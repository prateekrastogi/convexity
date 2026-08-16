# deploy/ — the convexity fleet runs as ONE self-contained container

Migrated off host systemd on 2026-08-16. The container carries its own scheduler, so the
deployment depends on nothing host-side: no systemd units, no `.venv`, no host python. Any host
with Docker plus this repo, `.env`, and `data/` runs the identical fleet.

```bash
docker compose up -d --build      # start (or rebuild + restart) the fleet
docker compose ps                 # is it alive
docker compose logs -f fleet      # watch it
docker compose down               # stop it
```

## Shape

| piece | what it is |
|---|---|
| `Dockerfile` | repo root. Multi-stage: `uv sync --frozen` into `/app/.venv`, then a slim runtime. |
| `docker-compose.yml` | repo root. The whole deployment — one service, `restart: unless-stopped`. |
| `deploy/docker/crontab` | all 34 job lines, transcribed 1:1 from the 31 systemd timers. |
| `deploy/docker/job.sh` | per-job wrapper: alerting + one 15-minute retry. |

PID 1 is **supercronic**, not systemd. systemd in a container needs `--privileged` or careful
cgroup delegation and turns the container into a pet VM; supercronic is built for this — logs to
stdout, forwards signals, no setuid.

## What the container mounts, and why

- `./data:/app/data` — 24 GB lake + the journals that ARE the fleet's accounting truth. Never
  baked into the image. The per-venue flock (`data/<venue>/account.lock`) only serialises
  correctly when every process shares one inode, which a bind mount guarantees.
- `.env` via `env_file` — secrets injected at runtime, never layered into an image.
- `/home/speculation/backups` at the **same absolute path** — `journal_backup.sh` derives its
  rotation dir and remote branch from the repo path's basename, so `/home/speculation/convexity`
  exists in the image as a symlink to `/app`. Point it at `/app` and the branch silently becomes
  `journal-backups-app`, orphaning the backup history.
- `/home/ubuntu/.ssh:/home/app/.ssh:ro` — the deploy key for that push. The ssh config names the
  key as `~/.ssh/convexity_deploy` and `~` is `/home/app` inside, so it resolves. Host files are
  0600 owned by uid 1001, matching the container user — ssh refuses looser permissions.

Runs as uid/gid **1001** to match the host `ubuntu` that owns `data/`, so journals written from
inside stay readable by host tooling.

## Manual commands

The default ENTRYPOINT is the interpreter, so one-offs look like the old venv python:

```bash
docker compose run --rm --entrypoint /app/.venv/bin/python fleet -m convexity.paper status
docker exec convexity-fleet /app/deploy/docker/job.sh adhoc-recon python -m convexity.paper recon --live
```

## Known behavioural difference vs systemd

The old timers set `Persistent=true`: a job missed while the host was down ran on next boot.
**cron has no catch-up.** If the container is down at 00:25, that recon is skipped, not run late.
The Sunday watchdog surfaces a missed weekly cycle; after a longer outage, run the missed command
by hand with the `docker compose run` form above.

## Runbook

- **Account setup (once, per venue)**: futures account in SINGLE-ASSET margin mode so the exchange
  cannot draft BTC as cross-collateral — collateral mandate is USDT first, USDC only when USDT is
  exhausted, **BTC never** (PAPER.md §3.1).
- **Deposits/withdrawals**: journal them the same day with
  `... -m convexity.paper flow --amount <signed USDT>` so the TWR chain stays exact (OPS.md §15);
  a ≥10% withdrawal prints an INTERIM_RESIZE directive — execute it.
- **HALT**: the kill-switch never auto-re-arms (OPS.md §6). After investigating, re-arm by
  explicit operator action only.
- **Venue**: code defaults to demo; prod jobs set `CONVEXITY_VENUE=prod` per crontab line. Both
  key sets live in the one `.env`.

## Rollback

The systemd units are in git history (`deploy/*.service`, `deploy/*.timer`, `deploy/prod/`,
removed in the Docker migration commit). Recover with `git show <commit>^:deploy/...`, reinstall
to `/etc/systemd/system/`, recreate `.venv` with `uv sync`, and `docker compose down`.
