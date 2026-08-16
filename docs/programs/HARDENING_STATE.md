# Hardening + reorg program — STATE

Loop task (fires every 5 min). **Read this first every firing; continue from the first unchecked
box; never restart from scratch.** Update it after every stage and commit.

Scope guard, from the owner: **no changes to external functionality or core algorithm.** Every
edit here is tests, packaging, docs, layout, or ops. If a refactor would alter a number the
fleet produces, it is out of scope.

Branch: `hardening`. Fleet stays LIVE on `main`'s image throughout — the container is only
restarted once a change is proven.

## Stages

- [ ] **S0 — survey**: 68 test files, 196 src modules, 687 tests green as of f687e81.
- [ ] **S1 — RESTORE PERSISTENCE** (highest risk; I introduced this regression)
      systemd `Persistent=true` ran a missed job on next boot; cron has no catch-up, so a job
      missed while the container is down is silently skipped. Live-trading correctness gap.
- [ ] **S2 — coverage measurement**: add pytest-cov, build a `test` image stage, get the real
      baseline number per module.
- [ ] **S3 — test-gated image build**: image build FAILS if tests fail. Two layers, because the
      equity parity gate needs the 24 GB lake which is not in the build context:
        (a) in-image: full suite minus lake-dependent tests;
        (b) deploy script: parity/slow suite against the mounted data before promoting.
- [ ] **S4 — write tests to 90%**: target the modules the baseline shows thinnest.
- [ ] **S5 — single deploy shell script**: polls GitHub origin for new commits, rebuilds
      (tests gate it), redeploys, backgroundable on any VM.
- [ ] **S6 — reorg + stale docs**: README claims TWO strategies, fleet runs THREE
      (crypto / commodity / equity). `deploy/` layout is off. Sweep stale comments repo-wide.
- [ ] **S7 — final audit pass** + full suite + commit + verify fleet healthy.

## Facts worth not rediscovering

- Host has **no .venv** (removed in the Docker migration). All test runs happen in a container.
- Host is **aarch64**, uid/gid **1001**, passwordless sudo, Docker 29.7.2.
- `data/` is **24 GB** — never in build context; bind-mounted.
- The parity test **auto-skips** when `data/stitched` is absent, so it would silently skip inside
  an image build. That is exactly why S3 needs layer (b).
- Fleet container: `convexity-fleet`, supercronic PID 1, `restart: unless-stopped`.
- Next live job: Mon 00:25 UTC (demo crypto recon).
- Journals in `data/` are the accounting truth — never let a test write to the real ones.

## Log

- (start) branch created, state file committed.
