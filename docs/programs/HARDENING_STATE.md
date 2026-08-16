# Hardening + reorg program — STATE

Loop task (fires every 5 min). **Read this first every firing; continue from the first unchecked
box; never restart from scratch.** Update it after every stage and commit.

Scope guard, from the owner: **no changes to external functionality or core algorithm.** Every
edit here is tests, packaging, docs, layout, or ops. If a refactor would alter a number the
fleet produces, it is out of scope.

Branch: `hardening`. Fleet stays LIVE on `main`'s image throughout — the container is only
restarted once a change is proven.

## Stages

- [x] **S0 — survey**: 68 test files, 196 src modules, 687 tests green as of f687e81.
- [x] **S1 — RESTORE PERSISTENCE** DONE (highest risk; I introduced this regression)
      systemd `Persistent=true` ran a missed job on next boot; cron has no catch-up, so a job
      missed while the container is down is silently skipped. Live-trading correctness gap.
- [x] **S2 — coverage measurement** DONE: add pytest-cov, build a `test` image stage, get the real
      baseline number per module.
- [x] **S3 — test-gated image build** DONE: image build FAILS if tests fail. Two layers, because the
      equity parity gate needs the 24 GB lake which is not in the build context:
        (a) in-image: full suite minus lake-dependent tests;
        (b) deploy script: parity/slow suite against the mounted data before promoting.
- [ ] **S4 — write tests to 90%**: target the modules the baseline shows thinnest.
- [~] **S5 — single deploy shell script** written, UNTESTED: polls GitHub origin for new commits, rebuilds
      (tests gate it), redeploys, backgroundable on any VM.
- [~] **S6 — reorg + stale docs** layout done, DOCS PENDING: README claims TWO strategies, fleet runs THREE
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


## Progress log

**Checkpoint 1 (this commit).**

- **S1 persistence RESTORED and tested.** `ops/docker/catchup_missed.py` + `entrypoint.sh`;
  `job.sh` stamps `data/ops/lastrun/<job>` on success. Verified both paths: first boot seeds 1
  job and runs none; a stale stamp is detected as a missed 00:25 fire and the job is re-run.
  Two deliberate deviations from systemd, both toward safety: a 48h max age (recovering
  Saturday's rebalance on Sunday is right, on Wednesday it is an operator call), and first-boot
  seeds rather than stampeding all 34 jobs.
- **S2 baseline measured: 75%** full-suite (7749 stmts, 1960 uncovered, 688 passed).
  The `-m "not slow"` number is 71% — the parity test moves it 4 points.
- **S3 test gate SHIPPED and PROVEN.** Runtime stage does `COPY --from=test /app/.tests-passed`,
  which forces the test stage to build. Injected a failing test: build exited 1 and **no image
  was produced**. Lint is deliberately NOT in the gate (35 pre-existing findings, none
  behavioural) — blocking a correctness fix on an unsorted import is the wrong trade.
- **IMAGE 3.29 GB -> 778 MB (-76%)** as a side effect of the builder/test/runtime split: dev deps
  now exist only in the test stage.
- **S6 layout**: `deploy/` -> `ops/{docker,deploy}` + `ops/README.md`, all baked paths rewritten.

### Bugs found by doing this (all pre-existing)

1. **`tests/cost/test_cost_validate.py` used `parents[4]` for the repo root** — two levels above
   the repo. On the VM that resolved to `/home/data/stitched`, which never exists, so every
   lake-backed assertion in that file had been **silently skipping**. Its own comment says the
   check "only runs where data/stitched exists (it does on arb-vm)" — it never did. Fixed to
   `parents[2]`; the suite went 683 passed/3 skipped -> 688 passed.
2. **`job.sh` hardcoded the lastrun dir** while `catchup_missed.py` honoured
   `CONVEXITY_LASTRUN_DIR`. Identical in production, but if that var were ever set the stamps and
   the reader would diverge and every job would look permanently missed — re-running the whole
   fleet on every container start. Fixed to read the same override.
3. **35 ruff findings** across src/tests/scripts (67 unsorted-import, 20 deprecated-import,
   16 blind-except, 13 naive-`date.today`, 7 unused-variable...). None behavioural. Recorded, not
   gated.

### Next

- **S4 is the whole remaining job.** To reach 90% from 75% needs ~1160 more covered statements,
  and **`src/convexity/paper/cli.py` alone is 942 of the 1960 uncovered (16% covered)** — the
  live trading path: rebalance, finalize, catchup, reprice, tca, recon. It needs a mocked venue
  (gateway + account snapshot + filters) so sessions can be driven without touching Binance.
  That fixture is the single highest-value thing left in this program, coverage aside.
- Then `paper/gateway.py` (161), `panel/panel.py` (82), `paper/topup.py` (66), and the 17
  modules sitting at 0% (402 stmts, mostly `*/validate/*` research-support).
- S5 autodeploy.sh still needs a live `--once` test.
- README still claims TWO strategies; fleet runs THREE.
