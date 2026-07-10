# Equity Book — Cutover Report (Program D1v2)

Branch: `equity-book` @ `f9db22c` (design `b9b4fb4` → core `78bfe36` → infra/fleet `f9db22c`).
Status: **STOPPED FOR OWNER CONFIRMATION.** Nothing merged, no timer enabled, live system
verified untouched throughout (main @ `0bfe370`, clean, both timers active).

## PARITY TEST — the blocking gate: PASSED

The production signal/sizing/brake chain (`src/convexity/signal/equity_trend.py`, the same
functions `cli._equity_targets` calls live), run in backtest mode over the validated window,
reproduces the D1-validated reference exactly:
CAGR 28.26% / maxDD 30.02% / Sharpe 1.1676 / realized avg leverage 1.1103 — all within 1e-3
(observed ~1e-6), day-level return series max diff < 5e-4 across 2000+ overlapping days, and the
frozen `L_HALF=2.3256` re-derives exactly from the pinned fixture inputs
(`tests/paper/test_equity_parity.py`, runs in the full suite). The deployed book IS the
validated construction, by construction.

## Requirements 1–13 (file paths)

1. **Leverage+DD policy** — `signal/equity_trend.py` (exact port), `paper/fleetconfig.py`
   (frozen L_HALF, 50% brake ceiling), `cli._equity_targets` (breadth × half-Kelly × brake off
   the book's own journal-equity chain; disclosed weekly-vs-daily brake-cadence fidelity gap,
   conservative direction). Parity-pinned.
2. **Auto-rebalance + clean boot** — `deploy/convexity-equity-{rebalance,finalize,recon,
   datamaint}.{service,timer}` committed NOT enabled (Sat 01:10 / 03:20 / daily 01:05 / Sat
   05:30 UTC, staggered). Boot = journal replay (existing pure-recompute pattern); duplicate
   fire converges to zero trades via the no-trade band (verified: re-run produced turnover 0.0000)
   — same behavior as the existing books.
3. **Separate portfolio** — `EQUITY` BookSpec (sleeves=("equities",), own journal
   `data/paper/equity/journal.jsonl`, default NAV 250). Three-book domain disjointness verified
   (100/7/110 symbols, `assert_disjoint_domains()` passes; re-checked at every rebalance).
4. **Deposit/withdrawal rebalance** — `allocate-split --total X` / `flow-split --amount X` split
   400/350/250 across all three books atomically under the account lock.
5. **No orphans** — the existing domain-scoped orphan sweep now covers the equity domain;
   reconciliation + cancel-first idempotency inherited from the proven live flow; all of it now
   inside the lock.
6. **Config** — `src/convexity/paper/fleetconfig.py`: allocations, L_HALF, brake ceiling, margin
   thresholds, leverage dials, ETF exclusions, warmup, lock/pacing constants. One place.
7. **Failsafe restart** — journal is append-only; NEW: torn-final-line crash recovery in
   `Journal.read` (kill -9 mid-append tolerated loudly; mid-file corruption still fails fast) —
   a pre-existing FLEET bug found in review, fixed for all three books, tested both directions.
   Lock survives holder death by kernel flock semantics (chaos-tested: process killed −9 inside
   the critical section; next waiter proceeded in <2s).
8. **Coexistence** — one account lock serializes every venue-facing critical section across all
   three books + allocate/flow (10 entry sites); shared API pacing on every gateway call; timer
   staggering retained as defense-in-depth. Three-book dry-run smoke: all three rebalance
   cleanly with the new code; crypto/commodity outputs unchanged.
9. **Maker orders** — recon finding: maker-first (GTX post-only + finalize crossing) ALREADY
   exists fleet-wide as the v1 execution shape; the equity book inherits it unchanged. No new
   flags needed; no crypto/commodity behavior change.
10. **Quantization** — inherited: per-book `book_filters` + live exchangeInfo re-quantization +
    venue-exact decimal strings. Model(110-domain)-vs-tradeable gap journaled every live
    rebalance (`tradeable_gap` record; demo lists ~17-19 equities today).
11. **Kill switches** — `data/paper/<book>/KILL` checked INSIDE the lock at the top of every
    live rebalance/finalize/recon (flatten reduce-only + halt); CLI `kill --book X
    [--resume|--now]`; systemctl path documented. Isolation tested (equity kill does not touch
    commodity).
12. **Fleet bug-hunt** — findings: (a) `Journal.read` torn-line poisoning (CRITICAL,
    **fixed**, §7); (b) no serialization existed between the two live books (MAJOR, **fixed** —
    the lock now covers them; they were racing on timer stagger alone); (c) maker execution
    already fleet-wide (spec assumption corrected, no change); (d) books never set the leverage
    dial (documented; equity now sets 3x at boot; normalizing crypto/commodity dials = optional
    owner cutover step, not autonomous); (e) margin mode: NO code assumes isolated; the standing
    SINGLE-ASSET **asset-mode** mandate (capital.py, BTC-collateral protection) is a different
    axis than cross/isolated **position** mode and is unchanged. Account's actual position-margin
    mode must be verified at cutover (step B below).
13. **Data maintenance** — `scripts/equity_data_maintenance.py` + its own timer: new-symbol
    dukascopy backfill → stitch → validate → pending-until-260d (unresolvable → flagged, never
    silently traded/dropped); split-watch flags |1d|>25% non-gap moves → excluded from
    rebalances until operator clears (`equity_flags.json`); Binance-side incremental updates
    confirmed ALREADY covered by the weekly lake-refresh chain. Flag→exclude flow tested end to
    end, gap-day false-positives tested.

**Prerequisite data fix landed**: AAPLUSDT 2020-08-31 4:1 split added to KNOWN_SPLITS;
parquet regenerated (backup: `AAPLUSDT.parquet.pre_split_fix` in the lake); fake −74.12% day →
real +3.52%; NVDA/MSFT parquets verified byte-identical.

## Test & simulation results

- Full suite: **669 passed, 1 skipped** (654 baseline + 15 new), including the parity gate.
- 6-week dry-run replay (fresh journal, 2026-05-22→06-26): settles evolve on SCALED weights;
  brake engagement hand-verified at every step (−6.46% week → m=0.871; new high → m=1.0;
  −7.51% → m=0.850); breadth ~0.70; realized G_t 1.37–1.66.
- Chaos: kill −9 inside the lock's critical section → kernel releases, next waiter proceeds;
  torn-journal recovery both directions; lock timeout aborts loudly (never unserialized).
- Margin guardrails: budget scale-down, DERISK→equity-flatten, ratio journaling — tested mocked.

**Honest scope notes (not silently skipped):** (i) true live-path integration (real GTX
submissions, the real `/fapi/v1/leverage` call, live margin fields) is untested by this program —
inherent to the repo's own standing convention that the gateway is EXPERIMENTAL until a
supervised `--live` session, which is the first cutover step; (ii) genuinely-concurrent live
submissions from two books were validated at the lock level (process serialization), not against
the real venue; (iii) the brake's live cadence is weekly-settle + daily-recon marks vs the
backtest's daily marks — conservative direction, disclosed in design §1.

## Margin architecture as shipped

CROSS position-margin (owner decision; MN books' legs offset; worst-case account gross ~1.33x) +
SINGLE-ASSET asset-mode (existing BTC mandate, unchanged) + three software guardrails:
per-book IM budget (dial-divisor, scales g_t down), account maint-ratio monitor
(ALERT 40% / DERISK 60% — equity flattens first; justification: equity worst day −10.4% at ~1.9x
≈ −20% book ≈ −5% account at 250/1000), liquidation-distance journaling (floor = 3 × 12.1%).

## Cutover commands (owner executes; nothing below has been run)

**Group A — repo cutover:**
```
cd /home/speculation/convexity
git merge equity-book --no-ff -m "merge equity-book: D1v2 equity trend book + fleet hardening"
PYTHONPATH=src:. .venv/bin/python -m pytest tests -q       # expect 669 passed / 1 skipped
sudo cp deploy/convexity-equity-*.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload
git push origin main
```

**Group B — one-time LIVE-ACCOUNT state (touches the real demo account; do under supervision):**
```
# 1. Verify/normalize position-margin mode = CROSSED for the traded symbols (Binance UI or API);
#    confirm asset mode remains SINGLE-ASSET (BTC mandate).
# 2. Allocate the three books (writes journals only):
.venv/bin/python -m convexity.paper allocate-split --total <ACCOUNT_USDT>
# 3. First equity session SUPERVISED (sets the 3x dial on its symbols as it trades):
.venv/bin/python -m convexity.paper rebalance --live --book equity
.venv/bin/python -m convexity.paper finalize  --live --book equity   # ~2h later
# 4. Only after a clean supervised session:
sudo systemctl enable --now convexity-equity-rebalance.timer convexity-equity-finalize.timer \
                           convexity-equity-recon.timer convexity-equity-datamaint.timer
```

**Crypto/commodity behavior changes shipped in this branch (flagged):** account lock around
their live critical sections; kill-switch checks; margin-ratio journaling in their recons;
torn-journal recovery; API pacing. No changes to their signals, sizing, orders, or costs — their
dry-run outputs verified unchanged.
