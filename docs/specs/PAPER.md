# PAPER.md — build spec for the live/paper weekly rebalancer (`src/convexity/paper/`)

The ONE module `docs/EDGE.md` §(e) deferred, now built: a thin, scheduled composition of the
validated pillars over the Binance USDT-M **DEMO** environment. Nothing here re-models anything —
policy is imported from the pillars; this spec records the composition, its invariants, and its
honest v1 deferrals. Paper's mission (EDGE.md §(d)): accumulate the out-of-sample weeks that
resolve the two open empirical questions — maker fill rates (EXECUTION.md §16.1) and the Friday
phase premium — while the track record grows past its marginal DSR pass.

## 1. Cadence (systemd oneshot sessions — no daemon)

Every session is a FULL PURE-RECOMPUTE (the OPS restart doctrine): all strategy state is rebuilt
from the lake + journal + venue; there is no in-memory state to corrupt and a crashed session is
simply re-run. Timers (`deploy/`): **Sat 00:10 UTC** weekly rebalance (the Friday daily bar closes
at Sat 00:00 UTC; +10min lets it complete) → **Sat 02:20** finalize (cross unfilled maker
remainders — `tactic_for`'s final-slice fallback) → **Sat 04:00** lake refresh (dumps + stitch);
**daily 00:25** reconciliation + equity mark.

## 2. Data path (lake immutable; REST top-up in memory)

`build_panel` on the stitched lake + `paper/topup.py`: one `/fapi/v1/klines?interval=1d` and one
`/fapi/v1/fundingRate` request per symbol for the tail between the lake's last day and the last
COMPLETED UTC day, merged by `extend_panel` with build_panel's exact conventions (daily close,
midnight-funding-to-starting-day, `listed` = finite close, funding 0-fill/NaN-mask). The lake row
wins on overlap; the running day is never used; the lake file tree is NEVER written by a session
(the weekly refresh timer owns that, via the existing backfill + stitch scripts).

## 3. Policy composition (Invariant PAPER-PARITY)

Weekly targets = `policy.ensemble_target_weights` — the engine's own §6.2 band-ensemble
construction (L∈{12,14,16} quintile/vol-scaled/dollar-neutral, equal-weight member average),
**pinned BIT-IDENTICAL to `run_book` rows by `tests/paper`**: the book that trades is provably the
book that was validated; any drift is a test failure, never a silent divergence. Gross =
`LeverageEngine.size_book` at quarter-Kelly (`c_t = 0.25`) over the full validated weekly net
series (lake history + settled paper weeks); `g_t = 1` (§9 deferral). The plan is
`ExecutionEngine.on_target` — the no-trade band + trajectories, with the pipeline's RED picks
(ρ=1.0, band_c=0.5) and σ per-√hour.

### 3.1 Collateral waterfall (USER MANDATE, 2026-07-03)

**USDT funds the book first; USDC folds in ONLY when USDT alone cannot cover ~2× the estimated
initial margin (logged loudly); BTC NEVER enters NAV or margin** (`paper/capital.py`, test-pinned).
The book's BTCUSDT PERP positions are derivatives and do not touch the wallet BTC; keep the demo
futures account in SINGLE-ASSET margin mode so the exchange cannot draft BTC as cross-collateral
(runbook step, `deploy/README.md`).

## 4. Orders (venue-legal by construction)

Per surviving name: `child_order_qty` against the REAL per-symbol filters (`crypto_filters` —
lot/min-qty/min-notional; EXECUTION.md §16) → maker-first tactic (`tactic_for`, EXECUTION.md
§16.1) → tick-quantized side-aware reference price → **venue-exact decimal strings** via
`fmt_step` (a raw `str(float)` like `63512.600000000006` fails Binance's precision filters; the
strings are quantized to the step's own decimals, test-pinned). v1 execution shape (documented
simplification): ONE post-only maker order per name for the full surviving trade; the
Almgren-Chriss slice schedule is journaled for TCA but not sliced live until measured fill rates
justify the added order traffic; the finalize pass cancels resting makers and crosses remainders
as MARKET.

## 5. Journal (`data/paper/journal.jsonl`, append-only)

The persisted history pure-recompute rebuilds from: `equity_mark`, `flow` (+`directive`),
`targets` (weights, g_t, NAV, funding assets), `orders` (full intents incl. exact strings),
`settle` (r_twr AND r_model side by side — the live-vs-frozen-accounting gap is itself a
measurement), `recon`, `submitted`/`finalized`. Corrections are new records, never edits.

## 6. CLI (`python -m convexity.paper …`)

`rebalance [--live] [--asof] [--nav] [--no-topup]` (settles the prior week FIRST: frozen-
accounting `settle_week` on the journaled targets + `twr_weekly_return` over journaled flows —
OPS.md §15's Invariant FLOW-TWR live), `finalize [--live]`, `recon [--live]`,
`flow --amount <signed>` (journals the FlowEvent AND prints the `flow_risk_check` directive —
a ≥10% withdrawal demands the reduce-only INTERIM_RESIZE), `status`. Dry-run is the default and
needs no keys: equity evolves by the settled returns from `--nav`, carried book = prior scaled
targets, orders are journaled not sent.

## 7. Restart / deposits / withdrawals

Restart = re-run the session (§1; `Persistent=true` timers catch missed runs). Flows: journal
them same-day via the `flow` command; the settle links them into the exact TWR so they contribute
ZERO to every detector input (OPS.md §15.1); the directive table (§15.2) governs the immediate
response. HALT never auto-re-arms (OPS.md §6).

## 8. Gateway status: EXPERIMENTAL until the first supervised session

`paper/gateway.py` reuses Nautilus's own Binance plumbing (`BinanceHttpClient` +
`BinanceFuturesAccountHttpAPI`, bound against the probed 1.229 surfaces; DEMO base
`https://demo-fapi.binance.com`; post-only = LIMIT+GTX). Every call fails LOUDLY on adapter
drift. **The first `--live` run is supervised by design** — do not enable the timers before it
passes (runbook). Dry-run exercises everything except the four gateway calls.

## 9. Honest v1 deferrals (each is a stated decision, not an omission)

1. `g_t = 1` — the REGIME live overlay needs its causal calibration wired to the live weekly
   grid; deferred, the book runs un-overlaid exactly like the recorded baseline.
2. Slice-level Almgren-Chriss execution (§4) — deferred until measured maker fill rates exist.
3. Flow auto-detection via the income API — flows are operator-journaled v1 (`flow` command).
4. Per-name real-cost friction in the live plan — the flat 13bp is the recorded parity anchor;
   the per-name model remains the validated research restatement (study R2).
5. UNIVERSE shortability gates — all listed perps treated shortable (the recorded convention).
6. The `finalize` remainder heuristic is position-presence-based v1; exact per-order fill
   accounting arrives with the first supervised session's TCA.

## 10. Acceptance

`tests/paper/`: PAPER-PARITY bit-equality; `extend_panel` lake conventions; `fmt_step` venue
exactness; intent lot/tick/min-notional legality + exact strings; the collateral waterfall
(USDT-first, USDC fold-in, BTC never). Plus the dry-run weekly cycle exercised end-to-end on the
real lake (two consecutive sessions: settle → target → size → plan → journal).
