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

### 4.1 Reduce-only closes + the orphan sweep (added 2026-07-05)

A FULL CLOSE (target weight 0, trade nonzero) is flagged ``reduceOnly`` end-to-end (intent →
gateway ``new_order``), so the venue exempts it from MIN_NOTIONAL and it can never open or
flip a position. After the intents submit, the ORPHAN SWEEP closes (reduce-only MARKET) every
venue position that no intent covers and no target wants — the leak observed live 2026-07-05:
names that rotate OUT of the 100-name panel between weeks could never be closed by the plan
(11 stale bootstrap positions), and band-suppressed or min-notional-dropped closes strand the
same way. Sub-lot dust that cannot be closed by quantity is journaled (``orphan_sweep.dust``),
never silently retried. The finalize pass propagates each order's own ``reduce_only`` flag.

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

## 11. Multi-book: the commodity RV pair book beside the crypto book (added 2026-07-05)

ONE demo account, N strategies — raw venue state MISLEADS per-strategy book-keeping, so every
venue-facing surface is scoped by a **book** (`paper/books.py`; `--book {crypto,commodity}` on
every CLI command; each book = its own systemd units, so `journalctl -u convexity-commodity-*`
monitors the commodity book independently of crypto).

### 11.1 Books, domains, and the scoped orphan sweep

A book owns: a SYMBOL DOMAIN (candidates.md sleeves; pairwise-DISJOINT, test-pinned), its own
append-only JOURNAL (`data/paper/journal.jsonl` crypto — unchanged; `data/paper/commodity/…`
commodity), and its dry-run quantization surface (`sleeve_filters`; commodity entries now carry
the real venue filters). `orphan_close_intents` takes a REQUIRED `domain` — without it the crypto
sweep would market-close the commodity book's fresh positions every Saturday. Carried-book reads,
recon counts and the sweep all see only domain positions.

### 11.2 Virtual capital + book-scoped settles (`allocate`)

`python -m convexity.paper allocate --book X --amount N` splits the shared account into virtual
books (append-only; Σ allocations ≤ account USDT+USDC is a runbook check the recon journals). In
allocation mode a book's equity evolves by its OWN frozen-accounting settle (+ its journaled
flows); every live session journals a `book_recon` (book model equity, domain positions, domain
unrealized P&L via position_risk, shared account equity) so live-vs-model drift is MEASURED per
book, never blended. The crypto book keeps its legacy whole-account settle until its first
`allocate` (zero behavior change at merge); ONCE THE COMMODITY BOOK TRADES LIVE, BOTH BOOKS MUST
BE ALLOCATED (runbook). Targets records now journal their `symbols` list; settles re-align by
symbol (a book's panel can grow via discovery — positional alignment would silently mis-settle).

### 11.3 The commodity strategy (frozen research config)

`signal/rv_pairs.py` = the RV pair study's pre-named deployable EXACTLY (research/entry_exit/
study/RV_PREREG.md, net 0.73 backtest / plan-around 0.5–0.6 live): 5 pre-named pairs (XAU-XAG,
XAU-XPT, XAG-XPT, XPT-XPD, CL-NATGAS), band-ensemble z {26,52,104}wk on weekly-Friday log-ratio
spreads, continuous sizing −clip(z,±2)/2, AR(1) half-life gate [1,26]wk, pair risk parity
(26wk, [0.25,4]), unit gross, 10% vol target (26wk, [0.5,2]) — all recomputed from the panel
each session (no journal state). Causality is canary-pinned (the study's unshifted-levels book
prints Sharpe −3.8; tests re-pin the shift). Sizing = the study's own vol target; quarter-Kelly
is DEFERRED until a paper track exists. Settle friction 17 bp RT (5 taker + 3 half-spread + 0.5
impact, ×2), inside the study's {12,24} bp scenario band; the settle evolves the commodity book
on its SCALED (vol-targeted) weights — the study's deployable series. Price basis = the stitched
proxy+perp lake; measuring proxy→perp transfer is this paper book's mission.

### 11.4 Discovery + data freshness (never auto-trade)

`scripts/discover_tradfi_perps.py` (weekly timer, Fri 12:00 UTC): venue commodity perps
(`underlyingType == COMMODITY`, the verified predicate) not yet in the universe are appended to
the gitignored overlay `data/universe/discovered.json` (merged by `load_universe`, so panels,
backfills and the ORPHAN-SWEEP DOMAIN see them), backfilled immediately (klines + funding) and
perp-only-stitched. Filter drift and venue-missing symbols are reported loudly; new CRYPTO perps
are reported FYI only — the crypto universe is frozen by design (survivorship-clean validated
set). DISCOVERED SYMBOLS NEVER AUTO-TRADE: `rv_pairs.PAIRS` is frozen research config; promotion
requires a new study. The weekly lake refresh now runs `backfill_tradfi_perp_1m.py` so the
commodity lake stays as current as crypto's.

### 11.5 Commodity runbook (deploy — DO NOT enable timers before this)

1. Merge to main, redeploy units (`deploy/README.md`), keep timers DISABLED.
2. `allocate --book commodity --amount <N>` AND `allocate --book crypto --amount <M>` with
   N+M ≤ account USDT+USDC (BTC never counts).
3. One SUPERVISED `rebalance --live --book commodity` (same doctrine as §8) — verify the demo
   lists all 5 pairs' legs, fills, the orphan sweep touches nothing outside the domain.
4. Only then enable `convexity-commodity-{rebalance,finalize,recon,discovery}.timer`.
