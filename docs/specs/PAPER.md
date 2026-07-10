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
prints Sharpe −3.8; tests re-pin the shift). LIVE sessions gate pairs on the EXECUTION venue's
actual listings (`fetch_execution_venue_tradeable`, demo exchangeInfo): a pair whose leg the
venue does not list is excluded WHOLE — learned in the first supervised session (2026-07-06:
demo lists the 4 metals, not CL/NATGAS; an ungated model would settle energy-pair P&L on
positions that can never exist). Sizing = the study's own vol target; quarter-Kelly is DEFERRED
until a paper track exists. Settle friction 17 bp RT (5 taker + 3 half-spread + 0.5
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

## 12. Multi-book III: the equity trend book + fleet hardening (added 2026-07-10)

Third book on the same demo account (`--book equity`; own units, `journalctl -u
convexity-equity-*`). OWNER DECISION OF RECORD: the certified equity research programs
(E1/E2/E3 `research/equity/`, I1 `research/indices/`, C1 `research/cta/`) all closed as clean
negatives — the owner deliberately deploys the SIMPLE ROBUST construction anyway, accepting the
survivorship-curated universe as a feature and bull-market lag as the premium. This is TIMED
BETA with beta-like drawdowns, capitalized small (300/1000), never presented as market-neutral
alpha.

### 12.1 The equity strategy (frozen validated config; Invariant EQUITY-PARITY)

`signal/equity_trend.py` = the D1 drawdown-overlay program's validated construction EXACTLY
(record: `research/cta/_dd_combined_waterfall*`; validated CAGR 28.26% / maxDD 30.02% / Sharpe
1.1676 / realized avg leverage 1.11): per name, LONG/FLAT equal-weight ensemble of {200d SMA
±2% hysteresis band (daily); 12-month total-return sign (monthly); 10-month SMA of month-end
closes (monthly)} → state ∈ {0,⅓,⅔,1}; inverse-20d-vol weights (0.5× cross-sectional-median
floor); book gross = **L_HALF × breadth_frac × m_brake** where L_HALF=2.3256 is the FROZEN
half-Kelly constant (never re-derived at boot — that would drift with new data; a test
re-derives it from pinned fixtures), breadth_frac = Σ(states over eligible)/n_eligible (NO
unit-gross renormalization — the breadth-proportional design, zero parameters), and m_brake =
clip(1 − DD_prev/0.50, 0, 1) off the book's OWN journal-equity running peak (weekly settles +
latest recon mark — a disclosed, conservative-direction fidelity gap vs the backtest's daily
marks). Decide Friday close, act next session; settle friction 13 bp RT (the backtest's own
primary cost); the settle evolves the book on its SCALED weights (g_t carries the leverage —
an unlevered settle would feed the brake wrong drawdowns). Eligibility: ≥260 real trading days
(names below it are tracked-but-pending, journaled, never traded — perp-only listings mature
through the same rule), MINUS the 6 leveraged/inverse/decay ETFs
({UVXY,SQQQ,TQQQ,SOXL,KORU,MVLL}USDT — validated finding: structural daily-reset decay
contaminates trend signals), MINUS split-watch-flagged names (§12.3). LIVE sessions gate on the
execution venue's actual listings (§11.3's pattern) and journal the model-vs-tradeable gap
(`tradeable_gap`; demo lists ~17–19 equities today).

**Invariant EQUITY-PARITY**: `tests/paper/test_equity_parity.py` runs the PRODUCTION signal/
sizing/brake chain in backtest mode and pins CAGR/maxDD/Sharpe/avg-leverage AND the day-level
return series to the validated reference (`tests/fixtures/equity_book/`, provenance from the D1
record). A deployment whose backtest-mode output diverges from the validated artifact is broken
by definition. Owner-tunable values live in ONE place: `paper/fleetconfig.py` (allocations
400/300/300 per 1000, L_HALF, brake ceiling, margin thresholds, leverage dials, ETF exclusions,
warmup, lock/pacing constants). AAPLUSDT's unadjusted 2020-08-31 4:1 split (fake −74.12% day)
was fixed at source with this book (KNOWN_SPLITS; pre-fix parquet kept in the lake).

### 12.2 Fleet hardening (shipped with this book; applies to ALL THREE books)

- **THE ACCOUNT LOCK** (`paper/lock.py`): recon found the two live books had NO serialization
  (timer stagger only). One `flock` on `data/paper/account.lock` now serializes every
  venue-facing critical section (state read → margin checks → submissions → orphan sweep) and
  the allocate/flow writes, fleet-wide. Kernel releases on holder death (kill −9 can never
  wedge the fleet — chaos-tested); a live-but-slow holder blocks waiters up to 300 s, then the
  waiter ABORTS loudly, never proceeds unserialized. Signal computation stays outside the lock;
  timer stagger remains defense-in-depth.
- **KILL SWITCHES** (per book, independent): `data/paper/<book>/KILL` checked INSIDE the lock at
  the top of every live rebalance/finalize/recon → cancel + flatten reduce-only + halt; CLI
  `kill --book X [--resume|--now]`; systemctl disable is the third layer. Isolation test-pinned
  (killing one book cannot touch another — domain scoping).
- **CROSS-MARGIN + GUARDRAILS**: position-margin mode is CROSS (owner decision: the MN books'
  legs offset — isolated would double their margin for zero risk benefit; worst-case account
  gross ≈1.33×). This is a DIFFERENT AXIS from §3.1's SINGLE-ASSET asset-mode mandate (BTC
  protection), which is unchanged. Software guardrails: (1) per-book IM budget — projected
  initial margin (gross/dial) must fit the book's NAV, else g_t scales down (journaled);
  (2) account maint-margin-ratio monitor, ALERT 40% / DERISK 60% (equity — the levered
  directional book — flattens FIRST; justification: its worst day −10.4% at ~1.9× ≈ −6% of
  account at 300/1000); (3) liquidation-distance journaled each session (floor = 3× the worst
  historical one-day move, 12.1%). Monitors run daily via recon.
- **INTEGER LEVERAGE DIAL**: Binance sets per-symbol leverage as an INTEGER; the dial is a
  cap + margin divisor, NEVER the sizing mechanism (effective fractional leverage comes from
  notional vs NAV). The equity book sets its symbols to 3× once per session
  (`gateway.set_leverage`, idempotent, errors journaled never fatal). Crypto/commodity never
  set the dial (venue default stands; normalizing them is an owner option, not autonomous).
- **SPLIT ALLOCATION**: `allocate-split --total X` / `flow-split --amount X` write all three
  books' records proportionally (400/300/300) under the lock — deposits/withdrawals cannot
  change NAV under a sizing session (NAV is snapshotted once inside the lock).
- **JOURNAL CRASH RECOVERY**: `Journal.read` now tolerates a torn FINAL line (kill −9
  mid-append; skipped loudly, the record was never acknowledged) while a torn line anywhere
  else still fails fast — a pre-existing fleet bug fixed for all three books, test-pinned both
  directions.
- **API PACING** (`paper/ratelimit.py`): min-interval gate on every gateway call — deliberately
  simple at tens-of-orders-per-week volume; the lock is the real concurrency control.

### 12.3 Equity data maintenance (own timer; trading NEVER blocks on it)

`scripts/equity_data_maintenance.py` (Sat 05:30 UTC, after the §11.4 lake refresh it depends
on): (a) NEW-SYMBOL BACKFILL — an equities entry with a `dukascopy` mapping but no stitched
parquet gets dukascopy-backfilled (the established candle-mode machinery, politeness-spaced,
resumable) → stitched → validated; unresolvable mappings or failed validation → flagged for
human review in `data/universe/equity_flags.json`, never silently traded or dropped; new names
then mature through the 260d warmup. Binance-side incremental updates for EXISTING symbols are
already the lake refresh's job. (b) SPLIT/CORPORATE-ACTION WATCH — any |1-day| move >25% on a
non-gap day in the fresh tail flags the name; `_equity_targets` EXCLUDES flagged names until an
operator clears the flag (the AAPL failure mode, contained at the data layer; a genuine
earnings blowup can flag too — accepted: a human confirms and clears, the book holds the prior
week's position meanwhile). Failures degrade gracefully: trading proceeds on last-good data,
staleness journaled per symbol (own audit log `data/universe/equity_datamaint_log.jsonl`).

### 12.4 Equity runbook (deploy — DO NOT enable timers before this)

Code merged 2026-07-10 (PR #3 + the research-record merges); timers NOT enabled. Honest scope
note: live-path integration (real GTX submissions, the real `/fapi/v1/leverage` call, live
margin fields) is untested until the supervised session below — the §8 doctrine.

1. Redeploy units (`deploy/README.md`), keep the four equity timers DISABLED.
2. ONE-TIME ACCOUNT STATE (supervised; touches the real demo account): verify position-margin
   mode is CROSSED for traded symbols; confirm asset mode remains SINGLE-ASSET (§3.1).
3. `allocate-split --total <ACCOUNT_USDT>` — all three books re-allocated 400/300/300 (the
   equity book REFUSES to trade live without an allocation, §11.2's guard).
4. One SUPERVISED `rebalance --live --book equity` (+ `finalize --live --book equity` ~2 h
   later): watch the 3× dial get set, maker fills, the margin/liq-distance journal records, and
   confirm the orphan sweep touches ONLY equity-domain symbols.
5. Only then enable `convexity-equity-{rebalance,finalize,recon,datamaint}.timer`.

### 12.5 Manual-position carve-out (all books)

The operator trades the same venue account by hand. Domain scoping protects each book from the
OTHERS, but not from the operator: a discretionary position in an in-domain symbol would be
absorbed into that book's carried state, traded toward its target, orphan-swept, or
kill-flattened. The carve-out:

- **Ledger**: `data/paper/manual_reserve.json` — symbol → signed operator-owned qty. Maintained
  ONLY via `python -m convexity.paper manual --reserve SYM QTY | --release SYM | (no args =
  list)`; edits run under the account lock. Never auto-populated — explicit ownership beats
  silent adoption.
- **Netting**: every venue-position read a book performs (carried book, orphan sweep, kill/DERISK
  flatten, finalize remainder check, recon counts, book-share unrealized) first subtracts the
  reserve: `book_qty = venue_qty − reserved_qty`. The book manages only its share; carve-outs are
  journaled as `manual_reserve` records.
- **Safety clamp**: a book share never flips sign against the venue position — if the reserve
  exceeds what exists (operator closed part without releasing), the share clamps to 0 and the
  discrepancy is journaled loudly.
- **Unowned hygiene**: reserved symbols outside every domain are silenced from the
  `unowned_positions` warning (they're yours); unreserved ones stay loud.
- **Margin**: cross mode means manual positions share collateral. Guardrails deliberately read
  WHOLE-ACCOUNT margin (a large manual position tightens the books' budget — conservative), but
  DERISK/kill only ever close book shares, never the reserve.
- **Operator contract**: reserve BEFORE or promptly after opening a manual position in an
  in-domain symbol (worst case before the next timer fire: recon daily ~01:05 UTC, rebalance Sat);
  release when you close it. Out-of-domain symbols are safe even unreserved (unowned warning
  only). `manual` with no args lists the ledger.
