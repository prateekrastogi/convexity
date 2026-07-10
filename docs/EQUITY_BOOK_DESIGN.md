# Equity Trend Book — Design (Program D1v2)

Status: DESIGN FROZEN before implementation. Owner cutover required before any live enablement.

## Owner decision of record (verbatim)

C1's research verdict was DO-NOT-DEPLOY on its optimized construction; the owner is deliberately
deploying the SIMPLE ROBUST construction instead, accepting the survivorship-curated universe as
a feature and bull-market lag as the premium. This is a timed-beta book with beta-like drawdowns,
capitalized small (250/1000).

## 1. The deployable construction → code map

Validated reference (D1 drawdown-overlay program, independently verified bit-for-bit):
CAGR 28.26%, maxDD 30.02%, Sharpe 1.1676, realized avg leverage 1.1103, EV 54.8bp/week.
Reference artifacts pinned at `tests/fixtures/equity_book/` (netret CSV + research scripts +
leverage_policy.py copy, all with provenance from the `equity-cta` worktree's D1 analyses).

| Piece | Design |
|---|---|
| Signal | `src/convexity/signal/equity_trend.py` — per-name ensemble of {200d SMA ±2% hysteresis band; 12-month (252d) total-return sign, monthly; 10-month SMA of month-end closes, monthly} → state ∈ {0,1/3,2/3,1}. LONG ONLY. All rules causal (decide at Friday close t, act next session). |
| Eligibility | Point-in-time: a name trades only with ≥260 real trading days of history (generic warmup — also covers perp-only names until they mature). The 6 leveraged/inverse/decay ETFs {UVXY,SQQQ,TQQQ,SOXL,KORU,MVLL}USDT are excluded by config (validated finding: structural decay contaminates trend signals). Names flagged by the split-watch (see §8) are excluded until cleared. |
| Per-name sizing | inverse 20d realized vol, floored at 0.5× cross-sectional median (the validated convention). Unit-gross normalized weights `w_unit`. |
| Book gross | `g_t = L_HALF × breadth_frac × m_brake` where breadth_frac = Σ(states)/n_eligible (NO renormalization to unit gross — the validated breadth-proportional design), and m_brake = clip(1 − DD_prev/0.50, 0, 1). |
| L_HALF | Frozen constant **2.325628534** in config. Deliberately NOT re-derived at boot (a boot-time re-derivation would drift with new data — the frozen backtest constant is the deployable). A test re-derives it from the fixture CSV via the pinned reference leverage_policy code and asserts equality, keeping provenance verifiable without runtime drift. |
| Brake input | The book's own journal equity chain: running peak over `equity_mark` records (+ latest `book_recon` unrealized when present). **Disclosed fidelity gap**: live brake updates weekly at settle (+ daily recon), vs daily marks in the backtest. Conservative direction (slower re-risking); documented, not hidden. |
| Rebalance | Weekly. Decide Friday close; timers run Saturday (UTC) like the other books. |
| Settle friction | `round_trip_friction()` = 13bp RT — matching the validated backtest's primary cost. |
| Demo gating | Live sessions gate on `fetch_execution_venue_tradeable()` (the commodity book's pattern) — only ~17–19 equities trade on demo; the model(93)-vs-tradeable gap is journaled every rebalance (`tradeable_gap` record). |

**Parity test (blocking)**: `tests/paper/test_equity_parity.py` rebuilds the duka-only daily-close
panel for the backtest window from the stitched lake (AAPL now fixed at source), runs the
PRODUCTION signal/sizing/brake chain in backtest mode, and asserts CAGR/maxDD/Sharpe against
`tests/fixtures/equity_book/dd_combined_waterfall_netret.csv` within 1e-3 absolute (float-path
tolerance; expected agreement ~1e-6). Marked `slow`; runs in CI/full suite.

## 2. Integer exchange leverage

Binance USDT-M leverage is a per-symbol INTEGER dial. Architecture separates:
(i) the dial — set ONCE at equity-book boot to **3** for its tradeable symbols (covers the 2.33×
nominal ceiling; also caps accidental over-leverage at 3× NAV); idempotent (same-value re-set is
a no-op; "cannot change with open position" errors logged, not fatal); journaled (`leverage_set`).
(ii) effective leverage — fractional (≈1.11× average), purely via notional sizing vs allocated NAV.
The dial is NEVER touched during rebalances.
Crypto/commodity: recon found NEITHER book sets the dial today (venue default applies). Their
current live behavior is not changed by this program (owner-approval boundary); the equity boot
step journals the observed dial for all three domains so the fleet state is visible. Normalizing
the other two books' dials is listed in the cutover report as an optional owner step.

## 3. Margin architecture — CROSS positions + SINGLE-ASSET mode, with software guardrails

Two independent axes, both documented:
- **Asset mode: SINGLE-ASSET (existing user mandate, capital.py)** — so the exchange can never
  draft wallet BTC as collateral. Unchanged.
- **Position margin mode: CROSS** (owner decision + technically correct: the MN books' legs
  offset; aggregate worst-case gross ≈1.33×; isolated would add per-position top-up failure
  modes). One-time cutover step verifies/sets this on the account (listed in the cutover report;
  not executed by this program).

Guardrails (all software, all journaled):
1. **Per-book margin budget**: before submitting, projected initial margin = Σ|target notional|/dial
   must fit inside the book's NAV allocation; the book scales down `g_t` proportionally if not
   (`margin_budget` journal record with before/after).
2. **Account monitor**: every live rebalance/recon journals `margin_ratio` =
   totalMaintMargin/totalMarginBalance. Thresholds (config): ALERT at 40%, DERISK at 60%.
   Justification: equity book worst historical day −10.4% at ~1.9× ≈ −20% book NAV ≈ −5% account
   (at 250/1000); a 60% maint ratio still leaves >3× that shock before liquidation territory.
   DERISK action: the equity book (levered, directional — declared priority: equity first,
   MN books last) flattens itself reduce-only at its next session; a `directive` record + loud log
   instructs the operator. (Fully autonomous flattening from the recon path is deliberately
   deferred v1 — demo account, low leverage, human-in-the-loop preferred; deferral documented.)
3. **Liquidation distance**: after every live rebalance, journal
   (marginBalance − maintMargin)/max(gross notional, ε) and assert > 3× the worst historical
   one-day move (12.1%); breach journals a `liq_distance_breach` directive.

## 4. Race-condition elimination

Recon finding: NO locking exists today; the two live books rely purely on timer staggering.
New module `src/convexity/paper/lock.py`:
- `account_lock(book, timeout=300s)` context manager: `fcntl.flock` on
  `data/paper/account.lock` (the SHARED lake path — one lock for every book/worktree/process),
  lock metadata (pid, book, ts) written inside, acquisition/release logged.
- Stale-lock policy: flock releases automatically on process death (kernel guarantee), so a
  kill −9 holder can never wedge the lock; the metadata file is advisory context only. A live
  holder blocks others up to `timeout`, then the waiter aborts LOUDLY (journaled `lock_timeout`)
  rather than proceeding unserialized — no false-stale takeover class of bug by construction.
- Critical section: venue-state read → margin checks → submissions → orphan sweep (all of
  `_submit_live` + the state reads feeding it), `cmd_finalize --live`, `cmd_recon --live`'s
  venue reads, `allocate`/`flow` journal writes. Signal computation stays OUTSIDE the lock.
- NAV snapshot semantics: NAV read once inside the lock, used for the whole rebalance.
- Kill-switch check: INSIDE the lock, first thing in the critical section.
- All three books get the lock (a fleet fix shipped in this branch, flagged in the report).
Timer staggering retained as defense-in-depth (slots in §7).

## 5. Rate limiting

`src/convexity/paper/ratelimit.py`: minimal shared limiter — module-level min-interval pacing
(authed calls ≥ 120ms apart, public ≥ 60ms) via a monotonic-clock gate, used by gateway and the
live submission loop. Deliberately simple (order counts are tens/week); a token bucket sized to
documented weights is overkill at this volume — documented simplicity, revisit if order traffic
grows.

## 6. Kill switches (all three books)

- Layer 1: control file `data/paper/<book>/KILL` — checked inside the lock at the top of every
  live rebalance/finalize/recon; if present: cancel open orders, flatten the book's domain
  positions reduce-only (maker-first, then finalize-style MARKET), journal `killed`, exit.
- Layer 2: CLI `python -m convexity.paper kill --book X [--resume]` — creates/removes the file
  + journals a `directive`.
- Layer 3: `systemctl disable convexity-<book>-*` (documented in deploy/README).
- Isolation: killing one book cannot touch another (domain scoping already guarantees it);
  freed margin is naturally visible to the others' fresh account fetches on their next cycle.

## 7. Books, config, timers, allocation

- `EQUITY = BookSpec(name="equity", sleeves=("equities",), journal_path="data/paper/equity/journal.jsonl", default_nav=250.0)` added to BOOKS. `assert_disjoint_domains()` unchanged (equities sleeve disjoint by construction).
- `src/convexity/paper/fleetconfig.py`: ALLOC_UNITS = {crypto: 400, commodity: 350, equity: 250}
  (per 1000); L_HALF; BRAKE_CEILING=0.50; MARGIN_ALERT=0.40; MARGIN_DERISK=0.60; LEVERAGE_DIAL
  = {equity: 3}; EQUITY_ETF_EXCLUSIONS; warmup=260. One location, no code edits to retune.
- New CLI `allocate-split --total X` and `flow-split --amount X`: proportional split across the
  three books per ALLOC_UNITS, all records written under the account lock (requirement 4).
- Timers (UTC, staggered; existing slots unchanged): crypto reb Sat 00:10 / commodity reb Sat
  00:40 / **equity reb Sat 01:10** / crypto fin 02:20 / commodity fin 02:50 / **equity fin
  03:20** / recons daily 00:25 / 00:45 / **01:05** / lake-refresh Sat 04:00 / **equity
  data-maintenance Sat 05:30** (after the lake refresh it depends on).
- New systemd files (committed, NOT enabled): convexity-equity-rebalance.{service,timer},
  convexity-equity-finalize.{service,timer}, convexity-equity-recon.{service,timer},
  convexity-equity-datamaint.{service,timer}.

## 8. Data maintenance

Recon finding: Binance-side incremental updates (13b) ALREADY exist — the lake-refresh chain
(backfill_crypto + backfill_tradfi_perp + backfill_funding + stitch) covers equity perps weekly.
New `scripts/equity_data_maintenance.py` (own service+timer, after lake refresh):
(a) NEW-SYMBOL DUKASCOPY BACKFILL: diff candidates.md equities (with a `dukascopy` mapping)
    against data/stitched/; missing → backfill via the existing dukascopy machinery
    (backfill_equity_hist.py conventions: politeness ≥1s, resumable), stitch that symbol,
    validate (max-move + seam checks); names failing mapping/validation → flagged for human
    review in `data/universe/equity_flags.json`, never silently traded or dropped. New names
    then mature through the generic 260d warmup before trading.
(b) SPLIT/CORPORATE-ACTION WATCH: scan each equity name's recent stitched closes; any
    |1-day move| > 25% (the established stitch-validation threshold) → flag in
    equity_flags.json + journal; `_equity_targets` excludes flagged names from the next
    rebalance until an operator clears the flag (the AAPL-class failure mode, contained).
(c) Idempotent, API-polite, bounded (only new symbols/new rows), own log; any failure degrades
    gracefully — trading proceeds on last-good data; staleness journaled per symbol.

## 9. Prerequisite data fix

`scripts/stitch_tradfi_1m.py` KNOWN_SPLITS += AAPLUSDT 2020-08-31 4:1 (TSLA/KLAC/KORU pattern);
regenerate AAPLUSDT.parquet only (original backed up in the lake first); re-validate; verify
neighbors byte-identical.

## 10. Existing files modified (fleet-affecting flagged ★)

- ★ `src/convexity/paper/cli.py` — equity targets fn + registration; lock wrapping of live
  sections (all books); kill-switch checks (all books); margin guardrails (live paths);
  allocate-split/flow-split/kill CLI commands; leverage-dial boot (equity).
- ★ `src/convexity/paper/books.py` — EQUITY BookSpec.
- `src/convexity/paper/gateway.py` — set_leverage + margin/maint fields on AccountSnapshot
  (additive).
- ★ `scripts/stitch_tradfi_1m.py` — AAPL KNOWN_SPLITS entry (data fix).
- New: signal/equity_trend.py, paper/lock.py, paper/ratelimit.py, paper/fleetconfig.py,
  scripts/equity_data_maintenance.py, deploy/convexity-equity-*.{service,timer}, tests.
Crypto/commodity BEHAVIOR changes shipped: lock + kill-switch + margin journaling on their live
paths (safety-additive only; no sizing/signal/order changes). Highlighted for the cutover report.

## 11. Test plan → requirement map

Parity (blocking, §1) · signal/sizing/brake units + causal-injection with naive control (1) ·
boot idempotency & duplicate-fire no-op (2) · domain disjointness incl. equity (3) ·
allocate/flow-split under lock (4) · orphan-sweep scoping incl. equity domain (5) · config (6) ·
chaos kill −9 at 3+ points incl. inside the lock — flock auto-release proves recovery (7) ·
three-book sim day + deliberate concurrent rebalances + deposit-mid-rebalance (8) · maker/GTX
utility mocked-exchange tests (9) · quantization edges (10) · kill-switch isolation +
margin-release visibility (11) · fleet review findings (12) · data-maintenance: new-symbol flow
(mocked fetch), append leaves history byte-identical, fake-split flag→exclusion (13) ·
30-day dry-run: realized leverage ≈1.1×, brake/breadth behavior sane.
