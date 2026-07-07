# I1 TIMED-BETA OVERLAY (SPY/QQQ) — PRE-REGISTRATION (Program 8; FROZEN 2026-07-07, before any
pipeline code)

Owner-directed follow-up to the closed E1/E2/E3 equity swing-research programs (all clean
negatives, research/equity/ on main). Isolation: runs in the `indices` worktree
(/home/speculation/convexity-indices, branch `indices`); reads the shared on-disk lake at
`data/stitched/` and the newly-acquired `data/fomc/` (both untracked, symlinked/populated per the
Setup and Data Acquisition phases — reads/writes confined to this worktree's own data directory);
NEVER touches the live paper system (`/home/speculation/convexity` on branch `main`, its systemd
timers, or its `data/paper/` journals — that directory is not even symlinked into this worktree,
confirmed in Setup). Executed by `i1_pipeline.py` (new, self-contained file under
`research/indices/`, not yet written); outcome to be recorded in `I1_REPORT.md`.

## PROGRAM CLASS (read first — this is NOT another edge-certification program like E1/E2/E3)

E1/E2/E3 were cross-sectional edge-certification programs: breadth ~93-110 names, DSR at a full
N_trials ledger as the hard adoption gate. **This program is different in kind.** It is an
OVERLAY-VALIDATION program on n=2 instruments (SPYUSDT, QQQUSDT). With breadth 2 and ~8 years of
daily data, no timed-beta construction can honestly clear a DSR significance gate — this is a
mathematical reality of the sample size (DSR's own denominator scales with the variance of
Sharpes across an assumed-large trial population; a 2-instrument overlay family does not resemble
that population), not a flaw to hide or a shortfall to apologize for.

The intellectual basis for a deployment recommendation here is instead three-legged:

(a) **Decades of published out-of-sample evidence** for each individual component from OUTSIDE
    this dataset entirely — 200-day/10-month trend filtering on equity indices (Faber 2007 and the
    much older SMA-crossover literature it formalizes), volatility targeting (Moreira-Muir 2017 and
    the broad risk-parity literature), turn-of-month seasonality (Ariel 1987 and its many
    replications), and pre-FOMC drift (Lucca-Moench 2015, one of the most replicated calendar
    anomalies in modern equity microstructure).
(b) **Our own data REPRODUCING the documented behavior of each component** — confirming known
    effects in this specific SPY/QQQ sample, not discovering new ones. The S1 regime-sanity
    signature below exists precisely to check this reproduction honestly, including the parts
    where the textbook story does NOT play out cleanly (e.g. a fast crash outrunning a 200-day
    filter).
(c) **The combined overlay dominating a buy-and-hold benchmark on risk-adjusted terms** in our
    sample, robustly across parameter perturbations — not just at one cherry-picked corner of the
    grid.

The adoption gate below is therefore **BENCHMARK-RELATIVE, not DSR-based**. DSR and PBO are still
computed and reported HONESTLY as informational numbers at the full N_trials ledger (declared in
full below) — expect them to be low, and `I1_REPORT.md` will say so plainly rather than omitting
them or apologizing for them. Neither DSR nor PBO gates anything in this program.

This is **TIMED BETA** — directional market exposure with beta-sized drawdowns — explicitly NOT
market-neutral alpha like the crypto cross-sectional momentum book (the one currently-deployable
edge from the convexity research line, per `convexity-xsec-momentum-edge` notes). `I1_REPORT.md`
must never present this overlay as market-neutral or risk-comparable to that book. Any eventual
deployment recommendation from this program is for a separate, separately-capitalized **THIRD**
paper book, distinct from the existing crypto and commodity books — stated explicitly now: this
overlay would NOT be blended into either existing book, even if it clears its own gate.

## Thesis (one program, one overlay, three components)

A simple, fully mechanical long-or-flat trend/vol-target/calendar overlay on a 50/50 SPY/QQQ blend
should dominate static buy-and-hold on risk-adjusted terms in this ~8-year sample — shallower
drawdowns in slow bear markets and fast crashes (S1), a smoother, more consistently-sized risk
profile (S2), and a small, honestly-fragile extra edge from two well-documented calendar effects
(S3) — while giving up some upside in strong, low-volatility bull regimes (the 2023-2024 melt-up
is the anticipated admission price, stated before any run, not a post-hoc excuse).

## Data contract landmines (verified FRESH in this document, not assumed inherited from E1/E2/E3)

Per DATA_CONTRACT and the Setup-phase CONTEXT: `data/stitched/{SYMBOL}.parquet`, schema
`open_time(ms)|open|high|low|close|volume|source('duka'|'perp')|gap_before(bool)`. Setup confirmed
fresh via direct parquet read AND `load_universe()`: SPYUSDT `source=='duka'` rows=778,291, span
2018-05-01 -> 2026-04-06 (then splices to `perp`); QQQUSDT `source=='duka'` rows=777,249, span
2018-05-01 -> 2026-04-02 (then splices to `perp`). Both `on_demo=true` in
`src/convexity/panel/candidates.md` (fresh `load_universe()` call, not memory) — the DEMO GATE
required by landmine 5 below is satisfied for both names, so a paper deployment of the eventual
book is actually possible. This program uses ONLY the `source=='duka'` leg for both names (2018-
05-01 through the respective 2026-04 splice dates) — the short perp leg after the splice is
EXCLUDED from the core signal and return series entirely, not spliced in, because landmine (1)
below is about not crossing the seam, and the cleanest way to guarantee that is to never let the
perp rows enter the frame used for S1/S2/S3 at all. (This does shorten the usable sample by
several months at the end relative to the full lake; this is stated here explicitly as a
deliberate, conservative choice, not a silent truncation discovered later.)

## Data acquisition outcome (FIXED input, not revisited here)

The Data Acquisition phase succeeded in full: 67 regularly-scheduled FOMC meetings 2018-01 through
2026-07-04, cached to `data/fomc/fomc_dates.csv` (untracked), sourced from
federalreserve.gov's own calendar and historical pages, with 7 hand-checked spot validations all
PASS (including the 2020-03-15 emergency cut and the 2019-10-04 unscheduled conference call both
correctly EXCLUDED, and the cancelled 2020-03-17/18 regularly-scheduled meeting correctly absent).
Coverage summary recorded at `research/indices/i1_fomc_data_coverage.json`. **Therefore: S3(b)
(pre-FOMC tilt) is IN.** Both S3 tilts (turn-of-month and pre-FOMC) proceed as specified. The
contingency described in the task (drop S3(b), proceed with S1/S2/S3(a) only) was NOT triggered
and is not discussed further below except where the with/without-tilts robustness split is
declared.

## S1 — trend filter (frozen)

Per instrument (SPY, QQQ separately, each using its own `source=='duka'` daily close series).
Daily close = each day's last `duka`-sourced bar. Three signals, ALL using only information
through day/month t-1's close to set day t's exposure (see AVOIDING BUGS (3) below for the exact
mechanics of this):

- **(a) 200-day SMA with +-2% hysteresis band.** Long when price > SMA_200 * 1.02, flat when price
  < SMA_200 * 0.98, HOLD the prior day's state when price is inside the band. The band exists
  specifically as a whipsaw-cost control mechanism — a bare SMA crossover flips state every time
  price crosses the line even by a hair, generating cost-bleeding round trips in a choppy,
  range-bound tape; the +-2% band requires a decisive break before flipping, which is the entire
  reason it is in the grid rather than a bare crossover.
- **(b) 12-month (252-trading-day) total-return sign, evaluated monthly.** Long if the trailing
  252-trading-day total return (as of the prior month-end close) is positive, flat if negative or
  zero.
- **(c) 10-month SMA rule (Meb Faber convention), evaluated monthly.** Long if month-end price is
  above its own 10-month (210-trading-day) SMA, flat otherwise.

**Ensemble (the only S1 construction that is ever scored as a tradeable cell)**: equal-weight mean
of the three rules' exposure states (each in {0,1}), giving an ensemble output in {0, 1/3, 2/3, 1}.
Never argmax a single rule. Long-or-flat only — explicitly NO shorting the index. Reasoning stated
now: the published evidence base for trend filtering on broad equity indices (Faber 2007 and the
literature it summarizes) is a long/flat risk-reduction technique, not a symmetric long/short
strategy; shorting a broad equity index carries a structurally negative expected-return headwind
(the equity risk premium itself) that a trend filter of this kind is not designed to overcome, and
adding a short leg would conflate two different theses in one construction.

**S1 regime-sanity signature (all four clauses MANDATORY, checked and reported with real numbers
before anything else in this program is trusted — this IS the mechanism check):**

(i) **Drawdown reduction in 2020-03 (COVID).** Report the actual date(s) the ensemble's exposure
    dropped, the realized peak-to-trough drawdown of the S1 book vs. the buy-and-hold benchmark
    over Feb-Apr 2020, and be honest if the filter exited LATE — a fast V-shaped crash can outrun
    a 200-day-SMA-class filter, and if the data shows that, `I1_REPORT.md` says so plainly rather
    than implying a textbook save that didn't happen.
(ii) **Bull-market participation.** The ensemble must show clear, near-full exposure through the
    2020-H2/2021 recovery rally and the 2023-2024 melt-up. Report the actual exposure time-series
    statistics (mean exposure, % of days at full exposure) for both windows — a filter that stays
    flat through either is broken, not conservative.
(iii) **Visible whipsaw cost in 2018-Q4 and mid-2022.** Both are choppy, range-bound-to-declining
    periods where trend-following is known to underperform buy-and-hold via repeated false
    signals. Report the actual number of exposure flips and their realized cost in each window.
    The ABSENCE of visible whipsaw cost here is itself a red flag (the same diagnostic logic as
    `DURABLE_REPORT.md`'s trend-drought check on main) — a suspiciously clean curve through a
    known chop period is treated as a bug-hunt trigger, not a sign the strategy is unusually good.
(iv) **2022 full-year drawdown, S1 book vs. benchmark.** Report both maxDD numbers; the S1 book's
    2022 drawdown is expected to be materially shallower than buy-and-hold's — this is the slow-
    grind bear market the filter exists to protect against, distinct from the fast-crash case in
    (i).

**Survivorship for S1**: SPY/QQQ are not survivor-selected individual securities (unlike E1/E2/E3's
93-110-name rosters), but QQQ's own 2018-2026 sample sits almost entirely within one secular tech
bull regime. Per-year and per-regime numbers (declared in S1's own regime-sanity clauses above,
plus the per-year Sharpe table reused from `e3_pipeline.py`) are reported specifically so this is
visible in the data, not hidden inside one full-sample average.

**S1 grid cells: 1** (the ensemble is the only tradeable S1 cell; the three individual rules are
diagnostic inputs to the ensemble, not separately-credited trials). This is a DIFFERENT
multiplicity convention than E1/E2/E3, where individual lookbacks WERE separately credited as
trials — the reasoning here is not multiplicity-avoidance for this program specifically, but that
the trend-following literature converges on ensembling short/medium/long-style rules as standard
practice for exactly this reason (reducing single-rule whipsaw sensitivity), so scoring the
ensemble alone reflects how this class of strategy is actually built and used in the literature,
not a post-hoc choice to shrink this program's own ledger.

## S2 — volatility targeting (frozen)

Applied multiplicatively on top of S1's ensemble exposure state: `scale[t] = clip(target_vol /
realized_vol[t], 0, cap)`, where `realized_vol[t]` is the annualized realized volatility of daily
`duka`-sourced log returns computed over a trailing window through day t-1 (NEVER including day t
itself — see AVOIDING BUGS (3)). `target_vol = 10%` annualized (the house convention used
elsewhere in this codebase's vol-target overlays, e.g. `regime/overlay.py`).

**Grid (frozen, multiplicity ledger):**
- Window `W_vol` in {20, 60} trading days — 2 variants.
- Cap in {1.0, 1.5} — 2 variants. Cap=1.0 means the overlay can only DE-RISK relative to S1's base
  exposure (never lever above 1.0x); cap=1.5 allows modest re-leveraging in unusually calm regimes.
  Both are declared grid points, not a single fixed choice, because the de-risk-only vs.
  modest-releverage design decision is a real, debatable construction choice with a documented
  tradeoff (de-risk-only is more conservative and easier to justify to a risk committee; modest
  releverage captures more of genuinely calm bull regimes at the cost of larger drawdowns if
  calm-regime vol estimates are stale when a shock hits).
- **S2 grid cells: 2 x 2 = 4.**

Combined exposure: `exposure[t] = S1_ensemble_state[t] * scale[t]`, bounded to `[0, cap]` (cap
taken from whichever S2 cell is active; since S1's state is already in [0,1], the product's
natural ceiling is `cap` itself and no separate clip is needed beyond the `scale[t]` clip already
applied above).

## S3 — calendar tilts (frozen; small, capped, a minor attribution layer, not the program's main
thesis)

- **(a) Turn-of-month.** A +25% RELATIVE exposure boost (i.e. `exposure[t] *= 1.25`, still capped
  at S2's own cap for the active cell), applied over the trading-day window comprising the last 3
  trading days of the current calendar month plus the first 2 trading days of the next calendar
  month (5 trading days total per month boundary).
- **(b) Pre-FOMC.** The same +25% relative boost applied over the 24 hours immediately preceding
  each REGULARLY-SCHEDULED FOMC announcement, per the Data Acquisition outcome above (`data/fomc/
  fomc_dates.csv`, 67 meetings, `announcement_date` column = the statement-release day). Since
  this program marks P&L daily rather than intraday, "the 24 hours immediately preceding the
  announcement" is operationalized as the FULL TRADING DAY immediately preceding each
  `announcement_date` (i.e. exposure[announcement_date - 1 trading day] gets the boost) — this is
  the closest daily-granularity analog to a 24h-pre-announcement window and is declared explicitly
  here as the operationalization, not left ambiguous in the pipeline code. Unscheduled/emergency
  meetings (2020-03-02, 2020-03-15 unscheduled meetings; 2019-10-04 unscheduled conference call)
  are explicitly EXCLUDED from this tilt since they are not knowable/tradeable ex-ante — none of
  these three dates appear in `fomc_dates.csv` and therefore none can trigger this tilt by
  construction.

**Separable attribution (mandatory)**: the deployable construction is run and reported BOTH with
and without S3's tilts (two full metrics rows), so the tilts' marginal contribution to
Sharpe/CAGR/maxDD is directly visible and judgeable independently of S1+S2's own contribution.
Stated plainly now, before any run: this marginal contribution is EXPECTED to be small and
statistically fragile given only ~8 years / ~96 turn-of-month events and 67 FOMC events in the
sample — this is a minor, honest attribution exercise, not a claim of a robustly-proven edge.

**S3 grid cells: with-tilts vs. without-tilts = 2 variants** (applied as a multiplier on top of
whichever S1 x S2 cell is active; not an independent standalone signal).

## Pre-named deployable (decided now, before any run — never adjusted after seeing any cell's P&L)

**50/50 SPY/QQQ blend, S1 trend ensemble (all three rules, equal-weight) x S2 vol-target at
W_vol=60 trading days / cap=1.0 (the single most conservative cell in the declared S2 grid —
de-risk-only, longer/smoother vol window) x S3 both tilts active, at the 13bp primary cost
scenario, daily marks, decide-at-close/act-next-day convention throughout.**

The other grid cells (W_vol=20, cap=1.5, per-instrument-only books, the no-tilt variant) form the
declared ROBUSTNESS FAMILY reported alongside — they are not alternative picks, and the deployable
does not change based on which cell looks best after the run, identical discipline to E1/E2/E3's
pre-naming rule. The 50/50 blend (rather than either single name) is the deployable specifically
because a single-instrument pick would be a more concentrated, less diversified bet on one index's
idiosyncratic regime (QQQ's tech-heavy melt-up in particular) — the blend is the more honest
"broad US equity beta, timed" construction the thesis is actually about.

## AVOIDING E1'S/E2'S/E3'S FOUR APPLICABLE CARRIED-FORWARD BUG PATTERNS (mandatory, declared
before any code is written — verified FRESH in this document, not assumed automatically inherited)

**(1) Source-aware returns.** Every `data/stitched/SPYUSDT.parquet` and `data/stitched/
QQQUSDT.parquet` read for S1/S2/S3's core signal and return computation filters to
`df[df.source=='duka']` BEFORE any resampling, SMA computation, or return computation — the
underlying-to-perp splice seam is stamped `gap_before=False` and produces fake returns if crossed.
`i1_pipeline.py` reuses `e2_pipeline.py`/`e3_pipeline.py`'s `load_stitched_duka(symbol)`
filter-by-construction pattern verbatim (repointed at this worktree's `REPO`, see below), and its
`verify_no_perp_rows_in_signal_panel()`-style check is re-run fresh for SPYUSDT/QQQUSDT
specifically before any signal is trusted (not assumed to still hold just because it held for
other symbols in E1/E2/E3's panel).

**(2) Derived annualization.** `periods_per_yr = n_obs / span_years` is computed per net-return
array, asserted near the expected constant, never hardcoded — reusing
`derive_periods_per_yr(dates, label, expect_weekly)`'s exact pattern from E2/E3 (which prints
`n_obs`/`span_years`/the derived arithmetic and asserts within 5% of the expected constant). This
program marks P&L DAILY (the overlay is held through time and marked every trading day even on
days it does not trade) — expected to derive to roughly 252, but DERIVED explicitly and shown with
the full arithmetic for the deployable in `I1_REPORT.md`, exactly as E2/E3 did, never assumed or
hardcoded as `252` directly in a formula.

**(3) Exposure-timing alignment — THE CENTRAL RISK OF THIS ENTIRE PROGRAM.** Every component here
(S1's trend state, S2's vol-target scale, S3's calendar tilts) IS a time-varying exposure array.
The exposure value applied to day t's return must be computable STRICTLY from information
available at day t-1's close — decide-at-close, act-next-day, used everywhere in this program,
declared here explicitly as the binding convention for every single exposure construction below,
with no exceptions:

- S1(a) (200-day SMA + band): the SMA level, the band thresholds, and the hysteresis STATE used to
  set day t's exposure are all computed from `close[t-1]` and earlier only. The hysteresis "hold
  prior state" logic is itself a potential trap — the state being held must be the state AS OF
  t-1's close (i.e. `state[t] = f(state[t-1], close[t-1] vs bands computed through t-1)`), never a
  state that has already incorporated day t's own close.
- S1(b)/(c) (monthly rules): the trailing-252-day return and the 10-month SMA used to set THIS
  month's exposure are both computed using data strictly through the PRIOR month's final trading
  day, not the current month's own most recent (possibly still-forming) data. Evaluated monthly
  means the exposure state is set ONCE at each month's start and held constant through that
  month's daily marks — this is itself an exposure-timing claim that must be traced explicitly:
  the array index for "month t's exposure decision" is the PRIOR month-end close, and that same
  scalar exposure value is broadcast across every trading day of the CURRENT month's return array,
  not recomputed daily within the month.
- S2 (vol-target scale): `realized_vol[t]` is computed over `returns[t-W:t-1]` (a trailing window
  ending at t-1, NEVER including `returns[t]` itself) — reusing `regime/overlay.py::
  vol_target_scale`'s half-open-interval, excludes-t-itself convention and `metrics()`'s internal
  `vol_target()` helper's `.shift(1)` pattern, both already-correct templates identified during
  Setup, but re-implemented here at DAILY cadence with a periods_per_yr DERIVED per bug pattern
  (2), not the imported weekly `PERIODS_PER_YEAR≈52.18` constant from `convexity.system.metrics`
  (that constant is for the crypto book's weekly cadence and is NOT applicable here).
- S3 tilts: the turn-of-month window (last-3/first-2 trading days) and the pre-FOMC window
  (trading day immediately before `announcement_date`) are both determined from the CALENDAR
  (trading-day position within the month; a fixed, publicly-known FOMC date), not from any
  realized return or price data — there is no lookahead vector through the tilt WINDOW definition
  itself, but the tilt's INTERACTION with S1xS2's own exposure must still be checked: the tilt
  multiplies whatever S1xS2 exposure value was ALREADY correctly lagged per the rules above, it
  does not introduce a new same-day data dependency.

Before trusting any exposure construction, `i1_pipeline.py` prints an explicit diagnostic tracing
which calendar date each array index corresponds to for BOTH the return series and EVERY
exposure-scale series (S1 state, S2 scale, S3 multiplier, combined exposure), inspected by hand.
This is exactly the bug class that was E2's real, adversarially-discovered lookahead bug (a stress
multiplier for week t built from week t's own end-of-week reading instead of the start-of-week
reading) — an END-of-period-vs-START-of-period exposure misalignment. The Independent Verify phase
of this program will specifically re-derive this via a SYNTHETIC-OUTLIER INJECTION TEST: inject a
large fake move at one index and confirm no exposure value downstream of it reacts before the
correct lag. `i1_pipeline.py`'s exposure-construction functions are written now, from the start,
to survive that test (every exposure array is built via an explicit `.shift(1)`-equivalent or
strictly-prior-index slice, never a same-index computation followed by an implicit assumption that
"it'll be used next period anyway") — the naive version is not written first and hoped to pass;
the shift-safe version is the only version written.

**(4) Costs scale with realized exposure changes.** Trading cost = 13bp round-trip primary
({6.5, 26}bp scenarios) charged on the day's `|delta-exposure|` — a full trend flip (exposure
0->1 or 1->0) incurs the full round-trip cost, a small vol-target nudge (e.g. exposure 0.62->0.65)
incurs a proportional fraction of that cost, and zero change incurs zero cost. This directly
reuses the already-correct turnover-scaled fee pattern from `e3_pipeline.py`'s `apply_costs()`
(`turn_scaled = turn_raw * es`), adapted from a portfolio-turnover array to this program's
single-scalar daily exposure-delta array. PLUS funding at +-5bp/week scenarios PRORATED TO DAILY
and SCALED BY the day's actual gross exposure held (not a flat charge regardless of exposure —
this is exactly E2's flat-funding bug and E3's cadence-conversion bug, both explicitly avoided here
from the start). **Proration convention, declared and justified**: `funding_bp_per_day =
funding_bp_per_wk / 7` (calendar days, not 5 trading days) — because funding/borrow-style carry on
a real cash equity or equity-exposure instrument accrues on calendar time (weekends and holidays
are not free), matching how overnight financing actually works economically for a levered or
short equity position, unlike a trading fee which only fires on realized trading-day turnover.
This daily rate is then multiplied by the day's actual combined exposure (`S1 x S2 x S3`) for that
day, every day, never a flat per-period constant regardless of exposure — verified by an explicit
unit check (funding-scenario deltas must themselves scale with the exposure array, not appear as
an exposure-independent flat shift, before any funding-scenario number is trusted).

**Leveraged/inverse-ETF exclusion (E2/E3's landmine 5) — explicitly MOOT here.** SPY and QQQ are
plain, unleveraged index ETFs; this exclusion check does not apply to this program's 2-name
universe by construction. Stated explicitly as a deliberate non-applicable check, per the task's
own instruction, rather than silently omitted.

**Survivorship — structurally different from E1/E2/E3, stated explicitly.** SPY/QQQ are not
survivor-selected individual stocks (unlike the 93-110-name rosters in E1/E2/E3), so the dominant
survivorship concern here is NOT "delisted losers are invisible" but rather that the 2018-2026
sample is dominated by one exceptional secular tech bull regime for QQQ specifically. Per-year and
per-regime reporting (S1's regime-sanity clauses, the per-year Sharpe table, and the 2023-2024
melt-up discussion in the Adoption Gate section below) exists partly to surface this honestly, not
to hide it inside one full-sample average.

**Demo gate (E2/E3's landmine 5 / lesson 7).** SPYUSDT and QQQUSDT are BOTH confirmed
`on_demo=true` in `src/convexity/panel/candidates.md` (fresh `load_universe()` call performed in
Setup, not memory) — a paper deployment of the eventual book is actually possible. Restated here
per the methodology's demo-gate reporting convention.

## Reused machinery (declared now, repointed for this worktree)

`i1_pipeline.py` reuses `e3_pipeline.py`'s (most bug-fixed) `metrics()` / `cpcv_paths()` / `dsr()`
/ `per_obs_sharpe()` / `per_year_sharpe()` / `pbo_cscv()` function bodies verbatim, parameterized
by a DERIVED `periods_per_yr` per bug pattern (2). `i1_pipeline.py` points its own paths at
`/home/speculation/convexity-indices/data/{stitched,fomc}` and
`src/convexity/panel/candidates.md` directly — it does NOT reuse `e2_pipeline.py`/
`e3_pipeline.py`'s hardcoded `REPO = Path('/home/speculation/convexity-equity')` (that worktree no
longer exists, merged and deleted per Setup's own framing); `REPO` is redefined to
`/home/speculation/convexity-indices` at the top of `i1_pipeline.py`. `i1_pipeline.py` does NOT
call `convexity.panel.panel.build_panel()` / `load_bars()` for the core signal — `load_bars()`
does not read the `source` column and would silently reintroduce the splice-seam bug (bug pattern
1). `regime/overlay.py`'s `vol_target_scale()`/`drawdown_stop_scale()` are portfolio-level,
weekly-cadence, `XSecBook`-coupled constructions confirmed in Setup as NOT directly reusable for
this program's single-scalar-per-day overlay on 2 instruments — `i1_pipeline.py` writes its own
simpler daily-cadence vol-target following the same no-lookahead/shift(1) pattern in spirit, with
`periods_per_yr` DERIVED per bug pattern (2), not the imported weekly constant.

## N_trials ledger (declared in full before any run; INFORMATIONAL per PROGRAM CLASS — feeds DSR/
PBO, does NOT gate adoption)

- **S1: 1** — the ensemble is the only S1 construction ever scored as a tradeable cell (see S1
  section above for the explicit reasoning on why the 3 individual rules are not separately
  credited here, unlike E1/E2/E3's per-lookback crediting convention).
- **S2: 2 (W_vol) x 2 (cap) = 4** cells.
- **S3: with-tilts vs. without-tilts = 2** variants.
- **Base construction cells so far: 1 x 4 x 2 = 8** (S1 is a fixed multiplier of 1 since there is
  only one S1 construction, so this is really S2 x S3 = 8 distinct exposure constructions).
- **Cost x funding scenarios**: {6.5, 13, 26}bp trading cost x {0, +5, -5}bp/week funding = 3 x 3 =
  **9** scenarios, applied to EACH of the 8 base construction cells, AND to the no-overlay
  buy-and-hold benchmark itself (the benchmark is costed too, at minimum an initial-purchase cost
  and funding drag, for the comparison to be fair under bug pattern (4)'s logic) — benchmark
  scenarios are reported alongside but are not separately-credited "trials" since the benchmark is
  not a competing construction, it is the comparator every cell is measured against.
- **Overlay trial-cells: 8 x 9 = 72.**
- **Per-instrument decomposition (SPY-only, QQQ-only, 50/50 blend) = 3 variants**, reported for
  TRANSPARENCY, declared here as a REPORTING BREAKDOWN of the same trial, not separate trials: the
  50/50 blend IS the deployable, and the per-instrument numbers exist purely to show whether the
  blend's performance is being driven by one name or is genuinely shared — they are diagnostic,
  not alternative constructions competing for the deployable slot. Not added to N_trials.
- **Lookahead canaries** (one per component: S1 trend-state canary, S2 vol-scale canary, S3
  tilt-window canary) — excluded from N_trials per convention, reported separately in
  `I1_REPORT.md`.

**Final N_trials = 72.** This feeds an INFORMATIONAL DSR/PBO figure only (per PROGRAM CLASS above)
— it is emphatically NOT the adoption gate for this program. Reiterated here: even at this
relatively modest N_trials=72 (small compared to E1's 179 or E3's 11), a 2-instrument, ~8-year
overlay family is not the kind of large, richly-varied trial population DSR was designed to
penalize meaningfully — DSR is expected to come back LOW regardless of whether the underlying
overlay construction is a genuinely good idea, and `I1_REPORT.md` states this plainly rather than
treating a low DSR as itself a negative finding.

## Validation battery

CPCV(8 groups, k=2, embargo >=1 month) reported for the deployable. CSCV-style PBO over the full
72-cell family reported informationally (not gating, same PROGRAM CLASS reasoning as DSR).
Drop-top-5-periods Sharpe reported for the deployable. Per-year AND per-regime metrics (regime
windows = the four S1 regime-sanity windows declared above: 2020-03 COVID, 2020-H2/2021 recovery,
2018-Q4 chop, mid-2022 chop/decline, plus 2022 full-year and 2023-2024 melt-up as their own
reporting windows). Full metrics table per cell: net Sharpe, EV bp/obs, win%, maxDD, CAGR raw +
at-10%-vol, turnover via `turn=` (wired and populated, not left as the E2-documented
silently-`None` bug), GROSS shown alongside NET everywhere, vs. the 50/50 SPY/QQQ buy-and-hold
benchmark computed on the IDENTICAL daily series with IDENTICAL conventions (decide-at-close/
act-next-day is moot for a static buy-and-hold's own exposure, since it never changes, but the
COST and FUNDING conventions applied to the benchmark must be identical in form to those applied
to the overlay — this fairness check is itself one of the three mandatory adversarial lenses
below).

**Three mandatory adversarial lenses (stated explicitly, to be run in the Independent Verify
phase, designed for from the start in `i1_pipeline.py`):**
1. **Exposure-timing synthetic-outlier injection** (per bug pattern (3) above) — inject a large
   fake move at one index, confirm no exposure value downstream reacts before the correct lag.
2. **Benchmark-construction fairness check** — confirm the buy-and-hold benchmark is computed on
   the IDENTICAL `duka`-filtered daily return series, with the IDENTICAL cost/funding scenario
   machinery (just held at constant exposure=1 rather than the overlay's time-varying exposure),
   not a separately-sourced or differently-conventioned "index return" that would make the
   comparison invalid before any Sharpe is even compared.
3. **Lookahead canary battery** (per component, see below) — each canary must show an inflated,
   obviously-too-good Sharpe; absence of inflation on any canary triggers a re-audit of that
   component's honest pipeline before anything from it is reported.

**Lookahead canary per component (mandatory, excluded from real results/family/DSR/PBO
reporting):**
- **S1 canary**: trend state computed using day t's own close (not t-1's) — e.g. the 200-day SMA
  band and the monthly rules' lookback windows both include the day/month whose return they are
  about to be applied to.
- **S2 canary**: vol scale uses same-day realized vol (the trailing window includes `returns[t]`
  itself, not just through t-1).
- **S3 canary**: tilt windows shifted to overlap themselves (e.g. the turn-of-month window is
  applied to the SAME days used to detect it, or the pre-FOMC window is shifted onto or after the
  announcement date itself rather than strictly before it).

Each must show a large, fake, obviously-too-good Sharpe relative to its honest counterpart; if any
does not, the honest pipeline for that component is treated as suspect for the same bug and is
re-audited before its result is reported.

## Full test suite (mandatory before every commit)

`cd /home/speculation/convexity-indices && PYTHONPATH=/home/speculation/convexity-indices/src:/home/speculation/convexity-indices /home/speculation/convexity/.venv/bin/python -m pytest tests -q`
must pass at >=654 passed/1 skipped (the confirmed-unchanged baseline through E1/E2/E3) before
every commit in this program; the pass count is included in every report.

## Adoption gate (frozen, pre-outcome, BENCHMARK-RELATIVE per PROGRAM CLASS — NOT a DSR-based gate)

Recommend a separate PAPER-FIRST timed-beta book ONLY IF ALL of the following hold, net of the
13bp primary cost scenario, measured against the 50/50 SPY/QQQ buy-and-hold benchmark computed on
the IDENTICAL daily return series with IDENTICAL conventions (checked by adversarial lens 2 above):

1. **Net Sharpe**: deployable net Sharpe > benchmark net Sharpe + 0.10.
2. **Drawdown**: deployable maxDD < 0.65 x benchmark maxDD.
3. **Regime sanity**: every S1 regime-sanity clause above holds, INCLUDING the honest whipsaw-cost
   admission in 2018-Q4/2022 — a suspiciously clean curve with no whipsaw cost anywhere FAILS this
   gate on suspicion of a construction bug, exactly as a smooth equity curve through a known
   drought period was treated as a red flag in `DURABLE_REPORT.md`.
4. **Robustness**: every cell in the declared grid family (both `W_vol` values, both cap values,
   with and without tilts — the full 8-cell base-construction family) individually beats the
   benchmark on net Sharpe. A result that only works at one specific parameter corner is REJECTED
   regardless of the deployable cell's own number.
5. **Drop-top-5-periods**: the deployable's drop-top-5-periods Sharpe still beats the benchmark's
   own (undropped) Sharpe.
6. **Turnover sanity**: realized turnover is consistent with the low-trade-count expectation
   (roughly 5-15 full round-trip-equivalent exposure changes per year). Materially higher turnover
   is investigated and reported explicitly, since it would indicate the hysteresis band or
   vol-target smoothing is not functioning as intended.

DSR and PBO at the full N_trials=72 are reported ALONGSIDE this gate as informational numbers with
their PROGRAM CLASS caveat restated (expected to be low at breadth=2, this does NOT by itself
invalidate the finding) — never omitted, never used as a pass/fail criterion here.

If ANY of gates (1)-(6) fails, the negative is recorded plainly as this program's conclusion and
deployment is recommended AGAINST — identical discipline to E1/E2/E3: a correctly-identified
negative is a complete, valuable result, not a failure to find one.

**Honest expectation, stated before any run**: the deployable is expected to land around 0.7-0.9
net Sharpe against a benchmark around 0.5-0.7 in this sample, with the 2023-2024 melt-up likely
being a period where the overlay LAGS buy-and-hold. This is stated explicitly now as an
anticipated, ACCEPTABLE admission price of the strategy, not a failure if observed — `I1_REPORT.md`
reports the actual 2023-2024 numbers either way, without softening a negative-looking window into
something it isn't.

## What this program will NOT do

No shorting the index (S1's own reasoning above). No leverage beyond S2's declared cap grid
{1.0, 1.5}. No blending into the existing crypto or commodity paper books. No merge to `main`, no
PR, no systemd changes, no live capital — a positive result here recommends, at most, a new
PAPER-FIRST book proposal for a separate, later, explicit decision by the repo owner.
