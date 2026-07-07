# E2 STRESS-CONDITIONED RESIDUAL REVERSAL — PRE-REGISTRATION (Program 6; FROZEN 2026-07-06, before
any pipeline code)

Owner-directed follow-up to E1 (perp-native suite, closed — clean negative, see E1_REPORT.md).
Isolation: runs in the `equity` worktree (/home/speculation/convexity-equity); reads the shared
on-disk lake at `data/stitched/` (symlinked into the same lake as `main` and `commodity` — reads
only); NEVER touches the live paper system (main's checkout, its branch, its systemd timers, or
its `data/paper/` journals) or `src/convexity`'s live-trading paths. Executed by `e2_pipeline.py`
(new, self-contained file under `research/equity/`); outcome to be recorded in `E2_REPORT.md`.

## Thesis (1-2 sentences)

Nagel's liquidity-provision logic applied cross-sectionally to equities: names whose return, after
removing their sector's common move, was most extreme this week are noise/inventory-pressure driven
and should partially mean-revert next week, and this reversal premium should be fatter precisely
when fast capital is scarce (genuine market stress, proxied by elevated realized vol in SPYUSDT/
QQQUSDT) — this is the equity RV study (breadth ~100 cross-sectional spread reversion, the direct
analog of the commodity named-pairs RV book, implemented cross-sectionally because single share
classes here offer no coherent named-pairs opportunity; see RV NOTE below).

## RV note for the record (deliberate scoping decision, not to be re-litigated)

This program IS the equity RV study. Reversal-vs-sector-peers is spread mean reversion at
cross-sectional breadth ~100 — the direct analog of the commodity RV pair book
(`research/entry_exit/study/RV_REPORT.md`) implemented cross-sectionally rather than as named
pairs, because (a) a named-pairs book on mega-caps is a well-documented anti-edge in the literature
(crowded, decayed post-2002), and (b) this venue only lists single share classes (no dual-class/ADR
pairs to exploit). No separate named-pairs equity study will be built.

## Universe / data / calendar

- **Base universe**: the 102 names in `src/convexity/panel/candidates.md`'s `equities` array with a
  non-empty `dukascopy` field (legacy key, holds the databento code) — i.e. the names with real
  underlying history, per DATA.md and the CONTEXT verification (110 total equities, 102 with real
  history, 8 perp-only excluded entirely from E2: ANTHROPICUSDT, BBXUSDT, BSPUSDT, HYUNDAIUSDT,
  OPENAIUSDT, QNTXUSDT, SAMSUNGUSDT, SKHYNIXUSDT).
- **Point-in-time discipline**: a name enters the weekly cross-section only from its own first real
  `source=='duka'` trading day onward (per DATA.md's manifest: span from 2018-05-01 for the 68
  seasoned names, later dates for genuine new listings, e.g. CBRSUSDT from 2026-05-14). No
  backfill before a name's real listing date, no hold-through after a delist (none of the 102 have
  delisted as of the data cut, but the construction must not assume this).
- **Data source**: `data/stitched/{SYMBOL}.parquet`, columns
  `open_time|open|high|low|close|volume|source|gap_before`, filtered to `source=='duka'` ONLY for
  every step of the core signal (panel construction, residualization, ranking, P&L) — the core
  signal never reads a `source=='perp'` row. This is stronger than "avoid crossing the seam": E2's
  panel is built from a `source=='duka'`-only slice of each file from the start, so the seam
  physically cannot enter the daily-close series (see AVOIDING E1's BUGS below for the explicit
  verification this claim gets before being trusted).
- **Sector groups** (residualization key, `candidates.md`'s `sector` field, verified 2026-07-06 via
  `convexity.panel.candidates.load_universe()`): 46 distinct tags across 110 equities. Counts among
  the 102 with real history: semis=13, thematic=6 (2 of the 8 thematic-tagged names are perp-only
  and excluded), hardware=6, software=6, semicap=5, optical=4, financials=4, consumer-disc=3,
  internet=3, space=2, bitcoin-proxy=2, networking=2, staples=2, ai-cloud=2, media=2, fintech=2,
  healthcare=2, semis-storage=2, plus ~28 singleton tags (china, clean-energy, industrials,
  crypto-exchange, stablecoin, software-security, gaming-betting, semis-thematic, the 4
  country-etf-* tags, energy-storage, retail-meme, health-consumer, it-services, ai-crypto-infra,
  index-smallcap, telecom, drones, data-ai, index-largetech, ev, leveraged-etf-semis, space-proxy,
  index-sp500, inverse-etf-nasdaq, leveraged-etf-nasdaq, gaming, mobility, uranium-etf, rare-earth,
  vol-etf, energy-etf). **Declared plainly**: only 6 tags (semis=13, thematic=6, hardware=6,
  software=6, semicap=5, optical=4) clear a 4-name floor in the STATIC (present-day, full-roster)
  count; the other ~40 tags are 1-3 names each and fold into "other" even once every name that will
  ever join them has listed. The point-in-time universe is much smaller still in early weeks
  (2018-2020 has only the 68 seasoned names live, and several of even the 6 qualifying tags are
  thinner in earlier years than their eventual full-roster count), so the "other" bucket will hold
  the large majority of names in most weeks, especially early in the sample — this is reported as a
  headline caveat on every residualization result, not buried. The residualization groups are still
  computed correctly per the stated rule (fold sectors with <4 point-in-time members that week into
  "other" for that week, recomputed fresh every week as the universe grows) — declared here in full
  since the actual effect is a de facto SIX-named-sector-plus-one-large-"other"-bucket scheme for
  most of the sample, with the six named sectors' own point-in-time membership growing (and
  "other"'s share of the panel shrinking, though not disappearing) over the 8.18yr span as more
  names list. The full symbol→sector map is reproduced in `e2_pipeline.py`'s `SECTOR_MAP` constant,
  sourced directly from `candidates.md` at pipeline-write time (not hand-retyped) and asserted
  against a fresh `load_universe()` call at pipeline run time so drift between this document and
  the live `candidates.md` is caught, not silently trusted.
- **Calendar**: weekly, Friday-close UTC (reusing `weekly_friday_close`'s as-of/ffill-onto-W-FRI
  convention from `e1_pipeline.py`, adapted to the `source=='duka'`-only daily close series — see
  below), spanning each name's own real history through the lake's last bar (2026-07-04 per
  DATA.md), full universe span 2018-05-01 to 2026-07-04 (~8.18yr, ~427 Fridays) for the seasoned
  names, point-in-time-shorter for newer listings.

## AVOIDING E1'S BUG PATTERNS (mandatory, declared before any code is written)

E1 found two real, previously-undocumented pipeline bugs (see E1_REPORT.md's "Independent
verification and adversarial review" section and STATE.md's tail). Both are exactly the class of
defect E2 must be checked against from the start, not discovered post-hoc. Concretely:

**1. Source-aware daily-return construction (E1 bug (a)).** E1's `daily_close_series` correctly
excluded `gap_before==True` rows, but its `daily_returns_from_close` guarded the resulting
pct_change only on `gap_days > 5` calendar days — insufficient, because 55/110 names' splice seam
lands ~2 TRADING days apart (well under the 5-calendar-day guard) and is stamped `gap_before=False`
(the perp lists 9-25 minutes into the RTH session, not at a UTC day boundary, so it never trips the
flag). E1's own `verify_seam_is_gapped` function existed only as a logged WARNING, not an actual
excluding guard.

E2's fix, declared now: **the panel-construction step filters every symbol's parquet to
`df[df["source"] == "duka"]` BEFORE any daily-close resampling, ranking, or return computation
happens.** The `source=='perp'` rows are never loaded into the core-signal pipeline at all — not
filtered out after the fact, not guarded against with a boundary check, simply never read past the
initial filter. This means there is no seam for the core signal to cross, by construction, not by
vigilance. This claim is explicitly verified, not assumed: `e2_pipeline.py` includes a
`verify_no_perp_rows_in_signal_panel()` check that (a) asserts every row consumed by the daily-close
builder has `source=='duka'`, and (b) for at least one known splice-seam name (PAYPUSDT, verified in
prereg drafting: seam at `open_time=1774276200000`, `source` flips duka→perp, `gap_before=False` at
the seam row itself — i.e. exactly the E1 bug pattern, confirmed present in this lake) directly
confirms the constructed daily-close series terminates at (or before) the last `duka` day and never
contains a `perp`-sourced close. This check runs and its pass/fail is printed in `E2_REPORT.md`
before any headline number is reported — not assumed silently correct because "we filtered by
source this time."

**2. Annualization constant matched to true observation spacing, derived not assumed (E1 bug
(b)).** E1 hardcoded `WK_PER_YR=52.0` for every weekly-rebalanced cell without checking that the
underlying net-return array was actually weekly-spaced (it was daily-marked in S1/S3, understating
every Sharpe by ~2.2x). E2's reversal signal both rebalances weekly (Friday-close to Friday-close)
AND is scored weekly (one net-return observation per week, no daily marking in between, since there
is no daily P&L construction anywhere in E2's core signal) — so `periods_per_yr ≈ 52` is expected to
be correct here, unlike E1. This expectation is verified, not assumed: for every cell, `e2_pipeline.
py` computes `n_obs`, `span_years = (dates[-1] - dates[0]).days / 365.25`, and
`periods_per_yr_derived = n_obs / span_years`, and asserts this is within 5% of 52.18
(`365.25/7`, the Friday-to-Friday constant) before calling `metrics()`/`dsr()`/`cpcv_paths()` with
that derived value (not a hardcoded `52.0` literal) — if a future cell (e.g. a daily-marked overlay
sub-computation) has different spacing, the derived value is used for THAT cell, never inherited
from the weekly convention by assumption. The arithmetic (n_obs, span_in_years,
periods_per_year = n_obs/span_in_years) for the pre-named deployable is shown explicitly in
`E2_REPORT.md` so a reader can check it themselves, per METHODOLOGY (1c).

**Path hygiene** (per E1's own STATE.md note on this exact issue): `e2_pipeline.py` points its own
paths at `/home/speculation/convexity-equity/data/{stitched,funding}` and
`src/convexity/panel/candidates.md` directly — it does NOT reuse `swing_pipeline.py`'s hardcoded
main-repo paths, and does NOT call `convexity.panel.panel.build_panel()`/`load_bars()` for the core
signal's daily-close construction, because `load_bars()` does not even read the `source` column
(verified by inspecting `src/convexity/panel/data.py`'s `load_bars`: its `pd.read_parquet(...,
columns=[...])` list omits `source` entirely) and `build_panel()`'s `_daily_close` takes the last
1-min close of each UTC day with no source-awareness — calling it naively on the raw stitched file
would silently splice through the seam exactly like E1's bug. E2 writes its own small, source-aware
daily-close builder instead (filtering to `source=='duka'` first, per above), and reuses
`e1_pipeline.py`'s `metrics()`/`dsr()`/`cpcv_paths()`/`per_year_sharpe()`/`per_obs_sharpe()`
function bodies (parameterized-by-`periods_per_yr` versions, not `swing_pipeline.py`'s hardcoded
ones) verbatim or near-verbatim for the validation battery, and `regime/overlay.py`'s
`vol_target_scale`/`apply_scale` pattern for the book-level vol-target overlay ONLY after confirming
E2's own series matches (or is adapted to differ from) that module's hardcoded
`PERIODS_PER_YEAR=365.25/7` assumption — since E2 is weekly-marked, this is expected to match, but
is verified via the same `periods_per_yr_derived` check above before trusting `vol_target_scale`'s
internal annualization, not assumed compatible because "it's also weekly."

## Signal construction (frozen)

1. **Panel**: daily close per name from `source=='duka'` rows only (last such row per UTC calendar
   day), point-in-time masked to each name's own first/last real `duka` day. Weekly (Friday) close
   via as-of/ffill onto the W-FRI calendar, reusing `weekly_friday_close`'s convention.
2. **Weekly return**: `wret[t] = close[Friday t] / close[Friday t-1] − 1` (plain return, no
   dividend/adjustment beyond the already-applied split-adjustment in the lake per DATA.md).
   Forward-return convention for signal scoring: `wret[t]` means the return from week `t`'s
   Friday-close to week `t+1`'s Friday-close; a signal computed at week `t` may use information at
   or before week `t`'s close only, and is scored against `wret[t]` (t→t+1).
3. **Residualization**: within each week's point-in-time universe, group names by `sector`
   (candidates.md field; sectors with <4 point-in-time members that week fold into a single "other"
   bucket for that week, recomputed fresh every week). Demean each name's raw weekly return by its
   group's cross-sectional mean that week: `resid[i,t] = wret[i,t] − mean(wret[j,t] for j in
   group(i,t))`.
4. **Reversal lookback band (frozen grid, declared in full — this IS the multiplicity ledger)**:
   trailing residual return over L trading days, L ∈ {3, 5, 10} (i.e. approx. the most recent
   ~0.6/1/2 weeks of residual daily moves, computed from the same `source=='duka'`-only DAILY
   close series before the Friday-sampling step — NOT a 1/2/3-week-lag on already-weekly-sampled
   data, since the grid is stated in TRADING DAYS per the task's declared grid). Deployed as an
   EQUAL-WEIGHT ENSEMBLE of the three per-lookback books — never argmax a single lookback.
5. **Ranking & book**: at each Friday, rank names by trailing-L residual return (most negative =
   biggest residual LOSER, most positive = biggest residual WINNER). Quintile book: LONG the
   bottom quintile (biggest losers, reversal-sense) / SHORT the top quintile (biggest winners) —
   **sign check, stated explicitly**: this is REVERSAL, so the book is long losers / short winners;
   a momentum sign (long winners/short losers) would be a serious and easy-to-make error and is
   explicitly the WRONG sign for this study. Position sizing: inverse trailing-20-trading-day
   realized-vol within each leg, floored at 0.5× the cross-sectional median vol that week (reusing
   `_weights_from_signal`'s floor convention from `swing_pipeline.py`, cited in the task and
   verified present there). Dollar-neutral, unit gross book before any overlay.
6. **10% vol-target overlay**: applied on top of the unit-gross book, reusing
   `regime/overlay.py::vol_target_scale`/`apply_scale`'s convention (target_vol_ann=0.10), subject
   to the periods_per_yr compatibility check declared above.

## Conditioned variant — the pre-named deployable's conditioning layer

**Stress signal**: realized-vol z-score of SPYUSDT and/or QQQUSDT (from the same `source=='duka'`
underlying-history lake, computed identically to the core panel's daily-close construction — same
source filter, same point-in-time discipline, though both names span the full 2018-05-01 window so
point-in-time truncation is moot for the stress signal itself).
- **Realized-vol window grid**: rolling realized vol of daily log returns over W_rv ∈ {10, 20, 40}
  trading days — 3 variants.
- **Z-score lookback grid**: the realized-vol series is itself z-scored over a longer trailing
  lookback W_z ∈ {252, 504} trading days (≈1yr, ≈2yr — long enough to capture "genuine stress" as a
  multi-regime-relative concept, not a same-quarter recency artifact) — 2 variants.
- **Threshold/gating grid**: two conditioning mechanisms, each with its own threshold grid:
  - **(a) Hard gate**: book gross exposure = 1.0 if z ≥ threshold, else 0.0 (flat when calm).
    Threshold ∈ {0.5, 1.0, 1.5} — 3 variants.
  - **(b) Continuous scale**: book gross exposure = `clip(1 + max(0, z), 1, 3)` (scales UP with
    stress, never below unit gross, capped at 3x) — 1 variant (no threshold to sweep, the scaling
    function itself is the declared form).
  - Combined per (SPYUSDT vs QQQUSDT vs mean-of-both) stress-source choice — 3 variants — this
    multiplies the grid further (declared below in the N_trials ledger).
- **SPYUSDT vs QQQUSDT vs mean-of-both** — 3 stress-source variants, as noted above.

This is the regime-sanity mechanism check itself, not a separate afterthought (per METHODOLOGY): if
the conditioned book's P&L does not concentrate in real stress windows, the whole durability
thesis is undermined regardless of the backtest Sharpe (see Regime sanity signature below).

## Universe-size sweep (frozen grid)

N ∈ {20, 30, 40, 50, 60, 70, 80, 90, 100, 102}, ranked by trailing 20-trading-day dollar volume
(`close * volume` from the `source=='duka'` leg — real share-volume dollar turnover, a genuine
liquidity proxy, computed causally at each Friday from the prior 20 trading days, recomputed fresh
every week as the point-in-time universe and volume ranks change) — **10 sweep points**. Applied to
the pre-named deployable's exact construction (band-ensemble lookback, chosen conditioning, see
below) — not to every grid cell (that would multiply the ledger by 10x for no incremental
information; the sweep's purpose is to answer "does breadth matter for THE deployable," not to
re-run the entire grid at every breadth).

**Honest CI-overlap treatment (declared now, per METHODOLOGY)**: for each N, block-bootstrap the
weekly net-return series (block length = 8 weeks, matching the CPCV embargo below) 1000 times to
get a 90% Sharpe confidence interval. The frozen, declared criterion for calling a peak at some N*
real (`peak_beats_baseline_honest = True`) is: **N*'s CI lower bound must exceed N=102's (baseline)
CI lower bound, AND the two CIs' overlap (as a fraction of the narrower CI's width) must be less
than 50%** — i.e. a clearly non-overlapping or only-marginally-overlapping improvement, not a peak
that is visually appealing but statistically indistinguishable from baseline noise. This is
deliberately less strict than requiring zero overlap (which would rarely trigger at any realistic
sample size) but stricter than a bare point-estimate comparison. Any peak failing this is reported
as "peak inside overlapping noise, not a real breadth effect" — the same principle already applied
to the crypto cross-sectional momentum universe-size study, implemented here from first principles
since that study's own bootstrap code was not directly reusable across repos.

## Breadth decision for the pre-named deployable (declared NOW, before the sweep runs)

**The pre-named deployable commits to the FULL 102-name universe (not a liquidity-filtered
subset).** Reasoning, declared now: (a) this is the more conservative, less-curated choice — using
all available names rather than cherry-picking a breadth after seeing which one scored best; (b) a
short-horizon cross-sectional reversal signal's edge is expected (per Nagel's mechanism) to be
strongest in the NAMES with the LEAST natural liquidity provision (i.e. names fast capital would
most plausibly abandon during stress), so restricting to only the highest-dollar-volume names could
mechanically remove exactly the names carrying the mechanism, not just noise; (c) it matches E1's
own precedent of reporting both full-universe and on_demo-restricted headline numbers without
letting a sweep retroactively redefine "the" deployable. After the sweep runs, `E2_REPORT.md`
states explicitly whether the sweep's honest-CI result SUPPORTS this pre-registered choice (e.g.
"N=102 is inside or beats every honestly-distinguishable peak") or CONTRADICTS it (e.g. "N=40 is a
real, honestly-distinguishable peak strictly above N=102") — but the deployable named here does not
change after the fact either way, per the frozen-grid discipline (a contradicting sweep result is
reported as a robustness finding "the sweep suggests a narrower book would do better," not as a
silent redefinition of what was pre-named).

## Costs (frozen; perp-reality friction on the underlying-built signal)

Because the eventual deployable instrument is the young Binance perp (even though the signal is
built entirely on underlying `duka` history), trades are costed at perp-reality friction, not
cash-equity friction: **primary round-trip 13bp per name-week traded** (Binance USDT-M VIP0 taker
5bp/leg + slippage buffer, matching E1's convention), with declared scenario variants **{6.5bp,
26bp}** — 3 cost scenarios. Plus **funding ±5bp/week** scenarios (2 additional variants: +5bp/wk
funding drag, −5bp/wk funding tailwind, applied uniformly since no real synthetic-perp funding
history exists yet for the intended deployable instrument at this signal's design stage) — reversal
turnover is high (~2x gross/week for a weekly-rebalanced quintile book, since a name can flip from
top-quintile-short to bottom-quintile-long between adjacent weeks), so cost realism matters a great
deal here; GROSS and NET are shown at every single cell in every table, not just net, so a reader
can see exactly how much of any edge cost eats.

## N_trials ledger (the full multiplicity charged to DSR — every cell below is a real cell that
COULD be run and IS run)

- **Unconditional band-ensemble** (the 3-lookback ensemble, no stress conditioning): 1 cell
  (the ensemble itself; the 3 individual per-lookback books are reported for transparency but are
  sub-components of the declared ensemble, not separately-selectable trading cells — they are
  charged as 3 additional trials since each is independently inspectable and reported in the
  results table): **1 (ensemble) + 3 (per-lookback) = 4**.
- **Conditioned variant grid**: W_rv (3) × W_z (2) × stress-source (3) × conditioning-mechanism
  {hard-gate-with-3-thresholds = 3, continuous-scale = 1} (4) = 3×2×3×4 = **72** conditioned cells.
- **Universe-size sweep**: 10 breadth points applied to the ONE pre-named deployable construction
  (full-102, band-ensemble, and — declared now — the SPECIFIC conditioning variant chosen as primary
  within the 72-cell grid above, decided by the regime-sanity signature check, not by ex-post Sharpe
  — see Pre-named deployable below) = **10 cells** (N=102 is already covered by the deployable's own
  headline cell but is re-counted here since it's a genuine grid point in the sweep's own bootstrap
  procedure, not free).
- **Cost scenarios**: 3 primary/{6.5,26}bp × the two funding variants are reported as an ADDITIONAL
  multiplicative axis applied only to the pre-named deployable and the unconditional comparator (not
  to all 76 signal-construction cells above, which would 15x the ledger for no new information about
  which SIGNAL works) — declared as **2 headline cells (deployable + unconditional comparator) × 3
  cost scenarios × 3 funding scenarios (0, +5bp/wk, −5bp/wk, i.e. the "no funding" primary plus the
  two funding stress variants) = 18** cost/funding cells.
- **Lookahead canary**: 1 deliberately-wrong version of the pre-named deployable (excluded from
  family/PBO/DSR per METHODOLOGY, reported separately, NOT counted in N_trials).
- **N_trials = 4 (unconditional) + 72 (conditioned grid) + 10 (universe sweep) + 18 (cost/funding) =
  104.** This is the full, honestly-enumerated ledger charged to DSR's N_trials — no hand-picked
  subset. The canary is excluded per the mandatory convention (it is a bug-detector, not a candidate
  edge).

## Pre-named deployable (decided now, before any run — never "whichever cell looked best")

**The deployable is: the 3-lookback {3,5,10}-trading-day band-ensemble residual-reversal book, full
102-name universe, conditioned by the CONTINUOUS-SCALE stress gate (mechanism (b) above: gross
exposure = clip(1 + max(0, z), 1, 3)) using realized vol over W_rv=20 trading days, z-scored over
W_z=252 trading days, on the MEAN of SPYUSDT and QQQUSDT's stress z (averaging the two reduces
single-index idiosyncratic noise in the stress signal without adding a free parameter), at the
primary 13bp cost scenario with no funding adjustment (0bp/wk) as the headline, {6.5bp, 26bp} and
±5bp/wk funding scenarios reported alongside.**

Reasoning, declared now:
- The band-ensemble (never argmax) is mandated by METHODOLOGY and matches every prior program's
  convention (S1's lookback ensemble in E1, the TSMOM ensemble in DURABLE, the z-window ensemble in
  RV).
- Full 102-name breadth is chosen over a liquidity-filtered subset for the reasons stated in
  "Breadth decision" above — decided before the sweep runs, not after.
- The CONTINUOUS-SCALE conditioning mechanism (not the hard gate) is chosen as primary because the
  thesis is that the reversal premium FATTENS during stress, not that it is ZERO outside stress —
  Nagel's mechanism describes a continuously-varying price of liquidity-provision, not a binary
  on/off state, so a continuous scale is the more faithful implementation of the stated thesis. The
  hard-gate variants are reported as a robustness/comparison family, not the pick.
- W_rv=20 (one trading month) is chosen as the primary realized-vol window because it is the middle
  of the declared {10,20,40} grid and matches the conventional "realized vol" convention used
  elsewhere in this codebase (e.g. the inverse-vol leg-weighting's own 20-day window) — a consistent
  convention across the pipeline rather than a differently-tuned parameter for the conditioning
  layer alone.
- W_z=252 (one trading year) is chosen over 504 (two years) because it is long enough to
  characterize "genuine stress" relative to a full annual cycle without requiring two full years of
  runway before the conditioning layer can activate at all for the shortest-history names.
- Mean of SPYUSDT/QQQUSDT is chosen over either single index alone because both are used
  interchangeably in the task's own framing ("SPYUSDT and/or QQQUSDT") and averaging is the more
  conservative, less-curated choice than picking whichever single index's z-score happens to
  produce the better backtest.
- 13bp primary cost (not 6.5bp) is chosen as the headline because it is this program's own declared
  "primary" scenario (perp-reality round-trip), matching E1's and the commodity programs' identical
  convention of reporting the middle/primary scenario as the headline and the {6.5,26}bp bracket as
  sensitivity, not cherry-picking the cheapest scenario for the headline.

If the conditioned deployable's own gates fail (see Adoption gate below), the deployable
recommendation is NOT silently swapped to the unconditional variant, a different W_rv/W_z, the hard
gate, or a different breadth after the fact — a failed pre-named deployable is reported as the
program's headline result, exactly as the frozen-grid discipline requires, and any individual
positive cell elsewhere in the 104-cell family is reported separately as "does not change the
pre-named deployable's outcome."

## Lookahead canary (mandatory, excluded from real results/family/PBO)

A deliberately-wrong version of the pre-named deployable's construction, where the residual-return
ranking signal at week `t` is computed using week `t`'s OWN return (i.e. the signal peeks at
`wret[t]` before ranking on it, rather than only trailing information through week `t-1`'s close),
scored against the same `wret[t]` (t→t+1) forward-return convention as the honest pipeline. This
must show a large, fake, obviously-too-good Sharpe (a reversal signal that gets to see the return
it's about to trade against should look implausibly good) — if it does not clearly dominate the
honest version, the honest pipeline is treated as suspect for a construction bug and is re-audited
before its result is reported (per E1's own experience where one sleeve's degenerate honest
construction made its canary fail to look "dramatically better" — if that happens here, this is
stated explicitly in `E2_REPORT.md` rather than papered over). The canary is excluded from every
family/PBO/DSR calculation.

## Regime sanity signature (MANDATORY pre-belief check, declared exact signature)

The CONDITIONED sleeve's P&L must visibly CONCENTRATE in genuine stress windows present in this
data's span: **2018 Q4** (Dec 2018 selloff), **2020-03** (COVID crash), **2022** (the full-year
drawdown, esp. Q1-Q3), and **2024-08** (the August 2024 vol spike) — and be thin/flat through calm
stretches (e.g. 2017 [pre-sample here], 2019, most of 2021, 2023-2024 H1). Concretely, declared
before any run: **the exact signature that must hold** is that the conditioned book's cumulative
P&L, when decomposed into "weeks where the stress z was above its own trailing median" vs "weeks
below," shows the ABOVE-median weeks contributing a majority (>50%) of total P&L despite being a
minority (<50%, expected roughly 15-30% given the continuous-scale exposure function's shape) of
total calendar weeks — i.e. P&L per stress-week must be materially higher than P&L per calm-week,
not merely "positive in both." Additionally, the four named windows above (2018 Q4, 2020-03, 2022,
2024-08) must each individually show the conditioned book's weekly Sharpe during that window
exceeding its own full-sample Sharpe — a smooth, steady curve with no visible concentration in these
specific windows is the same red-flag pattern documented in `DURABLE_REPORT.md` (there: a smooth
equity line through the 2013-2019 trend drought meant overfit/lookahead/bug, not a real edge) and
triggers investigation BEFORE any positive number from this sleeve is believed, exactly per that
report's diagnostic discipline (re-check the stress-signal construction for lookahead, re-check the
window-return computation for the same splice-seam/gap_before hygiene as the core signal, and
re-run the lookahead canary specifically on the conditioning layer if the unconditional signal's own
canary already checked out).

## Costs & validation (identical gate family to prior programs, reused per METHODOLOGY)

CPCV(8 groups, k=2, embargo declared, ≥3 weeks for this weekly-cadence study — reusing
`cpcv_paths()`'s signature from `e1_pipeline.py`, parameterized by the derived `periods_per_yr`, not
a hardcoded one); CSCV-style PBO over the full 104-cell family (`pbo_cscv`); drop-top-5-weeks
robustness; per-year Sharpes (`per_year_sharpe`); full metrics table per cell (`metrics()` reused
verbatim from `e1_pipeline.py`'s parameterized signature — net Sharpe annualized per-observation, EV
in bp per name-week, win %, maxDD, CAGR raw and at the 10% vol-target overlay, **`turn_mean`
actually wired via the `turn=` parameter this time** — E1_REPORT.md flagged that `metrics()`'s
`turn=` parameter was accepted but never passed at any call site, leaving `turn_mean` null
everywhere; E2's pipeline passes the per-rebalance gross-turnover array to every `metrics()` call so
`turn_mean` is populated and can be sanity-checked against the declared ~2x/week cost assumption
from the artifact itself, not reconstructed after the fact).

DSR via `dsr()` (reused from `e1_pipeline.py`) at **N_trials = 104** (the full ledger above, never a
hand-picked subset).

**Annualization arithmetic shown explicitly for the pre-named deployable** (per METHODOLOGY 1c):
`E2_REPORT.md` states `n_obs` (count of weekly net-return observations), `span_in_years`
(`(dates[-1]-dates[0]).days/365.25`), and `periods_per_year = n_obs/span_in_years` for the
deployable cell, and confirms this value is used (not an assumed `52.0` literal) in every
`metrics()`/`dsr()`/`cpcv_paths()`/`per_year_sharpe()` call for that cell.

**Survivorship (mandatory, landmine 4)**: every report section presenting a headline number states
plainly that the 102-name roster is today's Binance winners, not a point-in-time index-constituent
universe (per DATA.md's own caveat #1) — this is LESS fatal for E2's short-horizon cross-sectional
reversal than it would be for a momentum study (the within-cross-section RANKING each week is what
matters, not which names happen to be in the roster today), but it is not nil: a name's mere
PRESENCE in the 2026 roster (because it is popular today) could still correlate with its historical
volatility/liquidity profile in a way that shapes which names appear in the reversal book's tails
most often. Stated explicitly in every report section with a headline number, not once in an
appendix.

**Demo-gate reporting (mandatory, landmine 5)**: every headline number in `E2_REPORT.md` is
reported twice — full 102-name universe and restricted to the on_demo=true subset (19/110 names:
AMZN, COIN, CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR, MVLL, NVDA, OPENAI, PAYP, PLTR, QNTX, QQQ,
SPCX, SPY, TSLA — note OPENAI and QNTX are on_demo=true but perp-only/no-real-history and are
EXCLUDED from E2 entirely per the base-universe declaration above, so the effective on_demo-and-
real-history subset for E2 is 17 names, stated explicitly in the report rather than silently using
19 and producing a mismatched denominator).

## Adoption gate (frozen, pre-outcome)

Recommend PAPER-FIRST deployment of the pre-named deployable (continuous-scale-conditioned
band-ensemble reversal, full 102-name universe, 13bp primary cost) ONLY IF ALL of the following
hold simultaneously:

1. Net Sharpe positive on the full 102-name universe AND on the on_demo-and-real-history (17-name)
   subset reported separately (both shown; the paper-deployability judgment uses the on_demo subset
   since that is what could actually be paper-traded, per E1's identical precedent).
2. DSR pass at N_trials = 104.
3. PBO ≤ 0.25 over the full 104-cell family.
4. Drop-top-5-weeks Sharpe still positive.
5. The regime-sanity signature holds in full: above-median-stress weeks contribute >50% of total
   P&L, AND each of the four named stress windows (2018 Q4, 2020-03, 2022, 2024-08) individually
   shows a weekly Sharpe during that window exceeding the deployable's own full-sample Sharpe.

If ANY gate fails, the negative is recorded plainly in `E2_REPORT.md` as the program's conclusion
for this construction, and deployment is recommended AGAINST — a correctly-identified dead end here
is a complete, valuable result, not a failure to find one, exactly per every prior program's
discipline. Any individual cell (a different lookback, a different conditioning mechanism, a
different breadth) that independently clears all five gates on its OWN construction, even if the
pre-named deployable does not, is reported as a secondary finding with the same full gate discipline
applied to it before it is called a finding — but it does NOT retroactively become "the deployable"
unless this document is revised and re-frozen first.

**Honest target (stated before any run, per the task and METHODOLOGY)**: 0.3-0.6 NET Sharpe for the
conditioned/deployable construction is the honest target; the unconditional variant is expected
thinner still (reported anyway, for comparison, per the frozen grid). Above ~1.0 net is more likely
a bug (lookahead, cost omission, survivorship leak, or one of the two E1 bug patterns above) than a
real, durable, cost-survived cross-sectional reversal edge at this breadth and turnover — if that is
seen, the bug hunt happens BEFORE reporting it as a finding, and what was checked is documented
either way, exactly as E1's report did for its own >1.5-gross-Sharpe near-miss.

## Full test suite

Before any commit, `cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/
convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/
python -m pytest tests -q` must pass with no fewer than the 654-passed/1-skipped baseline recorded
at the E1 commit — the pass count is reported in `E2_REPORT.md`.
