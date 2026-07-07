# I1 — Timed-Beta Overlay (SPY/QQQ) — Final Report

**Status: CLOSED. Clean negative. DO NOT DEPLOY.**

Program: I1 (Program 8 in the research sequence). Frozen prereg:
`research/indices/I1_PREREG.md` (frozen 2026-07-07, commit `202ad54`). Pipeline:
`research/indices/i1_pipeline.py`, commit `ca34133`. Raw output:
`research/indices/results_i1.json`. Worktree: `/home/speculation/convexity-indices`, branch
`indices`. This program never touched `main`, the live paper system at
`/home/speculation/convexity` on branch `main`, any systemd unit, or the crypto/commodity paper
books' journals.

---

## 1. Program class (read this first)

**This is NOT another edge-certification program like E1/E2/E3.** E1/E2/E3 (closed, all clean
negatives, `research/equity/` on `main`) were cross-sectional edge-certification programs with
breadth ~93-110 names, where DSR at a full N_trials ledger was the hard adoption gate.

I1 is an **overlay-validation program on n=2 instruments** (SPYUSDT, QQQUSDT). With breadth 2 and
~8 years of daily data, no timed-beta construction can honestly clear a DSR significance gate —
this is a mathematical reality of the sample size (DSR's denominator scales with the variance of
Sharpes across an assumed-large trial population; a 2-instrument overlay family does not resemble
that population), not a flaw to hide or a shortfall to apologize for.

The intellectual basis for a deployment recommendation here was three-legged, declared before any
run:

- **(a)** Decades of published out-of-sample evidence for each component from outside this
  dataset: 200-day/10-month trend filtering (Faber 2007 and the SMA-crossover literature),
  volatility targeting (Moreira-Muir 2017, risk-parity literature), turn-of-month seasonality
  (Ariel 1987), pre-FOMC drift (Lucca-Moench 2015).
- **(b)** Our own data reproducing the documented behavior of each component — confirming known
  effects, not discovering new ones.
- **(c)** The combined overlay dominating a buy-and-hold benchmark on risk-adjusted terms in our
  sample, robustly across parameter perturbations.

**The adoption gate is therefore BENCHMARK-RELATIVE, not DSR-based.** DSR and PBO are computed and
reported below at the full frozen N_trials=72 ledger as **informational numbers only** — they do
not gate anything in this program, and their low values (expected at breadth=2) are reported
plainly, not hidden or apologized for. Neither DSR nor PBO is the reason this program's result is
negative; see §5 for why.

This is **timed beta** — directional market exposure with beta-sized drawdowns — **explicitly NOT
market-neutral alpha** like the crypto cross-sectional momentum book (the one currently-deployable
edge from the convexity research line). This report never presents I1's overlay as market-neutral
or risk-comparable to that book. **Any eventual deployment recommendation from this program is for
a separate, separately-capitalized THIRD paper book**, distinct from the existing crypto and
commodity books — this overlay would NOT be blended into either existing book, even had it cleared
its own gate. Since the gate failed, no deployment of any kind is recommended.

---

## 2. Data and construction recap

- **Universe**: SPYUSDT, QQQUSDT, `source=='duka'` rows only (splice-seam guard applied before any
  resampling/return computation, per carried-forward bug pattern 1). SPY: 778,291 duka rows,
  2018-05-01 → 2026-04-06 splice. QQQ: 777,249 duka rows, 2018-05-01 → 2026-04-02 splice. Both
  confirmed `on_demo=true` in `src/convexity/panel/candidates.md` (fresh `load_universe()` call) —
  demo gate satisfied, a paper deployment would have been mechanically possible had the gate
  passed.
- **S1 (trend)**: equal-weight ensemble of (a) 200-day SMA ±2% hysteresis band, (b) 12-month
  (252-trading-day) total-return sign evaluated monthly, (c) 10-month (210-trading-day) SMA rule
  (Faber convention) evaluated monthly. Output in {0, 1/3, 2/3, 1}. Long-or-flat only, no
  shorting. **1 tradeable S1 cell** (the ensemble; individual rules are diagnostic inputs, not
  separately-credited trials, per the prereg's explicit multiplicity convention).
- **S2 (vol targeting)**: `scale[t] = clip(target_vol / realized_vol[t-W:t-1], 0, cap)`,
  `target_vol=10%` annualized. Grid: `W_vol ∈ {20, 60}` × `cap ∈ {1.0, 1.5}` = **4 cells**.
- **S3 (calendar tilts)**: turn-of-month (+25% relative exposure, last-3/first-2 trading days of
  each month boundary) and pre-FOMC (+25% relative exposure, full trading day immediately
  preceding each of 67 regularly-scheduled FOMC announcement dates). With-tilts vs. without-tilts
  = **2 variants**, reported as separable attribution (both rows always shown together).
- **Costs**: 13bp round-trip primary ({6.5, 26}bp scenarios) charged on `|Δexposure|`; funding at
  {0, +5, -5}bp/week scenarios, prorated to `bp/week / 7` calendar days and scaled by the day's
  held exposure. {6.5,13,26}bp × {0,+5,-5}bp/wk = 3×3=9 scenarios per base cell.
- **N_trials = 8 base construction cells (S2×S3 = 4×2) × 9 cost/funding scenarios = 72.**
  Per-instrument decomposition (3 variants) and lookahead canaries (3) are diagnostic/reporting
  breakdowns, explicitly excluded from the ledger per the prereg.
- **Pre-named deployable** (decided before any run, never adjusted after seeing P&L): 50/50
  SPY/QQQ blend, S1 ensemble × S2(`W_vol=60, cap=1.0`) × S3(both tilts on), 13bp primary cost,
  funding=0, daily marks, decide-at-close/act-next-day throughout.
- **FOMC data acquisition** (fixed input, not revisited here): 67 regularly-scheduled FOMC
  meetings, 2018-01 through 2026-07-04, sourced from federalreserve.gov's own calendar/historical
  pages, cached to `data/fomc/fomc_dates.csv` (untracked). 7/7 hand spot-checks PASS, including
  correct exclusion of the 2020-03-15 emergency inter-meeting cut, the 2019-10-04 unscheduled
  conference call, and the cancelled 2020-03-17/18 regularly-scheduled meeting. Coverage summary:
  `research/indices/i1_fomc_data_coverage.json`. Independently re-confirmed in the adversarial
  skeptic pass (§6) against public knowledge of the FOMC calendar (2018-01-31, 2022-06-15,
  2023-07-26, 2024-12-18 all correctly present; 2020-03-15 correctly absent). **S3(b) (pre-FOMC
  tilt) is confirmed IN; the drop-S3(b) contingency was never triggered.**

---

## 3. Headline metrics table

All net numbers at the 13bp primary cost scenario, funding=0, `periods_per_yr` derived (not
hardcoded) at ≈251.5 trading days/year for both SPY and QQQ individually and for the blend
(independently re-derived in §6: SPY 1993 obs / 7.9316 yr = 251.27; QQQ 1992 / 7.9206 = 251.50;
blend 1991 obs post-`diff()` / same span ≈ 251.5 — all within the prereg's declared 5% tolerance
of 252, shown with full arithmetic here rather than assumed).

| | n_obs | Net Sharpe (ann) | Gross Sharpe (ann) | EV bp/obs (net) | Win % | MaxDD % | CAGR % (raw) | CAGR % (@10% vol) | Drop-top-5 Sharpe | Turnover (RT-equiv/yr) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Deployable** (Wvol=60,cap=1.0,tilts ON) | 1991 | **0.6882** | 0.7586 | 2.473 | 45.96 | 14.067 | 5.981 | 6.901 | 0.5485 | 2.455 |
| Deployable, no tilts | 1991 | 0.7244 | 0.7568 | — | — | 13.479 | 5.991 | — | 0.5935 | — |
| **Benchmark** (50/50 SPY/QQQ B&H) | 1991 | **0.7691** | 0.7691 | 6.403 | 56.35 | 31.580 | 14.917 | 8.209 | 0.5497 | n/a (static) |

S3's tilts made the deployable's Sharpe **slightly worse** in this sample (0.6882 with tilts vs.
0.7244 without) — a small, honestly-fragile effect, exactly as anticipated in the prereg (~96
turn-of-month events + 67 FOMC events over 8 years is not enough to expect a robust standalone
edge from a +25% relative exposure nudge on those days alone).

### Per-instrument decomposition (diagnostic only, not a competing construction)

| | Net Sharpe | MaxDD % | CAGR % |
|---|---|---|---|
| SPY-only (deployable cell) | 0.5856 | 14.601 | 4.989 |
| QQQ-only (deployable cell) | 0.7523 | 13.763 | 6.935 |

The blend's performance is genuinely shared across both names, not driven by one — neither
instrument diverges wildly from the blend's 0.6882.

### Full 8-cell robustness family (net Sharpe vs. benchmark 0.7691)

| Cell | Net Sharpe | Beats benchmark? |
|---|---|---|
| W_vol=20, cap=1.0, with tilts | 0.7991 | **YES** |
| W_vol=20, cap=1.0, no tilts | 0.8161 | **YES** |
| W_vol=20, cap=1.5, with tilts | 0.6971 | no |
| W_vol=20, cap=1.5, no tilts | 0.7289 | no |
| **W_vol=60, cap=1.0, with tilts (= DEPLOYABLE)** | **0.6882** | **no** |
| W_vol=60, cap=1.0, no tilts | 0.7244 | no |
| W_vol=60, cap=1.5, with tilts | 0.6573 | no |
| W_vol=60, cap=1.5, no tilts | 0.7131 | no |

**Only 2 of 8 cells beat the benchmark, and both are at W_vol=20 — the pre-named deployable
(W_vol=60/cap=1.0) is not among them.** This is reported exactly as it fell out; the deployable was
named before any run and is not swapped for a better-looking cell after the fact.

### CPCV(8 groups, k=2, embargo=21 trading days) — deployable

| n_paths | mean Sharpe | median Sharpe | min Sharpe | max Sharpe | % paths positive |
|---|---|---|---|---|---|
| 28 | 0.7212 | 0.6745 | -0.0431 | 1.8512 | 96.43% |

### DSR / PBO — informational only, full N_trials=72 ledger

| sr_obs | psr_vs_0 | **DSR** | sr0_obs (expected max SR under H0) | N_trials | T |
|---|---|---|---|---|---|
| 0.0434 | 0.9714 | **0.3199** | 0.05407 | 72 | 1991 |

**CSCV-style PBO over the 8-cell family: 0.0143.**

Per PROGRAM CLASS: DSR=0.32 is low, exactly as expected at breadth=2 — this is a mathematical
property of running DSR on a 2-instrument family, not a red flag about the strategy's quality, and
it does not gate anything. PBO=0.014 (low probability of backtest overfitting) is informationally
reassuring but is likewise not used to validate anything. **Critically, neither number is the
reason this program failed** — the deployable failed on its own benchmark-relative Sharpe,
robustness, drop-top-5, and turnover terms (gates 1, 4, 5, 6 below), entirely independently of what
DSR/PBO say.

### Per-year Sharpe, deployable vs. benchmark

| Year | Deployable | Benchmark |
|---|---|---|
| 2018 | 0.0000 | -0.3421 |
| 2019 | 1.0455 | 2.1403 |
| 2020 | 0.6185 | 0.9878 |
| 2021 | 1.3298 | 1.6677 |
| 2022 | **-1.7366** | -0.9858 |
| 2023 | 1.6137 | 2.2633 |
| 2024 | 1.1616 | 1.5126 |
| 2025 | 0.6101 | 0.9144 |
| 2026 (partial) | -1.5319 | -1.0371 |

The overlay lags the benchmark in nearly every single year, including — notably — 2022, the year
its drawdown-protection thesis is supposed to shine on a *risk-adjusted* basis (it does protect on
*drawdown magnitude*, see §4(iv), but not on Sharpe that year, which is dragged down by whipsaw
cost). The anticipated 2023-2024 melt-up laggard behavior is confirmed exactly as the prereg
predicted (1.61 vs 2.26 in 2023; 1.16 vs 1.51 in 2024) — this was flagged as an acceptable,
anticipated admission price before any run, and the actual numbers are reported here without
softening.

---

## 4. S1 regime-sanity signature — all four clauses, real numbers

**(i) Drawdown reduction in 2020-03 (COVID) — reported honestly, including the late exit.**
SPY exposure first drops below 0.5 on **2020-03-02**; QQQ on **2020-03-04**. This is several
sessions into the crash, not a clean textbook save — a fast V-shaped crash outran the filter's
typical reaction time, and this is stated plainly rather than implying a save that didn't happen.
Despite the late exit, the realized peak-to-trough drawdown over the Feb 15–Apr 15 2020 window was
still materially shallower for the S1 book blend: **14.067% vs. the benchmark's 31.58%** — the late
exit cost some protection but the filter still worked directionally.

**(ii) Bull-market participation.** The ensemble shows genuine, substantial (not flat)
participation: 2020-H2/2021 mean exposure SPY=0.756 (19.5% of days at ≥0.95 full exposure),
QQQ=0.582 (7.1% full); 2023-2024 melt-up mean exposure SPY=0.757 (28.9% full), QQQ=0.563 (0%
full). QQQ's lower participation reflects its higher realized volatility interacting with S2's
vol-target scale-down — this is real, substantial engagement in both bull windows, not a broken
always-flat filter.

**(iii) Visible whipsaw cost in 2018-Q4 and mid-2022 — one genuine finding, one confirmed
data-coverage artifact (investigated, not swept under the rug).** 2018-Q4 shows **zero**
exposure-change days for both SPY and QQQ. Per the prereg's own instruction, a suspiciously clean
curve through a known chop period is a bug-hunt trigger, not a good sign — this was investigated
and the zero is a **confirmed data-coverage artifact**: SPY/QQQ duka history begins 2018-05-01, so
the 200-day SMA (warms up ~2019-02-14), the 12-month-return-sign rule (~2019-03), and the 10-month
SMA rule (~2019-06) are all still pinned at their `fillna(0.0)` default throughout all of 2018-Q4
— there is no live signal yet to whipsaw. This is recorded explicitly in `results_i1.json`
(`all_exposure_zero_in_window=true`, `warmup_caveat` field) and **excluded from gate 3's pass/fail
logic**, exactly as a false-comfort "no whipsaw" reading would otherwise have been. By contrast,
**mid-2022 is past warmup and DOES show genuine, costly whipsaw**: SPY 104 exposure-change days
costing ≈47.9bp implied cost; QQQ 83 days costing ≈16.5bp implied cost — a real, honest admission
price, confirming the mechanism functions correctly once the data permits it to be tested.

**(iv) 2022 full-year drawdown, S1 book vs. benchmark.** S1 book blend maxDD **8.406%** vs.
benchmark blend maxDD **30.203%** — materially shallower, the slow-grind bear-market protection
mechanism working exactly as designed, distinct from the fast-crash case in (i).

All four clauses were checked with real numbers, including two genuinely unflattering findings
(the somewhat-late 2020-03 exit, and the 2018-Q4 zero requiring investigation before being
excluded) reported plainly rather than smoothed over.

---

## 5. Adoption gate — six sub-checks, benchmark-relative, applied explicitly

Gate is evaluated at the 13bp primary cost scenario, funding=0, deployable vs. the identical-series
50/50 SPY/QQQ buy-and-hold benchmark. **All six must pass for a deploy recommendation.**

| # | Sub-check | Deployable value | Threshold | Result |
|---|---|---|---|---|
| 1 | Net Sharpe > benchmark net Sharpe + 0.10 | 0.6882 | 0.8691 (0.7691+0.10) | **FAIL** — deployable does not even exceed the raw benchmark Sharpe (0.6882 < 0.7691); underperforms outright, not just short of the margin |
| 2 | MaxDD < 0.65 × benchmark maxDD | 14.067% | 20.527% (0.65×31.58%) | **PASS** — genuine, substantial margin |
| 3 | Regime sanity holds (all 4 S1 clauses, 2018-Q4 warmup-artifact excluded per its own diagnostic honesty requirement) | — | — | **PASS** — 2022 drawdown materially shallower (8.41% vs 30.20%) AND genuine whipsaw visible in the warmed-up 2022 window (104/83 days); 2018-Q4 investigated and correctly excluded as a data-coverage artifact, not credited as a clean-avoidance finding |
| 4 | All 8 robustness-family cells individually beat benchmark net Sharpe | 2/8 | 8/8 | **FAIL** — only both W_vol=20 cells beat the benchmark; the deployable's own W_vol=60/cap=1.0 cell is NOT among the passing cells |
| 5 | Drop-top-5-periods Sharpe beats benchmark's own undropped Sharpe | 0.5485 | 0.7691 | **FAIL** |
| 6 | Turnover sanity, ~5–15 round-trip-equivalent exposure changes/year | 2.455/yr | 5–15/yr | **FAIL** — too LOW, not too high; investigated (§below) and explained as a genuine low-frequency-construction property, not a hysteresis/smoothing malfunction |

**2 of 6 sub-checks PASS (2, 3); 4 FAIL (1, 4, 5, 6).**

**ADOPTION RECOMMENDATION: DO NOT DEPLOY.** Per the frozen gate, all six must pass; they do not.
This is recorded as this program's complete, valuable conclusion — identical discipline to
E1/E2/E3, where a correctly-identified negative was treated as a full result, not a failure to
find one.

**Turnover gate 6, investigated in the adversarial cost-realism pass (§6):** independently rebuilt
the ±2% SMA-band hysteresis logic from scratch and compared to a bare (unbanded) 200-day SMA
crossover. The band is a real, functioning whipsaw suppressor (SPY 42→16 flips/yr-equivalent
frequency, a 62% reduction; QQQ 26→12, a 54% reduction), not an inert no-op. Even under the most
generous assumption, S1-ensemble-alone implied round-trip-equivalent turnover caps out around
1.7–2.0/yr per instrument — already below the gate's 5/yr floor before S2/S3 are even added. The
reported 2.455/yr blended figure is a genuine structural property of a correctly-implemented
low-frequency trend filter on two liquid, mostly-trending index ETFs over 8 years — a design-fit
issue against the pre-declared 5–15/yr expectation band, not a construction defect.

---

## 6. Independent verification and adversarial-skeptic findings

An independent, from-scratch re-implementation (no imports from `i1_pipeline.py`) plus a
three-lens adversarial skeptic pass were both run against the pipeline. Reported here per bug
pattern, explicitly stating whether each was repeated or genuinely avoided.

### (1) Source-aware returns (splice-seam guard) — **genuinely avoided, verified**

Independently confirmed a REAL duka→perp splice seam exists for both SPYUSDT (2026-04-06,
notably stamped `gap_before=False` despite a real -0.108% price discontinuity — this seam would
**not** have been caught by a `gap_before`-based filter, only by the `source=='duka'` filter the
pipeline actually uses) and QQQUSDT (2026-04-02→04-06, `gap_before=True`). Independently rebuilt
the duka-only daily close series for both symbols and confirmed zero perp rows ever enter it, and
that the series terminates at/before the last duka day for both. **Landmine 1 is genuinely fixed**,
and the SPY `gap_before=False`-at-a-real-seam finding retroactively validates why source-column
filtering (not `gap_before` filtering) was the correct design choice from the start.

### (2) Derived annualization — **genuinely avoided, verified**

Independently measured `n_obs`/`span_years` arithmetic: SPY 1993 obs / 7.9316 yr = 251.27; QQQ
1992 / 7.9206 = 251.50; blend 1992 (or 1991 post-`diff()`) / 7.9206 ≈ 251.50 — closely matching
the pipeline's own derivation and its ~252 expectation, within the prereg's 5% tolerance. No
hardcoded `252` used in any formula. **Landmine 2 is genuinely fixed.**

### (3) Exposure-timing alignment (the program's central risk) — **genuinely avoided, verified with real discriminating power**

The synthetic-outlier-injection test was independently rebuilt (not imported): a +50% fake price
spike was injected at index 996 (2022-04-13, the same index the pipeline used) into an
independent, from-scratch implementation of the 200-day SMA±band and the vol-target scale. The
**honest** (shift(1)-lagged) version showed zero divergence at or before the injection point, with
first divergence appearing exactly one index later (t_inject+1), for both S1 and S2. As a negative
control, a **naive** (unlagged) version of the identical logic was also run and **did** react
same-day — proving the test has real discriminating power, not a tautology that would pass
anything. This exactly reproduces the pipeline's own reported result (PASS=true for both SPYUSDT
and QQQUSDT at t_inject=996/2022-04-13).

A separate adversarial "lookahead" lens additionally hand-traced/numerically simulated five
specific boundary conditions directly against the pipeline's source
(`research/indices/i1_pipeline.py`, 1278 lines): the SMA-band warmup edge, the monthly-rule
first/last/partial-month edges, the turn-of-month calendar-only construction, the pre-FOMC
`searchsorted` boundary (cross-checked against the real 67-row `fomc_dates.csv` — all announcement
dates fall on Wednesday or Thursday, so the theoretical weekend-adjacency edge case never actually
triggers in this dataset), and the S2 vol-window slicing semantics. All five held up under direct
numerical simulation against the actual code. **One soft, non-bug observation**: the turn-of-month
window's calendar-position determination technically depends on knowing a month's full trading-day
count, which is public calendar/holiday-schedule information known far in advance, not
price/return information — this is a modeling nuance about calendar-vs-price information, not a
data leak, and does not change any reported number or the DO-NOT-DEPLOY conclusion.

**Landmine 3 is genuinely fixed** — the central risk of this entire program was checked from three
independent angles (the pipeline's own canary, an independent from-scratch re-implementation with
a negative control, and hand/numerical boundary tracing of the actual source) and survived all
three.

### (4) Costs scale with realized exposure changes — **one real bug found, does NOT affect the primary gate table**

**Trading-fee scaling: genuinely avoided, verified.** A toy-example check confirmed fee at
`|Δexposure|=0.03` (small vol-target nudge) = 0.39bp; fee at `|Δexposure|=1.00` (full trend flip)
= 13.00bp; ratio 33.33 exactly equals the delta-exposure ratio — confirming linear proportionality,
not a flat charge. Real-data spot checks on SPY (a big-delta day, 2026-03-27, |Δ|=0.8212 →
fee=10.675bp; a small-delta day, 2023-07-10, |Δ|=0.0014 → fee=0.0185bp) both scale exactly with
their driver.

**Funding scaling-with-exposure: genuinely avoided, verified.** Funding at exposure=0.20 →
0.1429bp; at exposure=0.90 → 0.6429bp; ratio 4.5 exactly equals the exposure ratio. Funding at
exposure=0.0 is exactly 0.0 — no flat E2-style charge regardless of exposure.

**However, a real bug was found in funding's absolute calendar-cadence magnitude** (this is a
distinct issue from the scaling-with-exposure property above, which is correct): the prereg
declares `funding_bp_per_day = funding_bp_per_wk / 7` specifically as **calendar-day** accrual
("weekends and holidays are not free"), but `apply_costs()` applies that flat per-day rate once
per **trading row**, not scaled by the actual elapsed calendar days since the prior row. Real SPY
calendar-row gap structure: 1554 one-day gaps, 361 three-day gaps (weekends), 24 two-day and 53
four-day gaps (holidays), mean 1.4543 calendar-days/row vs. the 1 row/row the code effectively
assumes. Over the full ~7.93yr SPY sample at exposure=1 constant, the pipeline charges 1422.86bp
total funding at the 5bp/week scenario; a calendar-accrual-correct version would charge 2069.29bp
— an understatement by a factor of 0.6876 (~31%). This reconciles almost exactly with
`251.5/52.18 ≈ 0.688` (trading-day count divided by calendar-week count).

**Mitigating factor, explicitly confirmed**: there is exactly one `apply_costs()` function with
exactly two call sites (deployable and benchmark), both funneling through the identical buggy
formula — so **the primary 13bp/funding=0 gate table that actually gates adoption is completely
unaffected**, since it uses funding=0 throughout. The bug only understates the magnitude of the
informational ±5bp/week funding-scenario deltas reported elsewhere in the 72-cell grid (reported
CAGR delta ±0.95pp at 13bp+±5bp/wk; a calendar-correct version would be roughly ±1.38pp). **This is
flagged honestly as a real, distinct bug from E2's flat-funding bug and E3's cadence-conversion
bug in form, but the same bug in substance** (a calendar-cadence rate applied at trading-cadence
frequency) — it should be fixed before this pipeline is reused for any future program where a
nonzero-funding scenario needs to be trusted at its correct magnitude. It does not change this
program's conclusion.

**Landmine 4 verdict: fee-scaling genuinely fixed; funding scales correctly with exposure but its
absolute calendar-cadence magnitude is understated by ~31% in the informational, non-gating
funding scenarios.** Recorded plainly rather than omitted, per this report's own standard.

### (5) Leveraged/inverse-ETF exclusion and survivorship — as declared

Confirmed moot: SPY and QQQ are plain, unleveraged index ETFs, stated explicitly as a deliberate
non-applicable check rather than silently omitted. Survivorship is structurally different from
E1/E2/E3 (no delisted-loser invisibility risk with 2 major index ETFs), but the 2018-2026 sample
is dominated by one exceptional secular tech bull regime for QQQ specifically — the per-year table
(§3) and per-regime signature (§4) exist precisely to surface this, and do (QQQ's standalone
Sharpe of 0.75 vs SPY's 0.59, and the 2023-2024 melt-up laggard years, are both visible rather than
hidden inside one full-sample average).

### Benchmark-fairness lens — the most important lens for this program class — **survives, no defect found**

This lens received the most scrutiny given its centrality to a benchmark-relative gate. Checked
directly against pipeline source and by independently re-executing its functions:

1. **Identical date range**: independently rebuilt both blends; symmetric difference between their
   calendars is exactly 0 (both span 2018-05-01 to 2026-04-02, n=1992). No shorter/longer window
   bias either direction.
2. **Identical weighting convention**: both blends compute `net = 0.5*spy_net + 0.5*qqq_net` at
   daily granularity (daily-rebalanced-to-50/50 in return space) — same code shape both sides. If
   anything this is mildly conservative *for* the deployable's relative comparison, since a
   drift-weighted "true" buy-and-hold would tilt toward QQQ's higher standalone Sharpe over 8 years
   and likely score even higher than the reported 0.7691.
3. **Benchmark cost machinery is genuinely live, not dead code**: the benchmark's net Sharpe is
   byte-identical across all three trading-cost scenarios (6.5/13/26bp all give 0.7691) — this
   looks suspicious at first glance, but is the mathematically correct signature of true
   buy-and-hold (a single one-time entry-day cost, amortized over ~1992 daily observations across
   8 years, is genuinely negligible on an annualized Sharpe). Confirmed the cost path is genuinely
   wired by checking funding sensitivity instead: the benchmark's Sharpe moves substantially across
   funding scenarios (0.6833 at +5bp/wk vs. 0.8548 at -5bp/wk) — a real, live cost/funding
   machinery, not bypassed for the benchmark. By contrast the deployable shows real sensitivity
   even on the trading-fee axis (0.7234→0.6882→0.6177 across 6.5/13/26bp) — exactly the expected
   qualitative difference between a near-zero-turnover benchmark and a real trading strategy, not a
   fairness bug.
4. **Jackknife of top exposure-change days** (adversarial concentration check): sorted the
   deployable's `Δexposure` array descending and excised the top 1/3/5/10 biggest single-day
   events (largest: 2026-04-01 at Δ=0.308; also the 2020-03-02 COVID de-risk at Δ=0.260). Result:
   Sharpe **decreases** monotonically as these days are removed (0.6882→0.6855→0.6669→0.6408→0.6344
   for N=1/3/5/10), and gate 1 remains FAILED at every N. This is the opposite of the failure mode
   the lens is designed to catch — the negative result is not propped up by a few lucky days; the
   biggest flip days (including the COVID de-risk, which realized +2.07% that day) were
   net-helpful to the deployable, so removing them makes the finding *more* negative, not less.
5. Test-suite baseline independently cross-checked: 654 passed, 1 skipped — matches exactly (see
   §7).

**No fairness defect found.** The date range, weighting convention, and cost/funding machinery are
identical in form between deployable and benchmark; the one construction that looked odd at first
glance (benchmark cost-scenario invariance) has a correct economic explanation, independently
confirmed via the funding-scenario cross-check. **The DO-NOT-DEPLOY conclusion survives this
adversarial pass.**

### Overall independent-verification summary

Hand-derived deployable Sharpe = 0.6877 vs. pipeline's 0.6882 (diff 0.0005); hand-derived benchmark
Sharpe = 0.7689 vs. pipeline's 0.7691 (diff 0.0002) — both effectively exact matches from a
completely independent, from-scratch implementation. 2020-03 and 2022 drawdowns both reproduce
within ~0.01–0.02 percentage points, including the exact SPY/QQQ exposure-drop dates. **Every one
of the adversarial checks corroborates the pipeline's own reported results and the DO-NOT-DEPLOY
conclusion rests on a numerically sound pipeline** — the one real bug found (funding's
calendar-cadence magnitude, §4 above) is confined to informational, non-gating scenarios and does
not touch the primary 13bp/funding=0 gate table. The negative finding is a genuine property of this
overlay construction in this sample, not a pipeline artifact.

---

## 7. Test suite

`cd /home/speculation/convexity-indices && PYTHONPATH=/home/speculation/convexity-indices/src:/home/speculation/convexity-indices /home/speculation/convexity/.venv/bin/python -m pytest tests -q`

**654 passed, 1 skipped** — matches the confirmed-unchanged baseline through E1/E2/E3, both before
and after the pipeline's two mid-development bug fixes, and independently reconfirmed by the
adversarial skeptic pass.

---

## 8. Conclusion

**DO NOT DEPLOY.** The pre-named deployable (50/50 SPY/QQQ blend, S1 trend ensemble × S2 vol-target
at W_vol=60/cap=1.0 × S3 both tilts, 13bp primary cost) fails 4 of the 6 frozen benchmark-relative
adoption-gate sub-checks: it does not beat the 50/50 SPY/QQQ buy-and-hold benchmark's net Sharpe at
all (0.6882 vs. 0.7691, let alone by the required +0.10 margin); only 2 of the 8 declared
robustness-family cells beat the benchmark, and the deployable's own cell is not among them; its
drop-top-5 Sharpe falls below the benchmark's own undropped Sharpe; and its realized turnover
(2.455 round-trip-equivalent/yr) is below the declared 5–15/yr sanity band. It does pass on
drawdown reduction (14.07% vs. 31.58% maxDD, comfortably under the 0.65× threshold) and on regime
sanity (the S1 signature's four clauses all hold honestly, including two genuinely unflattering
findings — a somewhat-late 2020-03 exit and a 2018-Q4 zero-whipsaw reading correctly diagnosed as a
data-coverage artifact rather than credited as a clean pass).

This negative was checked from every angle the methodology requires: source-aware returns,
derived annualization, exposure-timing (the program's central risk, verified three independent
ways), cost/funding scaling (one real but non-gating bug found and reported honestly), and —
specifically for this program class — benchmark-construction fairness, which survived adversarial
jackknifing. DSR (0.32) and PBO (0.014) are reported informationally at the full N_trials=72
ledger, exactly as low as expected at breadth=2, and are explicitly not the reason for this
result.

This is timed beta, not market-neutral alpha, and any deployment recommendation from this program
would have been for a separate, separately-capitalized third paper book, distinct from the
existing crypto and commodity books. Since the gate fails, **no new paper book is recommended from
this program.** The crypto cross-sectional momentum book remains the sole currently-deployable edge
from the convexity research line.

Per this program's own stated discipline (identical to E1/E2/E3): a correctly-identified,
adversarially-verified negative is a complete, valuable result, not a failure to find one.
