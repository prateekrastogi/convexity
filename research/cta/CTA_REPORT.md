# C1 EQUITY-CTA — FINAL REPORT (Program 9; Stage A + Stage B; CLOSED)

**Status: CLOSED. Comprehensive negative on the numeric adoption gate. DO NOT DEPLOY. PAPER-FIRST NOT recommended.**

This document is the complete, honest research record for Program C1 (the "equity-CTA" program),
run in the `equity-cta` worktree (`/home/speculation/convexity-cta`, branch `equity-cta`). It
supersedes nothing on `main` and is not merged there. Companion documents: `CTA_PREREG.md` (frozen
pre-registration, written before `ctaB_pipeline.py`), `CTA_MATH_NOTES.md` (full per-axis
derivations — referenced by section number below, not reproduced), `STATE.md` (running program
state, finalized alongside this report), `results_ctaA_full_grid.csv.gz` (Stage A's 29,760-row
exploratory grid), `results_ctaB.json` (Stage B's complete numeric record).

---

## 1. Mandate and established facts this program was built on

C1 is an owner-directed follow-up to the closed E1/E2/E3 equity swing-research programs, the closed
I1 timed-beta overlay program, and four exploratory CTA-style scans — all on `main`,
`research/equity/` and `research/indices/`. Its mandate: build a maximally-general, six-axis,
literature-grounded CTA-style construction (causal signal extraction, optimal-control entry/exit,
regime modeling, volatility forecasting, growth-optimal leverage, portfolio construction) on the
full 93-name equity/ETF universe, and test it honestly against a named, already-computed passive
benchmark, using **CAGR-at-a-declared-maxDD-budget** as the only valid objective.

Seven established facts were inherited, cited, and re-verified fresh where this program depended on
them directly (full text in the task's own HOST/SAFETY section, not reproduced verbatim here):

1. **Sharpe is leverage-invariant** in this cost structure (both fee and funding terms are exactly
   linear in exposure) — therefore Sharpe cannot be the optimization objective for any leverage
   question; the valid objective family is CAGR-at-a-maxDD-budget.
2. **Shorting equities is a proven net negative** on this universe (32/32 negative short-side cells
   in prior programs) — long/flat only by default.
3. **Ruin is a single-day, common-factor gap-risk problem**, not diversifiable across 93 correlated
   names (all 93 names crash together; worst days cluster on 2020-03-16 across SPY/QQQ/the
   93-name benchmark alike).
4. **Trailing realized volatility is reactive, not predictive** — it protected well in the gradual
   COVID vol ramp but was blind ahead of the sudden 2025-04-09 tariff-relief reversal.
5. **Single-timeframe signals are fragile** — multi-horizon ensembling is required (this is exactly
   what Axis A's IC-weighted combination implements).
6. **The benchmark to beat**: 93-name, point-in-time, equal-weight, unlevered buy-and-hold over the
   full sample — **Sharpe 1.353, CAGR 46.8%, maxDD 41.0%, worst single day −12.1% (2020-03-16)**.
   This is restated in full in §7 below and is the frozen comparator throughout.
7. **Five carried-forward bug patterns** (source-aware returns, derived annualization,
   exposure-timing alignment/decide-at-close-act-next-day, calendar-day-prorated costs/funding
   scaled by realized exposure, leveraged/inverse/decay-ETF exclusion) — binding on
   `ctaB_pipeline.py` exactly as on the Stage-A axis modules, re-verified fresh in Stage B (§4, §6).

**Program class**: per `CTA_PREREG.md`'s own explicit declaration (adopting I1's precedent), C1 is
a **benchmark-relative overlay-validation program**, not a cross-sectional DSR-certification program
like E1/E2/E3, despite E1/E2/E3-scale breadth (93 names, up to 29,760 grid cells). The adoption gate
is therefore entirely CAGR/Sharpe/maxDD-vs-benchmark; DSR/PBO are computed and reported honestly
(§6, §7) but never gate the decision.

---

## 2. Stage A mathematics — what each axis contributed

Full derivations live in `CTA_MATH_NOTES.md`; only headline content and real numbers are repeated
here.

| Axis | What it is | Section | Grid points (Stage-A actual) |
|---|---|---|---|
| A — Signal Extraction | Causal multi-horizon trend filter: EWMA continuum (exact crossover⇔momentum identity), local-linear-trend Kalman filter (exact equivalence to Holt's double-exponential smoothing), causal rolling L1 trend filtering (Kim-Koh-Boyd, ADMM warm-started for ~2x speedup), Grinold-Kahn IC-weighted multi-horizon combination | §A (lines 26-383) | 31 |
| B — Entry/Exit Control | Garleanu-Pedersen (2013) partial-adjustment "aim portfolio" (closed-form continuous-time HJB solve) and Davis-Norman/Shreve-Soner no-trade-band optimal control (smooth-pasting derivation) | §B (384-798) | 4 |
| C — Regime Modeling | Strictly-causal 2-state Gaussian HMM regime gate (Hamilton 1989 root; EM/Baum-Welch; filtering not smoothing — §C.4 is the causality-critical section) | §C (799-1110) | 3 (trimmed from 5; both 3-state variants dropped per Axis C's own short-history caveat) |
| D — Vol Forecasting | Yang-Zhang OHLC range estimator, EWMA/RiskMetrics, GARCH(1,1) fit causally, HAR-RV — evaluated by causal QLIKE/MSE | §D (1111-1559) | 3 (trimmed from 10 via Axis D's own top-3 ranking, plus one disclosed SPY/QQQ-specific GARCH retention) |
| E — Leverage Policy | Discrete-time Kelly fractions, Grossman-Zhou (1993) drawdown control, EVT/POT ruin-constrained ceiling, CDaR-constrained leverage (Chekhlov-Uryasev-Zabarankin 2005, closed-form scalar case) | §E (1560-1893) | 6 |
| F — Portfolio Construction | Equal-risk-weight vs. correlation-adjusted risk parity, correlation-window × spike-de-risking overlay | §F (1894-2084) | 8 |

Every module passed its own synthetic-outlier-injection causality test at Stage A (documented per
axis in `CTA_MATH_NOTES.md`, re-verified fresh at Stage B scale — §4.7, §6).

**Grid design** (§9 and §9-REDESIGN of `CTA_MATH_NOTES.md`): the original frozen full-factorial
design ({A,B,F} full cross × OAT-replicated {C,D,E}, block width 19) gave N=18,848/universe and was
**never run** — superseded before any cell was computed, for compute-tractability on this VM's 2
vCPUs. The redesign (ADMM warm-start, a 22-name `STAGE_A_22` pilot universe force-including
SPY/QQQ and excluding the confirmed-buggy AAPLUSDT, Axis C/D grid trims disclosed above, OAT block
width cut 19→10) gave the design that **actually ran**: N=9,920 cells/universe × 3 universes
(`STAGE_A_22`, SPY-alone, QQQ-alone) = **29,760 total rows**, launched 2026-07-07 13:02:21 UTC,
completed after 186.4 minutes wall-clock (disclosed as over the sub-60-minute aspiration — a
finding about `aggregate_portfolio`'s per-day Python-object-construction floor, not a bug). Row
count verified = 29,760 exactly, matching 9,920×3; committed as
`research/cta/results_ctaA_full_grid.csv.gz`.

---

## 3. Stage A selection funnel, winner, runner-up, and the melt-up finding

### 3.1 The funnel (frozen; zero-survivors disclosed plainly)

- **Stage 0 (leverage filter, avg_leverage ∈ [2,5], all 29,760 rows)**: **796 cells survive, all
  `e_key=half_kelly`, all in `STAGE_A_22`.** SPY-alone and QQQ-alone give 0 cells each. Every other
  leverage policy misses the band entirely (`gz_loose` 1.20-1.82, `quarter_kelly` 0.78-1.17,
  `gz_tight` 0.57-0.83, `cdar` 0.58-0.93 — all below 2.0; `evt_ceiling` 7.86-10.28 — already
  known-ruinous).
- **Stage 1 (intersect with each DD budget, i.e. the objective's own feasible set)**: **DD≤20%: 0.
  DD≤30%: 0. DD≤41%: 0.** Zero at every budget, every universe. Root cause: the only
  leverage-in-[2,5] cells (the 796 `half_kelly` cells) realize maxDD 57.9%–69.0%, blowing through
  even the 41% budget.
- **Robustness-filter stages (i) cost-scenario, (ii) drop-top-5, (iii) walk-forward**: inapplicable
  — start=0 → after (i)=0 → after (ii)=0 → after (iii)=0. **None of (i)-(iii) was actually exercised
  against a nonempty candidate pool in Stage A** — Stage B's own re-run (§4.3-4.5) is the first real
  execution of these checks in this program.
- Separately: **DD≤20% is infeasible for the entire `STAGE_A_22` grid** (0/9,920 cells at any
  leverage). DD≤30% is achievable by 1,984/9,920 cells and DD≤41% by 2,965/9,920, but only at
  leverage well below 2× (winning constructions run at avg_leverage ≈0.87-0.88×).

**`selection.zero_survivors = TRUE`.** Because leverage-in-[2,5] is infeasible jointly with any DD
budget, the robustness funnel could not start. Per the task's fallback rule, selection proceeded on
the single best cell by the raw objective — but the fallback itself required **dropping the
leverage-in-[2,5] restriction entirely**, not merely relaxing the robustness filter. This is a
flagged deviation, disclosed here plainly rather than hidden in a footnote.

**Selection rule actually used**: best CAGR-at-30%-DD-budget among all `STAGE_A_22` cells, no
leverage restriction.

### 3.2 Winner and runner-up (frozen; both flagged as selected without surviving a robustness
filter and without satisfying leverage-in-[2,5])

| Axis | Winner | Runner-up |
|---|---|---|
| A (signal) | `kalman_1e-06_1e-08_0.0001` — fastest-adapting Kalman triple (q_level,q_slope,r)=(1e-6,1e-8,1e-4) | same |
| B (entry/exit) | `aggressive` — ζ=0.25, η=0.5, narrowest band | same |
| C (regime) | `hmm2_rf63` — 2-state HMM, 63-day refit (Axis C's own baseline) | same |
| D (vol) | `ewma_0.94` — RiskMetrics λ=0.94 (Axis D's own baseline) | same |
| E (leverage) | `cdar` — CDaR-constrained, budget=0.25, α=0.95, closed-form L\* | same |
| F (portfolio) | `corr_risk_parity_w120_spike` — 120-day corr window, spike overlay ON | `corr_risk_parity_w60_nospike` — 60-day window, spike overlay OFF |

The two constructions differ in **exactly one axis** (F). The literal #2/#3 rows in the raw ranking
were exact numerical duplicates of the winner via the documented `w120_spike`/`w120_nospike`
byte-identical collapse (spike overlay never actually triggers on this book); the runner-up is the
first row that differs in a real parameter.

### 3.3 Full Stage-A metrics table (`STAGE_A_22` pilot, 13bp primary cost, funding=0)

| Metric | Winner | Runner-up |
|---|---:|---:|
| CAGR (full sample) | 18.491% | 18.096% |
| maxDD | 28.098% | 28.240% |
| Sharpe (net, ann.) | 1.0266 | 1.0201 |
| avg_leverage | 0.8821 | 0.8716 |
| win_rate | 56.0% | 56.0% |
| worst single day | −8.86% | −8.78% |
| EV/trade (bp) — see turnover caveat §6.3 | 0.815 | 0.798 |
| trades/yr (raw day-count metric — flawed, see §6.3) | 2,284.9 | 2,284.9 |
| n_obs / span_years | 2,053 / 8.167 | 2,053 / 8.167 |
| dd_budget_20 / 30 / 41 | FALSE / **TRUE** / TRUE | FALSE / **TRUE** / TRUE |
| CAGR at 6.5bp / 13bp / 26bp | 19.587% / 18.491% / 16.329% | 19.226% / 18.096% / 15.868% |
| CAGR delta vs. 46.8% benchmark | −28.31pp | −28.70pp |
| ret 2018-Q4 chop | 0.0% (no data) | 0.0% (no data) |
| ret 2020-03 COVID | −13.17% | −13.07% |
| ret 2022 full year | −18.99% | −19.24% |
| ret 2023-24 melt-up | +57.55% | +57.76% |
| ret 2025-04 squeeze week | −0.66% | −0.72% |

Cost-scenario sensitivity is monotonic and modest for both, no sign reversal. The 41%-budget
objective is actually **won by a different, more-levered cell** (`quarter_kelly`, avg_lev=1.17,
CAGR=24.36%, maxDD=36.1%) — the winner reported here is specifically the 30%-budget winner per the
task's tie-break, not a universal winner across all three budgets.

### 3.4 Melt-up concentration finding (mandatory first-order caveat)

Exact geometric decomposition (`total_full=(1+cagr)^span_years−1`;
`total_ex_meltup=(1+total_full)/(1+ret_2023_24_meltup)−1`;
`cagr_ex_meltup=(1+total_ex_meltup)^(1/(span_years−1.999))−1`, using the compounded 2023-01-01 to
2024-12-31 window return, 1.999 years, fully interior to the 8.17-year sample):

- **Winner**: `ret_2023_24_meltup=+57.55%`, `cagr_ex_meltup=16.29%` — a 2.20pp (~11.9% relative)
  haircut. Re-ranked by `cagr_ex_meltup` among the 1,984 DD≤30%-qualifying `STAGE_A_22` cells: the
  winner **falls from rank #1 to rank #4** (0-indexed 3). It is overtaken by an
  `ewma_5_20_80_shr1.0 × aggressive × cdar × corr_risk_parity_w120` construction (full-sample CAGR
  17.72%, lower than the winner's, but a smaller melt-up-window return of +48.20%, so it loses only
  ~1.28pp ex-meltup).
- **Runner-up**: `ret_2023_24_meltup=+57.76%`, `cagr_ex_meltup=15.76%` — falls out of the top spots
  ex-meltup in the same way.
- **Context, independently reconfirmed for this report** (direct query of the 1,984-row qualifying
  pool): its full-sample CAGR distribution has **mean 12.04%, std 2.60%**, and the entire top-20 by
  CAGR spans only **18.49% down to 17.30%** — a razor-thin, near-tied band. Whichever cell "wins"
  by full-sample vs. ex-meltup CAGR, or by CAGR vs. Sharpe, is close to a coin flip at the margin.

**Honest characterization**: the winner's (and runner-up's) edge is MATERIALLY, not just modestly,
concentrated in the 2023-24 melt-up — enough that excluding that single 2-year window flips the
identity of the best-performing construction in this DD-budget class. This is carried into Stage B
as a first-order caveat, not a footnote (and is reconfirmed at full-93-name scale in §4.6 below).

---

## 4. Stage B — fresh, independent execution of the frozen construction on the full 93-name universe

`ctaB_pipeline.py` re-runs the frozen winner and runner-up **exactly as specified** (no re-tuning)
on the reconciled full 93-name universe. `reconcile_universe()` independently re-verified = exactly
93 names (110 equities sleeve − 8 perp-only − 3 short-history − 6 leveraged/inverse/decay ETFs).
Panel: 93/93 names usable, panel shape (2,053 obs, 93 names), span_years=8.167,
periods_per_yr=251.377 (derived fresh, per bug pattern (b)). Wall-clock 348.3s.

**Fresh internal cross-check vs. Established Fact 6**: this run's own 93-name equal-weight
unlevered buy-and-hold = **CAGR 41.947% / maxDD 41.039%** — maxDD matches Fact 6's 41.0% almost
exactly; CAGR is somewhat below Fact 6's 46.8% (expected from point-in-time entry/exit
differences). **Fact 6 (CAGR 46.8% / maxDD 41.0%) remains the frozen comparator, not superseded.**

### 4.1 Full Stage-B metrics table, GROSS and NET, 13bp primary / funding=0

| Metric | Winner NET | Winner GROSS | Runner-up NET | Runner-up GROSS |
|---|---:|---:|---:|---:|
| avg_leverage (CDaR closed-form L) | 0.9718 | 0.9718 | 0.9866 | 0.9866 |
| CAGR | 19.357% | 22.098% | 21.260% | 25.698% |
| maxDD | 29.501% | 28.857% | 29.757% | 29.463% |
| Sharpe (ann.) | 1.0338 | 1.1541 | 1.0768 | 1.2587 |
| worst single day | −10.355% | −10.341% | −10.614% | −10.600% |
| win_rate | 55.33% | 55.61% | 55.39% | 56.00% |
| ruin_flag | False | False | False | False |
| CAGR delta vs. 46.8% benchmark | −27.44pp | −24.70pp | −25.54pp | −21.10pp |
| dd_budget_20 / 30 / 41 | F / T / T | F / T / T | F / T / T | F / T / T |
| n_obs / span_years | 2,053 / 8.167 | — | 2,053 / 8.167 | — |
| n_active_mean (names live) | 65.83 | — | 65.83 | — |

**Runner-up beats the winner on every headline number at full-93 scale** — a genuine, disclosed
reversal from Stage A's 22-name pilot (where the winner narrowly led). This is reported plainly, not
smoothed over: the "winner" label reflects the frozen selection rule applied at Stage-A-pilot scale,
per the prereg's own no-re-tuning discipline; it is not re-designated after the fact.

**Axis-F collapse re-check at full scale**: the diagnostic `corr_risk_parity_w120_nospike` variant
is numerically IDENTICAL to the winner (`max_abs_diff=0.0`) — confirms the Stage-A-documented
`w120_spike`/`w120_nospike` byte-identical collapse still holds at 93 names; the correlation-spike
de-risking overlay never actually triggers anywhere in this sample, at either universe size.

### 4.2 Cost-scenario robustness (gate 4-i)

| Scenario | Winner CAGR | Runner-up CAGR |
|---|---:|---:|
| 6.5bp | 20.720% | 23.460% |
| 13bp (primary) | 19.357% | 21.260% |
| 26bp | 16.676% | 16.976% |

Monotonic decreasing in both cases, no sign reversal. **PASS** for both.

### 4.3 Drop-top-5-contributing-days robustness (gate 4-ii)

| | Winner | Runner-up |
|---|---|---|
| Top-5 dates | 2025-04-09 (+8.55%), 2020-03-24 (+7.82%), 2020-03-13 (+7.45%), 2020-04-06 (+6.04%), 2026-06-11 (+5.53%) | 2025-04-09 (+8.68%), 2020-03-24 (+8.04%), 2020-03-13 (+7.60%), 2020-04-06 (+6.22%), 2020-03-17 (+5.56%) |
| CAGR after drop-5 | 14.466% | 16.198% |
| maxDD after drop-5 | 34.700% | 38.858% |
| dd_budget_30 survives? | **NO** | **NO** |
| dd_budget_41 survives? | **YES** (34.70%≤41%) | **YES** (38.86%≤41%) |

Criterion is defined on dd_budget_41 survival per the pipeline's own coded logic → **PASS** for
both, but note plainly: the tighter 30%-budget qualifying status is fragile — it does NOT survive
removing just 5 of 2,053 trading days.

### 4.4 Purged walk-forward / CPCV robustness (gate 4-iii)

CPCV(8 groups, k=2, embargo=252 trading days — Axis C's own `min_train=252`, the slowest
signal-component horizon in the frozen construction), reusing `cpcv_paths()` from
`research/indices/i1_pipeline.py`:

| | Winner | Runner-up |
|---|---:|---:|
| Folds computable / possible | 21 / 28 | 21 / 28 |
| Sharpe mean / median | 1.128 / 1.066 | 1.145 / 1.192 |
| Sharpe min / max | −0.057 / 2.721 | 0.027 / 2.620 |
| % folds positive Sharpe | 95.24% | 100% |

**PASS** for both (≥50% threshold used, per the pipeline's coded criterion).

Walk-forward thirds (chronological, no purging):

| Third | Winner CAGR / Sharpe / maxDD | Runner-up CAGR / Sharpe / maxDD |
|---|---|---|
| 2018-05-01 → 2021-01-15 | 18.39% / 0.927 / 29.14% | 18.36% / 0.916 / 29.76% |
| 2021-01-19 → 2023-10-05 | **4.56% / 0.339** / 29.50% | **4.45% / 0.326** / 29.56% |
| 2023-10-06 → 2026-07-01 | 37.30% / 1.840 / 20.64% | 44.13% / 1.961 / 20.43% |

The middle third is genuinely weak for both constructions — consistent with, and confirming at full
scale, the Stage-A melt-up-concentration caveat: most of the edge lives in the first and (especially)
third thirds, not spread evenly across the sample.

### 4.5 Drawdown-control exposure path through both named windows (gate 5)

**2020-03 COVID** (padded window 2020-01-15 to 2020-05-01; named window 2020-02-15 to 2020-04-15):
the combined regime-gate series starts at **0.7914** (2020-01-15), stays in the high-0.7s through
late Feb (0.7874 on 02-21), breaks sharply at 2020-02-25 (0.6302), continues down through early
March, and reaches its **low of 0.4975 on 2020-03-13** (staying near that floor through 03-17 at
0.4979) — a genuine ~37% relative de-risk. It recovers to 0.58-0.68 by mid-to-late April, ending the
window at 0.6436 (2020-05-01). **This de-risking is real but GRADUAL and somewhat LATE**: it does
not reach its floor until mid-March, several weeks after the crash began, and essentially coincides
with (not precedes) 2020-03-16 — the benchmark's own single worst day.

**2025-04 squeeze** (padded window 2025-03-15 to 2025-04-25; named window 2025-04-04 to 2025-04-11):
gate is 0.57-0.65 through mid-to-late March 2025, **drops sharply to 0.4034 on 2025-04-04**, reaches
its low of **0.3666 on 2025-04-07**, stays depressed through the named squeeze week (04-08=0.4221,
04-09=0.4521, 04-10=0.3771, 04-11=0.3791), then recovers from 04-14 onward, ending the window at
0.4885 (04-25). This is a real, sharp, ~26-percentage-point range of engagement — the HMM DID move
in real time — but it moved in the DE-RISKING direction in response to the initial down-move and
**stayed de-risked THROUGH the 04-09 relief-rally itself**: it did not "see" the reversal coming any
better than a reactive vol estimator would have (Fact 4's documented failure mode). Notably,
2025-04-09 was nonetheless the single LARGEST positive-return-contributing day in the winner's
entire path (+8.55%, see §4.3) — the book still captured a large gain that day despite the
depressed gate, which is a matter of favorable positioning/luck as much as protection.

Both windows: `covid_de_risked=True`, `squeeze_gate_moved=True` → **PASS**.

**Important qualification, independently confirmed** (see §6): 100% of this observed
"drawdown-control engagement" is Axis C (per-name regime reallocation) alone. Axis E's CDaR leverage
constant is fit ONCE for the whole backtest and never varies; `aggregate_portfolio()` renormalizes
total book weight to unit gross every day by construction, so the **book's aggregate deployed gross
exposure is bit-identical (to 10+ decimal places) across both crisis windows** — Axis E contributes
literally zero real-time risk reduction. `CTA_PREREG.md`'s narrative framing ("Axis C × Axis E
exposure control") overstates Axis E's role; the gate-5 PASS verdict itself is unaffected (it is
correctly coded to check only the regime-gate series), but the attribution should be corrected in
any future citation of this result.

### 4.6 Worst single day (gate 6) and melt-up decomposition re-run at full scale

Winner worst day: **−10.355%** (net), −10.341% (gross); runner-up: −10.614% / −10.600%. Both well
inside the −25% ruin threshold. **PASS** for both.

Melt-up decomposition, recomputed fresh on the full-93 run:

| | Winner | Runner-up |
|---|---:|---:|
| ret_2023_24_meltup (full-93) | +69.87% | +72.77% |
| cagr_full_sample | 19.357% | 21.260% |
| cagr_ex_meltup | 15.996% | 18.127% |
| haircut | 3.36pp | 3.13pp |

Confirms the Stage-A finding at full scale: a large fraction of both constructions' edge is
concentrated in the 2023-24 window.

### 4.7 Per-year and per-regime breakdown (winner, NET; runner-up nearly identical — see §4.1-4.6)

| Year | Winner net return | Winner net Sharpe |
|---|---:|---:|
| 2018 (partial) | 0.0% | 0.0 |
| 2019 | 6.84% | 0.604 |
| 2020 | 42.72% | 1.295 |
| 2021 | 28.45% | 1.801 |
| 2022 | −21.78% | −1.013 |
| 2023 | 31.42% | 1.905 |
| 2024 | 29.26% | 1.802 |
| 2025 | 23.71% | 1.192 |
| 2026 (partial) | 31.76% | 2.600 |

| Regime window (NET) | Winner | Runner-up |
|---|---:|---:|
| 2020-03 COVID | −15.09% | −15.64% |
| 2020H2/2021 recovery | +72.31% | +71.76% |
| 2018-Q4 chop | 0.0% (no data in window) | 0.0% |
| 2022 full year (chop/decline) | −21.78% | −21.76% |
| 2023-2024 melt-up | +69.87% | +72.77% |

A real, uncushioned −21.8% loss year in 2022 (the slow-grind bear year — Axis C's HMM did not
prevent this) stands out; strong performance is otherwise concentrated in 2020 recovery and
2023-2026.

### 4.8 Lookahead canary and injection re-verification (per-primitive)

Re-run on the full-93 universe's real signal arrays (not assumed from the 22-name pilot): Axis C
HMM injection test, Axis D vol-forecast injection tests (trailing-RV-20, Yang-Zhang-20, EWMA-0.94),
and Axis F correlation-causality injection test **all PASS** individually — each axis primitive is
confirmed causal in isolation. **However, see §6 for a genuine, confirmed, END-TO-END integration-
level lookahead violation found during independent verification that these per-primitive tests do
not catch.**

### 4.9 Trades/turnover and n_trades methodology caveat

`n_trades_total_all_names = 74,945` over 8.17y across 93 names (≈9,176/yr aggregate, ≈98.6/yr/name).
**This figure is flagged as a known-flawed metric** — see §6.3 — it counts continuous
partial-adjustment micro-steps as discrete "trades," a regression of a bug this same codebase's
lineage already found and fixed once (`research/indices/STATE.md` lines 50-58, I1's own
~185-198/yr raw-day-count bug). It does not corrupt the P&L (`apply_costs()` works directly off the
weight matrix), but should not be cited as a genuine round-trip count.

### 4.10 DSR / PBO (informational only, per PROGRAM CLASS)

| | Primary (N=29,760) | Sensitivity (N=9,920, `STAGE_A_22`-only) |
|---|---:|---:|
| sr_obs (per-obs) | 0.06522 | 0.06522 |
| sharpe_ann_obs | 1.0341 | 1.0341 |
| sr0_obs (expected max SR under N trials) | 0.09092 | 0.08516 |
| psr_vs0 | 0.9981 | 0.9981 |
| **DSR** | **0.1271** | **0.1881** |

Both low, as anticipated going into Stage B given this program's much larger breadth than I1's n=2
overlay family — but per PROGRAM CLASS, informational only, does not itself gate adoption (§6
discusses why this number should still weigh heavily on the reader's overall confidence).

PBO: the full-grid CSCV-PBO over the complete 29,760/9,920-cell family is **NOT reconstructable**
(Stage A persisted only per-cell summary metrics, not full return paths) — reported as `null`, not
fabricated. A genuine, clearly-scoped supplementary computation — CSCV-PBO (Bailey et al. 2015,
n_splits=8) over a real 12-member family (winner's and runner-up's own 13bp-cost book returns × all
6 frozen Axis-E leverage policies) — gives **PBO_proxy = 0.1857**. This is honestly computed but
explicitly disclosed as small and non-representative, not a stand-in for the full ledger's PBO.

---

## 5. The six-part adoption gate — applied explicitly, PASS/FAIL per gate

Per `CTA_PREREG.md`'s frozen, verbatim criteria, evaluated fresh on Stage B's winner construction at
the primary 13bp/funding-0 scenario:

| Gate | Criterion | Result | Verdict |
|---|---|---|---|
| **1** | CAGR at 41%-DD budget > 46.8% benchmark | 19.357% vs 46.8% | **FAIL** |
| **2** | CAGR at 30%-DD budget > 0.75×46.8% = 35.1% | 19.357% vs 35.1% | **FAIL** |
| **3** | Net Sharpe ≥ 1.353 − 0.05 = 1.303 | 1.0338 vs 1.303 | **FAIL** |
| **4** | Robustness (i) cost-scenario, (ii) drop-top-5, (iii) purged walk-forward | (i) monotonic no reversal — PASS; (ii) dd_budget_41 survives drop-5 — PASS; (iii) 95.24% of CPCV folds positive Sharpe (21/28 computable) — PASS | **PASS** |
| **5** | Drawdown-control visibly engages both 2020-03 and 2025-04 windows | COVID gate 0.79→0.50 (real, gradual/late); squeeze gate 0.63→0.37→recovers to 0.57 (real, sharp) | **PASS** |
| **6** | No ruin, worst single day > −25% | worst_day = −10.355% | **PASS** |

**ALL_GATES_PASS_PAPER_FIRST_RECOMMENDED = FALSE.**

Per the frozen prereg, this is recorded as this program's conclusion and **deployment is
recommended AGAINST** — a correctly-identified dead end is a complete, valuable result, not a
failure to find one, identical to the discipline used throughout E1/E2/E3/I1.

Note explicitly: the prereg's own pre-declared **Honest Prior** anticipated the *most likely*
outcome to be a pass on the budget-relative gates (2)/(3) and a fail only on the unconstrained gate
(1). **That specific pattern did NOT materialize.** Gate (2) — CAGR at the 30%-DD budget vs. a
discounted 35.1% threshold — also FAILS, by 15.7 percentage points. This is discussed fully, and
without softening, in §7.

---

## 6. Independent verification and adversarial-skeptic findings

Two independent processes were run against this Stage-B result: (a) a from-scratch independent
numerical reproduction using only the already-validated axis-module functions (not the pipeline's
own glue code), and (b) three adversarial-skeptic passes (lookahead, cost-realism, overfitting/
multiplicity) each targeting a specific class of failure mode. **All three skeptic passes returned
`verdict: concern`** — none found a "nothing wrong" result, and the overfitting pass in particular
is treated here, per its own instruction, as the single most important check for a two-stage
optimize-then-confirm program like this one; its conclusion is folded into the headline verdict
below, not buried.

### 6.1 Independent numerical reproduction

Hand-derived CAGR = 0.193566... and maxDD = 0.295008... vs. the pipeline's own reported 0.19357 /
0.29501 — **exact match**. All six gate sub-checks independently reproduced: **FAIL, FAIL, FAIL,
PASS, PASS, PASS → ALL_GATES_PASS=FALSE**, confirmed. Bug patterns (a) source-aware returns, (b)
derived annualization, (d) cost/funding scaling were independently re-verified correct at the
integration level (fresh 93-name buy-hold cross-check reproduced 41.947%/41.039%, matching the
pipeline exactly). Bug pattern (e) (leveraged/inverse/decay-ETF exclusion) reproduced correctly —
**but see §6.2 for a new, confirmed universe-reconciliation defect found in the process.**

### 6.2 Two confirmed defects not previously disclosed as such

1. **Integration-level lookahead bug (NEW, not one of the five documented bug patterns)**:
   `build_name_exposure()`'s per-name `target_daily_vol` calibration is the **harmonic mean of the
   ENTIRE (past+future) `fvol` series**, used as a single global constant applied uniformly to every
   date, including the very first tradable day. A permanent +50% price-level shock injected at the
   series midpoint (idx≈996-1001, tested on real SPYUSDT and MSFTUSDT) caused the exposure signal to
   diverge from the unshocked baseline starting at **idx=253 — roughly 750 days BEFORE the shock**,
   on both names identically. Quantified: the shock shifted `target_daily_vol` by +1.186%
   (0.0085041→0.0086050), shifting `vol_scale` at day 300 (696 days pre-shock) by +0.0156, affecting
   736/736 dates (100%) of the pre-shock 260-996 window. Each individual axis primitive (Axis A
   Kalman chain, Axis C HMM, Axis D forecasts) is independently confirmed causal; the leak is purely
   in this integration-level full-sample calibration step, present verbatim in `ctaB_pipeline.py`'s
   real code, shared identically by both winner and runner-up.
2. **AAPLUSDT unadjusted-split contamination present in the full-93 universe.** This program's own
   `CTA_MATH_NOTES.md` already documents a confirmed fake −74.12% single-day return on 2020-08-31
   from an unadjusted 4:1 split, and Stage A's own `STAGE_A_22` pilot deliberately EXCLUDES
   AAPLUSDT for exactly this reason — but `ctaB_pipeline.py`'s full-93-name `reconcile_universe()`
   INCLUDES it. Independently re-verified from the raw parquet (close 499.97 on 2020-08-28 → 129.40
   on 2020-08-31, exact match to the real 4:1 ratio, unadjusted). Quantified impact: AAPL's weight
   that day was 1.64%, contributing −1.22pp to the book's −1.37% return (≈89% of that day's loss).
   This is NOT the portfolio's worst day and does not appear in the drop-top-5 list, so it does not
   change the reported CAGR/maxDD/gate verdicts materially — but it is a confirmed, avoidable
   data-quality contamination of AAPL's own vol-forecast/HMM-regime/correlation inputs for months
   around that date, never flagged or reconciled between Stage A's exclusion and Stage B's universe.

Separately (found by the adversarial lookahead-lens pass, §6.4): Axis E's CDaR leverage constant is
**fit once on the FULL 2018-2026 realized path** (by explicit design, disclosed in the code
docstring but not flagged as a limitation in `CTA_PREREG.md`'s adoption-gate narrative). A late-
sample shock changes the full-sample L by ~19% (0.7656→0.6197) while a prefix-only fit is unchanged
— i.e. the deployed 0.9718× leverage was chosen with hindsight of the entire 8.17-year sample, which
a real 2018 deployment could not have known.

Neither defect reverses the qualitative conclusion — gates 1-3 fail by margins (27+pp on CAGR,
0.27 on Sharpe) far too large for a ~1-2%-scale vol-calibration artifact or a single 1.64%-weighted
name's one-day contamination to plausibly close — but the **specific point estimates
(CAGR=19.357%, maxDD=29.501%, Sharpe=1.0338, avg_leverage=0.9718) embed genuine hindsight** via both
the Axis-D calibration leak and Axis E's full-sample-fit leverage constant, and should not be cited
as a literal, real-time-achievable historical result without this caveat attached.

### 6.3 Cost-realism findings

`n_trades_total_all_names` (§4.9) is confirmed to reintroduce an already-fixed bug from this same
codebase's lineage (`research/indices/STATE.md` lines 50-58: I1 already found and fixed an identical
raw-day-count-vs-round-trip conflation). Independently measured TRUE round-trip-equivalent turnover
on 5 sample names (SPY 3.64, QQQ 3.99, AAPL 4.01, MSFT 4.40, NVDA 5.07 round-trips/yr/name) is 4-8×
higher than Axis B's own published "sane" baseline diagnostic (0.51-1.04/yr at `derived_baseline`,
`CTA_MATH_NOTES.md` §B.8) — expected given `aggressive`'s narrower band, but never flagged as a
cost-realism caveat in the winner-selection narrative. Back-solving implied turnover from the three
cost-scenario CAGRs gives an internally-consistent T_annual≈17.4-17.5 (book-level one-way,
~9× round-trip-equivalent/yr) — confirming the cost math itself scales genuinely linearly with no
hidden nonlinearity, but confirming a high-turnover regime relative to what a "CTA" construction
would typically run. No adoption-gate reversal found at 26bp cost.

### 6.4 Overfitting / multiplicity — the single most important finding of this report

This is the adversarial lens explicitly flagged as most consequential for a two-stage
optimize-then-confirm program, and its conclusion is stated here without softening, as instructed:

1. **The "robustness funnel" never actually ran before selection** — confirmed independently, not
   just as disclosed. The winner is not the survivor of any cost/drop-top-5/walk-forward filtering;
   it is simply the single highest-CAGR-at-30%-DD-budget cell out of 1,984 unfiltered candidates,
   picked by a plain `sort_values("cagr").head()`. This is disclosed in the prereg but deserves to
   be stated at full strength: **no cross-validation of any kind was applied before picking the
   winner.**
2. **The winner is statistically indistinguishable from the grid maximum.** Independently
   recomputed for this report: the winner's Stage-A Sharpe (1.0266) sits at the **99.96th
   percentile** of the full 29,760-cell grid; the true grid-max Sharpe (1.0277) belongs to a
   *different* combo (`ewma_5_20_80_shr1.0`/`aggressive`/`quarter_kelly`/`corr_risk_parity_w60_
   nospike`) that the CAGR-based selection rule didn't pick. This is the textbook look-elsewhere
   effect.
3. **DSR confirms this directly**: DSR = 0.1271 at the primary N=29,760 (0.1881 at the
   `STAGE_A_22`-only N=9,920) — both far below any conventional threshold for "distinguishable from
   search-luck" (typically >0.9). This is a decisive, already-computed answer: the observed Sharpe
   is **not** comfortably distinguishable from what you'd expect from the best of ~10,000-30,000
   correlated draws.
4. **The runner-up is a weak robustness check**, not independent corroboration — it differs from the
   winner in exactly one, already-known-to-be-nearly-inert axis (F: the `w120_spike`/`w120_nospike`
   collapse). A more meaningful comparator (a genuinely different Axis-A signal family scoring only
   ~0.6-0.7pp behind) sits in the same top-20 table and was not used as the designated comparator.
5. **The entire DD≤30% survivor pool comes from exactly 2 of 6 leverage policies** (`cdar` 992 cells,
   `gz_tight` 992 cells; every other policy has zero survivors) — and `cdar`'s own leverage constant
   is fit with full-sample hindsight (§6.2). The population the winner was even eligible to be drawn
   from exists partly because a hindsight-fit leverage constant happened to keep realized drawdown
   under 30% over this specific sample — compounding grid-search overfitting with a data-snooping
   mechanism baked into the candidate pool's own construction.
6. **Melt-up concentration** (§3.4, reconfirmed at full scale in §4.6) sharpens this further: the
   qual30 pool's CAGR distribution has mean 12.04%/std 2.6%, with the entire top-20 spanning only
   18.49% down to 17.30% — a razor-thin band where the "winner" is close to a coin flip.

**What could NOT be broken**: the qualitative FAIL verdict on gates 1-3 is not just correct but, per
this lens, *understated* as a "close miss" if reported as "informational only, does not gate." Given
DSR≈0.13, a grid-maximum-adjacent Sharpe, a hindsight-calibrated-leverage-policy monoculture in the
survivor pool, and material melt-up concentration, **this specific winner is largely
indistinguishable from the best of a large, correlated, partly hindsight-tainted, and never
cross-validated search** — independent of whether it beats the 46.8% benchmark. This qualification
is carried into the headline conclusion in §7 and §9, not left as a footnote.

---

## 7. Restated benchmark and exact comparison

**The benchmark** (Established Fact 6, restated in full): 93-name, point-in-time, equal-weight,
unlevered buy-and-hold over the full 2018-2026 sample — **Sharpe 1.353, CAGR 46.8%, maxDD 41.0%,
worst single day −12.1% (2020-03-16)**.

**Exact comparison, frozen construction (winner), Stage B, 13bp primary / funding=0:**

| Dimension | Construction | Benchmark | Delta |
|---|---:|---:|---:|
| CAGR (unconstrained / 41%-DD budget) | 19.357% net (22.098% gross) | 46.8% | **−27.44pp net (−24.70pp gross)** |
| CAGR at 30%-DD-budget vs. 0.75×-discounted threshold | 19.357% | 35.1% (=0.75×46.8%) | **−15.74pp** |
| Sharpe | 1.0338 net (1.1541 gross) | 1.353 | **−0.319 net (−0.199 gross)** |
| maxDD | 29.501% | 41.0% | **construction is 11.5pp / ~28% relatively tighter** |
| Worst single day | −10.355% | −12.1% | construction is modestly shallower |

**Honest characterization — stated plainly, per the prereg's own pre-authorization to report
whichever pattern actually occurs:**

The prereg's own Honest Prior, declared before Stage B ran, identified the *most plausible* positive
outcome as: "wins on CAGR-at-20%/30%-DD-budget (bands where the benchmark's own 41% drawdown
disqualifies it from even being a candidate) but does not win at the unconstrained top line." **That
specific pattern did NOT occur.** The construction fails gate (2) — CAGR at the 30%-DD budget
against the discounted 35.1% threshold — by 15.7 percentage points, not narrowly. It is not the case
that this construction "wins on a budget-relative basis but loses unconstrained"; it loses on both
the unconstrained basis (gate 1) and the budget-relative basis (gate 2), and also falls short on
risk-adjusted terms (gate 3, Sharpe). This is a **more decisively negative outcome than the prereg's
own central expectation**, and per the prereg's explicit instruction, is reported here without
softening or reframing as a partial win.

The one dimension where the construction genuinely, unambiguously outperforms the benchmark's own
risk profile is **maxDD**: 29.5% vs. 41.0%, an 11.5-percentage-point (~28% relative) reduction. Per
the prereg's own framing, had this drawdown reduction been paired with a CAGR clearing even the
discounted 35.1% bar, that combination — "beats the benchmark's own risk-adjusted terms even while
falling short of its raw, undiscounted number" — would have been a real, meaningful,
deployment-relevant finding about drawdown-efficiency, not a consolation prize. It is not paired with
such a CAGR here. A materially lower drawdown, standing alone, achieved at the cost of a CAGR that
does not clear even a return-scaled version of the same comparator, is not itself evidence of a
superior risk-adjusted construction — it is closer to what a passively-deleveraged (lower-exposure)
version of the same benchmark would plausibly deliver, though that specific counterfactual number was
not computed here and is not asserted.

Combined with §6.4's overfitting finding — the winner is not well-established as a distinguishable
signal from the best of a large, correlated, unfiltered, hindsight-tainted search — the honest
conclusion is a **comprehensive negative**, not a near-miss or a partial success dressed up
favorably.

---

## 8. Full test suite

```
cd /home/speculation/convexity-cta && PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
  /home/speculation/convexity/.venv/bin/python -m pytest tests -q
```

**Result, reconfirmed immediately before this report's own commit: `654 passed, 1 skipped` in
37.33s** — matches the established ≥654 passed/1 skipped baseline confirmed unchanged through
E1/E2/E3/I1 and through Stage A and Stage B of this program.

---

## 9. Recommendation

**PAPER-FIRST is NOT recommended.** The adoption gate fails decisively on all three numeric
criteria (CAGR-at-41%-DD, CAGR-at-30%-DD, net Sharpe), passes only the robustness/drawdown-
engagement/no-ruin criteria, and — per the overfitting-lens finding in §6.4, the single most
important adversarial check for a program of this shape — the winning construction is not even
well-established as a real, distinguishable edge separate from the best of a large, correlated,
never-cross-validated grid search (DSR 0.127-0.188, effectively grid-maximum Sharpe). This is
recorded plainly as this program's conclusion, exactly per the discipline used throughout
E1/E2/E3/I1: a correctly-identified dead end is a complete, valuable result.

Had the gate passed, the recommendation would have been a PAPER-FIRST proposal for a new,
separately-named, separately-capitalized book — distinct from the crypto cross-sectional momentum
book, distinct from any commodity book, and distinct from I1's own already-closed SPY/QQQ
timed-beta program — subject to a separate, later, explicit decision by the repo owner. **No such
proposal is being made.**

Per the prereg's own "What this program will NOT do" section: no re-tuning of the frozen
winner/runner-up parameters based on this result; no swap to the ex-meltup diagnostic leader (§3.4)
or to the runner-up as a replacement pick; no shorting; no leverage policy other than the frozen
CDaR construction's own closed-form output. No merge to `main`, no PR, no systemd changes, no live
capital. This program ends by reporting results in chat, per the task's own standing instruction.
