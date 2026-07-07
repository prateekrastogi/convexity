# C1 EQUITY-CTA STAGE B — PRE-REGISTRATION (Program 9; FROZEN 2026-07-07, before ctaB_pipeline.py)

Owner-directed follow-up to the closed E1/E2/E3 equity swing-research programs, the closed I1
timed-beta overlay program, and the four exploratory CTA-style scans (all on `main`,
research/equity/ and research/indices/), plus this program's own Stage A (six frozen axis modules
+ the redesigned 29,760-row exploratory sweep, all already committed to this branch). Isolation:
runs in the `equity-cta` worktree (`/home/speculation/convexity-cta`, branch `equity-cta`,
confirmed via fresh `git branch --show-current` immediately before this document was drafted);
reads the shared on-disk lake via the 8 real symlinks confirmed present with `ls -la data/`
(`cot`, `curves`, `dukascopy`, `earnings`, `funding`, `klines`, `stitched`, `universe` →
`/home/speculation/convexity/data/...`) — `paper` is deliberately NOT symlinked into this
worktree, confirmed absent. NEVER touches the live paper system (`/home/speculation/convexity` on
branch `main`, its systemd timers, or its `data/paper/` journals). All work under this program is
written to `research/cta/`, committed and pushed to `equity-cta` ONLY — no merge to `main`, no PR,
per the task's own standing instruction; this program ends by reporting results in chat. Executed
next by `ctaB_pipeline.py` (new, self-contained file under `research/cta/`, **not yet written as
of this freeze**); outcome to be recorded in `CTA_REPORT.md`.

## PROGRAM CLASS (restating I1's precedent explicitly — read this before the gate below)

I1 (`research/indices/I1_PREREG.md`) drew an explicit distinction between two kinds of program in
this codebase's history: **cross-sectional edge-certification programs** (E1/E2/E3 — breadth
~93-110 names, DSR at a full N_trials ledger as the *hard* adoption gate) and **benchmark-relative
overlay-validation programs** (I1 itself — an n=2-instrument timed-beta overlay judged against a
named buy-and-hold benchmark on CAGR/Sharpe/maxDD terms, with DSR/PBO reported only as
informational numbers that never gate anything).

**This program (C1) is, by its own frozen objective, in the SECOND class — despite having
E1/E2/E3-scale breadth.** Established Fact (1) is the reason: Sharpe is leverage-invariant in this
cost structure and therefore cannot be the optimization objective for any leverage question; the
correct objective family is **CAGR-at-a-declared-maxDD-budget**, benchmarked against the specific,
already-computed 93-name buy-and-hold benchmark (Established Fact 6: Sharpe 1.353, CAGR 46.8%,
maxDD 41.0%). That is a benchmark-relative question — "does this construction beat this named
comparator on these specific risk terms" — not a cross-sectional significance-certification
question. I1's own framing is adopted here explicitly, FOR THE SAME REASON, even though C1's
breadth (93-name full universe; 22-name Stage-A pilot; ~9,920–29,760 exploratory cells) is
structurally nothing like I1's n=2: DSR and PBO **can** be computed meaningfully at this breadth
and grid size (unlike I1, where the mathematics of DSR's own denominator genuinely does not apply
to a 2-instrument family) and will be, at the frozen denominator declared below — but they remain
**informational only**, because the adoption gate itself (below) was never framed as a
cross-sectional DSR pass/fail threshold. It is framed, identically in kind to I1's, as: does the
frozen construction beat a specific named benchmark on CAGR-at-a-DD-budget, Sharpe, and drawdown
terms, re-run fresh, robustly across cost scenarios and regime windows.

## Established facts and data contract (cited, not relitigated — see the task's own HOST/SAFETY
and ESTABLISHED EMPIRICAL FACTS sections for full text; verified fresh where this document depends
on them directly)

Facts (1)-(7) from the closed I1 program and four exploratory CTA scans are treated as inherited,
verified inputs: (1) CAGR-at-a-DD-budget is the only valid objective (Sharpe is
leverage-invariant); (2) shorting equities is net negative here — long/flat only by default; (3)
ruin is a single-day common-factor gap-risk problem, not diversifiable across 93 correlated names;
(4) trailing realized vol is reactive, not predictive (2025-04-09 is the standing counter-example);
(5) single-timeframe signals are fragile — ensembling is required (Axis A's multi-horizon
combination directly implements this); (6) **the benchmark to beat**: 93-name, point-in-time,
equal-weight, unlevered buy-and-hold, Sharpe 1.353, CAGR 46.8%, maxDD 41.0%, worst day -12.1%
(2020-03-16); (7) the five carried-forward bug patterns (source-aware returns, derived
annualization, exposure-timing alignment via decide-at-close/act-next-day, calendar-day-prorated
costs/funding scaled by realized exposure, leveraged/inverse/decay-ETF exclusion) — all five are
binding on `ctaB_pipeline.py` exactly as they were on the Stage-A axis modules and sweep script,
and will be re-verified fresh in Stage B via the same synthetic-outlier-injection convention used
throughout Stage A, not assumed to still hold just because they held for the axis modules.

Data contract: `data/stitched/{SYMBOL}.parquet`, schema
`open_time(ms)|open|high|low|close|volume|source('duka'|'perp')|gap_before(bool)`. Universe = the
93-name base reconciled in the closed E3 program and re-verified in the 93-name CTA scan (99
usable names minus the 6 leveraged/inverse/decay ETFs, per bug pattern (e)); Stage A's own pilot
subset (`STAGE_A_22`, deterministically built by `research/cta/_build_universe22.py` from
`research/cta/_universe_manifest_93.csv`) force-includes SPYUSDT/QQQUSDT and excludes AAPLUSDT (a
confirmed data-lake unadjusted-split bug, independently re-verified). Daily OHLC for any
range-based estimator is aggregated from 1-minute `source=='duka'` rows per real trading day, not
trusted from a pre-aggregated field.

## Stage A — what already ran (frozen context this document builds on; not re-litigated)

**Six frozen axis modules** (all committed to `equity-cta`, all independently causality-tested via
synthetic-outlier injection, all documented in `research/cta/CTA_MATH_NOTES.md`):

| Axis | What it is | Grid points (original) |
|---|---|---|
| A — Signal Extraction | causal multi-horizon trend filter (EWMA / Kalman local-linear-trend / causal rolling L1TF / IC-weighted combination) | 31 |
| B — Entry/Exit Control | Garleanu-Pedersen partial-adjustment / Davis-Norman no-trade-band optimal control | 4 |
| C — Regime Modeling | strictly-causal Gaussian HMM regime gate (filtering, not smoothing) | 5 |
| D — Vol Forecasting | Yang-Zhang / EWMA / GARCH(1,1) / HAR-RV, evaluated by causal QLIKE/MSE | 10 |
| E — Leverage Policy | Kelly fractions / Grossman-Zhou drawdown control / EVT ceiling / CDaR-constrained | 6 |
| F — Portfolio Construction | equal-risk-weight vs. correlation-aware risk parity, corr-window x spike-overlay | 8 |

**Original frozen design** (`CTA_MATH_NOTES.md` Section 9): full factorial on {A, B, F} = 992 core
cells, crossed with a replicated one-at-a-time sweep of {C, D, E} anchored at each axis's own
disclosed baseline (C=`hmm2_rf63`, D=`ewma_0.94`, E=`gz_loose`), OAT block width 19, giving
**N_cells = 992 × 19 = 18,848 per universe** — a 15.8x reduction from the naive 297,600-cell full
cross-product, with the exact cut (281 of 300 possible {C,D,E} triples per core cell) stated and
justified in full in Section 9.4. This is preserved verbatim in `CTA_MATH_NOTES.md` for the
record but was **never run** — it was superseded before any cell was computed.

**Redesign (Section 9-REDESIGN) — the design that actually ran.** Compute-tractability on this
VM's 2 vCPUs required a real reduction, disclosed in full rather than silently applied: (a) an
ADMM warm-start fix to Axis A's L1TF (~2x speedup, verified byte-close to cold-start, same
convex optimum); (b) a stratified 22-name Stage-A pilot universe (`STAGE_A_22`,
force-including SPY/QQQ, excluding the confirmed-buggy AAPLUSDT) in place of the full 93 for Stage
A only (Stage B, this document's own successor pipeline, uses the full 93); (c) Axis C trimmed
5→3 points (dropping both 3-state HMM variants, per Axis C's own disclosed short-history caveat)
and Axis D trimmed 10→3 points (Axis D's own top-3-by-QLIKE/MSE ranking, plus one disclosed
SPY/QQQ-specific GARCH retention) — Axis A, B, E, F grids all UNCHANGED. New OAT block = 10 (down
from 19). **New N_cells = 992 × 10 = 9,920 per universe**, run across **three universes**
(`STAGE_A_22`, SPY-alone, QQQ-alone) = **29,760 total rows** — a further 1.9x reduction on top of
the universe-subset and per-cell speedups, for a combined ~15.8×1.9 ≈ 30x reduction from the naive
297,600 baseline. Nothing from Axis A/B/E/F's own frozen grids was cut; the cut is confined to
Axis C (2 of 5 points) and Axis D (7 of 10 points), stated explicitly, not silently.

**The redesigned grid actually launched and completed**: 2026-07-07 13:02:21 UTC, polled to
completion (never killed/restarted) over 186.4 minutes wall-clock (over the sub-60-minute
aspiration, disclosed plainly — Phase 3's single-name portfolio-aggregation cost did not shrink
proportionally to its call-count reduction, a genuine finding about `aggregate_portfolio`'s
per-day Python-object-construction floor, not a bug). Result row count: **29,760**, exactly
matching 9,920 × 3, verified directly from `results_ctaA_full_grid.parquet`'s own row count and
`universe` value-counts (`STAGE_A_22` 9,920 / `SPY` 9,920 / `QQQ` 9,920). Committed to `equity-cta`
as `research/cta/results_ctaA_full_grid.csv.gz` (the repo's blanket `*.parquet` gitignore rule is
respected rather than force-added around, per the task's own explicit allowance).

**N_cells multiplicity ledger, final and frozen — this is the exact figure Stage B's own
informational DSR/PBO reporting below is keyed to**: the ORIGINAL 18,848/universe (56,544 total)
design is explicitly SUPERSEDED and never generated a single real result; the REDESIGNED
**9,920 cells per universe, 29,760 total rows across the three universes actually run**, is the
real, final multiplicity ledger for this program's Stage A.

## Selection funnel (frozen; the exact process that produced the winner and runner-up below)

**Stage 0 — leverage filter** (avg_leverage ∈ [2,5], applied to all 29,760 rows across all three
universes): **796 cells survive, ALL of them `e_key=half_kelly`, ALL in `STAGE_A_22`** (SPY-alone
and QQQ-alone each give exactly 0 cells with leverage in [2,5] anywhere in their own 9,920-row
grids). Every other leverage policy misses the [2,5] band entirely: `gz_loose` 1.20-1.82,
`quarter_kelly` 0.78-1.17, `gz_tight` 0.57-0.83, `cdar` 0.58-0.93 (all below 2.0); `evt_ceiling`
7.86-10.28 (already known-ruinous, well above 5.0).

**Stage 1 — intersect the leverage-filtered set with each DD budget** (the objective's OWN
feasible set, before any robustness filter): **DD≤20%: 0. DD≤30%: 0. DD≤41%: 0.** Checked across
all three universes — zero at every budget, everywhere. Root cause: the only cells landing
avg_leverage ∈ [2,5] (the 796 `half_kelly` cells) have realized maxDD ranging 57.9%-69.0%, which
blows through even the loosest 41% budget by a wide margin. There is no cell anywhere in the
29,760-row grid that combines meaningful leverage (2-5x) with a bounded drawdown.

**Robustness-filter stages (i)-(iii) — cost-scenario robustness, drop-top-5-contributing-periods
robustness, purged walk-forward robustness — are INAPPLICABLE / trivially 0, not because they were
run and failed, but because the candidate pool feeding them was already 0 before they could even
start**: start=0 → after (i) cost-scenario filter=0 → after (ii) drop-top-5 filter=0 → after (iii)
walk-forward filter=0. **None of (i)-(iii) has actually been coded or executed against a nonempty
candidate pool at any point in this program** — this is stated explicitly here because Adoption
Gate criterion (4) below requires Stage B to check the frozen construction against (i)-(iii), and
Stage B's own check is therefore the FIRST time these three robustness checks are ever actually
exercised in this program, not a re-confirmation of something already passed in Stage A.

Separately, and worth flagging on its own: even ignoring leverage entirely, **DD≤20% is infeasible
for the ENTIRE STAGE_A_22 grid** (0 of 9,920 cells achieve maxDD≤20% at any leverage). DD≤30% is
achievable by 1,984/9,920 cells and DD≤41% by 2,965/9,920 cells, but only at leverage well below
2x — the winning constructions run at avg_leverage ≈0.87-0.88x, i.e. **under 1x**.

### Zero-survivors disclosure (explicit, as required)

**`selection.zero_survivors = TRUE`.** Because the leverage-in-[2,5] restriction is infeasible
jointly with ANY declared DD budget (Stage 1 = 0 everywhere), the robustness-filter funnel could
not even start, so per the task's own fallback instruction, selection proceeds on the single best
cell by the raw objective. **That fallback itself required an additional, explicitly-flagged
deviation**: the raw objective's own leverage-in-[2,5] restriction had to be DROPPED entirely (not
merely the robustness filter relaxed), because enforcing it leaves literally zero cells at every
DD budget. This is reported plainly, as required, rather than silently relaxed or hidden inside a
footnote. **Selection rule actually used**: best CAGR-at-30%-DD-budget among all `STAGE_A_22`
cells, no leverage restriction — matching the tie-break already anchored in this program's prior
reporting.

## THE FROZEN CONSTRUCTION (the ONE selected; no further tuning after this point)

**Winner — frozen, exact parameters (every number below is a literal grid value, not
approximated):**

| Axis | Frozen value | Concrete meaning |
|---|---|---|
| A (signal) | `kalman_1e-06_1e-08_0.0001` | Local-linear-trend Kalman filter, `(q_level, q_slope, r) = (1e-6, 1e-8, 1e-4)` — the fastest-adapting of the grid's 3 Kalman triples (vs. the module's own coded default `(1e-5, 1e-7, 1e-4)`) |
| B (entry/exit) | `aggressive` | Garleanu-Pedersen/Davis-Norman partial-adjustment control, `cost_sensitivity (ζ) = 0.25`, `band_scale (η) = 0.5` — narrowest no-trade band and fastest convergence of the 4-point grid |
| C (regime) | `hmm2_rf63` | 2-state Gaussian HMM regime gate, refit every 63 trading days — Axis C's own baseline |
| D (vol forecast) | `ewma_0.94` | EWMA/RiskMetrics exponential smoothing, λ=0.94 — Axis D's own declared baseline/recommended default |
| E (leverage) | `cdar` | CDaR-constrained constant leverage (Chekhlov-Uryasev-Zabarankin 2005), `budget=0.25`, `alpha=0.95`, closed-form `L* = budget / CDaR_alpha(d)` |
| F (portfolio) | `corr_risk_parity_w120_spike` | Correlation-aware risk parity, 120-trading-day correlation window, spike-de-risking overlay ON — numerically tied byte-for-byte to the `_nospike` variant at this (22-name pilot) book size, a documented F-variant collapse Stage B must re-check at full 93-name scale |
| oat_axis | `baseline` | all three of {C,D,E} at their own disclosed baseline simultaneously (this cell sits at the OAT block's shared "1" cell) |

**Winner's Stage-A numbers (STAGE_A_22 pilot, 13bp primary cost, funding=0)**: CAGR = 18.491%,
maxDD = 28.098%, Sharpe = 1.027, avg_leverage = 0.882 (well below the originally-targeted 2-5x
band — the flagged deviation above), n_obs = 2,053, span_years = 8.167.

DD-budget results: `dd_budget_20 = FALSE` (fails; no `STAGE_A_22` cell at any leverage clears 20%
DD), `dd_budget_30 = TRUE` (the budget it was selected on), `dd_budget_41 = TRUE`. Note the
41%-budget objective is actually WON by a different, more-levered `quarter_kelly` cell
(avg_lev=1.17, CAGR=24.36%, maxDD=36.1%) — the construction frozen here is specifically the
30%-budget winner per the task's own tie-break, not a universal winner across all three budgets;
Stage B reports all three budgets for the frozen construction regardless.

Cost-scenario sensitivity: CAGR at 6.5bp = 19.587%, at 13bp (primary) = 18.491%, at 26bp =
16.329% — monotonic, modest sensitivity, no sign reversal.

**CAGR delta vs. the 46.8% 93-name buy-and-hold benchmark (Fact 6) = −28.31 percentage points** —
a large, decisive loss to the passive benchmark at the unconstrained top-line comparison (see the
Honest Prior section below for why this is not itself the gate).

**Runner-up — named robustness comparator, frozen, exact parameters (differs from the winner in
exactly one axis):**

Identical to the winner on every axis except **F (portfolio) = `corr_risk_parity_w60_nospike`**
(60-trading-day correlation window instead of 120, spike overlay OFF instead of ON). This is the
next distinct construction in the raw CAGR-at-30%-DD-budget ranking — the literal #2 and #3 rows
were exact numerical duplicates of the winner via the same `w120`-spike/nospike collapse, so this
is the first row that differs in a real parameter.

**Runner-up's Stage-A numbers**: CAGR = 18.096%, maxDD = 28.240%, Sharpe = 1.020, avg_leverage =
0.872, n_obs = 2,053, span_years = 8.167. `dd_budget_20 = FALSE`, `dd_budget_30 = TRUE`,
`dd_budget_41 = TRUE`. Cost scenarios: 6.5bp → 19.226%, 13bp → 18.096%, 26bp → 15.868%. CAGR delta
vs. benchmark = −28.70 percentage points. This is a near-clone of the winner (identical
signal/control/regime/vol/leverage axes, only the correlation-window/spike choice differs), 0.4pp
behind on headline CAGR and slightly worse on every cost scenario — a weak but genuine robustness
comparator, reported alongside the winner, never as an alternative pick.

### Melt-up concentration caveat (mandatory first-order caveat, not a footnote)

Exact geometric decomposition (`total_full = (1+cagr)^span_years − 1`; `total_ex_meltup =
(1+total_full)/(1+ret_2023_24_meltup) − 1`; `cagr_ex_meltup = (1+total_ex_meltup)^(1/(span_years
− 1.999)) − 1`, using the already-computed compounded 2023-01-01→2024-12-31 window return,
1.999 years, fully interior to the 8.17-year sample so the decomposition is exact):

- **Winner**: `ret_2023_24_meltup = +57.55%`, `cagr_ex_meltup = 16.29%` — a 2.20pp (≈11.9%
  relative) haircut from the full-sample 18.49%. Re-ranked by `cagr_ex_meltup` among the 1,984
  `STAGE_A_22` cells qualifying at DD≤30%: **the winner falls from rank #1 to rank #4** (0-indexed
  3). It is overtaken by an `ewma_5_20_80_shr1.0 × aggressive × cdar × corr_risk_parity_w120`
  construction (full-sample CAGR 17.72%, lower than the winner's 18.49%, but with a much smaller
  melt-up-window return of +48.20%, so it loses only ~1.28pp ex-meltup vs. the winner's ~2.20pp).
- **Runner-up**: `ret_2023_24_meltup = +57.76%`, `cagr_ex_meltup = 15.76%` — falls out of the top
  spots ex-meltup in the same way.

**Honest characterization, carried into Stage B as a first-order caveat, not a footnote**: the
winner's (and runner-up's) edge is MATERIALLY, not just modestly, concentrated in the 2023-24
melt-up — enough that excluding that single 2-year window flips the identity of the
best-performing construction in this DD-budget class. Both come from the same underlying signal
family (Kalman-filtered trend + aggressive entry control + CDaR leverage) that happened to have an
outsized directional read specifically during the melt-up window, and neither survives as the
field-leader once that window is removed. `ctaB_pipeline.py` reports the frozen construction's
melt-up-window contribution explicitly (both winner and runner-up), and separately reports the
ex-meltup leader identified here purely as DIAGNOSTIC CONTEXT — it is NOT promoted to a competing
pick or a third named comparator; per the frozen-grid discipline used throughout this program's
lineage, the winner and runner-up named above are not swapped out after the fact for a
better-looking ex-meltup cell.

## N_trials / DSR / PBO ledger for Stage B (informational only, per PROGRAM CLASS above)

Per the PROGRAM CLASS declaration above, DSR and PBO are computed and reported HONESTLY in
`CTA_REPORT.md` but **do not gate adoption** — the Adoption Gate below is entirely
benchmark-relative (CAGR/Sharpe/maxDD vs. the named 93-name buy-and-hold benchmark), never a
DSR/PBO pass/fail threshold, identical in kind to I1's own gate structure.

**Denominator, declared explicitly**: Stage B's DSR/PBO reporting for the frozen construction uses
**N_cells = 29,760** — the full, real, actually-executed Stage-A ledger across all three universes
(`STAGE_A_22` 9,920 + SPY-alone 9,920 + QQQ-alone 9,920) — NOT the original, superseded aspiration
of 56,544 (18,848 × 3), which never generated a single real result and is explicitly retired as
the denominator. Using the full 29,760 (rather than, say, restricting to `STAGE_A_22`'s own 9,920,
the universe the winner was actually drawn from) is the more conservative choice, consistent with
this program's own house discipline ("a large N_trials makes the DSR bar harder to clear, which is
the correct, conservative consequence of declaring an honest, complete grid rather than a curated
one" — CTA_MATH_NOTES.md Section 9.4 analog). `CTA_REPORT.md` states both figures (the
`STAGE_A_22`-only 9,920 and the full 29,760) so a reader can see the sensitivity of the
informational DSR number to which denominator is used, but 29,760 is the FROZEN primary figure.

**Expectation, stated before Stage B runs**: unlike I1's n=2 overlay family (where DSR was
expected to be low essentially by mathematical necessity), this program's much larger breadth
(93/22 names, 9,920-29,760 cells) means DSR/PBO are NOT expected to be trivially low or trivially
high for structural reasons alone — whatever Stage B computes is a real, informative number, it is
simply not the thing that gates adoption here, per the frozen objective (Fact 1) and the
benchmark-relative framing above.

## Stage B pipeline scope (`ctaB_pipeline.py` — not yet written; this section binds its contents)

Stage B re-runs the FROZEN winner and runner-up constructions above **fresh, on the FULL 93-name
universe** (not the 22-name Stage-A pilot — the pilot existed solely for Stage-A
compute-tractability; Stage B is the confirmatory run this program's adoption decision is actually
based on). No parameter in either construction may be re-tuned, re-optimized, or "nudged" based on
any Stage B result — the only degrees of freedom left to Stage B are: universe (93 vs. 22),
re-verification of the five bug patterns fresh on the full universe, and the specific
robustness/regime checks enumerated below. Concretely, `ctaB_pipeline.py` must:

1. **Re-run the frozen winner and runner-up on the full 93-name universe**, at the 13bp primary
   cost scenario and funding=0, reporting CAGR/maxDD/Sharpe/avg_leverage at all three DD budgets
   {20%, 30%, 41%} exactly as Stage A did, so the pilot-vs-full-universe delta is directly visible
   (including whether the documented `w120_spike`/`w120_nospike` byte-identical collapse (Axis F)
   still holds at 93 names — Stage A's own note flags this as worth re-checking at scale).
2. **Cost-scenario robustness (gate criterion 4-i)**: {6.5bp, 13bp, 26bp} — first real execution
   of this check against a nonempty candidate (only the frozen construction itself; not a search).
3. **Drop-top-5-contributing-periods robustness (gate criterion 4-ii)**: recompute CAGR/maxDD with
   the 5 largest positive-return-contributing periods removed from the frozen construction's own
   realized path; report whether the DD-budget-qualifying status and CAGR-vs-benchmark comparison
   survive.
4. **Purged walk-forward robustness (gate criterion 4-iii)**: an embargoed out-of-sample split
   (reusing the CPCV/purged-walk-forward machinery already present in this codebase —
   `research/equity/e3_pipeline.py` / `research/indices/i1_pipeline.py`'s `cpcv_paths()` — rather
   than reinventing it) applied to the frozen construction specifically.
5. **Drawdown-control exposure path, both named crash/squeeze windows (gate criterion 5)**: the
   frozen construction's combined regime-gate (Axis C) × CDaR leverage-sizing (Axis E) exposure
   control is traced explicitly through (a) the 2020-03 COVID crash (the benchmark's own worst day,
   -12.1% on 2020-03-16, per Fact 6) and (b) the 2025-04-09-style tariff-relief squeeze week (per
   Fact 4, the episode where trailing realized vol failed as a predictive signal) — report the
   actual exposure time series through both, not a summary statistic, so it is visible whether the
   policy de-risked in time for the slow-onset 2020 crash and whether it avoided over-de-risking
   into the sudden 2025-04 reversal (Fact 4's own documented failure mode for reactive vol
   estimators).
6. **Worst single-day check (gate criterion 6)**: report the frozen construction's own worst single
   trading day at its actually-deployed (CDaR-derived, time-varying) leverage — not the fixed
   0.882x average, since CDaR leverage is a level, not a guarantee of a fixed daily exposure.
7. **Lookahead canary and synthetic-outlier-injection re-verification**, reusing each axis module's
   own already-passed injection test but re-run on the full 93-name universe's actual signal
   arrays, not assumed to still pass unchanged from the 22-name pilot.
8. Full metrics table, per-year and per-regime breakdown (2020-03 COVID, 2020-H2/2021 recovery,
   2018-Q4 chop, mid-2022 chop/decline, 2022 full-year, 2023-2024 melt-up — the same regime windows
   I1 used, reused here rather than re-derived), GROSS shown alongside NET everywhere, and the
   melt-up decomposition (already computed once in Stage A, per the caveat above) recomputed fresh
   on the full-93 re-run.
9. DSR/PBO at N_cells=29,760 (informational, per the ledger above), reported plainly with the
   PROGRAM CLASS caveat restated.

**Reused machinery** (declared now, to steer `ctaB_pipeline.py`'s implementation and avoid
reinventing already-correct code): the six Stage-A axis modules under `research/cta/lib/`
(`signal_extraction.py`, `entry_exit_control.py`, `regime_hmm.py`, `vol_forecast.py`,
`leverage_policy.py`, `portfolio_construction.py`) are called with the frozen parameter values
above, unchanged; `metrics()` / `cpcv_paths()` / `dsr()` / `per_year_sharpe()` / `pbo_cscv()`
function bodies are reused from `research/equity/e3_pipeline.py` / `research/indices/i1_pipeline.py`
(the most bug-fixed versions in this codebase's lineage), parameterized by a DERIVED
`periods_per_yr` per bug pattern (b), never a hardcoded constant.

## Adoption gate (frozen, pre-outcome — declared EXACTLY as specified, not softened or altered)

PAPER-FIRST recommended only if ALL hold at the primary 13bp/funding-0 scenario:
1. CAGR at the 41% DD budget > 46.8% (the benchmark's own CAGR);
2. CAGR at the 30% DD budget > 0.75 × 46.8%;
3. net Sharpe >= 1.353 - 0.05;
4. the frozen construction (re-run fresh in Stage B's own Pipeline, not just Stage A's exploratory
   number) still passes the (i)-(iii) robustness checks;
5. the derived drawdown-control policy visibly engages during both the 2020-03 crash window and
   the 2025-04-09-style squeeze week (report the actual exposure path through both);
6. no ruin, worst single day > -25% at the deployed leverage policy.

If ANY gate fails, the negative is recorded plainly in `CTA_REPORT.md` as this program's
conclusion, and deployment is recommended AGAINST — a correctly-identified dead end is a complete,
valuable result, not a failure to find one, identical discipline to every prior program in this
codebase's lineage (E1/E2/E3/I1).

## Honest prior (stated explicitly, before Stage B runs)

The most likely outcome, stated now rather than after the fact, is that the frozen construction
**wins on CAGR-at-20%/30%-DD-budgets** (bands where the benchmark's own 41% drawdown disqualifies
the benchmark ITSELF from even being a candidate at those budgets) **but does NOT win at the
unconstrained top line** — gate (1) (CAGR at the 41% budget > 46.8%) is, on the Stage-A evidence
already in hand (the winner's own 41%-budget CAGR of 18.49% is far below 46.8%, and even the
actual 41%-budget WINNER among all Stage-A cells, `quarter_kelly` at 24.36%, is still far below
46.8%), the gate this construction is least likely to clear. Gates (2) and (3), which compare the
construction against a fraction of the benchmark's CAGR and against a small Sharpe shortfall
rather than the benchmark's own unconstrained number, are the gates where a real, honest pass is
plausible. **If Stage B finds exactly this pattern — pass on (2)/(3)-style budget-relative
comparisons, fail on the unconstrained (1) — this prereg's own gate structure already treats that
as a real, meaningful, and already-anticipated pass/fail distinction, not a consolation prize
manufactured after the fact.** `CTA_REPORT.md` must report whichever pattern Stage B actually
finds without softening it in either direction — including the possibility that the construction
fails even the budget-relative gates, given the melt-up concentration caveat above already shows
the construction's headline number is not fully robust to excluding a single 2-year window.

## What this program will NOT do

No re-tuning of the frozen winner/runner-up parameters based on any Stage B result. No swapping in
the ex-meltup diagnostic leader (Melt-up section above) as a new pick. No shorting (Fact 2). No
leverage policy other than the frozen CDaR construction's own closed-form output for the deployed
book (gate 6's ruin check applies to whatever that policy actually realizes, not a separately
imposed cap). No merge to `main`, no PR, no systemd changes, no live capital — a positive result
here recommends, at most, a PAPER-FIRST book proposal for a separate, later, explicit decision by
the repo owner, exactly as I1's own gate concluded for its own (negative) result.

## Full test suite (mandatory before every commit)

```
cd /home/speculation/convexity-cta && PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
  /home/speculation/convexity/.venv/bin/python -m pytest tests -q
```
must pass at the established ≥654 passed/1 skipped baseline before every commit in this program;
the pass count is included in every report. Reconfirmed immediately before this document's own
commit below.
