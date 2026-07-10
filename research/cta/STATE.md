# CTA (C1) research state

**PROGRAM STATUS: CLOSED. Both stages (A and B) complete. Comprehensive negative. DO NOT DEPLOY.
No PAPER-FIRST recommendation. See `research/cta/CTA_REPORT.md` for the full, final record.**

- **Stage A phase**: complete. Six frozen axis modules (A signal extraction, B entry/exit control,
  C regime HMM, D vol forecasting, E leverage policy, F portfolio construction) implemented,
  causality-tested, and documented in `research/cta/CTA_MATH_NOTES.md`. Original frozen exploratory
  grid (Section 9) was N_cells=18,848/universe; redesigned for compute-tractability on this VM's 2
  vCPUs (Section 9-REDESIGN: ADMM warm-start, 22-name `STAGE_A_22` pilot universe, Axis C/D grid
  trims) to N_cells=9,920/universe x 3 universes (`STAGE_A_22`, SPY-alone, QQQ-alone) = 29,760
  total rows. Redesigned grid actually launched and completed (186.4 min wall-clock, disclosed as
  over the sub-60-min aspiration), row count verified = 29,760 exactly. Results committed at
  `research/cta/results_ctaA_full_grid.csv.gz`. Selection funnel: Stage 0 (leverage in [2,5])
  leaves 796 cells, all `half_kelly`, all in `STAGE_A_22`; Stage 1 (intersect with any DD budget)
  = 0 candidates at every budget in every universe -- `selection.zero_survivors = TRUE`. Per the
  task's fallback rule, selection proceeded on the single best cell by the raw CAGR-at-30%-DD-
  budget objective with the leverage-in-[2,5] restriction explicitly DROPPED (flagged deviation).
  Winner: `kalman_1e-06_1e-08_0.0001` x `aggressive` x `hmm2_rf63` x `ewma_0.94` x `cdar` x
  `corr_risk_parity_w120_spike` (CAGR 18.49%, maxDD 28.10%, Sharpe 1.03, avg_leverage 0.882).
  Runner-up: identical except `f_key=corr_risk_parity_w60_nospike` (CAGR 18.10%, maxDD 28.24%).
  Both flagged: selected without surviving a robustness filter and without satisfying the
  leverage-in-[2,5] restriction. Melt-up concentration check: winner's edge is materially
  concentrated in the 2023-24 melt-up window (ex-meltup CAGR 16.29% vs full-sample 18.49%; winner
  falls from rank #1 to rank #4 of 1,984 DD<=30%-qualifying cells when re-ranked ex-meltup) --
  carried into Stage B as a first-order caveat, not a footnote.

- **Stage B prereg phase**: complete. `research/cta/CTA_PREREG.md` frozen -- states the ONE
  selected construction (winner above) and the runner-up as a named robustness comparator, both
  with exact frozen parameters (no further tuning permitted). Declares PROGRAM CLASS explicitly:
  benchmark-relative CAGR-at-DD-budget objective (per Established Fact 1), same class as I1's
  overlay-validation framing, NOT a DSR-certification program like E1/E2/E3 -- despite 93-name
  breadth. DSR/PBO reported informationally only at the frozen N_cells=29,760 denominator (the
  real, redesigned ledger; the original superseded 56,544 explicitly retired). Zero-survivors
  disclosed plainly per the task's requirement. Adoption gate declared verbatim, unaltered:
  (1) CAGR@41%DD > 46.8%; (2) CAGR@30%DD > 0.75x46.8%; (3) net Sharpe >= 1.353-0.05; (4) frozen
  construction re-run fresh in Stage B passes cost-scenario / drop-top-5 / walk-forward robustness
  checks (never yet actually executed against a nonempty pool -- Stage A's own funnel hit zero
  before these checks could run); (5) drawdown-control exposure visibly engages during both the
  2020-03 crash and the 2025-04-09-style squeeze week; (6) no ruin, worst day > -25%. Honest prior
  stated explicitly: most likely outcome is a pass on budget-relative gates (2)/(3) but a fail on
  the unconstrained gate (1), given the winner's own 41%-budget CAGR (18.49%) is far below 46.8%.
  Full test suite reconfirmed at 654 passed/1 skipped immediately before commit. Frozen and
  committed to branch `equity-cta` at commit `a22fcb5`, pushed to origin.

- **Stage B pipeline phase**: complete. `research/cta/ctaB_pipeline.py` re-runs the frozen winner
  and runner-up FRESH on the full 93-name reconciled base universe (confirmed exactly 93 via
  `reconcile_universe()`, identical logic to Stage A's own reconciliation). Wall-clock 348s. Fresh
  internal cross-check: 93-name equal-weight unlevered buy-and-hold on THIS run's own panel gives
  CAGR=41.95%/maxDD=41.04% (maxDD matches Established Fact 6's 41.0% almost exactly; CAGR a bit
  below Fact 6's 46.8%, expected given point-in-time entry/exit differences -- Fact 6 itself
  remains the frozen comparator, not superseded).
  **Winner** (13bp primary, funding=0): CAGR=19.36%, maxDD=29.50%, Sharpe=1.034, avg_leverage
  (CDaR closed-form constant L)=0.972, worst_day=-10.35%, no ruin. dd_budget_20=FALSE,
  dd_budget_30=TRUE, dd_budget_41=TRUE. Gross Sharpe=1.154 (net-of-cost drag ~0.12 Sharpe).
  **Runner-up** actually BEATS the winner at full-93 scale on every headline number (a genuine,
  disclosed reversal from Stage A's 22-name pilot, where the winner narrowly led): CAGR=21.26%,
  maxDD=29.76%, Sharpe=1.077, avg_leverage=0.987, worst_day=-10.61%.
  Axis F w120_spike/w120_nospike byte-identical collapse (flagged in Stage A as worth re-checking
  at scale): CONFIRMED still holds at 93 names, max_abs_diff=0.0 -- the correlation-spike overlay
  never actually triggers anywhere in this sample for this construction, at either universe size.
  **Adoption gate -- ALL SIX sub-checks computed fresh, real numbers**: (1) CAGR@41%DD 19.36% vs
  46.8% benchmark -- FAIL. (2) CAGR@30%DD 19.36% vs 35.1% threshold -- FAIL (this DEVIATES from
  the prereg's own stated honest prior, which expected a plausible pass here; the prereg itself
  explicitly anticipated and pre-authorized reporting this exact pattern without softening). (3)
  net Sharpe 1.034 vs 1.303 threshold -- FAIL. (4) robustness (i) cost-scenario monotonic
  (6.5bp/13bp/26bp = 20.72%/19.36%/16.68%, no sign reversal) PASS; (ii) drop-top-5-contributing-
  days (top5 = 2025-04-09, 2020-03-24, 2020-03-13, 2020-04-06, 2026-06-11) -- maxDD rises to
  34.70%, dd_budget_30 does NOT survive but dd_budget_41 does -- PASS (41%-survival used as the
  criterion); (iii) CPCV(8 groups,k=2,embargo=252d, embargo = Axis C's own min_train=252d, the
  slowest signal-component horizon in the frozen construction) -- 21/28 folds computable,
  95.2% positive Sharpe, mean 1.13 -- PASS. Gate 4 overall PASS. (5) drawdown-control exposure
  visibly engages both windows: COVID gate 0.79->0.50 (gradual, somewhat-late de-risking, bottoming
  mid-March after the crash was already underway); 2025-04 squeeze gate 0.63->0.37->recovers to
  0.57 (sharp, real movement exactly around the 4/4-4/11 window) -- PASS. (6) no ruin, worst day
  -10.35% > -25% -- PASS. **ALL_GATES_PASS = FALSE** -> deployment recommended AGAINST; PAPER-FIRST
  NOT justified by this frozen construction. DSR (informational, N=29,760 primary / 9,920
  secondary) = 0.127 / 0.188 -- low, as anticipated by PROGRAM CLASS, not itself a failure. PBO:
  full-grid PBO NOT reconstructable (Stage A persisted only per-cell summary metrics, not full
  return paths) -- disclosed honestly as None rather than fabricated; a genuine 12-member Axis-E x
  Axis-F proxy family gives PBO=0.186 (small, clearly-scoped supplementary number only). Per-year
  net returns show a real -21.8% loss in 2022 (the slow-grind bear year) and strong 2023-2026
  years; walk-forward thirds show the middle third (2021-01 to 2023-10) was weak (CAGR 4.6%,
  Sharpe 0.34) while the first and third thirds were much stronger -- consistent with the Stage-A
  melt-up-concentration caveat, now confirmed at full scale. Full test suite reconfirmed at 654
  passed/1 skipped immediately before commit. `results_ctaB.json` holds the complete numeric
  record (per-year, per-regime GROSS+NET, full CPCV fold detail, exposure time series through both
  named windows, melt-up decomposition for both constructions).

- **Independent verification + adversarial-skeptic phase**: complete. Independent from-scratch
  numerical reproduction (axis-module functions only, no pipeline glue reused) matched the
  pipeline's CAGR/maxDD/Sharpe/gate verdicts exactly, but surfaced two genuine, previously-
  undisclosed-as-such defects: (i) a NEW integration-level lookahead bug (not one of the five
  documented bug patterns) in `build_name_exposure()`'s per-name `target_daily_vol` calibration --
  it is the harmonic mean of the FULL (past+future) vol-forecast series, proven via injection
  testing to leak information ~750 days before an injected shock, into every date including the
  very first tradable day; (ii) the confirmed-buggy AAPLUSDT unadjusted-4:1-split name (excluded
  from Stage A's own `STAGE_A_22` pilot for exactly this reason) is present in Stage B's full-93
  universe, contributing a quantified but immaterial -1.22pp one-day contamination. Three
  adversarial-skeptic passes (lookahead, cost-realism, overfitting/multiplicity) were run; **all
  three returned `verdict: concern`**. Lookahead lens: Axis E's CDaR leverage constant is fit ONCE
  on the full 2018-2026 realized path (hindsight; a late-sample shock moves it ~19%), contaminating
  the CPCV/walk-forward CAGR/maxDD figures and the very DD-budget-qualifying status used for
  selection (Sharpe unaffected, by Established Fact 1's scale-invariance). Cost-realism lens: the
  pipeline's `n_trades_total_all_names` metric reintroduces an already-fixed raw-day-count bug from
  this codebase's own I1 program; true round-trip turnover is ~4-9x the module's own "sane"
  baseline, though the cost math itself is verified linear with no gate-flip at 26bp.
  **Overfitting/multiplicity lens (the single most consequential check for this program's
  two-stage optimize-then-confirm shape): the winner's Stage-A Sharpe sits at the 99.96th
  percentile of the full 29,760-cell grid (independently reconfirmed), is statistically
  indistinguishable from the grid maximum, and DSR = 0.127 (N=29,760) / 0.188 (N=9,920) -- both far
  below any threshold for "distinguishable from search luck." The entire Stage-A "robustness
  funnel" never actually filtered anything before selection (zero-survivors meant no
  cross-validation of any kind was applied); the DD<=30%-qualifying candidate pool is drawn
  exclusively from 2 of 6 leverage policies, one of which (`cdar`) is itself hindsight-calibrated;
  and the runner-up "robustness comparator" differs from the winner only in the near-inert Axis F.
  This construction is not well-established as a real, distinguishable edge, independent of
  whether it beats the benchmark.** None of this reverses the qualitative gate verdict (gates 1-3
  fail by margins far too large for these artifacts to close), but it means the FAIL verdict should
  be read as decisive, not as a near-miss softened by "informational only" DSR framing.

- **Final report phase**: complete. `research/cta/CTA_REPORT.md` written -- the complete, final,
  honest research record for Program C1 (mandate, Stage A math and selection funnel, Stage B
  full-93 confirmatory run, six-part adoption gate applied explicitly, independent-verification and
  adversarial-skeptic findings with the overfitting lens's conclusion carried into the headline
  verdict per its own instruction, restated benchmark comparison with the honest "did NOT achieve
  even the anticipated budget-relative pass" characterization, full test suite result, and final
  recommendation). Full test suite reconfirmed at 654 passed/1 skipped immediately before this
  commit. Committed and pushed to `equity-cta` at commit `c6baa2a` (see git log).

- **Program conclusion**: C1 (Stage A + Stage B) is a complete, honest NEGATIVE result -- the frozen
  construction fails all three numeric adoption-gate criteria (CAGR@41%DD, CAGR@30%DD, net Sharpe)
  at the full 93-name confirmatory scale, despite passing the three robustness checks and the
  drawdown-control/no-ruin checks. The prereg's own Honest Prior anticipated the most likely outcome
  as a pass on the budget-relative gates (2)/(3) with a fail only on the unconstrained gate (1);
  **that did not happen** -- gate (2) also fails, by 15.7 percentage points, a more decisively
  negative outcome than the prereg's own central expectation, reported here without softening or
  reframing as a partial win. The one genuine, unambiguous advantage over the benchmark is a tighter
  realized maxDD (29.5% vs 41.0%, ~28% relative reduction) -- but per the prereg's own framing, a
  lower drawdown standing alone, not paired with a CAGR that clears even a return-scaled version of
  the same comparator, is not itself evidence of a superior risk-adjusted construction. Compounding
  this, the overfitting-lens finding above means the winning construction is not even well-
  established as a real signal distinct from the best of a large, correlated, never-cross-validated
  search. No further tuning, no swap to the runner-up or any diagnostic-only cell as a replacement
  pick, no re-optimization -- per the prereg's own "What this program will NOT do" section. No
  PAPER-FIRST recommendation. No merge to `main`, no PR, no systemd changes, no live capital
  (program ends by reporting in chat, per the task's standing instruction).

  **The six-program arc and what it collectively suggests.** Counting E1, E2, E3 (equity
  swing-research: carry/basis/overnight/reversal/momentum/BAB/PEAD, comprehensive negative,
  apparent edges traced to 2-4 idiosyncratic names), I1 (SPY/QQQ timed-beta overlay: trend +
  vol-target + calendar tilts, clean negative -- CAGR gate failed at every N tested, though its own
  drawdown-reduction gate genuinely PASSED, 14.07% vs a 31.58% benchmark maxDD), the four
  exploratory CTA-style scans (established the CAGR-at-DD-budget objective, the 93-name benchmark,
  the shorting-is-negative and reactive-vol-is-not-predictive facts this program built on), and now
  C1 (a maximally-general six-axis CTA construction -- causal multi-horizon signal extraction,
  optimal-control entry/exit, regime-switching, vol forecasting, growth-optimal leverage,
  correlation-aware portfolio construction -- Stage A exploratory + Stage B confirmatory), this is
  **six research programs, spanning single-name swing signals, cross-sectional equity factors,
  index-level timed-beta overlays, and now a full state-of-the-art CTA stack, applied to
  essentially the same 93-110-name US equity/ETF universe over the same 2018-2026 sample.** Every
  one of them has failed to beat simply holding that universe, passively, unlevered, equal-weight.
  The one specific, identifiable, non-trivial thing that recurs across the arc -- not a fluke of
  one program -- is that **modern drawdown-control mathematics genuinely works mechanically**: both
  I1's vol-target/trend overlay and C1's HMM-regime-gated, CDaR-leverage-sized construction produce
  materially and measurably shallower realized drawdowns than the passive benchmark (I1: 14.07% vs
  31.58%; C1: 29.5% vs 41.0%). But in NEITHER program has that drawdown reduction been accompanied
  by a CAGR that survives even a proportionally-discounted comparison against the benchmark's own
  return -- the growth-optimal leverage math that is supposed to convert a lower-vol path into a
  competitive or superior CAGR has not, in any construction tested across this entire arc, actually
  delivered on that conversion. C1 adds a further, sharper qualification specific to this program's
  scale: its own overfitting-lens finding (DSR ~0.13-0.19, effectively grid-maximum Sharpe, a never-
  cross-validated selection funnel) shows that even the modest apparent edge surfaced by a
  large-scale grid search should not be over-interpreted as a durable discovery in the first place.
  **Honest carried-forward conclusion**: on this specific 93-110-name universe over this specific
  2018-2026 secular-bull sample, systematic timing/sizing/regime-detection edge-seeking is dead --
  five programs and dozens of construction variants deep -- for beating raw CAGR; the ONE
  specific, identifiable, worth-carrying-forward finding is that drawdown-control mathematics
  (vol-targeting, regime-gating, CDaR/Kelly-family leverage sizing) demonstrably and repeatably
  shrinks realized drawdown relative to a passive benchmark, even though it has not yet, in any
  construction tried, been paired with a leverage/growth policy that recovers competitive CAGR from
  that lower-risk path. Whether a genuinely causal (not hindsight-fit) leverage policy, or genuine
  diversification across return streams that do NOT share this universe's single common crash
  factor (Established Fact 3), could close that gap remains an OPEN question for a future program,
  not one this arc has foreclosed.

- **Next step**: none scheduled -- C1 concludes with this Stage B result unless the repo owner
  directs a new program.
