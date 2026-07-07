# CTA (C1) research state

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

- **Next step**: Stage B Pipeline (`ctaB_pipeline.py`, not yet written) -- re-run the frozen
  winner and runner-up on the full 93-name universe, execute the adoption gate's six criteria for
  real (including the cost-scenario / drop-top-5 / walk-forward robustness checks for the first
  time in this program), trace the drawdown-control exposure path through 2020-03 and 2025-04-09,
  and report the result in `CTA_REPORT.md`.
