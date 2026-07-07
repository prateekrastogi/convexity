# Indices research state

- **Setup phase**: complete. Worktree `/home/speculation/convexity-indices` created from
  `main`@`f6797be` on branch `indices`. Data symlinks recreated for 8 of 9 real `data/` subdirs
  (cot, curves, dukascopy, earnings, funding, klines, stitched, universe) — `paper` deliberately
  EXCLUDED (live paper-trading journal, confirmed unreadable from this worktree). SPYUSDT/QQQUSDT
  `source=='duka'` history confirmed (778,291 / 777,249 rows, both spanning 2018-05-01 through
  their respective 2026-04 perp splice dates). Both confirmed `on_demo=true` in
  `src/convexity/panel/candidates.md` (fresh `load_universe()` call).

- **Data Acquisition phase**: complete. 67 regularly-scheduled FOMC meeting dates, 2018-01 through
  2026-07-04, scraped from federalreserve.gov's own calendar/historical pages, cached to
  `data/fomc/fomc_dates.csv` (untracked, gitignored, confirmed via `git check-ignore -v`). 7/7
  hand spot-checks PASS, including correct exclusion of the 2020-03-15 emergency cut and the
  2019-10-04 unscheduled conference call. No fabrication/approximation needed; contingency
  (dropping S3(b)) NOT triggered. Coverage summary at
  `research/indices/i1_fomc_data_coverage.json`.

- **I1 prereg phase**: complete. `research/indices/I1_PREREG.md` frozen — SPY/QQQ timed-beta
  overlay (S1 trend-filter ensemble [200d SMA+/-2% band, 12mo total-return sign, 10mo SMA], S2
  vol-targeting [W_vol in {20,60}d x cap in {1.0,1.5}], S3 calendar tilts [turn-of-month +
  pre-FOMC, both IN per the Data Acquisition outcome]). N_trials = 72 (S2 4 cells x S3 2 variants
  x cost/funding 3x3=9 scenarios = 72; S1 fixed at 1 ensemble construction; per-instrument
  decomposition and lookahead canaries explicitly excluded from the ledger). Pre-named deployable:
  50/50 SPY/QQQ blend, S1 ensemble x S2(W_vol=60,cap=1.0) x S3(both tilts on), 13bp primary cost.
  PROGRAM CLASS declared explicitly as overlay-validation (benchmark-relative adoption gate), NOT
  edge-certification like E1/E2/E3 -- DSR/PBO reported informationally only, expected low at
  breadth=2, not gating. Full test suite reconfirmed at 654 passed/1 skipped before commit.
  Frozen and committed at commit `202ad54` on branch `indices`, pushed to origin.

- **I1 pipeline phase**: complete. `research/indices/i1_pipeline.py` implements the frozen grid
  in full: S1 ensemble (200d SMA+/-2% band, 12mo-sign, 10mo-SMA), S2 vol-target (4 cells), S3
  calendar tilts (turn-of-month + pre-FOMC, with/without variants), the 50/50 SPY/QQQ buy-and-hold
  benchmark on an identical daily series/cost/funding convention, all 72 declared trial-cells, the
  8-cell base-construction robustness family, per-component lookahead canaries, the mandatory
  synthetic-outlier-injection test, CPCV(8,2,embargo=21d), DSR/PBO (informational), drop-top-5,
  per-year and per-regime metrics, and the full S1 regime-sanity signature.

  Two real bugs were found and fixed DURING pipeline development (before any headline number was
  trusted), both surfaced by the mandated adversarial/regime-sanity checks working exactly as
  designed:
    1. **S1(b)/(c) unit bug**: `.pct_change(252)` / `.rolling(210)` were initially applied to the
       already-MONTHLY-resampled close series (implying 252/210 *months* ~ 21yr/17.5yr lookbacks,
       which never validate in an 8yr sample) instead of the DAILY series (252/210 *trading days*
       ~ 12mo/10mo, per I1_PREREG.md's own stated units). Symptom that surfaced it: the 2018Q4
       whipsaw-cost regime-sanity check showed a suspiciously-clean all-zero S1 ensemble for ALL
       THREE component rules, not just the SMA. Fixed by computing the 252d-return/210d-SMA on the
       daily series and THEN sampling at month-ends; both rules now correctly warm up and activate
       ~2019-03/2019-06 given SPY/QQQ duka history starts 2018-05-01.
    2. **Turnover-metric definition bug**: gate 6 initially counted RAW DAYS with any nonzero
       `|delta-exposure|` as "trades/year", conflating S2's continuously-recomputed daily
       vol-target micro-adjustments (nonzero on a large majority of days) with discrete round-trip
       events, producing a nonsensical ~185-198/year figure. Fixed to the prereg's actual stated
       definition ("full ROUND-TRIP-EQUIVALENT exposure changes"): sum(|delta-exposure|)/2/year.
       Corrected value: 2.455/year (still fails the gate, but for the opposite, mechanistically
       sensible reason -- too LOW relative to the 5-15/yr expectation, not too high). The raw
       day-count and the S1-ensemble-only discrete-flip frequency (SPY 4.0/yr, QQQ 2.9/yr) are
       both retained in the output as diagnostics for transparency.

  Also added, beyond the bug fixes: an explicit data-coverage-artifact caveat on the 2018Q4
  whipsaw-cost window (`whipsaw_cost.*.warmup_caveat` / `all_exposure_zero_in_window` in
  `results_i1.json`) -- SPY/QQQ duka history only starts 2018-05-01, so ALL THREE S1 components
  are structurally un-warmed-up through all of 2018Q4, meaning the zero-flip result there is a
  DATA-COVERAGE ARTIFACT, not evidence the filter cleanly avoided whipsaw. Gate 3's regime-sanity
  check was correspondingly narrowed to require visible whipsaw specifically in the 2022 window
  (which IS warmed-up and testable), not treat the pre-warmup 2018Q4 window as satisfying that
  check.

  After both fixes: seam checks PASS for SPY/QQQ specifically (all_duka_only=true, splice-seam
  correctly detected and excluded for both symbols), all 3 lookahead canaries show correctly
  inflated Sharpe (S1: 0.688->1.002 ann.; S2: 0.688->0.784 ann.; S3's canary shows a small,
  honestly-expected effect of 0.688->0.677 given it only touches ~5/month+~1/FOMC single days out
  of ~2000 trading days), and the synthetic-outlier-injection test PASSES for both instruments (no
  exposure array reacts before the correct t-1 lag; confirmed on SPY and QQQ separately). Full
  test suite: 654 passed, 1 skipped (baseline unchanged) both before and after the fixes.

  **RESULT: clean negative, per PROGRAM CLASS's own stated discipline.** Deployable (Wvol=60,
  cap=1.0, both tilts, 13bp primary cost, funding=0) headline: net Sharpe 0.6882 (gross 0.7586,
  maxDD 14.067%, CAGR 5.98%) vs. the 50/50 SPY/QQQ buy-and-hold benchmark (identical daily
  series/cost/funding construction) net Sharpe 0.7691 (maxDD 31.58%, CAGR 14.92%) -- the overlay
  UNDERPERFORMS the benchmark on risk-adjusted terms in this sample, not merely fails to clear the
  +0.10 margin. Drawdown protection genuinely works as designed (maxDD 14.07% vs 31.58%, comfortably
  under gate 2's 0.65x=20.53% threshold; 2022 full-year drawdown 8.41% vs 30.20%), and the S1
  regime-sanity signature holds honestly on all 4 declared clauses (2020-03 COVID: real but
  somewhat-late de-risking, exposure crosses below 0.5 starting 2020-03-02/03-04, book maxDD 14.07%
  vs benchmark 31.58% in the Feb-Apr window; bull participation: mean exposure 0.76/0.58 (SPY/QQQ)
  through 2020H2-2021 and 0.76/0.56 through 2023-2024; 2022 whipsaw genuinely visible -- 104/83
  exposure-change days for SPY/QQQ costing ~48/16bp; 2022 full-year drawdown materially shallower).
  But only 2 of the 8 robustness-family cells (both W_vol=20/cap=1.0 variants) beat the benchmark's
  Sharpe at all -- the deployable's own pre-named W_vol=60/cap=1.0 cell is NOT among them, failing
  gate 4's robustness requirement outright. Drop-top-5 (0.5485 < benchmark's undropped 0.7691) and
  turnover-sanity (2.455/yr round-trip-equivalent, vs. 5-15/yr expected) both fail too.

  **Adoption gate: 2 of 6 sub-checks PASS (2: drawdown, 3: regime sanity), 4 FAIL (1: Sharpe,
  4: robustness, 5: drop-top-5, 6: turnover) -> DO NOT DEPLOY.** DSR=0.3199/PBO=0.0143
  (informational, N_trials=72, reported honestly per PROGRAM CLASS -- not gating, and notably NOT
  the reason this failed: the overlay fails on its own benchmark-relative terms regardless of what
  DSR/PBO say). The with/without-tilts separable attribution shows S3's tilts made things slightly
  WORSE in this sample (0.6882 with tilts vs. 0.7244 without) -- a small, honestly-fragile effect,
  exactly the kind the prereg flagged as expected going in. Full numbers in
  `research/indices/results_i1.json`.

  Committed at commit `ca34133` on branch `indices`, pushed to origin.

- **Independent Verify phase**: complete. Independent from-scratch re-implementation (no imports
  from `i1_pipeline.py`) plus a three-lens adversarial skeptic pass (lookahead, cost realism,
  benchmark fairness) were both run. All 4 carried-forward bug patterns re-checked fresh:
  (1) source-aware returns genuinely fixed (independently confirmed a real duka->perp splice seam
  for both symbols, notably SPY's is stamped `gap_before=False` despite a real price discontinuity
  -- retroactively validating the source-column-filter design choice over a gap_before-based one);
  (2) derived annualization genuinely fixed (independent n_obs/span_years arithmetic matches, no
  hardcoded 252); (3) exposure-timing genuinely fixed (synthetic-outlier-injection test
  independently rebuilt with an honest-vs-naive negative control, both confirming real
  discriminating power and a clean pass for the honest pipeline; separately, 5 boundary conditions
  hand-traced/numerically simulated directly against the pipeline source, all held); (4) cost
  scaling: trading-fee proportionality-to-|delta-exposure| genuinely fixed and verified both in a
  toy example and on real SPY data, BUT one real (non-gating) bug found -- funding's bp/week-to-
  bp/day proration is declared as calendar-day accrual (`/7`) but applied once per TRADING ROW
  rather than scaled by actual elapsed calendar days, understating funding magnitude by ~31% in
  the informational +-5bp/week scenarios only. Does NOT affect the primary 13bp/funding=0 gate
  table (funding=0 there), confirmed via the single shared `apply_costs()` code path feeding both
  deployable and benchmark identically. Flagged for fix if this pipeline is ever reused for a
  program needing correct nonzero-funding magnitudes.

  The benchmark-fairness lens (the most important lens for this program class) SURVIVES: identical
  date range (symmetric difference=0), identical daily-rebalance weighting convention, benchmark
  cost/funding machinery confirmed genuinely live (not dead code) via funding-scenario
  sensitivity, and an adversarial jackknife of the top 1/3/5/10 biggest exposure-change days showed
  the deployable's Sharpe DECREASES as those days are removed (0.6882->0.6344 at N=10) --the
  opposite of the failure mode the lens looks for, confirming the negative result is not propped up
  by a few lucky days. Hand-derived Sharpes matched the pipeline's own to within ~0.0005 for both
  deployable and benchmark. Test suite independently reconfirmed: 654 passed, 1 skipped.

  **RESULT: DO NOT DEPLOY, confirmed independently.** All adversarial checks corroborate the
  pipeline's numbers; the negative is a genuine property of this overlay construction in this
  sample, not a pipeline artifact. Full detail in `research/indices/I1_REPORT.md`.

- **I1 program: CLOSED.** Final report `research/indices/I1_REPORT.md` committed. Adoption gate:
  2 of 6 sub-checks PASS (drawdown, regime sanity), 4 FAIL (Sharpe, robustness, drop-top-5,
  turnover) -> **DO NOT DEPLOY**. DSR=0.3199/PBO=0.0143 reported informationally per PROGRAM CLASS,
  not gating and not the cause of the failure. This program is a **standalone follow-up**, not part
  of the closed E1/E2/E3 equity cross-sectional program -- it validates a fundamentally different
  construction class (timed-beta overlay on n=2 instruments vs. cross-sectional edge certification
  on ~93-110 names) and used a benchmark-relative, not DSR-based, adoption gate throughout, per its
  own PROGRAM CLASS declaration. Final commit: `<FINAL_COMMIT_HASH>` on branch `indices`, pushed to
  origin. No merge to `main`, no PR, no systemd changes, no live capital -- consistent with the
  program's frozen scope throughout.

  **Natural follow-up, if the repo owner wants to pursue this line further**: the 8-cell robustness
  family shows the W_vol=20/cap=1.0 corner (faster vol-target reaction, de-risk-only) beating the
  benchmark in BOTH the with- and without-tilts variants (0.7991 and 0.8161 vs. 0.7691) -- this
  was NOT the pre-named deployable and per this program's own pre-naming discipline is not
  promoted post-hoc, but a genuinely new, separately pre-registered program testing W_vol=20/
  cap=1.0 as its OWN frozen deployable (with a fresh, independent adoption-gate run, not a re-use
  of this program's results) would be a legitimate, honestly-scoped next step. A second candidate
  follow-up: broadening the instrument set beyond SPY/QQQ (e.g. adding IWM/DIA or a small basket of
  major sector ETFs) would increase breadth enough to make DSR a more informative (though still not
  necessarily gating) number, and would test whether the trend/vol-target/calendar overlay
  construction generalizes beyond two large, correlated, tech-tilted US equity indices. Neither
  follow-up is started by this program; both are left as explicit, separate, later decisions for
  the repo owner.
