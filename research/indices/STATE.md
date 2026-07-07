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

  Committed at commit `<see next commit>` on branch `indices`, pushed to origin.

  **Next step**: none required by the frozen program -- a negative is a complete, valuable result
  per this program's own stated discipline (identical to E1/E2/E3). Any eventual deployment
  decision is explicitly NOT triggered by this program's gate outcome; the repo owner may choose
  to review the full grid family and results separately. No merge to `main`, no PR, no systemd
  changes were made or are proposed by this program.
