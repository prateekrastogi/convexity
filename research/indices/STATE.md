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

  **Next step: Pipeline phase.** `i1_pipeline.py` (not yet written) implements the frozen grid
  above, reusing `e3_pipeline.py`'s bug-fixed `metrics()`/`cpcv_paths()`/`dsr()`/`per_obs_sharpe()`/
  `per_year_sharpe()`/`pbo_cscv()` bodies verbatim (repointed `REPO` to
  `/home/speculation/convexity-indices`, since the old `convexity-equity` worktree no longer
  exists), plus a new daily-cadence, no-lookahead vol-target/exposure-timing implementation
  (NOT `regime/overlay.py`'s weekly/XSecBook-coupled version, confirmed not directly reusable in
  Setup). The exposure-timing alignment check (bug pattern 3, the central risk of this program)
  and its synthetic-outlier-injection test are the top implementation priority before any
  headline number is trusted.
