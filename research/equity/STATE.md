# Equity research state

- **Data phase**: complete. `research/equity/DATA.md` frozen (databento equity data lake, 102
  underlyings + 8 perp-only, full data-quality audit), landed at commit `601ef67` on branch
  `equity`.
- **E1 prereg phase**: complete. `research/equity/E1_PREREG.md` frozen — perp-native suite
  (S1 funding-carry cross-section, S2 perp<->underlying basis mean reversion, S3 overnight/weekend
  structure), N_trials = 179, pre-named deployable = S1 weekly band-ensemble + S3(c)
  overnight-execution overlay. Frozen and committed at commit `e2e586d` on branch `equity`,
  pushed to origin.
- **E1 pipeline + report phase**: COMPLETE. `research/equity/e1_pipeline.py` implements the full
  frozen grid (S1: 6 cells x 3 fee scenarios = 18; S2: 48 cells x 3 cost scenarios = 144; S3: 4
  trading cells x 3 cost scenarios = 12; + 5 ensemble/robustness trials = 179 total, N_trials
  matches the frozen ledger exactly). Outcome recorded in `research/equity/E1_REPORT.md` and
  `research/equity/results_e1.json` (md5 `6b8273ea85ed4441a04a6686b7af6a4e`).

  **Verdict: pre-named deployable (S1 weekly band-ensemble + S3(c)) FAILS the adoption gate --
  DO NOT DEPLOY.** Net Sharpe -1.67 ann full universe / -1.97 ann on_demo (pipeline's own √52
  annualization); DSR 0.00; PBO 0.586 (S1 weekly family); drop-top-5 -2.10 ann. S1's regime-sanity
  mechanism is technically positive but economically negligible and fragile (median funding pooled
  +1.4e-06, frac_pos 50.3% -- a jackknife shows this flips sign under single-symbol removal). S3(a)
  fact-check HOLDS (overnight Sharpe 3.44 > intraday 1.11 at SPYUSDT n=88; 2.99 > 1.51 at QQQUSDT
  n=89; pooled 0.64 > -0.17 across 102 names n=4832). S3(b) cross-sectional cells small-but-
  plausible (+0.71/+0.47/-0.23 ann across L in {4,13,26} weeks), none clear DSR at N=179. S3(c)
  overlay is a null result by construction (flat-bp cost model has no time-of-day variation to
  exploit, Delta Sharpe = 0.000 exactly).

  S2 (never the pre-named deployable) required two real bug fixes during the mandatory Sharpe>1.5
  bug hunt before its numbers could be trusted -- see E1_REPORT.md's "Bugs found and fixed"
  section: (1) the raw pre-stitch underlying (`data/dukascopy/{code}/{code}.parquet`, needed
  because `data/stitched/` is a sequential splice with ZERO simultaneous both-legs overlap) is not
  split-adjusted, so CRWDUSDT's real 2026-07-02 4:1 split produced a fake 139% single-bar basis
  jump -- fixed by reusing `scripts/stitch_tradfi_1m.py`'s own split-detection/adjustment
  functions directly; (2) S2's "daily" rebalance mask was `np.ones(n)` on a 1-minute-native
  calendar (rebalancing every bar, not once per real day), turning a real +259 ann GROSS
  microstructure signal into an impossible -521 to -635 ann NET Sharpe via self-inflicted
  overtrading -- fixed to rebalance once per real RTH day. Post-fix S2 daily-rebalance cells are
  small and flat/negative (-9.13 to +0.65 ann, no DSR pass anywhere) -- a clean dead end. S2
  WEEKLY-rebalance cells remain large (-3.04 to -118.74 ann) even after both fixes -- diagnosed as
  stale week-old vol-scaled sizing colliding with a fast 1-minute-timescale signal, producing
  94-99% single-name drawdowns -- flagged as unresolved/untrustworthy rather than force-fit clean.

  **Independent verification and adversarial review** (post-run, before this report was frozen)
  bit-exact-confirmed S1's regime-sanity number from a second, independent pandas-only
  re-derivation, confirmed both S1 and S3's lookahead canaries show the correct "artificially too
  good" direction, and additionally found TWO REAL, PREVIOUSLY UNDOCUMENTED BUGS in
  `e1_pipeline.py` (neither changes the adoption-gate conclusion, both are documented in full in
  E1_REPORT.md's "Independent verification and adversarial review" section, and both should be
  fixed before this pipeline is reused for a future sleeve):
    (a) `daily_returns_from_close()` (~line 308-318) has no `source`-aware guard alongside its
        `gap_days > 5` check, so the underlying-to-perp splice-seam day's "daily return" is
        computed across the basis-splice boundary at 55 of 110 names (e.g. PAYPUSDT shows a fake
        +30.0% "daily return" on its splice day) -- this contaminates the trailing-20-day RVOL
        used for S1's inverse-vol leg weighting for roughly a month per affected name.
    (b) every "weekly-rebalance" cell in S1 and S3(b) (including the pre-named deployable and both
        their canaries) is annualized with `WK_PER_YR=52.0` against a DAILY-marked P&L series that
        should use `DAY_PER_YR=252.0` -- understates every currently-negative Sharpe/CAGR by
        ~2.2x in magnitude (correct S1 deployable Sharpe is ~-3.67 ann, not -1.67) and would
        equally overstate a future positive weekly cell by the same factor. This already caused
        one concrete near-miss: S3's xsec_L4 cell's corrected Sharpe (~1.57) crosses the program's
        own "above ~1.5 gross triggers a bug hunt" threshold, which the originally-displayed 0.71
        number stayed under only because of this bug. DSR/PBO and the sign of every gate are
        VERIFIED UNAFFECTED (DSR uses per-obs Sharpe and true T directly, not the annualization
        constant) -- the adoption-gate FAIL verdict is robust to this bug.
  A dependence (single-name/single-period jackknife) lens independently re-derived S1's deployable
  from scratch (bit-exact match) and confirmed the negative result is broadly distributed across
  102 names and 5 months of data, not concentrated in one name or period -- the one fragility found
  is in the regime-sanity gate itself (5a), which sits at a near-zero, sign-flippable value under
  single-symbol jackknife.

  Full test suite: **654 passed, 1 skipped, 9 warnings in 37.81s** -- unchanged from the
  pre-pipeline baseline, confirming `e1_pipeline.py` (a new, self-contained file) did not touch or
  break any existing `src/` behavior. Run immediately before the E1_REPORT.md commit below.

  Committed and pushed to branch `equity`.
- **Next step: E2 is next.** E1 is closed (a correctly-identified, fully-diagnosed negative is a
  complete result, not a failure to find one). Any E1 follow-up would need either (a) a
  fundamentally different S2 construction that doesn't hold a vol-scaled continuous position for a
  full week against a fast-decaying 1-minute signal, or (b) a wider/older venue before S1's
  funding-carry mechanism accrues enough history to move past "short-history, directionally
  suggestive." Per the frozen prereg's discipline, S2's noisy weekly-cell numbers do NOT
  retroactively become a finding or a new deployable candidate without a fresh, separately-frozen
  pre-registration. Whoever picks up E2 should read E1_REPORT.md's "Independent verification and
  adversarial review" section first -- the two documented pipeline bugs (splice-seam daily-return
  contamination; weekly-cadence annualization mismatch) are exactly the class of defect E2's own
  pipeline should be checked against from the start, not discovered post-hoc again.
- **Final E1 commit**: `<FILLED IN AFTER COMMIT>`.
