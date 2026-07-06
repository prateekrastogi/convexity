# Equity research state

- **Data phase**: complete. `research/equity/DATA.md` frozen (databento equity data lake, 102
  underlyings + 8 perp-only, full data-quality audit), landed at commit `601ef67` on branch
  `equity`.
- **E1 prereg phase**: complete. `research/equity/E1_PREREG.md` frozen — perp-native suite
  (S1 funding-carry cross-section, S2 perp<->underlying basis mean reversion, S3 overnight/weekend
  structure), N_trials = 179, pre-named deployable = S1 weekly band-ensemble + S3(c)
  overnight-execution overlay. Frozen and committed at commit `e2e586d` on branch `equity`,
  pushed to origin.
- **E1 pipeline phase**: complete. `research/equity/e1_pipeline.py` implements the full frozen
  grid (S1: 6 cells x 3 fee scenarios = 18; S2: 48 cells x 3 cost scenarios = 144; S3: 4 trading
  cells x 3 cost scenarios = 12; + 5 ensemble/robustness trials = 179 total, N_trials matches the
  frozen ledger exactly). Outcome recorded in `research/equity/E1_REPORT.md` and
  `research/equity/results_e1.json`.
  **Verdict: pre-named deployable (S1 weekly band-ensemble + S3(c)) FAILS the adoption gate --
  net Sharpe -1.67 ann full universe / -1.97 ann on_demo, DSR 0.00, PBO 0.586 (S1 weekly family),
  drop-top-5 -2.10 ann. DO NOT DEPLOY.** S1's regime-sanity mechanism is technically positive but
  economically negligible (median funding pooled +1.4e-06, frac_pos 50.3% -- barely above a coin
  flip, a much weaker signature than the retail-long-bias carry story needs). S3(a)'s fact-check
  HOLDS (overnight Sharpe 3.44 > intraday 1.11 at SPYUSDT n=88; 2.99 > 1.51 at QQQUSDT n=89; pooled
  0.64 > -0.17 across 102 names n=4832). S3(b) cross-sectional cells are small-but-plausible
  (+0.71/+0.47/-0.23 ann across L in {4,13,26} weeks), none clear DSR at N=179. S3(c) overlay is a
  null result by construction (flat-bp cost model has no time-of-day variation to exploit, Delta
  Sharpe = 0.000 exactly, reported honestly not fabricated).
  S2 (not the pre-named deployable) required two real bug fixes during the mandatory Sharpe>1.5
  bug hunt before its numbers could be trusted -- see E1_REPORT.md's "Bugs found and fixed"
  section for the full diagnostic trail: (1) `data/dukascopy/{code}/{code}.parquet` (the raw
  pre-stitch underlying S2 needs, since `data/stitched/` turned out to be a sequential splice with
  ZERO simultaneous both-legs overlap anywhere) is NOT split-adjusted, so CRWDUSDT's real
  2026-07-02 4:1 split showed up as a fake 139% single-bar basis jump -- fixed by reusing
  `scripts/stitch_tradfi_1m.py`'s own `_detect_splits`/`_split_adjust`/`_mask_bad_windows`
  directly, plus dropping the split-effective day entirely (the perp itself re-pegs a few minutes
  into the underlying's own RTH session on the split day -- a real cross-venue timing lag, not a
  bug, but still needs excluding); (2) S2's `freq == "daily"` rebalance mask was `np.ones(n)` on a
  1-minute-native calendar -- i.e. rebalancing EVERY BAR, not once per real day -- which reshuffled
  ~half the book's gross every minute against a 13bp cost, turning a real (if itself already
  bug-hunt-triggering) +259 ann GROSS microstructure signal into an impossible -521 to -635 ann NET
  Sharpe purely from self-inflicted overtrading; fixed to rebalance once per real RTH day. Post-fix
  S2 daily-rebalance cells are small and flat/negative (-9.8 to +0.65 ann, no DSR pass anywhere) --
  a clean dead end. S2 WEEKLY-rebalance cells remain large (-7.6 to -122 ann) even after both
  fixes -- diagnosed (not just observed) as stale week-old vol-scaled sizing colliding with a fast
  1-minute-timescale signal, producing occasional 96-99% single-name drawdowns (vs 7-19% on the
  otherwise-identical daily cells) -- flagged explicitly as an unresolved/untrustworthy
  construction rather than force-fit clean. S2 was never the pre-named deployable, so none of this
  changes the program's headline verdict, per the frozen-grid discipline.
  Full test suite: 654 passed, 1 skipped -- unchanged from the pre-pipeline baseline, confirming
  `e1_pipeline.py` (a new, self-contained file) did not touch or break any existing `src/`
  behavior.
  Committed and pushed to branch `equity`.
- **Next step**: E1 is closed (a correctly-identified, fully-diagnosed negative is a complete
  result, not a failure to find one). Any follow-up would need either (a) a fundamentally
  different S2 construction that doesn't hold a vol-scaled continuous position for a full week
  against a fast-decaying 1-minute signal, or (b) a wider/older venue before S1's funding-carry
  mechanism accrues enough history to move past "short-history, directionally suggestive." Per the
  frozen prereg's discipline, S2's noisy weekly-cell numbers do NOT retroactively become a finding
  or a new deployable candidate without a fresh, separately-frozen pre-registration.
