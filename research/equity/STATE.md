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
- **Final E1 commit**: `8eb25d9` (research(equity): E1 final report -- pre-named deployable fails
  adoption gate (clean negative), records independent-verification/adversarial-review findings) —
  placeholder above was left unfilled in the copy previously on disk; confirmed via `git log` and
  filled in now as part of the E2 prereg commit below.
- **E2 prereg phase**: complete. `research/equity/E2_PREREG.md` frozen — stress-conditioned
  residual reversal (weekly Friday cross-sectional reversal on the 102-name real-underlying-history
  panel, sector-residualized per `candidates.md`, lookback band {3,5,10} trading days as a
  band-ensemble, conditioned by a SPYUSDT/QQQUSDT realized-vol-z stress signal). N_trials = 104
  (4 unconditional + 72 conditioned grid + 10 universe-size sweep + 18 cost/funding scenario
  cells, canary excluded). Pre-named deployable = full 102-name universe, 3-lookback band-ensemble,
  continuous-scale stress conditioning (W_rv=20d, W_z=252d, mean of SPYUSDT/QQQUSDT z), 13bp
  primary cost. Explicitly declares source-aware (`source=='duka'`-only) daily-close construction
  and derived-not-assumed `periods_per_yr` to avoid repeating E1's two documented bug patterns —
  verified during drafting that `build_panel()`/`load_bars()` in `src/convexity/panel/{panel,
  data}.py` are NOT source-aware (load_bars doesn't even read the `source` column) and so are
  deliberately NOT reused for E2's core daily-close construction; a fresh, source-filtering builder
  is written instead. Frozen and committed at commit `573edbb` on branch `equity`, pushed to
  origin.
- **E2 pipeline phase**: COMPLETE. `research/equity/e2_pipeline.py` implements the full frozen
  grid exactly (unconditional band-ensemble + 3 per-lookback = 4 cells; conditioned grid
  W_rv{10,20,40} x W_z{252,504} x source{SPY,QQQ,MEAN} x mechanism{hard-gate x3 thresholds,
  continuous} = 72 cells; universe-size sweep N in {20..102} = 10 cells; cost/funding grid
  (deployable + unconditional comparator) x 3 RT x 3 funding = 18 cells; N_trials = 104 exactly,
  verified in the artifact's own `n_trials_ledger_breakdown.matches_declared_104=True`). Lookahead
  canary excluded from family/PBO/DSR per the mandatory convention. Outcome recorded in
  `research/equity/results_e2.json` (runtime 18.6s on the VM; 99 of the 102 base-universe names
  actually usable -- STXXUSDT/CBRSUSDT/DRAMUSDT excluded, each has too little real duka history
  (11-33 calendar days) to clear even the L=3 lookback + 20-day vol window + 5-day buffer at the
  data cut, stated here since the prereg's 102-name figure is the base universe declaration, not
  a claim every one of the 102 contributes an observation).

  **Verdict: pre-named deployable (continuous-scale-conditioned band-ensemble, full 99-usable-name
  universe, W_rv=20/W_z=252/mean-SPY-QQQ-z, 13bp primary) FAILS the adoption gate -- DO NOT
  DEPLOY.** Net Sharpe -0.9539 ann full-universe / -0.6441 ann on_demo-and-real-history (17-name)
  subset (both negative); DSR 0.0 at N_trials=104; drop-top-5-weeks Sharpe still negative (-1.4534
  full-universe); PBO over the full 104-cell family (86 cells actually poolable into one aligned
  CSCV family) = 0.20, which DOES clear the <=0.25 gate on its own but does not save the verdict
  since gates 1/2/4/5 all fail. turn_mean wired via metrics()'s `turn=` parameter this time (per
  the task's explicit note about E1's oversight there): 1.80/week gross for the full-universe
  deployable, sanity-checking against the declared ~2x/week cost assumption.

  **Regime-sanity signature (the mechanism check) FAILS.** Above-trailing-median-stress weeks
  contribute 47.91% of total P&L (needed >50%) while being 42.09% of calendar weeks; of the four
  named stress windows only 2018Q4 (weekly Sharpe +0.399 > full-sample -0.954, n=13) and 2022
  (-0.582 > -0.954, n=51) individually beat the full-sample Sharpe -- 2020-03 COVID (-2.148, n=4)
  and 2024-08 (-3.432, n=5) do NOT, so `all_named_windows_exceed_full_sample_sharpe=False` and
  `SIGNATURE_HOLDS=False`. This is reported plainly as a failed mechanism check, not papered over;
  the 2020-03/2024-08 windows have very few observations (n=4, n=5) so this is also a genuinely
  noisy small-sample read, stated as such.

  **Lookahead canary behaves correctly** (a real, important verification, not just a formality):
  net Sharpe +10.526 ann / gross +10.646 ann, clearly dominating the honest deployable's -0.954 --
  confirms the honest pipeline is not silently degenerate the way one E1 sleeve's canary was.

  **Universe-size sweep**: all 10 points N in {20,...,102} show negative point-estimate Sharpes
  (-0.54 to -0.95); N=20's point estimate (-0.584) is less negative than N=102's (-0.954) but the
  honest block-bootstrap 90% CI-overlap treatment finds NO N clears the "CI lower bound beats
  baseline AND <50% overlap" bar (`any_honest_peak_found=False`) -- verdict:
  "SUPPORTS (N=102 baseline is not honestly beaten by any sweep point)", i.e. the sweep does not
  contradict the pre-registered full-102-name breadth decision (there is simply no deployable
  edge at any breadth to have a peak in).

  **Two real, previously-undocumented pipeline bugs were found and fixed during this program's own
  mandatory pre-run smoke-testing** (caught BEFORE any headline number was trusted, per the task's
  explicit instruction to design the source-aware/derived-ppy logic correctly the first time and
  verify it -- these are a different class of bug from that mandate but were caught by the same
  discipline of testing every component before trusting the full run):
    (a) **Stress-signal ffill-past-end-of-history bug.** `build_stress_signal`'s SPY/QQQ daily-close
        series used `reindex(cal).ffill()`, which forward-filled each name's LAST REAL `duka` close
        indefinitely across the ~3 months after its own duka->perp splice date (SPY's real duka
        history ends 2026-04-06, QQQ's 2026-04-02, but `cal` runs through 2026-07-01) -- a flat,
        fabricated price series with zero daily variance, which made the rolling realized-vol
        z-score decay smoothly toward a manufactured value (verified: SPY's z-score under the
        buggy version decayed smoothly to -1.85 by 2026-07-01 instead of correctly going NaN) --
        exactly the kind of fake, non-mechanism-driven smoothness the regime-sanity gate is
        designed to catch, had it gone undetected. Fixed by explicitly NaN-ing any day past each
        stress name's own real last `duka` day before computing log-returns, so the stress signal
        (and hence `gate_continuous`'s exposure, which treats non-finite z as base 1.0x gross) goes
        genuinely inert past the real data's end rather than manufacturing a fake reading.
    (b) **Lookahead-canary sign/construction bug (caught by the mandatory canary-dominance check
        itself working as designed).** The first canary construction shifted the daily trailing-sum
        window forward by 1 day (a diluted, mostly-non-overlapping leak of only 1 of L trading
        days into the NEXT week) and showed a canary Sharpe of -6.07 ann, WORSE than the honest
        -1.44 -- immediately flagged as wrong by the mandatory "canary must clearly dominate"
        check. Root-caused and rebuilt: the canary now ranks directly on the actual realized
        forward return `wret[t]` itself (via a new `weekly_forward_name_returns` helper), but must
        be NEGATED before feeding into the same "long-bottom-quintile/short-top-quintile" reversal
        weight function (an un-negated first attempt at this also produced an impossible SHARPLY
        NEGATIVE canary Sharpe of -18.88, since "long the most-negative-signal names" applied
        directly to `+wret[t]` means "long the names about to have the WORST return" -- exactly
        backwards for a canary). Post-fix the canary shows +10.53 net / +10.65 gross ann, clearly
        dominating the honest pipeline, as required.
    Both bugs were caught and fixed BEFORE the reported run (not discovered post-hoc in an
    adversarial-review pass the way E1's two bugs were) -- a smoke-testing discipline directly
    informed by having just read E1's own "two real bugs found late" lesson.

  Full test suite: **654 passed, 1 skipped, 9 warnings in 36.88s** -- unchanged from the E1
  baseline, confirming `e2_pipeline.py` (a new, self-contained file) did not touch or break any
  existing `src/` behavior.

  Survivorship (landmine 4) and demo-gate (landmine 5) caveats restated per the frozen prereg:
  the 99-usable-of-102-declared name roster is today's Binance winners, not a point-in-time
  universe (less fatal for this short-horizon cross-sectional reversal than for a momentum study,
  per DATA.md, but not nil); on_demo-and-real-history restricts to 17 of the 19 raw on_demo names
  (OPENAIUSDT/QNTXUSDT excluded, perp-only/no real history) -- both headline numbers reported
  above (full-universe AND on_demo-17) per the mandatory dual-reporting convention.

  Committed and pushed to branch `equity` (see commit hash in the commit this STATE.md update is
  part of).
- **Next step**: E2 is closed (a correctly-identified, fully-diagnosed negative is a complete
  result -- the frozen prereg's discipline explicitly treats this as valuable, not a failure to
  find one). A future follow-up (E3 or beyond) would need either (a) a fundamentally different
  residualization/conditioning construction, (b) a genuinely different stress proxy less prone to
  the "only 4-5 named-window observations" small-sample problem documented above (2020-03 and
  2024-08 both had n<=5 weekly observations within this study's exact window definitions), or
  (c) acceptance that short-horizon cross-sectional reversal on this ~100-name Binance-listed
  TradFi-equity venue, at 13bp perp-reality cost and ~1.8x/week turnover, is simply not a
  deployable edge at this breadth -- per the task's own framing, a clean negative here is a
  complete, valuable result, not a failure to find one. Whoever picks up a follow-up should read
  this section plus `results_e2.json` directly before re-deriving anything.
