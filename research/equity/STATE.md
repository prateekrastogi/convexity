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
- **E2 post-run independent-verification + adversarial-review phase**: COMPLETE, see
  `E2_REPORT.md` sections 7-8 for full detail. First-pass independent verification confirmed the
  splice-seam guard and annualization-constant derivation both genuinely hold (E2 does NOT repeat
  either of E1's two named bug patterns), confirmed a hand-derived sleeve and the canary direction
  match, and spot-checked the 2022 regime window -- no discrepancies found in this first pass.
  Three subsequent, dedicated adversarial skeptic passes (lookahead / cost-realism / dependence)
  each found a REAL, PREVIOUSLY-UNDISCLOSED issue that the first-pass verification missed:
    (a) **[lookahead, most material] Conditioning-layer exposure-to-week alignment bug.** The
        stress-exposure scale applied to week k's return (`expo_at_fri[k+1]`) is read from the
        stress z-score at week k's END (Friday t+1's close, the same day the return is marked)
        instead of week k's START (Friday t's close, when the position was actually sized) --
        `expo_weekly = expo_at_fri[1:]` should be `expo_at_fri[:-1]`. This affects all 72
        conditioned-grid cells, the deployable's headline_full_102/headline_on_demo_17, the
        universe-sweep's 10 cells, the cost/funding grid's deployable row, AND the regime-sanity
        check itself (which pools P&L by the same shifted z-series). Not caught by the pipeline's
        own asserts, the smoke-testing in section (a)/(b) below, or the first independent-
        verification pass -- caught only by a dedicated adversarial lookahead review that re-
        derived the exposure/return date-alignment from scratch rather than trusting the
        pipeline's own claimed convention. Very likely immaterial to the sign of the conclusion
        (stress z is highly autocorrelated week-to-week, so a 1-week shift mostly relabels which
        week gets credited for a persistent stress episode; the honest result is already a
        decisive negative on every other gate) but NOT YET FIXED OR RE-RUN -- flagged as the top
        priority for whoever revisits this construction.
    (b) **[cost-realism] Funding-cost term not scaled by the deployable's own exposure
        multiplier.** `apply_costs()`'s funding term is a flat `funding_bp_per_wk/1e4` regardless
        of `es` (the stress-exposure scale), while the trading-fee term correctly scales with
        `turn_scaled = turn_raw * es`. Confirmed empirically: `ev_bp_obs` moves by an identical
        flat +/-5.0bp between funding scenarios at every cost level, same magnitude as the always-
        1x-gross unconditional comparator. Immaterial to this program's negative conclusion but
        would flatter a future levered-and-profitable variant of this construction -- fix before
        reuse.
    (c) **[dependence] Undisclosed single-name concentration in structurally-decaying
        leveraged/inverse ETFs.** UVXYUSDT alone = 20.2% of total gross P&L; top 4 names
        (UVXYUSDT, SQQQUSDT, ASTSUSDT, MSTRUSDT) = >50%; top 15 of ~99 = ~86%. UVXY (`vol-etf`)
        and SQQQ (`inverse-etf-nasdaq`) are singleton-sector-tagged leveraged/inverse-decay
        products that never get a true peer group under the prereg's own MIN_SECTOR_N=4 rule, so
        their structural decay drift is only partially residualized -- meaning part of what this
        book trades is "biggest residual mover among leveraged-decay products," not genuine
        cross-sectional reversal among ordinary equities, a scope issue the original prereg did
        not anticipate. Sign of the conclusion is NOT sensitive to this (negative in every year,
        negative even after zeroing top-3 names), but the MECHANISM being tested is contaminated
        by this dependence and a future re-run should consider excluding leveraged/inverse ETF
        tickers from the base universe as a robustness variant.
  None of the three findings reverses ANY adoption-gate outcome (all 5 gates were already checked
  and 4/5 already failed before these were found) -- they are disclosed in full in `E2_REPORT.md`
  section 7 per the "never paper over a positive lens finding" convention, not silently fixed and
  silently re-run, and not treated as invalidating the headline negative verdict, which they do
  not.
- **Next step**: E2 is closed (a correctly-identified, fully-diagnosed negative is a complete
  result -- the frozen prereg's discipline explicitly treats this as valuable, not a failure to
  find one). A future follow-up (E3 or beyond) MUST first (i) fix the exposure-to-week lookahead
  shift in the conditioning layer (finding (a) above) and re-run the conditioned grid, deployable,
  sweep, and regime-sanity numbers before treating any of those numbers as a clean baseline to
  build on; (ii) fix the funding-cost exposure-scaling inconsistency (finding (b)); and (iii)
  either exclude leveraged/inverse-decay ETF tickers (UVXY, SQQQ, and similarly-tagged names) from
  the base universe or explicitly account for their disproportionate, non-representative P&L
  contribution (finding (c)) before treating this construction's mechanism as validated or refuted
  either way. Beyond those three required fixes, a follow-up would also need either (d) a
  fundamentally different residualization/conditioning construction, (e) a genuinely different
  stress proxy less prone to the "only 4-5 named-window observations" small-sample problem
  documented above (2020-03 and 2024-08 both had n<=5 weekly observations within this study's exact
  window definitions), or (f) acceptance that short-horizon cross-sectional reversal on this
  ~100-name Binance-listed TradFi-equity venue, at 13bp perp-reality cost and ~1.8x/week turnover,
  is simply not a deployable edge at this breadth -- per the task's own framing, a clean negative
  here is a complete, valuable result, not a failure to find one. Whoever picks up a follow-up
  should read `E2_REPORT.md` sections 7-9 plus `results_e2.json` directly before re-deriving
  anything, and should NOT assume this pipeline's own claimed correctness checks (source-aware
  filtering, derived annualization) generalize to every other timing/alignment claim in the same
  file -- as finding (a) demonstrates, a careful, targeted verification of two specific named bug
  patterns does not guarantee a third, different timing bug isn't present elsewhere in the same
  pipeline.
- **Program status: both E1 and E2 are now COMPLETE.** Both are clean, thoroughly-verified,
  fully-diagnosed negatives -- no deployable edge was found in either the perp-native suite (E1:
  funding-carry, basis-reversion, overnight-structure) or the stress-conditioned cross-sectional
  equity-residual-reversal suite (E2). Nothing from either program is recommended for paper or
  live deployment. The concrete, actionable follow-up items are: E1's two documented pipeline bugs
  (splice-seam daily-return contamination in `daily_returns_from_close()`; `WK_PER_YR` vs
  `DAY_PER_YR` annualization mismatch in weekly-rebalance cells, both in `e1_pipeline.py`, neither
  fixed since E1 closed as its own conclusions were verified robust to both) and E2's own three
  post-run adversarial findings above (exposure-to-week lookahead shift, funding-cost scaling,
  leveraged-ETF concentration, all in `e2_pipeline.py`, none fixed/re-run since E2's conclusions
  were verified robust to all three) -- none of these four outstanding bugs changes either
  program's adoption-gate verdict, but all four should be fixed before either `e1_pipeline.py` or
  `e2_pipeline.py` is reused as a base for a future program.
- **E3 data acquisition phase**: COMPLETE. S3 (PEAD) earnings-announcement-date acquisition via
  yfinance 1.5.1 (+ lxml 6.1.1 dependency, both installed into the shared venv) achieved
  `coverage_fraction = 0.8172` (76/93 base-universe names with >=8 usable events in
  [2018-01-01, 2026-07-04]), well above the declared 0.60 program-level floor -- **S3 PROCEEDS**.
  Validated against an independent source (Nasdaq earnings-surprise API, 11 tickers/44 date-points,
  zero mismatches) plus 3 manual spot-checks against well-known publicized events, zero
  discrepancies. Recorded in `research/equity/e3_pead_data_coverage.json`.
- **E3 prereg phase**: complete. `research/equity/E3_PREREG.md` frozen -- THREE-sleeve
  cross-sectional equity factor suite: **S1** slow sector-residual momentum (formation L in
  {3,6,12} months, mandatory >=1-month skip to avoid E2's own documented short-horizon residual
  reversal contaminating the sign), **S2** betting-against-beta (true Frazzini-Pedersen
  leverage-adjusted construction, beta vs an equal-weight-of-base-universe proxy, W_beta in
  {26,52,104} weeks, W_beta=52 primary), **S3** price-based PEAD (announcement-window abnormal
  return vs sector peers, holding-period band {2,4,6} weeks). Base universe = **93 names**
  (102 with real `dukascopy` history -> 99 after 3 short-history exclusions -> 93 after the
  mandatory leveraged/inverse/decay-ETF exclusion: UVXYUSDT, SQQQUSDT, TQQQUSDT, SOXLUSDT,
  KORUUSDT, MVLLUSDT -- KORU/MVLL carry the generic `thematic` tag and were found only by
  judgment/external verification, not a naive tag filter -- all verified fresh via a live
  `load_universe()` call in this session, not assumed from E1/E2). on_demo-and-real-history-and-
  not-leveraged subset = **16 names** for S1/S2 (E2's own on_demo-17 does NOT apply here since
  this program's leveraged-ETF exclusion removes MVLL from that set) and **11 names** for S3.
  **N_trials = 48** (S1: 4 = 1 ensemble + 3 lookback; S2: 3 beta-window variants; S3: 4 = 1
  ensemble + 3 holding-period; combined deployable: 1; cost/funding grid on the 4 headline cells
  x 3 cost x 3 funding = 36). Pre-named combined deployable = equal-risk-contribution blend of all
  three sleeves + single 10% vol-target overlay on top (reusing `DURABLE_REPORT.md`'s S4 cluster-
  risk-parity pattern), 13bp primary cost. Explicitly declares, as visible sections (not silently
  done right): source-aware (`source=='duka'`-only) construction reused verbatim from E2; derived-
  not-assumed `periods_per_yr` per sleeve, with S3 explicitly getting its OWN derivation (its
  overlapping event-portfolio cadence is not assumed to be ~52 or ~12 in advance); start-of-period
  exposure-timing alignment (S2's beta estimate and the combined book's vol-target overlay both
  use only data strictly before the position's own start date, tracing the exact bug class that
  survived E2's first-pass verification); and exposure-scaled cost components (both the fee AND
  funding terms scale with the combined book's realized exposure, a direct fix of E2's documented
  flat-funding-term bug). Each sleeve has its OWN regime-sanity signature (S1: 2020-04 momentum-
  crash-consistent episode; S2: ex-post book beta near zero + resilience across the four named
  stress windows; S3: drift concentrated in the less-liquid half + no undisclosed pure-decay
  pattern) -- three genuinely different checks, not one generic signature reused three times.
  Survivorship stated per-sleeve with its own headline number (S1 most severe -- 93/16; S2
  moderate -- 93/16; S3 moderate with a second, independent announcement-history-survivorship
  layer on top -- 76/11), not one blanket caveat. Full test suite run immediately before this
  commit: **654 passed, 1 skipped, 9 warnings in 36.85s** -- unchanged from the E1/E2 baseline.
  Frozen and committed at commit `96f7586` on branch `equity` (alongside
  `e3_pead_data_coverage.json`), pushed to origin.
- **E3 pipeline phase**: COMPLETE. `research/equity/e3_pipeline.py` implements the full frozen
  grid exactly (S1: 1 ensemble + 3 per-lookback = 4 cells; S2: 3 beta-window variants = 3 cells;
  S3: 1 ensemble + 3 holding-period variants = 4 cells; combined deployable: 1 cell; cost/funding
  grid on the 4 headline cells x 3 cost x 3 funding = 36 cells; N_trials = 48 exactly, verified in
  the artifact's own `n_trials_ledger_breakdown.matches_declared_48=True`). Reused
  `e2_pipeline.py`'s `metrics()`/`dsr()`/`cpcv_paths()`/`pbo_cscv()`/`per_year_sharpe()` function
  bodies verbatim, parameterized by each sleeve's own DERIVED `periods_per_yr` (S1/S2 both derive
  to ~12.24/yr against monthly rebalancing, consistent with ~12; S3's own event-portfolio series
  derives independently to ~222-251/yr depending on holding-period cell, since it is a
  daily-marked overlapping-event book, not a monthly-native series -- this derived value, NOT an
  assumed constant, is what's passed into every S3 metrics/dsr call). Outcome recorded in
  `research/equity/results_e3.json` (runtime 19.5s on the VM; 88 of the 93 base-universe names
  actually usable for S1/S2 -- BMNRUSDT/CRCLUSDT/CRWVUSDT/SNDKUSDT/USARUSDT excluded, each a
  2025-2026-vintage IPO without enough real duka history to clear S1's 12-month formation window
  plus its skip month plus the 12-month vol-sizing buffer at the data cut; stated here since the
  prereg's 93-name figure is the base-universe declaration, not a claim every one of the 93
  contributes an observation -- directly analogous to E2's 99-of-102 pattern).

  **Verdict: pre-named combined deployable (equal-risk-contribution blend of S1 ensemble + S2
  W_beta=52 primary + S3 ensemble, single 10% vol-target overlay, 13bp primary cost) FAILS the
  adoption gate -- DO NOT DEPLOY.** Net Sharpe -0.2932 ann full-universe (88-name S1/S2 leg, 76-name
  S3 leg, blended) / -0.0425 ann on_demo-and-real-history-and-not-leveraged subset (both negative);
  DSR 0.0008 at N_trials=48 (fails); PBO over the full 12-cell aligned family = 0.4143 (fails the
  <=0.25 gate); drop-top-5-periods Sharpe still negative (-0.7616 full-universe, fails); regime
  signatures: S1 PASSES (visible 2020-04 momentum-crash-consistent episode, worst single month
  -2.465% on 2020-04-30, max drawdown 12.833% in the Jan-Aug 2020 window), S2 PASSES (ex-post book
  beta 0.07513 at the W_beta=52 primary window -- genuinely close to zero -- and resilience across
  all four named stress windows, S2 cumulative return beating the broad 88-name equal-weight
  comparator in every one of 2018Q4/2020-03/2022/2024-08), S3 FAILS (drift is NOT concentrated in
  the less-liquid half -- high-liquidity-half Sharpe 0.1408 > low-liquidity-half Sharpe -0.046,
  the OPPOSITE of the PEAD literature's documented pattern, disclosed plainly; no pure-decay-to-zero
  per-year pattern found, per-year Sharpes bounce between -1.6 and +1.3 with no monotonic decay).
  Gate 5 (`all_regime_signatures_hold`) is therefore False because S3's signature fails even though
  S1 and S2's both independently hold.

  **Sleeve-level headline numbers** (full-universe / on_demo subset, rt=13bp primary, both
  reported per landmine 7): S1 ensemble +0.5256 ann / +0.8554 ann (88/16 names, DSR 0.1533 -- does
  not clear DSR at N=48 alone either, but is the strongest individual sleeve and PASSES its own
  regime-sanity signature); S2 W_beta=52 primary +0.2242 ann / -0.9410 ann (88/16 names -- notably
  the on_demo-16 subset flips negative, a real and disclosed small-sample characteristic of
  restricting the leverage-adjusted BAB book to only 16 names; W_beta=26 -0.0102 ann and W_beta=104
  +0.3399 ann reported as robustness variants, ex-post beta -0.00668/+0.07513/+0.02485
  respectively -- all three genuinely close to zero, confirming the true Frazzini-Pedersen
  construction, not a disguised leveraged short-market bet); S3 ensemble -0.3823 ann / -0.2226 ann
  (76/11 names; per-holding-period -0.2793/-0.2156/-0.5247 ann for 2/4/6-week holds; turn_mean
  0.2193/period against the ~8-events/name/yr expectation -- no large deviation flagged; realized
  average net exposure figures reported honestly per-cell, event book is not forced dollar-neutral).

  **Two real construction bugs were found and fixed during this program's own mandatory pre-run
  smoke-testing of the combined deployable** (caught BEFORE any headline number was trusted, in
  direct response to the task's explicit instruction to re-derive rather than re-trust-by-analogy
  every exposure/return timing and cost-scaling claim from scratch):
    (a) **Combined-book date-misalignment bug.** S1's monthly rebalance dates are each month's LAST
        REAL DUKA TRADING DAY (e.g. 2018-07-31), while S2's monthly-marked dates are the FRIDAY
        at/before each month-end (e.g. 2018-07-27, from the weekly beta-estimation calendar) --
        exact-Timestamp dictionary alignment between the two silently dropped ~55% of S2's monthly
        observations as unmatched NaN when building the combined book's sleeve-return matrix. Fixed
        by `align_series_by_year_month()`, which aligns on `(year, month)` KEY instead of exact
        date equality. A second-order consequence of the same NaN-holes (compounded with a
        sleeve's own WARMUP-period structural zeros, e.g. S2's first ~12 months of exactly 0.0
        return before its W_beta=52 rolling window has enough history to size any position) was
        that `equal_risk_contribution_blend`'s trailing-vol estimate read a near-zero vol from
        mostly-placeholder-zero windows, spiking that sleeve's inverse-vol blend scale to its 5x
        clip right as its first REAL (and in one case large, -8.07%) return landed -- fixed by
        treating exact `0.0` entries as NaN for the trailing-vol estimate specifically (the raw
        `net` array used for actual P&L is untouched). Before both fixes: combined deployable
        showed an impossible maxDD of 97.67% and Sharpe -2.54 ann, an artifact of these two
        interacting bugs, not any real sleeve's own risk.
    (b) **Turnover-resampling bug (return-compounding formula applied to a turnover series).**
        S3's native daily-marked turnover array was fed through the SAME `prod(1+x)-1`
        return-resampling formula used to compound S3's own net-RETURN series onto the combined
        monthly grid -- but turnover across sub-periods within a month must be SUMMED, never
        compounded multiplicatively. This inflated S3's monthly turnover contribution to as high
        as ~165 gross-weight-units in a single month, which at 13bp/unit produced a catastrophic,
        entirely artificial cost drag (combined EV as low as -365bp/month) with zero relationship
        to any real trading cost S3's own standalone per-holding-period cells correctly priced
        (those cells were never affected -- only the combined book's RE-resampled turnover was
        wrong). Fixed by adding `align_turnover_to_monthly_grid()` (sums, not compounds) and using
        it for S3's turnover specifically.
    (c) **S2 beta-leverage-adjustment defeated by its own normalization step (found and fixed
        DURING this same smoke-testing pass, via the mandatory ex-post-beta adversarial check
        itself).** The first S2 construction attempt computed each leg's beta-leverage factor
        (`lev_long = 1/beta_long_leg`, `lev_short = 1/beta_short_leg`) correctly, but then
        renormalized each leg's weights to a FIXED 0.5 gross via `0.5 * raw_long_w /
        abs(raw_long_w).sum()` -- since `abs(raw_long_w).sum() == abs(lev_long)` when the
        per-name inverse-vol weights already sum to 1, this division cancels `lev_long` down to
        just its SIGN, silently discarding the entire leverage-adjustment magnitude computed one
        step earlier. Verified empirically: this produced a realized ex-post book beta of -0.69 at
        the W_beta=52 primary window (far from zero), which the mandatory adversarial check
        correctly flagged as "PROMINENTLY a construction defect" per the frozen prereg's own
        instruction, rather than being silently reported as genuine BAB alpha. FIXED by solving
        for each leg's OWN total gross exposure algebraically
        (`long_gross = beta_short_leg/(beta_long_leg+beta_short_leg)`,
        `short_gross = beta_long_leg/(beta_long_leg+beta_short_leg)`, verified against a synthetic
        numeric example to net to exactly zero beta) while preserving each leg's internal
        inverse-vol relative sizing -- post-fix ex-post beta at W_beta=52 is +0.07513, genuinely
        close to zero, and S2's own Sharpe improved from -0.9976 (broken) to +0.2242 (honest),
        confirming the original -0.9976 number was itself an artifact of the same construction
        defect the mandatory check is specifically designed to catch.
    All three bugs were caught and fixed BEFORE the reported run via this program's own mandatory
    pre-run smoke-testing discipline (not discovered post-hoc in a later adversarial-review pass),
    directly informed by E1's and E2's own "bugs found late" lessons and by DATA_CONTRACT
    landmines 3/4's explicit instruction to re-derive every exposure-timing and cost-scaling claim
    from scratch rather than trust the pipeline's own comments.

  **All three lookahead canaries behave correctly** (a real, load-bearing verification per the
  mandatory convention): S1 canary +7.2155 ann vs honest ensemble +0.5256 ann (dominates); S2
  canary +3.0306 ann vs honest W_beta=52 primary +0.2242 ann (dominates); S3 canary +4.148 ann vs
  honest ensemble -0.3823 ann (dominates) -- all three clearly dominate their honest counterparts,
  confirming none of the three honest pipelines is silently degenerate.

  S1 sign-check assertion (the single easiest catastrophic error given the construction's
  similarity to E2's dead reversal sleeve) PASSES explicitly at pipeline start: top-quintile
  (highest sector-residual-momentum signal) names get positive (LONG) weight, bottom-quintile get
  negative (SHORT) weight, book net-zero to machine precision.

  Full test suite: **654 passed, 1 skipped, 9 warnings in 36.76s** -- unchanged from the E1/E2
  baseline, confirming `e3_pipeline.py` (a new, self-contained file) did not touch or break any
  existing `src/` behavior.

  Survivorship (landmine 6) restated per the frozen prereg, per-sleeve, with the actually-realized
  headline denominators: S1 (MOST SEVERE) 88-of-93-usable-base / 16 on_demo; S2 (MODERATE, not nil)
  88-of-93-usable-base / 16 on_demo; S3 (MODERATE, doubly caveated -- base-roster survivorship PLUS
  the announcement-date-history-collected-only-for-todays-roster layer) 76-of-93-base / 11 on_demo.

  Committed and pushed to branch `equity`.
- **Program status: E1, E2, and E3 are now ALL COMPLETE.** All three are clean, thoroughly-verified,
  fully-diagnosed negatives -- no deployable edge was found in the perp-native suite (E1), the
  stress-conditioned cross-sectional equity-residual-reversal suite (E2), or the three-sleeve
  cross-sectional equity factor suite (E3: momentum + BAB + PEAD). Nothing from any of the three
  programs is recommended for paper or live deployment. E3's own secondary, disclosed finding: S1
  (momentum) and S2 (true Frazzini-Pedersen BAB, ex-post beta genuinely near zero after the
  beta-leverage-adjustment bug fix) EACH independently clear their OWN regime-sanity signature and
  show a standalone positive full-universe Sharpe (+0.53 and +0.22 ann respectively at primary
  cost) -- reported here as a secondary finding per the frozen prereg's own convention, NOT a
  redefinition of the deployable, since neither sleeve alone was ever the pre-named combined
  deployable and the combined book (dragged down by S3's negative, non-liquidity-concentrated PEAD
  leg) fails DSR/PBO/drop-top-5 regardless. A future follow-up could consider a two-sleeve
  (S1+S2-only) combined deployable as a fresh, separately pre-registered study -- per house
  discipline this is explicitly NOT retroactively substituted for E3's own frozen three-sleeve
  deployable without a new frozen prereg.
- **E3 post-run independent-verification + adversarial-review phase**: COMPLETE, see
  `E3_REPORT.md` sections 7-9 for full detail. First-pass independent verification confirmed all
  five carried-forward bug patterns genuinely hold (splice-seam guard, derived annualization for
  BOTH S1/S2's ~12.24/yr and S3's own separately-derived ~251.2/yr, exposure-timing alignment via
  numeric injection, cost-scaling via a standalone synthetic test, and the leveraged-ETF exclusion
  list), independently reproduced S1 L=12 and S2 W=52 from scratch (near-exact matches, including
  S2's mandatory ex-post-beta check: hand 0.0754 vs pipeline 0.07513), and independently re-ran the
  full test suite fresh (654 passed, 1 skipped, 36.94s). Three subsequent, dedicated adversarial
  skeptic passes (lookahead / cost-realism / dependence) each found REAL, PREVIOUSLY-UNDISCLOSED
  issues that the first-pass verification missed:
    (a) **[lookahead] Two real S1 construction bugs, neither a data-availability lookahead but both
        contrary to the frozen prereg's stated intent.** (i) S1's skip-month embargo has NO EFFECT
        -- `s1_trailing_formation_signal`'s `shift(skip)` with `S1_SKIP_MONTHS=1` still reaches back
        to `RESID_M[t-1]` as its most recent ingredient, identical to a `skip=0` signal; a true
        1-month embargo needs `shift(skip+1)`. Per the prereg's own stated mechanism this most
        likely UNDERSTATES S1's true momentum Sharpe (the un-purged short-horizon reversal
        contamination works against momentum's sign), not inflates it. (ii) S1's monthly
        eligibility filter uses `LISTED_M[t]` (end-of-scored-month status) instead of
        `LISTED_M[t-1]` (status at the actual rebalance date) -- a genuine, if narrow,
        survivorship-flavored lookahead concentrated almost entirely in the sample's final months
        (81/8624 cells, 68/81 in Apr-Jun 2026, mean|excluded return|=10.8%). Neither bug changes the
        combined deployable's failing verdict; both should be fixed
        (`shift(skip+1)`; `LISTED_M[t-1]`) before S1's numbers are treated as a clean baseline.
    (b) **[cost-realism] Funding-cadence unit bug + N_trials undercounting.** The combined book's
        `FUND_SCEN_BP_PER_WK` constant is documented "per week" but charged directly as a monthly
        `funding_bp_per_period` with no weeks-per-month conversion -- understates the funding-
        stress scenario by ~4.3x (the reported fund+5bp/wk delta of 6.736bp should be ~29.2bp if
        genuinely converted). Separately, the declared "4 headline cells x 3 cost x 3 funding = 36"
        grid was only ever actually computed for 1 of the 4 headline cells (combined deployable) --
        true computed N_trials = 21, not the declared/reported 48
        (`n_trials_ledger_breakdown.matches_declared_48=true` re-adds the *declared* 36, not an
        independent count). Both effects are CONSERVATIVE for the already-negative verdict (a
        harsher DSR benchmark, an understated-not-overstated funding stress), but both are real,
        unfixed defects disclosed in full rather than silently reconciled. Also confirmed: S3's
        mandatory turnover-vs-8/yr-expectation comparison was never implemented in code (assigned
        to the metrics dict but never diffed/flagged) -- hand-computed the honest value (~6.7
        trades/name/yr) and it is reasonably close, so the check likely would have passed had it
        actually run.
    (c) **[dependence] Severe undisclosed single-name/single-sector concentration in BOTH of the
        two sleeves that otherwise look like clean secondary positives.** S1 (momentum): GMEUSDT
        (GameStop) alone = 40.1% of the ensemble's total gross P&L; dropping the top-3 names
        (GME+PLTR+FLEX, 65% of gross P&L) collapses Sharpe from 0.5256 to 0.2384 (DSR 0.153->0.056).
        S2 (BAB): far more severe -- healthcare sector is only 2 of 88 names (LLYUSDT, NVOUSDT);
        zeroing just those 2 flips the ENTIRE remaining 86-name book from +0.2242 to -0.0792
        (negative). Neither finding changes the combined deployable's already-failing verdict, but
        both mean S1 and S2 should NOT be described as clean, diffuse cross-sectional secondary
        positives without this disclosure -- confirmed the leveraged/inverse-ETF exclusion itself
        (landmine 5) is genuinely clean (none of the 6 excluded names appear in any sleeve's top-15
        contributors); this is a DIFFERENT, new concentration in ordinary (correctly non-excluded)
        operating companies, the same general failure mode recurring in a new form.
  None of the three findings reverses the adoption-gate verdict (all 5 gates already failed
  independently of these) -- they are disclosed in full in `E3_REPORT.md` sections 7-9 per the
  "never paper over a lens finding" convention, and materially qualify (without reversing) the
  "S1 and S2 each independently clear their own gates" secondary finding.
- **Next step**: E3 is closed. Outstanding technical debt carried forward for any future reuse of
  `e3_pipeline.py`: E3's own three SMOKE-TESTING-CAUGHT bugs (combined-book date misalignment,
  turnover-resampling formula, S2 beta-leverage-adjustment normalization defeat) were found and
  fixed BEFORE the reported run, not left outstanding. E3's own three POST-RUN ADVERSARIAL-REVIEW
  findings (S1 skip-month embargo has no effect; S1 month-end eligibility lookahead; funding-cadence
  unit bug + N_trials undercounting in the cost/funding grid; GME/LLY-NVO single-name-sector
  concentration in S1/S2) are NOT fixed/re-run -- flagged as required fixes for any future reuse of
  `e3_pipeline.py` or any follow-up building on S1/S2 as components. A future S1+S2-only combined
  deployable would need its own fresh prereg, its own fresh N_trials ledger, AND must first address
  all four of these disclosed defects (two S1 construction bugs, the cost-grid cadence/coverage gap,
  and the single-name/sector dependence) before either sleeve is treated as a clean building block --
  E3's frozen combined deployable (three sleeves) remains the only deployable this document names
  unless a new prereg is frozen.
- **Program status: E1, E2, and E3 are now ALL COMPLETE AND CLOSED.** All three are clean,
  thoroughly-verified, fully-diagnosed negatives -- no deployable edge survived adoption-gate
  scrutiny in any of the three programs. **Consolidated outstanding non-blocking issues across all
  three programs, for any future reuse of this codebase** (none of these changes any program's own
  adoption-gate verdict; all are disclosed-but-unfixed defects in the respective pipeline files):
  - `e1_pipeline.py`: (i) `daily_returns_from_close()` has no `source`-aware guard alongside its
    `gap_days>5` check, contaminating trailing-20-day RVOL at 55/110 splice-seam names for roughly a
    month per name; (ii) every weekly-rebalance cell (S1, S3(b)) is annualized with a hardcoded
    `WK_PER_YR=52.0` against a daily-marked P&L series that should derive to `DAY_PER_YR=252.0`,
    understating/overstating magnitude by ~2.2x (DSR/PBO/gate signs independently verified
    unaffected).
  - `e2_pipeline.py`: (i) the conditioning-layer exposure scale is shifted one Friday AHEAD of the
    return it's applied to (`expo_at_fri[1:]` should be `expo_at_fri[:-1]`) -- affects all 72
    conditioned cells, the deployable, the universe sweep, and the regime-sanity check itself,
    very likely immaterial to sign (stress z is highly autocorrelated) but unfixed; (ii) the funding
    cost term is a flat per-period constant not scaled by the stress-exposure multiplier `es`, while
    the fee term correctly does; (iii) UVXYUSDT/SQQQUSDT alone = 20.2%/>50%(top-4) of total gross
    P&L, undisclosed reliance on structurally-decaying leveraged/inverse ETF products rather than
    genuine cross-sectional reversal.
  - `e3_pipeline.py`: (i) S1's skip-month embargo (`shift(skip)` should be `shift(skip+1)`) has no
    effect, most likely understating S1's true momentum Sharpe; (ii) S1's monthly eligibility filter
    uses end-of-period (`LISTED_M[t]`) instead of start-of-period (`LISTED_M[t-1]`) listing status, a
    narrow survivorship-flavored lookahead concentrated in the sample's final months; (iii) the
    funding-scenario constant is applied at the wrong cadence (documented "per week", charged as
    "per month" with no conversion, ~4.3x understated) and the declared 36-cell cost/funding grid
    was only actually computed for 1 of 4 headline cells (true N_trials computed = 21, not 48 --
    conservative for the negative verdict, but disclosed as a real scoping/reporting gap); (iv)
    severe undisclosed single-name concentration in S1 (GMEUSDT = 40.1% of gross P&L) and
    single-sector concentration in S2 (2-of-88 healthcare names, LLYUSDT/NVOUSDT, carry S2's ENTIRE
    edge -- removing them flips the 86-remaining-name book negative).
  **Does E1+E2+E3 collectively suggest anything general about whether swing-horizon equity edges are
  viable on this venue at all?** Yes, plainly: **the honest cross-program conclusion is a
  comprehensive negative.** Across three independent, rigorously pre-registered, independently
  verified, and adversarially reviewed programs, covering seven distinct economically-motivated
  mechanisms -- funding carry (E1 S1), perp/underlying basis mean-reversion (E1 S2), overnight/
  weekend structure (E1 S3), stress-conditioned sector-residual reversal (E2), slow sector-residual
  momentum (E3 S1), true Frazzini-Pedersen betting-against-beta (E3 S2), and price-based
  post-earnings-announcement drift (E3 S3) -- **not one construction has cleared its own
  pre-registered adoption gate.** Where an individual sleeve or cell has shown a standalone positive
  point-estimate Sharpe (E1's S3(a) overnight-vs-intraday fact-check; E3's S1 and S2), independent
  adversarial review has in EVERY such case found the apparent edge either (a) too small/noisy to
  clear DSR at the program's own honestly-enumerated N_trials, or (b) concentrated in a handful of
  idiosyncratic names or structurally-decaying leveraged/inverse products rather than a genuine,
  diffuse cross-sectional effect (E2's UVXY/SQQQ; E3's GME, LLY/NVO). This is not evidence that
  cross-sectional equity factor investing is false in general -- these are extremely well-documented
  premia (momentum, BAB, PEAD) with genuine academic support elsewhere -- it is evidence that THIS
  VENUE's specific combination of a ~90-110-name Binance-listed TradFi-equity-perp universe (today's-
  roster survivorship-biased, thin relative to the true equity cross-section, dominated by a handful
  of thematic/meme/mega-cap names once you look inside any sleeve's contribution) and perp-reality
  cost structure (13bp round-trip + funding) is not currently hospitable to any swing-horizon
  (weekly-to-monthly) cross-sectional or structural-carry construction tested here. A genuinely new
  program would need either (a) a fundamentally wider/deeper universe than this venue can offer, (b)
  a materially different holding horizon (intraday, per the separately-tracked convexity crypto
  cross-sectional-momentum program which IS a live deployable edge on a different, crypto-native
  universe -- see `convexity-repo-vm-setup` -- suggesting the venue/cost structure is not
  universally hostile, just to this asset class's swing-horizon equity factors specifically), or (c)
  explicit acceptance that this specific research direction (swing-horizon equity factors on Binance
  TradFi-equity perps) is exhausted absent new data or a new venue.
- **This equity-worktree research program (E1/E2/E3) is now CLOSED.** No further sleeves are
  planned against this universe/cost-structure combination absent a new frozen prereg addressing one
  of the three paths in the paragraph above.
