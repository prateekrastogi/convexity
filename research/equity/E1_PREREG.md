# E1 PERP-NATIVE SUITE — PRE-REGISTRATION (Program 5; FROZEN 2026-07-06, before any run)

Owner-directed follow-up to the databento equity data lake build (research/equity/DATA.md).
Isolation: runs in the `equity` worktree (/home/speculation/convexity-equity); reads the shared
on-disk lake at data/stitched/ and data/funding/ (symlinked into the same lake as `main` and
`commodity` — reads only); NEVER touches the live paper system (main @ 601ef67's ancestor,
untouched by this branch) or src/convexity's live-trading paths. Executed by e1_pipeline.py;
outcome to be recorded in E1_REPORT.md. PAPER-FIRST BY CONSTRUCTION: this program's thesis is that
the *venue itself* — a new, immature TradFi-perp market on Binance — is the edge, so even a
passing gate below leads to a demo-paper deployment recommendation, never straight to live capital.

## Thesis (one program, three genuinely distinct signal families)

**S1 funding-carry cross-section.** Retail long-bias on a new perp venue should show up as a
persistent positive median funding rate; a dollar-neutral book short the richest-funding names /
long the cheapest-funding names, vol-scaled, should then harvest that as carry income — the direct
analog of the already-validated crypto funding-carry sleeve, but expected to look thinner here
because these perps are weeks-to-months old, not years old.

**S2 perp<->underlying basis mean reversion.** For each name with both legs overlapping, the ratio
of perp price to real underlying price ("basis") should be temporarily push-able away from its own
historical norm by perp-specific flow (funding pressure, retail one-sidedness, thin perp liquidity)
and should mean-revert; z-scoring each name's own basis history and betting on reversion when the
z is extreme should be more exploitable here than in most venues, because this dataset uniquely has
*both* legs at 1-minute resolution simultaneously.

**S3 overnight/weekend structure.** Because the perp trades 24/7 but the real economic
information for an equity (earnings, macro releases, order flow of the actual market-makers)
concentrates in the underlying's RTH session, the Lou-Polk-Skouras overnight/intraday
decomposition should show a real split: the perp's overnight leg (yesterday's RTH close to today's
RTH open, on the perp, spanning the full 24/7 window including the actual RTH-closed underlying)
should carry a disproportionate share of the return and/or a better Sharpe than the intraday leg,
and this should be both cross-sectionally exploitable and usable as an execution-timing overlay for
other sleeves.

## Universe / data / calendar

Universe = the 110-symbol roster in `src/convexity/panel/candidates.md`'s equities JSON array
(105 EQUITY incl. ETF subtypes, 3 KR_EQUITY, 2 PREMARKET), cross-checked against
`data/stitched/*.parquet` (all 110 present, verified in the setup phase — see CONTEXT). Price
basis = `data/stitched/{SYMBOL}.parquet`, 1-min OHLCV, `source` ∈ {`duka` (legacy field name; in
this lake it actually means "databento underlying proxy", see DATA.md), `perp`}, `gap_before`
(bool). Funding basis = `data/funding/{SYMBOL}.parquet` (`calc_time`, `funding_interval_hours`,
`last_funding_rate`), perp-era-only by construction (funding does not exist pre-listing) — 8
perp-only names (3 KR, 2 pre-IPO, 3 synthetic thematics BBX/BSP/QNTX) have funding but no
underlying and are therefore usable in S1 but excluded from S2/S3.

**Session/return hygiene (all sleeves, non-negotiable — landmines from the task, restated here as
frozen rules):**
- Never compute a 1-minute return across a `gap_before=True` boundary, and never across a row where
  `source` changes between consecutive rows (the underlying-to-perp splice seam) — both are treated
  identically to a data gap.
- Daily bars = each day's LAST bar with `gap_before=False` among that day's rows (the real RTH
  close for underlying-sourced days; for `perp`-sourced days post-splice, "daily" for S1/S2 purposes
  means the last bar of the UTC day, since the perp trades 24/7 and has no RTH boundary of its own
  — S3 instead uses the underlying's RTH calendar applied to perp prices, see below).
- Weekly returns (S1's declared weekly-rebalance variant) = Friday-close to Friday-close on daily
  closes built per the above rule.
- RTH boundaries for S3 = 09:30/16:00 America/New_York, DST-aware, via `pd.Timestamp.tz_convert`
  (reusing the exact convention already verified in `scripts/dbn_fetch.py`, not a re-derived
  timezone rule).

**Calendar span**: per-name, from each symbol's own funding-history start (S1) or perp-listing
date (S2, S3) through 2026-07-04 (the lake's last bar). No name is backfilled or extrapolated
before its real listing. This means N_eff differs sharply by sleeve and is reported per-sleeve,
not assumed — see Regime sanity and Costs sections below for how this is handled honestly.

## Sleeves (frozen; grids ARE the multiplicity ledger — every cell below is charged to N_trials)

### S1 — funding-carry cross-section
- **Universe**: every symbol with funding history in data/funding/ (up to 110, actual count is
  whatever currently has a non-empty funding parquet — perp-only names included, since S1 needs
  funding, not underlying).
- **Cadence grid**: {daily, weekly (Friday close)} — 2 variants.
- **Book construction**: dollar-neutral quintile book, short the top-quintile trailing-mean-funding
  names, long the bottom-quintile, using trailing mean `last_funding_rate` over lookback
  L ∈ {7, 14, 30} days as the ranking signal — 3 variants × 2 cadences = 6 cells.
- **Leg weighting**: inverse trailing-20-period realized-vol scaling within each leg (reusing the
  same inverse-vol convention documented in `src/convexity/portfolio/portfolio.py`'s
  `liqcap_volscaled` / risk-parity pattern — the crypto xsec momentum book's own vol-scaling
  convention, applied here rather than re-derived), floored at 0.5× the cross-sectional median vol
  exactly as swing_pipeline's `_weights_from_signal` floors low-vol names.
- **Mechanism-only cell (not charged as a separate trial, reported alongside)**: the frozen
  regime-sanity check itself — median `last_funding_rate` across the full perp-era panel — is NOT
  a backtest cell, it is the pre-belief gate (see Regime sanity below).
- Total S1 cells: **6**.

### S2 — perp<->underlying basis mean reversion
- **Universe**: the 102 names with real underlying history (perp-only names have no basis to
  compute).
- **Sampling grid**: {1-min raw, 5-min resample} for the basis series — 2 variants (declared here:
  1-min is the primary because both legs genuinely exist at that resolution; 5-min is a noise-
  robustness check).
- **Basis definition**: `basis[t] = perp_close[t] / underlying_close[t]`, computed ONLY on RTH
  minutes where both legs are present (post-splice overlap window), respecting `gap_before` and
  never spanning the source-change seam (per the session-hygiene rule above).
- **Normalization**: per-name z-score of `log(basis[t])` (log, not raw ratio, so names like PAYP's
  structural 0.43x level don't distort the z-scale) over a rolling window W ∈ {5, 20, 60} trading
  days — 3 variants.
- **Book structure (declared choice, frozen)**: CROSS-SECTIONAL, not single-name. At each rebalance
  point, rank all names' current basis z together and go long the most-negative-z (cheap perp
  relative to underlying) / short the most-positive-z (rich perp), dollar-neutral, vol-scaled legs
  — same inverse-vol convention as S1. Reasoning: per-name z removes the level effect (landmine 2),
  but a purely single-name mean-reversion book has no natural dollar-neutral hedge and would leave
  net market exposure that is really just S3's overnight/RTH exposure in disguise; cross-sectional
  ranking isolates the basis-specific signal and stays market-neutral by construction.
- **Rebalance-frequency grid**: {daily, weekly} applied to the ranking/rebalance step — 2 variants.
- **Extreme-entry threshold grid**: |z| trigger ∈ {1.5, 2.0, 2.5} for a threshold-gated variant
  (position only when |z| exceeds the trigger, else flat) reported alongside the always-in
  continuous-sizing variant (`position = -clip(z, ±3)/3`) — 3 threshold cells + 1 continuous cell
  = 4 variants.
- Total S2 cells: 2 (sampling) × 3 (window) × 2 (rebalance freq) × 4 (sizing) = **48**. This is
  large; it is declared in full here because every one of these is a real cell that could be run
  and every one is charged to DSR's N_trials — see Costs & validation.

### S3 — overnight/weekend structure
- **Decomposition**: for each RTH day, on the PERP price series, `overnight[t]` = yesterday's RTH
  close → today's RTH open (spans the full 24/7 non-RTH window), `intraday[t]` = today's RTH open →
  today's RTH close. `weekend[t]` = Friday RTH close → Monday RTH open, reported as its own cell
  (a special case of overnight, not summed into the daily overnight series to avoid double-count).
- **(a) In-sample fact-check (mandatory gate, not itself a trading cell)**: at the INDEX level
  (SPYUSDT, QQQUSDT) and cross-sectionally across all 102 underlying-backed names, does mean
  |overnight return| exceed mean |intraday return|, and does overnight Sharpe exceed intraday
  Sharpe? This must hold before (b) or (c) are trusted — see Regime sanity below.
- **(b) Cross-sectional variant**: rank names weekly by trailing overnight-Sharpe strength over
  lookback L ∈ {4, 13, 26} weeks, go long the top-quintile / short the bottom-quintile (dollar-
  neutral, vol-scaled) — 3 variants.
- **(c) Execution-timing overlay**: for S1's pre-named deployable cell (see below) ONLY, test
  whether concentrating that sleeve's rebalance/exposure timing into the overnight window (vs an
  evenly-spread intraday execution assumption) changes realized Sharpe — 1 variant (declared as an
  overlay ON TOP of S1's cell, not a standalone signal, and charged as its own trial since it is a
  genuine grid cell that could have come out either way).
- Total S3 cells: 3 (cross-sectional L) + 1 (overlay-on-S1) = **4** trading cells, plus the
  mandatory (a) fact-check which is a gate, not a trial.

### Cost scenarios (multiply every trading cell above)
Binance USDT-M VIP0 taker 5bp per leg; primary round-trip = 13bp per name-period traded, with
declared scenario variants {6.5bp, 26bp} — **3 cost scenarios** per cell for S2 and S3 (funding is
not a cost in S1, see below).
- **S1 exception (declared per METHODOLOGY)**: funding IS the P&L mechanism in S1, not a cost —
  only the taker trading fee is scenario'd there (3 fee scenarios: {6.5bp, 13bp, 26bp} applied to
  the rebalance turnover only, funding itself never scenario'd).

### N_trials ledger (the full multiplicity charged to DSR)
- S1: 6 cells × 3 cost scenarios = 18
- S2: 48 cells × 3 cost scenarios = 144
- S3: 4 trading cells × 3 cost scenarios = 12 (the (a) fact-check is excluded — it is a gate, not
  a scored trial)
- Ensembles/robustness reported on top of the family (declared now, added to the ledger): S1
  band-ensemble across the 3 lookbacks (both cadences) = 2; S2 band-ensemble across the 3 z-windows
  (both sampling grids, continuous sizing only, since threshold gates are themselves already a
  declared robustness axis) = 2; combined-book overlay (S1+S2+S3, see below) = 1.
- **N_trials = 18 + 144 + 12 + 2 + 2 + 1 = 179.** This is a deliberately large, fully-enumerated
  ledger (S2's sampling×window×rebalance×sizing grid dominates it) — DSR is charged the full 179,
  not a hand-picked subset, exactly as the methodology requires ("every cell you compute...must be
  counted"). A large N_trials makes the DSR bar harder to clear, which is the correct, conservative
  consequence of declaring an honest, complete grid rather than a curated one.

## Pre-named deployable (decided now, before any run — never "whichever cell looked best")

**The deployable is: S1 band-ensemble (equal-weight mean of the 3 lookback-L books {7,14,30}d,
WEEKLY cadence) + S3(c) overnight-execution-timing overlay on top of it, at the primary 13bp cost
scenario, vol-scaled legs, restricted for the paper-deployability report to the on_demo=true
subset (in addition to the full-110/full-102 research cut).**

Reasoning, declared now: S1 is the sleeve with the clearest a priori mechanism (retail long-bias
on a new venue → positive funding → carry), the most direct analog to an already-validated crypto
sleeve, and — critically — the only sleeve usable on perp-only names, which matters because several
of the highest-conviction "new venue" names (synthetic thematics, KR equities) have no underlying
at all and would be invisible to S2/S3. Weekly cadence is chosen over daily as the primary because
funding settles every 8h but meaningfully drifts on a slower clock, and weekly matches the already-
validated crypto carry sleeve's own cadence (comparability). S3(c) is layered on as an overlay,
not a competing standalone pick, because the task frames it that way explicitly and because it is
free to test once S1 exists. S2 is NOT part of the pre-named deployable: its 1-minute two-leg
requirement is the most novel part of the research (worth running in full) but also the noisiest
and shortest-history construction of the three, and a 48-cell family is exactly the kind of grid
that should be validated hard, not defaulted into the headline pick, before it is trusted.

If S1's own gates fail (see Adoption gate below), the deployable recommendation is NOT silently
swapped to S2 or S3 after the fact — a failed pre-named deployable is reported as the program's
headline result, and any positive S2/S3 cell is reported separately as "does not change the
pre-named deployable's outcome," per the frozen-grid discipline in METHODOLOGY.

## Regime sanity signature (MANDATORY pre-belief check, per sleeve)

- **S1**: median `last_funding_rate` across the full perp-era panel (pooled across names and
  periods) must be persistently POSITIVE for the retail-long-bias carry mechanism to have a story.
  A preliminary spot-check during prereg drafting (TSLAUSDT, 382 funding obs) showed a pooled mean
  of −0.0004bp with only 45.5% of observations positive — i.e. NOT unambiguously supporting the
  retail-long-bias prior on this one name. This is exactly why the signature must be checked on
  the FULL panel before any S1 result is believed: if the full-panel median comes back negative or
  near-zero, the report must say plainly that the carry mechanism has no support in this data,
  regardless of what the backtested Sharpe shows.
- **S2**: no separate regime signature beyond the general per-name basis-varies-by-name caveat
  (landmine 2) — the check here is a construction check, not a regime check: the z-scored book must
  show its P&L concentrated in genuine extreme-z episodes (visible mean-reversion after a threshold
  crossing), not smoothly steady, or the "reversion" story is not actually what is being captured.
- **S3**: the in-sample fact-check (a) — overnight return magnitude/quality exceeding intraday —
  MUST hold at the SPYUSDT/QQQUSDT index level before (b) or (c) are credited at all. If overnight
  does not exceed intraday at the index level, both S3(b) and S3(c) are reported as built on a
  foundation that failed its own precondition, and any positive number from them is treated as
  suspect (bug-hunt trigger) rather than a finding.

## Lookahead canary (mandatory, one per sleeve, excluded from real results/family/PBO)

For each of S1, S2, S3(b), run one deliberately-wrong version where the signal at time t is
computed using information dated AFTER t (the wrong-direction shift), scored against the same
forward return convention as the honest pipeline (wret[t] = return from t to t+1; a signal at t
uses info ≤ t only in the honest version — the canary breaks this on purpose). Each canary must
show a large, fake, obviously-too-good Sharpe; if any canary does NOT show this, the honest
pipeline for that sleeve is treated as suspect for the same bug and is re-audited before its
result is reported. Canary Sharpes are reported explicitly in E1_REPORT.md as the clean-pipeline
proof, and are excluded from every family/PBO/DSR calculation.

## Costs & validation

Costs: Binance USDT-M VIP0 taker 5bp per leg; primary round-trip 13bp per name-period traded for
S2/S3, with {6.5bp, 26bp} scenario variants (see N_trials ledger). S1 scenarios the fee only
({6.5bp, 13bp, 26bp}), never funding.

Validation battery (identical gate family to Programs 2-4, reused verbatim where the machinery
fits): CPCV (8 groups, k=2, embargo ≥3 weeks for weekly-cadence cells, embargo ≥5 trading days for
daily-cadence cells — declared explicitly per cadence since S1/S2/S3 mix daily and weekly) with
mean-path and frac_pos reported; CSCV-style PBO over the full 179-cell family; drop-top-5-periods
robustness; per-year Sharpes; and the full metrics table per cell (`metrics()` reused near-verbatim
from swing_pipeline.py: net Sharpe annualized per-observation, EV in bp per name-period, win %,
maxDD, CAGR raw and at a 10% vol-target overlay via `vol_target()`, turnover, Sharpe under each
declared cost scenario). DSR via swing_pipeline.py's `dsr()` at N_trials = 179 (the full ledger
above), never a hand-picked subset.

**Short-history honesty (mandatory, restated per METHODOLOGY)**: S1 and S2 are perp-era-only by
construction — N_eff will be small (weeks to low hundreds of periods per name, and the panel-level
pooled N_eff is bounded by the youngest perp's listing date for any pooled cross-sectional cell).
The report states plainly, per sleeve, whatever the real N_eff implies: if it is too short to
support annualization-based claims, the sleeve is labeled "short-history, directionally
suggestive, not yet backtest-confirmable at this length" rather than dressed up with an annualized
Sharpe that outruns its own sample size. S3 has more history where it uses the underlying's RTH
calendar over the full 2018-2026+ window (the perp side of S3's overnight leg is still perp-era-
only wherever it needs perp prices specifically — declared here: S3(a)'s fact-check and S3(b)'s
book both require the PERP price series to be traded through the overnight window, so S3 is ALSO
perp-era-length-bound in practice, not full-databento-length-bound; only a hypothetical underlying-
only version of the overnight/intraday split, which is not what task's decomposition asks for,
would get the long window).

**Honest target (stated before any run, per METHODOLOGY)**: 0.8-1.2 GROSS Sharpe per sleeve is a
believable new-uncrowded-venue outcome; above ~1.5 gross triggers a bug hunt (lookahead, cost
omission, survivorship leak) before anything is reported as a finding, documented either way.

**Demo-gate reporting (mandatory, landmine 5)**: every headline number in E1_REPORT.md is reported
twice — once on the full research universe (110/102/perp-history-only as appropriate per sleeve)
and once restricted to the on_demo=true subset (currently ~19/110 names) — since the demo subset
is the only one that could actually be paper-traded today without the venue adding more demo names.

**Survivorship (mandatory, landmine 4)**: every report section presenting a headline number states
plainly that the 110-name roster is today's Binance winners, not a point-in-time universe, in that
section — not deferred to a single caveats appendix.

## Combined book (declared for completeness, NOT the pre-named deployable)

A combined S1+S2+S3(b) risk-parity book (equal trailing-vol contribution across the three sleeves'
band-ensemble books, then 10% annualized vol-target overlay, scale capped [0.5, 2], reusing
`vol_target()`'s convention) is reported as a robustness/completeness cell — it is charged 1 trial
in the N_trials ledger above. It is explicitly NOT the pre-named deployable; per the frozen-grid
discipline, the deployable was named above before any run and does not change based on which
combination looks best after the fact.

## Adoption gate (frozen, pre-outcome)

Deploy the pre-named deployable (S1 weekly band-ensemble + S3(c) overlay, 13bp primary cost) to
DEMO PAPER TRADING (never live, per this program's PAPER-FIRST-BY-CONSTRUCTION framing) ONLY IF
ALL of the following hold simultaneously:
1. Net Sharpe positive on the full research universe AND on the on_demo=true subset reported
   separately (both shown, deployment judged on the on_demo subset since that is what could
   actually be paper-traded).
2. DSR pass at N_trials = 179.
3. PBO ≤ 0.25 over the full 179-cell family.
4. Drop-top-5-periods Sharpe still positive.
5. The S1 regime-sanity signature holds (full-panel median funding rate persistently positive) AND
   the S3(a) fact-check holds (overnight exceeds intraday at the SPYUSDT/QQQUSDT index level) —
   since S3(c) rides on S1, both signatures must hold, not just S1's own.

If ANY gate fails, the negative is recorded plainly in E1_REPORT.md as the program's conclusion for
that construction, and deployment is recommended AGAINST — a correctly-identified dead end here is
a complete, valuable result, not a failure to find one. Any individual sleeve (S1, S2, or S3) that
independently clears all five gates on its OWN construction, even if the combined pre-named
deployable does not, is reported as a secondary finding with the same full gate discipline applied
to it before it is called a finding — but it does NOT retroactively become "the deployable" unless
this document is revised and re-frozen first.
