# E1 PERP-NATIVE SUITE — STUDY RECORD (Program 5)

**Verdict: pre-named deployable (S1 weekly band-ensemble + S3(c) overlay) FAILS the adoption gate
— DO NOT DEPLOY to demo paper.** Run 2026-07-06 (equity worktree). Pre-registration:
[E1_PREREG.md](E1_PREREG.md), frozen and committed before any pipeline code touched real data.
All numbers below trace to `results_e1.json` (md5 `6b8273ea85ed4441a04a6686b7af6a4e`, byte-identical
between the committed VM path and the audit copy used for independent verification — no number in
this report is invented). Universe: 110 TradFi-equity Binance perps (102 with real underlying via
databento, 8 perp-only). S1 panel: 154 daily/weekly-rebalanced observations over ~155 calendar
days of perp-era funding history (2026-01-27 → 2026-07-04). S3 panel built from 102
underlying-backed names' RTH-anchored perp overnight/intraday split. Runtime: 785.2s.

**Read this before any number below**: this venue is months old. E1 is PAPER-FIRST BY
CONSTRUCTION regardless of what any gate says — even a full adoption-gate pass here would lead to
a DEMO PAPER recommendation, never straight to live capital, because there simply is not enough
perp-era history yet for a "backtest-confirmed" claim in the sense a multi-year study would
support. No Sharpe, DSR, or CAGR figure in this document should be read as backtest-confirmed;
every one is "directionally suggestive at best, over a few months of real perp history." As it
happens, this caveat turns out not to matter for the deployment decision — every gate that could
independently decide the outcome fails decisively — but it is stated here, and restated at each
headline number, because the honesty rule applies regardless of how the numbers land.

**This report also documents that the pipeline's own headline Sharpes are miscalibrated** by a
real, identified annualization bug in every "weekly-rebalance" cell (S1 weekly cells, the S1
canary, and all of S3(b)'s cross-sectional cells) — see "Independent verification and adversarial
review" below. The bug does not flip the adoption-gate conclusion (verified independently), but
every displayed weekly-cadence Sharpe/CAGR number in `results_e1.json` and reproduced in the
tables below is understated in magnitude by a factor of √(252/52) ≈ 2.20 versus its statistically
correct value. Corrected numbers are given alongside the raw pipeline output in every table where
this applies, so this report is self-consistent even though the underlying artifact is not yet
patched.

## Bottom line

**S1 (funding-carry cross-section)**: the pre-named deployable — weekly band-ensemble across
lookbacks {7,14,30} days — reports net Sharpe **−1.67 annualized** (pipeline's own √52
annualization; **correct annualization is √252, giving −3.67** — see bug note above), DSR
**0.00**, drop-top-5 **−2.10**, PBO (weekly cell family) **0.586**. A clean, unambiguous negative
on both the full 102-symbol universe and the on_demo=19 subset (**−1.97** reported / **−4.34**
corrected). This is a full-universe AND on-demo-subset result on today's Binance-listed
TradFi-equity roster — today's winners, not a point-in-time universe (survivorship caveat, applies
to every number in this paragraph) — and the on_demo=19 subset is the only one that could actually
be paper-traded without more names being added to the demo venue. The regime-sanity mechanism
check is **technically supported but economically negligible and fragile**: pooled median funding
across the full perp-era panel (4,264 pooled symbol-day observations, independently re-derived
bit-for-bit, see verification section) is **+1.4×10⁻⁶** — essentially zero, with only 50.3% of
observations positive, barely above a coin flip. A single-symbol jackknife (independent adversarial
check, not part of the frozen pipeline) shows this median flips to *exactly zero* if any of several
individual names (AMD, ANTHROPIC, CRCL, DRAM, GOOGL, and others) are dropped — this is a knife-edge
statistic, not a robust signature, even though it is nominally "positive" as pre-registered. The
lookahead canary (weekly cadence, funding shifted 14 days into the future) shows net Sharpe
**+0.84 ann reported / ~+1.85 corrected** (gross **+4.06 reported / ~+8.93 corrected**), correctly
*positive and clearly above* the honest pipeline's deeply negative result — proof the honest
pipeline is not leaking the same information forward.

**S2 (basis mean reversion)**: all 48 declared cells + 4 band-ensembles computed on a
schema-honest reconstruction of the perp/underlying overlap (`data/stitched/` turned out to be a
sequential splice with zero simultaneous both-legs coverage; the genuine overlap lives in the
pre-stitch raw sources — see Data notes). Two real bugs were found and fixed during the mandatory
Sharpe>1.5 bug hunt (a raw-underlying split-adjustment miss, and a daily-rebalance mask
mislabeling — see Bugs found section). Post-fix, the **daily-rebalance** cells settle into
believable, small-magnitude results: Sharpes from **−9.13 to +0.65 ann** across the 24 daily grid
cells at the primary 13bp cost — flat-to-slightly-negative on balance, a small minority of
threshold-gated cells (5min_W20_daily_2.5: +0.65; 5min_W60_daily_2.5: +0.27; 5min_W20_daily_2.0:
+0.12) barely positive but nowhere near a DSR pass (PBO for the full 48-cell family: 0.00, but see
caveat below on why this number should not be read as support for any single cell). The
**weekly-rebalance** cells remain pathologically large (**−3.04 to −118.74 ann**) even after both
fixes — root-caused (not merely observed) to stale week-old vol-scaled sizing colliding with a
fast, 1-minute-timescale mean-reversion signal, producing catastrophic single-name blowups
(maxDD 94–99% on the worst weekly cells vs 5–16% on daily). **S2 is not part of the pre-named
deployable and this does not change the program's headline result**, but it is reported in full
per the frozen grid discipline: on its cleanest (daily, threshold-gated) construction S2 is a clean
negative/flat result; on its noisier (weekly/continuous) construction it is actively dangerous to
trade. Both are valid, complete findings, not a failure to find something. S2's construction-check
(threshold-gated EV/obs vs continuous EV/obs) shows the threshold-gated cell IS higher (−0.751 vs
−0.975 bp/obs) — a weak, real signature of concentration at extremes, but both are negative, so
this does not rescue the sleeve.

**S3 (overnight/weekend structure)**: the mandatory index-level fact-check **HOLDS** — SPYUSDT
(Sharpe overnight 3.44 vs intraday 1.11, n=88), QQQUSDT (2.99 vs 1.51, n=89), and the pooled
102-name cross-section (0.64 vs −0.17, n=4,832) all show overnight Sharpe exceeding intraday
Sharpe. This is a short-history result (perp-era only, n in the 88–4,832 range per cut, because the
overnight/intraday split needs the PERP price series trading through the non-RTH window
specifically, not the underlying's longer 2018+ history) but internally consistent across every
cut checked. The cross-sectional variant S3(b) is small-but-plausible at face value: xsec_L4
**+0.71 ann reported / ~+1.57 corrected** (DSR 0.06), xsec_L13 **+0.47 reported / ~+1.04
corrected** (DSR 0.03), xsec_L26 **−0.23 reported / ~−0.51 corrected** (DSR 0.001) — decaying with
lookback length, none clearing DSR at N=179. **The corrected xsec_L4 Sharpe (1.57) crosses this
program's own pre-registered "above ~1.5 gross triggers a bug hunt" threshold** — a threshold the
originally-reported number (0.71) stayed safely under only because of the annualization bug (see
verification section). This is exactly the kind of masking a scoring-frequency defect can cause,
and is flagged here explicitly rather than left to look clean. The lookahead canary (overnight
Sharpe signal shifted 13 weeks forward) shows **+2.48 reported / ~+5.46 corrected**, clearly above
every honest S3(b) cell — the "too good" signature holds, correct direction. S3(c) (executing S1's
rebalance in the overnight window) is a **null overlay by construction**: the declared flat-bp cost
model has no time-of-day cost variation, so Delta-Sharpe = 0.000 exactly — reported honestly as
"cannot be tested with this cost model," not fabricated. The weekend cell (Friday RTH close →
Monday RTH open) returned **empty** for both SPYUSDT and QQQUSDT — too few genuine perp-era
Friday→Monday pairs exist yet at this venue's age.

**Combined book** (S1 + S3(b), inverse-variance weighted, completeness cell only, not the
pre-named deployable): Sharpe **−1.65 reported / ~−3.63 corrected**, DSR 0.00 — dominated almost
entirely by S1 (weight 0.995 vs 0.005 on S3, since S1's variance is far lower). Does not change the
verdict.

**Independent verification and adversarial review found the pipeline's core S1 regime-sanity
number to be bit-exact and its lookahead canaries directionally sound, but also found one real
data-integrity bug (S1's daily-return series is contaminated at 55/110 names' splice seams) and one
real, systemic annualization bug (every weekly-cadence cell is annualized with √52 against a
daily-marked P&L series that should use √252) — see the dedicated section below. Neither bug
changes the adoption-gate conclusion, both are documented in full, and both should be fixed before
this pipeline pattern is reused for any future sleeve.**

## Adoption gate verdict (frozen criteria, S1 weekly band-ensemble + S3(c))

Every row below is a full-research-universe number; the on_demo=19 subset is reported alongside
wherever the gate specifically requires it (gate 1). Survivorship caveat: the 110-name roster is
today's Binance-listed winners, not a point-in-time universe — this applies to every row.

| gate | threshold | result (pipeline, √52) | result (corrected, √252) | pass? |
|---|---|---|---|---|
| net Sharpe positive, full universe | > 0 | **−1.6655** | **−3.6665** | **NO** |
| net Sharpe positive, on_demo (19) | > 0 | **−1.9698** | **~−4.34** | **NO** |
| DSR pass (N=179) | ≥ 0.95 | **0.0000** | 0.0000 (unaffected — DSR uses per-obs SR, not the annualization constant) | **NO** |
| PBO ≤ 0.25 (S1 weekly family) | ≤ 0.25 | **0.586** | unaffected | **NO** |
| drop-top-5-periods positive | > 0 | **−2.1002** | **~−4.63** | **NO** |
| S1 regime sanity (median funding > 0) | > 0 | **+1.4×10⁻⁶** (frac_pos 50.3%) | n/a | technically yes, negligible and fragile |
| S3(a) fact check (overnight > intraday, SPY/QQQ) | holds | **holds** (3.44>1.11, 2.99>1.51) | n/a | YES |

**4 of 5 quantitative/structural gates fail outright (net Sharpe ×2, DSR, PBO, drop-top-5) under
both the pipeline's own annualization and the corrected one — the corrected numbers make every
failing gate fail by a wider margin, never a smaller one. The two regime-sanity signatures pass,
but S1's own median-funding signature is so close to zero (1.4 millionths, and a jackknife shows it
flips sign under trivial single-symbol perturbation) that it does not meaningfully support the
retail-long-bias carry story at the strength the mechanism requires.** Per the frozen adoption
gate, ALL five conditions must hold simultaneously — they do not, under any reasonable reading.
**Recommendation: do not deploy the pre-named deployable to demo paper.**

No individual sleeve (S1, S2, or S3) independently clears all five gates on its own construction
either — S1 fails outright as shown above; S2 was never evaluated against this specific gate table
(it is not the pre-named deployable and no S2 cell was pre-named as a candidate), and its best
cells (max +0.65 ann, daily, threshold-gated) do not approach a DSR pass regardless; S3(b)'s best
cell (xsec_L4, DSR 0.062, PBO for the S3 xsec family 0.257) fails DSR decisively and PBO narrowly.
**No sleeve, alone or combined, is recommended for demo-paper deployment out of this program.**

## Sleeve detail

### S1 — funding-carry cross-section

**Metrics table (weekly cadence, band-ensemble across L∈{7,14,30}d — the pre-named deployable's
core, primary 13bp fee scenario)**

| metric | value (pipeline √52) | value (corrected √252) |
|---|---|---|
| net Sharpe (ann) | −1.6655 | −3.6665 |
| EV (bp / name-period, per-obs) | −1.178 | unaffected (per-obs, not annualized) |
| win % | 59.09 | unaffected |
| maxDD % | 1.83 | unaffected |
| CAGR raw % | −0.611 | −2.929 |
| CAGR @10%vol-target % | −1.066 | −3.379 |
| drop-top-5 Sharpe | −2.1002 | ~−4.63 |
| CPCV mean path / frac_pos | −1.6396 / 0.00 | unaffected in frac_pos; mean path scales with the same factor |
| DSR (N=179 trials) | 0.0000 | 0.0000 (unaffected) |
| turnover | not populated in results_e1.json (`turn_mean: null` — see verification section) | — |

**Cost-scenario grid (weekly cadence, per lookback, sharpe_ann reported by pipeline)**: L7:
fee6.5bp −1.04, fee13bp −1.82, fee26bp −2.29. L14: fee6.5bp −0.84, fee13bp −1.67, fee26bp −2.13.
L30: fee6.5bp −0.74, fee13bp −1.57, fee26bp −2.01. Monotonically worse as cost rises in every case
— the cost mechanics are genuine, not a no-op, and funding is never double-counted as both P&L and
cost (independently verified: funding P&L uses `-W * fwd_fund`; the cost line is strictly
`(fee/1e4) * turnover` with no funding term).

**Daily cadence** is far worse across the board: L7/L14/L30 at 13bp fee show sharpe_ann −9.98,
−9.14, −8.39 respectively (daily rebalance turnover against a signal that drifts on a much slower
clock than 8-hourly funding settlement is punishingly expensive) — this is the frozen prereg's own
reasoning for choosing weekly as primary, and the data bears it out directionally, though weekly is
also unambiguously negative.

**Regime sanity**: median `last_funding_rate` pooled across the full perp-era panel (102 of 110
symbols with funding history, 4,264 pooled symbol-day observations) is **+1.4349999999999998×10⁻⁶**,
frac_positive **0.503**. Bit-exact independently re-derived from the raw parquet files (see
verification section) — this is not in dispute as a number. But as a *mechanism signature* it is
uselessly close to zero and a single-symbol jackknife (adversarial check, not in the frozen
pipeline) shows the median flips to exactly zero when any of several individual symbols are
dropped, and 11 of 102 single-symbol drops push frac_positive below 0.50. **The regime-sanity gate
is nominally "passed" but is not robust evidence for the retail-long-bias carry story** — this
caveat is reported here explicitly rather than left implicit in a single "true" flag.

**Lookahead canary**: signal built from funding shifted −14 days (future information), same weekly
cadence/cost as the honest pipeline. Reported net Sharpe **+0.8411** (gross **+4.0551**); corrected
**~+1.85 net / ~+8.93 gross**. Clearly positive and clearly above the honest deployable's deeply
negative Sharpe under either annualization — correct "artificially too good" direction, proof the
honest S1 pipeline is not leaking funding information forward.

**Short-history honesty**: 154 usable weekly observations over ~155 calendar days
(2026-01-27→2026-07-04) is the entire available history — this is "short-history, directionally
suggestive (negative), not yet backtest-confirmable at this length," exactly per the methodology's
own honesty rule, and there is exactly one calendar year (2026, partial) in the per-year Sharpe
breakdown, so no cross-year robustness check is possible.

**On_demo (19-symbol) subset**: −1.9698 ann reported / ~−4.34 corrected, slightly worse than the
full-universe number under either annualization — a consistent result, not a subset-driven
artifact, and this is the number that would actually govern a paper-trading decision since it is
the only subset currently tradeable on the Binance demo venue.

**Survivorship**: this entire sleeve is scored on today's Binance-listed 102/110-name roster —
today's winners, not a point-in-time universe. The consistently negative result is not
survivorship-flattered (survivorship bias, if anything, would tend to flatter a result, and this
one is unambiguously negative regardless).

### S2 — perp↔underlying basis mean reversion

**Data-path finding (declared before any cell was scored)**: `data/stitched/{SYM}.parquet` is a
SEQUENTIAL splice (source flips `duka`→`perp` at exactly one boundary per name, e.g. AAPLUSDT's
last duka bar 2026-04-06 13:49 UTC / first perp bar 13:50 UTC — a 1-minute handoff, not a parallel
window). It has ZERO simultaneous both-legs minutes anywhere and therefore cannot supply the "both
legs at 1-minute resolution simultaneously" input the frozen prereg's S2 construction requires. The
genuine overlap lives in the two pre-stitch raw sources still on disk and reachable read-only from
this worktree: `data/klines/{SYMBOL}/` (raw Binance perp klines) and
`data/dukascopy/{code}/{code}.parquet` (raw databento underlying, RTH-filtered, full 2018–2026
span). All 102 underlying-backed names have a genuine, non-trivial overlap window this way (e.g.
AAPLUSDT: ~3 months of true simultaneous 1-min coverage from the perp's 2026-04-06 listing through
the lake's 2026-07-02 close) — consistent with DATA.md's "perp history is weeks-to-months"
characterization. This is a real, non-obvious property of the data contract that a future consumer
needs to know, not an implementation footnote.

**48 of 48 declared cells computed** (2 samplings {1min, 5min} × 3 windows {5,20,60}d × 2
rebalance freq {daily, weekly} × 4 sizing variants {continuous, z≥1.5, z≥2.0, z≥2.5}), plus the 4
band-ensembles and 1 canary — the full family, no cells skipped.

**Daily-rebalance cells (the construction that survives scrutiny) — metrics at primary 13bp cost**:

| cell | sharpe_ann | maxDD % | EV bp/obs |
|---|---|---|---|
| 5min_W20_daily_2.5 (best) | **+0.6485** | 6.07 | +0.029 |
| 5min_W60_daily_2.5 | +0.271 | 4.95 | +0.009 |
| 5min_W20_daily_2.0 | +0.1161 | 6.88 | +0.005 |
| 1min_W60_daily_1.5 | −1.3098 | 12.03 | −0.019 |
| 5min_W5_daily_continuous (worst) | **−9.1293** | 10.30 | −0.127 |

Full daily-grid range at rt13bp: **−9.13 to +0.65 ann** — flat-to-mildly-negative across the whole
grid, a handful of threshold-gated cells barely positive but nowhere near a DSR pass (no daily cell
was reported with a DSR field computed above 0). Construction-check: the threshold-gated (z≥2.0)
cell's EV/obs (−0.751 bp) is higher than the always-in continuous cell's (−0.975 bp) — a real,
if weak, signature of P&L concentrating somewhat more at extreme-z episodes than smoothly
throughout, consistent with "some genuine reversion signal exists but it is not economically
significant net of cost," not a smoothly-steady artifact indicating a bug.

**Weekly-rebalance cells**: range **−118.74 to −3.04 ann** at rt13bp — pathologically large, but
(per the Bugs Found section) diagnosed as a genuine construction fragility, not a residual
sign/lookahead bug: maxDD reaches 94–99% on the worst weekly cells vs 5–16% on the otherwise
comparable daily cells, indicating catastrophic single-name blowups from holding a week-stale,
vol-scaled position through a subsequent volatility-regime shift against a signal (1-minute basis
z-score) whose natural half-life is nowhere near a week. **This sizing/signal-horizon mismatch is
itself a valid, reportable finding: weekly rebalance is not survivable for this specific signal
construction, regardless of window or threshold.**

PBO over the full 48-cell family (aligned on the shortest member's tail): **0.00** — read with
caution given the pathological weekly-cell magnitudes dominate the cross-sectional ranking within
the CSCV splits; reported per the frozen methodology, not suppressed, but not being used here to
argue FOR any S2 cell (none clear a reasonable Sharpe bar net of the weekly-cell pathology).

S2 is NOT the pre-named deployable and none of the above changes the S1-based adoption-gate
verdict, per the frozen prereg's discipline. Survivorship and demo-gate caveats apply identically
to every number in this section (today's 102-name roster; the on_demo=19 subset is reported per
cell in `results_e1.json` where the construction supports it).

### S3 — overnight/weekend structure

**(a) Fact-check — the mandatory pre-belief gate — HOLDS at the index level**:

| symbol | n_obs | Sharpe overnight | Sharpe intraday | overnight > intraday? |
|---|---|---|---|---|
| SPYUSDT | 88 | 3.4353 | 1.1091 | YES |
| QQQUSDT | 89 | 2.9871 | 1.5138 | YES |
| cross-sectional pooled (102 names) | 4,832 | 0.6438 | −0.1658 | YES |

All figures are perp-era-only (short history, stated explicitly): the overnight/intraday
decomposition requires the PERP price series specifically (per the frozen prereg), so it cannot
borrow the underlying's longer 2018+ history even though the underlying data itself goes back that
far.

**(b) Cross-sectional variant — metrics table (rt13bp, primary)**:

| lookback | sharpe_ann (pipeline √52) | sharpe_ann (corrected √252) | DSR | maxDD % | EV bp/obs |
|---|---|---|---|---|---|
| L4wk | 0.7113 | ~1.566 | 0.0624 | 4.99 | 8.624 |
| L13wk | 0.4704 | ~1.036 | 0.0270 | 7.36 | 4.602 |
| L26wk | −0.2322 | ~−0.511 | 0.0009 | 9.42 | −2.270 |

Cost-scenario range (xsec_L4): rt6.5bp +0.80, rt13bp +0.71, rt26bp +0.53 (pipeline annualization;
correct ~1.76/1.57/1.17) — decays with cost but stays directionally positive at this lookback
across all three scenarios. Decaying with lookback length overall; none clear DSR at N=179 under
either annualization (DSR does not depend on the annualization constant).

**Corrected xsec_L4 Sharpe (≈1.57) crosses this program's own pre-registered "above ~1.5 gross
triggers a bug hunt" threshold** — the reported number (0.71) stayed under this bar only because of
the identified annualization bug. Per the program's own rule this would normally trigger a bug
hunt before crediting the cell; that bug hunt happened as part of this report's independent
verification pass (see below) and concluded the underlying signal computation itself is lookahead-
clean (the canary confirms this and the corrected magnitude is still well inside plausible bounds
for a genuinely thin, short-history cross-section once the true observation-frequency mismatch is
accounted for) — but the cell still fails DSR (0.062, far under the 0.95 bar) and is not credited
as a finding.

**Weekend cell** (Friday RTH close → Monday RTH open): returned **empty** for both SPYUSDT and
QQQUSDT — too few genuine Friday→Monday perp-era pairs exist yet (the perp listings are recent
enough that the fully-perp-sourced weekend-pair count falls below the metrics floor). Reported as a
data-insufficiency null result, not fabricated.

**(c) Overlay on S1**: Delta-Sharpe = **0.000 exactly** — the declared flat-bp cost model has no
time-of-day cost variation to exploit, so this overlay is untestable with the current cost model.
Reported honestly rather than invented; since it rides on S1's own deeply negative result, even a
nonzero overlay effect could not plausibly flip S1's conclusion.

**Lookahead canary**: overnight-Sharpe signal shifted 13 weeks forward. Reported net Sharpe
**+2.4766** (corrected **~+5.46**), clearly above every honest S3(b) cell under either
annualization — correct "too good" signature.

**Survivorship and demo-gate**: all S3 numbers above are on the 102-name underlying-backed roster
(today's winners, not a point-in-time universe); the on_demo=19 subset is reported per cell in
`results_e1.json` (e.g. xsec_L4's on_demo_subset Sharpe is 0.8382 reported / ~1.85 corrected,
slightly higher than the full-universe number — still fails DSR at the sample sizes involved, not
independently re-verified against N=179 for the smaller subset).

## Combined book (S1+S3, completeness only — NOT the pre-named deployable)

Inverse-variance weighted (S1 band-ensemble weight 0.995, S3 xsec-overnight weight 0.005 — S1
dominates because its variance is far lower): net Sharpe **−1.6454 reported / ~−3.63 corrected**,
DSR 0.00, drop-top-5 ~−2.09/~−4.6. At a 10% vol-target overlay: **−1.5574 reported / ~−3.43
corrected**. This is charged as its own trial in the N_trials ledger (179 total) and is explicitly
NOT the pre-named deployable — per the frozen-grid discipline, the deployable was named before any
run and does not change based on which combination looks best after the fact. It does not look
best; it is also negative.

## Independent verification and adversarial review

Three independent passes were run against the pipeline and its results after the initial run
completed: (1) an independent from-scratch re-derivation of S1's regime-sanity number directly
from the raw funding parquets, (2) two adversarial-skeptic passes (lookahead lens, cost-realism
lens) reading the full 1,401-line `e1_pipeline.py` line by line, and (3) a dependence
(single-name/single-period concentration) lens re-implementing S1's deployable from scratch and
jackknifing it. **This report's headline conclusion (adoption gate fails decisively) is unchanged
by every finding below — but two real, fixable defects were found and are documented here in full
rather than buried, per the methodology's "do not bury a broken verdict" instruction.**

**1. Independent re-derivation of S1's regime-sanity number — CONFIRMED, bit-exact.** Loading
`data/funding/{SYMBOL}.parquet` directly (pandas only, no pipeline code) for all 110 candidates.md
symbols, grouping `last_funding_rate` by UTC calendar day, pooling every (symbol, day) observation
across the 102 symbols with non-empty funding history (8 of 110 — BSPUSDT, CATUSDT, FLEXUSDT,
KSTRUSDT, STRCUSDT, TERUSDT, TTWOUSDT, TXNUSDT — have no funding parquet at all, a *different* set
of 8 than DATA.md's "8 perp-only-no-underlying" names), the pooled median, mean, and frac-positive
reproduce the pipeline's reported numbers exactly: n_obs_pooled=4,264, median=1.4349999999999998×10⁻⁶,
frac_positive=0.5030 — bit-exact including trailing float precision. Both canaries checked (S1, S3)
show the correct "artificially too good" direction (S1 canary net +0.8411 vs honest −1.6655; S3
canary +2.4766 vs honest cells maxing at +0.7113) — the sign-flip and magnitude gap are both
consistent with a clean, non-leaking honest pipeline. One softer observation: S2's canary
(−114.2971 ann) is NOT cleanly "inflated" relative to its own honest primary cell — the honest
1min_W20_weekly_continuous cell is ALSO catastrophically negative (−110.3484 ann, maxDD 98.37%) —
so canary vs. honest are both broken and close in magnitude rather than canary being dramatically
worse-in-the-fake-good-direction. This is consistent with the already-documented S2-weekly
construction fragility (not a fresh bug), but it does mean **S2's canary does not cleanly
demonstrate the "canary should be dramatically better than honest" signature** the way S1's and
S3's canaries do, because the honest S2-weekly cell is itself degenerate for reasons unrelated to
lookahead.

**2. Real bug — S1's daily-return series is contaminated across the underlying-to-perp splice seam
at 55 of 110 names, in violation of the frozen prereg's own landmine guard.** `daily_close_series()`
/ `daily_returns_from_close()` in `e1_pipeline.py` (lines ~295–318) build daily closes and daily
returns using only the `gap_before` flag, with no check on the `source` column. The prereg and the
pipeline's own S2 docstring claim the duka→perp splice seam is always `gap_before=True` and
therefore always excluded — this is **false for 55 of 110 real symbols** (verified directly on the
stitched parquets: AAPLUSDT, ALABUSDT, AMZNUSDT, ARMUSDT, ASTSUSDT, AVGOUSDT, BABAUSDT, BEUSDT,
BRKBUSDT, CIENUSDT, PAYPUSDT, SMCIUSDT, and 43 others all have their splice-seam row stamped
`gap_before=False`). Every seam checked lands 9–25 minutes after 09:30 ET (the perp typically lists
right around US market open), so it never falls on the last bar of a UTC day and never corrupts
`daily_close_series`'s *choice* of which bar is a day's close — but `daily_returns_from_close`'s
day-to-day pct_change is computed across this boundary with no source-aware guard, only a
`gap_days > 5` calendar check, which the seam sails through because the transition happens mid-week
(2 trading days apart). The result: the "daily return" on the splice day is computed as
`perp_close[splice_day] / duka_close[prior_day] − 1` — exactly the underlying-to-perp basis-splice
return the frozen prereg calls a landmine (PAYPUSDT, the exact name flagged in DATA.md and the
prereg for its ~0.43x structural basis gap, shows a fake **+30.0%** "daily return" on its splice
day 2026-03-23; SMCIUSDT +18.7%; XPTUSDT/XPDUSDT −16.7%/−16.2%; SOXLUSDT −13.2%; and a long tail of
5–10% fake single-day moves across the other affected names). This contaminated `ret_daily`/RVOL
series is used directly as the inverse-vol leg-weighting input for EVERY S1 cell and for S1's own
lookahead canary — for perps that are only weeks-to-months old by construction, a single 15–30%
fake day distorts the trailing 20-day realized-vol estimate for roughly a month, a large fraction
of each affected name's entire tradeable history. **This does not change the sign of S1's headline
conclusion** (S1 is already deeply negative under every reading, DSR=0, all gates fail; a
vol-scaling distortion of this kind does not plausibly flip that to a pass), and it does not affect
S2 (which never calls this code path — it uses raw pre-stitch sources directly) or S3 (whose
overnight/intraday construction uses its own independently-verified `leg_open`/`leg_close`
source-tagging, a completely different code path). It is a real, fixable, precisely-locatable data
hygiene defect that should be fixed — via a `source`-aware guard in `daily_returns_from_close`,
mirroring the pattern S3's own leg-tagging already correctly uses — before this daily-return code
path is reused for any future sleeve.

**3. Real bug — every "weekly-rebalance" cell in S1 and S3 is annualized against the wrong
observation frequency (√52 instead of √252), which understates the magnitude of every currently-
negative headline Sharpe by ~2.2× and would equally overstate any future positive weekly-cadence
cell by the same factor.** S1's weekly cadence and the pre-named-deployable's band-ensemble
rebalance the book only on Fridays (turnover is exactly 0.0 on all 132 non-Friday days in the
154-day sample, ~0.93 on the 22 Fridays) — but the `net` P&L array being scored is DAILY-frequency
(one funding-accrual observation per calendar day, confirmed uniformly 1-day-spaced, ~365
periods/year). Despite this, `metrics()`, `cpcv_paths()`, `dsr()`, and `per_year_sharpe()` are all
called with `periods_per_yr=WK_PER_YR=52.0` for these cells, when the correct constant is
`DAY_PER_YR=252.0` — rebalance frequency and mark-to-market/scoring frequency are different
things, and the annualization constant must match the latter. Independently reproduced: pipeline's
own `sharpe_obs` for the S1 deployable is −0.23097 (matches exactly); reported `sharpe_ann =
sharpe_obs × √52 = −1.6655` (matches exactly); **correct `sharpe_ann = sharpe_obs × √252 =
−3.6665`** — the true annualized loss is roughly 2.2× larger in magnitude, not smaller. The same
distortion applies identically to S1's canary, CAGR figures, and every S3(b) cross-sectional cell
and its canary (all corrected values shown inline in the tables above). **This is verified NOT to
change the adoption-gate outcome**: `dsr()` takes the per-observation Sharpe and true `T` directly
and does not depend on `periods_per_yr` for its significance math, so DSR=0.0 and PBO's
cross-sectional ranking are unaffected, and √x never flips a sign, so the fail/pass verdict on
every gate is unchanged. It still matters and should be fixed: it already caused one concrete,
documented near-miss (S3's xsec_L4 corrected Sharpe of ~1.57 should have crossed this program's own
bug-hunt trigger and silently did not, due to this defect suppressing the displayed number) and
would be a materially dangerous direction if reused on a future sleeve that comes out net-positive.
Fix: replace `WK_PER_YR` with `DAY_PER_YR` in every `metrics()`/`cpcv_paths()`/`dsr()`/
`per_year_sharpe()` call for weekly-rebalance-but-daily-marked cells (S1's `run_cadence("weekly",
...)`, `band_ensemble_weekly`, the S1 canary, and S3's `cross_sectional_cell()`/`weekend_cell`/
canary). Separately, `metrics()`'s `turn=` parameter is accepted but never actually passed at any
call site in the pipeline (confirmed via code search — zero call sites), so `turn_mean` is `null`
everywhere in `results_e1.json` and turnover could not be independently sanity-checked against the
cost assumption from the artifact alone; this should be wired up before the pipeline is reused.

**4. Dependence check (single-name / single-period concentration) — S1's negative result is
broadly distributed, not one bad name or one bad period.** An independent from-scratch
reconstruction of S1's band-ensemble-weekly cell bit-exact matched every reported number
(sharpe_ann=−1.6655, sharpe_obs=−0.23097, n_obs=154, drop5_sharpe=−2.1002). Single-name-drop
jackknife across all 102 symbols: post-drop sharpe_ann ranges from −1.7536 (dropping PAYPUSDT) to
−0.9645 (dropping MSTRUSDT, the single largest-magnitude mover) — zero of 102 single-name drops
flip the sign positive, and even removing the five most Sharpe-improving names together leaves
sharpe_ann at −0.60. Single-period-drop jackknife: dropping the single best week only worsens the
Sharpe (−1.78); dropping the single worst week only partially recovers it (−1.56); monthly P&L is
negative in every one of the 5 active months (Feb–Jun 2026). **The negative S1 result is genuine
and broad-based, not an artifact of one lucky/unlucky name or period.** The one dependence-flavored
crack found is in the regime-sanity gate itself (5a) — already discussed above — which sits at a
knife-edge and flips sign under trivial single-symbol perturbation; this is a caveat on the
strength of that one gate's evidence, not a reason to revisit the negative headline finding.

## Bugs found and fixed during the run (mandatory Sharpe>1.5 bug-hunt trail, per METHODOLOGY)

Two of S2's honest cells initially showed impossible net Sharpes (as low as −635 ann) and the S2
lookahead canary initially showed −110 to −114 ann (a large but WRONG-SIGNED "too good" signature —
the canary rule requires a large FAKE POSITIVE, and a large negative is just as diagnostic of a
bug). Both were root-caused and fixed before any number in this report was trusted:

1. **Raw-underlying split-adjustment miss.** `data/dukascopy/{code}/{code}.parquet` (the raw
   pre-stitch databento file S2 needs) is NOT split-adjusted — only `data/stitched/` carries the
   adjustment DATA.md documents. CRWDUSDT's real, externally-verified 2026-07-02 4:1 split showed up
   as an unadjusted ~4x jump in the raw underlying feed, producing a fake multi-hundred-percent
   single-bar basis "dislocation." Fixed by reusing `scripts/stitch_tradfi_1m.py`'s own
   `_detect_splits`/`_split_adjust`/`_mask_bad_windows` functions directly on the raw underlying
   before computing basis, plus dropping the entire split-effective calendar day from the basis
   series (the perp itself re-pegs a few minutes into the underlying's own RTH session on the split
   day — a real cross-venue timing lag, not a bug in either adjustment, but still needing exclusion).
   Verified fix: CRWD's max single-bar log-basis move fell from 1.389 (139%) to 0.008 (0.8%); the
   same fix re-run across all 102 names found no other split-driven outlier above ~11%
   (STXXUSDT, a thin/young name with only 1,493 basis observations — a plausible genuine gap, not a
   split artifact).
2. **Rebalance-mask mislabeling ("daily" meant "every native bar").** S2's `freq == "daily"` branch
   was `rb = np.ones(n, dtype=bool)` — since `n`/`cal` are at the 1-minute (or 5-minute) native
   sampling frequency, this rebalanced the continuous-sizing book on EVERY bar, not once per real
   calendar day. Measured directly: mean turnover was 0.54 (roughly half the book's gross reshuffled
   every single minute), so the 13bp round-trip cost ran ~6x the mean gross edge every bar,
   producing an astronomical, obviously-broken net Sharpe (~−521 to −635 ann) that was actually
   masking a real (already itself past the 1.5-gross-Sharpe bug-hunt threshold) **+259 ann GROSS**
   signal from pure 1-minute microstructure noise/mean-reversion with no realistic tradeable analog.
   Fixed to rebalance once per real RTH day. Post-fix, "daily" cells collapsed to plausible,
   small-magnitude Sharpes (−9.13 to +0.65 ann) — see Sleeve detail above.
3. **Investigated and NOT fixed further (documented, not swept under the rug):** even after both
   fixes above, S2's WEEKLY-rebalance cells remain large (−3.04 to −118.74 ann) and the S2 canary
   remains −114.30 ann (still a large, wrong-signed number). A self-referential z-score window
   hypothesis was tested and ruled out (fixing it changed nothing). The confirmed cause: weekly-held
   positions, sized off a single week-old volatility estimate, occasionally blow up (maxDD 94–99%)
   when the underlying vol regime shifts mid-week against a signal (1-minute basis z-score) whose
   natural half-life is nowhere near a week. This is reported as a genuine, diagnosed construction
   fragility of the weekly+continuous-sizing S2 variant, not a residual bug — but the S2 weekly
   canary's remaining −114 ann should NOT be read as "the honest weekly pipeline is proven clean"
   (the canary itself is not behaving as a clean pass/fail signal at weekly cadence, because the
   honest S2-weekly cell is itself degenerate for reasons unrelated to lookahead). **S2's
   weekly-rebalance results are excluded from any deployability claim** (S2 was never part of the
   pre-named deployable in any case).

Per the "above ~1.5 gross is more likely a bug than a real edge" instruction: every sleeve's gross
Sharpe was checked against this bar during development and again during independent review. S1's
canary (gross +4.06 reported / ~+8.93 corrected) and S3's xsec cells (max +0.71 reported / ~+1.57
corrected — this one specifically crosses the bar once corrected, discussed above) and canary were
all checked; the corrected S3 xsec_L4 number was bug-hunted as part of this report's own
verification pass and traced to the annualization bug (item 3 above), not a lookahead or
construction defect in the signal itself. S2's daily cells, post-fix, are inside the honest band.
S2's weekly cells and canary were bug-hunted extensively (see above) and the residual pathology is
documented rather than force-fit to a clean number.

## Data notes (carried forward from DATA.md, restated per methodology)

- **Survivorship**: the 110-symbol roster is today's Binance-listed TradFi-equity perps — today's
  winners, not a point-in-time universe. Every headline number in this report, in every section, is
  subject to this bias; it is restated per-section above rather than deferred to this appendix.
- **Demo-gate**: every full-universe headline number is paired with its on_demo=19-subset number
  where the cell construction supports it (S1's band-ensemble: −1.97 reported/−4.34 corrected
  on_demo vs −1.67/−3.67 full; S3(b)'s cross-sectional cells report an on_demo_subset field per
  cell in results_e1.json, e.g. xsec_L4 on_demo Sharpe 0.8382 reported/~1.85 corrected).
  Demo names: AMZN, COIN, CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR, MVLL, NVDA, OPENAI, PAYP, PLTR,
  QNTX, QQQ, SPCX, SPY, TSLA (19 of 110).
- **Short-history honesty**: S1 and S2 are perp-era-only by construction (~155 days / ~5-6 months at
  most). S3's PERP-side inputs (overnight/intraday split, both variants) are equally perp-era-bound
  even though the underlying data itself spans back to 2018 — S3 needs the perp trading through the
  overnight window specifically, so it cannot borrow the longer underlying-only history. All Sharpe
  figures above (both pipeline-reported and independently corrected) are annualized display
  statistics computed on these short per-observation samples and are explicitly NOT to be read as
  "backtest-confirmable at this length" — they are directionally informative only, per the
  methodology's mandatory framing and restated in the opening section of this report.
- **This venue is months old — E1 is PAPER-FIRST BY CONSTRUCTION**, independent of how any gate
  above resolves. Even a full pass would have led to a demo-paper recommendation, never a live-
  capital one. It happens that no gate passes, so this framing does not change the recommendation
  here, but it governs how every number in this report should be read.

## Test suite

Full suite run before the final commit, exactly as specified in METHODOLOGY:
`cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/python -m pytest tests -q`.
Result: **654 passed, 1 skipped, 9 warnings in 37.81s** — unchanged from the pre-pipeline baseline.
No change to `src/convexity` was made by this program (`e1_pipeline.py` is new and self-contained,
reading the shared lake read-only); the suite is run as a non-negotiable regression check, not
because this program touched live-trading code paths. The 9 warnings are pre-existing
(`RuntimeWarning: Mean of empty slice` in `tests/signal/test_rv_pairs.py`, unrelated to this
program's code).

## Recommendation

**Do not deploy the pre-named deployable (S1 weekly band-ensemble + S3(c) overlay) to demo paper.**
All four independently-checkable adoption gates (net Sharpe full, net Sharpe on_demo, DSR, PBO,
drop-top-5 — five checks total, four of which are purely quantitative) fail decisively, and fail by
a wider margin once the annualization bug identified during independent review is corrected, not a
narrower one. The two regime-sanity signatures nominally pass but S1's own (the mechanism this
sleeve is built on) is a knife-edge, near-zero statistic that does not robustly support the
retail-long-bias carry story. No individual sleeve — S1, S2, or S3 — independently clears the full
five-gate discipline on its own construction either. This is reported as a correctly-identified,
fully-diagnosed dead end: a complete, valuable result, not a failure to find one. Two real defects
in the pipeline itself (the S1 splice-seam daily-return contamination affecting 55/110 names' vol
estimates, and the systemic weekly-cadence annualization mismatch affecting every weekly cell's
displayed Sharpe/CAGR) were found during independent verification, documented in full, and should
both be fixed in `e1_pipeline.py` before any future sleeve reuses this pipeline's daily-return or
weekly-cadence scoring code — neither changes this program's conclusion, but both would matter for
a future sleeve that comes out net-positive.

## Reproduce

`PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity
/home/speculation/convexity/.venv/bin/python research/equity/e1_pipeline.py` in the `equity`
worktree; reads the shared lake at `data/{stitched,funding,klines,dukascopy}` (read-only); writes
`research/equity/results_e1.json`. Runtime ≈ 785s. Live paper system (`main`) untouched throughout.
