# STATE — Program 3 DURABLE COMMODITY BOOK — ★ COMPLETE 2026-07-05 ★. Loop fdd1649a: CANCEL.

## FINAL VERDICT: DEAD net-of-cost, NO DEPLOY. Report delivered in chat + DURABLE_REPORT.md committed.
Combined book Sharpe 0.092 (0.045 vol-targeted), DSR 0.079, PBO 0.543, drop-top-5 NEGATIVE, 4/5
adoption gates FAIL. Thesis direction confirmed (slow trend L52 +0.149 & carry-on-storables +0.139
are the only positive cells; KRT −0.78 destructive) but magnitude sub-noise. Root cause = breadth
(7 names, eff ~4, metals-heavy) — same wall as Program 2 xsec, now confirmed for time-series.
Engine validated (2022 carry +0.79 / gate +1.31 paid the backwardation → return series & signs
correct; pure-trend 2022 weakness is real regime not bug). CLOSES commodity-directional book.
Nothing further to do; loop should self-cancel. If re-fired: verify commit present, then CronDelete.

# (history below)
# STATE — Program 3 DURABLE COMMODITY BOOK (commodity worktree resume brain). Loop: cron fdd1649a (5h).

Worktree /home/speculation/convexity-commodity (branch commodity); data read from
/home/speculation/convexity/data (shared, on-disk). Live paper = main @ 31d3238 (orphan fix),
UNTOUCHED. Reuses swing_pipeline.py (data layer + metrics/cpcv/dsr).

## Progress
- Prereg DURABLE_PREREG.md FROZEN + committed. durable_pipeline.py built + runs (839 wk breadth-7).
- FIRST RUN produced results_durable.json but with TWO unresolved issues (do NOT report a verdict):

## OPEN ISSUE 1 — combined book raw metrics broken
cagr=-100%, maxdd=nan: per-name vol-scaled positions (1/rvol) are NOT normalized to a book gross
before combining + cluster_risk_parity multiplies, so raw leverage compounds through zero. FIX:
normalize each sleeve/book to a target gross (or unit ex-ante vol) BEFORE metrics; report the
vol-targeted series for level metrics (cagr/maxDD). Sharpe/per-year are scale-invariant so already
valid.

## OPEN ISSUE 2 — REGIME-SANITY FAILURE (the important one; prereg mandates investigate-before-belief)
S1_band Sharpe ~0.016 (~zero) AND per-year 2022 = -0.29. 2022 was THE commodity-trend year — a slow
trend book MUST be strongly positive in 2022 (every CTA printed). Negative 2022 ⇒ the trend
construction is very likely mis-capturing trend, NOT "trend dead". HYPOTHESES to check next firing:
(a) weekly position staleness — position at t applied to wret[t]=t->t+1 is correct, but verify the
trailing-return sign uses lr[t-L..t-1] (known at t), not including t; (b) roll-clean return in deep
2022 backwardation — natgas/crude front-roll returns may be distorted (verify a naive always-long
CL/NG 2022 net return is strongly POSITIVE in our series; if it is but the trend book isn't, the
sign/lag is the bug; if buy-and-hold itself is wrong, the return series is); (c) vol-scaling by σ20d
(daily) — a 2022 vol spike shrinks positions right when trend pays, muting 2022 (plausible real
effect, but shouldn't flip to NEGATIVE); (d) natgas 2022 round-trip (spike+crash within the year)
could net a whipsaw for a slow book — decompose per-name 2022 (CL vs NG vs metals).
S1_band per-year (scale-invariant, valid): 2010 1.58, 2011 .35, 2012 -2.18, 2013 .28, 2014 -.81,
2015 1.41, 2016 -.88, 2017 -.58, 2018 -1.02, 2019 .01, 2020 .28, 2021 .95, 2022 -.29, 2023 -1.2,
2024 .28, 2025 1.3, 2026 .11.

## NEXT ACTION
Diagnose ISSUE 2 first (durable_diag.py: per-name 2022 buy-and-hold vs trend-sign vs trend-book
return, for CL/NG/GC). If a construction bug -> fix durable_pipeline, rerun, re-check the regime
signature (2022 MUST be strongly positive for a valid trend book). THEN fix ISSUE 1 (normalize book
before level metrics). THEN full report + gates (DSR/PBO/drop-top-5). Only after 2022 reads sane may
any Sharpe be reported as a finding. If, after correct construction, 2022 is strongly positive but
the full-sample net is still ~0-0.3 after cost, THAT is the honest durable verdict (weak but real),
report it. Never report the current ~0.016 as "dead" — it fails its own sanity gate.
Deliver report in chat -> cancel loop fdd1649a.
