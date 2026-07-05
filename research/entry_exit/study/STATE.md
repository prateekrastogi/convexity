# STUDY STATE — research/entry_exit/study (resume brain). Loop: cron ea55c4d6 (5h backstop).

## Program 2 — SWING XSEC COMMODITY (active). Data ALL landed.

- COT ✅ (data/cot/cot_disagg.parquet, 7 syms x ~1046 reports, usable_from lag).
- Databento curves ✅ (data/curves/databento/*.csv raw WITH instrument_id; parsed parquet lacks it —
  the pipeline reads the RAW csvs). 7 roots x ranks 0-7, 2010-06+.
- Dukascopy extension ✅ (HIST DONE) but **NOT USED** by this daily study (superseded — see below).

## KEY DECISION (supersedes SWING_PREREG_DRAFT steps 1-2)

Price basis = **databento FRONT-MONTH daily settles**, roll-aware returns via instrument_id change
on rank 0 (roll day return = c0_t / c1_{t-1}). Uniform panel: **839 Friday weeks, breadth 7
throughout, 2010-06..2026-07** — longer & cleaner than the duka-stitch (which had per-name floors +
XPT/XPD only 2021). No 1-min re-stitch / prep_frames needed (those were the directional-ORB frames).
Perp = execution vehicle only (tracking error negligible at swing horizon, covered by cost model).

## BUG FOUND + FIXED (why never trust an all-negative first pass)

First run: ALL factors negative incl. carry (−0.38 gross) — a red flag vs 35yr carry evidence.
Diagnostic proved it was DATA CONTAMINATION: WTI 2020-04-20 settled −$2.67 → spurious −116%/−439%
"returns" through the negative price, detonating the dollar-neutral book. Carry decile spread is
correctly POSITIVE (+8.1bp/wk most-backwardated vs most-contango). FIX (swing_pipeline.root_daily,
declared data hygiene): invalidate any return computed through a non-positive price; winsorize daily
returns to ±40% (no legit daily move for these 7 reaches it). Re-running now.

## Pipeline (research/entry_exit/study/swing_pipeline.py) — self-contained

Loads raw curve csvs + COT → roll-aware daily returns → weekly Friday panel (adj index, carry =
annualized log(c0/c1) by data-derived per-root roll spacing, rvol20, skew40/60, COT dmm/pnet) →
factors F1 carry, F2 basis-mom (4wk Δcarry), F3 reversal (−prev1wk, COT-flow-top-half conditioned;
+uncond variant), F4 mom{6,8,10}, F5 skew{40,60}, F6 mom8×hedging-pressure → inverse-vol
dollar-neutral gross-1 weight matrices → net {6,12,24}bp + funding → ensembles {F3+F4+F5},{+F6},
{+F1+F2 all} + vol-target overlay → CPCV(8,k2) paths, per-obs DSR @ N_eff≈13, drop-top-5, Kelly.
Output research/entry_exit/study/results_swing.json.

## STATUS: ✅ DONE 2026-07-05 — reported to owner; loop ea55c4d6 cancelled.

VERDICT: NO deployable edge on the 7-perp universe. Clean, verified NEGATIVE (not a bug):
- Panel 839 wk breadth-7 2010-2026 (databento front-month, roll-aware).
- Every factor net-negative; best = skew ~−0.2; ensembles −0.6 to −0.9; DSR ≈ 0.00 all.
- Carry raw extreme spread +1.68bp/wk (Sharpe +0.01) ≈ zero → net-negative after 2/2 book + cost.
- Bug found+fixed first (WTI 2020-04-20 −$2.67 → spurious −116%/−439% returns; winsorized). Engine
  then VERIFIED: pipeline k=1 gross (+1.69bp) == independent decile (+1.68bp), no sign/align bug.
- Mechanism: 4/7 are precious metals (degenerate carry) + breadth 7 too small for a quantile book
  (fundamental law) + short-mom decay + turnover. Curve factors make it WORSE (metal-heavy).
Artifacts committed on `commodity`: SWING_PREREG.md (frozen), REPORT.md, results_swing.json,
swing_pipeline.py, diag.py, ab.py. To ever revisit: needs a 20-30 name CME/LME dated-futures
universe (different data + execution program), NOT a construction tweak on these 7 perps.

## Invariants
Prereg-before-claim; net-of-cost; no-lookahead (weight_t uses data ≤ t, scored on t→t+1; COT
usable_from ≤ Friday t); deployed pick = pre-named ensemble passing gates, never best cell; never
fabricate; VM bg compute; write-local-then-scp; never touch src/convexity or live paper.
