# No-trade band calibration — `band_c` 0.5 → 0.03

**Date:** 2026-08-14 · **Branch:** `bandcal` · **Artifacts:** `band_calibration.py`,
`band_robust.py`, `results_bandcal.json`

## How it surfaced

Routine question about the weekly digest's `drift` line: demo showed model-vs-venue drift of
+187 on 4,414 (4.2%) with crypto coverage at 124%. Tracing that gap found the defect below.
Ruled out first, in order: orphan/delisted positions (zero across all books), demo-vs-prod price
divergence (534 shared symbols, only `KORUUSDT` disagrees >5% and it is in no book), and a stale
`w_carried` (live mode correctly reads venue positions, [cli.py:801](../../src/convexity/paper/cli.py:801)).

## The defect

`ExecutionDefaults` makes `band_c` a **required** field with no default. Its docstring states c is
*"fitted once offline to sigma_w"* and that *"refusing to invent a number is the contract."*
`policy.build_plan` then supplied `band_c: float = 0.5` as a Python default — inventing exactly
the number the config layer refused to invent.

Band half-width is `b = c·(θ/ρ)^⅓`, `θ = 2(taker + half_spread)`. At c=0.5:

| book | band | median \|w\| | names whose **entire** position < band |
|---|---|---|---|
| crypto | 5.31% of NAV | 1.41% | **57/57 (100%)** |
| equity | 5.85% of NAV | 1.68% | **68/68 (100%)** |
| commodity | 5.85% of NAV | 7.36% | 2/6 (33%) |

Only `REASON_FLIP_FULL` trades — open-from-flat, close-to-flat, sign change — bypass the band
([execution.py:872](../../src/convexity/execution/execution.py:872)). **Every resize of a held
same-sign position was frozen.** That produces high turnover and permanent drift simultaneously:
names churn in and out fine, but any name that stays keeps whatever size it first got.

Demo crypto is the proof. NAV fell 4,996 → 1,735 over six weeks; targets shrank with it; held
positions never resized down. CHZ sat at −89 against a −15.26 target because −89 was a sane weight
when NAV was 5,000. `position_drift` logged ~640 USDT with an identical name list every day for a
week — a fixed point, exactly as the mechanism predicts. Commodity escaped because its 6 RV legs
are ~7.4% each, above the band.

## Calibration

`band_calibration.py` reproduces `equity_trend.backtest_book` and inserts the live band rule at
each weekly decision so the carried book ratchets as production does.

**Faithfulness gate:** at c=0 the band is inert, so the harness must reproduce the D1-validated
reference. It does, **bit-exactly** — CAGR/maxDD/Sharpe max diff `0.00e+00`. Every row below is
therefore trustworthy.

| c | band | CAGR | Sharpe | maxDD | turnover/yr | skipped | TE/yr |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.000% | 28.26% | 1.168 | 30.02% | 10.15× | 0.0% | — |
| 0.02 | 0.234% | 29.45% | 1.199 | 30.28% | 6.95× | 44.3% | 0.86% |
| **0.03** | **0.351%** | **29.71%** | **1.204** | 30.47% | 6.06× | 53.0% | 1.24% |
| 0.05 | 0.585% | 29.59% | 1.200 | 30.70% | 4.79× | 65.0% | 1.94% |
| 0.10 | 1.170% | 25.92% | 1.108 | 30.66% | 3.14× | 80.6% | 3.65% |
| **0.50 (was shipped)** | **5.848%** | **15.21%** | **1.015** | 25.75% | 1.39× | **95.5%** | **11.25%** |

A *small* band genuinely beats no band — it saves ~4×/yr of turnover at 13bp, which is what §4's
band is for. 0.5 was simply 15–35× too large.

## Robustness

`band_robust.py`, history 2018-05-07 → 2026-06-26, split 2022-05-27:

- **Sub-period:** the 0.02–0.05 plateau wins in **both halves** (H1 argmax 0.03, H2 0.05).
- **Per-year:** c=0.03 beat c=0.5 in **7 of 9 years**. The two losses are 2018 (warmup, 0%) and
  2022 (−21.7% vs −18.8%).
- **The 2022 loss is the honest caveat:** a frozen book de-risks passively, so the wide band posts
  a lower maxDD (25.75% vs 30.47%). It costs 13 CAGR points to buy that. On CAGR:DD, c=0.03 gives
  **0.98** vs 0.59 for c=0.5.

The choice *within* 0.02–0.05 is noise. Moving off 0.5 is not.

## Shipped

- `fleetconfig.BAND_C = 0.03`, with the calibration recorded above it.
- `policy.build_plan` takes `band_c: float = BAND_C` — no invented default.
- `tests/paper/test_fleet_infra.py::test_band_c_calibrated_and_smaller_than_a_position` pins the
  constant and asserts the property that matters: the band must stay under half a median position
  or the book freezes.

Resulting bands: crypto 0.319%, commodity 0.351%, equity 0.351% of NAV.

## Limitations

- **Fitted on the equity book only** — the sole book with a reproducible validated backtest.
  Crypto and commodity inherit the constant. Their bands land at ~23% and ~5% of a median
  position, which is sane, but they are **not** independently calibrated. Recalibrate per book if
  either grows its own validated backtest.
- Selection is in-sample over 14 grid points on one book. Mitigated by the two-half split and the
  per-year table, not eliminated.
- The band's drawdown-suppressing side effect is real and is being given up deliberately.
