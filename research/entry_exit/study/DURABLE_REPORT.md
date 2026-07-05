# DURABLE COMMODITY BOOK — STUDY RECORD (Program 3)

**Verdict: DEAD net-of-cost on the 7-name Binance USDT-M perp universe. DO NOT DEPLOY.**
Run 2026-07-05 (commodity worktree). Pre-registration: [DURABLE_PREREG.md](DURABLE_PREREG.md), frozen
+ committed before any run. Every number below is from `results_durable.json` (actual run, no
fabrication). Panel: 839 weekly Fridays, 2010-06-11 → 2026-07-03, median breadth 7.

## Bottom line
The two families the durability thesis singled out — **slow time-series trend** and
**carry-on-storables** — are the *only* two positive standalone cells (L52 Sharpe +0.149, carry
+0.139). That is genuine directional confirmation of the thesis. But the magnitude is **sub-noise**:
every cell is DSR ~0.11 or worse, PBO 0.543, and **drop-top-5-weeks is negative everywhere** — the
tiny positive is a handful of lucky weeks. The pre-named combined book fails 4 of 5 adoption gates.

Root cause is **breadth**, not construction: 7 names (4 of them degenerate-carry metals, effective
breadth ≈ 4) cannot aggregate real-but-small per-name risk premia into a deployable book net of
12 bp. This is the same wall Program 2's cross-sectional swing study hit — now confirmed for the
time-series construction too. Directional/carry commodity investing on a Binance-perp-constrained
universe is closed.

## Sleeve results (gross-normalized to unit book, 12 bp RT, weekly Friday)
| sleeve | Sharpe | EV bp/nm-wk | win% | maxDD% | CAGR% | drop-top5 Sh | CPCV frac+ | DSR |
|---|---|---|---|---|---|---|---|---|
| S1 TSMOM L13 | −0.128 | −3.3 | 46.9 | 63.2 | −2.54 | −0.395 | 0.32 | 0.012 |
| S1 TSMOM L26 | −0.047 | −1.1 | 47.3 | 44.0 | −1.37 | −0.225 | 0.46 | 0.025 |
| S1 TSMOM L39 | −0.096 | −2.3 | 46.5 | 46.5 | −1.89 | −0.270 | 0.50 | 0.016 |
| **S1 TSMOM L52** | **+0.149** | +3.6 | 48.8 | 32.6 | +1.08 | −0.011 | 0.68 | 0.121 |
| S1 TSMOM band (pre-reg'd) | −0.100 | −2.8 | 48.4 | 59.0 | −2.42 | −0.292 | 0.43 | 0.015 |
| **S2 carry-on-storables** | **+0.139** | +5.1 | 50.1 | 52.9 | +0.86 | −0.069 | 0.68 | 0.112 |
| S3 gate (carry∧trend) | −0.007 | −0.4 | 43.2 | 75.1 | −5.52 | −0.242 | 0.46 | 0.036 |
| S3 veto | −0.005 | −0.3 | 43.1 | 75.0 | −5.44 | −0.241 | 0.50 | 0.037 |
| S5 KRT (COT reversal) | −0.778 | −24.4 | 45.7 | 93.5 | −13.1 | −1.004 | 0.14 | 0.000 |

## Pre-named combined book (the only thing eligible for a deploy call)
Construction = preciouses S1-band trend + storables (S2 + S3-gate) + 0.2×S5 → cluster risk parity →
10% vol-target overlay. (Never the best single cell.)

| | Sharpe | EV bp | win% | maxDD% | CAGR% | drop-top5 Sh | top5 share | DSR |
|---|---|---|---|---|---|---|---|---|
| combined (raw net) | 0.092 | 2.7 | 48.8 | 61.8 | 0.27 | −0.153 | 2.45 | 0.079 |
| combined @10% vol-target | 0.045 | — | — | 54.3 | −0.15 | — | — | — |
| combined **ex-S5** (robustness) | 0.131 | 3.8 | 48.6 | 59.6 | 0.86 | −0.103 | 1.70 | 0.103 |

Cost/funding fragility (combined): Sharpe **0.19 @6bp → 0.092 @12bp → −0.102 @24bp**; funding
−5bp/wk → +0.263, +5bp/wk → −0.078. **The entire "edge" lives inside the cost/funding assumption** —
the hallmark of no real edge. PBO over the 9-config family = **0.543** (IS winner lands below the OOS
median in 54% of CSCV splits).

Per-year Sharpe (combined ex-S5): 2010 +2.06, 2011 +1.68, 2012 −0.24, 2013 +1.17, 2014 −0.64,
2015 +1.16, 2016 −1.13, 2017 −1.05, 2018 −0.84, 2019 −0.31, 2020 −0.36, 2021 −0.63, 2022 +0.01,
2023 −1.01, 2024 +0.59, 2025 +0.77, 2026 +1.08.

## Adoption gate verdict (frozen criteria)
| gate | threshold | result | pass? |
|---|---|---|---|
| net Sharpe positive | > 0 | 0.092 raw / 0.045 vol-targeted | technically yes (trivial) |
| DSR pass | ≥ ~0.9 | **0.079** | **NO** |
| PBO | ≤ 0.25 | **0.543** | **NO** |
| drop-top-5 positive | > 0 | **−0.153** | **NO** |
| regime signature | drought weak ✓ **and** 2020–22 carries | drought weak ✓; **2020–22 did NOT carry** (−0.51/−0.64/−0.10) | **NO** |

**4 of 5 gates fail → NO DEPLOY.** Quarter/full-Kelly and DD-vs-leverage were NOT computed: the
prereg gates them on a positive, gate-passing edge, which does not exist here.

## Regime-sanity check (the mandatory honesty gate) — PASSED as a *check*, and it validated the engine
- **No overfit/lookahead signature:** the 2011–2019 trend drought shows as flat-to-weak/negative
  (2016 −1.24, 2017 −1.21, 2018 −0.73, 2019 −0.47). A smooth equity line through that window would
  have meant a bug — there isn't one. Good.
- **Engine validated by cross-sleeve coherence in 2022:** in the raw-invvol run, S2 carry = **+0.79**
  and S3-gate = **+1.31** in 2022 — carry and carry-gated trend paid *exactly* in the energy
  backwardation year the thesis predicts, using the *same* roll-aware return series and sign
  convention as the trend sleeve. A sign/lag bug would have corrupted carry too; it didn't. So the
  pure-trend 2022 weakness (−0.10 to −0.29) is **real regime** — a metals-heavy 7-name basket
  (4/7 choppy-to-down preciouses+copper, NG round-tripped) — not a defect. An initial "trend must
  print in 2022" alarm was miscalibrated to *diversified multi-asset* CTAs (rates+FX+equities), which
  this universe cannot express.

## What we learned (why this is a clean negative, not a failed run)
1. **Thesis direction confirmed, magnitude un-deployable.** Slowest trend (L52) beats faster
   monotonically (L52 +0.15 > L26 −0.05 > L13/L39 ≈ −0.1/−0.13); carry-on-storables is positive;
   the reactive KRT overlay is destructive (−0.78). The ranking is exactly what "slow, painful,
   structurally-compelled" predicts — but the two winners are only ~0.14 Sharpe, DSR ~0.11, and
   their profit is a few weeks (drop-top-5 negative).
2. **Construction-fragility = noise.** S1-band flips sign between raw-inverse-vol (+0.016) and
   equal-gross (−0.10); S3 interaction flips positive→negative. A robust edge does not flip on the
   weighting convention. Both constructions nonetheless give the same verdict.
3. **The pre-registered band ensemble under-performs its best member** (L52) because equal-weighting
   drags in the cost-dead fast lookbacks — but reporting L52 as "the pick" is cherry-picking the best
   cell, which the prereg forbids. The honest deployable is the ensemble, and it is dead.
4. **Breadth is the binding constraint.** Slow trend + carry are real premia, but they compound into
   a deployable Sharpe only across cross-asset breadth (rates, FX, equity indices, ags, softs, dozens
   of commodities — the classic 50–100-market CTA portfolio). A Binance-USDT-M-perp slice of 7
   commodity names (effective breadth ≈ 4) is structurally too narrow, net of 12 bp.

## Recommendation
- **Do not deploy** any commodity directional or carry book on the current 7-name perp universe.
- **Do not allocate further research** to commodity-directional/carry unless the universe widens
  beyond Binance perps (a futures venue adding rates/FX/ags/softs would change the breadth math).
- This **closes the commodity-directional book**: fast breakout (dead, gross≈0), cross-sectional
  swing (dead, breadth-starved), and now slow TSMOM + carry (dead net, breadth-starved). The only
  surviving convexity edge remains **crypto cross-sectional momentum** (net Sharpe ~1.0–1.2,
  DSR-pass) — the live-paper focus. Next build stays there, not here.

## Reproduce
`/home/speculation/convexity/.venv/bin/python
research/entry_exit/study/durable_pipeline.py` in the `commodity` worktree; reads the shared lake at
`/home/speculation/convexity/data`; writes `results_durable.json`. Live paper system (main @ 31d3238)
untouched throughout.
