# RV PAIR STUDY — STUDY RECORD (Program 4)

**Verdict: FIRST SURVIVING CANDIDATE in this universe — net Sharpe ~0.73, robust to every
post-hoc attack — but the frozen adoption gate does NOT fully pass (PBO 0.386 > 0.25, DSR 0.856
marginal). Recommendation: NO real-money deploy off the backtest; graduate to PAPER validation.**

Run 2026-07-05 (commodity worktree). Pre-registration: [RV_PREREG.md](RV_PREREG.md), frozen +
committed before any run. All numbers from `results_rv.json` / `results_rv_posthoc.json`. Panel:
839 weekly Fridays 2010-06 → 2026-07 (positions from ~2012 — the 104-wk windows need history).

## The pre-named deployable (band z {26,52,104}, β=1 ratio, HL-gate [1,26]wk, pair RP, 10% vol)
| | value |
|---|---|
| net Sharpe (12bp RT) | **0.732** (vol-targeted 0.705) |
| EV / win% | 18.0 bp/nm-wk · 46.7% |
| maxDD / CAGR | **18.5%** / 8.9% (vt: 19.4% / 7.3%) |
| **drop-top-5 Sharpe** | **+0.566** (first-ever pass in this universe) · top-5 share 0.29 |
| CPCV (8,2, emb 8wk) | mean path **0.731**, **frac_pos 1.00** (28/28) |
| DSR (N=20 trials) | **0.856** (PSR vs 0: 0.9987) — just under the ~0.9–0.95 pass bar |
| PBO (CSCV, 12 configs) | **0.386** — FAILS the ≤0.25 gate (moderate selection risk) |
| cost scenarios | 6bp → **0.827** · 12bp → 0.732 · **24bp → 0.541** (survives doubled cost) |
| funding ±5bp/wk | 0.910 / 0.554 |
| turnover / cost drag | 0.39 gross/wk → 4.7 bp/wk at 12bp |

Per-year (pre-named): 2012 +2.19, 2013 −0.98, 2014 −0.54, 2015 +1.71, 2016 +1.24, 2017 +0.35,
2018 −0.37, 2019 +0.98, 2020 +0.53, 2021 +0.78, 2022 +1.83, 2023 +0.30, 2024 +1.25, 2025 +0.63,
2026 +2.40 (partial). Positive 12/15 — no single-regime dependence, real drawdown years present.

## Per-pair (band, ungated, solo) & post-hoc jackknife (labeled: NOT pre-registered)
| pair | solo Sh | EV bp | note | drop-pair book Sh (H1/H2) |
|---|---|---|---|---|
| SI–PL | **0.755** | 23.6 | best; gated solo 0.672 | **0.540** (0.45/0.62) |
| CL–NG | **0.550** | 33.8 | gated solo 0.576 | 0.593 (0.50/0.68) |
| GC–PL | 0.275 | 7.1 | | 0.648 (0.64/0.69) |
| PL–PA | 0.112 | 3.8 | 2017 structural inversion → gate cuts it to 66% active | 0.716 (0.42/0.94) |
| GC–SI | 0.010 | 0.2 | ratio trended 80→30→90 → no reversion, correctly ~0 | 0.728 (0.68/0.78) |

Full book halves: **H1 0.608 / H2 0.843** — stable, second half stronger (not a decayed artifact).
No drop-one-pair book goes negative in any half; worst case (lose the best pair) is still 0.54.

## Honesty checks — all passed
- **Lookahead canary** (the critical one for mean reversion): unshifted price levels (z includes the
  t→t+1 return being traded) produce Sharpe **−3.81** — a ~4.5-Sharpe swing on the off-by-one. Our
  book uses the causal shifted levels (lp[t] = Σ_{k<t} log1p(wret[k])); the canary proves the
  sensitivity and that we are on the honest side of it.
- **Structural-break signature** (prereg): the two pairs with known breaks — GC–SI (ratio regime
  shifts) and PL–PA (2017 permanent inversion) — score ~0, and PL–PA solo carries a 64% maxDD. The
  breaks show up as they must; nothing fake-profits through them. The HL gate correctly de-activates
  broken spreads (PL–PA active only 66% vs 76–85% for others) and adds +0.11 vs ungated (0.62).
- **Above-ceiling scrutiny** (prereg: result 0.73 > stated 0.3–0.5 ceiling, below the 0.8 alarm):
  triggered the post-hoc battery above — jackknife, halves, turnover, cost-doubling all hold. The
  ceiling prior was likely too conservative for this bet type (continuous sizing + 5 spreads +
  16 years); no evidence of a bug found.

## Frozen adoption gate
| gate | threshold | result | pass? |
|---|---|---|---|
| net Sharpe positive | > 0 | 0.732 | ✓ |
| DSR pass | ≳ 0.9 | 0.856 | **marginal fail** |
| PBO | ≤ 0.25 | 0.386 | **NO** |
| drop-top-5 positive | > 0 | +0.566 | ✓ |
| structural-break signature | breaks visible as losses | ✓ (GC–SI/PL–PA ~0, DD years real) | ✓ |

3/5 pass → per the frozen rule, **no deploy recommendation off this backtest**. But this is
categorically unlike Programs 2–3 (which failed 4–5/5 with negative drop-top-5): every robustness
axis holds, and PBO 0.386 is the same neighborhood the crypto xsec book showed (0.39) before it was
mitigated by band-deployment and paper validation.

## Caveats that only paper can resolve
1. **Instrument transfer**: backtest is on futures settles; the tradeable is Binance spot-indexed
   perps. Ratio reversion transfers far better than curve carry (spot ratios ≈ front ratios for
   these), and funding scenarios are covered (±5bp → 0.55–0.91) — but it must be measured.
2. **Thin legs**: PL, PA, SI perps are thinner than the majors; the doubled-cost scenario (0.54 @
   24bp) is the honest planning number for them, not 0.73.
3. **Selection risk** (PBO 0.386): mitigated by deploying only the pre-named construction — never a
   best cell — but 0.386 means best-in-family picking is ~39% likely to disappoint OOS.

## Recommendation
- **Do not deploy real money on the backtest.** Plan around **net ~0.5–0.6 live** (24bp-scenario /
  haircut basis), not 0.73 — honest answer to the 0.8–1.0 ask: this book alone won't reliably get
  there; it can plausibly get 0.5–0.7 as a **diversifier** nearly uncorrelated with the crypto
  xsec book (different asset class, different bet type).
- **Graduate to paper**: add the pre-named RV book as a small second paper strategy (same weekly
  Friday cadence as the crypto rebalancer, ~5 pairs = 10 legs, tiny operational surface) to measure
  perp-vs-futures transfer, real fills on the thin legs, and funding drag. Adopt only if paper
  tracks the backtest.
- Sizing if/when adopted: quarter-Kelly convention; at Sharpe ~0.5–0.7 and 10% vol target this is a
  modest sleeve next to the crypto book, not a co-equal.

## Reproduce
`rv_pipeline.py` (study) + `rv_posthoc.py` (labeled post-hoc) in the commodity worktree; reads the
shared lake; live paper (main @ 31d3238) untouched throughout.
