# P5 — RV BREADTH + DD SHAPING — STUDY RECORD (Program 5)

**Verdict: NEGATIVE. The owner objective (net CAGR:maxDD ≈ 1:1) is NOT honestly achievable on
this construction/universe. Every pre-registered lever failed its acceptance rule; the deployed
5-pair book stands unchanged. 4 of 7 adoption gates fail.**

Run 2026-07-12. Pre-registration: [P5_PREREG.md](P5_PREREG.md) (frozen 0b123c6, before any run).
All numbers from `results_p5.json`. S0 reproduction of the deployed book is EXACT
(0.732 / 18.5 — pipeline faithful to Program 4).

## Step A — HG breadth extension: REJECTED
| | Sharpe | maxDD | drop-5 |
|---|---|---|---|
| S0 (deployed 5-pair) | 0.732 | 18.5% | +0.566 |
| U1 (8-pair, +HG–CL, HG–PL, GC–HG) | 0.732 | 18.5% | +0.556 |

New-pair solos: HG–PL **0.709** (real, but adds ~nothing after pair risk-parity — highly
correlated with the existing metals cluster), HG–CL 0.074, GC–HG 0.112 (the copper/gold macro
ratio trends rather than reverts — the GC–SI signature, exactly as the prereg predicted).
Acceptance required Sharpe(U1) > Sharpe(S0): 0.732 > 0.732 is false → **rejected**. Copper adds
no independent reversion premium to this book.

## Step B — DD waterfall: ALL THREE REJECTED (rule: relCAGR-cost ≤ ½ relDD-gain)
| candidate | Sharpe | maxDD | CAGR | relDD gain | relCAGR cost | verdict |
|---|---|---|---|---|---|---|
| B1 GZ-brake@25% | 0.655 | **14.8%** | 6.13% | 20% | **31%** | reject |
| B2 active-fraction gross | 0.726 | 15.5% | 7.80% | 16% | 12% (needed ≤8%) | reject (near-miss) |
| B3 per-pair stop-reset | 0.746 | 18.5% | 9.10% | 0% | −2% | reject (no DD effect) |

**The structural finding worth keeping**: the GZ brake — which IMPROVED the equity trend book
(D1: 22/38.7 → 28/30) — is **counterproductive on a mean-reversion book**. Trend-book drawdowns
cluster and persist (de-levering into them saves tail); RV drawdowns are stretched spreads about
to pay (de-levering into them sells the edge at its cheapest). Same overlay, opposite sign of
effect, for a principled reason. B2's near-miss (12% vs 8% allowed) is noted and NOT adopted —
the acceptance rule is the pre-registration's whole point.

## Gates on the final (= unchanged deployed) construction
sharpe ≥ 0.75 **FAIL** (0.732) · CPCV pass (1.00) · DSR ≥ 0.9 **FAIL** (0.81 at N_eff=30) ·
PBO ≤ 0.25 **FAIL** (0.629 — mining the same panel again raised it, as it must) ·
drop-top-5 pass (+0.566) · 24bp pass (0.541) · **ratio ≥ 0.8 FAIL**.

## Why the 1:1 objective fails structurally (the useful takeaway)
The sizing frontier (k = 0.5→3.0) shows CAGR:maxDD ≈ **0.48:1 at every leverage** — the ratio is
scale-invariant to first order. It is a property of Sharpe and tail shape, not of sizing:
equity reaches ~0.94 with Sharpe 1.17 + a brake that fits trend tails; crypto ~0.9 with Sharpe
0.8–1.0 + dollar-neutral tails. Commodity RV at honest Sharpe 0.73 on ≤8 effective spreads has
neither the Sharpe nor a tail-shaper that survives its cost rule. To reach 1:1 this book would
need ~1.1+ net Sharpe — i.e., materially more independent spreads than this venue's commodity
universe contains (Program 3 closed directional/carry; this closes breadth-within-RV).

## Standing recommendation
Deployed 5-pair book unchanged (10% vol target, plan Sharpe 0.5–0.6 live). The commodity book's
honest role in the fleet is a diversifier at ~0.5:1 ratio, not a 1:1 engine. Revisit only if the
venue lists genuinely new spread families (e.g. refined products vs crude, more ags/softs) —
that is a universe event, not a construction question.
