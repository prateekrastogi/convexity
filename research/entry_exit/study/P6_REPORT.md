# P6 — LITERATURE DD-SHAPERS FOR THE RV BOOK — STUDY RECORD (Program 6)

**Verdict: NEGATIVE — all five literature-canonical risk shapers REJECTED by the ½-rule; four
of five actively RAISED drawdown or destroyed return. The deployed construction is
overlay-resistant for structural reasons (below). DD:CAGR ≈ 0.5 is intrinsic to this book.**

Run 2026-07-12. Prereg: [P6_PREREG.md](P6_PREREG.md) (frozen 61ae544 before any run; canonical
parameters from the pairs-trading literature, zero grids). All numbers: `results_p6.json`.
S0 reproduction exact (0.732 / 18.5%).

## Waterfall (frozen order; acceptance = relDD ≥15%, relCAGR ≤ ½·relDD, CPCV held)
| candidate | Sharpe | maxDD | CAGR | ΔDD | ΔCAGR | verdict |
|---|---|---|---|---|---|---|
| L1 z-divergence stop @3σ (re-arm 0.5) | 0.546 | **28.1%** | 6.55% | **+52% WORSE** | −26% | reject |
| L2 time stop @2×half-life | 0.558 | 23.7% | 7.48% | +28% worse | −16% | reject |
| L3 Avellaneda-Lee discrete bands 1.25/0.5 | 0.355 | **29.7%** | 3.93% | +61% worse | −56% | reject |
| L4 Hurst<0.5 regime gate (104wk) | 0.723 | 18.6% | 8.88% | ~0 | ~0 | reject (no-op) |
| L5 crowding de-gross (Khandani-Lo) | 0.690 | 17.7% | 7.31% | −4% | −18% | reject (½-rule) |

Final = unchanged deployed book. Gates 5/7 (ratio 0.479 fails, DSR 0.778 fails). PBO 0.214.

## Why the canonical toolbox fails HERE (the finding worth keeping)
The deployed construction — continuous band-z sizing × half-life gate × pair risk-parity —
already embodies, in continuous form, what each discrete overlay tries to add:
1. **Stops amputate the payoff, not the risk.** With size ∝ stretch, the position is largest
   exactly at |z|≈3 — and the 3→0.5 reversion leg is where the edge LIVES. A divergence stop
   (L1) or time stop (L2) realizes the worst mark as a permanent loss and then misses the
   recovery it just paid for. MaxDD goes UP (18.5→28.1 / 23.7): the overlays convert temporary
   drawdown into realized drawdown. (Mirror image of P5's brake finding, same root cause: on
   mean reversion, drawdown IS the entry signal.)
2. **Break protection is already priced in.** The HL gate de-activates spreads whose AR(1)
   dynamics die — which is why L4 (Hurst) is a pure no-op (redundant estimator of the same
   regime) and why L1's break-insurance has nothing left to insure (P4 showed PL–PA/GC–SI
   correctly score ~0 through their breaks).
3. **The canonical AL thresholds are strictly dominated** by continuous sizing on this panel
   (L3: Sharpe 0.355 vs 0.732) — binary entry/exit throws away the information in |z| magnitude.
4. **Concentration is already risk-paritied**; halving gross on correlation spikes (L5) cuts
   return 4.5× more than drawdown.

## Cumulative program map for this universe (all committed records)
P2 directional swing: dead · P3 TSMOM/carry: dead · P4 RV pairs: SURVIVOR (deployed) ·
P5 RV breadth + generic DD overlays: negative · P6 literature DD shapers: negative.
**The commodity construction question is CLOSED.** Eight distinct risk-shaping techniques from
two programs all fail one pre-registered cost-benefit rule; the 18.5% maxDD at 8.9% CAGR is the
price of this edge, not removable engineering fat. The book's fleet role stays: diversifier.
Reopen only on a universe event (new spread families listed on venue).
