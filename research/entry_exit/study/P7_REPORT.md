# P7 — INSTITUTIONAL RISK STACK FOR THE RV BOOK — STUDY RECORD (Program 7)

**Verdict: NEGATIVE — all seven institutional techniques REJECTED under the pre-registered
rules (½-rule OR Sharpe-branch). With P5+P6 this makes FIFTEEN risk-shaping techniques across
three programs, spanning generic → literature-canonical → institutional, all failing one frozen
cost-benefit rule on this book. The Program-4 construction is at its efficient point; the
question is closed at maximum depth.**

Run 2026-07-12. Prereg: [P7_PREREG.md](P7_PREREG.md) (frozen 0792dd9). Numbers:
`results_p7.json`. S0 repro exact (0.732/18.5).

## Waterfall
| candidate | Sharpe | maxDD | CAGR | ΔDD | ΔCAGR | verdict |
|---|---|---|---|---|---|---|
| K1 Kalman dynamic hedge ratio | 0.505 | **32.1%** | 6.47% | **+74% WORSE** | −27% | reject |
| K2 CUSUM on AR(1) innovations | 0.575 | 21.5% | 6.71% | +16% worse | −25% | reject |
| K3 HMM high-vol de-gross | 0.720 | **14.8%** | 4.80% | −20% ✓ | **−46%** | reject (½-rule) |
| K4 Expected-shortfall sizing | 0.683 | 17.5% | 7.15% | −5% | −20% | reject |
| K5 ensemble vol target | 0.682 | 19.6% | 6.85% | +6% worse | −23% | reject |
| K6 PCA-1 factor neutralization | 0.741 | 17.1% | 8.37% | −8% | −6% | reject (both branches, near-miss) |
| K7 Amihud liquidity de-gross | 0.634 | 17.4% | 7.43% | −6% | −17% | reject |

Final = unchanged deployed book. Gates 5/7 (ratio 0.479, DSR 0.751@N_eff=50, PBO 0.600).

## The three structural reasons the institutional toolbox does not transfer here
1. **β=1 is economics, not naivety (K1's −74% DD lesson).** These pairs are ECONOMIC ratios
   (gold/silver, crude/gas); the mispricing LIVES in the fixed ratio. A Kalman β adapts toward
   whatever co-movement is recent — i.e., it absorbs the very dislocation being traded into the
   hedge ratio, then holds the residual noise. Dynamic hedging is for statistical pairs whose
   relationship is an estimate; it is anti-alpha for structural ratios.
2. **In RV, stress IS the payment moment.** Every de-risk-into-stress overlay — HMM high-vol
   gate (K3), ES sizing (K4), liquidity de-gross (K7), P5's brake, P6's stops — cuts exposure
   precisely when spreads are widest and about to converge. K3 is the cleanest demonstration:
   a real −20% DD bought at −46% CAGR, a strictly bad trade under any ratio objective.
3. **The construction already contains the list's load-bearing items.** Vol targeting (#5),
   adaptive reversion speed (#15, the HL gate), stability monitoring (#7), risk budgeting
   (#20), robust inputs (#13) were in the deployed book before P5 began — which is WHY the
   remaining overlays only had payoff to amputate.

K6 (PCA-1 neutralization) is the single technique that improved both axes (mildly: Sharpe
+0.009, DD −8%) yet cleared neither acceptance branch. It is noted for re-examination ON NEW
DATA ONLY (the accumulating live track) — never re-mined on this panel.

## Cumulative closure
P2 dead · P3 dead · P4 SURVIVOR (deployed) · P5 negative (3 overlays) · P6 negative
(5 literature) · P7 negative (7 institutional). DD:CAGR ≈ 0.48 is intrinsic; the book's value
is realized at FLEET level (40/30/30 fleet: 30.9%/10.8% ≈ 2.86:1 on the common window, vs
~2.5 without commodity). Reopen only on a universe event.
