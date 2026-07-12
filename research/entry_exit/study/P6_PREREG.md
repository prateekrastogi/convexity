# P6 PRE-REGISTRATION — LITERATURE DD-SHAPERS FOR THE RV BOOK (Program 6)

**Frozen before any run, 2026-07-12 (same day as, but AFTER, P5's negative — P6's candidate set
and gates are written with P5's results known but before any P6 number is computed; the
candidates are all NEW relative to P5's B1–B3 and come from the pairs-trading literature with
CANONICAL parameters, no grids).**

Owner objective: minimize the RV book's drawdown per unit CAGR using techniques documented for
statistical-arbitrage/pairs books specifically — the families P5's generic overlays did not
cover. Sources surveyed 2026-07-12: Avellaneda & Lee (2010) entry/exit band practice;
Leung & Li optimal-stopping OU thresholds; Göncü et al. (arXiv:1706.07021) stop-loss in optimal
stat-arb; Lipton & López de Prado closed-form OU trading; Khandani & Lo (2011) crowding/
deleveraging; Hurst-exponent regime gating (practitioner canon).

## Base
The deployed 5-pair book (Program 4 construction, frozen: band z {26,52,104}, β=1, HL-gate,
pair RP, weekly, 12bp). S0 must reproduce 0.732/18.5 exactly. HG pairs stay excluded (P5).

## Candidates — FROZEN order, canonical parameters, ONE shot each, composed like D1/P5
1. **L1 z-divergence stop (structural-break stop)**: pair de-arms when |z| ≥ 3.0 (the classic
   "the relationship broke" stop — divergence stops, not P&L stops); re-arms when |z| ≤ 0.5.
2. **L2 time stop**: a position leg older than 2 × trailing half-life (HL[t], clamped [4,52]wk)
   de-arms; re-arms on |z| ≤ 0.5. (Avellaneda-Lee practice: unconverged = thesis stale.)
3. **L3 discrete AL bands**: replace continuous −z/2 sizing with the canonical enter |z|≥1.25 /
   exit |z|≤0.5 unit-position state machine (Avellaneda & Lee 2010's published thresholds).
   Expected effect: less time-in-trade, lower turnover, fatter per-trade edge.
4. **L4 Hurst regime gate**: pair tradeable only while trailing 104wk aggregated-variance Hurst
   of the spread < 0.5 (mean-reversion regime confirmation, independent of the AR(1) HL gate).
5. **L5 crowding de-gross (Khandani-Lo)**: book gross × 0.5 while trailing 13wk mean pairwise
   |correlation| of pair returns exceeds its expanding 80th percentile (≥104wk burn-in) —
   de-risk when all spreads move together (systemic stress signature).

**Acceptance per candidate (identical to P5's ½-rule)**: relative maxDD reduction ≥ 15% AND
relative CAGR cost ≤ ½ the relative DD reduction AND CPCV frac_pos not down by > 0.05.
Composition: evaluated in order on the composition of previously accepted candidates. No
re-ordering, no parameter changes after results.

## Adoption gates (final composed construction; set BEFORE any run, differ from P5's
deliberately: this is a RISK-SHAPE program at held Sharpe, not a Sharpe-raise program)
| gate | threshold | rationale |
|---|---|---|
| net Sharpe | ≥ 0.70 | must not degrade the deployed 0.732 beyond noise |
| ratio CAGR:maxDD (at 25%-DD-budget sizing) | ≥ 0.60 | meaningful improvement over 0.48 |
| CPCV frac_pos | ≥ 0.90 | |
| drop-top-5 Sharpe | > 0 | |
| 24bp scenario Sharpe | ≥ 0.45 | |
| DSR (N_eff = 40 cumulative: P4 20 + P5 10 + P6 ≤10) | reported; adoption bar 0.80 | overlay-selection risk is bounded by canonical params + ½-rule, so the bar sits below P4/P5's discovery bar — stated openly |
| PBO over P6 family (≤8 configs) | ≤ 0.35 reported-with-caveat | same rationale |

**Verdict rules**: gates pass → recommend the composed construction for the PAPER book first
(prod resize remains a separate owner decision). Gates fail → record negative; deployed book
stands; accepted-but-ungated candidates are NOT adopted piecemeal.
