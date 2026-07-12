# P8 — CROSS-ASSET RV SPREADS — STUDY RECORD (Program 8)

**Verdict: NEGATIVE, decisively — 0/8 gates. Neither pre-named cross-asset spread carries a
deployable reversion premium at the weekly horizon. The last honest lever for lifting the RV
CAGR:DD (new independent return streams on unmined data) is now also closed.**

Run 2026-07-12. Prereg: [P8_PREREG.md](P8_PREREG.md) (frozen c64d4fe). Numbers:
`results_p8.json`. Panel: fresh weekly closes from the stitched lake (never touched by P4–P7).

## Results
| | Sharpe | maxDD | CAGR | drop-5 | CPCV frac+ |
|---|---|---|---|---|---|
| XLE–CL (2018→) | 0.217 | 25.4% | 2.0% | **−0.223** | 0.77 |
| MSTR–BTC (2020-09→) | **−0.181** | **68.3%** | −4.5% | −0.444 | 0.28 |
| book (pair RP) | 0.041 | 44.6% | −0.6% | −0.314 | 0.32 |

- **MSTR–BTC**: the mNAV premium is NOT a weekly-horizon mean-reverting spread on this sample —
  it regime-trends (expansion phases then collapse), and betting reversion got run over in both
  directions (solo maxDD 68%). The famous trade of this cycle is a REGIME/positioning trade,
  not an RV spread — exactly the GC–HG/GC–SI signature class, now confirmed cross-asset.
- **XLE–CL**: weakly positive but a lucky-weeks profile (drop-top-5 NEGATIVE) — the same
  disqualifier that killed Programs 2–3.

## Canary note (honest self-correction, recorded)
The lookahead canary produced −2.22 Sharpe; the pipeline's gate as written expected a large
POSITIVE value — a sign-convention miscode in the GATE, not in the causality: for reversion
sizing ∝ −z, including the traded return in the level makes the canary hugely NEGATIVE
(Program 4's own canary was −3.81, the recorded precedent). The −2.22 magnitude therefore
CONFIRMS the main path is on the honest side of the off-by-one; the gate row is void as
miswritten and superseded by this interpretation. This does not affect the verdict (all seven
substantive gates fail on their own).

## Program arc closure (P4 → P8)
Every honest route to improving the commodity/RV CAGR:DD has now been tested and recorded:
generic overlays (P5), literature stops/regimes (P6), the institutional stack (P7), breadth
within commodities (P5/U1), and new cross-asset spread families on fresh data (P8) — fifteen
overlay techniques plus three universe extensions, one frozen rule, zero survivors. The
deployed 5-pair book at 0.73 Sharpe / 0.48 ratio is the local optimum of everything reachable
from here. The ratio objective is met at FLEET level (~2.86:1) through diversification.
Standing revisit triggers: venue universe events (new spread families), and the K6 (PCA-1)
near-miss re-examined on accumulated LIVE data only, not before ~2027.
