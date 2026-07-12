# P8 PRE-REGISTRATION — CROSS-ASSET RV SPREADS (Program 8)

**Frozen before any run, 2026-07-12. Motivation: P5–P7 established (15 techniques, one frozen
rule) that the commodity RV book's CAGR:DD is intrinsic — overlays cannot lift it. The only
honest lever is NEW independent return streams on NEW data. This program tests spread families
that exist because the venue uniquely lists both legs across asset classes. NO overlay mining;
constructions frozen here.**

## The two pre-named pairs (economic pinning stated; universe is these two, full stop)
1. **XLE–CL** (energy equities vs crude): ECONOMIC ratio → the Program-4 construction VERBATIM
   (β=1 on weekly log closes, band z {26,52,104}, HL gate [1,26]wk, continuous −clip(z,2)/2).
   Sample: 2018-05 (data start) → now.
2. **MSTR–BTC** (the mNAV premium): STATISTICAL ratio — MSTR ≈ β_t·BTC + premium where β_t
   drifts with BTC-per-share — so the hedge is Kalman-filtered (P7's K1 machinery, canonical
   δ=1e-4, λ=0.98; the construction P7 proved WRONG for economic ratios and which is standard
   for exactly this kind of pair). z on the Kalman innovation, band {13,26,52} (pre-named
   shorter: the documented mNAV cycle is weeks–months and the sample is ~300wk), HL gate,
   continuous sizing. Sample: 2020-09-01 (MSTR's first BTC treasury purchase) → now.

Book = the two pairs under pair risk-parity (P4 convention). Weekly Friday, from the stitched
lake's daily closes (W-FRI last). Costs 12bp RT primary, {6,24}bp scenarios; 24bp is the
planning number (MSTR/XLE perps are thin).

## Cells (DSR honesty — FRESH data, fresh count): pair XLE–CL, pair MSTR–BTC, book = 3 cells
+ 2 diagnostics (each pair ungated) → N_eff = 6. PBO family = 5 series.

## Adoption gates (short-sample honesty; graduation target = PAPER sleeve only, never direct
prod; any deployment ALSO requires an owner architecture ruling — the legs belong to the
equity/crypto books' domains today)
| gate | threshold |
|---|---|
| book net Sharpe (12bp) | ≥ 0.60 |
| CPCV (8,2, embargo 8wk) frac_pos | ≥ 0.75 |
| drop-top-5-weeks Sharpe | > 0 |
| 24bp scenario Sharpe | ≥ 0.40 |
| per-obs DSR (N_eff=6) | ≥ 0.70 |
| PBO (5 configs) | ≤ 0.35 |
| each pair's solo maxDD | ≤ 60% |
| lookahead canary (unshifted levels) | must fake a large positive edge |

Verdict rules: pass → recommend a PAPER cross-asset sleeve (sizing/architecture = owner);
fail → record the negative; nothing changes anywhere. Either way the deployed books are
untouched by this program.
