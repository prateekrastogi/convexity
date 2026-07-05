# RV PAIR STUDY — PRE-REGISTRATION (Program 4; FROZEN 2026-07-05, before any run)

Owner-directed follow-up to Program 3: with all directional families dead at breadth 7, test the one
untested bet type — **relative-value mean reversion on economically coherent pairs**. Different bet
family: stationary-spread reversion can carry higher per-bet IC than outright premia, and
dollar-neutral pairs kill the regime exposure that whipsawed trend. Isolation: `commodity` worktree;
reads the shared lake; live paper (main) untouched. Executed by rv_pipeline.py.

## Honest prior (stated before the run)
Ceiling **0.3–0.5 net Sharpe**; **>0.8 = alarm/bug-hunt** (the fundamental-law arithmetic at 5 pairs
does not support it; the owner's desired 0.8–1.0 is explicitly NOT the expectation). Known structural
breaks — PL–PA permanent inversion (~2017), GC/SI spike (2020) — MUST appear as drawdowns for a
β=1 ratio-reversion book. A smooth equity line through them = lookahead/bug, investigate first.
Mean-reversion lookahead canary: the pipeline also computes the band book on UNSHIFTED price levels
(z includes the t→t+1 return being traded) purely as a bug detector — it should show a huge fake
Sharpe, and is excluded from the family/PBO.

## Universe / data / calendar
Same panel machinery as Programs 2–3 (swing_pipeline verbatim): databento front-month settles,
roll-aware via instrument_id, negative-price invalidation + ±40% winsor, weekly Friday grid
2010-06→2026-07 (~839 wk). Log price levels reconstructed causally: lp[t] = Σ_{k<t} log(1+wret[k])
(wret[t] is the FORWARD t→t+1 return, so levels are shifted — position at t uses info ≤ t only).

## Pre-named pairs (economically coherent only; 5, frozen)
GC–SI, GC–PL, SI–PL, PL–PA (precious cluster), CL–NG (energy). Copper unpaired (no coherent partner
in-universe). No other pairs will be reported.

## Signal & book construction (frozen; grids ARE the multiplicity ledger)
- **Spread**: s = ln P_A − β·ln P_B. Primary β=1 (log ratio). Variant: rolling 104-wk OLS β
  (cov/var, clipped [0.2, 5]; signal-only — execution legs stay equal-notional dollar-neutral).
- **Signal**: rolling z of s over W ∈ {26, 52, 104} wk (mean/std through t, causal). Deployed as the
  **BAND ENSEMBLE** (mean of the 3 z's) — never argmax. Continuous sizing: pair position
  = −clip(z, ±2)/2 ∈ [−1, 1] (short the rich leg, long the cheap leg, 50/50 notional).
- **Half-life gate variant**: trailing 104-wk AR(1) on the spread (Δs on lagged s); trade only if
  half-life ∈ [1, 26] wk (a spread that doesn't mean-revert in ≤2 quarters isn't an RV trade).
- **Book**: pair risk parity (inverse trailing 26-wk pair-P&L vol, bounded [0.25, 4]) across the 5
  pairs → unit gross → 10% ann vol-target overlay (26-wk trailing, cap [0.5, 2]).
- **Pre-named deployable** = band-ensemble z, β=1 ratio spreads, HALF-LIFE-GATED, pair risk parity,
  10% vol target. (OLS-β and ungated versions are robustness variants, not the pick.)

## Costs & validation (identical gates to Program 3)
12bp RT primary per name-week traded; {6, 24}bp + funding ±5bp/wk scenarios. CPCV(8, k=2, embargo
8wk); per-observation DSR at **N_trials = 20** (5 pairs + 3 z-windows + band + OLS-β + HL-gate +
combined + cost/robustness variants); CSCV PBO over the ~12-config family; drop-top-5-weeks;
per-year Sharpes.

## Adoption gate (frozen)
Deploy the pre-named book ONLY if net Sharpe positive AND DSR pass AND PBO ≤ 0.25 AND drop-top-5
positive AND the structural-break signature holds (PL–PA 2017 / GC-SI 2020 visible as losses). Else
record the negative plainly — a fourth clean kill closes RV too, and with it this universe.
