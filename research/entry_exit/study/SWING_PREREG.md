# SWING XSEC COMMODITY — PRE-REGISTRATION (FROZEN 2026-07-05, executed same day)

Frozen from SWING_PREREG_DRAFT v0.1 after all data landed. Executed by
research/entry_exit/study/swing_pipeline.py; outcome in REPORT.md (verdict: no deployable edge on
the 7-perp universe — a clean, verified negative). Frozen for the record so the result is auditable.

## Deviations from the draft (declared, all pre-outcome)
- **Price basis = databento front-month daily settles (rank 0), roll-aware via instrument_id**, NOT
  the 1-min duka re-stitch + perp splice. Reason: databento gives a uniform 2010-06+ panel for all 7
  roots (breadth 7 throughout, 839 wk) — longer/cleaner than the duka stitch (per-name floors; XPT/XPD
  only 2021). Perp = execution vehicle; tracking error negligible at swing horizon, covered by the
  cost model. Duka extension retained but unused by this daily study.
- **Data hygiene (added before any factor claim)**: a return computed through a non-positive price is
  invalidated (WTI 2020-04-20 = −$2.67); daily returns winsorized to ±40% (no legitimate daily move
  for these 7 reaches it). This corrected a contamination that had faked a +8bp carry spread.

## Universe / calendar
7 Binance perps → CME/COMEX/NYMEX roots: CL, NG, GC, SI, HG, PL, PA. Weekly Friday-close UTC grid,
2010-06-11→2026-07-03 (839 wk, breadth 7). COT joined at usable_from ≤ Friday (report+3d release lag).

## Factor ledger (frozen; N_eff = 13 = 10 factor variants + 3 ensembles)
F1 carry (rank0→rank1 annualized log-slope by data-derived per-root roll spacing; long
backwardated-2 / short contango-2), F2 basis-momentum (4wk Δcarry), F3 1wk reversal (−prev-week
return, conditioned on COT |Δmm_net|/OI top-half; + an unconditional variant), F4 momentum band
{6,8,10}wk (top-2/bottom-2), F5 realized skew {40,60}d (long negative-skew), F6 mom8 × hedging-
pressure conditioner. All: inverse-vol legs (20d daily, floored at 0.5× cross-sectional median),
dollar-neutral gross 1.0. Ensembles: {F3+F4+F5}, {+F6}, {+F1+F2 all}, equal-risk mean + 10%
vol-target overlay (scale capped [0.5,2]).

## Costs & validation
Net of 12bp RT primary per name-week traded ({6,24}bp + funding ±5bp/wk scenarios reported). CPCV(8
groups, k=2, embargo 6wk) mean-path + frac_pos; per-observation DSR at N_eff=13; drop-top-5-weeks;
per-year. No-lookahead by construction (weight_t from info ≤ t, scored t→t+1).

## Adoption gate (frozen, pre-outcome)
Deploy the pre-named ensemble ONLY if it beats the incumbent (do-not-trade) at net Sharpe with DSR
pass + PBO ≤ 0.25 + drop-top-5 positive + CPCV frac_pos > 0.5. **OUTCOME: no ensemble cleared any
gate (all net-negative, DSR ≈ 0.00). Recorded negative; nothing deployed.**
