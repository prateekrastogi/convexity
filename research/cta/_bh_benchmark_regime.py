import sys
sys.path.insert(0, "/home/speculation/convexity-cta/research/cta")
import numpy as np
import pandas as pd
import ctaB_pipeline as cb

names = cb.reconcile_universe()
print(f"universe: {len(names)} names")

frames = {}
for s in names:
    f = cb.load_name_frames(s)
    if f is not None:
        frames[s] = f["simple_ret"]

# build a point-in-time equal-weight daily return panel
all_ret = pd.DataFrame(frames)
# equal-weight across names ACTIVE that day (non-NaN), point-in-time (no backfill: NaN before a
# name's first real day acts as "not yet listed", pandas .mean(axis=1) with skipna handles this)
bh_ret = all_ret.mean(axis=1, skipna=True).dropna()
dates = bh_ret.index

print(f"benchmark n_obs={len(bh_ret)}, span={(dates[-1]-dates[0]).days/365.25:.3f}yr")

cagr, maxdd = cb._cagr_from_returns(bh_ret.values, dates)
sharpe = cb.per_obs_sharpe(bh_ret.values) * np.sqrt(cb.derive_periods_per_yr(dates))
print(f"benchmark full-sample: CAGR={cagr:.4%} maxDD={maxdd:.4%} Sharpe={sharpe:.4f}")

regimes = cb.regime_window_returns(bh_ret.values, dates)
print()
print("=== benchmark regime_window_returns ===")
for k, v in regimes.items():
    print(f"  {k}: {v:.4%}")

peryear = cb.per_year_return(bh_ret.values, dates)
print()
print("=== benchmark per_year_return ===")
for k, v in peryear.items():
    print(f"  {k}: {v:.4%}")
