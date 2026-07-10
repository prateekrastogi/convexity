"""
Isolate WHY the full-chain injection test reacted at index 253 (far before the shock at ~996-1001)
on SPYUSDT/MSFTUSDT. Hypothesis: build_name_exposure()'s per-name `target_daily_vol` (harmonic
mean of the ENTIRE fvol.dropna() series, i.e. past+future) is a single GLOBAL constant applied
uniformly to vol_target_exposure_magnitude() at every t -- so a shock ANYWHERE in the sample
changes vol_scale[t] for ALL t, including t far before the shock. This script isolates that
component specifically (not the HMM gate, not the raw_dir signal) to confirm/deny.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO_ROOT}/research/cta/lib")
sys.path.insert(0, REPO_ROOT)
import vol_forecast as vf
import regime_hmm as rh

STITCHED = Path(REPO_ROOT)/"data"/"stitched"

def load_close(symbol):
    df = pd.read_parquet(STITCHED/f"{symbol}.parquet")
    df = df[df["source"]=="duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = vf.aggregate_daily_ohlc_from_duka(df)
    return ohlc["close"]

close = load_close("SPYUSDT")
n = len(close)
shock_idx = n // 2
base = close.copy()
shocked = close.copy()
shocked.iloc[shock_idx:] = shocked.iloc[shock_idx:] * 1.5

fvol_base = vf.ewma_vol_forecast(base, lam=0.94)
fvol_shock = vf.ewma_vol_forecast(shocked, lam=0.94)

# 1) confirm fvol ITSELF is causal (should be identical before shock_idx+1)
fv_diff = (fvol_base - fvol_shock).abs()
fv_first_react = int(np.where(fv_diff.to_numpy() > 1e-9)[0][0]) if (fv_diff > 1e-9).any() else None
print(f"[fvol per-day] first_react={fv_first_react} (shock_idx={shock_idx}) "
      f"causal={'OK' if (fv_first_react is None or fv_first_react >= shock_idx) else 'FAIL'}")

# 2) compute the harmonic-mean target_daily_vol EXACTLY as build_name_exposure() does
def harmonic_target(fvol):
    v = fvol.dropna(); v = v[v > 0]
    return float(len(v) / np.sum(1.0 / v))

tvol_base = harmonic_target(fvol_base)
tvol_shock = harmonic_target(fvol_shock)
print(f"[target_daily_vol] base={tvol_base:.8f} shocked={tvol_shock:.8f} "
      f"rel_diff={(tvol_shock-tvol_base)/tvol_base:.6%}")

# 3) vol_scale at an EARLY date (well before shock_idx) using each target_daily_vol
vol_scale_base = vf.vol_target_exposure_magnitude(fvol_base, target_daily_vol=tvol_base, leverage_target=1.0, cap_mult=2.0)
vol_scale_shock = vf.vol_target_exposure_magnitude(fvol_shock, target_daily_vol=tvol_shock, leverage_target=1.0, cap_mult=2.0)
early_idx = 300  # well before shock_idx, well after HMM min_train=252
print(f"[vol_scale @ idx {early_idx}, well before shock] base={vol_scale_base.iloc[early_idx]:.8f} "
      f"shocked={vol_scale_shock.iloc[early_idx]:.8f} "
      f"diff={vol_scale_shock.iloc[early_idx]-vol_scale_base.iloc[early_idx]:.3e}")

# 4) Confirm this vol_scale diff propagates at EVERY early index, not just one
diffs = (vol_scale_shock - vol_scale_base).abs()
early_diffs = diffs.iloc[260:shock_idx]
print(f"[vol_scale diff over idx 260..{shock_idx}] max={early_diffs.max():.3e} "
      f"min_nonzero={early_diffs[early_diffs>0].min() if (early_diffs>0).any() else None:.3e} "
      f"n_nonzero={(early_diffs>1e-9).sum()} / {len(early_diffs)}")

# 5) Cross-check: HMM gate ALONE (isolated) is causal (own injection_test on real log returns)
simple_ret = close.pct_change()
log_ret = np.log(1.0+simple_ret).dropna()
hmm_test = rh.injection_test(log_ret, injection_idx=shock_idx-1, shock=0.4, n_states=2, refit_every=63, n_restarts=3)
print("[HMM standalone injection_test]", json.dumps(hmm_test, indent=2))

out = {
    "fvol_first_react": fv_first_react, "shock_idx": shock_idx,
    "target_daily_vol_base": tvol_base, "target_daily_vol_shocked": tvol_shock,
    "target_daily_vol_rel_diff": (tvol_shock-tvol_base)/tvol_base,
    "vol_scale_diff_at_idx300": float(vol_scale_shock.iloc[early_idx]-vol_scale_base.iloc[early_idx]),
    "vol_scale_early_window_max_diff": float(early_diffs.max()),
    "vol_scale_early_window_n_nonzero": int((early_diffs>1e-9).sum()),
    "vol_scale_early_window_n_total": int(len(early_diffs)),
    "hmm_standalone_injection_test": hmm_test,
}
with open(f"{REPO_ROOT}/research/cta/_diagnose_lookahead_results.json","w") as f:
    json.dump(out, f, indent=2, default=str)
print("DONE_DIAGNOSE", flush=True)
