"""
Independent synthetic-outlier-injection test on the FULL frozen per-name exposure chain
(Axis A kalman+normalize+shift -> Axis C HMM gate -> Axis D ewma vol-target scale -> Axis B
partial-adjustment control), run on REAL SPYUSDT and MSFTUSDT daily closes -- exactly the chain
build_name_exposure() in ctaB_pipeline.py uses for the frozen construction, but written from
scratch here (own glue), calling only the already-validated axis-module functions.
"""
from __future__ import annotations
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
import signal_extraction as se
import entry_exit_control as eec
import regime_hmm as rh
import vol_forecast as vf

STITCHED = Path(REPO_ROOT)/"data"/"stitched"
KALMAN_PARAMS = {"q_level":1e-6, "q_slope":1e-8, "r":1e-4}

def load_close(symbol):
    df = pd.read_parquet(STITCHED/f"{symbol}.parquet")
    df = df[df["source"]=="duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = vf.aggregate_daily_ohlc_from_duka(df)
    return ohlc["close"]

def full_chain(close: pd.Series):
    logp = np.log(close)
    simple_ret = close.pct_change()
    kf = se.kalman_local_linear_trend(logp, **KALMAN_PARAMS)
    raw_dir = se.normalize_signal(kf["slope"]).shift(1)
    log_ret = np.log(1.0+simple_ret).dropna()
    post = rh.causal_hmm_regime_posterior(log_ret, n_states=2, refit_every=63, n_restarts=3)
    gate = rh.exposure_scale_from_posterior(post["p_bull"]).reindex(close.index)
    fvol = vf.ewma_vol_forecast(close, lam=0.94)
    v = fvol.dropna(); v=v[v>0]
    target_vol = float(len(v)/np.sum(1.0/v)) if len(v) else 0.01
    vol_scale = vf.vol_target_exposure_magnitude(fvol, target_daily_vol=target_vol, leverage_target=1.0, cap_mult=2.0)
    idx = close.index
    raw_dir=raw_dir.reindex(idx); gate_r=gate.reindex(idx); vol_scale_r=vol_scale.reindex(idx)
    gated_target = (np.clip(raw_dir*gate_r,0.0,None)*vol_scale_r).fillna(0.0)
    res = eec.partial_adjustment_control(gated_target, close, cost_bps=13.0, gamma=1.0,
                                          cost_sensitivity=0.25, band_scale=0.5, external_vol=fvol)
    return gated_target, res["exposure"]

def naive_unlagged_chain(close: pd.Series):
    """Deliberately WRONG control: skip the raw_dir outer .shift(1) (uses today's own kalman slope
    directly against yesterday's std -- a same-day leak), to prove the injection test has teeth."""
    logp = np.log(close)
    simple_ret = close.pct_change()
    kf = se.kalman_local_linear_trend(logp, **KALMAN_PARAMS)
    raw_dir_naive = se.normalize_signal(kf["slope"])   # NO outer shift(1) -- BUG for test purposes
    fvol = vf.ewma_vol_forecast(close, lam=0.94)
    v = fvol.dropna(); v=v[v>0]
    target_vol = float(len(v)/np.sum(1.0/v)) if len(v) else 0.01
    vol_scale = vf.vol_target_exposure_magnitude(fvol, target_daily_vol=target_vol, leverage_target=1.0, cap_mult=2.0)
    idx = close.index
    gated_naive = (np.clip(raw_dir_naive.reindex(idx),0.0,None)*vol_scale.reindex(idx)).fillna(0.0)
    return gated_naive

def run_test(symbol):
    close = load_close(symbol)
    n = len(close)
    shock_idx = n // 2
    base = close.copy()
    shocked = close.copy()
    # permanent +50% multiplicative level shock from shock_idx onward
    shocked.iloc[shock_idx:] = shocked.iloc[shock_idx:] * 1.5

    gt_base, exp_base = full_chain(base)
    gt_shock, exp_shock = full_chain(shocked)
    gn_base = naive_unlagged_chain(base)
    gn_shock = naive_unlagged_chain(shocked)

    def first_diff(a, b, atol=1e-9):
        d = (a - b).abs()
        idxs = np.where(d.to_numpy() > atol)[0]
        return int(idxs[0]) if len(idxs) else None

    idx_target = first_diff(gt_base, gt_shock)
    idx_exposure = first_diff(exp_base, exp_shock)
    idx_naive = first_diff(gn_base, gn_shock)

    out = {
        "symbol": symbol, "n": n, "shock_idx": shock_idx,
        "gated_target_first_react": idx_target,
        "exposure_first_react": idx_exposure,
        "naive_unlagged_first_react": idx_naive,
        "gated_target_causal_ok": (idx_target is None) or (idx_target >= shock_idx),
        "exposure_causal_ok": (idx_exposure is None) or (idx_exposure >= shock_idx),
        "naive_reacts_strictly_before_shock": (idx_naive is not None) and (idx_naive < shock_idx),
    }
    out["overall_pass"] = bool(out["gated_target_causal_ok"] and out["exposure_causal_ok"] and out["naive_reacts_strictly_before_shock"])
    print(json.dumps(out, indent=2), flush=True)
    return out

if __name__=="__main__":
    results = {}
    for sym in ["SPYUSDT", "MSFTUSDT"]:
        results[sym] = run_test(sym)
    with open(f"{REPO_ROOT}/research/cta/_injection_full_chain_results.json","w") as f:
        json.dump(results, f, indent=2)
    print("DONE_INJECTION", flush=True)
