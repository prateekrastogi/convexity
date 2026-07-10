"""
INDEPENDENT verification of ctaB_pipeline.py's WINNER construction.
Written from scratch: reuses ONLY the already-validated axis-module functions
(signal_extraction, regime_hmm, vol_forecast, entry_exit_control, portfolio_construction,
leverage_policy) -- does NOT import or call any function from ctaB_pipeline.py itself.
All glue/metrics code below is independently written.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, traceback
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
LIB_DIR = f"{REPO_ROOT}/research/cta/lib"
sys.path.insert(0, LIB_DIR)
sys.path.insert(0, REPO_ROOT)

import signal_extraction as se
import entry_exit_control as eec
import regime_hmm as rh
import vol_forecast as vf
import leverage_policy as lp
import portfolio_construction as pc
from convexity.panel.candidates import load_universe

STITCHED_DIR = Path(REPO_ROOT) / "data" / "stitched"
OUT = {}

EXCLUDE_LEVERAGED = {"UVXYUSDT","SQQQUSDT","TQQQUSDT","SOXLUSDT","KORUUSDT","MVLLUSDT"}
MIN_DUKA_SPAN_DAYS = 100.0

KALMAN_PARAMS = {"q_level":1e-6, "q_slope":1e-8, "r":1e-4}
B_PARAMS = {"cost_sensitivity":0.25, "band_scale":0.5}
C_PARAMS = {"n_states":2, "refit_every":63}
D_PARAMS = {"lam":0.94}

def my_reconcile_universe():
    u = load_universe()
    eq = [d["symbol"] for d in u["equities"]]
    kept = []
    for sym in eq:
        fp = STITCHED_DIR / f"{sym}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["open_time","source"])
        duka = df[df["source"]=="duka"]
        if len(duka)==0:
            continue
        span = (duka["open_time"].max()-duka["open_time"].min())/1000.0/86400.0
        if span < MIN_DUKA_SPAN_DAYS:
            continue
        if sym in EXCLUDE_LEVERAGED:
            continue
        kept.append(sym)
    return sorted(kept), sorted(eq)

def build_name(symbol):
    t0=time.time()
    try:
        fp = STITCHED_DIR / f"{symbol}.parquet"
        df = pd.read_parquet(fp)
        df = df[df["source"]=="duka"].copy()
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        ohlc = vf.aggregate_daily_ohlc_from_duka(df)
        if len(ohlc) < 60:
            return symbol, None
        close = ohlc["close"]
        logp = np.log(close)
        simple_ret = close.pct_change()

        kf = se.kalman_local_linear_trend(logp, **KALMAN_PARAMS)
        raw_dir = se.normalize_signal(kf["slope"]).shift(1)

        log_ret = np.log(1.0+simple_ret).dropna()
        post = rh.causal_hmm_regime_posterior(log_ret, n_states=C_PARAMS["n_states"],
                                               refit_every=C_PARAMS["refit_every"], n_restarts=5)
        gate = rh.exposure_scale_from_posterior(post["p_bull"]).reindex(close.index)

        fvol = vf.ewma_vol_forecast(close, lam=D_PARAMS["lam"])
        v = fvol.dropna(); v = v[v>0]
        target_vol = float(len(v)/np.sum(1.0/v)) if len(v) else np.nan
        if not np.isfinite(target_vol) or target_vol<=0:
            target_vol = 0.01
        vol_scale = vf.vol_target_exposure_magnitude(fvol, target_daily_vol=target_vol,
                                                       leverage_target=1.0, cap_mult=2.0)

        idx = close.index
        raw_dir = raw_dir.reindex(idx); gate_r = gate.reindex(idx); vol_scale_r = vol_scale.reindex(idx)
        gated_target = (np.clip(raw_dir*gate_r,0.0,None)*vol_scale_r).fillna(0.0)

        res = eec.partial_adjustment_control(gated_target, close, cost_bps=13.0, gamma=1.0,
                                              cost_sensitivity=B_PARAMS["cost_sensitivity"],
                                              band_scale=B_PARAMS["band_scale"], external_vol=fvol)
        exposure = res["exposure"]
        dt=time.time()-t0
        print(f"[name] {symbol} ok in {dt:.1f}s n={len(ohlc)}", flush=True)
        return symbol, {"close":close, "ret":simple_ret, "exposure":exposure, "fvol":fvol, "gate":gate_r}
    except Exception as e:
        print(f"[name] {symbol} FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return symbol, None

def cagr_maxdd(net, dates):
    n=len(net)
    span_years = (dates[-1]-dates[0]).days/365.25
    eq = np.cumprod(1.0+net)
    cagr = eq[-1]**(1.0/span_years)-1.0 if eq[-1]>0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = (peak-eq)/peak
    return float(cagr), float(np.max(dd)), span_years

def main():
    t0=time.time()
    OUT["started"]=str(pd.Timestamp.now())
    names, eq_all = my_reconcile_universe()
    OUT["universe_n"]=len(names)
    OUT["aapl_in_universe"] = "AAPLUSDT" in names
    print(f"[universe] {len(names)} names, AAPL in universe={'AAPLUSDT' in names}", flush=True)

    with mp.get_context("fork").Pool(2) as pool:
        outs = pool.map(build_name, names, chunksize=1)
    cache = {s:c for s,c in outs if c is not None}
    failed = [s for s,c in outs if c is None]
    OUT["names_usable"]=len(cache); OUT["names_failed"]=failed
    print(f"[phase1] done {time.time()-t0:.1f}s usable={len(cache)} failed={failed}", flush=True)

    active = sorted(cache.keys())
    idx=None
    for s in active:
        idx = cache[s]["close"].index if idx is None else idx.union(cache[s]["close"].index)
    idx = idx.sort_values()
    RET = pd.DataFrame({s: cache[s]["ret"].reindex(idx) for s in active})
    EXP = pd.DataFrame({s: cache[s]["exposure"].reindex(idx) for s in active})
    RV  = pd.DataFrame({s: cache[s]["fvol"].reindex(idx) for s in active})
    GATE = pd.DataFrame({s: cache[s]["gate"].reindex(idx) for s in active})
    dates = RET.index
    n_obs = len(dates); span_years=(dates[-1]-dates[0]).days/365.25
    periods_per_yr = n_obs/span_years
    OUT["panel_shape"]=[n_obs, len(active)]
    OUT["periods_per_yr"]=periods_per_yr

    # fresh equal-weight buy-hold cross-check
    ew = RET.mean(axis=1, skipna=True).fillna(0.0).to_numpy()
    ew_cagr, ew_dd, _ = cagr_maxdd(ew, dates)
    OUT["fresh_bh_cagr"]=ew_cagr; OUT["fresh_bh_maxdd"]=ew_dd
    print(f"[bh] cagr={ew_cagr:.4f} maxdd={ew_dd:.4f}", flush=True)

    # Axis F winner: corr_risk_parity, corr_window=120, spike overlay
    print("[axisF] building portfolio (winner) ...", flush=True)
    tF=time.time()
    pr120 = pc.aggregate_portfolio(EXP, RET, RV, scheme="corr_risk_parity", corr_window=120, apply_spike_overlay=False)
    avgcorr120 = pc.rolling_avg_pairwise_correlation(RET, window=120)
    spike = pc.correlation_spike_scale(avgcorr120)
    weights = pr120.weights.multiply(spike, axis=0)
    print(f"[axisF] done {time.time()-tF:.1f}s", flush=True)

    # my OWN cost function (13bp trading cost only, funding=0)
    w = weights.fillna(0.0).to_numpy()
    r = RET.reindex(columns=weights.columns).fillna(0.0).to_numpy()
    dw = np.diff(w, axis=0, prepend=w[[0]]); dw[0,:]=w[0,:]
    trad_cost = (13.0/1e4)*np.abs(dw)
    net_per_name = w*r - trad_cost
    book_1x = np.nansum(net_per_name, axis=1)
    gross_1x = np.nansum(w*r, axis=1)

    # my OWN CDaR leverage application (reuse lp.cdar_constrained_leverage, the axis-E primitive)
    finite = book_1x[np.isfinite(book_1x)]
    cdar = lp.cdar_constrained_leverage(finite, budget=0.25, alpha=0.95)
    L = cdar["L_closed_form"]
    final_net = L*np.nan_to_num(book_1x)
    final_gross = L*np.nan_to_num(gross_1x)

    cagr, maxdd, sy = cagr_maxdd(final_net, dates)
    mu, sd = np.mean(final_net), np.std(final_net, ddof=1)
    sharpe = mu/sd*np.sqrt(periods_per_yr)
    worst_day = float(np.min(final_net))
    ruin = bool(np.any(1.0+final_net<=0))

    OUT["winner_hand_derived"] = {
        "L_cdar": L, "cdar_diag": {k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in cdar.items()},
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "worst_day": worst_day, "ruin": ruin,
        "span_years": sy, "n_obs": n_obs,
    }
    print(f"[winner] L={L:.4f} CAGR={cagr:.4f} maxDD={maxdd:.4f} Sharpe={sharpe:.4f} worst_day={worst_day:.4f}", flush=True)

    # cost scenarios
    for bp in (6.5, 26.0):
        tc = (bp/1e4)*np.abs(dw)
        b1x = np.nansum(w*r - tc, axis=1)
        c,_,_ = cagr_maxdd(L*np.nan_to_num(b1x), dates)
        OUT[f"winner_cagr_{bp}bp"] = c
        print(f"[cost] {bp}bp -> CAGR={c:.4f}", flush=True)

    # drop-top-5
    order = np.argsort(final_net)[::-1]
    top5 = order[:5]
    drop5 = final_net.copy(); drop5[top5]=0.0
    c5, d5, _ = cagr_maxdd(drop5, dates)
    OUT["drop_top5"] = {"dates":[str(dates[i].date()) for i in top5], "cagr_after":c5, "maxdd_after":d5}
    print(f"[drop5] dates={OUT['drop_top5']['dates']} cagr_after={c5:.4f} maxdd_after={d5:.4f}", flush=True)

    # exposure path at gate5 dates -- mean gate + deployed exposure
    mean_gate = GATE.mean(axis=1, skipna=True)
    gross_exposure = L*weights.abs().sum(axis=1)
    def pull(window):
        s,e = window
        sub_g = mean_gate.loc[(mean_gate.index>=pd.Timestamp(s,tz="UTC"))&(mean_gate.index<=pd.Timestamp(e,tz="UTC"))]
        sub_x = gross_exposure.loc[(gross_exposure.index>=pd.Timestamp(s,tz="UTC"))&(gross_exposure.index<=pd.Timestamp(e,tz="UTC"))]
        return {str(d.date()):float(v) for d,v in sub_g.items()}, {str(d.date()):float(v) for d,v in sub_x.items()}
    covid_gate, covid_exp = pull(("2020-01-15","2020-05-01"))
    squeeze_gate, squeeze_exp = pull(("2025-03-15","2025-04-25"))
    OUT["covid_gate"]=covid_gate; OUT["squeeze_gate"]=squeeze_gate
    OUT["covid_exposure"]=covid_exp; OUT["squeeze_exposure"]=squeeze_exp

    # melt-up decomposition
    idxr = pd.DatetimeIndex(dates)
    ser = pd.Series(np.nan_to_num(final_net), index=idxr)
    meltup_window = ser.loc[(idxr>=pd.Timestamp("2023-01-01",tz="UTC"))&(idxr<=pd.Timestamp("2024-12-31",tz="UTC"))]
    ret_meltup = float(np.prod(1.0+meltup_window.to_numpy())-1.0)
    total_full = (1.0+cagr)**sy - 1.0
    total_ex = (1.0+total_full)/(1.0+ret_meltup)-1.0
    cagr_ex = (1.0+total_ex)**(1.0/(sy-1.999))-1.0
    OUT["meltup"] = {"ret_meltup":ret_meltup, "cagr_full":cagr, "cagr_ex_meltup":cagr_ex}
    print(f"[meltup] ret={ret_meltup:.4f} cagr_ex={cagr_ex:.4f}", flush=True)

    # AAPL specific data-bug materiality check
    if "AAPLUSDT" in active:
        aapl_ret = cache["AAPLUSDT"]["ret"].reindex(dates)
        aapl_w = weights["AAPLUSDT"].reindex(dates)
        aapl_contrib = (aapl_w.fillna(0.0)*aapl_ret.fillna(0.0))
        worst_aapl_day = aapl_ret.idxmin()
        OUT["aapl_bug_check"] = {
            "worst_aapl_return": float(aapl_ret.min()),
            "worst_aapl_date": str(worst_aapl_day.date()),
            "aapl_weight_that_day": float(aapl_w.reindex([worst_aapl_day]).iloc[0]) if worst_aapl_day in aapl_w.index else None,
            "aapl_contrib_to_book_that_day": float(aapl_contrib.reindex([worst_aapl_day]).iloc[0]) if worst_aapl_day in aapl_contrib.index else None,
            "book_return_that_day": float(pd.Series(final_net, index=dates).reindex([worst_aapl_day]).iloc[0]) if worst_aapl_day in dates else None,
        }
        print(f"[aapl] {OUT['aapl_bug_check']}", flush=True)

    # Kelly / EVT / GZ arithmetic re-derivation on winner's book_1x_13 (pre-leverage) returns
    kelly = lp.kelly_optimal_leverage(finite, l_max=25.0)
    half_k = lp.fractional_kelly_leverage(max(kelly["L_star"],1e-6), 0.5)
    quarter_k = lp.fractional_kelly_leverage(max(kelly["L_star"],1e-6), 0.25)
    evt = lp.evt_ceiling_leverage(finite, threshold_quantile=0.95, breach_years=10.0)
    gz_tight = lp.grossman_zhou_exposure(np.nan_to_num(book_1x), l_base=max(kelly["L_star"],1e-6), alpha_floor=0.75, rt_bp=13.0)
    gz_loose = lp.grossman_zhou_exposure(np.nan_to_num(book_1x), l_base=max(kelly["L_star"],1e-6), alpha_floor=0.50, rt_bp=13.0)
    OUT["leverage_arithmetic"] = {
        "kelly_L_star": kelly["L_star"], "half_kelly_L": half_k, "quarter_kelly_L": quarter_k,
        "evt_L_ceiling": evt.get("L_ceiling"), "cdar_L": L,
        "gz_tight_avg_exposure": float(np.mean(np.abs(np.asarray(gz_tight["exposure"])))) if "exposure" in gz_tight else None,
        "gz_loose_avg_exposure": float(np.mean(np.abs(np.asarray(gz_loose["exposure"])))) if "exposure" in gz_loose else None,
    }
    print(f"[leverage] {OUT['leverage_arithmetic']}", flush=True)

    OUT["total_wallclock_sec"] = time.time()-t0
    with open(f"{REPO_ROOT}/research/cta/_verify_ctaB_results.json","w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("DONE", flush=True)

if __name__=="__main__":
    main()
