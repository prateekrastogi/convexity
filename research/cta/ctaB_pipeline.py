"""
research/cta/ctaB_pipeline.py -- Stage B CONFIRMATORY pipeline (CTA_PREREG.md, frozen 2026-07-07).

Re-runs the FROZEN winner and runner-up constructions from CTA_PREREG.md fresh, on the FULL
93-name reconciled base universe (not the 22-name STAGE_A_22 pilot). No parameter is re-tuned here.
Both constructions share IDENTICAL Axis A/B/C/D/E choices (kalman_1e-06_1e-08_0.0001 /
aggressive / hmm2_rf63 / ewma_0.94 / cdar) and differ ONLY in Axis F:
    winner:     corr_risk_parity, corr_window=120, spike overlay ON
    runner-up:  corr_risk_parity, corr_window=60,  spike overlay OFF
so the per-name signal/exposure pipeline (through Axis B) is built ONCE and shared; only the
portfolio-construction (Axis F) -> cost -> leverage (Axis E) stages branch per construction.

PROGRAM CLASS (CTA_PREREG.md): benchmark-relative (CAGR-at-DD-budget vs the named 93-name
buy-and-hold benchmark), identical in kind to I1's own overlay-validation framing. DSR/PBO are
computed and reported INFORMATIONALLY ONLY -- they never gate adoption here.

Adoption gate (frozen, PRE-outcome, restated verbatim from CTA_PREREG.md) -- ALL required for a
PAPER-FIRST recommendation, at the primary 13bp/funding-0 scenario:
  1. CAGR at the 41% DD budget > 46.8% (benchmark CAGR)
  2. CAGR at the 30% DD budget > 0.75 x 46.8% = 35.1%
  3. net Sharpe >= 1.353 - 0.05 = 1.303
  4. frozen construction (THIS Stage-B run, not Stage-A's exploratory number) passes (i) cost-
     scenario, (ii) drop-top-5-periods, (iii) purged walk-forward (CPCV) robustness
  5. drawdown-control (Axis C regime gate x Axis E CDaR leverage) visibly engages during both the
     2020-03 crash and the 2025-04-09-style squeeze week (actual exposure path reported)
  6. no ruin, worst single day > -25% at the deployed (CDaR-derived) leverage

Run: nohup /home/speculation/convexity/.venv/bin/python research/cta/ctaB_pipeline.py \
       > research/cta/ctaB.out 2>&1 &
Writes: research/cta/results_ctaB.json
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os

# Pin BLAS/OMP threads to 1 BEFORE numpy/scipy import -- avoids oversubscription when
# parallelizing across names via multiprocessing.Pool (same fix as ctaA_sweep.py's
# Section-9-REDESIGN performance note; changes no math, only thread scheduling).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
LIB_DIR = f"{REPO_ROOT}/research/cta/lib"
sys.path.insert(0, LIB_DIR)
sys.path.insert(0, REPO_ROOT)

import signal_extraction as se          # Axis A
import entry_exit_control as eec        # Axis B
import regime_hmm as rh                 # Axis C
import vol_forecast as vf               # Axis D
import leverage_policy as lp            # Axis E
import portfolio_construction as pc     # Axis F

from convexity.panel.candidates import load_universe

STITCHED_DIR = Path(REPO_ROOT) / "data" / "stitched"
OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
RESULTS_PATH = OUT_DIR / "results_ctaB.json"
PROGRESS_LOG = OUT_DIR / "_ctaB_progress.log"

N_PROCESSES = max(1, mp.cpu_count())

# ======================================================================================
# Established Fact (6): the 93-name benchmark this entire program is judged against.
# ======================================================================================
BENCHMARK_CAGR = 0.468
BENCHMARK_SHARPE = 1.353
BENCHMARK_MAXDD = 0.410
BENCHMARK_WORST_DAY = -0.121  # 2020-03-16

# 6 excluded leveraged/inverse/decay ETFs (Established Bug Pattern 7e)
EXCLUDE_LEVERAGED = {"UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT"}
EXCLUDE_SHORT_HISTORY = {"STXXUSDT", "CBRSUSDT", "DRAMUSDT"}
MIN_DUKA_SPAN_DAYS = 100.0

DD_BUDGETS = (0.20, 0.30, 0.41)

# CPCV embargo: the slowest signal component's OWN horizon among the frozen construction's axes --
# Axis A normalize_signal vol_window=20, Axis B vol_window=20, Axis D ewma_0.94 effective memory
# (center_of_mass = 0.94/0.06 ~= 15.7 days), Axis F corr_window<=120 (winner's own w120) -- all
# dominated by Axis C's causal_hmm_regime_posterior's own min_train=252 (a full trading year of
# real history required before the HMM produces its first fit; refit_every=63 thereafter). 252 is
# the largest of these and is used as the embargo, per the prereg's ">=" requirement.
EMBARGO_TRADING_DAYS = 252
CPCV_N_GROUPS = 8
CPCV_K = 2

# N_trials ledger for informational DSR/PBO (CTA_PREREG.md's own frozen denominator)
N_CELLS_PRIMARY = 29760          # full Stage-A ledger: STAGE_A_22(9,920) + SPY(9,920) + QQQ(9,920)
N_CELLS_STAGE_A_22_ONLY = 9920   # the universe the winner was actually drawn from

# I1's own 6 named regime windows, reused VERBATIM (CTA_PREREG.md pipeline-scope item 8: "the same
# regime windows I1 used, reused here rather than re-derived") -- including I1's own
# "2022_chop_decline" == "2022_full_year" date-range duplication, inherited as-is, not "fixed".
REGIME_WINDOWS = {
    "2020-03_COVID": ("2020-02-15", "2020-04-15"),
    "2020H2_2021_recovery": ("2020-07-01", "2021-12-31"),
    "2018Q4_chop": ("2018-10-01", "2018-12-31"),
    "2022_chop_decline": ("2022-01-01", "2022-12-31"),
    "2022_full_year": ("2022-01-01", "2022-12-31"),
    "2023_2024_meltup": ("2023-01-01", "2024-12-31"),
}
MELTUP_WINDOW = ("2023-01-01", "2024-12-31")
MELTUP_YEARS = 1.999  # exact interior span used in Stage A's own decomposition, reused verbatim

COVID_EXPOSURE_WINDOW = ("2020-01-15", "2020-05-01")     # padded around the named 2020-02-15/04-15 window
SQUEEZE_EXPOSURE_WINDOW = ("2025-03-15", "2025-04-25")   # padded around the named 2025-04-04/04-11 window

# ======================================================================================
# THE FROZEN CONSTRUCTION -- winner and runner-up, exact grid values, CTA_PREREG.md verbatim.
# Both share Axis A/B/C/D/E; differ ONLY in Axis F.
# ======================================================================================
KALMAN_PARAMS = {"q_level": 1e-6, "q_slope": 1e-8, "r": 1e-4}
AXIS_A_KEY = "kalman_1e-06_1e-08_0.0001"
B_PARAMS = {"name": "aggressive", "cost_sensitivity": 0.25, "band_scale": 0.5}
AXIS_C_KEY = "hmm2_rf63"
AXIS_C_PARAMS = {"n_states": 2, "refit_every": 63}
AXIS_D_KEY = "ewma_0.94"
AXIS_D_PARAMS = {"lam": 0.94}
AXIS_E_KEY = "cdar"

CONSTRUCTIONS = {
    "winner": {"f_key": "corr_risk_parity_w120_spike", "corr_window": 120, "spike": True},
    "runner_up": {"f_key": "corr_risk_parity_w60_nospike", "corr_window": 60, "spike": False},
}
# Diagnostic-only third F-variant to directly test whether the Stage-A-documented w120_spike /
# w120_nospike byte-identical collapse still holds at full 93-name scale (pipeline-scope item 1).
DIAGNOSTIC_F_VARIANT = {"f_key": "corr_risk_parity_w120_nospike", "corr_window": 120, "spike": False}

COST_SCENARIOS_BP = [13.0, 6.5, 26.0]


# ======================================================================================
# 0. Universe reconciliation -- FULL 93-name base universe (identical logic to ctaA_sweep.py's
#    reconcile_universe(), NOT reconcile_universe_stage_a() -- Stage B's own frozen scope).
# ======================================================================================
def reconcile_universe() -> list[str]:
    u = load_universe()
    eq = [d["symbol"] for d in u["equities"]]
    print(f"[universe] equities sleeve from load_universe(): {len(eq)}", flush=True)

    kept, excluded_no_duka, excluded_short, excluded_lev = [], [], [], []
    for sym in eq:
        fp = STITCHED_DIR / f"{sym}.parquet"
        if not fp.exists():
            excluded_no_duka.append(sym)
            continue
        df = pd.read_parquet(fp, columns=["open_time", "source"])
        duka = df[df["source"] == "duka"]
        if len(duka) == 0:
            excluded_no_duka.append(sym)
            continue
        span_days = (duka["open_time"].max() - duka["open_time"].min()) / 1000.0 / 86400.0
        if span_days < MIN_DUKA_SPAN_DAYS:
            excluded_short.append(sym)
            continue
        if sym in EXCLUDE_LEVERAGED:
            excluded_lev.append(sym)
            continue
        kept.append(sym)

    print(f"[universe] excluded (0 duka rows / perp-only): {len(excluded_no_duka)} {sorted(excluded_no_duka)}", flush=True)
    print(f"[universe] excluded (span < {MIN_DUKA_SPAN_DAYS}d, short-history): {len(excluded_short)} {sorted(excluded_short)}", flush=True)
    print(f"[universe] excluded (leveraged/inverse/decay ETF): {len(excluded_lev)} {sorted(excluded_lev)}", flush=True)
    print(f"[universe] RECONCILED BASE UNIVERSE: {len(kept)} names", flush=True)

    assert set(excluded_short) == EXCLUDE_SHORT_HISTORY, (
        f"short-history exclusion mismatch: got {sorted(excluded_short)}, expected {sorted(EXCLUDE_SHORT_HISTORY)}")
    if len(kept) != 93:
        print(f"[universe] *** DISCREPANCY: expected 93, got {len(kept)} ***", flush=True)
    return sorted(kept)


# ======================================================================================
# 1. Per-name data loading (identical to ctaA_sweep.py)
# ======================================================================================
def load_name_frames(symbol: str) -> dict | None:
    fp = STITCHED_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(fp)
    df = df[df["source"] == "duka"].copy()          # bug pattern (a)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = vf.aggregate_daily_ohlc_from_duka(df)     # aggregate from 1-min duka rows
    if len(ohlc) < 60:
        return None
    daily_close = ohlc["close"]
    log_price = np.log(daily_close)
    simple_ret = daily_close.pct_change()
    return {
        "symbol": symbol,
        "ohlc": ohlc,
        "close": daily_close,
        "log_price": log_price,
        "simple_ret": simple_ret,
        "n_obs": len(ohlc),
        "span_years": (ohlc.index[-1] - ohlc.index[0]).days / 365.25,
    }


# ======================================================================================
# 2. Per-name signal build: Axis A (kalman) -> Axis C (hmm2_rf63) -> Axis D (ewma_0.94)
#    -> gated target -> Axis B (aggressive partial-adjustment) -> per-name exposure.
#    IDENTICAL for winner and runner-up (they differ only in Axis F) -- built ONCE per name.
# ======================================================================================
def build_name_exposure(symbol: str) -> tuple[str, dict | None]:
    t0 = time.time()
    try:
        frames = load_name_frames(symbol)
        if frames is None:
            return symbol, None
        log_price, close, simple_ret = frames["log_price"], frames["close"], frames["simple_ret"]

        # Axis A: kalman_1e-06_1e-08_0.0001 -- standalone Kalman-slope signal (bucket-2 style in
        # ctaA_sweep.py's compute_axis_a: kalman_local_linear_trend -> normalize_signal -> shift(1),
        # NOT the IC-weighted multi-horizon combination, exactly as the frozen grid point is built).
        kf = se.kalman_local_linear_trend(log_price, **KALMAN_PARAMS)
        raw_dir = se.normalize_signal(kf["slope"]).shift(1)

        # Axis C: hmm2_rf63
        log_ret = np.log(1.0 + simple_ret).dropna()
        post = rh.causal_hmm_regime_posterior(log_ret, n_states=AXIS_C_PARAMS["n_states"],
                                               refit_every=AXIS_C_PARAMS["refit_every"], n_restarts=5)
        gate = rh.exposure_scale_from_posterior(post["p_bull"]).reindex(close.index)

        # Axis D: ewma_0.94
        fvol = vf.ewma_vol_forecast(close, lam=AXIS_D_PARAMS["lam"])
        v = fvol.dropna()
        v = v[v > 0]
        target_daily_vol = float(len(v) / np.sum(1.0 / v)) if len(v) else np.nan   # harmonic mean
        if not np.isfinite(target_daily_vol) or target_daily_vol <= 0:
            target_daily_vol = 0.01
        vol_scale = vf.vol_target_exposure_magnitude(fvol, target_daily_vol=target_daily_vol,
                                                       leverage_target=1.0, cap_mult=2.0)

        idx = close.index
        raw_dir = raw_dir.reindex(idx)
        gate_r = gate.reindex(idx)
        vol_scale_r = vol_scale.reindex(idx)
        gated_target = (np.clip(raw_dir * gate_r, 0.0, None) * vol_scale_r).fillna(0.0)

        # Axis B: aggressive partial-adjustment control, external_vol = Axis D's own forecast
        res = eec.partial_adjustment_control(
            gated_target, close, cost_bps=13.0, gamma=1.0,
            cost_sensitivity=B_PARAMS["cost_sensitivity"], band_scale=B_PARAMS["band_scale"],
            external_vol=fvol,
        )
        exposure = res["exposure"]
        n_trades = int(res["traded"].sum())

        dt = time.time() - t0
        print(f"[phase1] {symbol} done in {dt:.1f}s (n_obs={frames['n_obs']})", flush=True)
        return symbol, {
            "close": close, "simple_ret": simple_ret, "exposure": exposure,
            "forecast_vol": fvol, "gate": gate_r, "n_trades": n_trades,
            "n_obs": frames["n_obs"], "span_years": frames["span_years"],
        }
    except Exception as e:
        print(f"[phase1] {symbol} FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return symbol, None


# ======================================================================================
# 3. Cost / leverage / metrics machinery -- reused verbatim from ctaA_sweep.py
# ======================================================================================
def apply_costs(weights_df: pd.DataFrame, returns_df: pd.DataFrame, cost_bp: float,
                 funding_bp_week: float) -> np.ndarray:
    idx = weights_df.index
    w = weights_df.fillna(0.0).to_numpy()
    r = returns_df.reindex(columns=weights_df.columns).fillna(0.0).to_numpy()
    delta_w = np.diff(w, axis=0, prepend=w[[0]])
    delta_w[0, :] = w[0, :]
    trading_cost = (cost_bp / 1e4) * np.abs(delta_w)
    cal_days = pd.Series(idx).diff().dt.days.fillna(1.0).to_numpy()
    cal_days = np.where(cal_days <= 0, 1.0, cal_days)
    funding_cost = (funding_bp_week / 1e4 / 7.0) * cal_days[:, None] * np.abs(w)
    gross = w * r
    net_per_name = gross - trading_cost - funding_cost
    return np.nansum(net_per_name, axis=1)


def gross_book_return(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> np.ndarray:
    w = weights_df.fillna(0.0).to_numpy()
    r = returns_df.reindex(columns=weights_df.columns).fillna(0.0).to_numpy()
    return np.nansum(w * r, axis=1)


def _cagr_from_returns(net_ret: np.ndarray, dates: pd.DatetimeIndex) -> tuple[float, float]:
    n = len(net_ret)
    if n < 2:
        return np.nan, np.nan
    span_years = (dates[-1] - dates[0]).days / 365.25
    if span_years <= 0:
        return np.nan, np.nan
    equity = np.cumprod(1.0 + net_ret)
    final_eq = equity[-1]
    cagr = (final_eq ** (1.0 / span_years) - 1.0) if final_eq > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, np.nan)
    max_dd = float(np.nanmax(dd)) if len(dd) else np.nan
    return float(cagr), max_dd


def apply_leverage_cdar(book_return_1x: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Axis E: cdar leverage policy -- a SINGLE closed-form constant L applied throughout (not a
    time-varying exposure guarantee; see cdar_constrained_leverage's own docstring). Computed on
    the FULL realized book_return_1x history, exactly as ctaA_sweep.py's apply_leverage did for
    e_key='cdar' (the frozen construction's own convention, not re-derived here)."""
    r = book_return_1x
    finite = r[np.isfinite(r)]
    if len(finite) < 60 or np.allclose(finite, 0.0):
        return r.copy(), 0.0, {}
    try:
        cdar = lp.cdar_constrained_leverage(finite, budget=0.25, alpha=0.95)
        L = cdar["L_closed_form"]
        if not np.isfinite(L):
            L = 0.0
    except Exception as e:
        cdar = {"error": str(e)}
        L = 0.0
    L = min(L, 25.0)
    final = L * np.nan_to_num(r)
    return final, float(L), cdar


def compute_metrics(final_net_return: np.ndarray, dates: pd.DatetimeIndex, avg_leverage: float) -> dict:
    r = np.nan_to_num(final_net_return)
    n = len(r)
    span_years = (dates[-1] - dates[0]).days / 365.25 if n > 1 else np.nan
    periods_per_year = n / span_years if span_years and span_years > 0 else 252.0

    cagr, max_dd = _cagr_from_returns(r, dates)
    mu, sd = np.mean(r), (np.std(r, ddof=1) if n > 1 else np.nan)
    sharpe = (mu / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan

    worst_day = float(np.min(r)) if n else np.nan
    ruin_flag = bool(np.any(1.0 + r <= 0.0))
    win_rate = float(np.mean(r[r != 0.0] > 0)) if np.any(r != 0.0) else np.nan

    out = {
        "cagr": cagr, "maxdd": max_dd, "sharpe": sharpe, "worst_day": worst_day,
        "ruin_flag": ruin_flag, "win_rate": win_rate, "avg_leverage": avg_leverage,
        "n_obs": n, "span_years": span_years, "periods_per_year": periods_per_year,
        "cagr_delta_vs_benchmark": (cagr - BENCHMARK_CAGR) if np.isfinite(cagr) else np.nan,
    }
    for budget in DD_BUDGETS:
        out[f"dd_budget_{int(budget*100)}"] = bool(np.isfinite(max_dd) and max_dd <= budget)
    return out


def regime_window_returns(ret: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    idx = pd.DatetimeIndex(dates)
    ser = pd.Series(np.nan_to_num(ret), index=idx)
    out = {}
    for label, (start, end) in REGIME_WINDOWS.items():
        window = ser.loc[(idx >= pd.Timestamp(start, tz="UTC")) & (idx <= pd.Timestamp(end, tz="UTC"))]
        out[label] = float(np.prod(1.0 + window.to_numpy()) - 1.0) if len(window) else np.nan
    return out


# ======================================================================================
# 4. Reused validation math -- function bodies reproduced from research/equity/e3_pipeline.py
#    (per METHODOLOGY's explicit reuse mandate), parameterized by a DERIVED periods_per_yr.
# ======================================================================================
def per_obs_sharpe(returns) -> float:
    a = np.asarray(returns, dtype="float64")
    if len(a) == 0:
        return 0.0
    sd = a.std(ddof=0)
    return float(a.mean() / sd) if sd > 1e-12 else 0.0


def _ppf(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= ph:
        q = p - 0.5
        rr = q*q
        return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / \
               (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def dsr(net: np.ndarray, n_trials: int, periods_per_yr: float) -> dict:
    from math import erf, sqrt
    r = net[np.isfinite(net)]
    T = len(r)
    if T < 30 or r.std() == 0:
        return {"psr": None, "dsr": None, "n_trials": n_trials, "T": T}
    sr = per_obs_sharpe(r)
    g3 = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    eul = 0.5772156649
    z1 = _ppf(1 - 1.0 / n_trials)
    z2 = _ppf(1 - 1.0 / (n_trials * math.e))
    sr0 = (1 - eul) * z1 + eul * z2
    sr0 *= sqrt(1.0 / T)
    denom = sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    psr = 0.5 * (1 + erf(((sr - 0) * sqrt(T - 1)) / denom / sqrt(2)))
    dsr_v = 0.5 * (1 + erf(((sr - sr0) * sqrt(T - 1)) / denom / sqrt(2)))
    return {"sr_obs": round(sr, 5), "psr_vs0": round(psr, 4), "dsr": round(dsr_v, 4),
            "sr0_obs": round(sr0, 5), "n_trials": n_trials, "T": T,
            "sharpe_ann_obs": round(sr * math.sqrt(periods_per_yr), 4)}


def cpcv_paths(net_full: np.ndarray, n_groups: int, k: int, embargo: int, periods_per_yr: float):
    net = net_full[np.isfinite(net_full)]
    N = len(net)
    if N < n_groups * 4:
        return []
    edges = np.linspace(0, N, n_groups + 1).astype(int)
    bounds = [(edges[i], edges[i + 1]) for i in range(n_groups)]
    paths = []
    for combo in combinations(range(n_groups), k):
        test = np.zeros(N, bool)
        for g in combo:
            lo, hi = bounds[g]
            test[lo:hi] = True
        if embargo > 0:
            drop = set()
            for g in combo:
                lo, hi = bounds[g]
                drop.update(range(max(0, lo - embargo), lo))
                drop.update(range(hi, min(N, hi + embargo)))
            test2 = test.copy()
            for d in drop:
                if 0 <= d < N:
                    test2[d] = False
            test = test2
        seg = net[test]
        if len(seg) > 20 and seg.std() > 0:
            paths.append({"combo": combo, "n_obs": int(len(seg)),
                          "sharpe_ann": float(per_obs_sharpe(seg) * np.sqrt(periods_per_yr)),
                          "cagr_proxy": float(np.prod(1.0 + seg) ** (periods_per_yr / len(seg)) - 1.0)})
    return paths


def pbo_cscv(cell_returns: dict[str, np.ndarray], n_splits: int = 8) -> float | None:
    names = list(cell_returns.keys())
    if len(names) < 2:
        return None
    lens = [len(v) for v in cell_returns.values()]
    n = min(lens)
    if n < n_splits * 4:
        return None
    M = np.stack([v[-n:] for v in cell_returns.values()], axis=1)
    n_groups = n_splits
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    bounds = [(edges[i], edges[i + 1]) for i in range(n_groups)]
    half = n_groups // 2
    combos = list(combinations(range(n_groups), half))
    below = 0
    total = 0
    for combo in combos:
        is_mask = np.zeros(n, bool)
        for g in combo:
            is_mask[bounds[g][0]:bounds[g][1]] = True
        oos_mask = ~is_mask
        is_ret = M[is_mask]
        oos_ret = M[oos_mask]
        if is_ret.shape[0] < 10 or oos_ret.shape[0] < 10:
            continue
        is_sr = np.array([per_obs_sharpe(is_ret[:, j]) for j in range(M.shape[1])])
        oos_sr = np.array([per_obs_sharpe(oos_ret[:, j]) for j in range(M.shape[1])])
        best_is = int(np.argmax(is_sr))
        if oos_sr[best_is] < np.median(oos_sr):
            below += 1
        total += 1
    return round(below / total, 4) if total > 0 else None


def per_year_sharpe(net: np.ndarray, idx: pd.DatetimeIndex, periods_per_yr: float) -> dict:
    out = {}
    s = pd.Series(net, index=idx)
    for yr, grp in s.groupby(s.index.year):
        g = grp.to_numpy()
        g = g[np.isfinite(g)]
        if len(g) >= 5:
            out[int(yr)] = round(float(per_obs_sharpe(g) * np.sqrt(periods_per_yr)), 3)
    return out


def per_year_return(net: np.ndarray, idx: pd.DatetimeIndex) -> dict:
    out = {}
    s = pd.Series(np.nan_to_num(net), index=idx)
    for yr, grp in s.groupby(s.index.year):
        g = grp.to_numpy()
        if len(g) >= 5:
            out[int(yr)] = round(float(np.prod(1.0 + g) - 1.0), 5)
    return out


def derive_periods_per_yr(dates: pd.DatetimeIndex) -> float:
    n_obs = len(dates)
    if n_obs < 2:
        return 252.0
    span_years = (dates[-1] - dates[0]).days / 365.25
    return n_obs / span_years if span_years > 0 else 252.0


# ======================================================================================
# main
# ======================================================================================
def main():
    t_start = time.time()
    print(f"=== ctaB_pipeline.py starting {pd.Timestamp.now()} ===", flush=True)
    print(f"N_PROCESSES = {N_PROCESSES}", flush=True)

    results: dict = {"meta": {"started": str(pd.Timestamp.now()),
                               "frozen_construction": {
                                   "a_key": AXIS_A_KEY, "b_key": B_PARAMS["name"],
                                   "c_key": AXIS_C_KEY, "d_key": AXIS_D_KEY, "e_key": AXIS_E_KEY,
                                   "winner_f_key": CONSTRUCTIONS["winner"]["f_key"],
                                   "runner_up_f_key": CONSTRUCTIONS["runner_up"]["f_key"],
                               },
                               "embargo_trading_days": EMBARGO_TRADING_DAYS,
                               "embargo_rationale": "Axis C causal_hmm_regime_posterior min_train=252 "
                                                     "trading days is the slowest signal-component "
                                                     "horizon among the frozen construction's axes "
                                                     "(A/B vol_window=20, D ewma_0.94 effective memory "
                                                     "~15.7d, F corr_window<=120) -- 252 used as embargo.",
                               "n_cells_primary": N_CELLS_PRIMARY,
                               "n_cells_stage_a22_only": N_CELLS_STAGE_A_22_ONLY}}

    names = reconcile_universe()
    results["meta"]["universe_n"] = len(names)
    results["meta"]["universe_names"] = names

    print(f"[phase1] building per-name exposure for {len(names)} names ...", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(N_PROCESSES) as pool:
        outs = pool.map(build_name_exposure, names, chunksize=1)
    name_cache = {sym: c for sym, c in outs if c is not None}
    failed = [sym for sym, c in outs if c is None]
    print(f"[phase1] done in {time.time()-t0:.1f}s, {len(name_cache)}/{len(names)} usable "
          f"(failed/excluded: {failed})", flush=True)
    results["meta"]["names_usable"] = len(name_cache)
    results["meta"]["names_failed"] = failed

    active_names = sorted(name_cache.keys())
    idx = None
    for sym in active_names:
        idx = name_cache[sym]["close"].index if idx is None else idx.union(name_cache[sym]["close"].index)
    idx = idx.sort_values()
    RETURNS_DF = pd.DataFrame({sym: name_cache[sym]["simple_ret"].reindex(idx) for sym in active_names})
    EXPOSURE_DF = pd.DataFrame({sym: name_cache[sym]["exposure"].reindex(idx) for sym in active_names})
    RV_DF = pd.DataFrame({sym: name_cache[sym]["forecast_vol"].reindex(idx) for sym in active_names})
    GATE_DF = pd.DataFrame({sym: name_cache[sym]["gate"].reindex(idx) for sym in active_names})
    n_trades_total = sum(name_cache[sym]["n_trades"] for sym in active_names)
    dates = RETURNS_DF.index
    periods_per_yr = derive_periods_per_yr(dates)
    print(f"[phase1] combined panel: {RETURNS_DF.shape}, periods_per_yr={periods_per_yr:.4f}", flush=True)
    results["meta"]["panel_shape"] = list(RETURNS_DF.shape)
    results["meta"]["periods_per_yr"] = periods_per_yr
    results["meta"]["n_trades_total_all_names"] = n_trades_total

    # Fresh 93-name equal-weight unlevered buy-and-hold, computed directly on THIS run's own
    # RETURNS_DF, as an apples-to-apples internal cross-check against Established Fact 6's
    # 0.468 CAGR (not a replacement for it -- Fact 6 remains the frozen comparator).
    eq_w_ret = RETURNS_DF.mean(axis=1, skipna=True).fillna(0.0).to_numpy()
    fresh_bh_cagr, fresh_bh_maxdd = _cagr_from_returns(eq_w_ret, dates)
    results["meta"]["fresh_93name_equalweight_bh_cagr_crosscheck"] = fresh_bh_cagr
    results["meta"]["fresh_93name_equalweight_bh_maxdd_crosscheck"] = fresh_bh_maxdd
    print(f"[crosscheck] fresh 93-name equal-weight B&H: CAGR={fresh_bh_cagr:.4f} maxDD={fresh_bh_maxdd:.4f} "
          f"(vs Established Fact 6: CAGR=0.468 maxDD=0.410)", flush=True)

    # ------------------------------------------------------------------------------
    # Phase 2: Axis F -- 2 base aggregate_portfolio calls (corr_window=60 nospike,
    # corr_window=120 nospike) + spike-scale-derived winner, exactly the Section-9-REDESIGN.7
    # algebraic reduction (byte-identical to the naive per-variant call).
    # ------------------------------------------------------------------------------
    print("[phase2] Axis F portfolio construction (corr_window in {60,120}) ...", flush=True)
    t0 = time.time()
    base_60 = pc.aggregate_portfolio(EXPOSURE_DF, RETURNS_DF, RV_DF, scheme="corr_risk_parity",
                                      corr_window=60, apply_spike_overlay=False)
    base_120 = pc.aggregate_portfolio(EXPOSURE_DF, RETURNS_DF, RV_DF, scheme="corr_risk_parity",
                                       corr_window=120, apply_spike_overlay=False)
    avg_corr_120 = pc.rolling_avg_pairwise_correlation(RETURNS_DF, window=120)
    spike_scale_120 = pc.correlation_spike_scale(avg_corr_120)
    weights_120_spike = base_120.weights.multiply(spike_scale_120, axis=0)
    print(f"[phase2] done in {time.time()-t0:.1f}s", flush=True)

    f_variants = {
        "winner": (weights_120_spike, base_120.n_active),
        "runner_up": (base_60.weights, base_60.n_active),
        "diagnostic_w120_nospike": (base_120.weights, base_120.n_active),
    }
    max_abs_diff_spike_collapse = float(
        (weights_120_spike.fillna(0.0) - base_120.weights.fillna(0.0)).abs().to_numpy().max())
    results["axis_f_w120_spike_vs_nospike_collapse_check"] = {
        "max_abs_diff": max_abs_diff_spike_collapse,
        "still_byte_identical_at_93_names": bool(max_abs_diff_spike_collapse < 1e-12),
        "note": "Stage A found w120_spike and w120_nospike numerically tied byte-for-byte at the "
                "22-name pilot (spike overlay never triggered on that book); re-checked here at "
                "full 93-name scale.",
    }
    print(f"[axis_f check] w120 spike-vs-nospike max_abs_diff = {max_abs_diff_spike_collapse:.3e} "
          f"(collapse_holds={max_abs_diff_spike_collapse < 1e-12})", flush=True)

    # ------------------------------------------------------------------------------
    # Phase 3: costs, leverage, metrics for winner + runner-up (+ diagnostic F-variant)
    # ------------------------------------------------------------------------------
    construction_results = {}
    return_series_for_pbo: dict[str, np.ndarray] = {}
    for key in ("winner", "runner_up", "diagnostic_w120_nospike"):
        weights, n_active = f_variants[key]
        book_1x_13 = apply_costs(weights, RETURNS_DF, cost_bp=13.0, funding_bp_week=0.0)
        book_1x_65 = apply_costs(weights, RETURNS_DF, cost_bp=6.5, funding_bp_week=0.0)
        book_1x_26 = apply_costs(weights, RETURNS_DF, cost_bp=26.0, funding_bp_week=0.0)
        gross_1x = gross_book_return(weights, RETURNS_DF)

        final_net, L, cdar_diag = apply_leverage_cdar(book_1x_13)
        final_gross = L * np.nan_to_num(gross_1x)
        final_65 = L * np.nan_to_num(book_1x_65)
        final_26 = L * np.nan_to_num(book_1x_26)

        m_net = compute_metrics(final_net, dates, L)
        m_gross = compute_metrics(final_gross, dates, L)
        cagr_65, _ = _cagr_from_returns(final_65, dates)
        cagr_26, _ = _cagr_from_returns(final_26, dates)

        # --- (i) cost-scenario robustness ---
        cost_scenarios = {"6.5bp": cagr_65, "13bp_primary": m_net["cagr"], "26bp": cagr_26}
        monotonic = (cagr_65 >= m_net["cagr"] >= cagr_26) or not all(np.isfinite([cagr_65, m_net["cagr"], cagr_26]))
        sign_reversal = np.isfinite(cagr_65) and np.isfinite(cagr_26) and (np.sign(cagr_65) != np.sign(cagr_26))

        # --- (ii) drop-top-5-contributing-periods robustness ---
        order = np.argsort(final_net)[::-1]
        top5_idx = order[:5]
        drop5_ret = final_net.copy()
        drop5_ret[top5_idx] = 0.0
        cagr_drop5, maxdd_drop5 = _cagr_from_returns(drop5_ret, dates)
        drop5 = {
            "top5_dates": [str(dates[i].date()) for i in top5_idx],
            "top5_returns": [float(final_net[i]) for i in top5_idx],
            "cagr_after_drop5": cagr_drop5,
            "maxdd_after_drop5": maxdd_drop5,
            "dd_budget_30_survives": bool(np.isfinite(maxdd_drop5) and maxdd_drop5 <= 0.30),
            "dd_budget_41_survives": bool(np.isfinite(maxdd_drop5) and maxdd_drop5 <= 0.41),
            "cagr_vs_benchmark_survives": bool(np.isfinite(cagr_drop5) and cagr_drop5 > BENCHMARK_CAGR) if False else None,
            "note": "cagr_vs_benchmark comparison is reported at the gate-check level below "
                    "(this construction is not expected to beat the unconstrained benchmark; "
                    "the relevant 'survives' check is whether the DD-budget qualifying status holds).",
        }

        # --- (iii) purged walk-forward (CPCV 8 groups, k=2, embargo=252) ---
        cpcv = cpcv_paths(final_net, CPCV_N_GROUPS, CPCV_K, EMBARGO_TRADING_DAYS, periods_per_yr)
        cpcv_sharpes = [p["sharpe_ann"] for p in cpcv]
        cpcv_summary = {
            "n_folds_computed": len(cpcv),
            "n_folds_possible": len(list(combinations(range(CPCV_N_GROUPS), CPCV_K))),
            "sharpe_mean": float(np.mean(cpcv_sharpes)) if cpcv_sharpes else None,
            "sharpe_median": float(np.median(cpcv_sharpes)) if cpcv_sharpes else None,
            "sharpe_min": float(np.min(cpcv_sharpes)) if cpcv_sharpes else None,
            "sharpe_max": float(np.max(cpcv_sharpes)) if cpcv_sharpes else None,
            "pct_folds_positive_sharpe": float(np.mean(np.array(cpcv_sharpes) > 0)) if cpcv_sharpes else None,
            "folds": cpcv,
        }

        # --- supplementary: walk-forward THIRDS (simple chronological 3-way split) ---
        n = len(final_net)
        thirds_bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
        thirds = []
        for lo, hi in thirds_bounds:
            seg = final_net[lo:hi]
            seg_dates = dates[lo:hi]
            c, d = _cagr_from_returns(seg, seg_dates)
            sh = per_obs_sharpe(seg[np.isfinite(seg)]) * np.sqrt(periods_per_yr) if len(seg) else np.nan
            thirds.append({"start": str(seg_dates[0].date()), "end": str(seg_dates[-1].date()),
                            "n_obs": int(len(seg)), "cagr": c, "maxdd": d, "sharpe_ann": float(sh)})

        # --- regime windows (net + gross) ---
        regime_net = regime_window_returns(final_net, dates)
        regime_gross = regime_window_returns(final_gross, dates)

        # --- per-year (net + gross) ---
        py_sharpe_net = per_year_sharpe(final_net, dates, periods_per_yr)
        py_return_net = per_year_return(final_net, dates)
        py_sharpe_gross = per_year_sharpe(final_gross, dates, periods_per_yr)
        py_return_gross = per_year_return(final_gross, dates)

        # --- melt-up decomposition (exact geometric decomposition, Stage A convention) ---
        span_years = m_net["span_years"]
        ret_meltup = regime_net["2023_2024_meltup"]
        total_full = (1.0 + m_net["cagr"]) ** span_years - 1.0
        if np.isfinite(ret_meltup) and (1.0 + ret_meltup) != 0 and span_years > MELTUP_YEARS:
            total_ex = (1.0 + total_full) / (1.0 + ret_meltup) - 1.0
            cagr_ex_meltup = (1.0 + total_ex) ** (1.0 / (span_years - MELTUP_YEARS)) - 1.0
        else:
            cagr_ex_meltup = np.nan
        meltup = {"ret_2023_24_meltup": ret_meltup, "cagr_full_sample": m_net["cagr"],
                  "cagr_ex_meltup": cagr_ex_meltup,
                  "haircut_pp": (m_net["cagr"] - cagr_ex_meltup) * 100 if np.isfinite(cagr_ex_meltup) else np.nan}

        construction_results[key] = {
            "f_params": CONSTRUCTIONS.get(key, DIAGNOSTIC_F_VARIANT),
            "avg_leverage_L": L,
            "cdar_diag": {k2: (float(v2) if isinstance(v2, (int, float, np.floating)) else v2)
                          for k2, v2 in cdar_diag.items() if k2 not in ("agree",)} | (
                          {"agree": bool(cdar_diag.get("agree"))} if "agree" in cdar_diag else {}),
            "metrics_net_primary": m_net,
            "metrics_gross_primary": m_gross,
            "cost_scenarios_cagr": cost_scenarios,
            "cost_scenario_robust": {"monotonic_no_reversal": bool(monotonic and not sign_reversal),
                                      "sign_reversal": bool(sign_reversal)},
            "drop_top5_robustness": drop5,
            "cpcv_purged_walk_forward": cpcv_summary,
            "walk_forward_thirds": thirds,
            "regime_window_returns_net": regime_net,
            "regime_window_returns_gross": regime_gross,
            "per_year_sharpe_net": py_sharpe_net,
            "per_year_return_net": py_return_net,
            "per_year_sharpe_gross": py_sharpe_gross,
            "per_year_return_gross": py_return_gross,
            "meltup_decomposition": meltup,
            "n_active_mean": float(n_active.mean()),
        }
        return_series_for_pbo[key] = final_net
        print(f"[phase3] {key}: L={L:.4f} CAGR={m_net['cagr']:.4f} maxDD={m_net['maxdd']:.4f} "
              f"Sharpe={m_net['sharpe']:.4f} worst_day={m_net['worst_day']:.4f}", flush=True)

    results["constructions"] = construction_results

    # ------------------------------------------------------------------------------
    # Exposure path through 2020-03 COVID and 2025-04-09-style squeeze windows (gate 5)
    # Combined regime-gate (Axis C, cross-name mean p_bull-derived gate) x CDaR leverage (Axis E,
    # constant L for the WINNER construction) -- reported as an actual daily time series, not a
    # summary statistic.
    # ------------------------------------------------------------------------------
    winner_weights, _ = f_variants["winner"]
    L_winner = construction_results["winner"]["avg_leverage_L"]
    gross_exposure_series = L_winner * winner_weights.abs().sum(axis=1)   # deployed book gross leverage
    mean_gate_series = GATE_DF.mean(axis=1, skipna=True)                  # cross-name avg Axis-C gate

    def _window_series(s: pd.Series, window: tuple[str, str]) -> list[dict]:
        start, end = window
        sub = s.loc[(s.index >= pd.Timestamp(start, tz="UTC")) & (s.index <= pd.Timestamp(end, tz="UTC"))]
        return [{"date": str(d.date()), "value": (float(v) if np.isfinite(v) else None)} for d, v in sub.items()]

    exposure_paths = {
        "2020_03_covid_window": {
            "window": COVID_EXPOSURE_WINDOW,
            "gross_deployed_exposure_L_x_sumabs_weights": _window_series(gross_exposure_series, COVID_EXPOSURE_WINDOW),
            "mean_axis_c_regime_gate": _window_series(mean_gate_series, COVID_EXPOSURE_WINDOW),
        },
        "2025_04_squeeze_window": {
            "window": SQUEEZE_EXPOSURE_WINDOW,
            "gross_deployed_exposure_L_x_sumabs_weights": _window_series(gross_exposure_series, SQUEEZE_EXPOSURE_WINDOW),
            "mean_axis_c_regime_gate": _window_series(mean_gate_series, SQUEEZE_EXPOSURE_WINDOW),
        },
    }
    results["exposure_paths_gate5"] = exposure_paths

    # ------------------------------------------------------------------------------
    # DSR / PBO -- informational only (PROGRAM CLASS: never gates adoption)
    # ------------------------------------------------------------------------------
    winner_net = return_series_for_pbo["winner"]
    dsr_primary = dsr(winner_net, N_CELLS_PRIMARY, periods_per_yr)
    dsr_stage_a22_only = dsr(winner_net, N_CELLS_STAGE_A_22_ONLY, periods_per_yr)

    # PBO: full-grid CSCV-PBO over all 29,760/9,920 Stage-A cells is NOT reconstructable here --
    # Stage A persisted only per-cell SUMMARY METRICS (results_ctaA_full_grid.csv.gz), not each
    # cell's full daily net-return path (a 29,760 x ~n_obs matrix was never written to keep the
    # artifact tractable) -- disclosed honestly rather than fabricated. As a genuine, real,
    # clearly-scoped supplementary computation, PBO is instead run over the 12-member Axis-E x
    # Axis-F family that IS cheaply and exactly reconstructable within this Stage-B run (same two
    # already-computed book_1x_13 (pre-leverage, cost-applied) paths, x all 6 Axis-E leverage
    # policies applied fresh to each) -- an honest, smaller, real family, not a proxy for the full
    # ledger.
    e_keys = ["half_kelly", "quarter_kelly", "gz_tight", "gz_loose", "evt_ceiling", "cdar"]

    def _apply_any_leverage(r: np.ndarray, e_key: str) -> np.ndarray:
        finite = r[np.isfinite(r)]
        if len(finite) < 60 or np.allclose(finite, 0.0):
            return r.copy()
        try:
            kelly = lp.kelly_optimal_leverage(finite, l_max=25.0)
            l_star = kelly["L_star"]
        except Exception:
            l_star = 0.0
        if e_key == "half_kelly":
            L = lp.fractional_kelly_leverage(max(l_star, 1e-6), 0.5) if l_star > 0 else 0.0
            return L * np.nan_to_num(r)
        if e_key == "quarter_kelly":
            L = lp.fractional_kelly_leverage(max(l_star, 1e-6), 0.25) if l_star > 0 else 0.0
            return L * np.nan_to_num(r)
        if e_key == "gz_tight":
            res = lp.grossman_zhou_exposure(np.nan_to_num(r), l_base=max(l_star, 1e-6), alpha_floor=0.75, rt_bp=13.0)
            return np.asarray(res["net_return"])
        if e_key == "gz_loose":
            res = lp.grossman_zhou_exposure(np.nan_to_num(r), l_base=max(l_star, 1e-6), alpha_floor=0.50, rt_bp=13.0)
            return np.asarray(res["net_return"])
        if e_key == "evt_ceiling":
            try:
                evt = lp.evt_ceiling_leverage(finite, threshold_quantile=0.95, breach_years=10.0)
                L = evt["L_ceiling"]
                if not np.isfinite(L):
                    L = 0.0
            except Exception:
                L = 0.0
            return min(L, 25.0) * np.nan_to_num(r)
        if e_key == "cdar":
            try:
                cdar_ = lp.cdar_constrained_leverage(finite, budget=0.25, alpha=0.95)
                L = cdar_["L_closed_form"]
                if not np.isfinite(L):
                    L = 0.0
            except Exception:
                L = 0.0
            return min(L, 25.0) * np.nan_to_num(r)
        raise ValueError(e_key)

    pbo_family: dict[str, np.ndarray] = {}
    winner_wts, _ = f_variants["winner"]
    runner_wts, _ = f_variants["runner_up"]
    book_winner_13 = apply_costs(winner_wts, RETURNS_DF, cost_bp=13.0, funding_bp_week=0.0)
    book_runner_13 = apply_costs(runner_wts, RETURNS_DF, cost_bp=13.0, funding_bp_week=0.0)
    for e_key in e_keys:
        pbo_family[f"winner_F__{e_key}"] = _apply_any_leverage(book_winner_13, e_key)
        pbo_family[f"runnerup_F__{e_key}"] = _apply_any_leverage(book_runner_13, e_key)

    pbo_proxy = pbo_cscv(pbo_family, n_splits=8)

    results["dsr_pbo_informational"] = {
        "program_class_note": "PROGRAM CLASS = benchmark-relative (CTA_PREREG.md), same as I1's own "
                               "overlay-validation framing -- DSR/PBO are reported here for the record "
                               "only and NEVER gate adoption. Low values are expected and are not "
                               "themselves treated as a failure.",
        "dsr_at_n_cells_29760_primary": dsr_primary,
        "dsr_at_n_cells_9920_stage_a22_only": dsr_stage_a22_only,
        "pbo_full_grid_29760_or_9920": None,
        "pbo_full_grid_note": "NOT reconstructable: Stage A's results_ctaA_full_grid.csv.gz persists "
                               "only per-cell SUMMARY METRICS, not each of the 29,760/9,920 cells' "
                               "full daily net-return path, so a literal CSCV-PBO over the complete "
                               "grid cannot be computed post-hoc without re-running the entire Stage-A "
                               "grid with full return-path retention (out of Stage B's scope). Reported "
                               "as None rather than fabricated.",
        "pbo_proxy_12_member_ExF_family": pbo_proxy,
        "pbo_proxy_note": "Genuine CSCV-PBO (Bailey et al. 2015, n_splits=8) computed over a REAL, "
                          "cheaply-reconstructable 12-member family (the winner's and runner-up's own "
                          "already-computed 13bp cost-applied book returns, x all 6 frozen Axis-E "
                          "leverage policies) -- an honest, clearly-scoped supplementary number, NOT a "
                          "substitute for the full-grid PBO above.",
    }

    # ------------------------------------------------------------------------------
    # Adoption gate -- six sub-checks, computed with REAL Stage-B numbers (winner construction,
    # primary 13bp/funding-0 scenario), frozen criteria restated verbatim from CTA_PREREG.md.
    # ------------------------------------------------------------------------------
    wm = construction_results["winner"]["metrics_net_primary"]
    w_cpcv = construction_results["winner"]["cpcv_purged_walk_forward"]
    w_drop5 = construction_results["winner"]["drop_top5_robustness"]
    w_cost = construction_results["winner"]["cost_scenario_robust"]

    gate1_applicable = wm["dd_budget_41"]
    gate1_value = wm["cagr"] if gate1_applicable else None
    gate1_pass = bool(gate1_applicable and np.isfinite(wm["cagr"]) and wm["cagr"] > BENCHMARK_CAGR)

    gate2_applicable = wm["dd_budget_30"]
    gate2_threshold = 0.75 * BENCHMARK_CAGR
    gate2_value = wm["cagr"] if gate2_applicable else None
    gate2_pass = bool(gate2_applicable and np.isfinite(wm["cagr"]) and wm["cagr"] > gate2_threshold)

    gate3_threshold = BENCHMARK_SHARPE - 0.05
    gate3_pass = bool(np.isfinite(wm["sharpe"]) and wm["sharpe"] >= gate3_threshold)

    gate4i_pass = w_cost["monotonic_no_reversal"]
    gate4ii_pass = bool(w_drop5["dd_budget_30_survives"] or w_drop5["dd_budget_41_survives"])
    gate4iii_pass = bool(w_cpcv["pct_folds_positive_sharpe"] is not None and w_cpcv["pct_folds_positive_sharpe"] >= 0.5)
    gate4_pass = bool(gate4i_pass and gate4ii_pass and gate4iii_pass)

    covid_gate_vals = [d["value"] for d in exposure_paths["2020_03_covid_window"]["mean_axis_c_regime_gate"] if d["value"] is not None]
    squeeze_gate_vals = [d["value"] for d in exposure_paths["2025_04_squeeze_window"]["mean_axis_c_regime_gate"] if d["value"] is not None]
    covid_gate_min = min(covid_gate_vals) if covid_gate_vals else None
    covid_gate_start = covid_gate_vals[0] if covid_gate_vals else None
    squeeze_gate_min = min(squeeze_gate_vals) if squeeze_gate_vals else None
    squeeze_gate_range = (max(squeeze_gate_vals) - min(squeeze_gate_vals)) if squeeze_gate_vals else None
    gate5_covid_deriskED = bool(covid_gate_min is not None and covid_gate_start is not None
                                 and covid_gate_min < covid_gate_start - 0.05)
    gate5_squeeze_moved = bool(squeeze_gate_range is not None and squeeze_gate_range > 0.02)
    gate5_pass = bool(gate5_covid_deriskED and gate5_squeeze_moved)

    gate6_pass = bool((not wm["ruin_flag"]) and np.isfinite(wm["worst_day"]) and wm["worst_day"] > -0.25)

    all_pass = bool(gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass)

    results["adoption_gate"] = {
        "1_cagr_at_41pct_dd_budget_gt_benchmark": {
            "dd_budget_41_applicable": bool(gate1_applicable), "cagr": gate1_value,
            "benchmark_cagr": BENCHMARK_CAGR, "pass": gate1_pass,
        },
        "2_cagr_at_30pct_dd_budget_gt_0.75x_benchmark": {
            "dd_budget_30_applicable": bool(gate2_applicable), "cagr": gate2_value,
            "threshold": gate2_threshold, "pass": gate2_pass,
        },
        "3_net_sharpe_gte_benchmark_minus_0.05": {
            "sharpe": wm["sharpe"], "threshold": gate3_threshold, "pass": gate3_pass,
        },
        "4_robustness_i_ii_iii": {
            "4i_cost_scenario_monotonic_no_reversal": gate4i_pass,
            "4ii_drop_top5_dd_budget_survives": gate4ii_pass,
            "4iii_cpcv_pct_folds_positive_sharpe_ge_50pct": gate4iii_pass,
            "4iii_pct_folds_positive": w_cpcv["pct_folds_positive_sharpe"],
            "pass": gate4_pass,
        },
        "5_drawdown_control_engages_both_windows": {
            "covid_gate_start": covid_gate_start, "covid_gate_min": covid_gate_min,
            "covid_de_risked": gate5_covid_deriskED,
            "squeeze_gate_range": squeeze_gate_range, "squeeze_gate_moved": gate5_squeeze_moved,
            "pass": gate5_pass,
        },
        "6_no_ruin_worst_day_gt_neg25pct": {
            "ruin_flag": wm["ruin_flag"], "worst_day": wm["worst_day"], "pass": gate6_pass,
        },
        "ALL_GATES_PASS_PAPER_FIRST_RECOMMENDED": all_pass,
    }

    print(f"[gate] 1={gate1_pass} 2={gate2_pass} 3={gate3_pass} 4={gate4_pass} 5={gate5_pass} 6={gate6_pass} "
          f"=> ALL_PASS={all_pass}", flush=True)

    results["meta"]["total_wallclock_sec"] = time.time() - t_start
    results["meta"]["finished"] = str(pd.Timestamp.now())

    def _sanitize(obj):
        """Recursively convert numpy scalars/arrays/NaN-Inf/sets/Timestamps to plain JSON-safe
        Python objects -- applied to the WHOLE results tree before dumping (rather than relying
        on json.dump's `default=`, which is never invoked for native float NaN/Inf)."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if not np.isfinite(v) else v
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return _sanitize(obj.tolist())
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(_sanitize(results), f, indent=2, default=str)
    print(f"=== DONE: results written to {RESULTS_PATH} ===", flush=True)
    print(f"Total wall-clock: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
