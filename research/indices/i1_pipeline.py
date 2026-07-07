#!/usr/bin/env python
"""I1 TIMED-BETA OVERLAY -- SPY/QQQ (research/indices/I1_PREREG.md, frozen 2026-07-07).

OVERLAY-VALIDATION program (NOT edge-certification like E1/E2/E3): n=2 instruments (SPYUSDT,
QQQUSDT), ~8 years of daily duka-sourced data. S1 trend-filter ensemble (200d SMA +/-2% band,
12mo total-return sign, 10mo SMA -- equal-weight ensemble, the ONLY tradeable S1 cell) x S2
vol-targeting (W_vol in {20,60} x cap in {1.0,1.5}, 10% annualized target) x S3 calendar tilts
(turn-of-month + pre-FOMC, both IN; with/without-tilts reported separately). Pre-named deployable:
50/50 SPY/QQQ blend, S1 ensemble x S2(W_vol=60, cap=1.0) x S3(both tilts on), 13bp primary cost,
daily marks, decide-at-close/act-next-day throughout.

PROGRAM CLASS: adoption gate is BENCHMARK-RELATIVE (vs. 50/50 SPY/QQQ buy-and-hold), NOT DSR-based.
DSR/PBO at N_trials=72 reported informationally only (expected low at breadth=2, per prereg).

AVOIDING THE FIVE CARRIED-FORWARD BUG PATTERNS (declared, checked explicitly, not assumed):
  (1) Source-aware returns: every stitched parquet filtered to source=='duka' BEFORE any
      resampling/SMA/return computation (load_stitched_duka), verify_no_perp_rows_in_signal_panel()
      re-run fresh for SPYUSDT/QQQUSDT specifically.
  (2) Derived annualization: periods_per_yr = n_obs/span_years computed FRESH per net-return array
      (derive_periods_per_yr), never hardcoded 252 in a formula.
  (3) Exposure-timing alignment (THE central risk): every exposure array (S1 state, S2 scale, S3
      multiplier, combined) is built via an explicit .shift(1)-equivalent / strictly-prior-index
      slice. A printed diagnostic traces calendar-date alignment for the return series and every
      exposure-scale series before anything is trusted. Verified via a synthetic-outlier injection
      test (inject a fake move at one index, confirm no exposure downstream reacts before the
      correct lag).
  (4) Costs scale with realized exposure changes: trading fee = 13bp(+scenarios) * |delta-exposure|
      per day; funding = bp/week PRORATED TO DAILY (/7 calendar days, declared+justified) * that
      day's actual combined exposure (never a flat charge).
  (5) Leveraged/inverse-ETF exclusion: MOOT (SPY/QQQ are plain unleveraged ETFs) -- stated
      explicitly as a deliberate non-applicable check.

Path hygiene: REPO points at THIS worktree (/home/speculation/convexity-indices), not the deleted
convexity-equity worktree. Does NOT call convexity.panel.panel.build_panel()/load_bars() for the
core signal (load_bars() does not read the source column).

Run: nohup /home/speculation/convexity/.venv/bin/python research/indices/i1_pipeline.py \
       > research/indices/i1.out 2>&1 &
Writes: research/indices/results_i1.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity-indices")
DATA_STITCHED = REPO / "data/stitched"
DATA_FOMC = REPO / "data/fomc"
OUT = REPO / "research/indices"
CANDIDATES_MD = REPO / "src/convexity/panel/candidates.md"

SYMBOLS = ("SPYUSDT", "QQQUSDT")
SIGMA_STAR = 0.10  # S2 target_vol, 10% annualized (house convention)

N_TRIALS = 72  # frozen ledger, I1_PREREG.md: S2(4) x S3(2) x cost/funding(3x3=9) = 72

RT_PRIMARY_BP = 13.0
RT_SCEN = (6.5, 13.0, 26.0)
FUND_SCEN_BP_PER_WK = (0.0, 5.0, -5.0)

# S2 grid (frozen)
W_VOL_GRID = (20, 60)
CAP_GRID = (1.0, 1.5)
# S3 grid (frozen)
TILT_VARIANTS = ("with_tilts", "no_tilts")
TILT_BOOST = 0.25  # +25% relative exposure boost

# Deployable cell (pre-named, I1_PREREG.md "Pre-named deployable")
DEPLOYABLE_W_VOL = 60
DEPLOYABLE_CAP = 1.0
DEPLOYABLE_TILTS = "with_tilts"
DEPLOYABLE_RT_BP = RT_PRIMARY_BP

# Regime windows (S1 regime-sanity + reporting, I1_PREREG.md)
REGIME_WINDOWS = {
    "2020-03_COVID": ("2020-02-15", "2020-04-15"),
    "2020H2_2021_recovery": ("2020-07-01", "2021-12-31"),
    "2018Q4_chop": ("2018-10-01", "2018-12-31"),
    "2022_chop_decline": ("2022-01-01", "2022-12-31"),
    "2022_full_year": ("2022-01-01", "2022-12-31"),
    "2023_2024_meltup": ("2023-01-01", "2024-12-31"),
}


# ======================================================================================
# reused validation math (metrics / dsr / cpcv_paths / pbo_cscv / per_obs_sharpe /
# per_year_sharpe / derive_periods_per_yr) -- reproduced near-verbatim from e3_pipeline.py's
# already-bug-fixed function bodies, per METHODOLOGY's explicit reuse mandate, parameterized by a
# DERIVED periods_per_yr (bug pattern 2).
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
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
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
            "sr0_obs": round(sr0, 5), "n_trials": n_trials, "T": T}


def vol_target(net: np.ndarray, periods_per_yr: float, win: int, sigma_star: float = SIGMA_STAR) -> np.ndarray:
    s = pd.Series(net)
    tv = s.rolling(win).std().shift(1) * np.sqrt(periods_per_yr)
    scale = np.clip(sigma_star / tv, 0.5, 2.0).fillna(1.0).to_numpy()
    return net * scale


def metrics(net: np.ndarray, periods_per_yr: float, turn: np.ndarray | None = None,
            gross: np.ndarray | None = None) -> dict:
    net = np.asarray(net, dtype="float64")
    valid = np.isfinite(net)
    net_v = net[valid]
    if len(net_v) < 30:
        return {"n_obs": int(len(net_v))}
    eq = np.cumprod(1 + net_v)
    peak = np.maximum.accumulate(eq)
    yrs = len(net_v) / periods_per_yr
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    vt = vol_target(net_v, periods_per_yr, win=max(int(periods_per_yr / 2), 10))
    eq_vt = np.cumprod(1 + vt)
    cagr_vt = eq_vt[-1] ** (1 / yrs) - 1 if eq_vt[-1] > 0 else -1.0
    srt = np.sort(net_v)[::-1]
    out = {
        "n_obs": int(len(net_v)),
        "sharpe_ann": round(float(per_obs_sharpe(net_v) * np.sqrt(periods_per_yr)), 4),
        "sharpe_obs": round(float(per_obs_sharpe(net_v)), 6),
        "ev_bp_obs": round(float(net_v.mean()) * 1e4, 4),
        "win_pct": round(float((net_v > 0).mean()) * 100, 2),
        "maxdd_pct": round(float((1 - eq / peak).max()) * 100, 3),
        "cagr_pct": round(float(cagr) * 100, 4),
        "cagr_voltarget_pct": round(float(cagr_vt) * 100, 4),
        "drop5_sharpe": round(float(per_obs_sharpe(np.sort(net_v)[:-5]) * np.sqrt(periods_per_yr)), 4) if len(net_v) > 10 else None,
        "top5_share": round(float(srt[:5].sum() / net_v.sum()), 4) if net_v.sum() != 0 else None,
        "turn_mean": round(float(np.nanmean(turn[valid])), 6) if turn is not None else None,
        "gross_sharpe_ann": None, "gross_cagr_pct": None,
    }
    if gross is not None:
        g_v = np.asarray(gross, dtype="float64")[valid]
        g_v = g_v[np.isfinite(g_v)]
        if len(g_v) >= 30:
            eqg = np.cumprod(1 + g_v)
            yrsg = len(g_v) / periods_per_yr
            cagrg = eqg[-1] ** (1 / yrsg) - 1 if eqg[-1] > 0 else -1.0
            out["gross_sharpe_ann"] = round(float(per_obs_sharpe(g_v) * np.sqrt(periods_per_yr)), 4)
            out["gross_cagr_pct"] = round(float(cagrg) * 100, 4)
            out["gross_ev_bp_obs"] = round(float(g_v.mean()) * 1e4, 4)
    return out


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
            lo = max(lo, 0)
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
            paths.append(float(per_obs_sharpe(seg) * np.sqrt(periods_per_yr)))
    return paths


def pbo_cscv(cell_returns: dict[str, np.ndarray], n_splits: int = 8) -> float | None:
    """CSCV-style PBO over the config family (Bailey et al. 2015): split the common aligned tail
    into n_splits contiguous blocks, form all C(n_splits, n_splits//2) IS/OOS combos, pick the
    IS-best config each time, measure the fraction of times that config lands BELOW the OOS
    median. INFORMATIONAL ONLY per PROGRAM CLASS -- does not gate this program's adoption
    decision."""
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
            out[int(yr)] = round(float(per_obs_sharpe(g) * np.sqrt(periods_per_yr)), 4)
    return out


def derive_periods_per_yr(dates: pd.DatetimeIndex, label: str, expect_const: float | None = None,
                           tol: float = 0.15) -> float:
    """Derive periods_per_yr from the ACTUAL observed calendar spacing of a net-return array's
    date index -- per bug pattern (2), never a hardcoded literal. n_obs / span_in_years. Prints
    the full arithmetic (n_obs, span_years, derived) so it is auditable."""
    n_obs = len(dates)
    if n_obs < 2:
        return expect_const if expect_const is not None else 252.0
    span_years = (dates[-1] - dates[0]).days / 365.25
    derived = n_obs / span_years if span_years > 0 else (expect_const or 252.0)
    msg = (f"    [periods_per_yr derivation] {label}: n_obs={n_obs} span_years={span_years:.6f} "
           f"derived={derived:.6f}")
    if expect_const is not None:
        msg += f" (expected~{expect_const:.4f})"
    print(msg, flush=True)
    if expect_const is not None and expect_const > 0:
        rel_err = abs(derived - expect_const) / expect_const
        if rel_err > tol:
            print(f"    WARNING: {label} derived periods_per_yr {derived:.4f} is >{tol:.0%} off "
                  f"the expected constant {expect_const:.4f} (rel_err={rel_err:.3%})", flush=True)
    return derived


# ======================================================================================
# universe / data loading (bug pattern 1: source-aware returns)
# ======================================================================================
def load_universe() -> list[dict]:
    txt = CANDIDATES_MD.read_text()
    m = re.search(r'"equities":\s*(\[.*?\n  \])', txt, re.S)
    if not m:
        raise RuntimeError("could not locate equities JSON array in candidates.md")
    return json.loads(m.group(1))


def load_universe_via_module() -> list[dict]:
    sys.path.insert(0, str(REPO / "src"))
    from convexity.panel.candidates import load_universe as _lu  # noqa: E402
    u = _lu()
    return u["equities"]


def load_stitched_duka(symbol: str) -> pd.DataFrame:
    """Load a symbol's stitched parquet, filtered to source=='duka' ONLY, before anything else
    touches it -- perp rows are never read past this point by the core-signal pipeline
    (bug pattern 1)."""
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df = df[df["source"] == "duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def verify_no_perp_rows_in_signal_panel(sample_symbols: list[str]) -> dict:
    """Mandatory check (reused pattern from e2/e3_pipeline.py), re-run FRESH for SPYUSDT/QQQUSDT
    specifically (not assumed to still hold just because it held for other symbols in E1/E2/E3's
    panel): (a) every row consumed by the daily-close builder has source=='duka'; (b) directly
    confirm the known splice seam is present in the raw file for each symbol, and that the
    source=='duka'-filtered daily-close series this pipeline actually builds terminates at/before
    the last duka day and contains zero perp-sourced closes."""
    out = {"checked_symbols": [], "all_duka_only": True, "seam_checks": {}}
    for sym in sample_symbols:
        df = load_stitched_duka(sym)
        ok = bool((df["source"] == "duka").all())
        out["checked_symbols"].append({"symbol": sym, "n_rows": int(len(df)), "all_duka": ok})
        if not ok:
            out["all_duka_only"] = False

        raw = pd.read_parquet(DATA_STITCHED / f"{sym}.parquet")
        perp_rows = raw[raw["source"] == "perp"]
        seam_confirmed = False
        seam_detail = {}
        if not perp_rows.empty:
            first_perp_ts = int(perp_rows["open_time"].min())
            row = raw[raw["open_time"] == first_perp_ts].iloc[0]
            prev_rows = raw[raw["open_time"] < first_perp_ts]
            if not prev_rows.empty:
                prev = prev_rows.iloc[-1]
                seam_confirmed = bool(row["source"] == "perp" and prev["source"] == "duka")
                seam_detail = {
                    "first_perp_open_time": first_perp_ts,
                    "source_at_seam": str(row["source"]),
                    "source_prev_row": str(prev["source"]),
                    "gap_before_at_seam": bool(row["gap_before"]),
                }
        filtered = load_stitched_duka(sym)
        last_duka_ts = filtered["ts"].max()
        daily_close = daily_close_series_duka(filtered)
        no_perp_in_daily_close = bool(daily_close.index.max() <= last_duka_ts.floor("D"))
        out["seam_checks"][sym] = {
            "raw_seam_confirmed_present_in_lake": seam_confirmed,
            "seam_detail": seam_detail,
            "last_duka_ts": str(last_duka_ts),
            "constructed_daily_close_series_max_day": str(daily_close.index.max()),
            "daily_close_terminates_at_or_before_last_duka_day": no_perp_in_daily_close,
            "PASS": bool(seam_confirmed and no_perp_in_daily_close and ok),
        }
    return out


def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    last_idx = d.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


# ======================================================================================
# FOMC calendar
# ======================================================================================
def load_fomc_dates() -> pd.DatetimeIndex:
    fp = DATA_FOMC / "fomc_dates.csv"
    df = pd.read_csv(fp)
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    if "scheduled" in df.columns:
        sched = df["scheduled"].astype(str).str.upper().isin(("TRUE", "1"))
        n_before = len(df)
        df = df[sched]
        n_after = len(df)
        print(f"  FOMC calendar: {n_before} rows loaded, {n_after} scheduled==TRUE retained "
              f"(unscheduled/emergency meetings excluded by construction)", flush=True)
    return pd.DatetimeIndex(sorted(df["announcement_date"].unique()))


# ======================================================================================
# S1: trend filter ensemble (frozen, decide-at-close/act-next-day throughout)
# ======================================================================================
def s1_sma_band_state(close: pd.Series, canary: bool = False) -> pd.Series:
    """200-day SMA +/-2% hysteresis band. Long when price > SMA*1.02, flat when price < SMA*0.98,
    HOLD prior state inside the band. Honest version: state[t] uses close[t-1] and SMA computed
    through close[t-1] only (SMA is itself shifted so its value at index t reflects the trailing
    200 days ending at t-1). Canary version: uses close[t] and SMA computed THROUGH close[t] (the
    day whose return the state is about to be applied to) -- deliberately wrong, for the
    lookahead-canary battery."""
    sma = close.rolling(200, min_periods=200).mean()
    if canary:
        px_ref = close
        sma_ref = sma
    else:
        px_ref = close.shift(1)
        sma_ref = sma.shift(1)
    upper = sma_ref * 1.02
    lower = sma_ref * 0.98
    n = len(close)
    state = np.zeros(n)
    prev = 0.0
    px_arr = px_ref.to_numpy()
    up_arr = upper.to_numpy()
    lo_arr = lower.to_numpy()
    for i in range(n):
        p, u, lo = px_arr[i], up_arr[i], lo_arr[i]
        if not (np.isfinite(p) and np.isfinite(u) and np.isfinite(lo)):
            state[i] = prev
            continue
        if p > u:
            cur = 1.0
        elif p < lo:
            cur = 0.0
        else:
            cur = prev  # hysteresis: hold prior state inside the band
        state[i] = cur
        prev = cur
    return pd.Series(state, index=close.index)


def s1_monthly_rule_state(close: pd.Series, month_end_idx: pd.DatetimeIndex,
                           rule: str, canary: bool = False) -> pd.Series:
    """Monthly rules (b: 12mo total-return sign, c: 10mo SMA), evaluated at each month-end using
    data strictly through the PRIOR month's final trading day, then broadcast as a constant
    exposure across every trading day of the CURRENT month. canary=True: computed through the
    CURRENT month's own most recent close instead (deliberately wrong lookahead).

    IMPORTANT UNIT NOTE (bug caught during smoke-testing, fixed before any headline number was
    trusted): I1_PREREG.md's "252-trading-day" / "210-trading-day" lookbacks are DAILY trading-day
    counts (~12 / ~10 CALENDAR months), NOT counts of monthly-sampled observations. The 252-day
    total return and 210-day SMA are therefore computed on the DAILY close series first (so the
    lookback windows are the correct ~12mo / ~10mo of calendar time), and ONLY THEN sampled at each
    month-end position -- computing `.pct_change(252)` / `.rolling(210)` directly on an
    already-monthly-resampled series would require 252/210 MONTHS (~21yr / ~17.5yr) of history,
    which never validates within this program's ~8yr sample and would leave S1 rules (b)/(c)
    silently stuck at flat (0) for the entire program -- exactly the kind of suspiciously-clean/
    all-zero result that should trigger a bug hunt (the 2018Q4 whipsaw-cost clause, which is
    supposed to show honest flip activity, was the check that surfaced this)."""
    if rule == "b":
        ret_252_daily = close.pct_change(252)
        raw_state_daily = (ret_252_daily > 0).astype(float)
    elif rule == "c":
        sma210_daily = close.rolling(210, min_periods=210).mean()
        raw_state_daily = (close > sma210_daily).astype(float)
    else:
        raise ValueError(rule)
    raw_state_at_me = raw_state_daily.reindex(month_end_idx)

    if canary:
        # deliberately wrong: broadcast month t's OWN decision (computed off month t's own
        # month-end data) across month t's own days -- the honest version instead uses month
        # (t-1)'s decision for month t's days (see below).
        decision_for_month = raw_state_at_me
    else:
        # honest: month t's exposure decision uses data strictly through month (t-1)'s month-end
        decision_for_month = raw_state_at_me.shift(1)

    # broadcast: for each day, the exposure is the decision made at the PRIOR month-end (honest)
    # or the CURRENT month-end (canary), constant through the day's own calendar month.
    daily_month_key = close.index.to_period("M")
    decision_by_month_key = decision_for_month.copy()
    decision_by_month_key.index = decision_by_month_key.index.to_period("M")
    # decision_by_month_key[m] = exposure to apply to every day IN month m
    out = daily_month_key.map(decision_by_month_key).astype(float)
    return pd.Series(out.to_numpy(), index=close.index).fillna(0.0)


def s1_ensemble(close: pd.Series, month_end_idx: pd.DatetimeIndex, canary: bool = False) -> pd.DataFrame:
    a = s1_sma_band_state(close, canary=canary)
    b = s1_monthly_rule_state(close, month_end_idx, "b", canary=canary)
    c = s1_monthly_rule_state(close, month_end_idx, "c", canary=canary)
    ens = (a + b + c) / 3.0
    return pd.DataFrame({"s1_a_sma_band": a, "s1_b_12mo_sign": b, "s1_c_10mo_sma": c,
                          "s1_ensemble": ens})


# ======================================================================================
# S2: volatility targeting (frozen, daily cadence, no-lookahead)
# ======================================================================================
def s2_vol_scale(log_ret: pd.Series, periods_per_yr: float, w_vol: int, cap: float,
                  canary: bool = False) -> pd.Series:
    """scale[t] = clip(target_vol / realized_vol[t], 0, cap). realized_vol[t] honest version is
    computed over returns[t-w_vol : t-1] (trailing window ending strictly before t, NEVER
    including returns[t] itself) -- rolling(w_vol).std() naturally uses [t-w_vol+1, t], so an
    explicit .shift(1) moves that window to [t-w_vol, t-1], excluding day t. canary=True: same-day
    realized vol (the trailing window INCLUDES returns[t] itself, i.e. no shift)."""
    roll_std = log_ret.rolling(w_vol, min_periods=w_vol).std()
    if canary:
        rv = roll_std  # includes returns[t] itself -- deliberately wrong
    else:
        rv = roll_std.shift(1)  # excludes returns[t]; uses [t-w_vol, t-1]
    ann_vol = rv * np.sqrt(periods_per_yr)
    scale = np.clip(SIGMA_STAR / ann_vol, 0.0, cap)
    return scale.fillna(0.0)


# ======================================================================================
# S3: calendar tilts (frozen)
# ======================================================================================
def s3_turn_of_month_mask(cal: pd.DatetimeIndex) -> pd.Series:
    """Last-3-trading-days-of-month + first-2-trading-days-of-next-month window, determined purely
    from the CALENDAR (trading-day position within the month), no return/price dependency."""
    df = pd.DataFrame({"date": cal})
    df["ym"] = cal.year * 100 + cal.month
    df["pos_from_end"] = df.groupby("ym").cumcount(ascending=False)  # 0 = last day of month
    df["pos_from_start"] = df.groupby("ym").cumcount(ascending=True)  # 0 = first day of month
    last3 = df["pos_from_end"] < 3
    first2 = df["pos_from_start"] < 2
    mask = (last3 | first2).to_numpy()
    return pd.Series(mask, index=cal)


def s3_pre_fomc_mask(cal: pd.DatetimeIndex, fomc_dates: pd.DatetimeIndex, canary: bool = False) -> pd.Series:
    """Honest: exposure[announcement_date - 1 trading day] gets the boost (the trading day
    IMMEDIATELY PRECEDING each regularly-scheduled announcement -- the daily-granularity analog of
    a 24h-pre-announcement window). canary=True: the window is shifted onto/after the announcement
    date itself instead of strictly before it (deliberately wrong)."""
    cal_arr = cal.normalize()
    ann_days = pd.DatetimeIndex(fomc_dates).tz_localize(cal_arr.tz) if cal_arr.tz is not None else pd.DatetimeIndex(fomc_dates)
    mask = np.zeros(len(cal), dtype=bool)
    cal_pos = {d: i for i, d in enumerate(cal_arr)}
    for ann in ann_days:
        # find the trading day immediately preceding (or, for canary, AT/after) ann in the calendar
        pos = cal_arr.searchsorted(ann)
        if canary:
            # deliberately wrong: boost the announcement day itself (and the day after), not the
            # day strictly before it
            if pos < len(cal):
                mask[pos] = True
            if pos + 1 < len(cal):
                mask[pos + 1] = True
        else:
            if pos - 1 >= 0:
                mask[pos - 1] = True
    return pd.Series(mask, index=cal)


def s3_combined_multiplier(cal: pd.DatetimeIndex, fomc_dates: pd.DatetimeIndex,
                            tilts: str, canary: bool = False) -> pd.Series:
    if tilts == "no_tilts":
        return pd.Series(1.0, index=cal)
    tom = s3_turn_of_month_mask(cal)
    fomc = s3_pre_fomc_mask(cal, fomc_dates, canary=canary)
    boosted = (tom | fomc)
    mult = np.where(boosted.to_numpy(), 1.0 + TILT_BOOST, 1.0)
    return pd.Series(mult, index=cal)


# ======================================================================================
# cost model (bug pattern 4: costs scale with realized exposure changes)
# ======================================================================================
def apply_costs(net_gross: np.ndarray, exposure: np.ndarray, rt_bp: float,
                 funding_bp_per_wk: float) -> dict:
    """Trading fee = rt_bp * |delta-exposure| per day (a full trend flip 0->1 incurs the FULL
    round-trip cost; a small vol-target nudge incurs a proportional fraction; zero change incurs
    zero cost). Funding = funding_bp_per_wk PRORATED TO DAILY via /7 CALENDAR DAYS (declared
    convention: funding/borrow-style carry on a real cash equity or equity-exposure instrument
    accrues on calendar time -- weekends/holidays are not free, matching how overnight financing
    actually works economically for a levered or short equity position; this is DIFFERENT from a
    trading fee, which only fires on realized trading-day turnover) -- SCALED BY the day's actual
    exposure held (never a flat charge regardless of exposure, per bug pattern 4 / E2's flat-
    funding bug)."""
    delta_exposure = np.abs(np.diff(exposure, prepend=exposure[0] if len(exposure) else 0.0))
    delta_exposure[0] = abs(exposure[0]) if len(exposure) else 0.0  # first day: entering from flat
    fee = (rt_bp / 1e4) * delta_exposure
    funding_bp_per_day = funding_bp_per_wk / 7.0  # DECLARED CONVENTION: /7 calendar days
    funding = (funding_bp_per_day / 1e4) * exposure  # scaled by actual exposure held, every day
    net = net_gross - fee - funding
    return {"net": net, "fee": fee, "funding": funding, "delta_exposure": delta_exposure,
            "funding_bp_per_day": funding_bp_per_day}


# ======================================================================================
# core per-instrument build: daily close, log returns, S1/S2 exposure arrays, all with an
# explicit calendar-date-alignment diagnostic (bug pattern 3, the central risk of this program).
# ======================================================================================
def build_instrument(symbol: str, fomc_dates: pd.DatetimeIndex) -> dict:
    df = load_stitched_duka(symbol)
    close = daily_close_series_duka(df)
    close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
    cal = close.index

    log_ret = np.log(close).diff()
    # month-end index: last real duka trading day of each calendar month
    ym = cal.year * 100 + cal.month
    me_pos = pd.Series(np.arange(len(cal)), index=cal).groupby(ym).max().to_numpy()
    month_end_idx = cal[me_pos]

    s1_honest = s1_ensemble(close, month_end_idx, canary=False)
    s1_canary = s1_ensemble(close, month_end_idx, canary=True)

    return {
        "symbol": symbol, "close": close, "log_ret": log_ret, "cal": cal,
        "month_end_idx": month_end_idx, "s1_honest": s1_honest, "s1_canary": s1_canary,
    }


def build_s2_scale(inst: dict, periods_per_yr: float, w_vol: int, cap: float, canary: bool) -> pd.Series:
    return s2_vol_scale(inst["log_ret"], periods_per_yr, w_vol, cap, canary=canary)


def combined_exposure_for_cell(inst: dict, periods_per_yr: float, w_vol: int, cap: float,
                                tilts: str, fomc_dates: pd.DatetimeIndex,
                                s1_canary: bool = False, s2_canary: bool = False,
                                s3_canary: bool = False) -> pd.Series:
    s1_state = inst["s1_canary"]["s1_ensemble"] if s1_canary else inst["s1_honest"]["s1_ensemble"]
    s2_scale = build_s2_scale(inst, periods_per_yr, w_vol, cap, canary=s2_canary)
    s3_mult = s3_combined_multiplier(inst["cal"], fomc_dates, tilts, canary=s3_canary)
    combined = s1_state * s2_scale * s3_mult
    combined = combined.clip(lower=0.0, upper=cap)
    return combined


def instrument_net_returns(inst: dict, periods_per_yr: float, w_vol: int, cap: float,
                            tilts: str, fomc_dates: pd.DatetimeIndex, rt_bp: float,
                            funding_bp_per_wk: float, s1_canary: bool = False,
                            s2_canary: bool = False, s3_canary: bool = False) -> dict:
    """Combine exposure[t] (decided using info through t-1) with return[t] (realized ON day t) --
    this IS the decide-at-close/act-next-day convention: exposure computed from t-1 information is
    applied to the return REALIZED on day t itself (arr index t in both series corresponds to
    calendar day cal[t]; exposure[t]'s VALUE was set using data through cal[t-1] only, per the
    shift(1) baked into s1/s2/s3 construction above, but it lives at array position t so that
    ret[t] * exposure[t] is a simple aligned elementwise product with no further shift needed
    here)."""
    exposure = combined_exposure_for_cell(inst, periods_per_yr, w_vol, cap, tilts, fomc_dates,
                                           s1_canary=s1_canary, s2_canary=s2_canary, s3_canary=s3_canary)
    ret = inst["log_ret"].reindex(exposure.index)
    simple_ret = np.expm1(ret)  # convert log return to simple return for compounding/costs
    gross = (exposure.to_numpy() * simple_ret.to_numpy())
    costed = apply_costs(gross, exposure.to_numpy(), rt_bp, funding_bp_per_wk)
    return {
        "cal": exposure.index, "exposure": exposure.to_numpy(), "gross": gross,
        "net": costed["net"], "fee": costed["fee"], "funding": costed["funding"],
        "delta_exposure": costed["delta_exposure"], "funding_bp_per_day": costed["funding_bp_per_day"],
    }


def blend_5050(res_spy: dict, res_qqq: dict) -> dict:
    """50/50 SPY/QQQ blend: align on the intersection of both calendars (both instruments must
    have a valid observation that day), equal-weight the two instruments' net (and gross) returns.
    Turnover (delta_exposure) is equal-weighted too so cost bookkeeping stays consistent."""
    idx = res_spy["cal"].intersection(res_qqq["cal"])
    def _align(res):
        s = pd.Series(np.arange(len(res["cal"])), index=res["cal"])
        pos = s.reindex(idx).to_numpy()
        return pos
    pos_s, pos_q = _align(res_spy), _align(res_qqq)
    ok = np.isfinite(pos_s) & np.isfinite(pos_q)
    idx_ok = idx[ok]
    ps, pq = pos_s[ok].astype(int), pos_q[ok].astype(int)
    net = 0.5 * res_spy["net"][ps] + 0.5 * res_qqq["net"][pq]
    gross = 0.5 * res_spy["gross"][ps] + 0.5 * res_qqq["gross"][pq]
    exposure = 0.5 * res_spy["exposure"][ps] + 0.5 * res_qqq["exposure"][pq]
    delta_exposure = 0.5 * res_spy["delta_exposure"][ps] + 0.5 * res_qqq["delta_exposure"][pq]
    return {"cal": idx_ok, "net": net, "gross": gross, "exposure": exposure,
            "delta_exposure": delta_exposure}


# ======================================================================================
# buy-and-hold benchmark (identical daily series/conventions -- adversarial lens 2)
# ======================================================================================
def benchmark_net_returns(inst: dict, rt_bp: float, funding_bp_per_wk: float) -> dict:
    """Static exposure=1 always (buy-and-hold): the SAME log_ret series, the SAME apply_costs()
    function, at constant exposure=1.0 (an initial purchase cost on day 0 -- a single delta-
    exposure event from 0->1 -- and thereafter funding drag at exposure=1 every day, IDENTICAL in
    form to the overlay's cost machinery, just held constant instead of time-varying)."""
    cal = inst["cal"]
    ret = inst["log_ret"].reindex(cal)
    simple_ret = np.expm1(ret)
    exposure = np.ones(len(cal))
    gross = exposure * simple_ret.to_numpy()
    costed = apply_costs(gross, exposure, rt_bp, funding_bp_per_wk)
    return {"cal": cal, "exposure": exposure, "gross": gross, "net": costed["net"],
            "delta_exposure": costed["delta_exposure"]}


def blend_5050_benchmark(bm_spy: dict, bm_qqq: dict) -> dict:
    idx = bm_spy["cal"].intersection(bm_qqq["cal"])
    def _align(res):
        s = pd.Series(np.arange(len(res["cal"])), index=res["cal"])
        return s.reindex(idx).to_numpy()
    pos_s, pos_q = _align(bm_spy), _align(bm_qqq)
    ok = np.isfinite(pos_s) & np.isfinite(pos_q)
    idx_ok = idx[ok]
    ps, pq = pos_s[ok].astype(int), pos_q[ok].astype(int)
    net = 0.5 * bm_spy["net"][ps] + 0.5 * bm_qqq["net"][pq]
    gross = 0.5 * bm_spy["gross"][ps] + 0.5 * bm_qqq["gross"][pq]
    exposure = 0.5 * bm_spy["exposure"][ps] + 0.5 * bm_qqq["exposure"][pq]
    return {"cal": idx_ok, "net": net, "gross": gross, "exposure": exposure}


# ======================================================================================
# synthetic-outlier injection test (adversarial lens 1 / bug pattern 3)
# ======================================================================================
def synthetic_outlier_injection_test(inst: dict, periods_per_yr: float) -> dict:
    """Inject a large fake move at one index into a COPY of the close series and confirm no
    exposure value downstream of it (S1 state, S2 scale, combined exposure) reacts BEFORE the
    correct lag (i.e. exposure[t_inject] must be IDENTICAL pre/post injection since it was decided
    using data through t_inject-1; exposure[t_inject+1] onward MAY differ, since that is exactly
    the correct, intended one-day-later reaction)."""
    close = inst["close"].copy()
    n = len(close)
    t_inject = n // 2  # somewhere in the middle, comfortably clear of the 200d SMA warmup
    close_shocked = close.copy()
    close_shocked.iloc[t_inject] = close_shocked.iloc[t_inject] * 1.50  # +50% fake spike

    cal = close.index
    ym = cal.year * 100 + cal.month
    me_pos = pd.Series(np.arange(len(cal)), index=cal).groupby(ym).max().to_numpy()
    month_end_idx = cal[me_pos]

    s1_orig = s1_ensemble(close, month_end_idx, canary=False)["s1_ensemble"]
    s1_shocked = s1_ensemble(close_shocked, month_end_idx, canary=False)["s1_ensemble"]

    log_ret_orig = np.log(close).diff()
    log_ret_shocked = np.log(close_shocked).diff()
    s2_orig = s2_vol_scale(log_ret_orig, periods_per_yr, 60, 1.0, canary=False)
    s2_shocked = s2_vol_scale(log_ret_shocked, periods_per_yr, 60, 1.0, canary=False)

    # exposure AT t_inject and BEFORE must be unaffected (decided using data through t_inject-1);
    # exposure AFTER t_inject (t_inject+1 onward) is ALLOWED (indeed expected) to differ.
    s1_before_ok = bool(np.allclose(
        s1_orig.iloc[:t_inject + 1].to_numpy(), s1_shocked.iloc[:t_inject + 1].to_numpy(), equal_nan=True))
    s2_before_ok = bool(np.allclose(
        s2_orig.iloc[:t_inject + 1].to_numpy(), s2_shocked.iloc[:t_inject + 1].to_numpy(), equal_nan=True, atol=1e-12))
    s1_after_differs = bool(not np.allclose(
        s1_orig.iloc[t_inject + 1:t_inject + 30].to_numpy(),
        s1_shocked.iloc[t_inject + 1:t_inject + 30].to_numpy(), equal_nan=True))
    s2_after_differs = bool(not np.allclose(
        s2_orig.iloc[t_inject + 1:t_inject + 90].to_numpy(),
        s2_shocked.iloc[t_inject + 1:t_inject + 90].to_numpy(), equal_nan=True, atol=1e-12))

    return {
        "symbol": inst["symbol"], "t_inject": int(t_inject), "inject_date": str(cal[t_inject].date()),
        "s1_exposure_through_t_inject_unaffected": s1_before_ok,
        "s2_exposure_through_t_inject_unaffected": s2_before_ok,
        "s1_exposure_reacts_after_t_inject_as_expected": s1_after_differs,
        "s2_exposure_reacts_after_t_inject_as_expected": s2_after_differs,
        "PASS": bool(s1_before_ok and s2_before_ok and s1_after_differs and s2_after_differs),
    }


# ======================================================================================
# calendar-date-alignment diagnostic (printed before anything is trusted, per bug pattern 3)
# ======================================================================================
def print_alignment_diagnostic(inst: dict, w_vol: int, cap: float) -> None:
    cal = inst["cal"]
    close = inst["close"]
    s1 = inst["s1_honest"]["s1_ensemble"]
    s2 = s2_vol_scale(inst["log_ret"], 252.0, w_vol, cap, canary=False)
    print(f"  --- alignment diagnostic for {inst['symbol']} (W_vol={w_vol}, cap={cap}) ---", flush=True)
    for i in [200, 201, 202, len(cal) // 2, len(cal) - 3, len(cal) - 2, len(cal) - 1]:
        if 0 <= i < len(cal):
            print(f"    idx={i:6d} date={cal[i].date()} close={close.iloc[i]:.4f} "
                  f"s1_ensemble[t]={s1.iloc[i]:.4f} (decided using data through {cal[i-1].date() if i>0 else 'N/A'}) "
                  f"s2_scale[t]={s2.iloc[i]:.4f} (decided using data through {cal[i-1].date() if i>0 else 'N/A'}) "
                  f"applied_to_return_realized_on={cal[i].date()}", flush=True)


# ======================================================================================
# S1 regime-sanity check (mandatory, all 4 clauses, real numbers)
# ======================================================================================
def window_mask(cal: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))


def realized_drawdown(net: np.ndarray) -> float:
    if len(net) == 0:
        return 0.0
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    return float((1 - eq / peak).max()) * 100


def count_flips_and_cost(exposure: np.ndarray, delta_exposure: np.ndarray, rt_bp: float) -> dict:
    n_flip_events = int(np.sum(delta_exposure > 1e-9))
    total_delta = float(np.sum(delta_exposure))
    cost_bp_total = total_delta * rt_bp
    return {"n_exposure_change_days": n_flip_events, "sum_abs_delta_exposure": round(total_delta, 4),
            "implied_cost_bp": round(cost_bp_total, 3)}


def s1_regime_sanity(deploy_spy: dict, deploy_qqq: dict, deploy_blend: dict,
                      bm_blend: dict, cal_blend: pd.DatetimeIndex) -> dict:
    out = {}

    # (i) 2020-03 COVID drawdown reduction / exit timing
    for label, res, cal in (("S1_book_blend", deploy_blend, cal_blend), ("benchmark_blend", bm_blend, cal_blend)):
        m = window_mask(cal, "2020-02-15", "2020-04-15")
        if m.any():
            dd = realized_drawdown(res["net"][m])
            out.setdefault("covid_2020_03", {})[label] = {"maxdd_pct_in_window": round(dd, 3)}

    # exposure drop dates: first date exposure fell below 0.5 within the window, per instrument
    for sym, res in (("SPYUSDT", deploy_spy), ("QQQUSDT", deploy_qqq)):
        cal_i = res["cal"]
        m = window_mask(cal_i, "2020-01-15", "2020-04-15")
        exp = res["exposure"][m]
        dates = cal_i[m]
        drop_dates = dates[exp < 0.5]
        out.setdefault("covid_2020_03", {})[f"{sym}_exposure_drop_dates_below_0.5"] = (
            [str(d.date()) for d in drop_dates[:5]] if len(drop_dates) else [])
        out["covid_2020_03"][f"{sym}_exposure_series_feb_apr_2020"] = [
            {"date": str(d.date()), "exposure": round(float(e), 4)}
            for d, e in zip(dates, exp)
        ][::5]  # every 5th day to keep it compact

    # (ii) bull-market participation: 2020H2/2021 recovery + 2023-2024 melt-up
    for win_label, (s, e) in (("2020H2_2021_recovery", ("2020-07-01", "2021-12-31")),
                               ("2023_2024_meltup", ("2023-01-01", "2024-12-31"))):
        for sym, res in (("SPYUSDT", deploy_spy), ("QQQUSDT", deploy_qqq)):
            cal_i = res["cal"]
            m = window_mask(cal_i, s, e)
            exp = res["exposure"][m]
            if len(exp):
                out.setdefault("bull_participation", {})[f"{win_label}_{sym}"] = {
                    "mean_exposure": round(float(np.mean(exp)), 4),
                    "pct_days_at_full_exposure_ge_0.95": round(float(np.mean(exp >= 0.95)) * 100, 2),
                    "n_days": int(len(exp)),
                }

    # (iii) whipsaw cost in 2018Q4 and mid-2022
    for win_label, (s, e) in (("2018Q4_chop", ("2018-10-01", "2018-12-31")),
                               ("2022_chop_decline", ("2022-01-01", "2022-12-31"))):
        for sym, res in (("SPYUSDT", deploy_spy), ("QQQUSDT", deploy_qqq)):
            cal_i = res["cal"]
            m = window_mask(cal_i, s, e)
            de = res["delta_exposure"][m]
            exp_in_window = res["exposure"][m]
            fc = count_flips_and_cost(exp_in_window, de, DEPLOYABLE_RT_BP)
            # DATA-COVERAGE CAVEAT (mandatory honesty check, per I1_PREREG.md's own diagnostic
            # logic that absence-of-whipsaw is a red flag needing investigation): SPY/QQQ duka
            # history starts 2018-05-01, so the 200-day SMA (warms up ~2019-02-14) and the
            # 252-day/210-day monthly rules (warm up ~2019-03 to 2019-06) are NOT YET VALID during
            # 2018Q4 -- all three S1 components are structurally pinned at their fillna(0.0)
            # default for the entire window, which is a DATA-COVERAGE ARTIFACT, not a genuine
            # "the filter avoided whipsaw" finding. Flagged explicitly rather than silently
            # reported as a clean zero-flip result.
            fc["all_exposure_zero_in_window"] = bool(np.allclose(exp_in_window, 0.0))
            fc["warmup_caveat"] = (
                "S1's 200d SMA / 252d-return / 210d-SMA components require data through "
                "~2019-02 (SMA) to ~2019-06 (12mo total-return sign) before they first validate, "
                "given SPY/QQQ duka history starts 2018-05-01; a zero-flip, all-zero-exposure "
                "result in the 2018Q4 window is therefore a DATA-COVERAGE ARTIFACT (pre-warmup), "
                "NOT evidence the filter cleanly avoided whipsaw in a live chop regime."
            ) if win_label == "2018Q4_chop" else None
            out.setdefault("whipsaw_cost", {})[f"{win_label}_{sym}"] = fc

    # (iv) 2022 full-year drawdown: S1 book (blend) vs benchmark (blend)
    m = window_mask(cal_blend, "2022-01-01", "2022-12-31")
    dd_s1 = realized_drawdown(deploy_blend["net"][m])
    dd_bm = realized_drawdown(bm_blend["net"][m])
    out["2022_full_year_drawdown"] = {"S1_book_blend_maxdd_pct": round(dd_s1, 3),
                                       "benchmark_blend_maxdd_pct": round(dd_bm, 3),
                                       "S1_shallower_than_benchmark": bool(dd_s1 < dd_bm)}
    return out


# ======================================================================================
# main
# ======================================================================================
def main() -> None:
    t0 = time.time()
    print("=== I1 pipeline start ===", flush=True)

    universe = load_universe()
    universe_via_module = load_universe_via_module()
    drift_check = sorted(e["symbol"] for e in universe) == sorted(e["symbol"] for e in universe_via_module)
    print(f"universe: {len(universe)} equities (regex-load); module-load matches: {drift_check}", flush=True)

    on_demo = {e["symbol"]: e.get("on_demo") for e in universe if e["symbol"] in SYMBOLS}
    print(f"  demo gate: {on_demo}", flush=True)
    demo_gate_pass = all(bool(on_demo.get(s)) for s in SYMBOLS)
    if not demo_gate_pass:
        print(f"WARNING: demo gate FAILED for one or more of {SYMBOLS}: {on_demo}", flush=True)

    print("=== mandatory bug-pattern-1 check: verify_no_perp_rows_in_signal_panel (SPY/QQQ) ===", flush=True)
    seam_check = verify_no_perp_rows_in_signal_panel(list(SYMBOLS))
    print(json.dumps(seam_check, indent=1, default=str), flush=True)
    if not seam_check["all_duka_only"] or not all(v["PASS"] for v in seam_check["seam_checks"].values()):
        print("FATAL: seam check did not pass -- aborting before any headline number is computed.", flush=True)
        sys.exit(1)

    print("=== FOMC calendar ===", flush=True)
    fomc_dates = load_fomc_dates()
    print(f"  {len(fomc_dates)} regularly-scheduled FOMC announcement dates loaded, "
          f"{fomc_dates.min().date()} -> {fomc_dates.max().date()}", flush=True)
    emergency_dates = [pd.Timestamp("2020-03-02"), pd.Timestamp("2020-03-15"), pd.Timestamp("2019-10-04")]
    for ed in emergency_dates:
        present = ed in fomc_dates
        print(f"  spot-check: emergency/unscheduled date {ed.date()} present in scheduled list: {present} "
              f"(expected False)", flush=True)
        if present:
            print(f"  WARNING: unscheduled date {ed.date()} incorrectly present in FOMC calendar!", flush=True)

    print("=== building per-instrument daily series + S1 ensembles (SPY, QQQ) ===", flush=True)
    instruments = {sym: build_instrument(sym, fomc_dates) for sym in SYMBOLS}
    for sym, inst in instruments.items():
        print(f"  {sym}: {len(inst['cal'])} duka daily obs, {inst['cal'].min().date()} -> "
              f"{inst['cal'].max().date()}, {len(inst['month_end_idx'])} month-ends", flush=True)

    # derive periods_per_yr per instrument's own daily calendar (bug pattern 2)
    ppy_by_sym = {}
    for sym, inst in instruments.items():
        ppy_by_sym[sym] = derive_periods_per_yr(inst["cal"], f"{sym} daily calendar", expect_const=252.0)

    print("=== calendar-date-alignment diagnostic (bug pattern 3, printed BEFORE anything is trusted) ===", flush=True)
    for sym, inst in instruments.items():
        print_alignment_diagnostic(inst, DEPLOYABLE_W_VOL, DEPLOYABLE_CAP)

    print("=== synthetic-outlier injection test (adversarial lens 1) ===", flush=True)
    outlier_tests = {}
    for sym, inst in instruments.items():
        res = synthetic_outlier_injection_test(inst, ppy_by_sym[sym])
        outlier_tests[sym] = res
        print(f"  {sym}: {json.dumps(res, default=str)}", flush=True)
        if not res["PASS"]:
            print(f"FATAL: synthetic-outlier injection test FAILED for {sym} -- lookahead bug suspected, aborting.",
                  flush=True)
            sys.exit(1)

    print("=== S1/S2/S3 lookahead canary battery ===", flush=True)
    canary_results = {}
    ppy_blend_est = float(np.mean(list(ppy_by_sym.values())))

    def _run_cell(w_vol, cap, tilts, rt_bp, funding_bp_wk, s1_canary=False, s2_canary=False, s3_canary=False):
        per_sym = {}
        for sym, inst in instruments.items():
            per_sym[sym] = instrument_net_returns(
                inst, ppy_by_sym[sym], w_vol, cap, tilts, fomc_dates, rt_bp, funding_bp_wk,
                s1_canary=s1_canary, s2_canary=s2_canary, s3_canary=s3_canary)
        blend = blend_5050(per_sym["SPYUSDT"], per_sym["QQQUSDT"])
        ppy_blend = derive_periods_per_yr(blend["cal"], "blend (canary/cell)", expect_const=252.0)
        m = metrics(blend["net"], ppy_blend, turn=blend["delta_exposure"], gross=blend["gross"])
        return per_sym, blend, m

    # S1 canary: trend state computed using day/month t's OWN close (not t-1's)
    _, _, m_s1_canary = _run_cell(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, DEPLOYABLE_TILTS, DEPLOYABLE_RT_BP, 0.0,
                                   s1_canary=True)
    _, _, m_s1_honest = _run_cell(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, DEPLOYABLE_TILTS, DEPLOYABLE_RT_BP, 0.0)
    canary_results["S1_trend_state"] = {
        "honest_sharpe_ann": m_s1_honest.get("sharpe_ann"), "canary_sharpe_ann": m_s1_canary.get("sharpe_ann"),
        "canary_inflated_vs_honest": bool((m_s1_canary.get("sharpe_ann") or -999) > (m_s1_honest.get("sharpe_ann") or -999)),
    }
    print(f"  S1 canary: {json.dumps(canary_results['S1_trend_state'])}", flush=True)

    # S2 canary: vol scale uses same-day realized vol
    _, _, m_s2_canary = _run_cell(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, DEPLOYABLE_TILTS, DEPLOYABLE_RT_BP, 0.0,
                                   s2_canary=True)
    canary_results["S2_vol_scale"] = {
        "honest_sharpe_ann": m_s1_honest.get("sharpe_ann"), "canary_sharpe_ann": m_s2_canary.get("sharpe_ann"),
        "canary_inflated_vs_honest": bool((m_s2_canary.get("sharpe_ann") or -999) > (m_s1_honest.get("sharpe_ann") or -999)),
    }
    print(f"  S2 canary: {json.dumps(canary_results['S2_vol_scale'])}", flush=True)

    # S3 canary: tilt windows shifted to overlap themselves (pre-FOMC boost lands ON/after the
    # announcement instead of strictly before it -- this alone is a subtle effect on daily-cadence
    # data since it only touches ~67 single days per instrument out of ~2000; reported honestly)
    _, _, m_s3_canary = _run_cell(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, DEPLOYABLE_TILTS, DEPLOYABLE_RT_BP, 0.0,
                                   s3_canary=True)
    canary_results["S3_tilt_window"] = {
        "honest_sharpe_ann": m_s1_honest.get("sharpe_ann"), "canary_sharpe_ann": m_s3_canary.get("sharpe_ann"),
        "note": "S3 tilts touch a small number of single days (~5/month + ~1/FOMC out of ~2000 total "
                "trading days); the canary's effect size on full-sample Sharpe is expected to be much "
                "smaller than S1/S2's canaries, which corrupt the ENTIRE exposure history. Reported "
                "honestly rather than expecting the same order-of-magnitude inflation.",
    }
    print(f"  S3 canary: {json.dumps(canary_results['S3_tilt_window'])}", flush=True)

    # ================================================================================
    # full grid: S2(4) x S3(2) = 8 base construction cells, x cost/funding 3x3=9 = 72 cells
    # ================================================================================
    print("=== running full 72-cell overlay family (informational DSR/PBO input) + benchmark ===", flush=True)
    base_cells = {}  # (w_vol, cap, tilts) -> {per_sym, blend, metrics at primary cost/funding=0}
    all_cell_metrics = {}  # full label -> metrics, for N_trials=72 ledger
    pbo_input_returns = {}  # base-cell label (at primary cost, funding=0) -> net array, for PBO

    for w_vol in W_VOL_GRID:
        for cap in CAP_GRID:
            for tilts in TILT_VARIANTS:
                base_label = f"Wvol{w_vol}_cap{cap}_{tilts}"
                for rt_bp in RT_SCEN:
                    for fund_bp in FUND_SCEN_BP_PER_WK:
                        per_sym, blend, m = _run_cell(w_vol, cap, tilts, rt_bp, fund_bp)
                        label = f"{base_label}_rt{rt_bp}_fund{fund_bp}"
                        all_cell_metrics[label] = m
                        if rt_bp == RT_PRIMARY_BP and fund_bp == 0.0:
                            base_cells[(w_vol, cap, tilts)] = {"per_sym": per_sym, "blend": blend, "metrics": m}
                            pbo_input_returns[base_label] = blend["net"]
    print(f"  {len(all_cell_metrics)} overlay trial-cells computed (expected 8 base x 9 cost/funding = 72)",
          flush=True)
    assert len(all_cell_metrics) == N_TRIALS, f"cell count mismatch: {len(all_cell_metrics)} != {N_TRIALS}"

    # benchmark: 50/50 SPY/QQQ buy-and-hold, same cost/funding grid (for fairness, not separately
    # credited as trials -- it's the comparator, per prereg)
    print("=== benchmark: 50/50 SPY/QQQ buy-and-hold (identical daily series/conventions) ===", flush=True)
    bm_cells = {}
    for rt_bp in RT_SCEN:
        for fund_bp in FUND_SCEN_BP_PER_WK:
            bm_spy = benchmark_net_returns(instruments["SPYUSDT"], rt_bp, fund_bp)
            bm_qqq = benchmark_net_returns(instruments["QQQUSDT"], rt_bp, fund_bp)
            bm_blend = blend_5050_benchmark(bm_spy, bm_qqq)
            ppy_bm = derive_periods_per_yr(bm_blend["cal"], f"benchmark blend rt{rt_bp} fund{fund_bp}", expect_const=252.0)
            m_bm = metrics(bm_blend["net"], ppy_bm, gross=bm_blend["gross"])
            bm_cells[f"rt{rt_bp}_fund{fund_bp}"] = {"blend": bm_blend, "metrics": m_bm}

    bm_primary = bm_cells["rt13.0_fund0.0"]
    print(f"  benchmark (13bp, funding=0) headline: {json.dumps(bm_primary['metrics'], default=str)}", flush=True)

    # ================================================================================
    # deployable: Wvol=60, cap=1.0, with_tilts, 13bp primary, funding=0
    # ================================================================================
    deploy_cell = base_cells[(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, DEPLOYABLE_TILTS)]
    deploy_blend = deploy_cell["blend"]
    deploy_metrics = deploy_cell["metrics"]
    deploy_spy = deploy_cell["per_sym"]["SPYUSDT"]
    deploy_qqq = deploy_cell["per_sym"]["QQQUSDT"]
    print(f"=== DEPLOYABLE headline (Wvol={DEPLOYABLE_W_VOL}, cap={DEPLOYABLE_CAP}, "
          f"tilts={DEPLOYABLE_TILTS}, rt=13bp, funding=0): {json.dumps(deploy_metrics, default=str)} ===",
          flush=True)

    # no-tilts counterpart for separable attribution
    no_tilt_cell = base_cells[(DEPLOYABLE_W_VOL, DEPLOYABLE_CAP, "no_tilts")]
    print(f"=== DEPLOYABLE (no tilts) headline: {json.dumps(no_tilt_cell['metrics'], default=str)} ===", flush=True)

    # per-instrument decomposition (diagnostic, not separate trials)
    ppy_spy = ppy_by_sym["SPYUSDT"]
    ppy_qqq = ppy_by_sym["QQQUSDT"]
    spy_only_metrics = metrics(deploy_spy["net"], ppy_spy, turn=deploy_spy["delta_exposure"], gross=deploy_spy["gross"])
    qqq_only_metrics = metrics(deploy_qqq["net"], ppy_qqq, turn=deploy_qqq["delta_exposure"], gross=deploy_qqq["gross"])
    print(f"  SPY-only deployable-cell metrics: {json.dumps(spy_only_metrics, default=str)}", flush=True)
    print(f"  QQQ-only deployable-cell metrics: {json.dumps(qqq_only_metrics, default=str)}", flush=True)

    # ================================================================================
    # CPCV(8,2,embargo>=1mo) for the deployable
    # ================================================================================
    print("=== CPCV(8 groups, k=2, embargo>=1mo) for the deployable ===", flush=True)
    ppy_blend = derive_periods_per_yr(deploy_blend["cal"], "deployable blend", expect_const=252.0)
    embargo_days = 21  # ~1 trading month
    cpcv_sh = cpcv_paths(deploy_blend["net"], n_groups=8, k=2, embargo=embargo_days, periods_per_yr=ppy_blend)
    cpcv_summary = {
        "n_paths": len(cpcv_sh), "mean_sharpe": round(float(np.mean(cpcv_sh)), 4) if cpcv_sh else None,
        "median_sharpe": round(float(np.median(cpcv_sh)), 4) if cpcv_sh else None,
        "min_sharpe": round(float(np.min(cpcv_sh)), 4) if cpcv_sh else None,
        "max_sharpe": round(float(np.max(cpcv_sh)), 4) if cpcv_sh else None,
        "pct_paths_positive": round(float(np.mean(np.array(cpcv_sh) > 0)) * 100, 2) if cpcv_sh else None,
    }
    print(f"  CPCV summary: {json.dumps(cpcv_summary, default=str)}", flush=True)

    # ================================================================================
    # DSR (informational, full N_trials=72) + PBO (informational, CSCV over 8-cell family)
    # ================================================================================
    print("=== DSR (informational, N_trials=72) ===", flush=True)
    dsr_result = dsr(deploy_blend["net"], N_TRIALS, ppy_blend)
    print(f"  DSR (deployable, N_trials={N_TRIALS}): {json.dumps(dsr_result, default=str)}", flush=True)
    print("  PROGRAM CLASS NOTE: DSR is expected to be LOW at breadth=2 (n=2 instruments, ~8yr "
          "sample) regardless of whether the underlying overlay is a genuinely good idea -- this "
          "is a mathematical reality of the sample size, NOT a gating criterion for this program.",
          flush=True)

    print("=== PBO (CSCV-style, informational, over the 8-cell base-construction family) ===", flush=True)
    pbo_result = pbo_cscv(pbo_input_returns, n_splits=8)
    print(f"  PBO (8-cell family): {pbo_result}", flush=True)

    # ================================================================================
    # drop-top-5-periods
    # ================================================================================
    deploy_drop5_sharpe = deploy_metrics.get("drop5_sharpe")
    bm_full_sharpe = bm_primary["metrics"].get("sharpe_ann")
    print(f"=== drop-top-5-periods: deployable={deploy_drop5_sharpe} vs benchmark(undropped)={bm_full_sharpe} ===",
          flush=True)

    # ================================================================================
    # per-year and per-regime metrics
    # ================================================================================
    print("=== per-year Sharpe (deployable blend vs benchmark blend) ===", flush=True)
    per_year_deploy = per_year_sharpe(deploy_blend["net"], deploy_blend["cal"], ppy_blend)
    ppy_bm_primary = derive_periods_per_yr(bm_primary["blend"]["cal"], "benchmark primary blend", expect_const=252.0)
    per_year_bm = per_year_sharpe(bm_primary["blend"]["net"], bm_primary["blend"]["cal"], ppy_bm_primary)
    print(f"  deployable per-year: {per_year_deploy}", flush=True)
    print(f"  benchmark per-year:  {per_year_bm}", flush=True)

    print("=== per-regime metrics (deployable blend vs benchmark blend) ===", flush=True)
    per_regime = {}
    cal_blend = deploy_blend["cal"]
    cal_bm = bm_primary["blend"]["cal"]
    for label, (s, e) in REGIME_WINDOWS.items():
        m_d = window_mask(cal_blend, s, e)
        m_b = window_mask(cal_bm, s, e)
        net_d = deploy_blend["net"][m_d]
        net_b = bm_primary["blend"]["net"][m_b]
        ppy_d = derive_periods_per_yr(cal_blend[m_d], f"regime {label} deployable", expect_const=252.0) if m_d.sum() > 2 else 252.0
        ppy_b = derive_periods_per_yr(cal_bm[m_b], f"regime {label} benchmark", expect_const=252.0) if m_b.sum() > 2 else 252.0
        per_regime[label] = {
            "deployable": metrics(net_d, ppy_d) if m_d.sum() >= 5 else {"n_obs": int(m_d.sum())},
            "benchmark": metrics(net_b, ppy_b) if m_b.sum() >= 5 else {"n_obs": int(m_b.sum())},
        }
    print(json.dumps(per_regime, indent=1, default=str), flush=True)

    # ================================================================================
    # S1 regime-sanity check (mandatory, 4 clauses, real numbers)
    # ================================================================================
    print("=== S1 REGIME-SANITY CHECK (mandatory, real numbers, all 4 clauses) ===", flush=True)
    regime_sanity = s1_regime_sanity(deploy_spy, deploy_qqq, deploy_blend, bm_primary["blend"], cal_blend)
    print(json.dumps(regime_sanity, indent=1, default=str), flush=True)

    # ================================================================================
    # ADOPTION GATE (benchmark-relative, 6 sub-checks)
    # ================================================================================
    print("=== ADOPTION GATE (benchmark-relative, frozen) ===", flush=True)
    bm_sharpe = bm_primary["metrics"].get("sharpe_ann") or 0.0
    bm_maxdd = bm_primary["metrics"].get("maxdd_pct") or 0.0
    deploy_sharpe = deploy_metrics.get("sharpe_ann") or -999
    deploy_maxdd = deploy_metrics.get("maxdd_pct") or 999

    gate1_sharpe = bool(deploy_sharpe > bm_sharpe + 0.10)
    gate2_dd = bool(deploy_maxdd < 0.65 * bm_maxdd)

    # Gate 3 checks whipsaw cost is VISIBLE in the one 2018Q4/2022 window where S1 is actually
    # WARMED UP and tradeable (2022_chop_decline) -- the 2018Q4 window is EXCLUDED from this
    # specific pass/fail check because it is a data-coverage artifact (S1's components have not
    # yet warmed up: SPY/QQQ duka history only starts 2018-05-01, and the 200d-SMA/252d-return/
    # 210d-SMA components do not validate until ~2019-02/2019-06), so an all-zero-exposure,
    # zero-flip result there is NOT evidence of "no whipsaw" and must not be credited as passing
    # this check -- it is reported plainly as a caveat instead (see whipsaw_cost.*.warmup_caveat).
    whipsaw_2022_has_flips = any(
        v.get("n_exposure_change_days", 0) > 0
        for k, v in regime_sanity["whipsaw_cost"].items() if k.startswith("2022_chop_decline"))
    whipsaw_2018q4_all_zero_warmup_artifact = all(
        v.get("all_exposure_zero_in_window", False)
        for k, v in regime_sanity["whipsaw_cost"].items() if k.startswith("2018Q4_chop"))
    gate3_regime = bool(
        regime_sanity["2022_full_year_drawdown"]["S1_shallower_than_benchmark"]
        and whipsaw_2022_has_flips
    )

    # gate 4: robustness -- every one of the 8 base-construction cells beats benchmark net sharpe
    robustness_detail = {}
    for (w_vol, cap, tilts), cell in base_cells.items():
        cell_sharpe = cell["metrics"].get("sharpe_ann") or -999
        robustness_detail[f"Wvol{w_vol}_cap{cap}_{tilts}"] = {
            "sharpe_ann": cell_sharpe, "beats_benchmark": bool(cell_sharpe > bm_sharpe)
        }
    gate4_robust = bool(all(v["beats_benchmark"] for v in robustness_detail.values()))

    gate5_drop5 = bool((deploy_drop5_sharpe or -999) > (bm_full_sharpe or 999))

    # Turnover metric (DECLARED, per I1_PREREG.md "roughly 5-15 full ROUND-TRIP-EQUIVALENT exposure
    # changes per year"): a full round trip (exposure 0->1->0) sums |delta-exposure| to 2.0 across
    # the two legs. The correct operationalization is therefore sum(|delta-exposure|) / 2, per
    # year -- NOT a raw count of days with any nonzero delta (S2's continuous daily-recomputed
    # vol-target scale makes almost every day show a tiny nonzero delta, which would conflate
    # smooth vol-target micro-adjustments with discrete round trips and wildly overstate "trade
    # count"). Also reported as a diagnostic cross-check: the day-count version, and S1-ensemble-
    # only flip frequency (the discrete trend-flip mechanism this gate is really sanity-checking).
    n_years_span = (cal_blend[-1] - cal_blend[0]).days / 365.25
    sum_abs_delta_exposure = float(np.sum(deploy_blend["delta_exposure"]))
    turnover_events_per_year = sum_abs_delta_exposure / 2.0 / n_years_span
    turnover_raw_day_count_per_year = float(np.sum(deploy_blend["delta_exposure"] > 1e-9) / n_years_span)
    s1_only_flips_per_year = {}
    for sym, inst in instruments.items():
        d = inst["s1_honest"]["s1_ensemble"].diff().abs()
        ny = (inst["cal"][-1] - inst["cal"][0]).days / 365.25
        s1_only_flips_per_year[sym] = round(float((d > 1e-9).sum() / ny), 3)
    gate6_turnover = bool(5 <= turnover_events_per_year <= 15)

    gate = {
        "1_net_sharpe_gt_benchmark_plus_0.10": {
            "deployable_sharpe_ann": deploy_sharpe, "benchmark_sharpe_ann": bm_sharpe,
            "threshold": round(bm_sharpe + 0.10, 4), "PASS": gate1_sharpe,
        },
        "2_maxdd_lt_0.65x_benchmark": {
            "deployable_maxdd_pct": deploy_maxdd, "benchmark_maxdd_pct": bm_maxdd,
            "threshold_pct": round(0.65 * bm_maxdd, 3), "PASS": gate2_dd,
        },
        "3_regime_sanity_holds": {
            "2022_full_year_drawdown": regime_sanity["2022_full_year_drawdown"],
            "whipsaw_2022_chop_decline_has_visible_flips": whipsaw_2022_has_flips,
            "whipsaw_2018Q4_all_zero_is_warmup_artifact_not_evaluated": whipsaw_2018q4_all_zero_warmup_artifact,
            "PASS": gate3_regime,
        },
        "4_robustness_all_8_cells_beat_benchmark": {"detail": robustness_detail, "PASS": gate4_robust},
        "5_drop5_sharpe_beats_benchmark_undropped": {
            "deployable_drop5_sharpe": deploy_drop5_sharpe, "benchmark_undropped_sharpe": bm_full_sharpe,
            "PASS": gate5_drop5,
        },
        "6_turnover_sanity_5_to_15_per_year": {
            "turnover_events_per_year_round_trip_equivalent": round(turnover_events_per_year, 4),
            "definition": "sum(|delta_exposure|) / 2 / n_years -- a full 0->1->0 round trip sums "
                           "|delta_exposure| to 2.0; this is the correct operationalization of "
                           "I1_PREREG.md's 'full round-trip-equivalent exposure changes per year', "
                           "NOT a raw day-count of any nonzero delta (S2's continuously-recomputed "
                           "vol-target scale makes nearly every day show some tiny nonzero delta).",
            "diagnostic_raw_day_count_with_any_nonzero_delta_per_year": round(turnover_raw_day_count_per_year, 3),
            "diagnostic_s1_ensemble_only_discrete_flips_per_year": s1_only_flips_per_year,
            "sum_abs_delta_exposure_full_sample": round(sum_abs_delta_exposure, 4),
            "PASS": gate6_turnover,
        },
    }
    all_pass = bool(gate1_sharpe and gate2_dd and gate3_regime and gate4_robust and gate5_drop5 and gate6_turnover)
    gate["ADOPTION_RECOMMENDATION"] = ("PAPER-FIRST DEPLOY (separate, separately-capitalized third book)"
                                        if all_pass else "DO NOT DEPLOY (negative/failed gate)")
    print(json.dumps(gate, indent=1, default=str), flush=True)

    # ================================================================================
    # assemble + write results
    # ================================================================================
    results = {
        "meta": {
            "n_trials_ledger": N_TRIALS,
            "program_class": "OVERLAY-VALIDATION (benchmark-relative gate), NOT edge-certification; "
                              "DSR/PBO informational only, expected low at breadth=2",
            "symbols": list(SYMBOLS),
            "demo_gate": on_demo, "demo_gate_pass": demo_gate_pass,
            "candidates_md_drift_check_passed": drift_check,
            "seam_check": seam_check,
            "fomc_calendar": {"n_meetings": int(len(fomc_dates)),
                               "span": [str(fomc_dates.min().date()), str(fomc_dates.max().date())],
                               "emergency_spotchecks_excluded": {str(ed.date()): bool(ed not in fomc_dates) for ed in emergency_dates}},
            "periods_per_yr_by_symbol": ppy_by_sym,
            "cost_model": {"primary_rt_bp": RT_PRIMARY_BP, "rt_scenarios_bp": list(RT_SCEN),
                           "funding_scenarios_bp_per_wk": list(FUND_SCEN_BP_PER_WK),
                           "funding_proration_convention": "funding_bp_per_day = funding_bp_per_wk / 7 (calendar days)"},
            "deployable_spec": {"w_vol": DEPLOYABLE_W_VOL, "cap": DEPLOYABLE_CAP, "tilts": DEPLOYABLE_TILTS,
                                 "rt_bp": DEPLOYABLE_RT_BP, "funding_bp_per_wk": 0.0},
        },
        "synthetic_outlier_injection_test": outlier_tests,
        "lookahead_canaries": canary_results,
        "deployable": {
            "with_tilts": deploy_metrics,
            "no_tilts": no_tilt_cell["metrics"],
            "spy_only": spy_only_metrics,
            "qqq_only": qqq_only_metrics,
        },
        "benchmark": {k: v["metrics"] for k, v in bm_cells.items()},
        "full_grid_72_cells": all_cell_metrics,
        "robustness_family_8_base_cells": {f"Wvol{w}_cap{c}_{t}": base_cells[(w, c, t)]["metrics"]
                                            for w in W_VOL_GRID for c in CAP_GRID for t in TILT_VARIANTS},
        "cpcv_deployable": cpcv_summary,
        "dsr_informational": dsr_result,
        "pbo_informational": pbo_result,
        "per_year_sharpe": {"deployable": per_year_deploy, "benchmark": per_year_bm},
        "per_regime_metrics": per_regime,
        "s1_regime_sanity": regime_sanity,
        "adoption_gate": gate,
    }

    results["meta"]["runtime_sec"] = round(time.time() - t0, 2)
    (OUT / "results_i1.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"=== DONE in {time.time() - t0:.1f}s -- wrote {OUT / 'results_i1.json'} ===", flush=True)


if __name__ == "__main__":
    main()
