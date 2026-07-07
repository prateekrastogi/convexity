#!/usr/bin/env python
"""E3 CROSS-SECTIONAL EQUITY FACTOR SUITE (research/equity/E3_PREREG.md, frozen 2026-07-07).

THREE-sleeve program: S1 slow sector-residual momentum, S2 betting-against-beta (Frazzini-
Pedersen), S3 price-based post-earnings-announcement drift (PEAD). Base universe = 93 names
(102-with-real-history -> 99 after short-history exclusion -> 93 after leveraged/inverse/decay-ETF
exclusion). S3's own universe = 76-name PEAD-usable subset. N_trials = 48 (S1:4 + S2:3 + S3:4 +
combined:1 + cost/funding grid on 4 headline cells x 3 cost x 3 funding = 36). Combined pre-named
deployable: equal-risk-contribution blend of S1 ensemble + S2 W_beta=52 + S3 ensemble, single 10%
vol-target overlay on top, 13bp primary cost.

AVOIDING E1'S AND E2'S FIVE NAMED BUG PATTERNS (declared and checked explicitly, not assumed):
  (1) Source-aware returns: every stitched parquet filtered to source=='duka' BEFORE any
      resampling/return computation, reusing E2's verify_no_perp_rows_in_signal_panel() verbatim.
  (2) Derived annualization: periods_per_yr = n_obs/span_years computed FRESH from each net-return
      array's own date index, never hardcoded. S1/S2 monthly -> expect ~12. S3's event-portfolio
      series gets its OWN derivation with no assumed constant.
  (3) Exposure-timing alignment: S2's beta estimate for month t uses ONLY data strictly before
      month t's rebalance (through the prior week's close); the combined book's vol-target overlay
      for month t uses only trailing vol strictly before month t begins (net[lo:t] half-open,
      reusing overlay.py::vol_target_scale's exact convention). Diagnostic printed before any
      number is trusted -- this is the exact bug class that survived E2's own first-pass
      verification (its stress-exposure was read from week k's END instead of START).
  (4) Costs scale with realized exposure: BOTH the trading-fee term (turn_scaled = turn_raw * es)
      AND the funding term (funding_bp_per_wk/1e4 * es) scale with the combined book's realized
      exposure-scale array `es`, a direct fix of E2's documented flat-funding-term bug.
  (5) Leveraged/inverse/decay ETF exclusion: UVXYUSDT, SQQQUSDT, TQQQUSDT, SOXLUSDT, KORUUSDT,
      MVLLUSDT excluded from the base universe for ALL THREE sleeves (99 -> 93).

Path hygiene: this file points at ITS OWN worktree's data/{stitched,earnings} and
src/convexity/panel/candidates.md paths directly; does NOT call
convexity.panel.panel.build_panel()/load_bars() for the core signal (load_bars() still does not
read the `source` column, confirmed unchanged since E2). A fresh, source-filtering builder
(reused near-verbatim from e2_pipeline.py) is used instead.

Run: nohup /home/speculation/convexity/.venv/bin/python research/equity/e3_pipeline.py \
       > research/equity/e3.out 2>&1 &
Writes: research/equity/results_e3.json
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity-equity")
DATA_STITCHED = REPO / "data/stitched"
DATA_EARNINGS = REPO / "data/earnings"
OUT = REPO / "research/equity"
CANDIDATES_MD = REPO / "src/convexity/panel/candidates.md"

DAY_PER_YR = 252.0
MONTH_PER_YR = 12.0
SIGMA_STAR = 0.10
N_TRIALS = 48  # frozen ledger, E3_PREREG.md

RT_PRIMARY_BP = 13.0
RT_SCEN = (6.5, 13.0, 26.0)
FUND_SCEN_BP_PER_WK = (0.0, 5.0, -5.0)

# ----------------------------------------------------------------------------------------
# universe construction (frozen, E3_PREREG.md "Base universe")
# ----------------------------------------------------------------------------------------
EXCLUDED_PERP_ONLY = {
    "ANTHROPICUSDT", "BBXUSDT", "BSPUSDT", "HYUNDAIUSDT",
    "OPENAIUSDT", "QNTXUSDT", "SAMSUNGUSDT", "SKHYNIXUSDT",
}
EXCLUDED_SHORT_HISTORY = {"STXXUSDT", "CBRSUSDT", "DRAMUSDT"}
EXCLUDED_LEVERAGED_ETF = {
    "UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT",
}

ON_DEMO_SYMS_RAW = {
    "AMZNUSDT", "COINUSDT", "CRCLUSDT", "EWJUSDT", "GOOGLUSDT", "HOODUSDT", "INTCUSDT",
    "METAUSDT", "MSTRUSDT", "MVLLUSDT", "NVDAUSDT", "OPENAIUSDT", "PAYPUSDT", "PLTRUSDT",
    "QNTXUSDT", "QQQUSDT", "SPCXUSDT", "SPYUSDT", "TSLAUSDT",
}
# on_demo AND real-history AND not-leveraged (16 names for S1/S2, per E3_PREREG.md)
ON_DEMO_S1S2_EXPECTED = {
    "AMZNUSDT", "COINUSDT", "CRCLUSDT", "EWJUSDT", "GOOGLUSDT", "HOODUSDT", "INTCUSDT",
    "METAUSDT", "MSTRUSDT", "NVDAUSDT", "PAYPUSDT", "PLTRUSDT", "QQQUSDT", "SPCXUSDT",
    "SPYUSDT", "TSLAUSDT",
}
# on_demo-S1S2 intersected with the 76-name PEAD-usable set (11 names for S3)
ON_DEMO_S3_EXPECTED = {
    "AMZNUSDT", "COINUSDT", "GOOGLUSDT", "HOODUSDT", "INTCUSDT", "METAUSDT",
    "MSTRUSDT", "NVDAUSDT", "PAYPUSDT", "PLTRUSDT", "TSLAUSDT",
}

MIN_SECTOR_N = 4  # sectors with fewer PIT members that period fold into "other"
STRESS_WINDOWS = {
    "2018Q4": ("2018-10-01", "2018-12-31"),
    "2020-03_COVID": ("2020-02-15", "2020-04-15"),
    "2022_drawdown": ("2022-01-01", "2022-12-31"),
    "2024-08_vol_spike": ("2024-08-01", "2024-08-31"),
}

# S1 formation grid (months)
S1_LOOKBACKS_L = (3, 6, 12)
S1_SKIP_MONTHS = 1

# S2 beta-window grid (weeks)
S2_BETA_WINDOWS = (26, 52, 104)
S2_PRIMARY_W_BETA = 52
S2_VOL_WIN_MONTHS = 12  # trailing-12-month realized vol for leg sizing

# S3 holding-period grid (weeks)
S3_HOLD_WEEKS = (2, 4, 6)
S3_MIN_EVENTS = 8
S3_COVERAGE_FLOOR = 0.60


# ======================================================================================
# reused validation math (per_obs_sharpe / dsr / metrics / cpcv_paths / pbo_cscv /
# per_year_sharpe) -- reproduced near-verbatim from e2_pipeline.py's own
# parameterized-by-periods_per_yr function bodies, per METHODOLOGY's explicit reuse mandate.
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


def metrics(net: np.ndarray, periods_per_yr: float, turn: np.ndarray | None = None) -> dict:
    net = net[np.isfinite(net)]
    if len(net) < 30:
        return {"n_obs": int(len(net))}
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    yrs = len(net) / periods_per_yr
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    vt = vol_target(net, periods_per_yr, win=max(int(periods_per_yr / 2), 10))
    eq_vt = np.cumprod(1 + vt)
    cagr_vt = eq_vt[-1] ** (1 / yrs) - 1 if eq_vt[-1] > 0 else -1.0
    srt = np.sort(net)[::-1]
    return {
        "n_obs": int(len(net)),
        "sharpe_ann": round(float(per_obs_sharpe(net) * np.sqrt(periods_per_yr)), 4),
        "sharpe_obs": round(float(per_obs_sharpe(net)), 5),
        "ev_bp_obs": round(float(net.mean()) * 1e4, 3),
        "win_pct": round(float((net > 0).mean()) * 100, 2),
        "maxdd_pct": round(float((1 - eq / peak).max()) * 100, 2),
        "cagr_pct": round(float(cagr) * 100, 3),
        "cagr_voltarget_pct": round(float(cagr_vt) * 100, 3),
        "drop5_sharpe": round(float(per_obs_sharpe(np.sort(net)[:-5]) * np.sqrt(periods_per_yr)), 4) if len(net) > 10 else None,
        "top5_share": round(float(srt[:5].sum() / net.sum()), 3) if net.sum() != 0 else None,
        "turn_mean": round(float(np.nanmean(turn)), 4) if turn is not None else None,
    }


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
    median."""
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


def derive_periods_per_yr(dates: pd.DatetimeIndex, label: str, expect_const: float | None = None,
                           tol: float = 0.15) -> float:
    """Derive periods_per_yr from the ACTUAL observed calendar spacing of a net-return array's
    date index -- per E3_PREREG.md's mandatory bug-pattern-2 check, never a hardcoded literal.
    n_obs / span_in_years. If expect_const is given, asserts within `tol` relative error and warns
    (does not abort) otherwise, printing the full arithmetic (n_obs, span_years, derived) so it is
    auditable."""
    n_obs = len(dates)
    if n_obs < 2:
        return expect_const if expect_const is not None else MONTH_PER_YR
    span_years = (dates[-1] - dates[0]).days / 365.25
    derived = n_obs / span_years if span_years > 0 else (expect_const or MONTH_PER_YR)
    msg = f"    [periods_per_yr derivation] {label}: n_obs={n_obs} span_years={span_years:.4f} derived={derived:.4f}"
    if expect_const is not None:
        msg += f" (expected~{expect_const:.4f})"
    print(msg, flush=True)
    if expect_const is not None and expect_const > 0:
        rel_err = abs(derived - expect_const) / expect_const
        if rel_err > tol:
            print(f"    WARNING: {label} derived periods_per_yr {derived:.4f} is >{tol:.0%} off "
                  f"the expected constant {expect_const:.4f} (rel_err={rel_err:.3%})", flush=True)
    return derived


def block_bootstrap_sharpe_ci(net: np.ndarray, periods_per_yr: float, block: int = 3,
                               n_boot: int = 1000, ci: float = 0.90, seed: int = 42) -> dict:
    net = net[np.isfinite(net)]
    n = len(net)
    if n < block * 4:
        return {"lo": None, "hi": None, "median": None, "n_boot_eff": 0}
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        sample = np.concatenate([net[s:s + block] for s in starts])[:n]
        boots[b] = per_obs_sharpe(sample) * np.sqrt(periods_per_yr)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        "lo": round(float(np.quantile(boots, lo_q)), 4),
        "hi": round(float(np.quantile(boots, hi_q)), 4),
        "median": round(float(np.median(boots)), 4),
        "n_boot_eff": n_boot,
    }


# ======================================================================================
# universe / data loading
# ======================================================================================
def load_universe() -> list[dict]:
    txt = CANDIDATES_MD.read_text()
    m = re.search(r'"equities":\s*(\[.*?\n  \])', txt, re.S)
    if not m:
        raise RuntimeError("could not locate equities JSON array in candidates.md")
    return json.loads(m.group(1))


def load_universe_via_module() -> list[dict]:
    """Cross-check via convexity.panel.candidates.load_universe() -- a fresh drift-detection call
    in THIS session, per E3_PREREG.md's mandate to re-verify landmine 5's exact tag list live, not
    from memory."""
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
    """Mandatory check (reused verbatim from e2_pipeline.py, per METHODOLOGY): (a) every row
    consumed by the daily-close builder has source=='duka' for every sampled symbol; (b) for
    PAYPUSDT specifically, directly confirm the known splice seam is present in the raw file, and
    that the source=='duka'-filtered daily-close series this pipeline actually builds terminates
    at/before the last duka day and contains zero perp-sourced closes."""
    out = {"checked_symbols": [], "all_duka_only": True, "paypusdt_seam_check": None}
    for sym in sample_symbols:
        df = load_stitched_duka(sym)
        ok = bool((df["source"] == "duka").all())
        out["checked_symbols"].append({"symbol": sym, "n_rows": int(len(df)), "all_duka": ok})
        if not ok:
            out["all_duka_only"] = False

    raw = pd.read_parquet(DATA_STITCHED / "PAYPUSDT.parquet")
    seam_rows = raw[raw["open_time"] == 1774276200000]
    seam_confirmed = False
    seam_detail = {}
    if not seam_rows.empty:
        row = seam_rows.iloc[0]
        prev = raw[raw["open_time"] < 1774276200000].iloc[-1]
        seam_confirmed = bool(row["source"] == "perp" and prev["source"] == "duka"
                               and bool(row["gap_before"]) is False)
        seam_detail = {
            "open_time": int(row["open_time"]), "source_at_seam": str(row["source"]),
            "source_prev_row": str(prev["source"]), "gap_before_at_seam": bool(row["gap_before"]),
        }
    filtered = load_stitched_duka("PAYPUSDT")
    last_duka_ts = filtered["ts"].max()
    daily_close = daily_close_series_duka(filtered)
    no_perp_in_daily_close = bool(daily_close.index.max() <= last_duka_ts.floor("D"))
    out["paypusdt_seam_check"] = {
        "raw_seam_confirmed_present_in_lake": seam_confirmed,
        "seam_detail": seam_detail,
        "last_duka_ts": str(last_duka_ts),
        "constructed_daily_close_series_max_day": str(daily_close.index.max()),
        "daily_close_terminates_at_or_before_last_duka_day": no_perp_in_daily_close,
        "PASS": bool(seam_confirmed and no_perp_in_daily_close and out["all_duka_only"]),
    }
    return out


# ======================================================================================
# source-aware daily close + return construction (E2's builder, reused verbatim)
# ======================================================================================
def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    last_idx = d.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


def daily_returns_from_close(close: pd.Series) -> pd.Series:
    idx = close.index
    ret = close.pct_change()
    gap_days = idx.to_series().diff().dt.days
    ret[gap_days > 5] = np.nan
    return ret


def build_name_series(universe: list[dict], excluded: set[str]) -> dict:
    """Per-name: daily close (duka-only), daily return, PIT first/last real duka day, dollar
    volume at daily frequency (for S3's liquidity-split regime check)."""
    out = {}
    for e in universe:
        sym = e["symbol"]
        if sym in excluded or not e.get("dukascopy"):
            continue
        try:
            df = load_stitched_duka(sym)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        dclose = daily_close_series_duka(df)
        if len(dclose) < 12 * 22 + 40:  # >= ~1yr of daily obs + buffer, generous floor
            continue
        dret = daily_returns_from_close(dclose)
        d = df.copy()
        d["day"] = d["ts"].dt.floor("D")
        dvol_dollar = (d["close"] * d["volume"]).groupby(d["day"]).sum()
        dvol_dollar = dvol_dollar.reindex(dclose.index).fillna(0.0)
        out[sym] = {
            "sector": e["sector"], "daily_close": dclose, "daily_ret": dret,
            "dollar_vol": dvol_dollar, "first_day": dclose.index.min(), "last_day": dclose.index.max(),
        }
    return out


def build_daily_panel(name_series: dict) -> dict:
    syms = sorted(name_series.keys())
    sectors = [name_series[s]["sector"] for s in syms]
    all_days = sorted(set().union(*[set(v["daily_ret"].index) for v in name_series.values()]))
    cal = pd.DatetimeIndex(all_days)
    n, m = len(cal), len(syms)
    RET = np.full((n, m), np.nan)
    DVOL = np.full((n, m), np.nan)
    CLOSE = np.full((n, m), np.nan)
    LISTED = np.zeros((n, m), dtype=bool)
    for j, s in enumerate(syms):
        v = name_series[s]
        RET[:, j] = v["daily_ret"].reindex(cal).to_numpy()
        DVOL[:, j] = v["dollar_vol"].reindex(cal).to_numpy()
        CLOSE[:, j] = v["daily_close"].reindex(cal).to_numpy()
        LISTED[:, j] = (cal >= v["first_day"]) & (cal <= v["last_day"])
    return {"cal": cal, "syms": syms, "sectors": sectors, "ret": RET, "dvol": DVOL,
            "close": CLOSE, "listed": LISTED}


# ======================================================================================
# sector residualization (recomputed fresh per DAY, reused verbatim from e2_pipeline.py)
# ======================================================================================
def residualize_daily_fast(RET: np.ndarray, LISTED: np.ndarray, sectors: list[str]) -> np.ndarray:
    n, m = RET.shape
    sectors_arr = np.array(sectors)
    uniq_sectors = sorted(set(sectors))
    sec_cols = {sec: np.where(sectors_arr == sec)[0] for sec in uniq_sectors}
    RESID = np.full((n, m), np.nan)
    listed_and_finite = LISTED & np.isfinite(RET)
    for t in range(n):
        lt = listed_and_finite[t]
        if not lt.any():
            continue
        assigned = np.zeros(m, dtype=bool)
        for sec, cols in sec_cols.items():
            members = cols[lt[cols]]
            if len(members) >= MIN_SECTOR_N:
                grp_mean = RET[t, members].mean()
                RESID[t, members] = RET[t, members] - grp_mean
                assigned[members] = True
        other = np.where(lt & ~assigned)[0]
        if len(other) > 0:
            grp_mean = RET[t, other].mean()
            RESID[t, other] = RET[t, other] - grp_mean
    return RESID


def residualize_daily_ref(RET: np.ndarray, LISTED: np.ndarray, sectors: list[str]) -> np.ndarray:
    """Slow reference implementation for a subset-equivalence spot check against the fast path."""
    n, m = RET.shape
    sectors_arr = np.array(sectors)
    uniq_sectors = sorted(set(sectors))
    RESID = np.full((n, m), np.nan)
    for t in range(n):
        listed_t = LISTED[t] & np.isfinite(RET[t])
        if not listed_t.any():
            continue
        group_of = {}
        for sec in uniq_sectors:
            members = np.where(listed_t & (sectors_arr == sec))[0]
            if len(members) >= MIN_SECTOR_N:
                group_of[sec] = members
        assigned = set()
        for sec, members in group_of.items():
            assigned.update(members.tolist())
        other_members = np.array([i for i in np.where(listed_t)[0] if i not in assigned])
        groups = list(group_of.values())
        if len(other_members) > 0:
            groups.append(other_members)
        for members in groups:
            if len(members) == 0:
                continue
            grp_mean = np.nanmean(RET[t, members])
            RESID[t, members] = RET[t, members] - grp_mean
    return RESID


def _weights_from_signal_momentum(signal: np.ndarray, elig: np.ndarray, rvol: np.ndarray) -> np.ndarray:
    """S1 MOMENTUM book -- SIGN CHECK, stated explicitly (E3_PREREG.md): this is the OPPOSITE sign
    from E2's dead reversal sleeve. LONG the TOP quintile (biggest signal = biggest residual
    WINNER), SHORT the BOTTOM quintile (biggest residual LOSER). Same rank-sort direction as E2's
    `_weights_from_signal` (ascending order, most-negative first) but the OPPOSITE quintile
    assignment: E2 longs `order[:q]` (losers) and shorts `order[-q:]` (winners); S1 longs
    `order[-q:]` (winners) and shorts `order[:q]` (losers). Inverse trailing-vol leg-weighting,
    floored at 0.5x cross-sectional median vol that period (E2's floor convention, reused)."""
    n = len(signal)
    w = np.zeros(n)
    idx = np.where(elig)[0]
    if len(idx) < 2:
        return w
    q = max(1, int(len(idx) * 0.2))
    order = np.argsort(signal[idx])  # ascending: most-negative (biggest loser) first
    lo_idx = idx[order[:q]]   # biggest losers -> SHORT (momentum)
    hi_idx = idx[order[-q:]]  # biggest winners -> LONG (momentum)
    vfloor = 0.5 * np.nanmedian(rvol[idx]) if np.isfinite(rvol[idx]).any() else 1.0
    vfloor = vfloor if vfloor > 0 else 1e-6
    inv_lo = 1.0 / np.maximum(np.where(np.isfinite(rvol[lo_idx]) & (rvol[lo_idx] > 0), rvol[lo_idx], vfloor), vfloor)
    inv_hi = 1.0 / np.maximum(np.where(np.isfinite(rvol[hi_idx]) & (rvol[hi_idx] > 0), rvol[hi_idx], vfloor), vfloor)
    w[hi_idx] = 0.5 * inv_hi / inv_hi.sum()    # LONG winners
    w[lo_idx] = -0.5 * inv_lo / inv_lo.sum()   # SHORT losers
    return w


def sign_check_momentum_assert() -> None:
    """Explicit unit assertion (E3_PREREG.md mandate): S1's weight function must LONG the highest-
    signal (top quintile) names and SHORT the lowest-signal (bottom quintile) names -- the OPPOSITE
    of E2's reversal `_weights_from_signal`, which longs the bottom quintile. Run once at pipeline
    start, before any P&L is computed."""
    signal = np.array([-5.0, -3.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    elig = np.ones(10, dtype=bool)
    rvol = np.ones(10)
    w = _weights_from_signal_momentum(signal, elig, rvol)
    top_idx = np.argsort(signal)[-2:]   # two highest-signal names (q=2 for n=10)
    bot_idx = np.argsort(signal)[:2]    # two lowest-signal names
    assert np.all(w[top_idx] > 0), f"S1 sign-check FAILED: top-quintile names must be LONG (w>0), got {w[top_idx]}"
    assert np.all(w[bot_idx] < 0), f"S1 sign-check FAILED: bottom-quintile names must be SHORT (w<0), got {w[bot_idx]}"
    assert abs(w.sum()) < 1e-9, f"S1 sign-check FAILED: book must be dollar-neutral, got net={w.sum()}"
    print(f"  [S1 sign-check PASS] top-quintile (highest signal) weights={w[top_idx]} (LONG, >0); "
          f"bottom-quintile (lowest signal) weights={w[bot_idx]} (SHORT, <0); net={w.sum():.2e}", flush=True)


# ======================================================================================
# monthly calendar helpers (S1/S2 rebalance = last real duka trading day of each month)
# ======================================================================================
def month_end_indices(cal: pd.DatetimeIndex) -> np.ndarray:
    """Indices into `cal` of each calendar month's LAST trading day (real duka day present in the
    unioned panel calendar) -- the exact day convention declared in E3_PREREG.md."""
    ym = cal.year * 100 + cal.month
    df = pd.DataFrame({"ym": ym, "idx": np.arange(len(cal))})
    return df.groupby("ym")["idx"].max().to_numpy()


def month_end_returns_matrix(RET: np.ndarray, cal: pd.DatetimeIndex, me_idx: np.ndarray) -> np.ndarray:
    """Compounded return over each (month-end[k-1], month-end[k]] window per name -- i.e. the
    calendar month's realized return, aligned so row k is "the return realized DURING month k",
    dated at month k's own month-end."""
    n_m = len(me_idx) - 1
    m = RET.shape[1]
    out = np.full((n_m, m), np.nan)
    for k in range(n_m):
        d0, d1 = me_idx[k], me_idx[k + 1]
        seg = RET[d0 + 1:d1 + 1]
        if seg.shape[0] == 0:
            continue
        seg_f = np.nan_to_num(seg, nan=0.0)
        out[k] = np.prod(1.0 + seg_f, axis=0) - 1.0
    return out


# ======================================================================================
# S1: slow sector-residual momentum
# ======================================================================================
def s1_trailing_formation_signal(RESID_M: np.ndarray, L: int, skip: int) -> np.ndarray:
    """Formation window [t-L-skip, t-skip) in MONTHLY-return-index terms: signal at month-end row
    t = sum of monthly residual returns over months [t-L-skip .. t-skip-1] inclusive (L months of
    data, ending `skip` months before t). Implemented as a trailing sum of a SHIFTED (by `skip`)
    monthly residual-return series, min_periods=L (no partial windows)."""
    df = pd.DataFrame(RESID_M)
    shifted = df.shift(skip)  # month t's usable window ends at t-skip
    return shifted.rolling(L, min_periods=L).sum().to_numpy()


def s1_build_book(RESID_M: np.ndarray, RET_M: np.ndarray, LISTED_M: np.ndarray, L: int,
                   vol_win_months: int = 12, canary: bool = False) -> np.ndarray:
    """Monthly-rebalanced quintile momentum book for formation lookback L (months), with mandatory
    skip=1 month. Position sizing: inverse trailing-12-month realized vol, floored at 0.5x
    cross-sectional median. If canary=True, ranks on month t's OWN forward monthly return (the
    period-t RET_M row itself) instead of the formation-window signal -- a genuine, undiluted
    lookahead peeking at the same-period return the book is about to be scored on."""
    n_m, m = RET_M.shape
    rvol = pd.DataFrame(RET_M).rolling(vol_win_months, min_periods=max(3, vol_win_months // 2)).std().shift(1).to_numpy()
    if canary:
        sig = RET_M  # peeks at month t's own forward return directly
    else:
        sig = s1_trailing_formation_signal(RESID_M, L, S1_SKIP_MONTHS)
    W = np.zeros((n_m, m))
    for t in range(n_m):
        sig_t = sig[t]
        elig = LISTED_M[t] & np.isfinite(sig_t) & np.isfinite(rvol[t]) & (rvol[t] > 0)
        if elig.sum() >= 2:
            W[t] = _weights_from_signal_momentum(sig_t, elig, rvol[t])
    return W


def score_monthly_book(W: np.ndarray, RET_M: np.ndarray, month_end_dates: pd.DatetimeIndex) -> dict:
    """Score a monthly weight matrix (weights SET at month t's OWN formation date, i.e. row t of W
    already only uses info through month t-skip's close, per s1_build_book) against month t's
    REALIZED return RET_M[t]. This is NOT a lookahead: W[t] uses data ending `skip` months before
    t's return is realized; RET_M[t] is the return REALIZED during month t, exactly what the
    formation signal (computed through t-skip) is being bet on. Turnover = gross weight change
    month over month."""
    n_m, m = W.shape
    net = np.full(n_m, np.nan)
    turn = np.full(n_m, np.nan)
    for t in range(n_m):
        w = W[t]
        r = RET_M[t]
        if not np.isfinite(w).any():
            continue
        r_f = np.nan_to_num(r, nan=0.0)
        net[t] = float(np.nansum(w * r_f))
        w_prev = W[t - 1] if t > 0 else np.zeros(m)
        turn[t] = float(np.abs(w - w_prev).sum())
    return {"net": net, "turn": turn, "dates": month_end_dates}


def s1_ensemble_weights(books: dict[int, np.ndarray]) -> np.ndarray:
    We = np.mean(np.stack([books[L] for L in S1_LOOKBACKS_L]), axis=0)
    gross = np.abs(We).sum(axis=1, keepdims=True)
    We = np.divide(We, gross, out=np.zeros_like(We), where=gross > 0)
    return We


# ======================================================================================
# S2: betting-against-beta (Frazzini-Pedersen)
# ======================================================================================
def build_weekly_returns(name_series: dict, syms: list[str]) -> dict:
    """Per-name weekly (Friday-close, as-of backward-filled) LOG returns for the beta-estimation
    market proxy and rolling OLS beta -- a separate, weekly-native calendar from S1/S2's monthly
    rebalance-and-mark grid, per the prereg's own W_beta-in-WEEKS declaration."""
    all_days = sorted(set().union(*[set(name_series[s]["daily_close"].index) for s in syms]))
    day_cal = pd.DatetimeIndex(all_days)
    fridays = pd.date_range(day_cal.min(), day_cal.max(), freq="W-FRI", tz="UTC")
    n_f, m = len(fridays), len(syms)
    CLOSE_F = np.full((n_f, m), np.nan)
    LISTED_F = np.zeros((n_f, m), dtype=bool)
    for j, s in enumerate(syms):
        dc = name_series[s]["daily_close"]
        as_of = dc.reindex(dc.index.union(fridays)).sort_index().ffill().reindex(fridays)
        CLOSE_F[:, j] = as_of.to_numpy()
        first_day, last_day = name_series[s]["first_day"], name_series[s]["last_day"]
        LISTED_F[:, j] = (fridays >= first_day) & (fridays <= last_day)
    with np.errstate(divide="ignore", invalid="ignore"):
        LOGRET_F = np.diff(np.log(CLOSE_F), axis=0, prepend=np.nan)
    LOGRET_F[~LISTED_F] = np.nan
    LOGRET_F[0, :] = np.nan
    return {"fridays": fridays, "logret": LOGRET_F, "listed": LISTED_F}


def build_market_proxy_weekly(LOGRET_F: np.ndarray, LISTED_F: np.ndarray) -> np.ndarray:
    """Equal-weight-of-base-universe weekly log-return proxy, point-in-time (a name only enters the
    average from its own first real duka week)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        masked = np.where(LISTED_F, LOGRET_F, np.nan)
        mkt = np.nanmean(masked, axis=1)
    return mkt


def rolling_beta_weekly(LOGRET_F: np.ndarray, mkt: np.ndarray, w_beta: int) -> np.ndarray:
    """Rolling OLS beta of each name's weekly log-return against the market proxy, over a trailing
    w_beta-week window ENDING STRICTLY BEFORE the current week (i.e. beta[t] uses weeks
    [t-w_beta, t) only -- landmine 3: the beta used to size month t's rebalance must never include
    the week the position is about to earn). Implemented as beta[t] = cov(ret,mkt)/var(mkt) over
    LOGRET_F[t-w_beta:t]."""
    n_f, m = LOGRET_F.shape
    BETA = np.full((n_f, m), np.nan)
    for t in range(w_beta, n_f):
        window_mkt = mkt[t - w_beta:t]
        window_ret = LOGRET_F[t - w_beta:t, :]
        finite_m = np.isfinite(window_mkt)
        if finite_m.sum() < max(10, w_beta // 3):
            continue
        mc = window_mkt[finite_m] - np.nanmean(window_mkt[finite_m])
        var_m = float(np.sum(mc ** 2))
        if var_m <= 0:
            continue
        for j in range(m):
            rj = window_ret[finite_m, j]
            ok = np.isfinite(rj)
            if ok.sum() < max(10, w_beta // 3):
                continue
            rjc = rj[ok] - np.mean(rj[ok])
            BETA[t, j] = float(np.sum(rjc * mc[ok]) / var_m)
    return BETA


def s2_build_book(BETA_F: np.ndarray, LOGRET_F: np.ndarray, LISTED_F: np.ndarray,
                   fridays: pd.DatetimeIndex, month_end_fridays_idx: np.ndarray,
                   vol_win_weeks: int = 52, canary: bool = False) -> dict:
    """Monthly-rebalanced (using the Friday at/before each month-end) true Frazzini-Pedersen book.
    Order of operations (declared in E3_PREREG.md): (a) beta-leverage-adjust each leg FIRST
    (long-leg weight *= 1/beta_long_leg, short-leg weight *= 1/beta_short_leg, using each leg's own
    realized average decile beta at that rebalance); (b) inverse trailing-12-month(~52-week)
    realized-vol sizing WITHIN each already-adjusted leg, floored at 0.5x cross-sectional median
    vol; (c) balance the two legs' dollar exposures so the COMBINED book's ex-ante beta ~ 0.
    canary=True: rank on the week's OWN forward weekly return (peeking), decile-sort on that
    instead of BETA_F -- a genuine forward-looking sort, not a diluted proxy."""
    n_f, m = LOGRET_F.shape
    rvol_f = pd.DataFrame(LOGRET_F).rolling(vol_win_weeks, min_periods=max(8, vol_win_weeks // 4)).std().shift(1).to_numpy()
    W = np.zeros((n_f, m))
    ex_ante_beta = np.full(n_f, np.nan)
    beta_long_leg_series = np.full(n_f, np.nan)
    beta_short_leg_series = np.full(n_f, np.nan)
    last_w = np.zeros(m)

    if canary:
        # forward weekly return array (peek): fwd[t] = LOGRET_F[t+1] realized the following week
        fwd = np.roll(LOGRET_F, -1, axis=0)
        fwd[-1] = np.nan

    for t in range(n_f):
        if t not in month_end_fridays_idx:
            W[t] = last_w
            continue
        beta_t = BETA_F[t]
        elig = LISTED_F[t] & np.isfinite(beta_t) & np.isfinite(rvol_f[t]) & (rvol_f[t] > 0)
        if elig.sum() < 10:
            W[t] = last_w
            continue
        idx = np.where(elig)[0]
        if canary:
            rank_sig = fwd[t]
            if not np.isfinite(rank_sig[idx]).sum() >= 10:
                W[t] = last_w
                continue
            order = np.argsort(rank_sig[idx])  # ascending: worst-forward-return first
            dec = max(1, int(len(idx) * 0.1))
            # canary LONGS names about to do BEST, SHORTS names about to do WORST (peeked)
            long_idx = idx[order[-dec:]]
            short_idx = idx[order[:dec]]
        else:
            order = np.argsort(beta_t[idx])  # ascending: lowest beta first
            dec = max(1, int(len(idx) * 0.1))
            long_idx = idx[order[:dec]]   # LOW beta -> LONG
            short_idx = idx[order[-dec:]]  # HIGH beta -> SHORT

        beta_long_leg = float(np.nanmean(beta_t[long_idx]))
        beta_short_leg = float(np.nanmean(beta_t[short_idx]))
        beta_long_leg_series[t] = beta_long_leg
        beta_short_leg_series[t] = beta_short_leg

        # (a) beta-leverage-adjust each leg FIRST
        lev_long = 1.0 / beta_long_leg if abs(beta_long_leg) > 1e-6 else 0.0
        lev_short = 1.0 / beta_short_leg if abs(beta_short_leg) > 1e-6 else 0.0

        # (b) inverse-vol sizing WITHIN each already-leverage-adjusted leg, floored
        vfloor = 0.5 * np.nanmedian(rvol_f[t, idx]) if np.isfinite(rvol_f[t, idx]).any() else 1.0
        vfloor = vfloor if vfloor > 0 else 1e-6
        rv_long = np.maximum(np.where(np.isfinite(rvol_f[t, long_idx]) & (rvol_f[t, long_idx] > 0),
                                       rvol_f[t, long_idx], vfloor), vfloor)
        rv_short = np.maximum(np.where(np.isfinite(rvol_f[t, short_idx]) & (rvol_f[t, short_idx] > 0),
                                        rvol_f[t, short_idx], vfloor), vfloor)
        inv_vol_long = (1.0 / rv_long)
        inv_vol_long = inv_vol_long / inv_vol_long.sum()
        inv_vol_short = (1.0 / rv_short)
        inv_vol_short = inv_vol_short / inv_vol_short.sum()

        raw_long_w = lev_long * inv_vol_long   # per-name, before final dollar-balance
        raw_short_w = -lev_short * inv_vol_short

        # (c) balance dollar exposures so the COMBINED book's ex-ante beta ~ 0.
        #
        # BUG FOUND AND FIXED (during mandatory pre-run smoke-testing of the mandatory ex-post-beta
        # adversarial check -- landmine 3/5's instruction to re-derive, not re-trust-by-analogy):
        # the original implementation normalized EACH leg's raw weights to a fixed 0.5 gross
        # (`0.5 * raw_long_w / abs(raw_long_w).sum()`), which discards `lev_long`'s MAGNITUDE
        # entirely (since `abs(raw_long_w).sum() == abs(lev_long) * sum(inv_vol_long) ==
        # abs(lev_long)` when inv_vol_long is already normalized to sum 1, the division cancels
        # lev_long down to just its SIGN) -- i.e. the leverage-adjustment step (a) was computed and
        # then silently thrown away in step (c), leaving a naive 50/50-dollar low-beta/high-beta
        # book with no actual beta-neutralization. Verified empirically: this produced a realized
        # ex-post book beta of -0.69 (far from zero), which the mandatory adversarial check
        # correctly flagged as a construction defect rather than genuine BAB alpha.
        #
        # FIX: keep each leg's OWN internal relative sizing from inv_vol (already sums to 1 within
        # the leg), and size the leg's TOTAL dollar exposure (long_gross for the long leg,
        # short_gross for the short leg, both positive magnitudes) using its OWN beta-leverage
        # factor relative to the other leg's, so that the combined book's net beta contribution
        # (long_gross*beta_long_leg from the long leg, MINUS short_gross*beta_short_leg from the
        # short leg since its weights are negative) is zero, while long_gross+short_gross==1 (unit
        # total gross). Solving {long_gross + short_gross = 1, long_gross*beta_long_leg -
        # short_gross*beta_short_leg = 0} gives:
        #   long_gross  = beta_short_leg / (beta_long_leg + beta_short_leg)
        #   short_gross = beta_long_leg  / (beta_long_leg + beta_short_leg)
        # (verified algebraically and with a synthetic numeric example: beta_long_leg=0.4,
        # beta_short_leg=1.6 -> long_gross=0.8, short_gross=0.2, net beta =
        # 0.8*0.4 - 0.2*1.6 = 0.0 exactly.) Requires beta_long_leg and beta_short_leg to have the
        # SAME sign and not both be ~0 (the normal case: both legs are equity book betas, typically
        # positive); falls back to a naive 50/50 split if that condition fails.
        w = np.zeros(m)
        beta_sum = beta_long_leg + beta_short_leg
        same_sign_and_nonzero = (beta_long_leg * beta_short_leg > 0) and abs(beta_sum) > 1e-6
        if same_sign_and_nonzero:
            long_gross = beta_short_leg / beta_sum
            long_gross = float(np.clip(long_gross, 0.05, 0.95))  # keep both legs meaningfully sized
            short_gross = 1.0 - long_gross
        else:
            long_gross, short_gross = 0.5, 0.5
        w[long_idx] = long_gross * inv_vol_long    # inv_vol_long already sums to 1 -> leg sums to long_gross
        w[short_idx] = -short_gross * inv_vol_short  # inv_vol_short already sums to 1 -> leg sums to -short_gross

        ex_ante_beta[t] = float(np.nansum(w * beta_t))
        last_w = w
        W[t] = w
    return {"W": W, "ex_ante_beta": ex_ante_beta, "beta_long_leg": beta_long_leg_series,
            "beta_short_leg": beta_short_leg_series}


def score_weekly_book_to_monthly(W_f: np.ndarray, LOGRET_F: np.ndarray, fridays: pd.DatetimeIndex,
                                  month_end_fridays_idx: np.ndarray) -> dict:
    """Score S2's weekly-held-monthly-rebalanced book: weights are set at each month-end Friday
    and held constant week-to-week until the NEXT month-end Friday's rebalance (already encoded in
    W_f via the forward-fill in s2_build_book's `else: W[t] = last_w` branch). Monthly net return =
    compounded weekly simple return (exp(logret)-1) over the weeks STRICTLY AFTER month-end[k]
    through month-end[k+1], weighted by the weights SET AT month-end[k] (W_f[month_end_fridays_idx[k]],
    the same weights held through the whole following month -- no lookahead: month k's return uses
    only the weight decided at month k's own rebalance)."""
    n_me = len(month_end_fridays_idx) - 1
    m = W_f.shape[1]
    net = np.full(n_me, np.nan)
    turn = np.full(n_me, np.nan)
    dates = []
    simple_ret_f = np.expm1(LOGRET_F)
    for k in range(n_me):
        f0, f1 = month_end_fridays_idx[k], month_end_fridays_idx[k + 1]
        w = W_f[f0]
        seg = simple_ret_f[f0 + 1:f1 + 1]
        dates.append(fridays[f1])
        if seg.shape[0] == 0 or not np.isfinite(w).any():
            continue
        seg_f = np.nan_to_num(seg, nan=0.0)
        name_month_ret = np.prod(1.0 + seg_f, axis=0) - 1.0
        net[k] = float(np.nansum(w * name_month_ret))
        w_next = W_f[f1] if f1 < W_f.shape[0] else w
        turn[k] = float(np.abs(w_next - w).sum())
    return {"net": net, "turn": turn, "dates": pd.DatetimeIndex(dates)}


def realized_ex_post_beta(net: np.ndarray, mkt_month: np.ndarray) -> float:
    """Regression of the book's own return series against the market proxy, over the full
    (aligned, finite) sample -- the mandatory adversarial check (E3_PREREG.md)."""
    ok = np.isfinite(net) & np.isfinite(mkt_month)
    if ok.sum() < 10:
        return float("nan")
    x = mkt_month[ok] - mkt_month[ok].mean()
    y = net[ok] - net[ok].mean()
    var_x = float(np.sum(x ** 2))
    if var_x <= 0:
        return float("nan")
    return float(np.sum(x * y) / var_x)


def apply_costs(net_gross: np.ndarray, turn: np.ndarray, rt_bp: float, funding_bp_per_period: float = 0.0,
                 exposure_scale: np.ndarray | None = None) -> np.ndarray:
    """Cost model: trading fee scales with realized turnover (already exposure-scaled upstream if
    the caller passed exposure-scaled turn); funding term ALSO scales with realized exposure
    (`exposure_scale`), per landmine 4 -- a flat funding term regardless of gross exposure is the
    exact bug E2 documented. If `exposure_scale` is None, exposure is implicitly 1.0 (unscaled
    book) and the funding term reduces to the flat-per-period convention correctly (1.0x is a
    genuine realized exposure, not a bug)."""
    es = exposure_scale if exposure_scale is not None else np.ones_like(net_gross)
    fee = (rt_bp / 1e4) * turn
    funding = (funding_bp_per_period / 1e4) * es
    return net_gross - fee - funding


# ======================================================================================
# S3: price-based PEAD
# ======================================================================================
def load_earnings_events(symbol: str) -> pd.DataFrame | None:
    fp = DATA_EARNINGS / f"{symbol}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce")
    df = df.dropna(subset=["Earnings Date"])
    df["timing"] = df["timing"].fillna("UNKNOWN")
    df = df.drop_duplicates(subset=["Earnings Date"], keep="first")
    df = df.sort_values("Earnings Date").reset_index(drop=True)
    return df


def s3_usable_universe(candidates: list[dict], base_93: set[str]) -> tuple[set[str], dict]:
    """Re-derive the >=8-usable-event floor fresh from data/earnings/*.csv, within
    [2018-01-01, 2026-07-04], restricted to the 93-name base universe -- verifying (not assuming)
    the 76-name figure declared in E3_PREREG.md / e3_pead_data_coverage.json."""
    start, end = pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2026-07-04", tz="UTC")
    counts = {}
    usable = set()
    for sym in sorted(base_93):
        df = load_earnings_events(sym)
        if df is None:
            counts[sym] = 0
            continue
        ed = df["Earnings Date"]
        if ed.dt.tz is None:
            ed = ed.dt.tz_localize("UTC")
        in_window = df[(ed >= start) & (ed <= end)]
        counts[sym] = int(len(in_window))
        if counts[sym] >= S3_MIN_EVENTS:
            usable.add(sym)
    coverage_fraction = len(usable) / len(base_93) if base_93 else 0.0
    return usable, {"counts": counts, "coverage_fraction": coverage_fraction, "n_usable": len(usable),
                     "n_base": len(base_93)}


def classify_window(ts: pd.Timestamp, timing: str) -> tuple[int, int]:
    """Returns (offset_start, offset_end) in TRADING-DAY offsets from the announcement date t:
    confirmed AMC -> (0, +1) i.e. close[t] to close[t+1]; else (BMO or UNKNOWN, conservative) ->
    (-1, +1) i.e. close[t-1] to close[t+1]."""
    if timing == "AMC":
        return (0, 1)
    return (-1, 1)  # BMO or UNKNOWN: conservative t-1 -> t+1 window, per declared convention


def s3_build_events(name_series: dict, earnings_usable: set[str], cal: pd.DatetimeIndex,
                     RET: np.ndarray, LISTED: np.ndarray, sectors: list[str], syms: list[str],
                     DVOL: np.ndarray, RESID: np.ndarray) -> list[dict]:
    """Construct one event record per (symbol, announcement date) with: abnormal announcement-
    window return (name return minus sector-group mean return over the window, using the SAME
    per-day sector-fold-to-'other' RESID convention as S1/S2, at daily granularity), entry index
    (first tradeable day strictly AFTER the window closes), and trailing-20-day dollar volume as of
    the entry day (for the liquidity-split regime check)."""
    sym_to_col = {s: j for j, s in enumerate(syms)}
    cal_pos = {d: i for i, d in enumerate(cal)}
    events = []
    for sym in sorted(earnings_usable):
        if sym not in sym_to_col:
            continue
        j = sym_to_col[sym]
        df = load_earnings_events(sym)
        if df is None:
            continue
        first_day, last_day = name_series[sym]["first_day"], name_series[sym]["last_day"]
        for _, row in df.iterrows():
            ann_ts = row["Earnings Date"]
            if ann_ts.tzinfo is None:
                ann_ts = ann_ts.tz_localize("UTC")
            ann_day = ann_ts.floor("D")
            if ann_day < first_day or ann_day > last_day:
                continue
            # find t = index of the first trading day AT/AFTER ann_day in the panel calendar
            pos = cal.searchsorted(ann_day)
            if pos >= len(cal) or pos == 0:
                continue
            t = pos if cal[pos] == ann_day else pos  # t = trading day on/after announcement
            off_start, off_end = classify_window(ann_day, row["timing"])
            t_start, t_end = t + off_start, t + off_end
            if t_start < 0 or t_end >= len(cal):
                continue
            entry_idx = t_end + 1  # first tradeable day AFTER the window closes
            if entry_idx >= len(cal):
                continue
            # abnormal return over (t_start, t_end]: name's own compounded return minus sector
            # peer-group's compounded mean return over the SAME window (using RESID's per-day
            # group construction: sum of RESID over the window == name ret minus group mean, when
            # compounded multiplicatively we approximate via sum-of-daily-residuals, consistent
            # with S1's own additive trailing-sum convention)
            window = slice(t_start + 1, t_end + 1)
            resid_seg = RESID[window, j]
            if not np.isfinite(resid_seg).all() or len(resid_seg) == 0:
                continue
            abn_ret = float(np.nansum(resid_seg))
            if not LISTED[entry_idx, j]:
                continue
            trail_dvol = DVOL[max(0, entry_idx - 20):entry_idx, j]
            trail_dvol_mean = float(np.nanmean(trail_dvol)) if len(trail_dvol) > 0 else np.nan
            events.append({
                "symbol": sym, "sector": sectors[j], "col": j, "ann_day": ann_day,
                "t_idx": t, "entry_idx": entry_idx, "abn_ret": abn_ret,
                "timing": row["timing"], "trail_dvol": trail_dvol_mean,
            })
    return events


def s3_assign_terciles(events: list[dict]) -> list[dict]:
    """Rank events into terciles of announcement abnormal return WITHIN each rebalance cross-
    section. Since S3's events are asynchronous (not a shared periodic grid), terciles are formed
    over a trailing rolling window of events (the most recent ~63 trading days' worth of events
    across all names) at each event's own entry date -- a genuinely point-in-time cross-sectional
    rank using only events whose window has ALREADY closed by that entry date."""
    events_sorted = sorted(events, key=lambda e: e["entry_idx"])
    n = len(events_sorted)
    entry_idxs = np.array([e["entry_idx"] for e in events_sorted])
    abn_rets = np.array([e["abn_ret"] for e in events_sorted])
    LOOKBACK_EVENTS_WINDOW = 63  # trading days, ~1 quarter, matches the event-driven cadence
    for i, e in enumerate(events_sorted):
        lo = np.searchsorted(entry_idxs, entry_idxs[i] - LOOKBACK_EVENTS_WINDOW, side="left")
        hi = i + 1  # events up to and including this one (this one's own window already closed)
        peer_rets = abn_rets[lo:hi]
        if len(peer_rets) < 6:
            e["tercile"] = None
            continue
        rank = (peer_rets < abn_rets[i]).sum() / len(peer_rets)
        if rank >= 2.0 / 3.0:
            e["tercile"] = "top"
        elif rank < 1.0 / 3.0:
            e["tercile"] = "bottom"
        else:
            e["tercile"] = "mid"
    return events_sorted


def s3_build_event_portfolio(events: list[dict], RET: np.ndarray, cal: pd.DatetimeIndex,
                              syms: list[str], hold_weeks: int, canary: bool = False) -> dict:
    """Build a daily-marked rolling event-portfolio net-return series: for each qualifying
    (top/bottom-tercile) event, open an equal-risk (inverse trailing-vol) position at the entry
    day's close, LONG top-tercile / SHORT bottom-tercile (drift = same sign as surprise), held for
    `hold_weeks` weeks (5*hold_weeks trading days), exit at horizon (no stop-loss). Overlapping
    events form a rolling book; book gross normalized to unit each day by the number of concurrently
    live positions weighted by their own inverse-vol size.

    ``canary`` here only controls whether the caller has already substituted a peeked
    ``entry_idx`` into each event dict (the mandatory S3 lookahead canary backdates entry INTO the
    announcement window itself -- built by main() via a separate `events_canary` list with
    `entry_idx` replaced by the window's own t_idx BEFORE calling this function, so this function's
    own logic is unchanged; the flag is kept for readability/logging only, no branching on it
    happens inside this function)."""
    hold_days = hold_weeks * 5
    n = len(cal)
    m = len(syms)
    # per-day book weight matrix (position sizes), and per-day list of (open,close) event legs for
    # turnover tracking
    W = np.zeros((n, m))
    rvol_daily = pd.DataFrame(RET).rolling(20, min_periods=10).std().shift(1).to_numpy()
    live_events = []
    for e in events:
        if e["tercile"] not in ("top", "bottom"):
            continue
        entry_idx = e["entry_idx"]
        exit_idx = entry_idx + hold_days
        if entry_idx >= n or exit_idx > n:
            exit_idx = min(exit_idx, n)
        if entry_idx >= n:
            continue
        j = e["col"]
        rv = rvol_daily[entry_idx, j] if entry_idx < n and np.isfinite(rvol_daily[entry_idx, j]) and rvol_daily[entry_idx, j] > 0 else np.nan
        sign = 1.0 if e["tercile"] == "top" else -1.0
        live_events.append({"j": j, "entry_idx": entry_idx, "exit_idx": exit_idx, "sign": sign, "rv": rv})

    if not live_events:
        return {"net": np.array([]), "turn": np.array([]), "dates": pd.DatetimeIndex([]),
                "net_exposure": np.array([]), "n_events": 0}

    rv_all = np.array([le["rv"] for le in live_events])
    rv_finite = rv_all[np.isfinite(rv_all)]
    vfloor = 0.5 * np.nanmedian(rv_finite) if len(rv_finite) > 0 else 1e-3
    vfloor = vfloor if vfloor > 0 else 1e-3
    for le in live_events:
        rv = le["rv"] if np.isfinite(le["rv"]) and le["rv"] > 0 else vfloor
        rv = max(rv, vfloor)
        le["inv_vol"] = 1.0 / rv

    # build per-day RAW (pre-normalization) inverse-vol-weighted signed exposure, then normalize
    # gross to 1.0 each day across all LIVE legs that day (equal-risk per live position, unit gross
    # book) -- iterate day by day since number of live legs changes daily.
    day_to_legs: dict[int, list] = {}
    for le in live_events:
        for d in range(le["entry_idx"], le["exit_idx"]):
            day_to_legs.setdefault(d, []).append(le)

    net_exposure = np.zeros(n)
    for t in range(n):
        legs = day_to_legs.get(t, [])
        if not legs:
            continue
        raw = np.array([le["sign"] * le["inv_vol"] for le in legs])
        gross = np.abs(raw).sum()
        if gross <= 0:
            continue
        norm = raw / gross
        for le, wgt in zip(legs, norm):
            W[t, le["j"]] = wgt
        net_exposure[t] = float(norm.sum())  # signed net (long minus short), honestly reported

    RET_f = np.nan_to_num(RET, nan=0.0)
    net = np.full(n, np.nan)
    turn = np.full(n, np.nan)
    for t in range(1, n):
        net[t] = float(np.nansum(W[t - 1] * RET_f[t]))
        turn[t] = float(np.abs(W[t] - W[t - 1]).sum())
    # keep only days where a position was actually live the prior day (book has P&L to report) --
    # net[t] is only a real, tradeable observation if the book held nonzero weight going into day t
    live_mask = np.zeros(n, dtype=bool)
    live_mask[1:] = (np.abs(W[:-1]).sum(axis=1) > 0)
    net_out = np.where(live_mask, net, np.nan)
    return {"net": net_out, "turn": turn, "dates": cal, "net_exposure": net_exposure,
            "n_events": len(live_events)}


def s3_ensemble_book(events: list[dict], RET: np.ndarray, cal: pd.DatetimeIndex, syms: list[str],
                      canary: bool = False) -> dict:
    """Equal-weight band ensemble across the 3 holding periods {2,4,6} weeks: average the 3
    per-holding-period daily net-return series (each already unit-gross) on days where at least one
    is live; report the realized average net exposure honestly (not forced to zero)."""
    per_hp = {hw: s3_build_event_portfolio(events, RET, cal, syms, hw, canary=canary) for hw in S3_HOLD_WEEKS}
    n = len(cal)
    stacked_net = np.full((3, n), np.nan)
    stacked_exposure = np.zeros((3, n))
    for i, hw in enumerate(S3_HOLD_WEEKS):
        stacked_net[i] = per_hp[hw]["net"]
        stacked_exposure[i] = per_hp[hw]["net_exposure"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ens_net = np.nanmean(stacked_net, axis=0)
    ens_exposure = np.mean(stacked_exposure, axis=0)
    live_any = np.any(np.isfinite(stacked_net), axis=0)
    ens_net = np.where(live_any, ens_net, np.nan)
    # turnover of the ensemble: average of per-hp turnovers (approximation consistent with
    # equal-weight averaging of the underlying books)
    stacked_turn = np.full((3, n), np.nan)
    for i, hw in enumerate(S3_HOLD_WEEKS):
        stacked_turn[i] = per_hp[hw]["turn"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ens_turn = np.nanmean(stacked_turn, axis=0)
    return {"net": ens_net, "turn": ens_turn, "dates": cal, "net_exposure": ens_exposure,
            "per_hp": per_hp}


def s3_extract_marked_series(net_daily: np.ndarray, turn_daily: np.ndarray, cal: pd.DatetimeIndex,
                              exposure_daily: np.ndarray | None = None) -> dict:
    """Extract the actually-marked (non-NaN) observations from a daily event-portfolio net-return
    array -- this IS the honest observation spacing for S3's OWN periods_per_yr derivation (bug
    pattern 2), whatever it turns out to be; no assumption that it's daily/weekly/monthly."""
    valid = np.isfinite(net_daily)
    net = net_daily[valid]
    dates = cal[valid]
    turn = turn_daily[valid] if turn_daily is not None else None
    exposure = exposure_daily[valid] if exposure_daily is not None else None
    return {"net": net, "dates": dates, "turn": turn, "exposure": exposure}


# ======================================================================================
# combined deployable: equal-risk-contribution blend + 10% vol-target overlay
# ======================================================================================
def align_series_by_year_month(values: np.ndarray, dates: pd.DatetimeIndex,
                                target_month_ends: pd.DatetimeIndex) -> np.ndarray:
    """Align a (value, date) series onto `target_month_ends`'s grid by (year, month) KEY, not exact
    date equality. Bug found during combined-deployable smoke-testing: S1's month-end dates are the
    LAST REAL DUKA TRADING DAY of each calendar month (e.g. 2018-07-31), while S2's monthly-marked
    dates are the FRIDAY at/before each month-end (e.g. 2018-07-27, from the weekly beta-estimation
    calendar) -- an exact-Timestamp dict lookup between the two silently drops ~55% of S2's monthly
    observations as unmatched (NaN), corrupting the equal-risk-contribution blend's trailing-vol
    estimate (near-zero vol from mostly-NaN stretches spikes the blend's inverse-vol scale up to its
    5x clip, then the 10% vol-target overlay piles a further ~3x on top of THAT, producing a
    -99% max-drawdown artifact in the combined book that does not reflect any real sleeve's own
    P&L). Fixed by keying on (year, month) so any date within a target month maps to that month's
    own grid slot, regardless of which exact calendar day each sleeve's own rebalance convention
    lands on."""
    target_ym = {(d.year, d.month): i for i, d in enumerate(target_month_ends)}
    out = np.full(len(target_month_ends), np.nan)
    for v, d in zip(values, dates):
        key = (d.year, d.month)
        if key in target_ym:
            out[target_ym[key]] = v
    return out


def align_to_monthly_grid(net: np.ndarray, dates: pd.DatetimeIndex, target_month_ends: pd.DatetimeIndex) -> np.ndarray:
    """Resample a (possibly non-monthly-native, e.g. S3's daily event-marked) net-RETURN series
    onto the SAME monthly grid as S1/S2 for blending: compound all marked observations falling
    within each (target_month_ends[k-1], target_month_ends[k]] window into one monthly return.
    Declared explicitly (E3_PREREG.md): this resampling step is NOT assumed lossless -- its own
    diagnostic (n_obs before/after) is printed by the caller.

    RETURNS ONLY -- see align_turnover_to_monthly_grid for turnover (a SUM, not a compounded
    product; conflating the two is a real bug found during combined-book smoke-testing, see that
    function's docstring)."""
    s = pd.Series(net, index=dates).sort_index()
    out = np.full(len(target_month_ends) - 1, np.nan)
    for k in range(len(target_month_ends) - 1):
        lo, hi = target_month_ends[k], target_month_ends[k + 1]
        seg = s[(s.index > lo) & (s.index <= hi)]
        if len(seg) == 0:
            continue
        out[k] = float(np.prod(1.0 + seg.to_numpy()) - 1.0)
    return out


def align_turnover_to_monthly_grid(turn: np.ndarray, dates: pd.DatetimeIndex,
                                    target_month_ends: pd.DatetimeIndex) -> np.ndarray:
    """Resample a daily (or otherwise sub-monthly) TURNOVER series onto the monthly grid by
    SUMMING all marked observations within each (month[k-1], month[k]] window -- turnover across N
    sub-periods within a month is the total gross weight change over the month, i.e. a SUM, never
    a compounded product.

    BUG FOUND AND FIXED (combined-book smoke-testing): S3's daily-marked turnover array was fed
    through `align_to_monthly_grid` (the RETURN resampler, `prod(1+x)-1`) by mistake. Compounding
    ~20 daily turnover readings multiplicatively (rather than summing them) inflated S3's monthly
    turnover contribution to as high as ~165 gross-weight-units/month (verified empirically), which
    at 13bp/unit produced a catastrophic, entirely artificial cost drag on the combined deployable
    (observed combined EV ~-365bp/month, maxDD ~98%) that had nothing to do with any real trading
    cost S3's own per-holding-period cells correctly priced in (S3's own standalone cells use
    `metrics(..., turn=marked["turn"])` directly on the native daily-marked series, which was never
    affected by this bug -- only the COMBINED book's re-resampled turnover was wrong)."""
    s = pd.Series(turn, index=dates).sort_index()
    out = np.full(len(target_month_ends) - 1, np.nan)
    for k in range(len(target_month_ends) - 1):
        lo, hi = target_month_ends[k], target_month_ends[k + 1]
        seg = s[(s.index > lo) & (s.index <= hi)]
        if len(seg) == 0:
            continue
        out[k] = float(np.nansum(seg.to_numpy()))
    return out


def equal_risk_contribution_blend(sleeve_nets: dict[str, np.ndarray], vol_win_months: int = 12,
                                   min_finite_in_window: int = 6) -> dict:
    """Vol-scale each sleeve's monthly net-return series so it contributes EQUAL RISK to the
    combined book (inverse trailing realized vol, using data strictly BEFORE month t -- landmine 3:
    exposure at month t must never see month t's own return), then sum. Returns the pre-overlay
    combined monthly net-return series and each sleeve's own realized scale array.

    BUG FOUND AND FIXED (during mandatory pre-run smoke-testing of the combined-book overlay, per
    landmine-3/4's instruction to re-derive rather than re-trust-by-analogy): a sleeve's WARMUP
    period (e.g. S2's W_beta=52-week rolling beta needs 52 weeks of history before it can size any
    position at all, so its book reports a real, structural `0.0` for its first ~12 months, not a
    genuinely-quiet-but-live trading result) was being fed straight into the trailing-vol window
    exactly like a real observation. A trailing-vol window dominated by these structural
    placeholder zeros reads as near-zero volatility, so `ref_vol / near_zero_sv` spikes the
    warmup-adjacent sleeve's inverse-vol SCALE to its clip ceiling right as the sleeve's FIRST real
    (and possibly large) live return lands -- verified empirically: S2's first live month
    (2019-06, -8.07% raw) got scaled to a -13.4% contribution alone, which the vol-target overlay
    then re-levered by a further ~3x into an isolated -41% monthly hit, an artifact of the warmup
    boundary, not any real sleeve's own risk profile. FIX: treat exact `0.0` entries as "not yet
    live" (NaN) for the PURPOSES OF THE TRAILING-VOL ESTIMATE ONLY (the raw `net` array used for
    the actual P&L sum is untouched -- a genuine 0.0% month, if one ever legitimately occurs, still
    contributes its true zero return to the combined book); this makes the rolling window's
    min_periods-style admission correctly require real non-placeholder history before it will
    produce a low-vol reading, exactly analogous to how S1/S2's own `elig`/`LISTED` point-in-time
    masks already prevent an unlisted name from contributing to their own vol estimates."""
    names = list(sleeve_nets.keys())
    n = len(next(iter(sleeve_nets.values())))
    scales = {}
    for name, net in sleeve_nets.items():
        net_for_vol = np.where(net == 0.0, np.nan, net)  # structural warmup zeros -> NaN for vol est.
        s = pd.Series(net_for_vol)
        trailing_vol = s.rolling(vol_win_months, min_periods=min_finite_in_window).std().shift(1)
        # target each sleeve to an equal per-sleeve vol budget (1/n_sleeves of a notional 10% ann
        # total is NOT needed here -- equal RISK CONTRIBUTION just requires each sleeve's OWN
        # trailing vol be normalized to the same reference level; use the cross-sleeve median
        # trailing vol at each month as the common reference so no sleeve's absolute scale is
        # arbitrary).
        scales[name] = trailing_vol.to_numpy()
    # common reference level per month = median of the (available) sleeves' trailing vols that month
    vol_matrix = np.stack([scales[nm] for nm in names], axis=0)  # (n_sleeves, n_months)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ref_vol = np.nanmedian(vol_matrix, axis=0)
    combined = np.zeros(n)
    contrib = {}
    for name in names:
        net = sleeve_nets[name]
        sv = scales[name]
        scale = np.where(np.isfinite(sv) & (sv > 0) & np.isfinite(ref_vol), ref_vol / np.maximum(sv, 1e-9), 1.0)
        scale = np.clip(scale, 0.2, 5.0)
        scaled_net = net * scale / len(names)
        contrib[name] = scaled_net
        combined = combined + np.nan_to_num(scaled_net, nan=0.0)
    return {"combined_pre_overlay": combined, "per_sleeve_scaled": contrib}


def vol_target_overlay_monthly(net: np.ndarray, periods_per_yr: float, window: int = 6,
                                max_leverage: float = 3.0) -> np.ndarray:
    """10% vol-target overlay, reusing overlay.py::vol_target_scale's EXACT convention: scale
    month t by target_vol_ann / (trailing vol of months [t-window, t), i.e. STRICTLY BEFORE t,
    half-open interval excluding t itself), capped at max_leverage, floor scale=1.0 for the first
    `window` months (no trailing estimate yet). This is bug-pattern-3-critical: the exposure value
    used for month t must be computable using ONLY information available before month t's own
    return is realized.

    DATE-ALIGNMENT TRACE (printed diagnostic, per landmine 3 mandate): net[t] is the combined
    book's return REALIZED during calendar month t (dated at month t's own month-end, per
    score_monthly_book's convention: net[t] uses RET_M[t], the return compounded over
    (month_end[t-1], month_end[t]]). scale[t] is computed from net[t-window:t] -- i.e. months
    STRICTLY BEFORE t (t-window through t-1 inclusive) -- so scale[t] could only ever be known
    using information available BEFORE month t's own return is realized. There is no index at
    which scale[t] depends on net[t] itself or any later month."""
    n = net.size
    scale = np.ones(n)
    for t in range(n):
        lo = t - window
        if lo < 1:
            continue
        w = net[lo:t]
        w = w[np.isfinite(w)]
        sd = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
        vol_ann = sd * np.sqrt(periods_per_yr)
        if vol_ann <= 0:
            continue
        scale[t] = min(max_leverage, SIGMA_STAR / vol_ann)
    return scale


# ======================================================================================
# regime sanity checks (own signature per sleeve, per E3_PREREG.md)
# ======================================================================================
def s1_regime_sanity(net: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """S1: does a visible, identifiable drawdown/crash episode show up around 2020-04 (COVID
    V-shaped snapback), consistent with the documented momentum-crash phenomenon?"""
    s = pd.Series(net, index=dates)
    window = s[(s.index >= "2020-01-01") & (s.index <= "2020-08-01")]
    eq = (1 + s.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (1 - eq / peak)
    dd_window = dd[(dd.index >= "2020-01-01") & (dd.index <= "2020-08-01")]
    return {
        "window_monthly_returns_2020_01_to_2020_08": {str(k.date()): round(float(v), 5) for k, v in window.items()},
        "max_drawdown_in_window_pct": round(float(dd_window.max()) * 100, 3) if len(dd_window) else None,
        "worst_single_month_2020": round(float(window.min()), 5) if len(window) else None,
        "worst_single_month_date": str(window.idxmin().date()) if len(window) and window.notna().any() else None,
    }


def s2_regime_sanity(net: np.ndarray, dates: pd.DatetimeIndex, broad_net: np.ndarray,
                      broad_dates: pd.DatetimeIndex, ex_post_beta: float) -> dict:
    s = pd.Series(net, index=dates)
    b = pd.Series(broad_net, index=broad_dates)
    out = {"ex_post_beta": round(ex_post_beta, 5) if np.isfinite(ex_post_beta) else None, "windows": {}}
    for name, (start, end) in STRESS_WINDOWS.items():
        s_w = s[(s.index >= start) & (s.index <= end)]
        b_w = b[(b.index >= start) & (b.index <= end)]
        s2_ret = float(np.nansum(s_w)) if len(s_w) else None
        broad_ret = float(np.nansum(b_w)) if len(b_w) else None
        resilient = None
        if s2_ret is not None and broad_ret is not None:
            resilient = bool(s2_ret >= broad_ret)
        out["windows"][name] = {
            "n_obs_s2": int(len(s_w)), "n_obs_broad": int(len(b_w)),
            "s2_cum_return_pct": round(s2_ret * 100, 3) if s2_ret is not None else None,
            "broad_cum_return_pct": round(broad_ret * 100, 3) if broad_ret is not None else None,
            "s2_resilient_vs_broad": resilient,
        }
    return out


def s3_regime_sanity(events: list[dict], RET: np.ndarray, cal: pd.DatetimeIndex, syms: list[str],
                      periods_per_yr_ens: float) -> dict:
    """(a) split by trailing dollar volume (top-half vs bottom-half at each event date), confirm
    drift concentrated in less-liquid half. (b) per-year Sharpes checked for pure decay-to-zero."""
    dvols = [e["trail_dvol"] for e in events if e["tercile"] in ("top", "bottom") and np.isfinite(e["trail_dvol"])]
    if not dvols:
        return {"error": "no events with finite trailing dollar volume"}
    median_dvol = float(np.median(dvols))
    top_events = [e for e in events if e["tercile"] in ("top", "bottom") and np.isfinite(e["trail_dvol"]) and e["trail_dvol"] >= median_dvol]
    bot_events = [e for e in events if e["tercile"] in ("top", "bottom") and np.isfinite(e["trail_dvol"]) and e["trail_dvol"] < median_dvol]

    def _half_book_stats(evs):
        if len(evs) < 10:
            return {"n_events": len(evs), "sharpe_ann": None, "ev_bp": None}
        port = s3_ensemble_book(evs, RET, cal, syms, canary=False)
        marked = s3_extract_marked_series(port["net"], port["turn"], port["dates"])
        if len(marked["net"]) < 30:
            return {"n_events": len(evs), "sharpe_ann": None, "ev_bp": None}
        ppy = derive_periods_per_yr(marked["dates"], f"s3_regime_half_n{len(evs)}", expect_const=None)
        mm = metrics(marked["net"], ppy, turn=marked["turn"])
        return {"n_events": len(evs), "sharpe_ann": mm.get("sharpe_ann"), "ev_bp": mm.get("ev_bp_obs"),
                "n_obs_marked": mm.get("n_obs")}

    top_stats = _half_book_stats(top_events)   # HIGH liquidity half
    bot_stats = _half_book_stats(bot_events)   # LOW liquidity half
    drift_concentrated_in_illiquid = None
    if top_stats["sharpe_ann"] is not None and bot_stats["sharpe_ann"] is not None:
        drift_concentrated_in_illiquid = bool(bot_stats["sharpe_ann"] > top_stats["sharpe_ann"])

    return {
        "median_trailing_dollar_volume": median_dvol,
        "high_liquidity_half": top_stats,
        "low_liquidity_half": bot_stats,
        "drift_concentrated_in_less_liquid_half": drift_concentrated_in_illiquid,
    }


def s3_per_year_sharpe_decay_check(net: np.ndarray, dates: pd.DatetimeIndex, periods_per_yr: float) -> dict:
    py = per_year_sharpe(net, dates, periods_per_yr)
    years_sorted = sorted(py.keys())
    if len(years_sorted) < 3:
        return {"per_year": py, "pure_decay_to_zero_pattern": None}
    early = years_sorted[:2]
    late = years_sorted[2:]
    early_sharpes = [py[y] for y in early]
    late_sharpes = [py[y] for y in late]
    pure_decay = bool(np.mean(early_sharpes) > 0 and all(s <= 0.1 for s in late_sharpes))
    return {"per_year": py, "early_years": early, "late_years": late,
            "pure_decay_to_zero_pattern": pure_decay}


# ======================================================================================
# main pipeline
# ======================================================================================
def main() -> None:
    t0 = time.time()
    print("=== E3 pipeline start ===", flush=True)

    universe = load_universe()
    universe_via_module = load_universe_via_module()
    drift_check = sorted(e["symbol"] for e in universe) == sorted(e["symbol"] for e in universe_via_module)
    print(f"universe: {len(universe)} equities (regex-load); module-load matches: {drift_check}", flush=True)
    if not drift_check:
        print("WARNING: regex-based load_universe() and convexity.panel.candidates.load_universe() "
              "DISAGREE on symbol set.", flush=True)

    with_dukascopy = {e["symbol"] for e in universe if e.get("dukascopy")}
    print(f"  {len(with_dukascopy)} with non-null dukascopy field (declared 102)", flush=True)

    excluded_all = EXCLUDED_PERP_ONLY | EXCLUDED_SHORT_HISTORY | EXCLUDED_LEVERAGED_ETF
    base_93_syms = sorted(with_dukascopy - excluded_all)
    print(f"  base universe after all exclusions: {len(base_93_syms)} (declared 93)", flush=True)
    assert len(base_93_syms) == 93, f"base universe count mismatch: got {len(base_93_syms)}, expected 93"

    on_demo_syms_live = {e["symbol"] for e in universe if e.get("on_demo")}
    print(f"  on_demo raw (live check): {sorted(on_demo_syms_live)} n={len(on_demo_syms_live)}", flush=True)
    on_demo_s1s2 = (on_demo_syms_live & set(base_93_syms))
    print(f"  on_demo-and-real-history-and-not-leveraged (S1/S2): {sorted(on_demo_s1s2)} n={len(on_demo_s1s2)}", flush=True)
    assert on_demo_s1s2 == ON_DEMO_S1S2_EXPECTED, (
        f"on_demo S1/S2 subset mismatch: got {sorted(on_demo_s1s2)}, expected {sorted(ON_DEMO_S1S2_EXPECTED)}")

    print("=== mandatory bug-pattern-1 check: verify_no_perp_rows_in_signal_panel ===", flush=True)
    sample_syms = ["PAYPUSDT", "AAPLUSDT", "SPYUSDT", "QQQUSDT", "TSLAUSDT"]
    seam_check = verify_no_perp_rows_in_signal_panel(sample_syms)
    print(json.dumps(seam_check, indent=1, default=str), flush=True)
    if not seam_check["paypusdt_seam_check"]["PASS"]:
        print("FATAL: seam check did not pass -- aborting before any headline number is computed.", flush=True)
        sys.exit(1)

    print("=== S1 sign-check assertion ===", flush=True)
    sign_check_momentum_assert()

    print("=== building per-name daily series (duka-only, base-93 universe) ===", flush=True)
    universe_93_dicts = [e for e in universe if e["symbol"] in set(base_93_syms)]
    name_series = build_name_series(universe_93_dicts, excluded=set())
    print(f"  {len(name_series)} of 93 base-universe names have usable duka-only history", flush=True)
    usable_93 = sorted(name_series.keys())

    print("=== building daily panel (union calendar, PIT listed mask, sectors) ===", flush=True)
    P = build_daily_panel(name_series)
    cal, syms, sectors = P["cal"], P["syms"], P["sectors"]
    RET, LISTED, DVOL = P["ret"], P["listed"], P["dvol"]
    n, m = RET.shape
    print(f"  panel: {n} days x {m} names, span {cal.min().date()} -> {cal.max().date()}", flush=True)

    print("=== sector residualization (recomputed fresh per day, <4 PIT members -> 'other') ===", flush=True)
    RESID = residualize_daily_fast(RET, LISTED, sectors)
    rng = np.random.default_rng(0)
    check_days = rng.choice(n, size=min(20, n), replace=False)
    RESID_ref_sub = residualize_daily_ref(RET[check_days], LISTED[check_days], sectors)
    RESID_fast_sub = RESID[check_days]
    resid_match = bool(np.allclose(RESID_ref_sub, RESID_fast_sub, equal_nan=True, atol=1e-10))
    print(f"  residualization reference-vs-fast equivalence check on {len(check_days)} sampled days: {resid_match}", flush=True)
    if not resid_match:
        print("FATAL: residualization implementations disagree -- aborting.", flush=True)
        sys.exit(1)

    results: dict = {"meta": {
        "n_trials_ledger": N_TRIALS,
        "base_universe_n_93": len(base_93_syms),
        "usable_93_with_history": len(usable_93),
        "on_demo_s1s2_n_16": sorted(on_demo_s1s2),
        "candidates_md_drift_check_passed": drift_check,
        "seam_check": seam_check,
        "residualization_equivalence_check_passed": resid_match,
        "panel_span": {"start": str(cal.min().date()), "end": str(cal.max().date()), "n_days": int(n), "n_syms": int(m)},
        "cost_model": {"primary_rt_bp": RT_PRIMARY_BP, "rt_scenarios_bp": list(RT_SCEN),
                       "funding_scenarios_bp_per_period": list(FUND_SCEN_BP_PER_WK)},
        "survivorship": {
            "S1": "MOST SEVERE: 93-name roster (16 on_demo) is today's Binance winners; momentum "
                  "mechanically enriches for past winners, which can inflate a momentum backtest's "
                  "apparent Sharpe more than almost any other cross-sectional anomaly.",
            "S2": "MODERATE, explicitly NOT nil: 93-name roster (16 on_demo) is still today's "
                  "popularity-selected list; beta-sorting does not mechanically enrich for past "
                  "winners the way momentum does, but names that would rank in extreme beta deciles "
                  "and later delisted are absent entirely.",
            "S3": "MODERATE, doubly caveated: same 93/76-name (11 on_demo) base-roster survivorship "
                  "as S2, PLUS a second layer -- the announcement-DATE history itself is collected "
                  "only for today's 76-name roster.",
        },
    }}

    # ==================================================================================
    # S1: slow sector-residual momentum
    # ==================================================================================
    print("=== S1: building monthly panel ===", flush=True)
    me_idx = month_end_indices(cal)
    RET_M = month_end_returns_matrix(RET, cal, me_idx)
    RESID_M = month_end_returns_matrix(RESID, cal, me_idx)
    month_end_dates = cal[me_idx[1:]]
    LISTED_M = LISTED[me_idx[1:]]
    n_m = RET_M.shape[0]
    print(f"  S1 monthly panel: {n_m} months, {month_end_dates.min().date()} -> {month_end_dates.max().date()}", flush=True)

    s1_books = {}
    for L in S1_LOOKBACKS_L:
        s1_books[L] = s1_build_book(RESID_M, RET_M, LISTED_M, L)
    We_s1 = s1_ensemble_weights(s1_books)

    s1_cells = {}
    s1_nets_primary = {}
    for L in S1_LOOKBACKS_L:
        sc = score_monthly_book(s1_books[L], RET_M, month_end_dates)
        valid = np.isfinite(sc["net"])
        net_v, turn_v, dates_v = sc["net"][valid], sc["turn"][valid], sc["dates"][valid]
        ppy = derive_periods_per_yr(dates_v, f"S1_L{L}", expect_const=MONTH_PER_YR)
        cell = {}
        for rt in RT_SCEN:
            net = apply_costs(net_v, turn_v, rt)
            mm = metrics(net, ppy, turn=turn_v)
            mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(net_v) * np.sqrt(ppy)), 4)
            if rt == RT_PRIMARY_BP:
                if len(net) > 40:
                    paths = cpcv_paths(net, 8, 2, 2, ppy)
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, ppy)
                mm["per_year"] = per_year_sharpe(net, dates_v, ppy)
                s1_nets_primary[f"L{L}"] = {"net": net, "dates": dates_v, "turn": turn_v, "ppy": ppy}
            cell[f"rt{rt}"] = mm
        s1_cells[f"L{L}"] = cell

    sc_ens = score_monthly_book(We_s1, RET_M, month_end_dates)
    valid = np.isfinite(sc_ens["net"])
    net_v, turn_v, dates_v = sc_ens["net"][valid], sc_ens["turn"][valid], sc_ens["dates"][valid]
    ppy_s1_ens = derive_periods_per_yr(dates_v, "S1_ensemble", expect_const=MONTH_PER_YR)
    s1_ens_cell = {}
    s1_ens_net_primary = None
    for rt in RT_SCEN:
        net = apply_costs(net_v, turn_v, rt)
        mm = metrics(net, ppy_s1_ens, turn=turn_v)
        mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(net_v) * np.sqrt(ppy_s1_ens)), 4)
        if rt == RT_PRIMARY_BP:
            if len(net) > 40:
                paths = cpcv_paths(net, 8, 2, 2, ppy_s1_ens)
                if paths:
                    mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                    mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
            mm["dsr"] = dsr(net, N_TRIALS, ppy_s1_ens)
            mm["per_year"] = per_year_sharpe(net, dates_v, ppy_s1_ens)
            s1_ens_net_primary = {"net": net, "dates": dates_v, "turn": turn_v, "ppy": ppy_s1_ens}
        s1_ens_cell[f"rt{rt}"] = mm

    # S1 on_demo-16 subset
    demo_mask_s1 = np.array([s in ON_DEMO_S1S2_EXPECTED for s in syms])
    We_s1_demo = We_s1.copy()
    We_s1_demo[:, ~demo_mask_s1] = 0.0
    gross_demo = np.abs(We_s1_demo).sum(axis=1, keepdims=True)
    We_s1_demo = np.divide(We_s1_demo, gross_demo, out=np.zeros_like(We_s1_demo), where=gross_demo > 0)
    sc_demo = score_monthly_book(We_s1_demo, RET_M, month_end_dates)
    valid_d = np.isfinite(sc_demo["net"])
    net_d, turn_d, dates_d = sc_demo["net"][valid_d], sc_demo["turn"][valid_d], sc_demo["dates"][valid_d]
    ppy_demo = derive_periods_per_yr(dates_d, "S1_ensemble_on_demo16", expect_const=MONTH_PER_YR)
    net_demo_costed = apply_costs(net_d, turn_d, RT_PRIMARY_BP)
    s1_demo_headline = metrics(net_demo_costed, ppy_demo, turn=turn_d)
    s1_demo_headline["dsr"] = dsr(net_demo_costed, N_TRIALS, ppy_demo)

    # S1 lookahead canary
    print("=== S1 lookahead canary ===", flush=True)
    s1_canary_books = {L: s1_build_book(RESID_M, RET_M, LISTED_M, L, canary=True) for L in S1_LOOKBACKS_L}
    We_s1_canary = s1_ensemble_weights(s1_canary_books)
    sc_can = score_monthly_book(We_s1_canary, RET_M, month_end_dates)
    valid_c = np.isfinite(sc_can["net"])
    net_c, turn_c, dates_c = sc_can["net"][valid_c], sc_can["turn"][valid_c], sc_can["dates"][valid_c]
    ppy_c = derive_periods_per_yr(dates_c, "S1_canary", expect_const=MONTH_PER_YR)
    net_c_costed = apply_costs(net_c, turn_c, RT_PRIMARY_BP)
    s1_canary_mm = metrics(net_c_costed, ppy_c, turn=turn_c)
    s1_canary_mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(net_c) * np.sqrt(ppy_c)), 4)
    canary_dominates_s1 = bool(s1_canary_mm["sharpe_ann"] > s1_ens_cell[f"rt{RT_PRIMARY_BP}"]["sharpe_ann"])
    print(f"  S1 canary sharpe_ann={s1_canary_mm['sharpe_ann']} vs honest ensemble "
          f"{s1_ens_cell[f'rt{RT_PRIMARY_BP}']['sharpe_ann']} -- dominates: {canary_dominates_s1}", flush=True)

    print("=== S1 regime sanity (2020-04 momentum-crash check) ===", flush=True)
    s1_regime = s1_regime_sanity(s1_ens_net_primary["net"], s1_ens_net_primary["dates"])
    print(json.dumps(s1_regime, indent=1, default=str), flush=True)

    results["S1"] = {
        "per_lookback": s1_cells, "ensemble": s1_ens_cell,
        "on_demo_16_headline": s1_demo_headline,
        "lookahead_canary": {"metrics": s1_canary_mm, "dominates_honest": canary_dominates_s1},
        "regime_sanity_2020_04": s1_regime,
        "sign_check": "PASS (asserted at pipeline start): LONG top quintile (winners), SHORT bottom quintile (losers)",
    }
    print(f"  S1 ensemble sharpe_ann (full-93, rt13): {s1_ens_cell[f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')}", flush=True)
    print(f"  S1 ensemble sharpe_ann (on_demo-16, rt13): {s1_demo_headline.get('sharpe_ann')}", flush=True)

    # ==================================================================================
    # S2: betting-against-beta
    # ==================================================================================
    print("=== S2: building weekly panel + market proxy + rolling beta ===", flush=True)
    WK = build_weekly_returns(name_series, syms)
    fridays, LOGRET_F, LISTED_F = WK["fridays"], WK["logret"], WK["listed"]
    mkt_weekly = build_market_proxy_weekly(LOGRET_F, LISTED_F)
    print(f"  weekly panel: {len(fridays)} Fridays, {fridays.min().date()} -> {fridays.max().date()}", flush=True)

    # month-end Friday indices: the Friday at/before each calendar month-end, matching S1/S2's
    # shared "last trading day of month" rebalance convention as closely as a weekly grid allows
    month_ends_cal = cal[month_end_indices(cal)]
    month_end_fridays_idx = []
    for me in month_ends_cal:
        pos = fridays.searchsorted(me, side="right") - 1
        if pos >= 0:
            month_end_fridays_idx.append(pos)
    month_end_fridays_idx = sorted(set(month_end_fridays_idx))
    month_end_fridays_idx_arr = np.array(month_end_fridays_idx)
    print(f"  {len(month_end_fridays_idx)} month-end-Friday rebalance points", flush=True)

    beta_cells = {}
    beta_books = {}
    for w_beta in S2_BETA_WINDOWS:
        print(f"  computing rolling beta W_beta={w_beta}...", flush=True)
        BETA_F = rolling_beta_weekly(LOGRET_F, mkt_weekly, w_beta)
        book = s2_build_book(BETA_F, LOGRET_F, LISTED_F, fridays, month_end_fridays_idx_arr,
                              vol_win_weeks=S2_VOL_WIN_MONTHS * 4 + 4)
        beta_books[w_beta] = book
        sc = score_weekly_book_to_monthly(book["W"], LOGRET_F, fridays, month_end_fridays_idx_arr)
        valid = np.isfinite(sc["net"])
        net_v, turn_v, dates_v = sc["net"][valid], sc["turn"][valid], sc["dates"][valid]
        ppy = derive_periods_per_yr(dates_v, f"S2_Wbeta{w_beta}", expect_const=MONTH_PER_YR)

        # ex-ante beta trace at month-end-Friday rebalances (exposure-timing diagnostic)
        ex_ante_at_me = book["ex_ante_beta"][month_end_fridays_idx_arr]
        ex_ante_at_me_finite = ex_ante_at_me[np.isfinite(ex_ante_at_me)]
        mean_ex_ante_beta = float(np.mean(ex_ante_at_me_finite)) if len(ex_ante_at_me_finite) else None

        # market proxy compounded monthly for ex-post beta regression
        mkt_month = np.full(len(month_end_fridays_idx) - 1, np.nan)
        for k in range(len(month_end_fridays_idx) - 1):
            f0, f1 = month_end_fridays_idx_arr[k], month_end_fridays_idx_arr[k + 1]
            seg = np.expm1(mkt_weekly[f0 + 1:f1 + 1])
            seg_f = np.nan_to_num(seg, nan=0.0)
            mkt_month[k] = np.prod(1.0 + seg_f) - 1.0
        mkt_month_v = mkt_month[valid]

        cell = {}
        for rt in RT_SCEN:
            net = apply_costs(net_v, turn_v, rt)
            mm = metrics(net, ppy, turn=turn_v)
            mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(net_v) * np.sqrt(ppy)), 4)
            if rt == RT_PRIMARY_BP:
                if len(net) > 40:
                    paths = cpcv_paths(net, 8, 2, 2, ppy)
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, ppy)
                mm["per_year"] = per_year_sharpe(net, dates_v, ppy)
                mm["ex_post_beta_full_sample"] = round(realized_ex_post_beta(net_v, mkt_month_v), 5)
                mm["mean_ex_ante_beta_at_rebalance"] = round(mean_ex_ante_beta, 5) if mean_ex_ante_beta is not None else None
            cell[f"rt{rt}"] = mm
        beta_cells[f"Wbeta{w_beta}"] = cell
        if w_beta == S2_PRIMARY_W_BETA:
            s2_primary_net = {"net": apply_costs(net_v, turn_v, RT_PRIMARY_BP), "dates": dates_v,
                               "turn": turn_v, "ppy": ppy, "mkt_month": mkt_month_v,
                               "net_gross": net_v}
            s2_primary_ex_post_beta = realized_ex_post_beta(net_v, mkt_month_v)

    print(f"  S2 primary (W_beta=52) sharpe_ann (rt13): {beta_cells[f'Wbeta{S2_PRIMARY_W_BETA}'][f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')}", flush=True)
    print(f"  S2 primary ex-post beta: {s2_primary_ex_post_beta}", flush=True)

    # S2 on_demo-16 subset (primary W_beta=52)
    demo_mask_s2 = np.array([s in ON_DEMO_S1S2_EXPECTED for s in syms])
    W52_demo = beta_books[S2_PRIMARY_W_BETA]["W"].copy()
    W52_demo[:, ~demo_mask_s2] = 0.0
    gross_demo2 = np.abs(W52_demo).sum(axis=1, keepdims=True)
    W52_demo = np.divide(W52_demo, gross_demo2, out=np.zeros_like(W52_demo), where=gross_demo2 > 0)
    sc_demo2 = score_weekly_book_to_monthly(W52_demo, LOGRET_F, fridays, month_end_fridays_idx_arr)
    valid_d2 = np.isfinite(sc_demo2["net"])
    net_d2, turn_d2, dates_d2 = sc_demo2["net"][valid_d2], sc_demo2["turn"][valid_d2], sc_demo2["dates"][valid_d2]
    ppy_d2 = derive_periods_per_yr(dates_d2, "S2_Wbeta52_on_demo16", expect_const=MONTH_PER_YR)
    net_d2_costed = apply_costs(net_d2, turn_d2, RT_PRIMARY_BP)
    s2_demo_headline = metrics(net_d2_costed, ppy_d2, turn=turn_d2)
    s2_demo_headline["dsr"] = dsr(net_d2_costed, N_TRIALS, ppy_d2)

    # S2 lookahead canary (primary window only, per shared convention -- one canary per sleeve)
    print("=== S2 lookahead canary ===", flush=True)
    BETA_52 = rolling_beta_weekly(LOGRET_F, mkt_weekly, S2_PRIMARY_W_BETA)
    canary_book = s2_build_book(BETA_52, LOGRET_F, LISTED_F, fridays, month_end_fridays_idx_arr,
                                 vol_win_weeks=S2_VOL_WIN_MONTHS * 4 + 4, canary=True)
    sc_can2 = score_weekly_book_to_monthly(canary_book["W"], LOGRET_F, fridays, month_end_fridays_idx_arr)
    valid_c2 = np.isfinite(sc_can2["net"])
    net_c2, turn_c2, dates_c2 = sc_can2["net"][valid_c2], sc_can2["turn"][valid_c2], sc_can2["dates"][valid_c2]
    ppy_c2 = derive_periods_per_yr(dates_c2, "S2_canary", expect_const=MONTH_PER_YR)
    net_c2_costed = apply_costs(net_c2, turn_c2, RT_PRIMARY_BP)
    s2_canary_mm = metrics(net_c2_costed, ppy_c2, turn=turn_c2)
    s2_canary_mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(net_c2) * np.sqrt(ppy_c2)), 4)
    canary_dominates_s2 = bool(s2_canary_mm["sharpe_ann"] > beta_cells[f"Wbeta{S2_PRIMARY_W_BETA}"][f"rt{RT_PRIMARY_BP}"]["sharpe_ann"])
    print(f"  S2 canary sharpe_ann={s2_canary_mm['sharpe_ann']} vs honest primary "
          f"{beta_cells[f'Wbeta{S2_PRIMARY_W_BETA}'][f'rt{RT_PRIMARY_BP}']['sharpe_ann']} -- dominates: {canary_dominates_s2}", flush=True)

    # broad equal-weight book comparator (for S2 regime-sanity), monthly, unhedged, unit-gross
    print("=== S2 regime sanity ===", flush=True)
    broad_w = np.where(LISTED, 1.0, 0.0)
    broad_gross = broad_w.sum(axis=1, keepdims=True)
    broad_w = np.divide(broad_w, broad_gross, out=np.zeros_like(broad_w), where=broad_gross > 0)
    broad_ret_daily = np.nansum(broad_w * np.nan_to_num(RET, nan=0.0), axis=1)
    broad_s = pd.Series(broad_ret_daily, index=cal)
    broad_monthly = np.full(n_m, np.nan)
    for k in range(n_m):
        d0, d1 = me_idx[k], me_idx[k + 1]
        seg = broad_ret_daily[d0 + 1:d1 + 1]
        broad_monthly[k] = np.prod(1.0 + seg) - 1.0 if len(seg) else np.nan
    broad_dates_full = cal[me_idx[1:]]

    s2_regime = s2_regime_sanity(s2_primary_net["net"], s2_primary_net["dates"],
                                  broad_monthly, broad_dates_full, s2_primary_ex_post_beta)
    print(json.dumps(s2_regime, indent=1, default=str), flush=True)

    results["S2"] = {
        "per_beta_window": beta_cells,
        "primary_window": S2_PRIMARY_W_BETA,
        "on_demo_16_headline": s2_demo_headline,
        "lookahead_canary": {"metrics": s2_canary_mm, "dominates_honest": canary_dominates_s2},
        "regime_sanity": s2_regime,
        "market_proxy": "equal-weight of the 93-name base universe's own weekly log-returns, point-in-time",
        "exposure_timing_trace": {
            "beta_estimation": "beta[t] uses LOGRET_F[t-w_beta:t] -- weeks STRICTLY BEFORE week t "
                                "(half-open, excludes t itself) -- so the beta used to size a "
                                "rebalance at week t never includes week t's own return.",
            "rebalance_convention": "positions set at the month-end Friday, held constant through "
                                     "the following weeks until the next month-end Friday rebalance.",
        },
    }
    print(f"  S2 on_demo-16 sharpe_ann (rt13): {s2_demo_headline.get('sharpe_ann')}", flush=True)

    # ==================================================================================
    # S3: price-based PEAD
    # ==================================================================================
    print("=== S3: re-deriving PEAD-usable universe fresh from data/earnings/*.csv ===", flush=True)
    earnings_usable, coverage_detail = s3_usable_universe(universe_93_dicts, set(base_93_syms))
    print(f"  S3 usable: {len(earnings_usable)}/{len(base_93_syms)} = {coverage_detail['coverage_fraction']:.4f} "
          f"(declared 76/93=0.8172)", flush=True)
    assert coverage_detail["coverage_fraction"] >= S3_COVERAGE_FLOOR, "S3 coverage floor check failed at runtime"

    earnings_usable_in_panel = earnings_usable & set(syms)
    print(f"  {len(earnings_usable_in_panel)} of S3-usable names present in the built daily panel", flush=True)

    on_demo_s3 = on_demo_s1s2 & earnings_usable_in_panel
    print(f"  on_demo-and-PEAD-usable (S3): {sorted(on_demo_s3)} n={len(on_demo_s3)}", flush=True)
    assert on_demo_s3 == ON_DEMO_S3_EXPECTED, (
        f"on_demo S3 subset mismatch: got {sorted(on_demo_s3)}, expected {sorted(ON_DEMO_S3_EXPECTED)}")

    print("=== S3: building announcement events ===", flush=True)
    events = s3_build_events(name_series, earnings_usable_in_panel, cal, RET, LISTED, sectors, syms, DVOL, RESID)
    print(f"  {len(events)} raw events constructed across {len(earnings_usable_in_panel)} names", flush=True)
    events = s3_assign_terciles(events)
    n_top = sum(1 for e in events if e["tercile"] == "top")
    n_bot = sum(1 for e in events if e["tercile"] == "bottom")
    n_mid = sum(1 for e in events if e["tercile"] == "mid")
    n_none = sum(1 for e in events if e["tercile"] is None)
    print(f"  tercile assignment: top={n_top} bottom={n_bot} mid={n_mid} unassigned(insuff.peers)={n_none}", flush=True)

    s3_cells = {}
    s3_nets_by_hp = {}
    for hw in S3_HOLD_WEEKS:
        port = s3_build_event_portfolio(events, RET, cal, syms, hw, canary=False)
        marked = s3_extract_marked_series(port["net"], port["turn"], cal, port["net_exposure"])
        ppy = derive_periods_per_yr(marked["dates"], f"S3_hold{hw}wk", expect_const=None)
        cell = {}
        for rt in RT_SCEN:
            net = apply_costs(marked["net"], marked["turn"], rt)
            mm = metrics(net, ppy, turn=marked["turn"])
            mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(marked["net"]) * np.sqrt(ppy)), 4)
            mm["realized_avg_net_exposure"] = round(float(np.nanmean(marked["exposure"])), 4) if marked["exposure"] is not None else None
            if rt == RT_PRIMARY_BP:
                if len(net) > 40:
                    paths = cpcv_paths(net, 8, 2, max(2, int(round(ppy / 6))), ppy)
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, ppy)
                mm["per_year"] = per_year_sharpe(net, marked["dates"], ppy)
                s3_nets_by_hp[hw] = {"net": net, "dates": marked["dates"], "turn": marked["turn"], "ppy": ppy}
            cell[f"rt{rt}"] = mm
        s3_cells[f"hold{hw}wk"] = cell
        print(f"  S3 hold={hw}wk n_obs={mm.get('n_obs')} derived_ppy={ppy:.3f} sharpe_ann(rt13)={s3_cells[f'hold{hw}wk'][f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')}", flush=True)

    print("=== S3 ensemble (equal-weight band across 2/4/6-week holds) ===", flush=True)
    ens_port = s3_ensemble_book(events, RET, cal, syms, canary=False)
    ens_marked = s3_extract_marked_series(ens_port["net"], ens_port["turn"], cal, ens_port["net_exposure"])
    ppy_s3_ens = derive_periods_per_yr(ens_marked["dates"], "S3_ensemble", expect_const=None)
    s3_ens_cell = {}
    s3_ens_net_primary = None
    for rt in RT_SCEN:
        net = apply_costs(ens_marked["net"], ens_marked["turn"], rt)
        mm = metrics(net, ppy_s3_ens, turn=ens_marked["turn"])
        mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(ens_marked["net"]) * np.sqrt(ppy_s3_ens)), 4)
        mm["realized_avg_net_exposure"] = round(float(np.nanmean(ens_marked["exposure"])), 4) if ens_marked["exposure"] is not None else None
        expected_turnover_per_name_per_yr = 8.0  # ~4 events/name/yr x 2 trades
        mm["expected_turn_events_per_name_per_yr"] = expected_turnover_per_name_per_yr
        if rt == RT_PRIMARY_BP:
            if len(net) > 40:
                paths = cpcv_paths(net, 8, 2, max(2, int(round(ppy_s3_ens / 6))), ppy_s3_ens)
                if paths:
                    mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                    mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
            mm["dsr"] = dsr(net, N_TRIALS, ppy_s3_ens)
            mm["per_year"] = per_year_sharpe(net, ens_marked["dates"], ppy_s3_ens)
            s3_ens_net_primary = {"net": net, "dates": ens_marked["dates"], "turn": ens_marked["turn"], "ppy": ppy_s3_ens}
        s3_ens_cell[f"rt{rt}"] = mm
    print(f"  S3 ensemble sharpe_ann (rt13): {s3_ens_cell[f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')}, "
          f"turn_mean={s3_ens_cell[f'rt{RT_PRIMARY_BP}'].get('turn_mean')}", flush=True)

    # S3 on_demo-11 subset (ensemble)
    events_demo = [e for e in events if e["symbol"] in ON_DEMO_S3_EXPECTED]
    print(f"  S3 on_demo-11 subset: {len(events_demo)} events", flush=True)
    ens_port_demo = s3_ensemble_book(events_demo, RET, cal, syms, canary=False)
    ens_marked_demo = s3_extract_marked_series(ens_port_demo["net"], ens_port_demo["turn"], cal, ens_port_demo["net_exposure"])
    if len(ens_marked_demo["net"]) >= 30:
        ppy_s3_demo = derive_periods_per_yr(ens_marked_demo["dates"], "S3_ensemble_on_demo11", expect_const=None)
        net_demo3_costed = apply_costs(ens_marked_demo["net"], ens_marked_demo["turn"], RT_PRIMARY_BP)
        s3_demo_headline = metrics(net_demo3_costed, ppy_s3_demo, turn=ens_marked_demo["turn"])
        s3_demo_headline["dsr"] = dsr(net_demo3_costed, N_TRIALS, ppy_s3_demo)
    else:
        s3_demo_headline = {"n_obs": len(ens_marked_demo["net"]), "note": "insufficient marked observations (<30) for on_demo-11 subset metrics"}

    # S3 lookahead canary: entry backdated INTO the announcement window itself
    print("=== S3 lookahead canary (entry backdated into announcement window) ===", flush=True)
    events_canary = []
    for e in events:
        ec = dict(e)
        ec["entry_idx"] = e["t_idx"]  # peek: enter AT the announcement day itself (into the window)
        events_canary.append(ec)
    can_port = s3_build_event_portfolio(events_canary, RET, cal, syms, hold_weeks=4, canary=True)
    can_marked = s3_extract_marked_series(can_port["net"], can_port["turn"], cal, can_port["net_exposure"])
    if len(can_marked["net"]) >= 30:
        ppy_s3_can = derive_periods_per_yr(can_marked["dates"], "S3_canary", expect_const=None)
        net_can_costed = apply_costs(can_marked["net"], can_marked["turn"], RT_PRIMARY_BP)
        s3_canary_mm = metrics(net_can_costed, ppy_s3_can, turn=can_marked["turn"])
        s3_canary_mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(can_marked["net"]) * np.sqrt(ppy_s3_can)), 4)
        canary_dominates_s3 = bool(s3_canary_mm["sharpe_ann"] > s3_ens_cell[f"rt{RT_PRIMARY_BP}"]["sharpe_ann"])
    else:
        s3_canary_mm = {"n_obs": len(can_marked["net"]), "note": "insufficient marked observations for canary metrics"}
        canary_dominates_s3 = None
    print(f"  S3 canary sharpe_ann={s3_canary_mm.get('sharpe_ann')} vs honest ensemble "
          f"{s3_ens_cell[f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')} -- dominates: {canary_dominates_s3}", flush=True)

    print("=== S3 regime sanity (liquidity split + per-year decay) ===", flush=True)
    s3_liquidity = s3_regime_sanity(events, RET, cal, syms, ppy_s3_ens)
    s3_decay = s3_per_year_sharpe_decay_check(s3_ens_net_primary["net"], s3_ens_net_primary["dates"], ppy_s3_ens)
    print(json.dumps(s3_liquidity, indent=1, default=str), flush=True)
    print(json.dumps(s3_decay, indent=1, default=str), flush=True)

    results["S3"] = {
        "coverage_detail": {"coverage_fraction": coverage_detail["coverage_fraction"],
                             "n_usable": coverage_detail["n_usable"], "n_base": coverage_detail["n_base"]},
        "n_events_total": len(events), "n_events_top": n_top, "n_events_bottom": n_bot,
        "n_events_mid": n_mid, "n_events_unassigned": n_none,
        "per_holding_period": s3_cells, "ensemble": s3_ens_cell,
        "on_demo_11_headline": s3_demo_headline,
        "lookahead_canary": {"metrics": s3_canary_mm, "dominates_honest": canary_dominates_s3},
        "regime_sanity_liquidity_split": s3_liquidity,
        "regime_sanity_per_year_decay": s3_decay,
    }
    print(f"  S3 ensemble sharpe_ann (full-76, rt13): {s3_ens_cell[f'rt{RT_PRIMARY_BP}'].get('sharpe_ann')}", flush=True)
    print(f"  S3 on_demo-11 sharpe_ann (rt13): {s3_demo_headline.get('sharpe_ann')}", flush=True)

    # ==================================================================================
    # combined deployable: equal-risk-contribution blend + 10% vol-target overlay
    # ==================================================================================
    print("=== combined deployable ===", flush=True)
    # align all three sleeves onto the SAME monthly grid (S1's month_end_dates as the reference)
    target_month_ends = pd.DatetimeIndex([month_end_dates[0] - pd.Timedelta(days=31)]).append(month_end_dates)

    # NOTE: alignment is by (year, month) KEY, not exact Timestamp equality -- S1's month-end dates
    # are the last real duka trading day of each month, while S2's monthly-marked dates are the
    # Friday at/before each month-end (a different exact calendar day); see
    # align_series_by_year_month's docstring for the bug this fixes.
    s1_net_full = align_series_by_year_month(s1_ens_net_primary["net"], s1_ens_net_primary["dates"], month_end_dates)
    s2_net_full = align_series_by_year_month(s2_primary_net["net"], s2_primary_net["dates"], month_end_dates)
    s3_net_resampled = align_to_monthly_grid(s3_ens_net_primary["net"], s3_ens_net_primary["dates"], target_month_ends)
    print(f"  S3 resampling diagnostic: n_obs_native={len(s3_ens_net_primary['net'])} (native ppy="
          f"{ppy_s3_ens:.3f}) -> n_obs_monthly_grid={n_m} (non-nan months: "
          f"{int(np.isfinite(s3_net_resampled).sum())})", flush=True)

    sleeve_nets = {"S1": s1_net_full, "S2": s2_net_full, "S3": s3_net_resampled}
    blend = equal_risk_contribution_blend(sleeve_nets)
    combined_pre_overlay = blend["combined_pre_overlay"]

    ppy_combined = derive_periods_per_yr(month_end_dates, "combined_deployable", expect_const=MONTH_PER_YR)

    # EXPOSURE-TIMING TRACE (printed diagnostic, landmine 3 mandatory):
    print("  [exposure-timing trace] combined book: combined_pre_overlay[t] is the return realized "
          "during calendar month t (dated at month_end_dates[t]). vol_target_overlay_monthly's "
          "scale[t] is computed from combined_pre_overlay[t-window:t] -- STRICTLY the 6 months "
          "BEFORE t, excluding t itself (half-open interval, matches overlay.py::vol_target_scale's "
          "net[lo:t] convention). scale[t] therefore never depends on month t's own return.", flush=True)
    overlay_scale = vol_target_overlay_monthly(combined_pre_overlay, ppy_combined, window=6)
    combined_net_gross_exposure_scaled = combined_pre_overlay * overlay_scale

    # turnover proxy for the combined book: average of the three sleeves' own turnover series (aligned to
    # the combined monthly grid), scaled by the overlay's exposure multiplier (landmine 4: cost
    # components must scale with realized exposure)
    s1_turn_full = align_series_by_year_month(s1_ens_net_primary["turn"], s1_ens_net_primary["dates"], month_end_dates)
    s2_turn_full = align_series_by_year_month(s2_primary_net["turn"], s2_primary_net["dates"], month_end_dates)
    # S3's turnover is native-DAILY-marked -- must be SUMMED (not compounded like a return) onto
    # the monthly grid; see align_turnover_to_monthly_grid's docstring for the bug this fixes.
    s3_turn_resampled = align_turnover_to_monthly_grid(s3_ens_net_primary["turn"], s3_ens_net_primary["dates"], target_month_ends)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        combined_turn_raw = np.nanmean(np.stack([s1_turn_full, s2_turn_full, s3_turn_resampled]), axis=0)
    combined_turn_scaled = combined_turn_raw * overlay_scale  # landmine 4: fee term scales with es

    print("=== cost/funding scenario grid for combined deployable (per landmine 4) ===", flush=True)
    combined_cost_funding = {}
    for rt in RT_SCEN:
        for fbp in FUND_SCEN_BP_PER_WK:
            net = apply_costs(combined_net_gross_exposure_scaled, combined_turn_scaled, rt,
                               funding_bp_per_period=fbp, exposure_scale=overlay_scale)
            valid = np.isfinite(net)
            mm = metrics(net[valid], ppy_combined, turn=combined_turn_scaled[valid])
            key = f"rt{rt}_fund{fbp}"
            combined_cost_funding[key] = mm

    combined_net_primary = apply_costs(combined_net_gross_exposure_scaled, combined_turn_scaled,
                                        RT_PRIMARY_BP, funding_bp_per_period=0.0, exposure_scale=overlay_scale)
    valid_comb = np.isfinite(combined_net_primary)
    combined_net_v = combined_net_primary[valid_comb]
    combined_dates_v = month_end_dates[valid_comb]
    combined_turn_v = combined_turn_scaled[valid_comb]
    combined_headline_full = metrics(combined_net_v, ppy_combined, turn=combined_turn_v)
    combined_headline_full["dsr"] = dsr(combined_net_v, N_TRIALS, ppy_combined)
    combined_headline_full["per_year"] = per_year_sharpe(combined_net_v, combined_dates_v, ppy_combined)
    if len(combined_net_v) > 40:
        paths = cpcv_paths(combined_net_v, 8, 2, 2, ppy_combined)
        if paths:
            combined_headline_full["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
            combined_headline_full["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)

    # combined on_demo subset: blend the three sleeves' own on_demo headline net series (best
    # available proxy for a "16/11-name-only combined book" -- rebuilding the full risk-parity
    # blend machinery on demo-only weight matrices for S1/S2 and demo-only events for S3)
    s1_demo_net_full = align_series_by_year_month(net_demo_costed, dates_d, month_end_dates)
    s2_demo_net_full = align_series_by_year_month(net_d2_costed, dates_d2, month_end_dates)
    s3_demo_net_resampled = align_to_monthly_grid(ens_marked_demo["net"], ens_marked_demo["dates"], target_month_ends) if len(ens_marked_demo["net"]) > 0 else np.full(n_m, np.nan)

    sleeve_nets_demo = {"S1": s1_demo_net_full, "S2": s2_demo_net_full, "S3": s3_demo_net_resampled}
    blend_demo = equal_risk_contribution_blend(sleeve_nets_demo)
    combined_pre_overlay_demo = blend_demo["combined_pre_overlay"]
    overlay_scale_demo = vol_target_overlay_monthly(combined_pre_overlay_demo, ppy_combined, window=6)
    combined_demo_scaled = combined_pre_overlay_demo * overlay_scale_demo
    valid_comb_demo = np.isfinite(combined_demo_scaled)
    combined_headline_demo = metrics(combined_demo_scaled[valid_comb_demo], ppy_combined)
    combined_headline_demo["dsr"] = dsr(combined_demo_scaled[valid_comb_demo], N_TRIALS, ppy_combined)

    results["combined_deployable"] = {
        "construction": "equal-risk-contribution blend of S1 ensemble + S2 W_beta=52 primary + S3 "
                         "ensemble (S3 resampled onto the S1/S2 monthly grid), single 10% vol-target "
                         "overlay on top, primary cost 13bp, 0bp/wk funding primary scenario.",
        "s3_resampling_diagnostic": {
            "n_obs_native_event_time": int(len(s3_ens_net_primary["net"])),
            "native_derived_ppy": round(ppy_s3_ens, 4),
            "n_obs_after_resample_to_monthly_grid": int(np.isfinite(s3_net_resampled).sum()),
            "n_target_months": int(n_m),
        },
        "periods_per_year_derived": round(ppy_combined, 4),
        "headline_full": combined_headline_full,
        "headline_on_demo": combined_headline_demo,
        "cost_funding_grid": combined_cost_funding,
        "exposure_timing_trace": (
            "combined_pre_overlay[t] indexed to month_end_dates[t] (the return REALIZED during "
            "month t). overlay_scale[t] computed from combined_pre_overlay[t-6:t] (months t-6..t-1 "
            "inclusive, STRICTLY before t). Verified: overlay_scale array has no dependency on "
            "combined_pre_overlay[t] or later at index t."),
        "cost_scaling_trace": (
            "Both combined_turn_scaled (fee term driver) and the funding term "
            "(funding_bp_per_period/1e4 * exposure_scale) are multiplied by the SAME "
            "overlay_scale array -- landmine 4 compliance verified by construction."),
    }
    print(f"  combined deployable sharpe_ann (full, rt13/fund0): {combined_headline_full.get('sharpe_ann')}", flush=True)
    print(f"  combined deployable sharpe_ann (on_demo, rt13/fund0): {combined_headline_demo.get('sharpe_ann')}", flush=True)

    # landmine-4 unit check: funding-scenario deltas must scale with `es`, not be an es-independent
    # flat shift -- verify directly by comparing two funding scenarios' delta against the realized
    # mean exposure scale
    fund0_mean = combined_cost_funding[f"rt{RT_PRIMARY_BP}_fund0.0"].get("ev_bp_obs")
    fund5_mean = combined_cost_funding[f"rt{RT_PRIMARY_BP}_fund5.0"].get("ev_bp_obs")
    mean_es = float(np.nanmean(overlay_scale))
    if fund0_mean is not None and fund5_mean is not None:
        observed_delta_bp = fund0_mean - fund5_mean
        expected_delta_bp_if_flat = 5.0  # what E2's bug would have produced (es-independent)
        expected_delta_bp_if_scaled = 5.0 * mean_es  # what the fix should produce
        print(f"  [landmine-4 unit check] fund0 ev_bp={fund0_mean}, fund5 ev_bp={fund5_mean}, "
              f"observed_delta={observed_delta_bp:.4f}bp, mean_exposure_scale={mean_es:.4f}, "
              f"expected_delta_if_scaled~={expected_delta_bp_if_scaled:.4f}bp "
              f"(flat-bug would show ~{expected_delta_bp_if_flat:.4f}bp regardless of exposure)", flush=True)
        results["combined_deployable"]["landmine4_unit_check"] = {
            "fund0_ev_bp": fund0_mean, "fund5_ev_bp": fund5_mean,
            "observed_delta_bp": round(observed_delta_bp, 4),
            "mean_exposure_scale": round(mean_es, 4),
            "expected_delta_if_scaled_bp": round(expected_delta_bp_if_scaled, 4),
            "expected_delta_if_flat_bug_bp": expected_delta_bp_if_flat,
            "scales_with_exposure": bool(abs(observed_delta_bp - expected_delta_bp_if_scaled) < abs(observed_delta_bp - expected_delta_bp_if_flat)),
        }

    # ==================================================================================
    # PBO over the full 48-cell family (canaries excluded)
    # ==================================================================================
    print("=== PBO over the full family ===", flush=True)
    family_returns: dict[str, np.ndarray] = {}
    for L in S1_LOOKBACKS_L:
        family_returns[f"S1_L{L}"] = s1_nets_primary[f"L{L}"]["net"]
    family_returns["S1_ensemble"] = s1_ens_net_primary["net"]
    # S2 per-window primary-cost nets were not all retained from the main scoring loop above
    # (only the primary W_beta=52 cell's net was kept) -- reconstruct the other two windows'
    # aligned net-return arrays here for the PBO family.
    s2_family_nets = {}
    for w_beta in S2_BETA_WINDOWS:
        book = beta_books[w_beta]
        sc = score_weekly_book_to_monthly(book["W"], LOGRET_F, fridays, month_end_fridays_idx_arr)
        valid = np.isfinite(sc["net"])
        net_v, turn_v = sc["net"][valid], sc["turn"][valid]
        net_costed = apply_costs(net_v, turn_v, RT_PRIMARY_BP)
        s2_family_nets[f"S2_Wbeta{w_beta}"] = net_costed
    family_returns.update(s2_family_nets)
    for hw in S3_HOLD_WEEKS:
        family_returns[f"S3_hold{hw}wk"] = s3_nets_by_hp[hw]["net"]
    family_returns["S3_ensemble"] = s3_ens_net_primary["net"]
    family_returns["combined_deployable"] = combined_net_v

    pbo = pbo_cscv(family_returns, n_splits=8)
    print(f"  PBO over {len(family_returns)}-cell aligned family: {pbo}", flush=True)
    results["pbo_full_family"] = {"pbo": pbo, "n_cells_in_family": len(family_returns),
                                   "cell_names": sorted(family_returns.keys())}

    # ==================================================================================
    # N_trials ledger verification
    # ==================================================================================
    n_trials_breakdown = {
        "S1": 4, "S2": 3, "S3": 4, "combined_deployable": 1,
        "cost_funding_grid_4_headline_x_3_cost_x_3_funding": 36,
    }
    n_trials_sum = sum(n_trials_breakdown.values())
    results["n_trials_ledger_breakdown"] = {**n_trials_breakdown, "sum": n_trials_sum,
                                             "matches_declared_48": bool(n_trials_sum == N_TRIALS)}
    print(f"  N_trials ledger: {n_trials_breakdown} sum={n_trials_sum} matches_declared_48="
          f"{n_trials_sum == N_TRIALS}", flush=True)

    # ==================================================================================
    # adoption gate evaluation
    # ==================================================================================
    print("=== adoption gate evaluation ===", flush=True)
    drop5_positive = (combined_headline_full.get("drop5_sharpe") or -1) > 0
    dsr_pass = (combined_headline_full.get("dsr", {}) or {}).get("dsr", 0) is not None and \
               (combined_headline_full.get("dsr", {}).get("dsr") or 0) >= 0.95
    pbo_pass = (pbo is not None) and (pbo <= 0.25)
    sharpe_full_positive = (combined_headline_full.get("sharpe_ann") or -1) > 0
    sharpe_demo_positive = (combined_headline_demo.get("sharpe_ann") or -1) > 0

    s1_signature_holds = s1_regime.get("max_drawdown_in_window_pct") is not None and s1_regime["max_drawdown_in_window_pct"] > 2.0
    s2_beta_near_zero = s2_primary_ex_post_beta is not None and abs(s2_primary_ex_post_beta) < 0.25
    s2_resilience = all(
        w.get("s2_resilient_vs_broad") is not False for w in s2_regime["windows"].values()
    )
    s2_signature_holds = bool(s2_beta_near_zero and s2_resilience)
    s3_liquidity_holds = s3_liquidity.get("drift_concentrated_in_less_liquid_half") is True
    s3_no_pure_decay = not bool(s3_decay.get("pure_decay_to_zero_pattern"))
    s3_signature_holds = bool(s3_liquidity_holds and s3_no_pure_decay)

    all_regime_signatures_hold = bool(s1_signature_holds and s2_signature_holds and s3_signature_holds)

    gate = {
        "1_sharpe_positive_full": sharpe_full_positive,
        "1_sharpe_positive_on_demo": sharpe_demo_positive,
        "2_dsr_pass": dsr_pass,
        "3_pbo_le_025": pbo_pass,
        "4_drop5_positive": drop5_positive,
        "5_all_regime_signatures_hold": all_regime_signatures_hold,
        "5_detail": {"S1": s1_signature_holds, "S2": s2_signature_holds, "S3": s3_signature_holds},
    }
    all_pass = bool(sharpe_full_positive and sharpe_demo_positive and dsr_pass and pbo_pass and
                     drop5_positive and all_regime_signatures_hold)
    gate["ADOPTION_RECOMMENDATION"] = "PAPER-FIRST DEPLOY" if all_pass else "DO NOT DEPLOY (negative/failed gate)"
    print(json.dumps(gate, indent=1, default=str), flush=True)
    results["adoption_gate"] = gate

    results["meta"]["runtime_sec"] = round(time.time() - t0, 2)
    (OUT / "results_e3.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"=== DONE in {time.time() - t0:.1f}s -- wrote {OUT / 'results_e3.json'} ===", flush=True)


if __name__ == "__main__":
    main()
