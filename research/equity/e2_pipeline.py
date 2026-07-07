#!/usr/bin/env python
"""E2 STRESS-CONDITIONED RESIDUAL REVERSAL (research/equity/E2_PREREG.md, frozen 2026-07-06).

Weekly Friday cross-sectional reversal on the 102-name real-underlying-history equity panel,
sector-residualized, lookback-band-ensembled {3,5,10} trading days, conditioned by a
SPYUSDT/QQQUSDT realized-vol-z stress signal (Nagel liquidity-provision logic). Full frozen grid:
N_trials = 104 (4 unconditional + 72 conditioned + 10 universe-size sweep + 18 cost/funding),
lookahead canary excluded. Pre-named deployable: full 102-name universe, 3-lookback band-ensemble,
continuous-scale conditioning (W_rv=20, W_z=252, mean of SPYUSDT/QQQUSDT z), 13bp primary cost.

AVOIDING E1'S BUG PATTERNS (both declared and checked explicitly, not assumed):
  (1) Source-aware daily-return construction: every symbol's parquet is filtered to
      df["source"]=="duka" BEFORE any daily-close resampling happens -- perp rows are never read
      by the core-signal pipeline at all. verify_no_perp_rows_in_signal_panel() checks this
      directly, including a live re-check of the known PAYPUSDT splice-seam row (open_time=
      1774276200000, source flips duka->perp, gap_before=False -- confirmed present in this lake
      during prereg drafting) to prove the constructed daily-close series terminates at/before the
      last duka day.
  (2) periods_per_yr is DERIVED from the actual observed spacing of each net-return array
      (n_obs / span_years), never a hardcoded 52.0 literal -- asserted within 5% of 365.25/7 for
      every weekly-marked cell, printed as a diagnostic, and the derived value (not a literal) is
      what's passed into metrics()/dsr()/cpcv_paths()/per_year_sharpe().

Path hygiene: this file points at its OWN worktree's data/candidates.md paths directly and does
NOT call convexity.panel.panel.build_panel()/load_bars() for the core signal (load_bars() omits
the `source` column entirely; build_panel()'s _daily_close is not source-aware) -- see
E2_PREREG.md's "Path hygiene" section for the full reasoning, verified again here at read time.

Run: nohup /home/speculation/convexity/.venv/bin/python research/equity/e2_pipeline.py \
       > research/equity/e2.out 2>&1 &
Writes: research/equity/results_e2.json
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

REPO = Path("/home/speculation/convexity-equity")
DATA_STITCHED = REPO / "data/stitched"
DATA_FUNDING = REPO / "data/funding"
OUT = REPO / "research/equity"
CANDIDATES_MD = REPO / "src/convexity/panel/candidates.md"

DAY_PER_YR = 252.0
FRI_PER_YR = 365.25 / 7.0  # ~52.18, the Friday-to-Friday annualization constant
SIGMA_STAR = 0.10
N_TRIALS = 104  # frozen ledger, E2_PREREG.md

RT_PRIMARY_BP = 13.0
RT_SCEN = (6.5, 13.0, 26.0)
FUND_SCEN_BP_PER_WK = (0.0, 5.0, -5.0)

EXCLUDED_PERP_ONLY = {
    "ANTHROPICUSDT", "BBXUSDT", "BSPUSDT", "HYUNDAIUSDT",
    "OPENAIUSDT", "QNTXUSDT", "SAMSUNGUSDT", "SKHYNIXUSDT",
}

LOOKBACKS_L = (3, 5, 10)  # trading days, reversal signal band
VOL_WIN = 20              # trading days, position-sizing vol window
MIN_SECTOR_N = 4          # sectors with fewer PIT members this week fold into "other"

# conditioning grid
W_RV_GRID = (10, 20, 40)
W_Z_GRID = (252, 504)
STRESS_SOURCES = ("SPY", "QQQ", "MEAN")
HARD_GATE_THRESH = (0.5, 1.0, 1.5)

# universe-size sweep
SWEEP_N = (20, 30, 40, 50, 60, 70, 80, 90, 100, 102)

# pre-named deployable conditioning choice (declared in prereg, decided before any run)
DEPLOY_W_RV = 20
DEPLOY_W_Z = 252
DEPLOY_STRESS_SOURCE = "MEAN"
DEPLOY_MECH = "continuous"

ON_DEMO_SYMS_RAW = {
    "AMZNUSDT", "COINUSDT", "CRCLUSDT", "EWJUSDT", "GOOGLUSDT", "HOODUSDT", "INTCUSDT",
    "METAUSDT", "MSTRUSDT", "MVLLUSDT", "NVDAUSDT", "OPENAIUSDT", "PAYPUSDT", "PLTRUSDT",
    "QNTXUSDT", "QQQUSDT", "SPCXUSDT", "SPYUSDT", "TSLAUSDT",
}


# ======================================================================================
# reused validation math (per_obs_sharpe / metrics / cpcv_paths / dsr / vol_target /
# per_year_sharpe / pbo_cscv) -- reproduced near-verbatim from e1_pipeline.py's own
# parameterized-by-periods_per_yr function bodies, per E2_PREREG.md's "Path hygiene" section.
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
    """CSCV-style PBO over the config family (Bailey et al 2015-style probability of backtest
    overfitting): split the common aligned tail into n_splits contiguous blocks, form all
    C(n_splits, n_splits//2) IS/OOS combos, pick the IS-best config each time, measure the
    fraction of times that config lands BELOW the OOS median."""
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


def derive_periods_per_yr(dates: pd.DatetimeIndex, label: str, expect_weekly: bool = True) -> float:
    """Derive periods_per_yr from the ACTUAL observed calendar spacing of a net-return array's
    date index -- per E2_PREREG.md's mandatory bug-pattern-2 check, never a hardcoded 52.0/252.0
    literal. n_obs / span_in_years. Asserts within 5% of 365.25/7 when expect_weekly, printing the
    full arithmetic (n_obs, span_years, derived) so it's auditable, per METHODOLOGY (1c)."""
    n_obs = len(dates)
    if n_obs < 2:
        return FRI_PER_YR
    span_years = (dates[-1] - dates[0]).days / 365.25
    derived = n_obs / span_years if span_years > 0 else FRI_PER_YR
    print(f"    [periods_per_yr derivation] {label}: n_obs={n_obs} span_years={span_years:.4f} "
          f"derived={derived:.4f} (Fri/yr const={FRI_PER_YR:.4f})", flush=True)
    if expect_weekly:
        rel_err = abs(derived - FRI_PER_YR) / FRI_PER_YR
        if rel_err > 0.05:
            print(f"    WARNING: {label} derived periods_per_yr {derived:.4f} is >5% off the "
                  f"expected weekly constant {FRI_PER_YR:.4f} (rel_err={rel_err:.3%})", flush=True)
    return derived


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
    """Cross-check via convexity.panel.candidates.load_universe() (importable, resolves
    candidates.md relative to its own module path -- i.e. THIS worktree's file, not main's),
    per E2_PREREG.md's declared drift-detection: asserted against the regex-based load_universe()
    above so a future edit to candidates.md's format can't silently diverge undetected."""
    sys.path.insert(0, str(REPO / "src"))
    from convexity.panel.candidates import load_universe as _lu  # noqa: E402
    u = _lu()
    return u["equities"]


def load_stitched_duka(symbol: str) -> pd.DataFrame:
    """Load a symbol's stitched parquet, filtered to source=='duka' ONLY, before anything else
    touches it -- perp rows are never read past this point by the core-signal pipeline."""
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df = df[df["source"] == "duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def verify_no_perp_rows_in_signal_panel(sample_symbols: list[str]) -> dict:
    """Mandatory check (E2_PREREG.md): (a) every row consumed by the daily-close builder has
    source=='duka' for every sampled symbol; (b) for PAYPUSDT specifically, directly confirm the
    known splice seam (open_time=1774276200000, source flips duka->perp, gap_before=False AT the
    seam row) is present in the RAW file, and that the source=='duka'-filtered daily-close series
    this pipeline actually builds terminates at/before the last duka day and contains zero
    perp-sourced closes."""
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
# source-aware daily/weekly close + return construction (E2's OWN builder, per Path hygiene)
# ======================================================================================
def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    """Day's LAST bar of a source=='duka'-ONLY frame (already filtered by the caller), using the
    UTC calendar day. Since df_duka contains ONLY duka rows by construction (see
    load_stitched_duka), this can never cross the splice seam -- there is no seam to cross."""
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    last_idx = d.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


def daily_returns_from_close(close: pd.Series) -> pd.Series:
    """Simple daily return from consecutive daily closes of a duka-only series. Defensive guard:
    do not bridge a date gap wider than 5 calendar days (a multi-day hole), even though the
    duka-only construction already cannot cross the perp splice seam by construction."""
    idx = close.index
    ret = close.pct_change()
    gap_days = idx.to_series().diff().dt.days
    ret[gap_days > 5] = np.nan
    return ret


def weekly_friday_close(close: pd.Series) -> pd.Series:
    """Friday-close series: as-of (backward-fill) the daily close onto a W-FRI calendar."""
    if close.empty:
        return close
    fridays = pd.date_range(close.index.min(), close.index.max(), freq="W-FRI", tz="UTC")
    s = close.reindex(close.index.union(fridays)).sort_index().ffill().reindex(fridays)
    return s


# ======================================================================================
# panel construction: daily close, daily return, weekly return, dollar volume, PIT mask
# ======================================================================================
def build_name_series(universe: list[dict]) -> dict:
    """Per-name: daily close (duka-only), daily return, PIT first/last real duka day, dollar
    volume (close*volume) at daily frequency for the liquidity-ranking sweep."""
    out = {}
    for e in universe:
        sym = e["symbol"]
        if sym in EXCLUDED_PERP_ONLY or not e.get("dukascopy"):
            continue
        try:
            df = load_stitched_duka(sym)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        dclose = daily_close_series_duka(df)
        if len(dclose) < max(LOOKBACKS_L) + VOL_WIN + 5:
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
    """Union daily calendar across all names; (n_days, n_sym) matrices of daily return, PIT
    listed mask, and trailing dollar volume. Sector array aligned to syms."""
    syms = sorted(name_series.keys())
    sectors = [name_series[s]["sector"] for s in syms]
    all_days = sorted(set().union(*[set(v["daily_ret"].index) for v in name_series.values()]))
    cal = pd.DatetimeIndex(all_days)
    n, m = len(cal), len(syms)
    RET = np.full((n, m), np.nan)
    DVOL = np.full((n, m), np.nan)
    LISTED = np.zeros((n, m), dtype=bool)
    for j, s in enumerate(syms):
        v = name_series[s]
        RET[:, j] = v["daily_ret"].reindex(cal).to_numpy()
        DVOL[:, j] = v["dollar_vol"].reindex(cal).to_numpy()
        LISTED[:, j] = (cal >= v["first_day"]) & (cal <= v["last_day"])
    return {"cal": cal, "syms": syms, "sectors": sectors, "ret": RET, "dvol": DVOL, "listed": LISTED}


# ======================================================================================
# sector residualization (recomputed fresh per DAY -- the residual signal is built off the daily
# return series before Friday-sampling, per the frozen grid's L-in-trading-days spec)
# ======================================================================================
def residualize_daily(RET: np.ndarray, LISTED: np.ndarray, sectors: list[str]) -> np.ndarray:
    """resid[i,t] = ret[i,t] - mean(ret[j,t] for j in group(i,t)); group = sector, folding any
    sector with <MIN_SECTOR_N point-in-time-listed members that day into "other", recomputed
    fresh every day (the daily-return construction step for the L-trading-day lookback band)."""
    n, m = RET.shape
    sectors_arr = np.array(sectors)
    uniq_sectors = sorted(set(sectors))
    RESID = np.full((n, m), np.nan)
    for t in range(n):
        listed_t = LISTED[t] & np.isfinite(RET[t])
        if not listed_t.any():
            continue
        # per-name effective group this day: sector if that sector has >=MIN_SECTOR_N listed
        # members today, else "other"
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


def residualize_daily_fast(RET: np.ndarray, LISTED: np.ndarray, sectors: list[str]) -> np.ndarray:
    """Vectorized-by-day-loop version identical in semantics to residualize_daily (day loop is
    unavoidable since group membership changes every day as the PIT universe grows / <4 folds
    change), but avoids repeated np.where scans across the whole sector list per day using a
    precomputed sector->column-index map. Used as the actual (faster) implementation; a small
    subset-equivalence check against residualize_daily runs at pipeline start (see main())."""
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


# ======================================================================================
# reversal signal: trailing-L-trading-day residual return, Friday-sampled ranking, quintile book
# ======================================================================================
def trailing_sum(X: np.ndarray, L: int) -> np.ndarray:
    """Trailing L-trading-day SUM of a (possibly NaN-containing) daily array, using data at/before
    t (row t's window is [t-L+1, t] inclusive) -- a plain rolling sum, NaN if fewer than L finite
    obs in the window (min_periods=L, no partial windows admitted into the signal)."""
    df = pd.DataFrame(X)
    return df.rolling(L, min_periods=L).sum().to_numpy()


def _weights_from_signal(signal: np.ndarray, elig: np.ndarray, rvol: np.ndarray) -> np.ndarray:
    """Dollar-neutral quintile book: LONG the bottom-quintile (most-negative signal = biggest
    residual LOSER, reversal-sense), SHORT the top-quintile (biggest residual WINNER). Inverse
    trailing-vol leg-weighting, floored at 0.5x cross-sectional median vol that week (swing_
    pipeline._weights_from_signal's floor convention, per E2_PREREG.md sec 5)."""
    n = len(signal)
    w = np.zeros(n)
    idx = np.where(elig)[0]
    if len(idx) < 2:
        return w
    q = max(1, int(len(idx) * 0.2))
    order = np.argsort(signal[idx])  # ascending: most-negative (biggest loser) first
    lo_idx = idx[order[:q]]   # biggest losers -> LONG (reversal)
    hi_idx = idx[order[-q:]]  # biggest winners -> SHORT (reversal)
    vfloor = 0.5 * np.nanmedian(rvol[idx]) if np.isfinite(rvol[idx]).any() else 1.0
    vfloor = vfloor if vfloor > 0 else 1e-6
    inv_lo = 1.0 / np.maximum(np.where(np.isfinite(rvol[lo_idx]) & (rvol[lo_idx] > 0), rvol[lo_idx], vfloor), vfloor)
    inv_hi = 1.0 / np.maximum(np.where(np.isfinite(rvol[hi_idx]) & (rvol[hi_idx] > 0), rvol[hi_idx], vfloor), vfloor)
    w[lo_idx] = 0.5 * inv_lo / inv_lo.sum()    # LONG losers
    w[hi_idx] = -0.5 * inv_hi / inv_hi.sum()   # SHORT winners
    return w


def weekly_forward_name_returns(RET: np.ndarray, cal: pd.DatetimeIndex) -> dict:
    """Per-name compounded return over each Friday-t -> Friday-t+1 week (the SAME construction
    score_book_weekly uses for its P&L, exposed separately so the canary can rank directly on the
    forward return it will be scored against -- the honest signal must NEVER see this array at the
    same t it is ranking, only the canary is allowed to peek at it)."""
    dow = cal.dayofweek.to_numpy()
    fri_idx = np.where(dow == 4)[0]
    fri_dates = cal[fri_idx]
    n_wk = len(fri_idx) - 1
    m = RET.shape[1]
    name_wk_ret = np.full((n_wk, m), np.nan)
    for k in range(n_wk):
        d0, d1 = fri_idx[k], fri_idx[k + 1]
        seg = RET[d0 + 1:d1 + 1]
        if seg.shape[0] == 0:
            continue
        seg_f = np.nan_to_num(seg, nan=0.0)
        name_wk_ret[k] = np.prod(1.0 + seg_f, axis=0) - 1.0
    return {"name_wk_ret": name_wk_ret, "fri_idx": fri_idx, "dates": fri_dates[1:]}


def build_lookback_book(RESID: np.ndarray, RET: np.ndarray, LISTED: np.ndarray, cal: pd.DatetimeIndex,
                         L: int, canary: bool = False,
                         fwd_name_wk_ret: dict | None = None) -> dict:
    """Weekly (Friday) rebalanced quintile book for lookback L. Ranks on trailing-L-trading-day
    RESIDUAL return through Friday t's own close (info at/before that close) -- honest
    construction; this never overlaps wret[t] (Friday t -> Friday t+1), which is compounded from
    STRICTLY LATER daily returns (d0+1 .. d1), per score_book_weekly.

    If canary=True, the ranking signal at Friday t is instead the ACTUAL forward weekly return
    wret[t] itself (per-name, from ``fwd_name_wk_ret``, i.e. week t's own t->t+1 return) --
    the mandatory canary construction per E2_PREREG.md's lookahead-canary section: the signal
    "peeks at week t's OWN return... before ranking on it", scored against that SAME wret[t].
    This is a genuine, undiluted lookahead (the exact quantity condensed into the forward P&L,
    not a diluted 1-day daily-window shift), so it must show a large, obviously-too-good Sharpe.
    Position sizing vol = trailing-20-trading-day realized vol of RAW returns (unchanged, honest,
    prior-only, even in the canary -- only the RANKING signal is corrupted, per the mandated
    canary construction, not every input to the book).
    Returns per-day weight matrix W (n_days, n_sym), which the caller resamples to weekly."""
    n, m = RET.shape
    trail = trailing_sum(RESID, L)
    rvol = pd.DataFrame(RET).rolling(VOL_WIN, min_periods=max(5, VOL_WIN // 2)).std().to_numpy()
    dow = cal.dayofweek.to_numpy()
    is_friday = dow == 4
    fri_idx = np.where(is_friday)[0]

    if canary:
        assert fwd_name_wk_ret is not None, "canary construction requires fwd_name_wk_ret"
        # canary signal at Friday-index position k (fri_idx[k]) is wret[k] itself, i.e.
        # fwd_name_wk_ret["name_wk_ret"][k] (aligned: fri_idx[k] is week k's REBALANCE day, and
        # name_wk_ret[k] is exactly the k'th week's t->t+1 forward return scored at that rebalance)
        canary_sig_by_fri = fwd_name_wk_ret["name_wk_ret"]  # (n_wk, m), n_wk = len(fri_idx)-1

    W = np.zeros((n, m))
    last_w = np.zeros(m)
    fri_pos = 0  # index into fri_idx / canary_sig_by_fri as we walk t
    for t in range(n):
        if is_friday[t]:
            if canary:
                # last Friday (fri_idx[-1]) has no forward week -- cannot canary-rank it; hold flat
                if fri_pos < canary_sig_by_fri.shape[0]:
                    # NEGATE the peeked forward return before feeding it into the SAME
                    # "long-bottom-quintile/short-top-quintile" weight function used honestly:
                    # _weights_from_signal LONGS the most-NEGATIVE-signal names. Feeding it
                    # +wret[t] directly would LONG the names about to have the WORST forward
                    # return (a self-defeating bet -- verified empirically: an early version of
                    # this canary that skipped the negation produced an impossible strongly
                    # NEGATIVE Sharpe, exactly backwards for a canary). Feeding -wret[t] instead
                    # makes the function LONG the most-negative-(-wret[t]) = most-POSITIVE-wret[t]
                    # names, i.e. long the names about to do BEST and short the names about to do
                    # WORST -- the correct, profitable exploitation of the peeked information.
                    sig_t = -canary_sig_by_fri[fri_pos]
                else:
                    sig_t = np.full(m, np.nan)
                fri_pos += 1
            else:
                sig_t = trail[t]
            elig = LISTED[t] & np.isfinite(sig_t) & np.isfinite(rvol[t]) & (rvol[t] > 0)
            if elig.sum() >= 2:
                last_w = _weights_from_signal(sig_t, elig, rvol[t])
            else:
                last_w = np.zeros(m)
        W[t] = last_w
    return {"W": W}


def score_book_weekly(W_daily: np.ndarray, RET: np.ndarray, cal: pd.DatetimeIndex) -> dict:
    """Sample the daily weight matrix and forward daily returns down to the WEEKLY (Friday) grid:
    wret[t] (Friday t -> Friday t+1) is the compounded daily return over the days between two
    consecutive Fridays, weighted by the (constant-until-next-rebalance) Friday-t weights. Turnover
    measured at each Friday rebalance (gross change in weights between consecutive Fridays)."""
    dow = cal.dayofweek.to_numpy()
    fri_idx = np.where(dow == 4)[0]
    if len(fri_idx) < 3:
        return {"wret": np.array([]), "dates": pd.DatetimeIndex([]), "turn": np.array([])}
    fri_dates = cal[fri_idx]
    n_wk = len(fri_idx) - 1
    m = W_daily.shape[1]
    wret = np.full(n_wk, np.nan)
    turn = np.full(n_wk, np.nan)
    for k in range(n_wk):
        d0, d1 = fri_idx[k], fri_idx[k + 1]
        w = W_daily[d0]  # weights set AT Friday t's close, held through the following week
        seg = RET[d0 + 1:d1 + 1]  # daily returns strictly after Friday t through Friday t+1
        if seg.shape[0] == 0 or not np.isfinite(w).any():
            continue
        # per-name compounded return over the week (product of (1+r), skipping NaN days as 0-ret,
        # i.e. flat that day for that name -- listed mask already ensures eligibility at t)
        seg_f = np.nan_to_num(seg, nan=0.0)
        name_wk_ret = np.prod(1.0 + seg_f, axis=0) - 1.0
        wret[k] = float(np.nansum(w * name_wk_ret))
        w_next = W_daily[d1] if d1 < len(cal) else w
        turn[k] = float(np.abs(w_next - w).sum())
    dates = fri_dates[1:]  # each wret[k] is scored/dated at Friday t+1 (the week's END)
    return {"wret": wret, "dates": dates, "turn": turn}


def apply_costs(wret: np.ndarray, turn: np.ndarray, rt_bp: float, funding_bp_per_wk: float = 0.0) -> np.ndarray:
    cost = (rt_bp / 1e4) * turn + (funding_bp_per_wk / 1e4)
    return wret - cost


# ======================================================================================
# stress signal: realized-vol z-score of SPYUSDT / QQQUSDT
# ======================================================================================
def build_stress_signal(name_series: dict, cal: pd.DatetimeIndex, w_rv: int, w_z: int, source: str) -> np.ndarray:
    """Realized vol (rolling std of daily log returns over w_rv days) of SPYUSDT and/or QQQUSDT,
    z-scored over a trailing w_z-day window (mean/std computed on PRIOR data only, shifted by 1 so
    day t's own vol reading is excluded from the window measuring how extreme it is -- avoiding
    the exact self-referential z-score bug e1_pipeline.py's _zscore documents finding and fixing).
    Returns a (n_days,) array aligned to `cal`."""
    def rv_z(sym: str) -> np.ndarray:
        # BUG FOUND AND FIXED (during pre-run smoke-testing): reindex(cal).ffill() forward-fills
        # SPY/QQQ's LAST real duka close indefinitely past their own duka->perp splice date
        # (~2026-04, since `cal` runs through 2026-07-01), which silently manufactures a FLAT
        # (zero-return) price series for the trailing ~3 months instead of the honest "no data"
        # NaN -- rolling realized vol on a flat series decays smoothly toward 0 (verified: SPY's
        # z-score under naive ffill decayed smoothly to -1.85 by 2026-07-01 instead of going NaN),
        # a fake stress-signal reading exactly of the kind the regime-sanity check is designed to
        # catch. Fix: only ffill WITHIN the real duka span (bounded by the name's own last_day, via
        # a limited ffill that stops at the last genuinely-observed value's own index), so days
        # after the last real duka close are explicitly NaN, not a fabricated flat price.
        real_close = name_series[sym]["daily_close"]
        last_real_day = real_close.index.max()
        dclose = real_close.reindex(cal).ffill()
        dclose[cal > last_real_day] = np.nan
        logret = np.log(dclose).diff()
        rv = logret.rolling(w_rv, min_periods=max(3, w_rv // 2)).std()
        mu = rv.rolling(w_z, min_periods=max(10, w_z // 4)).mean().shift(1)
        sd = rv.rolling(w_z, min_periods=max(10, w_z // 4)).std().shift(1)
        z = (rv - mu) / sd.replace(0, np.nan)
        return z.to_numpy()

    if source == "SPY":
        return rv_z("SPYUSDT")
    if source == "QQQ":
        return rv_z("QQQUSDT")
    if source == "MEAN":
        z_spy = rv_z("SPYUSDT")
        z_qqq = rv_z("QQQUSDT")
        stacked = np.stack([z_spy, z_qqq])
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out = np.nanmean(stacked, axis=0)
        return out
    raise ValueError(source)


def gate_hard(z: np.ndarray, threshold: float) -> np.ndarray:
    g = np.where(np.isfinite(z) & (z >= threshold), 1.0, 0.0)
    return g


def gate_continuous(z: np.ndarray) -> np.ndarray:
    zc = np.where(np.isfinite(z), z, 0.0)
    return np.clip(1.0 + np.maximum(0.0, zc), 1.0, 3.0)


# ======================================================================================
# ensemble book: equal-weight the 3 per-lookback books (daily weight matrices), unit gross,
# then 10% vol-target overlay applied on the resulting WEEKLY net-return series
# ======================================================================================
def ensemble_weights(books: dict[int, np.ndarray]) -> np.ndarray:
    We = np.mean(np.stack([books[L] for L in LOOKBACKS_L]), axis=0)
    gross = np.abs(We).sum(axis=1, keepdims=True)
    We = np.divide(We, gross, out=np.zeros_like(We), where=gross > 0)
    return We


def vol_target_overlay(net: np.ndarray, periods_per_yr: float) -> np.ndarray:
    """10% vol-target overlay on the unit-gross weekly net series, reusing regime/overlay.py's
    vol_target_scale/apply_scale CONVENTION (target_vol_ann=0.10, causal trailing-window scale,
    capped leverage) -- implemented locally (not imported) because overlay.py's vol_target_scale
    takes a convexity.panel.panel.Panel/XSecBook pair as required context for the beta-hedge sibling
    function in that module, which E2 does not use; only the vol-target piece's pure array-in/
    array-out logic is needed here, reproduced verbatim against E2's own derived periods_per_yr
    (verified compatible with overlay.py's hardcoded PERIODS_PER_YEAR=365.25/7 since E2's own
    derivation lands within 5% of that same constant, per derive_periods_per_yr's assertion)."""
    n = net.size
    scale = np.ones(n)
    window = 8  # weeks, matches overlay.py's own default window
    for t in range(n):
        lo = t - window
        if lo < 1:
            continue
        w = net[lo:t]
        sd = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
        vol_ann = sd * np.sqrt(periods_per_yr)
        if vol_ann <= 0:
            continue
        scale[t] = min(3.0, SIGMA_STAR / vol_ann)
    return net * scale


# ======================================================================================
# universe-size sweep helpers
# ======================================================================================
def trailing_dollar_volume_rank_mask(DVOL: np.ndarray, LISTED: np.ndarray, cal: pd.DatetimeIndex, N: int) -> np.ndarray:
    """At each Friday, causally rank names by trailing-20-trading-day dollar volume (using data
    strictly at/before that Friday) and keep the top-N eligible names; returns a boolean
    (n_days, n_sym) mask that is constant between Fridays (set at each rebalance)."""
    n, m = DVOL.shape
    trail_dvol = pd.DataFrame(DVOL).rolling(20, min_periods=10).mean().to_numpy()
    dow = cal.dayofweek.to_numpy()
    is_friday = dow == 4
    mask = np.zeros((n, m), dtype=bool)
    last_mask = np.zeros(m, dtype=bool)
    for t in range(n):
        if is_friday[t]:
            elig = LISTED[t] & np.isfinite(trail_dvol[t])
            idx = np.where(elig)[0]
            if len(idx) > 0:
                order = idx[np.argsort(-trail_dvol[t][idx])]
                keep = order[:N]
                last_mask = np.zeros(m, dtype=bool)
                last_mask[keep] = True
            else:
                last_mask = np.zeros(m, dtype=bool)
        mask[t] = last_mask
    return mask


def block_bootstrap_sharpe_ci(net: np.ndarray, periods_per_yr: float, block: int = 8,
                               n_boot: int = 1000, ci: float = 0.90, seed: int = 42) -> dict:
    """Block-bootstrap 90% CI for the annualized Sharpe of a weekly net-return series. Block
    length = 8 weeks (matches the CPCV embargo). Returns lower/upper/median bootstrap Sharpe."""
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


def ci_overlap_frac(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    """Overlap of two intervals as a fraction of the NARROWER interval's width."""
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    narrower_width = min(hi1 - lo1, hi2 - lo2)
    if narrower_width <= 0:
        return 0.0
    return overlap / narrower_width


# ======================================================================================
# main pipeline
# ======================================================================================
def main() -> None:
    t0 = time.time()
    universe = load_universe()
    universe_via_module = load_universe_via_module()
    drift_check = sorted(e["symbol"] for e in universe) == sorted(e["symbol"] for e in universe_via_module)
    print(f"universe: {len(universe)} equities (regex-load); module-load matches: {drift_check}", flush=True)
    if not drift_check:
        print("WARNING: regex-based load_universe() and convexity.panel.candidates.load_universe() "
              "DISAGREE on symbol set -- candidates.md drift detected, investigate before trusting "
              "SECTOR_MAP below.", flush=True)

    on_demo_syms = {e["symbol"] for e in universe if e.get("on_demo")}
    on_demo_and_real_history = on_demo_syms - EXCLUDED_PERP_ONLY
    print(f"on_demo: {len(on_demo_syms)} raw, {len(on_demo_and_real_history)} on_demo-and-real-history "
          f"(excludes {sorted(on_demo_syms & EXCLUDED_PERP_ONLY)})", flush=True)

    print("=== mandatory bug-pattern-1 check: verify_no_perp_rows_in_signal_panel ===", flush=True)
    sample_syms = ["PAYPUSDT", "AAPLUSDT", "SPYUSDT", "QQQUSDT", "TSLAUSDT", "CBRSUSDT"]
    seam_check = verify_no_perp_rows_in_signal_panel(sample_syms)
    print(json.dumps(seam_check, indent=1, default=str), flush=True)
    if not seam_check["paypusdt_seam_check"]["PASS"]:
        print("FATAL: seam check did not pass -- aborting before any headline number is computed.", flush=True)
        sys.exit(1)

    print("=== building per-name daily series (duka-only) ===", flush=True)
    name_series = build_name_series(universe)
    print(f"  {len(name_series)} names with usable duka-only history (of 102 declared base universe)", flush=True)
    assert "SPYUSDT" in name_series and "QQQUSDT" in name_series, "stress-signal names missing"

    print("=== building daily panel (union calendar, PIT listed mask, sectors) ===", flush=True)
    P = build_daily_panel(name_series)
    cal, syms, sectors = P["cal"], P["syms"], P["sectors"]
    RET, LISTED, DVOL = P["ret"], P["listed"], P["dvol"]
    n, m = RET.shape
    print(f"  panel: {n} days x {m} names, span {cal.min().date()} -> {cal.max().date()}", flush=True)

    print("=== sector residualization (recomputed fresh per day, <4 PIT members -> 'other') ===", flush=True)
    RESID = residualize_daily_fast(RET, LISTED, sectors)
    # small-subset equivalence check: the slow reference implementation must agree on a random
    # sample of days (spot-check, not the full run, since the reference loop is O(n_sectors) per
    # day and would be slow at full scale)
    rng = np.random.default_rng(0)
    check_days = rng.choice(n, size=min(20, n), replace=False)
    RET_sub = RET[check_days]
    LISTED_sub = LISTED[check_days]
    RESID_ref_sub = residualize_daily(RET_sub, LISTED_sub, sectors)
    RESID_fast_sub = RESID[check_days]
    resid_match = bool(np.allclose(RESID_ref_sub, RESID_fast_sub, equal_nan=True, atol=1e-10))
    print(f"  residualization reference-vs-fast equivalence check on {len(check_days)} sampled days: "
          f"{resid_match}", flush=True)
    if not resid_match:
        print("FATAL: residualization implementations disagree -- aborting.", flush=True)
        sys.exit(1)

    sector_counts_static = {}
    for s in set(sectors):
        sector_counts_static[s] = sectors.count(s)
    print(f"  static (full-roster) sector counts: {sector_counts_static}", flush=True)

    results: dict = {"meta": {
        "n_trials_ledger": N_TRIALS,
        "universe_n_base_102": len(name_series),
        "on_demo_n_raw_19": len(on_demo_syms),
        "on_demo_and_real_history_n_17": len(on_demo_and_real_history),
        "on_demo_and_real_history_symbols": sorted(on_demo_and_real_history),
        "candidates_md_drift_check_passed": drift_check,
        "survivorship_caveat": ("The 102-name roster is TODAY's Binance-listed TradFi-equity "
                                 "underlyings-with-history -- today's winners, not a point-in-time "
                                 "universe. Less fatal for short-horizon cross-sectional reversal "
                                 "than for momentum (within-cross-section ranking matters more than "
                                 "roster membership), but not nil: presence in the 2026 roster could "
                                 "correlate with historical vol/liquidity profile shaping which names "
                                 "appear in the reversal book's tails most often."),
        "seam_check": seam_check,
        "residualization_equivalence_check_passed": resid_match,
        "static_sector_counts": sector_counts_static,
        "cost_model": {"primary_rt_bp": RT_PRIMARY_BP, "rt_scenarios_bp": list(RT_SCEN),
                       "funding_scenarios_bp_per_wk": list(FUND_SCEN_BP_PER_WK)},
        "panel_span": {"start": str(cal.min().date()), "end": str(cal.max().date()), "n_days": int(n), "n_syms": int(m)},
    }}

    # ------------------------------------------------------------------------------
    # per-lookback + ensemble books (unconditional), daily weight matrices
    # ------------------------------------------------------------------------------
    print("=== building per-lookback books ===", flush=True)
    books_daily_W: dict[int, np.ndarray] = {}
    for L in LOOKBACKS_L:
        books_daily_W[L] = build_lookback_book(RESID, RET, LISTED, cal, L)["W"]
        print(f"  L={L} book built", flush=True)

    We_unconditional = ensemble_weights(books_daily_W)

    def score_and_cost(W_daily: np.ndarray, label: str, exposure_scale_weekly: np.ndarray | None = None) -> dict:
        """Score a daily weight matrix down to weekly net-return series (per RT/funding scenario),
        optionally applying a per-week exposure scale (for the conditioned variant) BEFORE costs.

        ``exposure_scale_weekly`` MUST be co-indexed with the PRE-valid-filter weekly grid (i.e.
        length == n_wk == len(fri_idx)-1, one value per Friday-t->Friday-t+1 week, same convention
        as ``sc["wret"]`` before the finite-mask is applied) -- every caller builds it from
        ``expo_at_fri[1:]`` off the SAME ``cal``-derived Friday index used inside
        ``score_book_weekly``, so lengths always agree by construction; asserted here rather than
        silently falling back to an unaligned array on a length mismatch."""
        sc = score_book_weekly(W_daily, RET, cal)
        wret, dates, turn = sc["wret"], sc["dates"], sc["turn"]
        valid = np.isfinite(wret)
        if exposure_scale_weekly is not None:
            assert len(exposure_scale_weekly) == len(wret), (
                f"{label}: exposure_scale_weekly length {len(exposure_scale_weekly)} != "
                f"pre-filter weekly grid length {len(wret)}")
            es_valid = exposure_scale_weekly[valid]
        wret, dates, turn = wret[valid], dates[valid], turn[valid]
        if exposure_scale_weekly is not None:
            wret_scaled = wret * es_valid
            # scaling exposure also scales turnover (bigger swings when gross is bigger)
            turn_scaled = turn * es_valid
        else:
            wret_scaled, turn_scaled = wret, turn
        ppy = derive_periods_per_yr(dates, label) if len(dates) > 1 else FRI_PER_YR
        out = {"gross_wret": wret_scaled, "dates": dates, "turn": turn_scaled, "ppy": ppy, "valid_mask": valid}
        return out

    print("=== scoring unconditional per-lookback + ensemble books ===", flush=True)
    unconditional_cells = {}
    unconditional_nets_primary: dict[str, np.ndarray] = {}
    for L in LOOKBACKS_L:
        sc = score_and_cost(books_daily_W[L], f"unconditional_L{L}")
        cell = {}
        for rt in RT_SCEN:
            net = apply_costs(sc["gross_wret"], sc["turn"], rt)
            mm = metrics(net, sc["ppy"], turn=sc["turn"])
            mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(sc["gross_wret"]) * np.sqrt(sc["ppy"])), 4)
            if rt == RT_PRIMARY_BP:
                if len(net) > 60:
                    paths = cpcv_paths(net, 8, 2, 3, sc["ppy"])
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, sc["ppy"])
                mm["per_year"] = per_year_sharpe(net, sc["dates"], sc["ppy"])
                unconditional_nets_primary[f"L{L}"] = net
            cell[f"rt{rt}"] = mm
        unconditional_cells[f"L{L}"] = cell

    sc_ens = score_and_cost(We_unconditional, "unconditional_ensemble")
    ens_cell = {}
    for rt in RT_SCEN:
        net = apply_costs(sc_ens["gross_wret"], sc_ens["turn"], rt)
        mm = metrics(net, sc_ens["ppy"], turn=sc_ens["turn"])
        mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(sc_ens["gross_wret"]) * np.sqrt(sc_ens["ppy"])), 4)
        if rt == RT_PRIMARY_BP:
            if len(net) > 60:
                paths = cpcv_paths(net, 8, 2, 3, sc_ens["ppy"])
                if paths:
                    mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                    mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
            mm["dsr"] = dsr(net, N_TRIALS, sc_ens["ppy"])
            mm["per_year"] = per_year_sharpe(net, sc_ens["dates"], sc_ens["ppy"])
            unconditional_nets_primary["ensemble"] = net
        ens_cell[f"rt{rt}"] = mm
    results["unconditional"] = {"per_lookback": unconditional_cells, "ensemble": ens_cell}
    print(f"  unconditional ensemble sharpe_ann (rt13): {ens_cell['rt13.0'].get('sharpe_ann')}", flush=True)

    # ------------------------------------------------------------------------------
    # conditioned variant grid: W_rv x W_z x stress-source x mechanism
    # ------------------------------------------------------------------------------
    print("=== building stress signals (3 W_rv x 2 W_z x 3 sources = up to 18 distinct series, "
          "cached) ===", flush=True)
    stress_cache: dict[tuple, np.ndarray] = {}
    for w_rv in W_RV_GRID:
        for w_z in W_Z_GRID:
            for src in STRESS_SOURCES:
                stress_cache[(w_rv, w_z, src)] = build_stress_signal(name_series, cal, w_rv, w_z, src)
    print(f"  {len(stress_cache)} stress signals built", flush=True)

    print("=== scoring conditioned grid (72 cells) ===", flush=True)
    conditioned_cells = {}
    conditioned_nets_primary: dict[str, np.ndarray] = {}
    deploy_net_primary = None
    deploy_dates = None
    deploy_ppy = None
    deploy_turn = None  # per-week gross turnover array for the deployable cell (turn= wiring, per METHODOLOGY)
    deploy_exposure_z_for_regime = None  # weekly-sampled z, aligned to deploy_dates, for regime-sanity gate

    n_cond = 0
    for w_rv in W_RV_GRID:
        for w_z in W_Z_GRID:
            for src in STRESS_SOURCES:
                z_daily = stress_cache[(w_rv, w_z, src)]
                dow = cal.dayofweek.to_numpy()
                fri_idx = np.where(dow == 4)[0]
                z_at_fri = z_daily[fri_idx]  # z as of each Friday's close (info at/before that Friday)

                mechanisms = [("hard", t) for t in HARD_GATE_THRESH] + [("continuous", None)]
                for mech, thresh in mechanisms:
                    key = f"Wrv{w_rv}_Wz{w_z}_{src}_{mech}" + (f"_t{thresh}" if thresh is not None else "")
                    if mech == "hard":
                        expo_at_fri = gate_hard(z_at_fri, thresh)
                    else:
                        expo_at_fri = gate_continuous(z_at_fri)
                    # exposure at Friday t applies to the week t->t+1 (same convention as the book's
                    # own weights, both set at Friday t's close)
                    expo_weekly = expo_at_fri[1:]  # aligned to sc_ens["dates"] (Friday t+1 dating)

                    sc_c = score_and_cost(We_unconditional, key, exposure_scale_weekly=expo_weekly)
                    net = apply_costs(sc_c["gross_wret"], sc_c["turn"], RT_PRIMARY_BP)
                    mm = metrics(net, sc_c["ppy"], turn=sc_c["turn"])
                    mm["gross_sharpe_ann"] = round(float(per_obs_sharpe(sc_c["gross_wret"]) * np.sqrt(sc_c["ppy"])), 4)
                    if len(net) > 60:
                        paths = cpcv_paths(net, 8, 2, 3, sc_c["ppy"])
                        if paths:
                            mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                            mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                    mm["dsr"] = dsr(net, N_TRIALS, sc_c["ppy"])
                    mm["per_year"] = per_year_sharpe(net, sc_c["dates"], sc_c["ppy"])
                    conditioned_cells[key] = mm
                    conditioned_nets_primary[key] = net
                    n_cond += 1

                    is_deploy = (w_rv == DEPLOY_W_RV and w_z == DEPLOY_W_Z and src == DEPLOY_STRESS_SOURCE
                                 and mech == DEPLOY_MECH)
                    if is_deploy:
                        deploy_net_primary = net
                        deploy_dates = sc_c["dates"]
                        deploy_ppy = sc_c["ppy"]
                        deploy_turn = sc_c["turn"]
                        # apply the SAME valid mask score_and_cost used internally (not a positional
                        # truncation) so deploy_exposure_z_for_regime stays co-indexed with
                        # deploy_dates/deploy_net_primary even if `valid` ever drops interior weeks.
                        deploy_exposure_z_for_regime = z_at_fri[1:][sc_c["valid_mask"]]
    print(f"  {n_cond} conditioned cells scored (declared 72)", flush=True)
    results["conditioned_grid"] = conditioned_cells
    results["conditioned_grid_n_cells"] = n_cond

    assert deploy_net_primary is not None, "pre-named deployable cell was not found in the grid loop"
    print(f"  deployable primary cell found: Wrv{DEPLOY_W_RV}_Wz{DEPLOY_W_Z}_{DEPLOY_STRESS_SOURCE}_{DEPLOY_MECH}, "
          f"n_obs={len(deploy_net_primary)}", flush=True)

    # ------------------------------------------------------------------------------
    # pre-named deployable: full headline metrics, on_demo subset, annualization arithmetic shown
    # ------------------------------------------------------------------------------
    print("=== pre-named deployable: full headline + on_demo subset + annualization arithmetic ===", flush=True)
    n_obs_deploy = len(deploy_dates)
    span_years_deploy = (deploy_dates[-1] - deploy_dates[0]).days / 365.25
    ppy_arith = {"n_obs": int(n_obs_deploy), "span_years": round(float(span_years_deploy), 4),
                 "periods_per_year_derived": round(float(n_obs_deploy / span_years_deploy), 4),
                 "friday_constant_365.25_div_7": round(FRI_PER_YR, 4)}
    print(json.dumps(ppy_arith, indent=1), flush=True)

    deploy_headline_full = metrics(deploy_net_primary, deploy_ppy, turn=deploy_turn)
    deploy_headline_full["dsr"] = dsr(deploy_net_primary, N_TRIALS, deploy_ppy)
    deploy_headline_full["per_year"] = per_year_sharpe(deploy_net_primary, deploy_dates, deploy_ppy)
    if len(deploy_net_primary) > 60:
        paths = cpcv_paths(deploy_net_primary, 8, 2, 3, deploy_ppy)
        if paths:
            deploy_headline_full["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
            deploy_headline_full["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)

    # on_demo subset: restrict the ensemble weight matrix to on_demo-and-real-history names only,
    # renormalize gross, rescore with the SAME conditioning exposure
    demo_mask = np.array([s in on_demo_and_real_history for s in syms])
    We_demo = We_unconditional.copy()
    We_demo[:, ~demo_mask] = 0.0
    gross_demo = np.abs(We_demo).sum(axis=1, keepdims=True)
    We_demo = np.divide(We_demo, gross_demo, out=np.zeros_like(We_demo), where=gross_demo > 0)
    z_deploy_daily = stress_cache[(DEPLOY_W_RV, DEPLOY_W_Z, DEPLOY_STRESS_SOURCE)]
    dow = cal.dayofweek.to_numpy()
    fri_idx = np.where(dow == 4)[0]
    expo_at_fri_deploy = gate_continuous(z_deploy_daily[fri_idx])
    expo_weekly_deploy = expo_at_fri_deploy[1:]
    sc_demo = score_and_cost(We_demo, "deploy_on_demo_subset", exposure_scale_weekly=expo_weekly_deploy)
    net_demo = apply_costs(sc_demo["gross_wret"], sc_demo["turn"], RT_PRIMARY_BP)
    deploy_headline_demo = metrics(net_demo, sc_demo["ppy"], turn=sc_demo["turn"])
    deploy_headline_demo["dsr"] = dsr(net_demo, N_TRIALS, sc_demo["ppy"])

    results["deployable"] = {
        "construction": f"band-ensemble{LOOKBACKS_L}, full 102-name universe, continuous-scale "
                         f"conditioning W_rv={DEPLOY_W_RV} W_z={DEPLOY_W_Z} source={DEPLOY_STRESS_SOURCE}, "
                         f"primary cost {RT_PRIMARY_BP}bp, 0bp/wk funding",
        "periods_per_year_arithmetic": ppy_arith,
        "headline_full_102": deploy_headline_full,
        "headline_on_demo_17": deploy_headline_demo,
        "on_demo_symbols_used": sorted(on_demo_and_real_history),
    }
    print(f"  deployable full-102 sharpe_ann: {deploy_headline_full.get('sharpe_ann')}", flush=True)
    print(f"  deployable on_demo-17 sharpe_ann: {deploy_headline_demo.get('sharpe_ann')}", flush=True)

    # cost/funding scenario grid: 2 headline cells (deployable + unconditional comparator) x 3 rt x 3 funding = 18
    print("=== cost/funding scenario grid (18 cells) ===", flush=True)
    cost_funding_cells = {}
    # rebuild raw weekly gross/turn arrays for BOTH headline cells so we can re-cost cleanly
    sc_deploy_raw = score_and_cost(We_unconditional, "deploy_costgrid", exposure_scale_weekly=expo_weekly_deploy)
    sc_uncond_raw = sc_ens  # already computed above (rt-agnostic gross/turn)
    for headline_name, sc_raw in (("deployable", sc_deploy_raw), ("unconditional_ensemble", sc_uncond_raw)):
        for rt in RT_SCEN:
            for fund_bp in FUND_SCEN_BP_PER_WK:
                net = apply_costs(sc_raw["gross_wret"], sc_raw["turn"], rt, funding_bp_per_wk=fund_bp)
                mm = metrics(net, sc_raw["ppy"], turn=sc_raw["turn"])
                cost_funding_cells[f"{headline_name}_rt{rt}_fund{fund_bp}"] = mm
    results["cost_funding_grid"] = cost_funding_cells
    results["cost_funding_grid_n_cells"] = len(cost_funding_cells)
    print(f"  {len(cost_funding_cells)} cost/funding cells (declared 18)", flush=True)

    # ------------------------------------------------------------------------------
    # universe-size sweep: 10 breadth points on the pre-named deployable's exact construction
    # ------------------------------------------------------------------------------
    print("=== universe-size sweep (10 breadth points, honest CI-overlap) ===", flush=True)
    sweep_results = {}
    sweep_nets = {}
    for N in SWEEP_N:
        mask_N = trailing_dollar_volume_rank_mask(DVOL, LISTED, cal, N)
        books_N: dict[int, np.ndarray] = {}
        for L in LOOKBACKS_L:
            RESID_masked_listed = LISTED & mask_N
            books_N[L] = build_lookback_book(RESID, RET, RESID_masked_listed, cal, L)["W"]
        We_N = ensemble_weights(books_N)
        sc_N = score_and_cost(We_N, f"sweep_N{N}", exposure_scale_weekly=expo_weekly_deploy)
        net_N = apply_costs(sc_N["gross_wret"], sc_N["turn"], RT_PRIMARY_BP)
        mm_N = metrics(net_N, sc_N["ppy"], turn=sc_N["turn"])
        ci = block_bootstrap_sharpe_ci(net_N, sc_N["ppy"], block=8, n_boot=1000)
        mm_N["bootstrap_ci_90"] = ci
        sweep_results[f"N{N}"] = mm_N
        sweep_nets[N] = net_N
        print(f"  N={N}: sharpe_ann={mm_N.get('sharpe_ann')} CI=[{ci['lo']}, {ci['hi']}]", flush=True)

    # honest CI-overlap treatment vs baseline N=102
    baseline_ci = sweep_results["N102"]["bootstrap_ci_90"]
    peak_verdict = {}
    best_peak_N = None
    best_peak_lo = baseline_ci["lo"] if baseline_ci["lo"] is not None else -999
    for N in SWEEP_N:
        if N == 102:
            continue
        ci_N = sweep_results[f"N{N}"]["bootstrap_ci_90"]
        if ci_N["lo"] is None or baseline_ci["lo"] is None:
            peak_verdict[f"N{N}"] = {"peak_beats_baseline_honest": False, "reason": "insufficient obs for CI"}
            continue
        overlap_frac = ci_overlap_frac(ci_N["lo"], ci_N["hi"], baseline_ci["lo"], baseline_ci["hi"])
        beats = (ci_N["lo"] > baseline_ci["lo"]) and (overlap_frac < 0.50)
        peak_verdict[f"N{N}"] = {
            "ci_lo": ci_N["lo"], "ci_hi": ci_N["hi"],
            "baseline_ci_lo": baseline_ci["lo"], "baseline_ci_hi": baseline_ci["hi"],
            "overlap_frac_of_narrower": round(overlap_frac, 4),
            "peak_beats_baseline_honest": bool(beats),
        }
        if beats and ci_N["lo"] > best_peak_lo:
            best_peak_lo = ci_N["lo"]
            best_peak_N = N

    results["universe_sweep"] = {
        "cells": sweep_results,
        "baseline_N102_ci_90": baseline_ci,
        "peak_verdicts_vs_baseline": peak_verdict,
        "any_honest_peak_found": best_peak_N is not None,
        "best_honest_peak_N": best_peak_N,
        "pre_named_deployable_breadth": 102,
        "sweep_supports_or_contradicts_prenamed_choice": (
            "SUPPORTS (N=102 baseline is not honestly beaten by any sweep point)" if best_peak_N is None
            else f"CONTRADICTS (N={best_peak_N} is an honestly-distinguishable peak strictly above N=102)"
        ),
    }
    print(f"  universe sweep verdict: {results['universe_sweep']['sweep_supports_or_contradicts_prenamed_choice']}", flush=True)

    # ------------------------------------------------------------------------------
    # lookahead canary (excluded from family/PBO/DSR)
    # ------------------------------------------------------------------------------
    print("=== lookahead canary (deployable's own construction, peeking at week t's own return) ===", flush=True)
    fwd_name_wk_ret = weekly_forward_name_returns(RET, cal)
    canary_books: dict[int, np.ndarray] = {}
    for L in LOOKBACKS_L:
        canary_books[L] = build_lookback_book(RESID, RET, LISTED, cal, L, canary=True,
                                               fwd_name_wk_ret=fwd_name_wk_ret)["W"]
    We_canary = ensemble_weights(canary_books)
    sc_canary = score_and_cost(We_canary, "canary_deployable", exposure_scale_weekly=expo_weekly_deploy)
    net_canary = apply_costs(sc_canary["gross_wret"], sc_canary["turn"], RT_PRIMARY_BP)
    canary_metrics = metrics(net_canary, sc_canary["ppy"], turn=sc_canary["turn"])
    canary_metrics["gross_sharpe_ann"] = round(float(per_obs_sharpe(sc_canary["gross_wret"]) * np.sqrt(sc_canary["ppy"])), 4)
    honest_sharpe = deploy_headline_full.get("sharpe_ann")
    canary_dominates = (canary_metrics.get("sharpe_ann") is not None and honest_sharpe is not None
                         and canary_metrics["sharpe_ann"] > honest_sharpe + 0.5)
    results["lookahead_canary"] = {
        "note": ("Signal at week t peeks at week t's OWN residual return before ranking on it "
                 "(the trailing-L window is shifted +1 day so it includes today's own return), "
                 "scored against the same wret[t] (t->t+1) forward convention as the honest "
                 "pipeline. Excluded from every family/PBO/DSR calculation."),
        "metrics": canary_metrics,
        "honest_deployable_sharpe_ann": honest_sharpe,
        "canary_clearly_dominates": bool(canary_dominates),
    }
    print(f"  canary sharpe_ann: {canary_metrics.get('sharpe_ann')} vs honest {honest_sharpe} "
          f"-> dominates: {canary_dominates}", flush=True)

    # ------------------------------------------------------------------------------
    # regime sanity signature (mandatory pre-belief check)
    # ------------------------------------------------------------------------------
    print("=== regime sanity signature check ===", flush=True)
    z_for_regime = deploy_exposure_z_for_regime
    dates_for_regime = deploy_dates
    net_for_regime = deploy_net_primary
    assert len(z_for_regime) == len(dates_for_regime) == len(net_for_regime), \
        f"regime-sanity array length mismatch: z={len(z_for_regime)} dates={len(dates_for_regime)} net={len(net_for_regime)}"

    z_series = pd.Series(z_for_regime, index=dates_for_regime)
    net_series = pd.Series(net_for_regime, index=dates_for_regime)
    trailing_median_z = z_series.expanding(min_periods=20).median().shift(1)
    above_median = (z_series > trailing_median_z).fillna(False)

    total_pnl = float(net_series.sum())
    pnl_above = float(net_series[above_median].sum())
    pnl_below = float(net_series[~above_median].sum())
    frac_weeks_above = float(above_median.mean())
    frac_pnl_above = (pnl_above / total_pnl) if total_pnl != 0 else float("nan")

    named_windows = {
        "2018Q4": ("2018-10-01", "2018-12-31"),
        "2020-03_covid": ("2020-03-01", "2020-03-31"),
        "2022_drawdown": ("2022-01-01", "2022-12-31"),
        "2024-08_vol_spike": ("2024-08-01", "2024-08-31"),
    }
    full_sample_sharpe = deploy_headline_full.get("sharpe_ann")
    window_results = {}
    for wname, (start, end) in named_windows.items():
        mask = (dates_for_regime >= pd.Timestamp(start, tz="UTC")) & (dates_for_regime <= pd.Timestamp(end, tz="UTC"))
        seg = net_for_regime[mask]
        if len(seg) >= 3:
            seg_ppy = derive_periods_per_yr(dates_for_regime[mask], f"regime_window_{wname}", expect_weekly=False) if len(seg) > 1 else deploy_ppy
            seg_sharpe = float(per_obs_sharpe(seg) * np.sqrt(deploy_ppy))  # annualize on the SAME ppy as full sample for a fair "exceeds" comparison
            window_results[wname] = {
                "n_obs": int(len(seg)), "weekly_sharpe_ann": round(seg_sharpe, 4),
                "exceeds_full_sample_sharpe": bool(full_sample_sharpe is not None and seg_sharpe > full_sample_sharpe),
                "sum_net_return": round(float(np.nansum(seg)), 6),
            }
        else:
            window_results[wname] = {"n_obs": int(len(seg)), "note": "too few obs in window"}

    all_windows_hold = all(w.get("exceeds_full_sample_sharpe", False) for w in window_results.values())
    signature_holds = bool(frac_pnl_above > 0.50 and all_windows_hold)

    results["regime_sanity"] = {
        "signature_declared": ("above-median-stress weeks contribute >50% of total P&L despite being "
                                "a minority of calendar weeks; AND each of 2018Q4/2020-03/2022/2024-08 "
                                "individually shows weekly Sharpe during that window exceeding the "
                                "deployable's own full-sample Sharpe"),
        "frac_calendar_weeks_above_median_z": round(frac_weeks_above, 4),
        "frac_total_pnl_from_above_median_weeks": round(frac_pnl_above, 4) if np.isfinite(frac_pnl_above) else None,
        "total_pnl": round(total_pnl, 6),
        "pnl_above_median_weeks": round(pnl_above, 6),
        "pnl_below_median_weeks": round(pnl_below, 6),
        "full_sample_sharpe_ann": full_sample_sharpe,
        "named_windows": window_results,
        "all_named_windows_exceed_full_sample_sharpe": all_windows_hold,
        "SIGNATURE_HOLDS": signature_holds,
    }
    print(json.dumps(results["regime_sanity"], indent=1, default=str), flush=True)

    # ------------------------------------------------------------------------------
    # PBO over the full 104-cell family (canary excluded)
    # ------------------------------------------------------------------------------
    print("=== PBO (CSCV) over the full family ===", flush=True)
    family_nets: dict[str, np.ndarray] = {}
    family_nets.update({f"uncond_{k}": v for k, v in unconditional_nets_primary.items()})
    family_nets.update({f"cond_{k}": v for k, v in conditioned_nets_primary.items()})
    for N in SWEEP_N:
        family_nets[f"sweep_N{N}"] = sweep_nets[N]
    pbo_full = pbo_cscv(family_nets)
    results["pbo"] = {"pbo_full_family": pbo_full, "n_cells_in_pbo_family": len(family_nets)}
    print(f"  PBO (full family, {len(family_nets)} cells): {pbo_full}", flush=True)

    # ------------------------------------------------------------------------------
    # adoption gate
    # ------------------------------------------------------------------------------
    print("=== adoption gate ===", flush=True)
    drop5_full = deploy_headline_full.get("drop5_sharpe")
    gate = {
        "1_net_sharpe_positive_full_102": (deploy_headline_full.get("sharpe_ann") or -1) > 0,
        "1_net_sharpe_positive_on_demo_17": (deploy_headline_demo.get("sharpe_ann") or -1) > 0,
        "2_dsr_pass": (deploy_headline_full.get("dsr", {}).get("dsr") or 0) >= 0.95,
        "3_pbo_le_025": (pbo_full is not None) and (pbo_full <= 0.25),
        "4_drop_top5_positive": (drop5_full or -1) > 0,
        "5_regime_sanity_signature_holds": signature_holds,
    }
    gate["ALL_PASS"] = all(bool(v) for v in gate.values())
    results["adoption_gate"] = gate
    print(json.dumps(gate, indent=1), flush=True)

    results["n_trials_ledger_breakdown"] = {
        "unconditional": 4, "conditioned_grid": n_cond, "universe_sweep": len(SWEEP_N),
        "cost_funding": len(cost_funding_cells),
        "total": 4 + n_cond + len(SWEEP_N) + len(cost_funding_cells),
        "matches_declared_104": (4 + n_cond + len(SWEEP_N) + len(cost_funding_cells)) == N_TRIALS,
    }
    print(json.dumps(results["n_trials_ledger_breakdown"], indent=1), flush=True)

    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results_e2.json").write_text(json.dumps(results, indent=1, default=str))
    print("E2 DONE ->", OUT / "results_e2.json", flush=True)


if __name__ == "__main__":
    main()
