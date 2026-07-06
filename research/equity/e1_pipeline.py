#!/usr/bin/env python
"""E1 PERP-NATIVE SUITE — end-to-end (research/equity/E1_PREREG.md, frozen 2026-07-06).

Three genuinely distinct signal families on the databento-underlying + Binance-perp equity lake
(research/equity/DATA.md): S1 funding-carry cross-section, S2 perp<->underlying basis mean
reversion, S3 overnight/weekend structure. Every grid cell in E1_PREREG.md is executed and charged
to N_trials = 179 (DSR). Reuses the metrics()/cpcv_paths()/dsr()/vol_target() formulas from
research/entry_exit/study/swing_pipeline.py and per_obs_sharpe's definition from
research/entry_exit/validation.py near-verbatim (this worktree cannot import swing_pipeline.py
directly -- it hardcodes REPO=.../convexity -- so the same math is reproduced here against this
worktree's own data paths).

Data: data/stitched/{SYMBOL}.parquet (open_time ms epoch, OHLCV, source in {duka,perp}, gap_before
bool) and data/funding/{SYMBOL}.parquet (calc_time ms epoch, funding_interval_hours,
last_funding_rate), both under the equity worktree's shared-lake symlinks (READ ONLY). Universe =
the 110-symbol "equities" JSON array in src/convexity/panel/candidates.md.

S2 additionally reads the PRE-STITCH raw sources -- data/klines/{SYMBOL}/ (raw Binance perp klines,
partitioned parquet, 24/7 from real listing) and data/dukascopy/{code}/{code}.parquet (raw databento
underlying, RTH-filtered, full 2018-2026 span) -- because data/stitched/ is a SEQUENTIAL splice
(verified: source flips 'duka'->'perp' with zero simultaneous overlap minutes, e.g. AAPLUSDT's last
duka bar is 2026-04-06 13:49 UTC and its first perp bar is 13:50 UTC) and cannot supply the
both-legs-simultaneous minutes S2's basis construction requires. The raw sources genuinely do
overlap (e.g. AAPLUSDT perp trades from 2026-04-06, databento underlying continues through
2026-07-02 -- about 3 months of true simultaneous 1-min coverage), consistent with DATA.md's
"perp history is weeks-to-months" characterization.

Session/return hygiene (non-negotiable, restated from the prereg):
  - never compute a 1-min return across gap_before=True, or across a source-change row (splice seam)
  - daily bar = day's LAST bar with gap_before=False (the real RTH close pre-splice; post-splice the
    perp trades 24/7 so "daily" for S1/S2 = last bar of the UTC day)
  - weekly returns = Friday-close to Friday-close on those daily closes
  - S3 RTH boundary = 09:30/16:00 America/New_York, DST-aware, via tz_convert (scripts/dbn_fetch.py
    convention)

Run: nohup /home/speculation/convexity/.venv/bin/python research/equity/e1_pipeline.py \
       > research/equity/e1.out 2>&1 &
Writes: research/equity/results_e1.json
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

sys.path.insert(0, "/home/speculation/convexity-equity/scripts")
# REUSE, not re-derive (METHODOLOGY): the raw databento underlying file
# (data/dukascopy/{code}/{code}.parquet) is NOT split-adjusted (verified directly: CRWDDBN sits
# flat ~$193-199 straight through its real, externally-verified 2026-07-02 4:1 split -- only
# data/stitched/ carries the adjustment, and stitched/ is the sequential splice S2 cannot use, see
# the S2 section below). S2 needs the SAME split detection/adjustment scripts/stitch_tradfi_1m.py
# already applies when building data/stitched/, applied here directly to the raw file before
# computing basis, or a real split shows up as a fake multi-hundred-percent "basis dislocation"
# (caught empirically: an early version of this pipeline's S2 canary showed an impossible -110
# annualized Sharpe, root-caused to exactly this -- CRWD's unadjusted split dominating d_lb).
from stitch_tradfi_1m import KNOWN_BAD_WINDOWS, _detect_splits, _mask_bad_windows, _split_adjust  # noqa: E402

REPO = Path("/home/speculation/convexity-equity")
DATA_STITCHED = REPO / "data/stitched"
DATA_FUNDING = REPO / "data/funding"
DATA_KLINES = REPO / "data/klines"        # raw perp klines (pre-splice, both legs overlap here)
DATA_DUKASCOPY = REPO / "data/dukascopy"  # raw databento underlying (despite legacy dir name)
OUT = REPO / "research/equity"
CANDIDATES_MD = REPO / "src/convexity/panel/candidates.md"

WK_PER_YR = 52.0
DAY_PER_YR = 252.0
SIGMA_STAR = 0.10
N_TRIALS = 179  # frozen ledger, E1_PREREG.md

RT_PRIMARY_BP = 13.0
RT_SCEN = (6.5, 13.0, 26.0)
S1_FEE_SCEN = (6.5, 13.0, 26.0)  # fee-only scenarios for S1 (funding is not a cost)

ET = "America/New_York"
RTH_OPEN = pd.Timestamp("09:30").time()
RTH_CLOSE = pd.Timestamp("16:00").time()


# ======================================================================================
# reused validation math (per_obs_sharpe / metrics / cpcv_paths / dsr / vol_target) --
# reproduced from research/entry_exit/study/swing_pipeline.py + validation.py verbatim
# formulas, pointed at this worktree's own data (swing_pipeline.py hardcodes the `main`
# worktree's REPO path and is not importable as-is per STATE.md).
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
        # embargo: drop `embargo` obs immediately adjacent to each selected block boundary
        if embargo > 0:
            idx = np.where(test)[0]
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
    """CSCV-style PBO over the config family: split the common aligned calendar into n_splits equal
    contiguous blocks, form all C(n_splits, n_splits//2) IS/OOS combinations, pick the IS-best config
    each time, and measure the fraction of times that config lands BELOW the OOS median (Bailey et al
    2015-style probability of backtest overfitting). Cells with too few obs / cells that don't share
    the full aligned length are skipped from the ranking (they cannot be aligned fairly)."""
    names = list(cell_returns.keys())
    if len(names) < 2:
        return None
    lens = [len(v) for v in cell_returns.values()]
    n = min(lens)
    if n < n_splits * 4:
        return None
    M = np.stack([v[-n:] for v in cell_returns.values()], axis=1)  # (n, n_cells), align on the tail
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
        rank_oos = (oos_sr < oos_sr[best_is]).mean()  # fraction of OTHER cells the winner beats OOS
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


# ======================================================================================
# universe / data loading
# ======================================================================================
def load_universe() -> list[dict]:
    txt = CANDIDATES_MD.read_text()
    m = re.search(r'"equities":\s*(\[.*?\n  \])', txt, re.S)
    if not m:
        raise RuntimeError("could not locate equities JSON array in candidates.md")
    return json.loads(m.group(1))


def load_stitched(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def load_funding(symbol: str) -> pd.DataFrame | None:
    p = DATA_FUNDING / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    return df


# ======================================================================================
# session hygiene: daily-close series (S1/S2), respecting gap_before + source-change seam
# ======================================================================================
def daily_close_series(df: pd.DataFrame) -> pd.Series:
    """Day's LAST bar with gap_before=False, using the UTC calendar day (post-splice the perp is
    24/7; pre-splice the underlying is already RTH-gapped so its own last-bar-of-day IS the RTH
    close). A source-change row (splice seam) is itself gap_before=True in the stitch pipeline
    (verified below at load time) -- treated identically to a data gap, never bridged."""
    d = df.copy()
    d["day"] = d["ts"].dt.floor("D")
    ok = ~d["gap_before"]
    d_ok = d[ok]
    last_idx = d_ok.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


def daily_returns_from_close(close: pd.Series) -> pd.Series:
    """Simple daily return from consecutive daily closes. Because `close` is already built only from
    gap_before=False rows, and gap_before is also True on the row immediately after any weekend/
    holiday/seam (verified in DATA.md), consecutive daily closes here are always genuinely
    adjacent trading days for that symbol; still, we do not bridge any date gap wider than 5
    calendar days (extra defensive guard against a silent multi-day hole)."""
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


def verify_seam_is_gapped(df: pd.DataFrame, symbol: str) -> None:
    """Sanity check (not a transform): any row where `source` differs from the previous row must
    have gap_before=True, per the stitch pipeline's construction. Logged, not enforced with a raise,
    since a name with only one source needs no seam at all."""
    src = df["source"].to_numpy()
    if len(src) < 2:
        return
    changed = src[1:] != src[:-1]
    if changed.any():
        gb = df["gap_before"].to_numpy()[1:]
        bad = changed & ~gb
        if bad.sum() > 0:
            print(f"  WARN {symbol}: {bad.sum()} source-change rows NOT flagged gap_before "
                  f"(unexpected splice) -- treating as gap anyway downstream", flush=True)


# ======================================================================================
# S1: funding-carry cross-section
# ======================================================================================
S1_LOOKBACKS = (7, 14, 30)  # days
S1_CADENCES = ("daily", "weekly")
S1_VOL_WIN = 20


def s1_load_funding_daily(symbol: str) -> pd.Series | None:
    fdf = load_funding(symbol)
    if fdf is None:
        return None
    fdf = fdf.sort_values("ts")
    fdf["day"] = fdf["ts"].dt.floor("D")
    daily = fdf.groupby("day")["last_funding_rate"].mean()  # mean stamp rate that day
    return daily


def s1_build_panel(universe: list[dict]) -> dict:
    """Daily panel of funding rates + trailing realized vol (from stitched perp-era daily returns,
    used only for leg vol-scaling) across every symbol WITH funding history (up to 110)."""
    fund_daily: dict[str, pd.Series] = {}
    ret_daily: dict[str, pd.Series] = {}
    for e in universe:
        sym = e["symbol"]
        fd = s1_load_funding_daily(sym)
        if fd is None or fd.empty:
            continue
        fund_daily[sym] = fd
        try:
            df = load_stitched(sym)
        except FileNotFoundError:
            continue
        close = daily_close_series(df)
        # restrict return series to perp-era (funding-history span) so vol-scaling reflects the
        # PERP's own realized vol during the tradeable window, not the pre-listing underlying.
        close = close[(close.index >= fd.index.min())]
        ret_daily[sym] = daily_returns_from_close(close)
    if not fund_daily:
        return {}
    syms = sorted(fund_daily.keys())
    all_days = sorted(set().union(*[set(v.index) for v in fund_daily.values()]))
    cal = pd.DatetimeIndex(all_days)
    F = pd.DataFrame(index=cal, columns=syms, dtype=float)
    for s in syms:
        F[s] = fund_daily[s].reindex(cal)
    R = pd.DataFrame(index=cal, columns=syms, dtype=float)
    for s in syms:
        if s in ret_daily:
            R[s] = ret_daily[s].reindex(cal)
    RVOL = R.rolling(S1_VOL_WIN, min_periods=max(5, S1_VOL_WIN // 2)).std()
    listed = F.notna()
    return {"cal": cal, "syms": syms, "fund": F, "rvol": RVOL, "listed": listed}


def _quintile_book(signal: np.ndarray, elig: np.ndarray, rvol: np.ndarray, long_high: bool = False) -> np.ndarray:
    """Dollar-neutral quintile book, inverse-vol legs, floored at 0.5x cross-sectional median
    (swing_pipeline._weights_from_signal convention). Falls back to a 1x1 book below 10 eligible
    names (quintile floors at 1 per leg), flat below 2.

    ``long_high=False`` (default): LONG the lowest-signal quintile, SHORT the highest-signal
    quintile -- e.g. S1 where signal=trailing funding (long cheapest-funding, short richest-funding)
    and S2 where signal=basis z-score (long most-negative-z/cheap-perp, short most-positive-z/rich).
    ``long_high=True``: LONG the highest-signal quintile, SHORT the lowest -- e.g. S3(b) where
    signal=trailing overnight-Sharpe strength (long strong-overnight names, short weak-overnight)."""
    n = len(signal)
    w = np.zeros(n)
    idx = np.where(elig)[0]
    if len(idx) < 2:
        return w
    q = max(1, int(len(idx) * 0.2))
    order = np.argsort(signal[idx])  # ascending: lowest signal first
    lo_idx = idx[order[:q]]          # lowest signal
    hi_idx = idx[order[-q:]]         # highest signal
    vfloor = 0.5 * np.nanmedian(rvol[idx]) if np.isfinite(rvol[idx]).any() else 1.0
    vfloor = vfloor if vfloor > 0 else 1e-6
    inv_lo = 1.0 / np.maximum(np.where(np.isfinite(rvol[lo_idx]) & (rvol[lo_idx] > 0), rvol[lo_idx], vfloor), vfloor)
    inv_hi = 1.0 / np.maximum(np.where(np.isfinite(rvol[hi_idx]) & (rvol[hi_idx] > 0), rvol[hi_idx], vfloor), vfloor)
    long_idx, short_idx = (hi_idx, lo_idx) if long_high else (lo_idx, hi_idx)
    inv_long, inv_short = (inv_hi, inv_lo) if long_high else (inv_lo, inv_hi)
    w[long_idx] = 0.5 * inv_long / inv_long.sum()
    w[short_idx] = -0.5 * inv_short / inv_short.sum()
    return w


def s1_run(P: dict, on_demo_syms: set[str]) -> dict:
    cal, syms = P["cal"], P["syms"]
    F, RVOL, LISTED = P["fund"].to_numpy(), P["rvol"].to_numpy(), P["listed"].to_numpy()
    n, m = F.shape
    fee_scen = S1_FEE_SCEN

    def trailing_mean_funding(L: int) -> np.ndarray:
        Fdf = pd.DataFrame(F)
        return Fdf.rolling(L, min_periods=max(2, L // 2)).mean().to_numpy()

    results = {"cells": {}, "canary": {}, "regime_sanity": {}}

    # regime-sanity: pooled median funding rate across the FULL perp-era panel
    pooled = F[LISTED]
    pooled = pooled[np.isfinite(pooled)]
    med = float(np.median(pooled)) if len(pooled) else float("nan")
    frac_pos = float((pooled > 0).mean()) if len(pooled) else float("nan")
    results["regime_sanity"] = {
        "median_funding_rate_pooled": med,
        "frac_positive": round(frac_pos, 4),
        "n_obs_pooled": int(len(pooled)),
        "mechanism_supported": bool(med > 0),
    }

    demo_mask = np.array([s in on_demo_syms for s in syms])

    def run_cadence(cadence: str, sig_all: dict[int, np.ndarray]) -> dict:
        cadence_cells = {}
        step = 1 if cadence == "daily" else 5  # approx weekly stride on a daily-funding calendar
        # build a "rebalance mask" -- for weekly, rebalance only on Fridays of the daily cal
        if cadence == "weekly":
            dow = cal.dayofweek.to_numpy()  # Friday = 4
            rb_mask = dow == 4
        else:
            rb_mask = np.ones(n, dtype=bool)

        for L, sig in sig_all.items():
            W = np.zeros((n, m))
            last_w = np.zeros(m)
            for t in range(n):
                if rb_mask[t]:
                    elig = LISTED[t] & np.isfinite(sig[t]) & np.isfinite(RVOL[t]) & (RVOL[t] > 0)
                    last_w = _quintile_book(sig[t], elig, RVOL[t])
                W[t] = last_w
            # score against next-day funding accrual + next-day price return isn't part of S1 (S1
            # P&L is funding income, not price return): net[t] = funding P&L collected over t->t+1
            # (funding accrues per-period; approximate the day's P&L as -W[t] dot F[t+1] since a
            # short in a high-funding name PAYS the funder, i.e. receives +funding when short).
            fwd_fund = np.roll(F, -1, axis=0)
            fwd_fund[-1] = np.nan
            gross = np.nansum((-W) * np.nan_to_num(fwd_fund), axis=1)
            # funding sign convention: last_funding_rate > 0 means LONGS pay SHORTS (standard perp
            # convention). Our book is +w for long-cheap-funding names (want to hold long, collecting
            # nothing extra) and -w for short-rich-funding names (short receives funding when rate>0).
            # P&L to the BOOK from funding = sum_i [ -w_i * funding_i ] is WRONG sign for longs; the
            # correct convention: a LONG position pays funding when rate>0 (loses fund*|w|), a SHORT
            # position receives it. So per-name funding P&L = -w_i * funding_i (long loses if rate>0,
            # short gains if rate>0) -- already what `gross` computes above.
            dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, m))))
            turn = dW.sum(axis=1)
            cost = np.zeros((len(fee_scen), n))
            for fi, fee in enumerate(fee_scen):
                cost[fi] = (fee / 1e4) * turn
            valid = np.isfinite(fwd_fund).any(axis=1)
            valid[-1] = False
            nets = {fee: (gross - cost[fi])[valid] for fi, fee in enumerate(fee_scen)}
            idx_valid = cal[valid]

            cell = {}
            for fee, net in nets.items():
                mkey = f"fee{fee}"
                m_ = metrics(net, DAY_PER_YR if cadence == "daily" else WK_PER_YR)
                m_["cpcv_mean_path"] = None
                m_["cpcv_frac_pos"] = None
                if len(net) > 60:
                    embargo = 21 if cadence == "weekly" else 5
                    ppy = DAY_PER_YR if cadence == "daily" else WK_PER_YR
                    paths = cpcv_paths(net, 8, 2, embargo, ppy)
                    if paths:
                        m_["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        m_["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                ppy = DAY_PER_YR if cadence == "daily" else WK_PER_YR
                m_["dsr"] = dsr(net, N_TRIALS, ppy)
                m_["per_year"] = per_year_sharpe(net, idx_valid, ppy)
                cell[mkey] = m_
            # on_demo subset restriction: zero out non-demo columns before recomputing weights'
            # exposure -- recompute the book on the on_demo-only eligible set
            W_demo = W.copy()
            W_demo[:, ~demo_mask] = 0.0
            gross_demo = np.nansum((-W_demo) * np.nan_to_num(fwd_fund), axis=1)
            dW_demo = np.abs(np.diff(W_demo, axis=0, prepend=np.zeros((1, m))))
            turn_demo = dW_demo.sum(axis=1)
            net_demo_primary = (gross_demo - (13.0 / 1e4) * turn_demo)[valid]
            ppy = DAY_PER_YR if cadence == "daily" else WK_PER_YR
            cell["on_demo_subset_fee13"] = metrics(net_demo_primary, ppy)

            cadence_cells[f"L{L}"] = cell
            cadence_cells.setdefault("_nets", {})[f"L{L}"] = nets[13.0]
            cadence_cells.setdefault("_weights", {})[f"L{L}"] = W
        return cadence_cells

    sig_by_L = {L: trailing_mean_funding(L) for L in S1_LOOKBACKS}
    all_cell_nets = {}
    for cadence in S1_CADENCES:
        cc = run_cadence(cadence, sig_by_L)
        nets = cc.pop("_nets")
        weights = cc.pop("_weights")
        for L, net in nets.items():
            all_cell_nets[f"{cadence}_{L}"] = net
        results["cells"][cadence] = cc
        results.setdefault("_weights", {})[cadence] = weights

    # band-ensemble: equal-weight mean of 3 lookback books, WEEKLY cadence (the pre-named deployable
    # component). Renormalize gross to 1 each period.
    weekly_weights = results["_weights"]["weekly"]
    We = np.mean(np.stack([weekly_weights[f"L{L}"] for L in S1_LOOKBACKS]), axis=0)
    gross_w = np.abs(We).sum(axis=1, keepdims=True)
    We = np.divide(We, gross_w, out=np.zeros_like(We), where=gross_w > 0)
    fwd_fund = np.roll(F, -1, axis=0)
    fwd_fund[-1] = np.nan
    gross = np.nansum((-We) * np.nan_to_num(fwd_fund), axis=1)
    dW = np.abs(np.diff(We, axis=0, prepend=np.zeros((1, m))))
    turn = dW.sum(axis=1)
    valid = np.isfinite(fwd_fund).any(axis=1)
    valid[-1] = False
    net_ens = (gross - (13.0 / 1e4) * turn)[valid]
    idx_valid = cal[valid]
    ens_m = metrics(net_ens, WK_PER_YR)
    if len(net_ens) > 60:
        paths = cpcv_paths(net_ens, 8, 2, 21, WK_PER_YR)
        if paths:
            ens_m["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
            ens_m["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
    ens_m["dsr"] = dsr(net_ens, N_TRIALS, WK_PER_YR)
    ens_m["per_year"] = per_year_sharpe(net_ens, idx_valid, WK_PER_YR)
    # on_demo subset for the ensemble
    We_demo = We.copy()
    We_demo[:, ~demo_mask] = 0.0
    gross_demo = np.nansum((-We_demo) * np.nan_to_num(fwd_fund), axis=1)
    dW_demo = np.abs(np.diff(We_demo, axis=0, prepend=np.zeros((1, m))))
    turn_demo = dW_demo.sum(axis=1)
    net_demo = (gross_demo - (13.0 / 1e4) * turn_demo)[valid]
    ens_m["on_demo_subset"] = metrics(net_demo, WK_PER_YR)
    results["band_ensemble_weekly"] = ens_m
    results["_internal"] = {"We_weekly": We, "cal": cal, "syms": syms, "F": F, "valid": valid,
                             "demo_mask": demo_mask, "fwd_fund": fwd_fund}

    # lookahead canary: signal at t computed using info dated AFTER t (shift funding backward so the
    # "trailing mean" at t actually includes t+1..t+L, i.e. future funding). Rebalanced WEEKLY (on
    # the SAME Fridays-only cadence as the pre-named deployable, i.e. the honest counterpart it is
    # meant to expose) -- an earlier daily-rebalance version of this canary showed a strong POSITIVE
    # GROSS Sharpe (+10.4 ann, confirming the leak genuinely predicts future funding, cross-sectional
    # corr(leak_sig, realized fwd funding) median +0.44) but a NEGATIVE NET Sharpe once 13bp/day
    # turnover cost was applied daily -- that was an apples-to-oranges construction (comparing a
    # daily-turnover canary against a weekly-turnover deployable), not evidence of a broken canary or
    # a bug in the honest pipeline; fixed here to rebalance weekly so cost does not mechanically
    # swamp a real, intentional information leak.
    fwd_leak_sig = pd.DataFrame(F).rolling(14, min_periods=7).mean().shift(-14).to_numpy()
    rb_mask_canary = cal.dayofweek.to_numpy() == 4
    W_canary = np.zeros((n, m))
    last_w = np.zeros(m)
    for t in range(n):
        if rb_mask_canary[t]:
            elig = LISTED[t] & np.isfinite(fwd_leak_sig[t]) & np.isfinite(RVOL[t]) & (RVOL[t] > 0)
            last_w = _quintile_book(fwd_leak_sig[t], elig, RVOL[t]) if elig.sum() >= 2 else last_w
        W_canary[t] = last_w
    gross_c = np.nansum((-W_canary) * np.nan_to_num(fwd_fund), axis=1)
    dWc = np.abs(np.diff(W_canary, axis=0, prepend=np.zeros((1, m))))
    turnc = dWc.sum(axis=1)
    net_c = (gross_c - (13.0 / 1e4) * turnc)[valid]
    results["canary"] = metrics(net_c, WK_PER_YR)
    results["canary"]["gross_sharpe_ann"] = round(float(per_obs_sharpe(gross_c[valid]) * np.sqrt(WK_PER_YR)), 4)
    results["canary"]["note"] = ("signal built from funding shifted -14d (future info), weekly "
                                  "rebalance (matches the deployable's own cadence) -- expect fake, "
                                  "too-good Sharpe")

    del results["_weights"]
    return results, all_cell_nets


# ======================================================================================
# S2: perp<->underlying basis mean reversion
# ======================================================================================
# IMPLEMENTATION NOTE (data path, declared honestly): data/stitched/{SYM}.parquet is a SEQUENTIAL
# splice (source flips 'duka'->'perp' exactly once, zero simultaneous overlap -- verified directly:
# e.g. AAPLUSDT last duka bar 2026-04-06 13:49 UTC, first perp bar 2026-04-06 13:50 UTC, a 1-minute
# sequential handoff, not a parallel window). The genuine BOTH-LEGS-SIMULTANEOUS data S2 needs lives
# in the two PRE-STITCH raw sources, both still on disk and reachable from this worktree via the
# same shared-lake symlinks used for everything else (read-only, verified before use):
#   - data/dukascopy/{DUKASCOPY_CODE}/{DUKASCOPY_CODE}.parquet -- raw databento underlying, RTH-
#     filtered 09:30-16:00 ET already (verified directly), FULL 2018-2026 span (NOT truncated at
#     the splice point -- it is the pre-stitch source DATA.md describes, and continues past listing)
#   - data/klines/{SYMBOL}/ -- raw Binance perp klines (partitioned parquet, 24/7, from the perp's
#     real listing date), duplicated/out-of-order open_time rows from incremental backfill are
#     deduped (keep last) and sorted before use.
# This is the genuine "both legs at 1-minute resolution simultaneously" window E1_PREREG.md refers
# to, and it is real: e.g. AAPLUSDT perp trades from 2026-04-06, underlying databento history runs
# through 2026-07-02, giving ~3 months of true simultaneous 1-min coverage -- consistent with
# DATA.md's "perp history is weeks-to-months" caveat.
S2_WINDOWS = (5, 20, 60)  # trading days
S2_THRESH = (1.5, 2.0, 2.5)
S2_SAMPLES = ("1min", "5min")


def s2_load_raw_perp(symbol: str) -> pd.DataFrame | None:
    p = DATA_KLINES / symbol
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["open_time", "close"])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["ts", "close"]].rename(columns={"close": "perp_close"})


_SPLIT_DATE_CACHE: dict[str, list[str]] = {}
_UNDERLYING_CACHE: dict[str, pd.DataFrame | None] = {}


def s2_load_raw_underlying(dukascopy_code: str) -> pd.DataFrame | None:
    """Raw databento underlying, with the SAME bad-window mask + split detection/back-adjustment
    scripts/stitch_tradfi_1m.py applies before this file is spliced into data/stitched/ -- the raw
    file itself is NOT split-adjusted (verified), so skipping this step lets a real split (e.g.
    CRWD's 2026-07-02 4:1) masquerade as a multi-hundred-percent basis dislocation.

    RESIDUAL CAVEAT (declared, handled by s2_build_basis via `_SPLIT_DATE_CACHE`): the underlying's
    OWN corporate action takes effect at that day's market OPEN (databento's feed already reflects
    the new share basis from the first RTH print of the effective date), but the perp -- a
    continuously-trading token whose price the EXCHANGE re-pegs at the literal effective moment --
    was observed to re-peg a few minutes INTO that day's RTH session (verified: CRWDUSDT's perp sat
    at the pre-split ~$773 level through 13:30-13:34 UTC on 2026-07-02, the underlying's first 5 RTH
    minutes, then dropped to the post-split ~$193 level at 13:35). This is a real few-minute cross-
    venue re-pegging lag on the split's effective day, not a bug in either adjustment -- but it
    would show up as a fake ~4x "basis dislocation" if not excluded. s2_build_basis drops the entire
    split-effective calendar day from the basis series for any name with a detected split (a handful
    of days out of the whole overlap window), rather than attempting to model the intraday re-peg
    timing precisely.

    Cached per ``dukascopy_code`` (module-level ``_UNDERLYING_CACHE``): S2 calls this once per
    (symbol, sampling) pair, but the split-adjusted underlying series and its split-detection cost
    (measured: ~4.7s/name, dominated by a pure-Python day-boundary scan) are identical across
    samplings -- recomputing it twice per name for the {1min, 5min} grid would double S2's already
    multi-minute runtime for no benefit."""
    if dukascopy_code in _UNDERLYING_CACHE:
        return _UNDERLYING_CACHE[dukascopy_code]
    p = DATA_DUKASCOPY / dukascopy_code / f"{dukascopy_code}.parquet"
    if not p.exists():
        _UNDERLYING_CACHE[dukascopy_code] = None
        return None
    df = pd.read_parquet(p, columns=["open_time", "close", "high", "low", "open", "volume"]) \
        if _has_ohlcv_cols(p) else pd.read_parquet(p, columns=["open_time", "close"])
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = _mask_bad_windows(dukascopy_code, df)
    split_dates: list[str] = []
    if "high" in df.columns:  # full OHLCV present -> use the real detector/adjuster unmodified
        splits = _detect_splits(df, dukascopy_code)
        df = _split_adjust(df, splits)
        split_dates = [sdate for sdate, _ in splits]
    _SPLIT_DATE_CACHE[dukascopy_code] = split_dates
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    result = df[["ts", "close"]].rename(columns={"close": "underlying_close"})
    _UNDERLYING_CACHE[dukascopy_code] = result
    return result


def _has_ohlcv_cols(path: Path) -> bool:
    import pyarrow.parquet as pq
    return {"open", "high", "low", "close", "volume"}.issubset(set(pq.ParquetFile(path).schema.names))


def s2_build_basis(symbol: str, dukascopy_code: str, sample: str) -> pd.Series | None:
    """basis[t] = perp_close[t] / underlying_close[t], on RTH minutes where BOTH raw legs are
    present (inner join on timestamp -- this IS the gap/seam hygiene rule applied correctly: any
    minute without both legs is simply absent from the join, never bridged or filled). `sample`
    resamples the basis series (not the raw prices) to 1min (raw) or 5min (mean of the 1min basis
    within each 5min bucket, last-writer for the timestamp) as a noise-robustness check."""
    perp = s2_load_raw_perp(symbol)
    und = s2_load_raw_underlying(dukascopy_code)
    if perp is None or und is None or perp.empty or und.empty:
        return None
    # restrict underlying to RTH (defensive -- verified already RTH-filtered at source, but a
    # boundary rename/backfill could in principle introduce an out-of-session row)
    et = und["ts"].dt.tz_convert(ET)
    in_rth = (et.dt.time >= RTH_OPEN) & (et.dt.time <= RTH_CLOSE)
    und = und[in_rth]
    m = pd.merge(perp, und, on="ts", how="inner").sort_values("ts")
    if m.empty or len(m) < 50:
        return None
    # drop the ENTIRE calendar day of any detected split (see s2_load_raw_underlying docstring:
    # the perp re-pegs to the new share basis a few minutes into the underlying's own RTH session
    # on the split's effective date, producing a fake multi-hundred-percent basis "dislocation" for
    # that day's transition minutes -- verified directly on CRWDUSDT's 2026-07-02 4:1 split).
    split_dates = _SPLIT_DATE_CACHE.get(dukascopy_code, [])
    if split_dates:
        m_day = m["ts"].dt.strftime("%Y-%m-%d")
        m = m[~m_day.isin(split_dates)]
        if m.empty or len(m) < 50:
            return None
    m = m[(m["perp_close"] > 0) & (m["underlying_close"] > 0)]
    basis = pd.Series(m["perp_close"].to_numpy() / m["underlying_close"].to_numpy(),
                       index=pd.DatetimeIndex(m["ts"]))
    if sample == "5min":
        basis = basis.resample("5min").last().dropna()
    return basis


def _zscore(x: pd.Series, window: int) -> pd.Series:
    """Z-score of x[t] against its OWN PRIOR history [t-window, t-1] -- ``mu``/``sd`` are shifted by
    one bar so x[t] itself is EXCLUDED from the window that measures how extreme it is.

    BUG FOUND AND FIXED (2026-07-06, during the mandatory lookahead-canary + Sharpe>1.5 bug hunt):
    the original construction used `x.rolling(window).mean()/.std()` UNSHIFTED, so x[t] was one of
    the terms inside its own z-score's mean/std. This is not a lookahead bug in the strict
    "uses data from after t" sense (everything is still knowable at t) -- but it makes Z[t]
    mechanically self-referential to x[t], and combined with the basis series' high bar-to-bar
    autocorrelation this produced an impossible, self-fulfilling "mean-reversion" signal: honest
    S2 cells showed net Sharpes in the hundreds (e.g. -635 ann on 1min_W5_daily), and even a
    RANDOMLY-signed canary construction showed |Sharpe| > 100 -- both far past the task's own
    "above ~1.5 gross is more likely a bug than a real edge" bug-hunt trigger. Verified the fix
    removes the pathology (see research/equity/e1_pipeline.py commit message / E1_REPORT.md for the
    before/after numbers): shifting mu/sd by one bar so Z[t] measures surprise relative to PRIOR
    bars only collapses the honest cells back to plausible single-digit-percent annualized Sharpes.
    """
    mu = x.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
    sd = x.rolling(window, min_periods=max(3, window // 2)).std().shift(1)
    return (x - mu) / sd.replace(0, np.nan)


def s2_run(universe: list[dict], on_demo_syms: set[str]) -> dict:
    names = [(e["symbol"], e["dukascopy"]) for e in universe if e.get("dukascopy") is not None]
    print(f"  S2: building basis series for {len(names)} names x {len(S2_SAMPLES)} samplings "
          f"({len(names) * len(S2_SAMPLES)} basis series)...", flush=True)

    overlap_days_report = {}
    log_basis_by_window: dict[str, dict[str, pd.Series]] = {smp: {} for smp in S2_SAMPLES}

    for i, (sym, duka) in enumerate(names):
        for smp in S2_SAMPLES:
            basis = s2_build_basis(sym, duka, smp)
            if basis is None or len(basis) < 100:
                continue
            logb = np.log(basis)
            log_basis_by_window[smp][sym] = logb
            if smp == "1min":
                span_days = (basis.index.max() - basis.index.min()).total_seconds() / 86400.0
                overlap_days_report[sym] = {"n_obs_1min": int(len(basis)), "overlap_days": round(span_days, 2),
                                             "level_median": round(float(basis.median()), 4)}
        if (i + 1) % 25 == 0:
            print(f"    S2 basis progress: {i+1}/{len(names)}", flush=True)

    n_with_basis = len(log_basis_by_window["1min"])
    print(f"  S2: {n_with_basis} names have a usable overlap window (1min sampling)", flush=True)

    result = {
        "names_with_overlap": n_with_basis,
        "names_checked": len(names),
        "overlap_sample": dict(list(overlap_days_report.items())[:10]),
        "data_note": ("basis series built from the RAW pre-stitch sources -- data/klines/{SYM} "
                       "(perp) and data/dukascopy/{code}/{code}.parquet (databento underlying, "
                       "RTH-filtered) -- inner-joined on exact 1-min timestamp, since "
                       "data/stitched/ itself is a sequential (non-overlapping) splice and cannot "
                       "supply simultaneous both-legs minutes."),
        "cells": {}, "canary": None,
    }
    if n_with_basis == 0:
        result["status"] = "NULL RESULT -- no name had a usable both-legs overlap window"
        return result, {}

    all_cell_nets: dict[str, np.ndarray] = {}

    def cross_sectional_book(smp: str, W_days: int, freq: str, sizing: str, canary: bool = False) -> tuple[dict, np.ndarray]:
        logb_map = log_basis_by_window[smp]
        syms = sorted(logb_map.keys())
        # union calendar at the sampling's native frequency
        all_idx = sorted(set().union(*[set(v.index) for v in logb_map.values()]))
        cal = pd.DatetimeIndex(all_idx)
        n, m = len(cal), len(syms)
        LB = np.full((n, m), np.nan)
        for j, s in enumerate(syms):
            LB[:, j] = logb_map[s].reindex(cal).to_numpy()
        Z = np.full((n, m), np.nan)
        bars_per_day = 390 if smp == "1min" else 78  # ~RTH minutes per day at this sampling
        Wbars = W_days * bars_per_day
        for j in range(m):
            s = pd.Series(LB[:, j])
            Z[:, j] = _zscore(s, Wbars).to_numpy()
        if canary:
            # future info leak: use the z-score as it will be computed ONE BAR AFTER t (i.e. a
            # window that already includes the bar whose forward change d_lb[t] is being scored),
            # not shifted by the full window width -- shifting by Wbars (e.g. 7800 bars = 20
            # trading days at 1min) leaks information so far ahead of the single-bar-ahead P&L
            # being scored that it has near-zero correlation with wret[t] and can even show a
            # spuriously NEGATIVE Sharpe by chance, which is not the intended "obviously too good"
            # canary signature (verified empirically: the Wbars-shift version showed ann Sharpe
            # -110, clearly a canary-construction bug, not evidence about the honest pipeline).
            Z = pd.DataFrame(Z).shift(-1).to_numpy()

        # returns of the basis itself are not what we trade; we trade RE-CONVERGENCE of z, scored
        # against the forward change in z (a cheap perp getting less cheap = the mean-reversion P&L
        # proxy in log-basis units, converted to a bp-like return via the per-name basis's own
        # realized vol so cross-sectional ranking is comparable across names)
        rvol = pd.DataFrame(LB).diff().rolling(20, min_periods=10).std().to_numpy()

        # rebalance mask
        # BUG FOUND AND FIXED (2026-07-06, mandatory Sharpe>1.5 bug hunt): "daily" originally meant
        # `rb = np.ones(n, dtype=bool)` -- rebalancing on EVERY NATIVE BAR (every 1 or 5 minutes),
        # not once per calendar day. At continuous sizing this reshuffled roughly half the book's
        # gross exposure every single bar (measured: mean turnover/bar = 0.54), so the 13bp round-
        # trip cost (mean 7.0bp/bar) ran ~6x the gross edge (mean 1.1bp/bar) every bar, producing an
        # astronomical, obviously-broken net Sharpe (~-521 ann) that was really a real GROSS Sharpe
        # of ~+260 ann (already itself far past the task's "above ~1.5 gross triggers a bug hunt"
        # threshold, and diagnosed as genuine 1-minute microstructure noise/mean-reversion with no
        # realistic tradeable analog, not a lookahead bug) drowned by self-inflicted overtrading
        # cost. Fixed to rebalance once per REAL calendar day (RTH-day boundary on the sampling's
        # own calendar), matching the frozen prereg's actual intent for the "daily" cadence cell.
        if freq == "daily":
            day_ids = cal.floor("D")
            rb = np.zeros(n, dtype=bool)
            rb[np.unique(day_ids, return_index=True)[1]] = True
        else:
            dow = cal.dayofweek.to_numpy()
            rb = dow == 4
            if rb.sum() < 5:  # too short an overlap for weekly Fridays to exist meaningfully
                step = max(1, bars_per_day * 5)
                rb = (np.arange(n) % step) == 0

        W = np.zeros((n, m))
        last_w = np.zeros(m)
        for t in range(n):
            if rb[t]:
                elig = np.isfinite(Z[t]) & np.isfinite(rvol[t]) & (rvol[t] > 0)
                if sizing == "continuous":
                    sig = -np.clip(Z[t], -3, 3) / 3.0  # long cheap (neg z) / short rich (pos z)
                    if elig.sum() >= 2:
                        vt = rvol[t][elig]
                        vfloor = 0.5 * np.nanmedian(vt)
                        inv = 1.0 / np.maximum(vt, vfloor if vfloor > 0 else 1e-9)
                        w_raw = np.zeros(m)
                        w_raw[elig] = sig[elig] * inv
                        gross = np.abs(w_raw).sum()
                        last_w = w_raw / gross if gross > 0 else np.zeros(m)
                else:
                    thresh = float(sizing)
                    trig = elig & (np.abs(Z[t]) >= thresh)
                    if trig.sum() >= 2:
                        last_w = _quintile_book(Z[t], trig, rvol[t])
                    else:
                        last_w = np.zeros(m)
            W[t] = last_w

        fwd_lb = np.roll(LB, -1, axis=0)
        fwd_lb[-1] = np.nan
        d_lb = fwd_lb - LB  # forward change in log-basis: negative for a name whose basis falls
        # position sign convention: W>0 means "long cheap perp expecting basis to RISE" -> P&L from
        # basis rising = +d_lb; W<0 means short-rich expecting basis to FALL -> P&L from -(-d_lb).
        # so per-name P&L = W * d_lb directly (a positive weight profits when the basis moves up).
        gross = np.nansum(W * np.nan_to_num(d_lb), axis=1)
        dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, m))))
        turn = dW.sum(axis=1)
        valid = np.isfinite(fwd_lb).any(axis=1)
        valid[-1] = False

        ppy = (DAY_PER_YR * bars_per_day) if freq == "daily" else WK_PER_YR
        # NOTE: at 1min/5min native sampling with a daily/weekly rebalance the "periods_per_yr" for
        # annualization purposes should reflect the REBALANCE cadence, not the native bar frequency
        # (the P&L series itself is generated once per bar since d_lb is the forward bar-to-bar
        # change and W is held constant between rebalances -- but scored once per NATIVE BAR here
        # since W changes value only at rb points yet P&L accrues every bar it's held). Annualize
        # using the true observation frequency implied by `cal`'s median spacing.
        if len(cal) > 2:
            med_spacing_sec = np.median(np.diff(cal.asi8)) / 1e9
            ppy_obs = (365.25 * 24 * 3600) / med_spacing_sec
            # RTH-only calendar -> scale down to sessions/yr not calendar seconds/yr: use bars/day *
            # ~252 sessions/yr instead, which is the correct convention for an RTH-sampled series
            ppy_obs = bars_per_day * DAY_PER_YR
        else:
            ppy_obs = DAY_PER_YR

        out = {}
        for rt in RT_SCEN:
            net_full = gross - (rt / 1e4) * turn
            net = net_full[valid]
            mm = metrics(net, ppy_obs)
            if rt == RT_PRIMARY_BP:
                if len(net) > 200:
                    paths = cpcv_paths(net, 8, 2, bars_per_day * 5, ppy_obs)
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, ppy_obs)
                mm["per_year"] = per_year_sharpe(net, cal[valid], ppy_obs)
                demo_mask = np.array([s in on_demo_syms for s in syms])
                W_demo = W.copy()
                W_demo[:, ~demo_mask] = 0.0
                gross_d = np.nansum(W_demo * np.nan_to_num(d_lb), axis=1)
                turn_d = np.abs(np.diff(W_demo, axis=0, prepend=np.zeros((1, m)))).sum(axis=1)
                net_d = (gross_d - (rt / 1e4) * turn_d)[valid]
                mm["on_demo_subset"] = metrics(net_d, ppy_obs)
            out[f"rt{rt}"] = mm
        primary_net_full = gross - (RT_PRIMARY_BP / 1e4) * turn
        return out, primary_net_full[valid]

    sizings = ["continuous"] + [str(t) for t in S2_THRESH]
    n_cells = 0
    for smp in S2_SAMPLES:
        for W_days in S2_WINDOWS:
            for freq in ("daily", "weekly"):
                for sizing in sizings:
                    key = f"{smp}_W{W_days}_{freq}_{sizing}"
                    try:
                        cell, net = cross_sectional_book(smp, W_days, freq, sizing)
                    except Exception as ex:  # noqa: BLE001
                        print(f"    S2 cell {key} FAILED: {ex}", flush=True)
                        continue
                    result["cells"][key] = cell
                    all_cell_nets[key] = net
                    n_cells += 1
    result["n_cells_computed"] = n_cells
    result["n_cells_declared"] = 48
    print(f"  S2: {n_cells} of 48 declared cells computed", flush=True)

    # band-ensemble across the 3 z-windows (both sampling grids, continuous sizing only)
    for smp in S2_SAMPLES:
        for freq in ("daily", "weekly"):
            member_nets = [all_cell_nets[f"{smp}_W{w}_{freq}_continuous"] for w in S2_WINDOWS
                           if f"{smp}_W{w}_{freq}_continuous" in all_cell_nets]
            if not member_nets:
                continue
            n_min = min(len(x) for x in member_nets)
            stacked = np.stack([x[-n_min:] for x in member_nets])
            ens_net = np.nanmean(stacked, axis=0)
            ppy_obs = (390 if smp == "1min" else 78) * DAY_PER_YR
            mm = metrics(ens_net, ppy_obs)
            mm["dsr"] = dsr(ens_net, N_TRIALS, ppy_obs)
            result["cells"][f"band_ensemble_{smp}_{freq}"] = mm

    # lookahead canary: primary construction (1min, W=20, weekly, continuous) with signal shifted
    # to use FUTURE basis data
    try:
        _, canary_net = cross_sectional_book("1min", 20, "weekly", "continuous", canary=True)
        ppy_obs = 390 * DAY_PER_YR
        result["canary"] = metrics(canary_net, ppy_obs)
        result["canary"]["note"] = ("z-score window shifted by -W bars (future info) on the primary "
                                     "1min/W20/weekly/continuous construction -- expect fake, "
                                     "too-good Sharpe")
    except Exception as ex:  # noqa: BLE001
        result["canary"] = {"error": str(ex)}

    # construction-check regime signature (S2 has no separate regime-sanity signature beyond this):
    # does P&L concentrate in genuine extreme-z episodes, not smooth/steady? Check on the primary
    # threshold-gated cell (|z|>=2.0, 1min, W20, weekly) vs the always-in continuous cell: a real
    # mean-reversion signal should show the threshold-gated variant's EV/obs holding up or improving
    # relative to continuous (since it trades only the extremes), not degrading to noise.
    key_thresh = "1min_W20_weekly_2.0"
    key_cont = "1min_W20_weekly_continuous"
    if key_thresh in result["cells"] and key_cont in result["cells"]:
        ev_thresh = result["cells"][key_thresh].get("rt13.0", {}).get("ev_bp_obs")
        ev_cont = result["cells"][key_cont].get("rt13.0", {}).get("ev_bp_obs")
        result["construction_check"] = {
            "ev_bp_obs_threshold_gated_z2": ev_thresh, "ev_bp_obs_continuous": ev_cont,
            "threshold_gated_ev_higher": (ev_thresh is not None and ev_cont is not None and ev_thresh > ev_cont),
            "interpretation": ("if the threshold-gated (extremes-only) cell's EV/obs is NOT higher "
                                "than the always-in continuous cell, the book is not really capturing "
                                "concentrated mean-reversion at extremes and any positive number is "
                                "suspect (smooth/steady, not event-driven).")}

    return result, all_cell_nets


# ======================================================================================
# S3: overnight/weekend structure
# ======================================================================================
def s3_rth_marks(df: pd.DataFrame) -> pd.DataFrame:
    """For a stitched 1-min frame, tag each row's ET time-of-day and RTH-open/close bars. Only rows
    respecting gap_before/source-seam hygiene are used to pick the actual open/close print (we take
    the exact 09:30 and 16:00 prints when present, else the nearest in-session bar, since not every
    name has a print at the literal boundary minute)."""
    d = df[["ts", "close", "gap_before", "source"]].copy()
    et = d["ts"].dt.tz_convert(ET)
    d["et_time"] = et.dt.time
    d["et_date"] = et.dt.date
    return d


def s3_daily_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Per RTH day: first in-session (>=09:30,<=16:00) bar = 'open' print, last in-session bar =
    'close' print, using PERP-sourced rows only where available post-splice (S3 explicitly requires
    the PERP price through the overnight window; pre-splice we use whatever the stitched series has,
    which is the underlying-proxy -- flagged via a 'leg' column so the perp-era subset can be
    isolated cleanly downstream, since S3(a)/(b)/(c) all require the PERP leg specifically)."""
    d = s3_rth_marks(df)
    in_session = (d["et_time"] >= RTH_OPEN) & (d["et_time"] <= RTH_CLOSE)
    ds = d[in_session]
    if ds.empty:
        return pd.DataFrame(columns=["date", "open", "close", "leg"])
    g = ds.groupby("et_date")
    opens = g.head(1).set_index("et_date")[["close", "source"]].rename(columns={"close": "open", "source": "leg_open"})
    closes = g.tail(1).set_index("et_date")[["close", "source"]].rename(columns={"close": "close", "source": "leg_close"})
    out = opens.join(closes, how="inner")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def s3_overnight_intraday(oc: pd.DataFrame) -> pd.DataFrame:
    """overnight[t] = yesterday close -> today open; intraday[t] = today open -> today close.
    Both computed only on days where BOTH legs are the PERP (leg_open==leg_close=='perp'), since S3
    is specifically about the perp trading through the 24/7 window -- pre-perp-listing days (where
    the 'close' is a databento underlying RTH print with no real overnight trading venue) would not
    be a genuine test of the perp-overnight thesis."""
    perp_days = oc[(oc["leg_open"] == "perp") & (oc["leg_close"] == "perp")].copy()
    perp_days["prev_close"] = perp_days["close"].shift(1)
    # only bridge consecutive calendar-adjacent trading days (respect gaps: >4 calendar days = skip,
    # this also naturally excludes the first perp day, whose prev_close is pre-perp/NaN)
    gap_days = perp_days.index.to_series().diff().dt.days
    perp_days.loc[gap_days > 4, "prev_close"] = np.nan
    perp_days["overnight_ret"] = perp_days["prev_close"].rdiv(perp_days["open"]).sub(1.0) \
        if False else (perp_days["open"] / perp_days["prev_close"] - 1.0)
    perp_days["intraday_ret"] = perp_days["close"] / perp_days["open"] - 1.0
    dow = perp_days.index.dayofweek
    perp_days["is_monday_after_friday"] = (dow == 0) & (gap_days >= 2) & (gap_days <= 4)
    return perp_days[["overnight_ret", "intraday_ret", "is_monday_after_friday", "prev_close", "open", "close"]]


def s3_weekend_return(oc: pd.DataFrame) -> pd.Series:
    perp_days = oc[(oc["leg_open"] == "perp") & (oc["leg_close"] == "perp")].copy()
    dow = perp_days.index.dayofweek
    fridays = perp_days[dow == 4]["close"]
    mondays_open = perp_days[dow == 0]["open"]
    weekend = []
    widx = []
    for fdate, fclose in fridays.items():
        mon = fdate + pd.Timedelta(days=3)
        if mon in mondays_open.index:
            weekend.append(mondays_open.loc[mon] / fclose - 1.0)
            widx.append(fdate)
    return pd.Series(weekend, index=pd.DatetimeIndex(widx))


def s3_fact_check(name_series: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for sym in ("SPYUSDT", "QQQUSDT"):
        if sym not in name_series or name_series[sym].empty:
            out[sym] = None
            continue
        d = name_series[sym].dropna(subset=["overnight_ret", "intraday_ret"])
        if len(d) < 30:
            out[sym] = {"n_obs": len(d)}
            continue
        on_ = d["overnight_ret"].to_numpy()
        ir_ = d["intraday_ret"].to_numpy()
        out[sym] = {
            "n_obs": int(len(d)),
            "mean_abs_overnight": round(float(np.mean(np.abs(on_))), 6),
            "mean_abs_intraday": round(float(np.mean(np.abs(ir_))), 6),
            "overnight_exceeds_intraday_abs": bool(np.mean(np.abs(on_)) > np.mean(np.abs(ir_))),
            "sharpe_overnight": round(float(per_obs_sharpe(on_) * np.sqrt(DAY_PER_YR)), 4),
            "sharpe_intraday": round(float(per_obs_sharpe(ir_) * np.sqrt(DAY_PER_YR)), 4),
            "overnight_sharpe_exceeds_intraday": bool(per_obs_sharpe(on_) > per_obs_sharpe(ir_)),
        }
    # cross-sectional pooled check across all 102 underlying-backed names
    all_on, all_ir = [], []
    for sym, d in name_series.items():
        if d.empty:
            continue
        dd = d.dropna(subset=["overnight_ret", "intraday_ret"])
        all_on.append(dd["overnight_ret"].to_numpy())
        all_ir.append(dd["intraday_ret"].to_numpy())
    if all_on:
        on_pool = np.concatenate(all_on)
        ir_pool = np.concatenate(all_ir)
        out["cross_sectional_pooled"] = {
            "n_obs": int(len(on_pool)),
            "mean_abs_overnight": round(float(np.mean(np.abs(on_pool))), 6),
            "mean_abs_intraday": round(float(np.mean(np.abs(ir_pool))), 6),
            "overnight_exceeds_intraday_abs": bool(np.mean(np.abs(on_pool)) > np.mean(np.abs(ir_pool))),
            "sharpe_overnight": round(float(per_obs_sharpe(on_pool) * np.sqrt(DAY_PER_YR)), 4),
            "sharpe_intraday": round(float(per_obs_sharpe(ir_pool) * np.sqrt(DAY_PER_YR)), 4),
            "overnight_sharpe_exceeds_intraday": bool(per_obs_sharpe(on_pool) > per_obs_sharpe(ir_pool)),
        }
    fact_holds = bool(out.get("SPYUSDT", {}) and out["SPYUSDT"].get("overnight_sharpe_exceeds_intraday") and
                       out.get("QQQUSDT", {}) and out["QQQUSDT"].get("overnight_sharpe_exceeds_intraday"))
    out["FACT_CHECK_HOLDS_AT_INDEX_LEVEL"] = fact_holds
    return out


S3_LOOKBACKS_WK = (4, 13, 26)


def s3_run(universe: list[dict], on_demo_syms: set[str], s1_ens_weekly: dict) -> dict:
    name_oc: dict[str, pd.DataFrame] = {}
    demo_syms_present = []
    for e in universe:
        sym = e["symbol"]
        if e.get("dukascopy") is None:
            continue
        try:
            df = load_stitched(sym)
        except FileNotFoundError:
            continue
        verify_seam_is_gapped(df, sym)
        oc = s3_daily_open_close(df)
        od = s3_overnight_intraday(oc)
        name_oc[sym] = od
        if sym in on_demo_syms:
            demo_syms_present.append(sym)

    fact = s3_fact_check(name_oc)

    results = {"fact_check": fact, "cells": {}, "canary": None}

    weekend = {sym: s3_weekend_return(s3_daily_open_close(load_stitched(sym)))
               for sym in ("SPYUSDT", "QQQUSDT") if sym in name_oc}
    results["weekend_cell"] = {
        sym: metrics(w.to_numpy(), WK_PER_YR) for sym, w in weekend.items() if len(w) > 30
    }

    # build a daily aligned overnight-Sharpe panel for the cross-sectional variant (b)
    syms = sorted(name_oc.keys())
    all_days = sorted(set().union(*[set(d.index) for d in name_oc.values() if not d.empty]))
    cal = pd.DatetimeIndex(all_days)
    n, m = len(cal), len(syms)
    ON = np.full((n, m), np.nan)
    IR = np.full((n, m), np.nan)
    for j, s in enumerate(syms):
        d = name_oc[s].reindex(cal)
        ON[:, j] = d["overnight_ret"].to_numpy()
        IR[:, j] = d["intraday_ret"].to_numpy()
    listed = np.isfinite(ON)
    demo_mask = np.array([s in on_demo_syms for s in syms])

    rvol_on = pd.DataFrame(ON).rolling(20, min_periods=10).std().to_numpy()

    # NOTE: ON/sh are indexed on the DAILY perp-era calendar (`cal` here, not a weekly grid), so
    # this is a 13-TRADING-DAY future shift (~2.6 weeks), not 13 weeks -- a modest, same-order-of-
    # magnitude leak relative to the 1-day-ahead forward return being scored (unlike S1's original
    # canary bug, which shifted by an entire window's worth of bars and washed out the correlation
    # with the scored forward return entirely -- see S1's canary comment for that lesson).
    canary_shift = -13

    def cross_sectional_cell(L_weeks: int, canary: bool = False) -> tuple[dict, np.ndarray]:
        L = L_weeks * 5  # trading days per week approx
        roll_sh = np.full((n, m), np.nan)
        on_df = pd.DataFrame(ON)
        mu = on_df.rolling(L, min_periods=max(5, L // 2)).mean()
        sd = on_df.rolling(L, min_periods=max(5, L // 2)).std()
        sh = (mu / sd).to_numpy()
        if canary:
            sh = pd.DataFrame(sh).shift(canary_shift).to_numpy()  # future info leak
        W = np.zeros((n, m))
        # weekly rebalance on Fridays of this daily calendar
        dow = cal.dayofweek.to_numpy()
        rb = dow == 4
        last_w = np.zeros(m)
        for t in range(n):
            if rb[t]:
                elig = listed[t] & np.isfinite(sh[t]) & np.isfinite(rvol_on[t]) & (rvol_on[t] > 0)
                if elig.sum() >= 2:
                    last_w = _quintile_book(sh[t], elig, rvol_on[t], long_high=True)
            W[t] = last_w
        fwd_on = np.roll(ON, -1, axis=0)
        fwd_on[-1] = np.nan
        gross = np.nansum(W * np.nan_to_num(fwd_on), axis=1)
        dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, m))))
        turn = dW.sum(axis=1)
        valid = np.isfinite(fwd_on).any(axis=1)
        valid[-1] = False
        out = {}
        for rt in RT_SCEN:
            net = (gross - (rt / 1e4) * turn)[valid]
            mm = metrics(net, WK_PER_YR)
            if rt == RT_PRIMARY_BP:
                if len(net) > 60:
                    paths = cpcv_paths(net, 8, 2, 21, WK_PER_YR)
                    if paths:
                        mm["cpcv_mean_path"] = round(float(np.mean(paths)), 4)
                        mm["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 3)
                mm["dsr"] = dsr(net, N_TRIALS, WK_PER_YR)
                mm["per_year"] = per_year_sharpe(net, cal[valid], WK_PER_YR)
                W_demo = W.copy()
                W_demo[:, ~demo_mask] = 0.0
                gross_d = np.nansum(W_demo * np.nan_to_num(fwd_on), axis=1)
                turn_d = np.abs(np.diff(W_demo, axis=0, prepend=np.zeros((1, m)))).sum(axis=1)
                net_d = (gross_d - (rt / 1e4) * turn_d)[valid]
                mm["on_demo_subset"] = metrics(net_d, WK_PER_YR)
            out[f"rt{rt}"] = mm
        primary_net = (gross - (RT_PRIMARY_BP / 1e4) * turn)[valid]
        return out, primary_net

    cell_nets = {}
    for L in S3_LOOKBACKS_WK:
        cell, net = cross_sectional_cell(L)
        results["cells"][f"xsec_L{L}"] = cell
        cell_nets[f"xsec_L{L}"] = net

    canary_cell, canary_net = cross_sectional_cell(13, canary=True)
    results["canary"] = metrics(canary_net, WK_PER_YR)
    results["canary"]["note"] = "overnight-Sharpe signal shifted -13wk (future info) -- expect fake, too-good Sharpe"

    # S3(c): execution-timing overlay on S1's weekly band-ensemble. Test: does concentrating S1's
    # rebalance/exposure into the overnight window (vs an evenly-spread intraday assumption) change
    # realized Sharpe? S1's P&L mechanism is funding (accrues continuously, 24/7, regardless of
    # execution timing) -- the overlay we can honestly test is TRANSACTION-COST timing: execute S1's
    # weekly rebalance trades at the overnight OPEN print (perp liquidity/spread may differ overnight
    # vs intraday) vs spread evenly through the RTH session. Since this lake has no intraday spread/
    # slippage model beyond the flat bp assumption, the honest overlay we can score is: does
    # SPYUSDT/QQQUSDT-average overnight vol (a proxy for execution cost/slippage regime) predict weeks
    # where evenly-spread-vs-concentrated execution would have differed -- reported as a null/inert
    # overlay given the flat-bp cost model has no time-of-day cost variation to exploit.
    results["s3c_overlay_on_s1"] = {
        "construction": ("S1 weekly band-ensemble trades executed AT the overnight-window print "
                          "(perp's post-RTH-close/pre-RTH-open price) instead of an evenly-spread "
                          "intraday assumption; scored against the SAME S1 funding P&L stream since "
                          "funding accrual is execution-timing-independent -- the overlay can only "
                          "affect realized turnover cost, not the funding leg itself"),
        "note": None, "sharpe_ann": None,
    }
    if s1_ens_weekly and "_internal" in s1_ens_weekly:
        internal = s1_ens_weekly["_internal"]
        We, s1_cal, s1_syms, F, valid_s1, demo_mask_s1, fwd_fund = (
            internal["We_weekly"], internal["cal"], internal["syms"], internal["F"],
            internal["valid"], internal["demo_mask"], internal["fwd_fund"])
        # overnight-concentrated timing: no flat-bp cost model varies by time of day in this lake, so
        # the overlay's honest effect on realized Sharpe is ZERO under the declared cost model; we
        # state this explicitly rather than inventing a fake slippage differential.
        gross = np.nansum((-We) * np.nan_to_num(fwd_fund), axis=1)
        turn = np.abs(np.diff(We, axis=0, prepend=np.zeros((1, We.shape[1])))).sum(axis=1)
        net_baseline = (gross - (13.0 / 1e4) * turn)[valid_s1]
        results["s3c_overlay_on_s1"]["sharpe_ann_baseline_s1"] = metrics(net_baseline, WK_PER_YR).get("sharpe_ann")
        results["s3c_overlay_on_s1"]["sharpe_ann_overnight_timed"] = metrics(net_baseline, WK_PER_YR).get("sharpe_ann")
        results["s3c_overlay_on_s1"]["note"] = (
            "identical to baseline: the declared flat-bp cost model (13bp RT primary) carries no "
            "time-of-day cost variation, so concentrating S1's execution into the overnight window "
            "cannot change realized Sharpe under this cost model -- reported honestly as a null "
            "overlay result (Delta Sharpe = 0.000) rather than fabricating a slippage differential "
            "the data does not support estimating.")
        results["s3c_overlay_on_s1"]["delta_sharpe"] = 0.0

    results["_internal"] = {"cell_nets": cell_nets, "name_oc": name_oc}
    return results


# ======================================================================================
# combined book (S1 + S3(b), completeness cell; S2 excluded -- no valid cells)
# ======================================================================================
def combined_book(s1_ens_net: np.ndarray, s1_cal: pd.DatetimeIndex,
                   s3_cell_nets: dict[str, np.ndarray]) -> dict:
    if not s3_cell_nets:
        return {"note": "S3 cross-sectional cells unavailable"}
    s3_best = s3_cell_nets[sorted(s3_cell_nets.keys())[0]]  # xsec_L4 as the representative member
    n = min(len(s1_ens_net), len(s3_best))
    a = s1_ens_net[-n:]
    b = s3_best[-n:]
    v1, v2 = np.var(a), np.var(b)
    w1 = (1.0 / v1) / (1.0 / v1 + 1.0 / v2) if v1 > 0 and v2 > 0 else 0.5
    w2 = 1 - w1
    combo = w1 * a + w2 * b
    combo_vt = vol_target(combo, WK_PER_YR, win=26)
    m = metrics(combo, WK_PER_YR)
    m["voltarget"] = metrics(combo_vt, WK_PER_YR)
    m["weights"] = {"s1_band_ensemble": round(float(w1), 3), "s3_xsec_overnight": round(float(w2), 3)}
    m["dsr"] = dsr(combo, N_TRIALS, WK_PER_YR)
    return m


# ======================================================================================
# main
# ======================================================================================
def main() -> None:
    t0 = time.time()
    universe = load_universe()
    on_demo_syms = {e["symbol"] for e in universe if e.get("on_demo")}
    print(f"universe: {len(universe)} symbols, {len(on_demo_syms)} on_demo", flush=True)

    results: dict = {"meta": {
        "n_trials_ledger": N_TRIALS,
        "universe_n": len(universe),
        "on_demo_n": len(on_demo_syms),
        "on_demo_symbols": sorted(on_demo_syms),
        "survivorship_caveat": ("The 110-symbol roster is TODAY's Binance-listed TradFi-equity perps "
                                 "-- today's winners, not a point-in-time universe. Every headline "
                                 "number in this file is subject to that selection bias."),
        "cost_model": {"primary_rt_bp": RT_PRIMARY_BP, "scenarios_bp": list(RT_SCEN),
                        "s1_fee_only_scenarios_bp": list(S1_FEE_SCEN)},
    }}

    print("=== S1 funding-carry cross-section ===", flush=True)
    P1 = s1_build_panel(universe)
    if not P1:
        results["s1"] = {"status": "NO FUNDING DATA FOUND"}
    else:
        print(f"S1 panel: {len(P1['cal'])} days, {len(P1['syms'])} syms with funding", flush=True)
        s1_res, s1_cell_nets = s1_run(P1, on_demo_syms)
        s1_internal = s1_res.pop("_internal")
        results["s1"] = s1_res
        results["s1"]["n_symbols_with_funding"] = len(P1["syms"])
    print(f"  S1 done @ {time.time()-t0:.1f}s", flush=True)

    print("=== S2 perp<->underlying basis mean reversion ===", flush=True)
    s2_res, s2_cell_nets = s2_run(universe, on_demo_syms)
    results["s2"] = s2_res
    print(f"  S2 done @ {time.time()-t0:.1f}s: {s2_res.get('status', 'ok')}", flush=True)

    print("=== S3 overnight/weekend structure ===", flush=True)
    s3_res = s3_run(universe, on_demo_syms, {"_internal": s1_internal} if P1 else {})
    s3_internal = s3_res.pop("_internal")
    results["s3"] = s3_res
    print(f"  S3 done @ {time.time()-t0:.1f}s", flush=True)
    print(f"  S3(a) fact check holds: {s3_res['fact_check'].get('FACT_CHECK_HOLDS_AT_INDEX_LEVEL')}", flush=True)

    # PBO over the full cell family (S1 daily+weekly x 3 lookbacks x 3 fee scenarios primary-fee net
    # series + S3 cross-sectional cells) -- run PBO on the primary-cost-scenario net series per cell
    # since PBO needs a common aligned calendar per sleeve; report per-sleeve PBO plus a note that a
    # single cross-sleeve PBO would require forcing daily/weekly calendars onto one grid (declared:
    # PBO computed WITHIN each sleeve's own natural cadence, which is the honest CSCV application
    # here since S1(daily)/S1(weekly)/S3(weekly) are not naturally the same clock).
    print("=== PBO (CSCV) per sleeve ===", flush=True)
    pbo_s1_weekly = pbo_cscv({k: v for k, v in s1_cell_nets.items() if k.startswith("weekly_")}) if P1 else None
    pbo_s1_daily = pbo_cscv({k: v for k, v in s1_cell_nets.items() if k.startswith("daily_")}) if P1 else None
    pbo_s3 = pbo_cscv(s3_internal["cell_nets"])
    pbo_s2 = pbo_cscv(s2_cell_nets) if s2_cell_nets else None
    results["pbo"] = {"s1_weekly_cells": pbo_s1_weekly, "s1_daily_cells": pbo_s1_daily,
                       "s3_xsec_cells": pbo_s3, "s2_full_48cell_family": pbo_s2,
                       "note": "CSCV-PBO computed within each sleeve's own native cadence family "
                                "(S1 daily cells, S1 weekly cells, S3 weekly cells, S2's full 48-cell "
                                "family aligned on its shortest member's tail) since S1(daily) and "
                                "S1/S3(weekly)/S2(mixed daily+weekly, 1min/5min) do not share a "
                                "common native clock to pool into one CSCV split without resampling "
                                "one onto the other's calendar."}
    print(f"  PBO done @ {time.time()-t0:.1f}s", flush=True)

    print("=== combined book (completeness cell, NOT the deployable) ===", flush=True)
    if P1 and "band_ensemble_weekly" in results["s1"]:
        we_net_key = "weekly_L14"  # any single-L net as a stand-in isn't the ensemble; recompute below
    combo = None
    if P1:
        We = s1_internal["We_weekly"]
        F = s1_internal["F"]
        valid_s1 = s1_internal["valid"]
        fwd_fund = s1_internal["fwd_fund"]
        gross = np.nansum((-We) * np.nan_to_num(fwd_fund), axis=1)
        turn = np.abs(np.diff(We, axis=0, prepend=np.zeros((1, We.shape[1])))).sum(axis=1)
        s1_ens_net = (gross - (13.0 / 1e4) * turn)[valid_s1]
        combo = combined_book(s1_ens_net, P1["cal"][valid_s1], s3_internal["cell_nets"])
    results["combined_book_s1_s3"] = combo if combo else {"status": "S1 or S3 unavailable"}
    print(f"  combined book done @ {time.time()-t0:.1f}s", flush=True)

    # adoption gate evaluation
    print("=== adoption gate ===", flush=True)
    gate = {}
    if P1 and "band_ensemble_weekly" in results["s1"]:
        ens = results["s1"]["band_ensemble_weekly"]
        gate["1_net_sharpe_positive_full"] = ens.get("sharpe_ann", 0) is not None and ens.get("sharpe_ann", -1) > 0
        gate["1_net_sharpe_positive_on_demo"] = ens.get("on_demo_subset", {}).get("sharpe_ann", -1) is not None and \
            (ens.get("on_demo_subset", {}).get("sharpe_ann") or -1) > 0
        gate["2_dsr_pass"] = (ens.get("dsr", {}).get("dsr") or 0) >= 0.95
        pbo_val = pbo_s1_weekly
        gate["3_pbo_le_025"] = (pbo_val is not None) and (pbo_val <= 0.25)
        gate["4_drop_top5_positive"] = (ens.get("drop5_sharpe") or -1) > 0
        gate["5a_s1_regime_sanity"] = results["s1"]["regime_sanity"]["mechanism_supported"]
        gate["5b_s3_fact_check"] = s3_res["fact_check"].get("FACT_CHECK_HOLDS_AT_INDEX_LEVEL", False)
        gate["ALL_PASS"] = all(bool(v) for v in gate.values())
    results["adoption_gate"] = gate
    print(json.dumps(gate, indent=1), flush=True)

    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "results_e1.json").write_text(json.dumps(results, indent=1, default=str))
    print("E1 DONE ->", OUT / "results_e1.json", flush=True)


if __name__ == "__main__":
    main()
