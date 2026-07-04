#!/usr/bin/env python3
"""ADDENDUM 1 — session-anchor reproduction diagnostic (see ADDENDUM1.md; frozen before run).

3 session anchors x 2 stops x 7 symbols, + CL period splits and per-year detail for the
MCX-faithful cell. Session frames are built from the stitched 1-min bars per calendar day
(bars in [start,end) UTC); the strong-close ORB signal is evaluated on SESSION h/l/close at the
session end; the stop is checked on ALL 24/7 minute bars during the 5-session-day hold
(gap-through at any bar open beyond the stop). Net 12bp RT primary; 6/24 reported.
-> results_anchor_repro.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity")
STUDY = REPO / "research/entry_exit/study"
SYMS = ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT", "COPPERUSDT", "XPTUSDT", "XPDUSDT"]
SESSIONS = {  # name: (start_min_utc, end_min_utc, or_minutes)
    "MCX": (3 * 60 + 30, 18 * 60 + 24, 70),
    "LDN": (8 * 60, 16 * 60 + 30, 40),
    "NY": (13 * 60 + 30, 19 * 60 + 30, 30),
}
STOPS = {"fix0.75": ("fixed", 0.0075), "atr0.5": ("atr", 0.5)}
HOLD = 5
RT_BPS = (6.0, 12.0, 24.0)
STRONG_CLOSE = 0.7
MIN_SESSION_BARS = 200
ATR_N = 20
CL_WINDOW = (pd.Timestamp("2023-10-01"), pd.Timestamp("2025-02-28"))


def load_minutes(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(REPO / f"data/stitched/{sym}.parquet",
                         columns=["open_time", "open", "high", "low", "close"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["date"] = ts.dt.date
    df["mod"] = ts.dt.hour * 60 + ts.dt.minute
    return df


def session_frame(mins: pd.DataFrame, s0: int, s1: int, or_min: int) -> pd.DataFrame:
    """Per calendar day: session OR/h/l/close (+ ATR20 over session TR, shifted 1d)."""
    m = mins[(mins["mod"] >= s0) & (mins["mod"] < s1)]
    rows = []
    for d, day in m.groupby("date", sort=True):
        if len(day) < MIN_SESSION_BARS:
            continue
        hi = day["high"].to_numpy()
        lo = day["low"].to_numpy()
        in_or = day["mod"].to_numpy() < s0 + or_min
        if not in_or.any():
            continue
        rows.append({"d": pd.Timestamp(d), "or_hi": float(hi[in_or].max()),
                     "or_lo": float(lo[in_or].min()), "s_hi": float(hi.max()),
                     "s_lo": float(lo.min()), "s_close": float(day["close"].iloc[-1])})
    f = pd.DataFrame(rows).set_index("d").sort_index()
    prev = f["s_close"].shift(1)
    tr = np.maximum(f["s_hi"] - f["s_lo"],
                    np.maximum((f["s_hi"] - prev).abs(), (f["s_lo"] - prev).abs()))
    f["atr20"] = tr.rolling(ATR_N).mean().shift(1)
    return f


def day_bar_index(mins: pd.DataFrame) -> dict:
    out = {}
    for d, day in mins.groupby("date", sort=True):
        out[d] = (day["open"].to_numpy(), day["high"].to_numpy(), day["low"].to_numpy())
    return out


def signal(row) -> int:
    rng = row.s_hi - row.s_lo if row.s_hi > row.s_lo else np.inf
    if row.s_hi > row.or_hi and row.s_close > row.or_hi and (row.s_close - row.s_lo) / rng > STRONG_CLOSE:
        return 1
    if row.s_lo < row.or_lo and row.s_close < row.or_lo and (row.s_hi - row.s_close) / rng > STRONG_CLOSE:
        return -1
    return 0


def simulate(frame: pd.DataFrame, bars: dict, stop_kind: str, stop_val: float):
    days = frame.index
    n = len(frame)
    close = frame["s_close"].to_numpy()
    atr = frame["atr20"].to_numpy()
    trades = []
    i = ATR_N + 5
    recs = list(frame.itertuples())
    # calendar days between session days (stop runs on ALL 24/7 bars incl. non-session days)
    all_days = sorted(bars.keys())
    pos_of = {d: k for k, d in enumerate(all_days)}
    while i < n - 1:
        side = signal(recs[i])
        if side == 0:
            i += 1
            continue
        dist = stop_val * (atr[i] if stop_kind == "atr" else 0.0) + \
            (stop_val * close[i] if stop_kind == "fixed" else 0.0)
        if stop_kind == "atr" and (not np.isfinite(atr[i]) or atr[i] <= 0):
            i += 1
            continue
        entry = close[i]
        stop = entry - side * dist
        end = min(i + HOLD, n - 1)
        # walk EVERY calendar day from the day after the signal session-day to the end session-day
        d_start = days[i].date()
        d_end = days[end].date()
        k0 = pos_of[d_start] + 1 if d_start in pos_of else 0
        exit_ret, exit_i = None, end
        for k in range(k0, len(all_days)):
            d = all_days[k]
            if d > d_end:
                break
            o, h, lo = bars[d]
            if side == 1:
                if o[0] <= stop:
                    exit_ret = o[0] / entry - 1.0
                elif lo.min() <= stop:
                    exit_ret = stop / entry - 1.0
            else:
                if o[0] >= stop:
                    exit_ret = 1.0 - o[0] / entry
                elif h.max() >= stop:
                    exit_ret = 1.0 - stop / entry
            if exit_ret is not None:
                # exit position = the first session day >= d (for non-overlap bookkeeping)
                later = np.searchsorted(days.values, np.datetime64(pd.Timestamp(d)))
                exit_i = min(max(int(later), i + 1), end)
                break
        if exit_ret is None:
            exit_ret = side * (close[end] / entry - 1.0)
        trades.append((days[i], days[exit_i], float(exit_ret)))
        i = exit_i + 1
    return trades


def metrics(trades, rt_bp: float, lo=None, hi=None) -> dict:
    sel = [t for t in trades if (lo is None or t[0] >= lo) and (hi is None or t[0] <= hi)]
    if not sel:
        return {"n": 0}
    net = np.array([t[2] for t in sel]) - rt_bp / 1e4
    yrs = max((sel[-1][0] - sel[0][0]).days / 365.25, 1e-9)
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    return {
        "n": len(net), "win_pct": round(float((net > 0).mean()) * 100, 1),
        "ev_bp": round(float(net.mean()) * 1e4, 1),
        "sharpe_tr": round(float(net.mean() / net.std()) if net.std() > 0 else 0.0, 3),
        "sharpe_ann": round(float(net.mean() / net.std() * np.sqrt(len(net) / yrs))
                            if net.std() > 0 else 0.0, 2),
        "maxdd_pct": round(float((1 - eq / peak).max()) * 100, 1),
        "ev_drop3_bp": round(float(np.sort(net)[:-3].mean()) * 1e4, 1) if len(net) > 3 else None,
        "trades_per_yr": round(len(net) / yrs, 1),
    }


def main() -> None:
    out: dict = {}
    for sym in SYMS:
        t0 = time.time()
        mins = load_minutes(sym)
        bars = day_bar_index(mins)
        out[sym] = {}
        for sname, (s0, s1, orm) in SESSIONS.items():
            f = session_frame(mins, s0, s1, orm)
            for stname, (kind, val) in STOPS.items():
                trades = simulate(f, bars, kind, val)
                cell = {f"rt{int(rt)}": metrics(trades, rt) for rt in RT_BPS}
                if sym == "CLUSDT":
                    cell["rt12_origwin"] = metrics(trades, 12.0, *CL_WINDOW)
                    cell["rt12_excl_origwin_pre"] = metrics(trades, 12.0, None,
                                                            CL_WINDOW[0] - pd.Timedelta(days=1))
                    by_yr = {}
                    for t in trades:
                        y = t[0].year
                        by_yr.setdefault(y, []).append(t)
                    cell["rt12_by_year_ev"] = {y: round(float(np.mean([x[2] for x in v]) * 1e4 - 12), 1)
                                               for y, v in sorted(by_yr.items())}
                out[sym][f"{sname}_{stname}"] = cell
        print(f"{sym} done in {time.time()-t0:.1f}s", flush=True)
    (STUDY / "results_anchor_repro.json").write_text(json.dumps(out, indent=1))
    print("ANCHOR REPRO DONE", flush=True)


if __name__ == "__main__":
    main()
