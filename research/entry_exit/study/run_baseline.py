#!/usr/bin/env python3
"""Stage 2 — the BASELINE (intraday-README incumbent on 24/7 perps), per COMMODITY_PREREG.md §2.

6 declared cells (W in {30,120} x s in {0.25,0.5,1.0}) x 7 symbols. Day-close entry on the verbatim
strong-close ORB signal; ATR20-scaled stop checked on every 1-min bar of the hold with
gap-through-at-open fills; 5-valid-day hold; non-overlapping. Net-of-cost {6,12,24}bp RT (+/-5bp/day
funding scenario reported at the primary cost). Full-sample metrics + CPCV OOS path distribution
(the baseline has NO fitted params, so CPCV here = the path Sharpes on each split's test days).
-> results_baseline.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path("/home/speculation/convexity/research/entry_exit/study")
sys.path.insert(0, str(STUDY))
from cpcv_daily import path_sharpes, sharpe_ann_daily  # noqa: E402

REPO = Path("/home/speculation/convexity")
SYMS = ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT", "COPPERUSDT", "XPTUSDT", "XPDUSDT"]
OR_WINDOWS = (30, 120)
STOP_S = (0.25, 0.5, 1.0)
HOLD = 5
RT_BPS = (6.0, 12.0, 24.0)  # 12 primary
FUNDING_BP_PER_DAY = 5.0
STRONG_CLOSE = 0.7


def minute_arrays(sym: str) -> dict:
    df = pd.read_parquet(REPO / f"data/stitched/{sym}.parquet",
                         columns=["open_time", "open", "high", "low"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    d = ts.dt.date.to_numpy()
    out = {}
    # split positions at day changes (bars are time-sorted)
    change = np.nonzero(d[1:] != d[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(d)]])
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    for s0, e0 in zip(starts, ends):
        out[d[s0]] = (o[s0:e0], h[s0:e0], lo[s0:e0])
    return out


def signal_row(row, w: int) -> int:
    orh, orl = row[f"or{w}_hi"], row[f"or{w}_lo"]
    dh, dl, c = row["day_high"], row["day_low"], row["day_close"]
    if not np.isfinite(orh) or not np.isfinite(orl):
        return 0
    rng = dh - dl if dh > dl else np.inf
    if dh > orh and c > orh and (c - dl) / rng > STRONG_CLOSE:
        return 1
    if dl < orl and c < orl and (dh - c) / rng > STRONG_CLOSE:
        return -1
    return 0


def simulate(frame: pd.DataFrame, minutes: dict, w: int, s: float):
    """-> trades [(entry_i, exit_i, side, gross_ret, days_held)] over frame positions."""
    days = frame.index.to_numpy()
    n = len(frame)
    atr = frame["atr20"].to_numpy()
    close = frame["day_close"].to_numpy()
    trades = []
    i = 25  # ATR warm-up
    rows = frame.to_dict("records")
    while i < n - 1:
        side = signal_row(rows[i], w)
        if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        entry = close[i]
        stop = entry - side * s * atr[i]
        end = min(i + HOLD, n - 1)
        exit_ret, exit_j, held = None, end, end - i
        for j in range(i + 1, end + 1):
            key = pd.Timestamp(days[j]).date()
            if key not in minutes:
                continue
            o, h, lo = minutes[key]
            if side == 1:
                if o[0] <= stop:  # gap-through at the open
                    exit_ret, exit_j, held = o[0] / entry - 1.0, j, j - i
                    break
                if lo.min() <= stop:  # find first touching bar; fill AT the stop
                    exit_ret, exit_j, held = stop / entry - 1.0, j, j - i
                    break
            else:
                if o[0] >= stop:
                    exit_ret, exit_j, held = 1.0 - o[0] / entry, j, j - i
                    break
                if h.max() >= stop:
                    exit_ret, exit_j, held = 1.0 - stop / entry, j, j - i
                    break
        if exit_ret is None:
            exit_ret = side * (close[end] / entry - 1.0)
        trades.append((i, exit_j, side, float(exit_ret), int(held)))
        i = exit_j + 1
    return trades


def metrics(trades, n_days: int, rt_bp: float, funding_bp_day: float = 0.0) -> dict:
    if not trades:
        return {"n": 0}
    gross = np.array([t[3] for t in trades])
    held = np.array([t[4] for t in trades])
    net = gross - rt_bp / 1e4 - funding_bp_day / 1e4 * held
    daily = np.zeros(n_days)
    for t, r in zip(trades, net):
        daily[t[1]] += r  # assign the trade's net return to its exit day (embargo covers overlap)
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    srt = np.sort(net)[::-1]
    tot = net.sum() if net.sum() != 0 else 1e-12
    return {
        "n": len(net),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "ev_bp": round(float(net.mean()) * 1e4, 1),
        "sharpe_trade": round(float(net.mean() / net.std()) if net.std() > 0 else 0.0, 3),
        "sharpe_ann": round(sharpe_ann_daily(daily), 3),
        "maxdd_pct": round(float((1 - eq / peak).max()) * 100, 1),
        "ev_drop1_bp": round(float(np.sort(net)[:-1].mean()) * 1e4, 1) if len(net) > 1 else None,
        "ev_drop3_bp": round(float(np.sort(net)[:-3].mean()) * 1e4, 1) if len(net) > 3 else None,
        "top3_share": round(float(srt[:3].sum() / tot), 2),
        "avg_hold_d": round(float(held.mean()), 2),
        "_daily": daily,
    }


def main() -> None:
    results: dict = {}
    for sym in SYMS:
        t0 = time.time()
        frame = pd.read_parquet(STUDY / f"frames/{sym}.parquet")
        minutes = minute_arrays(sym)
        results[sym] = {}
        for w in OR_WINDOWS:
            for s in STOP_S:
                trades = simulate(frame, minutes, w, s)
                cell: dict = {}
                for rt in RT_BPS:
                    m = metrics(trades, len(frame), rt)
                    daily = m.pop("_daily", None)
                    cell[f"rt{int(rt)}"] = m
                    if rt == 12.0 and daily is not None:
                        ps = path_sharpes(daily)
                        cell["cpcv_rt12"] = {
                            "mean_path_sharpe": round(float(np.mean(ps)), 3),
                            "frac_pos": round(float(np.mean([p > 0 for p in ps])), 2),
                        }
                for fd in (-FUNDING_BP_PER_DAY, FUNDING_BP_PER_DAY):
                    m = metrics(trades, len(frame), 12.0, fd)
                    m.pop("_daily", None)
                    cell[f"rt12_fund{int(fd):+d}"] = {k: m[k] for k in ("n", "ev_bp", "sharpe_ann")
                                                      if k in m}
                results[sym][f"W{w}_s{s}"] = cell
        print(f"{sym} done in {time.time()-t0:.1f}s", flush=True)
    (STUDY / "results_baseline.json").write_text(json.dumps(results, indent=1))
    print("BASELINE DONE", flush=True)


if __name__ == "__main__":
    main()
