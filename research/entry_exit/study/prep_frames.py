#!/usr/bin/env python3
"""Stage 1 — per-symbol DAILY decision frames from the stitched 1-min lake.

Per UTC day and symbol: opening-range highs/lows for W in {30,120} minutes, day OHLC, bar count,
gap count, and ATR20 (20-day EMA-free simple mean of daily true range, shifted 1 day so it is
F_{t-1}-measurable at the day-t close decision). Valid decision day: >= 300 one-minute bars
(COMMODITY_PREREG.md §1). Frames -> research/entry_exit/study/frames/{SYM}.parquet (gitignored via
the repo *.parquet rule); the 1-min bars are re-read at simulation time for stop-touch fills.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity")
OUT = REPO / "research/entry_exit/study/frames"
SYMS = ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT", "COPPERUSDT", "XPTUSDT", "XPDUSDT"]
MIN_BARS = 300
OR_WINDOWS = (30, 120)
ATR_N = 20


def build_frame(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(REPO / f"data/stitched/{sym}.parquet",
                         columns=["open_time", "open", "high", "low", "close", "gap_before"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.assign(d=ts.dt.date, minute_of_day=(ts.dt.hour * 60 + ts.dt.minute))
    g = df.groupby("d", sort=True)
    rows = []
    for d, day in g:
        n = len(day)
        if n < MIN_BARS:
            continue
        hi = day["high"].to_numpy()
        lo = day["low"].to_numpy()
        rec = {
            "d": pd.Timestamp(d),
            "n_bars": n,
            "n_gaps": int(day["gap_before"].sum()),
            "day_open": float(day["open"].iloc[0]),
            "day_high": float(hi.max()),
            "day_low": float(lo.min()),
            "day_close": float(day["close"].iloc[-1]),
        }
        mod = day["minute_of_day"].to_numpy()
        for w in OR_WINDOWS:
            m = mod < w  # first w minutes of the UTC day (bars are UTC-day anchored)
            if not m.any():
                rec[f"or{w}_hi"] = np.nan
                rec[f"or{w}_lo"] = np.nan
            else:
                rec[f"or{w}_hi"] = float(hi[m].max())
                rec[f"or{w}_lo"] = float(lo[m].min())
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("d").sort_index()
    # ATR20, F_{t-1}: true range uses previous day close; the rolling mean is shifted one day
    prev_close = out["day_close"].shift(1)
    tr = np.maximum(out["day_high"] - out["day_low"],
                    np.maximum((out["day_high"] - prev_close).abs(),
                               (out["day_low"] - prev_close).abs()))
    out["atr20"] = tr.rolling(ATR_N).mean().shift(1)
    out["ret_1d"] = np.log(out["day_close"] / out["day_close"].shift(1))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in SYMS:
        t0 = time.time()
        f = build_frame(sym)
        f.to_parquet(OUT / f"{sym}.parquet")
        print(f"{sym}: {len(f)} valid days [{f.index[0].date()}..{f.index[-1].date()}] "
              f"median_bars={int(f.n_bars.median())} in {time.time()-t0:.1f}s", flush=True)
    print("ALL FRAMES DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
