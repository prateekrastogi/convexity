#!/usr/bin/env python
"""Backward-extend the Dukascopy commodity lake — CANDLE MODE (verified 2026-07-05).

Dukascopy's datafeed serves pre-aggregated day files `{BID,ASK}_candles_min_1.bi5` (LZMA, 1440
records of `>iiiiif` = sec-of-day, open, close, low, high as point-scaled int32, volume float32).
Verified against our tick-built bars: BID/ASK carry symmetric half-spread offsets, so the
(bid+ask)/2 mid reproduces the tick-built mid to ~1bp MAD. 2 requests/day instead of 24 -> ~12x
fewer requests than tick mode. Output schema and file paths are IDENTICAL to the tick path
(resume-compatible both ways). SELF-CHECK at startup: per code, 2 already-tick-built days are
rebuilt via candles and must agree (>=90% minute overlap, mid MAD <= 3bp) else that code falls
back to the module's tick-mode `_do_day`.

  nohup .venv/bin/python scripts/backfill_dukascopy_hist.py >> data/backfill_duka_hist.out 2>&1 &
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import lzma
import pathlib
import random
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("duka", REPO / "scripts" / "backfill_dukascopy_1m.py")
duka = importlib.util.module_from_spec(spec)
sys.modules["duka"] = duka
spec.loader.exec_module(duka)
duka.WORKERS = 12

FLOORS = {
    "XAUUSD": dt.date(2009, 1, 1),
    "XAGUSD": dt.date(2008, 6, 1),
    "LIGHTCMDUSD": dt.date(2013, 1, 1),
    "GASCMDUSD": dt.date(2013, 1, 1),
    "COPPERCMDUSD": dt.date(2012, 1, 1),
}
END = dt.date(2021, 5, 31)
REC = struct.Struct(">iiiiif")  # sec_of_day, open, close, low, high (int, x0.001), volume (f32)


def _candle_url(code: str, day: dt.date, side: str) -> str:
    return (f"https://datafeed.dukascopy.com/datafeed/{code}/"
            f"{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5")


def _fetch(client: httpx.Client, url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            r = client.get(url, timeout=30)
            if r.status_code == 404:
                return b""
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPError(f"throttle {r.status_code}")
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1) + random.random())
    return b""


def _side_frame(content: bytes) -> pd.DataFrame | None:
    if not content:
        return None
    try:
        raw = lzma.decompress(content)
    except lzma.LZMAError:
        return None
    n = len(raw) // REC.size
    if n == 0 or len(raw) % REC.size:
        return None
    rows = [REC.unpack_from(raw, k * REC.size) for k in range(n)]
    df = pd.DataFrame(rows, columns=["sec", "open", "close", "low", "high", "vol"])
    for c in ("open", "close", "low", "high"):
        df[c] = df[c] * duka.POINT
    return df


def candles_day(client: httpx.Client, code: str, day: dt.date) -> pd.DataFrame | None:
    """Mid 1-min bars for one day from the BID+ASK candle files (None => no data)."""
    bid = _side_frame(_fetch(client, _candle_url(code, day, "BID")))
    time.sleep(0.3 + random.random() * 0.5)  # same politeness spacing as the tick path
    ask = _side_frame(_fetch(client, _candle_url(code, day, "ASK")))
    if bid is None or ask is None:
        return None
    m = bid.merge(ask, on="sec", suffixes=("_b", "_a"))
    m["volsum"] = m["vol_b"] + m["vol_a"]
    m = m[m["volsum"] > 0]
    if m.empty:
        return None
    base = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
    out = pd.DataFrame({
        "open_time": base + m["sec"].astype("int64") * 1000,
        "open": (m["open_b"] + m["open_a"]) / 2,
        "high": (m["high_b"] + m["high_a"]) / 2,
        "low": (m["low_b"] + m["low_a"]) / 2,
        "close": (m["close_b"] + m["close_a"]) / 2,
        "volume": m["volsum"],
        "ticks": 0,
    })
    return out.sort_values("open_time").reset_index(drop=True)


def _do_day_candles(client: httpx.Client, code: str, day: dt.date):
    out = duka.LAKE / code / f"{code}-1m-{day:%Y-%m-%d}.parquet"
    if out.exists():
        return "skip", 0
    if day.weekday() == 5:  # Saturday: commodities closed
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "ticks"]
                     ).to_parquet(out, index=False)
        return "empty", 0
    bars = candles_day(client, code, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    if bars is None or bars.empty:
        pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "ticks"]
                     ).to_parquet(out, index=False)
        return "empty", 0
    tmp = out.with_suffix(".parquet.tmp")
    bars.to_parquet(tmp, index=False)
    tmp.rename(out)
    return "ok", len(bars)


def self_check(client: httpx.Client, code: str) -> bool:
    """Rebuild 2 already-tick-built days via candles; require >=90% overlap and mid MAD <= 3bp."""
    files = sorted((duka.LAKE / code).glob(f"{code}-1m-*.parquet"))
    picked = 0
    for f in files[len(files) // 4::11]:
        ticks = pd.read_parquet(f)
        if len(ticks) < 300:
            continue
        day = dt.date.fromisoformat(f.stem.split("-1m-")[1])
        cand = candles_day(client, code, day)
        if cand is None:
            return False
        j = ticks.merge(cand, on="open_time", suffixes=("_t", "_c"))
        if len(j) < 0.9 * len(ticks):
            print(f"SELF-CHECK FAIL {code} {day}: overlap {len(j)}/{len(ticks)}", flush=True)
            return False
        mad_bp = float(np.median(np.abs((j["close_c"] - j["close_t"]) / j["close_t"])) * 1e4)
        if mad_bp > 3.0:
            print(f"SELF-CHECK FAIL {code} {day}: mid MAD {mad_bp:.1f}bp", flush=True)
            return False
        print(f"self-check {code} {day}: overlap {len(j)}/{len(ticks)}, MAD {mad_bp:.2f}bp OK",
              flush=True)
        picked += 1
        if picked == 2:
            return True
    return picked > 0


def main() -> None:
    with httpx.Client(headers={"User-Agent": duka.UA}) as probe:
        modes = {code: ("candles" if self_check(probe, code) else "ticks") for code in FLOORS}
    print(f"modes: {modes}", flush=True)
    grid = []
    for code, floor in FLOORS.items():
        d = floor
        while d <= END:
            grid.append((code, d))
            d += dt.timedelta(days=1)
    print(f"START hist extension (candle-mode): {len(grid)} code-days", flush=True)
    ok = skip = empty = err = bars = 0
    with httpx.Client(headers={"User-Agent": duka.UA}) as client, \
            ThreadPoolExecutor(duka.WORKERS) as ex:
        def work(code, day):
            if modes[code] == "candles":
                return _do_day_candles(client, code, day)
            return duka._do_day(client, code, day, False)
        futs = {ex.submit(work, code, day): (code, day) for code, day in grid}
        for i, f in enumerate(as_completed(futs), 1):
            code, day = futs[f]
            try:
                tag, n = f.result()
                if tag == "ok":
                    ok += 1
                    bars += n
                elif tag == "skip":
                    skip += 1
                else:
                    empty += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                duka.log.error(f"hist {code} {day} failed: {e}")
            if i % 1000 == 0:
                print(f"hist progress {i}/{len(futs)} ok={ok} skip={skip} empty={empty} err={err} "
                      f"bars={bars / 1e6:.2f}M", flush=True)
    print(f"HIST DONE ok={ok} skip={skip} empty={empty} err={err} bars={bars / 1e6:.2f}M", flush=True)


if __name__ == "__main__":
    main()
