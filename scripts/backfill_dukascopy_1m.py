#!/usr/bin/env python
"""Resumable, polite 1-minute backfill of Dukascopy underlying proxies for the convexity TradFi sleeves.

Recipe (live-verified): GET datafeed.dukascopy.com/datafeed/{CODE}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
where MM0 = month-1 (ZERO-INDEXED; the file still carries real-month timestamps). Body = LZMA blob of
20-byte big-endian records `>3i2f` = (ms_since_hour, ask_int, bid_int, askVol_f, bidVol_f); price =
int * 0.001 for all our instruments (metals/commodities/indices/equity-CFDs). Aggregate tick mids to
1-min OHLCV. Empty hour = 404 OR 200-with-empty-LZMA; 503/timeout = throttle -> backoff+retry.

Politeness: low concurrency + per-request jitter spacing + exponential backoff (Dukascopy 503s on
un-spaced hammering). Resumable: one parquet per (code, day); re-runs skip done days (incl. empty
weekend/holiday markers) so it can be killed/restarted and finish properly.

Usage (on the VM, via the venv binary so uv can't recreate the env mid-run):
  cd ~/convexity
  nohup .venv/bin/python scripts/backfill_dukascopy_1m.py commodities >> data/backfill_duka_commodities.out 2>&1 </dev/null &
  .venv/bin/python scripts/backfill_dukascopy_1m.py --test XAUUSD 2026-06-24   # single-day verify
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import lzma
import pathlib
import random
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIVERSE_MD = REPO / "src" / "convexity" / "selection" / "UNIVERSE.md"
LAKE = REPO / "data" / "dukascopy"
LOG = REPO / "data" / "backfill_dukascopy.log"
START = dt.date(2021, 6, 1)  # 5-yr floor; per-instrument history floor is handled by empty hours
POINT = 0.001                # divisor is the same for metals/commodities/indices/equity CFDs
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
WORKERS = 6
REC = struct.Struct(">3i2f")  # ms_since_hour, ask, bid, askVol, bidVol (big-endian)

LAKE.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("duka")


def load_tradfi(sleeves: list[str]) -> list[tuple[str, str, bool]]:
    """Return (binance_symbol, dukascopy_code, is_equity) for proxied instruments in the sleeves."""
    block = re.search(r"```json\n(.*?)\n```", UNIVERSE_MD.read_text(), re.S).group(1)
    data = json.loads(block)
    out = []
    for s in sleeves:
        for d in data[s]:
            if d.get("dukascopy"):
                out.append((d["symbol"], d["dukascopy"], s == "equities"))
    return out


def _url(code: str, day: dt.date, hh: int) -> str:
    return (f"https://datafeed.dukascopy.com/datafeed/{code}/"
            f"{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{hh:02d}h_ticks.bi5")


def _fetch_hour(client: httpx.Client, code: str, day: dt.date, hh: int, tries: int = 5) -> bytes:
    """Return raw .bi5 bytes, or b'' for an empty hour (404). Raises after `tries` throttles/timeouts."""
    for i in range(tries):
        try:
            r = client.get(_url(code, day, hh), timeout=30)
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


def _ticks(content: bytes, day: dt.date, hh: int) -> list[tuple[dt.datetime, float, float]]:
    if not content:
        return []
    try:
        raw = lzma.decompress(content)
    except lzma.LZMAError:
        return []
    base = dt.datetime(day.year, day.month, day.day, hh, tzinfo=dt.timezone.utc)
    out = []
    for k in range(len(raw) // 20):
        ms, ask_i, bid_i, _av, _bv = REC.unpack_from(raw, k * 20)
        out.append((base + dt.timedelta(milliseconds=ms), (ask_i + bid_i) * 0.5 * POINT, _av + _bv))
    return out


def _hours(day: dt.date, is_equity: bool):
    if is_equity:
        if day.weekday() >= 5:  # equities: weekdays only, 12-22 UTC covers US RTH incl. DST
            return []
        return range(12, 23)
    if day.weekday() == 5:  # commodities/metals ~24x5: skip Saturday
        return []
    return range(0, 24)


def _do_day(client: httpx.Client, code: str, day: dt.date, is_equity: bool):
    out = LAKE / code / f"{code}-1m-{day:%Y-%m-%d}.parquet"
    if out.exists():
        return "skip", 0
    hrs = list(_hours(day, is_equity))
    ticks: list = []
    for hh in hrs:
        ticks += _ticks(_fetch_hour(client, code, day, hh), day, hh)
        time.sleep(0.3 + random.random() * 0.5)  # politeness spacing
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ticks:  # weekend / holiday / pre-listing -> empty marker so resume skips it
        pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "ticks"]).to_parquet(out, index=False)
        return "empty", 0
    df = pd.DataFrame(ticks, columns=["ts", "mid", "vol"])
    df["m"] = df["ts"].dt.floor("1min")
    g = df.groupby("m")
    bars = pd.DataFrame({
        "open": g["mid"].first(), "high": g["mid"].max(), "low": g["mid"].min(),
        "close": g["mid"].last(), "volume": g["vol"].sum(), "ticks": g.size(),
    }).reset_index()
    # epoch MS, resolution-independent (pandas 3 defaults to us; the crypto lake uses ms — must match)
    _ms = bars["m"].dt.tz_convert("UTC").to_numpy().astype("datetime64[ms]")
    bars["open_time"] = _ms.astype("int64")
    bars = bars[["open_time", "open", "high", "low", "close", "volume", "ticks"]]
    tmp = out.with_suffix(".parquet.tmp")
    bars.to_parquet(tmp, index=False)
    tmp.rename(out)
    return "ok", len(bars)


def _days():
    today = dt.date.today()
    d, out = START, []
    while d < today:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "--test":
        code, day = args[1], dt.date.fromisoformat(args[2])
        with httpx.Client(headers={"User-Agent": UA}) as client:
            is_eq = not (code.endswith("CMDUSD") or code in ("XAUUSD", "XAGUSD"))
            tag, n = _do_day(client, code, day, is_eq)
            f = LAKE / code / f"{code}-1m-{day:%Y-%m-%d}.parquet"
            print("result:", tag, n, "bars")
            if f.exists():
                df = pd.read_parquet(f)
                print(df.head(3).to_string())
                print("...")
                print(df.tail(2).to_string())
                print("rows:", len(df), "price range:", round(df.low.min(), 4), "-", round(df.high.max(), 4))
        return

    sleeves = args or ["commodities"]
    insts = load_tradfi(sleeves)
    days = _days()
    log.info(f"START dukascopy {sleeves}: {len(insts)} instruments x up to {len(days)} days (workers={WORKERS})")
    print(f"START {sleeves}: {len(insts)} instruments x {len(days)} days", flush=True)
    ok = skip = empty = err = bars = 0
    with httpx.Client(headers={"User-Agent": UA}) as client, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(_do_day, client, code, day, is_eq): (code, day)
                for (_sym, code, is_eq) in insts for day in days}
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
                log.error(f"{code} {day} failed: {e}")
            if i % 500 == 0:
                msg = f"progress {i}/{len(futs)} ok={ok} skip={skip} empty={empty} err={err} bars={bars/1e6:.2f}M"
                log.info(msg)
                print(msg, flush=True)
    done = f"DONE {sleeves} ok={ok} skip={skip} empty={empty} err={err} bars={bars/1e6:.2f}M lake={LAKE}"
    log.info(done)
    print(done, flush=True)


if __name__ == "__main__":
    main()
