#!/usr/bin/env python
"""Resumable 1-minute backfill of the convexity TRADFI-PERP legs from data.binance.vision dumps.

Companion to backfill_crypto_1m.py — identical mechanics (no-rate-limit dump CDN, monthly + current
incomplete-month daily, atomic per-(symbol,period) parquet, resumable/idempotent, 404 = pre-listing),
but the symbol set is the commodities + equities sleeves (the Binance TradFi perps: CLUSDT, XAUUSDT,
NVDAUSDT, ...). These are the RECENT real-tradeable legs that the Dukascopy underlyings get ratio-stitched
onto. They are brand-new listings, so most months 404 (handled); the current-month DAILY dumps carry the
bulk of an equity perp's short life, so they are essential. Writes alongside crypto in data/klines/.

Run (on the VM, via the data venv so uv can't recreate the env mid-run):
  cd ~/convexity
  nohup .venv-data/bin/python scripts/backfill_tradfi_perp_1m.py >> data/backfill_tradfi_perp.out 2>&1 </dev/null &
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import pathlib
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIVERSE_MD = REPO / "src" / "convexity" / "panel" / "candidates.md"
LAKE = REPO / "data" / "klines"  # same lake + schema as crypto; perp symbols never collide with crypto
LOG = REPO / "data" / "backfill_tradfi_perp.log"
HISTORY_YEARS = 2  # TradFi perps are <1yr old; 2yr covers all with little 404 noise
WORKERS = 8
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume"]

LAKE.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("backfill_tradfi")


def load_tradfi_perp_symbols() -> list[str]:
    """The Binance perp `symbol` of every commodities + equities universe entry (SONY/NVO included)."""
    block = re.search(r"```json\n(.*?)\n```", UNIVERSE_MD.read_text(), re.S).group(1)
    d = json.loads(block)
    return [x["symbol"] for sleeve in ("commodities", "equities") for x in d[sleeve]]


def months_back(years: int) -> list[str]:
    today = dt.date.today()
    last_full = (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    y, m, out = last_full.year, last_full.month, []
    for _ in range(years * 12):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y -= 1
            m = 12
    return sorted(out)


def current_month_days() -> list[str]:
    today = dt.date.today()
    d, out = today.replace(day=1), []
    while d < today:  # exclude today (incomplete)
        out.append(d.strftime("%Y-%m-%d"))
        d += dt.timedelta(days=1)
    return out


def _fetch(url: str, client: httpx.Client, tries: int = 5) -> bytes | None:
    for i in range(tries):
        try:
            r = client.get(url, timeout=90)
            if r.status_code == 404:
                return None  # pre-listing or not yet published
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(2 ** i, 30))
    return None


def _parse_zip(content: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(content))
    raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if str(df.iloc[0, 0]).lower().startswith("open"):  # newer dumps carry a header row
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = COLS[: df.shape[1]]
    df = df[KEEP].apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time"])
    df["open_time"] = df["open_time"].astype("int64")
    return df


def _pull(kind: str, sym: str, period: str, client: httpx.Client):
    out = LAKE / sym / f"{sym}-1m-{period}.parquet"
    if out.exists():
        return ("skip", sym, period, 0)
    url = f"{BASE}/{kind}/klines/{sym}/1m/{sym}-1m-{period}.zip"
    content = _fetch(url, client)
    if content is None:
        return ("404", sym, period, 0)
    df = _parse_zip(content)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)  # atomic
    return ("ok", sym, period, len(df))


def main() -> None:
    syms = load_tradfi_perp_symbols()
    months = months_back(HISTORY_YEARS)
    days = current_month_days()
    log.info(f"START {len(syms)} tradfi-perp symbols x {len(months)} months + {len(days)} days (workers={WORKERS})")
    print(f"START {len(syms)} tradfi-perp symbols x {len(months)} months + {len(days)} days", flush=True)

    jobs = [("monthly", s, mth) for s in syms for mth in months] + \
           [("daily", s, d) for s in syms for d in days]
    ok = skip = miss = err = rows = 0
    with httpx.Client(headers={"User-Agent": "convexity-backfill"}) as client, ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_pull, k, s, p, client) for (k, s, p) in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                tag, _s, _p, n = f.result()
                if tag == "ok":
                    ok += 1
                    rows += n
                elif tag == "skip":
                    skip += 1
                else:
                    miss += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                log.error(f"task failed: {e}")
            if i % 250 == 0:
                msg = f"progress {i}/{len(jobs)} ok={ok} skip={skip} 404={miss} err={err} rows={rows/1e6:.2f}M"
                log.info(msg)
                print(msg, flush=True)
    done = f"DONE ok={ok} skip={skip} 404={miss} err={err} total_rows={rows/1e6:.2f}M lake={LAKE}"
    log.info(done)
    print(done, flush=True)


if __name__ == "__main__":
    main()
