#!/usr/bin/env python
"""Resumable funding-rate backfill for the convexity universe from data.binance.vision dumps.

WHY: the COST pillar charges funding only if ``FundingRateUpdate`` data is fed to the BacktestEngine
(COST §5.5). Binance publishes the realized funding per perp at
``/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip`` (+ a current-month daily
mirror), CSV columns ``calc_time(ms), funding_interval_hours, last_funding_rate`` — ~3 rows/day
(8h stamps), so the files are tiny. Same dump-CDN mechanics as backfill_crypto_1m.py: no API key, no
rate limit, idempotent/resumable (one parquet per (symbol,period); re-running skips on-disk files),
atomic writes, 404 = pre-listing.

Output: ``data/funding/{SYMBOL}.parquet`` (consolidated per symbol), columns
``calc_time | funding_interval_hours | last_funding_rate`` sorted by ``calc_time`` and de-duplicated.

Run (on the VM, via the data venv so uv can't recreate the env mid-run):
  cd ~/convexity
  nohup .venv-data/bin/python scripts/backfill_funding_1m.py >> data/backfill_funding.out 2>&1 </dev/null &

Limit the symbol set with positional args (e.g. ``... backfill_funding_1m.py BTCUSDT ETHUSDT``).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import pathlib
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIVERSE_MD = REPO / "src" / "convexity" / "selection" / "UNIVERSE.md"
LAKE = REPO / "data" / "funding"
LOG = REPO / "data" / "backfill_funding.log"
HISTORY_YEARS = 6
WORKERS = 8
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

LAKE.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("backfill_funding")


def load_universe_symbols() -> list[str]:
    """Every perp symbol in the universe (crypto + tradfi equities + commodities)."""
    block = re.search(r"```json\n(.*?)\n```", UNIVERSE_MD.read_text(), re.S).group(1)
    u = json.loads(block)
    syms: list[str] = []
    for d in u["crypto"]:
        syms.append(d["symbol"])
    for d in u["equities"] + u["commodities"]:
        syms.append(d["symbol"])  # the Binance perp leg (e.g. NVDAUSDT, XAUUSDT)
    return sorted(set(syms))


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
                return None
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(2**i, 30))
    return None


def _parse_zip(content: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(content))
    raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if str(df.iloc[0, 0]).lower().startswith("calc"):  # newer dumps carry a header row
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = COLS[: df.shape[1]]
    df = df[COLS].apply(pd.to_numeric, errors="coerce").dropna(subset=["calc_time"])
    df["calc_time"] = df["calc_time"].astype("int64")
    return df


def _pull_period(kind: str, sym: str, period: str, client: httpx.Client) -> pd.DataFrame | None:
    url = f"{BASE}/{kind}/fundingRate/{sym}/{sym}-fundingRate-{period}.zip"
    content = _fetch(url, client)
    if content is None:
        return None
    return _parse_zip(content)


def _backfill_symbol(sym: str, months: list[str], days: list[str], client: httpx.Client):
    out = LAKE / f"{sym}.parquet"
    if out.exists():
        return ("skip", sym, 0)
    frames: list[pd.DataFrame] = []
    for mth in months:
        df = _pull_period("monthly", sym, mth, client)
        if df is not None and len(df):
            frames.append(df)
    for day in days:
        df = _pull_period("daily", sym, day, client)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return ("404", sym, 0)
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["calc_time"]).sort_values("calc_time")
    full = full.reset_index(drop=True)
    tmp = out.with_suffix(".parquet.tmp")
    full.to_parquet(tmp, index=False)
    tmp.rename(out)  # atomic
    return ("ok", sym, len(full))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    syms = args if args else load_universe_symbols()
    months = months_back(HISTORY_YEARS)
    days = current_month_days()
    log.info(f"START funding {len(syms)} symbols x {len(months)} months + {len(days)} days")
    print(f"START funding {len(syms)} symbols x {len(months)} months + {len(days)} days", flush=True)

    ok = skip = miss = err = 0
    rows = 0
    results: list[tuple[str, int]] = []
    with httpx.Client(headers={"User-Agent": "convexity-backfill"}) as client, ThreadPoolExecutor(
        WORKERS
    ) as ex:
        futs = {ex.submit(_backfill_symbol, s, months, days, client): s for s in syms}
        for f in as_completed(futs):
            try:
                tag, sym, n = f.result()
                if tag == "ok":
                    ok += 1
                    rows += n
                    results.append((sym, n))
                elif tag == "skip":
                    skip += 1
                else:
                    miss += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                log.error(f"task failed: {e}")
    done = f"DONE ok={ok} skip={skip} 404={miss} err={err} total_rows={rows} lake={LAKE}"
    log.info(done)
    print(done, flush=True)
    for sym, n in sorted(results):
        print(f"  {sym}: {n}", flush=True)


if __name__ == "__main__":
    main()
