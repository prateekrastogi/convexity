#!/usr/bin/env python
"""One-off CME futures-curve backfill via databento (GLBX.MDP3, daily) -> data/curves/.

CONTINUOUS symbology, expiry-ranked: {ROOT}.c.{0..7} = the 1st..8th listed contract each day —
exactly the curve points the carry/basis-momentum factors need, with rolls handled by databento
and NO calendar-spread instruments inflating the bill (parent symbology cost ~$30; this ~cents).
Schemas: ohlcv-1d (per-rank daily bars) + statistics (settlement price + open interest).
ALWAYS run --cost first (prints the exact USD debit, spends nothing).

  .venv/bin/python scripts/backfill_databento_curves.py --cost
  .venv/bin/python scripts/backfill_databento_curves.py --pull
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "curves"
BASE = "https://hist.databento.com/v0"
DATASET = "GLBX.MDP3"
START = "2010-06-06"  # GLBX.MDP3 dataset start
END = dt.date.today().isoformat()
ROOTS = ["CL", "NG", "GC", "SI", "HG", "PL", "PA"]
DEPTH = 8
SCHEMAS = ["ohlcv-1d", "statistics"]


def api_key() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DATABENTO_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DATABENTO_API_KEY not found in .env")


def syms(root: str) -> str:
    return ",".join(f"{root}.c.{i}" for i in range(DEPTH))


def get_cost(client: httpx.Client, root: str, schema: str) -> float:
    r = client.get(f"{BASE}/metadata.get_cost", params={
        "dataset": DATASET, "symbols": syms(root), "stype_in": "continuous",
        "schema": schema, "start": START, "end": END})
    r.raise_for_status()
    return float(r.text)


def pull_rank(client: httpx.Client, root: str, rank: int, schema: str) -> pd.DataFrame | None:
    tag = f"{root}_c{rank}_{schema.replace('-', '')}"
    cache = OUT / "databento" / f"{tag}.csv"
    if not cache.exists():
        r = client.get(f"{BASE}/timeseries.get_range", params={
            "dataset": DATASET, "symbols": f"{root}.c.{rank}", "stype_in": "continuous",
            "schema": schema, "start": START, "end": END,
            "encoding": "csv", "pretty_px": "true", "pretty_ts": "true",
            "map_symbols": "true", "compression": "none"})
        if r.status_code != 200:
            print(f"  {tag}: HTTP {r.status_code} {r.text[:120]}", flush=True)
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
        print(f"  {tag}: {len(r.content) / 1e6:.2f} MB", flush=True)
    df = pd.read_csv(cache)
    if df.empty:
        return None
    df["root"] = root
    df["rank"] = rank
    return df


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--cost"
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(auth=(api_key(), ""), timeout=120) as client:
        if mode == "--cost":
            total = 0.0
            for root in ROOTS:
                for schema in SCHEMAS:
                    c = get_cost(client, root, schema)
                    total += c
                    print(f"{root:3s} {schema:11s} ${c:8.4f}", flush=True)
            print(f"TOTAL ${total:.2f} (debit if pulled; this dry-run spends nothing)")
            return
        if mode == "--pull":
            bars, stats = [], []
            for root in ROOTS:
                for rank in range(DEPTH):
                    b = pull_rank(client, root, rank, "ohlcv-1d")
                    if b is not None:
                        bars.append(b)
                    s = pull_rank(client, root, rank, "statistics")
                    if s is not None:
                        stats.append(s)
            bar = pd.concat(bars, ignore_index=True)
            scol = "symbol" if "symbol" in bar.columns else "raw_symbol"
            bar["date"] = pd.to_datetime(bar["ts_event"], utc=True, format="mixed").dt.date
            bar = bar[["date", "root", "rank", scol, "open", "high", "low", "close", "volume"]]
            bar = bar.rename(columns={scol: "contract"})
            bar.to_parquet(OUT / "curve_ohlcv1d.parquet", index=False)
            st = pd.concat(stats, ignore_index=True)
            st["date"] = pd.to_datetime(st["ts_event"], utc=True, format="mixed").dt.date
            keep = [c for c in ("date", "root", "rank", "stat_type", "price", "quantity")
                    if c in st.columns]
            st[keep].to_parquet(OUT / "curve_statistics.parquet", index=False)
            print(bar.groupby("root")["date"].agg(["min", "max", "count"]).to_string(), flush=True)
            print(f"CURVES DONE: {len(bar)} bar-rows, {len(st)} stat-rows -> {OUT}", flush=True)
            return
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
