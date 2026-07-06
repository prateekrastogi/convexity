#!/usr/bin/env python
"""Databento equity fetch: full 2018->now 1-min per US-listed name from its PRIMARY venue,
RTH-filtered (correct daily close = the 16:00 closing-auction bar), written in the lake schema
so stitch_tradfi_1m.py consumes it unchanged. Single clean source; replaces Dukascopy for equities.

Reads /tmp/dbn_routing_all.json. Resumable (skip written), per-symbol isolation, disk-guarded.
No databento client — REST + encoding=csv. Codes: <TICKER>DBN in data/dukascopy/<CODE>/ so the
existing stitch (which globs DUKA/<code>/*.parquet) picks them up with no code change.
  nohup .venv-data/bin/python scripts/dbn_fetch.py >> data/dbn_fetch.out 2>&1 &
"""
from __future__ import annotations

import io
import json
import shutil
import time
from pathlib import Path

import httpx
import pandas as pd

BASE = "https://hist.databento.com/v0"
REPO = Path("/home/speculation/convexity-equity")
DUKA = Path("/home/speculation/convexity/data/dukascopy")  # shared lake (via symlink target)
ROUTING = json.loads(Path("/tmp/dbn_routing_all.json").read_text())
KEY = next(l.split("=", 1)[1].strip() for l in (REPO / ".env").read_text().splitlines()
           if l.startswith("DATABENTO_API_KEY=")) if (REPO / ".env").exists() else \
    next(l.split("=", 1)[1].strip() for l in Path("/home/speculation/convexity/.env").read_text().splitlines()
         if l.startswith("DATABENTO_API_KEY="))
AUTH = (KEY, "")
START, END = ROUTING["start"], ROUTING["end"]
DISK_ABORT_GB = 8.0
CORE = ["open_time", "open", "high", "low", "close", "volume", "ticks"]


def code_for(ticker: str) -> str:
    return f"{ticker}DBN"


def pull(client, venue, ticker, tries=4):
    """Full-history ohlcv-1m CSV -> (raw df or None, warning or None). Retries transient errors.

    databento answers 206 (not just 200) when the response body is fully valid CSV but the venue
    flags data-quality caveats for specific days (x-warning header) -- e.g. a young listing whose
    requested range starts before its IPO still returns 206 with the real post-IPO rows. Treating
    206 as an error (the original bug, caught live 2026-07-06 on ALAB/Astera Labs) silently wrote
    an EMPTY file for every such name. Any 2xx with a body is accepted; the warning is surfaced
    in the manifest instead of being discarded.
    """
    for i in range(tries):
        try:
            r = client.get(f"{BASE}/timeseries.get_range", params={
                "dataset": venue, "symbols": ticker, "schema": "ohlcv-1m", "stype_in": "raw_symbol",
                "start": START, "end": END, "encoding": "csv", "compression": "none"}, timeout=600)
            if 200 <= r.status_code < 300:
                warn = r.headers.get("x-warning")
                if not r.text.strip():
                    return pd.DataFrame(), warn
                return pd.read_csv(io.StringIO(r.text)), warn
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(3 * (i + 1)); continue
            print(f"    {ticker}@{venue} HTTP{r.status_code}: {r.text[:100]}", flush=True)
            return None, None
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print(f"    {ticker}@{venue} err: {e}", flush=True)
                return None, None
            time.sleep(3 * (i + 1))
    return None, None


def to_lake(raw: pd.DataFrame) -> pd.DataFrame:
    """databento ohlcv-1m raw -> lake bars, RTH-filtered (09:30-16:00 ET, DST-aware)."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CORE)
    import numpy as np
    ts_ns = raw["ts_event"].to_numpy("int64")                  # int ns since epoch UTC
    et = pd.to_datetime(ts_ns, utc=True).tz_convert("America/New_York")
    t = et.time
    keep = (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())
    if not keep.any():
        return pd.DataFrame(columns=CORE)
    out = pd.DataFrame({
        "open_time": ts_ns[keep] // 1_000_000,                 # ns -> ms, straight from int col
        "open": raw["open"].to_numpy(float)[keep] / 1e9,
        "high": raw["high"].to_numpy(float)[keep] / 1e9,
        "low": raw["low"].to_numpy(float)[keep] / 1e9,
        "close": raw["close"].to_numpy(float)[keep] / 1e9,
        "volume": raw["volume"].to_numpy(float)[keep],
        "ticks": 0,
    })
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def main() -> None:
    routing = ROUTING["routing"]
    print(f"START databento equity fetch: {len(routing)} names, {START}..{END}, "
          f"free={shutil.disk_usage(DUKA).free/1e9:.1f}G", flush=True)
    manifest = []
    with httpx.Client(auth=AUTH) as c:
        for i, (perp, r) in enumerate(sorted(routing.items()), 1):
            ticker, venue, hist_tks = r["ticker"], r["venue"], r["hist_tickers"]
            code = code_for(ticker)
            outdir = DUKA / code
            outfile = outdir / f"{code}.parquet"
            if outfile.exists():
                manifest.append({"perp": perp, "code": code, "status": "skip"})
                continue
            if shutil.disk_usage(DUKA).free / 1e9 < DISK_ABORT_GB:
                print(f"ABORT: low disk", flush=True)
                break
            pulled = [pull(c, venue, tk) for tk in hist_tks]              # META: [META, FB]
            warnings_ = [w for _, w in pulled if w]
            frames = [to_lake(raw) for raw, _ in pulled]
            bars = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
                not f.empty for f in frames) else pd.DataFrame(columns=CORE)
            bars = bars.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            outdir.mkdir(parents=True, exist_ok=True)
            tmp = outfile.with_suffix(".parquet.tmp")
            bars.to_parquet(tmp, index=False)
            tmp.rename(outfile)
            span = ("", "")
            if len(bars):
                t = pd.to_datetime(bars["open_time"], unit="ms")
                span = (str(t.iloc[0].date()), str(t.iloc[-1].date()))
            manifest.append({"perp": perp, "code": code, "ticker": ticker, "venue": venue,
                             "rows": len(bars), "first": span[0], "last": span[1],
                             "status": "ok" if len(bars) else "empty",
                             "warnings": warnings_ or None})
            warn_tag = " [QUALITY-WARN]" if warnings_ else ""
            print(f"  [{i}/{len(routing)}] {perp:14s} {code:10s} @{venue} rows={len(bars):>8} "
                  f"{span[0]}..{span[1]} free={shutil.disk_usage(DUKA).free/1e9:.1f}G{warn_tag}",
                  flush=True)
    Path("/tmp/dbn_fetch_manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"DBN FETCH DONE: {ok} ok / {len(manifest)} names; manifest /tmp/dbn_fetch_manifest.json",
          flush=True)


if __name__ == "__main__":
    main()
