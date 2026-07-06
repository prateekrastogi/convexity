#!/usr/bin/env python
"""Fast candle-mode Dukascopy backfill for ALL equity underlyings from 2017 (equity branch, data
acquisition). Reuses the verified candle machinery (backfill_dukascopy_hist.candles_day/self_check)
and the tick fallback + constants (backfill_dukascopy_1m); adds equity day-handling (weekday RTH
only) + a 2017 floor + a disk guard.

  cd /home/speculation/convexity-equity
  nohup .venv-data-or-venv/bin/python scripts/backfill_equity_hist.py \
      >> data/backfill_equity_hist.out 2>&1 </dev/null &

Resumable/idempotent: one parquet per (code, day); existing files skip; weekends write an empty
marker without a network call. NEVER restart to "check on it" — tail the .out log instead.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIVERSE_MD = REPO / "src" / "convexity" / "panel" / "candidates.md"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


duka = _load("backfill_dukascopy_1m")           # LAKE, POINT, UA, _do_day, log
hist = _load("backfill_dukascopy_hist")          # candles_day, self_check (reuse duka internally)

FLOOR = dt.date(2017, 1, 1)                      # Dukascopy equity history floor
END = dt.date.today() - dt.timedelta(days=1)     # yesterday (today incomplete)
WORKERS = 20
DISK_ABORT_GB = 8.0


def equity_codes() -> list[str]:
    """Dukascopy codes for equities that HAVE an underlying proxy (dukascopy != null)."""
    block = re.search(r"```json\n(.*?)\n```", UNIVERSE_MD.read_text(), re.S).group(1)
    u = json.loads(block)
    return sorted({d["dukascopy"] for d in u["equities"] if d.get("dukascopy")})


def _do_day_equity(client: httpx.Client, code: str, day: dt.date):
    """Candle-mode one equity day: weekends -> empty marker (no fetch); weekday -> BID+ASK candles."""
    out = duka.LAKE / code / f"{code}-1m-{day:%Y-%m-%d}.parquet"
    if out.exists():
        return "skip", 0
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["open_time", "open", "high", "low", "close", "volume", "ticks"]
    if day.weekday() >= 5:  # equities closed Sat/Sun
        pd.DataFrame(columns=cols).to_parquet(out, index=False)
        return "empty", 0
    bars = hist.candles_day(client, code, day)   # None => no data (holiday / pre-listing)
    if bars is None or bars.empty:
        pd.DataFrame(columns=cols).to_parquet(out, index=False)
        return "empty", 0
    tmp = out.with_suffix(".parquet.tmp")
    bars.to_parquet(tmp, index=False)
    tmp.rename(out)
    return "ok", len(bars)


def _free_gb() -> float:
    # measure the ACTUAL write target (resolves through the shared-lake symlink) so the guard
    # watches the mount data lands on, not a hardcoded parent that may be a different filesystem.
    return shutil.disk_usage(duka.LAKE).free / 1e9


def main() -> None:
    codes = equity_codes()
    print(f"START equity hist backfill: {len(codes)} codes x {FLOOR}..{END} "
          f"(candle mode, workers={WORKERS}); free={_free_gb():.1f}G", flush=True)
    if _free_gb() < DISK_ABORT_GB:
        print(f"ABORT: only {_free_gb():.1f}G free (< {DISK_ABORT_GB}G)", flush=True)
        return

    # per-code mode: candles by default; only fall to ticks if self_check EXPLICITLY fails on a
    # code that already has tick-built days (new codes have none -> trust candles, verified <=1bp).
    with httpx.Client(headers={"User-Agent": duka.UA}) as probe:
        modes = {}
        for code in codes:
            has_existing = (duka.LAKE / code).exists() and any(
                (duka.LAKE / code).glob(f"{code}-1m-*.parquet"))
            if not has_existing:
                modes[code] = "candles"
            else:
                try:
                    modes[code] = "candles" if hist.self_check(probe, code) else "ticks"
                except Exception as e:  # noqa: BLE001
                    print(f"self_check {code} errored ({e!r}) -> candles", flush=True)
                    modes[code] = "candles"
    n_tick = sum(v == "ticks" for v in modes.values())
    print(f"modes resolved: {len(codes)-n_tick} candles, {n_tick} ticks", flush=True)

    grid = []
    d = FLOOR
    while d <= END:
        for code in codes:
            grid.append((code, d))
        d += dt.timedelta(days=1)
    print(f"grid: {len(grid)} code-days", flush=True)

    ok = skip = empty = err = bars = 0
    with httpx.Client(headers={"User-Agent": duka.UA}) as client, \
            ThreadPoolExecutor(WORKERS) as ex:
        def work(code, day):
            if modes[code] == "candles":
                return _do_day_equity(client, code, day)
            return duka._do_day(client, code, day, True)  # tick fallback, is_equity=True
        futs = {ex.submit(work, code, day): (code, day) for code, day in grid}
        for i, f in enumerate(as_completed(futs), 1):
            code, day = futs[f]
            try:
                tag, n = f.result()
                if tag == "ok":
                    ok += 1; bars += n
                elif tag == "skip":
                    skip += 1
                else:
                    empty += 1
            except Exception as e:  # noqa: BLE001
                err += 1
                duka.log.error(f"equity_hist {code} {day} failed: {e}")
            if i % 2000 == 0:
                free = _free_gb()
                print(f"progress {i}/{len(grid)} ok={ok} skip={skip} empty={empty} err={err} "
                      f"bars={bars/1e6:.2f}M free={free:.1f}G", flush=True)
                if free < DISK_ABORT_GB:
                    print(f"ABORT mid-run: free {free:.1f}G < {DISK_ABORT_GB}G", flush=True)
                    return
    print(f"EQUITY HIST DONE ok={ok} skip={skip} empty={empty} err={err} "
          f"bars={bars/1e6:.2f}M free={_free_gb():.1f}G", flush=True)


if __name__ == "__main__":
    main()
