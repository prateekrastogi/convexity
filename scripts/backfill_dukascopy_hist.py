#!/usr/bin/env python
"""Backward-extend the Dukascopy commodity lake for the SWING xsec program (probe-set floors).

Reuses the committed backfill module's fetch/aggregate/resume machinery UNCHANGED (loaded by path);
only the (code, day) grid differs: per-code historical floors (probed 2026-07-04) up to the
existing lake's 2021-06-01 START. XPT/XPD have no pre-2021 Dukascopy history (probed) — skipped.
Same politeness (spacing lives inside _do_day), same one-parquet-per-day resume semantics.

  nohup .venv/bin/python scripts/backfill_dukascopy_hist.py >> data/backfill_duka_hist.out 2>&1 &
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("duka", REPO / "scripts" / "backfill_dukascopy_1m.py")
duka = importlib.util.module_from_spec(spec)
sys.modules["duka"] = duka
spec.loader.exec_module(duka)
# 2x the module's conservative default: this run is 5 codes (not 35 instruments), and per-thread
# request spacing is unchanged, so aggregate rate ~doubles; Dukascopy 503s (if any) self-regulate
# via the module's existing backoff, and errored days are simply retried on the next run.
duka.WORKERS = 12

FLOORS = {  # probed earliest-with-data (2026-07-04); generous by <=1yr is cheap, wrong by 3yr is not
    "XAUUSD": dt.date(2009, 1, 1),
    "XAGUSD": dt.date(2008, 6, 1),
    "LIGHTCMDUSD": dt.date(2013, 1, 1),
    "GASCMDUSD": dt.date(2013, 1, 1),
    "COPPERCMDUSD": dt.date(2012, 1, 1),
}
END = dt.date(2021, 5, 31)  # the existing lake starts 2021-06-01


def main() -> None:
    grid = []
    for code, floor in FLOORS.items():
        d = floor
        while d <= END:
            grid.append((code, d))
            d += dt.timedelta(days=1)
    print(f"START hist extension: {len(FLOORS)} codes, {len(grid)} code-days "
          f"(resume: existing files skipped)", flush=True)
    ok = skip = empty = err = bars = 0
    with httpx.Client(headers={"User-Agent": duka.UA}) as client, \
            ThreadPoolExecutor(duka.WORKERS) as ex:
        futs = {ex.submit(duka._do_day, client, code, day, False): (code, day)
                for code, day in grid}
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
            if i % 500 == 0:
                msg = (f"hist progress {i}/{len(futs)} ok={ok} skip={skip} empty={empty} "
                       f"err={err} bars={bars / 1e6:.2f}M")
                duka.log.info(msg)
                print(msg, flush=True)
    print(f"HIST DONE ok={ok} skip={skip} empty={empty} err={err} bars={bars / 1e6:.2f}M", flush=True)


if __name__ == "__main__":
    main()
