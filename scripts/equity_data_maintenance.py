#!/usr/bin/env python
"""Equity data maintenance (PAPER.md §12.3) — its own timer, NEVER blocks trading.

Two jobs, both idempotent, both degrade gracefully (trading proceeds on last-good data):

(a) NEW-SYMBOL DUKASCOPY BACKFILL: any equities-sleeve entry with a ``dukascopy`` mapping but no
    stitched parquet gets its underlying history backfilled (reusing the established
    backfill_equity_hist machinery: candle mode, ~politeness-spaced, resumable), then stitched
    (stitch_tradfi_1m's stitch_one: source tagging, seam ratio-adjust, split detection/overrides)
    and VALIDATED (max-move check below). A name whose mapping cannot be resolved or whose
    validation fails is FLAGGED for human review — never silently traded or silently dropped.
    Newly-stitched names then mature through the book's generic 260d warmup before trading.
    A new symbol WITHOUT a dukascopy mapping is expected (perp-only listings) and simply noted:
    the weekly lake refresh (backfill_tradfi_perp_1m + stitch) grows its perp history until the
    warmup admits it.

(b) SPLIT/CORPORATE-ACTION WATCH: scan every equity name's recent stitched daily closes; any
    |1-day move| > EQUITY_SPLIT_WATCH_THRESHOLD (25%, the stitch-validation convention) within
    the lookback flags the name in data/universe/equity_flags.json — the rebalance excludes
    flagged names until an operator clears the flag (sets "cleared": true). This is the
    AAPL-2020-class failure mode (unadjusted split → fake −74% day poisoning trend signals),
    contained at the data layer. Genuine >25% moves (earnings blowups) can flag too — that is
    accepted: a human confirms and clears, typically within a day, and the book simply holds the
    prior week's position meanwhile.

Binance-side incremental updates for EXISTING symbols are NOT this script's job — the weekly
lake-refresh chain already appends perp klines and re-stitches (recon-verified, PAPER.md §12.3).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from convexity.paper.fleetconfig import (  # noqa: E402
    EQUITY_FLAGS_PATH,
    EQUITY_SPLIT_WATCH_THRESHOLD,
)

STITCHED = REPO / "data" / "stitched"
FLAGS = REPO / EQUITY_FLAGS_PATH
UNIVERSE_MD = REPO / "src" / "convexity" / "panel" / "candidates.md"
SPLIT_WATCH_LOOKBACK_DAYS = 10  # scan the freshest tail only; older history was validated at stitch
MAINT_LOG = REPO / "data" / "universe" / "equity_datamaint_log.jsonl"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _log(event: str, **payload) -> None:
    MAINT_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": int(time.time() * 1000), "event": event, **payload}
    with MAINT_LOG.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
    print(f"[datamaint] {event}: {json.dumps(payload, default=str)[:300]}", flush=True)


def _equities() -> list[dict]:
    """Committed candidates.md block + the discovery overlay (PAPER.md §12.3): discovered
    perp-only equities get split-watch coverage from day one; without a dukascopy mapping the
    new-symbol backfill correctly notes them perp_only and leaves history to the lake refresh."""
    from convexity.panel.candidates import load_universe

    return load_universe()["equities"]


def _load_flags() -> dict:
    if FLAGS.exists():
        try:
            return json.loads(FLAGS.read_text())
        except Exception:
            return {}
    return {}


def _save_flags(flags: dict) -> None:
    FLAGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FLAGS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(flags, indent=1, default=str))
    tmp.rename(FLAGS)


def backfill_new_symbols() -> None:
    """(a) — one new symbol at a time, full pipeline, validated before it can ever trade."""
    eq = _equities()
    missing = [x for x in eq if not (STITCHED / f"{x['symbol']}.parquet").exists()]
    if not missing:
        _log("new_symbols_none")
        return
    flags = _load_flags()
    stitch = _load_script("stitch_tradfi_1m")
    for x in missing:
        sym, code = x["symbol"], x.get("dukascopy")
        if not code:
            _log("new_symbol_perp_only", symbol=sym,
                 note="no dukascopy mapping; perp history accrues via lake refresh + warmup")
            continue
        try:
            _log("new_symbol_backfill_start", symbol=sym, duka_code=code)
            # Reuse the established equity-hist backfill for JUST this code (its per-day cache
            # makes re-runs cheap/resumable; politeness spacing inside the reused machinery).
            beh = _load_script("backfill_equity_hist")
            import httpx

            days_done = 0
            with httpx.Client(headers={"User-Agent": beh.duka.UA}, timeout=30) as client:
                day = beh.FLOOR
                while day <= beh.END:
                    status, _n = beh._do_day_equity(client, code, day)
                    if status == "ok":
                        days_done += 1
                        time.sleep(1.0)  # politeness only on real fetches; skips are free
                    day += dt.timedelta(days=1)
            _log("new_symbol_backfill_done", symbol=sym, real_days_fetched=days_done)
            inst = next(i for i in stitch.universe() if i["symbol"] == sym)
            m = stitch.stitch_one(inst)
            _log("new_symbol_stitched", symbol=sym, manifest={k: m.get(k) for k in
                 ("rows", "span", "splits", "ratio_adj", "overlap_min", "note")})
            ok, reason = validate_symbol(sym)
            if not ok:
                flags[sym] = {"reason": f"new-symbol validation failed: {reason}",
                              "flagged_ts": int(time.time() * 1000), "cleared": False}
                _log("new_symbol_flagged", symbol=sym, reason=reason)
            else:
                _log("new_symbol_validated", symbol=sym)
        except Exception as e:  # noqa: BLE001 — one bad symbol must not sink the rest
            flags[sym] = {"reason": f"backfill/stitch error: {e!r}"[:300],
                          "flagged_ts": int(time.time() * 1000), "cleared": False}
            _log("new_symbol_error", symbol=sym, error=repr(e)[:300])
    _save_flags(flags)


def validate_symbol(sym: str, lookback_days: int | None = None) -> tuple[bool, str]:
    """Max-single-day-move check on the stitched daily closes (gap_before-aware: a flagged
    session/splice gap is not a same-session move and is skipped, the stitch-validation
    convention)."""
    fp = STITCHED / f"{sym}.parquet"
    if not fp.exists():
        return False, "no stitched parquet"
    df = pd.read_parquet(fp, columns=["open_time", "close", "gap_before"])
    if len(df) == 0:
        return False, "empty parquet"
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    day = ts.dt.floor("D")
    daily = df.assign(day=day.to_numpy()).groupby("day").agg(
        close=("close", "last"), gap=("gap_before", "max"))
    if lookback_days:
        daily = daily.iloc[-(lookback_days + 1):]
    ret = daily["close"].pct_change().to_numpy()
    gap = daily["gap"].to_numpy(bool)
    bad = np.where(np.isfinite(ret) & (np.abs(ret) > EQUITY_SPLIT_WATCH_THRESHOLD) & ~gap)[0]
    if len(bad):
        worst = int(bad[np.argmax(np.abs(ret[bad]))])
        return False, (f"|1d move| {ret[worst]:+.1%} on "
                       f"{daily.index[worst].date()} > {EQUITY_SPLIT_WATCH_THRESHOLD:.0%}")
    return True, "ok"


def split_watch() -> None:
    """(b) — recent-tail scan over every equity name already stitched."""
    flags = _load_flags()
    n_scanned = n_flagged = 0
    for x in _equities():
        sym = x["symbol"]
        if not (STITCHED / f"{sym}.parquet").exists():
            continue
        if sym in flags and not flags[sym].get("cleared", False):
            continue  # already flagged, awaiting operator
        n_scanned += 1
        ok, reason = validate_symbol(sym, lookback_days=SPLIT_WATCH_LOOKBACK_DAYS)
        if not ok:
            flags[sym] = {"reason": f"split-watch: {reason}",
                          "flagged_ts": int(time.time() * 1000), "cleared": False}
            n_flagged += 1
            _log("split_watch_flagged", symbol=sym, reason=reason)
    _save_flags(flags)
    active = sorted(s for s, m in flags.items() if not m.get("cleared", False))
    _log("split_watch_done", scanned=n_scanned, newly_flagged=n_flagged, active_flags=active)


def autopromote_equities() -> None:
    """Weekly dukascopy auto-promotion of discovered perp-only equities (owner decision
    2026-07-10, PAPER.md §12.3): probe <TICKER>USUSD; a hit backfills ~450 days of underlying
    history and re-stitches — the name enters the ensemble the FOLLOWING Saturday (this timer
    runs after the morning rebalance: a one-week veto window). Validation failures are flagged
    like any other data problem; probe misses (true IPOs) stay on the perp-only warmup path."""
    ap = _load_script("duka_autopromote")
    promoted, failed = ap.autopromote(
        "equities", window_days=450, max_promotions=2, log=_log, validator=validate_symbol)
    if failed:
        flags = _load_flags()
        for sym in failed:
            flags[sym] = {"reason": "auto-promotion validation failed (see datamaint log)",
                          "flagged_ts": int(time.time() * 1000), "cleared": False}
        _save_flags(flags)
    _log("autopromote_done", promoted=promoted, failed_validation=failed)


def main() -> None:
    _log("run_start")
    try:
        autopromote_equities()
    except Exception as e:  # noqa: BLE001
        _log("autopromote_phase_error", error=repr(e)[:300])
    try:
        backfill_new_symbols()
    except Exception as e:  # noqa: BLE001
        _log("backfill_phase_error", error=repr(e)[:300])
    try:
        split_watch()
    except Exception as e:  # noqa: BLE001
        _log("split_watch_phase_error", error=repr(e)[:300])
    _log("run_end")


if __name__ == "__main__":
    main()
