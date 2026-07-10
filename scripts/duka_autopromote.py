#!/usr/bin/env python
"""Weekly dukascopy AUTO-PROMOTION of discovered perp-only names (owner decision 2026-07-10).

Discovered names (data/universe/discovered.json overlay) start perp-only. This module probes
dukascopy for the underlying; a hit sets the overlay's ``dukascopy`` mapping, deep-backfills the
underlying candles into the lake, and re-stitches proxy+perp — making the name signal-ready
without waiting out the perp-only warmup:

  * EQUITIES (called from equity_data_maintenance, Sat 05:30 UTC): backfill window ~450 calendar
    days ≥ the 260-trading-day warmup + buffer. Because datamaint runs AFTER that morning's
    rebalance, a promoted name first trades the FOLLOWING Saturday — a one-week veto window
    (owner reviews the alert; leveraged/inverse/decay ETFs go to EQUITY_ETF_EXCLUSIONS).
    Validation (split/max-move) runs before the name can ever trade; failures are FLAGGED.
  * COMMODITIES (called from discover_tradfi_perps, Fri 12:00 UTC): full 2017 depth — RESEARCH
    readiness only. The RV book trades frozen pre-registered PAIRS; a new commodity NEVER
    auto-trades, promotion just means years of proxy history exist when research promotes it.
  * Names dukascopy does not carry (true IPOs, KR_EQUITY, PREMARKET) probe-miss and stay
    perp-only — they mature through the warmup naturally ("only those which we don't have any
    data need the full year", owner 2026-07-10).

Probing is cheap (a few candle days); backfills are politeness-paced and capped per run
(MAX bounds the unit's TimeoutStartSec); everything is idempotent — the per-day parquet cache
makes re-runs skip-fast, and the overlay mapping is written only after a verified probe."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import pathlib
import sys
import time

import httpx

REPO = pathlib.Path(__file__).resolve().parents[1]
OVERLAY = REPO / "data" / "universe" / "discovered.json"

# dukascopy datafeed code candidates per sleeve (committed candidates.md provenance):
# equities: NVDAUSUSD-style; commodities: XAUUSD (metal spot) / COPPERCMDUSD-style + renames.
COMMODITY_ALIASES = {"CL": "LIGHTCMD", "NATGAS": "GASCMD", "BZ": "BRENTCMD"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def code_candidates(symbol: str, sleeve: str) -> list[str]:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if sleeve == "equities":
        return [f"{base}USUSD"]
    alias = COMMODITY_ALIASES.get(base)
    out = [f"{alias}USD"] if alias else []
    return out + [f"{base}CMDUSD", f"{base}USD"]


def _recent_weekdays(n: int, back_from: dt.date | None = None) -> list[dt.date]:
    d = back_from or (dt.date.today() - dt.timedelta(days=2))
    out: list[dt.date] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def probe(code: str, client: httpx.Client, hist, days: int = 6, need: int = 2) -> bool:
    """True iff dukascopy serves candle data for ``code`` on >= ``need`` recent weekdays."""
    hits = 0
    for day in _recent_weekdays(days):
        try:
            if hist.candles_day(client, code, day) is not None:
                hits += 1
                if hits >= need:
                    return True
        except Exception:  # noqa: BLE001 — a transport blip is a miss, not a crash
            pass
        time.sleep(0.4)
    return False


def backfill_window(code: str, start: dt.date, client: httpx.Client, beh) -> int:
    """Politeness-paced candle backfill [start .. yesterday] into the duka lake (idempotent:
    existing per-day parquets skip free)."""
    ok = 0
    day = start
    end = dt.date.today() - dt.timedelta(days=1)
    while day <= end:
        status, _n = beh._do_day_equity(client, code, day)
        if status == "ok":
            ok += 1
            time.sleep(0.4)
        day += dt.timedelta(days=1)
    return ok


def _update_overlay(sleeve: str, symbol: str, code: str) -> None:
    overlay = json.loads(OVERLAY.read_text())
    for e in overlay.get(sleeve, []):
        if e["symbol"] == symbol:
            e["dukascopy"] = code
            e["promoted_ts"] = int(time.time() * 1000)
            break
    else:
        raise RuntimeError(f"{symbol} not in overlay[{sleeve}] — refusing to promote")
    tmp = OVERLAY.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(overlay, indent=1))
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(OVERLAY)


def _push(msg: str) -> None:
    """Best-effort operator push (same channels as fleet_alert.sh); never raises."""
    try:
        topic = os.environ.get("NTFY_TOPIC")
        if topic:
            httpx.post(f"https://ntfy.sh/{topic}", content=msg, timeout=10)
        tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if tok and chat:
            httpx.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                       data={"chat_id": chat, "text": msg}, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def autopromote(sleeve: str, *, window_days: int | None, max_promotions: int,
                log, validator=None) -> tuple[list[str], list[str]]:
    """Probe + promote up to ``max_promotions`` unmapped overlay names of ``sleeve``.

    ``window_days=None`` -> full depth from the 2017 equity-history floor (commodities).
    ``validator(symbol) -> (ok, reason)`` runs after the stitch (equities: split/max-move).
    Returns (promoted, failed_validation) — the caller owns the flags file for the latter."""
    if not OVERLAY.exists():
        return [], []
    entries = [e for e in json.loads(OVERLAY.read_text()).get(sleeve, [])
               if not e.get("dukascopy")]
    if not entries:
        log("autopromote_none", sleeve=sleeve)
        return [], []
    beh = _load("backfill_equity_hist")
    hist = sys.modules["backfill_dukascopy_hist"]  # loaded by beh
    duka = sys.modules["backfill_dukascopy_1m"]
    stitch = _load("stitch_tradfi_1m")
    promoted: list[str] = []
    failed_validation: list[str] = []
    start = (dt.date(2017, 1, 1) if window_days is None
             else dt.date.today() - dt.timedelta(days=window_days))
    with httpx.Client(headers={"User-Agent": duka.UA}, timeout=30) as client:
        for e in entries:
            if len(promoted) >= max_promotions:
                log("autopromote_deferred", sleeve=sleeve,
                    symbols=[x["symbol"] for x in entries if x["symbol"] not in promoted],
                    note=f"max {max_promotions}/run — next weekly run continues")
                break
            sym = e["symbol"]
            hit = next((c for c in code_candidates(sym, sleeve)
                        if probe(c, client, hist)), None)
            if hit is None:
                log("autopromote_probe_miss", sleeve=sleeve, symbol=sym,
                    tried=code_candidates(sym, sleeve),
                    note="no dukascopy data (true IPO / non-US) — perp-only warmup path")
                continue
            log("autopromote_hit", sleeve=sleeve, symbol=sym, code=hit,
                window_start=str(start))
            n_days = backfill_window(hit, start, client, beh)
            _update_overlay(sleeve, sym, hit)
            inst = next(i for i in stitch.universe() if i["symbol"] == sym)
            m = stitch.stitch_one(inst)
            log("autopromote_stitched", sleeve=sleeve, symbol=sym, code=hit,
                days_fetched=n_days, manifest={k: m.get(k) for k in
                ("rows", "span", "splits", "ratio_adj", "overlap_min", "note")})
            if validator is not None:
                ok, reason = validator(sym)
                if not ok:
                    failed_validation.append(sym)
                    log("autopromote_validation_failed", sleeve=sleeve, symbol=sym,
                        reason=reason)
                    _push(f"convexity: {sym} auto-promotion FAILED validation ({reason}) — "
                          f"flagged, will not trade")
                    promoted.append(sym)  # counts against the cap: the work was done
                    continue
            promoted.append(sym)
            _push(f"convexity: {sym} AUTO-PROMOTED ({sleeve}, dukascopy {hit}, "
                  f"{n_days} days) — "
                  + ("first trades NEXT Saturday; veto via EQUITY_ETF_EXCLUSIONS/flags"
                     if sleeve == "equities" else
                     "research/pair-ready only (RV PAIRS stay frozen)"))
    return promoted, failed_validation
