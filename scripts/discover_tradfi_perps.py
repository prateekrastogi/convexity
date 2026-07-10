#!/usr/bin/env python
"""Commodity-perp DISCOVERY (PAPER.md §11.4): keep the commodity universe + lake tracking the venue.

Weekly timer (deploy/convexity-commodity-discovery.timer). What it does:

  1. Pull the venue's current exchangeInfo ONCE and select every TRADING
     ``contractType=TRADIFI_PERPETUAL`` with ``underlyingType == "COMMODITY"`` — and, since
     2026-07-10 (owner decision), ``== "EQUITY"`` as well: new stock/ETF perps are onboarded
     data-first exactly like commodities. ``KR_EQUITY``/``PREMARKET`` types are FYI-only by
     design (no liquidity/history validation surface; PREMARKET marks pre-IPO books).
     Discovered EQUITIES are doubly quarantined: perp-only history (no dukascopy mapping unless
     the owner later curates one into candidates.md) means the equity book's 260d warmup blocks
     them for ~a year from listing — review the weekly report for leveraged/inverse/decay ETFs
     (-> EQUITY_ETF_EXCLUSIONS) well before any could mature.
  2. Any symbol not already in the universe (committed candidates.md block + overlay) is APPENDED
     to ``data/universe/discovered.json`` — the gitignored overlay ``load_universe`` merges — with
     its real venue filters + onboard date. From that moment panels, backfills and the commodity
     book's ORPHAN-SWEEP DOMAIN all see it.
  3. For each new symbol the data backfill runs immediately (klines + funding via the existing
     resumable dump scripts) and a perp-only stitched series is written so the panel can read it.
  4. Filter DRIFT on known symbols (venue revised tick/lot) is reported loudly (live sessions
     already re-quantize against current exchangeInfo, so drift never blocks trading).
  5. New CRYPTO perps are counted and named in the report FYI — the crypto book's candidates.md
     is a frozen, survivorship-clean, validated universe; it does NOT auto-grow (R7 disclosure).

DISCOVERED SYMBOLS ARE NEVER AUTO-TRADED: the RV book trades its frozen pre-registered PAIRS only
(signal/rv_pairs.PAIRS). Discovery gives a new perp data + universe membership so that WHEN
research promotes it, years of lake already exist. Every action is journaled to
``data/paper/commodity/discovery.jsonl`` (append-only).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from convexity.panel.candidates import load_universe  # noqa: E402

OVERLAY = REPO / "data" / "universe" / "discovered.json"
DISCOVERY_JOURNAL = REPO / "data" / "paper" / "commodity" / "discovery.jsonl"
KLINES = REPO / "data" / "klines"
STITCHED = REPO / "data" / "stitched"
FAPI = "https://fapi.binance.com"


def _journal(kind: str, **payload) -> None:
    DISCOVERY_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "ts": int(time.time() * 1000), **payload}
    with DISCOVERY_JOURNAL.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def venue_tradfi_perps(underlying: str, info: dict | None = None) -> dict[str, dict]:
    """symbol -> {filters, onboard} for every TRADING venue TRADIFI perp of ``underlying``."""
    if info is None:
        r = httpx.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=30)
        r.raise_for_status()
        info = r.json()
    out: dict[str, dict] = {}
    for s in info["symbols"]:
        if (s.get("underlyingType") != underlying or s.get("status") != "TRADING"
                or s.get("contractType") != "TRADIFI_PERPETUAL"):
            continue  # dated/delivery futures are NOT these books' instrument
        fs = {f["filterType"]: f for f in s["filters"]}
        try:
            out[s["symbol"]] = {
                "tick_size": float(fs["PRICE_FILTER"]["tickSize"]),
                "lot_size": float(fs["LOT_SIZE"]["stepSize"]),
                "min_qty": float(fs["LOT_SIZE"]["minQty"]),
                "min_notional": float(fs.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
                "onboard": int(s.get("onboardDate", 0)),
            }
        except KeyError:
            continue
    return out


def stitch_perp_only(sym: str) -> int:
    """Concat the symbol's kline parquets into a perp-only stitched series (no proxy history —
    a NEW listing has none; the panel's listed mask starts at its first live day). Schema matches
    the existing stitched lake (open_time/open/high/low/close/volume/source/gap_before).
    ``gap_before`` is written False throughout — a perp-only series has no proxy splice; intra-
    session gaps are not annotated here (the daily resample the paper book consumes ignores it)."""
    parts = sorted((KLINES / sym).glob(f"{sym}-1m-*.parquet"))
    if not parts:
        return 0
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    out = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    out["source"] = "binance"
    out["gap_before"] = False
    STITCHED.mkdir(parents=True, exist_ok=True)
    tmp = STITCHED / f"{sym}.parquet.tmp"
    out.to_parquet(tmp, index=False)
    tmp.rename(STITCHED / f"{sym}.parquet")
    return len(out)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write/backfill nothing")
    args = ap.parse_args()

    r = httpx.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=30)
    r.raise_for_status()
    info = r.json()
    u = load_universe()  # committed block + current overlay
    # (sleeve, venue underlyingType, overlay sector tag, never-traded note)
    SLEEVES = (
        ("commodities", "COMMODITY", "commodity-discovered",
         "NEVER auto-traded (RV PAIRS are frozen research config)"),
        ("equities", "EQUITY", "equity-discovered",
         "NEVER auto-traded (perp-only history -> ~1y 260d warmup; review for leveraged/"
         "inverse ETF -> EQUITY_ETF_EXCLUSIONS before maturity)"),
    )
    new_by_sleeve: dict[str, list[str]] = {}
    venue_by_sleeve: dict[str, dict] = {}
    # delisted/halted is judged against EVERYTHING trading on the venue, not the sleeve's
    # underlyingType slice — a known name classified KR_EQUITY/PREMARKET is not "missing"
    all_trading = {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING"}
    for sleeve, underlying, _sector, _note in SLEEVES:
        venue = venue_tradfi_perps(underlying, info)
        known = {d["symbol"]: d for d in u[sleeve]}
        new = sorted(set(venue) - set(known))
        drift = {
            s: {k: (known[s].get(k), venue[s][k])
                for k in ("tick_size", "lot_size", "min_qty", "min_notional")
                if known[s].get(k) is not None and float(known[s][k]) != venue[s][k]}
            for s in set(venue) & set(known)
        }
        drift = {s: d for s, d in drift.items() if d}
        delisted = sorted(set(known) - all_trading)
        print(f"[discover] venue {sleeve} perps={len(venue)} known={len(known)} "
              f"new={new or 'none'} filter_drift={list(drift) or 'none'} "
              f"venue_missing={delisted or 'none'}")
        if drift:
            if not args.dry_run:
                _journal("filter_drift", sleeve=sleeve, detail=drift)
            print(f"[discover] FILTER DRIFT ({sleeve}; live re-quantizes anyway): {drift}")
        if delisted:
            if not args.dry_run:
                _journal("venue_missing", sleeve=sleeve, symbols=delisted)
            print(f"[discover] {sleeve} known symbols missing from venue (halted/delisted?): "
                  f"{delisted} — the listed mask + orphan sweep handle the position side; "
                  f"investigate")
        new_by_sleeve[sleeve] = new
        venue_by_sleeve[sleeve] = venue
    # FYI-only underlying types: never onboarded by design (KR_EQUITY: no validation surface;
    # PREMARKET: pre-IPO marks with no underlying history — not trend material).
    exotic: dict[str, list[str]] = {}
    for s in info["symbols"]:
        if s.get("contractType") == "TRADIFI_PERPETUAL" and s.get("status") == "TRADING":
            ut = str(s.get("underlyingType"))
            if ut not in ("COMMODITY", "EQUITY"):
                exotic.setdefault(ut, []).append(s["symbol"])
    if exotic:
        print(f"[discover] FYI non-onboarded TradFi types (by design): "
              f"{ {k: sorted(v) for k, v in exotic.items()} }")

    new_total = [s for lst in new_by_sleeve.values() for s in lst]
    if new_total and args.dry_run:
        print(f"[discover] DRY-RUN: would add {new_by_sleeve} to {OVERLAY} and backfill — "
              f"no writes")
    elif new_total:
        overlay = json.loads(OVERLAY.read_text()) if OVERLAY.exists() else {}
        for sleeve, _u_, sector, note in SLEEVES:
            have = {d["symbol"] for d in overlay.get(sleeve, [])}
            added = [s for s in new_by_sleeve[sleeve] if s not in have]
            for sym in added:
                overlay.setdefault(sleeve, []).append(
                    {"symbol": sym, "sector": sector, "source": "binance",
                     **venue_by_sleeve[sleeve][sym]})
            if added:
                print(f"[discover] {sleeve} overlay +{len(added)} — data+universe only, {note}")
        OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        tmp = OVERLAY.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(overlay, indent=1))
        tmp.rename(OVERLAY)
        _journal("discovered", by_sleeve=new_by_sleeve,
                 detail={s: venue_by_sleeve[sl][s]
                         for sl, lst in new_by_sleeve.items() for s in lst})

        # backfill the new symbols' data NOW (resumable dump scripts read the merged universe,
        # so the just-written overlay is already in their symbol list), then write perp-only
        # stitched series so the panel sees them.
        py = sys.executable
        for script in ("backfill_tradfi_perp_1m.py", "backfill_funding_1m.py"):
            print(f"[discover] running {script} ...")
            try:
                rc = subprocess.run([py, str(REPO / "scripts" / script)], cwd=REPO,
                                    timeout=7200, check=False).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            _journal("backfill", script=script, returncode=rc)
            if rc != 0:
                print(f"[discover] {script} exited rc={rc} — investigate")
        # a symbol enters the panel ONLY with a stitched series; a failed stitch must ROLL the
        # symbol BACK OUT of the overlay, else every later session's build_panel dies on a
        # missing parquet (review finding 2026-07-05). Backfills are resumable — next weekly
        # run retries the rolled-back symbol from wherever the dumps got to.
        failed = []
        for sym in new_total:
            n = stitch_perp_only(sym)
            _journal("stitched", symbol=sym, rows=n)
            print(f"[discover] stitched {sym}: {n} rows (perp-only)")
            if n == 0:
                failed.append(sym)
        if failed:
            overlay = json.loads(OVERLAY.read_text())
            for sleeve, _u_, _s_, _n_ in SLEEVES:
                overlay[sleeve] = [d for d in overlay.get(sleeve, [])
                                   if d["symbol"] not in failed]
            tmp = OVERLAY.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(overlay, indent=1))
            tmp.rename(OVERLAY)
            _journal("rolled_back", symbols=failed)
            print(f"[discover] ROLLED BACK (no stitched data yet, retried next week): {failed}")

    # R7 disclosure: the crypto universe is FROZEN by design — report drift, never act on it.
    r = info  # same exchangeInfo snapshot fetched above
    venue_crypto = {s["symbol"] for s in r["symbols"]
                    if s.get("underlyingType") == "COIN" and s.get("status") == "TRADING"
                    and s.get("contractType") == "PERPETUAL"}
    known_crypto = {d["symbol"] for d in u["crypto"]}
    new_crypto = sorted(venue_crypto - known_crypto)
    print(f"[discover] FYI crypto: venue has {len(new_crypto)} perps outside the frozen "
          f"{len(known_crypto)}-name candidates universe (by design — no auto-add); "
          f"sample: {new_crypto[:8]}")
    if not args.dry_run:
        _journal("report", n_venue=len(venue), n_new=len(new), n_drift=len(drift),
                 n_crypto_outside=len(new_crypto))


if __name__ == "__main__":
    main()
