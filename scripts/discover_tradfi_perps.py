#!/usr/bin/env python
"""Commodity-perp DISCOVERY (PAPER.md §11.4): keep the commodity universe + lake tracking the venue.

Weekly timer (deploy/convexity-commodity-discovery.timer). What it does:

  1. Pull the venue's current exchangeInfo and select every TRADING perp with
     ``underlyingType == "COMMODITY"`` (Binance marks its TradFi commodity perps
     ``contractType=TRADIFI_PERPETUAL`` — the empirically verified predicate, 2026-07-05).
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


def venue_commodity_perps() -> dict[str, dict]:
    """symbol -> {filters, onboard} for every TRADING venue commodity perp."""
    r = httpx.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=30)
    r.raise_for_status()
    out: dict[str, dict] = {}
    for s in r.json()["symbols"]:
        if (s.get("underlyingType") != "COMMODITY" or s.get("status") != "TRADING"
                or s.get("contractType") != "TRADIFI_PERPETUAL"):
            continue  # dated/delivery commodity futures are NOT this book's instrument
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

    venue = venue_commodity_perps()
    u = load_universe()  # committed block + current overlay
    known = {d["symbol"]: d for d in u["commodities"]}

    new = sorted(set(venue) - set(known))
    drift = {
        s: {k: (known[s].get(k), venue[s][k])
            for k in ("tick_size", "lot_size", "min_qty", "min_notional")
            if known[s].get(k) is not None and float(known[s][k]) != venue[s][k]}
        for s in set(venue) & set(known)
    }
    drift = {s: d for s, d in drift.items() if d}
    delisted = sorted(set(known) - set(venue))

    print(f"[discover] venue commodity perps={len(venue)} known={len(known)} "
          f"new={new or 'none'} filter_drift={list(drift) or 'none'} "
          f"venue_missing={delisted or 'none'}")

    if new and args.dry_run:
        print(f"[discover] DRY-RUN: would add {new} to {OVERLAY} and backfill — no writes")
    elif new:
        overlay = json.loads(OVERLAY.read_text()) if OVERLAY.exists() else {"commodities": []}
        have = {d["symbol"] for d in overlay.get("commodities", [])}
        for sym in new:
            if sym in have:
                continue
            overlay.setdefault("commodities", []).append(
                {"symbol": sym, "sector": "commodity-discovered", "source": "binance",
                 **venue[sym]})
        OVERLAY.parent.mkdir(parents=True, exist_ok=True)
        tmp = OVERLAY.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(overlay, indent=1))
        tmp.rename(OVERLAY)
        _journal("discovered", symbols=new, detail={s: venue[s] for s in new})
        print(f"[discover] overlay updated: {OVERLAY} (+{len(new)}) — data+universe only, "
              f"NEVER auto-traded (RV PAIRS are frozen research config)")

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
        for sym in new:
            n = stitch_perp_only(sym)
            _journal("stitched", symbol=sym, rows=n)
            print(f"[discover] stitched {sym}: {n} rows (perp-only)")
            if n == 0:
                failed.append(sym)
        if failed:
            overlay = json.loads(OVERLAY.read_text())
            overlay["commodities"] = [d for d in overlay.get("commodities", [])
                                      if d["symbol"] not in failed]
            tmp = OVERLAY.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(overlay, indent=1))
            tmp.rename(OVERLAY)
            _journal("rolled_back", symbols=failed)
            print(f"[discover] ROLLED BACK (no stitched data yet, retried next week): {failed}")

    if drift:
        if not args.dry_run:
            _journal("filter_drift", detail=drift)
        print(f"[discover] FILTER DRIFT (venue revised; live re-quantizes anyway): {drift}")
    if delisted:
        if not args.dry_run:
            _journal("venue_missing", symbols=delisted)
        print(f"[discover] known symbols missing from venue (halted/delisted?): {delisted} — "
              f"the listed mask + orphan sweep handle the position side; investigate")

    # R7 disclosure: the crypto universe is FROZEN by design — report drift, never act on it.
    r = httpx.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=30).json()
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
