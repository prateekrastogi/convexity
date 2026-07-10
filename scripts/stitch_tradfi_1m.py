#!/usr/bin/env python
"""Stitch the convexity universe into one continuous, analysis-ready 1-min series per instrument.

Two cases, one output schema (data/stitched/{SYMBOL}.parquet):

  (A) Dukascopy-underlying + Binance-perp (35 TradFi names, src=dukascopy+binance):
      1. consolidate the Dukascopy daily shards -> continuous series (sort, dedup);
      2. SPLIT-ADJUST equities (Dukascopy equity prices are RAW: NVDA 1204->121 at its 2024-06-10
         10:1, TSLA 900->291 at its 3:1 — a naive splice feeds the backtest a fake -90% crash).
         Splits are detected from day-boundary close ratios near a clean integer factor and the
         pre-split prices are back-adjusted onto the current share basis (volume scaled inversely);
      3. RATIO-ADJUST the underlying to the perp over their overlap (adj = median(perp/duka)),
         removing the units/basis level difference so the splice has no jump;
      4. SPLICE: adjusted underlying BEFORE the perp's first bar, real perp from there on.

  (B) perp-only (50 crypto + SONY/NVO, src=binance): just consolidate the klines shards.

Then for both: a `gap_before` flag marks the first bar after any non-contiguous (>1.5 min) step —
session/weekend/holiday gaps in the Dukascopy underlying (equities are session-bound, commodities
~24x5) that the backtest must NOT treat as a tradeable 1-min return. Bars are kept, not dropped.

Output schema: open_time(ms,int64) | open high low close volume(float) | source('duka'|'perp') | gap_before(bool)
Also writes data/stitched/_manifest.json (per-instrument provenance: spans, overlap, ratio, splits, seam).

Run (on the VM, data venv):  cd ~/convexity && .venv-data/bin/python scripts/stitch_tradfi_1m.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import re

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIVERSE_MD = REPO / "src" / "convexity" / "panel" / "candidates.md"
KLINES = REPO / "data" / "klines"
DUKA = REPO / "data" / "dukascopy"
OUT = REPO / "data" / "stitched"
CORE = ["open_time", "open", "high", "low", "close", "volume"]

SPLIT_FACTORS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20]  # candidate integer split ratios
SPLIT_TOL = 0.03      # day-boundary close ratio must be within 3% of a clean factor
SPLIT_MIN = 1.5       # ...and the move must be large (real earnings gaps are rarely > 1.5x)
MIN_OVERLAP = 300     # min common minutes to trust the perp/duka ratio-adjust
GAP_MIN = 1.5         # > this many minutes between consecutive bars => a session/weekend gap

# Known corrupted windows in the source feed: bars inside these windows are DROPPED before any
# processing (never enter split-detection, ratio-adjust, or the stitched output). METADBN:
# databento XNAS.ITCH mixed in erroneous sub-$100 prints for a large share (up to 96% on the
# worst days) of 1-min bars from 2021-06-30 through 2022-02-02 -- verified 2026-07-06 by direct
# inspection (median/mean/last all land in the erroneous cluster on many days; the erroneous
# prints REVERT day-to-day rather than persisting, ruling out a real corporate action -- a
# systematic per-symbol price-quality scan found this as the one severe case among 102 fetched
# equities; see research/equity/DATA.md).
KNOWN_BAD_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "METADBN": [("2021-06-30", "2022-02-02")],
}

# Real corporate actions verified externally (WebSearch, 2026-07-06) that the automated
# day-boundary-ratio detector missed -- normal price drift around the effective date can push
# the observed ratio outside SPLIT_TOL of the clean factor. Applied in ADDITION to auto-detection
# (union, deduplicated by date).
KNOWN_SPLITS: dict[str, list[tuple[str, int]]] = {
    "KLACDBN": [("2026-06-12", 10)],   # KLA Corp 10-for-1, effective 2026-06-12 (verified)
    "TSLADBN": [("2020-08-31", 5)],    # Tesla 5-for-1, effective 2020-08-31 (well known; the
                                       # observed raw ratio was 4.44 vs the true 5.00, outside
                                       # SPLIT_TOL due to same-day price movement -- same failure
                                       # shape as KLAC, verified 2026-07-06)
    "KORUDBN": [("2025-02-10", -10)],  # Direxion KORU 1-for-10 REVERSE split, split-adjusted
                                       # trading began 2025-02-10 (verified); observed ratio
                                       # 10.54 vs true 10.00, outside SPLIT_TOL for the same
                                       # same-day-price-movement reason as KLAC/TSLA
    "AAPLDBN": [("2020-08-31", 4)],    # Apple 4-for-1, effective 2020-08-31 (same effective date
                                       # as TSLA's -- both announced for that Monday; well known).
                                       # Missed by the detector for the same drift-outside-
                                       # SPLIT_TOL reason; live-caught 2026-07 as a fake -74.12%
                                       # single-day return poisoning trend signals (D1v2 fix).
    "CRWDDBN": [("2026-07-02", 4)],    # CrowdStrike 4-for-1 eff 2026-07-02 — split-watch live
                                       # catch 2026-07-10 (fake -74.8% day); a PERP-ERA split:
                                       # handled by the combined-series pass in stitch_one.
}

# Explicit REJECTIONS: (symbol, date) pairs where the ratio+persistence check finds a clean-ish
# factor that is NOT a real corporate action -- confirmed by direct verification. GME's only real
# split was July 2022 (4:1, calmly detected/adjusted normally); this entry is a coincidental ~2x
# day-over-day jump on 2021-02-24 during a SECOND wave of the 2021 meme-stock squeeze (price kept
# climbing well past 2x over the following week -- genuine sustained momentum, not a one-time
# mechanical reset). A general statistical guard (trailing volatility before the candidate day)
# was tried and rejected: GME's trailing 10-day vol (4.94%) was statistically indistinguishable
# from KLAC's real-split trailing vol (4.83%) -- not a safe basis for a global rule with the
# available calibration data, so this confirmed case is excluded explicitly instead (verified
# 2026-07-06 by tracing daily closes for two weeks either side of the date).
KNOWN_FALSE_SPLITS: set[tuple[str, str]] = {
    ("GMEDBN", "2021-02-24"),
}


def _mask_bad_windows(code: str, df: pd.DataFrame) -> pd.DataFrame:
    windows = KNOWN_BAD_WINDOWS.get(code)
    if not windows or df.empty:
        return df
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    keep = pd.Series(True, index=df.index)
    for start, end in windows:
        keep &= ~((day >= start) & (day <= end))
    return df[keep].reset_index(drop=True)


def universe() -> list[dict]:
    """Merged loader (committed block + the gitignored discovery overlay) so a commodity perp
    discovered mid-cycle keeps getting RE-stitched by the weekly lake refresh — a raw-regex
    parse of candidates.md would freeze it at discovery day (review finding 2026-07-05)."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from convexity.panel.candidates import load_universe
    d = load_universe(UNIVERSE_MD, discovered_path=REPO / "data/universe/discovered.json")
    out = []
    for sleeve in ("commodities", "equities", "crypto"):
        for x in d[sleeve]:
            out.append({"symbol": x["symbol"], "sleeve": sleeve, "duka": x.get("dukascopy")})
    return out


def _load(folder: pathlib.Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(folder / "*.parquet")))
    if not fs:
        return pd.DataFrame(columns=CORE)
    parts = []
    for f in fs:
        df = pd.read_parquet(f, columns=None)
        if len(df) and set(CORE).issubset(df.columns):
            parts.append(df[CORE])
    if not parts:
        return pd.DataFrame(columns=CORE)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["open_time"]).astype({"open_time": "int64"})
    df = df.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    return df


def _detect_splits(df: pd.DataFrame, code: str | None = None) -> list[tuple[str, int]]:
    """Day-boundary close ratios near a clean integer factor => a split on the later day.

    Symmetric: a FORWARD split (price drops, share count up) gives prev/close ~ +factor; a
    REVERSE split (price jumps, share count down -- routine for leveraged/inverse ETFs fighting
    volatility decay: SQQQ/UVXY/SOXL/TQQQ) gives prev/close ~ 1/factor. Both are checked; the
    returned factor is NEGATIVE for a reverse split (encodes the adjustment direction for
    ``_split_adjust``). Missing this direction entirely (live-caught 2026-07-06) left several
    leveraged-ETF names with 300-3700% unadjusted single-day "moves" in the raw series.

    Requires PERSISTENCE: the day AFTER a candidate split must NOT revert toward the pre-split
    level. This rejects single-day bad-tick artifacts that happen to land near a clean ratio by
    chance -- a real split changes the share count permanently and never reverts (live-caught
    case: three spurious ~20x ratios in METADBN's corrupted window each reverted the very next
    day; excluded separately via KNOWN_BAD_WINDOWS, but this check is a general safety net for
    any symbol/period not manually audited). When there's no next day to check (the candidate is
    the last row in the series), the split is accepted -- e.g. CRWDUSDT's real 2026-07-02 split,
    confirmed independently by whole-day price-level consistency.

    Known externally-verified splits the ratio check itself misses (KNOWN_SPLITS -- KLAC's
    2026-06-12 10:1 and TSLA's 2020-08-31 5:1: normal price drift around the effective date
    pushed the raw ratio outside SPLIT_TOL) are unioned in, deduplicated by date.
    """
    if df.empty:
        return sorted(set(KNOWN_SPLITS.get(code, [])))
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    daily = df.assign(day=day).groupby("day")["close"].last()
    days = daily.index.to_numpy()
    closes = daily.to_numpy()
    splits = []
    for i in range(1, len(closes)):
        prev, c = closes[i - 1], closes[i]
        if c <= 0 or prev <= 0:
            continue
        candidates = [(prev / c, 1) for _ in [0]] + [(c / prev, -1)]  # (ratio, sign)
        matched = None
        for r, sign in candidates:
            for f in SPLIT_FACTORS:
                if r >= SPLIT_MIN and abs(r - f) / f <= SPLIT_TOL:
                    matched = sign * f
                    break
            if matched is not None:
                break
        if matched is None:
            continue
        if i + 1 < len(closes):
            nxt = closes[i + 1]
            if abs(nxt - prev) < abs(nxt - c):   # closer to OLD level => reverted, reject
                continue
        if code is not None and (code, str(days[i])) in KNOWN_FALSE_SPLITS:
            continue
        splits.append((str(days[i]), matched))
    splits = sorted(set(splits) | set(KNOWN_SPLITS.get(code, [])))
    return splits


def _split_adjust(df: pd.DataFrame, splits: list[tuple[str, int]]) -> pd.DataFrame:
    """Back-adjust: bars BEFORE a split are divided by the cumulative later-split factor.

    A negative factor (reverse split) means the price JUMPED at the event -- pre-event bars are
    on the OLD (higher-share-count, lower-price) basis and must be brought UP to the current
    basis, i.e. multiplied rather than divided; encoded here as dividing by 1/|f|.
    """
    if df.empty or not splits:
        return df
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    divisor = pd.Series(1.0, index=df.index)
    for sdate, f in splits:
        step = f if f > 0 else (1.0 / abs(f))
        divisor[day.values < sdate] *= step
    df = df.copy()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] / divisor
    df["volume"] = df["volume"] * divisor
    return df


def _gap_flag(df: pd.DataFrame) -> pd.Series:
    dmin = df["open_time"].diff() / 60000.0
    g = dmin > GAP_MIN
    if len(g):
        g.iloc[0] = True
    return g.fillna(True)


def stitch_one(inst: dict) -> dict:
    sym, sleeve, code = inst["symbol"], inst["sleeve"], inst["duka"]
    perp = _load(KLINES / sym)
    note, splits, ratio, overlap, seam = "", [], None, 0, None

    if code:  # (A) dukascopy underlying present
        duka = _load(DUKA / code)
        duka = _mask_bad_windows(code, duka)
        if sleeve == "equities":
            splits = _detect_splits(duka, code)
            duka = _split_adjust(duka, splits)
        if not perp.empty and not duka.empty:
            common = duka.merge(perp[["open_time", "close"]], on="open_time", suffixes=("_d", "_p"))
            overlap = len(common)
            if overlap >= MIN_OVERLAP:
                ratio = float((common["close_p"] / common["close_d"]).median())
                for c in ["open", "high", "low", "close"]:
                    duka[c] = duka[c] * ratio
            else:
                note += f"overlap={overlap}<{MIN_OVERLAP}: ratio NOT applied (level may be off); "
        if not perp.empty:
            seam = int(perp["open_time"].min())
            hist = duka[duka["open_time"] < seam].copy()
            hist["source"] = "duka"
            perp = perp.copy()
            perp["source"] = "perp"
            out = pd.concat([hist[CORE + ["source"]], perp[CORE + ["source"]]], ignore_index=True)
        else:  # perp not (yet) fetched -> underlying only
            note += "no perp leg; underlying-only; "
            out = duka.copy()
            out["source"] = "duka"
    else:  # (B) perp-only (crypto, SONY, NVO)
        out = perp.copy()
        out["source"] = "perp"

    if out.empty:
        return {"symbol": sym, "sleeve": sleeve, "rows": 0, "note": "NO DATA"}

    out = out.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    # PERP-ERA SPLITS (live-caught 2026-07-10: CRWD 4:1 on 2026-07-02): the pass above adjusts
    # only the duka frame, but a split AFTER the perp lists re-bases the PERP price too --
    # back-adjust the COMBINED series for splits dated inside the perp era. Proxy-era splits
    # are excluded (already applied to duka; re-applying would double-adjust). Equities only:
    # crypto has no corporate actions and a crash near a clean factor must never be "adjusted".
    if sleeve == "equities" and not perp.empty:
        perp_start = pd.to_datetime(int(perp["open_time"].min()),
                                    unit="ms", utc=True).strftime("%Y-%m-%d")
        perp_splits = [(d, f) for (d, f) in _detect_splits(perp, code) if d >= perp_start]
        if perp_splits:
            out = _split_adjust(out, perp_splits)
            splits = sorted(set(splits) | set(perp_splits))
            note += f"perp-era splits applied: {perp_splits}; "
    out["gap_before"] = _gap_flag(out)
    OUT.mkdir(parents=True, exist_ok=True)
    out[CORE + ["source", "gap_before"]].to_parquet(OUT / f"{sym}.parquet", index=False)

    def span(d):
        if d.empty:
            return None
        return [pd.to_datetime(d["open_time"].iloc[i], unit="ms", utc=True).strftime("%Y-%m-%d") for i in (0, -1)]
    return {
        "symbol": sym, "sleeve": sleeve, "duka_code": code,
        "rows": int(len(out)),
        "span": span(out),
        "perp_span": span(perp) if not perp.empty else None,
        "overlap_min": overlap, "ratio_adj": None if ratio is None else round(ratio, 6),
        "splits": splits,
        "seam": None if seam is None else pd.to_datetime(seam, unit="ms", utc=True).strftime("%Y-%m-%d"),
        "n_gap": int(out["gap_before"].sum()),
        "note": note.strip(),
    }


def main() -> None:
    insts = universe()
    print(f"stitching {len(insts)} instruments -> {OUT}", flush=True)
    manifest = []
    for inst in insts:
        try:
            m = stitch_one(inst)
        except Exception as e:  # noqa: BLE001
            m = {"symbol": inst["symbol"], "sleeve": inst["sleeve"], "rows": 0, "note": f"ERROR {e!r}"}
        manifest.append(m)
        tag = f"{m['symbol']:<12} {m.get('rows',0):>9,} rows  {m.get('span')}"
        extra = []
        if m.get("splits"):
            extra.append(f"splits={m['splits']}")
        if m.get("ratio_adj") is not None:
            extra.append(f"ratio={m['ratio_adj']} (overlap={m.get('overlap_min')})")
        if m.get("note"):
            extra.append(m["note"])
        print(tag + ("  | " + " | ".join(extra) if extra else ""), flush=True)
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(1 for m in manifest if m.get("rows", 0) > 0)
    print(f"\nDONE: {n_ok}/{len(insts)} instruments written to {OUT}; manifest -> {OUT}/_manifest.json", flush=True)


if __name__ == "__main__":
    main()
