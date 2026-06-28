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
UNIVERSE_MD = REPO / "src" / "convexity" / "selection" / "UNIVERSE.md"
KLINES = REPO / "data" / "klines"
DUKA = REPO / "data" / "dukascopy"
OUT = REPO / "data" / "stitched"
CORE = ["open_time", "open", "high", "low", "close", "volume"]

SPLIT_FACTORS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20]  # candidate integer split ratios
SPLIT_TOL = 0.03      # day-boundary close ratio must be within 3% of a clean factor
SPLIT_MIN = 1.5       # ...and the move must be large (real earnings gaps are rarely > 1.5x)
MIN_OVERLAP = 300     # min common minutes to trust the perp/duka ratio-adjust
GAP_MIN = 1.5         # > this many minutes between consecutive bars => a session/weekend gap


def universe() -> list[dict]:
    block = re.search(r"```json\n(.*?)\n```", UNIVERSE_MD.read_text(), re.S).group(1)
    d = json.loads(block)
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


def _detect_splits(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Day-boundary close ratios near a clean integer factor => a split on the later day."""
    if df.empty:
        return []
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    daily = df.assign(day=day).groupby("day")["close"].last()
    splits = []
    prev = None
    for d, c in daily.items():
        if prev is not None and c > 0:
            r = prev / c            # forward split (price drops) => r ~ factor
            for f in SPLIT_FACTORS:
                if r >= SPLIT_MIN and abs(r - f) / f <= SPLIT_TOL:
                    splits.append((d, f))
                    break
        prev = c
    return splits


def _split_adjust(df: pd.DataFrame, splits: list[tuple[str, int]]) -> pd.DataFrame:
    """Back-adjust: bars BEFORE a split are divided by the cumulative later-split factor."""
    if df.empty or not splits:
        return df
    day = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    divisor = pd.Series(1.0, index=df.index)
    for sdate, f in splits:
        divisor[day.values < sdate] *= f
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
        if sleeve == "equities":
            splits = _detect_splits(duka)
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
