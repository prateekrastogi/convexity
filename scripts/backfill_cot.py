#!/usr/bin/env python
"""CFTC Disaggregated COT (futures-only) for the 7 commodity underlyings -> data/cot/cot_disagg.parquet.

Free official archives: fut_disagg_txt_hist_2006_2016.zip (2006-06-13 onward) + fut_disagg_txt_{YYYY}.zip
(2017+). Extracts Managed-Money and Producer/Merchant long/short + open interest per report Tuesday,
maps CME/NYMEX/COMEX market names to our perp symbols, and adds `usable_from` = report date + 3 days
(the Friday ~15:30 ET release) so a signal can never read a report before it was public.
"""
from __future__ import annotations

import datetime as dt
import io
import pathlib
import re
import zipfile

import httpx
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "cot"
BASE = "https://www.cftc.gov/files/dea/history/"
THIS_YEAR = dt.date.today().year
SOURCES = ["fut_disagg_txt_hist_2006_2016.zip"] + [
    f"fut_disagg_txt_{y}.zip" for y in range(2017, THIS_YEAR + 1)]

MARKET_MAP = [  # (regex on Market_and_Exchange_Names, our symbol) — NYMEX/COMEX-anchored ONLY:
    # CFTC renamed several markets over time (Feb-2022: NATURAL GAS->NAT GAS NYME,
    # COPPER-GRADE #1->COPPER- #1; ~2018: CRUDE OIL LIGHT SWEET->WTI-PHYSICAL). The ICE Europe
    # "CRUDE OIL, LIGHT SWEET-WTI" look-alike must NOT match (double-mapping corrupts the panel).
    (r"^(CRUDE OIL, LIGHT SWEET - NEW YORK|WTI-PHYSICAL - NEW YORK)", "CLUSDT"),
    (r"^(NATURAL GAS - NEW YORK|NAT GAS NYME - NEW YORK)", "NATGASUSDT"),
    (r"^GOLD - COMMODITY EXCHANGE", "XAUUSDT"),
    (r"^SILVER - COMMODITY EXCHANGE", "XAGUSDT"),
    (r"^COPPER-\s?(GRADE\s?)?#1 - COMMODITY EXCHANGE", "COPPERUSDT"),
    (r"^PLATINUM - NEW YORK", "XPTUSDT"),
    (r"^PALLADIUM - NEW YORK", "XPDUSDT"),
]


def col(df: pd.DataFrame, pattern: str) -> str:
    hits = [c for c in df.columns if re.fullmatch(pattern, c.strip(), flags=re.I)]
    if not hits:
        raise KeyError(f"no column matching {pattern!r}; have {list(df.columns)[:8]}...")
    return hits[0]


def load_zip(client: httpx.Client, name: str) -> pd.DataFrame | None:
    r = client.get(BASE + name, timeout=120, follow_redirects=True)
    if r.status_code != 200:
        print(f"  {name}: HTTP {r.status_code} (skipped)", flush=True)
        return None
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    inner = zf.namelist()[0]
    df = pd.read_csv(zf.open(inner), low_memory=False)
    mkt = col(df, r"Market_and_Exchange_Names?")
    date_c = col(df, r"Report_Date_as_YYYY-MM-DD|Report_Date_as_MM_DD_YYYY")
    keep = {
        "market_raw": df[mkt].astype(str).str.strip(),
        "report_date": pd.to_datetime(df[date_c]),
        "oi": pd.to_numeric(df[col(df, r"Open_Interest_All")], errors="coerce"),
        "mm_long": pd.to_numeric(df[col(df, r"M_Money_Positions_Long_All")], errors="coerce"),
        "mm_short": pd.to_numeric(df[col(df, r"M_Money_Positions_Short_All")], errors="coerce"),
        "prod_long": pd.to_numeric(df[col(df, r"Prod_Merc_Positions_Long_All")], errors="coerce"),
        "prod_short": pd.to_numeric(df[col(df, r"Prod_Merc_Positions_Short_All")], errors="coerce"),
        "swap_long": pd.to_numeric(df[col(df, r"Swap_Positions_Long_All")], errors="coerce"),
        "swap_short": pd.to_numeric(df[col(df, r"Swap__?Positions_Short_All")], errors="coerce"),
    }
    out = pd.DataFrame(keep)
    sym = pd.Series(pd.NA, index=out.index, dtype="object")
    for pat, s in MARKET_MAP:
        sym = sym.mask(out["market_raw"].str.match(pat, case=False), s)
    out["symbol"] = sym
    out = out.dropna(subset=["symbol"])
    print(f"  {name}: {len(out)} mapped rows", flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    with httpx.Client(headers={"User-Agent": "convexity-research/1.0"}) as client:
        for name in SOURCES:
            f = load_zip(client, name)
            if f is not None and len(f):
                frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    # LOUD double-mapping guard: two different market_raw for one (symbol, report_date) means a
    # regex caught a look-alike contract — fail rather than silently dedup to an arbitrary winner.
    dupes = df[df.duplicated(["symbol", "report_date"], keep=False)]
    conflict = dupes.groupby(["symbol", "report_date"])["market_raw"].nunique()
    if (conflict > 1).any():
        bad = dupes[dupes.set_index(["symbol", "report_date"]).index.isin(
            conflict[conflict > 1].index)]["market_raw"].unique()
        raise SystemExit(f"DOUBLE-MAPPING: {sorted(bad)}")
    df = df.drop_duplicates(["symbol", "report_date"])  # identical rows from overlapping archives
    df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)
    # release lag: report Tuesday, published Friday ~15:30 ET -> usable strictly AFTER Friday UTC
    df["usable_from"] = df["report_date"] + pd.Timedelta(days=3)
    df.to_parquet(OUT / "cot_disagg.parquet", index=False)
    per = df.groupby("symbol")["report_date"].agg(["min", "max", "count"])
    print(per.to_string(), flush=True)
    print(f"COT DONE: {len(df)} rows -> {OUT / 'cot_disagg.parquet'}", flush=True)


if __name__ == "__main__":
    main()
