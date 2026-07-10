"""In-memory panel top-up from Binance REST (PAPER.md §2) — the lake stays immutable.

The stitched lake refreshes on its own weekly timer (backfill dumps + stitch); a rebalance session
must not wait for T+1 dump publication, so the tail between the lake's last day and the last
COMPLETED UTC day is fetched live: one ``/fapi/v1/klines?interval=1d`` and one
``/fapi/v1/fundingRate`` request per symbol (~200 unauthenticated requests weekly — far inside the
public limits), then merged into the panel by ``extend_panel`` using build_panel's OWN conventions
(daily close = the day's last close; funding = the day's stamp sum with midnight belonging to the
STARTING day; ``listed`` = finite close; funding 0-filled on live days, NaN off-span).
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import pandas as pd

from convexity.panel.panel import MS_PER_DAY, Panel

# DATA venue — ALWAYS prod (PAPER.md §12.6): daily klines/funding tails feed the research
# panel and demo market history is thin/laggy. EXECUTION surfaces below resolve through
# convexity.paper.venue instead.
FAPI = "https://fapi.binance.com"


def fetch_ticker_prices() -> dict[str, float]:
    """Current last price for ALL symbols in one public call — EXECUTION venue (these prices
    mark the book we actually trade: pegs, drift metric; PAPER.md §12.6)."""
    from convexity.paper.venue import execution_base

    with httpx.Client(timeout=15.0) as client:
        r = client.get(f"{execution_base()}/fapi/v1/ticker/price")
        r.raise_for_status()
        return {row["symbol"]: float(row["price"]) for row in r.json()}


def fetch_book_tickers() -> dict[str, tuple[float, float]]:
    """(best_bid, best_ask) for ALL symbols in one public call.

    The maker peg (PAPER.md §4): a BUY joins the BEST BID and a SELL the BEST ASK — pegging to the
    last trade can sit on the crossing side and draws Binance -5022 post-only rejections (learned
    in the first supervised session, 2026-07-03). EXECUTION venue: the peg must come from the
    order book the order actually rests in (audit P0-1/P1-7)."""
    from convexity.paper.venue import execution_base

    with httpx.Client(timeout=15.0) as client:
        r = client.get(f"{execution_base()}/fapi/v1/ticker/bookTicker")
        r.raise_for_status()
        return {row["symbol"]: (float(row["bidPrice"]), float(row["askPrice"]))
                for row in r.json()}


def fetch_execution_venue_tradeable() -> frozenset[str]:
    """Symbols with status TRADING on the EXECUTION venue (the Binance DEMO the gateway trades).

    The research surfaces (filters, pegs, klines) read PROD fapi — near-identical for crypto,
    but the demo's TradFi listings LAG prod (2026-07-06: metals live, CL/NATGAS absent). Live
    sessions gate strategy universes on this set so the journaled book only holds what the
    venue can hold; one public unauthenticated call. Venue via the ONE switch (PAPER.md §12.6).
    """
    from convexity.paper.venue import execution_base

    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{execution_base()}/fapi/v1/exchangeInfo")
        r.raise_for_status()
        return frozenset(s["symbol"] for s in r.json()["symbols"]
                         if s.get("status") == "TRADING")


def effective_filters(tick: float, lot: float, price_prec: int | None,
                      qty_prec: int | None) -> tuple[float, float]:
    """Clamp tick/step by the venue's decimal-precision caps (pure helper, test-pinned).

    Binance enforces BOTH the PRICE_FILTER/LOT_SIZE grids AND per-symbol pricePrecision/
    quantityPrecision (max decimals). A step finer than the precision draws -1111 "Precision is
    over the maximum defined for this asset" — live-caught in the equity book's first supervised
    session (2026-07-10: SPCXUSDT demo has quantityPrecision=0, whole shares only, while the
    stale filter source implied a fractional lot). The effective grid is the COARSER of the two
    constraints."""
    if price_prec is not None:
        tick = max(tick, 10.0 ** -int(price_prec))
    if qty_prec is not None:
        lot = max(lot, 10.0 ** -int(qty_prec))
    return tick, lot


def fetch_live_filters() -> dict[str, tuple[float, float, float, float]]:
    """(tick, lot, min_qty, min_notional) from the EXECUTION venue's CURRENT exchangeInfo.

    The candidates.md filters are a committed SNAPSHOT (EXECUTION.md §16's documented limitation);
    Binance revises tick/lot occasionally, and a stale tick draws -4014 at submission (learned in
    the first supervised session). Live sessions quantize against THESE; research keeps the
    snapshot. Reads the EXECUTION venue (the venue the gateway actually trades — prod filters
    drift from demo on TradFi listings, the SPCX -1111 lesson; venue via the ONE switch,
    PAPER.md §12.6) and clamps tick/step by the venue's own pricePrecision/quantityPrecision
    via ``effective_filters``."""
    from convexity.paper.venue import execution_base

    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{execution_base()}/fapi/v1/exchangeInfo")
        r.raise_for_status()
        out: dict[str, tuple[float, float, float, float]] = {}
        for s in r.json()["symbols"]:
            fs = {f["filterType"]: f for f in s["filters"]}
            try:
                tick, lot = effective_filters(
                    float(fs["PRICE_FILTER"]["tickSize"]),
                    float(fs["LOT_SIZE"]["stepSize"]),
                    s.get("pricePrecision"), s.get("quantityPrecision"))
                out[s["symbol"]] = (
                    tick, lot,
                    float(fs["LOT_SIZE"]["minQty"]),
                    float(fs.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
                )
            except KeyError:
                continue
        return out


def last_completed_day_ms(now_ms: int | None = None) -> int:
    """Epoch-ms of 00:00 UTC of the last COMPLETED UTC day (never the running day)."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return (now // MS_PER_DAY - 1) * MS_PER_DAY


def fetch_daily_tail(
    symbol: str,
    start_day_ms: int,
    *,
    client: httpx.Client,
    end_day_ms: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(day_ms, close, funding_daily) for COMPLETED days in [start_day_ms, end_day_ms].

    Klines: ``interval=1d`` open-time-stamped; a day is kept only if fully closed. Funding: raw
    stamps summed per UTC day with ``fundingTime // MS_PER_DAY`` (the panel's _daily_funding
    convention: the midnight stamp belongs to the STARTING day). Missing funding days are 0 —
    exactly the panel's live-day-no-stamp convention. 404/no-data (pre-listing) -> empty arrays.
    """
    end_ms = last_completed_day_ms() if end_day_ms is None else int(end_day_ms)
    if start_day_ms > end_ms:
        return np.array([], dtype=np.int64), np.array([]), np.array([])

    r = client.get(f"{FAPI}/fapi/v1/klines", params={
        "symbol": symbol, "interval": "1d", "startTime": int(start_day_ms),
        "endTime": int(end_ms + MS_PER_DAY - 1), "limit": 1500})
    r.raise_for_status()
    rows = r.json()
    days, closes = [], []
    for k in rows:
        open_ms = int(k[0])
        if open_ms > end_ms:  # running (incomplete) day — never used
            continue
        days.append(open_ms)
        closes.append(float(k[4]))

    fr = client.get(f"{FAPI}/fapi/v1/fundingRate", params={
        "symbol": symbol, "startTime": int(start_day_ms),
        "endTime": int(end_ms + MS_PER_DAY - 1), "limit": 1000})
    fr.raise_for_status()
    fund_by_day: dict[int, float] = {}
    for f in fr.json():
        d = (int(f["fundingTime"]) // MS_PER_DAY) * MS_PER_DAY
        if start_day_ms <= d <= end_ms:
            fund_by_day[d] = fund_by_day.get(d, 0.0) + float(f["fundingRate"])

    day_arr = np.array(days, dtype=np.int64)
    close_arr = np.array(closes, dtype=float)
    fund_arr = np.array([fund_by_day.get(int(d), 0.0) for d in day_arr], dtype=float)
    return day_arr, close_arr, fund_arr


def fetch_tails(
    symbols: tuple[str, ...],
    start_day_ms: int,
    *,
    pause_s: float = 0.12,
    end_day_ms: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Politeness-paced tail fetch for every symbol (~2 requests/symbol)."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with httpx.Client(timeout=20.0) as client:
        for sym in symbols:
            try:
                out[sym] = fetch_daily_tail(sym, start_day_ms, client=client,
                                            end_day_ms=end_day_ms)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404):  # pre-listing / delisted
                    out[sym] = (np.array([], dtype=np.int64), np.array([]), np.array([]))
                else:
                    raise
            time.sleep(pause_s)
    return out


def extend_panel(
    panel: Panel,
    tails: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Panel:
    """Append the fetched tail days to ``panel`` — PURE, lake untouched (PAPER.md §2).

    Only days strictly AFTER ``panel.dates[-1]`` are appended (the lake row wins on overlap — the
    stitched 1-min resample is the more granular truth). ``listed``/``first_day``/``last_day`` are
    recomputed with build_panel's exact rules (listed = finite close; funding 0 where listed with
    no stamp, NaN off-span).
    """
    last_lake_ms = int(panel.dates[-1].value // 1_000_000)
    new_days: set[int] = set()
    for day_arr, _c, _f in tails.values():
        for d in day_arr.tolist():
            if d > last_lake_ms:
                new_days.add(int(d))
    if not new_days:
        return panel

    days_sorted = sorted(new_days)
    n_new, n_sym = len(days_sorted), panel.n_symbols
    day_pos = {d: i for i, d in enumerate(days_sorted)}
    close_new = np.full((n_new, n_sym), np.nan)
    fund_new = np.full((n_new, n_sym), np.nan)
    for j, sym in enumerate(panel.symbols):
        day_arr, c_arr, f_arr = tails.get(
            sym, (np.array([], dtype=np.int64), np.array([]), np.array([])))
        for d, c, f in zip(day_arr.tolist(), c_arr.tolist(), f_arr.tolist(), strict=True):
            if d > last_lake_ms:
                i = day_pos[int(d)]
                close_new[i, j] = c
                fund_new[i, j] = f

    close = np.vstack([panel.close, close_new])
    listed = ~np.isnan(close)
    # panel.funding is already 0-filled/NaN-masked; apply the same rule to the new rows only.
    fund_new = np.where(listed[-n_new:], np.nan_to_num(fund_new, nan=0.0), np.nan)
    funding = np.vstack([panel.funding, fund_new])
    dates = panel.dates.append(pd.DatetimeIndex(
        pd.to_datetime(np.array(days_sorted, dtype=np.int64), unit="ms")))

    first_day: dict[str, int] = {}
    last_day: dict[str, int] = {}
    for j, sym in enumerate(panel.symbols):
        live = np.flatnonzero(listed[:, j])
        if live.size == 0:
            continue
        first_day[sym] = int(live[0])
        last_day[sym] = int(live[-1])

    return Panel(dates=dates, symbols=panel.symbols, close=close, funding=funding,
                 listed=listed, first_day=first_day, last_day=last_day)
