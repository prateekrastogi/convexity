"""1-min multi-sleeve data loader for the research harness (pure pandas/numpy, NautilusTrader-free).

Reads ``data/stitched/{SYMBOL}.parquet`` (cols ``open_time|open|high|low|close|volume|source|
gap_before``; ``open_time`` is epoch-ms, the bar's OPEN) into the shared ``convexity.core.types.Bars``
the breakout/trend pillars consume, and a wall-clock funding schedule.

The lake spans three sleeves (crypto perps, Dukascopy equity underlyings, Dukascopy commodity
underlyings) under one ``{SYMBOL}USDT.parquet`` naming; the sleeve is resolved per symbol in
``convexity.panel.assetclass``. FUNDING is sleeve-specific (``load_funding``): crypto reads its native
``data/funding/{SYMBOL}.parquet``; equities/commodities have NO native funding, so a constant synthetic
schedule at the ASSUMED perp rate is built over the bar span (the user trades them as perps/CFDs).

NO-LOOKAHEAD: ``Bars.ts`` is set to the bar CLOSE (open_time + 60_000 ms) so a signal known at the
close of bar t is timestamped at t's close, and the funding accrual that compares trade timestamps to
funding stamps is comparing close-to-close (the live convention). ``gap_before`` flags a discontinuity
in the 1-min grid (the RTH/overnight/weekend boundary for the gappy sleeves) and is surfaced as
``is_session_open`` so the pillars' gap handling fires identically to the live path; it is ALSO read
by the gap-through fill model in ``driver.cell``. The InstrumentSpec is built per asset class (AMBER).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexity.core.types import Bars, InstrumentSpec

BAR_MS: int = 60_000


@dataclass(frozen=True)
class FundingSchedule:
    """Wall-clock funding stamps for one instrument: epoch-ms ``ts`` (ascending) + signed ``rate``.

    A perp LONG paying positive funding accrues a COST; a SHORT a credit (COST §5.5 sign convention).
    ``interval_ms`` is the nominal stamp interval (8h for Binance) used only for diagnostics.
    """

    ts: np.ndarray
    rate: np.ndarray
    interval_ms: int

    def __len__(self) -> int:
        return int(self.ts.shape[0])


# Nominal fallback ONLY for the dead equities/commodities sleeves (no live cross-sectional edge,
# EDGE.md scope note) which candidates.md does not carry real Binance filters for (they are
# Dukascopy-underlying proxies, not genuine perp order books). NEVER used for crypto (below).
_NOMINAL_TICK: float = 0.01
_NOMINAL_LOT: float = 0.001


def crypto_spec(symbol: str, funding_interval_hours: int = 8) -> InstrumentSpec:
    """Build an AMBER-only Binance USDT-M perp ``InstrumentSpec`` (no RED/BLUE modelling knob).

    ``calendar_id=""`` marks 24/7 (no session boundary) so the breakout barrier family resolves to
    rolling Donchian (BREAKOUT §2/§3). ``tick_size``/``lot_size``/``min_notional`` are REAL per-symbol
    Binance USDT-M filters read from ``candidates.md`` (``convexity.panel.candidates.crypto_filters``,
    EXECUTION.md's quantization section) -- e.g. XRPUSDT's real lot_size is 0.1, DOGEUSDT's is 1, NOT
    a uniform placeholder. Raises ``KeyError`` if ``symbol`` is not a crypto candidate (never silently
    falls back to a wrong number for a real crypto perp).
    """
    from convexity.panel.candidates import crypto_filters

    f = crypto_filters(symbol)
    return InstrumentSpec(
        symbol=symbol,
        asset_class="crypto_perp",
        venue="BINANCE",
        timezone="UTC",
        tick_size=f.tick_size,
        lot_size=f.lot_size,
        price_multiplier=1.0,
        calendar_id="",
        min_notional=f.min_notional,
        funding_interval_ms=funding_interval_hours * 3600_000,
    )


def instrument_spec(symbol: str, funding_interval_hours: int = 8) -> InstrumentSpec:
    """Build the AMBER ``InstrumentSpec`` for ``symbol`` with its resolved asset class.

    Crypto keeps the 24/7 (``calendar_id=""``) spec and delegates to :func:`crypto_spec` for REAL
    per-symbol ``tick_size``/``lot_size``/``min_notional`` (never the nominal fallback). Equities/
    commodities are gappy underlyings with no genuine Binance order book (Dukascopy-proxied, dead
    sleeve per EDGE.md scope) -- we keep ``calendar_id=""`` (the lake is a stitched 1-min grid whose
    discontinuities are carried by ``gap_before``/``is_session_open``, which the pillars read
    directly), tag the correct ``asset_class`` so any class-aware pillar behaviour (gap-survival
    sigmas etc.) resolves right, and use the NOMINAL tick/lot (documented, never consumed by the live
    crypto-only cross-sectional book).
    """
    from convexity.panel.assetclass import asset_class_of

    asset_class = asset_class_of(symbol)
    if asset_class == "crypto_perp":
        return crypto_spec(symbol, funding_interval_hours=funding_interval_hours)

    return InstrumentSpec(
        symbol=symbol,
        asset_class=asset_class,
        venue="BINANCE",
        timezone="UTC",
        tick_size=_NOMINAL_TICK,
        lot_size=_NOMINAL_LOT,
        price_multiplier=1.0,
        calendar_id="",
        funding_interval_ms=funding_interval_hours * 3600_000,
    )


def load_bars(
    symbol: str,
    *,
    lake_dir: str = "data/stitched",
    days: int | None = None,
) -> Bars:
    """Load 1-min OHLCV for ``symbol`` into ``Bars`` (ts = bar CLOSE epoch-ms; no lookahead).

    ``days`` (when set) keeps only the most-recent ``days`` of bars (a fast slice for quick runs).
    The ``gap_before`` lake column becomes ``is_session_open`` (a 1-min-grid discontinuity marker);
    ``is_valid`` is all-True (the lake is already validated/stitched).
    """
    path = pathlib.Path(lake_dir) / f"{symbol}.parquet"
    df = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "volume", "gap_before"])
    if days is not None and len(df):
        cutoff = int(df["open_time"].iloc[-1]) - days * 86_400_000
        df = df[df["open_time"] >= cutoff]
    df = df.reset_index(drop=True)
    n = len(df)
    ts_close = df["open_time"].to_numpy(np.int64) + BAR_MS  # bar CLOSE (no lookahead)
    return Bars(
        ts=ts_close,
        o=df["open"].to_numpy(float),
        h=df["high"].to_numpy(float),
        l=df["low"].to_numpy(float),
        c=df["close"].to_numpy(float),
        v=df["volume"].to_numpy(float),
        is_session_open=df["gap_before"].to_numpy(bool),
        is_valid=np.ones(n, dtype=bool),
    )


def load_funding(
    symbol: str,
    *,
    funding_dir: str = "data/funding",
    bars: Bars | None = None,
    assumed_perp_rate_8h: float | None = None,
) -> FundingSchedule:
    """Load the wall-clock funding schedule for ``symbol`` (sleeve-specific).

    * CRYPTO — read the native ``data/funding/{SYMBOL}.parquet`` (cols ``calc_time(ms)|
      funding_interval_hours|last_funding_rate``); missing file ⇒ empty schedule (rate 0).
    * EQUITIES / COMMODITIES — there is NO native funding. The user trades these underlyings as
      perps/CFDs, so a CONSTANT synthetic schedule at the ASSUMED perp rate (``assumed_perp_rate_8h``)
      is built every 8h across the bar span (``bars`` required for the span). Same accrual mechanism
      as crypto, just a flat rate — this is the funding HEADWIND the non-crypto sleeves must clear.
    """
    from convexity.panel.assetclass import uses_native_funding

    if uses_native_funding(symbol):
        path = pathlib.Path(funding_dir) / f"{symbol}.parquet"
        if not path.exists():
            return FundingSchedule(ts=np.zeros(0, np.int64), rate=np.zeros(0), interval_ms=8 * 3600_000)
        df = pd.read_parquet(path)
        df = df.sort_values("calc_time").reset_index(drop=True)
        ts = df["calc_time"].to_numpy(np.int64)
        rate = df["last_funding_rate"].to_numpy(float)
        interval_h = int(df["funding_interval_hours"].iloc[0]) if len(df) else 8
        return FundingSchedule(ts=ts, rate=rate, interval_ms=interval_h * 3600_000)

    # non-crypto: synthetic constant-rate perp schedule over the bar span.
    if bars is None or len(bars) == 0 or assumed_perp_rate_8h is None:
        return FundingSchedule(ts=np.zeros(0, np.int64), rate=np.zeros(0), interval_ms=8 * 3600_000)
    interval_ms = 8 * 3600_000
    t0 = int(bars.ts[0])
    t1 = int(bars.ts[-1])
    stamps = np.arange(t0, t1 + 1, interval_ms, dtype=np.int64)
    rate = np.full(stamps.shape[0], float(assumed_perp_rate_8h))
    return FundingSchedule(ts=stamps, rate=rate, interval_ms=interval_ms)
