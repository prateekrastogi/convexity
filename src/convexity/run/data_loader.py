"""Real-data loaders for the parity-backtest harness — bar lake, funding lake, instrument builder.

NautilusTrader-importing, so VM-only (Invariant N). Three offline loaders that turn the convexity
data lake into the inputs a ``BacktestEngine`` consumes, with NO live provider/network:

  * :func:`load_bars` — ``data/stitched/{SYMBOL}.parquet`` (cols open_time|open|high|low|close|
    volume|source|gap_before; open_time is epoch-ms) → a list of Nautilus ``Bar`` for a ``BarType``.
    The ``gap_before`` column is CONSUMED (FRICTION #2): with ``return_gaps=True`` the loader also
    returns the bar-close timestamps the lake flagged as gaps, which the backtest threads into the
    strategy so its per-bar ``Bars.is_session_open`` flag matches the live ts-discontinuity path.
  * :func:`load_funding` — ``data/funding/{SYMBOL}.parquet`` (cols calc_time(ms)|
    funding_interval_hours|last_funding_rate) → a list of ``FundingRateUpdate`` (COST §5.5: the
    engine charges funding only if these are fed).
  * :func:`build_instrument` — a Binance USDT-M ``CryptoPerpetual`` constructed programmatically from
    the universe :class:`~convexity.core.types.InstrumentSpec` (tick_size/lot_size/price_multiplier),
    mirroring ``TestInstrumentProvider.btcusdt_perp_binance`` but for any universe symbol.
"""

from __future__ import annotations

import math
import pathlib
from decimal import Decimal
from typing import Any

from convexity.core.types import InstrumentSpec

__all__ = [
    "DEFAULT_LAKE",
    "DEFAULT_FUNDING_LAKE",
    "load_bars",
    "load_funding",
    "build_instrument",
    "bar_type_for",
]

DEFAULT_LAKE = pathlib.Path("data/stitched")
DEFAULT_FUNDING_LAKE = pathlib.Path("data/funding")


def _precision_from_increment(increment: float) -> int:
    """Decimal places implied by a tick/size increment (e.g. 0.1 → 1, 0.001 → 3, 1.0 → 0)."""
    if increment >= 1.0:
        return 0
    # round to avoid 0.1 → 0.09999999 style float noise.
    s = f"{increment:.12f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def instrument_id_str(symbol: str, venue: str = "BINANCE") -> str:
    """The Nautilus instrument-id string for a universe perp symbol (``BTCUSDT`` → ``BTCUSDT-PERP.BINANCE``)."""  # noqa: E501
    return f"{symbol}-PERP.{venue}"


def bar_type_for(symbol: str, venue: str = "BINANCE") -> str:
    """The 1-min LAST EXTERNAL bar-type descriptor for a universe perp symbol."""
    return f"{instrument_id_str(symbol, venue)}-1-MINUTE-LAST-EXTERNAL"


def build_instrument(spec: InstrumentSpec, venue: str = "BINANCE") -> Any:
    """Build a Binance USDT-M ``CryptoPerpetual`` from the AMBER :class:`InstrumentSpec`.

    Programmatic/offline (no live provider): the price/size increments come from
    ``spec.tick_size``/``spec.lot_size``, fees from the bps spec, and the base currency is derived
    from the symbol (``ETHUSDT`` → ``ETH`` quoted/settled in USDT). The quantity bounds are wide so
    the LEVERAGE-sized lots are never clamped by the instrument metadata in the slice.
    """
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import CurrencyType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Currency, Money, Price, Quantity

    symbol = spec.symbol
    base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol
    base_currency = Currency(
        base_asset, precision=8, iso4217=0, name=base_asset, currency_type=CurrencyType.CRYPTO
    )
    price_prec = _precision_from_increment(spec.tick_size)
    size_prec = _precision_from_increment(spec.lot_size)
    min_notional = max(float(spec.min_notional), 0.0)
    return CryptoPerpetual(
        instrument_id=InstrumentId(
            symbol=Symbol(f"{symbol}-PERP"), venue=Venue(venue)
        ),
        raw_symbol=Symbol(symbol),
        base_currency=base_currency,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=price_prec,
        price_increment=Price(spec.tick_size, price_prec),
        size_precision=size_prec,
        size_increment=Quantity(spec.lot_size, size_prec),
        max_quantity=Quantity(1_000_000.0, size_prec),
        min_quantity=Quantity(spec.lot_size, size_prec),
        max_notional=None,
        min_notional=Money(min_notional, USDT),
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=Decimal(str(spec.maker_fee_bps / 10_000.0)),
        taker_fee=Decimal(str(spec.taker_fee_bps / 10_000.0)),
        ts_event=0,
        ts_init=0,
    )


def load_bars(
    symbol: str,
    instrument: Any,
    *,
    lake: pathlib.Path | str = DEFAULT_LAKE,
    limit: int | None = None,
    offset: int = 0,
    venue: str = "BINANCE",
    return_gaps: bool = False,
) -> list[Any] | tuple[list[Any], frozenset[int]]:
    """Load ``data/stitched/{SYMBOL}.parquet`` → a list of Nautilus ``Bar`` for this instrument.

    ``open_time`` is epoch-ms (the bar's open); we stamp ``ts_event``/``ts_init`` at the bar CLOSE
    (open_time + 1 min) in ns so the engine processes a closed bar (the LAST kline convention). The
    high/low already bracket open/close in the stitched lake so Nautilus's bar validity holds.
    ``offset``/``limit`` take a contiguous slice (for a small deterministic test window).

    CROSS-MODE GAP MARKER (FRICTION #2): the lake's authoritative ``gap_before`` column — True on the
    first bar after a stitch boundary — is now CONSUMED, not just documented. A Nautilus ``Bar`` cannot
    carry a custom flag, so the gap is threaded out-of-band: with ``return_gaps=True`` this returns
    ``(bars, gap_close_ms)`` where ``gap_close_ms`` is the frozenset of bar-CLOSE timestamps (ms, the
    same key ``Bar.ts_event`` carries) that the lake flagged as a gap. ``run/backtest.run_session``
    wires that set into ``StrategyDeps.gap_close_ms_by_instrument`` so the strategy's ``on_bar`` raises
    the SAME ``Bars.is_session_open`` flag the live ts-discontinuity path raises — true cross-mode
    parity for BREAKOUT's session-gap masking + TREND's once-per-boundary gap_var. The set covers the
    rare stitch boundary a 24/7 feed marks WITHOUT a timestamp hole (which ts-discontinuity alone would
    miss); where a hole IS present, ``RollingBars`` OR-detects it anyway, so the two agree.
    """
    import pandas as pd
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.objects import Price, Quantity

    path = pathlib.Path(lake) / f"{symbol}.parquet"
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    # Read gap_before when present (older lakes may predate the column); fall back gracefully.
    try:
        df = pd.read_parquet(path, columns=[*cols, "gap_before"])
        have_gap = True
    except Exception:  # noqa: BLE001 — column absent on an older lake snapshot
        df = pd.read_parquet(path, columns=cols)
        have_gap = False
    df = df.sort_values("open_time").reset_index(drop=True)
    if offset:
        df = df.iloc[offset:].reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    bar_type = BarType.from_str(bar_type_for(symbol, venue))
    price_prec = instrument.price_precision
    size_prec = instrument.size_precision
    interval_ns = 60_000_000_000  # 1-min bar; close = open + 1 min
    bars: list[Any] = []
    ot = df["open_time"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    gap = df["gap_before"].to_numpy() if have_gap else None
    gap_close_ms: set[int] = set()
    for j in range(len(df)):
        close_ms = int(ot[j]) + 60_000  # bar close = open_time(ms) + 1 min, in ms
        close_ns = int(ot[j]) * 1_000_000 + interval_ns
        if gap is not None and bool(gap[j]):
            gap_close_ms.add(close_ms)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(float(o[j]), price_prec),
                high=Price(float(h[j]), price_prec),
                low=Price(float(low[j]), price_prec),
                close=Price(float(c[j]), price_prec),
                volume=Quantity(max(float(v[j]), 0.0), size_prec),
                ts_event=close_ns,
                ts_init=close_ns,
            )
        )
    if return_gaps:
        return bars, frozenset(gap_close_ms)
    return bars


def load_funding(
    symbol: str,
    instrument: Any,
    *,
    lake: pathlib.Path | str = DEFAULT_FUNDING_LAKE,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[Any]:
    """Load ``data/funding/{SYMBOL}.parquet`` → a list of Nautilus ``FundingRateUpdate``.

    Columns ``calc_time(ms) | funding_interval_hours | last_funding_rate``. ``calc_time`` is the
    funding-stamp epoch-ms; we stamp ``ts_event``/``ts_init`` at that time in ns. The COST pillar
    sums these realized funding events over the hold (COST §5.5) — the engine only charges funding
    when these are added to the data stream. ``start_ms``/``end_ms`` clip to the bar window so the
    feed lines up with the loaded slice. Returns ``[]`` if no funding file exists (non-perp / not yet
    backfilled), in which case the run is directional-only (no funding charged).
    """
    import pandas as pd
    from nautilus_trader.model.data import FundingRateUpdate

    path = pathlib.Path(lake) / f"{symbol}.parquet"
    if not pathlib.Path(path).exists():
        return []
    df = pd.read_parquet(path)
    df = df.sort_values("calc_time").reset_index(drop=True)
    if start_ms is not None:
        df = df[df["calc_time"] >= start_ms]
    if end_ms is not None:
        df = df[df["calc_time"] <= end_ms]
    out: list[Any] = []
    for _, row in df.iterrows():
        rate = float(row["last_funding_rate"])
        if math.isnan(rate):
            continue
        ts_ns = int(row["calc_time"]) * 1_000_000
        out.append(
            FundingRateUpdate(
                instrument_id=instrument.id,
                rate=Decimal(str(rate)),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
        )
    return out
