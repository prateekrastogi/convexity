"""Load the trading universe from candidates.md (the machine-readable ```json block)."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

_UNIVERSE_MD = pathlib.Path(__file__).with_name("candidates.md")


def load_universe(path: str | pathlib.Path | None = None) -> dict:
    """Return the parsed universe dict: keys `meta`, `crypto`, `equities`, `commodities`."""
    text = pathlib.Path(path or _UNIVERSE_MD).read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if not m:
        raise ValueError("no ```json block found in candidates.md")
    return json.loads(m.group(1))


def all_symbols(sleeves: tuple[str, ...] = ("crypto", "equities", "commodities")) -> list[str]:
    """Flat list of Binance perp symbols across the requested sleeves."""
    u = load_universe()
    return [d["symbol"] for s in sleeves for d in u[s]]


@dataclass(frozen=True)
class ExchangeFilters:
    """One crypto symbol's real Binance USDT-M futures order-sizing filters (candidates.md meta's
    ``crypto_filters_source``/``crypto_filters_note``): ``LOT_SIZE``/``PRICE_FILTER``/``MIN_NOTIONAL``.

    A CURRENT exchange snapshot, not proven point-in-time-accurate across the full backtest window
    (Binance does occasionally revise these) -- see EXECUTION.md's quantization section for the
    documented limitation and how ``execution.quantize_qty``/``quantize_price`` consume this.
    """

    tick_size: float
    lot_size: float
    min_qty: float
    min_notional: float


def crypto_filters(symbol: str) -> ExchangeFilters:
    """Real per-symbol Binance filters for one crypto candidate (KeyError if not a crypto entry).

    Equities/commodities carry no filters (dead sleeve, no live cross-sectional edge, EDGE.md scope
    note) -- this function is crypto-only by construction, matching candidates.md's own note.
    """
    u = load_universe()
    for d in u["crypto"]:
        if d["symbol"] == symbol:
            return ExchangeFilters(
                tick_size=d["tick_size"],
                lot_size=d["lot_size"],
                min_qty=d["min_qty"],
                min_notional=d["min_notional"],
            )
    raise KeyError(f"{symbol!r} is not a crypto candidate in candidates.md (no exchange filters)")
