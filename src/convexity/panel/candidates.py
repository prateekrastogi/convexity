"""Load the trading universe from candidates.md (the machine-readable ```json block).

The committed block is the source of truth for the RESEARCHED universe. A gitignored overlay
(``data/universe/discovered.json``, written by ``scripts/discover_tradfi_perps.py``) carries venue
commodity perps discovered AFTER the last commit — merged here so panels, backfills and the paper
books' symbol domains all see them without a repo edit. Discovered symbols are data/universe-only:
nothing trades them until research promotes them into a frozen strategy config.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

_UNIVERSE_MD = pathlib.Path(__file__).with_name("candidates.md")
_DISCOVERED_JSON = pathlib.Path("data/universe/discovered.json")  # cwd-relative like the lake dirs


def load_universe(
    path: str | pathlib.Path | None = None,
    *,
    include_discovered: bool = True,
    discovered_path: str | pathlib.Path | None = None,
) -> dict:
    """Return the parsed universe dict: keys `meta`, `crypto`, `equities`, `commodities`.

    ``include_discovered`` merges the gitignored discovery overlay (deduped by symbol; the
    committed entry wins so a hand-tuned entry is never shadowed by the auto-discovered one).
    """
    text = pathlib.Path(path or _UNIVERSE_MD).read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if not m:
        raise ValueError("no ```json block found in candidates.md")
    u = json.loads(m.group(1))
    if include_discovered:
        dp = pathlib.Path(discovered_path or _DISCOVERED_JSON)
        if dp.exists():
            overlay = json.loads(dp.read_text())
            for sleeve, entries in overlay.items():
                if sleeve == "meta" or not isinstance(entries, list):
                    continue
                have = {d["symbol"] for d in u.get(sleeve, [])}
                u.setdefault(sleeve, []).extend(
                    d for d in entries if d.get("symbol") and d["symbol"] not in have)
    return u


def all_symbols(sleeves: tuple[str, ...] = ("crypto", "equities", "commodities")) -> list[str]:
    """Flat list of Binance perp symbols across the requested sleeves (incl. discovered overlay)."""
    u = load_universe()
    return [d["symbol"] for s in sleeves for d in u[s]]


@dataclass(frozen=True)
class ExchangeFilters:
    """One symbol's real Binance USDT-M futures order-sizing filters (candidates.md meta's
    ``crypto_filters_source``/``crypto_filters_note``): ``LOT_SIZE``/``PRICE_FILTER``/``MIN_NOTIONAL``.

    A CURRENT exchange snapshot, not proven point-in-time-accurate across the full backtest window
    (Binance does occasionally revise these) -- see EXECUTION.md's quantization section for the
    documented limitation and how ``execution.quantize_qty``/``quantize_price`` consume this.
    Live sessions re-quantize against the venue's current exchangeInfo (``fetch_live_filters``).
    """

    tick_size: float
    lot_size: float
    min_qty: float
    min_notional: float


_FILTER_KEYS = ("tick_size", "lot_size", "min_qty", "min_notional")


def _entry_filters(d: dict) -> ExchangeFilters | None:
    if not all(k in d for k in _FILTER_KEYS):
        return None
    return ExchangeFilters(tick_size=d["tick_size"], lot_size=d["lot_size"],
                           min_qty=d["min_qty"], min_notional=d["min_notional"])


def sleeve_filters(sleeves: tuple[str, ...]) -> dict[str, ExchangeFilters]:
    """symbol -> filters for every entry in ``sleeves`` that carries the real venue filter fields.

    The paper books consume this as their DRY-RUN quantization surface (one map per book domain);
    live sessions override with the venue's current exchangeInfo. An entry without filter fields
    (equities, the dead sleeve) is simply absent — ``order_intents`` skips what it cannot quantize.
    """
    u = load_universe()
    out: dict[str, ExchangeFilters] = {}
    for s in sleeves:
        for d in u.get(s, []):
            f = _entry_filters(d)
            if f is not None:
                out[d["symbol"]] = f
    return out


def crypto_filters(symbol: str) -> ExchangeFilters:
    """Real per-symbol Binance filters for one crypto candidate (KeyError if not a crypto entry).

    Kept for back-compat with the original crypto-only paper book; new callers use
    ``sleeve_filters`` for a whole book domain.
    """
    u = load_universe()
    for d in u["crypto"]:
        if d["symbol"] == symbol:
            f = _entry_filters(d)
            if f is not None:
                return f
    raise KeyError(f"{symbol!r} is not a crypto candidate in candidates.md (no exchange filters)")
