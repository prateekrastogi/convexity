"""Per-symbol asset-class detection + the class-level policy the scalp harness branches on.

The harness now spans THREE sleeves (``selection/UNIVERSE.md``): crypto (50 USDT-M perps, native
funding), equities (30 Dukascopy underlyings, RTH-only + overnight gaps), commodities (7 Dukascopy
underlyings, near-continuous). Every sleeve-specific difference is funnelled through here so the
driver/cell/netcost code stays one path:

  * ``asset_class_of(symbol)`` — map a lake symbol to a ``core.types.AssetClass`` via the UNIVERSE
    sleeve it belongs to (crypto -> crypto_perp; equities -> us_equity; commodities -> the precious /
    industrial / commodity_future sub-class from the UNIVERSE sector). REUSES the universe loader.
  * ``Sleeve`` — the coarse {crypto, equity, commodity} bucket used for pooled-per-class reporting.
  * ``models_gap_through`` — only the gappy sleeves (equities, and commodities to a lesser extent)
    can GAP THROUGH a stop/TP overnight; crypto is 24/7 and never gaps.
  * ``uses_native_funding`` — only crypto has native funding data. Equities/commodities have NONE,
    but the user trades them as PERPS/CFDs, so the harness ASSUMES a representative perp funding rate
    (the median 8h funding across the crypto lake; see ``config.assumed_perp_funding_rate``) applied
    by wall-clock — same mechanism as crypto, just a constant synthetic schedule.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from convexity.core.types import AssetClass
from convexity.selection.universe import load_universe

# Coarse sleeve bucket used for pooled-per-class report rows.
Sleeve = Literal["crypto", "equity", "commodity"]

# UNIVERSE commodity sectors -> the core AssetClass sub-class (drives the Kalman preset + gap sigmas).
_COMMODITY_SECTOR_CLASS: dict[str, AssetClass] = {
    "precious-metal": "precious_metal",
    "industrial-metal": "industrial_metal",
    "energy": "commodity_future",
}


@lru_cache(maxsize=1)
def _symbol_index() -> dict[str, tuple[Sleeve, AssetClass]]:
    """Build ``{symbol: (sleeve, asset_class)}`` once from UNIVERSE.md (cached)."""
    u = load_universe()
    idx: dict[str, tuple[Sleeve, AssetClass]] = {}
    for d in u["crypto"]:
        idx[d["symbol"]] = ("crypto", "crypto_perp")
    for d in u["equities"]:
        idx[d["symbol"]] = ("equity", "us_equity")
    for d in u["commodities"]:
        ac = _COMMODITY_SECTOR_CLASS.get(d.get("sector", ""), "commodity_future")
        idx[d["symbol"]] = ("commodity", ac)
    return idx


def sleeve_of(symbol: str) -> Sleeve:
    """Coarse {crypto, equity, commodity} sleeve for ``symbol`` (defaults to crypto if unknown)."""
    return _symbol_index().get(symbol, ("crypto", "crypto_perp"))[0]


def asset_class_of(symbol: str) -> AssetClass:
    """The fine ``core.types.AssetClass`` for ``symbol`` (defaults to crypto_perp if unknown)."""
    return _symbol_index().get(symbol, ("crypto", "crypto_perp"))[1]


def models_gap_through(symbol: str) -> bool:
    """True when a hold can GAP THROUGH the stop/TP across a session break.

    Crypto perps are 24/7 (no gaps). Equities are RTH-only with real overnight/weekend gaps, and
    commodities have session breaks too — both are modelled with gap-through (fill at the gapped open
    when the next session opens beyond the barrier). Driven by ``gap_before`` marks in the lake.
    """
    return sleeve_of(symbol) in ("equity", "commodity")


def uses_native_funding(symbol: str) -> bool:
    """True only for crypto (native funding data). Equities/commodities use the ASSUMED perp rate."""
    return sleeve_of(symbol) == "crypto"
