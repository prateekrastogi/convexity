"""InstrumentLeverageSpec — AMBER per-instrument metadata ONLY (LEVERAGE §4, Invariant C).

This is the ONLY config object that travels per-symbol. It holds AMBER market
metadata: the lot/contract step, contract multiplier / point value, tick size, the
exchange/broker MAXIMUM position (the hard §8.6.4 cap), and the financing-rate
descriptor. A RED constant (c, Kelly fraction, governor breakpoint, ES level, H_book,
σ_target, vol multiple, …) appearing here is THE bug Invariant C forbids — those
reach the model only via ``resolve_config(defaults=...)``. The Inv-C test runs
``assert_no_red_in_amber`` over this dataclass's field names against
:data:`LEVERAGE_PARAM_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import AssetClass


@dataclass(frozen=True)
class InstrumentLeverageSpec:
    """AMBER metadata for one instrument's sizing/quantization (§4, §8.6)."""

    symbol: str
    asset_class: AssetClass
    # §8.6.2 the tradable lot step (futures: lot size, e.g. NIFTY 65, BANKNIFTY 30;
    # equities: 1 share; crypto: the venue minimum order increment).
    lot_step: float
    # §8.6.2 notional per one tradable unit = contract_multiplier * price (point value).
    contract_multiplier: float
    tick_size: float
    # §8.6.2 minimum tradable notional / one lot's minimum (venue min_notional).
    min_notional: float = 0.0
    # §8.6.2 exchange/broker MAXIMUM lots (hard, ceiling-independent §8.6.4 cap). None -> unbounded.
    exchange_max_lots: int | None = None
    # AMBER financing/MTF interest annualized rate descriptor (the carry feed; the SIGNED
    # carry itself comes from COST's CarryTerm — this is only the metadata flag).
    financing_rate_annual: float = 0.0


# The AMBER field names — the Inv-C test asserts none is a RED/BLUE param name.
INSTRUMENT_LEVERAGE_SPEC_FIELDS = frozenset(
    {
        "symbol",
        "asset_class",
        "lot_step",
        "contract_multiplier",
        "tick_size",
        "min_notional",
        "exchange_max_lots",
        "financing_rate_annual",
    }
)
