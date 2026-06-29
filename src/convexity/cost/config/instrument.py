"""InstrumentCostSpec — AMBER per-instrument metadata ONLY (COST §4, Invariant C-cfg).

This is the ONLY config object that travels per-symbol. It holds AMBER market
metadata: the venue fee schedule key, funding interval/cap, carry-class, and the
roll/financing/storage descriptors. A RED constant (half-life, shrinkage, band
quantile, residual floor, tier proximity, max-hold quantile, edge buffer) appearing
here is THE bug Invariant C-cfg forbids — those reach the model only via
``resolve_config(defaults=...)``. The Inv-C test runs ``assert_no_red_in_amber`` over
this dataclass's field names against :data:`COST_PARAM_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.cost.types import CarryClass, CommodityFinancingMode, InterestRateUnit


@dataclass(frozen=True)
class InstrumentCostSpec:
    """AMBER metadata for one instrument's cost/carry (§4)."""

    symbol: str
    carry_class: CarryClass
    fee_schedule_venue: str  # key into the AMBER FeeSchedule table
    funding_interval_ms: int  # AMBER (Binance USDT-M = 8h = 28_800_000)
    funding_cap: float  # AMBER venue per-interval funding cap
    interest_rate_unit: InterestRateUnit  # AMBER; gates the (8/24) scaling (§5.5b); no default
    pre_publishes_funding: bool = True  # AMBER; venue pre-publishes next funding (§5.5)
    # carry descriptors (AMBER) — None when the class does not use them (§5.6).
    tau_spread_days: float | None = None  # far-near expiry gap (dated/commodity basis, §5.6-time)
    mtf_compounding: bool = False  # §5.6-time; default SIMPLE Actual/365
    commodity_financing_mode: CommodityFinancingMode | None = None  # §5.6-commodity
    commodity_basis_observable: bool = False  # §5.6-commodity; else storage-only, convenience=0


# The AMBER field names — the Inv-C test asserts none is a RED/BLUE param name.
INSTRUMENT_COST_SPEC_FIELDS = frozenset(
    {
        "symbol",
        "carry_class",
        "fee_schedule_venue",
        "funding_interval_ms",
        "funding_cap",
        "interest_rate_unit",
        "pre_publishes_funding",
        "tau_spread_days",
        "mtf_compounding",
        "commodity_financing_mode",
        "commodity_basis_observable",
    }
)
