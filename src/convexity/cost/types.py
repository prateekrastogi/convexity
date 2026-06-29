"""COST-pillar frozen data objects (COST §10).

Pure-Python, NautilusTrader-free. The cross-pillar boundary types
(``Side``, ``CostComponent``, ``RoundTripCost``, ``HoldHorizon``) already live in
``convexity.core.types`` and are RE-EXPORTED here, never redefined — COST is the
producer of ``CostComponent``/``RoundTripCost`` and the consumer of ``HoldHorizon``
(COST P1 "one cost truth"; ARCHITECTURE §3). The objects defined *here* are the
COST-internal market-state view and the three consumer decisions (§8) plus the
post-trade TCA report (§9), none of which cross into a sibling pillar's namespace.

The carry-model class axis (``CarryClass``, §5.6) is a COST-internal modelling
concept selected by AMBER metadata — it is NOT the foundation's instrument
``AssetClass`` (which buckets the *whole* universe). One ``CarryModel`` interface,
per-class body, keyed by this literal (Invariant X).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

# Re-export the shared boundary types so COST code imports them from one place
# without redefining them (ARCHITECTURE §3 / COST P1).
from convexity.core.types import (
    CostComponent,
    HoldHorizon,
    RoundTripCost,
    Side,
)

__all__ = [
    "Side",
    "CostComponent",
    "RoundTripCost",
    "HoldHorizon",
    "LiquidityFlag",
    "CarryClass",
    "InterestRateUnit",
    "SpreadRegime",
    "CommodityFinancingMode",
    "CostMarketState",
    "NetEVDecision",
    "HoldCarryDecision",
    "TCAReport",
]

# §5.2 each leg is maker or taker; the fee rate is selected by this flag.
LiquidityFlag = Literal["maker", "taker"]

# §5.6 the carry-model branch. One interface, per-class body keyed by this AMBER
# literal (Invariant X). This is the COST modelling axis, distinct from the
# foundation universe-level ``core.types.AssetClass``.
CarryClass = Literal["crypto_perp", "dated_future", "equity_mtf", "commodity"]

# §5.5b — gates the (8/24) interest scaling; NO silent default (a missing unit is
# a config error, not an assumed per_day).
InterestRateUnit = Literal["per_day", "per_interval"]

# §5.3 spread is regime-bucketed (TEAL) per instrument; the gate forecast uses the
# current regime's expected half-spread.
SpreadRegime = Literal["calm", "stressed"]

# §5.6-commodity — financing booked once, either zero (unlevered) or via MTF.
CommodityFinancingMode = Literal["unlevered", "mtf"]


@dataclass(frozen=True)
class CostMarketState:
    """All-``F_t``-measurable (causal) market state COST prices a round trip against (§10).

    Every field is either FACT read from Nautilus/EXECUTION at ``t`` or AMBER
    metadata. There is no future-dated input here; the deterministic funding
    SCHEDULE is derived from ``funding_interval_ms`` on the ``HoldHorizon``, the
    funding RATES are forecast (§5.5). When any input is missing/stale the
    conservative fallback fires (P6) — recorded in ``stale_flags``.
    """

    mid: float
    bid: float | None
    ask: float | None
    perp_mark: float | None  # basis (§5.5b)
    index_price: float | None  # basis (§5.5b)
    funding_cap: float  # AMBER venue per-interval cap; r_hat clamped to [-cap,+cap]
    interest_quote: float  # AMBER interest-baseline component (§5.5b)
    interest_base: float  # AMBER interest-baseline component (§5.5b)
    interest_rate_unit: InterestRateUnit  # AMBER; gates (8/24) scaling; NO silent default
    premium_index_8h: float | None  # AMBER published premium index; snapshot fallback if None
    published_next_funding: float | None  # venue pre-published rate; j==1 OVERRIDE (§5.5)
    recent_funding_rates: np.ndarray  # realized history -> EWMA (§5.5a)
    recent_funding_times: np.ndarray
    n_funding_events: int  # event-count floor (§5.5a); < floor -> fallback
    sigma: float  # PER-BAR return vol for impact (§5.4), same clock as k_i
    v_hat_exit_window: float  # CAUSAL forecast notional volume over the leg window (§5.4, Inv W)
    rolling_30d_volume: float  # fee tier (FACT from Nautilus fills)
    carry_class: CarryClass  # §5.6 per-class carry-model branch (AMBER)
    # impact coefficient — TEAL fit on this instrument's own realized series (§5.4);
    # None -> conservative linear fallback (RED). k_units flags fraction-vs-bps at ingestion.
    impact_k: float | None = None
    impact_k_units: Literal["fraction", "bps"] = "fraction"
    # residual quantile shock for the comonotone band (§5.5c). e_q is the CI-EDGE
    # block-bootstrap one-step residual quantile; None -> class fallback (fallback_band_thin).
    residual_e_q_ci_edge: float | None = None
    n_residual_events: int = 0  # event-count floor for the band (§5.5c)
    # cross-asset carry inputs (§5.6) — all AMBER, None when the class does not use them.
    futures_basis: float | None = None  # observed (F_far-F_near)/F_near for dated/commodity
    tau_spread_days: float | None = None  # far-near expiry gap in days (§5.6-time)
    roll_gap_quote: float | None = None  # discrete roll-gap cost booked on a roll in-hold
    mtf_rate: float | None = None  # AMBER, equities + financed commodity proxy (annual nominal)
    mtf_compounding: bool = False  # §5.6-time; default SIMPLE Actual/365
    borrowed_notional: float | None = None  # the leverage above 1x carrying MTF financing
    storage_rate: float | None = None  # AMBER quote-ccy per unit per YEAR (§5.6-commodity)
    commodity_financing_mode: CommodityFinancingMode | None = None  # AMBER (§5.6-commodity)
    commodity_basis_observable: bool = False  # if False -> storage-only, convenience=0 (P6)
    price: float | None = None  # underlying price for storage_rate/price (§5.6-commodity)
    spread_regime: SpreadRegime = "calm"
    pre_publishes_funding: bool = True  # AMBER; if False the blend governs j==1 too (§5.5)
    stale_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetEVDecision:
    """To BREAKOUT — numbers, never an action (P2, Invariant D).

    ``passes`` is priced at the conservative ``m_max`` (§5.5d) AND the side-selected
    conservative funding BAND (Invariant F-band-gate) — never the point forecast.
    ``binding_funding == round_trip.funding_band_conservative`` is the funding line
    actually summed into ``passes`` (Gap 1).
    """

    passes: bool
    expected_favorable: float
    round_trip: RoundTripCost
    binding_funding: float  # = round_trip.funding_band_conservative (Gap 1)
    edge_buffer: float
    margin: float  # expected_favorable - total - buffer (total uses binding_funding)
    funding_sensitivity: float  # dF/dm at m_max (§5.5d), mandatory telemetry
    funding_m_sensitive: bool  # verdict flips between m=1 and m=m_max
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class HoldCarryDecision:
    """To TREND — numbers, never an action (P2, Invariant D).

    ``forecast_residual_carry`` is signed (P7) and priced at the SAME side-selected
    conservative band the gate uses (Invariant T). A funding-earning short's
    conservative band is a SMALLER credit, never an inflated cost (Invariant P-band).
    """

    forecast_residual_carry: float  # signed, conservative m_max + side-selected band
    expected_residual_favorable: float
    carry_exceeds_residual: bool  # conservative-band carry > expected residual move
    funding_band_low: float  # raw -e_q comonotone sum (§5.5c)
    funding_band_high: float  # raw +e_q comonotone sum (§5.5c)
    funding_band_conservative: float  # side-selected, CI-edge band (Inv P-band/G-est)
    funding_sensitivity: float  # dF/dm (§5.5d)
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class TCAReport:
    """Post-trade implementation-shortfall decomposition, all FACT (§9, Invariant R).

    The eight IS lines (seven named + ``unattributed_residual``) sum to
    ``implementation_shortfall = gross - net`` EXACTLY by construction, because
    ``unattributed_residual`` is defined as the plug. Each cost line is POSITIVE when
    it REDUCES P&L; a funding CREDIT is a NEGATIVE line. ``funding_cost`` equals
    ``FundingAccrual`` (FACT), never the forecast.
    """

    delay_cost: float
    fee_cost: float
    half_spread_cost: float
    impact_cost: float
    funding_cost: float  # = FundingAccrual (FACT)
    carry_cost: float
    opportunity_cost: float
    unattributed_residual: float  # plug = (gross-net) - Σ(7 named); closes identity EXACTLY
    implementation_shortfall: float  # = gross - net; == sum of the 8 lines, by construction
    realized_vs_expected: Mapping[str, tuple[float, float]]  # line -> (forecast, realized)
    attribution_pct: Mapping[str, float]  # sums to 1.0
    engine_version: str
