"""CostBook — additive RoundTripCost assembly (COST §5, P3, Invariant B/A).

Sums the six independent component estimators into the additive ``RoundTripCost``
(P3 — never a multiplicative score). ``total_quote_ccy == sum(c.quote_ccy)`` exactly
(Invariant B). The funding line summed into the total is the side-selected
CONSERVATIVE BAND ``funding_band_conservative`` (the GATE's binding number, Gap 1);
the point ``funding_forecast`` and the raw band ends are carried as separate
telemetry fields, never summed into the total (Invariant F-band-gate). Accrued FACT
funding is NEVER merged here — that is a TCA input only (Invariant A). CostBook
NEVER decides (P2): it returns numbers.
"""

from __future__ import annotations

from convexity.core.types import HoldHorizon

from convexity.cost.components.carry import CarryModel
from convexity.cost.components.fees import FeeModel
from convexity.cost.components.funding_forecast import FundingForecast
from convexity.cost.components.impact import ImpactModel
from convexity.cost.components.spread import SpreadModel
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import (
    CostComponent,
    CostMarketState,
    LiquidityFlag,
    RoundTripCost,
    Side,
)


class CostBook:
    """Assemble the additive round-trip cost (§5). One instance serves all consumers (Inv T)."""

    def __init__(
        self,
        fees: FeeModel | None = None,
        spread: SpreadModel | None = None,
        impact: ImpactModel | None = None,
        funding: FundingForecast | None = None,
        carry: CarryModel | None = None,
    ) -> None:
        self.fees = fees if fees is not None else FeeModel()
        self.spread = spread if spread is not None else SpreadModel()
        self.impact = impact if impact is not None else ImpactModel()
        self.funding = funding if funding is not None else FundingForecast()
        self.carry = carry if carry is not None else CarryModel(self.funding)

    def round_trip(
        self,
        side: Side,
        notional: float,
        liquidity_flag_in: LiquidityFlag,
        liquidity_flag_out: LiquidityFlag,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
        *,
        expected_favorable: float | None = None,
        gate_fee_tier: bool = True,
    ) -> RoundTripCost:
        """The additive ``RoundTripCost`` (§5, P3). Funding summed at the conservative BAND.

        ``gate_fee_tier`` selects the conservative near-breakpoint tier (§7, Invariant
        J-tier) for the GATE; the assembled cost is what the gate/regime cost-gate
        compares against. ``cost_fraction`` is filled when ``expected_favorable`` is
        supplied (§6) — diagnostic only, never the gate boolean.
        """
        # --- fee tier (conservative near a breakpoint for the gate, §7).
        if gate_fee_tier:
            tier, tier_reason = self.fees.gate_tier(
                market.rolling_30d_volume, cfg.fee_schedule, cfg
            )
        else:
            # TCA / non-gate path: the plain in-force tier (FACT, no conservative bump, §7).
            tier = cfg.fee_schedule.tier_for_volume(market.rolling_30d_volume)
            tier_reason = "tier_lookup"

        fee_in = self.fees.leg_component(
            liquidity_flag_in, notional, tier, "fee_in", is_fact=False, reason=tier_reason
        )
        fee_out = self.fees.leg_component(
            liquidity_flag_out, notional, tier, "fee_out", is_fact=False, reason=tier_reason
        )

        # --- half-spread (§5.3).
        hs_in = self.spread.leg_component(market, notional, "half_spread_in", cfg)
        hs_out = self.spread.leg_component(market, notional, "half_spread_out", cfg)

        # --- impact (§5.4, FRACTION × notional, NO /1e4).
        imp_in = self.impact.leg_component(market, notional, "impact_in", cfg)
        imp_out = self.impact.leg_component(market, notional, "impact_out", cfg)

        # --- funding/carry over the conservative m_max (§5.5d), at the side-selected band.
        carry_value, carry_diag = self.carry.carry(
            side, notional, horizon, market, cfg, use_band=True
        )
        # The funding/carry component summed into the total IS the binding band value
        # (Gap 1). For crypto_perp the diagnostics carry the point + raw band ends.
        funding_point = float(carry_diag.get("funding_forecast_point", carry_value))
        band_low = float(carry_diag.get("funding_band_low", carry_value))
        band_high = float(carry_diag.get("funding_band_high", carry_value))
        band_conservative = float(carry_diag.get("funding_band_conservative", carry_value))
        funding_bps = 1e4 * band_conservative / notional if notional else 0.0
        funding_component = CostComponent(
            name="funding_forecast",
            quote_ccy=band_conservative,  # the BINDING band value, summed into the total (Gap 1)
            bps_of_notional=funding_bps,
            is_fact=False,
            reason=carry_diag.get("reason", "carry"),
        )

        components = (
            fee_in,
            fee_out,
            hs_in,
            hs_out,
            imp_in,
            imp_out,
            funding_component,
        )
        total_quote = sum(c.quote_ccy for c in components)  # Invariant B: additive sum
        total_bps = 1e4 * total_quote / notional if notional else 0.0

        cost_fraction: float | None = None
        if expected_favorable is not None and expected_favorable != 0.0:
            cost_fraction = total_quote / expected_favorable

        return RoundTripCost(
            components=components,
            total_quote_ccy=total_quote,
            total_bps=total_bps,
            funding_forecast=funding_point,  # POINT blend, telemetry (Inv F-band-gate)
            funding_band_low=band_low,
            funding_band_high=band_high,
            funding_band_conservative=band_conservative,  # the gate summand (Gap 1)
            cost_fraction=cost_fraction,
        )

    def funding_sensitivity(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> float:
        """``dF/dm`` at ``m_max`` (§5.5d), exposed for the gate/hold telemetry."""
        _, diag = self.carry.carry(side, notional, horizon, market, cfg, use_band=True)
        return float(diag.get("funding_sensitivity", 0.0))
