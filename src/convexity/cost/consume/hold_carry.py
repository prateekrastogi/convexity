"""HoldCarryInput — the funding-aware HOLD input handed to TREND (COST §8.2, P2).

Returns a ``HoldCarryDecision`` (numbers, never an action — Invariant D):

    forecast_residual_carry = side-selected CONSERVATIVE-BAND funding+carry over remaining_hold
    expected_residual_move  = supplied by TREND (edge, owned upstream)
    carry_exceeds_residual  = forecast_residual_carry > expected_residual_favorable

TREND consumes ``carry_exceeds_residual`` as ONE input to its hold/reduce/exit policy
("stop holding when forecast carry over remaining life exceeds the expected residual
favorable move"). COST does not exit; it states the relationship. The forecast carry
uses the SAME band the gate uses (Invariant T), priced over the residual hold's
conservative ``m_max`` (§5.5d) at the side-selected conservative band (§5.5c,
Invariant P-band/G-est) — NOT the point forecast. Signed (P7): a funding-earning
short's conservative band is a SMALLER credit, never an inflated cost, so the input
correctly RELAXES the hold for a short being paid.
"""

from __future__ import annotations

from convexity.core.types import HoldHorizon

from convexity.cost.components.carry import CarryModel
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostMarketState, HoldCarryDecision, Side


class HoldCarryInput:
    """TREND hold/reduce input (§8.2). Thin transform over the SAME carry model (Inv T)."""

    def __init__(self, carry: CarryModel | None = None) -> None:
        self.carry = carry if carry is not None else CarryModel()

    def residual(
        self,
        side: Side,
        notional: float,
        remaining_hold: HoldHorizon,
        expected_residual_favorable: float,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> HoldCarryDecision:
        """Funding-aware hold input (§8.2). The conservative-band carry vs the residual move."""
        carry_value, diag = self.carry.carry(
            side, notional, remaining_hold, market, cfg, use_band=True
        )
        band_low = float(diag.get("funding_band_low", carry_value))
        band_high = float(diag.get("funding_band_high", carry_value))
        band_conservative = float(diag.get("funding_band_conservative", carry_value))
        sensitivity = float(diag.get("funding_sensitivity", 0.0))

        # carry_exceeds_residual: the conservative-band CARRY (a cost when positive) eats
        # the residual move. A credit (negative carry) never exceeds a positive move, so a
        # funding-earning short correctly relaxes the hold (P7).
        carry_exceeds = carry_value > expected_residual_favorable

        reasons = [diag.get("reason", "carry")]
        if carry_exceeds:
            reasons.append("carry_exceeds_residual")
        if market.stale_flags:
            reasons.append("conservative_fallback")

        return HoldCarryDecision(
            forecast_residual_carry=carry_value,
            expected_residual_favorable=expected_residual_favorable,
            carry_exceeds_residual=carry_exceeds,
            funding_band_low=band_low,
            funding_band_high=band_high,
            funding_band_conservative=band_conservative,
            funding_sensitivity=sensitivity,
            reason_codes=tuple(reasons),
        )
