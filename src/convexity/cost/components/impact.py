"""ImpactModel — expected slippage/impact, the square-root FRACTION (COST §5.4).

The Almgren square-root law ``k_i * sigma_i * sqrt(participation)`` is already a
RETURN FRACTION, not basis points. So the impact cost in quote currency is the
fraction times notional, with NO ``/1e4`` (Invariant U-dim — the rogue ``/1e4`` is
the bug this closes):

    impact_frac_leg = k_i * sigma_i * sqrt(participation_leg)   # a FRACTION
    C_impact_leg    = impact_frac_leg * Notional_leg            # quote ccy, NO /1e4
    impact_bps_leg  = 1e4 * impact_frac_leg                     # bps column only
    participation_leg = Notional_leg / V_hat_leg

``V_hat_leg`` is a CAUSAL forecast notional volume built only from PAST volume
(Invariant W) — never realized future volume. A conservative LOW volume forecast
(inflating participation, hence impact) is the P6 fallback. When ``k_i`` cannot be
fitted, a conservative linear fallback (RED fraction coefficient) governs. For
realized TCA the EXECUTION-supplied realized slippage replaces this forecast (§9).
"""

from __future__ import annotations

import math

from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostComponent, CostMarketState


class ImpactModel:
    """Expected-impact component (§5.4). Emits one ``CostComponent`` per leg."""

    def impact_fraction(
        self, market: CostMarketState, notional_leg: float, cfg: ResolvedCostConfig
    ) -> tuple[float, str]:
        """Return (impact_frac, reason). The dimensionless square-root fraction (§5.4).

        ``participation = Notional_leg / v_hat`` uses the CAUSAL forecast volume
        (Invariant W). When ``impact_k`` is missing/stale, the conservative linear
        fallback coefficient (RED, fraction) governs (P6).
        """
        v_hat = market.v_hat_exit_window
        if v_hat <= 0.0:
            # Degenerate volume forecast -> maximally conservative participation of 1.
            participation = 1.0
        else:
            participation = notional_leg / v_hat

        if market.impact_k is None or "impact" in market.stale_flags:
            # Conservative LINEAR fallback: k_fallback * participation (RED fraction).
            frac = cfg.fallback_impact_k_fraction * participation
            return frac, "fallback_linear_impact"

        # k_i may arrive in bps (an AMBER class fallback) — convert to a fraction ONCE.
        k = market.impact_k / 1e4 if market.impact_k_units == "bps" else market.impact_k
        frac = k * market.sigma * math.sqrt(max(participation, 0.0))
        reason = "sqrt_impact" if market.impact_k_units == "fraction" else "sqrt_impact_k_bps"
        return frac, reason

    def leg_component(
        self, market: CostMarketState, notional_leg: float, name: str, cfg: ResolvedCostConfig
    ) -> CostComponent:
        """One impact leg: ``impact_frac * Notional_leg`` — NO ``/1e4`` (Invariant U-dim)."""
        frac, reason = self.impact_fraction(market, notional_leg, cfg)
        quote = frac * notional_leg  # fraction multiplies notional directly (U-dim)
        bps = 1e4 * frac  # bps recovered by MULTIPLYING, never by dividing the cost
        return CostComponent(
            name=name, quote_ccy=quote, bps_of_notional=bps, is_fact=False, reason=reason
        )
