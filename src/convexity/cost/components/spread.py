"""SpreadModel — half-spread cost from bid/ask, regime-bucketed, with fallback (COST §5.3).

The cost of crossing from mid to the touch on each leg:

    half_spread_bps = 1e4 * (ask - bid) / (2 * mid)
    C_hs_leg        = (half_spread_bps / 1e4) * Notional_leg

The ``/1e4`` here is CORRECT because ``half_spread_bps`` is genuinely in bps (§5.3,
contrast Invariant U-dim for impact). When bid/ask is unavailable, fall back to the
class-level conservative half-spread (RED, P6) — never zero (Invariant Q). Spread is
regime-bucketed (calm/stressed) per instrument; the gate uses the current regime's
expected half-spread, not a single global average.
"""

from __future__ import annotations

from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostComponent, CostMarketState


class SpreadModel:
    """Half-spread component (§5.3). Emits one ``CostComponent`` per leg."""

    def half_spread_bps(self, market: CostMarketState, cfg: ResolvedCostConfig) -> tuple[float, str]:
        """Return (half_spread_bps, reason). Falls back to the class wide spread when
        bid/ask is missing or stale (P6, Invariant Q) — never zero."""
        if (
            market.bid is None
            or market.ask is None
            or market.mid <= 0.0
            or "spread" in market.stale_flags
        ):
            base = cfg.fallback_half_spread_bps
            # A stressed regime widens the conservative fallback (§5.3 regime bucketing).
            mult = 2.0 if market.spread_regime == "stressed" else 1.0
            return base * mult, "fallback_wide_spread"
        hs = 1e4 * (market.ask - market.bid) / (2.0 * market.mid)
        if hs <= 0.0:
            return cfg.fallback_half_spread_bps, "fallback_nonpositive_spread"
        return hs, "observed_spread"

    def leg_component(
        self, market: CostMarketState, notional_leg: float, name: str, cfg: ResolvedCostConfig
    ) -> CostComponent:
        """One half-spread leg: ``(half_spread_bps / 1e4) * Notional_leg`` (§5.3)."""
        hs_bps, reason = self.half_spread_bps(market, cfg)
        quote = (hs_bps / 1e4) * notional_leg
        return CostComponent(
            name=name,
            quote_ccy=quote,
            bps_of_notional=hs_bps,
            is_fact=False,
            reason=reason,
        )
