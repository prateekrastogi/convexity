"""FeeModel — rolling-volume -> VIP tier -> effective maker/taker fee (COST §7, §5.2).

The tier is a deterministic STEP function over AMBER volume breakpoints (no
interpolation, no learned schedule). ``effective_fees`` is the forward lookup (given
a volume, the tier in force). ``gate_tier`` is the GATE's conservative selection: it
prices fees at the HIGHER-FEE (worse) tier whenever ``rolling_30d_volume`` is within
``tier_proximity`` of a breakpoint (or sits on it), because the tier a trade pays is
a forecast question, not a lookup (P6, Invariant J-tier). ``volume_to_next`` feeds
the proximity test — telemetry that informs the FEE estimate only, never a trade
trigger (Invariant D). TCA still books realized fees at the in-force tier (FACT).
"""

from __future__ import annotations

from convexity.cost.config.fee_schedule import FeeSchedule, FeeTier
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostComponent, LiquidityFlag


class FeeModel:
    """Effective fee component (§5.2/§7). Emits one ``CostComponent`` per leg."""

    def effective_fees(
        self,
        rolling_30d_volume: float,
        fee_schedule: FeeSchedule,
    ) -> tuple[float, float, dict]:
        """Forward lookup: volume -> tier -> (maker_rate, taker_rate, diagnostics) (§7).

        Returns the effective rates and diagnostics ``{tier, next_breakpoint,
        volume_to_next}``. ``volume_to_next`` surfaces how close the book is to a
        cheaper tier — informational, never a trade trigger (Invariant D).
        """
        tier = fee_schedule.tier_for_volume(rolling_30d_volume)
        idx = fee_schedule.tier_index(rolling_30d_volume)
        next_breakpoint: float | None = None
        if idx + 1 < len(fee_schedule.tiers):
            next_breakpoint = fee_schedule.tiers[idx + 1].min_30d_volume
        volume_to_next = (
            next_breakpoint - rolling_30d_volume if next_breakpoint is not None else float("inf")
        )
        diag = {
            "tier": tier.name,
            "next_breakpoint": next_breakpoint,
            "volume_to_next": volume_to_next,
        }
        return tier.maker_rate, tier.taker_rate, diag

    def gate_tier(
        self, rolling_30d_volume: float, fee_schedule: FeeSchedule, cfg: ResolvedCostConfig
    ) -> tuple[FeeTier, str]:
        """The tier the GATE prices the round-trip fee at — conservative near a breakpoint (§7).

        Within ``tier_proximity`` (a fraction of the breakpoint) of any VIP breakpoint
        — including sitting exactly on it — price at the WORSE of the in-force tier and
        the next-tier-up-in-fee (P6, Invariant J-tier). Away from any breakpoint use
        the plain step lookup. Returns (tier, reason).
        """
        tier = fee_schedule.tier_for_volume(rolling_30d_volume)
        idx = fee_schedule.tier_index(rolling_30d_volume)
        prox = cfg.defaults.tier_proximity
        for bp in fee_schedule.breakpoints:
            if bp <= 0:
                continue
            if abs(rolling_30d_volume - bp) <= prox * bp:
                # The fee changes ACROSS this breakpoint. Lower volume = the worse
                # (higher-fee) side. Compare the in-force tier with the tier just below
                # the breakpoint (the higher-fee neighbour).
                below_idx = max(idx - 1, 0) if rolling_30d_volume >= bp else idx
                worse = fee_schedule.worse_fee_tier(tier, fee_schedule.tiers[below_idx])
                return worse, "tier_proximity_conservative"
        return tier, "tier_lookup"

    def leg_component(
        self,
        liquidity_flag: LiquidityFlag,
        notional_leg: float,
        tier: FeeTier,
        name: str,
        *,
        is_fact: bool,
        reason: str,
    ) -> CostComponent:
        """One fee leg: ``rate(tier) * Notional_leg`` (§5.2). ``rate`` may be negative
        (maker rebate); the line is dual-unit (quote ccy + bps)."""
        rate = tier.taker_rate if liquidity_flag == "taker" else tier.maker_rate
        quote = rate * notional_leg
        bps = 1e4 * rate
        return CostComponent(
            name=name, quote_ccy=quote, bps_of_notional=bps, is_fact=is_fact, reason=reason
        )
