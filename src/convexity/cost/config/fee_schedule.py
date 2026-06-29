"""FeeSchedule / FeeTier — the ONE venue maker/taker fee truth (COST §7, P1).

AMBER metadata: venue maker/taker rates by VIP tier and the rolling-30d volume
breakpoints. This absorbs the legacy ``convexity.costs.binance_fees`` stub so there
is exactly one fee table (COST P1; ARCHITECTURE §5.1/§6). Rates are stored as
FRACTIONS of notional (0.0005 == 5.0 bps), per §5.7's worked example.

The tier is a deterministic STEP function over the breakpoints — no interpolation,
no learned schedule (§7). For the ~$1000 retail book the realistic tier is VIP-0,
but the same code reflects a tier change without a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeTier:
    """One VIP tier: the rolling-30d-volume floor that unlocks it and its rates (fractions)."""

    name: str
    min_30d_volume: float  # inclusive lower breakpoint in quote ccy (rolling 30d traded volume)
    maker_rate: float  # FRACTION of notional (negative == maker rebate)
    taker_rate: float  # FRACTION of notional


@dataclass(frozen=True)
class FeeSchedule:
    """A venue's full VIP ladder, ascending by ``min_30d_volume``.

    ``tier_for_volume`` is the deterministic step lookup; ``breakpoints`` exposes the
    volume thresholds so the gate's conservative near-breakpoint selection (§7,
    Invariant J-tier) can test proximity.
    """

    venue: str
    tiers: tuple[FeeTier, ...]

    def __post_init__(self) -> None:
        vols = [t.min_30d_volume for t in self.tiers]
        if vols != sorted(vols):
            raise ValueError("FeeSchedule tiers must be ascending by min_30d_volume")
        if not self.tiers:
            raise ValueError("FeeSchedule needs at least one tier")

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """The volume thresholds (excluding the VIP-0 zero floor) used by §7 proximity."""
        return tuple(t.min_30d_volume for t in self.tiers if t.min_30d_volume > 0.0)

    def tier_for_volume(self, rolling_30d_volume: float) -> FeeTier:
        """Deterministic step lookup: the highest tier whose floor the volume clears (§7)."""
        chosen = self.tiers[0]
        for tier in self.tiers:
            if rolling_30d_volume >= tier.min_30d_volume:
                chosen = tier
            else:
                break
        return chosen

    def tier_index(self, rolling_30d_volume: float) -> int:
        idx = 0
        for i, tier in enumerate(self.tiers):
            if rolling_30d_volume >= tier.min_30d_volume:
                idx = i
            else:
                break
        return idx

    def worse_fee_tier(self, a: FeeTier, b: FeeTier) -> FeeTier:
        """The higher-fee (worse-for-the-trade) of two tiers, taker rate as the tiebreaker (§7)."""
        return a if (a.taker_rate, a.maker_rate) >= (b.taker_rate, b.maker_rate) else b


# The Binance USDT-M perpetual-futures VIP ladder (AMBER). Rates as FRACTIONS.
# Absorbed from the retired ``convexity.costs.binance_fees`` stub — ONE fee truth (P1).
# VIP-0 retail: taker 0.0005 (5.0 bps), maker 0.0002 (2.0 bps). Higher tiers from the
# published USDT-M schedule; numbers are AMBER metadata, refreshed from the venue.
BINANCE_USDM_FEE_SCHEDULE = FeeSchedule(
    venue="BINANCE",
    tiers=(
        FeeTier("VIP0", 0.0, maker_rate=0.00020, taker_rate=0.00050),
        FeeTier("VIP1", 15_000_000.0, maker_rate=0.00016, taker_rate=0.00040),
        FeeTier("VIP2", 50_000_000.0, maker_rate=0.00014, taker_rate=0.00035),
        FeeTier("VIP3", 100_000_000.0, maker_rate=0.00012, taker_rate=0.00032),
        FeeTier("VIP4", 600_000_000.0, maker_rate=0.00010, taker_rate=0.00030),
        FeeTier("VIP5", 1_000_000_000.0, maker_rate=0.00008, taker_rate=0.00027),
    ),
)
