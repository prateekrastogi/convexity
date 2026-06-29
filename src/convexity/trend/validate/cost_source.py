"""TrendCostSource — the COST.md-sourced friction for TREND's net-of-cost gates (§6.1, Inv M).

§6.1 / Inv M FORBID the local ``CostModel`` stub for any net-of-cost decision (the §1.5c net-EV gate
and the §13 promotion gate). Costs MUST come from COST.md's shared estimators — the SAME ones
BREAKOUT / LEVERAGE / TCA call — so TREND's gate and SYSTEM's "net-of-cost" label agree on one cost
source (COST P3 / Inv T). This adapter is the seam: it wraps the imported ``convexity.cost.cost_api``
(``net_ev_gate`` / ``hold_carry``) and returns the two friction numbers TREND's labeler needs:

  * ``round_trip_log`` — the per-trade round-trip friction in side-adjusted log-PnL units (the same
    units as ``Pi``), taken from the COST ``RoundTripCost`` produced by ``net_ev_gate``
    (``total_bps / 1e4``). This is what is subtracted from each gross ``Pi`` to get ``net_pi``.
  * ``cost_per_loss_r`` — the round-trip friction expressed in R0 units (round-trip log / R0), the
    ``avg_loss`` premium-bleed charge added on every failed cut in the §1.5c expectancy.

The funding-aware ``hold_carry`` residual is folded into the round-trip charge (a positive carry over
the remaining hold is a real friction on a held trend; a credit relaxes it), so the carry the gate
prices equals the COST ``HoldCarryInput`` (Inv T). VALIDATION-ONLY: imported by ``validate/`` and
``tune/``, never by ``estimate/`` or ``policy/`` (Inv G).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import HoldHorizon

# COST is DONE + importable; this is the one authoritative cost source (§6.1, Inv M / Inv T).
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.cost_api import CostAPI
from convexity.cost.types import CostMarketState, Side


@dataclass(frozen=True)
class TrendCostSource:
    """COST-sourced friction for TREND's §1.5c / §13 net gates (§6.1, Inv M). NOT the local stub."""

    api: CostAPI
    market: CostMarketState
    cfg: ResolvedCostConfig
    notional: float = 1.0

    def round_trip_log(
        self,
        side: Side,
        r0_risk: float,
        hold: HoldHorizon,
        expected_residual_favorable: float = 0.0,
    ) -> float:
        """Round-trip friction in log-PnL units from COST's ``net_ev_gate`` round trip, plus the
        funding-aware ``hold_carry`` residual over the remaining hold (Inv T). Same shared estimators
        BREAKOUT/LEVERAGE/TCA use — never the local ``CostModel`` stub (§6.1)."""
        gate = self.api.net_ev_gate(
            side, self.notional, expected_residual_favorable, "taker", "taker",
            hold, self.market, self.cfg,
        )
        rt_log = gate.round_trip.total_bps / 1e4  # bps -> log-PnL fraction
        # funding-aware hold carry over the remaining life (COST §8.2); a positive carry is a real
        # friction on a held trend, a credit (negative) relaxes it. Priced on the SAME carry model.
        carry = self.api.hold_carry(
            side, self.notional, hold, expected_residual_favorable, self.market, self.cfg,
        )
        carry_log = carry.forecast_residual_carry / max(self.notional, 1e-12)
        return float(rt_log + max(carry_log, 0.0))

    def cost_per_loss_r(
        self,
        side: Side,
        r0_risk: float,
        hold: HoldHorizon,
        expected_residual_favorable: float = 0.0,
    ) -> float:
        """The §1.5c premium-bleed charge per failed cut, in R0 units: round-trip friction / R0."""
        rt_log = self.round_trip_log(side, r0_risk, hold, expected_residual_favorable)
        return float(rt_log / r0_risk) if r0_risk > 0 else rt_log
