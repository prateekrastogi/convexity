"""CostAPI — the public entry point: ONE CostBook feeds the four consumers (COST §8, Inv T).

The single live cost truth consumed four ways (P1). One ``CostBook`` (one set of
estimators, one resolved config) backs the net-EV gate (BREAKOUT), the funding-aware
hold input (TREND), the carry term (LEVERAGE), and the carry side of TCA — so the
carry the gate priced at entry equals ``CarryTerm.over_horizon`` equals what
``HoldCarryInput`` uses, bit-for-bit (Invariant T). COST is a PROVIDER, never a
decider (P2): every method returns numbers/booleans/decompositions, never an action
(Invariant D). Satisfies the ``convexity.core.protocols.CostEngine`` seam.
"""

from __future__ import annotations

from typing import Mapping

from convexity.core.types import HoldHorizon, RoundTripCost

from convexity.cost.assemble.cost_book import CostBook
from convexity.cost.components.carry import CarryModel
from convexity.cost.components.fees import FeeModel
from convexity.cost.components.funding_accrual import FundingAccrual
from convexity.cost.components.funding_forecast import FundingForecast
from convexity.cost.components.impact import ImpactModel
from convexity.cost.components.spread import SpreadModel
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.consume.carry_term import CarryTerm
from convexity.cost.consume.hold_carry import HoldCarryInput
from convexity.cost.consume.net_ev_gate import NetEVGate
from convexity.cost.consume.tca import TCA
from convexity.cost.facts.execution_adapter import ExecutionFacts
from convexity.cost.facts.nautilus_adapter import (
    ClosedTrade,
    NautilusFacts,
    ReferenceMarks,
)
from convexity.cost.types import (
    CostMarketState,
    HoldCarryDecision,
    LiquidityFlag,
    NetEVDecision,
    Side,
    TCAReport,
)


class CostAPI:
    """One cost truth, four consumers (COST §8). Build once, share the estimators (Inv T)."""

    def __init__(self) -> None:
        # ONE shared estimator stack — every consumer reads the SAME models (Invariant T).
        self.funding = FundingForecast()
        self.carry = CarryModel(self.funding)
        self.cost_book = CostBook(
            fees=FeeModel(),
            spread=SpreadModel(),
            impact=ImpactModel(),
            funding=self.funding,
            carry=self.carry,
        )
        self.net_ev_gate_engine = NetEVGate(self.cost_book)
        self.hold_carry_engine = HoldCarryInput(self.carry)
        self.carry_term_engine = CarryTerm(self.carry)
        self.tca_engine = TCA(FundingAccrual())

    # ---------------------------------------------------------------- BREAKOUT (§8.1)

    def net_ev_gate(
        self,
        side: Side,
        notional: float,
        expected_favorable: float,
        liquidity_flag_in: LiquidityFlag,
        liquidity_flag_out: LiquidityFlag,
        expected_hold: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> NetEVDecision:
        """BREAKOUT entry gate (§8.1)."""
        return self.net_ev_gate_engine.evaluate(
            side, notional, expected_favorable, liquidity_flag_in, liquidity_flag_out,
            expected_hold, market, cfg,
        )

    # ---------------------------------------------------------------- TREND (§8.2)

    def hold_carry(
        self,
        side: Side,
        notional: float,
        remaining_hold: HoldHorizon,
        expected_residual_favorable: float,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> HoldCarryDecision:
        """TREND funding-aware hold/reduce input (§8.2)."""
        return self.hold_carry_engine.residual(
            side, notional, remaining_hold, expected_residual_favorable, market, cfg,
        )

    # ---------------------------------------------------------------- LEVERAGE (§8.3)

    def carry_term(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> tuple[float, Mapping[str, float]]:
        """LEVERAGE stop-ceiling SIGNED carry term (§8.3). LEVERAGE applies max(carry,0)."""
        return self.carry_term_engine.over_horizon(side, notional, horizon, market, cfg)

    # ---------------------------------------------------------------- REGIME cost gate

    def round_trip(
        self,
        side: Side,
        notional: float,
        liquidity_flag_in: LiquidityFlag,
        liquidity_flag_out: LiquidityFlag,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
        expected_favorable: float | None = None,
    ) -> RoundTripCost:
        """REGIME's cost gate input — the additive round-trip estimate (ARCHITECTURE §4)."""
        return self.cost_book.round_trip(
            side, notional, liquidity_flag_in, liquidity_flag_out, horizon, market, cfg,
            expected_favorable=expected_favorable,
        )

    # ---------------------------------------------------------------- post-trade (§9)

    def tca(
        self,
        trade: ClosedTrade,
        marks: ReferenceMarks,
        nautilus_facts: NautilusFacts,
        execution_facts: ExecutionFacts,
        cfg: ResolvedCostConfig,
        *,
        t_entry: int | None = None,
        t_now: int | None = None,
    ) -> TCAReport:
        """Post-trade TCA decomposition (§9)."""
        return self.tca_engine.decompose(
            trade, marks, nautilus_facts, execution_facts, cfg, t_entry=t_entry, t_now=t_now,
        )
