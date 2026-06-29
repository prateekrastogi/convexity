"""TCA — post-trade implementation-shortfall decomposition (COST §9, Invariant R/E).

Total IS = (decision price -> final P&L) gap, attributed to additive, NAMED lines.
The decomposition is an EXACT accounting IDENTITY by construction: the seven named
cost lines plus an explicit ``unattributed_residual`` PLUG sum to ``(gross - net)``
with ZERO residual by definition (Invariant R). The plug is defined as
``(gross - net) - Σ(seven named)`` so the eight-line sum equals the gap for EVERY
trade — there is no open sum that can drift.

SIGN CONVENTION (pinned, §9): every cost line is POSITIVE when it REDUCES P&L
(money out / shortfall), so the lines SUM to the measured ``(gross - net)`` gap
(itself positive when costs ate P&L). A funding CREDIT (short earning funding) is a
NEGATIVE cost line. ALL named terms are FACT (realized), never forecast —
``funding_cost`` equals ``FundingAccrual`` (Invariant A/R). Realized-vs-expected is
emitted per line (Invariant E) so forecast bias feeds validation.
"""

from __future__ import annotations

from convexity.cost.components.funding_accrual import FundingAccrual
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.facts.execution_adapter import ExecutionFacts
from convexity.cost.facts.nautilus_adapter import (
    ClosedTrade,
    NautilusFacts,
    ReferenceMarks,
)
from convexity.cost.types import TCAReport

_ENGINE_VERSION = "cost-tca-1.0.0"


class TCA:
    """Post-trade implementation-shortfall decomposition (§9). All FACT."""

    def __init__(self, funding_accrual: FundingAccrual | None = None) -> None:
        self.funding_accrual = funding_accrual if funding_accrual is not None else FundingAccrual()

    def decompose(
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
        """Decompose the closed trade's IS into the eight lines (§9, Invariant R).

        Each cost line POSITIVE when P&L-reducing. ``funding_cost`` is the accrued FACT
        summed from Nautilus events (Invariant A/N). The plug closes the identity exactly.
        """
        gross_minus_net = trade.gross_pnl - trade.net_pnl  # the measured IS gap

        # --- delay cost: decision_price -> arrival_price, signed by side, on filled qty.
        # A long whose arrival is above the decision price paid the lag (P&L-reducing).
        delay_cost = trade.side * (marks.arrival_price - marks.decision_price) * abs(
            trade.filled_qty
        )

        # --- fee cost: realized maker/taker fees (FACT, = Σ Nautilus Commission, Inv N).
        fee_cost = nautilus_facts.accrued_fees

        # --- half-spread cost: realized touch-vs-mid on each leg (FACT, from EXECUTION).
        half_spread_cost = execution_facts.realized_half_spread_cost

        # --- impact cost: realized slippage/impact (FACT, from EXECUTION, §5.4).
        impact_cost = execution_facts.realized_impact_cost

        # --- funding cost: F_accrued (FACT, summed from Nautilus funding events, §5.5).
        if t_entry is not None and t_now is not None:
            funding_cost = self.funding_accrual.accrued(
                trade.side, nautilus_facts.funding_events, t_entry, t_now
            )
        else:
            funding_cost = sum(
                trade.side * ev.notional_at_stamp * ev.rate
                for ev in nautilus_facts.funding_events
            )

        # --- carry cost: realized roll/MTF/storage carry over the hold (FACT).
        carry_cost = trade.realized_carry_cost

        # --- opportunity cost: unfilled/missed-portion shortfall valued at final mid.
        missed_qty = trade.intended_qty - trade.filled_qty
        opportunity_cost = trade.side * (marks.final_mid - marks.arrival_price) * missed_qty

        named_sum = (
            delay_cost
            + fee_cost
            + half_spread_cost
            + impact_cost
            + funding_cost
            + carry_cost
            + opportunity_cost
        )
        # The PLUG: closes the identity EXACTLY (Invariant R).
        unattributed_residual = gross_minus_net - named_sum

        # --- realized-vs-expected per line (Invariant E).
        exp = trade.expected_at_entry
        realized = {
            "delay_cost": delay_cost,
            "fee_cost": fee_cost,
            "half_spread_cost": half_spread_cost,
            "impact_cost": impact_cost,
            "funding_cost": funding_cost,
            "carry_cost": carry_cost,
            "opportunity_cost": opportunity_cost,
        }
        realized_vs_expected = {
            line: (float(exp.get(line, 0.0)), float(val)) for line, val in realized.items()
        }

        # --- attribution sums to 1.0 (over the eight lines including the residual).
        lines = {**realized, "unattributed_residual": unattributed_residual}
        denom = gross_minus_net if gross_minus_net != 0.0 else 1.0
        attribution_pct = {line: val / denom for line, val in lines.items()}

        return TCAReport(
            delay_cost=delay_cost,
            fee_cost=fee_cost,
            half_spread_cost=half_spread_cost,
            impact_cost=impact_cost,
            funding_cost=funding_cost,
            carry_cost=carry_cost,
            opportunity_cost=opportunity_cost,
            unattributed_residual=unattributed_residual,
            implementation_shortfall=gross_minus_net,
            realized_vs_expected=realized_vs_expected,
            attribution_pct=attribution_pct,
            engine_version=_ENGINE_VERSION,
        )
