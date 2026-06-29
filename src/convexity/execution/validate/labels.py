"""RealizedLabeler — realized IS / adverse-selection / slippage labels (§12, firewalled, Invariant V).

VALIDATION-ONLY: uses FUTURE bars (the post-fill adverse-selection horizon, the realized fill path),
so it lives here and is NEVER imported by the live microstate / cost / policy / nautilus layers
(Invariant V — asserted by the AST import-graph test). The objective is execution COST (IS), never
trade P&L (which belongs to the signal).

Each label is computed vs the FIXED ``arrival_mid`` (§5.2): the realized mid walk from arrival to the
fill is the delay term, the post-fill adverse drift over the horizon is the adverse-selection term,
the half-spread paid is the spread term, and impact is the residual at the $1k clip (~0, Invariant I).
"""

from __future__ import annotations

import numpy as np

from convexity.execution.types import ExecTarget, RealizedFill, RealizedStopFill

__all__ = ["RealizedLabeler"]


class RealizedLabeler:
    """Builds firewalled realized-cost labels from future bars (§12 / Invariant V)."""

    def label_fill(
        self,
        target: ExecTarget,
        avg_fill_price: float,
        future_mids: np.ndarray,
        half_spread_bps: float,
        filled_qty: float,
        maker_filled_qty: float,
        realized_funding_bps: float = 0.0,
        aborted: bool = False,
    ) -> RealizedFill:
        """Decompose realized IS vs the FIXED arrival mid into spread / impact / adverse / delay (§12).

        ``future_mids`` is the post-fill mid path over the adverse-selection horizon (FUTURE bars —
        firewalled). All terms signed bps for the trade's own side: a cost > 0 is money lost.
        """
        arrival = target.arrival_mid
        side = target.side
        # Total IS = signed (fill - arrival) for the side, in bps.
        is_bps = side * 10_000.0 * (avg_fill_price - arrival) / arrival if arrival else 0.0
        # Spread term: the half-spread paid on the taker-filled share (maker keeps it).
        taker_share = max(0.0, filled_qty - maker_filled_qty) / max(filled_qty, 1e-12)
        spread_bps = half_spread_bps * taker_share
        # Adverse term: the post-fill adverse drift over the horizon (future mids), conditional on
        # the fill — for a long, a continued DOWN move is adverse.
        if future_mids.size:
            end_mid = float(future_mids[-1])
            adverse_bps = -side * 10_000.0 * (end_mid - avg_fill_price) / avg_fill_price
            adverse_bps = max(adverse_bps, 0.0)  # adverse excursion only (favorable is not a cost)
        else:
            adverse_bps = 0.0
        # Delay term: the realized arrival->fill mid walk (booked vs arrival, §5.2).
        delay_bps = max(is_bps - spread_bps, 0.0)
        # Impact: residual (~0 at the $1k clip, Invariant I).
        impact_bps = max(is_bps - spread_bps - delay_bps, 0.0)
        return RealizedFill(
            intent_id=target.intent_id,
            realized_avg_price=avg_fill_price,
            realized_is_bps=is_bps,
            realized_spread_bps=spread_bps,
            realized_impact_bps=impact_bps,
            realized_adverse_sel_bps=adverse_bps,
            realized_delay_bps=delay_bps,
            realized_funding_bps=realized_funding_bps,
            realized_aborted=aborted,
            filled_qty=filled_qty,
            maker_filled_qty=maker_filled_qty,
            taker_filled_qty=max(0.0, filled_qty - maker_filled_qty),
        )

    def label_stop_fill(
        self,
        intent_id: str,
        trigger_price: float,
        realized_fill_price: float,
        bot_was_down: bool,
    ) -> RealizedStopFill:
        """The protective stop's realized gap-slippage, measured SEPARATELY from IS (§12).

        ``gap_slippage_bps`` is the cost a STOP_MARKET pays when price gaps THROUGH the trigger; it
        feeds LEVERAGE's gap-survival ceiling, NOT entry/exit IS (§12). ``gapped_through`` is True when
        the fill is worse than the trigger (price jumped past it without trading at it).
        """
        slip = 0.0
        if trigger_price:
            slip = 10_000.0 * abs(realized_fill_price - trigger_price) / trigger_price
        gapped = abs(realized_fill_price - trigger_price) > 0.0
        return RealizedStopFill(
            intent_id=intent_id,
            trigger_price=trigger_price,
            realized_fill_price=realized_fill_price,
            gap_slippage_bps=slip,
            gapped_through=gapped,
            bot_was_down=bot_was_down,
        )
