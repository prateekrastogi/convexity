"""SpreadCost — the half-spread + fee term, signed bps vs arrival mid (§5.1).

Crossing pays the effective half-spread ``s`` plus the taker fee ``f_t``; resting AT the touch saves
the full half-spread and pays the maker fee ``f_m`` (negative if a rebate). The captured price
improvement at offset δ is ``PI = s − δ·tick_bps`` (the spread you keep — NOT a flat ``−δ``, §5.1).

This component is computed INDEPENDENTLY of impact / adverse / delay (Invariant B): a bad impact
model cannot corrupt the spread term. Fees come from the AMBER schedule on the instrument spec (§4).
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig

__all__ = ["SpreadCost"]


class SpreadCost:
    """The half-spread + fee leg of the cost decomposition (§5.1). Independent of other tactics (Inv B)."""

    def taker_cost_bps(self, half_spread_bps: float, cfg: ResolvedConfig) -> float:
        """``s + f_t`` — crossing pays the half-spread and the taker fee (§5.1). Impact added elsewhere."""
        return float(half_spread_bps + cfg.instrument.taker_fee_bps)

    def maker_fee_bps(self, cfg: ResolvedConfig) -> float:
        """``f_m`` — the maker fee in bps, negative if a rebate (§5.1)."""
        return float(cfg.instrument.maker_fee_bps - cfg.instrument.maker_rebate_bps)

    def price_improvement_bps(
        self, half_spread_bps: float, offset_ticks: float, tick_bps: float
    ) -> float:
        """``PI = s − δ·tick_bps`` — the captured half-spread when a maker fills at offset δ (§5.1).

        At δ = 0 (rest at touch) ``PI = s`` (the whole half-spread saved); a deeper post captures
        less; ``PI`` can go negative for a deep post past the opposite touch (correctly priced — such
        offsets are dominated and pruned by the δ* argmin).
        """
        return float(half_spread_bps - offset_ticks * tick_bps)
