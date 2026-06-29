"""default_cost_model — the local validation-friction STUB (TREND §6.1).

The ``CostModel`` dataclass (shared core) is the local validation-friction STUB only, so the
``validate/`` layer can run self-contained in isolation. This module does NOT consume COST and is
NOT authoritative for any net-of-cost gate: the net-expectancy gate (§1.5c), the promotion gate
(§13), and anything SYSTEM labels "net-of-cost" MUST source costs from COST.md's shared estimators
(the same ones BREAKOUT / LEVERAGE / TCA call). That real wiring lives in
``validate/cost_source.TrendCostSource``, which imports ``convexity.cost.cost_api`` and feeds the
COST-sourced friction (``net_ev_gate`` round trip + ``hold_carry`` residual) into the labeler / the
promotion gate's ``e_pi_net`` (Inv M / Inv T). This stub may be used ONLY for standalone,
clearly-labeled validation friction when the COST estimators are not wired in (e.g. a unit test of
``validate/labels.py``).
"""

from __future__ import annotations

from convexity.core.types import CostModel


def default_cost_model() -> CostModel:
    """A conservative default validation-friction stub (§6.1). NOT for promotion/net-EV gates."""
    return CostModel(
        spread_bps=2.0,
        slippage_bps=1.0,
        fees_bps=4.0,  # taker round-trip-ish for crypto-perp
        min_tick_slippage=1.0,
    )
