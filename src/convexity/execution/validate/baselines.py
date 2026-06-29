"""Baselines — always-MARKET / always-POST-ONLY / naive-cross / TWAP (§12, firewalled, Invariant V).

The required §12 baselines, run under IDENTICAL fills / costs / benchmark. The engine must BEAT
always-MARKET on patient / calm flow and must NOT lose to always-MARKET on urgent breakout flow
(where posting is the trap it avoids). Validation-only — never imported by the live layers.

Each baseline returns an expected IS in bps from the same :class:`TacticCosts` the engine sees, so
the comparison is apples-to-apples (the engine's choice is just the argmin; a baseline FIXES the
tactic regardless of cost).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.types import TacticCosts

__all__ = ["BaselineCosts", "compute_baselines"]


@dataclass(frozen=True)
class BaselineCosts:
    """Expected IS (bps) for each fixed-tactic baseline + the engine's argmin (§12)."""

    always_market: float
    always_post: float
    naive_cross: float  # cross immediately on the signal (== marketable limit, the urgent default)
    twap: float  # equal-weight average of the crossing tactics (a coarse slice proxy)
    engine_argmin: float


def compute_baselines(costs: TacticCosts) -> BaselineCosts:
    """Expected IS for each baseline from one :class:`TacticCosts` (§12). Validation-only (Inv V)."""
    market = costs.by_tactic["MARKET"]
    post = costs.by_tactic["POST_ONLY_MAKER"]
    marketable = costs.by_tactic["MARKETABLE_LIMIT"]
    limit_ioc = costs.by_tactic["LIMIT_IOC"]
    twap = (market + marketable + limit_ioc) / 3.0
    argmin = min(costs.by_tactic.values())
    return BaselineCosts(
        always_market=market,
        always_post=post,
        naive_cross=marketable,
        twap=twap,
        engine_argmin=argmin,
    )
