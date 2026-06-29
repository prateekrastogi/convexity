"""The hold-incumbent firewall + rotation test (SYSTEM §4, Invariant HI).

The most consequential conflict: selection-wants-to-drop vs trend-wants-to-ride. Resolved by an
explicit monotone rule — NEVER by comparing a selection score to a trend score on the same axis
(they measure different things: selection ranks risk-diversification + candidate quality, trend
measures whether the open move is still alive). P4: an open, trend-alive position is governed by
TREND until TREND says exit; selection alone may not evict it.

For an open position ``i`` NOT in ``sel.keep`` (selection wants it out), Section 4:

    if TREND says EXIT                       -> close.            reason=trend_exit
    elif a LEVERAGE ceiling collapsed        -> reduce/close.     reason=ceiling_collapse
    elif TREND says trend NOT alive          -> allow demotion.   reason=sel_demote_dead
    elif TREND says CONTINUE (trend alive):
         replacement = best eligible, qualified, not-open candidate
         if replacement AND (rank margin > Δ_rank) AND (RISK improvement clears round-trip cost)
                                              -> rotate.          reason=sel_rotate
         else                                 -> HOLD.            reason=hold_incumbent

The rotation test is a RISK/COST test (does the swap improve book diversification per unit heat by
more than the round-trip cost?), NOT an edge test. We never sell a live winner to buy a higher-alpha
name (Invariant A). No sizing/exit math here (Invariant P1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import OpenPosition, ProposedPortfolio

IncumbentVerdict = Literal[
    "hold",          # held (in sel.keep, or trend-alive with no qualifying replacement)
    "trend_exit",    # TREND exit overrides selection
    "ceiling_collapse",  # a ceiling collapsed
    "sel_demote_dead",   # trend not alive ⇒ selection demotion allowed
    "sel_rotate",    # rotation cleared rank hysteresis + cost margin
]


@dataclass(frozen=True)
class IncumbentDecision:
    """The per-incumbent verdict (Section 4). ``replacement`` names the rotation target on
    ``sel_rotate`` (else None); ``reason`` is the §4 reason code."""

    instrument: str
    verdict: IncumbentVerdict
    replacement: str | None
    reason: str


def best_replacement(
    sel: ProposedPortfolio,
    open_instruments: frozenset[str],
    qualified: frozenset[str],
    incumbent: str,
) -> str | None:
    """The best selection-eligible, breakout-qualified, NOT-open candidate to replace ``incumbent``.

    "Best" is by the selection RISK rank (higher rank = better diversifier; Invariant A — never
    edge). Ties broken by instrument id (deterministic). Returns None if no candidate qualifies.
    """
    pool = [
        s
        for s in sel.eligible
        if s in qualified and s not in open_instruments and s != incumbent
    ]
    if not pool:
        return None
    return max(sorted(pool), key=lambda s: sel.rank.get(s, float("-inf")))


def resolve_incumbent(
    position: OpenPosition,
    trend: TrendAction,
    sel: ProposedPortfolio,
    *,
    ceiling_collapsed: bool,
    open_instruments: frozenset[str],
    qualified: frozenset[str],
    delta_rank: float,
    risk_improvement: float,
    round_trip_cost: float,
) -> IncumbentDecision:
    """Resolve one open position against selection's wish (Section 4, Invariant HI).

    ``risk_improvement`` is the diversification/heat improvement of the swap on a RISK basis
    (caller-computed from the MCTR decomposition; NOT an edge); ``round_trip_cost`` is the expected
    round-trip spread+impact+funding of the swap. A rotation requires BOTH the rank hysteresis AND
    that the risk improvement clears the cost.
    """
    inst = position.instrument

    # In sel.keep ⇒ selection is content; precedence (B) already handled trend exit/reduce upstream.
    if inst in sel.keep:
        return IncumbentDecision(inst, "hold", None, "hold_incumbent")

    # Selection wants it OUT. The §4 ladder, top wins:
    if trend.kind == "exit":
        return IncumbentDecision(inst, "trend_exit", None, "trend_exit")
    if ceiling_collapsed:
        return IncumbentDecision(inst, "ceiling_collapse", None, "ceiling_collapse")
    if not trend.alive:  # reduce / death ⇒ selection demotion allowed
        return IncumbentDecision(inst, "sel_demote_dead", None, "sel_demote_dead")

    # Trend ALIVE: the incumbent is winning. Selection alone may not evict it (P4) — only a
    # rotation that clears the rank hysteresis AND the round-trip cost on a RISK basis.
    replacement = best_replacement(sel, open_instruments, qualified, inst)
    if replacement is not None:
        incumbent_rank = sel.rank.get(inst, float("-inf"))
        replacement_rank = sel.rank.get(replacement, float("-inf"))
        rank_margin = replacement_rank - incumbent_rank
        if rank_margin > delta_rank and risk_improvement > round_trip_cost:
            return IncumbentDecision(inst, "sel_rotate", replacement, "sel_rotate")
    return IncumbentDecision(inst, "hold", None, "hold_incumbent")


def reconcile(
    positions: Mapping[str, OpenPosition],
    trend_actions: Mapping[str, TrendAction],
    sel: ProposedPortfolio,
    *,
    collapsed: frozenset[str],
    qualified: frozenset[str],
    delta_rank: float,
    risk_improvements: Mapping[str, float] | None = None,
    round_trip_costs: Mapping[str, float] | None = None,
) -> dict[str, IncumbentDecision]:
    """Resolve every open position (Section 2 step C / Section 4). Deterministic over sorted ids."""
    open_instruments = frozenset(positions)
    risk_improvements = risk_improvements or {}
    round_trip_costs = round_trip_costs or {}
    out: dict[str, IncumbentDecision] = {}
    for inst in sorted(positions):
        trend = trend_actions.get(
            inst, TrendAction(kind="continue", alive=True, target_fraction=1.0, time_cap=False)
        )
        out[inst] = resolve_incumbent(
            positions[inst],
            trend,
            sel,
            ceiling_collapsed=inst in collapsed,
            open_instruments=open_instruments,
            qualified=qualified,
            delta_rank=delta_rank,
            risk_improvement=risk_improvements.get(inst, 0.0),
            round_trip_cost=round_trip_costs.get(inst, float("inf")),
        )
    return out
