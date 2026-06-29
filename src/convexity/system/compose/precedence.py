"""The deterministic A..G precedence pipeline (SYSTEM §2).

Each step is a pure function of F_t and the current BookState. The order is fixed and is itself the
conflict-resolution rule:

    TREND exit > ceiling collapse > governor scaling > hold incumbent (trend-alive)
              > selection demotion-with-margin > new-signal admission

There is NO place where a higher breakout score promotes a trade ahead of a risk constraint
(Invariant CB). SYSTEM's only multi-input combination is ``min()`` over ceilings (delegated to
LEVERAGE), set membership (selection), and this precedence — no product of scores anywhere. A
composed decision that depends on the *magnitude* of a breakout score is the bug (Invariant CB).

This module holds the per-step helpers; :mod:`convexity.system.compose.orchestrator` wires them into
``SystemOrchestrator.decide`` and runs the fixed point (Section 3) between steps D and F.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.core.types import EntrySignal
from convexity.system.compose.budget import Candidate
from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import BookState, OpenPosition, SystemAction


@dataclass(frozen=True)
class GovernorVerdict:
    """Step (A) GOVERNOR / KILL. ``flat_all`` halts new opens and flats the book at ≥43% drawdown
    (governor multiplier 0); else ``multiplier`` m ∈ [0,1] scales every size forward."""

    flat_all: bool
    multiplier: float
    reason: str


def governor_kill(book: BookState) -> GovernorVerdict:
    """Step (A): read the LEVERAGE drawdown governor on the whole book (Section 2).

    ``multiplier == 0`` (≥43% drawdown) ⇒ FLAT_ALL, halt new opens. Else carry m ∈ [0,1] forward; it
    scales every size. SYSTEM does not recompute the governor (Invariant P1) — it reads the
    multiplier off the BookState that LEVERAGE produced.
    """
    m = float(book.governor_multiplier)
    if m <= 0.0:
        return GovernorVerdict(flat_all=True, multiplier=0.0, reason="governor_flat_all")
    return GovernorVerdict(flat_all=False, multiplier=m, reason="governor_scale")


def open_position_actions(
    positions: Mapping[str, OpenPosition],
    trend_actions: Mapping[str, TrendAction],
    collapsed: frozenset[str],
) -> list[SystemAction]:
    """Step (B) OPEN POSITIONS FIRST (Section 2). Apply TrendAction; a collapsed ceiling REDUCES an
    incumbent (leverage may only reduce here, never add — adds go through D). Deterministic over ids.

    Precedence within (B): trend EXIT (close) > ceiling collapse (reduce) > trend REDUCE > continue.
    """
    actions: list[SystemAction] = []
    for inst in sorted(positions):
        pos = positions[inst]
        trend = trend_actions.get(
            inst, TrendAction(kind="continue", alive=True, target_fraction=1.0, time_cap=False)
        )
        if trend.kind == "exit":
            reason = ("trend_exit",) + (("time_cap",) if trend.time_cap else ())
            actions.append(
                SystemAction(inst, "close", 0, pos.side, "trend", "trend_exit", reason)
            )
        elif inst in collapsed:
            actions.append(
                SystemAction(inst, "reduce", 0, pos.side, "leverage", "ceiling_collapse",
                             ("ceiling_collapse",))
            )
        elif trend.kind == "reduce":
            target = int(pos.units * trend.target_fraction)
            actions.append(
                SystemAction(inst, "reduce", target, pos.side, "trend", "trend_reduce",
                             ("trend_reduce",))
            )
        else:  # continue ⇒ INCUMBENT, reserves its current risk budget
            actions.append(
                SystemAction(inst, "hold", pos.units, pos.side, "trend", "trend_continue",
                             ("trend_continue",))
            )
    return actions


def new_signal_candidates(
    sel_eligible: frozenset[str],
    qualified: Mapping[str, EntrySignal],
    open_instruments: frozenset[str],
    slack: Mapping[str, float],
    risk_weight: Mapping[str, float],
    heat: Mapping[str, float],
) -> list[Candidate]:
    """Step (D) NEW-SIGNAL ADMISSION candidate set (Section 2/3.4).

    A candidate is selection-eligible AND breakout-qualified AND not-open. The :class:`Candidate`
    carries only RISK quantities (ceiling-slack, risk-weight delivered, heat consumed) — NO breakout
    score (Invariant A/CB: side + qualification only). Deterministic over sorted ids.
    """
    out: list[Candidate] = []
    for inst in sorted(qualified):
        if inst in sel_eligible and inst not in open_instruments:
            out.append(
                Candidate(
                    instrument=inst,
                    ceiling_slack=float(slack.get(inst, 0.0)),
                    risk_weight_delivered=float(risk_weight.get(inst, 0.0)),
                    heat_consumed=float(heat.get(inst, 1.0)),
                )
            )
    return out
