"""Residual-budget priority by measurable risk (SYSTEM §3.4, Invariant A).

When the residual budget ``H_book − Σ_incumbent MCTR`` cannot admit all qualified new candidates at
minimum tradable size, admit by MEASURABLE-RISK priority, in order (Section 3.4):

  1. Highest binding-ceiling SLACK (room before the nearest failure mode) — admit positions
     furthest from any ceiling first.
  2. Highest MCTR-efficiency = (risk-weight delivered) / (book heat consumed) — the most
     diversifying instrument per unit of shared budget.
  3. Tie-break: instrument id (deterministic).

Breakout score, expected return, expected favorable excursion, and trend confidence are FORBIDDEN
as priority inputs (the firewall, Invariant A inherited from LEVERAGE.md). There is NO edge field on
:class:`Candidate` — the firewall is enforced by the *absence* of the field. A position that cannot
be admitted is logged ``budget_deferred``, not silently dropped (Section 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Candidate:
    """A new-capital candidate competing for the residual budget (Section 3.4).

    Carries NO edge field (no breakout_score / expected_return / trend_confidence): the edge/risk
    firewall is the absence of the field (Invariant A). ``ceiling_slack`` = room before the nearest
    binding ceiling; ``risk_weight_delivered`` = the diversifying risk-weight one lot adds;
    ``heat_consumed`` = the book-heat (MCTR) one lot consumes.
    """

    instrument: str
    ceiling_slack: float
    risk_weight_delivered: float
    heat_consumed: float

    @property
    def mctr_efficiency(self) -> float:
        """(risk-weight delivered) / (book heat consumed). Higher = more diversifying per unit heat."""
        if self.heat_consumed <= 0.0:
            return float("inf")
        return self.risk_weight_delivered / self.heat_consumed


def priority_order(
    candidates: Sequence[Candidate], *, budget_priority: str = "slack_then_mctr"
) -> list[Candidate]:
    """Order candidates by the §3.4 measurable-risk priority (Invariant A).

    Default ``slack_then_mctr``: ceiling-slack DESC, then MCTR-efficiency DESC, then instrument id
    ASC (deterministic). The BLUE ``mctr_then_slack`` variant flips the first two keys — both are
    pure risk keys; neither admits any edge/score input. The tie-break on id makes the order a
    deterministic total order (Section 3.3).
    """
    if budget_priority == "mctr_then_slack":
        key = lambda c: (-c.mctr_efficiency, -c.ceiling_slack, c.instrument)  # noqa: E731
    else:
        key = lambda c: (-c.ceiling_slack, -c.mctr_efficiency, c.instrument)  # noqa: E731
    return sorted(candidates, key=key)
