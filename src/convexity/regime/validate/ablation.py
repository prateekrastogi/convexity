"""Which measure binds + the ARL audit (REGIME §9, §3.6d, firewalled — Invariant V).

Two first-class acceptance reports:
  * **Binding ablation** — how often each of the four sub-states is the binding one. A measure that
    NEVER binds is redundant or mis-scaled; one that ALWAYS binds is the only real signal. Mirrors
    LEVERAGE's "which ceiling binds" ablation (REGIME §9).
  * **ARL audit** — on held-out CALM stretches, the REALIZED per-bar escalation rate into each state
    vs the ``1/alpha_k_effective`` budget (REGIME §3.6d). The LIQ channel is audited on TWO budgets:
    its decision-bar ``ARL0^liq`` (off the ``1 − p²`` null) and, SEPARATELY, the event-driven path's
    per-MINUTE ``ARL0^liq_min`` — the decision-bar audit does NOT cover the per-minute path. "How
    often does the brake fire on noise" is answerable here, not in production (P7b).

Nothing in the live layers imports this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from convexity.core.types import RegimeState
from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.state.bands import arl0_budget
from convexity.regime.types import SubName


def binding_ablation(states: Iterable[RegimeState]) -> Mapping[SubName, int]:
    """Count how often each sub-state was the binding one (REGIME §9 binding ablation)."""
    counts: Counter[SubName] = Counter()
    for s in states:
        counts[s.binding_sub] += 1
    return dict(counts)


@dataclass(frozen=True)
class ARLAuditRow:
    """One state's budgeted vs realized false-escalation rate on a calm stretch (REGIME §3.6d)."""

    measure: str
    state: str
    budget_alpha_eff: float
    budget_arl0: float
    realized_rate: float
    realized_count: int
    n_bars: int


def arl_audit(
    measure: str,
    realized_escalations: Mapping[str, int],
    n_bars: int,
    cfg: PolicyDefaults,
) -> tuple[ARLAuditRow, ...]:
    """Realized per-bar escalation rate per state vs the ``alpha_k``/``ARL0`` budget (REGIME §3.6d).

    ``measure`` is a PERCENTILE measure ("vol"/"turb"/"liq" — liq uses the ``1 − p²`` null);
    ``realized_escalations`` maps a state name to the realized escalation count on the calm stretch.
    A realized rate materially ABOVE budget => threshold too tight (over-firing); materially BELOW
    => too loose (late in a real crisis).
    """
    rows: list[ARLAuditRow] = []
    for budget in arl0_budget(measure, cfg):
        count = int(realized_escalations.get(budget.state, 0))
        realized = (count / n_bars) if n_bars > 0 else 0.0
        rows.append(
            ARLAuditRow(
                measure=measure, state=budget.state,
                budget_alpha_eff=budget.alpha_eff, budget_arl0=budget.arl0,
                realized_rate=realized, realized_count=count, n_bars=n_bars,
            )
        )
    return tuple(rows)


def detection_delay(g_path: Sequence[float], shift_bar: int) -> int:
    """Realized ARL1: bars from a true shift at ``shift_bar`` until ``g_t`` first cuts (REGIME §9).

    Escalation is immediate (§3.2), so a true shift clearing the entry mark should give ``ARL1 ≈ 1``.
    Returns the number of bars until the first ``g_t`` reduction at/after ``shift_bar`` (or -1).
    """
    g = list(g_path)
    for i in range(shift_bar, len(g)):
        if g[i] < g[shift_bar - 1] if shift_bar > 0 else g[i] < 1.0:
            return i - shift_bar + 1
    return -1
