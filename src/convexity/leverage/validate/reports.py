"""LEVERAGE validation-report helpers (LEVERAGE.md §9 layout — the NOTE-only module).

Per §9, LEVERAGE does NOT implement its own block-bootstrap CI / stats module: ``SR_LCB_t``
(``edge/lcb.py``) calls ``convexity.system.validate.robustness.block_bootstrap_ci`` DIRECTLY, and a
duplicate ``stats.py``/bootstrap module here would violate Invariant K-CI (no re-derivation of
SYSTEM's estimator). This module therefore contains NO statistical machinery — only presentation:
it renders an already-computed sizing decision and its acceptance-gate check into a flat, auditable
report record. Any CI/DSR/ARL number it displays is produced upstream (edge/lcb.py, validate/arl.py,
validate/backtest.py) and merely passed through here (Invariant K-CI).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.leverage.types import LeverageDecision


@dataclass(frozen=True)
class SizingReport:
    """A flat, auditable render of a :class:`LeverageDecision` (LEVERAGE.md §9, presentation only).

    Every field is COPIED from an upstream-computed decision — this record derives no new statistic
    (Invariant K-CI). It exists so an operator/audit log can persist the binding ceiling and its
    four terms in one row without re-touching the estimators.
    """

    g_t: float
    binding_ceiling: str
    kelly_term: float
    capacity_term: float
    governor_term: float
    hard_cap_term: float
    reason_codes: tuple[str, ...]


def render_decision(decision: LeverageDecision) -> SizingReport:
    """Render a decision into a flat report record (LEVERAGE.md §9 — pass-through, no re-derivation).

    Pure projection of the already-computed :class:`LeverageDecision`; introduces no CI/DSR/ARL math
    (that lives in SYSTEM's robustness + LEVERAGE's edge/validate layers, Invariant K-CI). The
    ``capacity_term`` may be ``+inf`` for an empty book and is passed through unchanged so the report
    faithfully shows "no capacity constraint" rather than fabricating a number.
    """
    return SizingReport(
        g_t=decision.g_t,
        binding_ceiling=decision.binding_ceiling,
        kelly_term=decision.kelly_term,
        capacity_term=decision.capacity_term,
        governor_term=decision.governor_term,
        hard_cap_term=decision.hard_cap_term,
        reason_codes=decision.reason_codes,
    )
