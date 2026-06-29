"""BLUE composition grids + the joint-search coordinate (SYSTEM §5).

SYSTEM's BLUE grid is the small set of *composition* choices that are legitimately uncertain:
cadence ``τ_sel``, deadband ``δ``, rotation hysteresis ``Δ_rank``, budget-priority rule variants,
precedence variants. Each is finite and enumerated; the joint product is recorded as
``n_sys_nominal`` telemetry (the size of the space we *could* have searched), and every config
actually scored against a fold is appended to the persistent ledger (Invariant SG), from which the
EFFECTIVE ``N_sys`` is derived (Section 7.1) — the product is NEVER fed to ``SR*₀``.

A config that changes ONLY a BLUE composition field re-scores from cache for free (Invariant CA):
its module evidence keys are unchanged. That is the whole point of P7 / the overnight budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping

# The enumerated BLUE composition grids (class-level, never per-symbol on P&L; Section 5).
TAU_SEL_GRID: tuple[str, ...] = ("daily", "session")
DEADBAND_GRID: tuple[float, ...] = (0.01, 0.02, 0.04)
DELTA_RANK_GRID: tuple[float, ...] = (1.0, 2.0, 3.0)
BUDGET_PRIORITY_GRID: tuple[str, ...] = ("slack_then_mctr", "mctr_then_slack")


def nominal_grid_cardinality() -> int:
    """The Cartesian product of the BLUE composition grids — ``n_sys_nominal`` telemetry ONLY.

    This is the size of the joint space we *could* have searched; it is NEVER fed to ``SR*₀``
    (Section 7.1 consistency law). Only the configs actually scored (the ledger) feed the deflation.
    """
    return (
        len(TAU_SEL_GRID)
        * len(DEADBAND_GRID)
        * len(DELTA_RANK_GRID)
        * len(BUDGET_PRIORITY_GRID)
    )


@dataclass(frozen=True)
class CompositionPoint:
    """One BLUE composition grid coordinate — a combiner-only variant (changes no module evidence).

    Two CompositionPoints with the same module versions share all module-evidence cache keys, so a
    change here re-scores from cache for free (Invariant CA / P7).
    """

    tau_sel: str = "daily"
    deadband_weight: float = 0.02
    delta_rank: float = 2.0
    budget_priority: str = "slack_then_mctr"

    def as_blue_point(self) -> Mapping[str, object]:
        """The joint BLUE grid coordinate recorded on each :class:`TrialRecord` (Invariant SG)."""
        return dict(asdict(self))

    def config_hash(self, module_versions: Mapping[str, str]) -> str:
        """Content-addressed hash over the combiner coordinate + module versions (EvidenceCache key).

        A combiner-only change perturbs the COMPOSITION part of the hash but NOT the module-version
        part, so module-evidence keyed on ``(module, module_version, fold_id, …)`` still hits cache.
        """
        payload = {"composition": dict(asdict(self)), "module_versions": dict(sorted(module_versions.items()))}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class CompositionGrid:
    """The finite enumerated BLUE composition search space (Section 5)."""

    tau_sel: tuple[str, ...] = TAU_SEL_GRID
    deadband: tuple[float, ...] = DEADBAND_GRID
    delta_rank: tuple[float, ...] = DELTA_RANK_GRID
    budget_priority: tuple[str, ...] = BUDGET_PRIORITY_GRID
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def enumerate(self) -> tuple[CompositionPoint, ...]:
        """Every grid coordinate (the joint search the deflation must account for)."""
        points: list[CompositionPoint] = []
        for ts in self.tau_sel:
            for db in self.deadband:
                for dr in self.delta_rank:
                    for bp in self.budget_priority:
                        points.append(
                            CompositionPoint(tau_sel=ts, deadband_weight=db, delta_rank=dr, budget_priority=bp)
                        )
        return tuple(points)

    @property
    def cardinality(self) -> int:
        return len(self.tau_sel) * len(self.deadband) * len(self.delta_rank) * len(self.budget_priority)
