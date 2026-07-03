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


# ------------------------------------------------------------------ xsec search grid (SYSTEM §5.1)

# The Section 5.1 enumerable cross-sectional search grid (BLUE: finite, enumerated, ledger-tracked,
# never fitted continuously or per-symbol on raw P&L; SYSTEM §10). The Cartesian product of these
# axes is telemetry ONLY (``n_sys_nominal``) — it is NEVER fed to SR*₀ (SYSTEM §5.6, Invariant DSR);
# only configs actually scored against a validation fold enter the ledger (Invariant SG).
XSEC_LOOKBACK_BAND: tuple[int, ...] = (12, 14, 16)  # sparse band anchored on TESTED points (§5.1)
XSEC_SIGNAL_GRID: tuple[str, ...] = (
    "momentum_L12", "momentum_L14", "momentum_L16", "carry", "combined",
)
XSEC_QUANTILE_GRID: tuple[str, ...] = ("quintile", "decile", "tercile")
XSEC_WEIGHTING_GRID: tuple[str, ...] = ("inverse_realized_vol", "liquidity_capped", "equal_weight")
XSEC_UNIVERSE_GRID: tuple[int, ...] = (50, 60, 75, 100)


def xsec_nominal_grid_cardinality() -> int:
    """The §5.1 Cartesian product (3 lookbacks folded into the signal axis) — telemetry ONLY.

    Recorded as ``n_sys_nominal``; never fed to ``SR*₀`` (SYSTEM §5.6 consistency law / Invariant
    DSR). Most cells were never composed and scored; only the persistent ledger feeds deflation.
    """
    return (
        len(XSEC_SIGNAL_GRID)
        * len(XSEC_QUANTILE_GRID)
        * len(XSEC_WEIGHTING_GRID)
        * len(XSEC_UNIVERSE_GRID)
    )


# ------------------------------------------------------- promotion-gate RED constants (SYSTEM §6/§10)

# The committed CPCV partition (SYSTEM §4.2) and the PRE-REGISTERED, ledger-independent alternative
# partitions for the PBO sensitivity range (SYSTEM §4.4 / §12 edge case 9). Fixed BEFORE any
# sensitivity check is run — a promotion report whose alternatives differ from this list is rejected
# exactly as one missing the range entirely (Invariant PBO-RANGE).
COMMITTED_CPCV_PARTITION: tuple[int, int] = (8, 2)
PREREGISTERED_PBO_PARTITIONS: tuple[tuple[int, int], ...] = ((6, 2), (10, 3))


@dataclass(frozen=True)
class PromotionGateDefaults:
    """RED constants of the SYSTEM §6.1 PROMOTABLE conjunction (§10 parameter governance).

    Every threshold here is RED and shared — unreachable from any per-instrument or per-signal-cell
    config (Invariant C). ``k_min=3`` is the ``src/convexity/system/xsec_validate.py::weekly_defaults`` floor
    (the value this book actually runs with) — NOT ``SystemDefaults.k_min=8`` (the unrelated
    1-min/directional-era default) and NOT the retracted, never-grounded ``n_true_distinct >= 30``
    of an earlier spec draft (SYSTEM §6 constant-reconciliation note). The Gate 6 constants
    (``stress_multiplier=1.5``, ``stress_retention=0.5``, ``stressed_frac_pos_floor=0.75``) are the
    ONE RED constant set shared with COST.md §8.1–8.2, restated, not independently re-derived
    (SYSTEM §6.1 Gate 6) — the spec itself flags the 0.5×/0.75 pair as judgment-call constants.
    """

    # GATE 1 — trial floor (SYSTEM §6.1 Gate 1; weekly_defaults' k_min, a deliberately LOW floor).
    k_min: int = 3
    # GATE 2 — edge (SYSTEM §6.1 Gate 2; branch 2a clean DSR pass, branch 2b track-length).
    dsr_threshold: float = 0.95
    psr_zero_threshold: float = 0.95
    # GATE 3 — CPCV robustness (SYSTEM §6.1 Gate 3; frac_pos, NEVER p05 — Invariant CPCV-STAT).
    frac_pos_threshold: float = 0.8
    # GATE 4 — overfitting (SYSTEM §6.1 Gate 4 / §4.4 three-zone convention, Invariant PBO-RANGE).
    pbo_threshold: float = 0.5   # upper bound of the reported range must be strictly below this
    pbo_band_floor: float = 0.2  # PBO in [band_floor, threshold) forces ensemble/band deployment
    # GATE 5 — regime breadth (SYSTEM §6.1 Gate 5: positive Sharpe in >= 4 of 5 tested regimes).
    regime_pos_required: int = 4
    regime_total: int = 5
    # GATE 6 — cost stress (SYSTEM §6.1 Gate 6 == COST.md §8.1–8.2, the shared RED constants).
    stress_multiplier: float = 1.5
    stress_retention: float = 0.5
    stressed_frac_pos_floor: float = 0.75
    # §12 edge cases 8+9 — ledger/PBO snapshot cross-consistency (in scoring cycles).
    max_snapshot_skew: int = 1
    committed_partition: tuple[int, int] = COMMITTED_CPCV_PARTITION
    preregistered_alternatives: tuple[tuple[int, int], ...] = PREREGISTERED_PBO_PARTITIONS
