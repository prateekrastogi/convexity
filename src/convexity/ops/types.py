"""OPS shared data objects (``docs/specs/OPS.md`` sec 9).

``OpsDirective`` and ``Alarm`` are defined canonically in :mod:`convexity.core.types` (the migration
OPS.md sec 9 mandates: ``core/types.py``'s pre-pivot ``OpsDirective`` was replaced in-place with
this spec's shape, and ``Alarm`` was added there so ``OpsDirective.alarms`` is constructible outside
this package). They are re-exported here so ``ops.types`` and ``core.types`` name the SAME object,
not two structurally-similar classes (OPS.md sec 12 cross-consistency gate). This module adds only
the OPS-internal recursion-state object (``DetectorState``) and the alarm/state literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from convexity.core.types import Alarm, AlarmKind, OpsDirective, OpsState

if TYPE_CHECKING:
    from convexity.ops.streams.recon import PositionSnapshot

__all__ = [
    "Alarm",
    "AlarmKind",
    "OpsDirective",
    "OpsState",
    "DetectorState",
    "RebalanceFill",
    "OpsObs",
]


@dataclass(frozen=True)
class DetectorState:
    """One detector's internal recursion state, checkpointed for restart-safety (OPS.md sec 9).

    ``statistic`` is the accumulated CUSUM ``C_k`` (or the analogous value for another detector);
    ``n_obs`` counts observations folded in; ``last_update`` is epoch-ms of the last fold.
    """

    detector: str  # "cusum_sharpe", "recon", "sla", ...
    statistic: float  # C_k or the analogous accumulated value
    n_obs: int
    last_update: int


@dataclass(frozen=True)
class RebalanceFill:
    """One rebalance's realized completion-time + slippage facts (OPS.md sec 5, sec 2 streams table).

    ``completion_lag`` is the realized wall-clock time the rebalance's last fill settled after start;
    ``realized_slippage`` is EXECUTION's realized ``IS_total`` (COST's realized-fill TCA interim proxy,
    sec 5.2); ``expected_slippage`` the modeled friction the strategy was sized against (sec 5.2).
    """

    completion_lag: float
    realized_slippage: float
    expected_slippage: float


@dataclass(frozen=True)
class OpsObs:
    """One causal observation the ``OpsEngine.on_obs`` seam consumes (OPS.md sec 2, sec 3, sec 12).

    A single obs may carry any subset of the sec-2 monitored streams — a settled ``weekly_net_return``
    (sec 4 CUSUM), a daily ``position_snapshots`` reconciliation read (sec 1), a realized
    ``rebalance_fill`` (sec 5 SLA), and/or a ``heartbeat_silent`` flag (sec 2). ``as_of`` is epoch-ms
    of the observation (P1: the risk unit is the whole book, never a per-instrument P&L). Every field
    is optional so the engine can be driven on a finer clock than it trades on (P2): a daily recon obs
    carries no weekly return, an intraday SLA poll carries only a fill, etc.
    """

    as_of: int
    weekly_net_return: float | None = None
    position_snapshots: "list[PositionSnapshot] | None" = None
    rebalance_fill: RebalanceFill | None = None
    heartbeat_silent: bool = False
