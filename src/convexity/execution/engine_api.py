"""engine_api — the stable public surface the strategy adapter calls (§3 / §11).

``build_engine`` wires the microstate / cost / policy stack into one :class:`ExecutionEngine`;
``on_target`` is the single entry point that turns a fully-decided :class:`ExecTarget` + a
:class:`MicroState` + the Nautilus-read :class:`WorkingState` into one :class:`ExecDecision` (the
``ExecutionEngine`` Protocol surface in :mod:`convexity.core.protocols`). Pure-Python,
NautilusTrader-free — the ``nautilus/adapter.py`` bridge is the only engine-facing code (Invariant N).
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.policy.engine import ExecutionEngine
from convexity.execution.types import ExecDecision, ExecTarget, MicroState, WorkingState

__all__ = ["build_engine", "on_target", "ExecutionEngine"]


def build_engine() -> ExecutionEngine:
    """Construct the default :class:`ExecutionEngine` with the standard microstate/cost/policy stack."""
    return ExecutionEngine()


def on_target(
    engine: ExecutionEngine,
    target: ExecTarget,
    micro: MicroState,
    working: WorkingState,
    cfg: ResolvedConfig,
    *,
    as_of_ts: int = 0,
) -> ExecDecision:
    """The public ``on_target`` (§11): one :class:`ExecDecision` per call. Deterministic (Invariant E)."""
    return engine.on_target(target, micro, working, cfg, as_of_ts=as_of_ts)
