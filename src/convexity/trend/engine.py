"""TrendLifeEngine — batch replay + one-bar online update with parity (TREND §11, Inv I/HC).

``run_batch`` replays history with causal state only; ``update`` advances exactly one bar and must
match ``run_batch`` after warm-up (Inv I parity). The engine owns the streaming estimator state (via
``BeliefBuilder``), the convex-exit state, and the decision policy. It FORCE-EMITS
``stop_cause="time_cap"`` at the first bar the active clock crosses ``cfg.tau_cap == H_sys`` (Inv HC),
so no realized hold exceeds SYSTEM's contractual bound — DecisionPolicy already implements rung 5,
and the engine stops iterating once the position has exited so no decision carries a hold beyond the
cap. The engine never places orders (Inv N); it returns ``ActionDecision``s carrying the intent.
"""

from __future__ import annotations

from dataclasses import replace

from convexity.core.types import Bars, InstrumentSpec, Position
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.estimate.state import BeliefBuilder
from convexity.trend.policy.convex_exit import ConvexExit
from convexity.trend.policy.decision import DecisionPolicy
from convexity.trend.types import ActionDecision, ExitState


class TrendLifeEngine:
    """The composed live/batch engine (§11)."""

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.builder = BeliefBuilder(cfg)
        self.convex_exit = ConvexExit()
        self.decision = DecisionPolicy(convex_exit=self.convex_exit)
        self._exit_state: ExitState | None = None
        self._closed = False
        self._last_idx = -1

    # ---- batch ----------------------------------------------------------------------------
    def run_batch(
        self, bars: Bars, position: Position, spec: InstrumentSpec, cfg: ResolvedConfig
    ) -> list[ActionDecision]:
        """Run historical replay with causal state only (§11). Stops once the position exits."""
        self._reset(bars, position, cfg)
        decisions: list[ActionDecision] = []
        for idx in range(position.entry_idx, len(bars)):
            decision = self._step(bars, idx, position, cfg)
            decisions.append(decision)
            if decision.action == "exit":
                break  # no decision carries a realized hold beyond the stopping time (Inv HC)
        return decisions

    # ---- online ---------------------------------------------------------------------------
    def update(
        self, bars: Bars, position: Position, spec: InstrumentSpec, cfg: ResolvedConfig, idx: int
    ) -> ActionDecision | None:
        """Advance exactly one bar at ``idx`` (§11). Returns the decision, or None once closed.

        Matches ``run_batch`` after warm-up (Inv I): the engine threads the same streaming state. The
        caller drives ``idx`` from ``entry_idx`` upward; once an exit fires the engine is closed.
        """
        if idx == position.entry_idx:
            self._reset(bars, position, cfg)
        if self._closed:
            return None
        decision = self._step(bars, idx, position, cfg)
        if decision.action == "exit":
            self._closed = True
        return decision

    # ---- internals ------------------------------------------------------------------------
    def _reset(self, bars: Bars, position: Position, cfg: ResolvedConfig) -> None:
        self.builder.reset()
        e = position.entry_idx
        vol0 = self.builder.vol.update(bars, e, position, cfg, prev_integrated_var=0.0)
        self._exit_state = self.convex_exit.initial(bars, position, vol0, cfg)
        self._closed = False
        self._last_idx = e - 1

    def _step(
        self, bars: Bars, idx: int, position: Position, cfg: ResolvedConfig
    ) -> ActionDecision:
        assert self._exit_state is not None
        belief = self.builder.update(bars, idx, position, cfg)
        # trail update (the initial state stands on the entry bar).
        if idx > position.entry_idx:
            self._exit_state = self.convex_exit.update(
                bars, idx, position, belief, self._exit_state, cfg
            )
        decision = self.decision.decide(bars, idx, position, belief, self._exit_state, cfg)
        # carry the reduced size fraction forward (reduce scales the held fraction; Inv: never death).
        if decision.action == "reduce":
            self._exit_state = replace(self._exit_state, size_fraction=decision.size_fraction)
        self._last_idx = idx
        return decision
