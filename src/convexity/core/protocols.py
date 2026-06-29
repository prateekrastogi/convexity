"""Pillar composition interfaces — the typing.Protocol seams the strategy adapter composes against.

The architecture principle: each pillar's POLICY logic is pure-Python and NautilusTrader-free, and
the thin ``strategy/convexity_strategy.py`` Nautilus adapter COMPOSES the pillars. These Protocols
make the composition graph statically explicit: the adapter (and SYSTEM's orchestrator) depend on
these shapes, not on concrete pillar classes, so a pillar can be built and unit-tested against a
mock that satisfies its Protocol before any sibling exists.

Each Protocol is the SINGLE outward method the spec prescribes as the pillar's composition surface:
  * BREAKOUT -> ``BreakoutModule.propose(instrument) -> EntrySignal | None``      (BREAKOUT §detect/combine)
  * TREND    -> ``TrendEngine.update(position, ...) -> ActionDecision``           (TREND §11)
  * LEVERAGE -> ``size_position(signal, risk, cfg, book) -> LeverageDecision``    (LEVERAGE engine_api)
  * EXECUTION-> ``ExecutionEngine.on_target(target, ...) -> ExecDecision``        (EXECUTION §11)
  * COST     -> the four consumers off one CostBook (net-EV gate / hold-carry / carry-term / TCA)  (COST §8)
  * REGIME   -> ``RegimeEngine.regime_state(...) -> RegimeState``                 (REGIME §8)
  * OPS      -> ``OpsEngine.on_obs(obs) -> OpsDirective``                         (OPS §3)
  * SYSTEM   -> ``SystemOrchestrator.decide(...) -> SystemDecision``              (SYSTEM §12)

The flow these wire up (the COMPOSITION GRAPH, see docs/ARCHITECTURE.md):

    SELECTION ──risk_weights/rank──▶ SYSTEM ◀── REGIME g_t ──▶ LEVERAGE (g_t·H_book)
        BREAKOUT.propose ─EntrySignal▶ SYSTEM ─SignalInput▶ LEVERAGE ─LeverageDecision▶ SYSTEM
        SYSTEM ─Position▶ TREND ─ActionDecision/ExecutionIntent▶ EXECUTION ─OrderCommand▶ Nautilus
        COST ◀──(net-EV gate, carry term, hold-carry, TCA)── BREAKOUT/LEVERAGE/TREND
        OPS  ◀──(monitors all streams)── EXECUTION/COST/LEVERAGE/SYSTEM ──OpsDirective▶ live loop

Concrete return types beyond the shared core (``ActionDecision``, ``ExecDecision``, ``Ceilings``,
``NetEVDecision``, ``SystemDecision``) live in each pillar's own ``types.py``; here they are forward
``Protocol``-typed via the shared boundary objects so this module stays import-light and pillar-
agnostic. We annotate them as ``object`` where the full shape is pillar-private, with the spec
reference in the docstring — the adapter only needs the boundary fields.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from convexity.core.types import (
    Bars,
    EntrySignal,
    ExecTarget,
    HoldHorizon,
    InstrumentSpec,
    LeverageDecision,
    OpsDirective,
    Position,
    RegimeState,
    RoundTripCost,
    Side,
    SignalInput,
)


@runtime_checkable
class BreakoutModule(Protocol):
    """BREAKOUT's outward composition surface (BREAKOUT detect/combine.py).

    Returns at most one currently-actionable highest-quality un-deduped 'entry' at the latest bar,
    else None. The ONLY method SYSTEM imports; exposes only direction/status=='entry'/urgency/price
    levels downstream (Inv K).
    """

    def propose(self, instrument: InstrumentSpec) -> EntrySignal | None: ...


@runtime_checkable
class LeverageEngine(Protocol):
    """LEVERAGE's public sizing surface (LEVERAGE engine_api.size_position).

    Pure-Python, Nautilus-free. ``book`` is the SYSTEM-owned BookState (incumbents + reserved heat +
    candidate weights + existing margin) — its concrete shape is owned by SYSTEM, so it is typed
    ``object`` here (LEVERAGE risk note: BookState undefined in LEVERAGE).
    """

    def size_position(
        self, signal: SignalInput, risk: object, cfg: object, book: object
    ) -> LeverageDecision: ...


@runtime_checkable
class TrendEngine(Protocol):
    """TREND's live surface (TREND §11). Returns an ``ActionDecision`` (TREND types.py) carrying the
    action (continue/reduce/exit), the ExecutionIntent for EXECUTION, and the HoldHorizonEstimate /
    ResidualMoveEstimate consumed read-only by LEVERAGE/COST.

    ``cfg`` exposes ``tau_cap == H_sys`` injected by SYSTEM (Inv HC); ``ActionDecision`` is pillar-
    private so typed ``object``.
    """

    def update(self, position: Position, bars: Bars, idx: int, cfg: object) -> object: ...


@runtime_checkable
class ExecutionEngine(Protocol):
    """EXECUTION's surface (EXECUTION §11). One ``ExecDecision`` per call (tactic/price/qty/
    client_order_id/cost decomposition/binding_gate). Reads working/position state from Nautilus
    (Inv N) via the passed-in ``working`` snapshot; never tracks position itself."""

    def on_target(
        self, target: ExecTarget, micro: object, working: object, cfg: object
    ) -> object: ...


@runtime_checkable
class CostEngine(Protocol):
    """COST's surface (COST cost_api). One ``CostBook`` feeds all four consumers with the SAME
    estimators + resolved config (Inv T 'one cost truth').

    The boundary methods the other pillars consume:
      * ``net_ev_gate`` -> BREAKOUT entry gate (passes/expected_favorable/round_trip/binding_funding)
      * ``carry_term``  -> LEVERAGE stop-ceiling SIGNED carry (LEVERAGE clamps max(carry,0))
      * ``hold_carry``  -> TREND hold/reduce input (signed residual carry)
      * ``round_trip``  -> REGIME's cost gate (round_trip_estimate(delta_gross))
    """

    def round_trip(
        self,
        side: Side,
        notional: float,
        liquidity_flag_in: str,
        liquidity_flag_out: str,
        horizon: HoldHorizon,
        market: object,
        cfg: object,
        expected_favorable: float | None = None,
    ) -> RoundTripCost: ...

    def carry_term(
        self, side: Side, notional: float, horizon: HoldHorizon, market: object, cfg: object
    ) -> tuple[float, Mapping[str, float]]: ...


@runtime_checkable
class RegimeEngine(Protocol):
    """REGIME's surface (REGIME engine_api). Pure function of (panel, exec_state, cost_state, book,
    prev, cfg). Emits ``g_t``/``s_t`` only (Inv SCOPE); enters LEVERAGE through ``g_t·H_book`` alone
    (Inv NODOUBLE2). ``panel``/``book`` are pillar/neighbour-owned, typed ``object``."""

    def regime_state(
        self,
        panel: object,
        exec_state: object,
        cost_state: object,
        book: object,
        prev: RegimeState | None,
        cfg: object,
    ) -> RegimeState: ...


@runtime_checkable
class OpsEngine(Protocol):
    """OPS's surface (OPS §3). Consumes one causal ``StreamObs`` per call, emits the
    ``OpsDirective`` ({run,degrade,halt} + flatten/cancel flags + reason chain). Monitors only,
    never trades (Inv PROV); the only size/exposure output is the binary state by precedence."""

    def on_obs(self, obs: object) -> OpsDirective: ...

    def directive(self) -> OpsDirective: ...


@runtime_checkable
class SystemOrchestrator(Protocol):
    """SYSTEM's one public composition method (SYSTEM §12). Composes the four pillar adapters via
    the §2 precedence pipeline + §3 fixed point into ONE target book delta (``SystemDecision``).
    Never reaches inside modules (P1); deterministic; emits deltas, not raw orders.

    ``book``/``sel``/``open_actions``/``candidates`` are SYSTEM-owned; the ``SystemDecision`` return
    is SYSTEM-private so typed ``object``.
    """

    def decide(
        self,
        book: object,
        sel: object,
        open_actions: Mapping[str, object],
        candidates: Mapping[str, object],
        cfg: object,
    ) -> object: ...
