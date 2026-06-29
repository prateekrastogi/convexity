"""The single Nautilus ``Strategy`` that COMPOSES all eight pillars (ARCHITECTURE §4).

Written ONCE, run unchanged in parity-backtest, paper, and live — the engine owns backtest↔live
parity, so the composition is a pure function of (BookState, per-bar pillar outputs) and the same
class drives every venue (ARCHITECTURE §1/§4).

Two layers, split so the composition is unit/integration-testable without a live node:

  * :class:`ConvexityComposer` — PURE-PYTHON, NautilusTrader-FREE. On each decision bar it builds the
    pillar inputs from the rolling :class:`~convexity.core.types.Bars`, runs TREND.update on every
    open position (exit/trail :class:`ExecutionIntent`), runs BREAKOUT.propose per eligible-not-open
    instrument, sizes each via the LEVERAGE engine (RISK metadata only), runs SYSTEM's orchestrator
    to a :class:`SystemDecision`, lowers each open/close/reduce delta through the EXECUTION engine
    into :class:`OrderCommand`s, rests a venue reduce-only protective stop per fresh open
    (EXECUTION P6), and folds the OPS directive (halt ⇒ flatten) over the top. It also runs the
    :class:`StopReconciler` on start so a restart rebuilds the protective stops idempotently. This
    object is what the backtest↔live parity integration test replays against a scripted pillar
    sequence (SYSTEM Inv PAR).

  * :class:`ConvexityStrategy` — the thin ``nautilus_trader`` ``Strategy`` SEAM. It is the ONLY code
    here that imports nautilus (lazily, via :func:`make_strategy_class` on the VM). Phase 8 ships it
    as the typed shell; the event wiring that will subscribe to bars / order / position / fill
    events, project the engine's ``Portfolio``/``Cache`` into the pure pillar types, call the
    composer, and submit the resulting :class:`OrderCommand`s lands in the DEPLOYMENT phase with
    ``run/{backtest,paper,live}.py``. It must NEVER track position/account state itself — it reads
    Nautilus's authoritative state (Invariant N). See :func:`make_strategy_class` for the scope note.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from convexity.core.types import (
    Bars,
    EntrySignal,
    ExecTarget,
    InstrumentSpec,
    OpsDirective,
)
from convexity.execution.config.resolve import ResolvedConfig as ExecResolvedConfig
from convexity.execution.nautilus.reconcile import (
    IntendedState,
    ReconcilePlan,
    StopReconciler,
)
from convexity.execution.policy.order_issuer import slice_client_order_id
from convexity.execution.policy.protective_stop import ProtectiveStop
from convexity.execution.types import (
    MicroState,
    OrderCommand,
    PositionState,
    VenueStop,
    WorkingState,
)
from convexity.ops.nautilus.adapter import OpsAdapter
from convexity.ops.nautilus.flatten import FlattenCommand
from convexity.ops.types import EquityRef
from convexity.system.compose.orchestrator import CandidateSignal, SystemOrchestrator
from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import BookState, ProposedPortfolio, SystemDecision

__all__ = [
    "ConvexityComposer",
    "ComposerResult",
    "OpenPositionView",
    "ConvexityStrategy",
    "StrategyDeps",
    "make_strategy_class",
    "intent_id_for",
]

STRATEGY_VERSION = "convexity-strategy-1.0.0"


# --------------------------------------------------------------------------- pure-Python views


@dataclass(frozen=True)
class OpenPositionView:
    """One open position as the composer sees it, projected from Nautilus state (Invariant N).

    ``trend_position`` is the BREAKOUT-owned :class:`~convexity.core.types.Position` TREND consumes
    read-only (Inv O); ``invalidation`` + ``invalidation_generation`` version the venue stop's id
    (derived from observed state, never an in-process counter — Inv Q-STOP); ``venue_stops`` are the
    reduce-only stops currently resting for this intent (read back for reconciliation).
    """

    instrument: str
    intent_id: str
    state: PositionState
    trend_position: Any
    invalidation: float
    invalidation_generation: int
    held_risk_weight: float
    entry_idx: int
    venue_stops: tuple[VenueStop, ...] = ()


@dataclass(frozen=True)
class ComposerResult:
    """Everything one decision bar produces (the deltas Nautilus executes + the audit trail).

    ``commands`` is the ordered venue-order plan (the EXECUTION-tactic'd opens, the venue reduce-only
    stop places/amends, exit/reduce orders, OPS flatten cancels). ``decision`` is SYSTEM's full
    reason-chained book delta. ``directive`` is the OPS {run,degrade,halt} state. ``intents`` maps
    instrument → the TREND :class:`ExecutionIntent` (audit). Deterministic given inputs.
    """

    decision: SystemDecision
    commands: tuple[OrderCommand, ...]
    directive: OpsDirective
    intents: Mapping[str, object]
    flatten: FlattenCommand | None
    reason_codes: tuple[str, ...]


# --------------------------------------------------------------------------- the pure composer


class ConvexityComposer:
    """Compose the eight pillars into one venue-order plan per decision bar (ARCHITECTURE §4).

    Pure-Python and NautilusTrader-free: every input is one of the shared ``core.types`` boundary
    objects or a pillar's public type, so the whole composition replays identically inside a Nautilus
    backtest and against a scripted pillar sequence (the parity invariant, SYSTEM Inv PAR). The
    pillars are held behind their Protocol surfaces (``core.protocols``) so a mock satisfies the seam.
    """

    def __init__(
        self,
        *,
        breakout: Any,
        leverage: Any,
        trend: Any,
        execution: Any,
        orchestrator: SystemOrchestrator | None = None,
        ops_adapter: OpsAdapter | None = None,
        protective_stop: ProtectiveStop | None = None,
        reconciler: StopReconciler | None = None,
    ) -> None:
        self.breakout = breakout
        self.leverage = leverage
        self.trend = trend
        self.execution = execution
        self.orchestrator = orchestrator or SystemOrchestrator()
        self.ops = ops_adapter
        self.protective_stop = protective_stop or ProtectiveStop()
        self.reconciler = reconciler or StopReconciler()

    # ---------------------------------------------------------------- restart reconciliation

    def reconcile_on_start(
        self,
        open_positions: Mapping[str, OpenPositionView],
        exec_cfg: ExecResolvedConfig,
    ) -> tuple[OrderCommand, ...]:
        """Rebuild the protective stops on restart (EXECUTION §9 / Inv S). Idempotent BY CONSTRUCTION.

        For each open position, project observed (position, resting venue stops) onto ``STOP_OK`` via
        the :class:`StopReconciler`. A position with no resting stop gets one placed NOW (the gap-
        protection point); a stale/duplicate set is collapsed to exactly one current-generation stop.
        Running it a second time on an already-``STOP_OK`` book emits an EMPTY plan (the idempotency
        proof). Returns the flat command list to submit.
        """
        commands: list[OrderCommand] = []
        for inst in sorted(open_positions):
            view = open_positions[inst]
            intended = IntendedState(
                intent_id=view.intent_id,
                current_invalidation=view.invalidation,
                current_generation=view.invalidation_generation,
            )
            plan: ReconcilePlan = self.reconciler.reconcile(
                intended, view.venue_stops, view.state, exec_cfg
            )
            commands.extend(plan.commands)
        return tuple(commands)

    # ---------------------------------------------------------------- the per-bar composition

    def on_decision_bar(
        self,
        *,
        bars_by_instrument: Mapping[str, Bars],
        idx_by_instrument: Mapping[str, int],
        specs: Mapping[str, InstrumentSpec],
        book: BookState,
        open_positions: Mapping[str, OpenPositionView],
        selection: ProposedPortfolio,
        leverage_risk: Mapping[str, Any],
        leverage_cfg: Any,
        trend_cfg: Any,
        exec_cfg: ExecResolvedConfig,
        micro_by_instrument: Mapping[str, MicroState],
        sys_cfg: Any,
        equity_ref: EquityRef | None = None,
        recon_mismatch: bool = False,
        heartbeat_code: float = 0.0,
        bar_ts: int = 0,
        risk_improvements: Mapping[str, float] | None = None,
        round_trip_costs: Mapping[str, float] | None = None,
    ) -> ComposerResult:
        """Run the full ARCHITECTURE §4 composition for one decision bar.

        Order of operations (the §4 graph):
          1. TREND.update on every OPEN position → per-instrument :class:`ExecutionIntent` +
             ``trend_state`` for the orchestrator (open positions governed by TREND, SYSTEM §2 B).
          2. BREAKOUT.propose on every selection-eligible, NOT-open instrument → candidate
             :class:`EntrySignal`; size each through the LEVERAGE engine to attach RISK metadata
             (ceiling slack, lot weight) — never the breakout score (SYSTEM Inv CB).
          3. SYSTEM.orchestrator.decide → one :class:`SystemDecision` (the §2 precedence pipeline +
             §3 fixed point). SYSTEM emits the integer book delta; it places no orders (§2 G).
          4. Lower each delta through EXECUTION.on_target → :class:`OrderCommand`s; rest a venue
             reduce-only protective stop per fresh open (EXECUTION P6) and amend it per TREND trail
             intent on incumbents.
          5. Fold the OPS directive on top: a HALT directive overrides everything with a
             reduce-only flatten (OPS §9.1b).
        """
        reason_codes: list[str] = []

        # ---- 1. OPEN POSITIONS → TREND.update (exit/reduce/continue + ExecutionIntent). ----
        open_actions: dict[str, TrendAction] = {}
        intents: dict[str, object] = {}
        for inst in sorted(open_positions):
            view = open_positions[inst]
            bars = bars_by_instrument[inst]
            idx = idx_by_instrument[inst]
            decision = self.trend.update(bars, view.trend_position, specs[inst], trend_cfg, idx)
            if decision is None:
                open_actions[inst] = _trend_action_from(None)
                continue
            intents[inst] = getattr(decision, "execution_intent", None)
            open_actions[inst] = _trend_action_from(decision)

        # ---- 2. CANDIDATES → BREAKOUT.propose + LEVERAGE size (RISK metadata only). ----
        candidates: dict[str, CandidateSignal] = {}
        open_set = set(open_positions)
        for inst in selection.eligible:
            if inst in open_set:
                continue
            signal: EntrySignal | None = self.breakout.propose(specs[inst])
            if signal is None or signal.status != "entry" or signal.direction == 0:
                continue
            cand = self._size_candidate(
                inst, signal, leverage_risk.get(inst), leverage_cfg, book
            )
            if cand is not None:
                candidates[inst] = cand

        # ---- 3. SYSTEM orchestrator → one SystemDecision. ----
        decision = self.orchestrator.decide(
            book,
            selection,
            open_actions,
            candidates,
            sys_cfg,
            risk_improvements=risk_improvements,
            round_trip_costs=round_trip_costs,
            bar_ts=bar_ts,
        )
        reason_codes.extend(decision.reason_codes)

        # ---- 4. lower the deltas through EXECUTION + rest the venue protective stop. ----
        commands: list[OrderCommand] = []
        for action in decision.actions:
            commands.extend(
                self._lower_action(
                    action,
                    open_positions=open_positions,
                    candidates=candidates,
                    micro_by_instrument=micro_by_instrument,
                    exec_cfg=exec_cfg,
                    intents=intents,
                    specs=specs,
                )
            )

        # ---- 5. OPS overlay (run/degrade/halt). HALT ⇒ flatten via the venue reduce-only stops. ----
        directive, flatten = self._run_ops(
            equity_ref=equity_ref, recon_mismatch=recon_mismatch, heartbeat_code=heartbeat_code
        )
        if directive.state == "halt":
            # HALT ⇒ "stop ALL" (OPS §7): no new entries may EVER be admitted while halted, on the
            # flatten-issue bar OR on a latched HALT bar where the §9.1a latch has already fired
            # (``flatten is None``). On the issue bar the flatten wave REPLACES the plan; on every
            # subsequent latched HALT bar we still strip new-signal admissions so a re-proposing
            # breakout cannot leak an entry past the kill (the latch suppresses re-flatten, NOT
            # re-entry). The venue reduce-only stop remains the standing floor either way (§9.1b).
            reason_codes.append("ops_halt")
            if flatten is not None:
                commands = list(self._flatten_commands(open_positions))
            else:
                commands = [
                    c for c in commands if "new_signal_admission" not in c.reason_codes
                ]
        elif directive.cancel_entries:
            reason_codes.append("ops_cancel_entries")
            commands = [
                c for c in commands if "new_signal_admission" not in c.reason_codes
            ]

        return ComposerResult(
            decision=decision,
            commands=tuple(commands),
            directive=directive,
            intents=intents,
            flatten=flatten,
            reason_codes=tuple(reason_codes),
        )

    # ---------------------------------------------------------------- internals

    def _size_candidate(
        self,
        instrument: str,
        signal: EntrySignal,
        risk: Any,
        leverage_cfg: Any,
        book: BookState,
    ) -> CandidateSignal | None:
        """Size a fresh breakout candidate to attach the RISK metadata the orchestrator admits on.

        The :class:`LeverageDecision` yields the per-lot heat (``wpc``) and ceiling weight; the
        ceiling slack is the room before the binding ceiling. NONE of the breakout score enters here
        (SYSTEM Inv CB / LEVERAGE Inv A) — only side + measurable risk.
        """
        if risk is None:
            return None
        sig_in = _signal_input(instrument, signal, leverage_cfg)
        lev = self.leverage.size_position(sig_in, risk, leverage_cfg, book)
        if lev.rounded_size <= 0.0:
            return None
        wpc = max(lev.risk_budget / max(lev.rounded_size, 1.0), 1e-9)
        return CandidateSignal(
            signal=signal,
            ceiling_weight=lev.risk_budget,
            wpc=wpc,
            ceiling_slack=max(lev.leverage, 0.0),
            equity_loss_per_lot=lev.planned_loss_on_equity,
        )

    def _lower_action(
        self,
        action: Any,
        *,
        open_positions: Mapping[str, OpenPositionView],
        candidates: Mapping[str, CandidateSignal],
        micro_by_instrument: Mapping[str, MicroState],
        exec_cfg: ExecResolvedConfig,
        intents: Mapping[str, object],
        specs: Mapping[str, InstrumentSpec],
    ) -> tuple[OrderCommand, ...]:
        """Lower ONE SYSTEM action into venue OrderCommands via EXECUTION + the protective stop.

        open   → EXECUTION.on_target(entry) → entry order(s) + a venue reduce-only stop at the
                 signal invalidation (the stop rests the moment the position opens, EXECUTION P6).
        reduce → EXECUTION.on_target(reduce-only) at TREND's target fraction.
        close  → EXECUTION.on_target(reduce-only exit).
        hold   → if TREND moved the trail, amend the venue stop; else a no-op.
        flat_all → reduce-only exit (governor flat).

        LOTS→UNITS GUARD (ARCHITECTURE §5.3): SYSTEM's ``target_units`` is an INTEGER LOT count
        (post-quantize, round-DOWN — LEVERAGE Inv U), but ``PositionState.qty`` and every venue
        :class:`ExecTarget`/:class:`OrderCommand` qty is in raw UNITS. The two are only equal when the
        contract multiplier (``InstrumentSpec.lot_size``, units-per-lot) is 1.0. We therefore convert
        every lot count through :func:`_lots_to_units` so a non-unit lot size (e.g. a 0.001-BTC perp
        contract) can never silently miscompute ``close_units`` or the entry qty.
        """
        inst = action.instrument
        kind = action.kind
        if kind == "hold":
            return self._maybe_amend_stop(inst, open_positions, intents, exec_cfg)
        micro = micro_by_instrument.get(inst)
        if micro is None:
            return ()
        spec = specs[inst]

        if kind == "open":
            cand = candidates.get(inst)
            if cand is None:
                return ()
            signal = cand.signal
            qty = _lots_to_units(action.target_units, spec) * action.side
            target = ExecTarget(
                intent_id=intent_id_for(inst, signal.entry_idx),
                instrument=inst,
                side=action.side,
                target_qty=qty,
                arrival_mid=micro.mid,
                invalidation=signal.invalidation,
                urgency="URGENT",  # BREAKOUT entries are adversarial-urgent (EXECUTION P4)
                reduce_only=False,
            )
            dec = self.execution.on_target(target, micro, WorkingState(), exec_cfg)
            stop = self.protective_stop.place(
                target.intent_id,
                PositionState(instrument=inst, side=action.side, qty=abs(qty)),
                signal.invalidation,
                exec_cfg,
                amend_generation=0,
            )
            # Tag every order belonging to a NEW-signal admission (the entry slices AND the fresh
            # protective stop) so an OPS DEGRADE directive (`cancel_entries`, OPS §7) can suppress
            # exactly the new opens while leaving incumbent management untouched. The EXECUTION layer
            # stamps tactic codes only, so the admission tag is added here at the composition seam.
            return tuple(
                replace(c, reason_codes=(*c.reason_codes, "new_signal_admission"))
                for c in (*dec.commands, stop)
            )

        # reduce / close / flat_all are reduce-only protective fills.
        view = open_positions.get(inst)
        if view is None:
            return ()
        if kind == "reduce":
            # ``target_units`` is the LOTS to KEEP; convert to UNITS before differencing against the
            # UNIT-denominated open qty so a non-unit contract multiplier cannot miscompute the close.
            keep_units = _lots_to_units(action.target_units, spec)
            close_units = abs(view.state.qty) - keep_units
        else:  # close / flat_all
            close_units = abs(view.state.qty)
        if close_units <= 0.0:
            return ()
        exit_side = -view.state.side
        target = ExecTarget(
            intent_id=view.intent_id,
            instrument=inst,
            side=exit_side,
            target_qty=close_units * exit_side,
            arrival_mid=micro.mid,
            invalidation=view.invalidation,
            urgency="IMMEDIATE",  # exits never abort (EXECUTION P5/P6)
            reduce_only=True,
        )
        working = WorkingState(
            filled_qty=view.state.qty,
            invalidation_generation=view.invalidation_generation,
        )
        dec = self.execution.on_target(target, micro, working, exec_cfg)
        return tuple(dec.commands)

    def _maybe_amend_stop(
        self,
        inst: str,
        open_positions: Mapping[str, OpenPositionView],
        intents: Mapping[str, object],
        exec_cfg: ExecResolvedConfig,
    ) -> tuple[OrderCommand, ...]:
        """If TREND emitted an ``amend_stop`` trail move on a held position, amend the venue stop.

        The venue stop is the real risk control (EXECUTION P6); a continue with a moved invalidation
        re-rests it place-before-cancel so there is never a no-stop window (Inv S).
        """
        intent = intents.get(inst)
        if intent is None or getattr(intent, "kind", "none") != "amend_stop":
            return ()
        view = open_positions.get(inst)
        if view is None:
            return ()
        return self.protective_stop.amend(
            view.intent_id,
            view.state,
            getattr(intent, "invalidation_level", view.invalidation),
            exec_cfg,
            amend_generation=view.invalidation_generation,
        )

    def _flatten_commands(
        self, open_positions: Mapping[str, OpenPositionView]
    ) -> tuple[OrderCommand, ...]:
        """The OPS HALT flatten: a reduce-only MARKET exit per open position (the venue reduce-only
        stop is the standing floor, OPS §9.1b; the live loop cancels working entries first)."""
        cmds: list[OrderCommand] = []
        for inst in sorted(open_positions):
            view = open_positions[inst]
            qty = abs(view.state.qty)
            if qty <= 0.0:
                continue
            exit_side = -view.state.side
            cmds.append(
                OrderCommand(
                    intent_id=view.intent_id,
                    instrument=inst,
                    side=exit_side,
                    kind="MARKET",
                    qty=qty,
                    price=0.0,
                    time_in_force="IOC",
                    post_only=False,
                    reduce_only=True,
                    client_order_id=slice_client_order_id(view.intent_id, 99, 0),
                    action="submit",
                    reason_codes=("ops_halt_flatten",),
                )
            )
        return tuple(cmds)

    def _run_ops(
        self,
        *,
        equity_ref: EquityRef | None,
        recon_mismatch: bool,
        heartbeat_code: float,
    ) -> tuple[OpsDirective, FlattenCommand | None]:
        """Advance the OPS kill machine for the bar and pull back the directive + any flatten."""
        if self.ops is None:
            return (
                OpsDirective("run", False, False, ("ops_disabled",), "ops-1.0.0"),
                None,
            )
        return self.ops.on_bar(
            equity=equity_ref, recon_mismatch=recon_mismatch, heartbeat_code=heartbeat_code
        )


# --------------------------------------------------------------------------- helpers


def intent_id_for(instrument: str, entry_idx: int) -> str:
    """Deterministic per-position intent id (stable across restart; the stop id derives from it)."""
    return f"{instrument}:{entry_idx}"


def _lots_to_units(lots: int, spec: InstrumentSpec) -> float:
    """Convert an integer SYSTEM LOT count into raw position UNITS via the contract multiplier.

    The contract multiplier is ``InstrumentSpec.lot_size`` (units per lot — e.g. 0.001 BTC for the
    Binance USDT-M perp). SYSTEM emits an INTEGER ``target_units`` that is really a LOT count
    (post-quantize round-DOWN, LEVERAGE Inv U); the venue order/position layer is UNIT-denominated.
    This is the single seam where LOTS meet UNITS, so it asserts the contract is sane (a positive
    finite multiplier) before scaling — a misconfigured ``lot_size`` can never silently turn N lots
    into N units (which only coincides when ``lot_size == 1.0``).
    """
    mult = float(spec.lot_size)
    assert mult > 0.0 and mult == mult, (  # mult==mult rejects NaN
        f"contract multiplier (lot_size) for {spec.symbol!r} must be positive/finite, got {mult!r}"
    )
    return float(lots) * mult


def _signal_input(instrument: str, signal: EntrySignal, leverage_cfg: Any):
    """Build a LEVERAGE ``SignalInput`` from a breakout entry (no edge field — Inv A)."""
    from convexity.leverage.types import SignalInput as LevSignalInput

    position_margin = float(getattr(leverage_cfg.defaults, "position_margin", 1.0))
    capital_cap = float(getattr(leverage_cfg.defaults, "capital_cap", 0.12))
    return LevSignalInput(
        instrument=instrument,
        side=signal.direction,
        entry_price=signal.entry_ref_price,
        position_margin=position_margin,
        capital_cap=capital_cap,
    )


def _trend_action_from(decision: object) -> TrendAction:
    """Project a TREND ``ActionDecision`` (or None=closed) onto the SYSTEM :class:`TrendAction` view.

    Reads only the boundary fields (action kind, reduce fraction, time-cap cause); SYSTEM never
    recomputes trend state (P1). ``alive`` is True only on ``continue``.
    """
    if decision is None:
        return TrendAction(kind="exit", alive=False, target_fraction=0.0, time_cap=False,
                           reason_codes=("trend_closed",))
    action = getattr(decision, "action", "continue")
    frac = float(getattr(decision, "size_fraction", 1.0))
    time_cap = getattr(decision, "stop_cause", None) == "time_cap"
    return TrendAction(
        kind=action,
        alive=action == "continue",
        target_fraction=frac if action == "reduce" else (0.0 if action == "exit" else 1.0),
        time_cap=time_cap,
        reason_codes=tuple(getattr(decision, "reason_codes", ())),
    )


# --------------------------------------------------------------------------- the Nautilus Strategy


@dataclass(frozen=True)
class StrategyDeps:
    """The pure-Python dependency bundle the live :class:`_ConvexityStrategy` hosts (NautilusTrader-
    free). Injected after construction via :meth:`_ConvexityStrategy.configure` so the msgspec
    ``StrategyConfig`` stays a thin serializable descriptor while the heavy, non-serializable
    composer + resolved pillar configs are passed by reference (the same instance runs unchanged in
    backtest / paper / live — ARCHITECTURE §1/§4).

    ``micro_builder`` derives the per-bar :class:`MicroState`; defaulting to
    :func:`projection.micro_from_bar` keeps the 1-min-kline path runnable. ``per_instrument`` maps a
    Nautilus instrument-id string → its pure :class:`InstrumentSpec` so the strategy can resolve the
    AMBER metadata for the lots↔units conversion and the microstate.
    """

    composer: "ConvexityComposer"
    specs: Mapping[str, InstrumentSpec]
    leverage_risk: Mapping[str, Any]
    leverage_cfg: Any
    trend_cfg: Any
    exec_cfg: ExecResolvedConfig
    sys_cfg: Any
    instrument_to_symbol: Mapping[str, str]
    symbol_to_instrument: Mapping[str, str]
    warmup_bars: int = 32
    # CROSS-MODE GAP MARKER (FRICTION #2): the BACKTEST half of the per-bar gap signal. The stitched
    # lake's authoritative ``gap_before`` column is threaded here as ``{instrument_id: {bar_close_ms}}``
    # — the set of bar-CLOSE timestamps the lake flagged as the first bar after a stitch boundary. The
    # strategy's ``on_bar`` passes ``gap_before=ts in this set`` into the rolling buffer so a stitch
    # gap that leaves NO timestamp hole (a 24/7 crypto source boundary) still raises the SAME
    # ``Bars.is_session_open`` flag the live ts-discontinuity path raises. ``None`` ⇒ live/paper, where
    # the buffer auto-derives the flag from the bar-timestamp discontinuity (reconnect / missing bar).
    gap_close_ms_by_instrument: Mapping[str, frozenset[int]] | None = None


def make_strategy_class():  # pragma: no cover - VM-only (imports nautilus_trader)
    """Build the live ``_ConvexityStrategy``, importing ``nautilus_trader`` LAZILY (Invariant N).

    The class is constructed inside this function so importing this module in the pure-Python unit
    venv (which has no nautilus) never pulls in the engine.

    This is the COMPOSITION HOST (ARCHITECTURE §4): the load-bearing logic — the §4 per-bar
    composition, restart reconciliation, the reduce-only protective stop, the OPS overlay — lives in
    the pure-Python :class:`ConvexityComposer`; this class is the thin Nautilus EVENT ROUTER that
    hosts it. It subscribes to bars/order/position/fill events, projects the engine's authoritative
    ``Portfolio``/``Cache`` into the pure pillar types via :mod:`convexity.strategy.projection`,
    calls ``composer.reconcile_on_start`` from :meth:`on_start` (after Nautilus has run ITS own
    reconciliation — the composer then rebuilds the venue protective stops idempotently, Inv S),
    calls ``composer.on_decision_bar`` from :meth:`on_bar`, and submits the returned
    :class:`OrderCommand`s as Nautilus ``SubmitOrder``/``ModifyOrder``/``CancelOrder``. It holds NO
    composition logic of its own and NEVER tracks position/account state itself (Invariant N) — it
    reads Nautilus and hands the numbers to the composer. Managing N simultaneous per-pair positions
    is just N entries in the open-position projection; the composer is per-instrument keyed.
    """
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide, TriggerType
    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.trading.strategy import Strategy

    from convexity.strategy.projection import (
        PositionSnapshot,
        RollingBars,
        book_state_from_snapshot,
        micro_from_bar,
        open_views_from_snapshot,
    )

    _TRIGGER = {
        "mark": TriggerType.MARK_PRICE,
        "index": TriggerType.INDEX_PRICE,
        "last": TriggerType.LAST_PRICE,
    }

    class _ConvexityStrategy(Strategy):
        """The live ``Strategy`` host (ARCHITECTURE §4): a thin event router over the pure composer."""

        def __init__(self, config) -> None:
            super().__init__(config)
            self._deps: StrategyDeps | None = None
            self._bar_types: list[str] = list(getattr(config, "bar_types", ()))
            self._rolling: dict[str, RollingBars] = {}
            self._instruments: dict[str, Any] = {}
            self._reconciled = False
            # Cold-start warmup parity (paper/live): the backtest PRELOADS the full bar slice, so the
            # rolling buffer is warm by the first decision bar; a live/paper node only subscribes, so
            # it would sit idle for ``warmup_bars`` live minutes after every (re)start. When
            # ``request_warmup`` is set (paper/live build it True; backtest leaves it False because it
            # preloads), ``on_start`` requests ``warmup_bars`` of history per bar-type and
            # ``on_historical_data`` primes the SAME rolling buffer the live bars feed — so the same
            # config reaches its first decision at the same buffer depth in every mode (FRICTION #7).
            self._request_warmup = bool(getattr(config, "request_warmup", False))
            self.historical_bars_seen = 0
            # Funding-subscription parity audit (FRICTION #8): the backtest charges funding from the
            # fed ``FundingRateUpdate`` lake; paper/live rely on the venue funding stream. A silently
            # swallowed subscribe-failure would make funding zero in live while it is charged in
            # backtest — a parity gap that must be VISIBLE, not hidden. Counted + logged here.
            self.funding_subscribed = 0
            self.funding_subscribe_failed = 0
            # Heartbeat/staleness tracking: the live OPS §7.6 trigger reads bar lateness/gaps off the
            # engine clock vs the last bar ts (NOT position/account state — Inv N). A gap beyond the
            # staleness window classifies HB_STALE (DEGRADE); a clean bar is HB_OK.
            self._last_bar_ts_ms: dict[str, int] = {}
            self._heartbeat_code = 0.0  # HB_OK
            self._heartbeat_override: float | None = None  # set by an external feed-health monitor
            self._bar_interval_ms = 60_000  # 1-min klines; refined from the observed cadence
            # Reconciliation mismatch latch: a fill/position event flagged ``reconciliation`` (the venue
            # carried state the engine had to backfill) raises it for the §7.5 trigger (Inv PROV/RECON).
            self._recon_mismatch = False
            self._recon_detail = ""
            # Running-max equity peak for the DD-kill (§7.1a); an optional override lets an external
            # equity reconciler / the e2e OPS test drive the equity series directly (Inv N preserved —
            # this is the SHARED equity reference, not position/account bookkeeping).
            self._equity_peak = 0.0
            self._equity_override: float | None = None
            # Per-bar-clock entry de-dup (Inv: a re-proposing breakout must not submit the same
            # deterministic entry client_order_id twice within one bar before the first fill).
            self._submitted_coids_this_bar: set[str] = set()
            self._bar_clock_ms = -1
            # Fill-rate health for the §7.4 trigger (event-driven, read off filled-vs-submitted).
            self._fills_seen = 0
            self._fills_full = 0
            # PERSISTENT OPS stream adapters: one instance per stream for the whole run so each
            # observation carries a MONOTONE stream-local sample_idx (Inv STREAM/CAUSAL) — a fresh
            # adapter per event would reset the index and break the detectors' ARL counting.
            self._ops_streams: dict[str, Any] = {}
            # Audit counters (NOT position/account state — those are read from Nautilus, Inv N).
            self.bars_seen = 0
            self.decisions = 0
            self.orders_submitted = 0
            self.stops_submitted = 0
            self.entries_submitted = 0
            self.reduces_submitted = 0
            self.ops_obs_pushed = 0
            self.entries_canceled_on_halt = 0
            self.flatten_outcomes_reported = 0
            # The last OPS directive state the node acted on (audit only — proves the kill machine is
            # actually escalating RUN→DEGRADE→HALT off the live reads, not stuck at a stubbed RUN).
            self.last_directive_state = "run"
            self.degrade_bars = 0
            self.halt_bars = 0
            # COST funding intake (§5.5): the realized ``FundingRateUpdate`` stamps Nautilus ingests
            # are accumulated per instrument here (Inv N: a re-key of the venue's funding events, not
            # a re-model) and summed into the realized funding accrual at each position close via the
            # validated COST ``FundingAccrual``. The accrual is only charged because the funding feed
            # is fed to the engine (COST §5.5 — no feed ⇒ directional-only, no funding term).
            self._funding_events: dict[str, list[Any]] = {}
            self._funding_accrual: Any = None
            self.funding_events_seen = 0
            self.funding_accrued_quote = 0.0

        # -------------------------------------------------------------- dependency injection

        def configure(self, deps: StrategyDeps) -> None:
            """Inject the pure-Python composer + resolved pillar configs (called before the run)."""
            self._deps = deps

        # -------------------------------------------------------------- lifecycle

        def on_start(self) -> None:
            assert self._deps is not None, "_ConvexityStrategy.configure(deps) must run before start"
            for bt_str in self._bar_types:
                bar_type = BarType.from_str(bt_str)
                iid = str(bar_type.instrument_id)
                instrument = self.cache.instrument(bar_type.instrument_id)
                if instrument is not None:
                    self._instruments[iid] = instrument
                # The rolling buffer derives the per-bar gap flag (Bars.is_session_open) from the SAME
                # bar-interval the heartbeat uses, so the live ts-discontinuity gap (reconnect/missing
                # bar) and the backtest's lake gap_before raise the identical flag into the pillars.
                self._rolling.setdefault(iid, RollingBars(bar_interval_ms=self._bar_interval_ms))
                self.subscribe_bars(bar_type)
                # Cold-start parity: prime the rolling buffer from venue history (paper/live only;
                # backtest preloads and leaves ``request_warmup`` False). One request per bar-type;
                # the response lands in ``on_historical_data`` which pushes into ``_rolling``.
                if self._request_warmup and self._deps is not None:
                    self._request_warmup_history(bar_type)
                # Subscribe to the venue funding stream so COST can sum the realized stamps over the
                # hold (§5.5). Guarded: a feed-less venue/instrument simply yields no events. The
                # outcome is COUNTED + logged (not silently swallowed) so a live funding-feed gap vs
                # the backtest funding lake is observable (FRICTION #8).
                if instrument is not None:
                    try:
                        self.subscribe_funding_rates(instrument.id)
                        self.funding_subscribed += 1
                    except Exception as exc:  # noqa: BLE001 — non-perp / no funding feed
                        self.funding_subscribe_failed += 1
                        self.log.warning(
                            f"funding-rate subscription failed for {instrument.id} "
                            f"({type(exc).__name__}: {exc}); funding will be ZERO for this "
                            "instrument live while the backtest charges it — parity gap."
                        )
            # Restart reconciliation: rebuild the venue protective stops idempotently (Inv S). Runs
            # AFTER Nautilus's own reconciliation, against the open positions Nautilus reports.
            snapshots = self._position_snapshots()
            if snapshots:
                views = open_views_from_snapshot(snapshots)
                cmds = self._deps.composer.reconcile_on_start(views, self._deps.exec_cfg)
                for cmd in cmds:
                    self._submit_command(cmd)
            self._reconciled = True

        def _request_warmup_history(self, bar_type: "BarType") -> None:
            """Request ``warmup_bars`` of history for one bar-type so the rolling buffer is primed at
            (re)start — the live/paper analogue of the backtest's bar preload (FRICTION #7).

            Asks for a little MORE than ``warmup_bars`` (a small margin) so the first live bar after the
            history window already clears warmup. The window is computed off the engine clock so it is
            identical paper↔live. Guarded: a venue/catalog that cannot serve history simply yields no
            historical-data callback (the node then warms from live bars, as before)."""
            from datetime import timedelta

            deps = self._deps
            assert deps is not None
            margin = 8  # request a few extra so the first LIVE bar is already past warmup
            n = deps.warmup_bars + margin
            interval_ms = self._bar_interval_ms
            try:
                now = self.clock.utc_now()
                start = now - timedelta(milliseconds=interval_ms * n)
                self.request_bars(bar_type, start=start, limit=n)
            except Exception as exc:  # noqa: BLE001 — no history client/catalog: warm from live bars
                self.log.warning(
                    f"warmup history request failed for {bar_type} "
                    f"({type(exc).__name__}: {exc}); buffer will warm from live bars instead."
                )

        def on_historical_data(self, data) -> None:
            """Prime the rolling buffer from requested warmup history (FRICTION #7).

            The response to :meth:`_request_warmup_history`. Historical bars are pushed into the SAME
            ``_rolling`` buffer the live ``on_bar`` feeds — but they DO NOT trigger decisions (no
            composition, no OPS heartbeat classification): they only fill the buffer so the first LIVE
            bar reaches a warm buffer. Idempotent across restarts (``RollingBars`` is a bounded deque)."""
            try:
                iterable = list(data)
            except TypeError:
                iterable = [data]
            for item in iterable:
                bar_type = getattr(item, "bar_type", None)
                if bar_type is None:
                    continue  # not a Bar (e.g. some other historical data type) — ignore
                iid = str(bar_type.instrument_id)
                roll = self._rolling.get(iid)
                if roll is None:
                    continue
                hist_ts_ms = int(item.ts_event // 1_000_000)
                roll.push(
                    ts=hist_ts_ms,
                    o=float(item.open), h=float(item.high), low=float(item.low),
                    c=float(item.close), v=float(item.volume),
                    gap_before=self._lake_gap_before(iid, hist_ts_ms),
                )
                self.historical_bars_seen += 1

        # -------------------------------------------------------------- the per-bar composition

        def on_bar(self, bar: Bar) -> None:
            deps = self._deps
            assert deps is not None
            iid = str(bar.bar_type.instrument_id)
            roll = self._rolling.get(iid)
            if roll is None:
                return
            bar_ts_ms = int(bar.ts_event // 1_000_000)  # ns → ms (Bars.ts is epoch-ms)
            self._classify_heartbeat(iid, bar_ts_ms)
            # Per-bar gap flag (Bars.is_session_open), threaded IDENTICALLY in every mode (FRICTION #2):
            #   * backtest: the lake's authoritative gap_before column, keyed by bar-close ts (deps).
            #   * live/paper: None ⇒ the buffer auto-derives it from the bar-timestamp discontinuity.
            # In both cases the flag is OR-ed with the ts-discontinuity test inside RollingBars, so a
            # real timestamp hole always raises it regardless of what the upstream column said.
            roll.push(
                ts=bar_ts_ms,
                o=float(bar.open), h=float(bar.high), low=float(bar.low),
                c=float(bar.close), v=float(bar.volume),
                gap_before=self._lake_gap_before(iid, bar_ts_ms),
            )
            self.bars_seen += 1
            if len(roll) < deps.warmup_bars:
                return
            self._decide_for(iid, bar)

        def _lake_gap_before(self, iid: str, bar_ts_ms: int) -> bool | None:
            """The backtest's explicit per-bar gap flag for this bar, or None in live/paper.

            Returns True iff the stitched lake flagged THIS bar (by its close ts) as the first bar
            after a stitch boundary (``StrategyDeps.gap_close_ms_by_instrument``). Returns None when no
            lake-gap map is wired (the live/paper path), so ``RollingBars`` auto-derives the flag from
            the bar-timestamp discontinuity instead — the one cross-mode gap mechanism (FRICTION #2)."""
            deps = self._deps
            if deps is None or deps.gap_close_ms_by_instrument is None:
                return None
            return bar_ts_ms in deps.gap_close_ms_by_instrument.get(iid, frozenset())

        def _classify_heartbeat(self, iid: str, bar_ts_ms: int) -> None:
            """Derive the OPS §7.6 heartbeat/staleness code from the inter-bar gap (P6, fail-safe).

            A 1-min kline feed must produce a bar every ``_bar_interval_ms``; a gap materially beyond
            that window means the feed went dark / the bar arrived late — HB_STALE (DEGRADE). A clean
            on-cadence bar is HB_OK. The code is read by ``_decide_for`` and fed to the OPS engine so
            the §7.6 trigger can actually fire in production (it can never fire on a hard-coded 0.0).
            """
            from convexity.ops.streams.heartbeat import HB_OK, HB_STALE

            last = self._last_bar_ts_ms.get(iid)
            self._last_bar_ts_ms[iid] = bar_ts_ms
            # An explicit external override (a dedicated feed-health monitor, or a HALT-class condition
            # no bar arrives to signal — HB_MISS/HB_CLOCK_SKEW) takes precedence over the bar-gap read.
            if self._heartbeat_override is not None:
                self._heartbeat_code = self._heartbeat_override
                return
            if last is None:
                self._heartbeat_code = HB_OK
                return
            gap = bar_ts_ms - last
            # Allow up to 1.5× the bar interval before declaring the feed stale (one missed bar).
            self._heartbeat_code = (
                HB_STALE if gap > int(self._bar_interval_ms * 1.5) else HB_OK
            )

        def set_heartbeat_code(self, code: float | None) -> None:
            """Inject the OPS §7.6 heartbeat/staleness/clock-skew code directly (used by an external
            feed-health monitor and by the end-to-end OPS test). It LATCHES as an override that takes
            precedence over the per-bar gap classification so a true heartbeat-loss (HB_MISS/
            HB_CLOCK_SKEW → HALT) — which no bar arrives to signal — can still drive the kill machine.
            Pass ``None`` to clear the override and resume the bar-gap classification."""
            self._heartbeat_override = None if code is None else float(code)
            if code is not None:
                self._heartbeat_code = float(code)

        def flag_recon_mismatch(self, mismatch: bool, detail: str = "") -> None:
            """Set the SYSTEM/Nautilus reconciliation-mismatch flag for the OPS §7.5 trigger (P5)."""
            self._recon_mismatch = bool(mismatch)
            self._recon_detail = detail

        def _decide_for(self, decision_iid: str, bar: Bar) -> None:
            """Build the composer inputs across ALL instruments and run one decision bar."""
            deps = self._deps
            assert deps is not None
            bars_by_instrument: dict[str, Bars] = {}
            idx_by_instrument: dict[str, int] = {}
            micro_by_instrument: dict[str, MicroState] = {}
            specs: dict[str, InstrumentSpec] = {}
            eligible: list[str] = []
            for iid, roll in self._rolling.items():
                if len(roll) < deps.warmup_bars:
                    continue
                symbol = deps.instrument_to_symbol[iid]
                spec = deps.specs[symbol]
                bars = roll.as_bars()
                bars_by_instrument[symbol] = bars
                idx_by_instrument[symbol] = len(bars) - 1
                specs[symbol] = spec
                mid = float(bars.c[-1])
                micro_by_instrument[symbol] = micro_from_bar(
                    spec, mid=mid,
                    bar_range=float(bars.h[-1] - bars.l[-1]),
                    volume=float(bars.v[-1]) if bars.v is not None else 1.0,
                )
                eligible.append(symbol)
            if not bars_by_instrument:
                return

            snapshots = self._position_snapshots()
            equity = self._equity()
            peak_equity = self._peak_equity()
            book = book_state_from_snapshot(
                equity=equity, peak_equity=peak_equity,
                free_margin=self._free_margin(), snapshots=snapshots, specs=deps.specs,
            )
            open_views = open_views_from_snapshot(snapshots)
            selection = ProposedPortfolio(
                keep=tuple(open_views), eligible=tuple(sorted(eligible)),
                risk_weights={s: 0.05 for s in eligible}, rank={s: 0.0 for s in eligible},
                reason_codes=("live_selection",),
            )
            bar_ts_ms = int(bar.ts_event // 1_000_000)
            # Reset the per-bar-clock entry de-dup set when the bar clock advances (N-pair guard).
            if bar_ts_ms != self._bar_clock_ms:
                self._bar_clock_ms = bar_ts_ms
                self._submitted_coids_this_bar = set()
            # Feed the OPS safety net the REAL production reads (Inv PROV/RECON — wired, not stubbed):
            #   * a live EquityRef from Nautilus's Portfolio (DD-kill §7.1),
            #   * the reconciliation-mismatch flag (§7.5),
            #   * the heartbeat/staleness code derived from the bar cadence (§7.6).
            result = deps.composer.on_decision_bar(
                bars_by_instrument=bars_by_instrument,
                idx_by_instrument=idx_by_instrument,
                specs=specs,
                book=book,
                open_positions=open_views,
                selection=selection,
                leverage_risk=deps.leverage_risk,
                leverage_cfg=deps.leverage_cfg,
                trend_cfg=deps.trend_cfg,
                exec_cfg=deps.exec_cfg,
                micro_by_instrument=micro_by_instrument,
                sys_cfg=deps.sys_cfg,
                bar_ts=bar_ts_ms,
                equity_ref=EquityRef(equity=equity, peak=peak_equity),
                recon_mismatch=self._recon_mismatch,
                heartbeat_code=self._heartbeat_code,
            )
            self.decisions += 1
            self.last_directive_state = result.directive.state
            if result.directive.state == "degrade":
                self.degrade_bars += 1
            elif result.directive.state == "halt":
                self.halt_bars += 1
            # OPS §9.1b HALT discipline: cancel working ENTRY orders FIRST, then submit the reduce-only
            # flatten, so a resting entry cannot fill mid-flatten. The composer's flatten directive
            # (`cancel_entries_first`) drives this; the venue reduce-only stop is the standing floor.
            if result.flatten is not None and result.flatten.cancel_entries_first:
                self._cancel_working_entries(open_views, specs)
            for cmd in result.commands:
                self._submit_command(cmd)
            # Relay the flatten outcome back to the kill machine (§9.1c): once the reduce-only exits are
            # sent, report flat iff Nautilus now reports no open position (venue-confirmed, never on
            # merely-sent). A still-open book leaves OPS latched-HALT-with-retry (never spurious "flat").
            if result.flatten is not None and deps.composer.ops is not None:
                still_open = bool(self._position_snapshots())
                deps.composer.ops.report_flatten_outcome(
                    venue_confirmed_flat=not still_open, partial=still_open
                )
                self.flatten_outcomes_reported += 1

        def _cancel_working_entries(
            self, open_views, specs: Mapping[str, InstrumentSpec]
        ) -> None:
            """Cancel every working (open, NON-reduce-only) ENTRY order across the traded instruments
            before a HALT flatten (OPS §9.1b). Reduce-only protective stops are LEFT resting (they are
            the floor); only resting entry/limit opens are pulled so they cannot fill mid-flatten."""
            deps = self._deps
            assert deps is not None
            for iid_str, instrument in self._instruments.items():
                for order in self.cache.orders_open(instrument_id=instrument.id):
                    if getattr(order, "is_reduce_only", False):
                        continue
                    if order.is_closed:
                        continue
                    self.cancel_order(order)
                    self.entries_canceled_on_halt += 1

        # -------------------------------------------------------------- OPS event intake (Inv PROV)

        def _ops(self):
            """The wired :class:`OpsAdapter` (or None when OPS is disabled). The composer owns it."""
            deps = self._deps
            return deps.composer.ops if deps is not None else None

        def _ops_stream(self, name: str):
            """A PERSISTENT OPS stream adapter (monotone sample_idx across the run, Inv STREAM)."""
            s = self._ops_streams.get(name)
            if s is None:
                if name == "fills":
                    from convexity.ops.streams.fills import FillsStream
                    s = FillsStream()
                elif name == "cost_tca":
                    from convexity.ops.streams.cost_tca import CostTcaStream
                    s = CostTcaStream()
                else:  # edge
                    from convexity.ops.streams.edge import EdgeStream
                    s = EdgeStream()
                self._ops_streams[name] = s
            return s

        def on_order_filled(self, event) -> None:
            """Push the EXECUTION fill into OPS's statistical streams (§5.1 fills + §5.4 edge) and the
            §7.4 fill-rate health, and raise the §7.5 reconciliation flag if Nautilus had to backfill
            this fill (``OrderFilled.reconciliation`` — the venue carried state the engine did not).

            Net-of-cost edge is read only at POSITION CLOSE (``on_position_closed``); a per-fill push
            here drives the fills/cost residual streams + the realized fill-rate the §7.4 collapse
            trigger consumes. Each push is one causal :class:`StreamObs` (Inv STREAM/CAUSAL)."""
            ops = self._ops()
            if ops is None:
                return
            from convexity.ops.streams.cost_tca import TcaFact
            from convexity.ops.streams.fills import FillFact

            if getattr(event, "reconciliation", False):
                # A reconciliation fill means the venue was ground truth and the engine backfilled —
                # surface it to the §7.5 mismatch trigger (cleared once a clean decision bar reconciles).
                self._recon_mismatch = True
                self._recon_detail = "order_filled_via_reconciliation"

            self._fills_seen += 1
            self._fills_full += 1  # a fill landed (vs a reject/expire which would not)
            ts_ms = int(getattr(event, "ts_event", 0) // 1_000_000)
            # Implementation-shortfall + fill-rate residuals (live==expected in the synthetic path; a
            # live cost layer supplies the realized-vs-forecast bps). The push exists so the §7.4/§5.1
            # detectors accumulate in production — the residual values are the data layer's to supply.
            fill_obs = self._ops_stream("fills").observe(
                FillFact(is_realized_bps=0.0, is_expected_bps=0.0,
                         fillrate_realized=1.0, fillrate_expected=1.0),
                ts=ts_ms,
            )
            ops.on_obs(fill_obs)
            cost_obs = self._ops_stream("cost_tca").observe_cost(
                TcaFact(cost_realized_bps=0.0, cost_forecast_bps=0.0), ts=ts_ms
            )
            ops.on_obs(cost_obs)
            ops.on_fill_health(
                fillrate_realized=self._fills_full / max(self._fills_seen, 1),
                n_orders=self._fills_seen,
            )
            self.ops_obs_pushed += 2

        def on_order_rejected(self, event) -> None:
            """A rejected order degrades realized fill-rate (the §7.4 collapse signal). If the rejected
            order was a reduce-only protective/exit order, that is an ``exit cannot fill`` condition
            (§9, escalate); a rejected ENTRY just lowers the realized fill rate."""
            ops = self._ops()
            if ops is None:
                return
            self._fills_seen += 1
            exit_cannot_fill = False
            coid = getattr(getattr(event, "client_order_id", None), "value", "")
            existing = self.cache.order(ClientOrderId(coid)) if coid else None
            if existing is not None and getattr(existing, "is_reduce_only", False):
                exit_cannot_fill = True
            ops.on_fill_health(
                fillrate_realized=self._fills_full / max(self._fills_seen, 1),
                n_orders=self._fills_seen,
                exit_cannot_fill=exit_cannot_fill,
            )

        def on_funding_rate(self, update) -> None:
            """Record a realized funding stamp Nautilus ingested (COST §5.5, FACT — Inv N).

            Each :class:`FundingRateUpdate` is re-keyed (not re-modelled) into a COST
            :class:`FundingEvent` carrying the SIGNED position notional at the stamp (read off
            Nautilus's authoritative position state). These are summed over the hold by the validated
            :class:`FundingAccrual` at each position close — the engine only charges funding because
            the feed is fed (no feed ⇒ no events ⇒ directional-only)."""
            from convexity.cost.facts.nautilus_adapter import FundingEvent

            deps = self._deps
            if deps is None:
                return
            iid_str = str(update.instrument_id)
            symbol = deps.instrument_to_symbol.get(iid_str)
            if symbol is None:
                return
            instrument = self._instruments.get(iid_str)
            notional = 0.0
            if instrument is not None:
                positions = self.cache.positions_open(instrument_id=instrument.id)
                if positions:
                    pos = positions[0]
                    signed = float(pos.signed_qty)
                    notional = signed * float(getattr(pos, "avg_px_open", 0.0) or 0.0)
            ts_ms = int(getattr(update, "ts_event", 0) // 1_000_000)
            self._funding_events.setdefault(symbol, []).append(
                FundingEvent(ts=ts_ms, rate=float(update.rate), notional_at_stamp=notional)
            )
            self.funding_events_seen += 1

        def on_position_closed(self, event) -> None:
            """Push the realized net-of-cost EDGE of the closed round-trip into OPS (§5.4 edge / §7.7
            edge-decay) and sum the realized funding accrued over the hold (COST §5.5). The Bernoulli
            hit (profitable?) + the R-multiple edge feed the CUSUM-SPRT + edge CUSUM so the validated
            edge is monitored in production, not just in backtest."""
            self._accrue_funding_on_close(event)
            ops = self._ops()
            if ops is None:
                return
            from convexity.ops.streams.edge import TradeFact

            pnl = float(getattr(getattr(event, "realized_pnl", None), "as_double", lambda: 0.0)())
            ret = float(getattr(event, "realized_return", 0.0) or 0.0)
            stream = self._ops_stream("edge")
            ts_ms = int(getattr(event, "ts_event", 0) // 1_000_000)
            fact = TradeFact(edge_realized=ret, profitable=pnl > 0.0)
            ops.on_obs(stream.observe_hit(fact, ts=ts_ms))
            ops.on_obs(stream.observe_edge(fact, ts=ts_ms))
            self.ops_obs_pushed += 2

        def _accrue_funding_on_close(self, event) -> None:
            """Sum the realized funding stamps over [entry, close] for the closed round-trip via the
            validated COST :class:`FundingAccrual` (§5.5). Pure re-key/aggregate of the ingested
            events (Inv N); the SIGNED accrual is recorded for telemetry/attribution."""
            deps = self._deps
            if deps is None:
                return
            iid_str = str(getattr(event, "instrument_id", ""))
            symbol = deps.instrument_to_symbol.get(iid_str)
            if symbol is None:
                return
            events = tuple(self._funding_events.get(symbol, ()))
            if not events:
                return
            from convexity.cost.components.funding_accrual import FundingAccrual

            if self._funding_accrual is None:
                self._funding_accrual = FundingAccrual()
            opened_ns = int(getattr(event, "ts_opened", 0) or 0)
            closed_ns = int(getattr(event, "ts_closed", getattr(event, "ts_event", 0)) or 0)
            t_entry = opened_ns // 1_000_000
            t_now = closed_ns // 1_000_000
            # The closed position's entry side (+1 long / -1 short) signs the funding (Inv P).
            side = -1 if str(getattr(event, "entry", "")).upper() == "SELL" else 1
            accrued = self._funding_accrual.accrued(side, events, t_entry, t_now)
            self.funding_accrued_quote += float(accrued)

        # -------------------------------------------------------------- command → Nautilus order

        def _submit_command(self, cmd: OrderCommand) -> None:
            deps = self._deps
            assert deps is not None
            iid_str = deps.symbol_to_instrument.get(cmd.instrument)
            if iid_str is None:
                return
            instrument = self._instruments.get(iid_str)
            if instrument is None:
                return
            order_side = OrderSide.BUY if cmd.side == 1 else OrderSide.SELL
            qty = instrument.make_qty(round(cmd.qty, 9))
            coid = ClientOrderId(cmd.client_order_id)

            if cmd.action == "cancel":
                existing = self.cache.order(coid)
                if existing is not None and not existing.is_closed:
                    self.cancel_order(existing)
                return

            # N-PAIR DUPLICATE-SUBMIT GUARD (Inv: a re-proposing breakout re-runs the full cross-
            # instrument composition every per-instrument bar, so the SAME deterministic entry
            # client_order_id can be emitted twice within one bar-clock BEFORE the first fill —
            # Nautilus rejects duplicate client_order_ids. For a SUBMIT, guard on both (i) the per-bar
            # set and (ii) Nautilus's authoritative cache (Inv N), mirroring the cancel/modify path.
            if cmd.action == "submit":
                if cmd.client_order_id in self._submitted_coids_this_bar:
                    return
                if self.cache.order(coid) is not None:
                    return
                self._submitted_coids_this_bar.add(cmd.client_order_id)

            if cmd.kind in ("STOP_MARKET", "STOP_LIMIT"):
                trigger = instrument.make_price(round(cmd.price, 9))
                if cmd.action == "modify":
                    existing = self.cache.order(coid)
                    if existing is None or existing.is_closed:
                        # Native-modify amend: the policy versions the stop id to the NEW generation
                        # (Inv Q-STOP), but a venue-native modify keeps the RESTING order's id — so the
                        # new-gen id is not yet in the cache. Modify the resting reduce-only stop for
                        # this intent IN PLACE (Nautilus carries its id forward) rather than submitting
                        # a fresh stop, which would orphan the prior generation (Inv S: exactly one
                        # resting protective stop per intent — never a duplicate at the venue).
                        existing = self._resting_stop_order(instrument, cmd.intent_id)
                    if existing is not None and not existing.is_closed:
                        self.modify_order(existing, trigger_price=trigger)
                        return
                order = self.order_factory.stop_market(
                    instrument_id=instrument.id, order_side=order_side, quantity=qty,
                    trigger_price=trigger, trigger_type=_TRIGGER[cmd.trigger_reference],
                    reduce_only=cmd.reduce_only, client_order_id=coid,
                )
                self.submit_order(order)
                self.stops_submitted += 1
                self.orders_submitted += 1
                return

            if cmd.kind == "LIMIT":
                order = self.order_factory.limit(
                    instrument_id=instrument.id, order_side=order_side, quantity=qty,
                    price=instrument.make_price(round(cmd.price, 9)),
                    post_only=cmd.post_only, reduce_only=cmd.reduce_only, client_order_id=coid,
                )
            else:  # MARKET
                order = self.order_factory.market(
                    instrument_id=instrument.id, order_side=order_side, quantity=qty,
                    reduce_only=cmd.reduce_only, client_order_id=coid,
                )
            self.submit_order(order)
            self.orders_submitted += 1
            if cmd.reduce_only:
                self.reduces_submitted += 1
            else:
                self.entries_submitted += 1

        # -------------------------------------------------------------- Nautilus state readers (Inv N)

        def _position_snapshots(self) -> list[PositionSnapshot]:
            """Read the open positions off Nautilus's ``Cache`` (Invariant N — never tracked here)."""
            deps = self._deps
            assert deps is not None
            out: list[PositionSnapshot] = []
            for iid_str, instrument in self._instruments.items():
                positions = self.cache.positions_open(instrument_id=instrument.id)
                if not positions:
                    continue
                pos = positions[0]
                symbol = deps.instrument_to_symbol[iid_str]
                signed = float(pos.signed_qty)
                if abs(signed) <= 0.0:
                    continue
                side = 1 if signed > 0.0 else -1
                entry_idx, invalidation, generation = self._recover_intent(iid_str, instrument, side)
                out.append(PositionSnapshot(
                    instrument=symbol, side=side, qty=abs(signed),
                    entry_price=float(pos.avg_px_open or 0.0),
                    entry_idx=entry_idx, invalidation=invalidation, generation=generation,
                    venue_stops=self._resting_stops(instrument, abs(signed)),
                ))
            return out

        def _recover_intent(self, iid_str, instrument, side) -> tuple[int, float, int]:
            """Recover (entry_idx, invalidation, generation) from the resting stop's id + trigger.

            The intent id (and hence the stop id) is derived from observed state (Inv Q-STOP), so a
            restart recovers the same intent without an in-process counter: the entry_idx is parsed
            from the resting stop's client-order-id, the invalidation from its trigger price, the
            generation from the id's ``G`` field. Falls back to a fresh bar-indexed intent if no
            stop is resting yet (the reconciler will then place one).
            """
            stops = self._resting_stops(instrument, 0.0)
            if stops:
                s = stops[0]
                # id form: "{symbol}:{entry_idx}|STOP|G{gen}|{hash}"
                head = s.client_order_id.split("|")
                try:
                    entry_idx = int(head[0].split(":")[-1])
                    generation = int(head[2][1:]) if len(head) > 2 and head[2].startswith("G") else 0
                except (ValueError, IndexError):
                    entry_idx, generation = self.bars_seen, 0
                return entry_idx, float(s.trigger), generation
            return self.bars_seen, 0.0, 0

        def _resting_stop_order(self, instrument, intent_id: str):
            """The live (open, non-closed) reduce-only STOP order resting for ``intent_id``, if any.

            A native-modify amend reuses this resting order (Nautilus keeps its id) instead of
            submitting a fresh-id stop, so there is never more than one protective stop per intent
            (Inv S). The id form is ``{intent_id}|STOP|G{gen}|{hash}``; we match the intent-id prefix.
            """
            prefix = f"{intent_id}|STOP|"
            for order in self.cache.orders_open(instrument_id=instrument.id):
                if not getattr(order, "is_reduce_only", False):
                    continue
                if getattr(order, "trigger_price", None) is None:
                    continue
                if order.client_order_id.value.startswith(prefix):
                    return order
            return None

        def _resting_stops(self, instrument, qty: float) -> tuple[VenueStop, ...]:
            """The resting reduce-only STOP orders for this instrument, as :class:`VenueStop`s (§9)."""
            out: list[VenueStop] = []
            for order in self.cache.orders_open(instrument_id=instrument.id):
                if not getattr(order, "is_reduce_only", False):
                    continue
                trigger = getattr(order, "trigger_price", None)
                if trigger is None:
                    continue
                out.append(VenueStop(
                    client_order_id=order.client_order_id.value,
                    trigger=float(trigger),
                    qty=float(order.quantity),
                    reduce_only=True,
                    kind="STOP_MARKET",
                ))
            return tuple(out)

        def _account(self):
            for iid_str in self._instruments:
                venue = self._instruments[iid_str].id.venue
                acct = self.portfolio.account(venue)
                if acct is not None:
                    return acct
            return None

        def _equity(self) -> float:
            if self._equity_override is not None:
                return self._equity_override
            acct = self._account()
            if acct is None:
                return 0.0
            total = acct.balance_total()
            return float(total.as_double()) if total is not None else 0.0

        def _free_margin(self) -> float:
            acct = self._account()
            if acct is None:
                return 0.0
            free = acct.balance_free()
            return float(free.as_double()) if free is not None else 0.0

        def _peak_equity(self) -> float:
            # The DD backstop reads a running-max peak over the SAME equity series the governor uses
            # (§7.1a / ARCH §5.12). We maintain that running max here off the live equity read so the
            # §7.1 DD-kill can actually fire on a peak-to-trough drawdown (a no-drawdown floor would
            # make 7.1 dead in the live path — part of the wiring bug).
            eq = self._equity()
            if eq > self._equity_peak:
                self._equity_peak = eq
            return self._equity_peak

        def set_equity_override(self, equity: float | None) -> None:
            """Override the live equity read (the DD-kill §7.1 series) — used by an external equity
            reconciler when Nautilus's per-venue balance is not the authoritative book equity, and by
            the end-to-end OPS test to drive a deterministic peak-to-trough drawdown past ``DD_kill``.
            The running-max peak still advances off this series so 45% > 43% holds by construction."""
            self._equity_override = None if equity is None else float(equity)

    return _ConvexityStrategy


# A module-level alias so callers can ``from ... import ConvexityStrategy``. On the VM (nautilus
# installed) this is the live ``Strategy`` subclass; in the pure-Python unit venv it falls back to
# the composer (nautilus is never imported there — Invariant N).
try:  # pragma: no cover - VM-only branch
    ConvexityStrategy = make_strategy_class()
except Exception:
    ConvexityStrategy = ConvexityComposer  # type: ignore[assignment,misc]
