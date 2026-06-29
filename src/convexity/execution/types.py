"""Shared frozen dataclasses for the EXECUTION pillar (EXECUTION §10).

Pure-Python, NautilusTrader-free (composed by ``strategy/convexity_strategy.py``). The cross-pillar
boundary type ``ExecTarget`` lives in :mod:`convexity.core.types` (the load-bearing input contract,
ARCHITECTURE §5.3 — no edge field, Invariant A); everything else here is EXECUTION-internal.

The types map one-to-one onto EXECUTION.md §10:
  * ``MicroState``      — the causal, F_t-measurable market inputs the cost model reads (§10).
  * ``TacticCosts``     — per-tactic expected IS in bps + decomposition (§10).
  * ``WorkingState``    — READ from Nautilus; never owned here (§10 / Invariant N).
  * ``ExecDecision``    — the single log record the engine emits per call (§10).
  * ``OrderCommand``    — the venue order the policy layer builds; submitted by Nautilus (§7/§9).
  * ``RealizedFill`` / ``RealizedStopFill`` — VALIDATION-ONLY, firewalled (§10/§12, Invariant V).

See docs/ARCHITECTURE.md for the package layout and docs/BUILD_PLAN.md Phase 5 for the build order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

# Re-export the cross-pillar aliases + the input contract so EXECUTION code imports them from one
# place. ``ExecTarget`` is defined ONCE in core (ARCHITECTURE §5.3); we never redefine it.
from convexity.core.types import ExecTarget, ExecUrgency, Side, Tactic

__all__ = [
    "ExecTarget",
    "ExecUrgency",
    "Side",
    "Tactic",
    "Urgency",
    "SpreadRung",
    "BindingGate",
    "AbortReason",
    "OrderKind",
    "TimeInForce",
    "MicroState",
    "TacticCosts",
    "WorkingState",
    "OrderCommand",
    "ExecDecision",
    "PositionState",
    "VenueStop",
    "RealizedFill",
    "RealizedStopFill",
]

# EXECUTION's urgency ladder is the 4-level ``ExecUrgency`` (EXECUTION §6). Aliased ``Urgency`` to
# match the spec's local name (§10 prose) without redefining the Literal.
Urgency = ExecUrgency

# Which spread-estimator rung fired (microstate/spread.py ladder, §5 / §11).
SpreadRung = Literal["quoted", "roll", "corwin_schultz", "class_proxy"]

# The gate that bound the tactic choice (§6); "none" when the pure min() decided.
BindingGate = Literal[
    "none", "protective_exit", "halt", "maker_floor", "rate_limit", "min_notional"
]

# Why an action terminated without fully filling (§5.1.2 / §8).
AbortReason = Literal["", "chase_budget_exhausted", "below_min_notional", "halt"]

# Venue order primitives the issuer builds (§7 / §9). The nautilus/adapter translates these.
OrderKind = Literal["LIMIT", "MARKET", "STOP_MARKET", "STOP_LIMIT"]
TimeInForce = Literal["GTC", "IOC", "FOK"]


@dataclass(frozen=True)
class MicroState:
    """The causal, F_t-measurable market microstate the cost model reads (EXECUTION §10).

    Every field is computable at or before the bar on which the engine acts (P7). ``drift_g`` is the
    shared latent continuation-drift / regime input (§5.1.1) that ``p_fill``, ``A``, and ``D`` all
    condition on (Invariant J); it is RE-READ per chase step, never frozen at arrival. ``tick_bps``
    is ``10_000 * tick / mid`` against the CURRENT mid (§5.1). ``fill_prob_curve`` is the causally,
    online-fitted callable (Invariant L).
    """

    mid: float
    half_spread_bps: float
    spread_rung: SpreadRung
    sigma_horizon_bps: float
    volume_proxy: float
    drift_g: float
    tick_bps: float
    # callable p_fill(offset_ticks, horizon_s, g) -> probability in [0,1] (microstate/fill_prob.py).
    fill_prob_curve: Callable[[float, float, float], float]
    funding_aware: bool
    predicted_funding_rate: float
    secs_to_next_funding: float
    halted: bool
    stale: bool
    rate_limit_ok: bool
    # The active fill-prob rung + its calibration-CI half-width feed the maker-floor LCB gate (§6.3).
    fill_prob_rung: Literal["bucket", "global", "class_prior"] = "bucket"
    fill_prob_ci_halfwidth: float = 0.0
    fill_prob_calib_error: float = 0.0
    cold_start: bool = False  # symbol below N_red own passive fills -> maker GATED OFF (§5.5 #2)
    funding_interval_secs: float = 8.0 * 3600.0  # AMBER stamp cadence (Binance USDT-M = 8h)


@dataclass(frozen=True)
class TacticCosts:
    """Per-tactic expected IS in bps + the full decomposition (EXECUTION §10).

    All values are SIGNED bps vs the arrival mid (a cost > 0 is money lost; a saving < 0). The
    spread/impact/fee terms are computed independently (Invariant B); the maker-branch trio
    (``p_fill``, ``A``, ``D``) shares the one ``drift_g`` (Invariant J). The maker-floor gate keys on
    ``p_fill_lcb_at_offset``, NOT the point estimate (§6.3 gate 3).
    """

    by_tactic: Mapping[Tactic, float]
    spread_bps: Mapping[Tactic, float]
    impact_bps: Mapping[Tactic, float]
    adverse_sel_bps: Mapping[Tactic, float]
    delay_bps: Mapping[Tactic, float]
    funding_bps: Mapping[Tactic, float]
    price_improve_bps: float  # PI(delta*) = s - delta*tick_bps captured if maker fills (§5.1)
    p_fill_at_offset: float  # p_fill(delta*, h, g) point estimate — the high-variance leg (§5.5)
    p_fill_lcb_at_offset: float  # LOWER-confidence bound; the maker-floor gate keys on THIS (§6.3)
    p_abort_if_nofill: float  # entry no-fill abort mass (§5.1.2); 0 for exits
    drift_g: float  # the shared regime driver (Invariant J)
    chosen_offset_ticks: float


@dataclass(frozen=True)
class WorkingState:
    """The working order/position snapshot READ from Nautilus — not owned here (EXECUTION §10 / Inv N).

    ``worst_entry`` is the hard chase bound (§5.3). ``invalidation_generation`` increments on each
    invalidation change and DERIVES the stop's ``amend_generation`` -> stop client_order_id, so it
    survives a restart (Invariant Q-STOP); it is read from observed state, never an in-process
    counter.
    """

    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    working_orders: tuple = ()
    chase_step: int = 0
    worst_entry: float = 0.0
    invalidation_generation: int = 0


@dataclass(frozen=True)
class OrderCommand:
    """A venue order the policy layer BUILDS; Nautilus submits it (EXECUTION §7 / §9).

    ``client_order_id`` is deterministic (Invariant Q for slices, Q-STOP for stops). ``reduce_only``
    is True on every protective order (Invariant R). ``action`` distinguishes a fresh submit from a
    cancel/replace (the chase ladder and the amend path).
    """

    intent_id: str
    instrument: str
    side: Side
    kind: OrderKind
    qty: float
    price: float  # 0.0 / ignored for MARKET; trigger price for STOP_* (§9)
    time_in_force: TimeInForce
    post_only: bool
    reduce_only: bool
    client_order_id: str
    action: Literal["submit", "cancel", "modify"] = "submit"
    trigger_reference: Literal["mark", "index", "last"] = "mark"  # §9 trigger ref (AMBER)
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecDecision:
    """The single log record the engine emits per call (EXECUTION §10).

    Commands go to Nautilus (``commands``); this record is the decomposable audit. ``binding_gate``
    names which §6 gate bound the choice ("none" when the pure min() decided). ``cost_decomposition``
    carries the spread/impact/adverse/delay/funding lines for the chosen tactic (Invariant H — log
    only, never alters the choice). Deterministic given (target, micro, working, cfg).
    """

    intent_id: str
    instrument: str
    tactic: Tactic
    price: float
    qty: float
    reduce_only: bool
    client_order_id: str
    expected_cost_bps: float
    cost_decomposition: Mapping[str, float]
    binding_gate: BindingGate
    chase_step: int
    aborted: bool
    abort_reason: AbortReason
    reason_codes: tuple[str, ...]
    engine_version: str
    commands: tuple[OrderCommand, ...] = ()


# --------------------------------------------------------------------------- protective stop I/O


@dataclass(frozen=True)
class PositionState:
    """The open position the protective stop covers — READ from Nautilus (EXECUTION §9 / Inv N)."""

    instrument: str
    side: Side
    qty: float  # current FILLED qty (>= 0 magnitude); the stop covers exactly this (§8/§9)
    entry_price: float = 0.0


@dataclass(frozen=True)
class VenueStop:
    """An observed venue-resident reduce-only stop, as read back for reconciliation (§9).

    The reconciler treats the SET of these as observed state and projects onto ``STOP_OK`` (§9).
    """

    client_order_id: str
    trigger: float
    qty: float
    reduce_only: bool
    kind: OrderKind = "STOP_MARKET"


# --------------------------------------------------------------------------- VALIDATION-ONLY (Inv V)


@dataclass(frozen=True)
class RealizedFill:
    """VALIDATION-ONLY realized fill outcome — uses future bars; firewalled (EXECUTION §10 / Inv V)."""

    intent_id: str
    realized_avg_price: float
    realized_is_bps: float  # vs arrival_mid (§5.2)
    realized_spread_bps: float
    realized_impact_bps: float
    realized_adverse_sel_bps: float
    realized_delay_bps: float
    realized_funding_bps: float
    realized_aborted: bool
    filled_qty: float
    maker_filled_qty: float
    taker_filled_qty: float


@dataclass(frozen=True)
class RealizedStopFill:
    """VALIDATION-ONLY protective-stop realized cost (EXECUTION §10 / §12, Invariant V).

    ``gap_slippage_bps`` is measured SEPARATELY from ``realized_is_bps``: it is the cost a STOP_MARKET
    pays when price gaps THROUGH the trigger, and it feeds LEVERAGE's gap-survival ceiling, not the
    entry/exit IS (§12).
    """

    intent_id: str
    trigger_price: float
    realized_fill_price: float
    gap_slippage_bps: float
    gapped_through: bool
    bot_was_down: bool
