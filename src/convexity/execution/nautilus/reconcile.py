"""StopReconciler — the restart-/gap-proof projection of observed state onto STOP_OK (§9 / Inv S/Q-STOP).

On startup (after Nautilus's own reconciliation completes) and periodically, verify the INVARIANT:
every open position has EXACTLY ONE resting venue reduce-only stop at the current invalidation,
covering the full open qty.

IDEMPOTENCY AS A PROOF, NOT AN ASSERTION. The target invariant is a PURE PREDICATE over OBSERVED
state only — (Nautilus position qty, the SET of venue reduce-only stops for this intent):

    STOP_OK(pos, stops)  <=>  |stops| == 1
                              and stops[0].qty     == pos.qty
                              and stops[0].trigger == current_invalidation
                              and stops[0].id      == hash(intent_id,"STOP",cur_gen)
                              and stops[0].reduce_only

``reconcile`` is DEFINED as the projection that drives observed state onto STOP_OK, computed PURELY
from (position, stops, current_invalidation, cur_gen) — NO hidden in-process counter, no memory of
prior runs. A second run OBSERVES STOP_OK already true and emits an EMPTY plan: re-running is a no-op
BY CONSTRUCTION, which is the PROOF (not the assertion) of idempotency. Plan empty <=> predicate holds.

This module reads OBSERVED state (the position + stop set the Nautilus adapter projects); it imports
no nautilus_trader API itself (the adapter does that), so the STOP_OK projection is unit-testable in
``.venv-test`` without the engine. It returns a plan of :class:`OrderCommand`s; it does not submit.
This method NEVER second-guesses Nautilus's position/account state — it READS it as truth and only
repairs the POLICY layer (the resting stop's existence, price, qty, single-stop invariant).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.policy.order_issuer import stop_client_order_id
from convexity.execution.policy.protective_stop import ProtectiveStop
from convexity.execution.types import OrderCommand, PositionState, VenueStop

__all__ = ["IntendedState", "ReconcilePlan", "stop_ok", "StopReconciler"]

_PRICE_EPS = 1e-9


@dataclass(frozen=True)
class IntendedState:
    """The current intended stop state, all DERIVED from observed Nautilus state (§9). No counter.

    ``current_invalidation`` + ``current_generation`` come from WorkingState (the invalidation
    generation), so the expected stop id is recomputable on every run.
    """

    intent_id: str
    current_invalidation: float
    current_generation: int


@dataclass(frozen=True)
class ReconcilePlan:
    """The order commands that project observed state onto STOP_OK (§9). Empty <=> predicate holds."""

    commands: tuple[OrderCommand, ...]
    position_closed: bool
    reason_codes: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.commands) == 0 and not self.position_closed


def stop_ok(
    intended: IntendedState, position: PositionState, stops: tuple[VenueStop, ...]
) -> bool:
    """The PURE predicate over observed state (§9). True iff exactly one correct resting stop."""
    if abs(position.qty) <= 0.0:
        return len(stops) == 0  # flat: no orphan stops
    if len(stops) != 1:
        return False
    s = stops[0]
    expected_id = stop_client_order_id(intended.intent_id, intended.current_generation)
    return (
        abs(s.qty - abs(position.qty)) <= _PRICE_EPS
        and abs(s.trigger - intended.current_invalidation) <= _PRICE_EPS
        and s.client_order_id == expected_id
        and s.reduce_only
    )


class StopReconciler:
    """The STOP_OK projection (§9 / Invariant S). Idempotent BY CONSTRUCTION (no hidden counter)."""

    def __init__(self) -> None:
        self._stop = ProtectiveStop()

    def reconcile(
        self,
        intended: IntendedState,
        nautilus_orders: tuple[VenueStop, ...],
        nautilus_position: PositionState,
        cfg: ResolvedConfig,
    ) -> ReconcilePlan:
        """Project observed state onto STOP_OK (§9). A second run on a satisfying state -> empty plan."""
        pos = nautilus_position
        stops = tuple(s for s in nautilus_orders if s.reduce_only)

        # Predicate already holds (the idempotency proof): emit an EMPTY plan.
        if stop_ok(intended, pos, stops):
            return ReconcilePlan((), False, ("stop_ok",))

        expected_id = stop_client_order_id(intended.intent_id, intended.current_generation)

        # --- venue stop(s) exist but NO position: orphan(s) (stop fired while down, or pos closed).
        if abs(pos.qty) <= 0.0:
            cancels = tuple(self._cancel(intended.intent_id, s, cfg) for s in stops)
            return ReconcilePlan(cancels, True, ("orphan_stop_position_flat", "reconcile_pnl"))

        # --- position open, NO venue stop: place one IMMEDIATELY (the gap-protection point, Inv S).
        if not stops:
            place = self._stop.place(
                intended.intent_id, pos, intended.current_invalidation, cfg,
                amend_generation=intended.current_generation,
            )
            return ReconcilePlan((place,), False, ("no_stop_place_now",))

        # --- MULTIPLE venue stops (amend place-before-cancel crash, Inv S case d): keep the one
        # matching CURRENT (trigger, qty, current-generation id); cancel every extra (stale gens).
        if len(stops) > 1:
            keepers = [
                s for s in stops
                if s.client_order_id == expected_id
                and abs(s.qty - abs(pos.qty)) <= _PRICE_EPS
                and abs(s.trigger - intended.current_invalidation) <= _PRICE_EPS
            ]
            if keepers:
                keep = keepers[0]
                cancels = tuple(
                    self._cancel(intended.intent_id, s, cfg)
                    for s in stops if s.client_order_id != keep.client_order_id
                )
                return ReconcilePlan(cancels, False, ("multiple_stops_keep_current",))
            # NONE matches current: amend one to current, cancel the rest.
            amend = self._stop.place(
                intended.intent_id, pos, intended.current_invalidation, cfg,
                amend_generation=intended.current_generation,
            )
            cancels = tuple(self._cancel(intended.intent_id, s, cfg) for s in stops)
            return ReconcilePlan((amend, *cancels), False, ("multiple_stops_none_current_amend",))

        # --- exactly one stop but it violates STOP_OK: stale trigger and/or qty mismatch (covers the
        # partial-STOP_MARKET-fire case e — re-cover the ACTUAL remaining position qty). Amend to
        # current invalidation + actual qty + current-generation id.
        amend = self._stop.place(
            intended.intent_id, pos, intended.current_invalidation, cfg,
            amend_generation=intended.current_generation,
        )
        single = stops[0]
        reason = []
        if abs(single.trigger - intended.current_invalidation) > _PRICE_EPS:
            reason.append("stale_invalidation_amend")
        if abs(single.qty - abs(pos.qty)) > _PRICE_EPS:
            reason.append("qty_mismatch_amend")
        if single.client_order_id != expected_id:
            reason.append("stale_generation_amend")
        # If the existing single stop is a prior generation, cancel it after placing the new one.
        cmds: tuple[OrderCommand, ...] = (amend,)
        if single.client_order_id != expected_id:
            cmds = (amend, self._cancel(intended.intent_id, single, cfg))
        return ReconcilePlan(cmds, False, tuple(reason) or ("amend_to_current",))

    @staticmethod
    def _cancel(intent_id: str, stop: VenueStop, cfg: ResolvedConfig) -> OrderCommand:
        """A cancel command for an observed venue stop, by its observed id (§9)."""
        return OrderCommand(
            intent_id=intent_id,
            instrument=cfg.instrument.symbol,
            side=1,  # side is immaterial for a cancel; the id identifies the order
            kind=stop.kind,
            qty=abs(stop.qty),
            price=0.0,
            time_in_force="GTC",
            post_only=False,
            reduce_only=True,
            client_order_id=stop.client_order_id,
            action="cancel",
            reason_codes=("reconcile_cancel",),
        )
