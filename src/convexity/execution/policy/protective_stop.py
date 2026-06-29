"""ProtectiveStop — the venue-native reduce-only protective stop (§9 / Invariants R / S / Q-STOP).

The single most important operational guarantee in the module: the stop that protects the trade lives
on the VENUE so it fires while the bot is down (P6). This module BUILDS the place / amend commands;
Nautilus submits them.

  * ``reduce_only = True`` ALWAYS — a stop can only flatten, never flip/add (Invariant R).
  * ``qty`` = the current FILLED position qty (reconciled on every fill event, §8); the venue stop
    covers exactly the open size.
  * ``trigger`` = the invalidation level handed down by upstream (TREND owns WHERE; execution owns
    that it RESTS on the venue).
  * ``kind`` = STOP_MARKET where supported (guaranteed fire, accepts gap slippage) — the default; a
    STOP_LIMIT only where the class preset says STOP_MARKET is unsafe (§9).
  * ``client_order_id`` uses the DEDICATED stop scheme ``hash(intent_id, "STOP", amend_generation)``
    (Invariant Q-STOP), DISJOINT from the entry/exit slice scheme. ``amend_generation`` starts at 0
    and is DERIVABLE from observed state (the current invalidation generation in WorkingState), so it
    survives restart.

``amend`` increments ``amend_generation`` so the replacement gets a FRESH, venue-accepted id, and
places-before-cancel where the venue lacks modify (NEVER a no-stop window). A crash between the place
and the cancel can leave TWO stops; reconcile() (nautilus/reconcile.py) collapses to exactly one.
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.policy.order_issuer import stop_client_order_id
from convexity.execution.types import OrderCommand, OrderKind, PositionState

__all__ = ["ProtectiveStop"]


class ProtectiveStop:
    """Builds the venue-native reduce-only stop place / amend commands (§9 / Invariants R/S/Q-STOP)."""

    def place(
        self,
        intent_id: str,
        position: PositionState,
        invalidation: float,
        cfg: ResolvedConfig,
        *,
        amend_generation: int = 0,
    ) -> OrderCommand:
        """Place a VENUE-NATIVE reduce-only protective stop the instant a position is (partially) open.

        ``qty`` = the current filled position qty (the stop covers exactly the open size); ``side`` is
        the CLOSING side (opposite the position) so reduce_only is consistent. The id is the dedicated
        stop scheme (Invariant Q-STOP). STOP_MARKET unless the class preset marks it unsafe (§9).
        """
        kind: OrderKind = "STOP_MARKET" if cfg.preset.stop_market_safe else "STOP_LIMIT"
        closing_side = -position.side  # a stop flattens: opposite the open side (reduce_only)
        coid = stop_client_order_id(intent_id, amend_generation)
        return OrderCommand(
            intent_id=intent_id,
            instrument=position.instrument,
            side=closing_side,  # type: ignore[arg-type]
            kind=kind,
            qty=abs(position.qty),
            price=invalidation,  # trigger price (§9)
            time_in_force="GTC",
            post_only=False,
            reduce_only=True,  # ALWAYS (Invariant R)
            client_order_id=coid,
            action="submit",
            trigger_reference=cfg.venue.stop_trigger_reference,
            reason_codes=("protective_stop_place", f"gen:{amend_generation}"),
        )

    def amend(
        self,
        intent_id: str,
        position: PositionState,
        new_invalidation: float,
        cfg: ResolvedConfig,
        *,
        amend_generation: int,
    ) -> tuple[OrderCommand, ...]:
        """Modify the resting venue stop when TREND moves the invalidation (§9.amend).

        Each amend INCREMENTS ``amend_generation`` so the replacement gets a FRESH, venue-accepted id
        (reusing the same id would be deduped and re-open the no-stop window). Where the venue lacks
        native modify, this is place-before-cancel: the NEW stop is emitted before the cancel of the
        old, so there is NEVER a window with no resting stop. Returns the command(s) in
        place-before-cancel order.
        """
        new_gen = amend_generation + 1
        new_stop = self.place(intent_id, position, new_invalidation, cfg, amend_generation=new_gen)

        if cfg.venue.supports_native_modify:
            # Native modify carries the SAME id forward conceptually but we still version the
            # generation for the reconciler's current-generation check; emit one modify command.
            modify = OrderCommand(
                intent_id=intent_id,
                instrument=position.instrument,
                side=new_stop.side,
                kind=new_stop.kind,
                qty=abs(position.qty),
                price=new_invalidation,
                time_in_force="GTC",
                post_only=False,
                reduce_only=True,
                client_order_id=stop_client_order_id(intent_id, new_gen),
                action="modify",
                trigger_reference=cfg.venue.stop_trigger_reference,
                reason_codes=("protective_stop_amend_modify", f"gen:{new_gen}"),
            )
            return (modify,)

        # No native modify: place the new stop FIRST, then cancel the prior generation (Inv S — never
        # a no-stop window). The reconciler collapses any transient two-stop state.
        cancel_old = OrderCommand(
            intent_id=intent_id,
            instrument=position.instrument,
            side=new_stop.side,
            kind=new_stop.kind,
            qty=abs(position.qty),
            price=0.0,
            time_in_force="GTC",
            post_only=False,
            reduce_only=True,
            client_order_id=stop_client_order_id(intent_id, amend_generation),  # the PRIOR gen
            action="cancel",
            reason_codes=("protective_stop_amend_cancel_prior", f"gen:{amend_generation}"),
        )
        return (new_stop, cancel_old)
