"""Restart RECONCILIATION (ARCHITECTURE §8.4 b / EXECUTION Inv S / Q-STOP).

On restart, the :class:`StopReconciler` rebuilds the venue protective stops by projecting the
observed (position, resting-stop) set onto the ``STOP_OK`` predicate. The projection is computed
PURELY from observed state — no in-process counter, no memory of prior runs — so re-running it on an
already-``STOP_OK`` book emits an EMPTY plan. That empty-second-run is the PROOF (not the assertion)
of idempotency.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from convexity.execution.policy.order_issuer import stop_client_order_id  # noqa: E402
from convexity.execution.types import VenueStop  # noqa: E402

from .conftest import build_composer, exec_cfg, open_position_view  # noqa: E402

pytestmark = pytest.mark.nautilus


def test_restart_with_no_stop_places_one():
    """A position whose venue stop vanished while the bot was down gets one placed NOW (gap floor)."""
    composer = build_composer()
    view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=0)
    cmds = composer.reconcile_on_start({view.instrument: view}, exec_cfg())
    assert len(cmds) == 1
    (place,) = cmds
    assert place.reduce_only
    assert place.kind in ("STOP_MARKET", "STOP_LIMIT")
    assert place.price == 29_800.0
    assert place.qty == 2.0


def test_reconcile_is_idempotent_when_stop_ok():
    """A book already at STOP_OK reconciles to an EMPTY plan (idempotency by construction)."""
    composer = build_composer()
    view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=0)
    coid = stop_client_order_id(view.intent_id, 0)
    resting = VenueStop(client_order_id=coid, trigger=29_800.0, qty=2.0, reduce_only=True)
    view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=0,
                              venue_stops=(resting,))
    cmds = composer.reconcile_on_start({view.instrument: view}, exec_cfg())
    assert cmds == ()


def test_reconcile_collapses_duplicate_stops_to_one():
    """A place-before-cancel crash can leave TWO stops; reconcile collapses to exactly one current."""
    composer = build_composer()
    intent_view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=1)
    current = VenueStop(client_order_id=stop_client_order_id(intent_view.intent_id, 1),
                        trigger=29_800.0, qty=2.0, reduce_only=True)
    stale = VenueStop(client_order_id=stop_client_order_id(intent_view.intent_id, 0),
                      trigger=29_700.0, qty=2.0, reduce_only=True)
    view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=1,
                              venue_stops=(current, stale))
    cmds = composer.reconcile_on_start({view.instrument: view}, exec_cfg())
    # keep current, cancel the stale generation.
    assert all(c.action == "cancel" for c in cmds)
    cancelled_ids = {c.client_order_id for c in cmds}
    assert stale.client_order_id in cancelled_ids
    assert current.client_order_id not in cancelled_ids


def test_double_reconcile_after_repair_is_noop():
    """Repairing (place a missing stop) then re-observing that stop ⇒ the second run is empty."""
    composer = build_composer()
    view = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=0)
    first = composer.reconcile_on_start({view.instrument: view}, exec_cfg())
    assert len(first) == 1
    placed = first[0]
    repaired = VenueStop(client_order_id=placed.client_order_id, trigger=placed.price,
                         qty=placed.qty, reduce_only=True)
    view2 = open_position_view(side=1, qty=2.0, invalidation=29_800.0, generation=0,
                               venue_stops=(repaired,))
    second = composer.reconcile_on_start({view2.instrument: view2}, exec_cfg())
    assert second == ()
