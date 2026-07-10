"""Shared API pacing (PAPER.md §12.2) — deliberately simple.

A monotonic-clock minimum-interval gate per call class (authed vs public). At this fleet's order
volume (tens of orders per WEEK across three books) a token bucket sized to Binance's documented
weights is overkill; a floor on inter-call spacing keeps even the worst concurrent burst far
inside the limits. Documented simplicity — revisit if order traffic grows by an order of
magnitude. Process-local by design: cross-process bursts are already serialized by the account
lock (lock.py), which is the actual concurrency-control mechanism.
"""

from __future__ import annotations

import threading
import time

from convexity.paper.fleetconfig import (
    AUTHED_CALL_MIN_INTERVAL_S,
    PUBLIC_CALL_MIN_INTERVAL_S,
)

_LOCKS = {"authed": threading.Lock(), "public": threading.Lock()}
_LAST = {"authed": 0.0, "public": 0.0}
_INTERVAL = {"authed": AUTHED_CALL_MIN_INTERVAL_S, "public": PUBLIC_CALL_MIN_INTERVAL_S}


def pace(kind: str = "authed") -> None:
    """Block just long enough to keep calls of this class at least the configured interval apart."""
    with _LOCKS[kind]:
        now = time.monotonic()
        wait = _INTERVAL[kind] - (now - _LAST[kind])
        if wait > 0:
            time.sleep(wait)
        _LAST[kind] = time.monotonic()
