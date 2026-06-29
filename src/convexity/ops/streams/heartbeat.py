"""Nautilus heartbeat / feed-staleness / clock-skew → the ``heartbeat`` binary stream (§5.6, P6).

Spec: OPS §5.6, P6. Binary triggers (§7.6): ``N_hb`` consecutive missed heartbeats, feed age beyond
the staleness window (session-aware via the AMBER calendar so a market close is not read as an
outage), or clock-skew beyond tolerance (a skewed clock corrupts every funding-stamp and bar-timestamp
downstream). These FAIL TOWARD SAFE (P6): a dark feed is treated as a break, not "in control by
silence". This adapter classifies the binary condition; the brake maps it to DEGRADE (stale feed) or
HALT (heartbeat loss / clock-skew) per §7.6/§9.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.types import StreamObs

# Heartbeat condition codes carried in ``StreamObs.meta['code']`` so the trigger layer can branch
# without re-classifying. OK==0 is in control; the rest fail toward SAFE.
HB_OK = 0.0
HB_MISS = 1.0          # a missed heartbeat (accumulated against N_hb_miss → HALT at the count)
HB_STALE = 2.0         # feed age beyond the staleness window → DEGRADE (session-aware)
HB_CLOCK_SKEW = 3.0    # clock-skew beyond tolerance → HALT


@dataclass(frozen=True)
class HeartbeatFact:
    """One heartbeat/feed/clock observation from Nautilus (§5.6). ``in_session`` is the AMBER calendar
    flag — a gap during a closed session is NOT an outage (P6 session-aware)."""

    heartbeat_received: bool
    feed_age_ms: int
    clock_skew_ms: int
    in_session: bool = True


class HeartbeatStream:
    """Causal adapter heartbeat/feed/clock → ``heartbeat`` binary :class:`StreamObs` (§5.6)."""

    name = "heartbeat"

    def __init__(self, staleness_window_ms: int, clock_skew_tol_ms: int):
        self.staleness_window_ms = int(staleness_window_ms)
        self.clock_skew_tol_ms = int(clock_skew_tol_ms)
        self._idx = 0

    def observe(self, fact: HeartbeatFact, ts: int = 0) -> StreamObs:
        """Classify the binary condition (§5.6/§7.6), fail-toward-SAFE (P6).

        Clock-skew is checked first (the most corrupting), then a missed heartbeat, then feed
        staleness — but only when in session (a closed-session gap is in control, P6).
        """
        self._idx += 1
        code = HB_OK
        if abs(fact.clock_skew_ms) > self.clock_skew_tol_ms:
            code = HB_CLOCK_SKEW
        elif not fact.heartbeat_received:
            code = HB_MISS
        elif fact.in_session and fact.feed_age_ms > self.staleness_window_ms:
            code = HB_STALE
        return StreamObs(
            stream="heartbeat",
            sample_idx=self._idx,
            value=0.0 if code == HB_OK else 1.0,
            ts=ts,
            binary=True,
            meta={
                "code": code,
                "feed_age_ms": float(fact.feed_age_ms),
                "clock_skew_ms": float(fact.clock_skew_ms),
                "in_session": float(fact.in_session),
            },
        )


__all__ = ["HeartbeatFact", "HeartbeatStream", "HB_OK", "HB_MISS", "HB_STALE", "HB_CLOCK_SKEW"]
