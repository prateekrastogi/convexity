"""The graceful-degradation policy (§9) — deterministic, leaning on venue-resident reduce-only stops.

Spec: OPS §9. On partial failure the response is a DETERMINISTIC policy, not discretion, and it leans
on EXECUTION's venue-resident reduce-only stops (P6 of EXECUTION) so the risk control survives an
OPS/bot outage. The unifying rule (P6): every partial failure resolves to the state that cannot lose
more than the venue stop allows — never the permissive one (fail toward SAFE).

This module maps each named §9 failure to its directive (state + flatten/cancel flags) as a pure
lookup; the kill machine consumes it. The venue reduce-only stop is the floor under all of this; the
degradation policy is about *not making it worse* while the bot is impaired, never substituting
in-process logic for the venue stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from convexity.core.types import KillState

# The §9 failure taxonomy.
FailureKind = Literal[
    "feed_gap_single_venue",     # one venue stale
    "heartbeat_loss",            # bot blind
    "clock_skew",                # every timestamp suspect
    "venue_down",                # orders rejected on a venue
    "reconciliation_mismatch",   # book disagrees with venue
    "exit_cannot_fill",          # protective action cannot fill
    "detector_self_diagnostic",  # a detector's own diagnostics failed
    "flatten_rejected",          # the flatten DIRECTIVE itself was rejected/partial (§9.1c)
]


@dataclass(frozen=True)
class SafeStateDirective:
    """The deterministic directive for a §9 failure. ``stop_new_entries`` holds existing positions;
    ``flatten`` closes to venue stops; ``cancel_working`` cancels working entry orders."""

    state: KillState
    stop_new_entries: bool
    flatten: bool
    cancel_working: bool
    page: bool
    message: str


# The §9 failure → safe-state directive table (deterministic, not discretion).
_POLICY: dict[FailureKind, SafeStateDirective] = {
    "feed_gap_single_venue": SafeStateDirective(
        "degrade", stop_new_entries=True, flatten=False, cancel_working=False, page=True,
        message="single venue stale: stop new entries on its instruments, HOLD existing (venue "
                "stops live), page (§9).",
    ),
    "heartbeat_loss": SafeStateDirective(
        "halt", stop_new_entries=True, flatten=False, cancel_working=True, page=True,
        message="bot blind: HALT, venue reduce-only stops already resting, cancel working entries, "
                "do NOT place new orders blind; reconcile before re-arming (§9/§8.4).",
    ),
    "clock_skew": SafeStateDirective(
        "halt", stop_new_entries=True, flatten=False, cancel_working=True, page=True,
        message="clock-skew over tolerance: every timestamp suspect, flatten is unsafe to compute "
                "too — HOLD to venue stops and page for manual intervention (§9).",
    ),
    "venue_down": SafeStateDirective(
        "degrade", stop_new_entries=True, flatten=False, cancel_working=False, page=True,
        message="venue down (orders rejected): HOLD to venue stops there, stop new entries there, "
                "continue other venues, page (§9).",
    ),
    "reconciliation_mismatch": SafeStateDirective(
        "halt", stop_new_entries=True, flatten=True, cancel_working=True, page=True,
        message="reconciliation mismatch: HALT + flatten to the KNOWN venue state (venue is ground "
                "truth), page; does not re-derive recon (§9/P5).",
    ),
    "exit_cannot_fill": SafeStateDirective(
        "halt", stop_new_entries=True, flatten=True, cancel_working=True, page=True,
        message="exit cannot fill on a protective action: escalate to MARKET via EXECUTION (exits "
                "never abort), venue reduce-only stop is the backstop, page HALT (§9).",
    ),
    "detector_self_diagnostic": SafeStateDirective(
        "degrade", stop_new_entries=True, flatten=False, cancel_working=False, page=True,
        message="detector self-diagnostic failed: treat that stream as OUT of control (P6), degrade "
                "on its trigger class, page (§9).",
    ),
    "flatten_rejected": SafeStateDirective(
        "halt", stop_new_entries=True, flatten=True, cancel_working=True, page=True,
        message="flatten order rejected/partial: venue reduce-only stop remains the floor, stay "
                "LATCHED HALT, re-attempt flatten on the RED retry cadence, page HALT-unconfirmed; "
                "NEVER treat an un-confirmed flatten as 'flat' (§9.1c).",
    ),
}


def safe_state_for(kind: FailureKind) -> SafeStateDirective:
    """The deterministic §9 directive for a partial failure (Invariant SAFE — fail toward SAFE)."""
    return _POLICY[kind]


__all__ = ["FailureKind", "SafeStateDirective", "safe_state_for"]
