"""Issue a flatten via EXECUTION's venue reduce-only stops (P6 of EXECUTION) — import-isolated.

Spec: OPS §9.1, §nautilus/flatten (P6 EXEC). OPS emits a flatten DIRECTIVE consumed asynchronously by
the live loop (P1) — it does not place orders itself. This module is the thin bridge that turns the
kill machine's flatten directive into the cancel-entries-then-reduce-only-flatten command sequence the
live loop hands to EXECUTION; the actual venue order is EXECUTION's venue-resident reduce-only stop,
already resting the moment a position opens (so even a position born in the race window has the venue
stop as its floor, §9.1b).

PURE-PYTHON, NautilusTrader-free: this file does NOT import nautilus (import-isolated per the build
directive). The strategy/ adapter on the VM wires the resulting :class:`FlattenCommand` to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlattenCommand:
    """The race-free flatten command sequence (§9.1b). ``cancel_entries_first`` is always True so the
    live loop cancels working ENTRY orders before issuing the reduce-only flatten."""

    cancel_entries_first: bool
    reduce_only: bool
    reason: str

    @classmethod
    def make(cls, reason: str = "ops_halt_flatten") -> "FlattenCommand":
        """The canonical HALT flatten: cancel entries first, then a reduce-only flatten (§9.1b)."""
        return cls(cancel_entries_first=True, reduce_only=True, reason=reason)


def flatten_command_for(verdict_flatten: bool, reason: str = "ops_halt_flatten") -> FlattenCommand | None:
    """Build the :class:`FlattenCommand` from a kill verdict's flatten flag, or ``None`` when no flatten
    is directed this bar (the §9.1a latch already ensures it is issued exactly once upstream)."""
    if not verdict_flatten:
        return None
    return FlattenCommand.make(reason)


__all__ = ["FlattenCommand", "flatten_command_for"]
