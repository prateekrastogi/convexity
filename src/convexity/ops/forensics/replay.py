"""Reconstruct the trigger chain from the forensic log (§P7, Invariant AUDIT).

Spec: OPS Invariant AUDIT — "any halt is reconstructed from the log alone, including the stream,
detector, statistic, limit, sample index, and in-control parameters." This module replays the
append-only :class:`ForensicLog` into a human-/machine-readable trigger chain for each halt, so a
halt's cause is auditable without re-running the live panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.forensics.log import ForensicLog
from convexity.ops.types import AlarmRecord, KillVerdict


@dataclass(frozen=True)
class HaltReconstruction:
    """The full reconstructed cause of one halt (Invariant AUDIT). Built from the log alone."""

    verdict: KillVerdict
    alarms: tuple[AlarmRecord, ...]

    def summary(self) -> str:
        """A one-line legible cause: the triggers + the binding alarm statistics/limits."""
        parts = [f"HALT triggers={list(self.verdict.triggers)}"]
        for a in self.alarms:
            ic = a.in_control
            ic_str = (f" mu0={ic.mu0} sigma0={ic.sigma0}" if ic is not None else "")
            parts.append(
                f"  [{a.stream}/{a.detector}] {a.trigger}: stat={a.statistic:.4f} "
                f"limit={a.limit:.4f} @sample={a.sample_idx}{ic_str}"
            )
        return "\n".join(parts)


def replay_halts(log: ForensicLog) -> list[HaltReconstruction]:
    """Reconstruct every halt's trigger chain from the log alone (Invariant AUDIT).

    Pairs each HALT verdict with the alarms whose trigger appears in its chain — so the stream,
    detector, statistic, limit, sample index, and in-control parameters that caused the halt are all
    recoverable without the live panel.
    """
    out: list[HaltReconstruction] = []
    for kill in log.kills:
        if kill.state != "halt":
            continue
        chain = set(kill.triggers)
        alarms = tuple(a for a in log.alarms if a.trigger in chain) + tuple(kill.alarms)
        out.append(HaltReconstruction(verdict=kill, alarms=alarms))
    return out


__all__ = ["HaltReconstruction", "replay_halts"]
