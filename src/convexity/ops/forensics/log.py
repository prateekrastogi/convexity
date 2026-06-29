"""Append-only forensic audit log (every alarm/kill, replayable) (§P7, Invariant AUDIT).

Spec: OPS P7, §13 (provenance), Invariant AUDIT. Every alarm carries the stream, the detector, the
statistic value, the limit it crossed, the sample index, and the in-control parameters it was tested
against; every kill carries the full trigger chain. The log is APPEND-ONLY and is the input to
COST.md's post-hoc TCA and SYSTEM's next overnight re-validation. A halt whose cause cannot be
replayed from the log is an un-auditable halt — forbidden.

This is a pure in-memory append-only store (the live wiring serializes it to disk via the adapter); it
holds the typed :class:`AlarmRecord` / :class:`KillVerdict` records so ``replay.py`` can reconstruct
the full trigger chain from the log alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.ops.types import AlarmRecord, KillVerdict


@dataclass
class ForensicLog:
    """Append-only log of alarms + kills (Invariant AUDIT). Never mutated, only appended."""

    alarms: list[AlarmRecord] = field(default_factory=list)
    kills: list[KillVerdict] = field(default_factory=list)

    def append_alarm(self, alarm: AlarmRecord) -> None:
        """Append one alarm record (append-only — Invariant AUDIT)."""
        self.alarms.append(alarm)

    def append_kill(self, verdict: KillVerdict) -> None:
        """Append one kill verdict with its full trigger chain (append-only — Invariant AUDIT)."""
        self.kills.append(verdict)

    def halts(self) -> list[KillVerdict]:
        """Every HALT verdict in the log (the input to a replay of the halt cause)."""
        return [k for k in self.kills if k.state == "halt"]


__all__ = ["ForensicLog"]
