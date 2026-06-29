"""Severity router + deterministic dedup + the ARL-budgeted paging cadence over independent SOURCES.

Spec: OPS §8 (alerting). Severity is trigger-defined (§8.1) and routed, not re-decided per alarm. Dedup
is deterministic (§8.2): an alarm fingerprint is ``(stream, detector, severity)``; the first crossing
pages, subsequent crossings within the dedup window are COALESCED (counted, logged, not re-paged); a
*new* fingerprint or an *escalation* (higher severity) always pages. The paging cadence is the ARL math
(§8.3/§6.6): the panel false-page rate is ``Σ_d(1/ARL0,d)`` summed over the **independent ALARM
SOURCES** — single detectors at their own ARL0, each 2-of-2 AND-pair at its JOINT ARL0, the edge stream
at its §6.5a ARL0-equivalent — NOT the raw stream count nor the raw detector count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from convexity.ops.types import AlarmRecord, Severity


@dataclass(frozen=True)
class _Fingerprint:
    stream: str
    detector: str | None
    severity: Severity


@dataclass
class AlarmRouter:
    """Deterministic severity router + dedup (§8.1/§8.2).

    ``route(alarms)`` returns the subset that should PAGE this bar (a held breach pages once + a
    coalesced count, not every bar); the rest are coalesced (counted, logged). A higher-severity
    fingerprint always pages (escalation, §8.2). This is alarm-state hysteresis, NOT threshold
    hysteresis — the limit never moves (P2/LIM), only the *notification* is suppressed.
    """

    dedup_window: int
    _seen: dict[_Fingerprint, int] = field(default_factory=dict)   # fingerprint -> last page sample
    _coalesced: dict[_Fingerprint, int] = field(default_factory=dict)
    _max_sev: dict[tuple[str, str | None], Severity] = field(default_factory=dict)
    _now: int = 0

    def tick(self) -> None:
        """Advance the internal sample clock one bar (the dedup window is counted in samples)."""
        self._now += 1

    def route(self, alarms: Iterable[AlarmRecord]) -> list[AlarmRecord]:
        """Return the alarms to PAGE this bar; coalesce the rest into their fingerprint's count."""
        to_page: list[AlarmRecord] = []
        for a in alarms:
            fp = _Fingerprint(a.stream, a.detector, a.severity)
            key = (a.stream, a.detector)
            prev_max = self._max_sev.get(key)
            escalation = prev_max is None or a.severity > prev_max
            last = self._seen.get(fp)
            fresh = last is None or (self._now - last) >= self.dedup_window
            if escalation or fresh:
                count = self._coalesced.pop(fp, 0)
                self._seen[fp] = self._now
                self._max_sev[key] = max(prev_max or a.severity, a.severity)
                to_page.append(
                    AlarmRecord(
                        stream=a.stream, detector=a.detector, trigger=a.trigger,
                        severity=a.severity, statistic=a.statistic, limit=a.limit,
                        sample_idx=a.sample_idx, in_control=a.in_control, message=a.message,
                        coalesced_count=count,
                    )
                )
            else:
                self._coalesced[fp] = self._coalesced.get(fp, 0) + 1
        return to_page

    def clear_fingerprint(self, stream: str, detector: str | None, severity: Severity) -> None:
        """Forget a fingerprint on a gated re-arm (§8.4) so it pages afresh if it breaks again."""
        fp = _Fingerprint(stream, detector, severity)
        self._seen.pop(fp, None)
        self._coalesced.pop(fp, None)
        self._max_sev.pop((stream, detector), None)


def panel_false_page_rate(source_arl0s: Iterable[float]) -> float:
    """Panel false-page rate ``Σ_d(1/ARL0,d)`` over the INDEPENDENT alarm sources (§6.6/§8.3).

    ``source_arl0s`` is the per-source ARL0 of each *independent* source — a single detector at its own
    ARL0, an AND-combined pair counted ONCE at its joint ARL0, the edge stream at its §6.5a ARL0-
    equivalent. NOT the raw stream count nor the raw detector count (the naive ``S/ARL0`` is wrong).
    The expected time between any false page is ``1 / Σ_d(1/ARL0,d)``.
    """
    return sum(1.0 / a for a in source_arl0s if a > 0.0)


def expected_samples_between_pages(source_arl0s: Iterable[float]) -> float:
    """``1 / Σ_d(1/ARL0,d)`` — the expected samples between any false page (§8.3)."""
    rate = panel_false_page_rate(source_arl0s)
    return float("inf") if rate <= 0.0 else 1.0 / rate


__all__ = [
    "AlarmRouter",
    "panel_false_page_rate",
    "expected_samples_between_pages",
]
