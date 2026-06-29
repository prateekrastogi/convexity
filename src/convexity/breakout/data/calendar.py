"""Sessionizer — session-open flags + session-aware metadata (BREAKOUT §7 data/calendar / §3).

Attaches an ``is_session_open`` flag array. Supports RTH/ETH equities, exchange-traded futures,
metals sessions, and crypto 24/7 UTC segmentation. Deterministic output (BREAKOUT §7): for 24/7
instruments the "session" is the UTC day, with the open flag on the first bar of each UTC day; the
UTC day/week boundary is the canonical anchor any anchored barrier variant uses (BREAKOUT §3 crypto
anchoring), never an ad-hoc midnight.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from convexity.core.types import Bars, InstrumentSpec

from convexity.breakout.config.resolve import has_session_boundary

_DAY_MS = 24 * 60 * 60 * 1000


class Sessionizer:
    """Attach session-open flags (BREAKOUT §7). One public method, ``apply``."""

    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Return ``Bars`` with a deterministic ``is_session_open`` flag (§7)."""
        if bars.is_session_open is not None:
            return bars  # caller-supplied session marks win (AMBER metadata)

        if has_session_boundary(spec.calendar_id):
            flags = self._session_marks_from_gaps(bars.ts)
        else:
            flags = self._utc_day_marks(bars.ts)  # 24/7 crypto: UTC-day segmentation
        return replace(bars, is_session_open=flags)

    def _utc_day_marks(self, ts: np.ndarray) -> np.ndarray:
        """Mark the first bar of each UTC day (the deterministic 24/7 anchor, §3)."""
        n = ts.size
        marks = np.zeros(n, dtype=bool)
        if n == 0:
            return marks
        day = ts.astype(np.int64) // _DAY_MS
        marks[0] = True
        marks[1:] = day[1:] != day[:-1]
        return marks

    def _session_marks_from_gaps(self, ts: np.ndarray) -> np.ndarray:
        """For session-bearing calendars without explicit marks: a large inter-bar gap opens a new
        session. Deterministic and past-only (uses only the timestamp grid)."""
        n = ts.size
        marks = np.zeros(n, dtype=bool)
        if n == 0:
            return marks
        marks[0] = True
        if n < 3:
            return marks
        dt = np.diff(ts.astype(np.int64))
        step = np.median(dt[dt > 0]) if np.any(dt > 0) else 0
        if step > 0:
            marks[1:] = dt > 4 * step  # an overnight/weekend break opens a session
        return marks
