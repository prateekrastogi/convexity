"""Log returns + gap flags + session-open handling (BREAKOUT §7 standardize/returns).

Close-to-close log returns with the session-open gap handled separately: the first bar of a session
is a gap (overnight/weekend) not a continuous return, so it is flagged and excluded from the rolling
volatility estimate (BREAKOUT §standardize/volatility: "Must handle session-open gaps separately").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.core.types import Bars


@dataclass(frozen=True)
class ReturnSeries:
    """Raw log returns + a gap flag. ``r[0] == 0`` (no return on the first bar)."""

    ts: np.ndarray
    r: np.ndarray  # close-to-close log return
    is_gap: np.ndarray  # True where the return spans a session boundary / missing-bar gap


def log_returns(bars: Bars) -> ReturnSeries:
    """Compute close-to-close log returns with session/gap flags (§7)."""
    c = np.clip(bars.c, 1e-12, None)
    n = len(bars)
    r = np.zeros(n)
    r[1:] = np.diff(np.log(c))

    is_gap = np.zeros(n, dtype=bool)
    if bars.is_session_open is not None:
        is_gap |= bars.is_session_open.astype(bool)
    # also treat a timestamp discontinuity as a gap (missing bars between t-1 and t).
    if n >= 3:
        dt = np.diff(bars.ts.astype(np.int64))
        step = np.median(dt[dt > 0]) if np.any(dt > 0) else 0
        if step > 0:
            gap = np.zeros(n, dtype=bool)
            gap[1:] = dt > 1.5 * step
            is_gap |= gap
    is_gap[0] = True
    return ReturnSeries(ts=bars.ts, r=r, is_gap=is_gap)
