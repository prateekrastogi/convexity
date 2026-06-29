"""RangeGeometry — strict-pre-bar barriers, range width, compression, invalidation (BREAKOUT §7 / Inv I).

THE strict-pre-bar barrier module. ``upper_barrier[t]`` / ``lower_barrier[t]`` are functions of bars
``[.. t-1]`` ONLY (Invariant I): the inclusive rolling extremum shifted forward exactly one bar,
``barrier[t] = inclusive_extremum[t-1]``. Bar t's own high/low/close are NEVER in the level it must
break — an inclusive max would contain ``h[t]`` and be unbreakable by construction (the single most
common breakout-detector defect, BREAKOUT §5 Inv I / §12.4).

This is distinct from the future-needle test (Invariant B): that checks bars AFTER t; this is the
self-reference defect AT t. The cross test in ``detect/candidates.py`` compares ``c[t]`` against
this strictly-causal level.

The rolling max/min uses an amortized-O(1) monotonic deque (each bar pushed/popped at most once,
BREAKOUT §1 hot-path cost). This module is the BATCH / validation reference array implementation; the
ONLINE hot path mirrors it bar-for-bar in ``online/streaming.MonotonicExtremum`` (whose deque state
lives in the ``online/state.py`` snapshot for batch/live parity, Invariant E). The strict-pre-bar
``compression_pct`` expanding empirical rank here is realised on the hot path as the §1 fixed-bin
streaming histogram (``online/streaming.StreamingHistogram``), so the online update stays O(1)
amortized while this reference stays exact.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.config.asset_class import BarrierFamily
from convexity.breakout.types import BreakoutGeometry, StdSeries


def rolling_max_strict_prebar(h: np.ndarray, window: int) -> np.ndarray:
    """``out[t] = max(h[t-window .. t-1])`` via a monotonic deque (amortized O(1)/bar, Inv I).

    The window ends at ``t-1`` (strictly pre-bar): the inclusive rolling max over ``[.. t-1]``,
    equivalently the inclusive max shifted forward one bar. NaN until at least one prior bar exists.
    """
    n = h.size
    out = np.full(n, np.nan)
    dq: deque[int] = deque()  # holds indices of a decreasing sequence of h
    for t in range(n):
        # the level bar t must break is the max over the window ending at t-1.
        while dq and dq[0] <= t - 1 - window:
            dq.popleft()
        if dq:
            out[t] = h[dq[0]]
        # now push bar t-? No: push index t AFTER reading, so t never sees its own high.
        # We push index (t) representing h[t]; it becomes available to bar t+1.
        while dq and h[dq[-1]] <= h[t]:
            dq.pop()
        dq.append(t)
    return out


def rolling_min_strict_prebar(low: np.ndarray, window: int) -> np.ndarray:
    """``out[t] = min(l[t-window .. t-1])`` via a monotonic deque (amortized O(1)/bar, Inv I)."""
    n = low.size
    out = np.full(n, np.nan)
    dq: deque[int] = deque()  # holds indices of an increasing sequence of l
    for t in range(n):
        while dq and dq[0] <= t - 1 - window:
            dq.popleft()
        if dq:
            out[t] = low[dq[0]]
        while dq and low[dq[-1]] >= low[t]:
            dq.pop()
        dq.append(t)
    return out


class RangeGeometry:
    """Barriers + range geometry + invalidation (BREAKOUT §7). One public method, ``compute``."""

    def __init__(self, *, family: BarrierFamily = "donchian") -> None:
        self.family = family

    def compute(self, bars: Bars, s: StdSeries, windows: tuple[int, ...]) -> BreakoutGeometry:
        """Compute strictly-pre-bar barriers, range width (in sigma), compression pct, invalidation,
        and stop distance for all bars. Does NOT decide whether to enter (§7).

        Uses the SHORTEST window for the actionable barrier and the LONGEST for the invalidation
        range so the invalidation sits at the opposite side of the pre-break range.
        """
        h, low, c = bars.h, bars.l, bars.c
        n = len(bars)
        w_fast = min(windows)
        w_slow = max(windows)

        upper = rolling_max_strict_prebar(h, w_fast)
        lower = rolling_min_strict_prebar(low, w_fast)
        # the slow extrema are retained for range-maturity / context, but the invalidation uses the
        # FAST (entry) range opposite side so the stop is tight (the convex-breakout thesis: small
        # invalidation distance relative to expected expansion, §9).
        upper_slow = rolling_max_strict_prebar(h, w_slow)
        lower_slow = rolling_min_strict_prebar(low, w_slow)
        _ = (upper_slow, lower_slow)  # available for diagnostics; not the invalidation anchor

        # candidate direction: which barrier the close has crossed (raw geometry, the vol-normalized
        # decision is candidates.py's job, Inv J). 0 where inside the range or undefined.
        direction = np.zeros(n, dtype=int)
        up_cross = np.isfinite(upper) & (c > upper)
        dn_cross = np.isfinite(lower) & (c < lower)
        direction[up_cross] = 1
        direction[dn_cross] = -1

        # range width in sigma units (the fast Donchian range scaled by the predictive sigma).
        rng = np.where(np.isfinite(upper) & np.isfinite(lower), upper - lower, np.nan)
        range_width_sigma = np.where(
            np.isfinite(rng), rng / (s.sigma * c + 1e-12), np.nan
        )

        # compression percentile: PAST-ONLY expanding rank of the (negative) range width — a narrow
        # range => high compression pct. Computed causally (no future quantiles, §12.4).
        compression_pct = self._compression_pct(range_width_sigma)

        # invalidation: opposite side of the FAST pre-break range (known at signal time, §9). A long
        # breakout is invalidated back below the range low; a short above the range high.
        invalidation = np.where(direction > 0, lower, np.where(direction < 0, upper, np.nan))
        entry_ref = c.astype(float)  # bar-close convention: ref price known at signal time
        stop_distance = np.abs(entry_ref - invalidation)

        barrier_age = self._barrier_age(upper, lower)

        return BreakoutGeometry(
            ts=bars.ts,
            direction=direction,
            upper_barrier=upper,
            lower_barrier=lower,
            invalidation=invalidation,
            stop_distance=stop_distance,
            range_width_sigma=range_width_sigma,
            compression_pct=compression_pct,
            barrier_age=barrier_age,
            entry_ref_price=entry_ref,
        )

    def _compression_pct(self, width_sigma: np.ndarray) -> np.ndarray:
        n = width_sigma.size
        out = np.zeros(n)
        warm = 30
        for t in range(n):
            if t < warm or not np.isfinite(width_sigma[t]):
                continue
            past = width_sigma[:t]
            past = past[np.isfinite(past)]
            if past.size < 5:
                continue
            # fraction of past widths >= current => high when current range is unusually narrow.
            out[t] = float(np.mean(past >= width_sigma[t]))
        return out

    def _barrier_age(self, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
        """Bars since the active barrier level last changed (a proxy for range maturity)."""
        n = upper.size
        age = np.zeros(n, dtype=int)
        for t in range(1, n):
            same_u = np.isfinite(upper[t]) and np.isfinite(upper[t - 1]) and upper[t] == upper[t - 1]
            same_l = np.isfinite(lower[t]) and np.isfinite(lower[t - 1]) and lower[t] == lower[t - 1]
            age[t] = age[t - 1] + 1 if (same_u and same_l) else 0
        return age
