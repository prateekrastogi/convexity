"""BarValidator — point-in-time bar schema validation (BREAKOUT §7 data/schema).

Returns cleaned ``Bars`` with an ``is_valid`` flag array. Checks sorted timestamps, nonnegative
prices/volume, high-low-open-close consistency, duplicate bars, missing bars, and stale bars. It
does NOT forward-fill prices silently — suspect bars are FLAGGED (``is_valid=False``), not patched.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from convexity.core.types import Bars, InstrumentSpec


class BarValidator:
    """Validate + flag a bar panel (BREAKOUT §7). One public method, ``validate``."""

    def __init__(self, *, stale_run: int = 5) -> None:
        # a price-flat run >= stale_run bars is flagged stale (no real trading).
        self.stale_run = stale_run

    def validate(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Return cleaned ``Bars`` with ``is_valid`` flags (never forward-filling silently)."""
        n = len(bars)
        valid = np.ones(n, dtype=bool)
        ts, o, h, low, c = bars.ts, bars.o, bars.h, bars.l, bars.c

        # sorted, strictly-increasing timestamps (duplicates flagged on the later bar).
        dup = np.zeros(n, dtype=bool)
        dup[1:] = ts[1:] <= ts[:-1]
        valid &= ~dup

        # nonnegative prices; finite.
        finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
        nonneg = (o > 0) & (h > 0) & (low > 0) & (c > 0)
        valid &= finite & nonneg

        # OHLC consistency: h >= max(o,c,l) and l <= min(o,c,h).
        hi_ok = (h >= o) & (h >= c) & (h >= low)
        lo_ok = (low <= o) & (low <= c) & (low <= h)
        valid &= hi_ok & lo_ok

        # volume nonnegative if present.
        if bars.v is not None:
            valid &= np.isfinite(bars.v) & (bars.v >= 0)

        # stale-price runs: a flat close held for >= stale_run bars.
        valid &= ~self._stale_mask(c)

        # missing bars: a timestamp gap > median spacing flags the later bar (data discontinuity).
        valid &= ~self._missing_mask(ts)

        # preserve any caller-supplied invalid flags (AND, never overwrite to True).
        if bars.is_valid is not None:
            valid &= bars.is_valid.astype(bool)

        return replace(bars, is_valid=valid)

    def _stale_mask(self, c: np.ndarray) -> np.ndarray:
        n = c.size
        mask = np.zeros(n, dtype=bool)
        run = 1
        for t in range(1, n):
            if c[t] == c[t - 1]:
                run += 1
            else:
                run = 1
            if run >= self.stale_run:
                mask[t] = True
        return mask

    def _missing_mask(self, ts: np.ndarray) -> np.ndarray:
        n = ts.size
        mask = np.zeros(n, dtype=bool)
        if n < 3:
            return mask
        dt = np.diff(ts.astype(np.int64))
        step = np.median(dt[dt > 0]) if np.any(dt > 0) else 0
        if step <= 0:
            return mask
        gap = np.zeros(n, dtype=bool)
        gap[1:] = dt > 1.5 * step
        return gap
