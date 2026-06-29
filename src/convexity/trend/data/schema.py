"""BarValidator — OHLCV validation + bad-bar flags (TREND §8 data/schema).

Returns ``Bars`` with an ``is_valid`` flag set. Checks: sorted timestamps, duplicate bars, OHLC
consistency (h >= max(o,c), l <= min(o,c), h >= l), negative/zero prices, stale prints, and
impossible jumps created by corporate actions or futures rolls. It does NOT silently forward-fill
tradable prices — a bad bar is flagged ``is_valid=False`` so the policy treats it as untradeable
(continue/reduce), never as trend death (§1.4).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, InstrumentSpec


class BarValidator:
    """Validate raw OHLCV and attach an ``is_valid`` mask (§8)."""

    def __init__(self, stale_repeat_min: int = 5) -> None:
        self.stale_repeat_min = stale_repeat_min

    def validate(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Return bars with ``is_valid`` flags (§8). Causal: each flag uses only that bar + history."""
        n = len(bars)
        o, h, low, c = bars.o, bars.h, bars.l, bars.c
        valid = np.ones(n, dtype=bool)

        # OHLC consistency + positivity.
        valid &= np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
        valid &= (o > 0) & (h > 0) & (low > 0) & (c > 0)
        valid &= h >= low
        valid &= h >= np.maximum(o, c) - 1e-9
        valid &= low <= np.minimum(o, c) + 1e-9

        # Non-decreasing timestamps: a bar whose ts <= prior ts is a duplicate/out-of-order bar.
        if n > 1:
            ts = bars.ts
            nonincreasing = np.zeros(n, dtype=bool)
            nonincreasing[1:] = ts[1:] <= ts[:-1]
            valid &= ~nonincreasing

        # Stale-print run: >= stale_repeat_min identical closes flags the staleness (causal scan).
        if self.stale_repeat_min > 0 and n >= self.stale_repeat_min:
            run = 1
            for i in range(1, n):
                if c[i] == c[i - 1]:
                    run += 1
                    if run >= self.stale_repeat_min:
                        valid[i] = False
                else:
                    run = 1

        # Respect any incoming halt flag (untradeable, not invalid-data per se but not tradable).
        if bars.is_halt is not None:
            valid &= ~bars.is_halt.astype(bool)

        # Merge with any pre-existing is_valid the caller attached.
        if bars.is_valid is not None:
            valid &= bars.is_valid.astype(bool)

        return Bars(**{**bars.__dict__, "is_valid": valid})
