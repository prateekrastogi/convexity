"""Futures roll/stitch metadata hooks (BREAKOUT §7 data/rolls / §3 commodity-futures).

Marks roll-window bars and back-adjusts the roll price step so a stitched contract does NOT fabricate
a breakout near the roll (BREAKOUT §3: "verify that stitched contracts do not fabricate breakouts
near rolls"). The roll rule (front-month / volume-OI switch / constant-maturity) is AMBER metadata;
the caller supplies the roll events. This module applies the price back-adjustment and exposes the
roll-window mask so the candidate generator can suppress roll-window crosses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from convexity.core.types import Bars


@dataclass(frozen=True)
class RollEvent:
    """One AMBER-supplied roll. ``adjust`` is the additive price gap removed by back-adjustment (the
    difference between expiring and new front-month price at the roll bar)."""

    roll_ts_ms: int
    adjust: float = 0.0
    window_bars: int = 2


class RollAdjuster:
    """Back-adjust roll steps + mark roll windows (BREAKOUT §7). One public method, ``adjust``."""

    def adjust(self, bars: Bars, rolls: tuple[RollEvent, ...]) -> Bars:
        """Back-adjust the roll price step and attach an ``is_roll_window`` mask."""
        if not rolls:
            return bars
        ts = bars.ts.astype(np.int64)
        n = len(bars)
        add = np.zeros(n)
        window = np.zeros(n, dtype=bool)
        if bars.is_roll_window is not None:
            window |= bars.is_roll_window.astype(bool)
        for r in rolls:
            idx = int(np.searchsorted(ts, r.roll_ts_ms))
            # back-adjust all bars before the roll by the additive gap (continuous stitched series).
            add[:idx] += r.adjust
            lo = max(0, idx - r.window_bars)
            hi = min(n, idx + r.window_bars + 1)
            window[lo:hi] = True
        return replace(
            bars,
            o=bars.o + add,
            h=bars.h + add,
            l=bars.l + add,
            c=bars.c + add,
            is_roll_window=window,
        )
