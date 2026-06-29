"""FuturesRollGuard — futures roll/stitch metadata hooks (TREND §8 data/rolls).

Roll artifacts must NOT trigger trend death by themselves (§8). The guard reads the ``is_roll_window``
flag the feed attaches and ensures a stitch discontinuity inside a roll window does not feed the jump
detector or the death union: bars inside the flagged window have their jump-eligibility suppressed by
marking them roll-protected (the estimators read ``is_roll_window`` and skip the synthetic return).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, InstrumentSpec


class FuturesRollGuard:
    """Protect roll-window stitch artifacts from reading as trend death (§8)."""

    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Ensure an ``is_roll_window`` mask exists (all-False if the feed supplied none)."""
        n = len(bars)
        if bars.is_roll_window is not None:
            return bars
        return Bars(**{**bars.__dict__, "is_roll_window": np.zeros(n, dtype=bool)})

    @staticmethod
    def is_roll_protected(bars: Bars, idx: int) -> bool:
        """True iff bar ``idx`` is inside a flagged roll window (jump-suppressed)."""
        rw = bars.is_roll_window
        return rw is not None and 0 <= idx < len(bars) and bool(rw[idx])
