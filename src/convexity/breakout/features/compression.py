"""CompressionState — squeeze / stored-energy evidence before the breakout (BREAKOUT §7 / §6.2).

Returns calibrated evidence in [0, 1] that a squeeze preceded the breakout. High evidence = stored
directional energy (a tight, low-entropy, contracting range); low evidence = ordinary noise. Uses
past-only percentiles of realized range / ATR width and the entropy of recent return signs — all
F_{t-1}-measurable (no future quantiles, §12.4).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.types import BreakoutGeometry, StdSeries


class CompressionState:
    """Squeeze evidence (BREAKOUT §7). One public method, ``evidence``."""

    def __init__(self, *, window: int = 60) -> None:
        self.window = window

    def evidence(self, bars: Bars, s: StdSeries, geom: BreakoutGeometry) -> np.ndarray:
        """Return calibrated squeeze evidence in [0, 1] for every bar (§features/compression).

        Blends (a) the range-width compression percentile already computed in the geometry, with
        (b) a volatility-contraction ratio: recent realized vol over a short window vs a longer one
        (NR-style squeeze). Both are past-only.
        """
        comp = np.clip(geom.compression_pct, 0.0, 1.0)
        vol_contraction = self._vol_contraction(s.x)
        ev = 0.5 * comp + 0.5 * vol_contraction
        return np.clip(ev, 0.0, 1.0)

    def _vol_contraction(self, x: np.ndarray) -> np.ndarray:
        """Short-window realized vol / long-window realized vol, mapped so contraction -> high.

        Uses only past returns (the windows end at t-1 via cumulative sums shifted one bar)."""
        n = x.size
        short_w = max(self.window // 4, 5)
        long_w = self.window
        sq = x * x
        out = np.zeros(n)
        for t in range(long_w + 1, n):
            short_v = np.mean(sq[t - short_w : t])  # ends at t-1
            long_v = np.mean(sq[t - long_w : t])
            ratio = np.sqrt(short_v / (long_v + 1e-12))
            # contraction (ratio < 1) -> evidence near 1; expansion -> near 0.
            out[t] = float(np.clip(1.0 - ratio, 0.0, 1.0))
        return out
