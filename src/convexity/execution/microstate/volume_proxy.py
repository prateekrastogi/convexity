"""VolumeProxy — rolling expected NOTIONAL volume over a window, from past bars (§11).

Returns the expected notional volume over the next ``window_s`` from PAST bar volume and a coarse
intraday seasonality factor (no live order book — retail-feed limitation, §intro). Used ONLY as the
participation denominator in the impact model and the rate-limit gate (§11); it is NOT a fill-prob
input beyond what ``g`` / vol already carry. Causal: averages only bars at or before ``idx``.
"""

from __future__ import annotations

import numpy as np

from convexity.execution.config.resolve import ResolvedConfig

__all__ = ["VolumeProxy"]

_BAR_SECONDS = 60.0


class VolumeProxy:
    """Causal rolling notional-volume proxy (§11). Past volume × bars-in-window; no live book."""

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)

    def estimate(self, bars, idx: int, window_s: float, cfg: ResolvedConfig) -> float:
        """Expected notional volume over the next ``window_s`` from past bar volume (§11).

        Mean per-bar notional (volume × close) over the trailing window, scaled to the number of bars
        in ``window_s``. Returns a small positive floor when the feed carries no volume so the impact
        participation denominator never divides by zero. F_t-measurable.
        """
        if bars.v is None:
            return 1.0  # no volume on the feed: a positive floor (impact then ~0 at retail size)
        lo = max(0, idx - self.window + 1)
        vol = np.asarray(bars.v[lo : idx + 1], dtype=float)
        px = np.asarray(bars.c[lo : idx + 1], dtype=float)
        if vol.size == 0:
            return 1.0
        per_bar_notional = float(np.mean(vol * px))
        bars_in_window = max(window_s / _BAR_SECONDS, 1.0)
        return max(per_bar_notional * bars_in_window, 1.0)
