"""HorizonVol — short-horizon (seconds-to-minute) volatility in bps over a fill horizon (§11).

Returns the volatility in bps over ``horizon_s``, scaled by sqrt(horizon) from a robust per-bar
return scale (the family's robust vol estimator — a MAD-based scale, NOT a fresh model). Drives the
fill probability (how far price wanders to reach a resting offset) and the delay / adverse terms.
Causal: uses only bars at or before ``idx`` (never idx+1).
"""

from __future__ import annotations

import numpy as np

from convexity.execution.config.resolve import ResolvedConfig

__all__ = ["HorizonVol"]

# Per-bar interval in seconds (1-minute bars, §intro). Horizon scaling is sqrt(horizon_s / bar_s).
_BAR_SECONDS = 60.0
# MAD -> sigma consistency factor for a normal (1 / Phi^{-1}(0.75)).
_MAD_TO_SIGMA = 1.4826


class HorizonVol:
    """Causal short-horizon volatility estimator (§11). Reuses a robust MAD scale, not a new model."""

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)

    def estimate(self, bars, idx: int, horizon_s: float, cfg: ResolvedConfig) -> float:
        """Short-horizon vol in bps over ``horizon_s``, sqrt-scaled from a robust per-bar return scale.

        Per-bar bps return scale via MAD (robust to the fat tails of 1-min crypto); scaled to the
        horizon by sqrt(horizon_s / bar_seconds). Always finite and positive (floored at a tiny
        epsilon so downstream divisions are safe). F_t-measurable.
        """
        lo = max(1, idx - self.window + 1)
        prices = np.asarray(bars.c[lo - 1 : idx + 1], dtype=float)
        if prices.size < 2:
            return 1.0  # degenerate cold-start: a 1 bp floor, never zero/NaN
        rets_bps = 10_000.0 * np.diff(prices) / prices[:-1]
        med = np.median(rets_bps)
        mad = np.median(np.abs(rets_bps - med))
        per_bar = float(_MAD_TO_SIGMA * mad)
        if per_bar <= 0.0 or not np.isfinite(per_bar):
            per_bar = float(np.std(rets_bps)) if rets_bps.size else 1.0
        per_bar = max(per_bar, 1e-6)
        scale = np.sqrt(max(horizon_s, 0.0) / _BAR_SECONDS)
        return float(per_bar * scale)
