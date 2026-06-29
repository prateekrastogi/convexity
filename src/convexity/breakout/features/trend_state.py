"""TrendState — directional persistence evidence + false-break risk (BREAKOUT §7 / §6.4).

Returns a pair ``(trend_evidence, false_break_risk)``, both per-bar in [0, 1], using only data up to
signal time (BREAKOUT §features/trend_state: "Do not use future continuation"):

  * trend evidence: slope stability, higher-high/lower-low structure, close-location value.
  * false-break risk: wick rejection, poor close location, immediate mean-reversion diagnostics.

These are the §8 ``trend_e`` channel and the ``false_break_risk`` penalty.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.types import BreakoutGeometry, StdSeries


class TrendState:
    """Persistence + false-break risk (BREAKOUT §7). One public method, ``evidence``."""

    def __init__(self, *, slope_window: int = 20) -> None:
        self.slope_window = slope_window

    def evidence(
        self, bars: Bars, s: StdSeries, geom: BreakoutGeometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(trend_evidence, false_break_risk)`` per bar (§features/trend_state)."""
        clv = self._close_location_value(bars)  # [-1, 1]: where the close sits in the bar's range
        slope = self._slope_stability(bars.c)  # [0, 1]: consistency of the recent drift sign

        direction = geom.direction.astype(float)
        # trend evidence: a strong same-direction CLV + stable slope in the cross direction.
        directional_clv = np.clip(direction * clv, 0.0, 1.0)
        trend_e = np.clip(0.5 * directional_clv + 0.5 * slope, 0.0, 1.0)

        # false-break risk: wick against the move (poor CLV) + immediate reversion of x[t].
        wick_rejection = np.clip(-direction * clv, 0.0, 1.0)  # close back inside => rejection
        reversion = self._immediate_reversion(s.x, direction)
        false_break_risk = np.clip(0.6 * wick_rejection + 0.4 * reversion, 0.0, 1.0)
        # no candidate => no false-break risk attached.
        false_break_risk = np.where(direction != 0, false_break_risk, 0.0)
        return trend_e, false_break_risk

    def _close_location_value(self, bars: Bars) -> np.ndarray:
        """CLV in [-1, 1]: +1 close at the high, -1 close at the low (Williams' CLV, current bar)."""
        h, low, c = bars.h, bars.l, bars.c
        rng = np.where(h > low, h - low, np.nan)
        clv = ((c - low) - (h - c)) / (rng + 1e-12)
        return np.nan_to_num(clv, nan=0.0)

    def _slope_stability(self, c: np.ndarray) -> np.ndarray:
        """Fraction of the last ``slope_window`` log-return signs matching the dominant sign (past
        only — the window ends at t-1)."""
        n = c.size
        lr = np.zeros(n)
        lr[1:] = np.diff(np.log(np.clip(c, 1e-12, None)))
        out = np.zeros(n)
        w = self.slope_window
        for t in range(w + 1, n):
            seg = np.sign(lr[t - w : t])  # ends at t-1
            up = np.mean(seg > 0)
            dn = np.mean(seg < 0)
            out[t] = float(max(up, dn))
        return out

    def _immediate_reversion(self, x: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Risk that the breakout bar's thrust is a one-bar spike: high |x[t]| with a thin prior
        build-up. Uses x[t] (F_t, allowed at signal time) and the past, never the future."""
        n = x.size
        out = np.zeros(n)
        for t in range(2, n):
            if direction[t] == 0:
                continue
            # a big single-bar move relative to the prior two bars => spike-like => higher risk.
            recent = abs(x[t])
            prior = abs(x[t - 1]) + abs(x[t - 2]) + 1e-9
            spike = recent / (recent + prior)
            out[t] = float(np.clip(spike, 0.0, 1.0))
        return out
