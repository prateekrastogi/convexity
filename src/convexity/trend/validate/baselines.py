"""BaselineSuite — the cheap baseline + ablation trails (TREND §12 baselines).

Evaluates the required §12 baselines under identical labels + timing so the convex exit can be
compared on equal footing (the promotion gate, §13, requires beating them on median MFE capture and
giveback). Each baseline is a simple causal exit rule producing a synthetic decision stream the
labeler scores; the heavy run lives in research, not the live loop (§14).

Required baselines (§12): plain vol-normalized trailing stop, fixed ATR/Chandelier, symmetric
trailing stop, high-watermark drawdown stop, time stop, SuperTrend-style, and the no-detector /
no-Kalman / no-jump / no-hazard ablations.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Position

BASELINE_NAMES = (
    "plain_vol_trail",
    "fixed_atr_chandelier",
    "symmetric_trail",
    "hwm_drawdown",
    "time_stop",
    "supertrend",
    "no_sequential",
    "no_kalman",
    "no_jump",
    "no_hazard",
)


class BaselineSuite:
    """Cheap baseline exit rules for the §12/§13 comparison."""

    def symmetric_trail_exit(
        self, bars: Bars, position: Position, k: float = 2.5, window: int = 60
    ) -> int:
        """Symmetric (fixed-multiple, profit-INDEPENDENT) trailing stop. Affine payoff (Inv L)."""
        side = float(position.side)
        e = position.entry_idx
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.diff(np.log(np.where(bars.c > 0, bars.c, np.nan)))
        hwm = bars.c[e]
        lwm = bars.c[e]
        for i in range(e + 1, len(bars)):
            lo = max(1, i - window)
            sub = ret[lo - 1 : i]
            sub = sub[np.isfinite(sub)]
            sigma = float(np.std(sub)) * bars.c[i] if sub.size else 0.0
            dist = max(k * sigma, bars.c[i] * 1e-4)
            if side > 0:
                hwm = max(hwm, bars.c[i])
                if bars.l[i] <= hwm - dist:
                    return i
            else:
                lwm = min(lwm, bars.c[i])
                if bars.h[i] >= lwm + dist:
                    return i
        return len(bars) - 1

    def time_stop_exit(self, bars: Bars, position: Position, hold_bars: int) -> int:
        """Fixed k-bar time stop (the degenerate clock the dynamic horizon is contrasted against)."""
        return min(position.entry_idx + hold_bars, len(bars) - 1)

    def hwm_drawdown_exit(
        self, bars: Bars, position: Position, dd_frac: float = 0.3
    ) -> int:
        """High-watermark fractional-drawdown stop (D4-style, policy-light retracement)."""
        side = float(position.side)
        e = position.entry_idx
        entry = position.entry_price
        m = 0.0
        for i in range(e + 1, len(bars)):
            x = side * float(np.log(bars.c[i] / entry))
            m = max(m, x)
            if m > 0 and (m - x) >= dd_frac * m:
                return i
        return len(bars) - 1
