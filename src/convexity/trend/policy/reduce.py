"""ReducePolicy — reduce-only rules, no hard exits (TREND §10 reduce).

Reduce may use hazard, slope flip (Kalman ``filtered_slope`` against the position), gap survival, and
SEVERE volatility expansion. It does NOT hard-exit (that is DecisionPolicy's job by the fixed
priority order, §10) — this keeps hazard from becoming an underpowered exit classifier (§10). The
result is a (size_fraction, reason_codes) pair; the fraction is the MIN of the applicable reduce
caps, floored by the gap-survival minimum. R_t lowers size, never counts as trend death (§1.4).
"""

from __future__ import annotations

import math

from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.policy.convex_exit import ConvexExit
from convexity.trend.types import BeliefState, ExitState


class ReducePolicy:
    """Reduce-only sizing rules (§10). Never hard-exits."""

    def __init__(self, convex_exit: ConvexExit | None = None) -> None:
        self.convex_exit = convex_exit if convex_exit is not None else ConvexExit()

    def fraction(
        self, belief: BeliefState, state: ExitState, side: int, cfg: ResolvedConfig
    ) -> tuple[float, tuple[str, ...]]:
        """Return ``(size_fraction, reason_codes)`` (§10). The MIN of applicable reduce caps.

        ``side`` (+1 long / -1 short) side-adjusts the slope-flip rule (§13a.3 D3): the trigger is
        the Kalman ``filtered_slope`` crossing zero AGAINST the position, i.e. ``side*slope < 0`` —
        for a SHORT a favorable falling price (negative slope) must NOT spuriously reduce.
        """
        reasons: list[str] = []
        frac = 1.0

        # gap survival: cap size by the fraction that survives an adverse gap stress.
        gap_frac = self.convex_exit.gap_survival_fraction(belief.vol, cfg)
        if gap_frac < 1.0:
            frac = min(frac, gap_frac)
            reasons.append("gap_survival")

        # slope flip AGAINST the position (Kalman filtered_slope, §13a.3 D3; NOT a hard exit). The
        # filtered_slope is on log-price; side*slope < 0 is a deteriorating trend for EITHER side.
        if math.isfinite(belief.filtered_slope) and float(side) * belief.filtered_slope < 0.0:
            frac = min(frac, 0.5)
            reasons.append("slope_flip")

        # severe volatility expansion (top vol bucket) without a clean jump_with continuation.
        if belief.vol.vol_bucket >= cfg.defaults.n_vol_buckets - 1 and not belief.jump_with:
            frac = min(frac, 0.75)
            reasons.append("vol_expansion")

        # hazard (reduce-only; disabled -> hazard == 0 so no effect).
        if belief.hazard > 0.0:
            frac = min(frac, max(1.0 - belief.hazard, cfg.defaults.gap_survival_min_fraction))
            reasons.append("hazard")

        frac = max(frac, cfg.defaults.gap_survival_min_fraction)
        return float(frac), tuple(reasons)
