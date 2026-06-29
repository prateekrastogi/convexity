"""SpreadEstimator — the causal effective HALF-spread in bps via a 4-rung ladder (§5 / §11).

Estimator ladder (first that is well-defined wins; the chosen rung is logged for the validation
layer):

  1. QUOTED half-spread where the feed carries bid/ask.
  2. ROLL (1984): half-spread = sqrt(-cov(dp_t, dp_{t-1})) over the rolling window. DEGENERATE GUARD
     — Roll REQUIRES negative serial autocovariance; on trending 1-min bars it is FREQUENTLY POSITIVE
     so sqrt(-cov) is undefined (NaN). When cov >= -eps, Roll is UNDEFINED and falls through to rung 3.
  3. CORWIN-SCHULTZ (2012) high-low spread estimator: does NOT require negative serial covariance and
     is the documented fallback for trending bars. A non-positive estimate falls through to rung 4.
  4. CLASS-LEVEL PROXY FLOOR: the asset-class preset's typical half-spread (AMBER/preset), a finite
     conservative constant, NEVER NaN/inf/zero.

The return is ALWAYS a finite positive half-spread in bps. F_t-measurable; never uses bar idx+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import SpreadRung

__all__ = ["SpreadEstimate", "SpreadEstimator"]

_EPS = 1e-12


@dataclass(frozen=True)
class SpreadEstimate:
    """The half-spread in bps + the rung that produced it (logged for validation, §11)."""

    half_spread_bps: float
    rung: SpreadRung


class SpreadEstimator:
    """Causal effective-half-spread estimator with the 4-rung degeneracy-guarded ladder (§11)."""

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)

    def estimate(self, bars, idx: int, cfg: ResolvedConfig) -> SpreadEstimate:
        """Return the causal effective HALF-spread in bps at bar ``idx`` (§11). Never NaN/inf/zero."""
        mid = self._mid(bars, idx)

        quoted = self._quoted(bars, idx, mid)
        if quoted is not None:
            return SpreadEstimate(quoted, "quoted")

        roll = self._roll(bars, idx, mid)
        if roll is not None:
            return SpreadEstimate(roll, "roll")

        cs = self._corwin_schultz(bars, idx, mid)
        if cs is not None:
            return SpreadEstimate(cs, "corwin_schultz")

        # Rung 4 — conservative class-proxy floor (AMBER/preset). Finite positive by construction.
        return SpreadEstimate(float(cfg.class_proxy_half_spread_bps), "class_proxy")

    # ------------------------------------------------------------------ rungs

    def _mid(self, bars, idx: int) -> float:
        if bars.bid is not None and bars.ask is not None:
            b, a = float(bars.bid[idx]), float(bars.ask[idx])
            if b > 0.0 and a > 0.0:
                return 0.5 * (b + a)
        return float(bars.c[idx])

    def _quoted(self, bars, idx: int, mid: float) -> float | None:
        """Rung 1 — quoted half-spread (ask-bid)/2 in bps, where the feed carries bid/ask."""
        if bars.bid is None or bars.ask is None:
            return None
        b, a = float(bars.bid[idx]), float(bars.ask[idx])
        if not (b > 0.0 and a > 0.0 and a >= b and mid > 0.0):
            return None
        half = 0.5 * (a - b)
        return 10_000.0 * half / mid

    def _roll(self, bars, idx: int, mid: float) -> float | None:
        """Rung 2 — Roll (1984): half-spread = sqrt(-cov(dp_t, dp_{t-1})). UNDEFINED if cov >= 0."""
        lo = idx - self.window
        if lo < 1 or mid <= 0.0:
            return None
        prices = np.asarray(bars.c[lo - 1 : idx + 1], dtype=float)  # need one extra for the lag
        dp = np.diff(prices)
        if dp.size < 2:
            return None
        cov = float(np.cov(dp[1:], dp[:-1], bias=True)[0, 1])
        # DEGENERATE GUARD: Roll needs negative serial autocov; positive (trending) -> undefined.
        if cov >= -_EPS:
            return None
        spread = 2.0 * np.sqrt(-cov)  # Roll's full spread; half = spread/2 in price units
        half = 0.5 * spread
        return 10_000.0 * half / mid

    def _corwin_schultz(self, bars, idx: int, mid: float) -> float | None:
        """Rung 3 — Corwin-Schultz (2012) high-low spread estimator. Non-positive -> fall through."""
        if idx < 1 or mid <= 0.0:
            return None
        h0, l0 = float(bars.h[idx - 1]), float(bars.l[idx - 1])
        h1, l1 = float(bars.h[idx]), float(bars.l[idx])
        if min(l0, l1) <= 0.0 or h0 <= 0.0 or h1 <= 0.0:
            return None
        # beta = sum of squared single-bar log(high/low); gamma = squared 2-bar log(high/low).
        beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
        hi2 = max(h0, h1)
        lo2 = min(l0, l1)
        gamma = np.log(hi2 / lo2) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        denom = k
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        ea = np.exp(alpha)
        spread = 2.0 * (ea - 1.0) / (1.0 + ea)  # proportional spread S
        if not np.isfinite(spread) or spread <= 0.0:
            return None
        # S is a proportional FULL spread; half-spread in bps = 10_000 * S/2.
        return 10_000.0 * 0.5 * float(spread)
