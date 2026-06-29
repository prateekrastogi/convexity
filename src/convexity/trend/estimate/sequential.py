"""SequentialDeterioration — EXACTLY one of CUSUM/SPRT (TREND §9 sequential, Inv B).

Configure exactly one detector (Inv B): do NOT run both and reconcile them, and do NOT add a second
detector for the same latent quantity (drift decay). CUSUM is the cheap-compute default (§20).

CUSUM mode (one-sided deterioration):
    S_t = max(0, S_{t-1} + (-x_t) - k)         # accumulates evidence that drift turned unfavorable
    alarm when S_t > h, with h calibrated to a target ARL under the empirical/Student-t null.

SPRT mode (favorable H0 vs dead/unfavorable H1):
    LLR_t = sum log( f(x; mu1) / f(x; mu0) )    # Gaussian increments
    alarm when LLR_t >= log((1 - beta) / alpha).

The increment ``side_adjusted_std_return`` is the side-adjusted standardized return (favorable
positive). A pure volatility shift must not reliably trigger deterioration (acceptance gate) — the
standardization removes scale, so only a genuine drift-sign change drifts the statistic.
"""

from __future__ import annotations

import math

from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.types import DetectorKind, VolState


class SequentialDeterioration:
    """One configured deterioration detector (§9, Inv B)."""

    def __init__(self, kind: DetectorKind, cfg: ResolvedConfig) -> None:
        if kind not in ("cusum", "sprt"):
            raise ValueError(f"detector kind must be 'cusum' or 'sprt' (Inv B), got {kind!r}")
        self.kind = kind
        self.cfg = cfg
        self._s = 0.0  # CUSUM accumulator / SPRT log-likelihood ratio
        self._alarm = False

    def reset(self) -> None:
        self._s = 0.0
        self._alarm = False

    def update(
        self, side_adjusted_std_return: float, vol: VolState
    ) -> tuple[float, bool]:
        """Return ``(seq_stat, seq_alarm)`` for one bar (§9). Causal increments only."""
        d = self.cfg.defaults
        x = float(side_adjusted_std_return)

        if self.kind == "cusum":
            # One-sided: accumulate UNFAVORABLE evidence (-x), with reference slack k.
            self._s = max(0.0, self._s + (-x) - d.cusum_k)
            self._alarm = self._s > d.cusum_h
            return float(self._s), bool(self._alarm)

        # SPRT: Gaussian increments, H0 favorable mu0, H1 unfavorable mu1, unit std (standardized).
        mu0, mu1 = d.sprt_mu0, d.sprt_mu1
        # log f(x;mu1)/f(x;mu0) for unit-variance Gaussian = (mu1-mu0)*x - (mu1^2 - mu0^2)/2
        llr_incr = (mu1 - mu0) * x - 0.5 * (mu1**2 - mu0**2)
        self._s += llr_incr
        upper = math.log((1.0 - d.sprt_beta) / d.sprt_alpha)
        lower = math.log(d.sprt_beta / (1.0 - d.sprt_alpha))
        if self._s <= lower:  # accept H0 (still favorable) -> reset accumulation
            self._s = 0.0
        self._alarm = self._s >= upper
        return float(self._s), bool(self._alarm)
