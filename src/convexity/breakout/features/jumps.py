"""JumpEvidence — continuous-breakout vs isolated-jump diagnostics (BREAKOUT §7 / §6.4).

Separates a continuous, multi-bar breakout from an isolated one-bar jump / liquidity sweep that
mean-reverts (a high jump score => LOWER breakout quality, since a jump is not stored-energy
expansion).

CLAUDE.md EXCEPTION (docs/ARCHITECTURE.md §5.7): this defaults to the ROBUST STANDARDIZED-RETURN
EXCEEDANCE branch (the CLAUDE.md-compatible path for crypto), NOT the bipower-variation /
Lee-Mykland test. The Lee-Mykland branch is config-gated OFF by default and documented as the
exception (the CLAUDE.md forbid was scoped to the entry SIGNAL, not this feature diagnostic). Assets
with scheduled gaps / 24-7 trading get separate calibration via the vol bucket carried in StdSeries.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.types import StdSeries


class JumpEvidence:
    """Jump-vs-continuous evidence (BREAKOUT §7). One public method, ``evidence``."""

    def __init__(self, *, method: str = "robust_exceedance", threshold: float = 4.0) -> None:
        # "robust_exceedance" is the DEFAULT CLAUDE.md-compatible branch; "lee_mykland" is gated off.
        self.method = method
        self.threshold = threshold

    def evidence(self, bars: Bars, s: StdSeries) -> np.ndarray:
        """Return per-bar jump evidence in [0, 1]: high => bar t looks like an isolated jump."""
        if self.method == "lee_mykland":
            return self._lee_mykland(s)  # config-gated; off by default (CLAUDE.md exception)
        return self._robust_exceedance(s)

    def _robust_exceedance(self, s: StdSeries) -> np.ndarray:
        """Default: |x[t]| (already vol-standardized by the F_{t-1} sigma) exceeding a robust
        threshold AND not sustained into t+ ... but we only use t and the PAST: a jump is a big |x[t]|
        whose neighbours x[t-1] were quiet. F_t-measurable, no future bars (Inv B)."""
        x = s.x
        n = x.size
        out = np.zeros(n)
        for t in range(2, n):
            big = abs(x[t]) / self.threshold
            quiet_before = self.threshold / (abs(x[t - 1]) + abs(x[t - 2]) + self.threshold)
            out[t] = float(np.clip(np.tanh(big) * quiet_before, 0.0, 1.0))
        return out

    def _lee_mykland(self, s: StdSeries) -> np.ndarray:
        """Bipower-variation Lee-Mykland-style statistic (CONFIG-GATED OFF by default). Returns the
        normalised jump statistic squashed to [0, 1]. Uses only x[:t+1] (no future bars)."""
        x = s.x
        n = x.size
        out = np.zeros(n)
        w = 30
        for t in range(w + 1, n):
            seg = x[t - w : t]  # ends at t-1
            bpv = (np.pi / 2.0) * np.mean(np.abs(seg[1:]) * np.abs(seg[:-1]))
            scale = np.sqrt(bpv) + 1e-12
            stat = abs(x[t]) / scale
            out[t] = float(np.clip(np.tanh(stat / self.threshold), 0.0, 1.0))
        return out
