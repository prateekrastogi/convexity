"""Baselines + ablations the panel must beat on detection-delay-at-fixed-false-alarm (§11.3).

Spec: OPS §11.3. The panel must beat these on detection-delay-at-fixed-false-alarm across the synthetic
breaks; a detector that NEVER fires on its designed break is mistuned/redundant; a detector that floods
on in-control OOS data has too-tight limits. The objective is statistical (detection quality), never
P&L (Invariant OOS). Part of the MODEL — never imported by the live layers (Invariant V).

Baselines:
  * NO-MONITOR        : no detector; measures the loss a real break causes unhalted.
  * NAIVE-3sigma-ONLY : Shewhart only — blind to small persistent drifts (edge-decay / cost-staleness).
  * FIXED-WINDOW-MEAN : a fixed-N rolling-mean t-test — too slow (blind) or too small-N (floods).
"""

from __future__ import annotations

import numpy as np

from convexity.ops.detectors.cusum import Cusum
from convexity.ops.detectors.shewhart import Shewhart


def no_monitor_delay(_stream: np.ndarray) -> float:
    """NO-MONITOR baseline: never detects — the detection delay is the full stream (the break runs
    unhalted). Returns ``len(stream)`` as the "delay" (the loss a real break causes unhalted, §11.3)."""
    return float(len(_stream))


def shewhart_only_delay(stream: np.ndarray, change_idx: int, L_s: float = 3.0) -> float:
    """NAIVE-3sigma-ONLY baseline: Shewhart alone (§11.3). Blind to a small persistent drift — on a
    sub-3σ cost-staleness drift it often never fires (returns the full post-change length)."""
    det = Shewhart(L_s)
    for i, z in enumerate(stream):
        st = det.update(float(z))
        if st.alarm and i >= change_idx:
            return float(i - change_idx + 1)
    return float(len(stream) - change_idx)


def fixed_window_mean_delay(stream: np.ndarray, change_idx: int, window: int = 20,
                            crit: float = 3.0) -> float:
    """FIXED-WINDOW-MEAN baseline: a fixed-N rolling-mean z-test (§11.3). Too slow at small shifts (the
    window must fill) or floods at small N — it cannot match the sequential detectors' delay."""
    for i in range(window, len(stream)):
        w = stream[i - window:i]
        zbar = float(np.mean(w)) * np.sqrt(window)   # standardized window mean (σ0==1 standardized)
        if abs(zbar) > crit and i >= change_idx:
            return float(i - change_idx + 1)
    return float(len(stream) - change_idx)


def cusum_delay(stream: np.ndarray, change_idx: int, k: float = 0.5, h: float = 4.77) -> float:
    """The panel's CUSUM detection delay on the same stream (§11.3) — the value the baselines must be
    beaten on. The sequential CUSUM catches a small persistent drift the Shewhart-only baseline misses.
    """
    det = Cusum(k, h)
    for i, z in enumerate(stream):
        st = det.update(float(z))
        if st.alarm and i >= change_idx:
            return float(i - change_idx + 1)
    return float(len(stream) - change_idx)


__all__ = [
    "no_monitor_delay",
    "shewhart_only_delay",
    "fixed_window_mean_delay",
    "cusum_delay",
]
