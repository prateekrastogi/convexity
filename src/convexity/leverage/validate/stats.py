"""Block-bootstrap CIs + multiple-test control (LEVERAGE §12, firewalled).

VALIDATION-ONLY (Invariant V). Block bootstrap respects vol clustering and correlation
(non-overlapping blocks); a Bonferroni helper for multiple-test control. Pure numpy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def block_bootstrap_ci(
    series: Sequence[float],
    *,
    block: int = 10,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Block-bootstrap (point, lo, hi) for the MEAN of ``series``, resampling contiguous
    blocks of length ``block`` to respect vol clustering / autocorrelation (§12)."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n_blocks = max(int(np.ceil(n / block)), 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    alpha = 1.0 - level
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(x.mean()), lo, hi


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Bonferroni multiple-test control: reject H0_i when p_i <= alpha/m (§12)."""
    p = list(p_values)
    m = max(len(p), 1)
    thresh = alpha / m
    return [pv <= thresh for pv in p]
