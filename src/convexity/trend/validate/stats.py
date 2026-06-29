"""Statistics — block-bootstrap CIs + multiple-test correction (TREND §12 stats).

Block bootstrap preserves autocorrelation in the per-trade metric series (§12: median and tail
metrics, not only mean/Sharpe). The multiple-test correction (Benjamini-Hochberg FDR) penalizes the
BLUE grid trials (§12 / §4: expanding a grid after seeing results needs a recorded multiple-test
penalty). Validation-only.
"""

from __future__ import annotations

import numpy as np


def block_bootstrap_ci(
    series: np.ndarray,
    statistic=np.median,
    block: int = 10,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point, ci_low, ci_high)`` for ``statistic`` via the moving-block bootstrap (§12)."""
    series = np.asarray(series, dtype=float)
    n = series.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    point = float(statistic(series))
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        sample = np.concatenate([series[s : s + block] for s in starts])[:n]
        stats[b] = statistic(sample)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR control over the BLUE-grid trial p-values (§12). Returns a reject mask."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject
