"""Validation statistics — block-bootstrap CIs + DSR/PSR (COST §12). Firewalled (Inv V).

The block bootstrap runs ONCE on the pooled out-of-fold residual series (NOT nested
per CPCV fold — the blow-up fix, §12), with block length ``L_funding`` set to the
funding-autocorrelation decorrelation length. DSR/PSR control the BLUE-grid
multiple-test comparison (forecast half-life, basis λ) so a candidate that "wins" by
chance over many candidates is not promoted (§12).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def block_bootstrap_quantile(
    series: np.ndarray, q: float, *, block_length: int, n_resamples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Block-bootstrap ``q``-quantile of a (clustered) series (§5.5c/§12).

    Returns (median, ci_lo, ci_hi) over ``n_resamples`` block-resamples. The block
    bootstrap (not an i.i.d. quantile) is required because residuals cluster. Run ONCE
    on the pooled residuals (§12 nesting discipline).
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    if n == 0:
        return 0.0, 0.0, 0.0
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    quantiles = np.empty(n_resamples)
    for b in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        sample = np.concatenate([series[s : s + block_length] for s in starts])[:n]
        quantiles[b] = np.quantile(sample, q)
    return (
        float(np.median(quantiles)),
        float(np.quantile(quantiles, 0.025)),
        float(np.quantile(quantiles, 0.975)),
    )


def probabilistic_sharpe_ratio(sr_hat: float, n_obs: int, sr_benchmark: float = 0.0,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR — the probability the true Sharpe exceeds ``sr_benchmark`` (§12)."""
    if n_obs <= 1:
        return 0.0
    denom = np.sqrt(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2)
    z = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(sr_hat: float, n_obs: int, n_trials: int,
                          sr_variance: float = 1.0, skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR — PSR against the expected-max-Sharpe benchmark from ``n_trials`` (§12).

    Controls the multiple-testing inflation when comparing the BLUE grid (half-life,
    λ): the benchmark Sharpe is the expected maximum of ``n_trials`` independent
    candidates under the null.
    """
    n_trials = max(1, n_trials)
    emc = 0.5772156649015329  # Euler-Mascheroni
    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_benchmark = np.sqrt(sr_variance) * ((1.0 - emc) * z1 + emc * z2)
    return probabilistic_sharpe_ratio(sr_hat, n_obs, sr_benchmark, skew, kurt)
