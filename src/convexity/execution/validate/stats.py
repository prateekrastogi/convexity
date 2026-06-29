"""Statistics for the execution validation layer (§12, firewalled, Invariant V).

  * block-bootstrap CIs on IS and each cost component (blocks respect intraday volume/vol clustering,
    so adjacent fills are not treated as independent — Künsch / Politis-Romano);
  * deflated-Sharpe-style multiple-test control: deflate the apparent IS-improvement Sharpe for the
    number of trials (a χ-grid winner must clear the deflated threshold, not the raw one);
  * drop-top-N robustness: re-compute the IS advantage after dropping the N best (and N worst) fills;
    if the edge collapses when a handful of lucky fills are removed, the advantage is noise.

Validation-only; never imported by the live layers (Invariant V). scipy is allowed here per
ARCHITECTURE §5.8 (validation / research code); the deflation uses a numpy normal CDF to avoid the
hard dependency.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["block_bootstrap_ci", "deflated_sharpe", "drop_top_n"]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def block_bootstrap_ci(
    series: np.ndarray, block_len: int = 10, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of ``series`` (§12).

    Returns (mean, lo, hi) at the (1−alpha) level. Blocks of length ``block_len`` preserve the
    intraday clustering so adjacent fills are not treated as independent.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(np.ceil(n / block_len)))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        sample = np.concatenate([series[s : s + block_len] for s in starts])[:n]
        means[b] = sample.mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(series.mean()), lo, hi


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int) -> float:
    """A deflated-Sharpe-style probability the observed Sharpe beats the multiple-test threshold (§12).

    A coarse PSR/DSR: the expected MAX of ``n_trials`` standard-normal Sharpes is the deflation
    benchmark; the returned probability is ``Φ((SR − E[max]) · √n_obs)``. A χ-grid winner must clear
    a high deflated probability, not the raw Sharpe (§12).
    """
    if n_obs <= 1 or n_trials < 1:
        return 0.0
    # Expected max of n_trials iid N(0,1) (a standard approximation).
    e_max = (
        (1.0 - np.euler_gamma) * _ppf(1.0 - 1.0 / n_trials)
        + np.euler_gamma * _ppf(1.0 - 1.0 / (n_trials * math.e))
        if n_trials > 1
        else 0.0
    )
    z = (sharpe - e_max) * math.sqrt(n_obs - 1)
    return float(_norm_cdf(z))


def _ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation) — avoids a scipy hard dependency."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def drop_top_n(advantage: np.ndarray, n: int) -> tuple[float, float, float]:
    """Mean IS advantage, and the mean after dropping the N best and the N worst fills (§12).

    Returns (full_mean, drop_best_mean, drop_worst_mean). If the edge collapses when the N best are
    dropped, the advantage was a handful of lucky fills, not policy.
    """
    adv = np.sort(np.asarray(advantage, dtype=float))
    if adv.size == 0:
        return 0.0, 0.0, 0.0
    n = min(n, adv.size - 1) if adv.size > 1 else 0
    full = float(adv.mean())
    drop_best = float(adv[:-n].mean()) if n else full
    drop_worst = float(adv[n:].mean()) if n else full
    return full, drop_best, drop_worst
