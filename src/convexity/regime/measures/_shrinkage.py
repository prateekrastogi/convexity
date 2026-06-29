"""Closed-form Ledoit–Wolf shrinkage + the post-shrink condition gate (REGIME §2.2/§2.3).

Both absorption (correlation ``C_t``) and turbulence (covariance ``Sigma_t``) condition their matrix
with Ledoit–Wolf shrinkage toward the **constant-correlation target** BEFORE the eigendecomposition /
inverse. The shrinkage intensity ``δ`` is the analytic Ledoit–Wolf (2004) estimator — the ratio of
estimated dispersion ``π`` to the squared Frobenius distance ``γ`` between sample and target — clamped
to ``[0, 1]``. It is computed CAUSALLY from the trailing panel only and is TEAL (closed-form,
data-derived, NOT a RED/BLUE knob, NOT in ``K_regime``; Invariant ARCOND/COND).

NO ML: these are closed-form linear-algebra estimators (``scipy.linalg.eigh`` for the symmetric
eigendecomposition), exactly the no-tuning shrinkage Kritzman/Ledoit–Wolf prescribe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class ShrinkResult:
    """A shrunk PSD matrix + its closed-form intensity, eigenvalues, and condition number."""

    matrix: np.ndarray  # the shrunk matrix (correlation: unit diagonal preserved)
    intensity: float  # delta in [0, 1] — TEAL closed-form (Invariant ARCOND/COND)
    eigenvalues: np.ndarray  # descending, tiny-negative clamped to 0 (Invariant TRACE)
    cond_number: float  # lambda_max / lambda_min, post-shrink (Invariant CONDNUM)


def _constant_correlation_corr_target(corr: np.ndarray) -> np.ndarray:
    """Constant-correlation target for a CORRELATION matrix: unit diagonal, mean off-diag elsewhere.

    Being itself a correlation matrix (unit diagonal), the shrink ``δ·F + (1−δ)·S`` keeps a unit
    diagonal and therefore ``trace = n`` exactly — the TRACE identity survives the shrink (§2.2).
    """
    n = corr.shape[0]
    if n <= 1:
        return corr.copy()
    off = corr[~np.eye(n, dtype=bool)]
    r_bar = float(off.mean()) if off.size else 0.0
    target = np.full((n, n), r_bar, dtype=float)
    np.fill_diagonal(target, 1.0)
    return target


def _eig_clamped(mat: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of a symmetric matrix, tiny-negative round-off clamped to 0 (TRACE)."""
    w = eigh(mat, eigvals_only=True)
    w = np.clip(w, 0.0, None)
    return np.sort(w)[::-1]


def _cond_number(eigvals: np.ndarray) -> float:
    """lambda_max / lambda_min with a guarded denominator (post-shrink condition, CONDNUM)."""
    if eigvals.size == 0:
        return 1.0
    lam_max = float(eigvals[0])
    lam_min = float(eigvals[-1])
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def lw_shrink_correlation(returns: np.ndarray) -> ShrinkResult:
    """Ledoit–Wolf constant-correlation shrink of the rolling CORRELATION matrix (REGIME §2.2).

    ``returns`` is the ``(W, n)`` causally de-meaned panel. Returns the shrunk correlation matrix
    (unit diagonal preserved so ``trace = n`` exactly — TRACE) together with the closed-form
    intensity ``δ_AR`` (TEAL), its clamped descending eigenvalues, and its post-shrink condition
    number. ``δ`` is the analytic Ledoit–Wolf estimator computed from the same sample moments used
    to form the correlation matrix (cost ``O(n²·W)``; Invariant ARCOND).
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if n <= 1:
        c = np.ones((max(n, 1), max(n, 1)), dtype=float)
        return ShrinkResult(c, 0.0, _eig_clamped(c), 1.0)

    std = x.std(axis=0, ddof=0)
    safe = np.where(std > 0, std, 1.0)
    z = x / safe  # standardized returns: sample correlation S = (z^T z)/t
    s = (z.T @ z) / t
    np.fill_diagonal(s, 1.0)

    f = _constant_correlation_corr_target(s)
    delta = _lw_intensity(z, s, f, t)
    shrunk = delta * f + (1.0 - delta) * s
    np.fill_diagonal(shrunk, 1.0)  # guard the unit diagonal -> trace = n exactly (TRACE)
    shrunk = 0.5 * (shrunk + shrunk.T)  # symmetrize against round-off

    eigvals = _eig_clamped(shrunk)
    return ShrinkResult(shrunk, float(delta), eigvals, _cond_number(eigvals))


def lw_shrink_covariance(returns: np.ndarray) -> ShrinkResult:
    """Ledoit–Wolf constant-correlation shrink of the rolling COVARIANCE matrix (REGIME §2.3).

    Same closed-form family as :func:`lw_shrink_correlation`, applied to the covariance ``Sigma_t``
    REGIME inverts for the Mahalanobis distance. Returns the shrunk ``Sigma_t`` plus ``δ_turb``
    (TEAL), eigenvalues, and post-shrink condition number (Invariant COND/CONDNUM).
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if n <= 1:
        v = float(x.var(ddof=0)) if x.size else 1.0
        cov = np.array([[max(v, 1e-12)]], dtype=float)
        return ShrinkResult(cov, 0.0, _eig_clamped(cov), 1.0)

    std = x.std(axis=0, ddof=0)
    s = (x.T @ x) / t
    safe = np.where(std > 0, std, 1.0)
    z = x / safe
    corr = (z.T @ z) / t
    np.fill_diagonal(corr, 1.0)
    r_bar = float(corr[~np.eye(n, dtype=bool)].mean())
    # Constant-correlation covariance target: F_ij = r_bar * std_i * std_j off-diag, var_i on diag.
    f = r_bar * np.outer(safe, safe)
    np.fill_diagonal(f, np.diag(s))

    delta = _lw_intensity(x, s, f, t)
    shrunk = delta * f + (1.0 - delta) * s
    shrunk = 0.5 * (shrunk + shrunk.T)

    eigvals = _eig_clamped(shrunk)
    return ShrinkResult(shrunk, float(delta), eigvals, _cond_number(eigvals))


def _lw_intensity(data: np.ndarray, sample: np.ndarray, target: np.ndarray, t: int) -> float:
    """Analytic Ledoit–Wolf shrinkage intensity δ = clamp(π / γ ... , 0, 1) (Ledoit–Wolf 2004).

    ``π`` is the sum of asymptotic variances of the sample entries (the estimated dispersion); ``γ``
    is the squared Frobenius distance between sample and target. The cross term ``ρ`` toward the
    constant-correlation target is dominated by the diagonal/π contribution; we use the standard
    ``κ = (π − ρ_diag)/γ`` simplification with ``ρ`` approximated by its diagonal part, which is the
    closed-form, no-tuning estimator (it is data-derived and causal; TEAL).
    """
    data = np.asarray(data, dtype=float)
    n = sample.shape[0]
    # pi: sum over i,j of Var(sample_ij). With y = data column-centred already (panel is de-meaned),
    # the asymptotic variance estimator is mean_t (data_ti*data_tj - sample_ij)^2.
    pi_mat = np.zeros((n, n), dtype=float)
    for k in range(t):
        outer = np.outer(data[k], data[k]) - sample
        pi_mat += outer * outer
    pi_mat /= t
    pi = float(pi_mat.sum())

    gamma = float(np.sum((sample - target) ** 2))
    if gamma <= 0.0:
        return 0.0
    # rho diagonal contribution only (off-diagonal ~ rho cross term is small toward const-corr).
    rho = float(np.trace(pi_mat))
    kappa = (pi - rho) / gamma
    delta = kappa / t
    return float(min(max(delta, 0.0), 1.0))
