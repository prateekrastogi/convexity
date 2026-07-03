"""Ledoit-Wolf-shrunk covariance + absorption ratio -- the BLUE-only multivariate FALLBACK (REGIME.md §2.4).

This is NOT part of the v1 live detector (which uses only the univariate ``D_t``/``V_t`` -- Invariant MP
forbids any unshrunk covariance in a LIVE statistic, and Section 2.1/2.4 keep the multivariate path out
of v1 on Marchenko-Pastur grounds). ``turbulence.py`` is the file §9 reserves for the "accepted fallback
construction" a future BLUE-grid research variant may use. Its purpose HERE is to make Invariant MP's
stated empirical demonstration binding: on an ``N x T`` i.i.d. panel (no true structure), the UNSHRUNK
absorption ratio over-concentrates variance into a spurious top eigenvalue at a materially higher rate
than the Ledoit-Wolf-SHRUNK one at the same nominal threshold (§2.4). Every estimator below is
closed-form (Ledoit & Wolf 2004, Theorem 1 -- an analytic shrinkage intensity, NO optimization loop),
so nothing here is an "ML fit" in the P3-forbidden sense.
"""

from __future__ import annotations

import numpy as np


def sample_covariance(returns_nt: np.ndarray) -> np.ndarray:
    """Raw sample covariance ``Sigma_sample`` over an ``N x T`` (symbols x obs) return matrix.

    The UNSHRUNK estimator Invariant MP names as the bug when fed directly into an absorption ratio /
    Mahalanobis distance with ``N`` approaching ``T`` (§2.4). Provided only so the MP demonstration can
    exhibit its failure against the shrunk version -- never a live detector path.
    """
    x = np.asarray(returns_nt, dtype=float)
    return np.cov(x, ddof=1)


def ledoit_wolf_shrunk_covariance(returns_nt: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation target (REGIME.md §2.4).

    Shrinks ``Sigma_sample`` toward the structured target ``F`` (constant-correlation: off-diagonals
    replaced by the average sample correlation times the geometric mean of the two variances, diagonal
    kept) with the closed-form optimal intensity ``delta* = min(1, max(0, (pi_hat - rho_hat) / gamma_hat / T))``
    of Ledoit & Wolf's Theorem 1 (their honey-vs-target decomposition; here we use the standard
    ``pi``/``rho``/``gamma`` estimators). Analytic -- computed once, NO numerical optimization loop --
    so it stays closed-form and NOT an ML fit (§2.4, P3). Returns ``(Sigma_LW, delta_star)``.
    """
    x = np.asarray(returns_nt, dtype=float)
    n, t = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    s = (xc @ xc.T) / t  # MLE sample covariance (divide by T, per Ledoit-Wolf convention)
    var = np.diag(s)
    std = np.sqrt(var)
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = s / denom
    off = ~np.eye(n, dtype=bool)
    rbar = float(corr[off].mean()) if n > 1 else 0.0  # average off-diagonal sample correlation
    f = rbar * denom
    np.fill_diagonal(f, var)  # constant-correlation target F
    # pi_hat: sum of asymptotic variances of the sample covariance entries
    xc2 = xc**2
    pi_mat = (xc2 @ xc2.T) / t - s**2
    pi_hat = float(pi_mat.sum())
    # rho_hat: sum of asymptotic covariances of F's entries with the sample covariance entries
    theta_ii = ((xc2 * xc) @ xc.T) / t - var[:, None] * s
    theta_jj = (xc2 @ (xc * xc).T) / t - var[None, :] * s  # symmetric partner
    rho_diag = float(np.diag(pi_mat).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = (std[None, :] / std[:, None]) * theta_ii + (std[:, None] / std[None, :]) * theta_jj
    cross[np.diag_indices(n)] = 0.0
    # LW 2003 Appendix B: rho_hat = sum_i pi_ii + sum_{i!=j} (rbar/2)[sqrt(s_jj/s_ii) theta_ii,ij
    # + sqrt(s_ii/s_jj) theta_jj,ij] — NO further division. An earlier revision divided `cross` by
    # sqrt(var_i*var_j), making delta* SCALE-VARIANT (delta*(x) != delta*(c*x); at realistic
    # decimal-return scales the extra ~1/4e-4 factor drove kappa negative and delta* clamped to 0,
    # silently disabling shrinkage). Audit R-1; the scale-invariance test pins the fix.
    rho_hat = rho_diag + rbar * 0.5 * float(cross[off].sum())
    # gamma_hat: squared Frobenius distance between sample covariance and target
    gamma_hat = float(((f - s) ** 2).sum())
    if gamma_hat <= 0.0:
        delta_star = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta_star = max(0.0, min(1.0, kappa / t))
    sigma_lw = delta_star * f + (1.0 - delta_star) * s
    return sigma_lw, delta_star


def absorption_ratio(cov: np.ndarray, *, n_pc: int = 1) -> float:
    """Kritzman-Li absorption ratio: fraction of total variance in the top ``n_pc`` eigenvalues (§2.4).

    ``AR = (sum of top-n_pc eigenvalues) / (sum of all eigenvalues)``. On a truly-unstructured panel the
    population value is ``n_pc / N``; a sample estimate that reports a materially larger value is
    absorbing estimation NOISE, not real co-movement -- the Marchenko-Pastur failure Invariant MP names.
    """
    evals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    evals = np.sort(np.clip(evals, 0.0, None))[::-1]
    total = float(evals.sum())
    if total <= 0.0:
        return 0.0
    return float(evals[:n_pc].sum() / total)
