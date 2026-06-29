"""Risk-budget allocation + covariance conditioning (LEVERAGE §7, Inv P/COND-L).

Two RISK-based jobs (no μ, no Σ⁻¹μ — Inv P):

  * ``condition_covariance`` — the MANDATORY Ledoit–Wolf shrink + post-shrink
    condition-number / minimum-eigenvalue gate every MCTR/inversion runs first
    (Inv COND-L). Closed-form, data-derived shrinkage toward a constant-correlation /
    scaled-identity target — NO P&L fit, NO ML. On a gate breach it returns a robust
    DIAGONAL fallback and flags ``cov_cond_breach``. Shared by ``PortfolioCeiling`` and
    this allocator so the panel is conditioned IDENTICALLY (same family REGIME pins).

  * ``RiskBudgetAllocator.allocate`` — splits the total risk budget across
    simultaneously-qualified signals by EQUAL RISK CONTRIBUTION (ERC) within
    correlation clusters; a correlated cluster shares a budget partially-netted by MCTR
    (MODERATE netting, §7). This is a SECTION not a module: for a small retail book
    (<= ~6 positions) ERC/MCTR is sufficient — NO HRP, NO hierarchical dendrogram (§7.2).

MCTR (the standard risk-parity decomposition, §17):
    MCTR_i = w_i · (Σw)_i / sqrt(wᵀΣw)
and Σ_i MCTR_i == sqrt(wᵀΣw) (total portfolio volatility) — the sum identity tests
assert this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from convexity.leverage.allocate.clusters import cluster_by_correlation

_EPS: float = 1e-12


# --------------------------------------------------------------------------- covariance conditioning

@dataclass(frozen=True)
class ConditionedCov:
    """Result of the Ledoit–Wolf shrink + condition gate (Inv COND-L)."""

    cov: np.ndarray
    shrinkage: float
    condition_number: float
    min_eig_ratio: float
    breached: bool  # True -> the diagonal fallback was used; flag ``cov_cond_breach``


def ledoit_wolf_shrink(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form Ledoit–Wolf shrinkage toward a constant-correlation target.

    Shrinks the sample covariance toward a structured target whose off-diagonals use
    the average sample correlation and whose diagonals are the sample variances. The
    intensity δ∈[0,1] here is a robust closed-form proxy derived from the dispersion of
    the correlations (no P&L fit). Returns (shrunk_cov, delta). Pure numpy.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n <= 1:
        return cov.copy(), 0.0
    d = np.sqrt(np.clip(np.diag(cov), _EPS, None))
    corr = cov / np.outer(d, d)
    off = corr[~np.eye(n, dtype=bool)]
    r_bar = float(np.mean(off)) if off.size else 0.0
    target_corr = np.full((n, n), r_bar)
    np.fill_diagonal(target_corr, 1.0)
    target = (d[:, None] * target_corr) * d[None, :]
    # robust intensity: more shrink when correlations are dispersed / matrix is small
    disp = float(np.var(off)) if off.size else 0.0
    delta = float(np.clip(0.5 + disp - 1.0 / n, 0.0, 1.0))
    shrunk = (1.0 - delta) * cov + delta * target
    return shrunk, delta


def condition_covariance(
    cov: np.ndarray,
    *,
    max_condition: float,
    min_eig_ratio: float,
) -> ConditionedCov:
    """Ledoit–Wolf shrink then a condition-number / min-eigenvalue gate (Inv COND-L).

    If the shrunk matrix is near-singular or ill-conditioned (condition number above
    ``max_condition`` or smallest/largest eigenvalue ratio below ``min_eig_ratio``), the
    MCTR shares and therefore ``L_portfolio`` would be nonsense, so we fall back to a
    robust DIAGONAL of the (shrunk) variances and flag ``breached``.
    """
    cov = np.asarray(cov, dtype=float)
    shrunk, delta = ledoit_wolf_shrink(cov)
    eig = np.linalg.eigvalsh(0.5 * (shrunk + shrunk.T))
    eig = np.clip(eig, 0.0, None)
    max_eig = float(eig[-1]) if eig.size else 0.0
    min_eig = float(eig[0]) if eig.size else 0.0
    cond = (max_eig / min_eig) if min_eig > _EPS else np.inf
    eig_ratio = (min_eig / max_eig) if max_eig > _EPS else 0.0

    breached = (cond > max_condition) or (eig_ratio < min_eig_ratio)
    if breached:
        out = np.diag(np.clip(np.diag(shrunk), _EPS, None))
    else:
        out = shrunk
    return ConditionedCov(
        cov=out,
        shrinkage=delta,
        condition_number=float(cond),
        min_eig_ratio=float(eig_ratio),
        breached=bool(breached),
    )


# --------------------------------------------------------------------------- MCTR

def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Total portfolio volatility sqrt(wᵀΣw)."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def mctr(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal contribution to risk per instrument: MCTR_i = w_i·(Σw)_i / sqrt(wᵀΣw).

    Sums to the total portfolio vol (the risk-parity decomposition, §17). Returns a
    zero vector when the book has no risk.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    pv = portfolio_vol(w, cov)
    if pv <= _EPS:
        return np.zeros_like(w)
    sigma_w = cov @ w
    return (w * sigma_w) / pv


# --------------------------------------------------------------------------- ERC allocator

class RiskBudgetAllocator:
    """ERC/MCTR split of the total risk budget across simultaneous signals (§7.2)."""

    def allocate(
        self,
        signals: list[str],
        cov: np.ndarray,
        cov_index: tuple[str, ...],
        total_budget: float,
        cfg,
    ) -> Mapping[str, float]:
        """Return {instrument: risk_budget_share}. Equal-risk-contribution WITHIN each
        correlation cluster; a correlated cluster shares a budget partially-netted by
        MCTR (MODERATE netting). NO Σ⁻¹μ (Inv P) — purely covariance/risk based.

        The covariance is conditioned (Inv COND-L) before any cluster/risk computation.
        ``signals`` are the instrument ids being sized this step.
        """
        cond = condition_covariance(
            cov,
            max_condition=cfg.defaults.cov_max_condition,
            min_eig_ratio=cfg.defaults.cov_min_eig_ratio,
        )
        clusters = cluster_by_correlation(
            cond.cov, cov_index, cfg.defaults.cluster_corr_threshold
        )
        # map instrument -> its cluster's standalone vol (sqrt of its own variance)
        idx_of = {name: i for i, name in enumerate(cov_index)}
        sig = [s for s in signals if s in idx_of]
        if not sig:
            return {}

        # Cluster-level budget split: each CLUSTER gets an equal share of the total
        # budget (diversifying clusters each get more); members of a correlated cluster
        # then SHARE that cluster's budget partially-netted by their MCTR weights.
        active_clusters = [[m for m in c if m in sig] for c in clusters]
        active_clusters = [c for c in active_clusters if c]
        if not active_clusters:
            return {}
        per_cluster = float(total_budget) / len(active_clusters)

        out: dict[str, float] = {}
        for cl in active_clusters:
            if len(cl) == 1:
                out[cl[0]] = per_cluster
                continue
            # within-cluster ERC: inverse-vol weights, partially netted via MCTR shares
            sub_idx = [idx_of[m] for m in cl]
            sub_cov = cond.cov[np.ix_(sub_idx, sub_idx)]
            inv_vol = 1.0 / np.sqrt(np.clip(np.diag(sub_cov), _EPS, None))
            w = inv_vol / inv_vol.sum()
            contrib = mctr(w, sub_cov)
            contrib = np.clip(contrib, 0.0, None)
            shares = contrib / contrib.sum() if contrib.sum() > _EPS else w
            for m, s in zip(cl, shares):
                out[m] = per_cluster * float(s)
        return out
