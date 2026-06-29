"""Correlation clustering — SIMPLE, capped grouping for budget sharing (LEVERAGE §7).

Used ONLY to decide which positions share a partially-netted risk budget — NOT a full
HRP dendrogram (forbidden, §0 / §7.2: HRP is built for large universes and overfits a
small retail book of <= ~6 positions). The grouping is a single-linkage threshold on
pairwise correlation: instruments whose pairwise |ρ| exceeds the RED
``cluster_corr_threshold`` are transitively grouped (NIFTY/BANKNIFTY/bank-stocks land in
one cluster; Gold/USDINR/NatGas stay separate). Pure connected-components over the
correlation graph — no linkage matrix, no tree.
"""

from __future__ import annotations

import numpy as np


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix (D^-1/2 Σ D^-1/2)."""
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    return np.clip(corr, -1.0, 1.0)


def cluster_by_correlation(
    cov: np.ndarray, index: tuple[str, ...], threshold: float
) -> list[list[str]]:
    """Group instruments into correlation clusters via single-linkage connected components.

    Two instruments are linked when ``|ρ_ij| >= threshold``; clusters are the connected
    components of that graph. Returns a list of clusters (each a list of instrument ids),
    deterministic by ``index`` order. A singleton instrument is its own cluster.
    """
    corr = correlation_from_cov(cov)
    n = len(index)
    adj = np.abs(corr) >= float(threshold)
    seen = [False] * n
    clusters: list[list[str]] = []
    for i in range(n):
        if seen[i]:
            continue
        # BFS over the correlation graph from i
        stack = [i]
        comp: list[int] = []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in range(n):
                if not seen[k] and adj[j, k]:
                    seen[k] = True
                    stack.append(k)
        clusters.append(sorted(index[j] for j in comp))
    return clusters
