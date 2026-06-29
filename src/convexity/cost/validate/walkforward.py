"""Purged walk-forward — forecast-vs-realized funding/cost (COST §12). Firewalled (Inv V).

Scores the forecast cost/funding against realized, net, on a purged walk-forward with
an embargo around funding stamps and rolls. The forecaster is a causal callable
``(history) -> r_hat``; the realized rates are the labels (future-only). Returns the
one-step residual series ``e_k = r_k - r_hat(τ_k)`` (§5.5c) pooled across folds — the
input to the SINGLE pooled block bootstrap (§12 nesting discipline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class WalkForwardResult:
    """Pooled out-of-fold residuals + summary error (§12)."""

    residuals: np.ndarray  # e_k = r_k - r_hat(τ_k), pooled across folds (for the bootstrap)
    bias: float  # mean residual (a negative funding-forecast bias = miscalibration, Inv E)
    rmse: float


def purged_walkforward(
    rates: np.ndarray,
    forecaster: Callable[[np.ndarray], float],
    *,
    n_folds: int = 4,
    embargo: int = 1,
) -> WalkForwardResult:
    """Purged walk-forward one-step residuals of a causal funding forecaster (§12).

    For each test point ``k`` the forecaster sees only history up to ``k - 1 - embargo``
    (the embargo purges funding-stamp leakage). The residual ``e_k = r_k - r_hat`` is
    accumulated. ``n_folds`` partitions the series only to bound compute; the residuals
    are pooled for the single downstream bootstrap.
    """
    rates = np.asarray(rates, dtype=float)
    n = rates.size
    residuals = []
    start = max(1 + embargo, n // (n_folds + 1))
    for k in range(start, n):
        history = rates[: max(k - embargo, 0)]
        if history.size == 0:
            continue
        r_hat = forecaster(history)
        residuals.append(rates[k] - r_hat)
    res = np.asarray(residuals, dtype=float)
    bias = float(np.mean(res)) if res.size else 0.0
    rmse = float(np.sqrt(np.mean(res**2))) if res.size else 0.0
    return WalkForwardResult(residuals=res, bias=bias, rmse=rmse)
