"""WalkForward — purged walk-forward + sampled CPCV with a fill-horizon embargo (§12, Invariant V).

Sampled combinatorial purged CV (CPCV) over time folds, with an EMBARGO spanning the fill horizon, so
a fill's realized-cost label cannot leak across the train/test boundary (López de Prado). Purged
walk-forward is the primary split. Validation-only — never imported by the live layers.

This module produces the FOLD INDEX sets (the embargo is the load-bearing part); scoring a model on
each fold is the caller's job (it reuses :mod:`shortfall` / :mod:`baselines`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Fold", "purged_walkforward_folds"]


@dataclass(frozen=True)
class Fold:
    """One purged/embargoed train/test split over fill-ordered indices (§12)."""

    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walkforward_folds(
    n: int, n_folds: int = 5, embargo: int = 5
) -> list[Fold]:
    """Purged walk-forward folds with a fill-horizon embargo (§12).

    The test block is a contiguous time slice; the train block is everything strictly before it minus
    an ``embargo`` gap (the fill-horizon window) so a fill whose outcome straddles the boundary cannot
    leak its label into training. Walk-forward => train always precedes test.
    """
    if n <= 0 or n_folds < 1:
        return []
    fold_size = max(1, n // (n_folds + 1))
    folds: list[Fold] = []
    for k in range(1, n_folds + 1):
        test_lo = k * fold_size
        test_hi = min(n, test_lo + fold_size)
        if test_lo >= n:
            break
        test_idx = np.arange(test_lo, test_hi)
        train_hi = max(0, test_lo - embargo)  # embargo gap before the test block
        train_idx = np.arange(0, train_hi)
        if train_idx.size and test_idx.size:
            folds.append(Fold(train_idx, test_idx))
    return folds
