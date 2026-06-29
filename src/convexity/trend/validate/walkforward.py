"""WalkForward — purged, embargoed walk-forward split plan (TREND §12 walkforward).

All TEAL fits, null calibrations, hazard fits, and class-preset choices in a fold must train only on
permitted training history (§12). This module produces the purged/embargoed split plan; the embargo
length must cover the maximum realized hold (``H_sys``) so a test position cannot overlap a train
fit (consistent with SYSTEM's integrated-validation embargo, §1.6a). Validation-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitPlan:
    """A purged/embargoed train/test split plan over bar indices (§12)."""

    folds: tuple[tuple[np.ndarray, np.ndarray], ...]  # (train_idx, test_idx) per fold
    embargo: int


class WalkForward:
    """Purged walk-forward split builder (§12)."""

    def plan(self, n: int, n_folds: int, embargo: int) -> SplitPlan:
        """Build expanding-window folds with a purge+embargo gap (§12)."""
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        fold_size = n // (n_folds + 1)
        folds = []
        for k in range(1, n_folds + 1):
            train_end = fold_size * k
            test_start = train_end + embargo  # purge + embargo gap (>= H_sys-equivalent)
            test_end = min(test_start + fold_size, n)
            if test_start >= n:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if test_idx.size:
                folds.append((train_idx, test_idx))
        return SplitPlan(folds=tuple(folds), embargo=embargo)
