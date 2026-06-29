"""Purged walk-forward of realized vs planned loss, net of costs (LEVERAGE §12, firewalled).

VALIDATION-ONLY (Invariant V). A thin purged walk-forward harness: split a time-ordered
decision/outcome series into train/test folds with an embargo gap so a label that spans
the boundary cannot leak across it (López de Prado purging). The combine/ceiling layers
NEVER import this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    train_idx: tuple[int, ...]
    test_idx: tuple[int, ...]


def purged_splits(n: int, n_folds: int, embargo: int) -> list[WalkForwardSplit]:
    """Forward-chaining folds with an ``embargo`` purge between train and test (§12).

    Fold k trains on ``[0, test_start - embargo)`` and tests on the k-th contiguous block;
    the embargo gap drops the ``embargo`` observations immediately before each test block
    so a label overlapping the boundary cannot leak (purged walk-forward).
    """
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    out: list[WalkForwardSplit] = []
    for k in range(1, n_folds + 1):
        test_lo, test_hi = int(bounds[k - 1]), int(bounds[k])
        train_hi = max(test_lo - embargo, 0)
        if train_hi <= 0:
            continue
        out.append(WalkForwardSplit(
            train_idx=tuple(range(0, train_hi)),
            test_idx=tuple(range(test_lo, test_hi)),
        ))
    return out


def exceedance_rate(planned: Sequence[float], realized: Sequence[float]) -> float:
    """Fraction of trades whose REALIZED loss exceeded its PLANNED loss (§12): systematic
    exceedance ⇒ gaps breaking stops ⇒ tail/gap/liquidity ceiling too loose."""
    p = np.asarray(planned, dtype=float)
    r = np.asarray(realized, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean(r > p + 1e-12))
