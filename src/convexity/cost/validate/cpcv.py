"""Sampled combinatorial purged CV (COST §12). Firewalled (Inv V).

CPCV groups ``N = 6``, test size ``k = 2`` -> ``C(6,2) = 15`` train/test splits and
<= 15 path evaluations (the ceiling; do NOT raise N past 8, ``C(8,2)=28``). CPCV gives
the path-robust point estimates; the SINGLE pooled block bootstrap (``stats.py``)
gives the CI on the pooled residuals — ``15 paths + 1×1000 bootstrap``, not
``15×1000`` (§12 nesting discipline collapses the product to a sum).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def cpcv_splits(n_groups: int = 6, k_test: int = 2) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The ``C(n_groups, k_test)`` (train_groups, test_groups) splits (§12).

    Bounded to <= 15 paths at the default ``N=6, k=2``; a guard raises past the
    ``C(8,2)=28`` ceiling so the combinatorial cost cannot grow unbounded on the VM.
    """
    if n_groups > 8:
        raise ValueError("CPCV N capped at 8 on the target VM (C(8,2)=28 ceiling, §12)")
    all_groups = tuple(range(n_groups))
    splits = []
    for test in combinations(all_groups, k_test):
        train = tuple(g for g in all_groups if g not in test)
        splits.append((train, test))
    return splits


def group_indices(n_obs: int, n_groups: int = 6) -> list[np.ndarray]:
    """Partition ``range(n_obs)`` into ``n_groups`` contiguous index blocks (§12)."""
    return [np.array(idx) for idx in np.array_split(np.arange(n_obs), n_groups)]
