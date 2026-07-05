"""Purged CPCV splitter on a daily index (COMMODITY_PREREG.md §5).

N contiguous groups over the valid-day index, all C(N,2) test pairs; train = the remaining groups
minus an `embargo`-day purge on EACH side of every test-group boundary (kills hold-overlap leakage
for holds <= embargo). The baseline has no fitted params (a-priori cells), so its CPCV use is the
OOS path distribution; stage-3 candidates FIT on the train days of each split and score the test
days with params FROZEN — same splitter, matched comparison.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

N_GROUPS = 8
K_TEST = 2
EMBARGO_DAYS = 10


def group_bounds(n_days: int, n_groups: int = N_GROUPS) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_days, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def splits(n_days: int, n_groups: int = N_GROUPS, k: int = K_TEST,
           embargo: int = EMBARGO_DAYS) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_idx, test_idx), ...] over positions 0..n_days-1, purged+embargoed."""
    bounds = group_bounds(n_days, n_groups)
    out = []
    for combo in combinations(range(n_groups), k):
        test_mask = np.zeros(n_days, dtype=bool)
        banned = np.zeros(n_days, dtype=bool)
        for g in combo:
            lo, hi = bounds[g]
            test_mask[lo:hi] = True
            banned[max(0, lo - embargo):min(n_days, hi + embargo)] = True
        train_mask = ~banned
        out.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return out


def sharpe_ann_daily(x: np.ndarray, periods: int = 365) -> float:
    """Annualized Sharpe of a daily net-return series (24/7 grid => 365)."""
    x = np.asarray(x, dtype=float)
    if x.size < 20 or np.std(x) == 0:
        return 0.0
    return float(np.mean(x) / np.std(x) * np.sqrt(periods))


def path_sharpes(daily: np.ndarray, n_groups: int = N_GROUPS, k: int = K_TEST,
                 embargo: int = EMBARGO_DAYS) -> list[float]:
    """OOS Sharpe on each split's test days for a FIXED (non-fitted) strategy's daily series."""
    return [sharpe_ann_daily(daily[test]) for _, test in splits(len(daily), n_groups, k, embargo)]
