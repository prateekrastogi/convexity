"""Ablations — which cost components matter (COST §12). Firewalled (Inv V).

Required ablations: funding forecast on/off in the gate and hold input; EWMA vs basis
vs published-next, each alone and blended; impact on/off; fee-tier dynamic vs flat;
drop-top-N robustness (remove the N most-impactful trades/regimes and confirm the
funding model's edge is not a handful of funding spikes). Each ablation toggles ONE
estimator and re-prices the same trades through the same CostBook.
"""

from __future__ import annotations

import numpy as np


def drop_top_n(values: np.ndarray, n: int) -> np.ndarray:
    """Remove the ``n`` largest-magnitude entries (§12 drop-top-N robustness)."""
    values = np.asarray(values, dtype=float)
    if n <= 0 or values.size == 0:
        return values
    order = np.argsort(np.abs(values))[::-1]
    keep = order[n:]
    return values[np.sort(keep)]


def ablation_delta(with_component: np.ndarray, without_component: np.ndarray) -> dict:
    """Summary of a single on/off ablation: mean/abs-mean shift in decision quality (§12)."""
    a = np.asarray(with_component, dtype=float)
    b = np.asarray(without_component, dtype=float)
    return {
        "mean_with": float(np.mean(a)) if a.size else 0.0,
        "mean_without": float(np.mean(b)) if b.size else 0.0,
        "mean_delta": float(np.mean(a) - np.mean(b)) if a.size and b.size else 0.0,
    }
