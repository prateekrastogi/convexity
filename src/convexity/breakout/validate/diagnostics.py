"""Diagnostics — per-regime breakdown + score deciles (BREAKOUT §7 / §10).

Validation-only reporting (Invariant C). Log-only: diagnostics never change a signal.
The quality-score decile-monotonicity table has a SINGLE canonical implementation in
`stats.py` (re-exported here so callers can reach it from either module); this module adds
the per-volatility-regime breakdown, which is unique to it.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, EntrySignal

from convexity.breakout.types import EntryLabel
from convexity.breakout.validate.stats import score_decile_table

__all__ = ["score_decile_table", "by_regime_table"]


def by_regime_table(
    bars: Bars, signals: list[EntrySignal], labels: list[EntryLabel], vol_bucket: np.ndarray
) -> list[dict]:
    """Net forward return by volatility bucket (§10 by-regime breakdown)."""
    if not labels:
        return []
    by_sig = {s.signal_idx: s for s in signals}
    groups: dict[int, list[float]] = {}
    for lb in labels:
        s = by_sig.get(lb.signal_idx)
        if s is None:
            continue
        b = int(vol_bucket[s.signal_idx])
        groups.setdefault(b, []).append(lb.net_return_to_horizon_bps)
    return [{"vol_bucket": b, "n": len(v), "net_bps": float(np.mean(v))}
            for b, v in sorted(groups.items())]
