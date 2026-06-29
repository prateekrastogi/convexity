"""QualityProfiler — missing/stale/outlier bar profiling (TREND §3/§8, TEAL).

A causal profiler of the data-health signals the policy reads as "untradeable" (priority rung 1 of
§10 decision): stale prints, missing-bar gaps (a ts step materially larger than the modal step), and
the fraction of invalid bars in a trailing window. TEAL: every statistic at bar t uses only bars at
or before t. The output is diagnostics only (log-only via BeliefDiagnostics), never an action.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars


class QualityProfiler:
    """Causal data-health profiler (§3/§8). All statistics are past-only (TEAL)."""

    def __init__(self, gap_multiple: float = 1.5) -> None:
        self.gap_multiple = gap_multiple

    def is_missing_gap(self, bars: Bars, idx: int) -> bool:
        """True iff the ts step into ``idx`` is materially larger than the modal step (missing bars)."""
        if idx <= 1:
            return False
        steps = np.diff(bars.ts[: idx + 1].astype(np.int64))
        if steps.size == 0:
            return False
        modal = np.median(steps[:-1]) if steps.size > 1 else steps[0]
        return bool(steps[-1] > self.gap_multiple * modal) if modal > 0 else False

    def is_tradeable(self, bars: Bars, idx: int) -> bool:
        """True iff bar ``idx`` is valid and not halted (the §10 rung-1 untradeable check)."""
        if bars.is_valid is not None and not bool(bars.is_valid[idx]):
            return False
        if bars.is_halt is not None and bool(bars.is_halt[idx]):
            return False
        return True
