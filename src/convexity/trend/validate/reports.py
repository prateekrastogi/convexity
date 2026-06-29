"""Reports — asset-class + regime reports (TREND §12/§13 reports).

Aggregates per-trade labels into the §13 headline table: net E[Pi] with p_win/avg_win/p_loss/
avg_loss, premium_bleed reported ALONGSIDE tail capture, MFE-capture, giveback ratio, exit-reason
distribution, and the regime breakdowns (asset class, vol/liq bucket, gap vs no-gap, etc.).
Validation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Report:
    """One §13 headline report (per asset class / regime bucket)."""

    label: str
    n_trades: int
    median_mfe_capture: float
    median_giveback: float
    e_pi: float
    premium_bleed: float
    tail_capture: float
    exit_reason_dist: Mapping[str, float]


class Reporter:
    """Build the §13 headline + regime reports from per-trade labels."""

    def headline(self, label: str, per_trade_labels: list[list[dict]], conv: dict) -> Report:
        """Aggregate one bucket of trades into a :class:`Report` (§13)."""
        mfe = [t[-1]["mfe_capture"] for t in per_trade_labels if t]
        gb = [t[-1]["giveback_ratio"] for t in per_trade_labels if t]
        causes: dict[str, int] = {}
        for t in per_trade_labels:
            if not t:
                continue
            cause = t[-1].get("stop_cause") or "alive"
            causes[cause] = causes.get(cause, 0) + 1
        total = max(sum(causes.values()), 1)
        dist = {k: v / total for k, v in causes.items()}
        tail_capture = float(np.median(mfe)) if mfe else 0.0
        return Report(
            label=label,
            n_trades=len(per_trade_labels),
            median_mfe_capture=float(np.median(mfe)) if mfe else 0.0,
            median_giveback=float(np.median(gb)) if gb else 0.0,
            e_pi=float(conv.get("E_Pi", 0.0)),
            premium_bleed=float(conv.get("premium_bleed", 0.0)),
            tail_capture=tail_capture,
            exit_reason_dist=dist,
        )
