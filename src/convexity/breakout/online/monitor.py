"""Online drift / stale-data / alerting metrics (BREAKOUT §7 online/monitor).

Lightweight, log-only telemetry over the live detector: realized alert rate vs the target, stale/
invalid-bar fraction, and the empirical-null bucket occupancy. Diagnostics only — they never gate or
alter a signal (BREAKOUT keeps composition decomposable; monitoring is OPS's job downstream).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.core.types import Bars, EntrySignal


@dataclass(frozen=True)
class MonitorMetrics:
    """Drift / staleness telemetry (BREAKOUT §7). Log-only, never a gate."""

    n_bars: int
    n_signals: int
    realized_alert_rate: float
    target_alert_rate: float
    alert_rate_drift: float  # realized - target
    invalid_fraction: float


def compute_metrics(
    bars: Bars, signals: list[EntrySignal], target_alert_rate: float
) -> MonitorMetrics:
    """Summarise realized vs target alert rate + data staleness (log-only telemetry)."""
    n = len(bars)
    realized = (len(signals) / n) if n else 0.0
    invalid = (
        float(np.mean(~bars.is_valid.astype(bool))) if bars.is_valid is not None else 0.0
    )
    return MonitorMetrics(
        n_bars=n,
        n_signals=len(signals),
        realized_alert_rate=realized,
        target_alert_rate=target_alert_rate,
        alert_rate_drift=realized - target_alert_rate,
        invalid_fraction=invalid,
    )
