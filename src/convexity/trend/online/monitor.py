"""LiveMonitor — drift / calibration / data-health monitor (TREND §11 online/monitor, log-only).

Aggregates per-bar diagnostics into running health stats (jump-flag rate, vol-quality flags, detector
margin, cap-binding fraction). Log-only (Inv F): it NEVER feeds a decision. The engine/strategy logs
these; OPS may consume them downstream, but they cannot change the action here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.trend.types import ActionDecision, BeliefState


@dataclass
class MonitorStats:
    n: int = 0
    n_cap_binding: int = 0
    n_ohlc_poor: int = 0
    n_exits: int = 0
    exit_causes: dict = field(default_factory=dict)


class LiveMonitor:
    """Running data-health/calibration monitor (§11, log-only)."""

    def __init__(self) -> None:
        self.stats = MonitorStats()

    def observe(self, belief: BeliefState, decision: ActionDecision) -> None:
        s = self.stats
        s.n += 1
        if belief.horizon.cap_binding_warning:
            s.n_cap_binding += 1
        if belief.diag.vol_quality_flag in ("ohlc_poor", "warmup", "nonfinite_fallback"):
            s.n_ohlc_poor += 1
        if decision.action == "exit":
            s.n_exits += 1
            cause = decision.stop_cause or "unknown"
            s.exit_causes[cause] = s.exit_causes.get(cause, 0) + 1

    def health(self) -> dict:
        s = self.stats
        return {
            "n": s.n,
            "cap_binding_frac": s.n_cap_binding / s.n if s.n else 0.0,
            "ohlc_poor_frac": s.n_ohlc_poor / s.n if s.n else 0.0,
            "exit_causes": dict(s.exit_causes),
        }
