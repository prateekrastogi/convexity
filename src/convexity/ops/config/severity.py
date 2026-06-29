"""AMBER severity tiering + paging routes + dedup windows (OPS §4 AMBER, §8.1).

Spec: OPS §8.1 (the response ladder LOG<PAGE<DEGRADE<HALT), §8.2 (deterministic dedup). Severity is a
property of the *trigger* (§7), tabulated and AMBER-routed, not re-decided per alarm (Invariant
ALARM). This module pins the trigger→severity table and the per-route metadata; the router itself
lives in ``brake/alarms.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.types import Severity, TriggerName

# The §7 trigger → §8.1 severity table. ``cost_divergence`` pages+degrades at h; its *escalated* form
# halts at h2. The DEGRADE-class alarms (vol/edge/funding) page; the hard binary triggers halt.
TRIGGER_SEVERITY: dict[TriggerName, Severity] = {
    "max_drawdown_backstop": Severity.HALT,        # 7.1
    "vol_regime_break": Severity.DEGRADE,          # 7.2
    "cost_divergence": Severity.DEGRADE,           # 7.3
    "cost_divergence_escalated": Severity.HALT,    # 7.3 → h2
    "fill_rate_collapse": Severity.DEGRADE,        # 7.4 (→ HALT if an exit cannot fill)
    "reconciliation_mismatch": Severity.HALT,      # 7.5
    "data_staleness": Severity.DEGRADE,            # 7.6 stale feed (HALT on heartbeat/clock-skew)
    "edge_decay": Severity.DEGRADE,                # 7.7
    "funding_blowout": Severity.DEGRADE,           # 7.8
}


@dataclass(frozen=True)
class SeverityConfig:
    """AMBER routing metadata: the dedup window + the page-route name per severity (§8.2).

    ``dedup_window`` coalesces repeated breaches of one ``(stream, detector, severity)`` fingerprint;
    a *new* fingerprint or an *escalation* always pages (§8.2). ``halt_route`` is the highest-priority
    page channel for HALT / ``HALT-unconfirmed`` (§9.1c).
    """

    dedup_window: int = 50
    page_route: str = "ops-page"
    halt_route: str = "ops-halt-priority"


DEFAULT_SEVERITY_CONFIG = SeverityConfig()


def severity_of(trigger: TriggerName) -> Severity:
    """The tabulated §8.1 severity for a trigger (Invariant ALARM: trigger-defined, not per-alarm)."""
    return TRIGGER_SEVERITY[trigger]


__all__ = ["TRIGGER_SEVERITY", "SeverityConfig", "DEFAULT_SEVERITY_CONFIG", "severity_of"]
