"""Public OPS entry point used by the live loop (§3, the ``OpsEngine.on_obs`` composition surface).

Spec: OPS §3 (the public ``OpsEngine`` seam), §16 build order. This is the thin public facade the
strategy adapter / SYSTEM orchestrator composes against: build a frozen config from a SYSTEM in-control
report, construct the engine, and emit directives. OPS holds NO edge/sizing/exit parameter — its only
outputs are the {run, degrade, halt} directive + alarms + the flatten directive (Invariant PROV/P1).
"""

from __future__ import annotations

from typing import Any, Mapping

from convexity.core.types import AssetClass
from convexity.ops.brake.engine import OpsEngine
from convexity.ops.config.defaults import OpsDefaults
from convexity.ops.config.resolve import ResolvedOpsConfig, resolve_config
from convexity.ops.forensics.log import ForensicLog


def build_engine(
    *,
    system_report: Mapping[str, Mapping[str, Any]] | None = None,
    defaults: OpsDefaults | None = None,
    asset_class: AssetClass | None = None,
    blue: Mapping[str, Any] | None = None,
    log: ForensicLog | None = None,
) -> OpsEngine:
    """Construct an :class:`OpsEngine` from a SYSTEM in-control report (§3/§11.1).

    ``system_report`` is SYSTEM's validation report (defaults to the §10 fixture so OPS is testable in
    isolation). RED limits enter only through ``defaults`` (Invariant C); BLUE through ``blue`` against
    the registered grid (NOFIT). The engine is the live panel; feed it observations via its
    ``on_obs``/``on_equity``/``on_recon``/``on_heartbeat`` methods and call ``evaluate`` per bar.
    """
    cfg: ResolvedOpsConfig = resolve_config(
        defaults=defaults, system_report=system_report, asset_class=asset_class, blue=blue
    )
    return OpsEngine(cfg, log=log)


__all__ = ["build_engine", "OpsEngine", "ResolvedOpsConfig", "resolve_config"]
