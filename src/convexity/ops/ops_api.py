"""OPS public entry point: ``OpsEngine.on_obs() -> OpsDirective`` (``docs/specs/OPS.md`` sec 8).

sec 8 names ``ops_api.py`` as the public entry point re-exporting the composing ``OpsEngine`` and the
boundary objects it produces/consumes. A caller (the live loop, or EXECUTION.md sec 7.2 routing its
realized ``IS_total`` through the monitoring seam) imports from here and never reaches into the
``brake``/``streams``/``detectors`` internals.
"""

from __future__ import annotations

from convexity.core.types import Alarm, OpsDirective, OpsState
from convexity.ops.brake.engine import OpsEngine
from convexity.ops.config.defaults import DEFAULTS, OpsDefaults
from convexity.ops.types import OpsObs, RebalanceFill

__all__ = [
    "OpsEngine",
    "OpsObs",
    "RebalanceFill",
    "OpsDirective",
    "Alarm",
    "OpsState",
    "OpsDefaults",
    "DEFAULTS",
]
