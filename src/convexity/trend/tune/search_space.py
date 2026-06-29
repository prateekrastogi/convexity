"""Finite BLUE search grids (TREND §15 search_space).

Re-exports the §15 finite grids from config/tuning so tune/ has a single import surface. BLUE values
are finite candidate presets searched at the CLASS level only (§4 BLUE) — never free optimization,
never per-symbol on realized P&L.
"""

from __future__ import annotations

from convexity.trend.config.tuning import (  # noqa: F401  (re-export)
    ARL_TARGETS,
    DETECTOR_KINDS,
    HAZARD_MODES,
    KALMAN_CLASS_PRESETS,
    TRAIL_MULTIPLIER_PRESETS,
    TRAIL_PRESETS,
    VOL_METHODS,
)


def grid_cardinality() -> int:
    """Total BLUE grid cardinality fed to SYSTEM's trial ledger (§14 / SYSTEM DSR)."""
    return (
        len(TRAIL_PRESETS)
        * len(DETECTOR_KINDS)
        * len(ARL_TARGETS)
        * len(VOL_METHODS)
        * len(KALMAN_CLASS_PRESETS)
        * len(HAZARD_MODES)
    )
