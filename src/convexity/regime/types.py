"""REGIME's internal frozen dataclasses (REGIME.md §9 ``types.py``).

The cross-pillar boundary object ``RegimeState`` lives in ``convexity.core.types`` (the single shared
core, migrated to this spec's §4/§8 contract). This module holds REGIME-PRIVATE state that never
crosses a pillar boundary: the four-CUSUM live calibration bundle and the per-week statistic snapshot
the engine threads through the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.regime.cusum import ARLCalibration


@dataclass(frozen=True)
class RegimeCalibration:
    """The frozen live calibration for both monitored statistics (REGIME.md §3.4, §6).

    One :class:`ARLCalibration` per statistic (``dispersion`` = ``D_t`` §2.2, ``btc_vol`` = ``V_t``
    §2.3), each fit ONCE offline on the frozen in-control training window and FROZEN for live use
    (§0 P4 — no online re-estimation). Both share the RED ``h_warn_frac``/``n_confirm`` governance via
    the resolved config; only the TEAL moments and RED-derived ``(k, h)`` differ per statistic."""

    dispersion: ARLCalibration
    btc_vol: ARLCalibration


@dataclass(frozen=True)
class WeekStatistics:
    """One week's computed statistics + CUSUM levels (REGIME.md §2-§4), for attribution/telemetry.

    ``d_t``/``v_t`` are the raw weekly statistics (NaN when the week is stale/insufficient-breadth,
    §8.1); ``s_d_up``/``s_d_down``/``s_v_up`` are the current CUSUM accumulator levels; ``n_days_used``
    is how many daily-dispersion values cleared the breadth floor (§8.1)."""

    d_t: float
    v_t: float
    s_d_up: float
    s_d_down: float
    s_v_up: float
    n_days_used: int
