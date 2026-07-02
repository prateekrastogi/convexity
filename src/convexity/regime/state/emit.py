"""Construct the frozen ``RegimeState`` boundary record with its P7 reason chain (REGIME.md §4, §8).

The single place a :class:`convexity.core.types.RegimeState` is built from the machine state, the
band table, and the triggering CUSUM statistic/threshold. Every ``g_t`` change carries a reason chain
(§0 P7): which detector fired, at what statistic value, against what threshold. ``g_t`` is the band
FLOOR for the state (§4.1) — the conservative live value, never interpolated.
"""

from __future__ import annotations

from convexity.core.types import RegimeDetector, RegimeState
from convexity.regime.config.resolve import ResolvedRegimeConfig
from convexity.regime.state.bands import g_t_for_state
from convexity.regime.state.machine import MachineState, WeekAlarms

ENGINE_VERSION = "regime-v1"


def _binding_detector(state: str, a: WeekAlarms, axes: frozenset[str]) -> RegimeDetector:
    """Which detector drove the current state (REGIME.md §4 reason chain, §3.3 vocabulary).

    Priority follows the alarm severity that set the state: the §5 drawdown-stop and multi-statistic
    HALT surface their strongest contributor; a single-statistic STRESSED names that statistic; WATCH
    names dispersion-warn or insufficient-breadth; HEALTHY names none (§4.1/§8.1). ``axes`` is the set
    of STILL-ACTIVE alarm axes (this week's fresh crossings unioned with prior unresolved ones, §4.3)
    so a §4.3 sequential-week HALT — where §3.1 already reset the earlier axis's accumulator and only
    the vol axis crosses THIS week — still surfaces the highest-priority contributing detector."""
    if a.dd_stop:
        return "drawdown_stop"
    if a.d_up or "d_up" in axes:
        return "dispersion_cusum_upper"
    if a.d_down or "d_down" in axes:
        return "dispersion_cusum_lower"
    if a.v_up or "v_up" in axes:
        return "btc_vol_cusum_upper"
    if a.stale:
        return "insufficient_breadth"
    if a.d_warn:
        return "dispersion_cusum_upper"
    return "none"


def _reason_codes(state: str, a: WeekAlarms, axes: frozenset[str]) -> tuple[str, ...]:
    """Decomposable reason chain for this week's state (REGIME.md §0 P7, §4.3 worked example).

    ``axes`` is the still-active alarm-axis set (fresh ∪ unresolved-prior, §4.3). At a §4.3
    sequential-week HALT the reason chain "lists both triggering detectors" even though §3.1 already
    reset the earlier axis's accumulator, so the alarm codes are emitted for the UNION of active axes,
    not just this week's fresh crossings."""
    codes: list[str] = [f"regime_{state.lower()}"]
    if a.dd_stop:
        codes.append("drawdown_stop_triggered")
    if a.d_up or "d_up" in axes:
        codes.append("dispersion_cusum_upper_alarm")
    if a.d_down or "d_down" in axes:
        codes.append("dispersion_cusum_lower_alarm")
    if a.v_up or "v_up" in axes:
        codes.append("btc_vol_cusum_upper_alarm")
    if a.stale and not (a.d_up or a.d_down or a.v_up or axes):
        codes.append("insufficient_breadth")
    if a.d_warn and not (a.d_up or a.d_down or "d_up" in axes or "d_down" in axes):
        codes.append("dispersion_cusum_warn")
    return tuple(codes)


def emit(
    machine: MachineState,
    alarms: WeekAlarms,
    transition: str,
    *,
    cfg: ResolvedRegimeConfig,
    s_stat: float,
    h_threshold: float,
    t: int,
) -> RegimeState:
    """Build the frozen ``RegimeState`` for week ``t`` (REGIME.md §4, §8).

    ``g_t`` is the band floor for ``machine.state`` (§4.1). ``s_stat``/``h_threshold`` are the
    triggering CUSUM value and the decision interval it crossed (0.0 when HEALTHY / no alarm). ``t``
    indexes the weekly Friday-close rebalance grid (§0 P1 — ``as_of`` is a week index, not a bar
    count)."""
    axes = machine.active_axes
    return RegimeState(
        g_t=g_t_for_state(machine.state, cfg),
        state=machine.state,
        binding_detector=_binding_detector(machine.state, alarms, axes),
        s_stat=float(s_stat),
        h_threshold=float(h_threshold),
        dwell_weeks=machine.dwell_weeks,
        transition=transition,  # type: ignore[arg-type]  # narrowed to escalate/deescalate/hold
        reason_codes=_reason_codes(machine.state, alarms, axes),
        as_of=int(t),
        engine_version=ENGINE_VERSION,
    )
