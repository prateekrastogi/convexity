"""Branch-coverage teeth for REGIME emit.py detector-classification paths.

The e2e real-data pass + the state-machine suite exercised the dispersion-upper / insufficient-
breadth / warn / none branches, but left the §5 drawdown-stop, the low-dispersion (d_down) and the
BTC-vol (v_up) binding-detector + reason-code branches of ``_binding_detector`` / ``_reason_codes``
untriggered. Each assertion binds one branch: it fails if that classification is removed or mis-keyed.
"""

from __future__ import annotations

from dataclasses import replace

from convexity.regime.config.resolve import resolve_config
from convexity.regime.state.emit import emit
from convexity.regime.state.machine import MachineState, WeekAlarms

_CFG = resolve_config()


def _alarms(**kw) -> WeekAlarms:
    base = dict(d_up=False, d_down=False, v_up=False, d_warn=False, stale=False,
                dd_stop=False, all_calm=False)
    base.update(kw)
    return WeekAlarms(**base)


def _machine(state: str, axes=frozenset()) -> MachineState:
    return MachineState(state=state, dwell_weeks=1, calm_streak=0, active_axes=axes)


def test_emit_drawdown_stop_binds_detector_and_reason() -> None:
    """The §5 drawdown-stop (-> HALT) surfaces binding_detector=drawdown_stop + its reason code."""
    rs = emit(_machine("HALT"), _alarms(dd_stop=True), "escalate",
              cfg=_CFG, s_stat=0.0, h_threshold=0.0, t=5)
    assert rs.binding_detector == "drawdown_stop"
    assert "drawdown_stop_triggered" in rs.reason_codes


def test_emit_low_dispersion_binds_dispersion_lower() -> None:
    """A fired lower-dispersion CUSUM names dispersion_cusum_lower + its alarm reason code."""
    rs = emit(_machine("STRESSED"), _alarms(d_down=True), "escalate",
              cfg=_CFG, s_stat=3.0, h_threshold=2.0, t=6)
    assert rs.binding_detector == "dispersion_cusum_lower"
    assert "dispersion_cusum_lower_alarm" in rs.reason_codes


def test_emit_btc_vol_binds_vol_cusum_upper() -> None:
    """A fired BTC-vol CUSUM names btc_vol_cusum_upper + its alarm reason code."""
    rs = emit(_machine("STRESSED"), _alarms(v_up=True), "escalate",
              cfg=_CFG, s_stat=4.0, h_threshold=2.5, t=7)
    assert rs.binding_detector == "btc_vol_cusum_upper"
    assert "btc_vol_cusum_upper_alarm" in rs.reason_codes


def test_emit_vol_axis_via_active_axes_only() -> None:
    """A §4.3 sequential HALT (vol axis carried in active_axes, not fresh this week) still surfaces
    btc_vol_cusum_upper — the axes-membership branch, distinct from the fresh-crossing branch."""
    rs = emit(_machine("HALT", axes=frozenset({"v_up"})), _alarms(), "hold",
              cfg=_CFG, s_stat=0.0, h_threshold=0.0, t=8)
    assert rs.binding_detector == "btc_vol_cusum_upper"
    assert "btc_vol_cusum_upper_alarm" in rs.reason_codes
    # sanity: a fresh-crossing variant classifies identically (equivariance of the two entry paths)
    fresh = emit(_machine("HALT"), _alarms(v_up=True), "hold",
                 cfg=_CFG, s_stat=0.0, h_threshold=0.0, t=8)
    assert fresh.binding_detector == rs.binding_detector


def test_emit_cfg_is_not_mutated() -> None:
    """emit reads cfg (band floor) but never mutates it (frozen boundary object)."""
    before = replace(_CFG)
    emit(_machine("WATCH"), _alarms(d_warn=True), "escalate",
         cfg=_CFG, s_stat=1.0, h_threshold=0.5, t=1)
    assert _CFG == before


def test_binding_stat_names_the_down_and_vol_axes() -> None:
    """_binding_stat returns the low-dispersion (d_down) and BTC-vol (v_up) statistic/threshold
    pairs (engine_api.py:128,130) — the two axes the dispersion-upper-only synthetic panels and the
    real-data slice never drove through the engine loop."""
    from convexity.regime.engine_api import _binding_stat

    down = _binding_stat(_alarms(d_down=True), s_d_up=0.1, s_d_down=3.3, s_v_up=0.2,
                         h_d=2.0, h_v=2.5)
    assert down == (3.3, 2.0)  # the LOWER dispersion accumulator against the dispersion threshold
    vol = _binding_stat(_alarms(v_up=True), s_d_up=0.1, s_d_down=0.2, s_v_up=4.4,
                        h_d=2.0, h_v=2.5)
    assert vol == (4.4, 2.5)  # the vol accumulator against the vol threshold
