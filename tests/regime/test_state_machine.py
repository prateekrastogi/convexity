"""HEALTHY/WATCH/STRESSED/HALT state machine tests (REGIME.md §4).

Pins the §4.1 state->g_t band table, the §4.1 trigger conditions (one alarm -> STRESSED, two-or-more
-> HALT, warn/stale -> WATCH), and the §4.2 causal re-arm hysteresis (immediate escalation,
one-level-at-a-time de-escalation gated by n_confirm calm weeks; a single good week never clears a
stress state). Also §8.3: severity does NOT escalate on single-alarm duration alone.
"""

import pytest

from convexity.regime.config.defaults import RegimeDefaults
from convexity.regime.config.resolve import resolve_config
from convexity.regime.state import bands, machine
from convexity.regime.state.emit import emit


def _alarms(*, d_up=False, d_down=False, v_up=False, d_warn=False, stale=False, dd_stop=False):
    all_calm = (not d_up) and (not d_down) and (not v_up) and (not d_warn)
    return machine.WeekAlarms(
        d_up=d_up, d_down=d_down, v_up=v_up, d_warn=d_warn,
        stale=stale, dd_stop=dd_stop, all_calm=all_calm,
    )


# --------------------------------------------------------------------------- §4.1 bands

def test_g_t_band_floors_per_state():
    """§4.1: HEALTHY=1.00, WATCH=0.75, STRESSED=0.40, HALT=0.00 (HALT floors at EXACTLY 0)."""
    cfg = resolve_config()
    assert bands.g_t_for_state("HEALTHY", cfg) == 1.00
    assert bands.g_t_for_state("WATCH", cfg) == 0.75
    assert bands.g_t_for_state("STRESSED", cfg) == 0.40
    assert bands.g_t_for_state("HALT", cfg) == 0.00  # not the pre-pivot .25 floor


# --------------------------------------------------------------------------- §4.1 triggers

def test_one_alarm_is_stressed():
    """§4.1: exactly one of {S_D^+, S_D^-, S_V^+} in alarm -> STRESSED."""
    st, tr = machine.step(machine.initial_state(), _alarms(d_up=True), n_confirm=2)
    assert st.state == "STRESSED" and tr == "escalate"


def test_two_alarms_is_halt():
    """§4.1: two-or-more simultaneous alarms -> HALT (the cross-statistic co-occurrence, §3.3)."""
    st, _ = machine.step(machine.initial_state(), _alarms(d_up=True, v_up=True), n_confirm=2)
    assert st.state == "HALT"


def test_sequential_week_distinct_axis_alarm_escalates_stressed_to_halt():
    """§4.3 (the sequential-week worked example): week t S_D^+ alarms -> STRESSED; §3.1 immediately
    resets that accumulator to 0, so in week t+1 S_D^+ is NOT re-firing. If a DIFFERENT axis (S_V^+)
    alarms in week t+1 while the dispersion stress is still UNRESOLVED, the two statistics are
    co-occurring in alarm at the STATE layer and the machine escalates to HALT — exactly §4.3's
    'the following week ... two statistics are now simultaneously in alarm ... escalates to HALT'.

    Before the active-axis fix this was UNREACHABLE: with s_d_up reset to 0, week t+1 saw only one
    fresh alarm (v_up) -> n_full=1 -> STRESSED, never HALT. This test fails on that old behaviour."""
    st = machine.initial_state()
    st, tr = machine.step(st, _alarms(d_up=True), n_confirm=2)     # week t: dispersion spike
    assert st.state == "STRESSED" and tr == "escalate"
    assert st.active_axes == frozenset({"d_up"})
    # week t+1: dispersion accumulator was reset (d_up False now); ONLY vol alarms this week.
    st, tr = machine.step(st, _alarms(v_up=True), n_confirm=2)
    assert st.state == "HALT" and tr == "escalate"                 # co-occurrence -> HALT (§4.3)
    assert st.active_axes == frozenset({"d_up", "v_up"})


def test_prolonged_single_axis_alarm_after_reset_stays_stressed_not_halt():
    """§8.3 guard on the §4.3 fix: the SAME axis re-firing week after week (each time reset by §3.1)
    keeps active_axes a SINGLETON -> STRESSED forever, never HALT on duration. Only a DISTINCT second
    axis escalates (contrast the test above)."""
    st = machine.initial_state()
    st, _ = machine.step(st, _alarms(d_up=True), n_confirm=2)       # STRESSED, active {d_up}
    for _ in range(5):
        st, tr = machine.step(st, _alarms(d_up=True), n_confirm=2)  # same axis re-fires
        assert st.state == "STRESSED" and tr == "hold"
        assert st.active_axes == frozenset({"d_up"})


def test_rearm_clears_active_axes_so_new_single_alarm_is_only_stressed():
    """§4.2: after a full re-arm (n_confirm calm weeks de-escalate out of STRESSED), the active-axis
    set is CLEARED — a subsequent lone alarm is STRESSED again, not an instant HALT (the prior stress
    is resolved, so it does not co-occur with the new one)."""
    st = machine.initial_state()
    st, _ = machine.step(st, _alarms(d_up=True), n_confirm=2)       # STRESSED, active {d_up}
    st, _ = machine.step(st, _alarms(), n_confirm=2)                # calm 1, hold (active persists)
    st, tr = machine.step(st, _alarms(), n_confirm=2)              # calm 2 -> de-escalate to WATCH
    assert st.state == "WATCH" and tr == "deescalate" and st.active_axes == frozenset()
    st, _ = machine.step(st, _alarms(), n_confirm=2)               # calm -> hold at WATCH (streak 1)
    st, _ = machine.step(st, _alarms(), n_confirm=2)               # calm -> WATCH->HEALTHY
    assert st.state == "HEALTHY" and st.active_axes == frozenset()
    # a fresh lone vol alarm is STRESSED, not HALT — no unresolved prior axis to co-occur with.
    st, tr = machine.step(st, _alarms(v_up=True), n_confirm=2)
    assert st.state == "STRESSED" and tr == "escalate"


def test_drawdown_stop_is_halt():
    """§4.1: the independent §5 drawdown-stop trigger -> HALT."""
    st, _ = machine.step(machine.initial_state(), _alarms(dd_stop=True), n_confirm=2)
    assert st.state == "HALT"


def test_warn_or_stale_is_watch():
    """§4.1/§8.1/§0 P6: a warn-crossing OR stale/insufficient data -> WATCH (never HEALTHY)."""
    st_w, _ = machine.step(machine.initial_state(), _alarms(d_warn=True), n_confirm=2)
    assert st_w.state == "WATCH"
    st_s, _ = machine.step(machine.initial_state(), _alarms(stale=True), n_confirm=2)
    assert st_s.state == "WATCH"


# --------------------------------------------------------------------------- §4.2 hysteresis

def test_escalation_is_immediate():
    """§4.1: escalation to a higher severity happens on the alarm week itself."""
    st = machine.initial_state()
    st, tr = machine.step(st, _alarms(v_up=True), n_confirm=2)
    assert st.state == "STRESSED" and tr == "escalate"


def test_single_good_week_never_clears_stressed():
    """§4.2: a STRESSED state does NOT clear on one calm week; needs n_confirm consecutive."""
    st = machine.MachineState(state="STRESSED", dwell_weeks=3, calm_streak=0)
    st, tr = machine.step(st, _alarms(), n_confirm=2)  # one calm week only
    assert st.state == "STRESSED" and tr == "hold" and st.calm_streak == 1


def test_deescalates_one_level_after_n_confirm_calm_weeks():
    """§4.2: STRESSED->WATCH after n_confirm calm weeks, one level at a time (not straight to HEALTHY)."""
    st = machine.MachineState(state="STRESSED", dwell_weeks=3, calm_streak=0)
    st, _ = machine.step(st, _alarms(), n_confirm=2)   # calm streak 1, hold
    st, tr = machine.step(st, _alarms(), n_confirm=2)  # calm streak 2 -> de-escalate one level
    assert st.state == "WATCH" and tr == "deescalate"


def test_halt_deescalates_to_stressed_not_healthy():
    """§4.2 one-level-at-a-time: HALT re-arms to STRESSED, never jumping to HEALTHY."""
    st = machine.MachineState(state="HALT", dwell_weeks=1, calm_streak=1)
    st, tr = machine.step(st, _alarms(), n_confirm=2)  # streak reaches 2 -> down one level
    assert st.state == "STRESSED" and tr == "deescalate"


def test_duration_alone_does_not_escalate(monkeypatch=None):
    """§8.3: a single alarm staying in prolonged alarm holds STRESSED, never auto-escalates to HALT
    on duration alone — only co-occurrence of independent signals escalates."""
    st = machine.MachineState(state="STRESSED", dwell_weeks=10, calm_streak=0)
    st, tr = machine.step(st, _alarms(d_up=True), n_confirm=2)  # still just one alarm, week 11
    assert st.state == "STRESSED" and tr == "hold" and st.dwell_weeks == 11


# --------------------------------------------------------------------------- emit / reason chain (§0 P7)

def test_emit_carries_reason_chain_and_band_g_t():
    """§0 P7 / §4.3: the emitted RegimeState carries g_t=band floor + a detector reason chain."""
    cfg = resolve_config()
    a = _alarms(d_up=True, v_up=True)  # HALT
    st, tr = machine.step(machine.initial_state(), a, n_confirm=2)
    rs = emit(st, a, tr, cfg=cfg, s_stat=0.0544, h_threshold=0.05419, t=42)
    assert rs.state == "HALT" and rs.g_t == 0.0
    assert rs.binding_detector == "dispersion_cusum_upper"  # highest-priority contributor
    assert "dispersion_cusum_upper_alarm" in rs.reason_codes
    assert "btc_vol_cusum_upper_alarm" in rs.reason_codes
    assert rs.as_of == 42 and rs.h_threshold == pytest.approx(0.05419)


def test_defaults_are_the_spec_red_constants():
    """§3.4/§4.1/§4.2/§6: RED constants match the spec's stated values."""
    d = RegimeDefaults()
    assert d.delta_target == 1.5 and d.arl_0_target == 104.0  # §3.5
    assert d.h_warn_frac == 0.6 and d.n_confirm == 2          # §4.1/§4.2
    assert d.n_min_dispersion == 15                            # §8.1
    assert d.n_weeks_min_cpcv == 156                           # §5.4
    assert (d.g_healthy, d.g_watch, d.g_stressed, d.g_halt) == (1.00, 0.75, 0.40, 0.00)
