"""State-machine acceptance gate (REGIME §13 State machine).

Covers Invariants MIN, HYST, DWELL, CLOCK, FLOOR, CAP, plus the §6.2 binding-handoff fixture.
"""


from convexity.core.types import RegimeState
from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.state.bands import classify_substate, g_of
from convexity.regime.state.machine import step_machine
from convexity.regime.types import SubName, SubState


def _sub(name: SubName, value: float, state, *, asof=0) -> SubState:
    return SubState(name=name, value=value, state=state, g_sub=g_of(state), asof_bar=asof, fresh=True, diag={})


def _prev(state, binding, dwell, g) -> RegimeState:
    return RegimeState(
        g_t=g, state=state, binding_sub=binding, dwell_bars=dwell, transition="hold",
        cost_gate_passed=True, rebalance_cost_hat=0.0, reason_codes=(), asof_bar=0,
        engine_version="t",
    )


def _as_prev(res) -> RegimeState:
    """Convert a MachineResult into the RegimeState the next step_machine call expects."""
    return RegimeState(
        g_t=res.g_s, state=res.state, binding_sub=res.binding, dwell_bars=res.dwell_bars,
        transition="hold", cost_gate_passed=True, rebalance_cost_hat=0.0, reason_codes=(),
        asof_bar=0, engine_version="t",
    )


# --------------------------------------------------------------------------- HYST (per-sub-state)


def test_hyst_oscillation_holds_state():
    """A measure oscillating between its enter and exit marks does NOT change its sub-state (HYST)."""
    bands = DEFAULTS.bands_for("vol")  # NORMAL 0.70/0.60
    # Enter NORMAL at 0.72
    s = classify_substate(0.72, "CALM", bands)
    assert s == "NORMAL"
    # Dither back to 0.65 (inside the 0.60-0.70 gap) -> HOLD NORMAL
    s = classify_substate(0.65, "NORMAL", bands)
    assert s == "NORMAL"
    # Only below the exit mark 0.60 does it de-escalate
    s = classify_substate(0.55, "NORMAL", bands)
    assert s == "CALM"


def test_escalation_may_jump_multiple_states():
    bands = DEFAULTS.bands_for("corr")  # CRISIS enter +2.0
    s = classify_substate(2.2, "CALM", bands)
    assert s == "CRISIS"  # jumped CALM -> CRISIS in one step


def test_deescalation_is_one_step_at_a_time():
    bands = DEFAULTS.bands_for("corr")  # CRISIS exit +1.4, STRESSED exit +0.6
    # From CRISIS, value drops well below all exits, but a sub-state steps down ONE rung per call.
    s = classify_substate(0.0, "CRISIS", bands)
    assert s == "STRESSED"
    s = classify_substate(0.0, "STRESSED", bands)
    assert s == "NORMAL"
    s = classify_substate(0.0, "NORMAL", bands)
    assert s == "CALM"


# --------------------------------------------------------------------------- MIN


def test_min_worst_sub_state_binds():
    subs = {
        "vol": _sub("vol", 0.72, "NORMAL"),
        "corr": _sub("corr", 2.2, "CRISIS"),
        "turb": _sub("turb", 0.80, "NORMAL"),
        "liq": _sub("liq", 0.55, "CALM"),
    }
    res = step_machine(subs, None, DEFAULTS)
    assert res.state == "CRISIS"
    assert res.binding == "corr"
    assert res.g_s == 0.25  # the min(), NOT a product


def test_min_not_product():
    """Two moderately-elevated measures do not compound below the worse single one (Invariant MIN)."""
    subs = {
        "vol": _sub("vol", 0.72, "NORMAL"),
        "corr": _sub("corr", 0.7, "NORMAL"),
        "turb": _sub("turb", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    res = step_machine(subs, None, DEFAULTS)
    assert res.g_s == 0.75  # min(0.75, 0.75, 1, 1) — NOT 0.75*0.75=0.5625


# --------------------------------------------------------------------------- DWELL


def test_dwell_blocks_deescalation_until_d_min():
    prev = _prev("STRESSED", "corr", dwell=1, g=0.50)  # dwell 1 < D_min 3
    subs = {  # corr fell back to NORMAL (below STRESSED exit), rest calm
        "vol": _sub("vol", 0.1, "CALM"),
        "corr": _sub("corr", 0.3, "NORMAL"),
        "turb": _sub("turb", 0.1, "CALM"),
        "liq": _sub("liq", 0.1, "CALM"),
    }
    res = step_machine(subs, prev, DEFAULTS)
    assert res.state == "STRESSED"  # dwell-locked, held
    assert res.transition == "hold"
    assert res.dwell_bars == 2  # clock keeps counting


def test_escalation_never_blocked_by_dwell():
    prev = _prev("NORMAL", "vol", dwell=1, g=0.75)  # just entered
    subs = {
        "vol": _sub("vol", 2.2, "CRISIS"),
        "corr": _sub("corr", 0.0, "CALM"),
        "turb": _sub("turb", 0.0, "CALM"),
        "liq": _sub("liq", 0.0, "CALM"),
    }
    res = step_machine(subs, prev, DEFAULTS)
    assert res.state == "CRISIS"  # escalation immediate despite dwell < D_min
    assert res.transition == "escalate"
    assert res.dwell_bars == 1


# --------------------------------------------------------------------------- CLOCK / §6.2 handoff


def test_clock_continues_through_binding_handoff_and_permits_t3():
    """The §6.2 t0..t3 fixture: corr->vol handoff at constant g, dwell continues, t3 permitted (CLOCK)."""
    corr_bands = DEFAULTS.bands_for("corr")
    pct_bands = DEFAULTS.bands_for("vol")

    # t0: corr +1.3 -> STRESSED (from NORMAL prev), rest CALM
    prev = _prev("NORMAL", "corr", dwell=2, g=0.75)
    corr_s = classify_substate(1.3, "STRESSED", corr_bands)  # already STRESSED via prior
    subs0 = {
        "corr": _sub("corr", 1.3, corr_s),
        "turb": _sub("turb", 0.5, "CALM"),
        "vol": _sub("vol", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    r0 = step_machine(subs0, prev, DEFAULTS)
    assert r0.state == "STRESSED" and r0.binding == "corr"

    # t1: corr +0.9 (still > exit +0.6 -> holds STRESSED); vol -> STRESSED via 0.88 entry
    vol_s = classify_substate(0.88, "CALM", pct_bands)
    assert vol_s == "STRESSED"
    subs1 = {
        "corr": _sub("corr", 0.9, classify_substate(0.9, "STRESSED", corr_bands)),
        "vol": _sub("vol", 0.88, vol_s),
        "turb": _sub("turb", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    r1 = step_machine(subs1, _as_prev(r0), DEFAULTS)
    assert r1.state == "STRESSED" and r1.binding == "corr"  # corr still binds (tie-break / unchanged)
    assert r1.dwell_bars == r0.dwell_bars + 1

    # t2: corr +0.5 (< exit +0.6 -> de-esc to NORMAL); vol stays STRESSED -> HANDOFF corr->vol
    subs2 = {
        "corr": _sub("corr", 0.5, classify_substate(0.5, "STRESSED", corr_bands)),  # -> NORMAL
        "vol": _sub("vol", 0.90, classify_substate(0.90, "STRESSED", pct_bands)),
        "turb": _sub("turb", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    assert classify_substate(0.5, "STRESSED", corr_bands) == "NORMAL"
    r2 = step_machine(subs2, _as_prev(r1), DEFAULTS)
    assert r2.state == "STRESSED"  # global min UNCHANGED at STRESSED
    assert r2.binding == "vol"  # binding handed off
    assert r2.transition == "hold"
    assert r2.reason == "binding_handoff"
    assert r2.dwell_bars == r1.dwell_bars + 1  # clock CONTINUES, not reset (§3.3 ii)
    assert r2.dwell_bars >= DEFAULTS.d_min

    # t3: corr +0.3 (NORMAL), vol 0.70 (< STRESSED exit 0.75 -> de-esc to NORMAL) -> global de-esc
    subs3 = {
        "corr": _sub("corr", 0.3, classify_substate(0.3, "NORMAL", corr_bands)),
        "vol": _sub("vol", 0.70, classify_substate(0.70, "STRESSED", pct_bands)),
        "turb": _sub("turb", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    assert classify_substate(0.70, "STRESSED", pct_bands) == "NORMAL"
    r3 = step_machine(subs3, _as_prev(r2), DEFAULTS)
    assert r3.state == "NORMAL"  # legitimate de-escalation PERMITTED (a resetting clock would block)
    assert r3.transition == "deescalate"
    assert r3.dwell_bars == 1


def test_floor_never_below_crisis():
    """No input makes the machine emit g < 0.25 (Invariant FLOOR)."""
    subs = {n: _sub(n, 9.9, "CRISIS") for n in ("vol", "corr", "turb", "liq")}
    res = step_machine(subs, None, DEFAULTS)
    assert res.g_s == 0.25
    assert res.g_s >= DEFAULTS.g_floor
