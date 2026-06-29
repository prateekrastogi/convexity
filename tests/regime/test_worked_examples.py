"""Worked-example regression fixtures — reproduce §6.1/§6.2/§6.3 bit-for-bit (Invariant WE)."""

import pytest

from convexity.core.types import RegimeState
from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.measures.absorption import absorption_ratio, standardized_shift, top_k
from convexity.regime.state.bands import classify_substate, g_of
from convexity.regime.state.cost_gate import cost_gate
from convexity.regime.state.emit import emit_state
from convexity.regime.state.machine import step_machine
from convexity.regime.types import BookState, SubName, SubState


def _sub(name: SubName, value: float, state) -> SubState:
    return SubState(name=name, value=value, state=state, g_sub=g_of(state), asof_bar=0, fresh=True, diag={})


def test_6_1_correlation_spike_cuts_the_book():
    """§6.1: n=80, k=16, AR_t=0.750, delta_AR=+2.20, g_t=0.25 binding corr, gross 750->250 (WE)."""
    # Step: AR_t via trace identity. Σλ_{1..16}=60, denominator n_t=80.
    n_t = 80
    k = top_k(0.20, n_t)
    assert k == 16
    import numpy as np

    eig = np.zeros(n_t)
    eig[:16] = 60.0 / 16.0  # any spectrum summing to 60 in the top-16; AR is the SUM, basis-invariant
    ar_t = absorption_ratio(eig, n_t, k)
    assert ar_t == pytest.approx(0.750)

    delta_ar = standardized_shift(0.710, 0.600, 0.050)
    assert delta_ar == pytest.approx(2.20, abs=1e-9)

    # Sub-states (§6.1 Step 1)
    subs = {
        "corr": _sub("corr", delta_ar, classify_substate(delta_ar, "NORMAL", DEFAULTS.bands_for("corr"))),
        "vol": _sub("vol", 0.72, classify_substate(0.72, "CALM", DEFAULTS.bands_for("vol"))),
        "turb": _sub("turb", 0.80, classify_substate(0.80, "CALM", DEFAULTS.bands_for("turb"))),
        "liq": _sub("liq", 0.55, classify_substate(0.55, "CALM", DEFAULTS.bands_for("liq"))),
    }
    assert subs["corr"].state == "CRISIS"
    assert subs["vol"].state == "NORMAL"
    assert subs["turb"].state == "NORMAL"
    assert subs["liq"].state == "CALM"

    # prev NORMAL, dwell 5
    prev = RegimeState(
        g_t=0.75, state="NORMAL", binding_sub="vol", dwell_bars=5, transition="hold",
        cost_gate_passed=True, rebalance_cost_hat=0.0, reason_codes=(), asof_bar=0, engine_version="t",
    )
    machine = step_machine(subs, prev, DEFAULTS)
    assert machine.state == "CRISIS"
    assert machine.binding == "corr"
    assert machine.g_s == 0.25
    assert machine.transition == "escalate"

    # Step 4 — cost gate. book_gross_at_full=1000, current_gross=750.
    book = BookState(book_gross_at_full=1000.0, current_gross=750.0)
    gate = cost_gate(g_prev=0.75, g_new=0.25, transition="escalate", book=book, cfg=DEFAULTS)
    assert gate.passed is True
    assert gate.delta_gross == pytest.approx(500.0)
    assert 0.10 * 750.0 == pytest.approx(75.0)
    assert gate.delta_gross >= 75.0
    # Step 6 — ~20 bps one-way on 500 -> ~1.00 USDT
    assert gate.rebalance_cost_hat == pytest.approx(1.00, abs=1e-9)

    out = emit_state(
        machine=machine, gate=gate, g_prev=0.75, binding_override=None, cfg=DEFAULTS,
        asof_bar=0, engine_version="t",
    )
    assert out.g_t == 0.25  # single gross channel -> realized gross 0.25*1000 = 250, NOT 62.50
    assert out.state == "CRISIS"
    assert out.binding_sub == "corr"
    target_gross = out.g_t * book.book_gross_at_full
    assert target_gross == pytest.approx(250.0)
    assert target_gross != pytest.approx(0.25 ** 2 * 1000.0)  # NODOUBLE2: not g_t^2*full=62.50


def test_6_1_hysteresis_tail_blocks_snapback():
    """Next bar delta_AR=+1.5 is NOT below the CRISIS exit +1.4 AND dwell reset -> stays at 0.25."""
    prev = RegimeState(
        g_t=0.25, state="CRISIS", binding_sub="corr", dwell_bars=1, transition="escalate",
        cost_gate_passed=True, rebalance_cost_hat=1.0, reason_codes=(), asof_bar=0, engine_version="t",
    )
    corr_bands = DEFAULTS.bands_for("corr")
    # +1.5 > exit +1.4 -> corr sub-state holds CRISIS
    corr_state = classify_substate(1.5, "CRISIS", corr_bands)
    assert corr_state == "CRISIS"
    subs = {
        "corr": _sub("corr", 1.5, corr_state),
        "vol": _sub("vol", 0.1, "CALM"),
        "turb": _sub("turb", 0.1, "CALM"),
        "liq": _sub("liq", 0.1, "CALM"),
    }
    res = step_machine(subs, prev, DEFAULTS)
    assert res.state == "CRISIS"  # no snapback


def test_6_3_drifted_gross_deadband_defers():
    """§6.3: drifted current_gross=120, book_gross_at_full=160 -> STRESSED->NORMAL DEFERRED (WE)."""
    book = BookState(book_gross_at_full=160.0, current_gross=120.0)
    gate = cost_gate(g_prev=0.50, g_new=0.75, transition="deescalate", book=book, cfg=DEFAULTS)
    assert gate.delta_gross == pytest.approx(0.0)
    assert gate.passed is False
    assert gate.reason == "regime_change_cost_gated"
    assert gate.g_emit == 0.50
