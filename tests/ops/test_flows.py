"""External capital flows (OPS.md sec 15): TWR stripping + the flow-event risk check.

The load-bearing property: a deposit/withdrawal contributes EXACTLY zero to the return every
detector consumes — the false-HALT / false-promotion protection, demonstrated end-to-end against
the real whiten + CUSUM chain."""

from __future__ import annotations

import pytest

from convexity.ops.config.defaults import OpsDefaults
from convexity.ops.detectors.cusum import CUSUM
from convexity.ops.streams.flows import FlowEvent, flow_risk_check, twr_weekly_return
from convexity.ops.streams.whiten import whiten

D = OpsDefaults()


# --------------------------------------------------------------------------- TWR exactness

def test_no_flows_reduces_to_raw_return_exactly():
    assert twr_weekly_return(100.0, 103.0) == pytest.approx(0.03, abs=1e-15)


def test_deposit_with_flat_prices_is_exactly_zero_return():
    # raw equity return would be +50% — the corruption this module strips.
    f = FlowEvent(ts=10, amount=50.0, equity_before=100.0)
    assert twr_weekly_return(100.0, 150.0, (f,)) == pytest.approx(0.0, abs=1e-15)


def test_withdrawal_with_flat_prices_is_exactly_zero_return():
    # raw equity return would be −30% — the false-HALT scenario.
    f = FlowEvent(ts=10, amount=-30.0, equity_before=100.0)
    assert twr_weekly_return(100.0, 70.0, (f,)) == pytest.approx(0.0, abs=1e-15)


def test_multi_flow_linking_matches_hand_computation():
    # start 100 -> +2% to 102; deposit 48 (equity 150); -1% to 148.5; withdraw 50 (98.5);
    # +3.0456...% to 101.5. TWR = 1.02 * (148.5/150) * (101.5/98.5) - 1.
    flows = (FlowEvent(ts=2, amount=48.0, equity_before=102.0),
             FlowEvent(ts=5, amount=-50.0, equity_before=148.5))
    got = twr_weekly_return(100.0, 101.5, flows)
    want = 1.02 * (148.5 / 150.0) * (101.5 / 98.5) - 1.0
    assert got == pytest.approx(want, abs=1e-15)


def test_trading_return_survives_the_flow_exactly():
    # -10% trading loss then a deposit doubling equity: TWR must report exactly -10%.
    f = FlowEvent(ts=3, amount=90.0, equity_before=90.0)
    assert twr_weekly_return(100.0, 180.0, (f,)) == pytest.approx(-0.10, abs=1e-15)


def test_flow_window_and_ordering_guards():
    f = FlowEvent(ts=10, amount=10.0, equity_before=100.0)
    with pytest.raises(ValueError, match="before the week start"):
        twr_weekly_return(100.0, 110.0, (f,), ts_start=10, ts_end=20)
    with pytest.raises(ValueError, match="after the week end"):
        twr_weekly_return(100.0, 110.0, (f,), ts_start=0, ts_end=9)
    unordered = (FlowEvent(ts=8, amount=10.0, equity_before=100.0),
                 FlowEvent(ts=5, amount=10.0, equity_before=100.0))
    with pytest.raises(ValueError, match="time-ordered"):
        twr_weekly_return(100.0, 120.0, unordered)


def test_flow_event_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="nonzero"):
        FlowEvent(ts=1, amount=0.0, equity_before=100.0)
    with pytest.raises(ValueError, match="equity_before"):
        FlowEvent(ts=1, amount=10.0, equity_before=0.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        FlowEvent(ts=1, amount=-100.0, equity_before=100.0)


# --------------------------------------------------------------- the false-HALT protection, live

def test_large_withdrawal_does_not_walk_the_kill_switch_toward_halt():
    """A 30% withdrawal in a flat week: the RAW return (-30%) is a ~7.7-sigma downward shock that
    jumps the downward CUSUM most of the way to h* in ONE observation; the TWR return (0) leaves
    it essentially at rest. This is Invariant FLOW-TWR demonstrated against the real chain."""
    mu0, sig0 = D.mu_0, D.sigma_0
    f = FlowEvent(ts=10, amount=-30.0, equity_before=100.0)
    r_twr = twr_weekly_return(100.0, 70.0, (f,))
    r_raw = 70.0 / 100.0 - 1.0

    c_raw = CUSUM(d=D.d)
    c_twr = CUSUM(d=D.d)
    s_raw = c_raw.update(whiten(r_raw, mu_0=mu0, sigma_0=sig0))
    s_twr = c_twr.update(whiten(r_twr, mu_0=mu0, sigma_0=sig0))
    assert s_raw.c_down > 0.6 * D.h_star  # raw: one flow ~ most of the way to a false HALT
    assert s_twr.c_down < 0.1 * D.h_star  # TWR: a quiet week stays quiet


# --------------------------------------------------------------------------- flow_risk_check

def test_small_flow_waits_for_friday():
    d = flow_risk_check(FlowEvent(ts=1, amount=-5.0, equity_before=100.0))
    assert d.action == "NONE" and d.resize_factor == 1.0
    assert d.relative == pytest.approx(0.05)


def test_material_withdrawal_draws_reduce_only_interim_resize():
    d = flow_risk_check(FlowEvent(ts=1, amount=-30.0, equity_before=100.0))
    assert d.action == "INTERIM_RESIZE"
    assert d.resize_factor == pytest.approx(0.70)  # scale positions to the new equity
    assert d.resize_factor < 1.0  # reduce-only, never adds risk (P6)


def test_material_deposit_is_reconcile_only_never_midweek_risk_add():
    d = flow_risk_check(FlowEvent(ts=1, amount=25.0, equity_before=100.0))
    assert d.action == "RECONCILE_ONLY" and d.resize_factor == 1.0  # CAL-PRIMARY


def test_threshold_is_the_red_constant():
    just_below = flow_risk_check(
        FlowEvent(ts=1, amount=-(D.flow_check_threshold - 1e-9) * 100.0, equity_before=100.0))
    at = flow_risk_check(
        FlowEvent(ts=1, amount=-D.flow_check_threshold * 100.0, equity_before=100.0))
    assert just_below.action == "NONE"
    assert at.action == "INTERIM_RESIZE"
