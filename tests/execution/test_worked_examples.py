"""EXECUTION worked-example regression fixtures (ARCHITECTURE §8 / EXECUTION §15).

The load-bearing numbered examples: PI at touch / one-tick; entry vs exit maker cost split;
worst_entry long/short; funding half-open window; sqrt impact → 0; p_fill LCB gate; STOP_OK
predicate; disjoint slice/stop id schemes.
"""

import math

import numpy as np
import pytest

from convexity.execution.config.resolve import resolve_config
from convexity.execution.cost.adverse_selection import AdverseSelection
from convexity.execution.cost.delay_cost import DelayCost
from convexity.execution.cost.impact import ImpactModel
from convexity.execution.cost.spread_cost import SpreadCost
from convexity.execution.cost.tactic_cost import TacticCostModel, fund_bps
from convexity.execution.microstate.fill_prob import FillProbability, logistic_fill_curve
from convexity.execution.microstate.spread import SpreadEstimator
from convexity.execution.policy.chase import worst_entry
from convexity.execution.policy.order_issuer import (
    slice_client_order_id,
    stop_client_order_id,
)
from convexity.execution.synth.generate import ExecutionSynth, perp_instrument
from convexity.execution.types import ExecTarget, MicroState
from convexity.core.types import Bars


def test_pi_at_touch_equals_full_half_spread():
    """PI = s − δ·tick_bps: at δ=0 PI == s (the whole half-spread saved) (§5.1)."""
    sp = SpreadCost()
    s, tick_bps = 8.0, 0.033
    assert sp.price_improvement_bps(s, 0.0, tick_bps) == pytest.approx(8.0)


def test_pi_one_tick_deeper():
    """PI one tick deeper = s − tick_bps (§5.1)."""
    sp = SpreadCost()
    s, tick_bps = 8.0, 0.5
    assert sp.price_improvement_bps(s, 1.0, tick_bps) == pytest.approx(7.5)
    # Two ticks deeper drops by 2·tick_bps.
    assert sp.price_improvement_bps(s, 2.0, tick_bps) == pytest.approx(7.0)


def _model():
    return TacticCostModel(SpreadCost(), ImpactModel(), AdverseSelection(), DelayCost())


def test_entry_vs_exit_maker_split_differ():
    """The ENTRY maker cost (no-fill -> split with free abort) is <= the EXIT maker cost (§5.1.2/3)."""
    cfg = resolve_config(perp_instrument())
    mid = 30_000.0
    curve = logistic_fill_curve(base_rate=0.5, sigma_bps=8.0, tick_bps=10_000.0 * 0.1 / mid)
    micro = MicroState(
        mid=mid, half_spread_bps=5.0, spread_rung="quoted", sigma_horizon_bps=8.0,
        volume_proxy=5e6, drift_g=2.0, tick_bps=10_000.0 * 0.1 / mid, fill_prob_curve=curve,
        funding_aware=False, predicted_funding_rate=0.0, secs_to_next_funding=float("inf"),
        halted=False, stale=False, rate_limit_ok=True,
    )
    entry = ExecTarget("e", "BTCUSDT", 1, 0.01, mid, mid * 0.98, "NORMAL", reduce_only=False)
    ex = ExecTarget("x", "BTCUSDT", -1, -0.01, mid, mid * 1.02, "NORMAL", reduce_only=True)
    m = _model()
    ce = m.costs(entry, micro, cfg)
    cx = m.costs(ex, micro, cfg)
    # The entry zeroes the abort mass's cross cost; the exit charges the full cross to all no-fill
    # mass — so the exit maker branch is at least as expensive at the same offset structure.
    assert cx.by_tactic["POST_ONLY_MAKER"] >= ce.by_tactic["POST_ONLY_MAKER"] - 1e-9
    assert ce.p_abort_if_nofill > 0.0  # entry has abort mass
    assert cx.p_abort_if_nofill == 0.0  # exit never aborts


def test_worst_entry_long_short():
    """worst_entry long is above entry0, short below; magnitude χ·|inv−entry0| (§5.3)."""
    cfg = resolve_config(perp_instrument())
    chi = cfg.defaults.chase_fraction_chi
    long_t = ExecTarget("l", "BTCUSDT", 1, 0.01, 100.0, 90.0, "URGENT", False)
    short_t = ExecTarget("s", "BTCUSDT", -1, -0.01, 100.0, 110.0, "URGENT", False)
    assert worst_entry(long_t, 100.0, cfg) == pytest.approx(100.0 + chi * 10.0)
    assert worst_entry(short_t, 100.0, cfg) == pytest.approx(100.0 - chi * 10.0)


def test_funding_half_open_window():
    """A stamp at secs == h is NOT counted (half-open [now, now+h)); a stamp inside IS (§5.4)."""
    cfg = resolve_config(perp_instrument())
    mid = 30_000.0
    curve = logistic_fill_curve(base_rate=0.9, sigma_bps=3.0, tick_bps=0.033)
    base = dict(
        mid=mid, half_spread_bps=6.0, spread_rung="quoted", sigma_horizon_bps=3.0, volume_proxy=5e6,
        drift_g=0.0, tick_bps=0.033, fill_prob_curve=curve, funding_aware=True,
        predicted_funding_rate=0.005, halted=False, stale=False, rate_limit_ok=True,
        funding_interval_secs=8 * 3600.0,
    )
    t = ExecTarget("f", "BTCUSDT", 1, 0.01, mid, mid * 0.98, "PATIENT", False)
    h = 30.0
    boundary = MicroState(**base, secs_to_next_funding=h)  # exactly h -> 0
    inside = MicroState(**base, secs_to_next_funding=10.0)  # inside -> 1
    assert fund_bps(t, boundary, h, cfg) == 0.0
    assert fund_bps(t, inside, h, cfg) == pytest.approx(0.005 * 10_000.0 * 1)


def test_funding_two_stamps_in_long_horizon():
    """Two stamps inside a long window are counted (half-open) (§5.4)."""
    cfg = resolve_config(perp_instrument())
    mid = 30_000.0
    curve = logistic_fill_curve(base_rate=0.9, sigma_bps=3.0, tick_bps=0.033)
    micro = MicroState(
        mid=mid, half_spread_bps=6.0, spread_rung="quoted", sigma_horizon_bps=3.0, volume_proxy=5e6,
        drift_g=0.0, tick_bps=0.033, fill_prob_curve=curve, funding_aware=True,
        predicted_funding_rate=0.004, secs_to_next_funding=100.0, halted=False, stale=False,
        rate_limit_ok=True, funding_interval_secs=1000.0,
    )
    t = ExecTarget("f", "BTCUSDT", 1, 0.01, mid, mid * 0.98, "PATIENT", False)
    # window [now, now+2500): stamps at 100, 1100, 2100 -> 3 stamps inside.
    assert fund_bps(t, micro, 2500.0, cfg) == pytest.approx(0.004 * 10_000.0 * 3)


def test_sqrt_impact_to_zero():
    """sqrt impact -> ~0 as participation -> 0; equals η·σ·sqrt(part) exactly (§11 / Inv I)."""
    cfg = resolve_config(perp_instrument())
    mid = 30_000.0
    curve = logistic_fill_curve(base_rate=0.9, sigma_bps=10.0, tick_bps=0.033)
    micro = MicroState(
        mid=mid, half_spread_bps=5.0, spread_rung="quoted", sigma_horizon_bps=10.0,
        volume_proxy=1e9, drift_g=0.0, tick_bps=0.033, fill_prob_curve=curve, funding_aware=True,
        predicted_funding_rate=0.0, secs_to_next_funding=float("inf"), halted=False, stale=False,
        rate_limit_ok=True,
    )
    im = ImpactModel()
    part = im.participation(0.01, micro)
    expected = cfg.defaults.impact_eta * 10.0 * math.sqrt(part)
    assert im.cost_bps(0.01, micro, cfg) == pytest.approx(expected)
    assert im.cost_bps(0.01, micro, cfg) < 0.5  # negligible at the $300 clip on deep volume


def test_p_fill_lcb_gate_predicate():
    """p_fill_LCB = point − z·ci; the gate keys on the LOWER bound, not the point (§6.3)."""
    cfg = resolve_config(perp_instrument())
    point, ci = 0.8, 0.30
    lcb = FillProbability.lcb(point, ci, 0.0, cfg)
    assert lcb == pytest.approx(max(0.0, 0.8 - cfg.defaults.z_lcb * 0.30))
    # A high-point but wide-CI estimate falls below the floor; a tight-CI one clears it.
    floor = FillProbability.floor_eff(0.0, cfg)
    assert FillProbability.lcb(0.8, 0.40, 0.0, cfg) < floor
    assert FillProbability.lcb(0.95, 0.02, 0.0, cfg) >= floor


def test_floor_eff_widens_with_calibration_error():
    """floor_eff = floor_RED + κ·max(0, calib_error − calib_RED) (§6.3)."""
    cfg = resolve_config(perp_instrument())
    base = FillProbability.floor_eff(0.0, cfg)
    high = FillProbability.floor_eff(0.20, cfg)
    assert base == pytest.approx(cfg.defaults.floor_red)
    assert high == pytest.approx(
        cfg.defaults.floor_red + cfg.defaults.kappa_calib * (0.20 - cfg.defaults.calib_red)
    )
    assert high > base


def test_stop_ok_predicate():
    """STOP_OK holds iff exactly one correct resting reduce-only stop at current (id, trigger, qty)."""
    from convexity.execution.nautilus.reconcile import IntendedState, stop_ok
    from convexity.execution.types import PositionState, VenueStop
    intended = IntendedState("X", 29_000.0, 0)
    pos = PositionState("BTCUSDT", 1, 0.01)
    good = (VenueStop(stop_client_order_id("X", 0), 29_000.0, 0.01, True),)
    assert stop_ok(intended, pos, good)
    # Wrong trigger fails.
    bad_trig = (VenueStop(stop_client_order_id("X", 0), 28_000.0, 0.01, True),)
    assert not stop_ok(intended, pos, bad_trig)
    # Two stops fail.
    two = good + (VenueStop(stop_client_order_id("X", 1), 29_000.0, 0.01, True),)
    assert not stop_ok(intended, pos, two)
    # Flat with no stops is OK; flat with an orphan stop is not.
    flat = PositionState("BTCUSDT", 1, 0.0)
    assert stop_ok(intended, flat, ())
    assert not stop_ok(intended, flat, good)


def test_disjoint_id_schemes():
    """Slice ids and stop ids never collide (§7 / §9, Invariant Q / Q-STOP)."""
    slice_ids = {slice_client_order_id("X", i, a) for i in range(30) for a in range(4)}
    stop_ids = {stop_client_order_id("X", g) for g in range(30)}
    assert slice_ids.isdisjoint(stop_ids)


def test_spread_ladder_roll_degenerate_falls_through():
    """On trending bars (positive serial autocov) the spread estimator never NaN/infs (§5 / §11)."""
    path = ExecutionSynth().trending_roll_degenerate()
    n = path.size
    bars = Bars(
        ts=np.arange(n, dtype=np.int64) * 60_000,
        o=path, h=path * 1.001, l=path * 0.999, c=path, v=np.full(n, 1000.0),
    )
    cfg = resolve_config(perp_instrument())
    est = SpreadEstimator(window=30)
    out = est.estimate(bars, n - 1, cfg)
    assert np.isfinite(out.half_spread_bps) and out.half_spread_bps > 0.0
    # Roll is undefined on this trending series -> the rung is NOT "roll".
    assert out.rung in ("corwin_schultz", "class_proxy")


def test_spread_ladder_quoted_when_bidask_present():
    """The quoted rung fires when the feed carries a valid bid/ask (§11 rung 1)."""
    n = 50
    mid = np.full(n, 100.0)
    bars = Bars(
        ts=np.arange(n, dtype=np.int64) * 60_000, o=mid, h=mid, l=mid, c=mid,
        v=np.full(n, 1000.0), bid=np.full(n, 99.9), ask=np.full(n, 100.1),
    )
    cfg = resolve_config(perp_instrument())
    out = SpreadEstimator().estimate(bars, n - 1, cfg)
    assert out.rung == "quoted"
    assert out.half_spread_bps == pytest.approx(10_000.0 * 0.1 / 100.0)  # half of 0.2 / mid


def test_short_side_mirror_term_by_term():
    """side=-1 on a sign-mirrored path equals side=+1 maker cost term-by-term (§5.1)."""
    cfg = resolve_config(perp_instrument())
    long_s, short_s = ExecutionSynth().short_side_mirror()
    m = _model()
    cl = m.costs(long_s.target, long_s.micro, cfg)
    cs = m.costs(short_s.target, short_s.micro, cfg)
    # Same microstate magnitudes (g, sigma, spread) -> identical maker-branch bps (side-agnostic).
    assert cl.by_tactic["POST_ONLY_MAKER"] == pytest.approx(cs.by_tactic["POST_ONLY_MAKER"])
    assert cl.price_improve_bps == pytest.approx(cs.price_improve_bps)
    assert cl.adverse_sel_bps["POST_ONLY_MAKER"] == pytest.approx(cs.adverse_sel_bps["POST_ONLY_MAKER"])
