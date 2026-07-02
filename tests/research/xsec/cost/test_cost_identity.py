"""Turnover-based cost identity against accounting.py's actual output (COST.md P3, Section 5.5-5.6).

accounting.py is the FROZEN parity anchor — these tests consume it, they never re-derive it
(Invariant N). The identity under test: cost_t = turnover_t * round_trip_friction (scalar path),
cost_t = 0.5 * sum_i |dw_i| * friction_i (per-symbol path), the latter reducing exactly to the
former at constant friction.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.cost.config.fee_schedule import BINANCE_USDM_FEE_SCHEDULE

from research.xsec.accounting import (
    HALF_SPREAD_FRAC,
    IMPACT_FRAC,
    round_trip_friction,
    settle_week,
)


def _flat_panel(n_days: int, n_sym: int):
    close = np.full((n_days, n_sym), 100.0)
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return close, funding, listed


def test_invariant_n_friction_rate_from_fee_schedule_not_a_literal():
    # Invariant N: round_trip_friction == 2*(taker + half_spread + impact) with the taker read
    # from BINANCE_USDM_FEE_SCHEDULE (the ONE fee truth), never a hand-copied constant.
    taker = BINANCE_USDM_FEE_SCHEDULE.tier_for_volume(0.0).taker_rate
    assert round_trip_friction() == 2.0 * (taker + HALF_SPREAD_FRAC + IMPACT_FRAC)
    # the assembled default is the recorded 13bp round-trip anchor (COST.md Section 5.5).
    assert round_trip_friction() == pytest.approx(0.0013, rel=1e-12)


def test_cost_identity_scalar_path():
    # cost_t = turnover_t * round_trip_friction, verified against settle_week's actual output.
    close, funding, listed = _flat_panel(30, 4)
    friction = round_trip_friction()
    w_target = np.array([0.3, 0.2, -0.25, -0.25])
    res, _ = settle_week(
        close, funding, listed,
        d0=7, d1=14, w_target=w_target, w_prev=np.zeros(4), friction=friction,
    )
    assert res.turnover == pytest.approx(0.5, rel=1e-15)  # 0.5 * sum|dw| from a flat start
    assert res.cost == res.turnover * friction  # the identity, bitwise
    assert res.net == pytest.approx(res.gross_price + res.funding - res.cost, rel=1e-15)


def test_cost_identity_includes_drift_turnover():
    # P3: names STAYING in a leg but drifting weight are charged too — holding the same book
    # through a price move still produces turnover = 0.5*sum|w_target - w_carried| > 0.
    close, funding, listed = _flat_panel(30, 2)
    close[7:, 0] = 110.0  # name 0 ran +10% over the elapsed week (days 0->7)
    friction = round_trip_friction()
    w = np.array([0.5, -0.5])
    res, _ = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=friction
    )
    w_carried_0 = 0.5 * 1.1  # prior long drifted up
    expected_turnover = 0.5 * abs(0.5 - w_carried_0)
    assert res.turnover == pytest.approx(expected_turnover, rel=1e-12)
    assert res.cost == res.turnover * friction


def test_per_symbol_path_reduces_exactly_to_scalar_at_constant_friction():
    # COST.md Section 5.6: the per-name model is a strict generalization, not a parallel system.
    close, funding, listed = _flat_panel(30, 4)
    f = round_trip_friction()
    w_target = np.array([0.25, 0.25, -0.25, -0.25])
    res_scalar, _ = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w_target, w_prev=np.zeros(4), friction=f
    )
    res_vec, _ = settle_week(
        close, funding, listed,
        d0=7, d1=14, w_target=w_target, w_prev=np.zeros(4), friction=np.full(4, f),
    )
    assert res_vec.cost == pytest.approx(res_scalar.cost, rel=1e-14)
    assert res_vec.turnover == res_scalar.turnover


def test_worked_example_section_5_6_per_name_contributions():
    # Median name A (13bp) vs 16x-thinner tail name B (22bp), equal |dw|: the tail name costs
    # ~1.7x more per unit of turnover (COST.md Section 5.6, reproduced bit-for-bit).
    close, funding, listed = _flat_panel(30, 2)
    w_prev = np.array([0.0, -0.04])
    w_target = np.array([0.04, 0.0])
    friction_i = np.array([0.0013, 0.0022])
    res, _ = settle_week(
        close, funding, listed,
        d0=7, d1=14, w_target=w_target, w_prev=w_prev, friction=friction_i,
    )
    contrib_a = 0.5 * 0.04 * 0.0013  # 0.000026
    contrib_b = 0.5 * 0.04 * 0.0022  # 0.000044
    assert res.cost == pytest.approx(contrib_a + contrib_b, rel=1e-15)
    assert res.cost == pytest.approx(7.0e-5, rel=1e-12)
    assert contrib_b / contrib_a == pytest.approx(0.0022 / 0.0013, rel=1e-15)  # ~1.69x
