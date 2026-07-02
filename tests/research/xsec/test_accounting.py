"""Accounting: round-trip friction, signed funding cash flow, turnover incl. drift, delist handling."""

from __future__ import annotations

import numpy as np

from convexity.cost.accounting import (
    BINANCE_VIP0_TAKER,
    round_trip_friction,
    settle_week,
)


def test_taker_rate_is_the_cost_pillar_truth():
    # ONE fee truth: Binance USDT-M VIP0 taker = 5 bps = 0.0005.
    assert abs(BINANCE_VIP0_TAKER - 0.0005) < 1e-12


def test_round_trip_friction_both_legs():
    # (5 + 1 + 0.5) bps/leg * 2 legs = 13 bps = 0.0013.
    assert abs(round_trip_friction() - 0.0013) < 1e-12


def _flat_panel(n_days: int, n_sym: int):
    close = np.ones((n_days, n_sym)) * 100.0
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return close, funding, listed


def test_short_on_positive_funding_is_credited():
    # a SHORT (w<0) on a positive-funding name receives the funding as a CREDIT.
    close, funding, listed = _flat_panel(30, 2)
    funding[:, 0] = 0.0003  # daily funding on the name we short
    w = np.array([-0.5, 0.5])  # short name 0, long name 1
    res, _ = settle_week(close, funding, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0)
    # 7 days of 0.0003 funding on name 0, cash flow = -w*f = -(-0.5)*(7*0.0003) = +0.00105.
    assert res.funding > 0
    assert abs(res.funding - 0.5 * 7 * 0.0003) < 1e-9


def test_long_on_positive_funding_pays():
    close, funding, listed = _flat_panel(30, 2)
    funding[:, 0] = 0.0003
    w = np.array([0.5, -0.5])  # LONG the positive-funding name
    res, _ = settle_week(close, funding, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0)
    assert res.funding < 0  # long pays positive funding


def test_zero_turnover_when_book_unchanged_and_flat_prices():
    close, funding, listed = _flat_panel(30, 2)
    w = np.array([0.5, -0.5])
    res, _ = settle_week(close, funding, listed, d0=14, d1=21, w_target=w, w_prev=w, friction=0.0013)
    # flat prices -> no drift -> same book -> zero turnover -> zero cost.
    assert res.turnover < 1e-12
    assert res.cost < 1e-12


def test_full_turnover_from_flat_start():
    close, funding, listed = _flat_panel(30, 2)
    w = np.array([0.5, -0.5])
    res, _ = settle_week(close, funding, listed, d0=14, d1=21, w_target=w, w_prev=np.zeros(2), friction=0.0013)
    # from a flat prior book, turnover = 0.5*sum|w| = 0.5*(1.0) = 0.5; cost = 0.5*0.0013.
    assert abs(res.turnover - 0.5) < 1e-12
    assert abs(res.cost - 0.5 * 0.0013) < 1e-12


def test_delisted_name_closed_no_holdthrough():
    close, funding, listed = _flat_panel(30, 2)
    listed[18:, 0] = False  # name 0 delists mid-week (entry d0=14, exit d1=21)
    w = np.array([0.5, -0.5])
    res, w_end = settle_week(close, funding, listed, d0=14, d1=21, w_target=w, w_prev=w, friction=0.0)
    # name 0 not live at d1 -> its return is NaN -> excluded from price P&L, weight ends 0.
    assert w_end[0] == 0.0


def test_price_pnl_dollar_neutral_long_wins():
    close, funding, listed = _flat_panel(30, 2)
    close[14:, 0] = 110.0  # long name rises 10% from day 14
    w = np.array([0.5, -0.5])
    res, _ = settle_week(close, funding, listed, d0=13, d1=20, w_target=w, w_prev=w, friction=0.0)
    # long name 0 +10%, short name 1 flat -> gross price = 0.5*0.10 = +0.05.
    assert abs(res.gross_price - 0.05) < 1e-9
