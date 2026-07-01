"""Breadth sweep support: per-symbol (liquidity-scaled) friction + trailing dollar-volume rank."""

from __future__ import annotations

import numpy as np

from research.xsec.accounting import settle_week
from research.xsec.panel import trailing_dollar_volume


def _flat_panel(n_days: int, n_sym: int):
    close = np.ones((n_days, n_sym)) * 100.0
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return close, funding, listed


def test_scalar_and_constant_vector_friction_agree():
    # a per-symbol friction that is CONSTANT must give the identical cost as the scalar path.
    close, funding, listed = _flat_panel(30, 4)
    w = np.array([0.5, -0.5, 0.0, 0.0])
    scal, _ = settle_week(close, funding, listed, d0=14, d1=21, w_target=w,
                          w_prev=np.zeros(4), friction=0.0013)
    vec, _ = settle_week(close, funding, listed, d0=14, d1=21, w_target=w,
                         w_prev=np.zeros(4), friction=np.full(4, 0.0013))
    assert abs(scal.cost - vec.cost) < 1e-15
    assert abs(scal.net - vec.net) < 1e-15


def test_per_symbol_friction_charges_each_name_its_own_rate():
    # name 0 turns over with a wide spread, name 1 with the base -> cost = 0.5*(|dw0|*f0 + |dw1|*f1).
    close, funding, listed = _flat_panel(30, 2)
    w = np.array([0.5, -0.5])
    fric = np.array([0.010, 0.001])  # name 0 costs 10x name 1
    res, _ = settle_week(close, funding, listed, d0=14, d1=21, w_target=w,
                         w_prev=np.zeros(2), friction=fric)
    expected = 0.5 * (0.5 * 0.010 + 0.5 * 0.001)
    assert abs(res.cost - expected) < 1e-15


def test_trailing_dollar_volume_ranks_and_masks_offspan():
    # 3 symbols, constant per-day dollar volume A>B>C; C not live at the query day -> NaN.
    n_days, n_sym = 40, 3
    dvol = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    dvol[:, 0] = 3.0e9
    dvol[:, 1] = 1.0e9
    dvol[:, 2] = 5.0e8
    listed[35:, 2] = False  # C delists before the query day
    dvol = np.where(listed, dvol, np.nan)
    tv = trailing_dollar_volume(dvol, listed, d=38, window=30)
    assert np.isfinite(tv[0]) and np.isfinite(tv[1])
    assert np.isnan(tv[2])  # not live at d -> excluded
    assert tv[0] > tv[1]  # liquidity ordering preserved
