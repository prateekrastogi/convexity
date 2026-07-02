"""Invariant F-stamp companion cases (COST.md Section 6.2, Section 10 delist edge case).

The plain sign convention (short credited on positive funding, long pays) is already covered by
tests/research/xsec/test_accounting.py; these tests cover what that file does not: the accrual is
a listed-masked PER-STAMP sum, so a mid-week sign flip combined with a mid-week delisting is
priced on the days actually held — never a flat weekly-average applied across the delist gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.cost.accounting import settle_week


def _flat_panel(n_days: int, n_sym: int):
    close = np.full((n_days, n_sym), 100.0)
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return close, funding, listed


def test_fstamp_sign_flip_with_delist_beats_flat_average():
    # Funding +0.001/day for 2 days then -0.002/day for 5 days; the name delists after day 2 of
    # the hold week. Masked per-stamp accrual = +0.002 (days held). The flat-average-over-the-week
    # alternative would book -0.008 — wrong magnitude AND wrong sign for the short's cash flow.
    close, funding, listed = _flat_panel(30, 2)
    funding[8:10, 0] = 0.001  # hold week is (d0=7, d1=14] -> days 8..14
    funding[10:15, 0] = -0.002
    listed[10:, 0] = False  # delists after day 9: only days 8, 9 are held
    w = np.array([-0.5, 0.5])  # SHORT the flipping name
    res, w_end = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0
    )
    masked_accrual = 0.001 * 2  # the days actually held
    assert res.funding == pytest.approx(-(-0.5) * masked_accrual, rel=1e-12)  # +0.001, credited
    naive_flat = -(-0.5) * (0.001 * 2 - 0.002 * 5)  # -0.004: the forbidden approximation
    assert np.sign(res.funding) != np.sign(naive_flat)
    assert w_end[0] == 0.0  # position closed at the delist, no hold-through


def test_fstamp_accrual_is_path_dependent_under_delist():
    # Two funding paths with the SAME weekly total, different timing vs a mid-week delist: the
    # per-stamp masked sum prices them differently (a lump-sum weekly rate could not).
    w = np.array([-0.5, 0.5])
    close, funding_early, listed = _flat_panel(30, 2)
    funding_late = funding_early.copy()
    funding_early[8, 0] = 0.003  # stamp lands while still listed
    funding_late[13, 0] = 0.003  # same total, stamps after the delist
    listed[11:, 0] = False  # live for days 8..10 of the hold week only
    res_early, _ = settle_week(
        close, funding_early, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0
    )
    res_late, _ = settle_week(
        close, funding_late, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0
    )
    assert res_early.funding == pytest.approx(0.5 * 0.003, rel=1e-12)  # short credited
    assert res_late.funding == 0.0  # never accrued: the name was gone
    assert res_early.funding != res_late.funding
