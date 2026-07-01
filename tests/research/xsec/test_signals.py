"""Cross-sectional signals: no-lookahead, point-in-time eligibility, ranking, carry sign."""

from __future__ import annotations

import numpy as np

from research.xsec import signals as sig
from research.xsec.signals import _pct_rank


def test_pct_rank_monotone_and_neutral_single():
    r = _pct_rank(np.array([3.0, 1.0, 2.0, np.nan]))
    # highest value (3.0) gets highest rank; NaN passes through.
    assert r[0] > r[2] > r[1]
    assert np.isnan(r[3])
    assert np.isclose(_pct_rank(np.array([5.0]))[0], 0.5)


def test_momentum_ranks_strongest_trend_highest(simple_panel):
    # at day 80 A has the strongest uptrend, C the downtrend -> A ranks top, C bottom.
    rank = sig.momentum_signal(simple_panel, 80, lookback=28, skip=0)
    a, b, c, d = rank
    assert a == max(rank)
    assert c == min(rank)


def test_momentum_excludes_unlisted_symbol_no_lookahead(simple_panel):
    # at day 30 symbol D is NOT yet listed (lists day 40) -> its rank must be NaN (never ranked early).
    rank = sig.momentum_signal(simple_panel, 30, lookback=28, skip=0)
    assert np.isnan(rank[3])
    assert np.isfinite(rank[0])


def test_momentum_needs_full_trailing_window(simple_panel):
    # at day 40 (D just listed) D lacks a 28d trailing window -> NaN, but A/B/C are fine.
    rank = sig.momentum_signal(simple_panel, 45, lookback=28, skip=0)
    assert np.isnan(rank[3])  # only 5 days of D history < 28
    assert np.isfinite(rank[0])


def test_momentum_skip_shifts_window_back_no_lookahead(simple_panel):
    # skip must measure the return over [d-skip-L, d-skip], all indices <= d (never past d).
    # verify directly against the panel closes for symbol A (a strict uptrend).
    d, L, skip = 80, 14, 7
    c = simple_panel.close
    expect = c[d - skip, 0] / c[d - skip - L, 0] - 1.0
    # both windows end at or before d -> no lookahead; ranks stay top for A either way.
    r0 = sig.momentum_signal(simple_panel, d, lookback=L, skip=0)
    r7 = sig.momentum_signal(simple_panel, d, lookback=L, skip=skip)
    assert r0[0] == max(r0) and r7[0] == max(r7)
    # the skipped window's endpoint index (d-skip) is strictly before d -> no lookahead.
    assert expect > 0  # A trends up over the pre-skip window


def test_carry_sign_high_funding_shorted(simple_panel):
    # A has the HIGHEST positive funding -> we SHORT it -> LOWEST long-rank.
    # C has negative funding -> LONG it -> HIGHEST long-rank.
    rank = sig.carry_signal(simple_panel, 80, lookback=7)
    a, b, c, d = rank
    assert c == max(rank)  # negative funding -> long
    assert a == min(rank)  # highest funding -> short


def test_combined_intersection_eligibility(simple_panel):
    # at day 45, D fails momentum eligibility -> combined must also be NaN for D.
    rank = sig.combined_signal(simple_panel, 45, lookback=28, skip=0, carry_lookback=7)
    assert np.isnan(rank[3])
    assert np.isfinite(rank[0])
