"""Drawdown governor ladder m_t (LEVERAGE.md §6.4, §10)."""

from __future__ import annotations

import numpy as np

from convexity.leverage.ceilings.governor import governor_state, ladder
from convexity.leverage.config.defaults import LeverageDefaults


def test_ladder_monotone_non_increasing_in_dd():
    # §6.4 / §10 gate: m_t is a monotone NON-INCREASING step in dd_t.
    d = LeverageDefaults()
    dds = np.linspace(0.0, 0.6, 200)
    ms = [ladder(dd, d) for dd in dds]
    assert all(ms[i + 1] <= ms[i] for i in range(len(ms) - 1))
    assert all(0.0 <= m <= 1.0 for m in ms)


def test_ladder_breakpoints_match_25pct_anchor_multiples():
    # §6.4: breakpoints at 0.5×/1.0×/1.5× of the ~25% max-DD anchor; terminal at 1.6×.
    d = LeverageDefaults()
    assert ladder(0.10, d) == 1.00   # inside 0.5·0.25 = 0.125 -> no de-risk
    assert ladder(0.20, d) == 0.60   # between 0.125 and 0.25 -> de-risk
    assert ladder(0.30, d) == 0.35   # past the 0.25 anchor -> de-risk hard
    assert ladder(0.45, d) == 0.00   # past 0.40 terminal -> full flat


def test_ladder_terminal_is_full_flat():
    # §6.4: m_t -> 0 at the RED terminal threshold (governor kill).
    d = LeverageDefaults()
    assert ladder(d.dd_terminal, d) == 0.0
    assert ladder(1.0, d) == 0.0


def test_governor_reads_book_equity_from_weekly_net():
    # §6.4: dd_t computed from the compounded weekly net series (book equity), not per-trade P&L.
    d = LeverageDefaults()
    # a -30% cumulative drawdown from peak: +10% then a run of losses to ~0.7 of peak.
    r = np.concatenate([np.array([0.10]), np.full(20, -0.02)])
    gs = governor_state(r, d)
    equity = np.cumprod(1.0 + r)
    expected_dd = 1.0 - equity[-1] / np.maximum.accumulate(equity)[-1]
    assert np.isclose(gs.dd_t, expected_dd)
    assert gs.dd_t > 0.25  # past the anchor


def test_governor_no_drawdown_gives_full_multiplier():
    d = LeverageDefaults()
    gs = governor_state(np.full(30, 0.01), d)  # monotone up -> no drawdown
    assert gs.dd_t == 0.0
    assert gs.m_t == 1.0
    assert not gs.terminal


def test_governor_terminal_flag_and_kill():
    d = LeverageDefaults()
    # a >40% drawdown triggers terminal kill.
    r = np.concatenate([np.array([0.05]), np.full(30, -0.03)])
    gs = governor_state(r, d)
    assert gs.dd_t >= d.dd_terminal
    assert gs.terminal
    assert gs.m_t == 0.0


def test_carry_dd_stop_regression_governor_not_over_tight():
    # §8 cautionary edge case: the ladder must NOT de-risk inside the book's normal DD range (which
    # would reproduce the carry-sleeve collapse). A modest 8% drawdown (inside 0.5·0.25) stays at
    # m_t=1.0 -> the governor does not touch a book operating inside its natural DD scale.
    d = LeverageDefaults()
    assert ladder(0.08, d) == 1.0
