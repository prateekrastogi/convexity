"""Invariant HYST — membership is path-dependent, monotone, F_t-measurable (UNIVERSE.md §2.2, §11.2)."""

from __future__ import annotations

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG
from convexity.universe.liquidity.hysteresis import hysteresis_step

Q_IN = DEFAULT_UNIVERSE_CONFIG.q_in  # 0.80
Q_OUT = DEFAULT_UNIVERSE_CONFIG.q_out  # 0.65


def _simulate(ranks: np.ndarray, start_in: bool) -> np.ndarray:
    """Replay a scalar rank series through the FSM (carrying state forward), return per-step IN/OUT."""
    state = np.array([start_in], dtype=bool)
    states = np.empty(len(ranks), dtype=bool)
    for i, r in enumerate(ranks):
        state = hysteresis_step(np.array([r]), state, q_in=Q_IN, q_out=Q_OUT)
        states[i] = state[0]
    return states


def test_no_flip_inside_band():
    """Inv HYST (§2.2): a rank oscillating strictly inside (q_out, q_in) produces ZERO flips over the
    whole series, from EITHER start state."""
    rng = np.random.default_rng(0)
    n_weeks = 104
    mid = 0.5 * (Q_IN + Q_OUT)
    span = 0.5 * (Q_IN - Q_OUT) - 1e-3  # stay strictly inside the band
    ranks = mid + (rng.random(n_weeks) - 0.5) * 2.0 * span
    assert np.all(ranks > Q_OUT) and np.all(ranks < Q_IN)
    for start in (True, False):
        states = _simulate(ranks, start_in=start)
        assert len(set(states.tolist())) == 1, "no membership flip inside the band"
        assert bool(states[0]) is start, "state holds the prior value inside the band"


def test_decay_to_exactly_q_out_never_exits():
    """Inv HYST (§2.2): a name that enters at q_in then only ever decays to exactly q_out+eps (never
    strictly below q_out) never exits."""
    ranks = [Q_IN]  # enter
    r = Q_IN
    while r > Q_OUT + 1e-9:
        r = max(Q_OUT + 1e-9, r - 0.01)
        ranks.append(r)
    ranks.append(Q_OUT + 1e-9)  # rest exactly at q_out + eps, never below
    states = _simulate(np.array(ranks), start_in=False)
    assert states[0], "crossing q_in from OUT enters"
    assert states[-1], "decay to exactly q_out + eps never triggers exit (exit needs rank < q_out)"


def test_entry_requires_higher_threshold_exit_lower():
    """Inv HYST asymmetry: entry needs rank >= q_in; a rank in (q_out, q_in) does NOT enter from OUT;
    exit needs rank < q_out; a rank in (q_out, q_in) does NOT exit from IN."""
    below_in = 0.5 * (Q_IN + Q_OUT)  # inside band
    # From OUT, a mid-band rank does not enter.
    assert not hysteresis_step(np.array([below_in]), np.array([False]), q_in=Q_IN, q_out=Q_OUT)[0]
    # Reaching q_in from OUT enters.
    assert hysteresis_step(np.array([Q_IN]), np.array([False]), q_in=Q_IN, q_out=Q_OUT)[0]
    # From IN, a mid-band rank does not exit.
    assert hysteresis_step(np.array([below_in]), np.array([True]), q_in=Q_IN, q_out=Q_OUT)[0]
    # Dropping below q_out from IN exits.
    assert not hysteresis_step(np.array([Q_OUT - 1e-6]), np.array([True]), q_in=Q_IN, q_out=Q_OUT)[0]


def test_conservative_fallback():
    """Cross-cutting (§2.4 P6): a NaN rank (name outside U_t^base or with stale liquidity) transitions
    to OUT regardless of prior state — never admitted under an assumed rank."""
    # From IN, NaN forces exit.
    assert not hysteresis_step(np.array([np.nan]), np.array([True]), q_in=Q_IN, q_out=Q_OUT)[0]
    # From OUT, NaN stays OUT (cannot enter without a valid >= q_in rank).
    assert not hysteresis_step(np.array([np.nan]), np.array([False]), q_in=Q_IN, q_out=Q_OUT)[0]
