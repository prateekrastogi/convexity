"""Invariant SIZE — N caps an already-liquidity-ranked set, monotone nesting (UNIVERSE.md §3.2, §11.2)."""

from __future__ import annotations

import numpy as np

from convexity.universe.liquidity.ranking import liquidity_rank, top_n_universe


def test_universe_monotone_nesting():
    """Inv SIZE (§3.2): for N1 < N2, ``U_t(N1) ⊆ U_t(N2)`` on the same rank_t and hysteresis-filtered
    base set (changing N changes HOW MANY survive, never WHICH statistic ranks them)."""
    rng = np.random.default_rng(1)
    n_sym = 120
    base = np.ones(n_sym, dtype=bool)
    dv = rng.random(n_sym) * 1e9
    rank = liquidity_rank(dv, base)
    state = np.ones(n_sym, dtype=bool)  # all IN (base-liquid), so top-N picks purely by rank
    for n1, n2 in [(50, 60), (60, 75), (75, 100)]:
        u1 = set(np.flatnonzero(top_n_universe(rank, state, N=n1)).tolist())
        u2 = set(np.flatnonzero(top_n_universe(rank, state, N=n2)).tolist())
        assert u1 <= u2, f"top-N({n1}) must nest inside top-N({n2})"
        assert len(u1) == n1 and len(u2) == n2


def test_size_keeps_most_liquid_and_uses_one_ranking():
    """§3.2: top-N keeps the N MOST liquid IN names by the SAME rank_i used as the hysteresis gate —
    not a second liquidity notion."""
    n_sym = 10
    base = np.ones(n_sym, dtype=bool)
    dv = np.arange(n_sym, dtype=float) * 1e8 + 1e6  # strictly increasing: sym 9 most liquid
    rank = liquidity_rank(dv, base)
    state = np.ones(n_sym, dtype=bool)
    keep = np.flatnonzero(top_n_universe(rank, state, N=3))
    assert set(keep.tolist()) == {7, 8, 9}, "top-3 = the three highest-DV names"


def test_fewer_than_N_returns_whole_in_set_never_backfilled():
    """§3.2 / §9 edge 3: if fewer than N names clear the hysteresis gate, U_t is the whole IN set,
    never backfilled with ineligible (OUT) names."""
    n_sym = 10
    dv = np.arange(n_sym, dtype=float) * 1e8 + 1e6
    rank = liquidity_rank(dv, np.ones(n_sym, dtype=bool))
    state = np.zeros(n_sym, dtype=bool)
    state[[2, 5]] = True  # only 2 names IN
    keep = np.flatnonzero(top_n_universe(rank, state, N=50))
    assert set(keep.tolist()) == {2, 5}, "U_t is the whole IN set (2 names), never backfilled to 50"


def test_conservative_fallback():
    """Cross-cutting (§2.4 P6): a base-eligible name with stale/NaN dollar volume ranks 0 (least
    liquid), never a neutral/average rank, so top-N can only exclude it, never admit it under an
    assumed rank."""
    n_sym = 5
    base = np.ones(n_sym, dtype=bool)
    dv = np.array([1e9, 8e8, 6e8, np.nan, 2e8])  # sym 3 has missing DV
    valid = np.array([30, 30, 30, 0, 30])  # sym 3 has zero valid days
    rank = liquidity_rank(dv, base, valid_days=valid)
    assert rank[3] == 0.0, "stale/missing-DV name falls to rank 0 (§2.4), never neutral"
    # With N small enough to bind, the rank-0 name is excluded, not admitted.
    state = np.ones(n_sym, dtype=bool)
    keep = np.flatnonzero(top_n_universe(rank, state, N=3))
    assert 3 not in keep.tolist(), "rank-0 stale name is excluded by the size cutoff"
