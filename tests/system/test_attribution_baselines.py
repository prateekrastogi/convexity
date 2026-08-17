"""Pillar attribution and the required baselines (SYSTEM §8.2/§8.3).

Both modules shipped at 0% coverage and both carry named invariants. ATTR is the interesting one:
Shapley's order-independence is a theorem ONLY when all 16 coalitions are provisioned, so the
module must REFUSE to call a 5-ablation result "Shapley". That refusal is the thing worth
testing — an honest label on an attribution is what stops a pillar being kept on false credit.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from convexity.system.validate.attribution import (
    ALL_PILLARS,
    PillarAttribution,
)
from convexity.system.validate.baselines import (
    OFF_SEMANTICS,
    SINGLE_PILLAR_ABLATIONS,
    buy_and_hold,
    coalition_subsets,
    random_matched_frequency,
)


def all_coalitions(value_of) -> dict:
    out = {}
    for size in range(len(ALL_PILLARS) + 1):
        for combo in itertools.combinations(ALL_PILLARS, size):
            out[frozenset(combo)] = value_of(frozenset(combo))
    return out


# ---------------------------------------------------------------- Invariant ATTR
def test_shapley_requires_all_sixteen_coalitions():
    """The invariant: claiming Shapley from fewer than 16 coalitions is an error, not a warning."""
    partial = {frozenset(): 0.0, frozenset({"selection"}): 1.0}
    with pytest.raises(ValueError, match="16"):
        PillarAttribution().attribute(partial, "shapley", ALL_PILLARS)


def test_shapley_of_an_additive_game_returns_each_pillars_own_value():
    """With additive value, every ordering gives the same marginal — Shapley must recover it
    exactly. This is the property that makes the number attributable at all."""
    own = {"selection": 3.0, "breakout": 5.0, "leverage": 2.0, "trend": -1.0}
    pnl = all_coalitions(lambda s: sum(own[p] for p in s))
    rep = PillarAttribution().attribute(pnl, "shapley", ALL_PILLARS)
    assert rep.attribution_mode == "shapley"
    assert rep.fixed_order is None
    for p, v in own.items():
        assert rep.pillar_contribution[p] == pytest.approx(v)


def test_shapley_contributions_sum_to_the_full_system_value():
    """Efficiency: the pillar contributions must account for the whole system, no residue."""
    own = {"selection": 3.0, "breakout": 5.0, "leverage": 2.0, "trend": -1.0}
    pnl = all_coalitions(lambda s: sum(own[p] for p in s) + 0.5 * len(s) * (len(s) - 1))
    rep = PillarAttribution().attribute(pnl, "shapley", ALL_PILLARS)
    total = pnl[frozenset(ALL_PILLARS)] - pnl[frozenset()]
    assert sum(rep.pillar_contribution.values()) == pytest.approx(total)


def test_shapley_is_order_independent_but_sequential_is_not():
    """The whole reason Mode B is labelled honestly: with interactions, knock-in order changes the
    answer. Shapley does not."""
    pnl = all_coalitions(lambda s: sum({"selection": 3.0, "breakout": 5.0, "leverage": 2.0,
                                        "trend": -1.0}[p] for p in s)
                         + 2.0 * (len(s) >= 2))
    a = PillarAttribution().attribute(pnl, "shapley", ALL_PILLARS)
    b = PillarAttribution().attribute(pnl, "shapley", tuple(reversed(ALL_PILLARS)))
    assert a.pillar_contribution == b.pillar_contribution

    fwd = PillarAttribution().attribute(pnl, "sequential", ALL_PILLARS)
    rev = PillarAttribution().attribute(pnl, "sequential", tuple(reversed(ALL_PILLARS)))
    assert fwd.attribution_mode == "sequential_fixed_order"
    assert fwd.fixed_order == ALL_PILLARS
    assert fwd.pillar_contribution != rev.pillar_contribution


def test_sequential_needs_only_the_knock_in_chain():
    """Mode B's point: it is computable from 5 runs, not 16."""
    chain, active, val = {}, set(), 0.0
    chain[frozenset()] = 0.0
    for i, p in enumerate(ALL_PILLARS, start=1):
        active.add(p)
        val += i
        chain[frozenset(active)] = val
    rep = PillarAttribution().attribute(chain, "sequential", ALL_PILLARS)
    assert rep.attribution_mode == "sequential_fixed_order"
    assert [rep.pillar_contribution[p] for p in ALL_PILLARS] == [1.0, 2.0, 3.0, 4.0]


def test_a_pillar_earning_nothing_is_flagged():
    """Mirror of TREND.md's Final Rule — a pillar with ~zero contribution must be surfaced, not
    quietly carried."""
    own = {"selection": 3.0, "breakout": 0.0, "leverage": 2.0, "trend": 1.0}
    pnl = all_coalitions(lambda s: sum(own[p] for p in s))
    rep = PillarAttribution().attribute(pnl, "shapley", ALL_PILLARS)
    assert "breakout" in rep.zero_contribution
    assert "selection" not in rep.zero_contribution


def test_an_unknown_mode_is_rejected():
    pnl = all_coalitions(lambda s: float(len(s)))
    with pytest.raises(ValueError):
        PillarAttribution().attribute(pnl, "bayesian-magic", ALL_PILLARS)


# ---------------------------------------------------------------- Invariant BL
def test_there_are_exactly_sixteen_coalition_subsets_and_they_are_unique():
    subs = coalition_subsets()
    assert len(subs) == 16
    assert len(set(subs)) == 16
    assert subs[0] == frozenset()
    assert subs[-1] == frozenset(ALL_PILLARS)


def test_coalitions_are_ordered_by_size():
    sizes = [len(s) for s in coalition_subsets()]
    assert sizes == sorted(sizes)


def test_every_pillar_declares_its_off_semantics():
    """A coalition is only runnable under the identical contract if OFF is defined for each."""
    assert set(OFF_SEMANTICS) == set(ALL_PILLARS)
    for p, pol in OFF_SEMANTICS.items():
        assert pol.pillar == p
        assert pol.description


def test_only_breakout_and_leverage_off_invalidate_cached_evidence():
    """selection-off and trend-off are pure combiner toggles — if they were marked as changing
    module evidence, every coalition would needlessly recompute."""
    changed = {p for p, pol in OFF_SEMANTICS.items() if pol.changes_module_evidence}
    assert changed == {"breakout", "leverage"}


def test_random_matched_frequency_is_a_zero_edge_null():
    r = random_matched_frequency(5000, 200, np.array([5, 10]), seed=11, ret_scale=0.01)
    assert r.shape == (5000,)
    assert abs(float(r.mean())) < 1e-3      # no edge, by construction
    assert int(np.count_nonzero(r)) <= 200  # matched trade count


def test_random_matched_frequency_is_deterministic_and_degrades_safely():
    a = random_matched_frequency(100, 10, np.array([3]), seed=5, ret_scale=0.01)
    b = random_matched_frequency(100, 10, np.array([3]), seed=5, ret_scale=0.01)
    assert np.array_equal(a, b)
    assert not random_matched_frequency(0, 10, np.array([3]), seed=5).any()
    assert not random_matched_frequency(100, 0, np.array([3]), seed=5).any()


def test_buy_and_hold_is_the_total_return():
    assert buy_and_hold(np.array([100.0, 150.0])) == pytest.approx(0.5)
    assert buy_and_hold(np.array([100.0, 50.0])) == pytest.approx(-0.5)


def test_buy_and_hold_degrades_rather_than_dividing_by_zero():
    assert buy_and_hold(np.array([100.0])) == 0.0
    assert buy_and_hold(np.array([])) == 0.0
    assert buy_and_hold(np.array([0.0, 10.0])) == 0.0


def test_the_five_single_pillar_ablations_are_named():
    assert len(SINGLE_PILLAR_ABLATIONS) == 5
    assert len(set(SINGLE_PILLAR_ABLATIONS)) == 5
