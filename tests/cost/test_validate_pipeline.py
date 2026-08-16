"""The remaining COST §12 validation pipeline: ablation, CPCV splits, realized labels, and the
purged walk-forward. All four shipped at 0%.

The one that matters most is the walk-forward's embargo. It is the no-lookahead guarantee for the
funding forecaster, and "the forecaster only saw history" is not a property you can eyeball — so
the rates here are ``arange``, which makes every value encode its own index and turns the
guarantee into an exact arithmetic assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.cost.validate.ablation import ablation_delta, drop_top_n
from convexity.cost.validate.cpcv import cpcv_splits, group_indices
from convexity.cost.validate.labels import (
    build_labels,
    realized_funding,
    realized_round_trip_cost,
)
from convexity.cost.validate.walkforward import purged_walkforward


# ---------------------------------------------------------------- drop-top-N
def test_drop_top_n_removes_the_largest_magnitude_entries():
    """Robustness check: does the result survive without its biggest contributors?"""
    out = drop_top_n(np.array([1.0, -9.0, 2.0, 8.0]), 2)
    assert sorted(out.tolist()) == [1.0, 2.0]


def test_drop_top_n_judges_by_MAGNITUDE_not_by_value():
    """A huge LOSS is as much of an outlier as a huge gain — dropping only positives would
    flatter the result."""
    assert (-50.0) not in drop_top_n(np.array([1.0, -50.0, 2.0]), 1).tolist()


def test_drop_top_n_preserves_the_original_order_of_survivors():
    out = drop_top_n(np.array([5.0, 1.0, 4.0, 2.0, 100.0]), 1)
    assert out.tolist() == [5.0, 1.0, 4.0, 2.0]


def test_drop_top_n_is_a_noop_for_a_nonpositive_n_or_empty_input():
    vals = np.array([3.0, 1.0])
    assert drop_top_n(vals, 0).tolist() == [3.0, 1.0]
    assert drop_top_n(vals, -1).tolist() == [3.0, 1.0]
    assert drop_top_n(np.array([]), 3).size == 0


def test_ablation_delta_reports_the_mean_shift():
    out = ablation_delta(np.array([2.0, 4.0]), np.array([1.0, 1.0]))
    assert out["mean_with"] == pytest.approx(3.0)
    assert out["mean_without"] == pytest.approx(1.0)
    assert out["mean_delta"] == pytest.approx(2.0)


def test_ablation_delta_degrades_on_an_empty_side():
    out = ablation_delta(np.array([]), np.array([1.0]))
    assert out["mean_delta"] == 0.0


# ---------------------------------------------------------------- CPCV splits
def test_default_cpcv_yields_the_fifteen_paths():
    splits = cpcv_splits()
    assert len(splits) == 15                     # C(6,2)
    assert len(set(splits)) == 15


def test_every_split_partitions_the_groups_exactly():
    """Train and test must be disjoint and together cover everything — an overlap would leak."""
    for train, test in cpcv_splits(n_groups=6, k_test=2):
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(6))
        assert len(test) == 2


def test_cpcv_is_capped_so_combinatorics_cannot_blow_up_the_VM():
    cpcv_splits(n_groups=8, k_test=2)            # C(8,2)=28, the stated ceiling
    with pytest.raises(ValueError, match="capped"):
        cpcv_splits(n_groups=9, k_test=2)


def test_group_indices_partition_every_observation_exactly_once():
    groups = group_indices(23, n_groups=6)
    assert len(groups) == 6
    joined = np.concatenate(groups)
    assert sorted(joined.tolist()) == list(range(23))


def test_group_indices_are_contiguous_blocks():
    """Contiguity is what makes the purge/embargo meaningful — shuffled groups would leak
    across time."""
    for g in group_indices(20, n_groups=4):
        assert g.tolist() == list(range(g[0], g[-1] + 1))


# ---------------------------------------------------------------- realized labels
def test_realized_funding_sums_the_signed_path():
    n = np.array([1000.0, 1000.0])
    r = np.array([1e-4, 2e-4])
    assert realized_funding(1, n, r) == pytest.approx(0.3)
    assert realized_funding(-1, n, r) == pytest.approx(-0.3)


def test_round_trip_cost_is_additive_over_its_five_components():
    assert realized_round_trip_cost(1.0, 2.0, 3.0, 4.0, 5.0) == pytest.approx(15.0)


def test_build_labels_assembles_cost_funding_and_shortfall():
    lab = build_labels(fees=1.0, half_spread=0.5, impact=0.25, realized_funding_value=2.0,
                       carry=0.25, gross_pnl=100.0, net_pnl=96.0)
    assert lab.realized_round_trip_cost == pytest.approx(4.0)
    assert lab.realized_funding_accrual == pytest.approx(2.0)
    assert lab.realized_implementation_shortfall == pytest.approx(4.0)


# ---------------------------------------------------------------- purged walk-forward
def test_the_embargo_actually_withholds_the_recent_past():
    """THE no-lookahead guarantee. rates=arange means every value equals its own index, so a
    last-value forecaster produces a residual of exactly (embargo + 1) at every step. Any leak of
    fresher data would shrink it."""
    rates = np.arange(60, dtype=float)
    last_value = lambda h: float(h[-1])  # noqa: E731
    for embargo in (1, 2, 5):
        res = purged_walkforward(rates, last_value, n_folds=4, embargo=embargo)
        assert res.residuals.size > 0
        assert np.allclose(res.residuals, embargo + 1)


def test_the_forecaster_is_never_handed_the_point_it_must_predict():
    seen_max = []

    def spy(history):
        seen_max.append(float(history[-1]))
        return float(history[-1])

    rates = np.arange(40, dtype=float)
    purged_walkforward(rates, spy, n_folds=4, embargo=1)
    # with embargo=1 the freshest admissible observation is k-2, so the final call cannot have
    # seen the last index.
    assert max(seen_max) <= len(rates) - 2


def test_a_perfect_forecaster_leaves_no_residual():
    rates = np.full(50, 0.0003)
    res = purged_walkforward(rates, lambda h: 0.0003, n_folds=4, embargo=1)
    assert res.bias == pytest.approx(0.0)
    assert res.rmse == pytest.approx(0.0)


def test_a_biased_forecaster_shows_up_as_bias_not_just_rmse():
    """Invariant E: a systematically low funding forecast is miscalibration, and the sign of the
    bias is what surfaces it."""
    rates = np.full(50, 0.001)
    res = purged_walkforward(rates, lambda h: 0.0005, n_folds=4, embargo=1)
    assert res.bias == pytest.approx(0.0005)
    assert res.rmse == pytest.approx(0.0005)


def test_a_series_too_short_to_score_degrades_to_empty():
    res = purged_walkforward(np.array([0.1, 0.2]), lambda h: 0.0, n_folds=4, embargo=5)
    assert res.residuals.size == 0
    assert res.bias == 0.0 and res.rmse == 0.0
