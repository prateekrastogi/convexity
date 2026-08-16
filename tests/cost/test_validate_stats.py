"""COST §12 validation statistics: block-bootstrap CIs and PSR/DSR.

These gate promotion — DSR is what stops a candidate that "won" by chance across the BLUE grid
from being deployed — and the whole module sat at 0% coverage. The assertions below are the
mathematical properties the gate depends on, not smoke tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.cost.validate.baselines import (
    flat_bps_cost,
    last_funding_only_forecast,
    no_funding_cost,
    perfect_foresight_funding,
)
from convexity.cost.validate.stats import (
    block_bootstrap_quantile,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


# ---------------------------------------------------------------- block bootstrap
def test_block_bootstrap_brackets_the_sample_quantile():
    rng = np.random.default_rng(0)
    series = rng.normal(0.0, 1.0, 2000)
    med, lo, hi = block_bootstrap_quantile(series, 0.5, block_length=20, n_resamples=200, seed=1)
    assert lo <= med <= hi
    assert abs(med) < 0.2                      # median of a centred normal
    assert lo < 0.0 < hi                       # the CI contains the truth


def test_block_bootstrap_is_deterministic_under_a_seed():
    series = np.linspace(-1.0, 1.0, 500)
    a = block_bootstrap_quantile(series, 0.9, block_length=10, n_resamples=50, seed=7)
    b = block_bootstrap_quantile(series, 0.9, block_length=10, n_resamples=50, seed=7)
    assert a == b


def test_block_bootstrap_on_an_empty_series_is_zero_not_an_exception():
    """Validation runs unattended; an empty fold must degrade, not raise."""
    assert block_bootstrap_quantile(np.array([]), 0.5, block_length=5) == (0.0, 0.0, 0.0)


def test_block_bootstrap_ci_widens_as_the_quantile_goes_further_into_the_tail():
    rng = np.random.default_rng(3)
    series = rng.normal(size=3000)
    _m, lo50, hi50 = block_bootstrap_quantile(series, 0.5, block_length=20, n_resamples=200, seed=2)
    _m, lo99, hi99 = block_bootstrap_quantile(series, 0.99, block_length=20, n_resamples=200, seed=2)
    assert (hi99 - lo99) > (hi50 - lo50)


# ---------------------------------------------------------------- PSR
def test_psr_is_a_probability():
    for sr in (-1.0, 0.0, 0.5, 2.0):
        p = probabilistic_sharpe_ratio(sr, n_obs=500)
        assert 0.0 <= p <= 1.0


def test_psr_rises_with_the_observed_sharpe():
    lo = probabilistic_sharpe_ratio(0.2, n_obs=500)
    hi = probabilistic_sharpe_ratio(1.5, n_obs=500)
    assert hi > lo


def test_psr_rises_with_sample_size_for_a_positive_sharpe():
    """More evidence for the same edge must raise confidence."""
    assert probabilistic_sharpe_ratio(0.8, n_obs=2000) > probabilistic_sharpe_ratio(0.8, n_obs=100)


def test_psr_at_the_benchmark_is_one_half():
    assert probabilistic_sharpe_ratio(0.7, n_obs=1000, sr_benchmark=0.7) == pytest.approx(0.5)


def test_psr_degrades_to_zero_with_no_observations():
    assert probabilistic_sharpe_ratio(2.0, n_obs=1) == 0.0
    assert probabilistic_sharpe_ratio(2.0, n_obs=0) == 0.0


# ---------------------------------------------------------------- DSR
def test_dsr_equals_psr_when_only_one_candidate_was_tried():
    """With n_trials=1 there is no multiple-testing inflation to deflate."""
    assert deflated_sharpe_ratio(1.0, n_obs=500, n_trials=1) == pytest.approx(
        probabilistic_sharpe_ratio(1.0, n_obs=500, sr_benchmark=0.0))


def test_dsr_falls_as_more_candidates_are_tried():
    """THE point of the gate: the same Sharpe is less impressive after more trials."""
    few = deflated_sharpe_ratio(1.0, n_obs=500, n_trials=2)
    many = deflated_sharpe_ratio(1.0, n_obs=500, n_trials=200)
    assert many < few


def test_dsr_never_leaves_the_unit_interval():
    for trials in (1, 5, 50, 5000):
        assert 0.0 <= deflated_sharpe_ratio(1.2, n_obs=800, n_trials=trials) <= 1.0


def test_dsr_treats_a_nonpositive_trial_count_as_one():
    assert deflated_sharpe_ratio(1.0, 500, n_trials=0) == deflated_sharpe_ratio(1.0, 500, 1)


# ---------------------------------------------------------------- §12 baselines
def test_flat_bps_cost_is_linear_in_notional():
    assert flat_bps_cost(10_000.0, 13.0) == pytest.approx(13.0)
    assert flat_bps_cost(20_000.0, 13.0) == pytest.approx(2 * flat_bps_cost(10_000.0, 13.0))


def test_no_funding_cost_sums_its_three_components():
    assert no_funding_cost(1.0, 2.0, 0.5) == pytest.approx(3.5)


def test_last_funding_only_scales_with_horizon_and_side():
    long = last_funding_only_forecast(1, 1000.0, 0.0001, 3)
    short = last_funding_only_forecast(-1, 1000.0, 0.0001, 3)
    assert long == pytest.approx(0.3)
    assert short == pytest.approx(-long)


def test_perfect_foresight_is_the_realized_sum():
    notionals = np.array([1000.0, 1000.0, 500.0])
    rates = np.array([1e-4, -2e-4, 3e-4])
    assert perfect_foresight_funding(1, notionals, rates) == pytest.approx(
        float(np.sum(notionals * rates)))


def test_perfect_foresight_flips_with_side():
    n = np.array([100.0, 100.0])
    r = np.array([1e-4, 1e-4])
    assert perfect_foresight_funding(-1, n, r) == pytest.approx(-perfect_foresight_funding(1, n, r))
