"""CUSUM ARL Monte-Carlo verification on synthetic N(0,1) only (LEVERAGE.md §6.3, §10)."""

from __future__ import annotations

from convexity.leverage.validate.arl import monte_carlo_arl


def test_mc_arl_in_control_order_of_magnitude():
    # §10: the empirical in-control ARL under N(0,1) surprises is on the order of the target ~52wk
    # (the closed-form is a conservative approximation; h_promote=5.0 sits above the 52-target so
    # the true ARL is comfortably >= a year of weekly obs — an infrequent false promotion).
    est = monte_carlo_arl(0.25, 5.0, n_runs=400, max_weeks=3_000, seed=1)
    # a strictly-in-control book should almost never falsely promote inside one year.
    assert est.arl_mc > 52.0
    # and the closed form should be the same order of magnitude as the MC estimate (both large).
    assert est.arl_closed_form > 52.0


def test_mc_arl_shorter_threshold_promotes_faster():
    # sanity / monotonicity: a LOWER h_promote crosses sooner (shorter ARL) under the same model.
    low = monte_carlo_arl(0.25, 2.0, n_runs=300, max_weeks=3_000, seed=2)
    high = monte_carlo_arl(0.25, 5.0, n_runs=300, max_weeks=3_000, seed=2)
    assert low.arl_mc < high.arl_mc
