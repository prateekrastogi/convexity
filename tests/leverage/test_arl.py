"""CUSUM ARL Monte-Carlo verification on synthetic N(0,1) only (LEVERAGE.md §6.3, §10)."""

from __future__ import annotations

import pytest

from convexity.leverage.validate.arl import monte_carlo_arl


def test_mc_arl_in_control_order_of_magnitude():
    # §10: h_promote=5.0 sits above the >=52wk floor; the corrected-Siegmund closed form (141.9)
    # and the MC agree to ~0.1% at scale (100k runs) — at this test's n_runs=400 the MC standard
    # error is ~7wk, so assert agreement within 20% (≈3 sigma), which the OLD uncorrected form
    # (97.5, a 31% gap) fails.
    est = monte_carlo_arl(0.25, 5.0, n_runs=400, max_weeks=3_000, seed=1)
    # a strictly-in-control book should almost never falsely promote inside one year.
    assert est.arl_mc > 52.0
    assert est.arl_closed_form > 52.0
    # closed form within MC error of the empirical truth (the audit-found 45% gap would fail this).
    assert est.arl_closed_form == pytest.approx(est.arl_mc, rel=0.20)


def test_mc_arl_shorter_threshold_promotes_faster():
    # sanity / monotonicity: a LOWER h_promote crosses sooner (shorter ARL) under the same model.
    low = monte_carlo_arl(0.25, 2.0, n_runs=300, max_weeks=3_000, seed=2)
    high = monte_carlo_arl(0.25, 5.0, n_runs=300, max_weeks=3_000, seed=2)
    assert low.arl_mc < high.arl_mc
