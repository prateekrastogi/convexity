"""Almgren-Chriss trajectory solver (EXECUTION.md §2.4-2.5, §11.2): Invariant N-exec,
TWAP-limit recovery, schedule boundary conditions, front-loading mean-variance trade-off."""

from __future__ import annotations

import math

import numpy as np
import pytest

from convexity.cost.friction.spread_impact import SpreadImpactModel
from convexity.execution.execution import (
    ExecTarget,
    ExecutionDefaults,
    ExecutionEngine,
    check_lower_delta_bound,
    cost_facts_from_spread_impact,
    solve_trajectory,
)


def test_impact_coeffs_match_cost_byte_for_byte():
    # Invariant N-exec (§11.2 table): eta/gamma used inside the solve are the LOCAL LINEARIZATION of
    # COST's fitted SpreadImpactModel square-root impact law at the per-slice trade-size scale
    # (§2.2) — read byte-for-byte off the REAL model.impact_frac(...), exact float equality, no
    # tolerance, no stub. (Earlier version asserted only against a fabricated .eta/.gamma stub.)
    rng = np.random.default_rng(7)
    n = 6
    # a REAL fitted COST model (src/convexity/cost/friction/spread_impact.py), anchored to a median universe.
    model = SpreadImpactModel.anchor(adv_median=5e7, amihud_median=2e-9)
    amihud = rng.uniform(5e-10, 8e-9, n)  # per-name Amihud_i illiquidity (COST §5.3)
    slice_notional = rng.uniform(2e4, 2e5, n)  # per-slice dollar trade size (§2.2 linearization scale)
    theta = np.full(n, 1.2e-4)
    sigma = rng.uniform(0.001, 0.01, n)
    facts = cost_facts_from_spread_impact(
        model, amihud, theta=theta, sigma=sigma, slice_notional=slice_notional
    )
    # eta_i is EXACTLY COST's own impact_frac at the slice scale — byte-for-byte, no tolerance.
    for i in range(n):
        assert facts.eta[i] == model.impact_frac(float(amihud[i]), float(slice_notional[i]))
    assert np.array_equal(facts.gamma, np.zeros(n))  # default permanent_fraction=0 -> all-temporary

    # and the solved kappa is exactly the value implied by those byte-identical coefficients.
    cfg = ExecutionDefaults(rho=1.0, band_c=0.0, lambda_=2.5)  # band off: every drift trades
    target = ExecTarget(w_target=np.full(n, 0.05), shortable=np.ones(n, dtype=bool))
    carried = np.full(n, 0.03)
    plan = ExecutionEngine().on_target(target, carried, facts, False, cfg)
    for i, traj in plan.trajectories.items():
        expected = math.sqrt(cfg.lambda_ * float(sigma[i]) ** 2 / float(facts.eta[i]))
        assert traj.kappa == expected  # same floats, same expression -> exact


def test_permanent_fraction_splits_cost_impact():
    # §2.2: the temporary/permanent split of COST's ONE fitted impact number is a RED governance
    # choice, not a fitted quantity. With permanent_fraction=f, eta+gamma == COST's impact_frac and
    # gamma == f * impact_frac exactly (byte-for-byte), never a fabricated separate permanent fit.
    model = SpreadImpactModel.anchor(adv_median=5e7, amihud_median=2e-9)
    amihud = np.array([1e-9, 3e-9, 6e-9])
    slice_notional = np.array([5e4, 1e5, 2e5])
    theta = np.full(3, 1.2e-4)
    sigma = np.full(3, 0.004)
    f = 0.3
    facts = cost_facts_from_spread_impact(
        model, amihud, theta=theta, sigma=sigma, slice_notional=slice_notional,
        permanent_fraction=f,
    )
    for i in range(3):
        impact = model.impact_frac(float(amihud[i]), float(slice_notional[i]))
        assert facts.gamma[i] == f * impact
        assert facts.eta[i] == (1.0 - f) * impact
    with pytest.raises(ValueError):
        cost_facts_from_spread_impact(
            model, amihud, theta=theta, sigma=sigma, slice_notional=slice_notional,
            permanent_fraction=1.5,
        )


def test_schedule_sums_to_x_and_boundary_conditions():
    for lam in (0.0, 1.0, 25.0):
        res = solve_trajectory(
            -0.037, delta=2.0, n=10, sigma_i=0.004, eta_i=2e-4, gamma_i=1e-5, lambda_i=lam
        )
        assert res.schedule.shape == (10,)
        assert abs(float(res.schedule.sum()) - (-0.037)) < 1e-15  # X_0 = x_i, X_n = 0 (§2.1)


def test_front_loading_reduces_variance_and_raises_expected_cost():
    # §2.3's mean-variance trade-off: kappa > 0 lowers Var[C] below TWAP and raises E[C].
    kw = dict(delta=2.0, n=20, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0)
    twap = solve_trajectory(0.05, lambda_i=0.0, **kw)
    fast = solve_trajectory(0.05, lambda_i=50.0, **kw)
    assert fast.kappa > 0.0
    assert fast.var_cost < twap.var_cost
    assert fast.e_cost > twap.e_cost
    # front-loaded: first slice strictly larger than the uniform slice (§2.4).
    assert abs(fast.schedule[0]) > abs(twap.schedule[0])


def test_twap_variance_converges_to_closed_form():
    # §2.5: the kappa=0 TWAP variance is x^2 sigma^2 Delta / 3 in the continuum limit.
    x, sigma, delta = 0.04, 0.006, 3.0
    res = solve_trajectory(x, delta=delta, n=500, sigma_i=sigma, eta_i=1e-4, gamma_i=0.0, lambda_i=0.0)
    assert res.var_cost == pytest.approx(x * x * sigma * sigma * delta / 3.0, rel=5e-3)


def test_delta_regularity_bounds_raise():
    kw = dict(n=10, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=0.0)
    for bad_delta in (0.0, -1.0, 168.0, 500.0, float("nan")):  # §2.1: 0 < Delta << 7 days (UPPER)
        with pytest.raises(ValueError):
            solve_trajectory(0.01, delta=bad_delta, **kw)
    with pytest.raises(ValueError):
        solve_trajectory(0.01, delta=2.0, n=0, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=0.0)
    with pytest.raises(ValueError):  # missing COST fit is never silently defaulted (P6, §13)
        solve_trajectory(0.01, delta=2.0, n=10, sigma_i=0.004, eta_i=0.0, gamma_i=0.0, lambda_i=0.0)
    with pytest.raises(ValueError):
        solve_trajectory(0.01, delta=2.0, n=10, sigma_i=0.004, eta_i=float("nan"), gamma_i=0.0, lambda_i=0.0)


def test_lower_delta_bound_enforced_by_check_helper():
    # §2.1 LOWER bound + §11.1's "raises if delta fails EITHER bound" contract: the lower bound is a
    # VENUE-metadata condition (delta >= n / lambda_hat_fill,i, using this name's causal fill-rate
    # forecast), NOT an Almgren-Chriss quantity. It is enforced by check_lower_delta_bound (and threaded
    # through _plan from CostFacts.lambda_fill) strictly UPSTREAM of the closed-form solve, so
    # solve_trajectory keeps §11.1's exact 7-parameter signature (no venue arg leaks into the AC math)
    # while the bound is still enforceable end-to-end — the earlier version added an extra
    # `lambda_fill_i` kwarg to solve_trajectory beyond §11.1's named signature (the auditor's gap).
    # lambda_fill = 2 slices/hour, n=10 -> need delta >= 5.0 hours.
    with pytest.raises(ValueError):  # delta=2.0 < 5.0: too short to clear 10 slices, rejected (P6)
        check_lower_delta_bound(2.0, 10, 2.0)
    check_lower_delta_bound(6.0, 10, 2.0)  # delta=6.0 >= 5.0: clears the bound, no raise
    check_lower_delta_bound(5.0, 10, 2.0)  # exactly at the bound is allowed (>=, not >)
    # a degenerate (non-positive / non-finite) fill-rate forecast is itself a violation.
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            check_lower_delta_bound(6.0, 10, bad)
    # and solve_trajectory ITSELF no longer takes a lambda_fill_i argument (spec-exact signature).
    import inspect
    assert "lambda_fill_i" not in inspect.signature(solve_trajectory).parameters


def test_cost_facts_lambda_fill_threads_lower_bound_into_plan():
    # the CostFacts.lambda_fill array threads §2.1's lower bound through the plan path: an on_target
    # rebalance whose delta is too short for a name's fill rate raises rather than silently solving.
    from convexity.cost.friction.spread_impact import SpreadImpactModel
    from convexity.execution.execution import (
        ExecutionDefaults, ExecutionEngine, ExecTarget, cost_facts_from_spread_impact,
    )
    n = 3
    model = SpreadImpactModel.anchor(adv_median=5e7, amihud_median=2e-9)
    amihud = np.full(n, 2e-9)
    slice_notional = np.full(n, 5e4)
    # cfg.delta_hours=2.0, cfg.n_slices=10 -> need lambda_fill >= 5.0 slices/hr; supply 1.0 (too slow).
    facts = cost_facts_from_spread_impact(
        model, amihud, theta=np.full(n, 1.2e-4), sigma=np.full(n, 0.004),
        slice_notional=slice_notional, lambda_fill=np.full(n, 1.0),
    )
    cfg = ExecutionDefaults(rho=1.0, band_c=0.0, lambda_=2.5)  # band off: names trade
    target = ExecTarget(w_target=np.full(n, 0.05), shortable=np.ones(n, dtype=bool))
    with pytest.raises(ValueError):
        ExecutionEngine().on_target(target, np.full(n, 0.03), facts, False, cfg)


def test_sign_conventions():
    # sell trade: every slice is a sell; permanent+temporary expected cost still positive (§2.2).
    res = solve_trajectory(
        -0.05, delta=2.0, n=8, sigma_i=0.004, eta_i=2e-4, gamma_i=1e-5, lambda_i=10.0
    )
    assert np.all(res.schedule < 0.0)
    assert res.e_cost > 0.0 and res.var_cost > 0.0


def test_twap_limit_recovery():
    # §2.4's lambda -> 0 limit: kappa == 0 and the schedule is EXACTLY uniform.
    res = solve_trajectory(0.02, delta=2.0, n=16, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=0.0)
    assert res.kappa == 0.0
    assert np.all(res.schedule == 0.02 / 16)


def test_delta_to_zero_recovers_instantaneous_backtest_fill():
    # Sec 13 "Delta->0 degenerate window": as Delta->0, kappa*Delta->0 and the trajectory
    # degenerates to an instantaneous fill -- the backtest frictionless assumption is the
    # Delta->0, lambda->0 corner of this pillar own framework. Var[C] vanishes in that limit.
    x = 0.05
    prev_v = float("inf")
    var_at_wide = None
    res = None
    for delta in (10.0, 1.0, 0.1, 0.01, 1e-3):
        res = solve_trajectory(
            x, delta=delta, n=10, sigma_i=0.004, eta_i=2e-4, gamma_i=1e-5, lambda_i=25.0
        )
        assert abs(float(res.schedule.sum()) - x) < 1e-13  # boundary conditions hold throughout
        assert res.var_cost <= prev_v + 1e-18  # exposure-during-window risk shrinks as Delta->0
        if var_at_wide is None:
            var_at_wide = res.var_cost
        prev_v = res.var_cost
    # Var[C] ~ sigma^2 * Delta * x^2 -> 0 as Delta -> 0: the Delta=1e-3 window carries <0.1% of the
    # Delta=10 window's execution-exposure risk (the instantaneous-fill limit, section 13).
    assert res.var_cost < 1e-3 * var_at_wide
    # permanent-impact expected cost is path/Delta-independent (gamma*x^2/2), the residual floor.
    assert res.e_cost >= 0.5 * 1e-5 * x * x - 1e-18
