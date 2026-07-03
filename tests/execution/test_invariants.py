"""Cross-cutting invariants (EXECUTION.md §13, §11.2): Invariant W (causality firewall),
SHORTABLE-GATE (composed quantile-flip/unshortable edge case), C-cfg-exec, and the Perold
shortfall decomposition arithmetic (§7.1)."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from convexity.execution.execution import (
    CostFacts,
    ExecTarget,
    ExecutionDefaults,
    ExecutionEngine,
    implementation_shortfall,
    lambda_from_budget,
    solve_trajectory,
)

_N = 5
_CFG = ExecutionDefaults(rho=2.0, band_c=0.2, lambda_=5.0)
_FACTS = CostFacts(
    eta=np.full(_N, 2e-4),
    gamma=np.full(_N, 1e-5),
    theta=np.full(_N, 1.3e-4),
    sigma=np.full(_N, 0.004),
)


def _inputs() -> tuple[ExecTarget, np.ndarray]:
    w_target = np.array([0.05, 0.03, -0.04, -0.04, 0.0])
    carried = np.array([0.02, 0.031, -0.01, -0.06, 0.01])
    return ExecTarget(w_target, np.ones(_N, dtype=bool)), carried


def test_on_target_signature_admits_no_realized_window_channel():
    # Invariant W (§13, §11.2), STRUCTURAL half: on_target's signature is EXACTLY §11.1/§12's build
    # contract (target, carried, cost, regime_defensive, cfg) — no realized-(t_0, t_0+Delta] window
    # argument. The plan CANNOT depend on window data because there is no parameter through which it
    # could enter. This is not a tautology: the only way to wire realized data into the plan is to
    # add a parameter carrying it, and that FAILS this assertion (the firewall is the signature).
    params = list(inspect.signature(ExecutionEngine.on_target).parameters)
    assert params == ["self", "target", "carried", "cost", "regime_defensive", "cfg"]
    # every one of those parameters is an F_{t_0}-measurable object per §11.1 (w_target/w_carried are
    # set at the Friday close; cost is COST's fit at t_0; regime_defensive/cfg are t_0 config) — none
    # is a realized-window series, so no future-data channel exists at all.


def test_planned_trajectory_ignores_realized_window_data():
    # Invariant W (§13, §11.2), BEHAVIORAL half: the PLANNED RebalancePlan is a pure function of its
    # F_{t_0} inputs — two calls with identical inputs return identical plans, and ONLY the post-hoc
    # ShortfallReport (the sole function permitted to read (t_0, t_0+Delta] data) depends on the
    # realized window. There is deliberately NO window parameter on on_target (see the signature test
    # above): realized data reaches the shortfall report DIRECTLY, never through the plan.
    target, carried = _inputs()
    engine = ExecutionEngine()
    rng = np.random.default_rng(0)

    decision = np.full(_N, 100.0)
    # realistic vs. wildly-corrupt realized window observations (volume/price/fill records).
    window_real = {"fill_price": decision * (1.0 + rng.normal(0.0, 1e-3, _N))}
    window_garbage = {"fill_price": decision * (1.0 + rng.normal(0.0, 0.5, _N))}  # 50% garbage

    # PLAN side: the plan never sees the window at all; two identical-input calls agree bit-for-bit.
    plan_a = engine.on_target(target, carried, _FACTS, False, _CFG)
    plan_b = engine.on_target(target, carried, _FACTS, False, _CFG)
    assert np.array_equal(plan_a.trade, plan_b.trade)
    assert np.array_equal(plan_a.x, plan_b.x)
    assert np.array_equal(plan_a.band, plan_b.band)
    assert plan_a.reason == plan_b.reason
    assert set(plan_a.trajectories) == set(plan_b.trajectories)
    for i in plan_a.trajectories:
        assert np.array_equal(plan_a.trajectories[i].schedule, plan_b.trajectories[i].schedule)
        assert plan_a.trajectories[i].kappa == plan_b.trajectories[i].kappa

    # REPORT side: only the post-hoc ShortfallReport may (and does) depend on realized window data.
    notional = np.array([3e4, 0.0, -2e4, 2e4, -1e4])
    fees = np.abs(notional) * 5e-4
    report_real = implementation_shortfall(
        decision, window_real["fill_price"], notional, nav=1e6, fee_component=fees
    )
    report_garbage = implementation_shortfall(
        decision, window_garbage["fill_price"], notional, nav=1e6, fee_component=fees
    )
    assert report_real.is_total != report_garbage.is_total  # only the realized report differs


def test_unshortable_short_target_raises_ops_observation():
    # Invariant SHORTABLE-GATE (§13): a non-zero short target without shortable=True is (a) never
    # traded, (b) never silently dropped, (c) raised to OPS as an observation.
    i = 2
    w_target = np.array([0.05, 0.03, -0.02, -0.04, 0.0])
    shortable = np.ones(_N, dtype=bool)
    shortable[i] = False
    carried = np.zeros(_N)
    plan = ExecutionEngine().on_target(ExecTarget(w_target, shortable), carried, _FACTS, False, _CFG)
    assert plan.trade[i] == 0.0
    assert plan.reason[i] == "SHORTABLE-GATE"
    assert plan.ops_observations[i].code == "SHORTABLE-GATE"
    assert i not in plan.trajectories
    # the other short leg (shortable) still trades in full (quantile flip from flat, §13).
    assert plan.trade[3] == plan.x[3]


def test_shortfall_decomposition_lines_match_independent_computation():
    # §7.1: EACH of the four Perold lines is verified against an INDEPENDENT hand computation, not
    # just the sum. (The auditor flagged the sum-to-total assertion as a by-construction tautology,
    # since impact absorbs the residual; this test defends the delay/spread/impact SPLIT itself, with
    # all realized inputs supplied so no line falls back to a residual/constant.) is_total is ALSO
    # verified against the standalone Perold identity fees/nav + price-shortfall/nav, independently of
    # the four-line reconstruction — so if the sub-line split were wrong, is_total (computed the
    # other way) would disagree with the reconstructed sum and this test would fail.
    nav = 1e6
    decision = np.array([100.0, 50.0, 200.0])
    arrival = np.array([100.05, 50.10, 200.0])  # decision -> arrival drift (delay)
    fill = np.array([100.2, 49.9, 200.0])  # arrival -> fill drift (impact, net of spread)
    notional = np.array([1e4, -5e3, 0.0])  # buy, sell, no-trade
    fees = np.array([5.0, 2.5, 0.0])
    half_spread = np.array([1.5e-4, 2.0e-4, 0.0])  # realized per-name half-spread (frac of notional)
    rep = implementation_shortfall(
        decision, fill, notional, nav=nav, fee_component=fees,
        arrival_price=arrival, realized_half_spread=half_spread,
    )

    # (1) explicit: fees / nav.
    assert rep.explicit_cost == (5.0 + 2.5) / nav
    # (2) delay: signed-notional (arrival-decision)/decision, hand-computed per name.
    exp_delay = (
        (100.05 - 100.0) / 100.0 * 1e4 + (50.10 - 50.0) / 50.0 * (-5e3)
    ) / nav
    assert rep.delay_cost == exp_delay
    # (3) spread: realized half-spread on |notional|, hand-computed.
    exp_spread = (1.5e-4 * 1e4 + 2.0e-4 * 5e3) / nav
    assert rep.realized_spread == exp_spread
    # (4) impact: arrival->fill drift on signed notional, MINUS the spread line — hand-computed
    #     INDEPENDENTLY (not read back off the report's residual), so the split is genuinely checked.
    exp_impact = (
        (100.2 - 100.05) / 100.0 * 1e4 + (49.9 - 50.10) / 50.0 * (-5e3)
    ) / nav - exp_spread
    assert rep.realized_impact == pytest.approx(exp_impact, rel=0, abs=1e-18)

    # is_total via the STANDALONE Perold identity (fees/nav + total price shortfall/nav), computed a
    # DIFFERENT way than the four-line sum — a genuine cross-check of the whole decomposition.
    price_shortfall = (
        (100.2 - 100.0) / 100.0 * 1e4 + (49.9 - 50.0) / 50.0 * (-5e3)
    ) / nav
    exp_is_total = (5.0 + 2.5) / nav + price_shortfall
    assert rep.is_total == pytest.approx(exp_is_total, rel=0, abs=1e-18)
    # and the four lines reconcile to it (by-construction, but now every line is independently pinned).
    assert rep.is_total == pytest.approx(
        rep.explicit_cost + rep.realized_spread + rep.realized_impact + rep.delay_cost,
        rel=0, abs=1e-18,
    )
    assert rep.is_total > 0.0  # buy above / sell below the decision print: both costs


def test_shortfall_realized_delay_and_spread_lines():
    # §7.1 (gap-close): when the caller supplies the REALIZED arrival price and realized half-spread,
    # the delay/spread lines are the GENUINE Perold numbers (decision->arrival drift; realized
    # half-spread paid), not a hardcoded 0 / a fallback constant. The four lines still sum to the
    # exact price-based total, and delay is nonzero exactly when arrival != decision.
    decision = np.array([100.0, 50.0])
    arrival = np.array([100.1, 49.95])  # price drifted between decision and first child fill
    fill = np.array([100.3, 49.80])  # further drift from arrival to the notional-weighted fill
    notional = np.array([2e4, -1e4])  # buy, sell
    fees = np.array([10.0, 5.0])
    half_spread = np.array([2e-4, 1e-4])  # realized per-name half-spread paid (fraction of notional)
    rep = implementation_shortfall(
        decision, fill, notional, nav=1e6, fee_component=fees,
        arrival_price=arrival, realized_half_spread=half_spread,
    )
    # delay = realized decision->arrival drift on signed notional (both are costs here).
    exp_delay = (
        (100.1 - 100.0) / 100.0 * 2e4 + (49.95 - 50.0) / 50.0 * (-1e4)
    ) / 1e6
    assert rep.delay_cost == exp_delay
    assert rep.delay_cost > 0.0  # genuinely nonzero, not the hardcoded-0 fallback
    # spread = realized half-spread on |notional|, NOT the accounting.HALF_SPREAD_FRAC fallback.
    exp_spread = (2e-4 * 2e4 + 1e-4 * 1e4) / 1e6
    assert rep.realized_spread == exp_spread
    # the four lines still reconcile to is_total exactly.
    assert rep.is_total == (
        rep.explicit_cost + rep.realized_spread + rep.realized_impact + rep.delay_cost
    )
    # and the arrival-given delay line differs from the no-arrival (delay=0) fallback.
    rep_no_arrival = implementation_shortfall(
        decision, fill, notional, nav=1e6, fee_component=fees, realized_half_spread=half_spread
    )
    assert rep_no_arrival.delay_cost == 0.0
    assert rep_no_arrival.delay_cost != rep.delay_cost


def test_shortfall_input_validation():
    with pytest.raises(ValueError):
        implementation_shortfall(
            np.array([100.0]), np.array([100.0]), np.array([1e4]),
            nav=0.0, fee_component=np.array([0.0]),
        )
    with pytest.raises(ValueError):  # traded name with a NaN fill is never silently priced
        implementation_shortfall(
            np.array([100.0]), np.array([np.nan]), np.array([1e4]),
            nav=1e6, fee_component=np.array([0.0]),
        )


def test_red_constants_are_not_per_instrument():
    # Invariant C-cfg-exec (§10): every ExecutionDefaults field is a book/class-level SCALAR —
    # a per-instrument array-valued lambda/rho/c/Delta/n is the bug.
    for f in dataclasses.fields(ExecutionDefaults):
        assert f.type in ("float", "int"), f"{f.name} must be a scalar, got {f.type}"
    cfg = ExecutionDefaults(rho=1.0, band_c=0.1)
    for f in dataclasses.fields(cfg):
        assert np.isscalar(getattr(cfg, f.name))


def test_sign_conventions():
    # long trade filled ABOVE decision = cost; short trade filled ABOVE decision = credit (§7.1).
    decision = np.array([100.0])
    above = np.array([101.0])
    buy = implementation_shortfall(decision, above, np.array([1e4]), nav=1e6,
                                   fee_component=np.array([0.0]))
    sell = implementation_shortfall(decision, above, np.array([-1e4]), nav=1e6,
                                    fee_component=np.array([0.0]))
    assert buy.is_total > 0.0
    assert sell.is_total < buy.is_total  # selling above the decision print is the good side


def test_twap_limit_recovery():
    res = solve_trajectory(0.05, delta=1.0, n=4, sigma_i=0.01, eta_i=1e-4, gamma_i=0.0, lambda_i=0.0)
    assert res.kappa == 0.0
    assert np.all(res.schedule == 0.05 / 4)


def test_lambda_from_budget_realizes_the_beta_to_lambda_derivation():
    # §2.5 (gap-close): lambda is DERIVED from a stated variance budget beta via the closed form
    # kappa_i^2 = (15/(2*Delta^2))*[1 - (3*beta/Delta)*(w_target/x)^2] and
    # lambda = kappa^2 * eta / sigma^2 — never a raw injected scalar, never fitted on P&L. This test
    # pins the 2/15 expansion coefficient (audit-corrected: an earlier revision shipped 17/45,
    # wrong by 17/6 — see test_solved_trajectory_respects_the_variance_budget for the NON-circular
    # check that the solved trajectory actually honors the budget) and the edge case.
    eta, sigma, delta, beta = 2e-4, 0.004, 0.0119, 0.01
    # a name with a LARGE required trade relative to its target weight (illiquid-tail regime): the
    # bracket is positive, kappa > 0 binds, and lambda recovers the exact closed form.
    x_i, w_i = 0.05, 0.02  # w_target/x = 0.4 -> (w/x)^2 = 0.16, small enough to keep bracket > 0
    bracket = 1.0 - (3.0 * beta / delta) * (w_i / x_i) ** 2
    assert bracket > 0.0  # this regime genuinely needs front-loading
    exp_kappa_sq = (15.0 / (2.0 * delta * delta)) * bracket
    exp_lambda = exp_kappa_sq * eta / (sigma * sigma)
    got = lambda_from_budget(x_i, w_i, delta=delta, beta=beta, eta_i=eta, sigma_i=sigma)
    assert got == pytest.approx(exp_lambda, rel=1e-12)
    # feeding that lambda back through kappa^2 = lambda*sigma^2/eta recovers the budget kappa exactly.
    assert (got * sigma * sigma / eta) == pytest.approx(exp_kappa_sq, rel=1e-12)

    # §2.5 EDGE CASE: the worked example (x/w_target ~ 0.30, i.e. w_target/x ~ 3.33, beta=0.01,
    # Delta~0.0119) lands the bracket NEGATIVE -> pure TWAP, lambda = 0. This is the expected
    # non-degenerate liquid-core behavior, NOT a failure.
    x_liq, w_liq = 0.30, 1.0  # w_target/x ~ 3.33 (§2.5's worked numeric example)
    edge_bracket = 1.0 - (3.0 * beta / delta) * (w_liq / x_liq) ** 2
    assert edge_bracket < 0.0
    assert lambda_from_budget(x_liq, w_liq, delta=delta, beta=beta, eta_i=eta, sigma_i=sigma) == 0.0

    # governance guards: beta must be a real budget in (0,1), x_i nonzero, eta/sigma positive.
    for bad_beta in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            lambda_from_budget(0.05, 0.02, delta=delta, beta=bad_beta, eta_i=eta, sigma_i=sigma)
    with pytest.raises(ValueError):
        lambda_from_budget(0.0, 0.02, delta=delta, beta=beta, eta_i=eta, sigma_i=sigma)
    with pytest.raises(ValueError):
        lambda_from_budget(0.05, 0.02, delta=delta, beta=beta, eta_i=0.0, sigma_i=sigma)


def test_shipped_lambda_default_is_the_anchored_twap_limit():
    # EXECUTION.md §2.5 ("shipped RED default" para) + §10 anchor lambda_=0.0 as a DESIGNED
    # knob: the risk-neutral / TWAP limit (§2.4's λ→0 ⇒ κ=0 identity), the value that reproduces
    # the λ=0 full-rebalance backtest in docs/EDGE.md. This pins the default's ORIGIN and value,
    # and confirms anchoring it changed no numeric result (the default still yields pure TWAP).
    cfg = ExecutionDefaults(rho=1.0, band_c=0.1)  # only the two genuinely-unassigned RED fields
    assert cfg.lambda_ == 0.0  # spec-cited shipped default, not a fitted or invented number
    # the default, fed through the solver, IS the TWAP baseline: kappa == 0, uniform schedule.
    res = solve_trajectory(
        0.05, delta=2.0, n=8, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=cfg.lambda_
    )
    assert res.kappa == 0.0
    assert np.all(res.schedule == 0.05 / 8)
    # the injection path stays open: a stated positive budget still produces a positive lambda
    # (illiquid-tail regime) that overrides the neutral default.
    injected = lambda_from_budget(0.05, 0.02, delta=0.0119, beta=0.01, eta_i=2e-4, sigma_i=0.004)
    assert injected > 0.0
    assert ExecutionDefaults(rho=1.0, band_c=0.1, lambda_=injected).lambda_ == injected


def test_solved_trajectory_respects_the_variance_budget():
    """NON-circular §2.5 check (audit E-1 test-gap): the trajectory solved from lambda_from_budget's
    lambda must ACTUALLY deliver Var[C] <= beta * w_target^2 * sigma^2 (the stated budget), within
    the small-bracket expansion tolerance — and the OLD 17/45 coefficient's lambda must violate the
    same bound, so this test cannot pass by construction."""
    delta_wk = 2.0 / 168.0  # 2h window as a fraction of a week (the §2.5 units convention)
    sigma_wk = 0.04  # per-sqrt-week price vol
    eta = 2e-4
    w, x = 0.02, 0.06
    # choose beta so the bracket is small (0.05) — inside the expansion's stated domain of validity.
    beta = 0.95 * delta_wk / (3.0 * (w / x) ** 2)
    lam = lambda_from_budget(x, w, delta=delta_wk, beta=beta, eta_i=eta, sigma_i=sigma_wk)
    res = solve_trajectory(x, delta=delta_wk, n=50, sigma_i=sigma_wk, eta_i=eta, gamma_i=0.0,
                           lambda_i=lam)
    budget = beta * w * w * sigma_wk * sigma_wk
    assert res.var_cost <= budget * 1.05  # measured 1.0345 at bracket=0.05
    # the OLD wrong coefficient (45/17 in kappa^2, i.e. lambda scaled by (45/17)/(15/2)) violates:
    lam_old = lam * (45.0 / 17.0) / (15.0 / 2.0)
    res_old = solve_trajectory(x, delta=delta_wk, n=50, sigma_i=sigma_wk, eta_i=eta, gamma_i=0.0,
                               lambda_i=lam_old)
    assert res_old.var_cost > budget * 1.05  # measured 1.0662 — fails the same bound


def test_tiny_lambda_falls_back_to_exact_twap_not_nan():
    """Audit E-2: 0 < kappa*tau <~ 1.5e-8 used to produce an all-NaN schedule (acosh underflow ->
    0/0). The TWAP fallback must catch the whole underflow region; schedules stay finite and sum
    to x exactly."""
    for lam in (1e-18, 1e-16, 1e-14, 1e-10):
        res = solve_trajectory(0.05, delta=2.0, n=10, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0,
                               lambda_i=lam)
        assert np.all(np.isfinite(res.schedule)), f"non-finite schedule at lambda={lam}"
        assert res.schedule.sum() == pytest.approx(0.05, abs=1e-15)


def test_hours_valued_delta_rejected_by_lambda_from_budget():
    """Audit E-4: lambda is not unit-invariant; passing delta_hours (e.g. 2.0) instead of the
    week-fraction 2/168 silently mis-sizes lambda ~121x. The guard turns the trap into an error."""
    with pytest.raises(ValueError, match="FRACTION of a week"):
        lambda_from_budget(0.05, 0.02, delta=2.0, beta=0.01, eta_i=2e-4, sigma_i=0.004)


def test_gamma_positive_uses_eta_tilde():
    """Audit E-8: AC 2001 eq. 12-14 use eta_tilde = eta - gamma*tau/2, not raw eta, in kappa and
    the temporary cost. gamma=0 (shipped default) is numerically identical to the old behavior."""
    import math as _math
    eta, sigma, lam = 2e-4, 0.004, 5.0
    r0 = solve_trajectory(0.05, delta=2.0, n=10, sigma_i=sigma, eta_i=eta, gamma_i=0.0,
                          lambda_i=lam)
    assert r0.kappa == pytest.approx(_math.sqrt(lam * sigma**2 / eta), rel=1e-15)
    tau = 2.0 / 10
    gamma = 1e-5
    rg = solve_trajectory(0.05, delta=2.0, n=10, sigma_i=sigma, eta_i=eta, gamma_i=gamma,
                          lambda_i=lam)
    assert rg.kappa == pytest.approx(_math.sqrt(lam * sigma**2 / (eta - 0.5 * gamma * tau)),
                                     rel=1e-15)
    # eta_tilde <= 0 (permanent impact too large for the slice) is rejected, never silently solved.
    with pytest.raises(ValueError, match="eta_tilde"):
        solve_trajectory(0.05, delta=2.0, n=10, sigma_i=sigma, eta_i=1e-6, gamma_i=1e-5,
                         lambda_i=lam)
