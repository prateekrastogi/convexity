"""Cube-root no-trade band (EXECUTION.md §4, §6.4, §11.2): Invariants E-monotone, B-band,
SIGMA-W-DISTINCT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsec import portfolio
from research.xsec.execution import (
    CostFacts,
    ExecTarget,
    ExecutionDefaults,
    ExecutionEngine,
    band_width,
    solve_trajectory,
    weight_drift_vol,
)
from research.xsec.panel import Panel


def _facts(n: int, theta: np.ndarray) -> CostFacts:
    return CostFacts(
        eta=np.full(n, 2e-4),
        gamma=np.full(n, 1e-5),
        theta=theta,
        sigma=np.full(n, 0.004),
    )


def test_live_turnover_never_exceeds_backtest():
    # Invariant E-monotone (§6.4, P4): over a multi-year synthetic replay, live turnover is <=
    # the backtest's full-rebalance turnover, per-name and in aggregate, every week.
    rng = np.random.default_rng(11)
    n, weeks = 20, 156
    theta = rng.uniform(0.5e-4, 6e-4, n)  # band widths drawn from §4.2's formula at varying theta
    cfg = ExecutionDefaults(rho=2.0, band_c=0.5)
    engine = ExecutionEngine()
    shortable = np.ones(n, dtype=bool)
    w_prev = np.zeros(n)
    for _ in range(weeks):
        w_target = rng.normal(0.0, 0.02, n)
        w_target -= w_target.mean()  # dollar-neutral-ish synthetic book
        w_carried = w_prev * (1.0 + rng.normal(0.0, 0.03, n))  # price drift of the held book
        plan = engine.on_target(ExecTarget(w_target, shortable), w_carried, _facts(n, theta), False, cfg)
        assert plan.turnover_live <= plan.turnover_backtest + 1e-12
        assert np.all(np.abs(plan.trade) <= np.abs(plan.x) + 1e-12)  # per-name, not just aggregate
        # the gap is attributable line-by-line to the band's skipped/clipped names (§6.4):
        # every skipped name contributes |x_i|/2; every banded trade contributes b_i/2.
        skipped = np.array([r == "BAND-SKIP" for r in plan.reason])
        banded = np.array([r == "TRADE" for r in plan.reason])
        gap = plan.turnover_backtest - plan.turnover_live
        explained = 0.5 * (np.sum(np.abs(plan.x[skipped])) + np.sum(plan.band[banded]))
        assert abs(gap - explained) < 1e-12
        w_prev = w_target


def test_band_widens_monotonically_with_impact_inclusion():
    # Invariant B-band (§4.4): adding the convex impact term to theta can only WIDEN the band,
    # so computing b_i from the LINEAR component only is conservative, never anti-conservative.
    rho, c = 2.0, 0.5
    for theta_linear in (0.5e-4, 1.2e-4, 6e-4):
        for impact_term in (0.0, 0.2e-4, 1e-4, 5e-4):
            b_linear = band_width(theta_linear, rho, c=c)
            b_with_impact = band_width(theta_linear + impact_term, rho, c=c)
            assert b_with_impact >= b_linear


def test_band_edge_rule_and_zero_cost():
    # §4.3: trade only to the band EDGE, never back to the exact target.
    n = 4
    theta = np.full(n, 1.3e-4)
    cfg = ExecutionDefaults(rho=1.0, band_c=1.0)
    b = band_width(1.3e-4, 1.0, c=1.0)
    w_carried = np.array([0.05, 0.05, -0.05, -0.05])
    w_target = w_carried + np.array([b * 0.5, b * 3.0, -b * 0.5, -b * 3.0])
    plan = ExecutionEngine().on_target(
        ExecTarget(w_target, np.ones(n, dtype=bool)), w_carried, _facts(n, theta), False, cfg
    )
    assert plan.trade[0] == 0.0 and plan.reason[0] == "BAND-SKIP"
    assert plan.trade[1] == plan.x[1] - b  # stops at the band edge
    assert plan.trade[2] == 0.0
    assert plan.trade[3] == plan.x[3] + b  # negative drift: edge on the other side
    assert band_width(0.0, 1.0, c=1.0) == 0.0  # zero linear cost -> zero band


def _fixture_panel(rng: np.ndarray, n_days: int = 90, n_sym: int = 5) -> Panel:
    vols = np.linspace(0.01, 0.08, n_sym)
    rets = rng.normal(0.0, vols, size=(n_days, n_sym))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    listed = np.ones((n_days, n_sym), dtype=bool)
    symbols = tuple(f"S{i}USDT" for i in range(n_sym))
    return Panel(
        dates=pd.date_range("2023-01-01", periods=n_days, freq="D"),
        symbols=symbols,
        close=close,
        funding=np.zeros((n_days, n_sym)),
        listed=listed,
        first_day={s: 0 for s in symbols},
        last_day={s: n_days - 1 for s in symbols},
    )


def test_weight_drift_vol_differs_from_price_return_vol():
    # Invariant SIGMA-W-DISTINCT (§4.2, §10): sigma_w,i is a different code path AND a different
    # value from portfolio.realized_vol — weight drift, not price return.
    rng = np.random.default_rng(3)
    panel = _fixture_panel(rng)
    d = 60
    sigma_price = portfolio.realized_vol(panel, d)
    assert np.isfinite(sigma_price).all()

    # weekly vol-scaled targets + drifted carries: the vol-scaled weighting's convexity is active,
    # so the price-return -> weight-drift map is not the identity (§4.2).
    weeks = list(range(30, d + 1, 7))
    w_t, w_c = [], []
    prev = None
    for dd in weeks:
        rank = np.argsort(np.argsort(panel.close[dd] / panel.close[dd - 14])) / (panel.n_symbols - 1)
        w = portfolio.quantile_weights(
            rank.astype(float), quantile="quintile", weighting="volscaled",
            vol=portfolio.realized_vol(panel, dd, window=28),
        )
        if prev is not None:
            r_week = panel.close[dd] / panel.close[dd - 7] - 1.0
            w_c.append(prev * (1.0 + r_week))
            w_t.append(prev)
        prev = w
    sigma_w = weight_drift_vol(np.array(w_c), np.array(w_t), half_life_weeks=8.0)
    finite = np.isfinite(sigma_w)
    assert finite.any()

    # SIGMA-W-DISTINCT, defended by VALUE (the auditor flagged the old `is not` object-identity line
    # as decorative — a check between two distinct functions that can never fail; it is removed).
    # The weight-drift statistic differs from the price-return statistic not merely by rescaling: the
    # vol-scaled weighting maps price returns through 1/sigma_i, so a name with LARGE price-return vol
    # (small weight) can have SMALL weight drift, and vice versa — the two are not even monotone in
    # each other. Assert they differ on the names that are genuinely held (nonzero drift), and that no
    # single scalar rescaling turns sigma_price into sigma_w (which a mere unit change would allow).
    held = finite & (sigma_w > 0.0)
    assert held.any()  # at least one genuinely-traded name with real weight drift
    assert not np.allclose(sigma_w[held], sigma_price[held])
    # rule out "same statistic in different units": no constant c makes sigma_w == c * sigma_price.
    ratios = sigma_w[held] / sigma_price[held]
    assert ratios.size >= 2
    assert float(np.std(ratios)) > 1e-9  # ratios vary -> genuinely different statistic, not a rescale


def test_sign_conventions():
    # a wider band never flips the direction of a surviving trade (§4.3).
    b = band_width(2e-4, 1.0, c=1.0)
    x = 3.0 * b
    assert np.sign(x - np.sign(x) * b) == np.sign(x)
    x = -3.0 * b
    assert np.sign(x - np.sign(x) * b) == np.sign(x)


def test_twap_limit_recovery():
    res = solve_trajectory(0.01, delta=2.0, n=5, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=0.0)
    assert res.kappa == 0.0
    assert np.all(res.schedule == 0.01 / 5)


def test_quantile_flip_trades_full_while_equal_indrift_is_banded():
    # §13 bullet 2 + Invariant B-band: a quantile-membership FLIP (a name entering/exiting a leg,
    # w_target or w_carried crossing 0) is NEVER band-gated -- it trades in full -- while an
    # in-leg drift of the SAME magnitude is clipped to the band edge. The band derivation (§4.2)
    # assumes small deviations around a continuous optimum; a 0->full-weight move is not one.
    n = 3
    theta = np.full(n, 1.3e-4)
    cfg = ExecutionDefaults(rho=1.0, band_c=1.0)
    b = band_width(1.3e-4, 1.0, c=1.0)
    small = 0.5 * b  # a drift strictly INSIDE the band
    # name 0: pure in-leg drift of `small` -> banded (skipped). name 1: flip from flat (0 -> +small)
    # -> the SAME magnitude move, but a membership flip, so it trades in FULL. name 2: sign flip
    # (short -> long) of the same magnitude -> also full.
    w_carried = np.array([0.05, 0.0, -small / 2.0])
    w_target = np.array([0.05 + small, small, small / 2.0])
    plan = ExecutionEngine().on_target(
        ExecTarget(w_target, np.ones(n, dtype=bool)), w_carried, _facts(n, theta), False, cfg
    )
    assert plan.reason[0] == "BAND-SKIP" and plan.trade[0] == 0.0  # in-leg drift < band: skipped
    assert plan.reason[1] == "FLIP-FULL" and plan.trade[1] == plan.x[1]  # flat->long flip: full
    assert plan.reason[2] == "FLIP-FULL" and plan.trade[2] == plan.x[2]  # short->long flip: full
    # the flip names are NOT band-clipped even though |x_i| = small < b (the whole point of §13).
    assert abs(plan.x[1]) < b and abs(plan.x[2]) < b
