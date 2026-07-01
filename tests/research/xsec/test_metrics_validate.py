"""Weekly metrics + discipline layer (purged WF + DSR reuse) sanity."""

from __future__ import annotations

import numpy as np

from research.xsec import metrics
from research.xsec.validate import deflated_sharpe, purged_walkforward, weekly_defaults


def test_weekly_metrics_sharpe_sign_and_attribution():
    rng = np.random.default_rng(0)
    net = 0.002 + 0.01 * rng.standard_normal(200)  # positive drift
    turnover = np.full(200, 0.3)
    cost = turnover * 0.0013
    funding = np.full(200, 0.0004)
    m = metrics.compute(net, turnover=turnover, cost=cost, funding=funding)
    assert m.n_weeks == 200
    assert m.sharpe_ann > 0
    assert m.turnover_per_year > 0
    # cost drag = mean cost * periods/yr; funding income likewise.
    assert abs(m.cost_drag_per_year - 0.3 * 0.0013 * metrics.PERIODS_PER_YEAR) < 1e-9
    assert abs(m.funding_income_per_year - 0.0004 * metrics.PERIODS_PER_YEAR) < 1e-9


def test_purged_walkforward_oos_only_uses_test_slices():
    net = np.linspace(0.0, 0.01, 100)
    wf = purged_walkforward(net, n_folds=3, embargo_weeks=1)
    assert wf.n_folds == 3
    assert wf.embargo_weeks == 1
    assert 0 < wf.oos_n_weeks < 100  # OOS is only the test tiles, not the whole series
    assert len(wf.fold_sharpes) == 3


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(1)
    net = 0.003 + 0.01 * rng.standard_normal(150)
    sr_ann = metrics.compute(net).sharpe_ann
    # few trials, all mediocre -> the headline is less deflated than under many lucky trials.
    few = deflated_sharpe(net, np.array([sr_ann, 0.1, 0.2]), n_trials_effective=3, k_distinct=3)
    many = deflated_sharpe(
        net, np.concatenate([[sr_ann], rng.normal(0.5, 0.5, 30)]), n_trials_effective=30, k_distinct=30
    )
    assert many.sr_star0 >= few.sr_star0  # more trials -> higher luck benchmark
    assert many.dsr <= few.dsr


def test_weekly_defaults_uses_weekly_ppy():
    d = weekly_defaults()
    assert 50.0 < d.ppy < 53.0  # ~52.18, not the 1-min 36
