"""Engine + panel: point-in-time universe growth, Friday cadence, weekly-series shape, no-lookahead."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.portfolio.engine import XSecConfig, run
from convexity.panel.panel import universe_size_by_year


def test_universe_size_grows_with_listings(simple_panel):
    # D lists on day 40; before that the live count is 3, after it is 4.
    live = simple_panel.listed.sum(axis=1)
    assert live[10] == 3
    assert live[50] == 4
    by_year = universe_size_by_year(simple_panel)
    assert max(by_year.values()) == 4


def test_engine_produces_weekly_series(simple_panel):
    cfg = XSecConfig(signal="momentum", lookback=28, skip=0, quantile="tercile")
    res = run(simple_panel, cfg)
    assert len(res) > 0
    # aligned arrays
    assert res.net.shape == res.turnover.shape == res.funding.shape == res.dates.shape
    # net = price + funding - cost, exactly.
    assert np.allclose(res.net, res.gross_price + res.funding - res.cost)


def test_engine_rebalances_on_fridays(simple_panel):
    cfg = XSecConfig(signal="carry", carry_lookback=7, quantile="tercile")
    res = run(simple_panel, cfg)
    # every rebalance date is a Friday (weekday 4).
    assert all(d.weekday() == 4 for d in res.dates)
    # weekly cadence: consecutive dates 7 days apart.
    if len(res) > 1:
        gaps = np.diff(res.dates.values).astype("timedelta64[D]").astype(int)
        assert set(gaps.tolist()) == {7}


def test_carry_book_earns_funding_when_signal_is_carry(simple_panel):
    # carry shorts the high-funding name (A) and longs the negative-funding name (C):
    # both legs COLLECT funding -> net funding income should be positive on average.
    cfg = XSecConfig(signal="carry", carry_lookback=7, quantile="tercile")
    res = run(simple_panel, cfg)
    assert res.funding.mean() > 0


def test_universe_size_reported_per_rebalance_is_pointintime(simple_panel):
    # a rebalance at a date before day 40 must see universe size < 4.
    cfg = XSecConfig(signal="momentum", lookback=7, skip=0, quantile="tercile")
    res = run(simple_panel, cfg)
    # early rebalances (before D has 7d history past day 40) see 3 eligible names.
    assert res.universe_size.min() <= 3
    assert res.universe_size.max() <= 4


def test_carried_weights_drift_exactly_once_across_consecutive_settles():
    """Audit PF-1 (CRITICAL, was live in every recorded backtest): the engine loops used to feed
    settle_week's returned w_end (ALREADY drifted) back in as w_prev, which settle_week drifts
    AGAIN over the same elapsed week -- carried weights of w_prev*(1+r)^2 doubled turnover's
    drift component and overstated cost every week. This composition test pins single-drift:
    re-targeting the SAME +-0.5/-0.5 book across a deterministic +10%/-10% week must produce
    turnover = 0.5*(|0.5-0.55|+|-0.5-(-0.45)|)/... = 0.05 exactly, not 0.10."""
    from convexity.cost.accounting import settle_week
    from tests.conftest import make_panel

    n_days = 22
    close = np.zeros((n_days, 2))
    # deterministic weekly moves: sym0 +10%/wk, sym1 -10%/wk, flat within the week
    for d in range(n_days):
        wk = d // 7
        close[d, 0] = 100.0 * (1.10 ** wk)
        close[d, 1] = 100.0 * (0.90 ** wk)
    # jump exactly at week boundaries: make the within-week path constant at the week's level
    funding = np.zeros((n_days, 2))
    listed = np.ones((n_days, 2), dtype=bool)
    panel = make_panel(close, funding, listed, ("UP", "DN"))

    w_tgt = np.array([0.5, -0.5])
    friction = 0.0013

    # week 1: settle [7, 14] with w_prev = the PRIOR TARGET set at d0=7's previous rebalance (d0=0)
    res1, w_end1 = settle_week(panel.close, panel.funding, panel.listed,
                               d0=7, d1=14, w_target=w_tgt, w_prev=w_tgt, friction=friction)
    # single drift: carried = [0.5*1.1, -0.5*0.9] = [0.55, -0.45];
    # turnover = 0.5*(|0.5-0.55| + |-0.5+0.45|) = 0.05
    assert res1.turnover == pytest.approx(0.05, abs=1e-12)

    # ENGINE COMPOSITION (the actual bug site): two consecutive engine-style settles. The second
    # call's w_prev must be the prior TARGET (w_tgt), not w_end1 -- feeding w_end1 double-drifts.
    res2_correct, _ = settle_week(panel.close, panel.funding, panel.listed,
                                  d0=14, d1=21, w_target=w_tgt, w_prev=w_tgt, friction=friction)
    assert res2_correct.turnover == pytest.approx(0.05, abs=1e-12)
    res2_doubled, _ = settle_week(panel.close, panel.funding, panel.listed,
                                  d0=14, d1=21, w_target=w_tgt, w_prev=w_end1, friction=friction)
    # the OLD wiring's number -- shown here as the counterfactual this test exists to forbid:
    assert res2_doubled.turnover == pytest.approx(0.10, abs=1e-3)
    assert res2_doubled.turnover > res2_correct.turnover * 1.9
