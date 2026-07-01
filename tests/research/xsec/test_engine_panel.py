"""Engine + panel: point-in-time universe growth, Friday cadence, weekly-series shape, no-lookahead."""

from __future__ import annotations

import numpy as np

from research.xsec.engine import XSecConfig, run
from research.xsec.panel import universe_size_by_year


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
