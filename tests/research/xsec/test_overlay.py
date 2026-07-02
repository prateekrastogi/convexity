"""Beta-hedge + risk-overlay layer (run_book detail, trailing beta, vol-target, drawdown-stop)."""

from __future__ import annotations

import numpy as np

from convexity.portfolio.engine import XSecConfig, run, run_book
from convexity.regime.overlay import (
    apply_scale,
    beta_hedge_returns,
    drawdown_stop_scale,
    trailing_beta,
    vol_target_scale,
)

from tests.research.xsec.conftest import make_panel


def _btc_panel():
    """8-symbol, 200-day panel; symbol 0 = BTCUSDT market proxy, others beta-correlated + noise."""
    rng = np.random.default_rng(3)
    n_days, n_sym = 200, 8
    syms = ("BTCUSDT", "ETHUSDT", "S2", "S3", "S4", "S5", "S6", "S7")
    mkt_ret = 0.001 + 0.02 * rng.standard_normal(n_days)
    close = np.zeros((n_days, n_sym))
    funding = np.full((n_days, n_sym), 0.0001)
    listed = np.ones((n_days, n_sym), dtype=bool)
    betas = np.array([1.0, 1.2, 0.8, 1.5, 0.5, 1.1, 0.9, 0.7])
    for j in range(n_sym):
        r = betas[j] * mkt_ret + 0.01 * rng.standard_normal(n_days)
        close[:, j] = 100.0 * np.cumprod(1.0 + r)
    return make_panel(close, funding, listed, syms)


def test_run_book_aggregate_matches_run(simple_panel):
    cfg = XSecConfig(signal="momentum", lookback=14, skip=0, quantile="tercile", weighting="equal")
    res_a = run(simple_panel, cfg)
    res_b, book = run_book(simple_panel, cfg)
    assert np.allclose(res_a.net, res_b.net)
    assert np.allclose(res_a.funding, res_b.funding)
    assert len(book) == len(res_a)
    # per-name contribution row-sums equal the gross_price + funding of that week.
    wk = book.contrib.sum(axis=1)
    assert np.allclose(wk, res_a.gross_price + res_a.funding, atol=1e-9)


def test_trailing_beta_recovers_known_beta():
    panel = _btc_panel()
    from convexity.regime.overlay import _daily_returns

    daily = _daily_returns(panel)
    mkt = daily[:, 0]
    beta = trailing_beta(daily, mkt, d0=180, window=120)
    # market vs itself -> beta 1; a 1.5-beta name should estimate near 1.5.
    assert abs(beta[0] - 1.0) < 1e-6
    assert 1.2 < beta[3] < 1.8


def test_beta_hedge_reduces_market_exposure():
    panel = _btc_panel()
    cfg = XSecConfig(signal="carry", carry_lookback=7, quantile="tercile", weighting="equal")
    res, book = run_book(panel, cfg)
    bh = beta_hedge_returns(panel, book, res.net, window=60)
    # hedged series correlation with the market weekly return is closer to 0 than the raw series.
    mkt = book.mkt_ret
    if np.std(mkt) > 0 and np.std(res.net) > 0:
        raw_corr = abs(np.corrcoef(res.net, mkt)[0, 1])
        hed_corr = abs(np.corrcoef(bh.hedged_net, mkt)[0, 1])
        assert hed_corr <= raw_corr + 1e-9
    assert bh.hedged_net.shape == res.net.shape


def test_vol_target_scale_delevers_high_vol():
    rng = np.random.default_rng(0)
    net = 0.01 * rng.standard_normal(60)
    net[30:] *= 4.0  # vol spike in the back half
    scale = vol_target_scale(net, target_vol_ann=0.15, window=8, max_leverage=3.0)
    assert scale.shape == net.shape
    assert np.all(scale >= 0) and np.all(scale <= 3.0)
    # after the spike, average scale should be lower than before it (de-levering into vol).
    assert np.mean(scale[40:]) < np.mean(scale[10:29]) + 1e-9


def test_drawdown_stop_cuts_after_trigger():
    # a series that drops sharply then recovers: stop should flatten during the drawdown.
    net = np.concatenate([np.full(10, 0.01), np.full(5, -0.10), np.full(20, 0.01)])
    scale = drawdown_stop_scale(net, dd_trigger=0.15, dd_reset=0.05, floor=0.0)
    assert scale.shape == net.shape
    # once the cumulative drop breaches 15%, some subsequent week must be cut to 0.
    assert np.any(scale == 0.0)
    # applying the stop yields a shallower max drawdown than the raw series.
    def _maxdd(r):
        eq = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(eq)
        return float((-(eq - peak) / peak).max())
    assert _maxdd(apply_scale(net, scale)) < _maxdd(net) + 1e-9
