"""Panel acceptance gate (REGIME §13 Panel, Invariant CAUSAL)."""

import numpy as np

from convexity.regime.panel.build import build_panel, log_returns


def _series(rng, n_stamps, n_sym):
    return {f"S{i}": rng.normal(0.0, 0.01, size=n_stamps) for i in range(n_sym)}


def test_panel_is_causal_no_future_bar():
    rng = np.random.default_rng(0)
    rets = _series(rng, 300, 10)
    ac = {s: "crypto_perp" for s in rets}
    p1 = build_panel(returns_by_symbol=rets, asset_class_by_symbol=ac, window=100, asof_bar=200)
    # Mutate a FUTURE bar (index 250 > asof_bar=200); the panel at t must be unchanged (CAUSAL).
    rets2 = {k: v.copy() for k, v in rets.items()}
    for k in rets2:
        rets2[k][250] += 5.0
    p2 = build_panel(returns_by_symbol=rets2, asset_class_by_symbol=ac, window=100, asof_bar=200)
    assert np.allclose(p1.returns, p2.returns)


def test_panel_is_causally_demeaned():
    rng = np.random.default_rng(1)
    rets = _series(rng, 200, 8)
    ac = {s: "crypto_perp" for s in rets}
    p = build_panel(returns_by_symbol=rets, asset_class_by_symbol=ac, window=120, asof_bar=150)
    assert np.allclose(p.returns.mean(axis=0), 0.0, atol=1e-12)  # trailing causal de-mean
    assert np.isfinite(p.returns).all()
    assert p.n_t == 8


def test_staleness_drop_shrinks_n_t():
    rng = np.random.default_rng(2)
    rets = _series(rng, 200, 6)
    ac = {s: "us_equity" for s in rets}  # staleness_stamps=1 for equities
    # S0 is closed for the last 3 stamps -> exceeds the 1-stamp window -> dropped.
    is_open = {s: np.ones(200, dtype=bool) for s in rets}
    is_open["S0"][-3:] = False
    p = build_panel(
        returns_by_symbol=rets, asset_class_by_symbol=ac, window=100, asof_bar=199,
        stamp_is_open=is_open,
    )
    assert "S0" not in p.instruments
    assert p.n_t == 5


def test_log_returns_shape():
    prices = np.array([100.0, 101.0, 102.0, 100.0])
    r = log_returns(prices)
    assert r.shape == (3,)
    assert np.isclose(r[0], np.log(101.0 / 100.0))
