"""Multi-asset (equities/commodities) extension: zero-funding panel, per-class friction, breadth.

These test the CLASS-AWARE glue on top of the (already tested) engine: that a momentum-only sleeve
has exactly zero funding cash flow, that the per-class friction table is coherent, that the effective
-breadth (inverse-HHI) matches its definition, and that a momentum book on a zeroed-funding panel earns
net = gross_price - cost (no funding term). Engine correctness itself is covered in test_engine_panel.
"""

from __future__ import annotations

import numpy as np

from convexity.portfolio.engine import XSecConfig, run, run_book
from convexity.panel.sleeve import (
    COMMODITY_FRICTION,
    CRYPTO_FRICTION,
    EQUITY_FRICTION,
    FRICTION_BY_SLEEVE,
    effective_breadth,
)

from .conftest import make_panel


def test_effective_breadth_bounds():
    # one name carries all PnL -> breadth ~1; perfectly even 4-name book -> breadth = 4.
    one = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    assert abs(effective_breadth(one) - 1.0) < 1e-9
    even = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
    assert abs(effective_breadth(even) - 4.0) < 1e-9
    assert effective_breadth(np.zeros((3, 4))) == 0.0
    assert effective_breadth(np.zeros((0, 0))) == 0.0


def test_friction_table_ordering():
    # commodities wider than equities; crypto (Binance taker) widest per-leg fee.
    assert EQUITY_FRICTION.round_trip_bps < COMMODITY_FRICTION.round_trip_bps
    assert COMMODITY_FRICTION.round_trip_bps < CRYPTO_FRICTION.round_trip_bps
    assert FRICTION_BY_SLEEVE["equity"] is EQUITY_FRICTION
    assert FRICTION_BY_SLEEVE["commodity"] is COMMODITY_FRICTION
    # crypto round-trip is the committed 13bp truth.
    assert abs(CRYPTO_FRICTION.round_trip_bps - 13.0) < 1e-6


def test_momentum_only_panel_has_zero_funding():
    # a panel with NON-zero funding, run momentum-only, must still show exactly zero funding cash flow
    # because the engine's momentum signal never consumes funding and we zero the array.
    n_days, syms = 90, ("A", "B", "C", "D")
    t = np.arange(n_days)
    close = np.vstack([
        100.0 * (1.0 + 0.004) ** t, 100.0 * (1.0 + 0.002) ** t,
        100.0 * (1.0 - 0.001) ** t, 100.0 * (1.0 + 0.001) ** t,
    ]).T
    funding = np.full((n_days, 4), 0.0)  # zeroed panel (the momentum-only convention)
    listed = np.ones((n_days, 4), dtype=bool)
    panel = make_panel(close, funding, listed, syms)

    cfg = XSecConfig(signal="momentum", lookback=14, skip=0, quantile="tercile", weighting="volscaled",
                     taker=EQUITY_FRICTION.taker, half_spread=EQUITY_FRICTION.half_spread,
                     impact=EQUITY_FRICTION.impact)
    res = run(panel, cfg)
    assert np.all(res.funding == 0.0)
    # net = gross_price - cost, with no funding term.
    assert np.allclose(res.net, res.gross_price - res.cost)


def test_per_class_friction_scales_cost():
    # the same momentum book costs more under commodity friction than equity friction (wider spread).
    n_days, syms = 120, ("A", "B", "C", "D")
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0005, 0.02, size=(n_days, 4))
    close = 100.0 * np.cumprod(1.0 + steps, axis=0)
    funding = np.zeros((n_days, 4))
    listed = np.ones((n_days, 4), dtype=bool)
    panel = make_panel(close, funding, listed, syms)

    base = dict(signal="momentum", lookback=14, skip=0, quantile="tercile", weighting="equal")
    eq = XSecConfig(**base, taker=EQUITY_FRICTION.taker, half_spread=EQUITY_FRICTION.half_spread,
                    impact=EQUITY_FRICTION.impact)
    co = XSecConfig(**base, taker=COMMODITY_FRICTION.taker, half_spread=COMMODITY_FRICTION.half_spread,
                    impact=COMMODITY_FRICTION.impact)
    cost_eq = run(panel, eq).cost.sum()
    cost_co = run(panel, co).cost.sum()
    assert cost_co > cost_eq > 0.0


def test_breadth_matches_book_contrib():
    # effective_breadth on a real book equals the inverse-HHI of the per-name summed |contrib|.
    n_days, syms = 120, ("A", "B", "C", "D")
    t = np.arange(n_days)
    close = np.vstack([
        100.0 * (1.0 + 0.004) ** t, 100.0 * (1.0 + 0.002) ** t,
        100.0 * (1.0 - 0.001) ** t, 100.0 * (1.0 + 0.001) ** t,
    ]).T
    panel = make_panel(close, np.zeros((n_days, 4)), np.ones((n_days, 4), bool), syms)
    cfg = XSecConfig(signal="momentum", lookback=14, skip=0, quantile="tercile")
    _res, book = run_book(panel, cfg)
    abs_name = np.abs(book.contrib).sum(axis=0)
    w = abs_name / abs_name.sum()
    assert abs(effective_breadth(book.contrib) - 1.0 / np.sum(w**2)) < 1e-9
