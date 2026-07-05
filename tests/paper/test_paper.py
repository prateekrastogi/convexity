"""Paper rebalancer (PAPER.md): parity pin, top-up merge, venue formatting, collateral waterfall."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.paper.capital import sizing_nav
from convexity.paper.policy import (
    BAND,
    ensemble_target_weights,
    fmt_step,
    order_intents,
    orphan_close_intents,
    build_plan,
)
from convexity.paper.topup import extend_panel
from convexity.panel.panel import MS_PER_DAY
from convexity.portfolio.engine import XSecConfig, run_book
from tests.conftest import make_panel


def _panel(n_days: int = 120, syms=("BTCUSDT", "XRPUSDT", "DOGEUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT")):
    rng = np.random.default_rng(9)
    n_sym = len(syms)
    drifts = np.linspace(-0.004, 0.005, n_sym)
    t = np.arange(n_days)[:, None]
    close = 100.0 * np.exp(drifts[None, :] * t
                           + rng.normal(0, 0.003, (n_days, n_sym)).cumsum(axis=0))
    funding = np.full((n_days, n_sym), 1e-4)
    listed = np.ones((n_days, n_sym), dtype=bool)
    return make_panel(close, funding, listed, tuple(syms))


def test_paper_parity_targets_equal_engine_band_ensemble_rows():
    """Invariant PAPER-PARITY (PAPER.md §3): the paper book's weekly target is BIT-IDENTICAL to
    the engine's own band-ensemble construction — any drift is a test failure, never a silent
    divergence between what was validated and what trades."""
    panel = _panel()
    cfgs = [XSecConfig(signal="momentum", lookback=lb, quantile="quintile",
                       weighting="volscaled") for lb in BAND]
    books = [run_book(panel, c)[1] for c in cfgs]
    # a rebalance row present in every member (post-warmup common grid)
    common = set(books[0].d_index.tolist())
    for b in books[1:]:
        common &= set(b.d_index.tolist())
    d = sorted(common)[-1]
    rows = [b.weights[list(b.d_index).index(d)] for b in books]
    engine_row = np.mean(np.stack(rows, axis=0), axis=0)
    paper_row = ensemble_target_weights(panel, int(d))
    assert np.array_equal(paper_row, engine_row)


def test_extend_panel_appends_only_new_days_with_lake_conventions():
    panel = _panel(n_days=30)
    last_ms = int(panel.dates[-1].value // 1_000_000)
    tails = {
        # one already-in-lake day (ignored) + two new days; second symbol missing entirely
        "BTCUSDT": (np.array([last_ms, last_ms + MS_PER_DAY, last_ms + 2 * MS_PER_DAY]),
                    np.array([1.0, 101.0, 102.0]), np.array([0.0, 1e-4, 2e-4])),
        "ETHUSDT": (np.array([last_ms + MS_PER_DAY]), np.array([55.0]), np.array([0.0])),
    }
    out = extend_panel(panel, tails)
    assert out.n_days == panel.n_days + 2
    j_btc = out.symbols.index("BTCUSDT")
    j_eth = out.symbols.index("ETHUSDT")
    j_xrp = out.symbols.index("XRPUSDT")
    # lake row wins on overlap: the last lake close is untouched
    assert out.close[panel.n_days - 1, j_btc] == panel.close[-1, j_btc]
    assert out.close[-2, j_btc] == 101.0 and out.close[-1, j_btc] == 102.0
    # ETH listed on its one new day only; XRP unlisted (NaN) on both new days
    assert out.listed[-2, j_eth] and not out.listed[-1, j_eth]
    assert not out.listed[-2, j_xrp] and not out.listed[-1, j_xrp]
    # funding: 0-filled where listed, NaN off-span (the panel convention)
    assert out.funding[-2, j_btc] == pytest.approx(1e-4)
    assert np.isnan(out.funding[-1, j_xrp])
    assert out.last_day["BTCUSDT"] == out.n_days - 1
    # no-op when tails carry nothing new
    assert extend_panel(panel, {"BTCUSDT": (np.array([last_ms]), np.array([1.0]),
                                            np.array([0.0]))}) is panel


def test_fmt_step_produces_venue_exact_decimals():
    assert fmt_step(63512.600000000006, 0.1) == "63512.6"
    assert fmt_step(0.30000000000000004, 0.001) == "0.300"
    assert fmt_step(438.0, 1.0) == "438"
    assert fmt_step(0.21799000000000002, 1e-05) == "0.21799"


def test_order_intents_are_lot_and_tick_legal_with_exact_strings():
    panel = _panel()
    d = panel.n_days - 1
    w = ensemble_target_weights(panel, d) * 1.0
    vols = np.full(panel.n_symbols, 0.03)
    plan = build_plan(w, np.zeros(panel.n_symbols), vols)
    intents = order_intents(plan, panel.symbols, panel.close[d], nav=5_000.0)
    assert intents, "expected at least one intent from a fresh book"
    from convexity.panel.candidates import crypto_filters

    for it in intents:
        f = crypto_filters(it.symbol)
        n_lots = abs(it.qty) / f.lot_size
        assert abs(n_lots - round(n_lots)) < 1e-6  # lot-grid legal
        assert abs(it.qty) * it.ref_price >= f.min_notional - 1e-9  # dollar floor
        assert float(it.qty_str) == pytest.approx(abs(it.qty))
        assert float(it.price_str) == pytest.approx(it.ref_price)
        assert "e" not in it.qty_str and "e" not in it.price_str  # no sci-notation artifacts
        assert it.tactic == "POST_ONLY_MAKER"  # v1 maker-first; finalize crosses remainders


def test_collateral_waterfall_usdt_first_usdc_fallback_btc_never():
    """USER MANDATE (2026-07-03, PAPER.md §3.1): USDT funds the book alone while it can; USDC
    folds in only when USDT cannot cover the margin headroom; BTC NEVER counts."""
    assets = {"USDT": 5_000.0, "USDC": 5_000.0, "BTC": 0.5}
    d = sizing_nav(assets, est_initial_margin=500.0)
    assert d.nav == 5_000.0 and d.used_assets == ("USDT",)
    assert d.excluded == {"BTC": 0.5}
    # USDT below the 2x headroom vs required margin -> USDC folds in, BTC still excluded
    d2 = sizing_nav({"USDT": 400.0, "USDC": 5_000.0, "BTC": 0.5}, est_initial_margin=500.0)
    assert d2.nav == pytest.approx(5_400.0) and d2.used_assets == ("USDT", "USDC")
    assert d2.excluded == {"BTC": 0.5}
    with pytest.raises(ValueError, match="BTC is excluded"):
        sizing_nav({"BTC": 1.0}, est_initial_margin=0.0)


def test_order_intents_flag_full_closes_reduce_only():
    """PAPER.md §4.1: a FULL CLOSE (target 0, carried nonzero) is reduce_only; opens are not."""
    panel = _panel()
    d = panel.n_days - 1
    w_t = np.array([0.3, -0.3, 0.2, -0.2, 0.0, 0.0])
    w_c = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0])
    vols = np.full(panel.n_symbols, 0.03)
    plan = build_plan(w_t, w_c, vols)
    intents = order_intents(plan, panel.symbols, panel.close[d], nav=5_000.0, w_target=w_t)
    flags = {i.symbol: (i.reduce_only, i.qty) for i in intents}
    sym_close = panel.symbols[4]
    assert sym_close in flags, "the full close must produce an intent"
    ro, q = flags[sym_close]
    assert ro is True and q < 0  # closing a long => sell, reduce-only
    assert all(not v[0] for s, v in flags.items() if s != sym_close)


def test_orphan_close_intents_reduce_only_market_and_dust():
    """PAPER.md §4.1: orphans closed reduce-only MARKET; targeted/covered/dust names untouched."""
    import pytest as _pt

    positions = {"AAAUSDT": 5.0, "BBBUSDT": -3.2, "DUSTUSDT": 0.0004,
                 "HELDUSDT": 2.0, "TGTUSDT": 1.0}
    covered = {"HELDUSDT"}                      # has an intent this cycle
    targets = {"TGTUSDT": 0.01}                 # targeted, inside the band -> NOT an orphan
    filters = {s: (0.01, 0.001, 0.001, 5.0) for s in positions}
    prices = {s: 10.0 for s in positions}
    out = orphan_close_intents(positions, covered, targets, filters, prices)
    by = {i.symbol: i for i in out}
    assert set(by) == {"AAAUSDT", "BBBUSDT"}    # dust skipped, held/targeted untouched
    assert by["AAAUSDT"].qty == _pt.approx(-5.0) and by["AAAUSDT"].reduce_only
    assert by["AAAUSDT"].tactic == "MARKET"
    assert by["BBBUSDT"].qty == _pt.approx(3.2) and by["BBBUSDT"].reduce_only
    # a $4 notional orphan (0.35 * 10 < min_notional 5) is STILL closed: reduceOnly exempts it
    out2 = orphan_close_intents({"CCCUSDT": 0.35}, set(), {}, {"CCCUSDT": (0.01, 0.01, 0.01, 5.0)},
                                {"CCCUSDT": 10.0})
    assert len(out2) == 1 and out2[0].qty == _pt.approx(-0.35) and out2[0].reduce_only
