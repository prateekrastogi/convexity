"""Multi-book invariants (PAPER.md §11): disjoint domains, the DOMAIN-SCOPED orphan sweep,
book filters, journal separation, weight re-alignment."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.paper.books import (
    BOOKS,
    COMMODITY,
    CRYPTO,
    assert_disjoint_domains,
    book_domain,
    book_filters,
)
from convexity.paper.policy import orphan_close_intents


def test_book_domains_are_pairwise_disjoint():
    """The invariant every venue-facing scope relies on: no symbol belongs to two books."""
    assert_disjoint_domains()
    assert book_domain(CRYPTO) & book_domain(COMMODITY) == frozenset()


def test_book_domains_cover_their_sleeves():
    dom_c = book_domain(CRYPTO)
    dom_k = book_domain(COMMODITY)
    assert "BTCUSDT" in dom_c and "BTCUSDT" not in dom_k
    for s in ("XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "COPPERUSDT", "CLUSDT", "NATGASUSDT"):
        assert s in dom_k and s not in dom_c


def test_commodity_book_has_real_filters():
    f = book_filters(COMMODITY)
    assert set(f) >= {"XAUUSDT", "CLUSDT", "NATGASUSDT"}
    assert f["CLUSDT"].lot_size == pytest.approx(0.01)
    assert f["NATGASUSDT"].lot_size == pytest.approx(0.1)
    assert all(v.min_notional > 0 for v in f.values())


def test_orphan_sweep_never_touches_the_other_books_positions():
    """R6/PAPER.md §11.1 — the shared account holds BOTH books; the crypto sweep must treat the
    commodity book's fresh positions as untouchable (and vice versa), else every Saturday one
    book would market-close the other's week."""
    positions = {
        "XAUUSDT": 0.5, "CLUSDT": -3.0,     # the commodity book's open pairs
        "DOGEUSDT": 1000.0,                  # a genuine crypto orphan (rotated out)
    }
    filters = {s: (0.01, 0.001, 0.001, 5.0) for s in positions}
    prices = {s: 100.0 for s in positions}

    crypto_sweep = orphan_close_intents(positions, covered=set(), targets_by_symbol={},
                                        live_filters=filters, prices_by_symbol=prices,
                                        domain=book_domain(CRYPTO))
    assert {i.symbol for i in crypto_sweep} == {"DOGEUSDT"}

    commodity_sweep = orphan_close_intents(positions, covered=set(), targets_by_symbol={},
                                           live_filters=filters, prices_by_symbol=prices,
                                           domain=book_domain(COMMODITY))
    assert {i.symbol for i in commodity_sweep} == {"XAUUSDT", "CLUSDT"}
    assert all(i.reduce_only and i.tactic == "MARKET" for i in commodity_sweep)


def test_domain_is_required_keyword():
    with pytest.raises(TypeError):
        orphan_close_intents({}, set(), {}, {}, {})  # missing domain must never compile silently


def test_books_registry_shape():
    assert set(BOOKS) == {"crypto", "commodity"}
    assert BOOKS["crypto"].journal_path == "data/paper/journal.jsonl"  # legacy path unchanged
    assert BOOKS["commodity"].journal_path != BOOKS["crypto"].journal_path


def test_order_intents_with_commodity_filters_are_lot_legal():
    """The book's filters map drives quantization (PAPER.md §11.1): NATGAS lot 0.1, CL lot 0.01;
    a symbol missing from the map (a discovered-but-unpromoted perp) is never traded."""
    from convexity.paper.policy import build_plan, order_intents

    syms = ("XAUUSDT", "CLUSDT", "NATGASUSDT", "NEWCOMMUSDT")
    w_t = np.array([0.3, -0.3, 0.2, 0.2])
    w_c = np.zeros(4)
    plan = build_plan(w_t, w_c, np.full(4, 0.02))
    prices = np.array([3300.0, 65.0, 3.5, 10.0])
    filters = book_filters(COMMODITY)
    intents = order_intents(plan, syms, prices, nav=2_000.0, w_target=w_t, filters=filters)
    by = {i.symbol: i for i in intents}
    assert "NEWCOMMUSDT" not in by            # outside the quantization surface
    assert set(by) <= {"XAUUSDT", "CLUSDT", "NATGASUSDT"}
    for sym, it in by.items():
        lot = filters[sym].lot_size
        n_lots = abs(it.qty) / lot
        assert abs(n_lots - round(n_lots)) < 1e-6
        assert abs(it.qty) * it.ref_price >= filters[sym].min_notional - 1e-9
        assert "e" not in it.qty_str and "e" not in it.price_str


def _bday_panel(n_days: int = 30, syms=("XAUUSDT", "XAGUSDT")):
    """A 24x5 panel (business days ONLY — the commodity proxy-era shape, no weekend rows)."""
    import pandas as pd

    from convexity.panel.panel import Panel

    dates = pd.bdate_range("2026-01-05", periods=n_days)  # Mon 2026-01-05 start
    n = len(syms)
    close = np.full((n_days, n), 100.0)
    listed = np.ones((n_days, n), dtype=bool)
    funding = np.zeros((n_days, n))
    first = {s: 0 for s in syms}
    last = {s: n_days - 1 for s in syms}
    return Panel(dates, tuple(syms), close, funding, listed, first, last)


def test_settle_window_is_calendar_days_not_rows(tmp_path):
    """Review finding 2026-07-05 (CRITICAL): the commodity panel has no weekend rows, so a
    '+7 rows' settle window overshoots the week — the window must be CALENDAR-day based.
    On this business-day panel, Fri 2026-01-09 -> Fri 2026-01-16 is 5 rows, and the settle
    must still land exactly on 2026-01-16."""
    from convexity.paper import cli
    from convexity.paper.journal import Journal

    panel = _bday_panel()
    j = Journal(tmp_path / "j.jsonl")
    j.append("targets", "2026-01-09", weights=[0.5, -0.5], symbols=list(panel.symbols),
             g_t=1.0, nav=100.0, mode="dry", book="commodity")
    d_now = int(np.flatnonzero(panel.dates == np.datetime64("2026-01-16"))[0])
    cli._settle_prior_week(j, COMMODITY, panel, d_now, live=False)
    settle = j.last("settle")
    assert settle is not None, "settle must fire on a weekend-less panel"
    assert settle["asof"] == "2026-01-16"


def test_settle_charges_real_rebalance_turnover(tmp_path):
    """Review finding 2026-07-05 (MAJOR): with flat prices and zero funding, the settled model
    return must equal MINUS the turnover cost of moving from the week-before book to the settled
    book — not the ~zero drift-only cost of w_prev == w_target."""
    from convexity.paper import cli
    from convexity.paper.journal import Journal

    panel = _bday_panel()
    j = Journal(tmp_path / "j.jsonl")
    # week 1 book +0.5/-0.5; week 2 book flipped -0.5/+0.5 => sum|dw| = 2.0, turnover 1.0.
    # (settle window 01-16 -> 01-23 so settle_week's one-week drift lookback stays inside the
    # panel; a lookback before row 0 zeroes the carried book by design.)
    j.append("targets", "2026-01-09", weights=[0.5, -0.5], symbols=list(panel.symbols),
             g_t=1.0, nav=100.0, mode="dry", book="commodity")
    j.append("targets", "2026-01-16", weights=[-0.5, 0.5], symbols=list(panel.symbols),
             g_t=1.0, nav=100.0, mode="dry", book="commodity")
    d_now = int(np.flatnonzero(panel.dates == np.datetime64("2026-01-23"))[0])
    cli._settle_prior_week(j, COMMODITY, panel, d_now, live=False)
    settle = j.last("settle")
    assert settle is not None and settle["asof"] == "2026-01-23"
    expected_cost = cli.SETTLE_FRICTION["commodity"] * 1.0  # turnover 1.0 on flat prices
    assert settle["r_model"] == pytest.approx(-expected_cost, rel=1e-9)


def test_aligned_weights_realigns_by_symbol_and_zeroes_departed():
    """PAPER.md §11.2: a book's panel can GROW between weeks (discovery overlay) — the settle
    must re-align journaled weights by symbol, never by position."""
    from convexity.paper.cli import _aligned_weights
    from tests.conftest import make_panel

    close = np.full((10, 3), 100.0)
    panel = make_panel(close, np.zeros_like(close), np.ones((10, 3), bool),
                       ("XAUUSDT", "NEWUSDT", "CLUSDT"))
    rec = {"weights": [0.5, -0.5, 0.25], "symbols": ["XAUUSDT", "CLUSDT", "GONEUSDT"]}
    w = _aligned_weights(rec, panel)
    assert w.tolist() == [0.5, 0.0, -0.5]  # NEW gets 0, GONE dropped, order = panel order

    legacy = {"weights": [0.1, 0.2, 0.3]}  # no symbols -> positional (frozen crypto behavior)
    assert _aligned_weights(legacy, panel).tolist() == [0.1, 0.2, 0.3]
    with pytest.raises(SystemExit, match="cannot align"):
        _aligned_weights({"weights": [0.1, 0.2]}, panel)
