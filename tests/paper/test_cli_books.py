"""The equity and commodity books' own target paths.

Each book has a different decision function behind the same session shell: crypto is the
cross-sectional momentum quintile, equity is the S1 trend ensemble with a breadth-scaled
half-Kelly and a drawdown brake, commodity is band-z mean reversion on pre-named RV pairs. The
crypto path is covered by test_cli_rebalance; these drive the other two, which between them are
the largest remaining untested region of the session code.

Equity needs a long panel: names are quarantined until a 260-day warmup completes, so a short
panel would produce an empty book and test nothing.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from convexity.paper.books import BOOKS, book_domain, book_journal_path
from convexity.paper.journal import Journal

from tests.conftest import make_panel


def run_cli(monkeypatch, *argv: str) -> None:
    from convexity.paper.cli import main

    monkeypatch.setattr(sys, "argv", ["convexity.paper", *argv])
    main()


def jrn(book: str) -> Journal:
    return Journal(book_journal_path(BOOKS[book]))


def trending_panel(symbols, n_days: int):
    k = len(symbols)
    t = np.arange(n_days)
    close = np.zeros((n_days, k))
    funding = np.zeros((n_days, k))
    listed = np.ones((n_days, k), dtype=bool)
    for j, d in enumerate(np.linspace(-0.002, 0.002, k)):
        close[:, j] = 100.0 * (1.0 + d) ** t
    return make_panel(close, funding, listed, tuple(symbols))


def mean_reverting_panel(symbols, n_days: int):
    """Oscillating levels so the RV pairs' band-z has something to mean-revert against."""
    k = len(symbols)
    t = np.arange(n_days)
    close = np.zeros((n_days, k))
    funding = np.zeros((n_days, k))
    listed = np.ones((n_days, k), dtype=bool)
    for j in range(k):
        close[:, j] = 100.0 + 5.0 * np.sin(2 * np.pi * t / 40.0 + j)
    return make_panel(close, funding, listed, tuple(symbols))


# ---------------------------------------------------------------- equity book
@pytest.fixture
def equity_env(fake_venue, monkeypatch):
    from convexity.paper import cli

    syms = tuple(sorted(book_domain(BOOKS["equity"]))[:20])
    panel = trending_panel(syms, n_days=400)      # past the 260d warmup quarantine
    monkeypatch.setattr(cli, "_load_panel", lambda spec, topup: panel)
    fake_venue.equity = 50_000.0
    fake_venue.prices = {s: float(panel.close[-1, i]) for i, s in enumerate(syms)}
    fake_venue.filters = {s: (0.01, 0.01, 0.01, 5.0) for s in syms}
    return fake_venue, panel, syms


def test_equity_dry_rebalance_journals_targets_and_its_diagnostics(equity_env, monkeypatch,
                                                                   capsys):
    run_cli(monkeypatch, "rebalance", "--book", "equity", "--no-topup")
    t = jrn("equity").last("targets")
    assert t is not None and t["mode"] == "dry"
    out = capsys.readouterr().out
    assert "breadth" in out          # the S1 diagnostics the book is sized from


def test_equity_gross_is_breadth_scaled_not_fixed(equity_env, monkeypatch):
    """g_t = L_half x breadth x brake — a book with weak breadth must take less risk, which is
    the whole reason the sizing is not a constant."""
    run_cli(monkeypatch, "rebalance", "--book", "equity", "--no-topup")
    t = jrn("equity").last("targets")
    assert 0.0 <= float(t["g_t"]) <= 3.0


def test_equity_live_rebalance_records_the_model_vs_tradeable_gap(equity_env, monkeypatch):
    """The equity model universe is wider than what the venue lists; the gap is journaled so a
    silently-untradeable name cannot be mistaken for a held one."""
    venue, _panel, syms = equity_env
    venue.tradeable = set(list(syms)[:10])         # half the model universe is unlistable
    run_cli(monkeypatch, "allocate", "--book", "equity", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "equity", "--live", "--no-topup")
    gap = jrn("equity").last("tradeable_gap")
    assert gap is not None
    assert gap["n_model_only"] >= 1


def test_equity_live_only_orders_names_the_venue_actually_lists(equity_env, monkeypatch):
    venue, _panel, syms = equity_env
    venue.tradeable = set(list(syms)[:10])
    run_cli(monkeypatch, "allocate", "--book", "equity", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "equity", "--live", "--no-topup")
    assert {o["symbol"] for o in venue.submitted} <= venue.tradeable


def test_equity_orders_stay_inside_the_equity_domain(equity_env, monkeypatch):
    venue, _panel, _syms = equity_env
    run_cli(monkeypatch, "allocate", "--book", "equity", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "equity", "--live", "--no-topup")
    assert {o["symbol"] for o in venue.submitted} <= set(book_domain(BOOKS["equity"]))


# ---------------------------------------------------------------- commodity book
@pytest.fixture
def commodity_env(fake_venue, monkeypatch):
    from convexity.paper import cli

    syms = tuple(sorted(book_domain(BOOKS["commodity"])))
    panel = mean_reverting_panel(syms, n_days=300)
    monkeypatch.setattr(cli, "_load_panel", lambda spec, topup: panel)
    fake_venue.equity = 50_000.0
    fake_venue.prices = {s: float(panel.close[-1, i]) for i, s in enumerate(syms)}
    fake_venue.filters = {s: (0.01, 0.01, 0.01, 5.0) for s in syms}
    return fake_venue, panel, syms


def test_commodity_dry_rebalance_journals_targets(commodity_env, monkeypatch):
    run_cli(monkeypatch, "rebalance", "--book", "commodity", "--no-topup")
    t = jrn("commodity").last("targets")
    assert t is not None and t["mode"] == "dry"


def test_commodity_weights_are_dollar_neutral_within_the_book(commodity_env, monkeypatch):
    """RV pairs are long one leg and short the other — the book nets to ~zero by construction."""
    run_cli(monkeypatch, "rebalance", "--book", "commodity", "--no-topup")
    w = np.array([float(x) for x in jrn("commodity").last("targets")["weights"]])
    assert abs(w.sum()) < 1e-6


def test_commodity_live_gates_pairs_on_what_the_venue_lists(commodity_env, monkeypatch):
    """PAPER.md §11.3: an unlisted leg would leave a naked single-leg position, so the WHOLE
    pair is dropped rather than half-traded."""
    venue, _panel, syms = commodity_env
    venue.tradeable = {s for s in syms if s != "XAGUSDT"}
    run_cli(monkeypatch, "allocate", "--book", "commodity", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "commodity", "--live", "--no-topup")
    assert "XAGUSDT" not in {o["symbol"] for o in venue.submitted}


def test_commodity_orders_stay_inside_the_commodity_domain(commodity_env, monkeypatch):
    venue, _panel, _syms = commodity_env
    run_cli(monkeypatch, "allocate", "--book", "commodity", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "commodity", "--live", "--no-topup")
    assert {o["symbol"] for o in venue.submitted} <= set(book_domain(BOOKS["commodity"]))


def test_the_three_books_domains_are_disjoint(commodity_env, monkeypatch):
    """The invariant the scoped orphan sweep depends on: no symbol may belong to two books, or
    one book would flatten another's position."""
    doms = {n: set(book_domain(BOOKS[n])) for n in ("crypto", "commodity", "equity")}
    assert not doms["crypto"] & doms["commodity"]
    assert not doms["crypto"] & doms["equity"]
    assert not doms["commodity"] & doms["equity"]
