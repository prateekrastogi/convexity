"""The operator carve-out (PAPER.md §12.5).

The owner trades the same Binance account from the phone app. The carve-out is what stops the
fleet treating those positions as its own and flattening them: every fleet order is stamped
``cvx-``, and any terminal fill WITHOUT that stamp is adopted into a reserve ledger that each
book nets out of its sizing, orphan sweep and kill.

``bootstrap`` is the one-time prod-cutover step, and its refusal conditions are the important
part: seeding a ledger on an account where a book has already traded would hand the operator
positions the FLEET opened, and the fleet would then never touch them again.
"""

from __future__ import annotations

import sys

import pytest

from convexity.paper.books import BOOKS, book_journal_path
from convexity.paper.journal import Journal

ASOF = "2026-08-14"


def run_cli(monkeypatch, *argv: str) -> None:
    from convexity.paper.cli import main

    monkeypatch.setattr(sys, "argv", ["convexity.paper", *argv])
    main()


def reserve() -> dict:
    """Read the ledger through the module's own loader rather than parsing the file — the
    on-disk shape is the module's business, and duplicating it here would test my guess at it."""
    from convexity.paper.manual import load_reserve

    return dict(load_reserve())


@pytest.fixture
def mv(fake_venue):
    fake_venue.equity = 5_000.0
    fake_venue.prices = {"BTCUSDT": 100.0, "ETHUSDT": 100.0}
    return fake_venue


# ---------------------------------------------------------------- read-only view
def test_manual_view_on_a_virgin_account_reports_no_reserves(mv, monkeypatch, capsys):
    run_cli(monkeypatch, "manual")
    out = capsys.readouterr().out
    assert "no operator reserves" in out
    assert reserve() == {}


def test_manual_view_lists_existing_reserves(mv, monkeypatch, capsys):
    from convexity.paper.manual import save_reserve

    save_reserve({"BTCUSDT": 0.25, "ETHUSDT": -1.5})
    run_cli(monkeypatch, "manual")
    out = capsys.readouterr().out
    assert "BTCUSDT" in out and "ETHUSDT" in out


def test_the_view_never_mutates_the_ledger(mv, monkeypatch, capsys):
    from convexity.paper.manual import save_reserve

    save_reserve({"BTCUSDT": 0.25})
    run_cli(monkeypatch, "manual")
    assert reserve() == {"BTCUSDT": 0.25}


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_on_a_virgin_account_seeds_every_open_position(mv, monkeypatch, capsys):
    """The prod-cutover case: no book has traded, so everything on the venue is the operator's."""
    mv.positions = {"BTCUSDT": 0.5, "ETHUSDT": -2.0}
    run_cli(monkeypatch, "manual", "--bootstrap")
    out = capsys.readouterr().out
    assert "seeded" in out
    assert reserve() == {"BTCUSDT": 0.5, "ETHUSDT": -2.0}


def test_bootstrap_with_no_venue_positions_still_anchors_attribution(mv, monkeypatch, capsys):
    """An empty ledger is a real outcome — the point of the run is to anchor deploy_ms so
    pre-cutover history stays out of attribution."""
    run_cli(monkeypatch, "manual", "--bootstrap")
    out = capsys.readouterr().out
    assert "no venue positions" in out
    assert reserve() == {}


def test_bootstrap_refuses_once_a_book_has_traded_live(mv, monkeypatch):
    """THE guard. After a live cycle some venue positions belong to the FLEET; seeding them as
    operator-owned would mean the fleet never touches its own book again."""
    j = Journal(book_journal_path(BOOKS["crypto"]))
    j.append("targets", ASOF, symbols=["BTCUSDT"], weights=[1.0], g_t=1.0, nav=1000.0,
             mode="live", book="crypto", nav_assets=["ALLOC"])
    mv.positions = {"BTCUSDT": 0.5}
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "manual", "--bootstrap")


def test_a_dry_cycle_does_not_block_bootstrap(mv, monkeypatch):
    """Only LIVE targets mean the fleet actually holds something; a dry run traded nothing."""
    j = Journal(book_journal_path(BOOKS["crypto"]))
    j.append("targets", ASOF, symbols=["BTCUSDT"], weights=[1.0], g_t=1.0, nav=1000.0,
             mode="dry", book="crypto", nav_assets=["ALLOC"])
    mv.positions = {"BTCUSDT": 0.5}
    run_cli(monkeypatch, "manual", "--bootstrap")
    assert reserve() == {"BTCUSDT": 0.5}


def test_bootstrap_refuses_when_the_ledger_is_already_populated(mv, monkeypatch):
    """Re-running it would re-seed whatever is on the venue now, silently adopting fleet
    positions opened since the first bootstrap."""
    from convexity.paper.manual import save_reserve

    save_reserve({"BTCUSDT": 0.1})
    mv.positions = {"BTCUSDT": 0.5, "ETHUSDT": 1.0}
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "manual", "--bootstrap")
    assert reserve() == {"BTCUSDT": 0.1}
