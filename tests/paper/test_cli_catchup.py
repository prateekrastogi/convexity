"""catchup's deployment rules.

catchup exists to put NEW capital to work between weekly rebalances — a deposit lands on Tuesday
and the book should not sit under-deployed until Saturday. That makes it the one command that
places orders outside a rebalance, so its guard rails matter: it must only ever close a gap
TOWARD the target, never trade the other way, and never exceed its per-run cap.
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


def jrn(book: str = "crypto") -> Journal:
    return Journal(book_journal_path(BOOKS[book]))


def seed_targets(nav: float = 1000.0, symbols=("BTCUSDT", "ETHUSDT"), g: float = 1.0):
    j = jrn()
    j.append("allocate", ASOF, amount=nav, book="crypto")
    n = len(symbols)
    j.append("targets", ASOF, symbols=list(symbols), weights=[1.0 / n] * n,
             g_t=g, nav=nav, mode="live", book="crypto", nav_assets=["ALLOC"])
    return j


@pytest.fixture
def cu(fake_venue):
    fake_venue.equity = 5_000.0
    fake_venue.prices = {"BTCUSDT": 100.0, "ETHUSDT": 100.0}
    fake_venue.filters = {s: (0.01, 0.001, 0.001, 5.0) for s in fake_venue.prices}
    return fake_venue


def test_an_empty_book_is_deployed_toward_target(cu, monkeypatch, capsys):
    seed_targets(nav=1000.0)
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "1.0")
    out = capsys.readouterr().out
    assert "coverage 0.0%" in out
    assert len(cu.submitted) == 2
    assert {o["symbol"] for o in cu.submitted} == {"BTCUSDT", "ETHUSDT"}


def test_orders_are_signed_toward_the_target(cu, monkeypatch):
    """A long target must be bought, never sold."""
    seed_targets(nav=1000.0)
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "1.0")
    assert all(o["qty"] > 0 for o in cu.submitted)


def test_an_OVER_held_name_is_left_alone(cu, monkeypatch, capsys):
    """catchup closes a gap toward target; trimming an over-held name is the weekly rebalance's
    job, and doing it here would fight the no-trade band."""
    seed_targets(nav=1000.0)
    # BTC 600 held vs a 500 target — over its own target, but the BOOK is still only 60% covered
    # so catchup does run. (Coverage is book-wide: making BTC hugely over-held would push
    # coverage above the threshold and catchup would correctly do nothing at all.)
    cu.positions = {"BTCUSDT": 6.0, "ETHUSDT": 0.0}
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "1.0")
    assert {o["symbol"] for o in cu.submitted} == {"ETHUSDT"}


def test_a_short_target_is_sold_not_bought(cu, monkeypatch):
    j = jrn()
    j.append("allocate", ASOF, amount=1000.0, book="crypto")
    j.append("targets", ASOF, symbols=["BTCUSDT", "ETHUSDT"], weights=[0.5, -0.5],
             g_t=1.0, nav=1000.0, mode="live", book="crypto", nav_assets=["ALLOC"])
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "1.0")
    by = {o["symbol"]: o["qty"] for o in cu.submitted}
    assert by["BTCUSDT"] > 0
    assert by["ETHUSDT"] < 0


def test_a_symbol_that_left_TRADING_is_skipped(cu, monkeypatch):
    seed_targets(nav=1000.0)
    cu.tradeable = {"BTCUSDT"}
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "1.0")
    assert {o["symbol"] for o in cu.submitted} == {"BTCUSDT"}


def test_a_gap_below_min_notional_is_not_sent(cu, monkeypatch):
    """Sub-minimum orders are rejected by the venue — the planner must drop them."""
    seed_targets(nav=8.0)          # ~4 USDT per name, under the 5 USDT floor
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live")
    assert cu.submitted == []


def test_deployment_is_capped_at_a_fraction_of_nav(cu, monkeypatch, capsys):
    """The cap is what stops a mis-journaled NAV turning into an outsized single-run deployment."""
    seed_targets(nav=1000.0)
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--max-frac", "0.3")
    out = capsys.readouterr().out
    assert "capped at 30%" in out
    deployed = sum(abs(o["qty"]) * o["price"] for o in cu.submitted)
    assert deployed <= 0.3 * 1000.0 + 1e-6


def test_a_book_already_at_target_coverage_does_nothing(cu, monkeypatch, capsys):
    seed_targets(nav=1000.0)
    cu.positions = {"BTCUSDT": 5.0, "ETHUSDT": 5.0}    # 500 + 500 = target
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live")
    out = capsys.readouterr().out
    assert "nothing to do" in out
    assert cu.submitted == []


def test_dry_catchup_reports_but_places_nothing(cu, monkeypatch, capsys):
    seed_targets(nav=1000.0)
    run_cli(monkeypatch, "catchup", "--book", "crypto")
    assert cu.submitted == []
    assert "coverage" in capsys.readouterr().out


def test_if_flow_acts_once_a_deposit_has_been_journaled(cu, monkeypatch):
    """The automatic trigger: no flow means no action, a journaled flow releases it."""
    seed_targets(nav=1000.0)
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--if-flow")
    assert cu.submitted == []

    run_cli(monkeypatch, "flow", "--book", "crypto", "--amount", "500",
            "--equity-before", "1000")
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--if-flow")
    assert len(cu.submitted) > 0
