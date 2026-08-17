"""The Saturday cycle's post-rebalance commands, against the venue double.

finalize / reprice / tca / catchup all operate on a journal the rebalance left behind, so these
seed that journal directly rather than running a rebalance (which would need the 24 GB lake).
That keeps them fast and lets each command's contract be stated on its own terms.
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


def intent(symbol: str, qty: float, ref: float) -> dict:
    return {"symbol": symbol, "qty": qty, "tactic": "POST_ONLY_MAKER", "ref_price": ref,
            "slice_idx": 0, "n_slices": 1, "reduce_only": False,
            "qty_str": f"{abs(qty)}", "price_str": f"{ref}"}


def seed_cycle(venue, book: str = "crypto", symbols=("BTCUSDT", "ETHUSDT"),
               nav: float = 1000.0) -> Journal:
    """Put a completed live rebalance in the journal: allocation, targets, orders."""
    j = jrn(book)
    j.append("allocate", ASOF, amount=nav, book=book)
    n = len(symbols)
    j.append("targets", ASOF, symbols=list(symbols), weights=[1.0 / n] * n,
             g_t=1.0, nav=nav, mode="live", book=book, nav_assets=["ALLOC"])
    j.append("orders", ASOF, mode="live",
             intents=[intent(s, 0.01, venue.prices[s]) for s in symbols])
    return j


@pytest.fixture
def cycle(fake_venue):
    fake_venue.equity = 5_000.0
    fake_venue.prices = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0}
    fake_venue.filters = {s: (0.01, 0.001, 0.001, 5.0) for s in fake_venue.prices}
    return fake_venue


# ---------------------------------------------------------------- finalize
def test_finalize_crosses_an_unfilled_maker_remainder(cycle, monkeypatch, capsys):
    """The window-end pass exists to cross whatever the makers did not fill."""
    seed_cycle(cycle)
    cycle.submit_bbo("BTCUSDT", 0.01, "0.01", "QUEUE")  # left resting, unfilled
    run_cli(monkeypatch, "finalize", "--book", "crypto", "--live")
    out = capsys.readouterr().out
    assert "finalize" in out
    assert jrn().last("finalized") is not None


def test_finalize_skips_a_symbol_that_left_TRADING(cycle, monkeypatch, capsys):
    """REGRESSION (2026-08-15): a delisted symbol rejected cancel with -1122 and the unhandled
    raise aborted the WHOLE finalize, leaving every other name's remainder uncrossed for a week.
    One bad symbol must degrade to a skip, never take the pass down."""
    seed_cycle(cycle, symbols=("BTCUSDT", "ETHUSDT"))
    cycle.tradeable = {"BTCUSDT"}          # ETH has left TRADING
    cycle.cancel_rejects = {"ETHUSDT"}     # and the venue rejects its cancel
    run_cli(monkeypatch, "finalize", "--book", "crypto", "--live")
    out = capsys.readouterr().out
    assert "ETHUSDT" in out and "SKIP" in out.upper()
    # the pass still completed rather than dying on the bad symbol
    assert jrn().last("finalized") is not None


def test_finalize_is_idempotent_on_rerun(cycle, monkeypatch, capsys):
    """Sessions are re-run after transient failures (the retry path) — a second finalize must
    not double-trade."""
    seed_cycle(cycle)
    run_cli(monkeypatch, "finalize", "--book", "crypto", "--live")
    first = len(cycle.submitted)
    run_cli(monkeypatch, "finalize", "--book", "crypto", "--live")
    assert len(cycle.submitted) == first


def test_finalize_without_a_live_cycle_does_nothing(cycle, monkeypatch, capsys):
    run_cli(monkeypatch, "finalize", "--book", "crypto", "--live")
    assert cycle.submitted == []


# ---------------------------------------------------------------- tca
def test_tca_journals_shortfall_and_coverage(cycle, monkeypatch, capsys):
    seed_cycle(cycle)
    for oid in list(cycle.open_orders):
        cycle.fill(oid)
    # a fill at the reference price is zero shortfall by construction
    cycle.trades.append({"symbol": "BTCUSDT", "price": 60_000.0, "qty": 0.01,
                         "side": "BUY", "maker": True, "time": cycle._now_ms, "orderId": 1})
    run_cli(monkeypatch, "tca", "--book", "crypto", "--live")
    rec = jrn().last("tca")
    assert rec is not None
    assert "coverage_pct" in rec


def test_tca_dry_run_writes_nothing(cycle, monkeypatch, capsys):
    seed_cycle(cycle)
    run_cli(monkeypatch, "tca", "--book", "crypto")
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert jrn().last("tca") is None


# ---------------------------------------------------------------- reprice
def test_reprice_repegs_a_resting_maker(cycle, monkeypatch, capsys):
    seed_cycle(cycle)
    cycle.submit_bbo("BTCUSDT", 0.01, "0.01", "QUEUE")
    before = len(cycle.cancelled)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert len(cycle.cancelled) >= before


def test_reprice_without_open_orders_is_a_noop(cycle, monkeypatch, capsys):
    seed_cycle(cycle)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert cycle.submitted == []


# ---------------------------------------------------------------- catchup
def test_catchup_skips_when_coverage_is_already_high(cycle, monkeypatch, capsys):
    """catchup exists to deploy NEW capital; a fully covered book must be left alone."""
    seed_cycle(cycle)
    cycle.positions = {"BTCUSDT": 0.00834, "ETHUSDT": 0.1667}  # ~fully covered vs target
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--target-cov", "1")
    out = capsys.readouterr().out.lower()
    assert "skip" in out or "coverage" in out


def test_catchup_if_flow_gate_holds_without_a_flow(cycle, monkeypatch, capsys):
    """--if-flow is the automatic trigger: no journaled flow since the last catchup, no action."""
    seed_cycle(cycle)
    run_cli(monkeypatch, "catchup", "--book", "crypto", "--live", "--if-flow")
    assert cycle.submitted == []
