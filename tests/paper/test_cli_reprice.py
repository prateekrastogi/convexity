"""reprice's re-peg rules.

Between the rebalance and the finalize the fleet runs three reprice passes. Their job is to chase
a moving touch with the UNFILLED remainder so the cycle completes on maker fills instead of
paying the crossing spread at finalize. The remainder is computed from OUR journaled orderIds'
actual fills — never inferred from a position snapshot — so the interesting cases are all about
what it does when that fill state is partial, unreadable, or already complete.
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


def intent(symbol: str, qty: float, ref: float, reduce_only: bool = False) -> dict:
    return {"symbol": symbol, "qty": qty, "tactic": "POST_ONLY_MAKER", "ref_price": ref,
            "slice_idx": 0, "n_slices": 1, "reduce_only": reduce_only,
            "qty_str": str(abs(qty)), "price_str": str(ref)}


@pytest.fixture
def rp(fake_venue):
    fake_venue.equity = 5_000.0
    fake_venue.prices = {"BTCUSDT": 100.0, "ETHUSDT": 100.0}
    fake_venue.filters = {s: (0.01, 0.001, 0.001, 5.0) for s in fake_venue.prices}
    return fake_venue


def seed_cycle(venue, intents, with_ids: bool = True):
    """A live cycle plus, optionally, the submission_ids record reprice needs to read fills."""
    j = jrn()
    j.append("allocate", ASOF, amount=1000.0, book="crypto")
    j.append("targets", ASOF, symbols=[i["symbol"] for i in intents],
             weights=[1.0 / len(intents)] * len(intents), g_t=1.0, nav=1000.0,
             mode="live", book="crypto", nav_assets=["ALLOC"])
    j.append("orders", ASOF, mode="live", intents=intents)
    posted = []
    for it in intents:
        r = venue.submit_bbo(it["symbol"], float(it["qty"]), it["qty_str"], "QUEUE")
        posted.append({"symbol": it["symbol"], "orderId": r["orderId"],
                       "clientOrderId": r["clientOrderId"], "qty": float(it["qty"])})
    if with_ids:
        j.append("submission_ids", ASOF, orders=posted)
    return j, posted


# ---------------------------------------------------------------- gating
def test_a_journal_without_submission_ids_is_skipped(rp, monkeypatch, capsys):
    """A v1 journal has no orderIds to read fills from, so there is no safe remainder to re-peg."""
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)], with_ids=False)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert "v1 journal" in capsys.readouterr().out
    assert jrn().last("repriced") is None


def test_reprice_cancels_resting_orders_before_reposting(rp, monkeypatch):
    """Re-pegging without cancelling would leave two live orders for one intent — a double fill."""
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert "BTCUSDT" in rp.cancelled


# ---------------------------------------------------------------- remainder handling
def test_an_unfilled_intent_is_reposted_at_the_touch(rp, monkeypatch):
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    before = len(rp.submitted)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert len(rp.submitted) > before
    rec = jrn().last("repriced")
    assert rec["n_posted"] == 1


def test_a_fully_filled_intent_is_counted_complete_and_not_reposted(rp, monkeypatch):
    _j, posted = seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.fill(posted[0]["orderId"])
    before = len(rp.submitted)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert len(rp.submitted) == before
    assert jrn().last("repriced")["n_complete"] == 1


def test_only_the_unfilled_remainder_is_reposted(rp, monkeypatch):
    """The whole point: a partially filled intent must be chased for what is LEFT, not re-sent
    in full, which would over-trade the target."""
    _j, posted = seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.fill(posted[0]["orderId"], qty=0.4)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    repost = rp.submitted[-1]
    assert repost["qty"] == pytest.approx(0.6, abs=1e-6)


def test_a_residue_below_min_notional_is_treated_as_done(rp, monkeypatch):
    """Chasing a 1 USDT tail would only draw rejections."""
    _j, posted = seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.fill(posted[0]["orderId"], qty=0.99)      # 1 USDT left, under the 5 USDT floor
    before = len(rp.submitted)
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert len(rp.submitted) == before
    assert jrn().last("repriced")["n_complete"] == 1


# ---------------------------------------------------------------- skips
def test_a_symbol_that_left_TRADING_is_skipped(rp, monkeypatch):
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0), intent("ETHUSDT", 1.0, 100.0)])
    rp.tradeable = {"BTCUSDT"}
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    rec = jrn().last("repriced")
    assert rec["n_skipped"] >= 1
    assert rec["n_posted"] == 1


def test_a_crossed_or_empty_book_is_skipped_rather_than_pegged(rp, monkeypatch):
    """Pegging against a crossed book would post through the spread — worse than not repricing."""
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.spread_bp = -50.0                          # ask < bid
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert jrn().last("repriced")["n_skipped"] == 1


def test_a_symbol_with_no_venue_filters_is_skipped(rp, monkeypatch):
    seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.filters = {"ETHUSDT": (0.01, 0.001, 0.001, 5.0)}   # BTC absent
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    assert jrn().last("repriced")["n_skipped"] == 1


def test_the_pass_is_always_journaled(rp, monkeypatch):
    """Even a pass that posts nothing must leave a record — the audit trail for the window."""
    _j, posted = seed_cycle(rp, [intent("BTCUSDT", 1.0, 100.0)])
    rp.fill(posted[0]["orderId"])
    run_cli(monkeypatch, "reprice", "--book", "crypto", "--live")
    rec = jrn().last("repriced")
    assert rec is not None
    assert {"n_posted", "n_complete", "n_skipped", "n_failed"} <= set(rec)
