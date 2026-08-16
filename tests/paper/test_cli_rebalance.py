"""The rebalance session itself, on a synthetic panel.

``cmd_rebalance`` is the largest single block of untested code in the repo and the one that
decides what the fleet actually holds. It is untested for a structural reason: it reads the 24 GB
lake. ``_load_panel`` is the seam — swapping it for an in-memory panel lets the whole session run
(settle -> targets -> sizing -> renormalisation -> plan -> orders -> journal) in milliseconds,
against the venue double.

The panel is built from REAL crypto-domain symbols so the book's filters, price map and domain
gate all line up with production rather than being bypassed.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from convexity.paper.books import BOOKS, book_domain, book_journal_path
from convexity.paper.journal import Journal

from tests.conftest import make_panel

N_DAYS = 220


def run_cli(monkeypatch, *argv: str) -> None:
    from convexity.paper.cli import main

    monkeypatch.setattr(sys, "argv", ["convexity.paper", *argv])
    main()


def jrn(book: str = "crypto") -> Journal:
    return Journal(book_journal_path(BOOKS[book]))


def crypto_symbols(n: int = 24) -> tuple[str, ...]:
    return tuple(sorted(book_domain(BOOKS["crypto"]))[:n])


def trending_panel(symbols: tuple[str, ...], n_days: int = N_DAYS):
    """Deterministic dispersed trends so the cross-sectional quantiles are unambiguous.

    Drifts fan from clearly negative to clearly positive across the universe, which gives the
    momentum band a stable ranking instead of a coin-flip on noise.
    """
    k = len(symbols)
    t = np.arange(n_days)
    close = np.zeros((n_days, k))
    funding = np.zeros((n_days, k))
    listed = np.ones((n_days, k), dtype=bool)
    drifts = np.linspace(-0.003, 0.003, k)
    for j, d in enumerate(drifts):
        close[:, j] = 100.0 * (1.0 + d) ** t
        funding[:, j] = 1e-4 * (j - k / 2) / k
    return make_panel(close, funding, listed, symbols)


@pytest.fixture
def rebalance_env(fake_venue, monkeypatch):
    """A crypto book with a synthetic panel, priced and filtered like the real one."""
    from convexity.paper import cli

    syms = crypto_symbols()
    panel = trending_panel(syms)
    monkeypatch.setattr(cli, "_load_panel", lambda spec, topup: panel)
    # The crypto sizing step ranks liquidity from the 1-min lake (build_dollar_volume ->
    # load_bars), which is the OTHER lake dependency in this path. Supply it directly: a flat,
    # ample notional keeps every name eligible so the test exercises the construction rather
    # than an accidental liquidity screen.
    monkeypatch.setattr(cli, "build_dollar_volume",
                        lambda p, **kw: np.full((p.n_days, len(p.symbols)), 5.0e8))

    fake_venue.equity = 50_000.0
    fake_venue.prices = {s: float(panel.close[-1, i]) for i, s in enumerate(syms)}
    fake_venue.filters = {s: (0.001, 0.001, 0.001, 5.0) for s in syms}
    fake_venue.assets = {"USDT": 50_000.0}
    return fake_venue, panel, syms


# ---------------------------------------------------------------- dry run
def test_dry_rebalance_journals_targets_without_touching_the_venue(rebalance_env, monkeypatch):
    """A dry run is the safe inspection path — it must place nothing."""
    venue, _panel, _syms = rebalance_env
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--no-topup")
    t = jrn().last("targets")
    assert t is not None and t["mode"] == "dry"
    assert venue.submitted == []
    assert venue.cancelled == []


def test_dry_targets_are_dollar_neutral_and_unit_gross(rebalance_env, monkeypatch):
    """The crypto book's construction: dollar-neutral quintiles at unit gross before sizing."""
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--no-topup")
    w = np.array([float(x) for x in jrn().last("targets")["weights"]])
    assert abs(w.sum()) < 1e-6                      # dollar neutral
    assert np.isclose(np.abs(w).sum(), 1.0, atol=1e-6)   # unit gross
    assert (w > 0).any() and (w < 0).any()          # genuinely two-sided


# ---------------------------------------------------------------- live run
def test_live_rebalance_places_orders_and_journals_the_cycle(rebalance_env, monkeypatch):
    venue, _panel, _syms = rebalance_env
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")

    t = jrn().last("targets")
    assert t["mode"] == "live"
    orders = jrn().last("orders")
    assert orders is not None and orders["mode"] == "live"
    assert len(orders["intents"]) > 0
    assert len(venue.submitted) > 0


def test_live_rebalance_only_ever_touches_its_own_domain(rebalance_env, monkeypatch):
    """The disjoint-domain invariant, enforced where it matters: on real submissions."""
    venue, _panel, _syms = rebalance_env
    domain = book_domain(BOOKS["crypto"])
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")
    assert {o["symbol"] for o in venue.submitted} <= set(domain)


def test_live_rebalance_journals_a_renorm_record(rebalance_env, monkeypatch):
    """Partial-book renormalisation is ON for all three books (owner decision 2026-08-13), so a
    live cycle must record what it scaled — the audit trail for a live-risk setting."""
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")
    r = jrn().last("renorm")
    assert r is not None
    assert "k" in r and "n_kept" in r


def test_orders_are_quantized_onto_the_venue_grid(rebalance_env, monkeypatch):
    """A qty off the lot grid is rejected by Binance — quantization is not cosmetic."""
    venue, _panel, syms = rebalance_env
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")
    lot = 0.001
    for it in jrn().last("orders")["intents"]:
        q = abs(float(it["qty"]))
        assert q > 0
        assert abs(round(q / lot) * lot - q) < 1e-9


def test_every_order_clears_min_notional(rebalance_env, monkeypatch):
    """Sub-minimum orders are rejected by the venue; the planner must drop them, not send them."""
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")
    for it in jrn().last("orders")["intents"]:
        if not it.get("reduce_only"):
            assert abs(float(it["qty"])) * float(it["ref_price"]) >= 5.0 - 1e-9


# ---------------------------------------------------------------- guards
def test_live_without_an_allocation_refuses_on_a_non_crypto_book(fake_venue, monkeypatch):
    """The shared account's equity belongs to three books — a book with no allocation cannot
    infer its own capital, and must refuse rather than guess."""
    from convexity.paper import cli

    syms = tuple(sorted(book_domain(BOOKS["commodity"])))
    panel = trending_panel(syms)
    monkeypatch.setattr(cli, "_load_panel", lambda spec, topup: panel)
    fake_venue.prices = {s: 100.0 for s in syms}
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "rebalance", "--book", "commodity", "--live", "--no-topup")


def test_kill_switch_blocks_a_live_rebalance_before_any_order(rebalance_env, monkeypatch):
    venue, _panel, _syms = rebalance_env
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "20000")
    run_cli(monkeypatch, "kill", "--book", "crypto", "--reason", "unit-test")
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--live", "--no-topup")
    assert venue.submitted == []


def test_rerunning_a_dry_rebalance_is_stable(rebalance_env, monkeypatch):
    """Same panel, same inputs -> same targets. Non-determinism here would be untrackable live."""
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--no-topup")
    first = json.dumps(jrn().last("targets")["weights"])
    run_cli(monkeypatch, "rebalance", "--book", "crypto", "--no-topup")
    assert json.dumps(jrn().last("targets")["weights"]) == first
