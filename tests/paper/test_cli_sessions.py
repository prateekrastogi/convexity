"""Live-session CLI coverage, driven against the in-memory venue double.

``paper/cli.py`` is the live trading path and was the least-covered module in the repo — 942 of
1120 statements uncovered — because every command talks to an exchange. These drive the real
commands through ``main()`` (so argparse is exercised too) against ``fakevenue``, with all state
isolated in tmp_path.

Read as a description of what a session is contracted to do: journal what it did, never touch
positions outside its own book's domain, and stay idempotent on re-run.
"""

from __future__ import annotations

import json
import sys

import pytest

from convexity.paper.books import BOOKS, book_journal_path
from convexity.paper.journal import Journal


def run_cli(monkeypatch, *argv: str) -> None:
    """Invoke the CLI exactly as the container's crontab does."""
    from convexity.paper.cli import main

    monkeypatch.setattr(sys, "argv", ["convexity.paper", *argv])
    main()


def jrn(book: str = "crypto") -> Journal:
    return Journal(book_journal_path(BOOKS[book]))


def kinds(book: str = "crypto") -> list[str]:
    path = book_journal_path(BOOKS[book])
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line)["kind"])
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return out


# ---------------------------------------------------------------- capital flows
def test_flow_journals_a_signed_amount(fake_venue, monkeypatch, capsys):
    """A deposit/withdrawal must be journaled the same day or the TWR chain silently drifts."""
    run_cli(monkeypatch, "flow", "--book", "crypto", "--amount", "500", "--equity-before", "1000")
    rec = jrn().last("flow")
    assert rec is not None and float(rec["amount"]) == 500.0
    assert "flow" in kinds()


def test_flow_accepts_a_withdrawal(fake_venue, monkeypatch):
    run_cli(monkeypatch, "flow", "--book", "crypto", "--amount", "-250",
            "--equity-before", "1000")
    assert float(jrn().last("flow")["amount"]) == -250.0


def test_allocate_records_the_books_capital(fake_venue, monkeypatch):
    run_cli(monkeypatch, "allocate", "--book", "equity", "--amount", "300")
    rec = jrn("equity").last("allocate")
    assert rec is not None and float(rec["amount"]) == 300.0


def test_allocate_split_divides_the_total_across_all_three_books(fake_venue, monkeypatch):
    """The fleet is THREE books at 40/30/30 — the split must reach every one of them."""
    run_cli(monkeypatch, "allocate-split", "--total", "1000")
    amounts = {b: float(jrn(b).last("allocate")["amount"]) for b in BOOKS}
    assert set(amounts) == {"crypto", "commodity", "equity"}
    assert amounts["crypto"] == pytest.approx(400.0, abs=1e-6)
    assert amounts["commodity"] == pytest.approx(300.0, abs=1e-6)
    assert amounts["equity"] == pytest.approx(300.0, abs=1e-6)
    assert sum(amounts.values()) == pytest.approx(1000.0, abs=1e-6)


def test_flow_split_divides_a_deposit_across_all_three_books(fake_venue, monkeypatch):
    run_cli(monkeypatch, "flow-split", "--amount", "500")
    total = sum(float(jrn(b).last("flow")["amount"]) for b in BOOKS)
    assert total == pytest.approx(500.0, abs=1e-6)


# ---------------------------------------------------------------- kill switch
def test_kill_arms_and_blocks_a_live_session(fake_venue, monkeypatch, capsys):
    """The kill switch must never auto-re-arm, and must stop a live session dead."""
    run_cli(monkeypatch, "kill", "--book", "crypto", "--reason", "unit-test")
    capsys.readouterr()
    run_cli(monkeypatch, "recon", "--book", "crypto", "--live")
    out = capsys.readouterr().out.lower()
    assert "kill" in out or "halt" in out


def test_kill_resume_clears_the_halt(fake_venue, monkeypatch, capsys):
    run_cli(monkeypatch, "kill", "--book", "crypto", "--reason", "unit-test")
    run_cli(monkeypatch, "kill", "--book", "crypto", "--resume")
    capsys.readouterr()
    run_cli(monkeypatch, "recon", "--book", "crypto", "--live")
    out = capsys.readouterr().out.lower()
    assert "recon" in out


# ---------------------------------------------------------------- reconciliation
def test_recon_live_reports_positions_against_the_venue(fake_venue, monkeypatch, capsys):
    fake_venue.equity = 5_000.0
    fake_venue.prices = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0}
    fake_venue.positions = {"BTCUSDT": 0.01}
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "1000")
    run_cli(monkeypatch, "recon", "--book", "crypto", "--live")
    assert "recon" in kinds()
    rec = jrn().last("recon")
    assert rec is not None and rec["n_positions"] >= 1


def test_recon_ignores_symbols_outside_the_books_domain(fake_venue, monkeypatch):
    """A position the book does not own must never be counted as the book's."""
    fake_venue.prices = {"BTCUSDT": 60_000.0}
    fake_venue.positions = {"BTCUSDT": 0.01, "NOTAREALSYMBOLUSDT": 5.0}
    run_cli(monkeypatch, "allocate", "--book", "crypto", "--amount", "1000")
    run_cli(monkeypatch, "recon", "--book", "crypto", "--live")
    rec = jrn().last("recon")
    assert rec["n_positions"] == 1


# ---------------------------------------------------------------- status
def test_status_is_read_only_and_survives_an_empty_journal(fake_venue, monkeypatch, capsys):
    run_cli(monkeypatch, "status", "--book", "commodity")
    out = capsys.readouterr().out
    assert "status" in out.lower()
    assert kinds("commodity") == []
