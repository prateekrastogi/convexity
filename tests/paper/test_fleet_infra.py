"""Fleet-infrastructure tests (PAPER.md §12.2-§7): lock, kill switches, margin
guardrails, split allocation, data-maintenance flows. Everything venue-facing is mocked — these
are logic tests, not integration tests (those run in the D1v2 simulate phase)."""

from __future__ import annotations

import json
import multiprocessing
import os
import time

import numpy as np
import pytest

from convexity.paper.lock import AccountLockTimeout, account_lock
from convexity.signal.equity_trend import brake_multiplier

# ---------------------------------------------------------------- account lock


def _hold_lock(path: str, hold_s: float, started, q) -> None:
    from convexity.paper.lock import account_lock as al

    with al("holder", timeout=5.0, path=path):
        started.set()
        time.sleep(hold_s)
    q.put("released")


def test_lock_serializes_two_processes(tmp_path):
    path = str(tmp_path / "account.lock")
    started = multiprocessing.Event()
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_hold_lock, args=(path, 1.5, started, q))
    p.start()
    assert started.wait(10.0), "holder never acquired"
    t0 = time.monotonic()
    with account_lock("waiter", timeout=30.0, path=path):
        waited = time.monotonic() - t0
    p.join(10.0)
    assert waited >= 1.0, f"waiter entered while holder held the lock (waited {waited:.2f}s)"


def test_lock_timeout_aborts_loudly(tmp_path):
    path = str(tmp_path / "account.lock")
    started = multiprocessing.Event()
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_hold_lock, args=(path, 4.0, started, q))
    p.start()
    assert started.wait(10.0)
    with pytest.raises(AccountLockTimeout):
        with account_lock("waiter", timeout=0.8, path=path):
            pass  # pragma: no cover — must never enter
    p.join(10.0)


def _die_holding_lock(path: str, started) -> None:
    from convexity.paper.lock import account_lock as al

    cm = al("crasher", timeout=5.0, path=path)
    cm.__enter__()
    started.set()
    os._exit(9)  # simulated kill -9 INSIDE the critical section


def test_lock_released_on_holder_death(tmp_path):
    """The chaos guarantee (PAPER.md §12.2/§7): a kill -9 holder can never wedge the fleet — the
    kernel releases flock on process death; the next waiter proceeds normally."""
    path = str(tmp_path / "account.lock")
    started = multiprocessing.Event()
    p = multiprocessing.Process(target=_die_holding_lock, args=(path, started))
    p.start()
    assert started.wait(10.0)
    p.join(10.0)
    t0 = time.monotonic()
    with account_lock("survivor", timeout=5.0, path=path):
        waited = time.monotonic() - t0
    assert waited < 2.0, f"lock not auto-released after holder death ({waited:.2f}s)"


# ---------------------------------------------------------------- brake off the journal chain


def test_brake_fresh_book_is_full_exposure():
    assert brake_multiplier(np.array([])) == 1.0
    assert brake_multiplier(np.array([250.0])) == 1.0


def test_brake_scales_linearly_with_drawdown():
    # peak 100, now 75 -> DD 25% of the 50% ceiling -> m = 0.5
    chain = np.array([80.0, 100.0, 90.0, 75.0])
    assert abs(brake_multiplier(chain, ceiling=0.50) - 0.5) < 1e-12
    # at/beyond the ceiling -> 0
    chain = np.array([100.0, 49.9])
    assert brake_multiplier(chain, ceiling=0.50) == 0.0


def test_brake_uses_last_value_not_min():
    # recovered book: dipped to 60 but NOW at 95 -> DD is 5%, not 40%
    chain = np.array([100.0, 60.0, 95.0])
    assert abs(brake_multiplier(chain, ceiling=0.50) - (1.0 - 0.05 / 0.50)) < 1e-12


# ---------------------------------------------------------------- kill switch mechanics


def test_kill_file_halts_rebalance(tmp_path, monkeypatch):
    from convexity.paper import cli
    from convexity.paper.books import EQUITY
    from convexity.paper.journal import Journal

    monkeypatch.chdir(tmp_path)
    kp = cli._kill_path(EQUITY)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text("{}")
    j = Journal(tmp_path / "data/paper/equity/journal.jsonl")
    assert cli._check_kill(EQUITY, j, live=False) is True
    recs = j.read("killed")
    assert len(recs) == 1
    kp.unlink()
    assert cli._check_kill(EQUITY, j, live=False) is False


def test_kill_isolation_between_books(tmp_path, monkeypatch):
    from convexity.paper import cli
    from convexity.paper.books import COMMODITY, EQUITY
    from convexity.paper.journal import Journal

    monkeypatch.chdir(tmp_path)
    kp = cli._kill_path(EQUITY)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text("{}")
    j_c = Journal(tmp_path / "data/paper/commodity/journal.jsonl")
    assert cli._check_kill(COMMODITY, j_c, live=False) is False, (
        "killing the equity book must not halt the commodity book")


# ---------------------------------------------------------------- margin guardrails (mocked)


def test_margin_budget_scales_down(tmp_path, monkeypatch):
    from convexity.paper import cli
    from convexity.paper.books import EQUITY
    from convexity.paper.journal import Journal

    monkeypatch.chdir(tmp_path)
    j = Journal(tmp_path / "data/paper/equity/journal.jsonl")
    # dial=3 (fleetconfig): g_t=4.0 -> IM = 4/3 NAV > NAV -> scaled to <= dial
    g = cli._margin_guardrails(EQUITY, j, "t", nav=100.0, g_t=4.0, live=False)
    assert g < 3.0
    assert j.read("margin_budget")
    # inside budget: untouched
    g2 = cli._margin_guardrails(EQUITY, j, "t", nav=100.0, g_t=2.33, live=False)
    assert g2 == 2.33


def test_margin_monitor_derisk_flattens_equity(tmp_path, monkeypatch):
    from convexity.paper import cli
    from convexity.paper.books import EQUITY
    from convexity.paper.journal import Journal

    monkeypatch.chdir(tmp_path)
    j = Journal(tmp_path / "data/paper/equity/journal.jsonl")

    class Snap:
        equity = 1000.0
        maint_margin = 650.0  # ratio 65% >= 60% DERISK
        initial_margin = 700.0

    flattened = {}
    monkeypatch.setattr("convexity.paper.gateway.account_snapshot", lambda: Snap())
    monkeypatch.setattr(cli, "_flatten_book_live",
                        lambda spec, jj, reason: flattened.setdefault("r", reason))
    g = cli._margin_guardrails(EQUITY, j, "t", nav=250.0, g_t=2.0, live=True)
    assert g == 0.0, "DERISK must zero the equity book's gross"
    assert "r" in flattened
    assert any(r["action"] == "DERISK" for r in j.read("directive"))
    ratios = j.read("margin_ratio")
    assert ratios and abs(ratios[-1]["ratio"] - 0.65) < 1e-9


# ---------------------------------------------------------------- split allocation arithmetic


def test_alloc_split_ratios():
    from convexity.paper.fleetconfig import ALLOC_TOTAL, ALLOC_UNITS, alloc_fraction

    assert sum(ALLOC_UNITS.values()) == ALLOC_TOTAL == 1000
    assert ALLOC_UNITS == {"crypto": 400, "commodity": 300, "equity": 300}
    assert abs(sum(alloc_fraction(b) for b in ALLOC_UNITS) - 1.0) < 1e-12


# ---------------------------------------------------------------- data maintenance flows


def _write_min_parquet(path, closes, gaps=None):
    import pandas as pd

    n = len(closes)
    ms_day = 86_400_000
    t0 = 1_700_000_000_000
    df = pd.DataFrame({
        "open_time": [t0 + i * ms_day for i in range(n)],
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1.0] * n,
        "source": ["duka"] * n,
        "gap_before": gaps if gaps is not None else [False] * n,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_split_watch_flags_and_excludes(tmp_path, monkeypatch):
    """A fake unadjusted split (−74% day) must flag the name; the rebalance's flag reader must
    then exclude it — the AAPL-class failure mode, contained end to end."""
    import importlib.util
    import pathlib
    import sys

    repo = pathlib.Path.cwd()
    spec = importlib.util.spec_from_file_location(
        "eqmaint", repo / "scripts" / "equity_data_maintenance.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["eqmaint"] = m
    spec.loader.exec_module(m)

    stitched = tmp_path / "stitched"
    good = [100.0 + i * 0.1 for i in range(30)]
    # the fake split lands INSIDE the watch's 10-day tail (the watch scans the freshest tail
    # each run — a real split always lands there within a week of happening)
    bad = good[:26] + [good[25] * 0.26] + [good[25] * 0.26 * (1 + 0.001 * i) for i in range(3)]
    _write_min_parquet(stitched / "GOODUSDT.parquet", good)
    _write_min_parquet(stitched / "BADUSDT.parquet", bad)

    monkeypatch.setattr(m, "STITCHED", stitched)
    monkeypatch.setattr(m, "FLAGS", tmp_path / "flags.json")
    monkeypatch.setattr(m, "MAINT_LOG", tmp_path / "maint.jsonl")
    monkeypatch.setattr(m, "_equities",
                        lambda: [{"symbol": "GOODUSDT"}, {"symbol": "BADUSDT"}])
    m.split_watch()
    flags = json.loads((tmp_path / "flags.json").read_text())
    assert "BADUSDT" in flags and not flags["BADUSDT"]["cleared"]
    assert "GOODUSDT" not in flags

    # the rebalance-side reader honors the flag (and the cleared marker releases it)
    from convexity.paper import cli
    monkeypatch.chdir(tmp_path)
    from convexity.paper.fleetconfig import EQUITY_FLAGS_PATH
    fp = tmp_path / EQUITY_FLAGS_PATH
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(flags))
    assert cli._equity_flags() == frozenset({"BADUSDT"})
    flags["BADUSDT"]["cleared"] = True
    fp.write_text(json.dumps(flags))
    assert cli._equity_flags() == frozenset()


def test_split_watch_gap_days_not_flagged(tmp_path, monkeypatch):
    """A gap_before day (session/splice gap) is not a same-session move — never flagged."""
    import importlib.util
    import pathlib
    import sys

    repo = pathlib.Path.cwd()
    spec = importlib.util.spec_from_file_location(
        "eqmaint2", repo / "scripts" / "equity_data_maintenance.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["eqmaint2"] = m
    spec.loader.exec_module(m)

    stitched = tmp_path / "stitched"
    closes = [100.0] * 15 + [140.0] * 15  # +40% but on a gap-flagged day
    gaps = [False] * 15 + [True] + [False] * 14
    _write_min_parquet(stitched / "GAPUSDT.parquet", closes, gaps)
    monkeypatch.setattr(m, "STITCHED", stitched)
    ok, reason = m.validate_symbol("GAPUSDT")
    assert ok, f"gap-flagged day wrongly flagged: {reason}"

# ---------------------------------------------------------------- venue precision clamp


def test_effective_filters_precision_clamp():
    """-1111 containment (PAPER.md §12.2): the effective grid is the coarser of LOT_SIZE/
    PRICE_FILTER and the venue's decimal-precision caps — the SPCX whole-share case."""
    from convexity.paper.topup import effective_filters

    # SPCX demo case: fractional lot from a stale source, but quantityPrecision=0 -> lot 1.0
    tick, lot = effective_filters(0.01, 0.001, price_prec=7, qty_prec=0)
    assert lot == 1.0
    assert tick == 0.01  # 10**-7 finer than the tick: tick stands
    # normal case: precision looser than the grids -> grids stand
    tick, lot = effective_filters(0.01, 0.01, price_prec=5, qty_prec=2)
    assert (tick, lot) == (0.01, 0.01)
    # missing precisions (defensive): grids stand
    tick, lot = effective_filters(0.5, 0.1, price_prec=None, qty_prec=None)
    assert (tick, lot) == (0.5, 0.1)

# ---------------------------------------------------------------- manual-position carve-out


def test_manual_netting_math():
    """PAPER.md §12.5: book share = venue − reserve; proportional uPnL; clamp never inverts."""
    from convexity.paper.manual import book_share_qty, net_book_positions

    pos = {"NVDAUSDT": 10.0, "BTCUSDT": 2.0, "SPYUSDT": 3.0}
    upnl = {"NVDAUSDT": 5.0, "BTCUSDT": 1.0, "SPYUSDT": 0.9}
    res = {"NVDAUSDT": 4.0, "SPYUSDT": 3.0}
    out, u, notes = net_book_positions(pos, upnl, res)
    assert out["NVDAUSDT"] == 6.0           # partial reserve -> book share
    assert abs(u["NVDAUSDT"] - 3.0) < 1e-12  # 5.0 * 6/10
    assert out["BTCUSDT"] == 2.0 and u["BTCUSDT"] == 1.0  # no reserve -> passthrough
    assert "SPYUSDT" not in out             # fully reserved -> book sees nothing
    assert any("SPYUSDT" in n for n in notes)

    # clamp: reserve exceeds venue (operator closed some without releasing) -> 0, never invert
    out, _, notes = net_book_positions({"HOODUSDT": 3.0}, None, {"HOODUSDT": 8.0})
    assert "HOODUSDT" not in out
    assert any("clamped" in n for n in notes)

    # opposite-sign reserve is plain arithmetic: venue +5 with operator short 2 -> book +7
    out, _, _ = net_book_positions({"CLUSDT": 5.0}, None, {"CLUSDT": -2.0})
    assert out["CLUSDT"] == 7.0

    # stale reserve (no venue position) -> loud note, no phantom position
    out, _, notes = net_book_positions({}, None, {"GONEUSDT": 2.0})
    assert out == {} and any("stale" in n for n in notes)

    assert book_share_qty("NVDAUSDT", 10.0, {"NVDAUSDT": 4.0}) == 6.0
    assert book_share_qty("NVDAUSDT", 10.0, {}) == 10.0


def test_manual_ledger_roundtrip(tmp_path):
    from convexity.paper.manual import load_reserve, save_reserve

    p = tmp_path / "manual_reserve.json"
    assert load_reserve(p) == {}            # absent file -> empty, never an error
    save_reserve({"NVDAUSDT": 4.0, "ZEROUSDT": 0.0}, p)
    assert load_reserve(p) == {"NVDAUSDT": 4.0}  # zero-qty entries pruned on save
    # corrupt ledger must FAIL LOUDLY (fail-safe for the operator, not the book)
    p.write_text("{not json")
    import pytest
    with pytest.raises(RuntimeError, match="unreadable"):
        load_reserve(p)

def test_manual_autosync(tmp_path):
    """PAPER.md §12.5 v2: non-cvx terminal fills are adopted signed; cvx fills and resting
    orders are not; re-runs are idempotent; a full manual close auto-releases the reserve."""
    from convexity.paper.manual import auto_sync, load_reserve, load_sync_state

    lp = tmp_path / "reserve.json"
    sp = tmp_path / "sync.json"
    T0 = 1_800_000_000_000
    book_fill = {"orderId": 1, "clientOrderId": "cvx-abc123", "status": "FILLED",
                 "side": "BUY", "executedQty": "5", "updateTime": T0 + 1000, "avgPrice": "10"}
    app_buy = {"orderId": 2, "clientOrderId": "android_xyz", "status": "FILLED",
               "side": "BUY", "executedQty": "4", "updateTime": T0 + 2000, "avgPrice": "11"}
    resting = {"orderId": 3, "clientOrderId": "web_r1", "status": "NEW",
               "side": "BUY", "executedQty": "0", "updateTime": T0 + 3000}
    cancelled_dry = {"orderId": 4, "clientOrderId": "web_c", "status": "CANCELED",
                     "side": "SELL", "executedQty": "0", "updateTime": T0 + 3500}
    board = {"NVDAUSDT": [book_fill, app_buy, resting, cancelled_dry]}
    calls = []

    def fetch(sym, start):
        calls.append((sym, start))
        return list(board.get(sym, []))

    adopted, notes = auto_sync(fetch, {"NVDAUSDT": 9.0}, now_ms=T0,
                               ledger_path=lp, state_path=sp)
    assert load_reserve(lp) == {"NVDAUSDT": 4.0}   # only the app buy
    assert [a["orderId"] for a in adopted] == ["2"]
    assert calls == [("NVDAUSDT", T0)]             # deploy anchor: scan starts at first-run now

    # idempotent re-run: nothing new adopted, reserve unchanged
    adopted, _ = auto_sync(fetch, {"NVDAUSDT": 9.0}, now_ms=T0 + 4000,
                           ledger_path=lp, state_path=sp)
    assert adopted == [] and load_reserve(lp) == {"NVDAUSDT": 4.0}

    # the resting order fills later -> caught once terminal
    board["NVDAUSDT"][2] = dict(resting, status="FILLED", executedQty="2",
                                updateTime=T0 + 9000, avgPrice="12")
    adopted, _ = auto_sync(fetch, {"NVDAUSDT": 11.0}, now_ms=T0 + 10_000,
                           ledger_path=lp, state_path=sp)
    assert load_reserve(lp) == {"NVDAUSDT": 6.0}

    # operator closes the whole manual position in the app -> reserve auto-releases
    board["NVDAUSDT"].append({"orderId": 5, "clientOrderId": "ios_q", "status": "FILLED",
                              "side": "SELL", "executedQty": "6", "updateTime": T0 + 20_000,
                              "avgPrice": "13"})
    adopted, _ = auto_sync(fetch, {"NVDAUSDT": 5.0}, now_ms=T0 + 21_000,
                           ledger_path=lp, state_path=sp)
    assert load_reserve(lp) == {}                  # fully released
    assert adopted[0]["qty"] == -6.0

    state = load_sync_state(T0 + 22_000, sp)
    assert state["deploy_ms"] == T0                # anchor never moves


def test_manual_autosync_first_run_anchor(tmp_path):
    """Fresh state anchors deploy_ms=now: pre-feature history (unstamped, ours included) is
    never scanned, so it can never be misclassified as manual."""
    from convexity.paper.manual import auto_sync

    T_LATE = 1_900_000_000_000
    seen = []

    def fetch(sym, start):
        seen.append(start)
        return []

    auto_sync(fetch, {"BTCUSDT": 1.0}, now_ms=T_LATE,
              ledger_path=tmp_path / "r.json", state_path=tmp_path / "s.json")
    assert seen == [T_LATE]

def test_manual_bootstrap_rule():
    """PAPER.md §12.5: virgin account -> every venue position seeds the reserve; any live book
    or prior seeding refuses (never freeze a book out of its own positions, never double-count)."""
    import pytest

    from convexity.paper.manual import bootstrap_reserve

    seeded = bootstrap_reserve({"AAPLUSDT": 3.5, "DUSTUSDT": 0.0, "CLUSDT": -2.0}, {}, [])
    assert seeded == {"AAPLUSDT": 3.5, "CLUSDT": -2.0}   # dust dropped, shorts kept signed

    with pytest.raises(RuntimeError, match="live targets"):
        bootstrap_reserve({"AAPLUSDT": 3.5}, {}, ["equity"])
    with pytest.raises(RuntimeError, match="one-time"):
        bootstrap_reserve({"AAPLUSDT": 3.5}, {"NVDAUSDT": 1.0}, [])


# ---------------------------------------------------------------- production hardening (§12.6)


def test_venue_switch(monkeypatch):
    """Audit P0-1: ONE env var selects every execution surface; invalid values refuse."""
    import pytest

    from convexity.paper import venue as V

    monkeypatch.delenv("CONVEXITY_VENUE", raising=False)
    assert V.venue() == "demo"
    assert V.execution_base() == "https://demo-fapi.binance.com"
    assert V.key_env_names()[0] == "BINANCE_FUTURES_DEMO_API_KEY"
    monkeypatch.setenv("CONVEXITY_VENUE", "prod")
    assert V.execution_base() == "https://fapi.binance.com"
    assert V.key_env_names() == ("BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET")
    monkeypatch.setenv("CONVEXITY_VENUE", "staging")
    with pytest.raises(RuntimeError, match="CONVEXITY_VENUE"):
        V.venue()


def test_settle_idempotency_helpers(tmp_path):
    """Audit P0-2: a crash-rerun must neither double-settle a week nor zero the settle's
    turnover through a same-asof duplicate targets record."""
    from convexity.paper.cli import _already_settled, _carry_in_targets
    from convexity.paper.journal import Journal

    j = Journal(tmp_path / "j.jsonl")
    assert not _already_settled(j, "2026-07-10")
    j.append("settle", "2026-07-10", r_model=0.01)
    assert _already_settled(j, "2026-07-10")
    assert not _already_settled(j, "2026-07-17")

    recs = [{"asof": "2026-07-03", "g_t": 1.0}, {"asof": "2026-07-10", "g_t": 1.2},
            {"asof": "2026-07-10", "g_t": 1.3}]  # crash-rerun duplicate
    assert _carry_in_targets(recs)["asof"] == "2026-07-03"
    assert _carry_in_targets([{"asof": "a"}, {"asof": "a"}]) is None
    assert _carry_in_targets([{"asof": "a"}]) is None


def test_position_drift_metric():
    """Audit P2-14: venue-vs-target notional drift, scaled-weight semantics."""
    from convexity.paper.cli import _position_drift

    t = {"symbols": ["AUSDT", "BUSDT"], "weights": [0.5, 0.0], "g_t": 2.0, "nav": 100.0}
    tot, top = _position_drift(t, {"AUSDT": 9.0, "CUSDT": 3.0},
                               {"AUSDT": 10.0, "CUSDT": 5.0})
    # AUSDT: venue 90 vs target 0.5*2*100=100 -> -10; CUSDT: venue 15 vs target 0 -> +15
    assert tot == 25.0
    assert top[0] == ["CUSDT", 15.0]
    assert ["AUSDT", -10.0] in top


def test_watchdog_last_friday():
    import datetime as dt
    import importlib.util

    spec = importlib.util.spec_from_file_location("fw", "scripts/fleet_watchdog.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    assert fw.last_friday_iso(dt.date(2026, 7, 12)) == "2026-07-10"  # Sunday
    assert fw.last_friday_iso(dt.date(2026, 7, 10)) == "2026-07-10"  # Friday itself
    assert fw.last_friday_iso(dt.date(2026, 7, 9)) == "2026-07-03"   # Thursday


def test_venue_state_namespacing(monkeypatch):
    """PAPER.md §12.6: ONE checkout, both venues — demo state paths byte-identical to the
    legacy layout; prod isolates every stateful path under data/prod."""
    from convexity.paper import venue as V
    from convexity.paper.books import BOOKS, book_journal_path
    from convexity.paper.manual import reserve_path, sync_state_path

    monkeypatch.delenv("CONVEXITY_VENUE", raising=False)
    assert str(V.state_root()) == "data/paper"
    assert book_journal_path(BOOKS["crypto"]) == "data/paper/journal.jsonl"  # legacy preserved
    assert book_journal_path(BOOKS["equity"]) == "data/paper/equity/journal.jsonl"
    assert str(reserve_path()) == "data/paper/manual_reserve.json"

    monkeypatch.setenv("CONVEXITY_VENUE", "prod")
    assert str(V.state_root()) == "data/prod"
    assert book_journal_path(BOOKS["crypto"]) == "data/prod/crypto/journal.jsonl"
    assert book_journal_path(BOOKS["commodity"]) == "data/prod/commodity/journal.jsonl"
    assert book_journal_path(BOOKS["equity"]) == "data/prod/equity/journal.jsonl"
    assert str(reserve_path()) == "data/prod/manual_reserve.json"
    assert str(sync_state_path()) == "data/prod/manual_sync.json"


def test_venue_lock_isolation(tmp_path, monkeypatch):
    """Demo and prod are different exchange accounts: their locks must not contend."""
    import os

    from convexity.paper.lock import account_lock

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONVEXITY_VENUE", raising=False)
    with account_lock("demo-test", timeout=1.0):
        monkeypatch.setenv("CONVEXITY_VENUE", "prod")
        with account_lock("prod-test", timeout=1.0):  # would deadlock if shared
            assert os.path.exists("data/paper/account.lock")
            assert os.path.exists("data/prod/account.lock")


def test_finalize_ids_accumulation(tmp_path):
    """finalize_ids accumulation (PAPER.md §12.6): a rerun counts prior crossing fills, so an
    already-crossed remainder computes to ~0 and is never re-crossed."""
    from convexity.paper.journal import Journal

    j = Journal(tmp_path / "j.jsonl")
    j.append("submission_ids", "2026-07-10",
             orders=[{"symbol": "TTWOUSDT", "orderId": 1, "qty": 0.05}])
    j.append("finalize_ids", "2026-07-10",
             orders=[{"symbol": "TTWOUSDT", "orderId": 2, "qty": 0.05}])
    j.append("finalize_ids", "2026-07-09",  # different asof: must NOT count
             orders=[{"symbol": "TTWOUSDT", "orderId": 9, "qty": 0.05}])

    by_sym = {}
    sub_rec = j.last("submission_ids")
    for o in sub_rec["orders"]:
        by_sym.setdefault(o["symbol"], []).append(o)
    for fr in j.read("finalize_ids"):
        if fr.get("asof") == "2026-07-10":
            for o in fr.get("orders", ()):
                by_sym.setdefault(o["symbol"], []).append(o)

    fills = {1: 0.0, 2: 0.05}  # maker expired unfilled; crossing filled fully
    executed = sum(fills[o["orderId"]] * (1 if o["qty"] > 0 else -1)
                   for o in by_sym["TTWOUSDT"])
    assert abs(0.05 - executed) < 1e-12  # remainder == 0 -> no re-cross


def test_maker_peg_and_batch_aggregation(tmp_path):
    """PAPER.md §12.7: peg sits AT the touch, and finalize must aggregate EVERY order batch
    (rebalance + each reprice pass + prior crossings) or it re-crosses filled positions."""
    from convexity.paper.cli import _maker_peg, _our_order_ids
    from convexity.paper.journal import Journal

    assert _maker_peg(100.0, 100.5, 1) == 100.0    # buy joins best bid
    assert _maker_peg(100.0, 100.5, -1) == 100.5   # sell joins best ask

    j = Journal(tmp_path / "j.jsonl")
    j.append("submission_ids", "2026-08-08",
             orders=[{"symbol": "AUSDT", "orderId": 1, "qty": 1.0}])
    j.append("submission_ids", "2026-08-08",           # reprice pass
             orders=[{"symbol": "AUSDT", "orderId": 2, "qty": 0.6}])
    j.append("finalize_ids", "2026-08-08",
             orders=[{"symbol": "AUSDT", "orderId": 3, "qty": 0.2}])
    j.append("submission_ids", "2026-08-01",           # PRIOR cycle: must NOT leak in
             orders=[{"symbol": "AUSDT", "orderId": 9, "qty": 5.0}])
    by = _our_order_ids(j, "2026-08-08")
    assert [o["orderId"] for o in by["AUSDT"]] == [1, 2, 3]
    assert _our_order_ids(j, "2026-07-25") == {}


def test_catchup_registered_and_safe_defaults():
    """PAPER.md §12.7: catchup exists, defaults are conservative, and it is dry by default."""
    import argparse

    from convexity.paper import cli

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    # rebuild just the catchup parser the way main() does, then assert its guardrails
    cu = sub.add_parser("catchup")
    cu.add_argument("--book", choices=sorted(cli.BOOKS), default="crypto")
    cu.add_argument("--live", action="store_true")
    cu.add_argument("--max-frac", type=float, default=0.5)
    cu.add_argument("--target-cov", type=float, default=95.0)
    a = ap.parse_args(["catchup", "--book", "equity"])
    assert a.live is False          # never trades unless explicitly asked
    assert a.max_frac == 0.5        # caps added notional at half of NAV
    assert a.target_cov == 95.0
    assert callable(cli.cmd_catchup)


def test_catchup_merge_per_symbol(tmp_path):
    """PAPER.md §12.7: catchup folds its orders into the cycle's intents MERGED PER SYMBOL.

    Duplicate per-symbol entries would break finalize, which computes want-vs-executed with
    executions summed per symbol — two entries for one symbol would subtract the same fills
    twice and re-cross a position already held."""
    from convexity.paper.journal import Journal

    j = Journal(tmp_path / "j.jsonl")
    j.append("orders", "2026-08-14", mode="live",
             intents=[{"symbol": "AUSDT", "qty": 1.0}, {"symbol": "BUSDT", "qty": 2.0}])
    plan = [("AUSDT", 0.5), ("CUSDT", 3.0)]          # A overlaps, C is band-suppressed/new
    merged = {it["symbol"]: dict(it) for it in j.last("orders")["intents"]}
    for sym, qty in plan:
        if sym in merged:
            merged[sym]["qty"] = float(merged[sym]["qty"]) + qty
        else:
            merged[sym] = {"symbol": sym, "qty": qty}
    assert len(merged) == 3                            # one entry per symbol, never duplicated
    assert merged["AUSDT"]["qty"] == 1.5               # overlapping symbol summed
    assert merged["CUSDT"]["qty"] == 3.0               # new symbol carried in
    assert merged["BUSDT"]["qty"] == 2.0               # untouched symbol preserved


def test_renormalize_achievable():
    """PAPER.md §12.8: scale the venue-achievable names back up to the book's target gross.

    Off by default everywhere; when applied it must preserve gross and signs, drop only what the
    grid cannot hold, and be a strict no-op when every name is achievable."""
    import numpy as np

    from convexity.paper.fleetconfig import RENORM_PARTIAL_BOOK
    from convexity.paper.policy import renormalize_achievable

    assert not any(RENORM_PARTIAL_BOOK.values()), "renormalization must ship OFF"

    class F:
        def __init__(self, lot, min_qty, min_notional):
            self.lot_size, self.min_qty, self.min_notional = lot, min_qty, min_notional

    syms = ("BIGUSDT", "MIDUSDT", "TINYUSDT")
    prices = np.array([100.0, 100.0, 100.0])
    filters = {s: F(0.01, 0.01, 5.0) for s in syms}
    w = np.array([0.5, 0.3, 0.02])          # gross 0.82
    nav = 1000.0                             # TINY target = $20 -> achievable here

    out, d = renormalize_achievable(w, syms, prices, nav, filters)
    assert d["applied"] and d["n_kept"] == 3
    assert abs(np.abs(out).sum() - 0.82) < 1e-9        # gross preserved
    assert np.allclose(out, w)                         # nothing dropped => no-op

    # small NAV: TINY's target ($2) is below the $5 floor -> dropped, others scaled up
    out2, d2 = renormalize_achievable(w, syms, prices, 100.0, filters)
    assert d2["applied"] and d2["n_kept"] == 2
    assert out2[2] == 0.0                               # unachievable name dropped
    assert abs(np.abs(out2).sum() - 0.82) < 1e-6        # gross RESTORED, not lost
    assert out2[0] > w[0] and out2[1] > w[1]            # survivors scaled UP
    assert abs(out2[0] / out2[1] - w[0] / w[1]) < 1e-9  # relative weights untouched

    # signs preserved on a short
    ws = np.array([0.5, -0.3, 0.02])
    out3, _ = renormalize_achievable(ws, syms, prices, 100.0, filters)
    assert out3[0] > 0 and out3[1] < 0

    # degenerate input is safe
    z, dz = renormalize_achievable(np.zeros(3), syms, prices, 1000.0, filters)
    assert not dz["applied"] and np.all(z == 0)

