"""The venue gateway, faked at the transport boundary.

``gateway.py`` is the code that actually talks to Binance, so it cannot be tested by substituting
its own functions — those ARE the thing under test. ``_build_api()`` is the seam instead: replacing
it leaves all the real logic running (response parsing, error mapping, retry policy, the -1122
tolerance) and fakes only the wire.

The property most worth locking down here is the retry asymmetry. Reads retry, because demo-fapi
returned nginx 502s across a 40-minute window and failed three daily recons. Order submissions
must NEVER retry, because a re-sent POST whose first attempt actually landed double-fills.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from convexity.paper import gateway
from convexity.paper.gateway import GatewayError


class FakeApi:
    """Async stand-in for BinanceFuturesAccountHttpAPI."""

    def __init__(self, info=None, risk=None, raises=None):
        self._info = info
        self._risk = risk or []
        self._raises = raises
        self.calls: list[str] = []

    async def query_futures_account_info(self, **kw):
        self.calls.append("account_info")
        if self._raises:
            raise self._raises
        return self._info

    async def query_futures_position_risk(self, **kw):
        self.calls.append("position_risk")
        if self._raises:
            raise self._raises
        return self._risk

    async def cancel_all_open_orders(self, symbol=None, **kw):
        self.calls.append(f"cancel:{symbol}")
        if self._raises:
            raise self._raises
        return {"code": 200}


def install(monkeypatch, api) -> None:
    monkeypatch.setattr(gateway, "_build_api", lambda: (object(), api, object()))
    # the module paces every call; real sleeps would only slow the suite
    monkeypatch.setattr("convexity.paper.ratelimit.pace", lambda kind="authed": None)


def info(margin_balance=1000.0, maint=None, initial=None):
    ns = SimpleNamespace(totalMarginBalance=margin_balance)
    if maint is not None:
        ns.totalMaintMargin = maint
    if initial is not None:
        ns.totalInitialMargin = initial
    return ns


def pos(symbol, amt, upnl=0.0):
    return SimpleNamespace(symbol=symbol, positionAmt=str(amt), unRealizedProfit=str(upnl))


# ---------------------------------------------------------------- account_snapshot
def test_snapshot_reads_equity_and_open_positions(monkeypatch):
    api = FakeApi(info(2500.0, maint=100.0, initial=400.0),
                  [pos("BTCUSDT", 0.5, 12.0), pos("ETHUSDT", -2.0, -3.0)])
    install(monkeypatch, api)
    snap = gateway.account_snapshot()
    assert snap.equity == 2500.0
    assert snap.positions == {"BTCUSDT": 0.5, "ETHUSDT": -2.0}
    assert snap.unrealized == {"BTCUSDT": 12.0, "ETHUSDT": -3.0}
    assert snap.maint_margin == 100.0
    assert snap.initial_margin == 400.0


def test_snapshot_drops_flat_symbols(monkeypatch):
    """A zero positionAmt row is not a position — carrying it would put phantom names into
    every book's recon."""
    install(monkeypatch, FakeApi(info(), [pos("BTCUSDT", 0.0), pos("ETHUSDT", 1.5)]))
    snap = gateway.account_snapshot()
    assert snap.positions == {"ETHUSDT": 1.5}


def test_snapshot_survives_a_null_position_risk(monkeypatch):
    """V3 account info returns positions=None (learned live 2026-07-03)."""
    install(monkeypatch, FakeApi(info(), None))
    assert gateway.account_snapshot().positions == {}


def test_snapshot_defaults_margin_fields_when_the_venue_omits_them(monkeypatch):
    """The margin guardrails must degrade to 0.0, not crash a live session on a missing field."""
    install(monkeypatch, FakeApi(info(900.0), [pos("BTCUSDT", 1.0)]))
    snap = gateway.account_snapshot()
    assert snap.maint_margin == 0.0
    assert snap.initial_margin == 0.0


# ---------------------------------------------------------------- error mapping
def test_an_arbitrary_adapter_error_becomes_a_GatewayError(monkeypatch):
    """Adapter drift must surface as our own typed failure, loudly."""
    install(monkeypatch, FakeApi(raises=RuntimeError("adapter drift")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(GatewayError):
        gateway.account_snapshot()


# ---------------------------------------------------------------- retry policy
def test_reads_retry_and_then_succeed(monkeypatch):
    """The 2026-07-27 outage: demo-fapi 502s across 40 minutes failed three daily recons."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GatewayError("502 bad gateway")
        return "ok"

    monkeypatch.setattr(gateway, "_run", lambda coro: flaky())
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert gateway._run_read(lambda: None) == "ok"
    assert calls["n"] == 3


def test_reads_give_up_after_the_attempt_budget_and_raise_the_last_error(monkeypatch):
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise GatewayError(f"attempt {calls['n']}")

    monkeypatch.setattr(gateway, "_run", lambda coro: always_fail())
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(GatewayError, match="attempt 4"):
        gateway._run_read(lambda: None)
    assert calls["n"] == 4


def test_the_coro_factory_is_rebuilt_each_attempt(monkeypatch):
    """A coroutine can only be awaited once — retrying a stored coroutine would raise instead of
    retrying, which is why _run_read takes a FACTORY."""
    built = {"n": 0}

    def factory():
        built["n"] += 1
        return f"coro{built['n']}"

    seen = []

    def run(c):
        seen.append(c)
        if len(seen) < 2:
            raise GatewayError("transient")
        return c

    monkeypatch.setattr(gateway, "_run", run)
    monkeypatch.setattr("time.sleep", lambda s: None)
    gateway._run_read(factory)
    assert built["n"] == 2
    assert seen == ["coro1", "coro2"]


def test_order_submission_is_not_wrapped_in_the_read_retry():
    """A re-sent POST whose first attempt actually landed would double-fill. submit_bbo and
    submit_intent must call _run directly, never _run_read."""
    import inspect

    for fn in (gateway.submit_bbo, gateway.submit_intent):
        src = inspect.getsource(fn)
        assert "_run_read" not in src, f"{fn.__name__} must never retry a POST"
        assert "_run(" in src


# ---------------------------------------------------------------- cancel tolerance
def test_cancel_returns_true_on_success(monkeypatch):
    api = FakeApi()
    install(monkeypatch, api)
    assert gateway.cancel_open_orders("BTCUSDT") is True
    assert "cancel:BTCUSDT" in api.calls


def test_cancel_of_a_non_TRADING_symbol_returns_false_instead_of_raising(monkeypatch):
    """REGRESSION (2026-08-15): -1122 raised, and the unhandled exception aborted the whole
    finalize before a single remainder was crossed."""
    install(monkeypatch, FakeApi(raises=RuntimeError("{'code': -1122, 'msg': 'Invalid symbol status.'}")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert gateway.cancel_open_orders("ICPUSDT") is False


def test_cancel_still_raises_on_any_other_failure(monkeypatch):
    """The tolerance is for ONE known venue state, not for failures in general."""
    install(monkeypatch, FakeApi(raises=RuntimeError("{'code': -1021, 'msg': 'timestamp'}")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(GatewayError):
        gateway.cancel_open_orders("BTCUSDT")
