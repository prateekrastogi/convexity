"""The gateway's SIGNED endpoints, faked at the transport boundary.

Everything here goes through ``client.sign_request``, so one fake client exercises the real
payload construction, response decoding and pagination. The headline is ``fetch_recent_orders``'s
cursor pagination — the P2-8 audit fix. Before it, a sync gap containing more than 1000 orders
TRUNCATED silently, and truncated adoption means the fleet mistakes the operator's manual
positions for its own.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from convexity.paper import gateway
from convexity.paper.gateway import GatewayError


class FakeResp:
    """The adapter sometimes returns an object with .body and sometimes the raw payload; the
    gateway handles both, so the double must be able to present both."""

    def __init__(self, body):
        self.body = body


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def sign_request(self, http_method=None, url_path=None, payload=None):
        self.requests.append({"method": str(http_method), "path": url_path,
                              "payload": dict(payload or {})})
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url_path}")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def install(monkeypatch, client=None, api=None):
    clock = SimpleNamespace(timestamp_ms=lambda: 1_700_000_000_000)
    monkeypatch.setattr(gateway, "_build_api", lambda: (client, api, clock))
    monkeypatch.setattr("convexity.paper.ratelimit.pace", lambda kind="authed": None)
    monkeypatch.setattr("time.sleep", lambda s: None)


def page(n, first_id=1):
    return json.dumps([{"orderId": first_id + i, "symbol": "BTCUSDT"} for i in range(n)])


# ---------------------------------------------------------------- pagination (audit P2-8)
def test_a_single_short_page_stops_immediately(monkeypatch):
    c = FakeClient([FakeResp(page(3))])
    install(monkeypatch, client=c)
    out = gateway.fetch_recent_orders("BTCUSDT", 1_000)
    assert len(out) == 3
    assert len(c.requests) == 1
    assert c.requests[0]["payload"]["startTime"] == "1000"


def test_a_full_page_triggers_another_request_with_an_orderId_cursor(monkeypatch):
    """The fix itself: >1000 orders in a sync gap must not truncate adoption."""
    c = FakeClient([FakeResp(page(1000, first_id=1)), FakeResp(page(2, first_id=1001))])
    install(monkeypatch, client=c)
    out = gateway.fetch_recent_orders("BTCUSDT", 1_000)
    assert len(out) == 1002
    assert len(c.requests) == 2
    # page 2 switches from a time window to an id cursor, one past the last id seen
    assert "startTime" not in c.requests[1]["payload"]
    assert c.requests[1]["payload"]["orderId"] == "1001"


def test_pagination_is_bounded_so_a_pathological_account_cannot_spin(monkeypatch):
    """10 pages x 1000 is far beyond any real gap; the loop must stop regardless."""
    c = FakeClient([FakeResp(page(1000, first_id=1 + 1000 * i)) for i in range(12)])
    install(monkeypatch, client=c)
    out = gateway.fetch_recent_orders("BTCUSDT", 0)
    assert len(c.requests) == 10
    assert len(out) == 10_000


def test_recent_orders_hits_the_allOrders_endpoint(monkeypatch):
    c = FakeClient([FakeResp(page(1))])
    install(monkeypatch, client=c)
    gateway.fetch_recent_orders("ETHUSDT", 5)
    assert c.requests[0]["path"] == "/fapi/v1/allOrders"
    assert c.requests[0]["payload"]["symbol"] == "ETHUSDT"


# ---------------------------------------------------------------- single-object reads
def test_fetch_order_decodes_the_body(monkeypatch):
    c = FakeClient([FakeResp(json.dumps({"orderId": 7, "executedQty": "0.5"}))])
    install(monkeypatch, client=c)
    d = gateway.fetch_order("BTCUSDT", 7)
    assert d["executedQty"] == "0.5"
    assert c.requests[0]["path"] == "/fapi/v1/order"
    assert c.requests[0]["payload"]["orderId"] == "7"


def test_a_raw_payload_without_a_body_attribute_is_accepted(monkeypatch):
    """The adapter is not consistent about wrapping; the gateway tolerates both shapes."""
    c = FakeClient([json.dumps({"orderId": 9})])
    install(monkeypatch, client=c)
    assert gateway.fetch_order("BTCUSDT", 9)["orderId"] == 9


def test_open_orders_are_fetched_across_the_whole_account(monkeypatch):
    c = FakeClient([FakeResp(json.dumps([{"symbol": "A"}, {"symbol": "B"}]))])
    install(monkeypatch, client=c)
    out = gateway.fetch_open_orders()
    assert len(out) == 2
    assert c.requests[0]["path"] == "/fapi/v1/openOrders"
    assert "symbol" not in c.requests[0]["payload"]   # no per-symbol filter


def test_user_trades_pass_the_time_window(monkeypatch):
    c = FakeClient([FakeResp(json.dumps([{"price": "10", "qty": "1", "maker": True}]))])
    install(monkeypatch, client=c)
    out = gateway.fetch_user_trades("BTCUSDT", 100, 200)
    assert out[0]["maker"] is True
    p = c.requests[0]["payload"]
    assert (p["startTime"], p["endTime"]) == ("100", "200")
    assert c.requests[0]["path"] == "/fapi/v1/userTrades"


# ---------------------------------------------------------------- leverage dial
def test_set_leverage_posts_an_integer_dial(monkeypatch):
    c = FakeClient([FakeResp(json.dumps({"leverage": 3}))])
    install(monkeypatch, client=c)
    out = gateway.set_leverage("BTCUSDT", 3)
    assert out["leverage"] == 3
    req = c.requests[0]
    assert req["path"] == "/fapi/v1/leverage"
    assert "POST" in req["method"].upper()
    assert req["payload"]["leverage"] == "3"


def test_set_leverage_coerces_to_int(monkeypatch):
    """The venue dial is an integer; sending 3.0 draws a rejection."""
    c = FakeClient([FakeResp(json.dumps({}))])
    install(monkeypatch, client=c)
    gateway.set_leverage("BTCUSDT", 3.9)
    assert c.requests[0]["payload"]["leverage"] == "3"


# ---------------------------------------------------------------- collateral waterfall
def test_account_assets_reports_nonzero_balances_only(monkeypatch):
    class Api:
        async def query_futures_account_info(self, **kw):
            return SimpleNamespace(assets=[
                SimpleNamespace(asset="USDT", marginBalance="1000.5"),
                SimpleNamespace(asset="USDC", marginBalance="0"),
                SimpleNamespace(asset="BTC", marginBalance="0.25"),
            ])

    install(monkeypatch, api=Api())
    out = gateway.account_assets()
    assert out == {"USDT": pytest.approx(1000.5), "BTC": pytest.approx(0.25)}
    assert "USDC" not in out


def test_account_assets_survives_a_venue_without_an_assets_field(monkeypatch):
    class Api:
        async def query_futures_account_info(self, **kw):
            return SimpleNamespace()

    install(monkeypatch, api=Api())
    assert gateway.account_assets() == {}


# ---------------------------------------------------------------- signed reads retry
def test_a_signed_read_retries_a_transient_failure(monkeypatch):
    c = FakeClient([GatewayError("502"), FakeResp(json.dumps({"orderId": 1}))])
    install(monkeypatch, client=c)
    assert gateway.fetch_order("BTCUSDT", 1)["orderId"] == 1
    assert len(c.requests) == 2
