"""Binance USDT-M DEMO gateway via the NautilusTrader adapter (PAPER.md §8) — EXPERIMENTAL.

Reuses Nautilus's own Binance HTTP plumbing (``BinanceHttpClient`` + ``BinanceFuturesAccountHttpAPI``
— never a hand-rolled signer, CLAUDE.md reuse rule), bound against the probed 1.229 surfaces:
``get_http_base_url(USDT_FUTURES, DEMO, is_us=False) -> https://demo-fapi.binance.com``,
``query_futures_account_info``, ``new_order`` (LIMIT+GTX = post-only maker; MARKET = the crossing
fallback), ``cancel_all_open_orders``. Weekly batch sessions need nothing else.

EXPERIMENTAL until the first supervised ``--live`` session (PAPER.md §8): every call is wrapped so
an adapter drift fails LOUDLY, and dry-run needs none of this module. Keys come from the
environment (``BINANCE_FUTURES_DEMO_API_KEY``/``_SECRET``; systemd injects ``.env``). External
flows (deposits/withdrawals) are journaled via the ``flow`` CLI command — the demo income API is
not part of this v1 surface (PAPER.md §9 deferral).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from convexity.paper.policy import OrderIntent


class GatewayError(RuntimeError):
    """Loud wrapper: the EXPERIMENTAL Nautilus binding failed — fix under supervision, never trade on."""


def _require_keys() -> tuple[str, str]:
    from convexity.paper.venue import key_env_names, venue

    k_env, s_env = key_env_names()
    key = os.environ.get(k_env, "")
    sec = os.environ.get(s_env, "")
    if not key or not sec:
        raise GatewayError(
            f"{k_env}/_SECRET not set — --live on venue={venue()!r} needs these keys "
            f"(.env); dry-run needs no keys")
    return key, sec


def _build_api():
    try:
        from nautilus_trader.adapters.binance.common.enums import (
            BinanceAccountType,
        )
        from nautilus_trader.adapters.binance.common.urls import get_http_base_url
        from nautilus_trader.adapters.binance.futures.http.account import (
            BinanceFuturesAccountHttpAPI,
        )
        from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
        from nautilus_trader.common.component import LiveClock
    except Exception as e:  # pragma: no cover - environment-dependent
        raise GatewayError(f"nautilus_trader Binance adapter import failed: {e!r}") from e

    from convexity.paper.venue import nautilus_environment

    key, sec = _require_keys()
    clock = LiveClock()
    # venue selection through the ONE switch (PAPER.md §12.6) — never a hardcoded environment
    base = get_http_base_url(
        BinanceAccountType.USDT_FUTURES, nautilus_environment(), False)
    client = BinanceHttpClient(clock=clock, api_key=key, api_secret=sec, base_url=base)
    api = BinanceFuturesAccountHttpAPI(
        client=client, clock=clock, account_type=BinanceAccountType.USDT_FUTURES)
    return client, api, clock


def _run_read(make_coro, attempts: int = 4, backoff: float = 2.0):
    """Retry wrapper for READ-ONLY (GET) venue calls — live catch 2026-07-27: demo-fapi
    returned nginx 502 across a >40min window and failed three daily recons.

    ONLY safe for reads. Order submission is NEVER retried here: a re-sent POST whose first
    attempt actually landed would double-fill. ``make_coro`` is a factory because a coroutine
    can only be awaited once."""
    import time as _t

    last = None
    for i in range(attempts):
        try:
            return _run(make_coro())
        except GatewayError as e:
            last = e
            if i < attempts - 1:
                _t.sleep(backoff * (2 ** i))
    raise last


def _run(coro):
    from convexity.paper.ratelimit import pace

    pace("authed")  # shared API pacing (PAPER.md §12.2)
    try:
        return asyncio.run(coro)
    except GatewayError:
        raise
    except Exception as e:  # pragma: no cover - live-path only
        raise GatewayError(f"gateway call failed (adapter drift? supervise): {e!r}") from e


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float  # totalMarginBalance (wallet + unrealized PnL), USDT
    positions: dict[str, float]  # symbol -> signed position qty (base units)
    unrealized: dict[str, float]  # symbol -> unrealized PnL (quote units), open positions only
    # Margin-guardrail context (PAPER.md §12.2); 0.0 when unavailable.
    maint_margin: float = 0.0  # totalMaintMargin (quote units)
    initial_margin: float = 0.0  # totalInitialMargin (quote units)


def account_snapshot() -> AccountSnapshot:
    """Equity + open positions from the demo account (EXPERIMENTAL).

    ``unrealized`` (per symbol, from position_risk) lets a BOOK sum the venue-truth unrealized
    P&L over ITS OWN symbol domain (PAPER.md §11 book_recon) — the shared account's aggregate
    equity blends every book and cannot attribute. ``maint_margin``/``initial_margin`` feed the
    cross-margin software guardrails (PAPER.md §12.2).
    """
    _client, api, _clock = _build_api()

    async def go() -> AccountSnapshot:
        info = await api.query_futures_account_info(recv_window="5000")
        # V3 account info returns positions=None (learned live 2026-07-03); position_risk is the
        # authoritative per-symbol position surface.
        risk = await api.query_futures_position_risk(recv_window="5000")
        positions: dict[str, float] = {}
        unrealized: dict[str, float] = {}
        for p in risk or ():
            amt = float(p.positionAmt)
            if amt != 0.0:
                positions[str(p.symbol)] = amt
                unrealized[str(p.symbol)] = float(getattr(p, "unRealizedProfit", 0.0) or 0.0)
        return AccountSnapshot(equity=float(info.totalMarginBalance), positions=positions,
                               unrealized=unrealized,
                               maint_margin=float(getattr(info, "totalMaintMargin", 0.0) or 0.0),
                               initial_margin=float(
                                   getattr(info, "totalInitialMargin", 0.0) or 0.0))

    return _run_read(go)


def set_leverage(symbol: str, leverage: int) -> dict:
    """Set the symbol's INTEGER leverage dial (PAPER.md §12.2) — once at book
    boot, never per-trade. Idempotent: re-setting the same value is a no-op on the venue.
    Official SIGNED endpoint ``POST /fapi/v1/leverage``; reuses the Nautilus client's own
    ``sign_request`` (never a hand-rolled signer, the sign_tradfi_agreement precedent)."""
    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> dict:
        resp = await client.sign_request(
            http_method=HttpMethod.POST,
            url_path="/fapi/v1/leverage",
            payload={"symbol": symbol, "leverage": str(int(leverage)),
                     "timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"},
        )
        return {"symbol": symbol, "leverage": int(leverage), "resp": str(resp)[:120]}

    return _run(go())


def submit_intent(intent: OrderIntent) -> dict:
    """Submit one OrderIntent (EXPERIMENTAL): POST_ONLY_MAKER -> LIMIT+GTX at ref_price;
    everything else crosses as MARKET (the finalize fallback)."""
    from nautilus_trader.adapters.binance.common.enums import (
        BinanceOrderSide,
        BinanceOrderType,
        BinanceTimeInForce,
    )

    import uuid

    from convexity.paper.manual import OUR_ORDER_PREFIX

    from convexity.paper.ratelimit import pace

    _client, api, _clock = _build_api()
    # order-placement cap (PAPER.md §12.6, prod -1015 live catch 2026-07-11): orders pace on
    # their OWN class (20/10s account cap) on top of the general authed floor in _run
    pace("order")
    side = BinanceOrderSide.BUY if intent.qty > 0 else BinanceOrderSide.SELL
    # fleet stamp (PAPER.md §12.5): everything the gateway submits carries this prefix, so any
    # fill WITHOUT it is the operator's manual order by construction (auto-reserve attribution)
    cid = f"{OUR_ORDER_PREFIX}{uuid.uuid4().hex[:20]}"

    async def go() -> dict:
        if intent.tactic == "POST_ONLY_MAKER":
            resp = await api.new_order(
                symbol=intent.symbol, side=side, order_type=BinanceOrderType.LIMIT,
                time_in_force=BinanceTimeInForce.GTX,  # post-only: cancels if it would cross
                reduce_only=("true" if intent.reduce_only else None),
                quantity=intent.qty_str or str(abs(intent.qty)),
                price=intent.price_str or str(intent.ref_price),
                new_client_order_id=cid)
        else:
            resp = await api.new_order(
                symbol=intent.symbol, side=side, order_type=BinanceOrderType.MARKET,
                reduce_only=("true" if intent.reduce_only else None),
                quantity=intent.qty_str or str(abs(intent.qty)),
                new_client_order_id=cid)
        return {"symbol": intent.symbol, "orderId": getattr(resp, "orderId", None),
                "status": str(getattr(resp, "status", "")), "clientOrderId": cid}

    return _run(go())


def submit_bbo(symbol: str, qty: float, qty_str: str, price_match: str,
               tif: str = "GTX", reduce_only: bool = False) -> dict:
    """LIMIT order priced SERVER-SIDE from the venue's own book (Binance BBO/``priceMatch``).

    ``price_match``: QUEUE / QUEUE_5|10|20 (own side, maker) or OPPONENT / OPPONENT_5|10|20
    (across the spread, taker). No ``price`` is sent — the venue reads its own book at
    placement, which removes the fetch->submit latency race that made -5022 rejections a
    problem and previously forced the peg one tick behind the touch (verified accepted with GTX
    on crypto, equity and commodity perps, 2026-08-06). Not retried on failure (POST)."""
    import uuid

    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    from convexity.paper.manual import OUR_ORDER_PREFIX

    client, _api, clock = _build_api()
    cid = f"{OUR_ORDER_PREFIX}{uuid.uuid4().hex[:20]}"
    payload = {
        "symbol": symbol, "side": "BUY" if qty > 0 else "SELL", "type": "LIMIT",
        "timeInForce": tif, "quantity": qty_str, "priceMatch": price_match,
        "newClientOrderId": cid, "timestamp": str(clock.timestamp_ms()), "recvWindow": "5000",
    }
    if reduce_only:
        payload["reduceOnly"] = "true"

    async def go() -> dict:
        import json as _json

        from convexity.paper.ratelimit import pace

        pace("order")
        resp = await client.sign_request(http_method=HttpMethod.POST,
                                        url_path="/fapi/v1/order", payload=payload)
        body = resp.body if hasattr(resp, "body") else resp
        d = _json.loads(body) if body else {}
        return {"symbol": symbol, "orderId": d.get("orderId"), "status": str(d.get("status", "")),
                "clientOrderId": cid, "price": d.get("price")}

    return _run(go())


def cancel_open_orders(symbol: str) -> bool:
    """Cancel all open orders for ``symbol`` (the finalize pass, EXPERIMENTAL).

    Returns ``False`` — never raises — when the venue reports the symbol is not in a cancellable
    status (-1122 "Invalid symbol status": delisted, halted, or settling). One such symbol must
    NEVER abort the finalize: the caller cancels in a loop before crossing anything, so a raise
    here leaves every OTHER name's unfilled maker remainder uncrossed for the week. That is
    exactly what happened on 2026-08-15 (demo crypto, NFPUSDT — listed on demo, delisted on prod).
    Every other error still raises: this tolerates a known venue state, not failures in general.
    """
    _client, api, _clock = _build_api()

    async def go() -> None:
        await api.cancel_all_open_orders(symbol=symbol)

    try:
        _run(go())
    except GatewayError as e:
        if "-1122" in str(e) or "Invalid symbol status" in str(e):
            return False
        raise
    return True


def fetch_recent_orders(symbol: str, start_ms: int) -> list[dict]:
    """All of ``symbol``'s orders placed after ``start_ms`` (SIGNED ``/fapi/v1/allOrders``) —
    the manual-fill attribution surface (PAPER.md §12.5). PAGINATED by orderId cursor
    (audit P2-8): >1000 orders in one sync gap no longer truncates adoption; auto_sync's
    14-day lookback covers orders that rested before going terminal (allOrders filters by
    placement time)."""
    import json as _json

    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> list[dict]:
        out: list[dict] = []
        cursor: int | None = None
        for _page in range(10):  # 10k orders/symbol/sync — far beyond any real gap
            payload = {"symbol": symbol, "limit": "1000",
                       "timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"}
            if cursor is None:
                payload["startTime"] = str(int(start_ms))
            else:
                payload["orderId"] = str(cursor)  # venue: orders with orderId >= this
            resp = await client.sign_request(
                http_method=HttpMethod.GET, url_path="/fapi/v1/allOrders", payload=payload)
            body = resp.body if hasattr(resp, "body") else resp
            batch = _json.loads(body)
            out.extend(batch)
            if len(batch) < 1000:
                break
            cursor = int(batch[-1]["orderId"]) + 1
        return out

    return _run_read(go)


def fetch_order(symbol: str, order_id: int) -> dict:
    """One order's venue state incl. ``executedQty`` (SIGNED ``/fapi/v1/order``) — the finalize
    v2 exact-remainder surface (PAPER.md §12.6): remainders are computed from OUR journaled
    orderIds' actual fills, never inferred from position snapshots."""
    import json as _json

    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> dict:
        resp = await client.sign_request(
            http_method=HttpMethod.GET,
            url_path="/fapi/v1/order",
            payload={"symbol": symbol, "orderId": str(int(order_id)),
                     "timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"},
        )
        body = resp.body if hasattr(resp, "body") else resp
        return _json.loads(body)

    return _run_read(go)


def fetch_open_orders() -> list[dict]:
    """All resting orders on the account (SIGNED ``/fapi/v1/openOrders``, no symbol filter)."""
    import json as _json

    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> list[dict]:
        resp = await client.sign_request(
            http_method=HttpMethod.GET, url_path="/fapi/v1/openOrders",
            payload={"timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"})
        body = resp.body if hasattr(resp, "body") else resp
        return _json.loads(body)

    return _run_read(go)


def fetch_user_trades(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Our executed trades for ``symbol`` in a window (SIGNED ``/fapi/v1/userTrades``) — the TCA
    surface: price, qty, commission, and the maker flag per fill."""
    import json as _json

    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> list[dict]:
        resp = await client.sign_request(
            http_method=HttpMethod.GET, url_path="/fapi/v1/userTrades",
            payload={"symbol": symbol, "startTime": str(int(start_ms)),
                     "endTime": str(int(end_ms)), "limit": "1000",
                     "timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"})
        body = resp.body if hasattr(resp, "body") else resp
        return _json.loads(body)

    return _run_read(go)


def sign_tradfi_agreement() -> str:
    """Sign the venue's TradFi-Perps agreement for this account (one-time; PAPER.md §11.5).

    Any TradFi-perp order draws ``-4411 "Please sign TradFi-Perps agreement contract fapi"``
    until the account has accepted the agreement — signable via the official SIGNED endpoint
    ``POST /fapi/v1/stock/contract`` (USER_DATA, weight 50, response "SUCCESS"). Reuses the
    Nautilus client's own ``sign_request`` (never a hand-rolled signer). Idempotent to re-run.
    """
    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()

    async def go() -> str:
        resp = await client.sign_request(
            http_method=HttpMethod.POST,
            url_path="/fapi/v1/stock/contract",
            # the raw sign_request does NOT inject timestamp (the endpoint classes normally do)
            payload={"timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"},
        )
        return str(resp)

    try:
        return _run(go())
    except GatewayError as e:
        # the venue answers HTTP 200 with the literal body "SUCCESS" (documented) — Nautilus's
        # JSON parser raises on it; that IS the success path (verified live 2026-07-06).
        if "SUCCESS" in str(e):
            return "SUCCESS"
        raise


def account_assets() -> dict[str, float]:
    """Per-asset margin balances (EXPERIMENTAL) — feeds the USDT->USDC collateral waterfall."""
    _client, api, _clock = _build_api()

    async def go() -> dict[str, float]:
        info = await api.query_futures_account_info(recv_window="5000")
        out: dict[str, float] = {}
        for a in getattr(info, "assets", None) or ():
            bal = float(a.marginBalance)
            if bal != 0.0:
                out[str(a.asset)] = bal
        return out

    return _run_read(go)
