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
    key = os.environ.get("BINANCE_FUTURES_DEMO_API_KEY", "")
    sec = os.environ.get("BINANCE_FUTURES_DEMO_API_SECRET", "")
    if not key or not sec:
        raise GatewayError(
            "BINANCE_FUTURES_DEMO_API_KEY/_SECRET not set — --live needs the demo keys "
            "(.env, see .env.example); dry-run needs no keys"
        )
    return key, sec


def _build_api():
    try:
        from nautilus_trader.adapters.binance.common.enums import (
            BinanceAccountType,
            BinanceEnvironment,
        )
        from nautilus_trader.adapters.binance.common.urls import get_http_base_url
        from nautilus_trader.adapters.binance.futures.http.account import (
            BinanceFuturesAccountHttpAPI,
        )
        from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
        from nautilus_trader.common.component import LiveClock
    except Exception as e:  # pragma: no cover - environment-dependent
        raise GatewayError(f"nautilus_trader Binance adapter import failed: {e!r}") from e

    key, sec = _require_keys()
    clock = LiveClock()
    base = get_http_base_url(
        BinanceAccountType.USDT_FUTURES, BinanceEnvironment.DEMO, False)
    client = BinanceHttpClient(clock=clock, api_key=key, api_secret=sec, base_url=base)
    api = BinanceFuturesAccountHttpAPI(
        client=client, clock=clock, account_type=BinanceAccountType.USDT_FUTURES)
    return client, api, clock


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

    return _run(go())


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

    _client, api, _clock = _build_api()
    side = BinanceOrderSide.BUY if intent.qty > 0 else BinanceOrderSide.SELL

    async def go() -> dict:
        if intent.tactic == "POST_ONLY_MAKER":
            resp = await api.new_order(
                symbol=intent.symbol, side=side, order_type=BinanceOrderType.LIMIT,
                time_in_force=BinanceTimeInForce.GTX,  # post-only: cancels if it would cross
                reduce_only=("true" if intent.reduce_only else None),
                quantity=intent.qty_str or str(abs(intent.qty)),
                price=intent.price_str or str(intent.ref_price))
        else:
            resp = await api.new_order(
                symbol=intent.symbol, side=side, order_type=BinanceOrderType.MARKET,
                reduce_only=("true" if intent.reduce_only else None),
                quantity=intent.qty_str or str(abs(intent.qty)))
        return {"symbol": intent.symbol, "orderId": getattr(resp, "orderId", None),
                "status": str(getattr(resp, "status", ""))}

    return _run(go())


def cancel_open_orders(symbol: str) -> None:
    """Cancel all open orders for ``symbol`` (the finalize pass, EXPERIMENTAL)."""
    _client, api, _clock = _build_api()

    async def go() -> None:
        await api.cancel_all_open_orders(symbol=symbol)

    _run(go())


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

    return _run(go())
