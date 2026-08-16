"""An in-memory stand-in for the Binance USDT-M venue, so live CLI sessions can be tested.

``src/convexity/paper/cli.py`` is the live trading path — rebalance, finalize, catchup, reprice,
tca, recon — and it was the least-covered module in the repo (16%, 942 uncovered statements)
precisely because every command talks to an exchange. This gives those commands a venue they can
talk to that costs nothing and asserts everything.

DESIGN NOTES

*Faithful signatures.* Every method mirrors the real ``gateway``/``topup`` function it replaces,
argument for argument. A double whose shape has drifted from the thing it doubles tests nothing,
so the fixture patches by name onto the real modules and any signature change breaks loudly.

*Orders are stateful.* Submissions land in an open-order book that ``fetch_open_orders`` returns
and ``cancel_open_orders`` clears, and ``fill()`` converts them into user trades. That is what
lets finalize's remainder accounting and tca's shortfall maths be exercised for real rather than
mocked away.

*cwd isolation, not path patching.* Journals, the per-venue account lock and the manual-reserve
ledger are all resolved cwd-relative (``data/paper/...``, ``state_root()``). The fixture chdirs
into ``tmp_path``, which isolates every one of them in a single move — and guarantees a test can
never write to the real journals, which are the fleet's accounting truth.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pytest

# Matches gateway's stamp, so manual-position carve-out logic classifies these as fleet orders.
OUR_PREFIX = "cvx-"


@dataclass
class FakeOrder:
    order_id: int
    symbol: str
    qty: float          # signed; buy > 0
    price: float
    client_order_id: str
    status: str = "NEW"
    filled: float = 0.0


@dataclass
class FakeVenue:
    """Mutable venue state. Construct, mutate, assert."""

    equity: float = 10_000.0
    positions: dict[str, float] = field(default_factory=dict)
    unrealized: dict[str, float] = field(default_factory=dict)
    maint_margin: float = 0.0
    initial_margin: float = 0.0
    prices: dict[str, float] = field(default_factory=dict)
    spread_bp: float = 2.0
    filters: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    tradeable: set[str] | None = None
    assets: dict[str, float] = field(default_factory=lambda: {"USDT": 10_000.0})

    open_orders: dict[int, FakeOrder] = field(default_factory=dict)
    # Every order ever placed, including filled ones. A real venue remembers them, and
    # finalize's v2 remainder accounting asks for executedQty BY ORDER ID — if filled orders
    # vanish here, it reads 0 executed and crosses the same remainder a second time.
    all_orders: dict[int, FakeOrder] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    submitted: list[dict] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    leverage_set: dict[str, int] = field(default_factory=dict)
    # Symbols whose cancel raises the venue's "not TRADING" rejection (-1122).
    cancel_rejects: set[str] = field(default_factory=set)

    _ids: itertools.count = field(default_factory=lambda: itertools.count(1000))
    _now_ms: int = 1_700_000_000_000

    # ---------------------------------------------------------------- gateway surface
    def account_snapshot(self):
        from convexity.paper.gateway import AccountSnapshot

        return AccountSnapshot(
            equity=self.equity,
            positions=dict(self.positions),
            unrealized=dict(self.unrealized),
            maint_margin=self.maint_margin,
            initial_margin=self.initial_margin,
        )

    def account_assets(self) -> dict[str, float]:
        return dict(self.assets)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        self.leverage_set[symbol] = leverage
        return {"symbol": symbol, "leverage": leverage}

    def _record(self, symbol: str, qty: float, price: float) -> dict:
        oid = next(self._ids)
        cid = f"{OUR_PREFIX}{oid:020d}"
        o = FakeOrder(oid, symbol, qty, price, cid)
        self.open_orders[oid] = o
        self.all_orders[oid] = o
        rec = {"symbol": symbol, "orderId": oid, "status": "NEW", "clientOrderId": cid}
        self.submitted.append({**rec, "qty": qty, "price": price})
        return rec

    def submit_bbo(self, symbol, qty, qty_str, price_match, tif="GTX", reduce_only=False):
        bid, ask = self.book(symbol)
        crossing = price_match.startswith("OPPONENT")
        px = (ask if qty > 0 else bid) if crossing else (bid if qty > 0 else ask)
        rec = self._record(symbol, qty, px)
        # A CROSSING order fills on submission and moves the position. Leaving it resting made
        # this double unfaithful: finalize re-run saw the same remainder and crossed twice, which
        # looked like an idempotency bug in the product when it was a gap in the fake.
        if crossing:
            self.fill(rec["orderId"], maker=False)
        return rec

    def submit_intent(self, intent):
        rec = self._record(intent.symbol, intent.qty, intent.ref_price)
        # gateway: POST_ONLY_MAKER -> resting LIMIT+GTX; everything else crosses as MARKET.
        if getattr(intent, "tactic", "") != "POST_ONLY_MAKER":
            self.fill(rec["orderId"], maker=False)
        return rec

    def cancel_open_orders(self, symbol: str) -> bool:
        self.cancelled.append(symbol)
        if symbol in self.cancel_rejects:
            return False  # the real gateway maps -1122 to False
        for oid in [o for o, v in self.open_orders.items() if v.symbol == symbol]:
            del self.open_orders[oid]
        return True

    def fetch_open_orders(self) -> list[dict]:
        return [{"symbol": o.symbol, "orderId": o.order_id, "clientOrderId": o.client_order_id,
                 "side": "BUY" if o.qty > 0 else "SELL",
                 "origQty": abs(o.qty), "executedQty": o.filled, "price": o.price,
                 "status": o.status}
                for o in self.open_orders.values()]

    def fetch_order(self, symbol: str, order_id: int) -> dict:
        o = self.all_orders.get(order_id)
        if o is None:
            return {"symbol": symbol, "orderId": order_id, "status": "CANCELED",
                    "executedQty": "0", "origQty": "0"}
        return {"symbol": o.symbol, "orderId": o.order_id, "status": o.status,
                "executedQty": str(o.filled), "origQty": str(abs(o.qty)),
                "clientOrderId": o.client_order_id,
                "side": "BUY" if o.qty > 0 else "SELL"}

    def fetch_recent_orders(self, symbol: str, start_ms: int) -> list[dict]:
        return [self.fetch_order(o.symbol, o.order_id)
                for o in self.all_orders.values() if o.symbol == symbol]

    def fetch_user_trades(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        return [t for t in self.trades
                if t["symbol"] == symbol and start_ms <= t["time"] <= end_ms]

    # ---------------------------------------------------------------- topup surface
    def book(self, symbol: str) -> tuple[float, float]:
        p = self.prices.get(symbol, 100.0)
        half = p * (self.spread_bp / 2.0) / 1e4
        return p - half, p + half

    def fetch_ticker_prices(self) -> dict[str, float]:
        return dict(self.prices)

    def fetch_book_tickers(self) -> dict[str, tuple[float, float]]:
        return {s: self.book(s) for s in self.prices}

    def fetch_live_filters(self) -> dict[str, tuple[float, float, float, float]]:
        if self.filters:
            return dict(self.filters)
        return {s: (0.01, 0.001, 0.001, 5.0) for s in self.prices}

    def fetch_execution_venue_tradeable(self) -> frozenset[str]:
        return frozenset(self.tradeable if self.tradeable is not None else self.prices)

    # ---------------------------------------------------------------- test controls
    def fill(self, order_id: int, qty: float | None = None, maker: bool = True) -> None:
        """Fill an open order (fully by default) and move the position, as the venue would."""
        o = self.open_orders[order_id]
        q = abs(o.qty) if qty is None else abs(qty)
        o.filled += q
        signed = q if o.qty > 0 else -q
        self.positions[o.symbol] = self.positions.get(o.symbol, 0.0) + signed
        self.trades.append({"symbol": o.symbol, "price": o.price, "qty": q,
                            "side": "BUY" if o.qty > 0 else "SELL", "maker": maker,
                            "time": self._now_ms, "orderId": o.order_id})
        if o.filled >= abs(o.qty) - 1e-12:
            o.status = "FILLED"
            del self.open_orders[o.order_id]

    def fill_all(self, maker: bool = True) -> None:
        for oid in list(self.open_orders):
            self.fill(oid, maker=maker)


@pytest.fixture
def fake_venue(monkeypatch, tmp_path):
    """A FakeVenue wired over gateway/topup, with all fleet state isolated under tmp_path.

    The chdir is the important part: journals, the account lock and the manual-reserve ledger are
    all cwd-relative, so this single move guarantees a test cannot touch the real ones.
    """
    monkeypatch.chdir(tmp_path)
    v = FakeVenue()

    for name in ("account_snapshot", "account_assets", "set_leverage", "submit_bbo",
                 "submit_intent", "cancel_open_orders", "fetch_open_orders", "fetch_order",
                 "fetch_recent_orders", "fetch_user_trades"):
        monkeypatch.setattr(f"convexity.paper.gateway.{name}", getattr(v, name))
    for name in ("fetch_ticker_prices", "fetch_book_tickers", "fetch_live_filters",
                 "fetch_execution_venue_tradeable"):
        monkeypatch.setattr(f"convexity.paper.topup.{name}", getattr(v, name))

    # Live sessions cross-check that the Nautilus adapter resolved the same base URL we did.
    # There is no adapter here, and the check is not what these tests are about.
    monkeypatch.setattr("convexity.paper.venue.assert_consistent", lambda: None)
    return v
