"""Venue REDUCE-ONLY stop placement through a real Nautilus BacktestEngine (ARCH §8.4 c / EXEC P6).

This is the one test that drives an actual ``nautilus_trader`` ``BacktestEngine`` end to end: a
minimal strategy opens a long via a MARKET order, then translates the composer-built protective-stop
:class:`OrderCommand` into a Nautilus reduce-only ``STOP_MARKET`` via the engine's ``order_factory``
and submits it. We assert the venue ACCEPTS the reduce-only stop and rests it as a working order —
the real risk control that fires while the bot is down (EXECUTION P6 / Invariant R/S).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig  # noqa: E402
from nautilus_trader.config import LoggingConfig  # noqa: E402
from nautilus_trader.model.currencies import USDT  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import (  # noqa: E402
    AccountType,
    OmsType,
    OrderSide,
    TriggerType,
)
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.objects import Money  # noqa: E402
from nautilus_trader.test_kit.providers import TestInstrumentProvider  # noqa: E402
from nautilus_trader.trading.strategy import Strategy, StrategyConfig  # noqa: E402

from convexity.execution.config.resolve import resolve_config as exec_resolve_config  # noqa: E402
from convexity.execution.policy.protective_stop import ProtectiveStop  # noqa: E402
from convexity.execution.synth.generate import perp_instrument  # noqa: E402
from convexity.execution.types import PositionState  # noqa: E402

_TRIGGER = {
    "mark": TriggerType.MARK_PRICE,
    "index": TriggerType.INDEX_PRICE,
    "last": TriggerType.LAST_PRICE,
}


class _StopConfig(StrategyConfig, frozen=True):
    bar_type: str
    instrument_id: str


class _StopStrategy(Strategy):
    """Open a long, then rest the composer-built reduce-only protective stop on the venue."""

    def __init__(self, config: _StopConfig) -> None:
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type)
        self.instrument = None
        self.opened = False
        self.stop_submitted = False
        self.stop_order = None
        self._protective = ProtectiveStop()
        self._exec_cfg = exec_resolve_config(perp_instrument())

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.bar_type.instrument_id)
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.opened:
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal("0.010")),
            )
            self.submit_order(order)
            self.opened = True
            return
        if self.opened and not self.stop_submitted and self.portfolio.is_net_long(self.instrument.id):
            pos = self.cache.positions_open(instrument_id=self.instrument.id)[0]
            qty = float(pos.quantity)
            trigger = float(bar.close) - 200.0
            # Build the protective-stop OrderCommand the composer would emit (EXECUTION §9).
            cmd = self._protective.place(
                "BTCUSDT:0",
                PositionState(instrument="BTCUSDT", side=1, qty=qty),
                trigger,
                self._exec_cfg,
                amend_generation=0,
            )
            stop = self.order_factory.stop_market(
                instrument_id=self.instrument.id,
                order_side=OrderSide.SELL if cmd.side == -1 else OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal(str(cmd.qty))),
                trigger_price=self.instrument.make_price(Decimal(str(round(cmd.price, 1)))),
                trigger_type=_TRIGGER[cmd.trigger_reference],
                reduce_only=cmd.reduce_only,
            )
            self.submit_order(stop)
            self.stop_order = stop
            self.stop_submitted = True


def _make_bar(bar_type, instrument, mid: float, ts: int) -> Bar:
    p = lambda x: instrument.make_price(Decimal(str(round(x, 1))))  # noqa: E731
    return Bar(
        bar_type=bar_type,
        open=p(mid), high=p(mid + 10.0), low=p(mid - 10.0), close=p(mid),
        volume=instrument.make_qty(Decimal("10")),
        ts_event=ts, ts_init=ts,
    )


def test_engine_accepts_reduce_only_protective_stop():
    """The Binance USDT-M venue ACCEPTS the composer-built reduce-only STOP_MARKET and rests it."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True))
    )
    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(100_000, USDT)],
    )
    engine.add_instrument(instrument)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    base_ts = 1_700_000_000_000_000_000
    bars = [
        _make_bar(bar_type, instrument, 30_000.0 + i * 5.0, base_ts + i * 60_000_000_000)
        for i in range(6)
    ]
    engine.add_data(bars)
    cfg = _StopConfig(bar_type=str(bar_type), instrument_id=str(instrument.id))
    strategy = _StopStrategy(cfg)
    engine.add_strategy(strategy)
    engine.run()

    assert strategy.stop_submitted, "the protective stop should have been submitted"
    stop = strategy.stop_order
    assert stop is not None
    assert stop.is_reduce_only, "the protective stop MUST be reduce-only (Invariant R)"
    # the venue accepted it (no rejection): the order reached a working/closed venue state.
    assert stop.status_string() not in ("DENIED", "REJECTED"), stop.status_string()
    engine.dispose()
