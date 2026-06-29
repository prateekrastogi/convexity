"""GENUINE end-to-end run of the live node over synthetic bars (ARCHITECTURE §8.4a / Inv N/PAR).

Unlike the rest of the integration suite (which drives the pure :class:`ConvexityComposer` directly),
this test runs the FULL ``run/backtest.py`` harness: a real ``nautilus_trader`` ``BacktestEngine``
hosting the live ``_ConvexityStrategy``, which subscribes to the engine's bars, projects the engine's
authoritative ``Cache``/``Portfolio`` into the pure pillar types, calls the composer, and submits the
returned orders back through the engine. We assert the strategy ACTUALLY traded:

  * orders were submitted and FILLED (entries reached the venue),
  * N simultaneous per-pair positions are managed (one open per pair),
  * a venue reduce-only protective stop rests for each open position (EXECUTION P6 / Inv S),
  * restart reconciliation is idempotent (a second on_start emits no duplicate stop),
  * the OPS DEGRADE (cancel-entries) and HALT (reduce-only flatten) paths are REACHABLE through the
    very composer the node ran (OPS §9.1b).
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from convexity.ops.types import EquityRef  # noqa: E402
from convexity.run.backtest import build_synthetic_composer, run_backtest  # noqa: E402
from convexity.strategy.projection import open_views_from_snapshot  # noqa: E402


def _open_positions(engine):
    return list(engine.cache.positions_open())


def test_node_runs_and_trades_n_positions():
    """The full node runs to completion and the strategy opens N managed per-pair positions."""
    result = run_backtest(n_bars=72, n_pairs=2, warmup_bars=16, fire_after_idx=20)
    strat = result.strategy
    engine = result.engine
    try:
        assert strat.bars_seen >= 72 * 2, "the strategy must have consumed every bar of both pairs"
        assert strat.decisions > 0, "the composer must have been driven on decision bars"
        assert strat.orders_submitted > 0, "the strategy must have submitted orders"
        assert strat.entries_submitted >= 2, "one entry per pair must have been submitted"
        # Fills actually happened: the engine has open positions.
        opens = _open_positions(engine)
        assert len(opens) == 2, f"expected 2 simultaneous per-pair positions, got {len(opens)}"
        # Orders were filled (every open position came from a filled entry).
        fills = [o for o in engine.cache.orders() if o.filled_qty > 0]
        assert fills, "at least one entry order must have filled"
    finally:
        engine.dispose()


def test_node_rests_a_reduce_only_protective_stop_per_open():
    """Every open position has a resting venue reduce-only STOP (the EXECUTION P6 floor)."""
    result = run_backtest(n_bars=72, n_pairs=2, warmup_bars=16, fire_after_idx=20)
    engine = result.engine
    try:
        assert result.strategy.stops_submitted >= 2, "a protective stop per pair must be submitted"
        for inst in engine.cache.instruments():
            opens = engine.cache.positions_open(instrument_id=inst.id)
            if not opens:
                continue
            resting = [
                o for o in engine.cache.orders_open(instrument_id=inst.id)
                if getattr(o, "is_reduce_only", False)
                and getattr(o, "trigger_price", None) is not None
            ]
            assert resting, f"open position on {inst.id} must have a resting reduce-only stop"
            assert all(o.is_reduce_only for o in resting)
    finally:
        engine.dispose()


def test_restart_reconciliation_is_idempotent():
    """Calling on_start a SECOND time against the same (already-STOP_OK) book places no new stop."""
    result = run_backtest(n_bars=72, n_pairs=2, warmup_bars=16, fire_after_idx=20)
    engine = result.engine
    strat = result.strategy
    try:
        before = strat.stops_submitted
        # The book already has one correct resting stop per open (STOP_OK); reconciling again must be
        # a no-op by construction (StopReconciler idempotency, Inv S).
        strat.on_start()
        assert strat.stops_submitted == before, "idempotent reconcile must place no duplicate stop"
    finally:
        engine.dispose()


def test_funding_subscription_is_counted_not_swallowed():
    """FRICTION #8: the funding subscription outcome is AUDITED (counted), not silently swallowed.

    The backtest charges funding from the fed lake; paper/live rely on the venue funding stream. A
    silently-swallowed subscribe-failure would make funding zero live while it is charged in backtest.
    The strategy now counts ``funding_subscribed`` / ``funding_subscribe_failed`` per perp at start so
    the parity gap is observable. Over a 2-perp run both perps must register a successful subscription
    and none must have silently failed."""
    result = run_backtest(n_bars=48, n_pairs=2, warmup_bars=16, fire_after_idx=20)
    strat = result.strategy
    engine = result.engine
    try:
        assert strat.funding_subscribed == 2, "both perps must register a funding subscription"
        assert strat.funding_subscribe_failed == 0, "no funding subscription may silently fail"
    finally:
        engine.dispose()


def test_warmup_history_primes_buffer_without_deciding():
    """FRICTION #7: ``on_historical_data`` primes the rolling buffer (cold-start parity) but never
    triggers a decision — the live/paper analogue of the backtest's bar preload.

    Drives a hosted strategy's ``on_historical_data`` with synthetic warmup bars and asserts the
    rolling buffer fills (``historical_bars_seen`` counts them) WITHOUT advancing the live-bar counter
    or running the composer. This is what lets paper/live reach the first decision at the same buffer
    depth the backtest preloads to, instead of idling for ``warmup_bars`` live minutes."""
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.objects import Price, Quantity

    # Stand up a node so a real hosted strategy exists, but DON'T run it — just drive on_historical_data.
    result = run_backtest(n_bars=20, n_pairs=1, warmup_bars=64, fire_after_idx=5)
    strat = result.strategy
    engine = result.engine
    try:
        iid = next(iter(strat._rolling))
        bt = BarType.from_str(f"{iid}-1-MINUTE-LAST-EXTERNAL")
        depth_before = len(strat._rolling[iid])
        decisions_before = strat.decisions
        live_bars_before = strat.bars_seen
        hist = [
            Bar(
                bar_type=bt,
                open=Price(30_000.0, 1), high=Price(30_010.0, 1),
                low=Price(29_990.0, 1), close=Price(30_000.0, 1),
                volume=Quantity(50.0, 3),
                ts_event=1_700_000_000_000_000_000 + j * 60_000_000_000,
                ts_init=1_700_000_000_000_000_000 + j * 60_000_000_000,
            )
            for j in range(30)
        ]
        strat.on_historical_data(hist)
        assert strat.historical_bars_seen == 30, "every historical bar must prime the buffer"
        assert len(strat._rolling[iid]) == depth_before + 30, "the rolling buffer must have grown"
        # Priming is NOT a decision: the composer was not driven and no live bar was counted.
        assert strat.decisions == decisions_before, "historical priming must not run the composer"
        assert strat.bars_seen == live_bars_before, "historical priming is not a live bar"
    finally:
        engine.dispose()


def test_degrade_and_halt_paths_reachable_through_the_run_composer():
    """The DEGRADE (cancel-entries) and HALT (reduce-only flatten) overlays are reachable.

    We drive the SAME composer the node ran against its realized open positions: a stale heartbeat
    yields DEGRADE (suppress the new entry), a catastrophic drawdown yields HALT (flatten every open
    reduce-only). This proves the OPS overlay the node hosts is wired, not dead code.
    """
    composer, breakout = build_synthetic_composer(mids={"BTCUSDT": 30_000.0}, fire_after_idx=0)
    breakout.idx = 5
    from convexity.run.backtest import _LevCfg
    from convexity.execution.config.resolve import resolve_config as exec_resolve_config
    from convexity.execution.synth.generate import perp_instrument
    from convexity.system.config.resolve import resolve_config as system_resolve_config
    from convexity.system.config.universe import load_universe_meta
    from convexity.strategy.projection import PositionSnapshot, RollingBars, micro_from_bar
    from convexity.system.types import BookState
    import numpy as np

    spec = perp_instrument("BTCUSDT")
    roll = RollingBars()
    for i in range(40):
        roll.push(ts=i * 60_000, o=30_000.0, h=30_010.0, low=29_990.0, c=30_000.0, v=50.0)
    bars = roll.as_bars()
    snap = PositionSnapshot(
        instrument="BTCUSDT", side=1, qty=0.005, entry_price=30_000.0, entry_idx=5,
        invalidation=29_800.0, generation=0,
    )
    open_views = open_views_from_snapshot([snap])
    book = BookState(equity=1000.0, peak_equity=1000.0, free_margin=800.0,
                     positions={}, cov=np.eye(1), governor_multiplier=1.0)

    def _common(open_positions):
        return dict(
            bars_by_instrument={"BTCUSDT": bars}, idx_by_instrument={"BTCUSDT": len(bars) - 1},
            specs={"BTCUSDT": spec}, book=book, open_positions=open_positions,
            selection=_proposed(), leverage_risk={"BTCUSDT": object()}, leverage_cfg=_LevCfg(),
            trend_cfg=object(), exec_cfg=exec_resolve_config(spec),
            micro_by_instrument={"BTCUSDT": micro_from_bar(spec, mid=30_000.0, bar_range=20.0,
                                                           volume=50.0)},
            sys_cfg=system_resolve_config(load_universe_meta([])), bar_ts=int(bars.ts[-1]),
        )

    # DEGRADE: a stale heartbeat ⇒ the fresh breakout entry (+ its stop) is suppressed, no flatten.
    degrade = composer.on_decision_bar(
        equity_ref=EquityRef(equity=1000.0, peak=1000.0), heartbeat_code=2.0, **_common({})
    )
    assert degrade.directive.state == "degrade"
    assert degrade.flatten is None
    assert "ops_cancel_entries" in degrade.reason_codes
    assert degrade.commands == (), "DEGRADE must suppress the new-signal entry orders + fresh stop"

    # HALT: 50% drawdown ⇒ flatten every open position reduce-only.
    halt = composer.on_decision_bar(
        equity_ref=EquityRef(equity=500.0, peak=1000.0), **_common(open_views)
    )
    assert halt.directive.state == "halt"
    assert halt.flatten is not None and halt.flatten.reduce_only
    assert halt.commands and all(c.reduce_only for c in halt.commands)
    assert "ops_halt" in halt.reason_codes


def _proposed():
    from convexity.system.types import ProposedPortfolio
    return ProposedPortfolio(keep=("BTCUSDT",), eligible=("BTCUSDT",),
                             risk_weights={"BTCUSDT": 0.05}, rank={"BTCUSDT": 0.0},
                             reason_codes=("synthetic",))
