"""REAL-DATA end-to-end run: the live node over the REAL data lake + REAL pillar engines (ARCH §8.4a).

Unlike ``test_backtest_end_to_end.py`` (synthetic klines + the deterministic ``_Synthetic*`` pillar
stubs), this drives the FULL ``run/backtest.run_backtest_real`` harness over a small deterministic
slice of the REAL ``data/stitched/{SYMBOL}.parquet`` lake, with the REAL Binance USDT-M perp
instruments built programmatically from the universe ``InstrumentSpec``, the REAL
``FundingRateUpdate`` feed from ``data/funding/{SYMBOL}.parquet``, and the REAL eight-pillar
composition (``run/real_deps.build_real_composer``: the validated BREAKOUT ``PipelineOrchestrator`` +
LEVERAGE ``LeverageEngineAPI`` + TREND ``TrendLifeEngine`` behind the composer's Protocol seams, plus
the live EXECUTION/OPS engines).

We assert:
  * the node runs to completion over the real bars (every bar of every pair consumed, composer driven);
  * the composer's pillars are the REAL engines, NOT the synthetic stubs (the crux of this task);
  * the funding feed is CONSUMED (the node ingested the realized funding stamps, COST §5.5);
  * if a breakout fires in the slice, the position is opened AND a venue reduce-only protective stop
    rests for it (EXECUTION P6 / Inv S).

Deterministic + fast: a fixed 600-bar slice of two majors (BTCUSDT/ETHUSDT) at a fixed offset; the
slice spans one 8h funding stamp per symbol so the funding intake is exercised.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("nautilus_trader")

_LAKE = pathlib.Path("data/stitched")
_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_BARS = 600
_OFFSET = 100_000


def _have_lake() -> bool:
    return all((_LAKE / f"{s}.parquet").exists() for s in _SYMBOLS)


pytestmark = [
    pytest.mark.nautilus,
    pytest.mark.skipif(not _have_lake(), reason="real data lake not present on this host"),
]


def _run():
    from convexity.run.backtest import run_backtest_real

    return run_backtest_real(
        symbols=_SYMBOLS,
        bars_limit=_BARS,
        bars_offset=_OFFSET,
        warmup_bars=64,
        with_funding=True,
        with_ops=True,
        dispose=False,
    )


def test_real_node_runs_to_completion_over_real_bars():
    """The full node runs end-to-end over the REAL lake slice and the composer is driven each bar."""
    result = _run()
    strat = result.strategy
    engine = result.engine
    try:
        assert strat.bars_seen == _BARS * len(_SYMBOLS), "every bar of every pair must be consumed"
        assert strat.decisions > 0, "the composer must have been driven on decision bars"
        assert result.n_bars == _BARS
        assert set(result.symbols) == set(_SYMBOLS)
    finally:
        engine.dispose()


def test_real_pillars_are_wired_not_synthetic():
    """The composer's BREAKOUT/LEVERAGE/TREND are the REAL engines — NOT the synthetic stubs."""
    from convexity.run.backtest import _SyntheticBreakout, _SyntheticLeverage, _SyntheticTrend
    from convexity.run.real_deps import _RealBreakout, _RealLeverage, _RealTrend

    result = _run()
    engine = result.engine
    try:
        comp = result.strategy._deps.composer
        assert isinstance(comp.breakout, _RealBreakout)
        assert isinstance(comp.leverage, _RealLeverage)
        assert isinstance(comp.trend, _RealTrend)
        # And explicitly NOT the deterministic stubs the synthetic harness injects.
        assert not isinstance(comp.breakout, _SyntheticBreakout)
        assert not isinstance(comp.leverage, _SyntheticLeverage)
        assert not isinstance(comp.trend, _SyntheticTrend)
        # The real breakout actually drove its validated detector pipeline (one per traded symbol).
        assert comp.breakout._pipelines, "the real BREAKOUT pipeline must have run over the bars"
        # EXECUTION + OPS are the live engines (the composer holds the real OpsAdapter).
        assert comp.ops is not None
    finally:
        engine.dispose()


def test_real_node_consumes_the_funding_feed():
    """The funding feed is fed AND consumed: the node ingested the realized funding stamps (§5.5)."""
    result = _run()
    engine = result.engine
    try:
        assert result.strategy.funding_events_seen > 0, (
            "the node must have ingested FundingRateUpdate stamps from the real funding lake"
        )
    finally:
        engine.dispose()


def test_real_breakout_fire_opens_position_with_protective_stop():
    """If a breakout fires in the slice, the position opens AND a reduce-only stop rests (P6/Inv S)."""
    result = _run()
    engine = result.engine
    try:
        strat = result.strategy
        if strat.entries_submitted == 0:
            pytest.skip("no breakout fired in this real-data slice (acceptable; nothing to manage)")
        # Entries fired → at least one filled position and a protective stop submitted per open.
        assert strat.stops_submitted >= 1, "a venue reduce-only protective stop must be submitted"
        opens = engine.cache.positions_open()
        assert opens, "a fired+filled breakout must leave an open position"
        for inst in engine.cache.instruments():
            if not engine.cache.positions_open(instrument_id=inst.id):
                continue
            resting = [
                o
                for o in engine.cache.orders_open(instrument_id=inst.id)
                if getattr(o, "is_reduce_only", False) and getattr(o, "trigger_price", None)
            ]
            assert resting, f"an open {inst.id} must have a resting reduce-only protective stop"
            # Inv S: EXACTLY ONE protective stop per intent — a native-modify trail must amend the
            # resting order in place, never orphan the prior generation as a second resting stop.
            by_intent: dict[str, int] = {}
            for o in resting:
                intent = o.client_order_id.value.split("|", 1)[0]
                by_intent[intent] = by_intent.get(intent, 0) + 1
            assert all(n == 1 for n in by_intent.values()), (
                f"each open {inst.id} intent must have exactly one resting protective stop, "
                f"got {by_intent}"
            )
    finally:
        engine.dispose()


def test_shared_session_builder_drives_the_real_backtest():
    """The ONE shared builder path: RunConfig → build_session → run_session over the real lake.

    Proves backtest now runs through the SAME ``run/session.build_session`` assembly paper/live use
    (config-only mode switch), with a POPULATED SystemUniverse (not the old ``load_universe_meta([])``
    inline path) and the real-pillar composer the plan carries.
    """
    from convexity.run.backtest import run_session
    from convexity.run.real_deps import _RealBreakout
    from convexity.run.session import RunConfig, build_session

    cfg = RunConfig(
        mode="backtest", symbols=_SYMBOLS,
        bars_limit=_BARS, bars_offset=_OFFSET, warmup_bars=64,
        with_funding=True, with_ops=True,
    )
    plan = build_session(cfg)
    # the plan SYSTEM config sees the real universe (regression guard for the empty-universe bug)
    assert set(plan.deps.sys_cfg.universe.specs) == set(_SYMBOLS)
    assert isinstance(plan.composer.breakout, _RealBreakout)

    result = run_session(plan, dispose=False)
    engine = result.engine
    try:
        assert result.strategy.bars_seen == _BARS * len(_SYMBOLS)
        assert result.strategy.decisions > 0
        assert set(result.symbols) == set(_SYMBOLS)
    finally:
        engine.dispose()
