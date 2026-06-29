"""The ONE shared session builder — universe→specs→deps, the config-only mode switch (run/session).

Pure-Python (NautilusTrader-free, Invariant N): :func:`resolve_universe` and :func:`build_session`
build the specs + real-pillar composer + ``StrategyDeps`` without importing the engine, so this runs
in the pillar venv. The crux these assert: backtest / paper / live now share ONE assembly path — the
plan a backtest runs is the same object paper/live consume; switching modes is just ``RunConfig.mode``.
"""

from __future__ import annotations

import pytest

from convexity.run.session import (
    RunConfig,
    bar_type_for,
    build_session,
    instrument_id_str,
    resolve_universe,
)


def test_resolve_universe_explicit_symbols():
    specs = resolve_universe(symbols=("BTCUSDT", "ETHUSDT"))
    assert set(specs) == {"BTCUSDT", "ETHUSDT"}
    for sym, spec in specs.items():
        assert spec.symbol == sym
        assert spec.asset_class == "crypto_perp"
        assert spec.venue == "BINANCE"
        assert spec.tick_size > 0 and spec.lot_size > 0
        assert spec.funding_interval_ms is not None  # perp ⇒ funding-aware


def test_resolve_universe_unknown_symbol_raises():
    with pytest.raises(KeyError):
        resolve_universe(symbols=("NOTAREALSYMBOLUSDT",))


def test_resolve_universe_sleeve_membership_matches_universe_md():
    crypto = resolve_universe(sleeves=("crypto",))
    assert "BTCUSDT" in crypto and "ETHUSDT" in crypto
    # equities sleeve members are still Binance -USDT perps for the live venue
    eq = resolve_universe(sleeves=("equities",))
    assert all(sym.endswith("USDT") and spec.venue == "BINANCE" for sym, spec in eq.items())


def test_spec_overrides_correct_per_symbol_market_metadata():
    """A symbol whose real venue tick/lot differs from the class default is corrected in ONE place."""
    specs = resolve_universe(
        symbols=("DOGEUSDT",),
        spec_overrides={"DOGEUSDT": {"tick_size": 0.00001, "lot_size": 1.0}},
    )
    assert specs["DOGEUSDT"].tick_size == 0.00001
    assert specs["DOGEUSDT"].lot_size == 1.0


def test_descriptors_are_shared_across_modes():
    assert instrument_id_str("BTCUSDT") == "BTCUSDT-PERP.BINANCE"
    assert bar_type_for("BTCUSDT") == "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"


def test_build_session_assembles_real_pillar_deps():
    """build_session wires the REAL eight-pillar composer + a POPULATED SystemUniverse (not empty)."""
    from convexity.run.real_deps import _RealBreakout, _RealLeverage, _RealTrend

    cfg = RunConfig(mode="backtest", symbols=("BTCUSDT", "ETHUSDT"), warmup_bars=64)
    plan = build_session(cfg)

    assert set(plan.specs) == {"BTCUSDT", "ETHUSDT"}
    assert plan.instrument_ids == ("BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE")
    assert plan.bar_types[0] == "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"

    deps = plan.deps
    # the composer's pillars are the REAL engines, not synthetic stubs
    assert isinstance(plan.composer.breakout, _RealBreakout)
    assert isinstance(plan.composer.leverage, _RealLeverage)
    assert isinstance(plan.composer.trend, _RealTrend)
    # the id↔symbol maps are bidirectional + consistent
    assert deps.instrument_to_symbol["BTCUSDT-PERP.BINANCE"] == "BTCUSDT"
    assert deps.symbol_to_instrument["BTCUSDT"] == "BTCUSDT-PERP.BINANCE"
    assert deps.warmup_bars == 64
    # SYSTEM gets the REAL universe (the old inline path passed load_universe_meta([]) — empty)
    assert set(deps.sys_cfg.universe.specs) == {"BTCUSDT", "ETHUSDT"}


def test_mode_switch_is_config_only():
    """Changing only RunConfig.mode yields the same deps assembly — the plan is venue-agnostic."""
    bt = build_session(RunConfig(mode="backtest", symbols=("BTCUSDT",)))
    paper = build_session(RunConfig(mode="paper", symbols=("BTCUSDT",)))
    live = build_session(RunConfig(mode="live", symbols=("BTCUSDT",)))
    assert bt.instrument_ids == paper.instrument_ids == live.instrument_ids
    assert bt.bar_types == paper.bar_types == live.bar_types
    assert set(bt.specs) == set(paper.specs) == set(live.specs)
