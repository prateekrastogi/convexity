"""ONE shared session builder — the single ``universe -> (specs, instruments, bar_types, deps)`` path
that backtest / paper / live all call, so switching modes is config-only (ARCHITECTURE §1/§4).

Before this module, ``StrategyDeps`` was assembled ONLY inside ``run/backtest.run_backtest_real``
(``build_real_composer`` + ``build_real_deps_configs`` + ``perp_instrument`` per symbol +
``load_universe_meta([])`` inline), while ``run/paper.build_trading_node`` and
``run/live.build_live_node`` took ``deps`` / ``instrument_ids`` / ``bar_types`` as PARAMETERS — the
caller had to hand-replicate backtest's assembly to ever run paper/live. That divergence is the core
friction this module removes: all three modes now resolve the universe, build the per-symbol specs,
the real eight-pillar composer + resolved configs, and the ``StrategyDeps`` through the SAME two
functions here.

PURE-PYTHON, NautilusTrader-free (Invariant N): :func:`resolve_universe` and :func:`build_session`
import no engine, so they are importable in the pillar venv and unit-testable off-VM. The bar/
instrument loading that DOES need the engine stays behind the VM-only ``run/backtest`` +
``run/data_loader`` seams; this module assembles everything else and hands the mode entrypoints a
ready :class:`SessionPlan`.

The flow every mode shares::

    cfg  = RunConfig(mode=..., symbols=[...] | sleeves=(...), ...)
    plan = build_session(cfg)                 # specs + composer + configs + StrategyDeps + bar_types
    # backtest: run/backtest.run_session(plan, cfg)     -> BacktestEngine over the lake
    # paper:    run/paper.run_paper(plan)                -> TradingNode, BinanceEnvironment.DEMO
    # live:     run/live.run_live(plan, i_understand_real_money=True) -> ... LIVE
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Mapping, Sequence

from convexity.core.types import InstrumentSpec
from convexity.selection.universe import load_universe

if TYPE_CHECKING:  # type-only; the heavy deps are built lazily so this module stays import-light.
    from convexity.strategy.convexity_strategy import ConvexityComposer, StrategyDeps

__all__ = [
    "RunConfig",
    "SessionPlan",
    "resolve_universe",
    "instrument_id_str",
    "bar_type_for",
    "build_session",
]

# --------------------------------------------------------------------------- universe -> specs
#
# The ONE place a universe symbol becomes an AMBER ``InstrumentSpec``. UNIVERSE.md carries only
# membership (symbol/sector/source); the per-asset-class MARKET metadata (tick/lot/fees/funding) is
# attached here. A per-symbol ``spec_overrides`` map on RunConfig lets a symbol whose real venue
# tick/lot differs from the class default be corrected in ONE place (see FRICTION note: the live
# venue's InstrumentProvider carries exact metadata; the backtest builds from these specs, so any
# symbol with a non-default tick/lot must be overridden here or backtest precision diverges from live).

# Asset-class market-metadata defaults (Binance USDT-M perp + Dukascopy-proxy equity/commodity).
_CLASS_DEFAULTS: dict[str, dict] = {
    "crypto_perp": dict(
        venue="BINANCE", timezone="UTC", tick_size=0.1, lot_size=0.001,
        min_notional=5.0, maker_fee_bps=1.0, taker_fee_bps=4.0,
        funding_interval_ms=8 * 3600 * 1000, funding_cap=0.0075,
    ),
}

# UNIVERSE.md sleeves are all Binance USDT-M perps (equities/commodities trade as -USDT perps too —
# the synthetic Dukascopy history is for warmup only; the live/paper venue is always the Binance perp).
_SLEEVE_ASSET_CLASS = {
    "crypto": "crypto_perp",
    "equities": "crypto_perp",
    "commodities": "crypto_perp",
}


def resolve_universe(
    *,
    symbols: Sequence[str] | None = None,
    sleeves: Sequence[str] = ("crypto", "equities", "commodities"),
    spec_overrides: Mapping[str, Mapping[str, object]] | None = None,
    universe_path: str | None = None,
) -> dict[str, InstrumentSpec]:
    """Resolve the trading universe to ``{symbol: InstrumentSpec}`` — the SINGLE symbol→spec map.

    If ``symbols`` is given it is the exact set (each must appear in UNIVERSE.md so its asset-class is
    known); otherwise the union of the requested ``sleeves`` from UNIVERSE.md is used. Each symbol's
    AMBER spec is the per-asset-class default, with any ``spec_overrides[symbol]`` field merged in
    (the one place a symbol's real venue tick/lot/fees can be corrected).
    """
    u = load_universe(universe_path)
    # symbol -> sleeve, from the membership doc.
    sym_sleeve: dict[str, str] = {}
    for sleeve in ("crypto", "equities", "commodities"):
        for row in u.get(sleeve, []):
            sym_sleeve[row["symbol"]] = sleeve

    if symbols is not None:
        want = list(symbols)
    else:
        # preserve UNIVERSE.md order within the requested sleeves
        want = [row["symbol"] for sleeve in sleeves for row in u.get(sleeve, [])]

    overrides = spec_overrides or {}
    out: dict[str, InstrumentSpec] = {}
    for sym in want:
        sleeve = sym_sleeve.get(sym)
        if sleeve is None:
            raise KeyError(
                f"{sym!r} is not in UNIVERSE.md — cannot resolve its asset-class/market metadata. "
                "Add it to UNIVERSE.md or pass an explicit spec_overrides entry with asset_class."
            )
        asset_class = _SLEEVE_ASSET_CLASS[sleeve]
        meta = dict(_CLASS_DEFAULTS[asset_class])
        spec = InstrumentSpec(symbol=sym, asset_class=asset_class, **meta)
        ov = overrides.get(sym)
        if ov:
            spec = replace(spec, **dict(ov))
        out[sym] = spec
    return out


def instrument_id_str(symbol: str, venue: str = "BINANCE") -> str:
    """Nautilus instrument-id string for a universe perp symbol (``BTCUSDT`` → ``BTCUSDT-PERP.BINANCE``)."""  # noqa: E501
    return f"{symbol}-PERP.{venue}"


def bar_type_for(symbol: str, venue: str = "BINANCE") -> str:
    """The 1-min LAST EXTERNAL bar-type descriptor for a universe perp symbol (shared by all modes)."""
    return f"{instrument_id_str(symbol, venue)}-1-MINUTE-LAST-EXTERNAL"


# --------------------------------------------------------------------------- the run config


@dataclass(frozen=True)
class RunConfig:
    """The ONE config object that selects a mode and parameterises a run — switching modes is just
    changing ``mode`` (+ creds for paper/live). A thin ``conf/{mode}.yaml`` can populate this, but the
    canonical form is config-as-code so the pillar venv needs no YAML dependency.

    ``symbols`` (exact set) takes precedence over ``sleeves`` (UNIVERSE.md membership). The
    pillar/run-level knobs (``warmup_bars``/``h_sys``/``position_margin``/``starting_equity``) and the
    feature toggles (``with_funding``/``with_ops``) are identical across modes — only ``mode`` and the
    venue creds differ, which is the whole point.
    """

    mode: str = "backtest"  # backtest | paper | live
    symbols: tuple[str, ...] | None = None
    sleeves: tuple[str, ...] = ("crypto", "equities", "commodities")
    spec_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    venue: str = "BINANCE"

    warmup_bars: int = 64
    h_sys: float = 30.0
    position_margin: float = 5_000.0
    starting_equity: float = 1_000_000.0
    with_funding: bool = True
    with_ops: bool = True

    # backtest-only data window
    bars_limit: int = 400
    bars_offset: int = 0

    # paper/live creds (env by Nautilus convention if None)
    api_key: str | None = None
    api_secret: str | None = None
    trader_id: str | None = None


# --------------------------------------------------------------------------- the session plan


@dataclass
class SessionPlan:
    """Everything the three mode entrypoints need, assembled ONCE by :func:`build_session`.

    ``deps`` is the fully-wired :class:`StrategyDeps` (real composer + resolved pillar configs +
    populated SystemUniverse); ``instrument_ids`` / ``bar_types`` are the venue descriptors paper/live
    feed their ``InstrumentProviderConfig`` and the strategy subscribes to. ``specs`` is the shared
    symbol→spec map (the backtest builds its programmatic instruments from these). All three modes
    consume the SAME plan — only the venue wiring downstream differs.
    """

    cfg: RunConfig
    specs: Mapping[str, InstrumentSpec]
    instrument_ids: tuple[str, ...]
    bar_types: tuple[str, ...]
    instrument_to_symbol: Mapping[str, str]
    symbol_to_instrument: Mapping[str, str]
    deps: "StrategyDeps"
    composer: "ConvexityComposer"


def build_session(cfg: RunConfig) -> SessionPlan:
    """Build the shared :class:`SessionPlan` from a :class:`RunConfig` — the one assembly path.

    Resolves the universe → specs (:func:`resolve_universe`), builds the REAL eight-pillar composer +
    resolved pillar configs (``run/real_deps``), assembles the ``StrategyDeps`` with a POPULATED
    :class:`SystemUniverse` (the old inline path passed ``load_universe_meta([])`` — an empty universe
    — to SYSTEM; this passes the real specs so the orchestrator composes over the actual instruments),
    and computes the venue instrument-id + bar-type descriptors. NautilusTrader-free: the engine-side
    instrument/bar loading happens in the mode entrypoint, driven by this plan.
    """
    from convexity.execution.config.resolve import resolve_config as exec_resolve_config
    from convexity.run.real_deps import build_real_composer, build_real_deps_configs
    from convexity.strategy.convexity_strategy import StrategyDeps
    from convexity.system.config.resolve import resolve_config as system_resolve_config
    from convexity.system.config.universe import load_universe_meta

    specs = resolve_universe(
        symbols=cfg.symbols, sleeves=cfg.sleeves,
        spec_overrides=cfg.spec_overrides, universe_path=None,
    )
    if not specs:
        raise ValueError("build_session: resolved an empty universe — check symbols/sleeves")

    instrument_to_symbol: dict[str, str] = {}
    symbol_to_instrument: dict[str, str] = {}
    instrument_ids: list[str] = []
    bar_types: list[str] = []
    for sym in specs:
        iid = instrument_id_str(sym, cfg.venue)
        instrument_to_symbol[iid] = sym
        symbol_to_instrument[sym] = iid
        instrument_ids.append(iid)
        bar_types.append(bar_type_for(sym, cfg.venue))

    configs = build_real_deps_configs(
        specs, h_sys=cfg.h_sys, position_margin=cfg.position_margin
    )
    composer, _breakout = build_real_composer(
        specs=specs, configs=configs, with_ops=cfg.with_ops
    )

    first = next(iter(specs.values()))
    deps = StrategyDeps(
        composer=composer,
        specs=specs,
        leverage_risk={sym: sym for sym in specs},
        leverage_cfg=configs.leverage_cfg,
        trend_cfg=configs.trend_cfg,
        exec_cfg=exec_resolve_config(first),
        sys_cfg=system_resolve_config(load_universe_meta(list(specs.values()))),
        instrument_to_symbol=instrument_to_symbol,
        symbol_to_instrument=symbol_to_instrument,
        warmup_bars=cfg.warmup_bars,
    )
    return SessionPlan(
        cfg=cfg,
        specs=specs,
        instrument_ids=tuple(instrument_ids),
        bar_types=tuple(bar_types),
        instrument_to_symbol=instrument_to_symbol,
        symbol_to_instrument=symbol_to_instrument,
        deps=deps,
        composer=composer,
    )
