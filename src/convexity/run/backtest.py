"""Parity-backtest harness — a RUNNABLE end-to-end Nautilus ``BacktestEngine`` (ARCHITECTURE §8.4a).

This is the deployment-phase backtest entrypoint: it stands up a real ``nautilus_trader``
``BacktestEngine`` with the Binance USDT-M perp venue, feeds it synthetic 1-minute klines for N
perps, hosts the live :class:`~convexity.strategy.convexity_strategy._ConvexityStrategy`, and runs
the node to completion. The SAME strategy class runs unchanged in paper / live — only the venue
wiring differs (the engine owns backtest↔live parity, ARCHITECTURE §1/§4).

NautilusTrader-importing, so VM-only (the pure-Python pillar venv has no engine — Invariant N). It is
exercised by ``tests/integration/test_backtest_end_to_end.py`` which asserts the strategy ACTUALLY
trades over the synthetic bars (entries + fills + N managed positions + a resting protective stop,
and the DEGRADE/flatten path reachable) — a genuine end-to-end run, not the composer-direct path.

The composition under test is injected via :func:`build_synthetic_composer` (the validated
EXECUTION/SYSTEM/OPS engines wired to deterministic Protocol-satisfying pillar stubs, exactly the
path the integration composer suite covers); a live deployment swaps in the real per-bar pillar
bundle at the same ``strategy.configure(deps)`` seam. Funding is only charged if ``FundingRateUpdate``
data is fed (none here — the synthetic run is directional only).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Sequence

import numpy as np

from convexity.core.types import EntrySignal, InstrumentSpec
from convexity.execution.config.resolve import resolve_config as exec_resolve_config
from convexity.execution.engine_api import build_engine as build_exec_engine
from convexity.execution.synth.generate import perp_instrument
from convexity.leverage.types import LeverageDecision
from convexity.ops.nautilus.adapter import OpsAdapter
from convexity.ops.ops_api import build_engine as build_ops_engine
from convexity.strategy.convexity_strategy import (
    ConvexityComposer,
    StrategyDeps,
    make_strategy_class,
)
from convexity.system.config.resolve import resolve_config as system_resolve_config
from convexity.system.config.universe import load_universe_meta

__all__ = [
    "BacktestResult",
    "run_backtest",
    "run_backtest_real",
    "run_session",
    "build_synthetic_composer",
    "make_synthetic_bars",
]


# --------------------------------------------------------------------------- deterministic pillars
#
# These satisfy the ``core.protocols`` BREAKOUT/LEVERAGE/TREND seams with deterministic, scriptable
# output so the end-to-end node TRADES on cue (the real pillar math is the 613-test pillar suite).
# A live deployment replaces them with the real per-bar pillar bundle at strategy.configure(deps).


class _SyntheticBreakout:
    """Fire ONE entry per symbol once the host's bar clock clears ``fire_after_idx`` (one open per
    pair), then go quiet for that symbol so the position is opened exactly once."""

    def __init__(self, *, fire_after_idx: int, mids: dict[str, float], direction: int = 1) -> None:
        self._fire_after = fire_after_idx
        self._mids = mids
        self._dir = direction
        self._fired: set[str] = set()
        self.idx = 0

    def propose(self, instrument: InstrumentSpec) -> EntrySignal | None:
        sym = instrument.symbol
        if sym in self._fired or self.idx < self._fire_after or sym not in self._mids:
            return None
        self._fired.add(sym)
        mid = self._mids[sym]
        invalidation = mid - 200.0 if self._dir == 1 else mid + 200.0
        return EntrySignal(
            symbol=sym, signal_idx=self.idx - 1, entry_idx=self.idx,
            direction=self._dir,  # type: ignore[arg-type]
            status="entry", quality_score=1.0, threshold=0.5,
            entry_ref_price=mid, invalidation=invalidation,
            stop_distance=abs(mid - invalidation),
            barrier_price=mid + self._dir * 10.0, range_width_sigma=2.0,
            compression_pct=0.3, false_break_risk=0.1, urgency="urgent",
            evidence={"glr": 3.0}, reason_codes=("synthetic_entry",), detector_version="synthetic",
        )


class _SyntheticLeverage:
    """A fixed, safe size (RISK metadata only — no edge, Inv A)."""

    def size_position(self, signal, risk, cfg, book) -> LeverageDecision:
        return LeverageDecision(
            instrument=signal.instrument, leverage=0.5, size=5.0, rounded_size=5.0,
            binding_ceiling="stop", kelly_fraction=0.1, governor_multiplier=1.0, g_t=1.0,
            risk_budget=0.06, planned_loss_on_margin=5.0, planned_loss_on_equity=5.0,
            reason_codes=("synthetic",), engine_version="synthetic-lev",
        )


class _TrendDecision:
    def __init__(self, action: str, intent=None) -> None:
        self.action = action
        self.size_fraction = 1.0
        self.stop_cause = None
        self.reason_codes = (f"trend_{action}",)
        self.execution_intent = intent


class _SyntheticTrend:
    """Hold the position open (the venue reduce-only protective stop is the real exit floor)."""

    def update(self, bars, position, spec, cfg, idx):
        return _TrendDecision("continue")


class _LevCfg:
    class defaults:
        position_margin = 100.0
        capital_cap = 0.13


def build_synthetic_composer(
    *, mids: dict[str, float], fire_after_idx: int, with_ops: bool = True, direction: int = 1
) -> tuple[ConvexityComposer, _SyntheticBreakout]:
    """Wire a :class:`ConvexityComposer` from the REAL EXECUTION/SYSTEM/OPS engines + deterministic
    BREAKOUT/LEVERAGE/TREND stubs. Returns the composer + the breakout stub (whose ``idx`` the host
    advances each bar so the entry fires on schedule)."""
    breakout = _SyntheticBreakout(fire_after_idx=fire_after_idx, mids=mids, direction=direction)
    composer = ConvexityComposer(
        breakout=breakout,
        leverage=_SyntheticLeverage(),
        trend=_SyntheticTrend(),
        execution=build_exec_engine(),
        ops_adapter=OpsAdapter(build_ops_engine()) if with_ops else None,
    )
    return composer, breakout


# --------------------------------------------------------------------------- synthetic market data


def make_synthetic_bars(
    n: int = 96, *, start_mid: float = 30_000.0, seed: int = 7
) -> tuple[np.ndarray, ...]:
    """A deterministic 1-min OHLCV panel as raw arrays (the same generator the integration synth
    uses). Returns (ts_ms, open, high, low, close, volume)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 5.0, size=n).cumsum()
    close = start_mid + steps
    open_ = np.concatenate([[start_mid], close[:-1]])
    # The high/low must bracket BOTH open and close (Nautilus enforces low<=o,c<=high). Build them
    # off the per-bar (open,close) range, not just the close, so the synthetic bar is always valid.
    oc_hi = np.maximum(open_, close)
    oc_lo = np.minimum(open_, close)
    high = oc_hi + rng.uniform(1.0, 8.0, size=n)
    low = oc_lo - rng.uniform(1.0, 8.0, size=n)
    ts = (np.arange(n, dtype=np.int64) * 60_000) + 1_700_000_000_000
    vol = rng.uniform(10.0, 100.0, size=n)
    return ts, open_, high, low, close, vol


# --------------------------------------------------------------------------- the runnable harness


@dataclass(frozen=True)
class BacktestResult:
    """What an end-to-end run reports (the node ran; the strategy traded). Engine kept for asserts."""

    engine: Any
    strategy: Any
    n_bars: int
    symbols: tuple[str, ...]


def run_backtest(
    *,
    n_bars: int = 96,
    n_pairs: int = 2,
    warmup_bars: int = 16,
    fire_after_idx: int = 20,
    start_mid: float = 30_000.0,
    seed: int = 7,
    composer_factory: Callable[..., tuple[ConvexityComposer, Any]] | None = None,
    on_bar_hook: Callable[[Any, Any], None] | None = None,
    bypass_logging: bool = True,
    dispose: bool = False,
) -> BacktestResult:
    """Stand up a real ``BacktestEngine``, host the ``_ConvexityStrategy``, feed synthetic klines for
    ``n_pairs`` distinct perps, and run to completion. Returns a :class:`BacktestResult` (the engine
    + strategy left live so the caller can assert on fills/positions, unless ``dispose``).

    This is a GENUINE end-to-end run: the strategy subscribes to the engine's bars, projects the
    engine's ``Cache``/``Portfolio`` into the pure pillar types, calls the composer, and submits the
    resulting orders back through the engine — the same code path paper/live drive. The composer
    manages all N pairs in one ``on_decision_bar`` call (N simultaneous per-pair positions).
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    # Distinct Binance USDT-M perps, one per pair (the test kit ships BTC/ETH; a 3rd reuses ADA).
    providers = [
        TestInstrumentProvider.btcusdt_perp_binance,
        TestInstrumentProvider.ethusdt_perp_binance,
        TestInstrumentProvider.onethousandrats_perp_binance,
    ]
    n_pairs = max(1, min(n_pairs, len(providers)))

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(bypass_logging=bypass_logging))
    )
    engine.add_venue(
        venue=Venue("BINANCE"), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=USDT, starting_balances=[Money(1_000_000, USDT)],
    )

    bar_types: list[str] = []
    instrument_to_symbol: dict[str, str] = {}
    symbol_to_instrument: dict[str, str] = {}
    specs: dict[str, InstrumentSpec] = {}
    mids: dict[str, float] = {}
    instruments = []
    base_ts_ns = 1_700_000_000_000_000_000

    all_bars = []
    for i in range(n_pairs):
        inst = providers[i]()
        engine.add_instrument(inst)
        instruments.append(inst)
        iid_str = str(inst.id)
        sym = iid_str.split("-")[0].split(".")[0]  # e.g. "ETHUSDT" from "ETHUSDT-PERP.BINANCE"
        bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")
        bar_types.append(str(bar_type))
        instrument_to_symbol[iid_str] = sym
        symbol_to_instrument[sym] = iid_str
        specs[sym] = perp_instrument(sym)
        pair_mid = start_mid * (1.0 + 0.1 * i)
        mids[sym] = pair_mid

        ts, o, h, low, c, v = make_synthetic_bars(n_bars, start_mid=pair_mid, seed=seed + i)
        p = lambda x: inst.make_price(Decimal(str(round(float(x), 1))))  # noqa: E731
        for j in range(n_bars):
            all_bars.append(Bar(
                bar_type=bar_type, open=p(o[j]), high=p(h[j]), low=p(low[j]), close=p(c[j]),
                volume=inst.make_qty(Decimal(str(round(float(v[j]), 3)))),
                ts_event=base_ts_ns + j * 60_000_000_000, ts_init=base_ts_ns + j * 60_000_000_000,
            ))
    all_bars.sort(key=lambda b: b.ts_init)
    engine.add_data(all_bars)

    # Build (or inject) the composer.
    fac = composer_factory or build_synthetic_composer
    composer, breakout_stub = fac(mids=mids, fire_after_idx=fire_after_idx)

    # Host the strategy + inject the pure-Python deps.
    strategy_cls = make_strategy_class()
    config = _make_strategy_config(bar_types)
    strategy = strategy_cls(config)
    deps = StrategyDeps(
        composer=composer, specs=specs,
        leverage_risk={sym: object() for sym in specs},
        leverage_cfg=_LevCfg(), trend_cfg=object(),
        exec_cfg=exec_resolve_config(perp_instrument(next(iter(specs)))),
        sys_cfg=system_resolve_config(load_universe_meta([])),
        instrument_to_symbol=instrument_to_symbol,
        symbol_to_instrument=symbol_to_instrument,
        warmup_bars=warmup_bars,
    )
    strategy.configure(deps)
    _attach_breakout_clock(strategy, breakout_stub)
    # An optional per-bar hook (h(strategy, bar)) the integration suite uses to drive a deterministic
    # equity/heartbeat trajectory WHILE the engine is connected (so an OPS HALT mid-run actually
    # submits orders to the live venue, not against a disconnected post-run exec client).
    if on_bar_hook is not None:
        _attach_on_bar_hook(strategy, on_bar_hook)
    engine.add_strategy(strategy)

    engine.run()
    result = BacktestResult(
        engine=engine, strategy=strategy, n_bars=n_bars, symbols=tuple(specs),
    )
    if dispose:
        engine.dispose()
    return result


def _make_strategy_config(bar_types: Sequence[str]):
    """Build the (serializable) ``StrategyConfig`` carrying the bar-type descriptors (VM-only)."""
    from nautilus_trader.trading.strategy import StrategyConfig

    class _ConvexityConfig(StrategyConfig, frozen=True):
        bar_types: tuple[str, ...]

    return _ConvexityConfig(bar_types=tuple(bar_types))


def _attach_breakout_clock(strategy: Any, breakout_stub: Any) -> None:
    """Wrap the strategy's ``on_bar`` so the synthetic breakout's ``idx`` tracks the bar count (so
    the scripted entry fires on schedule). A no-op for an injected real breakout without ``idx``."""
    if not hasattr(breakout_stub, "idx"):
        return
    original = strategy.on_bar

    def _wrapped(bar):
        breakout_stub.idx = strategy.bars_seen
        return original(bar)

    strategy.on_bar = _wrapped  # type: ignore[method-assign]


def _attach_on_bar_hook(strategy: Any, hook: Callable[[Any, Any], None]) -> None:
    """Wrap ``on_bar`` so ``hook(strategy, bar)`` runs BEFORE each bar is processed — the seam the
    integration suite uses to drive equity/heartbeat trajectories while the engine is connected."""
    original = strategy.on_bar

    def _wrapped(bar):
        hook(strategy, bar)
        return original(bar)

    strategy.on_bar = _wrapped  # type: ignore[method-assign]


# --------------------------------------------------------------------------- REAL-DATA harness
#
# The same runnable ``BacktestEngine`` node, but driven by the REAL data lake (data/stitched +
# data/funding) + the REAL eight-pillar composition (run/real_deps.build_real_composer) instead of
# synthetic klines + the deterministic pillar stubs. The strategy class, the per-bar §4 composition,
# the projection seam and the order/stop/OPS wiring are IDENTICAL — only the data source and the
# pillar engines behind the Protocols differ (ARCHITECTURE §1/§4: the engine owns backtest↔live
# parity; the same node runs in paper/live, swapping the venue, not the composition).


def run_session(
    plan: Any,
    *,
    lake: Any = None,
    funding_lake: Any = None,
    bypass_logging: bool = True,
    dispose: bool = False,
) -> BacktestResult:
    """Run a ``BacktestEngine`` over the REAL lake driven by a shared :class:`SessionPlan`.

    This is the BACKTEST consumer of the one shared builder (``run/session.build_session``): the
    plan's ``specs`` / ``deps`` / ``bar_types`` are EXACTLY what paper/live consume — only the venue
    wiring below (a simulated ``BacktestEngine`` over ``data/stitched`` + ``data/funding``) differs.
    Builds a Binance USDT-M ``CryptoPerpetual`` per symbol from the plan's AMBER ``InstrumentSpec``,
    loads the bar + funding feeds, hosts the live ``_ConvexityStrategy`` with the plan's already-wired
    real-pillar ``StrategyDeps``, and runs to completion.
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money

    from convexity.run.data_loader import (
        DEFAULT_FUNDING_LAKE,
        DEFAULT_LAKE,
        build_instrument,
        load_bars,
        load_funding,
    )

    cfg = plan.cfg
    lake = lake if lake is not None else DEFAULT_LAKE
    funding_lake = funding_lake if funding_lake is not None else DEFAULT_FUNDING_LAKE

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(bypass_logging=bypass_logging))
    )
    engine.add_venue(
        venue=Venue(cfg.venue), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=USDT, starting_balances=[Money(cfg.starting_equity, USDT)],
    )

    all_bars: list[Any] = []
    all_funding: list[Any] = []
    # CROSS-MODE GAP MARKER (FRICTION #2): collect the lake's gap_before bar-close timestamps per
    # instrument id so the strategy's on_bar raises the SAME Bars.is_session_open flag the live
    # ts-discontinuity path raises (the gap can't ride the Nautilus Bar, so it is threaded via deps).
    gap_close_ms_by_instrument: dict[str, frozenset[int]] = {}
    for sym, spec in plan.specs.items():
        inst = build_instrument(spec, venue=cfg.venue)
        engine.add_instrument(inst)
        bars, gap_close_ms = load_bars(
            sym, inst, lake=lake, limit=cfg.bars_limit, offset=cfg.bars_offset,
            venue=cfg.venue, return_gaps=True,
        )
        gap_close_ms_by_instrument[str(inst.id)] = gap_close_ms
        all_bars.extend(bars)
        if cfg.with_funding and bars:
            start_ms = int(bars[0].ts_event // 1_000_000)
            end_ms = int(bars[-1].ts_event // 1_000_000)
            all_funding.extend(
                load_funding(sym, inst, lake=funding_lake, start_ms=start_ms, end_ms=end_ms)
            )

    all_bars.sort(key=lambda b: b.ts_init)
    engine.add_data(all_bars)
    if all_funding:
        all_funding.sort(key=lambda f: f.ts_init)
        engine.add_data(all_funding)

    strategy_cls = make_strategy_class()
    strategy = strategy_cls(_make_strategy_config(plan.bar_types))
    # Thread the lake gap map into the (frozen) deps so on_bar can key the per-bar gap flag (FRICTION #2).
    deps = dataclasses.replace(
        plan.deps, gap_close_ms_by_instrument=gap_close_ms_by_instrument
    )
    strategy.configure(deps)
    engine.add_strategy(strategy)

    engine.run()
    result = BacktestResult(
        engine=engine, strategy=strategy, n_bars=cfg.bars_limit, symbols=tuple(plan.specs),
    )
    if dispose:
        engine.dispose()
    return result


def run_backtest_real(
    *,
    symbols: Sequence[str],
    bars_limit: int = 400,
    bars_offset: int = 0,
    warmup_bars: int = 64,
    h_sys: float = 30.0,
    position_margin: float = 5_000.0,
    starting_equity: float = 1_000_000.0,
    with_funding: bool = True,
    with_ops: bool = True,
    lake: Any = None,
    funding_lake: Any = None,
    bypass_logging: bool = True,
    dispose: bool = False,
) -> BacktestResult:
    """Stand up a real ``BacktestEngine`` over the REAL lake + funding feed + the REAL pillar engines.

    Thin compatibility wrapper over the shared builder: assembles a :class:`RunConfig` from these args,
    builds the :class:`SessionPlan` via ``run/session.build_session`` (the SAME path paper/live use),
    and runs it through :func:`run_session`. Kept so existing callers/tests keep their signature while
    the assembly is now the single shared path (no inline deps construction here anymore).
    """
    from convexity.run.session import RunConfig, build_session

    cfg = RunConfig(
        mode="backtest", symbols=tuple(symbols),
        bars_limit=bars_limit, bars_offset=bars_offset, warmup_bars=warmup_bars,
        h_sys=h_sys, position_margin=position_margin, starting_equity=starting_equity,
        with_funding=with_funding, with_ops=with_ops,
    )
    plan = build_session(cfg)
    return run_session(
        plan, lake=lake, funding_lake=funding_lake,
        bypass_logging=bypass_logging, dispose=dispose,
    )
