"""REAL pillar composition — the deployment-grade bundle that drives the validated engines per-bar.

This replaces the deterministic ``_Synthetic{Breakout,Leverage,Trend}`` stubs in ``run/backtest.py``
with the ACTUAL validated pillar engines, satisfying the same ``core.protocols`` seams the
:class:`~convexity.strategy.convexity_strategy.ConvexityComposer` composes against (ARCHITECTURE §4).
Pure-Python, NautilusTrader-free (the engines never import the engine; this module wires them).

The composition seam the composer drives is per-instrument and per-bar, but two real engines need a
little adaptation to fit the stateless Protocol shapes the composer calls:

  * BREAKOUT (``BreakoutModule.propose(spec)``): the real :class:`PipelineOrchestrator` reads a
    point-in-time :class:`Bars` panel, but the composer's ``propose`` takes only the spec. The
    :class:`_RealBreakout` adapter therefore reads the CURRENT per-instrument bar buffer the strategy
    accumulates (published into a shared :class:`_BarContext` by :class:`_RealComposer` each bar) and
    runs the validated detector batch over it — the exact same ``run_batch`` the BREAKOUT suite
    covers, with the latest-bar single-or-None contract (Inv K).

  * LEVERAGE (``size_position(signal, risk, cfg, book)``): the real engine reads a
    :class:`BookStateFixture` (with ``cov_index``/``candidate_weights``) and a measurable
    :class:`RiskInputs`. The composer passes the SYSTEM ``BookState`` (no ``cov_index``) and an
    opaque ``risk`` handle, so :class:`_RealLeverage` builds a proper ``BookStateFixture`` from the
    per-bar equity/cov context + a ``RiskInputs`` whose ``forecast_vol`` is the validated
    Yang-Zhang/EWMA/HAR blend over the instrument's bar buffer, then delegates to the real
    :class:`LeverageEngineAPI`.

  * TREND (``update(bars, position, spec, cfg, idx)``): the real :class:`TrendLifeEngine` is
    PER-POSITION stateful (it threads one position's exit/trail lifecycle), but the composer holds a
    single trend object and calls it for every open position. :class:`_RealTrend` multiplexes one
    real engine per open intent and forwards the call, returning the real ``ActionDecision``.

  * EXECUTION / OPS: the real engines already expose ``build_engine`` (EXECUTION) and an
    ``OpsAdapter`` over ``build_engine`` (OPS); they are wired unchanged, exactly as live.

:func:`build_real_composer` assembles them into a :class:`_RealComposer` (a thin
``ConvexityComposer`` subclass that publishes the per-bar bar/equity context to the adapters before
running the §4 composition) and resolves each pillar's config via its own ``config/resolve.py`` for
the universe instruments. The returned ``(composer, breakout)`` tuple matches the synthetic factory's
shape so it drops straight into ``run/backtest.py``'s ``composer_factory`` seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from convexity.breakout.config.resolve import resolve_config as breakout_resolve_config
from convexity.breakout.detect.combine import BreakoutModule
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.core.types import Bars, EntrySignal, InstrumentSpec, Position
from convexity.execution.engine_api import build_engine as build_exec_engine
from convexity.leverage.config.asset_class import DEFAULT_PRESETS as LEV_PRESETS
from convexity.leverage.config.defaults import PolicyDefaults as LevPolicyDefaults
from convexity.leverage.config.instrument import InstrumentLeverageSpec
from convexity.leverage.config.resolve import (
    ResolvedConfig as LevResolvedConfig,
    resolve_config as leverage_resolve_config,
)
from convexity.leverage.engine_api import LeverageEngineAPI
from convexity.leverage.types import (
    BookStateFixture,
    ConditionalMAE,
    GapModel,
    GPDFit,
    LiquidityState,
    RiskInputs,
)
from convexity.leverage.vol.forecast import forecast_vol
from convexity.ops.nautilus.adapter import OpsAdapter
from convexity.ops.ops_api import build_engine as build_ops_engine
from convexity.strategy.convexity_strategy import ConvexityComposer
from convexity.trend.config.resolve import resolve_config as trend_resolve_config
from convexity.trend.engine import TrendLifeEngine

__all__ = [
    "build_real_composer",
    "build_real_deps_configs",
    "RealDepsConfigs",
]


# --------------------------------------------------------------------------- per-bar shared context


@dataclass
class _BarContext:
    """The per-bar inputs the stateless Protocol adapters need but the composer does not forward.

    :class:`_RealComposer` publishes these every ``on_decision_bar`` BEFORE running the §4
    composition: the current per-instrument :class:`Bars` panel (what BREAKOUT/LEVERAGE read), the
    latest decision index per instrument, and the account equity/free-margin the LEVERAGE book needs.
    The adapters read this shared object so the composition contract (per-instrument keyed) is
    preserved without changing the composer's call signature.
    """

    bars_by_instrument: Mapping[str, Bars] = field(default_factory=dict)
    idx_by_instrument: Mapping[str, int] = field(default_factory=dict)
    equity: float = 0.0
    peak_equity: float = 0.0
    free_margin: float = 0.0
    eligible: tuple[str, ...] = ()


# --------------------------------------------------------------------------- BREAKOUT adapter


class _RealBreakout:
    """Drives the validated :class:`PipelineOrchestrator` over the strategy's live bar buffer.

    Satisfies ``core.protocols.BreakoutModule``: ``propose(spec) -> EntrySignal | None``. One
    persistent pipeline per instrument so the streaming state amortizes; the actual fire path is the
    validated ``BreakoutModule.propose`` (``run_batch`` + latest-bar single-or-None, Inv K).
    """

    def __init__(self, ctx: _BarContext, cfgs: Mapping[str, Any]) -> None:
        self._ctx = ctx
        self._cfgs = cfgs
        self._pipelines: dict[str, PipelineOrchestrator] = {}

    def propose(self, instrument: InstrumentSpec) -> EntrySignal | None:
        sym = instrument.symbol
        bars = self._ctx.bars_by_instrument.get(sym)
        cfg = self._cfgs.get(sym)
        if bars is None or cfg is None or len(bars) < 2:
            return None
        pipeline = self._pipelines.get(sym)
        if pipeline is None:
            pipeline = PipelineOrchestrator()
            self._pipelines[sym] = pipeline
        module = BreakoutModule(pipeline, cfg, bars)
        return module.propose(instrument)


# --------------------------------------------------------------------------- LEVERAGE adapter


class _RealLeverage:
    """Wraps the validated :class:`LeverageEngineAPI`, building the measurable RISK view per call.

    Satisfies ``core.protocols.LeverageEngine``: ``size_position(signal, risk, cfg, book)``. The
    composer passes a per-instrument ``risk`` handle (the symbol) + the resolved leverage cfg + the
    SYSTEM ``BookState``; this adapter ignores the (incomplete) SYSTEM book and constructs the proper
    :class:`BookStateFixture` (with ``cov_index``/``candidate_weights``) from the shared per-bar
    equity/cov context, plus a :class:`RiskInputs` whose ``forecast_vol`` is the validated YZ/EWMA/
    HAR blend over the instrument's bar buffer (LEVERAGE §15). Pure RISK metadata only (Inv A).
    """

    def __init__(self, ctx: _BarContext) -> None:
        self._ctx = ctx
        self._api = LeverageEngineAPI()

    def size_position(self, signal, risk, cfg, book) -> Any:
        sym = str(risk) if isinstance(risk, str) else getattr(signal, "instrument", "")
        bars = self._ctx.bars_by_instrument.get(sym)
        # Horizon in PERIODS for the σ·√T scaling — use the SYSTEM hold cap as a hold proxy (bars).
        horizon_periods = max(float(getattr(cfg, "_holding_horizon_periods", 30.0)), 1.0)
        sigma = self._forecast_vol(bars, horizon_periods)
        risk_inputs = _risk_inputs_from_vol(sigma, horizon_periods)
        equity = max(self._ctx.equity, 1.0)
        free_margin = max(self._ctx.free_margin, 0.0)
        idx = (sym,)
        cov = np.array([[max(sigma, 1e-6) ** 2]], dtype=float)
        book_fixture = BookStateFixture(
            equity=equity,
            peak_equity=max(self._ctx.peak_equity, equity),
            free_margin=free_margin,
            positions={},
            cov=cov,
            cov_index=idx,
            candidate_weights={},
            governor_mult=1.0,
        )
        return self._api.size_position(signal, risk_inputs, cfg, book_fixture)

    @staticmethod
    def _forecast_vol(bars: Bars | None, horizon_periods: float) -> float:
        if bars is None or len(bars) < 3:
            return 0.02
        sigma = forecast_vol(
            bars.o, bars.h, bars.l, bars.c, horizon_periods=horizon_periods
        )
        # Floor so a near-flat window never produces a divide-by-zero ceiling.
        return max(float(sigma), 1e-4)


def _risk_inputs_from_vol(forecast_vol_value: float, horizon_periods: float) -> RiskInputs:
    """Assemble the measurable :class:`RiskInputs` from the forecast vol + conservative defaults.

    The vol-derived gap/tail/MAE distributions scale with the forecast σ (a richer live data layer
    supplies trailing empirical fits; the backtest uses σ-scaled conservative stand-ins). NO edge
    quantity enters (Inv A): ``mu_hat`` is a small Kelly-only forecast, isolated to ``L_kelly``.
    """
    sigma = max(forecast_vol_value, 1e-4)
    gap = GapModel(
        worst_adverse_gap=2.0 * sigma,
        typical_gap=0.5 * sigma,
        n_gaps=40,
        class_stress_gap=3.0 * sigma,
    )
    tail = GPDFit(
        threshold=1.5 * sigma,
        shape=0.1,
        scale=sigma,
        n_exceed=40,
        tail_prob=0.05,
        ok=True,
        emp_es=3.0 * sigma,
    )
    mae = ConditionalMAE(
        quantile_by_bucket={"low_vol": 1.5 * sigma, "high_vol": 3.0 * sigma, "global": 2.0 * sigma},
        n_by_bucket={"low_vol": 50, "high_vol": 50, "global": 50},
    )
    liq = LiquidityState(
        spread_bps=2.0,
        expected_volume_exit_window=1.0e7,
        adv_notional=1.0e8,
        top_of_book_notional=1.0e6,
        impact_bps=3.0,
        stale_or_halted=False,
    )
    return RiskInputs(
        forecast_vol=sigma,
        vol_percentile=0.5,
        corr_instability=0.1,
        jump_intensity=0.5,
        gap_hist=gap,
        loss_tail=tail,
        cond_mae=mae,
        regime_bucket="global",
        financing_rate=0.0,
        liquidity=liq,
        g_t=1.0,
        mu_hat=0.02,
        holding_horizon_periods=horizon_periods,
    )


# --------------------------------------------------------------------------- TREND adapter


@dataclass
class _TrendSlot:
    engine: TrendLifeEngine
    entry_local: int  # the buffer-local index at which this position was first managed
    last_local: int  # the buffer-local index of the previous update (monotone walk)
    position: Position  # the position re-anchored into the buffer's local index space


class _RealTrend:
    """Multiplexes one validated :class:`TrendLifeEngine` per open intent, in the buffer's index space.

    Satisfies ``core.protocols.TrendEngine``: ``update(bars, position, spec, cfg, idx)``. The real
    engine is per-position stateful and EXPECTS the caller to drive ``idx`` from ``position.entry_idx``
    upward, one bar at a time, indexing into ``bars`` (it resets its exit state only when
    ``idx == entry_idx``, then walks). But the composer hands the position's GLOBAL ``entry_idx``
    (the breakout signal's index, recovered from the stop id) and ``idx = len(bars)-1`` (the rolling
    buffer's latest index) — two different index spaces. This adapter therefore ANCHORS each new
    position at the buffer-local index where it is first managed, re-anchors the position's
    ``entry_idx`` to that local index, and drives the engine with a monotone local ``idx``
    (entry_local, entry_local+1, ...) so the validated reset/step/time-cap logic runs exactly as the
    TREND suite covers. Keyed on (symbol, global entry_idx) — the position's intent — and dropped
    once the engine reports closed (returns None).
    """

    def __init__(self, cfgs: Mapping[str, Any]) -> None:
        self._cfgs = cfgs
        self._slots: dict[tuple[str, int], _TrendSlot] = {}

    def update(self, bars: Bars, position: Position, spec: InstrumentSpec, cfg: Any, idx: int):
        from dataclasses import replace as _replace

        sym = spec.symbol
        key = (sym, int(getattr(position, "entry_idx", 0)))
        resolved = self._cfgs.get(sym, cfg)
        slot = self._slots.get(key)
        current_local = len(bars) - 1
        if slot is None:
            # First time we manage this position: anchor it at the current buffer-local index and
            # re-anchor the engine's view of entry so ``_reset`` fires on this very bar.
            entry_local = current_local
            local_pos = _replace(position, entry_idx=entry_local)
            slot = _TrendSlot(
                engine=TrendLifeEngine(resolved),
                entry_local=entry_local,
                last_local=entry_local - 1,
                position=local_pos,
            )
            self._slots[key] = slot
        # Walk the engine forward one bar at a time to the current local index (never backward).
        local_idx = max(current_local, slot.last_local + 1)
        local_idx = min(local_idx, len(bars) - 1)
        decision = None
        for step_idx in range(slot.last_local + 1, local_idx + 1):
            decision = slot.engine.update(bars, slot.position, spec, resolved, step_idx)
            if decision is None:
                break
        slot.last_local = local_idx
        if decision is None:
            self._slots.pop(key, None)
        return decision


# --------------------------------------------------------------------------- the real composer


class _RealComposer(ConvexityComposer):
    """A :class:`ConvexityComposer` that publishes the per-bar bar/equity context to the adapters.

    Identical §4 composition; the ONLY override is ``on_decision_bar`` recording the current
    ``bars_by_instrument`` / ``idx_by_instrument`` and the account equity into the shared
    :class:`_BarContext` BEFORE delegating, so the stateless BREAKOUT/LEVERAGE adapters read the live
    buffer the strategy accumulated this bar.
    """

    def __init__(self, *, ctx: _BarContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ctx = ctx

    def on_decision_bar(self, **kwargs: Any):  # type: ignore[override]
        self._ctx.bars_by_instrument = kwargs.get("bars_by_instrument", {})
        self._ctx.idx_by_instrument = kwargs.get("idx_by_instrument", {})
        book = kwargs.get("book")
        if book is not None:
            self._ctx.equity = float(getattr(book, "equity", 0.0))
            self._ctx.peak_equity = float(getattr(book, "peak_equity", 0.0))
            self._ctx.free_margin = float(getattr(book, "free_margin", 0.0))
        selection = kwargs.get("selection")
        if selection is not None:
            self._ctx.eligible = tuple(getattr(selection, "eligible", ()))
        return super().on_decision_bar(**kwargs)


# --------------------------------------------------------------------------- config resolution


@dataclass(frozen=True)
class RealDepsConfigs:
    """The resolved per-pillar configs the real composition needs (one set per symbol where the
    pillar is per-instrument). ``leverage_cfg``/``trend_cfg`` are the single representative configs
    the strategy passes; the per-symbol maps drive the adapters."""

    breakout_by_symbol: Mapping[str, Any]
    leverage_by_symbol: Mapping[str, Any]
    trend_by_symbol: Mapping[str, Any]
    leverage_cfg: Any
    trend_cfg: Any


def _lev_instrument_spec(spec: InstrumentSpec) -> InstrumentLeverageSpec:
    return InstrumentLeverageSpec(
        symbol=spec.symbol,
        asset_class=spec.asset_class,
        lot_step=float(spec.lot_size),
        contract_multiplier=float(spec.price_multiplier),
        tick_size=float(spec.tick_size),
        min_notional=float(spec.min_notional),
        financing_rate_annual=0.0,
    )


def _resolve_leverage_cfg(spec: InstrumentSpec, position_margin: float, h_sys: float) -> Any:
    """Resolve the validated leverage ``ResolvedConfig`` and decorate it with the per-position
    margin + holding horizon the composer/adapter read (the real ``PolicyDefaults`` carries neither
    — they are SYSTEM/run-level, supplied here at the composition seam)."""
    lev_spec = _lev_instrument_spec(spec)
    base = leverage_resolve_config(
        lev_spec, defaults=LevPolicyDefaults(), preset=LEV_PRESETS[spec.asset_class]
    )
    return _LevCfgProxy(base, position_margin=position_margin, holding_horizon_periods=h_sys)


class _LevCfgProxy:
    """A pass-through over the real leverage ``ResolvedConfig`` that adds ``defaults.position_margin``
    (the per-position committed margin the composer reads) without mutating the frozen real config.

    The real ``PolicyDefaults`` has no ``position_margin``/``capital_cap``-default issue: ``capital_cap``
    IS a RED default, ``position_margin`` is run-level (account fraction). This proxy forwards every
    real attribute/property the engine reads and overlays ``defaults.position_margin`` +
    ``_holding_horizon_periods`` (consumed by :class:`_RealLeverage`)."""

    def __init__(
        self,
        base: LevResolvedConfig,
        *,
        position_margin: float,
        holding_horizon_periods: float,
    ) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "defaults", _DefaultsProxy(base.defaults, position_margin))
        object.__setattr__(self, "_holding_horizon_periods", holding_horizon_periods)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base"), name)


class _DefaultsProxy:
    """Forwards every real ``PolicyDefaults`` attribute and adds ``position_margin``."""

    def __init__(self, base: Any, position_margin: float) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "position_margin", float(position_margin))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base"), name)


def build_real_deps_configs(
    specs: Mapping[str, InstrumentSpec],
    *,
    h_sys: float = 30.0,
    position_margin: float = 5_000.0,
) -> RealDepsConfigs:
    """Resolve breakout/leverage/trend configs for every universe symbol via each pillar's own
    ``config/resolve.py`` (the single merge point, Inv C/D). ``h_sys`` is the SYSTEM operational hold
    cap injected into TREND (Inv HC) and used as the leverage holding horizon; ``position_margin`` is
    the per-position committed margin the LEVERAGE budget is a fraction of."""
    breakout_by_symbol: dict[str, Any] = {}
    leverage_by_symbol: dict[str, Any] = {}
    trend_by_symbol: dict[str, Any] = {}
    for sym, spec in specs.items():
        breakout_by_symbol[sym] = breakout_resolve_config(spec)
        leverage_by_symbol[sym] = _resolve_leverage_cfg(spec, position_margin, h_sys)
        trend_by_symbol[sym] = trend_resolve_config(spec, h_sys=h_sys)
    first = next(iter(specs))
    return RealDepsConfigs(
        breakout_by_symbol=breakout_by_symbol,
        leverage_by_symbol=leverage_by_symbol,
        trend_by_symbol=trend_by_symbol,
        leverage_cfg=leverage_by_symbol[first],
        trend_cfg=trend_by_symbol[first],
    )


def build_real_composer(
    *,
    specs: Mapping[str, InstrumentSpec],
    configs: RealDepsConfigs,
    with_ops: bool = True,
    **_ignored: Any,
) -> tuple[ConvexityComposer, Any]:
    """Assemble a :class:`_RealComposer` from the REAL eight-pillar engines (ARCHITECTURE §4).

    The breakout/leverage/trend seams are the validated pillar engines (NOT the synthetic stubs);
    execution + ops are the same ``build_engine``/``OpsAdapter`` the live node uses. Returns the
    ``(composer, breakout)`` tuple matching the synthetic factory's shape so ``run/backtest.py``'s
    ``composer_factory`` seam accepts it unchanged (the breakout has no ``idx`` clock — the
    clock-attach is a no-op for the real engine)."""
    ctx = _BarContext()
    breakout = _RealBreakout(ctx, configs.breakout_by_symbol)
    leverage = _RealLeverage(ctx)
    trend = _RealTrend(configs.trend_by_symbol)
    composer = _RealComposer(
        ctx=ctx,
        breakout=breakout,
        leverage=leverage,
        trend=trend,
        execution=build_exec_engine(),
        ops_adapter=OpsAdapter(build_ops_engine()) if with_ops else None,
    )
    return composer, breakout
