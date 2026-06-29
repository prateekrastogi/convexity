"""Shared fixtures for the VM-only integration suite (``@pytest.mark.nautilus``).

Everything here is SYNTHETIC and deterministic — no live connectivity. We build:
  * a rolling synthetic :class:`~convexity.core.types.Bars` panel,
  * SCRIPTED pillar mocks that satisfy the ``core.protocols`` seams (the orchestrator + composition
    gate are validated against scripted module outputs per SYSTEM §15; the real pillar math is
    covered by the 613-test pillar suite),
  * a REAL EXECUTION engine (pure-Python) so the tactic→OrderCommand lowering is exercised end to end,
  * a :class:`ConvexityComposer` wiring them together,
  * a real ``nautilus_trader`` ``BacktestEngine`` with the Binance USDT-M perp instrument so the venue
    actually accepts the reduce-only protective stop the composer emits.

A module-level ``nautilus`` marker + an importorskip means the whole package is cleanly skipped where
nautilus is not installed (the pure-Python pillar venv).
"""

from __future__ import annotations

import numpy as np
import pytest

# The conftest imports ONLY pure-Python convexity modules (no nautilus), so it loads in the pillar
# venv too. The actual nautilus dependency lives in the test MODULES, each of which guards itself
# with ``pytest.importorskip("nautilus_trader")`` + the ``nautilus`` marker, so the whole package is
# cleanly SKIPPED (never an error) where nautilus is not installed (ARCHITECTURE §8.4).

from convexity.core.types import (  # noqa: E402
    Bars,
    EntrySignal,
    InstrumentSpec,
    Position,
)
from convexity.execution.config.resolve import resolve_config as exec_resolve_config  # noqa: E402
from convexity.execution.engine_api import build_engine as build_exec_engine  # noqa: E402
from convexity.execution.synth.generate import perp_instrument  # noqa: E402
from convexity.execution.types import PositionState, VenueStop  # noqa: E402
from convexity.leverage.types import LeverageDecision  # noqa: E402
from convexity.ops.nautilus.adapter import OpsAdapter  # noqa: E402
from convexity.ops.ops_api import build_engine as build_ops_engine  # noqa: E402
from convexity.strategy.convexity_strategy import (  # noqa: E402
    ConvexityComposer,
    OpenPositionView,
    intent_id_for,
)
from convexity.system.config.resolve import resolve_config as system_resolve_config  # noqa: E402
from convexity.system.config.universe import load_universe_meta  # noqa: E402

# The pure-Python pillar symbol (matches the EXECUTION synth perp spec). The Nautilus engine maps it
# to the venue instrument id "BTCUSDT-PERP.BINANCE" only in the engine-level stop-placement test.
SYMBOL = "BTCUSDT"
NAUTILUS_INSTRUMENT_ID = "BTCUSDT-PERP.BINANCE"


# --------------------------------------------------------------------------- synthetic market data


def synthetic_bars(n: int = 64, *, start_mid: float = 30_000.0, seed: int = 7) -> Bars:
    """A deterministic 1-min synthetic OHLCV panel (no external data; ARCHITECTURE §8)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 5.0, size=n).cumsum()
    close = start_mid + steps
    high = close + rng.uniform(1.0, 8.0, size=n)
    low = close - rng.uniform(1.0, 8.0, size=n)
    open_ = np.concatenate([[start_mid], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) * 60_000) + 1_700_000_000_000
    vol = rng.uniform(10.0, 100.0, size=n)
    return Bars(ts=ts, o=open_, h=high, l=low, c=close, v=vol)


# --------------------------------------------------------------------------- instrument metadata


def perp_spec() -> InstrumentSpec:
    """The pure-Python AMBER metadata for the perp (mirrors the EXECUTION synth instrument)."""
    return perp_instrument()


def exec_cfg():
    return exec_resolve_config(perp_spec())


def sys_cfg():
    return system_resolve_config(load_universe_meta([]))


# --------------------------------------------------------------------------- scripted pillar mocks


class ScriptedBreakout:
    """A BREAKOUT mock satisfying ``core.protocols.BreakoutModule`` — emits one scripted entry."""

    def __init__(self, signal: EntrySignal | None) -> None:
        self._signal = signal

    def propose(self, instrument: InstrumentSpec) -> EntrySignal | None:
        return self._signal


class ScriptedLeverage:
    """A LEVERAGE mock satisfying ``core.protocols.LeverageEngine`` — fixed safe size."""

    def __init__(self, rounded_size: float = 2.0, risk_budget: float = 0.06,
                 leverage: float = 0.5, planned_loss_on_equity: float = 5.0) -> None:
        self._size = rounded_size
        self._risk = risk_budget
        self._lev = leverage
        self._loss = planned_loss_on_equity

    def size_position(self, signal, risk, cfg, book) -> LeverageDecision:
        return LeverageDecision(
            instrument=signal.instrument,
            leverage=self._lev,
            size=self._size,
            rounded_size=self._size,
            binding_ceiling="stop",
            kelly_fraction=0.1,
            governor_multiplier=1.0,
            g_t=1.0,
            risk_budget=self._risk,
            planned_loss_on_margin=self._loss,
            planned_loss_on_equity=self._loss,
            reason_codes=("scripted",),
            engine_version="scripted-lev",
        )


class _ScriptedDecision:
    """A minimal TREND ``ActionDecision`` view (only the boundary fields the composer reads)."""

    def __init__(self, action: str, *, size_fraction: float = 1.0, intent=None) -> None:
        self.action = action
        self.size_fraction = size_fraction
        self.stop_cause = None
        self.reason_codes = (f"trend_{action}",)
        self.execution_intent = intent


class _ScriptedIntent:
    def __init__(self, kind: str, invalidation_level: float = 0.0) -> None:
        self.kind = kind
        self.invalidation_level = invalidation_level


class ScriptedTrend:
    """A TREND mock satisfying ``core.protocols.TrendEngine`` — replays a scripted action."""

    def __init__(self, action: str = "continue", *, size_fraction: float = 1.0,
                 intent=None) -> None:
        self._action = action
        self._frac = size_fraction
        self._intent = intent

    def update(self, bars, position, spec, cfg, idx):
        if self._action == "closed":
            return None
        return _ScriptedDecision(self._action, size_fraction=self._frac, intent=self._intent)


# --------------------------------------------------------------------------- builders


def make_entry(direction: int = 1, *, entry_idx: int = 60, mid: float = 30_000.0) -> EntrySignal:
    """A scripted downstream-actionable breakout entry (status=='entry')."""
    invalidation = mid - 200.0 if direction == 1 else mid + 200.0
    return EntrySignal(
        symbol=SYMBOL,
        signal_idx=entry_idx - 1,
        entry_idx=entry_idx,
        direction=direction,  # type: ignore[arg-type]
        status="entry",
        quality_score=1.0,
        threshold=0.5,
        entry_ref_price=mid,
        invalidation=invalidation,
        stop_distance=abs(mid - invalidation),
        barrier_price=mid + direction * 10.0,
        range_width_sigma=2.0,
        compression_pct=0.3,
        false_break_risk=0.1,
        urgency="urgent",
        evidence={"glr": 3.0},
        reason_codes=("scripted_entry",),
        detector_version="scripted",
    )


def build_composer(
    *,
    breakout=None,
    leverage=None,
    trend=None,
    with_ops: bool = False,
) -> ConvexityComposer:
    """Wire a :class:`ConvexityComposer` from scripted pillars + a REAL EXECUTION engine."""
    ops_adapter = OpsAdapter(build_ops_engine()) if with_ops else None
    return ConvexityComposer(
        breakout=breakout or ScriptedBreakout(make_entry()),
        leverage=leverage or ScriptedLeverage(),
        trend=trend or ScriptedTrend("continue"),
        execution=build_exec_engine(),
        ops_adapter=ops_adapter,
    )


def open_position_view(
    *,
    side: int = 1,
    qty: float = 2.0,
    entry_price: float = 30_000.0,
    entry_idx: int = 10,
    invalidation: float = 29_800.0,
    generation: int = 0,
    venue_stops: tuple[VenueStop, ...] = (),
) -> OpenPositionView:
    """An OPEN-position projection the composer reconciles / manages (Invariant N)."""
    inst = SYMBOL
    iid = intent_id_for(inst, entry_idx)
    return OpenPositionView(
        instrument=inst,
        intent_id=iid,
        state=PositionState(instrument=inst, side=side, qty=qty, entry_price=entry_price),  # type: ignore[arg-type]
        trend_position=Position(side=side, entry_idx=entry_idx, entry_price=entry_price,  # type: ignore[arg-type]
                                initial_invalidation=invalidation),
        invalidation=invalidation,
        invalidation_generation=generation,
        held_risk_weight=0.05,
        entry_idx=entry_idx,
        venue_stops=venue_stops,
    )


@pytest.fixture
def bars() -> Bars:
    return synthetic_bars()


@pytest.fixture
def specs() -> dict[str, InstrumentSpec]:
    return {SYMBOL: perp_spec()}


def perp_micro(mid: float = 30_000.0):
    """A causal MicroState for the perp (closed-form logistic fill curve, EXECUTION microstate)."""
    from convexity.execution.microstate.fill_prob import logistic_fill_curve
    from convexity.execution.types import MicroState

    tick = 0.1
    return MicroState(
        mid=mid,
        half_spread_bps=1.0,
        spread_rung="quoted",
        sigma_horizon_bps=5.0,
        volume_proxy=5_000_000.0,
        drift_g=0.2,
        tick_bps=10_000.0 * tick / mid,
        fill_prob_curve=logistic_fill_curve(base_rate=0.8, sigma_bps=5.0,
                                            tick_bps=10_000.0 * tick / mid),
        funding_aware=True,
        predicted_funding_rate=0.0,
        secs_to_next_funding=float("inf"),
        halted=False,
        stale=False,
        rate_limit_ok=True,
        fill_prob_rung="bucket",
        fill_prob_ci_halfwidth=0.02,
        fill_prob_calib_error=0.0,
        cold_start=False,
    )


@pytest.fixture
def micro_by_instrument():
    return {SYMBOL: perp_micro()}
