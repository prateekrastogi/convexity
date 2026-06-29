"""Frozen dataclasses + Literal aliases crossing every TREND boundary (TREND §6).

Only the cross-PILLAR handoffs live in :mod:`convexity.core.types` and are NOT redefined here:
``Position`` (BREAKOUT -> TREND, Inv O), ``ExecutionIntent`` / ``HoldHorizonEstimate`` /
``ResidualMoveEstimate`` (TREND -> EXECUTION/LEVERAGE/COST, Inv N/P), ``Bars`` / ``InstrumentSpec`` /
``CostModel``. This module holds the TREND-INTERNAL rich objects the estimate/policy layers build and
pass between themselves: ``ClockState``, ``HorizonState``, ``VolState``, ``BeliefDiagnostics``,
``BeliefState``, ``ExitState``, and the live ``ActionDecision`` emitted ONLY by
``policy/decision.py`` (Inv A).

Pure-Python + numpy-typed, NautilusTrader-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

# Re-exported from the shared core so TREND code imports the boundary objects from one place.
from convexity.core.types import (  # noqa: F401  (re-export)
    ExecutionIntent,
    HoldHorizonEstimate,
    Position,
    ResidualMoveEstimate,
    Side,
)

# --------------------------------------------------------------------------- type aliases (§6)

Action = Literal["continue", "reduce", "exit"]
DetectorKind = Literal["sprt", "cusum"]
VolMethod = Literal["ewma", "mad", "yang_zhang", "hybrid"]
HorizonClock = Literal[
    "calendar_days",
    "trading_sessions",
    "active_minutes",
    "bars_since_entry",
]
# §6 — every stopping event is reason-coded. The trend-death events (jump_against/seq_alarm/
# trail_breach) are deaths; time_cap is operational (Inv HC); data_censored/censored are
# data/observation-cap censoring.
StopCause = Literal[
    "jump_against",
    "seq_alarm",
    "trail_breach",
    "time_cap",
    "data_censored",
    "censored",
]
HazardRung = Literal["state", "single", "constant", "disabled"]


# --------------------------------------------------------------------------- clock / horizon (§1)

@dataclass(frozen=True)
class ClockState:
    """The four horizon clocks' elapsed values for one open position (§1.2)."""

    elapsed_calendar_ms: int
    elapsed_sessions: int
    elapsed_active_minutes: int
    bars_since_entry: int


@dataclass(frozen=True)
class HorizonState:
    """Causal dynamic-horizon path/cap state (§1, §9 horizon). All ``F_t``-measurable.

    ``M_t`` = ``running_favorable_excursion`` (running favorable high-watermark, NOT future MFE);
    ``G_t`` = ``running_giveback``; ``V_t`` = ``integrated_var_since_entry`` (§1.3). The cap is the
    SYSTEM-injected operational bound (``tau_cap = H_sys``, §1.6a / Inv HC) under ``active_clock``.
    """

    active_clock: HorizonClock
    horizon_cap: float
    elapsed_value: float
    cap_fraction_used: float
    cap_remaining: float
    integrated_var_since_entry: float
    running_favorable_excursion: float
    running_giveback: float
    cap_binding_warning: bool


# --------------------------------------------------------------------------- volatility (§9)

@dataclass(frozen=True)
class VolState:
    """Robust causal volatility state (§9 volatility). ``integrated_var_since_entry`` is monotone
    nondecreasing after entry; ``gap_var`` is applied once per session boundary."""

    method: VolMethod
    sigma_log: float
    sigma_price: float
    integrated_var_since_entry: float
    gap_var: float
    vol_bucket: int
    liq_bucket: int
    tail_nu: float
    quality_flag: str


# --------------------------------------------------------------------------- beliefs (§6, §9)

@dataclass(frozen=True)
class BeliefDiagnostics:
    """Log-only diagnostics (Inv F): policy decisions must not change when these are perturbed."""

    innovation_whiteness: float
    kalman_consistency: float
    detector_margin: float
    jump_flag_rate: float
    hazard_rung: HazardRung
    effective_sample_size: int
    tail_nu: float
    vol_method: VolMethod
    vol_quality_flag: str


@dataclass(frozen=True)
class BeliefState:
    """The full causal belief for one bar (§6, §9 state). Built ONLY by ``estimate/state.py``;
    carries NO action (Inv A) and NO drift posterior (Inv D — there is no ``P(trend_alive)``)."""

    idx: int
    clock: ClockState
    horizon: HorizonState
    vol: VolState
    filtered_slope: float
    slope_var: float
    z_drift: float
    adverse_excursion_r: float
    favorable_excursion_r: float
    seq_stat: float
    seq_alarm: bool
    jump_against: bool
    jump_with: bool
    hazard: float
    hazard_model: HazardRung
    reason_signals: Mapping[str, float]
    diag: BeliefDiagnostics


# --------------------------------------------------------------------------- exit / trail (§10)

@dataclass(frozen=True)
class ExitState:
    """Asymmetric convex-trail state (§10). ``invalidation`` is the trigger level EXECUTION rests
    on the venue (Inv N); ``trail_multiple`` is the open-profit-dependent multiplier (monotone in
    ``open_profit_r``, §10 / INV-CVX); ``size_fraction`` is scaled by the reduce path only."""

    trail_level: float
    trail_multiple: float
    high_watermark: float
    low_watermark: float
    open_profit_r: float
    initial_risk: float
    invalidation: float
    size_fraction: float


# --------------------------------------------------------------------------- live decision (§6)

@dataclass(frozen=True)
class ActionDecision:
    """The ONLY live action TREND emits, built ONLY by ``policy/decision.py`` (Inv A).

    ``decision_idx`` is the bar the decision becomes known; ``action_idx`` is the first bar the
    action can be evaluated/executed (normally ``decision_idx + 1``, §6 / Inv J).
    ``stopping_time_fired`` is True only at ``tau_live``; ``stop_cause`` reason-codes whether the
    stop was trend evidence, operational (``time_cap``, Inv HC), or censoring. ``hold_horizon`` and
    ``residual_move`` are the TREND-owned outputs LEVERAGE/COST consume read-only (Inv P);
    ``execution_intent`` is the exit/trail intent EXECUTION executes (Inv N).
    """

    idx: int
    decision_idx: int
    action_idx: int
    action: Action
    horizon: HorizonState
    stopping_time_fired: bool
    stop_cause: StopCause | None
    reason_codes: tuple[str, ...]
    exit_state: ExitState
    size_fraction: float
    hold_horizon: HoldHorizonEstimate
    residual_move: ResidualMoveEstimate
    action_price_kind: Literal["next_open", "next_mid", "close_if_live_executable"]
    execution_intent: ExecutionIntent
    engine_version: str
