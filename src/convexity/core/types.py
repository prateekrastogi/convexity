"""Shared cross-pillar data types — the single source of the shapes that cross pillar boundaries.

Every spec defines its own rich internal dataclasses, but the spec ``risks`` sections agree on one
thing: the types that flow *between* pillars (Side, Bars, InstrumentSpec, CostModel, Position,
HoldHorizon, the Urgency/Direction/SignalStatus enums, the EntrySignal/ExecTarget/LeverageDecision/
RegimeState handoffs) MUST be a single shared core, not redefined per pillar, or the pillars cannot
interoperate. This module is that core.

Design rules (from the architecture principle in the build directive):
  * Pure-Python + numpy-typed only. NautilusTrader-FREE so every pillar's policy logic is unit-
    testable without the engine. The thin ``strategy/`` adapters are the only Nautilus-facing code.
  * Frozen dataclasses with explicit, minimal field sets. Each cross-pillar handoff carries ONLY
    the fields the consumer is allowed to read (the per-spec Invariant-A / Invariant-K firewalls:
    e.g. ``ExecTarget`` deliberately carries no edge score; ``EntrySignal`` exposes only direction/
    status/urgency/price-levels downstream).
  * Time is int64 epoch-milliseconds, sorted ascending. Arrays are ``np.ndarray``.

Where a pillar needs a richer object than the cross-pillar handoff (e.g. BREAKOUT's full
``EvidenceFrame``, TREND's ``BeliefState``, LEVERAGE's seven ``Ceilings``), that lives in the
pillar's own ``<pillar>/types.py``; only the boundary objects live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np

# --------------------------------------------------------------------------- type aliases

# Shared across BREAKOUT/TREND/LEVERAGE/EXECUTION/COST/SYSTEM. -1 short, +1 long. (0 = none only
# in BREAKOUT's Direction, which keeps the 0 for "no candidate".)
Side = Literal[-1, 1]
Direction = Literal[-1, 0, 1]

# Six asset classes (BREAKOUT §4) — drives barrier-family policy + TEAL bucketing. TREND/LEVERAGE/
# EXECUTION/COST/REGIME use overlapping subsets; the convexity repo trades only the crypto_perp
# path live but the spec requires the full set faithfully.
AssetClass = Literal[
    "commodity_future",
    "precious_metal",
    "industrial_metal",
    "us_equity",
    "crypto_spot",
    "crypto_perp",
]

# Only "entry" is downstream-actionable (BREAKOUT Inv K). SYSTEM branches on status=="entry".
SignalStatus = Literal["candidate", "entry"]

# BREAKOUT entries are always "urgent"; EXECUTION reads the 4-level ladder off ExecTarget.
Urgency = Literal["urgent", "patient"]
ExecUrgency = Literal["PATIENT", "NORMAL", "URGENT", "IMMEDIATE"]

# TREND's live actions and EXECUTION's tactic ladder.
TrendActionKind = Literal["continue", "reduce", "exit"]
Tactic = Literal["POST_ONLY_MAKER", "LIMIT_IOC", "MARKETABLE_LIMIT", "MARKET"]

# REGIME's four ordered market-state labels (REGIME.md §4.1, post-pivot). Ordered by severity
# HEALTHY < WATCH < STRESSED < HALT. Migrated from the pre-pivot CALM/NORMAL/STRESSED/CRISIS
# ladder (which was tuned for a single-instrument directional-breakout book that is now proven
# dead); the cross-sectional momentum book gates on a continuous g_t floored at exactly 0 (HALT),
# not the old discrete {1,.75,.5,.25}-floored-.25 ladder (REGIME.md §0 P7, §4.1).
RegimeStateName = Literal["HEALTHY", "WATCH", "STRESSED", "HALT"]

# Which detector drove the current g_t (REGIME.md §3.3/§4 reason-chain vocabulary). The live
# g_t-affecting CUSUMs are the dispersion up/down and BTC-vol up sides (§3.3: S_V^- is retained
# for diagnostics only and never gates g_t); the drawdown state machine (§5) is the fourth source.
RegimeDetector = Literal[
    "dispersion_cusum_upper",
    "dispersion_cusum_lower",
    "btc_vol_cusum_upper",
    "drawdown_stop",
    "insufficient_breadth",
    "none",
]

# OPS operational state, totally ordered RUN < DEGRADE < HALT for max-severity precedence
# (OPS.md sec 9). Migrated in place from the pre-pivot ``KillState`` alias
# (``Literal["run","degrade","halt"]``, single-instrument-governor era) — the cross-sectional
# book emits UPPER-cased states matching this spec's OpsDirective shape (OPS.md sec 9 step 1-2).
OpsState = Literal["RUN", "DEGRADE", "HALT"]

# The fired-alarm taxonomy OPS attributes every directive to (OPS.md sec 9 Alarm.kind). One
# closed set shared by reconciliation (sec 1.4), the CUSUM kill-switch (sec 4.3-4.4), the SLA
# monitor (sec 5.3), and the heartbeat trigger (sec 2).
AlarmKind = Literal[
    "RECON_BREACH",
    "RECON_STALE",
    "CUSUM_ALARM",
    "CUSUM_WARN",
    "SLA_BREACH",
    "SLA_BREACH_SEVERE",
    "ANOMALOUS_OUTPERFORMANCE",
    "HEARTBEAT_SILENCE",
]


# --------------------------------------------------------------------------- market data

@dataclass(frozen=True)
class Bars:
    """Point-in-time OHLCV+ panel shared by every data-consuming pillar (BREAKOUT/TREND §4).

    ``ts`` is int64 epoch-ms, sorted ascending. ``o/h/l/c`` required; the rest optional (None when
    the feed lacks them — e.g. a 1-min kline feed has no bid/ask, so the spread estimators fall
    through their ladders). Flag arrays (``is_*``) are attached by the data layer's validator/
    sessionizer and are read-only thereafter.
    """

    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741 — OHLC 'low'; the o/h/l/c single-letter bar convention is deliberate
    c: np.ndarray
    v: np.ndarray | None = None
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    trade_count: np.ndarray | None = None
    is_session_open: np.ndarray | None = None
    is_valid: np.ndarray | None = None
    is_halt: np.ndarray | None = None
    is_roll_window: np.ndarray | None = None
    funding_rate: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.ts.shape[0])


@dataclass(frozen=True)
class InstrumentSpec:
    """AMBER per-instrument market metadata — the ONLY config object that may travel per-symbol.

    Invariant C (every pillar): this holds AMBER metadata ONLY. No RED/BLUE field (ARL0, threshold,
    horizon, lookback, weight, stop policy, Kelly fraction, chase χ, governor breakpoint, ES level,
    grid) may appear here. ``data_policy``/``optional_sources`` describe RTH/ETH/24-7, roll rule, CA
    source, funding availability, OFI availability — capability metadata, not modelling knobs.
    """

    symbol: str
    asset_class: AssetClass
    venue: str
    timezone: str
    tick_size: float
    lot_size: float
    price_multiplier: float = 1.0
    calendar_id: str = ""
    min_notional: float = 0.0
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    maker_rebate_bps: float = 0.0
    funding_interval_ms: int | None = None  # perps only; None for non-funding instruments
    funding_cap: float | None = None
    data_policy: Mapping[str, str] = field(default_factory=dict)
    optional_sources: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class CostModel:
    """Validation-friction STUB (BREAKOUT/TREND §6.1). SUPERSEDED by the COST pillar estimators for
    any net-of-cost gate (COST P1 "one cost truth"). Kept only for isolated unit tests + baselines
    that do not touch a promotion gate."""

    spread_bps: float
    slippage_bps: float
    fees_bps: float
    min_tick_slippage: float


@dataclass(frozen=True)
class HoldHorizon:
    """A RANGE of remaining hold, never a scalar (TREND Inv P, COST §5.5d).

    Hold length is unmeasurable, so the gate prices funding at the conservative ``m_max`` stamp
    count. ``m_max`` is derived from the quantile + funding interval via :meth:`from_quantile` so
    the field can never disagree with ``ceil(max_hold_quantile_ms / funding_interval_ms)`` (COST
    risk note: HoldHorizon producer ambiguity). ``expected_m_caller`` is DIAGNOSTIC ONLY and cannot
    relax the gate (COST Inv M-hold).
    """

    max_hold_quantile_ms: int
    funding_interval_ms: int
    m_max: int
    expected_m_caller: int | None = None

    @classmethod
    def from_quantile(
        cls,
        max_hold_quantile_ms: int,
        funding_interval_ms: int,
        expected_m_caller: int | None = None,
    ) -> "HoldHorizon":
        """Construct with ``m_max`` computed so it can never disagree with the ceil() rule."""
        m_max = -(-max_hold_quantile_ms // funding_interval_ms)  # ceil division
        return cls(max_hold_quantile_ms, funding_interval_ms, m_max, expected_m_caller)

    @property
    def n_funding_stamps_remaining(self) -> int:
        return self.m_max


# --------------------------------------------------------------------------- BREAKOUT -> SYSTEM/EXEC

@dataclass(frozen=True)
class EntrySignal:
    """The cross-pillar entry emitted by BREAKOUT's SignalAssembler (BREAKOUT §4, Inv A/K).

    Only ``detect/combine.py`` may construct it (Inv A). Downstream (SYSTEM breakout_adapter,
    EXECUTION) reads ONLY ``direction``, ``status=='entry'``, ``urgency`` and the price levels —
    ``quality_score`` magnitude is NOT a downstream ranking input (Inv K score-magnitude firewall).
    ``evidence`` + ``reason_codes`` make every signal decomposable (Inv H).
    """

    symbol: str
    signal_idx: int
    entry_idx: int
    direction: Direction
    status: SignalStatus
    quality_score: float
    threshold: float
    entry_ref_price: float
    invalidation: float
    stop_distance: float
    barrier_price: float
    range_width_sigma: float
    compression_pct: float
    false_break_risk: float
    urgency: Urgency
    evidence: Mapping[str, float]
    reason_codes: tuple[str, ...]
    detector_version: str


@dataclass(frozen=True)
class Position:
    """An already-open position handed from BREAKOUT's entry gate to TREND (TREND §6c, Inv O).

    TREND consumes read-only and NEVER recomputes entry. ``initial_invalidation``/``initial_risk``
    are BREAKOUT-owned; TREND's ConvexExit uses them when present, falling back to vol/swing only
    when absent.
    """

    side: Side
    entry_idx: int
    entry_price: float
    initial_invalidation: float | None = None
    initial_risk: float | None = None


# --------------------------------------------------------------------------- LEVERAGE I/O

@dataclass(frozen=True)
class SignalInput:
    """Qualified upstream signal handed to LEVERAGE (LEVERAGE types.py, Inv A).

    Deliberately carries NO edge field (trend_confidence/expected_return/breakout_score): the
    edge/risk firewall is enforced by the *absence* of fields. ``capital_cap`` is c (0.12–0.15 of
    position_margin, RED-sourced) — the L_stop numerator.
    """

    instrument: str
    side: Side
    entry_price: float
    position_margin: float
    capital_cap: float


@dataclass(frozen=True)
class LeverageDecision:
    """LEVERAGE engine output (LEVERAGE types.py). The size SYSTEM/EXECUTION act on.

    ``binding_ceiling`` names which of the seven ceilings (or governor/unit/class-cap) bound the
    min(); a regime cut surfaces as ``binding_ceiling=='portfolio'`` with reason_code
    'regime_heat_cut' (NOT a 'regime' literal). ``g_t`` + ``governor_multiplier`` are echoed for the
    NODOUBLE audit (Inv R): regime reads market-state, governor reads equity-drawdown, disjoint axes.
    """

    instrument: str
    leverage: float
    size: float
    rounded_size: float
    binding_ceiling: Literal[
        "stop", "vol", "kelly", "tail", "portfolio", "margin", "liquidity", "governor", "unit"
    ]
    kelly_fraction: float
    governor_multiplier: float
    g_t: float
    risk_budget: float
    planned_loss_on_margin: float
    planned_loss_on_equity: float
    reason_codes: tuple[str, ...]
    engine_version: str


# --------------------------------------------------------------------------- TREND -> EXEC/LEV/COST

@dataclass(frozen=True)
class ExecutionIntent:
    """Exit/trail INTENT TREND hands to EXECUTION (TREND §6a, Inv N). TREND never places orders.

    ``amend_stop`` on a trail move; ``exit_reduce_only`` + ``urgency=='immediate'`` on a hard exit;
    ``none`` on continue.
    """

    invalidation_level: float
    reduce_only: bool
    urgency: Literal["passive", "immediate"]
    kind: Literal["amend_stop", "exit_reduce_only", "none"]


@dataclass(frozen=True)
class HoldHorizonEstimate:
    """TREND-owned conservative hold-horizon RANGE, consumed read-only by LEVERAGE (σ·√T) and COST
    (funding gate m_max) (TREND §6b, Inv P). A range, never a point scalar."""

    active_clock: Literal["calendar_days", "trading_sessions", "active_minutes", "bars_since_entry"]
    expected_remaining_lo_ms: int
    expected_remaining_hi_ms: int
    max_hold_quantile_ms: int
    quantile_used: float
    op_cap_remaining_ms: int


@dataclass(frozen=True)
class ResidualMoveEstimate:
    """TREND-owned expected residual FAVORABLE move — a band with one conservative point that COST's
    funding-aware hold gate consumes read-only (TREND §6b / COST §8.2, Inv P)."""

    expected_residual_move_lo: float
    expected_residual_move_hi: float
    point_used_by_cost: float
    units: Literal["initial_risk", "robust_vol"]


# --------------------------------------------------------------------------- EXECUTION handoff

@dataclass(frozen=True)
class ExecTarget:
    """The fully-DECIDED trade handed to EXECUTION (EXECUTION types.py, Inv A).

    Carries NO breakout_score / expected_return / trend_confidence / kelly_fraction — execution
    chooses the *tactic*, never the trade. ``target_qty`` is signed; |qty| is never exceeded.
    ``arrival_mid`` is the IS benchmark, fixed for the intent's life. ``invalidation`` + its
    generation (owned by TREND) version the venue stop's id.
    """

    intent_id: str
    instrument: str
    side: Side
    target_qty: float
    arrival_mid: float
    invalidation: float
    urgency: ExecUrgency
    reduce_only: bool


# --------------------------------------------------------------------------- COST -> consumers

@dataclass(frozen=True)
class CostComponent:
    """One additive cost line, dual-unit (COST types.py, Inv B). ``is_fact`` separates realized
    FACT from forecast ESTIMATE (Inv A)."""

    name: str
    quote_ccy: float
    bps_of_notional: float
    is_fact: bool
    reason: str


@dataclass(frozen=True)
class RoundTripCost:
    """Additive cost decomposition (COST §5, Inv B): ``total_quote_ccy == sum(c.quote_ccy)`` exactly.

    ``funding_forecast`` is the POINT blend (telemetry); ``funding_band_conservative`` is the
    side-selected CI-edge band that the net-EV gate actually sums (Inv F-band-gate).
    """

    components: tuple[CostComponent, ...]
    total_quote_ccy: float
    total_bps: float
    funding_forecast: float
    funding_band_low: float
    funding_band_high: float
    funding_band_conservative: float
    cost_fraction: float | None = None


# --------------------------------------------------------------------------- REGIME -> LEVERAGE/SEL

@dataclass(frozen=True)
class RegimeState:
    """REGIME's only output (REGIME.md §4, §8), post-pivot cross-sectional-momentum shape.

    A single scalar de-risking multiplier ``g_t`` on the book's target gross exposure for the NEXT
    weekly rebalance, plus the labeled market state and a decomposable reason chain (§0 P7). This
    supersedes the pre-pivot shape (``g_t`` in {1,.75,.5,.25} floored .25, ``binding_sub`` over
    vol/corr/turb/liq, bar-indexed) which was tuned for a single-instrument directional-breakout
    book (proven dead). Key contract changes, per REGIME.md §4.1 / §0 P2:

      * ``g_t`` is CONTINUOUS in ``[0, 1]`` and floored at exactly ``0`` on HALT (§4.1), not a
        four-value ladder floored at .25. The state's band FLOOR is the conservative live value
        emitted (§4.1: HEALTHY=1.00, WATCH=0.75, STRESSED=0.40, HALT=0.00).
      * Enters the downstream sizing layer through the SINGLE ``g_t`` channel, consumed exactly
        once multiplicatively (``gross_live = g_t · gross_target``; once LEVERAGE is built,
        ``gross_live = g_t · G_t``) — never a second ceiling, never a per-instrument value,
        never additive (§0 P2).
      * The risk unit is NEXT WEEK's portfolio return; ``as_of`` indexes the weekly Friday-close
        rebalance grid (``t``), NOT a bar count (§0 P1, notation note).

    ``binding_detector`` names which detector drove the current state (§4 reason chain). ``s_stat``
    is the triggering CUSUM statistic value, ``h_threshold`` the decision interval it crossed
    (both 0.0 when HEALTHY / no alarm). ``dwell_weeks`` counts consecutive weeks in the current
    state (§4.2 re-arm hysteresis). ``transition`` records the last state move.
    """

    g_t: float
    state: RegimeStateName
    binding_detector: RegimeDetector
    s_stat: float
    h_threshold: float
    dwell_weeks: int
    transition: Literal["escalate", "deescalate", "hold"]
    reason_codes: tuple[str, ...]
    as_of: int
    engine_version: str


# --------------------------------------------------------------------------- OPS output

@dataclass(frozen=True)
class Alarm:
    """One fired OPS alarm, fully attributed (OPS.md sec 9).

    ``detail`` is human-readable attribution (which instrument / value / threshold); ``value`` is
    the triggering statistic's value and ``threshold`` the limit it crossed. Frozen because it
    crosses the OPS -> live-loop boundary as part of ``OpsDirective.alarms``."""

    kind: AlarmKind
    detail: str
    value: float
    threshold: float
    as_of: int


@dataclass(frozen=True)
class OpsDirective:
    """The one thing OPS emits (OPS.md sec 9, P3). Totally ordered RUN < DEGRADE < HALT.

    SUPERSEDES the pre-pivot ``OpsDirective`` (``state: KillState`` + ``flatten``/
    ``cancel_entries``/``reason_codes``/``engine_version``) that the retired single-instrument
    governor emitted — migrated in place per OPS.md sec 9 step 1 so ``core/protocols.py``'s
    ``OpsEngine`` Protocol and every consumer resolve to ONE shape, never two same-named classes.
    ``gross_multiplier`` is 1.0 at RUN, ``g_degrade`` at DEGRADE, 0.0 at HALT; ``alarms`` is the
    reason chain (empty at RUN); ``as_of`` is epoch-ms of the producing observation."""

    state: OpsState
    gross_multiplier: float
    alarms: tuple["Alarm", ...]
    as_of: int
