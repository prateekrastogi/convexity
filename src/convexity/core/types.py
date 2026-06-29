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

# REGIME's four ordered global states + its multiplier floor.
RegimeStateName = Literal["CALM", "NORMAL", "STRESSED", "CRISIS"]

# OPS kill-switch state, ordered RUN < DEGRADE < HALT for max-severity precedence.
KillState = Literal["run", "degrade", "halt"]


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
    """REGIME's only output (REGIME §8). ``g_t`` in {1.00,0.75,0.50,0.25}, floored at 0.25 (Inv FLOOR).

    Enters LEVERAGE through the SINGLE H_book channel (``g_t * H_book``) — never a second min() term
    or post-min scalar (LEVERAGE Inv R / REGIME Inv NODOUBLE2). ``binding_sub`` names which of the
    four measures bound the min(); a regime cut is legible via ``binding_sub`` + reason_codes.
    """

    g_t: float
    state: RegimeStateName
    binding_sub: Literal["vol", "corr", "turb", "liq"]
    dwell_bars: int
    transition: Literal["escalate", "deescalate", "hold", "cost_gated"]
    cost_gate_passed: bool
    rebalance_cost_hat: float
    reason_codes: tuple[str, ...]
    asof_bar: int
    engine_version: str


# --------------------------------------------------------------------------- OPS output

@dataclass(frozen=True)
class OpsDirective:
    """OPS's only output (OPS types.py, P1). A {run,degrade,halt} state by precedence (never a
    blended health score, Inv KILL) + the binary flatten/cancel flags + the full trigger-reason
    chain. Holds NO size/edge/exit parameter (Inv PROV)."""

    state: KillState
    flatten: bool
    cancel_entries: bool
    reason_codes: tuple[str, ...]
    engine_version: str
