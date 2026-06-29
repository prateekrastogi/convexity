"""SYSTEM shared data objects (SYSTEM §11).

The shapes that cross the orchestrator/validation boundary. ``BookState`` is the *canonical*
cross-pillar reference (ARCHITECTURE §5.9): LEVERAGE/REGIME/OPS all consume it as ``object`` at
their seams and SYSTEM owns the one true definition here (equity, peak_equity, free_margin,
positions, cov, governor_multiplier — the shared equity/peak reference behind the 43% governor and
the 45% OPS DD-kill, ARCHITECTURE §5.12).

Pure-Python + numpy only; NautilusTrader-free. Frozen dataclasses, minimal fields. The boundary
``Side``/``EntrySignal``/``SignalInput``/``LeverageDecision`` live in :mod:`convexity.core.types`;
only SYSTEM-private shapes live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np

from convexity.core.types import Side

PillarName = Literal["selection", "breakout", "leverage", "trend"]
ActionKind = Literal["open", "close", "reduce", "hold", "flat_all"]


# --------------------------------------------------------------------------- composition I/O


@dataclass(frozen=True)
class ProposedPortfolio:
    """FROM the Carver Ch.25 selection module (SYSTEM §11), via ``modules/selection.py``.

    ``keep`` = instruments selection wants capital in; ``eligible`` = instruments eligible for NEW
    entries this bar; ``risk_weights`` = target RISK weights (NOT sizes, no edge); ``rank`` = a
    diversification/quality RISK measure that NEVER drives sizing or budget priority (Invariant A).
    """

    keep: tuple[str, ...]
    eligible: tuple[str, ...]
    risk_weights: Mapping[str, float]
    rank: Mapping[str, float]
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenPosition:
    """One open position read from NautilusTrader (SYSTEM §11). ``held_risk_weight`` is the reserved
    MCTR budget an incumbent holds (P4); ``trend_state`` echoes TREND's alive/reduce/death."""

    instrument: str
    side: Side
    units: int
    entry_price: float
    held_risk_weight: float
    trend_state: str  # "alive" / "reduce" / "death"


@dataclass(frozen=True)
class BookState:
    """The canonical book snapshot read from NautilusTrader at F_t (SYSTEM §11, ARCHITECTURE §5.9).

    ``cov`` is the LeverageEngine RISK covariance (no Σ⁻¹μ; Invariant P inherited). ``governor_multiplier``
    is m ∈ [0,1] from the LEVERAGE drawdown governor (0 at ≥43% drawdown ⇒ FLAT_ALL). ``equity`` and
    ``peak_equity`` are the SHARED equity/running-max reference behind both the LEVERAGE 43% governor
    flat and the OPS 45% DD-kill (ARCHITECTURE §5.12; 45% > 43% holds by construction).
    """

    equity: float
    peak_equity: float
    free_margin: float
    positions: Mapping[str, OpenPosition]
    cov: np.ndarray
    governor_multiplier: float

    @property
    def drawdown(self) -> float:
        """Current drawdown from the peak (the axis the governor and OPS DD-kill share)."""
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


@dataclass(frozen=True)
class SystemAction:
    """One line of the book delta (SYSTEM §11). ``driven_by`` names the pillar/precedence stage that
    produced it; ``binding`` names the binding ceiling/precedence rule; ``reason_chain`` is the full
    cross-module audit trail (P8). ``target_units`` is post-quantize, round-DOWN (LEVERAGE Inv U)."""

    instrument: str
    kind: ActionKind
    target_units: int
    side: Side | None
    driven_by: PillarName | Literal["governor", "iterate", "system"]
    binding: str
    reason_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemDecision:
    """OUTPUT of one :meth:`SystemOrchestrator.decide` (SYSTEM §11/§12). The target book delta plus
    the fixed-point telemetry and the deferred-candidate ledger. Deterministic: identical inputs ⇒
    identical SystemDecision."""

    bar_ts: int
    actions: tuple[SystemAction, ...]
    iterations: int
    converged: bool
    book_heat_used: float
    equity_ladder_used: Mapping[str, float]
    deferred: tuple[str, ...]
    reason_codes: tuple[str, ...]
    system_version: str


# --------------------------------------------------------------------------- validation I/O


@dataclass(frozen=True)
class TrialRecord:
    """One row of the persistent trial ledger → effective ``N_sys`` (SYSTEM §11, Invariant SG).

    ``oos_returns_ref`` points at the per-config OOS path-return series so the cross-config
    correlation matrix (Section 7.1) can be rebuilt for the effective-trials count. ``blue_point`` is
    the joint BLUE grid coordinate; ``config_hash`` is the content address used by the EvidenceCache.
    ``oos_sharpe`` is the ANNUALIZED OOS Sharpe (the reporting convention); it is de-annualized to
    per-observation units inside the gate (Section 7.2a). ``oos_returns`` carries the per-observation
    OOS series in-memory for the cross-config correlation matrix.
    """

    config_hash: str
    module_versions: Mapping[str, str]
    blue_point: Mapping[str, object]
    oos_sharpe: float
    oos_returns_ref: str
    oos_returns: np.ndarray | None = None


@dataclass(frozen=True)
class DSRResult:
    """OUTPUT of :meth:`DeflatedSharpe.deflate` (SYSTEM §7/§12). Every input recorded so the
    deflation is fully auditable. ``sr_pp`` and ``sr_star0`` are PER-OBSERVATION (Section 7.2a); the
    headline is (``dsr`` | ``n_obs`` vs ``min_trl``)."""

    psr: float
    dsr: float
    sr_star0: float
    sr_pp: float
    var_trials: float
    var_floored: bool
    n_sys: float
    n_obs: int
    ppy: float
    min_trl: float
    n_obs_sufficient: bool
    k_distinct: int
    psr_denom_floored: bool
    passed: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemReport:
    """OUTPUT of integrated validation (SYSTEM §11). The HEADLINE is always (DSR | n_obs vs MinTRL),
    never the vanilla Sharpe (Invariant DSR/DSR-MINTRL)."""

    psr: float
    dsr: float
    sr_star0: float
    n_sys: float
    n_sys_nominal: int
    k_ledger: int
    k_distinct: int
    var_trials: float
    var_floored: bool
    n_obs: int
    ppy: float
    min_trl: float
    n_obs_sufficient: bool
    rho_cl: float
    walkforward: Mapping[str, float] = field(default_factory=dict)
    cpcv_path_sharpes: tuple[float, ...] = ()
    drop_top_n: Mapping[str, float] = field(default_factory=dict)
    bootstrap_ci: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    attribution_mode: Literal["shapley", "sequential_fixed_order"] = "sequential_fixed_order"
    pillar_contribution: Mapping[PillarName, float] = field(default_factory=dict)
    baselines: Mapping[str, float] = field(default_factory=dict)
    regime_attribution: Mapping[str, float] = field(default_factory=dict)
    forward_gate: Mapping[str, float] | None = None
    overnight_wall_clock_s: float = 0.0
    passed: bool = False
    reason_codes: tuple[str, ...] = ()

    def headline(self) -> str:
        """The (DSR | n_obs vs MinTRL) headline — never the DSR alone (Invariant DSR-MINTRL)."""
        ok = "≥" if self.n_obs_sufficient else "<"
        return f"DSR={self.dsr:.4f} @ n_obs={self.n_obs} ({ok} MinTRL={self.min_trl:.0f})"
