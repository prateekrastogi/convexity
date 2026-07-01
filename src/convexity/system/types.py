"""SYSTEM shared validation data objects (SYSTEM §11).

The shapes that cross the validation boundary and are consumed by ``system/validate/``. This module
was trimmed during the cross-sectional pivot: the directional-live composition shapes
(``ProposedPortfolio``/``OpenPosition``/``BookState``/``SystemAction``/``SystemDecision``) lived
here alongside the validation records but were removed with the dead directional pillars. Only the
three validation records the surviving ``validate/`` subtree imports (``TrialRecord``,
``DSRResult``, ``SystemReport``) remain.

Pure-Python + numpy only; NautilusTrader-free. Frozen dataclasses, minimal fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np

# Kept for the SystemReport.pillar_contribution mapping key (historical DSR attribution axis).
PillarName = Literal["selection", "breakout", "leverage", "trend"]


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
