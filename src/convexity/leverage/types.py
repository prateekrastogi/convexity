"""LEVERAGE-pillar frozen data objects (LEVERAGE §9).

Pure-Python, NautilusTrader-free. The cross-pillar boundary types
(``Side``, ``SignalInput``, ``LeverageDecision``) already live in
``convexity.core.types`` and are RE-EXPORTED here, never redefined (ARCHITECTURE §3):
``SignalInput`` carries NO edge field (Invariant A by absence) and a regime cut
surfaces on ``LeverageDecision`` as ``binding_ceiling='portfolio'`` (P7/Inv R).

The objects defined *here* are the LEVERAGE-internal market-state view
(``RiskInputs``, ``LiquidityState``, ``GapModel``, ``GPDFit``, ``ConditionalMAE``),
the seven-ceiling bundle (``Ceilings``), and the quantizer output (``QuantizedSize``).
``BookState`` is SYSTEM-owned (ARCHITECTURE §5.9); LEVERAGE consumes it via the
``BookState`` Protocol declared here and drives its unit tests off the minimal
``BookStateFixture`` below — documented SYSTEM-owned / to-be-unified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import numpy as np

# Re-export the shared boundary types so LEVERAGE code imports them from one place
# without redefining them (ARCHITECTURE §3). LEVERAGE is the producer of
# ``LeverageDecision`` and the consumer of ``SignalInput``/``RegimeState``.
from convexity.core.types import (
    LeverageDecision,
    RegimeState,
    Side,
    SignalInput,
)

__all__ = [
    "Side",
    "SignalInput",
    "LeverageDecision",
    "RegimeState",
    "RegimeBucket",
    "LiquidityState",
    "GapModel",
    "GPDFit",
    "ConditionalMAE",
    "RiskInputs",
    "Ceilings",
    "QuantizedSize",
    "BookState",
    "BookStateFixture",
]

# §8.1 the small, robust conditional-MAE regime bucket set (vol-regime, optionally
# trend/chop) — kept tiny so each bucket has enough winners to estimate its
# 80th-pct MAE stably or it collapses to global (same event-count discipline as
# TREND's hazard ladder).
RegimeBucket = Literal["low_vol", "high_vol", "global"]


# --------------------------------------------------------------------------- measurable inputs

@dataclass(frozen=True)
class LiquidityState:
    """Measurable-only liquidity snapshot for ``LiquidityCeiling`` (LEVERAGE §9, §8.5).

    No edge quantity. ``stale_or_halted`` forces ``L_liquidity`` to 0 (§8.5).
    """

    spread_bps: float
    expected_volume_exit_window: float
    adv_notional: float
    top_of_book_notional: float | None
    impact_bps: float
    stale_or_halted: bool


@dataclass(frozen=True)
class GapModel:
    """Trailing-only historical gap distribution by session boundary (TEAL, Inv CAUSAL-L).

    ``worst_adverse_gap`` is the worst overnight/weekend adverse gap over the
    configured lookback (a FRACTION of price). ``typical_gap`` and the multiplier let
    the tail ceiling fall back to ``mult * typical_gap`` when the worst-gap sample is
    thin (LEVERAGE §8.3). ``n_gaps`` gates EVT vs the empirical/stress fallback.
    """

    worst_adverse_gap: float
    typical_gap: float
    n_gaps: int
    class_stress_gap: float

    def plausible_adverse_gap(self, typical_multiple: float) -> float:
        """The modeled worst adverse gap (§8.3): the worst observed, floored by a
        multiple of the typical gap and the class stress gap, all FRACTIONS of price."""
        return max(
            self.worst_adverse_gap, typical_multiple * self.typical_gap, self.class_stress_gap
        )


@dataclass(frozen=True)
class GPDFit:
    """Peaks-over-threshold Generalized Pareto fit of the loss tail (TEAL, EVT §8.3).

    Records the exceedance count, threshold, shape ξ / scale β, and a diagnostic
    ``ok`` flag. A precise EVT ceiling may NEVER be emitted without these recorded
    (LEVERAGE §8.3): when ``ok`` is False the tail ceiling falls back to the
    conservative empirical/stress estimate. ``emp_es`` is the rolling empirical
    Expected Shortfall at the configured level, used by the fallback and as a floor.
    ``tail_prob`` is the empirical exceedance probability of ``threshold`` (P(X>u)),
    needed to map the conditional GPD tail to an unconditional quantile.
    """

    threshold: float
    shape: float  # ξ (xi)
    scale: float  # β (beta)
    n_exceed: int
    tail_prob: float
    ok: bool
    emp_es: float

    def expected_shortfall(self, level: float) -> float:
        """ES at confidence ``level`` (e.g. 0.975) via the GPD POT closed form when the
        fit is usable, else the rolling empirical ES. Returns a positive LOSS fraction.

        POT-ES (Embrechts/McNeil/Frey): with VaR_p the p-quantile from the GPD tail,
        ES_p = VaR_p/(1-ξ) + (β - ξ·threshold)/(1-ξ), valid for ξ < 1.
        """
        if not self.ok or self.shape >= 1.0 or self.n_exceed <= 0:
            return self.emp_es
        var_p = self.var(level)
        es = var_p / (1.0 - self.shape) + (self.scale - self.shape * self.threshold) / (
            1.0 - self.shape
        )
        return max(es, self.emp_es)

    def var(self, level: float) -> float:
        """POT VaR at confidence ``level`` from the GPD tail above ``threshold``.

        Unconditional inverse-CDF (McNeil): for the upper tail with exceedance
        probability ``tail_prob`` = P(X>u),
            VaR_p = u + (β/ξ)[((1-p)/tail_prob)^(-ξ) - 1]   (ξ != 0).
        """
        p_exceed = max(self.tail_prob, 1e-12)
        ratio = max(1.0 - level, 1e-12) / p_exceed
        if abs(self.shape) < 1e-8:
            return self.threshold + self.scale * (-np.log(max(ratio, 1e-12)))
        return self.threshold + (self.scale / self.shape) * (ratio ** (-self.shape) - 1.0)


@dataclass(frozen=True)
class ConditionalMAE:
    """Per-regime-bucket winner-MAE quantile, trailing-only (TEAL, §8.1, Inv CAUSAL-L).

    ``quantile_by_bucket`` maps each :data:`RegimeBucket` to its (e.g. 80th-pct) winner
    MAE as a FRACTION of price; ``n_by_bucket`` is the event count. A bucket below
    ``min_events`` collapses to ``global`` (same event-count floor as TREND's hazard
    ladder, §8.1).
    """

    quantile_by_bucket: Mapping[str, float]
    n_by_bucket: Mapping[str, int]

    def quantile(self, regime_bucket: str, min_events: int) -> tuple[float, str]:
        """Return (mae_quantile, bucket_used). Falls back to ``global`` when the
        requested bucket has fewer than ``min_events`` winners (§8.1 event-count floor)."""
        n = int(self.n_by_bucket.get(regime_bucket, 0))
        if (
            regime_bucket != "global"
            and n >= min_events
            and regime_bucket in self.quantile_by_bucket
        ):
            return float(self.quantile_by_bucket[regime_bucket]), regime_bucket
        return float(self.quantile_by_bucket.get("global", 0.0)), "global"


@dataclass(frozen=True)
class RiskInputs:
    """The MEASURABLE market-state risk view the engine consumes (LEVERAGE §9). NO edge quantity.

    ONE risk view, no divergent mirrors: the equity / peak / covariance / candidate-weights
    / free-margin that are CANONICALLY owned by ``BookState`` (ARCHITECTURE §5.9) are read
    from ``book`` by the engine and are deliberately ABSENT here, so there can never be two
    divergent pictures of the book. ``RiskInputs`` carries only the per-instrument /
    market-state quantities that are NOT book-owned.

    ``forecast_vol`` is HORIZON-SCALED once upstream by ``vol/forecast.py`` (σ·√T,
    Inv S) — the ceilings never re-scale it. ``g_t`` is REGIME's macro multiplier in
    [0.25,1.0]; it is market-state-derived and NEVER equity drawdown (Inv R) and
    scales ``H_book`` ONLY (P7). The drawdown governor's equity axis is read from
    ``book.peak_equity``/``book.equity`` (disjoint axis, P8). ``mu_hat`` is the
    Kelly-ONLY forecast excess return, isolated to ``L_kelly`` (Inv A).
    """

    forecast_vol: float
    vol_percentile: float
    corr_instability: float
    jump_intensity: float
    gap_hist: GapModel
    loss_tail: GPDFit
    cond_mae: ConditionalMAE
    regime_bucket: str
    financing_rate: float
    liquidity: LiquidityState
    g_t: float
    mu_hat: float
    holding_horizon_periods: float  # T for σ·√T scaling consistency with the stop (Inv S)


# --------------------------------------------------------------------------- ceiling bundle

@dataclass(frozen=True)
class Ceilings:
    """The seven failure-mode ceilings, each with diagnostics (LEVERAGE §9).

    ``L_portfolio`` is computed against ``g_t·H_book`` and carries the regime cut
    (P7). ``g_t`` is echoed for the NODOUBLE audit (Inv R). ``diag`` is LOG-ONLY
    (Inv H): it may NEVER alter the leverage except through a ceiling value. The
    portfolio ``reason_code`` ('regime_heat_cut' vs 'mctr_concentration_cut') lives
    in ``diag`` so telemetry can attribute a gross cut to REGIME vs MCTR.
    """

    L_stop: float
    L_vol: float
    L_kelly: float
    L_tail: float
    L_portfolio: float
    L_margin: float
    L_liquidity: float
    g_t: float
    diag: Mapping[str, object]

    # The canonical min() order (§1). ``CEILING_NAMES`` maps each slot to its
    # LeverageDecision.binding_ceiling literal.
    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (
            self.L_stop,
            self.L_vol,
            self.L_kelly,
            self.L_tail,
            self.L_portfolio,
            self.L_margin,
            self.L_liquidity,
        )


# The binding-ceiling literal for each slot of ``Ceilings.as_tuple`` (§1, §9).
CEILING_NAMES: tuple[str, ...] = (
    "stop",
    "vol",
    "kelly",
    "tail",
    "portfolio",
    "margin",
    "liquidity",
)


# --------------------------------------------------------------------------- quantizer output

@dataclass(frozen=True)
class QuantizedSize:
    """LotSizer output (LEVERAGE §8.6.5). Rounding may only REDUCE risk or reject (Inv U)."""

    units: int
    notional: float
    realized_leverage: float
    realized_loss_pct_margin: float
    realized_loss_pct_equity: float
    realized_span_margin: float
    free_margin_after: float
    divergence: float
    flags: tuple[str, ...]
    rejected: bool


# --------------------------------------------------------------------------- SYSTEM-owned BookState

@runtime_checkable
class BookState(Protocol):
    """The SYSTEM-owned book snapshot LEVERAGE consumes (ARCHITECTURE §5.9).

    Its canonical definition is ``system/types.BookState`` (not built yet); LEVERAGE
    depends only on this read-only shape and is unit-tested against
    :class:`BookStateFixture`. The drawdown governor reads ``peak_equity``/``equity``
    ONLY (the disjoint equity axis, P8); ``cov``/``positions``/``candidate_weights``
    feed the portfolio ceiling's MCTR; ``free_margin`` feeds the margin ceiling.
    ``governor_mult`` is echoed for telemetry only.
    """

    @property
    def equity(self) -> float: ...
    @property
    def peak_equity(self) -> float: ...
    @property
    def free_margin(self) -> float: ...
    @property
    def positions(self) -> Mapping[str, float]: ...
    @property
    def cov(self) -> np.ndarray: ...
    @property
    def cov_index(self) -> tuple[str, ...]: ...
    @property
    def candidate_weights(self) -> Mapping[str, float]: ...
    @property
    def governor_mult(self) -> float: ...


@dataclass(frozen=True)
class BookStateFixture:
    """Minimal concrete ``BookState`` for LEVERAGE's unit tests (SYSTEM-owned / to-be-unified).

    SYSTEM will replace this with ``system/types.BookState``; the fields mirror the
    ARCHITECTURE §5.9 contract (equity, peak_equity, free_margin, positions, cov,
    governor_mult) plus the ``cov_index`` / ``candidate_weights`` the MCTR ceiling needs.
    """

    equity: float
    peak_equity: float
    free_margin: float
    positions: Mapping[str, float]
    cov: np.ndarray
    cov_index: tuple[str, ...]
    candidate_weights: Mapping[str, float] = field(default_factory=dict)
    governor_mult: float = 1.0
