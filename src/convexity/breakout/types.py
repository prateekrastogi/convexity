"""BREAKOUT pillar internal boundary objects (BREAKOUT §4).

Only frozen dataclasses cross module boundaries. The cross-PILLAR handoffs (``EntrySignal``,
``Position``, ``Bars``, ``InstrumentSpec``, ``CostModel``) live in :mod:`convexity.core.types` and
are NOT redefined here — this module holds the BREAKOUT-internal rich objects that flow between
BREAKOUT's own modules: ``StdSeries`` (standardize -> features/detect), ``BreakoutGeometry``
(features -> detect), ``EvidenceFrame`` (detect -> combine), plus the evaluation-only ``EntryLabel``
/ ``Report`` (validate-only).

The downstream-facing ``EntrySignal`` is constructed ONLY in ``detect/combine.py`` (Invariant A);
nothing here builds it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- standardize output


@dataclass(frozen=True)
class StdSeries:
    """Standardized-return series + past-only regime buckets (BREAKOUT §4 / §12.2).

    ONE-STEP-AHEAD CONVENTION (pins Invariant J's measurability): ``sigma[t]`` is the PREDICTIVE
    scale formed from returns ``[..t-1]`` only (F_{t-1}-measurable; it does NOT include the
    contemporaneous ``r[t]``). The standardized return is ``x[t] = r[t] / sigma[t]`` (numerator
    F_t, denominator F_{t-1}). ``vol_bucket`` / ``liq_bucket`` are integer regime buckets computed
    without future quantiles (Invariant M: a within-instrument null-calibration device, NOT
    REGIME's global state).

    NOTE (§12.2 sealed surface): there is intentionally NO ``nu`` field. A Student-t tail dof must be
    fold-local / online (``standardize/tails.py``), never a full-sample fit carried on the live
    series. No live consumer needs it, so it is dropped from the live path entirely rather than
    carried as an unused full-sample value that could later be mistaken for a fold-local one.
    """

    ts: np.ndarray
    x: np.ndarray  # standardized close-to-close log returns r[t]/sigma[t]
    r: np.ndarray  # raw log returns
    sigma: np.ndarray  # F_{t-1}-measurable predictive scale used to standardize
    vol_bucket: np.ndarray  # integer regime bucket, past-only
    liq_bucket: np.ndarray  # integer liquidity bucket, past-only

    def __len__(self) -> int:
        return int(self.ts.shape[0])


# --------------------------------------------------------------------------- range geometry output


@dataclass(frozen=True)
class BreakoutGeometry:
    """Barriers, range geometry, invalidation, stop distance (BREAKOUT §4, features/range_geometry).

    Invariant I (strictly pre-bar): ``upper_barrier[t]`` / ``lower_barrier[t]`` are functions of
    bars ``[.. t-1]`` ONLY — the inclusive rolling extremum shifted forward one bar. They never
    contain bar t's own high/low/close. ``entry_ref_price[t]`` is the price available at signal time
    (the bar-t close under the bar-close convention).
    """

    ts: np.ndarray
    direction: np.ndarray  # -1, 0, +1 candidate direction
    upper_barrier: np.ndarray
    lower_barrier: np.ndarray
    invalidation: np.ndarray  # price level that invalidates the entry idea
    stop_distance: np.ndarray  # abs(entry_ref - invalidation)
    range_width_sigma: np.ndarray
    compression_pct: np.ndarray
    barrier_age: np.ndarray
    entry_ref_price: np.ndarray  # price available at signal time for evaluation

    def __len__(self) -> int:
        return int(self.ts.shape[0])


# --------------------------------------------------------------------------- evidence frame


@dataclass(frozen=True)
class EvidenceFrame:
    """Decomposable evidence frame (BREAKOUT §4 / §8). Every channel is an explicit array; missing
    optional channels are explicit zeros, never NaNs (detect/evidence.py docstring). ``quality_score``
    + ``threshold`` are BREAKOUT-internal only (Invariant K firewall)."""

    ts: np.ndarray
    candidate: np.ndarray
    direction: np.ndarray
    glr_e: np.ndarray
    bocpd_e: np.ndarray
    range_e: np.ndarray
    compression_e: np.ndarray
    trend_e: np.ndarray
    flow_e: np.ndarray
    jump_e: np.ndarray
    context_e: np.ndarray
    false_break_risk: np.ndarray
    quality_score: np.ndarray
    threshold: np.ndarray
    reason_codes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.ts.shape[0])


# --------------------------------------------------------------------------- evaluation-only labels


@dataclass(frozen=True)
class EntryLabel:
    """Future-looking evaluation label (BREAKOUT §4, validate/labels.py). FUTURE DATA — lives only
    under ``validate/`` (Invariant C). Never imported by detect/features/standardize/online/pipeline.
    Labels start from ``entry_idx``; the stop barrier uses ``signal.invalidation`` /
    ``signal.stop_distance`` known at entry time."""

    signal_idx: int
    entry_idx: int
    horizon: int
    mfe_r: float  # max favorable excursion in stop-distance units
    mae_r: float  # max adverse excursion in stop-distance units
    hit_plus_1r_before_stop: bool
    hit_plus_2r_before_stop: bool
    hit_stop_first: bool
    time_to_plus_1r: int | None
    false_breakout: bool
    net_return_to_horizon_bps: float


@dataclass(frozen=True)
class Report:
    """Validation report (BREAKOUT §4 / §10). Identical shape across detector + every baseline
    (Invariant F)."""

    n_signals: int
    n_folds: int
    net_return_mean_bps: float
    hit_plus_1r_rate: float
    hit_stop_first_rate: float
    median_mfe_r: float
    median_mae_r: float
    convexity_ratio: float  # median MFE_R / max(abs(median MAE_R), eps)
    false_breakout_rate: float
    score_decile_table: list[dict]
    calibration_table: list[dict]
    baseline_table: list[dict]
    ablation_table: list[dict]
    by_fold: list[dict]
    by_regime: list[dict]
