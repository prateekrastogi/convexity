"""Walk-forward Amihud/half-spread calibration stability check (COST.md Section 5.3, PARTICIPATION).

Firewalled validation companion (nothing here feeds a live rebalance). The derived model's two RED
anchors (``c`` for the half-spread, ``k_amihud`` for impact — COST.md Sections 5.2-5.3) are each
calibrated ONCE to "the median liquid name in the ~50-name universe." This module checks the
stability-of-the-liquidity-distribution assumption that calibration implicitly makes (COST.md
Section 5.3): if the universe's cross-sectional liquidity profile shifted materially release over
release, a single anchor calibrated at one point in time would under- or over-price the median name
until recalibrated.

The check re-anchors the model on each trailing walk-forward window (median ``ADV_i`` / ``Amihud_i``
as-of that window's right edge, causal) and reports how much the two anchor constants drift across
windows. A large drift is the signal to recalibrate the RED constants (a governance action, COST.md
Section 4) — NOT an automatic per-rebalance re-fit (P7's no-per-name-fit discipline; this module
diagnoses, it does not silently mutate the deployed constants).

PARTICIPATION (Invariant PARTICIPATION, COST.md Section 5.3). ``participation_rate`` reports
``Notional_leg / ADV_i`` per name so a promotion review that scales the book an order of magnitude
sees whether the sqrt-law is still being applied inside its calibrated small-single-digit-% regime.
Everything here is closed-form (median, ratio) — no fit, no ML (P7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.cost.friction.spread_impact import SpreadImpactModel, _median_positive, amihud_ratio


@dataclass(frozen=True)
class WindowCalibration:
    """The anchor constants a single trailing window would produce (COST.md Section 5.3)."""

    right_edge: int  # panel day index at the window's (inclusive) right edge — F_t-measurable
    adv_median: float
    amihud_median: float
    c: float  # half-spread anchor for this window
    k_amihud: float  # impact anchor for this window


def calibrate_window(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    right_edge: int,
    *,
    window: int = 30,
) -> WindowCalibration:
    """Anchor constants from the trailing ``window`` days ending at ``right_edge`` inclusive (causal).

    Computes the window's median ``ADV_i`` and median ``Amihud_i`` (over strictly-positive names) and
    the ``SpreadImpactModel.anchor`` constants they imply — the calibration that window's liquidity
    distribution would produce. All data is at or before ``right_edge`` (COST.md Section 0: the
    right edge is ``t``, never ``t+1``), so the result is ``F_t``-measurable.
    """
    start = max(0, right_edge - window + 1)
    r_win = np.asarray(returns, dtype=float)[start : right_edge + 1]
    dv_win = np.asarray(dollar_volume, dtype=float)[start : right_edge + 1]
    adv_i = np.full(dv_win.shape[1], np.nan)
    finite = np.isfinite(dv_win) & (dv_win > 0.0)
    has_any = finite.any(axis=0)
    if has_any.any():
        with np.errstate(invalid="ignore"):
            adv_i[has_any] = np.nanmean(np.where(finite[:, has_any], dv_win[:, has_any], np.nan), axis=0)
    amihud_i = amihud_ratio(r_win, dv_win)
    adv_med = _median_positive(adv_i)
    amihud_med = _median_positive(amihud_i)
    model = SpreadImpactModel.anchor(adv_med, amihud_med)
    return WindowCalibration(
        right_edge=right_edge,
        adv_median=adv_med,
        amihud_median=amihud_med,
        c=model.c,
        k_amihud=model.k_amihud,
    )


@dataclass(frozen=True)
class CalibrationDrift:
    """Cross-window drift of the two anchor constants (COST.md Section 5.3 stability check)."""

    windows: tuple[WindowCalibration, ...]
    c_rel_range: float  # (max c - min c) / median c across windows
    k_amihud_rel_range: float  # (max k - min k) / median k across windows


def walk_forward_drift(
    returns: np.ndarray,
    dollar_volume: np.ndarray,
    right_edges: np.ndarray,
    *,
    window: int = 30,
) -> CalibrationDrift:
    """Re-anchor on each trailing window at ``right_edges`` and report anchor-constant drift.

    A large ``c_rel_range`` / ``k_amihud_rel_range`` means the universe's liquidity distribution
    shifted enough that the single deployed RED anchor is stale for the median name — the signal to
    recalibrate (a governance action, COST.md Section 4), never an automatic re-fit (P7). This
    diagnoses stability; it does not mutate the deployed constants.

    REAL-LAKE RESULT (measured, no longer synthetic-only). Run causally against
    ``panel.build_dollar_volume`` on the true 100-crypto point-in-time lake over 2020->2024
    (``tests/cost/test_cost_validate.py::
    test_walk_forward_drift_on_real_lake_reports_large_governance_drift``), the observed drift is
    LARGE — ``c_rel_range ~= 1.4``, ``k_amihud_rel_range ~= 1.1`` — because crypto liquidity expanded
    by more than an order of magnitude over the sample. That large drift is precisely the intended
    governance recalibration signal, not a bug: it says the single point-in-time ``c``/``k_amihud``
    anchor calibrated on today's median name would have mispriced the 2020 median name and vice versa,
    so the RED anchor must be recalibrated per deployment epoch (Section 4) rather than pinned once for
    all history. The drift is a DIAGNOSTIC threshold-free number here (the caller/promotion review sets
    its own recalibration trigger against the deployment epoch it cares about); it is deliberately not
    a hard-coded pass/fail bound, since the tolerable drift depends on the epoch being priced.
    """
    cals = tuple(
        calibrate_window(returns, dollar_volume, int(e), window=window)
        for e in np.asarray(right_edges)
    )
    cs = np.array([c.c for c in cals if np.isfinite(c.c)])
    ks = np.array([c.k_amihud for c in cals if np.isfinite(c.k_amihud)])
    c_range = _rel_range(cs)
    k_range = _rel_range(ks)
    return CalibrationDrift(windows=cals, c_rel_range=c_range, k_amihud_rel_range=k_range)


def _rel_range(x: np.ndarray) -> float:
    """(max - min) / median, 0 for a degenerate (<2-point or non-positive-median) sample."""
    if x.size < 2:
        return 0.0
    med = float(np.median(x))
    if not (med > 0.0):
        return 0.0
    return float((x.max() - x.min()) / med)


def participation_rate(adv: np.ndarray, notional_leg: float) -> np.ndarray:
    """Per-name ``Notional_leg / ADV_i`` (COST.md Section 5.3, Invariant PARTICIPATION).

    The regularity-condition diagnostic: the sqrt-Amihud law is calibrated/validated only up to the
    book's historical small-single-digit-% participation range. A promotion review scaling the book
    checks this stays inside that regime before reusing ``k_amihud`` / ``c`` unchanged (COST.md
    Section 5.3 / Invariant PARTICIPATION). Non-positive / non-finite ``ADV_i`` -> ``inf`` (an
    untradeable name at any size).
    """
    a = np.asarray(adv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(a) & (a > 0.0), notional_leg / a, np.inf)
    return out
