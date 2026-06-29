"""CalibrationReport — the fill-prob reliability curve + the calibration-error -> floor_eff loop (§12/§6.3).

VALIDATION-ONLY (firewalled, Invariant V). The maker branch is the leg most likely to be wrong
(§5.5), so it is the leg most heavily validated AND the only leg whose gate self-tightens on its own
miscalibration. Two products:

  * the RELIABILITY CURVE: bucket passive orders by predicted ``p_fill``, plot the realized fill rate;
    it must lie on the diagonal within CI. A curve below the diagonal means posting is over-optimistic
    and the maker-floor gate is mis-set (§12).
  * the CALIBRATION ERROR (Brier / off-diagonal): this is read as an ``F_t`` quantity by the LIVE
    maker-floor gate to widen ``floor_eff = floor_RED + κ·max(0, calib_error − calib_RED)`` (§6.3),
    closing the §12 -> live-gate loop. This module COMPUTES the error; the live gate (in
    ``microstate/fill_prob.floor_eff``) consumes it — validation never imports the live layer, the
    live layer reads the error off MicroState (Invariant V).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReliabilityBin", "CalibrationReport", "CalibrationAnalyzer"]


@dataclass(frozen=True)
class ReliabilityBin:
    """One predicted-p_fill bucket and its realized fill rate (§12)."""

    predicted_mid: float
    realized_rate: float
    n: int


@dataclass(frozen=True)
class CalibrationReport:
    """The reliability curve + Brier / calibration error for the maker-floor loop (§12 / §6.3)."""

    bins: tuple[ReliabilityBin, ...]
    brier: float
    calibration_error: float  # mean |predicted − realized| across populated bins (off-diagonal)


class CalibrationAnalyzer:
    """Builds the fill-prob reliability curve + calibration error (§12). Validation-only (Invariant V)."""

    def report(
        self, predicted: np.ndarray, realized_filled: np.ndarray, n_bins: int = 10
    ) -> CalibrationReport:
        """Bucket by predicted ``p_fill``; compute realized rate per bin + Brier + calibration error.

        ``predicted`` in [0,1]; ``realized_filled`` 0/1 outcomes. The calibration error is the mean
        absolute off-diagonal gap across populated bins — the ``F_t`` quantity the §6.3 gate widens
        ``floor_eff`` with.
        """
        predicted = np.asarray(predicted, dtype=float)
        realized = np.asarray(realized_filled, dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins: list[ReliabilityBin] = []
        gaps: list[float] = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (predicted >= lo) & (predicted < hi if i < n_bins - 1 else predicted <= hi)
            if not mask.any():
                continue
            pmid = float(predicted[mask].mean())
            rate = float(realized[mask].mean())
            bins.append(ReliabilityBin(pmid, rate, int(mask.sum())))
            gaps.append(abs(pmid - rate))
        brier = float(np.mean((predicted - realized) ** 2)) if predicted.size else 0.0
        calib_error = float(np.mean(gaps)) if gaps else 0.0
        return CalibrationReport(tuple(bins), brier, calib_error)
