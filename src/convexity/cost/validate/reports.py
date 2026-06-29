"""Validation reports — forecast bias/RMSE, gate calibration, TCA residual (COST §12).

Firewalled (Invariant V). Assembles the scored quantities §12 requires:
forecast funding bias + RMSE per regime; net-EV gate hit-rate calibration; hold-input
calibration; and the TCA reconciliation-residual distribution (Invariant R). All
inputs are realized labels (future-only) paired with the forecasts the gates recorded
(Invariant E).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastErrorReport:
    """Forecast funding bias + RMSE (overall or per regime, §12)."""

    bias: float
    rmse: float
    n: int


def forecast_error(forecasts: np.ndarray, realized: np.ndarray) -> ForecastErrorReport:
    """Bias (mean signed error) and RMSE of a forecast vs realized series (§12, Inv E)."""
    f = np.asarray(forecasts, dtype=float)
    r = np.asarray(realized, dtype=float)
    err = f - r
    return ForecastErrorReport(
        bias=float(np.mean(err)) if err.size else 0.0,
        rmse=float(np.sqrt(np.mean(err**2))) if err.size else 0.0,
        n=int(err.size),
    )


def gate_calibration(admitted: np.ndarray, net_positive: np.ndarray) -> dict:
    """Of trades the gate admitted, what fraction were net-cost-positive ex-post (§12)."""
    admitted = np.asarray(admitted, dtype=bool)
    net_positive = np.asarray(net_positive, dtype=bool)
    if admitted.sum() == 0:
        admitted_hit = float("nan")
    else:
        admitted_hit = float(net_positive[admitted].mean())
    rejected = ~admitted
    rejected_hit = float(net_positive[rejected].mean()) if rejected.sum() else float("nan")
    return {"admitted_net_positive_rate": admitted_hit, "rejected_net_positive_rate": rejected_hit}


def tca_residual_distribution(residuals: np.ndarray, tolerance: float) -> dict:
    """Summary of the TCA unattributed-residual magnitudes across trades (Invariant R, §12)."""
    res = np.abs(np.asarray(residuals, dtype=float))
    return {
        "max_abs_residual": float(res.max()) if res.size else 0.0,
        "mean_abs_residual": float(res.mean()) if res.size else 0.0,
        "frac_over_tolerance": float((res > tolerance).mean()) if res.size else 0.0,
    }
