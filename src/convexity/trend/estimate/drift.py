"""VarianceIntegratedDrift — the path-normalized drift z + excursions (TREND §9 drift).

The old single-current-sigma z is too fragile; a trend held through volatility regimes must be
normalized by the variance accumulated over the PATH, not the current bar's sigma (§9). So

    z_drift = side * log(price_t / entry_price) / sqrt(max(integrated_var_since_entry, eps))

The worked example pins the constant: a +10% favorable move under integrated path-vol 0.1 gives
z_drift = log(1.1)/0.1 = 0.9531017980432486 (§13 / config.defaults.Z_DRIFT_WORKED). Favorable/adverse
excursions are in initial-risk R units when ``initial_risk`` exists, else in robust-vol units (§9).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Position
from convexity.trend.types import VolState


class VarianceIntegratedDrift:
    """Causal variance-integrated drift z + excursions (§9)."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def update(
        self,
        bars: Bars,
        idx: int,
        position: Position,
        vol: VolState,
    ) -> tuple[float, float, float]:
        """Return ``(z_drift, favorable_excursion_r, adverse_excursion_r)`` at ``idx`` (§9).

        ``z_drift`` is the path-variance-normalized side-adjusted log move; the excursions are the
        running favorable/adverse moves measured from entry in R units (or robust-vol units when
        ``initial_risk`` is absent). All ``F_t``-measurable.
        """
        side = float(position.side)
        p0 = position.entry_price
        pt = bars.c[idx]
        if p0 <= 0 or pt <= 0:
            return 0.0, 0.0, 0.0

        x_t = side * float(np.log(pt / p0))  # side-adjusted log PnL (§1.3)
        denom = float(np.sqrt(max(vol.integrated_var_since_entry, self.eps)))
        z_drift = x_t / denom

        # Excursion unit: initial-risk R when supplied, else robust vol (current sigma_log).
        if position.initial_risk is not None and position.initial_risk > 0:
            # initial_risk is a PRICE distance; convert the side-adjusted price move to R units.
            move_price = side * (pt - p0)
            r_unit = position.initial_risk
            move_r = move_price / r_unit
        else:
            unit = max(vol.sigma_log, self.eps)
            move_r = x_t / unit

        favorable = max(move_r, 0.0)
        adverse = max(-move_r, 0.0)
        return float(z_drift), float(favorable), float(adverse)
