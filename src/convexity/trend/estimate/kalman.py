"""FixedPresetLocalTrend — class-preset fixed-noise local-level/local-slope filter (TREND §9, Inv E).

A causal local-level + local-slope Kalman filter on log price. The process/observation noise ratio
``q`` is selected from a validated asset-class preset (config/asset_class.KALMAN_NOISE_RATIO) and is
FIXED for the run: it is NOT estimated per symbol and NOT adapted online from recent PnL (Inv E). The
filter is FILTERING ONLY in the live path — no smoothing (§16.5). Observations are
volatility-normalized via the supplied ``sigma_log`` so the observation variance tracks the regime.

State x = [level, slope]; transition F = [[1,1],[0,1]]; observation H = [1,0]. Process covariance is
``q * sigma_obs^2`` on the slope (a local-linear-trend random walk on the slope), observation variance
``sigma_obs^2``. Returns ``(filtered_slope, slope_var)`` per bar.
"""

from __future__ import annotations

import numpy as np


class FixedPresetLocalTrend:
    """Causal local-trend filter with a FIXED class-preset noise ratio (Inv E)."""

    def __init__(self, noise_ratio: float) -> None:
        if noise_ratio <= 0:
            raise ValueError("noise_ratio must be positive (class preset, Inv E)")
        self.noise_ratio = float(noise_ratio)
        self._init = False
        self.x = np.zeros(2)  # [level, slope]
        self.P = np.eye(2)  # state covariance
        self._F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._H = np.array([[1.0, 0.0]])

    def reset(self) -> None:
        self._init = False
        self.x = np.zeros(2)
        self.P = np.eye(2)

    def update(self, price_log: float, sigma_log: float) -> tuple[float, float]:
        """One causal filter step on log price (§9). Returns ``(filtered_slope, slope_var)``."""
        r = max(sigma_log, 1e-8) ** 2  # observation variance
        if not self._init:
            self.x = np.array([price_log, 0.0])
            self.P = np.diag([r, r])
            self._init = True
            return float(self.x[1]), float(self.P[1, 1])

        # Predict.
        x_pred = self._F @ self.x
        q = self.noise_ratio * r  # process noise scales with observation noise (fixed ratio)
        Q = np.array([[0.0, 0.0], [0.0, q]])
        P_pred = self._F @ self.P @ self._F.T + Q

        # Update.
        y = price_log - float((self._H @ x_pred)[0])  # innovation
        S = float((self._H @ P_pred @ self._H.T)[0, 0]) + r
        K = (P_pred @ self._H.T).ravel() / S
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, self._H.ravel())) @ P_pred

        return float(self.x[1]), float(self.P[1, 1])
