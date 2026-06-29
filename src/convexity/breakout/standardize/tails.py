"""Online / fold-local Student-t tail fit (BREAKOUT §7 standardize/tails / §2 TEAL).

A fold-local Student-t dof estimate for the standardized returns. Used to set the heavy-tail null
for the GLR/empirical-null channels. No full-sample Student-t tail fit for a fold (BREAKOUT §12.2):
the dof is fitted from the supplied (training-window) standardized returns only.
"""

from __future__ import annotations

import numpy as np


def fit_student_t_nu(x: np.ndarray, *, lo: float = 3.0, hi: float = 50.0) -> float:
    """Method-of-moments Student-t dof from excess kurtosis: nu = 6/kurt + 4 (clipped to [lo, hi]).

    A robust, allocation-free estimator (no scipy in the live path). For a fold, pass only that
    fold's training standardized returns — never the full sample (§12.2).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return 8.0
    m = np.mean(x)
    s = np.std(x) + 1e-12
    kurt = float(np.mean(((x - m) / s) ** 4) - 3.0)
    if kurt <= 0:
        return hi
    return float(np.clip(6.0 / kurt + 4.0, lo, hi))


def winsorize_causal(x: np.ndarray, *, z: float = 6.0, warm: int = 30) -> np.ndarray:
    """Causal winsorization: clip x[t] to +-z robust-sigma using a PAST-ONLY median/MAD (no global
    winsorization thresholds estimated from all data, §12.2)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    out = x.copy()
    for t in range(warm, n):
        past = x[:t]
        med = np.median(past)
        mad = np.median(np.abs(past - med)) + 1e-12
        lim = z * 1.4826 * mad
        out[t] = float(np.clip(x[t], med - lim, med + lim))
    return out
