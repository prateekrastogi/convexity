"""OfflineDiagnostics — Hurst / ADF / KPSS / PELT / Bai-Perron, OFFLINE ONLY (TREND §0).

Forbidden in v1 as LIVE decision signals (§0): Hurst/ADF/KPSS/Bai-Perron may be computed OFFLINE for
research, but live estimation and policy modules MUST NOT import them. This module is the offline
firewall: a guard test asserts no ``estimate/`` or ``policy/`` module imports it. The implementations
are cheap, self-contained, and clearly research-only.
"""

from __future__ import annotations

import numpy as np


class OfflineDiagnostics:
    """Offline-only research diagnostics (§0). NEVER imported by estimate/ or policy/."""

    @staticmethod
    def hurst_rs(series: np.ndarray) -> float:
        """Rescaled-range Hurst exponent (research only). H>0.5 trending, H<0.5 mean-reverting."""
        x = np.asarray(series, dtype=float)
        n = x.size
        if n < 20:
            return 0.5
        lags = np.unique(np.logspace(1, np.log10(n // 2), 8).astype(int))
        rs = []
        for lag in lags:
            chunks = n // lag
            if chunks < 1:
                continue
            vals = []
            for c in range(chunks):
                seg = x[c * lag : (c + 1) * lag]
                if seg.size < 2:
                    continue
                dev = np.cumsum(seg - seg.mean())
                r = dev.max() - dev.min()
                s = seg.std()
                if s > 0:
                    vals.append(r / s)
            if vals:
                rs.append((lag, np.mean(vals)))
        if len(rs) < 2:
            return 0.5
        lg = np.log([r[0] for r in rs])
        lr = np.log([r[1] for r in rs])
        slope = np.polyfit(lg, lr, 1)[0]
        return float(slope)

    @staticmethod
    def adf_like_stat(series: np.ndarray) -> float:
        """A cheap ADF-style coefficient (research only): the AR(1) gap from a unit root."""
        x = np.asarray(series, dtype=float)
        if x.size < 3:
            return 0.0
        dx = np.diff(x)
        lag = x[:-1]
        beta = np.polyfit(lag, dx, 1)[0]
        return float(beta)  # more negative => stronger mean reversion
