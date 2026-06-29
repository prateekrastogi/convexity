"""VolTargetCeiling — L_vol = σ_target / forecast_vol (LEVERAGE §8.2).

The NORMAL-layer volatility-regime ceiling. Rises in calm regimes (low forecast vol →
more leverage allowed) and falls in volatile ones (high forecast vol → less). The
spec signature takes NO ``cfg`` argument — only the RED ``σ_target`` and the
horizon-scaled ``forecast_vol`` (Inv S; already σ·√T from ``vol/forecast.py``). This
is the dominant risk-control step in vol-managed sizing (Moreira-Muir, §17).
"""

from __future__ import annotations

from typing import Mapping

_EPS: float = 1e-12


class VolTargetCeiling:
    """L_vol = σ_target / forecast_vol (§8.2)."""

    def ceiling(self, forecast_vol: float, sigma_target: float) -> tuple[float, Mapping[str, float]]:
        """Return (L_vol, diagnostics).

        ``sigma_target`` is the RED per-position target (e.g. 15-20% annualized, passed
        on the SAME horizon basis as ``forecast_vol``). ``forecast_vol`` is the
        horizon-scaled σ·√T. No cfg argument (per the §8.2 contract).
        """
        fv = max(float(forecast_vol), _EPS)
        l_vol = float(sigma_target) / fv
        return l_vol, {"sigma_target": float(sigma_target), "forecast_vol": float(forecast_vol)}
