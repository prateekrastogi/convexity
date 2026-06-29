"""Baselines incl. the Full-Kelly CAUTIONARY baseline (LEVERAGE §12, firewalled).

VALIDATION-ONLY (Invariant V). Full Kelly is included ONLY as a CAUTIONARY baseline to
demonstrate it repeatedly trips the 43% governor (P3, §12) — Half Kelly must show
materially lower max drawdown for comparable growth. Also fixed-fraction and
single-ceiling baselines for ablation contrast.
"""

from __future__ import annotations


def full_kelly_leverage(mu_hat: float, sigma_hat: float) -> float:
    """Full-Kelly leverage f* = μ/σ² (the classic, FORBIDDEN in production — P3). Cautionary
    baseline only: its drawdown distribution makes a 43% governor a near-certain repeated
    trip, maximizing liquidation rather than growth."""
    return float(mu_hat) / max(float(sigma_hat) ** 2, 1e-12)


def fixed_fraction_leverage(fraction: float, sigma_target: float, forecast_vol: float) -> float:
    """A naive fixed-fraction-of-vol-target baseline (no ceilings) for the ablation contrast."""
    return float(fraction) * float(sigma_target) / max(float(forecast_vol), 1e-12)


def single_ceiling_leverage(ceiling_value: float) -> float:
    """A single-ceiling baseline (e.g. only L_vol, no min()): shows the value of the full
    min() of seven ceilings vs relying on any one (§12 ablation)."""
    return float(ceiling_value)
