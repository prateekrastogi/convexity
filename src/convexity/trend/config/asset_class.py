"""AssetClassPreset — validated class-level overrides (TREND §5).

The module uses ONE code path; asset-class differences enter through AMBER metadata, TEAL fitted
profiles, and these class-level presets (validated, never per-symbol — Inv E for the Kalman ratio).
A preset overrides a subset of RED defaults at the CLASS level only; a ``None`` field means "use the
RED default". The Kalman noise-ratio preset groups are the eight named in §9.

The required class names span the seven §5 classes plus the eight Kalman preset groups; the
``crypto_perp`` / ``crypto_spot`` shared-core classes from :mod:`convexity.core.types` map onto the
``crypto_major`` / ``crypto_alt`` Kalman groups.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import AssetClass

# §9 — the eight allowed Kalman noise-ratio preset groups. Higher ratio = trust the observation less
# / smoother slope (illiquid, noisy); lower = more responsive (liquid). Class-level only (Inv E).
KALMAN_NOISE_RATIO: dict[str, float] = {
    "equity_liquid": 0.02,
    "equity_illiquid": 0.005,
    "futures_macro": 0.02,
    "futures_commodity": 0.01,
    "precious_metal": 0.01,
    "industrial_metal": 0.008,
    "crypto_major": 0.03,
    "crypto_alt": 0.015,
}


@dataclass(frozen=True)
class AssetClassPreset:
    """A validated class-level override bundle (§5). ``None`` => fall through to the RED default."""

    asset_class: AssetClass
    kalman_class_preset: str  # one of KALMAN_NOISE_RATIO's keys
    active_clock: str  # primary clock for this class (§1.2)
    trail_preset: str | None = None  # tight|balanced|tail_preserve (BLUE; None => default)
    arl0: int | None = None
    gap_survival_sigmas: float | None = None
    vol_method: str | None = None


# §1.2 / §5 — the per-shared-class defaults. crypto trades 24/7 (calendar clock); equities by
# trading sessions; futures/metals by exchange sessions.
DEFAULT_PRESETS: dict[AssetClass, AssetClassPreset] = {
    "commodity_future": AssetClassPreset(
        asset_class="commodity_future",
        kalman_class_preset="futures_commodity",
        active_clock="trading_sessions",
        gap_survival_sigmas=3.5,
    ),
    "precious_metal": AssetClassPreset(
        asset_class="precious_metal",
        kalman_class_preset="precious_metal",
        active_clock="trading_sessions",
        gap_survival_sigmas=3.0,
    ),
    "industrial_metal": AssetClassPreset(
        asset_class="industrial_metal",
        kalman_class_preset="industrial_metal",
        active_clock="trading_sessions",
        gap_survival_sigmas=3.0,
    ),
    "us_equity": AssetClassPreset(
        asset_class="us_equity",
        kalman_class_preset="equity_liquid",
        active_clock="trading_sessions",
        gap_survival_sigmas=4.0,  # overnight gaps are heavier for equities
    ),
    "crypto_spot": AssetClassPreset(
        asset_class="crypto_spot",
        kalman_class_preset="crypto_alt",
        active_clock="calendar_days",
        gap_survival_sigmas=3.0,
    ),
    "crypto_perp": AssetClassPreset(
        asset_class="crypto_perp",
        kalman_class_preset="crypto_major",
        active_clock="calendar_days",
        gap_survival_sigmas=3.0,
    ),
}


def noise_ratio_for(preset_name: str, default: float) -> float:
    """Return the class Kalman noise ratio, falling back to the RED default for an unknown group."""
    return KALMAN_NOISE_RATIO.get(preset_name, default)
