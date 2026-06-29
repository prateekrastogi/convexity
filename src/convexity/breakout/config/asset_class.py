"""AssetClassOverride — controlled per-asset-class RED overrides + the barrier-family policy.

The barrier family that defines ``upper_barrier``/``lower_barrier`` (Donchian, robust rolling
quantile, opening-range, anchored-session-range) is a RED *modelling policy chosen per asset-class*
(BREAKOUT §2 barrier-family policy), NOT a free per-instrument choice. It is governed:

  * 24/7 instruments (crypto, no session boundary): family is **rolling Donchian** by default;
    the session-anchored opening-range variant is DISABLED for them.
  * Instruments whose ``calendar_id`` declares sessions (futures/metals/equities): the
    **session-anchored opening-range** variant is permitted.

This module declares the per-class presets + the allowed barrier families; ``resolve.py`` owns the
actual disablement decision (so the disablement is in the single RED merge point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from convexity.core.types import AssetClass

BarrierFamily = Literal["donchian", "robust_quantile", "opening_range", "anchored_session"]

# Families that require a session boundary in calendar_id (BREAKOUT §2/§3). Disabled for 24/7.
SESSION_ANCHORED_FAMILIES: frozenset[str] = frozenset({"opening_range", "anchored_session"})


@dataclass(frozen=True)
class AssetClassOverride:
    """Validated class-level override (BREAKOUT §2 BLUE/§3). Carries only the small set of values a
    class may legitimately differ on; everything else inherits ``DetectorDefaults``."""

    asset_class: AssetClass
    barrier_family: BarrierFamily = "donchian"
    # optional class-level RED overrides (None => inherit DetectorDefaults)
    cross_k: float | None = None
    windows: tuple[int, ...] | None = None
    preset: str | None = None


# Default per-class presets. Crypto stays on rolling Donchian (no session boundary); the
# session-bearing classes default to opening-range where their calendar declares sessions.
DEFAULT_OVERRIDES: dict[AssetClass, AssetClassOverride] = {
    "crypto_perp": AssetClassOverride("crypto_perp", barrier_family="donchian"),
    "crypto_spot": AssetClassOverride("crypto_spot", barrier_family="donchian"),
    "commodity_future": AssetClassOverride("commodity_future", barrier_family="opening_range"),
    "precious_metal": AssetClassOverride("precious_metal", barrier_family="opening_range"),
    "industrial_metal": AssetClassOverride("industrial_metal", barrier_family="opening_range"),
    "us_equity": AssetClassOverride("us_equity", barrier_family="opening_range"),
}
