"""AssetClassCostPreset — validated per-class carry-model + fallback presets (COST §14).

Class-level only, never per instrument on P&L (§4, §14). These set the carry
MECHANICS and the conservative fallbacks for a ``CarryClass``; live numbers come
from TEAL fits and AMBER metadata. Per-class overrides enter ``resolve_config``
through this object — the SEPARATE channel from the RED ``CostDefaults`` (Invariant
C-cfg): a class may widen its band quantile or fallback spread, but a RED constant
is never reachable from a per-instrument spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.cost.types import CarryClass


@dataclass(frozen=True)
class AssetClassCostPreset:
    """One asset class's carry mechanics + fallbacks (§14).

    ``class_worst_funding_rate`` is the §5.5a priority-3 fallback (worst-recent
    funding) when an instrument has neither a published rate nor history.
    ``class_worst_residual_e_q`` is the §5.5c ``fallback_band_thin`` class-level
    worst residual quantile used when a residual series is below ``n_residual_min``.
    """

    carry_class: CarryClass
    # §5.3 class-level conservative half-spread (bps) — overrides the RED default per class.
    fallback_half_spread_bps: float = 5.0
    # §5.4 class-level conservative impact coefficient (fraction).
    fallback_impact_k_fraction: float = 0.50
    # §5.5a priority-3 fallback worst-recent funding rate (per 8h stamp, magnitude).
    class_worst_funding_rate: float = 0.0
    # §5.5c fallback_band_thin class-level worst one-step residual quantile (magnitude).
    class_worst_residual_e_q: float = 0.0
    # §5.5c band quantile q — may be widened for a high-funding-variance class.
    band_quantile_q: float | None = None


# §14 — the four cross-asset presets. Crypto-perp is the live path; the others make
# the ONE CarryModel interface exercisable across all four classes (Invariant X).
CRYPTO_PERP_PRESET = AssetClassCostPreset(
    carry_class="crypto_perp",
    fallback_half_spread_bps=5.0,
    fallback_impact_k_fraction=0.50,
    class_worst_funding_rate=0.0075,  # near the typical Binance USDT-M cap
    class_worst_residual_e_q=0.0020,
    band_quantile_q=0.90,
)

DATED_FUTURE_PRESET = AssetClassCostPreset(
    carry_class="dated_future",
    fallback_half_spread_bps=3.0,
    fallback_impact_k_fraction=0.40,
)

EQUITY_MTF_PRESET = AssetClassCostPreset(
    carry_class="equity_mtf",
    fallback_half_spread_bps=2.0,
    fallback_impact_k_fraction=0.30,
)

COMMODITY_PRESET = AssetClassCostPreset(
    carry_class="commodity",
    fallback_half_spread_bps=4.0,
    fallback_impact_k_fraction=0.45,
)

DEFAULT_PRESETS: dict[CarryClass, AssetClassCostPreset] = {
    "crypto_perp": CRYPTO_PERP_PRESET,
    "dated_future": DATED_FUTURE_PRESET,
    "equity_mtf": EQUITY_MTF_PRESET,
    "commodity": COMMODITY_PRESET,
}
