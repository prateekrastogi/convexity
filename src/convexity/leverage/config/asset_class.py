"""AssetClassPreset — validated per-class caps & params (LEVERAGE §13, §4).

Class-level only, never per instrument on P&L (§4, §13). These set the per-class
SOFT-ceiling starting points and the absolute hard guardrail ``L_class_max`` (§8.6.6).
Leverage ceilings fall as volatility and gap/tail risk rise; these are CEILINGS, not
targets — the min()/governor/tail ceiling routinely produce lower live leverage.
Per-class overrides enter ``resolve_config`` through this object — the SEPARATE
channel from the RED ``PolicyDefaults`` (Invariant C): a class may set a tighter
``L_class_max`` or vol target, but a RED constant is never reachable from a
per-instrument spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import AssetClass


@dataclass(frozen=True)
class AssetClassPreset:
    """One asset class's caps + starting points (§13). All overrides validated, class-level."""

    asset_class: AssetClass
    # §8.6.6 absolute HARD guardrail (clamps pre-governor leverage; binds only on a model fault).
    l_class_max: float
    # §8.2 class vol target override (None -> RED default).
    sigma_target: float | None = None
    # §8.3 class stress gap (fraction) used as the conservative tail/gap fallback.
    class_stress_gap: float = 0.10
    # §8.4 class margin stress multiplier override (None -> RED default).
    margin_stress_multiplier: float | None = None


# §13 — the cross-asset presets. Crypto-perp is the convexity live path; the others
# make the engine exercisable across the full universe the spec requires.
CRYPTO_PERP_PRESET = AssetClassPreset(
    asset_class="crypto_perp",
    l_class_max=3.0,  # §13 crypto 1-2x typical; cap conservatively above that
    sigma_target=0.20,
    class_stress_gap=0.20,  # violent gaps even without a close
)

CRYPTO_SPOT_PRESET = AssetClassPreset(
    asset_class="crypto_spot",
    l_class_max=3.0,
    sigma_target=0.20,
    class_stress_gap=0.20,
)

US_EQUITY_PRESET = AssetClassPreset(
    asset_class="us_equity",
    l_class_max=4.0,  # idiosyncratic earnings gaps -> lower caps (§13)
    sigma_target=0.15,
    class_stress_gap=0.12,
)

PRECIOUS_METAL_PRESET = AssetClassPreset(
    asset_class="precious_metal",
    l_class_max=6.0,
    sigma_target=0.15,
    class_stress_gap=0.08,
)

INDUSTRIAL_METAL_PRESET = AssetClassPreset(
    asset_class="industrial_metal",
    l_class_max=6.0,
    sigma_target=0.16,
    class_stress_gap=0.09,
)

COMMODITY_FUTURE_PRESET = AssetClassPreset(
    asset_class="commodity_future",
    l_class_max=6.0,  # session/geopolitical gaps; moderate caps (§13)
    sigma_target=0.18,
    class_stress_gap=0.12,
)

DEFAULT_PRESETS: dict[AssetClass, AssetClassPreset] = {
    "crypto_perp": CRYPTO_PERP_PRESET,
    "crypto_spot": CRYPTO_SPOT_PRESET,
    "us_equity": US_EQUITY_PRESET,
    "precious_metal": PRECIOUS_METAL_PRESET,
    "industrial_metal": INDUSTRIAL_METAL_PRESET,
    "commodity_future": COMMODITY_FUTURE_PRESET,
}
