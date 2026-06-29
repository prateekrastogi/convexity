"""AssetClassOpsPreset — per-class validated limit presets (OPS §4 BLUE, class-level only).

Spec: OPS §4 (BLUE — finite validated grids, CLASS-level only, NEVER per-symbol-on-PnL). Different
asset classes have different sample frequencies (a per-bar crypto-perp vol stream vs a per-trade
equity edge stream), so the per-source ARL0 is set per the stream's sample frequency (§6.6) at the
class level. These presets are searched OFFLINE inside SYSTEM's CPCV and promoted by the OOS gate
(§11.2) — never tuned per symbol on realized P&L (Invariant C/OOS).

A preset only *selects among* the BLUE grids in ``defaults.py``; it never introduces a new continuous
degree of freedom (NOFIT). The default returns the canonical :class:`OpsDefaults`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from convexity.core.types import AssetClass
from convexity.ops.config.defaults import DEFAULTS, OpsDefaults


@dataclass(frozen=True)
class AssetClassOpsPreset:
    """A class-level override of the per-source ARL0 target + warm-up (§6.6/§6.8), nothing per-symbol.

    ``panel_arl0_target`` is set higher for a high-frequency (per-bar) stream so the per-source false-
    page rate stays human-tolerable (§6.6: "per-source ARL0 is set per the stream's sample
    frequency"). ``ewma_warm`` reflects that an autocorrelated 24/7 crypto vol stream needs a longer
    settling window than a low-frequency equity stream.
    """

    asset_class: AssetClass
    panel_arl0_target: float
    n_warm_ewma: int


# Crypto-perp (24/7, per-minute bars) is the convexity live path: a high per-bar frequency, so a
# larger per-source ARL0 keeps the panel page rate sane. Lower-frequency classes can run a tighter
# ARL0 because each sample is rarer.
PRESETS: dict[AssetClass, AssetClassOpsPreset] = {
    "crypto_perp": AssetClassOpsPreset("crypto_perp", panel_arl0_target=2000.0, n_warm_ewma=50),
    "crypto_spot": AssetClassOpsPreset("crypto_spot", panel_arl0_target=2000.0, n_warm_ewma=50),
    "us_equity": AssetClassOpsPreset("us_equity", panel_arl0_target=500.0, n_warm_ewma=40),
    "commodity_future": AssetClassOpsPreset("commodity_future", panel_arl0_target=1000.0,
                                            n_warm_ewma=40),
    "precious_metal": AssetClassOpsPreset("precious_metal", panel_arl0_target=1000.0, n_warm_ewma=40),
    "industrial_metal": AssetClassOpsPreset("industrial_metal", panel_arl0_target=1000.0,
                                            n_warm_ewma=40),
}


def apply_preset(defaults: OpsDefaults, asset_class: AssetClass | None) -> OpsDefaults:
    """Apply a class-level preset to the RED defaults (§4 BLUE). No-op when ``asset_class`` is None.

    Only the per-source ARL0 target and EWMA warm-up are class-tunable; the SPC *limits* themselves
    are solved against the (class-chosen) ARL0 by ``derive_limits.py``, never edited per symbol.
    """
    if asset_class is None or asset_class not in PRESETS:
        return defaults
    p = PRESETS[asset_class]
    return replace(defaults, panel_arl0_target=p.panel_arl0_target, n_warm_ewma=p.n_warm_ewma)


__all__ = ["AssetClassOpsPreset", "PRESETS", "apply_preset", "DEFAULTS"]
