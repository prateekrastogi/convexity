"""AssetClassPreset — validated per-class fill / cost params + the conservative class-prior curve (§13).

Asset-class differences flow through AMBER metadata (tick, lot, fees, stop semantics, session
calendar, 24/7 flag) and TEAL fits (each instrument's own spread, fill-prob curve, adverse-selection
magnitude) — NOT through forked tactic logic (§13). The same :class:`ExecutionEngine` runs on every
class; this preset only supplies CLASS-LEVEL caps / starting points (never per instrument, Invariant
T) and the conservative class-prior fill curve used at cold-start (§5.5 #2).

These are class-level, validated values (BLUE/preset), distinct from RED globals: e.g. the typical
half-spread proxy FLOOR the spread ladder falls through to (§11 rung 4), and the conservative
class-prior fill rate the cold-start uses when the symbol has < ``N_red`` own passive fills.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import AssetClass


@dataclass(frozen=True)
class AssetClassPreset:
    """Class-level fill / cost caps + the cold-start class-prior curve (§13). Never per instrument."""

    asset_class: AssetClass
    # §11 rung 4 — the conservative half-spread proxy FLOOR (bps) the spread ladder falls through
    # to; a finite positive constant, NEVER NaN/inf/zero.
    class_proxy_half_spread_bps: float
    # §5.4 / §13 — True only for perps: the funding-over-horizon term is priced (Invariant W); 0
    # for every non-perp class (carry there belongs to LEVERAGE).
    funding_aware: bool
    # §5.5 #2 — the conservative class-prior passive fill rate used at cold-start (AMBER reliability),
    # priced LOW so the maker branch never looks cheap before the symbol earns its own curve.
    class_prior_fill_rate: float = 0.35
    # §5.5 #2 — the class-prior calibration-CI half-width (wide, so the LCB sits below floor_eff at
    # cold-start by construction — maker GATED OFF until N_red own fills accrue).
    class_prior_ci_halfwidth: float = 0.20
    # §9 — whether the venue's STOP_MARKET is safe (default) or a STOP_LIMIT must be used (§9).
    stop_market_safe: bool = True
    # §5.1 — class-typical fill horizon scale (seconds): thicker crypto fills faster than thin
    # session-bound proxies (a starting point; the live horizon is RED ``fill_horizon_s``).
    typical_fill_horizon_s: float = 30.0


# §13 — the four documented asset-class presets. crypto_perp is the only live path on this book
# (funding-aware); the rest are present so the SAME engine runs cross-asset faithfully.
DEFAULT_PRESETS: dict[AssetClass, AssetClassPreset] = {
    "crypto_perp": AssetClassPreset(
        asset_class="crypto_perp",
        class_proxy_half_spread_bps=2.0,  # tight tick/step, small min-notional (§13)
        funding_aware=True,  # 8h funding stamps priced as an execution-timing cost (§5.4/W)
        class_prior_fill_rate=0.35,
        class_prior_ci_halfwidth=0.20,
        stop_market_safe=True,  # native STOP_MARKET reduce-only, MARK trigger (§13)
        typical_fill_horizon_s=20.0,
    ),
    "crypto_spot": AssetClassPreset(
        asset_class="crypto_spot",
        class_proxy_half_spread_bps=3.0,
        funding_aware=False,  # spot has no perpetual funding
        typical_fill_horizon_s=20.0,
    ),
    "commodity_future": AssetClassPreset(
        asset_class="commodity_future",
        class_proxy_half_spread_bps=6.0,  # wider, more variable spreads (§13)
        funding_aware=False,  # carry is a HOLD cost owned by LEVERAGE (§5.4)
        typical_fill_horizon_s=60.0,  # longer fill horizons (§13)
    ),
    "precious_metal": AssetClassPreset(
        asset_class="precious_metal",
        class_proxy_half_spread_bps=5.0,
        funding_aware=False,
        typical_fill_horizon_s=60.0,
    ),
    "industrial_metal": AssetClassPreset(
        asset_class="industrial_metal",
        class_proxy_half_spread_bps=6.0,
        funding_aware=False,
        typical_fill_horizon_s=60.0,
    ),
    "us_equity": AssetClassPreset(
        asset_class="us_equity",
        class_proxy_half_spread_bps=4.0,  # wider off-RTH raises the maker-floor gate (§13)
        funding_aware=False,  # overnight/MTF financing is a HOLD cost owned by LEVERAGE (§5.4)
        typical_fill_horizon_s=45.0,
    ),
}
