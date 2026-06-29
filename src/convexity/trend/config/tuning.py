"""TuningGrids — finite BLUE candidate grids + the trail-multiplier presets (TREND §4/§15).

BLUE values are finite candidate presets searched by ``tune/autotune.py`` at the CLASS level only,
never freely optimized and never per-symbol (§4 BLUE). The trail-preset map is the §10 multiplier
schedule's three validated presets; ``resolve_config`` selects one and the resolved multipliers ride
on ``ResolvedConfig``.
"""

from __future__ import annotations

# §15 finite grids.
TRAIL_PRESETS = ("tight", "balanced", "tail_preserve")
DETECTOR_KINDS = ("cusum", "sprt")
ARL_TARGETS = (250, 500, 1000, 2000)
VOL_METHODS = ("ewma", "mad", "hybrid")
KALMAN_CLASS_PRESETS = (
    "equity_liquid",
    "equity_illiquid",
    "futures_macro",
    "futures_commodity",
    "precious_metal",
    "industrial_metal",
    "crypto_major",
    "crypto_alt",
)
HAZARD_MODES = ("disabled", "constant", "single", "state")


# §10 — the three trail-multiplier presets, keyed (tight, normal, loose, tail). All strictly
# increasing so INV-CVX (non-increasing marginal giveback) holds for every preset.
TRAIL_MULTIPLIER_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "tight": (1.0, 2.0, 3.0, 4.0),
    "balanced": (1.5, 2.5, 4.0, 6.0),
    "tail_preserve": (2.0, 3.5, 6.0, 9.0),
}
