"""resolve_config — the single merge point where REGIME RED enters (REGIME.md §3.4/§6 Inv C-cfg).

The ONE place the RED :class:`RegimeDefaults` are merged with the AMBER
:class:`RegimeAssetClassPreset` into the frozen :class:`ResolvedRegimeConfig` the detector/state
machine read. RED is reachable ONLY through the ``defaults=`` argument; the AMBER preset rides
alongside and carries NO RED field. This mirrors ``convexity.cost.config.resolve`` and
``convexity.system.config.resolve`` exactly (the identical already-implemented pattern §6 points to)
so the firewall ``assert_no_red_in_amber`` can run over the AMBER field names against the REGIME
param registry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from convexity.core.params import assert_no_red_in_amber

from convexity.regime.config.asset_class import CRYPTO_REGIME_PRESET, RegimeAssetClassPreset
from convexity.regime.config.defaults import REGIME_PARAM_REGISTRY, RegimeDefaults


@dataclass(frozen=True)
class ResolvedRegimeConfig:
    """The frozen, fully-merged config the detector and state machine read (REGIME.md §6).

    RED constants live on ``defaults`` (untouched by the AMBER preset); the AMBER sleeve metadata
    rides on ``preset``. Helper properties surface the derived state-machine bands so callers never
    re-merge or recompute a threshold from raw fields.
    """

    defaults: RegimeDefaults
    preset: RegimeAssetClassPreset

    @property
    def g_bands(self) -> dict[str, float]:
        """State -> g_t band floor (REGIME.md §4.1). The conservative live value per state."""
        d = self.defaults
        return {"HEALTHY": d.g_healthy, "WATCH": d.g_watch, "STRESSED": d.g_stressed, "HALT": d.g_halt}


def resolve_config(
    *,
    defaults: RegimeDefaults | None = None,
    preset: RegimeAssetClassPreset | None = None,
) -> ResolvedRegimeConfig:
    """Merge RED defaults with the AMBER sleeve preset (REGIME.md §3.4/§6 Invariant C-cfg).

    RED enters ONLY here, through ``defaults``. ``preset`` carries AMBER metadata only; a RED name
    leaking onto the AMBER preset is the bug ``assert_no_red_in_amber`` catches.
    """
    defaults = defaults or RegimeDefaults()
    preset = preset or CRYPTO_REGIME_PRESET
    amber_field_names = frozenset(f.name for f in fields(preset))
    assert_no_red_in_amber(amber_field_names, REGIME_PARAM_REGISTRY)
    return ResolvedRegimeConfig(defaults=defaults, preset=preset)
