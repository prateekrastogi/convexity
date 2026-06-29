"""resolve_config — the single merge point where RED enters (LEVERAGE §4, Invariant C).

The ONE place the RED ``PolicyDefaults`` are merged with the per-class
``AssetClassPreset`` and the per-instrument AMBER ``InstrumentLeverageSpec`` into the
frozen ``ResolvedConfig`` the ceilings/combine layers read. RED is reachable ONLY
through the ``defaults=`` argument; per-class overrides come through the SEPARATE
``preset=`` argument; the per-instrument spec carries AMBER only. A RED value on the
instrument spec is the bug this firewall catches — ``assert_no_red_in_amber`` runs
over the instrument field names against :data:`LEVERAGE_PARAM_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import assert_no_red_in_amber

from convexity.leverage.config.asset_class import DEFAULT_PRESETS, AssetClassPreset
from convexity.leverage.config.defaults import LEVERAGE_PARAM_REGISTRY, PolicyDefaults
from convexity.leverage.config.instrument import (
    INSTRUMENT_LEVERAGE_SPEC_FIELDS,
    InstrumentLeverageSpec,
)


@dataclass(frozen=True)
class ResolvedConfig:
    """The frozen, fully-merged config every ceiling/combine module reads (§4).

    RED constants live on ``defaults`` (untouched by per-instrument config); the class
    preset and the AMBER instrument spec ride alongside. Helper properties surface the
    few values a class may override (σ_target, L_class_max, stress multiplier) so
    callers never re-merge.
    """

    defaults: PolicyDefaults
    preset: AssetClassPreset
    instrument: InstrumentLeverageSpec

    @property
    def sigma_target(self) -> float:
        """Class override of the RED vol target when present (§8.2)."""
        if self.preset.sigma_target is not None:
            return self.preset.sigma_target
        return self.defaults.sigma_target

    @property
    def l_class_max(self) -> float:
        """The HARD per-class guardrail (§8.6.6). Class preset wins; falls back to the
        RED ``class_cap_by_asset`` map, then the RED default cap."""
        if self.preset.l_class_max is not None:
            return self.preset.l_class_max
        return self.defaults.class_cap_by_asset.get(
            self.instrument.asset_class, self.defaults.default_class_cap
        )

    @property
    def margin_stress_multiplier(self) -> float:
        """Class override of the RED portfolio-margin stress multiplier (§8.4, P4)."""
        if self.preset.margin_stress_multiplier is not None:
            return self.preset.margin_stress_multiplier
        return self.defaults.margin_stress_multiplier

    @property
    def class_stress_gap(self) -> float:
        """Class stress-gap fallback used by the tail/gap ceiling (§8.3)."""
        return self.preset.class_stress_gap


def resolve_config(
    instrument: InstrumentLeverageSpec,
    *,
    defaults: PolicyDefaults | None = None,
    preset: AssetClassPreset | None = None,
) -> ResolvedConfig:
    """Merge RED defaults + class preset + AMBER instrument spec into one ``ResolvedConfig``.

    RED enters ONLY through ``defaults`` (Invariant C). The per-class preset is the
    SEPARATE validated channel; if omitted, the asset-class default preset is used.
    Runs ``assert_no_red_in_amber`` so a RED/BLUE name leaking onto the instrument
    spec raises immediately.
    """
    assert_no_red_in_amber(INSTRUMENT_LEVERAGE_SPEC_FIELDS, LEVERAGE_PARAM_REGISTRY)
    red = defaults if defaults is not None else PolicyDefaults()
    cls = preset if preset is not None else DEFAULT_PRESETS[instrument.asset_class]
    return ResolvedConfig(defaults=red, preset=cls, instrument=instrument)
