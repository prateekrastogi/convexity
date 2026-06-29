"""resolve_config — the single merge point where RED enters (COST §4, Invariant C-cfg).

The ONE place RED ``CostDefaults`` are merged with the per-class
``AssetClassCostPreset`` and the per-instrument AMBER ``InstrumentCostSpec`` into the
frozen ``ResolvedCostConfig`` the components/consumers read. RED is reachable ONLY
through the ``defaults=`` argument; per-class overrides come through the SEPARATE
``preset=`` argument; the per-instrument spec carries AMBER only. A RED value on the
instrument spec is the bug this firewall catches — ``assert_no_red_in_amber`` runs
over the instrument field names against the COST param registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import assert_no_red_in_amber

from convexity.cost.config.asset_class import DEFAULT_PRESETS, AssetClassCostPreset
from convexity.cost.config.defaults import COST_PARAM_REGISTRY, CostDefaults
from convexity.cost.config.fee_schedule import (
    BINANCE_USDM_FEE_SCHEDULE,
    FeeSchedule,
)
from convexity.cost.config.instrument import (
    INSTRUMENT_COST_SPEC_FIELDS,
    InstrumentCostSpec,
)


@dataclass(frozen=True)
class ResolvedCostConfig:
    """The frozen, fully-merged config every COST component/consumer reads (§4).

    RED constants live on ``defaults`` (untouched by per-instrument config); the
    class preset and the AMBER instrument spec ride alongside. Helper properties
    surface the few values that depend on a class override (band quantile, fallbacks)
    so callers never re-merge.
    """

    defaults: CostDefaults
    preset: AssetClassCostPreset
    instrument: InstrumentCostSpec
    fee_schedule: FeeSchedule

    @property
    def band_quantile_q(self) -> float:
        """Class override of the RED band quantile when present (§5.5c)."""
        if self.preset.band_quantile_q is not None:
            return self.preset.band_quantile_q
        return self.defaults.band_quantile_q

    @property
    def fallback_half_spread_bps(self) -> float:
        """Class-level conservative half-spread fallback (§5.3)."""
        return self.preset.fallback_half_spread_bps

    @property
    def fallback_impact_k_fraction(self) -> float:
        """Class-level conservative impact coefficient (§5.4)."""
        return self.preset.fallback_impact_k_fraction


_FEE_SCHEDULES: dict[str, FeeSchedule] = {
    BINANCE_USDM_FEE_SCHEDULE.venue: BINANCE_USDM_FEE_SCHEDULE,
}


def resolve_config(
    instrument: InstrumentCostSpec,
    *,
    defaults: CostDefaults | None = None,
    preset: AssetClassCostPreset | None = None,
    fee_schedule: FeeSchedule | None = None,
) -> ResolvedCostConfig:
    """Merge RED defaults + class preset + AMBER instrument into one frozen config (§4).

    ``defaults`` is the SOLE RED entry point (Invariant C-cfg). ``preset`` is the
    SEPARATE per-class override channel; defaulted from the instrument's carry class.
    Raises if any RED/BLUE param name leaks into the instrument's AMBER field set.
    """
    # Invariant C-cfg: RED/BLUE must be unreachable from the per-instrument spec.
    assert_no_red_in_amber(INSTRUMENT_COST_SPEC_FIELDS, COST_PARAM_REGISTRY)

    defaults = defaults if defaults is not None else CostDefaults()
    preset = preset if preset is not None else DEFAULT_PRESETS[instrument.carry_class]
    if preset.carry_class != instrument.carry_class:
        raise ValueError(
            f"preset class {preset.carry_class!r} != instrument class {instrument.carry_class!r}"
        )
    if fee_schedule is None:
        fee_schedule = _FEE_SCHEDULES.get(instrument.fee_schedule_venue)
        if fee_schedule is None:
            raise ValueError(f"no fee schedule for venue {instrument.fee_schedule_venue!r}")

    return ResolvedCostConfig(
        defaults=defaults,
        preset=preset,
        instrument=instrument,
        fee_schedule=fee_schedule,
    )
