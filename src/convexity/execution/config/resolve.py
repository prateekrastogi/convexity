"""resolve_config — the single merge point where RED enters EXECUTION (§4, Invariant C).

The ONE place RED :class:`PolicyDefaults` are merged with the per-class :class:`AssetClassPreset`,
the AMBER per-instrument :class:`InstrumentSpec`, and the AMBER :class:`VenueSpec` into the frozen
:class:`ResolvedConfig` every microstate / cost / policy module reads (§4).

RED is reachable ONLY through the ``defaults=`` argument (Invariant C); per-class overrides come
through the SEPARATE ``preset=`` argument and only when validated (Invariant T); the per-instrument
spec and the venue spec carry AMBER only. A RED value on the instrument spec is the bug this firewall
catches — :func:`assert_no_red_in_amber` runs over the instrument field names against the EXECUTION
param registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import assert_no_red_in_amber

from convexity.execution.config.asset_class import DEFAULT_PRESETS, AssetClassPreset
from convexity.execution.config.defaults import EXECUTION_PARAM_REGISTRY, PolicyDefaults
from convexity.execution.config.instrument import INSTRUMENT_SPEC_FIELDS, InstrumentSpec
from convexity.execution.config.venue import VenueSpec, venue_spec


@dataclass(frozen=True)
class ResolvedConfig:
    """The frozen, fully-merged config every EXECUTION module reads (§4).

    RED constants live on ``defaults`` (untouched by per-instrument config); the class preset, the
    AMBER instrument spec, and the AMBER venue spec ride alongside. Helper properties surface the few
    values that depend on a class override so callers never re-merge — and so a RED constant is never
    re-derived per symbol (Invariant T).
    """

    defaults: PolicyDefaults
    preset: AssetClassPreset
    instrument: InstrumentSpec
    venue: VenueSpec

    @property
    def funding_aware(self) -> bool:
        """True only for perps (§5.4 / §13): the funding-over-horizon term is priced (Invariant W)."""
        return self.preset.funding_aware

    @property
    def class_proxy_half_spread_bps(self) -> float:
        """The conservative half-spread floor the spread ladder falls through to (§11 rung 4)."""
        return self.preset.class_proxy_half_spread_bps

    @property
    def tick_bps_at(self) -> float:
        """One tick in bps vs the instrument's arrival mid is computed live; this exposes tick size."""
        return self.instrument.tick_size


def resolve_config(
    instrument: InstrumentSpec,
    *,
    defaults: PolicyDefaults | None = None,
    preset: AssetClassPreset | None = None,
    venue: VenueSpec | None = None,
) -> ResolvedConfig:
    """Merge RED defaults + class preset + AMBER instrument + AMBER venue into one frozen config (§4).

    ``defaults`` is the SOLE RED entry point (Invariant C). ``preset`` is the SEPARATE per-class
    override channel, defaulted from the instrument's asset class. Raises if any RED/BLUE EXECUTION
    param name leaks into the instrument's AMBER field set.
    """
    # Invariant C: RED/BLUE must be unreachable from the per-instrument spec.
    assert_no_red_in_amber(INSTRUMENT_SPEC_FIELDS, EXECUTION_PARAM_REGISTRY)

    defaults = defaults if defaults is not None else PolicyDefaults()
    preset = preset if preset is not None else DEFAULT_PRESETS[instrument.asset_class]
    if preset.asset_class != instrument.asset_class:
        raise ValueError(
            f"preset class {preset.asset_class!r} != instrument class {instrument.asset_class!r}"
        )
    venue = venue if venue is not None else venue_spec(instrument.venue)

    return ResolvedConfig(defaults=defaults, preset=preset, instrument=instrument, venue=venue)
