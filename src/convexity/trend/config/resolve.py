"""resolve_config — the SINGLE merge point where RED enters TREND, and where SYSTEM injects the
operational cap (TREND §3, §1.6a, Inv C, Inv HC).

The ONE place ``PolicyDefaults`` (RED) is merged with the per-class ``AssetClassPreset`` and the
per-instrument AMBER ``InstrumentSpec`` into the frozen ``ResolvedConfig`` every TREND module reads.
RED is reachable ONLY through ``defaults=`` (Inv C); per-class overrides come through the SEPARATE
``preset=`` argument; the instrument carries AMBER only.

This module also owns the SYSTEM-injected operational cap (§1.6a / Inv HC): the operational
maximum-hold cap is NOT a TREND-internal constant — SYSTEM supplies ``h_sys`` (its contractual hold
bound) and TREND exposes it as ``ResolvedConfig.tau_cap`` (alias ``H_op``). The engine force-emits
``stop_cause="time_cap"`` at the first bar the active clock reaches ``tau_cap``. A separate, possibly
LARGER, ``h_obs`` observation cap is exposed for survival analysis (roles 1/3); ``None`` means
``h_obs == tau_cap``.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import assert_no_red_in_amber
from convexity.core.types import InstrumentSpec

from convexity.trend.config.asset_class import (
    DEFAULT_PRESETS,
    AssetClassPreset,
    noise_ratio_for,
)
from convexity.trend.config.defaults import (
    TREND_PARAM_REGISTRY,
    PolicyDefaults,
)
from convexity.trend.config.instrument import INSTRUMENT_SPEC_FIELDS
from convexity.trend.config.tuning import TRAIL_MULTIPLIER_PRESETS


@dataclass(frozen=True)
class ResolvedConfig:
    """The frozen, fully-merged config every TREND module reads (§3).

    RED constants live on ``defaults`` (untouched by per-instrument config); the class preset and
    AMBER instrument ride alongside. ``tau_cap`` (== ``H_sys``, alias ``H_op``) is the SYSTEM-injected
    operational cap (Inv HC); ``h_obs`` is the (possibly larger) survival-analysis observation cap.
    ``active_clock`` is the clock those caps are measured under. The resolved trail multipliers,
    Kalman noise ratio, and detector kind are surfaced as properties so callers never re-merge.
    """

    defaults: PolicyDefaults
    preset: AssetClassPreset
    instrument: InstrumentSpec
    tau_cap: float  # H_op == H_sys, in units of active_clock (§1.6a / Inv HC)
    h_obs: float  # observation cap for survival (>= tau_cap); roles 1/3 (§1.6a)
    active_clock: str
    trail_multipliers: tuple[float, float, float, float]  # (tight, normal, loose, tail)

    # --- resolved scalars (class preset beats default, default otherwise) ---
    @property
    def H_op(self) -> float:  # noqa: N802 — spec alias
        """Alias for ``tau_cap`` (the operational cap, §1.6a)."""
        return self.tau_cap

    @property
    def kalman_noise_ratio(self) -> float:
        """Class-preset Kalman process/observation ratio (Inv E — never per-symbol)."""
        return noise_ratio_for(self.preset.kalman_class_preset, self.defaults.kalman_noise_ratio)

    @property
    def detector_kind(self) -> str:
        return self.defaults.detector_kind

    @property
    def arl0(self) -> int:
        return self.preset.arl0 if self.preset.arl0 is not None else self.defaults.arl0

    @property
    def vol_method(self) -> str:
        return (
            self.preset.vol_method
            if self.preset.vol_method is not None
            else self.defaults.vol_method
        )

    @property
    def gap_survival_sigmas(self) -> float:
        return (
            self.preset.gap_survival_sigmas
            if self.preset.gap_survival_sigmas is not None
            else self.defaults.gap_survival_sigmas
        )

    @property
    def trail_breakpoints_r(self) -> tuple[float, float, float]:
        return self.defaults.trail_breakpoints_r


def resolve_config(
    instrument: InstrumentSpec,
    *,
    h_sys: float,
    defaults: PolicyDefaults | None = None,
    preset: AssetClassPreset | None = None,
    h_obs: float | None = None,
) -> ResolvedConfig:
    """Merge RED defaults + class preset + AMBER instrument into one frozen config (§3).

    ``defaults`` is the SOLE RED entry point (Inv C). ``preset`` is the SEPARATE per-class channel;
    defaulted from the instrument's asset class. ``h_sys`` is the SYSTEM-injected operational cap
    (Inv HC) — REQUIRED, never a TREND-internal constant — and becomes ``tau_cap``. ``h_obs`` is the
    optional larger survival observation cap (defaults to ``h_sys``). Raises if any RED/BLUE name
    leaks into the instrument's AMBER field set, or if ``h_obs < h_sys``.
    """
    # Inv C: RED/BLUE must be unreachable from the per-instrument spec.
    assert_no_red_in_amber(INSTRUMENT_SPEC_FIELDS, TREND_PARAM_REGISTRY)

    defaults = defaults if defaults is not None else PolicyDefaults()
    preset = preset if preset is not None else DEFAULT_PRESETS[instrument.asset_class]
    if preset.asset_class != instrument.asset_class:
        raise ValueError(
            f"preset class {preset.asset_class!r} != instrument class {instrument.asset_class!r}"
        )

    if h_sys <= 0:
        raise ValueError("h_sys (SYSTEM operational cap, Inv HC) must be positive")
    if h_obs is None:
        h_obs = h_sys
    if h_obs < h_sys:
        raise ValueError(f"h_obs ({h_obs}) must be >= tau_cap=h_sys ({h_sys}) (§1.6a)")

    trail_name = preset.trail_preset if preset.trail_preset is not None else "balanced"
    trail_multipliers = TRAIL_MULTIPLIER_PRESETS.get(
        trail_name,
        (
            defaults.trail_mult_tight,
            defaults.trail_mult_normal,
            defaults.trail_mult_loose,
            defaults.trail_mult_tail,
        ),
    )

    active_clock = (
        preset.active_clock if preset.active_clock is not None else defaults.active_clock
    )

    return ResolvedConfig(
        defaults=defaults,
        preset=preset,
        instrument=instrument,
        tau_cap=float(h_sys),
        h_obs=float(h_obs),
        active_clock=active_clock,
        trail_multipliers=trail_multipliers,
    )
