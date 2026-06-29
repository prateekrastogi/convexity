"""The single config merge point (OPS §4, Invariant C/LIM).

``resolve_config`` is the ONLY place a RED limit or a BLUE selection enters OPS. RED is reachable
solely through ``defaults=`` (never from a per-instrument ``InstrumentSpec`` — Invariant C); BLUE
comes in as a class-level selection from SYSTEM's promoted grid, validated against the registered
``BLUE_GRID`` so an unlisted value cannot sneak in (NOFIT). The frozen in-control reference comes from
a SYSTEM validation-report fixture (``in_control.py``), NOT from any live estimate (Invariant LIM).

The resulting :class:`ResolvedOpsConfig` is the deterministic input to every detector, the
reconcile map, the brake, and the rollout gate — no per-instrument field reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from convexity.core.types import AssetClass
from convexity.ops.config.asset_class import apply_preset
from convexity.ops.config.defaults import BLUE_GRID, DEFAULTS, OpsDefaults
from convexity.ops.config.in_control import fixture_system_report, in_control_from_report
from convexity.ops.config.severity import DEFAULT_SEVERITY_CONFIG, SeverityConfig
from convexity.ops.types import InControlParams, StreamName


@dataclass(frozen=True)
class ResolvedOpsConfig:
    """The resolved, frozen OPS config (OPS §4). The sole config surface the engine reads.

    ``defaults`` are the RED SPC limits (after any class preset + BLUE selection); ``in_control`` the
    frozen per-stream reference from SYSTEM's report; ``severity`` the AMBER routing. No per-instrument
    field is present (Invariant C).
    """

    defaults: OpsDefaults
    in_control: Mapping[StreamName, InControlParams]
    severity: SeverityConfig = DEFAULT_SEVERITY_CONFIG
    engine_version: str = "ops-1.0.0"


def _apply_blue(defaults: OpsDefaults, blue: Mapping[str, Any]) -> OpsDefaults:
    """Apply a validated BLUE selection onto the RED defaults (§4 BLUE). Each value must be in grid."""
    out = defaults
    if "cusum_kh" in blue:
        k, h = blue["cusum_kh"]
        out = replace(out, cusum=replace(out.cusum, k=k, h=h))
    if "ewma_lambda_L" in blue:
        lam, L = blue["ewma_lambda_L"]
        out = replace(out, ewma=replace(out.ewma, lam=lam, L=L))
    if "sprt_alpha_beta" in blue:
        a, b = blue["sprt_alpha_beta"]
        out = replace(out, sprt=replace(out.sprt, alpha=a, beta=b))
    if "arl0_target" in blue:
        out = replace(out, panel_arl0_target=float(blue["arl0_target"]))
    return out


def resolve_config(
    *,
    defaults: OpsDefaults | None = None,
    system_report: Mapping[str, Mapping[str, Any]] | None = None,
    asset_class: AssetClass | None = None,
    blue: Mapping[str, Any] | None = None,
    severity: SeverityConfig | None = None,
    engine_version: str = "ops-1.0.0",
) -> ResolvedOpsConfig:
    """Merge RED defaults + a class preset + a validated BLUE selection + the frozen in-control ref.

    ``defaults`` is the RED canonical set (RED enters ONLY here). ``system_report`` is the SYSTEM
    validation report the in-control reference is loaded from (defaults to the §10 fixture so OPS is
    testable in isolation). ``asset_class`` selects a class-level preset (§4 BLUE). ``blue`` is a
    class-level grid selection — each key must be a registered ``BLUE_GRID`` family and each value one
    of its candidates (NOFIT). Per-instrument config can never reach this function (Invariant C).
    """
    base = defaults if defaults is not None else DEFAULTS
    base = apply_preset(base, asset_class)
    if blue:
        for key, value in blue.items():
            if key not in BLUE_GRID:
                raise ValueError(f"BLUE key {key!r} is not a registered OPS grid family (NOFIT)")
            if value not in BLUE_GRID[key]:
                raise ValueError(
                    f"BLUE value {value!r} for {key!r} is not in the registered grid "
                    f"{BLUE_GRID[key]!r} (NOFIT — no free continuous fit)"
                )
        base = _apply_blue(base, blue)
    report = system_report if system_report is not None else fixture_system_report()
    in_control = in_control_from_report(report)
    return ResolvedOpsConfig(
        defaults=base,
        in_control=in_control,
        severity=severity if severity is not None else DEFAULT_SEVERITY_CONFIG,
        engine_version=engine_version,
    )


__all__ = ["ResolvedOpsConfig", "resolve_config"]
