"""The single config merge point (REGIME §4, Invariant C).

``resolve_config`` is the ONLY place a RED constant or a BLUE selection enters the pillar. RED is
reachable solely through ``defaults=`` (never from a per-instrument ``InstrumentSpec`` — Invariant
C); BLUE comes in as a class-level selection from SYSTEM's promoted grid, validated against the
registered ``BLUE_GRID`` so an unlisted value cannot sneak in. The resulting ``ResolvedConfig`` is
frozen and is the deterministic input to every measure and the state machine (Invariant DET).

The ``param_registry`` records each parameter's governance bucket so the Invariant-C / NOFIT tests
can introspect the pillar's degrees of freedom and assert ``K_regime <= K_regime_max`` (§4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry
from convexity.regime.config.defaults import BLUE_GRID, DEFAULTS, PolicyDefaults


@dataclass(frozen=True)
class ResolvedConfig:
    """The resolved, frozen policy fed to every measure + the state machine (REGIME §4, §8).

    ``policy`` holds the RED/BLUE constants (with any promoted BLUE selection applied). It is the
    sole config surface the engine reads — no per-instrument field reaches it (Invariant C).
    """

    policy: PolicyDefaults
    engine_version: str = "regime-1.0.0"


def resolve_config(
    *,
    defaults: PolicyDefaults | None = None,
    blue: Mapping[str, Any] | None = None,
    engine_version: str = "regime-1.0.0",
) -> ResolvedConfig:
    """Merge RED defaults with a validated BLUE selection (REGIME §4, Invariant C).

    ``defaults`` is the RED canonical set (RED enters ONLY here). ``blue`` is a class-level
    selection of BLUE parameters promoted by SYSTEM — each key must be in ``BLUE_GRID`` and each
    value must be one of that family's enumerated candidates, else it is an unlisted (forbidden)
    degree of freedom (NOFIT). Per-instrument config can never reach this function.
    """
    policy = defaults if defaults is not None else DEFAULTS
    if blue:
        for key, value in blue.items():
            if key not in BLUE_GRID:
                raise ValueError(f"BLUE key {key!r} is not a registered regime grid family (NOFIT)")
            if value not in BLUE_GRID[key]:
                raise ValueError(
                    f"BLUE value {value!r} for {key!r} is not in the registered grid "
                    f"{BLUE_GRID[key]!r} (NOFIT — no free continuous fit)"
                )
        policy = _apply_blue(policy, blue)
    return ResolvedConfig(policy=policy, engine_version=engine_version)


def _apply_blue(policy: PolicyDefaults, blue: Mapping[str, Any]) -> PolicyDefaults:
    """Apply a validated BLUE selection onto the RED policy (the only mutable surface, §4.1)."""
    updates: dict[str, Any] = {}
    if "kappa" in blue:
        updates["kappa"] = float(blue["kappa"])
    if "d_min" in blue:
        updates["d_min"] = int(blue["d_min"])
    if "tau_regime" in blue:
        updates["tau_regime"] = str(blue["tau_regime"])
    return replace(policy, **updates) if updates else policy


def param_registry() -> StaticParamRegistry:
    """Governance registry of every REGIME parameter (REGIME §4.1, Invariants C/NOFIT).

    BLUE families carry their finite grid so ``blue_grid_cardinality`` is the per-pillar trial count
    SYSTEM deflates against; the Ledoit–Wolf intensities are deliberately ABSENT (TEAL, closed-form,
    not searched DOF — §2.2/§2.3/§4.1).
    """
    specs: tuple[ParamSpec, ...] = (
        ParamSpec("g_of_state", Bucket.RED, "state -> gross multiplier ladder", "REGIME §3.1"),
        ParamSpec("g_floor", Bucket.RED, "lowest g; REGIME never flats (Inv FLOOR)", "REGIME §3.1"),
        ParamSpec("kappa", Bucket.BLUE, "absorption top-k fraction", "REGIME §2.2", BLUE_GRID["kappa"]),
        ParamSpec("w_ar", Bucket.RED, "absorption correlation window", "REGIME §2.2"),
        ParamSpec("w_ar_short", Bucket.RED, "delta_AR short trailing-mean window", "REGIME §2.2"),
        ParamSpec("w_ar_long", Bucket.RED, "delta_AR long trailing-mean + sigma_AR window", "REGIME §2.2"),
        ParamSpec("c_ar", Bucket.RED, "absorption obs-per-dim floor (ARFLOOR)", "REGIME §2.2"),
        ParamSpec("frac_dn", Bucket.RED, "fixed-n_t stratum / freeze band (ARDISPNORM)", "REGIME §2.2"),
        ParamSpec("w_turb", Bucket.RED, "turbulence mean/cov window", "REGIME §2.3"),
        ParamSpec("c_cov", Bucket.RED, "turbulence obs-per-dim floor (COND)", "REGIME §2.3"),
        ParamSpec("w_turb_pct", Bucket.RED, "d_norm causal-percentile window", "REGIME §2.3"),
        ParamSpec("w_pct", Bucket.RED, "sigma_agg causal-percentile window", "REGIME §2.1"),
        ParamSpec("w_vov", Bucket.RED, "vol-of-vol window", "REGIME §2.1"),
        ParamSpec("cond_max", Bucket.RED, "post-shrink condition gate (CONDNUM)", "REGIME §2.3"),
        ParamSpec("d_min", Bucket.BLUE, "global minimum dwell", "REGIME §3.3", BLUE_GRID["d_min"]),
        ParamSpec("d_liq_min", Bucket.RED, "minute-level liq dwell (3.6 b')", "REGIME §3.6"),
        ParamSpec("rebal_min_frac", Bucket.RED, "rebalance deadband (COSTGATE clause b)", "REGIME §5.2"),
        ParamSpec("rebal_cost_cap", Bucket.RED, "re-gross-up cost cap (COSTGATE clause c)", "REGIME §5.2"),
        ParamSpec("tau_regime", Bucket.BLUE, "decision cadence", "REGIME §3.5", BLUE_GRID["tau_regime"]),
        ParamSpec(
            "vol_stressed_entry", Bucket.BLUE, "vol STRESSED-entry level", "REGIME §3.2",
            BLUE_GRID["vol_stressed_entry"],
        ),
        ParamSpec(
            "corr_stressed_entry", Bucket.BLUE, "corr STRESSED-entry level", "REGIME §3.2",
            BLUE_GRID["corr_stressed_entry"],
        ),
        ParamSpec(
            "turb_stressed_entry", Bucket.BLUE, "turb STRESSED-entry level", "REGIME §3.2",
            BLUE_GRID["turb_stressed_entry"],
        ),
        ParamSpec(
            "liq_stressed_entry", Bucket.BLUE, "liq STRESSED-entry level", "REGIME §3.2",
            BLUE_GRID["liq_stressed_entry"],
        ),
    )
    return StaticParamRegistry(specs)
