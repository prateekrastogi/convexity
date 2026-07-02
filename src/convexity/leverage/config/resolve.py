"""Single config merge point for the LEVERAGE pillar (LEVERAGE.md §3, Invariant C-cfg).

RED constants (``LeverageDefaults``) are unreachable from any per-instrument field — they enter
only via ``resolve_config(defaults=...)`` (§3). This module is the ONE place that constructs the
resolved config, mirroring COST/SYSTEM's ``resolve_config`` discipline. It also exposes the
weekly-cadence ``SystemDefaults`` shim that the DSR gate (``edge/lcb.py``) reads — reused verbatim
from ``research/xsec/validate.weekly_defaults`` (§2.6 reuses SYSTEM's deflation machinery, never
re-implements it).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from research.xsec.validate import weekly_defaults as _weekly_defaults

from convexity.leverage.config.defaults import LeverageDefaults


@dataclass(frozen=True)
class ResolvedLeverageConfig:
    """The resolved LEVERAGE config carrier (LEVERAGE.md §3).

    Holds ONLY the RED ``LeverageDefaults`` — there is deliberately no per-instrument field on this
    object (Invariant C-cfg): the capacity ceiling reads per-instrument AMBER/TEAL data through the
    ``CapacityInput`` list at call time, never through this config.
    """

    defaults: LeverageDefaults


def resolve_config(*, defaults: LeverageDefaults | None = None) -> ResolvedLeverageConfig:
    """The single RED merge point (LEVERAGE.md §3, Invariant C-cfg).

    RED constants enter ONLY here. Passing per-instrument overrides is impossible by construction —
    the signature accepts only a ``LeverageDefaults`` (a RED bundle), never an instrument spec.
    """
    return ResolvedLeverageConfig(defaults=defaults or LeverageDefaults())


def weekly_dsr_defaults(*, dsr_threshold: float = 0.95, k_min: int = 3, n_obs_floor: int = 24):
    """The weekly-cadence ``SystemDefaults`` the DSR gate reads (LEVERAGE.md §2.6).

    Thin pass-through to ``research/xsec/validate.weekly_defaults`` — the SAME weekly-tuned
    ``SystemDefaults`` (ppy≈52, k_min=3) that ``deflated_sharpe`` uses, so LEVERAGE's DSR gate and
    SYSTEM's promotion gate deflate on IDENTICAL cadence/floors (§2.6). LEVERAGE never re-derives
    any part of the SYSTEM §5.3–§5.5 test (Invariant K-CI/K-LCB).
    """
    return _weekly_defaults(dsr_threshold=dsr_threshold, k_min=k_min, n_obs_floor=n_obs_floor)


def with_c(config: ResolvedLeverageConfig, c_t: float) -> ResolvedLeverageConfig:
    """Return a copy of ``config`` — the fractional-Kelly dial is state, not a RED field (§6.2).

    ``c_t`` is the ONLY free sizing dial (Invariant SIZE-ONE) and is carried in ``RatchetState``,
    not baked into the RED defaults; this helper exists purely so tests can assert that the two
    §2.3 grid points are the only reachable values (``c_t`` must be in ``defaults.c_grid``).
    """
    if c_t not in config.defaults.c_grid:
        raise ValueError(
            f"c_t={c_t} not in the §2.3 grid {config.defaults.c_grid} (LEVERAGE §2.3/§6.2)"
        )
    return replace(config)
