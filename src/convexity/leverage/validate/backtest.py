"""§7 worked-example reproduction harness (LEVERAGE.md §9 layout, §10 acceptance gate).

The §10 "Validation harness" gate requires that ``validate/backtest.py`` reproduces the Section 7
worked example BIT-FOR-BIT from its tabulated inputs. This module is that deliverable: it takes the
exact §7 input table (``σ̂_t``, ``SR_LCB_t``, ``c_t``, ``dd_t``/``m_t``, ``G_t^{cap}``, ``G_max``)
and drives the SAME live pillar modules (``ceilings/kelly_vol_target.py`` + ``combine/engine.py``)
that production uses, then asserts the four §7 steps reproduce:

  Step 1 — Kelly/vol-target term  ``c·SR_LCB/σ̂ = 0.25·0.091/0.0389 = 0.585``
  Step 2 — min() composition      ``G_t = min(0.585, 2.0, 1.0·1.5, 1.5) = 0.585``, binding = 'kelly'
  Step 3 — growth-capture check   ``c(2−c) at c=0.25 = 0.4375``
  Step 4 — governor-binding case  at ``dd=0.30, m_t=0.35`` → ``0.35·1.5 = 0.525`` < 0.585, binds

It is a ``validate/`` module (Invariant V, §10): it imports the live layers but no live layer
imports it. It reproduces the worked example by RUNNING the real composition, never by re-deriving
the arithmetic in a second place — so if a live-layer refactor breaks the §7 numbers, this harness
fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.leverage.ceilings.capacity import CapacityCeiling
from convexity.leverage.ceilings.kelly_vol_target import kelly_vol_target_term
from convexity.leverage.combine.engine import compose_g_t
from convexity.leverage.types import GovernorState, LeverageDecision


@dataclass(frozen=True)
class WorkedExampleInputs:
    """The §7 tabulated inputs, verbatim (LEVERAGE.md §7 input table).

    Every field is a CITED §7 row, not an invented number: ``sigma_hat`` = 3.89% weekly (28% ann ÷
    √52.18); ``sr_lcb`` = 0.091 (the §7 worked-example placeholder consistent with CPCV ≈0.91 ann ÷
    √52.18, shaded for the CI margin — §7 explicitly labels this a placeholder, the ACTUAL deploy
    number is ``max(0, block_bootstrap_ci(...)[0])``); the rest are the §7 illustrative ceilings.
    """

    sigma_hat: float = 0.0389
    sr_lcb: float = 0.091
    c_t: float = 0.25
    g_cap: float = 2.0
    m_t_base: float = 1.0
    g_max: float = 1.5
    # §7 Step 4 governor-binding contrast:
    dd_governor: float = 0.30
    m_t_governor: float = 0.35


@dataclass(frozen=True)
class WorkedExampleResult:
    """Reproduction outcome of the §7 worked example (LEVERAGE.md §7, §10 gate).

    ``base`` is Step 2's decision (kelly binds, ``G_t=0.585``); ``governor`` is Step 4's contrast
    (governor binds, ``G_t=0.525``); ``growth_capture`` is Step 3's ``c(2−c)=0.4375``. ``matches``
    is True iff all three reproduce their §7 targets to bit-for-bit rounding.
    """

    base: LeverageDecision
    governor: LeverageDecision
    growth_capture: float
    matches: bool


# The §7 target numbers (what a builder must reproduce), kept next to the check they gate.
_G_T_BASE = 0.585
_G_T_GOVERNOR = 0.525
_GROWTH_CAPTURE = 0.4375


def _cap(g_cap: float) -> CapacityCeiling:
    return CapacityCeiling(g_cap=g_cap, binding_symbol="cap_illustrative", n_unshortable_zeroed=0)


def reproduce_worked_example(
    inputs: WorkedExampleInputs | None = None,
) -> WorkedExampleResult:
    """Reproduce the §7 worked example bit-for-bit through the live modules (LEVERAGE.md §10 gate).

    Drives ``kelly_vol_target_term`` + ``compose_g_t`` (the SAME production path) on the §7 input
    table and returns both the base (kelly-binds) and governor-binds decisions plus the Step-3
    growth-capture value. ``matches`` asserts, to §7's tabulated 3-decimal rounding, that:
      Step 1/2: base ``G_t == 0.585`` and ``binding_ceiling == 'kelly'``;
      Step 3:   ``growth_capture == 0.4375`` at ``c=0.25``;
      Step 4:   governor-binds ``G_t == 0.525`` and ``binding_ceiling == 'governor'``.
    The harness reproduces by EXECUTION, never by restating the arithmetic (§10, Invariant K-CI
    spirit — one source of truth for the composition).
    """
    inp = inputs or WorkedExampleInputs()

    g_target = kelly_vol_target_term(inp.c_t, inp.sr_lcb, inp.sigma_hat)

    base = compose_g_t(
        g_target,
        _cap(inp.g_cap),
        GovernorState(dd_t=0.0, m_t=inp.m_t_base, terminal=False),
        g_max=inp.g_max,
    )
    governor = compose_g_t(
        g_target,
        _cap(inp.g_cap),
        GovernorState(dd_t=inp.dd_governor, m_t=inp.m_t_governor, terminal=False),
        g_max=inp.g_max,
    )

    matches = (
        round(base.g_t, 3) == _G_T_BASE
        and base.binding_ceiling == "kelly"
        and round(governor.g_t, 3) == _G_T_GOVERNOR
        and governor.binding_ceiling == "governor"
        and round(g_target.growth_capture, 4) == _GROWTH_CAPTURE
    )

    return WorkedExampleResult(
        base=base,
        governor=governor,
        growth_capture=g_target.growth_capture,
        matches=matches,
    )
