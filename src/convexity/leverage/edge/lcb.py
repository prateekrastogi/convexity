"""Confidence-adjusted edge ``SR_LCB_t`` (LEVERAGE.md §2.5–§2.6, the load-bearing sizing input).

The entire Kelly sizing term is a function of ``SR_LCB_t``, a *deflated, lower-confidence-bound
per-observation Sharpe* — NEVER the raw sample Sharpe (P2, Invariant K-LCB). Its construction is
the single most load-bearing pointer in the whole spec (§2.6):

    SR_LCB_t = 0                          if DeflatedSharpe.deflate(...).passed is False
             = max(0, ci_low)             otherwise,
    (ci_low, ci_high) = convexity.system.validate.robustness.block_bootstrap_ci(path_returns, ...)

This module does NOT define its own bootstrap or DSR math (Invariant K-CI): the CI is a pass-through
of the REAL ``block_bootstrap_ci`` in ``convexity.system.validate.robustness`` (never
``research/xsec/cpcv.PathSharpeSummary``, which has no CI field, §2.6), and the gate is a
pass-through of ``DeflatedSharpe.deflate(...).passed`` via ``research/xsec/validate.deflated_sharpe``
(the weekly-cadence wrapper, §2.6). Both read the SAME weekly net-of-cost series, so the CI lower
bound and the DSR gate can never silently reference different data (§2.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.system.validate.robustness import block_bootstrap_ci

from research.xsec.validate import deflated_sharpe

from convexity.leverage.config.defaults import LeverageDefaults


@dataclass(frozen=True)
class EdgeLCB:
    """OUTPUT of :func:`sr_lcb_t` (LEVERAGE.md §2.6).

    ``sr_lcb`` is the deployable ``SR_LCB_t = max(0, ci_low)`` (or ``0`` when the DSR gate fails).
    ``ci_low``/``ci_high`` are the raw bootstrap CI bounds (pre-floor). ``dsr_passed`` records the
    ``DeflatedSharpe.deflate(...).passed`` gate outcome; ``gated_to_zero`` True when the gate forced
    ``SR_LCB_t = 0`` regardless of the CI (§8 edge case: DSR failing ⇒ book runs flat).
    """

    sr_lcb: float
    ci_low: float
    ci_high: float
    dsr_passed: bool
    gated_to_zero: bool


def sr_lcb_t(
    path_returns: np.ndarray,
    trial_sharpes_ann: np.ndarray,
    defaults: LeverageDefaults,
    *,
    n_trials_effective: float | None = None,
    k_distinct: int | None = None,
    dsr_defaults=None,
) -> EdgeLCB:
    """The exact §2.6 verified call path for ``SR_LCB_t`` (LEVERAGE.md §2.5–§2.6, Invariant K-CI/K-LCB).

    ``path_returns`` = the promoted config's per-week net-of-cost return series (the SAME series the
    DSR gate reads, §2.6). ``trial_sharpes_ann`` = the annualized Sharpes of every config in the
    search family (the trial multiplicity the DSR deflates by, §2.6) — this is what forces
    ``SR_LCB_t`` to be sourced from the deflation object, not a bare ``mean(R)/std(R)`` (Invariant
    K-LCB; a raw scalar Sharpe passed here as ``path_returns`` would not be a series and cannot
    bypass the CI/DSR objects).

    Steps (§2.6):
      1. Gate: ``DeflatedSharpe.deflate(...).passed`` via ``research/xsec/validate.deflated_sharpe``
         (the weekly-cadence wrapper). If ``passed is False`` → ``SR_LCB_t = 0`` unconditionally
         (Invariant K-LCB, §8 edge case), regardless of how large the raw Sharpe looks.
      2. CI: ``block_bootstrap_ci`` (the real robustness function) with RED ``L_boot=6``, ``n_boot``,
         one-sided ``ci_level=0.90``, fixed seed, default per-observation-Sharpe statistic (§2.6).
      3. ``SR_LCB_t = max(0, ci_low)`` — the P6-style floor (never negative/NaN, §2.6).

    Widening the CI (more estimation uncertainty ⇒ lower ``ci_low``) can only LOWER ``SR_LCB_t``,
    never raise it, holding σ̂_t fixed (Invariant K-LCB) — an emergent property of returning
    ``max(0, ci_low)``.

    ``path_returns`` MUST be a 1-D return SERIES (length >= 2), not a pre-computed scalar Sharpe:
    passing a bare scalar (an attempt to bypass the CI/DSR objects) raises ``ValueError`` (Invariant
    K-LCB — the confidence-adjusted edge is never sized off a raw point-estimate Sharpe).
    """
    r = np.asarray(path_returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError(
            "path_returns must be a 1-D weekly return SERIES (len >= 2), not a scalar Sharpe "
            "(LEVERAGE §2.6, Invariant K-LCB — never size off a raw point estimate)"
        )

    # --- Step 1: the DSR gate (Invariant K-LCB) — pass-through of DeflatedSharpe.deflate(...).passed
    dsr = deflated_sharpe(
        r,
        np.asarray(trial_sharpes_ann, dtype=float),
        n_trials_effective=n_trials_effective,
        k_distinct=k_distinct,
        defaults=dsr_defaults,
    )
    if not dsr.passed:
        return EdgeLCB(sr_lcb=0.0, ci_low=float("nan"), ci_high=float("nan"),
                       dsr_passed=False, gated_to_zero=True)

    # --- Step 2: the block-bootstrap CI (Invariant K-CI) — the REAL robustness function, RED params
    ci_low, ci_high = block_bootstrap_ci(
        r,
        block_length=defaults.bootstrap_block_length,
        n_boot=defaults.bootstrap_n,
        ci_level=defaults.bootstrap_ci_level,
        seed=defaults.bootstrap_seed,
    )

    # --- Step 3: max(0, ci_low) floor (§2.6, P6)
    return EdgeLCB(
        sr_lcb=max(0.0, float(ci_low)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        dsr_passed=True,
        gated_to_zero=float(ci_low) <= 0.0,
    )
