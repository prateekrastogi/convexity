"""Derive every limit FROM the validated distribution + the ARL math (§11.1) — MC-authoritative.

Spec: OPS §11.1 (derive ``(μ0,σ0)`` per stream + the limits from a SYSTEM validation report + a target
ARL0), §6.6 (solve ``(k,h)`` against the TWO-SIDED ARL0, MC authoritative), §6.7c (Shewhart ``L_s`` is
the empirical-tail quantile). The limits are written to the RED config and FROZEN; derived ONCE per
validated config, never live (Invariant LIM). This module is part of the MODEL — never imported by the
live layers (Invariant V).

The load-bearing fact (§6.6): solving the bare one-sided Siegmund formula for ARL0=370 lands on
``h≈3.5`` and roughly HALVES the real limit (doubling the false-alarm rate) — the bug this module
exists to forbid. We solve for ``h`` so the **two-sided** ARL0 = the target, which returns ``h≈4.77``.
The Siegmund ``÷2`` is the analytic seed/bracket; the MC two-arm OR-chart is authoritative on
disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.detectors.cusum import siegmund_arl_oneside, two_sided_arl0_from_oneside
from convexity.ops.validate.arl import cusum_arl_mc, shewhart_Ls_empirical


@dataclass(frozen=True)
class DerivedCusumLimit:
    """A derived CUSUM limit (§6.6): ``(k, h)`` + the analytic seed + the authoritative MC ARL0."""

    k: float
    h: float
    arl0_analytic_two_sided: float   # the Siegmund ÷2 seed/cross-check
    arl0_mc_two_sided: float | None  # the authoritative MC value (None if MC skipped)


def solve_h_for_two_sided_arl0(k: float, target_two_sided_arl0: float, *,
                               h_lo: float = 1.0, h_hi: float = 12.0,
                               tol: float = 1e-4) -> float:
    """Solve for ``h`` so the ANALYTIC two-sided ARL0 == target (§6.6) — the seed/bracket for the MC.

    Bisection on ``two_sided_arl0(k,h) = siegmund_oneside(k,h)/2``. For ``k=0.5, target=370`` this
    returns ``h≈4.77`` (NOT the ≈3.5 the bare one-sided solve gives — the bug §6.6 forbids). The MC
    solver :func:`solve_h_for_two_sided_arl0_mc` refines it where the dependence/fat-tail correction
    matters.
    """
    def two_sided(h: float) -> float:
        return two_sided_arl0_from_oneside(siegmund_arl_oneside(k, h, shift=0.0))

    lo, hi = h_lo, h_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = two_sided(mid)
        if abs(val - target_two_sided_arl0) <= tol * target_two_sided_arl0:
            return mid
        if val < target_two_sided_arl0:
            lo = mid       # ARL0 increases with h → need a larger h
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_h_for_two_sided_arl0_mc(k: float, target: float, *, n_runs: int = 1500,
                                  seed: int = 0, stream_fn=None,
                                  h_lo: float = 3.0, h_hi: float = 7.0,
                                  iters: int = 14, max_len: int = 200_000) -> float:
    """Solve for ``h`` against the AUTHORITATIVE MC two-sided ARL0 (§6.6/§6.7) — the actual two-arm
    OR-chart on a (possibly dependent/fat-tailed) in-control synthetic.

    Bisection on the MC ARL0; the analytic :func:`solve_h_for_two_sided_arl0` is the initial bracket.
    Where the MC and the ``÷2`` disagree, this value is pinned (Invariant ARL). ``stream_fn`` supplies
    the in-control synthetic so the limit can be solved against the dependent/fat-tailed stream
    (§6.7d). ``max_len`` caps each MC run (a smaller cap keeps a fat-tailed solve tractable; keep it
    comfortably above the target ARL0 so in-control runs are not truncated).
    """
    lo, hi = h_lo, h_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        est = cusum_arl_mc(k, mid, shift=0.0, n_runs=n_runs, seed=seed, stream_fn=stream_fn,
                           max_len=max_len)
        if est.mean < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def derive_cusum_limit(target_two_sided_arl0: float = 370.0, k: float = 0.5, *,
                       use_mc: bool = False, n_runs: int = 1500, seed: int = 0,
                       stream_fn=None) -> DerivedCusumLimit:
    """Derive the CUSUM ``(k, h)`` for a target two-sided ARL0 (§6.6/§11.1).

    With ``use_mc=False`` (fast path) ``h`` is the analytic-seed solve (``h≈4.77`` for 370) and the MC
    field is ``None``; with ``use_mc=True`` ``h`` is refined against the authoritative MC two-sided
    ARL0 (slower; used at promotion time). The analytic ARL0 is always recorded as the cross-check.
    """
    h_seed = solve_h_for_two_sided_arl0(k, target_two_sided_arl0)
    arl0_analytic = two_sided_arl0_from_oneside(siegmund_arl_oneside(k, h_seed, shift=0.0))
    if not use_mc:
        return DerivedCusumLimit(k=k, h=h_seed, arl0_analytic_two_sided=arl0_analytic,
                                 arl0_mc_two_sided=None)
    h_mc = solve_h_for_two_sided_arl0_mc(k, target_two_sided_arl0, n_runs=n_runs, seed=seed,
                                         stream_fn=stream_fn)
    arl0_mc = cusum_arl_mc(k, h_mc, shift=0.0, n_runs=n_runs, seed=seed, stream_fn=stream_fn).mean
    return DerivedCusumLimit(k=k, h=h_mc, arl0_analytic_two_sided=arl0_analytic,
                             arl0_mc_two_sided=arl0_mc)


def derive_shewhart_Ls(arl0_target: float = 370.0, *, dof: float | None = None) -> float:
    """Derive the Shewhart ``L_s`` as the EMPIRICAL / fitted-t tail quantile (§6.7c), not Gaussian 3σ.

    The |z| quantile whose in-control exceedance == 1/ARL0. Under a fat-tailed in-control residual this
    is the t(``dof``) quantile (e.g. ≈4.27 for t(5) at ARL0=370), so the chart pages at the design rate
    instead of flooding (which the Gaussian ``L_s=3`` would on the fat-tailed stream)."""
    return shewhart_Ls_empirical(arl0_target, dof=dof)


def vif_inflated_h(k: float, target_two_sided_arl0: float, rho: float) -> float:
    """Alternative to prewhitening (§6.7b route ii): inflate ``h`` by the measured VIF so the REALIZED
    two-sided ARL0 on the RAW autocorrelated stream hits the target.

    ``VIF=(1+ρ)/(1-ρ)``; the inflated target one-sided ARL0 is raised by VIF before solving. v1 default
    is to PREWHITEN (route i) so the §6.6 numbers stand; this is provided for the raw-stream route.
    """
    vif = (1.0 + rho) / (1.0 - rho)
    return solve_h_for_two_sided_arl0(k, target_two_sided_arl0 * vif)


__all__ = [
    "DerivedCusumLimit",
    "solve_h_for_two_sided_arl0",
    "solve_h_for_two_sided_arl0_mc",
    "derive_cusum_limit",
    "derive_shewhart_Ls",
    "vif_inflated_h",
]
