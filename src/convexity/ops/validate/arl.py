"""Average Run Length / false-alarm-rate calculators — the MC value is AUTHORITATIVE (§6.6/§6.7).

Spec: OPS §6.6 (the ARL math), §6.7 (ARL under serial dependence + fat tails). The closed forms
(Siegmund CUSUM, Lucas–Saccucci EWMA) are design-time APPROXIMATIONS and the bracketing seed; the
pinned RED limits and their stated ARL0/ARL1 are the dependence-aware Monte-Carlo solver output, which
is authoritative wherever the two disagree (Invariant ARL). This module is part of the MODEL — it is
NEVER imported by the live layers (Invariant V); ``scipy`` lives here, not in the engine closure.

Key facts the tests pin:
  * two-sided CUSUM ARL0: the analytic ``÷2`` (one-sided Siegmund 743 → two-sided 370) is the SEED;
    the MC two-sided value (the actual two-arm OR-chart on an in-control stream) is authoritative.
  * realized ARL0 on an AR(1)-Student-t in-control synthetic lands within CI of 370 only AFTER
    prewhitening (§6.7a) / VIF-inflation (§6.7b); FAILS on the raw autocorrelated residual.
  * Shewhart ``L_s`` is the EMPIRICAL / fitted-t tail quantile (§6.7c), not Gaussian 3σ.
  * SPRT ASN is the boundary-weighted Wald form (ASN_H0≈99, ASN_H1≈118), not 40–50.
  * the 2-of-2 windowed-AND joint ARL0 is a dependence-aware MC number (§6.6a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from convexity.ops.detectors.cusum import Cusum, siegmund_arl_oneside, two_sided_arl0_from_oneside
from convexity.ops.detectors.page_hinkley import PageHinkley
from convexity.ops.detectors.sprt import asn as sprt_asn
from convexity.ops.detectors.sprt import llr_increment


@dataclass(frozen=True)
class ArlEstimate:
    """An MC ARL estimate with its CI (Invariant ARL records ARL0/ARL1 + the CI)."""

    mean: float
    ci_lo: float
    ci_hi: float
    n_runs: int

    def contains(self, target: float) -> bool:
        """Whether ``target`` is within the CI (the realized-ARL0 test, §6.7d)."""
        return self.ci_lo <= target <= self.ci_hi


# --------------------------------------------------------------------------- in-control synthetics

def iid_gaussian_stream(n: int, rng: np.random.Generator) -> np.ndarray:
    """An IID N(0,1) in-control stream — the closed-form cross-check synthetic (§6.7d)."""
    return rng.standard_normal(n)


def ar1_student_t_stream(n: int, rho: float, dof: float, rng: np.random.Generator) -> np.ndarray:
    """An AR(1)-Student-t in-control stream (§6.7d): autocorrelated AND fat-tailed.

    ``x_i = rho·x_{i-1} + innovation`` with Student-t(dof) innovations scaled to unit *marginal*
    variance, so the standardized residual has the right ρ and fat tail. This is the production-fidelity
    synthetic the realized ARL0 must hold on after prewhitening (Invariant ARL).
    """
    t_var = dof / (dof - 2.0)                       # variance of a standard t(dof)
    innov_scale = np.sqrt((1.0 - rho * rho) / t_var)  # so the marginal AR(1) variance is 1
    innov = rng.standard_t(dof, size=n) * innov_scale
    x = np.empty(n)
    x[0] = innov[0] / np.sqrt(max(1e-9, 1.0 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + innov[i]
    return x


# --------------------------------------------------------------------------- CUSUM ARL (MC)

def cusum_arl_mc(k: float, h: float, *, shift: float = 0.0, n_runs: int = 2000,
                 max_len: int = 200_000, seed: int = 0,
                 stream_fn=None) -> ArlEstimate:
    """MC estimate of the TWO-SIDED CUSUM ARL (§6.6) — AUTHORITATIVE over the analytic ``÷2``.

    Runs the actual two-arm OR-chart (``C+ > h`` OR ``C- > h``) on independent in-control streams (or
    a shifted stream for ARL1) and averages the run length to the first alarm. ``shift`` adds a mean
    shift in σ units (0 == ARL0). ``stream_fn(n, rng)`` supplies the in-control synthetic (default IID
    Gaussian); pass :func:`ar1_student_t_stream`-backed generator for the §6.7d dependent/fat-tailed
    check (which FAILS at the IID limit and holds after prewhitening).
    """
    rng = np.random.default_rng(seed)
    gen = stream_fn if stream_fn is not None else (lambda n, r: iid_gaussian_stream(n, r))
    runs = np.empty(n_runs)
    for r in range(n_runs):
        c_plus = 0.0
        c_minus = 0.0
        x = gen(max_len, rng) + shift
        rl = max_len
        for i in range(max_len):
            z = x[i]
            c_plus = max(0.0, c_plus + z - k)
            c_minus = max(0.0, c_minus - z - k)
            if c_plus > h or c_minus > h:
                rl = i + 1
                break
        runs[r] = rl
    mean = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(n_runs))
    return ArlEstimate(mean=mean, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, n_runs=n_runs)


def cusum_arl0_siegmund_two_sided(k: float, h: float) -> float:
    """The analytic two-sided ARL0 SEED: one-sided Siegmund ``÷2`` (§6.6). Cross-check, NOT pinned.

    ``(k=0.5,h=4.77)`` → one-sided ≈743 → two-sided ≈370. The MC :func:`cusum_arl_mc` is authoritative.
    """
    return two_sided_arl0_from_oneside(siegmund_arl_oneside(k, h, shift=0.0))


def cusum_arl1_siegmund(k: float, h: float, shift: float = 1.0) -> float:
    """The analytic ARL1 SEED at the design shift (one-directional, drives one arm only, §6.6).

    ``(k=0.5,h=4.77)`` at 1σ → ≈9.9 (the MC value ≈10.4 is authoritative, §10/Invariant WE)."""
    return siegmund_arl_oneside(k, h, shift=shift)


# --------------------------------------------------------------------------- EWMA / Shewhart ARL

def ewma_arl_mc(lam: float, L: float, *, shift: float = 0.0, n_runs: int = 2000,
                max_len: int = 200_000, seed: int = 0, stream_fn=None) -> ArlEstimate:
    """MC estimate of the EWMA ARL (§6.3/§6.6). Uses the asymptotic limit (the steady-state limit
    warm-up settles into). Cross-check against the Lucas–Saccucci tables (≈370 at ``(0.10, 2.70)``)."""
    rng = np.random.default_rng(seed)
    gen = stream_fn if stream_fn is not None else (lambda n, r: iid_gaussian_stream(n, r))
    limit = L * np.sqrt(lam / (2.0 - lam))
    runs = np.empty(n_runs)
    for r in range(n_runs):
        s = 0.0
        x = gen(max_len, rng) + shift
        rl = max_len
        for i in range(max_len):
            s = lam * x[i] + (1.0 - lam) * s
            if abs(s) > limit:
                rl = i + 1
                break
        runs[r] = rl
    mean = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(n_runs))
    return ArlEstimate(mean=mean, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, n_runs=n_runs)


def shewhart_arl0_gaussian(L_s: float) -> float:
    """The Gaussian Shewhart ARL0 ``1/(2Φ(-L_s))`` (§6.4) — the THIN-TAILED reference only.

    ``L_s=3 ⇒ 370``, ``L_s=3.5 ⇒ 2150``. On a fat-tailed stream this OVER-states the ARL0 (§6.7c);
    the empirical-quantile :func:`shewhart_Ls_empirical` is authoritative."""
    return 1.0 / (2.0 * stats.norm.cdf(-L_s))


def shewhart_Ls_empirical(arl0_target: float, *, dof: float | None = None) -> float:
    """The EMPIRICAL / fitted-t Shewhart ``L_s`` (§6.7c): the |z| quantile whose exceedance == 1/ARL0.

    Under a Student-t(``dof``) in-control residual this is the t-quantile, NOT the Gaussian 3σ — so the
    chart pages at the design rate on the fat-tailed stream. With ``dof=None`` it returns the Gaussian
    quantile (the thin-tailed reference). The tail probability is split two-sided (``1/(2·ARL0)`` each
    tail), matching ``ARL0 = 1/(2·P(|z|>L_s))``.
    """
    tail = 1.0 / (2.0 * arl0_target)
    if dof is None:
        return float(stats.norm.ppf(1.0 - tail))
    # standardize the t so its marginal variance is 1, then take the standardized quantile
    scale = np.sqrt((dof - 2.0) / dof)
    return float(stats.t.ppf(1.0 - tail, dof) * scale)


def shewhart_arl0_empirical_mc(L_s: float, dof: float, *, n: int = 2_000_000,
                               seed: int = 0) -> float:
    """MC ARL0 of a Shewhart chart on a unit-variance Student-t(``dof``) stream (§6.7c).

    Demonstrates the Gaussian formula OVER-states the ARL0 on a fat-tailed stream: feeding the Gaussian
    ``L_s=3`` here returns an ARL0 far below 370 (the chart floods), while :func:`shewhart_Ls_empirical`
    restores it. ARL0 = 1 / (empirical exceedance rate of ``|z| > L_s``).
    """
    rng = np.random.default_rng(seed)
    scale = np.sqrt((dof - 2.0) / dof)
    z = rng.standard_t(dof, size=n) * scale
    rate = float(np.mean(np.abs(z) > L_s))
    return float("inf") if rate <= 0.0 else 1.0 / rate


# --------------------------------------------------------------------------- SPRT ASN

@dataclass(frozen=True)
class SprtAsn:
    """The boundary-weighted Wald ASN pair (§6.5). ``asn_h0`` ≈ 99, ``asn_h1`` ≈ 118 at the worked
    operating point (NOT the dropped-weighting 40–50)."""

    asn_h0: float
    asn_h1: float


def sprt_asn_table(p0: float, p1: float, alpha: float, beta: float) -> SprtAsn:
    """The boundary-weighted Wald ASN (§6.5). For ``(0.55,0.45,0.05,0.10)`` → (≈99, ≈118)."""
    return SprtAsn(
        asn_h0=sprt_asn(p0, p0, p1, alpha, beta),
        asn_h1=sprt_asn(p1, p0, p1, alpha, beta),
    )


def sprt_asn_mc(p_true: float, p0: float, p1: float, alpha: float, beta: float, *,
                n_runs: int = 5000, max_len: int = 100_000, seed: int = 0) -> ArlEstimate:
    """MC of the bare SPRT's sample number under a true hit-rate ``p_true`` (§6.5).

    Reproduces the boundary-weighted ASN within CI (and FAILS the old 40–50 figure). Run with
    ``p_true=p0`` for ASN_H0, ``p_true=p1`` for ASN_H1.
    """
    rng = np.random.default_rng(seed)
    from convexity.ops.detectors.sprt import wald_boundaries
    A, B = wald_boundaries(alpha, beta)
    runs = np.empty(n_runs)
    for r in range(n_runs):
        llr = 0.0
        n = max_len
        for j in range(max_len):
            hit = 1 if rng.random() < p_true else 0
            llr += llr_increment(hit, p0, p1)
            if llr >= A or llr <= B:
                n = j + 1
                break
        runs[r] = n
    mean = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(n_runs))
    return ArlEstimate(mean=mean, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, n_runs=n_runs)


def cusum_sprt_arl0_mc(p0: float, p1: float, h_edge: float, *, n_runs: int = 2000,
                       max_len: int = 200_000, seed: int = 0) -> ArlEstimate:
    """MC ARL0 of the CUSUM-SPRT (§6.5a) under the in-control hit-rate ``p0`` — the single pinned number
    that enters the §6.6 panel budget (NOT the bare per-epoch α). The G recursion climbs on misses."""
    rng = np.random.default_rng(seed)
    runs = np.empty(n_runs)
    for r in range(n_runs):
        g = 0.0
        rl = max_len
        for j in range(max_len):
            hit = 1 if rng.random() < p0 else 0
            g = max(0.0, g + llr_increment(hit, p0, p1))  # +ve on a miss, -ve on a hit (§6.5a)
            if g > h_edge:
                rl = j + 1
                break
        runs[r] = rl
    mean = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(n_runs))
    return ArlEstimate(mean=mean, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, n_runs=n_runs)


# --------------------------------------------------------------------------- 2-of-2 joint ARL0

def and_rule_joint_arl0_mc(k: float, h: float, ph_delta: float, ph_lambda: float, m_conf: int, *,
                           n_runs: int = 2000, max_len: int = 20_000, seed: int = 0) -> ArlEstimate:
    """Dependence-aware MC joint ARL0 of the 2-of-2 windowed-AND cost confirmation (§6.6a).

    Runs the CUSUM and Page-Hinkley recursions on the SAME in-control stream (so their dependence is
    captured) and alarms only when both arms are breached within ``M_conf`` consecutive samples. The
    result exceeds either single-detector ARL0 and is ≤ the independence product bound — it enters the
    §6.6 panel budget as the cost stream's single source (the product is logged only as the bound).
    """
    rng = np.random.default_rng(seed)
    runs = np.empty(n_runs)
    for r in range(n_runs):
        cusum = Cusum(k, h)
        ph = PageHinkley(ph_delta, ph_lambda)
        last_a: int | None = None
        last_b: int | None = None
        x = iid_gaussian_stream(max_len, rng)
        rl = max_len
        for i in range(max_len):
            cs = cusum.update(x[i])
            ps = ph.update(x[i])
            if cs.statistic > h:
                last_a = i
            if ps.statistic > ph_lambda:
                last_b = i
            if last_a is not None and last_b is not None and abs(last_a - last_b) < m_conf:
                rl = i + 1
                break
        runs[r] = rl
    mean = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(n_runs))
    return ArlEstimate(mean=mean, ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se, n_runs=n_runs)


__all__ = [
    "ArlEstimate",
    "iid_gaussian_stream",
    "ar1_student_t_stream",
    "cusum_arl_mc",
    "cusum_arl0_siegmund_two_sided",
    "cusum_arl1_siegmund",
    "ewma_arl_mc",
    "shewhart_arl0_gaussian",
    "shewhart_Ls_empirical",
    "shewhart_arl0_empirical_mc",
    "SprtAsn",
    "sprt_asn_table",
    "sprt_asn_mc",
    "cusum_sprt_arl0_mc",
    "and_rule_joint_arl0_mc",
]
