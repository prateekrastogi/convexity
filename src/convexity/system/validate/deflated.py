"""DeflatedSharpe — PSR/DSR over the EFFECTIVE joint trials + all singularity guards (SYSTEM §7).

THE CENTERPIECE. Four modules can each pass their own gate and the composition can still have NO
edge, because the live config was selected from a large joint search space. The system Sharpe must
be deflated by that JOINT multiplicity. The math that stops the four-green-modules-but-no-edge trap.

Everything inside the z-test runs in PER-OBSERVATION units (Section 7.2a): the Bailey–López de Prado
non-normality denominator terms (γ3·SR, ((γ4−1)/4)·SR²) are only correct at annualization factor
f = 1, so we de-annualize every Sharpe (``SR_pp = SR_annual / sqrt(ppy)``) BEFORE it touches the
gate; annualization is reporting-only, applied AFTER the decision. Feeding an annualized Sharpe
inflates z by ≈sqrt(ppy) with no offset and manufactures a spurious near-1 DSR — rejected as
``annualization_mismatch`` (Invariant DSR-ANN).

The guards (Section 7.3a, Invariants DSR-GUARD / DSR-MINTRL):
  1. var_floor = (1 + SR_pp²/2)/n_obs (Lo-2002, per-observation); K_distinct ≥ K_min before PASS.
  2. N_sys = 1 ⇒ SR*₀ = 0, DSR = PSR(0) (the BLdP convention, never Φ⁻¹(0)).
  3. MinTRL track-length floor: n_obs ≥ max(MinTRL, n_obs_floor) or insufficient_track_record.

The §7.3b worked examples (a FAIL at n_obs=108 vs MinTRL≈1114 and a PASS contrast at n_obs=400 vs
MinTRL≈56) are reproduced exactly by these formulas.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from convexity.system.types import DSRResult

_EULER_GAMMA = 0.5772156649015329  # ɣ (Euler–Mascheroni)


# --------------------------------------------------------------------------- effective trials N_sys


def _correlation_matrix(returns: list[np.ndarray]) -> np.ndarray:
    """K×K correlation matrix of the ledger configs' OOS path-return series (Section 7.1)."""
    k = len(returns)
    n = min(len(r) for r in returns)
    mat = np.vstack([np.asarray(r, dtype=float)[:n] for r in returns])
    if n < 2:
        return np.eye(k)
    c = np.corrcoef(mat)
    return np.atleast_2d(c)


def _n_eff_kish(corr: np.ndarray) -> float:
    """Kish-style effective count N_eff = K / (1 + (K−1)·rho_bar), rho_bar CLIPPED at 0 (Section 7.1).

    A negative mean off-diagonal correlation is treated as 0 ⇒ rho_bar = 0 ⇒ N_eff = K (anti-
    correlated configs count as fully independent trials — the de-correlated floor, Invariant DSR-MONO).
    """
    k = corr.shape[0]
    if k <= 1:
        return float(k)
    off = corr[~np.eye(k, dtype=bool)]
    rho_bar = max(0.0, float(off.mean()))  # clip at 0
    return k / (1.0 + (k - 1) * rho_bar)


def _n_clusters_single_linkage(corr: np.ndarray, rho_cl: float) -> int:
    """Single-linkage cluster count at threshold ``rho_cl`` on |corr| (Section 7.1, Invariant DSR-MONO).

    Two configs share a cluster iff their OOS-return |correlation| ≥ rho_cl, transitively (single
    linkage). Tightening rho_cl (toward 1) splits clusters ⇒ MORE clusters ⇒ larger N_sys (monotone
    non-decreasing) — so loosening can only shrink the penalty, which is the forbidden quiet-shrink.
    """
    k = corr.shape[0]
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    absc = np.abs(corr)
    for i in range(k):
        for j in range(i + 1, k):
            if absc[i, j] >= rho_cl:
                union(i, j)
    return len({find(i) for i in range(k)})


def effective_trials(trial_returns: list[np.ndarray], rho_cl: float) -> tuple[float, int]:
    """N_sys = max(N_eff, N_clusters, 1) (Section 7.1). Returns (n_sys, k_distinct).

    ``k_distinct`` = the de-correlated trial count (= N_clusters), checked against K_min (guard 1).
    Uses the LARGER of the Kish count and the cluster count (the more conservative). Bounded
    1 ≤ N_sys ≤ K: you cannot pay a multiple-test tax for trials you never ran.
    """
    k = len(trial_returns)
    if k == 0:
        return 1.0, 0
    if k == 1:
        return 1.0, 1
    corr = _correlation_matrix(trial_returns)
    n_eff = _n_eff_kish(corr)
    n_clusters = _n_clusters_single_linkage(corr, rho_cl)
    n_sys = max(n_eff, float(n_clusters), 1.0)
    return min(n_sys, float(k)), n_clusters


# --------------------------------------------------------------------------- PSR / SR*₀ / MinTRL


def _psr_denom(sr_pp: float, skew: float, kurt: float, eps: float) -> tuple[float, bool]:
    """The PSR/z-test variance term denom = 1 − γ3·SR_pp + ((γ4−1)/4)·SR_pp² (Section 7.2b).

    ``kurt`` is NON-EXCESS (raw) kurtosis (normal ⇒ 3, so the normal denom is 1 + 0.5·SR_pp²).
    Floored at ``eps`` to avoid division blow-up (recorded as psr_denom_floored).
    """
    denom = 1.0 - skew * sr_pp + ((kurt - 1.0) / 4.0) * sr_pp * sr_pp
    if denom <= eps:
        return eps, True
    return denom, False


def psr(sr_pp: float, sr_star: float, n_obs: int, skew: float, kurt: float, eps: float) -> tuple[float, bool]:
    """PSR(SR*) = Φ((SR_pp − SR*)·sqrt(n_obs − 1)/sqrt(denom)) (Section 7.2b).

    Both ``sr_pp`` and ``sr_star`` MUST be PER-OBSERVATION (the denom's non-normality terms are only
    correct at f = 1). The (n_obs − 1) is OUTSIDE the sqrt of denom. Returns (psr, denom_floored).
    """
    denom, floored = _psr_denom(sr_pp, skew, kurt, eps)
    z = (sr_pp - sr_star) * math.sqrt(max(n_obs - 1, 0)) / math.sqrt(denom)
    return float(norm.cdf(z)), floored


def sr_star0(var_trials: float, n_sys: float) -> float:
    """The luck-max benchmark SR*₀ from N_sys independent trials (Section 7.3, PER-OBSERVATION units).

    SR*₀ = sqrt(Var[SR_trials])·[(1−ɣ)·Φ⁻¹(1 − 1/N_sys) + ɣ·Φ⁻¹(1 − 1/(N_sys·e))]. The N_sys = 1
    case is handled by the caller (BLdP convention SR*₀ = 0) BEFORE calling this, so Φ⁻¹(0) is never
    evaluated here (Section 7.3a guard 2).
    """
    sd = math.sqrt(max(var_trials, 0.0))
    q1 = norm.ppf(1.0 - 1.0 / n_sys)
    q2 = norm.ppf(1.0 - 1.0 / (n_sys * math.e))
    return sd * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2)


def min_trl(sr_pp: float, sr_star: float, skew: float, kurt: float, z_gate: float, eps: float) -> float:
    """The Bailey–Borwein Minimum Track-Record Length (Section 7.3a guard 3).

    MinTRL = 1 + denom·(Z_gate/(SR_pp − SR*₀))² — the algebraic inverse of the z-test: the n at which
    (SR_pp − SR*₀)·sqrt(n − 1)/sqrt(denom) = Z_gate. Uses SR*₀ (not 0) as the benchmark so the floor
    reflects the DEFLATED gate. If SR_pp ≤ SR*₀ the config cannot clear the benchmark at any length ⇒
    MinTRL = +∞ (fails on edge, not length).
    """
    edge = sr_pp - sr_star
    if edge <= 0.0:
        return math.inf
    denom, _ = _psr_denom(sr_pp, skew, kurt, eps)
    return 1.0 + denom * (z_gate / edge) ** 2


# --------------------------------------------------------------------------- the gate


class DeflatedSharpe:
    """PSR/DSR over N_sys with the singularity guards (SYSTEM §7/§12). The reported headline is the
    DSR (| n_obs vs MinTRL), never the vanilla Sharpe (Invariant DSR)."""

    def deflate(
        self,
        path_returns: np.ndarray,
        trial_sharpes: np.ndarray,
        n_obs: int,
        skew: float,
        kurt: float,
        n_trials_effective: float,
        cfg,
        *,
        sr_annual: float | None = None,
        trial_sharpes_annual: bool = True,
        k_distinct: int | None = None,
        path_returns_are_per_obs: bool = True,
        assume_annualized_input: bool = False,
    ) -> DSRResult:
        """Compute PSR and DSR for the composed system (Section 7.2 input contract).

        ``path_returns`` = the promoted config's NON-OVERLAPPING net-of-cost system return series;
        ``n_obs`` = COUNT of those non-overlapping observations (NOT bar/path/trade count);
        ``trial_sharpes`` = the per-LEDGER-CONFIG OOS Sharpes (cross-CONFIG, never cross-CPCV-path);
        ``skew``/``kurt`` are of ``path_returns`` (``kurt`` = NON-EXCESS / raw, normal ⇒ 3);
        ``n_trials_effective`` = N_sys (consistent with Var[SR_trials]=var(trial_sharpes), Section
        7.1). Everything is converted to PER-OBSERVATION units before the z-test (Section 7.2a):
        ``SR_pp = SR_annual / sqrt(ppy)`` and each trial Sharpe likewise; ``var_floor =
        (1 + SR_pp²/2)/n_obs``. Annualization is reporting-only.

        ``assume_annualized_input=True`` is the negative test: it (incorrectly) feeds the ANNUALIZED
        Sharpe straight into the z-test and is rejected as ``annualization_mismatch``.
        """
        defaults = cfg.defaults
        ppy = float(defaults.ppy)
        eps = defaults.psr_denom_eps
        z_gate = defaults.z_gate
        reasons: list[str] = []

        # ---- per-observation SR_pp (Section 7.2a) ----
        if sr_annual is None:
            mu = float(np.mean(path_returns))
            sd = float(np.std(path_returns, ddof=1)) if len(path_returns) > 1 else 0.0
            sr_per_obs = mu / sd if sd > 0.0 else 0.0
            sr_annual = sr_per_obs * math.sqrt(ppy)
        if assume_annualized_input:
            # The forbidden path: feed the annualized Sharpe into the per-observation z-test.
            return DSRResult(
                psr=float("nan"), dsr=float("nan"), sr_star0=float("nan"), sr_pp=sr_annual,
                var_trials=float("nan"), var_floored=False, n_sys=n_trials_effective, n_obs=n_obs,
                ppy=ppy, min_trl=float("nan"), n_obs_sufficient=False, k_distinct=k_distinct or 0,
                psr_denom_floored=False, passed=False, reason_codes=("annualization_mismatch",),
            )
        sr_pp = sr_annual / math.sqrt(ppy)

        # ---- per-observation trial Sharpes + cross-config variance (Section 7.2a / 7.2) ----
        ts = np.asarray(trial_sharpes, dtype=float)
        ts_pp = ts / math.sqrt(ppy) if trial_sharpes_annual else ts
        var_trials = float(np.var(ts_pp, ddof=1)) if ts_pp.size > 1 else 0.0

        # ---- variance floor (Lo-2002, per-observation; guard 1) ----
        var_floor = (1.0 + sr_pp * sr_pp / 2.0) / max(n_obs, 1)
        var_floored = False
        if defaults.var_floor_enabled and var_floor > var_trials:
            var_trials = var_floor
            var_floored = True

        # ---- distinct-trials guard (guard 1) ----
        kd = k_distinct if k_distinct is not None else ts_pp.size
        insufficient_trials = kd < defaults.k_min

        # ---- SR*₀ (BLdP N_sys=1 convention; guard 2) ----
        n_sys = max(float(n_trials_effective), 1.0)
        if n_sys <= 1.0:
            sr_star = 0.0  # no maximum-of-N inflation to deflate; never Φ⁻¹(0)
        else:
            sr_star = sr_star0(var_trials, n_sys)

        # ---- PSR / DSR (per-observation z-test) ----
        dsr, denom_floored = psr(sr_pp, sr_star, n_obs, skew, kurt, eps)
        psr_zero, _ = psr(sr_pp, 0.0, n_obs, skew, kurt, eps)
        if denom_floored:
            reasons.append("psr_denom_floored")

        # ---- MinTRL track-length floor (guard 3) ----
        mtrl = min_trl(sr_pp, sr_star, skew, kurt, z_gate, eps)
        floor = max(mtrl, float(defaults.n_obs_floor))
        n_obs_sufficient = bool(n_obs >= floor)

        # ---- the gate ----
        passed = bool(dsr >= defaults.dsr_threshold)
        if insufficient_trials:
            passed = False
            reasons.append("insufficient_trials")
        if not n_obs_sufficient:
            passed = False
            reasons.append("insufficient_track_record")

        return DSRResult(
            psr=psr_zero,
            dsr=dsr,
            sr_star0=sr_star,
            sr_pp=sr_pp,
            var_trials=var_trials,
            var_floored=var_floored,
            n_sys=n_sys,
            n_obs=n_obs,
            ppy=ppy,
            min_trl=mtrl,
            n_obs_sufficient=n_obs_sufficient,
            k_distinct=kd,
            psr_denom_floored=denom_floored,
            passed=passed,
            reason_codes=tuple(reasons),
        )
