"""Turbulence — Kritzman–Mancini Mahalanobis distance on the dof-normalized ``d_norm`` (REGIME §2.3).

Absorption flags a STRUCTURAL co-movement shift; turbulence flags a statistically UNUSUAL day: a
current cross-asset return vector far from its trailing mean given the trailing covariance.

  d_t      = (y_t − mu_t) Σ_t^{-1} (y_t − mu_t)^T      (squared Mahalanobis distance)
  d_norm_t = d_t / n_t                                 (per-dimension; E≈1 regardless of n_t)
  turb_pct = causal percentile of d_norm_t in its OWN trailing window (P6)

Faithful to §2.3:
  * **TURBNORM** — the percentile is taken on ``d_norm = d_t/n_t``, never raw ``d_t``: because
    ``n_t`` varies bar-to-bar (staleness drops, session closes) and ``d_t ~ χ²_{n_t}`` scales with
    ``n_t``, a raw-``d_t`` percentile would move when the panel shrinks. ``d_norm`` is ~scale-stable.
  * **COND** — ``Σ_t`` is Ledoit–Wolf constant-correlation shrunk before inversion, and we require
    ``W_turb ≥ c_cov·n_t``; below that we fall back to the robust diagonal (per-asset variance)
    inverse and flag it.
  * **CONDNUM** — the shrunk ``Σ_t`` passes the post-shrink ``cond ≤ cond_max`` gate; on breach we
    fall back to the diagonal inverse (``cond_breach_turb``) — never invert a near-singular matrix
    into a nonsense ``d_t`` spike.

The covariance inverse is ``O(n³)`` and runs only on the regime cadence (Invariant CHEAP). No ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve

from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.measures._shrinkage import lw_shrink_covariance


def causal_percentile(value: float, history: np.ndarray) -> float:
    """Percentile of ``value`` within its OWN trailing ``history`` (P6) — the MIDRANK fraction.

    Returns a value in ``[0, 1]``. Uses the midrank ``(count(<) + 0.5·count(==))/n`` rather than the
    plain ``count(≤)`` so that NEAR-CONSTANT / autocorrelated history (the typical calm regime, where
    consecutive aggregate-vol or ``d_norm`` values cluster tightly) does not tie-inflate the
    percentile toward 1.0 and manufacture spurious escalations. On exchangeable data the midrank is
    unbiased at 0.5 — the correct in-control null the §3.6 ARL budget assumes. With no history the
    bar is treated as median (0.5) — neutral, never a spurious escalation. Distribution-free (§2.3).
    """
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return 0.5
    below = float(np.count_nonzero(h < value))
    equal = float(np.count_nonzero(h == value))
    return (below + 0.5 * equal) / float(h.size)


def mahalanobis_sq(y: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """``d_t = (y − mu) Σ^{-1} (y − mu)^T`` — the squared Mahalanobis distance (REGIME §2.3)."""
    diff = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    return float(diff @ cov_inv @ diff)


@dataclass
class TurbulenceState:
    """Trailing causal history of ``d_norm`` for the percentile (REGIME §2.3).

    The percentile compares like with like across panel-size changes because it is taken on the
    dof-normalized ``d_norm`` (TURBNORM), so a single pooled trailing series is correct.
    """

    d_norm_history: list[float] = field(default_factory=list)

    def push(self, d_norm: float, max_len: int) -> None:
        self.d_norm_history.append(float(d_norm))
        if len(self.d_norm_history) > max_len:
            del self.d_norm_history[: len(self.d_norm_history) - max_len]


@dataclass(frozen=True)
class TurbulenceResult:
    """``turb_pct`` (the trigger) plus the §8 diagnostics (REGIME §2.3)."""

    turb_pct: float
    fresh: bool
    diag: dict[str, float]


def compute_turbulence(
    returns: np.ndarray,
    *,
    cfg: PolicyDefaults,
    state: TurbulenceState,
) -> TurbulenceResult:
    """One turbulence recompute on the current panel (REGIME §2.3). MUTATES ``state``.

    ``y_t`` is the current (last-row) return vector; ``mu_t``/``Σ_t`` are the trailing causal mean
    and (shrunk) covariance over the window. Emits ``turb_pct`` of ``d_norm = d_t/n_t`` (TURBNORM).
    """
    x = np.asarray(returns, dtype=float)
    w_turb, n_t = x.shape
    y = x[-1]
    mu = x.mean(axis=0)
    diag: dict[str, float] = {"n_t": float(n_t)}

    use_diagonal = False
    if w_turb < cfg.c_cov * n_t:
        use_diagonal = True
        diag["cov_window_deficient"] = 1.0

    if not use_diagonal:
        shrink = lw_shrink_covariance(x)
        diag["lw_intensity"] = shrink.intensity
        diag["cond_number"] = shrink.cond_number
        if shrink.cond_number > cfg.cond_max:
            use_diagonal = True
            diag["cond_breach_turb"] = 1.0
        else:
            d_t = _mahalanobis_via_solve(y, mu, shrink.matrix)

    if use_diagonal:
        var = x.var(axis=0, ddof=0)
        var = np.where(var > 0, var, 1e-12)
        diff = y - mu
        d_t = float(np.sum(diff * diff / var))  # robust diagonal (per-asset variance) inverse
        diag.setdefault("cond_number", 1.0)

    d_norm = d_t / float(n_t)
    diag["d_t"] = d_t
    diag["d_norm"] = d_norm

    turb_pct = causal_percentile(d_norm, np.asarray(state.d_norm_history, dtype=float))
    state.push(d_norm, cfg.w_turb_pct)  # push AFTER the percentile (today's value vs its OWN past)
    diag["turb_pct"] = turb_pct
    diag["fresh"] = 1.0
    return TurbulenceResult(turb_pct, fresh=True, diag=diag)


def _mahalanobis_via_solve(y: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """``d_t`` via a single linear solve (no explicit inverse) on the shrunk, conditioned ``Σ_t``."""
    diff = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    z = solve(cov, diff, assume_a="pos")
    return float(diff @ z)
