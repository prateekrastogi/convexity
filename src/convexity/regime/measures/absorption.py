"""Absorption ratio + the Kritzman–Li standardized shift ``delta_AR`` (REGIME §2.2).

The centerpiece systemic measure. When assets co-move, a few principal components capture most of
the cross-asset variance and diversification collapses; the absorption ratio ``AR_t`` is the share
of total variance in the top-``k`` eigenvectors of the rolling CORRELATION matrix.

Faithful to every pin in §2.2:
  * **ARCOND** — the correlation matrix is Ledoit–Wolf constant-correlation shrunk BEFORE the
    eigendecomposition, so ``AR_t`` is not MP-biased-high at marginal ``T/N``. The intensity
    ``δ_AR`` is TEAL (closed-form, in ``diag``, never a knob).
  * **TRACE** — the denominator is ``n_t`` by the correlation-matrix trace identity (the shrink
    preserves the unit diagonal), NOT a numerically re-summed eigenvalue total. Tiny-negative
    eigenvalues are clamped to 0; ``AR_t`` is a cluster-invariant eigenvalue sum so needs no
    eigenvector tie-break.
  * **ARFLOOR** — require ``W_AR ≥ c_AR·n_t``; below it ``C_t`` is rank-deficient and ``g^{corr}``
    is FROZEN at its last valid value (``ar_window_deficient``), never reset to 1.0.
  * **CONDNUM** — the shrunk ``C_t`` passes ``cond ≤ cond_max``; on breach ``g^{corr}`` FREEZES
    (``cond_breach_corr``).
  * **DISP** — ``delta_AR = (AR_bar_short − AR_bar_long)/sigma_AR`` where ``sigma_AR`` is the std of
    the RAW single-bar AR LEVEL over the long window (the Kritzman–Li displacement), NOT the
    mean-difference standard error. Substituting the SE would inflate every ``delta_AR`` ~3.7× and
    silently retune the correlation thresholds — forbidden.
  * **ARDISPNORM** — the AR-level series feeding ``AR_bar_short``/``AR_bar_long``/``sigma_AR`` is
    taken over the fixed-``n_t`` STRATUM, defined as a CURRENT-BAR-ANCHORED membership band: the set
    of all trailing bars whose ``n_t`` lies within ``frac_dn`` of the CURRENT bar's ``n_t``
    (``|n_trailing/n_t − 1| ≤ frac_dn``, equivalently ``|log n_trailing − log n_t| ≤ log1p(frac_dn)``),
    selected PER CURRENT BAR from one persistent chronological ``(n_t, AR_t)`` history POOLED across
    non-contiguous sessions/days — NOT a fixed global quantization grid (which would wrongly exclude
    in-band bars near a grid edge). When the current ``n_t`` steps outside the band around the prior
    sizes (i.e. too few trailing bars fall inside the band) ``g^{corr}`` FREEZES (``ar_nt_changed``)
    until the matching band-stratum has enough history; because the history persists across days the
    freeze is transient. The pooling band and the freeze band are ONE AND THE SAME test (one
    ``frac_dn`` criterion, not two).

The eigendecomposition is ``O(n³)`` and runs only on the regime cadence (Invariant CHEAP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.measures._shrinkage import lw_shrink_correlation


def absorption_ratio(eigenvalues: np.ndarray, n_t: int, k: int) -> float:
    """``AR_t = (Σ_{j≤k} λ_j)/n_t`` — top-k share with the TRACE-identity denominator (REGIME §2.2).

    ``eigenvalues`` must be the descending, clamped eigenvalues of the SHRUNK correlation matrix.
    The denominator is ``n_t`` exactly (the shrink preserves the unit diagonal so ``Σλ = n_t``); it
    is NOT a numerical re-sum of all eigenvalues (Invariant TRACE).
    """
    k = max(1, min(int(k), int(n_t)))
    top = float(np.sum(eigenvalues[:k]))
    return top / float(n_t)


def top_k(kappa: float, n_t: int) -> int:
    """``k = ceil(kappa·n_t)`` (REGIME §2.2). e.g. kappa=0.20, n=80 -> k=16."""
    return max(1, int(math.ceil(kappa * n_t)))


# A degenerate-dispersion floor for sigma_AR. AR_t levels live in [0, 1] and a genuine day-to-day
# AR dispersion is of order 1e-1..1e-2 (the §6.1 worked example uses sigma_AR=0.050). When the AR
# LEVEL is constant across the stratum (a stationary calm null — the AR matrix, hence AR_t, does not
# move bar-to-bar) the true sigma_AR is 0; the rolling std then returns only BLAS-dependent round-off
# (~1e-16, machine-eps of the level), and so does (AR_bar_short − AR_bar_long). Their ratio is a
# meaningless 0/0 that lands on a HARMLESS sign on Accelerate but on +2.0..+3.0 on OpenBLAS — a false
# CRISIS cut on calm data. Per §2.2 / the §3.6a labelling note, under a calm null delta_AR is
# "approximately mean-zero": a stratum with no real AR dispersion produces NO detectable displacement,
# so delta_AR = 0. We floor any sigma_AR negligible RELATIVE to the AR-level scale (~1e-12 of it — 4
# orders below a genuine 0.050 dispersion, 4 orders above the 1e-16 round-off) to that calm value.
# This is a numerical-degeneracy guard (same family as the eigenvalue tiny-negative clamp and the
# lam_min<=0 condition guard), NOT a retune of the RED threshold and NOT platform branching.
_SIGMA_AR_REL_FLOOR = 1e-12


def _sigma_ar_degenerate(sigma_ar: float, ar_level_scale: float) -> bool:
    """True when ``sigma_AR`` is only round-off relative to the AR-level scale (zero-dispersion null)."""
    return sigma_ar <= _SIGMA_AR_REL_FLOOR * max(1.0, abs(ar_level_scale))


def standardized_shift(ar_bar_short: float, ar_bar_long: float, sigma_ar: float) -> float:
    """The Kritzman–Li standardized DISPLACEMENT ``delta_AR`` (REGIME §2.2, Invariant DISP).

    ``= (AR_bar_short − AR_bar_long)/sigma_AR`` with ``sigma_AR`` the std of the RAW single-bar AR
    LEVEL over the long window — NOT the mean-difference SE. Reproduces the §6.1 worked example:
    ``(0.710 − 0.600)/0.050 = +2.20``.

    A ``sigma_AR`` that is only round-off relative to the AR level (a stationary calm null with a
    constant AR level) yields NO detectable displacement -> ``delta_AR = 0`` (REGIME §3.6a calm-null
    labelling), guarding the 0/0 BLAS-noise ratio rather than emitting a spurious cut.
    """
    if _sigma_ar_degenerate(sigma_ar, ar_bar_long):
        return 0.0
    return (ar_bar_short - ar_bar_long) / sigma_ar


def _in_band(n_trailing: int, n_t: int, frac_dn: float) -> bool:
    """Is a trailing bar at ``n_trailing`` within ``frac_dn`` of the CURRENT ``n_t``? (REGIME §2.2).

    The current-bar-anchored membership test — the SAME ``frac_dn`` criterion that gates the freeze,
    so the pooling band and the freeze band are one and the same test (Invariant ARDISPNORM). A bar
    is in the stratum iff ``|n_trailing/n_t − 1| ≤ frac_dn`` (equivalently, to first order,
    ``|log n_trailing − log n_t| ≤ log1p(frac_dn)``). This is NOT a fixed global quantization grid:
    at ``n_t=80`` with ``frac_dn=0.05`` the band is ``[76, 84]`` and 76/78/80/84 are ALL in-band (a
    fixed grid would split them into different keys and wrongly exclude genuinely in-band bars).
    """
    if n_trailing <= 0 or n_t <= 0:
        return False
    # Compare the ratio directly (|n_trailing/n_t − 1| ≤ frac_dn) with a tiny float tolerance so the
    # exact upper edge n_trailing = n_t·(1+frac_dn) lands IN-band; the log form is the same criterion.
    return abs(n_trailing / n_t - 1.0) <= frac_dn + 1e-12


@dataclass
class AbsorptionState:
    """Stateful, persistent chronological AR-level history pooled across non-contiguous sessions.

    ``history`` is one trailing list of ``(n_t, AR_t)`` pairs accumulated ACROSS calendar days (NOT a
    single contiguous run). The fixed-``n_t`` stratum is NOT stored as a fixed grid of buckets; it is
    derived PER CURRENT BAR by filtering this history to the bars whose ``n_t`` lies within
    ``frac_dn`` of the current ``n_t`` (see :func:`_in_band`). On a mixed crypto+equity book ``n_t``
    structurally alternates, so a recurring size re-uses every in-band trailing observation on
    re-entry — making the ARDISPNORM freeze transient (REGIME §2.2). ``last_delta_ar`` holds the
    value to FREEZE on a deficient window / condition breach / ``n_t`` step.
    """

    history: list[tuple[int, float]] = field(default_factory=list)
    last_delta_ar: float = 0.0

    def push(self, n_t: int, ar_level: float, max_history: int) -> None:
        """Append ``(n_t, AR_t)`` to the persistent cross-day history, trimmed to ``max_history``."""
        self.history.append((int(n_t), float(ar_level)))
        if len(self.history) > max_history:
            del self.history[: len(self.history) - max_history]

    def stratum(self, n_t: int, frac_dn: float, max_len: int) -> list[float]:
        """The current-bar-anchored fixed-``n_t`` stratum: in-band AR levels, newest ``max_len``.

        Filters the chronological history to bars within ``frac_dn`` of the CURRENT ``n_t`` (the same
        criterion that gates the freeze), pooled across non-contiguous sessions, capped at ``max_len``
        (= the long window) keeping the most recent in-band observations.
        """
        levels = [ar for (n_tr, ar) in self.history if _in_band(n_tr, n_t, frac_dn)]
        if len(levels) > max_len:
            levels = levels[len(levels) - max_len :]
        return levels


@dataclass(frozen=True)
class AbsorptionResult:
    """``delta_AR`` (the trigger), its freeze/validity flags, and the §8 diagnostics (REGIME §2.2)."""

    delta_ar: float  # the standardized-displacement trigger (frozen value if not fresh)
    fresh: bool  # False when frozen (deficient window / cond breach / n_t step)
    diag: dict[str, float]


def compute_absorption(
    returns: np.ndarray,
    *,
    cfg: PolicyDefaults,
    state: AbsorptionState,
) -> AbsorptionResult:
    """One absorption recompute on the current panel (REGIME §2.2). MUTATES ``state``.

    Returns ``delta_AR`` (or the frozen last value) plus the diagnostics. The freeze paths
    (ARFLOOR / CONDNUM / ARDISPNORM) hold ``g^{corr}`` at its last valid value via ``state``; this
    function emits the trigger value and flags, and the band layer maps it to ``g^{corr}``.
    """
    x = np.asarray(returns, dtype=float)
    w_ar, n_t = x.shape
    diag: dict[str, float] = {"n_t": float(n_t)}

    # ARFLOOR — rank-deficiency floor: W_AR >= c_AR * n_t.
    if w_ar < cfg.c_ar * n_t:
        diag["ar_window_deficient"] = 1.0
        diag["fresh"] = 0.0
        return AbsorptionResult(state.last_delta_ar, fresh=False, diag=diag)

    # ARCOND — shrink the correlation matrix BEFORE the eigendecomposition.
    shrink = lw_shrink_correlation(x)
    diag["lw_intensity"] = shrink.intensity
    diag["cond_number"] = shrink.cond_number
    diag["eig_sum"] = float(np.sum(shrink.eigenvalues))  # == n_t by TRACE, for the test

    # CONDNUM — post-shrink condition gate; on breach FREEZE g^corr.
    if shrink.cond_number > cfg.cond_max:
        diag["cond_breach_corr"] = 1.0
        diag["fresh"] = 0.0
        return AbsorptionResult(state.last_delta_ar, fresh=False, diag=diag)

    k = top_k(cfg.kappa, n_t)
    ar_t = absorption_ratio(shrink.eigenvalues, n_t, k)
    diag["AR_t"] = ar_t
    diag["eig_topk_share"] = ar_t
    diag["k"] = float(k)

    # ARDISPNORM — append this bar to the persistent cross-day history, then build the fixed-n_t
    # stratum as the CURRENT-BAR-ANCHORED in-band subset (one frac_dn test gates pooling AND freeze).
    # Bound the raw history generously above the long window so multiple recurring sizes coexist.
    state.push(n_t, ar_t, cfg.w_ar_long * 8)
    bucket = state.stratum(n_t, cfg.frac_dn, cfg.w_ar_long)
    diag["stratum_n"] = float(len(bucket))
    # Freeze until the matching band-stratum has enough bars to form both means + sigma_AR.
    if len(bucket) < cfg.w_ar_short + 1 or len(bucket) < 2:
        diag["ar_nt_changed"] = 1.0
        diag["fresh"] = 0.0
        return AbsorptionResult(state.last_delta_ar, fresh=False, diag=diag)

    levels = np.asarray(bucket, dtype=float)
    short = levels[-cfg.w_ar_short :]
    long_ = levels  # the long window is the whole accumulated stratum (capped at w_ar_long)
    ar_bar_short = float(short.mean())
    ar_bar_long = float(long_.mean())
    # DISP — sigma_AR is the std of the RAW single-bar AR LEVEL over the long window, NOT the
    # mean-difference standard error sigma_AR*sqrt(1/W_short + 1/W_long).
    sigma_ar = float(long_.std(ddof=0))
    diag["AR_bar_short"] = ar_bar_short
    diag["AR_bar_long"] = ar_bar_long
    diag["sigma_AR"] = sigma_ar

    # Degenerate-dispersion guard (shared with standardized_shift): a sigma_AR that is only round-off
    # relative to the AR LEVEL is a stationary calm null -> no detectable displacement -> delta_AR = 0.
    # This catches the BLAS-dependent ~1e-16 std-of-identical-levels that otherwise makes the 0/0 ratio
    # land on a spurious +2.0 CRISIS on OpenBLAS while reading harmless on Accelerate.
    if _sigma_ar_degenerate(sigma_ar, ar_bar_long):
        delta_ar = 0.0
    else:
        delta_ar = (ar_bar_short - ar_bar_long) / sigma_ar

    state.last_delta_ar = delta_ar
    diag["delta_AR"] = delta_ar
    diag["fresh"] = 1.0
    return AbsorptionResult(delta_ar, fresh=True, diag=diag)
