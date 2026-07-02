"""CUSUM change-detection with ARL calibration — the core REGIME detector (REGIME.md §3).

Closed-form, no-ML (REGIME.md §0 P3). This module implements:

  * The one-sided upper CUSUM recursion ``S_t = max(0, S_{t-1} + (x_t - mu_0 - k))`` and its
    downward-shift companion (§3.1, Page 1954), plus the ``S_t >= h`` alarm-and-reset rule.
  * The Siegmund (1985) continuity-corrected Wald ARL closed form (§3.2), used ONLY to INITIALIZE
    the ``(k, h)`` search — with the §3.2 correction that the in-control ARL is the ``theta -> 0``
    LIMIT ``b^2`` taken at ``mu_0 + k`` (the zero-drift reference), NOT a substitution of
    ``theta = -k/sigma`` (which answers a different question and gives ~3.9e6, not ~104).
  * The reference value ``k = Delta_target * sigma / 2`` (§3.4) and the ``h`` solved from
    ``ARL_0_target`` via ``h = sigma * (sqrt(ARL_0) - 1.166)`` (§3.5 Step 2).
  * The MANDATORY empirical block-bootstrap ARL_0 check (§3.5 Step 5 / Invariant ARL): the delivered
    ARL_0 is EMPIRICALLY verified on the frozen in-control window, never closed-form asserted.

Two distinct claims are kept separate (§3.1): (i) Page's (1954) construction IS the SPRT restarted at
its running minimum with the ``S_t >= h`` stopping rule; (ii) the ASYMPTOTIC MINIMAX OPTIMALITY of
that construction is Lorden's (1971) separate, later theorem — not attributed to Page here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Siegmund's (1985) continuity correction for the discreteness of the reflecting boundary at 0
# (REGIME.md §3.2): b := h/sigma + 1.166. This constant is the whole reason the worked example's
# h/sigma = 9.032 maps to b = 10.198 and b^2 = 103.99 ~ 104 (§3.5).
SIEGMUND_CORRECTION: float = 1.166


def cusum_upper(x: np.ndarray, *, mu_0: float, k: float) -> np.ndarray:
    """One-sided UPPER CUSUM for an upward mean shift (REGIME.md §3.1, Page 1954).

    ``S_0 = 0``; ``S_t = max(0, S_{t-1} + (x_t - mu_0 - k))``. Detects dispersion/vol SPIKING (the
    STRESSED direction). Returns the full ``S_t`` path (same length as ``x``); the caller applies the
    ``S_t >= h`` alarm-and-reset rule (:func:`run_length_to_alarm` / the state machine). This is the
    SPRT restarted at its running minimum (§3.1 derivation), up to the positive scale 2k/sigma^2 that
    does not affect the crossing time."""
    x = np.asarray(x, dtype=float)
    s = np.zeros(x.size)
    prev = 0.0
    for t in range(x.size):
        prev = max(0.0, prev + (x[t] - mu_0 - k))
        s[t] = prev
    return s


def cusum_lower(x: np.ndarray, *, mu_0: float, k: float) -> np.ndarray:
    """One-sided LOWER CUSUM for a downward mean shift (REGIME.md §3.3).

    ``S_t^- = max(0, S_{t-1}^- + (mu_0 - k - x_t))``. Detects dispersion COLLAPSE (momentum
    crowding/unwind — §2.2, §8.2), the risk a spike-only detector would completely miss. Mirror image
    of :func:`cusum_upper`; the two are mechanically ANTI-correlated through their shared input (§3.3)
    which is WHY HALT keys on cross-statistic co-occurrence (D and V), not S_D^+ with S_D^-."""
    x = np.asarray(x, dtype=float)
    s = np.zeros(x.size)
    prev = 0.0
    for t in range(x.size):
        prev = max(0.0, prev + (mu_0 - k - x[t]))
        s[t] = prev
    return s


def reference_value_k(delta_target: float, sigma: float) -> float:
    """Reference/slack ``k = Delta_target * sigma / 2`` — the standard half-shift convention (§3.4).

    ``Delta_target`` is the STANDARDIZED shift REGIME commits to detecting (e.g. 1.5-sigma, §3.5).
    Choosing ``k`` as half the target shift places the CUSUM's own zero-drift reference ``mu_0 + k``
    close to ``mu_0``, which is exactly what makes the ``b^2`` in-control approximation valid (§3.2)."""
    return delta_target * sigma / 2.0


def arl_wald(theta: float, h: float, sigma: float) -> float:
    """Siegmund (1985) continuity-corrected Wald ARL approximation (REGIME.md §3.2).

    ``theta`` is the STANDARDIZED DRIFT of the random walk S_t under the true mean (``theta = Delta -
    k/sigma`` where ``Delta = (mu - mu_0)/sigma``) — the reference-value drag ``-k/sigma`` stays INSIDE
    theta (§3.2). ``b := h/sigma + 1.166``. For ``theta != 0``:

        ARL(mu) ~ ( exp(-2*theta*b) - 1 + 2*theta*b ) / ( 2*theta^2 )

    At ``theta = 0`` (the zero-drift point ``mu = mu_0 + k``) this has a removable singularity whose
    limit is ``b^2`` (§3.2 L'Hopital / second-order Taylor: exp(-2tb)-1+2tb = 2t^2 b^2 - (4/3)t^3 b^3 +
    ...). We return that limit directly for tiny ``|theta|`` to avoid the 0/0. Used ONLY to initialize
    the (k,h) search (§3.2/§3.5) — the delivered ARL_0 is the EMPIRICAL one (Invariant ARL)."""
    b = h / sigma + SIEGMUND_CORRECTION
    if abs(theta) < 1e-9:
        return b * b
    return (np.exp(-2.0 * theta * b) - 1.0 + 2.0 * theta * b) / (2.0 * theta * theta)


def arl_in_control(h: float, sigma: float) -> float:
    """The delivered CLOSED-FORM in-control ARL: the ``theta -> 0`` limit ``b^2`` at ``mu_0 + k`` (§3.2).

    This is ``ARL(mu_0 + k) = (h/sigma + 1.166)^2`` — the conventional in-control ARL (Montgomery;
    Hawkins & Olwell 1998 §2.3), the analytically tractable ZERO-DRIFT point, valid because
    ``mu_0 + k ~ mu_0`` for the half-shift ``k`` (§3.4). It is NOT ``arl_wald(theta=-k/sigma, ...)``,
    which evaluates at the ACTUAL in-control mean ``mu_0`` (large negative drift) and gives ~3.9e6 — a
    different question (§3.2). This closed form seeds ``h`` only; Invariant ARL requires the empirical
    check before ship."""
    return arl_in_control_from_b(h / sigma + SIEGMUND_CORRECTION)


def arl_in_control_from_b(b: float) -> float:
    """``b^2`` — the in-control ARL as a function of the corrected standardized interval ``b`` (§3.2)."""
    return b * b


def solve_h_from_arl0(arl_0_target: float, sigma: float) -> float:
    """Invert the in-control closed form to seed ``h`` from a target ARL_0 (REGIME.md §3.5 Step 2).

    ``ARL_0 ~ (h/sigma + 1.166)^2  =>  h = sigma * (sqrt(ARL_0) - 1.166)``. Worked example (§3.5):
    ``ARL_0=104, sigma=0.0060 -> h = 0.0060 * (10.198 - 1.166) = 0.0060 * 9.032 = 0.05419``. This is
    only the SEED for the (k,h) search; the FINAL h is the empirically-validated one (§3.5 Step 5).

    Raises ``ValueError`` when ``sigma <= 0``: a degenerate in-control window with no variance cannot
    yield a meaningful threshold (conservative-under-uncertainty, §0 P6 — a silent 0 threshold would
    make the detector fire on every step, the opposite of a calibrated ARL_0)."""
    if sigma <= 0.0:
        raise ValueError(
            "sigma must be > 0 to solve h from ARL_0 (REGIME.md §3.5); a zero-variance in-control "
            "window is a degenerate calibration (§0 P6 conservative-under-uncertainty)."
        )
    return sigma * (np.sqrt(arl_0_target) - SIEGMUND_CORRECTION)


def arl_out_of_control(h: float, sigma: float, k: float) -> float:
    """Implied out-of-control ARL at the target shift ``mu_0 + 2k`` (REGIME.md §3.5 Step 4).

    At the target shift ``mu = mu_0 + 2k`` the standardized drift is ``theta = 2k/sigma - k/sigma =
    k/sigma = Delta_target/2`` (positive; §3.5 Step 4: for Delta_target=1.5, sigma=0.0060, k=0.0045
    this is theta = 0.75). The ``Delta_target`` dependence is ALREADY carried by ``k`` (which was
    derived as ``Delta_target*sigma/2``), so ``theta = k/sigma`` is the whole story — the raw
    ``Delta_target`` is not read again here. Worked example gives ARL_1 ~ 12.71 weeks vs ARL_0 ~ 104
    — the >8x CUSUM speed-up (§3.5)."""
    theta = k / sigma
    return arl_wald(theta, h, sigma)


def run_length_to_alarm(s_path: np.ndarray, h: float) -> int:
    """Number of steps until the CUSUM path first crosses ``h`` (REGIME.md §3.1 alarm rule).

    Returns the 1-based index of the first ``S_t >= h``, or ``len(s_path)`` (right-censored) if it
    never crosses within the path. Used by the empirical ARL check (:func:`empirical_arl0`) to average
    run-lengths to first crossing across bootstrap resamples (§3.5 Step 5)."""
    crossings = np.flatnonzero(s_path >= h)
    return int(crossings[0]) + 1 if crossings.size else int(s_path.size)


@dataclass(frozen=True)
class ARLCalibration:
    """A calibrated ``(mu_0, sigma, k, h)`` quadruple plus its empirical ARL_0 (REGIME.md §3.4/§3.5).

    ``mu_0``/``sigma`` are TEAL (fit on the frozen in-control window, §3.4); ``k``/``h`` are RED-derived
    (§3.4). ``arl0_closed_form`` is the ``b^2`` seed (§3.2); ``arl0_empirical`` is the DELIVERED,
    block-bootstrap-verified number (Invariant ARL) — a calibration whose ``arl0_empirical`` is None
    has NOT passed the mandatory §3.5 Step 5 check.

    Invariant ARL has TWO clauses and this record enforces BOTH via :attr:`ships`: (i) the delivered
    ARL_0 is the EMPIRICAL block-bootstrap value (``arl0_empirical is not None``), and (ii) it lands
    within the RED tolerance band of ``ARL_0_target`` — i.e.
    ``|arl0_empirical - arl0_target| <= arl_tol_frac * arl0_target`` (§3.5 Step 5:
    "matches the target ARL_0_target within a stated tolerance"). ``arl0_target``/``arl_tol_frac`` are
    the RED targets the calibration was gated against; ``within_tolerance`` caches clause (ii). A
    ``(k, h)`` with ``ships is False`` must NOT go live (Invariant ARL: "No calibrated (k,h) ships
    without ... landing within the RED tolerance band")."""

    mu_0: float
    sigma: float
    k: float
    h: float
    arl0_closed_form: float
    arl0_empirical: float | None = None
    arl1: float | None = None
    arl0_target: float | None = None
    arl_tol_frac: float | None = None

    @property
    def within_tolerance(self) -> bool:
        """Clause (ii) of Invariant ARL: empirical ARL_0 inside ``arl0_target +/- arl_tol_frac`` (§3.5).

        False when the empirical estimate, the target, or the tolerance is missing (an ungated
        calibration is conservatively treated as NOT within tolerance — §0 P6)."""
        if self.arl0_empirical is None or self.arl0_target is None or self.arl_tol_frac is None:
            return False
        return abs(self.arl0_empirical - self.arl0_target) <= self.arl_tol_frac * self.arl0_target

    @property
    def ships(self) -> bool:
        """Invariant ARL gate: True iff BOTH clauses hold (empirical estimate present AND in the RED
        tolerance band, §3.5 Step 5). A calibration with ``ships is False`` must not go live."""
        return self.arl0_empirical is not None and self.within_tolerance


def empirical_arl0(
    in_control_series: np.ndarray,
    *,
    mu_0: float,
    sigma: float,
    k: float,
    h: float,
    n_boot: int,
    block_length: int,
    seed: int,
) -> float:
    """Empirical in-control ARL_0 by moving-block bootstrap (REGIME.md §3.5 Step 5 / Invariant ARL).

    Because ``D_t``/``V_t`` are autocorrelated (dispersion clusters) and not exactly Gaussian, the
    closed form is only a starting point (§3.5). This resamples overlapping blocks of length
    ``block_length`` from the frozen in-control training series (Kunsch 1989; block length per Politis
    & White 2004, §3.5) — the SAME moving-block discipline ``convexity.system.validate.robustness``
    uses — runs the upper CUSUM on each resampled path, records the run-length to first ``S_t >= h``
    crossing, and averages. Right-censored paths (no crossing) contribute the full path length, so the
    estimate is a conservative LOWER bound on the true ARL_0 (Invariant ARL: the reported number is the
    EMPIRICAL, not the closed-form, value). Deterministic given ``seed``."""
    r = np.asarray(in_control_series, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return float("nan")
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    # Resample paths at least as long as the target ARL so the run-length is not truncated by a short
    # simulated stream (§3.5: "simulating many resampled D_t paths ... averaging the observed run
    # length to first crossing"). Path length = a few multiples of the closed-form ARL_0 seed.
    path_len = max(n, int(4 * arl_in_control_from_b(h / sigma + SIEGMUND_CORRECTION)))
    n_blocks = -(-path_len // block_length)  # ceil
    run_lengths = np.empty(n_boot)
    for b_i in range(n_boot):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        path = np.concatenate([r[s : s + block_length] for s in starts])[:path_len]
        s_path = cusum_upper(path, mu_0=mu_0, k=k)
        run_lengths[b_i] = run_length_to_alarm(s_path, h)
    return float(np.mean(run_lengths))


def calibrate(
    in_control_series: np.ndarray,
    *,
    delta_target: float,
    arl_0_target: float,
    n_boot: int,
    block_length: int,
    seed: int,
    arl_tol_frac: float | None = None,
    require_ships: bool = False,
    fit_moments: bool = True,
    mu_0: float | None = None,
    sigma: float | None = None,
) -> ARLCalibration:
    """Full calibration of one statistic's CUSUM (REGIME.md §3.4/§3.5, Steps 1-5).

    Fits ``mu_0``/``sigma`` on the frozen in-control training series (TEAL, §3.4) unless supplied,
    derives ``k = Delta_target*sigma/2`` (Step 1) and the seed ``h`` from ``ARL_0_target`` (Step 2),
    then runs the MANDATORY empirical block-bootstrap ARL_0 check (Step 5 / Invariant ARL) and records
    both the closed-form seed and the delivered empirical value. Does NOT re-estimate ``(mu_0, k, h)``
    online — calibration is a discrete, frozen, offline event (§0 P4).

    Invariant ARL tolerance gate: when ``arl_tol_frac`` is supplied, the returned :class:`ARLCalibration`
    carries the RED target/tolerance so ``.within_tolerance``/``.ships`` decide whether this ``(k, h)``
    may go live — the empirical ARL_0 must land within ``arl_0_target +/- arl_tol_frac*arl_0_target``
    (§3.5 Step 5: "matches the target ARL_0_target within a stated tolerance"). With
    ``require_ships=True`` a calibration that FAILS the tolerance gate raises rather than returning a
    non-shippable artifact — the strict form for a live-deploy path (Invariant ARL: "No calibrated
    (k,h) ships without ... landing within the RED tolerance band"). The (k,h) grid SEARCH that would
    move ``h`` until the empirical estimate lands in band lives in REGIME's ``validate/`` (§3.5, §7 —
    out of scope for the detector core); this gate is the shippability CHECK that search must satisfy."""
    r = np.asarray(in_control_series, dtype=float)
    finite = r[np.isfinite(r)]
    if fit_moments:
        mu_0 = float(np.mean(finite))
        sigma = float(np.std(finite, ddof=1))
    if mu_0 is None or sigma is None:
        raise ValueError("mu_0/sigma must be supplied when fit_moments=False (REGIME.md §3.4).")
    if sigma <= 0.0:
        raise ValueError(
            "in-control sigma must be > 0 to calibrate (REGIME.md §3.4/§3.5); a zero-variance "
            "training window is degenerate (§0 P6)."
        )
    k = reference_value_k(delta_target, sigma)
    h = solve_h_from_arl0(arl_0_target, sigma)
    arl0_cf = arl_in_control(h, sigma)
    arl1 = arl_out_of_control(h, sigma, k)
    arl0_emp = empirical_arl0(
        finite, mu_0=mu_0, sigma=sigma, k=k, h=h,
        n_boot=n_boot, block_length=block_length, seed=seed,
    )
    cal = ARLCalibration(
        mu_0=mu_0, sigma=sigma, k=k, h=h,
        arl0_closed_form=arl0_cf, arl0_empirical=arl0_emp, arl1=arl1,
        arl0_target=arl_0_target, arl_tol_frac=arl_tol_frac,
    )
    if require_ships and not cal.ships:
        raise ValueError(
            "Invariant ARL (REGIME.md §3.5 Step 5): calibrated (k, h) does not ship — empirical "
            f"ARL_0={arl0_emp} is outside the RED tolerance band {arl_0_target}"
            f" +/- {arl_tol_frac}*{arl_0_target}. No (k, h) goes live without landing in band; run "
            "the validate/ (k,h) grid search to find an h whose empirical ARL_0 matches target."
        )
    return cal
