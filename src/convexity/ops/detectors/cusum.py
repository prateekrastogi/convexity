"""CUSUM sequential change-detection (Page 1954) + Siegmund (1985) ARL calibration.

``docs/specs/OPS.md`` sec 4.1-4.3. This is the generic primitive REGIME reuses (OPS.md sec 8);
OPS's kill-switch (sec 4) is one caller. Closed-form only — no ML, no fitted threshold (P8): the
ARL calibration is a one-time offline scalar solve / Monte-Carlo under a fixed null, never a
live-fitted model.

Notation reconciliation (OPS.md sec 0.1): the classical CUSUM *reference/allowance value* is ``k``
in Page 1954 / Siegmund 1985 / REGIME.md sec 4; it is named ``d`` throughout this module's API to
avoid a symbol clash with the ``CUSUM`` class. ``d`` and ``k`` are the same quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CusumState:
    """The two one-sided CUSUM statistics after folding in ``n_obs`` standardized innovations.

    ``c_down`` is Page 1954's downward-drift statistic ``C_k`` (the only one that feeds the
    kill-switch action, OPS.md sec 4.3); ``c_up`` is the symmetric upward statistic ``C_k^+`` tracked
    for telemetry (ANOMALOUS_OUTPERFORMANCE, sec 4.3), never a HALT.
    """

    c_down: float
    c_up: float
    n_obs: int


class CUSUM:
    """One-sided (down) + telemetry-side (up) CUSUM on standardized innovations (OPS.md sec 4.3).

    The downward recursion (Page 1954), detecting a NEGATIVE drift in the standardized weekly
    innovation ``z_k`` (the only direction that matters for a kill-switch)::

        C_0 = 0
        C_k = max(0, C_{k-1} - z_k - d)

    and the symmetric upper statistic ``C_k^+ = max(0, C_{k-1}^+ + z_k - d)`` for telemetry.

    ``d`` is the classical reference/allowance value (== REGIME's/SPC's ``k``); the decision boundary
    ``h`` is calibrated externally via :func:`arl_0_siegmund` / :func:`solve_boundary` against a
    stated ARL_0 target. This class holds no threshold itself — it only advances the recursion; the
    RUN/DEGRADE/HALT mapping against ``h*`` lives in the brake state machine (sec 4.4).
    """

    def __init__(self, *, d: float) -> None:
        if d <= 0.0:
            raise ValueError(f"CUSUM reference value d must be > 0, got {d}")
        self.d = float(d)
        self._c_down = 0.0
        self._c_up = 0.0
        self._n = 0

    @property
    def state(self) -> CusumState:
        return CusumState(c_down=self._c_down, c_up=self._c_up, n_obs=self._n)

    def update(self, z_k: float) -> CusumState:
        """Fold one standardized innovation ``z_k`` into both recursions and return the new state."""
        self._c_down = max(0.0, self._c_down - z_k - self.d)
        self._c_up = max(0.0, self._c_up + z_k - self.d)
        self._n += 1
        return self.state

    def reset(self) -> None:
        """Reset both statistics to 0 (used only by the offline ARL Monte-Carlo, never live: the
        live operational state's cool-down is the brake state machine's job, OPS.md sec 4.4 P7)."""
        self._c_down = 0.0
        self._c_up = 0.0
        self._n = 0


def arl_0_siegmund(d: float, h: float) -> float:
    """Siegmund (1985) corrected-diffusion ARL_0 for a one-sided Gaussian CUSUM (OPS.md sec 4.3).

    ``ARL_0(d, h) ~= (exp(2*d*b) - 2*d*b - 1) / (2*d^2)`` with ``b = h + 1.166`` — the expected
    number of observations between false alarms when the standardized innovation really is iid
    N(0,1). The ``+1.166`` is Siegmund's excess-over-boundary (overshoot) constant ``rho`` for
    standardized Gaussian increments; OMITTING it (the raw Wald form ``b = h``) understates ARL_0
    ~3.2x at the shipped ``(d=0.5, h*=6.23)`` — MC-verified: corrected form 3242 vs 100k-run MC
    3201-3220, agreement ~1%. Valid for ``d, h > 0``; a large-ARL_0 asymptotic, appropriate here.
    """
    if d <= 0.0 or h <= 0.0:
        raise ValueError(f"d and h must be > 0, got d={d}, h={h}")
    b = h + 1.166  # Siegmund's Gaussian excess-over-boundary correction
    x = 2.0 * d * b
    return (math.exp(x) - x - 1.0) / (2.0 * d * d)


def solve_boundary(d: float, arl_0_target: float, *, lo: float = 1e-6, hi: float = 100.0,
                   tol: float = 1e-9, max_iter: int = 200) -> float:
    """Solve ``arl_0_siegmund(d, h) == arl_0_target`` for ``h`` by bisection (OPS.md sec 4.3).

    A one-time scalar root-find on the monotone-increasing Siegmund exponential — no ML, no grid
    search (P8). With ``d = 0.5`` and ``arl_0_target = 1000`` the corrected form gives
    ``h_min ~= 5.06`` — the FLOOR consistent with the 1000-week ARL_0 target; the shipped RED
    constant ``h* = 6.23`` sits deliberately ABOVE it (margin), delivering ARL_0 ~= 3242
    (sec 4.3 [DERIVED]; an exact-1000 ARL_0 would put the per-year false-HALT at ~5.1%,
    marginally OVER the <=5%/yr budget — the margin is required, not cosmetic).
    """
    if arl_0_target <= 0.0:
        raise ValueError(f"arl_0_target must be > 0, got {arl_0_target}")
    f_lo = arl_0_siegmund(d, lo) - arl_0_target
    f_hi = arl_0_siegmund(d, hi) - arl_0_target
    if f_lo > 0.0:
        raise ValueError("arl_0_target below the smallest achievable ARL_0 on this bracket")
    if f_hi < 0.0:
        raise ValueError("arl_0_target above the largest achievable ARL_0 on this bracket")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = arl_0_siegmund(d, mid) - arl_0_target
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
