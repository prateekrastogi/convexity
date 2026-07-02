"""Reproducible CUSUM ARL calibration + Monte-Carlo verification (``docs/specs/OPS.md`` sec 11.1).

sec 8 names ``ops/validate/arl.py`` as the home of Invariant CAL's reproducible calibration; sec 11.1
makes the ARL verification the acceptance gate for shipping any change to ``d``, ``h*``, or
``delta_target``. This module is the SINGLE source of that calibration: it (a) reproduces the shipped
boundary ``h* = 6.23`` from the Siegmund closed form (sec 4.3), and (b) Monte-Carlo simulates the
``N(0,1)`` null and a ``delta_target`` downward-shifted alternative through the calibrated ``(d, h*)``
recursion to report empirical ``ARL_0``/``ARL_1`` against the design targets (Inv CAL).

Offline / one-time (P8): the numeric root-find and the Monte-Carlo are a calibration run, never a
live-fitted model — the boundary ships as a RED constant (``OpsDefaults.h_star``), re-derived here for
audit, not re-solved on the hot path. numpy for the MC; the closed-form solve is scipy-free
(bisection on the monotone Siegmund exponential, :mod:`convexity.ops.detectors.cusum`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.ops.config.defaults import PERIODS_PER_YEAR
from convexity.ops.detectors.cusum import CUSUM, arl_0_siegmund, solve_boundary


@dataclass(frozen=True)
class ArlReport:
    """The reproducible-calibration report for one ``(d, h*, delta_target)`` triple (OPS.md sec 11.1).

    ``h_star_solved`` is the boundary the Siegmund solve produces for ``(d, arl_0_target)`` — the
    number the shipped RED constant must match (sec 12 kill-switch gate). ``arl_0_closed_form`` is the
    Siegmund ARL_0 at the shipped ``(d, h_star)``. ``arl_0_mc``/``arl_1_mc`` are the Monte-Carlo
    empirical average run lengths under the null and the injected shift. ``per_year_false_halt`` is
    the annualized false-HALT probability implied by ``arl_0_mc`` (the <=5%/yr budget, sec 4.3).
    """

    d: float
    h_star_solved: float
    arl_0_target: float
    arl_0_closed_form: float
    arl_0_mc: float
    arl_1_mc: float
    per_year_false_halt: float
    delta_target: float
    n_years_null: int
    n_reps_alt: int


def solve_h_star(*, delta_target: float, arl_0_target: float) -> float:
    """Reproduce the shipped boundary ``h*`` from the Siegmund ARL_0 solve (OPS.md sec 4.3 [DERIVED]).

    ``d = delta_target / 2`` (classical reference/allowance value); solve ``ARL_0(d, h) = arl_0_target``
    for ``h`` by bisection. With ``delta_target = 1.0`` and ``arl_0_target = 1000`` this reproduces
    ``h* ~= 6.23``, the RED constant this pillar ships. No ML, no grid search — a one-time scalar
    root-find on the monotone Siegmund exponential (P8).
    """
    d = delta_target / 2.0
    return solve_boundary(d, arl_0_target)


def mc_arl_0(*, d: float, h_star: float, n_years: int, seed: int = 0) -> float:
    """Monte-Carlo empirical ARL_0: mean weeks between false alarms under the iid ``N(0,1)`` null.

    Simulates ``n_years * PERIODS_PER_YEAR`` weeks of ``z_k ~ N(0,1)`` through the calibrated
    downward CUSUM, resetting after each detected alarm (the standard offline ARL restart convention).
    Returns ``total_weeks / max(alarms, 1)`` (OPS.md sec 11.1 / Inv CAL).
    """
    rng = np.random.default_rng(seed)
    total_weeks = int(n_years * PERIODS_PER_YEAR)
    z = rng.standard_normal(total_weeks)
    cusum = CUSUM(d=d)
    alarms = 0
    for zk in z:
        st = cusum.update(float(zk))
        if st.c_down >= h_star:
            alarms += 1
            cusum.reset()
    return total_weeks / max(alarms, 1)


def mc_arl_1(*, d: float, h_star: float, delta_target: float, n_reps: int, seed: int = 1,
             max_weeks: int = 5000) -> float:
    """Monte-Carlo empirical ARL_1: mean weeks to detect a persistent ``-delta_target`` shift.

    Each rep injects a persistent downward shift of ``delta_target`` in the mean of ``z_k`` from week
    1 and records the first passage of the downward CUSUM above ``h_star`` (Inv CAL detection side).
    Returns the mean detection time; by construction ``ARL_1 << ARL_0`` (OPS.md sec 4.3).
    """
    rng = np.random.default_rng(seed)
    cusum = CUSUM(d=d)
    detect_times: list[int] = []
    for _ in range(n_reps):
        cusum.reset()
        t = 0
        while True:
            t += 1
            zk = float(rng.standard_normal()) - delta_target
            st = cusum.update(zk)
            if st.c_down >= h_star or t > max_weeks:
                break
        detect_times.append(t)
    return float(np.mean(detect_times))


def calibrate_and_verify(
    *,
    delta_target: float,
    arl_0_target: float,
    h_star: float,
    n_years_null: int = 4000,
    n_reps_alt: int = 2000,
    seed: int = 0,
) -> ArlReport:
    """Full Inv CAL acceptance run: reproduce ``h*`` and Monte-Carlo-verify ARL_0/ARL_1 (sec 11.1).

    Reproduces the shipped ``h_star`` from the Siegmund solve, reports the closed-form ARL_0 at the
    shipped boundary, and Monte-Carlo estimates ARL_0 (null) and ARL_1 (injected ``delta_target``
    shift). The caller (the CAL test / an offline acceptance report) asserts: (i) ``h_star_solved``
    rounds to the shipped ``h_star``; (ii) ``per_year_false_halt < 0.05`` (the <=5%/yr budget); and
    (iii) ``arl_1_mc << arl_0_mc`` (fast detection). This is the acceptance gate for changing any of
    ``d``/``h*``/``delta_target`` (sec 11.1).
    """
    d = delta_target / 2.0
    h_star_solved = solve_boundary(d, arl_0_target)
    arl_0_cf = arl_0_siegmund(d, h_star)
    arl_0_mc_val = mc_arl_0(d=d, h_star=h_star, n_years=n_years_null, seed=seed)
    arl_1_mc_val = mc_arl_1(d=d, h_star=h_star, delta_target=delta_target, n_reps=n_reps_alt,
                            seed=seed + 1)
    per_year = 1.0 - (1.0 - 1.0 / arl_0_mc_val) ** PERIODS_PER_YEAR
    return ArlReport(
        d=d,
        h_star_solved=h_star_solved,
        arl_0_target=arl_0_target,
        arl_0_closed_form=arl_0_cf,
        arl_0_mc=arl_0_mc_val,
        arl_1_mc=arl_1_mc_val,
        per_year_false_halt=per_year,
        delta_target=delta_target,
        n_years_null=n_years_null,
        n_reps_alt=n_reps_alt,
    )
