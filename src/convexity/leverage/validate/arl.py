"""CUSUM average-run-length Monte-Carlo verification (LEVERAGE.md §6.3, §9, §10).

Verifies the one-sided-Gaussian-CUSUM in-control ARL against the Hawkins & Olwell (1998) closed
form ``ARL_0 ≈ exp(2·k_ref·h_promote)/(2·k_ref²)`` (§6.3). Runs ONLY on synthetic ``N(0,1)``
surprise sequences (the stated in-control model) — NEVER on realized live P&L, which would violate
P5/P7 (§9, §10). This is a ``validate/`` module: no live layer imports it (Invariant V firewall).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.leverage.ratchet.cusum import arl0_closed_form, cusum_step


@dataclass(frozen=True)
class ArlEstimate:
    """OUTPUT of :func:`monte_carlo_arl` (LEVERAGE.md §6.3).

    ``arl_mc`` is the empirical mean run length to a false crossing under the in-control ``N(0,1)``
    model; ``arl_closed_form`` the Hawkins & Olwell (1998) target; ``n_runs`` the MC sample count.
    """

    arl_mc: float
    arl_closed_form: float
    n_runs: int


def monte_carlo_arl(
    k_ref: float,
    h_promote: float,
    *,
    n_runs: int = 2_000,
    max_weeks: int = 5_000,
    seed: int = 0,
) -> ArlEstimate:
    """Empirical in-control ARL of the one-sided CUSUM on synthetic ``N(0,1)`` surprises (§6.3).

    For each run, draw i.i.d. ``z_k ~ N(0,1)`` (the stated in-control model, §6.3) and record the
    week the CUSUM first crosses ``h_promote`` (a false promotion); average over ``n_runs``. Runs
    that never cross within ``max_weeks`` are censored at ``max_weeks`` (a conservative lower bound
    on the true ARL). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    run_lengths = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        s = 0.0
        crossed_at = max_weeks
        z = rng.standard_normal(max_weeks)
        for k in range(max_weeks):
            s = cusum_step(s, float(z[k]), k_ref)
            if s >= h_promote:
                crossed_at = k + 1
                break
        run_lengths[i] = crossed_at
    return ArlEstimate(
        arl_mc=float(np.mean(run_lengths)),
        arl_closed_form=arl0_closed_form(k_ref, h_promote),
        n_runs=n_runs,
    )
