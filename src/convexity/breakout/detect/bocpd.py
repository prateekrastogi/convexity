"""BocpdChannel — Bayesian online changepoint detection evidence (BREAKOUT §7 detect/bocpd).

Adams & MacKay (2007) BOCPD with a constant hazard and a Gaussian observation model with a
Normal-inverse-Gamma conjugate prior (unknown mean and variance), so the predictive is Student-t.
Pruning bounds the run-length posterior to ``max_run_length`` (BREAKOUT §detect/bocpd). Hazard is
RED/BLUE, never a per-symbol hand parameter. Evidence at t uses only ``x[:t+1]`` (no future bars,
Inv B).

Returns ``(changepoint_evidence, map_run_length)``: the changepoint evidence is the posterior mass
on a SHORT run length (a recent changepoint), squashed to [0, 1]; the MAP run length is the most
probable run length at each bar.
"""

from __future__ import annotations

import numpy as np

from convexity.breakout.types import StdSeries


def _student_t_pdf(x: np.ndarray, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray,
                   beta: np.ndarray) -> np.ndarray:
    """Posterior-predictive Student-t pdf for a Normal-inverse-Gamma prior (vectorised over runs)."""
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / (nu * scale2)
    from math import lgamma
    # gamma terms depend only on nu (scalar-per-run); compute via vectorised exp(lgamma).
    lg = np.vectorize(lgamma)
    log_c = lg((nu + 1.0) / 2.0) - lg(nu / 2.0) - 0.5 * np.log(np.pi * nu * scale2)
    return np.exp(log_c) * (1.0 + z) ** (-(nu + 1.0) / 2.0)


class BocpdChannel:
    """BOCPD changepoint evidence (BREAKOUT §7). One public method, ``run_length_evidence``."""

    # Normal-inverse-Gamma prior hyperparameters (RED-ish modelling choice; weak, data-dominated).
    MU0 = 0.0
    KAPPA0 = 1.0
    ALPHA0 = 1.0
    BETA0 = 1.0

    def run_length_evidence(
        self, s: StdSeries, hazard: float, max_run_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(changepoint_evidence, map_run_length)`` per bar (uses x[:t+1], pruned)."""
        x = s.x
        n = x.size
        evidence = np.zeros(n)
        map_rl = np.zeros(n, dtype=int)

        # run-length posterior R and per-run NIG sufficient stats.
        R = np.array([1.0])
        mu = np.array([self.MU0])
        kappa = np.array([self.KAPPA0])
        alpha = np.array([self.ALPHA0])
        beta = np.array([self.BETA0])

        for t in range(n):
            xt = float(x[t]) if np.isfinite(x[t]) else 0.0
            pred = _student_t_pdf(np.full_like(mu, xt), mu, kappa, alpha, beta)

            growth = R * pred * (1.0 - hazard)
            cp = float(np.sum(R * pred * hazard))
            newR = np.concatenate([[cp], growth])
            total = newR.sum()
            if total <= 0:
                newR = np.zeros_like(newR)
                newR[0] = 1.0
            else:
                newR /= total

            # NIG conjugate update for the grown runs; prepend the fresh-run prior.
            mu_g = (kappa * mu + xt) / (kappa + 1.0)
            kappa_g = kappa + 1.0
            alpha_g = alpha + 0.5
            beta_g = beta + 0.5 * kappa * (xt - mu) ** 2 / (kappa + 1.0)
            mu = np.concatenate([[self.MU0], mu_g])
            kappa = np.concatenate([[self.KAPPA0], kappa_g])
            alpha = np.concatenate([[self.ALPHA0], alpha_g])
            beta = np.concatenate([[self.BETA0], beta_g])

            if newR.size > max_run_length + 1:
                newR = newR[: max_run_length + 1]
                mu = mu[: max_run_length + 1]
                kappa = kappa[: max_run_length + 1]
                alpha = alpha[: max_run_length + 1]
                beta = beta[: max_run_length + 1]
                newR /= newR.sum() + 1e-300
            R = newR

            short = min(5, R.size)
            evidence[t] = float(np.sum(R[:short]))
            map_rl[t] = int(np.argmax(R))
        return np.clip(evidence, 0.0, 1.0), map_rl
