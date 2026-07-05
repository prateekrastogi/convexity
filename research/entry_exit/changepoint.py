"""MODELS.changepoint -- Bayesian Online Change-Point Detection (Adams-MacKay 2007), Phase-C ENTRY #1.

The change-point / structural-break ENTRY candidate frozen in the Phase-C pre-registration (docs/EDGE.md
"The pre-registered candidate set (frozen)" #1, SIGNAL.md §C.2(a), SYSTEM.md §C.9 #1). It conforms to the
``intraday.models.base.EntryModel`` protocol: ``fit(train) -> self`` (parameters estimated on the TRAIN
fold, then FROZEN) and a causal one-step-ahead ``signal_at(cursor) -> Side`` in the ``{+1, 0, -1}`` sign
convention of ``signal/orb.py`` (SIGNAL §3), feeding the *unchanged* §4 strike selection.

GENERATING PROCESS (the assumption the synthetic recovery test controls -- SIGNAL §C.2(a)).
The per-bar log-return ``x_t`` is drawn from a segment with a piecewise-constant **drift** ``mu`` and
i.i.d. Gaussian innovations ``eps_t ~ N(0, sigma^2)``; segment durations follow a **constant hazard**
``H(r) = 1/lambda`` (a geometric run-length prior, expected regime length ``lambda``). Within a segment
the unknown ``(mu, sigma^2)`` carry a conjugate **Normal-inverse-Gamma** prior ``(mu0, kappa0, alpha0,
beta0)``, so the per-run-length predictive is a Student-t (the standard Adams-MacKay Gaussian UPM with
both mean and variance unknown). This is the *drift* change-point of SIGNAL §C.2(a), never the
retrospective MAP segmentation (that is the Bai-Perron / e-divisive DEFERRAL, §C.5).

THE FILTERED RUN-LENGTH POSTERIOR (Adams-MacKay 2007, causal by construction).
``run_length_posterior(x_{1:n})`` returns ``P(r_n | x_{1:n})`` -- the *filtered* run-length distribution
after the exact online recursion (growth message ``* (1 - H)`` + change-point message ``* H``, summed and
renormalized each step). It conditions ONLY on ``x_{1:n}`` (no future bar), so it is a filtered estimate,
never a smoothed / full-sample one (SIGNAL §C.1, the load-bearing purity clause). A change-point makes
the mass collapse toward ``r = 0`` (the recovery test asserts this near a known truth).

PURITY -- the pre-registered ONE-BAR OFFSET (binding, closes the same-bar look-ahead hole).
The BOCPD posterior available at the close of bar ``t`` (``P(r_t | F_t)``) conditions on bar ``t``; a
decision that reads it and transacts on bar ``t`` is a one-bar look-ahead (docs/EDGE.md #1, SYSTEM §C.2).
Phase-C fixes the decision->execution offset explicitly: an entry EXECUTED on ``cursor.t`` may only read a
posterior conditioned on information up to ``t-1``. ``signal_at`` therefore runs the recursion over
``cursor.past() == series[:t]`` (indices ``0 .. t-1``) ONLY -- the last datum consumed is ``x_{t-1}``, so
the posterior it reads is ``P(r_{t-1} | F_{t-1})`` and the entry at bar ``t`` is ``F_{t-1}``-measurable
(equivalently: the signal entering at bar ``s`` is ``P(r_{s-1} | F_{s-1})``, never ``P(r_s | F_s)``). The
final observation is fetched through ``cursor.read(t - 1)``, so a one-line mutation to ``read(t)`` trips
``require_causal`` and the pinning test FAILS -- the no-lookahead property is enforced, not asserted
(SIGNAL §C.1, SYSTEM §C.2). Cadence is ``DAILY``: any read of index ``>= t`` (same bar included) is a leak.

TRAIN-FIT, FROZEN (SIGNAL §C.1, SYSTEM §C.9 #1 grid).
``fit`` estimates, on the TRAIN fold ONLY, the segment innovation scale (``sigma`` -> the NIG prior
strength) and the prior drift mean from the fold, then FREEZES every parameter for OOS -- no re-fit / no
threshold re-tune on TEST. The pre-registered per-candidate grid (charged to SYSTEM ``N_eff`` §C.3) is the
hazard ``lambda``, the NIG prior ``(mu0, kappa0, alpha0, beta0)``, and the run-length-posterior-mass entry
threshold ``p_break`` -- all constructor arguments, set a priori, never P&L-tuned. ``k`` counts them for
the SYSTEM §C.4 parsimony penalty.

Imports ONLY ``intraday.models.base`` (the shared interface + causal helpers) and numpy + stdlib ``math``
-- NEVER a sibling Phase-C model module (SYSTEM §C.3 interface-driven-harness / off-ledger rule). Touches
no lake, emits no metric, adopts nothing (the empirical RUN is BLOCKED-ON-LAKE, SYSTEM §C.8); this module
is the candidate CODE, synthetic-tested by a known-generating-process recovery + a no-lookahead pinning
test (tests/models/test_changepoint.py). Moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import Cadence, ScanCursor, Side, TrainFold, warmup_ok

__all__ = ["BOCPDEntry", "run_length_posterior"]


def _student_t_logpdf(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
    """Log posterior-predictive of the Normal-inverse-Gamma UPM: Student-t (Adams-MacKay 2007, §2).

    A single observation ``x`` under the NIG posterior ``(mu, kappa, alpha, beta)`` has predictive
    ``t_{2 alpha}`` with location ``mu`` and scale ``sqrt(beta (kappa + 1) / (alpha kappa))`` -- the
    standard conjugate Gaussian-unknown-mean-and-variance predictive. Returned in LOG space so the
    recursion multiplies probabilities as log-sum stable additions; ``nu = 2 alpha`` degrees of freedom.
    """
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / (nu * scale2)
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - (nu + 1.0) / 2.0 * math.log1p(z)
    )


def run_length_posterior(
    x: np.ndarray,
    *,
    hazard: float,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> tuple[np.ndarray, float]:
    """The FILTERED Adams-MacKay run-length posterior ``P(r_n | x_{1:n})`` + the MAP-run filtered drift.

    Runs the exact online BOCPD recursion (Adams-MacKay 2007) over the observation vector ``x`` with a
    constant hazard ``H = hazard`` and a Normal-inverse-Gamma prior, tracking the per-run-length sufficient
    statistics conjugately. Returns ``(posterior, map_drift)`` where ``posterior[r]`` is
    ``P(r_n = r | x_{1:n})`` (length ``len(x) + 1``, index 0 = "a change-point just occurred") and
    ``map_drift`` is the posterior-mean drift ``E[mu | r*]`` of the MAP run-length ``r* = argmax posterior``
    -- the "post-break filtered drift" whose SIGN the entry uses (SIGNAL §C.2(a)).

    CAUSAL BY CONSTRUCTION: the message passing conditions only on ``x_{1:n}``; no future observation
    enters. An empty ``x`` returns the degenerate prior (all mass on ``r = 0``) and the prior drift ``mu0``.
    """
    n = len(x)
    if n == 0:
        return np.array([1.0]), mu0

    # NIG sufficient-statistic vectors, one entry per live run length (index r = run length).
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    r_post = np.array([1.0])  # P(r_0 = 0) = 1 before any datum

    for t in range(n):
        xt = float(x[t])
        # Predictive prob of x_t under each live run length's NIG posterior (Student-t UPM).
        pred = np.array([
            math.exp(_student_t_logpdf(xt, mu[i], kappa[i], alpha[i], beta[i]))
            for i in range(len(mu))
        ])
        growth = r_post * pred * (1.0 - hazard)          # run continues: r -> r+1
        cp = float(np.sum(r_post * pred * hazard))       # a change-point: mass into r = 0
        new_post = np.empty(len(growth) + 1)
        new_post[0] = cp
        new_post[1:] = growth
        s = new_post.sum()
        r_post = new_post / s if s > 0 else new_post

        # Conjugate NIG update of each continuing run + prepend the fresh r=0 prior (Adams-MacKay §2).
        new_mu = np.empty(len(mu) + 1)
        new_kappa = np.empty(len(mu) + 1)
        new_alpha = np.empty(len(mu) + 1)
        new_beta = np.empty(len(mu) + 1)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        new_kappa[1:] = kappa + 1.0
        new_mu[1:] = (kappa * mu + xt) / (kappa + 1.0)
        new_alpha[1:] = alpha + 0.5
        new_beta[1:] = beta + (kappa * (xt - mu) ** 2) / (2.0 * (kappa + 1.0))
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    r_star = int(np.argmax(r_post))
    return r_post, float(mu[r_star])


@dataclass
class BOCPDEntry:
    """BOCPD entry candidate (Phase-C #1): enter in the sign of the post-break filtered drift.

    Constructor args are the pre-registered, a-priori grid (SYSTEM §C.9 #1) -- set from the design, NEVER
    P&L-tuned; ``fit`` refines only the NIG prior strength / drift-scale from the TRAIN fold and FREEZES:
      * ``hazard``   -- constant change-point hazard ``H = 1/lambda`` (expected regime length ``lambda``).
      * ``p_break``  -- run-length-posterior-mass entry threshold: fire only when the collapsed mass at
        short run-lengths ``P(r <= r_break | F_{t-1})`` exceeds ``p_break`` (a break just fired), so a
        signal is emitted on a *detected* break, not on quiescent drift.
      * ``r_break``  -- the short-run-length window whose posterior mass ``p_break`` gates (default 1:
        ``r in {0, 1}`` = "a change-point occurred at or one bar before the decision cutoff").
      * ``mu0, kappa0, alpha0, beta0`` -- the Normal-inverse-Gamma prior; ``beta0`` (the innovation-scale
        prior) is re-set on ``fit`` from the train drift dispersion so the Student-t predictive is scaled
        to the instrument, then frozen. ``k`` counts the free parameters + the run-length latent
        dimension for the §C.4 parsimony penalty.
    """

    hazard: float = 0.02
    p_break: float = 0.5
    r_break: int = 1
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    warmup: int = 2
    # SYSTEM §C.4 parsimony count: hazard, p_break, r_break, and the 4 NIG params (7 fit/prior params)
    # + 1 run-length latent-state dimension = 8. Set at construction so isinstance-conformance sees it.
    k: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.hazard < 1.0:
            raise ValueError("hazard must lie strictly in (0, 1) (a per-bar change-point probability)")
        if not 0.0 < self.p_break <= 1.0:
            raise ValueError("p_break must lie in (0, 1] (a run-length-posterior mass threshold)")
        if self.r_break < 0:
            raise ValueError("r_break must be non-negative (a short-run-length window)")
        if self.kappa0 <= 0 or self.alpha0 <= 0 or self.beta0 <= 0:
            raise ValueError("NIG prior kappa0, alpha0, beta0 must be positive")
        self._fitted = False

    def fit(self, train: TrainFold) -> "BOCPDEntry":
        """Estimate the NIG innovation-scale prior on the TRAIN fold ONLY, then FREEZE (SIGNAL §C.1).

        Only the observation-scale prior ``beta0`` (and prior drift ``mu0``) are refined from train, so
        the Student-t predictive is scaled to the instrument's return dispersion; the hazard, thresholds,
        and the NIG shape are the frozen a-priori grid. No parameter is tuned on TEST (SYSTEM §C.9 #1).
        """
        body = train.series[train.warmup:]
        if len(body) >= 2:
            arr = np.asarray(body, dtype="float64")
            self.mu0 = float(arr.mean())
            var = float(arr.var(ddof=0))
            # Anchor the NIG innovation-scale prior to the train return variance by inverting the
            # relevant NIG moment of sigma^2 -- so the frozen Student-t predictive is scaled to the
            # instrument's dispersion for ANY a-priori alpha0 (closing the alpha0<=1 calibration gap).
            #   * alpha0 > 1: E[sigma^2] = beta0 / (alpha0 - 1) is defined -> invert the MEAN,
            #     beta0 = var * (alpha0 - 1)  (the intended, documented scaling; == var at alpha0 = 2).
            #   * alpha0 <= 1: the prior mean of sigma^2 is undefined (heavy inverse-Gamma tail), so
            #     invert the always-defined MODE  Mode[sigma^2] = beta0 / (alpha0 + 1)  instead,
            #     beta0 = var * (alpha0 + 1)  -- a principled train-var anchor, not an ad-hoc floor.
            # Either branch is set a priori from the frozen alpha0 and train var ONLY (SIGNAL §C.1);
            # never P&L-tuned. A degenerate (constant) train fold gives var == 0 -> the 1e-12 floor
            # keeps beta0 strictly positive so the Student-t scale stays well-defined.
            scale = (self.alpha0 - 1.0) if self.alpha0 > 1.0 else (self.alpha0 + 1.0)
            self.beta0 = max(var * scale, 1e-12)
        self._fitted = True
        return self

    def signal_at(self, cursor: ScanCursor) -> Side:
        """Causal one-step-ahead entry for the trade EXECUTED on ``cursor.t`` -- ``F_{t-1}``-measurable.

        Runs the BOCPD recursion over ``cursor.past() == series[:t]`` (indices ``0 .. t-1``) ONLY, so the
        filtered posterior read is ``P(r_{t-1} | F_{t-1})`` -- the pre-registered one-bar offset (docs/EDGE
        #1, SYSTEM §C.2): the decision that fires the bar-``t`` entry conditions on information up to
        ``t-1``, never bar ``t``. The final datum is fetched through ``cursor.read(t - 1)``, so a mutation
        to ``read(t)`` (a same-bar leak) trips the causal guard and the pinning test FAILS. Fires in the
        SIGN of the MAP-run filtered drift only when the short-run-length mass ``P(r <= r_break)`` clears
        ``p_break`` (a break just detected); else FLAT.
        """
        if not self._fitted:
            raise RuntimeError("signal_at before fit -- parameters not frozen yet")
        past = cursor.past()
        # Warm-up honesty: the run-length prior needs history within the fold before the scan may fire.
        if not warmup_ok(cursor.t, self.warmup) or len(past) < 2:
            return Side.FLAT
        # Touch the last causal datum through the guarded read so a same-bar leak is pinned (t-1 legal,
        # t or later RAISES under DAILY cadence). The value is already inside past(); this is the pin.
        _ = cursor.read(cursor.t - 1)
        x = np.asarray(past, dtype="float64")
        posterior, map_drift = run_length_posterior(
            x, hazard=self.hazard, mu0=self.mu0, kappa0=self.kappa0,
            alpha0=self.alpha0, beta0=self.beta0,
        )
        # Short-run-length mass: P(r <= r_break | F_{t-1}) -- large iff a change-point just fired.
        break_mass = float(posterior[: self.r_break + 1].sum())
        if break_mass < self.p_break:
            return Side.FLAT
        if map_drift > 0.0:
            return Side.LONG
        if map_drift < 0.0:
            return Side.SHORT
        return Side.FLAT


# Cadence is DAILY for this end-of-bar decision (the same-bar / one-bar-offset standard, SYSTEM §C.2);
# re-exported so a caller wiring a ScanCursor names the same tag the pinning test asserts against.
CADENCE = Cadence.DAILY
