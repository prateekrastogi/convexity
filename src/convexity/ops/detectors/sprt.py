"""Wald SPRT (Wald 1945) + the CUSUM-of-the-LLR change-point form ("CusumSprt", §6.5a v1 default).

Spec: OPS §6.5 (the Wald SPRT on the hit-rate Bernoulli stream), §6.5a (the repeated-epoch
false-alarm problem and the CUSUM-SPRT fix). The edge stream is the lowest-frequency, highest-variance
stream (§5.4), so its detector must be *sequential and sample-efficient*.

Bernoulli per-trade log-likelihood-ratio increment for ``H0: p=p0`` vs ``H1: p=p1`` (p1<p0):

    ΔΛ_j = hit_j·ln(p1/p0) + (1-hit_j)·ln((1-p1)/(1-p0))

Wald boundaries from the chosen ``(α, β)``:  ``A = ln((1-β)/α) > 0``,  ``B = ln(β/(1-α)) < 0``.

The bare :class:`Sprt` decides ONCE (Λ≥A ⇒ H1 "edge decayed"; Λ≤B ⇒ H0 reset/keep trading); restarting
it turns the lifetime stream into many epochs and the per-epoch ``α`` compounds (§6.5a). The v1
default :class:`CusumSprt` runs a one-sided CUSUM on the *same* ΔΛ increment (``k=0``):
``G_j = max(0, G_{j-1} + ΔΛ_j)``, alarm when ``G_j > h_edge`` — a genuine change-point detector whose
in-control ARL0 is a single pinned number that drops into the §6.6 panel budget like every other CUSUM.
"""

from __future__ import annotations

import math

from convexity.ops.detectors.base import WarmupMixin, make_state
from convexity.ops.types import DetectorName, DetectorState


def wald_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Wald SPRT boundaries ``(A, B)`` from the type-I/type-II rates (§6.5). ``B < 0 < A``."""
    a = math.log((1.0 - beta) / alpha)
    b = math.log(beta / (1.0 - alpha))
    return a, b


def llr_increment(hit: int, p0: float, p1: float) -> float:
    """One Bernoulli log-likelihood-ratio increment ``ΔΛ`` for a trade outcome ``hit∈{0,1}`` (§6.5)."""
    if hit:
        return math.log(p1 / p0)
    return math.log((1.0 - p1) / (1.0 - p0))


def expected_llr(p: float, p0: float, p1: float) -> float:
    """Mean per-trade LLR increment ``E[ΔΛ | p]`` (§6.5) — used by the boundary-weighted ASN."""
    return p * math.log(p1 / p0) + (1.0 - p) * math.log((1.0 - p1) / (1.0 - p0))


def asn(p: float, p0: float, p1: float, alpha: float, beta: float) -> float:
    """Boundary-weighted Wald Average Sample Number ``E[N|H]`` (§6.5) — the expected detection delay.

    ``E[N|H1] ≈ ((1-β)·A + β·B) / E[ΔΛ|p1]``  and  ``E[N|H0] ≈ (α·A + (1-α)·B) / E[ΔΛ|p0]``. This is
    the FULL boundary-probability-weighted form; the hand-wave ``boundary/increment`` is wrong by
    ~2.5× (it omits which boundary is hit). For ``(p0=0.55,p1=0.45,α=0.05,β=0.10)`` this gives
    ASN_H1 ≈ 118, ASN_H0 ≈ 99 (NOT the old 40–50). Pass ``p=p1`` for ASN_H1, ``p=p0`` for ASN_H0.
    """
    a, b = wald_boundaries(alpha, beta)
    e_inc = expected_llr(p, p0, p1)
    if p >= p0:                      # under H0 the test tends to the lower boundary B
        numer = alpha * a + (1.0 - alpha) * b
    else:                            # under H1 the test tends to the upper boundary A
        numer = (1.0 - beta) * a + beta * b
    return numer / e_inc


class Sprt(WarmupMixin):
    """Bare Wald SPRT on the hit-rate Bernoulli stream (§6.5, option 1 / non-inferiority gate).

    ``update(hit)`` consumes a 0/1 trade outcome (passed through the ``z`` slot of the protocol — for
    this detector ``z`` IS the Bernoulli outcome, not a standardized residual). On a decision the
    accumulator resets (an H0 decision keeps trading; an H1 decision is the §7.7 trigger). The
    lifetime false-halt budget of repeated epochs is owned by the caller (§6.5a option 1); the v1
    default is :class:`CusumSprt`.
    """

    name: DetectorName = "sprt"

    def __init__(self, p0: float, p1: float, alpha: float, beta: float, *, n_warm: int = 0):
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.A, self.B = wald_boundaries(alpha, beta)
        self._init_warmup(n_warm)
        self.llr = 0.0
        self.last_decision = "continue"

    def reset(self) -> None:
        self.llr = 0.0
        self.last_decision = "continue"
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """Accumulate one trade outcome ``z∈{0,1}`` and apply the Wald decision rule (§6.5)."""
        warming = self._tick_warmup()
        hit = 1 if z >= 0.5 else 0
        self.llr += llr_increment(hit, self.p0, self.p1)
        breach = self.llr >= self.A
        if breach:
            self.last_decision = "H1_decayed"
        elif self.llr <= self.B:
            self.last_decision = "H0_intact"
            self.llr = 0.0           # reset epoch; keep trading (§6.5)
        else:
            self.last_decision = "continue"
        return make_state(
            detector=self.name,
            statistic=self.llr,
            limit=self.A,
            breach=breach,
            sample_idx=self.n_seen,
            warming_up=warming,
            diagnostics={"A": self.A, "B": self.B, "p0": self.p0, "p1": self.p1},
        )

    def accepts_h0(self) -> bool:
        """Non-inferiority gate (§17.1): True once the SPRT has accepted H0 (live ``p ≥ p0``)."""
        return self.last_decision == "H0_intact"


class CusumSprt(WarmupMixin):
    """CUSUM-of-the-LLR change-point detector (§6.5a, the v1 default for the edge hit-rate arm).

    Runs a one-sided CUSUM on the per-trade Bernoulli LLR increment ``ΔΛ`` with ``k=0`` and an upper
    limit ``h_edge`` whose in-control ARL0 is a single pinned number (Siegmund/MC on the ΔΛ
    distribution under ``p0``) that enters the §6.6 panel budget like every other CUSUM. It auto-resets
    on a clean stretch and never "decides H0 and stops", so its lifetime false-halt rate is a
    *designed* number, not the bare per-epoch ``α``.
    """

    name: DetectorName = "cusum_sprt"

    def __init__(self, p0: float, p1: float, h_edge: float, *, n_warm: int = 0):
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.h_edge = float(h_edge)
        self._init_warmup(n_warm)
        self.g = 0.0

    def reset(self) -> None:
        self.g = 0.0
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """``G_j = max(0, G_{j-1} + ΔΛ_j)`` on the trade outcome ``z∈{0,1}``; alarm when ``G > h_edge``.

        ΔΛ is positive on a *miss* (evidence for the decayed ``p1``) and negative on a *hit*, so the
        accumulator climbs as misses pile up and resets on a clean run of hits (a genuine change-point
        detector — §6.5a).
        """
        warming = self._tick_warmup()
        hit = 1 if z >= 0.5 else 0
        # ΔΛ = log-LR for H1 (decayed p1): +ve on a MISS (ln((1-p1)/(1-p0))>0), -ve on a HIT
        # (ln(p1/p0)<0). G climbs as misses pile up, resets on a clean run of hits (§6.5a).
        delta_lambda = llr_increment(hit, self.p0, self.p1)
        self.g = max(0.0, self.g + delta_lambda)
        breach = self.g > self.h_edge
        return make_state(
            detector=self.name,
            statistic=self.g,
            limit=self.h_edge,
            breach=breach,
            sample_idx=self.n_seen,
            warming_up=warming,
            diagnostics={"g": self.g, "p0": self.p0, "p1": self.p1},
        )
