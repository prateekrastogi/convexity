"""One-sided CUSUM sequential test for the fractional-Kelly promotion ratchet (LEVERAGE.md §6.3).

Per P7, promotion from quarter- to half-Kelly is a SEQUENTIAL STATISTICAL TEST on LIVE weekly
returns, never a vibes-based judgment call. The test is a one-sided upper CUSUM (Page, 1954) on the
standardized weekly Sharpe surprise ``z_k``:

    z_k  = (SR_week,k − SR_ref) / SE(SR_week,k)     (standardized surprise; SR_ref = the deflated
                                                     SR_LCB_t frozen at promotion time)
    S_0  = 0
    S_k  = max(0, S_{k-1} + z_k − k_ref)            (upper CUSUM, reference offset k_ref)
    PROMOTE when S_k ≥ h_promote

In-control model (§6.3): ``z_k ~ N(0,1)`` i.i.d. — the same per-observation standardized convention
DSR's z-test uses. ``k_ref``/``h_promote`` are calibrated to a target ``ARL_0 ≈ 52`` weeks via the
Hawkins & Olwell (1998) closed form ``ARL_0 ≈ exp(2·k_ref·h_promote)/(2·k_ref²)``. The ``validate/``
Monte-Carlo check runs ONLY on synthetic ``N(0,1)`` draws, never on realized live P&L (P5/P7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def arl0_closed_form(k_ref: float, h_promote: float) -> float:
    """Closed-form one-sided-Gaussian-CUSUM in-control ARL (LEVERAGE.md §6.3, Hawkins & Olwell 1998).

    ``ARL_0 ≈ exp(2·k_ref·h_promote) / (2·k_ref²)`` — the expected number of in-control weeks before
    a FALSE promotion (§6.3). Used to derive ``h_promote`` from a target ``ARL_0`` and as the target
    the ``validate/arl.py`` Monte-Carlo check verifies the actual recursion against (§10).
    """
    return math.exp(2.0 * k_ref * h_promote) / (2.0 * k_ref * k_ref)


def cusum_step(s_prev: float, z_k: float, k_ref: float) -> float:
    """One upper-CUSUM update ``S_k = max(0, S_{k-1} + z_k − k_ref)`` (LEVERAGE.md §6.3, Page 1954).

    Accumulates the standardized surprise net of the reference offset ``k_ref``, floored at 0 (a
    one-sided upper CUSUM: it detects sustained OUTPERFORMANCE and never runs negative). Because it
    accumulates over the ENTIRE track record (never a forgetful rolling window), a single lucky
    week cannot carry it across ``h_promote`` if surrounding weeks pull it back down (Invariant
    RATCHET).
    """
    return max(0.0, s_prev + z_k - k_ref)


def cusum_path(z: np.ndarray, k_ref: float) -> np.ndarray:
    """The full CUSUM statistic path ``S_k`` over a standardized-surprise series (LEVERAGE.md §6.3).

    Returns ``S_1..S_n`` with ``S_0 = 0`` implicit. Vectorization is not worth the loss of clarity
    at weekly cadence (this runs once per week over a few-hundred-week history, §0 compute budget).
    """
    s = 0.0
    out = np.empty(len(z), dtype=float)
    for k, zk in enumerate(np.asarray(z, dtype=float)):
        s = cusum_step(s, float(zk), k_ref)
        out[k] = s
    return out


@dataclass(frozen=True)
class CusumResult:
    """OUTPUT of :func:`run_cusum` (LEVERAGE.md §6.3).

    ``s_k`` is the accumulated CUSUM at the most recent (current) week. ``crossed`` is the promotion
    signal: it is True iff the CURRENT accumulated ``s_k ≥ h_promote`` (§6.3 Invariant RATCHET —
    promotion is gated on the accumulated statistic, so a bad run that decays ``S_k`` back toward 0
    un-arms the ratchet: "the accumulated S_k from the surrounding weeks pulls it back below
    h_promote"). ``first_cross_week`` records the 1-indexed week the path first touched ``h_promote``
    (diagnostic only — NOT the promotion signal, since a single lucky spike that later decays must
    not promote).
    """

    s_k: float
    crossed: bool
    first_cross_week: int | None


def run_cusum(z: np.ndarray, *, k_ref: float, h_promote: float) -> CusumResult:
    """Run the one-sided CUSUM over standardized weekly surprises (LEVERAGE.md §6.3).

    ``z`` = standardized weekly Sharpe surprises ``(SR_week,k − SR_ref) / SE(SR_week,k)`` accumulated
    over the entire live track record since promotion eligibility began (§6.3, Invariant RATCHET —
    NOT a rolling window; the CUSUM remembers prior bad weeks by pinning ``S_k`` at 0 through them).
    The promotion signal ``crossed`` reads the CURRENT accumulated ``s_k`` (not a historical stopping
    time): a single lucky week that spikes ``S_k`` above ``h_promote`` and then decays back down as
    later weeks disappoint does NOT leave the ratchet armed — exactly the scenario Invariant RATCHET
    requires to fail promotion. ``first_cross_week`` is retained only for diagnostics/ARL analysis.
    """
    path = cusum_path(z, k_ref)
    first_cross: int | None = None
    for k, s in enumerate(path):
        if s >= h_promote:
            first_cross = k + 1
            break
    current = float(path[-1]) if path.size else 0.0
    return CusumResult(
        s_k=current,
        crossed=current >= h_promote,
        first_cross_week=first_cross,
    )


def standardized_surprise(
    sr_week: np.ndarray, sr_ref: float, se_sr: float
) -> np.ndarray:
    """Standardized weekly Sharpe surprise ``z_k = (SR_week,k − SR_ref) / SE(SR_week,k)`` (LEVERAGE.md §6.3).

    ``se_sr`` — the standardizing denominator — is assumed constant/known for the CUSUM recursion (a
    known-variance CUSUM, the classical Page 1954 setting, §6.3); ``σ̂_t``'s short half-life keeps
    it tracking regime change rather than stale (§6.3). ``se_sr <= 0`` is a degenerate input and
    raises rather than producing a NaN/inf surprise (§8 conservative).
    """
    if not (se_sr > 0.0):
        raise ValueError("SE(SR_week,k) must be > 0 for the standardized surprise (LEVERAGE §6.3)")
    return (np.asarray(sr_week, dtype=float) - float(sr_ref)) / float(se_sr)
