"""Adversarial synthetic streams so OPS is testable with no live feed (§12).

Spec: OPS §12. Generate the adversarial streams the module's behavioral assertions need: the in-control
ARL0 check, the AR(1)-Student-t dependent/fat-tailed check (§6.7d), the warm-up transient, the live-
variance-shift, the repeated edge-epoch stream, the §10 slow cost drift, the vol-regime jump, edge
decay, frequency collapse, single catastrophic fill, fill-rate collapse, recon mismatch, heartbeat
loss / staleness / clock-skew, the drawdown ramp through the governor → backstop, gap-through-both-
thresholds, bot-kill mid-position, double-consumed HALT, HALT-during-in-flight, flatten-rejected, and
the flapping stream. Pure numpy — no external data.
"""

from __future__ import annotations

import numpy as np

from convexity.ops.types import EquityRef, StreamObs


# --------------------------------------------------------------------------- in-control synthetics

def in_control_iid(n: int, *, mu0: float = 0.0, sigma0: float = 1.0, seed: int = 0) -> np.ndarray:
    """An IID-Gaussian in-control residual stream (the ARL0 check, §12). Detectors rarely fire."""
    rng = np.random.default_rng(seed)
    return mu0 + sigma0 * rng.standard_normal(n)


def in_control_ar1_student_t(n: int, *, mu0: float = 0.0, sigma0: float = 1.0, rho: float = 0.4,
                             dof: float = 5.0, seed: int = 0) -> np.ndarray:
    """An autocorrelated AND fat-tailed in-control residual (§6.7d/§12).

    The realized ARL0 lands within CI of 370 only AFTER prewhitening (§6.7a); it FLOODS if the raw
    residual is fed the IID limit. ``sigma0`` scales to the raw residual units; ``rho``/``dof`` are the
    measured in-control dependence/fat-tail params.
    """
    rng = np.random.default_rng(seed)
    t_var = dof / (dof - 2.0)
    innov_scale = np.sqrt((1.0 - rho * rho) / t_var)
    innov = rng.standard_t(dof, size=n) * innov_scale
    x = np.empty(n)
    x[0] = innov[0] / np.sqrt(max(1e-9, 1.0 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + innov[i]
    return mu0 + sigma0 * x


def warmup_transient(n_warm: int, *, spike_at: int = 2, spike: float = 6.0,
                     seed: int = 0) -> np.ndarray:
    """An in-control stream with a transient excursion INSIDE the warm-up window (§6.8/§12).

    The detector must emit zero PAGE/DEGRADE/HALT before ``N_warm`` (logs only) even on this spike,
    while its recursion still updates. ``n_warm`` sets the window; the spike lands at ``spike_at``.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_warm + 20)
    x[spike_at] = spike
    return x


def live_variance_shift(n: int, *, mu0: float = 0.0, sigma0: float = 1.0, vol_mult: float = 2.0,
                        change_at: int | None = None, seed: int = 0) -> np.ndarray:
    """A stream whose live VARIANCE shifts (mean unchanged) — live vol 2× validated (§12/Invariant
    LIM-(b)). Standardizing against the FROZEN ``σ0`` keeps the raw trip point at ``μ0+h·σ0``; the
    logged ``σ_live`` moves, the alarm sample does not."""
    rng = np.random.default_rng(seed)
    change_at = change_at if change_at is not None else n // 2
    s = np.full(n, sigma0)
    s[change_at:] = sigma0 * vol_mult
    return mu0 + s * rng.standard_normal(n)


# --------------------------------------------------------------------------- the §10 cost drift

def cost_drift_section10() -> tuple[np.ndarray, np.ndarray]:
    """The §10 worked-example cost stream (the regression fixture, Invariant WE).

    Samples 1–10 are the in-control noise around 0; from sample 11 realized cost runs +3.0 bps above
    forecast every fill (a +1.5σ shift at σ0=2.0). Returns ``(e_cost, z)`` so a test can reproduce the
    full ``C+`` recursion bit-for-bit (DEGRADE@15, HALT@18).
    """
    e_cost = np.array(
        [+0.4, -1.0, +0.8, -0.6, +1.2, -0.4, +0.2, -0.8, +0.6, -0.2]
        + [+3.0] * 8,
        dtype=float,
    )
    z = e_cost / 2.0  # standardize by σ0 = 2.0
    return e_cost, z


def slow_cost_drift(n: int = 60, *, change_at: int = 20, drift_per_step: float = 0.1,
                    sigma0: float = 1.0, seed: int = 0) -> tuple[np.ndarray, int]:
    """A slow ramping cost drift (§12): in control then a gradual positive drift the CUSUM/PH catch and
    the Shewhart-only baseline misses. Returns ``(stream, change_at)``."""
    rng = np.random.default_rng(seed)
    x = sigma0 * rng.standard_normal(n)
    for i in range(change_at, n):
        x[i] += drift_per_step * (i - change_at + 1)
    return x, change_at


# --------------------------------------------------------------------------- regime / edge / fills

def vol_regime_jump(n: int = 60, *, change_at: int = 20, step: float = 0.8,
                    seed: int = 0) -> tuple[np.ndarray, int]:
    """A vol-regime step up in ``e_vol`` (§12): EWMA fires; DEGRADE. Returns ``(stream, change_at)``."""
    rng = np.random.default_rng(seed)
    x = 0.25 * rng.standard_normal(n)
    x[change_at:] += step
    return x, change_at


def edge_decay_hits(n: int = 300, *, p0: float = 0.55, p1: float = 0.45, change_at: int = 100,
                    seed: int = 0) -> tuple[np.ndarray, int]:
    """A hit-rate stream that steps ``p0 → p1`` at ``change_at`` (§12): the SPRT/CUSUM-SPRT decide H1
    in ~ASN trades. Returns ``(hits, change_at)`` of 0/1 outcomes."""
    rng = np.random.default_rng(seed)
    hits = np.empty(n)
    for j in range(n):
        p = p0 if j < change_at else p1
        hits[j] = 1.0 if rng.random() < p else 0.0
    return hits, change_at


def repeated_edge_epochs(n: int = 5000, *, p0: float = 0.55, seed: int = 0) -> np.ndarray:
    """An IN-CONTROL hit-rate stream over many CUSUM-SPRT epochs (§12/§6.5a): the lifetime false-halt
    rate must match the budgeted ARL0-equivalent, NOT the per-epoch α a bare restart-SPRT inflates."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p0).astype(float)


def frequency_collapse(n: int = 40, *, validated_freq: float = 10.0, change_at: int = 20,
                       collapsed: float = 2.0, seed: int = 0) -> tuple[np.ndarray, int]:
    """A trade-frequency collapse (§12): the count per window drops far below the validated frequency —
    the system has stopped finding trades. Returns ``(count_residual, change_at)``."""
    rng = np.random.default_rng(seed)
    counts = validated_freq + rng.standard_normal(n)
    counts[change_at:] = collapsed
    return counts - validated_freq, change_at


def single_catastrophic_fill(n: int = 30, *, spike_at: int = 15, spike: float = 8.0,
                             seed: int = 0) -> tuple[np.ndarray, int]:
    """A single catastrophic fill (§12): Shewhart fires in ONE sample; CUSUM lags — the Shewhart+CUSUM
    complementarity. Returns ``(stream, spike_at)``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x[spike_at] = spike
    return x, spike_at


def fill_rate_collapse(n: int = 30, *, change_at: int = 15, floor_rate: float = 0.2,
                       seed: int = 0) -> tuple[np.ndarray, int]:
    """A fill-rate collapse (§12): realized fill rate drops below the floor; DEGRADE → HALT on exit-
    fail. Returns ``(realized_fillrate, change_at)``."""
    rng = np.random.default_rng(seed)
    fr = 0.95 + 0.02 * rng.standard_normal(n)
    fr[change_at:] = floor_rate
    return np.clip(fr, 0.0, 1.0), change_at


# --------------------------------------------------------------------------- binary / equity streams

def recon_mismatch_stream(n: int = 10, *, mismatch_at: int = 0) -> list[bool]:
    """A reconciliation stream that is OK then mismatches (§12): immediate HALT, no SPC. Returns a list
    of ``reconciliation_ok`` flags (False == mismatch)."""
    flags = [True] * n
    if 0 <= mismatch_at < n:
        flags[mismatch_at] = False
    return flags


def heartbeat_loss_codes(n: int = 6) -> list[float]:
    """A heartbeat-loss sequence (§12): consecutive missed heartbeats → HALT at the miss count. Returns
    a list of HB_MISS codes (the brake accumulates them)."""
    from convexity.ops.streams.heartbeat import HB_MISS
    return [HB_MISS] * n


def drawdown_ramp_through_governor(*, peak: float = 1000.0) -> list[EquityRef]:
    """A drawdown ramp to the 43% governor-flat, then SITTING at ≥45% → OPS HALT (§12).

    On a SHARED equity/peak reference so 45% > 43% never inverts. Returns a sequence of EquityRef from
    a small drawdown, through the governor flat, to a sit at 46% (past the 45% DD_kill)."""
    dds = [0.10, 0.30, 0.42, 0.43, 0.44, 0.46]
    return [EquityRef(equity=peak * (1.0 - dd), peak=peak) for dd in dds]


def gap_through_both_thresholds(*, peak: float = 1000.0) -> list[EquityRef]:
    """Equity at 38% DD that GAPS in one bar past both 43% and 45% (§12/§7.1a).

    The DD_kill fires AFTER the loss — it does NOT prevent gap ruin; only the venue stop + L_tail bound
    it. The synth asserts the honest concession, not a false guarantee."""
    return [EquityRef(equity=peak * 0.62, peak=peak),   # 38% DD
            EquityRef(equity=peak * 0.50, peak=peak)]   # gapped to 50% DD (past 45%) in one bar


def flapping_stream(n: int = 60, *, seed: int = 0) -> np.ndarray:
    """A clean-broken-clean stream (§12): the re-arm must NOT auto-un-trip on the clean stretch (§8.4).
    A big excursion in the middle then back to in-control."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x[n // 3: n // 3 + 8] += 3.0   # the broken stretch
    return x


def make_obs(stream: str, values: np.ndarray, *, binary: bool = False) -> list[StreamObs]:
    """Wrap a raw value array into a list of :class:`StreamObs` for a stream (test convenience)."""
    return [
        StreamObs(stream=stream, sample_idx=i + 1, value=float(v), binary=binary)  # type: ignore[arg-type]
        for i, v in enumerate(values)
    ]


__all__ = [
    "in_control_iid",
    "in_control_ar1_student_t",
    "warmup_transient",
    "live_variance_shift",
    "cost_drift_section10",
    "slow_cost_drift",
    "vol_regime_jump",
    "edge_decay_hits",
    "repeated_edge_epochs",
    "frequency_collapse",
    "single_catastrophic_fill",
    "fill_rate_collapse",
    "recon_mismatch_stream",
    "heartbeat_loss_codes",
    "drawdown_ramp_through_governor",
    "gap_through_both_thresholds",
    "flapping_stream",
    "make_obs",
]
