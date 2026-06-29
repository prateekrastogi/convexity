"""Adversarial synthetic cross-asset panels (REGIME §12).

Generators so the whole pillar is testable with NO external data. Each returns a ``(W, n)`` return
matrix (already the panel ``returns`` shape; the engine de-means it via the panel builder, but for
isolated measure tests these are fed directly after de-meaning). A seeded ``numpy.random.Generator``
keeps every panel deterministic (Invariant DET).

Covers the §12 scenarios: calm/diversified, correlation-to-1 spike, vol spike, turbulence day,
liquidity blowout (via the exec/cost helpers), oscillation, near-singular covariance, rank-deficient
window, marginal-T/N noise, correlation-structure-fixed panel-shrink, condition-breach, calm-null
(ARL), panel-shrink (TURBNORM), intraday liq-flash-on-stale-corr, multi-measure handoff, drifted
gross deadband, per-minute liq null.
"""

from __future__ import annotations

import numpy as np

from convexity.regime.types import CostState, ExecState


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _demean(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def calm_panel(w: int = 300, n: int = 80, seed: int = 0, sigma: float = 0.01) -> np.ndarray:
    """Diversified, low-correlation, normal-vol panel -> g_t = 1.00 (REGIME §12 calm)."""
    return _demean(_rng(seed).normal(0.0, sigma, size=(w, n)))


def correlation_spike_panel(
    w: int = 300, n: int = 80, seed: int = 1, rho: float = 0.85, sigma: float = 0.01
) -> np.ndarray:
    """Correlation-to-1 spike: a strong single common factor -> high AR, ``delta_AR`` crosses +2.0.

    The §6 worked-example scenario in spirit: one dominant factor loads on all names so the top-k
    eigenvalue share jumps (REGIME §12 correlation-to-1 spike).
    """
    g = _rng(seed)
    common = g.normal(0.0, sigma, size=(w, 1))
    idio = g.normal(0.0, sigma, size=(w, n))
    x = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    return _demean(x)


def vol_spike_panel(
    w: int = 300, n: int = 80, seed: int = 2, base: float = 0.01, spike: float = 0.06
) -> np.ndarray:
    """Aggregate-vol spike in the trailing tail (rising sigma_agg + vov + steep term, §12 vol)."""
    g = _rng(seed)
    x = g.normal(0.0, base, size=(w, n))
    x[-20:] = g.normal(0.0, spike, size=(20, n))  # recent vol regime jump
    return _demean(x)


def turbulence_day_panel(
    w: int = 300, n: int = 80, seed: int = 3, sigma: float = 0.01, shock: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """A single statistically-unusual cross-asset vector (large Mahalanobis distance), §12 turbulence.

    Returns ``(history, current_vector)``: the history defines the trailing mean/cov; the current
    vector is an UNUSUAL combination (alternating-sign large move) far from the calm geometry. No
    sustained correlation shift, so absorption does NOT fire — turbulence does, then reverts.
    """
    g = _rng(seed)
    history = _demean(g.normal(0.0, sigma, size=(w, n)))
    unusual = shock * sigma * np.tile([1.0, -1.0], n)[:n]  # big AND unusually combined
    return history, unusual


def oscillating_series(n_bars: int = 60, low: float = 0.55, high: float = 0.72, seed: int = 4) -> np.ndarray:
    """A measure value dithering across a single band (e.g. vol_pct around 70), §12 oscillation.

    Used to assert hysteresis + dwell hold the state (no whipsaw). Values straddle the NORMAL band.
    """
    g = _rng(seed)
    return g.uniform(low, high, size=n_bars)


def near_singular_panel(w: int = 200, n: int = 80, seed: int = 5, sigma: float = 0.01) -> np.ndarray:
    """Highly collinear assets + n_t near W (near-singular covariance), §12 near-singular.

    A near-duplicated block forces an ill-conditioned matrix; the shrink/diagonal fallback must
    engage with no nonsense turbulence spike (Invariant COND).
    """
    g = _rng(seed)
    base = g.normal(0.0, sigma, size=(w, 4))
    blocks = np.repeat(base, n // 4 + 1, axis=1)[:, :n]
    blocks = blocks + g.normal(0.0, sigma * 1e-3, size=(w, n))  # near-duplicate columns
    return _demean(blocks)


def rank_deficient_panel(n: int = 80, w: int = 60, seed: int = 6, sigma: float = 0.01) -> np.ndarray:
    """A correlation window with W_AR < c_AR*n_t (rank-deficient), §12 rank-deficient (ARFLOOR).

    With ``w < 2*n`` the correlation matrix is rank-deficient; absorption must FREEZE and flag
    ``ar_window_deficient`` rather than emit a spurious top-k share.
    """
    return _demean(_rng(seed).normal(0.0, sigma, size=(w, n)))


def marginal_tn_noise_panel(n: int = 80, seed: int = 7, sigma: float = 0.01, c_ar: float = 2.0) -> np.ndarray:
    """Pure i.i.d. noise at T/N ≈ c_ar (marginal conditioning), §12 marginal-T/N (ARCOND).

    No true co-movement; on the RAW sample correlation the top-k share is MP-biased HIGH, on the
    SHRUNK matrix it is materially less biased.
    """
    w = int(round(c_ar * n))
    return _demean(_rng(seed).normal(0.0, sigma, size=(w, n)))


def corr_fixed_shrink_pair(
    w: int = 300, n_full: int = 80, drop: int = 20, seed: int = 8, rho: float = 0.4, sigma: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Same correlation STRUCTURE, n_t dropped by a session close, §12 corr-structure-fixed shrink.

    Returns ``(full_panel, shrunk_panel)`` with the SAME single-factor loading; only ``n_t`` differs
    (the shrunk one drops the last ``drop`` columns). ``delta_AR`` must NOT move across the n_t step
    (Invariant ARDISPNORM) — it freezes/flags ``ar_nt_changed``.
    """
    g = _rng(seed)
    common = g.normal(0.0, sigma, size=(w, 1))
    idio = g.normal(0.0, sigma, size=(w, n_full))
    full = _demean(np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
    shrunk = full[:, : n_full - drop]
    return full, _demean(shrunk)


def condition_breach_panel(w: int = 200, n: int = 80, seed: int = 9, sigma: float = 0.01) -> np.ndarray:
    """A near-duplicated asset PAIR leaving the shrunk matrix cond > cond_max, §12 condition-breach.

    Two columns are made nearly identical (duplicate up to tiny noise); turbulence must fall back to
    the diagonal inverse and absorption must freeze (Invariant CONDNUM).
    """
    g = _rng(seed)
    x = g.normal(0.0, sigma, size=(w, n))
    x[:, 1] = x[:, 0] + g.normal(0.0, sigma * 1e-7, size=w)  # near-identical pair
    return _demean(x)


def panel_shrink_pair(
    w: int = 300, n_full: int = 80, drop: int = 30, seed: int = 10, sigma: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Same return distribution, n_t dropped (TURBNORM), §12 panel-shrink.

    Returns ``(full, shrunk)`` i.i.d. panels differing only in ``n_t``; ``turb_pct`` of ``d_norm``
    must NOT move across the drop (Invariant TURBNORM).
    """
    g = _rng(seed)
    full = _demean(g.normal(0.0, sigma, size=(w, n_full)))
    return full, _demean(full[:, : n_full - drop])


# --------------------------------------------------------------------------- liquidity / exec / cost


def liquidity_blowout(live: tuple[str, ...], level: float = 0.97, kind: str = "spread") -> tuple[ExecState, CostState]:
    """A book-wide spread (or funding) blowout -> g^liq binds (REGIME §12 liquidity blowout)."""
    spread = {s: (level if kind == "spread" else 0.30) for s in live}
    funding = {s: (level if kind == "funding" else 0.30) for s in live}
    return ExecState(spread_pct=spread), CostState(funding_pct=funding)


def calm_liquidity(live: tuple[str, ...], level: float = 0.40) -> tuple[ExecState, CostState]:
    """Calm spread + funding (below all bands) -> g^liq = CALM (REGIME §12)."""
    return (
        ExecState(spread_pct={s: level for s in live}),
        CostState(funding_pct={s: level for s in live}),
    )


def minute_liq_blip_stream(
    n_minutes: int = 600, blip_every: int = 50, blip: float = 0.98, calm: float = 0.40, seed: int = 11
) -> np.ndarray:
    """Calm minute stream with intermittent SINGLE-minute spread blips, §12 per-minute liq null.

    The minute-level dwell ``D_liq_min`` must suppress the isolated blips so the realized intraday
    escalation rate matches ``ARL0^liq_min`` (Invariant ARL (6), §3.6 b').
    """
    g = _rng(seed)
    stream = g.uniform(0.2, calm, size=n_minutes)
    stream[::blip_every] = blip  # isolated single-minute blips
    return stream
