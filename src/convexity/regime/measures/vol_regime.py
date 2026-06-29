"""Volatility regime — aggregate realized-vol percentile, vol-of-vol, term-structure (REGIME §2.1).

The aggregate volatility of the book's ENVIRONMENT and its instability:

  sigma_agg_t = sqrt( mean_i RV_i_t )     (equal-weight mean of per-asset causal realized variance)
  vol_pct_t   = causal percentile of sigma_agg_t within its OWN trailing window (P6)
  vov_t       = std(sigma_agg)/mean(sigma_agg) over a short window   (coefficient of variation)
  term_t      = sigma_agg(short) / sigma_agg(long)                   (vol term-structure slope)

``vol_pct`` is the PRIMARY driver of ``g^{vol}``; ``vov`` and ``term`` are confirming inputs that
can only TIGHTEN it (an elevated percentile WITH rising vol-of-vol and a steep term-structure is a
stronger signal than the percentile alone, §2.1) — applied in the band layer.

Invariant NODOUBLEVC: ``g^{vol}`` keys ONLY off the CROSS-asset AGGREGATE vol / vol-of-vol /
term-structure — quantities no single LEVERAGE ``L_vol`` term sees. REGIME never reads the
per-instrument ``forecast_vol`` vector LEVERAGE divides each position by; the two act on disjoint
quantities. (The DVOL-vs-realized choice is an AMBER metadata switch, §2.1 — a fall-through to
realized when implied vol is unavailable, never a forked code path.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.measures.turbulence import causal_percentile


def sigma_agg(returns: np.ndarray, rv_window: int) -> float:
    """``sqrt(mean_i RV_i)`` — cross-asset aggregate realized vol over the trailing ``rv_window``.

    ``RV_i`` is asset ``i``'s causal realized variance over the last ``rv_window`` rows of the panel
    (the equal-weight mean is the cross-asset aggregate the NODOUBLEVC boundary requires, §2.1).
    """
    x = np.asarray(returns, dtype=float)
    win = x[-rv_window:] if rv_window < x.shape[0] else x
    rv_per_asset = win.var(axis=0, ddof=0)  # per-asset realized variance
    return float(np.sqrt(rv_per_asset.mean()))


def term_structure(returns: np.ndarray, w_short: int, w_long: int) -> float:
    """``sigma_agg(short)/sigma_agg(long)`` — the vol term-structure slope (REGIME §2.1).

    ``> 1`` => short-term vol elevated vs baseline (stress onset); ``< 1`` => calm relative.
    """
    s_short = sigma_agg(returns, w_short)
    s_long = sigma_agg(returns, w_long)
    if s_long <= 0.0:
        return 1.0
    return s_short / s_long


@dataclass
class VolState:
    """Trailing causal history of ``sigma_agg`` for the percentile + vol-of-vol (REGIME §2.1)."""

    sigma_history: list[float] = field(default_factory=list)

    def push(self, sigma: float, max_len: int) -> None:
        self.sigma_history.append(float(sigma))
        if len(self.sigma_history) > max_len:
            del self.sigma_history[: len(self.sigma_history) - max_len]


@dataclass(frozen=True)
class VolResult:
    """``vol_pct`` (primary trigger) + the confirming ``vov``/``term`` + diagnostics (REGIME §2.1)."""

    vol_pct: float
    vov: float
    term: float
    fresh: bool
    diag: dict[str, float]


def compute_vol_regime(
    returns: np.ndarray,
    *,
    cfg: PolicyDefaults,
    state: VolState,
) -> VolResult:
    """One volatility-regime recompute on the current panel (REGIME §2.1). MUTATES ``state``.

    Emits ``vol_pct`` (the causal percentile of ``sigma_agg`` in its own past) plus the confirming
    ``vov`` and ``term`` that can only tighten ``g^{vol}`` in the band layer.
    """
    s_agg = sigma_agg(returns, cfg.w_rv)
    diag: dict[str, float] = {"sigma_agg": s_agg}

    vol_pct = causal_percentile(s_agg, np.asarray(state.sigma_history, dtype=float))
    state.push(s_agg, cfg.w_pct)  # push AFTER the percentile (today's value vs its OWN past)

    recent = np.asarray(state.sigma_history[-cfg.w_vov :], dtype=float)
    if recent.size >= 2 and recent.mean() > 0:
        vov = float(recent.std(ddof=0) / recent.mean())  # coefficient of variation
    else:
        vov = 0.0

    term = term_structure(returns, cfg.w_term_short, cfg.w_term_long)

    diag["vol_pct"] = vol_pct
    diag["vov"] = vov
    diag["term"] = term
    diag["fresh"] = 1.0
    return VolResult(vol_pct=vol_pct, vov=vov, term=term, fresh=True, diag=diag)
