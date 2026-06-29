"""ExecDNA — slow daily per-instrument priors: typical spread, fill-rate, fee tier (§3).

A slow-moving (daily) per-instrument prior over the microstate the live engine warms up against: the
typical effective half-spread, the realized maker fill-rate, and the active fee tier. These are TEAL
(fitted from the instrument's OWN past data, causally) — they seed the cold-start before enough live
fills accrue, but they are NEVER a hand knob and never P&L-fit (Invariant T). Distinct from the live,
per-bar microstate estimators (which this only warm-starts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ExecDNA", "fit_exec_dna"]


@dataclass(frozen=True)
class ExecDNA:
    """The slow daily per-instrument execution prior (§3, TEAL)."""

    symbol: str
    typical_half_spread_bps: float
    realized_maker_fill_rate: float
    n_passive_fills: int
    typical_volume_proxy: float


def fit_exec_dna(
    symbol: str,
    daily_half_spread_bps: np.ndarray,
    passive_fill_outcomes: np.ndarray,
    daily_volume_proxy: np.ndarray,
) -> ExecDNA:
    """Fit the slow daily prior from the instrument's OWN realized history (§3, TEAL / causal).

    ``passive_fill_outcomes`` are 0/1 fill flags on the instrument's realized passive posts. Robust
    medians over the daily history so a single outlier day cannot move the prior.
    """
    hs = np.asarray(daily_half_spread_bps, dtype=float)
    fills = np.asarray(passive_fill_outcomes, dtype=float)
    vol = np.asarray(daily_volume_proxy, dtype=float)
    return ExecDNA(
        symbol=symbol,
        typical_half_spread_bps=float(np.median(hs)) if hs.size else 0.0,
        realized_maker_fill_rate=float(fills.mean()) if fills.size else 0.0,
        n_passive_fills=int(fills.size),
        typical_volume_proxy=float(np.median(vol)) if vol.size else 1.0,
    )
