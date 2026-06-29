"""CandidateGenerator — vol-normalized barrier-cross candidates (BREAKOUT §7 / Invariant J).

The cross is VOL-NORMALIZED, not raw price (Invariant J): a 1-tick poke past a Donchian high and a
1-sigma thrust past it are NOT the same event, and treating them identically makes the trigger
non-comparable across the universe (a tick is a different fraction of sigma on BTC than on a 5-tick
equity). Penetration past the strictly-pre-bar barrier (Invariant I) is expressed in causal-sigma:

    fire long candidate at t  iff  c[t] - upper_barrier[t] >= k * sigma_t
    fire short candidate at t iff  lower_barrier[t] - c[t] >= k * sigma_t

where ``sigma_t = StdSeries.sigma[t]`` is the F_{t-1}-measurable predictive scale and ``k`` is the
single shared RED/BLUE scalar (the SAME k on crypto and equities). This is a recall-oriented gate —
loose, but it NEVER emits EntrySignal objects (Inv A). It suppresses invalid bars, session artifacts
(gap bars), roll-window bars, and duplicated same-zone crosses.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.types import BreakoutGeometry, StdSeries


class CandidateGenerator:
    """Vol-normalized cross candidates (BREAKOUT §7). One public method, ``candidates``."""

    def __init__(self, *, cross_k: float = 1.0) -> None:
        self.cross_k = cross_k  # the single shared RED/BLUE scalar k (Inv J); never per-symbol.

    def candidates(self, bars: Bars, s: StdSeries, geom: BreakoutGeometry) -> np.ndarray:
        """Return an int8 candidate-direction array: +1 long cross, -1 short cross, 0 otherwise.

        The penetration uses the PREDICTIVE sigma in PRICE units (``sigma_t * close``), since
        ``StdSeries.sigma`` is a return scale and the barrier is a price level. The barrier itself is
        strictly pre-bar (Inv I) — raising bar t's own high cannot change the level it must break.
        """
        c = bars.c
        n = len(bars)
        sigma_price = s.sigma * c  # convert the F_{t-1} return scale to a price scale at bar t

        upper = geom.upper_barrier
        lower = geom.lower_barrier
        thresh = self.cross_k * sigma_price

        long_cross = np.isfinite(upper) & ((c - upper) >= thresh)
        short_cross = np.isfinite(lower) & ((lower - c) >= thresh)

        cand = np.zeros(n, dtype=np.int8)
        cand[long_cross] = 1
        cand[short_cross] = -1

        cand = self._suppress_artifacts(bars, cand)
        cand = self._suppress_same_zone(cand, upper, lower, c)
        return cand

    def _suppress_artifacts(self, bars: Bars, cand: np.ndarray) -> np.ndarray:
        """Zero out invalid bars, session-open gap bars, and roll-window bars (§7 'suppress session
        artifacts')."""
        if bars.is_valid is not None:
            cand = np.where(bars.is_valid.astype(bool), cand, 0).astype(np.int8)
        if bars.is_session_open is not None:
            # the first bar of a session is a gap, not a continuous cross.
            cand = np.where(bars.is_session_open.astype(bool), 0, cand).astype(np.int8)
        if bars.is_roll_window is not None:
            cand = np.where(bars.is_roll_window.astype(bool), 0, cand).astype(np.int8)
        return cand

    def _suppress_same_zone(
        self, cand: np.ndarray, upper: np.ndarray, lower: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """Keep only the FIRST cross of a given barrier zone; suppress repeats while the close stays
        beyond the same level (the dedupe module handles cross-signal cooldown later)."""
        n = cand.size
        out = cand.copy()
        active = 0  # +1/-1 while we remain beyond the level that triggered the last candidate
        for t in range(n):
            if cand[t] == 1:
                if active == 1:
                    out[t] = 0
                else:
                    active = 1
            elif cand[t] == -1:
                if active == -1:
                    out[t] = 0
                else:
                    active = -1
            else:
                # reset the active zone once price falls back inside the barrier band.
                if active == 1 and np.isfinite(upper[t]) and c[t] <= upper[t]:
                    active = 0
                elif active == -1 and np.isfinite(lower[t]) and c[t] >= lower[t]:
                    active = 0
        return out.astype(np.int8)
