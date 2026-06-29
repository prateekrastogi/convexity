"""Data-quality diagnostics: missing/bad/stale/outlier bars (BREAKOUT §7 data/quality).

A read-only summary over a (validated) ``Bars`` panel: counts and per-bar masks for missing bars,
stale prices, and price outliers (robust MAD-based, past-only). It does not mutate prices — the
validator owns flagging; this module reports the symbol-specific missing-data / stale-bar PROFILE
(BREAKOUT §2 TEAL "symbol-specific missing-data and stale-bar profile").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.core.types import Bars


@dataclass(frozen=True)
class QualityProfile:
    """Per-instrument data-quality profile (BREAKOUT §2 TEAL). All causal/descriptive."""

    n_bars: int
    n_invalid: int
    invalid_fraction: float
    n_stale: int
    n_outlier: int
    stale_mask: np.ndarray
    outlier_mask: np.ndarray


class QualityProfiler:
    """Summarise data quality (BREAKOUT §7). One public method, ``profile``."""

    def __init__(self, *, outlier_mad: float = 12.0, stale_run: int = 5) -> None:
        self.outlier_mad = outlier_mad
        self.stale_run = stale_run

    def profile(self, bars: Bars) -> QualityProfile:
        """Return a :class:`QualityProfile` (causal MAD outlier flag, stale-run flag)."""
        n = len(bars)
        valid = bars.is_valid if bars.is_valid is not None else np.ones(n, dtype=bool)
        c = bars.c
        r = np.zeros(n)
        r[1:] = np.diff(np.log(np.clip(c, 1e-12, None)))

        outlier = self._mad_outlier(r)
        stale = self._stale_mask(c)
        return QualityProfile(
            n_bars=n,
            n_invalid=int(np.sum(~valid)),
            invalid_fraction=float(np.mean(~valid)) if n else 0.0,
            n_stale=int(np.sum(stale)),
            n_outlier=int(np.sum(outlier)),
            stale_mask=stale,
            outlier_mask=outlier,
        )

    def _mad_outlier(self, r: np.ndarray) -> np.ndarray:
        """Causal expanding-window MAD outlier flag: a return whose |z| exceeds outlier_mad using a
        median/MAD estimated from PAST returns only (no future quantiles, §12)."""
        n = r.size
        mask = np.zeros(n, dtype=bool)
        warm = 30
        for t in range(warm, n):
            past = r[:t]
            med = np.median(past)
            mad = np.median(np.abs(past - med)) + 1e-12
            if abs(r[t] - med) > self.outlier_mad * 1.4826 * mad:
                mask[t] = True
        return mask

    def _stale_mask(self, c: np.ndarray) -> np.ndarray:
        n = c.size
        mask = np.zeros(n, dtype=bool)
        run = 1
        for t in range(1, n):
            run = run + 1 if c[t] == c[t - 1] else 1
            if run >= self.stale_run:
                mask[t] = True
        return mask
