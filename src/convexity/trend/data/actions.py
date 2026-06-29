"""CorporateActionGuard — equity split/dividend artifact flags (TREND §8 data/actions).

A split adjustment must NOT look like a trend death or a jump (§8). The guard scans for a single-bar
multiplicative discontinuity consistent with a clean split ratio (e.g. 2:1 -> ~0.5, 3:1 -> ~0.333)
and flags the boundary bar ``is_valid=False`` so the jump detector and the death union never see the
synthetic return. This is a hook: real corporate-action data lives upstream; absent it, this is a
conservative ratio-snap detector.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, InstrumentSpec

# Common clean split/reverse-split ratios; a single-bar close ratio near one of these is a CA artifact.
_SPLIT_RATIOS: tuple[float, ...] = (0.5, 1.0 / 3.0, 0.25, 0.2, 2.0, 3.0, 4.0, 5.0)


class CorporateActionGuard:
    """Flag equity split/dividend artifacts so they do not read as trend death (§8)."""

    def __init__(self, tol: float = 0.02) -> None:
        self.tol = tol

    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Flag single-bar close ratios that snap to a clean split ratio (§8)."""
        if spec.asset_class != "us_equity":
            return bars  # splits are an equity artifact only
        n = len(bars)
        if n < 2:
            return bars
        c = bars.c
        ratio = np.ones(n)
        ratio[1:] = c[1:] / np.where(c[:-1] == 0, np.nan, c[:-1])
        artifact = np.zeros(n, dtype=bool)
        for r in _SPLIT_RATIOS:
            artifact |= np.abs(ratio - r) <= self.tol * r
        artifact[0] = False
        valid = bars.is_valid.copy() if bars.is_valid is not None else np.ones(n, dtype=bool)
        valid &= ~artifact
        return Bars(**{**bars.__dict__, "is_valid": valid})
