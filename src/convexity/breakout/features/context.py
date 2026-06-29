"""ContextEvidence — optional cross-asset context evidence (BREAKOUT §7 / §3 cross-asset).

OPTIONAL channel (BREAKOUT §3: cross-asset context must be evaluated by ablation; disabled unless it
improves OOS entry quality). Context bars must be timestamp-aligned and current/lagged only — no
asset may use a later-closing market's future close (BREAKOUT §12.7). Missing context returns zeros.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.core.types import Bars, InstrumentSpec


class ContextEvidence:
    """Optional cross-asset context (BREAKOUT §7). One public method, ``evidence``."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled  # disabled by default until ablation proves benefit (§3).

    def evidence(
        self, bars: Bars, context: Mapping[str, Bars] | None, spec: InstrumentSpec
    ) -> np.ndarray:
        """Return cross-asset confirmation evidence in [0, 1]; zeros when absent/disabled.

        Aligns each context series to ``bars.ts`` by last-known value (current/lagged only, §12.7);
        the sign of the context's recent return that AGREES with the instrument's recent return
        raises confirmation evidence.
        """
        n = len(bars)
        if not self.enabled or not context:
            return np.zeros(n)

        own_ret = np.zeros(n)
        own_ret[1:] = np.diff(np.log(np.clip(bars.c, 1e-12, None)))
        agree = np.zeros(n)
        for ctx in context.values():
            aligned = self._align_lagged(bars.ts, ctx)
            cret = np.zeros(n)
            cret[1:] = np.diff(np.log(np.clip(aligned, 1e-12, None)))
            agree += np.sign(own_ret) == np.sign(cret)
        agree /= max(len(context), 1)
        return np.clip(agree, 0.0, 1.0)

    def _align_lagged(self, ts: np.ndarray, ctx: Bars) -> np.ndarray:
        """Last-known context close at or before each ts (no future leakage, §12.7)."""
        idx = np.searchsorted(ctx.ts, ts, side="right") - 1
        idx = np.clip(idx, 0, len(ctx) - 1)
        return ctx.c[idx]

    @property
    def channel_absent(self) -> bool:
        return not self.enabled
