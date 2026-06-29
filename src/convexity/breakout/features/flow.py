"""FlowEvidence — optional order-flow / volume-surprise evidence (BREAKOUT §7 / §6.6).

OPTIONAL channel (BREAKOUT §8: optional channels default to zero, enabled only if ablation proves
benefit). If quote/trade data is absent, returns zeros and the channel is marked absent. OFI is
venue-specific (BREAKOUT §3 crypto); volume surprise is seasonally adjusted (delegated to
``standardize/liquidity.py``).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, InstrumentSpec

from convexity.breakout.standardize.liquidity import LiquidityProfile
from convexity.breakout.types import StdSeries


class FlowEvidence:
    """Optional flow evidence (BREAKOUT §7). One public method, ``evidence``."""

    def __init__(self, *, enabled: bool = False) -> None:
        # disabled by default (§8): only on after ablation proves OOS benefit.
        self.enabled = enabled
        self._liq = LiquidityProfile()

    def evidence(self, bars: Bars, spec: InstrumentSpec, s: StdSeries) -> np.ndarray:
        """Return optional flow/volume-surprise evidence in [0, 1]; zeros when absent/disabled."""
        n = len(bars)
        if not self.enabled or bars.v is None:
            return np.zeros(n)
        surprise = self._liq.volume_surprise(bars)
        # squash the seasonal volume z-score into [0, 1] evidence (a positive surprise on the cross
        # bar supports the breakout).
        return np.clip(0.5 * (1.0 + np.tanh(0.5 * surprise)), 0.0, 1.0)

    @property
    def channel_absent(self) -> bool:
        return not self.enabled
