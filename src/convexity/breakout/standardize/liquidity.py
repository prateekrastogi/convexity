"""Liquidity/volume/turnover seasonality (BREAKOUT §7 standardize/liquidity / §2 TEAL).

Past-only seasonal-adjusted volume "surprise": how unusual the current bar's volume is relative to
its own past intraday seasonal level. Feeds the flow channel's volume-surprise evidence (BREAKOUT
§features/flow: "Volume surprise must be seasonally adjusted"). All estimates are past-only.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from convexity.core.types import Bars


class LiquidityProfile:
    """Past-only seasonal volume surprise (BREAKOUT §7). One public method, ``volume_surprise``."""

    def __init__(self, *, lookback_sessions: int = 20) -> None:
        self.lookback_sessions = lookback_sessions

    def volume_surprise(self, bars: Bars) -> np.ndarray:
        """Return a per-bar standardized volume surprise (z of log-volume vs its past seasonal level).

        Zero where volume is absent. Uses sessions strictly before the current one for the seasonal
        baseline (no future quantiles, §12.1).
        """
        n = len(bars)
        if bars.v is None:
            return np.zeros(n)
        v = np.clip(np.asarray(bars.v, dtype=float), 1e-9, None)
        logv = np.log(v)

        opens = bars.is_session_open
        if opens is None:
            return self._expanding_z(logv)

        bar_in_sess = np.zeros(n, dtype=int)
        counter = 0
        for t in range(n):
            counter = 0 if opens[t] else counter + 1
            bar_in_sess[t] = counter

        out = np.zeros(n)
        history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max(self.lookback_sessions, 1))
        )
        for t in range(n):
            k = bar_in_sess[t]
            prior = history[k]
            if len(prior) >= 3:
                arr = np.fromiter(prior, dtype=float)
                mu, sd = float(np.mean(arr)), float(np.std(arr)) + 1e-9
                out[t] = (logv[t] - mu) / sd
            prior.append(logv[t])
        return out

    def _expanding_z(self, logv: np.ndarray) -> np.ndarray:
        n = logv.size
        out = np.zeros(n)
        for t in range(3, n):
            past = logv[:t]
            out[t] = (logv[t] - np.mean(past)) / (np.std(past) + 1e-9)
        return out
