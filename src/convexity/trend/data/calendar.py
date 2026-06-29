"""Sessionizer — session-open + active-minute/session metadata (TREND §8 data/calendar).

Attaches the clock metadata the dynamic horizon (§1.2) needs: which bars are session opens, how many
sessions have elapsed, and the active-minute count. Supports US-equity RTH/ETH, futures/metals
sessions, and crypto 24/7 UTC segmentation. For 24/7 instruments every bar is "active" and there are
no session opens unless the feed marks them (crypto day boundaries can be requested by the caller).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, InstrumentSpec

# calendar_ids that DECLARE session boundaries (mirrors the BREAKOUT resolve set).
_SESSION_CALENDARS: frozenset[str] = frozenset(
    {"us_equity_rth", "us_equity_eth", "cme_globex", "comex", "lme", "nyse", "nasdaq"}
)


class Sessionizer:
    """Attach session metadata used by the horizon clocks (§1.2 / §8)."""

    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Return bars with ``is_session_open`` set (if not already supplied).

        24/7 instruments (no session calendar) keep an all-False mask unless the feed supplied opens
        (e.g. crypto UTC-day boundaries). Session instruments without a supplied mask get a
        first-bar-only open as a degenerate fallback (real session detection lives upstream).
        """
        n = len(bars)
        if bars.is_session_open is not None:
            return bars
        so = np.zeros(n, dtype=bool)
        if spec.calendar_id in _SESSION_CALENDARS and n > 0:
            so[0] = True
        return Bars(**{**bars.__dict__, "is_session_open": so})

    @staticmethod
    def is_24_7(spec: InstrumentSpec) -> bool:
        """True iff the instrument has no declared session boundary (crypto)."""
        return spec.calendar_id not in _SESSION_CALENDARS
