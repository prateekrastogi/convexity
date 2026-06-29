"""Corporate-action adjustment hooks for US equities (BREAKOUT §7 data/corporate_actions / §3).

US-equity split/dividend adjustment so stitched/adjusted prices do NOT fabricate breakouts at the
action bar (BREAKOUT §13: "roll/corporate-action artifacts do not trigger after proper adjustment").
Back-adjusts a multiplicative split factor onto all bars BEFORE the ex-date so the price series is
continuous. The adjustment source is AMBER metadata (the caller supplies the action list); this
module only applies it deterministically and past-consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from convexity.core.types import Bars


@dataclass(frozen=True)
class CorporateAction:
    """One AMBER-supplied corporate action. ``ratio`` is the split ratio (2.0 => 2:1 split halves the
    post-date price), ``cash_dividend`` is a per-share cash dividend on the ex-date (quote ccy)."""

    ex_date_ms: int
    ratio: float = 1.0
    cash_dividend: float = 0.0


class CorporateActionAdjuster:
    """Apply split/dividend back-adjustment (BREAKOUT §7). One public method, ``adjust``."""

    def adjust(self, bars: Bars, actions: tuple[CorporateAction, ...]) -> Bars:
        """Back-adjust prices for splits/dividends so the action bar is not a fabricated breakout."""
        if not actions:
            return bars
        ts = bars.ts.astype(np.int64)
        factor = np.ones(len(bars))
        for a in actions:
            pre = ts < a.ex_date_ms  # bars strictly before the ex-date get scaled
            if a.ratio and a.ratio != 1.0:
                factor[pre] /= a.ratio  # pre-split prices divided down to match post-split scale
            if a.cash_dividend:
                # multiplicative total-return adjustment; uses the close at the action bar.
                idx = int(np.searchsorted(ts, a.ex_date_ms))
                if 0 < idx <= len(bars):
                    ref = bars.c[min(idx, len(bars) - 1)]
                    if ref > 0:
                        factor[pre] *= 1.0 - a.cash_dividend / ref
        return replace(
            bars,
            o=bars.o * factor,
            h=bars.h * factor,
            l=bars.l * factor,
            c=bars.c * factor,
        )
