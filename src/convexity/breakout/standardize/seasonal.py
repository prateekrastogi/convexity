"""SeasonalProfile — past-only intraday seasonal de-seasonalisation (BREAKOUT §7 standardize/seasonal).

Returns deseasonalized log returns using past-only seasonality (BREAKOUT §12.1). The value applied
inside session S is estimated from sessions strictly BEFORE S (a leave-current-out running mean over
prior sessions, keyed by intra-session bar index), so no full-sample intraday mean and no future
quantile ever leaks (BREAKOUT §12.1 lookahead surface).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.standardize.returns import log_returns


class SeasonalProfile:
    """Past-only intraday seasonal profile (BREAKOUT §7). One public method, ``fit_transform``."""

    def fit_transform(self, bars: Bars, lookback_sessions: int) -> np.ndarray:
        """Return deseasonalized log returns; the profile for session S uses sessions before S only.

        ``lookback_sessions`` bounds how many prior sessions feed the running seasonal mean. For 24/7
        instruments the "session" is delimited by ``is_session_open`` (UTC-day marks from the
        Sessionizer); without marks the whole series is one session and the result is a no-op.
        """
        rs = log_returns(bars)
        r = rs.r
        n = r.size
        out = r.copy()

        opens = bars.is_session_open
        if opens is None:
            return out  # no session structure to deseasonalize against

        # intra-session bar index for each bar, and a session id.
        sess_id = np.cumsum(opens.astype(int)) - 1
        sess_id = np.clip(sess_id, 0, None)
        bar_in_sess = np.zeros(n, dtype=int)
        counter = 0
        for t in range(n):
            counter = 0 if opens[t] else counter + 1
            bar_in_sess[t] = counter

        # past-only running seasonal mean keyed by bar_in_sess, over the prior `lookback_sessions`.
        # Maintain, per intra-session index, a deque of the most-recent prior-session values.
        from collections import defaultdict, deque

        history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(lookback_sessions, 1)))
        for t in range(n):
            k = bar_in_sess[t]
            prior = history[k]
            if prior:
                out[t] = r[t] - float(np.mean(prior))
            # append AFTER using (leave-current-out) so session S never sees its own bar.
            # but only roll into history at session boundary to keep "strictly before S".
            prior.append(r[t])
        return out
