"""VolatilityStandardizer — FIRST-CLASS one-step-ahead predictive scale (BREAKOUT §7 / §12.2 / Inv J).

THE load-bearing measurability module. ``sigma[t]`` is the PREDICTIVE scale formed from returns
``[..t-1]`` ONLY (F_{t-1}-measurable; it does NOT include the contemporaneous ``r[t]``). The
standardized return is ``x[t] = r[t] / sigma[t]`` — numerator F_t, denominator F_{t-1}. This is
exactly what seals the surface Invariant J relies on: the breakout cross test
``c[t] - upper_barrier[t] >= k * sigma[t]`` has a denominator fully known before t (the
"future-needle trap": raising bar t cannot change ``sigma[t]``).

Implementation: a causal EWMA of squared deseasonalized returns, shifted one bar forward so the
estimate applied at t is the EWMA computed through t-1 only. Session-open gaps are excluded from the
vol update (handled separately, §standardize/volatility). The vol/liquidity regime buckets are
formed from PAST-ONLY expanding quantiles (no future quantiles, Invariant M local conditioning).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

from convexity.breakout.standardize.returns import log_returns
from convexity.breakout.types import StdSeries


class VolatilityStandardizer:
    """Past-only predictive volatility + regime buckets (BREAKOUT §7/§12.2). Public: ``standardize``."""

    def __init__(self, *, n_vol_buckets: int = 3, n_liq_buckets: int = 3, floor: float = 1e-9) -> None:
        self.n_vol_buckets = n_vol_buckets
        self.n_liq_buckets = n_liq_buckets
        self.floor = floor

    def standardize(self, bars: Bars, deseasonalized: np.ndarray, window: int) -> StdSeries:
        """Return ``StdSeries`` with the one-step-ahead predictive ``sigma`` (Inv J), ``x=r/sigma``,
        and past-only ``vol_bucket`` / ``liq_bucket`` integers.

        ``window`` sets the EWMA half-life-equivalent span (alpha = 2/(window+1)). The contemporaneous
        squared return is folded in only AFTER ``sigma[t]`` is read, so ``sigma`` stays F_{t-1}.
        """
        rs = log_returns(bars)
        r = rs.r
        is_gap = rs.is_gap
        n = r.size
        alpha = 2.0 / (window + 1.0)

        sigma = np.empty(n)
        # seed: a robust scale from the first `window` deseasonalized returns is NOT allowed (would
        # peek). Instead seed the EWMA recursively; the warm-up bars get the running estimate so far.
        ewvar = 0.0
        seeded = False
        for t in range(n):
            if not seeded:
                # before any update, sigma[t] is the running estimate (or floor on bar 0).
                sigma[t] = np.sqrt(ewvar) if ewvar > 0 else self.floor
            else:
                sigma[t] = np.sqrt(max(ewvar, self.floor**2))
            # AFTER assigning sigma[t] (F_{t-1}), fold in bar t's deseasonalized return (-> F_t).
            if not is_gap[t]:
                d = deseasonalized[t]
                if seeded:
                    ewvar = (1.0 - alpha) * ewvar + alpha * d * d
                else:
                    ewvar = d * d
                    seeded = True
        sigma = np.clip(sigma, self.floor, None)

        x = r / sigma  # numerator F_t, denominator F_{t-1}

        vol_bucket = self._past_only_buckets(sigma, self.n_vol_buckets)
        liq_bucket = self._liquidity_buckets(bars, self.n_liq_buckets)

        # NOTE (§12.2): NO full-sample Student-t tail fit here. A dof estimate must be fold-local /
        # online (standardize/tails.fit_student_t_nu), so the live StdSeries carries no ``nu`` field.
        return StdSeries(
            ts=bars.ts, x=x, r=r, sigma=sigma,
            vol_bucket=vol_bucket, liq_bucket=liq_bucket,
        )

    def _past_only_buckets(self, series: np.ndarray, k: int) -> np.ndarray:
        """Assign each bar to one of ``k`` regime buckets via PAST-ONLY expanding quantiles
        (Invariant M / §12.3: regime bucket boundaries are online-recursive, never full-sample)."""
        n = series.size
        buckets = np.zeros(n, dtype=int)
        warm = 30
        for t in range(n):
            if t < warm:
                buckets[t] = k // 2
                continue
            past = series[:t]
            qs = np.quantile(past, np.linspace(0, 1, k + 1)[1:-1])
            buckets[t] = int(np.searchsorted(qs, series[t]))
        return np.clip(buckets, 0, k - 1)

    def _liquidity_buckets(self, bars: Bars, k: int) -> np.ndarray:
        if bars.v is None:
            return np.full(len(bars), k // 2, dtype=int)
        return self._past_only_buckets(np.asarray(bars.v, dtype=float), k)
