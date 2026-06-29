"""RobustVolatility — causal vol stack + integrated variance since entry (TREND §9 volatility).

A cheap-compute vol stack (§9):
  * EWMA close-to-close log-return volatility.
  * MAD/median robust fallback.
  * Yang-Zhang-style OHLC estimate only when OHLC quality is good.
  * Hybrid that chooses/fuses by data quality.

The output carries per-bar sigma AND the integrated variance since entry,

    integrated_var_since_entry = sum_{i=entry_idx+1..idx} sigma_i^2  (+ gap_var once per boundary)

which is MONOTONE NONDECREASING after entry (acceptance gate). Session gaps contribute to gap_var
ONCE at the boundary. No full-sample volatility, no future quantiles — every value at bar t uses only
bars at or before t (Inv H).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Position
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.types import VolState

_SQRT_4LN2 = 1.6651092223153954  # for the MAD-to-sigma constant if needed


class RobustVolatility:
    """Causal robust volatility (§9). ``update`` returns a :class:`VolState` for one bar."""

    def __init__(self, window: int = 60) -> None:
        self.window = window

    # ---- per-method one-bar sigma (all causal, past-only) ---------------------------------
    def _ewma_sigma(self, logret: np.ndarray, lam: float) -> float:
        """EWMA of squared log returns -> sigma. Uses returns up to and including the current bar."""
        if logret.size == 0:
            return 0.0
        w = lam ** np.arange(logret.size - 1, -1, -1)
        var = float(np.sum(w * logret**2) / np.sum(w))
        return float(np.sqrt(max(var, 0.0)))

    def _mad_sigma(self, logret: np.ndarray) -> float:
        """Robust MAD-scaled sigma over the trailing window (median-absolute-deviation)."""
        if logret.size == 0:
            return 0.0
        med = np.median(logret)
        mad = np.median(np.abs(logret - med))
        return float(1.4826 * mad)  # MAD -> Gaussian-consistent sigma

    def _yang_zhang_sigma(self, bars: Bars, lo: int, hi: int) -> float:
        """Yang-Zhang OHLC estimator over [lo, hi]. Per-bar sigma (sqrt of YZ variance)."""
        o, h, low, c = bars.o[lo:hi], bars.h[lo:hi], bars.l[lo:hi], bars.c[lo:hi]
        if o.size < 2:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            # overnight (close-to-open), open-to-close, Rogers-Satchell.
            o_c_prev = np.log(o[1:] / c[:-1])
            c_o = np.log(c / o)
            rs = np.log(h / c) * np.log(h / o) + np.log(low / c) * np.log(low / o)
        n = o_c_prev.size
        if n < 1:
            return 0.0
        var_o = np.var(o_c_prev, ddof=0)
        var_c = np.var(c_o[1:], ddof=0) if c_o[1:].size else 0.0
        var_rs = float(np.mean(rs[1:])) if rs[1:].size else 0.0
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
        yz = var_o + k * var_c + (1 - k) * var_rs
        return float(np.sqrt(max(yz, 0.0)))

    def _ohlc_quality_good(self, bars: Bars, lo: int, hi: int) -> bool:
        """OHLC quality is good iff highs/lows are non-degenerate and bars are valid in the window."""
        h, low = bars.h[lo:hi], bars.l[lo:hi]
        if h.size < 2:
            return False
        nondegenerate = np.mean(h > low) > 0.9
        if bars.is_valid is not None:
            nondegenerate &= bool(np.mean(bars.is_valid[lo:hi]) > 0.9)
        return bool(nondegenerate)

    # ---- public update --------------------------------------------------------------------
    def update(
        self,
        bars: Bars,
        idx: int,
        position: Position,
        cfg: ResolvedConfig,
        prev_integrated_var: float = 0.0,
        gap_applied: bool = False,
    ) -> VolState:
        """Return robust causal volatility state at ``idx`` (§9).

        ``prev_integrated_var`` is the integrated variance through ``idx-1`` so the result is monotone
        nondecreasing; the engine threads it. ``gap_applied`` records whether the boundary gap_var was
        already booked (so it is applied ONCE per session boundary).
        """
        d = cfg.defaults
        lo = max(0, idx - self.window)
        hi = idx + 1
        c = bars.c[lo:hi]
        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.diff(np.log(np.where(c > 0, c, np.nan)))
        logret = logret[np.isfinite(logret)]

        method = cfg.vol_method
        quality = "ok"
        if logret.size < d.vol_warmup:
            sigma = self._mad_sigma(logret) if logret.size else 0.0
            chosen: str = "mad"
            quality = "warmup"
        elif method == "ewma":
            sigma, chosen = self._ewma_sigma(logret, d.ewma_lambda), "ewma"
        elif method == "mad":
            sigma, chosen = self._mad_sigma(logret), "mad"
        elif method == "yang_zhang":
            sigma, chosen = self._yang_zhang_sigma(bars, lo, hi), "yang_zhang"
        else:  # hybrid: YZ when quality is good, else fuse EWMA + MAD
            if self._ohlc_quality_good(bars, lo, hi):
                yz = self._yang_zhang_sigma(bars, lo, hi)
                ewma = self._ewma_sigma(logret, d.ewma_lambda)
                sigma = float(np.sqrt(0.5 * (yz**2 + ewma**2)))
                chosen = "yang_zhang"
            else:
                ewma = self._ewma_sigma(logret, d.ewma_lambda)
                mad = self._mad_sigma(logret)
                sigma = max(ewma, mad)  # conservative: trust the larger when quality is poor
                chosen = "hybrid"
                quality = "ohlc_poor"

        if not np.isfinite(sigma) or sigma < 0:
            sigma = 0.0
            quality = "nonfinite_fallback"

        # Gap variance: applied ONCE at a session boundary (§9). gap = log(open_t / close_{t-1}).
        gap_var = 0.0
        if not gap_applied and idx >= 1 and bars.is_session_open is not None:
            if bool(bars.is_session_open[idx]):
                prev_c = bars.c[idx - 1]
                this_o = bars.o[idx]
                if prev_c > 0 and this_o > 0:
                    gap_log = np.log(this_o / prev_c)
                    gap_var = float(d.gap_var_multiple * gap_log**2)

        # Integrated variance: monotone nondecreasing after entry.
        incr = sigma**2 + gap_var if idx > position.entry_idx else 0.0
        integrated = float(prev_integrated_var + max(incr, 0.0))

        return VolState(
            method=chosen,  # type: ignore[arg-type]
            sigma_log=float(sigma),
            sigma_price=float(sigma * bars.c[idx]) if bars.c[idx] > 0 else 0.0,
            integrated_var_since_entry=integrated,
            gap_var=gap_var,
            vol_bucket=self._bucket(sigma, d.n_vol_buckets),
            liq_bucket=self._liq_bucket(bars, idx, d.n_liq_buckets),
            tail_nu=8.0,  # conservative Student-t dof for null calibration (not full-sample fit)
            quality_flag=quality,
        )

    @staticmethod
    def _bucket(sigma: float, n: int) -> int:
        """Coarse vol bucket by fixed log-spaced thresholds (past-only; no future quantiles)."""
        thresholds = (0.0008, 0.002)  # 1-min crypto-perp-ish edges
        b = int(np.searchsorted(thresholds, sigma))
        return min(b, n - 1)

    @staticmethod
    def _liq_bucket(bars: Bars, idx: int, n: int) -> int:
        """Coarse liquidity bucket by trailing volume (past-only)."""
        if bars.v is None:
            return 0
        lo = max(0, idx - 60)
        v = bars.v[lo : idx + 1]
        if v.size == 0:
            return 0
        cur = bars.v[idx]
        med = np.median(v)
        if med <= 0:
            return 0
        ratio = cur / med
        if ratio < 0.5:
            return 0
        if ratio < 1.5:
            return min(1, n - 1)
        return n - 1
