"""Synthetic 1-minute regimes + adversarial cases (BREAKOUT §13).

Every BREAKOUT unit test is driven from a seeded generator here, so the suite needs NO external
data (BREAKOUT §0.1). The thirteen regimes the spec enumerates are each a ``regime_*`` function
returning a :class:`convexity.core.types.Bars`; :func:`make_regime` dispatches by name and
:data:`REGIMES` lists all thirteen.

Synthetic behaviour the tests then verify (BREAKOUT §13):
  * true compression-to-expansion breakouts score high,
  * isolated mean-reverting spikes score lower or carry high false-break risk,
  * roll / corporate-action artifacts do not trigger after proper adjustment,
  * future changes do not alter past signals.

All paths are GBM-style on log price with volatility clustering; OHLC is synthesised from an
intrabar Brownian bridge so highs/lows are consistent (h >= max(o,c) >= min(o,c) >= l).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

MINUTE_MS = 60_000


def _ohlc_from_path(
    rng: np.random.Generator, close: np.ndarray, sigma_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build consistent OHLC from a close path + per-bar vol via an intrabar two-draw bridge."""
    n = close.size
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    # two interior samples per bar to give a plausible wick around o->c
    span = sigma_bar * close * 0.5
    up = rng.standard_normal(n) * span
    dn = rng.standard_normal(n) * span
    hi = np.maximum(open_, close) + np.abs(up)
    lo = np.minimum(open_, close) - np.abs(dn)
    return open_, hi, lo, close


def _assemble(
    rng: np.random.Generator,
    logret: np.ndarray,
    *,
    sigma_bar: np.ndarray,
    start_price: float = 100.0,
    base_volume: float = 1000.0,
    volume: np.ndarray | None = None,
    is_valid: np.ndarray | None = None,
    is_session_open: np.ndarray | None = None,
    is_roll_window: np.ndarray | None = None,
    start_ms: int = 1_600_000_000_000,
) -> Bars:
    # produce exactly `logret.size` bars: bar 0 is the start price, bar t applies cumulative returns
    # up to t-1 (so the regime's length-n return/sigma/flag arrays all align to n output bars).
    n = logret.size
    cum = np.concatenate([[0.0], np.cumsum(logret[:-1])])
    close = start_price * np.exp(cum)
    sb = sigma_bar if sigma_bar.size == n else np.resize(sigma_bar, n)
    o, h, low, c = _ohlc_from_path(rng, close, sb)
    ts = start_ms + np.arange(n, dtype=np.int64) * MINUTE_MS
    if volume is None:
        volume = base_volume * (1.0 + 0.3 * rng.standard_normal(n)).clip(0.1)
    return Bars(
        ts=ts,
        o=o,
        h=h,
        l=low,
        c=c,
        v=volume,
        is_valid=is_valid,
        is_session_open=is_session_open,
        is_roll_window=is_roll_window,
    )


def _clustered_vol(rng: np.random.Generator, n: int, base: float, persistence: float = 0.94) -> np.ndarray:
    """GARCH-ish clustered per-bar sigma (multiplicative)."""
    logv = np.zeros(n)
    shock = rng.standard_normal(n) * 0.15
    for t in range(1, n):
        logv[t] = persistence * logv[t - 1] + shock[t]
    return base * np.exp(logv - 0.5 * np.var(logv))


# --------------------------------------------------------------------------- the 13 regimes


def regime_null_random_walk(n: int = 600, seed: int = 0) -> Bars:
    """1. Null random walk with volatility clustering — no breakout should score high."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0006)
    r = rng.standard_normal(n) * sb
    return _assemble(rng, r, sigma_bar=sb)


def regime_drift_breakout(n: int = 600, seed: int = 1) -> Bars:
    """2. Drift breakout after compression — the canonical compression->expansion case."""
    rng = np.random.default_rng(seed)
    sb = np.full(n, 0.0006)
    cut = n // 2
    sb[cut - 120 : cut] = 0.00018  # squeeze: vol contracts before the break
    r = rng.standard_normal(n) * sb
    r[cut:] += 0.0011  # persistent upward drift expansion after the break
    return _assemble(rng, r, sigma_bar=sb)


def regime_variance_breakout(n: int = 600, seed: int = 2) -> Bars:
    """3. Variance breakout without drift — variance changes, mean does not."""
    rng = np.random.default_rng(seed)
    sb = np.full(n, 0.0004)
    sb[n // 2 :] = 0.0016
    r = rng.standard_normal(n) * sb
    return _assemble(rng, r, sigma_bar=sb)


def regime_jump_spike_revert(n: int = 600, seed: int = 3) -> Bars:
    """4. Jump spike that mean-reverts — should score LOW / high false-break risk."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0005)
    r = rng.standard_normal(n) * sb
    j = n // 2
    r[j] += 0.012  # one-bar jump up
    r[j + 1] -= 0.011  # immediately reverts
    return _assemble(rng, r, sigma_bar=sb)


def regime_liquidity_sweep(n: int = 600, seed: int = 4) -> Bars:
    """5. Liquidity sweep false breakout — pokes past the high then reverses."""
    rng = np.random.default_rng(seed)
    sb = np.full(n, 0.0004)
    cut = n // 2
    sb[:cut] = 0.00015  # tight range builds an obvious barrier
    r = rng.standard_normal(n) * sb
    j = min(cut + 20, n - 7)
    r[j] += 0.006  # sweep above the range high
    r[j + 1 : j + 6] -= 0.0018  # then fails back into the range
    return _assemble(rng, r, sigma_bar=sb)


def regime_slow_trend(n: int = 600, seed: int = 5) -> Bars:
    """6. Slow trend without a clean barrier — gentle drift, no discrete level."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0006)
    r = rng.standard_normal(n) * sb + 0.00025
    return _assemble(rng, r, sigma_bar=sb)


def regime_gap_continuation(n: int = 600, seed: int = 6) -> Bars:
    """7. Gap open then continuation — session-open gap followed by trend in the gap direction."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0005)
    r = rng.standard_normal(n) * sb
    g = n // 2
    r[g] += 0.009  # gap
    r[g:] += 0.0006  # continuation
    return _assemble(rng, r, sigma_bar=sb, is_session_open=_session_open_marks(n, g))


def regime_gap_fade(n: int = 600, seed: int = 7) -> Bars:
    """8. Gap open then fade — session-open gap that reverses (a false breakout)."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0005)
    r = rng.standard_normal(n) * sb
    g = n // 2
    r[g] += 0.009  # gap up
    r[g + 1 : g + 30] -= 0.0007  # fades back
    return _assemble(rng, r, sigma_bar=sb, is_session_open=_session_open_marks(n, g))


def regime_session_ushape(n: int = 600, seed: int = 8) -> Bars:
    """9. Session U-shaped volume/volatility — open/close busy, midday quiet."""
    rng = np.random.default_rng(seed)
    phase = np.linspace(0, np.pi, n)
    u = 0.4 + 0.6 * (np.cos(phase) ** 2)  # high at ends, low in middle
    sb = 0.0006 * u
    r = rng.standard_normal(n) * sb
    vol = 1500.0 * u * (1.0 + 0.2 * rng.standard_normal(n)).clip(0.1)
    return _assemble(rng, r, sigma_bar=sb, volume=vol, is_session_open=_session_open_marks(n, 0))


def regime_crypto_weekend(n: int = 600, seed: int = 9) -> Bars:
    """10. Crypto 24/7 weekend liquidity regime — thin weekend, no session boundary."""
    rng = np.random.default_rng(seed)
    weekend = np.zeros(n, dtype=bool)
    weekend[n // 3 : n // 3 + 120] = True
    sb = np.where(weekend, 0.0009, 0.0005)  # weekend: thinner, jumpier
    r = rng.standard_normal(n) * sb
    vol = np.where(weekend, 400.0, 1200.0) * (1.0 + 0.3 * rng.standard_normal(n)).clip(0.1)
    return _assemble(rng, r, sigma_bar=sb, volume=vol)


def regime_futures_roll(n: int = 600, seed: int = 10) -> Bars:
    """11. Futures roll artifact — a discontinuous price step at the roll bar (pre-adjustment)."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0005)
    r = rng.standard_normal(n) * sb
    roll = n // 2
    r[roll] += 0.02  # spurious roll step — must NOT fabricate a breakout after adjustment
    rollwin = np.zeros(n, dtype=bool)
    rollwin[roll - 2 : roll + 3] = True
    return _assemble(rng, r, sigma_bar=sb, is_roll_window=rollwin)


def regime_equity_split(n: int = 600, seed: int = 11) -> Bars:
    """12. Equity split / corporate-action artifact — a 2:1 price halving step (pre-adjustment)."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0005)
    r = rng.standard_normal(n) * sb
    split = n // 2
    r[split] += np.log(0.5)  # 2:1 split halves the price — corporate_actions must adjust it out
    return _assemble(rng, r, sigma_bar=sb)


def regime_missing_stale(n: int = 600, seed: int = 12) -> Bars:
    """13. Missing bars and stale bars — invalid/duplicate price runs the quality layer must flag."""
    rng = np.random.default_rng(seed)
    sb = _clustered_vol(rng, n, base=0.0006)
    r = rng.standard_normal(n) * sb
    bars = _assemble(rng, r, sigma_bar=sb)
    c = bars.c.copy()
    o = bars.o.copy()
    h = bars.h.copy()
    low = bars.l.copy()
    # a run of stale (repeated) prices around one third of the way in
    stale_a = n // 3
    stale_b = min(stale_a + 10, n)
    c[stale_a:stale_b] = c[stale_a]
    o[stale_a:stale_b] = c[stale_a]
    h[stale_a:stale_b] = c[stale_a]
    low[stale_a:stale_b] = c[stale_a]
    is_valid = np.ones(len(bars), dtype=bool)
    # a block of missing bars: introduce a timestamp gap from miss_b (two thirds in) onward
    miss_b = min(2 * n // 3, n - 1)
    ts = bars.ts.copy()
    ts[miss_b:] += 5 * MINUTE_MS  # introduce a 5-minute gap (missing bars)
    return Bars(ts=ts, o=o, h=h, l=low, c=c, v=bars.v, is_valid=is_valid)


def _session_open_marks(n: int, *opens: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[0] = True
    for o in opens:
        if 0 <= o < n:
            m[o] = True
    return m


REGIMES: dict[str, object] = {
    "null_random_walk": regime_null_random_walk,
    "drift_breakout": regime_drift_breakout,
    "variance_breakout": regime_variance_breakout,
    "jump_spike_revert": regime_jump_spike_revert,
    "liquidity_sweep": regime_liquidity_sweep,
    "slow_trend": regime_slow_trend,
    "gap_continuation": regime_gap_continuation,
    "gap_fade": regime_gap_fade,
    "session_ushape": regime_session_ushape,
    "crypto_weekend": regime_crypto_weekend,
    "futures_roll": regime_futures_roll,
    "equity_split": regime_equity_split,
    "missing_stale": regime_missing_stale,
}


def make_regime(name: str, n: int = 600, seed: int = 0) -> Bars:
    """Dispatch one of the thirteen named regimes (BREAKOUT §13)."""
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; known: {sorted(REGIMES)}")
    return REGIMES[name](n=n, seed=seed)  # type: ignore[operator]


def noise_universe(n_instruments: int, *, n: int = 400, seed: int = 0) -> list[Bars]:
    """A universe of pure-noise instruments for the cross-sectional FDR test (Invariant L)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_instruments)
    return [regime_null_random_walk(n=n, seed=int(s)) for s in seeds]
