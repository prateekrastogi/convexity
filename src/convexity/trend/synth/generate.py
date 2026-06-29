"""Adversarial synthetic trend paths (TREND §17).

The 17 scenarios the acceptance gates exercise. Every path returns a :class:`convexity.core.types.Bars`
with a fixed 1-minute clock and a paired ``entry_idx`` so the engine can be driven deterministically.
Paths are seeded for reproducibility. These are TEST fixtures (synth/, not validate/) — they carry no
future labels.

Required scenarios (§17):
  1  persistent trend, low noise
  2  persistent trend, vol expansion
  3  trend decays to flat
  4  trend reverses
  5  jump against position
  6  jump with position
  7  gap against position
  8  gap with position
  9  choppy mean-reverting
  10 blow-off top / capitulation
  11 futures roll artifact
  12 equity split artifact
  13 equity halt and reopen
  14 crypto weekend liquidity regime
  15 crypto funding shock
  16 missing and stale bars
  17 RTH/ETH session transition
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars

_MIN_MS = 60_000  # one-minute bars


def _ts(n: int, start_ms: int = 1_600_000_000_000) -> np.ndarray:
    return start_ms + np.arange(n, dtype=np.int64) * _MIN_MS


def _ohlc_from_close(c: np.ndarray, rng: np.random.Generator, wick: float = 0.0005) -> Bars:
    """Build a causal OHLCV Bars from a close path with small symmetric wicks."""
    n = c.shape[0]
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1]
    span = np.abs(c) * wick + 1e-9
    hi = np.maximum(o, c) + np.abs(rng.normal(0.0, 1.0, n)) * span
    lo = np.minimum(o, c) - np.abs(rng.normal(0.0, 1.0, n)) * span
    v = np.full(n, 1000.0)
    return Bars(
        ts=_ts(n),
        o=o,
        h=hi,
        l=lo,
        c=c,
        v=v,
        is_valid=np.ones(n, dtype=bool),
        is_session_open=np.zeros(n, dtype=bool),
        is_halt=np.zeros(n, dtype=bool),
        is_roll_window=np.zeros(n, dtype=bool),
    )


def _gbm(n: int, mu: float, sigma: float, p0: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric path: log-return drift ``mu`` per bar, vol ``sigma`` per bar."""
    r = rng.normal(mu, sigma, n)
    r[0] = 0.0
    return p0 * np.exp(np.cumsum(r))


# --------------------------------------------------------------------------- the 17 paths

def persistent_trend_low_noise(n: int = 400, seed: int = 1) -> tuple[Bars, int]:
    """1 — strong positive drift, low vol. Should be held long; convexity tail."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0012, sigma=0.0010, p0=100.0, rng=rng)
    return _ohlc_from_close(c, rng), 0


def persistent_trend_vol_expansion(n: int = 400, seed: int = 2) -> tuple[Bars, int]:
    """2 — positive drift with a widening vol regime. Trail must widen, not exit."""
    rng = np.random.default_rng(seed)
    sig = np.linspace(0.0008, 0.004, n)
    r = rng.normal(0.0012, 1.0, n) * sig
    r[0] = 0.0
    c = 100.0 * np.exp(np.cumsum(r))
    return _ohlc_from_close(c, rng), 0


def trend_decays_to_flat(n: int = 400, seed: int = 3) -> tuple[Bars, int]:
    """3 — drift decays to zero. Deterioration detector should exit with low giveback."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.0015, 0.0, n)
    r = rng.normal(0.0, 0.0010, n) + mu
    r[0] = 0.0
    c = 100.0 * np.exp(np.cumsum(r))
    return _ohlc_from_close(c, rng), 0


def trend_reverses(n: int = 400, seed: int = 4) -> tuple[Bars, int]:
    """4 — up then down. Must exit on the reversal, not ride it all the way back."""
    rng = np.random.default_rng(seed)
    half = n // 2
    mu = np.concatenate([np.full(half, 0.0015), np.full(n - half, -0.0015)])
    r = rng.normal(0.0, 0.0010, n) + mu
    r[0] = 0.0
    c = 100.0 * np.exp(np.cumsum(r))
    return _ohlc_from_close(c, rng), 0


def jump_against(n: int = 200, seed: int = 5, at: int = 120) -> tuple[Bars, int]:
    """5 — a discontinuous adverse jump for a long. Candidate hard exit (D5)."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 0.93  # -7% jump down against the long
    return _ohlc_from_close(c, rng), 0


def jump_with(n: int = 200, seed: int = 6, at: int = 120) -> tuple[Bars, int]:
    """6 — a discontinuous favorable jump for a long. Widens trail, does not exit."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 1.07  # +7% jump up with the long
    return _ohlc_from_close(c, rng), 0


def gap_against(n: int = 200, seed: int = 7, at: int = 120) -> tuple[Bars, int]:
    """7 — an adverse session-boundary gap. Gap survival must not pretend the stop held."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 0.95
    bars = _ohlc_from_close(c, rng)
    so = bars.is_session_open.copy()
    so[at] = True
    return Bars(**{**bars.__dict__, "is_session_open": so}), 0


def gap_with(n: int = 200, seed: int = 8, at: int = 120) -> tuple[Bars, int]:
    """8 — a favorable session-boundary gap with the long."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 1.05
    bars = _ohlc_from_close(c, rng)
    so = bars.is_session_open.copy()
    so[at] = True
    return Bars(**{**bars.__dict__, "is_session_open": so}), 0


def choppy_mean_reverting(n: int = 400, seed: int = 9) -> tuple[Bars, int]:
    """9 — zero-drift mean-reverting noise. No real trend; should die early, low giveback."""
    rng = np.random.default_rng(seed)
    c = np.empty(n)
    c[0] = 100.0
    for i in range(1, n):  # AR(1) mean reversion around 100
        c[i] = c[i - 1] + 0.3 * (100.0 - c[i - 1]) * 0.01 + rng.normal(0.0, 0.10)
    return _ohlc_from_close(c, rng), 0


def blowoff_top(n: int = 300, seed: int = 10, peak: int = 200) -> tuple[Bars, int]:
    """10 — parabolic blow-off then capitulation. Tail capture then fast cut on the break."""
    rng = np.random.default_rng(seed)
    up = np.linspace(0.0008, 0.006, peak)
    r_up = rng.normal(0.0, 0.0010, peak) + up
    down = np.full(n - peak, -0.004)
    r = np.concatenate([r_up, down]) + rng.normal(0.0, 0.0010, n)
    r[0] = 0.0
    c = 100.0 * np.exp(np.cumsum(r))
    return _ohlc_from_close(c, rng), 0


def futures_roll_artifact(n: int = 200, seed: int = 11, at: int = 120) -> tuple[Bars, int]:
    """11 — a stitch discontinuity inside a flagged roll window. Must NOT trigger false death."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] += 5.0  # contango stitch jump (price-level shift, not a return)
    bars = _ohlc_from_close(c, rng)
    rw = bars.is_roll_window.copy()
    rw[at - 1 : at + 2] = True
    return Bars(**{**bars.__dict__, "is_roll_window": rw}), 0


def equity_split_artifact(n: int = 200, seed: int = 12, at: int = 120) -> tuple[Bars, int]:
    """12 — a 2:1 split discontinuity. The data guard must flag it, not call it a jump-against."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 0.5  # 2:1 split halves the printed price
    return _ohlc_from_close(c, rng), 0


def equity_halt_reopen(n: int = 200, seed: int = 13, halt_at: int = 120, dur: int = 5) -> tuple[Bars, int]:
    """13 — a trading halt then reopen. Halt bars are untradeable, not trend death."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[halt_at + dur :] *= 0.97  # reopen gap
    bars = _ohlc_from_close(c, rng)
    halt = bars.is_halt.copy()
    halt[halt_at : halt_at + dur] = True
    valid = bars.is_valid.copy()
    valid[halt_at : halt_at + dur] = False
    return Bars(**{**bars.__dict__, "is_halt": halt, "is_valid": valid}), 0


def crypto_weekend_liquidity(n: int = 400, seed: int = 14) -> tuple[Bars, int]:
    """14 — thin weekend regime (low volume, wider noise). Trail must not whipsaw on noise."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0008, sigma=0.0012, p0=100.0, rng=rng)
    bars = _ohlc_from_close(c, rng)
    v = bars.v.copy()
    v[n // 2 :] *= 0.2  # weekend liquidity drop
    return Bars(**{**bars.__dict__, "v": v}), 0


def crypto_funding_shock(n: int = 400, seed: int = 15, at: int = 200) -> tuple[Bars, int]:
    """15 — a funding-rate spike (carry shock). Drives the COST hold-carry gate, not price death."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0008, sigma=0.0010, p0=100.0, rng=rng)
    bars = _ohlc_from_close(c, rng)
    fr = np.zeros(n)
    fr[at:] = 0.01  # 1% per interval funding shock
    return Bars(**{**bars.__dict__, "funding_rate": fr}), 0


def missing_and_stale_bars(n: int = 200, seed: int = 16) -> tuple[Bars, int]:
    """16 — repeated stale prints and a dropped-bar gap. Quality flags, not trend signals."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[80:85] = c[79]  # stale repeat
    bars = _ohlc_from_close(c, rng)
    ts = bars.ts.copy()
    ts[120:] += 10 * _MIN_MS  # a 10-bar gap (missing bars)
    return Bars(**{**bars.__dict__, "ts": ts}), 0


def rth_eth_transition(n: int = 400, seed: int = 17, session_len: int = 100) -> tuple[Bars, int]:
    """17 — RTH/ETH session boundaries every ``session_len`` bars, with overnight gaps."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=0.0010, sigma=0.0008, p0=100.0, rng=rng)
    bars = _ohlc_from_close(c, rng)
    so = bars.is_session_open.copy()
    so[::session_len] = True
    so[0] = False
    return Bars(**{**bars.__dict__, "is_session_open": so}), 0


def persistent_downtrend_low_noise(n: int = 400, seed: int = 21) -> tuple[Bars, int]:
    """SHORT-side mirror of scenario 1 — strong NEGATIVE drift, low vol. A falling price is the
    FAVORABLE direction for a short (side=-1): the low watermark should ratchet down, the trail
    should sit ABOVE the lwm and ratchet down, a discontinuous DOWN move is a jump_with (not against),
    and a falling Kalman slope must NOT trigger the side-adjusted reduce (§13a.3 D3). Held short to a
    convex tail, symmetric to ``persistent_trend_low_noise``."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=-0.0012, sigma=0.0010, p0=100.0, rng=rng)
    return _ohlc_from_close(c, rng), 0


def jump_with_short(n: int = 200, seed: int = 26, at: int = 120) -> tuple[Bars, int]:
    """SHORT-side mirror of scenario 6 — a discontinuous DOWN jump is FAVORABLE for a short
    (jump_with): it widens the trail and must not hard-exit (§9 / §10)."""
    rng = np.random.default_rng(seed)
    c = _gbm(n, mu=-0.0010, sigma=0.0008, p0=100.0, rng=rng)
    c[at:] *= 0.93  # -7% jump down: favorable for a short
    return _ohlc_from_close(c, rng), 0


# Registry so tests can enumerate all 17 (§17 acceptance).
ALL_PATHS = {
    "persistent_trend_low_noise": persistent_trend_low_noise,
    "persistent_trend_vol_expansion": persistent_trend_vol_expansion,
    "trend_decays_to_flat": trend_decays_to_flat,
    "trend_reverses": trend_reverses,
    "jump_against": jump_against,
    "jump_with": jump_with,
    "gap_against": gap_against,
    "gap_with": gap_with,
    "choppy_mean_reverting": choppy_mean_reverting,
    "blowoff_top": blowoff_top,
    "futures_roll_artifact": futures_roll_artifact,
    "equity_split_artifact": equity_split_artifact,
    "equity_halt_reopen": equity_halt_reopen,
    "crypto_weekend_liquidity": crypto_weekend_liquidity,
    "crypto_funding_shock": crypto_funding_shock,
    "missing_and_stale_bars": missing_and_stale_bars,
    "rth_eth_transition": rth_eth_transition,
}
