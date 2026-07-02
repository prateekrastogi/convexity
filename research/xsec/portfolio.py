"""Dollar-neutral quantile portfolio construction from a cross-sectional rank vector.

Given a rank vector (higher rank -> long conviction, NaN = ineligible) we form a dollar-neutral book:
LONG the top quantile, SHORT the bottom quantile, flat in between. Weights are normalized so the book
has a fixed reference GROSS exposure (100% gross = 50% long + 50% short); the long leg sums to +0.5,
the short leg to -0.5, net 0. Because the book is dollar-neutral and leverage-normalized, SHARPE is
leverage-invariant (scaling gross scales every weekly return identically, leaving Sharpe unchanged);
CAGR / maxDD scale with the chosen gross (the engine reports at 100% gross and notes the scaling).

Two within-leg weightings:
  * EQUAL — every name in a leg gets equal magnitude.
  * VOL-SCALED — inverse-realized-vol weighting within each leg (a low-vol name gets a larger weight),
    normalized to the same leg gross. This is the standard risk-parity-within-leg variant; it uses ONLY
    trailing realized vol (no lookahead).

Quantile granularity: ``quintile`` (top/bottom 20% each) or ``tercile`` (top/bottom 33%). At the full
~50-name universe a quintile is ~10 long / ~10 short; early, with a small universe, the quantile count
floors at 1 per leg so the book is always at least 1x1 (or flat if fewer than 2 eligible names).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from research.xsec.panel import Panel

Quantile = Literal["quintile", "tercile"]
# base engine weightings (equal, volscaled) + liquidity-aware within-leg schemes (research sweep).
Weighting = Literal["equal", "volscaled", "liqcap_volscaled", "liqtilt", "eqcap"]

_QFRAC: dict[str, float] = {"quintile": 0.2, "tercile": 1.0 / 3.0}


def _leg_count(n_eligible: int, quantile: Quantile) -> int:
    """Names per leg for ``n_eligible`` eligible symbols at the chosen quantile (>=1 if a book is possible).

    Floors at 1 so a small early universe still forms a 1x1 book; returns 0 only when fewer than 2
    names are eligible (no dollar-neutral book possible).
    """
    if n_eligible < 2:
        return 0
    k = int(np.floor(n_eligible * _QFRAC[quantile]))
    k = max(1, k)
    # never let the two legs overlap: cap so 2*k <= n_eligible.
    return min(k, n_eligible // 2)


def realized_vol(panel: Panel, d: int, *, window: int = 28) -> np.ndarray:
    """Trailing realized daily-return vol per symbol over ``[d - window, d]`` (no lookahead).

    Std of daily log-ish simple returns of the daily close over the trailing window. NaN where the
    symbol lacks a full window of live closes. Used only by the vol-scaled weighting.
    """
    n_sym = panel.n_symbols
    vol = np.full(n_sym, np.nan)
    start = d - window
    if start < 1:
        return vol
    block = panel.close[start : d + 1]  # (window+1, n_sym)
    live = panel.listed[start : d + 1].all(axis=0)
    if not live.any():
        return vol
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = block[1:, live] / block[:-1, live] - 1.0
        vol[live] = np.nanstd(rets, axis=0, ddof=1)
    return vol


def quantile_weights(
    rank: np.ndarray,
    *,
    quantile: Quantile,
    weighting: Weighting = "volscaled",
    gross: float = 1.0,
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """Dollar-neutral quantile weights from a rank vector. Long leg sums +gross/2, short leg -gross/2.

    ``rank`` — cross-sectional rank in [0, 1], NaN for ineligible symbols.
    ``weighting`` — within-leg scheme; the resolved default is ``'volscaled'`` (PORTFOLIO.md §5.4,
                Invariant L: the deployed within-leg scheme is inverse-vol risk-parity, never the
                liquidity-capped variant, absent an explicit logged AUM-threshold override).
    ``gross`` — total gross exposure (1.0 == 100% gross == 50% long + 50% short).
    ``vol``   — trailing realized vol per symbol (used for ``weighting='volscaled'``); a name with
                no finite vol falls back to the leg's mean inverse-vol (PORTFOLIO.md §11 E2).

    Returns a weight vector aligned to ``rank`` (0.0 for non-selected / ineligible names). If fewer
    than 2 names are eligible the book is all-flat (zeros).
    """
    n = rank.shape[0]
    w = np.zeros(n)
    elig = np.flatnonzero(np.isfinite(rank))
    m = elig.size
    k = _leg_count(m, quantile)
    if k == 0:
        return w
    order = elig[np.argsort(rank[elig], kind="mergesort")]  # ascending rank
    short_idx = order[:k]  # lowest ranks -> SHORT
    long_idx = order[-k:]  # highest ranks -> LONG
    half = gross / 2.0

    def _leg(idx: np.ndarray, target: float) -> None:
        if weighting == "volscaled" and vol is not None:
            iv = np.array([1.0 / vol[i] if np.isfinite(vol[i]) and vol[i] > 0 else np.nan for i in idx])
            if np.all(~np.isfinite(iv)):
                mag = np.full(idx.size, 1.0 / idx.size)
            else:
                # names with no vol fall back to the mean inverse-vol so they still participate.
                fill = np.nanmean(iv)
                iv = np.where(np.isfinite(iv), iv, fill)
                mag = iv / iv.sum()
        else:
            mag = np.full(idx.size, 1.0 / idx.size)
        w[idx] = target * mag

    _leg(long_idx, +half)
    _leg(short_idx, -half)
    return w


def _renorm_leg_to_target(mag: np.ndarray, target: float) -> np.ndarray:
    """Normalize a non-negative magnitude vector so it sums to ``|target|``, carrying ``target``'s sign.

    Empty / all-zero magnitude -> zeros (a degenerate leg contributes nothing). Used by the
    liquidity-aware within-leg weightings below, all of which build a per-name magnitude then rescale
    the leg back to its dollar-neutral target (+gross/2 long, -gross/2 short).
    """
    if mag.size == 0:
        return mag
    s = float(mag.sum())
    if s <= 0.0:
        return np.zeros_like(mag)
    return np.sign(target) * abs(target) * (mag / s)


def _leg_liq_cap(
    base_mag: np.ndarray, dvol_leg: np.ndarray, cap_mult: float
) -> np.ndarray:
    """Cap each name's within-leg magnitude at ``cap_mult * (its dollar-volume share of the leg)``.

    ``base_mag`` — non-negative starting magnitudes within ONE leg (e.g. inverse-vol), summing to any
    positive number. ``dvol_leg`` — point-in-time trailing dollar volume of the SAME names (>=0; a
    non-finite / non-positive entry is treated as 0 liquidity, so it gets the tightest cap). ``cap_mult``
    (C) — the tightness: a name may hold at most C times its liquidity share of the leg.

    Iterative water-filling: normalize base to leg-shares, clamp any name above its cap = C * liq_share,
    then redistribute the freed mass proportionally among the UNCAPPED names (respecting their own caps),
    repeating until no name exceeds its cap or all names are capped. C = inf (or a cap that binds on no
    name) returns the base magnitudes unchanged. This never over-allocates beyond C x a name's liquidity
    share and pushes mass OUT of low-vol majors INTO more names -> higher effective breadth.
    """
    m = base_mag.astype(float).copy()
    m[~np.isfinite(m)] = 0.0
    m[m < 0.0] = 0.0
    if m.sum() <= 0.0:
        return m
    if not np.isfinite(cap_mult):
        return m

    liq = np.where(np.isfinite(dvol_leg) & (dvol_leg > 0.0), dvol_leg, 0.0)
    liq_sum = float(liq.sum())
    if liq_sum <= 0.0:
        # no liquidity info for this leg -> fall back to the uncapped base magnitudes.
        return m
    liq_share = liq / liq_sum
    cap = cap_mult * liq_share  # per-name max share of the leg

    total = float(m.sum())
    w_share = m / total  # current leg-shares
    capped = np.zeros(m.shape, dtype=bool)
    # if the cap sums to < 1 the constraints are infeasible; the loop then caps every name (best effort).
    for _ in range(m.size + 1):
        over = (~capped) & (w_share > cap + 1e-15)
        if not over.any():
            break
        w_share[over] = cap[over]
        capped[over] = True
        free = 1.0 - float(w_share[capped].sum())  # mass to redistribute among uncapped
        open_mask = ~capped
        if free <= 0.0 or not open_mask.any():
            break
        base_open = m[open_mask]
        if float(base_open.sum()) <= 0.0:
            break
        w_share[open_mask] = free * (base_open / float(base_open.sum()))
    return w_share  # magnitudes as leg-shares (sum ~1 unless infeasible); renormalized by caller


def liquidity_weights(
    rank: np.ndarray,
    *,
    quantile: Quantile,
    scheme: Weighting,
    dvol: np.ndarray,
    gross: float = 1.0,
    vol: np.ndarray | None = None,
    cap_mult: float = float("inf"),
) -> np.ndarray:
    """Dollar-neutral quantile weights with a LIQUIDITY-aware within-leg magnitude (no lookahead).

    Same leg selection as ``quantile_weights`` (top/bottom quantile by ``rank``), but the within-leg
    magnitude depends on ``scheme`` and the point-in-time trailing dollar volume ``dvol`` (per symbol,
    as-of the rebalance; non-finite = ineligible-liquidity, treated as 0):

      * ``'liqcap_volscaled'`` — start from inverse-vol magnitudes, then CAP each name at
        ``cap_mult`` x its dollar-volume share of the leg and renormalize (water-fill). ``cap_mult=inf``
        reduces EXACTLY to plain vol-scaled. Requires ``vol``.
      * ``'liqtilt'`` — magnitude proportional to ``sqrt(dollar-volume)`` (capacity-aware tilt).
      * ``'eqcap'`` — equal-weight within leg, then capped by ``cap_mult`` x liquidity share (max-breadth
        reference). ``cap_mult=inf`` reduces to plain equal weight.

    Every leg is renormalized to its dollar-neutral target (+gross/2 long, -gross/2 short). A leg with no
    usable liquidity/vol info falls back to equal weight so the book still forms. Returns a weight vector
    aligned to ``rank`` (0.0 for non-selected / ineligible names).
    """
    n = rank.shape[0]
    w = np.zeros(n)
    elig = np.flatnonzero(np.isfinite(rank))
    m = elig.size
    k = _leg_count(m, quantile)
    if k == 0:
        return w
    order = elig[np.argsort(rank[elig], kind="mergesort")]  # ascending rank
    short_idx = order[:k]
    long_idx = order[-k:]
    half = gross / 2.0

    def _leg(idx: np.ndarray, target: float) -> None:
        dvol_leg = np.array([dvol[i] if np.isfinite(dvol[i]) else 0.0 for i in idx])
        if scheme == "liqtilt":
            mag = np.sqrt(np.clip(dvol_leg, 0.0, None))
            if float(mag.sum()) <= 0.0:  # no liquidity -> equal weight fallback
                mag = np.full(idx.size, 1.0)
        else:
            # base magnitude: inverse-vol (liqcap_volscaled) or equal (eqcap).
            if scheme == "liqcap_volscaled" and vol is not None:
                iv = np.array(
                    [1.0 / vol[i] if np.isfinite(vol[i]) and vol[i] > 0 else np.nan for i in idx]
                )
                if np.all(~np.isfinite(iv)):
                    base = np.full(idx.size, 1.0)
                else:
                    base = np.where(np.isfinite(iv), iv, np.nanmean(iv))
            else:  # 'eqcap' (or liqcap_volscaled with no vol) -> equal base
                base = np.full(idx.size, 1.0)
            mag = _leg_liq_cap(base, dvol_leg, cap_mult)
        w[idx] = _renorm_leg_to_target(mag, target)

    _leg(long_idx, +half)
    _leg(short_idx, -half)
    return w
