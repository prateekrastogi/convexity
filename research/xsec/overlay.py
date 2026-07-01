"""Deployable-risk overlays for a cross-sectional book: BTC-beta hedge + vol-target / drawdown-stop.

The raw carry book (SHORT high-funding, LONG low-funding names) harvests +12.9%/yr of funding income
but carries a 74.9% maxDD and fails DSR: it is dollar-neutral by construction yet NOT market-neutral —
its residual net BTC-beta drifts (the high-funding names it shorts are the high-beta momentum names in a
bull, so a rally gores the book). Two overlays, both computed with NO lookahead, try to turn that funding
harvest into a deployable positive-Sharpe / maxDD<25% strategy:

  * BETA HEDGE (``beta_hedge_returns``) — each week estimate every held name's BTC-beta over a trailing
    daily-return window (data strictly BEFORE the hold week), sum the book's signed dollar-beta, and add
    an offsetting BTC futures position. The hedge return each week is ``-book_beta * btc_week_return``;
    the hedge leg also accrues its own signed funding (a short-BTC hedge is credited BTC funding). This
    neutralizes the residual market exposure the dollar-neutral construction leaves behind.

  * VOL-TARGET (``vol_target_scale``) — scale the whole book each week by ``target_vol / trailing_vol``
    (trailing realized vol of the book's OWN net weekly returns, strictly prior weeks, capped at a max
    leverage). A carry book's vol spikes in a crisis; targeting a constant vol de-levers into the storm.

  * DRAWDOWN STOP (``drawdown_stop_scale``) — a state machine on the running equity: when drawdown from
    the running peak breaches ``dd_trigger`` the book is cut to ``floor`` exposure; it re-arms to full
    only after equity recovers to within ``dd_reset`` of the peak. Purely causal (uses only realized
    equity up to the prior week), so it is a legitimate deployable rule, not a hindsight stop.

All three return a MULTIPLIER or an ADDITIVE weekly-return series applied to the raw net series; they are
composable (hedge first, then scale the hedged series). Sharpe is leverage-invariant to a constant
multiplier but the vol-target / DD-stop multipliers are TIME-VARYING, so they DO change Sharpe (that is
the point — they suppress the fat left tail).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.xsec.engine import XSecBook
from research.xsec.metrics import PERIODS_PER_YEAR
from research.xsec.panel import Panel


def trailing_beta(
    ret: np.ndarray, mkt: np.ndarray, d0: int, window: int
) -> np.ndarray:
    """Per-symbol trailing OLS beta to the market over daily returns in ``[d0 - window, d0)`` (no lookahead).

    ``ret`` (n_days, n_sym) and ``mkt`` (n_days,) are daily simple returns aligned on the panel day index.
    Beta_i = cov(ret_i, mkt) / var(mkt) over the strictly-prior window. NaN where the symbol lacks a full
    finite window; the caller treats NaN beta as 0 (unhedgeable -> left in the residual).
    """
    n_sym = ret.shape[1]
    beta = np.full(n_sym, np.nan)
    lo = d0 - window
    if lo < 0:
        return beta
    m = mkt[lo:d0]
    finite_m = np.isfinite(m)
    if finite_m.sum() < 2:
        return beta
    mc = m - np.nanmean(m[finite_m])
    var_m = float(np.nansum(mc[finite_m] ** 2))
    if var_m <= 0:
        return beta
    block = ret[lo:d0]  # (window, n_sym)
    for j in range(n_sym):
        rj = block[:, j]
        ok = finite_m & np.isfinite(rj)
        if ok.sum() < 2:
            continue
        rjc = rj[ok] - np.mean(rj[ok])
        beta[j] = float(np.sum(rjc * mc[ok]) / var_m)
    return beta


@dataclass(frozen=True)
class BetaHedge:
    """The per-week BTC-beta hedge applied to a book's raw net series."""

    hedged_net: np.ndarray  # raw net + hedge price P&L + hedge funding
    book_beta: np.ndarray  # (n_weeks,) pre-hedge net dollar-beta of the book to BTC
    hedge_ret: np.ndarray  # (n_weeks,) hedge leg return (price + funding)
    hedge_funding: np.ndarray  # (n_weeks,) hedge leg funding cash flow


def _daily_returns(panel: Panel) -> np.ndarray:
    """(n_days, n_sym) daily simple returns of the panel close (NaN off-span / at the first day)."""
    c = panel.close
    out = np.full_like(c, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.where(
            panel.listed[1:] & panel.listed[:-1] & (c[:-1] > 0), c[1:] / c[:-1] - 1.0, np.nan
        )
    return out


def beta_hedge_returns(
    panel: Panel,
    book: XSecBook,
    raw_net: np.ndarray,
    *,
    market_symbol: str = "BTCUSDT",
    window: int = 28,
) -> BetaHedge:
    """Neutralize each week's net BTC-beta by adding an offsetting BTC futures leg (no lookahead).

    For hold week ``t`` starting at panel row ``d`` we estimate each held name's BTC-beta over the
    ``window`` daily returns STRICTLY BEFORE ``d``, sum the book's signed dollar-beta
    ``B_t = sum_i w_{t,i} * beta_i`` (NaN beta -> 0, left unhedged), and add a BTC position of ``-B_t``.
    That leg earns ``-B_t * btc_week_return`` in price P&L and ``-(-B_t) * btc_week_funding`` in funding
    (a short-BTC hedge, B_t>0, is credited positive BTC funding). The hedged net = raw net + hedge P&L +
    hedge funding. Beta uses only prior daily data; the hedge is set at rebalance and held the week.
    """
    daily = _daily_returns(panel)
    mkt_col = panel.symbols.index(market_symbol)
    mkt_daily = daily[:, mkt_col]
    n_wk = len(book)
    book_beta = np.zeros(n_wk)
    hedge_ret = np.zeros(n_wk)
    hedge_funding = np.zeros(n_wk)
    for t in range(n_wk):
        d0 = int(book.d_index[t])
        beta = trailing_beta(daily, mkt_daily, d0, window)
        w = book.weights[t]
        b = np.where(np.isfinite(beta), beta, 0.0)
        book_beta[t] = float(np.sum(w * b))
        # offsetting BTC position of -book_beta; its price return over the hold week:
        hedge_price = -book_beta[t] * book.mkt_ret[t]
        # funding on the hedge leg: cash flow = -w_hedge * f, with w_hedge = -book_beta.
        f_btc = float(book.asset_funding[t, mkt_col])
        hf = -(-book_beta[t]) * f_btc
        hedge_funding[t] = hf
        hedge_ret[t] = hedge_price + hf
    return BetaHedge(
        hedged_net=raw_net + hedge_ret,
        book_beta=book_beta,
        hedge_ret=hedge_ret,
        hedge_funding=hedge_funding,
    )


def vol_target_scale(
    net: np.ndarray, *, target_vol_ann: float, window: int = 8, max_leverage: float = 3.0
) -> np.ndarray:
    """Per-week exposure multiplier targeting a constant annualized vol (no lookahead).

    Scale week ``t`` by ``target_vol_ann / (trailing_vol of weeks [t-window, t) annualized)``, capped at
    ``max_leverage`` and floored at 0. The first ``window`` weeks (no trailing estimate) get scale 1.0.
    Uses only realized net returns STRICTLY BEFORE week ``t`` — a legitimate deployable rule.
    """
    n = net.size
    scale = np.ones(n)
    for t in range(n):
        lo = t - window
        if lo < 1:
            continue
        w = net[lo:t]
        sd = float(np.std(w, ddof=1)) if w.size > 1 else 0.0
        vol_ann = sd * np.sqrt(PERIODS_PER_YEAR)
        if vol_ann <= 0:
            continue
        scale[t] = min(max_leverage, target_vol_ann / vol_ann)
    return scale


def drawdown_stop_scale(
    net: np.ndarray, *, dd_trigger: float = 0.15, dd_reset: float = 0.05, floor: float = 0.0
) -> np.ndarray:
    """Causal drawdown-stop multiplier: cut to ``floor`` when DD breaches ``dd_trigger``, re-arm near peak.

    A state machine on the running equity of the RAW ``net`` series, evaluated with only weeks strictly
    before ``t`` known. When the running drawdown from the peak exceeds ``dd_trigger`` (e.g. 15%), the
    book scales to ``floor`` (0 = flat) starting the NEXT week; it re-arms to 1.0 once equity recovers to
    within ``dd_reset`` of the running peak. ``dd_trigger``/``dd_reset`` are positive fractions.
    """
    n = net.size
    scale = np.ones(n)
    equity = 1.0
    peak = 1.0
    armed = True  # True = full exposure
    for t in range(n):
        # decide THIS week's scale from state known before t, then realize the week.
        scale[t] = 1.0 if armed else floor
        # realize week t on the ACTUAL (scaled) return to advance equity the strategy really rides.
        equity *= 1.0 + scale[t] * net[t]
        peak = max(peak, equity)
        dd = 0.0 if peak <= 0 else (peak - equity) / peak
        if armed and dd >= dd_trigger:
            armed = False
        elif (not armed) and dd <= dd_reset:
            armed = True
    return scale


def apply_scale(net: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a per-week exposure multiplier to a weekly net-return series."""
    return net * scale
