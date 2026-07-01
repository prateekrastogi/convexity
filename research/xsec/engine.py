"""The cross-sectional backtest driver: panel -> signals -> dollar-neutral book -> net weekly series.

Ties the layers together for ONE config. The rebalance grid is every ``rebalance_days`` (7 = weekly)
along the panel's day index; at each rebalance day ``d`` we:

  1. compute the signal RANK at ``d`` (strictly from rows <= d — no lookahead),
  2. form dollar-neutral quantile weights (top/bottom quantile, equal or vol-scaled),
  3. settle the week ``[d, d + rebalance_days]``: price P&L + signed funding - turnover cost,
  4. carry the drifted weights forward to the next rebalance.

The first rebalance only starts once ``d`` is large enough for the signal's trailing window (the panel
NaN mask + signal eligibility already guarantee this per-symbol; the engine additionally skips leading
rebalances where NO book can form). The output is the aligned per-week arrays (net, price, funding,
turnover, cost) + the rebalance dates, consumed by ``metrics`` and ``validate``.

REBALANCE ON FRIDAY CLOSE: the brief asks for a Friday-close weekly cadence. We snap the first
rebalance to the first Friday at/after the warmup, then step 7 days; every rebalance day is a Friday
(the daily close of a Friday = ~Fri 23:59 UTC), so signals/holds are Fri-close to Fri-close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from research.xsec import signals as sig
from research.xsec.accounting import (
    WeekResult,
    _asset_week_return,
    _week_funding,
    round_trip_friction,
    settle_week,
)
from research.xsec.panel import Panel
from research.xsec.portfolio import Quantile, Weighting, quantile_weights, realized_vol

SignalKind = Literal["momentum", "carry", "combined"]


@dataclass(frozen=True)
class XSecConfig:
    """One cross-sectional backtest config (a single DSR 'trial')."""

    signal: SignalKind = "momentum"
    lookback: int = 28  # momentum lookback in days (ignored for pure carry)
    skip: int = 0  # momentum skip-most-recent in days
    carry_lookback: int = 7  # funding-carry trailing mean window in days
    quantile: Quantile = "quintile"
    weighting: Weighting = "equal"
    rebalance_days: int = 7  # weekly
    gross: float = 1.0  # reference gross exposure (100% = 50% long + 50% short)
    vol_window: int = 28  # trailing realized-vol window for vol-scaled weighting

    # friction (fractions of notional per leg); defaults match research.config / cost pillar.
    taker: float = 0.0005
    half_spread: float = 1.0e-4
    impact: float = 0.5e-4

    @property
    def label(self) -> str:
        if self.signal == "carry":
            base = f"carry-{self.carry_lookback}d"
        elif self.signal == "momentum":
            base = f"mom-{self.lookback}d-skip{self.skip}"
        else:
            base = f"combo-{self.lookback}d-skip{self.skip}-c{self.carry_lookback}"
        return f"{base}/{self.quantile}/{self.weighting}"


@dataclass(frozen=True)
class XSecResult:
    """Aligned per-week outputs of one backtest run."""

    dates: pd.DatetimeIndex  # rebalance (entry) date of each week
    net: np.ndarray
    gross_price: np.ndarray
    funding: np.ndarray
    turnover: np.ndarray
    cost: np.ndarray
    universe_size: np.ndarray  # eligible-symbol count at each rebalance

    def __len__(self) -> int:
        return int(self.net.shape[0])


@dataclass(frozen=True)
class XSecBook:
    """Per-week weight/return detail for one run — the raw material for overlays + attribution.

    All arrays are aligned to ``dates`` (one row per hold week). ``weights`` is the TARGET book set at
    each rebalance (n_weeks, n_symbols); ``asset_ret`` is each name's realized simple return over that
    hold week (NaN if not live across both endpoints); ``asset_funding`` is each name's summed funding
    over the week. ``contrib`` = per-name net price+funding contribution to that week's book return (the
    attribution atom). ``mkt_ret`` is the market (BTC) weekly return for beta-hedging.
    """

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    d_index: np.ndarray  # panel row index of each rebalance day
    weights: np.ndarray  # (n_weeks, n_symbols) target weights
    asset_ret: np.ndarray  # (n_weeks, n_symbols) week price return per name
    asset_funding: np.ndarray  # (n_weeks, n_symbols) week summed funding per name
    contrib: np.ndarray  # (n_weeks, n_symbols) per-name price+funding contribution
    mkt_ret: np.ndarray  # (n_weeks,) market weekly return (BTC)

    def __len__(self) -> int:
        return int(self.weights.shape[0])


def _rank_at(panel: Panel, d: int, cfg: XSecConfig) -> np.ndarray:
    if cfg.signal == "momentum":
        return sig.momentum_signal(panel, d, lookback=cfg.lookback, skip=cfg.skip)
    if cfg.signal == "carry":
        return sig.carry_signal(panel, d, lookback=cfg.carry_lookback)
    return sig.combined_signal(
        panel, d, lookback=cfg.lookback, skip=cfg.skip, carry_lookback=cfg.carry_lookback
    )


def _first_friday_index(panel: Panel, warmup: int) -> int:
    """First panel row index at/after ``warmup`` whose date is a Friday (weekday == 4)."""
    wd = panel.dates.weekday.to_numpy()
    for i in range(warmup, panel.n_days):
        if wd[i] == 4:
            return i
    return panel.n_days  # no valid start


def run(panel: Panel, cfg: XSecConfig) -> XSecResult:
    """Run the full cross-sectional backtest for ``cfg`` over ``panel``; return the weekly series."""
    step = cfg.rebalance_days
    # warmup: enough trailing history for the signal, +1 to allow a first drift week for turnover.
    if cfg.signal == "carry":
        warmup = cfg.carry_lookback + step
    else:
        warmup = cfg.lookback + cfg.skip + step
    warmup = max(warmup, cfg.vol_window + 1)

    friction = round_trip_friction(taker=cfg.taker, half_spread=cfg.half_spread, impact=cfg.impact)
    start = _first_friday_index(panel, warmup)

    dates: list[pd.Timestamp] = []
    net: list[float] = []
    gp: list[float] = []
    fnd: list[float] = []
    trn: list[float] = []
    cst: list[float] = []
    usz: list[int] = []

    w_prev = np.zeros(panel.n_symbols)
    d = start
    while d + step < panel.n_days:
        rank = _rank_at(panel, d, cfg)
        vol = realized_vol(panel, d, window=cfg.vol_window) if cfg.weighting == "volscaled" else None
        w_target = quantile_weights(
            rank, quantile=cfg.quantile, weighting=cfg.weighting, gross=cfg.gross, vol=vol
        )
        res: WeekResult
        res, w_end = settle_week(
            panel.close, panel.funding, panel.listed,
            d0=d, d1=d + step, w_target=w_target, w_prev=w_prev, friction=friction,
        )
        dates.append(panel.dates[d])
        net.append(res.net)
        gp.append(res.gross_price)
        fnd.append(res.funding)
        trn.append(res.turnover)
        cst.append(res.cost)
        usz.append(int(np.isfinite(rank).sum()))
        w_prev = w_end
        d += step

    return XSecResult(
        dates=pd.DatetimeIndex(dates),
        net=np.asarray(net),
        gross_price=np.asarray(gp),
        funding=np.asarray(fnd),
        turnover=np.asarray(trn),
        cost=np.asarray(cst),
        universe_size=np.asarray(usz, dtype=int),
    )


def run_book(
    panel: Panel, cfg: XSecConfig, *, market_symbol: str = "BTCUSDT"
) -> tuple[XSecResult, XSecBook]:
    """Like ``run`` but ALSO capture per-week weights / per-name returns / market return (for overlays).

    Re-runs the identical rebalance loop (same warmup, same Friday snap, same settle) and additionally
    records, per hold week: the target weight vector, each name's realized price return and summed
    funding, the per-name price+funding contribution to the book return, and the market (``market_symbol``,
    default BTC) weekly return used to beta-hedge. The aggregate ``XSecResult`` returned is byte-identical
    to ``run(panel, cfg)`` — this is a strict superset, so nothing about the headline series changes.
    """
    step = cfg.rebalance_days
    if cfg.signal == "carry":
        warmup = cfg.carry_lookback + step
    else:
        warmup = cfg.lookback + cfg.skip + step
    warmup = max(warmup, cfg.vol_window + 1)

    friction = round_trip_friction(taker=cfg.taker, half_spread=cfg.half_spread, impact=cfg.impact)
    start = _first_friday_index(panel, warmup)
    mkt_col = panel.symbols.index(market_symbol) if market_symbol in panel.symbols else -1

    dates: list[pd.Timestamp] = []
    net: list[float] = []
    gp: list[float] = []
    fnd: list[float] = []
    trn: list[float] = []
    cst: list[float] = []
    usz: list[int] = []
    d_idx: list[int] = []
    w_rows: list[np.ndarray] = []
    ar_rows: list[np.ndarray] = []
    af_rows: list[np.ndarray] = []
    contrib_rows: list[np.ndarray] = []
    mkt_rows: list[float] = []

    w_prev = np.zeros(panel.n_symbols)
    d = start
    while d + step < panel.n_days:
        rank = _rank_at(panel, d, cfg)
        vol = realized_vol(panel, d, window=cfg.vol_window) if cfg.weighting == "volscaled" else None
        w_target = quantile_weights(
            rank, quantile=cfg.quantile, weighting=cfg.weighting, gross=cfg.gross, vol=vol
        )
        res: WeekResult
        res, w_end = settle_week(
            panel.close, panel.funding, panel.listed,
            d0=d, d1=d + step, w_target=w_target, w_prev=w_prev, friction=friction,
        )
        # per-name detail (same window the settle used): price return + funding + contribution.
        r_wk = _asset_week_return(panel.close, panel.listed, d, d + step)
        f_wk = _week_funding(panel.funding, panel.listed, d, d + step)
        price_c = np.where(np.isfinite(r_wk) & (w_target != 0.0), w_target * r_wk, 0.0)
        fund_c = np.where(w_target != 0.0, -w_target * f_wk, 0.0)
        contrib = price_c + fund_c
        mkt_r = float(r_wk[mkt_col]) if mkt_col >= 0 and np.isfinite(r_wk[mkt_col]) else 0.0

        dates.append(panel.dates[d])
        net.append(res.net)
        gp.append(res.gross_price)
        fnd.append(res.funding)
        trn.append(res.turnover)
        cst.append(res.cost)
        usz.append(int(np.isfinite(rank).sum()))
        d_idx.append(d)
        w_rows.append(w_target.copy())
        ar_rows.append(np.where(np.isfinite(r_wk), r_wk, np.nan))
        af_rows.append(f_wk)
        contrib_rows.append(contrib)
        mkt_rows.append(mkt_r)
        w_prev = w_end
        d += step

    dt = pd.DatetimeIndex(dates)
    result = XSecResult(
        dates=dt,
        net=np.asarray(net),
        gross_price=np.asarray(gp),
        funding=np.asarray(fnd),
        turnover=np.asarray(trn),
        cost=np.asarray(cst),
        universe_size=np.asarray(usz, dtype=int),
    )
    book = XSecBook(
        dates=dt,
        symbols=panel.symbols,
        d_index=np.asarray(d_idx, dtype=int),
        weights=np.asarray(w_rows) if w_rows else np.zeros((0, panel.n_symbols)),
        asset_ret=np.asarray(ar_rows) if ar_rows else np.zeros((0, panel.n_symbols)),
        asset_funding=np.asarray(af_rows) if af_rows else np.zeros((0, panel.n_symbols)),
        contrib=np.asarray(contrib_rows) if contrib_rows else np.zeros((0, panel.n_symbols)),
        mkt_ret=np.asarray(mkt_rows),
    )
    return result, book
