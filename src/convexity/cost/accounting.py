"""Honest net weekly return: price P&L + SIGNED funding cash flow - turnover friction.

For each rebalance we hold the target weights over the next ``horizon`` days (weekly = 7). The net
return of the book over that week is:

    net = sum_i  w_i * asset_return_i(week)            # price P&L, dollar-neutral
        + sum_i  funding_cashflow_i(week)              # SIGNED per-leg funding accrual
        - turnover * round_trip_friction               # cost of getting INTO this book

FUNDING SIGN (COST §5.5). A perp LONG PAYS funding when funding is positive (cost), RECEIVES it when
funding is negative (credit); a SHORT is the mirror. With signed weight ``w_i`` (positive = long) and
a funding rate ``f`` the cash flow to the position over an interval is ``-w_i * f`` (long+positive
funding -> negative cash flow = pays; short (w_i<0) + positive funding -> positive cash flow =
credited). Summed over the week's daily funding accruals. So a SHORT on a high-positive-funding name
is CREDITED and a LONG on a negative-funding name is CREDITED — exactly the carry the signal targets.

TURNOVER + FRICTION. Turnover = 0.5 * sum_i |w_new_i - w_carried_i|, where ``w_carried`` is the PRIOR
book's weights DRIFTED by their realized returns over the just-elapsed week (a name that ran up carries
a larger weight into the rebalance, so even holding the same names incurs drift turnover). Round-trip
friction is charged per unit turnover: (taker + half_spread + impact) * 2 legs. The taker rate is the
Binance USDT-M VIP0 taker from the cost pillar's ONE fee truth (``BINANCE_USDM_FEE_SCHEDULE``); the
half-spread and impact are the conservative research fallbacks (crypto-major ~1bp half-spread, ~0.5bp
impact at low size), matching ``research.config``.

This is deliberately honest about turnover: names entering/leaving the legs AND weight drift both cost.
At weekly cadence the friction is paid ~52x/yr, not thousands of times — the whole point of the
low-turnover premise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.cost.config.fee_schedule import BINANCE_USDM_FEE_SCHEDULE

# ONE fee truth: Binance USDT-M VIP0 taker rate (fraction of notional) from the cost pillar.
BINANCE_VIP0_TAKER: float = BINANCE_USDM_FEE_SCHEDULE.tier_for_volume(0.0).taker_rate  # 0.0005

# conservative research fallbacks (match research.config), as FRACTIONS of notional.
HALF_SPREAD_FRAC: float = 1.0e-4  # 1.0 bp
IMPACT_FRAC: float = 0.5e-4  # 0.5 bp


def round_trip_friction(
    *,
    taker: float = BINANCE_VIP0_TAKER,
    half_spread: float = HALF_SPREAD_FRAC,
    impact: float = IMPACT_FRAC,
) -> float:
    """Round-trip friction per unit turnover: (taker + half_spread + impact) charged on BOTH legs.

    Turnover already counts one side of the trade (0.5 * sum|dw|), and a round trip is enter + exit, so
    the per-unit-turnover cost is 2 * (taker + half_spread + impact). Default 2*(5+1+0.5)bp = 13.0 bp,
    the same round-trip friction the 1-min harness charged — the difference here is that weekly cadence
    pays it ~52x/yr instead of thousands of times.
    """
    return 2.0 * (taker + half_spread + impact)


@dataclass(frozen=True)
class WeekResult:
    """One rebalance week's decomposed outcome (all in return units on the reference gross)."""

    gross_price: float  # price P&L before funding/cost
    funding: float  # signed funding cash flow (positive = net credit)
    turnover: float  # 0.5 * sum|w_new - w_carried|
    cost: float  # turnover * round_trip_friction (>= 0)
    net: float  # gross_price + funding - cost


def _asset_week_return(close: np.ndarray, listed: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Per-symbol simple return from day ``d0`` close to day ``d1`` close, NaN if not live across both.

    A name that delists mid-week (not live at d1) returns NaN so the caller can neutralize its weight
    (close the position at its last live day — no holding into the delist gap).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(listed[d0] & listed[d1] & (close[d0] > 0), close[d1] / close[d0] - 1.0, np.nan)
    return r


def _week_funding(funding: np.ndarray, listed: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Per-symbol total funding rate accrued over ``(d0, d1]`` (sum of daily funding), 0 where not live.

    Funding from the day AFTER entry through the exit day inclusive — the position is on the book across
    those days. Days a name is not live contribute 0 (it is not held then).
    """
    block = funding[d0 + 1 : d1 + 1]
    mask = listed[d0 + 1 : d1 + 1]
    with np.errstate(invalid="ignore"):
        acc = np.where(mask, np.nan_to_num(block, nan=0.0), 0.0).sum(axis=0)
    return acc


def settle_week(
    close: np.ndarray,
    funding: np.ndarray,
    listed: np.ndarray,
    *,
    d0: int,
    d1: int,
    w_target: np.ndarray,
    w_prev: np.ndarray,
    friction: float | np.ndarray,
) -> tuple[WeekResult, np.ndarray]:
    """Settle one week: apply ``w_target`` from day ``d0`` to ``d1``, return the decomposed outcome.

    ``w_prev`` is the PRIOR week's target weights (as set at the prior rebalance); we first DRIFT them
    by the just-elapsed week's returns to get the weights actually CARRIED into ``d0``, then turnover is
    ``0.5 * sum|w_target - w_carried|``. A name that delisted over the elapsed week drifts to 0 (closed
    at its last live day). Price P&L and funding use ``w_target`` over the coming week.

    ``friction`` is the round-trip friction per unit turnover: a SCALAR (flat cost, the default) OR a
    per-symbol array (LIQUIDITY-SCALED cost — a wider spread for lower-volume names). The scalar path is
    ``cost = turnover * friction``; the per-symbol path charges each name its own friction on its own
    weight change, ``cost = 0.5 * sum_i |dw_i| * friction_i`` (which reduces EXACTLY to the scalar form
    when friction is constant), so an illiquid name that turns over costs proportionally more.

    Returns ``(WeekResult, w_end)`` where ``w_end`` = the target weights DRIFTED by this week's returns
    (the carried weights for the NEXT rebalance).
    """
    # --- carried weights: prior target drifted by the elapsed week's returns ---
    r_elapsed = _asset_week_return(close, listed, d0 - (d1 - d0), d0) if d0 - (d1 - d0) >= 0 else None
    if r_elapsed is None:
        w_carried = np.zeros_like(w_target)
    else:
        drift = np.where(np.isfinite(r_elapsed) & (w_prev != 0.0), 1.0 + r_elapsed, 0.0)
        w_drifted = w_prev * drift
        # a name that delisted (NaN return) has drift 0 -> weight 0 (position closed).
        w_carried = w_drifted

    dw = np.abs(w_target - w_carried)
    turnover = 0.5 * float(np.sum(dw))
    # scalar friction -> cost = turnover*friction; per-symbol friction -> each name charged its own rate.
    if np.isscalar(friction):
        cost = turnover * float(friction)
    else:
        cost = 0.5 * float(np.sum(dw * np.asarray(friction, dtype=float)))

    # --- coming week: price P&L + signed funding on the TARGET book ---
    r = _asset_week_return(close, listed, d0, d1)
    active = np.isfinite(r) & (w_target != 0.0)
    gross_price = float(np.sum(np.where(active, w_target * r, 0.0)))

    f_acc = _week_funding(funding, listed, d0, d1)
    # cash flow = -w * f  (long+positive funding pays; short+positive funding is credited).
    funding_cf = float(np.sum(np.where(w_target != 0.0, -w_target * f_acc, 0.0)))

    net = gross_price + funding_cf - cost

    # weights drifted forward for next rebalance's turnover.
    w_end = np.where(active, w_target * (1.0 + r), 0.0)

    return WeekResult(gross_price, funding_cf, turnover, cost, net), w_end
