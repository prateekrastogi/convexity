"""Performance metrics on a WEEKLY net-return series (distinct from the per-trade ``metrics.perf``).

The scalp harness's ``research.metrics.perf`` scores per-TRADE outcomes in R units; a cross-sectional
book has no per-trade R — it has a regular WEEKLY return stream. So this module computes period-return
metrics on that stream: annualized Sharpe (52 periods/yr), CAGR (compounded), maxDD on the equity
curve, win-rate-of-weeks, turnover/yr, cost-drag/yr, funding-income/yr.

Annualization uses PERIODS_PER_YEAR = 365.25 / 7 ≈ 52.18 (weekly cadence). Sharpe here is the raw
annualized Sharpe for reporting; the DEFLATED Sharpe (over the config-trial count) is computed by
``research.xsec.validate`` reusing SYSTEM's ``DeflatedSharpe`` — that deflated number is the headline,
this raw one is diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERIODS_PER_YEAR: float = 365.25 / 7.0  # ~52.18 weekly periods per year


@dataclass(frozen=True)
class WeeklyMetrics:
    """Metric bundle for one config's weekly net-return series."""

    n_weeks: int
    mean_weekly: float
    sharpe_ann: float
    cagr_pct: float
    maxdd_pct: float
    win_rate_weeks: float
    vol_ann_pct: float
    turnover_per_year: float
    cost_drag_per_year: float  # annualized total cost as a return drag (positive number)
    funding_income_per_year: float  # annualized signed funding cash flow (positive = net credit)


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(-dd.min() * 100.0)


def compute(
    net_weekly: np.ndarray,
    *,
    turnover: np.ndarray | None = None,
    cost: np.ndarray | None = None,
    funding: np.ndarray | None = None,
) -> WeeklyMetrics:
    """Compute the weekly metric bundle from the net weekly-return series (+ optional decompositions).

    ``net_weekly`` — the per-week net returns (price + funding - cost). ``turnover``/``cost``/``funding``
    are the aligned per-week decompositions used only for the per-year attribution rows. Sharpe is
    annualized by sqrt(PERIODS_PER_YEAR); CAGR compounds the weekly returns over the realized span.
    """
    r = np.asarray(net_weekly, dtype=float)
    n = r.size
    if n == 0:
        return WeeklyMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_w = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    sharpe = (mean_w / sd) * np.sqrt(PERIODS_PER_YEAR) if sd > 0 else 0.0
    vol_ann = sd * np.sqrt(PERIODS_PER_YEAR) * 100.0

    equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    maxdd = _max_drawdown_pct(equity)
    span_years = n / PERIODS_PER_YEAR
    final = float(equity[-1])
    cagr = ((final ** (1.0 / span_years) - 1.0) * 100.0) if (span_years > 0 and final > 0) else -100.0

    win = float(np.mean(r > 0))

    turns_yr = float(np.mean(turnover) * PERIODS_PER_YEAR) if turnover is not None and turnover.size else 0.0
    cost_yr = float(np.mean(cost) * PERIODS_PER_YEAR) if cost is not None and cost.size else 0.0
    fund_yr = float(np.mean(funding) * PERIODS_PER_YEAR) if funding is not None and funding.size else 0.0

    return WeeklyMetrics(
        n_weeks=n,
        mean_weekly=mean_w,
        sharpe_ann=float(sharpe),
        cagr_pct=float(cagr),
        maxdd_pct=maxdd,
        win_rate_weeks=win,
        vol_ann_pct=float(vol_ann),
        turnover_per_year=turns_yr,
        cost_drag_per_year=cost_yr,
        funding_income_per_year=fund_yr,
    )
