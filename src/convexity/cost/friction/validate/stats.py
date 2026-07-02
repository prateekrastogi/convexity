"""Block-bootstrap CIs on cost-drag/net-Sharpe + win-rate-by-funding-sign (COST.md Section 6.4).

Firewalled validation companion: nothing here feeds a live rebalance. It delegates the moving-block
bootstrap to ``src/convexity/system/validate/robustness.py::block_bootstrap_ci`` (the ONE bootstrap
truth, Kunsch 1989 / Politis-White 2004 block-length rule) — it defines NO own bootstrap. The
statistics computed:

  * ``net_sharpe_ci`` / ``cost_drag_ci`` — moving-block CIs on the deployable net Sharpe and the
    annualized cost drag, so the honest Amihud-cost number (Invariant HONEST-SR) is reported with a
    dependence-preserving interval, never a naive-iid one (FORBIDDEN for dependent weekly returns).
  * ``win_rate_by_funding_sign`` — the Section 6.4 diagnostic: weekly win rate CONDITIONAL on the
    sign of that week's ``funding_t``, alongside the unconditional ~0.60. A future divergence between
    the two conditional rates (which would mean funding drives the SIGN of weekly P&L more than its
    ~3-5%-of-gross magnitude suggests, undercutting P4) becomes visible instead of silently absorbed
    into one blended number.

Annualization of Sharpe matches ``research/xsec/metrics.py::PERIODS_PER_YEAR`` (weekly) everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.system.validate.robustness import block_bootstrap_ci

from convexity.system.metrics import PERIODS_PER_YEAR, compute

# Default moving-block bootstrap knobs (COST.md Section 6.4 / SYSTEM Section 8.1). ``block_length``
# >= the weekly vol-clustering scale. 6 weeks is SYSTEM Section 7.4's retained committed default:
# the Politis & White (2004) automatic block-length optimum b* ~ c*n^(1/3) evaluates to 258^(1/3)
# ~= 6.36 ~= 6 at the pinned n=258 weekly sample (c ~= 1), reclassifying the empirical "block=6" from
# ad hoc to Politis-White-consistent. (This is SYSTEM's committed *bootstrap-length* default of
# Section 7.4, NOT the separate ``bootstrap_block_length`` config knob in
# src/convexity/system/config/defaults.py, which is a distinct, more conservative =4 setting; the
# value 6 here is grounded in SYSTEM 7.4's cube-root derivation, cited there, not invented locally.)
DEFAULT_BLOCK_WEEKS: int = 6
DEFAULT_N_BOOT: int = 2_000
DEFAULT_CI_LEVEL: float = 0.95


def _sharpe_ann(returns: np.ndarray) -> float:
    """Annualized weekly Sharpe, identical convention to metrics.compute (sqrt(52.18)/yr)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return (float(np.mean(r)) / sd) * np.sqrt(PERIODS_PER_YEAR) if sd > 0.0 else 0.0


def _cost_drag_per_year(cost_weekly: np.ndarray) -> float:
    """Annualized cost drag as a positive return drag (mean weekly cost * periods/yr)."""
    c = np.asarray(cost_weekly, dtype=float)
    return float(np.mean(c) * PERIODS_PER_YEAR) if c.size else 0.0


def net_sharpe_ci(
    net_weekly: np.ndarray,
    *,
    block_weeks: int = DEFAULT_BLOCK_WEEKS,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap CI on the deployable net Sharpe (COST.md Section 6.4, HONEST-SR).

    Delegates to ``block_bootstrap_ci`` with the ``_sharpe_ann`` statistic — the SAME annualization
    as ``metrics.compute``, so the reported point estimate and the CI are on one scale. Deterministic
    given ``seed``. iid bootstrap is FORBIDDEN for dependent returns (SYSTEM Section 17); this uses
    overlapping blocks preserving the weekly serial dependence.
    """
    return block_bootstrap_ci(
        np.asarray(net_weekly, dtype=float),
        block_length=block_weeks,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        statistic=_sharpe_ann,
    )


def cost_drag_ci(
    cost_weekly: np.ndarray,
    *,
    block_weeks: int = DEFAULT_BLOCK_WEEKS,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap CI on annualized cost drag (COST.md Section 6.4).

    Same block bootstrap, ``_cost_drag_per_year`` statistic — a dependence-preserving interval on the
    ~6.8%/yr-at-full-turnover cost line (COST.md Section 6.4) so cost realism is reported with an
    honest band, not a point estimate.
    """
    return block_bootstrap_ci(
        np.asarray(cost_weekly, dtype=float),
        block_length=block_weeks,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        statistic=_cost_drag_per_year,
    )


@dataclass(frozen=True)
class FundingSignWinRates:
    """Weekly win rate split by the sign of that week's funding cash flow (COST.md Section 6.4)."""

    unconditional: float  # win_rate_weeks over all weeks (~0.60 baseline)
    when_funding_positive: float  # win rate on weeks with funding_t > 0 (net credit weeks)
    when_funding_negative: float  # win rate on weeks with funding_t < 0 (net pay weeks)
    n_positive: int
    n_negative: int


def win_rate_by_funding_sign(
    net_weekly: np.ndarray, funding_weekly: np.ndarray
) -> FundingSignWinRates:
    """Win rate conditional on the sign of weekly funding (COST.md Section 6.4 diagnostic).

    The Section 6.4 cheap diagnostic: if losing weeks were disproportionately adverse-funding weeks,
    the two conditional rates would diverge, indicating funding drives the SIGN of weekly P&L more
    than its ~3-5%-of-gross magnitude suggests (undercutting P4's "turnover control, not funding
    timing" conclusion). Reporting both makes a future divergence visible. Zero-funding weeks are
    excluded from both conditional counts (they belong to neither sign); the unconditional rate
    matches ``metrics.compute(net_weekly).win_rate_weeks``.
    """
    net = np.asarray(net_weekly, dtype=float)
    fund = np.asarray(funding_weekly, dtype=float)
    if net.shape != fund.shape:
        raise ValueError(f"net_weekly {net.shape} and funding_weekly {fund.shape} must be aligned")
    uncond = compute(net).win_rate_weeks
    pos = fund > 0.0
    neg = fund < 0.0
    wr_pos = float(np.mean(net[pos] > 0.0)) if pos.any() else float("nan")
    wr_neg = float(np.mean(net[neg] > 0.0)) if neg.any() else float("nan")
    return FundingSignWinRates(
        unconditional=uncond,
        when_funding_positive=wr_pos,
        when_funding_negative=wr_neg,
        n_positive=int(pos.sum()),
        n_negative=int(neg.sum()),
    )
