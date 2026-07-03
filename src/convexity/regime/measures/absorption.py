"""Cross-sectional return dispersion ``D_t`` (REGIME.md §2.2).

Filename kept from the pre-pivot ``docs/ARCHITECTURE.md`` tree (§9 bracket note), but the CONTENT is
the NEW cross-sectional dispersion statistic, NOT the old per-instrument absorption ratio (which §2.4
explicitly rejects as the default on Marchenko-Pastur / random-matrix grounds — an ``N x N`` sample
covariance over ~50 names and ~50 weekly obs is ill-conditioned). ``D_t`` is a single UNIVARIATE
weekly scalar (§2.1), so it feeds one CUSUM without ever fitting a covariance matrix.

Definition (§2.2), over the point-in-time ELIGIBLE cross-section ``U_tau = Panel.listed[tau]`` — the
SAME eligibility mask ``src/convexity/signal/signals.py`` uses for the momentum rank, so ``D_t`` and the
momentum signal share an identical peer set by construction (§2.2, "no separate REGIME universe"):

    r_{i,tau} = close[tau,i]/close[tau-1,i] - 1        (daily simple return; supplied by panel/build)
    D_tau     = std_{i in U_tau}( r_{i,tau} )          (cross-sectional std across eligible i, one day)
    D_t       = sqrt( (1/n) * sum_{tau in week t} D_tau^2 )   (RMS of daily cross-sectional std over week t)

The RMS (not arithmetic-mean) aggregation is the §2.2 variance-additivity result: ``D_tau`` is itself
a std (a second-moment object), so the week's dispersion SCALE is the root-mean-square of the daily
values, the same argument that aggregates daily vol to weekly as sigma_week = sigma_day * sqrt(5).
"""

from __future__ import annotations

import numpy as np


def daily_dispersion(
    daily_returns: np.ndarray, listed: np.ndarray, tau: int, *, n_min: int
) -> tuple[float, int]:
    """Cross-sectional std of daily returns across the eligible universe on day ``tau`` (REGIME.md §2.2).

    Returns ``(D_tau, n_eligible)`` where ``D_tau`` is ``std_{i in U_tau}(r_{i,tau})`` over names that
    are BOTH listed on ``tau`` and have a finite daily return, and ``n_eligible`` is that count. When
    ``n_eligible < n_min`` (§8.1 early-history breadth floor) ``D_tau`` is returned as ``NaN`` so the
    caller can EXCLUDE the day from the weekly RMS entirely — never feed a degenerate value that would
    artificially suppress the CUSUM (§8.1: "not fed as a degenerate/zero value")."""
    row = daily_returns[tau]
    ok = listed[tau] & np.isfinite(row)
    n_eligible = int(ok.sum())
    if n_eligible < n_min:
        return (np.nan, n_eligible)
    # population-style cross-sectional std (ddof=0): the dispersion of the realized cross-section,
    # not an inference about a super-population (§2.2 treats D_tau as the realized spread).
    return (float(np.std(row[ok])), n_eligible)


def weekly_dispersion(
    daily_returns: np.ndarray,
    listed: np.ndarray,
    day_indices: np.ndarray,
    *,
    n_min: int,
) -> tuple[float, int]:
    """RMS-aggregate the daily cross-sectional std over one rebalance week (REGIME.md §2.2 ``D_t``).

    ``day_indices`` are the panel row indices ``tau`` belonging to rebalance week ``t`` (the daily grid
    within the week). Returns ``(D_t, n_days_used)`` where ``n_days_used`` is the number of days whose
    eligible breadth cleared ``n_min``.

    §8.1 breadth rule (two distinct clauses, both enforced here):

      * **Majority-of-days floor (the WATCH escalation).** If ``|U_tau| < n_min`` for a MAJORITY of
        the week's days (strictly more than half), the week's dispersion is UNDEFINED: ``D_t`` is
        returned as ``NaN`` so the caller EXCLUDES ``D_t`` from that week's CUSUM update entirely and
        reports ``WATCH``/``insufficient_breadth`` (never ``HEALTHY``, §0 P6). A minority of surviving
        days is too thin a base to trust the week's dispersion as HEALTHY-eligible — computing ``D_t``
        from that minority would be the more-permissive outcome P6 forbids.
      * **Per-day exclusion (not zero-fill).** When the majority DO clear the floor, individual
        sub-floor days are still EXCLUDED from the RMS (their :func:`daily_dispersion` is ``NaN``),
        never fed as a degenerate/zero value that would artificially suppress ``S_D^+``/``S_D^-``.

    An empty week (no day indices) is also undefined -> ``NaN`` (conservative WATCH, §0 P6)."""
    n_days = int(len(day_indices))
    if n_days == 0:
        return (np.nan, 0)
    sq: list[float] = []
    n_below = 0
    for tau in day_indices:
        d_tau, n_eligible = daily_dispersion(daily_returns, listed, int(tau), n_min=n_min)
        if n_eligible < n_min:
            n_below += 1
        if np.isfinite(d_tau):
            sq.append(d_tau * d_tau)
    # §8.1 majority rule: a MAJORITY (strictly > half) of days below the floor -> insufficient breadth
    # for the whole week; exclude D_t from the CUSUM and report WATCH. This also subsumes the
    # all-days-below case (n_below == n_days), which necessarily leaves ``sq`` empty.
    if n_below * 2 > n_days:
        return (np.nan, len(sq))
    return (float(np.sqrt(np.mean(sq))), len(sq))
