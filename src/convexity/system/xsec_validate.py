"""Discipline layer: purged/embargoed walk-forward on the WEEKLY series + Deflated Sharpe over trials.

REUSES the SYSTEM pillars (never reimplements them):

  * ``PurgedWalkForward`` / ``make_walkforward_folds`` — the anchored, embargoed walk-forward fold
    construction. We apply it to the WEEKLY return series (one observation = one non-overlapping week),
    with an embargo of >= 1 rebalance (>= 1 week) so no test week leaks into a training fold.
  * ``DeflatedSharpe.deflate`` — PSR/DSR over the EFFECTIVE joint trial count. The cross-sectional
    search space is (signal in {momentum-by-lookback, carry, combined}) x quantile x weighting; the
    headline Sharpe MUST be deflated by that multiplicity or four-green-configs-but-no-edge sneaks in.

Both pillars key on a ``SystemDefaults``; the committed default cadence is ppy=36 (1-min H_sys scale).
For a WEEKLY series the annualization cadence is ~52 periods/yr, so we build a weekly-tuned
``SystemDefaults`` (``weekly_defaults``) that overrides ppy and the observation-count floors — nothing
about the DSR MATH changes, only the cadence + floors that must match a weekly series. The embargo used
for the walk-forward is expressed directly in WEEKS (a small integer), not via the 1-min bar union rule
(that union is for the 1-min pillar composition; here one observation already IS one week).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from convexity.system.config.defaults import SystemDefaults
from convexity.system.validate.deflated import DeflatedSharpe
from convexity.system.validate.walkforward import make_walkforward_folds

from convexity.system.metrics import PERIODS_PER_YEAR


@dataclass(frozen=True)
class _CfgShim:
    """Minimal ``cfg`` carrier for ``DeflatedSharpe.deflate`` (which reads only ``cfg.defaults``)."""

    defaults: SystemDefaults


def weekly_defaults(*, dsr_threshold: float = 0.95, k_min: int = 3, n_obs_floor: int = 24) -> SystemDefaults:
    """A weekly-cadence ``SystemDefaults``: ppy = ~52 periods/yr, floors sized for a weekly series.

    Only the annualization cadence (ppy) and the trial/observation-count floors are overridden — the
    DSR gate math (BLdP z-test, guards) is untouched. ``k_min`` is lowered from the 1-min default (8)
    to a value sensible for a small cross-sectional config search, and ``n_obs_floor`` keeps the
    MinTRL/track-length floor from passing on a too-short weekly history.
    """
    return dataclasses.replace(
        SystemDefaults(),
        ppy=PERIODS_PER_YEAR,
        dsr_threshold=dsr_threshold,
        k_min=k_min,
        n_obs_floor=n_obs_floor,
    )


@dataclass(frozen=True)
class WalkForwardOOS:
    """Out-of-sample walk-forward summary on the weekly series."""

    n_folds: int
    embargo_weeks: int
    oos_mean_weekly: float
    oos_sharpe_ann: float
    oos_n_weeks: int
    fold_sharpes: tuple[float, ...]


def purged_walkforward(
    net_weekly: np.ndarray, *, n_folds: int = 3, embargo_weeks: int = 1, anchored: bool = True
) -> WalkForwardOOS:
    """Anchored purged walk-forward on the weekly series (REUSE SYSTEM ``make_walkforward_folds``).

    One observation = one non-overlapping week, so purging reduces to the embargo gap (a week label has
    horizon 0 — it is already realized within its own week). We concatenate the per-fold TEST-slice
    returns into the OOS stream and report its aggregate Sharpe + each fold's own Sharpe. Embargo is in
    WEEKS (>= 1 rebalance), satisfying the brief's ">= 1 rebalance" embargo.
    """
    r = np.asarray(net_weekly, dtype=float)
    n = r.size
    folds = make_walkforward_folds(n, n_folds, embargo=embargo_weeks, anchored=anchored)
    oos: list[float] = []
    fold_sh: list[float] = []
    for f in folds:
        seg = r[f.test_lo : f.test_hi]
        oos.extend(seg.tolist())
        if seg.size > 1 and np.std(seg, ddof=1) > 0:
            fold_sh.append(float(np.mean(seg) / np.std(seg, ddof=1) * np.sqrt(PERIODS_PER_YEAR)))
        else:
            fold_sh.append(0.0)
    oos_arr = np.asarray(oos)
    if oos_arr.size > 1 and np.std(oos_arr, ddof=1) > 0:
        oos_sharpe = float(np.mean(oos_arr) / np.std(oos_arr, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    else:
        oos_sharpe = 0.0
    return WalkForwardOOS(
        n_folds=len(folds),
        embargo_weeks=embargo_weeks,
        oos_mean_weekly=float(np.mean(oos_arr)) if oos_arr.size else 0.0,
        oos_sharpe_ann=oos_sharpe,
        oos_n_weeks=int(oos_arr.size),
        fold_sharpes=tuple(fold_sh),
    )


def deflated_sharpe(
    net_weekly: np.ndarray,
    trial_sharpes_ann: np.ndarray,
    *,
    n_trials_effective: float | None = None,
    k_distinct: int | None = None,
    defaults: SystemDefaults | None = None,
):
    """DSR/PSR of the weekly series over the config-trial count (REUSE SYSTEM ``DeflatedSharpe``).

    ``trial_sharpes_ann`` — the ANNUALIZED Sharpes of ALL configs in the search (the trial family the
    headline was selected from). ``n_trials_effective`` — N_sys; defaults to the count of trials (the
    conservative no-correlation-shrink stance when we do not pass the per-trial return matrix).
    ``k_distinct`` — de-correlated trial count for the guard; defaults to the trial count. Returns the
    full ``DSRResult`` (headline = dsr | n_obs vs min_trl).
    """
    d = defaults or weekly_defaults()
    r = np.asarray(net_weekly, dtype=float)
    n_obs = int(r.size)
    mu = float(np.mean(r)) if n_obs else 0.0
    sd = float(np.std(r, ddof=1)) if n_obs > 1 else 0.0
    sr_per_obs = mu / sd if sd > 0 else 0.0
    sr_annual = sr_per_obs * np.sqrt(PERIODS_PER_YEAR)

    # skew / non-excess kurtosis of the weekly returns (the BLdP non-normality terms).
    if n_obs > 2 and sd > 0:
        z = (r - mu) / sd
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4))  # NON-excess (normal -> 3)
    else:
        skew, kurt = 0.0, 3.0

    ts = np.asarray(trial_sharpes_ann, dtype=float)
    n_eff = float(n_trials_effective) if n_trials_effective is not None else float(ts.size)
    kd = k_distinct if k_distinct is not None else int(ts.size)

    return DeflatedSharpe().deflate(
        path_returns=r,
        trial_sharpes=ts,
        n_obs=n_obs,
        skew=skew,
        kurt=kurt,
        n_trials_effective=n_eff,
        cfg=_CfgShim(defaults=d),
        sr_annual=sr_annual,
        trial_sharpes_annual=True,
        k_distinct=kd,
    )
