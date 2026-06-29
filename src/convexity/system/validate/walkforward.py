"""Purged walk-forward + the union embargo keyed to H_sys (SYSTEM §6.2/§6.3, Inv SC/HC).

Anchored / rolling expanding windows. In each fold every module's TEAL fit, every threshold, every
selection rank, every covariance estimate is trained ONLY on that fold's permitted history; the
composed pipeline runs forward on the test slice; system triple-barrier labels score the realized
book deltas net of cost; folds are time-ordered, embargoed.

The embargo is the UNION (max horizon) across modules (Section 6.2):

    embargo_bars_sys = max(H_sys, max over modules of label_horizon_m) + max over modules of warmup_m

The system triple-barrier horizon is bounded by the contractual cap ``H_sys`` (Invariant HC), so
``H_sys`` — not a mean or typical realized horizon — enters the union; this closes the endogenous,
path-dependent-horizon lookahead. Purge drops training labels whose [start,end] overlaps any test
label after applying the embargo on BOTH sides of each test fold (Invariant SC).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.system.config.defaults import SystemDefaults, system_embargo_bars


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: time-ordered train/test bar ranges (half-open [lo, hi))."""

    train_lo: int
    train_hi: int
    test_lo: int
    test_hi: int


def system_embargo(defaults: SystemDefaults) -> int:
    """The union embargo width (Section 6.2, Invariants SC/HC). Re-exported from config so callers
    key on the contractual cap, never a mean/typical horizon."""
    return system_embargo_bars(defaults)


def purge_train_indices(
    train_lo: int,
    train_hi: int,
    test_lo: int,
    test_hi: int,
    *,
    embargo: int,
    label_horizon: int,
) -> tuple[int, int, int]:
    """Drop training labels that overlap the test fold after the embargo on BOTH sides (Section 6.2).

    A training label at bar ``i`` spans [i, i + label_horizon]; it overlaps the test fold (widened by
    the embargo on both sides) iff its span intersects [test_lo − embargo, test_hi + embargo).
    Returns the purged training range as (lo, hi_before, hi_after) — the kept training is
    [train_lo, hi_before) ∪ [hi_after, train_hi); for an anchored fold (train precedes test) only the
    right edge purges, so we return the kept upper bound. The returned tuple is (lo, kept_hi, gap_lo)
    so callers can reconstruct both kept segments.
    """
    embargo_lo = test_lo - embargo
    embargo_hi = test_hi + embargo
    # A train label at i overlaps iff [i, i+label_horizon] ∩ [embargo_lo, embargo_hi) ≠ ∅.
    # ⇒ i + label_horizon >= embargo_lo  AND  i < embargo_hi.
    kept_hi = min(train_hi, max(train_lo, embargo_lo - label_horizon))
    gap_lo = max(train_lo, embargo_hi)
    return train_lo, kept_hi, gap_lo


def make_walkforward_folds(
    n_bars: int, n_folds: int, *, embargo: int, anchored: bool = True
) -> list[Fold]:
    """Build time-ordered expanding (anchored) or rolling walk-forward folds with the embargo gap.

    Anchored: training is [0, test_lo − embargo) and grows; rolling keeps a fixed window. The test
    folds tile the back of the timeline; consecutive test folds are separated so the embargo never
    leaks a test label into the next fold's fit (Invariant SC).
    """
    folds: list[Fold] = []
    seg = n_bars // (n_folds + 1)
    for f in range(1, n_folds + 1):
        test_lo = seg * f
        test_hi = seg * (f + 1) if f < n_folds else n_bars
        train_hi = max(0, test_lo - embargo)
        train_lo = 0 if anchored else max(0, train_hi - seg)
        folds.append(Fold(train_lo=train_lo, train_hi=train_hi, test_lo=test_lo, test_hi=test_hi))
    return folds


class PurgedWalkForward:
    """Purged, embargoed walk-forward over the composed chain (Section 6.3). The embargo is the union
    keyed to H_sys (Invariants SC/HC); every fit/threshold/rank/covariance is fold-local."""

    def __init__(self, defaults: SystemDefaults) -> None:
        self._defaults = defaults

    @property
    def embargo_bars(self) -> int:
        return system_embargo(self._defaults)

    def folds(self, n_bars: int, n_folds: int, *, anchored: bool = True) -> list[Fold]:
        return make_walkforward_folds(n_bars, n_folds, embargo=self.embargo_bars, anchored=anchored)
