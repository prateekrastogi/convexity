"""WalkForward — purged walk-forward evaluation (BREAKOUT §7 / §10).

Purged, embargoed walk-forward: every fit/calibration/TEAL profile inside a fold is trained only on
that fold's permitted training history, and test labels do not overlap training labels after purge +
embargo (BREAKOUT §walkforward). Scores entry quality from ``entry_idx`` (Invariant G). The detector
itself is strictly causal, so the "training" here is the empirical-null/threshold calibration window;
the test segment is evaluated forward with labels computed only inside the test window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.core.types import Bars, CostModel, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.types import Report
from convexity.breakout.validate.labels import EntryLabeler
from convexity.breakout.validate.stats import aggregate_report


@dataclass(frozen=True)
class SplitPlan:
    """A purged/embargoed walk-forward split plan (BREAKOUT §10). ``folds`` is a list of
    ``(train_lo, train_hi, test_lo, test_hi)`` index tuples with embargo applied."""

    folds: list[tuple[int, int, int, int]]
    embargo_bars: int

    @classmethod
    def expanding(cls, n: int, n_folds: int, embargo_bars: int, min_train: int = 200) -> "SplitPlan":
        """Expanding-window walk-forward: each fold trains on ``[0, test_lo - embargo)`` and tests on
        the next contiguous block."""
        test_size = max((n - min_train) // n_folds, 1)
        folds = []
        for k in range(n_folds):
            test_lo = min_train + k * test_size
            test_hi = min(test_lo + test_size, n)
            if test_lo >= n:
                break
            train_hi = max(test_lo - embargo_bars, 0)
            folds.append((0, train_hi, test_lo, test_hi))
        return cls(folds=folds, embargo_bars=embargo_bars)


class WalkForward:
    """Purged walk-forward validation (BREAKOUT §7). One public method, ``evaluate``."""

    def evaluate(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        cost: CostModel,
        splits: SplitPlan,
    ) -> Report:
        """Purged walk-forward (§7). Labels start from ``entry_idx`` and live within each test fold."""
        po = PipelineOrchestrator()
        labeler = EntryLabeler()
        all_labels = []
        all_scores: list[float] = []
        by_fold = []
        for fi, (_tlo, _thi, test_lo, test_hi) in enumerate(splits.folds):
            # the detector is causal: run it over [0, test_hi) and keep only signals firing in the
            # test window (their evidence used only data <= signal bar, no train/test leakage).
            sub = _slice_to(bars, test_hi)
            signals = po.run_batch(sub, spec, cfg)
            test_signals = [s for s in signals if test_lo <= s.signal_idx < test_hi]
            labels = labeler.label(sub, test_signals, cost, cfg.defaults.horizons, cfg.defaults.r_targets)
            # thread the per-signal BREAKOUT-internal quality_score aligned 1:1 with the labels the
            # labeler actually emitted (it may skip a signal, §10 decile table; Invariant K firewall:
            # the score is a SEPARATE diagnostic array, never an EntryLabel field).
            score_by_idx = {s.signal_idx: s.quality_score for s in test_signals}
            all_scores.extend(score_by_idx[lb.signal_idx] for lb in labels)
            all_labels.extend(labels)
            if labels:
                by_fold.append(
                    {"fold": fi, "n": len(labels),
                     "net_bps": float(np.mean([lb.net_return_to_horizon_bps for lb in labels]))}
                )
        return aggregate_report(
            all_labels, scores=np.asarray(all_scores), n_folds=len(splits.folds), by_fold=by_fold
        )


def _slice_to(bars: Bars, hi: int) -> Bars:
    def s(arr):
        return None if arr is None else arr[:hi]

    return Bars(
        ts=bars.ts[:hi], o=bars.o[:hi], h=bars.h[:hi], l=bars.l[:hi], c=bars.c[:hi],
        v=s(bars.v), bid=s(bars.bid), ask=s(bars.ask), trade_count=s(bars.trade_count),
        is_session_open=s(bars.is_session_open), is_valid=s(bars.is_valid),
        is_roll_window=s(bars.is_roll_window), funding_rate=s(bars.funding_rate),
    )
