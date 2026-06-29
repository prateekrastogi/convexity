"""CombinatorialPurgedCV — combinatorial purged CV / sampled CPCV (BREAKOUT §7 / §10).

Splits the series into ``n_groups`` contiguous groups and forms every ``C(n_groups, k_test)``
test-group combination, purging + embargoing training groups adjacent to a test group (Lopez de
Prado). Returns a Report whose ``by_fold`` carries the distribution of OOS results across paths. When
``max_paths`` is set, paths are sampled deterministically with a seed stored in the report metadata.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from convexity.core.types import Bars, CostModel, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.types import Report
from convexity.breakout.validate.labels import EntryLabeler
from convexity.breakout.validate.stats import aggregate_report
from convexity.breakout.validate.walkforward import _slice_to


class CombinatorialPurgedCV:
    """Combinatorial purged CV (BREAKOUT §7). One public method, ``evaluate``."""

    def evaluate(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        cost: CostModel,
        n_groups: int,
        k_test: int,
        embargo_bars: int,
        max_paths: int | None = None,
    ) -> Report:
        """Return a distribution of OOS results across purged combinatorial paths (§7)."""
        n = len(bars)
        bounds = np.linspace(0, n, n_groups + 1).astype(int)
        groups = [(bounds[i], bounds[i + 1]) for i in range(n_groups)]
        combos = list(combinations(range(n_groups), k_test))
        seed = 0
        if max_paths is not None and len(combos) > max_paths:
            rng = np.random.default_rng(seed)
            combos = [combos[i] for i in rng.choice(len(combos), size=max_paths, replace=False)]

        po = PipelineOrchestrator()
        labeler = EntryLabeler()
        all_labels = []
        by_fold = []
        for pi, test_groups in enumerate(combos):
            test_mask = np.zeros(n, dtype=bool)
            for g in test_groups:
                lo, hi = groups[g]
                test_mask[lo:hi] = True
            hi = int(np.max(np.where(test_mask)[0])) + 1
            sub = _slice_to(bars, hi)
            signals = po.run_batch(sub, spec, cfg)
            sigs = [s for s in signals if test_mask[s.signal_idx]]
            labels = labeler.label(sub, sigs, cost, cfg.defaults.horizons, cfg.defaults.r_targets)
            all_labels.extend(labels)
            if labels:
                by_fold.append(
                    {"path": pi, "test_groups": list(test_groups), "n": len(labels),
                     "net_bps": float(np.mean([lb.net_return_to_horizon_bps for lb in labels]))}
                )
        rep = aggregate_report(all_labels, n_folds=len(combos), by_fold=by_fold)
        return rep
