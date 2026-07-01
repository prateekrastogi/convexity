"""CPCV wiring on the xsec momentum family: path distribution (reuse SystemCPCV) + PBO + purge."""

from __future__ import annotations

import itertools

import numpy as np

from convexity.system.validate.cpcv import CPCVDistribution

from research.xsec.cpcv import (
    _group_bounds,
    _split_weeks,
    compute_pbo,
    path_distribution,
    run_cpcv,
)


def _family(n_weeks: int = 259, n_configs: int = 7, seed: int = 0) -> dict[str, np.ndarray]:
    """A synthetic aligned lookback family: config 0 has a real edge, the rest are drifting noise."""
    rng = np.random.default_rng(seed)
    fam: dict[str, np.ndarray] = {}
    for c in range(n_configs):
        drift = 0.004 if c == 0 else 0.0004 * c
        fam[f"mom-{10 + c}d/quintile/volscaled"] = drift + 0.02 * rng.standard_normal(n_weeks)
    return fam


def test_group_bounds_tile_the_timeline():
    bounds = _group_bounds(259, 8)
    assert len(bounds) == 8
    assert bounds[0][0] == 0 and bounds[-1][1] == 259
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
        assert a_hi == b_lo  # contiguous, no gap/overlap


def test_split_purges_embargo_collar_and_partitions():
    n_weeks = 80
    bounds = _group_bounds(n_weeks, 8)
    test_idx, train_idx = _split_weeks(bounds, (3, 5), embargo=1, n_weeks=n_weeks)
    # test + train are disjoint; no train week sits within the embargo collar of a test block.
    assert set(test_idx).isdisjoint(set(train_idx))
    for g in (3, 5):
        lo, hi = bounds[g]
        collar = set(range(max(0, lo - 1), min(n_weeks, hi + 1)))
        assert collar.isdisjoint(set(train_idx))  # the whole collar is purged from train
        assert set(range(lo, hi)).issubset(set(test_idx))  # the block itself is the test path


def test_path_distribution_reuses_system_cpcv_and_counts_paths():
    rng = np.random.default_rng(1)
    series = 0.003 + 0.02 * rng.standard_normal(259)
    dist, summary = path_distribution(series, n_groups=8, k_test=2, embargo=1, seed=0)
    assert isinstance(dist, CPCVDistribution)  # we REUSE SystemCPCV, not a reimplementation
    n_expect = len(list(itertools.combinations(range(8), 2)))
    assert n_expect == 28
    assert summary.n_paths == 28
    assert dist.n_groups == 8 and dist.k_test == 2
    # haircut is defined as naive - mean; both are finite annualized Sharpes.
    assert np.isfinite(summary.naive_full_sharpe) and np.isfinite(summary.mean)
    assert abs(summary.haircut - (summary.naive_full_sharpe - summary.mean)) < 1e-12
    assert 0.0 <= summary.frac_pos <= 1.0 and 0.0 <= summary.frac_gt_half <= 1.0


def test_pbo_low_when_one_config_dominates_in_and_out_of_sample():
    fam = _family(n_weeks=259, n_configs=7, seed=0)
    pbo = compute_pbo(fam, n_groups=8, k_test=2, embargo=1)
    assert pbo.n_splits == 28
    assert pbo.n_configs == 7
    # config 0 has the real edge, so best-IS should track well OOS -> low overfitting probability.
    assert 0.0 <= pbo.pbo <= 1.0
    assert pbo.pbo < 0.5
    assert len(pbo.logits) == pbo.n_splits


def test_pbo_high_when_family_is_pure_noise():
    # No config has an edge: best-IS is luck -> it lands at/below OOS median about half the time or more.
    rng = np.random.default_rng(7)
    fam = {f"mom-{10 + c}d/quintile/volscaled": 0.02 * rng.standard_normal(259) for c in range(7)}
    pbo = compute_pbo(fam, n_groups=8, k_test=2, embargo=1)
    assert pbo.pbo >= 0.4  # noise family cannot reliably rank its own best OOS


def test_run_cpcv_end_to_end_report_shape():
    fam = _family(n_weeks=259, n_configs=7, seed=2)
    selected = next(iter(fam))  # the edge config
    report = run_cpcv(fam, selected, n_groups=8, k_test=2, embargo=1)
    assert report.n_paths == 28
    assert report.n_groups == 8 and report.k_test == 2 and report.embargo_weeks == 1
    assert report.selected_label == selected
    assert report.family_labels == tuple(fam.keys())
    assert report.pbo.n_splits == 28
    assert report.path_summary.n_paths == 28
