"""§9 validation deliverable: the execution-adjusted weekly series through the SAME battery
(EXECUTION.md §9). These tests assert the reconcile layer (a) is a pure per-week transform in the
P6-conservative direction, and (b) delegates to convexity.system.xsec_validate.{walkforward,deflated,cpcv}
BYTE-FOR-BYTE (reused via convexity.system.{validate,cpcv}), never a new statistical test."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.system import cpcv as xsec_cpcv
from convexity.system import xsec_validate as xsec_validate
from convexity.execution.execution import (
    ExecutionValidation,
    execution_adjusted_series,
    reconcile,
)


def _family(rng: np.random.Generator, n_weeks: int = 120, n_configs: int = 7) -> dict[str, np.ndarray]:
    """A synthetic aligned lookback family: one weakly-positive-mean weekly series per config."""
    fam: dict[str, np.ndarray] = {}
    for k in range(n_configs):
        mu = 8e-4 + 1e-4 * k  # small positive weekly drift, config-dependent
        fam[f"L{10 + 2 * k}"] = rng.normal(mu, 0.02, n_weeks)
    return fam


def test_execution_adjusted_series_is_conservative_per_week_transform():
    # §0/§9: net_adj = net_frictionless - realized_shortfall, per week. A POSITIVE shortfall can only
    # LOWER the realized return (P6: execution never improves on the frictionless proxy). Zero
    # shortfall recovers the frictionless series exactly (the Delta->0, lambda->0 corner, §13).
    net = np.array([0.01, -0.005, 0.02, 0.0])
    shortfall = np.array([1e-3, 2e-3, 0.0, 5e-4])  # all >= 0 (costs)
    adj = execution_adjusted_series(net, shortfall)
    assert np.array_equal(adj, net - shortfall)
    assert np.all(adj <= net + 1e-18)  # conservative direction
    assert np.array_equal(execution_adjusted_series(net, np.zeros_like(net)), net)  # zero => identity
    with pytest.raises(ValueError):
        execution_adjusted_series(net, shortfall[:-1])  # shape mismatch


def test_reconcile_delegates_byte_for_byte_to_the_existing_battery():
    # §9: reconcile is a SECOND INPUT SERIES through the existing machinery, not a new test. Assert
    # its outputs equal calling convexity.system.{validate.purged_walkforward, validate.deflated_sharpe,
    # cpcv.run_cpcv} directly on the SAME execution-adjusted series — byte-for-byte (Invariant N-exec).
    rng = np.random.default_rng(5)
    fam = _family(rng)
    selected = "L14"
    n_weeks = len(fam[selected])
    shortfall = rng.uniform(0.0, 5e-4, n_weeks)  # a realistic small positive per-week shortfall

    ev = reconcile(fam, selected, shortfall, n_groups=6, k_test=2, embargo=1, wf_folds=3, seed=0)
    assert isinstance(ev, ExecutionValidation)

    # reconstruct the EXACT same delegation by hand and demand equality.
    adjusted = fam[selected] - shortfall
    adj_family = {lab: (adjusted if lab == selected else s) for lab, s in fam.items()}
    wf = xsec_validate.purged_walkforward(adjusted, n_folds=3, embargo_weeks=1)
    cp = xsec_cpcv.run_cpcv(adj_family, selected, n_groups=6, k_test=2, embargo=1, seed=0)
    trial_sharpes = np.array([xsec_cpcv._sharpe_ann(adj_family[lab]) for lab in adj_family])
    dsr = xsec_validate.deflated_sharpe(adjusted, trial_sharpes)

    assert ev.wf_oos_sharpe_ann == wf.oos_sharpe_ann
    assert ev.naive_full_sharpe_ann == cp.path_summary.naive_full_sharpe
    assert ev.cpcv_mean_path_sharpe == cp.path_summary.mean
    assert ev.cpcv_frac_pos == cp.path_summary.frac_pos
    assert ev.cpcv_haircut == cp.path_summary.haircut
    assert ev.cpcv_pbo == cp.pbo.pbo
    assert ev.dsr_passed == bool(dsr.passed)


def test_reconcile_zero_shortfall_matches_frictionless_run():
    # §9/§13 (Delta->0, lambda->0 corner): an all-zero realized shortfall must reproduce the
    # frictionless run's CPCV numbers EXACTLY — the adjusted series IS the frictionless series, so the
    # execution layer adds no distortion when there is nothing to price.
    rng = np.random.default_rng(9)
    fam = _family(rng)
    selected = "L16"
    n_weeks = len(fam[selected])

    ev = reconcile(fam, selected, np.zeros(n_weeks), n_groups=6, k_test=2, embargo=1, seed=0)
    cp = xsec_cpcv.run_cpcv(fam, selected, n_groups=6, k_test=2, embargo=1, seed=0)
    assert ev.cpcv_mean_path_sharpe == cp.path_summary.mean
    assert ev.cpcv_frac_pos == cp.path_summary.frac_pos
    assert ev.naive_full_sharpe_ann == cp.path_summary.naive_full_sharpe
    assert ev.cpcv_pbo == cp.pbo.pbo


def test_reconcile_positive_shortfall_only_hurts_the_selected_series():
    # §9: a strictly-positive realized shortfall lowers the SELECTED config's full-sample Sharpe vs
    # the frictionless run (the honest haircut), while the frictionless family it is compared against
    # for PBO is unchanged. The adjusted naive Sharpe must be <= the frictionless naive Sharpe.
    rng = np.random.default_rng(13)
    fam = _family(rng)
    selected = "L14"
    n_weeks = len(fam[selected])
    shortfall = np.full(n_weeks, 1e-3)  # a persistent positive cost every week

    ev_adj = reconcile(fam, selected, shortfall, n_groups=6, k_test=2, embargo=1, seed=0)
    ev_frictionless = reconcile(fam, selected, np.zeros(n_weeks), n_groups=6, k_test=2, embargo=1, seed=0)
    assert ev_adj.naive_full_sharpe_ann < ev_frictionless.naive_full_sharpe_ann


def test_reconcile_rejects_unknown_selected_label():
    rng = np.random.default_rng(1)
    fam = _family(rng)
    with pytest.raises(KeyError):
        reconcile(fam, "L99", np.zeros(len(next(iter(fam.values())))))
