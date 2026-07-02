"""Invariant MP empirical demonstration (REGIME.md §2.4): the SPURIOUS-FIRE test the invariant names.

The structural half of Invariant MP -- "the v1 detector uses no unshrunk covariance" -- is pinned by
``test_measures.py::test_v1_detector_uses_no_unshrunk_covariance``. This file adds the invariant's
STATED empirical demonstration: on a synthetic ``N=50, T=50`` i.i.d. panel (NO true structure), the
UNSHRUNK absorption ratio must fire spuriously at a materially higher rate than the Ledoit-Wolf-shrunk
version at the same nominal threshold -- "the empirical demonstration that motivates defaulting to
Section 2.2/2.3's univariate statistics instead" (Invariant MP). This is what makes it binding: if a
future contributor deleted the shrinkage step (the exact Marchenko-Pastur bug), this test fails.
"""

from __future__ import annotations

from convexity.regime.measures import turbulence as tb
from convexity.regime.synth.generate import iid_no_structure_panel


def test_mp_empirical_demonstration_unshrunk_fires_far_more_than_shrunk():
    """§2.4 Invariant MP: unshrunk absorption ratio over-concentrates variance on a noise panel.

    Population absorption ratio on an i.i.d. panel is 1/N = 0.02; a nominal threshold of 3x that (0.06)
    is exceeded by the UNSHRUNK sample estimate on essentially every draw (estimation noise inflates
    the top eigenvalue) but by the Ledoit-Wolf-SHRUNK estimate on almost none, at N=T=50."""
    n_symbols, n_obs, n_seeds = 50, 50, 200
    threshold = 3.0 / n_symbols  # 3x the population absorption ratio 1/N

    unshrunk_fires = shrunk_fires = 0
    for seed in range(n_seeds):
        panel = iid_no_structure_panel(n_symbols, n_obs, seed=seed)
        cov_sample = tb.sample_covariance(panel)
        cov_lw, _ = tb.ledoit_wolf_shrunk_covariance(panel)
        unshrunk_fires += tb.absorption_ratio(cov_sample) > threshold
        shrunk_fires += tb.absorption_ratio(cov_lw) > threshold

    # The demonstration: unshrunk fires spuriously at a MATERIALLY higher rate than shrunk.
    assert unshrunk_fires > 5 * max(shrunk_fires, 1)
    assert unshrunk_fires >= int(0.9 * n_seeds)  # unshrunk over-fires on nearly every draw
    assert shrunk_fires <= int(0.2 * n_seeds)    # shrinkage suppresses the spurious concentration


def test_ledoit_wolf_shrinks_heavily_when_N_approaches_T():
    """§2.4: when N ~ T the optimal shrinkage intensity delta* is large (the Marchenko-Pastur regime)."""
    panel = iid_no_structure_panel(50, 50, seed=0)
    _, delta_star = tb.ledoit_wolf_shrunk_covariance(panel)
    assert 0.0 <= delta_star <= 1.0
    assert delta_star > 0.5  # heavy shrinkage: the sample covariance is mostly noise at N=T=50


def test_true_absorption_ratio_of_iid_panel_is_one_over_N_in_population():
    """§2.4 sanity: the shrunk estimate sits far closer to the population 1/N than the unshrunk one."""
    n_symbols = 50
    panel = iid_no_structure_panel(n_symbols, 50, seed=7)
    ar_sample = tb.absorption_ratio(tb.sample_covariance(panel))
    ar_lw = tb.absorption_ratio(tb.ledoit_wolf_shrunk_covariance(panel)[0])
    population = 1.0 / n_symbols
    assert abs(ar_lw - population) < abs(ar_sample - population)
