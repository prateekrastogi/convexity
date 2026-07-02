"""The BLUE composition grid + xsec search grid + config_hash (SYSTEM §5.1, Invariants CA/SG/DSR).

Covers the previously-untested surface of ``convexity.system.config.tuning``: the enumerated BLUE
composition grid (:class:`CompositionGrid` / :class:`CompositionPoint`), its ``config_hash``
content-addressing (Invariant CA — a combiner-only change perturbs the composition part of the hash
but NOT the module-version part, so module evidence still hits cache), and the §5.1 xsec search
grid cardinality (telemetry-only ``n_sys_nominal``, NEVER fed to SR*₀ — Invariant DSR). Every test
FAILS if the invariant it names is violated.
"""

from __future__ import annotations

from convexity.system.config.tuning import (
    BUDGET_PRIORITY_GRID,
    DEADBAND_GRID,
    DELTA_RANK_GRID,
    TAU_SEL_GRID,
    XSEC_LOOKBACK_BAND,
    XSEC_QUANTILE_GRID,
    XSEC_SIGNAL_GRID,
    XSEC_UNIVERSE_GRID,
    XSEC_WEIGHTING_GRID,
    CompositionGrid,
    CompositionPoint,
    nominal_grid_cardinality,
    xsec_nominal_grid_cardinality,
)


def test_composition_grid_enumerates_full_cartesian_product():
    """``enumerate()`` yields exactly the Cartesian product of the four BLUE axes (no gaps/dups)."""
    grid = CompositionGrid()
    points = grid.enumerate()
    expected = (
        len(TAU_SEL_GRID) * len(DEADBAND_GRID) * len(DELTA_RANK_GRID) * len(BUDGET_PRIORITY_GRID)
    )
    assert len(points) == expected == grid.cardinality == nominal_grid_cardinality()
    assert len(set(points)) == expected  # no duplicate coordinates
    # every enumerated coordinate is a legal member of the declared axes
    for p in points:
        assert p.tau_sel in TAU_SEL_GRID
        assert p.deadband_weight in DEADBAND_GRID
        assert p.delta_rank in DELTA_RANK_GRID
        assert p.budget_priority in BUDGET_PRIORITY_GRID


def test_config_hash_is_stable_and_content_addressed():
    """Same coordinate + same module versions => identical 16-hex hash (deterministic key)."""
    p = CompositionPoint()
    mv = {"signals": "v3", "portfolio": "v2"}
    h1 = p.config_hash(mv)
    h2 = p.config_hash(dict(mv))
    assert h1 == h2
    assert len(h1) == 16
    # dict ORDER of module_versions must not change the hash (payload sorts keys, Inv CA).
    assert p.config_hash({"portfolio": "v2", "signals": "v3"}) == h1


def test_config_hash_changes_when_composition_changes():
    """A combiner-only field change perturbs the hash (distinct coordinates key distinct evidence)."""
    mv = {"signals": "v3"}
    base = CompositionPoint()
    moved = CompositionPoint(delta_rank=3.0)
    assert base.delta_rank != moved.delta_rank
    assert base.config_hash(mv) != moved.config_hash(mv)


def test_config_hash_changes_when_module_version_changes():
    """Invariant CA: bumping a module version MUST change the cache key (stale evidence rejected)."""
    p = CompositionPoint()
    assert p.config_hash({"signals": "v3"}) != p.config_hash({"signals": "v4"})


def test_xsec_nominal_cardinality_matches_axis_product_and_excludes_lookback_axis():
    """§5.1: the 3 lookbacks are FOLDED INTO the signal axis (momentum_L12/L14/L16), so the
    nominal grid multiplies signal × quantile × weighting × universe — the lookback band is NOT a
    separate multiplicand (double-counting the search would inflate SR*₀; Invariant DSR)."""
    expected = (
        len(XSEC_SIGNAL_GRID)
        * len(XSEC_QUANTILE_GRID)
        * len(XSEC_WEIGHTING_GRID)
        * len(XSEC_UNIVERSE_GRID)
    )
    assert xsec_nominal_grid_cardinality() == expected
    # the deployed band is the m=3 lookback set {12,14,16}, NOT {12..16} (canonical config).
    assert XSEC_LOOKBACK_BAND == (12, 14, 16)
    assert len(XSEC_LOOKBACK_BAND) == 3
    # the three momentum lookbacks appear as signal-axis members, not as a separate axis.
    for lb in XSEC_LOOKBACK_BAND:
        assert f"momentum_L{lb}" in XSEC_SIGNAL_GRID
