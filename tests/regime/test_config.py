"""REGIME config governance tests (REGIME.md §3.4/§6 Invariant C-cfg).

RED enters ONLY through resolve_config; no RED constant appears in AMBER metadata, and REGIME has no
per-instrument axis at all (§3.4). Also pins the AMBER preset's reuse (not reforking) of
research/xsec/universe.CRYPTO_REGIMES and the §3.4 in-control-window choice.
"""

from convexity.core.params import Bucket, assert_no_red_in_amber
from research.xsec.universe import CRYPTO_REGIMES
from convexity.regime.config.asset_class import CRYPTO_REGIME_PRESET
from convexity.regime.config.defaults import REGIME_PARAM_REGISTRY, RegimeDefaults
from convexity.regime.config.resolve import ResolvedRegimeConfig, resolve_config


def test_resolve_returns_frozen_merged_config():
    cfg = resolve_config()
    assert isinstance(cfg, ResolvedRegimeConfig)
    assert isinstance(cfg.defaults, RegimeDefaults)


def test_amber_preset_carries_no_red_field():
    """Invariant C-cfg (§3.4/§6): the AMBER preset field names contain no RED constant name."""
    import dataclasses

    amber_names = frozenset(f.name for f in dataclasses.fields(CRYPTO_REGIME_PRESET))
    # this is exactly what resolve_config runs; it must not raise
    assert_no_red_in_amber(amber_names, REGIME_PARAM_REGISTRY)
    # explicit: no RED name leaked onto AMBER
    red_names = {"h_warn_frac", "n_confirm", "n_min_dispersion", "n_weeks_min_cpcv",
                 "bleed_return_tol", "k_regime_max"}
    assert not (amber_names & red_names)


def test_amber_reuses_crypto_regimes_not_reforked():
    """§6 AMBER: the regime windows are reused from research/xsec/universe.CRYPTO_REGIMES."""
    assert CRYPTO_REGIME_PRESET.regime_windows == dict(CRYPTO_REGIMES)


def test_in_control_windows_are_recovery_and_bull_not_bear():
    """§3.4: in-control training is 2023_recovery + 2024_bull; 2022_bear is a stress regime, excluded."""
    ic = CRYPTO_REGIME_PRESET.in_control_windows
    assert set(ic) == {"2023_recovery", "2024_bull"}
    assert "2022_bear" not in ic


def test_market_symbol_matches_overlay_default():
    """§6 AMBER / §2.3: V_t's market symbol is BTCUSDT, matching overlay.py's default."""
    assert CRYPTO_REGIME_PRESET.market_symbol == "BTCUSDT"


def test_no_per_instrument_axis():
    """§3.4 Invariant C-cfg / §0 P2: RegimeDefaults are all book-level scalars — no per-symbol field."""
    import dataclasses

    for f in dataclasses.fields(RegimeDefaults):
        assert f.type in ("float", "int"), f"{f.name}: RED constants are scalars, not per-instrument"


def test_k_regime_max_is_defined_here_not_a_magic_reference():
    """§6: K_regime_max has no field anywhere in the repo before this pillar — it is defined here."""
    assert RegimeDefaults().k_regime_max > 0


def test_live_delta_and_arl0_are_registered_RED_not_BLUE():
    """§3.4/§6: the FROZEN LIVE delta_target/arl_0_target are RED 'shared economic constants', NOT
    BLUE — registering the deployed constant itself as BLUE bucket-mislabels the RED value (§6). The
    BLUE candidate SETS are separate *_grid entries (COST's h_grid/RED-live pattern). This test fails
    on the pre-fix registry, which registered delta_target/arl_0_target themselves as BLUE."""
    by_name = {p.name: p for p in REGIME_PARAM_REGISTRY.params()}
    assert by_name["delta_target"].bucket is Bucket.RED
    assert by_name["arl_0_target"].bucket is Bucket.RED
    # a RED live constant carries NO blue_grid (that would re-introduce the mislabel via the back door)
    assert by_name["delta_target"].blue_grid is None
    assert by_name["arl_0_target"].blue_grid is None


def test_blue_search_grids_are_separate_entries_matching_the_spec_sets():
    """§6: the offline search grids are SEPARATE BLUE entries — delta_target_grid={1.0,1.5,2.0},
    arl_0_target_grid={52,104,208} — mirroring COST's h_grid. The live RED values sit inside them."""
    by_name = {p.name: p for p in REGIME_PARAM_REGISTRY.params()}
    dg, ag = by_name["delta_target_grid"], by_name["arl_0_target_grid"]
    assert dg.bucket is Bucket.BLUE and ag.bucket is Bucket.BLUE
    assert dg.blue_grid == (1.0, 1.5, 2.0)
    assert ag.blue_grid == (52, 104, 208)
    # the deployed RED values are members of their respective BLUE grids (§6).
    d = RegimeDefaults()
    assert d.delta_target in dg.blue_grid and d.arl_0_target in ag.blue_grid


def test_k_regime_max_bounds_the_blue_grid_cardinality_fed_to_the_system_ledger():
    """§6/§7.4 Invariant LEDGER (the in-scope, implementable half): the product of REGIME's BLUE grid
    cardinalities — the count of distinct (Delta_target, ARL_0_target[, dd_trigger, dd_reset]) tuples
    that can enter SYSTEM's ONE trial ledger — must not exceed K_regime_max, the overnight compute cap
    (§6). The ledger-EQUALITY half (§7.4) is now binding in regime/validate/backtest.py; see
    tests/regime/test_ledger.py::test_ledger_equality_scored_tuples_equal_regime_tagged_entries."""
    card = REGIME_PARAM_REGISTRY.blue_grid_cardinality()   # 3 (delta) * 3 (arl0) = 9 detector combos
    assert card == 9
    assert card <= RegimeDefaults().k_regime_max            # 9 <= 27 (leaves headroom for dd pairs)
