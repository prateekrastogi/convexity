"""Phase-0 foundation tests: governance buckets, cross-pillar types, composition protocols.

These pin the shared-foundation contract every pillar builds on (docs/ARCHITECTURE.md §3) and
guarantee the foundation imports cleanly + behaves before any pillar logic lands.
"""

import dataclasses

import pytest

from convexity.core.params import (
    Bucket,
    ParamSpec,
    StaticParamRegistry,
    assert_no_red_in_amber,
)
from convexity.core.types import (
    EntrySignal,
    ExecTarget,
    HoldHorizon,
    InstrumentSpec,
    RegimeState,
)


# --------------------------------------------------------------------------- governance

def test_buckets_are_the_four_governance_classes():
    assert {b.value for b in Bucket} == {"red", "amber", "blue", "teal"}


def test_blue_grid_rejected_on_non_blue_param():
    ParamSpec("window", Bucket.BLUE, "lookback", "BREAKOUT", blue_grid=(20, 40))  # ok
    with pytest.raises(ValueError):
        ParamSpec("chi", Bucket.RED, "chase frac", "EXECUTION", blue_grid=(0.25,))


def test_registry_bucketing_and_blue_cardinality():
    reg = StaticParamRegistry(
        (
            ParamSpec("chi", Bucket.RED, "chase frac", "EXECUTION"),
            ParamSpec("tick_size", Bucket.AMBER, "tick", "core"),
            ParamSpec("window", Bucket.BLUE, "lookback", "BREAKOUT", blue_grid=(20, 40, 80, 160)),
            ParamSpec("arl0", Bucket.BLUE, "arl target", "OPS", blue_grid=(250, 500, 1000)),
        )
    )
    assert {p.name for p in reg.of_bucket(Bucket.RED)} == {"chi"}
    # product of BLUE grids = the per-pillar trial count fed to SYSTEM's deflation
    assert reg.blue_grid_cardinality() == 4 * 3


def test_blue_cardinality_is_one_when_no_blue_params():
    reg = StaticParamRegistry((ParamSpec("c", Bucket.RED, "cap", "LEVERAGE"),))
    assert reg.blue_grid_cardinality() == 1


def test_invariant_c_helper_passes_and_raises():
    reg = StaticParamRegistry(
        (
            ParamSpec("chi", Bucket.RED, "chase frac", "EXECUTION"),
            ParamSpec("tick_size", Bucket.AMBER, "tick", "core"),
        )
    )
    # AMBER-only field set is fine
    assert_no_red_in_amber(frozenset({"tick_size", "lot_size"}), reg)
    # a RED name leaking into the per-instrument field set is the Invariant-C bug
    with pytest.raises(AssertionError, match="Invariant C"):
        assert_no_red_in_amber(frozenset({"chi", "tick_size"}), reg)


# --------------------------------------------------------------------------- shared types

def test_instrument_spec_carries_no_red_or_blue_fields():
    """Invariant C (every pillar): the cross-pillar InstrumentSpec is AMBER-only."""
    fields = {f.name for f in dataclasses.fields(InstrumentSpec)}
    forbidden = {
        "k_cross", "window", "arl0", "kelly_fraction", "phi", "chi", "edge_buffer",
        "governor_breakpoint", "es_level", "hazard", "threshold", "stop_policy",
    }
    assert not (fields & forbidden)


def test_hold_horizon_m_max_is_ceil_and_cannot_disagree():
    # 25h hold, 8h funding -> ceil(25/8) = 4 stamps
    h = HoldHorizon.from_quantile(25 * 3600_000, 8 * 3600_000)
    assert h.m_max == 4
    assert h.n_funding_stamps_remaining == 4
    # exact multiple -> no spurious extra stamp
    h2 = HoldHorizon.from_quantile(16 * 3600_000, 8 * 3600_000)
    assert h2.m_max == 2


def test_boundary_objects_are_frozen():
    for cls in (EntrySignal, ExecTarget, RegimeState, HoldHorizon, InstrumentSpec):
        params = dataclasses.fields(cls)
        assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, cls.__name__
        assert params  # has fields


def test_exec_target_has_no_edge_field():
    """EXECUTION Inv A: the decided trade carries no edge/sizing/exit score."""
    fields = {f.name for f in dataclasses.fields(ExecTarget)}
    forbidden = {"breakout_score", "expected_return", "trend_confidence", "kelly_fraction",
                 "quality_score"}
    assert not (fields & forbidden)
    assert fields == {
        "intent_id", "instrument", "side", "target_qty", "arrival_mid",
        "invalidation", "urgency", "reduce_only",
    }


def test_regime_state_g_t_floor_documented_range():
    """REGIME g_t lives in {1.00,0.75,0.50,0.25}; a constructed value is held as-is (policy
    enforces the floor, the type just carries it)."""
    rs = RegimeState(
        g_t=0.25, state="CRISIS", binding_sub="corr", dwell_bars=1, transition="escalate",
        cost_gate_passed=True, rebalance_cost_hat=1.0, reason_codes=("regime_escalate_corr_crisis",),
        asof_bar=100, engine_version="v0",
    )
    assert rs.g_t == 0.25 and rs.binding_sub == "corr"
