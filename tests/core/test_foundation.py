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
    Alarm,
    EntrySignal,
    ExecTarget,
    HoldHorizon,
    InstrumentSpec,
    OpsDirective,
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
    for cls in (EntrySignal, ExecTarget, RegimeState, HoldHorizon, InstrumentSpec,
                OpsDirective, Alarm):
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


def test_regime_state_post_pivot_contract():
    """REGIME.md §4.1 migrated shape: continuous g_t in [0,1] floored at exactly 0 on HALT (NOT the
    pre-pivot {1,.75,.5,.25}-floored-.25 ladder), states HEALTHY/WATCH/STRESSED/HALT, and a
    detector-named reason chain (§0 P7). A constructed value is held as-is (policy enforces the
    band floor, the type just carries it)."""
    # HALT floors g_t at exactly 0 — the pre-pivot .25 floor is gone (§4.1).
    rs = RegimeState(
        g_t=0.0, state="HALT", binding_detector="dispersion_cusum_upper", s_stat=0.0544,
        h_threshold=0.05419, dwell_weeks=1, transition="escalate",
        reason_codes=("regime_halt_dispersion_and_btcvol",), as_of=100, engine_version="v0",
    )
    assert rs.g_t == 0.0 and rs.state == "HALT"
    assert rs.binding_detector == "dispersion_cusum_upper"
    # STRESSED band value (§4.1) and a continuous — not four-valued — g_t.
    rs2 = RegimeState(
        g_t=0.40, state="STRESSED", binding_detector="btc_vol_cusum_upper", s_stat=0.06,
        h_threshold=0.05, dwell_weeks=3, transition="hold", reason_codes=("regime_stressed_btcvol",),
        as_of=101, engine_version="v0",
    )
    assert rs2.g_t == 0.40 and rs2.state == "STRESSED"


def test_regime_state_name_is_post_pivot_vocabulary():
    """The pre-pivot CALM/NORMAL/STRESSED/CRISIS labels are retired for the cross-sectional book;
    the migrated vocabulary is HEALTHY/WATCH/STRESSED/HALT (REGIME.md §4.1)."""
    import typing

    from convexity.core.types import RegimeStateName

    assert set(typing.get_args(RegimeStateName)) == {"HEALTHY", "WATCH", "STRESSED", "HALT"}


def test_ops_directive_post_pivot_migration_contract():
    """OPS.md §9 migrated shape (foundation-level coverage of the OpsDirective/Alarm dataclass
    migration in core/types.py — the load-bearing sec-9 change, distinct from the REGIME RegimeState
    migration above). The pre-pivot OpsDirective (state: KillState + flatten/cancel_entries/
    reason_codes/engine_version) is SUPERSEDED by (state: RUN|DEGRADE|HALT, gross_multiplier: float,
    alarms: tuple[Alarm,...], as_of: int); the pre-pivot KillState enum is removed entirely (§9 step 2)."""
    from convexity.core import types as ct

    # The pre-pivot enum is gone (§9 step 2) — a lingering KillState is exactly the stale residue the
    # sec-9 reconciliation exists to remove.
    assert not hasattr(ct, "KillState"), "pre-pivot KillState must be removed from core/types.py (§9)"

    # The migrated OpsDirective has §9's EXACT field set/shape, and the pre-pivot fields are gone.
    d = OpsDirective(state="DEGRADE", gross_multiplier=0.5, alarms=(), as_of=7)
    fields = {f.name for f in dataclasses.fields(OpsDirective)}
    assert fields == {"state", "gross_multiplier", "alarms", "as_of"}
    for pre_pivot in ("flatten", "cancel_entries", "reason_codes", "engine_version"):
        assert pre_pivot not in fields, f"pre-pivot OpsDirective field {pre_pivot} must be gone (§9)"
    assert d.state == "DEGRADE" and isinstance(d.gross_multiplier, float)

    # Alarm was added to core/types.py so OpsDirective.alarms is constructible outside ops/ (§9 step 3).
    a = Alarm(kind="CUSUM_ALARM", detail="x", value=7.0, threshold=6.23, as_of=7)
    assert {f.name for f in dataclasses.fields(Alarm)} == {"kind", "detail", "value", "threshold",
                                                           "as_of"}
    d2 = OpsDirective(state="HALT", gross_multiplier=0.0, alarms=(a,), as_of=7)
    assert d2.alarms == (a,)


def test_ops_directive_is_single_source_of_truth():
    """OPS.md §9 step 1 / §12 cross-consistency: core.types.OpsDirective and ops.types.OpsDirective are
    the SAME object (not two same-named classes shadowing across modules), and the OpsEngine Protocol's
    return annotation resolves to this one shape."""
    from convexity.core import types as ct
    from convexity.ops import types as ot

    assert ot.OpsDirective is ct.OpsDirective
    assert ot.Alarm is ct.Alarm


def test_ops_state_and_alarm_kind_vocabularies():
    """OPS.md §9: the OpsState total order is RUN/DEGRADE/HALT (the pre-pivot lowercase run/degrade/halt
    KillState is retired), and AlarmKind is the closed §9 set."""
    import typing

    from convexity.core.types import AlarmKind, OpsState

    assert set(typing.get_args(OpsState)) == {"RUN", "DEGRADE", "HALT"}
    assert set(typing.get_args(AlarmKind)) == {
        "RECON_BREACH", "RECON_STALE", "CUSUM_ALARM", "CUSUM_WARN", "SLA_BREACH",
        "SLA_BREACH_SEVERE", "ANOMALOUS_OUTPERFORMANCE", "HEARTBEAT_SILENCE",
    }
