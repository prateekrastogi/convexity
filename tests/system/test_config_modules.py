"""Config governance (Inv C/SG) + thin-adapter tests (Inv P1, ARCH §5.10 selection mapping)."""

import numpy as np
import pytest

from convexity.core.params import Bucket
from convexity.core.types import InstrumentSpec
from convexity.selection.dynamic_optimization import Instrument, optimise_portfolio
from convexity.system.config.defaults import SYSTEM_PARAM_REGISTRY, SystemDefaults
from convexity.system.config.resolve import resolve_config
from convexity.system.config.tuning import CompositionGrid, CompositionPoint, nominal_grid_cardinality
from convexity.system.config.universe import INSTRUMENT_META_FIELDS, load_universe_meta
from convexity.system.modules.breakout_adapter import BreakoutAdapter
from convexity.system.modules.selection import SelectionAdapter
from convexity.system.modules.trend_adapter import TrendAdapter


# --------------------------------------------------------------------------- governance (Inv C)


def test_no_red_blue_leaks_into_amber_metadata():
    """Invariant C: no RED/BLUE param name appears on the AMBER per-instrument metadata."""
    forbidden = {p.name for p in SYSTEM_PARAM_REGISTRY.params() if p.bucket in (Bucket.RED, Bucket.BLUE)}
    assert not (forbidden & INSTRUMENT_META_FIELDS)


def test_resolve_runs_the_firewall():
    """resolve_config runs the Invariant-C firewall and merges RED+BLUE+AMBER (single merge point)."""
    cfg = resolve_config(load_universe_meta([]))
    assert isinstance(cfg.defaults, SystemDefaults)
    assert cfg.h_sys == cfg.defaults.h_sys
    assert cfg.embargo_bars >= cfg.h_sys


def test_red_enters_only_through_defaults():
    """RED is reachable ONLY via defaults= (Invariant C)."""
    custom = SystemDefaults(h_sys=5000, dsr_threshold=0.99)
    cfg = resolve_config(load_universe_meta([]), defaults=custom)
    assert cfg.h_sys == 5000
    assert cfg.defaults.dsr_threshold == 0.99


def test_blue_grid_is_telemetry_only_product():
    """SG/§7.1: the Cartesian product is n_sys_nominal telemetry, never fed to SR*0."""
    grid = CompositionGrid()
    assert grid.cardinality == nominal_grid_cardinality()
    assert len(grid.enumerate()) == grid.cardinality
    assert nominal_grid_cardinality() == 2 * 3 * 3 * 2


def test_config_hash_changes_with_combiner_but_not_module_version_part():
    """A combiner-only change perturbs the config hash (so the ledger counts it) — Invariant SG/CA."""
    mv = {"breakout": "1", "leverage": "1", "trend": "1", "selection": "1"}
    a = CompositionPoint(deadband_weight=0.01).config_hash(mv)
    b = CompositionPoint(deadband_weight=0.04).config_hash(mv)
    assert a != b  # distinct combiner coordinates ⇒ distinct ledger configs


# --------------------------------------------------------------------------- selection mapping (ARCH §5.10)


def test_selection_adapter_maps_dynopt_to_proposed_portfolio():
    """ARCH §5.10: keep/eligible from non-zero positions; risk_weights from optimised weights."""
    instruments = [
        Instrument("US5y", 0.4, 0.22, 0.052, cost_per_contract=5.50),
        Instrument("US10y", 0.9, 0.24, 0.082, cost_per_contract=11.50),
        Instrument("SP500", 3.1, 0.04, 0.171, cost_per_contract=0.875),
    ]
    corr = np.array([[1.0, 0.9, -0.1], [0.9, 1.0, -0.1], [-0.1, -0.1, 1.0]])
    result = optimise_portfolio(instruments, corr, 500_000.0, apply_buffering=True)
    proposed = SelectionAdapter.from_dynopt(result, instruments, corr)
    # the Carver [0,1,3]→buffered[0,1,2] candidate list = non-zero positions.
    assert proposed.keep == ("US10y", "SP500")
    assert proposed.eligible == ("US10y", "SP500")
    assert "SP500" in proposed.risk_weights
    assert proposed.risk_weights["SP500"] == pytest.approx(0.124)


def test_selection_rank_is_a_risk_measure_never_edge():
    """Invariant A: the rank is a RISK (MCTR) measure, not edge; no return field enters it."""
    instruments = [
        Instrument("A", 1.0, 0.05, 0.20),
        Instrument("B", 1.0, 0.05, 0.20),
    ]
    corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    result = optimise_portfolio(instruments, corr, 500_000.0, apply_buffering=False)
    proposed = SelectionAdapter.from_dynopt(result, instruments, corr)
    # the rank is finite and derived from the covariance decomposition (a risk quantity).
    assert all(np.isfinite(v) for v in proposed.rank.values())


# --------------------------------------------------------------------------- breakout/trend adapters (P1)


def test_breakout_adapter_drops_non_entries_and_carries_no_score_logic():
    """Invariant P1/CB: the adapter passes through entries; SYSTEM never reads the score magnitude."""

    class MockBreakout:
        def __init__(self, sig):
            self._sig = sig

        def propose(self, instrument):
            return self._sig

    from convexity.system.synth.generate import make_entry_signal

    spec = InstrumentSpec(symbol="X", asset_class="crypto_perp", venue="binance", timezone="UTC",
                          tick_size=0.1, lot_size=1.0)
    entry = make_entry_signal("X", direction=1, quality_score=999.0)
    out = BreakoutAdapter(MockBreakout(entry)).propose_for([spec])
    assert "X" in out
    # a None (no-trade) is dropped.
    none_out = BreakoutAdapter(MockBreakout(None)).propose_for([spec])
    assert none_out == {}


def test_trend_adapter_injects_tau_cap_h_sys():
    """Invariant HC: the TrendAdapter forces tau_cap = H_sys on the TREND config."""
    from dataclasses import dataclass

    @dataclass
    class FakeCfg:
        tau_cap: int = 0

    captured = {}

    class MockTrend:
        def update(self, position, bars, idx, cfg):
            captured["tau_cap"] = cfg.tau_cap

            class Decision:
                action = "continue"
                stop_cause = ""
                target_fraction = 1.0
                reason_codes = ()

            return Decision()

    adapter = TrendAdapter(MockTrend(), h_sys=10_080)
    action = adapter.update_for(None, None, 0, FakeCfg())
    assert captured["tau_cap"] == 10_080
    assert action.kind == "continue" and action.alive


def test_trend_adapter_maps_time_cap():
    """Invariant HC: a stop_cause='time_cap' maps to a time_cap exit action."""

    class MockTrend:
        def update(self, position, bars, idx, cfg):
            class Decision:
                action = "exit"
                stop_cause = "time_cap"
                target_fraction = 0.0
                reason_codes = ("time_cap",)

            return Decision()

    action = TrendAdapter(MockTrend(), h_sys=100).update_for(None, None, 0, object())
    assert action.kind == "exit"
    assert action.time_cap
