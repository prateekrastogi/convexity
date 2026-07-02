"""End-to-end integration test for the full 9-pillar weekly pipeline (scripts/xsec_pipeline.py).

Lake-gated (runs only where the real crypto lake exists, i.e. arb-vm) exactly like COST's
real-lake e2e (tests/research/xsec/cost/test_cost_validate.py). It drives ``run_pipeline`` on a
real symbol slice and asserts the shared composition wiring actually fires EVERY pillar and
produces a coherent weekly book — the integration TODOs the pillar implementers each left
(UNIVERSE->SIGNAL->PORTFOLIO->LEVERAGE->REGIME->EXECUTION->COST->SYSTEM->OPS) are exercised in one
runnable path, not just unit-tested in isolation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]  # tests/integration/<file> -> repo root
_LAKE_DIR = _REPO / "data" / "stitched"
_HAS_LAKE = _LAKE_DIR.is_dir()


def _load_pipeline():
    """Import scripts/xsec_pipeline.py as a module (scripts/ is not a package)."""
    path = _REPO / "scripts" / "xsec_pipeline.py"
    spec = importlib.util.spec_from_file_location("xsec_pipeline", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _HAS_LAKE, reason="requires the data/stitched crypto lake (arb-vm only)")
def test_full_pipeline_composes_all_nine_pillars_and_produces_a_weekly_book() -> None:
    """One real-data pass fires every pillar and yields a coherent, invariant-respecting book."""
    mod = _load_pipeline()
    out = mod.run_pipeline(n_symbols=20, verbose=False)

    # It ACTUALLY produced a weekly book on real data (PORTFOLIO.run_ensemble over ~5yr).
    assert out["n_weeks"] > 200, "the ensemble must produce a real multi-year weekly book"
    assert out["universe_last"] > 0, "UNIVERSE eligibility FSM must admit at least one name"

    # SIGNAL: the m=3 lookback ensemble ran all three canonical members {12,14,16}.
    assert set(out["per_l_sharpe"]) == {12, 14, 16}, "SIGNAL must run the exact {12,14,16} band"

    # LEVERAGE: G_t is a finite positive gross in a sane range, and the min-composition named a
    # binding ceiling (kelly at this book size — capacity/governor never bind on ~$M ADV names).
    assert 0.0 < out["G_t"] < 3.0
    assert out["binding_ceiling"]

    # REGIME: the CUSUM detector produced a causal state path that EXERCISES more than one state
    # (HEALTHY plus at least one escalation), and g_t is the single [0,1] de-risk scalar.
    assert 0.0 <= out["g_t"] <= 1.0
    assert len(out["regime_states"]) >= 2, "the detector must exercise >1 market state on real data"
    assert set(out["regime_states"]) <= {"HEALTHY", "WATCH", "STRESSED", "HALT"}

    # ARCHITECTURE §2 single-call-site gross invariant: gross_live == g_t * G_t (never a double cut).
    assert out["gross_live"] == pytest.approx(out["g_t"] * out["G_t"], rel=1e-9)

    # EXECUTION Invariant E-monotone (P4): the no-trade band only SKIPS trades, never adds one.
    assert out["turnover_live"] <= out["turnover_backtest"] + 1e-12
    assert out["n_trajectories"] >= 0 and out["n_ops_obs"] >= 0

    # COST friction-stress produced a valid frac_pos and SYSTEM ran the full DSR/CPCV/PROMOTABLE gate.
    assert 0.0 <= out["cost_stress_frac_pos"] <= 1.0
    assert 0.0 <= out["cpcv_frac_pos"] <= 1.0
    assert 0.0 <= out["pbo"] <= 1.0
    assert isinstance(out["promotable"], bool)

    # OPS: the kill-switch folded every settled weekly return (RUN on the calibrated null) and the
    # forced-HALT path (heartbeat silence) latches HALT (the kill-switch codepath actually fires).
    assert out["ops_final_state"] in {"RUN", "DEGRADE", "HALT"}
    assert out["ops_halt_on_heartbeat"] == "HALT", "heartbeat silence must force an OPS HALT"


@pytest.mark.skipif(not _HAS_LAKE, reason="requires the data/stitched crypto lake (arb-vm only)")
def test_pipeline_is_deterministic_across_two_runs() -> None:
    """The whole composition is a pure function of the frozen lake + seeds (no hidden RNG drift)."""
    mod = _load_pipeline()
    a = mod.run_pipeline(n_symbols=20, verbose=False)
    b = mod.run_pipeline(n_symbols=20, verbose=False)
    assert a["ensemble_sharpe"] == pytest.approx(b["ensemble_sharpe"], rel=1e-12)
    assert a["G_t"] == pytest.approx(b["G_t"], rel=1e-12)
    assert a["regime_states"] == b["regime_states"]
    assert a["promotable"] == b["promotable"]
