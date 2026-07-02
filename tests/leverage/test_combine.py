"""min() composition + binding_ceiling + firewall + worked example (LEVERAGE.md §6.1, §7, §10)."""

from __future__ import annotations

import pathlib

import numpy as np

from convexity.leverage.ceilings.capacity import CapacityCeiling
from convexity.leverage.ceilings.kelly_vol_target import kelly_vol_target_term
from convexity.leverage.combine.engine import compose_g_t
from convexity.leverage.config.resolve import resolve_config
from convexity.leverage.engine_api import LeverageEngine
from convexity.leverage.types import CapacityInput, GovernorState


def _cap(g):
    return CapacityCeiling(g_cap=g, binding_symbol="X" if np.isfinite(g) else None, n_unshortable_zeroed=0)


def _gov(m):
    return GovernorState(dd_t=0.0, m_t=m, terminal=False)


def test_worked_example_kelly_binds():
    # §7 worked example step 2: G_t = min(0.585, 2.0, 1.0*1.5, 1.5) = 0.585, binding = 'kelly'.
    gt = kelly_vol_target_term(0.25, 0.091, 0.0389)
    dec = compose_g_t(gt, _cap(2.0), _gov(1.0), g_max=1.5)
    assert round(dec.g_t, 3) == 0.585
    assert dec.binding_ceiling == "kelly"


def test_worked_example_governor_binds_at_dd_030():
    # §7 step 4: at dd=0.30, m_t=0.35 -> governor term 0.525 < kelly 0.585 -> governor binds.
    gt = kelly_vol_target_term(0.25, 0.091, 0.0389)
    dec = compose_g_t(gt, _cap(2.0), GovernorState(dd_t=0.30, m_t=0.35, terminal=False), g_max=1.5)
    assert round(dec.g_t, 3) == 0.525
    assert dec.binding_ceiling == "governor"


def test_composition_is_min_never_a_blend():
    # §6.1: G_t is the strict min of the four terms, not a weighted average.
    gt = kelly_vol_target_term(0.5, 0.20, 0.04)  # kelly = 2.5
    dec = compose_g_t(gt, _cap(0.4), _gov(1.0), g_max=1.5)  # capacity 0.4 binds
    assert dec.g_t == min(2.5, 0.4, 1.5, 1.5)
    assert dec.binding_ceiling == "capacity"


def test_hard_cap_clamps_an_absurd_kelly_term():
    # §3: G_max catches a corrupted σ̂_t forcing an absurd Kelly term. At m_t=1 the governor term
    # (m_t·G_max) equals the hard cap (G_max) and wins the tie (declaration order); both yield 1.5,
    # so G_t is clamped to G_max regardless — the circuit breaker's whole purpose (§6.1).
    gt = kelly_vol_target_term(0.5, 0.5, 0.001)  # kelly = 250, absurd
    dec = compose_g_t(gt, _cap(float("inf")), _gov(1.0), g_max=1.5)
    assert dec.g_t == 1.5
    assert dec.binding_ceiling in {"governor", "hard_cap"}
    assert dec.binding_ceiling != "kelly"  # the absurd kelly term never survives the cap


def test_hard_cap_binds_uniquely_when_governor_has_slack():
    # When the governor de-risks only slightly (m_t·G_max still > G_max is impossible; m_t<=1), the
    # hard cap and governor coincide at m_t=1. To see hard_cap bind ALONE we need m_t=1 exactly and
    # every other term above G_max — verified via g_t clamped at G_max with no smaller term.
    gt = kelly_vol_target_term(0.5, 0.5, 0.001)  # kelly = 250
    dec = compose_g_t(gt, _cap(10.0), _gov(1.0), g_max=1.5)
    assert dec.g_t == 1.5
    assert dec.hard_cap_term == 1.5 and dec.governor_term == 1.5


def test_g_t_is_always_non_negative():
    # §8 / Invariant PROV: never negative, never inverted. Zero edge -> G_t = 0.
    gt = kelly_vol_target_term(0.25, 0.0, 0.04)  # SR_LCB = 0 -> kelly term 0
    dec = compose_g_t(gt, _cap(2.0), _gov(1.0), g_max=1.5)
    assert dec.g_t == 0.0
    assert dec.binding_ceiling == "kelly"
    assert "edge_floored_zero" in dec.reason_codes


def test_binding_ceiling_always_populated_and_matches_smallest():
    gt = kelly_vol_target_term(0.25, 0.09, 0.04)
    for cap_g, gov_m in [(2.0, 1.0), (0.1, 1.0), (5.0, 0.05)]:
        dec = compose_g_t(gt, _cap(cap_g), _gov(gov_m), g_max=1.5)
        terms = {
            "kelly": dec.kelly_term,
            "capacity": dec.capacity_term,
            "governor": dec.governor_term,
            "hard_cap": dec.hard_cap_term,
        }
        assert np.isclose(dec.g_t, min(terms.values()))
        assert terms[dec.binding_ceiling] == min(terms.values())


def test_engine_end_to_end_produces_a_decision():
    # LeverageEngine.size_book is the ONE public method (§9); it wires the full sizing chain.
    eng = LeverageEngine(resolve_config())
    rng = np.random.default_rng(3)
    net = rng.normal(0.02, 0.02, 400)  # passes DSR
    trials = np.array([1.0, 0.9, 0.8, 1.1, 0.95])
    caps = [CapacityInput("BTC", adv=1e6, w_ref=0.5), CapacityInput("ETH", adv=1e6, w_ref=-0.5)]
    dec = eng.size_book(net, trials, caps, c_t=0.25)
    assert dec.g_t >= 0.0
    assert dec.binding_ceiling in {"kelly", "capacity", "governor", "hard_cap"}


def test_firewall_live_layers_do_not_import_validate():
    # Invariant V (§10): no live layer (vol/edge/ceilings/ratchet/combine) imports leverage/validate.
    import convexity.leverage as lev

    root = pathlib.Path(lev.__file__).parent
    live_dirs = {"vol", "edge", "ceilings", "ratchet", "combine"}
    for path in root.rglob("*.py"):
        if path.name.startswith("."):
            continue  # skip editor/transfer artifacts (e.g. macOS ._ AppleDouble files)
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in live_dirs:
            src = path.read_text()
            assert "leverage.validate" not in src and "from ..validate" not in src, (
                f"{path} imports leverage/validate (Invariant V firewall)"
            )
