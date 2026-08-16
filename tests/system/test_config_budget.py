"""SYSTEM's AMBER universe wrapper and the overnight compute budget (§9.2, Invariant CB-BUDGET).

The budget is the thing that stops an overnight validation job being enqueued that cannot finish
by morning — a job that silently overruns burns a night of research time and, worse, tempts a
half-finished result into a decision. Both modules were at 0%.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from convexity.core.types import InstrumentSpec
from convexity.system.config.universe import SystemUniverse, load_universe_meta
from convexity.system.validate.budget import OvernightBudget


@dataclass(frozen=True)
class Defaults:
    """A stand-in for SystemDefaults carrying only the fields the budget model reads."""

    n_groups: int = 6
    k_test: int = 2
    max_paths: int | None = None
    t_mod_s: float = 1.0
    t_comp_s: float = 0.5
    t_night_s: float = 10_000.0
    k_max: int = 100
    max_distinct_module_configs: int = 50


def spec(symbol: str, lot: float = 0.001) -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol, asset_class="crypto", venue="BINANCE", timezone="UTC",
        tick_size=0.01, lot_size=lot, price_multiplier=1.0, min_notional=5.0,
    )


# ---------------------------------------------------------------- universe
def test_universe_symbols_are_sorted_for_deterministic_tie_breaks():
    """Section 3.3 — the total order is what makes tie-breaks reproducible run to run."""
    u = load_universe_meta([spec("ZUSDT"), spec("AUSDT"), spec("MUSDT")])
    assert u.symbols == ("AUSDT", "MUSDT", "ZUSDT")


def test_universe_exposes_specs_and_lot_size_by_symbol():
    u = load_universe_meta([spec("AUSDT", lot=0.25)])
    assert u.spec("AUSDT").symbol == "AUSDT"
    assert u.lot_size("AUSDT") == pytest.approx(0.25)


def test_an_unknown_symbol_raises_rather_than_returning_a_default():
    """A silently-defaulted lot size would quantize orders on a grid the venue never agreed to."""
    u = load_universe_meta([spec("AUSDT")])
    with pytest.raises(KeyError):
        u.lot_size("NOPEUSDT")


def test_an_empty_universe_is_representable():
    u = SystemUniverse(specs={})
    assert u.symbols == ()


# ---------------------------------------------------------------- budget model
def test_paths_are_the_full_combination_count_by_default():
    est = OvernightBudget(Defaults()).estimate(k=1, distinct_module_configs=1)
    assert est.paths == 15                       # C(6,2)


def test_max_paths_caps_the_scored_paths():
    est = OvernightBudget(Defaults(max_paths=4)).estimate(k=1, distinct_module_configs=1)
    assert est.paths == 4


def test_cost_decomposes_into_evidence_compose_and_attribution():
    d = Defaults()
    est = OvernightBudget(d).estimate(k=2, distinct_module_configs=3)
    assert est.cost_evidence_s == pytest.approx(3 * 15 * 1.0)
    assert est.cost_compose_s == pytest.approx(2 * 15 * 0.5)
    assert est.cost_attrib_s == pytest.approx(5 * 15 * (1.0 + 0.5))   # sequential = 5 coalitions
    assert est.total_s == pytest.approx(
        est.cost_evidence_s + est.cost_compose_s + est.cost_attrib_s)


def test_shapley_attribution_costs_sixteen_coalitions_not_five():
    """The honest-Shapley requirement has a price, and the budget must charge it."""
    d = Defaults()
    seq = OvernightBudget(d).estimate(k=1, distinct_module_configs=1)
    shap = OvernightBudget(d).estimate(k=1, distinct_module_configs=1,
                                       attribution_mode="shapley")
    assert shap.cost_attrib_s == pytest.approx(seq.cost_attrib_s * 16 / 5)
    assert shap.total_s > seq.total_s


def test_a_job_that_fits_is_admitted_with_budget_ok():
    b = OvernightBudget(Defaults())
    est = b.estimate(k=2, distinct_module_configs=2)
    assert est.fits and est.within_cap
    assert est.reason_codes == ("budget_ok",)
    assert b.admits(k=2, distinct_module_configs=2)


def test_exceeding_k_max_is_refused_and_says_so():
    """Invariant CB-BUDGET: K_max + 1 configs must be rejected."""
    b = OvernightBudget(Defaults(k_max=3))
    est = b.estimate(k=4, distinct_module_configs=1)
    assert not est.within_cap
    assert "exceeds_k_max" in est.reason_codes
    assert not b.admits(k=4, distinct_module_configs=1)


def test_exceeding_the_module_config_cap_is_reported_distinctly():
    b = OvernightBudget(Defaults(max_distinct_module_configs=2))
    est = b.estimate(k=1, distinct_module_configs=5)
    assert not est.within_cap
    assert "exceeds_module_config_cap" in est.reason_codes


def test_a_job_that_cannot_finish_by_morning_is_refused():
    b = OvernightBudget(Defaults(t_night_s=1.0))
    est = b.estimate(k=10, distinct_module_configs=10)
    assert not est.fits
    assert "exceeds_t_night" in est.reason_codes
    assert not b.admits(k=10, distinct_module_configs=10)


def test_both_failures_are_reported_together_not_just_the_first():
    """An operator resizing a job needs to see every reason at once."""
    b = OvernightBudget(Defaults(k_max=1, t_night_s=1.0))
    est = b.estimate(k=50, distinct_module_configs=50)
    assert "exceeds_k_max" in est.reason_codes
    assert "exceeds_t_night" in est.reason_codes
    assert "budget_ok" not in est.reason_codes


def test_estimate_never_raises_so_the_caller_can_inspect_before_enqueuing():
    """The contract: estimate() reports, admits() decides — an exception here would deny the
    caller the diagnosis."""
    est = OvernightBudget(Defaults(k_max=0, t_night_s=0.0)).estimate(
        k=9999, distinct_module_configs=9999)
    assert est.total_s > 0
    assert not est.fits and not est.within_cap
