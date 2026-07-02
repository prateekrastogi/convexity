"""Invariant SHORT — the short gate is a strict subset, never silently widened (UNIVERSE.md §4, §11.2)."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.universe.shortability.liquidity_gate import ShortLiquidityThreshold, short_liquid
from convexity.universe.types import EligibilitySnapshot
from convexity.universe.universe_api import build_shortability


class RankFloorThreshold:
    """A test-only INJECTED short-liquidity callable: require rank_i(t) >= a HIGHER percentile than q_in.

    This is a TEST fixture, not a spec default — UNIVERSE.md §4.2 forbids fabricating a numeric
    short-liquidity threshold in the shipped code (no short-borrow data exists). Injecting one here is
    exactly the contract §4.2 mandates: the day real data exists, calibration is supplying a callable.
    """

    def __init__(self, rank_floor: float):
        self.rank_floor = rank_floor

    def __call__(self, rank_t: np.ndarray, dv_t: np.ndarray) -> np.ndarray:
        return np.isfinite(rank_t) & (rank_t >= self.rank_floor)


def test_threshold_fn_is_a_protocol_instance():
    """§4.2: ShortLiquidityThreshold is an injected callable Protocol with no default."""
    thr = RankFloorThreshold(0.90)
    assert isinstance(thr, ShortLiquidityThreshold)


def test_short_subset_never_violated():
    """Inv SHORT (§4): ``build_shortability``'s OWN ``short_eligible`` field is a strict subset of ``U_t``
    (``short_eligible & ~universe`` is empty) EVEN WHEN the injected ``short_liquid`` test would admit names
    that sit OUTSIDE ``U_t``. This has teeth: the fixture deliberately makes ``short_liquid`` True for
    out-of-universe names, so if ``build_shortability`` dropped its ``& eligibility.universe`` term the
    assertion would fail. Asserts on the emitted ShortabilitySnapshot field, not a test-local
    re-composition (the §11.2 body's ``short_liquid(...) & ~hysteresis_state == 0`` intent, checked
    against the actual product not a tautology)."""
    from tests.universe.conftest import make_dvol, make_panel
    from convexity.universe.universe_api import build_shortability

    n_days = 120
    syms = tuple(f"S{i}" for i in range(20))
    n_sym = len(syms)
    listed = np.ones((n_days, n_sym), dtype=bool)
    # strictly-ordered constant dollar volume -> deterministic ranks (S0 least, S19 most liquid)
    dv_live = np.tile((np.arange(n_sym) + 1.0) * 1e8, (n_days, 1))
    panel = make_panel(np.full((n_days, n_sym), 100.0), listed, syms)
    dvol = make_dvol(dv_live, listed)

    t = 100
    from convexity.universe.liquidity.ranking import liquidity_rank

    rank = liquidity_rank(dvol[t], np.ones(n_sym, dtype=bool))
    # A universe that is only HALF the names (the top-10 by rank), so the bottom-10 are OUT of U_t.
    universe = rank >= 0.5
    snap = EligibilitySnapshot(
        t=t,
        base_eligible=np.ones(n_sym, dtype=bool),
        hysteresis_state=universe.copy(),
        rank=rank,
        universe=universe,
        n_hysteresis_entries=0,
        n_hysteresis_exits=0,
    )
    # A PERVERSE injected short test that passes EVERY finite-rank name — including the out-of-U_t bottom
    # half. If build_shortability failed to intersect with U_t, ``short_eligible`` would leak these names
    # and the subset assertion below would fail. This is the fixture that gives the invariant test teeth.
    class AdmitAll:
        def __call__(self, rank_t, dv_t):
            return np.isfinite(rank_t)

    ss = build_shortability(panel, dvol, snap, threshold_fn=AdmitAll(), k=1)
    assert int((ss.short_eligible & ~snap.universe).sum()) == 0, "U_t^short must be a subset of U_t"
    # And it is a PROPER subset here (the bottom half was admitted by the perverse test but gated out):
    assert bool((ss.short_liquid & ~snap.universe).any()), (
        "fixture must actually exercise the gate: short_liquid should admit out-of-U_t names so the "
        "intersection with U_t is doing real work (otherwise the subset assertion is vacuous)"
    )
    assert np.array_equal(ss.short_eligible, snap.universe), (
        "with AdmitAll and universe-only permission, short_eligible == U_t exactly"
    )


def test_short_starvation_flagged_not_silently_widened():
    """Inv SHORT (§4.4): when fewer than k tail names are short-eligible, the flag is raised AND the
    universe composition is UNCHANGED by the flag (UNIVERSE reports, never resolves)."""
    from tests.universe.conftest import make_dvol, make_panel

    n_days = 120
    syms = tuple(f"S{i}" for i in range(8))
    listed = np.ones((n_days, len(syms)), dtype=bool)
    # constant, strictly-ordered dollar volume so ranks are deterministic
    dv_live = np.tile((np.arange(len(syms)) + 1.0) * 1e8, (n_days, 1))
    panel = make_panel(np.full((n_days, len(syms)), 100.0), listed, syms)
    dvol = make_dvol(dv_live, listed)

    t = 100
    n_sym = len(syms)
    # Build a plausible EligibilitySnapshot directly: all names in U_t, ranks by DV order.
    from convexity.universe.liquidity.ranking import liquidity_rank

    rank = liquidity_rank(dvol[t], np.ones(n_sym, dtype=bool))
    universe = np.ones(n_sym, dtype=bool)
    snap = EligibilitySnapshot(
        t=t,
        base_eligible=np.ones(n_sym, dtype=bool),
        hysteresis_state=np.ones(n_sym, dtype=bool),
        rank=rank,
        universe=universe,
        n_hysteresis_entries=0,
        n_hysteresis_exits=0,
    )
    # A very strict short threshold: only the single MOST-liquid name is shortable -> the bottom tail is
    # entirely short-ineligible.
    thr = RankFloorThreshold(0.99)
    k = 3
    ss = build_shortability(panel, dvol, snap, threshold_fn=thr, k=k)
    assert ss.short_starved_flag is True, "starved short tail must raise the flag"
    assert ss.n_short_eligible_in_tail < k
    # Universe composition is UNCHANGED by the flag (§4.4, P2 provider-not-decider).
    assert np.array_equal(snap.universe, universe), "flag must not mutate U_t"
    # And U_t^short remains a strict subset of U_t.
    assert int((ss.short_eligible & ~snap.universe).sum()) == 0


def test_short_liquid_requires_injected_callable():
    """§4.2 disclosed limitation: no threshold is hard-coded; the callable must be injected."""
    rank = np.array([0.9, 0.8, 0.7, np.nan])
    dv = np.array([1e9, 8e8, 6e8, np.nan])
    with pytest.raises(NotImplementedError):
        short_liquid(rank, dv, threshold_fn=None)  # type: ignore[arg-type]


def test_conservative_fallback():
    """Cross-cutting (§2.4 P6): a NaN rank/DV can never be short-eligible, whatever the injected
    callable returns (a name that cannot be ranked fails the name closed)."""
    rank = np.array([np.nan, 0.95, 0.99])
    dv = np.array([1e9, np.nan, 5e8])

    class AlwaysTrue:
        def __call__(self, rank_t, dv_t):
            return np.ones(rank_t.shape, dtype=bool)

    liq = short_liquid(rank, dv, threshold_fn=AlwaysTrue())
    assert not liq[0], "NaN rank -> not short-liquid regardless of callable (P6)"
    assert not liq[1], "NaN DV -> not short-liquid regardless of callable (P6)"
    assert liq[2], "a valid rank+DV passes the always-true callable"


def test_stale_name_gets_floored_dv_inside_build_shortability():
    """Auditor gap 2 (§2.4 consistency): a name the long liquidity gate declared STALE (fewer than
    ``ceil(W * min_valid_frac)`` valid trailing days) must ALSO be treated as DV-invalid inside
    ``build_shortability`` — so the short-liquidity absolute-floor component can never see an un-floored
    ``DV_i(d)`` for a name UNIVERSE already declared unrankable. With DV NaN-floored, ``short_liquid``'s
    ``isfinite(dv_t)`` gate (P6) closes the name regardless of the injected callable.

    Fixture: symbol STALE has an all-NaN trailing dollar-volume window (only 0 valid days << the floor),
    so §2.4 forces its rank to 0 upstream AND its DV must be floored to NaN here. Under a perverse
    AdmitAll threshold it would leak into ``short_eligible`` if the gap-2 floor were absent."""
    from tests.universe.conftest import make_dvol, make_panel

    n_days = 120
    syms = ("LIVE0", "LIVE1", "STALE")
    n_sym = len(syms)
    listed = np.ones((n_days, n_sym), dtype=bool)
    dv_live = np.tile(np.array([3e8, 2e8, 1e8]), (n_days, 1))
    panel = make_panel(np.full((n_days, n_sym), 100.0), listed, syms)
    dvol = make_dvol(dv_live, listed)

    t = 100
    # Blank MOST of STALE's trailing W-window, leaving only a FEW valid days: below the ceil(W/2)=15
    # floor (§2.4) yet >=1 valid day, so trailing_dollar_volume (nanmean) still returns a FINITE DV.
    # This is the case where ONLY the gap-2 floor closes the name -- an all-NaN window would already be
    # NaN without any floor and would not exercise the fix.
    from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG as CFG

    dvol[t - CFG.W + 1 : t - 2, 2] = np.nan  # leave 3 valid days at the window end (< ceil(30/2)=15)

    # Build the EligibilitySnapshot exactly as the API would, so rank reflects the §2.4 floor (STALE->0).
    from convexity.universe.liquidity.dollar_volume import dollar_volume_at, valid_day_count
    from convexity.universe.liquidity.ranking import liquidity_rank

    dv_t = dollar_volume_at(panel, dvol, t, CFG)
    vdays = valid_day_count(dvol, t, CFG)
    rank = liquidity_rank(dv_t, np.ones(n_sym, dtype=bool), valid_days=vdays, cfg=CFG)
    assert rank[2] == 0.0, "STALE must be forced to rank 0 by §2.4 upstream"
    assert np.isfinite(dv_t[2]), (
        "fixture check: STALE's raw trailing DV is FINITE (nanmean over its few valid days); only the "
        "gap-2 §2.4 floor NaN-nulls it inside build_shortability"
    )

    snap = EligibilitySnapshot(
        t=t,
        base_eligible=np.ones(n_sym, dtype=bool),
        hysteresis_state=np.ones(n_sym, dtype=bool),
        rank=rank,
        universe=np.ones(n_sym, dtype=bool),  # STALE is (perversely) forced into U_t to isolate the gate
        n_hysteresis_entries=0,
        n_hysteresis_exits=0,
    )

    class AdmitAll:
        def __call__(self, rank_t, dv_t):
            return np.ones(rank_t.shape, dtype=bool)  # would admit even a rank-0 / NaN-DV name

    ss = build_shortability(panel, dvol, snap, threshold_fn=AdmitAll(), k=1)
    assert not ss.short_liquid[2], (
        "a §2.4-stale name must NOT be short-liquid: its DV is floored to NaN inside build_shortability, "
        "so short_liquid's isfinite(dv_t) gate closes it regardless of the injected callable (gap 2)"
    )
    assert not ss.short_eligible[2], "and therefore never short-eligible"
    # The two well-populated names are unaffected by the floor.
    assert ss.short_liquid[0] and ss.short_liquid[1]


def test_short_permitted_honors_venue_metadata_dict():
    """The AMBER short_permitted gate keys off the supplied venue-metadata dict (venue_gate.py:43).

    When permitted_by_symbol is None every symbol defaults to True (the conservative path); when a
    dict is supplied a symbol flagged False is not short-permitted, and one absent from the dict
    still defaults to True (§4.2). This binds the dict-lookup branch the None-default path skips.
    """
    from tests.universe.conftest import make_panel
    from convexity.universe.shortability.venue_gate import short_permitted

    syms = ("A", "B", "C")
    close = np.full((40, 3), 100.0)
    listed = np.ones((40, 3), dtype=bool)
    panel = make_panel(close, listed, syms)
    default = short_permitted(panel, 30)
    assert default.tolist() == [True, True, True]  # None -> all permitted (conservative)

    flagged = short_permitted(panel, 30, permitted_by_symbol={"A": False})
    assert flagged.tolist() == [False, True, True]  # A blocked; B absent -> defaults True; C True
