"""End-to-end wiring of the UNIVERSE contract layer over the panel substrate (UNIVERSE.md §1-4)."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.universe.config.defaults import UniverseConfig
from convexity.universe.universe_api import build_book, build_eligibility


def test_book_carries_hysteresis_state_forward(liquid_panel):
    """§2.2: the FSM state at t is the previous rebalance's returned state (one-step memory), never
    recomputed from scratch — build_book must thread it through the calendar."""
    panel, dvol = liquid_panel
    # small N so the size cutoff binds and membership is interesting
    cfg = UniverseConfig(N=3)
    rebalances = list(range(50, panel.n_days, 5))
    book = build_book(panel, dvol, rebalances, cfg)
    assert len(book) == len(rebalances)
    for snap in book:
        # U_t is a subset of the IN (hysteresis) set, which is a subset of base-eligible.
        assert np.all(snap.universe <= snap.hysteresis_state)
        assert np.all(snap.hysteresis_state <= snap.base_eligible | snap.hysteresis_state)
        assert int(snap.universe.sum()) <= cfg.N


def test_late_lister_only_enters_after_warmup(liquid_panel):
    """§1.4 + §2.2: symbol E (lists day 120) cannot be in U_t before day 120 + 46 = 166, and only after
    it also clears the q_in liquidity entry."""
    panel, dvol = liquid_panel
    j_E = panel.symbols.index("E")
    cfg = UniverseConfig(N=6)  # keep all names so only the warmup + hysteresis gate E
    rebalances = list(range(120, panel.n_days))
    book = build_book(panel, dvol, rebalances, cfg)
    for snap in book:
        if snap.t < 120 + 46:
            assert not snap.base_eligible[j_E], "E excluded before its own warmup completes"
            assert not snap.universe[j_E]


def test_single_rebalance_matches_book_replay(liquid_panel):
    """build_eligibility(t) (causal replay for the PIT test) must equal the last snapshot of build_book
    over the same calendar — the two entry points share the FSM, no divergent path."""
    panel, dvol = liquid_panel
    cfg = UniverseConfig(N=4)
    rebalances = list(range(50, 121, 5))
    book = build_book(panel, dvol, rebalances, cfg)
    t = rebalances[-1]
    single = build_eligibility(panel, t, dvol=dvol, rebalances=rebalances, cfg=cfg)
    assert np.array_equal(single.universe, book[-1].universe)
    assert np.array_equal(single.hysteresis_state, book[-1].hysteresis_state)


def test_mass_exit_diagnostic_surfaces(liquid_panel):
    """§9 edge 8: n_hysteresis_exits is a per-rebalance diagnostic. A rebalance where many names' ranks
    collapse (here: we blank all dollar volume after a cutoff) spikes the exit count — surfaced, not
    smoothed."""
    panel, dvol = liquid_panel
    cfg = UniverseConfig(N=6)
    dvol2 = dvol.copy()
    cutoff = 150
    dvol2[cutoff:, :] = np.nan  # mass liquidity/data outage from `cutoff` onward
    # The second rebalance is placed a FULL window (W days) past the outage so the entire trailing
    # W-window is NaN -> every name falls to rank 0 (§2.4), tripping a mass hysteresis exit. (A partial
    # outage window would still have >= ceil(W/2) valid days and NOT collapse -- exactly §2.4's rule.)
    rebalances = [cutoff - 5, cutoff + cfg.W + 5]
    book = build_book(panel, dvol2, rebalances, cfg)
    assert int(book[0].universe.sum()) >= 1, "some names IN before the outage (sanity)"
    # After the full-window outage, previously-IN names cross to OUT (rank 0 < q_out) -> spike in exits.
    assert book[1].n_hysteresis_exits >= 1, "a mass liquidity outage surfaces as a hysteresis-exit spike"
    # And no smoothing/capping: every affected name genuinely exits (universe shrinks, §9 edge 8).
    assert int(book[1].universe.sum()) < int(book[0].universe.sum())


def test_build_eligibility_requires_explicit_calendar(liquid_panel):
    """No implicit daily [0, t] default (UNIVERSE.md §11 build-contract note): build_eligibility MUST be
    given the weekly rebalance calendar. A silent daily replay would (a) be O(t) per weekly rebalance on
    the ~1000-day panel and (b) evaluate the memoryful FSM on a different calendar than the book. This
    test has teeth: it would FAIL (no exception raised, or a wrong-calendar U_t returned) if the old
    range(0, t+1) default were restored.
    """
    panel, dvol = liquid_panel
    cfg = UniverseConfig(N=4)
    t = 100
    # Omitting rebalances must raise, not silently fall back to a daily replay.
    with pytest.raises(ValueError, match="rebalance calendar"):
        build_eligibility(panel, t, dvol=dvol, cfg=cfg)
    # And the weekly-calendar replay must equal the weekly book (the ONLY on-contract path). This is the
    # value the removed daily default would have SILENTLY diverged from.
    weekly = list(range(50, t + 1, 5))
    single = build_eligibility(panel, t, dvol=dvol, rebalances=weekly, cfg=cfg)
    book = build_book(panel, dvol, weekly, cfg)
    assert np.array_equal(single.universe, book[-1].universe)
