"""Invariant PIT — the master structural point-in-time check (UNIVERSE.md §7, §9, §11.2)."""

from __future__ import annotations

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG
from convexity.universe.types import EligibilitySnapshot
from convexity.universe.universe_api import build_eligibility
from convexity.universe.validate.pit_check import corrupt_future, pit_check
from convexity.panel.panel import build_dollar_volume  # noqa: F401  (import-surface sanity)


def _make_eligibility_fn(rebalances):
    """Return a Callable[[Panel, int], EligibilitySnapshot] that builds dvol FROM the panel it is given.

    Building dvol from the SAME (possibly corrupted) panel is what makes the PIT test meaningful: if the
    construction leaked a future row, the corrupted-panel dvol/close would differ and U_t would change.
    We derive a synthetic dvol from close (close*const) rather than the lake so the test is hermetic.
    """

    def _dvol_from_panel(panel):
        # A deterministic dollar-volume proxy from the panel's own close, masked off-span exactly like
        # convexity.panel.panel.build_dollar_volume's post-condition (np.where(listed, ., nan)).
        base = np.nan_to_num(panel.close, nan=0.0) * 1.0e6
        return np.where(panel.listed, base, np.nan)

    def eligibility_fn(panel, t) -> EligibilitySnapshot:
        dvol = _dvol_from_panel(panel)
        return build_eligibility(panel, t, dvol=dvol, rebalances=rebalances, cfg=DEFAULT_UNIVERSE_CONFIG)

    return eligibility_fn


def test_future_corruption_does_not_change_past_universe(liquid_panel):
    """Inv PIT (§7): U_t is bit-identical whether computed on the full panel or on a copy with every row
    s > t garbage-overwritten, for a sample of t spanning the window."""
    panel, _ = liquid_panel
    rebalances = list(range(0, panel.n_days, 5))
    fn = _make_eligibility_fn(rebalances)
    sample_t = [r for r in rebalances if r >= 50][:: max(1, len(rebalances) // 8)]
    assert sample_t, "need at least one testable rebalance"
    for t in sample_t:
        assert pit_check(panel, t, eligibility_fn=fn) is True, f"U_t changed under future corruption at t={t}"


def test_corrupt_future_leaves_past_rows_untouched(liquid_panel):
    """The corruption helper only touches rows s > t (rows <= t are bit-identical) — so the test isolates
    a genuine forward-read leak, not incidental past-row damage."""
    panel, _ = liquid_panel
    t = 100
    corrupted = corrupt_future(panel, t, seed=0)
    assert np.array_equal(
        np.nan_to_num(panel.close[: t + 1], nan=-1.0),
        np.nan_to_num(corrupted.close[: t + 1], nan=-1.0),
    ), "rows <= t must be untouched by corruption"
    assert np.array_equal(panel.listed[: t + 1], corrupted.listed[: t + 1])


def test_pit_detects_a_planted_forward_read(liquid_panel):
    """A construction that DELIBERATELY reads a future row (t+1) must fail pit_check — proving the test
    has teeth (a no-op assertion would pass even a leaky construction)."""
    panel, _ = liquid_panel
    rebalances = list(range(0, panel.n_days, 5))
    good_fn = _make_eligibility_fn(rebalances)

    def leaky_fn(p, t) -> EligibilitySnapshot:
        snap = good_fn(p, t)
        # Plant a forward read: set one symbol's membership from a FUTURE property. Symbol A (col 0) is
        # live for the whole panel, so on the REAL panel no future close for A is NaN; the corrupted
        # panel NaN-poisons ~30% of future rows. So this bit is deterministically False on the real
        # panel and True on the corrupted one -> a genuine leak PIT must catch.
        if t + 1 < p.n_days:
            universe = snap.universe.copy()
            fut = p.close[t + 1 :, 0]
            universe[0] = bool(np.isnan(fut).any())
            snap = EligibilitySnapshot(
                t=snap.t,
                base_eligible=snap.base_eligible,
                hysteresis_state=snap.hysteresis_state,
                rank=snap.rank,
                universe=universe,
                n_hysteresis_entries=snap.n_hysteresis_entries,
                n_hysteresis_exits=snap.n_hysteresis_exits,
            )
        return snap

    assert pit_check(panel, 100, eligibility_fn=leaky_fn) is False, "a planted forward read must fail PIT"


def test_conservative_fallback(liquid_panel):
    """Cross-cutting (§7): the check is a bit-identity, not a statistical tolerance — even a single
    changed membership bit fails it (no averaging that could mask a stale/leaked cell)."""
    panel, _ = liquid_panel
    rebalances = list(range(0, panel.n_days, 5))
    fn = _make_eligibility_fn(rebalances)
    # A correct construction is bit-identical across independent corruption seeds.
    for seed in (0, 1, 7):
        assert pit_check(panel, 90, eligibility_fn=fn, seed=seed) is True
