"""Invariant HW — the minimum-history warmup is per-symbol, never panel-global (UNIVERSE.md §1.4, §11.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, UniverseConfig
from convexity.universe.eligibility.history_gate import base_eligible, history_warmup, tau_history
from convexity.universe.eligibility.stopping_times import in_tradeable_life
from research.xsec.panel import Panel


def _panel(listed: np.ndarray, symbols: tuple[str, ...]) -> Panel:
    n_days = listed.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n_days, freq="D"))
    close = np.where(listed, 100.0, np.nan)
    funding = np.where(listed, 0.0, np.nan)
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    return Panel(dates, symbols, close, funding, listed, first_day, last_day)


def test_history_gate_is_per_symbol_not_panel_global():
    """Inv HW (§1.4): a symbol listed exactly 30 trading days before ``t`` is EXCLUDED from ``U_t^base``
    at every ``t`` for which ``t - tau_list < L_max + w_warmup = 46``, regardless of how much history
    OTHER symbols carry.

    Fixture: OLD has decades of history from day 0; NEW lists 30 days before the test ``t``. NEW must be
    excluded (30 < 46) even though OLD (the panel's earliest symbol) has ~full history.
    """
    n_days = 500
    listed = np.ones((n_days, 2), dtype=bool)
    t = 400
    tau_list_new = t - 30
    listed[:tau_list_new, 1] = False  # NEW lists 30 trading days before t
    panel = _panel(listed, ("OLD", "NEW"))

    be = base_eligible(panel, t)
    j_old, j_new = 0, 1
    assert be[j_old], "OLD (full history) must be base-eligible"
    # 30 < 46 -> NEW is still inside its warmup window, excluded per-symbol.
    assert not be[j_new], "NEW listed 30d ago must be excluded (30 < L_max + w_warmup = 46)"

    # And exactly at the boundary: tau_list + 46 == the first eligible t.
    warm = history_warmup()
    assert warm == DEFAULT_UNIVERSE_CONFIG.l_max + DEFAULT_UNIVERSE_CONFIG.W == 46
    th = tau_history(panel)
    assert th[j_new] == tau_list_new + 46
    assert not base_eligible(panel, tau_list_new + 45)[j_new]
    assert base_eligible(panel, tau_list_new + 46)[j_new]


def test_history_gate_ignores_panel_global_earliest_date():
    """A NEW symbol's eligibility does not change when we lengthen OTHER symbols' history (§1.4 HW).

    Two panels differing only in OLD's listing depth must give NEW the SAME eligibility verdict — proof
    the gate reads ``tau_list_i`` per-symbol, never the panel-global first row.
    """
    n_days = 500
    t = 400
    tau_list_new = t - 40  # 40 < 46 -> still excluded
    # Panel 1: OLD from day 0. Panel 2: OLD from day 300 (much shallower panel-global history).
    for old_start in (0, 300):
        listed = np.ones((n_days, 2), dtype=bool)
        listed[:old_start, 0] = False
        listed[:tau_list_new, 1] = False
        panel = _panel(listed, ("OLD", "NEW"))
        assert not base_eligible(panel, t)[1], "NEW excluded regardless of OLD's history depth"


def test_conservative_fallback():
    """Cross-cutting (§2.4/§1.4 P6): a not-yet-warm name is EXCLUDED, never partially admitted.

    A name halfway through its warmup window is out of ``U_t^base`` — the gate never admits a name it
    would otherwise exclude on a partial window (P6 conservative-exclusion discipline).
    """
    cfg = UniverseConfig()
    n_days = 200
    t = 100
    tau_list_half = t - (cfg.l_max + cfg.W) // 2  # halfway through the 46-day warmup
    listed = np.ones((n_days, 1), dtype=bool)
    listed[:tau_list_half, 0] = False
    panel = _panel(listed, ("HALF",))
    assert not base_eligible(panel, t, cfg)[0], "halfway-warm name must be excluded, never partially admitted"


def test_in_tradeable_life_matches_listed_span():
    """§1.2 vocabulary: ``in_tradeable_life(panel, t)`` == ``1{ tau_list_i <= t < tau_delist_i }``, i.e.
    exactly the per-symbol live span. It is False before listing and False on/after the delist boundary,
    for each symbol independently. This is the mask ``base_eligible`` now delegates its delist bound to
    (auditor gap 4 — the function previously had zero callers)."""
    n_days = 50
    listed = np.ones((n_days, 2), dtype=bool)
    listed[:10, 0] = False  # A lists at day 10
    listed[40:, 1] = False  # B delists after day 39 (last live day 39)
    panel = _panel(listed, ("A", "B"))

    # A: not tradeable before its listing, tradeable from listing on.
    assert not in_tradeable_life(panel, 9)[0], "A not yet listed at t=9"
    assert in_tradeable_life(panel, 10)[0], "A tradeable at its listing day"
    # B: tradeable while live, not tradeable at/after the delist boundary (last live day = 39).
    assert in_tradeable_life(panel, 39)[1], "B tradeable on its last live day"
    assert not in_tradeable_life(panel, 40)[1], "B not tradeable at tau_delist (day 40)"


def test_in_tradeable_life_out_of_range_is_all_false():
    """§1.2: an out-of-range ``t`` (before the panel or past its end) yields all-False — no name is
    tradeable at a time outside the observed sample (conservative, F_t-measurable)."""
    listed = np.ones((20, 3), dtype=bool)
    panel = _panel(listed, ("A", "B", "C"))
    assert not in_tradeable_life(panel, -1).any()
    assert not in_tradeable_life(panel, 20).any()


def test_base_eligible_delegates_delist_bound_to_in_tradeable_life():
    """Inv (§1.4 + §1.2): ``base_eligible`` == ``(tau_history <= t) & in_tradeable_life(panel, t)`` for a
    panel that spans a full listing + warmup + delist lifecycle. Verifies the delegation the gap-4 fix
    introduced is EXACT (not an approximation): the warmup lower bound ANDed with the shared tradeable-life
    mask reproduces the gate's mask bit-for-bit at every t in the sample."""
    cfg = UniverseConfig()
    n_days = 120
    warm = cfg.l_max + cfg.w_warmup  # 46
    listed = np.ones((n_days, 1), dtype=bool)
    listed[:20, 0] = False  # lists day 20
    listed[100:, 0] = False  # delists after day 99
    panel = _panel(listed, ("X",))
    th = tau_history(panel, cfg)
    for t in range(0, n_days):
        expected = bool((th[0] <= t) and in_tradeable_life(panel, t)[0])
        assert bool(base_eligible(panel, t, cfg)[0]) == expected, f"mismatch at t={t}"
    # Sanity: the name is eligible only inside [tau_list+warm, tau_delist).
    assert base_eligible(panel, 20 + warm, cfg)[0]
    assert not base_eligible(panel, 20 + warm - 1, cfg)[0]
    assert not base_eligible(panel, 100, cfg)[0], "delisted name (t == tau_delist) is not base-eligible"
