"""Invariant PIT — the master structural point-in-time check (UNIVERSE.md §7, §9 item 9).

Per §7's explicit warning against circular validation: no downstream Sharpe/CPCV number can CERTIFY
§1's point-in-time property (a survivorship leak is invisible to every fold-based check by construction,
§1.3 corollary) — they can only be consistent or inconsistent with it. The ONLY direct check is
structural: does the eligibility computation at ``t``, given a dataset with all ``s > t`` rows corrupted,
produce a BIT-IDENTICAL ``U_t``?

    Invariant PIT (§9): for every rebalance ``t``, ``U_t`` and ``U_t^short`` are unchanged under ANY
    corruption/deletion of data at timestamps ``s > t``.

Methodology mirrors COST.md's Invariant W, re-aimed at universe construction: compute the eligibility
snapshot twice — once against the full panel, once against a copy with every row ``s > t`` overwritten
with garbage (NaN-poisoned or randomized) close/volume/listed — and assert bit-identical
``EligibilitySnapshot.universe`` (and, when a short field is supplied, ``U_t^short``). A construction that
reads any index ``> t`` (an off-by-one in day-bucketing, a forward-windowing ``.rolling()``) fails this
immediately, with no realized P&L needed (§7).
"""

from __future__ import annotations

import numpy as np

from convexity.universe.types import EligibilitySnapshot
from convexity.panel.panel import Panel


def corrupt_future(panel: Panel, t: int, *, seed: int = 0) -> Panel:
    """Return a copy of ``panel`` with every row ``s > t`` overwritten with garbage (UNIVERSE.md §7).

    Close/funding at ``s > t`` are NaN-poisoned then randomized; ``listed`` at ``s > t`` is flipped to
    a random mask; ``first_day`` is preserved (it is ``<= t`` for any name live by ``t``) while
    ``last_day`` is recomputed from the corrupted mask so the corrupted panel is internally consistent
    (a construction that reads ``last_day`` for names live at ``t`` would then see garbage — which is
    exactly the leak PIT must catch). Rows ``<= t`` are untouched.
    """
    rng = np.random.default_rng(seed)
    close = panel.close.copy()
    funding = panel.funding.copy()
    listed = panel.listed.copy()
    n_days = panel.n_days
    if t + 1 < n_days:
        fut = slice(t + 1, n_days)
        shape = close[fut].shape
        # Garbage: random finite values AND random NaNs, random listed flips — anything non-causal reads it.
        close[fut] = rng.normal(1e9, 1e9, size=shape)
        close[fut][rng.random(size=shape) < 0.3] = np.nan
        funding[fut] = rng.normal(0.0, 1.0, size=shape)
        listed[fut] = rng.random(size=shape) < 0.5

    first_day: dict[str, int] = {}
    last_day: dict[str, int] = {}
    for j, sym in enumerate(panel.symbols):
        live = np.flatnonzero(listed[:, j])
        # Preserve the causal first_day (it is <= t for names live by t); if the corruption emptied a
        # column entirely, fall back to the original span so the copy stays constructible.
        first_day[sym] = int(live[0]) if live.size else panel.first_day[sym]
        last_day[sym] = int(live[-1]) if live.size else panel.last_day[sym]
    return Panel(
        dates=panel.dates,
        symbols=panel.symbols,
        close=close,
        funding=funding,
        listed=listed,
        first_day=first_day,
        last_day=last_day,
    )


def pit_check(
    panel_full: Panel,
    t: int,
    *,
    eligibility_fn,
    seed: int = 0,
) -> bool:
    """Invariant PIT structural test (UNIVERSE.md §7, §9 item 9, Invariant PIT).

    ``panel_full`` -- the real historical panel (``research/xsec/panel.py``'s ``Panel``).
    ``t`` -- the rebalance index under test.
    ``eligibility_fn`` -- ``Callable[[Panel, int], EligibilitySnapshot]``: the construction under test
                          (a composition of the history gate §1.4, the hysteresis FSM §2.2, and the
                          top-N cutoff §3.2 — typically a closure that builds ``dvol`` FROM the panel it
                          is handed, so corrupting the panel corrupts the liquidity statistic too).

    Returns True iff ``eligibility_fn(panel_full, t).universe`` is bit-identical to
    ``eligibility_fn(panel_corrupted, t).universe`` where ``panel_corrupted`` has every row ``s > t``
    garbage-overwritten. Bit-identity check, not statistical (§7's anti-circular-validation warning).
    """
    snap_full: EligibilitySnapshot = eligibility_fn(panel_full, t)
    snap_corrupt: EligibilitySnapshot = eligibility_fn(corrupt_future(panel_full, t, seed=seed), t)
    return bool(np.array_equal(snap_full.universe, snap_corrupt.universe))


__all__ = ["pit_check", "corrupt_future"]
