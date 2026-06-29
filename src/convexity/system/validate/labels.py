"""System-level triple-barrier labels — firewalled, vertical = H_sys (SYSTEM §6.1, Inv FW/HC).

The label for a system decision at bar ``t`` (an open, a hold-resize, a rotate) is the realized
net-of-cost path outcome of the book delta it produced, resolved by a triple barrier:

    upper barrier : +b_up · σ_horizon       (profit target, RED multiple)
    lower barrier : -b_dn · σ_horizon       (stop / invalidation, RED multiple)
    vertical      : H_sys (the CONTRACTUAL system max-hold cap, in bars, RED)
    label         : which barrier touched first, signed, net of the execution-stub cost

Labels use FUTURE bars and therefore live ONLY here, structurally unreachable from any live module
(Invariant FW — nothing in modules/, compose/, nautilus/ imports validate/). Because TREND's
operational cap is set to ``tau_cap = H_sys`` (Invariant HC), no realized hold exceeds ``H_sys``, so
``H_sys`` is a true upper bound on every label horizon and the correct quantity to feed the embargo.
A position still alive at ``H_sys`` is force-marked ``time_cap`` (not death).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LabelOutcome = Literal["upper", "lower", "time_cap"]


@dataclass(frozen=True)
class TripleBarrierLabel:
    """One system triple-barrier label (Section 6.1). ``ret_net`` is signed and NET of the
    execution-stub cost; ``touch_idx`` is the first-touch bar; ``outcome`` names the barrier hit;
    ``horizon_bars`` ≤ H_sys by construction (Invariant HC)."""

    entry_idx: int
    touch_idx: int
    outcome: LabelOutcome
    ret_net: float
    horizon_bars: int


def triple_barrier_label(
    prices: np.ndarray,
    entry_idx: int,
    side: int,
    sigma_horizon: float,
    *,
    b_up: float,
    b_dn: float,
    h_sys: int,
    entry_cost: float = 0.0,
    exit_cost: float = 0.0,
) -> TripleBarrierLabel:
    """Resolve the first-touch triple barrier for one decision (Section 6.1, Invariant HC).

    Scans FUTURE bars from ``entry_idx`` up to the contractual cap ``H_sys``; the first of the
    ±b·σ_horizon barriers touched wins, signed for ``side``; if neither is touched by ``H_sys`` the
    position is force-marked ``time_cap`` at the cap (never beyond). ``ret_net`` subtracts the
    conservative execution-stub entry+exit cost (Section 6.1b) so labels are biased LOW.
    """
    p = np.asarray(prices, dtype=float)
    entry = p[entry_idx]
    up = entry * (1.0 + side * b_up * sigma_horizon)
    dn = entry * (1.0 - side * b_dn * sigma_horizon)
    last = min(entry_idx + h_sys, len(p) - 1)
    cost = entry_cost + exit_cost

    for i in range(entry_idx + 1, last + 1):
        px = p[i]
        # signed favorable/adverse excursion relative to side.
        if side > 0:
            if px >= up:
                return _label(entry_idx, i, "upper", side, entry, px, cost)
            if px <= dn:
                return _label(entry_idx, i, "lower", side, entry, px, cost)
        else:
            if px <= up:
                return _label(entry_idx, i, "upper", side, entry, px, cost)
            if px >= dn:
                return _label(entry_idx, i, "lower", side, entry, px, cost)
    # force time_cap at H_sys (Invariant HC: no realized hold exceeds H_sys).
    return _label(entry_idx, last, "time_cap", side, entry, p[last], cost)


def _label(entry_idx, touch_idx, outcome, side, entry, exit_px, cost) -> TripleBarrierLabel:
    gross = side * (exit_px - entry) / entry
    return TripleBarrierLabel(
        entry_idx=entry_idx,
        touch_idx=touch_idx,
        outcome=outcome,
        ret_net=float(gross - cost),
        horizon_bars=int(touch_idx - entry_idx),
    )
