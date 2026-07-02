"""Frozen boundary types for the LEVERAGE pillar (LEVERAGE.md §9 directory layout).

LEVERAGE is a *sizer of the whole book*, not of a position (LEVERAGE.md §0, P8): it emits ONE
scalar ``G_t`` (gross exposure as a multiple of equity) plus its full ceiling decomposition and the
binding-reason code. These dataclasses are the immutable carriers of that decision and of the
inputs the sizing chain reads; every field maps to a symbol in the §0.1 notation table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# The four ceilings the composed rule takes a min() over (LEVERAGE.md §6.1). ``binding_ceiling`` on
# every decision is exactly one of these — analogous to the pre-pivot ``LeverageDecision
# .binding_ceiling`` boundary type (§6.1).
BindingCeiling = Literal["kelly", "capacity", "governor", "hard_cap"]


@dataclass(frozen=True)
class GTarget:
    """The Kelly/vol-target sizing term and its diagnostic re-expression (LEVERAGE.md §2.5, §2.4).

    ``kelly_term`` is ``c_t · SR_LCB_t / σ̂_t`` (§2.5); ``sigma_target`` is the SIZE-ONE diagnostic
    ``c_t · μ̂ = c_t · SR_LCB_t`` re-expression (§2.4) reported alongside — never an independently
    tuned second dial (Invariant SIZE-ONE). ``growth_capture`` is the ``c(2−c)`` fraction of
    full-Kelly growth captured at ``c_t`` (§2.3).
    """

    kelly_term: float
    c_t: float
    sr_lcb: float
    sigma_hat: float
    sigma_target: float
    growth_capture: float


@dataclass(frozen=True)
class CapacityInput:
    """Per-instrument inputs to the capacity ceiling (LEVERAGE.md §5.1).

    ``adv`` is the TEAL trailing dollar-volume ``ADV_i`` (causal, no lookahead); ``w_ref`` the
    instrument's reference-gross (100%) dollar weight ``w_i^{ref}`` from ``research/xsec/
    portfolio.py``'s already-built book; ``shortable`` the AMBER venue flag — an unshortable name
    contributes ``ADV_i = 0`` on the short side (§5.1) so it cannot silently inflate capacity.
    """

    symbol: str
    adv: float
    w_ref: float
    shortable: bool = True


@dataclass(frozen=True)
class GovernorState:
    """Drawdown-governor state at rebalance ``t`` (LEVERAGE.md §6.4).

    ``dd_t`` is the running peak-to-trough book-equity drawdown; ``m_t`` the monotone
    non-increasing ladder multiplier in ``[0, 1]`` (§6.4). ``terminal`` records that the terminal
    drawdown threshold was breached (``m_t → 0``, full flat — governor kill).
    """

    dd_t: float
    m_t: float
    terminal: bool


@dataclass(frozen=True)
class RatchetState:
    """Fractional-Kelly ratchet state (LEVERAGE.md §6.2–§6.3).

    ``c_t ∈ {0.25, 0.50}`` (RED, §2.3); ``s_k`` the accumulated one-sided CUSUM statistic (§6.3);
    ``elapsed_weeks`` the live-time counter gating the minimum-elapsed-time floor ``N_min_weeks``
    (§6.3). ``promoted`` mirrors ``c_t == 0.50``; ``sr_ref`` is the deflated ``SR_LCB_t`` frozen at
    promotion-eligibility time and used as the CUSUM reference (§6.3).
    """

    c_t: float
    s_k: float
    elapsed_weeks: int
    promoted: bool
    sr_ref: float


@dataclass(frozen=True)
class LeverageDecision:
    """OUTPUT of :meth:`LeverageEngine.size_book` (LEVERAGE.md §6.1, P8/Invariant PROV).

    ``g_t`` is THE output scalar (gross multiplier). ``binding_ceiling`` names which of the four
    §6.1 terms was numerically smallest (always populated). The four ``*_term`` fields record every
    ceiling that entered the ``min()`` so the decision is fully auditable (§6.1). ``g_t ≥ 0`` always
    — never negative, never an inverted book (§8 edge cases).
    """

    g_t: float
    binding_ceiling: BindingCeiling
    kelly_term: float
    capacity_term: float
    governor_term: float
    hard_cap_term: float
    g_target: GTarget
    governor: GovernorState
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def annualized_to_weekly_vol(vol_annual: float, periods_per_year: float) -> float:
    """Convert an annualized vol to the per-week (per-observation) vol (LEVERAGE.md §4.2, §7).

    ``σ̂_weekly = vol_annual / √periods_per_year`` — the cited ~28% annualized → 3.9% weekly mapping
    at ``PERIODS_PER_YEAR = 365.25/7 ≈ 52.18`` (§7 worked example). Kept here so both the vol-floor
    fallback (§4.2) and the worked-example reproduction use one conversion, not two.
    """
    return float(vol_annual) / float(np.sqrt(periods_per_year))
