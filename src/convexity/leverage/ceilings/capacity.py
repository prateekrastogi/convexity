"""Capacity ceiling ``G_t^{cap}`` and the liquidity-capping dial (LEVERAGE.md §5).

Capacity is a HARD ceiling, never a soft preference (P6): ``G_t`` is capped so no single position
implied by the scaled book exceeds ``κ · ADV_i`` (§5.1). This binds via ``min()``, never a weighted
blend. An unshortable name (AMBER ``shortable=False``) contributes ``ADV_i = 0`` on the short side
regardless of its measured liquidity (§5.1), so it cannot silently inflate capacity.

The liquidity-cap dial ``C`` is a CAPACITY tool, never a default (Invariant CAP-DIAL, §5.2):
plain inverse-vol (``'volscaled'``) is the default within-leg scheme; ``'liqcap_volscaled'`` at
tightness ``C`` engages ONLY above a RED book-size trigger, logged as a distinct reason code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.types import CapacityInput


@dataclass(frozen=True)
class CapacityCeiling:
    """OUTPUT of :func:`capacity_ceiling` (LEVERAGE.md §5.1).

    ``g_cap`` is ``min_i(κ·ADV_i / |w_i^{ref}|)`` — the largest ``G_t`` consistent with EVERY
    instrument's capacity ceiling holding simultaneously. ``binding_symbol`` names the tightest
    instrument. ``n_unshortable_zeroed`` counts short-leg names forced to ``ADV_i=0`` (§5.1).
    """

    g_cap: float
    binding_symbol: str | None
    n_unshortable_zeroed: int


def capacity_ceiling(
    inputs: Sequence[CapacityInput], defaults: LeverageDefaults
) -> CapacityCeiling:
    """Capacity ceiling ``G_t^{cap} = min_i(κ·ADV_i / |w_i^{ref}|)`` (LEVERAGE.md §5.1).

    Evaluated on the FULLY SCALED book: ``|position_i(G_t)| = G_t · |w_i^{ref}| ≤ κ·ADV_i`` for the
    reference-gross weight ``w_i^{ref}``, so the per-instrument ceiling on ``G_t`` is
    ``κ·ADV_i / |w_i^{ref}|`` and the book-wide ceiling is the ``min`` over all instruments (§5.1).

    An unshortable name (``shortable=False``) on the SHORT side (``w_ref < 0``) is treated as
    ``ADV_i = 0`` (§5.1): its ceiling becomes ``0`` and the book cannot lever at all while holding
    that short — the conservative P6 fallback (an unmeasured/unshortable name never contributes
    unbounded capacity, §8 edge case). A name with ``w_ref == 0`` imposes no ceiling (skipped).
    Empty input ⇒ no capacity constraint (``g_cap = +inf``).
    """
    g_cap = float("inf")
    binding: str | None = None
    n_zeroed = 0
    for ci in inputs:
        w = abs(ci.w_ref)
        if w == 0.0:
            continue  # name carries no reference weight -> imposes no capacity ceiling
        adv = ci.adv
        if ci.w_ref < 0.0 and not ci.shortable:
            adv = 0.0  # unshortable short leg -> ADV_i = 0 (§5.1), regardless of measured liquidity
            n_zeroed += 1
        ceiling_i = defaults.kappa * adv / w
        if ceiling_i < g_cap:
            g_cap = ceiling_i
            binding = ci.symbol
    return CapacityCeiling(g_cap=g_cap, binding_symbol=binding, n_unshortable_zeroed=n_zeroed)


@dataclass(frozen=True)
class LiquidityCapDecision:
    """OUTPUT of :func:`resolve_within_leg_scheme` (LEVERAGE.md §5.2, Invariant CAP-DIAL).

    ``scheme`` is the within-leg weighting SELECTION/portfolio should use next week: ``'volscaled'``
    by default, ``'liqcap_volscaled'`` only when the projected book size exceeds the RED trigger.
    ``c`` is the liquidity-cap tightness when engaged (else ``None``). ``reason`` is the distinct
    ``'liquidity_cap_engaged'`` directive code (§5.2), or empty when off.
    """

    scheme: str
    c: float | None
    reason: str


def resolve_within_leg_scheme(
    projected_book_size: float, defaults: LeverageDefaults
) -> LiquidityCapDecision:
    """Resolve the within-leg weighting scheme from the projected book size (LEVERAGE.md §5.2).

    ``projected_book_size = equity × G_t × reference_gross`` (the ONE book-size number LEVERAGE
    computes, §5.2). Below ``liquidity_cap_book_threshold`` (RED, ~$3M anchor) the scheme resolves
    to ``'volscaled'`` (plain inverse-vol) — the DEFAULT (Invariant CAP-DEFAULT/CAP-DIAL). Only
    above it does ``'liqcap_volscaled'`` at tightness ``C`` engage, emitting the distinct
    ``'liquidity_cap_engaged'`` reason code (§5.2). This is applied to NEXT week's construction (a
    one-week lag), avoiding same-week circularity between SELECTION and LEVERAGE (§5.2).
    """
    if projected_book_size > defaults.liquidity_cap_book_threshold:
        return LiquidityCapDecision(
            scheme="liqcap_volscaled",
            c=defaults.liquidity_cap_multiple,
            reason="liquidity_cap_engaged",
        )
    return LiquidityCapDecision(scheme="volscaled", c=None, reason="")
