"""Minimum-history warmup gate ``tau_history_i`` (UNIVERSE.md §1.4).

A genuinely-listed, currently-live name is not immediately eligible: it needs enough trailing history
for the signal that consumes it (``L_max`` days, SIGNAL-owned) AND for UNIVERSE's own liquidity
statistic (``w_warmup = W`` days, §2.1). §1.4:

    tau_history_i = tau_list_i + L_max + w_warmup                (with L_max=16, W=30 -> tau_list + 46)
    1{ i ∈ U_t^base } = 1{ tau_history_i <= t < tau_delist_i }

``tau_history_i`` is ``F_t``-measurable at any ``t >= tau_history_i`` (a fixed offset from an
already-observed ``tau_list_i``), so this refinement preserves §1.1's point-in-time property. The two
warmups are SUMMED, not netted against shared calendar days (§1.4), so ``tau_history`` is a
conservative upper bound on the true minimum history (P6).

Invariant HW (§1.4): the warmup is computed from instrument ``i``'s OWN ``tau_list_i``, NEVER from the
earliest-listed symbol in the panel or a fixed calendar date. A symbol that listed ``k < L_max +
w_warmup`` days before ``t`` is excluded regardless of how much history OTHER symbols have — this is
the formal version of ``research/xsec/signals.py::momentum_signal``'s per-symbol endpoint guard, which
already checks ``panel.listed`` at BOTH window endpoints rather than the panel's global first row.
"""

from __future__ import annotations

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, L_MAX, UniverseConfig
from convexity.universe.eligibility.stopping_times import in_tradeable_life, tau_list
from research.xsec.panel import Panel


def history_warmup(cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG) -> int:
    """The total minimum-history warmup in trading days, ``L_max + w_warmup`` (UNIVERSE.md §1.4).

    ``w_warmup`` is pinned to ``cfg.W`` (§1.4), so this moves only when the BLUE grid moves ``W`` — never
    an independent knob. With the RED defaults (``L_max=16``, ``W=30``) this is 46 trading days.
    """
    return cfg.l_max + cfg.w_warmup


def tau_history(panel: Panel, cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG) -> np.ndarray:
    """Per-symbol minimum-history stopping time ``tau_history_i`` (UNIVERSE.md §1.4).

    Returns shape ``(n_symbols,)`` int: ``tau_list_i + L_max + w_warmup``, computed PER-SYMBOL from each
    name's own ``tau_list_i`` (Invariant HW). ``F_t``-measurable: a fixed offset from an observed
    listing index.
    """
    return tau_list(panel) + history_warmup(cfg)


def base_eligible(panel: Panel, t: int, cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG) -> np.ndarray:
    """Base eligibility mask ``1{ tau_history_i <= t < tau_delist_i }`` (UNIVERSE.md §1.4).

    Returns shape ``(n_symbols,)`` bool = ``U_t^base``, the set after the history-warmup refinement and
    before any liquidity gate. A name half-way through its warmup window is excluded, never partially
    admitted (P6). ``F_t``-measurable: reads only ``tau_list_i`` / ``tau_delist_i``, both fixed offsets
    from observed listing indices, and the scalar ``t``.

    Invariant HW: excludes any symbol with ``t - tau_list_i < L_max + w_warmup`` no matter how much
    trailing history OTHER symbols in the panel carry.
    """
    if t < 0 or t >= panel.n_days:
        return np.zeros(panel.n_symbols, dtype=bool)
    th = tau_history(panel, cfg)
    # The ``t < tau_delist_i`` upper bound (and the ``tau_list_i <= t`` liveness lower bound) is exactly
    # ``in_tradeable_life`` (§1.2 vocabulary); delegate to it rather than recomputing ``tau_delist`` here,
    # so §1.2's tradeable-life mask has ONE definition. The warmup refinement is the extra ``th <= t``
    # term §1.4 sums on top (``tau_history_i >= tau_list_i``, so the warmup subsumes the ``tau_list <= t``
    # half of the tradeable-life window; the ``t < tau_delist_i`` half comes from ``in_tradeable_life``).
    return (th <= t) & in_tradeable_life(panel, t)


__all__ = ["history_warmup", "tau_history", "base_eligible", "L_MAX"]
