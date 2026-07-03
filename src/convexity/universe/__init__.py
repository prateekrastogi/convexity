"""UNIVERSE pillar — point-in-time eligibility gate (docs/specs/UNIVERSE.md).

UNIVERSE answers one question, asked fresh at every weekly rebalance ``t`` (Friday 00:00 UTC):
of all names that have ever traded, which are it LEGITIMATE to rank, weight, and trade — long or
short — this week, using ONLY what was knowable strictly before ``t`` (UNIVERSE.md §0). It is a
GATE, not a forecaster (P2, provider-not-decider): it emits a boolean eligibility indicator and a
liquidity order statistic, never a trade, a size, or a lookback.

This package is a thin, contract-enforcing layer over the already-validated point-in-time substrate
in ``src/convexity/panel/panel.py`` (the ``listed`` mask, ``trailing_dollar_volume``, ``build_dollar_volume``)
and ``src/convexity/panel/candidates.py`` (the candidate symbol tables). It ADDS the mechanics UNIVERSE.md
specifies but that did not exist in code: the minimum-history warmup gate (§1.4), the per-symbol
hysteresis state machine (§2.2), the strict short-liquidity gate (§4.2), and the structural
point-in-time invariant check (§7, Invariant PIT). It never reimplements the proven panel numerics.

Public surface (``universe_api``): ``EligibilitySnapshot`` / ``ShortabilitySnapshot`` (§11.1),
``hysteresis_step`` (§2.2), ``short_liquid`` (§4.2), ``pit_check`` (§7).
"""

from __future__ import annotations

from convexity.universe.eligibility.history_gate import history_warmup, tau_history
from convexity.universe.liquidity.hysteresis import hysteresis_step
from convexity.universe.liquidity.ranking import top_n_universe
from convexity.universe.shortability.liquidity_gate import ShortLiquidityThreshold, short_liquid
from convexity.universe.types import EligibilitySnapshot, ShortabilitySnapshot
from convexity.universe.universe_api import build_eligibility, build_shortability
from convexity.universe.validate.pit_check import pit_check

__all__ = [
    "EligibilitySnapshot",
    "ShortabilitySnapshot",
    "ShortLiquidityThreshold",
    "hysteresis_step",
    "short_liquid",
    "top_n_universe",
    "pit_check",
    "history_warmup",
    "tau_history",
    "build_eligibility",
    "build_shortability",
]
