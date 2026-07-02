"""Trailing dollar-volume liquidity statistic ``DV_i(d)`` — a thin REUSE layer (UNIVERSE.md §2.1).

``DV_i(d) = mean_{s in [d-W+1, d]} ( close_i(s) * volume_i(s) )`` over ``i``'s live span, computed
causally (only ``s <= d``). UNIVERSE.md §2.1 pins this to ``research/xsec/panel.py::trailing_dollar_volume``
(``window: int = 30``) verbatim — this module DELEGATES to it, never recomputes it, so the proven
panel numerics (and their point-in-time behavior) are the single source of the liquidity statistic.

It adds one thing panel.py does not express: §2.4's conservative-fallback / minimum-valid-day rule.
If fewer than ``ceil(W/2)`` days in the trailing window are valid (missing/stale/zero-volume data),
``DV_i(d)`` is treated as UNRELIABLE and the name falls to rank 0 downstream (§2.4, P6: "a name that
cannot be liquidity-ranked fails the name closed, never admits it under an assumed rank"). This module
surfaces the valid-day count so ``ranking.py`` can apply that rule; it never imputes a neutral rank.
"""

from __future__ import annotations

import math

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, UniverseConfig
from research.xsec.panel import Panel, trailing_dollar_volume


def dollar_volume_at(
    panel: Panel,
    dvol: np.ndarray,
    d: int,
    cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG,
) -> np.ndarray:
    """Trailing ``W``-day mean dollar volume ``DV_i(d)`` (UNIVERSE.md §2.1); REUSES panel.py.

    ``dvol`` is the daily dollar-volume array from ``research.xsec.panel.build_dollar_volume`` (same
    dates x symbols order, NaN off-span). Thin pass-through to ``trailing_dollar_volume`` at ``cfg.W`` —
    no new liquidity notion. Returns shape ``(n_symbols,)`` float, NaN for names not live at ``d``.
    """
    return trailing_dollar_volume(dvol, panel.listed, d, window=cfg.W)


def valid_day_count(dvol: np.ndarray, d: int, cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG) -> np.ndarray:
    """Number of finite (valid) dollar-volume days in the trailing ``W``-window ending at ``d`` (§2.4).

    Returns shape ``(n_symbols,)`` int. Used to apply §2.4's ``ceil(W/2)``-valid-days conservative
    fallback: a name below the floor cannot be trusted to be liquidity-ranked and falls to rank 0
    (``ranking.py``). Causal — reads only rows ``<= d``.
    """
    n_sym = dvol.shape[1]
    start = max(0, d - cfg.W + 1)
    block = dvol[start : d + 1]  # (<=W, n_sym)
    if block.shape[0] == 0:
        return np.zeros(n_sym, dtype=np.int64)
    return np.isfinite(block).sum(axis=0).astype(np.int64)


def min_valid_days(cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG) -> int:
    """``ceil(W * min_valid_frac)`` — the §2.4 minimum-valid-day threshold (default ``ceil(W/2)``)."""
    return int(math.ceil(cfg.W * cfg.min_valid_frac))
