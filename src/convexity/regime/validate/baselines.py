"""Baselines for the regime overlay (REGIME §9, firewalled — Invariant V).

Compare the discrete sticky machine against (a) always-on (``g_t ≡ 1``), (b) each single measure
alone, and (c) the REJECTED continuous-multiplier design — to demonstrate the continuous design
whipsaws / over-trades vs the discrete sticky machine (REGIME §9 baselines). These helpers are used
OFFLINE only; nothing in ``panel/``, ``measures/`` or ``state/`` imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def always_on(n_bars: int) -> np.ndarray:
    """The no-overlay baseline: ``g_t ≡ 1`` for every bar (REGIME §9 baseline a)."""
    return np.ones(n_bars, dtype=float)


def continuous_multiplier(measure_pct: np.ndarray) -> np.ndarray:
    """The REJECTED continuous design: a smooth ``g = 1 − measure_pct`` (REGIME §9 baseline c).

    Re-grosses the book on every noisy tick of a statistic — used only to SHOW it over-trades vs the
    discrete sticky machine. Never used live (REGIME §5.1 deliberately rejects a continuous g_t).
    """
    p = np.clip(np.asarray(measure_pct, dtype=float), 0.0, 1.0)
    return np.clip(1.0 - p, 0.25, 1.0)


@dataclass(frozen=True)
class TurnoverReport:
    """How often a g_t path moves and the total absolute move (a turnover proxy, REGIME §9)."""

    n_changes: int
    total_abs_move: float


def turnover(g_path: np.ndarray) -> TurnoverReport:
    """Count the moves + total absolute change of a ``g_t`` path (the whipsaw / turnover audit, §9).

    The discrete sticky machine should print FAR fewer changes and a smaller total move than the
    continuous baseline on the same noisy measure stream.
    """
    g = np.asarray(g_path, dtype=float)
    diffs = np.diff(g)
    return TurnoverReport(
        n_changes=int(np.count_nonzero(diffs != 0.0)),
        total_abs_move=float(np.sum(np.abs(diffs))),
    )
