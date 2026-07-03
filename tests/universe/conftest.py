"""Synthetic panel + dollar-volume fixtures for the UNIVERSE eligibility tests (no lake I/O).

Mirrors ``tests/conftest.py``'s ``make_panel`` construction so the UNIVERSE contract layer
is exercised against the SAME ``Panel`` shape ``src/convexity/panel/panel.py`` produces, deterministically and
without touching the 137-instrument lake (read-only elsewhere). A companion synthetic daily
dollar-volume array is built directly (bypassing ``build_dollar_volume``'s lake read) so the liquidity
rank / hysteresis / short gates are testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from convexity.panel.panel import Panel


def make_panel(
    close: np.ndarray, listed: np.ndarray, symbols: tuple[str, ...]
) -> Panel:
    """Assemble a ``Panel`` from arrays (bypassing the lake), funding all-zero (momentum-only)."""
    n_days = close.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n_days, freq="D"))
    funding = np.zeros_like(close)
    close = np.where(listed, close, np.nan)
    funding = np.where(listed, funding, np.nan)
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    return Panel(dates, symbols, close, funding, listed, first_day, last_day)


def make_dvol(dvol_live: np.ndarray, listed: np.ndarray) -> np.ndarray:
    """Daily dollar-volume array aligned to a panel: ``dvol_live`` on listed cells, NaN off-span.

    Matches ``convexity.panel.panel.build_dollar_volume``'s post-condition (``np.where(listed, arr, nan)``)
    so ``trailing_dollar_volume`` behaves identically to production.
    """
    return np.where(listed, dvol_live, np.nan)


@pytest.fixture
def liquid_panel():
    """A 6-symbol, 200-day panel with clean per-symbol listing spans and distinct liquidity tiers.

    Symbols A..F. A/B/C live from day 0; D from day 60; E from day 120; F from day 0 but very illiquid.
    Dollar volume is constant-per-symbol so liquidity ranks are known and stable (no whipsaw), letting
    each test perturb exactly the dimension it targets. Returns (panel, dvol).
    """
    n_days = 200
    syms = ("A", "B", "C", "D", "E", "F")
    listed = np.ones((n_days, len(syms)), dtype=bool)
    listed[:60, 3] = False  # D lists day 60
    listed[:120, 4] = False  # E lists day 120
    t = np.arange(n_days)
    close = np.empty((n_days, len(syms)))
    close[:, 0] = 100.0 * (1.0 + 0.004) ** t
    close[:, 1] = 100.0 * (1.0 + 0.002) ** t
    close[:, 2] = 100.0 * (1.0 - 0.001) ** t
    close[:, 3] = 100.0 * (1.0 + 0.001) ** t
    close[:, 4] = 100.0 * (1.0 + 0.003) ** t
    close[:, 5] = 100.0 * (1.0 + 0.0005) ** t
    # Distinct constant dollar-volume tiers: A most liquid ... F least.
    dv_live = np.empty((n_days, len(syms)))
    dv_live[:, 0] = 1.0e9
    dv_live[:, 1] = 8.0e8
    dv_live[:, 2] = 6.0e8
    dv_live[:, 3] = 4.0e8
    dv_live[:, 4] = 2.0e8
    dv_live[:, 5] = 1.0e6  # F: two+ orders of magnitude thinner
    panel = make_panel(close, listed, syms)
    dvol = make_dvol(dv_live, listed)
    return panel, dvol
