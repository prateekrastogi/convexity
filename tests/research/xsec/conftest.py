"""Synthetic point-in-time panel fixtures for the cross-sectional engine tests (no lake I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.xsec.panel import Panel


def make_panel(
    close: np.ndarray, funding: np.ndarray, listed: np.ndarray, symbols: tuple[str, ...]
) -> Panel:
    """Assemble a ``Panel`` directly from arrays (bypassing the lake) for deterministic tests."""
    n_days = close.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n_days, freq="D"))  # a Monday start
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    close = np.where(listed, close, np.nan)
    funding = np.where(listed, funding, np.nan)
    return Panel(dates, symbols, close, funding, listed, first_day, last_day)


@pytest.fixture
def simple_panel() -> Panel:
    """A 4-symbol, 120-day panel where symbol D lists late (day 40) — tests point-in-time universe.

    A/B/C are live from day 0; D from day 40. Prices are deterministic monotone trends so momentum
    ranks are known; funding is constant per symbol so the carry sign is checkable.
    """
    n_days, syms = 120, ("A", "B", "C", "D")
    close = np.zeros((n_days, 4))
    funding = np.zeros((n_days, 4))
    listed = np.ones((n_days, 4), dtype=bool)
    listed[:40, 3] = False  # D lists on day 40
    t = np.arange(n_days)
    close[:, 0] = 100.0 * (1.0 + 0.004) ** t  # A: strongest uptrend
    close[:, 1] = 100.0 * (1.0 + 0.002) ** t  # B
    close[:, 2] = 100.0 * (1.0 - 0.001) ** t  # C: downtrend
    close[:, 3] = 100.0 * (1.0 + 0.001) ** t  # D
    funding[:, 0] = 0.0002  # A pays longs a lot -> carry says SHORT A
    funding[:, 1] = 0.0001
    funding[:, 2] = -0.0001  # C credits longs -> carry says LONG C
    funding[:, 3] = 0.0
    return make_panel(close, funding, listed, syms)
