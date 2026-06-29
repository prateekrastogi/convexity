"""The causal cross-asset return panel (REGIME §2 panel hygiene, §7 layer 1, Invariant CAUSAL).

Builds ``R_t ∈ ℝ^{W×n_t}`` from per-instrument price/return series aligned on the regime cadence.
No modelling beyond alignment + causal de-mean:

  * Returns are log-returns at the regime cadence.
  * A name that does not trade at a stamp (closed session, halt) is forward-filled to a ZERO
    return for that stamp ONLY within its AMBER staleness window (``staleness_stamps``); past that
    it is DROPPED from the panel for the bar, so ``n_t`` shrinks (every measure runs on the live
    ``n_t``). 24/7 crypto, session-bound equities and Dukascopy commodities are aligned by AMBER
    calendar, never by forking the math.
  * De-meaning uses the TRAILING causal mean only (P6) — never a centred/full-sample mean.

No future bar enters: the panel ends at ``asof_bar`` and uses only rows at or before it (Invariant
CAUSAL). The builder is a pure function of the supplied aligned series.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.core.types import AssetClass
from convexity.regime.config.asset_class import config_for
from convexity.regime.types import Panel


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Causal log-returns of a 1-D price series (length L -> length L-1). NaN where price absent."""
    prices = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(prices))
    return r


def build_panel(
    *,
    returns_by_symbol: Mapping[str, np.ndarray],
    asset_class_by_symbol: Mapping[str, AssetClass],
    window: int,
    asof_bar: int,
    stamp_is_open: Mapping[str, np.ndarray] | None = None,
) -> Panel:
    """Assemble the causal de-meaned panel at ``asof_bar`` over the trailing ``window`` (REGIME §2).

    Parameters
    ----------
    returns_by_symbol   : symbol -> 1-D causal log-return series at the regime cadence, indexed so
                          that index ``asof_bar`` is the current stamp (a NaN means the name had no
                          return at that stamp).
    asset_class_by_symbol : symbol -> AssetClass (drives the AMBER staleness window).
    window              : W trailing stamps to include (ending at ``asof_bar`` inclusive).
    asof_bar            : F_t index; only rows <= asof_bar are read (Invariant CAUSAL).
    stamp_is_open       : optional symbol -> bool array; False marks a stamp the name did not trade
                          (forward-filled to 0 within the staleness window, else the name is dropped
                          for the whole panel bar). When omitted, a NaN return is treated as closed.

    Returns
    -------
    Panel with ``returns`` shaped ``(window, n_t)``, causally de-meaned per surviving column, and
    ``instruments`` the surviving live universe (length ``n_t``).
    """
    lo = asof_bar - window + 1
    if lo < 0:
        raise ValueError(f"window {window} reaches before the start of history at asof_bar {asof_bar}")

    symbols = sorted(returns_by_symbol)  # deterministic column order (Invariant DET)
    cols: list[np.ndarray] = []
    live: list[str] = []

    for sym in symbols:
        series = np.asarray(returns_by_symbol[sym], dtype=float)
        win = series[lo : asof_bar + 1].copy()  # only causal rows (Invariant CAUSAL)
        if win.shape[0] != window:
            raise ValueError(f"{sym}: expected {window} rows, got {win.shape[0]}")

        open_mask = _open_mask(sym, stamp_is_open, win, lo, asof_bar)
        ac = config_for(asset_class_by_symbol.get(sym, "crypto_perp"))
        col = _fill_with_staleness(win, open_mask, ac.staleness_stamps)
        if col is None:  # dropped: too many consecutive stale stamps up to the current bar
            continue
        cols.append(col)
        live.append(sym)

    if not cols:
        raise ValueError("no instrument survived the staleness drop — empty panel")

    raw = np.column_stack(cols)  # (window, n_t)
    demeaned = raw - raw.mean(axis=0, keepdims=True)  # TRAILING causal de-mean (P6)
    return Panel(returns=demeaned, instruments=tuple(live), asof_bar=asof_bar)


def _open_mask(
    sym: str,
    stamp_is_open: Mapping[str, np.ndarray] | None,
    win: np.ndarray,
    lo: int,
    asof_bar: int,
) -> np.ndarray:
    """True where the name traded this stamp; falls back to 'has a finite return' if no calendar."""
    if stamp_is_open is not None and sym in stamp_is_open:
        m = np.asarray(stamp_is_open[sym], dtype=bool)[lo : asof_bar + 1]
        return m
    return np.isfinite(win)


def _fill_with_staleness(
    win: np.ndarray, open_mask: np.ndarray, staleness_stamps: int
) -> np.ndarray | None:
    """Forward-fill closed stamps to a 0 return within the staleness window; drop the name else.

    Returns the filled column, or ``None`` if the current bar's gap exceeds the staleness window
    (the name is dropped, shrinking ``n_t`` — REGIME §2 panel hygiene).
    """
    col = win.copy()
    # Drop the name if it is stale at the CURRENT bar beyond its staleness budget.
    if not open_mask[-1]:
        run = 0
        for k in range(len(open_mask) - 1, -1, -1):
            if open_mask[k]:
                break
            run += 1
        if run > staleness_stamps:
            return None
    col[~open_mask] = 0.0  # forward-fill a non-trading stamp to a zero return
    col[~np.isfinite(col)] = 0.0
    return col
