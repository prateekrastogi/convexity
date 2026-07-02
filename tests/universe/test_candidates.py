"""Invariant CANDIDATE-VS-ELIGIBLE — the symbol list is a superset, never the answer (§6, §11.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from convexity.universe.config.candidates import candidate_symbols, crypto_candidates
from convexity.universe.eligibility.history_gate import base_eligible
from convexity.universe.eligibility.stopping_times import tau_delist, tau_list
from research.xsec.panel import Panel


def _panel(listed: np.ndarray, symbols: tuple[str, ...]) -> Panel:
    n_days = listed.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=n_days, freq="D"))
    close = np.where(listed, 100.0, np.nan)
    funding = np.where(listed, 0.0, np.nan)
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    return Panel(dates, symbols, close, funding, listed, first_day, last_day)


def test_candidate_list_loads_and_is_nonempty():
    """§6/§8 AMBER: the crypto candidate pool loads from research/UNIVERSE.md's curated table."""
    cands = crypto_candidates()
    assert len(cands) > 0
    assert cands == candidate_symbols(("crypto",))


def test_candidate_symbol_excluded_before_listing():
    """Inv CANDIDATE-VS-ELIGIBLE (§6): a candidate-list symbol with ``tau_list_i > t`` is excluded from
    U_t at every such t, even though it stays permanently in the candidate list."""
    cands = crypto_candidates()
    sym = cands[0]  # a real candidate-list symbol
    n_days = 300
    t = 100
    listed = np.ones((n_days, 1), dtype=bool)
    listed[: t + 50, 0] = False  # this candidate lists AFTER t (tau_list = t+50 > t)
    panel = _panel(listed, (sym,))
    assert sym in cands, "symbol is (and stays) in the candidate superset"
    assert tau_list(panel)[0] > t, "tau_list > t for this rebalance"
    assert not base_eligible(panel, t)[0], "candidate not yet listed at t is excluded from U_t"


def test_candidate_symbol_excluded_after_delisting():
    """§6 / §1.2: a candidate symbol already delisted by t (``tau_delist_i <= t``) is excluded from U_t,
    even though it remains permanently on the candidate list."""
    cands = crypto_candidates()
    sym = cands[0]
    n_days = 300
    t = 250
    listed = np.ones((n_days, 1), dtype=bool)
    listed[100:, 0] = False  # delists at row 100 -> tau_delist = 100 <= t
    panel = _panel(listed, (sym,))
    assert tau_delist(panel)[0] <= t
    assert not base_eligible(panel, t)[0], "already-delisted candidate is excluded from U_t"


def test_conservative_fallback():
    """Cross-cutting (§6, P6): membership is a t-dependent SUBSET computed from §1-4, never the pool's
    mere existence — a listed-but-not-yet-warm candidate is still excluded."""
    cands = crypto_candidates()
    sym = cands[0]
    n_days = 200
    t = 100
    listed = np.ones((n_days, 1), dtype=bool)
    listed[: t - 20, 0] = False  # lists 20 days before t: warm (20) < 46 -> excluded
    panel = _panel(listed, (sym,))
    assert not base_eligible(panel, t)[0], "listed candidate still inside warmup is excluded (§1.4)"
