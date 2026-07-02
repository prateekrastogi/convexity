"""FundingModel: signed real-8h-stamp accrual, delegating to accounting.py (COST.md Section 6).

Invariants exercised: P (sign — short credited on positive funding, long pays), F-stamp (per-real-
stamp masked accrual, never a weekly average; delist stops accrual at the last live day), and N (the
accrual/sign is accounting.py's ONE truth — FundingModel mirrors it, never a second implementation).
"""

from __future__ import annotations

import numpy as np
import pytest

from research.xsec.accounting import _week_funding, settle_week
from research.xsec.cost.funding import FundingModel


def _flat_panel(n_days: int, n_sym: int):
    close = np.full((n_days, n_sym), 100.0)
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return close, funding, listed


def test_p_sign_short_credited_long_pays():
    # Invariant P / COST.md Section 6.1: cash_flow = -w * f. Long + positive funding PAYS (<0);
    # short + positive funding is CREDITED (>0).
    fm = FundingModel()
    w = np.array([0.5, -0.5])
    f = np.array([0.001, 0.001])  # both names have positive funding
    cf = fm.cash_flow(w, f)
    assert cf[0] < 0.0  # long pays
    assert cf[1] > 0.0  # short credited
    assert cf[0] == pytest.approx(-0.5 * 0.001, rel=1e-12)
    assert cf[1] == pytest.approx(0.5 * 0.001, rel=1e-12)


def test_zero_weight_contributes_no_funding():
    fm = FundingModel()
    cf = fm.cash_flow(np.array([0.0, -0.3]), np.array([0.002, 0.002]))
    assert cf[0] == 0.0
    assert cf[1] > 0.0


def test_accrued_over_delegates_to_accounting_verbatim():
    # Invariant N: accrued_over IS accounting._week_funding — the ONE accrual truth, not re-derived.
    fm = FundingModel()
    _, funding, listed = _flat_panel(30, 3)
    funding[8, 0] = 0.0004
    funding[9, 0] = 0.0005
    funding[10, 1] = 0.0003
    acc = fm.accrued_over(funding, listed, d0=7, d1=14)
    assert np.array_equal(acc, _week_funding(funding, listed, d0=7, d1=14))
    assert acc[0] == pytest.approx(0.0009, rel=1e-12)


def test_fstamp_per_stamp_sum_not_weekly_average_with_delist():
    # Invariant F-stamp / COST.md Section 6.2: +0.001/day for 2 days then -0.002/day for 5 days, name
    # delists after day 2 of the hold week. Masked per-stamp accrual = +0.002 (days held); the
    # flat-weekly-average alternative would be wrong sign AND magnitude for the short's cash flow.
    fm = FundingModel()
    _, funding, listed = _flat_panel(30, 2)
    funding[8:10, 0] = 0.001  # hold week (d0=7, d1=14] -> days 8..14
    funding[10:15, 0] = -0.002
    listed[10:, 0] = False  # delists after day 9: only days 8, 9 held
    w = np.array([-0.5, 0.5])  # SHORT the flipping name
    total = fm.weekly_cash_flow(funding, listed, w, d0=7, d1=14)
    masked_accrual = 0.001 * 2  # days actually held
    assert total == pytest.approx(-(-0.5) * masked_accrual, rel=1e-12)  # +0.001, credited
    naive_flat = -(-0.5) * (0.001 * 2 - 0.002 * 5)  # -0.004: the forbidden approximation
    assert np.sign(total) != np.sign(naive_flat)


def test_weekly_cash_flow_matches_settle_week_funding_term():
    # The FundingModel scalar equals settle_week's funding_t term on the same inputs (Invariant N:
    # one funding truth, whether reached via the model helper or the full weekly settlement).
    fm = FundingModel()
    close, funding, listed = _flat_panel(30, 3)
    funding[8, 0] = 0.0004
    funding[9, 0] = 0.0005
    funding[11, 2] = -0.0006
    w = np.array([0.4, -0.4, 0.2])
    res, _ = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w, w_prev=w, friction=0.0
    )
    assert fm.weekly_cash_flow(funding, listed, w, d0=7, d1=14) == pytest.approx(
        res.funding, rel=1e-12
    )


def test_cash_flow_rejects_misaligned_series():
    fm = FundingModel()
    with pytest.raises(ValueError):
        fm.cash_flow(np.zeros(3), np.zeros(2))


# --------------------------------------------------------------- Invariant SEP (COST.md Section 6.3)


def _sep_panel(n_days: int = 250, n_sym: int = 8, seed: int = 7):
    """Synthetic all-listed panel with a mid-sample momentum flip so the book actually turns over."""
    import pandas as pd

    from research.xsec.panel import Panel

    rng = np.random.default_rng(seed)
    symbols = tuple(f"S{i}" for i in range(n_sym))
    drift = np.linspace(-0.003, 0.003, n_sym)
    r = np.empty((n_days, n_sym))
    r[: n_days // 2] = drift + rng.normal(0.0, 0.01, (n_days // 2, n_sym))
    r[n_days // 2 :] = drift[::-1] + rng.normal(0.0, 0.01, (n_days - n_days // 2, n_sym))
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    funding = rng.normal(1e-4, 2e-4, (n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    dates = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n_days, freq="D"))
    first = {s: 0 for s in symbols}
    last = {s: n_days - 1 for s in symbols}
    return Panel(dates, symbols, close, funding, listed, first, last)


def test_sep_permuting_funding_leaves_weights_gross_turnover_byte_identical():
    # Invariant SEP (COST.md Section 6.3): the momentum book accrues INCIDENTAL signed funding on
    # whatever weights the (funding-blind) signal produces; funding never drives a weight. Permuting
    # the funding panel (holding price/rank fixed, momentum signal reads no funding) must change
    # funding_t in settle_week's output while leaving w_target -- and therefore gross_price_t and
    # turnover_t -- BYTE-identical. If portfolio.py ever read funding, this test would fail.
    from research.xsec.engine import XSecConfig, run_book
    from research.xsec.panel import Panel

    cfg = XSecConfig(
        signal="momentum", lookback=14, quantile="tercile", weighting="equal", vol_window=14
    )
    panel = _sep_panel()
    base_res, base_book = run_book(panel, cfg)

    # Permute the funding panel across ALL cells (breaks any time/symbol structure) -- a stronger
    # scramble than a within-column shuffle, guaranteeing funding_t moves.
    rng = np.random.default_rng(99)
    flat = panel.funding.flatten()
    permuted = flat[rng.permutation(flat.size)].reshape(panel.funding.shape)
    panel_perm = Panel(
        panel.dates,
        panel.symbols,
        panel.close,
        permuted,
        panel.listed,
        panel.first_day,
        panel.last_day,
    )
    perm_res, perm_book = run_book(panel_perm, cfg)

    # w_target byte-identical (the load-bearing SEP assertion: weights are funding-independent).
    assert np.array_equal(perm_book.weights, base_book.weights)
    # and therefore the two funding-independent settlement outputs are byte-identical too.
    assert np.array_equal(perm_res.gross_price, base_res.gross_price)
    assert np.array_equal(perm_res.turnover, base_res.turnover)
    assert np.array_equal(perm_res.universe_size, base_res.universe_size)
    # while funding_t DID change (non-vacuous: the permutation actually moved the funding series).
    assert np.any(perm_res.funding != base_res.funding)


def test_sep_quantile_weights_take_no_funding_argument():
    # Invariant SEP structurally: research/xsec/portfolio.py's weight builders take NO funding input,
    # so a funding rate physically cannot reach a weight. Asserted on the signatures themselves so a
    # future refactor that threads funding into weight formation trips this guard.
    import inspect

    from research.xsec.portfolio import liquidity_weights, quantile_weights

    for fn in (quantile_weights, liquidity_weights):
        params = set(inspect.signature(fn).parameters)
        assert "funding" not in params, f"{fn.__name__} must not take a funding argument (SEP)"
        assert not any("fund" in p.lower() for p in params), f"{fn.__name__} leaks funding (SEP)"
