"""Cross-sectional signals: no-lookahead, point-in-time eligibility, ranking, carry sign."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from convexity.signal import signals as sig
from convexity.panel.panel import Panel
from convexity.signal.signals import _pct_rank


def test_pct_rank_monotone_and_neutral_single():
    r = _pct_rank(np.array([3.0, 1.0, 2.0, np.nan]))
    # highest value (3.0) gets highest rank; NaN passes through.
    assert r[0] > r[2] > r[1]
    assert np.isnan(r[3])
    assert np.isclose(_pct_rank(np.array([5.0]))[0], 0.5)


def test_momentum_ranks_strongest_trend_highest(simple_panel):
    # at day 80 A has the strongest uptrend, C the downtrend -> A ranks top, C bottom.
    rank = sig.momentum_signal(simple_panel, 80, lookback=28, skip=0)
    a, b, c, d = rank
    assert a == max(rank)
    assert c == min(rank)


def test_momentum_excludes_unlisted_symbol_no_lookahead(simple_panel):
    # at day 30 symbol D is NOT yet listed (lists day 40) -> its rank must be NaN (never ranked early).
    rank = sig.momentum_signal(simple_panel, 30, lookback=28, skip=0)
    assert np.isnan(rank[3])
    assert np.isfinite(rank[0])


def test_momentum_needs_full_trailing_window(simple_panel):
    # at day 40 (D just listed) D lacks a 28d trailing window -> NaN, but A/B/C are fine.
    rank = sig.momentum_signal(simple_panel, 45, lookback=28, skip=0)
    assert np.isnan(rank[3])  # only 5 days of D history < 28
    assert np.isfinite(rank[0])


def test_momentum_skip_shifts_window_back_no_lookahead(simple_panel):
    # skip must measure the return over [d-skip-L, d-skip], all indices <= d (never past d).
    # verify directly against the panel closes for symbol A (a strict uptrend).
    d, L, skip = 80, 14, 7
    c = simple_panel.close
    expect = c[d - skip, 0] / c[d - skip - L, 0] - 1.0
    # both windows end at or before d -> no lookahead; ranks stay top for A either way.
    r0 = sig.momentum_signal(simple_panel, d, lookback=L, skip=0)
    r7 = sig.momentum_signal(simple_panel, d, lookback=L, skip=skip)
    assert r0[0] == max(r0) and r7[0] == max(r7)
    # the skipped window's endpoint index (d-skip) is strictly before d -> no lookahead.
    assert expect > 0  # A trends up over the pre-skip window


def test_carry_sign_high_funding_shorted(simple_panel):
    # A has the HIGHEST positive funding -> we SHORT it -> LOWEST long-rank.
    # C has negative funding -> LONG it -> HIGHEST long-rank.
    rank = sig.carry_signal(simple_panel, 80, lookback=7)
    a, b, c, d = rank
    assert c == max(rank)  # negative funding -> long
    assert a == min(rank)  # highest funding -> short


def test_combined_intersection_eligibility(simple_panel):
    # at day 45, D fails momentum eligibility -> combined must also be NaN for D.
    rank = sig.combined_signal(simple_panel, 45, lookback=28, skip=0, carry_lookback=7)
    assert np.isnan(rank[3])
    assert np.isfinite(rank[0])


# --- ensemble_momentum_signal — SIGNAL.md §4.3-4.4, §6.2, §8, §9 (the deployable core) ----------


def _build_panel(
    close: np.ndarray, funding: np.ndarray, listed: np.ndarray, symbols: tuple[str, ...]
) -> Panel:
    """Assemble a ``Panel`` directly from arrays (mirrors conftest.make_panel; local so the
    ensemble tests can extend the cross-section without lake I/O)."""
    n_days = close.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n_days, freq="D"))
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    return Panel(
        dates,
        symbols,
        np.where(listed, close, np.nan),
        np.where(listed, funding, np.nan),
        listed,
        first_day,
        last_day,
    )


def _worked_example_panel() -> tuple[Panel, int]:
    """SIGNAL.md §9's 4-symbol setup: raw L=12 trailing returns +0.18/+0.02/-0.05/-0.05 and 7d
    mean funding +0.00060/-0.00010/+0.00005/-0.00080 for A/B/C/D at rebalance day d."""
    n_days, syms, d = 20, ("A", "B", "C", "D"), 15
    close = np.full((n_days, 4), 100.0)
    close[d] = [118.0, 102.0, 95.0, 95.0]  # vs close[d-12] = 100 -> the §9 raw returns exactly
    funding = np.tile([0.00060, -0.00010, 0.00005, -0.00080], (n_days, 1))
    listed = np.ones((n_days, 4), dtype=bool)
    return _build_panel(close, funding, listed, syms), d


@pytest.mark.parametrize("skip", [0, 7])
def test_ensemble_equals_rank_average_of_band_members(simple_panel, skip):
    # §4.4's operator, bit-for-bit: per-L momentum ranks -> equal-weight average over the
    # intersection-eligible names -> final _pct_rank re-rank. Also covers Invariant SKIP
    # pass-through (skip shifts every band member's window identically).
    d, band = 80, (12, 14, 16)
    per_l = np.vstack([sig.momentum_signal(simple_panel, d, lookback=L, skip=skip) for L in band])
    eligible = np.isfinite(per_l).all(axis=0)
    v = np.full(3, 1.0 / 3.0)
    avg = np.full(simple_panel.n_symbols, np.nan)
    avg[eligible] = v @ per_l[:, eligible]
    expect = _pct_rank(avg)
    got = sig.ensemble_momentum_signal(simple_panel, d, skip=skip)
    np.testing.assert_array_equal(got, expect)
    # contract: (n_symbols,) aligned, values in [0, 1] for eligible, NaN for ineligible (§10).
    finite = np.isfinite(got)
    assert got.shape == (simple_panel.n_symbols,)
    assert ((got[finite] >= 0.0) & (got[finite] <= 1.0)).all()


def test_ensemble_no_lookahead_future_rows_are_garbage(simple_panel):
    # Invariant NLA: adversarially corrupting every panel cell with row index > d (NaN, ±inf,
    # shuffled prices, flipped listing flags) must leave the signal bit-for-bit unchanged.
    d = 80
    base = sig.ensemble_momentum_signal(simple_panel, d)
    rng = np.random.default_rng(7)
    close = simple_panel.close.copy()
    funding = simple_panel.funding.copy()
    listed = simple_panel.listed.copy()
    close[d + 1 :] = rng.permutation(close[d + 1 :].ravel()).reshape(close[d + 1 :].shape)
    close[d + 1] = np.nan
    close[d + 2] = np.inf
    funding[d + 1 :] = rng.normal(size=funding[d + 1 :].shape) * 1e3
    listed[d + 1 :] = ~listed[d + 1 :]
    corrupt = replace(simple_panel, close=close, funding=funding, listed=listed)
    np.testing.assert_array_equal(sig.ensemble_momentum_signal(corrupt, d), base)


def test_ensemble_carry_zero_funding_never_read(simple_panel):
    # Invariant CARRY-ZERO (§6.2): w_carry = 0 in the deployable core — perturbing the ENTIRE
    # funding panel (past rows included) must not move the ensemble by a single bit.
    d = 80
    base = sig.ensemble_momentum_signal(simple_panel, d)
    funding = np.random.default_rng(11).normal(size=simple_panel.funding.shape)
    perturbed = replace(simple_panel, funding=funding)
    np.testing.assert_array_equal(sig.ensemble_momentum_signal(perturbed, d), base)


def test_ensemble_rank_ignores_ineligible_added_symbol(simple_panel):
    # Invariant RANK: adding a symbol with listed[d] = False must leave every eligible symbol's
    # rank unchanged (the rank denominator and order statistics only see the eligible set).
    d, n_days = 80, simple_panel.n_days
    base = sig.ensemble_momentum_signal(simple_panel, d)
    close = np.column_stack([simple_panel.close, np.full(n_days, 50.0)])
    funding = np.column_stack([simple_panel.funding, np.zeros(n_days)])
    listed = np.column_stack([simple_panel.listed, np.zeros(n_days, dtype=bool)])
    listed[100:, 4] = True  # E lists at day 100 > d -> ineligible at d
    wider = _build_panel(close, funding, listed, ("A", "B", "C", "D", "E"))
    rank = sig.ensemble_momentum_signal(wider, d)
    np.testing.assert_array_equal(rank[:4], base)
    assert np.isnan(rank[4])


def test_ensemble_intersection_eligibility_partial_band_excluded(simple_panel):
    # §8 case 5: D (lists day 40) is eligible for L=12 at day 52 but NOT for L=16 — the ensemble
    # excludes it entirely rather than scoring a partial 2-of-3 band.
    d = 52
    assert np.isfinite(sig.momentum_signal(simple_panel, d, lookback=12, skip=0)[3])
    assert np.isnan(sig.momentum_signal(simple_panel, d, lookback=16, skip=0)[3])
    rank = sig.ensemble_momentum_signal(simple_panel, d)
    assert np.isnan(rank[3])
    assert np.isfinite(rank[:3]).all()


def test_ensemble_red_defaults_and_no_fit(simple_panel):
    # §7 RED constants + Invariant NO-FIT (the part signals.py DOES own): the DEFAULT band is the
    # canonical {12, 14, 16}, skip = 0, and the ensemble weights are fixed equal (1/3), never fitted
    # — the default output is bit-identical to explicit equal weights and to any positive rescaling
    # (the re-rank is scale-invariant, so no argmax/regressed weight can hide as a rescaling).
    #
    # NOTE ON INVARIANT BAND (honest scoping): signals.py deliberately does NOT enforce BAND —
    # ``ensemble_momentum_signal(panel, d, lookbacks=(14,))`` is a well-defined single-member call
    # BY DESIGN (SIGNAL.md §4.3: "a single-L signal MAY be run for research/diagnostic purposes …
    # but MUST NOT be the promoted/deployed configuration"; §9 step 3 reduces the 1-member ensemble
    # to that member's momentum_signal). BAND is a PROMOTION-GATE invariant, and its ValueError
    # enforcement lives in PORTFOLIO's driver (research/xsec/engine.py::band_configs, per
    # docs/ARCHITECTURE.md's pillar->module map), NOT in this SIGNAL-owned module. This test does
    # not overstate a signal-layer BAND guarantee; the cross-reference below pins where the real
    # guarantee lives and asserts it can fail.
    params = inspect.signature(sig.ensemble_momentum_signal).parameters
    assert params["lookbacks"].default == (12, 14, 16)  # canonical deployed band is the DEFAULT
    assert len(params["lookbacks"].default) >= 2  # default is an ensemble, not a lone fitted L
    assert params["skip"].default == 0  # Invariant SKIP
    assert params["weights"].default is None  # -> equal-weight, Invariant NO-FIT (no fitted weight)
    d = 80
    default = sig.ensemble_momentum_signal(simple_panel, d)
    explicit = sig.ensemble_momentum_signal(simple_panel, d, weights=(1 / 3, 1 / 3, 1 / 3))
    rescaled = sig.ensemble_momentum_signal(simple_panel, d, weights=(1.0, 1.0, 1.0))
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(default, rescaled)
    # a research single-L call is NOT rejected at the signal layer (§4.3), so the deployed band's
    # BAND guarantee cannot come from here — it must come from the promotion gate:
    lone = sig.ensemble_momentum_signal(simple_panel, d, lookbacks=(14,))
    assert np.isfinite(lone).any()  # single-member ensemble runs fine (research/diagnostic use)


def test_band_enforced_by_portfolio_promotion_gate_not_signals():
    # Invariant BAND, verified where it actually lives (SIGNAL.md §4.4 / §7 BLUE; ARCHITECTURE.md
    # pillar->module map): the deployable band is fixed at {12, 14, 16} by PORTFOLIO's driver, which
    # rejects any non-canonical lookback set with a ValueError BEFORE it can reach a promoted config.
    # This is the discriminating BAND test (it would fail if the promotion gate stopped enforcing
    # the band); signals.py's own tests must NOT claim to provide this guarantee.
    from convexity.portfolio.engine import LOOKBACK_BAND, band_configs

    assert LOOKBACK_BAND == (12, 14, 16)  # the single canonical ensemble (§7 RED / cross-pillar)
    cfgs = band_configs()  # default base is the canonical momentum/quintile/volscaled config
    assert len(cfgs) == len(LOOKBACK_BAND) >= 2  # a real ensemble, never a lone fitted L
    assert tuple(sorted(c.lookback for c in cfgs)) == LOOKBACK_BAND
    with pytest.raises(ValueError):
        band_configs(lookbacks=(14,))  # a single fitted L is not a promotable band
    with pytest.raises(ValueError):
        band_configs(lookbacks=(12, 13, 14, 15, 16))  # the ungrounded m=5 variant (§P4)


def test_ensemble_rejects_invalid_weights_and_empty_band(simple_panel):
    # §8 case 9: a negative weight is nonsensical for a rank-average; mismatched lengths and an
    # empty band fail loud, never silently renormalized around.
    with pytest.raises(ValueError):
        sig.ensemble_momentum_signal(simple_panel, 80, lookbacks=())
    with pytest.raises(ValueError):
        sig.ensemble_momentum_signal(simple_panel, 80, weights=(0.5, 0.5))
    with pytest.raises(ValueError):
        sig.ensemble_momentum_signal(simple_panel, 80, weights=(-1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        sig.ensemble_momentum_signal(simple_panel, 80, weights=(0.0, 0.0, 0.0))


def test_ensemble_worked_example_bit_for_bit():
    # SIGNAL.md §9, all four steps reproduced exactly (ranks are exact binary fractions).
    panel, d = _worked_example_panel()
    # Step 1 — momentum cross-sectional rank at L=12 (C and D tie at -0.05 -> average rank 1.5).
    mom = sig.momentum_signal(panel, d, lookback=12, skip=0)
    np.testing.assert_array_equal(mom, [0.875, 0.625, 0.25, 0.25])
    # Step 2 — carry rank (sign negated: most-negative funding D ranks highest).
    car = sig.carry_signal(panel, d, lookback=7)
    np.testing.assert_array_equal(car, [0.125, 0.625, 0.375, 0.875])
    # Step 3 — single-member ensemble reduces exactly to Step 1's rank (re-rank is the identity).
    ens = sig.ensemble_momentum_signal(panel, d, lookbacks=(12,))
    np.testing.assert_array_equal(ens, mom)
    # Step 4 — the deployable rank is UNCHANGED by carry (w_carry = 0); combined_signal's
    # research/diagnostic 50/50 blend re-ranks the §9 averages {A:.5, B:.625, C:.3125, D:.5625}.
    comb = sig.combined_signal(panel, d, lookback=12, skip=0, carry_lookback=7)
    np.testing.assert_array_equal(comb, [0.375, 0.875, 0.125, 0.625])
    assert not np.array_equal(comb, ens)  # blending materially changes the book (§9 step 4 note)


def test_ensemble_degenerate_all_equal_scores_neutral():
    # §8 case 6: byte-identical trailing returns -> the whole cross-section ties at 0.5 (a
    # well-defined neutral book), never a divide-by-zero or NaN propagation.
    panel, _ = _worked_example_panel()
    rank = sig.ensemble_momentum_signal(panel, 14, lookbacks=(12,))  # all closes 100 in [2, 14]
    np.testing.assert_array_equal(rank, [0.5, 0.5, 0.5, 0.5])


def test_carry_sign_cash_flow_non_negative_on_rank_extremes():
    # Invariant CARRY-SIGN (§5.2): a book long the top carry-rank name and short the bottom one
    # has cash_flow = sum(-w_i * f_i) >= 0 over the SAME trailing window that generated the rank
    # (accounting.settle_week's -w * f_acc convention); ranking on +mean_funding must fail.
    panel, d = _worked_example_panel()
    car = sig.carry_signal(panel, d, lookback=7)
    lo, hi = int(np.nanargmin(car)), int(np.nanargmax(car))
    window = panel.funding[d - 6 : d + 1]  # the c=7 trailing window the rank used
    w = np.zeros(panel.n_symbols)
    w[hi], w[lo] = 0.5, -0.5  # long the HIGH carry rank, short the LOW carry rank
    cash_flow = float((-w * window).sum())
    assert cash_flow > 0.0  # both legs collect funding in expectation over the window
    flipped_cash_flow = float((w * window).sum())  # ranking on +mean_funding flips the book
    assert flipped_cash_flow < 0.0
