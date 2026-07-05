"""RV pair signal (PAPER.md §11.3): causality, gate behavior, construction invariants.

The study's lookahead canary showed this construction swings ~4.5 Sharpe on an off-by-one — the
causality test here is the single most important pin in the module.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.signal import rv_pairs
from convexity.signal.rv_pairs import PAIRS, book_weights, half_life, roll_z
from tests.conftest import make_panel

SYMS = ("XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "COPPERUSDT", "CLUSDT", "NATGASUSDT")
N_DAYS = 1600  # ~228 weeks — clears the 104wk z / HL windows with margin


def _panel(seed: int = 7, n_days: int = N_DAYS, syms=SYMS, spread_phi: float = 0.9):
    """Synthetic co-integrated commodity panel: pair legs share a random walk; the log spread is
    a stationary AR(1) with weekly half-life ~ -ln2/ln(phi) (phi=0.9 -> ~6.6wk, inside the gate)."""
    rng = np.random.default_rng(seed)
    n = len(syms)
    base = rng.normal(0, 0.01, (n_days, n)).cumsum(axis=0)
    close = 100.0 * np.exp(base)
    # impose a stationary spread on each pre-named pair: leg B = leg A minus an AR(1) spread
    ix = {s: i for i, s in enumerate(syms)}
    for a, b in PAIRS:
        if a not in ix or b not in ix:
            continue
        s = np.zeros(n_days)
        eps = rng.normal(0, 0.02, n_days)
        daily_phi = spread_phi ** (1.0 / 5.0)  # weekly phi expressed on the daily grid
        for t in range(1, n_days):
            s[t] = daily_phi * s[t - 1] + eps[t]
        close[:, ix[b]] = close[:, ix[a]] * np.exp(-s)
    funding = np.zeros((n_days, n))
    listed = np.ones((n_days, n), dtype=bool)
    return make_panel(close, funding, listed, syms)


def _last_friday(panel) -> int:
    wd = panel.dates.weekday.to_numpy()
    return int(np.flatnonzero(wd == 4)[-1])


def test_causality_weights_at_d_ignore_future_prices():
    """THE canary pin: mutate every price AFTER the rebalance row -> the target is bit-identical.
    (The study's unshifted-levels book prints Sharpe -3.8; an off-by-one here would leak the
    scored week into the signal.)"""
    panel = _panel()
    wd = panel.dates.weekday.to_numpy()
    d = int(np.flatnonzero(wd == 4)[-10])  # a Friday with real future days after it
    w1 = book_weights(panel, d).w_target

    close2 = panel.close.copy()
    close2[d + 1 :] *= np.exp(np.linspace(0.5, 2.0, panel.n_symbols))[None, :]  # violent future
    panel2 = make_panel(close2, np.zeros_like(close2), np.ones_like(close2, bool), panel.symbols)
    w2 = book_weights(panel2, d).w_target
    assert np.array_equal(w1, w2)


def test_non_friday_rebalance_row_rejected():
    panel = _panel()
    wd = panel.dates.weekday.to_numpy()
    monday = int(np.flatnonzero(wd == 0)[-1])
    with pytest.raises(ValueError, match="not a Friday"):
        book_weights(panel, monday)


def test_mean_reverting_pairs_produce_a_dollar_neutral_unit_gross_book():
    panel = _panel()
    d = _last_friday(panel)
    rvb = book_weights(panel, d)
    w = rvb.w_target
    assert np.isfinite(w).all()
    gross_unit = float(np.abs(rvb.w_unit[-1]).sum())
    # unit gross before the vol overlay (0 only if every gate closed — not with phi=0.9 spreads)
    assert gross_unit == pytest.approx(1.0, abs=1e-9)
    # each pair's legs are equal-and-opposite -> the whole book is dollar-neutral
    assert float(w.sum()) == pytest.approx(0.0, abs=1e-9)
    # vol-target factor honors the frozen caps and is applied multiplicatively
    assert rv_pairs.VT_CLIP[0] <= rvb.vt_scale <= rv_pairs.VT_CLIP[1]
    assert np.allclose(w, rvb.w_unit[-1] * rvb.vt_scale)


def test_position_sign_short_rich_leg_long_cheap_leg():
    """Force pair A-B's spread far ABOVE its trailing mean at the last Friday: A rich vs B ->
    the book must be SHORT A, LONG B (continuous reversion sizing)."""
    panel = _panel()
    ix = {s: i for i, s in enumerate(panel.symbols)}
    a, b = PAIRS[0]
    close = panel.close.copy()
    close[-30:, ix[a]] *= np.exp(np.linspace(0.0, 0.35, 30))  # pump the A leg into the close
    p2 = make_panel(close, np.zeros_like(close), np.ones_like(close, bool), panel.symbols)
    d = _last_friday(p2)
    w = book_weights(p2, d).w_target
    # the A-leg weight aggregates A's side across its pairs; the pumped spread dominates
    assert w[ix[a]] < 0.0 and w[ix[b]] > 0.0


def test_random_walk_spread_is_gated_out():
    """A pure random-walk pair (phi -> 1, no mean reversion) must be inactive via the HL gate."""
    rng = np.random.default_rng(3)
    n_days = N_DAYS
    syms = ("XAUUSDT", "XAGUSDT")  # one pre-named pair, both legs independent random walks
    close = 100.0 * np.exp(rng.normal(0, 0.01, (n_days, 2)).cumsum(axis=0))
    panel = make_panel(close, np.zeros((n_days, 2)), np.ones((n_days, 2), bool), syms)
    d = _last_friday(panel)
    rvb = book_weights(panel, d)
    pair_diag = rvb.diag["pairs"].get("XAUUSDT-XAGUSDT", {})
    # independent random walks: the AR(1) phi is ~0 -> half-life ~1wk CAN slip inside the gate
    # occasionally, but a NON-mean-reverting *integrated* spread must not be persistently active.
    assert pair_diag.get("active_frac", 0.0) < 0.5


def test_half_life_flags_known_ar1():
    rng = np.random.default_rng(11)
    n = 400
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 1.0)
    hl = half_life(s)
    est = np.nanmedian(hl[rv_pairs.HL_WIN + 10 :])
    assert 3.0 < est < 15.0  # true half-life ln2/ln(1/0.9) ~ 6.6


def test_roll_z_is_causal_window_stat():
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, 300)
    z = roll_z(s, 50)
    assert np.isnan(z[:49]).all()
    t = 200
    seg = s[t - 49 : t + 1]
    assert z[t] == pytest.approx((s[t] - seg.mean()) / seg.std(ddof=1))


def test_missing_leg_deactivates_only_that_pair():
    """Drop NATGASUSDT from the panel: CL-NG is reported inactive; the metal pairs still trade."""
    panel = _panel(syms=("XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT", "COPPERUSDT", "CLUSDT"))
    d = _last_friday(panel)
    rvb = book_weights(panel, d)
    assert any("NATGASUSDT" in s for s in rvb.diag["inactive_pairs"])
    assert float(np.abs(rvb.w_unit[-1]).sum()) == pytest.approx(1.0, abs=1e-9)
    ix = {s: i for i, s in enumerate(panel.symbols)}
    assert rvb.w_target[ix["CLUSDT"]] == 0.0  # CL's only pair is inactive -> never traded


def test_symbols_outside_pairs_always_zero():
    panel = _panel()
    d = _last_friday(panel)
    w = book_weights(panel, d).w_target
    ix = {s: i for i, s in enumerate(panel.symbols)}
    assert w[ix["COPPERUSDT"]] == 0.0  # copper is unpaired by the frozen prereg
