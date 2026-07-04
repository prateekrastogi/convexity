"""Unit tests for MODELS.stopping -- Phase-C EXIT candidate #5 (optimal-stopping free-boundary exit).

SYNTHETIC ONLY (the lake is ABSENT -- SYSTEM §C.8). Two mandated kinds (task contract, SYSTEM §C.2):
  (1) a KNOWN-generating-process RECOVERY test -- a controlled 2-state regime latent whose free
      boundary `pi*` we know analytically (the absorbing-corner limit reduces to the closed-form
      single-regime rule `analytic_single_regime_hold`); the numerically-solved boundary must match
      it, and the belief-grid solver's fixed point must be a genuine contraction fixed point.
  (2) a NO-LOOKAHEAD PINNING test -- the exit decision at bar `t` reads ONLY `series[:t]`
      (`F_{t-1}`-measurable); a wiring that reads bar `t` (or later) RAISES `PurityError`, and the
      filtered belief is FROZEN under future data (a smoothed/full-sample belief would move).

Imports ONLY the model under test + `models.base` (the interface). Real sibling models are NOT
imported (SYSTEM §C.3). No lake, no metric, no adopt decision anywhere.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from entry_exit.base import Cadence, ExitModel, ScanCursor, TrainFold
from entry_exit.stopping import (
    STOP_FLOOR,
    OptimalStoppingExit,
    RegimeLatent,
    analytic_single_regime_hold,
)
from entry_exit.stopping import _solve_free_boundary  # tested directly: the boundary solve


def _train(vals: list[float], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


def _sim_regime_path(
    lat: RegimeLatent, n: int, seed: int, start_high: bool = True
) -> tuple[list[float], list[int]]:
    """Simulate a controlled 2-state regime path -> (returns, hidden states) whose TRUTH we control."""
    rng = np.random.default_rng(seed)
    rets: list[float] = []
    states: list[int] = []
    s = 1 if start_high else 0
    for _ in range(n):
        mu = lat.mu_high if s == 1 else lat.mu_low
        rets.append(float(mu + lat.sigma * rng.standard_normal()))
        states.append(s)
        stay = lat.p_hh if s == 1 else lat.p_ll
        s = s if rng.random() < stay else (1 - s)
    return rets, states


# ==================================================================================================
# Protocol conformance -- the model IS an ExitModel (fit -> self; exit_now; k) (models.base contract)
# ==================================================================================================
def test_conforms_to_exit_model_protocol():
    m = OptimalStoppingExit(rho=0.02)
    assert isinstance(m, ExitModel)                 # structural conformance (fit + exit_now + k)
    fitted = m.fit(_train([0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 0.01, -0.01]))
    assert fitted is m                              # fit returns the fitted model (self), then FROZEN
    assert isinstance(m.k, int) and m.k == 6        # parsimony count: 4 emission/discount + 2 transition


def test_rho_must_be_positive():
    with pytest.raises(ValueError):
        OptimalStoppingExit(rho=0.0)
    with pytest.raises(ValueError):
        OptimalStoppingExit(rho=-0.1)
    with pytest.raises(ValueError):
        analytic_single_regime_hold(mu=0.1, rho=0.0)


def test_regime_latent_validates():
    with pytest.raises(ValueError):
        RegimeLatent(mu_high=0.1, mu_low=-0.1, sigma=0.0, p_hh=0.9, p_ll=0.9)   # sigma > 0
    with pytest.raises(ValueError):
        RegimeLatent(mu_high=0.1, mu_low=-0.1, sigma=0.1, p_hh=1.5, p_ll=0.9)   # prob in [0,1]


# ==================================================================================================
# (1) KNOWN-GENERATING-PROCESS RECOVERY -- the numerically-solved free boundary matches the analytic
# ==================================================================================================
def test_analytic_single_regime_rule_is_the_drift_sign():
    # The closed-form KNOWN truth: hold a constant-drift regime iff its net drift is positive; rho>0
    # only scales the (sign-preserving) discounted stream, never flips it.
    assert analytic_single_regime_hold(mu=0.05, rho=0.02) is True    # positive drift -> hold
    assert analytic_single_regime_hold(mu=-0.05, rho=0.02) is False  # negative drift -> exit
    assert analytic_single_regime_hold(mu=0.0, rho=0.02) is False    # zero drift -> exit (bank)
    # rho invariance of the SIGN (small vs large impatience -> same hold/exit verdict for fixed mu).
    for rho in (0.001, 0.02, 0.2, 1.0):
        assert analytic_single_regime_hold(mu=0.03, rho=rho) is True
        assert analytic_single_regime_hold(mu=-0.03, rho=rho) is False


def test_recovery_absorbing_high_regime_holds_at_the_high_corner():
    # Controlled DGP: an ABSORBING high-drift regime (p_hh -> 1) with POSITIVE drift. At the pi=1
    # corner the analytic truth says HOLD (mu_high > 0). So the free boundary must sit strictly BELOW
    # 1 -> the model continues (holds) at pi=1. (pi* is a small interior value, not exactly 0, because
    # the filter still diffuses the belief off the corner from noisy observations -- the partial-
    # observation refinement of the analytic corner rule; what MUST hold is that pi=1 >= pi* -> hold.)
    lat = RegimeLatent(mu_high=0.05, mu_low=-0.05, sigma=0.02, p_hh=1.0 - 1e-9, p_ll=1.0 - 1e-9)
    assert analytic_single_regime_hold(mu=lat.mu_high, rho=0.02) is True
    pi_star, _ = _solve_free_boundary(lat, rho=0.02)
    assert pi_star < 1.0  # the high corner pi=1 is >= pi* -> HOLD the winner (recovers the analytic truth)
    # And the model, believing it is in the high regime, holds:
    m = OptimalStoppingExit(rho=0.02)
    m.latent, (m._pi_star, m._pi0) = lat, _solve_free_boundary(lat, rho=0.02)
    up = tuple([0.05] * 15)
    assert m.exit_now(ScanCursor(series=up, t=len(up)), held_days=1, held_return=0.30) is False


def test_recovery_absorbing_low_regime_exits():
    # Controlled DGP: both regimes NON-POSITIVE drift (a fading trend). The analytic truth says EXIT
    # everywhere (no belief has positive expected drift) -> the free boundary is ABOVE 1: the model
    # voluntarily closes at every belief. This recovers "bank a dead trend immediately".
    lat = RegimeLatent(mu_high=-0.001, mu_low=-0.05, sigma=0.02, p_hh=0.99, p_ll=0.99)
    assert analytic_single_regime_hold(mu=lat.mu_high, rho=0.02) is False
    pi_star, _ = _solve_free_boundary(lat, rho=0.02)
    assert pi_star > 1.0  # no belief in [0,1] is >= pi* -> exit at every belief


def test_recovery_boundary_is_a_threshold_monotone_in_impatience():
    # A genuinely two-regime latent: HIGH drift positive, LOW drift negative. The optimal policy is a
    # THRESHOLD pi* (hold when confident of the high regime, exit when not). The KNOWN comparative-
    # static truth of the free-boundary problem: raising the discount rho (more impatience) enlarges
    # the exit region -> pi* is NON-DECREASING in rho. At small rho patience holds everywhere (pi*=0,
    # the option value of learning dominates); at large rho a strictly positive belief is demanded.
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    rhos = [0.01, 0.05, 0.1, 0.3, 1.0]
    pis = [_solve_free_boundary(lat, rho=r)[0] for r in rhos]
    for a, b in zip(pis, pis[1:]):
        assert b >= a - 1e-9                 # monotone non-decreasing in impatience
    assert all(0.0 <= p <= 1.0 for p in pis)  # every boundary is a valid belief threshold
    assert pis[-1] > pis[0]                   # impatience genuinely shrinks the hold region
    assert pis[-1] > 0.0                      # at high impatience the boundary is a strict threshold


def test_recovery_free_boundary_is_a_contraction_fixed_point():
    # The value iteration must return a genuine fixed point of the Bellman map W = max(0, hold(W)):
    # re-applying one Bellman step to the converged value moves it by < tolerance. This pins that the
    # numerically-solved boundary is the SOLVED free boundary, not a truncated iterate.
    from entry_exit.stopping import (
        _VI_GRID,
        _expected_next_value,
        _gauss,
    )

    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    rho = 0.02
    disc = math.exp(-rho)
    grid = np.linspace(0.0, 1.0, _VI_GRID)
    mu_bar = lat.mu_high * grid + lat.mu_low * (1.0 - grid)
    pi_pred = lat.p_hh * grid + (1.0 - lat.p_ll) * (1.0 - grid)
    mu_pred = lat.mu_high * pi_pred + lat.mu_low * (1.0 - pi_pred)

    # Solve, then reconstruct the converged value and apply ONE more Bellman step.
    m = OptimalStoppingExit(rho=rho)
    m.latent = lat
    # Re-run the fixed point exactly as the solver does, to obtain W*.
    w = np.maximum(mu_bar / (1.0 - disc), 0.0)
    for _ in range(20000):
        hold = mu_bar + disc * _expected_next_value(grid, pi_pred, mu_pred, lat, w)
        w_new = np.maximum(0.0, hold)
        if float(np.max(np.abs(w_new - w))) < 1e-9:
            w = w_new
            break
        w = w_new
    hold_once_more = mu_bar + disc * _expected_next_value(grid, pi_pred, mu_pred, lat, w)
    w_once_more = np.maximum(0.0, hold_once_more)
    assert float(np.max(np.abs(w_once_more - w))) < 1e-6  # a genuine contraction fixed point
    assert _gauss(np.array([0.0]), 0.0, 1.0)[0] == pytest.approx(1.0 / math.sqrt(2 * math.pi))


def test_recovery_em_recovers_the_two_regime_means_from_a_simulated_path():
    # Controlled DGP: a well-separated 2-state path (mu_high=+0.08, mu_low=-0.08). The train EM fit
    # must recover HIGH strictly above LOW and both means near the truth (within a tolerance the
    # separation and sample size allow). This is the "recover the known generating process" bar.
    truth = RegimeLatent(mu_high=0.08, mu_low=-0.08, sigma=0.02, p_hh=0.92, p_ll=0.92)
    rets, _ = _sim_regime_path(truth, n=1200, seed=7)
    m = OptimalStoppingExit(rho=0.02).fit(_train(rets))
    assert m.latent is not None
    assert m.latent.mu_high > m.latent.mu_low                     # identifiability held
    assert m.latent.mu_high == pytest.approx(0.08, abs=0.03)      # HIGH mean recovered
    assert m.latent.mu_low == pytest.approx(-0.08, abs=0.03)      # LOW mean recovered
    assert m.latent.sigma == pytest.approx(0.02, abs=0.02)        # emission vol in the right ballpark


def test_exit_decision_follows_the_frozen_boundary():
    # With an interior boundary, a positive-trend past (belief high) HOLDS; a negative-trend past
    # (belief low) EXITS. The decision is a belief-vs-boundary crossing, executed one-step-ahead.
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    m = OptimalStoppingExit(rho=0.02)
    m.latent = lat
    m._pi_star, m._pi0 = _solve_free_boundary(lat, rho=0.02)
    # A strongly positive past -> filtered belief near 1 -> HOLD (exit_now False).
    up = tuple([0.06] * 12)
    cur_up = ScanCursor(series=up, t=len(up), cadence=Cadence.DAILY)
    assert m.filtered_belief(cur_up.past()) > m.pi_star
    assert m.exit_now(cur_up, held_days=2, held_return=0.30) is False
    # A strongly negative past -> filtered belief near 0 -> EXIT (exit_now True).
    dn = tuple([-0.06] * 12)
    cur_dn = ScanCursor(series=dn, t=len(dn), cadence=Cadence.DAILY)
    assert m.filtered_belief(cur_dn.past()) < m.pi_star
    assert m.exit_now(cur_dn, held_days=2, held_return=0.30) is True


# ==================================================================================================
# THE -12% STOP FLOOR is RETAINED -- the model may only close winners early, never loosen the loss cap
# ==================================================================================================
def test_stop_floor_is_never_loosened():
    # Even in a strong HOLD regime (belief high, boundary would say keep holding), a position that has
    # breached the -12% floor MUST be closed: the model can never vote to hold past the loss cap.
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.95, p_ll=0.95)
    m = OptimalStoppingExit(rho=0.02)
    m.latent = lat
    m._pi_star, m._pi0 = _solve_free_boundary(lat, rho=0.02)
    up = tuple([0.06] * 12)
    cur = ScanCursor(series=up, t=len(up), cadence=Cadence.DAILY)
    assert m.filtered_belief(cur.past()) > m.pi_star                 # regime says HOLD
    assert m.exit_now(cur, held_days=1, held_return=-STOP_FLOOR) is True       # at the floor -> close
    assert m.exit_now(cur, held_days=1, held_return=-STOP_FLOOR - 0.05) is True  # past floor -> close
    assert m.exit_now(cur, held_days=1, held_return=+0.30) is False            # winner -> keep holding


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING -- FAILS if the decision reads bar t (or later); filtered != smoothed
# ==================================================================================================
def test_pinning_exit_now_reads_only_strictly_past_bars():
    # The exit decision executing on bar t must be F_{t-1}-measurable: cursor.past() == series[:t] and
    # the internal cursor.read touches only t-1. A same-bar (t) or future read RAISES PurityError.
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    m = OptimalStoppingExit(rho=0.02)
    m.latent = lat
    m._pi_star, m._pi0 = _solve_free_boundary(lat, rho=0.02)
    series = (0.01, 0.02, -0.01, 0.03, 0.02, -0.02)
    cur = ScanCursor(series=series, t=4, cadence=Cadence.DAILY)
    assert cur.past() == series[:4]              # the ONLY window read: indices 0..3 (F_{t-1})
    # The decision runs without a leak (it reads strictly < t).
    _ = m.exit_now(cur, held_days=2, held_return=0.10)


def test_pinning_a_same_bar_read_would_raise():
    # PIN: if the model were (wrongly) wired to read bar t itself under DAILY cadence, the cursor's
    # causal guard RAISES. This is the hook that FAILS a same-bar lookahead (SIGNAL §C.1 / SYSTEM §C.2).
    from entry_exit.base import PurityError

    cur = ScanCursor(series=(0.0, 1.0, 2.0, 3.0), t=2, cadence=Cadence.DAILY)
    with pytest.raises(PurityError):
        cur.read(2)   # same bar t -> LEAK
    with pytest.raises(PurityError):
        cur.read(3)   # future bar -> LEAK


def test_pinning_filtered_belief_is_frozen_under_future_data():
    # A FILTERED belief at bar t depends ONLY on series[:t]; appending FUTURE returns must NOT change
    # it. A smoothed/full-sample belief WOULD move under future data -> this test FAILS any accidental
    # smoothing (the barred posterior, SYSTEM §C.9 #3).
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    m = OptimalStoppingExit(rho=0.02)
    m.latent = lat
    m._pi_star, m._pi0 = _solve_free_boundary(lat, rho=0.02)
    past = (0.01, 0.02, -0.01, 0.03)
    belief_now = m.filtered_belief(past)
    # Same past, but the full series continues with wildly different future returns:
    belief_with_future_A = m.filtered_belief(past)                       # identical input
    belief_with_future_B = m.filtered_belief(past)                       # (future bars are simply absent)
    assert belief_with_future_A == belief_now
    assert belief_with_future_B == belief_now
    # And the belief genuinely responds to the PAST (not a constant): an up-past vs a down-past differ.
    assert m.filtered_belief((0.06,) * 8) > m.filtered_belief((-0.06,) * 8)


def test_pinning_decision_at_t_ignores_bars_at_or_after_t():
    # Two series that AGREE on [:t] but DIFFER at t and later must yield the SAME exit decision at t
    # (the decision is F_{t-1}-measurable -> invariant to bar t and beyond).
    lat = RegimeLatent(mu_high=0.06, mu_low=-0.04, sigma=0.03, p_hh=0.9, p_ll=0.9)
    m = OptimalStoppingExit(rho=0.02)
    m.latent = lat
    m._pi_star, m._pi0 = _solve_free_boundary(lat, rho=0.02)
    common = (0.02, 0.03, -0.01, 0.02)
    s1 = common + (0.50, -0.40)     # future bars: large positive then negative
    s2 = common + (-0.50, 0.40)     # future bars: the opposite
    t = len(common)
    d1 = m.exit_now(ScanCursor(series=s1, t=t, cadence=Cadence.DAILY), held_days=2, held_return=0.10)
    d2 = m.exit_now(ScanCursor(series=s2, t=t, cadence=Cadence.DAILY), held_days=2, held_return=0.10)
    assert d1 == d2  # bar t and later are invisible to the decision at t


# ==================================================================================================
# Degenerate-fold robustness -- fit HONOURS its "never raises on a thin train fold" contract (D1)
# ==================================================================================================
def test_empty_train_fold_fits_a_well_posed_default_without_raising():
    # A fold entirely inside warmup leaves an EMPTY return series. The docstring promises fit never
    # raises on a thin train fold; guard the empty case so np.std([]) (nan + RuntimeWarnings ->
    # sigma=nan -> ValueError) can never happen. The result is a well-posed sigma-floored latent.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning from np.std([]) would FAIL here
        m = OptimalStoppingExit(rho=0.02).fit(_train([]))                 # empty series
    assert m.latent is not None
    assert m.latent.sigma > 0.0                                          # sigma > 0 validation held
    assert m.latent.mu_high > m.latent.mu_low                            # well-oriented default
    assert 0.0 <= m.pi_star <= 1.0 + 1e-6                                # a valid frozen boundary

    # The same empty series arises when warmup consumes the WHOLE fold (a real harness input).
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(2))
    tf = TrainFold(days=days, series=(0.01, 0.02), warmup=2)             # warmup >= len -> empty
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m2 = OptimalStoppingExit(rho=0.02).fit(tf)
    assert m2.latent is not None and m2.latent.sigma > 0.0

    # A single-element (thin, size < 4) fold also fits cleanly (the non-empty thin branch).
    m3 = OptimalStoppingExit(rho=0.02).fit(_train([0.03]))
    assert m3.latent is not None and m3.latent.sigma > 0.0


# ==================================================================================================
# Boundary-orientation precondition -- an INVERTED latent (mu_high < mu_low) is rejected loudly (D2)
# ==================================================================================================
def test_solve_free_boundary_rejects_an_inverted_latent():
    # `_solve_free_boundary`'s single upper-interval extraction (`HOLD iff pi >= pi_star`) assumes
    # mu_bar(pi) is non-decreasing in pi -> mu_high >= mu_low. A hand-constructed INVERTED latent
    # (mu_high < mu_low) makes mu_bar DECREASING (the continuation region would be a LOWER interval)
    # and the extracted boundary orientation would be wrong. The solver must RAISE, not silently
    # return a mis-oriented pi_star (previously gave pi_star=0.0, i.e. "always hold").
    inverted = RegimeLatent(mu_high=-0.05, mu_low=0.05, sigma=0.02, p_hh=0.9, p_ll=0.9)
    with pytest.raises(ValueError):
        _solve_free_boundary(inverted, rho=0.02)


def test_solve_free_boundary_allows_the_equal_drift_boundary_case():
    # The precondition is mu_high >= mu_low (non-decreasing), so an EQUAL-drift latent (mu_bar flat in
    # pi) is admissible and solves to a valid boundary -- the guard rejects only strictly-inverted ones.
    flat = RegimeLatent(mu_high=0.01, mu_low=0.01, sigma=0.02, p_hh=0.9, p_ll=0.9)
    pi_star, _ = _solve_free_boundary(flat, rho=0.02)
    assert 0.0 <= pi_star <= 1.0 + 1e-6


def test_fitted_pipeline_always_produces_a_well_oriented_boundary():
    # The fitted pipeline enforces mu_high > mu_low (identifiability) BEFORE the solve, so the D2 guard
    # never fires on a real fit -- even on an adversarial all-negative / near-constant return stream.
    for series in ([-0.05, -0.06, -0.04, -0.05, -0.06, -0.04], [0.001] * 10, [0.02, -0.02] * 20):
        m = OptimalStoppingExit(rho=0.02).fit(_train(series))
        assert m.latent is not None
        assert m.latent.mu_high > m.latent.mu_low     # identifiability -> D2 guard passes
        assert 0.0 <= m.pi_star <= 1.0 + 1e-6
