"""Synthetic tests for the 2-state Gaussian HMM entry (ledger #3). SYNTHETIC ONLY -- no real lake.

Two mandated tests (SYSTEM §C.2, task contract), plus protocol conformance and warm-up honesty:

  (1) KNOWN-GENERATING-PROCESS RECOVERY -- simulate an HMM path whose TRUTH we control (a fixed state
      sequence with well-separated bull/bear drifts). After fitting on a train realization, the
      Hamilton FILTER's per-bar argmax must recover the true hidden state sequence within tolerance
      on a held-out realization; and the fitted `(mu, var, P)` must land near the true params.

  (2) NO-LOOKAHEAD PINNING -- (a) the FILTERED posterior differs from the Kim SMOOTHER (`filtered !=
      smoothed`), proving the two are genuinely distinct and the model does not silently substitute
      the (lookahead) smoother; (b) `signal_at` reads ONLY `cursor.past()` -- a mutant path that lets
      it read bar `t` via `cursor.read(t)` RAISES `PurityError`; (c) the smoother is BARRED as a live
      signal (calling it without the diagnostics flag RAISES); (d) `signal_at`'s decision is
      invariant to future bars appended after `t` (a definitional filtered/causal property) AND that
      invariance test DISCRIMINATES a leak: it runs on an OVERLAPPING-regime DGP (not well-separated,
      where the posterior saturates at 0/1 and a leak would pass unnoticed -- HMM-1) and carries an
      explicit full-sample-smoother mutant that MUST flip, so the test provably fails a lookahead.

Also: FLAT-reachability -- the default candidate has a non-empty neutral no-trade band (threshold
> 0.5), so it is NOT always in the market (HMM-2), and thresholds <= 0.5 are rejected.

The real Phase-C sibling models are NOT imported (interface-driven harness, SYSTEM §C.3). No lake,
no metric, no adopt decision is touched. `random` is seeded so the recovery is deterministic.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from entry_exit.base import (
    Cadence,
    EntryModel,
    PurityError,
    ScanCursor,
    Side,
    TrainFold,
)
from entry_exit.hmm import GaussianHMMEntry, HMMParams


# --- Controlled DGP: simulate a 2-state Gaussian HMM with a KNOWN state sequence -------------------
def _simulate(n, mu, sd, p_stay, seed):
    """Simulate `n` obs from a 2-state Gaussian HMM. Returns (observations, true_states).

    State 0 = bull (mu[0] > 0), state 1 = bear (mu[1] < 0); symmetric self-persistence `p_stay`.
    The returned `true_states` is the ground truth the FILTER must recover (test 1).
    """
    rng = random.Random(seed)
    states, obs = [], []
    s = 0
    for _ in range(n):
        if rng.random() > p_stay:  # switch
            s = 1 - s
        states.append(s)
        obs.append(rng.gauss(mu[s], sd[s]))
    return obs, states


def _train(vals, warmup=0):
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


# Well-separated truth so the filter can confidently recover the regime (large drift gap vs sd).
_TRUE_MU = (0.02, -0.02)
_TRUE_SD = (0.005, 0.005)
_P_STAY = 0.95


# ==================================================================================================
# Protocol conformance
# ==================================================================================================
def test_hmm_conforms_to_entry_model():
    m = GaussianHMMEntry()
    assert isinstance(m, EntryModel)  # structural: fit + signal_at + k
    obs, _ = _simulate(300, _TRUE_MU, _TRUE_SD, _P_STAY, seed=1)
    assert m.fit(_train(obs)) is m  # fit returns the fitted model (self)
    assert isinstance(m.k, int) and m.k >= 0
    assert m.params is not None  # parameters frozen after fit


def test_signal_at_before_fit_raises():
    with pytest.raises(RuntimeError):
        GaussianHMMEntry().signal_at(ScanCursor(series=(0.01, 0.02), t=2))


def test_bad_threshold_and_warmup_rejected():
    # threshold must be strictly in (0.5, 1). Out-of-(0,1) values AND anything <= 0.5 are rejected:
    # 0.5 collapses the FLAT band (always-in-market, HMM-2), < 0.5 makes the branches overlap.
    for bad in (0.0, 1.0, -0.1, 1.5, 0.5, 0.4, 0.3):
        with pytest.raises(ValueError):
            GaussianHMMEntry(threshold=bad)
    with pytest.raises(ValueError):
        GaussianHMMEntry(warmup=-1)


def test_flat_state_is_reachable_by_default():
    # HMM-2: the default candidate MUST have a non-empty FLAT no-trade band (not always-in-market).
    # Default threshold > 0.5 leaves the band (1 - threshold, threshold); a p_high in it -> FLAT.
    m = GaussianHMMEntry()
    assert m.threshold > 0.5  # the band (1 - threshold, threshold) is non-empty
    # Fit on an overlapping-regime path so the filtered posterior genuinely visits the neutral band,
    # then confirm FLAT actually occurs on a decision (not merely during warm-up).
    obs, _ = _simulate(400, (0.01, -0.01), (0.02, 0.02), 0.90, seed=41)
    m.fit(_train(obs))
    series = tuple(obs)
    sides = [
        m.signal_at(ScanCursor(series=series, t=t)) for t in range(m.warmup + 1, len(series) + 1)
    ]
    assert Side.FLAT in sides, "default candidate never emits FLAT -- it is always in the market"
    assert Side.LONG in sides and Side.SHORT in sides  # and both directions still reachable


# ==================================================================================================
# (1) KNOWN-GENERATING-PROCESS RECOVERY -- the FILTER recovers the true hidden state sequence
# ==================================================================================================
def test_recovery_filter_recovers_true_state_sequence():
    train_obs, _ = _simulate(600, _TRUE_MU, _TRUE_SD, _P_STAY, seed=7)
    m = GaussianHMMEntry().fit(_train(train_obs))

    # Held-out realization with its KNOWN truth; filter it and compare per-bar argmax to the truth.
    test_obs, true_states = _simulate(600, _TRUE_MU, _TRUE_SD, _P_STAY, seed=99)
    filt = m.filtered(tuple(test_obs))
    pred_states = [0 if hi >= lo else 1 for (hi, lo) in filt]

    # Allow filter lag at switch points, but the bulk of the path must be recovered.
    agree = sum(1 for a, b in zip(pred_states, true_states) if a == b) / len(true_states)
    assert agree > 0.9, f"filter recovered only {agree:.2%} of the true state sequence"


def test_recovery_fitted_params_near_truth():
    obs, _ = _simulate(1500, _TRUE_MU, _TRUE_SD, _P_STAY, seed=3)
    p = GaussianHMMEntry().fit(_train(obs)).params
    assert p is not None
    # State 0 is canonicalized to the high-drift state; recovers the true bull/bear drifts.
    assert p.mu[0] == pytest.approx(_TRUE_MU[0], abs=0.006)
    assert p.mu[1] == pytest.approx(_TRUE_MU[1], abs=0.006)
    assert p.mu[0] > 0 > p.mu[1]  # sign structure recovered
    # Self-persistence recovered (diagonal-dominant transition matrix).
    assert p.trans[0][0] > 0.7 and p.trans[1][1] > 0.7


def test_recovery_signal_direction_matches_regime():
    obs, _ = _simulate(600, _TRUE_MU, _TRUE_SD, _P_STAY, seed=11)
    m = GaussianHMMEntry(threshold=0.6, warmup=5).fit(_train(obs))
    # A strongly-bull recent past -> filtered high-drift prob high -> LONG.
    bull_past = tuple([0.02] * 30)
    assert m.signal_at(ScanCursor(series=bull_past, t=len(bull_past))) is Side.LONG
    # A strongly-bear recent past -> filtered high-drift prob low -> SHORT (sign-antisymmetric).
    bear_past = tuple([-0.02] * 30)
    assert m.signal_at(ScanCursor(series=bear_past, t=len(bear_past))) is Side.SHORT


def test_hmmparams_canonicalizes_high_drift_as_state0():
    # HMMParams enforces the mu[0] >= mu[1] canonicalization (no label-switching ambiguity).
    with pytest.raises(ValueError):
        HMMParams(mu=(-0.01, 0.01), var=(1e-4, 1e-4), trans=((0.9, 0.1), (0.1, 0.9)), pi0=(0.5, 0.5))


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING -- filtered != smoothed, and signal_at uses the filtered path ONLY
# ==================================================================================================
def test_pinning_filtered_differs_from_smoothed():
    # The FILTERED posterior (causal, F_t) and the Kim SMOOTHER (full-sample, F_T) are genuinely
    # DISTINCT -- if the model silently returned the smoother from `filtered`, this fails. Uses an
    # OVERLAPPING-regime DGP (drift gap ~= sd) so the filter is genuinely uncertain and the smoother's
    # future information visibly moves the interior posterior (with well-separated states the filter
    # is already near-certain and the two would coincide -- a degenerate, uninformative case here).
    obs, _ = _simulate(200, (0.01, -0.01), (0.02, 0.02), 0.90, seed=5)
    m = GaussianHMMEntry().fit(_train(obs))
    series = tuple(obs)
    filt = m.filtered(series)
    smooth = m.smoothed(series, for_diagnostics_only=True)
    # They must differ on at least one interior bar (the smoother uses future info there).
    assert any(abs(filt[t][0] - smooth[t][0]) > 1e-6 for t in range(len(series) - 1)), (
        "filtered and smoothed posteriors are identical -- the smoother adds no future info?")
    # The LAST bar is the one place they coincide (F_T == F_t at t=T): a sanity anchor.
    assert filt[-1][0] == pytest.approx(smooth[-1][0], abs=1e-9)


def test_pinning_smoother_is_barred_as_a_signal():
    # The Kim smoother is LOOKAHEAD; calling it without the explicit diagnostics flag RAISES.
    obs, _ = _simulate(50, _TRUE_MU, _TRUE_SD, _P_STAY, seed=13)
    m = GaussianHMMEntry().fit(_train(obs))
    with pytest.raises(PurityError):
        m.smoothed(tuple(obs))  # no for_diagnostics_only -> BARRED


def test_pinning_signal_at_never_reads_bar_t():
    # A leaky wiring that reads bar t (or later) for the decision at t must RAISE via the guard.
    obs, _ = _simulate(60, _TRUE_MU, _TRUE_SD, _P_STAY, seed=17)
    m = GaussianHMMEntry().fit(_train(obs))
    cur = ScanCursor(series=tuple(obs), t=30, cadence=Cadence.DAILY)
    # signal_at itself is causal (reads past() only) -- it does not raise.
    _ = m.signal_at(cur)
    # But the SAME-BAR read a leaky mutant would use is caught by the cursor guard:
    with pytest.raises(PurityError):
        cur.read(30)  # bar t under DAILY -> leak
    with pytest.raises(PurityError):
        cur.read(45)  # future bar -> leak


def _leaky_smoothed_signal_at(model, cursor):
    """A LOOKAHEAD mutant of `signal_at`: decide from the FULL-SAMPLE Kim smoother over the WHOLE
    `cursor.series` (which reads bars >= t), read at index t-1. This is exactly the leak SIGNAL
    §C.2(c) bars -- the pinning test below must FAIL against it (it flips when future bars change).
    The shipped `signal_at` (filter over `past()` only) must NOT flip.
    """
    from entry_exit.base import warmup_ok

    past = cursor.past()
    if not warmup_ok(len(past), model.warmup) or not past:
        return Side.FLAT
    sm = model.smoothed(cursor.series, for_diagnostics_only=True)  # reads bars >= t -> LOOKAHEAD
    p_high = sm[cursor.t - 1][0]
    if p_high >= model.threshold:
        return Side.LONG
    if p_high <= 1.0 - model.threshold:
        return Side.SHORT
    return Side.FLAT


def test_pinning_signal_invariant_to_future_bars():
    # Definitional filtered/causal property: the decision at bar t is UNCHANGED by any bars appended
    # after t. A model that (illegally) peeked at the future (e.g. read the Kim SMOOTHER over the
    # whole series) would flip when the future changes.
    #
    # HMM-1: this test MUST run on an OVERLAPPING-regime DGP (drift gap ~= sd), NOT the well-separated
    # _TRUE_MU/_TRUE_SD. Under well-separated states the filtered posterior is SATURATED at 0/1 on
    # essentially every bar, so a genuinely-leaky full-sample-smoother signal_at ALSO stays invariant
    # (its p_high is pinned at 0/1 too) and the test would PASS a leak -- a degenerate, uninformative
    # config, the same reason `test_pinning_filtered_differs_from_smoothed` switched to this DGP.
    # Under the overlapping DGP the interior posterior is genuinely uncertain, so the smoother's future
    # information moves the decision and the leak is caught (verified: the mutant below flips here).
    obs, _ = _simulate(80, (0.01, -0.01), (0.02, 0.02), 0.90, seed=23)
    m = GaussianHMMEntry(warmup=5).fit(_train(obs))
    t = 40
    tails = ([0.5] * 40, [-0.5] * 40, [0.0] * 40)

    # (i) The SHIPPED causal signal_at (filter over past() only) is invariant to appended future bars.
    sig_base = m.signal_at(ScanCursor(series=tuple(obs), t=t))
    for tail in tails:
        mutated = tuple(obs[:t]) + tuple(tail)
        cur = ScanCursor(series=mutated, t=t)
        assert m.signal_at(cur) is sig_base, "signal_at at t changed when only future bars changed"

    # (ii) DISCRIMINATION guard: a leaky full-sample-smoother signal_at DOES flip under the SAME
    # inputs, so this test provably FAILS a lookahead wiring (HMM-1) -- it is not a no-op invariance.
    leak_base = _leaky_smoothed_signal_at(m, ScanCursor(series=tuple(obs), t=t))
    leak_flips = any(
        _leaky_smoothed_signal_at(m, ScanCursor(series=tuple(obs[:t]) + tuple(tail), t=t))
        is not leak_base
        for tail in tails
    )
    assert leak_flips, (
        "the full-sample-smoother lookahead mutant did NOT flip on this DGP -- the invariance test "
        "cannot discriminate a leak here (HMM-1: use an overlapping-regime DGP, not well-separated)")


# ==================================================================================================
# Warm-up honesty (SIGNAL §C.1)
# ==================================================================================================
def test_warmup_suppresses_signal_until_enough_history():
    obs, _ = _simulate(100, _TRUE_MU, _TRUE_SD, _P_STAY, seed=29)
    m = GaussianHMMEntry(warmup=10).fit(_train(obs))
    # Fewer past bars than warm-up -> FLAT regardless of how bullish the little history looks.
    short = ScanCursor(series=tuple([0.02] * 5), t=5)
    assert m.signal_at(short) is Side.FLAT
    # Enough history -> the filter fires.
    ok = ScanCursor(series=tuple([0.02] * 20), t=20)
    assert m.signal_at(ok) is not Side.FLAT
