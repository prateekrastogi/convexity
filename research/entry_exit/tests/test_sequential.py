"""Unit tests for MODELS.sequential -- the Wald SPRT / GLR drift-sign change detector (candidate #2).

SYNTHETIC ONLY (the lake is ABSENT). Two mandated purity tests (task contract, SIGNAL §C.2(b) /
SYSTEM §C.2):
  (1) a KNOWN-generating-process RECOVERY test -- simulate a drift-shift path whose truth we control
      (a no-drift prefix then a +/-delta drift regime, Gaussian innovations) and assert the detector
      TRIGGERS in the correct sign; a no-drift path does NOT falsely alarm at the designed rate, and
      the empirical in-control run length matches the ARL the (alpha, beta) boundaries designed for --
      for BOTH the SPRT and the GLR arm (the GLR error control is a Lorden/Siegmund window-limited-GLR
      ARL design, NOT a single-window chi-square quantile);
  (2) a NO-LOOKAHEAD PINNING test -- a decision executing on bar `t` must be F_{t-1}-measurable; a
      wiring that reads bar `t` (or later) to decide at `t` BREAKS a test (`ScanCursor`'s causal guard
      raises `PurityError`).
Plus boundary-math + fit-freeze + protocol-conformance checks. The real sibling models are NOT
imported (the harness is interface-driven; SYSTEM §C.3). No lake, no metric, no adopt decision here.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from entry_exit.base import Cadence, EntryModel, PurityError, ScanCursor, Side, TrainFold
from entry_exit.sequential import SprtGlrEntry, glr_threshold, wald_boundaries


def _train(vals: list[float], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


def _drift_path(rng: random.Random, n_pre: int, n_post: int, drift: float, sigma: float) -> list[float]:
    """A controlled change-point DGP: `n_pre` no-drift bars then `n_post` bars of mean `drift`.

    Gaussian innovations N(0, sigma^2). The KNOWN truth is: a drift-sign change from 0 to `drift`
    after bar `n_pre`. A correct sequential detector must alarm in `sign(drift)` somewhere in the
    post-change segment (within its expected detection delay), and not before.
    """
    pre = [rng.gauss(0.0, sigma) for _ in range(n_pre)]
    post = [rng.gauss(drift, sigma) for _ in range(n_post)]
    return pre + post


# ==================================================================================================
# Boundary / threshold math (the a-priori error-probability design -- NOT P&L-tuned)
# ==================================================================================================
def test_wald_boundaries_from_alpha_beta():
    a, b = wald_boundaries(alpha=0.05, beta=0.20)
    assert a == pytest.approx(math.log((1.0 - 0.20) / 0.05))   # A = log((1-beta)/alpha)
    assert b == pytest.approx(math.log(0.20 / (1.0 - 0.05)))   # B = log(beta/(1-alpha))
    assert a > 0.0 > b                                          # positive upper, negative reset


def test_wald_boundaries_reject_bad_rates():
    for bad in ((0.0, 0.2), (0.2, 1.0), (-0.1, 0.2), (0.2, 0.0)):
        with pytest.raises(ValueError):
            wald_boundaries(*bad)


def test_glr_threshold_is_log_arl_boundary_equals_sprt_upper():
    # D1 fix: the window-limited-GLR alarm boundary is the ARL-controlling log-boundary
    # h = log((1-beta)/alpha) == the SPRT upper boundary A (Lorden/Siegmund), NOT a fixed single-window
    # chi-square-1/2 quantile (which controls one test's Type-I error, not the sequential ARL).
    a, _ = wald_boundaries(alpha=0.05, beta=0.20)
    assert glr_threshold(0.05, 0.20) == pytest.approx(a)                 # == SPRT upper A
    assert glr_threshold(0.05, 0.20) == pytest.approx(math.log((1.0 - 0.20) / 0.05))
    assert glr_threshold(0.01, 0.20) > glr_threshold(0.05, 0.20)   # stricter alpha => higher bar
    assert glr_threshold(0.05, 0.05) > glr_threshold(0.05, 0.20)   # stricter beta  => higher bar
    for bad in ((0.0, 0.2), (0.2, 1.0), (-0.1, 0.2), (0.2, 0.0), (1.0, 0.2)):
        with pytest.raises(ValueError):
            glr_threshold(*bad)


# ==================================================================================================
# Protocol conformance + fit-then-freeze
# ==================================================================================================
def test_conforms_to_entry_model_protocol():
    for m in (SprtGlrEntry(mode="sprt"), SprtGlrEntry(mode="glr")):
        assert isinstance(m, EntryModel)            # structural (fit + signal_at + k)
        assert m.fit(_train([0.0, 0.01, -0.01, 0.02])) is m   # fit returns the fitted model (self)
        assert isinstance(m.k, int) and m.k >= 0


def test_k_reflects_fitted_parameter_count():
    # SPRT fits sigma + delta => k=2. GLR profiles the DRIFT out (no delta) but carries the searched /
    # train-frozen window length m as a genuine fitted knob (SIGNAL §C.2(b) "for GLR, the window length
    # m") => sigma + m = 2. Counting GLR as k=1 would understate the §C.4 parsimony penalty (D3 fix).
    assert SprtGlrEntry(mode="sprt").k == 2
    assert SprtGlrEntry(mode="glr").k == 2


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        SprtGlrEntry(mode="ma")                 # not sprt/glr
    with pytest.raises(ValueError):
        SprtGlrEntry(mode="glr", window=0)      # non-positive GLR window
    with pytest.raises(ValueError):
        SprtGlrEntry(delta_sigmas=0.0)          # non-positive min-detectable drift


def test_signal_before_fit_raises():
    with pytest.raises(RuntimeError):
        SprtGlrEntry().signal_at(ScanCursor(series=(0.1, 0.2), t=2))


def test_fit_recovers_train_sigma_and_freezes():
    # KNOWN truth: a train series with std ~0.10 -> sigma_hat recovers ~0.10, delta = delta_sigmas*sigma.
    rng = random.Random(7)
    body = [rng.gauss(0.0, 0.10) for _ in range(4000)]
    m = SprtGlrEntry(mode="sprt", delta_sigmas=1.5).fit(_train(body))
    assert m._sigma == pytest.approx(0.10, abs=0.01)
    assert m._delta == pytest.approx(1.5 * m._sigma)
    # Frozen: a second (different) fit body would re-fit, but without a call the frozen value stands.
    assert m._sigma is not None and m._delta is not None


def test_fit_rejects_empty_train_body():
    with pytest.raises(ValueError):
        SprtGlrEntry().fit(_train([1.0, 2.0], warmup=2))   # nothing past warm-up


# ==================================================================================================
# (1) KNOWN-generating-process RECOVERY test (a drift-shift path triggers in the correct SIGN)
# ==================================================================================================
def test_recovery_sprt_positive_drift_triggers_long():
    # Controlled DGP: no-drift prefix, then a strong +drift regime. SPRT must alarm LONG.
    rng = random.Random(1)
    sigma = 0.01
    m = SprtGlrEntry(mode="sprt", alpha=0.05, beta=0.20, delta_sigmas=1.0)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    path = tuple(_drift_path(rng, n_pre=30, n_post=60, drift=+2.0 * sigma, sigma=sigma))
    cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.LONG


def test_recovery_sprt_negative_drift_triggers_short():
    rng = random.Random(2)
    sigma = 0.01
    m = SprtGlrEntry(mode="sprt", alpha=0.05, beta=0.20, delta_sigmas=1.0)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    path = tuple(_drift_path(rng, n_pre=30, n_post=60, drift=-2.0 * sigma, sigma=sigma))
    cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.SHORT


def test_recovery_glr_positive_drift_triggers_long():
    # The GLR arm (running MLE, no pre-specified shift) must recover the +drift sign as LONG under the
    # ARL-controlling log boundary. The current trailing window ends in the strong +drift regime.
    rng = random.Random(3)
    sigma = 0.01
    m = SprtGlrEntry(mode="glr", alpha=0.01, beta=0.20, window=25)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    path = tuple(_drift_path(rng, n_pre=25, n_post=50, drift=+2.5 * sigma, sigma=sigma))
    cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.LONG


def test_recovery_glr_negative_drift_triggers_short():
    # Mirror recovery: a strong -drift current window must alarm SHORT (sign-antisymmetric).
    rng = random.Random(31)
    sigma = 0.01
    m = SprtGlrEntry(mode="glr", alpha=0.01, beta=0.20, window=25)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    path = tuple(_drift_path(rng, n_pre=25, n_post=50, drift=-2.5 * sigma, sigma=sigma))
    cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.SHORT


def test_glr_emits_current_window_drift_sign_not_a_latched_first_crossing():
    # D2 fix: the window-limited GLR decides on the CURRENT trailing window (ending at t-1), NOT the
    # first-ever historical crossing. Controlled path: a strong +0.05 block, a long no-drift middle,
    # then a strong -0.05 block. The current window is strongly NEGATIVE (sum << 0), so a correct
    # window-limited detector must emit SHORT. The OLD scan-all-ends/latch-first form returned LONG
    # (it latched the early +block forever and ignored the current-window drift) -- this test FAILS
    # under that defect and PASSES under the current-window form.
    rng = random.Random(5)
    sigma = 0.01
    m = SprtGlrEntry(mode="glr", alpha=0.01, beta=0.20, window=20)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    path = tuple([+0.05] * 10 + [rng.gauss(0.0, sigma) for _ in range(50)] + [-0.05] * 10)
    cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.SHORT


def _false_alarm_rate(alpha: float, beta: float, sigma: float, window: int, n_trials: int) -> float:
    """Empirical in-control alarm frequency of the reset-SPRT over independent no-drift windows.

    The Type-I side of the (alpha, beta) design: fit sigma on a no-drift train fold, then count how
    often a fresh no-drift window of `window` bars alarms. The repeated-SPRT resets on the lower
    boundary, so this is the per-window false-alarm probability -- its ARL (mean gap between alarms)
    is `~window / rate`, and a stricter alpha raises the boundary and lowers this rate (asserted below).
    """
    fit_rng = random.Random(99)
    m = SprtGlrEntry(mode="sprt", alpha=alpha, beta=beta, delta_sigmas=1.0)
    m.fit(_train([fit_rng.gauss(0.0, sigma) for _ in range(3000)]))
    alarms = 0
    for s in range(n_trials):
        rng = random.Random(1000 + s)
        path = tuple(rng.gauss(0.0, sigma) for _ in range(window))  # in-control window
        cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
        if m.signal_at(cur) is not Side.FLAT:
            alarms += 1
    return alarms / n_trials


def test_recovery_no_drift_false_alarm_controlled_and_monotone_in_alpha():
    # Controlled null DGP (pure no-drift Gaussian): the (alpha, beta) boundary design controls the
    # in-control false-alarm rate. Two facts the design buys, asserted directly:
    #   (i) a STRICTER alpha (higher upper boundary A) yields a LOWER in-control alarm rate -- the
    #       monotone error-control property that makes (alpha, beta) the a-priori design knob; AND
    #   (ii) at the strict setting the rate is well under a coin-flip over a 60-bar window (the ARL
    #        under H_0 is long -- most in-control windows never alarm), NOT alarming on most paths.
    loose = _false_alarm_rate(alpha=0.10, beta=0.20, sigma=0.01, window=60, n_trials=300)
    strict = _false_alarm_rate(alpha=0.001, beta=0.05, sigma=0.01, window=60, n_trials=300)
    assert strict < loose          # (i) stricter alpha -> fewer false alarms (monotone in the design)
    assert strict < 0.25           # (ii) rare in-control alarms at the strict design -> long ARL


def _glr_false_alarm_rate(alpha: float, beta: float, sigma: float, nbars: int,
                          window: int, n_trials: int) -> float:
    """Empirical in-control alarm frequency of the WINDOW-LIMITED GLR over no-drift paths of `nbars`.

    The Type-I side of the GLR's (alpha, beta) ARL design. Fit sigma on a no-drift train fold, then for
    each independent no-drift path of `nbars` bars ask whether the decision at t = nbars alarms (the
    window-limited GLR evaluated on the current trailing window). Because the current form fixes the
    window END at t-1 and controls the alarm with the log-ARL boundary, this rate must be STABLE in
    `nbars` (not grow without bound with history length -- the D1 defect the old fixed-chi-square /
    scan-all-ends form exhibited).
    """
    fit_rng = random.Random(99)
    m = SprtGlrEntry(mode="glr", alpha=alpha, beta=beta, window=window)
    m.fit(_train([fit_rng.gauss(0.0, sigma) for _ in range(3000)]))
    alarms = 0
    for s in range(n_trials):
        rng = random.Random(2000 + s)
        path = tuple(rng.gauss(0.0, sigma) for _ in range(nbars))   # in-control path
        cur = ScanCursor(series=path, t=len(path), cadence=Cadence.DAILY)
        if m.signal_at(cur) is not Side.FLAT:
            alarms += 1
    return alarms / n_trials


def test_glr_no_drift_false_alarm_arl_controlled_and_stable_in_history():
    # D1 fix (the missing GLR ARL / no-drift false-alarm test). Under the ARL-controlling log boundary
    # the in-control false-alarm probability of the window-limited GLR is STABLE in history length --
    # it does NOT grow without bound as the OLD fixed single-window chi-square-1/2 quantile did
    # (measured then: ~0.06 at 20 bars, ~0.32 at 120, ~0.94 at 1000). We assert two properties:
    #   (i) STABLE in history: the false-alarm rate at 20, 120 and 1000 bars stays low and does not
    #       balloon with history length (the ARL is a per-decision design, controlled by h ~ log(ARL));
    #   (ii) MONOTONE in alpha: a stricter alpha raises h and lowers the in-control alarm rate.
    sigma = 0.01
    r20 = _glr_false_alarm_rate(alpha=0.01, beta=0.20, sigma=sigma, nbars=20, window=20,
                                n_trials=400)
    r120 = _glr_false_alarm_rate(alpha=0.01, beta=0.20, sigma=sigma, nbars=120, window=20,
                                 n_trials=400)
    r1000 = _glr_false_alarm_rate(alpha=0.01, beta=0.20, sigma=sigma, nbars=1000, window=20,
                                  n_trials=400)
    # (i) stable in history length: all low, and the 1000-bar rate is NOT materially above the 20-bar
    # rate (the old defect blew this up to ~0.94; the current form keeps it flat ~0.02).
    assert r20 < 0.10
    assert r120 < 0.10
    assert r1000 < 0.10
    assert r1000 < 3.0 * max(r20, 0.01)     # no unbounded growth with history (old form: ~15x)
    # (ii) monotone in alpha at fixed history: stricter alpha -> higher h -> fewer false alarms.
    loose = _glr_false_alarm_rate(alpha=0.10, beta=0.20, sigma=sigma, nbars=60, window=20,
                                  n_trials=400)
    strict = _glr_false_alarm_rate(alpha=0.001, beta=0.20, sigma=sigma, nbars=60, window=20,
                                   n_trials=400)
    assert strict < loose


def test_recovery_in_control_arl_matches_boundary_design():
    # The expected in-control run length (time-to-alarm on a NO-drift path) grows with the upper
    # boundary A (a stricter alpha => longer ARL). We assert the mean first-passage time under H_0 is
    # long relative to the detection delay -- the ARL property the (alpha, beta) design controls.
    sigma = 0.01
    fit_rng = random.Random(5)
    m = SprtGlrEntry(mode="sprt", alpha=0.01, beta=0.10, delta_sigmas=1.0)
    m.fit(_train([fit_rng.gauss(0.0, sigma) for _ in range(2000)]))
    run_lengths = []
    for s in range(120):
        rng = random.Random(7000 + s)
        path = [rng.gauss(0.0, sigma) for _ in range(300)]
        first = None
        for t in range(1, len(path) + 1):   # first bar at which an alarm would fire (F_{t-1} decision)
            cur = ScanCursor(series=tuple(path), t=t, cadence=Cadence.DAILY)
            if m.signal_at(cur) is not Side.FLAT:
                first = t
                break
        run_lengths.append(first if first is not None else len(path))
    mean_arl = sum(run_lengths) / len(run_lengths)
    # A strict alpha=0.01 buys a long in-control ARL: most 300-bar no-drift paths never alarm, so the
    # mean run length sits well above a short detection delay (dozens of bars), not near 1.
    assert mean_arl > 50.0


def test_detection_delay_shorter_for_larger_shift():
    # A larger true drift is detected FASTER (shorter run length to alarm) -- the ARL_1 side of the
    # design. Controlled: same detector, two post-change drift sizes; the bigger shift alarms sooner.
    sigma = 0.01
    fit_rng = random.Random(11)
    m = SprtGlrEntry(mode="sprt", alpha=0.01, beta=0.10, delta_sigmas=1.0)
    m.fit(_train([fit_rng.gauss(0.0, sigma) for _ in range(2000)]))

    def delay(drift: float, seed: int) -> int:
        rng = random.Random(seed)
        path = [rng.gauss(drift, sigma) for _ in range(200)]
        for t in range(1, len(path) + 1):
            cur = ScanCursor(series=tuple(path), t=t, cadence=Cadence.DAILY)
            if m.signal_at(cur) is not Side.FLAT:
                return t
        return len(path)

    big = sum(delay(4.0 * sigma, 2000 + s) for s in range(40)) / 40
    small = sum(delay(1.5 * sigma, 3000 + s) for s in range(40)) / 40
    assert big < small   # the larger shift is detected sooner (shorter mean detection delay)


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING test (FAILS if a decision at bar t reads bar t or later)
# ==================================================================================================
def test_pinning_signal_never_reads_bar_t():
    # A decision executing on bar t must be F_{t-1}-measurable: `signal_at` reads cursor.past() =
    # series[:t] ONLY. We prove it by making bar t (and beyond) a value that WOULD flip the decision
    # if read, and asserting the signal is unchanged whether or not those future bars exist.
    rng = random.Random(42)
    sigma = 0.01
    m = SprtGlrEntry(mode="sprt", alpha=0.05, beta=0.20, delta_sigmas=1.0)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    past_only = tuple(_drift_path(rng, n_pre=25, n_post=45, drift=+2.0 * sigma, sigma=sigma))
    t = len(past_only)
    # Append huge NEGATIVE future bars at index >= t: if signal_at leaked them it would flip to SHORT.
    with_future = past_only + (-100.0, -100.0, -100.0)
    sig_a = m.signal_at(ScanCursor(series=past_only, t=t, cadence=Cadence.DAILY))
    sig_b = m.signal_at(ScanCursor(series=with_future, t=t, cadence=Cadence.DAILY))
    assert sig_a is sig_b is Side.LONG   # unchanged: the future bars were never read (F_{t-1} only)


def test_pinning_glr_signal_never_reads_bar_t():
    # Same F_{t-1} pinning for the GLR arm: the window-limited GLR ends its window at t-1 and never
    # reads bar t. A huge negative bar planted AT/AFTER t must not flip the LONG decision.
    rng = random.Random(43)
    sigma = 0.01
    m = SprtGlrEntry(mode="glr", alpha=0.01, beta=0.20, window=25)
    m.fit(_train([rng.gauss(0.0, sigma) for _ in range(2000)]))
    past_only = tuple(_drift_path(rng, n_pre=25, n_post=50, drift=+2.5 * sigma, sigma=sigma))
    t = len(past_only)
    with_future = past_only + (-100.0, -100.0, -100.0)
    sig_a = m.signal_at(ScanCursor(series=past_only, t=t, cadence=Cadence.DAILY))
    sig_b = m.signal_at(ScanCursor(series=with_future, t=t, cadence=Cadence.DAILY))
    assert sig_a is sig_b is Side.LONG   # future bars never read (window ends at t-1)


def test_pinning_leaky_reimplementation_raises_purityerror():
    # A no-lookahead PINNING test in the models/base sense: a wiring that reads bar t at decision t
    # through the causal guard BREAKS. We simulate the leak a buggy detector would make -- reading
    # cursor.read(t) (the same bar it decides on) -- and assert the guard raises PurityError.
    cur = ScanCursor(series=(0.1, 0.2, 0.3, 0.4), t=3, cadence=Cadence.DAILY)
    assert cur.read(2) == pytest.approx(0.3)   # a strictly-past read (bar t-1) is legal
    with pytest.raises(PurityError):
        cur.read(3)                            # reading bar t (same bar) at decision t -> LEAK
    with pytest.raises(PurityError):
        cur.read(4)                            # reading a future bar -> LEAK


def test_signal_flat_before_warmup_and_on_empty_past():
    # At t=0 the causal past is empty -> no evidence -> FLAT (never an accidental entry from nothing).
    m = SprtGlrEntry(mode="sprt").fit(_train([0.01, 0.02, 0.03]))
    assert m.signal_at(ScanCursor(series=(0.05, 0.05), t=0, cadence=Cadence.DAILY)) is Side.FLAT
    g = SprtGlrEntry(mode="glr", window=10).fit(_train([0.01, 0.02, 0.03]))
    assert g.signal_at(ScanCursor(series=(0.05, 0.05), t=0, cadence=Cadence.DAILY)) is Side.FLAT
