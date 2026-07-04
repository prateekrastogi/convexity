"""Unit tests for MODELS.hazard -- Phase-C candidate #7 discrete-time reversal-hazard ExitModel.

SYNTHETIC ONLY, no real lake (SYSTEM §C.8). The two mandated tests plus conformance/guards:
  (1) KNOWN-generating-process RECOVERY -- durations simulated from a hazard whose truth we control
      are fit back and the estimator RECOVERS that truth within tolerance. This covers BOTH the
      duration coefficients (β0, β_dur) AND the COVARIATE coefficient β_cov (a DGP with a true nonzero
      covariate effect is fit back and the covariate coefficient recovered) -- the covariates are
      genuinely fitted, not frozen at 0; plus the monotonic-hold sanity (a positive duration
      coefficient => hazard monotone increasing in `u` => a well-defined "hold found by the data"
      threshold crossing) and its ENFORCEMENT (a negative-duration-dependence fold RAISES);
  (2) NO-LOOKAHEAD PINNING -- the exit decision at bar `t` reads ONLY F_{t-1} info; a wiring that
      reads bar `t` (or later) through the causal `ScanCursor` RAISES `PurityError` / breaks a test.
      The pinning is NON-VACUOUS: it is asserted with a GENUINELY-FITTED nonzero covariate coefficient
      (so a leaky covariate builder that read bar t WOULD change the hazard and break the test).
The harness / sibling Phase-C models are NOT imported (interface-driven; SYSTEM §C.3). No lake, no
metric, no adopt decision is touched anywhere here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from entry_exit.base import STOP_FLOOR, Cadence, ExitModel, PurityError, ScanCursor, TrainFold
from entry_exit.hazard import (
    DiscreteHazardFit,
    HazardReversalExit,
    hazard_prob,
    simulate_reversal_durations,
)


def _durations_fold(durations: list[int], warmup: int = 0) -> TrainFold:
    """A TrainFold carrying observed reversal durations as its per-day causal series (the survival response)."""
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(durations)))
    return TrainFold(days=days, series=tuple(float(d) for d in durations), warmup=warmup)


def _zero_cov(n: int, p: int) -> list[list[float]]:
    """`n` per-trade covariate rows, each `p` zeros -- a covariate design with no train response.

    Used where the test cares about the DURATION recovery / a decision path, not the covariate
    coefficient: a genuinely-fitted-but-uninformative covariate (its train response is flat) sits at
    ~0 without needing the test to hand-craft an effect. Distinct from the D1 facade (coefficients
    NEVER estimated): here the coefficient IS estimated -- it just happens to land near 0 on a flat
    covariate, which the covariate-recovery test below (with a real nonzero effect) proves.
    """
    return [[0.0] * p for _ in range(n)]


# ==================================================================================================
# Protocol conformance (runtime_checkable structural typing -- it IS an ExitModel)
# ==================================================================================================
def test_hazard_conforms_to_exit_model():
    m = HazardReversalExit()  # default: return covariate ON (n_cov == 1)
    assert isinstance(m, ExitModel)                       # fit + exit_now + k structurally present
    durs = [2, 3, 4, 5, 3, 4]
    assert m.fit(_durations_fold(durs), covariates=_zero_cov(len(durs), 1)) is m  # fit returns self
    assert isinstance(m.k, int) and m.k >= 0                # parsimony count present, non-negative int


def test_k_counts_fitted_params_plus_threshold():
    # k = (β0, β_dur) + one GENUINELY-FITTED coef per covariate + the fitted h* threshold (SYSTEM §C.4).
    assert HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).k == 3  # 2 coef + h*
    assert HazardReversalExit(use_return_covariate=True, trend_halflife=0.0).k == 4   # + return cov
    assert HazardReversalExit(use_return_covariate=True, trend_halflife=5.0).k == 5   # + trend cov


def test_constructor_validates_params():
    with pytest.raises(ValueError):
        HazardReversalExit(exit_hazard_quantile=0.0)   # must be strictly in (0, 1)
    with pytest.raises(ValueError):
        HazardReversalExit(exit_hazard_quantile=1.0)
    with pytest.raises(ValueError):
        HazardReversalExit(trend_halflife=-1.0)


def test_methods_raise_before_fit():
    m = HazardReversalExit()
    cur = ScanCursor(series=(0.0, 0.0, 0.0), t=2)
    with pytest.raises(RuntimeError):
        m.exit_now(cur, held_days=1, held_return=0.1)
    with pytest.raises(RuntimeError):
        _ = m.coef
    with pytest.raises(RuntimeError):
        _ = m.h_star


# ==================================================================================================
# (1) KNOWN-generating-process RECOVERY test (fit recovers the hazard truth within tolerance)
# ==================================================================================================
def test_recovery_of_known_hazard_coefficients():
    # Controlled DGP: a discrete-time hazard with a KNOWN (β0, β_dur) -- positive duration dependence.
    true_coef = (-3.0, 1.2)  # low base hazard, hazard rises with log-duration
    durations = simulate_reversal_durations(true_coef, n=8000, seed=7)
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold(durations))
    b0, b_dur = m.coef
    # The MLE recovers the generating coefficients within sampling tolerance (large-n consistency).
    assert b0 == pytest.approx(true_coef[0], abs=0.20)
    assert b_dur == pytest.approx(true_coef[1], abs=0.20)


def test_recovery_of_known_covariate_coefficient():
    # D3 FIX: the estimator recovers the truth of the FULL covariate model, not just the duration
    # sub-model. Controlled DGP: a hazard with a KNOWN nonzero covariate effect β_cov; each trade gets
    # its own covariate value, its duration drawn from that covariate-conditioned hazard. fit() must
    # recover β_cov (~ the truth), proving the covariate coefficient is GENUINELY estimated (not the
    # D1 facade where it is hardcoded to 0).
    rng = np.random.default_rng(19)
    n = 12000
    true_coef = (-3.0, 0.8, 1.5)  # (β0, β_dur, β_cov) -- a real, sizeable covariate effect
    cov_rows = [[float(rng.normal(0.0, 1.0))] for _ in range(n)]
    durations = simulate_reversal_durations(true_coef, n=n, seed=23, covariates=cov_rows)
    m = HazardReversalExit(use_return_covariate=True, trend_halflife=0.0).fit(
        _durations_fold(durations), covariates=cov_rows)
    b0, b_dur, b_cov = m.coef
    assert b0 == pytest.approx(true_coef[0], abs=0.20)
    assert b_dur == pytest.approx(true_coef[1], abs=0.20)
    # The load-bearing assertion: the covariate coefficient is recovered NONZERO near its truth --
    # it is fitted, not frozen at 0 (D1). A facade fit (β_cov == 0) FAILS this by ~1.5.
    assert b_cov == pytest.approx(true_coef[2], abs=0.25)
    assert abs(b_cov) > 0.5  # unmistakably nonzero -- the covariate is not inert


def test_covariate_actually_moves_the_hazard_and_decision():
    # D1 FIX (behavioural): with a genuinely-fitted covariate coefficient, held_return must MOVE the
    # hazard (and can flip the exit decision) -- the exact inertness the defect reported (both +5 and
    # -5 giving the identical 0.15895) must be gone.
    rng = np.random.default_rng(31)
    n = 12000
    true_coef = (-2.5, 0.6, 2.0)  # strong positive covariate effect -> higher return => higher hazard
    cov_rows = [[float(rng.normal(0.0, 1.0))] for _ in range(n)]
    durations = simulate_reversal_durations(true_coef, n=n, seed=41, covariates=cov_rows)
    m = HazardReversalExit(use_return_covariate=True, trend_halflife=0.0).fit(
        _durations_fold(durations), covariates=cov_rows)
    cur = ScanCursor(series=(0.0,) * 10, t=5, cadence=Cadence.DAILY)
    h_hi = m.hazard_at(cur, held_days=2, held_return=+2.0)
    h_lo = m.hazard_at(cur, held_days=2, held_return=-2.0)
    assert h_hi != h_lo                        # held_return is NOT inert (the D1 facade is gone)
    assert h_hi > h_lo                          # sign matches the positive fitted β_cov


def test_recovery_of_known_hazard_curve():
    # Stronger: the FITTED hazard CURVE h(u) matches the KNOWN truth at every early duration.
    true_coef = (-2.5, 0.9)
    durations = simulate_reversal_durations(true_coef, n=10000, seed=11)
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold(durations))
    cur = ScanCursor(series=(), t=0)  # no covariates enabled -> the past window is unused here
    for u in range(0, 8):
        fitted = m.hazard_at(cur, held_days=u, held_return=0.0)
        truth = hazard_prob(true_coef, u)
        assert fitted == pytest.approx(truth, abs=0.03)


def test_monotonic_hold_sanity():
    # Positive duration dependence => the reversal hazard is MONOTONE INCREASING in u, so there is a
    # well-defined "hold found by the data": a duration u* below which we HOLD (h <= h*) and at/above
    # which we EXIT (h > h*). This is the load-bearing dynamic-hold property (SIGNAL §C.3(c)).
    true_coef = (-4.0, 1.5)
    durations = simulate_reversal_durations(true_coef, n=8000, seed=3)
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold(durations))
    cur = ScanCursor(series=(), t=0)
    hazards = [m.hazard_at(cur, held_days=u, held_return=0.0) for u in range(0, 12)]
    assert all(hazards[u + 1] >= hazards[u] - 1e-12 for u in range(len(hazards) - 1))  # non-decreasing
    # And the exit decision flips exactly once, HOLD -> EXIT, as u grows (a single crossing of h*).
    decisions = [m.exit_now(cur, held_days=u, held_return=0.0) for u in range(0, 12)]
    first_exit = next(i for i, d in enumerate(decisions) if d)
    assert all(not d for d in decisions[:first_exit])   # HOLD before the crossing
    assert all(d for d in decisions[first_exit:])        # EXIT at/after the crossing (monotone hold)


def test_negative_duration_dependence_is_rejected():
    # D6 FIX: a fold with NEGATIVE duration dependence fits β_dur < 0 (hazard DECREASING in u, which
    # would INVERT the reversal-exit -- exit early, hold longer as the trade ages). The monotone-hold
    # guard must RAISE rather than silently ship the inverted logic.
    inverted = [1, 1, 1, 1, 1, 20]  # short durations dominate then one long -> decreasing hazard MLE
    with pytest.raises(ValueError, match="β_dur"):
        HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
            _durations_fold(inverted))
    # ...and the diagnostic escape hatch fits it WITHOUT raising (for inspection only, never shipped).
    m = HazardReversalExit(
        use_return_covariate=False, trend_halflife=0.0, require_monotone_hold=False
    ).fit(_durations_fold(inverted))
    assert m.coef[1] < 0.0  # the (rejected-by-default) decreasing-hazard fit is inspectable here


def test_incumbent_style_constant_hazard_gives_finite_hold():
    # A near-flat hazard fold (durations tightly clustered ~ the incumbent's fixed horizon) still fits
    # and produces a finite, sensible crossing -- the dynamic hold degrades gracefully to a fixed-ish
    # horizon when the data show no duration dependence (the incumbent is the k-minimal limit).
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold([5, 5, 5, 4, 5, 6, 5, 5, 4, 6]))
    cur = ScanCursor(series=(), t=0)
    # Some early duration holds and some later duration exits (a real crossing exists).
    assert m.exit_now(cur, held_days=0, held_return=0.0) is False
    assert m.exit_now(cur, held_days=50, held_return=0.0) is True


def test_h_star_is_per_trade_not_survival_time_weighted():
    # D6 (threshold) FIX: for the SAME frozen fit, h* is the quantile over ONE hazard per trade
    # (h(T_i|x_i) at each trade's OBSERVED reversal duration), NOT over all person-period survival
    # rows. We reconstruct BOTH quantiles from the model's own frozen coef and assert h_star matches
    # the PER-TRADE one and NOT the (survival-time-weighted) row one -- isolating the weighting choice
    # from the coefficient fit (adding a trade would re-fit coef and confound the two effects).
    # A monotone-increasing spread (each trade's reversal at a distinct later u): the per-trade
    # endpoint hazards h(T_i) sit HIGH on the rising curve, while the bulk of survival rows (u < T_i)
    # sit LOW -- so the two weightings' medians diverge widely. This is a VALID (β_dur > 0) fit, so no
    # diagnostic escape hatch is needed.
    durs = [4, 5, 6, 7, 8, 9, 10, 11]
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(_durations_fold(durs))
    coef = m.coef
    assert coef[1] > 0.0  # monotone-increasing hazard (the guard passed)
    # Per-trade hazards: ONE value per trade, at its observed reversal duration T_i.
    per_trade = np.array([hazard_prob(coef, T) for T in durs])
    # Row-weighted hazards: every person-period row u=0..T_i for every trade (survival-time-weighted).
    all_rows = np.array([hazard_prob(coef, u) for T in durs for u in range(T + 1)])
    q = 0.5
    per_trade_q = float(np.quantile(per_trade, q))
    row_q = float(np.quantile(all_rows, q))
    assert per_trade_q - row_q > 0.1       # the two weightings genuinely, widely differ here
    assert m.h_star == pytest.approx(per_trade_q, abs=1e-9)  # h* uses the PER-TRADE quantile (the fix)
    assert m.h_star != pytest.approx(row_q, abs=1e-6)        # NOT the row-weighted one (the defect)


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING test (FAILS if the decision reads bar t or later) -- NON-VACUOUS
# ==================================================================================================
def _fitted_trend_model() -> HazardReversalExit:
    """A model with a GENUINELY-FITTED nonzero TREND coefficient (so the pinning test is non-vacuous).

    DGP: each trade's duration is drawn from a hazard whose covariate is that trade's trend value, with
    a real nonzero β_trend. The fitted trend coefficient is therefore nonzero -- so if `_trend` leaked
    a future bar, the hazard WOULD change and the pinning test WOULD break (D2: the old test could not
    fail because the trend coefficient was frozen at 0 and multiplied any trend value to 0).
    """
    rng = np.random.default_rng(101)
    n = 12000
    true_coef = (-3.0, 0.7, 1.8)  # (β0, β_dur, β_trend) -- a real, sizeable trend effect
    trend_vals = [[float(rng.normal(0.0, 1.0))] for _ in range(n)]
    durations = simulate_reversal_durations(true_coef, n=n, seed=202, covariates=trend_vals)
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=3.0).fit(
        _durations_fold(durations), covariates=trend_vals)
    assert abs(m.coef[2]) > 0.5  # the trend coefficient is genuinely fitted nonzero (non-vacuous)
    return m


def test_pinning_exit_reads_only_causal_past():
    # The trend covariate is built from cursor.past() (indices 0..t-1); mutating bar t (and later)
    # MUST NOT change the decision -- if it did, the model would be reading F_t (a leak). NON-VACUOUS:
    # the trend coefficient is fitted NONZERO (see _fitted_trend_model), so a leak WOULD move the
    # hazard (unlike the D2 facade where β_trend == 0 made bar-t mutations inconsequential by design).
    m = _fitted_trend_model()
    past = (0.01, 0.02, -0.01, 0.03)
    cur_a = ScanCursor(series=(*past, 999.0, -999.0), t=4, cadence=Cadence.DAILY)   # bar t = +999
    cur_b = ScanCursor(series=(*past, -999.0, 999.0), t=4, cadence=Cadence.DAILY)   # bar t = -999
    # Same causal past (series[:4]) => identical hazard/decision regardless of bar t or later.
    h_a = m.hazard_at(cur_a, held_days=2, held_return=0.0)
    h_b = m.hazard_at(cur_b, held_days=2, held_return=0.0)
    assert h_a == h_b
    assert m.exit_now(cur_a, held_days=2, held_return=0.0) == m.exit_now(cur_b, held_days=2, held_return=0.0)


def test_pinning_leaky_trend_that_reads_bar_t_would_change_the_hazard():
    # D2 FIX (the certificate the vacuous test lacked): prove the pinning test has TEETH. With the
    # trend coefficient fitted NONZERO, a `_trend` that leaked bar t (a same-bar read) produces a
    # DIFFERENT hazard on the two cursors that differ only at bar t -- so the clean-vs-leaky contrast
    # is observable. We simulate the leak by monkeypatching `_trend` to read cursor.series[cursor.t]
    # (the exact maximally-leaky mutation the defect described) and assert the two hazards now DIFFER,
    # which is precisely what test_pinning_exit_reads_only_causal_past forbids for the real (causal)
    # builder. Thus a genuine future-reading leak DOES break the pinning test (it did not before).
    m = _fitted_trend_model()
    past = (0.01, 0.02, -0.01, 0.03)
    cur_a = ScanCursor(series=(*past, 999.0, -999.0), t=4, cadence=Cadence.DAILY)
    cur_b = ScanCursor(series=(*past, -999.0, 999.0), t=4, cadence=Cadence.DAILY)
    clean_a = m.hazard_at(cur_a, held_days=2, held_return=0.0)
    clean_b = m.hazard_at(cur_b, held_days=2, held_return=0.0)
    assert clean_a == clean_b  # the real causal builder is pinned (identical)
    # Now install the leak and re-evaluate: a `_trend` that reads bar t (cursor.series[cursor.t]) --
    # the maximally-leaky mutation the defect described -- makes the two hazards diverge. (`_trend` is
    # called as `self._trend(cursor)`, so an instance-attribute function receives `cursor` only.)
    leaky = HazardReversalExit(use_return_covariate=False, trend_halflife=3.0)
    leaky._fit = m._fit  # reuse the frozen (nonzero-β_trend) fit
    leaky._trend = lambda cursor: float(cursor.series[cursor.t])  # type: ignore[assignment,method-assign]
    leak_a = leaky.hazard_at(cur_a, held_days=2, held_return=0.0)
    leak_b = leaky.hazard_at(cur_b, held_days=2, held_return=0.0)
    assert leak_a != leak_b  # the leak IS observable => the pinning test genuinely certifies causality


def test_pinning_cursor_read_of_bar_t_raises():
    # A wiring that tried to read bar t (the same-bar leak) through the causal cursor MUST raise --
    # this is the guard a leaky refactor of the covariate builder would trip (SIGNAL §C.1 / SYSTEM §C.2).
    cur = ScanCursor(series=(0.01, 0.02, 0.03, 0.04), t=2, cadence=Cadence.DAILY)
    assert cur.past() == (0.01, 0.02)          # only F_{t-1} exposed
    with pytest.raises(PurityError):
        cur.read(2)                            # same bar t -> leak
    with pytest.raises(PurityError):
        cur.read(3)                            # future bar -> leak


def test_pinning_probe_same_bar_read_raises_via_the_models_own_guard():
    # D4 FIX: the model carries its OWN pinning hook wired to require_causal (not only base.py's guard
    # in isolation). `_probe_same_bar_read` performs the exact same-bar read a leaky covariate builder
    # would do and RAISES PurityError -- and the live decision path routes cursor.read(t-1), so the
    # tripwire is in the model's decision machinery, matching the docstring's causal-safety claim.
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold([2, 3, 4, 5, 3, 4]))
    cur = ScanCursor(series=(0.01, 0.02, 0.03, 0.04), t=2, cadence=Cadence.DAILY)
    with pytest.raises(PurityError):
        m._probe_same_bar_read(cur)            # the model's own same-bar read tripwire
    # And the live decision at a legal cursor does NOT raise (it reads only bar t-1, F_{t-1}-measurable).
    assert m.exit_now(cur, held_days=1, held_return=0.0) in (True, False)


def test_pinning_held_days_is_the_duration_not_cursor_derived():
    # The duration u is the FIRST-CLASS `held_days` covariate, F_{t-1}-measurable by construction (a
    # count of already-elapsed bars); the hazard changes with held_days at a FIXED cursor.t, proving u
    # is not smuggled from a future cursor read (base.py ExitModel contract / EDGE #7).
    m = HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
        _durations_fold(simulate_reversal_durations((-3.0, 1.2), n=6000, seed=5)))
    cur = ScanCursor(series=(0.0,) * 10, t=7, cadence=Cadence.DAILY)  # t fixed
    h_early = m.hazard_at(cur, held_days=1, held_return=0.0)
    h_late = m.hazard_at(cur, held_days=8, held_return=0.0)
    assert h_late > h_early   # duration dependence enters via held_days (u), not via cursor.t


# ==================================================================================================
# fit covariate-design validation (the covariates are never silently dropped -- D1 guard)
# ==================================================================================================
def test_fit_requires_covariate_rows_when_covariates_enabled():
    # D1 FIX: a model that ENABLES a covariate must be GIVEN per-trade covariate rows to fit against --
    # it may not silently proceed with the covariate frozen at 0 (the facade). Omitting them RAISES.
    with pytest.raises(ValueError, match="covariate"):
        HazardReversalExit(use_return_covariate=True, trend_halflife=0.0).fit(
            _durations_fold([2, 3, 4, 5]))                       # covariates enabled but none supplied


def test_fit_rejects_covariates_when_none_enabled():
    with pytest.raises(ValueError, match="no covariates"):
        HazardReversalExit(use_return_covariate=False, trend_halflife=0.0).fit(
            _durations_fold([2, 3, 4, 5]), covariates=[[0.0], [0.0], [0.0], [0.0]])


def test_fit_rejects_covariate_shape_mismatch():
    m = HazardReversalExit(use_return_covariate=True, trend_halflife=0.0)  # n_cov == 1
    with pytest.raises(ValueError, match="one row per"):
        m.fit(_durations_fold([2, 3, 4]), covariates=[[0.0], [0.0]])        # 2 rows for 3 durations
    m2 = HazardReversalExit(use_return_covariate=True, trend_halflife=3.0)  # n_cov == 2
    with pytest.raises(ValueError, match="exactly 2"):
        m2.fit(_durations_fold([2, 3, 4]), covariates=[[0.0], [0.0], [0.0]])  # 1 value, need 2


# ==================================================================================================
# The RETAINED -12% stop floor (an exit model may never loosen the loss cap -- SIGNAL §C.3)
# ==================================================================================================
def test_stop_floor_is_retained_and_unchanged():
    durs = [2, 3, 4, 5]
    m = HazardReversalExit().fit(_durations_fold(durs), covariates=_zero_cov(len(durs), 1))
    assert m.stop_floor == STOP_FLOOR == 0.12  # the loss cap this model surfaces but never loosens
    # exit_now is a boolean "close this WINNER early?" decision -- it never returns a loss threshold,
    # so it structurally cannot loosen the -12% stop (the stop stays the floor, SIGNAL §C.3).
    cur = ScanCursor(series=(0.0, 0.0, 0.0), t=2)
    assert m.exit_now(cur, held_days=0, held_return=0.5) in (True, False)


# ==================================================================================================
# Fit-input validation + the shared hazard definition
# ==================================================================================================
def test_fit_rejects_empty_and_negative_durations():
    with pytest.raises(ValueError):
        HazardReversalExit(use_return_covariate=False).fit(_durations_fold([]))  # no observed durations
    with pytest.raises(ValueError):
        HazardReversalExit(use_return_covariate=False).fit(
            _durations_fold([2, -1, 3]))                        # a negative duration is malformed


def test_hazard_prob_shape_and_bounds():
    # h in (0, 1); coef length must be 2 + n_covariates; u must be non-negative.
    assert 0.0 < hazard_prob((-2.0, 0.5), 3) < 1.0
    assert 0.0 < hazard_prob((-2.0, 0.5, 1.0), 3, (0.2,)) < 1.0
    # A large positive eta saturates toward 1 (clamped below 1), a large negative toward 0 (above 0).
    assert hazard_prob((50.0, 0.0), 0) < 1.0
    assert hazard_prob((-50.0, 0.0), 0) > 0.0
    with pytest.raises(ValueError):
        hazard_prob((-2.0, 0.5), -1)              # negative u
    with pytest.raises(ValueError):
        hazard_prob((-2.0, 0.5), 3, (0.1, 0.2))   # coef length != 2 + n_covariates


def test_discrete_hazard_fit_is_immutable_value_object():
    f = DiscreteHazardFit(coef=(-1.0, 0.5), h_star=0.3, n_covariates=0)
    assert f.coef == (-1.0, 0.5) and f.h_star == pytest.approx(0.3) and f.n_covariates == 0
    with pytest.raises(AttributeError):          # __slots__ -> no stray attributes
        f.extra = 1  # type: ignore[attr-defined]


def test_simulator_is_deterministic_given_seed():
    a = simulate_reversal_durations((-2.5, 0.9), n=200, seed=42)
    b = simulate_reversal_durations((-2.5, 0.9), n=200, seed=42)
    assert a == b                                 # reproducible synthetic fixture (SYSTEM §C.8)
    assert all(isinstance(d, int) and d >= 0 for d in a)
    # A higher base hazard reverses sooner on average (a sanity check on the generator's direction).
    hi = simulate_reversal_durations((0.0, 0.0), n=2000, seed=1)   # h(u)=0.5 flat -> short durations
    lo = simulate_reversal_durations((-4.0, 0.0), n=2000, seed=1)  # h(u)~0.018 flat -> long durations
    assert sum(hi) / len(hi) < sum(lo) / len(lo)


def test_simulator_covariate_shifts_durations():
    # The covariate-aware simulator: a positive covariate value + positive β_cov shortens durations
    # (higher hazard). Sanity check that the DGP the covariate-recovery test relies on is directional.
    hi_cov = simulate_reversal_durations(
        (-3.0, 0.0, 2.0), n=3000, seed=9, covariates=[[1.0]] * 3000)   # high covariate -> high hazard
    lo_cov = simulate_reversal_durations(
        (-3.0, 0.0, 2.0), n=3000, seed=9, covariates=[[-1.0]] * 3000)  # low covariate -> low hazard
    assert sum(hi_cov) / len(hi_cov) < sum(lo_cov) / len(lo_cov)
