"""Unit tests for MODELS.ou_exit -- the OU / Leung-Li mean-reversion optimal EXIT (EDGE #6). SYNTH.

Two mandated Phase-C tests (SYSTEM §C.2, task contract), no real lake touched anywhere:
  1. a KNOWN-generating-process RECOVERY test -- simulate an OU path whose truth ``(theta, m, sig)``
     we control, assert ``ou_mle`` recovers it within tolerance, AND assert the Leung-Li ``b*`` the
     solver returns satisfies the paper's smooth-fit OPTIMALITY CONDITION ``F(b*) = (b*-c) F'(b*)``;
  2. a NO-LOOKAHEAD PINNING test -- ``OUExit.exit_now`` deciding at bar ``t`` must be F_{t-1}-
     measurable; a wiring that reads bar ``t`` (or later) via the ``ScanCursor`` guard raises.

The real Phase-C harness / sibling models are NOT imported (interface-driven; SYSTEM §C.3). numpy is
used only to SIMULATE the controlled OU path (the truth we recover) -- never to touch a lake.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings

import numpy as np
import pytest

from entry_exit.base import STOP_FLOOR, Cadence, ExitModel, PurityError, ScanCursor, TrainFold
from entry_exit.ou_exit import (
    OUExit,
    OUParams,
    OUStationarityError,
    fg_F,
    fg_F_prime,
    leung_li_boundary,
    ou_mle,
)


# --- Controlled OU path simulator (the KNOWN truth the recovery test asserts against) -------------
def _simulate_ou(theta: float, m: float, sigma: float, dt: float, n: int, x0: float, seed: int):
    """Exact-transition OU simulation ``dX = theta(m-X)dt + sigma dW`` -- the controlled DGP.

    Uses the EXACT discrete transition (not Euler): ``X_{i+1} = m + (X_i - m) e^{-theta dt} +
    N(0, sigma^2 (1 - e^{-2 theta dt}) / (2 theta))`` -- the same law ``ou_mle`` inverts, so a good
    estimator recovers the truth as ``n -> inf``. Returns a length-``n`` path as a Python tuple.
    """
    rng = np.random.default_rng(seed)
    b = math.exp(-theta * dt)
    sd = math.sqrt(sigma * sigma * (1.0 - b * b) / (2.0 * theta))
    x = np.empty(n)
    x[0] = x0
    eps = rng.normal(0.0, sd, size=n)
    for i in range(1, n):
        x[i] = m + (x[i - 1] - m) * b + eps[i]
    return tuple(float(v) for v in x)


def _train(vals: tuple[float, ...], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=vals, warmup=warmup)


# ==================================================================================================
# Protocol conformance -- OUExit is a structural ExitModel (fit + exit_now + k), no harness import
# ==================================================================================================
def test_ouexit_conforms_to_exit_model():
    m = OUExit()
    assert isinstance(m, ExitModel)                 # structural conformance
    assert isinstance(m.k, int) and m.k == 3        # theta, m, sigma fit on train (r, c fixed)


def test_ouexit_constructor_guards():
    with pytest.raises(ValueError):
        OUExit(r=0.0)          # discount must be > 0 (else the F/G integral diverges)
    with pytest.raises(ValueError):
        OUExit(dt=0.0)


# ==================================================================================================
# (1a) KNOWN-generating-process RECOVERY: ou_mle recovers the simulated (theta, m, sigma)
# ==================================================================================================
def test_recovery_ou_mle_recovers_known_params():
    theta_true, m_true, sigma_true, dt_ = 0.30, 0.10, 0.05, 1.0
    path = _simulate_ou(theta_true, m_true, sigma_true, dt_, n=60000, x0=m_true, seed=7)
    p = ou_mle(path, dt=dt_)
    # A consistent estimator on a long controlled path recovers the truth within a loose tolerance.
    assert p.theta == pytest.approx(theta_true, rel=0.15)
    assert p.m == pytest.approx(m_true, abs=0.02)
    assert p.sigma == pytest.approx(sigma_true, rel=0.10)


def test_recovery_ou_mle_is_the_ar1_closed_form():
    # Recovery is EXACT for the AR(1) it inverts: build X_{i+1} = a + b X_i (tiny noise) and
    # check theta = -ln(b)/dt, m = a/(1-b) come back. (A tiny noise keeps Var[eps] > 0 for sigma.)
    b_true, a_true, dt_ = 0.8, 0.02, 1.0  # -> theta = -ln0.8, m = 0.02/0.2 = 0.1
    rng = np.random.default_rng(3)
    x = [0.0]
    for _ in range(40000):
        x.append(a_true + b_true * x[-1] + rng.normal(0.0, 0.01))
    p = ou_mle(tuple(x), dt=dt_)
    assert p.theta == pytest.approx(-math.log(b_true) / dt_, rel=0.05)
    assert p.m == pytest.approx(a_true / (1.0 - b_true), abs=0.01)


# ==================================================================================================
# (1b) KNOWN optimality: solved b* satisfies the Leung-Li smooth-fit condition F(b*)=(b*-c)F'(b*)
# ==================================================================================================
def test_recovery_boundary_satisfies_leung_li_optimality_condition():
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    r, c = 0.05, 0.0
    b = leung_li_boundary(p, r=r, c=c)
    F = fg_F(p, r, b)
    Fp = fg_F_prime(p, r, b)
    # The paper's optimality condition (value-matching + smooth-pasting): F(b*) = (b*-c) F'(b*).
    assert F == pytest.approx((b - c) * Fp, rel=1e-4, abs=1e-6)
    # And the take-profit sits ABOVE the mean (bank a reverting winner as it overshoots up).
    assert b > p.m


def test_recovery_boundary_with_sale_cost_shifts_up():
    # A positive net sale cost c raises the liquidation bar (demand a higher level to clear cost).
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    b0 = leung_li_boundary(p, r=0.05, c=0.0)
    bc = leung_li_boundary(p, r=0.05, c=0.03)
    assert bc > b0
    # Still an exact optimality root at the shifted boundary.
    assert fg_F(p, 0.05, bc) == pytest.approx(
        (bc - 0.03) * fg_F_prime(p, 0.05, bc), rel=1e-4, abs=1e-6)


def test_recovery_F_prime_matches_finite_difference():
    # F' from differentiating-under-the-integral must match a central finite difference of F.
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    r, x, hstep = 0.05, 0.12, 1e-5
    fd = (fg_F(p, r, x + hstep) - fg_F(p, r, x - hstep)) / (2.0 * hstep)
    assert fg_F_prime(p, r, x) == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_recovery_fit_end_to_end_freezes_boundary():
    # fit() on a known OU path recovers params AND freezes a b* that clears the optimality cond.
    path = _simulate_ou(0.30, 0.10, 0.05, 1.0, n=40000, x0=0.10, seed=11)
    model = OUExit(r=0.05, c=0.0, dt=1.0).fit(_train(path))
    assert model.params is not None and model.b_star is not None
    assert model.params.theta == pytest.approx(0.30, rel=0.20)
    F = fg_F(model.params, model.r, model.b_star)
    Fp = fg_F_prime(model.params, model.r, model.b_star)
    assert F == pytest.approx((model.b_star - model.c) * Fp, rel=1e-4, abs=1e-6)


# ==================================================================================================
# (1d) REACHABILITY + DYNAMIC HOLD: the boundary is HIT along a controlled OU path and exit_now FIRES
# ==================================================================================================
# Defect-#3 fix: the default fit (theta=.30, m=.10, sigma=.05, r=.05) solves b*=.44, ~5.3 stationary
# std above m, so a return reverting around m essentially NEVER crosses it -- an economically inert /
# unreachable boundary the incumbent's fixed horizon dominates, yet every prior test only checked
# b*>m and the smooth-fit identity, never that the exit actually FIRES. The distance of b* above m
# (in stationary-std units) is governed by r/theta: a smaller r/theta puts b* within ~1 std of m, a
# REACHABLE take-profit. This test pins the DYNAMIC-HOLD behavior on such a reachable config: the
# boundary is crossed by a controlled OU path a non-trivial fraction of the time, exit_now FIRES on
# those crossings and HOLDS below -- so the model is not degenerate-vs-the-incumbent (never-fires).
def test_reachability_boundary_is_hit_and_exit_fires_along_an_ou_path():
    theta, m, sigma, r = 0.05, 0.0, 0.20, 0.05  # r/theta = 1 -> b* ~= 0.84 stationary std above m
    p = OUParams(theta=theta, m=m, sigma=sigma, dt=1.0)
    b = leung_li_boundary(p, r=r, c=0.0)
    std = sigma / math.sqrt(2.0 * theta)
    # REACHABILITY: the boundary sits within a few stationary std of m (a level a reverting winner
    # actually overshoots to), NOT the ~5-std unreachable wall the default config produces.
    assert m < b < m + 3.0 * std
    # Simulate the controlled OU and confirm the path crosses b* a non-trivial fraction of the time.
    path = np.asarray(_simulate_ou(theta, m, sigma, dt=1.0, n=4000, x0=m, seed=99))
    frac_above = float((path >= b).mean())
    assert frac_above > 0.02, f"boundary effectively unreachable (frac above b*={frac_above:.4f})"
    # DYNAMIC HOLD: drive exit_now with the running return = the path value; it must FIRE on crossings
    # and HOLD strictly below -- the boundary-crossing hold/exit semantics the model claims to add.
    model = OUExit(r=r, c=0.0, dt=1.0)
    model.params, model.b_star = p, b
    fired = 0
    for t in range(1, len(path)):
        cur = ScanCursor(series=tuple(float(v) for v in path[: t + 1]), t=t, cadence=Cadence.DAILY)
        decision = model.exit_now(cur, held_days=t, held_return=float(path[t]))
        assert decision is (float(path[t]) >= b)  # exact boundary-crossing semantics, every bar
        fired += int(decision)
    assert fired > 0  # the dynamic exit ACTUALLY triggers on this reachable config (not inert)


# ==================================================================================================
# (1c) ERROR-CONTROL: the F/G quadrature is OVERFLOW-SAFE; it NEVER fabricates a boundary (defect #1)
# ==================================================================================================
# The F/G integrand `u^{power} exp(sign*z*u - u^2/2)` peaks near u~=z=c0(x-m) with log-height ~z^2/2;
# a naive exp(log_g) overflows float64 (> ~709) for a moderate discount r (power=r/theta-1 grows) or a
# larger state x, silently returning the bracket-growth CAP as a fabricated b* (the reported defect).
# The peak-centered (log-sum-exp) core must (a) emit NO overflow warning and (b) either return a
# GENUINE smooth-fit root or RAISE -- never fabricate the cap artifact on a valid in-range (r, c).
def test_error_control_fg_quadrature_does_not_overflow_and_stays_finite():
    # F/G value & derivative at the OLD overflow trigger (r=1.0 on the default fit) must be FINITE and
    # emit no RuntimeWarning -- the log-sum-exp core keeps the exponentials bounded. (F itself may be
    # astronomically large but finite; the point is: no silent overflow -> nan/inf-driven fabrication.)
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # any overflow warning becomes a failure
        F = fg_F(p, r=1.0, x=0.30)
        Fp = fg_F_prime(p, r=1.0, x=0.30)
    assert math.isfinite(F) and math.isfinite(Fp)


def test_error_control_boundary_never_fabricated_on_wide_r_band():
    # ACROSS a wide discount band the OLD code overflowed and returned the fabricated bracket cap
    # (b*=3.1999999999) on r=1.0/5.0. Now each r must EITHER return a genuine smooth-fit root (clears
    # the optimality identity F(b*)=(b*-c)F'(b*) AND is not the cap artifact) OR raise a clear
    # ValueError -- and MUST NOT emit an overflow warning. No fabricated frozen boundary is admitted.
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    for r in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                b = leung_li_boundary(p, r=r, c=0.0)
            except ValueError:
                continue  # refused (immediate-liquidation / unbracketable band) -- not fabricated
        # If it returned, it is a GENUINE root: clears the smooth-fit identity and is NOT the old cap.
        assert abs(b - 3.1999999999) > 1e-6, f"r={r} returned the fabricated bracket-cap artifact"
        F, Fp = fg_F(p, r, b), fg_F_prime(p, r, b)
        assert F == pytest.approx((b - 0.0) * Fp, rel=1e-4, abs=1e-6)


def test_error_control_out_of_band_discount_is_refused_not_faked():
    # A HIGH discount relative to theta (r/theta large) makes immediate liquidation optimal: there is
    # NO take-profit boundary above the mean (h(m+) <= 0). The OLD code overflowed and fabricated the
    # cap; the fixed solver RAISES ValueError (a valid stationary OU, an out-of-band r -- refused).
    p = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            leung_li_boundary(p, r=1.0, c=0.0)


def test_error_control_boundary_scale_invariant_no_overflow():
    # A larger state SCALE (m and sigma scaled up together) drives z=c0(x-m) and the log-peak up; the
    # peak-centered core stays overflow-free and returns a consistently-scaled genuine root (the OLD
    # code overflowed here too). r/theta held in the bracketable band so a root exists.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p = OUParams(theta=0.05, m=0.0, sigma=1.0, dt=1.0)  # sd = 3.16, big log-peak at the root
        b = leung_li_boundary(p, r=0.05, c=0.0)
    assert math.isfinite(b) and b > p.m
    F, Fp = fg_F(p, 0.05, b), fg_F_prime(p, 0.05, b)
    assert F == pytest.approx(b * Fp, rel=1e-4, abs=1e-6)


# ==================================================================================================
# Stationarity check -- a non-stationary / degenerate train fold is REFUSED, never silently fit
# ==================================================================================================
def test_stationarity_random_walk_is_refused():
    # A pure random walk (b ~ 1, theta ~ 0) is NOT a stationary OU -> refuse.
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0.0, 1.0, size=5000))
    with pytest.raises(OUStationarityError):
        ou_mle(tuple(float(v) for v in rw))


def test_stationarity_constant_series_is_refused():
    with pytest.raises(OUStationarityError):
        ou_mle((0.5,) * 100)


def test_stationarity_too_few_samples_is_refused():
    with pytest.raises(OUStationarityError):
        ou_mle((0.1, 0.2))  # < 3 samples -> < 2 transition pairs


def test_ouparams_rejects_nonstationary_theta():
    with pytest.raises(OUStationarityError):
        OUParams(theta=0.0, m=0.1, sigma=0.05)   # theta <= 0 is non-stationary
    with pytest.raises(ValueError):
        OUParams(theta=0.3, m=0.1, sigma=0.0)    # sigma must be > 0


def test_fit_refuses_nonstationary_train_fold():
    rng = np.random.default_rng(2)
    rw = tuple(float(v) for v in np.cumsum(rng.normal(0.0, 1.0, size=4000)))
    with pytest.raises(OUStationarityError):
        OUExit().fit(_train(rw))


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING: exit_now at bar t must be F_{t-1}-measurable; a >= t read RAISES
# ==================================================================================================
def _fitted_model() -> OUExit:
    path = _simulate_ou(0.30, 0.10, 0.05, 1.0, n=20000, x0=0.10, seed=5)
    return OUExit(r=0.05, c=0.0, dt=1.0).fit(_train(path))


def test_pinning_exit_reads_only_strictly_past_bars():
    # The DAILY exit decision at cursor.t routes its causal read through the guard; a legal cursor
    # (t within range) decides WITHOUT raising -- it only ever reads bar t-1.
    model = _fitted_model()
    cur = ScanCursor(series=(0.0, 0.05, 0.09, 0.12, 0.20), t=4, cadence=Cadence.DAILY)
    # held_return above b* -> close; below -> hold. Decision is F_{t-1}-measurable (no raise).
    assert model.exit_now(cur, held_days=3, held_return=model.b_star + 0.01) is True
    assert model.exit_now(cur, held_days=3, held_return=model.b_star - 0.01) is False


def test_pinning_leaky_wiring_breaks_via_the_same_guard_the_model_uses():
    # PINNING: a candidate wiring that reads bar t (or later) at a DAILY decision is a LEAK the
    # ScanCursor guard `exit_now` routes through must catch. `exit_now` reads cursor.read(t-1) (ok);
    # a mutated/leaky exit that instead read bar t would call cursor.read(t) -> PurityError. We
    # assert the guard the model uses breaks on the exact same-bar / future reads a leaky exit does.
    cur = ScanCursor(series=(0.10, 0.20, 0.30), t=1, cadence=Cadence.DAILY)
    cur.read(0)  # strictly-past read is fine (this is what exit_now does: read(t-1))
    with pytest.raises(PurityError):
        cur.read(1)  # same bar t=1 -> leak (the exact read a look-ahead exit wiring would do)
    with pytest.raises(PurityError):
        cur.read(2)  # future bar -> leak


def test_pinning_exit_now_never_reads_the_execution_bar():
    # The model's own exit_now, at a legal cursor, must NOT raise -- it reads only bar t-1 (the
    # guarded F_{t-1} read exit_now performs), never the execution bar t. The DAILY guard behind it
    # then confirms reading bar t itself IS a leak, so exit_now stays strictly one-step-ahead.
    model = _fitted_model()
    cur = ScanCursor(series=(0.0, 0.05, 0.09, 0.20), t=3, cadence=Cadence.DAILY)
    # No raise: exit_now reads only strictly-past bars (F_{t-1}-measurable) to decide at t=3.
    assert model.exit_now(cur, held_days=2, held_return=model.b_star + 0.05) in (True, False)
    # And the guard behind it is the DAILY one: reading bar t=3 (the execution bar) is a leak.
    with pytest.raises(PurityError):
        cur.read(3)


def test_pinning_exit_before_fit_raises():
    m = OUExit()
    cur = ScanCursor(series=(0.0, 0.1, 0.2), t=2)
    with pytest.raises(RuntimeError):
        m.exit_now(cur, held_days=1, held_return=0.3)  # boundary not frozen yet


# --- (2b) DECISION-PATH pinning: the leak must break in the DECISION, not merely on an incidental read
# The earlier pinning tests exercise ScanCursor.read(t) (a base.ScanCursor property already covered by
# test_base) but NOT exit_now's DECISION path: the real decision is bool(held_return >= b*), so a leaky
# rewrite would trip only incidentally on read(). These two tests pin the DECISION itself: (i) a leaky
# exit whose OUTPUT depends on bar-t information routed through the guard MUST raise (the guard breaks a
# same-bar decision), and (ii) the real exit's decision is INVARIANT to future / garbage cursor content
# (it is a pure function of the F_{t-1}-measurable held_return), so it cannot be reading bar t.
class _LeakyOUExit(OUExit):
    """A deliberately look-ahead exit whose DECISION consumes bar-t (the same-bar leak to pin against).

    It mirrors ``OUExit.exit_now`` but folds the execution bar's value into the decision via the guarded
    ``cursor.read(cursor.t)`` -- exactly the one-bar look-ahead the purity line forbids. Because the read
    that FEEDS the decision is a same-bar read, the DAILY ``ScanCursor`` guard RAISES ``PurityError`` at
    the point the decision would consume bar t. This is the wiring a pinning test must FAIL, and it fails
    ON THE DECISION PATH (the read is load-bearing for the returned bool), not on a decorative side read.
    """

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:  # type: ignore[override]
        if self.b_star is None:
            raise RuntimeError("exit_now before fit")
        same_bar = cursor.read(cursor.t)  # LEAK: reads the execution bar to decide -> PurityError
        return bool(held_return + same_bar >= self.b_star)


def test_pinning_decision_that_reads_bar_t_raises_on_the_decision_path():
    # A leaky exit whose RETURNED DECISION depends on bar-t (read through the same guard the real model
    # uses) MUST raise PurityError -- the guard breaks a same-bar DECISION, not merely a stray read.
    leaky = _LeakyOUExit(r=0.05, c=0.0, dt=1.0)
    leaky.params = OUParams(theta=0.30, m=0.10, sigma=0.05, dt=1.0)
    leaky.b_star = 0.4
    cur = ScanCursor(series=(0.0, 0.1, 0.2, 0.3, 0.4), t=3, cadence=Cadence.DAILY)
    with pytest.raises(PurityError):
        leaky.exit_now(cur, held_days=2, held_return=0.35)  # decision consumes bar t=3 -> leak
    # The HONEST model, at the identical cursor, decides WITHOUT raising (reads only bar t-1).
    honest = _fitted_model()
    assert honest.exit_now(cur, held_days=2, held_return=honest.b_star + 0.01) is True


def test_pinning_decision_is_invariant_to_future_cursor_content():
    # The real exit_now's DECISION is a pure function of the F_{t-1}-measurable held_return: mutating
    # the cursor's FUTURE / same-bar content (bars >= t) must NOT change the returned bool. If the
    # decision depended on bar t (a leak), these two calls would differ. (Whether the supplied
    # held_return is itself F_{t-1} vs same-bar F_t is the HARNESS's measurability contract -- it
    # supplies held_return -- and is asserted where the harness wires trades, not in this unit.)
    model = _fitted_model()
    hr = model.b_star + 0.02
    # Same t, same strictly-past bars; only bars >= t differ wildly (garbage / future-laden tail).
    cur_a = ScanCursor(series=(0.0, 0.05, 0.09, 0.0, 0.0), t=3, cadence=Cadence.DAILY)
    cur_b = ScanCursor(series=(0.0, 0.05, 0.09, 999.0, -999.0), t=3, cadence=Cadence.DAILY)
    assert model.exit_now(cur_a, held_days=2, held_return=hr) == \
        model.exit_now(cur_b, held_days=2, held_return=hr)
    # And flipping held_return across b* DOES flip the decision -- the decision tracks the F_{t-1}
    # covariate, nothing on the execution bar.
    assert model.exit_now(cur_b, held_days=2, held_return=model.b_star + 0.02) is True
    assert model.exit_now(cur_b, held_days=2, held_return=model.b_star - 0.02) is False


# ==================================================================================================
# The -12% stop floor is RETAINED -- OUExit only banks winners earlier, never loosens the loss cap
# ==================================================================================================
def test_stop_floor_is_not_loosened_by_the_exit_model():
    # OUExit carries no loss-cap parameter at all; the retained floor is base.STOP_FLOOR and the
    # model's take-profit boundary lives ABOVE the mean (a winner level), never below the -12% cap.
    assert STOP_FLOOR == 0.12
    model = _fitted_model()
    assert model.b_star > model.params.m           # take-profit is a winner level, not a loss cap
    assert model.b_star > -STOP_FLOOR              # never a level looser than the -12% floor


def test_exit_is_monotone_in_held_return():
    # Given a frozen b*, exit fires iff held_return >= b* -- monotone, boundary-crossing semantics.
    model = _fitted_model()
    cur = ScanCursor(series=(0.0, 0.05, 0.10), t=2, cadence=Cadence.DAILY)
    below = model.exit_now(cur, held_days=2, held_return=model.b_star - 1e-6)
    at = model.exit_now(cur, held_days=2, held_return=model.b_star)
    above = model.exit_now(cur, held_days=2, held_return=model.b_star + 1e-6)
    assert below is False and at is True and above is True
