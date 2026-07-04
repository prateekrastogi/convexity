"""SYNTHETIC tests for MODELS.changepoint -- BOCPD (Adams-MacKay) entry. NO real lake (SYSTEM §C.8).

Two mandated tests (SYSTEM §C.2, task contract), plus protocol conformance / purity plumbing:
  (1) a KNOWN-generating-process RECOVERY test -- simulate a path whose truth we CONTROL (a piecewise-
      constant drift with an INJECTED change-point where the drift flips sign) and assert (a) the
      filtered run-length posterior COLLAPSES toward short run-lengths near the true change-point, and
      (b) the post-break filtered drift SIGN is recovered (SIGNAL §C.2(a) generating process);
  (2) a NO-LOOKAHEAD PINNING test -- a decision executing on bar t that reads bar t (or later) trips the
      causal guard and RAISES PurityError; the honest one-bar-offset form never reads bar t.

The BOCPD recursion is the code under test; the base-interface helpers (ScanCursor / require_causal /
Side / TrainFold) come from intraday.models.base. No sibling Phase-C model is imported (interface-driven
harness / off-ledger rule, SYSTEM §C.3). No lake, no metric, no adopt decision is touched anywhere here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from entry_exit.base import Cadence, EntryModel, PurityError, ScanCursor, Side, TrainFold
from entry_exit.changepoint import BOCPDEntry, run_length_posterior


def _train(vals: list[float], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


def _changepoint_path(seed: int, n_seg: int, mu_pre: float, mu_post: float, sigma: float) -> np.ndarray:
    """A controlled DGP: two constant-drift segments of length n_seg with a change-point at index n_seg.

    Gaussian innovations N(0, sigma^2) around a piecewise-constant drift (mu_pre then mu_post). This is
    EXACTLY the SIGNAL §C.2(a) generating process the BOCPD model is the correct model for; the change-
    point truth (index n_seg) and the post-break drift sign (sign of mu_post) are KNOWN by construction.
    """
    rng = np.random.default_rng(seed)
    pre = mu_pre + sigma * rng.standard_normal(n_seg)
    post = mu_post + sigma * rng.standard_normal(n_seg)
    return np.concatenate([pre, post])


# ==================================================================================================
# Protocol conformance (runtime_checkable structural typing) + validation
# ==================================================================================================
def test_bocpd_conforms_to_entrymodel():
    m = BOCPDEntry()
    assert isinstance(m, EntryModel)                 # fit + signal_at + k -> structural conformance
    assert m.fit(_train([0.01, -0.01, 0.02])) is m   # fit returns the fitted model (self)
    assert isinstance(m.k, int) and m.k >= 0         # parsimony count present (SYSTEM §C.4)


def test_bocpd_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        BOCPDEntry(hazard=0.0)          # hazard must be a probability in (0, 1)
    with pytest.raises(ValueError):
        BOCPDEntry(hazard=1.0)
    with pytest.raises(ValueError):
        BOCPDEntry(p_break=0.0)         # a posterior-mass threshold in (0, 1]
    with pytest.raises(ValueError):
        BOCPDEntry(beta0=0.0)           # NIG scale must be positive
    with pytest.raises(ValueError):
        BOCPDEntry(r_break=-1)


def test_signal_before_fit_raises():
    with pytest.raises(RuntimeError):
        BOCPDEntry().signal_at(ScanCursor(series=(0.01, 0.02, 0.03), t=3))


# ==================================================================================================
# (1) KNOWN-generating-process RECOVERY -- the posterior collapses at a known change-point AND the
#     post-break filtered drift sign is recovered (SIGNAL §C.2(a))
# ==================================================================================================
def test_recovery_posterior_collapses_near_true_changepoint():
    # Controlled truth: strong up-drift for 40 bars, then a flip to strong down-drift (change-point at 40).
    n_seg = 40
    x = _changepoint_path(seed=7, n_seg=n_seg, mu_pre=+0.05, mu_post=-0.05, sigma=0.01)

    # Filtered posterior on all data up to JUST BEFORE the change-point: the run has grown ~n_seg long,
    # so the MAP run-length should be LARGE (no break yet) -- the pre-break control.
    post_before, _ = run_length_posterior(
        x[:n_seg], hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    assert int(np.argmax(post_before)) > 10  # a long, uninterrupted run before the break

    # Filtered posterior a FEW bars AFTER the change-point: the mass must COLLAPSE toward short run-
    # lengths (the segment reset), i.e. the MAP run-length is small and short-run mass is substantial.
    lag = 3
    post_after, drift_after = run_length_posterior(
        x[: n_seg + lag], hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    map_run = int(np.argmax(post_after))
    assert map_run <= lag + 2                     # detected the reset near the true change-point index
    assert map_run < int(np.argmax(post_before))  # strictly shorter than the pre-break run
    short_mass = float(post_after[: lag + 3].sum())
    assert short_mass > 0.5                        # posterior mass collapsed onto the new (short) segment
    # And the post-break FILTERED drift recovers the true post-change sign (mu_post < 0 -> negative).
    assert drift_after < 0.0


def test_recovery_post_break_drift_sign_both_directions():
    # Symmetric control: down->up flip must recover a POSITIVE post-break filtered drift (mirror of above).
    n_seg = 40
    x = _changepoint_path(seed=11, n_seg=n_seg, mu_pre=-0.05, mu_post=+0.05, sigma=0.01)
    _, drift_after = run_length_posterior(
        x[: n_seg + 3], hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    assert drift_after > 0.0


def test_recovery_entry_fires_in_post_break_direction():
    # End-to-end through the model: fit on a train fold, then a DOWN-flip change-point path must make the
    # entry EXECUTED just after the break fire SHORT (the sign of the recovered post-break drift). The
    # firing window is the bars immediately after a detected break, when the run-length posterior mass is
    # still collapsed onto short run-lengths (P(r<=r_break) high) -- BOCPD's run-length grows one bar per
    # step, so the short-run mass clears p_break only right after the reset (the honest detection window).
    n_seg = 40
    x = _changepoint_path(seed=7, n_seg=n_seg, mu_pre=+0.05, mu_post=-0.05, sigma=0.01)
    train = _train(list(x[:n_seg]), warmup=2)  # fit the NIG scale on the (pre-break) train fold
    m = BOCPDEntry(hazard=0.02, p_break=0.5, r_break=2, alpha0=2.0, beta0=1e-3).fit(train)
    # Execute two bars after the change-point: cursor.past() = series[:t] ends at x_{t-1} (F_{t-1}-
    # measurable), one datum into the new segment -> short-run mass is high, the break is detected.
    t = n_seg + 2
    side = m.signal_at(ScanCursor(series=tuple(x[:t + 1]), t=t, cadence=Cadence.DAILY))
    assert side is Side.SHORT

    # The mirror (up-flip) fires LONG under the same wiring -- cross-direction recovery.
    x_up = _changepoint_path(seed=11, n_seg=n_seg, mu_pre=-0.05, mu_post=+0.05, sigma=0.01)
    m_up = BOCPDEntry(hazard=0.02, p_break=0.5, r_break=2, alpha0=2.0, beta0=1e-3).fit(
        _train(list(x_up[:n_seg]), warmup=2))
    side_up = m_up.signal_at(ScanCursor(series=tuple(x_up[:t + 1]), t=t, cadence=Cadence.DAILY))
    assert side_up is Side.LONG


def test_stable_drift_no_false_break_is_flat():
    # A single quiescent segment (no change-point): the run grows long, short-run mass stays below
    # p_break, so the entry stays FLAT -- no spurious signal on an undisturbed drift.
    rng = np.random.default_rng(3)
    x = 0.02 + 0.01 * rng.standard_normal(60)
    m = BOCPDEntry(hazard=0.02, p_break=0.5, r_break=1, alpha0=2.0, beta0=1e-3).fit(
        _train(list(x[:30]), warmup=2))
    side = m.signal_at(ScanCursor(series=tuple(x), t=len(x) - 1, cadence=Cadence.DAILY))
    assert side is Side.FLAT


def test_fit_freezes_and_recovers_train_drift_scale():
    # Train-fit, frozen: fit refines mu0/beta0 from the TRAIN fold and freezes them (SIGNAL §C.1).
    m = BOCPDEntry(alpha0=2.0)
    train = _train([0.10, 0.10, 0.10, 0.10], warmup=0)
    m.fit(train)
    assert m.mu0 == pytest.approx(0.10)  # prior drift recovered to the known train mean
    assert m.beta0 > 0.0                 # innovation-scale prior positive and frozen


def test_fit_recovers_beta0_from_train_variance_at_any_alpha0():
    # KNOWN-generating-process scale recovery: with a controlled train fold of KNOWN variance, fit must
    # anchor the NIG scale beta0 to that variance by inverting the correct NIG moment of sigma^2 -- for
    # BOTH a-priori alpha0 branches (closing the OBS-5 calibration gap: beta0 must track the train var
    # even at the constructor default alpha0 == 1.0, where E[sigma^2] is undefined).
    rng = np.random.default_rng(19)
    body = list(0.0 + 0.03 * rng.standard_normal(400))  # controlled dispersion, drift ~ 0
    var = float(np.asarray(body[0:]).var(ddof=0))       # the KNOWN train variance (warmup=0 here)

    # alpha0 > 1: E[sigma^2] = beta0 / (alpha0 - 1) is defined -> beta0 == var * (alpha0 - 1) exactly.
    m2 = BOCPDEntry(alpha0=2.0).fit(_train(body, warmup=0))
    assert m2.beta0 == pytest.approx(var * (2.0 - 1.0))
    m3 = BOCPDEntry(alpha0=3.0).fit(_train(body, warmup=0))
    assert m3.beta0 == pytest.approx(var * (3.0 - 1.0))

    # alpha0 == 1.0 (default): mean undefined -> invert the always-defined MODE beta0/(alpha0+1),
    # so beta0 == var * (alpha0 + 1) and the scale is STILL tied to the train variance (not an ad-hoc
    # floor). Both fitted scales are strictly positive and monotone in the controlled variance.
    m1 = BOCPDEntry(alpha0=1.0).fit(_train(body, warmup=0))
    assert m1.beta0 == pytest.approx(var * (1.0 + 1.0))
    assert m1.beta0 > 0.0 and m2.beta0 > 0.0

    # A degenerate (constant) train fold has var == 0 -> the positive floor keeps beta0 well-defined.
    m_flat = BOCPDEntry(alpha0=1.0).fit(_train([0.05] * 8, warmup=0))
    assert m_flat.beta0 > 0.0


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING -- the entry at bar t is F_{t-1}-measurable; a same-bar read RAISES
# ==================================================================================================
def test_pinning_signal_never_reads_bar_t():
    # The honest one-bar-offset form reads ONLY series[:t] (up to x_{t-1}); it must NOT raise, and the
    # cursor's own guard proves a same-bar read WOULD be a leak (the pin the model relies on).
    x = _changepoint_path(seed=7, n_seg=40, mu_pre=0.05, mu_post=-0.05, sigma=0.01)
    t = 44
    cur = ScanCursor(series=tuple(x[:t + 1]), t=t, cadence=Cadence.DAILY)
    m = BOCPDEntry(alpha0=2.0, beta0=1e-3).fit(_train(list(x[:40]), warmup=2))
    m.signal_at(cur)                     # F_{t-1}-measurable: no raise
    cur.read(cur.t - 1)                  # the last legal (strictly-past) read the model performs
    with pytest.raises(PurityError):
        cur.read(cur.t)                  # a same-bar read (the leak the pin forbids) -> PurityError
    with pytest.raises(PurityError):
        cur.read(cur.t + 1)              # a future read -> PurityError


def test_pinning_leaky_recursion_over_series_including_bar_t_would_change_the_read():
    # Demonstrate the leak the offset closes: conditioning on x_{1:t} (INCLUDING bar t) is a DIFFERENT,
    # forbidden posterior than the honest x_{1:t-1}. A wiring that fed series[:t+1] would read bar t --
    # exactly the P(r_t|F_t) same-bar look-ahead the pre-registration bars (docs/EDGE #1).
    x = _changepoint_path(seed=7, n_seg=40, mu_pre=0.05, mu_post=-0.05, sigma=0.01)
    t = 44
    honest, _ = run_length_posterior(x[:t], hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    leaky, _ = run_length_posterior(x[:t + 1], hazard=0.02, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    # The leaky posterior is one bar longer and materially different -- the offset is not cosmetic.
    assert len(leaky) == len(honest) + 1
    assert not np.allclose(honest, leaky[: len(honest)])


def test_flat_when_still_warming_up():
    # Warm-up honesty: before the run-length prior has enough history the scan stays FLAT (no borrow).
    m = BOCPDEntry(warmup=5, alpha0=2.0).fit(_train([0.01, 0.02, 0.03, 0.02, 0.01], warmup=0))
    assert m.signal_at(ScanCursor(series=(0.01, 0.02, 0.03), t=2, cadence=Cadence.DAILY)) is Side.FLAT


def test_run_length_posterior_is_normalized_and_causal():
    # The filtered posterior is a proper distribution of length n+1 (index 0 = "change just occurred").
    x = _changepoint_path(seed=1, n_seg=10, mu_pre=0.03, mu_post=-0.03, sigma=0.01)
    post, _ = run_length_posterior(x, hazard=0.05, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1e-3)
    assert len(post) == len(x) + 1
    assert post.sum() == pytest.approx(1.0)
    assert (post >= 0.0).all()
    # Empty history -> degenerate prior on r=0 (causal base case, no future peek).
    post0, drift0 = run_length_posterior(np.array([]), hazard=0.05, mu0=0.5, kappa0=1.0, alpha0=2.0, beta0=1.0)
    assert post0.tolist() == [1.0] and drift0 == 0.5
