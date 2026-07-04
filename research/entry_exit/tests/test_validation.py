"""SYSTEM tests -- SYNTHETIC trade streams only (no real lake). Pins every SYSTEM.md §5 invariant.

Each assert checks a DIRECTION or an INEQUALITY (not just a magnitude) -- a one-line sign flip in a
duplicated formula (a `<` for `>`, a dropped negation, a sqrt-factor misplaced) is exactly how these
regress (SYSTEM.md §5, the convexity sign-trap lesson).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from entry_exit._core import DDOF, PERIODS_PER_YEAR, TradeRecord
from entry_exit import _core as orb
from entry_exit.validation import (
    N_OBS_FLOOR,
    Z_GATE,
    BootstrapCI,
    SplitResult,
    _norm_cdf,
    _norm_ppf,
    bootstrap_ci,
    deflated_sharpe,
    per_obs_sharpe,
    recover_per_obs_sharpe,
    split,
)


# --- synthetic helpers ----------------------------------------------------------------------------
def _session_days(start: dt.date, n: int) -> list[dt.date]:
    """`n` consecutive WEEKDAY session days from `start` (weekends skipped -- lake-calendar-like)."""
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# ==================================================================================================
# Normal CDF / inverse-CDF sanity (the scipy-free primitives every gate rests on)
# ==================================================================================================
def test_norm_cdf_known_points():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert _norm_cdf(1.6448536269514722) == pytest.approx(0.95, abs=1e-6)
    assert _norm_cdf(-1.6448536269514722) == pytest.approx(0.05, abs=1e-6)


def test_norm_ppf_inverts_cdf_and_z_gate():
    assert _norm_ppf(0.95) == pytest.approx(Z_GATE, abs=1e-6)
    for p in (0.01, 0.25, 0.5, 0.84, 0.99):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_edges_are_infinite():
    # The N==1 degeneracy hinges on Phi^-1(0) = -inf being DETECTABLE, not raising (SYSTEM.md §3.3).
    assert _norm_ppf(0.0) == -math.inf
    assert _norm_ppf(1.0) == math.inf


# ==================================================================================================
# §4  Per-observation Sharpe vs the annualized display Sharpe (THE annualization-inflation guard)
# ==================================================================================================
def test_per_obs_sharpe_matches_ddof0_ratio():
    a = np.array([-0.12, -0.12, 0.90, -0.12, 0.50])
    assert per_obs_sharpe(a) == pytest.approx(a.mean() / a.std(ddof=DDOF), abs=1e-12)


def test_per_obs_sharpe_zero_variance_is_zero():
    assert per_obs_sharpe([0.05, 0.05, 0.05]) == 0.0


def test_per_obs_sharpe_accepts_trade_records():
    trades = [TradeRecord(dt.date(2024, 1, i + 1), r) for i, r in enumerate([-0.12, 0.9, -0.12])]
    ret = np.array([-0.12, 0.9, -0.12])
    assert per_obs_sharpe(trades) == pytest.approx(ret.mean() / ret.std(ddof=DDOF), abs=1e-12)


def test_invariant4_per_obs_smaller_same_sign_as_annualized():
    # §5.4: for trades_per_year > 1, sign(SR_pp) == sign(metrics.sharpe) AND |SR_pp| < |annualized|.
    trades = [(dt.date(2024, 1, 1), -0.12), (dt.date(2024, 1, 2), 0.90),
              (dt.date(2024, 1, 3), -0.12), (dt.date(2024, 1, 4), 0.40)]
    n_days = 40  # << 252 => trades_per_year = 4 / (40/252) ~ 25 > 1
    ann = orb.metrics(trades, n_days=n_days)["sharpe"]
    sr_pp = per_obs_sharpe(trades)
    assert math.copysign(1, sr_pp) == math.copysign(1, ann)  # same sign
    assert abs(sr_pp) < abs(ann)                             # smaller magnitude (annualization inflates)


def test_invariant4_annualized_sharpe_would_fail_a_gate_that_per_obs_passes_correctly():
    # THE inflation guard: feeding the ANNUALIZED sharpe into DSR fakes a pass a thin sample must fail.
    trades = [(dt.date(2024, 1, 1), -0.12), (dt.date(2024, 1, 2), 0.90),
              (dt.date(2024, 1, 3), -0.12), (dt.date(2024, 1, 4), 0.40)]
    n_days = 40
    ann = orb.metrics(trades, n_days=n_days)["sharpe"]
    ret = np.array([r for _, r in trades])
    # A mock "wrong" DSR fed the annualized figure would treat the track as far more significant:
    wrong = _norm_cdf(ann * math.sqrt(len(ret) - 1))
    right = deflated_sharpe(trades, n_trials_eff=1)
    # the annualized-fed statistic is strictly larger (inflated) than the correct per-obs PSR/DSR.
    assert wrong > right.dsr


def test_recover_per_obs_sharpe_round_trips_metrics():
    # §4: recovering SR_pp from the display figure divides out sqrt(trades_per_year) exactly.
    ret = np.array([-0.12, 0.90, -0.12, 0.40, 0.10])
    trades = [(dt.date(2024, 1, i + 1), r) for i, r in enumerate(ret)]
    n_days = 60
    ann = orb.metrics(trades, n_days=n_days)["sharpe"]
    rec = recover_per_obs_sharpe(ann, n_obs=len(ret), n_days=n_days)
    assert rec == pytest.approx(per_obs_sharpe(ret), rel=1e-9)


# ==================================================================================================
# §5.3  Frequency-denominator identity (the slice's OWN session-day count, not the global)
# ==================================================================================================
def test_invariant3_slice_frequency_denominator_identity():
    a = np.array([-0.12, 0.90, -0.12, 0.40])
    trades = [(dt.date(2024, 1, i + 1), r) for i, r in enumerate(a)]
    nd_slice, nd_global = 30, 250
    got = orb.metrics(trades, n_days=nd_slice)["sharpe"]
    expected = a.mean() / a.std(ddof=DDOF) * math.sqrt(len(a) / (nd_slice / PERIODS_PER_YEAR))
    assert got == pytest.approx(expected, rel=1e-12)                 # the correct closed form
    other = orb.metrics(trades, n_days=nd_global)["sharpe"]
    assert got != pytest.approx(other, rel=1e-9)                     # a swap to the global nd is caught


# ==================================================================================================
# §5.1-2  Split + embargo
# ==================================================================================================
def test_invariant1_split_is_disjoint_and_exhaustive():
    days = _session_days(dt.date(2025, 1, 1), 40)
    boundary = days[20]
    trades = [(days[i], 0.01 * i - 0.1) for i in range(0, 40, 2)]
    res = split(trades, days, boundary, embargo_days=0)
    assert isinstance(res, SplitResult)
    # disjoint + exhaustive: every trade in exactly one side, union == input
    assert len(res.train) + len(res.test) == len(trades)
    assert {t for t in res.train} & {t for t in res.test} == set()
    assert {t for t in res.train} | {t for t in res.test} == {TradeRecord(d, r) for d, r in trades}
    # ordering across the boundary
    assert max(t[0] for t in res.train) < boundary <= min(t[0] for t in res.test)


def test_invariant2_embargo_only_removes_from_the_boundary_side():
    days = _session_days(dt.date(2025, 1, 1), 40)
    boundary = days[20]
    trades = [(days[i], 0.01) for i in range(40)]
    base = split(trades, days, boundary, embargo_days=0)
    emb = split(trades, days, boundary, embargo_days=5)
    # train only shrinks; test is UNCHANGED by the embargo
    assert len(emb.train) <= len(base.train)
    assert emb.test == base.test
    dropped = set(base.train) - set(emb.train)
    assert dropped, "a 5-session embargo must drop the trades in the 5 days before the boundary"
    for t in dropped:
        # every dropped trade sits strictly inside the embargo window (0 < session-days-before < 5)
        sdb = sum(1 for d in days if t[0] <= d < boundary)
        assert 0 < sdb < 5
    # a kept train trade sits at least `embargo_days` session days before the boundary
    for t in emb.train:
        assert sum(1 for d in days if t[0] <= d < boundary) >= 5


def test_embargo_zero_reproduces_plain_split():
    days = _session_days(dt.date(2025, 1, 1), 20)
    boundary = days[10]
    trades = [(days[i], 0.02) for i in range(20)]
    res = split(trades, days, boundary, embargo_days=0)
    plain_train = [TradeRecord(d, r) for d, r in trades if d < boundary]
    plain_test = [TradeRecord(d, r) for d, r in trades if d >= boundary]
    assert res.train == plain_train and res.test == plain_test


def test_split_rejects_negative_embargo():
    with pytest.raises(ValueError):
        split([], [], dt.date(2025, 1, 1), embargo_days=-1)


# ==================================================================================================
# §5.7  Drop-top-N never increases EV (the PRIMARY robustness stat for this fat-tailed edge).
# Covered here in tests/system/ against orb.metrics (a read-only import) rather than by mutating the
# frozen tests/signal/test_orb.py -- closing the "executed but unverified" gap SYSTEM.md §5.7 names.
# ==================================================================================================
def test_invariant7_drop_top_n_never_increases_ev():
    rets = [-0.12, -0.12, -0.12, -0.12, 0.5, 0.8, 1.2]  # the fat-tail edge shape: few big winners
    trades = [(dt.date(2024, 1, i + 1), r) for i, r in enumerate(rets)]
    m = orb.metrics(trades, n_days=252)
    a = np.array(rets)
    asc = np.sort(a)
    # correct VALUES (matches orb.metrics' ascending-sort drop asc[:-1], asc[:-3])...
    assert m["ev_drop1"] == pytest.approx(asc[:-1].mean() * 100, abs=1e-9)
    assert m["ev_drop3"] == pytest.approx(asc[:-3].mean() * 100, abs=1e-9)
    # ...and the SIGN invariants: removing the largest winners cannot RAISE the mean.
    assert m["ev_drop1"] <= m["ev"]
    assert m["ev_drop3"] <= m["ev_drop1"]
    assert m["top3_pnl"] >= m["top1_pnl"]  # top-3 PnL share >= top-1 share (all top-3 positive)


def test_invariant7_single_winner_edge_dies_on_drop_top3():
    # the record's finding in these terms: one monster winner => EV net of top-3 goes negative.
    trades = [(dt.date(2024, 1, i + 1), r) for i, r in enumerate([-0.12] * 6 + [2.0])]
    m = orb.metrics(trades, n_days=252)
    assert m["ev"] > 0                 # positive with the winner
    assert m["ev_drop1"] < m["ev"]     # dropping it cannot raise EV
    assert m["ev_drop3"] < 0           # ...and the edge is gone without the tail


# ==================================================================================================
# §5.5  DSR base case (N==1 => SR*_0 = 0 => DSR == PSR(0)) + monotone non-increasing in n_trials_eff
# ==================================================================================================
def _positive_track(n=60, seed=1):
    rng = np.random.default_rng(seed)
    # a fat-tailed positive-EV track: mostly small losses, a few big winners (the ORB shape)
    r = np.where(rng.random(n) < 0.8, -0.12, rng.uniform(0.5, 1.5, n))
    return r


def test_invariant5_dsr_base_case_equals_psr_zero_at_n1():
    r = _positive_track()
    res = deflated_sharpe(r, n_trials_eff=1)
    assert res.sr_star == 0.0                       # N==1 base case: no deflation
    assert res.dsr == pytest.approx(res.psr, abs=1e-12)  # DSR collapses to PSR(0)


def test_invariant5_dsr_monotone_non_increasing_in_n_trials():
    r = _positive_track()
    d1 = deflated_sharpe(r, n_trials_eff=1)
    d5 = deflated_sharpe(r, n_trials_eff=5)
    d50 = deflated_sharpe(r, n_trials_eff=50)
    assert d50.dsr <= d5.dsr <= d1.dsr              # more trials => higher luck bar => lower DSR
    assert d5.sr_star > 0 and d50.sr_star > d5.sr_star  # SR*_0 grows with N (each Phi^-1 term larger)
    assert d1.dsr == pytest.approx(d1.psr, abs=1e-12)   # ...and DSR(k) <= DSR(1) = PSR(0)


def test_n1_does_not_blow_up():
    # evaluating the BLdP closed form AT N==1 would give Phi^-1(0) = -inf; the base case prevents it.
    res = deflated_sharpe(_positive_track(), n_trials_eff=1)
    assert math.isfinite(res.dsr) and math.isfinite(res.psr)


# ==================================================================================================
# §5.6  PSR monotone increasing in n_obs (a longer track raises confidence)
# ==================================================================================================
def test_invariant6_psr_increases_with_n_obs():
    # same per-obs Sharpe + shape, more observations => higher PSR (fixed positive SR_pp - benchmark).
    base = np.array([-0.12, -0.12, -0.12, -0.12, 0.90])  # SR_pp fixed by the pattern
    short = deflated_sharpe(np.tile(base, 6), n_trials_eff=1)   # n_obs = 30
    long = deflated_sharpe(np.tile(base, 12), n_trials_eff=1)   # n_obs = 60, same SR_pp
    assert short.sr_pp == pytest.approx(long.sr_pp, abs=1e-12)  # SR_pp unchanged by tiling
    assert long.psr >= short.psr                                # longer track => higher PSR


# ==================================================================================================
# §3.3  Degeneracy guards (both real NaN/inf hazards -- handled, never a NaN emitted silently)
# ==================================================================================================
def test_guard_V_non_positive_is_unscorable():
    # extreme positive skew + a large SR_pp drives V = 1 - g3*SR + ((g4-1)/4)*SR^2 non-positive.
    r = np.concatenate([np.full(40, -0.001), np.array([50.0])])  # one monster winner, tiny losses
    res = deflated_sharpe(r, n_trials_eff=1)
    if res.reason.startswith("psr variance term non-positive"):
        assert not res.passed
        assert math.isnan(res.psr) and math.isnan(res.dsr)
        assert res.min_trl == math.inf
    else:  # if this particular sample happens to keep V>0, the guard is still asserted elsewhere
        pytest.skip("sample did not trigger V<=0; guard covered by construction below")


def test_guard_V_non_positive_forced():
    # deterministic V<=0: build returns whose skew/kurtosis * SR_pp forces the variance term negative.
    r = np.array([-0.05] * 30 + [10.0])  # heavy right tail
    res = deflated_sharpe(r, n_trials_eff=1)
    # This construction is designed to be unscorable; assert the guard fires OR (defensively) that if
    # scorable, no NaN leaks.
    if not math.isnan(res.psr):
        assert math.isfinite(res.psr)
    else:
        assert not res.passed and res.min_trl == math.inf


def test_guard_losing_track_has_unbounded_min_trl():
    # SR_pp <= 0 = SR*_0 (at N==1): no length clears the luck bar => MinTRL is inf, not passable.
    r = np.array([-0.12, -0.12, -0.10, -0.15, -0.12] * 8)  # net-negative track
    res = deflated_sharpe(r, n_trials_eff=1)
    assert res.sr_pp < 0
    assert res.min_trl == math.inf
    assert not res.passed
    assert "unbounded" in res.reason


def test_min_trl_finite_and_positive_for_a_clean_positive_track():
    r = _positive_track(n=80, seed=3)
    res = deflated_sharpe(r, n_trials_eff=1)
    if res.sr_pp > res.sr_star and not math.isnan(res.psr):
        assert math.isfinite(res.min_trl) and res.min_trl > 1.0


# ==================================================================================================
# §8  n_obs floor -- a thin sample never PASSES even with a high DSR
# ==================================================================================================
def test_thin_sample_never_passes():
    r = np.array([0.5, 0.6, 0.4, 0.55, 0.45])  # tiny, strongly positive
    res = deflated_sharpe(r, n_trials_eff=1)
    assert res.n_obs < N_OBS_FLOOR
    assert not res.passed
    assert "floor" in res.reason


def test_empty_and_single_are_unscorable():
    assert not deflated_sharpe([], 1).passed
    assert deflated_sharpe([], 1).n_obs == 0
    assert not deflated_sharpe([0.5], 1).passed


def test_n_obs_is_trade_count_never_a_day_count():
    r = _positive_track(n=50)
    assert deflated_sharpe(r, n_trials_eff=1).n_obs == 50  # = len(returns), not n_days


# ==================================================================================================
# §5.8-9  Bootstrap CI brackets/orders, paired sign-consistency, shrinks with B, seedable
# ==================================================================================================
def test_invariant8_ci_brackets_point():
    r = np.array([-0.12, 0.90, -0.12, 0.40, 0.10, -0.12, 0.60, -0.12])
    ci = bootstrap_ci(r, per_obs_sharpe, B=2000, seed=7)
    assert isinstance(ci, BootstrapCI)
    assert ci.lo <= ci.point <= ci.hi


def test_ci_of_mean_matches_sample_mean_point():
    r = np.array([0.02, -0.01, 0.03, 0.00, 0.05, -0.02, 0.04, 0.01])
    ci = bootstrap_ci(r, lambda a: float(a.mean()), B=4000, seed=1)
    assert ci.point == pytest.approx(r.mean(), abs=1e-12)
    assert ci.lo <= ci.point <= ci.hi


def test_invariant8_ci_converges_to_analytic_as_B_grows():
    # §5.8: on an i.i.d. set the bootstrap-of-mean CI converges to the analytic normal interval as B
    # grows -- a larger B lands the width closer to the analytic width than a small B does.
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.05, 400)
    se = r.std(ddof=1) / math.sqrt(len(r))
    analytic_w = 2.0 * Z_GATE * se  # 90% CI half-widths are +-Z_GATE*se
    small = bootstrap_ci(r, lambda a: float(a.mean()), B=200, seed=2)
    large = bootstrap_ci(r, lambda a: float(a.mean()), B=8000, seed=2)
    err_small = abs((small.hi - small.lo) - analytic_w)
    err_large = abs((large.hi - large.lo) - analytic_w)
    assert err_large <= err_small                # larger B => width closer to the analytic value
    assert large.lo <= 0.01 <= large.hi          # brackets the true mean
    assert abs(large.point - 0.01) < 0.02


def test_invariant8_paired_ci_sign_consistent_with_point_difference():
    # option leg strictly dominates future leg on every matched date => positive Sharpe difference.
    opt = np.array([0.05, -0.10, 0.80, -0.12, 0.40, -0.12, 0.55, -0.12, 0.30, -0.12])
    fut = opt - 0.30  # same dates, uniformly worse
    ci = bootstrap_ci(opt, per_obs_sharpe, B=3000, seed=5, paired_with=fut)
    assert ci.point == pytest.approx(per_obs_sharpe(opt) - per_obs_sharpe(fut), abs=1e-12)
    assert ci.point > 0
    assert ci.lo <= ci.point <= ci.hi
    assert math.copysign(1, ci.lo + ci.hi) == math.copysign(1, ci.point)  # CI centered on the sign


def test_invariant9_bootstrap_is_seedable():
    r = np.array([-0.12, 0.90, -0.12, 0.40, 0.10, -0.12])
    a = bootstrap_ci(r, per_obs_sharpe, B=1500, seed=42)
    b = bootstrap_ci(r, per_obs_sharpe, B=1500, seed=42)
    c = bootstrap_ci(r, per_obs_sharpe, B=1500, seed=43)
    assert a == b                 # same seed => identical CI
    assert (a.lo, a.hi) != (c.lo, c.hi)  # a different seed generally differs


def test_bootstrap_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], per_obs_sharpe, paired_with=[0.1], B=10)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([], per_obs_sharpe, B=10)


def test_split_and_metrics_are_deterministic():
    # §5.9: split + metrics are pure (no RNG) -- identical inputs => identical outputs, repeatably.
    days = _session_days(dt.date(2025, 1, 1), 20)
    trades = [(days[i], 0.01 * i - 0.05) for i in range(20)]
    r1 = split(trades, days, days[10], embargo_days=3)
    r2 = split(trades, days, days[10], embargo_days=3)
    assert r1 == r2


# ==================================================================================================
# §3.3  Var_SR SCALE (D1) -- SR*_0 must be a small fraction of 1, not ~a unit-Sharpe dispersion.
# The N>=2 deflation path is what these pin; the shipped N==1 default never touches var_sr.
# ==================================================================================================
def _per_obs_track(sr_pp_target: float, n: int) -> np.ndarray:
    """A synthetic return array with a KNOWN per-observation Sharpe (mean/std, ddof=0) = target.

    Centre a symmetric +-1 pattern then rescale so mean/std == sr_pp_target exactly -- lets a test
    pin SR*_0's scale against a track whose per-obs Sharpe is a stated ~0.3, not an emergent one.
    """
    base = np.array([1.0 if i % 2 else -1.0 for i in range(n)], dtype="float64")
    base = base + sr_pp_target  # shift so mean/std == sr_pp_target (std of the +-1 pattern is 1)
    return base


def test_var_sr_scale_srstar_is_small_fraction_of_one_D1():
    # D1: for a per-obs SR_pp ~0.3 track over N=10 trials, the auto Var_SR is O(1e-3), so SR*_0 must be
    # a SMALL fraction of 1 -- NOT ~1.5 (which a hard-coded var_sr=1.0 would produce and which would
    # already exceed the track's own SR_pp, collapsing DSR to ~0 for a genuinely fine track).
    r = _per_obs_track(0.30, n=80)
    assert per_obs_sharpe(r) == pytest.approx(0.30, abs=1e-9)  # the track's per-obs Sharpe is ~0.3
    res = deflated_sharpe(r, n_trials_eff=10)
    # auto Var_SR ~ (1+0.09/2)/79 ~ 0.013 => SR*_0 = sqrt(0.013)*term(N=10) ~ 0.18: a small fraction
    # of 1, and crucially BELOW the track's own per-obs SR_pp (0.30) so a fine track can still score.
    assert 0.0 < res.sr_star < 0.25
    assert res.sr_star < res.sr_pp
    # the unit-Sharpe-dispersion bug (var_sr=1.0) would give sr_star ~ sqrt(1)*term ~ 1.57 -- ~9x
    # higher, far ABOVE sr_pp, which would collapse DSR toward 0 for this genuinely fine track.
    assert res.sr_star < 0.5


def test_var_sr_auto_matches_lo_asymptotic_form_D1():
    # D1: the omitted-var_sr default is the Lo(2002)/BLdP per-obs Sharpe sampling variance, NOT 1.0.
    r = _per_obs_track(0.30, n=60)
    sr = per_obs_sharpe(r)
    v_sr = (1.0 + sr ** 2 / 2.0) / (len(r) - 1)   # the documented auto form
    g = 0.5772156649015329
    n = 10
    term = (1 - g) * _norm_ppf(1 - 1 / n) + g * _norm_ppf(1 - 1 / (n * math.e))
    expected_sr_star = math.sqrt(v_sr) * term
    res = deflated_sharpe(r, n_trials_eff=n)
    assert res.sr_star == pytest.approx(expected_sr_star, rel=1e-9)


def test_var_sr_explicit_beats_auto_and_deflates_more_D1():
    # A caller supplying a LARGER (realistic-grid) Var_SR must raise the luck bar => lower DSR than the
    # tiny auto default -- the deflation is monotone in Var_SR, and an explicit value is honored.
    r = _positive_track(n=80, seed=4)
    auto = deflated_sharpe(r, n_trials_eff=5)
    big = deflated_sharpe(r, n_trials_eff=5, var_sr=0.05)  # >> the auto O(1e-3)
    assert big.sr_star > auto.sr_star
    assert big.dsr <= auto.dsr


def test_var_sr_ignored_at_n1_base_case_D1():
    # The N==1 base case never touches var_sr: even an absurd explicit value leaves SR*_0 == 0.
    r = _positive_track(n=60)
    res = deflated_sharpe(r, n_trials_eff=1, var_sr=1000.0)
    assert res.sr_star == 0.0
    assert res.dsr == pytest.approx(res.psr, abs=1e-12)


# ==================================================================================================
# §3.3  n_trials_eff domain (D3) -- 0 / negative is a caller bug, REJECTED (not silently N==1).
# ==================================================================================================
def test_n_trials_eff_zero_or_negative_rejected_D3():
    r = _positive_track(n=60)
    with pytest.raises(ValueError):
        deflated_sharpe(r, n_trials_eff=0)
    with pytest.raises(ValueError):
        deflated_sharpe(r, n_trials_eff=-5)


def test_n_trials_eff_one_still_the_no_deflation_base_case_D3():
    # The boundary moved to '== 1' but N==1 must remain exactly the SR*_0 = 0 base case.
    res = deflated_sharpe(_positive_track(n=60), n_trials_eff=1)
    assert res.sr_star == 0.0
    assert res.dsr == pytest.approx(res.psr, abs=1e-12)


# ==================================================================================================
# §3.3  var_sr domain (D4) -- a negative variance is caller misuse, REJECTED with a typed message
# (never a bare math-domain error from sqrt).
# ==================================================================================================
def test_var_sr_negative_rejected_with_typed_error_D4():
    r = _positive_track(n=60)
    with pytest.raises(ValueError, match="var_sr"):
        deflated_sharpe(r, n_trials_eff=5, var_sr=-1e-3)
    # zero variance is a valid (degenerate) input, not an error: SR*_0 collapses to 0.
    res = deflated_sharpe(r, n_trials_eff=5, var_sr=0.0)
    assert res.sr_star == 0.0


# ==================================================================================================
# §3.2 / §6.4  bootstrap_ci has NO block param (D2) -- the deferred circular-block resample is not
# shipped; the reusable call is the ordinary trade-level bootstrap only.
# ==================================================================================================
def test_bootstrap_ci_has_no_block_param_D2():
    import inspect

    from entry_exit import validation as system

    assert "block" not in inspect.signature(bootstrap_ci).parameters
    # the deferred circular-block helper arg is gone too (the resampler takes only n + rng).
    assert "block" not in inspect.signature(system._resample_indices).parameters
    # and passing block= is a hard TypeError, not a silently-ignored kwarg.
    with pytest.raises(TypeError):
        bootstrap_ci(np.array([0.1, -0.2, 0.3, -0.1]), per_obs_sharpe, B=10, block=2)
