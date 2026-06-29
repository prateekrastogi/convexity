"""ARL / false-alarm math — MC authoritative (Invariant ARL, §6.6, §6.7, §6.5).

The heavier MC runs use modest run counts so the suite stays fast; the closed-form cross-checks are
exact. ``validate/`` carries scipy and is never imported by the live layers (Invariant V).
"""

from __future__ import annotations

import numpy as np

from convexity.ops.streams.whiten import CausalPrewhitener
from convexity.ops.types import InControlParams
from convexity.ops.validate import arl, derive_limits as dl


def test_two_sided_cusum_arl0_mc_matches_370():
    """MC two-sided ARL0 at (0.5, 4.77) lands within CI of 370 on the IID Gaussian synthetic (§6.6)."""
    est = arl.cusum_arl_mc(0.5, 4.77, n_runs=1200, seed=1)
    assert est.ci_lo <= 376 <= est.ci_hi or 330 < est.mean < 420


def test_derive_limits_returns_h_about_477_not_35():
    """§6.6 / Invariant ARL: solving for two-sided ARL0=370 returns h≈4.77, NOT the ≈3.5 the bare
    one-sided formula gives (the bug)."""
    h = dl.solve_h_for_two_sided_arl0(0.5, 370.0)
    assert abs(h - 4.77) < 0.05
    # the bare one-sided solve gives a distinctly smaller (wrong) h
    from convexity.ops.detectors.cusum import siegmund_arl_oneside
    lo, hi = 1.0, 8.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if siegmund_arl_oneside(0.5, m, 0.0) < 370.0:
            lo = m
        else:
            hi = m
    h_bug = 0.5 * (lo + hi)
    assert h_bug < h - 0.4        # the one-sided solve roughly halves the limit's distance


def test_ar1_student_t_raw_floods_prewhitened_holds():
    """§6.7d / Invariant ARL (the panel-does-not-flood proof, MC-AUTHORITATIVE h):

    The RAW autocorrelated + fat-tailed residual at the IID limit (h=4.77) FLOODS (realized ARL0 <<
    370). Prewhitening (§6.7a) lifts it sharply but, at the IID-derived h, the t(5) fat tail still
    bites — realized ARL0 lands well below 370 (~240). The Invariant's literal claim is that the
    realized ARL0 HOLDS at 370 *after the correction*; the honest correction is the MC-authoritative
    solve (§6.6): derive_limits.solve_h_for_two_sided_arl0_mc runs the actual two-arm OR-chart on the
    AR(1)-Student-t (prewhitened) synthetic and solves for the h whose REALIZED two-sided ARL0 = 370.
    We then assert the realized ARL0 at that solved h CONTAINS 370 (the panel does not flood).
    The IID h=4.77 is retained only as the floods-at-the-wrong-limit contrast."""
    ic = InControlParams(stream="cost_tca", mu0=0.0, sigma0=2.0, phi=(0.4,), rho=0.4, tail_dof=5.0)

    def raw(n, rng):
        return arl.ar1_student_t_stream(n, rho=0.4, dof=5.0, rng=rng)  # standardized raw (unit σ)

    def prewhitened(n, rng):
        r = arl.ar1_student_t_stream(n, rho=0.4, dof=5.0, rng=rng) * 2.0
        pw = CausalPrewhitener(ic)
        return np.array([pw.update(x) for x in r])

    # contrast: raw floods at the IID limit; prewhitening alone does NOT reach 370 at the IID h.
    raw_arl0 = arl.cusum_arl_mc(0.5, 4.77, n_runs=400, max_len=4000, seed=2, stream_fn=raw).mean
    pw_at_iid = arl.cusum_arl_mc(0.5, 4.77, n_runs=400, max_len=8000, seed=3,
                                 stream_fn=prewhitened).mean
    assert raw_arl0 < 150.0                  # floods at the IID limit
    assert pw_at_iid > 1.8 * raw_arl0        # prewhitening sharply lifts the realized ARL0 …
    assert pw_at_iid < 320.0                 # … but the t(5) tail keeps it below 370 at the IID h

    # MC-authoritative correction: solve h on the fat-tailed prewhitened stream so realized ARL0 = 370.
    h_star = dl.solve_h_for_two_sided_arl0_mc(
        0.5, 370.0, n_runs=350, seed=3, stream_fn=prewhitened,
        h_lo=4.0, h_hi=8.0, iters=12, max_len=10000,
    )
    assert h_star > 4.77                     # the fat tail needs a WIDER limit than the IID h
    realized = arl.cusum_arl_mc(0.5, h_star, n_runs=600, max_len=22000, seed=23,
                                stream_fn=prewhitened)
    assert realized.contains(370.0)          # the literal Invariant claim: realized ARL0 HOLDS at 370


def test_shewhart_empirical_quantile_not_gaussian_on_fat_tail():
    """§6.7c: the empirical-t L_s for ARL0=370 exceeds the Gaussian 3σ; the Gaussian L_s=3 floods on
    a t(5) stream (ARL0 far below 370)."""
    ls_emp = dl.derive_shewhart_Ls(370.0, dof=5.0)
    assert ls_emp > 3.0                                  # fatter tail needs a wider limit
    flood = arl.shewhart_arl0_empirical_mc(3.0, dof=5.0, n=400000, seed=0)
    assert flood < 200.0                                 # Gaussian 3σ floods on t(5)
    held = arl.shewhart_arl0_empirical_mc(ls_emp, dof=5.0, n=400000, seed=0)
    assert held > 250.0                                  # the empirical quantile restores it


def test_sprt_asn_mc_matches_boundary_weighted():
    """§6.5: the MC SPRT sample number matches the boundary-weighted ASN (≈99 H0, ≈118 H1), NOT 40–50."""
    table = arl.sprt_asn_table(0.55, 0.45, 0.05, 0.10)
    assert abs(table.asn_h0 - 99.0) < 2.0 and abs(table.asn_h1 - 118.0) < 2.0
    mc_h0 = arl.sprt_asn_mc(0.55, 0.55, 0.45, 0.05, 0.10, n_runs=3000, seed=4)
    assert mc_h0.ci_lo <= table.asn_h0 <= mc_h0.ci_hi or abs(mc_h0.mean - table.asn_h0) < 12.0
    assert mc_h0.mean > 60.0                              # explicitly NOT 40–50


def test_and_rule_joint_arl0_bounded_by_product_and_at_least_single():
    """§6.6a: the windowed-AND joint ARL0 is ≥ either single detector's ARL0 (the AND never makes the
    halt MORE frequent) and ≤ the independence product bound — and because CUSUM and PH on the SAME
    stream are STRONGLY POSITIVELY DEPENDENT, the joint is only MODESTLY larger than the single (far
    below the product), which is exactly why the pinned number is the dependence-aware MC value, not
    the product (the cost stream's single panel source)."""
    single = arl.cusum_arl_mc(0.5, 4.77, n_runs=300, seed=5).mean
    joint = arl.and_rule_joint_arl0_mc(0.5, 4.77, 0.5, 4.77, m_conf=3, n_runs=250, seed=6).mean
    assert joint >= 0.9 * single                          # never more frequent than the single (± MC)
    product_bound = single * single / 3.0
    assert joint <= product_bound                         # ≤ the loose independence product bound
    assert joint < 0.25 * product_bound                   # strong dependence: far below the product


def test_cusum_sprt_lifetime_arl0_not_per_epoch_alpha():
    """§6.5a: the CUSUM-SPRT in-control ARL0 is a single pinned number (the panel source), not the
    bare per-epoch α — many in-control trades pass before a false halt."""
    est = arl.cusum_sprt_arl0_mc(0.55, 0.45, h_edge=8.0, n_runs=400, seed=7)
    assert est.mean > 100.0                               # a designed lifetime rate, not α=0.05


def test_panel_budget_source_count_not_stream_or_detector_count():
    """§6.6: the panel rate sums over INDEPENDENT sources (AND-pairs counted once)."""
    from convexity.ops.validate.reports import StreamLimitRow, build_panel_report
    rows = [
        StreamLimitRow("cost_tca", "cusum+ph(AND)", 4000.0, 10.0, 4000.0),   # one source (the pair)
        StreamLimitRow("vol", "ewma", 2000.0, 10.0, 2000.0),
        StreamLimitRow("fills", "cusum", 2000.0, 10.0, 2000.0),
        StreamLimitRow("edge", "cusum_sprt", 2000.0, 100.0, 2000.0),
    ]
    rep = build_panel_report(rows)
    assert rep.n_independent_sources() == 4
    # rate = 1/4000 + 3·(1/2000); NOT counted as 6+ raw detectors
    assert abs(rep.panel_false_page_rate - (1 / 4000 + 3 / 2000)) < 1e-9
