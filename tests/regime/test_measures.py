"""Measure-layer acceptance gates (REGIME §13: Volatility / Absorption / Turbulence / Liquidity).

Covers Invariants TRACE, DISP, ARFLOOR, ARCOND, ARDISPNORM, CONDNUM, COND, TURBNORM, NREBUILD.
"""

import numpy as np
import pytest

from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.measures._shrinkage import lw_shrink_correlation
from convexity.regime.measures.absorption import (
    AbsorptionState,
    absorption_ratio,
    compute_absorption,
    standardized_shift,
    top_k,
)
from convexity.regime.measures.liquidity import compute_liquidity
from convexity.regime.measures.turbulence import TurbulenceState, compute_turbulence
from convexity.regime.measures.vol_regime import VolState, compute_vol_regime
from convexity.regime.synth import generate as g
from convexity.regime.types import CostState, ExecState


# --------------------------------------------------------------------------- absorption / TRACE


def test_trace_identity_denominator_is_n_t():
    """Σλ = n_t to tolerance on a SHRUNK correlation matrix (Invariant TRACE/ARCOND)."""
    panel = g.calm_panel(w=300, n=40, seed=0)
    shrink = lw_shrink_correlation(panel)
    assert np.isclose(shrink.eigenvalues.sum(), 40.0, atol=1e-8)  # trace identity survives shrink


def test_absorption_ratio_uses_n_t_denominator():
    eig = np.array([5.0, 3.0, 1.0, 1.0])  # sum=10, but denominator must be n_t=4 by TRACE identity
    # AR with k=2 = (5+3)/4 = 2.0 (denominator n_t, NOT re-summed eigenvalue total 10).
    assert absorption_ratio(eig, n_t=4, k=2) == pytest.approx(8.0 / 4.0)


def test_top_k_worked_example():
    assert top_k(0.20, 80) == 16  # k = ceil(0.20*80)


def test_disp_is_raw_level_std_not_mean_difference_se():
    """delta_AR scales by the RAW AR-level std, reproducing the §6.1 +2.20 (Invariant DISP)."""
    assert standardized_shift(0.710, 0.600, 0.050) == pytest.approx(2.20, abs=1e-9)
    # The mean-difference SE substitution would inflate it ~3.7x -> ~8.2; we must NOT do that.
    se = 0.050 * np.sqrt(1 / 15 + 1 / 250)
    wrong = (0.710 - 0.600) / se
    assert wrong > 7.0  # demonstrates the forbidden inflation; our value stays +2.20


def test_arfloor_freezes_on_rank_deficient_window():
    """W_AR < c_AR*n_t -> absorption FREEZES g^corr, flags ar_window_deficient (Invariant ARFLOOR)."""
    panel = g.rank_deficient_panel(n=80, w=60)  # 60 < 2*80
    st = AbsorptionState(last_delta_ar=0.42)
    res = compute_absorption(panel, cfg=DEFAULTS, state=st)
    assert res.fresh is False
    assert res.diag.get("ar_window_deficient") == 1.0
    assert res.delta_ar == 0.42  # frozen at last valid, NOT reset to a fresh trigger


def test_arcond_shrink_reduces_mp_bias_on_noise():
    """On pure-noise T/N≈2, AR on the SHRUNK C_t is less biased-high than on the RAW (Invariant ARCOND)."""
    panel = g.marginal_tn_noise_panel(n=60, c_ar=2.0, seed=3)
    n_t = panel.shape[1]
    k = top_k(DEFAULTS.kappa, n_t)
    # raw sample correlation eigenvalues
    std = panel.std(axis=0, ddof=0)
    z = panel / np.where(std > 0, std, 1.0)
    raw_corr = (z.T @ z) / panel.shape[0]
    np.fill_diagonal(raw_corr, 1.0)
    raw_eig = np.sort(np.linalg.eigvalsh(raw_corr))[::-1]
    ar_raw = raw_eig[:k].sum() / n_t
    shrink = lw_shrink_correlation(panel)
    ar_shrunk = absorption_ratio(shrink.eigenvalues, n_t, k)
    assert ar_shrunk < ar_raw  # shrink pulls the inflated top eigenvalues down


def test_ardispnorm_freezes_across_n_t_step():
    """Corr-structure-fixed panel-shrink: delta_AR does NOT move across the n_t step (ARDISPNORM)."""
    full, shrunk = g.corr_fixed_shrink_pair(w=300, n_full=80, drop=20, seed=8)
    st = AbsorptionState()
    # Warm the FULL-size stratum so it has a valid delta_AR.
    for _ in range(DEFAULTS.w_ar_short + 5):
        compute_absorption(full, cfg=DEFAULTS, state=st)
    res_full = compute_absorption(full, cfg=DEFAULTS, state=st)
    assert res_full.fresh is True
    # Now n_t drops beyond frac_dn -> a DIFFERENT band stratum with no history -> freeze.
    res_shrunk = compute_absorption(shrunk, cfg=DEFAULTS, state=st)
    assert res_shrunk.fresh is False
    assert res_shrunk.diag.get("ar_nt_changed") == 1.0
    assert res_shrunk.delta_ar == res_full.delta_ar  # held frozen across the n_t step


def test_ardispnorm_band_is_current_bar_anchored_membership():
    """A trailing bar WITHIN frac_dn of the current n_t is INCLUDED in the stratum, one BEYOND is
    EXCLUDED — a current-bar-anchored band, NOT a fixed global quantization grid (ARDISPNORM)."""
    from convexity.regime.measures.absorption import _in_band

    # Near-boundary unit test: at n_t=80, frac_dn=0.05 the band is [76, 84] (|n'/80 − 1| ≤ 0.05).
    assert _in_band(77, 80, 0.05) is True  # just inside the lower edge
    assert _in_band(84, 80, 0.05) is True  # at the upper edge (ratio 1.05)
    assert _in_band(70, 80, 0.05) is False  # well below the band
    assert _in_band(90, 80, 0.05) is False  # well above the band
    assert _in_band(85, 80, 0.05) is False  # one past the upper edge -> excluded
    # The grid bug: 76/78/80/84 map to DIFFERENT fixed-grid keys; the anchored band pools them ALL.
    for n in (76, 77, 78, 80, 84):
        assert _in_band(n, 80, 0.05) is True  # every bar within 5% of 80 is in the SAME stratum

    # Integration: warm history ONLY at in-band neighbor sizes (77 and 84), never at 80 itself.
    # The current n_t=80 bar must be FRESH because the in-band neighbors POOL into its stratum.
    full = g.correlation_spike_panel(w=300, n=90, seed=21, rho=0.5)
    st_in = AbsorptionState()
    for i in range(DEFAULTS.w_ar_short + 4):
        size = 77 if i % 2 == 0 else 84  # alternate two in-band sizes, both within 5% of 80
        compute_absorption(full[:, :size], cfg=DEFAULTS, state=st_in)
    res_in = compute_absorption(full[:, :80], cfg=DEFAULTS, state=st_in)
    assert res_in.fresh is True  # in-band neighbors counted -> enough bars -> fresh delta_AR
    assert res_in.diag["stratum_n"] >= DEFAULTS.w_ar_short + 1

    # Now warm history ONLY at OUT-of-band sizes (70 and 90); the n_t=80 bar finds an EMPTY stratum
    # (its own bar aside) and must FREEZE — the beyond-band bars are excluded.
    st_out = AbsorptionState()
    for i in range(DEFAULTS.w_ar_short + 4):
        size = 70 if i % 2 == 0 else 90  # both beyond 5% of 80
        compute_absorption(full[:, :size], cfg=DEFAULTS, state=st_out)
    res_out = compute_absorption(full[:, :80], cfg=DEFAULTS, state=st_out)
    assert res_out.fresh is False
    assert res_out.diag.get("ar_nt_changed") == 1.0
    assert res_out.diag["stratum_n"] == 1.0  # only the current bar is in its own band


def test_condnum_freezes_absorption_on_breach():
    """A near-duplicated pair breaching cond_max -> absorption freezes, cond_breach_corr (CONDNUM)."""
    panel = g.condition_breach_panel(w=200, n=40, seed=9)
    # The breach/freeze path is reachable: force a tiny cond_max so the near-duplicate pair breaches.
    from dataclasses import replace

    tight = replace(DEFAULTS, cond_max=1.5)
    res2 = compute_absorption(panel, cfg=tight, state=AbsorptionState(last_delta_ar=0.1))
    assert res2.diag.get("cond_breach_corr") == 1.0
    assert res2.fresh is False


# --------------------------------------------------------------------------- turbulence / TURBNORM


def test_turbnorm_percentile_stable_across_panel_shrink():
    """Dropping assets on an unchanged-distribution panel does NOT shift turb_pct (Invariant TURBNORM)."""
    full, shrunk = g.panel_shrink_pair(w=300, n_full=80, drop=30, seed=10)
    # Build a common d_norm history from the full panel, then compare a fresh d_norm on full vs shrunk.
    st_full = TurbulenceState()
    st_shrunk = TurbulenceState()
    # seed identical histories
    hist = np.random.default_rng(0).uniform(0.8, 1.2, size=200).tolist()
    st_full.d_norm_history = list(hist)
    st_shrunk.d_norm_history = list(hist)
    r_full = compute_turbulence(full, cfg=DEFAULTS, state=st_full)
    r_shrunk = compute_turbulence(shrunk, cfg=DEFAULTS, state=st_shrunk)
    # d_norm (E≈1) should be close across the n_t change; raw d_t would differ by ~the drop ratio.
    assert r_full.diag["d_norm"] == pytest.approx(r_shrunk.diag["d_norm"], rel=0.5)
    assert abs(r_full.turb_pct - r_shrunk.turb_pct) < 0.25


def test_cond_diagonal_fallback_no_nonsense_spike():
    """Near-singular covariance -> shrink/diagonal fallback, finite d_t (Invariant COND/CONDNUM)."""
    panel = g.near_singular_panel(w=200, n=80, seed=5)
    st = TurbulenceState()
    res = compute_turbulence(panel, cfg=DEFAULTS, state=st)
    assert np.isfinite(res.diag["d_t"])
    assert res.diag["d_t"] < 1e6  # no nonsense inverse-amplified spike


def test_turbulence_window_deficient_uses_diagonal():
    from dataclasses import replace

    panel = g.calm_panel(w=60, n=40, seed=1)  # 60 < 2*40 -> deficient
    cfg = replace(DEFAULTS, w_turb=60)
    res = compute_turbulence(panel, cfg=cfg, state=TurbulenceState())
    assert res.diag.get("cov_window_deficient") == 1.0


# --------------------------------------------------------------------------- vol


def test_vol_pct_is_causal_percentile():
    panel = g.vol_spike_panel(w=300, n=40, seed=2)
    st = VolState()
    # feed a calm history first so the recent spike reads HIGH
    for _ in range(60):
        compute_vol_regime(g.calm_panel(w=300, n=40, seed=99), cfg=DEFAULTS, state=st)
    res = compute_vol_regime(panel, cfg=DEFAULTS, state=st)
    assert 0.0 <= res.vol_pct <= 1.0
    assert res.vol_pct > 0.8  # the vol spike sits high in its own trailing history


# --------------------------------------------------------------------------- liquidity / NREBUILD


def test_liquidity_is_pure_max_of_two_aggregations():
    """liq_stress = max(spread, funding), pure aggregation of EXEC/COST (Invariant NREBUILD)."""
    live = ("A", "B", "C")
    ex = ExecState(spread_pct={"A": 0.9, "B": 0.8, "C": 0.7})
    co = CostState(funding_pct={"A": 0.2, "B": 0.3, "C": 0.1})
    res = compute_liquidity(exec_state=ex, cost_state=co, live=live)
    assert res.spread_blowout == pytest.approx(0.8)
    assert res.funding_extreme == pytest.approx(0.2)
    assert res.liq_stress == pytest.approx(0.8)  # the worse of the two


def test_liquidity_only_live_names_contribute():
    live = ("A", "B")  # C not live
    ex = ExecState(spread_pct={"A": 0.5, "B": 0.5, "C": 0.99})
    co = CostState(funding_pct={"A": 0.5, "B": 0.5})
    res = compute_liquidity(exec_state=ex, cost_state=co, live=live)
    assert res.spread_blowout == pytest.approx(0.5)  # C excluded
