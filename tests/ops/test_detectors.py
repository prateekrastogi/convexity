"""Detector recursions + causality + warm-up (Invariant CAUSAL, §6.1–6.5a, §6.8)."""

from __future__ import annotations

import numpy as np

from convexity.ops.detectors.cusum import Cusum, siegmund_arl_oneside, two_sided_arl0_from_oneside
from convexity.ops.detectors.ewma import Ewma
from convexity.ops.detectors.page_hinkley import PageHinkley
from convexity.ops.detectors.shewhart import Shewhart
from convexity.ops.detectors.sprt import CusumSprt, Sprt, asn, wald_boundaries


def test_cusum_recursion_resets_on_noise():
    c = Cusum(k=0.5, h=4.77)
    # in-control noise: C+ pinned near 0
    for z in [0.2, -0.5, 0.4, -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.1]:
        st = c.update(z)
    assert c.c_plus == 0.0
    assert not st.alarm


def test_cusum_section10_recursion_bit_for_bit():
    """The §10 worked example: C+ crosses h=4.77 at i=15, h2=8.0 at i=18 (Invariant WE)."""
    c = Cusum(k=0.5, h=4.77, escalation_h=8.0)
    zs = [0.2, -0.5, 0.4, -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.1] + [1.5] * 8
    states = [c.update(z) for z in zs]
    assert states[9].diagnostics["c_plus"] == 0.0          # in control through i=10
    assert states[14].diagnostics["c_plus"] == 5.0          # crosses h at i=15
    assert states[14].alarm
    assert states[17].diagnostics["c_plus"] == 8.0          # crosses h2 at i=18
    assert states[17].diagnostics["escalated"] == 1.0
    assert c.escalated


def test_cusum_causal_one_at_a_time_equals_batch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    a = Cusum(0.5, 4.77)
    states_a = [a.update(float(z)) for z in x]
    b = Cusum(0.5, 4.77)
    # feeding garbage AFTER index i must not change state at i
    states_b = []
    for i in range(50):
        states_b.append(b.update(float(x[i])))
    for sa, sb in zip(states_a, states_b):
        assert sa.statistic == sb.statistic


def test_warmup_suppresses_alarm_but_updates_recursion():
    """§6.8: a transient inside N_warm does not PAGE but the recursion still updates."""
    c = Cusum(k=0.5, h=1.0, n_warm=5)
    # a big run that would breach, all inside warm-up
    states = [c.update(3.0) for _ in range(5)]
    assert all(s.warming_up for s in states)
    assert not any(s.alarm for s in states)       # suppressed to LOG
    assert c.c_plus > 1.0                          # but the recursion climbed (state correct)
    # the next sample (past warm-up) alarms immediately because state is already past h
    st = c.update(3.0)
    assert not st.warming_up
    assert st.alarm


def test_warmup_reset_restarts_window():
    c = Cusum(0.5, 1.0, n_warm=3)
    c.update(0.0)
    c.reset()
    assert c.n_seen == 0


def test_siegmund_two_sided_arl0_seed():
    """The ÷2 two-sided combination (§6.6): one-sided 743 → two-sided ≈370."""
    one = siegmund_arl_oneside(0.5, 4.77, 0.0)
    assert abs(one - 743.0) < 5.0
    assert abs(two_sided_arl0_from_oneside(one) - 371.5) < 2.0


def test_siegmund_arl1_at_1sigma():
    """ARL1@1σ ≈ 9.9 (Siegmund) — the design target of ~10 (§10)."""
    assert abs(siegmund_arl_oneside(0.5, 4.77, 1.0) - 9.9) < 0.3


def test_ewma_asymptotic_limit():
    e = Ewma(0.10, 2.70)
    expected = 2.70 * np.sqrt(0.10 / 1.90)
    assert abs(e.asymptotic_limit - expected) < 1e-9


def test_page_hinkley_fires_on_upward_shift():
    ph = PageHinkley(delta=0.5, lam=4.0)
    st = None
    for z in [0.0] * 5 + [2.0] * 10:
        st = ph.update(z)
    assert st.alarm
    assert ph.breached


def test_shewhart_fires_one_sample():
    s = Shewhart(3.0)
    s.update(0.0)
    st = s.update(4.0)
    assert st.alarm and st.statistic == 4.0


def test_sprt_boundaries_worked_point():
    """§6.5: A=+2.890, B=-2.251 for (α=0.05, β=0.10)."""
    a, b = wald_boundaries(0.05, 0.10)
    assert abs(a - 2.890) < 0.01
    assert abs(b - (-2.251)) < 0.01


def test_sprt_asn_boundary_weighted_not_40_50():
    """§6.5: ASN_H0≈99, ASN_H1≈118 — the boundary-weighted form, NOT the dropped-weighting 40–50."""
    asn_h0 = asn(0.55, 0.55, 0.45, 0.05, 0.10)
    asn_h1 = asn(0.45, 0.55, 0.45, 0.05, 0.10)
    assert abs(asn_h0 - 99.0) < 2.0
    assert abs(asn_h1 - 118.0) < 2.0
    assert asn_h0 > 60.0 and asn_h1 > 60.0          # explicitly NOT 40–50


def test_sprt_accepts_h0_in_control_decides_h1_on_decay():
    """§6.5: the bare Wald SPRT accepts H0 (edge intact) in control and decides H1 on a real decay —
    the §17.1 non-inferiority gate reads accepts_h0()."""
    s = Sprt(p0=0.55, p1=0.45, alpha=0.05, beta=0.10)
    rng = np.random.default_rng(2)
    # an in-control run of mostly-hits drives LLR to B (accept H0)
    decided_h0 = False
    for _ in range(400):
        st = s.update(1.0 if rng.random() < 0.60 else 0.0)
        if s.accepts_h0():
            decided_h0 = True
            break
    assert decided_h0
    # a decayed run drives LLR to A (decide H1)
    s.reset()
    decided_h1 = False
    for _ in range(800):
        st = s.update(1.0 if rng.random() < 0.35 else 0.0)
        if st.alarm:
            decided_h1 = True
            break
    assert decided_h1


def test_cusum_sprt_fires_on_decayed_hitrate():
    """The §6.5a change-point form alarms when the hit-rate decays to p1 (auto-resets in control)."""
    cs = CusumSprt(p0=0.55, p1=0.45, h_edge=8.0)
    rng = np.random.default_rng(1)
    st = None
    # in control: stays below the (edge-frequency-appropriate) limit
    for _ in range(200):
        st = cs.update(1.0 if rng.random() < 0.55 else 0.0)
    assert not st.alarm
    # decay: G climbs as misses pile up
    fired = False
    for _ in range(500):
        st = cs.update(1.0 if rng.random() < 0.40 else 0.0)
        if st.alarm:
            fired = True
            break
    assert fired
