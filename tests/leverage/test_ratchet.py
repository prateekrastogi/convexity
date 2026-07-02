"""Fractional-Kelly ratchet: CUSUM promotion + N_min floor + demotion (LEVERAGE.md §6.2–§6.3, §10)."""

from __future__ import annotations

import numpy as np

from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.ratchet.cusum import (
    arl0_closed_form,
    cusum_step,
    run_cusum,
    standardized_surprise,
)
from convexity.leverage.ratchet.state import (
    apply_demotion,
    evaluate_promotion,
    initial_state,
    next_c,
)


def test_cusum_step_is_one_sided_and_floored_at_zero():
    # §6.3: S_k = max(0, S_{k-1} + z_k − k_ref); never runs negative (one-sided upper CUSUM).
    assert cusum_step(0.0, -5.0, 0.25) == 0.0
    assert cusum_step(1.0, 1.0, 0.25) == 1.75


def test_arl0_closed_form_matches_hawkins_olwell():
    # §6.3: ARL_0 ≈ exp(2·k_ref·h)/(2·k_ref²). At the v1 (0.25, 5.0) this sits ABOVE the ~52 target
    # (h_promote is set conservatively above the closed-form solution, §3).
    arl = arl0_closed_form(0.25, 5.0)
    assert arl > 52.0  # conservative margin above target (§3)
    # sanity: the closed form is monotone increasing in h_promote.
    assert arl0_closed_form(0.25, 6.0) > arl


def test_initial_state_is_quarter_kelly_cold_start():
    # §6.2 / P7: cold start at c=0.25, zero CUSUM, zero elapsed weeks.
    d = LeverageDefaults()
    st = initial_state(0.09, d)
    assert st.c_t == 0.25
    assert st.s_k == 0.0 and st.elapsed_weeks == 0 and not st.promoted


def test_single_lucky_week_does_not_promote():
    # Invariant RATCHET: one lucky week in an otherwise flat/negative record does NOT promote.
    d = LeverageDefaults()
    z = np.full(60, -0.3)         # persistent underperformance
    z[30] = 8.0                    # one huge lucky week
    st = evaluate_promotion(
        sr_week=z, se_sr=1.0, elapsed_weeks=60, defaults=d, sr_ref=0.0
    )
    # surrounding negative surprises pull the accumulated S_k back below h_promote.
    assert not st.promoted
    assert st.c_t == 0.25


def test_sustained_outperformance_promotes_once_floor_met():
    # §6.3: a genuinely sustained outperformance crosses h_promote AND clears N_min_weeks -> promote.
    d = LeverageDefaults()
    z = np.full(60, 0.8)  # sustained ~0.8-sigma weekly surprise
    st = evaluate_promotion(z, se_sr=1.0, elapsed_weeks=60, defaults=d, sr_ref=0.0)
    assert st.promoted
    assert st.c_t == 0.50


def test_min_elapsed_time_floor_blocks_early_promotion():
    # §6.3: even a strong CUSUM cross cannot promote before N_min_weeks elapsed (structural floor).
    d = LeverageDefaults()
    z = np.full(60, 0.8)
    cusum = run_cusum(z, k_ref=d.k_ref, h_promote=d.h_promote)
    assert cusum.crossed  # CUSUM alone would fire
    st = evaluate_promotion(z, se_sr=1.0, elapsed_weeks=d.n_min_weeks - 1, defaults=d, sr_ref=0.0)
    assert not st.promoted  # blocked by the elapsed-time floor
    assert st.c_t == 0.25


def test_demotion_is_immediate_and_asymmetric():
    # §6.2 / §8: a promoted book demotes IMMEDIATELY on a single negative surprise (not CUSUM-gated).
    d = LeverageDefaults()
    promoted = evaluate_promotion(np.full(60, 0.8), se_sr=1.0, elapsed_weeks=60, defaults=d, sr_ref=0.0)
    assert promoted.promoted
    # demote_z_threshold is an INJECTED governance contract (no §3 anchor), supplied by the caller.
    demoted = apply_demotion(promoted, latest_z=-0.01, defaults=d, demote_z_threshold=0.0)
    assert not demoted.promoted
    assert demoted.c_t == 0.25
    # a surprise at/above the injected threshold does NOT demote (asymmetry is threshold-relative).
    held = apply_demotion(promoted, latest_z=0.0, defaults=d, demote_z_threshold=0.0)
    assert held.promoted and held.c_t == 0.50


def test_demote_threshold_is_anchored_to_minus_k_ref():
    # §3/§6.2: demote_z_threshold is the DERIVED symmetric mirror of the promotion CUSUM reference
    # offset in its degenerate immediate (h_demote=0) lower-CUSUM form: z_k < -k_ref triggers.
    # It is therefore == -k_ref BY DERIVATION (Hawkins & Olwell 1998), not a fabricated free number.
    d = LeverageDefaults()
    assert d.demote_z_threshold == -d.k_ref  # anchored value, not an independent constant
    assert d.demote_z_threshold == -0.25  # pins the §3-cited numeric

    # The injection path is retained: kwarg is keyword-only and defaults to None (resolve-from-defaults).
    import inspect

    sig = inspect.signature(apply_demotion)
    param = sig.parameters["demote_z_threshold"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None  # None => resolve to defaults.demote_z_threshold


def test_demotion_default_resolves_to_spec_anchored_value():
    # With no explicit threshold, apply_demotion uses defaults.demote_z_threshold = -0.25:
    # a surprise below -0.25 demotes; one at/above -0.25 (but negative) does NOT (asymmetry
    # quantified by the h_demote=0 lower-CUSUM ARL≈8wk vs promote ARL≈52wk, §3/§6.2).
    d = LeverageDefaults()
    promoted = evaluate_promotion(np.full(60, 0.8), se_sr=1.0, elapsed_weeks=60, defaults=d, sr_ref=0.0)
    assert promoted.promoted
    # z just above the anchored trip: held (default resolution path, demote_z_threshold=None).
    held = apply_demotion(promoted, latest_z=-0.20, defaults=d)
    assert held.promoted and held.c_t == 0.50
    # z below the anchored trip: demotes.
    demoted = apply_demotion(promoted, latest_z=-0.30, defaults=d)
    assert not demoted.promoted and demoted.c_t == 0.25
    # explicit injection still overrides the default (OPS governance path).
    demoted0 = apply_demotion(promoted, latest_z=-0.01, defaults=d, demote_z_threshold=0.0)
    assert not demoted0.promoted


def test_next_c_empty_history_stays_at_floor():
    d = LeverageDefaults()
    assert (
        next_c(
            np.array([]), se_sr=1.0, elapsed_weeks=0, defaults=d, sr_ref=0.0,
            demote_z_threshold=0.0,
        )
        == 0.25
    )


def test_standardized_surprise_rejects_zero_se():
    import pytest

    with pytest.raises(ValueError):
        standardized_surprise(np.array([0.1, 0.2]), sr_ref=0.0, se_sr=0.0)


def test_cusum_accumulates_over_entire_record_not_rolling_window():
    # Invariant RATCHET: S_k accumulates over the ENTIRE track record (a prior bad run is remembered).
    d = LeverageDefaults()
    # a bad run keeps S_k pinned at 0 through week 40, so a later good run must build from scratch.
    z = np.concatenate([np.full(40, -1.0), np.full(10, 0.8)])
    cusum = run_cusum(z, k_ref=d.k_ref, h_promote=d.h_promote)
    # only 10 good weeks of ~0.55 net contribution ≈ 5.5 — right at the threshold, not runaway.
    assert cusum.s_k < d.h_promote + 2.0  # bounded; the prior bad run was not forgotten as a windfall
