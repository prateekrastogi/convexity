"""Invariant STOP-XFER tests (REGIME.md §5.4).

The carry DD-stop collapse finding (§5.3: Sharpe 0.57->0.10) must NOT be assumed to transfer to
momentum. A (dd_trigger, dd_reset) pair validated on one sleeve may not be applied to another
without its own regime-conditional ablation passing at FULL statistical power. The gate raises when
(1) the ablation is keyed to a different strategy's return series, or (2) gate (a) is claimed from a
short-history (n_weeks < n_weeks_min_cpcv) frac_pos, or (3) the per-window bleed check (b) failed.
The overlay numerics are NOT touched (Invariant N) — the gate only delegates to the frozen mechanic.
"""

import numpy as np
import pytest

from convexity.regime.overlay import drawdown_stop_scale
from convexity.regime.config.resolve import resolve_config
from convexity.regime.overlay_gate import (
    DDStopAblationRecord,
    ablation_with_recomputed_bleed_check,
    aggregate_bleed_check,
    bleed_check_by_window,
    gated_drawdown_stop_scale,
)


def _promotable_record(strategy_id="momentum"):
    return DDStopAblationRecord(
        strategy_id=strategy_id, dd_trigger=0.15, dd_reset=0.05,
        n_weeks=200, frac_pos=0.7, passed_bleed_check=True,
    )


def test_carry_calibration_cannot_transfer_to_momentum_by_strategy_key():
    """§5.4: an ablation keyed to 'carry' applied to 'momentum' raises (STOP-XFER key mismatch)."""
    cfg = resolve_config()
    carry_rec = _promotable_record(strategy_id="carry")
    net = np.random.default_rng(0).normal(0, 0.02, size=200)
    with pytest.raises(ValueError, match="STOP-XFER"):
        gated_drawdown_stop_scale(net, carry_rec, strategy_id="momentum", cfg=cfg)


def test_short_history_frac_pos_cannot_satisfy_gate_a():
    """§5.4 edge case: n_weeks < n_weeks_min_cpcv means frac_pos alone can't promote -> raises."""
    cfg = resolve_config()
    short = DDStopAblationRecord(
        strategy_id="momentum", dd_trigger=0.15, dd_reset=0.05,
        n_weeks=80, frac_pos=0.9, passed_bleed_check=True,  # short history, high frac_pos
    )
    net = np.random.default_rng(0).normal(0, 0.02, size=80)
    with pytest.raises(ValueError, match="STOP-XFER"):
        gated_drawdown_stop_scale(net, short, strategy_id="momentum", cfg=cfg)


def test_failed_bleed_check_rejects_regardless_of_frac_pos():
    """§5.4 gate (b): a single bleed-like window rejects the candidate regardless of aggregate Sharpe."""
    cfg = resolve_config()
    bled = DDStopAblationRecord(
        strategy_id="momentum", dd_trigger=0.15, dd_reset=0.05,
        n_weeks=200, frac_pos=0.99, passed_bleed_check=False,
    )
    net = np.random.default_rng(0).normal(0, 0.02, size=200)
    with pytest.raises(ValueError, match="STOP-XFER"):
        gated_drawdown_stop_scale(net, bled, strategy_id="momentum", cfg=cfg)


def test_promotable_record_delegates_to_frozen_overlay_bit_for_bit():
    """A promotable, correctly-keyed record delegates to overlay.drawdown_stop_scale UNCHANGED
    (Invariant N — the gate does not alter the numerics)."""
    cfg = resolve_config()
    rec = _promotable_record("momentum")
    net = np.random.default_rng(1).normal(0.001, 0.03, size=200)
    gated = gated_drawdown_stop_scale(net, rec, strategy_id="momentum", cfg=cfg)
    direct = drawdown_stop_scale(net, dd_trigger=rec.dd_trigger, dd_reset=rec.dd_reset)
    np.testing.assert_array_equal(gated, direct)  # byte-for-byte identical (no numeric change)


def test_bleed_check_flags_a_bleed_window():
    """§5.4 gate (b): a window where the stopped series' return is materially below the raw
    series' return is flagged False (the carry-bleed signature)."""
    cfg = resolve_config()
    # raw earns +0.10 over the window; a stop that sits flat gives up all of it -> bleed.
    raw = np.full(10, 0.01)
    scale = np.zeros(10)  # fully stopped out the whole window
    res = bleed_check_by_window(raw, scale, {"win": slice(0, 10)}, cfg=cfg)
    assert res["win"] is False  # stopped 0.0 << raw 0.10 - tol -> bleed detected
    # a window where the stop avoids a genuine loss (raw negative) does NOT flag.
    raw_loss = np.full(10, -0.01)
    res2 = bleed_check_by_window(raw_loss, scale, {"win": slice(0, 10)}, cfg=cfg)
    assert res2["win"] is True


# ------------------------------------------------ gate (b) wiring: recompute, don't trust (§5.4)

def test_aggregate_bleed_check_is_and_over_windows_and_rejects_empty():
    """§5.4 gate (b): the scalar verdict is AND over EVERY window — a single failing window fails the
    whole candidate — and an empty window map is conservatively NOT a pass (§0 P6)."""
    assert aggregate_bleed_check({"a": True, "b": True}) is True
    assert aggregate_bleed_check({"a": True, "b": False}) is False   # one bleed window -> reject
    assert aggregate_bleed_check({}) is False                          # no evidence -> not a pass


def test_recompute_overrides_a_fabricated_passed_bleed_check():
    """Invariant STOP-XFER gate (b) WIRING: a record can arrive with passed_bleed_check=True that is
    NOT backed by the windows. ablation_with_recomputed_bleed_check re-derives the flag from the raw
    per-window bleed check and CORRECTS a fabricated pass DOWN to fail — so a promotion gated on the
    recomputed record raises instead of trusting the hand-supplied bool. Before this wiring nothing
    aggregated bleed_check_by_window into the record; a fabricated True sailed through."""
    cfg = resolve_config()
    # raw earns +0.10 in the window; a fully-stopped scale gives it all up -> the window BLEEDS.
    raw = np.full(10, 0.01)
    scale = np.zeros(10)
    windows = {"2022_bear": slice(0, 10)}
    fabricated = DDStopAblationRecord(
        strategy_id="momentum", dd_trigger=0.15, dd_reset=0.05,
        n_weeks=200, frac_pos=0.9, passed_bleed_check=True,   # LIE: claims no bleed
    )
    corrected = ablation_with_recomputed_bleed_check(fabricated, raw, scale, windows, cfg=cfg)
    assert corrected.passed_bleed_check is False              # recomputed from the bleeding window
    assert corrected.is_promotable(cfg) is False              # gate (b) now fails
    with pytest.raises(ValueError, match="STOP-XFER"):
        gated_drawdown_stop_scale(raw, corrected, strategy_id="momentum", cfg=cfg)


def test_recompute_confirms_a_genuine_pass():
    """The wiring is not one-directional: a record whose windows genuinely do NOT bleed recomputes to
    passed_bleed_check=True and (with gate (a) satisfied) is promotable."""
    cfg = resolve_config()
    raw = np.full(10, -0.01)   # raw LOSES; a stop sitting flat avoids the loss -> no bleed
    scale = np.zeros(10)
    windows = {"2022_bear": slice(0, 10)}
    rec = DDStopAblationRecord(
        strategy_id="momentum", dd_trigger=0.15, dd_reset=0.05,
        n_weeks=200, frac_pos=0.7, passed_bleed_check=False,   # understated; recompute lifts it
    )
    corrected = ablation_with_recomputed_bleed_check(rec, raw, scale, windows, cfg=cfg)
    assert corrected.passed_bleed_check is True
    assert corrected.is_promotable(cfg) is True
