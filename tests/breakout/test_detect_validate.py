"""Detect + validate + tune behavioural tests (BREAKOUT §15 / §13).

Candidate generation; GLR/BOCPD no-decision (they return arrays, never EntrySignal); empirical-null
fold isolation; evidence decomposition; dedupe/cooldown; stop floor/ceiling; label firewall import;
baseline identical-contract; autotune promotion statuses; and the §13 synthetic-behavior
expectations (true compression->expansion scores high; isolated mean-reverting spikes carry high
false-break risk; roll/CA artifacts do not trigger after adjustment).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import EntrySignal, InstrumentSpec

from convexity.breakout.config.cost import default_cost_model
from convexity.breakout.config.defaults import DetectorDefaults
from convexity.breakout.config.resolve import resolve_config
from convexity.breakout.config.tuning import TuningPolicy
from convexity.breakout.detect.bocpd import BocpdChannel
from convexity.breakout.detect.candidates import CandidateGenerator
from convexity.breakout.detect.dedupe import SignalDeduper
from convexity.breakout.detect.empirical_null import EmpiricalNullCalibrator
from convexity.breakout.detect.glr import RobustJointGLR
from convexity.breakout.features.range_geometry import RangeGeometry
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.standardize.seasonal import SeasonalProfile
from convexity.breakout.standardize.volatility import VolatilityStandardizer
from convexity.breakout.synth.generate import make_regime
from convexity.breakout.tune.autotune import AutoTuner
from convexity.breakout.validate.labels import EntryLabeler


def crypto_spec() -> InstrumentSpec:
    return InstrumentSpec("BTCUSDT", "crypto_perp", "BINANCE", "UTC", tick_size=0.1, lot_size=0.001)


def _std(bars, spec, cfg):
    des = SeasonalProfile().fit_transform(bars, 20)
    return VolatilityStandardizer(
        n_vol_buckets=cfg.defaults.n_vol_buckets, n_liq_buckets=cfg.defaults.n_liq_buckets
    ).standardize(bars, des, cfg.defaults.vol_window)


# --------------------------------------------------------------------------- candidate generation


def test_candidates_return_int_directions_only():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    s = _std(bars, spec, cfg)
    geom = RangeGeometry(family="donchian").compute(bars, s, cfg.windows)
    cand = CandidateGenerator(cross_k=cfg.cross_k).candidates(bars, s, geom)
    assert set(np.unique(cand)).issubset({-1, 0, 1})
    assert cand.dtype == np.int8


def test_candidates_suppress_roll_window():
    spec = InstrumentSpec("CL", "commodity_future", "CME", "UTC", 0.01, 1.0, calendar_id="cme_globex")
    cfg = resolve_config(spec)
    bars = make_regime("futures_roll", 600, 10)
    # mark the roll window so candidates there are suppressed.
    rw = np.zeros(len(bars), dtype=bool)
    mid = len(bars) // 2
    rw[mid - 2 : mid + 3] = True
    from dataclasses import replace

    bars = replace(bars, is_roll_window=rw)
    s = _std(bars, spec, cfg)
    geom = RangeGeometry(family=cfg.barrier_family).compute(bars, s, cfg.windows)
    cand = CandidateGenerator(cross_k=cfg.cross_k).candidates(bars, s, geom)
    assert np.all(cand[mid - 2 : mid + 3] == 0)


# --------------------------------------------------------------------------- GLR / BOCPD no-decision


def test_glr_returns_array_not_signal():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    s = _std(make_regime("variance_breakout", 300, 2), spec, cfg)
    stat = RobustJointGLR().statistic(s, cfg.windows)
    assert isinstance(stat, np.ndarray) and stat.shape == (len(s),)
    assert not isinstance(stat, EntrySignal)


def test_bocpd_returns_arrays_and_responds_to_change():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    s = _std(make_regime("variance_breakout", 400, 2), spec, cfg)
    ev, rl = BocpdChannel().run_length_evidence(s, cfg.defaults.bocpd_hazard, cfg.defaults.bocpd_max_run_length)
    assert ev.shape == rl.shape == (len(s),)
    assert rl.max() > 5  # run length grows in stable stretches


# --------------------------------------------------------------------------- empirical-null isolation


def test_empirical_null_is_past_only():
    """A future perturbation must not change an earlier calibrated value (fold isolation / §12.3)."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 400, 2)
    s = _std(bars, spec, cfg)
    geom = RangeGeometry(family="donchian").compute(bars, s, cfg.windows)
    cand = CandidateGenerator(cross_k=cfg.cross_k).candidates(bars, s, geom)
    stat = RobustJointGLR().statistic(s, cfg.windows)
    cal = EmpiricalNullCalibrator()
    a = cal.calibrate({"glr": stat}, s, cand, cfg.defaults.target_alert_rate)["glr"]
    stat2 = stat.copy()
    stat2[300:] *= 2.0  # perturb the FUTURE statistic
    b = cal.calibrate({"glr": stat2}, s, cand, cfg.defaults.target_alert_rate)["glr"]
    assert np.allclose(a[:250], b[:250])


# --------------------------------------------------------------------------- evidence decomposition


def test_evidence_frame_has_all_channels_no_nans():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    ev, _ = PipelineOrchestrator().build_frame(bars, spec, cfg)
    for arr in (ev.glr_e, ev.bocpd_e, ev.range_e, ev.compression_e, ev.trend_e, ev.flow_e,
                ev.jump_e, ev.context_e, ev.false_break_risk, ev.quality_score):
        assert np.all(np.isfinite(arr)), "no NaNs in any evidence channel"


# --------------------------------------------------------------------------- dedupe / cooldown


def test_dedupe_is_causal_keeps_earliest_per_side():
    sigs = [
        _sig(idx, 1, q) for idx, q in [(100, 2.0), (110, 5.0), (200, 3.0)]
    ] + [_sig(105, -1, 4.0)]
    out = SignalDeduper().apply(sigs, cooldown_bars=30)
    longs = [s for s in out if s.direction == 1]
    # within cooldown of 100, the 110 (higher quality) is suppressed; the EARLIEST 100 is kept.
    assert [s.signal_idx for s in longs] == [100, 200]
    assert any(s.direction == -1 for s in out)  # short kept (different zone)


# --------------------------------------------------------------------------- stop floor / ceiling


def test_stop_floor_rejects_below_noise_floor():
    """A signal whose stop distance is below the tick/spread noise floor is rejected (§9)."""
    spec = InstrumentSpec("X", "crypto_perp", "V", "UTC", tick_size=100.0, lot_size=0.001)
    cfg = resolve_config(spec)  # huge tick => huge stop floor => most signals rejected
    bars = make_regime("drift_breakout", 400, 1)
    sigs = PipelineOrchestrator().run_batch(bars, spec, cfg)
    # with a 100-wide tick the floor is 2*(100+100)=400 in price, far above any real stop -> none fire.
    assert len(sigs) == 0


# --------------------------------------------------------------------------- label firewall import


def test_labeler_is_validation_only_and_labels_from_entry_idx():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    sigs = PipelineOrchestrator().run_batch(bars, spec, cfg)
    labels = EntryLabeler().label(bars, sigs, default_cost_model("crypto_perp"),
                                  cfg.defaults.horizons, cfg.defaults.r_targets)
    for lb, s in zip(labels, sigs):
        assert lb.entry_idx == s.entry_idx == s.signal_idx + 1


# --------------------------------------------------------------------------- baseline identical contract


def test_baseline_signals_share_entry_timestamp_convention():
    from convexity.breakout.validate.baselines import BaselineSuite

    spec = crypto_spec()
    bars = make_regime("drift_breakout", 400, 1)
    for name in ("donchian", "atr_keltner", "cusum", "random"):
        sigs = BaselineSuite().signals(name, bars, spec)
        for s in sigs:
            assert s.entry_idx == s.signal_idx + 1
            assert s.urgency == "urgent"


# --------------------------------------------------------------------------- §13 synthetic behavior


def test_compression_breakout_scores_higher_than_noise():
    """A true compression->expansion breakout scores higher than the null random walk (§13)."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    drift = make_regime("drift_breakout", 600, 1)
    noise = make_regime("null_random_walk", 600, 1)
    ev_d, _ = PipelineOrchestrator().build_frame(drift, spec, cfg)
    ev_n, _ = PipelineOrchestrator().build_frame(noise, spec, cfg)
    assert ev_d.quality_score.max() > ev_n.quality_score.max()


def test_jump_spike_has_high_false_break_or_jump_evidence():
    """An isolated mean-reverting spike carries high jump evidence / false-break risk (§13)."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("jump_spike_revert", 600, 3)
    ev, _ = PipelineOrchestrator().build_frame(bars, spec, cfg)
    j = int(np.argmax(np.abs(np.diff(bars.c)))) + 1
    # near the spike, either jump evidence or false-break risk is elevated.
    window = slice(max(0, j - 1), min(len(bars), j + 3))
    assert ev.jump_e[window].max() > 0.2 or ev.false_break_risk[window].max() > 0.2


# --------------------------------------------------------------------------- quality-score decile table


def test_score_decile_top_outranks_bottom_net_return():
    """BREAKOUT §10 headline: the TOP quality-score decile out-net-returns the BOTTOM decile.

    A direct unit check of ``score_decile_table`` with a monotone score->net relationship: the table
    must decile by the per-signal quality_score (NOT realized MFE_R) and flag the §10 promotion gate
    ``top_decile_outperforms_bottom``."""
    from convexity.breakout.types import EntryLabel
    from convexity.breakout.validate.stats import score_decile_table

    rng = np.random.default_rng(0)
    n = 200
    scores = np.linspace(0.0, 5.0, n)
    # net return increases with score (+ noise) — the real signal the gate must detect.
    nets = 2.0 * scores + rng.normal(0, 0.5, n)
    labels = [
        EntryLabel(signal_idx=i, entry_idx=i + 1, horizon=30, mfe_r=0.0, mae_r=0.0,
                   hit_plus_1r_before_stop=False, hit_plus_2r_before_stop=False,
                   hit_stop_first=False, time_to_plus_1r=None, false_breakout=False,
                   net_return_to_horizon_bps=float(nets[i]))
        for i in range(n)
    ]
    table = score_decile_table(labels, scores=scores, n_deciles=5)
    assert table, "decile table must be populated when scores are threaded"
    # the table is keyed by SCORE (score_lo/score_hi present), not MFE_R.
    assert "score_lo" in table[0] and "score_hi" in table[-1]
    # deciles are score-monotone-ascending and the top decile out-net-returns the bottom.
    assert table[-1]["net_bps"] > table[0]["net_bps"]
    assert table[-1]["top_decile_outperforms_bottom"] is True


def test_walkforward_report_carries_quality_score_decile_table():
    """The walk-forward Report deciles by the BREAKOUT-internal quality_score (threaded as a separate
    array, Inv K), not by a realized-label field, and emits the §10 promotion-gate flag."""
    from convexity.breakout.config.cost import default_cost_model
    from convexity.breakout.validate.walkforward import SplitPlan, WalkForward

    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 900, 1)
    sp = SplitPlan.expanding(len(bars), n_folds=3, embargo_bars=cfg.defaults.embargo_bars)
    rep = WalkForward().evaluate(bars, spec, cfg, default_cost_model("crypto_perp"), sp)
    if rep.n_signals >= 5:
        assert rep.score_decile_table, "score decile table must be populated with threaded scores"
        assert "score_lo" in rep.score_decile_table[0]
        assert "top_decile_outperforms_bottom" in rep.score_decile_table[-1]


# --------------------------------------------------------------------------- fold isolation (§12 leakage)


def test_fold_isolation_refit_without_test_fold_equals_strict_replay():
    """BREAKOUT §12 leakage list: refitting/recalibrating WITHOUT the test fold yields the SAME
    test-time features/evidence as strict fold replay (refit-without-test-fold == strict-replay
    equality on calibrated evidence at the test-window bars).

    The detector is strictly causal: evidence at bar t uses only ``[..t]``. So building the evidence
    frame on the truncated series ``bars[:test_hi]`` (the WalkForward path that never sees the later
    folds) must reproduce, bar-for-bar at every test-window bar, the evidence computed by a strict
    replay over the FULL series (which DOES contain the later folds). Equality there proves the
    calibration/standardization at test time did not leak any post-test-fold data."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 600, 2)
    test_lo, test_hi = 300, 450

    po = PipelineOrchestrator()
    # strict replay: full series (the later "training" fold [test_hi:] is present but must be unused
    # at <= test_hi-1 by causality).
    ev_full, geom_full = po.build_frame(bars, spec, cfg)
    # refit-without-test-fold: truncate to [0, test_hi); recalibrate everything on that window only.
    sub = _slice_to(bars, test_hi)
    ev_sub, geom_sub = PipelineOrchestrator().build_frame(sub, spec, cfg)

    sl = slice(test_lo, test_hi)
    for name in ("glr_e", "bocpd_e", "range_e", "compression_e", "trend_e", "jump_e",
                 "quality_score", "false_break_risk"):
        a = getattr(ev_full, name)[sl]
        b = getattr(ev_sub, name)[sl]
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), f"fold leak in {name}"
    # the calibrated barriers/geometry at the test window must also match bar-for-bar.
    assert np.allclose(geom_full.upper_barrier[sl], geom_sub.upper_barrier[sl],
                       atol=1e-9, equal_nan=True)
    assert np.allclose(geom_full.compression_pct[sl], geom_sub.compression_pct[sl],
                       atol=1e-9, equal_nan=True)


def _slice_to(bars, hi):
    from convexity.breakout.validate.walkforward import _slice_to as _s

    return _s(bars, hi)


# --------------------------------------------------------------------------- autotune promotion status


def test_autotune_insufficient_data_is_shadow():
    spec = crypto_spec()
    bars = make_regime("drift_breakout", 400, 1)
    dec = AutoTuner().onboard_symbol(bars, spec, DetectorDefaults(), TuningPolicy(),
                                     default_cost_model("crypto_perp"))
    assert dec.status in ("approved", "shadow", "rejected")
    # 400 bars yields few OOS signals -> insufficient evidence -> shadow (not forced approval).
    assert dec.status == "shadow"


def test_autotune_returns_decision_fields():
    spec = crypto_spec()
    bars = make_regime("drift_breakout", 400, 1)
    dec = AutoTuner().onboard_symbol(bars, spec, DetectorDefaults(), TuningPolicy(),
                                     default_cost_model("crypto_perp"))
    assert dec.symbol == "BTCUSDT"
    assert dec.detector_version == "breakout-v1"
    assert dec.resolved_config_hash and dec.reason


# --------------------------------------------------------------------------- helpers


def _sig(idx: int, direction: int, q: float) -> EntrySignal:
    return EntrySignal(
        symbol="X", signal_idx=idx, entry_idx=idx + 1, direction=direction, status="entry",
        quality_score=q, threshold=1.0, entry_ref_price=100.0, invalidation=99.0,
        stop_distance=1.0, barrier_price=100.5, range_width_sigma=2.0, compression_pct=0.5,
        false_break_risk=0.1, urgency="urgent", evidence={"glr_e": 1.0}, reason_codes=("glr",),
        detector_version="breakout-v1",
    )
