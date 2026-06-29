"""BREAKOUT architectural-invariant tests (BREAKOUT §5 / §15, BUILD_PLAN Phase 2).

One test per named invariant: A (only combine emits EntrySignal), B (no future data / future-needle),
C (label firewall), D (RED unreachable from InstrumentSpec), E (batch/live parity + snapshot restore),
F (baseline identical contract), G (confirmation delay moves entry), H (decomposable scores), I
(strict-pre-bar barriers), J (vol-normalized cross, single k), K (score-magnitude firewall), L
(cross-sectional FDR at promotion), M (local conditioning bucket, no REGIME g_t import) + barrier-family
determinism + urgency non-null.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from convexity.core.types import EntrySignal, InstrumentSpec

from convexity.breakout.config.defaults import BREAKOUT_PARAM_REGISTRY
from convexity.breakout.config.instrument import INSTRUMENT_SPEC_FIELDS
from convexity.breakout.config.resolve import resolve_config
from convexity.breakout.features.range_geometry import (
    RangeGeometry,
    rolling_max_strict_prebar,
)
from convexity.breakout.online.replay import assert_batch_live_parity
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.synth.generate import make_regime

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "breakout"


def crypto_spec() -> InstrumentSpec:
    return InstrumentSpec("BTCUSDT", "crypto_perp", "BINANCE", "UTC", tick_size=0.1, lot_size=0.001)


def equity_spec() -> InstrumentSpec:
    return InstrumentSpec(
        "AAPL", "us_equity", "NASDAQ", "America/New_York", tick_size=0.01, lot_size=1.0,
        calendar_id="us_equity_rth",
    )


# ----------------------------------------------------------------------- Inv A: only combine emits


def test_inv_a_only_combine_constructs_entry_signal():
    """Invariant A: only detect/combine.py may construct EntrySignal (and validate/baselines for the
    identical-contract baseline emitter). No feature/detector/standardize/online module does."""
    allowed = {"detect/combine.py", "validate/baselines.py"}
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EntrySignal":
                offenders.append(rel)
    assert offenders == [], f"EntrySignal constructed outside combine: {offenders}"


# ----------------------------------------------------------------------- Inv B: future-needle


def test_inv_b_future_needle_no_leak():
    """Invariant B: perturbing bars AFTER t must not change any signal at or before t."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    po = PipelineOrchestrator()
    base = po.run_batch(bars, spec, cfg)

    tamper = 350
    c2, h2 = bars.c.copy(), bars.h.copy()
    c2[tamper:] *= 1.10
    h2[tamper:] *= 1.10
    tampered = po.run_batch(replace(bars, c=c2, h=h2), spec, cfg)

    before = {(s.signal_idx, s.direction, round(s.quality_score, 9)) for s in base if s.signal_idx < tamper}
    after = {(s.signal_idx, s.direction, round(s.quality_score, 9)) for s in tampered if s.signal_idx < tamper}
    assert before == after


# ----------------------------------------------------------------------- Inv C: label firewall


def test_inv_c_label_firewall():
    """Invariant C: no live layer (detect/features/standardize/online/pipeline) imports validate.labels."""
    live_dirs = ("detect", "features", "standardize", "online")
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        is_live = rel == "pipeline.py" or any(rel.startswith(f"{d}/") for d in live_dirs)
        if not is_live:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            if mod and "validate.labels" in mod:
                offenders.append(rel)
    assert offenders == [], f"live layer imports validate.labels: {offenders}"


def test_inv_c_label_import_is_reachable_from_validate_only():
    """The label module itself imports cleanly (it is the firewalled future-data home)."""
    from convexity.breakout.validate import labels  # noqa: F401


# ----------------------------------------------------------------------- Inv D: RED unreachable


def test_inv_d_red_unreachable_from_instrument_spec():
    """Invariant D: no RED/BLUE param name leaks into InstrumentSpec's AMBER field set."""
    from convexity.core.params import assert_no_red_in_amber

    assert_no_red_in_amber(INSTRUMENT_SPEC_FIELDS, BREAKOUT_PARAM_REGISTRY)  # must not raise


def test_inv_d_resolve_asserts_firewall():
    """resolve_config runs the firewall (a smoke check it does not raise on a valid spec)."""
    resolve_config(crypto_spec())


# ----------------------------------------------------------------------- Inv E: batch/live parity


def test_inv_e_batch_live_parity():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 2)
    assert assert_batch_live_parity(bars, spec, cfg)


def test_inv_e_snapshot_restore_parity():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 400, 3)
    assert assert_batch_live_parity(bars, spec, cfg, snapshot_at=200)


# ----------------------------------------------------------------------- Inv F: baseline contract


def test_inv_f_baselines_identical_label_shape():
    from convexity.breakout.config.cost import default_cost_model
    from convexity.breakout.validate.baselines import BaselineSuite
    from convexity.breakout.validate.walkforward import SplitPlan

    spec = crypto_spec()
    bars = make_regime("drift_breakout", 400, 1)
    sp = SplitPlan.expanding(len(bars), 3, 120)
    reports = BaselineSuite().evaluate_all(bars, spec, default_cost_model("crypto_perp"), sp)
    # every baseline returns the SAME Report shape (Inv F).
    for r in reports:
        assert hasattr(r, "convexity_ratio") and hasattr(r, "false_breakout_rate")


# ----------------------------------------------------------------------- Inv G: confirmation delay


def test_inv_g_entry_idx_is_signal_idx_plus_one():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    sigs = PipelineOrchestrator().run_batch(bars, spec, cfg)
    assert sigs, "expected at least one signal"
    for s in sigs:
        assert s.entry_idx == s.signal_idx + 1


# ----------------------------------------------------------------------- Inv H: decomposable


def test_inv_h_scores_are_decomposable():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    sigs = PipelineOrchestrator().run_batch(bars, spec, cfg)
    for s in sigs:
        assert s.evidence, "every signal carries component evidence"
        assert s.reason_codes, "every signal carries reason codes"


# ----------------------------------------------------------------------- Inv I: strict-pre-bar


def test_inv_i_barrier_is_strictly_pre_bar():
    """upper_barrier[t] == rolling_max(h[t-W .. t-1]); raising bar t's own high does NOT raise it."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 300, 1)
    s = _std(bars, spec, cfg)
    W = min(cfg.windows)
    geom = RangeGeometry(family="donchian").compute(bars, s, (W,))

    # identity: the barrier equals the explicit strict-pre-bar rolling max.
    expect = rolling_max_strict_prebar(bars.h, W)
    finite = np.isfinite(expect) & np.isfinite(geom.upper_barrier)
    assert np.allclose(geom.upper_barrier[finite], expect[finite])

    # raising bar t's own high must not raise the level it must break.
    t = 150
    h2 = bars.h.copy()
    h2[t] *= 2.0
    geom2 = RangeGeometry(family="donchian").compute(replace(bars, h=h2), s, (W,))
    assert geom2.upper_barrier[t] == geom.upper_barrier[t]


def test_inv_i_inclusive_max_never_fires_but_strict_does():
    """A true compression->expansion breakout fires under the STRICT barrier; a self-referential
    inclusive-max barrier (max includes h[t]) is unbreakable (c[t] > max is impossible)."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    W = min(cfg.windows)

    strict_upper = rolling_max_strict_prebar(bars.h, W)
    strict_fires = int(np.sum(np.isfinite(strict_upper) & (bars.c > strict_upper)))

    # inclusive max over [t-W+1 .. t] contains h[t] >= c[t], so c[t] > inclusive_max never holds.
    incl = np.full(len(bars), -np.inf)
    for t in range(len(bars)):
        lo = max(0, t - W + 1)
        incl[t] = np.max(bars.h[lo : t + 1])
    incl_fires = int(np.sum(bars.c > incl))

    assert strict_fires > 0
    assert incl_fires == 0


# ----------------------------------------------------------------------- Inv J: vol-normalized cross


def test_inv_j_one_tick_poke_does_not_fire_but_k_sigma_does():
    """A 1-tick poke past the barrier produces NO candidate; a >= k*sigma thrust does."""
    from convexity.breakout.detect.candidates import CandidateGenerator

    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("null_random_walk", 300, 1)
    s = _std(bars, spec, cfg)
    geom = RangeGeometry(family="donchian").compute(bars, s, cfg.windows)

    gen = CandidateGenerator(cross_k=cfg.cross_k)
    t = 200
    upper_t = geom.upper_barrier[t]
    assert np.isfinite(upper_t)
    sigma_price = s.sigma[t] * bars.c[t]

    # a sub-threshold poke (a small fraction of k*sigma past the barrier) must NOT fire — the cross
    # is vol-normalized, not raw price (Inv J): a tiny poke is not a k-sigma event.
    c_poke = bars.c.copy()
    c_poke[t] = upper_t + 0.1 * cfg.cross_k * sigma_price
    cand_poke = gen.candidates(replace(bars, c=c_poke), s, geom)
    assert cand_poke[t] == 0, "a sub-k-sigma poke must not be a candidate (Inv J)"

    # a >= k*sigma thrust must fire.
    c_thrust = bars.c.copy()
    c_thrust[t] = upper_t + 5.0 * cfg.cross_k * sigma_price
    cand_thrust = gen.candidates(replace(bars, c=c_thrust), s, geom)
    assert cand_thrust[t] == 1, "a k-sigma thrust must be a candidate (Inv J)"


def test_inv_j_same_k_runs_on_crypto_and_equity():
    """The SAME k runs on a crypto fixture and an equity fixture (no per-symbol k)."""
    ccfg = resolve_config(crypto_spec())
    ecfg = resolve_config(equity_spec())
    assert ccfg.cross_k == ecfg.cross_k  # single shared RED scalar


def test_inv_j_k_unreachable_from_instrument_spec():
    """A per-symbol PnL fit of k is structurally impossible: k is RED, not an InstrumentSpec field."""
    assert "cross_k" not in INSTRUMENT_SPEC_FIELDS


# ----------------------------------------------------------------------- Inv K: score firewall


def test_inv_k_score_scale_invariance():
    """Scaling every quality_score AND threshold by c>0 must not change the emitted entry set, their
    direction, or the downstream-facing price levels (Invariant K / SYSTEM Inv CB)."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    po = PipelineOrchestrator()
    base = po.run_batch(bars, spec, cfg)

    c = 7.0
    scaled_weights = {k: c * v for k, v in cfg.weights.items()}
    cfg2 = replace(cfg, weights=scaled_weights, defaults=replace(cfg.defaults,
                                                                  quality_threshold=c * cfg.defaults.quality_threshold))
    scaled = po.run_batch(bars, spec, cfg2)

    bkey = [(s.signal_idx, s.direction, s.invalidation, s.stop_distance, s.urgency) for s in base]
    skey = [(s.signal_idx, s.direction, s.invalidation, s.stop_distance, s.urgency) for s in scaled]
    assert bkey == skey


# ----------------------------------------------------------------------- Inv L: cross-sectional FDR


def test_inv_l_noise_universe_fdr_controls_approvals():
    """On a pure-noise universe the promotion gate approves ~the FDR-nominal fraction, not the naive
    uncorrected fraction (Invariant L)."""
    from convexity.breakout.tune.promotion import promote_universe

    rng = np.random.default_rng(0)
    n_syms = 50
    # noise universe: uniform p-values; the single gate provisionally 'approved' for all.
    provisional = [(f"S{i}", "approved", float(rng.uniform())) for i in range(n_syms)]
    out = promote_universe(provisional, fdr_alpha=0.10)
    n_approved = sum(1 for v in out.values() if v == "approved")
    # uncorrected naive @ alpha=0.10 would approve ~ n_syms*1.0 (all 'approved'); BH approves few.
    assert n_approved <= max(1, int(0.20 * n_syms)), f"FDR did not control approvals: {n_approved}"


def test_inv_l_registry_records_universe_scan_n():
    from convexity.breakout.tune.registry import RegistryArtifact

    art = RegistryArtifact(
        symbol="BTCUSDT", detector_version="breakout-v1", resolved_config_hash="abc",
        data_hash="def", fit_window=(0, 100), validation_window=(50, 100),
        universe_scan_n=87, blue_grid_size=48,
    )
    assert art.universe_scan_n == 87 and art.blue_grid_size == 48


# ----------------------------------------------------------------------- Inv M: no REGIME g_t import


def test_inv_m_no_regime_import():
    """Invariant M: BREAKOUT must not consume/reproduce/gate on REGIME's global g_t — no breakout
    module imports convexity.regime."""
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(nm.startswith("convexity.regime") for nm in names):
                offenders.append(path.relative_to(SRC).as_posix())
    assert offenders == [], f"breakout imports REGIME (violates Inv M): {offenders}"


def test_inv_m_buckets_are_local_and_past_only():
    """The vol/liq buckets are within-instrument, past-only integers (the local null-calibration
    device), not a global state."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 300, 2)
    s = _std(bars, spec, cfg)
    assert s.vol_bucket.dtype.kind == "i"
    assert set(np.unique(s.vol_bucket)).issubset(set(range(cfg.defaults.n_vol_buckets)))


# ----------------------------------------------------------------------- barrier-family determinism


def test_barrier_family_determinism_crypto_disables_opening_range():
    """A 24/7 instrument (no session boundary) cannot select the opening-range/anchored variant; it
    falls back deterministically to rolling Donchian (§2/§3)."""
    from convexity.breakout.config.asset_class import AssetClassOverride

    spec = crypto_spec()
    override = AssetClassOverride("crypto_perp", barrier_family="opening_range")
    cfg = resolve_config(spec, override=override)
    assert cfg.barrier_family == "donchian"
    assert not cfg.session_anchored_available


def test_barrier_family_session_instrument_keeps_opening_range():
    cfg = resolve_config(equity_spec())  # us_equity_rth declares sessions
    assert cfg.barrier_family == "opening_range"
    assert cfg.session_anchored_available


# ----------------------------------------------------------------------- urgency non-null


def test_urgency_is_urgent_and_non_null():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    sigs = PipelineOrchestrator().run_batch(bars, spec, cfg)
    assert sigs
    for s in sigs:
        assert s.urgency == "urgent"
        assert s.urgency is not None


# ----------------------------------------------------------------------- public adapter surface


def test_public_adapter_exposes_only_boundary_fields():
    """BreakoutModule.propose returns an EntrySignal whose downstream-read fields are present; a mock
    SYSTEM adapter reads only direction/status/urgency/levels (Inv K companion)."""
    from convexity.breakout.detect.combine import BreakoutModule

    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 400, 1)
    mod = BreakoutModule(PipelineOrchestrator(), cfg, bars)
    sig = mod.propose(spec)
    if sig is not None:
        # the downstream-facing fields a SYSTEM adapter is allowed to read.
        _ = (sig.direction, sig.status == "entry", sig.urgency, sig.invalidation,
             sig.stop_distance, sig.entry_ref_price, sig.barrier_price, sig.entry_idx)
        assert isinstance(sig, EntrySignal)


# ----------------------------------------------------------------------- helpers


def _std(bars, spec, cfg):
    from convexity.breakout.standardize.seasonal import SeasonalProfile
    from convexity.breakout.standardize.volatility import VolatilityStandardizer

    des = SeasonalProfile().fit_transform(bars, 20)
    return VolatilityStandardizer(
        n_vol_buckets=cfg.defaults.n_vol_buckets, n_liq_buckets=cfg.defaults.n_liq_buckets
    ).standardize(bars, des, cfg.defaults.vol_window)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
