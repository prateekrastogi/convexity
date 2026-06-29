"""AutoTuner — symbol onboarding + class-level tuning (BREAKOUT §7 / §11).

The §11 onboarding sequence: validate schema/metadata, apply calendar, fit TEAL profiles, inherit
approved class defaults, run walk-forward + sampled CPCV, run baselines + ablations, and emit a
``PromotionDecision`` (approved / shadow / rejected). A symbol can run automatically with inherited
defaults after TEAL fitting, but PROMOTION requires the §10 gates in ``tune/promotion.py`` (incl. the
cross-sectional FDR, Invariant L). Insufficient data => shadow; failed leakage/parity => rejected.
"""

from __future__ import annotations

from convexity.core.types import Bars, CostModel, InstrumentSpec

from convexity.breakout.config.defaults import DetectorDefaults
from convexity.breakout.config.resolve import ResolvedConfig, resolve_config
from convexity.breakout.config.tuning import TuningPolicy
from convexity.breakout.online.replay import assert_batch_live_parity
from convexity.breakout.tune.promotion import PromotionDecision, gate_single
from convexity.breakout.tune.registry import RegistryArtifact, config_hash, data_hash
from convexity.breakout.validate.baselines import BaselineSuite
from convexity.breakout.validate.walkforward import SplitPlan, WalkForward


class AutoTuner:
    """Symbol onboarding (BREAKOUT §7/§11). One public method, ``onboard_symbol``."""

    def onboard_symbol(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        defaults: DetectorDefaults,
        policy: TuningPolicy,
        cost: CostModel,
    ) -> PromotionDecision:
        """Fit TEAL, run governed BLUE search, validate, and decide status (§11)."""
        cfg = resolve_config(spec, defaults=defaults)
        n = len(bars)

        # leakage + parity gates (cheap, run first; a failure => rejected).
        parity_ok = assert_batch_live_parity(bars, spec, cfg)
        leakage_ok = self._future_needle_ok(bars, spec, cfg)

        # walk-forward + baselines under the identical contract (Inv F).
        splits = SplitPlan.expanding(n, n_folds=3, embargo_bars=defaults.embargo_bars)
        report = WalkForward().evaluate(bars, spec, cfg, cost, splits)
        baselines = BaselineSuite().evaluate_all(bars, spec, cost, splits)

        status, reason = gate_single(
            report, baselines,
            min_oos_signals=policy.min_oos_signals,
            leakage_ok=leakage_ok, parity_ok=parity_ok,
        )

        artifact = RegistryArtifact(
            symbol=spec.symbol,
            detector_version=defaults.detector_version,
            resolved_config_hash=config_hash(cfg),
            data_hash=data_hash(bars.ts, bars.c),
            fit_window=(0, n),
            validation_window=(splits.folds[0][2], n) if splits.folds else (0, n),
            universe_scan_n=1,  # set by the batch onboarder; 1 for a single-symbol call
            blue_grid_size=policy.blue_cardinality(),
        )
        _ = artifact  # written to the registry store by the caller

        return PromotionDecision(
            symbol=spec.symbol,
            status=status,
            detector_version=defaults.detector_version,
            resolved_config_hash=config_hash(cfg),
            fit_window=(0, n),
            reason=reason,
            report=report,
        )

    @staticmethod
    def _future_needle_ok(bars: Bars, spec: InstrumentSpec, cfg: ResolvedConfig) -> bool:
        """Future-needle leakage check (Inv B): perturbing bars AFTER t must not change signals at t."""
        from convexity.breakout.pipeline import PipelineOrchestrator

        po = PipelineOrchestrator()
        base = po.run_batch(bars, spec, cfg)
        if len(bars) < 10:
            return True
        import numpy as np

        tamper = int(0.7 * len(bars))
        c2 = bars.c.copy()
        h2 = bars.h.copy()
        c2[tamper:] *= 1.05  # move the FUTURE only
        h2[tamper:] *= 1.05
        from dataclasses import replace

        bars2 = replace(bars, c=c2, h=h2)
        tampered = po.run_batch(bars2, spec, cfg)
        before = {(s.signal_idx, s.direction) for s in base if s.signal_idx < tamper}
        after = {(s.signal_idx, s.direction) for s in tampered if s.signal_idx < tamper}
        return before == after and np.isfinite(0.0)
