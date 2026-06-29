"""AblationRunner — channel-on/off and interaction tests (BREAKOUT §7 / §10).

Measures the contribution of every evidence channel by zeroing its weight (channel-off), keeping
only it (channel-only), and the interaction. A channel that does not improve OOS quality net of
costs must be disabled by default (BREAKOUT §ablation / §8 optional-channel rule). Runs under the
identical label contract (Invariant F).
"""

from __future__ import annotations

from dataclasses import replace


from convexity.core.types import Bars, CostModel, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.validate.labels import EntryLabeler
from convexity.breakout.validate.stats import aggregate_report
from convexity.breakout.validate.walkforward import SplitPlan

_WEIGHT_KEYS = ("w_glr", "w_bocpd", "w_range", "w_compression", "w_trend", "w_flow", "w_jump",
                "w_context")


class AblationRunner:
    """Channel ablation (BREAKOUT §7). One public method, ``run``."""

    def run(
        self, bars: Bars, spec: InstrumentSpec, cfg: ResolvedConfig, cost: CostModel, splits: SplitPlan
    ) -> list[dict]:
        """Return per-channel channel-off / channel-only contribution rows (§ablation)."""
        po = PipelineOrchestrator()
        labeler = EntryLabeler()

        def score(weights: dict) -> dict:
            c2 = replace(cfg, weights=weights)
            sigs = po.run_batch(bars, spec, c2)
            labels = labeler.label(bars, sigs, cost, cfg.defaults.horizons, cfg.defaults.r_targets)
            rep = aggregate_report(labels, n_folds=len(splits.folds))
            return {"n": rep.n_signals, "convexity": rep.convexity_ratio,
                    "false_break": rep.false_breakout_rate, "net_bps": rep.net_return_mean_bps}

        full = score(dict(cfg.weights))
        rows = [{"channel": "full", **full}]
        for key in _WEIGHT_KEYS:
            if key not in cfg.weights:
                continue
            off = dict(cfg.weights)
            off[key] = 0.0
            r_off = score(off)
            rows.append({
                "channel": key, "mode": "off",
                "delta_convexity": full["convexity"] - r_off["convexity"],
                "delta_false_break": r_off["false_break"] - full["false_break"],
                **{f"off_{k}": v for k, v in r_off.items()},
            })
        return rows
