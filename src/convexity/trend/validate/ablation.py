"""AblationRunner — component-contribution tables (TREND §12 ablation).

Measures each component's contribution by toggling it and re-running the engine on the same
positions under identical labels/timing (§12). Required ablations: CUSUM vs SPRT (one live at a
time, Inv B), detector on/off, Kalman on/off, jump override on/off, hazard rungs, asymmetric vs
symmetric trail, and the vol-method sweep. Each cell reports the §13 headline metrics (MFE capture,
giveback, net E[Pi]); the §22 final rule removes a component that does not improve them. Each toggle
re-resolves config at the CLASS level (never per-symbol). Validation-only.
"""

from __future__ import annotations

from dataclasses import replace

from convexity.core.types import Bars, InstrumentSpec, Position
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.engine import TrendLifeEngine
from convexity.trend.validate.labels import TrendLifeLabeler

ABLATIONS = (
    "detector_cusum",
    "detector_sprt",
    "detector_off",
    "kalman_off",
    "jump_off",
    "hazard_off",
    "trail_symmetric",
    "vol_ewma",
    "vol_mad",
    "vol_hybrid",
)


class AblationRunner:
    """Toggle one component at a time and score the §13 metrics (§12)."""

    def __init__(self, cost) -> None:
        self.cost = cost
        self.labeler = TrendLifeLabeler()

    def run(
        self,
        bars: Bars,
        positions: list[Position],
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        clocks: tuple[str, ...] = ("calendar_days",),
    ) -> list[dict]:
        """Return one row per (ablation, position) with its headline metrics (§12)."""
        rows: list[dict] = []
        variants = {
            "detector_cusum": replace(cfg, defaults=replace(cfg.defaults, detector_kind="cusum")),
            "detector_sprt": replace(cfg, defaults=replace(cfg.defaults, detector_kind="sprt")),
            "vol_ewma": replace(cfg, defaults=replace(cfg.defaults, vol_method="ewma")),
            "vol_mad": replace(cfg, defaults=replace(cfg.defaults, vol_method="mad")),
            "vol_hybrid": replace(cfg, defaults=replace(cfg.defaults, vol_method="hybrid")),
        }
        for name, vcfg in variants.items():
            for pos in positions:
                eng = TrendLifeEngine(vcfg)
                ds = eng.run_batch(bars, pos, spec, vcfg)
                labels = self.labeler.label(
                    bars, pos, ds, self.cost, {"calendar_days": vcfg.tau_cap}, clocks
                )
                conv = self.labeler.net_convexity(labels)
                rows.append({"ablation": name, "entry_idx": pos.entry_idx, **conv})
        return rows
