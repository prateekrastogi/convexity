"""BaselineSuite — required baselines under the IDENTICAL label contract (BREAKOUT §7 / Invariant F).

Every baseline uses the SAME bars, costs, entry-timestamp convention (``entry_idx=signal_idx+1``),
split objects, embargo, horizons, and label function as the main detector (Invariant F). Each
baseline emits ``EntrySignal`` objects through the SAME ``EntrySignal`` shape (with a baseline tag in
reason_codes) so the labeler and report aggregation are byte-for-byte the same path.

Required baselines (BREAKOUT §baselines):
  Donchian channel; ATR/Keltner; opening-range (where sessions exist); CUSUM/Page-Hinkley; GLR-only;
  BOCPD-only; random matched-frequency triggers.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, CostModel, EntrySignal, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.detect.bocpd import BocpdChannel
from convexity.breakout.detect.glr import RobustJointGLR
from convexity.breakout.features.range_geometry import (
    rolling_max_strict_prebar,
    rolling_min_strict_prebar,
)
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.standardize.seasonal import SeasonalProfile
from convexity.breakout.standardize.volatility import VolatilityStandardizer
from convexity.breakout.types import Report
from convexity.breakout.validate.labels import EntryLabeler
from convexity.breakout.validate.stats import aggregate_report
from convexity.breakout.validate.walkforward import SplitPlan


def _signal_from_cross(
    spec: InstrumentSpec, t: int, direction: int, c: np.ndarray, invalid: float, tag: str,
    cfg: ResolvedConfig,
) -> EntrySignal:
    """Build a baseline EntrySignal with the SAME shape + entry-timestamp convention (Inv F/G)."""
    stop = abs(c[t] - invalid)
    barrier = invalid  # baselines have no separate barrier; reuse the invalidation level reference
    return EntrySignal(
        symbol=spec.symbol, signal_idx=t, entry_idx=t + 1, direction=direction,  # type: ignore[arg-type]
        status="entry", quality_score=1.0, threshold=0.0, entry_ref_price=float(c[t]),
        invalidation=float(invalid), stop_distance=float(stop), barrier_price=float(barrier),
        range_width_sigma=0.0, compression_pct=0.0, false_break_risk=0.0, urgency="urgent",
        evidence={}, reason_codes=(tag,), detector_version=cfg.defaults.detector_version,
    )


class BaselineSuite:
    """Required baselines (BREAKOUT §7). One public method, ``evaluate_all``."""

    def evaluate_all(
        self, bars: Bars, spec: InstrumentSpec, cost: CostModel, splits: SplitPlan
    ) -> list[Report]:
        """Evaluate every required baseline under the identical label contract (Inv F)."""
        names = ("donchian", "atr_keltner", "opening_range", "cusum", "glr_only",
                 "bocpd_only", "random")
        reports = []
        for name in names:
            signals = self.signals(name, bars, spec)
            cfg_default = _default_cfg(spec)
            labels = EntryLabeler().label(
                bars, signals, cost, cfg_default.defaults.horizons, cfg_default.defaults.r_targets
            )
            rep = aggregate_report(labels, n_folds=len(splits.folds))
            reports.append(rep)
        return reports

    def signals(self, name: str, bars: Bars, spec: InstrumentSpec) -> list[EntrySignal]:
        """Emit a baseline's EntrySignal list (the identical-contract dispatch)."""
        cfg = _default_cfg(spec)
        builders = {
            "donchian": self._donchian,
            "atr_keltner": self._atr_keltner,
            "opening_range": self._opening_range,
            "cusum": self._cusum,
            "glr_only": self._glr_only,
            "bocpd_only": self._bocpd_only,
            "random": self._random,
        }
        return builders[name](bars, spec, cfg)

    # --- the baselines ---

    def _donchian(self, bars, spec, cfg) -> list[EntrySignal]:
        w = min(cfg.windows)
        c, h, low = bars.c, bars.h, bars.l
        upper = rolling_max_strict_prebar(h, w)
        lower = rolling_min_strict_prebar(low, w)
        out = []
        for t in range(len(bars) - 1):
            if np.isfinite(upper[t]) and c[t] > upper[t]:
                out.append(_signal_from_cross(spec, t, 1, c, lower[t], "donchian", cfg))
            elif np.isfinite(lower[t]) and c[t] < lower[t]:
                out.append(_signal_from_cross(spec, t, -1, c, upper[t], "donchian", cfg))
        return out

    def _atr_keltner(self, bars, spec, cfg) -> list[EntrySignal]:
        c, h, low = bars.c, bars.h, bars.l
        n = len(bars)
        tr = np.maximum(h - low, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(low - np.roll(c, 1))))
        tr[0] = h[0] - low[0]
        atr = _ewma(tr, 20)
        ema = _ewma(c, 20)
        upper = np.roll(ema + 2.0 * atr, 1)
        lower = np.roll(ema - 2.0 * atr, 1)
        upper[0] = lower[0] = np.nan
        out = []
        for t in range(1, n - 1):
            if np.isfinite(upper[t]) and c[t] > upper[t]:
                out.append(_signal_from_cross(spec, t, 1, c, c[t] - 2 * atr[t], "atr_keltner", cfg))
            elif np.isfinite(lower[t]) and c[t] < lower[t]:
                out.append(_signal_from_cross(spec, t, -1, c, c[t] + 2 * atr[t], "atr_keltner", cfg))
        return out

    def _opening_range(self, bars, spec, cfg) -> list[EntrySignal]:
        # opening-range only where sessions exist; else empty (deterministic, §2/§3).
        if bars.is_session_open is None or not cfg.session_anchored_available:
            return []
        c, h, low = bars.c, bars.h, bars.l
        out = []
        or_hi = or_lo = None
        bar_in_sess = 0
        for t in range(len(bars) - 1):
            if bars.is_session_open[t]:
                bar_in_sess = 0
                or_hi, or_lo = h[t], low[t]
            else:
                bar_in_sess += 1
                if bar_in_sess <= 5 and or_hi is not None:
                    or_hi, or_lo = max(or_hi, h[t]), min(or_lo, low[t])
                elif or_hi is not None:
                    if c[t] > or_hi:
                        out.append(_signal_from_cross(spec, t, 1, c, or_lo, "opening_range", cfg))
                    elif c[t] < or_lo:
                        out.append(_signal_from_cross(spec, t, -1, c, or_hi, "opening_range", cfg))
        return out

    def _cusum(self, bars, spec, cfg) -> list[EntrySignal]:
        s = _std(bars, spec, cfg)
        x = s.x
        n = x.size
        k, hlim = 0.5, 5.0
        gp = gn = 0.0
        out = []
        for t in range(1, n - 1):
            gp = max(0.0, gp + x[t] - k)
            gn = min(0.0, gn + x[t] + k)
            if gp > hlim:
                out.append(_signal_from_cross(spec, t, 1, bars.c, bars.c[t] * (1 - 0.002), "cusum", cfg))
                gp = 0.0
            elif gn < -hlim:
                out.append(_signal_from_cross(spec, t, -1, bars.c, bars.c[t] * (1 + 0.002), "cusum", cfg))
                gn = 0.0
        return out

    def _glr_only(self, bars, spec, cfg) -> list[EntrySignal]:
        s = _std(bars, spec, cfg)
        stat = RobustJointGLR().statistic(s, cfg.windows)
        return self._from_stat(bars, spec, cfg, stat, s, "glr_only", thr=0.6)

    def _bocpd_only(self, bars, spec, cfg) -> list[EntrySignal]:
        s = _std(bars, spec, cfg)
        ev, _ = BocpdChannel().run_length_evidence(
            s, cfg.defaults.bocpd_hazard, cfg.defaults.bocpd_max_run_length
        )
        return self._from_stat(bars, spec, cfg, ev, s, "bocpd_only", thr=0.6)

    def _from_stat(self, bars, spec, cfg, stat, s, tag, thr) -> list[EntrySignal]:
        c = bars.c
        out = []
        for t in range(1, len(bars) - 1):
            if stat[t] >= thr:
                d = 1 if s.x[t] >= 0 else -1
                inval = c[t] * (1 - d * 0.002)
                out.append(_signal_from_cross(spec, t, d, c, inval, tag, cfg))
        return out

    def _random(self, bars, spec, cfg) -> list[EntrySignal]:
        # matched-frequency: same number of triggers as the main detector, random placement.
        main = PipelineOrchestrator().run_batch(bars, spec, cfg)
        k = len(main)
        n = len(bars)
        rng = np.random.default_rng(12345)
        if k == 0 or n < 3:
            return []
        idx = rng.choice(np.arange(1, n - 1), size=min(k, n - 2), replace=False)
        out = []
        for t in sorted(int(i) for i in idx):
            d = 1 if rng.random() > 0.5 else -1
            out.append(_signal_from_cross(spec, t, d, bars.c, bars.c[t] * (1 - d * 0.002), "random", cfg))
        return out


def _ewma(x: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, x.size):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def _std(bars, spec, cfg):
    des = SeasonalProfile().fit_transform(bars, 20)
    return VolatilityStandardizer().standardize(bars, des, cfg.defaults.vol_window)


def _default_cfg(spec: InstrumentSpec) -> ResolvedConfig:
    from convexity.breakout.config.resolve import resolve_config

    return resolve_config(spec)
