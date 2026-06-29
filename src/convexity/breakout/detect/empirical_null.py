"""EmpiricalNullCalibrator — past-only conditional null calibration (BREAKOUT §7 / Invariant M).

Turns each raw evidence channel into a past-only calibrated e-value / p-value transform, CONDITIONAL
on the instrument-local vol/liquidity regime bucket (Invariant M: a within-instrument
null-calibration device, NOT REGIME's global state). Calibration is conditional on the bucket where
enough data exists, falling back to the pooled (global) null when it does not (BREAKOUT
§detect/empirical_null). Online thresholds at t use observations available before or at t only
(§12.3). Block/bootstrap nulls are preferred over iid simulation for 1-minute data, so the calibrator
uses the EMPIRICAL distribution of the statistic within each bucket (a block-resampling-equivalent
rank), never a parametric iid null.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.breakout.types import StdSeries


class EmpiricalNullCalibrator:
    """Conditional past-only null calibration (BREAKOUT §7). One public method, ``calibrate``."""

    def __init__(self, *, min_events: int = 50, warm: int = 60) -> None:
        self.min_events = min_events  # below this in a bucket, fall back to the pooled null.
        self.warm = warm

    def calibrate(
        self,
        evidence: Mapping[str, np.ndarray],
        s: StdSeries,
        candidates: np.ndarray,
        target_alert_rate: float,
    ) -> Mapping[str, np.ndarray]:
        """Return a calibrated e-value array per channel: the past-only conditional exceedance rank.

        For each channel and bar t, the calibrated value is ``-log(p_hat)`` where ``p_hat`` is the
        past-only fraction of same-bucket history whose statistic was >= the current value (a
        block-empirical right-tail p-value). Conditioning is on the joint (vol_bucket, liq_bucket);
        buckets with fewer than ``min_events`` history fall back to the pooled null. The
        ``target_alert_rate`` floors p_hat so an e-value cannot blow up past the alert budget.
        """
        vb = s.vol_bucket
        lb = s.liq_bucket
        key = vb.astype(np.int64) * 1000 + lb.astype(np.int64)
        n = len(s)
        floor = max(target_alert_rate, 1e-4)

        out: dict[str, np.ndarray] = {}
        for name, stat in evidence.items():
            cal = np.zeros(n)
            for t in range(self.warm, n):
                same = key[:t] == key[t]
                hist = stat[:t][same]
                if hist.size < self.min_events:
                    hist = stat[:t]  # fall back to the pooled (global) past-only null
                if hist.size < 5:
                    continue
                p_hat = max(float(np.mean(hist >= stat[t])), floor)
                cal[t] = -np.log(p_hat)
            out[name] = cal
        return out
