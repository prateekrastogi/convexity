"""EvidenceBuilder — assemble the decomposable EvidenceFrame (BREAKOUT §7 / §8).

Combines the calibrated evidence channels into one :class:`EvidenceFrame` with a monotone, decomposable
``quality_score`` (BREAKOUT §8): a fixed weighted sum of GLR / BOCPD / range / compression / trend /
flow / jump / context evidence MINUS a false-break-risk penalty. Weights are RED/BLUE preset values
(never fit freely per symbol, §8). Missing optional channels are explicit zeros, NOT NaNs
(§detect/evidence). The score is monotone in good evidence (Invariant H decomposability), and its
magnitude is opaque to downstream composition (Invariant K — only combine's status/direction matter).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.breakout.types import BreakoutGeometry, EvidenceFrame

_CHANNELS = ("glr", "bocpd", "range", "compression", "trend", "flow", "jump", "context")


class EvidenceBuilder:
    """Build a decomposable EvidenceFrame (BREAKOUT §7). One public method, ``build``."""

    def __init__(self, *, weights: Mapping[str, float], threshold: float) -> None:
        self.weights = dict(weights)
        self.threshold = float(threshold)

    def build(
        self,
        components: Mapping[str, np.ndarray],
        calibrated: Mapping[str, np.ndarray],
        geom: BreakoutGeometry,
    ) -> EvidenceFrame:
        """Return a decomposable EvidenceFrame (§8 weighted-sum, false-break penalty).

        ``components`` carries the raw range/compression/trend/flow/jump/context evidence + the
        ``false_break_risk`` and ``candidate`` arrays; ``calibrated`` carries the e-value transforms
        of the change statistics (glr/bocpd) plus the calibrated range/compression where available.
        Any absent channel resolves to zeros.
        """
        n = len(geom)
        ts = geom.ts

        def ch(src: Mapping[str, np.ndarray], name: str) -> np.ndarray:
            arr = src.get(name)
            return np.zeros(n) if arr is None else np.nan_to_num(np.asarray(arr, dtype=float))

        glr_e = ch(calibrated, "glr")
        bocpd_e = ch(calibrated, "bocpd")
        range_e = ch(components, "range")
        compression_e = ch(components, "compression")
        trend_e = ch(components, "trend")
        flow_e = ch(components, "flow")
        jump_e = ch(components, "jump")
        context_e = ch(components, "context")
        false_break_risk = ch(components, "false_break_risk")
        candidate = ch(components, "candidate").astype(int)
        direction = ch(components, "direction").astype(int)

        w = self.weights
        # a jump (isolated spike) is BAD for a breakout, so jump_e enters with a negative-ish weight
        # by subtracting its evidence: a continuous breakout has LOW jump_e.
        quality = (
            w.get("w_glr", 1.0) * glr_e
            + w.get("w_bocpd", 1.0) * bocpd_e
            + w.get("w_range", 1.0) * range_e
            + w.get("w_compression", 1.0) * compression_e
            + w.get("w_trend", 0.8) * trend_e
            + w.get("w_flow", 0.4) * flow_e
            - w.get("w_jump", 0.5) * jump_e
            + w.get("w_context", 0.3) * context_e
            - w.get("w_false", 1.2) * false_break_risk
        )
        threshold = np.full(n, self.threshold)

        return EvidenceFrame(
            ts=ts,
            candidate=candidate,
            direction=direction,
            glr_e=glr_e,
            bocpd_e=bocpd_e,
            range_e=range_e,
            compression_e=compression_e,
            trend_e=trend_e,
            flow_e=flow_e,
            jump_e=jump_e,
            context_e=context_e,
            false_break_risk=false_break_risk,
            quality_score=quality,
            threshold=threshold,
            reason_codes=_CHANNELS,
        )
