"""TrendStateSnapshot — serializable deterministic live state (TREND §11 online/snapshot).

Serializes the engine's streaming estimator + trail + timing state so a restored engine produces the
SAME next decisions (batch/live parity is a release gate, §11). The streaming state is small and
pure-Python (Kalman state vector/cov, CUSUM/SPRT accumulator, integrated variance, running M_t/X_t,
the trail ExitState), so a JSON-able dict round-trips deterministically.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from convexity.trend.engine import TrendLifeEngine
from convexity.trend.types import ExitState


class TrendStateSnapshot:
    """Snapshot/restore the deterministic engine state (§11)."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    @classmethod
    def of(cls, engine: TrendLifeEngine) -> "TrendStateSnapshot":
        """Capture the engine's streaming + trail state."""
        st = engine.builder._stream
        k = engine.builder.kalman
        seq = engine.builder.seq
        payload = {
            "stream": {
                "integrated_var": st.integrated_var,
                "running_x": st.running_x,
                "running_max_x": st.running_max_x,
                "gap_applied_at": sorted(st.gap_applied_at),
                "jump_count": st.jump_count,
                "bar_count": st.bar_count,
            },
            "kalman": {"init": k._init, "x": list(k.x), "P": [list(r) for r in k.P]},
            "seq": {"s": seq._s, "alarm": seq._alarm, "kind": seq.kind},
            "exit_state": asdict(engine._exit_state) if engine._exit_state is not None else None,
            "closed": engine._closed,
            "last_idx": engine._last_idx,
        }
        return cls(payload)

    def snapshot(self) -> bytes:
        """Serialize to deterministic bytes."""
        return json.dumps(self.payload, sort_keys=True).encode("utf-8")

    @classmethod
    def restore_bytes(cls, data: bytes) -> "TrendStateSnapshot":
        return cls(json.loads(data.decode("utf-8")))

    def into(self, engine: TrendLifeEngine) -> None:
        """Restore the captured state into a fresh engine (same config)."""
        import numpy as np

        p = self.payload
        st = engine.builder._stream
        st.integrated_var = p["stream"]["integrated_var"]
        st.running_x = p["stream"]["running_x"]
        st.running_max_x = p["stream"]["running_max_x"]
        st.gap_applied_at = set(p["stream"]["gap_applied_at"])
        st.jump_count = p["stream"]["jump_count"]
        st.bar_count = p["stream"]["bar_count"]
        k = engine.builder.kalman
        k._init = p["kalman"]["init"]
        k.x = np.array(p["kalman"]["x"])
        k.P = np.array(p["kalman"]["P"])
        seq = engine.builder.seq
        seq._s = p["seq"]["s"]
        seq._alarm = p["seq"]["alarm"]
        engine._exit_state = ExitState(**p["exit_state"]) if p["exit_state"] is not None else None
        engine._closed = p["closed"]
        engine._last_idx = p["last_idx"]
