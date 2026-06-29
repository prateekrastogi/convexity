"""DetectorState — serializable online streaming-structure state (BREAKOUT §7 / Invariant E).

Invariant E requires the snapshot to contain the FULL streaming-structure state, not just summary
scalars or — as the previous design did — the raw bar HISTORY: the monotonic-deque contents for every
rolling-extremum barrier, the fixed-bin streaming-histogram state for every rolling quantile /
percentile (compression, regime buckets, empirical null), the EWMA/vol accumulators, the per-bucket
empirical-null counters, the pruned BOCPD posterior, the seasonal prior-session deques, and the
cooldown / active-zone state. Restoring then replaying reproduces identical signals to an
uninterrupted run (asserted by ``online/replay.py``).

This serializes the :class:`~convexity.breakout.online.streaming.StreamingState` engine — the actual
O(1)-amortized per-bar structures — NOT the bar history, which is the whole point of §1 / Inv E: the
detector restarts from the streaming state alone.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque

from convexity.breakout.online.streaming import (
    MonotonicExtremum,
    StreamingHistogram,
    StreamingState,
)


class DetectorState:
    """Serializable online detector state (BREAKOUT §7). ``snapshot`` / ``restore`` round-trip the
    full streaming-structure state of a :class:`StreamingState` (Invariant E)."""

    def __init__(self, stream: StreamingState | None = None, rows: list | None = None,
                 dedupe_last_idx_by_side: dict | None = None,
                 ts_tail: list | None = None) -> None:
        self.stream = stream
        self.rows = rows or []
        self.dedupe_last_idx_by_side = dedupe_last_idx_by_side or {}
        self.ts_tail = ts_tail or []

    def snapshot(self) -> bytes:
        """Serialize the full streaming-structure state for deterministic restart (Inv E)."""
        st = self.stream
        if st is None:
            return json.dumps({"empty": True}).encode()
        dedupe = {str(k): int(v) for k, v in self.dedupe_last_idx_by_side.items()}

        def hist(h: StreamingHistogram | None):
            return None if h is None else h.to_state()

        payload = {
            "empty": False,
            # scalar accumulators
            "ewvar": st.ewvar, "seeded": st.seeded, "t": st.t,
            "prev_ts": st.prev_ts, "ts_step": st.ts_step, "prev_c": st.prev_c,
            "bar_in_sess": st.bar_in_sess, "first_bar": st._first_bar,
            "prev_upper": _f(st.prev_upper), "prev_lower": _f(st.prev_lower),
            "barrier_age": st.barrier_age, "active_zone": st.active_zone,
            # monotonic-deque barrier state
            "up_fast": st.up_fast.to_state(), "dn_fast": st.dn_fast.to_state(),
            # streaming-histogram quantile/percentile state
            "comp_hist": hist(st.comp_hist), "sigma_hist": hist(st.sigma_hist),
            "liq_hist": hist(st.liq_hist),
            # per-bucket + pooled empirical-null counters
            "null_hist": {str(k): {nm: h.to_state() for nm, h in d.items()}
                          for k, d in st.null_hist.items()},
            "null_pool": {nm: h.to_state() for nm, h in st.null_pool.items()},
            # bounded tail buffers
            "x_buf": list(st.x_buf), "c_buf": list(st.c_buf),
            "h_buf": list(st.h_buf), "l_buf": list(st.l_buf),
            # pruned BOCPD posterior
            "R": st.R.tolist(), "bo_mu": st.bo_mu.tolist(), "bo_kappa": st.bo_kappa.tolist(),
            "bo_alpha": st.bo_alpha.tolist(), "bo_beta": st.bo_beta.tolist(),
            # seasonal prior-session deques
            "seasonal_hist": {str(k): list(v) for k, v in st.seasonal_hist.items()},
            # the per-bar rows produced so far (so the array assembler can re-fire the latest bar)
            "rows": [list(_row_tuple(r)) for r in self.rows],
            # persistent forward-only dedupe cooldown state
            "dedupe": dedupe,
            # bounded ts tail (for indexing the actionable bar on continuation)
            "ts_tail": [int(x) for x in self.ts_tail],
        }
        return json.dumps(payload).encode()

    @classmethod
    def restore(cls, payload: bytes, cfg) -> "DetectorState":
        """Restore a streaming-structure snapshot into a live :class:`StreamingState` (Inv E)."""
        import numpy as np

        data = json.loads(payload.decode())
        if data.get("empty", True):
            return cls(stream=None, rows=[])
        st = StreamingState(cfg)
        st.ewvar = float(data["ewvar"])
        st.seeded = bool(data["seeded"])
        st.t = int(data["t"])
        st.prev_ts = data["prev_ts"]
        st.ts_step = float(data["ts_step"])
        st.prev_c = data["prev_c"]
        st.bar_in_sess = int(data["bar_in_sess"])
        st._first_bar = bool(data["first_bar"])
        st.prev_upper = _u(data["prev_upper"])
        st.prev_lower = _u(data["prev_lower"])
        st.barrier_age = int(data["barrier_age"])
        st.active_zone = int(data["active_zone"])
        st.up_fast = MonotonicExtremum.from_state(data["up_fast"])
        st.dn_fast = MonotonicExtremum.from_state(data["dn_fast"])
        st.comp_hist = _h(data["comp_hist"])
        st.sigma_hist = _h(data["sigma_hist"])
        st.liq_hist = _h(data["liq_hist"])
        st.null_hist = {int(k): {nm: StreamingHistogram.from_state(hs) for nm, hs in d.items()}
                        for k, d in data["null_hist"].items()}
        st.null_pool = {nm: StreamingHistogram.from_state(hs) for nm, hs in data["null_pool"].items()}
        st.x_buf = deque(data["x_buf"], maxlen=st._tail)
        st.c_buf = deque(data["c_buf"], maxlen=st._tail)
        st.h_buf = deque(data["h_buf"], maxlen=st._tail)
        st.l_buf = deque(data["l_buf"], maxlen=st._tail)
        st.R = np.asarray(data["R"], dtype=float)
        st.bo_mu = np.asarray(data["bo_mu"], dtype=float)
        st.bo_kappa = np.asarray(data["bo_kappa"], dtype=float)
        st.bo_alpha = np.asarray(data["bo_alpha"], dtype=float)
        st.bo_beta = np.asarray(data["bo_beta"], dtype=float)
        sh: dict = defaultdict(lambda: deque(maxlen=max(st._lookback_sessions, 1)))
        for k, v in data["seasonal_hist"].items():
            sh[int(k)] = deque(v, maxlen=max(st._lookback_sessions, 1))
        st.seasonal_hist = sh
        from convexity.breakout.online.streaming import BarRow
        rows = [BarRow(*tuple(r)) for r in data["rows"]]
        dedupe = {int(k): int(v) for k, v in data.get("dedupe", {}).items()}
        ts_tail = [int(x) for x in data.get("ts_tail", [])]
        return cls(stream=st, rows=rows, dedupe_last_idx_by_side=dedupe, ts_tail=ts_tail)


def _f(v):
    import math
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _u(v):
    return float("nan") if v is None else float(v)


def _h(st):
    return None if st is None else StreamingHistogram.from_state(st)


def _row_tuple(r):
    from dataclasses import astuple
    return astuple(r)
