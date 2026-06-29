"""Batch-vs-live deterministic replay (BREAKOUT §7 online/replay / Invariant E).

Drives the online ``PipelineOrchestrator.update`` path one bar at a time and asserts the emitted
signals match the batch ``run_batch`` over the same series (Invariant E batch/live parity), including
a mid-stream snapshot/restore of the deque/histogram streaming state via ``online/state.py``.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, EntrySignal, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.online.state import DetectorState
from convexity.breakout.pipeline import PipelineOrchestrator


def _slice_bar(bars: Bars, t: int) -> Bars:
    """Single-bar slice preserving every present optional array."""

    def s(arr):
        return None if arr is None else arr[t : t + 1]

    return Bars(
        ts=bars.ts[t : t + 1],
        o=bars.o[t : t + 1],
        h=bars.h[t : t + 1],
        l=bars.l[t : t + 1],
        c=bars.c[t : t + 1],
        v=s(bars.v),
        bid=s(bars.bid),
        ask=s(bars.ask),
        trade_count=s(bars.trade_count),
        is_session_open=s(bars.is_session_open),
        is_valid=s(bars.is_valid),
        is_roll_window=s(bars.is_roll_window),
        funding_rate=s(bars.funding_rate),
    )


def replay_online(
    bars: Bars, spec: InstrumentSpec, cfg: ResolvedConfig, *, snapshot_at: int | None = None
) -> list[EntrySignal]:
    """Feed ``bars`` one at a time through ``update``; optionally snapshot/restore mid-stream.

    Returns the concatenation of every signal the online path emits. Compare against
    ``PipelineOrchestrator().run_batch(bars, spec, cfg)`` for the parity assertion (Inv E).
    """
    from collections import deque

    po = PipelineOrchestrator()
    po.reset()
    collected: list[EntrySignal] = []
    n = len(bars)
    for t in range(n):
        if snapshot_at is not None and t == snapshot_at:
            # snapshot the FULL streaming-structure state (deques, histograms, EWMA, null counters,
            # BOCPD posterior, seasonal deques, cooldown, fireable-tail) — NOT the raw bar history
            # (Inv E / §1: the detector restarts from the streaming state alone).
            payload = DetectorState(
                stream=po._stream, rows=list(po._hist_rows),
                dedupe_last_idx_by_side=po._dedupe_last_idx_by_side,
                ts_tail=list(po._ts_tail),
            ).snapshot()
            restored = DetectorState.restore(payload, cfg)
            po2 = PipelineOrchestrator()
            po2._stream = restored.stream
            po2._hist_rows = deque(restored.rows, maxlen=2)
            po2._ts_tail = deque(restored.ts_tail, maxlen=2)
            po2._dedupe_last_idx_by_side = restored.dedupe_last_idx_by_side
            po = po2  # continue from the restored streaming state
        collected.extend(po.update(_slice_bar(bars, t), spec, cfg))
    return collected


def assert_batch_live_parity(
    bars: Bars, spec: InstrumentSpec, cfg: ResolvedConfig, *, snapshot_at: int | None = None
) -> bool:
    """True iff the online signals equal the batch signals on (signal_idx, direction) (Inv E)."""
    batch = PipelineOrchestrator().run_batch(bars, spec, cfg)
    online = replay_online(bars, spec, cfg, snapshot_at=snapshot_at)
    bkey = sorted((s.signal_idx, s.direction) for s in batch)
    okey = sorted((s.signal_idx, s.direction) for s in online)
    return bkey == okey and np.allclose(
        sorted(s.quality_score for s in batch),
        sorted(s.quality_score for s in online),
        atol=1e-9,
    )
