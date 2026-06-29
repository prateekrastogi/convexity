"""PipelineOrchestrator — full detector wiring, batch + online (BREAKOUT §7 pipeline / Invariant E).

Composes data -> standardize -> features -> detect into end-to-end signal generation in both batch
replay (``run_batch``) and one-bar online update (``update``) modes. Per Invariant E the two paths
produce tolerance-identical signals after warm-up.

HOT PATH (BREAKOUT §1, the 87-instrument / 2-core requirement): the per-bar feature + evidence
scalars are produced by the persistent, amortized-O(1) :class:`StreamingState` engine
(``online/streaming.py``) — monotonic deques for the rolling-extremum barriers, an EWMA accumulator
for sigma, fixed-bin streaming histograms for the compression / regime-bucket / empirical-null
percentiles, a pruned BOCPD posterior, and bounded tail buffers for the windowed GLR / trend / jump
channels. ``build_frame`` drives that SAME engine sequentially over the batch, and ``update`` advances
ONE persistent engine per bar — so ``run_batch == sequential update()`` holds by construction (Inv E),
and the stream is O(1)/bar rather than the O(n)/bar full-batch re-run the previous design used.

FIREWALL (Invariant C): this module MUST NOT import ``validate.labels`` or any future-looking label
(enforced by the AST import-graph test). It produces ``EntrySignal`` (-> SYSTEM) and seeds
``Position`` (-> TREND) only; no MFE/MAE/horizon return is reachable here.
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from convexity.core.types import Bars, EntrySignal, InstrumentSpec

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.data.calendar import Sessionizer
from convexity.breakout.data.schema import BarValidator
from convexity.breakout.detect.combine import SignalAssembler
from convexity.breakout.detect.dedupe import SignalDeduper
from convexity.breakout.detect.evidence import EvidenceBuilder
from convexity.breakout.features.context import ContextEvidence
from convexity.breakout.features.flow import FlowEvidence
from convexity.breakout.online.streaming import StreamingState
from convexity.breakout.standardize.seasonal import SeasonalProfile
from convexity.breakout.standardize.volatility import VolatilityStandardizer
from convexity.breakout.types import BreakoutGeometry, EvidenceFrame, StdSeries


class PipelineOrchestrator:
    """End-to-end detector (BREAKOUT §7). Public methods: ``run_batch`` and ``update``."""

    def __init__(self) -> None:
        self._validator = BarValidator()
        self._sessionizer = Sessionizer()
        self._seasonal = SeasonalProfile()
        self._deduper = SignalDeduper()
        # persistent online state (set up on the first update()).
        self._stream: StreamingState | None = None
        # bounded fireable tail (only [signal bar, current bar] are needed to fire — keeps O(1)/bar).
        self._hist_rows: deque = deque(maxlen=2)
        self._ts_tail: deque = deque(maxlen=2)
        self._latest_online_bar: Bars | None = None
        self._cfg_weights: dict = {}
        self._cfg_threshold: float = 0.0
        # persistent forward-only dedupe state (incremental cooldown, matches SignalDeduper, Inv E).
        self._dedupe_last_idx_by_side: dict = {}

    # --------------------------------------------------------------------- batch

    def run_batch(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        context: Mapping[str, Bars] | None = None,
    ) -> list[EntrySignal]:
        """Run the full detector in historical replay mode. No labels (Inv C firewall)."""
        prepared = self._prepared(bars, spec)
        ev, geom = self.build_frame(bars, spec, cfg, context)
        assembler = SignalAssembler()
        signals = assembler.fire(spec.symbol, prepared, spec, geom, ev, cfg)
        return self._deduper.apply(signals, cfg.defaults.cooldown_bars)

    def build_frame(
        self,
        bars: Bars,
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        context: Mapping[str, Bars] | None = None,
    ) -> tuple[EvidenceFrame, BreakoutGeometry]:
        """Compute the EvidenceFrame + BreakoutGeometry by driving the SAME streaming engine over the
        batch (the Inv E single source of truth). The optional flow/context channels are vectorised
        post-hoc (they default OFF, §8) so the default path the tests exercise matches ``update``."""
        prepared = self._prepared(bars, spec)
        s, geom, ev = self._frame_from_stream(prepared, spec, cfg, StreamingState(cfg), context)
        return ev, geom

    # --------------------------------------------------------------------- online

    def update(
        self,
        bar: Bars,
        spec: InstrumentSpec,
        cfg: ResolvedConfig,
        context_bar: Mapping[str, Bars] | None = None,
    ) -> list[EntrySignal]:
        """Run one online update in amortized O(1) (Inv E parity / §1 hot path).

        Advances the persistent :class:`StreamingState` by ONE bar (no full-history re-run). The
        per-bar rows are accumulated only so the array-based assembler/dedupe can re-evaluate the
        latest fireable bar; the streaming structures themselves are the O(1) state. Returns the
        signal that becomes actionable now (the entry whose ``signal_idx`` is the previous bar)."""
        prepared = self._prepared_online(bar, spec)
        if self._stream is None:
            self._stream = StreamingState(cfg)
            self._hist_rows = deque(maxlen=2)
        self._cfg_weights = cfg.weights
        self._cfg_threshold = cfg.defaults.quality_threshold

        # advance ONE bar through the persistent engine (O(1) amortized).
        fields = self._bar_fields(prepared, 0)
        row = self._stream.advance(*fields)
        self._hist_rows.append(row)  # bounded deque (maxlen 2) — only the fireable tail is needed.
        last = self._stream.t - 1  # authoritative global bar index (engine bar counter)

        # the bar that becomes ACTIONABLE now is the previous one (signal_idx == last-1, entry now);
        # evaluate ONLY that bar (O(1)) instead of re-firing the whole history (the previous O(n)/bar
        # design). geom/ev are built over the 2-row tail so ``fire_at`` can index it at t=0.
        sig_idx = last - 1
        if sig_idx < 0 or len(self._hist_rows) < 2:
            return []
        tail = list(self._hist_rows)[-2:]  # [signal bar, current bar]
        ts_tail = np.asarray(list(self._ts_tail))  # bounded 2-element ts tail
        geom = self._geom_from_rows(tail, ts_tail)
        ev = self._ev_from_rows(tail, geom, flow=None, context=None)
        assembler = SignalAssembler()
        sig = assembler.fire_at(spec.symbol, self._stream_bars(prepared), spec, geom, ev, cfg, t=0)
        if sig is None:
            return []
        # the assembler indexed the tail at 0; restore the true global indices.
        from dataclasses import replace as _replace
        sig = _replace(sig, signal_idx=sig_idx, entry_idx=sig_idx + 1)
        # incremental forward-only dedupe (matches SignalDeduper; persistent cooldown state, Inv E).
        side = sig.direction
        prev = self._dedupe_last_idx_by_side.get(side)
        if prev is not None and sig.signal_idx - prev <= cfg.defaults.cooldown_bars:
            return []
        self._dedupe_last_idx_by_side[side] = sig.signal_idx
        return [sig]

    def reset(self) -> None:
        """Clear the persistent online state (start a fresh stream)."""
        self._stream = None
        self._hist_rows = deque(maxlen=2)
        self._ts_tail = deque(maxlen=2)
        self._latest_online_bar = None
        self._dedupe_last_idx_by_side = {}

    # --------------------------------------------------------------------- internals

    def _frame_from_stream(self, prepared, spec, cfg, stream, context):
        n = len(prepared)
        rows = []
        for t in range(n):
            rows.append(stream.advance(*self._bar_fields(prepared, t)))
        # cache the resolved weights/threshold for the row assembler.
        self._cfg_weights = cfg.weights
        self._cfg_threshold = cfg.defaults.quality_threshold
        geom = self._geom_from_rows(rows, prepared.ts)
        # optional flow/context channels (vectorised; default OFF per §8). The standardize pass they
        # need is computed ONLY when at least one is enabled — otherwise the streaming-engine rows
        # already carry zeros for both (the hot path stays free of the offline O(t) standardize).
        flow_on = bool(spec.optional_sources.get("flow", False))
        ctx_on = bool(spec.optional_sources.get("context", False))
        flow = ctx = None
        if flow_on or ctx_on:
            s = self._std_for_optional(prepared, cfg)
            flow = FlowEvidence(enabled=flow_on).evidence(prepared, spec, s) if flow_on else None
            ctx = ContextEvidence(enabled=ctx_on).evidence(prepared, context, spec) if ctx_on else None
        ev = self._ev_from_rows(rows, geom, flow=flow, context=ctx)
        return None, geom, ev

    def _geom_from_rows(self, rows, ts) -> BreakoutGeometry:
        n = len(rows)

        def col(attr, dtype=float):
            return np.array([getattr(r, attr) for r in rows], dtype=dtype) if n else np.zeros(0, dtype=dtype)

        return BreakoutGeometry(
            ts=np.asarray(ts),
            direction=col("direction", int),
            upper_barrier=col("upper_barrier"),
            lower_barrier=col("lower_barrier"),
            invalidation=col("invalidation"),
            stop_distance=col("stop_distance"),
            range_width_sigma=col("range_width_sigma"),
            compression_pct=col("compression_pct"),
            barrier_age=col("barrier_age", int),
            entry_ref_price=col("entry_ref_price"),
        )

    def _ev_from_rows(self, rows, geom, *, flow, context) -> EvidenceFrame:
        n = len(rows)

        def col(attr):
            return np.array([getattr(r, attr) for r in rows], dtype=float) if n else np.zeros(0)

        components = {
            "candidate": col("candidate"),
            "direction": col("candidate"),
            "range": col("range_e"),
            "compression": col("compression_e"),
            "trend": col("trend_e"),
            "flow": flow if flow is not None else col("flow_e"),
            "jump": col("jump_e"),
            "context": context if context is not None else col("context_e"),
            "false_break_risk": col("false_break_risk"),
        }
        calibrated = {"glr": col("glr_e"), "bocpd": col("bocpd_e")}
        return EvidenceBuilder(
            weights=self._cfg_weights, threshold=self._cfg_threshold
        ).build(components, calibrated, geom)

    # --- standardize used ONLY for the optional flow/context channels (default off) ---
    def _std_for_optional(self, prepared, cfg) -> StdSeries:
        d = cfg.defaults
        deseason = self._seasonal.fit_transform(prepared, lookback_sessions=20)
        return VolatilityStandardizer(
            n_vol_buckets=d.n_vol_buckets, n_liq_buckets=d.n_liq_buckets
        ).standardize(prepared, deseason, d.vol_window)

    def _prepared(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        bars = self._validator.validate(bars, spec)
        bars = self._sessionizer.apply(bars, spec)
        self._ts_cache = bars.ts
        return bars

    def _prepared_online(self, bar: Bars, spec: InstrumentSpec) -> Bars:
        # validate the single incoming bar; do NOT sessionize per-bar (a single bar would mark itself
        # as a session open). The streaming engine derives the session-open mark incrementally from
        # the timestamp grid, bit-identically to the batch Sessionizer (so Inv E parity holds).
        #
        # NB: we deliberately do NOT accumulate the full bar history here (the previous design's
        # O(n)-per-bar np.concatenate -> O(n^2) stream). Only a bounded 2-bar ts tail is kept (for
        # indexing the actionable bar) and the latest bar is passed to the assembler for its O(1)
        # tick/half-spread floor lookup. This keeps the whole update amortized O(1) (§1).
        bar = self._validator.validate(bar, spec)
        self._ts_tail.append(int(bar.ts[0]))
        self._latest_online_bar = bar
        return bar

    def _stream_bars(self, latest: Bars) -> Bars:
        """The latest online bar (for the assembler's O(1) tick/half-spread floor lookup)."""
        return latest

    @staticmethod
    def _bar_fields(bars: Bars, t: int):
        v = None if bars.v is None else float(bars.v[t])
        so = None if bars.is_session_open is None else bool(bars.is_session_open[t])
        iv = None if bars.is_valid is None else bool(bars.is_valid[t])
        rw = None if bars.is_roll_window is None else bool(bars.is_roll_window[t])
        return (
            int(bars.ts[t]), float(bars.o[t]), float(bars.h[t]), float(bars.l[t]),
            float(bars.c[t]), v, so, iv, rw,
        )
