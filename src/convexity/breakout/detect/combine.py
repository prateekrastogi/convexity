"""SignalAssembler.fire (Invariant A: the ONLY EntrySignal emitter) + BreakoutModule.propose.

This is the ONLY place ``EntrySignal`` objects are born (Invariant A). Feature/detector modules
return arrays or an :class:`EvidenceFrame`; only :meth:`SignalAssembler.fire` constructs an entry.
The firing rule is fixed by ``cfg`` (BREAKOUT §detect/combine): quality threshold, false-break cap,
min/max stop distance, and the entry-timestamp convention. Under the bar-close convention the signal
is known at the close of bar t (``signal_idx=t``) and the first actionable fill is the next bar
(``entry_idx=t+1``, Invariant G: confirmation delay moves the entry).

Every emitted signal carries ``urgency='urgent'`` (breakout entries are adversarial, EXECUTION needs
this) and the full evidence decomposition + reason codes (Invariant H). ``quality_score`` is
BREAKOUT-internal for its own gating only; downstream reads only direction / status=='entry' /
urgency / price levels (Invariant K). :class:`BreakoutModule` is the thin outward adapter SYSTEM
wraps: ``propose`` returns at most one currently-actionable entry, or None.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, EntrySignal, InstrumentSpec, Position

from convexity.breakout.config.resolve import ResolvedConfig
from convexity.breakout.types import BreakoutGeometry, EvidenceFrame


class SignalAssembler:
    """The sole EntrySignal emitter (Invariant A). One public method, ``fire``."""

    def fire(
        self,
        symbol: str,
        bars: Bars,
        spec: InstrumentSpec,
        geom: BreakoutGeometry,
        ev: EvidenceFrame,
        cfg: ResolvedConfig,
    ) -> list[EntrySignal]:
        """Emit final EntrySignal objects (the ONLY place they are constructed, Inv A).

        Fires at every bar that is a candidate, clears the quality threshold, sits under the
        false-break cap, and has a stop distance within [floor, ceiling] (§9). ``entry_idx`` is
        ``signal_idx + 1`` under the bar-close convention (Inv G). The last bar cannot fire (no
        actionable next bar).
        """
        d = cfg.defaults
        n = len(bars)
        out: list[EntrySignal] = []

        # noise floor for the stop distance: a multiple of (tick + half-spread) in price units.
        half_spread = self._half_spread(bars, spec)
        stop_floor = d.min_stop_tick_mult * (spec.tick_size + half_spread)

        for t in range(n - 1):  # last bar has no actionable next bar
            direction = int(ev.candidate[t])
            if direction == 0:
                continue
            q = float(ev.quality_score[t])
            thr = float(ev.threshold[t])
            if q < thr:
                continue
            fbr = float(ev.false_break_risk[t])
            if fbr > d.false_break_cap:
                continue
            stop_dist = float(geom.stop_distance[t])
            if not np.isfinite(stop_dist) or stop_dist < stop_floor:
                continue
            # R-budget ceiling (§9): reject when the stop is so wide relative to the predictive
            # one-bar scale that trend convexity is gone. The unit is causal-sigma in PRICE terms;
            # the stop spans the pre-break range, so the ceiling is a generous multiple of sigma.
            sigma_price = self._sigma_price_at(geom, t)
            if sigma_price > 0 and stop_dist > d.max_stop_sigma_mult * sigma_price:
                continue

            out.append(self._make_signal(symbol, t, direction, geom, ev, cfg))
        return out

    def fire_at(
        self,
        symbol: str,
        bars: Bars,
        spec: InstrumentSpec,
        geom: BreakoutGeometry,
        ev: EvidenceFrame,
        cfg: ResolvedConfig,
        t: int,
    ) -> EntrySignal | None:
        """Evaluate the SINGLE bar ``t`` against the firing rule (the O(1) online hot-path emitter).

        Identical accept/reject logic to :meth:`fire`'s per-bar body, factored out so the online
        ``update`` can fire just the newly-actionable bar without re-scanning history (Inv E parity:
        the rule is bar-local, so a single-bar evaluation equals the batch loop's decision at t).

        ``bars`` here supplies only the tick/half-spread floor lookup (the latest online bar); the
        index ``t`` addresses the evidence/geometry tail, NOT ``bars`` — the batch loop's "last bar
        cannot fire" guard is enforced by the caller (the online actionable bar is never the last)."""
        if t < 0 or t >= len(ev):
            return None
        d = cfg.defaults
        direction = int(ev.candidate[t])
        if direction == 0:
            return None
        if float(ev.quality_score[t]) < float(ev.threshold[t]):
            return None
        if float(ev.false_break_risk[t]) > d.false_break_cap:
            return None
        half_spread = self._half_spread(bars, spec)
        stop_floor = d.min_stop_tick_mult * (spec.tick_size + half_spread)
        stop_dist = float(geom.stop_distance[t])
        if not np.isfinite(stop_dist) or stop_dist < stop_floor:
            return None
        sigma_price = self._sigma_price_at(geom, t)
        if sigma_price > 0 and stop_dist > d.max_stop_sigma_mult * sigma_price:
            return None
        return self._make_signal(symbol, t, direction, geom, ev, cfg)

    def _make_signal(
        self,
        symbol: str,
        t: int,
        direction: int,
        geom: BreakoutGeometry,
        ev: EvidenceFrame,
        cfg: ResolvedConfig,
    ) -> EntrySignal:
        barrier = float(geom.upper_barrier[t] if direction > 0 else geom.lower_barrier[t])
        evidence = {
            "glr_e": float(ev.glr_e[t]),
            "bocpd_e": float(ev.bocpd_e[t]),
            "range_e": float(ev.range_e[t]),
            "compression_e": float(ev.compression_e[t]),
            "trend_e": float(ev.trend_e[t]),
            "flow_e": float(ev.flow_e[t]),
            "jump_e": float(ev.jump_e[t]),
            "context_e": float(ev.context_e[t]),
        }
        reasons = tuple(c for c in ev.reason_codes if evidence.get(f"{c}_e", 0.0) > 0.0) or ("cross",)
        return EntrySignal(
            symbol=symbol,
            signal_idx=t,
            entry_idx=t + 1,  # bar-close convention: first actionable fill is next bar (Inv G)
            direction=direction,  # type: ignore[arg-type]
            status="entry",
            quality_score=float(ev.quality_score[t]),
            threshold=float(ev.threshold[t]),
            entry_ref_price=float(geom.entry_ref_price[t]),
            invalidation=float(geom.invalidation[t]),
            stop_distance=float(geom.stop_distance[t]),
            barrier_price=barrier,
            range_width_sigma=float(geom.range_width_sigma[t]),
            compression_pct=float(geom.compression_pct[t]),
            false_break_risk=float(ev.false_break_risk[t]),
            urgency="urgent",  # breakout entries are adversarial (EXECUTION needs URGENT)
            evidence=evidence,
            reason_codes=reasons,
            detector_version=cfg.defaults.detector_version,
        )

    @staticmethod
    def _half_spread(bars: Bars, spec: InstrumentSpec) -> float:
        if bars.bid is not None and bars.ask is not None:
            sp = np.nanmedian((bars.ask - bars.bid) / 2.0)
            if np.isfinite(sp) and sp > 0:
                return float(sp)
        return float(spec.tick_size)

    @staticmethod
    def _sigma_price_at(geom: BreakoutGeometry, t: int) -> float:
        """Recover the PRICE-units predictive sigma at t from the geometry, so the R-budget ceiling
        (§9) is in the same units as ``stop_distance``.

        ``range_width_sigma[t] = (upper-lower)/(sigma_price)`` by construction in range_geometry, so
        ``sigma_price = (upper-lower)/range_width_sigma``. Returns 0.0 (ceiling disabled) when the
        range is undefined.
        """
        rws = geom.range_width_sigma[t]
        u, low = geom.upper_barrier[t], geom.lower_barrier[t]
        if np.isfinite(rws) and rws > 0 and np.isfinite(u) and np.isfinite(low):
            return float((u - low) / rws)
        return 0.0


def position_seed(signal: EntrySignal) -> Position:
    """Seed a TREND :class:`Position` from a fired entry (BREAKOUT entry gate -> TREND, Inv O).

    BREAKOUT owns ``initial_invalidation`` / ``initial_risk``; TREND consumes them read-only and never
    recomputes entry. The fill price is the next-bar actionable reference; here we seed it with the
    signal's entry_ref_price (the strategy adapter overwrites it with the real fill).
    """
    return Position(
        side=signal.direction,  # type: ignore[arg-type]
        entry_idx=signal.entry_idx,
        entry_price=signal.entry_ref_price,
        initial_invalidation=signal.invalidation,
        initial_risk=signal.stop_distance,
    )


class BreakoutModule:
    """The outward composition surface SYSTEM wraps (BREAKOUT §detect/combine public adapter).

    Wraps the pipeline + SignalAssembler. ``propose`` returns at most one currently-actionable
    highest-quality un-deduped 'entry' at the LATEST bar, or None. SYSTEM's breakout_adapter reads
    ONLY direction / status=='entry' / urgency / price levels (Invariant K). Returning a list is an
    internal detail of ``fire``; the composition contract is single-or-None per instrument per bar.
    """

    def __init__(self, pipeline, cfg: ResolvedConfig, bars: Bars) -> None:
        # pipeline is a PipelineOrchestrator; injected to avoid an import cycle.
        self._pipeline = pipeline
        self._cfg = cfg
        self._bars = bars

    def propose(self, instrument: InstrumentSpec) -> EntrySignal | None:
        """Return the single currently-actionable entry for this instrument, else None (§adapter)."""
        signals = self._pipeline.run_batch(self._bars, instrument, self._cfg)
        if not signals:
            return None
        latest_signal_idx = max(s.signal_idx for s in signals)
        latest = [s for s in signals if s.signal_idx == latest_signal_idx and s.status == "entry"]
        if not latest:
            return None
        # highest-quality among ties at the latest bar (internal gating use of the score only).
        return max(latest, key=lambda s: s.quality_score)
