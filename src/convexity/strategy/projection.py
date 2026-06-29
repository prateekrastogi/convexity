"""Pure-Python projections the live ``_ConvexityStrategy`` SEAM feeds the composer (Invariant N).

NautilusTrader-FREE by construction — every function here takes plain primitives (the numbers the
strategy reads off Nautilus's authoritative ``Cache`` / ``Portfolio`` / ``Bar``) and returns one of
the shared ``core``/``execution``/``system`` boundary objects the :class:`ConvexityComposer`
consumes. Splitting the projection out of ``strategy/convexity_strategy.make_strategy_class`` keeps
it unit-testable in the pillar venv (no nautilus) and keeps the nautilus seam a thin event router:
the strategy NEVER tracks position/account state itself (Inv N) — it reads Nautilus, hands the
numbers here, and submits the composer's plan.

The pieces:
  * :class:`RollingBars` — accumulates the engine's bar stream into a point-in-time
    :class:`~convexity.core.types.Bars` panel BREAKOUT/TREND read (append-only, F_t-measurable).
  * :func:`micro_from_bar` — derives a causal :class:`~convexity.execution.types.MicroState` from a
    closed bar (the 1-min kline feed has no book, so spread/sigma fall through conservative AMBER
    defaults; the live OFI-aware microstate is a data-layer upgrade, not a composition change).
  * :func:`book_state_from_snapshot` / :func:`open_views_from_snapshot` — map the plain
    position/account snapshot the strategy reads off Nautilus into :class:`BookState` +
    :class:`OpenPositionView` (the composer's open-position projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from convexity.core.types import Bars, InstrumentSpec, Position
from convexity.execution.microstate.fill_prob import logistic_fill_curve
from convexity.execution.types import MicroState, PositionState, VenueStop
from convexity.strategy.convexity_strategy import OpenPositionView, intent_id_for
from convexity.system.types import BookState, OpenPosition

__all__ = [
    "RollingBars",
    "micro_from_bar",
    "book_state_from_snapshot",
    "open_views_from_snapshot",
    "PositionSnapshot",
]


@dataclass
class RollingBars:
    """Append-only OHLCV accumulator → a point-in-time :class:`Bars` panel (one per instrument).

    The strategy pushes each closed bar's (ts, o, h, l, c, v) as it arrives; :meth:`as_bars` builds
    the immutable numpy panel the data-consuming pillars read. Bounded by ``maxlen`` so a long live
    run does not grow without limit (the detectors only need a rolling lookback window).

    CROSS-MODE GAP MARKER (FRICTION #2). Each bar carries a ``gap_before`` flag — True when this bar
    is the first bar after a session gap / discontinuity (a missing bar, an overnight session
    boundary, a reconnect). This is the SINGLE per-bar gap signal the data-consuming pillars consume
    as :attr:`Bars.is_session_open`: BREAKOUT suppresses the cross on a gap bar (a session-open jump
    is not a continuous breakout, ``detect/candidates._suppress_artifacts``) and TREND applies its
    once-per-boundary ``gap_var`` there (``estimate/volatility``). The flag is derived IDENTICALLY in
    every mode so the pillars' gap handling behaves the same regardless of venue:

      * live / paper: a bar-timestamp DISCONTINUITY — ``ts - prev_ts`` materially beyond the expected
        bar interval (a reconnect or a missing/late bar) — exactly the cadence the OPS §7.6 heartbeat
        also reads. Auto-derived here when ``push(gap_before=None)`` (the live ``on_bar`` path).
      * backtest: the stitched lake's authoritative ``gap_before`` column threaded explicitly through
        ``run/data_loader.load_bars`` → ``push(gap_before=...)``. This covers the rare stitch boundary
        a 24/7 (crypto) feed marks WITHOUT leaving a timestamp hole, which pure ts-discontinuity would
        miss — but the explicit flag is OR-ed with the same ts-discontinuity test so the two paths
        agree whenever a hole IS present.

    ``bar_interval_ms`` is the expected cadence (1-min klines ⇒ 60_000); a gap is declared at
    > 1.5× it (one fully missed bar), matching the strategy's heartbeat-staleness window.
    """

    maxlen: int = 4096
    bar_interval_ms: int = 60_000
    _ts: list[int] = field(default_factory=list)
    _o: list[float] = field(default_factory=list)
    _h: list[float] = field(default_factory=list)
    _l: list[float] = field(default_factory=list)
    _c: list[float] = field(default_factory=list)
    _v: list[float] = field(default_factory=list)
    _gap: list[bool] = field(default_factory=list)

    def push(
        self,
        ts: int,
        o: float,
        h: float,
        low: float,
        c: float,
        v: float,
        *,
        gap_before: bool | None = None,
    ) -> None:
        """Append one closed bar, recording its gap-before flag (Bars.is_session_open).

        ``gap_before`` is the per-bar gap signal: pass the explicit lake column in the backtest; leave
        it ``None`` in live/paper to auto-derive from the bar-timestamp discontinuity. When an explicit
        flag is given it is OR-ed with the ts-discontinuity test so a real timestamp hole is never
        silently dropped just because the upstream column did not mark it. The very first bar of a run
        is never a gap bar (there is no prior bar to gap from).
        """
        ts_i = int(ts)
        discontinuous = bool(self._ts) and (ts_i - self._ts[-1]) > int(self.bar_interval_ms * 1.5)
        is_gap = (bool(gap_before) or discontinuous) if self._ts else False
        self._ts.append(ts_i)
        self._o.append(float(o))
        self._h.append(float(h))
        self._l.append(float(low))
        self._c.append(float(c))
        self._v.append(float(v))
        self._gap.append(is_gap)
        if len(self._ts) > self.maxlen:
            for buf in (self._ts, self._o, self._h, self._l, self._c, self._v, self._gap):
                del buf[0]

    def __len__(self) -> int:
        return len(self._ts)

    def as_bars(self) -> Bars:
        return Bars(
            ts=np.asarray(self._ts, dtype=np.int64),
            o=np.asarray(self._o, dtype=float),
            h=np.asarray(self._h, dtype=float),
            l=np.asarray(self._l, dtype=float),
            c=np.asarray(self._c, dtype=float),
            v=np.asarray(self._v, dtype=float),
            is_session_open=np.asarray(self._gap, dtype=bool),
        )


def micro_from_bar(
    spec: InstrumentSpec,
    *,
    mid: float,
    bar_range: float,
    volume: float,
    half_spread_bps: float = 1.0,
    sigma_horizon_bps: float = 5.0,
    predicted_funding_rate: float = 0.0,
    secs_to_next_funding: float = float("inf"),
    halted: bool = False,
    stale: bool = False,
    cold_start: bool = True,
) -> MicroState:
    """A causal :class:`MicroState` from a closed bar (EXECUTION §10; every field ≤ the bar, P7).

    A 1-min kline feed has NO resting book and NO passive-fill history, so ``half_spread_bps`` /
    ``sigma_horizon_bps`` fall through to conservative AMBER defaults and — critically —
    ``cold_start`` defaults TRUE: with no own-passive-fill record the maker floor is GATED OFF
    (EXECUTION §5.5 #2), so EXECUTION crosses the book (the marketable/taker branch) rather than
    fabricating a passive edge a kline feed cannot support. ``bar_range`` (high−low) feeds the cost
    stub's adverse-extreme slippage budget via EXECUTION. A book-/OFI-aware data layer is a DATA
    upgrade that passes ``cold_start=False`` here without touching the composition (the strategy
    reads the microstate; the composer is unchanged).
    """
    tick = float(spec.tick_size)
    safe_mid = mid if mid > 0.0 else 1.0
    tick_bps = 10_000.0 * tick / safe_mid
    return MicroState(
        mid=mid,
        half_spread_bps=half_spread_bps,
        spread_rung="quoted",
        sigma_horizon_bps=sigma_horizon_bps,
        volume_proxy=max(volume, 1.0),
        drift_g=0.0,
        tick_bps=tick_bps,
        fill_prob_curve=logistic_fill_curve(
            base_rate=0.8, sigma_bps=sigma_horizon_bps, tick_bps=tick_bps
        ),
        funding_aware=spec.funding_interval_ms is not None,
        predicted_funding_rate=predicted_funding_rate,
        secs_to_next_funding=secs_to_next_funding,
        halted=halted,
        stale=stale,
        rate_limit_ok=True,
        fill_prob_rung="class_prior" if cold_start else "bucket",
        fill_prob_ci_halfwidth=0.02,
        fill_prob_calib_error=0.0,
        cold_start=cold_start,
    )


@dataclass(frozen=True)
class PositionSnapshot:
    """The plain numbers the strategy reads off ONE open Nautilus position (Inv N — Nautilus owns it).

    ``side`` ∈ {+1,-1}; ``qty`` is the signed-magnitude open UNITS; ``entry_idx`` is the bar index at
    which the position opened (recovered from the intent id on restart); ``invalidation`` +
    ``generation`` version the protective-stop id; ``venue_stops`` is the SET of resting reduce-only
    stops read back for reconciliation. ``trend_state`` is the SYSTEM book-state tag.
    """

    instrument: str
    side: int
    qty: float
    entry_price: float
    entry_idx: int
    invalidation: float
    generation: int
    held_risk_weight: float = 0.05
    trend_state: str = "alive"
    venue_stops: tuple[VenueStop, ...] = ()


def open_views_from_snapshot(
    snapshots: Sequence[PositionSnapshot],
) -> dict[str, OpenPositionView]:
    """Project the open Nautilus positions onto the composer's :class:`OpenPositionView` map (Inv N)."""
    views: dict[str, OpenPositionView] = {}
    for s in snapshots:
        iid = intent_id_for(s.instrument, s.entry_idx)
        views[s.instrument] = OpenPositionView(
            instrument=s.instrument,
            intent_id=iid,
            state=PositionState(
                instrument=s.instrument,
                side=s.side,  # type: ignore[arg-type]
                qty=abs(s.qty),
                entry_price=s.entry_price,
            ),
            trend_position=Position(
                side=s.side,  # type: ignore[arg-type]
                entry_idx=s.entry_idx,
                entry_price=s.entry_price,
                initial_invalidation=s.invalidation,
            ),
            invalidation=s.invalidation,
            invalidation_generation=s.generation,
            held_risk_weight=s.held_risk_weight,
            entry_idx=s.entry_idx,
            venue_stops=s.venue_stops,
        )
    return views


def book_state_from_snapshot(
    *,
    equity: float,
    peak_equity: float,
    free_margin: float,
    snapshots: Sequence[PositionSnapshot],
    specs: Mapping[str, InstrumentSpec],
    governor_multiplier: float = 1.0,
    cov: np.ndarray | None = None,
) -> BookState:
    """Build the canonical :class:`BookState` from the account + open-position snapshot (ARCH §5.9).

    ``units`` on each :class:`OpenPosition` is SYSTEM's INTEGER LOT count — the inverse of the
    lots→units guard: the strategy reads raw UNITS off Nautilus and divides by the contract
    multiplier (``lot_size``) to recover the lot count SYSTEM reasons in (round to nearest; a partial
    fill is floored by SYSTEM's round-DOWN elsewhere). Equity/peak/free-margin/governor come from the
    LEVERAGE-owned reference (§5.12); SYSTEM never recomputes them.
    """
    positions: dict[str, OpenPosition] = {}
    for s in snapshots:
        spec = specs[s.instrument]
        mult = float(spec.lot_size)
        lots = int(round(abs(s.qty) / mult)) if mult > 0.0 else int(abs(s.qty))
        positions[s.instrument] = OpenPosition(
            instrument=s.instrument,
            side=s.side,  # type: ignore[arg-type]
            units=lots,
            entry_price=s.entry_price,
            held_risk_weight=s.held_risk_weight,
            trend_state=s.trend_state,
        )
    n = max(len(positions), 1)
    cov = cov if cov is not None else np.eye(n)
    return BookState(
        equity=float(equity),
        peak_equity=float(peak_equity),
        free_margin=float(free_margin),
        positions=positions,
        cov=np.asarray(cov, dtype=float),
        governor_multiplier=float(governor_multiplier),
    )
