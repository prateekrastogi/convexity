"""AdverseSelection — A(δ, g), the maker's short-option cost, signed bps vs arrival mid (§5.1 / §11).

The expected adverse mid-move CONDITIONAL on a passive fill at ``offset_ticks`` — you are filled
precisely when informed flow trades through you (Glosten-Milgrom). Conditions on the shared driver
``g`` (§5.1.1): LARGE when ``g`` is high (a live breakout — fills cluster on the wrong side of
continuation), ~0 for PATIENT/calm. Monotone NON-DECREASING in ``g`` and in ``offset_ticks`` (a
deeper fill only triggers after a LARGER adverse move has reached it). Shares ``g`` with p_fill and
DelayCost so the maker branch cannot assemble an incoherent joint (Invariant J).

ONLINE-CAUSAL UPDATE (Invariant L): the realized-drift magnitude is fitted online from the
instrument's OWN post-fill outcomes under the same point-in-time / embargo rule as p_fill — a fill's
realized adverse drift is admitted ONLY after its horizon has fully elapsed, so the live magnitude at
bar t never includes a fill still being scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import ExecTarget, MicroState

__all__ = ["AdverseFillRecord", "AdverseSelection"]


@dataclass(frozen=True)
class AdverseFillRecord:
    """One realized post-fill adverse-drift outcome (bps), embargoed by ``outcome_close_ts`` (Inv L)."""

    regime_bucket: str
    realized_adverse_bps: float
    outcome_close_ts: int


@dataclass
class AdverseSelection:
    """A(δ, g) — fitted post-fill adverse drift, shared-``g`` and online-embargoed (Inv J/L)."""

    records: list[AdverseFillRecord] = field(default_factory=list)

    def admit(self, record: AdverseFillRecord) -> None:
        """Append a realized adverse-drift outcome; used only when embargoed (see :meth:`magnitude`)."""
        self.records.append(record)

    def magnitude(self, regime_bucket: str, micro: MicroState, as_of_ts: int) -> float:
        """The fitted base adverse-drift magnitude in bps (a fraction of σ_h when un-fitted, §5.1).

        Online point-in-time (Invariant L): averages only records whose outcome horizon closed at or
        before ``as_of_ts``. With no resolved own fills it falls back to a conservative fraction of
        the short-horizon vol — a finite positive default, never zero.
        """
        resolved = [
            r for r in self.records
            if r.outcome_close_ts <= as_of_ts and r.regime_bucket == regime_bucket
        ]
        if resolved:
            return float(sum(abs(r.realized_adverse_bps) for r in resolved) / len(resolved))
        return 0.5 * micro.sigma_horizon_bps  # conservative un-fitted base (§5.1)

    def cost_bps(
        self,
        target: ExecTarget,
        micro: MicroState,
        offset_ticks: float,
        regime_bucket: str,
        g: float,
        cfg: ResolvedConfig,
        *,
        as_of_ts: int = 0,
    ) -> float:
        """``A(δ, g)`` in bps — NON-decreasing in ``g`` and in ``offset_ticks`` (§5.1 / Invariant J).

        ``A = base · (1 + g) · (1 + offset_bps/σ_h)``: the ``(1+g)`` factor makes it rise with the
        shared continuation drift (large on a live breakout, ~0 when g~0 since base ~ 0.5·σ); the
        offset factor makes a deeper fill (which only triggers after a larger adverse move) cost more.
        """
        base = self.magnitude(regime_bucket, micro, as_of_ts)
        g_factor = 1.0 + max(g, 0.0)
        offset_bps = max(offset_ticks, 0.0) * micro.tick_bps
        sig = max(micro.sigma_horizon_bps, 1e-6)
        offset_factor = 1.0 + offset_bps / sig
        return float(base * g_factor * offset_factor)
