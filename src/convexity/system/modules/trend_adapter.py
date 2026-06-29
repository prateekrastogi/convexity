"""TrendAdapter — wraps ``TrendEngine.update`` and injects ``tau_cap = H_sys`` (Invariant HC).

For each OPEN position, the adapter asks TREND for its action (continue / reduce / exit) under a
config whose operational maximum-hold cap is set to ``H_sys`` (Section 6.1, Invariant HC): no
realized holding period may exceed ``H_sys`` so the contractual cap is a true upper bound on every
label horizon, which is what the embargo is keyed to. No exit/trend-life math lives here (Invariant
P1) — the adapter only forces ``tau_cap = H_sys`` on the TREND config and reshapes the action into
the ``{continue, reduce, exit}`` literal SYSTEM's precedence reads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from convexity.core.protocols import TrendEngine
from convexity.core.types import Bars, Position

TrendActionLiteral = Literal["continue", "reduce", "exit"]


@dataclass(frozen=True)
class TrendAction:
    """The SYSTEM-side view of TREND's per-position action (SYSTEM §2 step B).

    ``kind`` ∈ {continue, reduce, exit}; ``alive`` is False on reduce/death (the demotion gate in
    Section 4 distinguishes a trend-alive incumbent from a dying one); ``target_fraction`` is the
    reduce target (1.0 on continue); ``time_cap`` flags a force-close at ``H_sys`` (Invariant HC).
    """

    kind: TrendActionLiteral
    alive: bool
    target_fraction: float
    time_cap: bool
    reason_codes: tuple[str, ...] = ()


class TrendAdapter:
    """Wrap a concrete (or mock) :class:`~convexity.core.protocols.TrendEngine` with ``tau_cap=H_sys``.

    ``update_for`` injects the contractual cap into the TREND config (Invariant HC), calls
    ``TrendEngine.update``, and maps the pillar-private ``ActionDecision`` into :class:`TrendAction`.
    The mapping reads only the boundary fields (action kind, alive flag, reduce fraction, time-cap
    cause); it never recomputes trend state.
    """

    def __init__(self, engine: TrendEngine, h_sys: int) -> None:
        self._engine = engine
        self._h_sys = h_sys

    def update_for(self, position: Position, bars: Bars, idx: int, cfg: object) -> TrendAction:
        """Run TREND with ``tau_cap == H_sys`` and map the action (Invariant HC)."""
        capped_cfg = self._with_tau_cap(cfg)
        decision = self._engine.update(position, bars, idx, capped_cfg)
        return self._to_action(decision)

    def _with_tau_cap(self, cfg: object) -> object:
        """Force ``tau_cap = H_sys`` so no realized hold exceeds the contractual cap (Invariant HC)."""
        try:
            return replace(cfg, tau_cap=self._h_sys)  # frozen dataclass cfg
        except TypeError:
            try:
                object.__setattr__(cfg, "tau_cap", self._h_sys)  # plain object
            except (AttributeError, TypeError):
                pass
            return cfg

    @staticmethod
    def _to_action(decision: object) -> TrendAction:
        """Reshape TREND's ``ActionDecision`` into the boundary :class:`TrendAction` (read-only)."""
        kind = str(getattr(decision, "action", getattr(decision, "kind", "continue")))
        if kind not in ("continue", "reduce", "exit"):
            kind = "continue"
        stop_cause = str(getattr(decision, "stop_cause", "") or "")
        time_cap = stop_cause == "time_cap"
        alive = kind == "continue"
        target_fraction = float(getattr(decision, "target_fraction", 1.0 if alive else 0.0))
        reason = tuple(getattr(decision, "reason_codes", ()) or ())
        return TrendAction(
            kind=kind,  # type: ignore[arg-type]
            alive=alive,
            target_fraction=target_fraction,
            time_cap=time_cap,
            reason_codes=reason,
        )
