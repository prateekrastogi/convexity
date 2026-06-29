"""DelayCost — D(δ, g), the no-fill-conditional adverse drift, signed bps vs arrival mid (§5.1.2 / §11).

The expected adverse drift of the arrival mid over ``horizon_s`` GIVEN a passive order at
``offset_ticks`` does NOT fill — the breakout runs away. It MUST take ``offset_ticks`` (§11): the
no-fill event and its conditional drift DEPEND on the posted offset (a deeper post is less likely to
fill, and it fails to fill precisely when the market is running, so its no-fill-conditional drift
differs from a shallow post's). Without this, ``δ* = argmin_δ E[cost_maker(δ)]`` would optimize only
half the δ-dependence (the fill branch via PI and A) and miss the no-fill branch's, so the argmin
would be inconsistent.

Conditions on the shared regime driver ``g`` (§5.1.1) so ``D`` rises with the same latent
continuation drift that lowers p_fill and raises ``A`` (Invariant J). Increasing in ``σ_horizon`` and
in ``g``; ~0 for PATIENT/calm. Uses MEASURABLE drift ``g`` (and the signal's urgency as a scale),
NEVER an edge / return forecast (Invariant A).
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import ExecTarget, MicroState

__all__ = ["DelayCost"]

# §6 urgency -> drift-scale mapping (RED-ish; urgency is the upstream INPUT, not an edge forecast).
_URGENCY_SCALE: dict[str, float] = {
    "PATIENT": 0.25,
    "NORMAL": 0.5,
    "URGENT": 1.0,
    "IMMEDIATE": 1.0,
}


class DelayCost:
    """D(δ, g) — the no-fill-conditional drift cost, shared ``g``, offset-dependent (§5.1.2 / Inv J)."""

    def cost_bps(
        self,
        target: ExecTarget,
        micro: MicroState,
        offset_ticks: float,
        horizon_s: float,
        g: float,
        cfg: ResolvedConfig,
    ) -> float:
        """``D(δ, g)`` in bps — NON-decreasing in ``g`` and in ``offset_ticks`` (§5.1.2 / Invariant J).

        ``D = σ_h · g · urgency_scale · (1 + offset_bps/σ_h)``: the ``g`` factor makes the no-fill
        drift scale with the same continuation drift that lowers p_fill (so it is ~0 when g~0, large
        on a runaway); the offset factor makes a deeper post (which fails to fill precisely when the
        market runs) carry a larger no-fill-conditional drift; urgency (an INPUT) scales it.
        """
        scale = _URGENCY_SCALE.get(str(target.urgency), 0.5)
        offset_bps = max(offset_ticks, 0.0) * micro.tick_bps
        sig = max(micro.sigma_horizon_bps, 1e-6)
        offset_factor = 1.0 + offset_bps / sig
        return float(micro.sigma_horizon_bps * max(g, 0.0) * scale * offset_factor)
