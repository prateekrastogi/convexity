"""Staged rollout with quantitative go/no-go gates (§17). New filename per ARCHITECTURE §5.13.

Spec: OPS §17. OPS owns the staged rollout because each stage gate is a STATISTICAL test of
live-vs-validated — the same §5 reconciliation streams, now used as promotion gates with a minimum
sample requirement. The book moves to the next stage only when the realized streams sit WITHIN the CI
of the validated distribution over a minimum sample, and demotes immediately on a kill.

  paper  → canary → scaled → full

§17.1: each gate is a statistical test, not a calendar. A stream CANNOT be a promotion gate until it
has cleared BOTH its warm-up ``N_warm`` (§6.8) AND its ``N_stage`` minimum sample — a stream still
warming up reports ``warming_up`` and DEFERS the advance exactly as an insufficient sample does. The
advance objective contains NO raw-P&L-milestone term (Invariant ROLLOUT). §17.2: a kill demotes
immediately (the same {degrade, halt} machinery); re-promotion re-runs the gate from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stage = Literal["paper", "canary", "scaled", "full"]
_ORDER: tuple[Stage, ...] = ("paper", "canary", "scaled", "full")


@dataclass(frozen=True)
class StageGateInput:
    """The realized live-vs-validated evidence for a stage gate (§17.1). NO raw-P&L term.

    ``n_trades`` is the realized sample; ``edge_in_ci``/``fill_in_ci``/``tca_in_ci``/``vol_in_ci`` are
    the two-sided CI checks against the validated distribution; ``sprt_accepts_h0`` is the §6.5 non-
    inferiority gate (live ``p ≥ p0``); ``recon_clean``/``no_kill_fired`` are the binary in-control
    flags; ``all_gates_warmed`` is True only when every gating detector is past its ``N_warm`` (§6.8).
    ``e_cost_cusum_breached`` blocks advance AND triggers a demote (Invariant XC, §17.1).
    """

    n_trades: int
    edge_in_ci: bool
    fill_in_ci: bool
    tca_in_ci: bool
    vol_in_ci: bool
    sprt_accepts_h0: bool
    recon_clean: bool
    no_kill_fired: bool
    all_gates_warmed: bool
    e_cost_cusum_breached: bool


@dataclass(frozen=True)
class StageGateResult:
    """The gate verdict (§17.1). ``advance`` only when ALL checks hold over the min sample; otherwise
    ``defer`` (insufficient sample / still warming up) or ``demote`` (a kill / cost breach)."""

    decision: Literal["advance", "defer", "demote", "hold"]
    reason: str


def stage_min_sample(stage: Stage, *, n_paper: int, n_canary: int, n_scaled: int) -> int:
    """The ``N_stage`` minimum realized-trade sample to ADVANCE out of a stage (§17.1, RED)."""
    return {"paper": n_paper, "canary": n_canary, "scaled": n_scaled, "full": 0}[stage]


def evaluate_gate(stage: Stage, ev: StageGateInput, *, n_paper: int, n_canary: int,
                  n_scaled: int) -> StageGateResult:
    """Evaluate one stage gate (§17.1). Statistical, never a calendar, never raw P&L (Invariant
    ROLLOUT).

    Demote first on a kill or an ``e_cost`` CUSUM breach (§17.1); else defer while warming up or below
    the minimum sample; else advance only when every live-vs-validated check holds.
    """
    if not ev.no_kill_fired:
        return StageGateResult("demote", "a kill fired at this stage (§17.2) — demote immediately")
    if ev.e_cost_cusum_breached:
        return StageGateResult("demote", "e_cost CUSUM breach blocks advance and demotes (XC/§17.1)")
    if not ev.all_gates_warmed:
        return StageGateResult("defer", "a gating detector is still in warm-up (warming_up, §6.8)")
    need = stage_min_sample(stage, n_paper=n_paper, n_canary=n_canary, n_scaled=n_scaled)
    if ev.n_trades < need:
        return StageGateResult("defer", f"insufficient_stage_sample ({ev.n_trades} < {need})")
    checks = (ev.edge_in_ci, ev.fill_in_ci, ev.tca_in_ci, ev.vol_in_ci,
              ev.sprt_accepts_h0, ev.recon_clean)
    if all(checks):
        return StageGateResult("advance", "all streams within validated CI over the min sample (§17.1)")
    return StageGateResult("hold", "a live-vs-validated check is not within the validated CI (§17.1)")


def next_stage(stage: Stage) -> Stage:
    """The stage one step up (``full`` is terminal)."""
    i = _ORDER.index(stage)
    return _ORDER[min(i + 1, len(_ORDER) - 1)]


def demote_stage(stage: Stage) -> Stage:
    """Demote one stage on a kill (§17.2; ``paper`` is the floor)."""
    i = _ORDER.index(stage)
    return _ORDER[max(i - 1, 0)]


__all__ = [
    "Stage",
    "StageGateInput",
    "StageGateResult",
    "stage_min_sample",
    "evaluate_gate",
    "next_stage",
    "demote_stage",
]
