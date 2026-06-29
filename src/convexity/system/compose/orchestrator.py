"""SystemOrchestrator.decide — the one public composition method (SYSTEM §2/§12, Invariant P1/CB).

Runs the deterministic precedence pipeline (Section 2) and the selection↔leverage fixed point
(Section 3) to produce ONE target book delta per bar. Pure function of F_t and BookState. Imports
the four modules only through ``modules/`` adapters; never reaches inside them (P1). Emits a full
reason chain per action. Deterministic: identical inputs ⇒ identical SystemDecision. Never places
raw orders — returns deltas for the Nautilus adapter to execute.

The §12 signature takes the already-resolved per-bar module outputs (so the orchestrator is
unit-testable against SCRIPTED module outputs / mock adapters, per the §15 composition gate):

    decide(book, sel, open_actions, candidates, cfg) -> SystemDecision

``candidates`` maps instrument → :class:`CandidateSignal` (the breakout entry + its RISK metadata:
per-position ceiling weight, lot weight-per-contract, ceiling slack, planned equity loss per lot).
The RISK metadata comes from the LEVERAGE adapter under the post-(B) book; the orchestrator never
recomputes it (Invariant P1). NO breakout score enters any decision (Invariant CB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.core.types import EntrySignal
from convexity.system.compose import precedence
from convexity.system.compose.budget import priority_order
from convexity.system.compose.incumbent import reconcile
from convexity.system.compose.iterate import FixedPointComposer
from convexity.system.compose.iterate import Position as FPPosition
from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import (
    BookState,
    ProposedPortfolio,
    SystemAction,
    SystemDecision,
)

SYSTEM_VERSION = "system-1.0.0"


@dataclass(frozen=True)
class CandidateSignal:
    """A new-capital candidate as the orchestrator sees it: the breakout entry + its RISK metadata.

    The RISK metadata (``ceiling_weight``, ``wpc``, ``ceiling_slack``, ``equity_loss_per_lot``) comes
    from the LEVERAGE adapter under the post-(B) book; the orchestrator does NOT recompute it
    (Invariant P1). The ``signal`` is read for ``direction`` only — never ``quality_score``
    (Invariant CB).
    """

    signal: EntrySignal
    ceiling_weight: float
    wpc: float
    ceiling_slack: float
    equity_loss_per_lot: float = 0.0


class SystemOrchestrator:
    """Compose the four pillar adapters into one target book delta (SYSTEM §12)."""

    def __init__(self, composer: FixedPointComposer | None = None) -> None:
        self._composer = composer or FixedPointComposer()

    def decide(
        self,
        book: BookState,
        sel: ProposedPortfolio,
        open_actions: Mapping[str, TrendAction],
        candidates: Mapping[str, CandidateSignal],
        cfg,
        *,
        collapsed: frozenset[str] = frozenset(),
        risk_improvements: Mapping[str, float] | None = None,
        round_trip_costs: Mapping[str, float] | None = None,
        bar_ts: int = 0,
    ) -> SystemDecision:
        """Run the §2 pipeline A→G + the §3 fixed point. Deterministic; emits deltas, not orders."""
        defaults = cfg.defaults
        reasons: list[str] = []

        # (A) GOVERNOR / KILL.
        gov = precedence.governor_kill(book)
        if gov.flat_all:
            actions = tuple(
                SystemAction(inst, "flat_all", 0, book.positions[inst].side, "governor",
                             "governor_flat_all", ("governor_flat_all",))
                for inst in sorted(book.positions)
            )
            return SystemDecision(
                bar_ts=bar_ts, actions=actions, iterations=0, converged=True,
                book_heat_used=0.0, equity_ladder_used={}, deferred=(),
                reason_codes=("governor_flat_all", "halt_new_opens"), system_version=SYSTEM_VERSION,
            )
        m = gov.multiplier
        reasons.append(gov.reason)

        # (B) OPEN POSITIONS FIRST — trend exit > ceiling collapse > reduce > continue.
        b_actions = precedence.open_position_actions(book.positions, open_actions, collapsed)
        # incumbents that survive (B): held or reduced (still on the book).
        surviving = {a.instrument: a for a in b_actions if a.kind in ("hold", "reduce")}

        # (C) RECONCILE SELECTION vs INCUMBENTS — the hold-incumbent firewall (Section 4).
        qualified = frozenset(candidates)
        incumbent_decisions = reconcile(
            {i: book.positions[i] for i in surviving},
            open_actions,
            sel,
            collapsed=collapsed,
            qualified=qualified,
            delta_rank=cfg.delta_rank,
            risk_improvements=risk_improvements,
            round_trip_costs=round_trip_costs,
        )
        demoted: set[str] = set()
        actions: list[SystemAction] = []
        for a in b_actions:
            dec = incumbent_decisions.get(a.instrument)
            if dec is None:  # closed in (B)
                actions.append(a)
                continue
            if dec.verdict in ("sel_demote_dead", "sel_rotate"):
                demoted.add(a.instrument)
                actions.append(
                    SystemAction(a.instrument, "close", 0, a.side, "selection", dec.reason,
                                 a.reason_chain + (dec.reason,))
                )
            else:  # hold (incumbent protected) or trend-driven action stands
                chain = a.reason_chain + ((dec.reason,) if dec.verdict == "hold" else ())
                actions.append(SystemAction(a.instrument, a.kind, a.target_units, a.side,
                                            a.driven_by, a.binding, chain))

        # (D) NEW-SIGNAL ADMISSION candidate set (selection-eligible AND qualified AND not-open).
        open_instruments = frozenset(surviving) - demoted
        slack = {i: c.ceiling_slack for i, c in candidates.items()}
        rweight = {i: sel.risk_weights.get(i, c.wpc) for i, c in candidates.items()}
        heat = {i: c.wpc for i, c in candidates.items()}
        cand_objs = precedence.new_signal_candidates(
            frozenset(sel.eligible), candidates, open_instruments, slack, rweight, heat
        )
        ordered = priority_order(cand_objs, budget_priority=cfg.budget_priority)

        # (E) ITERATE-TO-CONSISTENCY + (F) QUANTIZE + DUAL-CAP RECHECK (Section 3).
        fp_positions: list[FPPosition] = []
        # incumbents reserved first (P4): held lots from the post-(C) book.
        for inst, a in surviving.items():
            if inst in demoted:
                continue
            held = a.target_units if a.kind == "reduce" else book.positions[inst].units
            fp_positions.append(
                FPPosition(
                    instrument=inst,
                    wpc=max(book.positions[inst].held_risk_weight / max(book.positions[inst].units, 1), 1e-9),
                    ceiling_weight=book.positions[inst].held_risk_weight,
                    incumbent_lots=int(held),
                    target_weight=sel.risk_weights.get(inst, book.positions[inst].held_risk_weight),
                    priority=(0.0, 0.0, inst),  # incumbents seated before any new lot
                )
            )
        equity_loss_per_lot: dict[str, float] = {
            inst: candidates[inst].equity_loss_per_lot for inst in candidates
        }
        for rank_idx, c in enumerate(ordered):
            inst = c.instrument
            cs = candidates[inst]
            fp_positions.append(
                FPPosition(
                    instrument=inst,
                    wpc=max(cs.wpc * m, 1e-9),       # governor scales the lot heat (Section 3.2)
                    ceiling_weight=cs.ceiling_weight * m,  # governor scales the ceiling
                    incumbent_lots=0,
                    target_weight=rweight.get(inst, cs.wpc),
                    priority=(float(rank_idx), -c.mctr_efficiency, inst),
                )
            )

        equity_cap = book.equity * defaults.equity_ladder[-1]
        fp = self._composer.solve(
            fp_positions,
            h_book=cfg.h_book,
            equity_ladder_cap=equity_cap,
            eta=defaults.eta,
            r_max=defaults.r_max,
            tol=defaults.iterate_tol,
            equity_loss_per_lot=equity_loss_per_lot,
        )
        reasons.extend(fp.reason_codes)

        # (G) EMIT new-capital opens for admitted candidates (deadband-suppressed jitter excluded).
        for c in ordered:
            inst = c.instrument
            lots = fp.integer_book.get(inst, 0)
            if lots > 0:
                side = candidates[inst].signal.direction
                actions.append(
                    SystemAction(inst, "open", int(lots), side, "breakout", "new_signal_admission",
                                 ("breakout_qualified", "budget_admitted"))
                )

        equity_ladder_used = {
            inst: lots * equity_loss_per_lot.get(inst, 0.0) for inst, lots in fp.integer_book.items()
        }
        return SystemDecision(
            bar_ts=bar_ts,
            actions=tuple(actions),
            iterations=fp.iterations,
            converged=fp.converged,
            book_heat_used=fp.heat_used,
            equity_ladder_used=equity_ladder_used,
            deferred=fp.deferred,
            reason_codes=tuple(reasons),
            system_version=SYSTEM_VERSION,
        )
