"""The selection↔leverage fixed point Φ + monotone greedy lot fill (SYSTEM §3, Invariant IT).

This is the only genuine two-way coupling in the stack. A one-shot is wrong: adding position j
raises every other position's MCTR, lowering their ceilings, which can make some proposed positions
infeasible, which changes the covariance again. The consistent book is a fixed point of
``w* = Φ(w*)`` where Φ takes a candidate weight vector to the LeverageEngine-feasible weight vector
under the SHARED book-heat budget and the dual equity cap.

We do NOT claim the INTEGER iterate converges (lot rounding DOWN is discontinuous and a naive loop
can oscillate). Instead, Section 3.3:

  1. The CONTINUOUS relaxation converges: ``w_tilde^(r+1) = Π_F((1-η)·w^(r) + η·LeverageFeasible(w^(r)))``,
     a damped averaged projection onto a convex feasible set, firmly non-expansive (Krasnoselskii–
     Mann). Run ≤ r_max damped steps; the continuous limit only DEFINES the rounding target.
  2. The integer book is then resolved by a SINGLE monotone greedy fill (NOT a loop): rank
     candidates by §3.4 priority, greedily admit lots in priority order each time re-checking H_book
     and the equity ladder against the partially-built book, admit a lot only if BOTH still hold.
     Lots are only added, never revoked, so the A-drops/B-fits oscillation CANNOT occur and the
     procedure terminates by construction.

``converged`` is defined honestly (Section 3.3): the continuous relaxation settled within r_max
(``‖w^(r) − w^(r-1)‖ ≤ tol``). On non-convergence we still emit the greedy-filled FEASIBLE book with
``converged=False`` and reason ``iterate_fallback`` — never an infeasible book, never an unbounded
loop. No sizing/exit math here (Invariant P1): the per-position ceilings come from the LEVERAGE
adapter; this module only does the projection + the budget knapsack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Position:
    """A position in the fixed point: its lot weight-per-contract ``wpc`` (one lot as a book-heat /
    risk-weight fraction), its per-position ceiling weight (the min() of seven ceilings × governor,
    from the LEVERAGE adapter), whether it is an incumbent (reserved first, P4), and a deterministic
    ``priority`` key (lower = admitted first; built from §3.4 ceiling-slack/MCTR-efficiency/id)."""

    instrument: str
    wpc: float            # one lot as a fraction of the book-heat / risk-weight budget
    ceiling_weight: float  # box ceiling 0 ≤ w_i ≤ ceiling_i (already × governor m)
    incumbent_lots: int    # held lots (reserved; P4) — never reduced by the fill
    target_weight: float   # the selection-proposed target risk weight w^sel_i
    priority: tuple        # §3.4 priority key (ceiling-slack, -mctr-eff, id); sorted ASC


@dataclass(frozen=True)
class FixedPointResult:
    """Output of :meth:`FixedPointComposer.solve` (SYSTEM §12). ``integer_book`` maps instrument →
    admitted integer lots (incumbents + new); ``iterations`` ≤ r_max; ``converged`` is the honest
    Section-3.3 definition; ``residual`` is the last ‖Δw‖ (logged on fallback); ``heat_used`` is the
    sum MCTR after the fill; ``deferred`` are candidates that could not be admitted (budget_deferred)."""

    integer_book: dict[str, int]
    weights: dict[str, float]
    iterations: int
    converged: bool
    residual: float
    heat_used: float
    deferred: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _project_box_simplex(w: np.ndarray, ceilings: np.ndarray, h_book: float) -> np.ndarray:
    """Euclidean projection onto F = {0 ≤ w_i ≤ ceiling_i, Σ w_i ≤ h_book} (Section 3.3).

    Box-clip first, then if the budget is exceeded scale the (non-zero) weights down proportionally
    to land on the simplex face Σ w_i = h_book. Proportional scaling is the projection direction the
    spec uses (scale the over-budget weights down toward MCTR-efficiency); it keeps the map firmly
    non-expansive on the convex box ∩ simplex.
    """
    w = np.clip(w, 0.0, ceilings)
    total = float(w.sum())
    if total > h_book and total > 0.0:
        w = w * (h_book / total)
        w = np.minimum(w, ceilings)  # re-clip after the scale (box still binds)
    return w


def continuous_relaxation(
    target: np.ndarray,
    ceiling_fn: Callable[[np.ndarray], np.ndarray],
    *,
    h_book: float,
    eta: float,
    r_max: int,
    tol: float,
) -> tuple[np.ndarray, int, bool, float]:
    """Damped averaged projection (Section 3.3). Returns (w_inf, iterations, converged, residual).

    ``ceiling_fn(w)`` returns the per-position box ceilings evaluated at the CURRENT iterate (the
    LEVERAGE engine re-priced under the current book). ``converged`` iff ‖w^(r) − w^(r-1)‖ ≤ tol
    within r_max steps.
    """
    w = np.zeros_like(target, dtype=float)
    residual = float("inf")
    iterations = 0
    converged = False
    for r in range(1, r_max + 1):
        iterations = r
        ceilings = np.asarray(ceiling_fn(w), dtype=float)
        feasible = _project_box_simplex(np.minimum(target, ceilings), ceilings, h_book)
        w_next = _project_box_simplex((1.0 - eta) * w + eta * feasible, ceilings, h_book)
        residual = float(np.linalg.norm(w_next - w))
        w = w_next
        if residual <= tol:
            converged = True
            break
    return w, iterations, converged, residual


def greedy_lot_fill(
    positions: Sequence[Position],
    *,
    h_book: float,
    equity_ladder_cap: float,
    equity_loss_per_lot: Mapping[str, float] | None = None,
) -> tuple[dict[str, int], dict[str, float], float, tuple[str, ...]]:
    """The single monotone greedy fill (Section 3.3). Lots only added, never revoked ⇒ no oscillation.

    Incumbents are seated first (reserved, P4); then new lots are admitted in priority order, each
    re-checking H_book AND the equity ladder against the partially-built book. Admit a lot only if
    BOTH constraints still hold after adding it. Terminates in ≤ Σ ceiling-lots steps by
    construction. Returns (integer_book, weights, heat_used, deferred).
    """
    equity_loss_per_lot = equity_loss_per_lot or {}
    book: dict[str, int] = {}
    heat = 0.0
    equity_loss = 0.0

    # 1. Seat incumbents first (reserved; P4 — never dropped to make room).
    for p in positions:
        if p.incumbent_lots > 0:
            book[p.instrument] = p.incumbent_lots
            heat += p.incumbent_lots * p.wpc
            equity_loss += p.incumbent_lots * equity_loss_per_lot.get(p.instrument, 0.0)

    # 2. Greedily admit new lots in deterministic priority order (ceiling-slack/MCTR-eff/id).
    ordered = sorted(positions, key=lambda p: p.priority)
    deferred: list[str] = []
    for p in ordered:
        held = book.get(p.instrument, 0)
        max_lots = int(p.ceiling_weight / p.wpc) if p.wpc > 0.0 else 0
        admitted_any = held > p.incumbent_lots
        while held < max_lots:
            new_heat = heat + p.wpc
            new_loss = equity_loss + equity_loss_per_lot.get(p.instrument, 0.0)
            if new_heat > h_book + 1e-12 or new_loss > equity_ladder_cap + 1e-12:
                break  # cannot admit this lot without breaching a shared constraint
            heat = new_heat
            equity_loss = new_loss
            held += 1
            book[p.instrument] = held
            admitted_any = True
        # A new-capital candidate (no incumbent lots) that got zero lots is budget_deferred.
        if p.incumbent_lots == 0 and not admitted_any and max_lots > 0:
            deferred.append(p.instrument)

    weights = {inst: lots * next(p.wpc for p in positions if p.instrument == inst) for inst, lots in book.items()}
    return book, weights, heat, tuple(sorted(deferred))


class FixedPointComposer:
    """Damped projection fixed point Φ + monotone greedy fill (SYSTEM §12 FixedPointComposer).

    ``solve`` runs the continuous relaxation to define the rounding target, then resolves the integer
    book by the single monotone greedy fill. Incumbents reserved first (P4); new-capital weights only
    scale DOWN; lot rounding DOWN only. On non-convergence returns the last FEASIBLE integer book
    with ``converged=False`` (Invariant IT). Deterministic tie-break by the §3.4 priority key (id
    last), seeded nowhere.
    """

    def solve(
        self,
        positions: Sequence[Position],
        *,
        h_book: float,
        equity_ladder_cap: float,
        eta: float,
        r_max: int,
        tol: float,
        ceiling_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        equity_loss_per_lot: Mapping[str, float] | None = None,
    ) -> FixedPointResult:
        """Solve the fixed point (Section 3). ``ceiling_fn`` re-prices the box ceilings under the
        current continuous iterate (the LEVERAGE adapter under the current book); when omitted the
        static per-position ``ceiling_weight`` is used (benign-input / unit-test path)."""
        order = sorted(positions, key=lambda p: p.instrument)
        target = np.array([p.target_weight for p in order], dtype=float)
        static_ceilings = np.array([p.ceiling_weight for p in order], dtype=float)

        if ceiling_fn is None:
            ceiling_fn = lambda _w: static_ceilings  # noqa: E731

        _w_inf, iterations, converged, residual = continuous_relaxation(
            target, ceiling_fn, h_book=h_book, eta=eta, r_max=r_max, tol=tol
        )

        book, weights, heat, deferred = greedy_lot_fill(
            order,
            h_book=h_book,
            equity_ladder_cap=equity_ladder_cap,
            equity_loss_per_lot=equity_loss_per_lot,
        )

        reasons = ["fixed_point"]
        if not converged:
            reasons.append("iterate_fallback")
        if deferred:
            reasons.append("budget_deferred")
        return FixedPointResult(
            integer_book=book,
            weights=weights,
            iterations=iterations,
            converged=converged,
            residual=residual,
            heat_used=heat,
            deferred=deferred,
            reason_codes=tuple(reasons),
        )
