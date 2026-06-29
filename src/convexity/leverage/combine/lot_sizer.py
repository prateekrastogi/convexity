"""Quantizer — round-down-or-reject + dual-cap revalidation (LEVERAGE §8.6, Inv U/M).

Converts the continuous (raw_leverage, raw_notional) target into whole tradable units
under a strict rule: quantization may only REDUCE risk or REJECT the trade — never
increase it (Inv U). This is a GATE, not an optimizer: it rounds DOWN and re-checks.

Rounding rule (§8.6.2): ``units = floor(raw_notional / unit_notional)`` snapped to the
instrument's lot step, then capped at ``min(ceiling_implied_lots, exchange_max_lots,
class_cap_lots)`` — each only ever LOWERS units (Inv U/CLASSCAP). Rounding up "for
convenience" is the single most common silent risk-limit leak and is forbidden.

Re-validation after rounding (§8.6.3): realized capital loss at stop on BOTH bases
(position margin and total equity — Inv M dual basis), realized SPAN margin and the
resulting free-margin buffer, and liquidity/heat — all recomputed on the rounded size.
Zero-rounding is an EXPLICIT reported outcome (``below_min_unit``), never a silent drop,
including the governor-throttles-below-one-lot case.

EXISTING-BOOK MARGIN (§8.6.3, "the resulting free-margin buffer AFTER this position is
added to the existing book"). The post-rounding free-margin re-check must net out the
margin ALREADY consumed by the OPEN book (``existing_margin_used``), not just this
position's SPAN. Ignoring incumbents lets a position pass the buffer check yet breach
once the incumbents' margin counts: the buffer is therefore computed on the equity AFTER
incumbents — ``free_after = free_margin − existing_margin_used − span_margin`` and the
floor on that same post-incumbent base (``buffer_floor = buffer · (free_margin −
existing_margin_used)``). SYSTEM keeps incumbents FIXED (P4); their reserved margin is
subtracted off the top here exactly as the aggregate trim reserves their cluster heat.

Aggregate trim (§8.6.4): when run STAND-ALONE (no SYSTEM orchestrator), the book-aware
trim uses SYSTEM's priority order VERBATIM (ceiling-slack → MCTR-efficiency →
instrument id) and NEVER an edge input (Inv A); incumbents are FIXED (SYSTEM P4) and
never re-quantized/trimmed. When orchestrated, SYSTEM owns the single greedy lot-fill
and this module supplies only the per-instrument round-down+revalidate leaf.
"""

from __future__ import annotations

import math
from typing import Mapping

from convexity.leverage.config.margin import MarginModel
from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import QuantizedSize


def _floor_steps(raw_quantity: float, lot_step: float) -> int:
    """Round a continuous base-unit quantity DOWN to a whole number of LOT STEPS (§8.6.2).

    Returns the integer COUNT of lot steps (the ``units`` field). The realized tradable
    quantity is ``units * lot_step`` (e.g. 36 steps × 0.001 BTC = 0.036 BTC; for a 65-lot
    future, 2 steps × 65 = 130 contracts). Rounding DOWN guarantees the realized size is
    <= the continuous target, so the rounding never breaches a constraint (Inv U).
    """
    if lot_step <= 0:
        return int(math.floor(raw_quantity))
    return int(math.floor(raw_quantity / lot_step))


class Quantizer:
    """Per-instrument round-down + revalidate leaf (§8.6.1-§8.6.3)."""

    def quantize(
        self,
        raw_leverage: float,
        raw_notional: float,
        instrument,  # InstrumentLeverageSpec
        position_margin: float,
        account_equity: float,
        entry_price: float,
        stop_distance: float,
        capital_cap: float,
        margin_model: MarginModel,
        free_margin: float,
        existing_margin_used: float,
        cfg: ResolvedConfig,
        ceiling_implied_lots: float | None = None,
        class_cap_lots: float | None = None,
    ) -> QuantizedSize:
        """Round DOWN to whole tradable units and re-validate every binding constraint
        (§8.6.2/§8.6.3). ``raw_notional`` is the continuous target notional; ``stop_distance``
        is d (fractional adverse move); ``capital_cap`` is c. Returns a ``QuantizedSize``.
        """
        flags: list[str] = []
        price = max(float(entry_price), 1e-12)
        mult = float(instrument.contract_multiplier)
        lot_step = float(instrument.lot_step)
        # notional of ONE lot step (e.g. 0.001 BTC, or a 65-contract NIFTY lot)
        step_notional = mult * price * lot_step if lot_step > 0 else mult * price
        # continuous target as a COUNT OF LOT STEPS, and the raw base-unit quantity
        raw_quantity = float(raw_notional) / (mult * price) if mult * price > 0 else 0.0
        raw_steps = raw_quantity / lot_step if lot_step > 0 else raw_quantity

        units = _floor_steps(raw_quantity, lot_step)  # integer count of lot steps (round DOWN)
        if units > 0 and units < raw_steps:
            flags.append("rounded_down")

        # Upper caps (each only LOWERS units — Inv U/CLASSCAP). ``ceiling_implied_lots`` /
        # ``class_cap_lots`` are continuous base-unit quantities from the leverage ceilings.
        if ceiling_implied_lots is not None:
            cap_units = _floor_steps(float(ceiling_implied_lots), lot_step)
            if units > cap_units:
                units = cap_units
        if class_cap_lots is not None:
            cc_units = _floor_steps(float(class_cap_lots), lot_step)
            if units > cc_units:
                units = cc_units
                flags.append("class_cap_capped")
        if instrument.exchange_max_lots is not None:
            if units > instrument.exchange_max_lots:
                units = int(instrument.exchange_max_lots)
                flags.append("exchange_max_capped")

        # free margin net of the OPEN book (§8.6.3 existing-book margin): incumbents are
        # FIXED (SYSTEM P4), their margin reserved off the top before this position counts.
        existing = max(float(existing_margin_used), 0.0)
        avail = float(free_margin) - existing

        # Below one tradable lot step -> explicit zero outcome (§8.6.2.3), never silent.
        if units <= 0:
            divergence = 1.0 if raw_steps > 0 else 0.0
            return QuantizedSize(
                units=0,
                notional=0.0,
                realized_leverage=0.0,
                realized_loss_pct_margin=0.0,
                realized_loss_pct_equity=0.0,
                realized_span_margin=0.0,
                free_margin_after=avail,
                divergence=divergence,
                flags=tuple(flags + ["below_min_unit"]),
                rejected=True,
            )

        notional = units * step_notional
        realized_leverage = notional / max(float(position_margin), 1e-12)

        # --- §8.6.3 re-validation on the rounded size --------------------------------
        # realized loss at stop on BOTH bases (Inv M)
        loss_quote = notional * float(stop_distance)
        loss_pct_margin = loss_quote / max(float(position_margin), 1e-12)
        loss_pct_equity = loss_quote / max(float(account_equity), 1e-12)

        # realized SPAN margin + free-margin buffer after adding this position to the
        # EXISTING book (§8.4/§8.6.3). The buffer is computed on the margin remaining
        # AFTER incumbents already consumed ``existing_margin_used`` (SYSTEM P4: incumbents
        # FIXED, their margin reserved off the top) — NOT on gross free margin, else this
        # position can pass the buffer check yet breach once incumbents count.
        span_margin = margin_model.requirement(notional, price)
        free_after = avail - span_margin
        buffer_floor = cfg.defaults.free_margin_buffer * avail

        rejected = False
        # rounding down cannot breach c, but margin is LUMPY (one more lot is a jump):
        # if the integer lot's margin breaks the free-margin buffer, drop a lot or reject.
        while free_after < buffer_floor and units > 0:
            units -= 1
            if "margin_after_rounding" not in flags:
                flags.append("margin_after_rounding")
            if units <= 0:
                rejected = True
                notional = 0.0
                realized_leverage = 0.0
                loss_pct_margin = 0.0
                loss_pct_equity = 0.0
                span_margin = 0.0
                free_after = avail
                break
            notional = units * step_notional
            realized_leverage = notional / max(float(position_margin), 1e-12)
            loss_quote = notional * float(stop_distance)
            loss_pct_margin = loss_quote / max(float(position_margin), 1e-12)
            loss_pct_equity = loss_quote / max(float(account_equity), 1e-12)
            span_margin = margin_model.requirement(notional, price)
            free_after = avail - span_margin

        divergence = (raw_steps - units) / raw_steps if raw_steps > 0 else 0.0

        return QuantizedSize(
            units=int(units),
            notional=float(notional),
            realized_leverage=float(realized_leverage),
            realized_loss_pct_margin=float(loss_pct_margin),
            realized_loss_pct_equity=float(loss_pct_equity),
            realized_span_margin=float(span_margin),
            free_margin_after=float(free_after),
            divergence=float(divergence),
            flags=tuple(flags),
            rejected=bool(rejected),
        )


# --------------------------------------------------------------------------- aggregate trim

def aggregate_trim(
    sized: Mapping[str, QuantizedSize],
    priority: Mapping[str, tuple[float, float]],
    cluster_budget: float,
    total_free_margin: float,
    cluster_heat: Mapping[str, float],
    *,
    incumbents: frozenset[str] = frozenset(),
) -> dict[str, QuantizedSize]:
    """Book-aware aggregate trim using SYSTEM's priority order VERBATIM (§8.6.4).

    Trims the LOWEST-priority NON-incumbent position by a lot until the aggregate cluster
    heat and total free-margin buffer fit. Priority order (NEVER an edge input, Inv A):
      1. highest binding-ceiling SLACK first (room before the nearest failure mode),
      2. highest MCTR-efficiency = risk-weight delivered / book heat consumed,
      3. tie-break: instrument id (deterministic).
    Incumbents are FIXED (SYSTEM P4): never re-quantized/trimmed; their reserved heat is
    subtracted from the budget first. A breach trims and flags ``cluster_trimmed``.
    """
    out = dict(sized)
    # incumbent heat reserved off the top
    incumbent_heat = sum(cluster_heat.get(i, 0.0) for i in incumbents if i in out)
    budget = float(cluster_budget) - incumbent_heat

    def heat_used() -> float:
        return sum(cluster_heat.get(k, 0.0) for k in out if k not in incumbents)

    def margin_used() -> float:
        return sum(q.realized_span_margin for q in out.values())

    # lowest priority LAST: sort ascending by (slack, mctr_eff), then by id; trim from the
    # FRONT (lowest priority) of the non-incumbent list.
    def order() -> list[str]:
        non_inc = [k for k in out if k not in incumbents and out[k].units > 0]
        non_inc.sort(key=lambda k: (priority.get(k, (0.0, 0.0))[0],
                                    priority.get(k, (0.0, 0.0))[1], k))
        return non_inc

    guard = 0
    while (heat_used() > budget or margin_used() > total_free_margin) and guard < 10_000:
        guard += 1
        candidates = order()
        if not candidates:
            break
        victim = candidates[0]
        q = out[victim]
        new_units = int(q.units - 1)  # units are LOT-STEP counts; trim one step (§8.6.4)
        if new_units <= 0:
            # drops to zero -> remove from book; flag cluster_trimmed
            out[victim] = QuantizedSize(
                units=0, notional=0.0, realized_leverage=0.0,
                realized_loss_pct_margin=0.0, realized_loss_pct_equity=0.0,
                realized_span_margin=0.0, free_margin_after=q.free_margin_after,
                divergence=q.divergence, flags=tuple(q.flags) + ("cluster_trimmed",),
                rejected=True,
            )
        else:
            ratio = new_units / max(q.units, 1)
            out[victim] = QuantizedSize(
                units=new_units,
                notional=q.notional * ratio,
                realized_leverage=q.realized_leverage * ratio,
                realized_loss_pct_margin=q.realized_loss_pct_margin * ratio,
                realized_loss_pct_equity=q.realized_loss_pct_equity * ratio,
                realized_span_margin=q.realized_span_margin * ratio,
                free_margin_after=q.free_margin_after,
                divergence=q.divergence,
                flags=tuple(q.flags) + ("cluster_trimmed",) if "cluster_trimmed" not in q.flags
                else tuple(q.flags),
                rejected=False,
            )
            # the cluster_heat for the victim scales too
            if victim in cluster_heat:
                cluster_heat = {**cluster_heat, victim: cluster_heat[victim] * ratio}
    return out
