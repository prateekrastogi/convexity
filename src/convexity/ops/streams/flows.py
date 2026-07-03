"""External capital flows: TWR return stripping + the flow-event risk check (OPS.md sec 15).

THE PROBLEM (OPS.md sec 15, Invariant FLOW-TWR): every detector in this pillar — the kill-switch
CUSUM (sec 4), the fractional-Kelly promotion ratchet (LEVERAGE sec 6.3), the drawdown governor
(LEVERAGE sec 6.4) — consumes the book's weekly return series. A deposit or withdrawal changes
account equity WITHOUT being a trading return: fed raw, a deposit looks like a windfall (false
ratchet promotion / ANOMALOUS_OUTPERFORMANCE) and a withdrawal looks like a crash (false HALT that
then requires a manual operator re-arm). The fix is the standard TIME-WEIGHTED return: split the
week into sub-periods at each flow instant and geometrically link the sub-period returns, so
external flows contribute EXACTLY zero to the measured return (exact linking, never the
Modified-Dietz approximation — closed-form, no estimation).

Pure policy code (NautilusTrader-free): the live adapter supplies equity marks and flow events
(Nautilus account events carry both); nothing here reads venue state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from convexity.ops.config.defaults import OpsDefaults

FlowAction = Literal["NONE", "RECONCILE_ONLY", "INTERIM_RESIZE"]


@dataclass(frozen=True)
class FlowEvent:
    """One external capital flow (OPS.md sec 15.1). AMBER facts read off the account stream.

    ``amount`` is SIGNED quote-ccy: deposit > 0, withdrawal < 0. ``equity_before`` is the account
    equity at the flow instant, BEFORE the flow lands — the adapter must capture it from the same
    account event (it is what makes the TWR linking exact rather than approximated).
    """

    ts: int  # epoch-ms of the flow
    amount: float  # signed: deposit > 0, withdrawal < 0
    equity_before: float  # account equity immediately BEFORE the flow

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount) or self.amount == 0.0:
            raise ValueError(f"flow amount must be finite and nonzero, got {self.amount!r}")
        if not math.isfinite(self.equity_before) or self.equity_before <= 0.0:
            raise ValueError(f"equity_before must be finite and > 0, got {self.equity_before!r}")
        if self.equity_before + self.amount <= 0.0:
            raise ValueError(
                f"flow {self.amount!r} would take equity {self.equity_before!r} to <= 0 "
                "(a withdrawal cannot exceed the account, OPS.md sec 15.1)"
            )


def twr_weekly_return(
    equity_start: float,
    equity_end: float,
    flows: tuple[FlowEvent, ...] = (),
    *,
    ts_start: int | None = None,
    ts_end: int | None = None,
) -> float:
    """Flow-stripped (time-weighted) weekly return (OPS.md sec 15.1, Invariant FLOW-TWR).

    Sub-period geometric linking at each flow instant::

        r = [ E(f_1)/E_start ] * [ E(f_2)/(E(f_1)+F_1) ] * ... * [ E_end/(E(f_k)+F_k) ] - 1

    where ``E(f_j)`` is ``flows[j].equity_before`` and ``F_j`` its signed amount. With no flows this
    reduces EXACTLY to ``E_end/E_start - 1`` (the raw return). A flow with flat prices contributes
    exactly 0 (the property the raw return violates). This is the ONE return that may feed
    ``whiten``/the CUSUM/the ratchet/the governor when any flow occurred in the week (Invariant
    FLOW-TWR); feeding a raw equity return instead is the false-HALT/false-promotion bug this
    module exists to prevent.

    ``ts_start``/``ts_end`` (optional, epoch-ms) bound the week: when supplied, every flow must
    satisfy ``ts_start < flow.ts <= ts_end`` (a flow outside the settlement window belongs to a
    different week's linking — rejected loudly, never silently absorbed). Flows must be
    time-ordered. Raises on non-positive equities or a non-positive linking denominator.
    """
    if not math.isfinite(equity_start) or equity_start <= 0.0:
        raise ValueError(f"equity_start must be finite and > 0, got {equity_start!r}")
    if not math.isfinite(equity_end) or equity_end <= 0.0:
        raise ValueError(f"equity_end must be finite and > 0, got {equity_end!r}")

    prev_ts: int | None = None
    for f in flows:
        if ts_start is not None and f.ts <= ts_start:
            raise ValueError(f"flow at {f.ts} is at/before the week start {ts_start} (sec 15.1)")
        if ts_end is not None and f.ts > ts_end:
            raise ValueError(f"flow at {f.ts} is after the week end {ts_end} (sec 15.1)")
        if prev_ts is not None and f.ts < prev_ts:
            raise ValueError("flows must be time-ordered (sec 15.1)")
        prev_ts = f.ts

    growth = 1.0
    base = equity_start
    for f in flows:
        growth *= f.equity_before / base
        base = f.equity_before + f.amount  # flow lands; next sub-period starts here
    growth *= equity_end / base
    return growth - 1.0


@dataclass(frozen=True)
class FlowDirective:
    """The flow-event risk-check outcome (OPS.md sec 15.2). P3: report/act via ONE directive.

    ``action``: ``NONE`` (below threshold — Friday handles it), ``RECONCILE_ONLY`` (material
    DEPOSIT: leverage fell BELOW target, which is the safe direction — never add risk mid-week,
    Invariant CAL-PRIMARY; run reconciliation so tolerances track the new equity), or
    ``INTERIM_RESIZE`` (material WITHDRAWAL: realized gross/NAV now sits ABOVE target — scale every
    position by ``resize_factor`` reduce-only to restore it). ``resize_factor`` is
    ``(equity_before + amount)/equity_before`` (< 1) for INTERIM_RESIZE, else 1.0.
    """

    action: FlowAction
    relative: float  # |amount| / equity_before
    resize_factor: float
    reason: str


def flow_risk_check(flow: FlowEvent, defaults: OpsDefaults | None = None) -> FlowDirective:
    """Immediate risk check on one external flow (OPS.md sec 15.2).

    The weekly calendar remains the only always-on TRADE trigger (Invariant CAL-PRIMARY); a flow
    never triggers opportunistic re-entry. But a withdrawal at/above the RED
    ``flow_check_threshold`` mechanically levers the book ABOVE its target gross the instant it
    lands (positions unchanged, equity smaller) — the one flow case that adds risk — so it draws
    the reduce-only ``INTERIM_RESIZE`` directive instead of waiting for Friday. A deposit of any
    size only DE-levers (safe direction): ``RECONCILE_ONLY`` at/above threshold, ``NONE`` below.
    HALT precedence is unchanged: a flow never re-arms a HALTed book (OPS.md sec 6 step 5).
    """
    d = defaults if defaults is not None else OpsDefaults()
    relative = abs(flow.amount) / flow.equity_before
    if relative < d.flow_check_threshold:
        return FlowDirective(
            action="NONE",
            relative=relative,
            resize_factor=1.0,
            reason=(
                f"flow {relative:.1%} of equity < threshold "
                f"{d.flow_check_threshold:.0%}; weekly rebalance absorbs it (sec 15.2)"
            ),
        )
    if flow.amount > 0.0:
        return FlowDirective(
            action="RECONCILE_ONLY",
            relative=relative,
            resize_factor=1.0,
            reason=(
                f"deposit {relative:.1%} of equity: gross/NAV fell BELOW target (safe "
                "direction); reconcile tolerances now, sizing catches up at the Friday "
                "rebalance (CAL-PRIMARY, sec 15.2)"
            ),
        )
    factor = (flow.equity_before + flow.amount) / flow.equity_before
    return FlowDirective(
        action="INTERIM_RESIZE",
        relative=relative,
        resize_factor=factor,
        reason=(
            f"withdrawal {relative:.1%} of equity: gross/NAV now ABOVE target by "
            f"{1.0 / factor:.3f}x; reduce-only proportional resize by {factor:.4f} "
            "restores it (sec 15.2, P6 risk-off)"
        ),
    )
