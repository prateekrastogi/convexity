"""Reconciliation tolerance map: instrument -> ``epsilon_i`` and the WARN/BREACH escalation ladder.

``docs/specs/OPS.md`` sec 8 names this module ``ops/reconcile/streams_map.py`` — the per-instrument
tolerance ``epsilon_i`` (sec 1.2) and the OK/WARN/BREACH/STALE mapping (sec 1.4). The book-vs-exchange
position *snapshot* that feeds it is the adapter in :mod:`convexity.ops.streams.recon` (sec 2 /
sec 1); this module holds only the pure tolerance/tier logic (no live venue dependency).

At any instant the book OPS believes it holds and the position the exchange reports must agree, per
instrument, up to a stated tolerance (sec 1.1)::

    delta_i,t = book_position_i,t - exchange_position_i,t
    INVARIANT:  |delta_i,t| <= epsilon_i     for every live instrument i

The tolerance ``epsilon_i`` is SCALED to the instrument's lot size / notional, never a flat global
constant (sec 1.2) — a single cross-instrument epsilon is either too loose for majors or cries wolf
on alt dust, the classic uncalibrated-control-limit SPC failure. Escalation (sec 1.4): OK / WARN
(logged, no book action) / BREACH (immediate ``RECON_BREACH`` -> ``HALT``, a Shewhart single-point
rule, NOT waiting for a statistical filter). A failed READ is a ``RECON_STALE`` WARN, never a silent
``OK`` (sec 1.4: "I could not check" must never be indistinguishable from "I checked and it's fine").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from convexity.core.types import Alarm

# The tier a single instrument's reconciliation lands in (OPS.md sec 1.4).
ReconTier = Literal["OK", "WARN", "BREACH", "STALE"]


def epsilon_i(
    *,
    lot_size_i: float,
    target_notional_i: float,
    price_i: float,
    k_lot: float,
    k_notional: float,
) -> float:
    """Per-instrument reconciliation tolerance ``epsilon_i`` (OPS.md sec 1.2).

    ``epsilon_i = max( k_lot * lot_size_i , k_notional * target_notional_i / price_i )`` — the
    ``max()`` floors the tolerance at whichever failure mode binds: a tiny position is bounded by the
    lot-rounding floor even when 0.1% of its notional would be sub-lot; a large position is bounded
    by the proportional term (sec 1.2). Both terms are in position-QUANTITY units.

    ``k_lot``/``k_notional`` are RED (``OpsDefaults``); ``lot_size_i``/``target_notional_i``/
    ``price_i`` are AMBER per-instrument metadata (venue ``stepSize`` and the intended dollar size).
    """
    if price_i <= 0.0:
        raise ValueError(f"price_i must be > 0 for a live instrument, got {price_i}")
    lot_floor = k_lot * lot_size_i
    notional_floor = k_notional * target_notional_i / price_i
    return max(lot_floor, notional_floor)


@dataclass(frozen=True)
class ReconResult:
    """One instrument's reconciliation outcome (OPS.md sec 1.4).

    ``tier`` is the escalation tier; ``alarm`` is populated (non-``None``) exactly when ``tier`` is
    WARN/BREACH/STALE and is the fully-attributed reason record fed forward to the brake (sec 1.4).
    A BREACH's alarm ``kind`` is ``RECON_BREACH`` and its tier is BREACH (an immediate HALT
    trigger); a WARN-band POSITION divergence is ALSO ``RECON_BREACH`` kind (same event class,
    sub-severe) but tier WARN (surfaced, no book action); ``RECON_STALE`` is reserved strictly
    for a failed reconciliation READ (see :func:`reconcile_stale`), never a live drift (sec 1.4).

    NOTE on the WARN/kind relationship (a documented, closed-set choice, not an oversight): sec 9's
    ``AlarmKind`` Literal has no dedicated WARN kind, so a sub-severe live divergence carries kind
    ``RECON_BREACH`` with ``tier == "WARN"``. A downstream consumer keying a HALT purely on
    ``alarm.kind == "RECON_BREACH"`` WITHOUT also reading ``ReconResult.tier`` would over-halt on a
    WARN. The brake (:mod:`convexity.ops.brake.triggers`) therefore keys on ``tier``, never on
    ``kind`` alone, when deciding to escalate — the tier is the load-bearing severity discriminator,
    and the ``triggers`` module makes that explicit so the ambiguity cannot leak to the live loop.
    """

    instrument: str
    delta: float
    epsilon: float
    tier: ReconTier
    alarm: Alarm | None


def reconcile_instrument(
    *,
    instrument: str,
    book_position: float,
    exchange_position: float,
    epsilon: float,
    m_warn: float,
    as_of: int,
) -> ReconResult:
    """Reconcile ONE instrument's book vs exchange position against its tolerance (OPS.md sec 1.4).

    Escalation ladder (sec 1.4)::

        |delta| <= epsilon                     -> OK      (no action)
        epsilon < |delta| <= epsilon * m_warn  -> WARN    (log + notify, no book action)
        |delta| > epsilon * m_warn             -> BREACH  -> RECON_BREACH -> HALT (Shewhart single-point)

    A BREACH does NOT wait for the CUSUM statistic to accumulate — a large position-state divergence
    is not a statistical signal to filter for noise, it is a direct certainty that ground truth is
    wrong right now (sec 1.4). ``m_warn`` is RED.
    """
    delta = book_position - exchange_position
    abs_delta = abs(delta)
    warn_bound = epsilon * m_warn
    if abs_delta <= epsilon:
        return ReconResult(instrument, delta, epsilon, "OK", None)
    if abs_delta <= warn_bound:
        # WARN-band: a small live POSITION divergence (same event class as a BREACH, sub-severe),
        # NOT a stale read. It is a RECON_BREACH-KIND alarm (the divergence event) distinguished
        # from a halting breach by its WARN tier, never a RECON_STALE (which sec 1.4 reserves
        # strictly for a failed reconciliation READ) — overloading RECON_STALE here would make a
        # downstream consumer keying on it to mean "stale read" misattribute a live drift (sec 1.4,
        # sec 9 AlarmKind has no dedicated WARN kind). The tier, not the kind, is the severity gate
        # (see the ReconResult docstring + brake.triggers).
        alarm = Alarm(
            kind="RECON_BREACH",
            detail=f"{instrument}: |delta|={abs_delta:.8g} in ({epsilon:.8g}, {warn_bound:.8g}] "
                   "WARN band (divergence, sub-severe; tier=WARN, no book action)",
            value=abs_delta,
            threshold=warn_bound,
            as_of=as_of,
        )
        return ReconResult(instrument, delta, epsilon, "WARN", alarm)
    alarm = Alarm(
        kind="RECON_BREACH",  # immediate HALT trigger (sec 1.4)
        detail=f"{instrument}: |delta|={abs_delta:.8g} > m_warn*epsilon={warn_bound:.8g} BREACH",
        value=abs_delta,
        threshold=warn_bound,
        as_of=as_of,
    )
    return ReconResult(instrument, delta, epsilon, "BREACH", alarm)


def reconcile_stale(*, instrument: str, days_unreadable: int, staleness_max_days: int,
                    as_of: int) -> ReconResult:
    """A reconciliation READ that itself failed (exchange API error, stale websocket) (OPS.md sec 1.4).

    A failed read is a ``RECON_STALE`` WARN-tier event, escalating to BREACH-tier once the read has
    been unreadable for more than ``staleness_max_days`` (RED) — "I could not check" must never be
    indistinguishable from "I checked and it's fine" (sec 1.4, conservative fallback, mirrors
    COST.md P6's no-permissive-default rule). ``delta``/``epsilon`` are NaN-free sentinels (0.0)
    because there is no reading to compare.
    """
    if days_unreadable > staleness_max_days:
        alarm = Alarm(
            kind="RECON_BREACH",
            detail=f"{instrument}: reconciliation unreadable {days_unreadable}d > staleness_max "
                   f"{staleness_max_days}d -> escalate STALE to BREACH",
            value=float(days_unreadable),
            threshold=float(staleness_max_days),
            as_of=as_of,
        )
        return ReconResult(instrument, 0.0, 0.0, "BREACH", alarm)
    alarm = Alarm(
        kind="RECON_STALE",
        detail=f"{instrument}: reconciliation read failed ({days_unreadable}d unreadable) STALE",
        value=float(days_unreadable),
        threshold=float(staleness_max_days),
        as_of=as_of,
    )
    return ReconResult(instrument, 0.0, 0.0, "STALE", alarm)


def worst_tier(results: list[ReconResult]) -> ReconTier:
    """The most-severe tier across the book (OPS.md sec 1.4: a BREACH on ANY instrument escalates).

    Ordering BREACH > WARN ~ STALE > OK: any single BREACH means the whole book must HALT (P3: the
    unit of action is the whole book), regardless of the CUSUM state at that instant (Inv RECON).
    """
    order: dict[ReconTier, int] = {"OK": 0, "STALE": 1, "WARN": 1, "BREACH": 2}
    worst: ReconTier = "OK"
    for r in results:
        if order[r.tier] > order[worst]:
            worst = r.tier
    return worst
