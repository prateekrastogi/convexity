"""Per-name half-spread + Amihud impact estimator (COST.md Sections 5.2-5.4, 5.5).

The DERIVED per-name cost model that supersedes ``research/xsec/accounting.py``'s flat
``HALF_SPREAD_FRAC`` / ``IMPACT_FRAC`` constants for the crypto production sleeve. It supplies the
three per-leg friction components (``taker``, ``half_spread_i``, ``impact_frac_i``) that assemble
into the ONE per-name round-trip rate ``round_trip_friction_i = 2*(taker + half_spread_i +
impact_frac_i)`` — the exact signature ``research/xsec/accounting.py::round_trip_friction`` already
implements (Invariant N: this module supplies the rate INPUTS, it never re-derives the turnover /
settlement arithmetic of ``settle_week``).

HALF-SPREAD (COST.md Section 5.2). The square-root law of market impact (Grinold & Kahn 2000 ch.16;
Gatheral 2010) gives a concave, sub-linear cost in participation. Applied to a touch-crossing
half-spread proxy inversely proportional to sqrt-liquidity:

    half_spread_i = c / sqrt(ADV_i),   c = 1bp * sqrt(ADV_median)

so ``half_spread_i = 1bp * sqrt(ADV_median / ADV_i)`` — ONE calibration point (median name pays
~1bp, matching accounting.py's ``HALF_SPREAD_FRAC = 1e-4``) plus the sqrt form fixes every other
name's spread from its own causal trailing ``ADV_i``, no per-name fit (P7).

IMPACT (COST.md Section 5.3-5.4). The Amihud (2002) illiquidity ratio ``Amihud_i = mean|r|/$Vol`` is
the closed-form local linear-response slope. The square-root law forces the large-trade correction:

    impact_frac_i = k_amihud * sqrt(Amihud_i * Notional_leg)

with the exponent 1/2 on ``Notional_leg`` DERIVED (matching Section 5.2's concavity) and the scalar
``k_amihud`` CALIBRATED so the median name's impact matches accounting.py's ``IMPACT_FRAC = 0.5e-4``
at the book's typical rebalance size — exactly the ``c``/``ADV_median`` anchoring pattern. ``Amihud_i``
supplies the cross-sectional SHAPE (which names are relatively illiquid); ``k_amihud`` supplies the
LEVEL anchor. Dimensionally ``Amihud_i`` is [return]/[dollar], so ``sqrt(Amihud_i * Notional_leg)``
is sqrt([return]); ``k_amihud`` absorbs the residual sqrt(1/[return]) — the calibration IS the unit
fix (Section 5.4).

CONVENTION (Invariant CONV, COST.md Section 5.5). accounting.py treats its ``half_spread`` argument
as ALREADY the per-leg cost (no further /2). This module matches that convention: ``half_spread_i``
here is the per-leg crossing cost, fed straight into ``round_trip_friction`` with NO extra halving —
composing this spec's derivation with an already-halved constant is the double-discount bug CONV
exists to catch. At the median anchor the derived rate equals accounting.py's flat default
bit-for-bit (13bp round-trip), so the per-name model is a drop-in generalization.

SHORTABILITY (Invariant N50, COST.md Section 7). A ``sqrt(Amihud_i * N)`` estimate is a
LIQUIDITY-cost number, not a FEASIBILITY number: it returns a finite bps for any name regardless of
borrow availability. When the (upstream, UNIVERSE-owned) per-name ``is_shortable`` flag says a name
placed in the SHORT leg cannot be borrowed, ``round_trip_friction_i`` returns ``math.inf`` (never
NaN, never a large-but-finite placeholder) so ``settle_week`` rejects the infeasible allocation
arithmetically instead of silently pricing it. Until UNIVERSE exposes that flag, callers pass
``is_shortable=None`` and this is a DOCUMENTED KNOWN GAP (Section 7 item 1), not a silent success.

PARTICIPATION (Invariant PARTICIPATION, COST.md Section 5.3). The anchors are calibrated at the
realized small-single-digit-%-of-ADV participation of a ~50-name weekly dollar-neutral book. This
model must NOT be applied unmodified at an order-of-magnitude larger book size without re-running the
walk-forward calibration check (``research/xsec/cost/validate/amihud_calibration.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from convexity.cost.accounting import (
    BINANCE_VIP0_TAKER,
    HALF_SPREAD_FRAC,
    IMPACT_FRAC,
    round_trip_friction,
)
from convexity.panel.sleeve import CRYPTO_FRICTION

# The book's typical single-name rebalance notional (dollars) at which the impact anchor is
# calibrated (COST.md Section 5.3: "~0.5bp at the book's typical rebalance size"). A ~50-name
# dollar-neutral book at ~$1M NAV with ~4% single-name weight changes -> ~$40k per-name leg
# (COST.md Section 5.2 worked example). This is the calibration reference size, not a per-name fit.
ANCHOR_NOTIONAL: float = 40_000.0


def _median_positive(x: np.ndarray) -> float:
    """Median over strictly-positive, finite entries (NaN/0/negatives dropped), NaN if none."""
    v = np.asarray(x, dtype=float)
    good = v[np.isfinite(v) & (v > 0.0)]
    return float(np.median(good)) if good.size else float("nan")


@dataclass(frozen=True)
class SpreadImpactModel:
    """Per-name half-spread + Amihud impact, anchored to accounting.py's flat median defaults.

    Immutable value object holding the two calibrated RED anchor constants (COST.md Section 4) and
    the shared taker fee. ``anchor(...)`` builds an instance whose ``c`` / ``k_amihud`` are pinned to
    a universe's median liquidity so the median name reproduces ``HALF_SPREAD_FRAC`` / ``IMPACT_FRAC``
    exactly (Invariant CONV / Section 5.5). No per-instrument state lives here — ``ADV_i`` /
    ``Amihud_i`` are passed per call (AMBER supplied metadata, Section 4), never stored.
    """

    c: float  # half-spread anchor: half_spread_i = c / sqrt(ADV_i)  (COST.md Section 5.2)
    k_amihud: float  # impact anchor: impact_frac_i = k_amihud * sqrt(Amihud_i * Notional)  (Section 5.3)
    taker: float = BINANCE_VIP0_TAKER  # shared VIP0 taker (the ONE fee truth, via accounting.py)
    anchor_notional: float = ANCHOR_NOTIONAL

    @classmethod
    def anchor(
        cls,
        adv_median: float,
        amihud_median: float,
        *,
        half_spread_anchor: float = HALF_SPREAD_FRAC,
        impact_anchor: float = IMPACT_FRAC,
        anchor_notional: float = ANCHOR_NOTIONAL,
        taker: float = BINANCE_VIP0_TAKER,
    ) -> "SpreadImpactModel":
        """Calibrate ``c`` / ``k_amihud`` so the MEDIAN name reproduces the flat research anchors.

        ``c = half_spread_anchor * sqrt(adv_median)`` makes ``half_spread(adv_median) ==
        half_spread_anchor`` (COST.md Section 5.2). ``k_amihud = impact_anchor / sqrt(amihud_median *
        anchor_notional)`` makes ``impact_frac(amihud_median, anchor_notional) == impact_anchor``
        (Section 5.3). One anchor point + the derived functional form fixes every other name — no
        per-name fit (P7). Raises ``ValueError`` on a non-positive median (a degenerate universe with
        no liquid name has no well-defined anchor).
        """
        if not (adv_median > 0.0 and math.isfinite(adv_median)):
            raise ValueError(f"adv_median must be finite positive, got {adv_median}")
        if not (amihud_median > 0.0 and math.isfinite(amihud_median)):
            raise ValueError(f"amihud_median must be finite positive, got {amihud_median}")
        c = half_spread_anchor * math.sqrt(adv_median)
        k_amihud = impact_anchor / math.sqrt(amihud_median * anchor_notional)
        return cls(c=c, k_amihud=k_amihud, taker=taker, anchor_notional=anchor_notional)

    def half_spread(self, adv_i: float) -> float:
        """Per-leg half-spread cost (fraction of notional) for a name with trailing ``ADV_i``.

        ``half_spread_i = c / sqrt(ADV_i)`` (COST.md Section 5.2). Already the per-leg crossing cost
        (Invariant CONV) — fed to ``round_trip_friction`` with NO further /2. As ``ADV_i -> 0`` this
        diverges; the caller floors ``ADV_i`` at the universe's own liquidity filter (Section 10), it
        is not this method's job to invent a floor. A non-positive ``ADV_i`` (should be
        pre-excluded upstream) yields ``inf`` — an untradeably-illiquid name, never a silent 0.
        """
        if not (adv_i > 0.0 and math.isfinite(adv_i)):
            return math.inf
        return self.c / math.sqrt(adv_i)

    def impact_frac(self, amihud_i: float, notional_leg: float | None = None) -> float:
        """Per-leg impact fraction ``k_amihud * sqrt(Amihud_i * Notional_leg)`` (COST.md Section 5.3).

        ``Notional_leg`` is a LOAD-BEARING dependency of the sqrt-Amihud law (COST.md Section 5.3):
        the impact fraction genuinely scales with the trade's own dollar size. The ``notional_leg is
        None`` path therefore prices at the CALIBRATION reference size (``anchor_notional``, the
        median-anchor ~0.5bp reproduction of Section 5.5) and is intended ONLY for the anchor-
        reproduction / calibration surface — it must NOT be relied on by a per-rebalance caller, which
        always knows and must pass the real ``Notional_leg = |Δw_i| · NAV`` (the round-trip and vector
        surfaces below make ``notional_leg`` REQUIRED so this convenience default can never silently
        misprice a live trade at the fixed calibration size — the footgun the review flagged).
        If ``Amihud_i`` is undefined (NaN / non-positive: fewer than a minimum window of trading days),
        falls back to the class flat research constant ``CRYPTO_FRICTION.impact`` — never zero, never
        an extrapolated thin-sample fit (COST.md Section 10, P5).
        """
        notional = self.anchor_notional if notional_leg is None else notional_leg
        if not (amihud_i > 0.0 and math.isfinite(amihud_i)):
            return CRYPTO_FRICTION.impact
        return self.k_amihud * math.sqrt(amihud_i * notional)

    def round_trip_friction_i(
        self,
        adv_i: float,
        amihud_i: float,
        *,
        notional_leg: float,
        is_shortable: bool | None = None,
        in_short_leg: bool = False,
    ) -> float:
        """Assembled per-name round-trip friction, delegating to ``accounting.round_trip_friction``.

        ``2 * (taker + half_spread_i + impact_frac_i)`` (COST.md Section 5.5). The half-spread and
        impact per-leg components are the DERIVED per-name values (Sections 5.2-5.3); ``taker`` is the
        shared VIP0 truth. The assembly formula itself is NOT re-implemented here — it is
        ``accounting.round_trip_friction(...)`` (Invariant N).

        ``notional_leg`` is REQUIRED (no silent default): the impact term genuinely scales with the
        trade's own dollar size ``|Δw_i| · NAV`` (COST.md Section 5.3), so a per-rebalance caller must
        pass the ACTUAL leg notional — the review's footgun (silently pricing every trade at the fixed
        calibration ``anchor_notional``) is closed by forcing the caller to state the size. To
        reproduce the median-anchor ~13bp calibration point, pass ``notional_leg=model.anchor_notional``
        explicitly (its intent is then visible at the call site).

        SHORTABILITY (Invariant N50, Section 7): if a name is placed in the SHORT leg
        (``in_short_leg=True``) and ``is_shortable is False``, returns ``math.inf`` — the correct
        value (not a convenience), forcing ``settle_week`` to reject the infeasible short
        arithmetically. ``is_shortable=None`` is the current UNIVERSE gap (no borrow metadata
        upstream): the model prices the name at its normal finite liquidity cost and the caller must
        carry the Section 7 caveat for any universe wider than the validated N~50 baseline.
        """
        if in_short_leg and is_shortable is False:
            return math.inf
        hs = self.half_spread(adv_i)
        imp = self.impact_frac(amihud_i, notional_leg)
        # accounting.round_trip_friction is linear and monotone in each per-leg component, so an
        # infinite half-spread (ADV_i <= 0) propagates to an infinite rate — an untradeable name.
        if not math.isfinite(hs):
            return math.inf
        return round_trip_friction(taker=self.taker, half_spread=hs, impact=imp)

    def friction_vector(
        self,
        adv: np.ndarray,
        amihud: np.ndarray,
        *,
        notional_leg: float,
        is_shortable: np.ndarray | None = None,
        in_short_leg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-symbol round-trip friction array for ``settle_week``'s per-symbol cost path.

        Vectorized companion to ``round_trip_friction_i`` over aligned per-name ``adv`` / ``amihud``
        (AMBER trailing statistics, COST.md Section 4). The result feeds ``settle_week(...,
        friction=<this array>)`` which charges each name its own rate on its own weight change
        (COST.md Section 5.6) — reducing EXACTLY to the scalar path when all names share the anchor.
        ``notional_leg`` is REQUIRED (same reason as ``round_trip_friction_i``): the engine-wiring
        caller passes the actual per-rebalance leg notional, never a silent calibration-size default.
        ``is_shortable`` / ``in_short_leg`` are optional per-name bool arrays for the N50 gate.
        """
        adv = np.asarray(adv, dtype=float)
        amihud = np.asarray(amihud, dtype=float)
        n = adv.shape[0]
        if amihud.shape[0] != n:
            raise ValueError(f"adv {adv.shape} and amihud {amihud.shape} must be aligned")
        short_ok = None if is_shortable is None else np.asarray(is_shortable, dtype=bool)
        short_leg = None if in_short_leg is None else np.asarray(in_short_leg, dtype=bool)
        out = np.empty(n, dtype=float)
        for i in range(n):
            out[i] = self.round_trip_friction_i(
                float(adv[i]),
                float(amihud[i]),
                notional_leg=notional_leg,
                is_shortable=(None if short_ok is None else bool(short_ok[i])),
                in_short_leg=(False if short_leg is None else bool(short_leg[i])),
            )
        return out


def amihud_ratio(returns: np.ndarray, dollar_volume: np.ndarray) -> np.ndarray:
    """Amihud (2002) illiquidity ratio per symbol over a trailing window (COST.md Section 5.3).

    ``Amihud_i = (1/D) * sum_d |r_i,d| / $Vol_i,d`` — the average absolute daily return per dollar of
    volume traded over the ``D``-row causal window ``returns`` / ``dollar_volume`` (both (D, n_sym),
    aligned; the caller slices the trailing window ending at day ``t`` inclusive so the estimate is
    ``F_t``-measurable, COST.md Section 0). Days with non-positive / non-finite ``$Vol`` are dropped
    from the mean (never divided-by-zero); a name with no valid day yields NaN, which
    ``SpreadImpactModel.impact_frac`` maps to the ``CRYPTO_FRICTION.impact`` fallback (Section 10).

    Window length ``D`` is the caller's choice (COST.md Section 5.3 shares ``panel.py``'s ``window=30``
    for both ``ADV_i`` and ``Amihud_i`` — one causal window, no second fitted length, P7). Local
    stationarity of each name's illiquidity over the window is the assumption this estimator makes
    (Section 5.3), the same in kind as every trailing-window estimator in the project.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    dv = np.asarray(dollar_volume, dtype=float)
    if r.shape != dv.shape:
        raise ValueError(f"returns {r.shape} and dollar_volume {dv.shape} must be aligned")
    valid = np.isfinite(r) & np.isfinite(dv) & (dv > 0.0)
    per_day = np.where(valid, r / np.where(valid, dv, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        out = np.full(r.shape[1], np.nan)
        has_any = valid.any(axis=0)
        if has_any.any():
            out[has_any] = np.nanmean(per_day[:, has_any], axis=0)
    return out
