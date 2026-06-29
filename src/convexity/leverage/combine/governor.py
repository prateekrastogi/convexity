"""DrawdownGovernor — graduated equity-drawdown multiplier (LEVERAGE §6, Inv G/DDCAUSAL).

The 43% portfolio limit implemented as a GRADUATED risk multiplier on the final leverage
of every position, NOT a single kill-switch. Drawdown is peak-to-current on portfolio
equity, where the peak is a TRAILING running max (``F_t``-measurable, no look-ahead —
Inv CAUSAL-L). ``drawdown = 1 − equity/peak``.

    portfolio drawdown    risk multiplier
       0% – 10%               1.00
      10% – 20%               0.75
      20% – 30%               0.50
      30% – 40%               0.25
      40% – 43%               LINEAR RAMP 0.25 -> 0.00
         >= 43%               0.00  (flat the book; stop opening)

The 40–43% band is a PINNED LINEAR ramp: ``mult(dd) = 0.25·(0.43−dd)/(0.43−0.40)`` for
dd ∈ [0.40, 0.43), clamped to 0.00 at dd ≥ 0.43 (worked: dd=0.42 → 0.0833, dd=0.415 →
0.125). This de-risks SMOOTHLY as equity falls so 43% is engineered around, not a cliff.

BOUNDARY CONVENTION (pinned, §15 "steps EXACTLY at 10/20/30/40%"). Each breakpoint is
INCLUSIVE of the band it OPENS — i.e. the bands are ``[0.00,0.10) → 1.00``,
``[0.10,0.20) → 0.75``, ``[0.20,0.30) → 0.50``, ``[0.30,0.40) → 0.25``,
``[0.40,0.43) → ramp``, ``[0.43,∞) → 0.00``. So drawdown reaching EXACTLY a breakpoint
takes the LOWER (more conservative) multiplier: ``dd == 0.10`` → 0.75, not 1.00. This is
the de-risking-safe choice (a breakpoint hit steps DOWN, never holds the higher band).
``dd`` is computed as ``1 − equity/peak`` and is float-fragile at the literal breakpoints
(round-off can land just below 0.10 and wrongly hold 1.00); the comparison is therefore
EPSILON-ROBUST: ``dd`` is treated as having reached a breakpoint ``bp`` when
``dd >= bp − _DD_EPS``. ``_DD_EPS`` (1e-9) is far smaller than the 10-pp band width so it
only absorbs equity/peak round-off, never collapses adjacent bands.

Inv G: the governor ONLY reduces, never increases, exposure; it cannot lift leverage
above the min() of ceilings. It reads peak-to-current EQUITY ONLY and NEVER a regime
measure (the disjoint axis vs g_t, P8). The breakpoints are RED, economically-derived
(geometric de-risking), and OOS-paid in SYSTEM's multiplicity budget (Inv DDCAUSAL).
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig

# Epsilon-robust breakpoint tolerance (LEVERAGE §6/§15). ``dd = 1 − equity/peak`` is
# float-fragile at the literal 10/20/30/40% breakpoints; treat dd as having REACHED a
# breakpoint bp when dd >= bp − _DD_EPS. Far smaller than the 10-pp band width, so it
# only absorbs equity/peak round-off and never merges adjacent bands.
_DD_EPS: float = 1e-9


class DrawdownGovernor:
    """Graduated drawdown risk multiplier in [0,1], applied to final leverage (§6)."""

    def multiplier(
        self,
        portfolio_peak: float,
        portfolio_equity: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (risk_multiplier in [0,1], reason). >= 43% drawdown returns 0.0 (flat,
        halt-open). The peak MUST be the trailing running max supplied by the caller
        (Inv CAUSAL-L) — the governor does not compute it from a future series.
        """
        peak = float(portfolio_peak)
        equity = float(portfolio_equity)
        dd = 0.0 if peak <= 0.0 else max(1.0 - equity / peak, 0.0)

        d = cfg.defaults
        breakpoints = d.governor_breakpoints  # (0.10, 0.20, 0.30, 0.40)
        mults = d.governor_multipliers  # (1.00, 0.75, 0.50, 0.25)
        lo, hi = d.governor_ramp_lo, d.governor_ramp_hi  # 0.40, 0.43

        # Bands are INCLUSIVE of the breakpoint they OPEN, with an epsilon-robust
        # comparison (dd reaches bp when dd >= bp − _DD_EPS), so dd landing EXACTLY on a
        # breakpoint steps DOWN to the lower (more conservative) multiplier (§6 BOUNDARY
        # CONVENTION; §15 "steps EXACTLY at breakpoints"). De-risking-safe by construction.
        if dd >= hi - _DD_EPS:
            mult = 0.0
            band = ">=ramp_hi(flat)"
        elif dd >= lo - _DD_EPS:
            # pinned linear ramp 0.25 -> 0.00 across [lo, hi). Snap dd to lo when it is
            # within _DD_EPS of the breakpoint (either side) so dd EXACTLY at lo (modulo
            # equity/peak round-off) yields the band's top multiplier 0.25 exactly, rather
            # than a sliver above/below it (§6 boundary convention, §15 exact-breakpoint).
            dd_eff = lo if abs(dd - lo) <= _DD_EPS else dd
            mult = mults[-1] * (hi - dd_eff) / (hi - lo)
            band = "ramp"
        else:
            # step ladder: pick the multiplier for the band dd falls in. Crossing (or
            # landing exactly on) the i-th breakpoint drops to its SUCCESSOR multiplier
            # (10% -> 0.75, 20% -> 0.50, ...).
            mult = mults[0]
            band = f"[0,{breakpoints[0]})"
            for i, bp in enumerate(breakpoints):
                if dd >= bp - _DD_EPS:
                    nxt = breakpoints[i + 1] if i + 1 < len(breakpoints) else lo
                    mult = mults[i + 1] if i + 1 < len(mults) else mults[-1]
                    band = f"[{bp},{nxt})"
                else:
                    break

        mult = min(max(mult, 0.0), 1.0)  # Inv G: clamped to [0,1], only de-risks
        reason: dict[str, object] = {
            "drawdown": dd,
            "multiplier": mult,
            "band": band,
            "peak": peak,
            "equity": equity,
            "flat": dd >= hi,
        }
        return mult, reason
