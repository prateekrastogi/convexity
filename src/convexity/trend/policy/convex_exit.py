"""ConvexExit — the asymmetric convex trail + gap survival (TREND §10, INV-CVX / Inv L).

The trail boundary B_t = b(M_t, V_t, G_t, clock, vol_bucket, liq_bucket) is asymmetric (§1.5):
TIGHT early (cut failed trades), WIDER as ``open_profit_r`` grows (earn the right to be loose), with
extra widening after a jump_with / vol expansion if trend evidence has not deteriorated. The trail
NEVER tightens because of future MFE labels (Inv H).

The trail MULTIPLE is monotone in open profit (§10):

    profit_r <= 0.5        -> tight     multiple
    0.5 < profit_r <= 2.0  -> normal    multiple
    2.0 < profit_r <= 4.0  -> loose     multiple
    profit_r > 4.0         -> tail-preservation multiple

Because the multiple is non-decreasing in profit, the marginal giveback g'(M) is non-increasing over
the loose/tail regions, so the realized round-trip payoff Pi(M) is convex (option-like) and bounded
below by the initial-risk cut -R0 (INV-CVX, §1.5a; tested in §18). ``initial()`` uses BREAKOUT's
``initial_invalidation`` when supplied and only falls back to a causal vol/swing stop when absent
(Inv O / §6c).
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Position
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.types import BeliefState, ExitState, VolState


class ConvexExit:
    """Asymmetric convex trail + gap survival (§10)."""

    # ---- trail multiple schedule (monotone in profit_r, INV-CVX) --------------------------
    def trail_multiple(self, profit_r: float, cfg: ResolvedConfig) -> float:
        """Return the §10 trail multiple for the open-profit band (monotone non-decreasing)."""
        b0, b1, b2 = cfg.trail_breakpoints_r  # (0.5, 2.0, 4.0)
        tight, normal, loose, tail = cfg.trail_multipliers
        if profit_r <= b0:
            return tight
        if profit_r <= b1:
            return normal
        if profit_r <= b2:
            return loose
        return tail

    # ---- initial state --------------------------------------------------------------------
    def initial(
        self, bars: Bars, position: Position, vol: VolState, cfg: ResolvedConfig
    ) -> ExitState:
        """Create the initial invalidation + trail state (§10).

        Inv O / §6c: use ``position.initial_invalidation`` UNCHANGED when supplied; only fall back to
        a causal vol/swing stop when it is absent. The initial risk must exceed the tick/spread noise
        floor (§10).
        """
        entry = position.entry_price
        side = float(position.side)
        spec = cfg.instrument
        noise_floor = (spec.tick_size + 0.0) * cfg.defaults.min_stop_tick_mult

        if position.initial_invalidation is not None:
            invalidation = float(position.initial_invalidation)  # BREAKOUT-owned (Inv O)
            risk = abs(entry - invalidation)
        else:
            # Degenerate causal vol fallback (NOT an entry model, §6c): k * sigma_price below entry.
            tight_mult = cfg.trail_multipliers[0]
            dist = max(tight_mult * vol.sigma_price, noise_floor)
            invalidation = entry - side * dist
            risk = dist

        risk = max(risk, noise_floor)  # enforce the noise floor (§10)
        if position.initial_risk is not None and position.initial_risk > 0:
            risk = max(float(position.initial_risk), noise_floor)

        return ExitState(
            trail_level=invalidation,
            trail_multiple=cfg.trail_multipliers[0],
            high_watermark=entry,
            low_watermark=entry,
            open_profit_r=0.0,
            initial_risk=float(risk),
            invalidation=invalidation,
            size_fraction=1.0,
        )

    # ---- per-bar update -------------------------------------------------------------------
    def update(
        self,
        bars: Bars,
        idx: int,
        position: Position,
        belief: BeliefState,
        state: ExitState,
        cfg: ResolvedConfig,
    ) -> ExitState:
        """Update the asymmetric convex trail (§10). Tight early, wider with profit; widen on
        jump_with / vol expansion when trend evidence has not deteriorated. Never tighten on future
        labels (Inv H). The trail RATCHETS in the favorable direction only (monotone B_t)."""
        side = float(position.side)
        entry = position.entry_price
        price = bars.c[idx]

        # favorable high/low watermark (causal): ratchet only in the favorable direction.
        hwm = max(state.high_watermark, price) if side > 0 else state.high_watermark
        lwm = min(state.low_watermark, price) if side < 0 else state.low_watermark

        # open profit in R units.
        risk = state.initial_risk
        open_profit_r = (side * (price - entry)) / risk if risk > 0 else 0.0

        mult = self.trail_multiple(open_profit_r, cfg)
        # extra widening after a jump_with or material vol expansion, if not deteriorating.
        if (belief.jump_with or belief.vol.vol_bucket >= cfg.defaults.n_vol_buckets - 1) and not (
            belief.seq_alarm or belief.jump_against
        ):
            mult *= cfg.defaults.jump_with_widen

        # Trail distance is in PRICE units, scaled by the PATH volatility (integrated var since
        # entry, monotone) so a single quiet early bar cannot collapse the trail (§9 rationale). It
        # is FLOORED by the initial risk R0 so the early ("tight") regime keeps the stop near the
        # entry invalidation rather than ratcheting tight under price — the convex left-tail cut is
        # -R0 by construction (§1.5a / INV-CVX), not a microstructure-tight stop.
        path_sigma_price = float(np.sqrt(max(belief.vol.integrated_var_since_entry, 0.0))) * price
        vol_dist = mult * max(path_sigma_price, belief.vol.sigma_price)
        trail_dist = max(vol_dist, risk, cfg.instrument.tick_size)

        # Trail anchored to the favorable extreme; ratchet only in the favorable direction, and
        # never below the initial (BREAKOUT-owned) invalidation floor on the protective side.
        if side > 0:
            new_trail = hwm - trail_dist
            new_trail = max(new_trail, state.trail_level)  # never loosen the protective level
        else:
            new_trail = lwm + trail_dist
            new_trail = min(new_trail, state.trail_level)

        return ExitState(
            trail_level=float(new_trail),
            trail_multiple=float(mult),
            high_watermark=float(hwm),
            low_watermark=float(lwm),
            open_profit_r=float(open_profit_r),
            initial_risk=float(risk),
            invalidation=float(new_trail),
            size_fraction=state.size_fraction,
        )

    # ---- breach ---------------------------------------------------------------------------
    def breached(
        self, bars: Bars, idx: int, position: Position, state: ExitState
    ) -> bool:
        """Return whether the causal trail/invalidation was breached at ``idx`` (§10).

        Uses the bar low (long) / high (short) so an intrabar penetration of the resting stop is
        detected — the venue would have triggered. Causal: only bar ``idx`` data.
        """
        side = float(position.side)
        if side > 0:
            return bool(bars.l[idx] <= state.trail_level)
        return bool(bars.h[idx] >= state.trail_level)

    # ---- gap survival ---------------------------------------------------------------------
    def gap_survival_fraction(self, vol: VolState, cfg: ResolvedConfig) -> float:
        """Return the max size fraction that survives an adverse gap stress (§10).

        Not a portfolio sizer: it only prevents the policy from pretending stops protect through
        gaps. Stress = ``gap_survival_sigmas`` gap-sigmas of adverse move; the surviving fraction is
        the share of risk that stress would not exhaust, floored at ``gap_survival_min_fraction``.
        """
        gap_sigma = float(np.sqrt(max(vol.gap_var, 0.0)))
        if gap_sigma <= 0:
            return 1.0
        stress = cfg.gap_survival_sigmas * gap_sigma
        # surviving fraction shrinks as stress grows relative to one sigma_log of headroom.
        headroom = max(vol.sigma_log, 1e-9)
        frac = float(np.clip(headroom / max(stress, 1e-9), cfg.defaults.gap_survival_min_fraction, 1.0))
        return frac
