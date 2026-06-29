"""AdaptiveKellyFraction — rule-based φ in {0.5, 1/3} on observables (LEVERAGE §5.2, Inv K).

The Kelly fraction is time-varying but its trigger is a MEASURABLE risk quantity, never
discretion (§5.2):

  * φ = 0.5 (Half Kelly) under normal conditions;
  * φ = 1/3 when parameter/market uncertainty is unusually HIGH, where "high" is defined
    by OBSERVABLES — realized-vol percentile above a RED threshold, OR correlation-matrix
    instability above a threshold, OR recent jump-intensity elevated;
  * return toward 0.5 only after the observable subsides, with HYSTERESIS to avoid
    flapping (a separate, lower "lo" threshold per observable).

Inv K: no code path produces φ > 0.5; no discretionary / edge-derived input changes φ.
The trigger is always a LOGGED number (returned in the reason mapping). Hysteresis needs
the previous φ; the engine threads it through so the switch is deterministic.
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig


class AdaptiveKellyFraction:
    """φ ∈ {0.5, 1/3} switched by logged observables with hysteresis (§5.2)."""

    def fraction(
        self,
        vol_percentile: float,
        corr_instability: float,
        jump_intensity: float,
        cfg: ResolvedConfig,
        prev_phi: float | None = None,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (φ, reason). φ in {base 0.5, stressed 1/3}. ``prev_phi`` carries the
        hysteresis state: once stressed, an observable must fall below its LOWER ('lo')
        threshold to relax back to base. No discretionary input (Inv K)."""
        d = cfg.defaults
        base, stressed = d.kelly_phi_base, d.kelly_phi_stressed

        was_stressed = prev_phi is not None and abs(prev_phi - stressed) < 1e-9

        if was_stressed:
            # relax to base ONLY when every observable is below its lower (hysteresis) bound
            relax = (
                vol_percentile < d.kelly_vol_pct_lo
                and corr_instability < d.kelly_corr_instability_lo
                and jump_intensity < d.kelly_jump_intensity_lo
            )
            phi = base if relax else stressed
            triggered = not relax
        else:
            triggered = (
                vol_percentile >= d.kelly_vol_pct_hi
                or corr_instability >= d.kelly_corr_instability_hi
                or jump_intensity >= d.kelly_jump_intensity_hi
            )
            phi = stressed if triggered else base

        reason: dict[str, object] = {
            "phi": phi,
            "stressed": bool(abs(phi - stressed) < 1e-9),
            "vol_percentile": float(vol_percentile),
            "corr_instability": float(corr_instability),
            "jump_intensity": float(jump_intensity),
            "vol_trigger": vol_percentile >= d.kelly_vol_pct_hi,
            "corr_trigger": corr_instability >= d.kelly_corr_instability_hi,
            "jump_trigger": jump_intensity >= d.kelly_jump_intensity_hi,
            "hysteresis_was_stressed": bool(was_stressed),
            "triggered": bool(triggered),
        }
        return phi, reason
