"""PortfolioCeiling — MCTR vs the regime-scaled g_t·H_book (LEVERAGE §7, Inv R/P/COND-L).

The CATASTROPHIC-layer correlation-concentration ceiling AND the SOLE channel through
which REGIME's macro multiplier ``g_t`` enters LEVERAGE (P7). Under portfolio margin
this is the load-bearing constraint (P4): correlated positions (NIFTY/BANKNIFTY/bank
stocks) get a LOWER ceiling because their MARGINAL CONTRIBUTION TO RISK is high;
diversifying positions (Gold/USDINR/NatGas/NIFTY) get a higher one.

The book-heat budget is ``H_book_eff = g_t · H_book`` (P7, Inv R): ``g_t ∈ [0.25,1.0]``
(REGIME Inv CAP) so it can only LOWER the budget and only de-risk. ``g_t`` is applied
HERE and NOWHERE ELSE — not to L_vol, not to the Kelly fraction. When the regime
scaling is the active cut, the reason code is ``'regime_heat_cut'`` (vs
``'mctr_concentration_cut'`` for a correlation-MCTR cut) so telemetry can attribute a
gross cut to REGIME vs MCTR (the NODOUBLE audit test 2).

Uses covariance for RISK ONLY — never return-optimized Σ⁻¹μ (Inv P; μ is unmeasurable
and inversion amplifies its error). The covariance is Ledoit–Wolf-shrunk and
condition-gated BEFORE any MCTR (Inv COND-L); a near-singular Σ falls back to a robust
class-level heat budget and flags ``cov_cond_breach``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.leverage.allocate.risk_budget import condition_covariance, mctr, portfolio_vol
from convexity.leverage.config.resolve import ResolvedConfig

_EPS: float = 1e-12


class PortfolioCeiling:
    """L_portfolio from the instrument's MCTR share against g_t·H_book (§7, P7)."""

    @staticmethod
    def _largest_root(a: float, b: float, c: float) -> float:
        """Largest non-negative root x>=0 of a·x² + b·x + c <= 0 (the heat quadratic).

        ``c = pv² − H²`` is <= 0 when the book already fits the budget; the feasible
        region is [0, x+] with x+ the positive quadratic root, so x_max = x+ (clamped
        >= 0). If the book is ALREADY over budget (c > 0) there is no room: return 0.
        """
        if c > 0.0:  # book already exceeds the heat budget -> no room for more risk
            return 0.0
        if a <= _EPS:  # degenerate (zero own-variance): linear b·x + c <= 0
            return max(-c / b, 0.0) if b > _EPS else 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0.0
        x_plus = (-b + float(np.sqrt(disc))) / (2.0 * a)
        return max(x_plus, 0.0)

    def ceiling(
        self,
        instrument: str,
        candidate_weights: Mapping[str, float],
        cov: np.ndarray,
        cov_index: tuple[str, ...],
        g_t: float,
        position_margin: float,
        account_equity: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (L_portfolio, diagnostics).

        ``candidate_weights`` is the current/candidate book in PORTFOLIO-WEIGHT terms
        (notional / equity), keyed by instrument, aligned to ``cov_index``. ``g_t`` is
        REGIME's multiplier; ``H_book_eff = g_t·H_book`` (P7). The ceiling translates the
        instrument's allowed MCTR share into a leverage given its weight-per-unit
        (= position_margin / account_equity).
        """
        h_book = cfg.defaults.h_book
        h_book_eff = float(g_t) * h_book  # P7: the ONLY place g_t enters LEVERAGE

        cond = condition_covariance(
            cov,
            max_condition=cfg.defaults.cov_max_condition,
            min_eig_ratio=cfg.defaults.cov_min_eig_ratio,
        )

        idx_of = {name: i for i, name in enumerate(cov_index)}
        n = len(cov_index)
        w = np.zeros(n, dtype=float)
        for name, wt in candidate_weights.items():
            if name in idx_of:
                w[idx_of[name]] = float(wt)

        # weight-per-unit-leverage for THIS instrument (notional == L·position_margin,
        # so its portfolio weight is L · position_margin / account_equity).
        eq = max(float(account_equity), _EPS)
        weight_per_unit_lev = max(float(position_margin), _EPS) / eq

        # Covariance breach -> robust class-level heat budget fallback (Inv COND-L):
        # treat the instrument as standalone (its own vol), no netting benefit.
        if cond.breached or instrument not in idx_of:
            own_vol = float(np.sqrt(max(cond.cov[idx_of[instrument], idx_of[instrument]], _EPS))) \
                if instrument in idx_of else 1.0
            # standalone heat budget: L·weight_per_unit·own_vol <= h_book_eff
            denom = weight_per_unit_lev * own_vol
            l_port = h_book_eff / denom if denom > _EPS else 0.0
            return max(l_port, 0.0), {
                "reason_code": "regime_heat_cut" if float(g_t) < 1.0 else "mctr_concentration_cut",
                "cov_cond_breach": bool(cond.breached),
                "h_book_eff": h_book_eff,
                "g_t": float(g_t),
                "fallback_standalone": True,
                "condition_number": cond.condition_number,
            }

        i = idx_of[instrument]
        pv = portfolio_vol(w, cond.cov)
        contrib = mctr(w, cond.cov)

        # The instrument's CURRENT MCTR share of book risk (for the concentration cap).
        current_share = float(contrib[i] / pv) if pv > _EPS else 0.0

        # Two heat sub-constraints; take the more binding:
        #  (a) total book heat: scaling THIS instrument's leverage scales its weight; we
        #      cap the book MCTR sum (== pv) at h_book_eff. Per unit leverage, its risk
        #      contribution rate is its standalone marginal vol.
        sigma_i = float(np.sqrt(max(cond.cov[i, i], _EPS)))
        # (a) total book-heat constraint, EXACT. Adding instrument i at portfolio weight a
        # makes the book vol sqrt(pv^2 + 2a·(Σw)_i + a^2·σ_i^2). Cap that at H_book_eff and
        # solve the quadratic σ_i^2·a^2 + 2(Σw)_i·a + (pv^2 - H^2) <= 0 for the largest a;
        # then L = a / weight_per_unit_lev. The CROSS term 2a·(Σw)_i penalises a correlated
        # add (large (Σw)_i) and rewards a diversifying one ((Σw)_i ~ 0) — exactly the §7
        # "correlated positions get a lower ceiling" behaviour, no ad-hoc marginal rate.
        sigma_w_i = float(cond.cov[i] @ w)  # (Σw)_i — the correlation coupling to the book
        h2 = h_book_eff * h_book_eff
        qa = sigma_i * sigma_i
        qb = 2.0 * sigma_w_i
        qc = pv * pv - h2
        a_max = self._largest_root(qa, qb, qc)
        l_total_heat = a_max / weight_per_unit_lev if weight_per_unit_lev > _EPS else 0.0

        #  (b) single-instrument MCTR share cap (no correlated cluster dominates, §7).
        share_budget = cfg.defaults.mctr_share_cap * h_book_eff
        l_share = share_budget / (weight_per_unit_lev * sigma_i) if sigma_i > _EPS else 0.0

        l_port = min(l_total_heat, l_share)

        # Attribution: a regime cut (g_t<1 made the binding budget) vs a correlation cut.
        # If at g_t=1 the same min() would be slack-er, the cut is regime-driven.
        a_max_nr = self._largest_root(qa, qb, pv * pv - h_book * h_book)
        l_port_no_regime_total = a_max_nr / weight_per_unit_lev if weight_per_unit_lev > _EPS else 0.0
        l_port_no_regime_share = (cfg.defaults.mctr_share_cap * h_book) / (
            weight_per_unit_lev * sigma_i
        ) if sigma_i > _EPS else 0.0
        l_port_no_regime = min(l_port_no_regime_total, l_port_no_regime_share)
        regime_drove_cut = float(g_t) < 1.0 and l_port < l_port_no_regime - _EPS
        reason_code = "regime_heat_cut" if regime_drove_cut else "mctr_concentration_cut"

        diag: dict[str, object] = {
            "reason_code": reason_code,
            "cov_cond_breach": False,
            "h_book": float(h_book),
            "h_book_eff": h_book_eff,
            "g_t": float(g_t),
            "portfolio_vol": pv,
            "current_mctr_share": current_share,
            "L_total_heat": l_total_heat,
            "L_share": l_share,
            "L_no_regime": l_port_no_regime,
            "shrinkage": cond.shrinkage,
            "condition_number": cond.condition_number,
        }
        return max(l_port, 0.0), diag
