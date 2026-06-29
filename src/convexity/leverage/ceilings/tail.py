"""TailCeiling — EVT / Expected-Shortfall + gap-survival (LEVERAGE §8.3, CATASTROPHIC).

The catastrophic-layer ceiling truncating the left tail (P5). Two sub-constraints,
take the more binding:

  1. EXPECTED-SHORTFALL budget. Keep ES within budget using EVT (peaks-over-threshold /
     Generalized Pareto fit to the LOSS tail) rather than variance — the tail (gaps,
     crises) is the real leveraged risk (§17). ``L · ES_per_unit <= es_budget``, where
     ES is the per-unit-leverage Expected Shortfall at ``es_level`` (e.g. 97.5%).

  2. GAP SURVIVAL. The hard per-trade stop does NOT survive gaps; only this ceiling
     does. ``L · plausible_adverse_gap <= gap_ruin_threshold`` (RED, e.g. 30-40% of
     committed capital), where the plausible gap is the worst observed overnight/weekend
     gap (or a multiple of the typical gap / class stress gap, §8.3).

EVT is OPTIONAL and EVENT-COUNT GATED (§8.3): the GPD fit is used ONLY when its
exceedance count and diagnostics pass (``GPDFit.ok`` and ``n_exceed >= tail_min_exceed``);
otherwise the ceiling falls back to the conservative rolling empirical ES / class stress
gap. No GPD fit emits a precise ceiling without recording exceedance count, threshold,
shape/scale, and the fallback decision (all in diagnostics).
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import GapModel, GPDFit

_EPS: float = 1e-12


class TailCeiling:
    """L_tail = min(ES-budget ceiling, gap-survival ceiling) (§8.3)."""

    def ceiling(
        self,
        loss_tail: GPDFit,
        gap_hist: GapModel,
        es_budget: float,
        es_level: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (L_tail, diagnostics).

        ``es_budget``/``es_level`` are RED. The ES is a per-unit-leverage LOSS fraction;
        the gap is a per-unit-leverage adverse move fraction. Both bound L from above.
        """
        # --- 1. Expected-Shortfall sub-constraint (EVT or empirical fallback) -----------
        use_evt = loss_tail.ok and loss_tail.n_exceed >= cfg.defaults.tail_min_exceed
        es = loss_tail.expected_shortfall(float(es_level))  # GPD POT when usable, else empirical
        es = max(float(es), _EPS)
        l_es = float(es_budget) / es

        # --- 2. Gap-survival sub-constraint --------------------------------------------
        plausible_gap = gap_hist.plausible_adverse_gap(cfg.defaults.gap_typical_multiple)
        plausible_gap = max(float(plausible_gap), _EPS)
        l_gap = cfg.defaults.gap_ruin_threshold / plausible_gap

        l_tail = min(l_es, l_gap)

        diag: dict[str, object] = {
            "L_es": l_es,
            "L_gap": l_gap,
            "es": es,
            "es_level": float(es_level),
            "es_budget": float(es_budget),
            "use_evt": bool(use_evt),
            "evt_fallback": (not use_evt),
            "gpd_shape": float(loss_tail.shape),
            "gpd_scale": float(loss_tail.scale),
            "gpd_threshold": float(loss_tail.threshold),
            "gpd_n_exceed": int(loss_tail.n_exceed),
            "plausible_adverse_gap": plausible_gap,
            "gap_ruin_threshold": float(cfg.defaults.gap_ruin_threshold),
            "tail_binding": "es" if l_es <= l_gap else "gap",
        }
        return max(l_tail, 0.0), diag
