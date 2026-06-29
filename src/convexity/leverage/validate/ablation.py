"""Ablation — which ceilings actually bind / contribute (LEVERAGE §12, firewalled).

VALIDATION-ONLY (Invariant V). Reports how often each ceiling is the binding one across
a decision set. A ceiling that NEVER binds is redundant or miscalibrated; a ceiling that
ALWAYS binds may be the only real constraint (simplify) — §12.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from convexity.leverage.types import LeverageDecision


def binding_histogram(decisions: Iterable[LeverageDecision]) -> Mapping[str, int]:
    """Count of how often each ceiling/governor/unit was the binding constraint (§12)."""
    return dict(Counter(d.binding_ceiling for d in decisions))


def regime_vs_mctr_attribution(decisions: Iterable[LeverageDecision]) -> Mapping[str, int]:
    """Of the decisions bound by ``portfolio``, split ``regime_heat_cut`` (REGIME g_t) vs
    ``mctr_concentration_cut`` (correlation) so the cross-pillar NODOUBLE audit is
    verifiable from LEVERAGE telemetry (REGIME NODOUBLE test 2, §12)."""
    counts: Counter[str] = Counter()
    for d in decisions:
        if d.binding_ceiling != "portfolio":
            continue
        if "regime_heat_cut" in d.reason_codes:
            counts["regime_heat_cut"] += 1
        elif "mctr_concentration_cut" in d.reason_codes:
            counts["mctr_concentration_cut"] += 1
        else:
            counts["portfolio_unattributed"] += 1
    return dict(counts)


def guardrail_event_counts(decisions: Iterable[LeverageDecision]) -> Mapping[str, int]:
    """Count ``class_cap_capped`` / ``cov_cond_breach`` / lot-rejection events — routine
    occurrence means a ceiling or the covariance pipeline is miscalibrated (§12)."""
    counts: Counter[str] = Counter()
    for d in decisions:
        for code in ("class_cap_capped", "cov_cond_breach", "below_min_unit",
                     "margin_after_rounding", "cluster_trimmed", "exchange_max_capped"):
            if code in d.reason_codes:
                counts[code] += 1
    return dict(counts)
