"""Reports — layer-attribution / lot-rejection / regime-vs-MCTR (LEVERAGE §11/§12, firewalled).

VALIDATION-ONLY (Invariant V). Assembles the §11 normal/catastrophic/operational layer
attribution and the §12 required reports into plain dicts (no rendering dependency).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from convexity.leverage.types import LeverageDecision
from convexity.leverage.validate.ablation import (
    binding_histogram,
    guardrail_event_counts,
    regime_vs_mctr_attribution,
)

# §11 layer attribution of the seven ceilings.
_NORMAL = {"stop", "vol", "kelly"}
_CATASTROPHIC = {"tail", "portfolio"}
_OPERATIONAL = {"margin", "liquidity", "unit", "governor"}


def layer_attribution(decisions: Iterable[LeverageDecision]) -> Mapping[str, int]:
    """Fraction (count) of decisions bound by each §11 layer (normal/catastrophic/operational)."""
    out = {"normal": 0, "catastrophic": 0, "operational": 0}
    for d in decisions:
        if d.binding_ceiling in _NORMAL:
            out["normal"] += 1
        elif d.binding_ceiling in _CATASTROPHIC:
            out["catastrophic"] += 1
        else:
            out["operational"] += 1
    return out


def full_report(decisions: list[LeverageDecision]) -> Mapping[str, object]:
    """The §12 report bundle: binding histogram, layer attribution, regime-vs-MCTR split,
    and guardrail event counts (class_cap / cov_cond_breach / lot rejections)."""
    return {
        "binding_histogram": binding_histogram(decisions),
        "layer_attribution": layer_attribution(decisions),
        "regime_vs_mctr": regime_vs_mctr_attribution(decisions),
        "guardrail_events": guardrail_event_counts(decisions),
        "n_decisions": len(decisions),
    }
