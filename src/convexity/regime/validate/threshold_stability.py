"""Out-of-sample trigger stability (REGIME §9, Invariant NOFIT, firewalled — Invariant V).

Are the standardized triggers (``delta_AR``, ``vol_pct``, ``turb_pct``, ``liq_stress``) stable
across folds, or did a fold's best-PnL breakpoint move? A threshold that must be re-tuned per fold is
OVERFIT and is rejected (P5). Crucially, this measures the DISPERSION of the realized escalation
behaviour across folds at the FIXED canonical/BLUE thresholds — it never selects a breakpoint by
maximising a fold's PnL (no code path here looks at PnL at all, NOFIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StabilityReport:
    """Cross-fold dispersion of a trigger's realized escalation rate (REGIME §9)."""

    mean_rate: float
    std_rate: float
    cv: float  # coefficient of variation across folds — low == stable, NOT re-tuned per fold


def stability_across_folds(per_fold_escalation_rate: Sequence[float]) -> StabilityReport:
    """Dispersion of a FIXED threshold's escalation rate across folds (REGIME §9, Invariant NOFIT).

    A low coefficient of variation means the standardized trigger behaves consistently OOS (it is
    not a fold-specific overfit). This consumes only realized escalation rates at fixed thresholds —
    never PnL, never a per-fold breakpoint search.
    """
    rates = np.asarray(per_fold_escalation_rate, dtype=float)
    if rates.size == 0:
        return StabilityReport(0.0, 0.0, 0.0)
    mean = float(rates.mean())
    std = float(rates.std(ddof=0))
    cv = (std / mean) if mean > 0 else 0.0
    return StabilityReport(mean_rate=mean, std_rate=std, cv=cv)
