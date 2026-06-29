"""AutoTuner — governed CLASS-level preset selection (TREND §15 autotune).

Auto-tuning is allowed ONLY as governed preset selection (§15): pick the best validated CLASS-level
preset from the finite BLUE grid by an out-of-sample objective, never a per-symbol P&L optimization
and never online self-retuning (§4 forbidden). Returns the selected preset name; the registry records
the trial with its config hash + window.
"""

from __future__ import annotations

import numpy as np

from convexity.trend.tune.search_space import TRAIL_PRESETS


class AutoTuner:
    """Class-level preset selection over the finite BLUE grid (§15)."""

    def select_trail_preset(self, oos_scores: dict[str, float]) -> str:
        """Pick the trail preset with the best out-of-sample objective (§15).

        ``oos_scores`` maps a preset name -> its validated OOS net-E[Pi] (or composite). Ties break to
        the simpler (tighter) preset to avoid over-loosening. Unknown presets are ignored.
        """
        candidates = {k: v for k, v in oos_scores.items() if k in TRAIL_PRESETS}
        if not candidates:
            return "balanced"
        best = max(candidates.values())
        # tie-break: prefer the earlier (tighter) preset in TRAIL_PRESETS order.
        for name in TRAIL_PRESETS:
            if name in candidates and np.isclose(candidates[name], best):
                return name
        return "balanced"
