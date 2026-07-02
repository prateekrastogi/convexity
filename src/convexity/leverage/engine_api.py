"""The LEVERAGE pillar's ONE public method (LEVERAGE.md §9, P8/Invariant PROV).

``LeverageEngine.size_book(...) -> LeverageDecision`` is the single entry point: given the book's
weekly net-of-cost history, the search-family trial Sharpes, the per-instrument capacity inputs and
the active fractional-Kelly dial, it returns the ONE scalar ``G_t`` and its full ceiling
decomposition (§6.1). It never chooses instruments, sides, or timing (P8) — those are SELECTION's
job; LEVERAGE only scales the already-built reference-gross book.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from convexity.leverage.ceilings.capacity import capacity_ceiling
from convexity.leverage.ceilings.governor import governor_state
from convexity.leverage.ceilings.kelly_vol_target import kelly_vol_target_term
from convexity.leverage.combine.engine import compose_g_t
from convexity.leverage.config.resolve import ResolvedLeverageConfig, resolve_config
from convexity.leverage.edge.lcb import sr_lcb_t
from convexity.leverage.types import CapacityInput, LeverageDecision
from convexity.leverage.vol.forecast import sigma_hat_t


class LeverageEngine:
    """Sizer of the whole book — emits ``G_t`` (LEVERAGE.md §0, §6.1, Invariant PROV).

    ONE public method, ``size_book``. The engine holds only the resolved RED config; all TEAL/AMBER
    inputs (weekly returns, trial Sharpes, capacity inputs, the ratcheted ``c_t``) arrive per call.
    """

    def __init__(self, config: ResolvedLeverageConfig | None = None) -> None:
        self.config = config or resolve_config()

    def size_book(
        self,
        net_weekly: np.ndarray,
        trial_sharpes_ann: np.ndarray,
        capacity_inputs: Sequence[CapacityInput],
        c_t: float,
        *,
        n_trials_effective: float | None = None,
        k_distinct: int | None = None,
    ) -> LeverageDecision:
        """Compute ``G_t`` for next week's rebalance (LEVERAGE.md §6.1 composition).

        Pipeline (each step delegates to its owning module; §6.1):
          1. ``σ̂_t`` — causal weekly EWMA book vol with the §4.2 conservative floor.
          2. ``SR_LCB_t`` — the DSR-gated block-bootstrap CI lower bound (§2.6, Invariant K-CI/K-LCB).
          3. Kelly/vol-target term ``c_t · SR_LCB_t / σ̂_t`` (§2.5).
          4. Capacity ceiling ``min_i(κ·ADV_i/|w_i^{ref}|)`` (§5.1).
          5. Drawdown governor ``m_t`` from the same weekly equity path (§6.4).
          6. ``min()`` composition + ``binding_ceiling`` (§6.1).

        ``c_t`` is the active fractional-Kelly dial resolved by the ``ratchet/`` state machine
        (§6.2–§6.3) — passed in, not chosen here, so ``size_book`` stays a pure function of its
        inputs. ``G_t`` is estimated at the fixed reference gross of 1.0 and applied exactly once at
        the position-translation boundary — no ``G_t²`` double-multiply is possible (§8).
        """
        defaults = self.config.defaults
        vol = sigma_hat_t(net_weekly, defaults)
        edge = sr_lcb_t(
            net_weekly,
            trial_sharpes_ann,
            defaults,
            n_trials_effective=n_trials_effective,
            k_distinct=k_distinct,
        )
        g_target = kelly_vol_target_term(c_t, edge.sr_lcb, vol.sigma_hat)
        capacity = capacity_ceiling(capacity_inputs, defaults)
        governor = governor_state(net_weekly, defaults)
        return compose_g_t(g_target, capacity, governor, defaults.g_max)
