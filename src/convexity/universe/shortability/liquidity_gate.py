"""Strict short-liquidity gate ``short_liquid_i(t)`` — an INJECTED callable (UNIVERSE.md §4.2).

Being tradeable long and being tradeable SHORT are different operational conditions on a
perpetual-futures venue (§4.1): a long never binds on Binance USDT-M perps, but a short requires the
venue to actually let you open/hold it AND realistic borrow/funding conditions that keep it viable.
``short_liquid_i(t)`` is the STRICTER liquidity order-statistic test — e.g. requiring ``rank_i(t)`` to
clear a HIGHER percentile than ``q_in``, or an absolute trailing-dollar-volume floor independent of the
cross-sectional rank (so even a liquidity-poor cross-section still has a binding absolute test, §4.2).

DISCLOSED LIMITATION (§4.2, stated starkly): this repo has NO short-borrow-availability data, no
historical short-liquidity time series, and no record of which perps have had a short rejected /
restricted / economically forced closed. Every other RED constant in UNIVERSE.md (``W``, ``q_in``,
``q_out``, ``N``) is anchored to ``DV_i(d)``, a statistic this repo computes today, so a defensible
numeric default exists. The short-side threshold is different IN KIND: inventing a plausible number
(e.g. "require ``rank_i(t) >= 0.90``") would be indistinguishable from the anchored constants and would
smuggle a fabricated empirical claim into a document whose central discipline is that every number is
measured, illustrative-and-labeled, or derived. The honest state is an OPEN specification parameter
pending real short-borrow/short-liquidity data, NOT a tuning decision merely deferred.

Build contract (§4.2, §11.1): ``short_liquid`` accepts a ``threshold_fn`` INJECTED at call time — it
NEVER hard-codes a threshold inside this function or ``universe_api.py``. The day real data exists,
calibrating the gate is supplying a new callable + a governance-logged RED promotion, not a code
change to this call site.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ShortLiquidityThreshold(Protocol):
    """Injected per-rebalance short-liquidity test (UNIVERSE.md §4.2). NO default — see module docstring.

    A caller supplies a concrete instance ONLY when real short-borrow/short-liquidity data exists
    (a governance-logged RED promotion, §8's BLUE->RED pattern). Until then no instance should be
    fabricated; ``short_liquid`` raises if none is supplied, rather than inventing a threshold.
    """

    def __call__(self, rank_t: np.ndarray, dv_t: np.ndarray) -> np.ndarray:
        """Return shape (n_symbols,) bool: True where the STRICT short-liquidity test passes.

        ``rank_t`` -- ``rank_i(t)`` (§2.1). ``dv_t`` -- ``DV_i(d)`` (§2.1), for any absolute-floor component.
        Must be STRICTLY stricter than the long floor of §2 (that is the invariant §4.2 fixes today).
        """
        ...


def short_liquid(
    rank_t: np.ndarray,
    dv_t: np.ndarray,
    *,
    threshold_fn: ShortLiquidityThreshold,
) -> np.ndarray:
    """Per-symbol strict short-liquidity gate ``short_liquid_i(t)`` (UNIVERSE.md §4.2).

    ``rank_t`` -- shape (n_symbols,), same ``rank_i(t)`` as ``hysteresis_step`` (§2.1).
    ``dv_t`` -- shape (n_symbols,), this rebalance's ``DV_i(d)`` (§2.1), for any absolute-floor component
                of ``threshold_fn`` (§4.2's example: a floor independent of cross-sectional rank).
    ``threshold_fn`` -- an INJECTED callable, never a threshold hard-coded inside this function or
                        ``universe_api.py`` (§4.2 disclosed-limitation requirement: no numeric
                        short-liquidity threshold is fabricated; supplying one is a future, data-backed,
                        governance-logged RED promotion, not a code change to this signature).

    Returns shape (n_symbols,) bool. This is ONE of THREE ANDed terms (§4.2:
    ``1{i∈U_t} · short_permitted · short_liquid``) — callers compose it with ``base_eligible &
    hysteresis_state``; this function does NOT re-check them (§4.2), so it never admits a name the base
    gate excluded. A NaN rank/DV yields False (P6: cannot short a name that cannot be ranked).
    """
    if threshold_fn is None:  # pragma: no cover - defensive; the type contract already forbids None
        raise NotImplementedError(
            "short_liquid requires an injected threshold_fn; ShortLiquidityThreshold has no default "
            "(UNIVERSE.md §4.2 disclosed limitation — no short-borrow data exists; do not fabricate a threshold)"
        )
    rank_t = np.asarray(rank_t, dtype=float)
    dv_t = np.asarray(dv_t, dtype=float)
    passed = np.asarray(threshold_fn(rank_t, dv_t), dtype=bool)
    if passed.shape != rank_t.shape:
        raise ValueError(
            f"threshold_fn returned shape {passed.shape}, expected {rank_t.shape} (UNIVERSE.md §4.2)"
        )
    # P6: a name with a NaN rank or NaN DV cannot pass the short-liquidity test regardless of the
    # injected callable (a name that cannot be liquidity-ranked fails the name closed, §2.4).
    valid = np.isfinite(rank_t) & np.isfinite(dv_t)
    return passed & valid


__all__ = ["ShortLiquidityThreshold", "short_liquid"]
