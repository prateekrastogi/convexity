"""``z_k`` standardization of the weekly-return stream (``docs/specs/OPS.md`` sec 8 ``streams/whiten.py``).

sec 4.2 standardizes the realized weekly net return into the CUSUM innovation the kill-switch monitors.
sec 8's directory layout puts this adapter in ``ops/streams/whiten.py`` (the ``streams/`` package
adapts realized facts into the monitored series of sec 2); the CUSUM recursion that consumes ``z_k``
is :mod:`convexity.ops.detectors.cusum` and the state machine is :mod:`convexity.ops.brake.kill_switch`.
"""

from __future__ import annotations


def whiten(r_k: float, *, mu_0: float, sigma_0: float) -> float:
    """Standardize a realized weekly net return into the CUSUM innovation ``z_k`` (OPS.md sec 4.2).

    ``z_k = (r_k - mu_0) / sigma_0`` — mean 0 / var 1 under the validated-null (healthy-strategy)
    hypothesis. ``r_k`` is ``WeekResult.net`` (realized, post-settlement, sec 2). ``mu_0``/``sigma_0``
    are derived from EDGE.md sec b's honest column: ``SR_0 = 0.85`` (lower Sharpe bound, [MEASURED])
    and ``ann_vol_0 = 0.28`` ([DERIVED] from EDGE.md's lower (Sharpe 0.85, CAGR 0.20) pair via the
    Sharpe identity + geometric-arithmetic return relation, OPS.md sec 4.2 — EDGE.md prints no literal
    vol; see OpsDefaults.ann_vol_0 and test_ann_vol_0_is_derived_from_edge_sharpe_and_cagr).
    """
    if sigma_0 <= 0.0:
        raise ValueError(f"sigma_0 must be > 0, got {sigma_0}")
    return (r_k - mu_0) / sigma_0
