"""Firewalled realized weekly book-return labels (LEVERAGE.md §9 layout, §10 Invariant V).

A ``validate/`` module: it constructs the REALIZED weekly book-return series used to SCORE a
sizing decision after the fact (backtest/ARL harnesses), and it is firewalled from every live
layer (``vol/``, ``edge/``, ``ceilings/``, ``ratchet/``, ``combine/``) — no live layer may import
it (Invariant V, §10). The one job of a label is to be strictly FUTURE relative to the decision's
information set: the return realized over ``(t, t+1]`` labels the ``G_t`` decided at ``t``, so a
decision is never scored on data it could see when it was made (the causal/labeling firewall the
whole repo enforces).

This mirrors ``research/xsec/accounting.py``'s realized-return series but is the LEVERAGE-local,
future-only view the acceptance-gate harnesses read; it never leaks a future label into a live
estimator.
"""

from __future__ import annotations

import numpy as np


def future_weekly_labels(net_weekly: np.ndarray) -> np.ndarray:
    """Future-only weekly book-return labels aligned to the decisions they score (LEVERAGE.md §10).

    Given the realized weekly net-of-cost book series ``net_weekly`` (``net[k]`` = return realized
    over week ``k``), returns ``labels`` of length ``len(net_weekly) − 1`` where ``labels[k] =
    net_weekly[k + 1]`` — the return realized in the week AFTER decision ``k``. A decision made with
    information through week ``k`` is thus scored ONLY on strictly-future data (the firewall). The
    last decision has no realized future week yet and is intentionally unlabelled (dropped), never
    back-filled with a same-week or synthetic value.

    Raises on a series too short to form a single forward label (``len < 2``) — a silent empty
    label array would let a caller score against nothing.
    """
    r = np.asarray(net_weekly, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError(
            "net_weekly must be a 1-D series of length >= 2 to form a forward label "
            "(LEVERAGE §10 firewall — the last week has no realized future to score against)"
        )
    return r[1:].copy()


def align_decisions_to_labels(
    g_targets: np.ndarray, net_weekly: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align a decision series to its strictly-future labels (LEVERAGE.md §10 Invariant V).

    ``g_targets[k]`` is the ``G_t`` decided at week ``k`` on information through ``k``; it is paired
    with ``future_weekly_labels(net_weekly)[k] = net_weekly[k+1]``. Returns the truncated
    ``(g_targets[:-1], labels)`` pair of equal length so downstream scoring can never accidentally
    pair a decision with a contemporaneous or past return (the labeling firewall). Both inputs must
    share length; a mismatch raises rather than silently truncating to the shorter.
    """
    g = np.asarray(g_targets, dtype=float)
    r = np.asarray(net_weekly, dtype=float)
    if g.shape != r.shape:
        raise ValueError(
            f"g_targets {g.shape} and net_weekly {r.shape} must share one shape before alignment "
            "(LEVERAGE §10 firewall)"
        )
    labels = future_weekly_labels(r)
    return g[:-1].copy(), labels
