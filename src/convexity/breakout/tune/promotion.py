"""Promotion gates + cross-sectional selection-bias control (BREAKOUT §10 / §11 / Invariant L).

The promotion decision (``approved`` / ``shadow`` / ``rejected``) and the cross-sectional FDR/FWER
control over the universe scanned (Invariant L). SYSTEM's DSR deflates the WITHIN-symbol BLUE-grid
multiplicity; it does NOT see the cross-sectional instrument scan, so ``promote_universe`` controls
the BETWEEN-symbol multiplicity here via Benjamini-Hochberg across the universe. A symbol whose
apparent edge does not survive the correction is ``shadow``, not ``approved``. The universe-scan
trial count is written to the registry artifact (§11) and fed to SYSTEM's ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from convexity.breakout.types import Report
from convexity.breakout.validate.stats import benjamini_hochberg

PromotionStatus = Literal["approved", "shadow", "rejected"]


@dataclass(frozen=True)
class PromotionDecision:
    """The §11 onboarding decision (mirrors the spec's PromotionDecision dataclass)."""

    symbol: str
    status: PromotionStatus
    detector_version: str
    resolved_config_hash: str
    fit_window: tuple[int, int]
    reason: str
    report: Report


def gate_single(
    report: Report,
    baselines: list[Report],
    *,
    min_oos_signals: int,
    leakage_ok: bool,
    parity_ok: bool,
) -> tuple[PromotionStatus, str]:
    """Per-symbol gate BEFORE the cross-sectional correction (§10 promotion gates).

    Returns ``rejected`` on a failed leakage/parity test; ``shadow`` on insufficient OOS evidence or
    failing to beat the parameter-free baselines on BOTH convexity AND false-break rate; else a
    provisional ``approved`` (subject to the cross-sectional FDR gate in ``promote_universe``).
    """
    if not leakage_ok or not parity_ok:
        return "rejected", "failed leakage/batch-live parity test"
    if report.n_signals < min_oos_signals:
        return "shadow", f"insufficient OOS evidence ({report.n_signals} < {min_oos_signals})"

    # must beat Donchian + ATR on BOTH OOS median convexity AND false-breakout rate (§10).
    beat_convexity = all(report.convexity_ratio >= b.convexity_ratio for b in baselines if b.n_signals)
    beat_false = all(report.false_breakout_rate <= b.false_breakout_rate for b in baselines if b.n_signals)
    if not (beat_convexity and beat_false):
        return "shadow", "does not beat baselines on both convexity and false-break rate"
    return "approved", "passes single-symbol gates (pending cross-sectional FDR)"


def promote_universe(
    provisional: list[tuple[str, PromotionStatus, float]], *, fdr_alpha: float
) -> dict[str, PromotionStatus]:
    """Apply the cross-sectional Benjamini-Hochberg FDR over the universe (Invariant L).

    ``provisional`` is ``(symbol, single_gate_status, p_value)`` per symbol where ``p_value`` is the
    symbol's edge significance. Only symbols both ``approved`` by the single gate AND surviving the
    BH-FDR correction are finally ``approved``; the rest are downgraded to ``shadow``. On a pure-noise
    universe this approves ~the FDR-nominal fraction, not the naive uncorrected fraction.
    """
    syms = [s for s, _, _ in provisional]
    pvals = np.array([p for _, _, p in provisional], dtype=float)
    accept = benjamini_hochberg(pvals, fdr_alpha)
    out: dict[str, PromotionStatus] = {}
    for i, (sym, status, _p) in enumerate(provisional):
        if status == "approved" and accept[i]:
            out[sym] = "approved"
        elif status == "rejected":
            out[sym] = "rejected"
        else:
            out[sym] = "shadow"
    _ = syms
    return out
