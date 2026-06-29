"""Causal prewhitening / batch-means decorrelation — so the ARL holds under autocorr + fat tails.

Spec: OPS §6.7 (the SPC-on-financial-data trap). Every monitored residual is autocorrelated AND
fat-tailed; the as-designed ARL0 is NOT the realized ARL0 on a raw financial residual, so the panel
would page far more often than budgeted. Two RED-pinned routes take the raw residual to an
approximately-IID innovation the IID ARL math holds for:

  (a) **Prewhitened AR(p) innovation** (default for e_vol/e_cost/e_fill, §6.7a):
        z_i = (residual_i - φ·residual_{i-1}) / σ_innovation
      ``φ`` is a frozen RED least-squares AR fit on the IN-CONTROL validation residual (offline, never
      live — P2/LIM). The innovation is approximately IID N(0,1) in control.
  (b) **Non-overlapping BATCH MEANS** (alternative, esp. for e_edge, §6.7b): average over
      non-overlapping blocks of the RED bootstrap block length, then run the detector on the batch
      means — the per-source ARL0 is then counted in *batches* (§6.6).

Both are CAUSAL (the prewhitener reads only ``residual_i`` and the frozen prior residual + ``φ``;
the batch mean reads only the closed block) — Invariant CAUSAL.
"""

from __future__ import annotations

import numpy as np

from convexity.ops.types import InControlParams


class CausalPrewhitener:
    """Causal AR(p) prewhitening to an approximately-IID standardized innovation (§6.7a).

    ``update(residual)`` returns ``z_i = (residual_i - Σ_p φ_p·residual_{i-p}) / σ_innovation`` using
    the frozen RED coefficients ``φ`` and the frozen ``σ_innovation`` (the std of the in-control
    *innovations*, i.e. ``σ0·sqrt(1 - Σφ²)`` for an AR(1) when ``σ0`` is the residual std — but we
    take it directly from the in-control params so it is the validated number, not a re-derivation).
    When ``φ`` is empty the prewhitener degrades to plain standardization ``(residual-μ0)/σ0``.
    """

    def __init__(self, ic: InControlParams):
        self.ic = ic
        self.phi = np.asarray(ic.phi, dtype=float)
        self.p = int(self.phi.shape[0])
        # σ of the in-control innovations: for AR(1), Var(innov) = σ_resid²·(1 - φ²). We use σ0 as the
        # residual std and deflate by the AR coefficients so the innovation z is ~N(0,1) in control.
        var_deflate = max(1e-12, 1.0 - float(np.sum(self.phi ** 2)))
        self.sigma_innovation = ic.sigma0 * np.sqrt(var_deflate) if self.p else ic.sigma0
        self._history: list[float] = []  # bounded to length p (centered prior residuals)

    def reset(self) -> None:
        self._history = []

    def update(self, residual: float) -> float:
        """Causally prewhiten one raw residual to a standardized innovation (§6.7a)."""
        centered = residual - self.ic.mu0
        pred = 0.0
        for lag in range(self.p):
            if lag < len(self._history):
                pred += self.phi[lag] * self._history[-(lag + 1)]
        innovation = centered - pred
        z = innovation / self.sigma_innovation
        # push the centered residual; keep only the last p
        self._history.append(centered)
        if len(self._history) > self.p:
            self._history.pop(0)
        return z


class BatchMeans:
    """Non-overlapping batch-means decorrelator (§6.7b). Yields a standardized batch mean per block.

    ``update(residual)`` accumulates raw residuals; when ``block_length`` have arrived it returns the
    standardized block mean ``z = (mean - μ0)/(σ0/sqrt(b))`` (the mean's std shrinks by ``sqrt(b)``)
    and starts a fresh block. Between block boundaries it returns ``None`` (no detector update). The
    per-source ARL0 is counted in batches (§6.6).
    """

    def __init__(self, ic: InControlParams, block_length: int):
        self.ic = ic
        self.block_length = int(block_length)
        self._buf: list[float] = []

    def reset(self) -> None:
        self._buf = []

    def update(self, residual: float) -> float | None:
        """Accumulate one residual; emit a standardized batch mean at each block boundary, else None."""
        self._buf.append(residual)
        if len(self._buf) < self.block_length:
            return None
        mean = float(np.mean(self._buf))
        self._buf = []
        sigma_mean = self.ic.sigma0 / np.sqrt(self.block_length)
        return (mean - self.ic.mu0) / sigma_mean


def standardize(residual: float, ic: InControlParams) -> float:
    """Plain standardization ``z = (residual - μ0)/σ0`` against the FROZEN ``(μ0, σ0)`` (§6, §4 TEAL).

    The denominator is the frozen ``σ0`` — a live ``σ`` is diagnostic-only (Invariant LIM-(b)): the
    raw trip point stays ``μ0 + h·σ0`` regardless of live vol. Used directly by streams with no
    autocorrelation (``φ`` empty); autocorrelated streams use :class:`CausalPrewhitener`.
    """
    return (residual - ic.mu0) / ic.sigma0


__all__ = ["CausalPrewhitener", "BatchMeans", "standardize"]
