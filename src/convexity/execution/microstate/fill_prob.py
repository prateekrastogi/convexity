"""FillProbability — the causal, auditable, online-fitted P(fill | offset, horizon, g) model (§3/§5.5).

A CAUSAL ``p_fill(offset_ticks, horizon, g) in [0,1]``: the probability a passive limit at
``offset_ticks`` from the touch fills within ``horizon``, with NO order book. The closed form is a
LOGISTIC in (offset/sigma) and ``g`` (auditable, NOT a learned classifier):

    p_fill = base · σ_(  a0 − a_off·(offset_bps/σ_h) − a_g·g  )

Monotone DECREASING in offset and in ``g`` on the favorable side (a faster run away from a rester
lowers its fill probability), INCREASING in sigma/volume. Sharing ``g`` with AdverseSelection and
DelayCost is REQUIRED (Invariant J).

COARSENESS (§5.5): queue position is unobservable at 1-min with no order book, so this is a
regime-conditional AVERAGE fill rate, not a per-order probability — the highest-variance input to the
cost ``min()``. Three consequences are enforced here:

  * THREE-RUNG terminal fallback (§5.5 #2): bucket (>= N_bucket own passive fills) -> global
    (>= N_global) -> CONSERVATIVE class prior (AMBER) with the cold-start maker GATED OFF. No
    unfloored rung.
  * a calibration-CI half-width (~1/sqrt of the active rung's event count) so the maker-floor gate
    can key on the LOWER confidence bound, not the point estimate (§6.3).
  * ONLINE point-in-time / embargoed update (Invariant L): a realized passive fill enters the fit
    ONLY after its outcome horizon has fully elapsed and is embargoed, so a fill can never enter the
    curve it is being scored against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from convexity.execution.config.resolve import ResolvedConfig

__all__ = ["logistic_fill_curve", "FillCurveResult", "FillProbability", "PassiveFillRecord"]


def _logistic(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def logistic_fill_curve(
    *,
    base_rate: float,
    sigma_bps: float,
    tick_bps: float,
    a_offset: float = 1.0,
    a_g: float = 0.8,
) -> Callable[[float, float, float], float]:
    """Build an auditable closed-form ``p_fill(offset_ticks, horizon_s, g)`` (§3 / §5.5).

    Monotone DECREASING in ``offset_ticks`` (deeper post fills less) and in ``g`` (a faster run away
    lowers fill prob on the favorable side); INCREASING in sigma (more wandering reaches the offset)
    via the offset/sigma standardization. ``base_rate`` caps the at-touch fill rate (the active rung's
    fitted level). The horizon scales the effective reach (longer horizon -> higher fill prob).
    """
    base = float(min(max(base_rate, 0.0), 1.0))
    sig = max(float(sigma_bps), 1e-6)
    tb = max(float(tick_bps), 1e-9)

    def curve(offset_ticks: float, horizon_s: float, g: float) -> float:
        offset_bps = max(float(offset_ticks), 0.0) * tb
        std_offset = offset_bps / sig
        # Longer horizon increases reach: shrink the effective offset by sqrt(horizon/30).
        horizon_factor = math.sqrt(max(horizon_s, 1e-6) / 30.0)
        std_offset = std_offset / max(horizon_factor, 1e-6)
        z = 2.0 - a_offset * std_offset - a_g * max(float(g), 0.0)
        return base * _logistic(z)

    return curve


@dataclass(frozen=True)
class FillCurveResult:
    """The fitted curve at a regime bucket plus the metadata the maker-floor LCB gate reads (§5.5/§6.3)."""

    curve: Callable[[float, float, float], float]
    rung: str  # "bucket" | "global" | "class_prior"
    ci_halfwidth: float  # ~1/sqrt(active-rung event count); widens the LCB on thin data (§6.3)
    cold_start: bool  # True at the class-prior rung -> maker GATED OFF by default (§5.5 #2)
    n_events: int


@dataclass(frozen=True)
class PassiveFillRecord:
    """One realized passive-fill outcome used to fit the curve online (Invariant L).

    ``outcome_close_ts`` is when the fill's outcome horizon fully elapsed; a record enters the fit
    ONLY when ``outcome_close_ts <= as_of_ts`` (point-in-time / embargoed) so it never scores itself.
    """

    regime_bucket: str
    offset_ticks: float
    g: float
    filled: bool
    outcome_close_ts: int


@dataclass
class FillProbability:
    """Causal, online, three-rung-floored fill-probability fitter (§3 / §5.5 / Invariants J/L)."""

    # The instrument's OWN realized passive fills (TEAL), appended online under the embargo rule.
    records: list[PassiveFillRecord] = field(default_factory=list)

    def admit(self, record: PassiveFillRecord) -> None:
        """Append a realized passive fill. Admission to the FIT is gated by the embargo in :meth:`curve`
        (a record is only USED when its outcome horizon closed at or before the as-of time)."""
        self.records.append(record)

    def _resolved(self, regime_bucket: str, as_of_ts: int) -> tuple[list[PassiveFillRecord], int]:
        """Bucket records whose outcome horizon closed at/before ``as_of_ts`` (Invariant L embargo)."""
        bucket = [
            r for r in self.records
            if r.outcome_close_ts <= as_of_ts and r.regime_bucket == regime_bucket
        ]
        glob = [r for r in self.records if r.outcome_close_ts <= as_of_ts]
        return bucket, len(glob)

    def curve(
        self,
        micro_inputs,
        regime_bucket: str,
        cfg: ResolvedConfig,
        *,
        as_of_ts: int = 0,
        sigma_bps: float,
        tick_bps: float,
    ) -> FillCurveResult:
        """Return the fitted curve + LCB metadata via the THREE-RUNG terminal fallback (§5.5 #2).

        Rung selection on the embargoed (Invariant L) own-passive-fill counts:
          * per-BUCKET fit when its resolved count >= ``N_bucket``;
          * GLOBAL pooled curve when below N_bucket but the global count >= ``N_global``;
          * CONSERVATIVE class prior (AMBER) when even the global count < ``N_red`` — the cold-start,
            with maker GATED OFF by default. There is NO unfloored rung.
        The ci_halfwidth scales ~1/sqrt(active-rung event count) so a thin bucket widens the LCB and
        the maker-floor gate removes POST_ONLY automatically (§6.3).
        """
        d = cfg.defaults
        bucket, n_global = self._resolved(regime_bucket, as_of_ts)
        n_bucket = len(bucket)

        if n_bucket >= d.n_bucket:
            rate = sum(1 for r in bucket if r.filled) / max(n_bucket, 1)
            return FillCurveResult(
                logistic_fill_curve(base_rate=rate, sigma_bps=sigma_bps, tick_bps=tick_bps),
                "bucket", 1.0 / math.sqrt(n_bucket), False, n_bucket,
            )

        glob = [r for r in self.records if r.outcome_close_ts <= as_of_ts]
        if n_global >= d.n_global:
            rate = sum(1 for r in glob if r.filled) / max(n_global, 1)
            return FillCurveResult(
                logistic_fill_curve(base_rate=rate, sigma_bps=sigma_bps, tick_bps=tick_bps),
                "global", 1.0 / math.sqrt(n_global), False, n_global,
            )

        # Terminal rung — cold-start: conservative class prior, maker GATED OFF (§5.5 #2). Below N_red
        # own passive fills the symbol has not EARNED its curve, so default to crossing.
        cold = n_global < d.n_red
        return FillCurveResult(
            logistic_fill_curve(
                base_rate=cfg.preset.class_prior_fill_rate, sigma_bps=sigma_bps, tick_bps=tick_bps
            ),
            "class_prior", cfg.preset.class_prior_ci_halfwidth, cold, n_global,
        )

    @staticmethod
    def lcb(point: float, ci_halfwidth: float, calib_error: float, cfg: ResolvedConfig) -> float:
        """The LOWER-confidence bound the maker-floor gate keys on (§6.3 gate 3).

        ``p_fill_LCB = p_fill_point − z · ci_halfwidth`` — widened by calibration error implicitly via
        the floor (see :func:`floor_eff`). Clamped to [0, 1].
        """
        bound = point - cfg.defaults.z_lcb * ci_halfwidth
        return float(min(max(bound, 0.0), 1.0))

    @staticmethod
    def floor_eff(calib_error: float, cfg: ResolvedConfig) -> float:
        """``floor_eff = floor_RED + κ · max(0, calib_error − calib_RED)`` (§6.3).

        Rises when the §12 reliability-curve calibration error exceeds the RED threshold, so an
        off-diagonal curve AUTOMATICALLY tightens the live gate (closing the §12 loop).
        """
        d = cfg.defaults
        return float(d.floor_red + d.kappa_calib * max(0.0, calib_error - d.calib_red))
