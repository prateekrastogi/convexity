"""Gate ``research/xsec/overlay.py``'s mechanics behind the REGIME detector (REGIME.md §5, §5.4).

REGIME does NOT duplicate or modify the overlay MECHANICS (§1: ``vol_target_scale``,
``beta_hedge_returns``, ``drawdown_stop_scale`` are pure causal transforms it consumes). It is the
missing DETECTOR that decides WHEN and HOW HARD to apply them (§0). This module is that gate — it
calls the overlay functions UNCHANGED (their numerics are frozen, §1 Invariant N) and layers the
REGIME contract on top:

  * The ``g_t`` scalar (§4.1) is the primary de-risking channel; a promoted DD-stop is an OPTIONAL
    add-on (§5.4 step 5: "a DD-stop is a nice-to-have overlay, not a hard requirement").
  * Invariant STOP-XFER (§5.4): a ``(dd_trigger, dd_reset)`` calibration validated on ANOTHER sleeve
    (e.g. carry) may NOT be applied to the momentum book without its OWN regime-conditional ablation
    passing at full statistical power. This module enforces that gate structurally: a DD-stop scale
    can only be obtained through :func:`gated_drawdown_stop_scale`, which REQUIRES a
    :class:`DDStopAblationRecord` keyed to the SAME return series and run at ``n_weeks >=
    n_weeks_min_cpcv`` — otherwise it raises (§5.4 / Invariant STOP-XFER test contract).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from convexity.regime.overlay import drawdown_stop_scale
from convexity.regime.config.resolve import ResolvedRegimeConfig


@dataclass(frozen=True)
class DDStopAblationRecord:
    """The regime-conditional ablation evidence a DD-stop MUST carry before live use (REGIME.md §5.4).

    Keyed to the EXACT return series it was validated on (``strategy_id``) so Invariant STOP-XFER can
    reject a carry-validated pair applied to momentum. ``n_weeks`` is the live history the ablation
    ran on (must be ``>= n_weeks_min_cpcv`` for gate (a)'s frac_pos to count, §5.4). ``frac_pos`` is
    the fraction of CPCV OOS paths where the stopped Sharpe >= no-stop Sharpe (§5.4 gate (a)).
    ``passed_bleed_check`` is §5.4 gate (b): NO single CRYPTO_REGIMES window shows the carry-like
    bleed signature (stopped window-return materially BELOW raw by > ``bleed_return_tol``). A record
    is PROMOTABLE only when both gates pass at full power (:meth:`is_promotable`)."""

    strategy_id: str
    dd_trigger: float
    dd_reset: float
    n_weeks: int
    frac_pos: float
    passed_bleed_check: bool

    def is_promotable(self, cfg: ResolvedRegimeConfig) -> bool:
        """True iff this ablation satisfies BOTH §5.4 gates at full statistical power.

        Gate (a): frac_pos majority (> 0.5) AND ``n_weeks >= n_weeks_min_cpcv`` (a short-history
        frac_pos CANNOT by itself satisfy gate (a) — §5.4 edge case / Invariant STOP-XFER). Gate (b):
        the per-window bleed check passed (binding regardless of n_weeks, §5.4)."""
        d = cfg.defaults
        gate_a = (self.frac_pos > 0.5) and (self.n_weeks >= d.n_weeks_min_cpcv)
        return gate_a and self.passed_bleed_check


def gated_drawdown_stop_scale(
    net: np.ndarray,
    ablation: DDStopAblationRecord,
    *,
    strategy_id: str,
    cfg: ResolvedRegimeConfig,
) -> np.ndarray:
    """Apply ``overlay.drawdown_stop_scale`` ONLY behind a passing regime-conditional ablation.

    Enforces Invariant STOP-XFER (REGIME.md §5.4): raises ``ValueError`` if the ablation record is
    keyed to a DIFFERENT strategy's return series than ``strategy_id`` (a carry-validated pair may not
    be silently applied to momentum), or if the ablation is not promotable at full power
    (``is_promotable`` False — including the short-history ``n_weeks < n_weeks_min_cpcv`` case whose
    frac_pos cannot satisfy gate (a), and the failed-bleed-check case). Only when the gate passes does
    it delegate to the FROZEN overlay mechanics UNCHANGED (§1 Invariant N — the numerics are not
    touched here; ``dd_trigger``/``dd_reset`` come straight off the validated record)."""
    if ablation.strategy_id != strategy_id:
        raise ValueError(
            "Invariant STOP-XFER (REGIME.md §5.4): DD-stop ablation is keyed to strategy "
            f"{ablation.strategy_id!r} but is being applied to {strategy_id!r}; a calibration "
            "validated on one sleeve's returns may not transfer without its own ablation."
        )
    if not ablation.is_promotable(cfg):
        raise ValueError(
            "Invariant STOP-XFER (REGIME.md §5.4): DD-stop ablation is not promotable at full "
            f"statistical power (n_weeks={ablation.n_weeks}, min={cfg.defaults.n_weeks_min_cpcv}, "
            f"frac_pos={ablation.frac_pos}, bleed_check={ablation.passed_bleed_check}); a short-"
            "history frac_pos alone cannot satisfy gate (a), and a failed bleed check (b) rejects "
            "the candidate regardless of aggregate Sharpe."
        )
    # Gate passed: delegate to the FROZEN overlay mechanic, numerics unchanged (§1 Invariant N).
    return drawdown_stop_scale(net, dd_trigger=ablation.dd_trigger, dd_reset=ablation.dd_reset)


def bleed_check_by_window(
    raw_net: np.ndarray,
    scale: np.ndarray,
    window_slices: dict[str, slice],
    *,
    cfg: ResolvedRegimeConfig,
) -> dict[str, bool]:
    """Per-CRYPTO_REGIMES-window bleed check (REGIME.md §5.4 gate (b)).

    For each regime window, compare the STOPPED series' total window return against the RAW series'.
    A window FAILS (returns False) when the stopped return is materially BELOW the raw return by more
    than ``bleed_return_tol`` — the carry-like signature of a stop sitting out a genuine recovery
    (§5.2/§5.4). A single failing window is the carry lesson recurring and rejects the candidate
    regardless of aggregate Sharpe (§5.4 gate (b), "a single bleed-like window ...")."""
    tol = cfg.defaults.bleed_return_tol
    stopped = raw_net * scale
    results: dict[str, bool] = {}
    for name, sl in window_slices.items():
        raw_w = float(np.nansum(raw_net[sl]))
        stopped_w = float(np.nansum(stopped[sl]))
        results[name] = not (stopped_w < raw_w - tol)
    return results


def aggregate_bleed_check(bleed_by_window: dict[str, bool]) -> bool:
    """Gate (b)'s scalar verdict: True iff NO CRYPTO_REGIMES window bled (REGIME.md §5.4).

    §5.4 gate (b) rejects a candidate that fails in EVEN ONE historical window ("a single bleed-like
    window is the carry lesson recurring"), so the aggregate is a logical AND over every window's
    pass flag from :func:`bleed_check_by_window`. An EMPTY window map is conservatively NOT a pass
    (no evidence the check ran -> cannot claim gate (b), §0 P6) — a real ablation must cover the
    §5.4 window set."""
    if not bleed_by_window:
        return False
    return all(bleed_by_window.values())


def ablation_with_recomputed_bleed_check(
    record: DDStopAblationRecord,
    raw_net: np.ndarray,
    scale: np.ndarray,
    window_slices: dict[str, slice],
    *,
    cfg: ResolvedRegimeConfig,
) -> DDStopAblationRecord:
    """Rebuild ``record`` with ``passed_bleed_check`` RECOMPUTED from the raw per-window bleed check,
    not trusted from the caller (REGIME.md §5.4 gate (b) / Invariant STOP-XFER).

    This is the wiring §5.4's promotion pipeline (``validate/``, scoped out) must use so a candidate
    cannot smuggle a fabricated ``passed_bleed_check=True`` past the gate: the flag is derived from
    :func:`bleed_check_by_window` over the SAME return series and the CRYPTO_REGIMES window set, then
    aggregated by :func:`aggregate_bleed_check`. A record whose hand-supplied flag disagrees with the
    recomputation is corrected DOWNWARD to the evidence (a claimed pass over bleeding windows becomes
    a fail). Callers should gate on the RETURNED record, never the raw input record."""
    by_window = bleed_check_by_window(raw_net, scale, window_slices, cfg=cfg)
    recomputed = aggregate_bleed_check(by_window)
    return replace(record, passed_bleed_check=recomputed)
