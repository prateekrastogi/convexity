"""Re-promotion cadence + held-out PAIRED block-bootstrap forward gate (SYSTEM §7.5, Inv RP).

Freezing a config is not the end of the multiple-testing story: every night's pick is another trial.
If repeated searches are not accumulated into the ledger, ``N_sys`` understates the true
multiplicity over calendar time and the system overfits-by-iteration. The policy:

  * The live config is frozen between swaps; the overnight job proposes a swap only on ``τ_promote``.
  * Every search (even a non-swapping monitoring run) appends its scored configs to the SAME
    persistent ledger, so ``N_sys`` accumulates (Invariant RP / handled by :class:`TrialLedger`).
  * A swap candidate must clear a FORWARD GATE on a held-out, strictly-later, never-searched window
    AND still pass the accumulated-ledger DSR.

The forward gate is a PAIRED block-bootstrap test on the per-bucket candidate-minus-incumbent return
DIFFERENCE over the IDENTICAL forward block (Invariant BL), NOT a bare point margin: at ~36 obs/yr a
single fat-tail trade can flip the SIGN of a point Sharpe difference. The bootstrap CI on the paired
difference must EXCLUDE ZERO and its lower bound clear round-trip cost; iid bootstrap is FORBIDDEN
(use the §8.1 dependence-preserving block length). The forward window is embargoed from the search
window by ``H_sys`` (Invariant HC), and ``τ_forward`` must yield ``n_obs ≥ max(MinTRL, n_obs_floor)``
or the swap DEFERS with ``insufficient_forward_track``.

The SYSTEM §6 six-gate PROMOTABLE conjunction is a SEPARATE promotion-time gate battery and lives in
its own §6.2-named module :mod:`convexity.system.validate.promotion_gate`; this module implements
only the §7.5 forward/cadence gate that SYSTEM §11's directory layout assigns to ``promotion.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.system.validate.robustness import block_bootstrap_ci


@dataclass(frozen=True)
class ForwardGateResult:
    """Output of :meth:`ForwardGate.evaluate` (Section 7.5). ``swap`` is True only when the paired
    difference CI excludes zero AND its lower bound clears round-trip cost AND the forward block is
    long enough (n_obs ≥ floor). ``ci_low``/``ci_high`` are the paired-difference Sharpe CI bounds."""

    swap: bool
    ci_low: float
    ci_high: float
    diff_sharpe: float
    block_length: int
    forward_n_obs: int
    n_obs_sufficient: bool
    reason_codes: tuple[str, ...]


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r)) / sd if sd > 0.0 else 0.0


class ForwardGate:
    """The held-out PAIRED block-bootstrap forward gate (SYSTEM §7.5, Invariant RP)."""

    def evaluate(
        self,
        candidate_forward: np.ndarray,
        incumbent_forward: np.ndarray,
        cfg,
        *,
        round_trip_cost: float = 0.0,
        min_trl: float = 0.0,
        seed: int = 0,
    ) -> ForwardGateResult:
        """Run both on the IDENTICAL forward bars, form the per-bucket DIFFERENCE d_t = r_cand −
        r_incumbent, block-bootstrap d_t for a CI on the difference Sharpe, and require the CI to
        EXCLUDE ZERO and its lower bound to clear ``round_trip_cost`` (Section 7.5). The two series
        MUST be the same length (paired, not two independent samples; Invariant BL).
        """
        defaults = cfg.defaults
        cand = np.asarray(candidate_forward, dtype=float)
        inc = np.asarray(incumbent_forward, dtype=float)
        if cand.shape != inc.shape:
            raise ValueError("paired forward gate requires identical-length series (Invariant BL)")
        d = cand - inc  # the paired per-bucket difference series

        n_obs = d.size
        floor = max(min_trl, float(defaults.n_obs_floor))
        n_obs_sufficient = n_obs >= floor
        reasons: list[str] = []

        ci_low, ci_high = block_bootstrap_ci(
            d,
            block_length=defaults.forward_block_length,
            n_boot=defaults.forward_n_boot,
            ci_level=defaults.forward_ci_level,
            seed=seed,
        )
        diff_sharpe = _sharpe(d)

        if not n_obs_sufficient:
            reasons.append("insufficient_forward_track")
            return ForwardGateResult(
                swap=False, ci_low=ci_low, ci_high=ci_high, diff_sharpe=diff_sharpe,
                block_length=defaults.forward_block_length, forward_n_obs=n_obs,
                n_obs_sufficient=False, reason_codes=tuple(reasons),
            )

        excludes_zero = ci_low > 0.0
        clears_cost = ci_low > round_trip_cost
        swap = excludes_zero and clears_cost
        if not excludes_zero:
            reasons.append("paired_ci_includes_zero")
        elif not clears_cost:
            reasons.append("paired_ci_below_cost")
        else:
            reasons.append("forward_gate_passed")
        return ForwardGateResult(
            swap=swap, ci_low=ci_low, ci_high=ci_high, diff_sharpe=diff_sharpe,
            block_length=defaults.forward_block_length, forward_n_obs=n_obs,
            n_obs_sufficient=True, reason_codes=tuple(reasons),
        )
