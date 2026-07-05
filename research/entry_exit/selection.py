"""SELECT.protocol -- pure selection statistics + the single Phase-C adoption conjunction.

Everything the multi-family model-selection study needs to decide KEEP-INCUMBENT vs ADOPT on the
HELD-OUT OOS slice, expressed over ``CandidateResult`` objects (MODELS.base) and REUSING the SYSTEM
judge (``split`` / ``bootstrap_ci`` / ``deflated_sharpe``, SYSTEM §3) verbatim. This module adds ONLY
the selection pieces the incumbent-only report deferred (SYSTEM §6) and that a multi-family study
re-admits (SYSTEM §C.3): PBO/CSCV, the IC/parsimony-penalized objective, drop-top-N, cross-variant
consistency, and the 5-clause adoption rule. It is PURE: typed results, no prints, no lake.

THE binding invariants (all from the frozen pre-registration -- SYSTEM §C, SIGNAL §C, docs/EDGE.md):

- **Held-out OOS ONLY (SYSTEM §C.1, §C.3).** The adopt statistic and every §C.6 number is computed on
  the single pre-registered held-out slice ``SPLIT = 2025-06-01`` (`test` side of ``system.split``).
  Training folds only FIT + FREEZE parameters; the walk-forward folds are a CONSISTENCY-only overlay
  (SYSTEM §C.3) that feeds clause 4 (cross-variant), NEVER the adopt point estimate and NEVER
  ``N_eff`` -- so the per-fold refits are not extra scored looks at held-out data.
- **Per-observation Sharpe, never annualized (SYSTEM §4).** Every Sharpe fed to DSR / the bootstrap is
  ``system.per_obs_sharpe`` (the un-annualized ``mean/std``, ddof=0) via ``system``'s own defaults --
  this module never annualizes.
- **``N_eff`` is the COMPOSED, correlation-deflated scored-config space (SYSTEM §C.3, EDGE).** NOT the
  additive 8: it counts the entry x exit x {HAR on/off} configuration grid AND each config's
  within-model parameter search, deflated for correlation. The caller supplies the deflated integer
  (``n_trials_eff``); this module passes it straight to ``system.deflated_sharpe`` and NEVER
  substitutes the additive candidate count (feeding the additive 8 is the FAIL->PASS flip the gate
  exists to prevent).
- **Parsimony penalty is INSIDE clause 1 (SYSTEM §C.4, EDGE clause 1).** Out-performance is measured
  on the IC-penalized objective (OOS EV net of a per-parameter ``k`` penalty), never raw OOS fit. A
  richer model must beat the incumbent by MORE than its added-complexity penalty; a tie resolves to
  the incumbent (the parsimony floor).
- **The adoption rule is ONE conjunction (SYSTEM §C.6 == EDGE clauses 1-5), miss-any => not adopted.**
  (1) beat the incumbent OOS net of the parsimony penalty, (2) DSR >= ``DSR_PASS`` with the composed
  ``N_eff`` AND PBO below ``PBO_CEILING``, (3) drop-top-1 AND drop-top-3 robust, (4) cross-variant
  consistent, (5) paired-bootstrap Sharpe-difference 90% CI clears zero. The incumbent is the default
  and survives every tie; the negative is always recorded (no partial-credit adoption).
- **Off-ledger rule (SYSTEM §C.3, §C.9).** A candidate whose ``name`` is not one of the 8
  pre-registered ledger names is a snooped trial by definition and CANNOT be adopted -- ``adopt``
  fails it on an explicit off-ledger clause, never scores it.

Imports ONLY ``intraday.core`` / ``intraday.models.base`` / ``intraday.system`` (SYSTEM §C.3
interface-driven-harness rule -- NO sibling Phase-C model module). Pure numpy + stdlib. Map:
docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._core import TradeRecord
from .base import CandidateResult
from .validation import (
    DSR_PASS,
    DeflatedSharpeResult,
    bootstrap_ci,
    deflated_sharpe,
    per_obs_sharpe,
    split,
)

__all__ = [
    "SPLIT",
    "PBO_CEILING",
    "IC_PENALTY_PER_PARAM",
    "FROZEN_LEDGER",
    "FoldSpec",
    "ClauseResult",
    "DropTopNResult",
    "PBOResult",
    "AdoptionDecision",
    "ic_penalized_objective",
    "drop_top_n",
    "pbo_cscv",
    "incumbent_vs_candidate_ci",
    "cross_variant_consistent",
    "walk_forward_folds",
    "adopt",
]

# --- Pre-registered selection constants (frozen -- SYSTEM §C / docs/EDGE.md, NOT tuned) ------------
# The single held-out boundary that IS the headline gate (SYSTEM §2.1, §C.1); selection reads the
# `test` side ONLY. Frozen, never tuned.
SPLIT = dt.date(2025, 6, 1)

# The paired-bootstrap design (SYSTEM §C.6 clause 5 / §8 / docs/EDGE.md) -- the futures-vs-options
# study's exact parameters, so the CI that STRADDLED zero there behaves identically here.
BOOTSTRAP_B = 20000
BOOTSTRAP_BLOCK = 1  # whole-trade (block-1) resample; SIGNAL trades are non-overlapping (SYSTEM §3.2)
BOOTSTRAP_ALPHA = 0.10  # a 90% CI (SYSTEM §C.6 clause 5)

# PBO ceiling (SYSTEM §C.3 / §C.6 clause 2 / docs/EDGE.md). PBO is the probability the in-sample-best
# candidate under-performs the median OOS -- a fair-coin 0.5 is "no better than chance"; the
# pre-registered ceiling admits only a candidate whose in-sample rank SURVIVES the CSCV train/test
# swap. Set at the conventional Bailey-Lopez de Prado threshold; frozen here, not searched.
PBO_CEILING = 0.50

# The IC/parsimony penalty per free parameter, charged INSIDE clause 1 (SYSTEM §C.4, EDGE clause 1).
# The objective is OOS EV (fractional, per trade) penalized by `k` parameters; a richer model must
# beat the incumbent by MORE than (its k - incumbent k) * this. The incumbent is the parsimony floor.
IC_PENALTY_PER_PARAM = 0.0  # default: penalty magnitude the caller sets; 0 => raw-EV tie-break only

# The 8 pre-registered ledger names (SYSTEM §C.9 == SIGNAL §C.2-§C.4 == docs/EDGE.md frozen set). A
# candidate whose `name` is NOT here is off-ledger -- a snooped trial by definition (SYSTEM §C.3), and
# `adopt` fails it. This is the ONE enumerable set `N_eff` and the off-ledger rule are charged against.
FROZEN_LEDGER: tuple[str, ...] = (
    "bocpd",              # 1. Bayesian Online Change-Point Detection (entry)
    "sprt_glr",           # 2. Wald SPRT / GLR sequential detection (entry)
    "hmm_filtered",       # 3. 2-state Gaussian HMM, FILTERED posterior (entry)
    "lee_mykland",        # 4. Lee-Mykland jump test, directional continuation (entry)
    "optimal_stopping",   # 5. Optimal-stopping free-boundary (exit/hold)
    "ou_leung_li",        # 6. OU / Leung-Li mean-reversion optimal exit (exit/hold)
    "survival_hazard",    # 7. Survival / hazard duration reversal exit (exit/hold)
    "har_rv",             # 8. HAR-RV (Corsi) directional vol covariate (vol input)
)


# ==================================================================================================
# Typed results
# ==================================================================================================
@dataclass(frozen=True)
class FoldSpec:
    """One anchored walk-forward fold: an expanding train window and its forward OOS window (SYSTEM §C.2).

    ``train`` is the (embargoed) train trade stream fit against, ``test`` the fold's forward OOS window
    scored for the CONSISTENCY overlay ONLY (SYSTEM §C.3 -- WF folds never enter ``N_eff`` or the adopt
    point estimate; they feed cross-variant consistency, clause 4). Boundaries are the fold cut dates.
    """

    boundary: dt.date
    train: list[TradeRecord]
    test: list[TradeRecord]


@dataclass(frozen=True)
class ClauseResult:
    """One adoption clause's pass/fail plus the number that decided it (SYSTEM §C.6)."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DropTopNResult:
    """Drop-top-N winner-robustness of the candidate's OOS out-performance (SYSTEM §C.5, SIGNAL §11).

    The out-performance (candidate EV minus incumbent EV, on matched trades) must PERSIST after
    removing the top-1 AND the top-3 winners that most drive the candidate's margin -- else the "edge"
    is a one-to-three-trade lottery (the fat-tail tell; no vehicle in the record survives this yet).
    ``margin_full`` / ``margin_drop1`` / ``margin_drop3`` are the paired-difference EVs; ``passed`` iff
    both drops stay strictly positive.
    """

    margin_full: float
    margin_drop1: float
    margin_drop3: float
    passed: bool


@dataclass(frozen=True)
class PBOResult:
    """Probability of Backtest Overfitting via CSCV (Bailey-Lopez de Prado 2014, SYSTEM §C.3).

    Over the pooled candidate performance matrix (rows = time slices, cols = candidates), CSCV forms
    every balanced train/test split of the row-blocks, ranks candidates in-sample, and asks whether the
    in-sample-best candidate lands in the BOTTOM half out-of-sample. ``pbo`` is the fraction of splits
    where the IS-best candidate's OOS rank is below the median (its logit distribution's mass at
    negative logits). A candidate whose in-sample rank does NOT survive the swap has high PBO and is
    rejected regardless of its point OOS number. ``passed`` iff ``pbo <= ceiling``.
    """

    pbo: float
    n_splits: int
    ceiling: float
    passed: bool


@dataclass(frozen=True)
class AdoptionDecision:
    """The single KEEP-INCUMBENT / ADOPT verdict + every clause (SYSTEM §C.6 == EDGE clauses 1-5).

    ``adopted`` is True IFF every clause in ``clauses`` passed (the conjunction). ``candidate`` is the
    ledger name judged; ``incumbent`` names the reigning model. When ``adopted`` is False, ``reason``
    names the FIRST failing clause -- the negative that is recorded (SYSTEM §C.6, docs/EDGE.md
    append-never-rewrite). The incumbent is the default and survives every tie.
    """

    candidate: str
    incumbent: str
    adopted: bool
    reason: str
    clauses: tuple[ClauseResult, ...]


# ==================================================================================================
# Clause 1 -- IC/parsimony-penalized OOS out-performance (SYSTEM §C.4, EDGE clause 1)
# ==================================================================================================
def _ev(trades: Sequence[TradeRecord]) -> float:
    """Mean per-trade fractional net return (OOS EV/trade) -- the raw objective before the k-penalty."""
    r = [t[1] for t in trades]
    return float(np.mean(r)) if r else 0.0


def ic_penalized_objective(
    trades: Sequence[TradeRecord], k: int, *, penalty_per_param: float = IC_PENALTY_PER_PARAM
) -> float:
    """OOS EV/trade penalized by the free-parameter count ``k`` (AIC/BIC-style; SYSTEM §C.4).

    The objective clause 1 compares on: ``EV(trades) - penalty_per_param * k``. A richer model (higher
    ``k``) is DOCKED for its extra parameters, so it must earn them on OOS -- a candidate that only
    matches the incumbent's raw EV with more parameters scores STRICTLY LOWER and loses (the incumbent
    is the parsimony floor, SYSTEM §C.4 / §C.6). With ``penalty_per_param == 0`` this reduces to the
    raw EV (the penalty magnitude is a pre-registered constant the caller sets; the STRUCTURE -- the
    penalty is inside the objective, not a downstream tie-break -- is what is binding here). ``k`` is a
    non-negative count.
    """
    if k < 0:
        raise ValueError("k must be non-negative (a free-parameter / latent-state-dimension count)")
    if penalty_per_param < 0:
        raise ValueError("penalty_per_param must be non-negative")
    return _ev(trades) - penalty_per_param * k


# ==================================================================================================
# Clause 3 -- drop-top-N winner robustness (SYSTEM §C.5, SIGNAL §11)
# ==================================================================================================
def _paired_returns(
    candidate: Sequence[TradeRecord], incumbent: Sequence[TradeRecord]
) -> tuple[np.ndarray, np.ndarray]:
    """The two matched return arrays for a paired comparison -- MUST be the same length (SYSTEM §C.5).

    Candidate and incumbent are compared on the SAME matched ORB trade dates (matched-comparison
    discipline, SYSTEM §2.3 / §C.5): a candidate that only swaps the exit/hold or entry timing has one
    incumbent trade per candidate trade. A length mismatch is a caller wiring bug (unmatched pairing),
    not a silent truncation -- it RAISES.
    """
    a = np.asarray([t[1] for t in candidate], dtype="float64")
    b = np.asarray([t[1] for t in incumbent], dtype="float64")
    if len(a) != len(b):
        raise ValueError("candidate and incumbent must be matched pairs (same length) -- SYSTEM §C.5")
    return a, b


def drop_top_n(
    candidate: Sequence[TradeRecord], incumbent: Sequence[TradeRecord]
) -> DropTopNResult:
    """Does the candidate's OOS out-performance survive dropping its top-1 and top-3 winners (SYSTEM §C.5)?

    Applied to the PAIRED DIFFERENCE (SYSTEM §C.5): form the per-trade difference
    ``d = candidate_return - incumbent_return`` on matched dates, then drop the trades whose difference
    most drives the candidate's margin (the largest positive ``d``) and require the mean margin to stay
    strictly positive after dropping the top-1 AND the top-3. This is the PRIMARY robustness statistic
    for the convex, fat-tailed family -- an out-performance resting on one-to-three lucky trades is not
    an edge. Fewer than 4 matched pairs cannot clear a drop-top-3 (there is no surviving sample), so it
    fails on a thin stream.
    """
    a, b = _paired_returns(candidate, incumbent)
    if len(a) == 0:
        return DropTopNResult(0.0, 0.0, 0.0, passed=False)
    d = a - b
    order = np.sort(d)[::-1]  # descending: order[0] is the single most-margin-driving trade
    margin_full = float(order.mean())
    margin_drop1 = float(order[1:].mean()) if len(order) > 1 else 0.0
    # After dropping the top-3, at least one trade must remain for a defined mean; else there is no
    # surviving sample and the drop-top-3 cannot pass (a <=3-trade margin IS the lottery this catches).
    margin_drop3 = float(order[3:].mean()) if len(order) > 3 else 0.0
    passed = len(order) > 3 and margin_drop1 > 0.0 and margin_drop3 > 0.0
    return DropTopNResult(margin_full, margin_drop1, margin_drop3, passed=passed)


# ==================================================================================================
# Clause 2b -- PBO via CSCV over the pooled candidate performance matrix (SYSTEM §C.3)
# ==================================================================================================
def pbo_cscv(
    perf: np.ndarray, *, n_blocks: int = 8, ceiling: float = PBO_CEILING
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV (Bailey-Lopez de Prado 2014; SYSTEM §C.3, §C.6).

    ``perf`` is the pooled candidate PERFORMANCE MATRIX: rows = time slices (contiguous return blocks),
    columns = candidates. CSCV partitions the ``n_blocks`` row-blocks into every balanced train/test
    combination (``C(n_blocks, n_blocks/2)`` splits, each split's complement its OOS half), ranks
    candidates by IN-SAMPLE performance on the train blocks, takes the IS-best candidate, and measures
    its OOS relative rank ``w`` in ``(0, 1)`` on the test blocks. ``logit(w)`` is accumulated; the PBO
    is the fraction of splits where the IS-best under-performs the OOS median (``logit <= 0``) -- the
    probability the winner was overfit. A candidate whose in-sample rank does not survive the swap has
    PBO near or above 0.5 and is rejected regardless of its point OOS number (SYSTEM §C.6 clause 2).

    ``n_blocks`` must be EVEN (CSCV forms balanced halves) and the matrix must have >= 2 candidates
    (a single-candidate matrix has no rank to overfit -- PBO is undefined and RAISES; this is the
    multi-family study's re-admission of the SYSTEM §6 PBO deferral, valid only once the ledger is
    large enough to form the combinations). ``passed`` iff ``pbo <= ceiling``.
    """
    m = np.asarray(perf, dtype="float64")
    if m.ndim != 2:
        raise ValueError("perf must be a 2-D matrix (rows = time slices, cols = candidates)")
    n_rows, n_cand = m.shape
    if n_cand < 2:
        raise ValueError("PBO/CSCV needs >= 2 candidates to form a rank (SYSTEM §C.3 multi-family)")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 2 (CSCV forms balanced halves)")
    if n_rows < n_blocks:
        raise ValueError("perf needs at least n_blocks rows to partition into blocks")

    # Contiguous, near-equal row-blocks (np.array_split handles a non-multiple row count).
    block_idx = [b for b in np.array_split(np.arange(n_rows), n_blocks)]
    half = n_blocks // 2
    logits: list[float] = []
    for train_blocks in itertools.combinations(range(n_blocks), half):
        test_blocks = [b for b in range(n_blocks) if b not in train_blocks]
        tr = np.concatenate([block_idx[b] for b in train_blocks])
        te = np.concatenate([block_idx[b] for b in test_blocks])
        is_score = m[tr].mean(axis=0)   # in-sample per-candidate performance
        oos_score = m[te].mean(axis=0)  # out-of-sample per-candidate performance
        best = int(np.argmax(is_score))  # the candidate a snooper would have picked in-sample
        # OOS relative rank of that candidate in (0, 1): fraction of candidates it beats OOS.
        oos_rank = (oos_score < oos_score[best]).sum()
        w = (oos_rank + 1.0) / (n_cand + 1.0)  # (0, 1), never exactly 0/1 (logit finite)
        logits.append(math.log(w / (1.0 - w)))
    n_splits = len(logits)
    # PBO = P(logit <= 0) = probability the IS-best lands in the bottom half OOS (overfit).
    pbo = float(np.mean([1.0 if lg <= 0.0 else 0.0 for lg in logits]))
    return PBOResult(pbo=pbo, n_splits=n_splits, ceiling=ceiling, passed=pbo <= ceiling)


# ==================================================================================================
# Clause 5 -- paired-bootstrap Sharpe-difference CI vs the incumbent (SYSTEM §C.6, §3.2)
# ==================================================================================================
def incumbent_vs_candidate_ci(
    candidate: Sequence[TradeRecord],
    incumbent: Sequence[TradeRecord],
    *,
    B: int = BOOTSTRAP_B,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
):
    """Paired trade-level bootstrap CI of the candidate-minus-incumbent Sharpe difference (SYSTEM §C.6).

    REUSES ``system.bootstrap_ci`` in its paired mode (resample the shared trade INDEX once, apply to
    both legs) on the SAME matched ORB trade dates. The point/CI is the per-observation Sharpe
    difference ``per_obs_sharpe(candidate) - per_obs_sharpe(incumbent)`` -- the un-annualized ratio
    (SYSTEM §4). Clause 5 passes IFF the 90% CI EXCLUDES zero (a favorable point estimate is NOT
    enough): this is exactly the test that STRADDLED zero in the futures-vs-options study
    ([-0.27, +0.42], P(option > future) ~ 0.66) and killed that claim (docs/EDGE.md). Defaults are the
    frozen ``B=20000, block=1`` design (SYSTEM §8); paired mode requires matched-length streams.
    """
    a, b = _paired_returns(candidate, incumbent)  # raises on a length mismatch (unmatched pairing)
    return bootstrap_ci(a, per_obs_sharpe, B=B, alpha=alpha, seed=seed, paired_with=b)


# ==================================================================================================
# Clause 4 -- cross-variant consistency (the anti-snooping mechanism; SYSTEM §C.5)
# ==================================================================================================
def cross_variant_consistent(variant_margins: Sequence[float]) -> ClauseResult:
    """Does the out-performance hold across the candidate's variants / WF folds (SYSTEM §C.5, §C.6)?

    ``variant_margins`` is the candidate-minus-incumbent margin on EACH sibling variant / instrument /
    walk-forward fold (SYSTEM §C.3: the WF folds feed THIS check only, never ``N_eff`` or the adopt
    point estimate). Consistency = the out-performance is positive across ALL of them (a filter that
    helps one variant but not its siblings is the snooping trap the record already caught -- the
    consec-loss trap; SYSTEM §C.5). This is a pass/fail robustness read, NEVER a P&L threshold grid: it
    requires every margin > 0, not a tuned cutoff. An empty variant set cannot demonstrate consistency
    (there is nothing to be consistent across) and FAILS.
    """
    margins = list(variant_margins)
    if not margins:
        return ClauseResult("cross_variant", passed=False, detail="no variants/folds to check")
    all_positive = all(mg > 0.0 for mg in margins)
    worst = min(margins)
    return ClauseResult(
        "cross_variant",
        passed=all_positive,
        detail=f"{sum(1 for mg in margins if mg > 0)}/{len(margins)} variants positive; "
        f"worst margin {worst:+.4f}",
    )


# ==================================================================================================
# Walk-forward + embargo (the CONSISTENCY overlay -- SYSTEM §C.2; reuses system.split)
# ==================================================================================================
def walk_forward_folds(
    trades: Sequence[TradeRecord],
    days: Sequence[dt.date],
    boundaries: Sequence[dt.date],
    embargo_days: int,
) -> list[FoldSpec]:
    """Anchored walk-forward folds with an embargo at every train<->OOS boundary (SYSTEM §C.2).

    REUSES ``system.split`` at each boundary: the train side is the expanding (embargoed) window of
    trades that exit before the boundary, the test side is that fold's forward OOS window. Parameters
    are re-fit and frozen per fold by the caller; these folds feed the CONSISTENCY overlay ONLY
    (SYSTEM §C.3 -- never ``N_eff``, never the adopt point estimate). ``embargo_days`` is
    ``cfg.hold`` lake session days (SYSTEM §C.2, §3.1), the same width the headline split uses. The
    3-year sample bears only the SMALLEST fold count (few wide folds, not many thin ones, SYSTEM §C.2)
    -- the caller passes those few boundaries; this helper only slices, it does not proliferate folds.

    Each fold's ``test`` is the trades whose ``exit_date`` falls in ``[boundary, next_boundary)`` (the
    forward OOS window), so folds do not overlap on the OOS side; the ``train`` is the anchored
    (expanding-from-the-start) embargoed window before ``boundary``.
    """
    ordered = sorted(boundaries)
    folds: list[FoldSpec] = []
    for i, boundary in enumerate(ordered):
        sp = split(trades, days, boundary, embargo_days=embargo_days)
        # The forward OOS window ends at the next boundary (anchored WF steps forward, non-overlapping).
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            test = [t for t in sp.test if t[0] < nxt]
        else:
            test = sp.test
        folds.append(FoldSpec(boundary=boundary, train=sp.train, test=test))
    return folds


# ==================================================================================================
# The single adoption conjunction (SYSTEM §C.6 == docs/EDGE.md clauses 1-5)
# ==================================================================================================
def adopt(
    candidate: CandidateResult,
    incumbent: CandidateResult,
    *,
    n_trials_eff: int,
    variant_margins: Sequence[float],
    pbo: PBOResult,
    penalty_per_param: float = IC_PENALTY_PER_PARAM,
    dsr_pass: float = DSR_PASS,
    seed: int = 0,
) -> AdoptionDecision:
    """The ONE Phase-C adoption conjunction on the HELD-OUT OOS slice (SYSTEM §C.6 == EDGE clauses 1-5).

    A candidate REPLACES the incumbent IF AND ONLY IF, on the held-out OOS slice, ALL of:
      0. **On-ledger** (SYSTEM §C.3, §C.9) -- ``candidate.name`` is one of the 8 frozen ledger names;
         an off-ledger name is a snooped trial by definition and cannot be adopted (this precedes the
         5 numbered clauses -- an off-ledger candidate is never scored, only rejected).
      1. **OOS out-performance NET of the parsimony penalty** (SYSTEM §C.4, EDGE 1) -- the candidate's
         IC-penalized objective (``ic_penalized_objective`` = OOS EV net of its ``k``-penalty) STRICTLY
         beats the incumbent's. A tie resolves to the incumbent (the parsimony floor).
      2. **DSR / PBO multiplicity** (SYSTEM §C.3, EDGE 2) -- ``system.deflated_sharpe`` on the candidate
         OOS returns with the COMPOSED, correlation-deflated ``n_trials_eff`` (NOT the additive 8)
         PASSES (``dsr >= dsr_pass`` and ``n_obs >= floor``), AND ``pbo.passed`` (PBO below its ceiling
         via CSCV over the pooled candidate matrix).
      3. **Drop-top-N robust** (SYSTEM §C.5, EDGE 3) -- ``drop_top_n`` survives drop-top-1 AND
         drop-top-3 on the paired difference.
      4. **Cross-variant consistent** (SYSTEM §C.5, EDGE 4) -- ``cross_variant_consistent`` over the
         candidate's sibling variants / WF-fold margins.
      5. **Paired-bootstrap CI clears zero** (SYSTEM §C.6, EDGE 5) -- the ``B=20000, block=1`` 90% CI
         of the Sharpe difference vs the incumbent EXCLUDES zero.

    Miss ANY clause => NOT adopted; ``reason`` names the FIRST failing clause and the negative is
    recorded (the incumbent is kept -- SYSTEM §C.6, docs/EDGE.md append-never-rewrite). The candidate
    and incumbent OOS trade streams MUST be matched pairs (SYSTEM §C.5); a mismatch raises upstream in
    the paired helpers. This function computes NO metric on the training folds and touches NO lake --
    the WF-fold margins are pre-computed by the caller and passed in as ``variant_margins`` (SYSTEM
    §C.3 -- the folds feed clause 4 only).
    """
    clauses: list[ClauseResult] = []

    # Clause 0 -- on-ledger gate (SYSTEM §C.3 / §C.9): an off-ledger name is a snooped trial.
    on_ledger = candidate.name in FROZEN_LEDGER
    clauses.append(ClauseResult(
        "on_ledger", passed=on_ledger,
        detail=f"'{candidate.name}' {'in' if on_ledger else 'NOT in'} the frozen 8-candidate ledger"))

    # Clause 1 -- IC-penalized OOS out-performance (strict; tie -> incumbent, the parsimony floor).
    cand_obj = ic_penalized_objective(
        candidate.oos_trades, candidate.k, penalty_per_param=penalty_per_param)
    inc_obj = ic_penalized_objective(
        incumbent.oos_trades, incumbent.k, penalty_per_param=penalty_per_param)
    beats = cand_obj > inc_obj
    clauses.append(ClauseResult(
        "oos_outperformance_net_penalty", passed=beats,
        detail=f"candidate penalized-EV {cand_obj:+.5f} vs incumbent {inc_obj:+.5f} "
        f"(k {candidate.k} vs {incumbent.k}, penalty/param {penalty_per_param})"))

    # Clause 2 -- DSR (composed N_eff, never the additive 8) AND PBO (CSCV over the pooled matrix).
    # `dsr_pass` is threaded straight into `system.deflated_sharpe` so the clause-2 verdict
    # (`dsr.passed`) is decided against THIS gate threshold, not `system`'s own default -- the detail
    # string's "(pass {dsr_pass})" and the actual admission bar can never diverge (D1). Under the
    # FROZEN pre-registration this is `DSR_PASS == 0.95`; a stricter caller value (e.g. 0.99) now
    # genuinely raises the luck bar instead of being a dead, misleading knob.
    dsr: DeflatedSharpeResult = deflated_sharpe(
        candidate.oos_trades, n_trials_eff=n_trials_eff, dsr_pass=dsr_pass)
    dsr_ok = dsr.passed
    clauses.append(ClauseResult(
        "dsr", passed=dsr_ok,
        detail=f"dsr {dsr.dsr:.4f} (pass {dsr_pass}), n_obs {dsr.n_obs}, "
        f"n_trials_eff {n_trials_eff}: {dsr.reason}"))
    clauses.append(ClauseResult(
        "pbo", passed=pbo.passed,
        detail=f"pbo {pbo.pbo:.3f} <= ceiling {pbo.ceiling} over {pbo.n_splits} CSCV splits"))

    # Clause 3 -- drop-top-N robustness on the paired difference.
    dtn = drop_top_n(candidate.oos_trades, incumbent.oos_trades)
    clauses.append(ClauseResult(
        "drop_top_n", passed=dtn.passed,
        detail=f"margin full {dtn.margin_full:+.4f} / drop1 {dtn.margin_drop1:+.4f} / "
        f"drop3 {dtn.margin_drop3:+.4f}"))

    # Clause 4 -- cross-variant consistency (WF folds feed here only, never N_eff or the point est).
    cvc = cross_variant_consistent(variant_margins)
    clauses.append(cvc)

    # Clause 5 -- paired-bootstrap Sharpe-difference 90% CI clears zero.
    ci = incumbent_vs_candidate_ci(
        candidate.oos_trades, incumbent.oos_trades,
        B=BOOTSTRAP_B, alpha=BOOTSTRAP_ALPHA, seed=seed)
    ci_ok = ci.lo > 0.0  # the entire 90% CI is above zero (excludes zero on the favorable side)
    clauses.append(ClauseResult(
        "paired_bootstrap_ci", passed=ci_ok,
        detail=f"Sharpe-diff 90% CI [{ci.lo:+.4f}, {ci.hi:+.4f}], point {ci.point:+.4f}"))

    adopted = all(c.passed for c in clauses)
    if adopted:
        reason = "ADOPTED -- all clauses pass on the held-out OOS slice"
    else:
        first_fail = next(c for c in clauses if not c.passed)
        reason = f"KEEP INCUMBENT -- failed clause '{first_fail.name}': {first_fail.detail}"
    return AdoptionDecision(
        candidate=candidate.name, incumbent=incumbent.name,
        adopted=adopted, reason=reason, clauses=tuple(clauses))
