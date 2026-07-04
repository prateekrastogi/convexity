"""SELECT.protocol tests -- SYNTHETIC mock CandidateResults only (no real lake, no real models).

The select unit is the interface-driven SELECTION HARNESS (SYSTEM §C.3): it is exercised with
SYNTHETIC ``CandidateResult`` streams whose truth we CONTROL, never by importing the real Phase-C
model modules. Two mandated flavours (task contract):
  1. a KNOWN-DECISION RECOVERY test -- construct a candidate whose adopt/reject verdict is KNOWN by
     construction (a genuine edge that clears all 5 clauses => ADOPT; a lucky/friction/off-ledger one
     => KEEP INCUMBENT) and assert the harness recovers that verdict;
  2. a NO-PEEK PINNING test -- the harness reads the HELD-OUT OOS slice ONLY and never the training
     folds / WF-fold refits into the adopt point estimate (SYSTEM §C.3), and every clause is a
     DIRECTION/INEQUALITY (a `<`-for-`>` sign flip in a duplicated formula must break a test -- the
     convexity sign-trap lesson).
Each assert checks a direction or an inequality, not just a magnitude.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from entry_exit._core import TradeRecord
from entry_exit.base import CandidateResult
from entry_exit.selection import (
    DSR_PASS,
    FROZEN_LEDGER,
    SPLIT,
    adopt,
    cross_variant_consistent,
    drop_top_n,
    ic_penalized_objective,
    incumbent_vs_candidate_ci,
    pbo_cscv,
    walk_forward_folds,
)
from entry_exit.validation import DSR_PASS as SYSTEM_DSR_PASS
from entry_exit.validation import split


# --- synthetic helpers ----------------------------------------------------------------------------
def _session_days(start: dt.date, n: int) -> list[dt.date]:
    """`n` consecutive WEEKDAY session days from `start` (weekends skipped -- lake-calendar-like)."""
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _stream(returns, start=dt.date(2024, 1, 1)):
    """A CandidateResult-style trade stream: one TradeRecord per return on consecutive session days."""
    days = _session_days(start, len(returns))
    return [TradeRecord(d, float(r)) for d, r in zip(days, returns)]


def _candidate(name, returns, k=1):
    return CandidateResult(name=name, oos_trades=_stream(returns), k=k)


def _incumbent(returns, k=0):
    # The incumbent is the parsimony floor -- k minimal; name is a non-ledger baseline label.
    return CandidateResult(name="incumbent_orb_5d_stop", oos_trades=_stream(returns), k=k)


# ==================================================================================================
# The frozen ledger + constants (the ONE enumerable set N_eff and the off-ledger rule are charged on)
# ==================================================================================================
def test_frozen_ledger_is_the_eight_pre_registered_candidates():
    # SYSTEM §C.9 / SIGNAL §C.2-C.4 / docs/EDGE.md: exactly 8, 4 entry + 3 exit + 1 vol input.
    assert len(FROZEN_LEDGER) == 8
    assert len(set(FROZEN_LEDGER)) == 8  # no duplicate names
    for name in ("bocpd", "sprt_glr", "hmm_filtered", "lee_mykland",
                 "optimal_stopping", "ou_leung_li", "survival_hazard", "har_rv"):
        assert name in FROZEN_LEDGER


def test_split_boundary_is_the_frozen_headline_gate():
    assert SPLIT == dt.date(2025, 6, 1)  # SYSTEM §2.1 -- never tuned


def test_reuses_system_dsr_pass_threshold():
    # The gate threshold is SYSTEM-owned (0.95); select must not fork a second value.
    assert DSR_PASS == SYSTEM_DSR_PASS == 0.95


# ==================================================================================================
# Clause 1 -- the IC/parsimony penalty is INSIDE the objective (SYSTEM §C.4)
# ==================================================================================================
def test_ic_penalty_docks_the_richer_model():
    # Equal raw EV, more parameters => STRICTLY lower penalized objective (parsimony floor, §C.4).
    tr = _stream([0.1, 0.1, 0.1, 0.1])
    lean = ic_penalized_objective(tr, k=1, penalty_per_param=0.01)
    rich = ic_penalized_objective(tr, k=5, penalty_per_param=0.01)
    assert rich < lean  # DIRECTION: the extra parameters are docked
    assert lean - rich == pytest.approx(0.04, abs=1e-12)  # (5-1)*0.01


def test_ic_penalty_zero_reduces_to_raw_ev():
    tr = _stream([0.2, -0.1, 0.3])
    assert ic_penalized_objective(tr, k=9, penalty_per_param=0.0) == pytest.approx(
        np.mean([0.2, -0.1, 0.3]), abs=1e-12)


def test_ic_penalty_rejects_negative_k():
    with pytest.raises(ValueError):
        ic_penalized_objective(_stream([0.1]), k=-1)


# ==================================================================================================
# Clause 3 -- drop-top-N robustness on the PAIRED difference (SYSTEM §C.5, the fat-tail tell)
# ==================================================================================================
def test_drop_top_n_kills_a_one_trade_lottery():
    # Candidate ties the incumbent on all trades EXCEPT one giant winner -> margin is one lucky trade.
    inc = [0.0] * 8
    cand = [0.0] * 7 + [5.0]  # the whole margin is trade #8
    res = drop_top_n(_stream(cand), _stream(inc))
    assert res.margin_full > 0             # looks like an edge on the full sample
    assert res.margin_drop1 == pytest.approx(0.0, abs=1e-12)  # ... but evaporates on drop-top-1
    assert not res.passed                  # DIRECTION: a one-trade lottery is rejected


def test_drop_top_n_survives_a_broad_edge():
    # A genuinely broad edge: candidate beats the incumbent on EVERY trade -> survives both drops.
    inc = [0.0] * 8
    cand = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.08, 0.05]
    res = drop_top_n(_stream(cand), _stream(inc))
    assert res.passed
    assert res.margin_drop3 > 0            # the margin persists after dropping the top-3


def test_drop_top_n_thin_stream_cannot_clear_drop3():
    # <= 3 trades: dropping the top-3 leaves no surviving sample -> cannot pass (the lottery guard).
    res = drop_top_n(_stream([0.1, 0.2, 0.3]), _stream([0.0, 0.0, 0.0]))
    assert not res.passed


def test_drop_top_n_raises_on_unmatched_pairs():
    with pytest.raises(ValueError):
        drop_top_n(_stream([0.1, 0.2, 0.3]), _stream([0.0, 0.0]))


# ==================================================================================================
# Clause 2b -- PBO via CSCV (SYSTEM §C.3); a truly-overfit matrix scores high PBO
# ==================================================================================================
def test_pbo_flags_a_constructed_overfit_matrix():
    # A candidate ENGINEERED to be in-sample-best on the early blocks but out-of-sample-worst on the
    # late ones -> the snooper's in-sample pick is systematically the OOS loser => PBO high (overfit).
    # Two candidates: col 0 is huge on the first half of the rows (any train subset heavy on early
    # blocks makes it IS-best) and terrible on the second half (so its OOS half is worst).
    n = 64
    perf = np.zeros((n, 2))
    half = n // 2
    perf[:half, 0] = 10.0       # col 0 dominates in-sample on early blocks
    perf[half:, 0] = -10.0      # ... and is worst out-of-sample on late blocks
    perf[:half, 1] = -1.0       # col 1 is the mirror: weak early, strong late
    perf[half:, 1] = 1.0
    res = pbo_cscv(perf, n_blocks=8)
    assert res.n_splits == 70            # C(8,4) balanced CSCV splits
    assert res.pbo > 0.5                 # DIRECTION: the IS winner is the OOS loser -> overfit
    assert not res.passed                # above the 0.5 ceiling -> rejected regardless of point OOS


def test_pbo_passes_a_dominant_candidate_and_is_lower_than_the_overfit_case():
    # One candidate dominates in EVERY time slice -> its in-sample rank always survives OOS => PBO 0.
    rng = np.random.default_rng(1)
    perf = rng.standard_normal((64, 5)) * 0.1
    perf[:, 0] += 3.0  # candidate 0 dominates everywhere
    res = pbo_cscv(perf, n_blocks=8)
    assert res.pbo == pytest.approx(0.0, abs=1e-12)  # DIRECTION: the true winner never gets overfit
    assert res.passed
    # and it is strictly more robust than the constructed-overfit matrix (the discriminating direction)
    overfit = np.zeros((64, 2))
    overfit[:32, 0], overfit[32:, 0] = 10.0, -10.0
    overfit[:32, 1], overfit[32:, 1] = -1.0, 1.0
    assert res.pbo < pbo_cscv(overfit, n_blocks=8).pbo


def test_pbo_needs_two_candidates_and_even_blocks():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((32, 1)))          # a single candidate has no rank to overfit
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((32, 3)), n_blocks=7)  # odd blocks cannot form balanced halves


# ==================================================================================================
# Clause 5 -- paired-bootstrap Sharpe-difference CI (SYSTEM §C.6; the futures-vs-options killer)
# ==================================================================================================
def test_paired_ci_straddles_zero_for_a_marginal_candidate():
    # A candidate barely different from the incumbent -> the 90% CI STRADDLES zero (not adoptable).
    rng = np.random.default_rng(3)
    base = rng.standard_normal(60) * 0.2
    inc = _stream(base)
    cand = _stream(base + rng.standard_normal(60) * 0.01)  # near-identical, tiny noise
    ci = incumbent_vs_candidate_ci(cand, inc, B=2000, seed=0)
    assert ci.lo < 0.0 < ci.hi           # DIRECTION: the interval spans zero -> clause 5 FAILS


def test_paired_ci_clears_zero_for_a_strong_candidate():
    # Candidate strictly better on every matched trade -> a positive Sharpe-diff CI above zero.
    inc = _stream([-0.05, 0.02, -0.03, 0.01, -0.04, 0.03, -0.02, 0.02] * 4)
    cand = _stream([0.06, 0.07, 0.05, 0.06, 0.05, 0.07, 0.06, 0.05] * 4)
    ci = incumbent_vs_candidate_ci(cand, inc, B=2000, seed=0)
    assert ci.point > 0.0
    assert ci.lo > 0.0                   # the whole 90% CI clears zero -> clause 5 PASSES


def test_paired_ci_raises_on_unmatched_pairs():
    with pytest.raises(ValueError):
        incumbent_vs_candidate_ci(_stream([0.1, 0.2]), _stream([0.0, 0.0, 0.0]), B=10)


# ==================================================================================================
# Clause 4 -- cross-variant consistency (SYSTEM §C.5, the anti-snooping mechanism)
# ==================================================================================================
def test_cross_variant_needs_all_positive():
    assert cross_variant_consistent([0.02, 0.03, 0.01]).passed             # all positive -> consistent
    assert not cross_variant_consistent([0.02, -0.01, 0.03]).passed        # one negative -> snooping trap
    assert not cross_variant_consistent([]).passed                         # nothing to be consistent across


# ==================================================================================================
# Walk-forward + embargo overlay reuses system.split (SYSTEM §C.2)
# ==================================================================================================
def test_walk_forward_folds_partition_forward_and_embargo():
    days = _session_days(dt.date(2024, 1, 1), 40)
    # one trade per session day, exit_date = that day
    trades = [TradeRecord(d, 0.01) for d in days]
    b1, b2 = days[15], days[30]
    folds = walk_forward_folds(trades, days, [b2, b1], embargo_days=0)  # unsorted input -> sorted
    assert [f.boundary for f in folds] == [b1, b2]                      # folds are boundary-ordered
    # fold 0's OOS window is [b1, b2); fold 1's is [b2, end)
    assert all(b1 <= t[0] < b2 for t in folds[0].test)
    assert all(t[0] >= b2 for t in folds[1].test)
    # anchored: fold 1 trains on strictly more history than fold 0 (expanding window)
    assert len(folds[1].train) > len(folds[0].train)


def test_walk_forward_embargo_shrinks_train_only():
    days = _session_days(dt.date(2024, 1, 1), 40)
    trades = [TradeRecord(d, 0.01) for d in days]
    boundary = days[20]
    no_emb = walk_forward_folds(trades, days, [boundary], embargo_days=0)[0]
    emb = walk_forward_folds(trades, days, [boundary], embargo_days=5)[0]
    assert len(emb.train) <= len(no_emb.train)   # embargo only SHRINKS train
    assert len(emb.test) == len(no_emb.test)     # test is untouched by the embargo (SYSTEM §5.2)


# ==================================================================================================
# THE adoption conjunction -- KNOWN-decision RECOVERY (SYSTEM §C.6 == EDGE clauses 1-5)
# ==================================================================================================
def _strong_edge_setup():
    """A candidate KNOWN by construction to clear all 5 clauses: a broad, robust, on-ledger edge."""
    # 60 matched trades: candidate strictly beats the incumbent on every trade by a steady margin,
    # so EV up, Sharpe-diff CI clears zero, drop-top-N survives, cross-variant positive.
    rng = np.random.default_rng(7)
    inc_r = rng.standard_normal(60) * 0.02 - 0.01     # incumbent: small, noisy, slightly negative EV
    cand_r = inc_r + 0.04 + rng.standard_normal(60) * 0.005  # steady +0.04 edge on every matched trade
    cand = CandidateResult("hmm_filtered", _stream(cand_r), k=5)
    inc = CandidateResult("incumbent_orb_5d_stop", _stream(inc_r), k=2)
    return cand, inc


def test_adopt_recovers_a_genuine_edge():
    cand, inc = _strong_edge_setup()
    # A dominant pooled matrix -> low PBO; DSR fed a modest composed N_eff (not the additive 8).
    rng = np.random.default_rng(8)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += np.asarray(cand.returns())[:64] if cand.n_obs >= 64 else 1.0
    pbo = pbo_cscv(perf, n_blocks=8)
    decision = adopt(
        cand, inc, n_trials_eff=12, variant_margins=[0.03, 0.04, 0.02],
        pbo=pbo, penalty_per_param=0.001, seed=0)
    assert decision.adopted, decision.reason
    assert all(c.passed for c in decision.clauses)
    assert decision.candidate == "hmm_filtered"


def test_adopt_rejects_off_ledger_candidate():
    # An off-ledger NAME is a snooped trial by definition (SYSTEM §C.3) -- rejected before scoring.
    cand, inc = _strong_edge_setup()
    off = CandidateResult("rsi_crossover", cand.oos_trades, k=cand.k)  # NOT in FROZEN_LEDGER
    rng = np.random.default_rng(8)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    decision = adopt(off, inc, n_trials_eff=12, variant_margins=[0.03], pbo=pbo)
    assert not decision.adopted
    assert "on_ledger" in decision.reason  # the FIRST failing clause is the ledger membership gate


def test_adopt_rejects_when_parsimony_penalty_erases_the_margin():
    # Candidate beats raw EV by LESS than its added-complexity penalty -> clause 1 FAILS (§C.4).
    inc_r = [0.010] * 60
    cand_r = [0.011] * 60           # +0.001 raw EV, but 8 extra params at 0.01 each = -0.08 penalty
    cand = CandidateResult("bocpd", _stream(cand_r), k=10)
    inc = CandidateResult("incumbent_orb_5d_stop", _stream(inc_r), k=2)
    rng = np.random.default_rng(9)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    decision = adopt(
        cand, inc, n_trials_eff=12, variant_margins=[0.001],
        pbo=pbo, penalty_per_param=0.01)
    assert not decision.adopted
    assert "oos_outperformance_net_penalty" in decision.reason  # the penalty is INSIDE the gate


def test_adopt_rejects_when_ci_straddles_zero():
    # Near-identical candidate: raw EV barely higher, but the paired Sharpe-diff CI straddles zero.
    rng = np.random.default_rng(11)
    base = rng.standard_normal(60) * 0.2
    inc = CandidateResult("incumbent_orb_5d_stop", _stream(base), k=2)
    cand = CandidateResult("bocpd", _stream(base + 0.001 + rng.standard_normal(60) * 0.05), k=3)
    rng2 = np.random.default_rng(12)
    perf = rng2.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    decision = adopt(
        cand, inc, n_trials_eff=12, variant_margins=[0.001],
        pbo=pbo, penalty_per_param=0.0, seed=0)
    assert not decision.adopted
    # the negative names a specific failing clause (not adopted silently); incumbent is kept
    assert decision.incumbent == "incumbent_orb_5d_stop"


def test_adopt_rejects_when_cross_variant_inconsistent():
    cand, inc = _strong_edge_setup()
    rng = np.random.default_rng(8)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    # one variant NEGATIVE -> the snooping-trap check fails even though the pooled edge is strong
    decision = adopt(
        cand, inc, n_trials_eff=12, variant_margins=[0.03, -0.01, 0.02],
        pbo=pbo, penalty_per_param=0.001, seed=0)
    assert not decision.adopted
    assert "cross_variant" in decision.reason


def test_adopt_is_a_strict_conjunction_incumbent_wins_ties():
    # Equal penalized objective (a tie) resolves to the incumbent (the parsimony floor, §C.6).
    tied = [0.01] * 60
    cand = CandidateResult("bocpd", _stream(tied), k=2)
    inc = CandidateResult("incumbent_orb_5d_stop", _stream(tied), k=2)
    rng = np.random.default_rng(13)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    decision = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.0], pbo=pbo)
    assert not decision.adopted  # a tie is NOT a strict win -> incumbent stays


# ==================================================================================================
# NO-PEEK PINNING -- the adopt statistic reads the held-out OOS slice ONLY (SYSTEM §C.3)
# ==================================================================================================
def test_adopt_reads_held_out_oos_only_train_side_is_pinned_out():
    """PINNING (no-peek / no-train-leak): the verdict depends ONLY on the post-SPLIT held-out slice.

    The adopt point estimate is computed on the OOS slice ONLY (SYSTEM §C.3): the held-out portion is
    the `test` side of `system.split` at the frozen SPLIT. We lay TWO full pre+post-SPLIT streams whose
    POST-SPLIT (held-out) trades are BYTE-IDENTICAL but whose PRE-SPLIT (train) trades differ WILDLY
    (one benign, one forged with giant fake winners). Run each through `system.split(SPLIT)`, feed only
    the `test` side to `adopt`, and require the two verdicts to be identical. This FAILS the moment the
    slicing pipeline (or a future edit) lets a pre-SPLIT train trade reach the DSR / CI / drop-top-N /
    clause-1 objective -- a genuine train-leak, not merely bootstrap non-determinism. It is NOT a
    tautology: the two INPUT full-streams are materially different; only their held-out slices agree.
    """
    rng = np.random.default_rng(21)
    n = 260  # enough session days to straddle 2025-06-01 with >= 24 trades on the held-out side
    start = dt.date(2025, 1, 1)
    days = _session_days(start, n)
    inc_full = list(rng.standard_normal(n) * 0.02 - 0.01)
    cand_full = [ir + 0.04 + z for ir, z in zip(inc_full, rng.standard_normal(n) * 0.005)]

    def _splits(returns):
        full = [TradeRecord(d, float(r)) for d, r in zip(days, returns)]
        return split(full, days, SPLIT, embargo_days=0)

    cand_sp, inc_sp = _splits(cand_full), _splits(inc_full)
    cand_oos = CandidateResult("hmm_filtered", cand_sp.test, k=5)
    inc_oos = CandidateResult("incumbent_orb_5d_stop", inc_sp.test, k=2)
    assert cand_oos.n_obs >= 24 and inc_oos.n_obs >= 24  # a scorable held-out slice (n_obs floor)

    # A second stream: SAME post-SPLIT trades, FORGED pre-SPLIT train side (giant fake winners).
    forged = [999.0 if d < SPLIT else r for d, r in zip(days, cand_full)]
    forged_sp = _splits(forged)
    assert forged_sp.test == cand_sp.test          # held-out slices are byte-identical ...
    assert forged_sp.train != cand_sp.train        # ... while the train side is wildly different
    cand_forged = CandidateResult("hmm_filtered", forged_sp.test, k=5)

    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)
    base = adopt(cand_oos, inc_oos, n_trials_eff=12, variant_margins=[0.03, 0.04], pbo=pbo,
                 penalty_per_param=0.001, seed=0)
    leaky = adopt(cand_forged, inc_oos, n_trials_eff=12, variant_margins=[0.03, 0.04], pbo=pbo,
                  penalty_per_param=0.001, seed=0)
    assert leaky.adopted == base.adopted        # a divergent train side cannot move the verdict
    assert leaky.reason == base.reason          # ... nor the recorded rationale
    assert base.adopted                          # (and the genuine held-out edge is in fact ADOPTED)


def test_adopt_wf_fold_identity_is_irrelevant_except_through_clause4_sign():
    """PINNING the clause-4 direction: WF-fold ORDER is irrelevant, but a NEGATIVE margin BREAKS adoption.

    The walk-forward folds reach `adopt` ONLY through `variant_margins` (SYSTEM §C.3 consistency
    overlay) and clause 4 is the order-invariant all-positive rule (`cross_variant_consistent`). So:
      * permuting `variant_margins` (relabelling which fold is which) must NOT change the verdict, and
      * flipping any single margin NEGATIVE (a genuine sign event on one fold) MUST flip the verdict to
        NOT-adopted with the failing clause named `cross_variant`.
    The second half is the `<`-for-`>` sign-trap discharge: were clause 4 wired with a flipped
    inequality (accept-any-negative / require-all-negative), this assert would fail.
    """
    cand, inc = _strong_edge_setup()  # a known all-clauses-pass edge
    rng = np.random.default_rng(8)
    perf = rng.standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)

    ordered = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.03, 0.04, 0.02], pbo=pbo,
                    penalty_per_param=0.001, seed=0)
    permuted = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.02, 0.04, 0.03], pbo=pbo,
                     penalty_per_param=0.001, seed=0)
    assert ordered.adopted == permuted.adopted   # fold IDENTITY (order) is irrelevant
    assert ordered.reason == permuted.reason
    assert ordered.adopted                        # the base edge clears every clause

    # A single NEGATIVE fold margin is a real sign event -> clause 4 must FAIL and block adoption.
    flipped = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.03, -0.04, 0.02], pbo=pbo,
                    penalty_per_param=0.001, seed=0)
    assert not flipped.adopted                    # DIRECTION: one negative fold => snooping trap
    assert "cross_variant" in flipped.reason


def test_adopt_honours_the_dsr_pass_knob_in_the_actual_verdict():
    """PINNING D1: the `dsr_pass` gate threshold decides clause 2, it is not a display-only string.

    `adopt` accepts `dsr_pass` and must thread it into `system.deflated_sharpe`, so the actual clause-2
    verdict is computed against THIS threshold -- not `system`'s hardcoded default. We pin the edge:
    a candidate whose DSR lands strictly BETWEEN a lax and a strict pass bar is ADMITTED at the lax bar
    and REJECTED at the strict one, everything else held fixed. This FAILS if `dsr_pass` is only
    interpolated into the detail string (the D1 dead-knob) rather than passed to `deflated_sharpe`.
    """
    from entry_exit.validation import deflated_sharpe
    cand, inc = _strong_edge_setup()
    perf = np.random.default_rng(8).standard_normal((64, 4)) * 0.05
    perf[:, 0] += 5.0
    pbo = pbo_cscv(perf, n_blocks=8)

    # Locate the candidate's realized DSR so we can bracket it with a lax/strict gate.
    dsr_val = deflated_sharpe(cand.oos_trades, n_trials_eff=12).dsr
    lax = max(0.0, dsr_val - 0.05)      # a bar the candidate clears
    strict = min(1.0, dsr_val + 1e-6)   # a bar just above its DSR -> it must FAIL clause 2
    assert lax < dsr_val < strict       # the DSR is genuinely bracketed

    lax_dec = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.03, 0.04, 0.02], pbo=pbo,
                    penalty_per_param=0.001, dsr_pass=lax, seed=0)
    strict_dec = adopt(cand, inc, n_trials_eff=12, variant_margins=[0.03, 0.04, 0.02], pbo=pbo,
                       penalty_per_param=0.001, dsr_pass=strict, seed=0)
    # find the dsr clause verdict in each decision
    lax_dsr = next(c for c in lax_dec.clauses if c.name == "dsr")
    strict_dsr = next(c for c in strict_dec.clauses if c.name == "dsr")
    assert lax_dsr.passed        # DIRECTION: DSR clears the lax bar
    assert not strict_dsr.passed  # ... and FAILS the strict bar -> the knob is live, not decorative
    assert strict_dec.adopted is False and "dsr" in strict_dec.reason  # strict gate blocks adoption


def test_dsr_uses_composed_n_eff_not_additive_eight():
    """PINNING the multiplicity leak: a larger composed N_eff raises the luck bar -> lower/equal DSR.

    Feeding the additive 8 instead of the composed, deflated space UNDER-counts N_eff and understates
    SR*_0 -- the exact FAIL->PASS flip the gate exists to prevent (SYSTEM §C.3). DSR is monotone
    non-increasing in n_trials_eff, so a study that (correctly) charges the larger composed count is
    never EASIER to pass than the additive-8 shortcut. This asserts the harness passes the caller's
    composed count straight through, never substituting 8.
    """
    from entry_exit.validation import deflated_sharpe
    cand, _ = _strong_edge_setup()
    dsr_additive = deflated_sharpe(cand.oos_trades, n_trials_eff=8).dsr
    dsr_composed = deflated_sharpe(cand.oos_trades, n_trials_eff=24).dsr
    assert dsr_composed <= dsr_additive  # DIRECTION: more trials => higher bar => not-larger DSR
