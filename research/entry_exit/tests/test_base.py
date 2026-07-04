"""Unit tests for MODELS.base -- the Phase-C candidate INTERFACE. SYNTHETIC ONLY, no real lake.

Two mandated purity tests (SYSTEM §C.2, task contract), plus protocol conformance on a TRIVIAL
synthetic candidate:
  1. a KNOWN-generating-process RECOVERY test -- a mock EntryModel whose parameter (a train-fit sign)
     is recovered from a controlled drift path, exercised through the causal `ScanCursor`;
  2. a NO-LOOKAHEAD PINNING test -- `require_causal` / `ScanCursor.read` RAISE if a decision at bar
     `t` reads bar `t` (or later) under the DAILY cadence (and after-instant under WITHIN_DAY).
The real Phase-C models are NOT imported (the harness/interface is tested with synthetic mocks only,
SYSTEM §C.3 off-ledger rule). No lake, no metric, no adopt decision is touched anywhere here.
"""

from __future__ import annotations

import datetime as dt

import pytest

from entry_exit._core import TradeRecord
from entry_exit.base import (
    STOP_FLOOR,
    BlockedOnLakeError,
    Cadence,
    CandidateResult,
    EntryModel,
    ExitModel,
    PurityError,
    ScanCursor,
    Side,
    TrainFold,
    require_causal,
    warmup_ok,
)


# --- Trivial SYNTHETIC candidates (mocks -- the interface is tested against these, NOT real models) --
class _MeanSignEntry:
    """A trivial synthetic EntryModel: fit the SIGN of the train-fold mean drift, then FROZEN.

    Emits that frozen sign as the entry `Side` whenever the causal past-window mean agrees with it --
    a one-parameter (`k=1`) model with a KNOWN generating truth (the train drift sign) the recovery
    test asserts against. Reads ONLY `cursor.past()`, so it is `F_{t-1}`-measurable by construction.
    """

    def __init__(self) -> None:
        self.k = 1
        self._sign: int | None = None  # set on fit, then frozen

    def fit(self, train: TrainFold) -> "_MeanSignEntry":
        s = sum(train.series[train.warmup:])
        self._sign = 1 if s > 0 else (-1 if s < 0 else 0)
        return self

    def signal_at(self, cursor: ScanCursor) -> Side:
        if self._sign is None:
            raise RuntimeError("signal_at before fit -- parameters not frozen yet")
        past = cursor.past()
        if not past:
            return Side.FLAT
        m = sum(past) / len(past)
        if self._sign > 0 and m > 0:
            return Side.LONG
        if self._sign < 0 and m < 0:
            return Side.SHORT
        return Side.FLAT


class _RunReturnExit:
    """A trivial synthetic ExitModel: exit once the running return clears a train-fit threshold.

    `k=1` (the take-profit level, fit as the train mean + never below `STOP_FLOOR`-consistent floor).
    Decides from `held_return` + `cursor.past()` only; never loosens the -12% stop (the floor stays
    the loss cap, this model only closes still-alive WINNERS earlier -- SIGNAL §C.3).
    """

    def __init__(self) -> None:
        self.k = 1
        self._tp: float | None = None

    def fit(self, train: TrainFold) -> "_RunReturnExit":
        body = train.series[train.warmup:] or (0.0,)
        self._tp = max(sum(body) / len(body), 0.0)  # a non-negative take-profit target
        return self

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        if self._tp is None:
            raise RuntimeError("exit_now before fit")
        _ = cursor.past()  # touch the causal window (a real model reads the filtered state here)
        _ = held_days      # duration covariate u is on the interface (a hazard model reads it; EDGE #7)
        return held_return >= self._tp


class _MaxHoldExit:
    """A trivial synthetic ExitModel whose KNOWN truth is an elapsed-holding-time threshold.

    Encodes the incumbent's OWN rule -- exit when `held_days >= horizon` -- through the interface,
    proving the elapsed-hold covariate `u` is FIRST-CLASS (the incumbent it "replaces" is expressible,
    and EDGE-#7's reversal hazard `h(u|.)` reads `u` directly, never smuggled via cursor re-basing).
    `k=1` (the horizon). The recovery test fits it to a known train duration and asserts `held_days`
    is load-bearing (the decision flips ONLY on `held_days`, holding `held_return` fixed).
    """

    def __init__(self) -> None:
        self.k = 1
        self._horizon: int | None = None

    def fit(self, train: TrainFold) -> "_MaxHoldExit":
        # KNOWN truth: the fixed hold horizon = the (integer) train series length past warm-up.
        self._horizon = len(train.series[train.warmup:])
        return self

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        if self._horizon is None:
            raise RuntimeError("exit_now before fit")
        _ = cursor.past()   # causal window (a real hazard model reads the filtered state here)
        _ = held_return     # a real hazard conditions on it too; here the truth is duration-only
        return held_days >= self._horizon  # F_{t-1}-measurable duration crossing (EDGE #7 / incumbent)


def _train(vals: list[float], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


# ==================================================================================================
# Protocol conformance (runtime_checkable structural typing on the trivial synthetic candidates)
# ==================================================================================================
def test_entry_model_conforms():
    m = _MeanSignEntry()
    assert isinstance(m, EntryModel)          # structural conformance (fit + signal_at + k)
    assert m.fit(_train([0.1, 0.2, 0.3])) is m  # fit returns the fitted model (self)
    assert isinstance(m.k, int) and m.k >= 0    # parsimony count present and a non-negative int


def test_exit_model_conforms():
    for m in (_RunReturnExit(), _MaxHoldExit()):  # both the value-fn and the duration/hazard shape
        assert isinstance(m, ExitModel)           # structural conformance (fit + exit_now + k)
        assert m.fit(_train([0.05, 0.10])) is m
        assert isinstance(m.k, int) and m.k >= 0


def test_non_conforming_object_is_not_a_model():
    class _Bad:  # no fit / signal_at / k
        pass

    assert not isinstance(_Bad(), EntryModel)
    assert not isinstance(_Bad(), ExitModel)


def test_side_is_the_orb_sign_convention():
    # int subclass -> transparent drop-in for the {+1, 0, -1} that orb.signal returns / §4 consumes.
    assert int(Side.LONG) == 1 and int(Side.FLAT) == 0 and int(Side.SHORT) == -1


# ==================================================================================================
# (1) KNOWN-generating-process RECOVERY test (train-fit param recovered from a controlled truth)
# ==================================================================================================
def test_recovery_positive_drift_recovers_long():
    # Controlled DGP: a strictly positive drift path -> the train-fit sign MUST be +1 (LONG).
    m = _MeanSignEntry().fit(_train([0.01, 0.02, 0.015, 0.03, 0.02]))
    assert m._sign == 1
    # And on a causal positive-past cursor the entry recovers LONG (never reading bar t).
    cur = ScanCursor(series=(0.01, 0.02, 0.03, 0.05), t=3, cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.LONG


def test_recovery_negative_drift_recovers_short():
    # Controlled DGP: a strictly negative drift path -> the train-fit sign MUST be -1 (SHORT).
    m = _MeanSignEntry().fit(_train([-0.01, -0.02, -0.03, -0.02]))
    assert m._sign == -1
    cur = ScanCursor(series=(-0.01, -0.02, -0.03, -0.05), t=3, cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.SHORT


def test_recovery_exit_threshold_from_known_train_mean():
    # Controlled DGP: train running-returns average 0.10 -> take-profit target recovers to 0.10.
    m = _RunReturnExit().fit(_train([0.10, 0.10, 0.10]))
    assert m._tp == pytest.approx(0.10)
    cur = ScanCursor(series=(0.02, 0.05, 0.09), t=3)
    assert m.exit_now(cur, held_days=2, held_return=0.15) is True   # cleared target -> close winner
    assert m.exit_now(cur, held_days=2, held_return=0.05) is False  # below target -> keep holding


def test_recovery_elapsed_hold_horizon_is_first_class():
    # Controlled DGP: a known 5-day hold horizon -> the model recovers horizon=5 from train length,
    # and the exit decision is driven BY THE ELAPSED-HOLD COVARIATE `held_days` (u), first-class on
    # the interface -- this is exactly the incumbent's "exit at hold>=5" rule (SIGNAL §C.0, SYSTEM
    # §C.1) and EDGE-#7's reversal-hazard duration `h(u|.)` (SIGNAL §C.3(c), SYSTEM §C.9 #7),
    # expressible WITHOUT re-basing the cursor to smuggle in `u`.
    m = _MaxHoldExit().fit(_train([0.0] * 5))
    assert m._horizon == 5
    cur = ScanCursor(series=(0.0, 0.0, 0.0, 0.0), t=3)
    # held_return is held FIXED; only held_days moves -> the decision flips ONLY on the duration `u`,
    # proving elapsed-hold is load-bearing and cannot be derived from (cursor, held_return) alone.
    assert m.exit_now(cur, held_days=4, held_return=0.30) is False  # u<5 -> keep holding
    assert m.exit_now(cur, held_days=5, held_return=0.30) is True   # u==5 -> incumbent horizon hit
    assert m.exit_now(cur, held_days=6, held_return=0.30) is True   # u>5  -> still exit


def test_incumbent_fixed_5day_horizon_is_expressible_through_the_interface():
    # The interface FAITHFULLY encodes the incumbent it claims to "replace": fixed 5-day hold is
    # EXACTLY `held_days >= 5`, expressible via `exit_now` with no entry-index / no cursor re-basing.
    m = _MaxHoldExit().fit(_train([0.0] * 5))
    cur = ScanCursor(series=(0.0,) * 10, t=7)   # cursor.t is the scan calendar pos, NOT the duration
    # Same execution bar (t=7), differing durations -> distinct decisions: duration is NOT t-derived.
    assert m.exit_now(cur, held_days=3, held_return=0.0) is False
    assert m.exit_now(cur, held_days=5, held_return=0.0) is True


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING test (must FAIL if a decision at bar t reads bar t or later)
# ==================================================================================================
def test_pinning_daily_same_bar_read_raises():
    # DAILY: a decision executing on bar t reading bar t (same bar) is a LEAK -> PurityError.
    with pytest.raises(PurityError):
        require_causal(read_index=5, decision_index=5, cadence=Cadence.DAILY)


def test_pinning_daily_future_bar_read_raises():
    with pytest.raises(PurityError):
        require_causal(read_index=7, decision_index=5, cadence=Cadence.DAILY)


def test_pinning_daily_past_bar_read_is_legal():
    # Strictly F_{t-1}-measurable: reading bar t-1 for a decision on t is fine (no raise).
    require_causal(read_index=4, decision_index=5, cadence=Cadence.DAILY)


def test_pinning_within_day_at_instant_legal_after_instant_raises():
    # WITHIN_DAY: reading AT the decision instant is legal; strictly after is a leak.
    require_causal(read_index=5, decision_index=5, cadence=Cadence.WITHIN_DAY)  # legal
    with pytest.raises(PurityError):
        require_causal(read_index=6, decision_index=5, cadence=Cadence.WITHIN_DAY)


def test_pinning_scancursor_read_guard():
    # The cursor routes reads through the guard: a same-bar DAILY read via ScanCursor.read RAISES,
    # while past() never exposes bar t (so a same-bar leak cannot be written through past()).
    cur = ScanCursor(series=(1.0, 2.0, 3.0, 4.0), t=2, cadence=Cadence.DAILY)
    assert cur.past() == (1.0, 2.0)          # indices 0..t-1 only -- F_{t-1}-measurable
    assert cur.read(1) == 2.0                # a strictly-past read is fine
    with pytest.raises(PurityError):
        cur.read(2)                          # same bar t -> leak
    with pytest.raises(PurityError):
        cur.read(3)                          # future bar -> leak


def test_pinning_negative_read_index_is_a_caller_bug_not_a_leak():
    # A malformed (negative) index is a ValueError, distinct from a lookahead PurityError.
    with pytest.raises(ValueError):
        require_causal(read_index=-1, decision_index=5, cadence=Cadence.DAILY)


# ==================================================================================================
# Warm-up honesty guard
# ==================================================================================================
def test_warmup_ok():
    assert warmup_ok(decision_index=22, warmup=22) is True    # exactly enough history
    assert warmup_ok(decision_index=21, warmup=22) is False   # still warming up
    assert warmup_ok(decision_index=0, warmup=0) is True      # no burn-in required
    with pytest.raises(ValueError):
        warmup_ok(decision_index=5, warmup=-1)


# ==================================================================================================
# CandidateResult container (data-only: OOS stream + parsimony k; no metric, no adopt decision)
# ==================================================================================================
def test_candidate_result_holds_oos_stream_and_k():
    trades = [
        TradeRecord(dt.date(2025, 6, 2), -0.12),
        TradeRecord(dt.date(2025, 6, 5), 0.40),
    ]
    r = CandidateResult(name="BOCPD", oos_trades=trades, k=5)
    assert r.name == "BOCPD"
    assert r.n_obs == 2
    assert r.returns() == [-0.12, 0.40]
    assert r.k == 5


def test_candidate_result_defaults_empty():
    r = CandidateResult(name="incumbent")
    assert r.n_obs == 0 and r.returns() == [] and r.k == 0


def test_candidate_result_rejects_negative_k():
    with pytest.raises(ValueError):
        CandidateResult(name="bad", k=-1)


# ==================================================================================================
# Frozen contract facts (the -12% stop floor is the retained loss cap; the lake sentinel exists)
# ==================================================================================================
def test_stop_floor_constant():
    assert STOP_FLOOR == 0.12  # the retained loss cap an ExitModel may never loosen (SIGNAL §C.3)


def test_blocked_on_lake_error_is_a_runtime_error():
    # The sentinel the RUN raises while the lake is absent -- named ONCE, shared by harness+candidates.
    assert issubclass(BlockedOnLakeError, RuntimeError)
    with pytest.raises(BlockedOnLakeError):
        raise BlockedOnLakeError("wide-band lake absent")


def test_trainfold_validates_alignment_and_warmup():
    with pytest.raises(ValueError):
        TrainFold(days=(dt.date(2024, 1, 1),), series=(1.0, 2.0))  # misaligned lengths
    with pytest.raises(ValueError):
        _train([1.0, 2.0], warmup=-1)


def test_scancursor_rejects_out_of_range_t():
    with pytest.raises(ValueError):
        ScanCursor(series=(1.0, 2.0), t=3)   # t beyond len
    with pytest.raises(ValueError):
        ScanCursor(series=(1.0, 2.0), t=-1)
