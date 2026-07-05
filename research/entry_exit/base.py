"""MODELS.base -- the Phase-C candidate INTERFACE (the one shared contract, no models).

The interface-only layer the Phase-C model-selection study (docs/specs/SIGNAL.md §C, SYSTEM.md §C,
docs/EDGE.md "The pre-registered candidate set (frozen)") is built against. It fixes the SHAPE every
pre-registered candidate must present so the SYSTEM selection harness can score them uniformly --
WITHOUT importing any concrete model. It defines:

  * ``EntryModel``  -- a runtime-checkable ``Protocol``: ``fit(train) -> self`` (train-fit, then
    parameters FROZEN) plus a causal one-step-ahead ``signal_at`` returning ``{+1, 0, -1}`` in the
    SAME sign convention as ``signal/orb.py`` (SIGNAL §3), feeding the *unchanged* §4 strike
    selection. It carries the free-parameter count ``k`` (SYSTEM §C.4 parsimony).
  * ``ExitModel``   -- a runtime-checkable ``Protocol``: ``fit(train) -> self`` plus a causal
    ``exit_now(cursor, held_days, held_return)`` hold/exit decision that REPLACES the fixed 5-day
    horizon while the true -12% 1-min premium stop is RETAINED as the floor (never loosened -- SIGNAL
    §C.3, "the stop IS the edge"). ``held_days`` is the elapsed-holding-time covariate ``u`` that makes
    the incumbent (``held_days >= 5``) and the EDGE-#7 reversal hazard ``h(u|.)`` first-class (not
    smuggled via cursor re-basing). It too carries ``k``.
  * ``CandidateResult`` -- the container the harness collects per candidate: the OOS trade stream
    (``list[TradeRecord]`` -- CORE's atomic unit) plus the parsimony free-parameter count ``k`` that
    the SYSTEM §C.4 information-criterion penalty is charged against.
  * shared CAUSAL-ESTIMATION helpers -- the small, model-agnostic pieces every candidate reuses to
    stay on the right side of the SIGNAL §C.1 / SYSTEM §C.2 purity line: a ``PurityError``, a
    ``Cadence`` tag (daily-decision vs within-day trigger), a ``require_causal`` measurability
    assertion that RAISES on a bar-``>= t`` read (daily) / after-instant read (within-day), a
    ``warmup_ok`` warm-up-honesty guard, and a ``TrainFold`` / ``ScanCursor`` pair that make the
    ``F_{t-1}``-measurable "decide on ``t-1`` info, execute on ``t``" offset explicit.

PURITY IS BINDING (SIGNAL §C.1, SYSTEM §C.2). Every candidate is admitted ONLY in its causal FILTERED
/ one-step-ahead ``F_{t-1}``-measurable form; a smoothed / full-sample / retrospective estimate, or a
same-bar ``RV_t`` read on a *daily* decision, is a LOOKAHEAD and inadmissible. Parameters are fit on
the TRAIN fold and FROZEN for OOS -- no re-fit or threshold re-tune on TEST. The helpers here give the
concrete models a single, tested place to enforce that; the pinning tests each model later carries
(SYSTEM §C.2) assert against ``require_causal`` breaking under a 1-bar leak.

DATA GATE (SYSTEM §C.8): this module is interface + synthetic-testable helpers ONLY. The empirical
selection RUN is BLOCKED-ON-LAKE (the wide-band / position-following lake is ABSENT). Nothing here
touches the lake, emits a metric, or adopts a candidate; the guard that RAISES ``BlockedOnLakeError``
lives with the RUN, and the sentinel is exported here so the harness and every candidate name the
SAME error (never a fabricated backtest).

Imports ONLY ``intraday.core`` shared contracts (``TradeRecord``, ``DDOF``, ``PERIODS_PER_YEAR``) --
NEVER a sibling Phase-C model module (the harness is interface-driven and tested with synthetic mock
candidates, not by importing the real models; SYSTEM §C.3 off-ledger rule). Pure Python + stdlib;
moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ._core import DDOF, PERIODS_PER_YEAR, TradeRecord

__all__ = [
    "PERIODS_PER_YEAR",
    "DDOF",
    "STOP_FLOOR",
    "Side",
    "Cadence",
    "PurityError",
    "BlockedOnLakeError",
    "TrainFold",
    "ScanCursor",
    "EntryModel",
    "ExitModel",
    "CandidateResult",
    "require_causal",
    "warmup_ok",
]

# --- The -12% premium stop is the RETAINED floor, never a candidate parameter (SIGNAL §C.3) --------
# The true 1-minute premium stop (`cfg.stop == 0.12`) is the validated edge; an ExitModel may only
# change WHEN a still-alive winner is voluntarily closed, NEVER loosen this loss cap. It is surfaced
# here as a named constant so an ExitModel's own conformance test can assert it does not raise the
# floor; the authoritative operative value stays `Inst.stop` in `signal/config.py` (this is the
# contract's documented floor, not a second live copy of the sizing parameter).
STOP_FLOOR = 0.12


class Side(int, Enum):
    """The ``{+1, 0, -1}`` directional signal convention (IDENTICAL to ``signal/orb.py``, SIGNAL §3).

    An ``EntryModel`` emits one of these, sign-antisymmetric; ``LONG`` -> a ~delta-0.65 ITM CALL,
    ``SHORT`` -> the mirror PUT, ``FLAT`` -> no trade -- feeding the *unchanged* §4 strike selection.
    It is an ``int`` subclass so ``int(side)`` is exactly the ``+1 / 0 / -1`` that ``orb.signal``
    returns and the strike-target code (``spot * (1 - moneyness)`` for ``+1``) already consumes -- a
    transparent drop-in, never a new wire shape.
    """

    LONG = 1
    FLAT = 0
    SHORT = -1


class Cadence(Enum):
    """The decision cadence that FIXES a candidate's measurability standard (SYSTEM §C.2 -- binding).

    The two purity standards must NOT be conflated (SYSTEM §C.2):
      * ``DAILY``      -- an end-of-day decision at day ``t`` must be strictly ``F_{t-1}``-measurable:
        it reads only bars *before* ``t``. A same-bar ``RV_t`` / same-bar filtered ``E[.|F_t]`` read
        is a LEAK here (the classic same-bar lookahead the HAR-RV covariate rule closes).
      * ``WITHIN_DAY`` -- an intra-session trigger (mirroring the incumbent's transact-at-the-close
        convention, SIGNAL §7) may use information UP TO its decision instant (the partial bar seen so
        far), but NEVER a bar dated after that instant.
    ``require_causal`` reads this tag to pick the correct barred set, so a candidate declares its
    cadence once and the pinning test is generated against the right rule.
    """

    DAILY = "daily"
    WITHIN_DAY = "within_day"


class PurityError(AssertionError):
    """A causal-purity violation: a decision read information it must not (SIGNAL §C.1, SYSTEM §C.2).

    Raised by ``require_causal`` when a decision at ``t`` reads a bar at index ``>= t`` (DAILY cadence,
    same-bar included) or a bar dated after the decision instant (WITHIN_DAY), or when a model is
    wired to a SMOOTHED / full-sample estimate instead of the FILTERED / one-step-ahead one. Subclasses
    ``AssertionError`` so a no-lookahead PINNING test (SYSTEM §C.2) that expects the leak to *break a
    test* catches it with ``pytest.raises((PurityError, AssertionError))`` either way.
    """


class BlockedOnLakeError(RuntimeError):
    """The wide-band / position-following lake is ABSENT -- the empirical RUN must RAISE, not fabricate.

    The sentinel error the SYSTEM Phase-C selection RUN raises while the lake is absent (SYSTEM §C.8):
    BLOCKED-ON-LAKE is an acceptable TERMINAL state; a fabricated backtest is not. Exported here so the
    RUN, the harness, and every candidate name the SAME error and no code path silently emits a metric.
    This module never raises it (it touches no lake); it only OWNS the name.
    """


# ==================================================================================================
# Train fold + the F_{t-1}-measurable scan cursor (make the decide-on-(t-1)/execute-on-t offset explicit)
# ==================================================================================================
@dataclass(frozen=True)
class TrainFold:
    """The TRAIN slice a candidate's parameters are fit on, then FROZEN (SIGNAL §C.1, SYSTEM §C.2).

    ``days`` are the LAKE SESSION days (matching ``B["days"]`` in ``signal/orb.py``) of the train
    window ONLY -- a candidate estimates its parameters here and freezes them; it never re-fits or
    re-tunes on TEST (SYSTEM §C.1). ``series`` is the per-day causal feature the model fits against
    (e.g. daily log-returns for BOCPD/SPRT/HMM drift, or realized variance for HAR-RV), aligned to
    ``days`` one-to-one. ``warmup`` is the model's own burn-in length (HMM filter burn-in, HAR's
    22-day lag, a change-point run-length prior): the model's scan may start only after ``warmup``
    days *within this fold* -- never borrowing pre-sample or cross-fold information (SIGNAL §C.1
    warm-up honesty; enforced by ``warmup_ok``).
    """

    days: tuple[dt.date, ...]
    series: tuple[float, ...]
    warmup: int = 0

    def __post_init__(self) -> None:
        if len(self.days) != len(self.series):
            raise ValueError("TrainFold.days and .series must be the same length (aligned per-day)")
        if self.warmup < 0:
            raise ValueError("TrainFold.warmup must be non-negative (a burn-in length in days)")


@dataclass(frozen=True)
class ScanCursor:
    """A read-cursor that makes the ``F_{t-1}``-measurable decision->execution offset EXPLICIT.

    A decision that FIRES a trade executed on bar ``t`` must be measurable from information up to
    ``t-1`` (SIGNAL §C.1, SYSTEM §C.2; the BOCPD one-bar-lookahead hole in docs/EDGE.md). This cursor
    holds the execution index ``t`` and the frozen full ``series`` and exposes ONLY the causal view:

      * ``past()``      -- ``series[:t]`` (indices ``0 .. t-1``): the ONLY window a DAILY decision that
        executes on ``t`` may read. A model that decides from ``past()`` is ``F_{t-1}``-measurable by
        construction.
      * ``read(i)``     -- guarded single read; routes through ``require_causal(i, self.t, cadence)``
        so a DAILY read of index ``>= t`` (same bar included) or a WITHIN_DAY read after the instant
        RAISES ``PurityError`` -- the hook a pinning test asserts against.

    It NEVER exposes ``series[t]`` or later to a DAILY decision, so a same-bar leak cannot be written
    accidentally; a WITHIN_DAY trigger passes its own ``instant`` index to ``read`` for the up-to-the-
    instant rule. ``cadence`` fixes which rule applies (SYSTEM §C.2).
    """

    series: tuple[float, ...]
    t: int
    cadence: Cadence = Cadence.DAILY

    def __post_init__(self) -> None:
        if not 0 <= self.t <= len(self.series):
            raise ValueError("ScanCursor.t must be a valid execution index into series (0..len)")

    def past(self) -> tuple[float, ...]:
        """The strictly-causal window ``series[:t]`` (indices ``0 .. t-1``) -- ``F_{t-1}``-measurable."""
        return self.series[: self.t]

    def read(self, i: int) -> float:
        """Read ``series[i]`` through the causal guard -- RAISES ``PurityError`` on a lookahead read.

        For a WITHIN_DAY trigger, ``i`` may equal the decision instant ``t`` (the up-to-the-instant
        rule); for a DAILY decision, any ``i >= t`` is a leak. Delegates the rule to ``require_causal``.
        """
        require_causal(i, self.t, self.cadence)
        return self.series[i]


# ==================================================================================================
# The candidate protocols (fit -> frozen; causal one-step-ahead decision). Interface ONLY.
# ==================================================================================================
@runtime_checkable
class EntryModel(Protocol):
    """A Phase-C ENTRY candidate (replaces/gates the ORB strong-close trigger; SIGNAL §C.2, EDGE #1-4).

    Conformance contract (a candidate is admissible only if it satisfies ALL):
      * ``k: int`` -- the free-parameter count (params fit on train + latent-state dimension), the
        parsimony complexity the SYSTEM §C.4 information-criterion penalty is charged against. The
        incumbent is the parsimony floor; a richer model must beat it by MORE than its ``k`` penalty.
      * ``fit(train: TrainFold) -> EntryModel`` -- estimate parameters on the TRAIN fold ONLY and
        FREEZE them, returning the fitted (self) model. No parameter is tuned on TEST (SYSTEM §C.1).
      * ``signal_at(cursor: ScanCursor) -> Side`` -- the causal one-step-ahead directional signal for
        the trade executed on ``cursor.t``: ``{LONG, FLAT, SHORT}`` in the SIGNAL §3 sign convention,
        computed from ``cursor.past()`` / ``cursor.read`` ONLY (so it is ``F_{t-1}``-measurable / up-
        to-the-instant for a WITHIN_DAY cadence). It reads the FILTERED / one-step-ahead posterior,
        NEVER a smoothed / full-sample estimate (SIGNAL §C.1 -- the load-bearing purity clause).

    This is an interface, not a base class: the harness accepts any object that structurally conforms
    (``isinstance(obj, EntryModel)`` works via ``runtime_checkable``), and it is tested with a trivial
    SYNTHETIC mock candidate -- the real Phase-C entry models (BOCPD / SPRT-GLR / HMM / Lee-Mykland)
    are NOT imported here (SYSTEM §C.3 off-ledger / interface-driven-harness rule).
    """

    k: int

    def fit(self, train: TrainFold) -> EntryModel:
        """Fit parameters on ``train`` ONLY, freeze them, and return the fitted model (self)."""
        ...

    def signal_at(self, cursor: ScanCursor) -> Side:
        """Causal one-step-ahead entry signal ``{LONG, FLAT, SHORT}`` for the trade at ``cursor.t``."""
        ...


@runtime_checkable
class ExitModel(Protocol):
    """A Phase-C EXIT / dynamic-HOLD candidate (replaces the fixed 5-day horizon; SIGNAL §C.3, EDGE #5-7).

    Conformance contract (a candidate is admissible only if it satisfies ALL):
      * ``k: int`` -- the free-parameter count (§C.4 parsimony), as for ``EntryModel``.
      * ``fit(train: TrainFold) -> ExitModel`` -- fit boundaries / hazard coefficients / latent
        dynamics on TRAIN ONLY, then FREEZE (SYSTEM §C.1). The free-boundary / hazard threshold is
        solved once on train parameters and frozen for OOS.
      * ``exit_now(cursor: ScanCursor, held_days: int, held_return: float) -> bool`` -- the causal
        hold/exit decision for a still-alive winner at ``cursor.t``: ``True`` = voluntarily close now,
        ``False`` = keep holding. Computed from ``held_days`` + ``held_return`` + ``cursor.past()`` /
        ``cursor.read`` ONLY (``F_{t-1}``-measurable) and the FILTERED state, never a smoothed one.
        The two decision covariates are FIRST-CLASS on the interface, not derived by re-basing the
        cursor (which would conflict with ``cursor.past()``'s documented meaning as the causal feature
        window, ScanCursor above):
          - ``held_days: int`` is the ELAPSED HOLDING TIME ``u`` -- the number of lake session days the
            position has been held, entry day ``u = 0`` (so the incumbent's fixed 5-day horizon is
            EXACTLY ``held_days >= 5``, and this interface therefore FAITHFULLY expresses the incumbent
            it "replaces"; SIGNAL §C.0, SYSTEM §C.1). It is ``F_{t-1}``-measurable by construction (a
            count of already-elapsed bars, known before the ``cursor.t`` execution bar, never a future
            read), and it is precisely the survival/hazard duration covariate ``u`` of the reversal
            hazard ``h(u | covariates)`` (EDGE #7 / SIGNAL §C.3(c) / SYSTEM §C.9 #7) -- passed
            explicitly, so ``u`` is NOT smuggled in via an undocumented per-trade re-basing of
            ``ScanCursor.series``. The position's ENTRY calendar index is thus NOT needed on the
            interface: ``held_days`` carries the duration directly, while ``cursor.t`` stays the
            frozen-full-series causal-feature calendar position (ScanCursor above), the two never
            conflated.
          - ``held_return: float`` is the position's current running return (a value-function / hazard
            covariate). A dynamic exit records its ACTUAL decided exit day, an explicit refinement over
            the incumbent's scheduled-horizon bucketing (SIGNAL §C.3).

    THE -12% STOP IS RETAINED AS THE FLOOR (SIGNAL §C.3, ``STOP_FLOOR``): an ``ExitModel`` decides only
    WHEN a still-alive winner is voluntarily closed; the true 1-minute premium stop remains the loss
    cap and an exit rule may NEVER loosen it (a conformance test asserts the model does not raise the
    floor). Interface only -- tested with a synthetic mock, real models (optimal-stopping / OU-Leung-Li
    / survival-hazard) NOT imported here.
    """

    k: int

    def fit(self, train: TrainFold) -> ExitModel:
        """Fit exit boundaries / hazard on ``train`` ONLY, freeze them, and return the fitted model."""
        ...

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        """Causal hold/exit at ``cursor.t`` -- ``True`` to close a still-alive winner now.

        ``held_days`` = elapsed holding time ``u`` (session days since entry, ``u=0`` on the entry
        day; incumbent = ``held_days >= 5``, hazard covariate ``h(u|.)`` of EDGE #7). ``held_return`` =
        current running return. Both ``F_{t-1}``-measurable; the FILTERED state only, never smoothed.
        """
        ...


# ==================================================================================================
# The per-candidate result container (the OOS trade stream + the parsimony free-parameter count k)
# ==================================================================================================
@dataclass(frozen=True)
class CandidateResult:
    """One scored candidate's OOS output -- what the SYSTEM harness collects per candidate (SYSTEM §C).

    The interface between a candidate and the SYSTEM selection gates (SYSTEM §C.3-§C.6):
      * ``name``   -- the ledger name; MUST be one of the 8 pre-registered candidates (SYSTEM §C.9 /
        SIGNAL §C.2-§C.4 / docs/EDGE.md). An off-ledger name is a snooped trial by definition and the
        harness rejects it -- this container does not police the name, but records it verbatim so the
        harness can check membership against the frozen ledger.
      * ``oos_trades`` -- the candidate's OUT-OF-SAMPLE trade stream on the single pre-registered
        held-out slice (``SPLIT = 2025-06-01``): a list of ``TradeRecord`` (CORE's atomic
        ``(exit_date, net_return)`` unit), on MATCHED trades / hold / instrument vs the incumbent. The
        DSR / bootstrap / drop-top-N gates all consume THIS stream (SYSTEM §C.3, §C.5, §C.6).
      * ``k``      -- the parsimony free-parameter count (params fit on train + latent-state
        dimension) the §C.4 information-criterion penalty is charged against. A candidate is adopted
        only if it beats the incumbent NET of this ``k``-penalty (SYSTEM §C.4, §C.6 clause 1). ``k >= 0``
        (a count); the incumbent's ``k`` is minimal (the parsimony floor).

    It is a pure DATA container: it holds no metric and makes no adopt decision (those are the SYSTEM
    gates' job, on the held-out slice ONLY, SYSTEM §C.6). It NEVER carries a fabricated/lake-derived
    number -- while the lake is absent the harness raises ``BlockedOnLakeError`` and never constructs a
    real ``CandidateResult`` (SYSTEM §C.8); a synthetic-recovery-test fixture is the only admissible
    non-lake instance and must be tagged synthetic.
    """

    name: str
    oos_trades: list[TradeRecord] = field(default_factory=list)
    k: int = 0

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be non-negative (a free-parameter / latent-state-dimension count)")

    @property
    def n_obs(self) -> int:
        """Number of OOS trades -- the ``n_obs`` fed to DSR / MinTRL (never a bar/day count; SYSTEM §4)."""
        return len(self.oos_trades)

    def returns(self) -> list[float]:
        """The bare ``net_return`` stream (per-observation), the un-annualized unit every gate consumes."""
        return [rec[1] for rec in self.oos_trades]


# ==================================================================================================
# Shared causal-estimation helpers (the SINGLE tested place the purity line is enforced)
# ==================================================================================================
def require_causal(read_index: int, decision_index: int, cadence: Cadence = Cadence.DAILY) -> None:
    """Assert a read is CAUSAL for the decision cadence -- RAISE ``PurityError`` on a lookahead (SYSTEM §C.2).

    The measurability standard is fixed by the cadence (SYSTEM §C.2 -- the two cases must not be
    conflated):
      * ``Cadence.DAILY``      -- a decision executing on bar ``decision_index`` must be strictly
        ``F_{t-1}``-measurable: any ``read_index >= decision_index`` (INCLUDING the same bar) is a
        leak (the same-bar ``RV_t`` / same-bar filtered ``E[.|F_t]`` hole). Legal iff
        ``read_index < decision_index``.
      * ``Cadence.WITHIN_DAY`` -- an intra-session trigger may read UP TO its decision instant: a read
        AT the instant (``read_index == decision_index``) is legal, only ``read_index > decision_index``
        (a bar dated after the instant) is a leak.

    This is the hook a no-lookahead PINNING test (SYSTEM §C.2) drives: wiring that lets a decision read
    bar ``>= t`` (daily) breaks here. A negative ``read_index`` is a caller bug (RAISES ``ValueError``,
    not ``PurityError`` -- it is not a lookahead, it is malformed).
    """
    if read_index < 0:
        raise ValueError("read_index must be non-negative (an index into the series)")
    if cadence is Cadence.DAILY:
        if read_index >= decision_index:
            raise PurityError(
                f"DAILY lookahead: decision executing on bar {decision_index} read bar {read_index} "
                f"(must be < {decision_index}; a same-bar read is a leak) -- SIGNAL §C.1 / SYSTEM §C.2")
    else:  # WITHIN_DAY: up to the decision instant is legal, strictly after is a leak.
        if read_index > decision_index:
            raise PurityError(
                f"WITHIN_DAY lookahead: trigger at instant {decision_index} read bar {read_index} "
                f"(must be <= {decision_index}) -- SIGNAL §C.1 / SYSTEM §C.2")


def warmup_ok(decision_index: int, warmup: int) -> bool:
    """Warm-up honesty: has the model's OWN burn-in been satisfied WITHIN the fold (SIGNAL §C.1)?

    A candidate that needs history (HMM filter burn-in, HAR's 22-day lag, a change-point run-length
    prior) may begin its scan only after ``warmup`` days *within its own train/OOS window* -- never
    borrowing pre-sample or cross-fold information (SIGNAL §C.1 warm-up honesty). A decision that
    executes on ``decision_index`` has enough causal history iff at least ``warmup`` bars precede it,
    i.e. ``decision_index >= warmup``. Returns ``True`` when the scan may fire, ``False`` while still
    warming up (the caller skips the bar). ``warmup < 0`` is a caller bug (RAISES).
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative (a burn-in length in bars)")
    return decision_index >= warmup
