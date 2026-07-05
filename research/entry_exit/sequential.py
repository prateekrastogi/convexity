"""MODELS.sequential -- Phase-C ENTRY candidate #2: Wald SPRT / GLR drift-sign change detector.

The single **sequential-detection** representative of the frozen Phase-C ledger (docs/EDGE.md "The
pre-registered candidate set (frozen)" item 2, SIGNAL §C.2(b), SYSTEM §C.9 #2 -- named IDENTICALLY in
all three specs; CUSUM is the DEFERRED sequential sibling, §C.5, deferred for parsimony NOT lookahead).
It replaces/gates the incumbent ORB strong-close ENTRY trigger with a formal sequential test of a
drift-SIGN change, and emits the SIGNAL §3 ``{+1, 0, -1}`` signal into the *unchanged* §4 strike
selection. It is an ``EntryModel`` (models/base.py) -- ``fit(train) -> self`` (train-fit then FROZEN)
plus a causal one-step-ahead ``signal_at(cursor)``.

FORMAL MODEL (stated assumptions / estimator / error-control -- NOT a technical indicator).
  *Generating process (the null and the two alternatives).* The per-bar (daily-close-to-close within
  the session) log-return is ``x_t = mu + eps_t`` with i.i.d. Gaussian innovations ``eps_t ~ N(0,
  sigma^2)``. We run a sequential test of ``H_0: mu = 0`` (no directional drift) against the two
  composite one-sided alternatives ``H_+: mu = +delta`` and ``H_-: mu = -delta`` (a drift shift of
  unknown sign, magnitude ``delta``). This is Wald's SPRT with the Gaussian per-sample log-likelihood
  ratio increment for the ``+delta`` arm

      llr_+(x) = log[ f_{+delta}(x) / f_0(x) ]
               = (delta / sigma^2) * (x - delta/2)          (Gaussian mean-shift LLR),

  and ``llr_-(x) = (delta / sigma^2) * (-x - delta/2)`` for the mirror arm (antisymmetric in ``x``).
  The **GLR** generalization replaces the fixed ``+/- delta`` by the running post-change MLE of the
  drift over a trailing window of length ``m`` (Lorden / Siegmund window-limited GLR); it does not
  pre-specify the shift magnitude. Both are supported (``mode="sprt"`` / ``mode="glr"``), one detector.

  *Estimator (accumulate the LLR; first-passage of a boundary = entry).* For SPRT each arm carries an
  accumulator ``Lambda^{+/-}_t = Lambda^{+/-}_{t-1} + llr_{+/-}(x_t)``, reset to 0 on any downcross of
  the lower boundary ``B`` (the continue-region reset of the repeated-SPRT / Page-style change
  detector). The FIRST arm whose statistic crosses the UPPER boundary ``A`` triggers entry in that
  arm's drift SIGN (``+1`` for the ``+delta`` arm, ``-1`` for the ``-delta`` arm); until then the
  signal is ``FLAT``.
  For GLR the decision at the CURRENT bar is the Lorden/Siegmund *window-limited* generalized
  likelihood ratio evaluated at the decision point ``n`` (here ``n = t-1``, the last ``F_{t-1}``-
  measurable bar): the drift is profiled out over each candidate change-point ``k`` in the trailing
  window of length ``m`` and the window statistic is MAXIMIZED over those ``k``,

      g_n = max_{n-m+1 <= k <= n}  (sum_{i=k}^{n} x_i)^2 / (2 * (n-k+1) * sigma^2) ,

  i.e. ``max_k [ len_k/2 * (xbar_k / sigma)^2 ]`` over window starts ``k`` within the last ``m`` bars
  ending at ``n``. It alarms when ``g_n >= h``; the entry sign is ``sign(sum over the maximizing
  window)``. Crucially the window END is FIXED at the current decision bar ``n = t-1`` and the max is
  over the ``<= m`` window STARTS -- this is a genuine window-limited detector evaluating the CURRENT
  trailing window at each ``t`` (it does NOT scan every historical window end and latch the first-ever
  crossing forever; that older form ignored the current-window drift and had no ARL control -- see the
  error-control note and the D1/D2 fix below).

  *Error control (boundaries pre-set from a train Type-I / Type-II pair -- NOT P&L-tuned).* The Wald
  approximations set the SPRT boundaries from a pre-registered false-alarm / miss pair ``(alpha,
  beta)``:

      A ~= log((1 - beta) / alpha)   (upper -- accept the drift hypothesis / alarm),
      B ~= log(beta / (1 - alpha))   (lower -- reset to the no-signal continue region),

  equivalently a target in-control Average Run Length (ARL). ``(alpha, beta)`` is fixed A PRIORI from
  the error-probability design (the no-snoop way to fix the boundaries), never grid-searched for P&L
  (SIGNAL §C.2(b), SYSTEM §C.9 #2). For the window-limited GLR the analogous ARL-controlling boundary
  is the SAME log-boundary ``h = log((1 - beta) / alpha)`` (Lorden 1971; Siegmund & Venkatraman 1995;
  Lai 1998: the window-limited-GLR in-control ARL grows like ``e^h``, so ``h ~= log(ARL)`` is the
  ARL-calibrated threshold -- exactly the SPRT upper boundary ``A``). This REPLACES the earlier fixed
  single-window chi-square-1/2 quantile ``Phi^-1(1-alpha/2)^2/2``: that critical value controls ONE
  test's Type-I error, not the SEQUENTIAL false-alarm rate over the overlapping windows a detector
  scans, so under it the in-control false-alarm probability grew without bound with history length
  (measured ~0.06 at 20 bars -> ~0.94 at 1000 bars). Because the current form fixes the window end at
  ``t-1`` and controls the alarm with the log-ARL boundary, the in-control false-alarm probability is
  STABLE in history length (~0.02 across 20..1000 bars at ``alpha=0.01, beta=0.20``) and monotone in
  ``alpha`` -- the ARL property the ``(alpha, beta)`` design buys, identical to the SPRT arm.

TRAIN-FIT, THEN FROZEN (SIGNAL §C.1, SYSTEM §C.2 -- no re-fit / re-tune on TEST). ``fit(train)``
estimates ONLY the innovation scale ``sigma`` and (for SPRT) the minimum-detectable drift ``delta``
from the train fold; ``(alpha, beta)`` -> ``(A, B)`` and the GLR boundary ``h`` are the pre-registered
a-priori design constants, NOT fit to the data. The GLR window length ``m`` is a searched train-fit /
frozen knob (SIGNAL §C.2(b): "for GLR, the window length ``m``"), counted in ``k`` (§C.4 parsimony;
see ``k`` below). After ``fit`` every operative parameter is frozen for OOS.

PURITY IS BINDING (SIGNAL §C.1, SYSTEM §C.2 -- the load-bearing clause). The detector is a DAILY-cadence
``EntryModel``: the decision that FIRES the trade executed on bar ``t`` accumulates the LLR / evaluates
the window-limited GLR over ``cursor.past()`` = ``series[:t]`` ONLY -- strictly ``F_{t-1}``-measurable,
it NEVER reads bar ``t`` (a same-bar read is a one-bar look-ahead). Every read routes through
``ScanCursor``'s causal guard, so a mutation that lets the decision at ``t`` read bar ``>= t`` BREAKS
the pinning test (models/base.py ``require_causal`` raises ``PurityError``). The test is purely
recursive/online, so it is causal by construction (each update reads only ``x_t`` and the prior
accumulator / the trailing window); no smoothed / full-sample / retrospective estimate enters.

Imports ONLY ``intraday.models.base`` (the interface) + stdlib ``math`` -- NEVER a sibling Phase-C
model module (the harness is interface-driven, SYSTEM §C.3 off-ledger rule) and NEVER the lake (this
module touches no data; the empirical RUN is BLOCKED-ON-LAKE, owned by the harness). Pure Python +
stdlib; moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import ScanCursor, Side, TrainFold

__all__ = [
    "SprtGlrEntry",
    "wald_boundaries",
    "glr_threshold",
]

# Below this the train innovation scale is numerically indistinguishable from a constant series
# (a degenerate, drift-free fold); such a fold has no defined LLR scale, so `fit` floors sigma here.
_SIGMA_EPS = 1e-12


def wald_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Wald SPRT decision boundaries ``(A, B)`` from a Type-I / Type-II pair (SIGNAL §C.2(b)).

    ``A = log((1 - beta) / alpha)`` (upper -- alarm / accept the drift hypothesis) and
    ``B = log(beta / (1 - alpha))`` (lower -- reset to the no-signal continue region), the standard
    Wald approximations to a target ``(alpha, beta)`` false-alarm / miss design (equivalently a target
    in-control ARL). This is the ONLY source of the boundaries -- they are set A PRIORI from the
    error-probability design, NEVER grid-searched for P&L (the no-snoop rule). ``A > 0 > B`` for a
    sane ``(alpha, beta)`` (both small): a positive upper boundary to cross, a negative lower reset.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must lie strictly in (0, 1) (Type-I / Type-II error rates)")
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    return upper, lower


def glr_threshold(alpha: float, beta: float) -> float:
    """Window-limited-GLR ARL-controlling alarm boundary ``h`` (Lorden/Siegmund; SIGNAL §C.2(b)).

    The window-limited GLR is a SEQUENTIAL change detector: at each bar it maximizes the profiled-drift
    likelihood ratio over the ``<= m`` change-points in the trailing window and alarms on the first
    crossing of ``h``. Its error control is therefore an in-control **Average Run Length** (ARL) design,
    NOT a single test's Type-I error. A fixed single-window chi-square-1/2 critical value
    ``Phi^-1(1-alpha/2)^2/2`` controls ONE window's Type-I error and, used as a sequential threshold,
    lets the false-alarm probability grow without bound with history length (the D1 defect). The
    ARL-calibrated boundary is instead the SAME log-boundary the SPRT upper arm uses,

        h = log((1 - beta) / alpha)   (== the SPRT upper boundary A of ``wald_boundaries``),

    because the window-limited-GLR in-control ARL grows like ``e^h`` (Lorden 1971; Siegmund &
    Venkatraman 1995; Lai 1998), so ``h ~= log(ARL)`` is the a-priori ARL design knob -- identical
    ``(alpha, beta)`` error-probability design as the SPRT arm, keeping the two arms coherent under one
    frozen ``(alpha, beta)``. Set a priori from ``(alpha, beta)`` alone, never P&L-tuned. A stricter
    ``alpha`` (or ``beta``) raises ``h`` and lengthens the in-control ARL.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must lie strictly in (0, 1) (the GLR false-alarm / miss pair)")
    return math.log((1.0 - beta) / alpha)


@dataclass
class SprtGlrEntry:
    """Wald SPRT / GLR sequential drift-sign change detector -- Phase-C ENTRY candidate #2 (EntryModel).

    A formal sequential test of a drift-SIGN change (``H_0: mu=0`` vs ``H_+: mu=+delta`` / ``H_-:
    mu=-delta``); first-passage of the log-likelihood-ratio boundary ``A`` = entry in that arm's sign
    (SIGNAL §C.2(b), EDGE #2, SYSTEM §C.9 #2). Conforms to the ``EntryModel`` protocol (models/base.py):
    ``fit(train) -> self`` (train-fit then FROZEN) + causal ``signal_at(cursor) -> Side``.

    Pre-registered A-PRIORI design constants (NOT fit to data -- set from the error-probability design):
      * ``alpha``, ``beta`` -- the target Type-I (false-alarm) / Type-II (miss) pair. ``(A, B)`` follow
        by ``wald_boundaries`` (SPRT) and the GLR ARL boundary ``h`` by ``glr_threshold(alpha, beta)``
        (== the SPRT upper ``A``). Fixed a priori, NEVER grid-searched for P&L.
      * ``mode`` -- ``"sprt"`` (fixed ``+/- delta``, ``delta`` train-fit) or ``"glr"`` (running MLE over
        a trailing window ``m``, no pre-specified shift magnitude).
      * ``window`` -- the GLR trailing-window length ``m`` (ignored by SPRT). Per SIGNAL §C.2(b) it is a
        train-fit / frozen SEARCHED knob for the GLR arm (not an a-priori design constant), so it is
        counted in the GLR ``k`` (see ``k`` below).
      * ``delta_sigmas`` -- the SPRT minimum-detectable drift expressed in TRAIN innovation-sigma units
        (``delta = delta_sigmas * sigma_hat``); ``delta`` is thus data-scaled but its size (in sigmas)
        is a pre-registered design choice, not a P&L search node.

    Train-fit, then FROZEN (``fit``): the innovation scale ``sigma_hat`` (and, for SPRT,
    ``delta = delta_sigmas * sigma_hat``). Nothing is re-fit or re-tuned on TEST (SIGNAL §C.1).

    ``k`` = free-parameter count (§C.4 parsimony). SPRT fits ``sigma`` (1) + ``delta`` (1) => 2. GLR
    fits ``sigma`` (1) and carries the SEARCHED / train-frozen window length ``m`` (1) => 2 as well:
    the running-MLE profiles the DRIFT out (so GLR fits no ``delta``), but ``m`` is a genuine train-fit
    /frozen knob (SIGNAL §C.2(b), "for GLR, the window length ``m``") and enters the parsimony count,
    exactly matching the SPRT arm's 2. Counting GLR as ``k=1`` would understate the §C.4 penalty in the
    candidate's favor (the D3 reconciliation); ``(alpha, beta)`` are a-priori design constants and are
    NOT counted.

    PURITY (binding): DAILY cadence -- ``signal_at`` accumulates the LLR / evaluates the window-limited
    GLR over ``cursor.past()`` = ``series[:t]`` ONLY, so the decision firing the trade at bar ``t`` is
    ``F_{t-1}``-measurable and NEVER reads bar ``t`` (a same-bar read is a one-bar look-ahead the
    pinning test must catch). The recursion is causal by construction.
    """

    alpha: float = 0.05
    beta: float = 0.20
    mode: str = "sprt"
    window: int = 20
    delta_sigmas: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("sprt", "glr"):
            raise ValueError("mode must be 'sprt' or 'glr'")
        if self.mode == "glr" and self.window < 1:
            raise ValueError("GLR window m must be a positive integer bar count")
        if self.delta_sigmas <= 0:
            raise ValueError("delta_sigmas must be positive (the minimum-detectable drift in sigmas)")
        # A-priori boundaries from the pre-registered (alpha, beta) design -- NOT P&L-tuned. The GLR
        # arm uses the SAME log-ARL upper boundary as the SPRT arm (glr_threshold == wald A), so both
        # arms share one coherent (alpha, beta) error-probability / ARL design.
        self._upper, self._lower = wald_boundaries(self.alpha, self.beta)
        self._h = glr_threshold(self.alpha, self.beta)
        # k: free params counted for §C.4 parsimony. SPRT fits sigma + delta (2). GLR fits sigma (1)
        # and carries the searched/train-frozen window length m (1) => 2 (the running-MLE profiles the
        # DRIFT out, so no delta, but m is a genuine fitted knob per SIGNAL §C.2(b)). (alpha, beta are
        # a-priori design constants -> not counted.)
        self.k = 2
        self._sigma: float | None = None  # set on fit, then frozen
        self._delta: float | None = None  # set on fit (SPRT), then frozen

    def fit(self, train: TrainFold) -> "SprtGlrEntry":
        """Fit ONLY the innovation scale ``sigma`` (and, for SPRT, ``delta``) on TRAIN, then FREEZE.

        ``sigma_hat`` is the population std (ddof=0, matching the repo convention) of the train series
        past its warm-up; the SPRT drift ``delta = delta_sigmas * sigma_hat`` is scaled to it. The
        boundaries ``(A, B)`` / GLR ``h`` are already set from the a-priori ``(alpha, beta)`` in
        ``__post_init__`` and are NOT touched here (they are design constants, never data-fit). Returns
        the fitted model (self) with every operative parameter frozen for OOS.
        """
        body = train.series[train.warmup:]
        n = len(body)
        if n == 0:
            raise ValueError("TrainFold has no observations past warm-up to fit sigma")
        mu = sum(body) / n
        var = sum((x - mu) ** 2 for x in body) / n  # ddof=0 population variance (repo DDOF convention)
        self._sigma = math.sqrt(var) if var > _SIGMA_EPS else _SIGMA_EPS
        self._delta = self.delta_sigmas * self._sigma
        return self

    def signal_at(self, cursor: ScanCursor) -> Side:
        """Causal one-step-ahead entry ``{LONG, FLAT, SHORT}`` for the trade at ``cursor.t`` (F_{t-1}).

        Runs the sequential detector over ``cursor.past()`` = ``series[:t]`` ONLY (never bar ``t``):
        SPRT accumulates the two-arm Gaussian mean-shift LLR with a lower-boundary reset and returns the
        sign of the FIRST arm to cross ``A``; the window-limited GLR evaluates the profiled-drift
        statistic on the CURRENT trailing window ending at ``t-1`` and returns its sign on crossing
        ``h``. Else ``FLAT``. Raises if called before ``fit`` (parameters must be frozen first).
        """
        if self._sigma is None or self._delta is None:
            raise RuntimeError("signal_at before fit -- parameters not frozen yet")
        past = cursor.past()  # F_{t-1}-measurable window (indices 0..t-1); never reads bar t
        if self.mode == "sprt":
            return self._sprt_signal(past)
        return self._glr_signal(past)

    def _sprt_signal(self, past: tuple[float, ...]) -> Side:
        """Two-arm Wald SPRT with lower-boundary reset; first arm to cross ``A`` = entry sign."""
        sigma, delta = self._sigma, self._delta
        assert sigma is not None and delta is not None  # frozen by fit (guarded in signal_at)
        coef = delta / (sigma * sigma)      # Gaussian mean-shift LLR scale
        half = delta / 2.0                  # the -mu/2 centering term
        lam_up = 0.0                        # +delta arm accumulator
        lam_dn = 0.0                        # -delta arm accumulator
        for x in past:
            lam_up += coef * (x - half)     # llr_+(x) = (delta/sigma^2)(x - delta/2)
            lam_dn += coef * (-x - half)    # llr_-(x) = (delta/sigma^2)(-x - delta/2)
            if lam_up >= self._upper:
                return Side.LONG            # +delta arm alarmed first -> LONG
            if lam_dn >= self._upper:
                return Side.SHORT           # -delta arm alarmed first -> SHORT
            if lam_up <= self._lower:       # downcross the reset boundary -> back to continue region
                lam_up = 0.0
            if lam_dn <= self._lower:
                lam_dn = 0.0
        return Side.FLAT

    def _glr_signal(self, past: tuple[float, ...]) -> Side:
        """Lorden/Siegmund window-limited GLR on the CURRENT trailing window ending at ``t-1``.

        The decision at bar ``t`` fixes the window END at the last ``F_{t-1}``-measurable bar
        (``n = t-1``, the final element of ``past``) and MAXIMIZES the profiled-drift GLR statistic over
        the ``<= m`` candidate change-points (window STARTS) inside the trailing window::

            g = max_{n-m+1 <= k <= n}  (sum_{i=k}^{n} x_i)^2 / (2 * (n-k+1) * sigma^2) .

        It alarms (``g >= h``) in the SIGN of the sum over the maximizing window. This evaluates the
        CURRENT window's drift at every ``t`` (a genuine window-limited detector) -- it does NOT scan
        every historical window end and latch the first-ever crossing forever (the D1/D2 defect, which
        both broke ARL control and could emit a sign contradicting the current-window drift). ``h`` is
        the log-ARL boundary ``glr_threshold(alpha, beta)`` (== the SPRT upper ``A``).
        """
        sigma = self._sigma
        assert sigma is not None
        n = len(past)
        if n == 0:
            return Side.FLAT                # empty causal past -> no evidence -> FLAT
        var = sigma * sigma
        end = n                             # window ends at t-1 (past = series[:t], last elem = t-1)
        lo = max(0, end - self.window)      # candidate window STARTS within the last m bars
        best_stat = -1.0
        best_sum = 0.0
        csum = 0.0
        # Accumulate suffix sums from the end backwards; each start k gives a window [k, end).
        for start in range(end - 1, lo - 1, -1):
            csum += past[start]
            length = end - start
            stat = (csum * csum) / (2.0 * length * var)  # profiled-drift GLR over window [k, end)
            if stat > best_stat:
                best_stat = stat
                best_sum = csum
        if best_stat >= self._h:
            if best_sum > 0:
                return Side.LONG
            if best_sum < 0:
                return Side.SHORT
        return Side.FLAT
