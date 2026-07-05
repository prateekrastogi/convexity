"""MODELS.jumps -- Lee-Mykland (2008) jump test, RE-ADMITTED as a DIRECTIONAL jump-continuation ENTRY.

The Phase-C ENTRY candidate #4 (docs/EDGE.md "The pre-registered candidate set (frozen)", SIGNAL
§C.2(d), SYSTEM §C.9 #4). It replaces/gates the incumbent's ORB strong-close trigger: on a detected
*significant* jump it enters in the SIGN of that jump (jump-continuation) -- the scoped, protocol-gated
directional re-admission (owner un-ban 2026-07-04), NOT the refuted long-vol / straddle use.

THE MODEL (Lee & Mykland 2008, "Jumps in Financial Markets: A New Nonparametric Test and Jump
Dynamics"). For a log-return series, the standardized jump statistic at bar ``t`` is::

    L_t = r_t / sigma_hat_t

where ``r_t`` is the (log-)return realized OVER bar ``t`` and ``sigma_hat_t`` is a CAUSAL, backward-
looking spot-volatility estimate from the ``K`` bars STRICTLY BEFORE ``t`` -- the realized bipower
variation (Barndorff-Nielsen & Shephard) of the prior window::

    sigma_hat_t^2 = mu_1^{-2} * (1 / (K - 1)) * sum_{i} |r_{t-i}| |r_{t-i-1}|   (i over the window)

i.e. the MEAN of the ``K - 1`` adjacent cross-products of the ``K`` prior returns (divided by their
COUNT, ``K - 1``, exactly as ``bipower_sigma`` implements). This per-term mean converges to
``mu_1^2 * sigma^2``, so ``sigma_hat -> sigma``; the divisor is the number of cross-products (``K - 1``),
NOT ``K - 2`` -- the header formula and the code agree on the per-term-mean normalization.

with ``mu_1 = E|Z| = sqrt(2/pi)`` (the bipower scaling constant). Bipower variation is
robust to jumps IN the estimation window (a single |r| is annihilated by its jump-free neighbour), so
``sigma_hat_t`` measures the CONTINUOUS diffusive vol even when the window itself contains a jump.

DETECTION (the exact Lee-Mykland limiting law -- no free P&L threshold). Under the null of NO jump at
``t``, ``|L_t|`` has, over ``n`` observations, the Gumbel extreme-value limit::

    ( max_t |L_t| - C_n ) / S_n  -->  xi ,   P(xi <= x) = exp(-e^{-x})   (standard Gumbel)
    C_n = (2 log n)^{1/2} / mu_1 - (log pi + log log n) / (2 mu_1 (2 log n)^{1/2})
    S_n = 1 / ( mu_1 (2 log n)^{1/2} )

A jump is flagged at ``t`` iff ``|L_t| > G(alpha)`` where the Gumbel critical value for a
pre-registered significance ``alpha`` is ``G(alpha) = -log(-log(1 - alpha)) * S_n + C_n`` (Lee-Mykland
eq. for the rejection region). ``alpha`` and the local window ``K`` are the ONLY free parameters -- set
a priori on TRAIN and FROZEN (never P&L grid-searched), so ``k = 2``.

DIRECTIONAL SIGNAL. On a flagged jump the ENTRY is the SIGN of the jump return: ``r_t > 0 -> LONG``,
``r_t < 0 -> SHORT`` (jump-continuation); no jump -> ``FLAT``. Same ``{+1, 0, -1}`` convention as
``signal/orb.py`` (SIGNAL §3), feeding the *unchanged* §4 strike selection.

PURITY (binding -- SIGNAL §C.1, SYSTEM §C.2). The candidate is DAILY-cadence: the decision that fires
the trade executed on bar ``t`` is strictly ``F_{t-1}``-measurable. Concretely ``signal_at`` reads
ONLY ``cursor.past() == series[:t]`` -- it detects the jump in the return ENDING at ``t-1`` (the last
fully-observed bar before execution) using a bipower vol on bars strictly before that, and enters at
``t`` in the detected sign. NO ``series[t]`` (same-bar) or later read occurs; a same-bar read is the
one-bar-lookahead hole the pinning test FAILS. The local-vol estimator ``sigma_hat`` uses only bars
strictly before the tested return. Parameters are TRAIN-fit and FROZEN -- the fit reads only the train
fold, computes ``n_train`` (for the Gumbel ``C_n``/``S_n`` normalization it will use OOS -- pinned on
train, see ``fit``) and freezes ``(K, alpha)``; nothing is tuned on TEST.

Warm-up honesty (SIGNAL §C.1): the scan may fire only after the model's own burn-in -- the trailing
window needs ``K + 1`` prior returns (``K`` bipower cross-products + the tested return), i.e. the first
admissible execution index is ``warmup = K + 2`` price points (``K + 1`` returns). Before that
``signal_at`` returns ``FLAT`` (never borrowing pre-sample / cross-fold data).

DATA GATE (SYSTEM §C.8): this module is a candidate MODEL, synthetic-testable via known-generating-
process recovery ONLY (inject jumps at controlled times/signs; assert they are detected). The empirical
selection RUN is BLOCKED-ON-LAKE and lives in the harness, not here -- this module touches no lake,
emits no metric, adopts nothing.

Imports ONLY ``intraday.models.base`` (the shared interface + causal helpers) plus stdlib ``math`` --
``intraday.core`` shared contracts are reached transitively through ``base`` (which re-exports them),
NOT imported directly here, and NEVER a sibling Phase-C model module (the harness is interface-driven;
SYSTEM §C.3 off-ledger rule). Pure Python (no numpy needed for the per-bar recursion); the
``series`` a candidate consumes is the per-bar (log-)return stream on the ``TrainFold`` / ``ScanCursor``
contract. Moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import (
    Cadence,
    ScanCursor,
    Side,
    TrainFold,
    warmup_ok,
)

__all__ = [
    "MU1",
    "gumbel_critical_value",
    "bipower_sigma",
    "LeeMyklandJumpEntry",
]

# mu_1 = E|Z| for a standard normal Z = sqrt(2/pi) -- the bipower-variation scaling constant
# (Barndorff-Nielsen & Shephard; Lee-Mykland 2008). Bipower variation of the diffusive part converges
# to mu_1^2 * integrated variance, so sigma_hat^2 divides the raw cross-product sum by mu_1^2.
MU1 = math.sqrt(2.0 / math.pi)

# Below this a diffusive-vol estimate is numerically indistinguishable from zero; L_t = r/sigma is then
# undefined (a flat prior window has no spot vol), so the bar is treated as no-detection (FLAT).
_SIGMA_EPS = 1e-12


def gumbel_critical_value(alpha: float, n: int) -> float:
    """Lee-Mykland (2008) Gumbel critical value ``G(alpha)`` for the max-|L| jump test over ``n`` obs.

    Under the no-jump null, ``(max_t |L_t| - C_n) / S_n`` converges to a standard Gumbel; a jump is
    flagged where ``|L_t| > G(alpha)`` with::

        C_n = (2 log n)^{1/2} / mu_1 - (log pi + log log n) / (2 mu_1 (2 log n)^{1/2})
        S_n = 1 / ( mu_1 (2 log n)^{1/2} )
        G(alpha) = -log(-log(1 - alpha)) * S_n + C_n

    ``alpha`` is the pre-registered significance (TRAIN-frozen, NOT P&L-tuned); ``n`` is the sample size
    the Gumbel normalization uses (pinned on TRAIN, see ``LeeMyklandJumpEntry.fit`` -- the same ``n`` is
    reused OOS so the critical value is frozen, never re-derived from the TEST length). Requires
    ``0 < alpha < 1`` and ``n >= 3`` (``log log n`` needs ``n > 1`` and the extreme-value asymptotics a
    handful of points); a caller bug raises ``ValueError``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1) (a jump-test significance level)")
    if n < 3:
        raise ValueError("n must be >= 3 for the Gumbel extreme-value normalization (log log n)")
    log_n = math.log(n)
    two_log_n = 2.0 * log_n
    root = math.sqrt(two_log_n)
    c_n = root / MU1 - (math.log(math.pi) + math.log(log_n)) / (2.0 * MU1 * root)
    s_n = 1.0 / (MU1 * root)
    # Gumbel quantile at (1 - alpha): P(xi <= x) = exp(-e^{-x}) => x = -log(-log(1 - alpha)).
    xi = -math.log(-math.log(1.0 - alpha))
    return xi * s_n + c_n


def bipower_sigma(window: tuple[float, ...] | list[float]) -> float:
    """Causal spot-vol estimate: the jump-robust realized BIPOWER variation of a return ``window``.

    Estimates the CONTINUOUS diffusive volatility from the strictly-prior returns ``window`` (the ``K``
    returns before the tested bar). Bipower variation (Barndorff-Nielsen & Shephard 2004)::

        sigma_hat^2 = (1 / mu_1^2) * (1 / (K - 1)) * sum_{i=1..K-1} |r_i| * |r_{i-1}|

    Each cross-product ``|r_i||r_{i-1}|`` pairs adjacent returns, so a lone jump in the window is
    multiplied by a jump-free neighbour and its O(1) contribution is annihilated in the limit -- this
    is why the estimate is ROBUST TO JUMPS inside the window and measures the diffusive part only (the
    property Lee-Mykland rely on so a jump does not inflate its own local vol). Returns the volatility
    ``sigma_hat`` (not the variance). A window with ``< 2`` returns has no adjacent pair and returns 0.0
    (the caller treats a 0/degenerate sigma as no-detection).
    """
    w = tuple(window)
    if len(w) < 2:
        return 0.0
    bpv = 0.0
    for i in range(1, len(w)):
        bpv += abs(w[i]) * abs(w[i - 1])
    bpv /= (len(w) - 1)
    var = bpv / (MU1 * MU1)
    return math.sqrt(var) if var > 0.0 else 0.0


@dataclass
class LeeMyklandJumpEntry:
    """Lee-Mykland (2008) directional jump-continuation ENTRY candidate (EntryModel; EDGE #4).

    A causal, DAILY-cadence ``EntryModel`` (``intraday.models.base``): on a detected *significant* jump
    in the return ending at ``t-1`` it enters at bar ``t`` in the SIGN of that jump; else ``FLAT``. All
    reads are ``cursor.past()`` (indices ``0..t-1``) so the decision is strictly ``F_{t-1}``-measurable
    (SIGNAL §C.1, SYSTEM §C.2 DAILY purity -- a same-bar read is a leak the pinning test FAILS).

    Free parameters (fit on TRAIN, FROZEN for OOS -- ``k = 2``, the parsimony count SYSTEM §C.4 charges):
      * ``window`` (``K``) -- the trailing local-vol window length (# prior returns in the bipower
        estimate).
      * ``alpha``          -- the jump-test significance (the Gumbel critical value ``G(alpha)``).
    ``fit`` pins the Gumbel sample size ``n`` from the TRAIN return count and FREEZES it, so the critical
    value ``G(alpha)`` is computed once on train information and reused OOS unchanged (never re-derived
    from the TEST length -- that would be a TEST-side re-tune, SYSTEM §C.1). No parameter is P&L-tuned.

    Warm-up honesty: ``signal_at`` fires only once at least ``window + 1`` prior returns exist
    (``warmup = window + 2`` price points); before that it returns ``FLAT`` (SIGNAL §C.1). The candidate
    NEVER reads ``series[t]`` or later; a smoothed / full-sample vol is inadmissible (only the trailing
    causal bipower vol is used).
    """

    window: int = 20
    alpha: float = 0.01
    cadence: Cadence = Cadence.DAILY
    k: int = 2  # (window K, significance alpha) -- the two frozen free parameters (SYSTEM §C.4)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window (K) must be >= 2 (the bipower estimate needs an adjacent pair)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly in (0, 1) (a jump-test significance level)")
        # The scan needs `window` prior returns for the bipower vol PLUS the tested return itself; a
        # return spans two price points, so warm-up in PRICE indices is window + 2 (window+1 returns).
        self._warmup: int = self.window + 2
        self._n: int | None = None  # Gumbel sample size, pinned on fit() then frozen
        self._crit: float | None = None  # frozen Gumbel critical value G(alpha)

    def fit(self, train: TrainFold) -> "LeeMyklandJumpEntry":
        """Fit on TRAIN ONLY: pin the Gumbel sample size ``n`` and freeze ``G(alpha)`` (SYSTEM §C.1).

        The local window ``K`` and significance ``alpha`` are pre-registered constructor parameters
        (set a priori, not learned); what ``fit`` PINS from train is the Gumbel extreme-value sample
        size ``n`` used to normalize the critical value, so ``G(alpha)`` is FROZEN on train information
        and reused OOS -- it is never re-derived from the (larger/smaller) TEST length, which would be a
        TEST-side re-tune. ``n`` is the number of usable trailing-window jump tests in the train fold
        (the count of admissible scan bars past warm-up), floored at 3 (the Gumbel ``log log n``
        domain). Reads ONLY ``train`` -- no TEST data is touched. Returns the fitted (self) model.
        """
        # Usable train tests = train price points minus the warm-up burn-in (each admissible execution
        # bar past warm-up is one max-|L| test); floor at 3 for the log-log-n Gumbel domain.
        usable = max(3, len(train.series) - self._warmup)
        self._n = usable
        self._crit = gumbel_critical_value(self.alpha, usable)
        return self

    def signal_at(self, cursor: ScanCursor) -> Side:
        """Causal one-step-ahead entry for the trade executed on ``cursor.t`` -- ``F_{t-1}``-measurable.

        Reads ONLY ``cursor.past()`` (indices ``0..t-1``): the last fully-observed return is
        ``r = series[t-1] - series[t-2]`` (the return ENDING at ``t-1``); its Lee-Mykland statistic
        ``L = r / sigma_hat`` uses the jump-robust bipower vol of the ``window`` returns STRICTLY BEFORE
        that return. On ``|L| > G(alpha)`` (the frozen Gumbel critical value) it enters in the SIGN of
        the jump (``r > 0 -> LONG``, ``r < 0 -> SHORT``); else ``FLAT``. No ``series[t]`` (same-bar) or
        later read occurs -- the jump that fires an entry at ``t`` is measurable from information up to
        ``t-1`` (SIGNAL §C.1, SYSTEM §C.2). Returns ``FLAT`` while still warming up, or when the prior
        window has no diffusive vol (a degenerate ``sigma_hat``).
        """
        if self._crit is None:
            raise RuntimeError("signal_at before fit -- parameters (Gumbel critical value) not frozen")
        t = cursor.t
        # Warm-up honesty (SIGNAL §C.1): need window+1 prior returns => window+2 prior price points.
        if not warmup_ok(t, self._warmup):
            return Side.FLAT
        past = cursor.past()  # series[:t] -- indices 0..t-1, the ONLY causal window (F_{t-1}-measurable)
        # The tested return ends at the last fully-observed bar t-1: r_test = past[-1] - past[-2].
        # The `window` returns STRICTLY BEFORE it span past[-(window+2) : -1] (window+1 prices ->
        # `window` returns), so the bipower spot vol is jump-robust AND uses no bar >= the tested return.
        prices = past[-(self.window + 2):]
        rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        r_test = rets[-1]
        prior_rets = tuple(rets[:-1])  # strictly before the tested return -> causal local vol
        sigma = bipower_sigma(prior_rets)
        if sigma <= _SIGMA_EPS:
            return Side.FLAT  # no diffusive vol in the prior window -> statistic undefined, no signal
        stat = r_test / sigma
        if abs(stat) <= self._crit:
            return Side.FLAT  # not a significant jump at the frozen Gumbel level
        return Side.LONG if r_test > 0.0 else Side.SHORT  # directional jump-continuation (EDGE #4)
