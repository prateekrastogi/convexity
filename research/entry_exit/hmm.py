"""Phase-C ENTRY candidate #3 -- 2-state Gaussian HMM on the FILTERED trend-posterior.

Ledger #3 (docs/EDGE.md "The pre-registered candidate set (frozen)", SIGNAL §C.2(c), SYSTEM §C.9 #3):
a Hamilton (1989) 2-state Markov-switching mean model -- returns are drawn from a bull-drift state
(`mu_hi > 0`) and a bear-drift state (`mu_lo < 0`), each Gaussian with its own variance, governed by
a 2x2 transition matrix `P`. The parameters `(mu_hi, mu_lo, sigma_hi, sigma_lo, P)` and the entry
probability threshold are estimated on the TRAIN fold (Baum-Welch / EM) and then FROZEN; nothing is
re-fit or re-tuned on TEST (SIGNAL §C.1, SYSTEM §C.1).

PURITY -- the load-bearing clause for the HMM (SIGNAL §C.2(c), SYSTEM §C.2, binding):
  * The decision at execution bar `t` reads ONLY the causal FILTERED posterior
    `xi_{s|s} = P(state_s | F_s)` produced by the forward Hamilton recursion over `cursor.past()`
    (indices `0 .. t-1`), so the entry is `F_{t-1}`-measurable by construction -- it never touches
    bar `t` or later (the DAILY cadence same-bar hole).
  * The Kim (1994) SMOOTHER `P(state_s | F_T)` -- which conditions on the WHOLE sample, including
    bars after `s` -- is LOOKAHEAD and is BARRED as a signal. It is implemented here ONLY so a
    pinning test can assert `smoothed != filtered` and that `signal_at` uses the filtered path; it
    carries an explicit `for_diagnostics_only` guard and never feeds a decision.
  * ENTRY: run the filter over the past window, take the filtered probability of the HIGH-DRIFT
    (bull) state at the LAST past bar; emit `LONG` when it crosses the frozen upper threshold,
    `SHORT` when the bull-probability falls below `1 - threshold` (i.e. the bear-state filtered
    probability crosses the same level -- sign-antisymmetric, matching SIGNAL §3), else `FLAT`.

Warm-up honesty (SIGNAL §C.1): the filter needs a burn-in for the posterior to settle away from its
prior; the model exposes `warmup` and refuses to fire before `warmup_ok` (the caller/harness skips
the bar). Pure Python + stdlib `math` (no numpy needed here; the recursion is 2-state). Imports ONLY
`intraday.core` / `intraday.models.base` -- never a sibling Phase-C model module (SYSTEM §C.3
interface-driven-harness / off-ledger rule). Moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import (
    PurityError,
    ScanCursor,
    Side,
    TrainFold,
    warmup_ok,
)

__all__ = ["HMMParams", "GaussianHMMEntry"]

# --- Numerical / EM control constants (fixed a priori, NOT P&L-grid-searched -- SYSTEM §C.2) -------
_VAR_FLOOR = 1e-12  # variance floor so a degenerate near-zero-spread state cannot divide-by-zero
_EPS = 1e-300  # likelihood underflow floor before a log / normalization
_MAX_EM_ITERS = 200  # Baum-Welch iteration cap
_EM_TOL = 1e-8  # log-likelihood convergence tolerance
# Filtered high-drift-prob entry level when train gives no better default. It MUST be strictly
# > 0.5 so the two entry branches (`p_high >= threshold` -> LONG, `p_high <= 1 - threshold` ->
# SHORT) leave a NEUTRAL band `(1 - threshold, threshold)` in which the model is FLAT -- SIGNAL
# §C.2(c) frames entry as "enter +1 when the filtered bull-probability crosses a level, -1 for
# bear", i.e. a level with a no-trade band around 0.5. At exactly 0.5 the two branches partition
# all of [0, 1] and FLAT is unreachable outside warm-up (the default candidate would be always-in-
# market); 0.6 is a modest a-priori band (a caller normally passes a train-frozen level before OOS).
_DEFAULT_THRESHOLD = 0.6
_DEFAULT_WARMUP = 5  # filter burn-in in bars before the posterior is trusted (SIGNAL §C.1)


def _norm_pdf(x: float, mu: float, var: float) -> float:
    """Gaussian density N(x; mu, var) with a variance floor (var already floored by the caller)."""
    return math.exp(-0.5 * (x - mu) * (x - mu) / var) / math.sqrt(2.0 * math.pi * var)


@dataclass(frozen=True)
class HMMParams:
    """The FROZEN 2-state Gaussian HMM parameters (fit on train, then immutable -- SIGNAL §C.1).

    State 0 is the HIGH-drift (bull) state and state 1 is the LOW-drift (bear) state by CONSTRUCTION:
    `fit` canonicalizes the EM output so `mu[0] >= mu[1]` always. This removes the label-switching
    ambiguity of EM (the two states are otherwise exchangeable) so "the filtered high-drift-state
    probability" (SIGNAL §C.2(c)) names an unambiguous quantity -- `xi[0]`. `trans[i][j]` is
    `P(state_t = j | state_{t-1} = i)` (rows sum to 1); `pi0` is the initial state distribution.
    """

    mu: tuple[float, float]
    var: tuple[float, float]
    trans: tuple[tuple[float, float], tuple[float, float]]
    pi0: tuple[float, float]

    def __post_init__(self) -> None:
        if self.mu[0] < self.mu[1]:
            raise ValueError("HMMParams: state 0 must be the high-drift state (mu[0] >= mu[1])")
        if self.var[0] <= 0 or self.var[1] <= 0:
            raise ValueError("HMMParams: variances must be strictly positive")
        for row in self.trans:
            if abs(sum(row) - 1.0) > 1e-6 or any(p < 0 for p in row):
                raise ValueError("HMMParams: transition rows must be non-negative and sum to 1")
        if abs(sum(self.pi0) - 1.0) > 1e-6 or any(p < 0 for p in self.pi0):
            raise ValueError("HMMParams: pi0 must be a non-negative distribution summing to 1")


class GaussianHMMEntry:
    """2-state Gaussian HMM EntryModel -- entry on the FILTERED high-drift posterior (ledger #3).

    Conforms to `intraday.models.base.EntryModel` (fit + signal_at + k). `k = 6`: five distinct
    fitted parameters `(mu_hi, mu_lo, sigma_hi, sigma_lo, and the transition self-persistences
    p00,p11 -> counted as the 2-DOF transition matrix)` plus the entry probability threshold, with
    the 2 latent states carried in the parsimony count per the base-interface convention
    (`k = free params fit on train + latent-state dimension`). Concretely: mu(2) + var(2) + trans(2
    free) = 6 fitted reals; the threshold is `threshold` (set from train, not grid-searched) -- the
    latent-state dimension (2) is folded into these six as the standard 2-state Gaussian HMM DOF.

    The `threshold` is fixed a priori (default 0.6, leaving a `(0.4, 0.6)` neutral no-trade band)
    rather than P&L-grid-searched; a caller may pass a train-derived level, but it must be set
    BEFORE OOS and FROZEN (no TEST re-tune, SYSTEM §C.1). It is required strictly `> 0.5` so a
    genuine FLAT band exists (SIGNAL §C.2(c)'s "crosses a level" with a no-trade band); a bare 0.5
    would partition [0, 1] into LONG/SHORT with no gap and the default candidate would be always in
    the market -- see `_DEFAULT_THRESHOLD`.
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD, warmup: int = _DEFAULT_WARMUP) -> None:
        # threshold must be strictly in (0.5, 1): > 0.5 so the LONG (`p_high >= threshold`) and
        # SHORT (`p_high <= 1 - threshold`) branches leave a NEUTRAL band `(1 - threshold, threshold)`
        # where the model is FLAT (SIGNAL §C.2(c)). At exactly 0.5 the band collapses and the model is
        # always in the market; <= 0.5 is worse (branches overlap). < 1 keeps both sides reachable.
        if not 0.5 < threshold < 1.0:
            raise ValueError(
                "threshold must be strictly in (0.5, 1): > 0.5 leaves a FLAT no-trade band "
                "(1 - threshold, threshold); 0.5 collapses it (always-in-market) -- SIGNAL §C.2(c)")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        self.k = 6
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.params: HMMParams | None = None  # set on fit, then FROZEN

    # --- EM (Baum-Welch) fit on the TRAIN fold ONLY -----------------------------------------------
    def fit(self, train: TrainFold) -> "GaussianHMMEntry":
        """Baum-Welch EM on `train.series[train.warmup:]`, then CANONICALIZE + FREEZE the params.

        Estimates `(mu, var, trans, pi0)` by the forward-backward EM for a 2-state Gaussian HMM on
        the train drift series ONLY (SIGNAL §C.1 train-fit). Note: the forward-backward smoother is
        used HERE, at FIT time, on the train fold -- that is legitimate (fitting is not a live
        decision); the PURITY line is about the live SIGNAL, which uses the filter only (`signal_at`).
        """
        x = list(train.series[train.warmup:])
        n = len(x)
        if n < 2:
            raise ValueError("HMM fit needs >= 2 train observations past warm-up")

        # Data-driven init that already respects the high/low split (mean +/- 0.5 std per state),
        # so EM starts near a separated solution rather than a symmetric saddle.
        mean = sum(x) / n
        var0 = max(sum((v - mean) ** 2 for v in x) / n, _VAR_FLOOR)
        sd = math.sqrt(var0)
        mu = [mean + 0.5 * sd, mean - 0.5 * sd]
        var = [var0, var0]
        trans = [[0.9, 0.1], [0.1, 0.9]]
        pi0 = [0.5, 0.5]

        prev_ll = -math.inf
        for _ in range(_MAX_EM_ITERS):
            # E-step: forward-backward (used ONLY for fitting; never for a live signal).
            alpha, scale = self._forward(x, mu, var, trans, pi0)
            beta = self._backward(x, mu, var, trans, scale)
            ll = sum(math.log(s) for s in scale)  # scaled log-likelihood

            gamma = [[alpha[t][i] * beta[t][i] for i in range(2)] for t in range(n)]  # already normed
            # xi[t][i][j] = P(state_t=i, state_{t+1}=j | X), summed over t for the transition update.
            xi_sum = [[0.0, 0.0], [0.0, 0.0]]
            for t in range(n - 1):
                b = [_norm_pdf(x[t + 1], mu[j], var[j]) for j in range(2)]
                denom = 0.0
                cell = [[0.0, 0.0], [0.0, 0.0]]
                for i in range(2):
                    for j in range(2):
                        cell[i][j] = alpha[t][i] * trans[i][j] * b[j] * beta[t + 1][j]
                        denom += cell[i][j]
                denom = denom if denom > _EPS else _EPS
                for i in range(2):
                    for j in range(2):
                        xi_sum[i][j] += cell[i][j] / denom

            # M-step.
            pi0 = [gamma[0][i] for i in range(2)]
            for i in range(2):
                gsum = sum(gamma[t][i] for t in range(n))
                gsum = gsum if gsum > _EPS else _EPS
                mu[i] = sum(gamma[t][i] * x[t] for t in range(n)) / gsum
                var[i] = max(
                    sum(gamma[t][i] * (x[t] - mu[i]) ** 2 for t in range(n)) / gsum, _VAR_FLOOR
                )
                row_den = sum(xi_sum[i])
                row_den = row_den if row_den > _EPS else _EPS
                trans[i] = [xi_sum[i][j] / row_den for j in range(2)]

            if ll - prev_ll < _EM_TOL:
                break
            prev_ll = ll

        # Canonicalize: state 0 := high drift (mu[0] >= mu[1]); relabel everything consistently.
        if mu[0] < mu[1]:
            mu = [mu[1], mu[0]]
            var = [var[1], var[0]]
            trans = [[trans[1][1], trans[1][0]], [trans[0][1], trans[0][0]]]
            pi0 = [pi0[1], pi0[0]]

        self.params = HMMParams(
            mu=(mu[0], mu[1]),
            var=(var[0], var[1]),
            trans=((trans[0][0], trans[0][1]), (trans[1][0], trans[1][1])),
            pi0=(pi0[0], pi0[1]),
        )
        return self

    # --- The Hamilton FILTER (causal, one-step-ahead) -- the ONLY thing a live signal may read -----
    def filtered(self, series: tuple[float, ...]) -> list[tuple[float, float]]:
        """Forward Hamilton filter -> `xi_{t|t} = P(state_t | F_t)` for each t in `series`.

        The causal one-step-ahead posterior (SIGNAL §C.2(c)): each `xi[t]` conditions ONLY on
        observations `series[0..t]`, never on any later bar. This is the sole quantity a live entry
        decision may consume. Returns `[(P(high|F_t), P(low|F_t)), ...]`.
        """
        if self.params is None:
            raise RuntimeError("filtered() before fit -- parameters not frozen yet")
        p = self.params
        out: list[tuple[float, float]] = []
        prior = [p.pi0[0], p.pi0[1]]  # P(state_0) before seeing obs 0
        for t, xt in enumerate(series):
            if t > 0:
                # PREDICT: propagate the previous filtered posterior one step through P.
                prev = out[t - 1]
                prior = [
                    prev[0] * p.trans[0][0] + prev[1] * p.trans[1][0],
                    prev[0] * p.trans[0][1] + prev[1] * p.trans[1][1],
                ]
            b = [_norm_pdf(xt, p.mu[i], p.var[i]) for i in range(2)]
            joint = [prior[i] * b[i] for i in range(2)]
            s = joint[0] + joint[1]
            s = s if s > _EPS else _EPS
            out.append((joint[0] / s, joint[1] / s))
        return out

    def signal_at(self, cursor: ScanCursor) -> Side:
        """Causal one-step-ahead entry for the trade at `cursor.t` -- the FILTERED posterior ONLY.

        Runs the Hamilton filter over `cursor.past()` (indices `0 .. t-1`, `F_{t-1}`-measurable) and
        reads the filtered high-drift probability at the LAST past bar. Enters `LONG` when it crosses
        the frozen `threshold`, `SHORT` when it falls below `1 - threshold` (the bear-state filtered
        probability crosses `threshold` -- sign-antisymmetric), else `FLAT` -- the neutral no-trade
        band `(1 - threshold, threshold)` that `threshold > 0.5` guarantees is non-empty (so the
        model is not always in the market). It NEVER reads bar `t` or the smoothed posterior
        (SIGNAL §C.2(c) -- the barred lookahead).
        """
        if self.params is None:
            raise RuntimeError("signal_at before fit -- parameters not frozen yet")
        past = cursor.past()
        # Warm-up honesty: the filter must have seen at least `warmup` bars before it fires.
        if not warmup_ok(len(past), self.warmup) or not past:
            return Side.FLAT
        p_high = self.filtered(past)[-1][0]  # P(bull | F_{t-1}) -- filtered, at the last past bar
        if p_high >= self.threshold:
            return Side.LONG
        if p_high <= 1.0 - self.threshold:
            return Side.SHORT
        return Side.FLAT

    # --- Kim SMOOTHER -- DIAGNOSTIC ONLY, BARRED as a signal (SIGNAL §C.2(c) load-bearing clause) --
    def smoothed(
        self, series: tuple[float, ...], *, for_diagnostics_only: bool = False
    ) -> list[tuple[float, float]]:
        """FULL-SAMPLE Kim (1994) smoother `P(state_t | F_T)` -- LOOKAHEAD, never a live signal.

        `P(state_t | F_T)` conditions on the WHOLE sample (bars AFTER `t`), so it is inadmissible as a
        decision input (SIGNAL §C.1/§C.2(c), SYSTEM §C.2). It exists ONLY for the pinning/diagnostic
        test that asserts `smoothed != filtered`; calling it without `for_diagnostics_only=True`
        RAISES `PurityError` so it can never be wired into a decision path by accident.
        """
        if not for_diagnostics_only:
            raise PurityError(
                "Kim smoother P(state_t|F_T) is FULL-SAMPLE lookahead -- BARRED as a live signal "
                "(SIGNAL §C.2(c)); pass for_diagnostics_only=True only in a diagnostic/pinning test")
        if self.params is None:
            raise RuntimeError("smoothed() before fit")
        p = self.params
        n = len(series)
        if n == 0:
            return []
        filt = self.filtered(series)  # xi_{t|t}
        # Kim smoother: backward pass. gamma_T = filt_T; then propagate P(t|T) from P(t+1|T).
        sm: list[list[float]] = [[filt[t][0], filt[t][1]] for t in range(n)]
        for t in range(n - 2, -1, -1):
            # predicted P(state_{t+1} | F_t)
            pred = [
                filt[t][0] * p.trans[0][0] + filt[t][1] * p.trans[1][0],
                filt[t][0] * p.trans[0][1] + filt[t][1] * p.trans[1][1],
            ]
            pred = [v if v > _EPS else _EPS for v in pred]
            for i in range(2):
                s = 0.0
                for j in range(2):
                    s += sm[t + 1][j] * p.trans[i][j] / pred[j]
                sm[t][i] = filt[t][i] * s
            z = sm[t][0] + sm[t][1]
            z = z if z > _EPS else _EPS
            sm[t] = [sm[t][0] / z, sm[t][1] / z]
        return [(row[0], row[1]) for row in sm]

    # --- forward / backward used ONLY inside EM fit (train fold) -----------------------------------
    def _forward(self, x, mu, var, trans, pi0):
        n = len(x)
        alpha = [[0.0, 0.0] for _ in range(n)]
        scale = [0.0] * n
        b0 = [_norm_pdf(x[0], mu[i], var[i]) for i in range(2)]
        a0 = [pi0[i] * b0[i] for i in range(2)]
        s0 = a0[0] + a0[1]
        s0 = s0 if s0 > _EPS else _EPS
        alpha[0] = [a0[i] / s0 for i in range(2)]
        scale[0] = s0
        for t in range(1, n):
            b = [_norm_pdf(x[t], mu[i], var[i]) for i in range(2)]
            pred = [
                alpha[t - 1][0] * trans[0][j] + alpha[t - 1][1] * trans[1][j] for j in range(2)
            ]
            a = [pred[i] * b[i] for i in range(2)]
            s = a[0] + a[1]
            s = s if s > _EPS else _EPS
            alpha[t] = [a[i] / s for i in range(2)]
            scale[t] = s
        return alpha, scale

    def _backward(self, x, mu, var, trans, scale):
        n = len(x)
        beta = [[0.0, 0.0] for _ in range(n)]
        beta[n - 1] = [1.0, 1.0]
        for t in range(n - 2, -1, -1):
            b = [_norm_pdf(x[t + 1], mu[j], var[j]) for j in range(2)]
            for i in range(2):
                beta[t][i] = sum(trans[i][j] * b[j] * beta[t + 1][j] for j in range(2))
            sc = scale[t + 1] if scale[t + 1] > _EPS else _EPS
            beta[t] = [beta[t][i] / sc for i in range(2)]
        return beta
