"""MODELS.har -- HAR-RV (Corsi 2009) realized-volatility cascade, the Phase-C VOL covariate (EDGE #8).

The single pre-registered VOL-INPUT candidate (docs/EDGE.md "The pre-registered candidate set
(frozen)" #8, SIGNAL §C.4, SYSTEM §C.9 #8), RE-ADMITTED 2026-07-04 in its DIRECTIONAL use ONLY. This
is the heterogeneous-autoregressive realized-variance cascade of Corsi (2009):

    RV_t = c + beta_d * RV^(d)_{t-1} + beta_w * RV^(w)_{t-1} + beta_m * RV^(m)_{t-1} + eps_t

where the three regressors are backward-looking averages of PAST realized variance over the daily
(1-day), weekly (5-day) and monthly (22-day) horizons -- the long-memory cascade. The four
coefficients ``(c, beta_d, beta_w, beta_m)`` are fit by OLS on the TRAIN fold ONLY and then FROZEN
for OOS (SIGNAL §C.4 train-fit/frozen; SYSTEM §C.1). Free-parameter count ``k = 4``.

WHAT IT IS (and is NOT):
  * It is a VOL COVARIATE / INPUT to the pre-registered entry & exit candidates -- NEVER a standalone
    trade signal. Its three admissible directional roles (SIGNAL §C.4, SYSTEM §C.9 #8), all consuming
    the ONE one-step-ahead forecast ``RV_hat_{t|t-1}``:
      (i)   ``vol_scale``          -- vol-scaling of position size / stop-width: scale a target by
            ``target_vol / forecast_vol`` so the book self-normalizes to the current vol regime;
      (ii)  ``adaptive_threshold`` -- normalize a detector statistic (a BOCPD/SPRT/GLR/HMM statistic,
            §C.2) by the forecast vol so the detector self-scales;
      (iii) a covariate feeding the survival hazard (§C.3c) or the jump local-vol (§C.2d) -- consumed
            as the plain forecast level via ``forecast_at``.
  * It is NOT the refuted RV-IV spread signal -- that STAYS BARRED and is not on this menu in any form
    (SIGNAL §C.4, docs/EDGE.md). Nothing here touches implied vol; realized variance only.
  * It is NOT an ``EntryModel`` / ``ExitModel`` (it emits no ``Side`` and no hold/exit bool): it is a
    covariate consumed BY those, so it deliberately does not conform to those protocols. It carries
    the same ``fit(train) -> self`` / frozen-parameter / ``k`` shape and reuses base.py's causal
    machinery (``ScanCursor`` / ``require_causal`` / ``warmup_ok`` / ``TrainFold``).

PURITY IS BINDING (SIGNAL §C.1/§C.4, SYSTEM §C.2 -- the load-bearing clause for this candidate).
The forecast for the value dated ``t`` is a genuine ONE-STEP-AHEAD ``F_{t-1}``-measurable estimate:
its three regressors are averages over lags ``<= t-1`` ONLY (``RV^(d)_{t-1} = RV_{t-1}``,
``RV^(w)_{t-1} = mean(RV_{t-5..t-1})``, ``RV^(m)_{t-1} = mean(RV_{t-22..t-1})``). A SAME-BAR ``RV_t``
read is the classic same-bar lookahead and is BARRED under the DAILY cadence -- ``forecast_at`` reads
ONLY ``cursor.past()`` (indices ``0..t-1``) and every read routes through ``require_causal`` via the
cursor, so a decision that peeks at bar ``t`` (or later) RAISES ``PurityError``. The two-sided /
full-sample HAR fit is inadmissible (lookahead) and is never used for a decision. Warm-up honesty:
the cascade needs 22 days of history, so no forecast fires before ``warmup = MONTHLY_LAG`` bars have
elapsed within the fold (``warmup_ok``); the OLS design matrix on train is built only from rows whose
full 22-day monthly window is available (never borrowing pre-sample/cross-fold data, SIGNAL §C.1).

DATA GATE (SYSTEM §C.8): this module is candidate-model code, SYNTHETIC-TESTED ONLY via a
known-generating-process recovery test (a simulated RV cascade whose true coefficients OLS must
recover) plus a same-bar-RV no-lookahead pinning test. It touches NO lake, emits NO metric, adopts
NO candidate; the empirical selection RUN is BLOCKED-ON-LAKE (``BlockedOnLakeError`` lives with the
RUN, not here).

Imports ONLY ``intraday.models.base`` (the shared interface + causal helpers) and ``intraday.core``
(shared constants) -- NEVER a sibling Phase-C model module (the harness is interface-driven, SYSTEM
§C.3 off-ledger rule). Pure numpy + stdlib; realized variance, not greeks / not IV (CLAUDE.md). Map:
docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ScanCursor, TrainFold, require_causal, warmup_ok

__all__ = [
    "DAILY_LAG",
    "WEEKLY_LAG",
    "MONTHLY_LAG",
    "HARCoefficients",
    "HARRV",
    "har_regressors",
]

# --- Corsi (2009) cascade horizons -- the canonical daily/weekly/monthly lags (frozen, not tuned) --
# These are the STANDARD Corsi HAR horizons, not a searched grid: 1 trading day, 1 trading week
# (5 sessions), and 1 trading month (22 sessions). They fix the model's structure; only the four
# regression coefficients are FIT (on train) -- the lags themselves are part of the pre-registered
# HAR representative (SIGNAL §C.4 "daily/weekly/monthly averages of past RV").
DAILY_LAG = 1
WEEKLY_LAG = 5
MONTHLY_LAG = 22

# Below this the OLS normal-equations are numerically singular (a rank-deficient / constant design):
# a HAR fit on such a fold has no defined coefficients and `fit` raises rather than emit garbage.
_COND_EPS = 1e-12


@dataclass(frozen=True)
class HARCoefficients:
    """The four FROZEN HAR-RV coefficients (Corsi 2009), fit by OLS on train (SIGNAL §C.4).

    ``c`` is the intercept; ``beta_d`` / ``beta_w`` / ``beta_m`` weight the daily / weekly / monthly
    lagged-RV cascade regressors. These are estimated ONCE on the train fold and FROZEN for OOS -- no
    re-fit or re-tune on TEST (SYSTEM §C.1). The parsimony free-parameter count is ``k = 4`` (SYSTEM
    §C.4); the recovery test asserts OLS recovers a simulated cascade's true ``(c, beta_d, beta_w,
    beta_m)`` within tolerance.
    """

    c: float
    beta_d: float
    beta_w: float
    beta_m: float

    def predict(self, rv_d: float, rv_w: float, rv_m: float) -> float:
        """One-step-ahead RV forecast from the three lagged-RV regressors (all ``F_{t-1}``-measurable).

        ``RV_hat = c + beta_d*rv_d + beta_w*rv_w + beta_m*rv_m``. The caller (``HARRV.forecast_at``)
        builds ``rv_d/rv_w/rv_m`` from PAST RV only, so this is a genuine one-step-ahead forecast.
        """
        return self.c + self.beta_d * rv_d + self.beta_w * rv_w + self.beta_m * rv_m


def har_regressors(past_rv: np.ndarray) -> tuple[float, float, float]:
    """Build the daily/weekly/monthly HAR regressors from a strictly-PAST RV window (``F_{t-1}``).

    ``past_rv`` is the realized-variance series over indices ``0 .. t-1`` ONLY (i.e. ``cursor.past()``
    for a forecast dated ``t``) -- it must NOT include ``RV_t`` (that is the same-bar lookahead this
    whole candidate is scoped to avoid, SIGNAL §C.4 / SYSTEM §C.2). Returns::

        RV^(d)_{t-1} = RV_{t-1}                       (the last, most-recent past value)
        RV^(w)_{t-1} = mean(RV_{t-5 .. t-1})          (trailing WEEKLY_LAG average)
        RV^(m)_{t-1} = mean(RV_{t-22 .. t-1})         (trailing MONTHLY_LAG average)

    Averages use the AVAILABLE trailing window when fewer than the full lag exists at the head of a
    fold (a genuine causal partial average, never borrowing pre-sample data); callers gate on
    ``warmup_ok(t, MONTHLY_LAG)`` so a live forecast fires only once the full 22-day window exists.
    Raises if ``past_rv`` is empty (no history -> no regressors / no forecast).
    """
    n = len(past_rv)
    if n == 0:
        raise ValueError("har_regressors needs a non-empty PAST-RV window (>= 1 lagged day)")
    rv_d = float(past_rv[-1])
    rv_w = float(past_rv[-WEEKLY_LAG:].mean())
    rv_m = float(past_rv[-MONTHLY_LAG:].mean())
    return rv_d, rv_w, rv_m


class HARRV:
    """HAR-RV (Corsi 2009) realized-vol cascade -- the DIRECTIONAL VOL covariate (EDGE #8, SIGNAL §C.4).

    A vol INPUT to the entry & exit candidates, NEVER a standalone signal (SIGNAL §C.4). Contract:
      * ``k = 4`` -- the parsimony free-parameter count ``(c, beta_d, beta_w, beta_m)`` the SYSTEM §C.4
        information-criterion penalty is charged against.
      * ``fit(train: TrainFold) -> HARRV`` -- OLS-fit the four coefficients on the TRAIN fold's RV
        series ONLY, then FREEZE (``self.coef``). ``TrainFold.series`` is the per-day realized variance
        (base.py TrainFold: "realized variance for HAR-RV"), aligned to ``train.days`` one-to-one.
      * ``forecast_at(cursor: ScanCursor) -> float`` -- the one-step-ahead ``RV_hat_{t|t-1}`` for the
        value dated ``cursor.t``, built from ``cursor.past()`` (indices ``0..t-1``) ONLY, so it is
        strictly ``F_{t-1}``-measurable. A same-bar ``RV_t`` read is BARRED (a probe read of bar
        ``cursor.t`` routes through ``require_causal`` and RAISES ``PurityError`` under the DAILY
        cadence -- the pinning hook).

    Directional roles (all consuming the ONE ``forecast_at`` output; SIGNAL §C.4, SYSTEM §C.9 #8):
      * ``vol_scale(target, target_vol, cursor)`` -- scale a size / stop-width ``target`` by
        ``target_vol / forecast_vol`` where ``forecast_vol = sqrt(RV_hat)`` (vol regime self-norm);
      * ``adaptive_threshold(base_threshold, cursor)`` -- normalize a detector's base threshold by the
        forecast vol so a §C.2 detector self-scales to the current regime.

    Interface-style, not a base class: it is NOT an ``EntryModel``/``ExitModel`` (it emits no ``Side``
    / no hold-exit bool -- it is the covariate those consume). The real entry/exit candidates are NOT
    imported here (SYSTEM §C.3 off-ledger / interface-driven-harness rule).
    """

    k = 4

    def __init__(self) -> None:
        self.coef: HARCoefficients | None = None  # set on fit, then FROZEN
        self.warmup: int = MONTHLY_LAG            # the 22-day cascade burn-in (SIGNAL §C.1)

    # --- fit (train-only OLS, then frozen) --------------------------------------------------------
    def fit(self, train: TrainFold) -> HARRV:
        """OLS-fit ``(c, beta_d, beta_w, beta_m)`` on the TRAIN fold's RV series, then FREEZE it.

        Builds the HAR design matrix from the train RV series: for each target index ``t`` whose full
        monthly window is available (``t >= MONTHLY_LAG``, respecting ``train.warmup`` too), the row is
        ``[1, RV^(d)_{t-1}, RV^(w)_{t-1}, RV^(m)_{t-1}]`` (all lagged ``<= t-1``, built by
        ``har_regressors`` on ``rv[:t]``) and the target is ``RV_t``. So the fit itself is causal by
        construction -- no target row uses its own or a future RV as a regressor. Solves the normal
        equations by ``numpy.linalg.lstsq`` (scipy-free), freezes the coefficients, and returns self.

        Raises ``ValueError`` if the train fold cannot furnish at least 4 causal rows (an
        under-determined HAR fit) or if the design is numerically rank-deficient -- never emits garbage
        coefficients. Parameters are FROZEN for OOS: no re-fit / re-tune on TEST (SYSTEM §C.1).
        """
        rv = np.asarray(train.series, dtype="float64")
        start = max(MONTHLY_LAG, int(train.warmup))
        rows: list[list[float]] = []
        targets: list[float] = []
        for t in range(start, len(rv)):
            rv_d, rv_w, rv_m = har_regressors(rv[:t])  # strictly past -> F_{t-1}-measurable regressors
            rows.append([1.0, rv_d, rv_w, rv_m])
            targets.append(float(rv[t]))
        if len(rows) < 4:
            raise ValueError(
                "HAR-RV fit needs >= 4 causal rows (train fold too short for the 22-day cascade)")
        x = np.asarray(rows, dtype="float64")
        y = np.asarray(targets, dtype="float64")
        # Guard a rank-deficient / constant design (e.g. a flat RV path): normal-equations singular.
        if np.linalg.matrix_rank(x, tol=_COND_EPS) < x.shape[1]:
            raise ValueError("HAR-RV design matrix rank-deficient -- coefficients undefined")
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.coef = HARCoefficients(c=float(beta[0]), beta_d=float(beta[1]),
                                    beta_w=float(beta[2]), beta_m=float(beta[3]))
        return self

    # --- causal one-step-ahead forecast (the ONE output the roles consume) -----------------------
    def forecast_at(self, cursor: ScanCursor) -> float:
        """One-step-ahead ``RV_hat_{t|t-1}`` for the value dated ``cursor.t`` (``F_{t-1}``-measurable).

        Reads ONLY ``cursor.past()`` (indices ``0..t-1``): a SAME-BAR ``RV_t`` read is a leak and is
        prevented two ways -- (a) ``cursor.past()`` never exposes ``series[t]`` (the causal window is
        ``series[:t]``), and (b) a probe read of ``cursor.t`` routes through ``require_causal`` and
        RAISES ``PurityError`` under the DAILY cadence (the pinning hook). The regressors are built by
        ``har_regressors`` on the past window; the frozen coefficients turn them into the forecast.

        Raises if called before ``fit`` (parameters not frozen) or before the 22-day warm-up is
        satisfied at ``cursor.t`` (``warmup_ok`` -- warm-up honesty, SIGNAL §C.1); the caller skips
        the bar while warming up rather than borrowing pre-sample history.
        """
        if self.coef is None:
            raise RuntimeError("forecast_at before fit -- HAR coefficients not frozen yet")
        if not warmup_ok(cursor.t, self.warmup):
            raise ValueError(
                f"HAR-RV warm-up not satisfied at t={cursor.t} (need >= {self.warmup} past days)")
        past = np.asarray(cursor.past(), dtype="float64")  # series[:t] -- strictly F_{t-1}
        rv_d, rv_w, rv_m = har_regressors(past)
        return self.coef.predict(rv_d, rv_w, rv_m)

    def forecast_vol_at(self, cursor: ScanCursor) -> float:
        """The one-step-ahead forecast VOLATILITY ``sqrt(max(RV_hat, 0))`` (RV is a variance).

        The HAR cascade forecasts realized VARIANCE; the vol-scaling / adaptive-threshold roles need a
        standard deviation, so this is ``sqrt`` of the (non-negative-clamped) variance forecast. A
        fitted HAR can in principle emit a slightly-negative variance forecast on an out-of-regime
        input (OLS is not sign-constrained); it is clamped at 0 so ``sqrt`` is always defined -- a zero
        forecast vol then makes the roles fall back to the unscaled target (see below).
        """
        return float(np.sqrt(max(self.forecast_at(cursor), 0.0)))

    # --- the two directional VOL roles (both consume the ONE forecast; SIGNAL §C.4) ---------------
    def vol_scale(self, target: float, target_vol: float, cursor: ScanCursor) -> float:
        """Vol-scale a size / stop-width ``target`` by ``target_vol / forecast_vol`` (role i, §C.4).

        Self-normalizes ``target`` (a position size or a stop width) to the current vol regime: when
        the one-step-ahead forecast vol is high the scaled target shrinks, when it is low it grows,
        holding vol-adjusted exposure roughly constant. ``target_vol`` is the desired reference vol
        (a train-frozen constant the caller supplies, NOT tuned on TEST). A zero / non-finite forecast
        vol (a degenerate regime) returns ``target`` unscaled -- never a divide-by-zero. The forecast
        is ``F_{t-1}``-measurable (``forecast_vol_at``), so the scale applied to a bar-``t`` decision
        is itself causal.
        """
        if target_vol < 0:
            raise ValueError("target_vol must be non-negative (a reference volatility)")
        fv = self.forecast_vol_at(cursor)
        if fv <= 0.0 or not np.isfinite(fv):
            return target  # degenerate forecast -> fall back to the unscaled target (no div-by-zero)
        return target * (target_vol / fv)

    def adaptive_threshold(self, base_threshold: float, cursor: ScanCursor) -> float:
        """Scale a detector's ``base_threshold`` by the forecast vol so a §C.2 detector self-scales.

        The adaptive-threshold role (role ii, §C.4): a change-point / sequential / regime detector's
        trigger level is multiplied by the one-step-ahead forecast vol so the SAME detector adapts to
        the current regime (a wider band in a high-vol regime, tighter in a calm one) instead of using
        a single fixed level. ``base_threshold`` is the train-frozen level; the forecast is
        ``F_{t-1}``-measurable, so the adapted threshold used at bar ``t`` is causal. A zero forecast
        vol returns the un-adapted ``base_threshold`` (no spurious collapse to zero).
        """
        fv = self.forecast_vol_at(cursor)
        if fv <= 0.0 or not np.isfinite(fv):
            return base_threshold
        return base_threshold * fv

    # --- the same-bar-lookahead pinning HOOK (asserted by the pinning test) -----------------------
    def _probe_same_bar_read(self, cursor: ScanCursor) -> float:
        """PINNING HOOK: attempt to read bar ``cursor.t`` (the SAME-BAR ``RV_t``) through the guard.

        This exists ONLY so the no-lookahead pinning test can assert that reading the same bar the
        forecast is *for* RAISES ``PurityError`` under the DAILY cadence (the classic same-bar RV
        leak, SIGNAL §C.4 / SYSTEM §C.2). ``forecast_at`` itself NEVER calls this -- it reads only
        ``cursor.past()``; the model's live path is causal by construction and this probe merely pins
        that a same-bar read would break, exactly the mutation the pinning test forbids.
        """
        require_causal(cursor.t, cursor.t, cursor.cadence)  # DAILY: read_index == t -> leak -> RAISES
        return cursor.series[cursor.t]  # unreachable under DAILY (guard raised above)
