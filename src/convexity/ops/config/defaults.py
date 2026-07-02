"""OPS RED shared control constants (``docs/specs/OPS.md`` sec 7 / sec 4.2 / sec 4.3 / sec 5).

Every constant here is exactly one governance bucket (``core/params.py::Bucket``). All are RED
(shared control constants entering ONLY via ``resolve_config(defaults=...)``, Inv C) or EVIDENCE-
anchored [MEASURED]/[DERIVED] values whose derivation is cited inline. There is no BLUE grid: OPS
is a control layer whose constants are calibrated once against a stated target, never searched for
best backtest performance (OPS.md sec 7, P8).

Nothing here is per-instrument (Inv C): reconciliation tolerance multiples, the CUSUM boundary, and
the SLA budget are all cross-instrument RED, never an ``InstrumentSpec`` field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---- annualization (must match research/xsec/metrics.py; OPS introduces no second week-count) ----
# research/xsec/metrics.py::PERIODS_PER_YEAR == 365.25 / 7 ~= 52.18 (OPS.md sec 4.2).
PERIODS_PER_YEAR: float = 365.25 / 7.0


@dataclass(frozen=True)
class OpsDefaults:
    """The RED constants OPS calibrates against (OPS.md sec 7 RED bucket).

    Constructed once, injected via ``resolve_config(defaults=...)`` (Inv C). Field-by-field spec
    citation is in the attribute comments below.
    """

    # ---- sec 1.2 reconciliation tolerance scaling (RED) ----
    k_lot: float = 2.0  # OPS.md sec 1.2: tolerate a couple of lots of Quantizer rounding slop
    k_notional: float = 1.0e-3  # OPS.md sec 1.2: 0.1% of intended notional
    # ---- sec 1.4 escalation ----
    m_warn: float = 3.0  # OPS.md sec 1.4: WARN band = (epsilon_i, m_warn * epsilon_i]
    staleness_max_days: int = 2  # OPS.md sec 1.4: RECON_STALE -> BREACH after this many unreadable days
    # ---- sec 1.3 cadence ----
    reconciliation_cadence_days: int = 1  # OPS.md sec 1.3: daily, strictly < rebalance_days = 7
    rebalance_days: int = 7  # OPS.md sec 7 AMBER: matches XSecConfig.rebalance_days

    # ---- sec 4.2 the validated null the CUSUM standardizes against ----
    sr_0: float = 0.85  # [MEASURED] OPS.md sec 4.2: LOWER end of the honest net Sharpe band ~0.8-1.0 (EDGE.md sec b, "Honest performance" table)
    # [DERIVED] OPS.md sec 4.2's ~28% annualized book-vol anchor. NOT a literal EDGE.md figure (EDGE.md
    # sec b's "Honest performance" table prints Sharpe, CAGR, maxDD and win-rate but no vol); it is
    # DERIVED from TWO primary EDGE.md numbers via a closed-form identity, so no value is fabricated:
    #   * net Sharpe lower bound 0.85  == sr_0 above          [MEASURED, EDGE.md sec b]
    #   * CAGR @100% gross lower bound 0.20                    [MEASURED, EDGE.md sec b]
    # Annualized Sharpe identity SR = arith_mean/vol, and geometric<->arithmetic drift correction
    # arith_mean = CAGR + 0.5*vol**2, combine into the fixed point  vol = (CAGR + 0.5*vol**2)/SR.
    # Solving at EDGE.md's lower (SR=0.85, CAGR=0.20) gives vol* = 0.28211 -> 0.28 (see test
    # test_ann_vol_0_is_derived_from_edge_sharpe_and_cagr, and OPS.md sec 4.2's derivation block).
    # Round-trip: vol=0.28,SR=0.85 => arith=0.2380 => CAGR=0.1988 ~= EDGE.md's 0.20 lower bound; and
    # EDGE.md maxDD 0.25 is a coherent third leg (Calmar = 0.20/0.25 = 0.80 ~= sr_0). Lower-band inputs
    # widen the null (bigger vol -> smaller |z_k|), the same conservatism sr_0's lower bound states.
    ann_vol_0: float = 0.28

    # ---- sec 4.3 CUSUM design constants ----
    delta_target: float = 1.0  # [ILLUSTRATIVE-RED] OPS.md sec 4.3: standardized Sharpe shift the chart detects fastest
    # d = delta_target/2 (classical reference/allowance value; OPS.md sec 4.3, == REGIME.md's `k`).
    # h* = 6.23 is the boundary the Siegmund ARL solve produces for ARL_0_target=1000, d=0.5
    # (OPS.md sec 4.3 [DERIVED]; reproducible offline, this constant is not re-solved live).
    h_star: float = 6.23  # [DERIVED-RED] OPS.md sec 4.3
    arl_0_target: float = 1000.0  # [ILLUSTRATIVE-RED] OPS.md sec 4.3: ~19.5yr, >=5%/yr false-HALT budget

    # ---- sec 4.4 escalation-from-CUSUM constants ----
    g_degrade: float = 0.5  # [ILLUSTRATIVE-RED] OPS.md sec 4.4: gross multiplier while in DEGRADE
    n_confirm: int = 4  # [ILLUSTRATIVE-RED] OPS.md sec 4.4: consecutive healthy weeks to exit DEGRADE

    # ---- sec 7 TEAL-but-slow ----
    sigma_0_reestimation_weeks: int = 52  # OPS.md sec 7 / sec 13: annual, slow enough the null can't chase a break

    # ---- sec 5 SLA budget ----
    m_slip: float = 2.0  # [ILLUSTRATIVE-RED] OPS.md sec 5.2: realized slippage <= m_slip * expected
    sla_severe_lag_multiple: float = 2.0  # OPS.md sec 5.3: completion_lag > this * W_exec -> HALT
    sla_severe_slip_multiple: float = 3.0  # OPS.md sec 5.3: realized_slippage > this * m_slip * expected -> HALT

    @property
    def d(self) -> float:
        """The CUSUM reference/allowance value ``d = delta_target / 2`` (OPS.md sec 4.3).

        Renamed from the classical ``k`` (== REGIME.md sec 4's symbol) only to avoid a clash with
        the ``CUSUM`` Python class; calibrated independently here against OPS's own Sharpe-drift null.
        """
        return self.delta_target / 2.0

    @property
    def mu_0(self) -> float:
        """[DERIVED] Per-week mean return under the null, ``SR_0 / sqrt(PERIODS_PER_YEAR)`` (sec 4.2)."""
        return self.sr_0 / math.sqrt(PERIODS_PER_YEAR)

    @property
    def sigma_0(self) -> float:
        """[DERIVED] Per-week return std-dev under the null, ``ann_vol_0 / sqrt(PERIODS_PER_YEAR)`` (sec 4.2)."""
        return self.ann_vol_0 / math.sqrt(PERIODS_PER_YEAR)


DEFAULTS = OpsDefaults()
