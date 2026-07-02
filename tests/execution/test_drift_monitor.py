"""Drift-triggered path (EXECUTION.md §6.2-6.3, §11.2): Invariant CAL-PRIMARY — REGIME-gated,
risk-off only, never a parallel alpha cadence."""

from __future__ import annotations

import numpy as np

from research.xsec.execution import (
    CostFacts,
    ExecTarget,
    ExecutionDefaults,
    band_width,
    drift_check,
    solve_trajectory,
)

_N = 6
_CFG = ExecutionDefaults(rho=2.0, band_c=0.5)
_FACTS = CostFacts(
    eta=np.full(_N, 2e-4),
    gamma=np.full(_N, 1e-5),
    theta=np.full(_N, 1.3e-4),
    sigma=np.full(_N, 0.004),
)


def test_drift_trigger_is_noop_without_regime_flag():
    # Invariant CAL-PRIMARY (§6.2): without the REGIME defensive flag the drift path is a no-op,
    # regardless of drift magnitude — spanning 0 to 10x the band width (§11.2 table).
    b = band_width(float(_FACTS.theta[0]), _CFG.rho, c=_CFG.band_c)
    w_target = np.full(_N, 0.05)
    shortable = np.ones(_N, dtype=bool)
    for scale in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        carried = w_target + scale * b  # LARGE_SYNTHETIC_DRIFT at scale x band
        plan = drift_check(
            ExecTarget(w_target, shortable), carried, _FACTS, _CFG, regime_defensive=False
        )
        assert plan is None


def test_drift_armed_is_reduce_only():
    # §6.2: armed, the drift path only REDUCES gross — a name whose move toward target would
    # INCREASE its gross exposure is blocked, never traded early.
    b = band_width(float(_FACTS.theta[0]), _CFG.rho, c=_CFG.band_c)
    w_target = np.array([0.06, 0.01, -0.06, 0.02, 0.0, 0.0])
    carried = np.array([0.01, 0.06, -0.01, 0.02 + 0.5 * b, 0.03, 0.0])
    plan = drift_check(
        ExecTarget(w_target, np.ones(_N, dtype=bool)), carried, _FACTS, _CFG, regime_defensive=True
    )
    assert plan is not None
    # every planned trade weakly reduces that name's gross (risk-off valve).
    assert np.all(np.abs(carried + plan.trade) <= np.abs(carried) + 1e-15)
    assert plan.trade[0] == 0.0 and plan.reason[0] == "GROSS-INCREASE-BLOCKED"  # would add gross
    assert plan.trade[1] < 0.0  # trimming an over-weight long: allowed
    assert plan.trade[2] == 0.0 and plan.reason[2] == "GROSS-INCREASE-BLOCKED"  # bigger short = more gross
    assert plan.trade[3] == 0.0 and plan.reason[3] == "BAND-SKIP"  # inside the band: no trade (§6.3)
    assert plan.trade[4] < 0.0  # closing a name the target dropped: pure risk-off
    # and the plan still respects E-monotone against the naive full rebalance.
    assert plan.turnover_live <= plan.turnover_backtest + 1e-12


def test_sign_conventions():
    # armed drift trades move TOWARD the currently-valid w_target, never past it (§6.3).
    w_target = np.zeros(_N)
    carried = np.full(_N, 0.04)
    plan = drift_check(
        ExecTarget(w_target, np.ones(_N, dtype=bool)), carried, _FACTS, _CFG, regime_defensive=True
    )
    assert plan is not None
    assert np.all(plan.trade <= 0.0)  # closing longs: sells only
    assert np.all(np.abs(plan.trade) <= np.abs(plan.x) + 1e-15)


def test_twap_limit_recovery():
    res = solve_trajectory(-0.03, delta=2.0, n=8, sigma_i=0.004, eta_i=2e-4, gamma_i=0.0, lambda_i=0.0)
    assert res.kappa == 0.0
    assert np.all(res.schedule == -0.03 / 8)
