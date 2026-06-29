"""Synthetic adversarial scenarios so LEVERAGE is testable with no external data (§14).

Each generator returns a fully-wired ``LeverageScenario`` (signal + risk + cfg + book)
that drives a specific ceiling to bind, plus the behavioral assertion the test checks.
The 13 scenarios mirror §14 and BUILD_PLAN Phase 3:

  1.  calm                     — vol-target ceiling binds, leverage high but capped.
  2.  vol_spike                — vol/stop ceilings tighten; φ may drop to 1/3.
  3.  fat_tail                 — tail ceiling binds via EVT.
  4.  adverse_gap              — gap-survival sub-test of the tail ceiling.
  5.  correlation_crisis       — portfolio ceiling + governor + margin spike engage.
  6.  regime_cut_full_equity   — g_t=0.25, NO drawdown: L_portfolio binds via g_t·H_book,
                                 binding='portfolio'/'regime_heat_cut', governor=1.00 (Inv R/P8).
  7.  no_regime_overlay        — g_t=1.0 dual: reproduces the no-regime book.
  8.  estimator_pathology      — absurd forecast vol / near-singular Σ -> class cap clamps,
                                 conditioning gate trips cov_cond_breach.
  9.  progressive_drawdown     — governor steps 1.00->0.75->0.50->0.25->0.00.
  10. correlated_signals       — ERC/MCTR shares one partially-netted budget.
  11. margin_explosion         — stressed L_margin prevents forced liquidation.
  12. thin_liquidity           — L_liquidity binds or rejects.
  13. min_lot_too_large        — LotSizer rejects instead of over-risking.

Pure-Python + numpy, seeded, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.leverage.config.asset_class import DEFAULT_PRESETS
from convexity.leverage.config.defaults import PolicyDefaults
from convexity.leverage.config.instrument import InstrumentLeverageSpec
from convexity.leverage.config.resolve import ResolvedConfig, resolve_config
from convexity.leverage.types import (
    BookStateFixture,
    ConditionalMAE,
    GapModel,
    GPDFit,
    LiquidityState,
    RiskInputs,
    SignalInput,
)


@dataclass(frozen=True)
class LeverageScenario:
    name: str
    signal: SignalInput
    risk: RiskInputs
    cfg: ResolvedConfig
    book: BookStateFixture
    carry: float
    note: str


# --------------------------------------------------------------------------- builders

def _spec(symbol: str = "BTCUSDT", asset_class: str = "crypto_perp",
          lot_step: float = 0.001, mult: float = 1.0, exchange_max_lots: int | None = None,
          min_notional: float = 0.0) -> InstrumentLeverageSpec:
    return InstrumentLeverageSpec(
        symbol=symbol, asset_class=asset_class, lot_step=lot_step,
        contract_multiplier=mult, tick_size=0.01, min_notional=min_notional,
        exchange_max_lots=exchange_max_lots,
    )


def _cfg(spec: InstrumentLeverageSpec, defaults: PolicyDefaults | None = None) -> ResolvedConfig:
    return resolve_config(
        spec, defaults=defaults or PolicyDefaults(), preset=DEFAULT_PRESETS[spec.asset_class]
    )


def _healthy_cov(symbols: tuple[str, ...], vol: float = 0.2, rho: float = 0.1) -> np.ndarray:
    n = len(symbols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    sig = np.full(n, vol)
    return (sig[:, None] * corr) * sig[None, :]


def _cond_mae(low: float = 0.02, high: float = 0.05, glob: float = 0.03,
              n: int = 100) -> ConditionalMAE:
    return ConditionalMAE(
        quantile_by_bucket={"low_vol": low, "high_vol": high, "global": glob},
        n_by_bucket={"low_vol": n, "high_vol": n, "global": n},
    )


def _gpd(ok: bool = True, shape: float = 0.1, scale: float = 0.02, thresh: float = 0.03,
         n_exceed: int = 60, emp_es: float = 0.08, tail_prob: float = 0.05) -> GPDFit:
    return GPDFit(
        threshold=thresh, shape=shape, scale=scale, n_exceed=n_exceed,
        tail_prob=tail_prob, ok=ok, emp_es=emp_es,
    )


def _gap(worst: float = 0.04, typical: float = 0.01, n: int = 50,
         stress: float = 0.08) -> GapModel:
    return GapModel(worst_adverse_gap=worst, typical_gap=typical, n_gaps=n, class_stress_gap=stress)


def _liq(spread: float = 2.0, vol_window: float = 1.0e7, adv: float = 1.0e8,
         impact: float = 3.0, stale: bool = False) -> LiquidityState:
    return LiquidityState(
        spread_bps=spread, expected_volume_exit_window=vol_window, adv_notional=adv,
        top_of_book_notional=1.0e6, impact_bps=impact, stale_or_halted=stale,
    )


def _risk(*, forecast_vol: float, vol_pct: float, cov: np.ndarray, cov_index: tuple[str, ...],
          corr_instab: float = 0.1, jump: float = 0.5, gap: GapModel | None = None,
          tail: GPDFit | None = None, mae: ConditionalMAE | None = None,
          regime_bucket: str = "low_vol", free_margin: float = 10_000.0,
          financing: float = 0.0, liq: LiquidityState | None = None,
          peak: float = 10_000.0, equity: float = 10_000.0, account_equity: float = 10_000.0,
          g_t: float = 1.0, mu_hat: float = 0.04, horizon: float = 5.0,
          weights: dict | None = None) -> RiskInputs:
    # The book-owned mirrors (cov/cov_index/peak/equity/account_equity/free_margin/weights)
    # are NOT on RiskInputs (one risk view, §9); the scenario carries them on ``book`` only.
    # They remain ``_risk`` kwargs so each builder can pass ONE coherent set to BOTH _risk
    # and _book, keeping the fixture's book/risk pictures provably consistent.
    del cov, cov_index, free_margin, peak, equity, account_equity, weights  # -> _book
    return RiskInputs(
        forecast_vol=forecast_vol, vol_percentile=vol_pct,
        corr_instability=corr_instab, jump_intensity=jump, gap_hist=gap or _gap(),
        loss_tail=tail or _gpd(), cond_mae=mae or _cond_mae(), regime_bucket=regime_bucket,
        financing_rate=financing, liquidity=liq or _liq(),
        g_t=g_t, mu_hat=mu_hat, holding_horizon_periods=horizon,
    )


def _book(cov: np.ndarray, cov_index: tuple[str, ...], *, equity: float = 10_000.0,
          peak: float = 10_000.0, free_margin: float = 10_000.0,
          positions: dict | None = None, weights: dict | None = None) -> BookStateFixture:
    return BookStateFixture(
        equity=equity, peak_equity=peak, free_margin=free_margin,
        positions=positions or {}, cov=cov, cov_index=cov_index,
        candidate_weights=weights or {}, governor_mult=1.0,
    )


def _signal(symbol: str = "BTCUSDT", side: int = 1, entry: float = 30_000.0,
            margin: float = 1_000.0, c: float = 0.13) -> SignalInput:
    return SignalInput(
        instrument=symbol, side=side, entry_price=entry, position_margin=margin, capital_cap=c
    )


# --------------------------------------------------------------------------- the 13 scenarios

def calm() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.10)
    risk = _risk(forecast_vol=0.06, vol_pct=0.3, cov=cov, cov_index=idx, regime_bucket="low_vol")
    return LeverageScenario("calm", _signal(), risk, _cfg(spec), _book(cov, idx), 0.0,
                            "low vol -> L_vol high but capped; class cap may bind")


def vol_spike() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.6)
    risk = _risk(forecast_vol=0.35, vol_pct=0.95, cov=cov, cov_index=idx,
                 corr_instab=0.2, jump=1.0, regime_bucket="high_vol")
    return LeverageScenario("vol_spike", _signal(), risk, _cfg(spec), _book(cov, idx), 0.0,
                            "vol spike -> L_vol/L_stop tighten; phi may drop to 1/3")


def fat_tail() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.2)
    tail = _gpd(ok=True, shape=0.3, scale=0.06, thresh=0.05, n_exceed=80, emp_es=0.20)
    risk = _risk(forecast_vol=0.08, vol_pct=0.5, cov=cov, cov_index=idx, tail=tail)
    return LeverageScenario("fat_tail", _signal(), risk, _cfg(spec), _book(cov, idx), 0.0,
                            "fat loss tail -> L_tail binds via EVT ES budget")


def adverse_gap() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.2)
    gap = _gap(worst=0.18, typical=0.04, n=40, stress=0.15)
    risk = _risk(forecast_vol=0.08, vol_pct=0.5, cov=cov, cov_index=idx, gap=gap)
    return LeverageScenario("adverse_gap", _signal(), risk, _cfg(spec), _book(cov, idx), 0.0,
                            "large adverse gap -> L_tail gap-survival sub-constraint binds")


def correlation_crisis() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    cov = _healthy_cov(idx, vol=0.5, rho=0.95)  # correlations to ~1
    weights = {"ETHUSDT": 0.15, "SOLUSDT": 0.15}
    risk = _risk(forecast_vol=0.3, vol_pct=0.9, cov=cov, cov_index=idx, corr_instab=0.7,
                 g_t=0.5, peak=10_000.0, equity=7_500.0, weights=weights, free_margin=4_000.0)
    return LeverageScenario("correlation_crisis", _signal(), risk, _cfg(spec),
                            _book(cov, idx, equity=7_500.0, weights=weights, free_margin=4_000.0),
                            0.0, "corr->1 + drawdown: portfolio + governor + margin all engage")


def regime_cut_full_equity() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT", "ETHUSDT")
    cov = _healthy_cov(idx, vol=0.2, rho=0.3)
    weights = {"ETHUSDT": 0.1}  # light existing book so the regime-scaled budget still binds
    # g_t = 0.25, NO equity drawdown (peak == equity): L_portfolio binds via g_t*H_book,
    # governor stays 1.00 (Inv R / P8 disjoint axes).
    risk = _risk(forecast_vol=0.05, vol_pct=0.3, cov=cov, cov_index=idx, g_t=0.25,
                 peak=10_000.0, equity=10_000.0, weights=weights)
    return LeverageScenario("regime_cut_full_equity", _signal(), risk, _cfg(spec),
                            _book(cov, idx, weights=weights), 0.0,
                            "g_t=0.25 at full equity -> portfolio/regime_heat_cut, governor=1.0")


def no_regime_overlay() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT", "ETHUSDT")
    cov = _healthy_cov(idx, vol=0.2, rho=0.3)
    weights = {"ETHUSDT": 0.1}
    risk = _risk(forecast_vol=0.05, vol_pct=0.3, cov=cov, cov_index=idx, g_t=1.0,
                 weights=weights)
    return LeverageScenario("no_regime_overlay", _signal(), risk, _cfg(spec),
                            _book(cov, idx, weights=weights), 0.0,
                            "g_t=1.0 dual of scenario 6 -> identical to no-regime book")


def estimator_pathology() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    # near-singular cov: one tiny eigenvalue
    base = _healthy_cov(idx, vol=0.2, rho=0.999999)
    weights = {"ETHUSDT": 0.05, "SOLUSDT": 0.05}  # light book so heat is not the binder
    # absurdly small forecast vol -> L_vol explodes; the HARD class cap must clamp it.
    # Loose tail/gap so the class cap (not the tail) is the would-be binder after clamp.
    tail = _gpd(ok=True, shape=0.05, scale=0.005, thresh=0.01, n_exceed=80, emp_es=0.02)
    gap = _gap(worst=0.01, typical=0.005, n=80, stress=0.02)
    risk = _risk(forecast_vol=1e-6, vol_pct=0.1, cov=base, cov_index=idx, mu_hat=10.0,
                 tail=tail, gap=gap, weights=weights)
    return LeverageScenario("estimator_pathology", _signal(), risk, _cfg(spec),
                            _book(base, idx, weights=weights), 0.0,
                            "absurd L + near-singular Sigma -> class_cap_capped + cov_cond_breach")


def progressive_drawdown() -> list[LeverageScenario]:
    """Returns one scenario per drawdown band so the governor ladder can be swept."""
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.15)
    out: list[LeverageScenario] = []
    for dd, label in [(0.05, "1.00"), (0.15, "0.75"), (0.25, "0.50"), (0.35, "0.25"),
                      (0.415, "0.125"), (0.42, "0.0833"), (0.45, "0.00")]:
        equity = 10_000.0 * (1.0 - dd)
        risk = _risk(forecast_vol=0.06, vol_pct=0.3, cov=cov, cov_index=idx,
                     peak=10_000.0, equity=equity)
        out.append(LeverageScenario(
            f"progressive_drawdown_dd{dd}", _signal(), risk, _cfg(spec),
            _book(cov, idx, equity=equity), 0.0, f"dd={dd} -> governor mult {label}"))
    return out


def correlated_signals() -> LeverageScenario:
    spec = _spec()
    idx = ("NIFTY", "BANKNIFTY", "GOLD")
    # NIFTY/BANKNIFTY highly correlated, GOLD diversifying
    corr = np.array([[1.0, 0.9, 0.05], [0.9, 1.0, 0.05], [0.05, 0.05, 1.0]])
    sig = np.array([0.18, 0.20, 0.14])
    cov = (sig[:, None] * corr) * sig[None, :]
    weights = {"NIFTY": 0.4, "BANKNIFTY": 0.4, "GOLD": 0.4}
    risk = _risk(forecast_vol=0.08, vol_pct=0.4, cov=cov, cov_index=idx, weights=weights)
    return LeverageScenario("correlated_signals", _signal("NIFTY", entry=22_000.0), risk,
                            _cfg(spec), _book(cov, idx, weights=weights), 0.0,
                            "correlated cluster shares one partially-netted budget (< naive sum)")


def margin_explosion() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.3)
    # very tight free margin -> stressed L_margin binds first and prevents forced liquidation
    risk = _risk(forecast_vol=0.1, vol_pct=0.6, cov=cov, cov_index=idx, free_margin=400.0)
    return LeverageScenario("margin_explosion", _signal(), risk, _cfg(spec),
                            _book(cov, idx, free_margin=400.0), 0.0,
                            "tight free margin -> stressed L_margin prevents liquidation")


def thin_liquidity() -> LeverageScenario:
    spec = _spec()
    idx = ("BTCUSDT",)
    cov = _healthy_cov(idx, vol=0.2)
    # very thin exit-window volume so participation caps notional well below the stop ceiling
    liq = _liq(spread=60.0, vol_window=5_000.0, adv=50_000.0, impact=60.0)
    risk = _risk(forecast_vol=0.08, vol_pct=0.4, cov=cov, cov_index=idx, liq=liq)
    return LeverageScenario("thin_liquidity", _signal(), risk, _cfg(spec), _book(cov, idx), 0.0,
                            "thin book + wide spread -> L_liquidity binds or rejects")


def min_lot_too_large() -> LeverageScenario:
    # a futures lot whose minimum notional dwarfs the tiny account -> LotSizer rejects
    spec = _spec(symbol="NIFTY", asset_class="us_equity", lot_step=65.0, mult=1.0,
                 min_notional=0.0)
    idx = ("NIFTY",)
    cov = _healthy_cov(idx, vol=0.15)
    risk = _risk(forecast_vol=0.07, vol_pct=0.4, cov=cov, cov_index=idx,
                 free_margin=500.0, account_equity=500.0)
    sig = _signal("NIFTY", entry=22_000.0, margin=200.0)
    return LeverageScenario("min_lot_too_large", sig, risk, _cfg(spec),
                            _book(cov, idx, equity=500.0, free_margin=500.0), 0.0,
                            "min tradable lot > account risk -> units=0 below_min_unit")


def all_scenarios() -> list[LeverageScenario]:
    """Every scenario flattened (progressive_drawdown expands to its band sweep)."""
    out: list[LeverageScenario] = [
        calm(), vol_spike(), fat_tail(), adverse_gap(), correlation_crisis(),
        regime_cut_full_equity(), no_regime_overlay(), estimator_pathology(),
    ]
    out.extend(progressive_drawdown())
    out.extend([correlated_signals(), margin_explosion(), thin_liquidity(), min_lot_too_large()])
    return out
