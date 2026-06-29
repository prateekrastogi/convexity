"""LEVERAGE per-ceiling acceptance gates + properties (LEVERAGE §8/§15)."""

import numpy as np
import pytest

from convexity.leverage.allocate.clusters import cluster_by_correlation
from convexity.leverage.allocate.risk_budget import RiskBudgetAllocator
from convexity.leverage.ceilings.liquidity import LiquidityCeiling
from convexity.leverage.ceilings.margin import MarginCeiling
from convexity.leverage.ceilings.portfolio import PortfolioCeiling
from convexity.leverage.ceilings.tail import TailCeiling
from convexity.leverage.combine.lot_sizer import Quantizer, aggregate_trim
from convexity.leverage.config.margin import ScenarioMarginModel
from convexity.leverage.synth import generate as g
from convexity.leverage.types import GapModel, GPDFit, LiquidityState, QuantizedSize


def _cfg():
    return g.calm().cfg


# --------------------------------------------------------------------------- tail: EVT vs fallback

def test_tail_evt_used_when_count_passes():
    cfg = _cfg()
    tc = TailCeiling()
    fit = GPDFit(threshold=0.05, shape=0.2, scale=0.04, n_exceed=80, tail_prob=0.05,
                 ok=True, emp_es=0.10)
    gap = GapModel(0.04, 0.01, 50, 0.08)
    _, diag = tc.ceiling(fit, gap, cfg.defaults.es_budget, cfg.defaults.es_level, cfg)
    assert diag["use_evt"] is True
    assert diag["evt_fallback"] is False


def test_tail_falls_back_when_thin_or_bad_fit():
    cfg = _cfg()
    tc = TailCeiling()
    fit = GPDFit(threshold=0.05, shape=0.2, scale=0.04, n_exceed=3, tail_prob=0.05,
                 ok=False, emp_es=0.10)
    gap = GapModel(0.04, 0.01, 50, 0.08)
    _, diag = tc.ceiling(fit, gap, cfg.defaults.es_budget, cfg.defaults.es_level, cfg)
    assert diag["evt_fallback"] is True


def test_tail_gap_survival_caps_leverage():
    cfg = _cfg()
    tc = TailCeiling()
    fit = GPDFit(0.02, 0.05, 0.01, 80, 0.05, True, 0.04)
    # a huge worst gap -> gap-survival sub-constraint binds
    gap = GapModel(worst_adverse_gap=0.30, typical_gap=0.05, n_gaps=40, class_stress_gap=0.25)
    L, diag = tc.ceiling(fit, gap, cfg.defaults.es_budget, cfg.defaults.es_level, cfg)
    assert diag["tail_binding"] == "gap"
    # L * worst_gap <= ruin threshold
    assert L * 0.30 <= cfg.defaults.gap_ruin_threshold + 1e-9


# --------------------------------------------------------------------------- margin: free-margin buffer

def test_margin_stressed_buffer_enforced():
    cfg = _cfg()
    mc = MarginCeiling()
    mm = ScenarioMarginModel()
    L, diag = mc.ceiling("BTCUSDT", mm, free_margin=1000.0, financing_rate=0.0,
                         horizon=0.02, position_margin=1000.0, entry_price=30000.0, cfg=cfg)
    # usable = free*(1-buffer); per-unit stressed margin > 0 -> finite L
    assert 0.0 < L < np.inf
    assert diag["free_margin_buffer"] == pytest.approx(cfg.defaults.free_margin_buffer)


def test_margin_financing_reduces_ceiling():
    cfg = _cfg()
    mc = MarginCeiling()
    mm = ScenarioMarginModel()
    L0, _ = mc.ceiling("X", mm, 1000.0, 0.0, 1.0, 1000.0, 30000.0, cfg)
    L1, _ = mc.ceiling("X", mm, 1000.0, 0.5, 1.0, 1000.0, 30000.0, cfg)  # high financing
    assert L1 < L0


# --------------------------------------------------------------------------- liquidity: stale + thin

def test_liquidity_stale_halt_forces_zero():
    cfg = _cfg()
    lc = LiquidityCeiling()
    liq = LiquidityState(2.0, 1e7, 1e8, 1e6, 3.0, stale_or_halted=True)
    L, diag = lc.ceiling(liq, 30.0, 1000.0, cfg)
    assert L == 0.0
    assert diag["reason"] == "stale_or_halted"


def test_liquidity_thin_book_lowers_ceiling():
    cfg = _cfg()
    lc = LiquidityCeiling()
    thick = LiquidityState(2.0, 1e8, 1e9, 1e6, 2.0, False)
    thin = LiquidityState(2.0, 1e5, 1e6, 1e4, 2.0, False)
    Lt, _ = lc.ceiling(thick, 30.0, 1000.0, cfg)
    Ln, _ = lc.ceiling(thin, 30.0, 1000.0, cfg)
    assert Ln < Lt


# --------------------------------------------------------------------------- portfolio: netting

def test_portfolio_correlated_cluster_gets_less_than_diversified():
    cfg = _cfg()
    pc = PortfolioCeiling()
    idx = ("A", "B")
    # A heavy, high-vol existing B position so the total-HEAT constraint (the
    # correlation-sensitive one) binds below the per-instrument share cap.
    corr_hi = np.array([[1.0, 0.95], [0.95, 1.0]])
    corr_lo = np.array([[1.0, 0.0], [0.0, 1.0]])
    sig = np.array([0.4, 0.4])
    cov_hi = (sig[:, None] * corr_hi) * sig[None, :]
    cov_lo = (sig[:, None] * corr_lo) * sig[None, :]
    w = {"B": 0.4}  # heavy enough that the correlation-sensitive heat term binds for hi
    L_hi, d_hi = pc.ceiling("A", w, cov_hi, idx, 1.0, 1000.0, 10000.0, cfg)
    L_lo, d_lo = pc.ceiling("A", w, cov_lo, idx, 1.0, 1000.0, 10000.0, cfg)
    # the correlated add's total-heat room is strictly smaller (the cross term penalises it)
    assert d_hi["L_total_heat"] < d_lo["L_total_heat"]
    assert L_hi < L_lo  # correlated cluster -> lower ceiling


def test_portfolio_regime_heat_vs_mctr_reason_code():
    cfg = _cfg()
    pc = PortfolioCeiling()
    idx = ("A", "B")
    sig = np.array([0.2, 0.2])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    cov = (sig[:, None] * corr) * sig[None, :]
    w = {"B": 0.1}
    _, diag1 = pc.ceiling("A", w, cov, idx, 1.0, 1000.0, 10000.0, cfg)
    _, diag25 = pc.ceiling("A", w, cov, idx, 0.25, 1000.0, 10000.0, cfg)
    assert diag1["reason_code"] == "mctr_concentration_cut"
    assert diag25["reason_code"] == "regime_heat_cut"


# --------------------------------------------------------------------------- clusters (NOT HRP)

def test_cluster_by_correlation_groups_correlated():
    idx = ("NIFTY", "BANKNIFTY", "GOLD")
    corr = np.array([[1.0, 0.9, 0.05], [0.9, 1.0, 0.05], [0.05, 0.05, 1.0]])
    sig = np.array([0.18, 0.20, 0.14])
    cov = (sig[:, None] * corr) * sig[None, :]
    clusters = cluster_by_correlation(cov, idx, 0.6)
    # NIFTY+BANKNIFTY together, GOLD alone
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


# --------------------------------------------------------------------------- ERC allocator

def test_erc_allocator_splits_budget_no_mu():
    cfg = _cfg()
    alloc = RiskBudgetAllocator()
    idx = ("NIFTY", "BANKNIFTY", "GOLD")
    corr = np.array([[1.0, 0.9, 0.05], [0.9, 1.0, 0.05], [0.05, 0.05, 1.0]])
    sig = np.array([0.18, 0.20, 0.14])
    cov = (sig[:, None] * corr) * sig[None, :]
    out = alloc.allocate(list(idx), cov, idx, 1.0, cfg)
    assert set(out) == set(idx)
    # GOLD (its own cluster) gets a full cluster share; NIFTY+BANKNIFTY SHARE one cluster's
    # budget, so each is less than GOLD's.
    assert out["GOLD"] > out["NIFTY"]
    assert out["GOLD"] > out["BANKNIFTY"]


# --------------------------------------------------------------------------- aggregate trim

def _qs(units, span, flags=()):
    return QuantizedSize(units=units, notional=units * 10.0, realized_leverage=1.0,
                         realized_loss_pct_margin=0.1, realized_loss_pct_equity=0.01,
                         realized_span_margin=span, free_margin_after=0.0,
                         divergence=0.0, flags=flags, rejected=False)


def test_aggregate_trim_uses_priority_and_fixes_incumbents():
    sized = {"INC": _qs(5, 500.0), "NEW1": _qs(5, 500.0), "NEW2": _qs(5, 500.0)}
    # priority: lower (slack, mctr_eff) trimmed first. NEW2 lowest priority.
    priority = {"NEW1": (0.9, 0.9), "NEW2": (0.1, 0.1)}
    heat = {"INC": 0.05, "NEW1": 0.05, "NEW2": 0.05}
    out = aggregate_trim(sized, priority, cluster_budget=0.12, total_free_margin=1e9,
                         cluster_heat=heat, incumbents=frozenset({"INC"}))
    # incumbent never trimmed
    assert out["INC"].units == 5
    # NEW2 (lowest priority) trimmed before NEW1
    assert out["NEW2"].units <= out["NEW1"].units


def test_aggregate_trim_respects_free_margin():
    sized = {"A": _qs(5, 800.0), "B": _qs(5, 800.0)}
    priority = {"A": (0.9, 0.9), "B": (0.1, 0.1)}
    heat = {"A": 0.0, "B": 0.0}
    out = aggregate_trim(sized, priority, cluster_budget=1e9, total_free_margin=1000.0,
                         cluster_heat=heat)
    total_margin = sum(q.realized_span_margin for q in out.values())
    assert total_margin <= 1000.0 + 1e-6


# --------------------------------------------------------------------------- quantize round-down only

def test_quantize_round_down_never_increases():
    cfg = _cfg()
    q = Quantizer()
    mm = ScenarioMarginModel(min_margin_fraction=0.001)
    res = q.quantize(raw_leverage=1.0, raw_notional=1000.0, instrument=cfg.instrument,
                     position_margin=1000.0, account_equity=1e6, entry_price=30000.0,
                     stop_distance=0.12, capital_cap=0.13, margin_model=mm,
                     free_margin=1e6, existing_margin_used=0.0, cfg=cfg)
    assert res.notional <= 1000.0 + 1e-9  # never rounds UP


# --------------------------------------------------------------------------- §8.6.3 existing-book margin

def test_quantize_existing_book_margin_rejects_when_incumbents_counted():
    """§8.6.3: the post-rounding free-margin buffer is computed AFTER this position is
    added to the EXISTING book. Ignoring incumbents would WRONGLY pass; counting the
    margin already consumed by the open book correctly REJECTS.

    price=1, mult=1, lot_step=50 -> one lot = 50 notional = 50 margin (FlatRatio 1.0).
    raw_notional=100 -> 2 lots (notional 100, span 100). free_margin=1000, buffer 25%.
      - incumbents IGNORED (existing=0): free_after = 1000-100 = 900 >= 0.25*1000=250 -> PASS.
      - incumbents COUNTED (existing=960): avail = 40; 2 lots span 100 -> free_after=-60 ->
        drop to 1 lot span 50 -> free_after = 40-50 = -10 < 0.25*40=10 -> drop -> 0 -> REJECT.
    """
    from convexity.leverage.config.instrument import InstrumentLeverageSpec
    from convexity.leverage.config.margin import FlatRatioMarginModel
    from convexity.leverage.config.resolve import resolve_config

    spec = InstrumentLeverageSpec(
        symbol="X", asset_class="us_equity", lot_step=50.0, contract_multiplier=1.0,
        tick_size=0.01,
    )
    cfg = resolve_config(spec)
    q = Quantizer()
    mm = FlatRatioMarginModel(initial_margin_fraction=1.0)  # margin == notional
    kw = dict(
        raw_leverage=1.0, raw_notional=100.0, instrument=spec, position_margin=1000.0,
        account_equity=1e6, entry_price=1.0, stop_distance=0.001, capital_cap=0.13,
        margin_model=mm, free_margin=1000.0, cfg=cfg,
    )
    # ignoring incumbents -> passes with 2 lots
    res_ignore = q.quantize(existing_margin_used=0.0, **kw)
    assert res_ignore.units == 2
    assert not res_ignore.rejected
    # counting the open book's margin -> rejects (the buffer breaches once incumbents count)
    res_count = q.quantize(existing_margin_used=960.0, **kw)
    assert res_count.units == 0
    assert res_count.rejected
    assert "margin_after_rounding" in res_count.flags


def test_quantize_free_margin_after_nets_incumbents():
    """The reported free_margin_after is net of the open book (§8.6.3): avail - span."""
    from convexity.leverage.config.instrument import InstrumentLeverageSpec
    from convexity.leverage.config.margin import FlatRatioMarginModel
    from convexity.leverage.config.resolve import resolve_config

    spec = InstrumentLeverageSpec(
        symbol="X", asset_class="us_equity", lot_step=1.0, contract_multiplier=1.0,
        tick_size=0.01,
    )
    cfg = resolve_config(spec)
    q = Quantizer()
    mm = FlatRatioMarginModel(initial_margin_fraction=0.10)  # span = 0.10 * notional
    # raw_notional 100 -> 100 units, span = 10. free 1000, existing 500 -> avail 500.
    res = q.quantize(
        raw_leverage=1.0, raw_notional=100.0, instrument=spec, position_margin=1000.0,
        account_equity=1e6, entry_price=1.0, stop_distance=0.001, capital_cap=0.13,
        margin_model=mm, free_margin=1000.0, existing_margin_used=500.0, cfg=cfg,
    )
    assert res.units == 100
    assert res.free_margin_after == pytest.approx(500.0 - 10.0)  # avail - span, NOT 1000 - 10
