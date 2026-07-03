"""Pure policy composition for one paper rebalance (PAPER.md §3-§4). NautilusTrader-free.

Everything here is a thin composition of the validated pillars — nothing is recomputed or
re-modelled: targets are PARITY-PINNED to the engine's own band-ensemble construction
(``tests/paper`` asserts bit-equality against ``run_book`` rows), sizing is
``LeverageEngine.size_book`` (quarter-Kelly), the trade plan is ``ExecutionEngine.on_target``
(no-trade band + trajectories), and order quantities/tactics come from the EXECUTION pillar's
quantization + maker-first primitives against the REAL per-symbol Binance filters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

import numpy as np

from convexity.execution.execution import (
    CostFacts,
    ExecTarget,
    ExecutionDefaults,
    ExecutionEngine,
    RebalancePlan,
    child_order_qty,
    quantize_price,
    tactic_for,
)
from convexity.leverage.engine_api import LeverageEngine
from convexity.leverage.types import CapacityInput
from convexity.panel.candidates import crypto_filters
from convexity.panel.panel import Panel
from convexity.portfolio.portfolio import quantile_weights, realized_vol
from convexity.signal.signals import momentum_signal

BAND: tuple[int, ...] = (12, 14, 16)  # the deployed lookback band (SIGNAL.md §4.3)
VOL_WINDOW = 28  # the engine's vol_window (XSecConfig default; parity with the recorded book)


def ensemble_target_weights(panel: Panel, d: int) -> np.ndarray:
    """This week's band-ensemble dollar-neutral target weights at rebalance row ``d``.

    EXACTLY the engine's construction (equal-weight mean of the per-L quintile/vol-scaled books —
    ``run_ensemble``'s §6.2 average); pinned bit-for-bit against ``run_book`` weight rows by
    ``tests/paper``. Any drift between the paper book and the validated engine is a test failure,
    never a silent divergence (Invariant PAPER-PARITY, PAPER.md §3).
    """
    vol = realized_vol(panel, d, window=VOL_WINDOW)
    ws = []
    for lb in BAND:
        rank = momentum_signal(panel, d, lookback=lb, skip=0)
        ws.append(quantile_weights(rank, quantile="quintile", weighting="volscaled",
                                   gross=1.0, vol=vol))
    return np.mean(np.stack(ws, axis=0), axis=0)


def size_gross(
    hist_net: np.ndarray,
    trial_sharpes_ann: np.ndarray,
    *,
    dvol_row: np.ndarray,
    w_target: np.ndarray,
    symbols: tuple[str, ...],
):
    """Quarter-Kelly gross ``G_t`` via the LEVERAGE pillar (LEVERAGE.md §2, c_t = 0.25).

    ``hist_net`` = the full validated weekly net series up to (excluding) this week — the SAME
    series the recorded backtest produces, extended by the journal's settled paper weeks.
    ``g_t = 1`` (REGIME live overlay is a documented v1 deferral, PAPER.md §9).
    """
    cap_inputs = [
        CapacityInput(symbol=symbols[i], adv=float(dvol_row[i]),
                      w_ref=float(abs(w_target[i])), shortable=True)
        for i in range(len(symbols))
        if w_target[i] != 0.0 and np.isfinite(dvol_row[i]) and dvol_row[i] > 0
    ]
    return LeverageEngine().size_book(
        net_weekly=np.asarray(hist_net, dtype=float),
        trial_sharpes_ann=np.asarray(trial_sharpes_ann, dtype=float),
        capacity_inputs=cap_inputs, c_t=0.25)


def build_plan(
    w_scaled: np.ndarray,
    w_carried: np.ndarray,
    vols_daily: np.ndarray,
    *,
    taker: float = 0.0005,
    half_spread: float = 1.0e-4,
    impact: float = 0.5e-4,
    rho: float = 1.0,
    band_c: float = 0.5,
) -> RebalancePlan:
    """No-trade band + trajectories via the EXECUTION pillar (defaults = the pipeline's RED picks;
    sigma converted per-sqrt-hour for the 2h window, the audited X-4 convention)."""
    n = w_scaled.shape[0]
    med = float(np.nanmedian(vols_daily[np.isfinite(vols_daily)])) if np.isfinite(
        vols_daily).any() else 0.02
    sig = np.where(np.isfinite(vols_daily) & (vols_daily > 0), vols_daily, med) / np.sqrt(24.0)
    linear = taker + half_spread
    facts = CostFacts(eta=np.full(n, impact), gamma=np.zeros(n),
                      theta=np.full(n, 2.0 * linear), sigma=np.asarray(sig, float))
    cfg = ExecutionDefaults(rho=rho, band_c=band_c)
    target = ExecTarget(w_target=w_scaled, shortable=np.ones(n, dtype=bool))
    return ExecutionEngine().on_target(target, w_carried, facts, regime_defensive=False, cfg=cfg)


def fmt_step(value: float, step: float) -> str:
    """Exact decimal string on the venue's step grid (PAPER.md §4).

    ``str(float)`` artifacts like ``63512.600000000006`` FAIL Binance's precision filters; the
    venue needs the value expressed with exactly the step's decimals. Quantize via Decimal against
    the step's own string form (e.g. step 0.1 -> 1dp, 1e-05 -> 5dp).
    """
    d_step = Decimal(str(step)).normalize()
    exp = d_step.as_tuple().exponent
    q = Decimal(repr(value)).quantize(d_step) if exp < 0 else Decimal(int(round(value)))
    return format(q, "f")


@dataclass(frozen=True)
class OrderIntent:
    """One venue-ready child order (PAPER.md §4): quantized qty, maker-first tactic, ref price."""

    symbol: str
    qty: float  # signed, lot-quantized (buy > 0, sell < 0)
    tactic: str  # POST_ONLY_MAKER / LIMIT_IOC / ...
    ref_price: float  # tick-quantized limit reference (last close, side-aware rounding)
    slice_idx: int
    n_slices: int
    reduce_only: bool
    qty_str: str = ""  # venue-exact decimal string of |qty| on the lot grid
    price_str: str = ""  # venue-exact decimal string of ref_price on the tick grid

    def to_dict(self) -> dict:
        return asdict(self)


def order_intents(
    plan: RebalancePlan,
    symbols: tuple[str, ...],
    prices: np.ndarray,
    nav: float,
    *,
    consolidate: bool = True,
) -> list[OrderIntent]:
    """Turn the plan into venue-legal order intents against the REAL per-symbol filters.

    v1 execution shape (PAPER.md §4, a documented simplification): ONE maker order per name for
    the full surviving trade (``consolidate=True``) — the Almgren-Chriss 10-slice schedule is
    journaled in the plan for TCA but not sliced live until the fill-rate data exists to justify
    the added order traffic. The window-end ``finalize`` pass crosses any unfilled remainder
    (LIMIT_IOC -> MARKET), matching ``tactic_for``'s final-slice fallback.
    """
    out: list[OrderIntent] = []
    for i in np.flatnonzero(plan.trade != 0.0):
        sym = symbols[int(i)]
        try:
            f = crypto_filters(sym)
        except KeyError:
            continue  # non-crypto sleeve — never traded by this book
        px = float(prices[int(i)])
        if not (np.isfinite(px) and px > 0):
            continue
        trade = float(plan.trade[int(i)])
        side = 1 if trade > 0 else -1
        if consolidate:
            slices = [(0, 1, trade)]
        else:
            traj = plan.trajectories.get(int(i))
            n_s = len(traj.schedule) if traj is not None else 1
            slices = [(k, n_s, float(traj.schedule[k])) for k in range(n_s)] if traj is not None \
                else [(0, 1, trade)]
        for k, n_s, frac in slices:
            qty = child_order_qty(frac, nav=nav, price=px, lot_size=f.lot_size,
                                  min_qty=f.min_qty, min_notional=f.min_notional)
            if qty == 0.0:
                continue
            tactic = tactic_for("NORMAL", is_final_slice=(k == n_s - 1 and not consolidate))
            if consolidate:
                tactic = tactic_for("NORMAL", is_final_slice=False)  # maker-first; finalize crosses
            ref = quantize_price(px, f.tick_size, side=side)
            out.append(OrderIntent(
                symbol=sym, qty=qty, tactic=tactic, ref_price=ref,
                slice_idx=k, n_slices=n_s if not consolidate else 1,
                reduce_only=False,
                qty_str=fmt_step(abs(qty), f.lot_size),
                price_str=fmt_step(ref, f.tick_size)))
    return out
