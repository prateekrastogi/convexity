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
    quantize_qty,
    tactic_for,
)
from convexity.leverage.engine_api import LeverageEngine
from convexity.leverage.types import CapacityInput
from convexity.panel.candidates import crypto_filters
from convexity.panel.panel import Panel
from convexity.paper.fleetconfig import BAND_C
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
    band_c: float = BAND_C,
) -> RebalancePlan:
    """No-trade band + trajectories via the EXECUTION pillar (defaults = the pipeline's RED picks;
    sigma converted per-sqrt-hour for the 2h window, the audited X-4 convention).

    ``band_c`` defaults to the OFFLINE-FITTED ``fleetconfig.BAND_C`` (research/bandcal/, 2026-08-14).
    It previously defaulted to a hardcoded 0.5 — a number ``ExecutionDefaults`` explicitly refuses
    to invent — which put the band ABOVE the entire target position of every name in the equity and
    crypto books and froze every non-flip trade. See BAND_C's provenance block for the calibration
    and its robustness checks."""
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
    w_target: np.ndarray | None = None,
    filters: dict[str, "object"] | None = None,
) -> list[OrderIntent]:
    """Turn the plan into venue-legal order intents against the REAL per-symbol filters.

    v1 execution shape (PAPER.md §4, a documented simplification): ONE maker order per name for
    the full surviving trade (``consolidate=True``) — the Almgren-Chriss 10-slice schedule is
    journaled in the plan for TCA but not sliced live until the fill-rate data exists to justify
    the added order traffic. The window-end ``finalize`` pass crosses any unfilled remainder
    (LIMIT_IOC -> MARKET), matching ``tactic_for``'s final-slice fallback.

    ``w_target`` (optional): the plan's target weight vector — a FULL CLOSE (target 0, trade
    nonzero) is flagged ``reduce_only`` so the venue exempts it from MIN_NOTIONAL and it can
    never open or flip a position (PAPER.md §4.1, added 2026-07-05).

    ``filters``: the BOOK's symbol->ExchangeFilters quantization surface (PAPER.md §11 —
    ``books.book_filters``); a symbol absent from it is never traded by this book. ``None``
    keeps the original crypto-only lookup (single-book back-compat).
    """
    out: list[OrderIntent] = []
    for i in np.flatnonzero(plan.trade != 0.0):
        sym = symbols[int(i)]
        if filters is not None:
            f = filters.get(sym)
            if f is None:
                continue  # outside the book's quantization surface — never traded by this book
        else:
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
                reduce_only=bool(w_target is not None
                                 and abs(float(w_target[int(i)])) < 1e-12),
                qty_str=fmt_step(abs(qty), f.lot_size),
                price_str=fmt_step(ref, f.tick_size)))
    return out


def renormalize_achievable(
    w: np.ndarray,
    symbols: tuple[str, ...],
    prices: np.ndarray,
    nav: float,
    filters: dict,
    max_iter: int = 6,
) -> tuple[np.ndarray, dict]:
    """Scale the ACHIEVABLE names up to the book's target gross (PAPER.md §12.8).

    A name is achievable when its target notional survives the venue grid: at least one lot, and
    a quantized notional at or above the min-notional floor. Names that cannot be held are
    dropped (they already were); the remainder is scaled by k so that the surviving gross equals
    the ORIGINAL target gross. Iterated to a fixed point because scaling up can itself lift a
    name over the floor, which enlarges the surviving set and lowers k.

    Returns (weights, diagnostics). Weights are unchanged when every name is achievable, so the
    function is a no-op on a large book. Never changes signs or the relative ordering within the
    surviving set — only the common scale."""
    target_gross = float(np.abs(w).sum())
    if target_gross <= 0 or nav <= 0:
        return w, {"applied": False, "reason": "no gross"}
    k = 1.0
    keep: list[int] = []
    for _ in range(max_iter):
        keep = []
        for i, s in enumerate(symbols):
            wi = float(w[i]) * k
            p = float(prices[i]) if i < len(prices) else 0.0
            f = filters.get(s)
            if f is None or not np.isfinite(p) or p <= 0 or abs(wi) < 1e-12:
                continue
            lot = float(getattr(f, "lot_size", 0.0) or 0.0)
            min_qty = float(getattr(f, "min_qty", 0.0) or 0.0)
            min_not = float(getattr(f, "min_notional", 5.0) or 5.0)
            if lot <= 0:
                continue
            q = quantize_qty(abs(wi) * nav / p, lot_size=lot, min_qty=min_qty)
            if q != 0.0 and abs(q) * p >= max(min_not, 5.0):
                keep.append(i)
        ach = float(np.abs(w[keep]).sum()) if keep else 0.0
        if ach <= 0:
            return w, {"applied": False, "reason": "nothing achievable"}
        k_new = target_gross / ach
        if abs(k_new - k) < 1e-6:
            k = k_new
            break
        k = k_new
    out = np.zeros_like(w)
    out[keep] = w[keep] * k
    return out, {"applied": True, "k": round(float(k), 4), "n_kept": len(keep),
                 "n_total": int((np.abs(w) > 1e-12).sum()),
                 "gross_before_trunc": round(target_gross, 4),
                 "gross_after": round(float(np.abs(out).sum()), 4)}


def orphan_close_intents(
    positions: dict[str, float],
    covered: set[str],
    targets_by_symbol: dict[str, float],
    live_filters: dict[str, tuple[float, float, float, float]],
    prices_by_symbol: dict[str, float],
    *,
    domain: frozenset[str] | set[str],
) -> list[OrderIntent]:
    """Reduce-only MARKET closes for venue positions no rebalance intent covers (PAPER.md §4.1).

    THE ORPHAN SWEEP: a held name can rotate OUT of the panel between weeks (universe rotation), or
    its close can be band-suppressed / min-notional-dropped — without this sweep such positions are
    never closed (observed live 2026-07-05: 11 stale positions from the bootstrap book). Any venue
    position whose symbol has no intent this cycle AND whose target weight is zero (absent from the
    panel = zero) is closed reduce-only at market. ``reduceOnly`` exempts Binance's MIN_NOTIONAL, so
    no notional floor is applied here; the lot grid still binds — a residual below min_qty/lot is a
    venue-level dust position the CALLER journals loudly (it cannot be closed by quantity at all).
    A symbol with a nonzero target but no intent is NOT an orphan (its trade sits inside the
    no-trade band by design) and is left alone.

    ``domain`` (REQUIRED, PAPER.md §11): the calling BOOK's symbol domain. The shared venue account
    holds every book's positions — without this scope the crypto book's sweep would market-close
    the commodity book's freshly opened positions every Saturday (and vice versa). A position
    outside ``domain`` belongs to another book and is NEVER this sweep's business.
    """
    out: list[OrderIntent] = []
    for sym, amt in sorted(positions.items()):
        if sym not in domain:
            continue  # another book's position — not ours to touch
        if sym in covered or abs(amt) < 1e-12:
            continue
        if abs(targets_by_symbol.get(sym, 0.0)) > 1e-12:
            continue
        lf = live_filters.get(sym)
        if lf is None:
            continue
        _tick, lot, min_qty, _mn = lf
        qty = quantize_qty(-amt, lot_size=lot, min_qty=min_qty)
        if qty == 0.0:
            continue
        px = float(prices_by_symbol.get(sym, 0.0) or 0.0)
        out.append(OrderIntent(symbol=sym, qty=qty, tactic="MARKET", ref_price=px,
                               slice_idx=0, n_slices=1, reduce_only=True,
                               qty_str=fmt_step(abs(qty), lot), price_str=""))
    return out

