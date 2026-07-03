"""The cross-sectional backtest driver: panel -> signals -> dollar-neutral book -> net weekly series.

Ties the layers together for ONE config. The rebalance grid is every ``rebalance_days`` (7 = weekly)
along the panel's day index; at each rebalance day ``d`` we:

  1. compute the signal RANK at ``d`` (strictly from rows <= d — no lookahead),
  2. form dollar-neutral quantile weights (top/bottom quantile, equal or vol-scaled),
  3. settle the week ``[d, d + rebalance_days]``: price P&L + signed funding - turnover cost,
  4. carry the drifted weights forward to the next rebalance.

The first rebalance only starts once ``d`` is large enough for the signal's trailing window (the panel
NaN mask + signal eligibility already guarantee this per-symbol; the engine additionally skips leading
rebalances where NO book can form). The output is the aligned per-week arrays (net, price, funding,
turnover, cost) + the rebalance dates, consumed by ``metrics`` and ``validate``.

REBALANCE ON FRIDAY CLOSE: the brief asks for a Friday-close weekly cadence. We snap the first
rebalance to the first Friday at/after the warmup, then step 7 days; every rebalance day is a Friday
(the daily close of a Friday = ~Fri 23:59 UTC), so signals/holds are Fri-close to Fri-close.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from convexity.signal import signals as sig
from convexity.cost.accounting import (
    WeekResult,
    _asset_week_return,
    _week_funding,
    round_trip_friction,
    settle_week,
)
from convexity.panel.panel import Panel
from convexity.portfolio.portfolio import Quantile, Weighting, quantile_weights, realized_vol

SignalKind = Literal["momentum", "carry", "combined"]


@dataclass(frozen=True)
class XSecConfig:
    """One cross-sectional backtest config (a single DSR 'trial')."""

    signal: SignalKind = "momentum"
    lookback: int = 28  # momentum lookback in days (ignored for pure carry)
    skip: int = 0  # momentum skip-most-recent in days
    carry_lookback: int = 7  # funding-carry trailing mean window in days
    quantile: Quantile = "quintile"
    weighting: Weighting = "equal"
    rebalance_days: int = 7  # weekly
    gross: float = 1.0  # reference gross exposure (100% = 50% long + 50% short)
    vol_window: int = 28  # trailing realized-vol window for vol-scaled weighting

    # friction (fractions of notional per leg); defaults match research.config / cost pillar.
    taker: float = 0.0005
    half_spread: float = 1.0e-4
    impact: float = 0.5e-4

    @property
    def label(self) -> str:
        if self.signal == "carry":
            base = f"carry-{self.carry_lookback}d"
        elif self.signal == "momentum":
            base = f"mom-{self.lookback}d-skip{self.skip}"
        else:
            base = f"combo-{self.lookback}d-skip{self.skip}-c{self.carry_lookback}"
        return f"{base}/{self.quantile}/{self.weighting}"


@dataclass(frozen=True)
class XSecResult:
    """Aligned per-week outputs of one backtest run."""

    dates: pd.DatetimeIndex  # rebalance (entry) date of each week
    net: np.ndarray
    gross_price: np.ndarray
    funding: np.ndarray
    turnover: np.ndarray
    cost: np.ndarray
    universe_size: np.ndarray  # eligible-symbol count at each rebalance

    def __len__(self) -> int:
        return int(self.net.shape[0])


@dataclass(frozen=True)
class XSecBook:
    """Per-week weight/return detail for one run — the raw material for overlays + attribution.

    All arrays are aligned to ``dates`` (one row per hold week). ``weights`` is the TARGET book set at
    each rebalance (n_weeks, n_symbols); ``asset_ret`` is each name's realized simple return over that
    hold week (NaN if not live across both endpoints); ``asset_funding`` is each name's summed funding
    over the week. ``contrib`` = per-name net price+funding contribution to that week's book return (the
    attribution atom). ``mkt_ret`` is the market (BTC) weekly return for beta-hedging.
    """

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    d_index: np.ndarray  # panel row index of each rebalance day
    weights: np.ndarray  # (n_weeks, n_symbols) target weights
    asset_ret: np.ndarray  # (n_weeks, n_symbols) week price return per name
    asset_funding: np.ndarray  # (n_weeks, n_symbols) week summed funding per name
    contrib: np.ndarray  # (n_weeks, n_symbols) per-name price+funding contribution
    mkt_ret: np.ndarray  # (n_weeks,) market weekly return (BTC)

    def __len__(self) -> int:
        return int(self.weights.shape[0])


def _rank_at(panel: Panel, d: int, cfg: XSecConfig) -> np.ndarray:
    if cfg.signal == "momentum":
        return sig.momentum_signal(panel, d, lookback=cfg.lookback, skip=cfg.skip)
    if cfg.signal == "carry":
        return sig.carry_signal(panel, d, lookback=cfg.carry_lookback)
    return sig.combined_signal(
        panel, d, lookback=cfg.lookback, skip=cfg.skip, carry_lookback=cfg.carry_lookback
    )


def _first_friday_index(panel: Panel, warmup: int) -> int:
    """First panel row index at/after ``warmup`` whose date is a Friday (weekday == 4)."""
    wd = panel.dates.weekday.to_numpy()
    for i in range(warmup, panel.n_days):
        if wd[i] == 4:
            return i
    return panel.n_days  # no valid start


def run(panel: Panel, cfg: XSecConfig) -> XSecResult:
    """Run the full cross-sectional backtest for ``cfg`` over ``panel``; return the weekly series."""
    step = cfg.rebalance_days
    # warmup: enough trailing history for the signal, +1 to allow a first drift week for turnover.
    if cfg.signal == "carry":
        warmup = cfg.carry_lookback + step
    else:
        warmup = cfg.lookback + cfg.skip + step
    warmup = max(warmup, cfg.vol_window + 1)

    # SCALAR flat-cost path (research baseline): one round-trip rate for every name.
    # TODO(COST-integration, engine/PORTFOLIO-owned, not COST-owned): the DERIVED per-name cost model
    # of COST.md Section 5.2-5.3 is built and unit-tested in
    # src/convexity/cost/friction/spread_impact.py::SpreadImpactModel.friction_vector(adv, amihud, ...) but is
    # NOT yet consumed here — the production crypto book still runs on flat cfg.taker/half_spread/impact.
    # To wire it: per rebalance day d, compute causal trailing ADV_i / Amihud_i from
    # panel.build_dollar_volume + spread_impact.amihud_ratio over panel.py's window=30, build the
    # SpreadImpactModel via .anchor(median ADV, median Amihud), and pass friction_vector(...,
    # notional_leg=|dw_i|*NAV, is_shortable=<UNIVERSE flag>, in_short_leg=(w_target<0)) as
    # settle_week(..., friction=<that array>) (settle_week already supports the per-symbol array path).
    # BLOCKED on UNIVERSE exposing per-symbol-per-date is_shortable (COST.md Section 7 item 2, N50) so
    # the ~15-unshortable-tail cost is priced (inf) rather than silently finite for universes > N~50.
    # Until then the flat scalar below is the live cost truth and the CPCV parity anchor.
    friction = round_trip_friction(taker=cfg.taker, half_spread=cfg.half_spread, impact=cfg.impact)
    start = _first_friday_index(panel, warmup)

    dates: list[pd.Timestamp] = []
    net: list[float] = []
    gp: list[float] = []
    fnd: list[float] = []
    trn: list[float] = []
    cst: list[float] = []
    usz: list[int] = []

    w_prev = np.zeros(panel.n_symbols)
    d = start
    while d + step < panel.n_days:
        rank = _rank_at(panel, d, cfg)
        vol = realized_vol(panel, d, window=cfg.vol_window) if cfg.weighting == "volscaled" else None
        w_target = quantile_weights(
            rank, quantile=cfg.quantile, weighting=cfg.weighting, gross=cfg.gross, vol=vol
        )
        res: WeekResult
        res, w_end = settle_week(
            panel.close, panel.funding, panel.listed,
            d0=d, d1=d + step, w_target=w_target, w_prev=w_prev, friction=friction,
        )
        dates.append(panel.dates[d])
        net.append(res.net)
        gp.append(res.gross_price)
        fnd.append(res.funding)
        trn.append(res.turnover)
        cst.append(res.cost)
        usz.append(int(np.isfinite(rank).sum()))
        w_prev = w_end
        d += step

    return XSecResult(
        dates=pd.DatetimeIndex(dates),
        net=np.asarray(net),
        gross_price=np.asarray(gp),
        funding=np.asarray(fnd),
        turnover=np.asarray(trn),
        cost=np.asarray(cst),
        universe_size=np.asarray(usz, dtype=int),
    )


def run_book(
    panel: Panel, cfg: XSecConfig, *, market_symbol: str = "BTCUSDT"
) -> tuple[XSecResult, XSecBook]:
    """Like ``run`` but ALSO capture per-week weights / per-name returns / market return (for overlays).

    Re-runs the identical rebalance loop (same warmup, same Friday snap, same settle) and additionally
    records, per hold week: the target weight vector, each name's realized price return and summed
    funding, the per-name price+funding contribution to the book return, and the market (``market_symbol``,
    default BTC) weekly return used to beta-hedge. The aggregate ``XSecResult`` returned is byte-identical
    to ``run(panel, cfg)`` — this is a strict superset, so nothing about the headline series changes.
    """
    step = cfg.rebalance_days
    if cfg.signal == "carry":
        warmup = cfg.carry_lookback + step
    else:
        warmup = cfg.lookback + cfg.skip + step
    warmup = max(warmup, cfg.vol_window + 1)

    # SCALAR flat-cost path (research baseline): one round-trip rate for every name.
    # TODO(COST-integration, engine/PORTFOLIO-owned, not COST-owned): the DERIVED per-name cost model
    # of COST.md Section 5.2-5.3 is built and unit-tested in
    # src/convexity/cost/friction/spread_impact.py::SpreadImpactModel.friction_vector(adv, amihud, ...) but is
    # NOT yet consumed here — the production crypto book still runs on flat cfg.taker/half_spread/impact.
    # To wire it: per rebalance day d, compute causal trailing ADV_i / Amihud_i from
    # panel.build_dollar_volume + spread_impact.amihud_ratio over panel.py's window=30, build the
    # SpreadImpactModel via .anchor(median ADV, median Amihud), and pass friction_vector(...,
    # notional_leg=|dw_i|*NAV, is_shortable=<UNIVERSE flag>, in_short_leg=(w_target<0)) as
    # settle_week(..., friction=<that array>) (settle_week already supports the per-symbol array path).
    # BLOCKED on UNIVERSE exposing per-symbol-per-date is_shortable (COST.md Section 7 item 2, N50) so
    # the ~15-unshortable-tail cost is priced (inf) rather than silently finite for universes > N~50.
    # Until then the flat scalar below is the live cost truth and the CPCV parity anchor.
    friction = round_trip_friction(taker=cfg.taker, half_spread=cfg.half_spread, impact=cfg.impact)
    start = _first_friday_index(panel, warmup)
    mkt_col = panel.symbols.index(market_symbol) if market_symbol in panel.symbols else -1

    dates: list[pd.Timestamp] = []
    net: list[float] = []
    gp: list[float] = []
    fnd: list[float] = []
    trn: list[float] = []
    cst: list[float] = []
    usz: list[int] = []
    d_idx: list[int] = []
    w_rows: list[np.ndarray] = []
    ar_rows: list[np.ndarray] = []
    af_rows: list[np.ndarray] = []
    contrib_rows: list[np.ndarray] = []
    mkt_rows: list[float] = []

    w_prev = np.zeros(panel.n_symbols)
    d = start
    while d + step < panel.n_days:
        rank = _rank_at(panel, d, cfg)
        vol = realized_vol(panel, d, window=cfg.vol_window) if cfg.weighting == "volscaled" else None
        w_target = quantile_weights(
            rank, quantile=cfg.quantile, weighting=cfg.weighting, gross=cfg.gross, vol=vol
        )
        res: WeekResult
        res, w_end = settle_week(
            panel.close, panel.funding, panel.listed,
            d0=d, d1=d + step, w_target=w_target, w_prev=w_prev, friction=friction,
        )
        # per-name detail (same window the settle used): price return + funding + contribution.
        r_wk = _asset_week_return(panel.close, panel.listed, d, d + step)
        f_wk = _week_funding(panel.funding, panel.listed, d, d + step)
        price_c = np.where(np.isfinite(r_wk) & (w_target != 0.0), w_target * r_wk, 0.0)
        fund_c = np.where(w_target != 0.0, -w_target * f_wk, 0.0)
        contrib = price_c + fund_c
        mkt_r = float(r_wk[mkt_col]) if mkt_col >= 0 and np.isfinite(r_wk[mkt_col]) else 0.0

        dates.append(panel.dates[d])
        net.append(res.net)
        gp.append(res.gross_price)
        fnd.append(res.funding)
        trn.append(res.turnover)
        cst.append(res.cost)
        usz.append(int(np.isfinite(rank).sum()))
        d_idx.append(d)
        w_rows.append(w_target.copy())
        ar_rows.append(np.where(np.isfinite(r_wk), r_wk, np.nan))
        af_rows.append(f_wk)
        contrib_rows.append(contrib)
        mkt_rows.append(mkt_r)
        w_prev = w_end
        d += step

    dt = pd.DatetimeIndex(dates)
    result = XSecResult(
        dates=dt,
        net=np.asarray(net),
        gross_price=np.asarray(gp),
        funding=np.asarray(fnd),
        turnover=np.asarray(trn),
        cost=np.asarray(cst),
        universe_size=np.asarray(usz, dtype=int),
    )
    book = XSecBook(
        dates=dt,
        symbols=panel.symbols,
        d_index=np.asarray(d_idx, dtype=int),
        weights=np.asarray(w_rows) if w_rows else np.zeros((0, panel.n_symbols)),
        asset_ret=np.asarray(ar_rows) if ar_rows else np.zeros((0, panel.n_symbols)),
        asset_funding=np.asarray(af_rows) if af_rows else np.zeros((0, panel.n_symbols)),
        contrib=np.asarray(contrib_rows) if contrib_rows else np.zeros((0, panel.n_symbols)),
        mkt_ret=np.asarray(mkt_rows),
    )
    return result, book


# ----------------------------------------------------------------------------------------------
# PORTFOLIO.md §6.2 / §13 — the deployed lookback-band ensemble driver.
# ----------------------------------------------------------------------------------------------

# The deployable momentum lookback band (PORTFOLIO.md §6.2, Invariant B-band): a SPARSE set of
# individually-tested lookbacks, never a dense fill of [12, 16]. L=13 / L=15 appear in neither the
# peak-width refinement grid nor the CPCV/PBO validation grid (§6.1) — no backtest was ever run at
# those lookbacks, so deploying them would interpolate untested configs into the live ensemble.
LOOKBACK_BAND: tuple[int, ...] = (12, 14, 16)


def band_configs(
    lookbacks: tuple[int, ...] = LOOKBACK_BAND,
    *,
    base: XSecConfig | None = None,
) -> list[XSecConfig]:
    """Build the deployed lookback-band ensemble configs (PORTFOLIO.md §6.2, Invariant B-band).

    One momentum ``XSecConfig`` per lookback ``L`` in the band, identical in every other field,
    ready to hand to ``run_ensemble``. Invariant B-band: the deployable set is EXACTLY
    ``{12, 14, 16}`` — sizing or reporting off a single fitted ``L``, or off a widened band that
    contains a lookback with no tested Sharpe figure in the §6.1 evidence grids (e.g. ``L=13`` /
    ``L=15``), is a policy violation of P6, so any other ``lookbacks`` set raises ``ValueError``
    rather than constructing an unevidenced ensemble. Membership is order-insensitive, but a
    DUPLICATED member (e.g. ``(12, 12, 14, 16)``) is rejected too: silently de-duplicating it to
    the canonical band would double a single ``L``'s vote in a later averaging step (§6.2 assumes
    exactly ``|𝓛|`` distinct, equal-weight members), so B-band is checked on the multiset —
    length AND value — not on the de-duplicated set.

    ``base`` supplies every non-lookback field; the default is the canonical deployed config —
    momentum / quintile / vol-scaled / weekly Friday-close (PORTFOLIO.md §4, ``docs/EDGE.md``
    §(b)). Per §11 E7 the band's PBO evidence is momentum-lookback-specific and does not transfer
    to a carry/combined lookback without its own CPCV re-run, so a non-momentum ``base`` raises.
    """
    # B-band on the MULTISET (sorted values, no de-dup): a widened/substituted/dropped set is
    # rejected, and so is a duplicated-canonical set — collapsing (12, 12, 14, 16) to the band
    # would silently over-weight one L in the §6.2 equal-weight average.
    if tuple(sorted(lookbacks)) != LOOKBACK_BAND:
        raise ValueError(
            "Invariant B-band (PORTFOLIO.md §6.2): the deployed lookback band is exactly "
            f"{LOOKBACK_BAND}, got {tuple(lookbacks)}; a lookback without a tested Sharpe figure "
            "in the §6.1 evidence grids (or a duplicated member) may not be deployed"
        )
    if base is None:
        base = XSecConfig(signal="momentum", quantile="quintile", weighting="volscaled")
    if base.signal != "momentum":
        raise ValueError(
            "the lookback-band ensemble is momentum-only (PORTFOLIO.md §6.2; §11 E7: the PBO/band "
            f"evidence does not transfer to signal={base.signal!r} without its own CPCV re-run)"
        )
    return [replace(base, lookback=lb) for lb in LOOKBACK_BAND]


def run_ensemble(panel: Panel, cfgs: list[XSecConfig]) -> XSecResult:
    """Average the dollar-neutral weight vectors of |cfgs| independently-run configs (§6.2).

    Each ``cfg`` in ``cfgs`` (typically one per lookback L in the band 𝓛 = {12, 14, 16}, §6.1)
    is run against the SAME ``panel`` object via ``run_book(panel, cfg)`` — never a separately
    constructed Panel per cfg, even from identical underlying data (Invariant E-align;
    ``run_book``'s ``XSecResult`` is byte-identical to ``run``'s, §11 E8 sanctions either driver).
    The returned ``XSecResult`` is the equal-weight average of the per-cfg weight paths,
    re-settled through ``accounting.settle_week`` at the averaged weight (not an average of
    the per-cfg net returns, which would NOT equal the return of the averaged book whenever
    turnover/cost is nonlinear in the weight path).

    §13 raise-clause, reconciled with §11 E5: the bare §13 docstring states the driver raises
    "if any two cfgs yield XSecResult objects whose dates are not identical". That literal
    check is IMPLEMENTED as E5-front-trim-then-E8-residual-raise below: a pure warmup OFFSET on
    the shared Friday grid (a member with a longer lookback, §11 E5) is a legitimate difference
    that §13's bare wording predates §11's edge-case analysis for, so it is trimmed (not
    raised on);
    only a RESIDUAL post-trim date/row mismatch (a genuine cadence/universe break, §11 E8) is
    the "dates not identical" case §13 means to fail loudly. For the canonical band {12,14,16}
    all three members warm on the identical first Friday (warmup is vol_window-dominated), so
    the trim removes zero rows and §13's literal "dates identical" path holds exactly.

    §11 E5 (warmup TIMING): members with different lookbacks warm up on different first Fridays,
    but — sharing one ``panel`` and one weekly Friday cadence — their rebalance grids are nested
    suffixes of the panel's Friday sequence (each is ``first_friday>=warmup`` then ``+step`` rows).
    The ensemble is therefore computed only from the first rebalance at which ALL members are warm
    (``d_start = max_L d_index^{(L)}[0]``, the max over the band), trimming a few extra days off
    the front relative to any single ``L`` — the small, deterministic cost of ensembling that E5
    names. Each member is front-trimmed to that common start; the averaged/re-settled book runs
    over the shared suffix, and ``w_prev`` starts flat at the first common rebalance (the ensemble
    holds nothing before it is live).

    §11 E8 (genuine misalignment — RAISES): trimming repairs only a warmup OFFSET on a shared
    grid. It cannot repair a cadence/universe mismatch, so after trimming to the common start the
    members' rebalance rows and dates must be IDENTICAL — if they are not (e.g. a member on an
    off-grid, non-weekly cadence), ``run_ensemble`` raises ``ValueError`` rather than mis-adding
    misaligned weeks by position. It also raises if members disagree on the settle parameters
    (cadence / gross / friction — the averaged book is settled once, at one rate) or on ``symbols``
    ordering (E8's corrupted-input regression case).
    """
    if not cfgs:
        raise ValueError("run_ensemble requires at least one config (PORTFOLIO.md §13)")

    head = cfgs[0]
    for cfg in cfgs[1:]:
        if (
            cfg.rebalance_days != head.rebalance_days
            or cfg.gross != head.gross
            or cfg.taker != head.taker
            or cfg.half_spread != head.half_spread
            or cfg.impact != head.impact
        ):
            raise ValueError(
                "ensemble members disagree on settle parameters (rebalance_days/gross/friction): "
                f"{head.label!r} vs {cfg.label!r} — the averaged book settles once, at one rate "
                "(PORTFOLIO.md §6.2/§7)"
            )

    # Invariant E-align: every member runs against the SAME Panel object, passed by reference —
    # run_ensemble constructs no Panel internally.
    members = [run_book(panel, cfg) for cfg in cfgs]

    for cfg, (_res, book) in zip(cfgs, members, strict=True):
        if book.symbols != panel.symbols:
            raise ValueError(
                f"ensemble member {cfg.label!r} settled on a different symbol ordering than the "
                "shared panel (E8: refusing to average misaligned weight vectors)"
            )

    # §11 E5: front-trim each member to the first rebalance at which ALL members are warm —
    # d_start = max over the band of each member's first rebalance row. On a shared panel + shared
    # weekly Friday cadence the per-L grids are nested suffixes, so this is a pure front-trim.
    d_start = max(int(book.d_index[0]) for _res, book in members)
    trimmed: list[tuple[int, XSecResult, XSecBook]] = []
    for cfg, (res, book) in zip(cfgs, members, strict=True):
        off = int(np.searchsorted(book.d_index, d_start, side="left"))
        if off >= book.d_index.shape[0] or int(book.d_index[off]) != d_start:
            raise ValueError(
                f"ensemble member {cfg.label!r} has no rebalance at the common warmup row "
                f"{d_start} (E8: cadence/universe mismatch, not a mere warmup offset — trimming "
                "cannot align members that rebalance on different rows)"
            )
        trimmed.append((off, res, book))

    # §11 E8: after the front-trim the members must rebalance on IDENTICAL rows/dates. Trimming
    # repairs a warmup OFFSET on the shared Friday grid; it cannot repair a genuine cadence
    # mismatch (off-grid rebalances), which surfaces here as a residual row disagreement.
    off0, ref_res, ref_book = trimmed[0]
    ref_rows = ref_book.d_index[off0:]
    ref_dates = ref_res.dates[off0:]
    for cfg, (off, res, book) in zip(cfgs, trimmed, strict=True):
        rows = book.d_index[off:]
        if not np.array_equal(rows, ref_rows):
            raise ValueError(
                f"ensemble members {head.label!r} and {cfg.label!r} rebalance on different panel "
                "rows after trimming to the common warmup (E8: cadence mismatch; refusing to "
                "average misaligned weight paths)"
            )
        if not res.dates[off:].equals(ref_dates):
            raise ValueError(
                f"ensemble members {head.label!r} and {cfg.label!r} yield different rebalance "
                "dates after trimming to the common warmup (E8: failing loudly, never mis-adding)"
            )

    # §6.2: equal-weight average of the per-cfg TARGET weight paths over the common suffix. Each
    # member row is dollar-neutral, so the convex combination is zero-sum at every t (long leg =
    # -short leg); realized gross can sit BELOW cfg.gross wherever members disagree on leg
    # membership (a name long under one L and short under another partially cancels — §6.2's
    # independence property).
    w_avg = np.mean(
        np.stack([book.weights[off:] for off, _res, book in trimmed], axis=0), axis=0
    )

    # Re-settle the AVERAGED path through the same accounting seam (§7): cost is charged on the
    # averaged book's own turnover (<= the members' mean turnover by the triangle inequality),
    # which is exactly why this is NOT the mean of the members' net series.
    # SCALAR flat-cost path (see the TODO(COST-integration) note in run() above — the derived per-name
    # friction_vector of COST.md Section 5.2-5.3 is built+tested but not yet wired into the ensemble
    # averaged-path settle either; blocked on UNIVERSE is_shortable for the N50 gate).
    friction = round_trip_friction(
        taker=head.taker, half_spread=head.half_spread, impact=head.impact
    )
    step = head.rebalance_days
    net: list[float] = []
    gp: list[float] = []
    fnd: list[float] = []
    trn: list[float] = []
    cst: list[float] = []
    # The ensemble holds nothing before its first common rebalance, so w_prev starts flat at the
    # trimmed front (no phantom turnover from a member's pre-trim position).
    w_prev = np.zeros(panel.n_symbols)
    for j in range(w_avg.shape[0]):
        d = int(ref_rows[j])
        res_j: WeekResult
        res_j, w_prev = settle_week(
            panel.close, panel.funding, panel.listed,
            d0=d, d1=d + step, w_target=w_avg[j], w_prev=w_prev, friction=friction,
        )
        net.append(res_j.net)
        gp.append(res_j.gross_price)
        fnd.append(res_j.funding)
        trn.append(res_j.turnover)
        cst.append(res_j.cost)

    # §11 E5: the ensemble is only live where EVERY member is warm/eligible, so over the common
    # suffix the reported eligible count is the conservative element-wise MIN across members
    # (not max/first/last/mean). §11 E5 fixes the aggregation as "only live where every member
    # is warm", so min is the direct reading; pinned by test_ensemble_universe_size_is_the_min
    # _across_members + the trim-path assertion in the E5 front-trim test (would fail on drift).
    usz = np.min(
        np.stack([res.universe_size[off:] for off, res, _book in trimmed], axis=0), axis=0
    )

    return XSecResult(
        dates=ref_dates,
        net=np.asarray(net),
        gross_price=np.asarray(gp),
        funding=np.asarray(fnd),
        turnover=np.asarray(trn),
        cost=np.asarray(cst),
        universe_size=usz.astype(int),
    )
