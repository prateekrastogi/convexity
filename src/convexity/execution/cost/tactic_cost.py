"""TacticCostModel — assembles per-tactic E[cost] in bps and the optimal passive offset δ* (§5.1/§11).

All terms are SIGNED bps vs the arrival mid. Steps (§11 contract):
  * compute the shared regime driver ``g`` ONCE (§5.1.1) and pass it to p_fill, AdverseSelection, and
    DelayCost so the maker-branch trio is internally consistent (Invariant J);
  * price improvement ``PI(δ) = s − δ·tick_bps`` (the captured half-spread, NOT a flat ``−δ``);
  * maker cost uses the ENTRY split (fill + no-fill·(P_fill_in_budget·cross + P_abort·0), §5.1.2) for
    an entry, and the EXIT form (no-fill -> guaranteed cross, §5.1.3) for a reduce/exit;
  * for funding-aware (perp) instruments add the funding-over-horizon term on the paying side
    (§5.4 / Invariant W);
  * ``δ* = argmin`` over the tick grid of the appropriate maker cost.

The spread, impact, and fee components remain MUTUALLY independent and no component reads another
TACTIC's cost (Invariant B); the maker-branch trio's shared ``g`` is the one scoped, tested coupling
(Invariant J).

Funding-timing scope (§5.4). Funding enters as an EXECUTION-TIMING cost — it prices ONLY the rest
period a clip's chosen tactic actually exposes to a funding stamp, NOT the position's hold carry
(that belongs to LEVERAGE). Three rest horizons, deliberately distinct:
  * a TAKER tactic crosses NOW — it does not rest for the fill horizon ``h``; it spans only the
    immediate crossing latency, so its funding window is the (~0) ``cross_horizon_s`` and a normal
    cross counts ZERO stamps. Charging the full ``h`` here over-counted funding on every immediate
    cross (the §5.4 over-count fix);
  * the maker NO-FILL / continued-rest branch holds the position across the WHOLE resting window
    ``h`` waiting to fill — full-``h`` funding is correct there (this is the branch that can straddle
    a stamp and flip the min() to cross-now);
  * the maker FILL branch rests only until it fills (somewhere in [0, h]); charging it the full ``h``
    is a deliberate, one-signed CONSERVATIVE over-charge — it can only raise ``E[cost_maker]`` (never
    lower it), biasing the min() toward the lower-variance crossing tactic, the SAFE direction (the
    same one-signed conservatism as the receiving-side floor §5.4 and the maker-floor gate §6.3).
    Queue position / time-to-fill is unobservable at 1-min with no order book (§5.5), so we do NOT
    invent a time-to-fill model to shave it; we keep the bound and prove its direction with a test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.cost.adverse_selection import AdverseSelection
from convexity.execution.cost.delay_cost import DelayCost
from convexity.execution.cost.impact import ImpactModel
from convexity.execution.cost.spread_cost import SpreadCost
from convexity.execution.types import ExecTarget, MicroState, TacticCosts, Tactic

__all__ = ["fund_bps", "CROSS_HORIZON_S", "TacticCostModel"]

# §5.4 — a taker order that crosses NOW does not rest for the fill horizon; it spans only the
# immediate crossing latency. A near-zero crossing window so the half-open stamp count is ~0 for a
# normal cross (a stamp at the very instant of the cross is not paid — the position exits before the
# charge, matching the venue convention). RED-ish modelling constant; tiny by construction.
CROSS_HORIZON_S: float = 1e-9


def fund_bps(target: ExecTarget, micro: MicroState, horizon_s: float, cfg: ResolvedConfig) -> float:
    """Expected funding-over-horizon term in bps on the PAYING side (§5.4 / Invariant W).

        fund_bps(h) = side_pays_funding · |predicted_funding_rate| · 10_000 · (#stamps in [now, now+h))

    ``side_pays_funding`` is 1 when the trade's side PAYS over ``h`` (long pays when funding > 0;
    short pays when funding < 0), 0 when it RECEIVES (floored at 0 — receiving funding is a position
    property, not an execution saving; the floor is one-signed conservative, §5.4). Identically 0 for
    non-perp classes (carry there belongs to LEVERAGE). The stamp count uses the HALF-OPEN window
    [now, now+h): a stamp counted only if ``now <= stamp < now+h`` — a clip ending EXACTLY at a stamp
    does NOT pay it.
    """
    if not cfg.funding_aware:
        return 0.0
    rate = micro.predicted_funding_rate
    # Paying side: long pays when rate > 0, short pays when rate < 0.
    pays = (target.side == 1 and rate > 0.0) or (target.side == -1 and rate < 0.0)
    if not pays:
        return 0.0  # receiving side floored at 0 (one-signed conservative, §5.4)
    # Half-open stamp count from the F_t-known next-stamp time and interval.
    secs_to_first = micro.secs_to_next_funding
    interval = max(micro.funding_interval_secs, 1e-9)
    if secs_to_first >= horizon_s:
        n_stamps = 0  # first stamp at or beyond the horizon end -> none inside [now, now+h)
    else:
        n_stamps = 1 + int(math.floor((horizon_s - secs_to_first) / interval - 1e-9 / interval))
        n_stamps = max(n_stamps, 0)
    return float(abs(rate) * 10_000.0 * n_stamps)


@dataclass
class TacticCostModel:
    """Assembles the full per-tactic cost decomposition + δ* (§5.1 / §11)."""

    spread_cost: SpreadCost
    impact: ImpactModel
    adverse: AdverseSelection
    delay: DelayCost
    offset_grid_ticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)

    def costs(
        self, target: ExecTarget, micro: MicroState, cfg: ResolvedConfig, *, as_of_ts: int = 0
    ) -> TacticCosts:
        """Assemble E[cost] in bps for every tactic + δ* over the tick grid (§5.1 / §11)."""
        s = micro.half_spread_bps
        f_t = cfg.instrument.taker_fee_bps
        f_m = self.spread_cost.maker_fee_bps(cfg)
        g = micro.drift_g  # shared regime driver, computed ONCE and passed to the trio (Inv J)
        h = cfg.defaults.fill_horizon_s
        regime_bucket = self._bucket(micro)
        is_exit = target.reduce_only

        # --- crossing tactics: spread + fee (+ impact, ~0 at $1k) + funding-if-cross-spans-stamp.
        # A taker crosses NOW (§5.4): its funding window is the immediate crossing latency, NOT the
        # full resting horizon h — so a normal cross counts ZERO stamps and is not over-charged.
        impact_bps = self.impact.cost_bps(target.target_qty, micro, cfg)
        cross_funding = fund_bps(target, micro, CROSS_HORIZON_S, cfg)
        taker_total = s + f_t + impact_bps + cross_funding
        # MARKETABLE_LIMIT caps slippage at through_ticks of extra; MARKET pays full slip (~same at
        # retail size since impact ~ 0). LIMIT_IOC caps at the touch.
        through_bps = cfg.defaults.marketable_through_ticks * micro.tick_bps
        cost_market = taker_total
        cost_marketable = s + f_t + min(impact_bps, through_bps) + cross_funding
        cost_limit_ioc = s + f_t + impact_bps + cross_funding  # IOC at the touch cap

        # --- maker branch: minimize over the offset grid (the per-step myopic argmin, §5.1).
        best = self._best_maker_offset(target, micro, cfg, g, h, s, f_m, f_t, regime_bucket,
                                       is_exit, as_of_ts)

        by_tactic: dict[Tactic, float] = {
            "POST_ONLY_MAKER": best["cost"],
            "LIMIT_IOC": cost_limit_ioc,
            "MARKETABLE_LIMIT": cost_marketable,
            "MARKET": cost_market,
        }
        spread_map: dict[Tactic, float] = {
            "POST_ONLY_MAKER": -best["pi"],  # captured PI as a (negative) cost on the fill branch
            "LIMIT_IOC": s, "MARKETABLE_LIMIT": s, "MARKET": s,
        }
        impact_map: dict[Tactic, float] = {
            "POST_ONLY_MAKER": 0.0, "LIMIT_IOC": impact_bps,
            "MARKETABLE_LIMIT": min(impact_bps, through_bps), "MARKET": impact_bps,
        }
        adverse_map: dict[Tactic, float] = {
            "POST_ONLY_MAKER": best["adverse"], "LIMIT_IOC": 0.0,
            "MARKETABLE_LIMIT": 0.0, "MARKET": 0.0,
        }
        delay_map: dict[Tactic, float] = {
            "POST_ONLY_MAKER": best["delay"], "LIMIT_IOC": 0.0,
            "MARKETABLE_LIMIT": 0.0, "MARKET": 0.0,
        }
        funding_map: dict[Tactic, float] = {
            "POST_ONLY_MAKER": best["funding"], "LIMIT_IOC": cross_funding,
            "MARKETABLE_LIMIT": cross_funding, "MARKET": cross_funding,
        }

        from convexity.execution.microstate.fill_prob import FillProbability

        lcb = FillProbability.lcb(
            best["p_fill"], micro.fill_prob_ci_halfwidth, micro.fill_prob_calib_error, cfg
        )
        return TacticCosts(
            by_tactic=by_tactic,
            spread_bps=spread_map,
            impact_bps=impact_map,
            adverse_sel_bps=adverse_map,
            delay_bps=delay_map,
            funding_bps=funding_map,
            price_improve_bps=best["pi"],
            p_fill_at_offset=best["p_fill"],
            p_fill_lcb_at_offset=lcb,
            p_abort_if_nofill=best["p_abort"],
            drift_g=g,
            chosen_offset_ticks=best["offset"],
        )

    # ------------------------------------------------------------------ maker assembly

    def _best_maker_offset(
        self, target, micro, cfg, g, h, s, f_m, f_t, regime_bucket, is_exit, as_of_ts
    ) -> dict:
        """δ* = argmin over the tick grid of the appropriate maker cost (§5.1.2/§5.1.3)."""
        best: dict | None = None
        # A maker order rests waiting to fill across the window -> its funding window is the full
        # resting horizon h (§5.4). The NO-FILL/continued-rest branch is exposed to the whole window
        # (this is the branch that can straddle a stamp and flip the min() to cross-now); the FILL
        # branch rests only until it fills (somewhere in [0, h]) but is charged the full h as a
        # ONE-SIGNED CONSERVATIVE over-charge — it can only raise E[cost_maker], biasing toward the
        # lower-variance cross (the SAFE direction). Time-to-fill is unobservable at 1-min with no
        # order book (§5.5), so we keep the bound rather than invent a time-to-fill model.
        rest_funding = fund_bps(target, micro, h, cfg)
        for delta in self.offset_grid_ticks:
            pi = self.spread_cost.price_improvement_bps(s, delta, micro.tick_bps)
            p_fill = float(micro.fill_prob_curve(delta, h, g))
            a = self.adverse.cost_bps(target, micro, delta, regime_bucket, g, cfg, as_of_ts=as_of_ts)
            d = self.delay.cost_bps(target, micro, delta, h, g, cfg)
            fill_branch = -pi + f_m + a
            fund = rest_funding
            cross_cost = d + s + f_t + fund
            if is_exit:
                # EXIT: no-fill -> guaranteed cross (§5.1.3). No abort mass.
                p_abort = 0.0
                cost = p_fill * (fill_branch + fund) + (1.0 - p_fill) * cross_cost
            else:
                # ENTRY: split the no-fill mass into chase-fills-in-budget vs free abort (§5.1.2).
                p_abort = self._p_abort(g, cfg)
                p_in_budget = 1.0 - p_abort
                cost = (
                    p_fill * (fill_branch + fund)
                    + (1.0 - p_fill) * (p_in_budget * cross_cost + p_abort * 0.0)
                )
            cand = {
                "offset": delta, "cost": cost, "pi": pi, "p_fill": p_fill,
                "adverse": a, "delay": d, "funding": fund, "p_abort": p_abort,
            }
            if best is None or cost < best["cost"]:
                best = cand
        assert best is not None
        return best

    def _p_abort(self, g: float, cfg: ResolvedConfig) -> float:
        """Fitted no-fill abort mass; conservative thin-data default biases LOW (§5.1.2).

        Thin/uncertain data sets ``P(abort) -> p_abort_floor`` (a low RED value), pushing the residual
        no-fill mass onto the guaranteed cross — so thin data can only make the maker branch look
        weakly MORE expensive, never phantom-cheap (one-signed conservatism). When ``g`` is high the
        abort mass rises (a runaway is likely to hit worst_entry); the realized split is regressed
        against this in the validation layer (§12).
        """
        floor = cfg.defaults.p_abort_floor
        # Monotone increasing in g, bounded [floor, 0.95]; conservative default = floor.
        return float(min(0.95, max(floor, floor + 0.3 * max(g, 0.0))))

    @staticmethod
    def _bucket(micro: MicroState) -> str:
        """The regime bucket (g-quantile × spread-state) the TEAL fits key on (§5.1.1)."""
        g_q = "calm" if micro.drift_g < 0.5 else ("mid" if micro.drift_g < 1.5 else "runaway")
        sp = "wide" if micro.half_spread_bps >= 6.0 else "tight"
        return f"{g_q}:{sp}"
