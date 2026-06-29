"""Adversarial synthetic fill scenarios so the module is testable with no external data (§14).

Each generator returns a fully-specified ``Scenario`` — an :class:`ExecTarget`, a :class:`MicroState`
(with a causal closed-form fill-prob curve), and, where relevant, a forward price path and a
:class:`WorkingState` — driving one of the §14 acceptance cases. Seeded numpy so every fixture is
deterministic and reproducible (no live venue, no real feed).

The ~20 scenarios map to §14:
  * ``calm_wide_spread``               — POST_ONLY should win (high p_fill, A~0).
  * ``runaway_breakout``               — chase reprices, flips to cross, ABORTS at worst_entry (Inv D).
  * ``swept_g``                        — g calm->runaway sweep (Invariant J sign relations).
  * ``short_side_mirror``              — side=-1 on a sign-mirrored path (5.1 side-reflection).
  * ``cold_start``                     — no own passive fills; maker GATED OFF by default (5.5 #2).
  * ``trending_roll_degenerate``       — positive serial autocov; spread ladder must not NaN (§11).
  * ``funding_stamp_imminent``         — paying-side perp clip; funding flips min() to cross (Inv W).
  * ``funding_receiving_side``         — receiving side / non-perp: funding term 0 (Inv W).
  * ``funding_boundary``               — horizon ends EXACTLY at a stamp -> 0 stamps (half-open, W).
  * ``unresolved_fill_injected``       — online-causal fit excludes it (Invariant L).
  * ``spread_blowout_midchase``        — min() flips post->cross mid-ladder (§5.2).
  * ``thin_book_large_size``           — impact non-trivial; ~0 at $1k (Invariant I).
  * ``heavy_partial_fills``            — residual ladder + min-notional dust (§8).
  * ``maker_never_fills``              — maker-floor gate removes POST_ONLY; delayed cross priced (§6.3).
  * ``halt_stale_feed``                — halt gate forces aggressive exit, disables entries (§6.2).
  * ``outage_gap_restart``             — venue stop fired while down; reconcile finds flat (Inv S).
  * ``restart_no_stop``                — reconcile places a stop before anything else (Inv S).
  * ``amend_crash_two_stops``          — TWO venue stops; reconcile collapses to one (Inv S/d).
  * ``partial_stop_fire``              — partial STOP_MARKET fire; reconcile re-covers remaining (S/e).
  * ``n_invalidation_amends``          — distinct accepted stop ids per amend (Invariant Q-STOP).
  * ``duplicate_submits``              — idempotency: filled qty <= target qty (Invariant Q).
  * ``rate_limit_pressure``            — churn suppressed; single marketable preferred (§6.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.core.types import ExecTarget, InstrumentSpec
from convexity.execution.config.resolve import ResolvedConfig, resolve_config
from convexity.execution.microstate.fill_prob import logistic_fill_curve
from convexity.execution.policy.order_issuer import stop_client_order_id
from convexity.execution.types import MicroState, PositionState, VenueStop, WorkingState

__all__ = ["Scenario", "ExecutionSynth", "perp_instrument", "equity_instrument"]


@dataclass(frozen=True)
class Scenario:
    """One adversarial fixture: the decided target + its microstate + optional path / working state.

    ``price_path`` is the forward mid path (the chase walks along it, validation labels read it);
    ``stops`` is the observed venue stop set for reconciliation scenarios; ``position`` the open
    position the protective stop covers.
    """

    name: str
    target: ExecTarget
    micro: MicroState
    cfg: ResolvedConfig
    price_path: np.ndarray
    working: WorkingState
    position: PositionState | None = None
    stops: tuple[VenueStop, ...] = ()
    note: str = ""


def perp_instrument(symbol: str = "BTCUSDT") -> InstrumentSpec:
    """A Binance USDT-M perp AMBER spec (funding-aware, tight tick/lot, MARK-trigger stop)."""
    return InstrumentSpec(
        symbol=symbol,
        asset_class="crypto_perp",
        venue="BINANCE",
        timezone="UTC",
        tick_size=0.1,
        lot_size=0.001,
        min_notional=5.0,
        maker_fee_bps=1.0,
        taker_fee_bps=4.0,
        maker_rebate_bps=0.0,
        funding_interval_ms=8 * 3600 * 1000,
        funding_cap=0.0075,
    )


def equity_instrument(symbol: str = "NVDA") -> InstrumentSpec:
    """A Dukascopy-proxy US-equity AMBER spec (non-perp: funding term is identically 0, Inv W)."""
    return InstrumentSpec(
        symbol=symbol,
        asset_class="us_equity",
        venue="DUKASCOPY",
        timezone="America/New_York",
        tick_size=0.01,
        lot_size=1.0,
        min_notional=1.0,
        maker_fee_bps=0.0,
        taker_fee_bps=2.0,
    )


class ExecutionSynth:
    """Seeded generator of the §14 adversarial fill scenarios."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers

    def _micro(
        self,
        *,
        mid: float,
        tick: float,
        half_spread_bps: float,
        sigma_bps: float,
        drift_g: float,
        funding_aware: bool,
        predicted_funding_rate: float = 0.0,
        secs_to_next_funding: float = float("inf"),
        spread_rung: str = "quoted",
        cold_start: bool = False,
        ci_halfwidth: float = 0.02,
        calib_error: float = 0.0,
        halted: bool = False,
        stale: bool = False,
        rate_limit_ok: bool = True,
        base_fill: float = 0.8,
    ) -> MicroState:
        """Build a MicroState with a closed-form logistic fill curve (microstate/fill_prob.py)."""
        curve = logistic_fill_curve(
            base_rate=base_fill, sigma_bps=sigma_bps, tick_bps=10_000.0 * tick / mid
        )
        return MicroState(
            mid=mid,
            half_spread_bps=half_spread_bps,
            spread_rung=spread_rung,  # type: ignore[arg-type]
            sigma_horizon_bps=sigma_bps,
            volume_proxy=5_000_000.0,
            drift_g=drift_g,
            tick_bps=10_000.0 * tick / mid,
            fill_prob_curve=curve,
            funding_aware=funding_aware,
            predicted_funding_rate=predicted_funding_rate,
            secs_to_next_funding=secs_to_next_funding,
            halted=halted,
            stale=stale,
            rate_limit_ok=rate_limit_ok,
            fill_prob_rung="class_prior" if cold_start else "bucket",
            fill_prob_ci_halfwidth=ci_halfwidth,
            fill_prob_calib_error=calib_error,
            cold_start=cold_start,
        )

    # ------------------------------------------------------------------ scenarios

    def calm_wide_spread(self) -> Scenario:
        """Calm wide-spread market: POST_ONLY should win, maker fill-rate high, A ~ 0 (§14)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=8.0, sigma_bps=3.0,
            drift_g=0.0, funding_aware=True, base_fill=0.92,
        )
        target = ExecTarget(
            intent_id="calm-1", instrument=inst.symbol, side=1, target_qty=0.01,
            arrival_mid=mid, invalidation=mid * 0.99, urgency="PATIENT", reduce_only=False,
        )
        path = mid + np.zeros(20)
        return Scenario("calm_wide_spread", target, micro, cfg, path, WorkingState(worst_entry=mid))

    def runaway_breakout(self) -> Scenario:
        """Price sprints toward invalidation: chase must abort at worst_entry, never beyond (Inv D)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        inv = mid * 1.02  # long; invalidation ABOVE? no — long invalidation is BELOW. Set below.
        inv = mid * 0.98
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=4.0, sigma_bps=12.0,
            drift_g=3.0, funding_aware=True, base_fill=0.3,
        )
        target = ExecTarget(
            intent_id="runaway-1", instrument=inst.symbol, side=1, target_qty=0.01,
            arrival_mid=mid, invalidation=inv, urgency="URGENT", reduce_only=False,
        )
        # Long breakout runs UP (away from the below-invalidation), widening entry->stop.
        path = mid * (1.0 + np.linspace(0.0, 0.01, 20))
        return Scenario(
            "runaway_breakout", target, micro, cfg, path, WorkingState(worst_entry=0.0),
            note="entry; worst_entry recomputed by the chase ladder",
        )

    def swept_g(self) -> list[Scenario]:
        """g calm->runaway sweep: assert p_fill down, A up, D up (Invariant J sign relations)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        out: list[Scenario] = []
        for g in (0.0, 0.5, 1.0, 2.0, 3.0):
            micro = self._micro(
                mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=8.0,
                drift_g=g, funding_aware=True,
            )
            target = ExecTarget(
                intent_id=f"sweep-g{g}", instrument=inst.symbol, side=1, target_qty=0.01,
                arrival_mid=mid, invalidation=mid * 0.98, urgency="NORMAL", reduce_only=False,
            )
            out.append(Scenario(f"swept_g_{g}", target, micro, cfg, mid + np.zeros(10),
                                WorkingState(worst_entry=mid)))
        return out

    def short_side_mirror(self) -> tuple[Scenario, Scenario]:
        """A long on a path and a short on the sign-mirrored path (5.1 side-reflection rule)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=8.0,
            drift_g=1.5, funding_aware=True,
        )
        long_t = ExecTarget("long-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        short_t = ExecTarget("short-1", inst.symbol, -1, -0.01, mid, mid * 1.02, "NORMAL", False)
        up = mid * (1.0 + np.linspace(0.0, 0.005, 10))
        down = mid * (1.0 - np.linspace(0.0, 0.005, 10))
        return (
            Scenario("mirror_long", long_t, micro, cfg, up, WorkingState(worst_entry=mid)),
            Scenario("mirror_short", short_t, micro, cfg, down, WorkingState(worst_entry=mid)),
        )

    def cold_start(self) -> Scenario:
        """Symbol with no own passive fills: maker GATED OFF by default, defaults to crossing (5.5)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=10.0, sigma_bps=4.0, drift_g=0.0,
            funding_aware=True, cold_start=True, ci_halfwidth=cfg.preset.class_prior_ci_halfwidth,
            base_fill=cfg.preset.class_prior_fill_rate,
        )
        target = ExecTarget("cold-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "PATIENT", False)
        return Scenario("cold_start", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def trending_roll_degenerate(self) -> np.ndarray:
        """A trending price series with POSITIVE serial autocovariance (Roll degenerate, §11).

        Price changes follow a strongly positively-autocorrelated AR(1) (φ=0.8) so cov(dp_t, dp_{t-1})
        is POSITIVE and Roll's sqrt(-cov) is undefined — the documented trending-bar degeneracy that
        must fall through to Corwin-Schultz / the class proxy.
        """
        n = 200
        eps = self.rng.normal(0.0, 0.02, n)
        dp = np.empty(n)
        dp[0] = 0.05
        for i in range(1, n):
            dp[i] = 0.05 + 0.8 * (dp[i - 1] - 0.05) + eps[i]  # AR(1), positive serial autocov
        return 100.0 + np.cumsum(dp)

    def funding_stamp_imminent(self) -> Scenario:
        """PATIENT perp clip, paying side, stamp imminent: funding flips min() to cross-now (Inv W)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        # Long pays when funding > 0; stamp 10s away, horizon 30s -> 1 stamp inside [now, now+h).
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=6.0, sigma_bps=3.0, drift_g=0.0,
            funding_aware=True, predicted_funding_rate=0.0050, secs_to_next_funding=10.0,
            base_fill=0.9,
        )
        target = ExecTarget("fund-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "PATIENT", False)
        return Scenario("funding_stamp_imminent", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def funding_receiving_side(self) -> Scenario:
        """Receiving-side perp (short, funding>0): funding term floored at 0 (Inv W one-signed)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=6.0, sigma_bps=3.0, drift_g=0.0,
            funding_aware=True, predicted_funding_rate=0.0050, secs_to_next_funding=10.0,
            base_fill=0.9,
        )
        target = ExecTarget("fund-recv-1", inst.symbol, -1, -0.01, mid, mid * 1.02, "PATIENT", False)
        return Scenario("funding_receiving_side", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def funding_boundary(self) -> Scenario:
        """Horizon ends EXACTLY at a stamp -> 0 stamps counted (half-open [now, now+h), Inv W)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        h = cfg.defaults.fill_horizon_s
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=6.0, sigma_bps=3.0, drift_g=0.0,
            funding_aware=True, predicted_funding_rate=0.0050, secs_to_next_funding=h,  # == h
            base_fill=0.9,
        )
        target = ExecTarget("fund-bnd-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "PATIENT", False)
        return Scenario("funding_boundary", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def thin_book_large_size(self) -> tuple[Scenario, Scenario]:
        """Same thin venue, a $30 clip (impact ~0, Inv I) and a large clip (impact non-trivial)."""
        inst = equity_instrument()
        cfg = resolve_config(inst)
        mid = 100.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=8.0, sigma_bps=10.0, drift_g=0.0,
            funding_aware=False,
        )
        micro_thin = MicroState(**{**micro.__dict__, "volume_proxy": 50_000.0})
        small = ExecTarget("imp-small", inst.symbol, 1, 0.3, mid, mid * 0.98, "NORMAL", False)  # $30
        large = ExecTarget("imp-large", inst.symbol, 1, 300.0, mid, mid * 0.98, "NORMAL", False)
        return (
            Scenario("impact_small", small, micro_thin, cfg, mid + np.zeros(5),
                     WorkingState(worst_entry=mid)),
            Scenario("impact_large", large, micro_thin, cfg, mid + np.zeros(5),
                     WorkingState(worst_entry=mid)),
        )

    def maker_never_fills(self) -> Scenario:
        """Thin-bucket maker: LCB below floor_eff -> POST_ONLY removed; cross priced (§6.3)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=12.0, sigma_bps=3.0, drift_g=0.0,
            funding_aware=True, base_fill=0.6, ci_halfwidth=0.30, calib_error=0.12,
        )
        target = ExecTarget("nofill-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        return Scenario("maker_never_fills", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def halt_stale_feed(self) -> Scenario:
        """Halt / stale feed: halt gate forces aggressive exit, disables new entries (§6.2)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=20.0, sigma_bps=30.0, drift_g=0.0,
            funding_aware=True, halted=True, stale=True,
        )
        target = ExecTarget("halt-exit", inst.symbol, -1, -0.01, mid, mid * 1.0, "IMMEDIATE", True)
        # filled_qty tracks fills toward THIS exit target (starts at 0); the open position the stop
        # covers is separate. An exit not yet started has filled_qty 0.
        return Scenario("halt_stale_feed", target, micro, cfg, mid + np.zeros(5),
                        WorkingState(worst_entry=mid, filled_qty=0.0))

    def protective_exit(self) -> Scenario:
        """An invalidation-breach exit handed down by TREND (reduce_only + IMMEDIATE, gate 1)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=6.0, sigma_bps=10.0, drift_g=0.0,
            funding_aware=True,
        )
        target = ExecTarget("prot-exit", inst.symbol, -1, -0.01, mid, mid, "IMMEDIATE", True)
        return Scenario("protective_exit", target, micro, cfg, mid + np.zeros(5),
                        WorkingState(worst_entry=mid, filled_qty=0.0))

    def rate_limit_pressure(self) -> Scenario:
        """Rate-limit budget low: suppress cancel/replace churn, prefer a single marketable (§6.4)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=6.0, sigma_bps=5.0, drift_g=0.5,
            funding_aware=True, rate_limit_ok=False,
        )
        target = ExecTarget("rl-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        return Scenario("rate_limit_pressure", target, micro, cfg, mid + np.zeros(10),
                        WorkingState(worst_entry=mid))

    def heavy_partial_fills(self) -> Scenario:
        """A target large enough to fill in tranches, leaving a sub-min-notional dust residual (§8)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(
            mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=6.0, drift_g=0.3,
            funding_aware=True,
        )
        target = ExecTarget("partial-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        # most filled; residual = 0.0001 BTC -> $3 < $5 min-notional -> dust (close, not emit).
        working = WorkingState(filled_qty=0.0099, avg_fill_price=mid, worst_entry=mid * 1.005)
        return Scenario("heavy_partial_fills", target, micro, cfg, mid + np.zeros(5), working)

    # ------------------------------------------------------------------ stop / reconcile scenarios

    def restart_no_stop(self) -> Scenario:
        """Restart with an open position and NO venue stop: reconcile places one first (Inv S)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=5.0,
                            drift_g=0.0, funding_aware=True)
        target = ExecTarget("recon-1", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        pos = PositionState(inst.symbol, 1, 0.01, entry_price=mid)
        return Scenario("restart_no_stop", target, micro, cfg, mid + np.zeros(3),
                        WorkingState(filled_qty=0.01, worst_entry=mid, invalidation_generation=0),
                        position=pos, stops=())

    def amend_crash_two_stops(self) -> Scenario:
        """Crash inside amend's place-before-cancel window: TWO stops; collapse to one (Inv S/d)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=5.0,
                            drift_g=0.0, funding_aware=True)
        target = ExecTarget("recon-2", inst.symbol, 1, 0.01, mid, mid * 0.985, "NORMAL", False)
        pos = PositionState(inst.symbol, 1, 0.01, entry_price=mid)
        # gen 1 = current (matches invalidation 0.985); gen 0 = stale (the pre-amend one).
        stale = VenueStop(stop_client_order_id("recon-2", 0), mid * 0.98, 0.01, True)
        cur = VenueStop(stop_client_order_id("recon-2", 1), mid * 0.985, 0.01, True)
        working = WorkingState(filled_qty=0.01, worst_entry=mid, invalidation_generation=1)
        return Scenario("amend_crash_two_stops", target, micro, cfg, mid + np.zeros(3),
                        working, position=pos, stops=(stale, cur))

    def partial_stop_fire(self) -> Scenario:
        """A partial STOP_MARKET fire in a thin gap: re-cover the exact remaining qty (Inv S/e)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=5.0,
                            drift_g=0.0, funding_aware=True)
        target = ExecTarget("recon-3", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        # position reduced from 0.01 to 0.006 by a partial fire; stale stop still rests at full qty.
        pos = PositionState(inst.symbol, 1, 0.006, entry_price=mid)
        stale = VenueStop(stop_client_order_id("recon-3", 0), mid * 0.98, 0.01, True)
        working = WorkingState(filled_qty=0.006, worst_entry=mid, invalidation_generation=0)
        return Scenario("partial_stop_fire", target, micro, cfg, mid + np.zeros(3),
                        working, position=pos, stops=(stale,))

    def outage_gap_restart(self) -> Scenario:
        """Outage through an adverse gap: stop fired while down; reconcile finds flat (Inv S)."""
        inst = perp_instrument()
        cfg = resolve_config(inst)
        mid = 30_000.0
        micro = self._micro(mid=mid, tick=inst.tick_size, half_spread_bps=5.0, sigma_bps=5.0,
                            drift_g=0.0, funding_aware=True)
        target = ExecTarget("recon-4", inst.symbol, 1, 0.01, mid, mid * 0.98, "NORMAL", False)
        # position flat (stop fired while down); an orphan stop may still be reported, or none.
        pos = PositionState(inst.symbol, 1, 0.0, entry_price=mid)
        orphan = VenueStop(stop_client_order_id("recon-4", 0), mid * 0.98, 0.01, True)
        working = WorkingState(filled_qty=0.0, worst_entry=mid, invalidation_generation=0)
        return Scenario("outage_gap_restart", target, micro, cfg, mid + np.zeros(3),
                        working, position=pos, stops=(orphan,))

    def all_scenarios(self) -> list[Scenario]:
        """Flatten every single-Scenario generator (the multi-return ones are tested directly)."""
        out = [
            self.calm_wide_spread(), self.runaway_breakout(), self.cold_start(),
            self.funding_stamp_imminent(), self.funding_receiving_side(), self.funding_boundary(),
            self.maker_never_fills(), self.halt_stale_feed(), self.protective_exit(),
            self.rate_limit_pressure(), self.heavy_partial_fills(), self.restart_no_stop(),
            self.amend_crash_two_stops(), self.partial_stop_fire(), self.outage_gap_restart(),
        ]
        out.extend(self.swept_g())
        long_s, short_s = self.short_side_mirror()
        out.extend([long_s, short_s])
        small, large = self.thin_book_large_size()
        out.extend([small, large])
        return out
