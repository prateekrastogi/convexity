"""TrendLifeLabeler — future labels for evaluation ONLY (TREND §12, Inv G firewall).

Firewalled (Inv G): live modules (estimate/ + policy/) must NEVER import this. It builds the future
labels the §13 metrics need from a finished decision stream: realized ``tau_live``, death definition
/ censoring reason, ``cap_fraction_used`` + cap-binding, net PnL from ``action_idx``, MFE/MAE,
MFE-capture, round-trip giveback, time-to-exit, gap-through-stop, and the §1.5c net-convexity
quantities (``E[Pi]``, ``p_win``, ``avg_win``, ``avg_loss``, ``premium_bleed``).

The §1.5c / §1.5a payoff identity, in side-adjusted log-PnL units:

    Pi = X_{tau_live} = M - G_exit           (M = terminal favorable HWM, G_exit = giveback at exit)

A position still alive at the cap is labeled ``censored`` (NOT trend death) unless the live policy
emitted a ``time_cap`` action (§12).

COST WIRING (§6.1 / Inv M): every net-of-cost quantity here — the per-trade ``net_pi`` and the §1.5c
``net_convexity`` expectancy — sources its friction from COST.md via a COST-sourced ``cost_log``
(round-trip friction in log-PnL units, from :class:`~convexity.trend.validate.cost_source.TrendCostSource`
wrapping ``convexity.cost.cost_api``). When ``cost_log`` is supplied (the real wiring, and the only
path used by promotion), the local ``CostModel`` stub is NOT consulted. The stub
(``_round_trip_cost_log``) is the §6.1 validation-only fallback, used ONLY when COST estimators are
not wired in (a standalone, clearly-labeled ``validate/labels.py`` unit test) — never for a promotion
or net-EV gate.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.core.types import Bars, CostModel, Position
from convexity.trend.types import ActionDecision


class TrendLifeLabeler:
    """Future-label builder, evaluation-only (§12, Inv G)."""

    def label(
        self,
        bars: Bars,
        position: Position,
        decisions: list[ActionDecision],
        cost: CostModel,
        horizon_caps: Mapping[str, float],
        clocks: tuple[str, ...],
        cost_log: float | None = None,
    ) -> list[dict]:
        """One label dict per decision (§12). Future bars used here ONLY (Inv G).

        ``cost_log`` is the COST.md-sourced round-trip friction in log-PnL units (§6.1 / Inv M). When
        supplied it is the authoritative net charge subtracted from each gross ``Pi``; the local
        ``cost`` stub is consulted ONLY as the §6.1 validation-only fallback when ``cost_log is None``.
        """
        side = float(position.side)
        entry = position.entry_price
        n = len(bars)
        out: list[dict] = []
        exit_decision = next((d for d in decisions if d.action == "exit"), decisions[-1])
        exit_idx = exit_decision.action_idx

        # side-adjusted log path X_t and its running max (causal up to t, but used as a label here).
        with np.errstate(divide="ignore", invalid="ignore"):
            x = side * np.log(np.where(bars.c > 0, bars.c, np.nan) / entry)
        x = np.nan_to_num(x, nan=0.0)

        for d in decisions:
            i = d.idx
            # MFE/MAE AFTER this decision, up to the realized exit (future-looking label, Inv G).
            j0, j1 = i, min(exit_idx, n - 1)
            window = x[j0 : j1 + 1] if j1 >= j0 else x[j0 : j0 + 1]
            mfe = float(np.max(window)) if window.size else 0.0
            mae = float(np.min(window)) if window.size else 0.0
            x_exit = float(x[min(exit_idx, n - 1)])
            m_terminal = float(np.max(x[position.entry_idx : min(exit_idx, n - 1) + 1]))
            g_exit = max(m_terminal - x_exit, 0.0)
            pi = x_exit  # round-trip log payoff = M - G_exit = X_exit (§1.5a)
            mfe_capture = (m_terminal - g_exit) / m_terminal if m_terminal > 1e-12 else 0.0
            giveback_ratio = g_exit / m_terminal if m_terminal > 1e-12 else 0.0

            # COST-sourced friction (§6.1 / Inv M) when wired; the local stub is the validation-only
            # fallback used ONLY when COST estimators are absent (cost_log is None).
            rt_cost_log = cost_log if cost_log is not None else self._round_trip_cost_log(cost)
            net_pi = pi - rt_cost_log

            out.append(
                {
                    "idx": i,
                    "action": d.action,
                    "stop_cause": d.stop_cause,
                    "tau_live_idx": exit_idx,
                    "death_definition": self._death_def(exit_decision.stop_cause),
                    "censored": exit_decision.stop_cause in ("time_cap", "censored", "data_censored"),
                    "cap_fraction_used": d.horizon.cap_fraction_used,
                    "cap_binding": d.horizon.cap_binding_warning,
                    "mfe": mfe,
                    "mae": mae,
                    "m_terminal": m_terminal,
                    "g_exit": g_exit,
                    "pi": pi,
                    "net_pi": net_pi,
                    "mfe_capture": mfe_capture,
                    "giveback_ratio": giveback_ratio,
                    "time_to_exit_bars": exit_idx - position.entry_idx,
                }
            )
        return out

    def net_convexity(self, labels: list[dict], cost_per_loss_r: float = 0.0) -> dict:
        """The §1.5c net-of-cost expectancy: E[Pi] = p_win*avg_win - p_loss*avg_loss, with
        premium_bleed reported alongside (§13). Aggregates the per-trade ``net_pi`` of the terminal
        labels (which already net the COST-sourced ``cost_log``, §6.1). ``cost_per_loss_r`` is the
        COST.md-sourced round-trip charge per failed cut, in R0 units, from
        :meth:`~convexity.trend.validate.cost_source.TrendCostSource.cost_per_loss_r` (Inv M) — NOT
        the local stub."""
        # one terminal label per trade (the labels share an exit; use the exit row).
        terminal = [labels[-1]] if labels else []
        pis = np.array([t["net_pi"] for t in terminal])
        if pis.size == 0:
            return self._convex_summary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wins = pis[pis > 0]
        losses = pis[pis <= 0]
        p_win = wins.size / pis.size
        p_loss = losses.size / pis.size
        avg_win = float(np.mean(wins)) if wins.size else 0.0
        avg_loss = float(np.mean(-losses)) if losses.size else 0.0
        premium_bleed = float(p_loss * (avg_loss + cost_per_loss_r))
        e_pi = float(p_win * avg_win - p_loss * avg_loss)
        return self._convex_summary(e_pi, p_win, avg_win, p_loss, avg_loss, premium_bleed)

    def net_convexity_population(
        self, terminal_labels: list[dict], cost_per_loss_r: float = 0.0
    ) -> dict:
        """The §1.5c expectancy over a POPULATION of trades (one terminal label per trade). This is
        the proper population estimand: ``E[Pi] = p_win*avg_win - p_loss*avg_loss`` with the
        COST-sourced ``cost_per_loss_r`` charged on EVERY losing cut (each loss's premium bleed). A
        regime can be convex GROSS (positive skew/MFE capture) yet net-negative once the bleed across
        all the small left-tail cuts is charged — the §1.5c FAIL the promotion gate must catch
        (Inv M). Distinct from :meth:`net_convexity`, which scores a single trade's terminal row."""
        if not terminal_labels:
            return self._convex_summary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        gross = np.array([t["net_pi"] for t in terminal_labels])
        is_loss = gross <= 0.0
        # the COST round-trip friction (in the same units as Pi via cost_per_loss_r) is charged on
        # every cut; on the loss side it deepens avg_loss, which is the premium bleed.
        net = gross - is_loss.astype(float) * cost_per_loss_r
        wins = net[net > 0]
        losses = net[net <= 0]
        n = net.size
        p_win = wins.size / n
        p_loss = losses.size / n
        avg_win = float(np.mean(wins)) if wins.size else 0.0
        avg_loss = float(np.mean(-losses)) if losses.size else 0.0
        premium_bleed = float(p_loss * avg_loss)  # the cut + COST friction across undeveloped cuts
        e_pi = float(p_win * avg_win - p_loss * avg_loss)
        return self._convex_summary(e_pi, p_win, avg_win, p_loss, avg_loss, premium_bleed)

    @staticmethod
    def _convex_summary(
        e_pi: float, p_win: float, avg_win: float, p_loss: float, avg_loss: float,
        premium_bleed: float,
    ) -> dict:
        """Assemble the §13 net-convexity report dict (shared by the single-trade + population gates)."""
        return {
            "E_Pi": e_pi,
            "p_win": float(p_win),
            "avg_win": avg_win,
            "p_loss": float(p_loss),
            "avg_loss": avg_loss,
            "premium_bleed": premium_bleed,
            "net_convex_pass": e_pi > 0.0,
        }

    @staticmethod
    def _round_trip_cost_log(cost: CostModel) -> float:
        """§6.1 VALIDATION-ONLY round-trip cost in log-PnL units from the local ``CostModel`` stub.
        Used ONLY when no COST-sourced ``cost_log`` is wired in (a standalone labels unit test); it is
        FORBIDDEN for the §1.5c net-EV gate and the §13 promotion gate (Inv M)."""
        bps = cost.spread_bps + cost.slippage_bps + cost.fees_bps
        return bps / 10_000.0

    @staticmethod
    def _death_def(cause) -> str:
        """Map the stop cause to the §13a death definition label."""
        return {
            "trail_breach": "D1",
            "seq_alarm": "D2",
            "jump_against": "D5",
            "time_cap": "censored",
            "censored": "censored",
            "data_censored": "censored",
        }.get(cause, "all_cause")
