"""Signed real-8h-stamp funding accrual (COST.md Section 6).

A thin typed provider around the panel's daily funding array + ``research/xsec/accounting.py``'s
listed-masked per-stamp accrual. It does NOT re-implement the settlement arithmetic (Invariant N):
``settle_week`` remains the ONE place ``funding_t`` is folded into ``net_t``. This module exists so
the funding SIGN convention and the per-real-stamp (never weekly-average) accrual are a named,
independently-testable provider surface (COST.md Section 3), matching ``SpreadImpactModel`` on the
cost side.

SIGN (Invariant P / P6, COST.md Section 6.1). For a signed weight ``w_i`` (positive = long) and a
realized funding rate ``f_i``, the cash flow to the position is ``-w_i * f_i``: a long paying
positive funding bleeds carry (``w>0, f>0 -> <0``, PAYS); a short in the same regime is CREDITED
(``w<0, f>0 -> >0``). This is exactly ``accounting.settle_week``'s ``funding_cf = sum(-w * f_acc)``.

REAL STAMPS (Invariant F-stamp, COST.md Section 6.2). Funding accrues on Binance's actual 8h stamps
(00:00 / 08:00 / 16:00 UTC), each daily panel entry already the sum of that day's ~3 stamps.
``accrued_over(...)`` sums the daily array over the holding interval ``(d0, d1]`` masked to days the
name is ``listed`` — a name delisting mid-week stops accruing at its last live day, never across the
gap. This is NOT a flat weekly-average rate: a +/- sign flip inside the week is priced on its true
path.

MINOR (Invariant MINOR / SEP, COST.md Section 6.3-6.4). Funding is REPORTED, never a weight driver:
``research/xsec/portfolio.py`` takes no funding input, so the momentum book accrues incidental signed
funding on whatever weights the signal produces without importing the DSR-failing carry signal's
structural drawdown. The funding-ablation Sharpe SYSTEM's report consumes lives in
``research/xsec/cost/friction_stress.py::net_sharpe_ex_funding`` (COST.md Section 6.4), not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.xsec.accounting import _week_funding


@dataclass(frozen=True)
class FundingModel:
    """Signed per-leg funding accrual over a weekly hold, delegating to accounting.py (COST.md 6)."""

    def accrued_over(
        self,
        funding: np.ndarray,
        listed: np.ndarray,
        *,
        d0: int,
        d1: int,
    ) -> np.ndarray:
        """Per-symbol total funding RATE accrued over ``(d0, d1]`` (COST.md Section 6.2, F-stamp).

        Delegates verbatim to ``accounting._week_funding`` (Invariant N): the sum of the daily funding
        array over days ``d0+1 .. d1`` inclusive, masked to days the name is ``listed`` (0 where not
        live, so a mid-week delist stops accrual at the last live day). NaN daily entries (real gaps
        in the lake) contribute 0. This is a RATE, not yet a cash flow — apply ``cash_flow`` to sign
        it against the position.
        """
        return _week_funding(funding, listed, d0, d1)

    def cash_flow(self, w: np.ndarray, accrued_rate: np.ndarray) -> np.ndarray:
        """Signed per-name funding cash flow ``-w_i * f_acc_i`` (COST.md Section 6.1, Invariant P).

        Long (``w>0``) + positive funding -> negative cash flow (PAYS); short (``w<0``) + positive
        funding -> positive cash flow (CREDITED). Names with zero weight contribute 0. Matches
        ``accounting.settle_week``'s ``funding_cf`` term element-wise (Invariant N: the sign
        convention is defined ONCE, in accounting.py; this method mirrors it, never a second truth).
        """
        w = np.asarray(w, dtype=float)
        f = np.asarray(accrued_rate, dtype=float)
        if w.shape != f.shape:
            raise ValueError(f"w {w.shape} and accrued_rate {f.shape} must be aligned")
        return np.where(w != 0.0, -w * f, 0.0)

    def weekly_cash_flow(
        self,
        funding: np.ndarray,
        listed: np.ndarray,
        w: np.ndarray,
        *,
        d0: int,
        d1: int,
    ) -> float:
        """Total signed book funding over the hold week: ``sum_i -w_i * f_acc_i`` (COST.md 6.1-6.2).

        The scalar ``funding_t`` term ``settle_week`` produces. Composes ``accrued_over`` (F-stamp
        per-real-stamp masked accrual) with ``cash_flow`` (Invariant P signing) — a single number a
        report or a diagnostic can consume without re-running the full weekly settlement.
        """
        acc = self.accrued_over(funding, listed, d0=d0, d1=d1)
        return float(np.sum(self.cash_flow(w, acc)))
