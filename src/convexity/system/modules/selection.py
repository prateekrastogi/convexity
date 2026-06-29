"""SelectionAdapter — maps the Carver Ch.25 ``DynOptResult`` → ``ProposedPortfolio`` (ARCH §5.10).

The existing :mod:`convexity.selection.dynamic_optimization` exposes
``optimise_portfolio(...) -> DynOptResult``; SYSTEM needs a ``ProposedPortfolio`` (SYSTEM §11). This
adapter is the mapping resolved in ARCHITECTURE §5.10:

  * ``keep``/``eligible``  — the instruments with a NON-ZERO buffered (optimised) position: the
                             dynamically-optimised candidate list. ``keep`` = currently-wanted
                             capital; ``eligible`` = eligible for NEW entries this bar (same set,
                             since the optimiser already filtered the universe).
  * ``risk_weights``       — the optimiser's OPTIMISED weights (target RISK weights, NOT sizes; the
                             optimiser carries no edge — it minimises tracking error, Invariant A).
  * ``rank``               — a pure RISK measure: the tracking-error / correlation-distance
                             *contribution* per instrument. NEVER edge, NEVER a size, NEVER a
                             budget-priority input (Invariant A). Higher rank = better diversifier
                             (lower marginal tracking-error contribution), so a LOWER numeric value
                             of the raw contribution maps to a HIGHER rank; we expose the negated
                             contribution so "higher rank wins" reads naturally in the rotation test.

This adapter reimplements NO selection math (Invariant P1): it calls
``optimise_portfolio`` and reshapes its frozen result. The 19 selection tests are untouched.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from convexity.selection.dynamic_optimization import (
    DynOptResult,
    Instrument,
    covariance_matrix,
    optimise_portfolio,
)
from convexity.system.types import ProposedPortfolio


def _risk_rank(result: DynOptResult, instruments: Sequence[Instrument], corr) -> dict[str, float]:
    """Diversification/quality RANK as a RISK measure (Invariant A — never edge).

    Each instrument's marginal contribution to the optimised portfolio's tracking-error risk:
    ``MCTR_i = w_i (Σ w)_i / sqrt(wᵀΣw)`` on the optimised weight vector. A SMALL contribution means
    the instrument diversifies well, so we rank by the NEGATED contribution (higher rank = better
    diversifier). This is the standard MCTR decomposition (SYSTEM §17), used here purely to order
    candidates for the rotation test — never to size and never as a budget-priority input.
    """
    sigma = np.array([ins.sigma for ins in instruments], dtype=float)
    cov = covariance_matrix(sigma, corr)
    w = np.array(
        [result.optimised_positions[ins.ticker] * ins.weight_per_contract for ins in instruments],
        dtype=float,
    )
    port_var = float(w @ cov @ w)
    port_sd = float(np.sqrt(port_var)) if port_var > 0.0 else 0.0
    if port_sd <= 0.0:
        return {ins.ticker: 0.0 for ins in instruments}
    sigma_w = cov @ w
    mctr = (w * sigma_w) / port_sd
    return {ins.ticker: -float(m) for ins, m in zip(instruments, mctr)}


class SelectionAdapter:
    """Thin wrapper over the Carver Ch.25 dynamic optimiser (SYSTEM §12 SelectionAdapter contract).

    ``select`` returns a RISK-weighted :class:`ProposedPortfolio` + a diversification/quality rank
    that NEVER becomes a size and NEVER enters the budget-priority firewall (Invariant A). Cross-asset
    via AMBER metadata only.
    """

    def select(self, instruments: Sequence[Instrument], corr, capital: float, **kwargs) -> ProposedPortfolio:
        """Run dynamic optimisation and reshape the result into a ``ProposedPortfolio`` (ARCH §5.10)."""
        result = optimise_portfolio(list(instruments), corr, capital, **kwargs)
        return self.from_dynopt(result, instruments, corr)

    @staticmethod
    def from_dynopt(
        result: DynOptResult, instruments: Sequence[Instrument], corr
    ) -> ProposedPortfolio:
        """Map a precomputed ``DynOptResult`` (ARCH §5.10). Separated so tests can script the result."""
        candidates = tuple(result.candidates)  # non-zero buffered positions = the candidate list
        risk_weights = {
            t: float(result.optimal_weights.get(t, 0.0)) for t in result.tickers
        }
        rank = _risk_rank(result, instruments, corr)
        return ProposedPortfolio(
            keep=candidates,
            eligible=candidates,
            risk_weights=risk_weights,
            rank=rank,
            reason_codes=("carver_dynopt",),
        )
