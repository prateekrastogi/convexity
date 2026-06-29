"""MODE 3 — LIVE (REAL MONEY): the same ``TradingNode`` wiring as paper, ``BinanceEnvironment.LIVE``.

REAL wiring (not a stub): reuses :func:`convexity.run.paper.build_trading_node` with ``demo=False``
so the venue is ``BinanceEnvironment.LIVE`` (real money) while everything else — the bundled Binance
USDT-M adapter, the SAME :class:`~convexity.strategy.convexity_strategy._ConvexityStrategy`, the
reduce-only exec config — is identical to paper. The engine owns backtest↔paper↔live parity; only the
venue environment + the live guardrails differ.

CREDS-REQUIRED + HARD GATES — NOT RUN IN CI. Connecting needs LIVE Binance API keys. Run only AFTER a
clean paper run (the deployment hard gate). :func:`run_live` enforces a small set of pre-flight
guardrails (an explicit ``i_understand_real_money`` acknowledgement + presence of credentials) before
it will start the live event loop; the standing venue-side risk control is the reduce-only protective
stop that rests the instant a position opens (EXECUTION P6) and the OPS kill machine's HALT→flatten
(OPS §9.1b). Notional/leverage caps belong in ``conf/live.yaml`` + the LEVERAGE ceilings, applied
inside the composer — not bypassable here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from convexity.run.paper import build_node_from_plan, build_trading_node

if TYPE_CHECKING:  # type-only; never imported in the pillar venv (Invariant N).
    from convexity.run.session import SessionPlan
    from convexity.strategy.convexity_strategy import StrategyDeps


def build_live_node(
    *,
    instrument_ids: Sequence[str],
    bar_types: Sequence[str],
    deps: "StrategyDeps",
    api_key: str | None = None,
    api_secret: str | None = None,
    trader_id: str = "CONVEXITY-LIVE-001",
) -> Any:
    """Build the LIVE (real-money) Binance USDT-M ``TradingNode`` hosting the ``_ConvexityStrategy``.

    Identical to the paper node except ``demo=False`` ⇒ ``BinanceEnvironment.LIVE``. Returns the
    constructed, NOT-yet-started node — :func:`run_live` adds the pre-flight gate before connecting.
    """
    return build_trading_node(
        instrument_ids=instrument_ids, bar_types=bar_types, deps=deps,
        demo=False, api_key=api_key, api_secret=api_secret, trader_id=trader_id,
    )


def run_live(
    plan: "SessionPlan", *, i_understand_real_money: bool = False
) -> None:  # pragma: no cover - real-money, creds-required, NOT run in CI
    """Build the LIVE node from a shared :class:`SessionPlan` and run its event loop (BLOCKS, REAL
    MONEY). The config-only live entrypoint: ``build_session(RunConfig(mode="live", ...))`` then
    ``run_live(plan, i_understand_real_money=True)``. Hard pre-flight gate: requires
    ``i_understand_real_money=True`` and live credentials present before it will connect."""
    if not i_understand_real_money:
        raise RuntimeError(
            "run_live is REAL MONEY: pass i_understand_real_money=True after a clean paper run "
            "(deployment hard gate). The reduce-only protective stop + OPS HALT→flatten are the "
            "standing venue-side risk controls."
        )
    node = build_node_from_plan(plan, demo=False)
    try:
        node.run()
    finally:
        node.dispose()


__all__ = ["build_live_node", "run_live"]
