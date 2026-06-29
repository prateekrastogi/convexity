"""MODE 2 — PAPER: a real Nautilus ``TradingNode`` on Binance USDT-M DEMO (ARCHITECTURE §1/§4).

REAL wiring (not a stub): a ``TradingNode`` with the bundled Binance USDT-M futures adapter
(``BinanceLive{Data,Exec}ClientFactory``), ``BinanceEnvironment.DEMO`` on BOTH client configs (the
Binance demo / paper venue; TESTNET is the legacy fallback), hosting the SAME
:class:`~convexity.strategy.convexity_strategy._ConvexityStrategy` that the backtest runs unchanged —
the engine owns backtest↔live parity, so only the venue wiring differs.

CREDS-REQUIRED — NOT RUN IN CI. Connecting needs live Binance demo API keys; this module builds and
returns a fully-wired, IMPORTABLE node (the construction is what the deployment phase asserts) but
does not connect unless :func:`run_paper` is invoked with real credentials. The node is built with
``environment=Environment.LIVE`` (the node lifecycle) + ``BinanceEnvironment.DEMO`` (the venue) so it
is a genuine live event loop against the paper venue — a directional sanity check, NOT a P&L
estimate (the cost stub, not the demo fills, is the validation cost truth, COST Inv XC).

Keys come from the environment (``BINANCE_FUTURES_API_KEY`` / ``BINANCE_FUTURES_API_SECRET`` by
Nautilus convention) or are passed explicitly; the strategy's pure-Python composer + resolved pillar
configs are injected at the ``strategy.configure(deps)`` seam (see ``run/backtest.py`` for the
synthetic deps bundle; live deployment supplies the real per-bar pillar bundle there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # type-only; never imported in the pillar venv (Invariant N).
    from convexity.run.session import SessionPlan
    from convexity.strategy.convexity_strategy import StrategyDeps

VENUE = "BINANCE"


def build_node_from_plan(plan: "SessionPlan", *, demo: bool = True) -> Any:
    """Build the paper/live ``TradingNode`` from a shared :class:`SessionPlan` — the config-only path.

    This is the PAPER/LIVE consumer of the one shared builder (``run/session.build_session``): the
    plan carries the already-wired real-pillar ``deps`` + the venue ``instrument_ids`` / ``bar_types``,
    so paper/live no longer hand-replicate backtest's assembly. ``demo`` selects the venue environment
    (paper ⇒ ``BinanceEnvironment.DEMO``; live ⇒ ``LIVE``). Creds + trader-id come from the plan's
    ``RunConfig``.
    """
    cfg = plan.cfg
    return build_trading_node(
        instrument_ids=plan.instrument_ids,
        bar_types=plan.bar_types,
        deps=plan.deps,
        demo=demo,
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        trader_id=cfg.trader_id or ("CONVEXITY-001" if demo else "CONVEXITY-LIVE-001"),
    )


def build_trading_node(
    *,
    instrument_ids: Sequence[str],
    bar_types: Sequence[str],
    deps: "StrategyDeps",
    demo: bool = True,
    api_key: str | None = None,
    api_secret: str | None = None,
    trader_id: str = "CONVEXITY-001",
) -> Any:
    """Build a fully-wired Binance USDT-M ``TradingNode`` hosting the ``_ConvexityStrategy``.

    ``demo=True`` ⇒ ``BinanceEnvironment.DEMO`` (paper venue); ``demo=False`` ⇒
    ``BinanceEnvironment.LIVE`` (real money — used by ``run/live.py``). The node lifecycle
    ``environment`` is always ``Environment.LIVE`` (a real async event loop); only the BINANCE venue
    environment switches paper↔live. Returns the constructed, NOT-yet-started node — the caller runs
    :meth:`TradingNode.run` with real credentials present.
    """
    from nautilus_trader.adapters.binance.common.enums import (
        BinanceAccountType,
        BinanceEnvironment,
    )
    from nautilus_trader.adapters.binance.config import (
        BinanceDataClientConfig,
        BinanceExecClientConfig,
    )
    from nautilus_trader.adapters.binance.factories import (
        BinanceLiveDataClientFactory,
        BinanceLiveExecClientFactory,
    )
    from nautilus_trader.common.config import Environment, InstrumentProviderConfig
    from nautilus_trader.config import (
        LiveDataEngineConfig,
        LiveExecEngineConfig,
        LoggingConfig,
        TradingNodeConfig,
    )
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model.identifiers import InstrumentId

    binance_env = BinanceEnvironment.DEMO if demo else BinanceEnvironment.LIVE
    load_ids = frozenset(InstrumentId.from_str(i) for i in instrument_ids)
    provider = InstrumentProviderConfig(load_ids=load_ids)

    data_cfg = BinanceDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=BinanceAccountType.USDT_FUTURES,
        environment=binance_env,
        instrument_provider=provider,
    )
    exec_cfg = BinanceExecClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=BinanceAccountType.USDT_FUTURES,
        environment=binance_env,
        instrument_provider=provider,
        use_reduce_only=True,  # the protective stop + every exit is reduce-only (Invariant R)
    )

    config = TradingNodeConfig(
        trader_id=trader_id,
        environment=Environment.LIVE,
        logging=LoggingConfig(log_level="INFO"),
        data_clients={VENUE: data_cfg},
        exec_clients={VENUE: exec_cfg},
        data_engine=LiveDataEngineConfig(),
        exec_engine=LiveExecEngineConfig(reconciliation=True),  # Nautilus reconciles, THEN composer
    )

    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(VENUE, BinanceLiveExecClientFactory)
    node.build()

    # Host the SAME strategy class the backtest runs; inject the pure-Python composer + cfgs.
    from convexity.strategy.convexity_strategy import make_strategy_class
    from nautilus_trader.trading.strategy import StrategyConfig

    class _ConvexityConfig(StrategyConfig, frozen=True):
        bar_types: tuple[str, ...]
        request_warmup: bool = True

    # request_warmup=True ⇒ the strategy primes its rolling buffer from venue history at (re)start so
    # paper/live reach the first decision at the same buffer depth the backtest preloads to (the
    # backtest preloads its bars instead and leaves this False) — cold-start cross-mode parity.
    strategy = make_strategy_class()(
        _ConvexityConfig(bar_types=tuple(bar_types), request_warmup=True)
    )
    strategy.configure(deps)
    node.trader.add_strategy(strategy)
    return node


def run_paper(plan: "SessionPlan") -> None:  # pragma: no cover - creds-required, NOT run in CI
    """Build the DEMO node from a shared :class:`SessionPlan` and run its live event loop (BLOCKS).

    The config-only paper entrypoint: ``build_session(RunConfig(mode="paper", ...))`` then
    ``run_paper(plan)``. Requires Binance demo credentials in the environment or on the plan's
    ``RunConfig`` — this is the only function that actually connects."""
    node = build_node_from_plan(plan, demo=True)
    try:
        node.run()
    finally:
        node.dispose()


__all__ = ["build_trading_node", "build_node_from_plan", "run_paper", "VENUE"]
