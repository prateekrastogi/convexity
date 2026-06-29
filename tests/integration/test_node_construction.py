"""PAPER / LIVE NODE-CONSTRUCTION parity (FRICTION #1): the shared SessionPlan wires a real venue node.

The whole point of ``run/session.build_session`` is that ALL three modes consume ONE plan — so a
regression in the VENUE-side wiring of that shared plan (the Binance data/exec clients fed from
``plan.instrument_ids``, the ``_ConvexityStrategy`` hosted with ``plan.deps``) must be caught without
needing live credentials or a connection. This builds the paper node AND the live node from a plan
and asserts they CONSTRUCT and wire correctly — it NEVER calls ``.run()`` (no connect, no creds beyond
dummy placeholders the config object requires).

The same ``build_session(RunConfig(...))`` path the deployment phase drives:
  * ``run/paper.build_node_from_plan(plan, demo=True)``  → ``BinanceEnvironment.DEMO``  (paper venue)
  * ``run/live.build_live_node(..., demo=False)``         → ``BinanceEnvironment.LIVE``  (real money)

Both must: construct a ``TradingNode`` over the BINANCE venue, configure its data+exec clients from
``plan.instrument_ids``, and host the strategy with ``plan.deps`` injected (``configure`` ran). VM-only
(imports nautilus + builds the bundled Binance adapter); cleanly skipped in the pillar venv.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

pytestmark = pytest.mark.nautilus

# Dummy placeholders: the Binance client configs accept api_key/secret, but construction +
# ``node.build()`` never connect, so these are inert. We assert CONSTRUCTION only — never ``.run()``.
_DUMMY_KEY = "dummy-key-not-used"
_DUMMY_SECRET = "dummy-secret-not-used"


def _plan(mode: str):
    from convexity.run.session import RunConfig, build_session

    cfg = RunConfig(
        mode=mode,
        symbols=("BTCUSDT",),
        api_key=_DUMMY_KEY,
        api_secret=_DUMMY_SECRET,
    )
    plan = build_session(cfg)
    # The shared plan carries the venue descriptors + the wired real-pillar deps the node consumes.
    assert plan.instrument_ids == ("BTCUSDT-PERP.BINANCE",)
    assert plan.bar_types == ("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",)
    assert plan.deps.composer is not None
    return plan


def _assert_node_wired(node, *, expect_binance_env, plan) -> None:
    """Assert the constructed node wires the BINANCE venue clients from the plan + hosts the strategy.

    Construction-only: we read the node's TradingNodeConfig + the hosted strategy. We never connect."""
    from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment

    assert node.trader_id is not None  # the node constructed (has a trader id)

    node_cfg = node._config  # the TradingNodeConfig the node was built from
    # The data + exec clients are configured for the BINANCE venue from the plan's instrument ids.
    assert "BINANCE" in node_cfg.data_clients
    assert "BINANCE" in node_cfg.exec_clients
    data_cfg = node_cfg.data_clients["BINANCE"]
    exec_cfg = node_cfg.exec_clients["BINANCE"]
    assert data_cfg.environment is expect_binance_env
    assert exec_cfg.environment is expect_binance_env
    assert exec_cfg.use_reduce_only is True  # the protective stop / every exit is reduce-only (Inv R)
    # The instrument provider loads exactly the plan's instrument ids (the venue-side wiring of the
    # shared SessionPlan — a regression here is what this test guards).
    from nautilus_trader.model.identifiers import InstrumentId

    want_ids = frozenset(InstrumentId.from_str(i) for i in plan.instrument_ids)
    assert frozenset(data_cfg.instrument_provider.load_ids) == want_ids
    assert frozenset(exec_cfg.instrument_provider.load_ids) == want_ids
    assert expect_binance_env in (BinanceEnvironment.DEMO, BinanceEnvironment.LIVE)

    # The strategy is hosted AND configured with the plan's deps (configure ran ⇒ _deps is set).
    strategies = node.trader.strategies()
    assert len(strategies) == 1
    strat = strategies[0]
    assert getattr(strat, "_deps", None) is plan.deps
    # The strategy subscribes to the plan's bar-types (the same descriptors the backtest preloads).
    assert tuple(getattr(strat, "_bar_types")) == plan.bar_types
    # Paper/live request warmup history at (re)start (cold-start parity) — backtest preloads instead.
    assert getattr(strat, "_request_warmup") is True


def test_paper_node_constructs_from_shared_plan():
    """build_session(mode='paper') → paper.build_node_from_plan(demo=True): DEMO venue, no connect."""
    from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment

    from convexity.run.paper import build_node_from_plan

    plan = _plan("paper")
    node = build_node_from_plan(plan, demo=True)
    try:
        _assert_node_wired(node, expect_binance_env=BinanceEnvironment.DEMO, plan=plan)
    finally:
        node.dispose()


def test_live_node_constructs_from_shared_plan():
    """build_session(mode='live') → live.build_live_node(demo=False): LIVE venue, no connect."""
    from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment

    from convexity.run.live import build_live_node

    plan = _plan("live")
    node = build_live_node(
        instrument_ids=plan.instrument_ids,
        bar_types=plan.bar_types,
        deps=plan.deps,
        api_key=_DUMMY_KEY,
        api_secret=_DUMMY_SECRET,
    )
    try:
        _assert_node_wired(node, expect_binance_env=BinanceEnvironment.LIVE, plan=plan)
    finally:
        node.dispose()


def test_live_run_guard_blocks_without_acknowledgement():
    """run_live is a HARD real-money gate: it refuses to even build+run without the explicit ack.

    No node is built and nothing connects — the guard raises before ``build_node_from_plan``."""
    from convexity.run.live import run_live

    plan = _plan("live")
    with pytest.raises(RuntimeError, match="REAL MONEY"):
        run_live(plan, i_understand_real_money=False)
