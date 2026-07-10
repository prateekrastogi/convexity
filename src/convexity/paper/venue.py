"""Single venue switch (PAPER.md §12.6) — ONE config value selects demo|prod everywhere.

The 2026-07-10 production audit found venue selection scattered across four hardcoded sites
(gateway env enum, filters URL, tradeable-gate URL, order-peg URL) — a cutover by hand-edit
risks orders on one venue quantized/gated against another's filters (the SPCX -1111 class, with
money). This module is now the ONLY place venue resolution happens:

  * ``CONVEXITY_VENUE`` env var: ``demo`` (default) or ``prod``.
  * EXECUTION surfaces (orders, filters, tradeable gate, maker pegs, live prices) all resolve
    through :func:`execution_base` / :func:`nautilus_environment` / :func:`key_env_names`.
  * DATA surfaces (daily klines/funding tails for the panel) stay on PROD always —
    ``topup.FAPI`` — because demo market history is thin/laggy and research must never read it.
  * Every live session journals a ``venue`` record (:func:`describe`) and asserts the Nautilus
    adapter resolved the SAME base URL (:func:`assert_consistent`) — drift between the two
    resolution paths aborts the session before any order.
"""

from __future__ import annotations

import os

EXEC_BASES = {"demo": "https://demo-fapi.binance.com", "prod": "https://fapi.binance.com"}
KEY_ENVS = {
    "demo": ("BINANCE_FUTURES_DEMO_API_KEY", "BINANCE_FUTURES_DEMO_API_SECRET"),
    "prod": ("BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET"),
}


def venue() -> str:
    v = os.environ.get("CONVEXITY_VENUE", "demo").strip().lower()
    if v not in EXEC_BASES:
        raise RuntimeError(
            f"CONVEXITY_VENUE={v!r} invalid — must be 'demo' or 'prod' (.env); refusing to "
            f"guess an execution venue")
    return v


def execution_base() -> str:
    """Base URL for every EXECUTION-venue HTTP surface (filters, tradeable, pegs, raw signed)."""
    return EXEC_BASES[venue()]


def key_env_names() -> tuple[str, str]:
    return KEY_ENVS[venue()]


def nautilus_environment():
    from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment

    return BinanceEnvironment.DEMO if venue() == "demo" else BinanceEnvironment.LIVE


def assert_consistent() -> None:
    """The Nautilus adapter's URL resolution must agree with ours — checked at every live
    session start (PAPER.md §12.6). A mismatch means orders would go to one venue while
    filters/pegs read another: abort before any order."""
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
    from nautilus_trader.adapters.binance.common.urls import get_http_base_url

    nb = get_http_base_url(BinanceAccountType.USDT_FUTURES, nautilus_environment(), False)
    if nb.rstrip("/") != execution_base().rstrip("/"):
        raise RuntimeError(
            f"venue resolution drift: nautilus={nb} vs execution_base={execution_base()} "
            f"(CONVEXITY_VENUE={venue()}) — aborting before any order")


def state_root():
    """Venue-namespaced paper-state root (PAPER.md §12.6): demo keeps the original data/paper
    (journals, lock, ledgers — history byte-untouched); prod isolates under data/prod so BOTH
    fleets run from ONE checkout without ever blending accounting. Lake/universe stay shared
    (venue-agnostic data)."""
    import pathlib

    return pathlib.Path("data/paper" if venue() == "demo" else "data/prod")


def describe() -> dict:
    return {"venue": venue(), "execution_base": execution_base(),
            "key_env": key_env_names()[0]}
