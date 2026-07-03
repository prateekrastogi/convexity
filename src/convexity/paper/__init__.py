"""The live/paper weekly rebalancer (docs/specs/PAPER.md) — the ONE module EDGE.md §(e) deferred.

Thin composition of the validated pillars over a scheduled weekly session (systemd timers, not a
daemon — the pure-recompute doctrine means every session rebuilds all strategy state from
persisted history): lake + REST top-up -> panel -> band-ensemble targets (parity-pinned to the
engine) -> quarter-Kelly gross -> no-trade band + quantized maker-first order intents -> journal
-> (optional --live) Binance USDT-M DEMO submission via the NautilusTrader adapter.

Dry-run is the v1 primary mode and works with no API keys; the gateway is EXPERIMENTAL until the
first supervised demo session (PAPER.md §8).
"""
