"""Cross-sectional (dollar-neutral) crypto momentum + funding-carry research engine.

A DIFFERENT premise from the 1-min scalp harness (``research.driver``). All prior convexity research
tested TIME-SERIES DIRECTIONAL signals — "predict this one asset's next move" — and found gross
directional edge ~ 0 (the market is efficient at that game, and 13bp round-trip friction buried it).

This package tests two premises that are NOT single-asset price prediction:

  * CROSS-SECTIONAL MOMENTUM — a RELATIVE ranking. We do not predict a coin's absolute return; we bet
    that recent relative winners keep out-ranking recent relative losers, dollar-neutral (long the top
    quantile, short the bottom). The market-wide beta cancels; what is left is the cross-sectional
    dispersion premium.
  * FUNDING CARRY — a STRUCTURAL CASH FLOW, not a prediction. A perp with high positive funding pays
    its longs and credits its shorts every 8h; we SHORT high-funding names (collect the credit) and
    LONG low/negative-funding names (collect the credit). This is a carry harvest, not a directional call.

Both run at LOW turnover (weekly rebalance) so the friction that killed 1-min (13bp round-trip) barely
bites — a weekly holding period pays that friction ~52x/yr, not ~thousands of times.

Correctness spine (the whole reason cross-sectional backtests lie):

  * POINT-IN-TIME UNIVERSE (``panel``) — at each rebalance we include ONLY coins already listed with
    enough trailing history for the signal, and still live; we NEVER rank a coin before it existed and
    NEVER hold into a delist gap. No survivorship lookahead.
  * NO-LOOKAHEAD SIGNALS (``signals``) — every signal is computed strictly from data BEFORE the
    rebalance timestamp; the resulting weights are held forward over the next week.
  * HONEST COST (``accounting``) — turnover-based round-trip friction (REUSES the Binance USDT-M taker
    rate from the ``cost`` pillar's ONE fee truth) + SIGNED per-leg funding cash flow.
  * DISCIPLINE (``validate``) — purged/embargoed walk-forward on the WEEKLY series (REUSES SYSTEM's
    ``PurgedWalkForward``) + Deflated Sharpe over the config-trial count (REUSES SYSTEM's
    ``DeflatedSharpe``). The pillars are NOT reimplemented here.

Two-layer altitude discipline (CLAUDE.md rule 7): this is research glue on top of the intact
``src/convexity`` pillars; it is NOT imported by any live/paper module.
"""
