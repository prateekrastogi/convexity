"""Vectorized research backtest harness for the 1-min crypto scalp (NautilusTrader-free).

This package is the FAST sweep driver that sits ON TOP of the intact ``src/convexity`` pillars — it
REUSES their real math (breakout signal + quality_score, trend direction + strength, robust vol/ATR,
COST fees/spread/impact + funding, SYSTEM purged-walk-forward + Deflated-Sharpe) and adds only the
thin glue needed to:

  * load the 1-min crypto lake (``data/stitched/{SYMBOL}.parquet``) + funding lake,
  * fit TRAIN-only terciles for the trend-aligned-breakout alignment cells,
  * forward-scan each signal ONCE to resolve all RR brackets + max-adverse-excursion,
  * apply net cost (fees + half-spread + impact) + wall-clock funding + LIQUIDATION analytically
    across the fixed-leverage grid,
  * pool across instruments under a purged walk-forward and deflate the Sharpe over the trial count.

The committed Nautilus ``strategy/`` path is the thing that is eventually TRADED; this harness is the
research-only first pass (two-layer altitude discipline, CLAUDE.md rule 7). It is NOT imported by any
live/paper module.
"""
