# research/ — crypto cross-sectional momentum engine

Research-only, NautilusTrader-free code sitting **on top of the intact `src/convexity` pillars**
(it reuses their real math — the ONE Binance fee truth, `PurgedWalkForward`, `DeflatedSharpe`,
CPCV — and does not reimplement any of it). The prior 1-min directional scalp harness that lived
here has been removed (proven no gross directional edge; see `../docs/EDGE.md`).

## What this tests

A **cross-sectional** premise, NOT single-asset price prediction:

- **CROSS-SECTIONAL MOMENTUM** — a relative ranking. We do not predict a coin's absolute return; we
  bet recent relative winners keep out-ranking recent relative losers, dollar-neutral (long the top
  quantile, short the bottom). Market-wide beta cancels; what's left is the cross-sectional
  dispersion premium.
- **FUNDING CARRY** — a structural cash flow, not a prediction. Short high-funding names (collect
  the credit), long low/negative-funding names. Real income but ~70% structural DD standalone —
  income-overlay only.

Both run at **weekly** turnover, so the ~13bp round-trip friction that buried the 1-min harness is
paid ~52×/yr, not thousands of times.

## Correctness spine (why cross-sectional backtests lie)

- **Point-in-time universe** (`xsec/panel.py`) — at each rebalance include ONLY coins already
  listed with enough trailing history and still live. Never rank before existence; never hold into
  a delist gap. No survivorship lookahead.
- **No-lookahead signals** (`xsec/signals.py`) — every signal computed strictly from data BEFORE
  the rebalance stamp; weights held forward one week.
- **Honest cost** (`xsec/accounting.py`) — turnover-based round-trip friction (REUSES the Binance
  USDT-M taker rate from the `cost` pillar's ONE fee truth) + SIGNED per-leg funding cash flow.
- **Discipline** (`xsec/validate.py`, `xsec/cpcv.py`) — purged/embargoed walk-forward on the weekly
  series + Deflated Sharpe over the config-trial count + CPCV/PBO. All REUSE the SYSTEM pillars.

## Files

| File | Role |
|---|---|
| `xsec/panel.py` | point-in-time daily panel + universe-size-by-year |
| `xsec/signals.py` | no-lookahead momentum / carry signals |
| `xsec/portfolio.py` | quantile construction, vol-scaled weighting, realized vol |
| `xsec/accounting.py` | turnover cost (Binance taker) + signed funding cash flow |
| `xsec/engine.py` | `run(config)` -> weekly return series (`XSecBook` / `XSecResult`) |
| `xsec/overlay.py` | regime / breadth overlays |
| `xsec/metrics.py` | weekly-return metrics (annualized Sharpe, CAGR, DD, ...) — NOT per-trade R |
| `xsec/validate.py` | `deflated_sharpe`, `purged_walkforward` wiring |
| `xsec/cpcv.py` | combinatorial purged CV / PBO |
| `xsec/universe.py` | point-in-time membership helpers |
| `data.py` | `load_bars` / `load_funding` from the lake |
| `assetclass.py` | sleeve / asset-class detection, native-funding flag |

## Result (honest)

Weekly dollar-neutral quintile momentum, vol-scaled, ~60–75 point-in-time crypto names, 12–16d
lookback band: **net Sharpe ~0.8–1.0**, CAGR ~20–25% @100% gross, maxDD ~25%, ~60% of weeks
positive. Passed purged-WF / DSR / bootstrap / holdout / per-regime / breadth / CPCV-PBO
(PBO 0.36). Deploy the lookback BAND, not one fitted L. Full memo: `../docs/EDGE.md`.

Two-layer altitude discipline: this is research glue on the intact `src/convexity` pillars; it is
NOT imported by any live/paper module. The live rebalancer is not built yet.
