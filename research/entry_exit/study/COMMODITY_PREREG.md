# COMMODITY STUDY PRE-REGISTRATION — frozen 2026-07-04, BEFORE any run

New dated pre-registration for the **commodity/metal futures entry/exit model study**, as required
by `research/entry_exit/PREREGISTRATION.md` (the NIFTY-options prereg does not license this run).
Everything below is frozen before any backtest executes; only what is listed here may later be
claimed, and multiplicity is charged against this full ledger.

**Prior to beat / honesty bar.** convexity `docs/EDGE.md` measured simple breakout/trend
directional DEAD across assets INCLUDING commodities (gross R ≈ 0; loss = friction). The intraday
repo's README found MCX Crude ORB ~0.7 Sharpe on a thin anecdotal sample and NatGas reliably
negative. Expectation: the translated baseline is weak-to-dead; the genuine open question is
whether the FORMAL models beat it. Negatives are recorded plainly; nothing is adopted that fails
the gates.

## 1. Instruments & data

The 7 stitched commodity/metal perps (1-min, `data/stitched/{SYM}.parquet`): XAUUSDT, XAGUSDT,
CLUSDT, NATGASUSDT, COPPERUSDT, XPTUSDT, XPDUSDT. (BZUSDT is not in the lake — deduped vs CL.)
Spans ~2021-06→2026-06 (XPT from 2021-11, XPD from 2021-07). The lake splices a SESSIONED
Dukascopy underlying era onto the 24/7 Binance perp era; `gap_before` flags session/weekend gaps.
Day grid = **UTC day**. A valid decision day has ≥ 300 one-minute bars. Returns are never taken
across a `gap_before` boundary as if continuous; stop fills honor gap semantics (§3).

## 2. Baseline — the intraday README incumbent, translated to linear 24/7 perps

- **Opening range**: high/low of the first `W` minutes of the UTC day, W ∈ {30, 120}
  (120 ≈ 8% of a 24h session — the validated "OR scales with session" rule; 30 = the NIFTY window).
- **Entry signal (verbatim `signal()` logic)**: LONG iff day_high > OR_high AND day_close > OR_high
  AND (close−day_low)/day_range > 0.7; SHORT mirrored. Decision and entry mark at the **day close**
  (the incumbent's stated modeling convention, applied identically to every candidate — matched).
- **Stop (THE EDGE)**: vol-scaled underlying stop, distance = `s × ATR20` from entry,
  s ∈ {0.25, 0.5, 1.0} (a fixed % is indefensible across XAU-vol vs NATGAS-vol; ATR-scaling is the
  cross-instrument analog of the −12%-premium stop). Checked on every 1-min bar of the hold:
  intraday touch fills AT the stop; a bar OPENING beyond the stop (incl. after `gap_before`) fills
  at that open — **gap-through can exceed the stop** (duka-era sessions), while the 24/7 perp era
  fills cleanly (the README's own gappy-vs-continuous distinction, now within one sample).
- **Hold**: 5 valid days ("ride winners"); exit at day-5 close if never stopped. Non-overlapping.
- **PRIMARY baseline cell (named now): W=120, s=0.5.** The other 5 cells are declared sensitivities.
- **P&L**: side × (exit/entry − 1) per trade, minus cost (§4). No compounding inside a trade.

## 3. Candidate ledger (frozen; nothing off-ledger may be claimed)

Daily-grid models on UTC-day log-returns of each symbol; every parameter fit on the CPCV **train**
folds and FROZEN for test; causal/filtered estimates only (smoothed = lookahead; pinned by the
package's own tests).

- **Entries (replace the OR signal; same day-close entry mark)**: (E1) BOCPD run-length posterior,
  hazard 1/{50,100,200}, entry on posterior break prob > {0.5,0.7}; (E2) Wald-SPRT/GLR drift-sign,
  (α,β) ∈ {(0.05,0.2),(0.01,0.1)}, window m ∈ {20,60}; (E3) 2-state Gaussian HMM filtered
  P(high-drift) > {0.6,0.75}; (E4) Lee-Mykland jumps at 1% size, trailing K ∈ {16,32} days,
  jump-sign continuation. Direction: sign of the detected drift/jump/state mean.
- **Exits (replace fixed-5d; the ATR stop of §2 remains as the floor in ALL variants)**:
  (X0) fixed 5-day (incumbent's); (X1) optimal-stopping free boundary; (X2) OU/Leung-Li b* exit;
  (X3) hazard-duration, exit when 1-step reversal hazard > {0.5,0.7}. Max hold cap 10 days.
- **HAR-RV input (H ∈ {off,on})**: HAR forecast (1/5/22) as the stop-width scaler (replaces ATR20
  with an equivalent-scale HAR forecast) — one toggle, no extra grid.
- **Composed grid**: 4 entries × 4 exits × 2 HAR = 32 combos/symbol + 6 baseline cells. `N_eff`
  for DSR counts the WITHIN-model parameter grids ((E1):6, (E2):4, (E3):2, (E4):2, (X3):2, stop-s:3,
  W:2) per the PREREGISTRATION composed-grid rule — not the naive combo count.

## 4. Costs (net-of-cost everywhere)

Round trip = {6, 12, 24} bp scenarios; **12 bp primary** (Binance TradFi-perp taker ~5bp/side + 2bp
slippage; measured spreads on these perps are ≤ ~9bp single-side, mostly ~1–5bp). A stopped trade
pays the full RT. Funding: TradFi-perp funding history is too short to replay over 2021-26, so a
declared **±5bp per held day** drag scenario brackets it (reported alongside, never silently).

## 5. Evaluation protocol

- **CPCV**: N=8 contiguous groups on the daily grid, k=2 test groups (28 splits), **embargo 10
  days** at every train/test boundary; per-config per-symbol daily net-return series. PBO via
  `convexity.system.cpcv.compute_pbo` over the full family; DSR/PSR (per-observation Sharpe,
  `entry_exit.validation.deflated_sharpe`) with the §3 `N_eff`.
- **Metrics reported per config × symbol and pooled**: n trades, win rate, EV/trade (bp),
  per-trade & annualized Sharpe, maxDD, drop-top-1/3 EV, top-3 PnL share, CPCV path distribution.
- **Optimal leverage**: fractional Kelly off the NET per-trade stream — report **quarter-Kelly
  (headline)**, full Kelly, and the DD-vs-leverage curve. Leverage is Sharpe-invariant and creates
  no edge; it is reported as sizing, never as performance.
- **Ranking**: best commodities = highest pooled-CPCV OOS Sharpe surviving the gates below, with
  per-year sign stability shown.

## 6. Adoption / claim gates (all five, else record the negative and keep the incumbent)

A candidate is "better than baseline" ONLY if, on CPCV OOS: (1) it beats the PRIMARY baseline cell
net-of-cost after an AIC-style parsimony penalty on its parameter count; (2) DSR passes at the
ledger `N_eff`; (3) PBO ≤ 0.25; (4) EV survives drop-top-3; (5) paired-bootstrap Sharpe-difference
90% CI clears zero. Per-symbol claims additionally require the pooled result to hold (no
single-symbol cherry-pick). Baseline numbers are reported regardless of outcome.
