# convexity — directional convex breakout on Binance futures (development rules)

This repo trades ONE convex/fat-tailed edge on Binance USDT-M perpetual futures: a directional
**breakout entry** → cut losses fast at a true intrabar stop → ride winners on a bounded hold.
Low win rate, rare big winners. **The cut-losses-fast stop IS the edge**, not a risk add-on.

Two structural differences from the `arbitrage` repo it descends from:
1. **24/7 market, no session open** — the "opening range" is redefined (UTC-day-anchored range OR
   rolling Donchian N-bar break). *Owner to specify.*
2. **Many pairs traded SIMULTANEOUSLY** — a portfolio of independent convex breakout bets across a
   universe of perps, run at once. Diversification across instruments is the structural variation
   vs arbitrage's single-instrument bet (it smooths the fat-tailed single-name drawdown). *Owner to
   specify the universe, allocation, and per-pair sizing.*

Perps are **linear** — no strikes, no greeks, no moneyness. The tunable is position size / leverage
per pair. **This is NOT a hedged book** — pairs are independent directional bets, not offsetting legs.

> **Strategy specifics (entry rule, exit/hold, stop level, pair universe, allocation) are owner-defined
> and will be filled into the blank module slots.** This file fixes the engineering ethos + the
> verified Binance/Nautilus integration facts; it does not pre-decide the strategy.

## Refuted / forbidden (do not reintroduce)
Tested in `arbitrage`/`vol_research` and bled, or no crypto-perp analog:
- **long-vol straddle / strangle** structures — refuted.
- **realized-vs-implied vol-forecast** signals, incl. **HAR-RV** and **Lee-Mykland** — tested and bled.
- **backspread** structures — refuted.
- **OU / Ornstein-Uhlenbeck mean-reversion** exit rules — refuted.
- **options strikes / delta / greeks / moneyness** — perps are linear; no analog.
- **expiry-capped holds / contract roll** logic — perps have no expiry; bound the hold by an explicit
  time/bar rule instead.
- the **NSE fee stack** (STT/SEBI/stamp/GST/Indian brokerage) and the **breeze-connect** adapter —
  replaced by Binance maker/taker + funding and the Nautilus Binance adapter.
- **Reasoning trap**: presenting un-robust, lookahead-tainted, or gross-of-cost results as edge. The
  fat-tailed payoff makes false positives (one lucky winner, a funding/cost artifact) trivially easy.
  Results that don't survive **drop-top-N** and **OOS** are NOT edge.

## CORE PRINCIPLE (non-negotiable): quality open-source first, minimal custom code
Trading-infra bugs cost real money. Reuse battle-tested, actively-maintained, permissively-licensed
libraries on EVERY critical path; write only what no quality library covers. An abandoned/niche library
is a liability → write our own well-tested code instead. We do not re-implement an engine, a broker
connector, a dataframe layer, or a backtest framework — those are reused.

### Reuse table
| Need | Use | Notes |
|---|---|---|
| Paper + live + parity-backtest engine | **NautilusTrader** `nautilus_trader>=1.229,<1.230` | Rust event-driven, deterministic, LGPL-3.0. ONE Strategy class runs backtest/paper/live. Native Binance USDT-M adapter is **bundled in the core wheel** — there is NO `binance` extra; `nautilus_trader[binance]` is invalid. |
| Binance Futures connectivity | Nautilus native adapter | `BinanceLiveDataClientFactory` / `BinanceLiveExecClientFactory` on the `BINANCE` venue of a `TradingNode`. Configs: `BinanceDataClientConfig` / `BinanceExecClientConfig`, `account_type=BinanceAccountType.USDT_FUTURES`. |
| Dataframes / numerics / Parquet | **pandas** / **numpy** / **pyarrow** | All hard transitive deps of nautilus_trader (pandas>=2.3.3, numpy>=1.26.4, pyarrow>=24.0.0) — free to use. pandas feeds `BarDataWrangler`; pyarrow backs the lake + `ParquetDataCatalog`. |
| Bulk historical klines + funding | **httpx** + stdlib `zipfile`/`csv` over **data.binance.vision** | PRIMARY: `data/futures/um/monthly\|daily/klines/<SYM>/<INT>/` and `.../fundingRate/<SYM>/`. Free, no key, no rate limit. ~30-line resumable downloader we own (reference: `binance/binance-public-data`, MIT — vendor, don't depend). |
| Recent-tail data + ad-hoc / metadata | **ccxt** `fetch_ohlcv` / `fetch_funding_rate_history` | SECONDARY only. Patches the unpublished tail. ~1000 candles/call → loop on `since`. NEVER multi-year backfill. |
| Notebooks (dev only) | **ipykernel** (dev group) | Kept out of the paper/live runtime env. |

Do NOT add: polars (no engine payoff yet — Nautilus speaks pandas; defer until pandas-memory-bound),
statsmodels/scipy-stats (signals are arithmetic on numpy/pandas), freqtrade/hummingbot/vectorbt/zipline
(competing engines — fragment parity), raw python-binance (duplicates the adapter), PyPI dump-downloader
wrappers (thin; we own a ~30-liner).

### Write ourselves ONLY (the irreducible strategy surface — all currently blank slots)
1. **Breakout signal** (`strategy/breakout.py`) — redefined for 24/7. No-lookahead by construction.
2. **Pair universe + portfolio allocation** (`strategy/universe.py`) — the set of perps traded
   simultaneously and how capital is split across the independent bets.
3. **Binance cost model** (`costs/binance_fees.py`) — maker vs taker fee, per-8h funding (sign by side
   + rate), slippage. The backtest fill model MUST mirror this schedule or parity breaks.
4. **The intrabar stop + bounded hold** — true cut-losses-fast intrabar stop (re-validate the level for
   crypto vol/leverage; arbitrage's -12% is NOT gospel) and an explicit time/bar hold bound (NO expiry;
   funding accrues ~8h so long holds carry cost). In paper/live the stop is a real exchange/engine order.
5. **Thin data adapter** (`adapters/binance_data/`, `adapters/catalog.py`) — bulk resumable backfill +
   ccxt tail patch + parquet lake + load into `ParquetDataCatalog` (set `ts_init` to bar close — no
   lookahead). Let `BinanceFuturesInstrumentProvider` own instrument specs; no parallel spec table.
6. **The single Strategy class** (`strategy/convexity_strategy.py`) — written ONCE, run unchanged across
   parity-backtest, paper, and live; manages N simultaneous pair positions.

## The three modes (exploration → paper → live)
- **EXPLORATION** (`run/explore.py`) — lean pandas/numpy first-pass backtest on the historical lake.
  No engine. The ONLY place the code truly diverges from the engine path.
- **PAPER** (`run/paper.py`) — `TradingNode` with `environment=BinanceEnvironment.DEMO` on BOTH data and
  exec configs (Nautilus 1.229 prefers DEMO for new Futures sims; TESTNET is a legacy-but-working
  fallback). Directional sanity check only — sim fills are NOT representative P&L.
- **LIVE** (`run/live.py`) — identical TradingNode wiring but `environment=BinanceEnvironment.LIVE` + live keys.

**LIVE TRADES REAL MONEY.** Paper-before-live is a hard gate: never promote to live without a clean
paper run, and never run live without the `conf/live.yaml` guardrails (max notional, max leverage,
kill-switch) in force. Separate API keys per environment.

## Rules for Claude
1. **Search for a QUALITY library before writing anything non-trivial.** Reuse on every critical path.
2. Keep the custom surface **MINIMAL + well-tested with pytest, NO-LOOKAHEAD by construction, NET-OF-COST**
   (maker/taker + per-8h funding + slippage always charged; in backtest funding is silently zero unless
   you feed `FundingRateUpdate` data with `next_funding_ns`).
3. **MATCH comparisons** (same hold / same conditions across variants and pairs) and **REPORT
   winner-robustness** (drop-top-N, OOS held-out) — the fat-tailed payoff is trivially easy to fool yourself on.
4. **ONE strategy class** across parity-backtest/paper/live — the thing you research is the thing you trade.
5. Pin deps with **uv**; run with **`uv run --no-sync`**. NEVER commit secrets (`.env` gitignored).
   Commit/push only when explicitly asked.
6. Repo hygiene: hatchling build, ruff line-length 100, pytest, `uv.lock` committed. Keep
   **`.python-version` = 3.12** — the published cp312 `manylinux_2_35` aarch64 wheel is what makes
   `uv sync` resolve a binary (no source build); other Python versions / Alpine-musl have no wheel.
7. **Two-layer altitude discipline**: lean pandas/numpy for exploration; escalate to Nautilus only for
   parity-backtest/paper/live. Don't drag the engine into research.

## Binance gotchas (load-bearing)
- Symbology: the USDT-M perpetual is **`BTCUSDT-PERP`** in Nautilus; bare `BTCUSDT` is SPOT.
- `environment` replaced the old `testnet=True` bool — set `environment=BinanceEnvironment.DEMO`
  (preferred for Futures) on BOTH configs; DEMO and TESTNET are distinct sandboxes. Separate keys per env.
- Backtest funding only happens if you feed `FundingRateUpdate` data — otherwise perp funding is silently
  zero, overstating a breakout/carry P&L. `FundingRateUpdate.interval` is `None` for Binance (hardcode 8h).
- The strategy must tolerate **external/exchange-generated** fills/position changes (liquidation/ADL/
  settlement: `autoclose-`/`adl_autoclose`/`settlement_autoclose-` prefixes).
- data.binance.vision lag: monthly files settle ~first Monday, daily next day — patch the recent tail
  with ccxt before a fresh run.
- Run one client instance per product type (UM and CM share a 2400 weight/min IP pool). When Binance
  omits the commission field the adapter estimates taker at 0.04% — the backtest fill model must mirror
  the live maker/taker schedule.
- Pin to the released tag (1.229.0); the `develop` branch is nightly and config fields move.
