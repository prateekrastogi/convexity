# convexity — crypto cross-sectional momentum (development rules)

This repo researches ONE surviving edge on Binance USDT-M perpetual futures: **weekly,
dollar-neutral, quintile CROSS-SECTIONAL MOMENTUM** on a point-in-time crypto panel. It is a
RELATIVE ranking (long recent relative winners, short recent relative losers) — NOT a single-asset
price prediction. Market beta cancels; the harvested premium is the cross-sectional dispersion.
Low turnover (weekly) means the ~13bp round-trip friction that killed everything faster barely
bites. Full evidence + the deploy decision: **`docs/EDGE.md`**.

The repo was pivoted from a directional convex-breakout thesis. That thesis is **dead** (gross
directional edge ≈ 0) and its code has been removed. Do not resurrect it.

## Refuted / forbidden (do not reintroduce)
Tested here or in the `arbitrage`/`vol_research` lineage and bled — or superseded by the pivot:
- **Directional breakout / trend** across {1m,1h,4h,daily} × {fixed-%, vol-stop, no-stop, fixed-RR}
  × {breakout-strength, conviction, trend-alignment} × leverage × {crypto,equities,commodities} ×
  even INVERTED. Root cause = **gross directional R ≈ 0**; the loss is pure friction-in-R. The
  "anti-predictive strong-breakout" cell was a **cost artifact**, not a fadeable edge. DEAD.
- **Funding carry as a standalone book** — real +12.9%/yr income but ~70% structural drawdown.
  Income-overlay at most; never the book.
- **long-vol straddle/strangle, HAR-RV / Lee-Mykland vol forecasting, backspreads, OU
  mean-reversion exits** — refuted upstream.
- **options strikes / delta / greeks / moneyness / expiry-roll** — perps are linear; no analog.
- the **NSE fee stack** and the **breeze-connect** adapter — replaced by Binance maker/taker +
  funding.
- **Reasoning trap**: presenting un-robust, lookahead-tainted, or gross-of-cost results as edge.
  Cross-sectional backtests lie easily (survivorship, delist gaps, size-blind cost). Results that
  don't survive **DSR / PBO / purged-WF / drop-top-N** are NOT edge. Deploy the lookback BAND, not
  one fitted L (PBO 0.36 → the single best L is partition-fragile; `frac_pos` is the robust signal).

## CORE PRINCIPLE (non-negotiable): quality open-source first, minimal custom code
Trading-infra bugs cost real money. Reuse battle-tested, actively-maintained, permissively-licensed
libraries on EVERY critical path; write only the irreducible strategy surface. The cross-sectional
engine is thin glue implemented as nine per-pillar packages under `src/convexity/{panel,universe,
signal,portfolio,leverage,cost,execution,regime,system,ops}/` — it reuses their real math (the ONE
Binance fee truth, `PurgedWalkForward`, `DeflatedSharpe`, CPCV) and does not reimplement any of it.
We do not re-implement an engine, a broker connector, a dataframe layer, or a backtest framework —
those are reused.

### Reuse table
| Need | Use | Notes |
|---|---|---|
| Paper + live engine (when the rebalancer is built) | **NautilusTrader** `>=1.229,<1.230` | Rust event-driven, deterministic, LGPL-3.0. ONE Strategy class across paper/live. Native Binance USDT-M adapter is bundled in the core wheel (no `binance` extra). |
| Binance Futures connectivity | Nautilus native adapter | `BinanceLive{Data,Exec}ClientFactory` on the `BINANCE` venue; `account_type=USDT_FUTURES`. |
| Dataframes / numerics / Parquet | **pandas / numpy / pyarrow** | Hard transitive deps of nautilus_trader — free to use. pyarrow backs the lake. |
| Bulk historical klines + funding | **httpx** + stdlib over **data.binance.vision** | `data/futures/um/monthly\|daily/klines/` and `.../fundingRate/`. Free, no key. The `scripts/backfill_*` own a ~30-line resumable downloader. |
| Recent-tail / metadata | **ccxt** | SECONDARY only — patch the unpublished tail; never multi-year backfill. |
| Validation (DSR / purged-WF / CPCV) | `src/convexity/system/validate/` | REUSED by `src/convexity/system/xsec_validate.py` + `cpcv.py`. Do not fork. |

Do NOT add competing engines (freqtrade/hummingbot/vectorbt/zipline), raw python-binance
(duplicates the adapter), or PyPI dump-downloader wrappers (we own the ~30-liner).

### Write ourselves ONLY (the irreducible surface)
1. **Cross-sectional engine** (`src/convexity/{panel,signal,portfolio,cost,execution,regime,
   system,ops}/`) — panel / signals / portfolio / accounting / engine / validate / cpcv / metrics.
   No-lookahead + net-of-cost by construction. THIS EXISTS.
2. **Data-layer backfill** (`scripts/backfill_*_1m.py`, `stitch_tradfi_1m.py`) — resumable lake
   builders; self-contained, no convexity imports. THESE EXIST.
3. **The live/paper weekly rebalancer** — NOT built. The strategy slot is intentionally empty.
   Build it against NautilusTrader when papering starts (see `README.md` + `docs/EDGE.md` §e).
   Do NOT pre-build it.

## Rules for Claude
1. **Search for a QUALITY library before writing anything non-trivial.** Reuse on every critical path.
2. Keep the custom surface **MINIMAL + pytest-tested + NO-LOOKAHEAD + NET-OF-COST** (turnover-based
   Binance taker friction + signed per-leg funding always charged).
3. **Point-in-time universe** (no survivorship, no delist-gap holds). Signals strictly BEFORE the
   rebalance stamp; weights held forward one week.
4. **Report winner-robustness** — DSR, PBO/CPCV, purged-WF, drop-top-N, per-regime. Deploy the BAND.
5. Pin deps with **uv**; run with `.venv/bin/python` (or `uv run --no-sync`). NEVER commit secrets
   (`.env` gitignored). Commit/push only when explicitly asked.
6. Repo hygiene: hatchling build, ruff line-length 100, pytest, `uv.lock` committed. Keep
   **`.python-version` = 3.12** (the cp312 manylinux aarch64 wheel is what makes `uv sync` resolve
   a binary).
7. **Two-layer altitude discipline**: lean pandas/numpy for the research engine; escalate to
   Nautilus ONLY for the eventual paper/live rebalancer. Don't drag the engine into research.

## Data lake (load-bearing)
- `data/` is the 137-instrument 1-min lake (gitignored, ~15G). The xsec panel is built from it via
  `src/convexity/panel/data.py` (`load_bars` / `load_funding`). Do NOT touch or delete it.
- Symbology: the USDT-M perpetual is `BTCUSDT-PERP` in Nautilus; bare `BTCUSDT` is SPOT.
- Backtest funding only accrues if you feed it — `src/convexity/cost/accounting.py` charges signed
  funding explicitly. The scalp harness's silent-zero-funding trap does not apply here, but keep
  funding signed by side.
