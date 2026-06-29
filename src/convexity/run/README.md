# `run/` — the deployment entrypoints (backtest / paper / live share ONE path)

The whole point of this package: **switch modes by config alone.** All three deployment modes resolve
the universe, build the per-symbol specs, the real eight-pillar composer, the resolved pillar configs
and the `StrategyDeps` through the SAME builder — `run/session.build_session`. The only thing that
differs between modes is the **data source + venue environment + creds**; the strategy class, the §4
per-bar composition, the projection seam, the reduce-only protective stop and the OPS overlay are
identical (ARCHITECTURE §1/§4 — the engine owns backtest↔live parity).

## The one shared builder

```python
from convexity.run.session import RunConfig, build_session

cfg  = RunConfig(mode="backtest", symbols=("BTCUSDT", "ETHUSDT"))   # or sleeves=("crypto", ...)
plan = build_session(cfg)   # -> specs + composer + resolved configs + StrategyDeps + bar_types
```

`build_session` is pure-Python (NautilusTrader-free, Invariant N) so it is importable in the pillar
venv and unit-testable off-VM. It is the bridge from UNIVERSE.md membership (`selection/universe`) to
the AMBER specs and the SYSTEM `SystemUniverse` (`system/config/universe`) — the two universe loaders
that previously never met. The resulting `SessionPlan` is **venue-agnostic**: the same object is what
all three modes below consume.

## Running each mode (the ONLY differences)

| mode | data source | venue env | creds | entrypoint |
|------|-------------|-----------|-------|------------|
| **backtest** | `data/stitched` + `data/funding` lake (preloaded) | simulated `BacktestEngine` | none | `run/backtest.run_session(plan, cfg)` |
| **paper** | live Binance USDT-M klines | `BinanceEnvironment.DEMO` | `BINANCE_FUTURES_*` | `run/paper.run_paper(plan)` |
| **live** | live Binance USDT-M klines | `BinanceEnvironment.LIVE` (REAL MONEY) | `BINANCE_FUTURES_*` | `run/live.run_live(plan, i_understand_real_money=True)` |

```python
# backtest (VM-only — imports nautilus):
from convexity.run.backtest import run_session
result = run_session(plan)                       # BacktestEngine over the lake

# paper (creds-required, NOT run in CI):
from convexity.run.paper import run_paper
run_paper(build_session(RunConfig(mode="paper", symbols=(...))))

# live (creds-required + hard gate, NOT run in CI):
from convexity.run.live import run_live
run_live(build_session(RunConfig(mode="live", symbols=(...))), i_understand_real_money=True)
```

`run/backtest.run_backtest_real(...)` is a thin compatibility wrapper that assembles a `RunConfig` and
delegates to `build_session` → `run_session` — there is no inline deps assembly left in `backtest.py`.

## Cross-mode parity guarantees baked into the shared path

These are the things that make a config-only switch FAITHFUL, not just runnable:

- **Same instrument construction.** `run/data_loader.build_instrument` builds the backtest
  `CryptoPerpetual` from the plan's AMBER `InstrumentSpec`; the live `InstrumentProvider` loads the
  same `instrument_id` (`{SYMBOL}-PERP.BINANCE`). A symbol whose real venue tick/lot differs from the
  asset-class default is corrected in ONE place — `RunConfig.spec_overrides` (see FRICTION #4: the
  per-symbol `exchangeInfo` backfill is a separate data job).
- **Same id↔symbol symbology.** Both maps come from the single `session.instrument_id_str`, so the
  backtest key and the live key cannot diverge.
- **Same funding consumption path.** Backtest feeds `FundingRateUpdate` from `data/funding`; paper/
  live subscribe to the venue funding stream — both land in the strategy's `on_funding_rate` → COST
  `FundingAccrual`. The subscription outcome is **counted + logged** (`funding_subscribed` /
  `funding_subscribe_failed`), never silently swallowed, so a live funding-feed gap is observable
  rather than producing a silent zero-funding divergence from backtest.
- **Same warm-start buffer depth.** The backtest PRELOADS its bar slice, so the rolling buffer is warm
  at the first decision bar. Paper/live build the strategy config with `request_warmup=True`, so
  `on_start` requests `warmup_bars` of venue history and `on_historical_data` primes the same rolling
  buffer — the node reaches its first decision at the same depth in every mode, including after a
  restart (instead of idling for `warmup_bars` live minutes).
- **Same SYSTEM universe.** `build_session` passes the resolved specs to `load_universe_meta`, so the
  orchestrator composes over the actual instruments (the old inline path passed an EMPTY universe).

## `conf/*.yaml`

The canonical config object is `RunConfig` (config-as-code) so the pillar venv needs no YAML
dependency. The `conf/*.yaml` files document the venue/creds surface; a thin `conf → RunConfig` loader
can be added later but is not required for the seamless path.

## `explore.py` — MODE 1 (deliberately NOT the engine path)

`run/explore.py` is a **separate, no-engine research entrypoint**: a lean pandas/numpy first-pass over
the historical lake (net-of-cost, drop-top-N / OOS robustness, cross-pair match comparisons). It does
NOT share `build_session` because it does not stand up a Nautilus node — it is the fast exploration
loop that precedes the parity backtest. It is currently an unimplemented scaffold (`conf/explore.yaml`
lists the knobs); when built it reads the same `data/stitched` lake but stays engine-free by design.
