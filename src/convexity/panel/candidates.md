# convexity trading universe

The candidate universe consumed by the cross-sectional engine (`src/convexity/panel/sleeve.py`). Target
account ≈ **$1,000**, **1-minute** resolution, **5-year** history target. Built for diversification
*within* each sleeve and *across* the whole book, not for raw volume.

## Design rationale

**Three sleeves, deliberately decorrelated.** Measured correlation structure (Binance, hourly):
within-group avg pairwise — **commodities 0.21 < equities 0.31 ≪ crypto-majors 0.88**; to-crypto —
**commodities 0.17 < equities 0.37**. So commodities are the prize diversifier, equities a partial
one, and crypto is mostly one risk-on/off factor. The optimiser will hold ~15–25 of these on any day.

- **Crypto (50)** — curated for **large market-cap + sector spread**, *not* raw 24h volume (the
  volume leaderboard is full of low-mcap pump coins — LAB/SLX/SYN/HEI/MAGMA — which were excluded).
  Spread across L1s, infra/oracle/DA, payments, L2, DeFi, memes, AI, privacy, perp-DEX, gaming so the
  sleeve isn't 50× BTC-beta. Native Binance 1-min, ~5 yr for majors down to ~1–2 yr for newer alts.
  *Excluded:* PAXG/XAUT (gold tokens — duplicate of the XAU commodity); USDC (stablecoin).

- **Equities (30)** — **mega-cap, factor-diversified** global names. Semis **capped at 3**
  (NVDA/TSM/AVGO) to avoid the one-factor semiconductor cluster; **crypto-proxies excluded**
  (MSTR/COIN/CRCL are crypto beta, not diversification); meme/junk excluded. Spans software, internet,
  semis, hardware, consumer disc/staples, financials, healthcare, energy, China, IT-services, plus two
  index factors (QQQ large-tech, IWM small-cap). **Data: Dukascopy underlying (long history) stitched
  with the recent Binance perp.** NOTE: the Dukascopy proxy is the *underlying stock* (US-session only)
  → it has **no overnight-gap data** and isn't 24/7 like the perp. Treated as an **indicator-grade**
  series for the backtest, not an exact replica of the perp.

- **Commodities / metals (7)** — **all liquid ones, deduped.** Dropped **Brent (BZ)** — ~0.95
  correlated to **WTI (CL)**, the "cl and wti" duplicate; kept the more-liquid WTI. Precious metals
  (gold/silver/platinum/palladium) cluster ~0.5–0.7 among themselves (the optimiser handles that via
  the covariance), copper is the industrial metal, crude+natgas the energy factor (the genuine
  low-to-BTC diversifiers, ~0 correlation). **Data: Dukascopy underlying (long) stitched with Binance perp.**

**Total: 50 crypto + 30 equities + 7 commodities = 87 instruments.**

## Data sourcing & history

| Sleeve | Long-history source | Recent / live | Stitch |
|---|---|---|---|
| Crypto (50) | Binance `data.binance.vision` 1-min dumps, 5 yr | same | none — native |
| Commodities (7) | Dukascopy underlying 1-min, 5 yr | Binance TradFi perp 1-min | ratio-adjust at the seam |
| Equities (30) | Dukascopy underlying 1-min (US-session), 5 yr | Binance TradFi perp 1-min | ratio-adjust at the seam |

Dukascopy codes below are **live-verified** (probed 200/404 on the datafeed). Notes:
- US stocks use the `{TICKER}USUSD` pattern; **META is the legacy `FBUSUSD`**; platinum/palladium are
  `XPT/XPDCMDUSD` (not `…USD`); QQQ/IWM/XLE exist as real ETF CFDs (`{TICKER}USUSD`).
- **SONY and NVO have NO Dukascopy USD proxy** (only Tokyo-JPY / Copenhagen-DKK) → `source: binance`,
  Binance-only short history, no synthetic backfill.
- **Stock splits are NOT adjusted in the Dukascopy CFD feed** — NVDA (2021-07 4:1, 2024-06 10:1),
  AVGO (2024-07 10:1), TSLA (2020-08 5:1, 2022-08 3:1), WMT (2024-02 3:1). The backfill back-adjusts
  pre-split bars by the cumulative split factor before stitching, else a split looks like a −90% crash.
- History floors: metals ~2003, commodities ~2011–12, equity CFDs ~2017 (UBER/PLTR ~2020-10) — the
  full 5-yr window (from 2021-06) is covered for every proxied name.
- Equity proxies are **US-session-only** (no overnight/weekend bars) — accepted as indicator-grade;
  the real 24/7 behaviour exists only in the post-listing Binance perp era.

## Crypto (50) — by sector
- **L1 majors (13):** BTC, ETH, BNB, SOL, XRP, ADA, AVAX, TRX, DOT, NEAR, APT, SUI, ATOM
- **Infra / oracle / DA / storage (7):** LINK, INJ, SEI, TIA, HBAR, ICP, FIL
- **Payments / classic (3):** BCH, LTC, XLM
- **L2 / scaling (3):** ARB, OP, POL
- **DeFi (8):** AAVE, UNI, CRV, PENDLE, ONDO, ENA, JUP, JTO
- **Memes (5):** DOGE, 1000PEPE, 1000SHIB, WIF, 1000BONK
- **AI (5):** TAO, FET, RENDER, VIRTUAL, WLD
- **Privacy (2):** XMR, ZEC
- **Perp-DEX / restaking (3):** HYPE, EIGEN, DYDX
- **Gaming (1):** AXS

## Equities (30) — by factor
- **Software / cloud (4):** MSFT, ORCL, CRM, ADBE
- **Internet / media (4):** GOOGL, META, NFLX, DIS
- **Semis — capped (3):** NVDA, TSM, AVGO
- **Hardware (2):** AAPL, SONY
- **Consumer discretionary (3):** AMZN, TSLA, HD
- **Consumer staples (2):** WMT, COST
- **Financials (3):** JPM, V, BRKB
- **Healthcare (2):** LLY, NVO
- **Mobility / data (2):** UBER, PLTR
- **Energy (1):** XLE
- **China (1):** BABA
- **IT services (1):** IBM
- **Index factors (2):** QQQ (large-tech), IWM (small-cap)

## Commodities / metals (7)
- **Precious metals (4):** XAU (gold), XAG (silver), XPT (platinum), XPD (palladium)
- **Industrial metal (1):** COPPER
- **Energy (2):** CL (WTI crude), NATGAS

## Loading into Python

```python
import json, re, pathlib
def load_universe(path="src/convexity/panel/candidates.md"):
    block = re.search(r"```json\n(.*?)\n```", pathlib.Path(path).read_text(), re.S).group(1)
    return json.loads(block)

u = load_universe()
crypto_symbols = [d["symbol"] for d in u["crypto"]]          # Binance perp symbols
tradfi = u["equities"] + u["commodities"]                     # each has binance + dukascopy codes
```

The block below is the machine-readable source of truth. `symbol` = Binance USDT-M perp symbol;
`dukascopy` = underlying-proxy code (verified by the backfill job); `source` = how it's fetched.

```json
{
  "meta": {
    "capital_usd": 1000,
    "resolution": "1m",
    "history_target_years": 5,
    "optimiser": "carver_ch25_dynamic",
    "crypto_filters_source": "fapi.binance.com/fapi/v1/exchangeInfo (LOT_SIZE/PRICE_FILTER/MIN_NOTIONAL), live snapshot",
    "crypto_filters_note": "tick_size/lot_size/min_qty/min_notional on each crypto entry are REAL per-symbol Binance USDT-M futures filters, not a placeholder. COMMODITY entries carry the same real filters since the RV pair paper book (2026-07-05); equities keep no tick/lot fields (dead sleeve, no live cross-sectional edge -- EDGE.md scope note). Filters are a CURRENT snapshot, not proven point-in-time-accurate across the backtest window -- see EXECUTION.md quantization section; live sessions re-quantize against the venue's current exchangeInfo anyway.",
    "commodity_perp_note": "Binance marks these underlyingType=COMMODITY / contractType=TRADIFI_PERPETUAL (the discovery predicate, scripts/discover_tradfi_perps.py); onboard = venue listing epoch-ms. Newly discovered venue commodity perps land in data/universe/discovered.json (gitignored overlay merged by load_universe) with data backfill -- NEVER auto-traded (the RV book's PAIRS are frozen research config)."
  },
  "crypto": [
    {
      "symbol": "BTCUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.1,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 50.0
    },
    {
      "symbol": "ETHUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 20.0
    },
    {
      "symbol": "BNBUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0
    },
    {
      "symbol": "SOLUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0
    },
    {
      "symbol": "XRPUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ADAUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "AVAXUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "TRXUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "DOTUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "NEARUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "APTUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "SUIUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ATOMUSDT",
      "sector": "L1-major",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0
    },
    {
      "symbol": "LINKUSDT",
      "sector": "infra-oracle",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 20.0
    },
    {
      "symbol": "INJUSDT",
      "sector": "infra-oracle",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "SEIUSDT",
      "sector": "infra-oracle",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "TIAUSDT",
      "sector": "infra-da",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "HBARUSDT",
      "sector": "infra",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ICPUSDT",
      "sector": "infra",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "FILUSDT",
      "sector": "storage",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "BCHUSDT",
      "sector": "payments",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 20.0
    },
    {
      "symbol": "LTCUSDT",
      "sector": "payments",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 20.0
    },
    {
      "symbol": "XLMUSDT",
      "sector": "payments",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ARBUSDT",
      "sector": "L2",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "OPUSDT",
      "sector": "L2",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "POLUSDT",
      "sector": "L2",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "AAVEUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "UNIUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "CRVUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "PENDLEUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ONDOUSDT",
      "sector": "defi-rwa",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ENAUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "JUPUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "JTOUSDT",
      "sector": "defi",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "DOGEUSDT",
      "sector": "meme",
      "source": "binance",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "1000PEPEUSDT",
      "sector": "meme",
      "source": "binance",
      "tick_size": 1e-07,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "1000SHIBUSDT",
      "sector": "meme",
      "source": "binance",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "WIFUSDT",
      "sector": "meme",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "1000BONKUSDT",
      "sector": "meme",
      "source": "binance",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "TAOUSDT",
      "sector": "ai",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0
    },
    {
      "symbol": "FETUSDT",
      "sector": "ai",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "RENDERUSDT",
      "sector": "ai",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "VIRTUALUSDT",
      "sector": "ai",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "WLDUSDT",
      "sector": "ai",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "XMRUSDT",
      "sector": "privacy",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0
    },
    {
      "symbol": "ZECUSDT",
      "sector": "privacy",
      "source": "binance",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0
    },
    {
      "symbol": "HYPEUSDT",
      "sector": "perp-dex",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0
    },
    {
      "symbol": "EIGENUSDT",
      "sector": "restaking",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "DYDXUSDT",
      "sector": "perp-dex",
      "source": "binance",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "AXSUSDT",
      "sector": "gaming",
      "source": "binance",
      "tick_size": 0.001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ETCUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 20.0
    },
    {
      "symbol": "DASHUSDT",
      "sector": "payments",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0
    },
    {
      "symbol": "ALGOUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "VETUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ONTUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "IOTAUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ZENUSDT",
      "sector": "privacy",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "BSVUSDT",
      "sector": "payments",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.01,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "KASUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ARUSDT",
      "sector": "storage",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "GLMUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "PYTHUSDT",
      "sector": "infra-oracle",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "QNTUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.01,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "GRTUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "BICOUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "POWRUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "NMRUSDT",
      "sector": "ai",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "LDOUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "SNXUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "SUSHIUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "CAKEUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ETHFIUSDT",
      "sector": "restaking",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "STGUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "DODOXUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "MAGICUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "GALAUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "SANDUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "APEUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "AGLDUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "PORTALUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "STRKUSDT",
      "sector": "L2",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "MANTAUSDT",
      "sector": "L2",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ZROUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "WUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "IOUSDT",
      "sector": "ai",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "TNSRUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "ORDIUSDT",
      "sector": "meme",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "MEMEUSDT",
      "sector": "meme",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-07,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "TURBOUSDT",
      "sector": "meme",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-07,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "1000LUNCUSDT",
      "sector": "meme",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "CHZUSDT",
      "sector": "gaming",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "CELOUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "RIFUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "NFPUSDT",
      "sector": "ai",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-06,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "IDUSDT",
      "sector": "infra",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "ONGUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "BELUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-05,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "SFPUSDT",
      "sector": "defi",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.0001,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    },
    {
      "symbol": "GASUSDT",
      "sector": "L1-major",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0
    },
    {
      "symbol": "GUSDT",
      "sector": "ai",
      "source": "binance",
      "batch": "expand50",
      "tick_size": 1e-06,
      "lot_size": 1.0,
      "min_qty": 1.0,
      "min_notional": 5.0
    }
  ],
  "equities": [
    {
      "symbol": "AAOIUSDT",
      "sector": "optical",
      "source": "databento+binance",
      "dukascopy": "AAOIDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780910700000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AAOI"
    },
    {
      "symbol": "AAPLUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "AAPLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775483400000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AAPL"
    },
    {
      "symbol": "ADBEUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "ADBEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781169900000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ADBE"
    },
    {
      "symbol": "ALABUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "ALABDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782135600000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ALAB"
    },
    {
      "symbol": "AMATUSDT",
      "sector": "semicap",
      "source": "databento+binance",
      "dukascopy": "AMATDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780909800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AMAT"
    },
    {
      "symbol": "AMDUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "AMDDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778074200000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AMD"
    },
    {
      "symbol": "AMZNUSDT",
      "sector": "consumer-disc",
      "source": "databento+binance",
      "dukascopy": "AMZNDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770648000000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AMZN"
    },
    {
      "symbol": "ANTHROPICUSDT",
      "sector": "pre-ipo-ai",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780374600000,
      "on_demo": false
    },
    {
      "symbol": "ARMUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "ARMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779802800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ARM"
    },
    {
      "symbol": "ASMLUSDT",
      "sector": "semicap",
      "source": "databento+binance",
      "dukascopy": "ASMLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781170800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ASML"
    },
    {
      "symbol": "ASTSUSDT",
      "sector": "space",
      "source": "databento+binance",
      "dukascopy": "ASTSDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780322100000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ASTS"
    },
    {
      "symbol": "AVGOUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "AVGODBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1776692400000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AVGO"
    },
    {
      "symbol": "AXTIUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "AXTIDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780911300000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "AXTI"
    },
    {
      "symbol": "BABAUSDT",
      "sector": "china",
      "source": "databento+binance",
      "dukascopy": "BABADBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1776693000000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "BABA"
    },
    {
      "symbol": "BBXUSDT",
      "sector": "thematic",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780321200000,
      "on_demo": false
    },
    {
      "symbol": "BEUSDT",
      "sector": "clean-energy",
      "source": "databento+binance",
      "dukascopy": "BEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779803100000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "BE"
    },
    {
      "symbol": "BMNRUSDT",
      "sector": "bitcoin-proxy",
      "source": "databento+binance",
      "dukascopy": "BMNRDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781169300000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "BMNR"
    },
    {
      "symbol": "BRKBUSDT",
      "sector": "financials",
      "source": "databento+binance",
      "dukascopy": "BRKBDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779112200000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "BRK B"
    },
    {
      "symbol": "BSPUSDT",
      "sector": "thematic",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782985800000,
      "on_demo": false
    },
    {
      "symbol": "BXUSDT",
      "sector": "financials",
      "source": "databento+binance",
      "dukascopy": "BXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780909200000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "BX"
    },
    {
      "symbol": "CATUSDT",
      "sector": "industrials",
      "source": "databento+binance",
      "dukascopy": "CATDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782984000000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "CAT"
    },
    {
      "symbol": "CBRSUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "CBRSDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779183000000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "CBRS"
    },
    {
      "symbol": "CIENUSDT",
      "sector": "networking",
      "source": "databento+binance",
      "dukascopy": "CIENDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782136200000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "CIEN"
    },
    {
      "symbol": "COHRUSDT",
      "sector": "optical",
      "source": "databento+binance",
      "dukascopy": "COHRDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779803400000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "COHR"
    },
    {
      "symbol": "COINUSDT",
      "sector": "crypto-exchange",
      "source": "databento+binance",
      "dukascopy": "COINDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770649200000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "COIN"
    },
    {
      "symbol": "COSTUSDT",
      "sector": "staples",
      "source": "databento+binance",
      "dukascopy": "COSTDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780995900000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "COST"
    },
    {
      "symbol": "CRCLUSDT",
      "sector": "stablecoin",
      "source": "databento+binance",
      "dukascopy": "CRCLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770648600000,
      "on_demo": true,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "CRCL"
    },
    {
      "symbol": "CRDOUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "CRDODBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780910400000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "CRDO"
    },
    {
      "symbol": "CRMUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "CRMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780478100000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "CRM"
    },
    {
      "symbol": "CRWDUSDT",
      "sector": "software-security",
      "source": "databento+binance",
      "dukascopy": "CRWDDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780910100000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "CRWD"
    },
    {
      "symbol": "CRWVUSDT",
      "sector": "ai-cloud",
      "source": "databento+binance",
      "dukascopy": "CRWVDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779111000000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "CRWV"
    },
    {
      "symbol": "CSCOUSDT",
      "sector": "networking",
      "source": "databento+binance",
      "dukascopy": "CSCODBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778853000000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "CSCO"
    },
    {
      "symbol": "DELLUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "DELLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780477200000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "DELL"
    },
    {
      "symbol": "DISUSDT",
      "sector": "media",
      "source": "databento+binance",
      "dukascopy": "DISDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778852400000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "DIS"
    },
    {
      "symbol": "DKNGUSDT",
      "sector": "gaming-betting",
      "source": "databento+binance",
      "dukascopy": "DKNGDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781082300000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "DKNG"
    },
    {
      "symbol": "DRAMUSDT",
      "sector": "semis-thematic",
      "source": "databento+binance",
      "dukascopy": "DRAMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779112800000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "DRAM"
    },
    {
      "symbol": "EBAYUSDT",
      "sector": "internet",
      "source": "databento+binance",
      "dukascopy": "EBAYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780996800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "EBAY"
    },
    {
      "symbol": "EWJUSDT",
      "sector": "country-etf-japan",
      "source": "databento+binance",
      "dukascopy": "EWJDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1773927000000,
      "on_demo": true,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "EWJ"
    },
    {
      "symbol": "EWTUSDT",
      "sector": "country-etf-taiwan",
      "source": "databento+binance",
      "dukascopy": "EWTDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780321800000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "EWT"
    },
    {
      "symbol": "EWYUSDT",
      "sector": "country-etf-korea",
      "source": "databento+binance",
      "dukascopy": "EWYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1773667800000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "EWY"
    },
    {
      "symbol": "EWZUSDT",
      "sector": "country-etf-brazil",
      "source": "databento+binance",
      "dukascopy": "EWZDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781083500000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "EWZ"
    },
    {
      "symbol": "FLEXUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "FLEXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782984600000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "FLEX"
    },
    {
      "symbol": "FLNCUSDT",
      "sector": "energy-storage",
      "source": "databento+binance",
      "dukascopy": "FLNCDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779112500000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "FLNC"
    },
    {
      "symbol": "GLWUSDT",
      "sector": "optical",
      "source": "databento+binance",
      "dukascopy": "GLWDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781170200000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "GLW"
    },
    {
      "symbol": "GMEUSDT",
      "sector": "retail-meme",
      "source": "databento+binance",
      "dukascopy": "GMEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781082900000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "GME"
    },
    {
      "symbol": "GOOGLUSDT",
      "sector": "internet",
      "source": "databento+binance",
      "dukascopy": "GOOGLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1774536600000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "GOOGL"
    },
    {
      "symbol": "HDUSDT",
      "sector": "consumer-disc",
      "source": "databento+binance",
      "dukascopy": "HDDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778853300000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "HD"
    },
    {
      "symbol": "HIMSUSDT",
      "sector": "health-consumer",
      "source": "databento+binance",
      "dukascopy": "HIMSDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780996500000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "HIMS"
    },
    {
      "symbol": "HOODUSDT",
      "sector": "fintech",
      "source": "databento+binance",
      "dukascopy": "HOODDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770043500000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "HOOD"
    },
    {
      "symbol": "HPEUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "HPEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780909500000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "HPE"
    },
    {
      "symbol": "HYUNDAIUSDT",
      "sector": "kr-equity",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780370400000,
      "on_demo": false
    },
    {
      "symbol": "IBMUSDT",
      "sector": "it-services",
      "source": "databento+binance",
      "dukascopy": "IBMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780477500000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "IBM"
    },
    {
      "symbol": "INTCUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "INTCDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770042600000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "INTC"
    },
    {
      "symbol": "IRENUSDT",
      "sector": "ai-crypto-infra",
      "source": "databento+binance",
      "dukascopy": "IRENDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780478400000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "IREN"
    },
    {
      "symbol": "IWMUSDT",
      "sector": "index-smallcap",
      "source": "databento+binance",
      "dukascopy": "IWMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780911000000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "IWM"
    },
    {
      "symbol": "JPMUSDT",
      "sector": "financials",
      "source": "databento+binance",
      "dukascopy": "JPMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779111600000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "JPM"
    },
    {
      "symbol": "KLACUSDT",
      "sector": "semicap",
      "source": "databento+binance",
      "dukascopy": "KLACDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782135300000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "KLAC"
    },
    {
      "symbol": "KORUUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "KORUDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782136500000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "KORU"
    },
    {
      "symbol": "KSTRUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "KSTRDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782985500000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "KSTR"
    },
    {
      "symbol": "LITEUSDT",
      "sector": "optical",
      "source": "databento+binance",
      "dukascopy": "LITEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778851800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "LITE"
    },
    {
      "symbol": "LLYUSDT",
      "sector": "healthcare",
      "source": "databento+binance",
      "dukascopy": "LLYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780320600000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "LLY"
    },
    {
      "symbol": "LRCXUSDT",
      "sector": "semicap",
      "source": "databento+binance",
      "dukascopy": "LRCXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782135000000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "LRCX"
    },
    {
      "symbol": "METAUSDT",
      "sector": "internet",
      "source": "databento+binance",
      "dukascopy": "METADBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1774535400000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "META"
    },
    {
      "symbol": "MRVLUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "MRVLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778853900000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "MRVL"
    },
    {
      "symbol": "MSFTUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "MSFTDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1776691800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "MSFT"
    },
    {
      "symbol": "MSTRUSDT",
      "sector": "bitcoin-proxy",
      "source": "databento+binance",
      "dukascopy": "MSTRDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770647400000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "MSTR"
    },
    {
      "symbol": "MUUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "MUDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775568600000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "MU"
    },
    {
      "symbol": "MVLLUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "MVLLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782740100000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "MVLL"
    },
    {
      "symbol": "NBISUSDT",
      "sector": "ai-cloud",
      "source": "databento+binance",
      "dukascopy": "NBISDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779802200000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "NBIS"
    },
    {
      "symbol": "NFLXUSDT",
      "sector": "media",
      "source": "databento+binance",
      "dukascopy": "NFLXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780995600000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "NFLX"
    },
    {
      "symbol": "NOKUSDT",
      "sector": "telecom",
      "source": "databento+binance",
      "dukascopy": "NOKDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780321500000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "NOK"
    },
    {
      "symbol": "NOWUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "NOWDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780477800000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "NOW"
    },
    {
      "symbol": "NVDAUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "NVDADBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1774536000000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "NVDA"
    },
    {
      "symbol": "NVOUSDT",
      "sector": "healthcare",
      "source": "databento+binance",
      "dukascopy": "NVODBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780320900000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "NVO"
    },
    {
      "symbol": "ONDSUSDT",
      "sector": "drones",
      "source": "databento+binance",
      "dukascopy": "ONDSDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780478700000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ONDS"
    },
    {
      "symbol": "OPENAIUSDT",
      "sector": "pre-ipo-ai",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779784200000,
      "on_demo": true
    },
    {
      "symbol": "ORCLUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "ORCLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778852100000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "ORCL"
    },
    {
      "symbol": "PAYPUSDT",
      "sector": "fintech",
      "source": "databento+binance",
      "dukascopy": "PYPLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1774276200000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "PYPL"
    },
    {
      "symbol": "PLTRUSDT",
      "sector": "data-ai",
      "source": "databento+binance",
      "dukascopy": "PLTRDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1770649800000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "PLTR"
    },
    {
      "symbol": "QCOMUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "QCOMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778074500000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "QCOM"
    },
    {
      "symbol": "QNTXUSDT",
      "sector": "quantum-thematic",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780042500000,
      "on_demo": true
    },
    {
      "symbol": "QQQUSDT",
      "sector": "index-largetech",
      "source": "databento+binance",
      "dukascopy": "QQQDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775482200000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "QQQ"
    },
    {
      "symbol": "RIVNUSDT",
      "sector": "ev",
      "source": "databento+binance",
      "dukascopy": "RIVNDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781082600000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "RIVN"
    },
    {
      "symbol": "RKLBUSDT",
      "sector": "space",
      "source": "databento+binance",
      "dukascopy": "RKLBDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779113100000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "RKLB"
    },
    {
      "symbol": "SAMSUNGUSDT",
      "sector": "kr-equity",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780369800000,
      "on_demo": false
    },
    {
      "symbol": "SKHYNIXUSDT",
      "sector": "kr-equity",
      "source": "binance",
      "dukascopy": null,
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780369200000,
      "on_demo": false
    },
    {
      "symbol": "SMCIUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "SMCIDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782135900000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "SMCI"
    },
    {
      "symbol": "SNDKUSDT",
      "sector": "semis-storage",
      "source": "databento+binance",
      "dukascopy": "SNDKDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775569200000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "SNDK"
    },
    {
      "symbol": "SONYUSDT",
      "sector": "hardware",
      "source": "databento+binance",
      "dukascopy": "SONYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782136800000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "SONY"
    },
    {
      "symbol": "SOXLUSDT",
      "sector": "leveraged-etf-semis",
      "source": "databento+binance",
      "dukascopy": "SOXLDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778853600000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "SOXL"
    },
    {
      "symbol": "SPCXUSDT",
      "sector": "space-proxy",
      "source": "databento+binance",
      "dukascopy": "SPCXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779338700000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "SPCX"
    },
    {
      "symbol": "SPYUSDT",
      "sector": "index-sp500",
      "source": "databento+binance",
      "dukascopy": "SPYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775482800000,
      "on_demo": true,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "SPY"
    },
    {
      "symbol": "SQQQUSDT",
      "sector": "inverse-etf-nasdaq",
      "source": "databento+binance",
      "dukascopy": "SQQQDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782740700000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "SQQQ"
    },
    {
      "symbol": "STRCUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "STRCDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782983700000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "STRC"
    },
    {
      "symbol": "STXXUSDT",
      "sector": "thematic",
      "source": "databento+binance",
      "dukascopy": "STXXDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781170500000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "STXX"
    },
    {
      "symbol": "TERUSDT",
      "sector": "semicap",
      "source": "databento+binance",
      "dukascopy": "TERDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782984900000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "TER"
    },
    {
      "symbol": "TQQQUSDT",
      "sector": "leveraged-etf-nasdaq",
      "source": "databento+binance",
      "dukascopy": "TQQQDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782740400000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "TQQQ"
    },
    {
      "symbol": "TSLAUSDT",
      "sector": "consumer-disc",
      "source": "databento+binance",
      "dukascopy": "TSLADBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1769610600000,
      "on_demo": true,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "TSLA"
    },
    {
      "symbol": "TSMUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "TSMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775484000000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "TSM"
    },
    {
      "symbol": "TTWOUSDT",
      "sector": "gaming",
      "source": "databento+binance",
      "dukascopy": "TTWODBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782985200000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "TTWO"
    },
    {
      "symbol": "TXNUSDT",
      "sector": "semis",
      "source": "databento+binance",
      "dukascopy": "TXNDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1782984300000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "TXN"
    },
    {
      "symbol": "UBERUSDT",
      "sector": "mobility",
      "source": "databento+binance",
      "dukascopy": "UBERDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778852700000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "UBER"
    },
    {
      "symbol": "URNMUSDT",
      "sector": "uranium-etf",
      "source": "databento+binance",
      "dukascopy": "URNMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1780996200000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "URNM"
    },
    {
      "symbol": "USARUSDT",
      "sector": "rare-earth",
      "source": "databento+binance",
      "dukascopy": "USARDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1778074800000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "USAR"
    },
    {
      "symbol": "UVXYUSDT",
      "sector": "vol-etf",
      "source": "databento+binance",
      "dukascopy": "UVXYDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781169600000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "UVXY"
    },
    {
      "symbol": "VUSDT",
      "sector": "financials",
      "source": "databento+binance",
      "dukascopy": "VDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779111900000,
      "on_demo": false,
      "databento_venue": "XNYS.PILLAR",
      "databento_ticker": "V"
    },
    {
      "symbol": "WDCUSDT",
      "sector": "semis-storage",
      "source": "databento+binance",
      "dukascopy": "WDCDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779802500000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "WDC"
    },
    {
      "symbol": "WMTUSDT",
      "sector": "staples",
      "source": "databento+binance",
      "dukascopy": "WMTDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1779111300000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "WMT"
    },
    {
      "symbol": "XLEUSDT",
      "sector": "energy-etf",
      "source": "databento+binance",
      "dukascopy": "XLEDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781083200000,
      "on_demo": false,
      "databento_venue": "ARCX.PILLAR",
      "databento_ticker": "XLE"
    },
    {
      "symbol": "ZMUSDT",
      "sector": "software",
      "source": "databento+binance",
      "dukascopy": "ZMDBN",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1781082000000,
      "on_demo": false,
      "databento_venue": "XNAS.ITCH",
      "databento_ticker": "ZM"
    }
  ],
  "commodities": [
    {
      "symbol": "XAUUSDT",
      "sector": "precious-metal",
      "source": "dukascopy+binance",
      "dukascopy": "XAUUSD",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0,
      "onboard": 1765440300000
    },
    {
      "symbol": "XAGUSDT",
      "sector": "precious-metal",
      "source": "dukascopy+binance",
      "dukascopy": "XAGUSD",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0,
      "onboard": 1767780000000
    },
    {
      "symbol": "XPTUSDT",
      "sector": "precious-metal",
      "source": "dukascopy+binance",
      "dukascopy": "XPTCMDUSD",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0,
      "onboard": 1769767200000
    },
    {
      "symbol": "XPDUSDT",
      "sector": "precious-metal",
      "source": "dukascopy+binance",
      "dukascopy": "XPDCMDUSD",
      "tick_size": 0.01,
      "lot_size": 0.001,
      "min_qty": 0.001,
      "min_notional": 5.0,
      "onboard": 1769768100000
    },
    {
      "symbol": "COPPERUSDT",
      "sector": "industrial-metal",
      "source": "dukascopy+binance",
      "dukascopy": "COPPERCMDUSD",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0,
      "onboard": 1772787600000
    },
    {
      "symbol": "CLUSDT",
      "sector": "energy",
      "source": "dukascopy+binance",
      "dukascopy": "LIGHTCMDUSD",
      "tick_size": 0.01,
      "lot_size": 0.01,
      "min_qty": 0.01,
      "min_notional": 5.0,
      "onboard": 1775034000000
    },
    {
      "symbol": "NATGASUSDT",
      "sector": "energy",
      "source": "dukascopy+binance",
      "dukascopy": "GASCMDUSD",
      "tick_size": 0.001,
      "lot_size": 0.1,
      "min_qty": 0.1,
      "min_notional": 5.0,
      "onboard": 1775035200000
    }
  ]
}
```
