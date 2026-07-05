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
  "meta": {"capital_usd": 1000, "resolution": "1m", "history_target_years": 5, "optimiser": "carver_ch25_dynamic",
    "crypto_filters_source": "fapi.binance.com/fapi/v1/exchangeInfo (LOT_SIZE/PRICE_FILTER/MIN_NOTIONAL), live snapshot",
    "crypto_filters_note": "tick_size/lot_size/min_qty/min_notional on each crypto entry are REAL per-symbol Binance USDT-M futures filters, not a placeholder. COMMODITY entries carry the same real filters since the RV pair paper book (2026-07-05); equities keep no tick/lot fields (dead sleeve, no live cross-sectional edge -- EDGE.md scope note). Filters are a CURRENT snapshot, not proven point-in-time-accurate across the backtest window -- see EXECUTION.md quantization section; live sessions re-quantize against the venue's current exchangeInfo anyway.",
    "commodity_perp_note": "Binance marks these underlyingType=COMMODITY / contractType=TRADIFI_PERPETUAL (the discovery predicate, scripts/discover_tradfi_perps.py); onboard = venue listing epoch-ms. Newly discovered venue commodity perps land in data/universe/discovered.json (gitignored overlay merged by load_universe) with data backfill -- NEVER auto-traded (the RV book's PAIRS are frozen research config)."},
  "crypto": [
    {"symbol": "BTCUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.1, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 50.0},
    {"symbol": "ETHUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 20.0},
    {"symbol": "BNBUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.01, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 5.0},
    {"symbol": "SOLUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.01, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 5.0},
    {"symbol": "XRPUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ADAUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "AVAXUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "TRXUSDT", "sector": "L1-major", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "DOTUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "NEARUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "APTUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "SUIUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ATOMUSDT", "sector": "L1-major", "source": "binance", "tick_size": 0.001, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 5.0},
    {"symbol": "LINKUSDT", "sector": "infra-oracle", "source": "binance", "tick_size": 0.001, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 20.0},
    {"symbol": "INJUSDT", "sector": "infra-oracle", "source": "binance", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "SEIUSDT", "sector": "infra-oracle", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "TIAUSDT", "sector": "infra-da", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "HBARUSDT", "sector": "infra", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ICPUSDT", "sector": "infra", "source": "binance", "tick_size": 0.001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "FILUSDT", "sector": "storage", "source": "binance", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "BCHUSDT", "sector": "payments", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 20.0},
    {"symbol": "LTCUSDT", "sector": "payments", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 20.0},
    {"symbol": "XLMUSDT", "sector": "payments", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ARBUSDT", "sector": "L2", "source": "binance", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "OPUSDT", "sector": "L2", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "POLUSDT", "sector": "L2", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "AAVEUSDT", "sector": "defi", "source": "binance", "tick_size": 0.01, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "UNIUSDT", "sector": "defi", "source": "binance", "tick_size": 0.001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "CRVUSDT", "sector": "defi", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "PENDLEUSDT", "sector": "defi", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ONDOUSDT", "sector": "defi-rwa", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ENAUSDT", "sector": "defi", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "JUPUSDT", "sector": "defi", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "JTOUSDT", "sector": "defi", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "DOGEUSDT", "sector": "meme", "source": "binance", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "1000PEPEUSDT", "sector": "meme", "source": "binance", "tick_size": 1e-07, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "1000SHIBUSDT", "sector": "meme", "source": "binance", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "WIFUSDT", "sector": "meme", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "1000BONKUSDT", "sector": "meme", "source": "binance", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "TAOUSDT", "sector": "ai", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0},
    {"symbol": "FETUSDT", "sector": "ai", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "RENDERUSDT", "sector": "ai", "source": "binance", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "VIRTUALUSDT", "sector": "ai", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "WLDUSDT", "sector": "ai", "source": "binance", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "XMRUSDT", "sector": "privacy", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0},
    {"symbol": "ZECUSDT", "sector": "privacy", "source": "binance", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0},
    {"symbol": "HYPEUSDT", "sector": "perp-dex", "source": "binance", "tick_size": 0.001, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 5.0},
    {"symbol": "EIGENUSDT", "sector": "restaking", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "DYDXUSDT", "sector": "perp-dex", "source": "binance", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "AXSUSDT", "sector": "gaming", "source": "binance", "tick_size": 0.001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ETCUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 20.0},
    {"symbol": "DASHUSDT", "sector": "payments", "source": "binance", "batch": "expand50", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0},
    {"symbol": "ALGOUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "VETUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ONTUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "IOTAUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ZENUSDT", "sector": "privacy", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "BSVUSDT", "sector": "payments", "source": "binance", "batch": "expand50", "tick_size": 0.01, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "KASUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ARUSDT", "sector": "storage", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "GLMUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "PYTHUSDT", "sector": "infra-oracle", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "QNTUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 0.01, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "GRTUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "BICOUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "POWRUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "NMRUSDT", "sector": "ai", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "LDOUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "SNXUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "SUSHIUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "CAKEUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ETHFIUSDT", "sector": "restaking", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "STGUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "DODOXUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "MAGICUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "GALAUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "SANDUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "APEUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "AGLDUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "PORTALUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "STRKUSDT", "sector": "L2", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "MANTAUSDT", "sector": "L2", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ZROUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "WUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "IOUSDT", "sector": "ai", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "TNSRUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "ORDIUSDT", "sector": "meme", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "MEMEUSDT", "sector": "meme", "source": "binance", "batch": "expand50", "tick_size": 1e-07, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "TURBOUSDT", "sector": "meme", "source": "binance", "batch": "expand50", "tick_size": 1e-07, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "1000LUNCUSDT", "sector": "meme", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "CHZUSDT", "sector": "gaming", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "CELOUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "RIFUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "NFPUSDT", "sector": "ai", "source": "binance", "batch": "expand50", "tick_size": 1e-06, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "IDUSDT", "sector": "infra", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "ONGUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "BELUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 1e-05, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "SFPUSDT", "sector": "defi", "source": "binance", "batch": "expand50", "tick_size": 0.0001, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0},
    {"symbol": "GASUSDT", "sector": "L1-major", "source": "binance", "batch": "expand50", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0},
    {"symbol": "GUSDT", "sector": "ai", "source": "binance", "batch": "expand50", "tick_size": 1e-06, "lot_size": 1.0, "min_qty": 1.0, "min_notional": 5.0}
  ],
  "equities": [
    {"symbol": "MSFTUSDT", "sector": "software", "source": "dukascopy+binance", "dukascopy": "MSFTUSUSD"},
    {"symbol": "ORCLUSDT", "sector": "software", "source": "dukascopy+binance", "dukascopy": "ORCLUSUSD"},
    {"symbol": "CRMUSDT", "sector": "software", "source": "dukascopy+binance", "dukascopy": "CRMUSUSD"},
    {"symbol": "ADBEUSDT", "sector": "software", "source": "dukascopy+binance", "dukascopy": "ADBEUSUSD"},
    {"symbol": "GOOGLUSDT", "sector": "internet", "source": "dukascopy+binance", "dukascopy": "GOOGLUSUSD"},
    {"symbol": "METAUSDT", "sector": "internet", "source": "dukascopy+binance", "dukascopy": "FBUSUSD"},
    {"symbol": "NFLXUSDT", "sector": "media", "source": "dukascopy+binance", "dukascopy": "NFLXUSUSD"},
    {"symbol": "DISUSDT", "sector": "media", "source": "dukascopy+binance", "dukascopy": "DISUSUSD"},
    {"symbol": "NVDAUSDT", "sector": "semis", "source": "dukascopy+binance", "dukascopy": "NVDAUSUSD"},
    {"symbol": "TSMUSDT", "sector": "semis", "source": "dukascopy+binance", "dukascopy": "TSMUSUSD"},
    {"symbol": "AVGOUSDT", "sector": "semis", "source": "dukascopy+binance", "dukascopy": "AVGOUSUSD"},
    {"symbol": "AAPLUSDT", "sector": "hardware", "source": "dukascopy+binance", "dukascopy": "AAPLUSUSD"},
    {"symbol": "SONYUSDT", "sector": "hardware", "source": "binance", "dukascopy": null, "note": "no Dukascopy USD proxy (only Tokyo 6758JPJPY) -> Binance-only, no synthetic history"},
    {"symbol": "AMZNUSDT", "sector": "consumer-disc", "source": "dukascopy+binance", "dukascopy": "AMZNUSUSD"},
    {"symbol": "TSLAUSDT", "sector": "consumer-disc", "source": "dukascopy+binance", "dukascopy": "TSLAUSUSD"},
    {"symbol": "HDUSDT", "sector": "consumer-disc", "source": "dukascopy+binance", "dukascopy": "HDUSUSD"},
    {"symbol": "WMTUSDT", "sector": "staples", "source": "dukascopy+binance", "dukascopy": "WMTUSUSD"},
    {"symbol": "COSTUSDT", "sector": "staples", "source": "dukascopy+binance", "dukascopy": "COSTUSUSD"},
    {"symbol": "JPMUSDT", "sector": "financials", "source": "dukascopy+binance", "dukascopy": "JPMUSUSD"},
    {"symbol": "VUSDT", "sector": "financials", "source": "dukascopy+binance", "dukascopy": "VUSUSD"},
    {"symbol": "BRKBUSDT", "sector": "financials", "source": "dukascopy+binance", "dukascopy": "BRKBUSUSD"},
    {"symbol": "LLYUSDT", "sector": "healthcare", "source": "dukascopy+binance", "dukascopy": "LLYUSUSD"},
    {"symbol": "NVOUSDT", "sector": "healthcare", "source": "binance", "dukascopy": null, "note": "no Dukascopy USD proxy (only Copenhagen NOVOBDKDKK) -> Binance-only, no synthetic history"},
    {"symbol": "UBERUSDT", "sector": "mobility", "source": "dukascopy+binance", "dukascopy": "UBERUSUSD"},
    {"symbol": "PLTRUSDT", "sector": "data-ai", "source": "dukascopy+binance", "dukascopy": "PLTRUSUSD"},
    {"symbol": "XLEUSDT", "sector": "energy-etf", "source": "dukascopy+binance", "dukascopy": "XLEUSUSD"},
    {"symbol": "BABAUSDT", "sector": "china", "source": "dukascopy+binance", "dukascopy": "BABAUSUSD"},
    {"symbol": "IBMUSDT", "sector": "it-services", "source": "dukascopy+binance", "dukascopy": "IBMUSUSD"},
    {"symbol": "QQQUSDT", "sector": "index-largetech", "source": "dukascopy+binance", "dukascopy": "QQQUSUSD"},
    {"symbol": "IWMUSDT", "sector": "index-smallcap", "source": "dukascopy+binance", "dukascopy": "IWMUSUSD"}
  ],
  "commodities": [
    {"symbol": "XAUUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XAUUSD", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0, "onboard": 1765440300000},
    {"symbol": "XAGUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XAGUSD", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0, "onboard": 1767780000000},
    {"symbol": "XPTUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XPTCMDUSD", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0, "onboard": 1769767200000},
    {"symbol": "XPDUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XPDCMDUSD", "tick_size": 0.01, "lot_size": 0.001, "min_qty": 0.001, "min_notional": 5.0, "onboard": 1769768100000},
    {"symbol": "COPPERUSDT", "sector": "industrial-metal", "source": "dukascopy+binance", "dukascopy": "COPPERCMDUSD", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0, "onboard": 1772787600000},
    {"symbol": "CLUSDT", "sector": "energy", "source": "dukascopy+binance", "dukascopy": "LIGHTCMDUSD", "tick_size": 0.01, "lot_size": 0.01, "min_qty": 0.01, "min_notional": 5.0, "onboard": 1775034000000},
    {"symbol": "NATGASUSDT", "sector": "energy", "source": "dukascopy+binance", "dukascopy": "GASCMDUSD", "tick_size": 0.001, "lot_size": 0.1, "min_qty": 0.1, "min_notional": 5.0, "onboard": 1775035200000}
  ]
}
```
