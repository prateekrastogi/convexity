# convexity trading universe

The candidate universe fed to the Carver dynamic optimiser (`dynamic_optimization.py`). Target
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
def load_universe(path="src/convexity/selection/UNIVERSE.md"):
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
  "meta": {"capital_usd": 1000, "resolution": "1m", "history_target_years": 5, "optimiser": "carver_ch25_dynamic"},
  "crypto": [
    {"symbol": "BTCUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "ETHUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "BNBUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "SOLUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "XRPUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "ADAUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "AVAXUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "TRXUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "DOTUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "NEARUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "APTUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "SUIUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "ATOMUSDT", "sector": "L1-major", "source": "binance"},
    {"symbol": "LINKUSDT", "sector": "infra-oracle", "source": "binance"},
    {"symbol": "INJUSDT", "sector": "infra-oracle", "source": "binance"},
    {"symbol": "SEIUSDT", "sector": "infra-oracle", "source": "binance"},
    {"symbol": "TIAUSDT", "sector": "infra-da", "source": "binance"},
    {"symbol": "HBARUSDT", "sector": "infra", "source": "binance"},
    {"symbol": "ICPUSDT", "sector": "infra", "source": "binance"},
    {"symbol": "FILUSDT", "sector": "storage", "source": "binance"},
    {"symbol": "BCHUSDT", "sector": "payments", "source": "binance"},
    {"symbol": "LTCUSDT", "sector": "payments", "source": "binance"},
    {"symbol": "XLMUSDT", "sector": "payments", "source": "binance"},
    {"symbol": "ARBUSDT", "sector": "L2", "source": "binance"},
    {"symbol": "OPUSDT", "sector": "L2", "source": "binance"},
    {"symbol": "POLUSDT", "sector": "L2", "source": "binance"},
    {"symbol": "AAVEUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "UNIUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "CRVUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "PENDLEUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "ONDOUSDT", "sector": "defi-rwa", "source": "binance"},
    {"symbol": "ENAUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "JUPUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "JTOUSDT", "sector": "defi", "source": "binance"},
    {"symbol": "DOGEUSDT", "sector": "meme", "source": "binance"},
    {"symbol": "1000PEPEUSDT", "sector": "meme", "source": "binance"},
    {"symbol": "1000SHIBUSDT", "sector": "meme", "source": "binance"},
    {"symbol": "WIFUSDT", "sector": "meme", "source": "binance"},
    {"symbol": "1000BONKUSDT", "sector": "meme", "source": "binance"},
    {"symbol": "TAOUSDT", "sector": "ai", "source": "binance"},
    {"symbol": "FETUSDT", "sector": "ai", "source": "binance"},
    {"symbol": "RENDERUSDT", "sector": "ai", "source": "binance"},
    {"symbol": "VIRTUALUSDT", "sector": "ai", "source": "binance"},
    {"symbol": "WLDUSDT", "sector": "ai", "source": "binance"},
    {"symbol": "XMRUSDT", "sector": "privacy", "source": "binance"},
    {"symbol": "ZECUSDT", "sector": "privacy", "source": "binance"},
    {"symbol": "HYPEUSDT", "sector": "perp-dex", "source": "binance"},
    {"symbol": "EIGENUSDT", "sector": "restaking", "source": "binance"},
    {"symbol": "DYDXUSDT", "sector": "perp-dex", "source": "binance"},
    {"symbol": "AXSUSDT", "sector": "gaming", "source": "binance"}
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
    {"symbol": "XAUUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XAUUSD"},
    {"symbol": "XAGUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XAGUSD"},
    {"symbol": "XPTUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XPTCMDUSD"},
    {"symbol": "XPDUSDT", "sector": "precious-metal", "source": "dukascopy+binance", "dukascopy": "XPDCMDUSD"},
    {"symbol": "COPPERUSDT", "sector": "industrial-metal", "source": "dukascopy+binance", "dukascopy": "COPPERCMDUSD"},
    {"symbol": "CLUSDT", "sector": "energy", "source": "dukascopy+binance", "dukascopy": "LIGHTCMDUSD"},
    {"symbol": "NATGASUSDT", "sector": "energy", "source": "dukascopy+binance", "dukascopy": "GASCMDUSD"}
  ]
}
```
