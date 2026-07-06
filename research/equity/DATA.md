# Equity data lake — sources, spans, data-quality investigation (equity branch, 2026-07-06)

DATA ACQUISITION ONLY. This branch extends the convexity lake to **every Binance TradFi EQUITY
perp** (+ KR equities + pre-IPO) and their real primary-listing-venue underlying history, for a
LATER, separately pre-registered equity study. No strategy code, no strategy preregistration, no
backtests here — only the data pipeline and its verification.

## Universe (census 2026-07-06, prod `fapi/v1/exchangeInfo`)

110 TradFi non-commodity perps, `contractType=TRADIFI_PERPETUAL`, status TRADING:
- **105 EQUITY** (incl. 1 ETF subtype EWJ; plus SPY/QQQ/IWM/XLE/SOXL/TQQQ/SQQQ/UVXY/EWT/EWY/EWZ)
- **3 KR_EQUITY** (SAMSUNG, SKHYNIX, HYUNDAI)
- **2 PREMARKET pre-IPO** (OPENAI, ANTHROPIC)

30 were already in the universe (the seasoned mega-caps); 80 are new. Real venue filters
(tick/lot/min_qty/min_notional), `onboard` epoch, and `on_demo` (demo-venue availability) are
recorded per symbol in `src/convexity/panel/candidates.md`.

## Source: databento, not Dukascopy (decision made mid-build, see below)

The original plan was a Dukascopy candle-mode backfill (as used for commodities). That fetch was
started, found to be **throttle-bound** (~9s/request under contention — Dukascopy's per-IP rate
cap, not our parallelism), and was **stopped by the owner mid-run** in favor of databento (paid
credit available, no per-request throttle, and — decisively — Dukascopy's curated CFD feed simply
does not carry ~50 of these names at all: no leveraged/inverse ETFs, no 2023+ IPOs, no KR/pre-IPO).
The partial Dukascopy data was discarded; **databento is the sole source for all 102 underlyings**.

**Routing**: each ticker's PRIMARY listing venue (XNAS.ITCH / XNYS.PILLAR / ARCX.PILLAR — the only
venues with 2018-05+ history; the consolidated DBEQ/EQUS feeds only go back to 2023) was
determined by **trailing 5-day traded VOLUME**, not a naive default — an early attempt that
compared cost/data-availability instead of volume mis-routed the majority of tickers to the wrong
venue. Verified against known listings (COIN/MSTR/KLAC→Nasdaq, BX→NYSE, SPY/SOXL→Arca) before the
real pull. One ticker-alias recovered: Binance `PAYPUSDT`'s real security is PayPal = **PYPL**,
not "PAYP". One historical-rename merge: **META** pulls both `META` and its pre-2022 ticker `FB`
and stitches them into one continuous series (confirmed: 860,134 rows spanning the full
2018-2026 window, not truncated at the 2022 rename).

**Fetch**: `scripts/dbn_fetch.py`, full 2018-05-01→now 1-min OHLCV via the REST `timeseries.
get_range` endpoint (CSV encoding, no client library), **RTH-filtered to 09:30-16:00 America/
New_York** so the daily "close" is the real 4pm auction print, not an extended-hours tick. Total
routed cost: **$50.72** for 102 names (well under the ~$121 credit; a ticker-format bug —
`BRKB`→must be `BRK B` with a literal space — was caught and fixed before it silently dropped
Berkshire Hathaway). 3 names have no real US listing on databento at all (BBX, BSP, QNTX — these
read as Binance-native synthetic "thematic" tokens, not real single-stock tickers).

Perp legs (all 117 TradFi perps) + funding backfilled from the existing `data.binance.vision`
dump pipeline (unchanged; young listings mean most months 404, handled). Stitch
(`stitch_tradfi_1m.py`): consolidate → split-detect/adjust equities → ratio-adjust the underlying
onto the perp level at the overlap seam → splice [adjusted underlying before the perp's first bar
| real perp after] → flag any >1.5-min gap (session/weekend/pre-listing). Perp-only names
(`dukascopy: null` in candidates.md — legacy field name, now holds the databento code) stitch
perp-only, no proxy history.

## Data-quality investigation (the bulk of this branch's engineering)

A first full stitch run surfaced a genuine bug, which triggered a systematic re-audit. This
section is the record of what was found, how each item was resolved, and how it was verified —
kept in full because any future consumer of this lake needs to trust the same evidence, not just
the conclusion.

### 1. METADBN feed corruption — CONFIRMED, EXCLUDED

The databento XNAS.ITCH feed for META contains sparse but *dominant* erroneous sub-$100 prints
across **2021-06-30 → 2022-02-02** (up to 96% of a day's 1-min bars on the worst days — this is
not occasional noise, the median close is corrupted on many days too, ruling out a simple
robust-statistic fix). It first surfaced as three spurious ~20:1 "splits" in the auto-detector
(META has never split its stock). Verified via direct tick inspection: the erroneous prints
**revert day-to-day** (e.g. real $340 → garbage $17 → real $345 the next day), which a genuine
corporate action never does. **Fix**: `KNOWN_BAD_WINDOWS["METADBN"]` masks this exact window out
of the series entirely before any processing (no attempt to salvage individual ticks — the
contamination is too dominant to trust a partial repair). Post-mask: 860,134 rows (was 919,103),
zero splits detected (was 3 spurious), difference matches the masked window exactly (58,969 rows).

### 2. Split detector was structurally blind to reverse splits — FIXED

The original detector only checked `prev/close ≈ factor` (price *drops*), so it could never catch
a *reverse* split (price *jumps* — routine for leveraged/inverse ETFs fighting volatility decay:
SQQQ, UVXY, SOXL, TQQQ, and the newly-added KORU). This left several names with 300–3700%
unadjusted single-day "moves" sitting in the raw series (KORU 942%, SQQQ 395%, STRC 3721%, UVXY
933%). **Fix**: symmetric detection — both `prev/close` and `close/prev` are checked against the
factor grid; a reverse split is encoded as a negative factor and `_split_adjust` applies the
inverse (multiplicative, not divisive) adjustment to pre-event bars. Verified with unit tests
(synthetic real-reverse-split acceptance, synthetic bad-tick-shaped-as-reverse rejection, exact
price/volume-scaling math) and confirmed on real data: SQQQ, UVXY, STRC, KORU now all show their
real reverse splits, with max unadjusted single-day moves falling from 300–3700% into normal
single-digit-to-teens percent ranges.

### 3. Persistence check — added (rejects bad-tick artifacts, verified not to reject real splits)

Any single-day ratio match (forward or reverse) is only accepted if the *following* trading day
does not revert toward the pre-event level — this is what actually caught and rejected META's
three spurious ratios (each reverted within one day). Verified against all previously-confirmed
real splits (16 of them) with zero regressions, and against the one case with no following day to
check (CRWD's split lands on the very last day of the fetch window — accepted, independently
confirmed by whole-day price consistency: 391/391 minutes at the new ~$190-200 level).

### 4. Three real splits recovered that the ratio check itself missed — externally verified, added

Same-day price *drift* around a split's effective date can push the observed ratio a few percent
outside the ±3% match tolerance, causing the detector to miss a real, well-documented split
entirely. Three such cases were found and externally verified before being added as explicit
overrides (`KNOWN_SPLITS`):
- **TSLA, 2020-08-31, 5-for-1** (observed ratio 4.44 vs true 5.00) — Tesla's well-known first
  split; without this override, all TSLA history before Aug 2020 sits at 5× the wrong price level.
- **KLAC, 2026-06-12, 10-for-1** (observed ratio ~8.6, KLA's own trailing-day price movement
  pushed it well outside tolerance) — confirmed via [KLA's press release](https://www.prnewswire.com/news-releases/kla-corporation-announces-ten-to-one-stock-split-and-quarterly-cash-dividend-payment-302766137.html):
  approved May 7 2026, split-adjusted trading began exactly 2026-06-12.
- **KORU, 2025-02-10, 1-for-10 reverse** (observed ratio 10.54) — confirmed via
  [Direxion's press release](https://www.direxion.com/press-release/direxion-announces-reverse-split-of-koru):
  split-adjusted trading began exactly 2025-02-10.

### 5. GME false positive — confirmed real market data, explicitly excluded

The symmetric-detection fix (item 2) introduced one new false positive: a `[2021-02-24, -2]`
"reverse split" for GME. Direct verification shows GME genuinely moved $44.97 → $91.71 on that
date and **kept climbing** to $137.74 over the following week — real, sustained speculative
momentum during a second wave of the 2021 meme-stock squeeze, not a one-time mechanical split
(GameStop's only actual split was July 2022, in a calm period). A general statistical guard
(reject candidates when the stock's trailing volatility was already extreme) was calibrated and
**rejected**: GME's trailing 10-day volatility (4.94%) proved statistically indistinguishable
from KLAC's genuine real split (4.83%) — not a safe basis for a global rule given the available
calibration data (~18 cases, only one confirmed negative). Excluded explicitly instead
(`KNOWN_FALSE_SPLITS`), the same conservative pattern as the verified inclusions.

### 6. PAYP's basis — real market structure, not a bug

PAYPUSDT's underlying/perp ratio is 0.426 (every other name is 0.999–1.004). Verified this is not
a fetch error: both legs show internally consistent, real prices at their own timestamps
(underlying ~$40-46, matching PayPal's real recent price range; perp ~$14.6-14.7, consistently).
**Binance's tokenized PAYPUSDT perp genuinely trades at roughly 1/3 of PayPal's real share price**
— a structural basis gap in this specific synthetic product, not a data-pipeline defect. Flagged
here as a caveat: any basis/carry study should treat PAYP as an outlier requiring separate
handling, not average it in with the ~1.0-ratio majority.

### 7. Validation tooling fix: gap-blind "max move" statistic

Not a pipeline bug, but a bug in the QC scripts used to *find* the pipeline bugs above: the
diagnostic's max-single-minute-move statistic computed `diff(close)/close` across every
consecutive row without checking `gap_before` — so a real multi-month gap (e.g. SPCX's sparse
pre-IPO gray-market prints jumping to its real post-IPO perp price weeks later) produced a
spurious "827% move" alarm with nothing wrong in the actual stitched data (`gap_before` was
already correctly `True` at that exact row). Fixed to exclude gap-flagged boundaries from the
statistic (and fixed a resulting index-misalignment bug from filtering before `argmax` — now uses
`nanargmax` to preserve positional correspondence).

### Final validation state

Flag count dropped from 62 (first stitch pass) to **22**, and every remaining flag is one of two
fully expected, non-bug categories: short-history new listings (correctly reflecting real recent
IPO/listing dates: ALAB, ARM, BMNR, CRCL, CRWV, DRAM, MVLL, NBIS, SNDK, STXX, USAR, CBRS) and
thin/zero-overlap seam warnings for names with no meaningful underlying/perp overlap window (KR
equities, pre-IPO, the 3 synthetic thematics, KSTR's 260-minute overlap). CRWD's real,
independently-verified 2026-07-02 split shows a 75% residual single-day move because it lands on
the literal last day of the fetch window (no following day exists to smooth against) — expected
boundary noise on a genuine, correctly-adjusted event, not a defect.

## Caveats (must carry into any equity study)

1. **SURVIVORSHIP.** The venue roster is *today's winners* — Binance listed these perps in 2026
   because they are the names people want to trade now (mega-cap tech, AI, meme/retail
   favorites). A cross-section built from this roster is selection-biased: it excludes names
   prominent in 2018-2022 that have since faded or delisted. This flatters any backtest and is
   fatal for long-horizon momentum; it is *less* fatal (but not nil) for short-horizon
   cross-sectional reversal, which is far less sensitive to the survivorship of the roster than to
   the within-cross-section ranking. A truly clean test needs a point-in-time index-constituent
   universe, which this lake does NOT provide. State this in every equity report.
2. **Underlying ≠ perp; basis varies by name.** The databento proxy is the US-session (RTH)
   primary-listing price; the perp trades ~24/7 and can carry a material structural basis to the
   underlying (PAYP ~0.43x — see item 6 above). The stitched series is RTH-gapped before the perp
   seam and 24/7 after. Indicator-grade for backtest, not an exact replica of the tradeable perp.
3. **Perp history is weeks-to-months.** The tradeable instrument (the perp) is young; only the
   underlying gives multi-year history. Perp-native studies (funding carry, basis) have only the
   short perp window; underlying-history studies (reversal, announcement) use the long proxy.
4. **Demo-venue breadth.** Only **19 of the 110** equity perps are live on the Binance DEMO today
   (paper-trading venue). Research can use all 110/102; a future equity paper book starts at ~19
   names — the venue-availability gate (PAPER.md §11.3 pattern) already handles this class of
   problem for the commodity book and would need the same treatment here. Demo names: AMZN, COIN,
   CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR, MVLL, NVDA, OPENAI, PAYP, PLTR, QNTX, QQQ, SPCX, SPY,
   TSLA.
5. **Split-adjustment residual noise at boundaries.** Even correctly-detected splits show a small
   (single-digit to ~15%) residual move right at the effective date, from real trading on the
   split day itself compounding with the mechanical adjustment. This is expected and should not
   be treated as a data-quality signal in downstream studies (e.g. don't flag every split date as
   an outlier week — it will already show up, correctly, as a real (if boundary-noisy) return).

## Manifest — 102 names with real underlying history (8.18yr max, from the 2018-05-01 databento
floor; shorter spans are genuine recent listings)

| Symbol | First | Last | Years | Rows | Venue | Splits |
|---|---|---|---|---|---|---|
| AAOIUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 576,615 | XNAS | — |
| AAPLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 906,102 | XNAS | — |
| ADBEUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 826,641 | XNAS | — |
| AMATUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 829,997 | XNAS | — |
| AMDUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 870,154 | XNAS | — |
| AMZNUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 971,552 | XNAS | 2022-06-06 20:1 |
| ASMLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 769,968 | XNAS | — |
| AVGOUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 886,033 | XNAS | 2024-07-15 10:1 |
| AXTIUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 385,821 | XNAS | — |
| BABAUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 884,792 | XNYS | — |
| BRKBUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 847,322 | XNYS | — |
| BXUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 795,315 | XNYS | — |
| CATUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 789,093 | XNYS | — |
| CIENUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 735,863 | XNYS | — |
| COSTUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 825,945 | XNAS | — |
| CRMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 834,229 | XNYS | — |
| CSCOUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 857,772 | XNAS | — |
| DISUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 856,442 | XNYS | — |
| EBAYUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 821,191 | XNAS | — |
| EWJUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 843,540 | ARCX | — |
| EWTUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 723,415 | ARCX | — |
| EWYUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 875,185 | ARCX | — |
| EWZUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 807,840 | ARCX | — |
| FLEXUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 716,491 | XNAS | — |
| GLWUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 767,257 | XNYS | — |
| GMEUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 752,925 | XNYS | — |
| GOOGLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 917,532 | XNAS | 2022-07-18 20:1 |
| HDUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 851,464 | XNYS | — |
| HPEUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 746,796 | XNYS | — |
| IBMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 831,651 | XNYS | — |
| INTCUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 978,342 | XNAS | — |
| IWMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 832,650 | ARCX | — |
| JPMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 855,351 | XNYS | — |
| KLACUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 790,797 | XNAS | 2026-06-12 10:1 |
| KORUUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 263,778 | ARCX | 2025-02-10 -10x rev |
| LITEUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 794,372 | XNAS | — |
| LLYUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 820,158 | XNYS | — |
| LRCXUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 806,997 | XNAS | 2024-10-03 10:1 |
| METAUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 860,134 | XNAS | — |
| MRVLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 849,782 | XNAS | — |
| MSFTUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 889,296 | XNAS | — |
| MSTRUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 798,727 | XNAS | — |
| MUUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 904,116 | XNAS | — |
| NFLXUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 830,364 | XNAS | 2025-11-17 10:1 |
| NOKUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 595,977 | XNYS | — |
| NOWUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 779,566 | XNYS | 2025-12-18 5:1 |
| NVDAUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 919,621 | XNAS | 2021-07-20 4:1; 2024-06-10 10:1 |
| NVOUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 749,268 | XNYS | 2023-09-20 2:1 |
| ORCLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 852,780 | XNYS | — |
| PAYPUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 921,420 | XNAS | — |
| QCOMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 869,491 | XNAS | — |
| QQQUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 906,039 | XNAS | — |
| SMCIUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 533,375 | XNAS | 2024-10-01 10:1 |
| SOXLUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 813,270 | ARCX | — |
| SPYUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 907,071 | ARCX | — |
| SQQQUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 747,613 | ARCX | 2019-05-24 -4x rev |
| TERUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 787,599 | XNAS | — |
| TQQQUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 807,117 | XNAS | 2018-05-24 3:1; 2021-01-21 2:1 |
| TSLAUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 986,417 | XNAS | 2020-08-31 5:1; 2022-08-25 3:1 |
| TSMUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 894,714 | XNYS | — |
| TTWOUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 778,368 | XNAS | — |
| TXNUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 802,787 | XNAS | — |
| UVXYUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 793,541 | ARCX | 2018-09-18 -5x rev; 2023-06-23 -10x rev |
| VUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 854,130 | XNYS | — |
| WDCUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 843,846 | XNAS | — |
| WMTUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 854,239 | XNAS | 2024-02-26 3:1 |
| XLEUSDT | 2018-05-01 | 2026-07-04 | 8.18 | 828,509 | ARCX | 2025-12-05 2:1 |
| BEUSDT | 2018-07-25 | 2026-07-04 | 7.94 | 736,187 | XNYS | — |
| COHRUSDT | 2018-09-24 | 2026-07-04 | 7.78 | 415,062 | XNYS | — |
| DELLUSDT | 2018-12-28 | 2026-07-04 | 7.52 | 742,778 | XNYS | 2021-11-02 2:1 |
| ZMUSDT | 2019-04-18 | 2026-07-04 | 7.21 | 724,900 | XNAS | — |
| UBERUSDT | 2019-05-10 | 2026-07-04 | 7.15 | 757,037 | XNYS | — |
| CRWDUSDT | 2019-06-12 | 2026-07-04 | 7.06 | 715,763 | XNAS | 2026-07-02 4:1 |
| URNMUSDT | 2019-12-04 | 2026-07-04 | 6.58 | 291,567 | ARCX | 2022-12-21 2:1 |
| DKNGUSDT | 2020-04-24 | 2026-07-04 | 6.19 | 633,924 | XNAS | — |
| PLTRUSDT | 2020-09-30 | 2026-07-04 | 5.76 | 729,236 | XNAS | — |
| ONDSUSDT | 2020-12-04 | 2026-07-04 | 5.58 | 297,368 | XNAS | — |
| SPCXUSDT | 2020-12-16 | 2026-07-04 | 5.55 | 83,714 | XNAS | — |
| HIMSUSDT | 2021-01-21 | 2026-07-04 | 5.45 | 474,566 | XNYS | — |
| KSTRUSDT | 2021-01-27 | 2026-07-04 | 5.43 | 48,364 | ARCX | — |
| SONYUSDT | 2021-04-01 | 2026-07-04 | 5.26 | 423,165 | XNYS | 2024-10-09 5:1 |
| ASTSUSDT | 2021-04-07 | 2026-07-04 | 5.24 | 467,012 | XNAS | — |
| COINUSDT | 2021-04-14 | 2026-07-04 | 5.22 | 681,332 | XNAS | — |
| HOODUSDT | 2021-07-29 | 2026-07-04 | 4.93 | 653,671 | XNAS | — |
| RKLBUSDT | 2021-08-25 | 2026-07-04 | 4.86 | 490,150 | XNAS | — |
| STRCUSDT | 2021-09-27 | 2026-07-04 | 4.77 | 135,839 | XNAS | 2023-07-06 -5x rev |
| FLNCUSDT | 2021-10-28 | 2026-07-04 | 4.68 | 474,290 | XNAS | — |
| RIVNUSDT | 2021-11-10 | 2026-07-04 | 4.65 | 481,649 | XNAS | — |
| IRENUSDT | 2021-11-17 | 2026-07-04 | 4.63 | 374,801 | XNAS | — |
| CRDOUSDT | 2022-01-27 | 2026-07-04 | 4.43 | 420,016 | XNAS | — |
| ARMUSDT | 2023-09-14 | 2026-07-04 | 2.8 | 318,003 | XNAS | — |
| ALABUSDT | 2024-03-20 | 2026-07-04 | 2.29 | 231,216 | XNAS | — |
| NBISUSDT | 2024-10-21 | 2026-07-04 | 1.7 | 209,676 | XNAS | — |
| SNDKUSDT | 2025-02-24 | 2026-07-04 | 1.36 | 231,074 | XNAS | — |
| MVLLUSDT | 2025-03-07 | 2026-07-04 | 1.33 | 52,087 | XNAS | — |
| USARUSDT | 2025-03-14 | 2026-07-04 | 1.31 | 183,847 | XNAS | — |
| CRWVUSDT | 2025-03-28 | 2026-07-04 | 1.27 | 179,130 | XNAS | — |
| BMNRUSDT | 2025-06-05 | 2026-07-04 | 1.08 | 124,488 | XNYS | — |
| CRCLUSDT | 2025-06-05 | 2026-07-04 | 1.08 | 275,025 | XNYS | — |
| DRAMUSDT | 2026-04-02 | 2026-07-04 | 0.25 | 79,444 | ARCX | — |
| STXXUSDT | 2026-04-24 | 2026-07-04 | 0.19 | 35,978 | ARCX | — |
| CBRSUSDT | 2026-05-14 | 2026-07-04 | 0.14 | 68,070 | XNAS | — |

## Manifest — 8 perp-only names (no real underlying available)

| Symbol | Note |
|---|---|
| ANTHROPICUSDT | pre-IPO, no public listing |
| BBXUSDT | no real US-listed underlying found (synthetic/thematic) |
| BSPUSDT | no real US-listed underlying found (synthetic/thematic) |
| HYUNDAIUSDT | KR equity, no US listing |
| OPENAIUSDT | pre-IPO, no public listing |
| QNTXUSDT | no real US-listed underlying found (synthetic/thematic) |
| SAMSUNGUSDT | KR equity, no US listing |
| SKHYNIXUSDT | KR equity, no US listing |
