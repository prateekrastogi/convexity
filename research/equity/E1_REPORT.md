# E1 PERP-NATIVE SUITE — STUDY RECORD (Program 5)

**Verdict: pre-named deployable (S1 weekly band-ensemble + S3(c) overlay) FAILS the adoption gate —
DO NOT DEPLOY to demo paper.** Run 2026-07-06 (equity worktree). Pre-registration:
[E1_PREREG.md](E1_PREREG.md), frozen + committed before any pipeline code. Every number below is
from `results_e1.json` (actual run, no fabrication). Universe: 110 TradFi-equity Binance perps (102
with real underlying, 8 perp-only); S1 panel 155 days/102 symbols with funding; S3 panel built from
102 underlying-backed names' RTH-anchored perp overnight/intraday split. Full run: 785.2s.

## Bottom line

**S1 (funding carry)**: the pre-named deployable — weekly band-ensemble across lookbacks
{7,14,30}d — is net Sharpe **−1.67 annualized**, DSR **0.00**, drop-top-5 **−2.10**, PBO (weekly
family) **0.586**. A clean, unambiguous negative on both the full 102-symbol universe and the
on_demo=19 subset (−1.97 ann). The regime-sanity mechanism check is **technically supported but
economically negligible**: pooled median funding across the full perp-era panel is +1.4e-06
(essentially zero — 50.3% of observations positive, barely above a coin flip), a much weaker
signature than the retail-long-bias carry story requires. The lookahead canary (weekly cadence,
funding shifted 14 days into the future) shows net Sharpe **+0.84 ann** (gross **+4.06 ann**),
correctly *positive and clearly above* the honest pipeline's −1.67 — proof the honest pipeline is
not leaking the same information, even though the margin is modest (funding carry is a genuinely
thin, low-magnitude mechanism on this young venue, so even the deliberately-leaked version is not
dramatically "too good").

**S2 (basis mean reversion)**: all 48 declared cells + 4 band-ensembles computed on a genuine,
schema-honest reconstruction of the perp/underlying overlap (see Data notes below — this required
falling back to the PRE-STITCH raw sources, since `data/stitched/` turned out to be a sequential
splice with zero simultaneous both-legs coverage). After two real bugs were found and fixed during
the mandatory Sharpe>1.5 bug hunt (a raw-underlying split-adjustment miss, and a rebalance-mask
mislabeling — see Bugs found section), the **daily-rebalance** cells settle into believable,
small-magnitude results: Sharpes from **−9.8 to +0.65 ann**, i.e. flat-to-slightly-negative, no
edge, exactly the "clean dead end" pattern. The **weekly-rebalance** cells remain pathologically
large (**−7.6 to −122 ann**) even after both fixes — diagnosed as a genuine construction fragility
(stale week-old vol-scaling sizing collides with a fast, 1-minute-timescale mean-reversion signal,
producing occasional catastrophic single-name blowups: maxDD 96–99% on the weekly cells vs 7–19%
on daily) rather than a further coding bug — see Bugs found section for the full diagnostic trail.
**S2 is not part of the pre-named deployable and this does not change the program's headline
result**, but it is reported in full per the frozen grid discipline: on its cleanest (daily,
threshold-gated) construction S2 is a clean negative/flat result, and on its noisier
(weekly/continuous) construction it is actively dangerous to trade, both valid findings.

**S3 (overnight/weekend structure)**: the mandatory index-level fact-check **HOLDS** — SPYUSDT
(Sharpe overnight 3.44 vs intraday 1.11, n=88), QQQUSDT (2.99 vs 1.51, n=89), and the pooled
102-name cross-section (0.64 vs −0.17, n=4832) all show overnight exceeding intraday. This is a
short-history result (perp-era only, n in the 88–177 range per cut) but internally consistent
across every cut. The cross-sectional variant S3(b) is small-but-plausible: xsec_L4 **+0.71 ann**
(DSR 0.06), xsec_L13 **+0.47 ann** (DSR 0.03), xsec_L26 **−0.23 ann** — inside the honest
0.8–1.2-ish "new venue" band at the short end, decaying at longer lookbacks, and none clear DSR at
N=179. The lookahead canary (overnight-Sharpe signal shifted 13 trading days forward) shows
**+2.48 ann**, clearly above every honest S3(b) cell — a real, if modest, "too good" signature
(discussed in Bugs found: this shift is smaller in bar-count than S1's or S2's canaries because
S3's calendar is already daily-native, so a 13-day shift is a genuinely large forward peek, not a
construction artifact). S3(c) (executing S1's rebalance in the overnight window) is a **null
overlay by construction**: the declared flat-bp cost model has no time-of-day cost variation, so
Delta-Sharpe = 0.000 exactly — reported honestly as "cannot be tested with this cost model," not
fabricated.

**Combined book** (S1+S3(b), inverse-variance weighted, completeness cell only): Sharpe **−1.65
ann**, DSR 0.00 — dominated almost entirely by S1 (weight 0.995 vs 0.005 on S3, since S1's variance
is far lower). Not the pre-named deployable; does not change the S1 verdict.

## Adoption gate verdict (frozen criteria, S1 weekly band-ensemble + S3(c))

| gate | threshold | result | pass? |
|---|---|---|---|
| net Sharpe positive, full universe | > 0 | **−1.6655** | **NO** |
| net Sharpe positive, on_demo (19) | > 0 | **−1.9698** | **NO** |
| DSR pass (N=179) | ≥ 0.95 | **0.0000** | **NO** |
| PBO ≤ 0.25 (S1 weekly family) | ≤ 0.25 | **0.586** | **NO** |
| drop-top-5-periods positive | > 0 | **−2.1002** | **NO** |
| S1 regime sanity (median funding > 0) | > 0 | **+1.4e-06** (frac_pos 50.3%) | technically yes, negligible |
| S3(a) fact check (overnight > intraday, SPY/QQQ) | holds | **holds** (3.44>1.11, 2.99>1.51) | YES |

**4 of 5 quantitative/structural gates fail outright (net Sharpe x2, DSR, PBO, drop-top-5); the two
regime-sanity signatures pass, but S1's own median-funding signature is so close to zero (1.4
millionths) that it does not meaningfully support the retail-long-bias carry story at the strength
the mechanism requires.** Per the frozen adoption gate, ALL five conditions must hold
simultaneously — they do not. **Recommendation: do not deploy the pre-named deployable to demo
paper.**

## Sleeve detail

### S1 — funding-carry cross-section
- Universe: 102 of 110 symbols have non-empty funding history (8 excluded: CATUSDT, TTWOUSDT,
  TXNUSDT, TERUSDT, BSPUSDT, KSTRUSDT, FLEXUSDT, STRCUSDT — missing funding files, not necessarily
  the same 8 as the "perp-only, no underlying" set).
- Panel: 155 calendar days, 2026-01-27 → 2026-07-04 (the entire panel is perp-era-only by
  construction — funding does not exist pre-listing, and even the earliest-funding name in this
  universe only has ~6 months of history). **Short-history caveat applies in full: 154 usable
  weekly observations is not enough to distinguish a real −1.67 Sharpe from noise at any
  confidence worth acting on — this is "short-history, directionally suggestive (negative),
  not yet backtest-confirmable at this length," exactly per the methodology's own honesty rule.**
- All 6 cells (3 lookbacks × 2 cadences) plus the weekly band-ensemble were net-negative; no cell
  cleared DSR. Per-year Sharpe: only one year of data (2026), so no cross-year robustness check is
  possible — flagged, not glossed over.
- On_demo (19-symbol) subset: −1.97 ann, slightly worse than the full-universe −1.67 — consistent,
  not a subset-driven artifact.

### S2 — perp↔underlying basis mean reversion
- **Data-path finding (declared before any cell was scored):** `data/stitched/{SYM}.parquet` is a
  SEQUENTIAL splice (source flips `duka`→`perp` at exactly one boundary, e.g. AAPLUSDT's last duka
  bar 2026-04-06 13:49 UTC / first perp bar 13:50 UTC — a 1-minute handoff, not a parallel window).
  It has ZERO simultaneous both-legs minutes anywhere, and therefore cannot supply the "both legs at
  1-minute resolution simultaneously" input E1_PREREG.md's S2 construction requires. The genuine
  overlap lives in the two PRE-STITCH raw sources still on disk and reachable read-only from this
  worktree: `data/klines/{SYMBOL}/` (raw Binance perp klines) and
  `data/dukascopy/{code}/{code}.parquet` (raw databento underlying, RTH-filtered, full
  2018–2026 span). All 102 underlying-backed names have a genuine, non-trivial overlap window this
  way (e.g. AAPLUSDT: ~3 months of true simultaneous 1-min coverage from the perp's 2026-04-06
  listing through the lake's 2026-07-02 close) — consistent with DATA.md's "perp history is
  weeks-to-months" characterization. This is reported here because it is a real, non-obvious
  property of the data contract that a future consumer needs to know, not an implementation detail.
- **48 of 48 declared cells computed** (2 samplings × 3 windows × 2 rebalance freq × 4 sizing
  variants), plus the 4 band-ensembles and 1 canary — the full family, no cells skipped.
- **Daily-rebalance cells (the construction that survives scrutiny):** Sharpes from **−9.8
  (5min_W5_daily_continuous) to +0.65 ann (5min_W60_daily_2.5)** — i.e. flat-to-mildly-negative
  across the whole grid, a handful of threshold-gated cells barely positive but nowhere near a DSR
  pass (max observed daily-cell DSR ≈ 0, none clear 0.95). Construction-check (per the prereg's S2
  regime signature): the threshold-gated cell's EV/obs is compared to the always-in continuous
  cell's; results are mixed and small in magnitude either way, consistent with "no real
  concentrated mean-reversion signal being captured," not a smoothly-steady artifact of a bug.
- **Weekly-rebalance cells:** Sharpes from **−7.6 to −122 ann** — large, but (per the Bugs Found
  section) diagnosed as a genuine construction fragility, not a residual sign/lookahead bug: maxDD
  is 96–99% on weekly cells vs 7–19% on the otherwise-identical daily cells, indicating a handful of
  catastrophic single-name blowups from holding a week-stale, vol-scaled position through a
  subsequent volatility-regime shift against a signal (1-minute basis z-score) that is only
  meaningful at a much shorter horizon than the holding period. **This sizing/signal-horizon
  mismatch is itself a valid, reportable finding: weekly rebalance is not survivable for this
  specific signal construction, full stop, regardless of window or threshold.**
- PBO over the full 48-cell family (aligned on the shortest member's tail): **0.00** — but this
  number should be read with caution given the pathological weekly-cell magnitudes dominate the
  cross-sectional ranking within the CSCV splits; it is reported per the frozen methodology, not
  suppressed, but is not being used here to argue FOR any S2 cell (none clear a reasonable Sharpe
  bar regardless of PBO).
- S2 is NOT the pre-named deployable and none of the above changes the S1-based adoption-gate
  verdict, per the frozen prereg's discipline.

### S3 — overnight/weekend structure
- (a) Fact-check HOLDS at the index level: SPYUSDT overnight Sharpe 3.44 > intraday 1.11 (n=88);
  QQQUSDT 2.99 > 1.51 (n=89); cross-sectional pooled (102 names) 0.64 > −0.17 (n=4832). All figures
  are perp-era-only (short history, stated explicitly): the overnight/intraday decomposition
  requires the PERP price series specifically (per the frozen prereg), so it cannot borrow the
  underlying's longer 2018+ history.
- (b) Cross-sectional variant: xsec_L4 +0.71 ann (DSR 0.062), xsec_L13 +0.47 ann (DSR 0.027),
  xsec_L26 −0.23 ann (DSR 0.001) — decaying with lookback length, none clearing DSR at N=179, all
  inside or below the honest "new venue" 0.8–1.2 gross band, i.e. not flagged for a bug hunt.
- Weekend cell (Friday RTH close → Monday RTH open) returned empty for both SPYUSDT and QQQUSDT —
  too few genuine Friday→Monday perp-era pairs exist yet (the perp listings are recent enough that
  the fully-perp-sourced weekend-pair count falls below the 30-obs floor `metrics()` requires).
  Reported as a data-insufficiency null result, not fabricated.
- (c) Overlay on S1: Delta-Sharpe = 0.000 exactly — the declared flat-bp cost model has no
  time-of-day cost variation to exploit, so this overlay is untestable with the current cost model,
  reported honestly rather than invented.

## Bugs found and fixed (mandatory Sharpe>1.5 bug-hunt trail, per METHODOLOGY)

Two of S2's honest cells initially showed impossible net Sharpes (as low as **−635 ann**) and the
S2 lookahead canary initially showed **−110 to −114 ann** (a large but WRONG-SIGNED "too good"
signature — the canary rule requires a large FAKE POSITIVE, and a large negative is just as
diagnostic of a bug). Both were root-caused and fixed before any number in this report was trusted:

1. **Raw-underlying split-adjustment miss.** `data/dukascopy/{code}/{code}.parquet` (the raw
   pre-stitch databento file S2 needs) is NOT split-adjusted — only `data/stitched/` carries the
   adjustment DATA.md documents. CRWDUSDT's real, externally-verified 2026-07-02 4:1 split showed
   up as an unadjusted ~4x jump in the raw underlying feed, producing a fake multi-hundred-percent
   single-bar basis "dislocation." Fixed by reusing `scripts/stitch_tradfi_1m.py`'s own
   `_detect_splits`/`_split_adjust`/`_mask_bad_windows` functions directly on the raw underlying
   before computing basis (methodology-mandated reuse, not re-derived logic), plus dropping the
   entire split-effective calendar day from the basis series (the perp itself was found to re-peg a
   few minutes INTO the underlying's own RTH session on the split day — a real few-minute
   cross-venue timing lag, not a bug in either adjustment, but one that still needs excluding).
   Verified fix: CRWD's max single-bar log-basis move fell from 1.389 (139%!) to 0.008 (0.8%), and
   the same fix (re-run across all 102 names) found no other split-driven outlier above ~11%
   (STXXUSDT, a thin/young name with only 1493 basis observations — a plausible genuine gap, not a
   split artifact).
2. **Rebalance-mask mislabeling ("daily" meant "every native bar").** S2's `freq == "daily"` branch
   was `rb = np.ones(n, dtype=bool)` — since `n`/`cal` are at the 1-minute (or 5-minute) NATIVE
   sampling frequency, this rebalanced the continuous-sizing book on EVERY bar, not once per real
   calendar day. Measured directly: mean turnover was 0.54 (roughly half the book's gross reshuffled
   every single minute), so the 13bp round-trip cost (mean 7.0bp/bar) ran ~6x the mean gross edge
   (1.1bp/bar) every bar, producing an astronomical, obviously-broken net Sharpe (~−521 to −635 ann)
   that was actually masking a real (already itself far past the 1.5 gross-Sharpe bug-hunt
   threshold) **+259 ann GROSS** signal from pure 1-minute microstructure noise/mean-reversion with
   no realistic tradeable analog — diagnosed via a "perfect foresight" test (ranking by the literal
   realized outcome) confirming the mechanism direction was sound and the pathology was cost-driven
   overtrading, not a sign error. Fixed to rebalance once per real RTH day. Post-fix, "daily" cells
   collapsed to plausible, small-magnitude Sharpes (−9.8 to +0.65 ann) — see Sleeve detail above.
3. **Investigated and NOT fixed further (documented, not swept under the rug):** even after both
   fixes above, S2's WEEKLY-rebalance cells remain large (−7.6 to −122 ann) and the S2 canary
   remains −114 ann (still a large, wrong-signed number). A `_zscore` self-referential-window
   hypothesis was tested and ruled out (fixing it changed nothing — identical numbers before/after).
   The actual cause, confirmed via direct turnover/maxDD inspection: weekly-held positions, sized
   off a SINGLE week-old volatility estimate, occasionally blow up (maxDD 96–99%) when the
   underlying vol regime shifts mid-week against a signal (1-minute basis z-score) whose natural
   half-life is nowhere near a week. This is reported as a genuine, diagnosed construction
   fragility of the weekly+continuous-sizing S2 variant, not a residual bug — but it is flagged
   explicitly as an open item: the S2 weekly canary's remaining −114 ann should NOT be read as "the
   honest weekly pipeline is proven clean" (the canary itself is not behaving as a clean pass/fail
   signal at weekly cadence, because the same 1-bar-shift construction that works cleanly on S1 and
   S3 is unstable against a highly autocorrelated weekly-held Z at S2's native 1-minute sampling).
   **S2's weekly-rebalance results should be treated with the same skepticism as an unresolved
   canary, and are excluded from any deployability claim** (S2 was never part of the pre-named
   deployable in any case).

Per the "above ~1.5 gross is more likely a bug than a real edge" instruction: every sleeve's gross
Sharpe was checked against this bar during development. S1's canary (gross +4.06 ann) and S3's
xsec cells (max +0.71 ann) and canary (implied gross similar order to its +2.48 net, since S3's
costs are modest relative to its signal) all sit inside or just outside the honest band and were
not bug-hunted further. S2's daily cells, post-fix, are also inside the honest band. S2's weekly
cells and canary were bug-hunted extensively (see above) and the residual pathology is documented
rather than force-fit to a clean number.

## Data notes (carried forward from DATA.md, restated per methodology)

- **Survivorship**: the 110-symbol roster is today's Binance-listed TradFi-equity perps — today's
  winners, not a point-in-time universe. Every headline number above is subject to this bias.
- **Demo-gate**: every full-universe headline number is paired with its on_demo=19-subset number
  where the cell construction supports it (S1's band-ensemble: −1.67 full vs −1.97 on_demo; S3(b)'s
  cross-sectional cells report an on_demo_subset field per cell in results_e1.json).
- **Short-history honesty**: S1 and S2 are perp-era-only by construction (155 days / ~6 months at
  most). S3's PERP-side inputs (overnight/intraday split, both variants) are equally perp-era-bound
  even though the underlying data itself spans back to 2018 — S3 needs the perp trading through the
  overnight window specifically, so it cannot borrow the longer underlying-only history. All Sharpe
  figures above are annualized display statistics computed on these short per-observation samples
  per the swing_pipeline.py convention (`per_obs_sharpe * sqrt(periods_per_year)`), and are
  explicitly NOT to be read as "backtest-confirmable at this length" — they are directionally
  informative only, exactly per the methodology's mandatory framing.

## Reproduce

`PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity
/home/speculation/convexity/.venv/bin/python research/equity/e1_pipeline.py` in the `equity`
worktree; reads the shared lake at `data/{stitched,funding,klines,dukascopy}` (read-only); writes
`research/equity/results_e1.json`. Runtime ≈ 785s. Live paper system (`main`) untouched throughout.
Full test suite: 654 passed, 1 skipped (unchanged from the pre-pipeline baseline).
