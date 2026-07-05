# SWING XSEC COMMODITY — RESULTS REPORT (2026-07-05)

**Verdict: NO deployable edge on our 7-perp universe.** Every pre-registered factor is net-negative;
the best is realized-skew at ~−0.2 net Sharpe; all three pre-named ensembles land −0.6 to −0.9 net
Sharpe with DSR ≈ 0.00 and CPCV frac-positive < 0.3. This is a clean, verified negative — not a bug
(see §Provenance). The owner's honest 0.85–1.15 target is **not reachable on this universe**; the
mechanism is universe composition + breadth, detailed in §Why.

## Panel (better than planned)
- **839 Friday weeks, 2010-06-11 → 2026-07-03, breadth 7 every week.** Price basis = databento
  front-month daily settles (rank 0), ROLL-AWARE returns via instrument_id change (roll-day return =
  c0_t / c1_{t-1}, never the raw front jump). This uniform 2010+ panel is longer and cleaner than the
  duka-stitch plan (which had per-name floors and gave XPT/XPD only from 2021). Curve slope from
  rank0/rank1; COT joined at usable_from ≤ Friday (report+3d release lag). Net of 12bp RT primary.

## Per-factor (net Sharpe annualized; rt6≈gross, rt12=primary)
| Factor | gross(rt6) | net(rt12) | EV bp/nm-wk | win% | DSR | CPCV frac+ |
|---|---|---|---|---|---|---|
| F1 carry | −0.43 | −0.58 | −19.2 | 43.8 | 0.00 | 0.07 |
| F2 basis-momentum | −0.70 | −0.90 | −29.4 | 44.4 | 0.00 | 0.00 |
| F3 reversal (COT-cond) | −0.12 | −0.45 | −11.5 | 46.7 | 0.00 | 0.18 |
| F3 reversal (uncond) | −0.04 | −0.31 | −10.0 | 46.2 | 0.00 | 0.25 |
| F4 momentum 6w | −0.43 | −0.57 | −16.9 | 48.0 | 0.00 | 0.00 |
| F4 momentum 8w | −0.42 | −0.55 | −16.4 | 46.2 | 0.00 | 0.00 |
| F4 momentum 10w | −0.25 | −0.37 | −11.0 | 46.8 | 0.00 | 0.14 |
| F6 mom×hedging-pressure | −0.53 | −0.67 | −18.7 | 45.6 | 0.00 | 0.00 |
| F5 skew 40d | −0.11 | −0.21 | −6.1 | 48.2 | 0.00 | 0.29 |
| F5 skew 60d | −0.17 | −0.26 | −7.6 | 48.0 | 0.00 | 0.11 |

Carry's *raw extreme spread* (most-backwardated minus most-contango, gross) is +1.68bp/wk, **Sharpe
+0.01** — i.e. carry has ~zero premium here, and any 2-a-side book plus cost pushes it net-negative.

## Ensembles (equal-risk mean of members, dollar-neutral, + 10% vol-target overlay)
| Ensemble | net Sharpe | EV bp | CAGR | maxDD | DSR | vol-tgt Sharpe |
|---|---|---|---|---|---|---|
| F3+F4+F5 (core) | −0.61 | −16.6 | −9.2% | 82% | 0.00 | −0.68 |
| +F6 | −0.69 | −18.6 | −10.1% | 85% | 0.00 | −0.73 |
| +F1+F2 (all, curves) | −0.93 | −24.2 | −12.7% | 90% | 0.00 | −0.90 |

Adding the curve factors (F1+F2) makes it WORSE — consistent with §Why (curve signal is degenerate on
a metal-heavy set). No leverage/Kelly table is meaningful: full-Kelly is negative (the edge is
negative), so optimal leverage is zero. Nothing is deployable; nothing is adopted.

## Why (mechanism, not bug)
1. **Universe is 4/7 precious metals** (gold, silver, platinum, palladium) — all near-infinite
   inventory ⇒ convenience-yield ≈ 0 ⇒ carry/basis-momentum are structurally degenerate on them
   (exactly the owner-supplied preamble's warning about gold). The curve factors effectively have
   only 3 real-carry names (crude, natgas, copper), far below the breadth they need.
2. **Breadth 7 is too small for a quantile book.** Fundamental law IR≈IC·√breadth; published 0.5–0.9
   are breadth-20–30. At breadth 7 a top/bottom-2 book is ~half the universe each side — almost no
   cross-sectional selectivity — so the already-small premium is swamped by idiosyncratic name
   variance and does not clear the ~12bp round-trip.
3. **Decay + turnover.** Short-lookback momentum decayed post-2010 (literature); the reversal factor
   is the turnover hog (~1.4 gross/wk) and dies at 24bp. Skew is the least-bad but still net-negative.

## Provenance (why this negative is trustworthy)
- A data-contamination bug WAS found and fixed first: WTI 2020-04-20 settled −$2.67, producing
  spurious −116%/−439% "returns" through the negative price. Fixed by invalidating returns through
  any non-positive price + winsorizing daily returns to ±40% (declared data hygiene).
- The factor engine was then VERIFIED consistent: pipeline k=1 equal-weight gross (+1.69bp) matches
  the independent decile diagnostic (+1.68bp) to the basis point — no sign/alignment/weighting bug.
- No-lookahead by construction (weight_t uses info ≤ t, scored on t→t+1; COT usable_from ≤ Friday);
  net-of-cost throughout; DSR at N_eff=13 (the full pre-registered family); CPCV(8,k2,embargo6).

## What would change the answer (not pursued without a decision)
- **More breadth**: the crypto book works because it has ~60 names. Commodities on Binance perps cap
  at 7 (Brent deduped). Real breadth needs the CME/LME agriculture+softs+industrials universe (20–30
  names) via dated futures — a different data + execution program, not these perps.
- **Not** a construction tweak: z-score vs quantile weighting, k=1 vs k=3, were bounded by the ~0
  gross carry spread — no weighting recovers a premium that isn't there at this breadth.

Bottom line: the swing xsec commodity thesis is sound in the literature but **does not survive our
7-name, metal-heavy perp universe**; the incumbent recommendation is to NOT deploy it here.
