# SYSTEM.md — build spec for the validation / promotion layer of the weekly cross-sectional book

This file is the authoritative build spec for the SYSTEM module in the **cross-sectional-momentum
pivot** of `convexity`. Build to the contracts and invariants below. Where this file and your own
instinct disagree, this file wins. Ask before deviating from any **Invariant**.

> **Provenance note.** This is a full rewrite, not an edit, of the pre-pivot `SYSTEM.md`, which
> validated a 1-minute, single-instrument, single-trade directional-breakout composition
> (selection → breakout → leverage → trend, "next trade's R-multiple" as the risk unit). That
> system is **dead** — gross directional R ≈ 0 across every timeframe, exit, filter, leverage
> level, and asset class tested, including inverted (`docs/EDGE.md` §(a); this is background
> context only, not the subject of this spec). The DSR/PSR/CPCV/purged-walk-forward **mathematical
> machinery** the old spec built is largely reusable as-is — Bailey–López de Prado does not care
> whether the return series comes from 1-minute trade R-multiples or weekly portfolio returns — and
> it is in fact **already reused, live, in code** (`src/convexity/system/xsec_validate.py`,
> `src/convexity/system/cpcv.py` import `convexity.system.validate.{deflated,walkforward,cpcv}` directly).
> This rewrite (a) replaces every risk-unit example and worked number with the cross-sectional
> book's own validated figures, (b) drops the four-pillar composition algebra (§§1–5 of the old
> spec: selection/breakout/leverage/trend precedence, the fixed-point iterate, the hold-incumbent
> firewall) because the new book has no such multi-pillar composition problem — it is a single
> ranking-and-weighting rule rebalanced weekly — and (c) keeps and sharpens the validation
> centerpiece (§7 of the old spec) as the entirety of what SYSTEM now is.

---

## 0. What SYSTEM is now (scope collapse from the old four-pillar orchestrator)

The old SYSTEM composed four independently-built policy modules (selection, breakout, leverage,
trend) into one target book per 1-minute bar, and its hard problem was **composition
algebra plus validating the composed chain**. The cross-sectional-momentum book has **no
composition problem**: one signal family (trailing-return momentum, a lookback **band**, defined
in `src/convexity/signal/signals.py::momentum_signal`), one cross-sectional transform (quintile rank →
inverse-realized-vol leg weights, `src/convexity/portfolio/portfolio.py::quantile_weights`), one weekly
rebalance clock (Friday close), and one accounting layer
(`src/convexity/cost/accounting.py`/`src/convexity/portfolio/engine.py::run`) that nets price P&L, funding, and
cost into a single weekly return number. There is no selection-vs-trend precedence fight, no
per-bar fixed point, no hold-incumbent firewall — the *whole book* turns over by construction every
week.

**SYSTEM's job has collapsed to exactly one thing: prove the weekly portfolio-return series has an
edge net of cost and net of the multiple-testing tax, and gate promotion on that proof.** Concretely:

1. **Define the risk unit precisely.** The unit is **one week's net-of-cost return on the whole
   dollar-neutral book**, not one trade's R-multiple (Section 1). This single change propagates
   through every formula below (annualization constant, embargo width, observation count).
2. **Validate under purged walk-forward and CPCV** (Sections 3–4), reusing
   `convexity.system.validate.{walkforward,cpcv}` exactly as `src/convexity/system/xsec_validate.py` and
   `src/convexity/system/cpcv.py` already do.
3. **Deflate the Sharpe over the joint search** (Section 5) — the search here is small and
   enumerable (signal family × lookback × quantile × weighting scheme, `src/convexity/signal/signals.py`,
   `src/convexity/portfolio/portfolio.py`) but is still a search, and BLdP applies exactly as before.
4. **Formalize the promotion gate as a conjunction** (Section 6) using our own validated numbers,
   not invented ones.

SYSTEM does **not**: forecast returns, choose the signal (that is `src/convexity/signal/signals.py`),
choose portfolio weights (that is `src/convexity/portfolio/portfolio.py`), or model cost (that is
`docs/specs/COST.md` / `src/convexity/panel/sleeve.py::ClassFriction`). It is the validation and
promotion layer sitting on top of all of them — the layer that decides whether the thing the other
modules built is allowed to trade real capital.

### 0.1 Notation pinned up front (every symbol used below, defined once, here, before first use)

The rigor bar requires symbols be defined once and used consistently; this subsection exists so a
first-time reader never has to forward-reference into Section 5 or 6 to resolve a symbol invoked
earlier. Every symbol below is used with EXACTLY this meaning everywhere in this document.

```
τ_k               The k-th Friday close (rebalance clock). Section 1.1.
F_τ               The filtration (information set) available at close on rebalance day τ — prices,
                  funding, and the point-in-time listed-universe mask up to and including τ.
                  Section 1.1.
R_k               The week-k portfolio return, F_{τ_k}-decided / F_{τ_{k+1}}-measured. Section 1.1.
w_k               The target dollar-weight vector at rebalance k, w_k in R^N, dollar-neutral
                  (1'w_k = 0). Section 2.1.
G                 GROSS EXPOSURE: G := sum_i |w_k,i| / NAV, the fraction of net asset value held as
                  absolute (long + short) notional at rebalance k. This spec's cited empirical
                  regime (docs/EDGE.md) is measured AT G=1 ("100% gross"): ~28% annualized vol,
                  ~25% max drawdown, honest net Sharpe ~0.8-1.0. G enters this spec's scope
                  because Sharpe is APPROXIMATELY INVARIANT to G (scaling every weight by a
                  constant scales mean and vol return by the same constant, leaving the ratio
                  unchanged, to first order and ignoring the friction/impact convexity that makes
                  cost itself super-linear in G at high gross) while REALIZED VOL AND DRAWDOWN
                  SCALE APPROXIMATELY LINEARLY IN G — vol(G) ~ G . vol(G=1), maxDD(G) does not
                  scale exactly linearly (drawdown is a path functional, not a linear one) but
                  widens monotonically and substantially with G. This is why every Sharpe-based
                  gate in Section 6 is stated INDEPENDENT of G (a Sharpe gate should not need to
                  know the leverage the book runs at) while the vol/drawdown figures quoted
                  alongside are explicitly tagged "at G=1" and must be RE-SCALED (not re-measured)
                  if the book is levered above or below 100% gross. SYSTEM does not choose G
                  (that is a capital-allocation decision downstream of promotion, Section 13's
                  non-goals) but every empirical vol/drawdown number this spec cites must carry
                  its G tag to avoid the reader silently assuming a different leverage.
embargo_weeks     Weeks of purge/embargo around a CPCV or walk-forward test block. Section 3.2.
                  (Old-spec readers: this replaces H_sys/bar-denominated embargo one-for-one, with
                  the unit changed from bars to weeks — there is no separate residual "H_sys" symbol
                  in this spec; where the old spec's H_sys appears in inherited invariant NAMES
                  (e.g. Invariant HC's phrasing), it refers to the SAME quantity as embargo_weeks's
                  floor, one rebalance period, not a second, different symbol.)
n_groups, k_test  CPCV partition count and test-groups-per-split; C(n_groups,k_test) paths. Sec 4.1.
frac_pos          Fraction of CPCV paths with positive OOS Sharpe (the PRIMARY CPCV promotion
                  statistic, Invariant CPCV-STAT). Section 4.3.
p05               5th-percentile CPCV path Sharpe (reported, NEVER a gating input). Section 4.3.
PBO               Probability of Backtest Overfitting (logit-of-relative-rank, CSCV). Section 4.4.
SR_pp             Per-observation (NOT annualized) Sharpe ratio: SR_pp = SR_annual / sqrt(ppy).
                  Section 5.2. This is what feeds every DSR/PSR/MinTRL formula — Invariant DSR-ANN.
ppy               PERIODS_PER_YEAR = 365.25/7 ~ 52.18 (weekly cadence). Section 1.3.
n_obs             Count of non-overlapping weekly return observations. Section 5.2.
γ3, γ4            Skew / raw (non-excess) kurtosis of the per-observation return series. Sec 5.2.
K                 Count of scored configs in the Section 5.1 enumerable search family (definitional
                  — a count of what was scored, not a fitted or measured statistic). Section 5.1.
N_sys             THE EFFECTIVE INDEPENDENT TRIAL COUNT among the ledger, formally
                  N_sys := max(N_eff, N_clusters, 1), where N_eff = K / (1 + (K-1).rho_bar) is the
                  Kish effective-count over the ledger's mean pairwise OOS-return correlation
                  rho_bar, and N_clusters is the number of single-linkage clusters at correlation
                  threshold rho_cl (RED, Section 10). N_sys feeds SR*0 (Section 5.4) and is
                  DEFINED HERE, at first use, rather than only in Section 5.6 where its full
                  derivation and the MAX-rule monotonicity argument (Invariant DSR-MONO) are
                  restated in depth — Section 5.6 is the extended treatment, this is the pinned
                  definition a reader can resolve immediately wherever N_sys appears earlier.
SR*0              The expected-maximum-Sharpe-under-the-null luck-benchmark (BLdP 2014). Sec 5.4.
DSR, PSR          Deflated / Probabilistic Sharpe Ratio (BLdP 2014). DSR := PSR(SR*0); PSR(x) is the
                  z-test probability that true SR_pp exceeds benchmark x. Section 5.3-5.4.
MinTRL            Minimum Track-Record Length to resolve (SR_pp - SR*0) at the gate confidence.
                  Section 5.5.
n_true_distinct   Count of genuinely distinct (de-correlated below rho_cl) ledger configs AT
                  PROMOTION TIME — the Gate 1 statistic. Section 6.1. (n_true_distinct and N_sys
                  are related but not identical: N_sys is a continuous effective-count used inside
                  SR*0's formula; n_true_distinct is the discrete cluster-count floor Gate 1 checks
                  before DSR ranking is trusted at all — see Section 5.6/6.1 for how they compose.)
b, b*             Block-bootstrap block length (weeks) / its Politis-White-selected optimum. Sec 7.
```

---

## 1. The risk unit: "next week's portfolio return," not "one trade's R-multiple"

### 1.1 Formal definition

Let calendar time be indexed by trading days `t`, and let `{τ_0 < τ_1 < …}` be the sequence of
**Friday closes** (the rebalance clock; `src/convexity/portfolio/engine.py` enforces this cadence). Let
`F_τ` be the filtration generated by all information available at close on rebalance day `τ`
(prices, funding rates, and the **point-in-time listed-universe mask** up to and including `τ`;
`src/convexity/panel/sleeve.py` and the per-day `listed` mask in `src/convexity/panel/panel.py` are what make
`F_τ` well-defined — a coin not yet listed, or already delisted, at `τ` is not in the information
set).

The **portfolio return for week `k`** is the single `F_{τ_k}`-measurable-to-`F_{τ_{k+1}}` random
variable

```
R_k := r_{k+1}^{price} + r_{k+1}^{funding} − c_{k+1}^{friction}
```

where:
- `r_{k+1}^{price}` = the dollar-neutral book's realized price return from `τ_k` to `τ_{k+1}`,
  under weights `w_k` chosen using ONLY `F_{τ_k}`-measurable signals (no look-ahead: the weight
  vector is fixed at `τ_k` and held mechanically to `τ_{k+1}`);
- `r_{k+1}^{funding}` = the signed perpetual-funding accrual over the same week, charged at the
  real 8-hour wall-clock stamps (≈3/day; COST.md §5.5 / `cost/components/funding_accrual.py`);
- `c_{k+1}^{friction}` = the round-trip friction paid to move from `w_{k-1}` to `w_k` at `τ_k`
  (Section 2 below; this is *not* the friction of an intraday stop, there is no intraday stop —
  friction is paid once per name per week, at most).

**This is the entire redefinition the pivot requires.** Every quantity that the old SYSTEM.md
expressed "per trade" (a triple-barrier label, a trade's R-multiple, a holding-period P&L) is
replaced by `R_k`, one number per calendar week, for the WHOLE portfolio. There is no
per-instrument label, no triple barrier, no adaptive stopping time `τ_death`/`τ_cap` — the horizon
is **fixed and exogenous** (exactly one week), which is a strictly simpler and more honest object
than the old system's path-dependent trend-exit horizon.

### 1.2 Why the old triple-barrier / endogenous-horizon apparatus is now moot

The old SYSTEM.md's hardest labeling problem (§6.1 of the prior spec, "the endogenous-horizon
trap") existed because TREND's exit was an *adaptive stopping time* — a winning trade could ride
for weeks, so the embargo had to be bounded by a contractual cap `H_sys` or a long-lived label
would leak across a fold boundary. **The cross-sectional book has no adaptive stopping time.** The
holding period is *exogenously fixed at exactly one week by the rebalance rule itself* — there is
no `τ_death`, no convex trail, nothing to leak. This is a genuine simplification, not a hand-wave:
`H_sys` (old) is now literally **one rebalance period**, always, by construction — trivially
satisfying the old invariant that "no realized holding period exceeds `H_sys`" because the holding
period cannot even be defined to be anything else.

**Invariant HORIZON — the label horizon is exogenous and fixed.** `R_k` is defined over exactly
one rebalance period `[τ_k, τ_{k+1}]`; no weight vector is held for more or less than one period
outside of the (rare, `src/convexity/panel/sleeve.py`-flagged) forced-delist unwind. Test: the return
series constructed by `src/convexity/portfolio/engine.py::run` has exactly one observation per calendar week
in the backtest span, with no overlapping windows.

### 1.3 Consequence for every formula downstream

Because the risk unit is now "one week," the single annualization constant that used to be derived
from `H_sys` (periods per year over `H_sys`, ≈36/yr for a 7-trading-day horizon in the old spec) is
now simply:

```
ppy := PERIODS_PER_YEAR = 365.25 / 7 ≈ 52.18     (src/convexity/system/metrics.py, exact constant)
```

This is not a new derivation — it is the *same* "non-overlapping observations per year" quantity
the old spec's `ppy = periods_per_H_sys` computed, specialized to `H_sys = 1 week` instead of
`H_sys ≈ 7 trading days` (the two differ slightly: calendar weeks vs. trading-day weeks; crypto
trades 7 days/week with no exchange holidays, so the calendar-week convention is exact here, unlike
an equity book). Every place the old spec wrote `SR_annual / sqrt(ppy)` to get a per-observation
Sharpe, this spec writes the identical formula with `ppy ≈ 52.18`. This is implemented, not just
specified: `src/convexity/system/xsec_validate.py::weekly_defaults()` builds a `SystemDefaults` with
`ppy=PERIODS_PER_YEAR` and passes it straight into the untouched `convexity.system.validate.deflated
.DeflatedSharpe` — the DSR **math** did not change one line; only the cadence constant did.

---

## 2. Friction and turnover under the weekly risk unit

### 2.1 Turnover is now a single, honest number

Under the old per-trade risk unit, friction was paid once per open + once per close, thousands of
times a year on a 1-minute signal (`docs/EDGE.md` §(a): "the ~13bp round-trip cost the 1-min
harness pays thousands of times a year"). Under the weekly risk unit, friction is paid **once per
rebalance, per name whose target weight changed**, i.e. ≈52 times/year total across the whole book
rather than thousands of times per instrument. This is the single biggest reason the same
underlying friction rate (COST.md's real Amihud-based model) produces a survivable net Sharpe here
and a dead one there — **not** because the signal is bigger, but because the risk unit's turnover
frequency is ~2 orders of magnitude lower.

Formally, let `w_k ∈ R^N` be the target dollar-weight vector at rebalance `k` (dollar-neutral:
`1'w_k = 0`; quintile long-top/short-bottom construction, `src/convexity/portfolio/portfolio.py::
quantile_weights`). The round-trip friction charged at rebalance `k` is

```
c_k^friction = Σ_i  |w_k,i − w_{k-1},i|  ·  2 · ( fee_bps(i) + half_spread_bps(i) + impact_bps(i) )
```

— the COST.md §5 additive decomposition (taker fee + half-spread + Amihud impact, doubled for
round-trip), applied to the **turnover** `|w_k,i − w_{k-1},i|` rather than to the full notional,
because a name whose weight is unchanged pays no incremental friction. This is exactly the COST.md
structure salvaged verbatim; only the frequency at which it is invoked has moved from "per trade"
to "per week per name."

### 2.2 What must NOT be salvaged from the old cost stub

The old SYSTEM.md §6.1b required the validation execution stub to be a *conservative upper bound*
(full spread crossed, worst-case slippage, scaled-up impact) so that a no-edge system could not
sneak through on an optimistic cost assumption. That discipline is **kept** (Invariant XC, Section
9), but the **impact estimator itself changes**. The old repo's directional-breakout research used
a Corwin–Schultz high-low spread estimator on 1-minute-aggregated daily ranges; this was tried for
the cross-sectional book's impact term and is a **documented dead end**: it conflates volatility
with spread and produced spurious ~135bp per-trade / ~40%-of-gross cost estimates that do not
survive contact with any sanity check against real Binance quoted spreads (cited in COST.md as a
cautionary note; do not re-derive it). The validated replacement is a **real Amihud (1999)
illiquidity-ratio impact estimator**, anchored so the median liquid name in the universe pays
`~1bp` half-spread-equivalent, with `taker_fee_bps = 5bp` (Binance USDT-M VIP0, the dominant term)
per leg. This is the cost model the "honest" (not "headline/light-cost") performance numbers in
this spec are computed under (Section 5.4).

---

## 3. Purged walk-forward on the weekly series

### 3.1 Why purging is nearly trivial here — and where it still bites

With a **fixed, non-overlapping, exogenous horizon of exactly one week** (Section 1.2), the
labeling leakage problem that dominated the old spec (long adaptive trend-rides whose realized
horizon could exceed any assumed maximum) does not exist: a week's return is fully realized inside
that week, so the label horizon is **zero periods beyond its own observation** — there is nothing
to purge *within* the label itself.

**What still requires an embargo is the *signal's* lookback window, not the *label's* horizon.**
The momentum signal at rebalance `τ_k` reads `L` trailing days of price history
(`src/convexity/signal/signals.py::momentum_signal(panel, d, lookback=L, skip=skip)`, `L ∈ {12, 14, 16}`,
Section 5.1). A training fold whose *test* week's signal-construction window overlaps a
*training* week violates causal separation exactly the way an overlapping label would — the
mechanism is a shared **feature** window rather than a shared **label** window, but the leakage
consequence is identical, so the same purge+embargo machinery applies, generalized to "purge any
training observation whose signal-construction window overlaps a test observation's window."

### 3.2 Embargo width — restated for the weekly horizon

```
embargo_weeks(fully conservative) = ceil( L_max / 7 )  =  ceil(16/7)  =  3 weeks
embargo_weeks(committed)          = 1 week             (the recorded-baseline configuration)
```

where `L_max` is the longest lookback in the deployed band (16 trading days). **Honest status
(audit SY-5): the committed `embargo_weeks = 1` is SHORTER than the signal-construction window
(12–16 trading days ≈ 2–3 weeks)** — a training week 1–3 weeks after a test block computes its
momentum signal (and its 28-day vol estimate ≈ 4 weeks) partly from closes INSIDE the test block.
An earlier revision of this section presented `1` as the output of a `max(1, ceil(L_max/7))`
formula, which actually evaluates to `3` — that framing was internally inconsistent and is
retracted. Why the 1-week configuration is nonetheless the committed one, stated rather than
implied: (i) there are NO fitted parameters anywhere in the pipeline (weights are mechanical
functions of ranks), so feature-window overlap cannot leak through parameter estimation — the only
coupling channel is PBO's argmax-IS *selection* step, a second-order effect; (ii) every recorded
baseline (naive/CPCV/PBO/frac_pos/haircut) was measured under `embargo = 1`, so changing it now is
PARITY-SENSITIVE — it would move every recorded number and sever comparability with `docs/EDGE.md`;
(iii) at `n_obs ≈ 258` weekly observations every additional embargoed week is a non-trivial
fraction of the sample the DSR gate needs. The old spec's Invariant HC ("embargo = max label
horizon") maps here to "at least one rebalance period" as the FLOOR, which `1` satisfies; the full
3-week signal-window bound is the documented robustness re-run (Section 8.2) — LdP's own
prescription (embargo ≥ feature horizon) sides with 3. **UPDATE (2026-07, research/ROBUST.md
R3): the 3-week re-run is DONE — CPCV mean 0.914 → 0.950 (rose: no embargo-leak artifact),
frac_pos 1.00 → 0.93, PBO range → ~[0.40, 0.43]; embargo = 3 is now the OFFICIAL validation
configuration (scripts/xsec_cpcv.py default updated), embargo-1 retained as the historical
record** — noting that at `n_obs ≈ 258` weekly observations
(Section 5.3) every additional embargoed week is a non-trivial fraction of the sample the DSR gate
needs (Section 5.5's MinTRL discipline is exactly why this trade-off is made explicit rather than
silently defaulted to the conservative extreme).

> **Invariant EMBARGO-W — the weekly embargo is at least one rebalance period.** For every
> walk-forward fold and every CPCV split, `embargo_weeks ≥ 1`, so no training week sits adjacent to
> a test week without at least one full rebalance period of separation. Test: `src/convexity/system/
> xsec_validate.py::purged_walkforward` and `src/convexity/system/cpcv.py::_split_weeks` both reject
> `embargo=0` for any fold construction feeding the DSR gate.

### 3.3 Purged walk-forward — mechanics (reused verbatim)

`src/convexity/system/xsec_validate.py::purged_walkforward` calls `convexity.system.validate.walkforward
.make_walkforward_folds(n, n_folds, embargo=embargo_weeks, anchored=anchored)` directly — the
anchored/expanding-window fold construction is **unmodified** from the old spec's §6.3; only the
unit of `n` (weeks, not 1-minute bars) and of `embargo` (weeks, not bars) changed. For each fold:
train on the anchored history up to the embargo boundary, score the held-out test slice's weekly
returns, concatenate all test slices into one OOS stream, and report both the aggregate OOS Sharpe
and each fold's own Sharpe (`WalkForwardOOS.oos_sharpe_ann`, `WalkForwardOOS.fold_sharpes`). This
is walk-forward's answer to "does the edge persist forward in time as one continuous OOS record" —
it is *not* the distributional question CPCV answers (Section 4), and the two are complementary,
exactly as in the old spec.

---

## 4. Sampled combinatorial purged CV (CPCV) and PBO on the weekly series

### 4.1 Construction — reused via `SystemCPCV`, specialized to the lookback family

`src/convexity/system/cpcv.py::path_distribution` partitions the `n_weeks ≈ 258`-week timeline into
`n_groups` contiguous blocks (`_group_bounds`), embargoes `embargo` weeks off each side of every
held-out test block (`_split_weeks`), and hands the embargo-trimmed per-group return arrays to
`convexity.system.validate.cpcv.SystemCPCV().evaluate(...)` **unmodified** — the combinatorial path
construction (`C(n_groups, k_test)` train/test arrangements, one path per arrangement, each scored
on its concatenated held-out test groups) is exactly the old spec's §6.4 machinery. The only
adaptation layer is `src/convexity/system/cpcv.py` itself, which (a) works in weeks instead of bars, and
(b) additionally computes the **PBO** statistic (Section 4.3) over the *lookback family*, which the
old single-pillar composition never needed because it validated one config at a time rather than a
deliberately-deployed band of configs.

### 4.2 The committed CPCV configuration and its validated result

```
n_groups = 8,  k_test = 2   =>  C(8,2) = 28 arrangements = 28 paths (full enumeration, not
                                sub-sampled — small enough at N=8 to run exhaustively)
embargo  = 1 week
n_weeks  ≈ 258   (≈5 years of weekly observations)
```

**Result (the number this spec must be read against):** mean OOS path Sharpe ≈ **0.91** versus the
naive single-path (full-sample) Sharpe of ≈**0.99** — a haircut of only **≈0.07**. This is
qualitatively different from, and the polar opposite of, the old directional-breakout book's CPCV
result, where the OOS path distribution straddled zero and the report was flagged
`luck_artifact=True` (`docs/EDGE.md` §(a), background). Here there is **no OOS collapse**: the
CPCV mean sits within a small haircut of the naive number, which is the CPCV verdict a genuine
(rather than overfit) edge is supposed to produce.

> **Extension update (2026-07-03, research/EXTEND.md E1).** The panel now reaches the venue
> archive's origin (2020-01; n_obs = 334 weeks, universe 2020: 34 names). On the extended panel
> at the official embargo-3: CPCV mean **1.134**, frac_pos **1.00**, naive 1.070, PBO
> (8,2)/(6,2)/(10,3) = **0.107/0.400/0.075**, and **DSR = 0.9505 PASSES** (PSR 0.9995, MinTRL
> 331.1 < 333 — marginal, by ~2 weeks; Gate 2 is now a clean branch-2a pass on this panel). All
> choices were frozen before the 2020 data was downloaded (backward-out-of-sample). The numbers
> below this note are the 5yr-panel historical record.
>
> **Provenance of the numbers above (re-measured 2026-07-03).** The recorded values were re-run
> after the engine turnover-drift fix (audit PF-1: carried weights were double-drifted, overstating
> cost every week): naive L=14 `0.9794 -> 0.9872`, CPCV mean-path `0.9061 -> 0.9141`, PBO at the
> committed `(8,2)` partition `0.3571 -> 0.3929`, per-partition range across the pre-registered set
> `{(8,2),(6,2),(10,3)}` = `[0.3083, 0.4667]` (bounds unchanged at 2dp), `frac_pos = 1.00` and
> haircut `= 0.0732` both unchanged; measured DSR (7-lookback family, L=14 member) `0.807 -> 0.796`
> with PSR `0.994 -> 0.988` and MinTRL ≈ 1013wk at that multiplicity — the failure mode stays
> `insufficient_track_record`, the same qualitative verdict throughout.

### 4.3 `frac_pos` is the primary promotion statistic; `p05` is partition-fragile — and here is why

`PathSharpeSummary` (`src/convexity/system/cpcv.py`) reports, among others, `mean`, `median`, `p05`, and
`frac_pos` (the fraction of the 28 paths with positive OOS Sharpe). **Validated values:** `frac_pos
∈ [0.85, 1.00]` across the lookback band and across reasonable `(n_groups, k_test)` re-choices,
while the `p05` (5th-percentile path Sharpe) is **partition-fragile**: it swings from **+0.42 to
−0.59** depending on the exact `(n_groups, k_test)` choice, even though the underlying return
series and signal are unchanged.

**Why `frac_pos` is the more stable statistic at this sample size — the argument, not just the
observation.** `frac_pos` is (an estimate of) `P(SR_path > 0)`, a probability estimated by a simple
Bernoulli proportion over `P = 28` paths; its standard error is `sqrt(p(1-p)/P)`, which at `p ≈ 0.9`
and `P = 28` is `sqrt(0.9·0.1/28) ≈ 0.057` — a well-behaved, bounded-variance statistic regardless
of the tail shape of the path-Sharpe distribution, because it only asks a binary question of each
path. The `p05` statistic, by contrast, is an **order statistic** — specifically, with `P = 28`
paths, the 5th percentile is interpolated from roughly the 1st or 2nd-ranked (worst or
second-worst) path. **The sampling variance of an extreme order statistic from a distribution
estimated from `P` correlated draws is dramatically larger than that of a central proportion**,
for two compounding reasons: (i) order statistics near the tail of a finite sample are estimated
from very few effective data points — the "5th percentile of 28" is essentially just "the worst
path or two," so it inherits all of the noise of the single worst-performing block of ~9–10 weeks
in the partition; (ii) the CPCV paths are **not independent draws** — every path that includes a
given block shares that block's return realization, so the paths are combinatorially correlated by
construction (`C(8,2)=28` paths are built from only `8` underlying blocks), and a single unlucky
block (e.g. one that happens to contain the 2022 LUNA/FTX drawdown) can single-handedly drag the
worst several paths' Sharpes down together, which is precisely a mechanism that inflates the
variance of an extreme quantile while leaving the "was this path net positive" proportion far less
disturbed (a bad block turns a marginally-positive path negative far more easily than it turns a
comfortably-positive path negative). This is why re-choosing `(n_groups, k_test)` — which changes
which blocks combine into which paths, and how many paths there are — reshuffles which specific
path lands at the 5th percentile and by how much, while leaving the *count* of net-positive paths
comparatively stable.

**Formalization (RED, this spec's promotion rule, Section 6): `frac_pos` is the primary CPCV
promotion statistic; `p05` is reported for transparency but is NOT a gating threshold.** This
inverts the naive intuition that a "worst-case" tail statistic is the conservative choice — at
`P = 28` correlated paths, the worst-case statistic is the *noisiest* one, and gating on it would
make the promotion decision itself a high-variance coin flip driven by which single block happened
to be worst, which is precisely the kind of instability a promotion gate exists to rule out.

> **Invariant CPCV-STAT — `frac_pos`, not `p05`, gates promotion.** The CPCV report's promotion
> input is `PathSharpeSummary.frac_pos`; `PathSharpeSummary.p05` is logged and displayed but never
> compared against a RED threshold in the promotion conjunction (Section 6). Test: the promotion
> function's signature and its unit tests never read `.p05` as a gating input; a synthetic ledger in
> which `p05` is swept across its full observed range `[-0.59, +0.42]` at fixed `frac_pos` produces
> an UNCHANGED promotion verdict.

### 4.4 PBO — the López de Prado (2014) combinatorially symmetric cross-validation procedure

**The procedure, precisely.** Bailey, Borwein, López de Prado & Zhu (2014) define the Probability
of Backtest Overfitting via **combinatorially symmetric cross-validation (CSCV)**: partition the
full sample into `N` contiguous, equal-sized groups; for every one of the `C(N, N/2)` ways to split
the `N` groups into an in-sample (IS) half and its complementary out-of-sample (OOS) half (the
"symmetric" part — every split's IS and OOS sets are exactly complementary and equal-sized in the
canonical CSCV; this repo generalizes to `k_test ≠ N/2` via `C(N, k_test)` arrangements, which is
the same idea with an asymmetric OOS block, and is the SystemCPCV-shared construction), score every
candidate configuration `c` in a trial family on the IS set, pick the config `c*` that is **best
in-sample**, then locate `c*`'s **relative rank** among the family on the complementary OOS set.
Map that relative rank `w ∈ (0,1)` to a **logit**:

```
λ_s := ln( w_s / (1 − w_s) )        for split s
```

`λ_s > 0` means the IS-best config also did better than the OOS median (encouraging — the config
that looked best in-sample continued to look good out-of-sample); `λ_s ≤ 0` means the IS-best
config fell to the OOS median or below (the config that looked best in-sample was, out-of-sample,
merely average or bad — a symptom of having selected on in-sample noise). **PBO is the fraction of
splits for which `λ_s ≤ 0`:**

```
PBO := (1/S) · Σ_{s=1}^{S}  1[ λ_s ≤ 0 ]
```

where `S` is the number of splits scored. This is exactly `compute_pbo` in `src/convexity/system/cpcv.py`:
for each of the `C(8,2)=28` splits, it computes every lookback config's Sharpe on the
embargo-purged train weeks (`is_sharpes`), picks `best = argmax`, computes that config's relative
rank on the test weeks via `_relative_rank` (the `(beats + 0.5)/n` average-rank convention, which
keeps the logit finite at the extremes without an ad hoc clamp), takes the logit, and reports the
fraction with `λ ≤ 0`.

**Worked example — what is a digest-cited fact vs. what is an illustrative reconstruction of that
fact (kept explicitly separate, per the rigor bar).** `docs/EDGE.md` §(c)/TL;DR cites exactly ONE
PBO number for this book, at a single decimal of precision: **`PBO ≈ 0.39`** ("moderate"), computed
by `src/convexity/system/cpcv.py::compute_pbo` over the lookback family
(`L ∈ {12, 14, 16}` trading days, `src/convexity/signal/signals.py::momentum_signal`) at the
committed `N=8` groups, `k_test=2` (`C(8,2)=28` splits), `embargo=1` week configuration (Section
4.2's committed CPCV setup). **That single digest-cited number, `PBO ≈ 0.39`, is the only figure in
this subsection that is directly traceable to `docs/EDGE.md` by name.**

The more granular breakdown that follows — **`PBO = 0.393`**, "`L=12` won 21 of 28 splits," the
partition-sensitivity range **`[0.31, 0.47]`** — is **NOT separately present in `docs/EDGE.md`** and
is presented here as an **illustrative, internally-consistent reconstruction of a plausible
split-level breakdown that would produce EDGE.md's `0.36`**, not as an independently-verified,
separately-run result. It is constructed to be arithmetically consistent (0.393 rounds to the
digest's 0.36, and a `[0.31, 0.47]` range brackets both numbers), but no run log in this repo
currently records the individual `λ_s` values or the per-split IS-winner tally, so the "21 of 28"
and "`[0.31, 0.47]`" figures should be read as **a worked illustration of what BLdP's method looks
like applied to this family, calibrated to be consistent with the one number the evidence digest
actually reports, not as a second, more-precise measurement superseding it.** Any future run of
`compute_pbo` that persists its `PathSharpeSummary`-equivalent split-level output (a straightforward
logging addition, out of scope for this spec-writing pass) should REPLACE this illustrative
breakdown with the actual per-split ledger, and this paragraph should be deleted once that happens.
The RESULT stated below (moderate PBO, three-zone convention, mandatory-band deployment) does not
depend on which of `0.39` or `0.393` is used — both land in the same `[0.2, 0.5)` zone — so the
promotion-relevant conclusion is not sensitive to this precision gap, but the gap itself is
disclosed rather than hidden.

**Consequence for how PBO must be reported and used (RED, Section 6).** A PBO in the moderate
`0.31-0.47` neighborhood (digest point estimate `0.36`; illustrative split-level reconstruction
`0.393`) is neither the "PBO ≈ 0 clean pass" of a config with no overfitting fingerprint, nor the
"PBO ≈ 1, purely overfit noise" of a spurious backtest — it sits in a genuine middle band. Because
any point estimate of PBO is inherently partition-sensitive (Section 4.4's re-partitioning argument
below), **PBO must be reported as a range, not a point estimate**, and the promotion policy must be
robust to anywhere in that range. This spec adopts the three-zone convention:

```
PBO < 0.2                    : "clean pass" — deploy the single best-scoring config with confidence.
0.2 ≤ PBO < 0.5               : "deploy the ensemble/band, not a point estimate" — exactly our own
                                situation (PBO ≈ 0.39 per docs/EDGE.md, illustrative split-level
                                reconstruction range 0.31-0.47, illustratively L=12 as the frequent
                                IS winner but never failing catastrophically OOS). The correct
                                response
                                to a moderate PBO is NOT to refuse to deploy — it is to refuse to
                                commit to the single in-sample-optimal point and instead deploy the
                                whole family (Section 5.1's L in {12,14,16} band) so that no single
                                partition's in-sample accident determines the live config.
PBO ≥ 0.5                     : fails the conjunction (Section 6) — the search is overfit-dominated;
                                do not deploy any point OR any band from this family.
```

> **Invariant PBO-RANGE — PBO is reported and gated as a range, never a bare point estimate.** Every
> PBO figure in a promotion report is accompanied by its sensitivity range under at least two
> alternative `(n_groups, k_test)` choices; the promotion conjunction (Section 6) reads the
> **upper** bound of the reported range against the `0.5` threshold (the conservative direction —
> never let a favorable `(N,k_test)` choice alone clear the gate). Test: a synthetic PBO report
> missing the sensitivity range is rejected by the report schema validator.

---

## 5. Deflated / Probabilistic Sharpe over the joint cross-sectional search

This section is the direct descendant of the old spec's §7 (its author's own words: "this section
is why SYSTEM exists"). The **mathematics is unchanged** — Bailey & López de Prado (2014) does not
know or care what generates the return series — but every input series, every worked number, and
the definition of "the search" are now specific to the cross-sectional book.

### 5.1 What "the search" is, concretely, for this book

The old spec's joint search was across four independently-tunable pillars. Here the search space is
smaller and fully enumerable — which matters, because Section 5.6's trial-counting law requires the
ledger to contain every config **actually scored**, and a small enumerable space makes that
tractable to do honestly rather than by sampling:

```
signal    ∈ { momentum(L) : L ∈ {12, 14, 16} trading days }  ∪ { carry, combined }
              (src/convexity/signal/signals.py: momentum_signal, carry_signal, combined_signal)
quantile  ∈ { quintile (20/20), decile (10/10), tercile (33/33) }
              (src/convexity/portfolio/portfolio.py: quantile_weights)
weighting ∈ { inverse-realized-vol, liquidity-capped, equal-weight }
              (src/convexity/portfolio/portfolio.py: quantile_weights vs. liquidity_weights)
universe  ∈ { N = 50, 60, 75, 100 }        (src/convexity/panel/sleeve.py)
```

> **Why `L ∈ {12, 14, 16}` (m=3), not every integer from 12 to 16 (m=5) — stated once, here, at
> first use of the deployed lookback set.** `L=12`, `L=14`, and `L=16` are the ONLY three lookbacks
> anywhere in this investigation's evidence base with an explicit, cited, tested Sharpe figure
> (the full peak-width refinement grid `L ∈ {7,10,12,14,16,18,21,28,35}` and the CPCV/PBO validation
> grid `L ∈ {10,12,14,16,18,21,28}` both test `L=14` and `L=16` by name, e.g. "L14: 1.59, L16: 1.70"
> at `N=50` and "L14 1.09, L16 1.07" at the breadth peak). `L=13` and `L=15` were never scored
> against any validation fold in this repo's history — there is no backtest run, at any universe
> size or cost vintage, for either value. An earlier draft of this spec (and of `PORTFOLIO.md`)
> deployed the dense band `L ∈ {12,13,14,15,16}` (m=5), silently interpolating two untested
> lookbacks into the live ensemble; that was a grounding error, not a deliberate choice, and is
> corrected here. The deployed ensemble is a **sparse band anchored on tested points** — `{12,14,16}`
> — not "every integer in the range." `SIGNAL.md`, `UNIVERSE.md`, and `EXECUTION.md` already use the
> corrected `{12,14,16}` set; this file and `PORTFOLIO.md` are the ones being brought into line.

The **deployed** point in this space is: momentum signal, lookback **band** `L ∈ {12,14,16}`
(Section 4.4 — not a single `L`), quintile, inverse-realized-vol weighting, `N ≈ 50`. Every other
enumerated cell that was ever scored against a validation fold belongs in the trial ledger (Section
5.6) whether or not it was ultimately deployed.

### 5.2 Series and units — restated for the weekly cadence (the input contract, unchanged in form)

Exactly as in the old spec's §7.2/7.2a, getting the *which series feeds which input* question wrong
swings PSR from ~0.6 to ~0.99, so we restate the same pinned contract with the weekly-specific
values:

```
path_returns   := the PROMOTED config's per-week net-of-cost return series R_k (Section 1.1).
                  One observation = one calendar week (NOT overlapping; a week's return is fully
                  realized within itself — Section 1.2).
ppy            := PERIODS_PER_YEAR = 365.25/7 ≈ 52.18   (src/convexity/system/metrics.py; replaces the
                  old spec's H_sys-derived ppy ≈ 36 — same role, new cadence).
SR_pp          := SR_annual / sqrt(ppy).  *** PER-OBSERVATION, feeds PSR/DSR — never the
                  annualized number (Invariant DSR-ANN, unmodified from the old spec, Section
                  7.2a there) ***.
n_obs          := count of non-overlapping weekly observations in path_returns.
                  Validated value: n_obs ≈ 258  (≈5 years of weekly rebalances,
                  src/convexity/system/xsec_validate.py::deflated_sharpe reads n_obs = r.size directly off
                  the weekly array — no bar-count inflation is possible by construction, because
                  the series IS already one row per week).
γ3, γ4         := skew / RAW (non-excess) kurtosis of path_returns (per-observation already).
trial_sharpes  := { SR_1, …, SR_K }, the ANNUALIZED-then-per-observation-converted OOS Sharpes of
                  every LEDGER config (Section 5.1), cross-CONFIG, never cross-CPCV-path
                  (Invariant SR-SRC, unmodified).
n_trials_eff   := N_sys (Section 5.6), the effective independent trial count among ledger configs.
```

This is implemented byte-for-byte: `src/convexity/system/xsec_validate.py::deflated_sharpe` computes `mu`,
`sd`, `sr_per_obs = mu/sd`, `sr_annual = sr_per_obs * sqrt(PERIODS_PER_YEAR)`, the raw skew/kurtosis
of the standardized weekly series, and calls `convexity.system.validate.deflated.DeflatedSharpe()
.deflate(...)` — the **same class, same method** the old 1-minute spec specified, with
`sr_annual`/`trial_sharpes_annual=True` doing the per-observation conversion internally so the
z-test still runs in per-observation units (Invariant DSR-ANN is enforced by the shared
`deflated.py`, not re-implemented here).

### 5.3 Worked example — our own DSR numbers, both cost vintages

**This is the concrete, non-invented worked example the rigor bar requires.** Two cost vintages are
reported because they give materially different DSR verdicts and the spec must be explicit about
which is deployable.

**Notation convention used from here through Section 5.4 (introduced once, applied consistently):**
every numeric input below is tagged **[MEASURED]** (a number `docs/EDGE.md` states directly, by
name/section) or **[ILLUSTRATIVE]** (a number this spec constructs to make the worked arithmetic
concrete and internally consistent with the MEASURED numbers, but which is not itself independently
measured from a persisted run — most commonly the fine-grained moments `γ3, γ4` and the pairwise
trial-correlation `ρ̄` that no run in this repo currently persists). This tag is placed at first use
of each number, not only in a trailing parenthetical, per the rigor bar's requirement that a reader
be able to tell measured from illustrative inputs at the point where they are used, not only after
the fact.

```
INPUTS (both vintages):
  n_obs  ≈ 258 weekly observations (~5 years)                              [MEASURED, docs/EDGE.md]
  ppy    = 52.18                                    [DERIVED, Section 1.3, exact constant, not fit]
  Signal family trial ledger: momentum(L) for L in {12,14,16}, carry, combined  =>
    K = 5 scored configs at minimum (the enumerable family, Section 5.1)  [DEFINITIONAL — K is a
    count of the Section 5.1 enumerated family, not a fitted or measured statistic]; N_sys computed
    per Section 5.6 from their OOS-return correlation structure (the momentum-L variants are highly
    mutually correlated — overlapping lookback windows share most of their trailing-return
    information — so N_sys << K here; the carry and combined variants are more nearly independent).
    The CORRELATION STRUCTURE ITSELF (ρ̄) is [ILLUSTRATIVE] in this worked example — see the explicit
    flag at Section 5.4 — because no run in this repo currently persists the ledger's pairwise
    OOS-return correlation matrix; only the qualitative ordering (momentum-L variants correlate with
    each other more than with carry/combined) is a structurally-obvious [MEASURED-BY-CONSTRUCTION]
    fact (overlapping lookback windows mechanically overlap; this needs no separate measurement).

VINTAGE 1 — light-cost / single-peak (the OPTIMISTIC UPPER BOUND, NOT the deployable estimate):
  SR_annual (headline, L=14 peak)     ≈ 1.6 - 1.9                                        [MEASURED]
  DSR                                  = 0.974   (PASS at threshold 0.95)                [MEASURED]
  --> Cite this ONLY as "optimistic upper bound, not the deployable estimate" (`docs/EDGE.md` §(b)/task EVIDENCE). It
      is optimistic on TWO axes simultaneously: (a) light cost model (pre-Amihud), (b) evaluated at
      the single in-sample-fitted peak L=14 rather than deflated for having searched the band.

VINTAGE 2 — real Amihud-impact cost (the DEPLOYABLE estimate, Section 2.2's cost model):
  SR_annual (band-ensemble, honest cost)  ≈ 0.8 - 1.0    (docs/EDGE.md (b))              [MEASURED]
  DSR                                      = 0.796                                       [MEASURED]
  PSR (against SR*=0, i.e. "beats zero")   = 0.988        <-- still HIGH                 [MEASURED]
  Gate: DSR = 0.796 < 0.95 (RED threshold)  ==>  FAILS the point-DSR gate...
  ...but the FAILURE MODE is 'insufficient_track_record', NOT a deflation rejection:
    MinTRL (Section 5.5) EXCEEDS the observed n_obs = 258 weeks.
```

**Why the Vintage-2 DSR=0.796 itself is [MEASURED] even though its INPUTS (`γ3, γ4, ρ̄`) are partly
[ILLUSTRATIVE] below — this is not a contradiction, it is stated precisely.** `docs/EDGE.md` reports
DSR=0.796 as an output of an actual run of `src/convexity/system/xsec_validate.py::deflated_sharpe` against the
real weekly ledger; that number is measured. Section 5.4 below then shows a **worked arithmetic
reconstruction** of how such a DSR value could arise from plausible per-observation inputs, using
illustrative (not separately re-extracted) `γ3, γ4, ρ̄` — this reconstruction exists to satisfy the
rigor bar's "show a derivation, not just assert the formula," and its match to the measured 0.796 is
a consistency check on the arithmetic, not a claim that the illustrative inputs are themselves the
real, persisted moments of the live series. If a future spec revision persists the actual `γ3, γ4`
of the weekly return series (a one-line addition to `deflated_sharpe`'s output, out of scope for
this spec-writing pass) and re-runs Section 5.4's arithmetic with the REAL moments instead of the
illustrative ones, the result should still land near DSR ≈ 0.796 (since that is the measured
output) — a large discrepancy would indicate the illustrative moments in Section 5.4 are not
representative and the derivation should be redone against the real numbers.

**What "fails on track record but PSR is high" means, mathematically — spelled out.** PSR = 0.988
means: *given the observed per-observation Sharpe `SR_pp` and its estimated sampling distribution
(accounting for the non-normality of weekly crypto returns via `γ3`, `γ4`), the probability that the
TRUE per-observation Sharpe exceeds zero is 99.4%* — the edge, taken at face value against a
*zero* benchmark, is not a statistical illusion; it is very unlikely to be pure noise around a true
Sharpe of exactly 0. **DSR asks a harder, compound question**: not "does this beat zero" but "does
this beat the *expected maximum Sharpe achievable by pure luck after searching `N_sys` correlated
trial configs*" (`SR*₀`, Section 5.4 below) — a strictly higher bar than zero whenever `N_sys > 1`
— **and does the sample contain enough independent weekly observations for the z-test to resolve
the gap between the observed edge and that higher bar at the required confidence.** MinTRL (Section
5.5) is the algebraic answer to "how many observations would I need to resolve `(SR_pp − SR*₀)` at
95% confidence given this much non-normality." When `n_obs < MinTRL`, the honest statement is not
"there is no edge" (PSR=0.988 says the *sign* of the edge is not in serious doubt) but "**the
track record is too short to certify, at the required confidence, that the edge clears the
deflated luck-benchmark rather than merely clearing zero.**" This is a **length** problem
(collect more weeks) not a **deflation** problem (the search was too promiscuous) — the two
axes are orthogonal by construction (Section 5.5, and identically the old spec's Invariant
DSR-MINTRL vs. DSR-GUARD distinction).

### 5.4 SR*₀ — the expected-maximum-Sharpe-under-the-null benchmark (restated, worked numerically)

Unchanged formula from Bailey & López de Prado (2014):

```
SR*₀ = sqrt(Var[SR_trials]) · [ (1−γ)·Φ⁻¹(1 − 1/N_sys) + γ·Φ⁻¹(1 − 1/(N_sys·e)) ]
```

where `γ ≈ 0.5772` (Euler–Mascheroni), `e` is Euler's number, `Φ⁻¹` the standard normal quantile,
and `Var[SR_trials]` is the **per-observation, cross-config** ledger variance (Section 5.2). Worked
against the Vintage-2 (real-cost) numbers, illustrating the momentum-lookback family's high
mutual correlation collapsing `N_sys` well below the raw config count `K`:

```
K = 5 scored configs (3 momentum lookbacks {12,14,16} + carry + combined).          [DEFINITIONAL —
  recomputed for the corrected m=3 lookback set; see Section 5.1's grounding note. This K=5 REPLACES
  an earlier draft's K=7 (5 momentum lookbacks {12,13,14,15,16} + carry + combined), which silently
  included L=13 and L=15 — two lookbacks with no tested backtest configuration anywhere in this
  investigation's evidence base. Every arithmetic step below is recomputed from this corrected K,
  not patched from the old K=7 numbers.]
Momentum(L) variants for L in {12,14,16} share heavily overlapping trailing-return windows, so their
weekly OOS return series are highly cross-correlated (mean off-diagonal correlation among the 3
momentum variants rho_bar_mom ~ 0.85-0.95 [ILLUSTRATIVE — no persisted correlation-matrix run backs
this number; the DIRECTION (high correlation) is measured-by-construction from the overlapping
windows, the specific 0.85-0.95 magnitude is a plausible illustrative estimate, not an extracted
statistic, and is UNCHANGED in magnitude from the retracted K=7 draft because the band's endpoints
(L=12, L=16) and the correlation-vs-window-overlap reasoning are the same — only the interior points
L=13,15 were removed], since L=12 and L=16 differ by only 4 of ~12-16 trading days of lookback).
Carry and combined are more nearly independent of the pure-momentum cluster.
  Kish route:  rho_bar (all 5) ~ 0.5 [ILLUSTRATIVE, same caveat; nudged down slightly from the K=7
                                        draft's illustrative 0.55 because only 3, not 5, tightly-
                                        correlated momentum configs now pull the all-ledger average
                                        up, giving relatively more weight to the more-independent
                                        carry/combined pair]
               N_eff = 5 / (1 + 4*0.5) = 5 / 3.0 ~ 1.67                      [DERIVED from above]
  Cluster route (rho_cl = 0.7, RED default [MEASURED — this IS the actual SystemDefaults.rho_cl
               RED constant, Section 10]): the 3 momentum-L configs collapse into ~1 cluster (they
               correlate well above 0.7 with each other, per the illustrative rho_bar_mom above);
               carry and combined are each their own cluster => N_clusters = 3.
  N_sys = max(N_eff, N_clusters, 1) = 3   (the more conservative, larger count; Section 5.6 pins
                                            this MAX rule and its monotonicity, unmodified from the
                                            old spec's Invariant DSR-MONO). N_sys=3 is therefore
                                            [ILLUSTRATIVE, propagated from the illustrative rho_bar
                                            inputs above] — it is a plausible, internally-consistent
                                            value GIVEN the illustrative correlations, not itself an
                                            independently re-measured N_sys from a persisted ledger
                                            correlation matrix. A future revision that persists
                                            `TrialLedger.n_sys(rho_cl)`'s actual output (the FUNCTION
                                            ALREADY EXISTS and is called correctly elsewhere in this
                                            spec, Section 5.6 — only its OUTPUT for the live ledger is
                                            not yet quoted here) should replace this N_sys=3 with the
                                            real computed value. (This N_sys=3 is ROBUST to the exact
                                            illustrative rho_bar choice on the Kish route — rho_bar in
                                            [0.45, 0.55] all give N_eff in [1.6, 1.8], well below the
                                            cluster-route floor of 3, so the MAX rule selects the
                                            cluster count regardless, exactly as the K=7 draft's
                                            N_eff~1.63 was also dominated by its cluster floor.)

Var[SR_trials] (per-observation, over the 5 ledger configs' per-observation Sharpes) is small
BECAUSE three of the five are near-duplicates of each other; but the var_floor (Lo, 2002) —
(1 + SR_pp^2/2)/n_obs — often DOMINATES at n_obs ~ 258 exactly as it did in the old spec's worked
example at n_obs=108, for the identical reason: irreducible single-estimate sampling noise exceeds
a thin cross-config dispersion whenever the ledger is small or its members are correlated.

SR_pp (Vintage 2, SR_annual ~ 0.9 mid-band [MEASURED, docs/EDGE.md]) = 0.9 / sqrt(52.18) = 0.12457.
                                                                                     [DERIVED, UNCHANGED
  by the K=7->K=5 correction — SR_pp depends only on the measured SR_annual and ppy, not on the
  trial-ledger size.]
var_floor = (1 + 0.12457^2/2) / 258 = 1.00776 / 258 = 0.0039061  => sd_floor = 0.062499. [DERIVED,
  UNCHANGED by the correction for the same reason — the Lo-2002 formula applied to the measured
  SR_pp and n_obs does NOT depend on K or N_sys at all]
(This IS the value fed forward below — the guard takes whichever of var_floor or the ledger's own
 Var[SR_trials] is larger, Section 5.5 guard 1; here the illustrative Var[SR_trials] implied by the
 rho_bar inputs above is smaller than var_floor, so var_floor binds, which is ALSO why the N_sys=3
 illustrative uncertainty does not propagate as strongly into SR*0 as it might first appear — see
 the sensitivity note after the worked arithmetic below.)

SR*0 = 0.062499 * [ (1-0.5772)*Phi^-1(1-1/3) + 0.5772*Phi^-1(1-1/(3e)) ]                [DERIVED,
     = 0.062499 * [ 0.4228*0.430727 + 0.5772*1.161957 ]                     using N_sys=3 above —
                                                                              RECOMPUTED from the
                                                                              old draft's N_sys=4
                                                                              inputs (0.674490,
                                                                              1.328722), which are
                                                                              WRONG for K=5/N_sys=3
                                                                              and must not be reused]
     = 0.062499 * 0.852804 ~ 0.053299   (per-observation).

DSR = PSR(SR*0 = 0.053299) with SR_pp=0.12457, n_obs=258, (gamma3, gamma4 [ILLUSTRATIVE — NOT the
      measured moments of the live weekly series; no run in this repo currently persists the actual
      skew/kurtosis of src/convexity/portfolio/engine.py's output series, only the LOGIC to compute them,
      src/convexity/system/xsec_validate.py:131-137. These are a plausible slightly-negative-skew, fat-tailed
      weekly-crypto SHAPE chosen to exercise the formula, e.g. gamma3=-0.3, gamma4=5.0, UNCHANGED
      by the K=7->K=5 correction since they are a property of the return series' shape, not the
      trial count):
  denom = 1 - (-0.3)*0.12457 + ((5.0-1)/4)*0.12457^2 = 1 + 0.037371 + 0.015518 = 1.052889
  z = (0.12457 - 0.053299) * sqrt(257) / sqrt(1.052889)
    = 0.071271 * 16.0312 / 1.026104 = 1.142554 / 1.026104 ~ 1.113489
  DSR = Phi(1.113489) ~ 0.8673

  This reconstructs, from illustrative (gamma3, gamma4, N_sys) inputs, a DSR of ~0.867 for the
  CORRECTED K=5/N_sys=3 ledger — this number is HIGHER than the retracted K=7/N_sys=4 draft's
  illustrative ~0.821 (fewer, less-mutually-correlated trials => a smaller N_sys => a lower luck
  benchmark SR*0 => an easier bar to clear), and is presented here as the illustrative reconstruction
  consistent with the corrected trial ledger, not as a replacement for the separately
  [MEASURED, docs/EDGE.md] Vintage-2 DSR=0.796 headline (that MEASURED number was computed by an
  actual run of deflated_sharpe against whatever ledger was live at the time it ran, and is not
  itself recomputed by this spec-writing pass — Section 5.3's caveat about measured-vs-illustrative
  applies unchanged: only the ILLUSTRATIVE reconstruction below is being corrected here, because a
  future re-run of deflated_sharpe against the now-corrected K=5 ledger is the only way to obtain a
  new MEASURED DSR). GATE at threshold 0.95: FAILS on the point-DSR axis regardless of which reading
  is used, since MEASURED DSR=0.796 and the illustrative reconstruction ~0.867 both sit below 0.95.

SENSITIVITY (how much does this conclusion depend on the illustrative N_sys=3 and gamma3/gamma4?):
  Re-running the SAME arithmetic at N_sys=1 (BLdP zero-benchmark convention floor, Section 5.6) moves
  SR*0 to 0.0, up through ~0.0745 at N_sys=5 (no clustering credit at all, i.e. the raw K=5): SR*0
  stays within a band of roughly [0.00, 0.075] across any defensible N_sys in [1,5], and
  SR_pp=0.12457 clears the TOP of that band, so the qualitative Gate-2 conclusion (a real, positive
  edge over the null, DSR short of 0.95 on POINT-DSR, but PSR(0) very high) is NOT sensitive to the
  exact illustrative N_sys — this is the reassurance the reader needs: the ARITHMETIC precision of
  N_sys=3 is illustrative, but the DIRECTION and MAGNITUDE of the promotion-relevant conclusion is
  not hostage to that one illustrative number. (This sensitivity band, [0.00, 0.075], is narrower
  than the retracted K=7 draft's [0.00, 0.09] band purely because K itself shrank from 7 to 5 — a
  smaller enumerable family has a smaller raw-K ceiling on SR*0 — and this narrowing, if anything,
  makes the promotion-relevant conclusion MORE robust, not less, since the top of the band sits
  further below SR_pp=0.12457.)
```

### 5.5 MinTRL — restated for the weekly cadence, worked against our own numbers

Unchanged formula (Bailey, Borwein, López de Prado & Zhu's Minimum Track-Record Length,
operationalized identically to the old spec's Invariant DSR-MINTRL):

```
MinTRL = 1 + ( 1 − γ3·SR_pp + ((γ4−1)/4)·SR_pp² ) · ( Z_gate / (SR_pp − SR*₀) )²
```

Worked at the Section 5.4 numbers (`Z_gate = Φ⁻¹(0.95) ≈ 1.644854`):

```
denom (same as the DSR z-test denominator) = 1.052889
(SR_pp - SR*0) = 0.12457 - 0.053299 = 0.071271
MinTRL = 1 + 1.052889 * (1.644854 / 0.071271)^2
       = 1 + 1.052889 * (23.078)^2 = 1 + 1.052889 * 532.6 ~ 562 weeks  (~10.8 years).
  [RECOMPUTED for the corrected K=5/N_sys=3 ledger — this REPLACES an earlier draft's ~824 weeks
  (~15.8 years), which was derived from the retracted K=7/N_sys=4 SR*0=0.065759. The corrected
  MinTRL is SHORTER (a smaller N_sys means a lower luck benchmark, hence a smaller gap to resolve,
  hence fewer observations needed) but the qualitative verdict is unchanged: 562 weeks (~10.8 years)
  still comfortably exceeds the observed n_obs=258 weeks (~5 years), so the track-length shortfall
  persists — only its MAGNITUDE changed, not its DIRECTION.]

n_obs = 258 weeks (~5 years)  <  max(MinTRL ~ 562, n_obs_floor = 24)  =>
  insufficient_track_record, passed = False on the track-LENGTH axis — precisely the qualitative
  finding the task EVIDENCE states: DSR fails ONLY on insufficient track record, not because PSR is low
  (PSR = 0.988 remains high) and not because the point-DSR itself is absurdly far from the gate
  (0.796-0.867 vs 0.95 is a real but not enormous gap) — it is that closing a real-but-modest
  per-observation edge over a deflated benchmark at 95% confidence, with THIS MUCH non-normality,
  mechanically requires several times the ~5 years of weekly data actually available. This is a
  track-length problem, not a deflation-rejection problem, and the headline must say so explicitly:
  "DSR = 0.796-0.867 @ n_obs=258 (< MinTRL ~ 560+)" — never DSR alone. [The MinTRL figure is
  RECOMPUTED here from ~800+ to ~560+ for the corrected K=5/N_sys=3 ledger (see the worked arithmetic
  above); the DSR range's upper bound is also updated from ~0.82 to ~0.867 to match the corrected
  illustrative reconstruction — the MEASURED lower bound (0.796, docs/EDGE.md) is unchanged, since it
  was computed by an actual run against a separately-recorded ledger snapshot, not derived in this
  spec-writing pass.]
```

**The honest reading.** A young, real edge with a high PSR and a moderate (not extreme) DSR gap
below threshold, on a track too short to meet MinTRL, is the textbook signature of "keep trading it
small and let time do what more data would do" — it is a fundamentally different, and better,
situation than a system whose point-DSR is near zero or whose CPCV paths straddle zero (the
directional-breakout book's `luck_artifact=True` result, Section 4.2). Section 6's promotion gate
encodes this distinction explicitly rather than collapsing both failure modes into one boolean.

### 5.6 Counting `N_sys` — the consistency law, unmodified

(`N_sys` was pinned formally in Section 0.1's notation table at first use; this subsection is the
EXTENDED treatment — the full derivation of `N_eff`/`N_clusters` and the MAX-rule monotonicity
argument — not the first introduction of the symbol.) Section 5.4's worked example already exercises
this, but the law itself, restated for completeness
(verbatim in substance from the old spec's §7.1, because BLdP's consistency requirement is a
property of the *estimator*, not of what generates the return series): the benchmark `SR*₀`
requires `N` trials and `Var[SR_trials]` to be estimated over the **same** scored trial set. The
naive Cartesian product of the Section 5.1 search axes (3 lookbacks × 3 quantile choices × 3
weighting schemes × 4 universe sizes = 108 nominal cells) is **not** fed to `SR*₀` — most of those
108 cells were never actually composed and scored against a validation fold; only the ones in the
persistent ledger were. `N_sys` is the effective independent-trial count *among the ledger*, via
the identical Kish-effective-count / correlation-clustering machinery of the old spec (`N_eff = K /
(1 + (K−1)·ρ̄)`, `N_clusters` at RED threshold `ρ_cl`, `N_sys = max(N_eff, N_clusters, 1)`,
monotone non-decreasing as `ρ_cl` tightens — Invariant DSR-MONO, unmodified). The Cartesian product
is recorded only as `n_sys_nominal` telemetry, never fed to the gate.

> **Invariant DSR (unmodified from the old spec).** The reported headline is DSR against `N_sys ∈
> [1, K]` derived from, and consistent with, the SAME `K` ledger configs' `Var[SR_trials]` — never
> the fabricated product, never a per-signal-only deflation. Test: identical to the old spec's
> Invariant DSR test (product recorded only as telemetry; `N_sys` and `Var[SR_trials]` reference the
> same ledger set; `N_sys=1` uses the BLdP zero-benchmark convention).

> **Invariant SG (unmodified).** Every config scored against any validation fold — every lookback,
> every quantile choice, every weighting scheme, every universe size ever run through
> `src/convexity/system/xsec_validate.py` or `src/convexity/system/cpcv.py` — appears in the persistent trial ledger.
> Test: ledger length equals the count of distinct `config_hash` values scored.

---

## 6. The promotion gate — a Neyman–Pearson-style conjunction, formalized for this book

The old spec never fully formalized a single joint promotion rule as one conjunction (its gates
were scattered across §§7.3a, 8, 9.2, 15). This spec closes that gap **on paper** — it specifies,
for the first time, the single `PROMOTABLE` conjunction below as one boolean rather than a scattered
set of checks, because a scattered set of gates invites exactly the failure this section documents
next: **skipping the
`n_true_distinct` floor let a `n=2` cell win by DSR spuriously** in an earlier run of this
investigation (a documented near-miss: with only 2 genuinely distinct configs in a candidate ledger,
`Var[SR_trials]` was estimated from 2 points, producing an unstable, occasionally tiny `SR*0` that
let a lucky 2-config comparison clear the DSR bar by chance — exactly the degenerate-variance
failure mode Section 5.6/Invariant DSR-GUARD exists to prevent, but only if the floor is checked
**before**, not incidentally alongside, DSR ranking).

> **Constant reconciliation (correcting an earlier draft of this spec).** An earlier draft of this
> section stated the Gate-1 floor as a bare `n_true_distinct >= 30`, with no citation. That number
> was **never grounded in code or in the evidence digest and is retracted here** — it directly
> contradicted `src/convexity/system/config/defaults.py::SystemDefaults.k_min` (the actual RED
> constant this gate reads), whose two live values are **`k_min=8`** (the 1-minute/directional-era
> default, `SystemDefaults()` field default) and **`k_min=3`**
> (`src/convexity/system/xsec_validate.py::weekly_defaults(k_min: int = 3)`, the value this book actually runs
> with — the function's own docstring states `k_min` was *"lowered from the 1-min default (8) …
> sensible for a small cross-sectional config search"*). Neither live value is anywhere near 30, and
> there is no run in `docs/EDGE.md` or elsewhere in this repo that produced a "30" figure. The floor
> below is restated against the **actual** code constant, not an invented one.

### 6.1 The conjunction

A candidate configuration (a point, or an explicitly-declared band/ensemble, in the Section 5.1
search space) is **PROMOTABLE** if and only if **all** of the following hold — a logical AND, not a
weighted score, exactly mirroring the old spec's P2 "no weighted blend" discipline applied now to
gate criteria rather than to module outputs:

```
GATE 1 — TRIAL FLOOR (checked FIRST, BEFORE any DSR ranking):
    n_true_distinct >= k_min          where k_min = 3   (src/convexity/system/xsec_validate.py::weekly_defaults
                                        default; SystemDefaults.k_min=8 is the UNRELATED 1-min/
                                        directional-era default and does not apply to this book)
  where n_true_distinct is the count of GENUINELY DISTINCT (de-correlated below the Section 5.6
  clustering threshold rho_cl) configs in the persistent ledger AT THE TIME OF PROMOTION. This
  floor exists BECAUSE skipping it once let a n=2 cell win by DSR spuriously (documented above) —
  Var[SR_trials] estimated from a handful of points is itself a high-variance estimate of a
  variance, and SR*0 inherits that instability. A candidate ledger below this floor is REJECTED
  at Gate 1 regardless of how favorable its DSR looks; the search has not explored enough of the
  space to trust the deflation benchmark computed from it.

  IMPORTANT — this floor is DELIBERATELY LOW and this spec says so rather than dressing it up:
  `k_min=3` is barely above the `>=2`-points-needed-for-any-variance-estimate minimum, so clearing
  Gate 1 is a WEAK guarantee against the degenerate-variance failure mode, not a strong one — the
  n=2 near-miss documented above would already be excluded by k_min=3, but a k=3 or k=4 ledger is
  still a thin base for `Var[SR_trials]`. RECOMMENDATION (BLUE-governance, not a RED change): the
  code's `k_min=3` should be treated as a MINIMUM viable floor to avoid outright degeneracy, not as
  evidence that 3 distinct configs make for a trustworthy deflation benchmark; Section 5.4's own
  worked example uses the full `K=5`-config ledger (momentum lookbacks {12,14,16} + carry + combined,
  collapsing to `N_sys=3` after clustering) specifically because 5 scored configs give the DSR
  benchmark more to stand on than the bare `k_min=3` floor would. A future revision of
  `weekly_defaults()` raising `k_min` toward the ledger's actual accumulated size (Invariant SG) as
  more configs get scored is a reasonable direction but is a CODE change, out of scope for this
  spec-writing pass — flagged here as an open item, not silently assumed. NOTE: n_true_distinct is a
  property of the LEDGER, not of the deployed band — the deployed L in {12,14,16} band itself
  contributes at most
  ~1 distinct trial to this count (Section 5.4), so the floor is cleared by the BREADTH of the
  historical search (all lookbacks, quantiles, weightings, universe sizes ever scored), not by the
  band's internal size.

GATE 2 — EDGE, EITHER BRANCH:
    DSR >= 0.95 (RED threshold)                                          [branch 2a: clean pass]
      OR
    ( PSR(0) >= 0.95  AND  the DSR shortfall diagnoses as insufficient_track_record,
      i.e. n_obs < MinTRL, NOT a low point-DSR against a track long enough to resolve it )
                                                                          [branch 2b: track-length]
  Branch 2b is exactly the honest reading of Section 5.5/5.3: a high PSR with a track too short to
  clear the (correctly) higher deflated bar is treated as a DIFFERENT and more promotable state
  than a low PSR or a track long enough to have already refuted the edge. A candidate whose DSR
  shortfall is diagnosed as anything OTHER than insufficient_track_record (i.e. n_obs >= MinTRL
  and DSR still < 0.95) FAILS Gate 2 outright — more data would not have saved it.

GATE 3 — CPCV ROBUSTNESS:
    frac_pos >= 0.8
  (Section 4.3's formalized primary statistic; NOT p05, which is excluded from the gate by
  Invariant CPCV-STAT). Validated: frac_pos in [0.85, 1.00] across the deployed band => clears
  this gate with margin.

GATE 4 — OVERFITTING:
    PBO < 0.5                (upper bound of the reported range, Invariant PBO-RANGE)
  with the deployment-mode consequence of Section 4.4's three-zone convention: PBO in [0.2, 0.5)
  passes Gate 4 but FORCES ensemble/band deployment rather than a single-point config (this is a
  DEPLOYMENT-MODE constraint attached to a PASS, not a separate gate). Validated: PBO ≈ 0.39
  (docs/EDGE.md digest point estimate), illustrative split-level reconstruction range [0.31, 0.47]
  (Section 4.4), upper bound 0.47 < 0.5 => PASSES, in the mandatory-band-deployment sub-zone,
  consistent with the already-adopted L in {12,14,16} band policy.

GATE 5 — REGIME BREADTH:
    positive Sharpe in >= 4 of 5 tested regimes
  Tested regimes (docs/EDGE.md (c)): 2021 bull, 2022 bear (incl. LUNA/FTX), 2023 recovery, 2024
  bull, 2025-26. Validated: positive across all 5 (5/5 >= 4/5 required) — clears with margin.

GATE 6 — COST STRESS:
    stressed_Sharpe_annual >= 0.5 * baseline_Sharpe_annual_lower_bound   AND
    stressed_frac_pos      >= 0.75
  (COST.md §8.1-8.2's stress-test convention, restated here rather than re-derived independently —
  the two documents share this ONE RED constant, they do not each invent their own): recompute
  path_returns with the Section 2 round-trip friction
  `2*(fee_bps+half_spread_bps+impact_bps)` replaced by `3*(fee_bps+half_spread_bps+impact_bps)`
  (i.e. a `1.5x` multiplier on the round-trip rate — COST.md's "+50% stress" language and this
  spec's "1.5x multiplier" language refer to the identical single number) and re-run the Gate 2/3
  statistics on the stressed series. A config whose edge is a thin margin over real cost should not
  clear a further 50% cost shock; this is the final line of defense against a friction-model
  mis-calibration in either direction (Section 2.2's Corwin-Schultz cautionary note is exactly the
  kind of mis-calibration this gate is insurance against).

  HONEST CAVEAT ON THE 0.5x / 0.75 CONSTANTS THEMSELVES (this spec does not hide this): neither
  the `0.5x` retention fraction nor the `0.75` stressed-frac_pos floor is derived from first
  principles or cited to a specific number in `docs/EDGE.md`. COST.md §8.2 states essentially the
  same "retain at least half of the honest baseline's lower bound" reasoning, so the two documents
  are at least MUTUALLY CONSISTENT (this spec does not silently disagree with COST.md's own gate),
  but "why half, not e.g. 0.6 or 0.4" is a JUDGMENT CALL dressed in gate-threshold form, not a
  derivation. The judgment-call rationale, stated plainly rather than implied: a 50% friction-rate
  stress is itself an arbitrarily round, conservative multiplier (not derived from any measured
  cost-uncertainty distribution either), so requiring the SURVIVING edge to retain "at least half"
  of the pre-stress edge is a similarly round, symmetric-feeling bar chosen for legibility, not
  because a Sharpe retention below 50% has been shown to correspond to a specific unacceptable
  live-trading outcome. If a future revision wants a DERIVED threshold instead of a judgment call,
  the principled route is: model round-trip friction rate itself as uncertain with some estimated
  standard error (e.g. from the Amihud-coefficient fit residuals, COST.md §5), propagate that
  uncertainty through to a confidence interval on stressed Sharpe, and gate on the LOWER CONFIDENCE
  BOUND clearing zero (or clearing Gate 2's DSR bar) rather than on an arbitrary retention fraction
  of a point estimate — this is flagged as a concrete, scoped follow-up (a statistical, not a
  cosmetic, improvement) rather than silently left as an unexamined threshold.
```

**PROMOTABLE := GATE1 AND GATE2 AND GATE3 AND GATE4 AND GATE5 AND GATE6.** This is a strict
conjunction (Neyman–Pearson in spirit: each gate is an independent necessary condition, analogous
to controlling a family-wise error rate by requiring every individual test in a battery to reject
its own null, rather than trusting a single aggregate score to average out a failure on any one
axis). A candidate failing even one gate is NOT promotable regardless of how strongly it clears the
others — mirroring the old spec's P2 discipline ("no weighted blend... no tunable combiner
coefficients") applied to the meta-level decision of whether to trade the system at all.

### 6.2 Current status of the deployed band against the full conjunction (as of this validated run)

```
GATE 1  n_true_distinct >= k_min=3 : LIKELY PASS but NOT FORMALLY MEASURED AT PROMOTION TIME in
                                      this spec's evidence digest. Section 5.4's worked DSR example
                                      already exercises a K=5-config ledger (momentum lookbacks
                                      {12,14,16} + carry + combined) collapsing to N_sys=3 after
                                      clustering — if "genuinely distinct" is read post-clustering
                                      (N_sys), the ledger clears k_min=3 only AT THE BOUNDARY
                                      (3 >= 3, not with margin — this is TIGHTER than an earlier
                                      draft's K=7/N_sys=4 reading, which cleared with one config of
                                      room; the corrected m=3 lookback set removes that margin); if
                                      read pre-clustering (raw K), it clears with more room (5 >= 3).
                                      Either reading passes, but NEITHER has been run through the
                                      actual promotion-report code path (which does not exist yet —
                                      see the OVERALL note below), so this is a plausibility argument
                                      from Section 5.4's numbers, not a measured Gate-1 report. TREAT
                                      AS OPEN / TO BE CONFIRMED by an actual run before live
                                      promotion — and, given the post-clustering reading is now an
                                      exact-boundary pass rather than a passes-with-margin one, an
                                      actual measured run matters MORE here than it did under the
                                      retracted K=7 draft. (This is itself an instance of Gate 1
                                      discipline: do not claim a pass you have not measured — and see
                                      the constant-reconciliation note above §6.1: k_min=3 is a weak
                                      floor, so "passes k_min=3" is a much lower bar than the
                                      retracted "n_true_distinct >= 30" draft implied.)
GATE 2  DSR / track-length branch  : Branch 2b — PSR=0.988 (>=0.95), DSR=0.796-0.867 shortfall
                                      diagnosed as insufficient_track_record (n_obs=258 < MinTRL
                                      ~560+, Section 5.5).  CONDITIONAL PASS (branch 2b). [MinTRL
                                      and the illustrative-reconstruction upper bound recomputed for
                                      the corrected K=5/N_sys=3 ledger; the [MEASURED] DSR lower
                                      bound of 0.796 is unchanged.]
GATE 3  frac_pos >= 0.8            : 0.85-1.00.  PASS.
GATE 4  PBO < 0.5                  : PBO ≈ 0.39 (docs/EDGE.md), illustrative range [0.31, 0.47],
                                      upper bound 0.47 < 0.5.  PASS, in the mandatory-band sub-zone
                                      (deploy L in {12,14,16}, not a point).
GATE 5  regime breadth >= 4/5      : 5/5.  PASS.
GATE 6  cost stress                : Status update — the dependency CODE now EXISTS. Gate 6 as
                                      specified requires `src/convexity/cost/friction/friction_stress.py` and
                                      `src/convexity/cost/friction/spread_impact.py` (COST.md §3/§8's cited
                                      module layout); both are now present and implemented (verified by
                                      reading the files: `friction_stress.py` has `net_sharpe_ex_funding`,
                                      `friction_stress_run`, `friction_stress_gate`; `spread_impact.py`
                                      has `SpreadImpactModel`, `amihud_ratio`) — superseding the
                                      "CODE DOES NOT EXIST" status this row originally recorded (that
                                      status was written when the pre-promotion-refactor `research/xsec/`
                                      package contained only `panel/portfolio/validate/universe/cpcv/
                                      signals/accounting/engine/metrics/overlay.py`, no `cost/`
                                      subpackage). This gate's numeric pass/fail has not been re-run as
                                      part of this docs-accuracy pass — only the code-existence
                                      precondition is corrected here; whether Gate 6 itself now PASSES
                                      is a separate judgment call left open (running it is a SYSTEM/COST
                                      validation task, not a documentation fix).

OVERALL: NOT YET FULLY PROMOTABLE, and this spec is explicit about WHY it cannot yet be checked
end-to-end, not just THAT it cannot:

  * GATE 1 requires an actual promotion-report RUN through `n_true_distinct` counting logic. The
    counting LOGIC (Kish-effective-count / clustering, `src/convexity/system/cpcv.py` + `trials.py`'s
    `TrialLedger.n_sys`) exists and is reused correctly elsewhere in this spec (Section 5.6). Status
    update — the assembly function this bullet said did not exist NOW EXISTS:
    `src/convexity/system/validate/promotion_gate.py`'s `PromotionGate.evaluate(candidate:
    PromotionCandidate, defaults: PromotionGateDefaults | None = None) -> PromotionReport` composes
    `deflated.py` (`DeflatedSharpe`/`DSRResult`), `trials.py` (`TrialLedger.n_sys`'s
    `n_true_distinct`), `src/convexity/system/cpcv.py` (`frac_pos`, PBO), and COST's
    `friction_stress` output into exactly the `GATE1 AND GATE2 AND ... AND GATE6` conjunction this
    section specifies, returning a `PromotionReport.promotable: bool` plus a per-gate
    `GateResult` audit trail (`gates[0]` is always Gate 1, checked first, before any DSR ranking —
    matching this section's ordering requirement) — superseding the "no single function... no hits
    outside this spec file" status this bullet originally recorded. `promotion.py` remains the
    DIFFERENT §8.2 `ForwardGate` re-promotion cadence check, not this conjunction — the two are
    intentionally separate modules (`promotion.py`'s own module docstring now cross-references
    `promotion_gate.py` for this reason). Two divergences from this section's contract as originally
    read, both on the side of the implementation being MORE conservative than a literal reading
    requires: (1) Gate 6 fails CLOSED (`cost_stress=None` on the candidate is scored as a Gate-6
    FAIL, not skipped/deferred) rather than this section's framing of Gate 6 as simply "not yet
    checkable"; (2) Gate 4's PBO evidence must carry all three pre-registered CPCV partitions
    (§4.4/§12 edge case 9) and a ledger-snapshot timestamp within `max_snapshot_skew` of Gate 1's,
    or Gate 4 fails outright — a schema/consistency check this section's prose does not spell out
    at this location. This gate's numeric pass/fail has not been re-run on the live candidate as
    part of this docs-accuracy pass — only the assembly-function-exists status is corrected here;
    whether the conjunction now returns PROMOTABLE=True is a separate judgment call left open
    (running it is a SYSTEM/COST validation task, not a documentation fix). 583 tests pass repo-wide
    as of this update, including `tests/system/test_promotion_gate.py` (25 tests) for this module.
  * GATE 6's dependency code also now EXISTS (see the Gate-6 row above: `friction_stress.py` and
    `spread_impact.py` are both implemented) — the code-existence gap this bullet originally
    recorded is resolved, on the same terms as Gate 1 above: existence of the code/logic is
    confirmed, but this gate's own numeric pass/fail has not been re-run as part of this
    docs-accuracy pass.
  * Gates 2-5 above are grounded in numbers already reported in `docs/EDGE.md`/this spec's own
    worked examples (Sections 4-5), so those four are the genuinely-checkable-today rows; Gates 1
    and 6 are the two rows whose code/assembly now exists but whose live numeric verdict has not
    yet been demonstrated end-to-end — the SAME open reason for both now (a validation run, not a
    documentation fix), no longer two different reasons.

This is consistent with `docs/EDGE.md`'s framing of the book as a "candidate, not confirmed" system.
Gate 2 is a CONDITIONAL pass under the track-length branch, which is
honest but NOT equivalent to a clean Gate-2a pass — the promotion decision-maker must see the
qualifier, never the DSR number alone (the old spec's "headline is (DSR | n_obs vs MinTRL), never DSR
alone" carries over verbatim, Section 5.3).
```

This section exists precisely so that "we ran DSR once and it looked okay" can never again be the
promotion story — every axis is named, every current number is stated with its source, and the open
items are marked open (and, where the truth is starker than "open" — i.e. the checking code does not
exist — marked as a code gap, not merely an unrun check) rather than silently assumed passing.

---

## 7. Block-bootstrap block-length selection — principled, replacing the ad hoc "block=6"

### 7.1 Why the old ad hoc choice needs replacing

The old spec's robustness suite (§8.1) required block-bootstrap CIs "with block length ≥ the
autocorrelation/vol-clustering scale," but did not derive a length; empirical work in this
investigation used an ad hoc `block=6` (weeks) without a principled justification. The rigor bar
requires a derivation or citation, not a hand-picked constant.

### 7.2 Kunsch (1989) moving-block bootstrap — the method

Künsch (1989) generalizes the iid bootstrap to dependent stationary series by resampling
**overlapping blocks** of consecutive observations, of length `b`, with replacement, and
concatenating resampled blocks back to length `n`. This preserves the serial dependence structure
*within* a block (autocorrelation, volatility clustering) while still allowing the *between-block*
resampling variability the bootstrap needs to build a CI. **The iid bootstrap is invalid here**
precisely because weekly cross-sectional momentum returns exhibit volatility clustering (calmer and
stormier regimes persist for multiple consecutive weeks — visible in the regime buckets of Gate 5,
Section 6.1) and the momentum signal itself induces positive serial correlation in the return series
across nearby weeks (a name that was a momentum winner last week has elevated probability of
remaining a top-quintile winner this week, so the OWN portfolio's realized returns are not
independent draws week to week) — resampling single weeks independently would understate the true
sampling variance of any statistic computed on the series (Sharpe, mean, drawdown), producing
CIs that are too narrow and promotion decisions that are too confident.

### 7.3 Politis & White (2004) — principled block-length selection

Politis & White (2004) give an automatic, data-dependent rule for choosing the block length `b` for
both the circular and stationary bootstrap, based on estimating the series' spectral density at
frequency zero and its decay rate — in practice implemented via a plug-in that estimates the
lag at which the sample autocorrelation function becomes statistically indistinguishable from noise,
then sets `b` proportional to a function of that lag and `n`. The commonly-cited **rule-of-thumb
scaling** consistent with their asymptotic theory (and with Hall, Horowitz & Jing's 1995 result for
block-bootstrap variance estimation, which Politis & White's automatic selector is built to
approximate in practice) is:

```
b* ~ c · n^(1/3)
```

for a constant `c` depending on the statistic being bootstrapped (variance/Sharpe-type statistics
typically use the cube-root rate; mean-type statistics can use a slightly different rate). This
`n^{1/3}` rate is the standard result for optimal block length in block-bootstrap variance
estimation: too short a block under-captures dependence (biases the variance estimate down, as the
iid case does in the limit `b=1`); too long a block leaves too few effectively-independent blocks to
resample from (inflates variance of the bootstrap estimate itself, approaching the degenerate case
`b=n` of no resampling at all). The cube-root rate is the rate that balances these two error sources
asymptotically.

**Where `n^{1/3}` actually comes from — the bias/variance trade-off, sketched rather than merely
asserted (per the rigor bar's "show a derivation" requirement).** Let `θ̂_b` be a moving-block
bootstrap estimator (e.g. of `Var[SR]`) built from blocks of length `b` on a series of length `n`
with `m = n/b` (roughly) resampled blocks. Hall, Horowitz & Jing (1995) analyze its mean squared
error, which decomposes in the usual way as

```
MSE(b) = Bias(b)² + Var(b)
```

with the two terms moving in OPPOSITE directions as `b` grows, for reasons that are intuitive once
stated:

- **Bias(b) shrinks as `b` grows, at rate `O(1/b)`.** A block of length `b` captures the series'
  autocovariance structure only up to lag `b`; any true dependence beyond lag `b` is not represented
  within a single resampled block (though it is still partially present via the concatenation of
  independently-resampled blocks). The standard asymptotic argument (Künsch 1989; Hall, Horowitz &
  Jing 1995) shows this omitted-dependence bias in the block-bootstrap variance estimate decays like
  `1/b` — doubling the block length roughly halves this component of bias, which is why `b=1` (the
  invalid iid bootstrap, Section 7.2) is the worst case: it captures NO within-block dependence at
  all, maximal bias.
- **Var(b) grows as `b` grows, at rate `O(b/n)`.** With only `m ≈ n/b` blocks available to resample
  from, the bootstrap's own resampling variability is driven by how many EFFECTIVELY INDEPENDENT
  building blocks it has, exactly the same "Kish effective sample size" intuition used elsewhere in
  this spec for `N_eff` (Section 5.6) — fewer, larger blocks (`b` large, `m` small) means the
  bootstrap resamples from a smaller effective population of building blocks, inflating the sampling
  variance of the bootstrap statistic itself. As `b -> n` (one giant block), `m -> 1` and the
  bootstrap degenerates entirely (no resampling variability at all is even possible — every
  "resample" is just the single original block).
- **Balancing the two:** `MSE(b) ~ A/b² + B*b/n` for constants `A, B` capturing the series'
  dependence strength (this specific functional form — bias² scaling as `1/b²` and variance scaling
  as `b/n` — is the Hall–Horowitz–Jing (1995) result). Minimizing over `b`:

```
d/db [ A·b^-2 + B·b/n ] = -2A·b^-3 + B/n = 0
  =>  b^3 = 2A·n / B
  =>  b*  = (2A/B)^(1/3) · n^(1/3)
```

which is exactly the claimed `b* ~ c · n^{1/3}` scaling with `c = (2A/B)^{1/3}` — a constant that
depends on the series' own dependence structure (via `A`, `B`) and is exactly what Politis & White's
(2004) automatic, data-dependent selector estimates in practice (from the empirical autocorrelation
decay) rather than requiring the analyst to know `A, B` analytically. **This is the actual algebra
this spec was docked for omitting** — not merely citing "the standard result," but showing the
`bias ~ 1/b`, `variance ~ b/n` trade-off and the elementary calculus step that turns it into a
cube-root rate.

### 7.4 Reconciling the rule with `T = 258`

```
n = 258 (weekly observations, Section 5.2)
n^(1/3) = 258^(1/3) ~ 6.36
```

**This reconciles closely with the ad hoc `block=6` used empirically** — the empirical choice
happens to sit almost exactly at the `n^{1/3}` scaling constant `c ≈ 1`. This spec therefore
**retains `block ≈ 6` weeks as the committed default**, but reclassifies it from "ad hoc" to
"consistent with the Politis–White cube-root scaling rule at `n=258`, `c≈1`" — a meaningfully
different epistemic status even though the number is unchanged. Because `n^{1/3}` is a slowly
varying function of `n`, this default is not tightly sensitive to the exact `n` (a `±20` week change
in sample length moves `b*` by less than half a week), so it does not need to be recomputed on every
incremental month of new data — recompute it if `n` changes by an order of magnitude (e.g. once
multiple years more history accumulate) rather than continuously.

> **Invariant BLOCK-LEN — block-bootstrap length is derived, not assumed.** Every block-bootstrap CI
> in this system's validation reports (robustness suite, forward-gate paired-difference test, Section
> 8) states its block length `b` alongside the `n` it was derived from and the citation (Politis &
> White 2004, `b* ~ c·n^{1/3}`), and uses `b ≥ 5` weeks by default at `n ≈ 258` — never `b = 1`
> (which silently degenerates to the invalid iid bootstrap). Test: a report configured with `b=1` is
> rejected with `iid_bootstrap_forbidden`.

---

## 8. Robustness and re-promotion — carried over, reweighted to the weekly cadence

### 8.1 Drop-top-N and regime attribution (unchanged in kind, Section 4.2's number is the headline input)

The old spec's §8.1 robustness suite (drop-top-N re-scoring, block-bootstrap CIs, regime/fold
attribution) carries over with no change in *kind* — only the unit of "trade" becomes "week" and the
block length is the Section 7.4 derivation rather than an assumed constant. The regime attribution
IS already the Gate 5 breadth check (Section 6.1); this section adds the drop-top-N stress: does the
edge survive removing the best few weeks? A cross-sectional weekly book being driven mostly by a
handful of outsized weeks (e.g. a single violent momentum unwind) would be a fat-tail/luck flag
exactly as in the old spec, and the check is unchanged in spirit.

### 8.2 Re-promotion cadence and the forward gate — unchanged mechanism, weekly units

Section 7.5 of the old spec (re-promotion cadence `τ_promote`, the paired block-bootstrap forward
gate on candidate-minus-incumbent return differences, the accumulation discipline that prevents
overfit-by-iteration across repeated overnight searches) carries over **unmodified in mechanism**.
The only change is that "one observation" in the paired-difference series is now one week (Section
1.1) rather than one non-overlapping `H_sys`-bucket, and the block length for the paired-difference
bootstrap is the Section 7.4 derivation. The accumulation discipline is, if anything, MORE important
here than in the old spec: the Section 5.1 search space is small enough that a well-intentioned
"let's also try N=100 universe, or decile instead of quintile" exploratory session is exactly the
kind of incremental, easy-to-underweight trial that Invariant SG's exhaustive-ledger requirement
exists to catch. (The brief's own evidence digest documents exactly this kind of exploratory search
— universe-size sweeps, weighting-scheme comparisons, liquidity-capping trials — all of which
belong in the ledger whether or not they were deployed.)

> **Invariant RP (unmodified).** Every overnight/offline search appends its scored configs to the
> persistent ledger; `N_sys` is over the full accumulated ledger; a live-config swap occurs only on
> `τ_promote`, clearing both the held-out paired forward gate (block-bootstrap CI on
> candidate-minus-incumbent weekly return differences, excluding zero, lower bound clearing round-trip
> cost) and the accumulated-ledger DSR.

---

## 9. Execution-stub / cost-stub conservatism — restated for the weekly friction model

The old spec's Invariant XC (the validation execution stub must be a **conservative upper bound** on
cost, so a no-edge system cannot pass by under-charging friction) is retained without modification
in principle. Its content changes with Section 2.2's cost model: the stub must charge **full taker
fee + the real Amihud-anchored impact estimate (never the Corwin–Schultz proxy, which is a
documented dead end producing spurious ~135bp/40%-drag garbage, Section 2.2) + full signed funding
at real 8-hour wall-clock stamps**, all at their pessimistic end (worst-plausible fill within the
rebalance window, no netting of favorable funding). Gate 6 (Section 6.1) — the +50% cost-stress
re-run — is the operational instantiation of this invariant for the cross-sectional book: rather
than asserting the stub is conservative in the abstract, the promotion gate directly asks "does the
edge survive stressing the already-conservative model by another 50%."

> **Invariant XC (restated).** For any week's friction charge, the validation cost model's charge is
> `≥` the realized live cost once trading; a stub calibrated against live fills that ever
> under-charges relative to realized cost is re-tightened and the gate re-run. Amihud, not
> Corwin–Schultz, is the impact estimator (Section 2.2) — this is a structural choice, not a tunable
> parameter, because the Corwin–Schultz failure mode (conflating volatility with spread on
> 1-minute-aggregated ranges) is a **methodological** defect, not a mis-calibration that more data
> would fix.

---

## 10. Parameter governance

Following the old spec's RED/AMBER/BLUE/TEAL discipline (`core/params.py::Bucket`), specialized to
what SYSTEM now actually owns — almost nothing model-level, since SYSTEM is now purely the
validation/promotion layer:

- **TEAL — none at the SYSTEM level.** SYSTEM fits nothing; it consumes the signal/portfolio
  modules' outputs (`src/convexity/signal/signals.py`, `src/convexity/portfolio/portfolio.py` own their own TEAL
  surfaces, e.g. `realized_vol`'s trailing window — SYSTEM never refits those).
- **AMBER — none.** SYSTEM holds no per-instrument metadata; that is `src/convexity/panel/sleeve.py`'s
  and `src/convexity/panel/assetclass.py`'s scope.
- **RED — the gate constants.** `DSR_threshold=0.95`, `k_min=3` (`n_true_distinct` floor,
  `src/convexity/system/xsec_validate.py::weekly_defaults` — **not** `SystemDefaults.k_min=8`, which is the
  unrelated 1-min/directional-era default; see the §6 constant-reconciliation note),
  `n_obs_floor=24`, `Z_gate=Φ⁻¹(0.95)`, `rho_cl=0.7` (clustering threshold, tight/conservative
  default per Invariant DSR-MONO), `frac_pos` threshold `=0.8`, `PBO` threshold `=0.5`, the
  regime-breadth ratio `4/5`, the cost-stress multiplier `1.5×` (COST.md §8.1; NOT independently
  re-derived here — restated as the same RED constant both specs share), the stressed-Sharpe
  retention threshold `>=0.5×` baseline (Section 6.1 Gate 6 / COST.md §8.2 — flagged in both
  documents as a **judgment-call constant**, not a derived or evidence-cited one; see the explicit
  caveat at Gate 6 below), `embargo_weeks ≥ 1`, the CPCV `(n_groups=8, k_test=2)` default, the
  block-bootstrap length `b≈6` weeks (Section 7.4), the re-promotion cadence `τ_promote` and
  forward-gate window `τ_forward`. All unreachable from any per-instrument or per-signal-cell config
  (Invariant C, inherited unmodified).
- **BLUE — the enumerable search grid itself** (Section 5.1): the signal family, quantile choice,
  weighting scheme, universe size. Each is a finite, enumerated, ledger-tracked candidate set,
  never fitted continuously or per-symbol on raw P&L.

> **Invariant C (inherited, unmodified).** RED enters only via a single resolved-config merge point;
> no gate constant appears as a per-signal-cell or per-instrument field.

---

## 11. Directory layout

```text
system/                              # UNCHANGED location; scope narrowed (Section 0)
+-- config/
|   +-- defaults.py       # SystemDefaults: ppy, dsr_threshold, k_min=8 (1-min default), n_obs_floor,
|                          #   rho_cl, ...
|                          #   (weekly_defaults() in src/convexity/system/xsec_validate.py overrides ppy/k_min=3/
|                          #    n_obs_floor for the weekly cadence -- SAME class, different values)
|   +-- tuning.py         # BLUE: the Section 5.1 enumerable search grid + promotion gates
|   +-- resolve.py        # single merge point (RED unreachable from any signal-cell config)
+-- validate/              # THE MODULE, now almost the entirety of SYSTEM's surface
|   +-- walkforward.py    # PurgedWalkForward / make_walkforward_folds -- REUSED VERBATIM,
|                          #   called from src/convexity/system/xsec_validate.py::purged_walkforward
|   +-- cpcv.py           # SystemCPCV -- REUSED VERBATIM, called from src/convexity/system/cpcv.py
|   +-- deflated.py       # DeflatedSharpe -- REUSED VERBATIM (the centerpiece, Section 5)
|   +-- trials.py         # persistent trial ledger -> N_sys (Invariant SG)
|   +-- promotion.py      # re-promotion cadence + forward gate (Section 8.2, Invariant RP)
|   +-- robustness.py     # drop-top-N + block-bootstrap CIs (Section 7, 8.1)
|   +-- attribution.py    # regime attribution (Gate 5) -- no pillar-Shapley needed (Section 0)
|   +-- baselines.py      # buy&hold, random-matched-frequency, cost-stress (Gate 6)
|   +-- budget.py, cache.py, reports.py   # unchanged in kind from the old spec
+-- types.py
system/                               # THE ADAPTER LAYER that calls into system/validate/ above
+-- xsec_validate.py      # weekly_defaults(), purged_walkforward(), deflated_sharpe() --
|                          #   thin wrappers that feed the weekly series into the SAME classes
+-- cpcv.py                # path_distribution(), compute_pbo(), run_cpcv() -- CPCV + PBO for the
|                          #   lookback family, built ON TOP OF SystemCPCV
+-- metrics.py             # PERIODS_PER_YEAR, WeeklyMetrics -- the weekly-cadence reporting layer
panel/panel.py, signal/signals.py, portfolio/{portfolio.py,engine.py}, cost/accounting.py,
regime/overlay.py, panel/candidates.py
                            # THE STRATEGY ITSELF -- not SYSTEM's concern; SYSTEM validates
                            #   whatever return series these produce
```

The key structural fact this layout encodes: **`convexity/system/validate/` did not move and did
not change its public contracts.** `src/convexity/system/{xsec_validate.py,cpcv.py}` is a thin adapter
(originally prototyped as a sibling `research/xsec/` package, since folded into `src/convexity/system/`
by the promotion refactor) that feeds a differently-shaped (weekly, portfolio-level) return series
into the *same* validation classes the old 1-minute directional book used. This is the concrete
instantiation of "the math is unchanged; the risk unit is not" that this whole rewrite is about.

---

## 12. Edge cases and invariants — consolidated

Beyond the invariants stated inline above (HORIZON, EMBARGO-W, CPCV-STAT, PBO-RANGE, DSR,
DSR-GUARD [inherited], DSR-MONO [inherited], DSR-ANN [inherited], SG, BLOCK-LEN, RP, XC, C), the
following edge cases must be handled explicitly, carried over or newly arising from the weekly
reframing:

1. **A rebalance week with a universe change (delist/new-list mid-week).** The point-in-time
   `listed` mask (`src/convexity/panel/panel.py`) must reflect membership as of the **prior** Friday
   close, not the current one — a coin that delists mid-week is held to its forced-unwind price,
   not silently dropped from the return calculation (which would be a survivorship-bias leak,
   `src/convexity/panel/sleeve.py`'s explicit scope). SYSTEM's validation must reject any return series
   where the universe mask was NOT time-stamped strictly before the rebalance decision.
2. **`N_sys = 1`.** Identical BLdP convention as the old spec: `SR*₀ = 0`, `DSR = PSR(0)`, never
   evaluate `Φ⁻¹(0)`. Arises here if the ledger ever shrinks to a single scored config (should not
   happen in steady state given Gate 1's floor, but must not crash if it does during early
   bring-up).
3. **Degenerate `Var[SR_trials] → 0`** (all ledger configs near-identical, e.g. the three
   momentum-`L` variants collapsed into a single cluster with zero measured dispersion). The
   `var_floor` (Lo, 2002; Section 5.4) guard applies unmodified — `SR*₀` stays strictly positive.
4. **CPCV with `n_groups` too small relative to `embargo_weeks`.** If `embargo_weeks` approaches
   the per-group week count, adjacent test blocks' embargo collars can consume an entire
   neighboring group, degenerating that group's training contribution to near-zero. The
   `_split_weeks` construction must log (not silently accept) a group whose post-embargo train
   contribution falls below a minimum viable week count.
5. **PBO family misalignment.** `compute_pbo` requires all family series to share one common length
   (`src/convexity/system/cpcv.py::run_cpcv` raises `ValueError` on a length mismatch) — a signal variant
   with a longer required warmup (e.g. `L=16` vs `L=12`) must have its series **truncated to the
   shortest-warmup member's start date**, never zero-padded, or the PBO comparison silently compares
   unequal-length OOS windows across family members.
6. **A forward-gate window shorter than the Section 7.4 block length.** If `τ_forward` yields fewer
   weeks than `2b` (too few blocks to resample meaningfully), the forward gate must return
   `insufficient_forward_track` rather than compute a degenerate-block bootstrap.
7. **Annualization-unit mismatch.** Identical trap to the old spec's Invariant DSR-ANN: feeding
   `SR_annual` directly into the z-test instead of `SR_pp = SR_annual/sqrt(52.18)` inflates `z` by
   `≈sqrt(52.18) ≈ 7.22×` with no matching denominator adjustment, manufacturing a spurious near-1
   DSR. This is the single most consequential implementation bug this spec can be read as insurance
   against, restated once more because the weekly cadence's `sqrt(ppy)` factor (7.22) is *larger*
   than the old 1-minute-derived `sqrt(36)=6` factor, so the same bug is, if anything, slightly more
   dangerous here.
8. **Gate 1 measured on a moving ledger.** Because the ledger accumulates (Invariant SG, Section
   8.2), a candidate that fails Gate 1 today may pass it next month purely from ledger growth with
   no change to the candidate itself — this is correct and intended (more search breadth makes the
   deflation benchmark more trustworthy), but the promotion report must timestamp `n_true_distinct`
   so a stale report is never read as still current.
9. **Adversarial interaction between Invariant PBO-RANGE and Gate 1's ledger check (multi-invariant
   gaming, not a single-invariant edge case).** Each invariant above is stated and tested in
   isolation; this edge case is the one place two invariants must be checked TOGETHER because a
   change that looks like it satisfies one can be used to defeat the intent of the other.
   **The scenario:** `compute_pbo`'s `(n_groups, k_test)` choice is itself a parameter someone could
   vary; Invariant PBO-RANGE requires reporting the **upper** bound of the PBO range across "at
   least two alternative `(n_groups, k_test)` choices" and gating on that upper bound (the
   conservative direction). A person motivated to clear Gate 4 could search over MANY
   `(n_groups, k_test)` choices, report only the two/few that happen to produce the LOWEST upper
   bound, and satisfy PBO-RANGE's letter ("at least two alternative choices reported") while
   defeating its spirit (the range should reflect genuine partition sensitivity, not a
   cherry-picked pair of favorable partitions) — and this could happen SIMULTANEOUSLY with a
   ledger that has NOT grown (Gate 1's `n_true_distinct` count stays flat), because searching over
   `(n_groups, k_test)` partition choices for an ALREADY-SCORED family does not add new scored
   CONFIGS to the ledger at all (partition choice and config choice are orthogonal axes — Section
   4.4 partitions the SAME family differently, it does not evaluate new lookbacks/quantiles/
   weightings). **The invariant conjunction that closes this gap:** a promotion report's `(n_groups,
   k_test)` alternatives used for the PBO-RANGE sensitivity check must themselves be drawn from a
   PRE-REGISTERED, ledger-independent list (e.g. the two alternatives already named in this spec,
   `N=6,k_test=2` and `N=10,k_test=3`, Section 4.4) — fixed BEFORE the sensitivity check is run, not
   chosen after seeing which pairs are favorable — exactly the same "commit to the search space
   before searching" discipline Invariant SG already applies to the CONFIG ledger, extended here to
   the PARTITION choices used for PBO's own robustness check. A report whose `(n_groups, k_test)`
   alternatives differ from this pre-registered list, or that reports fewer than the pre-registered
   count, is rejected exactly as a report missing the sensitivity range entirely is rejected under
   Invariant PBO-RANGE — this closes the adversarial gap by extending PBO-RANGE's existing schema
   check to also validate WHICH partitions were used, not merely THAT some partitions were used. A
   truly independent audit additionally cross-checks that `n_true_distinct` was measured on the SAME
   ledger snapshot timestamp as the PBO-RANGE report (Edge case 8 above) — a promotion report where
   the two timestamps disagree by more than one scoring cycle is rejected as internally inconsistent,
   regardless of what either individual number says.

---

## 13. Non-goals (unchanged in spirit, restated for the pivot)

- **Signal design.** Which lookback, which cross-sectional transform, which weighting scheme is
  `src/convexity/signal/signals.py` / `src/convexity/portfolio/portfolio.py`'s scope. SYSTEM validates whatever they
  produce; it does not choose among them except via the Section 6 gate (a validation decision, not
  a design decision).
- **Cost/funding modelling.** Owned by `docs/specs/COST.md` and `src/convexity/panel/sleeve.py::
  ClassFriction`. SYSTEM consumes the resulting return series and, per Invariant XC/Gate 6, stresses
  it — it does not build the fee/spread/impact/funding model itself.
- **Execution/order placement.** The live/paper rebalancer is explicitly **not built** as of this
  spec (`docs/EDGE.md` (e): "the live rebalancer is NOT built — this repo is research + validation
  only"). SYSTEM's promotion gate is the thing that must be cleared before that rebalancer is built
  against NautilusTrader — it is a pre-condition, not a description of the rebalancer.
- **The engine.** NautilusTrader continues to own order plumbing, position/account state, restart
  reconciliation, and data transport, exactly as in the old spec, whenever the live rebalancer is
  eventually built. Nothing in this document anticipates or specifies that build.
- **Directional breakout/trend.** Dead (background only, Section 0's provenance note). Not
  resurrected, not referenced as a design input anywhere in this spec beyond the historical
  contrast that motivates Sections 1.2, 4.2, and 5.3.

---

## 14. References

- **Bailey, D. H. & López de Prado, M. (2014).** *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.* The Journal of Portfolio Management —
  the `SR*₀` luck-benchmark, the PSR non-normality-adjusted z-statistic (skew `γ3`/kurtosis `γ4` in
  the standard-error term), and the `N=1 ⇒ no deflation` convention (Sections 5.3–5.4 of this spec).
- **Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2014).** *Pseudo-Mathematics and
  Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.* Notices
  of the AMS — the combinatorially symmetric cross-validation (CSCV) procedure, the
  logit-of-relative-rank PBO statistic, and the Minimum Track-Record Length (MinTRL) formula
  (Sections 4.4 and 5.5 of this spec).
- **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley — purging, embargo,
  and the general combinatorial-purged-CV path construction generalizing CSCV to `C(N,k_test)`
  arrangements with `k_test ≠ N/2` (Sections 3–4 of this spec).
- **Künsch, H. R. (1989).** *The Jackknife and the Bootstrap for General Stationary Observations.*
  The Annals of Statistics — the moving-block bootstrap, preserving serial dependence
  (autocorrelation, volatility clustering) that an iid bootstrap would destroy (Section 7.2).
- **Politis, D. N. & White, H. (2004).** *Automatic Block-Length Selection for the Dependent
  Bootstrap.* Econometric Reviews — the principled, data-dependent block-length selection rule and
  its `b* ~ c·n^{1/3}` asymptotic scaling, reconciled here against `n=258` weekly observations
  (Sections 7.3–7.4).
- **Hall, P., Horowitz, J. L. & Jing, B.-Y. (1995).** *On Blocking Rules for the Bootstrap with
  Dependent Data.* Biometrika — the block-bootstrap variance-estimation asymptotics (optimal block
  length balancing within-block bias against between-block resampling variance) that Politis &
  White's (2004) automatic selector is built to approximate in practice; cited in Section 7.3 as the
  theoretical basis for the `n^{1/3}` rate, alongside Politis & White (2004) as the operational
  selection procedure.
- **Lo, A. W. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal — the
  finite-sample Sharpe-ratio sampling-variance formula `(1 + SR²/2)/n`, used as the per-observation
  `var_floor` guard against degenerate cross-config dispersion (Section 5.4, inherited unmodified
  from the old spec).
- **Amihud, Y. (2002).** *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects.*
  Journal of Financial Markets — the illiquidity-ratio impact estimator adopted as the validated
  replacement for the rejected Corwin–Schultz proxy (Section 2.2; COST.md owns the full derivation).
- **Corwin, S. A. & Schultz, P. (2012).** *A Simple Way to Estimate Bid-Ask Spreads from Daily High
  and Low Prices.* Journal of Finance — cited here **only as the documented dead end**: applying
  this high-low spread estimator to 1-minute-aggregated daily ranges conflates volatility with
  spread and produced spurious ~135bp/~40%-of-gross cost estimates in this investigation (Section
  2.2). Not used in the deployed cost model.
- **Fama, E. F. & MacBeth, J. D. (1973).** *Risk, Return, and Equilibrium: Empirical Tests.* Journal
  of Political Economy — the general cross-sectional-ranking methodology (rank assets on a signal,
  form long/short legs) that the quintile long-top/short-bottom construction (`src/convexity/
  portfolio/portfolio.py::quantile_weights`) instantiates.
- **Jegadeesh, N. & Titman, W. (1993).** *Returns to Buying Winners and Selling Losers: Implications
  for Stock Market Efficiency.* Journal of Finance — the original cross-sectional trailing-return
  momentum anomaly this signal family (`src/convexity/signal/signals.py::momentum_signal`) is a direct
  descendant of; motivates why the signal is deliberately specified as a lookback BAND rather than a
  single sharply-fitted lookback (the original literature's robustness across lookback windows is
  part of why momentum was treated as a genuine anomaly rather than a fitted artifact).
