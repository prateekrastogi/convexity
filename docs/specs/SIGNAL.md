# SIGNAL.md — build spec for the cross-sectional score (momentum + funding carry)

This file is the authoritative build spec for the SIGNAL pillar of the **crypto cross-sectional
momentum** system (the convexity repo's post-pivot core — see `docs/EDGE.md`). Build to the
contracts and invariants below. Where this file and your own instinct disagree, this file wins.
Ask before deviating from any **Invariant**.

**The risk unit changed and every formula in this file is scoped to it.** The old
`docs/specs/{COST,SYSTEM}.md` were written for a single-trade, stop-to-target directional book
whose risk unit was one trade's R-multiple (entry → stop/target → exit). That book is
**PROVEN DEAD**: gross directional R ≈ 0 across every timeframe, exit rule, entry filter, leverage
tier, and three asset classes, even inverted (`docs/EDGE.md` §a). It is cited here only as
background for salvaged math (cost decomposition, DSR/CPCV machinery), never as the subject.

The new risk unit is **next week's portfolio return** — a dollar-neutral cross-sectional book
rebalanced every Friday close. SIGNAL's job is narrow and precisely bounded:

> Given the point-in-time daily panel of closes and funding rates for the crypto perp universe as
> of rebalance time `t`, produce a cross-sectional RANK per eligible symbol — a real number in
> `[0, 1]`, monotone in "how much of this week's dollar-neutral long/short book this name should
> be on the long side of" — from (a) trailing-return momentum over a validated lookback BAND, and
> (b) trailing funding carry, combined by a small, fixed, non-fitted weight.

SIGNAL is a **provider, not a decider** (COST.md's P2 pattern, kept because it is still the right
firewall): it emits ranks, never target weights, never trade sizes, never a boolean trade/no-trade.
PORTFOLIO.md (not yet written) turns ranks into dollar-neutral quantile weights; SIGNAL never
computes a weight itself. The implementation already exists and this spec cross-references it
throughout — this is a "here is why the code is shaped this way, and here is the math it must
keep satisfying" document, not a green-field design.

Scope: crypto only. The cross-sectional momentum edge is **crypto-specific** — tested and found
NOT to extend to equities or commodities sleeves (`docs/EDGE.md` §d: "This is a crypto
cross-sectional dispersion premium, full stop"). Binance USDT-M perpetual futures, daily bars
aggregated from a 1-minute lake, weekly (Friday-close) rebalance, point-in-time universe of
~50 liquid names (UNIVERSE.md / `research/xsec/universe.py`).

---

## 0. Governing principles (read before anything else)

**P1 — SIGNAL emits a rank, never a weight, never an action.** The public contract is
`momentum_signal`, `carry_signal`, `combined_signal` (`research/xsec/signals.py`) — each returns
`np.ndarray` of shape `(n_symbols,)`, values in `[0, 1]` or `NaN`. Turning a rank into a
dollar-neutral quantile weight (top-quantile long, bottom-quantile short, inverse-vol scaled) is
PORTFOLIO's job (`research/xsec/portfolio.py`), not SIGNAL's. A rank vector that is monotone in
conviction is a complete and sufficient interface; anything more (a target notional, a leverage
multiplier) is scope creep across the firewall and is the bug.

**P2 — Every signal value at rebalance time `t` is measurable with respect to `F_t`, the
information set of the point-in-time panel up to and including day `t`'s close.** No signal reads
a panel row with index `> d` (the rebalance day's row index). This is stronger than "no lookahead
in wall-clock time" — it is a precise causality statement over the discrete filtration
`{F_d}_{d=0}^{n_days-1}` where `F_d = sigma(close[0..d], funding[0..d], listed[0..d])`. Every
signal function's domain is `F_d`-measurable by construction (Section 2, Invariant NLA).

**P3 — Cross-sectional standardization is a rank transform computed WITHIN the point-in-time
cross-section `U_t`, never a time-series z-score over full-sample statistics.** A raw trailing
return is compared only to the OTHER names eligible at the SAME rebalance, never to its own
historical distribution. `U_t` is defined formally in Section 2.3 (a set-builder anchor to
UNIVERSE.md §1.1's `F_t`-measurable eligibility indicator, intersected with SIGNAL's own
per-signal windowing) — every use of the symbol in this file refers to that one definition. This
is Invariant RANK (Section 3) and is the single most important correctness property distinguishing
a valid point-in-time cross-sectional signal from a lookahead-contaminated one.

**P4 — Single-lookback selection is measurably overfit; the deployable signal is a lookback-BAND
ensemble, not a fitted point.** CPCV over the lookback family shows PBO = 0.31–0.47 across
partitions (`docs/EDGE.md` §c: "PBO ≈ 0.36 (moderate)"; the wider CPCV run cited in this spec's
evidence digest gives the range 0.31–0.47 across partitions) and the best-in-sample lookback was
`L=12` on 21 of 28 CPCV splits, not the headline `L=14`. A signal that ships a single fitted `L`
is shipping the winner of a search whose own validation says the winner is partition-unstable.
Section 4 derives the ensemble-vs-point tradeoff formally and pins the deployable band to
`L ∈ {12, 14, 16}` trading days.

**Cross-pillar disclosure.** This `m=3` set `{12, 14, 16}` is the single canonical lookback
ensemble used consistently across all five pillars that reference it (SIGNAL, UNIVERSE,
PORTFOLIO, EXECUTION, SYSTEM). An earlier drafting pass introduced a divergent, ungrounded
`m=5` variant `{12, 13, 14, 15, 16}` in PORTFOLIO.md and SYSTEM.md, silently interpolating two
lookbacks (`L=13`, `L=15`) with no backtest history anywhere in this investigation. That
divergence has since been corrected in both files; `{12, 14, 16}` is now the only ensemble
deployed anywhere in the repo.

**P5 — Funding carry is signed by the perpetual-futures no-arbitrage mechanism, not by
convention.** A perp's funding payment exists to hold its price near the spot index; a structurally
positive-funding name is one the market is persistently long, so its longs pay its shorts. The
carry RANK is therefore the NEGATIVE of trailing mean funding (Section 5) — this is a
mechanism-derived sign, not an arbitrary choice, and it is unit-tested (Invariant CARRY-SIGN).

**P6 — Carry is a small, fixed-weight satellite overlay on the momentum core, never a fitted
blend.** Funding carry alone has honest Sharpe well below the DSR gate (DSR ≈ 0.10–0.45 across
overlay variants; `docs/EDGE.md` "not deployable standalone") and a structural ~70–75% maximum
drawdown driven by a regime-fragile short-the-squeeze exposure. That drawdown was TESTED, not
merely observed: a BTC-beta hedge lifts standalone carry Sharpe only from 0.48 to 0.57 (maxDD
74.9%→69.9%), and a causal drawdown-stop caps maxDD at ~24% only by collapsing Sharpe to 0.10 and
income to ~0.6%/yr (`research/xsec/overlay.py`; Section 6.1) — no overlay tried converts carry
into a standalone-deployable book, which is why it is a satellite, not a sizing input. Section 6
formalizes, via the Sharpe-additivity identity for near-orthogonal signals plus this overlay
evidence, why a combination weight large enough to matter for the combined Sharpe is not justified
by carry's weak and fragile standalone profile, and why the fixed weight must be derived
analytically (never fitted by regressing forward returns on candidate signals — that IS the
overfitting risk PBO already flagged, restated as Invariant NO-FIT below).

**P7 — No ML, no fitted/learned combination weights, no expensive search.** Every estimator in
this file is closed-form: rank statistics, trailing means, equal-weight or IC-weighted linear
ensembles (Grinold & Kahn 2000), and Sharpe-additivity algebra. There is no neural network, no
gradient-boosted tree, no regression of forward returns on candidate signals, no continuously
refit hidden-Markov or EM model, and no large hyperparameter grid search with retraining loops.
This mirrors the repo's `pyproject.toml` doctrine ("scipy: closed-form (NOT ML)") and COST.md's P4
("no ML / learned cost model"), extended here to signal construction and combination.

Forbidden in v1: SIGNAL emitting a weight, a trade, or a leverage multiplier (P1); any signal
computed from a row index `> d` (P2); a time-series z-score using full-sample or forward-looking
statistics in place of the point-in-time cross-sectional rank (P3); shipping a single fitted
lookback `L` as the deployable signal (P4); a fixed `funding_band_high`-style sign convention for
carry not derived from the funding mechanism (P5); a fitted/regressed combination weight between
momentum and carry (P6, P7); any neural network, GBDT, or large hyperparameter search anywhere in
signal construction (P7); per-symbol tuning of the RED ensemble weights or lookback set on
realized P&L (Section 7); an IC-weighted lookback-ensemble blend deployed WITHOUT a pre-registered,
frozen `IC_L` estimate and a fresh CPCV/PBO re-validation (Section 4.3's resolution of the
IC-weighting-vs-NO-FIT tension) — equal-weight is the only variant this spec deploys.

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its first point of use in
prose; this table is a lookup, not a substitute for those definitions). Rebalance/day index `d`
runs over integer panel rows; `t` (UNIVERSE's wall-clock timestamp) and `d` coincide at a Friday
rebalance (Section 2.3).

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `d` | Integer panel-row day index, `d = 0, …, n_days−1` | data | 2.1 |
| `F_d` | Filtration: sigma-algebra of every panel cell with row index `≤ d` | — | 2.1 |
| `close[d,i]`, `funding[d,i]`, `listed[d,i]` | `Panel` arrays: day-`d` close, aggregated daily funding, point-in-time listing flag for symbol `i` | data | 1, 2.1 |
| `L, k` | Momentum lookback (trading days) and skip (days) | RED | 2.2, 4.3 |
| `c` | Carry lookback (days), default `7` | RED | 2.2, 5.2 |
| `U_t` | UNIVERSE's `F_t`-measurable eligibility set (§1.1 of UNIVERSE.md) | UNIVERSE-owned | 2.3 |
| `U_d^{mom}(L,k)`, `U_d^{carry}(c)`, `U_d^{comb}` | `U_t` intersected with SIGNAL's own per-signal windowing (momentum / carry / both) | derived | 2.3 |
| `n` | `\|U_d^{(\cdot)}\|` — size of whichever eligible set is operative for the signal being ranked | derived | 2.3, 3.1 |
| `x ∈ R^n`, `x_i` | Raw score vector for the eligible cross-section; symbol `i`'s raw score | data | 3.1 |
| `r_i` | `i`'s 1-based rank of `x_i` among the `n` finite values (average-rank tie convention) | derived | 3.1 |
| `rank_i ≡ (r_i − 0.5)/n` | Point-in-time cross-sectional percentile rank, `∈ (0,1)` | derived | 3.1 |
| `Φ^{-1}` | Inverse standard normal CDF (Gaussian rank / normal-scores transform, named for completeness, not implemented) | — | 3.1 |
| `z_{i,t}`, `mu_hat_i`, `sigma_hat_i` | Time-series z-score and its own-history mean/vol estimators (the REJECTED alternative to `rank_i`) | — | 3.2 |
| `Rank(x, U_t)` | Percentile rank of `x` within cross-section `U_t` (formal notation for Section 3.2(b)'s ordering argument) | derived | 3.2 |
| `L_1,…,L_m` | The validated lookback family, `m=3`, `{12,14,16}` | RED | 4.2–4.3 |
| `SR_L`, `SR_hat_L` | True (population) and estimated in-sample Sharpe of lookback `L`'s weekly return series | data (measured) | 4.2 |
| `eps_L` | Mean-zero sampling-noise term, `SR_hat_L = SR_L + eps_L`, correlated across nearby `L` | — | 4.2 |
| `L*` | argmax-`L` (point-selected lookback) | derived | 4.2 |
| `v_L`, `v_s` | Lookback-ensemble weight (`=1/3` each, RED) / general ensemble component weight | RED | 4.2–4.4 |
| `r^{(L)}_t`, `r^{ens}_t` | Weekly net-return series of lookback `L`'s book; the ensemble's weighted-average return series | data, derived | 4.2 |
| `PBO` | Probability of backtest overfitting (López de Prado et al. logit-of-relative-rank estimator) | measured | 4.1 |
| `w_bar` | Relative rank of the best-in-sample config's OOS Sharpe within the family's OOS distribution (PBO's argument) | derived | 4.1 |
| `frac_pos`, `p05` | Fraction of CPCV paths with positive OOS Sharpe; 5th-percentile path Sharpe (SYSTEM-owned statistics, cited here) | SYSTEM-owned | 4.1 |
| `rank^{(1)},…,rank^{(m)}`, `avg_i`, `ensemble_rank_i` | Component rank vectors; their weighted average; the final re-ranked ensemble output | derived | 4.4 |
| `mean_funding_i,d` | Trailing mean daily funding rate over lookback `c` ending at `d`, for symbol `i` | derived | 5.2 |
| `carry_score_i,d ≡ -mean_funding_i,d` | Sign-negated carry score (Invariant CARRY-SIGN) | derived | 5.2 |
| `w`, `cash_flow(w,f)` | Signed position size (positive=long); funding cash flow at rate `f`, `= -w·f` | — | 5.2 |
| `r^{mom}_t`, `r^{carry}_t` | Weekly net-return series of the momentum-only and carry-only books (dollar-neutral, same universe, same cost model) | data | 6.1 |
| `w`, `1-w` | Risk-budget fraction allocated to the momentum book / carry book in the book-level blend | derived | 6.1 |
| `rho ≡ Corr(r^{mom}, r^{carry})` | Momentum/carry return correlation — UNMEASURED in this repo; treated as `≈0` on structural (not numeric) grounds, Section 6.1's sensitivity sweep bounds its effect | UNMEASURED | 6.1 |
| `IR_mom, IR_carry` | Standalone Sharpe/information ratios of the momentum-only and carry-only books | measured (ranged) | 6.1 |
| `w*` | Variance-minimizing (Sharpe-additivity-optimal) momentum weight, `= IR_mom^2/(IR_mom^2+IR_carry^2)` | derived | 6.1 |
| `w_carry` | Deployed core-signal carry weight — fixed at `0` (Invariant NO-FIT, Section 6.2) | RED | 6.1–6.2 |
| `DSR` | Deflated Sharpe Ratio (SYSTEM-owned gate; applied here to carry's standalone series) | SYSTEM-owned | P6, 6.1 |

---

## 1. What SIGNAL builds ON TOP OF (and what it must NOT rebuild)

SIGNAL builds on the point-in-time daily panel already constructed by `research/xsec/panel.py`
(`Panel.close`, `Panel.funding`, `Panel.listed`) and the universe/friction metadata in
`research/xsec/universe.py` / `research/universe.py`. SIGNAL does **not** rebuild:

| Already built (do NOT rebuild)                                      | SIGNAL consumes it as                          |
|-----------------------------------------------------------------------|-------------------------------------------------|
| `Panel` point-in-time close/funding/listed arrays (`panel.py`)        | the `F_t`-measurable raw data every signal reads |
| Point-in-time listing mask (`Panel.listed`, no survivorship backfill) | the eligibility gate (Section 2)                 |
| Daily funding aggregation (sum of 8h stamps → daily rate, `panel.py`) | the raw funding series `carry_signal` averages   |
| Point-in-time dollar-volume panel (`build_dollar_volume`, `panel.py`) | (used by PORTFOLIO/UNIVERSE, not by SIGNAL itself) |
| Sleeve/friction/regime metadata (`research/xsec/universe.py`)         | universe membership only (crypto sleeve)         |
| `convexity.system.validate.{deflated,cpcv,walkforward}` (SYSTEM pillar) | the DSR/CPCV machinery `research/xsec/validate.py` and `research/xsec/cpcv.py` re-use for signal-family validation |

Implemented in `research/xsec/signals.py` (the module this spec documents),
`research/xsec/panel.py` (upstream data), `research/xsec/cpcv.py` and `research/xsec/validate.py`
(the validation this spec's constants are drawn from).

---

## 2. Point-in-time causality: the filtration and the eligibility gate

### 2.1 The filtration `F_t`

Index calendar days by integer row `d = 0, 1, …, n_days − 1` (ascending UTC days,
`Panel.dates[d]`). Define the filtration

```
F_d = sigma( close[0..d, :], funding[0..d, :], listed[0..d, :] )
```

i.e. the sigma-algebra generated by every panel cell with row index `≤ d`. A rebalance at day `d`
may use any function measurable with respect to `F_d`; it may **never** use `close[d+1:]`,
`funding[d+1:]`, or `listed[d+1:]`. This is the formal statement of "no lookahead" for this panel.

`Panel.close[d]` is defined (`panel.py` docstring) as the last 1-minute close of UTC calendar day
`d` — i.e. information genuinely known only at day `d`'s close (~23:59 UTC), not an intra-day
value smeared forward. `Panel.funding[d]` is the SUM of day `d`'s three 8-hourly funding stamps,
also fully realized and knowable by day `d`'s close. Both are honest `F_d`-measurable quantities
by construction of the panel layer — SIGNAL inherits this and must not violate it by, e.g.,
peeking at `d+1`'s close to "confirm" a signal.

### 2.2 Eligibility

A symbol `i` is **momentum-eligible** at rebalance day `d` with lookback `L` and skip `k` iff:

```
listed[d - k, i]  AND  listed[d - k - L, i]  AND  (d - k - L) >= 0
```

i.e. both endpoints of the trailing-return window are inside the symbol's live point-in-time span
(`research/xsec/signals.py::momentum_signal`, using `Panel.listed`). Because `Panel.listed` is
constructed from each symbol's OWN first/last live day — never from the cross-sectional union
(`panel.py::build_panel` docstring: "NEVER from the union … no survivorship backfill, no delist
hold-through") — a symbol that has not yet listed, or has delisted, is correctly excluded without
any special-casing in `signals.py` itself.

A symbol is **carry-eligible** at `d` with carry lookback `c` (default 7 days) iff
`listed[d - c + 1 : d + 1, i]` is True on every day of the trailing window (full window required,
`carry_signal`'s `live_all` mask) — a partial funding window is treated as ineligible rather than
averaged over fewer days, because a short/partial funding history is disproportionately noisy
relative to a short/partial price history (funding realizations are ~three discrete draws/day vs.
a continuous price path).

A symbol is **combined-eligible** iff it is eligible for BOTH sub-signals (`combined_signal`'s
`both = isfinite(mom) & isfinite(car)` — an intersection, not a union, so the combined score never
silently substitutes one signal's information for the other's on a name that lacks it).

### 2.3 Formal definition of `U_t` — the point-in-time eligible cross-section

Every reference to `U_t` in this file (P3, Section 3) means the SAME object UNIVERSE.md defines
formally (`docs/specs/UNIVERSE.md` §1.1, the `F_t`-measurable eligibility indicator process
`1{i ∈ U_t}`) — SIGNAL does not re-derive eligibility, it composes UNIVERSE's gate with its own
per-signal windowing on top. Stated once, precisely, so the symbol used throughout Sections 3–4
has a set-builder definition and not only a prose gloss:

```
U_t  :=  { i : 1{i ∈ U_t^univ} = 1 }              (UNIVERSE's F_t-measurable gate; §1.1 of UNIVERSE.md)
```

For a given SIGNAL call, the operative eligible set is the intersection of UNIVERSE's gate with
that signal's OWN windowing requirement (Section 2.2 above), evaluated at the same rebalance day
`d` (`t` and `d` coincide at a Friday rebalance — UNIVERSE indexes by wall-clock timestamp `t`,
SIGNAL indexes by the underlying panel's integer day row `d`; `Panel.dates[d] = t` at a rebalance):

```
U_d^{mom}(L, k)  :=  U_t  ∩  { i : listed[d-k, i] ∧ listed[d-k-L, i] ∧ (d-k-L) ≥ 0 }     (Section 2.2, momentum)
U_d^{carry}(c)   :=  U_t  ∩  { i : listed[d-c+1 : d+1, i] all True }                      (Section 2.2, carry)
U_d^{comb}       :=  U_d^{mom}(L,k)  ∩  U_d^{carry}(c)                                    (combined-eligible, above)
```

`n` in Section 3.1's rank transform is `|U_d^{(\cdot)}|` for whichever of these three sets is
operative for the signal being ranked. Every occurrence of `U_t` in P3 and Section 3.2 refers to
this same `F_t`-measurable set — the reason `Rank(x_i, U_t) > Rank(x_j, U_t) <=> x_i > x_j` (Section
3.2(b)) is a well-defined statement is exactly because `U_t` is a precise, `F_t`-measurable set
membership test, not an informal "the names we happen to have" gloss. SIGNAL never constructs
`U_t` itself (that is UNIVERSE's job — P1's firewall applies here too: SIGNAL consumes `Panel.listed`,
which already encodes UNIVERSE's eligibility decision upstream at the panel-construction layer,
per Section 1's table) — this section exists only to give the symbol this file uses constantly a
single, unambiguous, set-builder anchor, matching the rigor UNIVERSE.md gives it and `F_d` already
has in Section 2.1.

> **Invariant NLA (No-LookAhead).** Every signal function's return value at day `d`, for any
> `L, k, c`, is a deterministic function of `{close[j], funding[j], listed[j] : j ≤ d}` only.
> Test: replacing every panel cell with row index `> d` with adversarial/garbage data (NaN, ±∞,
> shuffled) must leave `momentum_signal(panel, d, …)`, `carry_signal(panel, d, …)`, and
> `combined_signal(panel, d, …)` bit-for-bit unchanged. This is the same discipline as COST.md's
> Invariant W (causal volume forecast) and SYSTEM.md's firewalled validation labels, applied here
> to the signal layer.

---

## 3. Cross-sectional standardization — the rank transform, and why it is safer than a z-score

### 3.1 The transform

For a raw score vector `x ∈ R^n` (n = number of eligible symbols at `d`; ineligible symbols are
NaN and excluded from `n`), the point-in-time cross-sectional rank is

```
rank_i = ( r_i − 0.5 ) / n
```

where `r_i ∈ {1, …, n}` is `i`'s 1-based rank of `x_i` among the `n` finite values (ties broken by
the average-rank convention: a block of `m` tied values each receive the mean of the `m`
consecutive integer ranks they would occupy). This is exactly `_pct_rank` in
`research/xsec/signals.py`. The map is affine in the rank, not in `x`, so it is a nonparametric
(distribution-free) transform: it depends only on the ORDER STATISTICS of `x_1, …, x_n`, never on
their scale or shape.

Properties, directly from the construction:
- `rank_i ∈ (0, 1)` strictly, for every finite `n ≥ 1` (a lone eligible symbol maps to `0.5`,
  the neutral midpoint — `_pct_rank`'s `m == 1` branch).
- Symmetric: the map `x -> -x` produces `rank -> 1 - rank` (reversing the order reverses the
  ranks), so long/short symmetry is exact regardless of the underlying distribution of `x`.
- Invariant to any strictly increasing transform of `x` (rank position is preserved under
  monotone reparametrization) — a property no z-score has.

The Gaussian rank / normal-scores transform `Phi^{-1}((rank_i − 0.5)/n)` is an admissible
alternative standardization, named here for completeness. The two are related by
`normal_score_i = Phi^{-1}(rank_i)` — the normal-scores transform is the percentile rank pushed
through the inverse Gaussian CDF, which spreads the tails (a name ranked 99th percentile among 50
gets a much larger magnitude under the normal-scores map than a linear percentile). Both are valid
UNBIASED point-in-time cross-sectional standardizations under Invariant RANK below; `signals.py`
implements the plain percentile form because PORTFOLIO's quantile-weight construction
(`research/xsec/portfolio.py::quantile_weights`) only needs the ORDER of names to select the
top/bottom quantile, not their standardized magnitude — the percentile rank is the minimal
sufficient statistic for that downstream consumer, so introducing the Gaussian tail-spreading is a
free parameter this spec does not need and does not adopt. A future PORTFOLIO variant that used
rank MAGNITUDE (not just order) to set within-leg weights would be the point at which the
normal-scores transform becomes informative rather than cosmetic; that is out of scope here.

### 3.2 Why cross-sectional rank, not a time-series z-score (the safety argument, formal)

A time-series z-score of the same raw momentum value would be

```
z_i,t = ( x_i,t − mu_hat_i ) / sigma_hat_i,    mu_hat_i, sigma_hat_i estimated from i's OWN history
```

This is a DIFFERENT and materially less safe construction for two reasons, both provable from the
definitions:

**(a) Regime-conditioning leaks information about the FUTURE volatility regime unless `mu_hat`,
`sigma_hat` are strictly trailing, and even then it compares `i` to itself across time, not to its
peers at the same time.** If `sigma_hat_i` is estimated using centered or full-sample statistics
(common practice for "in-sample" reports), then `sigma_hat_i` is a function of `x_i,t'` for
`t' > t` — a direct violation of `F_t`-measurability (Section 2.1). Even a strictly-trailing
`sigma_hat_i` (using only `t' ≤ t`) remains regime-dependent: a coin's trailing return volatility
during a quiet 2023 stretch understates its z-score's true dispersion once a 2022-style regime
returns, so the SAME raw return produces a much larger z-score in a newly-volatile regime purely
from a stale `sigma_hat_i`, with no genuine change in relative momentum. The cross-sectional rank
has no such failure mode: `rank_i,t` is computed fresh at every `t` from ONLY the peer set
`U_t` alive at `t`, so it automatically re-normalizes to whatever cross-sectional dispersion
exists AT `t` — there is no stale trailing statistic to go stale.

**(b) The z-score's benchmark population is wrong for a dollar-neutral relative-value book.** The
economic bet here is "name `i` outperforms name `j`, both drawn from `U_t`," not "name `i`
outperforms its own history." A z-score answers the latter question; PORTFOLIO's quantile
construction (long the top quantile of `U_t`, short the bottom quantile of `U_t`) needs the
former. Formally, if `Rank(x, U_t)` denotes the percentile rank of `x` within cross-section
`U_t`, then for any two eligible names `i, j`:

```
Rank(x_i, U_t) > Rank(x_j, U_t)   <=>   x_i > x_j   (within U_t, at time t)
```

which is exactly the ordering PORTFOLIO's top/bottom-quantile split needs. A z-score comparison
`z_i,t > z_j,t` is NOT equivalent to `x_i > x_j` whenever `sigma_hat_i != sigma_hat_j` — it
answers "is `i` unusually strong FOR ITSELF" not "is `i` stronger THAN `j` RIGHT NOW," which is a
different economic question from the one the dollar-neutral cross-sectional book is pricing.

> **Invariant RANK — cross-sectional standardization uses only the point-in-time eligible set.**
> `rank_i,d` for any signal is a function of `{x_j,d : j ∈ U_d^{(\cdot)}}` only (Section 2.3's
> formal `U_t`/`U_d` definition for the signal in question) — never of `x_i`'s
> own history beyond what already entered the raw score `x_i,d` itself, and never of any `x_j,d'`
> for `d' > d`. Test: adding or removing an INELIGIBLE symbol (one with `listed[d]=False`) from the
> panel must leave every eligible symbol's rank unchanged (the rank denominator `n` and the order
> statistics only see the eligible set); adding a genuinely eligible new symbol must change other
> ranks only through the order statistic (never through a side-channel statistic like a shared
> rolling mean/variance).

---

## 4. The lookback-band ensemble — formal justification (the key design decision)

### 4.1 The empirical finding that forces this design

CPCV over the momentum lookback family (`research/xsec/cpcv.py::compute_pbo`, the López de Prado
logit-of-relative-rank estimator) gives **PBO = 0.31–0.47** across partitions of the ~259-week
crypto history, with the best-in-sample lookback landing on `L = 12` in **21 of 28** CPCV splits —
not the headline `L = 14` that the light-cost single-path backtest peaks at. PBO is defined
(Bailey, Borwein, López de Prado & Zhu, 2014 — the same citation as this file's References entry
below; an earlier draft of this section cited a differently-ordered/dated author list for the same
estimator, which was a citation-drift error across drafts, corrected here to use one canonical
name/year everywhere the citation appears in this file, matching the exemplars' one-name-one-
citation discipline) as the probability that the configuration selected as best IN-SAMPLE ranks
below the OUT-OF-SAMPLE median among the family:

```
PBO = P( logit(w_bar) <= 0 ),   w_bar = relative rank of the best-in-sample config's OOS Sharpe
                                          within the family's OOS Sharpe distribution
```

`compute_pbo` computes exactly this: for each of the `C(n_groups, k_test)` CPCV splits it picks
`argmax` in-sample Sharpe over the lookback family on the train complement, looks up that SAME
config's relative rank among the family on the embargoed test blocks, and takes the fraction of
splits whose logit is `<= 0`. PBO = 0.31–0.47 means roughly a third to nearly half of all splits
see the in-sample "winner" perform at or below the family median out-of-sample — a moderate but
real overfitting signal, never below `0.2` clean-pass in any partition tested. This is direct,
quantitative evidence that **which single `L` looks best is itself noisy** across the resampled
history, even though the FAMILY as a whole (momentum at lookbacks in the 12–16 day neighborhood)
is genuinely OOS-positive (Section 4.2 below; this is not "momentum is fake," it is "the exact
peak of the lookback curve is fake").

**PBO is the overfitting-of-the-lookback-DIMENSION statistic; it is not the only CPCV product,
and it is not the one this file should lead with when the question is "does the strategy survive
CPCV at all."** The SAME `N=8`-group / `k_test=2` / 28-path / 1-week-embargo split family
(`research/xsec/cpcv.py::run_cpcv`, Section 7's provenance table) also produces a PATH-SHARPE
distribution for the selected/deployed config (`PathSharpeSummary`, `research/xsec/cpcv.py`):
mean OOS path Sharpe **≈0.91** versus the naive full-sample Sharpe **≈0.98** — a haircut of only
**≈0.07**, i.e. **no OOS collapse** (`docs/EDGE.md` §c; the same number every sibling pillar spec
that touches CPCV cites verbatim — `docs/specs/PORTFOLIO.md` §"CPCV mean OOS path Sharpe" table,
`docs/specs/UNIVERSE.md` §CPCV, `docs/specs/EXECUTION.md`, `docs/specs/REGIME.md`,
`docs/specs/SYSTEM.md` §"Result" and `docs/specs/LEVERAGE.md`). This is the direct, quantitative
contrast to the directional-breakout book's CPCV verdict, which flagged `luck_artifact=True` (OOS
paths straddling zero, i.e. a haircut large enough to erase the mean) — SIGNAL's lookback family
shows the opposite pattern: a small, honest haircut with the OOS mean still comfortably positive.

Within that path distribution, `frac_pos` (the fraction of the 28 CPCV paths with positive OOS
Sharpe) is **0.85–1.00** and is the statistic the evidence digest names as ROBUST to the `N`/`k`
partition choice, in contrast to the 5th-percentile path Sharpe (`p05`), which is
**PARTITION-FRAGILE** — it swings from **+0.42 to −0.59** depending on the `N`/`k` grouping chosen
(`docs/EDGE.md` §c: "`frac_pos`… is the robust signal; the p05 tail is partition-fragile — trust
breadth of positivity, not the worst-partition tail"; restated identically as Invariant R-stat in
`docs/specs/PORTFOLIO.md`). **`frac_pos`, not PBO and not p05, is the statistic a SIGNAL-layer
promotion decision should weight most heavily when the question is "is the deployed lookback
family's OOS behavior robust,"** because PBO answers a narrower question (is the ARGMAX-`L`
selection biased — yes, moderately) while `frac_pos` answers the broader question this file's
Invariant BAND is actually trying to protect (does the FAMILY, ensembled, produce OOS-positive
weeks across the large majority of resampled partitions — yes, 85–100% of them). A future
promotion report that cites only PBO without `frac_pos` and the mean-path-Sharpe haircut is
citing a necessary but not sufficient piece of the CPCV evidence.

### 4.2 The formal ensemble-vs-point-selection tradeoff

Let `L ∈ {L_1, …, L_m}` be the validated lookback family (here `m = 3`, `{12, 14, 16}`), each
producing a weekly net-return series `r^{(L)}_t` with true (population) Sharpe `SR_L` and
sampling noise such that the ESTIMATED in-sample Sharpe `SR_hat_L` satisfies
`SR_hat_L = SR_L + eps_L`, `eps_L` a mean-zero, finite-variance selection-noise term correlated
across nearby `L` (adjacent lookbacks overlap heavily in the trailing window they read, so `eps_L`
and `eps_{L+2}` are strongly positively correlated — this is WHY the family is a "band," not `m`
independent draws).

**Point selection** (deploy the single argmax-`L`) delivers, in expectation over repeated
finite-sample draws, the max-order-statistic bias:

```
E[ SR_hat_{L*} ]  =  E[ max_L (SR_L + eps_L) ]  >=  max_L SR_L      (Jensen, strict > if Var[eps]>0)
```

i.e. the in-sample-selected peak systematically OVER-estimates the true Sharpe of whichever `L` it
happens to pick, by an amount that grows with the number of candidates searched and the variance
of the selection noise — this is the exact quantity DSR's `SR*_0` term (Section 7.1 of SYSTEM.md,
inherited machinery) deflates for at the PORTFOLIO/validation layer. At the SIGNAL layer, the
PBO estimator is measuring this same phenomenon directly on the lookback dimension alone: it is
telling us the argmax-`L` selection bias is large enough that the winner flips across resampled
partitions in the majority of splits (21/28 favor `L=12` over the headline `L=14`), which is
precisely the finite-sample symptom the inequality above predicts when `Var[eps_L]` is
non-negligible relative to the spread `max_L SR_L - median_L SR_L`.

**Ensemble selection** (deploy the equal-weight or IC-weighted average over the band) instead
targets

```
r^{ens}_t = sum_{L in family} v_L * r^{(L)}_t,     sum_L v_L = 1,  v_L >= 0
```

Because the `eps_L` are positively correlated but not perfectly so (adjacent lookbacks share most
but not all of their trailing window), the ensemble's Sharpe estimator has STRICTLY LOWER
selection-driven variance than the point estimator's max, at the cost of giving up the (biased,
partly illusory) peak:

```
Var[ SR_hat_{ens} ]  <  Var[ max_L SR_hat_L ]        whenever  Corr(eps_{L_i}, eps_{L_j}) < 1  (∃ i≠j)
E[ SR_hat_{ens} ]     =  sum_L v_L * SR_L             (a WEIGHTED AVERAGE of true Sharpes, not a max)
```

The ensemble's expectation is an honest average of the family's TRUE Sharpes (no max-order-statistic
inflation, because there is no `argmax` in the construction — every weight `v_L` is fixed BEFORE
looking at any realized `SR_hat_L`, satisfying P7's no-fitted-weight rule). The ensemble therefore
trades a small amount of point-estimate peak Sharpe (`max_L SR_L - sum_L v_L SR_L`, small whenever
the family members have similar true Sharpe, which the near-zero degradation from `L=14` to
`L=12/16` in the backtested family indicates) for a LARGE reduction in the selection-driven
variance that PBO = 0.31–0.47 is flagging as the dominant risk. This is the precise formal
statement, derived above from the max-order-statistic/variance-reduction inequalities, of the
design principle P4 states informally: **ensembling trades a small amount of peak Sharpe for a
large reduction in selection-driven overfitting variance.**

### 4.3 The deployable ensemble

```
DEPLOY:  L ∈ {12, 14, 16} trading days,  skip k = 0  (the repo's validated default; see Invariant SKIP below)
WEIGHT:  v_L = 1/3 each                            (equal-weight; RED, Section 7; the ONLY deployed weight, see below)
```

**Why the IC-weighted alternative named earlier in some drafts of this design is NOT offered as a
live "OR" here, and why that is not merely a stylistic choice.** An IC-weighted variant
(`v_L PROPORTIONAL to IC_L`, `IC_L` = the rank information coefficient of lookback `L`'s signal
against realized forward return, estimated over this repo's own ~5-year sample) is, on inspection,
**the identical failure mode Invariant NO-FIT (Section 6.1) forbids for the momentum/carry blend,
one dimension over.** Both are "estimate a linear combination weight by correlating each candidate
signal against realized forward returns on the same limited sample used to validate the family."
Section 6.1 forbids exactly this for `w_carry` ("never by an in-sample regression of forward
returns on candidate signals… fitting a 2-signal blend weight on the same limited sample is the
identical failure mode one dimension over") — there is no principled reason a 3-way IC-weighted
lookback blend is exempt from that argument while a 2-way IC/regression-weighted momentum/carry
blend is the paradigm case of what NO-FIT forbids. Grinold & Kahn (2000, ch. 6) is the correct,
precisely-scoped citation for the IC-weighting FORMULA used here (this is its only citation role in
this file — Section 6.1's book-level blend cites Sharpe (1994) / Markowitz two-asset tangency
theory instead, so the two sections no longer share one citation across two distinct applications),
but ch. 6 does not license running the IC-weighting formula on a sample this short: IC-weighting is
derived under an implicit assumption of many independent, stable IC
estimates (a large cross-section of stock-selection signals scored over a long history), which is
a different regime from estimating 3 highly-correlated `IC_L`'s (Section 4.2's `eps_L`
correlation-across-`L` argument applies here just as much as to raw Sharpe) off ~259 weekly
observations.

**Resolution: equal-weight is the ONLY weight this spec deploys; IC-weighting is demoted to a
research/diagnostic tool, never a promotable default, under the same discipline Section 6.2 gives
`combined_signal`.** If a future report wants to explore IC-weighting the lookback band, it must
clear the SAME bar Invariant NO-FIT sets for `w_carry`: the `IC_L` estimate must be PRE-REGISTERED
and FROZEN from a validation run that PRECEDES its use as a deployed weight (never computed
in-sample against the same period being scored), and the resulting weighted-ensemble config must
independently re-clear CPCV/PBO before promotion (it re-enters the BLUE bucket of Section 7, not
RED-by-assertion). Absent that discipline, IC-weighting the band is exactly the kind of small
estimation-noise-driven search over a 3-element family that Section 4.1's PBO evidence already
shows is overfit-prone — equal-weight is the only variant that is fixed BEFORE any `SR_hat_L`/`IC_L`
is observed, satisfying P7's no-fitted-weight rule without qualification.

`combined_signal`/`momentum_signal` in `research/xsec/signals.py` currently expose a single
`lookback: int` parameter (the point-selection interface used by the backtest/validation harness
to SCORE the family, which is exactly how `research/xsec/cpcv.py` produces the PBO evidence this
section cites). The DEPLOYABLE ensemble signal is a thin wrapper — not yet a named function in
`signals.py` — defined as:

```python
def ensemble_momentum_signal(panel: Panel, d: int, *, lookbacks: tuple[int, ...] = (12, 14, 16),
                              skip: int = 0, weights: tuple[float, ...] | None = None) -> np.ndarray:
    """Equal-weight (or IC-weighted) rank-average of momentum_signal across the lookback BAND.
    Re-ranks the weighted average cross-sectionally (same intersection-eligibility discipline as
    combined_signal: a symbol needs to be eligible for EVERY lookback in the band, else it is
    excluded rather than scored on a partial band)."""
```

This mirrors `combined_signal`'s existing pattern exactly (rank each sub-signal, average, re-rank)
— Section 4.4 states this as the formal general operator.

### 4.4 The general rank-ensemble operator (ties Section 3's rank transform to Section 4's ensemble)

Given `m` component rank vectors `rank^{(1)}, …, rank^{(m)}` (each already a point-in-time
cross-sectional rank per Section 3, one per lookback), the ensemble score and FINAL rank are:

```
avg_i = sum_{s=1}^{m} v_s * rank^{(s)}_i      (defined only where i is finite in EVERY component s)
ensemble_rank_i = _pct_rank(avg)_i             (re-rank the averaged score cross-sectionally)
```

This is literally the `combined_signal` construction in `research/xsec/signals.py` generalized
from `m = 2` (momentum, carry) sub-signals to `m = 3` (lookbacks 12/14/16), with `v_s = 1/2` each
in the shipped `combined_signal` and `v_s = 1/3` each for the lookback-band ensemble of this
section. The re-rank step matters: averaging two `[0,1]` ranks does not itself produce a uniform
`[0,1]` cross-section (the average of two independent uniforms is triangular, not uniform), so a
FINAL `_pct_rank` pass restores the flat, PORTFOLIO-consumable percentile-rank contract regardless
of `m` or the component weights.

> **Invariant BAND — the deployable signal is never a single fitted lookback.** The signal that
> feeds PORTFOLIO in any live/paper/validated-promotion path uses the ensemble operator over
> `L ∈ {12, 14, 16}` (or a RED-governed superset validated by the same CPCV procedure), never a
> lone `lookback=14` (or any other single value) selected because it scored highest in a single
> backtest pass. A single-`L` signal MAY be run for research/diagnostic purposes (exactly what
> produced the PBO evidence this section cites) but MUST NOT be the promoted/deployed
> configuration. Test: the promotion/validation report for any live config asserts
> `len(lookbacks) >= 2` and that the weights were fixed BEFORE the family's OOS Sharpes were
> computed (no `argmax`-derived weight in the deployed config).

> **Invariant SKIP — skip is a documented default, not silently zero.** `signals.py`'s
> `momentum_signal` docstring names `skip ∈ {0, 3, 7}` days as the researched grid (dodging the
> well-documented ~1-week crypto reversal effect). The repo's validated default is `skip = 0`,
> because the reversal effect at these lookbacks (12–16d, an order of magnitude longer than the
> 1-week reversal window) was not found to materially change the family's OOS Sharpe — this is a
> reported finding, not an assumption; a config that changes `skip` away from `0` re-triggers the
> family validation (CPCV/PBO) before promotion, per Invariant BAND's promotion discipline.

---

## 5. Funding carry score — the mechanism, the sign, and the formula

### 5.1 The perpetual-futures funding mechanism (why funding exists at all)

A perpetual future has no expiry, so it needs a mechanism other than convergence-to-expiry to keep
its price tethered to the spot index. Funding is that mechanism: every 8 hours, longs pay shorts
(or vice versa) a rate proportional to the PREMIUM of the perp mark price over the spot index
(COST.md §5.5b's `premium_8h = (perp_mark - index)/index`, plus a small interest-rate baseline
term). When the perp trades ABOVE spot (positive premium — the market is net long and bidding the
perp up), funding is positive and LONGS PAY SHORTS, which is a cost-of-carry force pushing new
capital to short the perp (or arbitrageurs to short perp/long spot), pulling the perp price back
toward the index. Symmetrically, a persistently negative-funding perp is one shorts are paying
longs to hold, pulling capital the other way. Funding is therefore a REALIZED, OBSERVABLE
compensator for the direction of one-sided demand pressure on the perp relative to spot — not a
forecast, not a model output, a fact recorded in the funding rate itself.

### 5.2 The carry score and why the sign is negated (formal)

Define the trailing mean daily funding rate over a lookback window `c` (default 7 days) ending at
rebalance day `d`:

```
mean_funding_i,d = (1/c) * sum_{tau=d-c+1}^{d} funding_i,tau
```

(`research/xsec/signals.py::carry_signal`, using `Panel.funding`'s daily-aggregated stamps —
Section 2.1's `F_d`-measurability holds because every `tau <= d`). The CARRY SIGNAL that ranks for
LONG conviction is the NEGATION:

```
carry_score_i,d = - mean_funding_i,d
```

**Why negate (the mechanism, stated as a derivation).** Consider a position of signed size
`w` (positive = long) held over a funding interval with realized rate `f`. The funding CASH FLOW
to the position is (COST.md §5.5, `accounting.py`'s `settle_week` docstring, both consistent):

```
cash_flow(w, f) = - w * f
```

A LONG (`w > 0`) in a POSITIVE-funding name (`f > 0`) has `cash_flow = -w*f < 0`: it PAYS. A SHORT
(`w < 0`) in the SAME positive-funding name has `cash_flow = -w*f > 0`: it RECEIVES. So to be on
the RECEIVING side of funding, a name with HIGH POSITIVE funding must be SHORTED, and a name with
LOW/NEGATIVE funding must be LONGED. PORTFOLIO's downstream convention (`portfolio.py`,
`quantile_weights`) is uniform across every signal SIGNAL emits: **high rank → long, low rank →
short.** For the carry rank to line up with that convention — i.e. for "high carry rank" to mean
"the side of this name that RECEIVES funding" — the raw ranking variable must be
`-mean_funding_i,d`, because:

```
mean_funding_i,d  HIGH (positive)   =>  want to SHORT i to receive funding  =>  need LOW rank
mean_funding_i,d  LOW (negative)    =>  want to LONG i to receive funding   =>  need HIGH rank
carry_score = -mean_funding   =>   HIGH mean_funding -> LOW carry_score -> LOW rank    (correct: short)
                                    LOW  mean_funding -> HIGH carry_score -> HIGH rank   (correct: long)
```

This is exactly `carry_signal`'s implementation (`raw[:] = -mean_funding`) and its docstring's
statement: "a perp with HIGH positive funding pays its longs, so we SHORT it… the carry SIGNAL
that we rank for LONG conviction is the NEGATED mean funding." The negation is not a convention
choice free to flip — it is forced by the definition of `cash_flow = -w*f` composed with the
"high rank = long" convention shared by every signal in this file.

> **Invariant CARRY-SIGN.** For any funding regime, a LONG position taken on the HIGH-carry-rank
> side of the book and a SHORT position on the LOW-carry-rank side both have `cash_flow >= 0` in
> expectation over the trailing window that generated the rank (mirrors COST.md's Invariant P
> sign discipline, restated at the signal layer). Test: constructing a two-name book from
> `carry_signal`'s extremes and computing `cash_flow` via `accounting.py::settle_week`'s
> `-w_i * f_acc` on the SAME trailing window the rank used must yield a non-negative combined
> cash flow; flipping the sign (ranking on `+mean_funding` instead) must fail this test on any
> input with nonzero funding dispersion.

### 5.3 Full-window eligibility (why a partial funding window is excluded, not averaged)

`carry_signal` requires `listed[d-c+1:d+1, i]` True on ALL `c` days (`live_all.all(axis=0)`), not
merely a majority. Funding is a per-8h-stamp discrete random variable with a small number of draws
even over a full week (`c=7` days × 3 stamps/day = 21 draws); a partial window (say, a coin that
listed 2 days ago) would average over as few as 6 draws, whose sample mean has a standard error
large relative to the persistent-regime signal the carry score is trying to capture. Excluding
partial windows entirely (treating them as NaN/ineligible, per Section 2.2) is the conservative
choice under uncertainty (the same P6 discipline COST.md states for its own funding forecast) —
it costs a small amount of early-life universe coverage in exchange for not ranking on a
statistically unreliable trailing mean.

---

## 6. Signal combination — why the deployable weight on carry is small/near-zero

### 6.1 The Sharpe-additivity identity for near-orthogonal signals (the derivation)

Let `r^{mom}_t` and `r^{carry}_t` be the weekly net-return series of the momentum-only and
carry-only books (both dollar-neutral, both from the SAME point-in-time universe, both net of the
SAME real cost model). Consider the combined book formed by allocating a fraction `w` of risk
budget to the momentum book and `1-w` to the carry book (a book-level blend — a stronger and more
conservative statement than blending the RANKS, since it lets us reason about realized Sharpe
directly; the rank-level `combined_signal` construction in `signals.py` is a special case with the
book-level intuition carrying over qualitatively). If `r^{mom}` and `r^{carry}` are uncorrelated
(`Corr(r^{mom}, r^{carry}) = rho ≈ 0`, the near-orthogonality condition — momentum and carry are
different information sources: trailing PRICE moves vs. trailing FUNDING-RATE levels, and the
repo's overlay analysis (`research/xsec/overlay.py`'s docstring) finds carry's own residual
exposure is to BTC-beta, not to the momentum book's cross-sectional dispersion), the combined
book's variance is

**How close to `rho ≈ 0` is this repo's actual evidence, stated numerically rather than only
qualitatively.** No single Pearson correlation between `r^{mom}_t` and `r^{carry}_t` is currently
computed and reported by `research/xsec/overlay.py` or elsewhere in this repo — that is a genuine
gap in the evidence base, flagged here rather than papered over with an invented number. What IS
measured, and what the near-orthogonality claim actually rests on, is INDIRECT but numerical:
`research/xsec/overlay.py::beta_hedge_returns` finds carry's residual market exposure is
concentrated in BTC-beta (a hedgeable, single-factor exposure — beta-hedging the raw carry book
lifts its Sharpe from 0.48 to 0.57, Section 6.2 above), which is evidence AGAINST carry's return
stream being driven by the SAME cross-sectional dispersion momentum trades on: momentum's edge is
explicitly the dispersion that survives netting OUT market beta (`docs/EDGE.md` §b: "market-wide
beta cancels; what remains is the cross-sectional dispersion premium"), whereas carry's dominant
residual risk factor IS market beta. Two return streams whose respective residual risk is
attributed to economically DIFFERENT factors (beta-driven squeeze risk vs. beta-neutral
dispersion) are a priori more likely to be weakly correlated than two streams sharing a risk
factor — but this is a STRUCTURAL argument for low `rho`, not a measured one, and this spec does
not claim otherwise. **The honest position: `rho ≈ 0` is a plausible, mechanism-supported
approximation, not a validated number** — a promotion report that wants to lean on Section 6.1's
`w*` formula with any weight materially above `w_carry = 0` should first compute
`Corr(r^{mom}_t, r^{carry}_t)` directly from the two books' realized weekly series and confirm it
is in fact small before trusting the cross-term-vanishes simplification; this spec does not need
that number because its OWN conclusion is `w_carry = 0` regardless of the precise `rho` (Section
6.2 — the overlay evidence alone is sufficient to reject a material carry weight even if `rho`
turned out to be non-negligible).

```
Var[w*r^{mom} + (1-w)*r^{carry}] = w^2 * Var[r^{mom}] + (1-w)^2 * Var[r^{carry}]
                                    + 2*w*(1-w)*rho*sigma_mom*sigma_carry
                                  ≈ w^2 * sigma_mom^2 + (1-w)^2 * sigma_carry^2      (rho ≈ 0)
```

and the combined book's mean return is `w*mu_mom + (1-w)*mu_carry`. The combined SHARPE (not
merely variance) under near-orthogonality decomposes approximately as:

```
IR_combined^2  ~=  w^2 * IR_mom^2  +  (1-w)^2 * IR_carry^2  +  cross-term(rho)
```

where `IR_mom = SR(r^{mom})`, `IR_carry = SR(r^{carry})` (Sharpe/information ratios of the
standalone books) and the cross-term vanishes as `rho -> 0`.

**Citation, corrected: this is a two-asset tangency-portfolio result, not primarily an
Information-Coefficient result, and is cited as such.** The algebra below (`d/dw` of the leading
`rho≈0` variance approximation, giving `w* = IR_1^2/(IR_1^2+IR_2^2)`) is the closed-form
variance-minimizing weight for combining two uncorrelated return streams — this is the classical
two-asset mean-variance tangency-portfolio weight in Sharpe-ratio form, i.e. Sharpe's own
generalized-Sharpe-ratio treatment of combining two return series (**Sharpe, W. F. (1994). "The
Sharpe Ratio." *Journal of Portfolio Management*, 21(1), 49–58** — References section), and is
textbook Markowitz mean-variance portfolio theory applied to `n=2` assets (e.g. Elton, Gruber,
Brown & Goetzmann, *Modern Portfolio Theory and Investment Analysis*, the two-asset tangency-weight
derivation). **Sharpe (1994) is the PRIMARY citation for this section's `w*` formula**, replacing
the earlier draft's reliance on Grinold & Kahn (2000) ch. 6 for this specific result. Grinold &
Kahn ch. 6 remains the correct, precisely-scoped citation for Section 4.3's DIFFERENT result
(IC-weighted combination of multiple stock-selection signals via their Information Coefficients) —
that citation is unchanged and stays there. The two sections no longer share one citation across
two applications: 4.3 cites Grinold & Kahn for IC-weighting; 6.1 cites Sharpe (1994) /
Markowitz two-asset tangency theory for the orthogonal-blend `w*` formula. This also resolves the
citation-double-duty concern that a shared citation would otherwise raise between the two sections,
since each now has its own precisely-scoped primary source. **The derivation of why it bounds the
usable carry weight:** differentiate the leading (`rho = 0`) approximation with respect to `w` and
set to zero to find the variance-minimizing (or, under the stated approximation, IR-maximizing)
blend:

```
d/dw [ w^2 IR_mom^2 + (1-w)^2 IR_carry^2 ] = 0
=>  2w*IR_mom^2 - 2(1-w)*IR_carry^2 = 0
=>  w* = IR_mom^2 / (IR_mom^2 + IR_carry^2)          (the orthogonal-blend optimal weight)
```

Plugging in this repo's numbers: momentum's honest net Sharpe (real Amihud-impact cost, the
number this spec's evidence digest names as the deployable estimate, not the light-cost headline)
is `IR_mom ≈ 0.8–1.0` [MEASURED, `docs/EDGE.md` §b honest column]; carry's standalone Sharpe is
`0.48` RAW and `0.57` BTC-beta-hedged [MEASURED, `research/xsec/overlay.py`, the same two figures
Section 6.1's overlay table above quotes] — using these two DIRECTLY MEASURED point Sharpes as
`IR_carry` avoids the extra, non-trivial step of converting the carry DSR range (`0.10–0.45`,
Section 6.1 above) back into an IR/Sharpe range, which the DSR formula does not invert
in closed form (DSR is a function of `SR_pp`, `n_obs`, and the higher moments jointly, Section
5.4-5.5 of SYSTEM.md — there is no single `SR` that maps to a given DSR without also fixing those
other inputs). Using the two measured point Sharpes directly is therefore the more defensible
sweep input; the illustrative `0.9^2/(0.9^2+0.3^2)` plug-in from an earlier draft is superseded by
the table below, which sweeps across the actually-cited uncertainty bands rather than one point:

```
w* = IR_mom^2 / (IR_mom^2 + IR_carry^2)          (the formula, restated)
```

**Sensitivity sweep 1 — `w*` across the full cited `IR_mom` and `IR_carry` ranges (algebra only, no
new measurement).** `IR_mom ∈ {0.8, 0.9, 1.0}` (the cited MEASURED band, Section 6.1 above) crossed
with `IR_carry ∈ {0.48, 0.57}` (MEASURED raw and beta-hedged carry Sharpe, Section 6.1's overlay
table) — 6 cells, each `w*` computed by the formula above:

| `IR_mom` \ `IR_carry` | `0.48` (raw) | `0.57` (beta-hedged) |
|---|---|---|
| `0.8` | `0.64/(0.64+0.2304)=0.735` | `0.64/(0.64+0.3249)=0.663` |
| `0.9` | `0.81/(0.81+0.2304)=0.779` | `0.81/(0.81+0.3249)=0.714` |
| `1.0` | `1.00/(1.00+0.2304)=0.813` | `1.00/(1.00+0.3249)=0.755` |

`w*` ranges `0.663–0.813` across every combination of the cited MEASURED bands — i.e. the
analytically-justified momentum weight is NEVER below `~0.66` and never above `~0.81` anywhere in
this grid, so `1 - w* ∈ [0.19, 0.34]` is the OPTIMISTIC Sharpe-additivity ceiling on carry's usable
risk-budget share BEFORE the drawdown evidence below is even considered — narrower and more
conservative than the single-point `~0.90` estimate an earlier draft plugged in (that point used
`IR_carry ≈ 0.3`, below the actual measured `0.48–0.57`, which understated carry's standalone
Sharpe and so overstated momentum's optimal share). Even at this sweep's most carry-favorable
corner (`IR_mom=0.8`, `IR_carry=0.57`, giving carry the largest defensible risk-budget share,
`1-w*≈0.34`), Section 6.1's overlay evidence below (drawdown 69.9% even after beta-hedging, and DSR
failure) is what actually drives `w_carry` to `0` in the deployed core — the Sharpe-additivity
bound alone would tolerate a modest nonzero carry weight, but the drawdown/DSR evidence overrides
it regardless of where in this sweep the true `IR_carry` sits.

**Sensitivity sweep 2 — how much does `w*` move if `rho` is NOT exactly `0`?** The `rho≈0`
simplification (discussed just above) drops a cross-term from `Var[w·r^mom+(1-w)·r^carry]`; a
non-zero `rho` changes the FIRST-ORDER CONDITION `d/dw[...]=0` used to derive `w*`. Re-deriving
with the cross-term retained:

```
Var(w) = w^2*sigma_mom^2 + (1-w)^2*sigma_carry^2 + 2*w*(1-w)*rho*sigma_mom*sigma_carry
d/dw Var(w) = 0
=>  w*(rho) = (sigma_carry^2 - rho*sigma_mom*sigma_carry) / (sigma_mom^2 + sigma_carry^2 - 2*rho*sigma_mom*sigma_carry)
```

which reduces exactly to `sigma_carry^2/(sigma_mom^2+sigma_carry^2)` — the variance-minimizing
form of the `rho=0` result above — when `rho=0`. This is an ILLUSTRATIVE (explicitly labeled, not
measured) arithmetic sweep of `w*(rho)` over a plausible band `rho ∈ {-0.3, 0, +0.3}`, holding
`sigma_mom, sigma_carry` at the ratio implied by `IR_mom=0.9, IR_carry=0.57` (the sweep-1 midpoint;
`sigma_carry/sigma_mom` cancels the actual scale and depends only on this ratio, so any consistent
choice of `mu_mom, mu_carry` giving these Sharpes works — take `sigma_mom=1, sigma_carry=0.9/0.57≈1.579`
for concreteness, i.e. carry's vol is higher per unit of Sharpe earned):

| `rho` | `w*(rho)` | `1 − w*(rho)` (carry's variance-minimizing share) |
|---|---|---|
| `-0.3` | `(1.579^2 - (-0.3)(1)(1.579)) / (1^2 + 1.579^2 - 2(-0.3)(1)(1.579)) = (2.493+0.474)/(1+2.493+0.947) = 2.967/4.440 = 0.668` | `0.332` |
| `0` (base case) | `1.579^2/(1+1.579^2) = 2.493/3.493 = 0.714` | `0.286` |
| `+0.3` | `(2.493-0.474)/(1+2.493-0.947) = 2.019/2.546 = 0.793` | `0.207` |

`w*(rho)` moves from `0.668` to `0.793` — a span of about `13 percentage points` — as `rho` sweeps
the plausible `[-0.3, +0.3]` band, versus `0.714` at the `rho=0` base case used throughout this
section. **This is the numeric answer to "how much does the `w*` conclusion depend on the
unmeasured `rho`": not much, in the sense that carry's OPTIMISTIC variance-minimizing share stays
within roughly `0.21–0.33` (never exceeding `1/3` of risk budget) across the full plausible `rho`
band, and — critically — this spread is a SHARPE-ADDITIVITY-ONLY bound that does not yet
incorporate the drawdown/DSR evidence below.** Once that evidence is folded in (as Section 6.2
does), the deployed conclusion `w_carry = 0` is unchanged across the ENTIRE swept `rho` range: no
point in either sweep table above produces a `w*` bound so generous to carry that the drawdown/DSR
disqualification would be overridden. This is the SYSTEM.md-style sensitivity move applied here:
the illustrative arithmetic precision of any single `rho` or `IR_carry` point is not load-bearing —
the qualitative and quantitative conclusion (`w_carry=0` in the deployed core) is invariant across
the full defensible range of both unmeasured/ranged inputs.

**The drawdown side of the argument is stronger than "carry is weak," and it was tested, not
assumed: overlays were TRIED on the standalone carry book and failed to fix the drawdown.**
`research/xsec/overlay.py` implements two causal (no-lookahead) defensive overlays on the raw
carry book (`beta_hedge_returns`, `vol_target_scale`, `drawdown_stop_scale`) and the results
argue for `w_carry = 0` more directly than the Sharpe-additivity bound above:

```
raw carry book                          Sharpe 0.48,  maxDD 74.9%,  income +12.9%/yr  (gross, standalone)
+ BTC-beta hedge only                   Sharpe 0.57,  maxDD 69.9%                      (real but modest lift)
+ causal drawdown-stop (on top)         Sharpe 0.10,  maxDD capped ~24%,  income ~0.6%/yr
```

(`docs/EDGE.md` §a/c; `docs/specs/REGIME.md` §5/§7's identical citation of
`research/xsec/overlay.py`'s `beta_hedge_returns`/`drawdown_stop_scale` outputs; `docs/specs/
PORTFOLIO.md` §"EVIDENCE records that overlays do not fix funding carry's structural drawdown".)
Beta-hedging the book's residual BTC-beta (the mechanism Section 6.1 above invokes qualitatively
to justify `rho ≈ 0`) recovers only a modest `0.48 -> 0.57` Sharpe lift and a `74.9% -> 69.9%`
maxDD reduction — nowhere near a fix. Forcing the drawdown down further with a causal
drawdown-stop (cutting exposure to a floor once running drawdown breaches a trigger, re-arming
only near a new peak — `overlay.py::drawdown_stop_scale`, a legitimate deployable rule since it
uses only realized equity strictly before the current week) DOES cap maxDD at ~24%, but it does so
by cutting the book to flat during precisely the stretches that also contain most of carry's
income, collapsing Sharpe to 0.10 and gross income to ~0.6%/yr. **The conclusion this supports is
that carry's drawdown is a persistent REGIME-BLEED (the book is structurally short the squeeze
regime, and a stop that avoids the squeeze also avoids the regime carry gets paid FOR being short
in), not one dodgeable tail event a smarter risk overlay could excise while keeping the income** —
a materially stronger argument for `w_carry = 0` in the CORE signal than the Sharpe-additivity
bound alone, because it rules out "just add a stop" as a way to make a larger carry weight safe.
Section 6.2's `w_carry = 0` rule is grounded in BOTH findings jointly: the analytic bound (`w* ∈
[0.663, 0.813]` for momentum across the sweep-1 grid, robust to `rho` per sweep 2) AND the
empirical failure of every overlay tried to convert carry into a standalone-deployable book.

### 6.2 The deployable combination rule

Given Section 6.1's bound and the additional (non-Sharpe) consideration that carry's drawdown is
regime-concentrated and fails DSR outright (not merely "weaker" than momentum but statistically
indistinguishable from a lucky draw at the DSR gate), this spec's combination rule is:

```
DEPLOYABLE SIGNAL  =  ensemble_momentum_signal(panel, d, lookbacks=(12,14,16))     [Section 4]
                       — i.e. w_carry = 0 in the CORE/deployable signal.

CARRY SATELLITE    =  carry_signal(panel, d, carry_lookback=7)                     [Section 5]
                       run and monitored SEPARATELY, sized (if at all) as a small satellite
                       allocation OUTSIDE the core book's risk budget, per docs/EDGE.md's
                       "Income overlay at most, never the book" verdict — never blended into
                       the core rank that PORTFOLIO consumes for the deployed book.
```

This differs from the shipped `combined_signal` function in `research/xsec/signals.py`, which
implements an EQUAL-weight (`v=0.5`) rank average of momentum and carry — `combined_signal` is
correctly built as a RESEARCH/diagnostic tool (it is exactly the artifact the validation harness
uses to show carry-blending does NOT improve the honest net Sharpe enough to justify its drawdown;
see `docs/EDGE.md` §a-b), and it remains available for that purpose. It is `w_carry = 0` (pure
`ensemble_momentum_signal`), not `combined_signal`, that Invariant BAND's promotion discipline
designates as the core deployable signal PORTFOLIO should consume for the live/paper book. A
future promotion of a nonzero carry weight requires re-running Section 6.1's bound with updated
`IR_mom`/`IR_carry` estimates AND clearing the DSR gate on the blended series — it is not a free
parameter to hand-tune.

> **Invariant NO-FIT — the combination weight is fixed and derived, never regressed.** `w_carry`
> (and the lookback-ensemble weights `v_L` of Section 4) are fixed by the closed-form derivation
> above, evaluated at PRE-REGISTERED estimates of `IR_mom`/`IR_carry` from validation runs that
> PRECEDE the weight's use — never by an in-sample regression of forward returns on
> `{momentum_score, carry_score}` over this repo's ~5-year sample. Such a regression is exactly
> the fitted-combiner search PBO (Section 4.1) already showed is overfit-prone at just 3 lookback
> candidates; fitting a 2-signal blend weight on the same limited sample is the identical failure
> mode one dimension over. Test: the deployed config's `w_carry` and `v_L` values must trace to a
> value computed by the Section 4.2/6.1 formulas from a FROZEN, timestamped `IR_mom`/`IR_carry`
> estimate — never to a value read off a regression coefficient or a grid-search argmax over the
> combination weight itself.

> **Invariant CARRY-ZERO — `w_carry` stays `0` unless BOTH a fresh `rho` measurement and a fresh
> DSR re-clearance occur.** This is a second, narrower invariant scoped specifically to Section
> 6.1's most consequential open uncertainty (the unmeasured `Corr(r^{mom}, r^{carry})`), isolating
> it behind its own explicit, machine-checkable guardrail rather than leaving it covered only by
> NO-FIT's broader "never regressed" rule — the same pattern LEVERAGE.md's Invariant K-LCB uses to
> isolate ITS single most load-bearing uncertainty (the deflated-Sharpe lower-confidence-bound
> input) behind a dedicated invariant rather than folding it into a more general one. Promoting
> `w_carry` away from `0` to any nonzero value requires ALL of: (1) a directly computed
> `Corr(r^{mom}_t, r^{carry}_t)` from the two books' realized weekly series (never the structural
> `rho≈0` argument of Section 6.1 alone — that argument licenses the `rho=0` APPROXIMATION used to
> derive `w*`, it does not license SKIPPING the measurement once a nonzero weight is actually being
> considered for promotion); (2) re-running Section 6.1's sweep 1/sweep 2 tables with the MEASURED
> `rho` and current `IR_mom`/`IR_carry` in place of the illustrative sweep inputs; (3) the resulting
> blended series independently clearing SYSTEM's DSR gate (Section 6.2's "not a free parameter to
> hand-tune" requirement, made mechanically checkable here). Test: a promotion report that sets
> `w_carry > 0` without a persisted, timestamped `Corr(r^{mom}_t, r^{carry}_t)` value AND a fresh
> DSR-gate pass on the blended series fails this invariant — the mere existence of Section 6.1's
> sensitivity sweep (which shows the CURRENT `w_carry=0` conclusion is robust to `rho`) is not
> itself grounds for shipping a nonzero weight; the sweep bounds the RISK of the `rho≈0`
> approximation, it does not substitute for measuring `rho` before acting on a different conclusion.

---

## 7. Parameter governance

Every SIGNAL parameter is exactly one bucket (COST.md/SYSTEM.md's RED/AMBER/BLUE/TEAL discipline,
adopted unchanged).

- **AMBER — supplied universe/panel metadata.** Universe membership and point-in-time listing
  span (from `UNIVERSE.md` / `Panel.listed`), the funding stamp schedule (8h, Binance USDT-M,
  ingested at the panel layer), the sleeve (crypto-only for this signal — Invariant SCOPE below).
- **RED — shared signal-construction constants, unreachable from per-symbol config.** The
  lookback BAND `{12, 14, 16}` trading days (Section 4.3, PBO-validated), the momentum skip
  `k = 0` (Invariant SKIP), the carry lookback `c = 7` days (Section 5.2), the ensemble weights
  `v_L = 1/3` each (Section 4.3), the carry combination weight `w_carry = 0` in the core signal
  (Section 6.2), and the eligibility discipline (full-window-only, Section 2.2/5.3). None of these
  may appear as a per-instrument override — a symbol with a "special" lookback or a
  symbol-specific carry weight is the bug (mirrors COST.md's Invariant C-cfg / SYSTEM.md's
  Invariant C).
- **BLUE — the finite, validated lookback family searched offline.** The `{12, 14, 16}` band
  itself is the surviving output of a BLUE grid search over a small candidate set of lookbacks
  (the search that produced the PBO = 0.31–0.47 evidence in Section 4.1), evaluated under CPCV and
  never refit on a rolling/continuous basis. Any FUTURE change to the band composition (adding
  `L=10` or `L=18`, say) re-enters this BLUE bucket and must re-run the full CPCV/PBO procedure of
  `research/xsec/cpcv.py` before promotion — it does not become RED by assertion.
- **TEAL — none.** SIGNAL fits nothing per-instrument at runtime; every quantity here is either a
  fixed RED constant or a raw read of the panel's `F_d`-measurable cells. (Contrast COST.md, whose
  EWMA funding forecast and impact coefficient are genuinely TEAL, fitted per-instrument online —
  SIGNAL has no analogous online fit.)

> **Invariant SCOPE — this signal is crypto-only.** `momentum_signal`/`carry_signal`/
> `combined_signal`/`ensemble_momentum_signal` are validated and deployable ONLY against a crypto
> perp panel (`research/xsec/panel.py::build_class_panel(sleeve="crypto", ...)`). The equity and
> commodity sleeves are momentum-ONLY, zero-funding underlyings in the panel layer
> (`research/xsec/universe.py`'s module docstring: "FUNDING = 0 for equities AND commodities") for
> an unrelated reason (they are cash underlyings with no perp funding to harvest) and were
> separately tested and found NOT to carry the cross-sectional momentum edge at all
> (`docs/EDGE.md` §d: "does NOT extend to equities or commodities… tested; those sleeves were
> directional-grade only"). Running SIGNAL's functions against a non-crypto panel is mechanically
> possible (the code is class-agnostic) but is explicitly NOT validated and NOT a deployable
> configuration; a report that promotes an equity/commodity cross-sectional signal without a fresh
> CPCV/PBO/DSR pass on THAT sleeve's own panel is the bug this invariant forbids.

---

## 8. Edge cases and invariants (consolidated)

This section consolidates every edge case a correct implementation must handle, cross-referencing
the invariant it is tested by.

1. **Empty or singleton cross-section.** `_pct_rank` on zero eligible names returns all-NaN; on
   exactly one eligible name returns `0.5` (neutral) rather than dividing by zero or raising.
   PORTFOLIO's `_leg_count` (consumer side) separately floors at "no book possible" below 2
   eligible names — SIGNAL's contract is simply to emit a well-defined `[0,1]`/NaN vector in every
   case, never `NaN/0` or an exception.

2. **Exact ties.** `_pct_rank`'s average-rank convention assigns tied raw scores the mean of the
   ranks the tied block would occupy (Section 3.1). This matters concretely for `carry_signal`
   whenever multiple names show byte-identical mean funding (common when a stretch of funding
   history is literally zero for several thinly-traded names) — ties must not be broken by
   insertion order (an implicit, non-reproducible tie-break) or by symbol name (an arbitrary bias
   toward alphabetically-early symbols); the implementation uses `argsort(kind="mergesort")`
   (stable sort) purely to make the SUBSEQUENT average-rank assignment deterministic given ties,
   not to break the tie itself.

3. **New listing at the edge of the lookback window.** A symbol whose `first_day` is exactly
   `d - k - L` (the earliest possible day it could be momentum-eligible) is included — the
   eligibility check is `>= 0` and `listed` at BOTH endpoints, an inclusive boundary. A symbol
   listed one day later than that is correctly excluded for this `d` and becomes eligible starting
   the day its live span first covers the full window. This boundary is a direct consequence of
   Section 2.2's eligibility definition and needs no special-casing.

4. **Funding gap on an otherwise-live day.** `Panel.funding` is `0.0` (not NaN) on a day the
   symbol was live but the funding lake recorded no stamp (`panel.py::build_panel`: "a real gap in
   the funding lake is a 0-accrual day"). `carry_signal`'s trailing mean therefore treats a
   funding gap as a genuine zero-accrual observation, not a missing one to be imputed or
   interpolated — this is a conservative, defensible choice (a gap is far more likely to be a data
   outage on an otherwise-uneventful funding period than a hidden extreme), but it is worth
   stating explicitly since a NaN-vs-zero confusion here would silently corrupt the trailing mean.

5. **Ensemble band member with different eligibility footprints.** Because `L=16` requires two
   more days of trailing history than `L=12`, a symbol newly past its `L=12` eligibility boundary
   is NOT yet eligible for the `L=16` component. `ensemble_momentum_signal` (Section 4.3) requires
   intersection eligibility across the WHOLE band (same discipline as `combined_signal`'s
   momentum/carry intersection) — such a symbol is excluded from the ensemble entirely for those
   few days rather than scored on a partial 2-of-3 band, keeping the ensemble's eligibility
   criterion simple and auditable at the cost of a few days' delayed entry for the newest listings.

6. **Degenerate / all-equal raw scores.** If every eligible name has byte-identical trailing
   return (e.g. a synthetic/test panel with constant prices), `_pct_rank` assigns every name the
   same average rank (the tied-block mean, which collapses to `0.5` when the tied block is the
   entire cross-section) — a well-defined neutral book (PORTFOLIO then forms no long/short
   distinction, correctly reflecting that the signal carries zero cross-sectional information that
   day), not a divide-by-zero or NaN propagation.

7. **`combined_signal`'s non-uniform average-of-ranks distribution.** As noted in Section 4.4, the
   simple average of two independent `[0,1]` percentile ranks is NOT itself uniformly distributed
   (it is approximately triangular for `m=2`, and more concentrated for larger `m`) — this is why
   every multi-signal combination in this spec re-ranks the averaged score with a FINAL `_pct_rank`
   pass rather than treating the raw average as the deliverable rank. Skipping the final re-rank
   would still produce a valid ORDERING (correctness for PORTFOLIO's quantile selection is
   unaffected) but would break the `[0,1]`-uniform-percentile CONTRACT other consumers (e.g. a
   future IC-weighting diagnostic that assumes a flat rank distribution) might rely on.

8. **Band-member disagreement — a symbol ranked top-quintile at one lookback and bottom-quintile
   at another.** This is the single most on-topic edge case for THIS spec given Section 4.1's own
   evidence: PBO = 0.31–0.47 and the `L=12`-wins-21/28-splits finding both document that the
   argmax-`L` selection is unstable, which is exactly the regime where individual band members can
   materially disagree on a given name (a coin with a sharp reversal 13–15 days ago can show a
   strongly positive `L=12` trailing return — the reversal is inside the window — while showing a
   strongly negative `L=16` trailing return — the reversal dominates a wider window that also
   captures the prior downtrend). `ensemble_momentum_signal` (Section 4.3–4.4) does not special-case
   this: it treats disagreement exactly like agreement, averaging the `m` component RANKS (not the
   raw returns) with equal weight `v_L=1/3` and re-ranking — a symbol on which two of three
   lookbacks agree strongly and one disagrees strongly lands near the AVERAGE of the three ranks,
   pulled toward the two-lookback consensus but not fully to either extreme. This is the intended
   behavior, not a bug to patch: the ensemble's entire justification (Section 4.2's variance
   inequality) is that averaging near-but-not-perfectly-correlated `eps_L` noise terms reduces
   selection-driven variance, and a band-member disagreement is a realized instance of that
   noise — an ensemble that instead special-cased disagreement (e.g. dropping the outlier
   lookback, or excluding the symbol from the book when band members disagree beyond a threshold)
   would itself be an ad hoc, non-pre-registered rule risking exactly the kind of per-name/per-day
   discretionary override P7 forbids. **No additional filtering on band-member disagreement is
   part of this spec's deployable signal**; a future report proposing one (e.g. "exclude names
   where the band's rank range exceeds `x`") must be validated with its own CPCV/PBO pass (BLUE
   bucket, Section 7) before promotion, exactly like any other change to the ensemble's
   construction, because a disagreement filter is itself a free parameter that could be
   overfit to the same ~5-year sample.

9. **IC-weighting selected with a negative or unstable `IC_L`.** If a future report follows Section
   4.3's pre-registration discipline and computes a frozen `IC_L` per lookback, it must handle the
   case where one or more `IC_L` are negative or close to zero over the frozen estimation window
   (plausible given the band members are similar-Sharpe by construction — Section 4.2 — so their
   ICs are not guaranteed to all be reliably positive on any given frozen sample). A weight
   `v_L PROPORTIONAL to IC_L` with a negative `IC_L` produces a negative combination weight, which
   is nonsensical for a rank-average (Section 4.4's `avg_i = sum_s v_s * rank^{(s)}_i` requires
   `v_s >= 0` for `avg` to remain a valid convex combination of `[0,1]` ranks). The well-defined,
   non-fitted fallback in that case is: clip negative `IC_L` to zero and renormalize the remaining
   weights to sum to 1, or — the more conservative option, and the one this spec recommends should
   IC-weighting ever be promoted — fall back to equal-weight entirely for that promotion cycle
   rather than silently zeroing out a band member, since a member with unstable IC on THIS frozen
   window is not evidence it lacks information on the LIVE window the deployed config will actually
   trade (the whole point of Invariant BAND is that no single lookback's estimated quality is
   trusted point-in-time). This spec does not deploy IC-weighting by default (Section 4.3), so this
   edge case is prospective — documented so a future promotion report is not the first place this
   failure mode is discovered.

---

## 9. Worked example (reproduce this bit-for-bit)

A minimal 4-symbol, 2-lookback illustration of the full pipeline: raw momentum → cross-sectional
rank → carry sign → ensemble re-rank, small enough to hand-check every number.

**Setup.** Rebalance day `d`. Four eligible crypto perps `A, B, C, D`. Trailing simple returns over
lookback `L=12` (skip `k=0`, so the window is `[d-12, d]`):

| Symbol | `r^{(12)}` (raw trailing return) | `mean_funding_7d` |
|---|---|---|
| A | +0.18 | +0.00060 (high positive — market crowded long) |
| B | +0.02 | -0.00010 (slightly negative) |
| C | -0.05 | +0.00005 |
| D | -0.05 | -0.00080 (deeply negative — market crowded short) |

**Step 1 — momentum cross-sectional rank (Section 3.1).** Sort `r^{(12)}`: `D=-0.05, C=-0.05` tied
for ranks 1–2 (average rank `1.5` each), `B=+0.02` rank 3, `A=+0.18` rank 4. With `n=4`:
`rank_i = (r_i - 0.5)/4`:

```
rank_D = (1.5 - 0.5)/4 = 0.25      rank_C = (1.5 - 0.5)/4 = 0.25
rank_B = (3 - 0.5)/4   = 0.625     rank_A = (4 - 0.5)/4   = 0.875
```

(Check: `_pct_rank`'s tie handling — C and D share raw value `-0.05`, so they share the tied-block
average rank `1.5`, giving both the identical momentum rank `0.25`, exactly as Section 8 edge
case 2 describes.)

**Step 2 — carry score and sign (Section 5.2).** `carry_score = -mean_funding`:

```
carry_score_A = -0.00060   carry_score_B = +0.00010   carry_score_C = -0.00005   carry_score_D = +0.00080
```

Sort ascending: `A=-0.00060` rank 1, `C=-0.00005` rank 2, `B=+0.00010` rank 3, `D=+0.00080` rank 4.
`carry_rank_i = (r_i - 0.5)/4`:

```
carry_rank_A = 0.125   carry_rank_C = 0.375   carry_rank_B = 0.625   carry_rank_D = 0.875
```

Sanity check against Section 5.2's derivation: `D` has the most NEGATIVE funding
(`-0.00080`) and correctly gets the HIGHEST carry rank (`0.875`, strong long conviction — it is
the name paying its shorts, so going long collects the credit); `A` has the most POSITIVE funding
(`+0.00060`) and correctly gets the LOWEST carry rank (`0.125`, strong short conviction).

**Step 3 — the deployable ensemble is momentum-only (`w_carry=0`, Section 6.2).** With only one
lookback shown here (`L=12`), `ensemble_momentum_signal`'s output for this single-member
illustration reduces to Step 1's `rank` exactly (Section 4.4's re-rank of a 1-element weighted
average is the identity on the ranks themselves — re-ranking a single already-uniform rank vector
returns the same vector). The full 3-member `{12,14,16}` ensemble would repeat Step 1 for `L=14`
and `L=16`, average the three rank vectors with `v_L=1/3` each, and re-rank; the arithmetic is
identical in kind, only wider.

**Step 4 — carry as a monitored satellite, not blended into the core.** The DEPLOYABLE signal
PORTFOLIO consumes for the core book is Step 3's momentum-ensemble rank `{A:0.875, B:0.625,
C:0.25, D:0.25}` UNCHANGED by carry. `combined_signal`'s research/diagnostic 50/50 blend (shown
for contrast, NOT the deployable path) would average Step 1 and Step 2's ranks
(`{A:(0.875+0.125)/2=0.5, B:(0.625+0.625)/2=0.625, C:(0.25+0.375)/2=0.3125, D:(0.25+0.875)/2=0.5625}`)
and re-rank — note how `A`'s strong momentum long-conviction is dragged down to a neutral `0.5` by
its crowded-positive-funding, illustrating concretely why blending materially changes which names
land in the long leg, and why Section 6.1's analysis says that blend is not warranted by carry's
weak, fragile standalone Sharpe.

---

## 10. Interfaces this spec's downstream consumers rely on

SIGNAL's complete outward contract (everything a PORTFOLIO.md / UNIVERSE.md spec may call):

```python
# research/xsec/signals.py  (existing; this spec documents its invariants)
def momentum_signal(panel: Panel, d: int, *, lookback: int, skip: int) -> np.ndarray: ...
def carry_signal(panel: Panel, d: int, *, lookback: int = 7) -> np.ndarray: ...
def combined_signal(panel: Panel, d: int, *, lookback: int, skip: int,
                     carry_lookback: int = 7) -> np.ndarray: ...   # RESEARCH/diagnostic only (§6.2)

# proposed addition (not yet in signals.py; specified by §4.3-4.4 of this document)
def ensemble_momentum_signal(panel: Panel, d: int, *, lookbacks: tuple[int, ...] = (12, 14, 16),
                              skip: int = 0, weights: tuple[float, ...] | None = None) -> np.ndarray: ...
```

Every function returns an `(n_symbols,)` array aligned to `panel.symbols`, values in `[0, 1]` for
eligible symbols and `NaN` for ineligible ones, satisfying Invariant NLA (causality), Invariant
RANK (point-in-time cross-sectional standardization), and — for the ensemble — Invariant BAND (no
single fitted lookback in the deployed path), Invariant NO-FIT (no regressed combination weight),
and Invariant CARRY-ZERO (`w_carry` stays `0` absent a fresh `rho` measurement and DSR
re-clearance). PORTFOLIO.md consumes exactly this contract to build dollar-neutral quantile weights
(`research/xsec/portfolio.py::quantile_weights` / `liquidity_weights`); SIGNAL never constructs a
weight, a trade, or a position itself (P1).

---

## References

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management,
  40(5), 94–107. — the max-order-statistic selection-bias inequality underlying Section 4.2's
  point-selection-vs-ensemble argument, and the DSR machinery `research/xsec/validate.py` reuses
  from `convexity.system.validate.deflated`.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-Mathematics and
  Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.*
  Notices of the AMS, 61(5), 458–471. — the combinatorial-purged-CV / PBO framework
  `research/xsec/cpcv.py::compute_pbo` implements (logit-of-relative-rank estimator, Section 4.1).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 11 (Combinatorial
  Purged Cross-Validation) and Chapter 12 (Backtesting through Cross-Validation). Wiley. — the
  CPCV path-construction and PBO estimator this repo's `research/xsec/cpcv.py` and
  `docs/specs/SYSTEM.md` §6.4/§7 both cite and reuse.
- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.), Chapter 6
  (Information Analysis / combining signals via the Information Coefficient). McGraw-Hill. — the
  IC-weighted signal-combination framework Section 4.3 names as the (non-deployed, see Invariant
  NO-FIT) alternative to equal-weight lookback ensembling. This is now this citation's ONLY role in
  this file — Section 6.1's momentum/carry book-level blend cites Sharpe (1994) and
  Elton/Gruber/Brown/Goetzmann below instead, since that section's result is a two-asset tangency-
  portfolio weight, not an IC-weighted multi-signal combination (an earlier draft cited Grinold &
  Kahn for both sections; this citation-scope correction removes the double-duty).
- Sharpe, W. F. (1994). *The Sharpe Ratio.* Journal of Portfolio Management, 21(1), 49–58. —
  PRIMARY citation for Section 6.1's `w* = IR_mom^2/(IR_mom^2+IR_carry^2)` orthogonal-blend weight:
  Sharpe's generalized-Sharpe-ratio treatment of combining two return series is the closed-form
  result Section 6.1's `d/dw` first-order condition reproduces, without needing Grinold & Kahn's
  IC-framework machinery (which is scoped to stock-selection signal combination, Section 4.3
  above, not whole-book return-stream blending).
- Elton, E. J., Gruber, M. J., Brown, S. J., & Goetzmann, W. N. *Modern Portfolio Theory and
  Investment Analysis* (any edition; the two-asset tangency-portfolio derivation is a standard,
  edition-stable textbook result). Wiley. — secondary/corroborating citation for the same two-asset
  mean-variance tangency-weight algebra Section 6.1 uses; named here because the `w*` formula is
  textbook Markowitz portfolio theory at `n=2` assets, independent of the Sharpe-ratio framing
  Sharpe (1994) gives it.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time Series Momentum.* Journal of
  Financial Economics, 104(2), 228–250. — background on trailing-return momentum as a signal
  class; this repo's edge is the cross-sectional (Jegadeesh & Titman-style relative-strength)
  variant, not the time-series variant, but the skip/lookback design vocabulary is shared.
- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications
  for Stock Market Efficiency.* Journal of Finance, 48(1), 65–91. — the foundational
  cross-sectional relative-strength momentum result this signal generalizes to the crypto perp
  universe.
- Binance USDT-Margined Futures funding-rate mechanism documentation (premium index + interest
  component, clamped to a funding cap) — the no-arbitrage mechanism Section 5.1's derivation
  states informally and `docs/specs/COST.md` §5.5b formalizes in full (premium/interest/clamp
  decomposition); Section 5's sign derivation is the direct consequence of that mechanism.

---

## Provenance of every numeric constant in this file

So no constant here is ever mistaken for an invented or "reasonable-sounding" number. Every
source below is either (a) a defaulted/implemented value read directly off a named function in
`research/xsec/*.py`, or (b) a validation RESULT recorded in `docs/EDGE.md` (the repo's decision
memo synthesizing the CPCV/PBO/DSR/overlay validation runs) and cross-cited, where the underlying
run lives, in `research/xsec/cpcv.py` / `research/xsec/overlay.py`'s own docstrings and the
sibling pillar specs that independently cite the same numbers from the same runs
(`docs/specs/PORTFOLIO.md`, `docs/specs/UNIVERSE.md`, `docs/specs/REGIME.md`,
`docs/specs/SYSTEM.md`, `docs/specs/LEVERAGE.md`). No entry cites this document's own commissioning
instructions as a source.

| Constant | Value | Source |
|---|---|---|
| Lookback band | `{12, 14, 16}` trading days | `docs/EDGE.md` §b "12–16 day BAND"; the CPCV/PBO grid search over the lookback family, `research/xsec/cpcv.py::compute_pbo` |
| Best-in-sample lookback (headline, biased) | `L=14` | `docs/EDGE.md` §b "headline" column (light-cost single-path peak) |
| Best-in-sample lookback (per-split mode) | `L=12` on 21/28 CPCV splits | `research/xsec/cpcv.py::compute_pbo`'s per-split `best_is_labels` output, synthesized in `docs/EDGE.md` §c |
| PBO range | `0.31–0.47` across partitions | `research/xsec/cpcv.py::compute_pbo` (the `pbo` field across `N_groups`/`k_test` partition choices); point estimate `0.36` in `docs/EDGE.md` §c |
| PBO floor | never `< 0.2` clean-pass | `research/xsec/cpcv.py::compute_pbo` output across the partition grid referenced above |
| CPCV mean OOS path Sharpe (selected config) | `≈0.91` vs. naive full-sample `≈0.98` (haircut `≈0.07`) | `research/xsec/cpcv.py::path_distribution` / `PathSharpeSummary.mean`/`.naive_full_sharpe`/`.haircut`; cited identically in `docs/specs/PORTFOLIO.md`, `docs/specs/SYSTEM.md` §"Result", `docs/specs/UNIVERSE.md`, `docs/specs/REGIME.md`, `docs/specs/LEVERAGE.md` |
| `frac_pos` (CPCV path-positivity) | `0.85–1.00`, the robust statistic (contrast fragile `p05` = `+0.42` to `−0.59`) | `research/xsec/cpcv.py::PathSharpeSummary.frac_pos`; `docs/EDGE.md` §c; `docs/specs/PORTFOLIO.md` Invariant R-stat |
| Momentum honest net Sharpe | `~0.8–1.0` | `docs/EDGE.md` §b honest column (real Amihud-impact cost reconciliation) |
| Momentum light-cost/single-peak Sharpe | `~1.6–1.9` (cite as upper bound only) | `docs/EDGE.md` §b headline column (light-cost proxy, single fitted `L`) |
| Carry standalone DSR | `0.10–0.45` across overlay variants | `convexity.system.validate.deflated` applied to `research/xsec/overlay.py`'s raw/hedged/DD-stopped carry series; `docs/EDGE.md` §d |
| Carry standalone gross income | `+12.9%/yr` | `docs/EDGE.md` §a "Funding carry"; the raw (no-overlay) series in `research/xsec/overlay.py` |
| Carry standalone max drawdown | `74.9%` (raw); `69.9%` (BTC-beta hedged); `~24%` (drawdown-stopped, at the cost of Sharpe 0.10 / income ~0.6%/yr) | `research/xsec/overlay.py::beta_hedge_returns`/`drawdown_stop_scale` output; `docs/EDGE.md` §a; identical figures in `docs/specs/REGIME.md` §5/§7 and `docs/specs/PORTFOLIO.md` |
| Carry overlay Sharpe path | raw `0.48` → beta-hedged `0.57` → DD-stopped `0.10` | `research/xsec/overlay.py`; `docs/specs/REGIME.md` §7 cites the identical three-point path |
| Carry lookback | `7` days | `research/xsec/signals.py::carry_signal` default parameter |
| Momentum skip default | `k=0` (grid `{0,3,7}`) | `research/xsec/signals.py::momentum_signal` docstring |
| Ensemble weights | `v_L = 1/3` each (equal-weight) | Section 4.2 derivation (fixed-before-scoring weight, P7-compliant); the repo ships equal-weight as the non-fitted default (Section 4.3) |
| Core-signal carry weight | `w_carry = 0` | Section 6.1 sensitivity sweep 1 (`w* ∈ [0.663, 0.813]` across the full cited `IR_mom∈{0.8,0.9,1.0}` × `IR_carry∈{0.48,0.57}` grid, so `1-w*∈[0.19,0.34]` bounds carry's OPTIMISTIC Sharpe-additivity share) and sweep 2 (`w*(rho)∈[0.668,0.793]` across `rho∈{-0.3,0,+0.3}`, ILLUSTRATIVE) jointly with the overlay evidence above (no overlay converts carry to standalone-deployable, so a nonzero core weight is not defensible even within either sweep's most carry-favorable corner) |
| `w*` sensitivity sweep inputs (sweep 1) | `IR_mom∈{0.8,0.9,1.0}` (MEASURED), `IR_carry∈{0.48,0.57}` (MEASURED raw/beta-hedged carry Sharpe) | `docs/EDGE.md` §b (IR_mom); `research/xsec/overlay.py` raw/`beta_hedge_returns` output (IR_carry) — Section 6.1 sweep 1 |
| `w*(rho)` sensitivity sweep inputs (sweep 2) | `rho∈{-0.3,0,+0.3}` (ILLUSTRATIVE — no persisted `Corr(r^mom,r^carry)` run backs any specific value), `sigma_carry/sigma_mom≈1.579` (derived from the `IR_mom=0.9,IR_carry=0.57` sweep-1 midpoint) | Section 6.1 sweep 2 derivation; see Invariant CARRY-ZERO for the promotion-time re-measurement requirement |
| Universe size | `~50` names (peak `60–75`, gain inside block-bootstrap CI) | `docs/EDGE.md` §c universe-size sweep (block-bootstrap CI on the weekly return series across `N`) |
| CPCV split family | `N_groups=8, k_test=2, 28 paths, 1-week embargo` | `research/xsec/cpcv.py::run_cpcv` defaults (`n_groups=8, k_test=2`, `C(8,2)=28` paths, `embargo=1` week) |
