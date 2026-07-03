# UNIVERSE.md — build spec for point-in-time universe eligibility

This file is the authoritative build spec for the UNIVERSE pillar. Build to the
contracts and invariants below. Where this file and your own instinct disagree,
this file wins. Ask before deviating from any **Invariant**.

UNIVERSE answers exactly one question, asked fresh at every weekly rebalance
timestamp `t`:

> Of all names that have ever traded, which ones is it LEGITIMATE to rank,
> weight, and trade — long or short — this week, using only what was knowable
> strictly before `t`?

UNIVERSE is a **gate, not a forecaster**. It never ranks names by expected
return, never sizes a position, never picks a lookback. It hands SIGNAL a
point-in-time eligible set `U_t` (and, separately, a stricter shortable subset
`U_t^short ⊆ U_t`); SIGNAL ranks within `U_t`, PORTFOLIO sizes within the ranked
set respecting `U_t^short` on the short leg, ACCOUNTING charges the realized
friction. UNIVERSE is a PROVIDER: an indicator process and a liquidity order
statistic, never a decision.

---

## 0. Risk-unit framing (read this before anything else)

This spec supersedes the two sibling specs' framing (`docs/specs/COST.md`,
`docs/specs/SYSTEM.md`), which were written for the OLD directional-breakout
system and priced a **single trade's stop-to-target R-multiple**. That system
is dead: **EDGE.md** records gross directional R ≈ 0 across every timeframe,
exit rule, entry filter, leverage level, and asset class tested, even inverted
— the breakout signal carries no information about the next move, so realized
loss was pure friction expressed in R-units. UNIVERSE does not inherit that
framing.

The live risk unit here is **next week's portfolio return** — one Friday-close
to Friday-close observation of a dollar-neutral, cross-sectionally ranked book.
UNIVERSE's job under this framing is narrower and cleaner than under the old
one: there is no per-trade stop, no per-instrument invalidation level, no
holding-period uncertainty (`5.5d`'s "the `m` problem" from COST.md does not
arise — the holding period is exogenously fixed at exactly one week by the
rebalance calendar, `Δt = 7` calendar days, always). What UNIVERSE still owes
every downstream module, unchanged in spirit from COST.md's discipline, is:

- **P1 (from COST.md, salvaged).** One membership truth, consumed identically
  by SIGNAL (ranks only within `U_t`), PORTFOLIO (sizes only within `U_t`,
  shorts only within `U_t^short`), and ACCOUNTING (attributes cost only to
  positions that were UNIVERSE-eligible when opened). Divergent per-module
  universe filters are the bug this spec forecloses.
- **P2 (from COST.md, salvaged).** UNIVERSE is a provider, never a decider. It
  emits a boolean indicator and an order statistic; it never emits a trade.
- **P6 (from COST.md, salvaged, re-aimed).** Conservative under uncertainty:
  missing/stale liquidity data excludes a name (fails it out of `U_t`), it
  never defaults a name IN. A name half-way through the minimum-history window
  is excluded, never partially admitted.
- **NEW — point-in-time-ness is the load-bearing property**, replacing COST's
  causality-of-cost-forecast focus. The single failure mode this spec exists
  to foreclose is **survivorship bias**: building `U_t` by looking at today's
  roster and reading it backward in time. Section 1 formalizes why that fails
  and Section 6 gives the negative worked example this repo already produced.

Scope: crypto USDT-M perpetuals on Binance (the ONLY sleeve with a live edge —
EDGE.md: momentum does not extend to the equities/commodities sleeves, tested),
weekly (Friday 00:00 UTC close) rebalance cadence, ~50-name target universe,
dollar-neutral quintile long/short book. Implemented today in
`src/convexity/panel/panel.py` and `src/convexity/panel/candidates.py`; this
spec is the formal contract those modules already approximate. **Status
update:** the liquidity/hysteresis/shortability logic this line originally
flagged as "not yet built" is now implemented — see
`src/convexity/universe/{liquidity/hysteresis.py, shortability/{venue_gate.py,
liquidity_gate.py}, eligibility/{stopping_times.py,history_gate.py},
liquidity/ranking.py, universe_api.py, validate/pit_check.py}` (verified by
reading each file: all fully implemented, with essentially zero
`NotImplementedError` stubs remaining).

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its
first point of use in prose; this table is a lookup, not a substitute for
those definitions — mirrors `docs/specs/LEVERAGE.md` §0.1's column structure
exactly, so a reader moving between pillar specs finds the same lookup shape
in each). The rigor bar this repo applies (`docs/specs/SYSTEM.md` §0.1: "a
first-time reader never has to forward-reference") requires this table to
exist BEFORE Section 1, not after it — every symbol below is used with
EXACTLY this meaning everywhere in this document. Week index `t` runs over
Friday 00:00 UTC rebalances, the same clock SIGNAL/PORTFOLIO/SYSTEM use
(`src/convexity/portfolio/engine.py`).

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `t`, `t_0, t_1, …` | Weekly rebalance timestamps (Friday 00:00 UTC) | — | 1.1 |
| `F_t` | Filtration: all market data (price, volume, funding, listing/delisting) knowable up to and including `t` | — | 1.1 |
| `1{i ∈ U_t}` | Eligibility indicator process for instrument `i` | — | 1.1 |
| `τ_list_i`, `τ_delist_i` | Listing / permanent-delisting stopping times for instrument `i` | AMBER | 1.2 |
| `R`, `D` | Terminal-date survivor set / delisted set (used only in the Section 1.3 bias proof, never in the deployed construction) | — | 1.3 |
| `g` | A generic return statistic used in the Section 1.3 bias inequality (e.g. mean return) | — | 1.3, 1.5 (A1) |
| `L, L_1, L_2, L_3` | SIGNAL's lookback family, `{12, 14, 16}` trading days (SIGNAL.md §4.2–4.3, restated not owned here) | SIGNAL-owned | 1.4 |
| `L_max` | `max` of the lookback family = 16 trading days | derived (SIGNAL-owned) | 1.4 |
| `w_warmup` | UNIVERSE's own additional warmup, pinned equal to `W` (never an independent free parameter) | RED (tied to `W`) | 1.4 |
| `τ_history_i` | Minimum-history stopping time, `τ_list_i + L_max + w_warmup` | derived | 1.4 |
| `U_t^base` | Base eligibility set after the history-warmup refinement (Section 1.4), before any liquidity gate | — | 1.4 |
| `W` | Trailing dollar-volume window (days) for `DV_i(d)` | RED (default 30, BLUE grid `{20,30,45}`) | 2.1 |
| `DV_i(d)` | Trailing `W`-day dollar volume of instrument `i` as of day `d` | derived (closed-form order statistic, no TEAL — Section 8) | 2.1 |
| `rank_i(d)` | Percentile rank (0,1] of `DV_i(d)` among `{j ∈ U_d^base}`; 1 = most liquid | derived | 2.1 |
| `q_in`, `q_out` | Hysteresis entry / exit percentile-rank thresholds, `q_in > q_out` | RED | 2.2–2.3 |
| `U_t^base-liquid` | Base-eligible set after the Section 2.2 hysteresis state machine | — | 2.2 |
| `Δ_i(t)` | Week-over-week change in `rank_i`, `rank_i(t) − rank_i(t-1)` | derived | 2.3 |
| `σ_rank` | Method-of-moments std. dev. of `Δ_i(t)`, pooled near the hysteresis boundary | ILLUSTRATIVE (Section 2.3) | 2.3 |
| `n`, `α` | Whipsaw-suppression horizon (weeks) / tolerance used in the `h_min` calibration | RED (calibration inputs, `n=4`, `α=0.10`) | 2.3 |
| `h_min(n, α)`, `h` | Conservative calibration target / actual deployed hysteresis band half-width | derived (`h_min`) / RED via `q_in,q_out` and BLUE-searchable (`h`) | 2.3 |
| `Φ`, `Φ⁻¹` | Standard normal CDF / quantile function | — | 2.3 |
| `N` | Target universe cardinality cap (RED default 50; BLUE grid `{50,60,75,100}`) | RED | 3.2 |
| `U_t` | Final size-`N` eligible universe = `top-N(U_t^base-liquid, key=rank_i(t))` | — | 3.2 |
| `C` | Liquidity-share cap multiple (capacity dial, off by default) | RED | 3.3 |
| `U_t^short` | Stricter shortable subset, `U_t^short ⊆ U_t` | — | 4.2 |
| `short_permitted_i(t)` | AMBER venue-contract-status flag | AMBER | 4.2 |
| `short_liquid_i(t)` | Stricter short-side liquidity order-statistic test | RED (functional form; Section 4.2 disclosed gap) | 4.2 |
| `k` | PORTFOLIO's quantile leg count (`portfolio.py::_leg_count`), restated not owned here | PORTFOLIO-owned | 4.4 |
| `CPCV`, `frac_pos`, `PBO`, `DSR`, `PSR`, `MinTRL` | SYSTEM's validation statistics, restated not owned here (SYSTEM §4–§5) | SYSTEM-owned | 7 |

---

## 1. Formal setup: eligibility as an F_t-measurable indicator process

### 1.1 Filtration and the eligibility indicator

Let `t ∈ {t_0, t_1, t_2, …}` index the weekly rebalance timestamps (Friday
00:00 UTC, matching `src/convexity/panel/panel.py`'s daily-close convention: the
close at UTC day boundary `D` is knowable at `D`'s close, so a Friday rebalance
at `t` may use the panel row for the day ending at `t` and every prior row, but
not the row that begins after `t`). Let `F_t` be the filtration generated by
all market data (prices, volumes, funding, listing/delisting events) up to and
including `t`: `F_t = σ({price_i(s), volume_i(s), listed_i(s) : s ≤ t, all i})`.
This is the SAME symbol pinned in the §0.1 notation table above — every later
reference to "`F_t`-measurable" (Section 1.4's `τ_history_i`, Section 2.2's
hysteresis state, Section 8's governance discussion) resolves to exactly this
definition and no other; nothing downstream silently narrows or widens the
filtration (e.g. Section 2.2's hysteresis FSM adds a one-step memory of prior
STATE, not a new field of information — the underlying `F_t` it is measurable
with respect to is unchanged from this definition).

For instrument `i`, define the **eligibility indicator process**:

```
1{i ∈ U_t} : Ω × T → {0, 1}
```

**Definition (point-in-time eligibility).** `1{i ∈ U_t}` is `F_t`-measurable:
its value at `t` is a deterministic function of information in `F_t` only. No
eligibility decision at `t` may read `price_i(s)`, `volume_i(s)`, or
`listed_i(s)` for any `s > t`.

This is the formal statement of "no lookahead" applied to universe
construction specifically (as opposed to signal construction, which
`src/convexity/signal/signals.py`'s docstring already states informally: "computed
STRICTLY from panel data at rows <= D"). UNIVERSE is the layer that must
satisfy this for the *membership* decision, before SIGNAL ever touches a rank.

### 1.2 Listing and delisting as stopping times

Model each instrument's tradeable life as an interval bounded by two **stopping
times** with respect to `F_t`:

```
τ_list_i    = inf{ t : instrument i has a valid, tradeable quote at t }
τ_delist_i  = inf{ t : instrument i permanently ceases to have a valid,
                       tradeable quote at t }        (= +∞ if never observed to delist)
```

Both are stopping times: the event `{τ_list_i ≤ t}` (respectively
`{τ_delist_i ≤ t}`) is `F_t`-measurable, because "did this instrument have a
valid quote by time `t`" is answerable from data available at `t` — you do not
need to look into the future to know a coin has *already* listed or *already*
delisted. This is the standard stopping-time definition (Karatzas & Shreve
1991, Ch. 1) specialized to a listing/delisting event.

Define the **base eligibility window**:

```
1{ τ_list_i ≤ t < τ_delist_i }
```

i.e. instrument `i` is in its tradeable life at `t` iff it has already listed
and has not yet permanently delisted. This is exactly the `listed[date, sym]`
mask `src/convexity/panel/panel.py::build_panel` computes: `listed = ~np.isnan(close)`,
with `first_day`/`last_day` derived **per-symbol from its own observed span**
("no union backfill" — the docstring's own words) — this is `τ_list_i` and
`τ_delist_i` read off the data, not assumed.

### 1.3 Worked derivation — why conditioning on trailing-only stopping times avoids survivorship bias

**Claim.** A universe construction rule of the form "`i ∈ U_t` iff
`τ_list_i ≤ t < τ_delist_i` AND `i`'s eligibility statistics use only data in
`F_t`" is survivorship-bias-free. A universe construction rule of the form
"`i ∈ U_t` for all `t` iff `i` is a member of the roster observed at the
terminal date `T`" is NOT, and the bias has a specific, derivable sign and
mechanism.

**Setup.** Let `R = {i : τ_delist_i > T}` be the set of names still alive at
the terminal (e.g. "today's") date `T` — the *survivors*. Let `D = {i :
τ_delist_i ≤ T}` be the set that delisted (went to zero, got hacked, rug-pulled,
or otherwise permanently stopped trading) before `T`. A backward-looking
"modern roster" construction sets `U_t := R` for every historical `t ≤ T`,
independent of `t`.

**Mechanism of the bias.** For any historical `t < T` at which some `i ∈ D` was
in fact tradeable (`τ_list_i ≤ t < τ_delist_i`), the survivorship rule sets
`1{i ∈ U_t} = 0` even though `i` truly belonged in the point-in-time cross-
section at `t`. Conversely it never excludes any `j ∈ R` at any `t` after
`τ_list_j`, regardless of how thin, new, or untested `j`'s history was at that
`t`. Two failures compound:

1. **Selection on the outcome.** `R` is defined using information from
   `[t, T]`, which is exactly the interval *strictly after* `t` for a
   historical rebalance at `t < T`. Membership in `R` is `F_T`-measurable, not
   `F_t`-measurable, so `1{i ∈ R}` **violates the Section 1.1 definition** for
   every `t < T`: the eligibility test literally uses information from the
   future relative to the rebalance it is being applied to.
2. **The direction of the bias is not neutral.** Names in `D` almost
   mechanically **underperformed** on their way to delisting (a name rarely
   delists after going up — it delists after collapsing, getting hacked, or
   being abandoned). So `R` is enriched for names whose *realized* history
   happens to look good ex post, and any cross-sectional statistic computed
   over `U_t := R` inherits a positive bias on average past return and a
   negative bias on realized cross-sectional dispersion of losers — exactly
   the two channels a momentum strategy's long and short legs are built on.
   Formally, for a return statistic `g` (e.g. mean return, or "does the
   bottom quintile really underperform"), `E[g | i ∈ R] ≥ E[g | i ∈ R ∪ D]`
   whenever `g` is monotone in realized terminal survival — which holds by
   construction for "did not go to zero." The inequality is generically
   strict when `D` is non-empty, so `E[g(U_t = R)] > E[g(U_t = R ∪ D restricted
   to F_t)]`: the survivorship-conditioned universe overstates the true
   point-in-time-achievable statistic.

**Why the F_t-only stopping-time rule is immune.** Under
`1{i ∈ U_t} = 1{τ_list_i ≤ t < τ_delist_i}`, whether `i` is later in `R` or `D`
plays no role: the test is answerable using only data through `t`, so it
satisfies Section 1.1 by construction, and no future outcome (survival,
delisting, or performance) can leak into the `t`-dated eligibility decision.
A name that will delist next month is included today if it is genuinely
tradeable today; a name that will 10x next year gets no head start. This is
precisely why `build_panel`'s `listed` mask is derived "from each symbol's OWN
valid span... NEVER from the union" (own docstring) — the union-of-all-history
approach is the `R`-style backward roster this section proves is biased; the
per-symbol-own-span approach is the stopping-time rule that is provably not.

**Corollary (used in Section 6).** Because the bias in the survivorship
construction operates through the DIRECTION of realized returns, its effect on
a measured Sharpe ratio is not merely "some noise" — it is a systematic
inflation with no in-sample symptom (the survivorship-biased backtest looks
internally consistent; nothing in the walk-forward or CPCV of *that*
construction reveals the bias, because the bias is baked into every fold
identically). This is why a purely statistical validation battery (Section
7 below) is a *necessary but not sufficient* check — Invariant PIT (Section 8)
must be enforced structurally, not discovered empirically.

### 1.4 Minimum-history requirement — a warmup stopping time

Even a genuinely-listed, currently-live name is not immediately eligible: it
needs enough trailing history for the signal that will consume it to be
computable. Restating SIGNAL.md's band notation exactly rather than pointing
at it blind: SIGNAL deploys an ENSEMBLE over the lookback family
`L ∈ {L_1, L_2, L_3} = {12, 14, 16}` trading days (SIGNAL.md §4.2-4.3; `m = 3`
members), never a single fitted `L` (PBO = 0.31–0.47 over this family,
Section 7 below).

**Cross-pillar disclosure.** This `m=3` set `{12, 14, 16}` is the single canonical lookback
ensemble used consistently across all five pillars that reference it (SIGNAL, UNIVERSE,
PORTFOLIO, EXECUTION, SYSTEM). An earlier drafting pass introduced a divergent, ungrounded
`m=5` variant `{12, 13, 14, 15, 16}` in PORTFOLIO.md and SYSTEM.md, silently interpolating two
untested lookbacks (`L=13`, `L=15`); that divergence has since been corrected in both files, and
`{12, 14, 16}` is now the only ensemble deployed anywhere in the repo. Define

```
L_max := max_{L in family} L = 16 trading days
```

— the top of the 12–16 day band, cited by name against the empirical
baseline (Section 0's Scope / this file's References), not against a
non-existent "Evidence Digest" section (there is none in this file; the
canonical source for the family and its PBO is SIGNAL.md §4.1, cross-checked
in Section 7 below).

Let `w_warmup` be the ADDITIONAL warmup consumed by any UNIVERSE-side
statistic itself, over and above the signal's own lookback. UNIVERSE's own
trailing statistic is the `W`-day dollar-volume order statistic of Section
2.1 (`src/convexity/panel/panel.py::trailing_dollar_volume`'s `window: int = 30`),
which needs `W` trailing days of its own to be a well-formed mean, exactly
symmetric to the signal's need for `L` trailing days. Set

```
w_warmup := W = 30 trading days       (RED, tied 1:1 to Section 2.1's window,
                                        never an independent free parameter)
```

i.e. `w_warmup` is NOT a new tunable — it is defined to equal whatever value
Section 2.1's RED liquidity-window constant `W` currently holds, so a change
to `W` (Section 8's BLUE grid `W ∈ {20, 30, 45}`) automatically propagates to
`τ_history_i` without a second governance decision. This is a deliberately
conservative choice: it does not attempt to net the two warmup requirements
against shared calendar days (`L_max` days and `W` days can, in principle,
overlap on the calendar since both are trailing windows ending at the same
`t`), it simply sums them, so `τ_history_i` below is an upper bound on the
true minimum history needed, never an underestimate — consistent with P6's
conservative-exclusion discipline (Section 0). With the current RED defaults
(`L_max = 16`, `W = 30`):

```
τ_history_i = τ_list_i + L_max + w_warmup = τ_list_i + 16 + 30 = τ_list_i + 46 trading days
```

(trading days; ≈2.2 months of calendar time before a newly-listed name is
even eligible to be RANKED, well before it can be sized — this is the
concrete number `src/convexity/panel/panel.py::universe_size_by_year`'s
growth-from-a-handful-of-majors shape reflects empirically, Section 3.2).

and the refined base eligibility test:

```
1{ i ∈ U_t^base } = 1{ τ_history_i ≤ t < τ_delist_i }
```

`τ_history_i` is itself `F_t`-measurable at any `t ≥ τ_history_i` (it is a
fixed offset from an already-observed `τ_list_i`), so this refinement
preserves the Section 1.1 property. This is the formal version of
`src/convexity/signal/signals.py::momentum_signal`'s runtime guard: `if start < 0:
return raw` (an all-NaN, i.e.-ineligible, vector) when day `d - skip - lookback`
predates the panel's first row for that lookback. UNIVERSE's contract is that
this check must be evaluated PER-SYMBOL against `τ_list_i` (as `panel.listed`
already does via the endpoint check `live = panel.listed[end] & panel.listed[start]`
in `momentum_signal`), not against the panel's global first row — a symbol
that listed in 2024 must not borrow eligibility from BTC's 2019 history.

> **Invariant HW — history warmup is per-symbol, never panel-global.**
> `τ_history_i` is computed from instrument `i`'s own `τ_list_i`, never from
> the earliest-listed symbol in the panel or from a fixed calendar date. Test:
> a symbol injected with a synthetic `τ_list` of "30 trading days before `t`"
> is excluded from `U_t^base` at every `t` for which `t − τ_list < L_max +
> w_warmup = 46`, regardless of how much history OTHER symbols in the panel
> have.

### 1.5 Assumptions and regularity conditions

Section 1.3's central inequality and Section 2.2's F_t-measurability claim
for the hysteresis FSM both rely on conditions that hold in this repo's
setting but are not universally true, and are stated explicitly here rather
than left implicit:

- **(A1) Monotonicity of `g` in realized terminal survival.** Section 1.3's
  inequality `E[g | i ∈ R] ≥ E[g | i ∈ R ∪ D]` requires `g` monotone
  non-decreasing in "survived to `T`." This holds by construction for `g` =
  "did not go to zero" (survival is binary and the statistic is the
  indicator itself). For `g` = mean realized return, the claim that "crypto
  delistings are overwhelmingly preceded by price collapse toward zero, not
  by delisting-after-a-rally" is **NOT formally verified anywhere in this
  repo** — grep against `src/convexity/panel/panel.py` and
  `src/convexity/cost/accounting.py` (the two modules that actually handle a
  delist event, at `panel.py:5,114` and `accounting.py:76-77,112,131`) finds
  only the MECHANICAL handling of a delisting (last-live-bar close, drift-to-
  zero weight, no hold-through), never a script or notebook that measures the
  realized return path INTO a delisting event across the panel's history.
  This spec therefore asserts the claim from general knowledge of crypto
  market structure (delistings are typically exchange risk-management actions
  triggered by a price/liquidity collapse, or exit scams/hacks that themselves
  crash the price — not typically actions taken against a name that just
  rallied), **not from a persisted empirical measurement in this repo** — the
  same MEASURED-vs-asserted distinction `docs/specs/SYSTEM.md` §5.3 applies to
  its own moments. What WOULD close this gap: a script (none exists yet) that,
  for every `i` with `τ_delist_i < ∞` in the panel, computes `g_i` = mean
  return over some trailing window ending at `τ_delist_i` and reports the
  sign/magnitude distribution — the natural home for it is
  `validate/survivorship_probe.py` (Section 11's target directory layout),
  which is not yet built. Until that script exists and is run, (A1)'s
  empirical half is a plausible, falsifiable, but UNVERIFIED assertion, and
  should be read as such rather than as a measured regularity. **If `g` is NOT
  monotone** — e.g. a mean-reverting statistic where names that delist did
  so AFTER a spike, or a re-listing event (Edge case 2, Section 9) that lets
  a name re-enter `R` under a new identity after a poor run under the old
  one — the DIRECTION of the bias in Section 1.3 is no longer guaranteed
  positive and must be re-derived case by case; the stopping-time
  construction of Section 1.2 remains bias-free regardless (it never
  depends on `g`'s monotonicity at all — this is exactly why Section 1.2's
  construction, not a corrected version of the `R`-style statistic, is the
  one this spec deploys).
- **(A2) Cross-sectional independence of delisting events is NOT assumed.**
  Section 1.3's derivation is a per-name statement (it conditions on a
  single `i`'s membership in `R` vs. `D`) and requires no independence
  assumption across `i, j` to hold — the inequality is proved name-by-name
  and then applied to the AGGREGATE statistic by linearity of expectation,
  which needs no independence, only that each per-name term is individually
  biased in the same direction. This matters concretely for crypto: delisting
  events are empirically CORRELATED (an exchange-wide de-risking event or a
  stablecoin depeg can trigger many simultaneous delistings — Edge case 8,
  Section 9), and a correlated-delisting regime would make the survivorship
  bias of a backward-roster construction WORSE, not better (more names
  simultaneously and non-independently excluded from `R`), which only
  strengthens Section 1.3's conclusion that the stopping-time construction
  must be used; it does not weaken it. UNIVERSE's own construction
  (Sections 1.2, 1.4) needs no cross-sectional independence assumption
  anywhere, by design — every eligibility test is evaluated per-instrument.
- **(A3) Staleness bound on Section 2.4's fallback.** The conservative
  fallback (missing/stale `DV_i(d)` → rank 0) is benign — i.e. it degrades
  gracefully to "exclude the name" rather than silently mis-ranking it —
  PROVIDED staleness is due to a genuine data or liquidity problem with
  instrument `i` specifically. The regularity condition this requires: the
  upstream data feed's staleness is not so pervasive, at any single `t`,
  that it strikes a large fraction of `U_t^base` simultaneously (which would
  look identical, under this rule, to a mass genuine liquidity collapse and
  could empty the eligible universe rather than merely trim it). This spec
  does not bound "how stale is too stale" for an INDIVIDUAL name's window
  with a numeric threshold beyond Section 2.4's `⌈W/2⌉`-valid-days rule; it
  DOES require (Edge case 8, Section 9) that a simultaneous mass-staleness
  or mass-delisting event is surfaced as an explicit diagnostic rather than
  silently shrinking `U_t`, precisely because (A3) is the condition under
  which the fallback's benignity assumption can break down.

These three conditions are regularity conditions on the THEOREMS of Sections
1.3/2.2, not additional gates UNIVERSE enforces at runtime — Sections 1.2,
1.4, and 2 are constructed to need none of them (they are per-instrument,
`F_t`-only rules by design), which is precisely why they are the ones
deployed rather than a "corrected" backward-roster construction.

---

## 2. Liquidity floor: trailing dollar-volume order statistic with hysteresis

### 2.1 The liquidity statistic

Define the **trailing dollar-volume** of instrument `i` as of day `d`, over a
window of `W` days (AMBER default `W = 30`, matching
`src/convexity/panel/panel.py::trailing_dollar_volume`'s `window: int = 30`):

```
DV_i(d) = mean_{s ∈ [d-W+1, d]} ( close_i(s) · volume_i(s) )     [only over s with i listed]
```

computed causally (only `s ≤ d`) and restricted to `i`'s live span (already
implemented, `trailing_dollar_volume`'s `finite = np.isfinite(block)` /
`np.nanmean` skip logic). `DV_i(d)` is undefined (`NaN`) if `i` is not listed
at `d`, per Section 1.4's base eligibility.

Because the target universe size is a fixed count (`N ≈ 50`, Section 4), the
liquidity floor is not an absolute dollar threshold but a **cross-sectional
order statistic**: at each rebalance, rank all base-eligible names by `DV_i(d)`
and admit by rank. Let `rank_i(d) ∈ (0, 1]` be `i`'s percentile rank of
`DV_i(d)` among `{j : j ∈ U_d^base}` (rank 1 = most liquid).

### 2.2 Why a bare threshold whipsaws, and the hysteresis fix

A hard rule "`i ∈ U_t` iff `rank_i(t) ≥ q`" (single threshold `q`) is unstable
at the boundary: a name whose trailing dollar-volume oscillates around the
`q`-th percentile enters and exits `U_t` every few rebalances as noise in
`DV_i` crosses the threshold both ways. Each membership flip forces PORTFOLIO
to open or unwind a position purely on a liquidity-rank tremor uncorrelated
with the momentum signal, paying the full round-trip friction
(`COST.md §5.1-5.4`'s salvaged fee+spread+impact decomposition, Section 5
below) for zero expected signal content. This is a real cost, not a
theoretical concern — it is functionally identical to the classic index
"reconstitution churn" problem.

**The hysteresis band.** Define two thresholds `q_in > q_out` (RED,
percentile ranks in `(0,1)`) and make membership **path-dependent**
(depends on `i`'s own membership state at `t^-`, not just today's rank):

```
i ∉ U_{t^-}^base-liquid  AND  rank_i(t) ≥ q_in    ⟹  i ENTERS  (i ∈ U_t^base-liquid)
i ∈ U_{t^-}^base-liquid  AND  rank_i(t) < q_out    ⟹  i EXITS   (i ∉ U_t^base-liquid)
otherwise                                          ⟹  membership UNCHANGED from t^-
```

A name must climb ABOVE `q_in` to earn entry but only falls out once it drops
BELOW the strictly lower `q_out`; inside the band `(q_out, q_in)` a name's
state is whatever it already was. This is a deterministic finite-state
machine per symbol (state ∈ {IN, OUT}, transition function above), so it is
still `F_t`-measurable (the state at `t` is a function of the state at `t^-`
— itself a function of `F_{t^-}` — and `rank_i(t) ∈ F_t`) and introduces no
lookahead; it only adds a one-step memory to the indicator process of Section
1.1.

### 2.3 Worked derivation — band width as an average-run-length calibration

The hysteresis band is not a free parameter to be fitted; it is derived from
the SAME closed-form logic used to calibrate a CUSUM control chart's
decision interval against a target average run length (Page 1954) — no
fitting, only a method-of-moments estimate of one noise parameter and a
classical formula relating band width to false-crossing rate.

**Notation convention used through the rest of this subsection (introduced
once here, applied consistently, identically to `docs/specs/SYSTEM.md` §5.3's
tagging discipline): every numeric input is tagged **[MEASURED]** (extracted
from a persisted run), **[ILLUSTRATIVE]** (constructed to make the worked
arithmetic concrete, not itself independently measured from a persisted run),
or **[DERIVED]** (an exact arithmetic consequence of tagged inputs). This tag
is placed at first use, not only in a trailing parenthetical.** **Status
update:** the hysteresis state machine itself is now implemented —
`src/convexity/universe/liquidity/hysteresis.py::hysteresis_step(rank_t,
state_prior, q_in, q_out)` is a complete function (verified by reading the
file), with `q_in`/`q_out` defaults (`0.80`/`0.65`) live in
`src/convexity/universe/config/defaults.py` — superseding Section 11's
original "not yet implemented" status for `q_in`/`q_out`. However, `σ_rank`
as a named symbol still does NOT exist anywhere in the codebase (verified:
no hits under `src/convexity/universe/`), because no run persisting
`std_t(Δ_i(t))` has been computed — so every numeric input in this
subsection remains necessarily [ILLUSTRATIVE] for that reason alone, not
because the state machine is unbuilt. The FUNCTIONAL FORM of the calibration
(`h_min` scales with `σ_rank·sqrt(n)`) is what this derivation establishes
and is reusable once real data exists; the specific `σ_rank ≈ 0.05–0.08`
figure below is a plausible order-of-magnitude placeholder chosen to make
the arithmetic concrete, not a number pulled from a persisted
`std_t(Δ_i(t))` computation.

**Setup.** Model `Δ_i(t) := rank_i(t) − rank_i(t-1)`, the week-over-week
change in instrument `i`'s percentile liquidity rank, as a mean-zero
increment with standard deviation `σ_rank` (a method-of-moments estimate:
`σ_rank = std_t(Δ_i(t))`, pooled across instruments and rebalances near the
band — this WOULD be a plain sample standard deviation, not a fitted model,
once computed from real data; today it is [ILLUSTRATIVE], per the tagging
convention above — no such computation is persisted anywhere in this repo).
Over
`n` consecutive weeks with NO genuine trend in `i`'s liquidity (the
"boundary noise" case Section 2.2 exists to suppress), the cumulative
unsigned drift of `rank_i` behaves like a mean-zero random walk with
per-step variance `σ_rank²`, so after `n` weeks the walk's typical
(one-standard-deviation) excursion is

```
Excursion(n) ≈ σ_rank * sqrt(n)                          (random-walk scaling)
```

**The false-crossing probability for a band of half-width `h`.** For a
symmetric band `[−h, +h]` around a stable rank (treating `q_in`, `q_out` as
approximately symmetric around the stable-membership rank for this
order-of-magnitude argument), the probability that pure noise crosses
BOTH thresholds within `n` weeks — a "whipsaw" — is bounded, by the
reflection principle for a mean-zero random walk (the same argument
underlying CUSUM's average-run-length calibration), by

```
P(whipsaw within n weeks) <= 2 * Phi( -h / (σ_rank * sqrt(n)) )      (reflection-principle bound)
```

where `Phi` is the standard normal CDF (a Gaussian increment approximation
to the rank-change distribution; adequate for an order-of-magnitude band
choice, not claimed exact). **Inverting for the band half-width** that
caps the `n`-week whipsaw probability at a target tolerance `α`:

```
h_min(n, α) = σ_rank * sqrt(n) * Phi^{-1}(1 - α/2)
```

This is the exact closed-form analogue of a CUSUM decision-interval
calibration against a target average run length (Page 1954): choosing `h`
large enough that noise-only crossings are rare over the horizon `n` you
care about suppressing churn on, in units of the noise scale `σ_rank`, is
the identical calibration logic, applied here to a two-threshold hysteresis
band instead of a one-sided CUSUM alarm.

**Applying the numbers.** `σ_rank ≈ 0.05`–`0.08` in percentile-rank units per
week **[ILLUSTRATIVE — no persisted `std_t(Δ_i(t))` run backs this number;
see the tagging convention above. The qualitative claim (a name's liquidity
rank moves a few percentile points week to week absent a real liquidity
regime change, rather than tens of points) is a plausible order-of-magnitude
judgment about a percentile-rank random walk over ~50 names, not an extracted
statistic]**. Targeting suppression of whipsaws over `n = 4` weeks (roughly a
month — long enough that a genuine liquidity regime change, not noise, should
be the dominant explanation for a persistent rank move) **[RED calibration
input]** at `α = 0.10` (10% tolerance for a spurious flip) **[RED calibration
input]** gives `Phi^{-1}(0.95) ≈ 1.645`, so, taking the midpoint
`σ_rank = 0.065` **[ILLUSTRATIVE, midpoint of the range above]**:

```
h_min(4, 0.10) ≈ 0.065 * sqrt(4) * 1.645 ≈ 0.065 * 2 * 1.645 ≈ 0.214    [DERIVED
                                                    from the ILLUSTRATIVE σ_rank above]
```

i.e. a full band width `q_in − q_out ≈ 2·h_min ≈ 0.43` would be needed to
suppress essentially ALL noise-driven whipsaws over a month at this
tolerance — wider than is operationally useful (it would also suppress many
genuine liquidity-driven transitions). The RED default actually deployed,
`q_in = 0.80`, `q_out = 0.65` (band width `0.15`), is DELIBERATELY narrower
than the fully-conservative `h_min` bound — it accepts a materially higher
residual whipsaw probability than `α = 0.10` over a month
(`P(whipsaw) ≈ 2·Phi(−0.15 / (0.065·2)) ≈ 2·Phi(−1.15) ≈ 0.25`, roughly a
1-in-4 chance of a boundary tremor within any given month for a name
oscillating right at the edge) in exchange for NOT delaying genuine
liquidity-driven entries/exits for a full month, matching Petajisto
(2011)'s empirical finding that index-committee buffer zones are
similarly a few-week, not multi-month, compromise between churn suppression
and responsiveness. **This is a BLUE-searchable width (Section 8 parameter
governance, `BLUE` grid on the band), never a fitted-to-maximize-Sharpe
parameter** — the derivation above fixes the FUNCTIONAL FORM (`h_min` scales
with `σ_rank·sqrt(n)`) and the qualitative direction of the trade-off; the
specific operating point on that trade-off (accepting ~25%/month residual
whipsaw risk rather than the ~10% the fully conservative bound would target)
is a governance choice validated by the SAME CPCV/DSR gates as any other
BLUE grid member, not a claim that `0.15` is itself derived to be optimal.

**SENSITIVITY (how much does the "accept ~25%/month residual whipsaw"
conclusion depend on the illustrative `σ_rank = 0.065` midpoint?).** Re-running
the SAME `P(whipsaw) ≈ 2·Phi(−0.15/(σ_rank·2))` arithmetic at the two ends of
the stated `σ_rank ∈ [0.05, 0.08]` range, holding the deployed band width
(`h = 0.075`, i.e. `q_in − q_out = 0.15`) fixed:

```
σ_rank = 0.05 (low end):  P(whipsaw) ≈ 2·Phi(−0.15/(0.05·2)) = 2·Phi(−1.50) ≈ 2·0.0668 ≈ 0.134
                           (~13%/month — a name whose liquidity rank moves LESS week to week
                           whipsaws LESS often at a fixed band width, as expected)
σ_rank = 0.065 (midpoint): P(whipsaw) ≈ 0.25   (~25%/month, the figure carried in the main text)
σ_rank = 0.08 (high end):  P(whipsaw) ≈ 2·Phi(−0.15/(0.08·2)) = 2·Phi(−0.9375) ≈ 2·0.1743 ≈ 0.349
                           (~35%/month — noisier names whipsaw close to a third of months)
```

Across the entire stated illustrative range, the residual whipsaw probability
at the deployed `0.15` band width spans roughly **13%–35% per month**, not a
single point estimate of exactly 25% — the qualitative conclusion of Section
2.3 ("the deployed band is deliberately narrower than the fully-conservative
bound, accepting materially higher residual whipsaw than `α=0.10` in exchange
for responsiveness") is UNCHANGED across this whole band: even at the
low-noise end (`σ_rank=0.05`), the deployed width still accepts roughly 13%
monthly whipsaw risk, well above the `10%` conservative target, and at the
high-noise end (`σ_rank=0.08`) the case for NOT widening the band further
(to avoid delaying genuine transitions) is if anything stronger, since a
wider band at high `σ_rank` would need to grow substantially to meaningfully
cut whipsaw risk. **What WOULD change this conclusion:** if a future
persisted `std_t(Δ_i(t))` run (Section 11's not-yet-built hysteresis module)
measures `σ_rank` an order of magnitude away from this illustrative
`0.05–0.08` band — e.g. `σ_rank > 0.20`, meaning liquidity rank itself is
extremely noisy week to week — the deployed `0.15` band width would sit
inside, not outside, the noise scale (`h/σ_rank < 1`), and the "accept
~25%/month" framing would understate the true whipsaw rate materially; that
scenario is not ruled out by anything in this derivation and is exactly the
kind of check `validate/survivorship_probe.py`-adjacent tooling (Section 11)
should run once real `DV_i(d)` history is wired through the hysteresis state
machine, before the RED `q_in`/`q_out` defaults are treated as final rather
than a first BLUE-grid candidate.

> **Invariant HYST — membership is path-dependent, monotone, and F_t-measurable.**
> Entry requires crossing the HIGHER threshold `q_in`; exit requires crossing
> the LOWER threshold `q_out < q_in`; membership inside the band holds the
> prior state. Test: a synthetic `rank_i(t)` series oscillating within
> `(q_out, q_in)` produces ZERO membership flips over the whole series,
> whichever state it started in; a series that visits `q_in` then only ever
> decays to exactly `q_out + ε` (never below) never exits.

### 2.4 Missing/stale liquidity data — conservative fallback (P6)

If `DV_i(d)` cannot be computed (missing volume data, a stale/zero-volume
window, or fewer than `⌈W/2⌉` valid days in the trailing window), `i` is
treated as **rank 0** (least liquid) for that rebalance — never imputed to a
neutral or average rank. A name with unreliable liquidity data can only be
excluded by this rule, never admitted by it (P6, salvaged from COST.md
verbatim: "a gate that cannot price a trade fails the trade closed, it does
not pass it free" — here, "a name that cannot be liquidity-ranked fails the
name closed, never admits it under an assumed rank").

---

## 3. Universe size N — the liquidity ranking used for the size decision

### 3.1 The empirical shape: peaked, not monotone

Universe size is a **separate decision** from the liquidity floor of Section 2
— the floor decides WHICH names clear a bar, size decides HOW MANY of the
liquidity-ranked names to keep. The empirical relationship between net Sharpe
and `N` is **peaked, not monotone increasing**: net Sharpe peaks empirically
around `N ≈ 60–75` (`docs/EDGE.md` §(c): "net edge peaks around 60-75 names"
[MEASURED]), and the peak's improvement over `N = 50` is **inside the
block-bootstrap confidence interval** — i.e. not statistically distinguishable
from noise. The bootstrap machinery itself is real and precisely locatable:
`convexity.system.validate.robustness.block_bootstrap_ci` (Kunsch 1989
moving-block bootstrap, block length auto-selected per Politis & White 2004
— **not** `src/convexity/cost/friction/validate/stats.py`, which does not exist; this is the same
correction `docs/specs/LEVERAGE.md` §2.6 already makes explicit for the
identical function). `docs/specs/PORTFOLIO.md` §9 independently cites the
same finding in the same words ("The gain of `N ≈ 60–75` over `N = 50` is
inside the block-bootstrap confidence interval") and is the cross-reference
of record for this claim.

**What this spec does NOT have, and does not invent.** Neither `docs/EDGE.md`
nor `docs/specs/PORTFOLIO.md` §9 records the actual Sharpe point estimates at
`N=50` vs. `N≈65` or the CI half-width itself — both documents state the
CI-overlap CONCLUSION ("not statistically distinguishable from noise") without
publishing the underlying `(ci_low, ci_high)` pair or the specific Sharpe
delta. This spec could not honestly tighten this claim beyond citing the
conclusion and its exact source, without fabricating numbers that exist in
neither `docs/EDGE.md` nor any persisted validation run this repo retains. If
a future `src/convexity/system/xsec_validate.py` run persists the actual
`block_bootstrap_ci` output for the `N`-sweep (the function already exists and
is already called correctly elsewhere in this repo, per LEVERAGE.md §2.6 —
only its output for THIS specific sweep is not currently quoted in any spec
or evidence doc), the numbers belong here and in PORTFOLIO.md §9 by reference,
replacing this qualitative citation with the real `(ci_low, ci_high, Δ)`
triple. Because
the gain from `N=50 → N≈65` is not a real, defensible edge, and because a
smaller universe is simpler to operate, monitor, and less exposed to tail
illiquidity, **`N ≈ 50` is the deployed choice** — a "prefer simplicity when
CIs overlap" decision, not a claim that 50 is a discovered optimum.

Expanding further, to `N = 100`, does **not** robustly beat `N = 50` either,
and for a specific, identifiable reason (not just "more noise"):

1. **Illiquid-tail cost.** The additional 50 names (ranked 51–100 by
   liquidity) sit further down the trailing-dollar-volume order statistic
   (Section 2.1) and carry materially wider realistic Amihud-illiquidity
   impact cost (COST.md §5.4's square-root impact form, `impact_frac = k_i ·
   σ_i · sqrt(participation)`, evaluated at these names' much smaller `V_hat`)
   — measured drag ≈ **+2.9%/yr** from the added tail.
2. **Short-availability bite.** Of the 50 added names, roughly **15 are
   effectively unshortable** in practice (Section 4 formalizes this gate).
   Part of the small apparent Sharpe gain at `N=100` in a backtest that does
   NOT model this friction leans on shorting names that could not actually be
   shorted live — i.e. the light-cost/unconstrained-short backtest overstates
   the `N=100` case specifically, not uniformly across `N`.

The honest `N=50`-vs-`N=100` comparison, done net of both frictions, shows
**overlapping confidence intervals** — expanding the universe is not a real,
deployable edge; it is backtest noise plus an uncorrected short-availability
assumption.

### 3.2 The liquidity ranking used for the size cutoff

Given the hysteresis-filtered eligible set `U_t^base-liquid` (Section 2), the
size-`N` universe is:

```
U_t = top-N(U_t^base-liquid, key = rank_i(t))       (N = 50, RED)
```

i.e. among names that already cleared the entry/exit hysteresis band, keep
exactly the `N` most liquid by the SAME trailing-dollar-volume order statistic
`rank_i(t)` from Section 2.1 — one liquidity ranking, used first as a
pass/fail hysteresis gate and then as a cardinality cutoff, never two
divergent liquidity notions. If fewer than `N` names clear the hysteresis
gate (early in the panel's history — see `panel.py::universe_size_by_year`,
which reports the universe growing from "a handful of majors" to the full
target over time, BY DESIGN, not as a bug), `U_t` is simply the whole
hysteresis-eligible set; `U_t` is never backfilled to a fixed count with
ineligible names.

> **Invariant SIZE — N is a RED cardinality cap on an already-liquidity-ranked
> set, never a separate liquidity model.** Changing `N` must not change which
> STATISTIC ranks names, only how many of the top-ranked survive. Test: for
> `N_1 < N_2`, `U_t(N_1) ⊆ U_t(N_2)` whenever both are computed from the same
> `rank_i(t)` and the same hysteresis-filtered base set (monotone nesting).

### 3.3 Capacity is a separate dial, not a default

A liquidity-SHARE cap within each leg (limiting any one name to at most `C ×`
its dollar-volume share of its leg, `src/convexity/portfolio/portfolio.py::_leg_liq_cap`)
was tested and **loses to plain inverse-realized-vol weighting** net of cost.
The reason is structural, not a tuning miss: in this universe, crypto majors
are SIMULTANEOUSLY low-vol and high-liquidity, so a liquidity-share cap
paradoxically RAISES the majors' weight ceiling (they already have the
largest liquidity share, so capping relative to liquidity share barely binds
on them while it redistributes weight among the tail) — effective breadth
(inverse-Herfindahl on absolute per-name contribution,
`src/convexity/panel/sleeve.py::effective_breadth`) FALLS under capping
(measured ≈40 → 28-34), the opposite of the diversification a cap is meant to
buy, at a cost of ≈0.4–0.6 Sharpe. Capping only buys deployable CAPACITY
(median deployable book ≈$3M uncapped → ≈$12-23M at cap tightness `C=2-3`).
**Consequence for this spec:** UNIVERSE's job is the eligibility gate and the
liquidity RANK (Sections 2-4); it does NOT impose a within-leg liquidity cap
by default. A liquidity-share cap is a capacity dial PORTFOLIO may apply
explicitly above roughly a $3M book, never a default UNIVERSE behavior, and
it is orthogonal to (post-dates) UNIVERSE's `U_t` / `U_t^short` gates.

---

## 4. Shortability — a separate, stricter gate on the short leg only

### 4.1 Why long-eligibility is not short-eligibility

`U_t` (Sections 1-3) answers "can this name be RANKED and, if it lands in
either tail, TRADED at all." It is symmetric by construction — the same test
admits a name to the long-candidate pool and the short-candidate pool. But
being tradeable long and being tradeable **short** are genuinely different
operational conditions on a perpetual-futures venue:

- A long-only constraint never binds on Binance USDT-M perps (any listed,
  liquid contract can be bought).
- A short requires the venue to actually let you open/hold a short (no
  isolated-margin restriction, no reduce-only lockout, no minimum-notional or
  leverage-tier gate that effectively excludes small books), AND requires
  realistic BORROW/FUNDING conditions that do not make the short economically
  non-viable even where technically permitted (e.g. some newly-listed or
  thin-float perps trade with structurally elevated or capped funding that
  makes a persistent short unfavorable independent of the momentum signal).

Define a second, STRICTER indicator:

```
1{ i ∈ U_t^short }    with    U_t^short ⊆ U_t
```

`U_t^short` is a SUBSET, never a superset or an independent set — a name
cannot be short-eligible without first being long/base-eligible (the
short-specific checks are additional necessary conditions layered on top of
Section 1-3's base gate, not a parallel gate that could admit a name
Section 1-3 excluded).

### 4.2 Formal gate

```
1{ i ∈ U_t^short } = 1{ i ∈ U_t } · 1{ short_permitted_i(t) } · 1{ short_liquid_i(t) }
```

- `short_permitted_i(t)`: an `F_t`-measurable AMBER metadata flag — venue
  contract status supports shorting at `t` with no active restriction
  (this is metadata, not a statistic; it changes rarely and is sourced from
  venue contract state, not fitted).
- `short_liquid_i(t)`: a STRICTER liquidity order-statistic test than the
  long-side floor of Section 2 — e.g. requiring `rank_i(t)` to clear a HIGHER
  percentile than `q_in`, or requiring a minimum absolute trailing dollar
  volume floor independent of the cross-sectional rank (so that even in a
  liquidity-poor cross-section, an absolute floor still binds). The STRUCTURE
  — strictly stricter than the long floor, evaluated on the same trailing
  dollar-volume statistic — is the invariant this spec fixes today.

  **DISCLOSED LIMITATION, stated starkly rather than left to read as merely
  "not yet decided."** Unlike every RED constant in `docs/specs/LEVERAGE.md`
  §3 (each of which carries an illustrative v1 NUMBER and an explicit
  anchoring derivation — e.g. `κ=0.02` anchored to the measured ~$3M
  deployable-book figure, `h_vol=8` weeks anchored to the CPCV embargo/group
  geometry), **this spec gives NO numeric value for `short_liquid_i(t)`'s
  threshold, and states explicitly that it MUST NOT be assigned one yet.**
  The reason is not an oversight: every other RED constant in this file
  (`W`, `q_in`, `q_out`, `N`) is anchored to a statistic this repo actually
  computes today (`DV_i(d)` flows through `src/convexity/panel/panel.py` on every
  panel build), so a numeric default can be chosen and defended even before
  the surrounding machinery (hysteresis, Section 2.2) is wired up end to end.
  The short-side threshold is different in kind: this repo has **no
  short-borrow-availability data, no historical short-liquidity time series,
  and no record of which Binance USDT-M perps have ever had a short
  position rejected, restricted, or economically forced closed for
  borrow/funding reasons** — inventing a plausible-sounding number (e.g.
  "require `rank_i(t) ≥ 0.90`") here would be indistinguishable, to a reader,
  from the anchored RED constants elsewhere in this file, and would silently
  smuggle a fabricated empirical claim into a document whose central
  discipline is that every number is either measured, illustrative-and-
  labeled, or derived. **The honest state of this parameter is: an open
  specification parameter pending real short-borrow/short-liquidity data,
  not a tuning decision merely deferred.** Section 11's directory layout
  names `shortability/liquidity_gate.py` as its target implementation file;
  the build-contract for that file is: accept a `short_liquid_i(t)` callable
  injected at construction time (never hard-code a threshold inside
  `universe_api.py`), so that the day real borrow/short-liquidity data
  exists, calibrating this gate is a matter of supplying a new callable and
  a governance-logged RED promotion (Section 8's BLUE→RED event pattern),
  not a code change to the gate's call site.

### 4.3 Why this gate matters empirically (grounding, not a hypothetical)

This is not a theoretical nicety: the `N=50 → N=100` universe-expansion
result (Section 3.1) is DIRECTLY explained by skipping this gate. Roughly 15
of the 50 added names at `N=100` are effectively unshortable in practice, and
part of the small apparent Sharpe improvement at `N=100` in a backtest that
does not enforce `U_t^short` leans on shorting exactly those unshortable
names — i.e. the backtest is silently assuming a short that could not be
executed live. Enforcing `U_t^short` as a real constraint (forcing the
short-leg quantile selection of `portfolio.py::quantile_weights` to draw its
`short_idx` only from `U_t^short`, never from the full `U_t`) is what converts
the light-cost, unconstrained-short `N=100` headline into the honest,
CI-overlapping-with-`N=50` result cited in Section 3.1. **A universe-size
decision made without this gate is not a decision about universe size — it is
partly a decision about how much illegal shorting to assume.**

### 4.4 Asymmetric consequence when the short leg is starved

If, at some `t`, fewer than the quantile leg-count `k` (Section 3 of
`portfolio.py::_leg_count`) names in the bottom-ranked tail are members of
`U_t^short`, the short leg cannot be filled at its designed size from the
intended candidates. UNIVERSE's contract is to EXPOSE this
(`n_short_eligible_in_tail < k` as a flagged diagnostic on every rebalance),
never to silently widen the short-eligible set or silently shrink the long
leg to compensate — that compensating decision belongs to PORTFOLIO/SIGNAL,
which may choose to (a) widen the short candidate pool further down the
rank (still requiring `U_t^short` membership), (b) reduce the short leg's
gross to match available shortable names (breaking the leg's dollar-neutral
match — an explicit PORTFOLIO policy, out of scope here), or (c) hold the
mismatch and accept a temporarily non-neutral book. UNIVERSE only reports the
constraint; it does not resolve it (P2, provider-not-decider).

> **Invariant SHORT — the short gate is a strict subset test, never
> independent, never silently relaxed.** `U_t^short ⊆ U_t` at every `t`
> (test: `U_t^short \ U_t = ∅` always). A rebalance where the bottom quantile
> tail is short-starved emits a diagnostic flag and does NOT fall back to
> drawing short candidates from outside `U_t^short`.

---

## 5. Cost context inherited from COST.md (salvaged, re-scoped)

UNIVERSE does not compute cost — that is `src/convexity/cost/accounting.py`'s
`round_trip_friction` — but the liquidity gates above exist BECAUSE realistic
cost is liquidity-dependent, so the additive decomposition COST.md defined
(salvaged) is the reason Sections 2-4's gates are shaped the way they are:

```
C_round_trip = 2 · (taker_fee_bps + half_spread_bps + impact_bps)     [both legs, COST.md §3 P3, re-scoped to weekly]
```

with the REAL numbers this repo has validated (never invented):

- `taker_fee_bps = 5.0` bps — the real Binance USDT-M VIP0 taker schedule,
  and the DOMINANT term of the three (`src/convexity/panel/sleeve.py::CRYPTO_FRICTION`,
  `taker=5.0e-4`).
- `half_spread_bps` — proxied by a dollar-volume-anchored `~1/sqrt(ADV)` curve,
  anchored so the median liquid name pays ≈1 bp — this is exactly what
  Section 2's `DV_i(d)` liquidity statistic exists to make tractable: the
  spread/impact estimate is a function of the SAME trailing dollar-volume
  order statistic used for the eligibility rank, so a name UNIVERSE excludes
  for thin liquidity is, not coincidentally, also a name whose cost model
  would otherwise be least reliable.
- `impact_bps` — measured via a REAL Amihud (1985/2002) illiquidity estimator.
  **Cautionary note, preserved from the investigation record:** a
  Corwin-Schultz (2012) high/low spread estimator was tried on 1-minute-
  aggregated daily H/L bars and found to CONFLATE volatility with spread,
  producing a garbage ≈135 bp / ≈40%-drag estimate — a documented dead end.
  UNIVERSE's liquidity ranking deliberately uses a dollar-VOLUME statistic
  (Section 2.1), not a high/low range statistic, for exactly this reason:
  volume-based liquidity proxies do not have the Corwin-Schultz
  volatility-conflation failure mode.
- Funding is charged by real wall-clock 8h stamps (≈3/day), signed per leg
  (COST.md §5.5's sign convention, P7: a long paying positive funding bleeds,
  a short in the same regime earns), and is confirmed a MINOR cost component
  (≈3-5% of gross) relative to friction (≈10× larger) — i.e. the liquidity
  gates of this spec are overwhelmingly about controlling fee+spread+impact,
  not funding.

Round-trip friction is charged on BOTH legs at weekly rebalance cadence
(≈52×/yr), not thousands of times per year the way the dead directional
system paid it intraday — this is why the same ≈13 bp round-trip that killed
the directional book (paid thousands of times) barely bites the weekly
cross-sectional book (paid ≈52 times): the risk-unit change from "one trade"
to "one week's rebalance" is what makes the identical cost structure
survivable, not a lower cost estimate.

---

## 6. The negative worked example — survivorship-flattered equity roster

This repo already produced, and explicitly flagged, an instance of the
failure mode Section 1.3 proves is biased — the exact quote lives in code,
not in the research memo, and is cited from its real location:
`src/convexity/panel/sleeve.py`'s sleeve-config docstring (line 14) documents the
equities sleeve (30 names) as **"SELECTED for modern liquidity (a
survivorship caveat the report must surface)"**, and the companion memo
`src/convexity/panel/UNIVERSE.md` independently describes the same roster's
construction as "mega-cap, factor-diversified" names — i.e. a roster built by
looking at TODAY's mega-cap survivors and applying that fixed list across the
whole 5-year backtest window. This is the `U_t := R` construction of Section
1.3 (backward-looking, `F_T`-measurable, not `F_t`-measurable), not the
stopping-time construction of Section 1.2 — the code's own docstring is
flagging Section 1.3's exact bias mechanism against itself, for the equities
sleeve specifically, before this spec ever formalized why it is a problem.

**What is and is not empirically measured here.** EDGE.md's tested
conclusion is that cross-sectional momentum "does NOT extend to equities or
commodities (tested); those sleeves were directional-grade only" — i.e. the
equities sleeve carries no measured cross-sectional-momentum edge, full
stop. This spec does NOT additionally claim a specific effective-breadth
number for the equities roster: no such number is recorded anywhere in this
repo's research notes for the equities sleeve, and inventing one here would
repeat, inside a document about survivorship bias, the exact discipline
violation (an ungrounded, uncited empirical claim) this repo's evidence-
grounding standard forbids. What IS a real, traceable measurement using the
SAME `effective_breadth` formula (Section 3.3, `src/convexity/panel/sleeve.py::
effective_breadth`, inverse-Herfindahl on per-name contribution) is the
liquidity-cap finding of Section 3.3 itself: breadth measured **≈40 → 28–34**
under a liquidity-share cap on the CRYPTO sleeve (PORTFOLIO.md §5.3's
independently-cited number, `src/convexity/panel/assetclass.py::effective_breadth`) —
cited there, correctly, as a within-crypto weighting artifact, and NOT
repeated here as if it were an equities-roster measurement; the two findings
share a formula, not a dataset, and must not be conflated.

**The lesson, stated at the level this repo can actually support.** A roster
hand-picked from today's liquid/large-cap survivors is not "a smaller but
still valid universe" — per Section 1.3's mechanism, it is systematically
enriched for names whose realized history looks good, which (a) inflates any
in-sample statistic computed on it, independent of whether that inflation
was separately quantified via an effective-breadth number for THIS sleeve,
and (b) is the reason `src/convexity/panel/sleeve.py`'s own docstring flags it as
a caveat the report "must surface" rather than treating it as settled. The
crypto sleeve is exempt from this specific caveat because it is NOT built the
same way: `src/convexity/panel/UNIVERSE.md`'s crypto symbol table is a fixed,
hand-curated 50-name candidate LIST (curated once, for sector/market-cap
diversity — see `src/convexity/panel/UNIVERSE.md`'s "Crypto (50) — by sector" table —
NOT for having survived or performed well, and explicitly excluding
"low-mcap pump coins" by sector representativeness, not by realized
return), while `src/convexity/panel/sleeve.py`'s equities docstring documents no
equivalent point-in-time listing/delisting construction for the equities
sleeve's ROSTER SELECTION itself (as distinct from the panel's own
`listed` mask, which IS point-in-time for whichever names are on the list).
The critical distinction, restated precisely: the crypto LIST is a superset
of ever-possibly-eligible tickers, and membership in `U_t` on any given
historical date is still governed entirely by the point-in-time
`τ_list_i`/`τ_delist_i`/liquidity-rank machinery of Sections 1-3, never by
the list's mere existence — whereas the equities sleeve's 30-name LIST
ITSELF is documented as selected using modern (survivorship-contaminated)
liquidity, a defect at the candidate-list layer that no amount of
point-in-time correctness inside `U_t`'s construction (Sections 1-4) can
repair, because Sections 1-4 only ever refine a candidate list, they cannot
retroactively fix a candidate list that was drawn non-causally in the first
place. A name on the crypto candidate list that had not yet listed at `t`,
or had already been effectively delisted (exchange-halted, liquidity
collapsed to zero) by `t`, is excluded at `t` regardless of its presence on
the list.

> **Invariant CANDIDATE-VS-ELIGIBLE — the symbol list is a superset, never
> the eligibility answer.** `src/convexity/panel/UNIVERSE.md`'s per-sleeve symbol tables
> are the maximum CANDIDATE pool (curated for sector/factor diversity, fixed
> once); `U_t` is always a `t`-dependent subset of that pool computed from
> Sections 1-4, never the pool itself. Test: a candidate-list symbol with
> `τ_list_i > t` or `τ_delist_i ≤ t` is excluded from `U_t` at every such `t`,
> even though it remains permanently present in the candidate list.

---

## 7. Validation — what a universe-construction bug looks like empirically

UNIVERSE itself is not "validated" by a Sharpe ratio (it emits no returns);
it is validated by checking that the DOWNSTREAM pipeline's honest numbers
match what a correct point-in-time construction should produce, and by
structural invariant tests (Section 8) that need no backtest at all. The
downstream numbers this spec's construction is responsible for keeping
honest, cited exactly (never invented):

- **CPCV** (Lopez de Prado 2018 combinatorial purged cross-validation; `N=8`
  groups, `k=2` test-groups, 28 paths, 1-week embargo — implemented in
  `src/convexity/system/cpcv.py`, reusing `convexity.system.validate.cpcv.SystemCPCV`):
  mean OOS path Sharpe ≈ **0.91** vs the naive single-path Sharpe ≈ **0.98** —
  a haircut of only ≈0.07, i.e. **no OOS collapse**. This is the direct
  contrast to what a survivorship-biased universe would show: Section 1.3's
  mechanism predicts a survivorship-inflated backtest looks internally
  consistent across folds (the bias is baked into every fold identically), so
  the absence-of-collapse here is necessary-but-not-sufficient evidence the
  construction is honest — it must be paired with the structural
  point-in-time proof of Section 1, not read alone as exoneration.
- **`frac_pos`** (fraction of CPCV paths with positive Sharpe) = **0.85-1.00**
  — the ROBUST statistic promotion criteria should weight. The 5th-percentile
  path Sharpe is explicitly PARTITION-FRAGILE (swings +0.42 to −0.59 depending
  on the `N`/`k` CPCV split choice) and must NOT be the primary promotion
  gate.
- **PBO** (probability of backtest overfitting, Lopez de Prado, Bailey,
  Borwein & Zhu 2014's logit-of-relative-rank estimator, implemented in
  `src/convexity/system/cpcv.py::PBOResult`) over the SIGNAL lookback family = **0.31
  to 0.47** across partitions — the best-in-sample lookback was `L=12` on
  21/28 CPCV splits, never fully clean (`<0.2`), which is SIGNAL's problem to
  solve by deploying the 12-16 day band rather than a single fitted `L`
  (SIGNAL.md), but it is cited here because a universe-construction leak
  (e.g. survivorship, or a hysteresis band tuned on the same data the PBO
  estimate is computed from) would inflate this number too, and the fact that
  it lands in a moderate — not pristine, not catastrophic — range is
  consistent with a genuinely point-in-time construction plus a real but
  modest lookback-selection fragility, not a construction-level leak (a leak
  would be expected to show as an anomalously LOW, too-good PBO).
- **DSR** (Bailey & Lopez de Prado 2014 deflated Sharpe, reusing
  `convexity.system.validate.deflated.DeflatedSharpe`) fails ONLY on
  insufficient track length: PSR ≈ **0.988** (high) against a MinTRL that
  exceeds the available ≈5-year / 258-weekly-observation sample once the
  multi-testing-inflated trial count is paid for — a track-RECORD-length
  problem, not a deflation rejection. This is orthogonal to universe
  construction (it is a statement about sample size, not about which names
  were in the sample) but is recorded here because a badly-constructed
  universe (survivorship, or a shrinking/whipsawing `N`) would ALSO reduce
  effective `n_obs` by introducing non-stationarity across the sample —
  Section 2's hysteresis band is partly what keeps `n_obs` honest by
  preventing membership churn from injecting spurious regime breaks into the
  weekly return series.

**Explicit warning against circular validation.** None of the numbers above
can certify Section 1's point-in-time property; they can only be CONSISTENT
or INCONSISTENT with it. The only way to check Section 1's property directly
is the structural test of Invariant PIT (Section 8): does the eligibility
computation at `t`, when given a dataset with all `s > t` rows corrupted or
deleted, produce a bit-identical `U_t`? A statistical validation battery
cannot substitute for this because — per Section 1.3's corollary — a
survivorship leak is invisible to every fold-based check by construction.

---

## 8. Parameter governance

Every UNIVERSE parameter is exactly one bucket, per the RED/AMBER/BLUE/TEAL
discipline this repo's `pyproject.toml` and sibling specs already enforce.

- **AMBER — supplied metadata, per instrument.** `τ_list_i`, `τ_delist_i`
  (derived from observed data, but per-instrument and not tunable);
  `short_permitted_i(t)` (venue contract-status flag); the candidate symbol
  list itself (`src/convexity/panel/UNIVERSE.md`'s curated per-sleeve tables — fixed
  once for sector/factor diversity, never re-tuned on realized P&L).
- **RED — shared, class-level constants.** The trailing dollar-volume window
  `W` (default 30 days); the hysteresis thresholds `q_in`, `q_out`
  (Section 2.3); the universe size `N` (default 50, Section 3); the minimum
  valid-day fraction for the liquidity statistic (Section 2.4); `w_warmup`
  (Section 1.4, defined to equal `W`, so it moves only when `W` does — not
  an independently-governed constant). The short-eligibility strictness
  parameter of Section 4.2 (`short_liquid_i(t)`'s threshold) is a RED
  constant BY BUCKET (it is shared, class-level, and must never become
  per-instrument config) but, unlike every other RED constant in this list,
  has **no assigned numeric value yet** — Section 4.2's disclosed limitation
  is the authoritative statement of why, and assigning it a value before
  real short-borrow/short-liquidity data exists would be the exact
  fabrication this repo's evidence-grounding standard forbids. RED is
  unreachable from per-instrument (AMBER) config — a per-symbol `q_in` or
  per-symbol `N` is the bug (Invariant C, inherited verbatim from
  COST.md/SYSTEM.md's identical governance rule).
- **BLUE — finite, validated grids, class-level only.** `N ∈ {50, 60, 75,
  100}` (the sweep that produced Section 3.1's peaked-shape finding); the
  hysteresis band width; `W ∈ {20, 30, 45}` days — each a small enumerated
  candidate set, evaluated under the SAME CPCV/DSR gates as SIGNAL's lookback
  band (Section 7), never a continuous fitted optimum, never per-symbol.
- **TEAL — none.** UNIVERSE fits nothing per-instrument at runtime; every
  input is either observed metadata (AMBER) or a shared constant (RED). This
  is a deliberate simplification versus COST.md/SYSTEM.md (which both have
  TEAL layers) — UNIVERSE's job is structural gating, not estimation, and the
  NO-ML / closed-form-only doctrine (this repo's `pyproject.toml`: "closed-
  form (NOT ML)") is satisfied here trivially because there is no fitted
  object at all, only order statistics and threshold crossings.

> **Invariant C — RED unreachable from per-instrument config** (inherited
> verbatim from COST.md/SYSTEM.md). `q_in`, `q_out`, `N`, `W`, or any
> short-eligibility RED constant appearing as a per-symbol field is the bug.

---

## 9. Edge cases and invariants (consolidated)

1. **A name delists mid-week, between rebalances.** `τ_delist_i` is only
   evaluated AT rebalance timestamps in this weekly-cadence spec (there is no
   intra-week eligibility check by design — the book is only rebalanced
   Friday-to-Friday). If `i` was `U_t`-eligible at rebalance `t` and
   delists before `t+1`, the position is held into whatever ACCOUNTING's
   delist-handling rule specifies (out of UNIVERSE's scope — this spec only
   governs the entry gate at each rebalance, and Section 1.2's `τ_delist_i`
   test simply ensures `i` is excluded from `U_{t+1}` onward).
2. **A name re-lists after a delisting (rare, e.g. a rebrand/relaunch under
   the same ticker).** `τ_list_i`/`τ_delist_i` are defined as the FIRST
   `inf{}` crossing in Section 1.2; a rule that must handle re-listing
   treats it as a NEW instrument identity (a fresh `τ_list_i'`), never as a
   resurrection of the old stopping times — merging the two would let the
   old instrument's pre-delist history count toward the new instrument's
   `τ_history_i` warmup (Section 1.4), which is lookahead-adjacent (borrowing
   informational credibility from a different, defunct listing).
3. **Fewer than `N` names clear the hysteresis gate (early history).**
   `U_t` is the whole hysteresis-eligible set, never backfilled — see Section
   3.2. This is BY DESIGN (`panel.py::universe_size_by_year`'s documented
   growth-from-a-handful-of-majors shape is the tell that the construction is
   genuinely point-in-time, not a bug to fix).
4. **All names in the bottom rank tail are short-ineligible.** The short leg
   is short-starved (Section 4.4); UNIVERSE flags it and does not resolve it.
5. **A name's liquidity data is stale or missing for an entire trailing
   window `W`.** Falls to rank 0 (Section 2.4), never imputed — this can
   push a previously-IN name below `q_out` and trigger a hysteresis exit
   exactly as a genuine liquidity collapse would (the fallback and a real
   collapse are, correctly, indistinguishable to the gate — P6 conservatism).
6. **The candidate symbol list itself changes (a new coin is added to
   `src/convexity/panel/UNIVERSE.md`'s curated table).** This is an AMBER, human,
   infrequent edit to the candidate superset (Section 6's
   Invariant CANDIDATE-VS-ELIGIBLE) — it must not be re-tuned in response to
   realized P&L on any specific name (that would reintroduce exactly the
   survivorship selection Section 1.3 proves is biased, just moved one layer
   up into which names are even considered).
7. **Two instruments share a stopping time exactly (`τ_list_i = τ_list_j`).**
   No tie-break is needed at the eligibility layer — Section 1's indicator is
   independent per instrument; SIGNAL's percentile-rank tie-averaging
   (`src/convexity/signal/signals.py::_pct_rank`'s documented average-rank-for-ties)
   handles coincident RANKS downstream, which is a separate concern from
   coincident LISTING dates.
8. **Correlated / mass delisting event (a liquidity-crisis scenario).** Unlike
   Edge case 1's single-name mid-week delist, a realistic crypto tail event
   (an exchange-wide de-risking action, a stablecoin depeg cascading into
   correlated delistings/liquidity collapses, or a venue-level outage) can
   push MANY names across `q_out` in the SAME rebalance. Section 1.5's
   assumption (A2) already establishes that Section 1.3's per-name bias proof
   needs no cross-sectional independence — so the point-in-time construction
   remains correct name-by-name even here — but the OPERATIONAL consequence
   is distinct from a single-name exit: a mass hysteresis exit in one week
   can shrink `U_t` sharply, potentially interacting with Section 4.4's
   short-starvation diagnostic simultaneously (a liquidity crisis plausibly
   hits shortability at the same time it hits the liquidity floor) and with
   Section 1.5 (A3)'s staleness-benignity condition (if the crisis manifests
   as a venue data outage rather than a genuine volume collapse, Section
   2.4's fallback and a real liquidity collapse become observationally
   identical, exactly as (A3) anticipates). UNIVERSE's contract: emit the
   SAME per-name hysteresis-exit and short-starvation diagnostics it would
   for any ordinary week (P2, provider-not-decider — there is no special-case
   "crisis mode" branch), but the count of simultaneous exits in a single
   rebalance is itself a diagnostic worth surfacing (`n_hysteresis_exits(t)`
   spiking far above its typical weekly value is the observable signature of
   this edge case, distinguishing it from ordinary one-or-two-name churn).
   UNIVERSE does not smooth, cap, or stagger a mass exit — every name that
   genuinely crosses `q_out` exits on schedule, because slow-walking a real
   liquidity collapse to reduce realized turnover would itself be a
   lookahead-adjacent, non-`F_t`-measurable decision (it would require
   knowing the exit is "part of a crisis" rather than a durable regime
   change, which is not knowable at `t`).
9. **A RED constant itself changes (a BLUE-to-RED promotion, or a RED value
   revision) — e.g. `N` moves from 50 to 60, or `q_in`/`q_out` are revised.**
   This is an AMBER-free, human, infrequent, class-level governance act
   (Section 8), never a per-instrument or realized-P&L-triggered change
   (Invariant C). Operationally, on the day a RED constant changes:
   - **`U_t` history is NEVER retroactively recomputed.** Every historical
     `U_t` for `t` strictly before the change's effective date was, and
     remains, `F_t`-measurable under the OLD constant value that was live at
     that `t` — recomputing history under the NEW value would inject
     information ("we later decided `N=60` was better") into a past
     eligibility decision that did not have access to that decision at the
     time, which is precisely the kind of lookahead Section 1.1 forecloses,
     just applied to a GOVERNANCE parameter instead of a market-data field.
   - **The new value applies PROSPECTIVELY ONLY**, starting at the first
     rebalance `t` at or after the change's effective timestamp. The
     hysteresis state machine of Section 2.2 is NOT reset or re-initialized
     across a RED-constant change — a name's IN/OUT state carries over
     unchanged (only the THRESHOLDS `q_in`/`q_out` it is compared against
     change), and `N`'s cardinality cutoff (Section 3.2) is simply
     re-applied at the new value from that rebalance forward, consistent
     with Invariant SIZE's monotone-nesting property (Section 3.2) holding
     at each `t` under whichever `N` was live at that `t`.
   - **A change is a validated BLUE→RED promotion event, documented like any
     other.** Because Section 8's BLUE grids (`N ∈ {50,60,75,100}`,
     `W ∈ {20,30,45}`, the hysteresis band width) are pre-enumerated and
     pre-validated under the SAME CPCV/DSR gates as SIGNAL's lookback band
     (Section 7), "promoting" a BLUE candidate to the live RED value is not a
     new fit — it is selecting among already-validated candidates — but it
     is still logged as a governance event with an effective timestamp, so
     that `U_t`'s dependence on "which RED value was live at `t`" is itself
     auditable, not implicit tribal knowledge.

> **Invariant PIT — the master invariant, structurally testable without a
> backtest.** For every rebalance `t`, `U_t` and `U_t^short` are unchanged
> under ANY corruption or deletion of data at timestamps `s > t`. Test
> (mirrors COST.md's Invariant W methodology, re-aimed at universe
> construction): compute `U_t` twice — once against the full historical
> panel, once against a copy with every row `s > t` overwritten with garbage
> — and assert bit-identical `U_t` / `U_t^short` both times, for a sample of
> `t` spanning the whole backtest window. A construction that reads
> `panel.close`/`panel.listed` at any index `> t` (e.g. via an off-by-one in
> the day-bucketing arithmetic of `panel.py::_daily_close`'s
> `(ts - 1) // MS_PER_DAY` convention, or a naive `.rolling()` call that
> windows forward instead of backward) fails this test immediately, without
> needing a single dollar of realized P&L to detect it.

---

## 10. References

- Karatzas, I. & Shreve, S. (1991). *Brownian Motion and Stochastic Calculus*,
  2nd ed. — stopping-time formalism used in Section 1.2.
- Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." *Annals of Statistics* — moving-block bootstrap,
  underlying the Section 3.1 block-bootstrap CI on the `N=50` vs `N≈65-75`
  comparison.
- Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for
  the Dependent Bootstrap." *Econometric Reviews* — block-length calibration
  for the same bootstrap.
- Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
  Time-Series Effects." *Journal of Financial Markets* — the illiquidity
  estimator underlying the real impact cost cited in Section 5 (as opposed
  to the rejected Corwin-Schultz alternative).
- Corwin, S. A. & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask
  Spreads from Daily High and Low Prices." *Journal of Finance* — the
  estimator explicitly tried and rejected (Section 5's cautionary note) for
  conflating volatility with spread on this data.
- López de Prado, M. (2018). *Advances in Financial Machine Learning* — the
  combinatorial purged cross-validation protocol (Section 7's CPCV numbers)
  and the general purge/embargo discipline this spec's Section 1 point-in-time
  requirement feeds.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). "Pseudo-
  Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting
  on Out-of-Sample Performance." *Notices of the AMS* — the PBO logit-of-
  relative-rank estimator cited in Section 7.
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management* — DSR/PSR, cited in Section 7 and shared
  verbatim with SYSTEM.md's §7 machinery (`convexity.system.validate.deflated`).
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts
  Journal* — the `var_floor` sampling-variance form reused (via SYSTEM.md)
  by the DSR gate cited in Section 7.
- Petajisto, A. (2011). "The Index Premium and Its Hidden Cost for Index
  Funds." *Journal of Empirical Finance* — index reconstitution buffer-zone
  practice, the empirical analogy motivating the Section 2.3 band-width
  operating point (a few-week, not multi-month, churn/responsiveness
  trade-off).
- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2),
  100–115. — the average-run-length calibration logic (decision-interval
  width vs. noise scale and target false-alarm rate) that Section 2.3's
  worked derivation of the hysteresis band half-width `h_min(n, α)` adapts
  from a one-sided CUSUM alarm to a two-threshold band.

---

## 11. Directory layout and build contract (implemented — see Status update below)

**Status update: this section's directory layout and build contract are now implemented.**
This section originally recorded Sections 2.2–2.3 (hysteresis) and Section 4 (shortability) as
mechanisms that did not yet exist in code. That is no longer the case: every module in the
directory layout below exists on disk exactly as laid out, and every function signature in
§11.1 is implemented — verified by reading each file: `src/convexity/universe/liquidity/hysteresis.py`
(`hysteresis_step`), `src/convexity/universe/shortability/liquidity_gate.py` (`short_liquid`),
`src/convexity/universe/shortability/venue_gate.py` (`short_permitted`),
`src/convexity/universe/validate/pit_check.py` (`pit_check`, `corrupt_future`),
`src/convexity/universe/eligibility/{stopping_times.py,history_gate.py}`,
`src/convexity/universe/liquidity/ranking.py`, and `src/convexity/universe/universe_api.py`
(`build_book`, `build_eligibility`, `build_shortability`) — essentially zero `NotImplementedError`
stubs remain (the one `raise NotImplementedError` in `liquidity_gate.py::short_liquid` is a
defensive guard against a caller passing `threshold_fn=None`, not an unimplemented stub — see
that module's docstring and §4.2's disclosed limitation, which still stands: no numeric
short-liquidity threshold is fabricated). The signatures in §11.1 below remain the BUILD
CONTRACT this spec laid out; the real implementations follow this same contract and style
(verified by reading each file), so what follows is preserved as the spec's own record of that
contract, not a claim that it is still unbuilt.

```text
universe/
+-- config/
|   +-- defaults.py        # RED: W, q_in, q_out, N, short-strictness constants
|   +-- candidates.py       # AMBER: loads src/convexity/panel/UNIVERSE.md candidate tables (per sleeve)
+-- eligibility/
|   +-- stopping_times.py   # tau_list_i / tau_delist_i from observed data (Section 1.2)
|   +-- history_gate.py     # tau_history_i = tau_list_i + L_max + w_warmup (Section 1.4)
+-- liquidity/
|   +-- dollar_volume.py    # DV_i(d), trailing W-day, causal (REUSE src/convexity/panel/panel.py)
|   +-- hysteresis.py       # per-symbol IN/OUT state machine (Section 2.2)
|   +-- ranking.py          # cross-sectional rank_i(t) + top-N cutoff (Section 3.2)
+-- shortability/
|   +-- venue_gate.py       # AMBER short_permitted_i(t)
|   +-- liquidity_gate.py   # stricter short_liquid_i(t) (Section 4.2)
+-- universe_api.py         # public U_t / U_t_short entry point
+-- types.py                # frozen dataclasses: EligibilitySnapshot, ShortabilitySnapshot
validate/
+-- pit_check.py            # Invariant PIT: future-corruption bit-identity test
+-- survivorship_probe.py   # Section 6-style effective-breadth check on any candidate roster
```

Existing, already-implemented substrate this spec formalizes and extends:
`src/convexity/panel/panel.py` (point-in-time `listed` mask, `trailing_dollar_volume`,
`build_dollar_volume`), `src/convexity/panel/sleeve.py` (`sleeve_symbols`,
`effective_breadth`), `src/convexity/panel/candidates.py` (`load_universe`, `all_symbols`),
`src/convexity/panel/UNIVERSE.md` (the candidate symbol tables). Also now implemented (Status
update above): the hysteresis state machine (Section 2.2, `liquidity/hysteresis.py`), the
explicit `U_t^short` gate (Section 4, `shortability/{venue_gate.py,liquidity_gate.py}` composed
in `universe_api.py::build_shortability`), and the `pit_check.py` structural test (Invariant PIT).

### 11.1 Function signatures (build contract — implemented; see Status update in §11)

> Every signature below is the **build contract** this spec laid out. It is now implemented —
> see the Status update at the top of §11 for the exact file/function mapping. The illustrative
> bodies below (each still showing `raise NotImplementedError`) are preserved verbatim as the
> spec's own record of the contract a future implementer had to match; they are SPECIFICATION
> TEXT, not a claim about the current state of `src/convexity/universe/`. Style matches
> `src/convexity/panel/panel.py`'s existing dataclass/type-hint conventions, and the real
> implementation follows it.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EligibilitySnapshot:
    """One rebalance's eligibility state (UNIVERSE.md Sec 1-3). Immutable, one per `t`."""

    t: int  # rebalance index (Friday-close ordinal, matching src/convexity/panel/panel.py's day index)
    base_eligible: np.ndarray  # shape (n_symbols,), bool — U_t^base (Sec 1.4)
    hysteresis_state: np.ndarray  # shape (n_symbols,), bool — U_t^base-liquid IN/OUT (Sec 2.2)
    rank: np.ndarray  # shape (n_symbols,), float in (0,1] or NaN — rank_i(t) (Sec 2.1)
    universe: np.ndarray  # shape (n_symbols,), bool — final U_t (Sec 3.2), universe <= hysteresis_state
    n_hysteresis_entries: int  # count of OUT->IN flips this rebalance (Sec 9 edge case 8 diagnostic)
    n_hysteresis_exits: int  # count of IN->OUT flips this rebalance (Sec 9 edge case 8 diagnostic)


def hysteresis_step(
    rank_t: np.ndarray,
    state_prior: np.ndarray,
    *,
    q_in: float,
    q_out: float,
) -> np.ndarray:
    """One rebalance's per-symbol hysteresis FSM transition (UNIVERSE.md Sec 2.2).

    ``rank_t`` -- shape (n_symbols,), this rebalance's `rank_i(t)` in (0,1], NaN for symbols not in
                  `U_t^base` (Sec 1.4) -- a NaN input MUST transition to OUT regardless of `state_prior`
                  (P6, Sec 2.4's conservative-fallback discipline extended to the FSM boundary).
    ``state_prior`` -- shape (n_symbols,), bool, this symbol's IN/OUT state at `t^-` (the FSM's own
                        memory -- Sec 2.2's `F_t`-measurability argument requires this be exactly the
                        prior call's returned `state_next`, never recomputed from scratch).
    ``q_in``, ``q_out`` -- RED thresholds, `q_in > q_out`, Sec 2.3 (values TBD pending the Sec 2.3
                            sensitivity-checked BLUE grid promotion, not fabricated here).

    Returns shape (n_symbols,) bool, the state AFTER this rebalance's transition rule (Sec 2.2's three
    cases: enter if `~state_prior & (rank_t >= q_in)`; exit if `state_prior & (rank_t < q_out)`;
    else unchanged). Test: Invariant HYST (Sec 2.2) -- a synthetic `rank_t` oscillating strictly inside
    `(q_out, q_in)` for the whole series produces zero flips over the whole series, from either start
    state (Sec 11.2, `test_hysteresis.py::test_no_flip_inside_band`).
    """
    raise NotImplementedError("specification only -- see UNIVERSE.md Sec 2.2")


def short_liquid(
    rank_t: np.ndarray,
    dv_t: np.ndarray,
    *,
    threshold_fn: "ShortLiquidityThreshold",
) -> np.ndarray:
    """Per-symbol strict short-liquidity gate `short_liquid_i(t)` (UNIVERSE.md Sec 4.2).

    ``rank_t`` -- shape (n_symbols,), same `rank_i(t)` as `hysteresis_step` (Sec 2.1).
    ``dv_t`` -- shape (n_symbols,), this rebalance's `DV_i(d)` (Sec 2.1), for any absolute-floor
                component of `threshold_fn` (Sec 4.2's example: a floor independent of cross-sectional
                rank, so a liquidity-poor cross-section still has a binding absolute test).
    ``threshold_fn`` -- an INJECTED callable, never a threshold hard-coded inside this function or
                         inside `universe_api.py`'s call site (Sec 4.2's disclosed-limitation
                         requirement: no numeric short-liquidity threshold is fabricated in this spec;
                         supplying one is a future, data-backed, governance-logged RED promotion, not
                         a code change to this function's signature).

    Returns shape (n_symbols,) bool. MUST be false wherever the corresponding `hysteresis_step`
    output (Sec 4.1's `U_t^short subseteq U_t` requirement) is false -- callers compose this with
    `base_eligible & hysteresis_state`, this function does not re-check them (Sec 4.2: `short_liquid_i`
    is one of THREE ANDed terms, not a standalone replacement for the other two).
    """
    raise NotImplementedError("specification only -- see UNIVERSE.md Sec 4.2 -- ShortLiquidityThreshold has no default; see Sec 4.2 disclosed limitation")


def pit_check(
    panel_full: "Panel",
    t: int,
    *,
    eligibility_fn: "Callable[[Panel, int], EligibilitySnapshot]",
) -> bool:
    """Invariant PIT structural test (UNIVERSE.md Sec 7, Sec 9 item 9, Invariant PIT).

    ``panel_full`` -- the real historical panel (src/convexity/panel/panel.py's `Panel` type).
    ``t`` -- the rebalance index under test.
    ``eligibility_fn`` -- the eligibility construction under test (e.g. a composition of
                           `hysteresis_step` + Sec 1.4's history gate + Sec 3.2's top-N cutoff).

    Returns True iff `eligibility_fn(panel_full, t) == eligibility_fn(panel_corrupted, t)` where
    `panel_corrupted` is `panel_full` with every row at index `s > t` overwritten with garbage
    (NaN-poisoned or randomized) close/volume/listed data. This is a BIT-IDENTITY check on
    `EligibilitySnapshot.universe` and any short-eligibility field, not a statistical check -- it
    needs no backtest and no realized P&L (Sec 7's explicit warning against circular validation).
    """
    raise NotImplementedError("specification only -- see UNIVERSE.md Sec 7, Invariant PIT")
```

### 11.2 Test files and assertion bodies (implemented — see Status update in §11)

> Every test file and function name below exists on disk exactly as named, verified by reading
> `tests/universe/`. Each row maps one of this spec's named Invariants to its pytest file,
> function name, and assertion — originally proposed at the same level of concreteness
> `docs/specs/EXECUTION.md` §11.2 and LEVERAGE.md's Invariant VOL-FLOOR give their own test
> tables; that concreteness is now realized code, not a proposal.

| Invariant | Test file | Function name | Assertion body |
|---|---|---|---|
| HW | `tests/universe/test_history_gate.py` | `test_history_gate_is_per_symbol_not_panel_global` | `sym.tau_list = t - 30; assert not eligible_base(sym, t)` for a panel where OTHER symbols have decades of history; must hold regardless of panel-wide earliest date |
| HYST | `tests/universe/test_hysteresis.py` | `test_no_flip_inside_band` | `ranks = oscillate_between(q_out + eps, q_in - eps, n_weeks=104); states = simulate(hysteresis_step, ranks); assert states.nunique() == 1` from either start state |
| HYST | `tests/universe/test_hysteresis.py` | `test_decay_to_exactly_q_out_never_exits` | `ranks = [q_in, q_in - delta, ..., q_out + 1e-9]; assert simulate(hysteresis_step, ranks)[-1] == IN` |
| SIZE | `tests/universe/test_ranking.py` | `test_universe_monotone_nesting` | `for n1, n2 in [(50,60),(60,75),(75,100)]: assert set(top_n(base, n1)) <= set(top_n(base, n2))` on the same `rank_i(t)` and hysteresis-filtered base set |
| SHORT | `tests/universe/test_shortability.py` | `test_short_subset_never_violated` | `assert (short_liquid(...) & ~hysteresis_state).sum() == 0` — no short-eligible symbol outside `U_t` |
| SHORT | `tests/universe/test_shortability.py` | `test_short_starvation_flagged_not_silently_widened` | `snap = build(fixture_with_starved_short_tail); assert snap.n_short_eligible_in_tail < k and snap.short_starved_flag is True` and `snap.universe` composition is UNCHANGED by the flag |
| C | `tests/universe/test_config.py` | `test_red_constants_are_not_per_symbol` | `assert not hasattr(SymbolConfig, "q_in") and not hasattr(SymbolConfig, "q_out") and not hasattr(SymbolConfig, "N")` — a static/schema check, no fixture data needed |
| CANDIDATE-VS-ELIGIBLE | `tests/universe/test_candidates.py` | `test_candidate_symbol_excluded_before_listing` | `sym in candidates.crypto_50 and sym.tau_list > t; assert sym not in eligibility(t).universe` |
| PIT | `tests/universe/test_pit_check.py` | `test_future_corruption_does_not_change_past_universe` | `assert pit_check(panel_full, t, eligibility_fn=build_eligibility) is True` for a sample of `t` spanning the full backtest window, per Sec 7's methodology |

Each file additionally carries a `test_conservative_fallback` (asserting
Section 2.4's stale/missing-data-to-rank-0 rule never admits a name it would
otherwise exclude) mirroring EXECUTION.md's own per-file convention of one
shared cross-cutting assertion per test module.
