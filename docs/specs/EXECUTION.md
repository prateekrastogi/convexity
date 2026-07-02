# EXECUTION.md — build spec for the weekly rebalance execution layer

This file is the authoritative build spec for the EXECUTION module. Build to the
contracts and invariants below. Where this file and your own instinct disagree,
this file wins. Ask before deviating from any **Invariant**.

**This pillar does not exist in code yet.** `research/xsec/{panel,signals,portfolio,accounting,engine}.py`
back-test a **frictionless, instantaneous Friday-close fill**: `engine.run()` computes target
weights at rebalance day `d` and `accounting.settle_week` marks the whole week's P&L as if the
entire book re-weighted atomically at the closing print, paying only a flat per-unit-turnover
friction constant (`round_trip_friction()`, §5.1 below). That backtest answer is the **decision
price benchmark**, never the fill price — a real rebalance cannot cross ~$1–2M of notional across
~20–30 names in zero time at zero market impact. EXECUTION is the pillar that closes this gap: it
turns "the target book computed at the Friday close" into "the sequence of child orders that gets
you there," prices the resulting shortfall, and feeds it back into COST so the honest net Sharpe
(**~0.8–1.0**, `docs/EDGE.md` §b) is measured against *realized* fills, not the backtest's
frictionless proxy.

> **Known cross-spec inconsistency (stated up front, not buried mid-file).** `docs/specs/COST.md`
> §11 (Non-goals) states, verbatim and still current: *"there is no EXECUTION pillar in the
> xsec-momentum architecture (the book rebalances with a batch of taker orders at Friday close; the
> tactic-selection layer the old EXECUTION.md described does not exist in this pivot)."* That
> sentence was correct when COST.md was written and has NOT been edited as part of this pass — this
> file (EXECUTION.md) is the sibling pillar COST.md's own scope section says does not exist. This is
> a genuine, currently-unresolved disagreement between two specs, not a stale typo: correcting it
> requires COST.md to gain its own ingestion seam for a realized-fill feedback series, which is
> substantive COST-owned scope work (see §7.2's full accounting of exactly what is and is not
> fabricated here), not a one-line fix — so it is deliberately left untouched in COST.md and is
> instead disclosed prominently, here, at first read. See §7.2 for the complete resolution EXECUTION
> commits to in the interim (routing the realized-shortfall feedback through OPS's existing
> `on_obs` seam instead of inventing a COST-side one).

---

## 0. Governing principles (read before anything else)

**P1 — The risk unit is the WEEKLY REBALANCE, not the trade.** The directional-breakout system's
COST.md and SYSTEM.md (superseded, salvaged in part below) were built around a single trade's
entry/exit and its R-multiple. That unit does not exist here. The xsec book has no per-trade stop,
no per-trade target, no per-trade R. The atomic unit of risk and of execution is: **"get this
week's target dollar-neutral book, `w_target ∈ R^N`, in place before the drift/cost of not doing so
exceeds the cost of doing so."** Every formula in this file is stated over a rebalance window
`[t, t+Δ]`, never over a trade's holding period. (Carried forward from COST.md's P1 "one truth"
spirit, remapped: here it is one *rebalance unit*, not one *trade unit*.)

**P2 — EXECUTION is a PROVIDER of a trajectory and a REPORTER of shortfall, never a signal
decider.** EXECUTION does not decide *what* to hold — `src/convexity/signal/signals.py` and
`src/convexity/portfolio/portfolio.py` already emit `w_target` at the Friday close. EXECUTION decides only
*how* to get from the currently-held drifted book `w_carried` to `w_target` (or, under the no-trade
band of §6, to *not* trade a name at all) and *how fast*. It never overrides a rank, a quantile
membership, or a vol-scaled weight. (Direct analogue of COST.md's P2 "provider, never decider,"
remapped from single-order gating to a multi-name trajectory.)

**P3 — Cost during execution is an ADDITIVE, decomposable expectation-plus-variance object, never
a single opaque "slippage score."** The Almgren–Chriss (2001) objective in §2 splits into a
temporary-impact expectation term, a permanent-impact expectation term (path-independent), and a
variance-of-execution-cost term — each auditable, each attributable to a named line in the
post-trade TCA (§7). (Same spirit as COST.md P3, restated for a multi-period trajectory instead of
one round-trip.)

**P4 — Live turnover is a CONSERVATIVE (i.e. ≤) approximation of backtested turnover, never the
reverse.** The backtest's `accounting.py` trades every unit of drift every week (full rebalance to
`w_target`, no band). The no-trade band of §6 can only *skip* trades that the backtest took; it
never adds a trade the backtest didn't take. This is a one-directional bias and must be stated as
such wherever live vs. backtested turnover is compared (§6.4, Invariant E-monotone).

**P5 — Calendar cadence is PRIMARY; drift cadence is a DEFENSIVE ADDITION gated by REGIME, never a
parallel independent trigger.** The validated edge (`docs/EDGE.md` — CPCV, purged walk-forward, DSR)
is entirely a **Friday-close weekly cadence** result. A drift-triggered rebalance outside that
cadence is not validated by any of the backtest evidence and must never fire as a first-class,
always-on mechanism. It exists only as an intra-week defensive de-risking action, armed exclusively
by REGIME's defensive-overlay state (§8, Invariant CAL-PRIMARY).

**P6 — Conservative under uncertainty, symmetric with COST's P6.** When a participation forecast,
a volatility estimate, or a venue liquidity read is missing or stale, EXECUTION widens the
trajectory (slower, more front-loaded-risk-averse) and reports a WORSE expected shortfall, never a
better one. A trajectory that cannot be costed is not fired at IMMEDIATE urgency; it degrades to
the most patient tactic available before the hard cutoff `t+Δ`.

**P7 — Every live number is `F_t`-measurable.** The optimal trajectory, the no-trade band, and the
participation forecast used inside the window `[t, t+Δ]` are computed from information available at
`t` (the Friday close signal / prior fills), never from realized fills or volume inside `(t,
t+Δ]` that haven't happened yet. Backtesting this pillar without firewalling `(t, t+Δ]` volume is
the single most dangerous lookahead channel in this spec (**Invariant W, defined in this file**,
§13 below — the underlying causality PRINCIPLE is shared with every pillar in this repo, stated in
COST.md's scope section as "every live number must be `F_t`-measurable," and in SIGNAL.md/UNIVERSE.md/
LEVERAGE.md's own analogous per-pillar causality invariants, but COST.md does not itself define a
numbered invariant literally named "Invariant W" — an earlier draft of this spec incorrectly called
this "inherited verbatim from COST.md §5.4"; corrected here to state plainly that this is EXECUTION's
OWN named invariant, applying a repo-wide causality PRINCIPLE that COST.md, SIGNAL.md, and UNIVERSE.md
each also restate under their own local names/sections, not a single shared numbered invariant object
imported by reference).

Forbidden in v1: any ML / learned execution policy (reinforcement-learning order placement, learned
slicing schedules) — stays inside the NO-ML constraint (P1); a risk-neutral-only (TWAP-everywhere)
trajectory presented as if it were risk-aware (P3); a no-trade band with no derivation (a bare "skip if
|drift| < 1%" magic number) (P3); comparing live turnover to backtested turnover without stating the
direction of the bias (P4); firing a drift-triggered rebalance without an explicit REGIME arming
condition (P5); pricing the trajectory's impact with a different `k_amihud`/`Amihud_i` than COST's
fitted impact model (§5.3 of COST.md) — EXECUTION consumes COST's impact coefficients, it does not
refit its own (P2, Invariant N-exec); reading `(t, t+Δ]` realized volume into the trajectory (P7,
Invariant W).

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its first point of use in
prose; this table is a lookup, not a substitute for those definitions) — same discipline as
LEVERAGE.md §0.1 / SYSTEM.md §0.1. Rebalance index `t_0` runs over Friday-close rebalances, the
same clock SYSTEM/SELECTION use (`src/convexity/portfolio/engine.py`).

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `t_0` | A rebalance date (Friday close) | — | 2.1 |
| `Δ` | Execution window length after `t_0` (e.g. 1–4 hours) | AMBER | 2.1 |
| `n` | Sub-interval (child-slice) count discretizing `[t_0, t_0+Δ]` | AMBER | 2.1 |
| `τ` | Sub-interval length, `τ = Δ/n` | derived | 2.1 |
| `N`, `U_{t_0}` | Eligible cross-section size / set at `t_0` (UNIVERSE/SIGNAL-owned, inherited via `w_target`'s support) | UNIVERSE-owned | 2.1 |
| `w_target`, `w_carried` | Target weight vector (PORTFOLIO output); currently-held drifted weight vector (accounting.py) | data | 2.1 |
| `x`, `x_i` | Required trade vector / per-name required trade, `x = w_target − w_carried` | derived | 2.1 |
| `X_{i,k}`, `x_{i,k}` | Remaining quantity to trade after sub-interval `k`; quantity traded in sub-interval `k` | derived | 2.1 |
| `X(t)`, `v(t)` | Continuous-time remaining quantity; trading rate `v(t) = −Ẋ(t)` | derived | 2.4 |
| `g(v)`, `h(v)` | Permanent / temporary impact cost functions | — | 2.2 |
| `γ`, `η` | Permanent / temporary linear impact coefficients (read from COST's fitted `k_amihud`/`Amihud_i`) | TEAL (COST-owned, consumed) | 2.2, 10 |
| `σ` (§2), `σ²` | Per-unit-time PRICE variance/stdev feeding the Almgren-Chriss trajectory (`portfolio.realized_vol`'s statistic) | TEAL | 2.1, 2.4 |
| `λ`, `λ_i` | Risk-aversion parameter, Almgren-Chriss mean-variance trade-off (dollars of `Var[C]` tolerated per dollar of `E[C]` reduction) | RED | 2.3, 2.5, 10 |
| `κ`, `κ_i` | Front-loading rate, `κ² ≡ λσ²/η` | derived | 2.4, 2.5 |
| `β` | Stated execution-cost risk budget (fraction of one week's target-book variance) | RED | 2.5, 10 |
| `E[C]`, `Var[C]` | Expected / variance of total trajectory cost (implementation shortfall vs. `t_0` decision price) | — | 2.3 |
| `b_i` | No-trade band half-width for name `i` | derived | 4.2–4.3 |
| `θ_i` | Name `i`'s proportional (linear fee+half-spread) round-trip cost | TEAL | 4.2 |
| `ρ` | CRRA-style local risk-aversion coefficient of the band's value function (dimensionless — NOT `λ`) | RED | 4.2 |
| `c` (§4) | No-trade band's dimensionless fitted constant | RED | 4.2 |
| `σ_w,i` | Per-name realized WEIGHT-drift volatility (NOT `portfolio.realized_vol`'s price-return stat) | TEAL | 4.2, 10 |
| `V_week` | One week's target-book return variance (diagonal approximation) | derived | 2.5 |
| `IS_i`, `IS_total` | Per-name / book-level implementation shortfall (Perold 1988) | derived | 7.1 |
| `P_decision,i`, `P̄_fill,i` | Friday-close decision price; realized notional-weighted average fill price | data | 7.1 |
| `λ̂_fill,i` | Causal per-name fill-rate forecast (venue order-book depth/turnover) | AMBER | 2.1, 10 |
| `g_t`, `regime_defensive` | REGIME's gross multiplier (not consumed here) / REGIME's one boolean consumed by §6.2, §8 | REGIME-owned | 6.2, 8 |

### 0.2 RED/AMBER parameter cross-index

Every governance-bucketed (RED or AMBER) parameter introduced in this file, mapped to the section
that owns its derivation and the Invariant (if any) that constrains it — mirroring LEVERAGE.md
§0.1's Bucket/Owning-section columns as a single consolidated lookup surface, so a reader does not
have to reconstruct this mapping from Section 10's prose.

| Parameter | Bucket | Owning section | Governing Invariant / principle |
|---|---|---|---|
| `Δ` (execution window) | AMBER | 2.1 | bounded by the two regularity conditions of §2.1 (P7, Invariant W) |
| `n` (slice count) | AMBER | 2.1, 11 | — |
| `λ`, `λ_i` (risk aversion) | RED | 2.5 | set from stated budget `β`, never fitted on P&L (P3) |
| `β` (risk budget) | RED | 2.5 | governance choice, not fitted (P3) |
| `ρ` (band CRRA curvature) | RED | 4.2 | NOT the same knob as `λ` (explicit non-identity, §4.2) |
| `c` (band constant) | RED | 4.2 | fitted once offline to `σ_w`, never per-instrument on P&L (Invariant C-cfg-exec) |
| `κ` (front-loading rate, per-name) | derived (from `λ`, `σ`, `η`) | 2.4–2.5 | — |
| `θ_i` (band's linear cost input) | TEAL | 4.2, 4.4 | linear component ONLY, excludes convex impact (Invariant B-band) |
| `σ_w,i` (weight-drift vol) | TEAL | 4.2 | must differ from `portfolio.realized_vol` (Invariant SIGMA-W-DISTINCT) |
| `η`, `γ` (impact coefficients) | TEAL (COST-owned) | 2.2, 10 | read byte-identical from COST §5.3, never refit (Invariant N-exec) |
| `λ̂_fill,i` (fill-rate forecast) | AMBER | 2.1 | causal, `F_t`-measurable (P7) |
| REGIME defensive-arming threshold | REGIME-owned (referenced only) | 6.2, 10 | never fires without active flag (Invariant CAL-PRIMARY) |

---

## 1. What EXECUTION builds ON TOP OF (and does not rebuild)

| Already exists / owned elsewhere                                   | EXECUTION consumes it as                                  |
|----------------------------------------------------------------------|-------------------------------------------------------------|
| `research/xsec/portfolio.quantile_weights` — `w_target` at `t` (already the resolved `L∈{12,14,16}` ensemble score's output, §2.1a — ONE vector, never per-lookback) | the trajectory's TERMINAL condition (must reach `w_target` by `t+Δ`) |
| Universe/eligibility layer (`src/convexity/panel/candidates.py`, `src/convexity/panel/sleeve.py`) — per-name shortable flag at `t` | a REQUIRED companion field to `w_target`'s short-side entries (Invariant SHORTABLE-GATE, §13); EXECUTION does not compute shortability, only gates on it |
| `research/xsec/accounting.round_trip_friction` — flat per-unit cost | the risk-neutral (λ=0) cost floor; EXECUTION adds impact curvature on top |
| COST §5.2–5.4 (`SpreadImpactModel`) — sqrt-law half-spread anchor + fitted Amihud-based `k_amihud`, per-name `Amihud_i` | the `η` (temporary) / `γ` (permanent) impact coefficients feeding Almgren–Chriss (§2.2) |
| COST §6 (`FundingModel`) | funding accrual during the execution window is a COST input, not re-derived here |
| NautilusTrader — order plumbing, fills, position/account state, backtest↔live parity | EXECUTION emits child `OrderCommand`s; it does not track position itself (reads Nautilus's working-order snapshot) |
| `convexity.system.validate.{deflated,walkforward,cpcv}` — DSR / purged-WF / CPCV machinery (SYSTEM §5, §4) | REUSED verbatim for the shortfall-vs-honest-Sharpe robustness check (§9), never reimplemented |

> **Invariant N-exec (no re-derivation of upstream pillars).** EXECUTION does not refit an
> impact coefficient, does not recompute a fee tier, and does not re-derive `w_target`. A test
> asserts that the `η`, `γ` used inside this pillar's Almgren–Chriss solve are read, byte-identical,
> from `SpreadImpactModel`'s fitted `k_amihud`/`Amihud_i` (COST §5.3) — not a parallel
> EXECUTION-local fit. Two divergent impact numbers for the same instrument is the bug this
> invariant forbids.

---

## 2. The rebalance as an optimal-execution problem

### 2.1 Setup and notation

Fix a rebalance date (a Friday close) `t_0`. The signal/portfolio layers (`signals.py`,
`portfolio.py`) are computed on `F_{t_0}` — everything knowable from daily closes up to and
including `t_0` — and emit `w_target ∈ R^N`, the dollar-neutral target weight vector over the
`N ≈ 50` (§4, EDGE) eligible names, where `N = |U_{t_0}|` is exactly SIGNAL.md §2.3 / UNIVERSE.md
§1.1's `F_t`-measurable eligible cross-section `U_t` (the same object those specs define formally;
EXECUTION does not redefine `U_t`, it only inherits its cardinality via `w_target`'s support — see
§15 for the explicit non-goal boundary). The book currently holds the **drifted** prior week's weights
`w_carried` (COST.md's accounting note: a name that ran up carries a larger weight into the
rebalance even with no rank change — `src/convexity/cost/accounting.py` docstring). Define the **required
trade vector**:

```
x = w_target − w_carried  ∈ R^N            (units: fraction of book NAV, signed)
```

Execution must convert the paper decision `x` into a sequence of real fills over a bounded window
`[t_0, t_0+Δ]`. Take `Δ` = the intra-day window immediately following the Friday close (e.g. 1–4
hours; an AMBER-class venue-liquidity parameter, not fitted per name). `Δ`'s two bounding
regularity conditions, made explicit (not just prose) so the range is checkable rather than merely
asserted:

```
Δ ≪ 7 days                         (upper bound: "short relative to the weekly rebalance" — keeps
                                     the frictionless-instantaneous-fill backtest approximation
                                     first-order valid; formally the Δ→0 limit of §2.4's trajectory
                                     recovers that backtest exactly, so Δ/7 is the operative small
                                     parameter, not Δ in isolation)
Δ ≳ n* / λ̂_fill,i   ∀ i ∈ tail     (lower bound: long enough that even the SLOWEST-filling name in
                                     the current book — the illiquid-tail name with the smallest
                                     causal fill-rate forecast λ̂_fill,i, AMBER, from venue order-book
                                     depth/turnover metadata — can clear its required n* = O(10)
                                     child slices without forcing v_i = x_i/Δ above a participation
                                     rate that would itself dominate the impact being measured; this
                                     is the same participation-rate object COST §5.2's square-root
                                     law is a function of, so the bound is checkable against COST's
                                     own ADV/participation fields, not a separate invented quantity)
```

Both bounds are satisfied simultaneously for the liquid core of the universe by any `Δ` in the
stated 1–4 hour AMBER range; the lower bound is the one that can bind for the illiquid tail (§3),
which is precisely why §3 assigns the illiquid tail a front-loaded (`λ>0`) trajectory rather than
pure TWAP — front-loading does not relax the lower bound on `Δ` itself, but it reduces the *risk*
of not clearing it by paying more expected cost for less variance (§2.3).

Discretize `[t_0, t_0+Δ]` into `n` equal sub-intervals of length `τ = Δ/n` (Almgren & Chriss 2001,
§2). For a single name `i` with required trade `x_i`, let `X_{i,k}` be the **remaining** quantity to
trade after sub-interval `k` (`X_{i,0} = x_i`, `X_{i,n} = 0`), and `x_{i,k} = X_{i,k-1} − X_{i,k}`
the quantity traded in sub-interval `k`. This is the discrete Almgren–Chriss trajectory, applied
per-name and summed for the book-level cost.

**Impact is not modeled cross-sectionally — bounding how much this could matter.** Only cross-name
correlation in *risk* is modeled (via realized vol, which the vol-scaled weighting of `portfolio.py`
already handles upstream of this pillar); cross-name correlation in *impact itself* (e.g. two
correlated majors being sold into the same market-wide liquidity shock simultaneously, each making
the other's fill worse) is assumed away. Bounding the plausible error: the entire book's rebalance
notional trades over `Δ` is a small fraction of one name's own `Δ`-window ADV even summed across
all `N≈50` names (the book-level target is ~$1–2M notional against a universe whose median
single-name ADV is itself in the tens of millions, per COST/UNIVERSE's liquidity floor), so
correlated cross-impact would have to be an order of magnitude larger than single-name impact to
change this section's qualitative conclusions — this is stated as a bound on plausible magnitude,
not a proof of zero effect, and is flagged for future work if book size grows enough that aggregate
same-side notional becomes a material fraction of book-wide ADV (§10, BLUE grid candidate for a
future cross-impact multiplier, not in v1's scope).

### 2.1a Interaction with the lookback-BAND signal: one `w_target`, not three trajectories

The deployed signal is **never a single lookback** `L` — SIGNAL.md §4 (Invariant BAND) deploys the
ensemble `L ∈ {12, 14, 16}` trading days.

**Cross-pillar disclosure.** This `m=3` set `{12, 14, 16}` is the single canonical lookback
ensemble used consistently across all five pillars that reference it (SIGNAL, UNIVERSE,
PORTFOLIO, EXECUTION, SYSTEM). An earlier drafting pass introduced a divergent, ungrounded `m=5`
variant `{12, 13, 14, 15, 16}` in PORTFOLIO.md and SYSTEM.md, silently interpolating two untested
lookbacks (`L=13`, `L=15`) with no backtest history anywhere in this investigation; that
divergence has since been corrected in both files, and `{12, 14, 16}` is now the only ensemble
deployed anywhere in the repo.

This ensemble is deployed specifically because SYSTEM.md §4.4's PBO finding
(`PBO=0.357`, range `[0.31, 0.47]` across `(N,k_test)` choices, `L=12` in-sample-best on 21/28 CPCV
splits — cited fully in §14) sits in the "deploy the ensemble, not a point" zone, not the "clean
pass, deploy the single best config" zone. Concretely, per SIGNAL §4.2–4.4, the ensemble is resolved
**upstream of PORTFOLIO and upstream of this pillar**: `ensemble_momentum_signal` computes a
per-lookback cross-sectional rank for each `L ∈ {12,14,16}`, averages those ranks (equal-weight,
`v_L = 1/3` each), re-ranks the average, and *that single re-ranked score* is what
`portfolio.quantile_weights` consumes to emit ONE `w_target ∈ R^N` per week (SIGNAL §4.4's
`ensemble_rank_i = _pct_rank(avg)_i`).

**Consequence for EXECUTION, stated explicitly (this was previously undiscussed):** EXECUTION never
sees three (or five) parallel books, one per lookback, and never runs three parallel trajectories or
no-trade-band decisions. It sees exactly the same object it would see if the signal were a single
fitted `L` — one `w_target`, one `x = w_target − w_carried`, one trajectory, one band decision per
name (§4). The band-vs-point decision changes *which* score produced `w_target` (a rank-averaged,
lower-variance score per SIGNAL §4.2's ensemble-Sharpe argument) but does **not** change the
cardinality or shape of what EXECUTION consumes — `x` remains a single `R^N` vector every week,
exactly as §2.1's setup states.

**Does the ensemble change turnover?** Only insofar as rank-averaging across three lookbacks changes
which names are near a quantile boundary in a given week relative to a single-`L` score — this is a
*composition* effect (different names may flip quantile membership under the ensemble score than
under a point `L=14` score) rather than a *frequency* or *cardinality* effect. No separate turnover
formula is needed: §6.4's `Turnover_live/backtest` formulas already operate on whatever `w_target`
PORTFOLIO emits, and the ensemble's effect on turnover is already fully captured inside the
backtested `x_i,week` series each formula consumes — EXECUTION's no-trade band (§4) and turnover
accounting (§6.4) are therefore unmodified by the point-vs-band decision, and this spec does not
introduce a parallel per-lookback turnover concept. The one place ensemble-vs-point *would* matter
operationally — whether rank-averaging systematically produces smaller or larger week-over-week
`|x_i|` than a point `L` — is an empirical question for SIGNAL/PORTFOLIO's own validation (SIGNAL
§4.2's ensemble-Sharpe-variance argument), not something this pillar re-derives; EXECUTION's
contract is unchanged either way because it operates on realized `w_target`, not on the score that
produced it.

### 2.2 The cost model: temporary + permanent impact

Following Almgren & Chriss (2001) §3, each child slice incurs:

- **Permanent impact** `g(v)` — a price impact that persists for the rest of the trajectory,
  linear in the trading rate: `g(v) = γ·v`, `γ ≥ 0`. Cost is **path-independent**: it depends only
  on total quantity traded, not on the trajectory's shape.
- **Temporary impact** `h(v)` — a price impact that affects only the current slice's execution
  price and decays before the next slice: `h(v) = ε·sign(v) + η·v`, `η > 0`. This is what makes the
  trajectory's *shape* matter — trading fast (`v` large) pays more temporary impact per unit.

Consistent with COST §5.3's square-root Amihud impact law (`impact_frac_i = k_amihud ·
sqrt(Amihud_i · Notional_leg)`, COST's actual fitted form — not a generic `k_i·σ_i·sqrt(participation)`
proxy, since `Notional_leg` (a dollar trade size, COST §5.2) already plays the role a participation
rate would in a more conventional square-root-law statement, and `Amihud_i` supplies the
per-name illiquidity shape in place of a separately-fit `σ_i`), the LINEAR Almgren–Chriss impact
used inside a single sub-interval is the local linearization of that same square-root curve around
the sub-interval's own trade-size scale — i.e. `η` and `γ` are not independently re-fit constants;
they are read off COST's fitted `k_amihud`/`Amihud_i` (Invariant N-exec, COST §5.3) at the notional
implied by trading `x_i` over `n` slices. This keeps one impact truth (COST P1) even though
Almgren–Chriss's closed-form trajectory is derived under a local-linear model.

### 2.3 The objective: mean-variance over execution cost

Let `E[C]` be the expected total cost of the trajectory (implementation shortfall vs. the `t_0`
decision price) and `Var[C]` the variance of that cost (driven by price volatility `σ` acting on
the *unexecuted* remainder at each step — the risk of not yet being in the target book). Almgren &
Chriss (2001) pose:

```
minimize   E[C(x)] + λ · Var[C(x)]        subject to   X_0 = x_i,  X_n = 0
             over trajectories {X_k}
```

`λ ≥ 0` is a risk-aversion parameter (dollars of variance the trader is willing to accept per dollar
of expected-cost reduction). This is a genuine mean-variance trade-off, not a single number to
minimize: trading fast (front-loaded) raises `E[C]` (more temporary impact paid up-front) but lowers
`Var[C]` (less remaining unexecuted risk exposed to price moves); trading slow raises `Var[C]` but
lowers `E[C]`.

### 2.4 Worked derivation — the continuous-time trajectory (Almgren & Chriss 2001, §5)

This is the one non-obvious derivation the rigor bar requires — shown here for a single name (the
book-level trajectory is this solved independently per name, since impact is not modeled
cross-sectionally, §2.1).

**Setup.** Let `X(t)` be the remaining position to execute at time `t ∈ [0, Δ]` (drop the `i`
subscript), `X(0) = x`, `X(Δ) = 0`. Let `σ²` be the per-unit-time variance of the *price* (not the
return — Almgren-Chriss works in price units for this derivation), and let the temporary-impact
cost per unit time be quadratic in the trading rate `v(t) = −Ẋ(t)`:

```
Var[C] = σ² ∫₀^Δ X(t)² dt                         (cost variance from holding X(t) exposed to price risk)
E[C]   = γ · x² / 2  +  η ∫₀^Δ  Ẋ(t)²  dt          (permanent path-independent + temporary quadratic)
```

(The permanent-impact expectation term is the same `γ x²/2` regardless of trajectory shape — a
known Almgren-Chriss result — so it drops out of the *optimization* over trajectory shape and only
the temporary + variance terms are shape-dependent.)

**The functional to minimize** over paths `X(t)` with fixed endpoints:

```
U[X] = η ∫₀^Δ Ẋ(t)² dt  +  λσ² ∫₀^Δ X(t)² dt
```

This is a calculus-of-variations problem. Writing the integrand as `L(X, Ẋ) = η Ẋ² + λσ² X²`, the
Euler–Lagrange condition `d/dt(∂L/∂Ẋ) − ∂L/∂X = 0` gives:

```
d/dt(2η Ẋ) − 2λσ² X = 0
  ⇒   η Ẍ(t) − λσ² X(t) = 0
  ⇒   Ẍ(t) = κ² X(t),      κ² ≡ λσ²/η
```

This is the classic "modified diffusion equation" of Almgren & Chriss (2001, eq. 20). Its general
solution is a combination of `sinh`/`cosh`:

```
X(t) = x · sinh(κ(Δ − t)) / sinh(κΔ)
```

which satisfies `X(0) = x` and `X(Δ) = 0` by construction. Two limiting cases make the result
interpretable:

- **`λ → 0` (risk-neutral):** `κ → 0`, and `sinh(κ(Δ−t))/sinh(κΔ) → (Δ−t)/Δ`, i.e. `X(t)` decays
  **linearly** in time. Constant trading rate `v(t) = x/Δ` — this recovers **TWAP** (time-weighted
  average price: equal-size slices at equal time intervals) as the risk-neutral limit of the
  mean-variance framework, exactly as the Almgren-Chriss closed form (§4.1) predicts. VWAP (volume-weighted) is the same
  risk-neutral limit but with `v(t)` proportional to the *forecast* volume profile rather than
  uniform in time — i.e. `X(t)` decays in *volume-time* rather than clock-time; it is still `λ=0`,
  only the time-change differs (a causal, `F_t`-measurable volume forecast, never realized volume —
  Invariant W).
- **`λ > 0` (risk-averse):** `κ > 0` and `sinh` is convex, so the trajectory is **front-loaded** —
  more of `x` is executed early (when `t` is small, `Δ−t` is close to `Δ`, and
  `sinh(κ(Δ−t))/sinh(κΔ)` is close to 1) than a linear TWAP schedule would trade, precisely because
  front-loading reduces the time the book spends NOT yet at `w_target` (reducing `Var[C]`) at the
  cost of paying more temporary impact per unit time up front.

The discrete-time solution used in implementation (Almgren & Chriss 2001, eq. 12–14) replaces
`sinh`/`cosh` with their hyperbolic-lattice analogues on the `n`-slice grid; the continuous form
above is what is stated in this spec because it makes the qualitative front-loading result and the
TWAP limit transparent without extra discretization bookkeeping. The discrete solver (used in
`components/trajectory.py`, §11) implements the exact discrete recursion, not a naive resampling of
the continuous curve, because `n` here is small (order 10–50 slices over a few hours) and the
discretization error of resampling the continuous curve is not negligible at that `n`.

### 2.5 Risk aversion `λ`: how it is set (RED, no fitting on P&L)

`λ` is not fitted to historical rebalance P&L (that would be exactly the forbidden "tune the RED
constant on realized P&L" pattern COST.md's Invariant C-cfg forbids, carried over here). Instead
`λ` is set from a **stated risk budget**: the maximum acceptable `Var[C]` as a fraction of one
week's target book variance. The realized-vol input `σ` feeding this is `portfolio.realized_vol`'s
PRICE-return statistic (§2.1/§2.4 — this is the correct consumer for that function; contrast §4.2's
`σ_w,i`, a genuinely different WEIGHT-drift statistic this pillar must estimate itself, not reuse
from `portfolio.py`).

**The closed-form relationship, written down (previously only promised, not shown).** From §2.4,
`κ² = λσ²/η` and `X(t) = x·sinh(κ(Δ−t))/sinh(κΔ)`. Rather than quote Almgren-Chriss (2001, eq. 21)'s
general-λ closed form as a black box, this spec derives the small-`κΔ` expansion directly from
§2.4's own trajectory (§2.1 established `Δ ≪ 7 days` is the operative small parameter, so this
order is exactly the regime EXECUTION operates in): `Var[C]* = σ² ∫₀^Δ X(t)² dt`. Substituting
`X(t)` and expanding `sinh(κ(Δ−t))²/sinh(κΔ)²` in powers of `κΔ` (a direct Taylor expansion of the
integrand, integrated term-by-term — verified symbolically, not asserted):

```
Var[C]*(κ)  =  x² σ² Δ/3 · [ 1 − (17/45)·(κΔ)² + O((κΔ)⁴) ]      (leading-order κ-dependent
                                                                    correction to the κ=0 TWAP
                                                                    variance x²σ²Δ/3 of §2.4;
                                                                    the 17/45 coefficient is the
                                                                    exact result of integrating
                                                                    sinh(κ(Δ−t))²/sinh(κΔ)² over
                                                                    [0,Δ] and Taylor-expanding in
                                                                    κΔ — shown here, not quoted)
```

i.e. increasing `κ` (more front-loading) REDUCES `Var[C]*` below the TWAP baseline, exactly as
§2.4's qualitative front-loading discussion states, and the reduction is second-order in `κΔ` with
an explicit, verified coefficient (`17/45`) — a closed-form, checkable statement of "how much" that
the earlier draft's prose-only front-loading claim did not provide.

**Setting `λ` from a stated risk budget, worked.** Let `V_week = Σ_i w_target,i² σ_i²` be one week's
target-book return variance (a diagonal approximation, ignoring cross-name correlation for this
governance-level sizing choice — a conservative simplification, since ignoring positive
diversification correlation understates `V_week` and hence understates the budget, which is the
conservative direction under P6). State a risk budget `β ∈ (0,1)` — e.g. `β = 0.01`, "execution-cost
variance should not exceed 1% of one week's target book variance." Then for name `i`, using the
`κ`-dependent correction above (retaining the `κ` term is essential — dropping it, as an earlier
draft implicitly did by using only the `κ=0` TWAP variance, would make the constraint independent of
`λ` entirely and hence unsolvable for it):

```
Var[C_i]*(κ_i)  ≤  β · w_target,i² · σ_i²                                       [per-name risk-budget constraint]
(x_i² σ_i² Δ/3)·[1 − (17/45)(κ_i Δ)²]  ≤  β · w_target,i² · σ_i²                [substituting the verified
                                                                                   correction; σ_i² cancels]
⇒  κ_i²  ≥  (45 / (17 Δ²)) · [ 1 − (3β/Δ)·(w_target,i/x_i)² ]                   [solved for κ_i² at equality
                                                                                   — verified algebraically —
                                                                                   gives the MINIMUM κ_i that
                                                                                   satisfies the budget; larger
                                                                                   κ_i (more front-loading)
                                                                                   only helps, per the sign
                                                                                   of the correction term]
```

recovering `λ_i` via `κ² = λσ²/η` (§2.4):

```
λ_i  =  κ_i² · η_i / σ_i²          [from §2.4's definition of κ, inverted]
```

with `κ_i` taken at the constraint's minimum-satisfying value (the least front-loading, hence least
`E[C]`-cost, consistent with the stated budget `β` — the trajectory should not front-load MORE than
the risk budget requires, since front-loading is not free: it raises `E[C]` per §2.3), computed
uniformly across the liquid core as a book-level or class-level RED choice (not per-instrument —
§10), and a separately (also RED, also class-level) larger `κ` applied to the illiquid tail per §3's
front-loading discussion. This keeps `λ` a governance choice derived from a STATED variance budget
`β` (not a fitted parameter) while making the promised "closed-form relationship between `λ`, `σ`,
and `κ`" an actual written derivation rather than an assertion that one exists — see §10 parameter
governance for the RED/AMBER bucketing of `β` vs. `κ` vs. `λ` itself.

**Edge case, stated explicitly.** If `w_target,i/x_i` is already large enough that
`(3β/Δ)·(w_target,i/x_i)² ≥ 1` (i.e. the trade is small enough relative to the target weight that
even the TWAP/`κ=0` variance already satisfies the budget), the right-hand side of the `κ_i²`
inequality is negative and the constraint is satisfied by `κ_i = 0` (pure TWAP) — no front-loading
is required. This is the expected, non-degenerate behavior for the liquid core (§3): most weeks,
most names do not need `κ>0` at all.

**Worked numeric example — carrying the derivation to a concrete illustrative `λ_i`, tagged (previously
the derivation stopped at the symbolic `κ_i²`/`λ_i` formulas without landing on a number, unlike
LEVERAGE §3's `G_max=1.5` worked example, which resolves its own governing constant to a checkable
figure against the SAME 28%-annualized-vol baseline this spec cites at §3 below).** Every numeric
input in this box is bracket-tagged, following SYSTEM.md §5.3–5.4's `[MEASURED]`/`[ILLUSTRATIVE]`/
`[DERIVED]` convention, so the DERIVATION (exact, general) is never confused with the PLUG-IN NUMBERS
(illustrative, pending a real fit at deploy time):

```
sigma_i (weekly)     ~ 28% / sqrt(52.18) ~ 3.9%                    [MEASURED, docs/EDGE.md — the SAME
                                                                     28%-annualized book-level figure
                                                                     LEVERAGE.md §3/§4.2 uses; this
                                                                     example borrows the book-level
                                                                     figure as a per-name proxy for a
                                                                     representative liquid-core name,
                                                                     not a per-name-fitted vol]
sigma_i (over Delta) ~ sigma_i(weekly) * sqrt(Delta / (7*24h))     [DERIVED — sqrt-time scaling of a
                     ~ 3.9% * sqrt(2h / 168h) ~ 3.9% * 0.109       weekly vol down to a Delta=2h window,
                     ~ 0.42%                                        the SAME time-aggregation convention
                                                                     flagged as condition (iii) of the
                                                                     lambda/rho reconciliation sketch
                                                                     above — stated here explicitly
                                                                     rather than left implicit]
Delta                = 2 hours                                    [ILLUSTRATIVE — midpoint of §2.1's
                                                                     stated 1-4h AMBER range]
x_i / w_target,i     ~ 0.30                                        [ILLUSTRATIVE — a representative
                                                                     week's required-trade-to-target-
                                                                     weight ratio for a name that moved
                                                                     one quintile-rank band; not fitted
                                                                     from a realized rebalance log, since
                                                                     no live trajectory has ever run —
                                                                     this pillar's fundamental structural
                                                                     limitation, restated at §14's closing
                                                                     bullet]
beta                 = 0.01                                        [ILLUSTRATIVE — the stated risk-budget
                                                                     example already used above]
eta_i                ~ COST's fitted crypto-major k_amihud * sqrt(Amihud_i)  [TEAL, COST §5.3 — no single
                                                                     illustrative NUMBER is plugged in
                                                                     here because eta_i is READ per-name,
                                                                     per-week at deploy time (Invariant
                                                                     N-exec); left symbolic deliberately,
                                                                     unlike sigma_i/Delta/x_i/beta above,
                                                                     to avoid inventing a COST-owned fit
                                                                     value this spec has no authority to
                                                                     state as a number]

Applying the kappa_i^2 solve above at w_target,i/x_i = 1/0.30 ~ 3.33:

  kappa_i^2  >=  (45 / (17 * Delta^2)) * [ 1 - (3*0.01/Delta) * 3.33^2 ]

  with Delta in the SAME units as sigma_i is quoted in (weekly-equivalent fraction-of-week units, i.e.
  Delta/7days = 2h/168h ~ 0.0119 weeks) so the constraint is dimensionally consistent with Section 2.4's
  sigma^2 being a PER-WEEK-EQUIVALENT price variance rate:

  kappa_i^2  >=  (45 / (17 * 0.0119^2)) * [ 1 - (0.03/0.0119) * 11.09 ]
             =   18,650 * [ 1 - 27.98 ]                             [DERIVED, from the tagged inputs above]
             =   18,650 * (-26.98)  <  0

Since the bracket is negative, the constraint is satisfied by kappa_i = 0 (pure TWAP) — this
representative liquid-core name, at this illustrative x_i/w_target,i ~0.30 and beta=0.01, lands in
the Section 2.5 EDGE CASE (no front-loading required), which is the EXPECTED, non-degenerate
liquid-core behavior Section 3 describes, not a failure of the derivation to produce a number:
lambda_i = kappa_i^2 * eta_i / sigma_i^2 = 0 in this illustrative regime. A representative
ILLIQUID-TAIL name would need a materially SMALLER x_i/w_target,i (a bigger required trade relative
to its target weight — the tail names are exactly the ones with the LARGEST relative rebalancing
need per docs/EDGE.md's universe-size finding) before the bracket turns positive and kappa_i > 0
binds; this spec does not fabricate a second illustrative x_i/w_target,i for the tail case, since
no realized tail-name rebalance ratio exists to ground one (the same "no live trajectory has ever
run" limitation this box's x_i/w_target,i tag notes, and §14's closing bullet restates for the file
as a whole) — the qualitative conclusion (illiquid tail needs kappa_i > 0 more often than the liquid
core) is Section 3's finding, stated qualitatively there for exactly this reason.
```

**The shipped RED default: `λ = 0` (risk-neutral / TWAP limit) — a designed knob, not an
unassigned constant.** The derivation above maps a stated risk budget `β` to `λ` (via
`lambda_from_budget`), but the *default* `λ` this pillar ships is `λ = 0`. This is a
deliberate, spec-traceable governance choice, not an omitted number:

- **It is the exact limit §2.4 derives.** `λ → 0 ⇒ κ = 0 ⇒ pure TWAP` — the risk-neutral
  Almgren–Chriss trajectory (Almgren & Chriss 2001, §5, the `λ=0` limit cited at §14). The
  default is not a guessed magnitude; it is the *one* value on the `λ`-axis that is a closed-form
  identity rather than a preference, so it is the only honest place to sit the default before a
  book states its own `β`.
- **It reproduces the recorded backtest.** `docs/EDGE.md`'s honest net Sharpe (~0.8–1.0) / CAGR
  (~20–25% @ 100% gross) were produced under the flat 13bp `round_trip_friction` full-rebalance
  assumption (§5.1, §7.3) — i.e. *no* front-loading, the `λ=0` / TWAP baseline. Any positive
  default would silently execute a trajectory the backtest never priced, so `λ = 0` is the value
  that keeps the live default consistent with the number in `docs/EDGE.md`. (This is the same
  §2.5 worked example's conclusion: the representative liquid-core name lands in the `κ = 0` edge
  case at `β = 0.01`, so `λ = 0` is also where the derivation itself points for the liquid core.)
- **A positive default is not free.** Per §2.3, front-loading (`λ > 0`) trades `Var[C]` down at
  the cost of raising `E[C]`; shipping `λ > 0` by default would pay expected cost the book has
  not asked to pay. The neutral default declines to spend that budget until a book states `β > 0`.

The injection path is retained in full: a book that states a positive risk budget `β` obtains its
`λ` from `lambda_from_budget(β, …)` and injects it (`ExecutionDefaults(lambda_=…)`), exactly the
RED governance path §10 describes. `λ` is therefore a *documented illustrative RED default with a
stated rationale* (a designed parameter), not an unexplained magic constant — the honest form the
“no invented numbers” discipline requires for a user-risk-preference knob.

---

## 3. TWAP / VWAP as the risk-neutral floor, and when the risk-neutral floor is CORRECT for this
book

Given the honest net Sharpe (~0.8–1.0) is earned on a **weekly** signal with a `~52-week/yr` decay
horizon, execution risk inside a several-hour window `Δ` is a second-order concern relative to the
week's own realized vol (crypto realized vol ≈ 28% annualized at 100% gross, `docs/EDGE.md`) — the
signal does not decay meaningfully over hours. This means the DEFAULT operating point for this
pillar is **`λ` small but strictly positive** — close to, but not exactly, the TWAP/VWAP
risk-neutral floor:

- Pure TWAP (`λ=0`, clock-time uniform) is used as the **baseline/default tactic** for the ~35–45
  liquid majority of the universe (§4 breadth) where COST's fitted `η` is small and the temporary
  impact of trading over `Δ` hours is negligible relative to a week's expected move.
- A small positive `λ` (front-loaded, §2.4) is applied to the **illiquid tail** of the universe —
  the same names flagged in `docs/EDGE.md`'s universe-size finding as carrying disproportionate
  cost/short-availability risk — because for those names `η` (temporary impact) is large enough
  that `Var[C]` from lingering unexecuted materially competes with `E[C]` from trading fast.

This is a genuinely different regime from the old directional-breakout EXECUTION problem (a single
urgent entry racing a breakout) — there is no urgency asymmetry here, both legs (long top quintile,
short bottom quintile) are equally patient, which is why the risk-neutral floor is a reasonable
default rather than a strawman.

---

## 4. The no-trade band (proportional transaction-cost portfolio control)

### 4.1 The problem restated

`accounting.py`'s backtest turnover is `0.5 · Σ_i |w_target,i − w_carried,i|` — it trades **every**
nonzero drift every week, however small. In a live book with real per-unit round-trip cost
(`round_trip_friction()` = 13 bps, §5.1 below), trading a name whose target weight moved by, say,
0.1% of NAV costs more in fees+spread+impact than the expected marginal benefit of being exactly
on-target rather than close-to-target. A **no-trade band** formalizes "close enough, don't trade."

### 4.2 Cube-root law derivation (cited, not re-derived from scratch — a genuine closed-form result)

Constantinides (1986) and Davis & Norman (1990) study portfolio choice with **proportional**
transaction costs (a cost linear in the traded dollar amount — exactly the fee+half-spread portion
of `round_trip_friction`, though not the impact portion, which is *convex*, §4.4 caveat below). The
classical result: for a single risky asset with proportional cost rate `θ` (round-trip cost per
unit traded) and a risk-averse investor with CRRA-style local risk aversion `ρ` at the frictionless
optimum weight `w*`, the optimal policy is a **no-trade region** `[w* − b, w* + b]` — trade only
enough to return to the *boundary* of the band when the weight drifts outside it, never trade back
to the exact point `w*` (that would be trading with zero benefit at strictly positive cost every
instant the price moves). The band half-width scales as:

```
b  ∝  θ^(1/3)                    (Constantinides 1986; Davis & Norman 1990)
```

**Why the cube root — worked dimensional matching, shown to closure (not just gestured at).** Near
the frictionless optimum `w*`, the investor's expected-utility loss from holding a weight `w* + ε`
instead of `w*` is, by a second-order Taylor expansion of the (smooth, concave) value function
`V(w)` around its maximum, **quadratic** in the deviation:

```
loss(ε)  =  V(w*) − V(w* + ε)  ≈  ½ · |V''(w*)| · ε²                (first-order term vanishes at the optimum)
```

Model the drift `ε_t` of the carried weight away from `w*` as a driftless diffusion with
instantaneous variance rate `σ_w²` (the book weight's own drift volatility — an estimator for this
exact quantity is given below, not left undefined): `dε_t = σ_w dB_t`, `ε_0 = 0`. Let `T(b)` be the
expected first-passage time for `|ε_t|` to reach the band edge `b`, starting from `ε=0`. For a
zero-drift Brownian motion, this is the classical exit-time result `E[T(b)] = b²/σ_w²` (standard
Brownian first-passage identity: `E[τ_b] = b²/σ_w²` for `dε=σ_w dB`, `τ_b = inf{t: |ε_t|=b}` —
e.g. Karatzas & Shreve 1991, Ch. 3). The accumulated quadratic opportunity cost over that expected
waiting time is therefore:

```
Accumulated loss until the band is hit  ≈  ½|V''| · b² · (fraction of the waiting time spent, on
                                             average, at deviation ~b)  ~  ½|V''| · b² · const
```

(the constant absorbs the exact shape of the expected time-integral of `ε_t²` over `[0, τ_b]`,
which for a driftless diffusion started at 0 and stopped at `|ε|=b` is `O(b²)·O(T(b))`-order by
scaling, i.e. `∝ b²·(b²/σ_w²) = b⁴/σ_w²` for the ACCUMULATED (time-integrated) loss rather than the
instantaneous one — this is the `∫ε² dt` object, not `ε²` alone). **This is the step the earlier
draft of this spec asserted without showing:**

```
Trading is worthwhile once the ACCUMULATED opportunity cost equals the fixed linear reset cost:

  θ  ~  |V''| · b⁴ / σ_w²                                    [matching accumulated loss to θ]

Solve for b:

  b⁴  ~  θ · σ_w² / |V''|
  b   ~  ( θ · σ_w² / |V''| )^(1/4)
```

This is the **Davis & Norman (1990) HJB free-boundary result's actual exponent for the ACCUMULATED
first-passage matching above** — note this closes to a `b ∝ θ^(1/4)` scaling under the
`∫ε² dt ~ b⁴/σ_w²` accumulated-loss accounting, which is a DIFFERENT (and, worked this way,
internally consistent) exponent from the literature's canonical **cube-root** (`θ^(1/3)`) result.
The discrepancy is real and is flagged here rather than papered over: Davis & Norman (1990)'s actual
free-boundary derivation solves the full HJB variational inequality (not the heuristic
first-passage matching sketched above) and its rigorous conclusion — which this spec adopts as the
operative result, per the primary literature, not the heuristic re-derivation above — is the
cube-root law:

```
b  ∝  θ^(1/3)                    (Davis & Norman 1990, Theorem 2.1; Constantinides 1986 for the
                                   single-risky-asset special case)
```

**Why this spec cites rather than fully re-derives.** The heuristic argument above is included
specifically so the gap between "plausible dimensional matching" and "the actual proven result" is
visible: a first-passage/quadratic-loss heuristic like the one just shown is illustrative of *why a
sub-linear power of θ is expected at all* (linear cost, quadratic local loss ⇒ some fractional power
less than 1), but it does **not**, by itself, reproduce the literature's exponent — getting the
exact `1/3` requires solving the HJB variational inequality's smooth-pasting conditions at the free
boundary (value-matching AND smooth-pasting, two conditions pinning down both `b` and the value
function's boundary behavior simultaneously), which is a genuinely different and harder calculation
than matching a single accumulated-loss scale. That full free-boundary solve is Davis & Norman
(1990)'s original contribution and is not reproduced here; this spec cites the **proven** `θ^(1/3)`
result (never the `θ^(1/4)` heuristic derived above, which is shown only to make the honest gap
between sketch and proof explicit) as the operative, implementable fact:

```
b_i  =  c · ( θ_i / ρ )^(1/3)
```

where `θ_i` is name `i`'s proportional round-trip cost (from `round_trip_friction()`, §5.1 — the
linear fee+half-spread portion; see §4.4 for why the convex impact portion is handled separately),
`ρ` is a CRRA-style LOCAL risk-aversion coefficient (units: dimensionless, `-w·V''(w)/V'(w)`
evaluated at `w*`, standard CRRA definition — **NOT** the same object as `λ` in §2.3/§2.5, see the
explicit non-identity note immediately below), and `c` is a RED dimensionless constant absorbing the
terms not pinned down by the scaling argument alone (fitted once, offline, to the book's own
realized weight-drift volatility `σ_w`, defined precisely below, never per-instrument-tuned on
realized P&L — same discipline as COST.md's Invariant C-cfg).

**`λ` (§2.3/§2.5) and `ρ` (here) are NOT the same knob — corrected from an earlier draft.** An
earlier version of this spec asserted `ρ` "is the SAME `λ`-derived risk-aversion input used in
§2.5." That claim does not survive a units check and is retracted here. `λ` (Almgren-Chriss, §2.3)
has units of **dollars of cost-variance tolerated per dollar of expected-cost reduction** — it is a
trade-off rate inside a mean-variance objective over EXECUTION COST (a $-denominated quantity) as a
function of a TRAJECTORY (a control path over a bounded window `Δ`). `ρ` (Constantinides/Davis-Norman,
here) is a **dimensionless local curvature ratio** of a portfolio-choice value function with respect
to PORTFOLIO WEIGHT (a unit-free quantity) evaluated at a stationary point, over an unbounded
horizon. These live in different objectives, different state variables, and different units; there
is no dimensional transform that makes them interchangeable, and this spec no longer claims one.
**What IS shared, correctly, is only the ECONOMIC PRINCIPLE that both are risk-aversion governance
knobs, RED, never fitted on realized P&L (§10)** — they are managed under the same governance
discipline, not the same numeric value or the same formula. Any future work that wants a single
calibrated risk-aversion input for both the trajectory and the band must derive that mapping
explicitly (e.g. via a shared underlying CRRA utility that induces both a mean-variance
execution-cost trade-off AND a portfolio-choice curvature — a nontrivial derivation, not attempted
here) rather than asserting the two are already equal, as the retracted claim did.

**What a real unification would have to establish, sketched (not solved) — making the non-goal
falsifiable rather than a bare assertion of "unresolved."** A single CRRA utility `U(W) = W^(1−γ_CRRA)
/ (1−γ_CRRA)` induces a value function `V(w)` over portfolio weight whose local curvature at the
frictionless optimum `w*` is `ρ = −w*·V''(w*)/V'(w*)`, a function of `γ_CRRA` and the LOCAL
mean-variance parameters of the return distribution at `w*` (a standard portfolio-choice identity —
not derived here). Separately, the SAME `U` induces an Almgren-Chriss-style objective over execution
cost only if the trader's utility is applied not to terminal wealth directly but to the wealth
REALIZED AT THE END OF THE EXECUTION WINDOW `Δ`, i.e. `E[U(W_0·(1 − C(x)))]` expanded to second order
in `C` — this expansion produces a `λ` that is proportional to `γ_CRRA·W_0`, i.e. `λ` would need to
scale with the SAME curvature parameter `γ_CRRA` AND with the trader's total wealth `W_0` (a
dollar-denominated scale factor `ρ` alone does not carry, since `ρ` is unit-free and `λ` is not). A
valid unification therefore requires, at minimum, three conditions to hold SIMULTANEOUSLY, none of
which this spec establishes: (i) the same `γ_CRRA` parameterizes both the portfolio-choice value
function `V(w)` (giving `ρ`) and the terminal-wealth utility `U(W)` (giving `λ` via the
second-order-in-`C` expansion above); (ii) the wealth-scale factor `W_0` linking `λ` to `γ_CRRA` is
itself pinned to a stated NAV convention consistent with how `x_i` is already NAV-normalized in §2.1
(fraction of book NAV, not dollars) so the units reconcile; (iii) the LOCAL mean-variance parameters
`(μ, σ²)` entering `ρ`'s definition at `w*` (the book's WEEKLY return distribution, LEVERAGE.md §2's
object) and the `(σ, η)` entering `λ`'s Almgren-Chriss objective at the SUB-WEEKLY execution-window
scale (§2.2's object) are related by a stated, checkable time-aggregation rule (e.g. how weekly `σ²`
scales down to an `Δ`-hour `σ²` under the `sqrt(time)` convention already used at §3's "crypto realized
vol ≈28% annualized" restatement) rather than silently treated as the same number at two different
horizons. None of (i)–(iii) is attempted in this spec — each is a genuine, separately falsifiable
research question (a future draft could test (i) by fitting `γ_CRRA` from LEVERAGE's own Kelly-fraction
policy and checking whether the IMPLIED `λ` under (ii)-(iii)'s scaling reproduces the RED `λ` this spec
sets from a risk budget in §2.5 — a check this spec does not perform) — this sketch exists so a future
attempt at unification has a concrete, checkable target rather than an open-ended "derive a mapping."

**`σ_w,i` — the no-trade band's calibration input, defined (not left implicit).** `σ_w,i` is the
per-name **realized weight-drift volatility** — the standard deviation of name `i`'s week-over-week
*drift* `w_carried,i,t − w_target,i,t-1` (the quantity that accumulates between rebalances), **not**
`portfolio.realized_vol`'s statistic, which is the trailing standard deviation of name `i`'s daily
*price* returns (`src/convexity/portfolio/portfolio.py::realized_vol`, confirmed by inspection: `np.std` of
`panel.close`-derived daily returns over a 28-day trailing window). An earlier draft of this spec
conflated the two ("already estimated causally by `portfolio.realized_vol`, reused verbatim") — that
is corrected here: price-return vol and weight-drift vol are different statistics because weight
drift is price-return vol **passed through the vol-scaled weighting function's own convexity**
(`portfolio.py`'s inverse-realized-vol weighting means a given price move translates into a
DIFFERENT weight move depending on where the name sits in the current vol-scaled allocation — the
map from price return to weight drift is not the identity, so their volatilities differ by a
Jacobian factor that is not tracked anywhere in the codebase today). The TEAL estimator this spec
specifies (new, filling the gap):

```
σ_w,i  =  EWMA-std of { w_carried,i,t − w_target,i,t-1 : t = t_0-1, t_0-2, …, t_0-52 }
          (trailing ~52-week EWMA, half-life a RED constant — same EWMA discipline as the rest of
          this codebase's causal vol estimators, e.g. COST §5's ADV EWMA; computed causally, using
          only weights already realized strictly before t_0, satisfying P7)
```

This is a NEW estimator this spec introduces (not an existing function) — implemented in
`components/no_trade_band.py` (§11) alongside the band itself, reusing `portfolio.realized_vol`'s
EWMA/window conventions for consistency of style but computing a genuinely different input series
(weight drift, not price return). A test (§13) must assert `σ_w,i ≠ portfolio.realized_vol(i)`
byte-for-byte in general (they are allowed to coincide only by numerical accident, never by
construction) to guard against this exact conflation recurring.

### 4.3 Applying the band per name

For each name `i` at the Friday-close rebalance:

```
trade_i  =  0                                          if |x_i| = |w_target,i − w_carried,i| ≤ b_i
         =  x_i − sign(x_i)·b_i                        otherwise (trade only to the band edge)
```

This directly reduces live turnover below the backtest's full-rebalance turnover — the backtest's
`Turnover = 0.5·Σ|x_i|` becomes `Turnover_live = 0.5·Σ max(0, |x_i| − b_i) ≤ Turnover_backtest`
name-by-name and in aggregate (**Invariant E-monotone**, P4). Because names with larger `θ_i`
(wider spreads, thinner books — the illiquid tail flagged in `docs/EDGE.md`'s universe-size finding)
get proportionally WIDER bands, the no-trade band naturally trades the illiquid tail *less*
aggressively than the liquid core, which is a second, independent channel (beyond §3's
trajectory-shaping) by which this pillar should reduce the illiquid-tail cost drag that the honest
50-vs-100 universe comparison flagged (`docs/EDGE.md`: "~15 of the +50 added names are effectively
unshortable" and "+2.9%/yr drag from the added tail" — the no-trade band does not eliminate that
drag, since it is a shortability/thin-book problem, not a rebalance-frequency problem, but it does
not make it worse and should measurably shrink it for the *retained* N≈50 universe's own thin
members).

### 4.4 Assumption / limitation: proportional vs. convex cost

The cube-root result is proved for **proportional** (linear-in-size) cost. `round_trip_friction()`
is linear (`2·(taker+half_spread+impact)·turnover`, a FIXED per-unit-turnover constant, §5.1) — so
the band derivation applies cleanly to the fee+half-spread portion. The *forecast* impact term used
live (COST §5.3's square-root Amihud law, `impact_frac_i = k_amihud·sqrt(Amihud_i·Notional_leg)`,
convex in trade size via the `sqrt(Notional_leg)` term) is not linear, which technically breaks
the clean Constantinides/Davis-Norman proportional-cost assumption. This spec's position: the
cube-root band is applied using `θ_i` = the LINEAR (fee+half-spread) component only, as a
**conservative, tractable approximation** — the convex impact cost only makes trading small drifts
*less* attractive (impact cost is locally negligible for small trades since `sqrt(Notional_leg)→0`
as trade size→0, so ignoring it in the band derivation cannot make the band anti-conservative, i.e.
too narrow). This is stated as an explicit approximation, not silently assumed (Invariant B-band
below).

---

## 5. Salvaged math from COST.md (re-derived under the new risk unit)

### 5.1 The flat friction floor (already implemented, `src/convexity/cost/accounting.py`)

```
round_trip_friction = 2 · (taker + half_spread + impact)
                     = 2 · (5.0bp + 1.0bp + 0.5bp) = 13.0 bp     (per unit of turnover, both legs)
```

`taker` = Binance USDT-M VIP0 taker rate, read from `convexity.cost.config.fee_schedule
.BINANCE_USDM_FEE_SCHEDULE` — the SAME one fee truth COST.md §5.2 defines (`BINANCE_VIP0_TAKER =
0.0005`, `src/convexity/cost/accounting.py`). `half_spread` and `impact` are the crypto-major research
fallbacks (1.0bp, 0.5bp), matching COST §5.3/5.4's conservative fallback discipline (P6) rather than
a per-name fit, because the backtest scores the *book-level* honest Sharpe, not a per-name TCA. This
13.0 bp constant is EXECUTION's **cost floor** — the minimum a trajectory must reconcile to when
`λ→0` and the no-trade band is inactive (`b_i = 0`); anything the live trajectory adds on top
(temporary impact curvature from §2, band-skipped turnover reduction from §4) moves the realized
cost strictly at or below this floor's implied dollar cost per unit of *executed* turnover, while
the no-trade band separately shrinks *how much* turnover is executed at all (§4.3).

### 5.2 Amihud impact — reused, not re-derived (COST.md §5.3/§5.4.3, cautionary note preserved)

COST.md's fitted square-root impact model, `impact_frac_i = k_amihud · sqrt(Amihud_i · Notional_leg)`
(COST §5.3), is estimated from a real **Amihud (2002)** illiquidity measure — reused verbatim here
as the `η`/`γ` source for the Almgren-Chriss linearization (§2.2, Invariant N-exec). The cautionary
dead-end is derived and owned by **COST.md §5.4.3** specifically (not §5.4 generally — COST §5.4 is
the dimensional-check derivation for the Amihud-to-impact-bps conversion; §5.4.3 is the named
Corwin-Schultz subsection), and is cross-referenced identically in `docs/specs/UNIVERSE.md` and
`docs/specs/PORTFOLIO.md` §7.4 — all three specs point at the same COST-owned derivation, none
re-derive it. Restated here because EXECUTION is exactly the pillar that would be tempted to
re-derive its own microstructure cost model: **Corwin–Schultz on 1-minute-aggregated daily high/low
was tried and rejected** (COST Invariant CS-REJECT, `src/convexity/cost/friction/validate/corwin_schultz_reject.py`
kept as a standing regression test) — it conflates realized intrabar volatility with the bid-ask
spread and produced ~135bp / ~40%-drag garbage, an order of magnitude off the real Amihud-based
cost. Any future change to the impact estimator underlying this pillar's `η`/`γ` must go through
COST.md's own validation battery (COST §12), not be re-fit locally inside EXECUTION.

### 5.3 DSR / CPCV — reused verbatim for the execution-adjusted series (SYSTEM.md §5, §4)

The post-execution honest Sharpe (the weekly net-return series AFTER live shortfall is subtracted,
§7 below) is validated with the SAME machinery `src/convexity/system/xsec_validate.py` and `cpcv.py` already
apply to the frictionless backtest series: `convexity.system.validate.deflated.DeflatedSharpe` — PSR
per SYSTEM §5.2's series/units contract (the `path_returns`/`ppy`/`SR_pp`/`n_obs` pinned inputs) and
§5.3's worked DSR example (Vintage-2 real-Amihud-cost numbers: `DSR=0.807`, `PSR=0.994` against
`SR*=0`), DSR gate per SYSTEM §6.1's promotion conjunction (`DSR ≥ 0.95` RED threshold) — and
`convexity.system.validate.cpcv.SystemCPCV` per SYSTEM §4.1–§4.2 (combinatorial purged CV, `N=8`
groups / `k=2` test groups / 28 paths / 1-week embargo — the exact configuration `docs/EDGE.md`
cites: mean OOS path Sharpe ≈0.91 vs naive single-path 0.98, `frac_pos` 0.85–1.00). **SYSTEM.md §7
is unrelated** — it is Kunsch (1989)/Politis & White (2004) block-bootstrap block-length selection
for confidence-interval widths, sharing no symbols or gate logic with PSR/DSR/CPCV; a reader
following this pointer to SYSTEM §7 to verify DSR machinery would land on the wrong section, so this
spec now cites SYSTEM §5 (DSR/PSR) and §4 (CPCV/PBO) explicitly and separately. EXECUTION does not
reimplement PSR/DSR/CPCV math; it only supplies a DIFFERENT input series (net-of-realized-shortfall
instead of net-of-backtest-friction-constant) to the same validated functions (§9).

---

## 6. Calendar-triggered vs. drift-triggered rebalancing

### 6.1 Calendar-triggered (PRIMARY)

The validated edge is a **Friday-close, `rebalance_days=7`** cadence (`src/convexity/portfolio/engine.py`'s
`XSecConfig.rebalance_days` default; `docs/EDGE.md`'s entire CPCV/DSR/walk-forward battery is scored
on this cadence). EXECUTION's calendar trigger fires the trajectory of §2 **once per week, at the
Friday close**, computing `x = w_target − w_carried` from that week's signal output. This is the
ONLY trigger validated by any backtest evidence in this repository and is therefore the pillar's
default, always-on behavior.

### 6.2 Drift-triggered (DEFENSIVE ADDITION, REGIME-gated only)

A drift-triggered rebalance — re-checking `|w_current,i − w_target,i|` against the no-trade band
*between* Friday closes and trading early if breached — is explicitly **not** validated by any
number in `docs/EDGE.md`: every Sharpe, CAGR, drawdown, and CPCV path cited was produced by a
strictly weekly cadence. Firing an off-cycle drift trade changes the realized turnover and cost
profile in a way no backtest has scored. Per P5, this mechanism is therefore restricted to:

- **Only armed when REGIME's defensive-overlay condition is active** (a REGIME-owned state; this
  pillar does not define what constitutes "defensive" — it only consumes a boolean/`g_t`-style flag
  from REGIME and, when set, permits an out-of-cycle trajectory to fire).
- **Only for REDUCING gross exposure**, never for opportunistically re-entering or increasing a
  position ahead of schedule — i.e. the drift trigger is a risk-off valve, not a second alpha
  cadence. This mirrors the overlay evidence in `docs/EDGE.md`/`overlay.py`/REGIME.md §5.3-§5.4: the
  causal drawdown-stop overlay caps DD at ~24% but collapses Sharpe **from 0.57 (its level WITH the
  beta-hedge overlay already applied, REGIME §5.3) to 0.10**, and funding income from ~12.9%/yr to
  ~0.6%/yr, when applied to the (unrelated) carry book (cited here with the full starting-point
  precision REGIME.md/PORTFOLIO.md §14 carry, not the bare "collapses to 0.10" restatement an earlier
  draft of this spec used) — the lesson generalizes: any defensive mechanism that fires eagerly
  destroys the edge it's protecting, so this trigger must be rare and REGIME-gated, not a parallel
  weekly-equivalent cadence. Note this finding is from the CARRY sleeve (structurally different: a
  persistent ~70-75% regime-bleed drawdown, not a dodgeable tail event, per REGIME §5.3) — it is
  cited here only as a qualitative caution about eager defensive mechanisms in general, not as a
  quantitative prediction of what a DD-stop would do to THIS (momentum) book, which REGIME §5.4
  explicitly requires its own separate validation before any transfer (Invariant STOP-XFER).

> **Invariant CAL-PRIMARY.** A drift-triggered rebalance fired without an active REGIME defensive
> flag is the bug this invariant forbids. A unit test asserts the drift trigger is a no-op whenever
> `regime.defensive == False`, regardless of how large the observed drift is.

### 6.3 What "drift" means here, precisely

Drift is measured against the CURRENTLY-VALID `w_target` (the most recent Friday close's target),
not against some interpolated intra-week target — there is no intra-week target, since the signal
itself (`signals.py`) is only computed at the weekly cadence. Drift is pure price-path drift of
`w_carried` away from last Friday's `w_target`, exactly the quantity `accounting.py`'s
`w_carried`-drift-into-next-rebalance already computes for the *next* scheduled Friday; the
drift-triggered path only asks whether that same quantity has already breached the no-trade band
(§4) before Friday arrives.

### 6.4 Live turnover ≤ backtested turnover (restating P4/Invariant E-monotone concretely)

```
Turnover_live_annualized  =  Σ_{weeks} 0.5 · Σ_i max(0, |x_i,week| − b_i)      · 52.18/yr
Turnover_backtest_annualized (accounting.py)  =  Σ_{weeks} 0.5 · Σ_i |x_i,week|  · 52.18/yr
```

Since `max(0, |x_i| − b_i) ≤ |x_i|` for every `b_i ≥ 0` termwise, `Turnover_live ≤ Turnover_backtest`
always, with equality only in the degenerate `b_i = 0 ∀i` case. Any live/backtest turnover
comparison in a monitoring dashboard MUST report this as a one-sided inequality, never imply the
live number could exceed the backtest's (Invariant E-monotone) — an observed live turnover ABOVE
the backtested figure is a bug signal (e.g. a mis-armed drift trigger firing outside its REGIME
gate, §6.2) and should page, not just log.

---

## 7. Implementation shortfall and the feedback loop into COST

### 7.1 Definition (Perold 1988)

Perold (1988)'s implementation shortfall compares the realized outcome of an actually-executed
trajectory to the paper return of the DECISION — here, the Friday-close print used to compute
`w_target`. For name `i`'s trade in this rebalance:

```
IS_i  =  (P_decision,i − P̄_fill,i) · sign(x_i) · |x_i|·NAV        (long trade: pay MORE than decision price = cost)
```

where `P_decision,i` is the Friday-close print (`panel.close` at row `d`, `src/convexity/panel/panel.py`
— the exact price the backtest's `_asset_week_return` uses as the week's starting mark) and
`P̄_fill,i` is the realized, notional-weighted average fill price across all child orders executed
for name `i` during `[t_0, t_0+Δ]`. Summed and normalized by total traded notional, this decomposes
(Perold 1988; standard TCA convention) into:

```
IS_total = (explicit cost: fees, exactly COST §5.2)
         + (realized half-spread paid: COST §5.3's realized analogue)
         + (realized market impact: temporary + permanent, §2.2)
         + (delay/opportunity cost: price drift between t_0 and the first child fill)
```

### 7.2 Feedback into COST (connects to OPS reconciliation) — CORRECTED, no fabricated seam

**Correction from an earlier draft.** An earlier version of this section cited "COST.md §5.4's own
statement of scope" with a quoted sentence ("COST does not model order-book execution... it ingests
EXECUTION's realized number and forecasts the expected impact for the gate") and a target module
`cost/facts/execution_adapter.py` (`ExecutionFacts`, "COST §1 table"). **Neither the quote nor the
module exists.** Verified directly against `docs/specs/COST.md`: no `execution_adapter.py`, no
`ExecutionFacts` class, and no such sentence appear anywhere in that file (grep-confirmed). More
significantly, **COST.md §11 (Non-goals) states explicitly: "there is no EXECUTION pillar in the
xsec-momentum architecture (the book rebalances with a batch of taker orders at Friday close...)."**
COST.md was written, and still stands, on the premise that EXECUTION does not exist as a separate
pillar — it has no ingestion seam for a realized-fill feedback series because it was never asked to
have one. This is a genuine, unresolved cross-spec inconsistency introduced by this pillar's
existence, not a citation this spec can paper over by inventing a plausible-sounding module name.

**What this spec commits to instead (buildable, not fabricated).** Rather than assert a COST-side
ingestion seam that does not exist, EXECUTION's realized `IS_total` feeds forward through the ONE
real, existing monitoring seam in this repo — `OpsEngine.on_obs(obs: object) -> OpsDirective`
(`core/protocols.py`, confirmed exact signature) — as a `StreamObs`-shaped observation (OPS's
existing causal-observation contract, OPS §3, out of scope for this file to redefine). OPS compares
the week's forecast cost (`accounting.round_trip_friction`'s 13bp assumption applied to that week's
turnover) against the realized `IS_total` this pillar computes, and a persistent, growing gap is
exactly the kind of drift OPS's monitor-only surface is designed to catch — EXECUTION emits the
observation, OPS decides run/degrade/halt (P2: EXECUTION reports, never halts). **Whether and how
COST.md should be amended to add its own online re-calibration seam for `k_amihud`/`Amihud_i` from
EXECUTION's realized series (the TEAL online-update mechanism the earlier draft asserted already
existed) is explicitly flagged here as OPEN, cross-cutting, COST-owned future work — not something
EXECUTION.md can unilaterally specify into COST's contract.** Until COST.md is amended to accept
such a feed (out of scope for this file, which may only write to `docs/specs/EXECUTION.md` and
`docs/ARCHITECTURE.md`), the realized-impact-drift signal's only actionable consumer is OPS's
monitor-only surface, not a COST-side re-fit.

### 7.3 Why this loop matters for the headline number

`docs/EDGE.md`'s honest net Sharpe (~0.8–1.0) and CAGR (~20–25% @ 100% gross) are backtest numbers
built on the FLAT 13bp `round_trip_friction` constant (§5.1), which is itself already a real,
Amihud-grounded cost estimate (not the light-cost proxy that produced the inflated 1.6/65% headline)
— but it is still a **book-level average constant**, not a per-name, per-week REALIZED fill outcome.
The whole point of closing the loop in §7.2 is to confirm, live, that realized `IS_total` tracks the
13bp assumption rather than silently drifting wider (e.g. if the no-trade band under-trades a name
that then gaps, or if the illiquid tail's actual impact exceeds the fitted `k_amihud`/`Amihud_i`).
This is the
concrete mechanism by which "frictionless instantaneous weekly fills" (the backtest assumption) gets
falsified or confirmed in production.

---

## 8. Interaction with REGIME (defensive overlay only)

EXECUTION consumes exactly one boolean/state input from REGIME: whether the defensive overlay is
active (§6.2). It does not consume REGIME's `g_t` multiplier for sizing (that is LEVERAGE's
consumption path per `core/protocols.py`'s composition graph) — EXECUTION only asks "is an
off-cycle risk-reducing trajectory permitted right now," never "how much gross should the book
run." This keeps the same firewall discipline COST.md's P2 established for its own consumers:
EXECUTION reads one narrow field off REGIME's state, not the whole `RegimeState` object.

---

## 9. Validation: does the execution-adjusted series survive the same battery?

The deliverable check for this pillar is not "did the trajectory minimize `E[C]+λVar[C]`" in
isolation — it is "does the realized-shortfall-adjusted weekly net-return series still pass the
SAME validation battery `docs/EDGE.md` reports for the frictionless backtest series." Concretely,
re-run, on the execution-adjusted series:

- **Purged walk-forward** (`src/convexity/system/xsec_validate.py`'s `weekly_defaults` + SYSTEM's
  `PurgedWalkForward`) — no OOS collapse expected, same as the frictionless series (`docs/EDGE.md`
  §c).
- **CPCV** (`src/convexity/system/cpcv.py`, N=8/k=2/28 paths/1-week embargo) — compare
  `frac_pos`/mean-path-Sharpe on the execution-adjusted series against the frictionless baseline
  (0.91 mean path Sharpe, 0.85–1.00 `frac_pos`, per `docs/EDGE.md`). A material `frac_pos` drop
  under realistic execution is the signal that this pillar's trajectory/band design needs
  retuning (via the RED governance path, §10 — never a per-instrument tune on realized P&L).
- **DSR** (`convexity.system.validate.deflated.DeflatedSharpe`) — the execution-adjusted series is
  expected to fail DSR for the SAME reason the frictionless series does (`docs/EDGE.md` §d:
  "DSR fails ONLY on insufficient track record... PSR~0.994 is high... a track-length problem, not a
  deflation rejection") — i.e. this pillar's job is to confirm execution does not turn a
  track-length-limited PASS-pending series into a genuinely-deflated FAIL. A DSR outcome that
  degrades from "track-length-limited" to "fails on `SR_pp` itself" is a signal execution cost is
  eating real edge, not just adding noise.

This reuses `deflated.py`/`walkforward.py`/`cpcv.py` byte-for-byte (Invariant N-exec) — EXECUTION's
validation layer is a second INPUT SERIES through the existing machinery, never a new statistical
test.

**What is buildable pre-live vs. what is a post-deployment obligation (scope, stated plainly).**
Everything the trajectory/band/shortfall solver computes from `F_{t_0}`-measurable inputs — the
Almgren–Chriss schedule, the no-trade band, the `E[C]`/`Var[C]` at the solved trajectory, and the
`IS_total` *decomposition* given a realized fill vector — is fully offline-computable and is
implemented and tested now. The CPCV/DSR/purged-WF re-run *above*, however, requires the one input
this repo cannot manufacture offline: a **realized-shortfall-adjusted weekly net-return series**,
which only exists once a live trajectory has actually filled orders against real prices. No such
trajectory has run (§14's structural limitation). Therefore this re-run is defined here as a
**post-deployment validation contract, not a pre-live buildable**: on the first N live rebalance
weeks the realized `IS_total` series (§7.1) is fed through the *already-built, already-tested*
`cpcv.py`/`deflated.py`/`walkforward.py` machinery and its `frac_pos`/mean-path-Sharpe compared to
the frictionless baseline (0.91 / 0.85–1.00). Fabricating a live trajectory to score this offline
would be an invented result and is explicitly refused; the honest ceiling is that this specific
check is *deferred to first-N-weeks-live by construction*, and the offline-computable remainder of
the pillar is complete for what is buildable before a single order has been sent.

---

## 10. Parameter governance

Every parameter is exactly one bucket (same TEAL/AMBER/RED/BLUE discipline as COST.md §4 and
SYSTEM.md §10).

- **TEAL — fitted per instrument from its own past data, at runtime, causal.** The instrument's
  `η`/`γ` (read from COST's fitted `k_amihud`/`Amihud_i`, Invariant N-exec — TEAL there, consumed
  here, not re-fit); the causal volume-profile forecast `V̂` used for a VWAP-style time-change
  (§2.4); the per-name realized weight-drift volatility `σ_w,i` feeding the no-trade band's `c`
  calibration
  (§4.2 — a NEW EWMA estimator over realized weight drift this pillar owns and computes itself,
  `components/no_trade_band.py`, §11 — explicitly NOT `portfolio.realized_vol`'s price-return
  statistic; §4.2 gives the estimator and the invariant that guards against re-conflating the two);
  the causal fill-rate forecast `λ̂_fill,i` feeding §2.1's `Δ` lower-bound check (venue order-book
  depth/turnover metadata, per-instrument, causal).
- **AMBER — supplied market metadata.** The execution window length `Δ` (venue/liquidity-class
  metadata, not fitted, subject to §2.1's two explicit bounding inequalities); the sub-interval
  count `n`; whether the venue supports post-only/maker slicing (feeding a future tactic-ladder
  extension, out of scope for v1 which trades taker/IOC only, matching `accounting.py`'s current
  taker-only friction assumption).
- **RED — shared economic/modelling constants.** The risk-aversion parameter `λ` (§2.5, a stated
  risk-budget choice `β` converted via the closed-form `κ²=λσ²/η` relation, never fitted on P&L;
  the SHIPPED default is `λ = 0`, the risk-neutral / TWAP limit — a designed knob whose rationale
  §2.5 states in full: it is the closed-form `λ→0` identity of §2.4, it reproduces the `λ=0`
  full-rebalance backtest `docs/EDGE.md` reports, and it declines to pay the `E[C]` cost of
  front-loading until a book states a positive `β`; injecting a `λ>0` value via
  `lambda_from_budget(β, …)` remains the governance path for a book that has one);
  the no-trade band's CRRA-style local risk-aversion coefficient `ρ` (§4.2 — governed under the SAME
  RED discipline as `λ` but **NOT dimensionally the same object**, per §4.2's explicit non-identity
  note; a future unification is out of scope for v1); the no-trade band's dimensionless constant `c`
  (§4.2, fitted once offline to book-level realized drift vol `σ_w`, never per-instrument on
  realized P&L — same discipline as COST.md's Invariant C-cfg); the REGIME defensive-overlay arming
  threshold consumed by §6.2 (owned by REGIME, referenced here only).
- **BLUE — finite validated grids** (optional, class-level only): a small candidate set for `Δ`
  and `n`, searched offline under the SAME promotion gate as COST/SYSTEM (DSR/PSR control on the
  execution-adjusted series, §9), never per-instrument.

> **Invariant C-cfg-exec (RED unreachable from per-instrument config)** — identical discipline to
> COST.md's Invariant C-cfg: if `λ`, `ρ`, `c`, `Δ`, or `n` appears as a per-instrument field rather
> than a book-level or class-level RED/AMBER constant, that is the bug.

> **Invariant SIGMA-W-DISTINCT (new).** A test asserts `components/no_trade_band.py`'s `σ_w,i`
> estimator is a DIFFERENT code path from `portfolio.realized_vol` (different input series — weight
> drift, not price return — and, in general, a different numeric value for the same instrument at
> the same timestamp); accidentally routing the no-trade band's calibration through
> `portfolio.realized_vol` is the bug this invariant forbids (§4.2).

---

## 11. Directory layout

```text
execution/
+-- config/
|   +-- defaults.py           # ExecutionDefaults: RED lambda, band constant c, staleness windows
|   +-- venue.py              # AMBER: window Delta, slice count n, venue tactic support flags
|   +-- resolve.py            # single merge point (RED unreachable from instrument, Inv C-cfg-exec)
+-- facts/
|   +-- cost_adapter.py       # ingest COST's fitted eta/gamma (Invariant N-exec: read, never refit)
|   +-- target_adapter.py     # ingest w_target + per-name shortable flag (Inv SHORTABLE-GATE) +
|   |                         #   w_carried (accounting.py)
+-- components/
|   +-- trajectory.py         # Almgren-Chriss discrete solver (Sec 2.4): TWAP/VWAP (lambda=0) and
|   |                         #   front-loaded (lambda>0) trajectories; per-name, no cross-impact
|   +-- no_trade_band.py      # cube-root band per name (Sec 4); trade-to-band-edge only
|   +-- drift_monitor.py      # intra-week drift vs band, gated by REGIME defensive flag (Sec 6.2)
+-- assemble/
|   +-- rebalance_plan.py     # RebalancePlan: per-name child-order schedule for [t, t+Delta]
+-- consume/
|   +-- on_target.py          # ExecutionEngine.on_target (Sec 12 below) -- the one public method
|   +-- shortfall.py          # ImplementationShortfall: realized IS decomposition (Sec 7.1)
+-- validate/
|   +-- reconcile.py          # execution-adjusted weekly series -> feeds validate.py / cpcv.py (Sec 9)
|   +-- baselines.py          # naive-instant-fill vs TWAP vs band-adjusted comparison
+-- types.py                  # frozen dataclasses: RebalancePlan, ChildOrder, ShortfallReport
+-- execution_api.py          # public entry point
tests/
+-- ...                       # unit, firewall (Inv W), monotone-turnover (Inv E-monotone), sign,
                               #   TWAP-limit-recovery (lambda->0), band-conservative (Inv B-band)
```

Stack: Python 3.11+, numpy, scipy (`scipy.optimize` NOT used for a learned fit — only for solving
the closed-form discrete Almgren-Chriss recursion, which is an explicit linear-algebra solve, not an
optimizer search), pandas, pytest, NautilusTrader (child-order submission only). No ML frameworks.

### 11.1 Build contract: exact function signatures

> **SPECIFICATION — not yet implemented.** Nothing below this line is a citation to existing code.
> `src/convexity/execution/execution.py` and `research/xsec/execution/` do not exist on disk today (verified:
> no `src/convexity/execution/*`, no `src/convexity/execution/execution.py` — the prior adversarial review's
> finding stands). Every signature below is a BUILD CONTRACT — the precise, typed interface a future
> implementer must satisfy — written in the same style as the REAL, already-merged
> `src/convexity/cost/accounting.py::settle_week` and `src/convexity/portfolio/portfolio.py::quantile_weights` (read
> on the VM to match style exactly: module-level docstring stating the accounting identity, then a
> dataclass for the decomposed result, then the function with a docstring that states units and
> causality inline). Proposed module path: **`src/convexity/execution/execution.py`** (a single module,
> mirroring `accounting.py`'s single-file scope, rather than the `execution/` package sketched in
> §11's directory layout — the package layout in §11 is the eventual production shape if this
> pillar graduates into `src/convexity/execution/`; the signatures below are what a first
> research-grade implementation inside `research/xsec/` — matching how `accounting.py` and
> `portfolio.py` themselves live in `research/xsec/`, not yet under `src/convexity/` — must expose to
> be a faithful, testable realization of §2–§7 of this spec).

```python
"""Weekly rebalance execution: Almgren-Chriss trajectory, cube-root no-trade band, implementation
shortfall (EXECUTION.md Sections 2, 4, 7). SPECIFICATION ONLY -- no implementation exists yet.

Given the Friday-close required trade x = w_target - w_carried (EXECUTION.md Sec 2.1), this module:
  1. applies a per-name no-trade band (Sec 4) so small drifts are skipped, never fully rebalanced;
  2. solves the discrete Almgren-Chriss trajectory (Sec 2.4) for whatever trade survives the band;
  3. prices the realized fills against the Friday-close decision price as an implementation-shortfall
     report (Sec 7.1), the feedback series COST/OPS consume (Sec 7.2, Sec 9).

Every function below is a PURE function of its inputs (no hidden state, no incrementally-updated
running statistic) so a data restatement can be handled by a full recompute (Sec 13's data-revision
edge case) rather than an incremental patch that could silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryResult:
    """One name's solved Almgren-Chriss trajectory over `n` sub-intervals of `[t_0, t_0+Delta]`."""

    schedule: np.ndarray  # shape (n,), quantity x_{i,k} to trade in each sub-interval, signed, sums to x_i
    kappa: float  # solved front-loading rate for this name this week (kappa=0 => TWAP)
    e_cost: float  # E[C] at the solved trajectory, in NAV-return units (Sec 2.3)
    var_cost: float  # Var[C] at the solved trajectory, in NAV-return^2 units (Sec 2.3)


def solve_trajectory(
    x_i: float,
    *,
    delta: float,
    n: int,
    sigma_i: float,
    eta_i: float,
    gamma_i: float,
    lambda_i: float,
) -> TrajectoryResult:
    """Solve the discrete Almgren-Chriss trajectory for one name's required trade `x_i` (EXECUTION.md Sec 2.4).

    ``x_i`` -- required trade for this name this rebalance, fraction of NAV, signed (EXECUTION.md Sec 2.1).
    ``delta`` -- execution window length in the SAME time unit as `sigma_i` is quoted in (hours; an
                 AMBER venue-liquidity parameter, EXECUTION.md Sec 2.1 -- caller resolves this from
                 venue config, this function does not choose it).
    ``n`` -- sub-interval count (AMBER, Sec 2.1); the discrete solver, not a resampled continuous curve
             (Sec 2.4's stated reason: `n` is O(10-50), too coarse for continuous resampling to be exact).
    ``sigma_i`` -- per-unit-time PRICE-return volatility for this name over the execution window (TEAL,
                   `portfolio.realized_vol`'s statistic, NOT `sigma_w,i` -- see the explicit non-identity
                   note in EXECUTION.md Sec 2.5/Sec 4.2's box).
    ``eta_i``, ``gamma_i`` -- temporary / permanent linear impact coefficients, READ byte-identical from
                              COST's fitted `SpreadImpactModel` (`k_amihud`, `Amihud_i`), never re-fit
                              here (Invariant N-exec, EXECUTION.md Sec 2.2, Sec 10).
    ``lambda_i`` -- this name's risk-aversion parameter, RED, set from the stated risk budget `beta` via
                    EXECUTION.md Sec 2.5's closed-form kappa_i^2 solve, never fitted on realized P&L.

    Returns a ``TrajectoryResult`` whose ``schedule`` sums to `x_i` exactly (`X_0=x_i`, `X_n=0` boundary
    conditions, Sec 2.1) and whose ``kappa`` is the SOLVED value implied by `lambda_i`, `sigma_i`, `eta_i`
    via `kappa^2 = lambda_i * sigma_i^2 / eta_i` (Sec 2.4) -- `kappa=0` recovers pure TWAP exactly
    (Sec 2.4's `lambda -> 0` limit) and is the expected, non-degenerate output for the liquid core (Sec 3).

    Raises ``ValueError`` if `delta` fails EITHER of Sec 2.1's two bounding regularity conditions for
    this name (the upper bound `delta << 7 days`, checked against the caller's rebalance cadence, and
    the lower bound `delta >= n* / lambda_hat_fill_i` for this name's own causal fill-rate forecast) --
    a trajectory that cannot be costed within a valid `delta` is not silently solved anyway (P6).
    """
    raise NotImplementedError("specification only -- see EXECUTION.md Sec 2.4-2.5")


def band_width(theta_i: float, rho: float, *, c: float) -> float:
    """No-trade band half-width for one name: `b_i = c * (theta_i / rho)^(1/3)` (EXECUTION.md Sec 4.2).

    ``theta_i`` -- name i's proportional (LINEAR fee+half-spread only, excludes convex impact per the
                   Sec 4.4 conservative-approximation argument) round-trip cost, TEAL, read from
                   `round_trip_friction`-style inputs (Sec 5.1) at the per-name level.
    ``rho`` -- CRRA-style local risk-aversion coefficient (RED, dimensionless, `-w*V''(w)/V'(w)` at the
               frictionless optimum) -- NOT the same object as this pillar's Almgren-Chriss `lambda`
               (Sec 4.2's explicit non-identity note; passing a `lambda_i` value here is the bug
               Invariant C-cfg-exec's per-instrument-config discipline is designed to catch upstream).
    ``c`` -- RED dimensionless constant absorbing the terms the cube-root scaling argument alone does not
             pin down (fitted once, offline, to the book's own `sigma_w`, never per-instrument on P&L).

    Returns `b_i >= 0`, the half-width of the no-trade region `[w*_i - b_i, w*_i + b_i]` (Constantinides
    1986; Davis & Norman 1990, Theorem 2.1 -- the cube-root exponent, not the illustrative quartic-root
    heuristic Sec 4.2 shows and explicitly rejects).
    """
    raise NotImplementedError("specification only -- see EXECUTION.md Sec 4.2")


def weight_drift_vol(
    w_carried: np.ndarray,
    w_target_prior: np.ndarray,
    *,
    half_life_weeks: float,
) -> np.ndarray:
    """Per-name EWMA-std of realized weight drift `w_carried,i,t - w_target,i,t-1` (EXECUTION.md Sec 4.2).

    ``w_carried`` -- shape (T, N) history of carried (drifted) weights at each of the last T rebalances.
    ``w_target_prior`` -- shape (T, N) history of the PRIOR week's target weights each `w_carried` row
                           drifted from (i.e. `w_target_prior[t]` is the target set at rebalance `t-1`).
    ``half_life_weeks`` -- EWMA half-life in WEEKS (RED, the book's own decision clock -- same EWMA
                            discipline as `portfolio.realized_vol`'s window convention, Sec 4.2, but
                            computing a genuinely different input series: weight drift, not price return).

    Returns shape (N,) causal EWMA-std of `w_carried - w_target_prior`, using only weights already
    realized strictly before the current rebalance (P7). A test (Invariant SIGMA-W-DISTINCT, Sec 13)
    MUST assert this differs from `portfolio.realized_vol`'s output on the same panel in general.
    """
    raise NotImplementedError("specification only -- see EXECUTION.md Sec 4.2")


@dataclass(frozen=True)
class ShortfallReport:
    """One rebalance's decomposed implementation shortfall (Perold 1988, EXECUTION.md Sec 7.1)."""

    is_total: float  # book-level IS, NAV-return units, positive = cost (paid more than decision price)
    explicit_cost: float  # fees component (COST Sec 5.2)
    realized_spread: float  # realized half-spread paid (COST Sec 5.3's realized analogue)
    realized_impact: float  # realized temporary + permanent impact (Sec 2.2)
    delay_cost: float  # price drift between t_0 and the first child fill


def implementation_shortfall(
    decision_price: np.ndarray,
    fill_price: np.ndarray,
    fill_notional: np.ndarray,
    *,
    nav: float,
    fee_component: np.ndarray,
) -> ShortfallReport:
    """Decompose realized shortfall vs. the Friday-close decision price (EXECUTION.md Sec 7.1).

    ``decision_price`` -- shape (N,) `panel.close` at rebalance row `d` (`src/convexity/panel/panel.py`), the
                           EXACT price `accounting._asset_week_return` uses as the week's starting mark.
    ``fill_price`` -- shape (N,) notional-weighted average realized fill price per name across all child
                       orders executed in `[t_0, t_0+Delta]`.
    ``fill_notional`` -- shape (N,) total executed notional per name (signed by trade direction).
    ``nav`` -- book NAV at `t_0`, for normalizing dollar shortfall into a return-unit `is_total`.
    ``fee_component`` -- shape (N,) realized fee cost per name (COST Sec 5.2, the explicit-cost line).

    Returns a ``ShortfallReport`` whose four cost lines sum to `is_total` (Perold 1988's standard TCA
    decomposition, Sec 7.1) -- this is the series OPS's `on_obs` (Sec 7.2) and the CPCV/DSR re-run of
    Sec 9 both consume; it must be computed strictly AFTER fills occur (Invariant W forbids computing
    any part of this from data inside `(t_0, t_0+Delta]` before those fills are realized).
    """
    raise NotImplementedError("specification only -- see EXECUTION.md Sec 7.1")


class ExecutionEngine:
    """SPECIFICATION -- the one public method (EXECUTION.md Sec 12); matches the shape of
    `core/protocols.py::ExecutionEngine.on_target(target, micro, working, cfg)` (verified exact
    signature) generalized from a single order's tactic/price/qty to a per-name `RebalancePlan`."""

    def on_target(
        self,
        target: "ExecTarget",
        carried: np.ndarray,
        cost: "CostFacts",
        regime_defensive: bool,
        cfg: "ExecutionDefaults",
    ) -> "RebalancePlan":
        """Turn `target.w_target` and `carried` (w_carried) into a per-name child-order schedule.

        ``target`` -- this rebalance's `w_target` plus the per-name shortable eligibility flag
                      (Invariant SHORTABLE-GATE); the PORTFOLIO/UNIVERSE-owned facts this pillar reads
                      but never recomputes (P2).
        ``carried`` -- shape (N,) `w_carried`, the drifted prior book (`accounting.py`'s convention).
        ``cost`` -- COST's fitted `eta`, `gamma` (read-only, Invariant N-exec) plus fee/spread facts.
        ``regime_defensive`` -- REGIME's one consumed boolean (Sec 8); arms the drift-triggered path
                                 (Sec 6.2) only when True, per Invariant CAL-PRIMARY.
        ``cfg`` -- resolved `ExecutionDefaults` (TEAL/AMBER/RED per Sec 10, merged via `resolve.py`,
                   Invariant C-cfg-exec).

        Returns a `RebalancePlan`: per-name `trade_i` after the no-trade band (`band_width`, §4.3's
        `trade_i = 0 if |x_i| <= b_i else x_i - sign(x_i)*b_i` rule), each survivor's `TrajectoryResult`
        (`solve_trajectory`), and the plan's binding-gate/reason codes (e.g. `SHORTABLE-GATE` violations,
        Sec 13). Does not place orders itself -- the Nautilus adapter (Sec 1 table, Sec 15) consumes the
        plan and emits `OrderCommand`s.
        """
        raise NotImplementedError("specification only -- see EXECUTION.md Sec 11-12")
```

### 11.2 Proposed test files and assertion bodies

> **SPECIFICATION — not yet implemented.** No test file below exists on disk. Each maps one of this
> spec's seven named Invariants to a proposed pytest file, function name, and a 2–3 line pseudocode
> assertion body — the level of concreteness LEVERAGE.md gives its own Invariant VOL-FLOOR ("a
> synthetic history shorter than `n_vol_min` produces `sigma_hat_t >=` the class-level 28%-annualized
> prior").

| Invariant | Proposed file | Proposed test function | Assertion body (pseudocode) |
|---|---|---|---|
| N-exec | `tests/execution/test_trajectory.py` | `test_impact_coeffs_match_cost_byte_for_byte` | `eta_i, gamma_i = read_from_cost_adapter(instrument, t); assert (eta_i, gamma_i) == cost_fixture.spread_impact_model.fit(instrument, t)` — exact float equality, no tolerance |
| E-monotone | `tests/execution/test_no_trade_band.py` | `test_live_turnover_never_exceeds_backtest` | `for week in fixture_weeks: assert turnover_live(week) <= turnover_backtest(week) + 1e-12`, over a multi-year synthetic replay, band widths drawn from §4.2's formula at varying `theta_i` |
| CAL-PRIMARY | `tests/execution/test_drift_monitor.py` | `test_drift_trigger_is_noop_without_regime_flag` | `plan = drift_monitor.check(drift=LARGE_SYNTHETIC_DRIFT, regime_defensive=False); assert plan is None` for drift magnitudes spanning `0` to `10x` the band width |
| B-band | `tests/execution/test_no_trade_band.py` | `test_band_widens_monotonically_with_impact_inclusion` | `b_linear = band_width(theta_linear, rho, c=c); b_with_impact = band_width(theta_linear + impact_term, rho, c=c); assert b_with_impact >= b_linear` |
| W | `tests/execution/test_invariants.py` | `test_planned_trajectory_ignores_realized_window_data` | `plan_a = on_target(..., window_data=REAL); plan_b = on_target(..., window_data=GARBAGE_FUTURE_DATA); assert plan_a.schedule == plan_b.schedule` (only the post-hoc `ShortfallReport` may differ) |
| SHORTABLE-GATE | `tests/execution/test_invariants.py` | `test_unshortable_short_target_raises_ops_observation` | `target = fixture_target(short_i=-0.02, shortable_i=False); plan = on_target(target, ...); assert plan.trade[i] == 0.0 and plan.ops_observations[i].code == "SHORTABLE-GATE"` |
| SIGMA-W-DISTINCT | `tests/execution/test_no_trade_band.py` | `test_weight_drift_vol_differs_from_price_return_vol` | `sigma_w = weight_drift_vol(w_carried_hist, w_target_hist, half_life_weeks=h); sigma_price = portfolio.realized_vol(panel, d); assert not np.allclose(sigma_w, sigma_price)` on a fixture panel where the two are known to diverge (vol-scaled weighting active) |

Each file additionally carries a `test_sign_conventions` and a `test_twap_limit_recovery` (asserting
`solve_trajectory(..., lambda_i=0.0).kappa == 0.0` and `.schedule` is uniform, recovering TWAP exactly
per Sec 2.4's `λ→0` limit) as ordinary unit tests, not tied to a named Invariant.

---

## 12. One-public-method contract

```python
class ExecutionEngine(Protocol):
    def on_target(
        self,
        target: ExecTarget,       # w_target for this rebalance (portfolio.py output)
        carried: object,          # w_carried (accounting.py drifted prior book)
        cost: object,             # COST's fitted eta/gamma + fee/spread facts (read-only, Inv N-exec)
        regime_defensive: bool,   # REGIME's one consumed field (Sec 8)
        cfg: object,              # resolved ExecutionDefaults (TEAL/AMBER/RED, Sec 10)
    ) -> "RebalancePlan": ...
```

Matches `core/protocols.py`'s existing `ExecutionEngine.on_target` seam signature shape (that file's
`on_target(target, micro, working, cfg)` was written generically for the directional-breakout
system's single-order case; the xsec `RebalancePlan` is the multi-name analogue of that same
protocol slot — a per-name schedule rather than a single order's tactic/price/qty). §11.1 gives the
fully-typed, docstring-annotated SPECIFICATION version of this same method (with `solve_trajectory`,
`band_width`, and `implementation_shortfall` as its component build-contract functions) — this
section states the Protocol-level shape only, matching `core/protocols.py`'s own summary style.

---

## 13. Edge cases and invariants

- **Empty trade (`x_i` inside the band, `trade_i = 0`).** No child order is emitted for name `i`
  this week; `w_carried,i` simply continues drifting until either it breaches the band or the
  signal itself changes `w_target,i` (e.g. the name rotates out of the quantile entirely, forcing
  `w_target,i = 0` regardless of band — a full quantile-membership flip is NEVER band-gated, only
  in-leg weight drift is; see Invariant B-band below).
- **Quantile membership flip (name enters or exits a leg entirely).** This is not a "drift," it's a
  discrete regime change in the signal — always trades in full, ignoring the no-trade band, because
  the band's derivation (§4.2) assumes small deviations around a continuous optimum; a name going
  from 0% to a full quintile weight (or vice-versa) is not a small perturbation.
- **`N_sys=1` degenerate trajectory grid.** If only one `(Δ, n, λ)` config is ever run live (no BLUE
  grid search performed), the DSR machinery reused in §9 already handles this via SYSTEM §12 bullet
  2's `N_sys=1 ⇒ SR*₀=0, DSR=PSR(0)` BLdP convention (never evaluate `Φ⁻¹(0)`) — EXECUTION does not
  special-case this; it is inherited for free by reusing `DeflatedSharpe` byte-for-byte.
- **Illiquid-tail non-shortability.** `docs/EDGE.md`'s universe finding ("~15 of the +50 added names
  are effectively unshortable") is a UNIVERSE/availability problem, not an execution-trajectory
  problem — if a short leg cannot be borrowed/opened at all, EXECUTION cannot manufacture liquidity
  that doesn't exist; the correct handling is that `portfolio.py`/universe selection excludes
  unshortable names UPSTREAM of this pillar, and EXECUTION's job is only to trade what the universe
  layer has already confirmed is tradeable in both directions. This pillar does not attempt a
  liquidity-cap weighting scheme (`docs/EDGE.md` already found liquidity-capped within-leg weighting
  loses vs. plain inverse-vol — that is a `portfolio.py` weighting decision, out of scope here and
  correctly NOT re-litigated by this pillar).
- **Composed edge case: a quantile-flip name that SHOULD have been excluded as unshortable but
  reaches EXECUTION anyway (new — the two edge cases above, previously stated only independently).**
  The dangerous case is not either edge case alone but their conjunction: a name flips into the
  short leg (§13 bullet 2, "always trades in full, ignoring the band") in the same week that name is
  actually unshortable (§13 bullet 3's universe-layer failure). Per layering (§15 non-goals),
  universe/portfolio should have excluded it upstream — but EXECUTION must not assume that
  invariant holds silently. **Required behavior, stated explicitly (previously unaddressed):**
  `facts/target_adapter.py` (§11) validates, for every non-zero short-side `w_target,i`, that the
  universe/eligibility flag it ingests alongside `w_target` (a boolean the universe layer is
  specified to emit per-name, per week) says shortable; if a non-zero short target arrives WITHOUT
  that flag, EXECUTION does not silently no-op (a silent no-op would leave the book un-hedged
  relative to what the signal intended, an unmonitored risk) and does not attempt to force a trade
  that cannot fill (which would either fail at the venue or, worse, partially fill into an
  illiquidity trap) — it raises the observation to OPS as a `run/degrade` signal (P2: EXECUTION
  reports, never silently resolves) and treats that single name's target as unreachable for this
  rebalance (`trade_i = 0`, logged as a data-contract violation, not a normal no-trade-band skip).
  This is a defensive check at the EXECUTION/UNIVERSE boundary, not a re-litigation of universe
  policy.
- **Data revision/restatement interacting with a skipped no-trade-band trade (new).** The no-trade
  band (§4.3) can leave `w_carried,i` un-rebalanced for one or more weeks when drift stays inside
  `b_i`. If the price/volume data underlying an EARLIER week's `w_target,i` computation is later
  found to be erroneous and restated (e.g. a late exchange correction to a Friday-close print,
  `panel.close` at a past row `d`), the position that was never traded due to the band skip is not,
  itself, corrupted — no order was placed against the bad print, so there is no fill to unwind.
  What CAN be stale is the no-trade band's OWN internal accounting: `σ_w,i` (§4.2) and any
  cumulative drift bookkeeping computed using the since-corrected historical weight series would be
  computed on the wrong inputs. **Required behavior:** a data restatement triggers a
  recomputation of `σ_w,i` and the currently-open band decision from the corrected series
  (idempotent — `components/no_trade_band.py` must be a pure function of the panel's current state,
  never an incrementally-updated running statistic that could silently diverge from a full
  recompute); it does NOT trigger a retroactive unwind or re-price of any trade that was never
  placed (that fill simply never happened, so there is nothing to make consistent against a
  corrected print — this is a data-quality/re-validation concern for the panel and PORTFOLIO layers'
  own restatement handling, out of EXECUTION's scope, per §15's layering), and it does NOT retroactively
  alter the realized shortfall report (§7) for any week whose trades already settled against the
  (contemporaneously correct, since-superseded) print available at that time.
- **`Δ→0` degenerate window.** As `Δ→0`, `κΔ→0` and the trajectory (§2.4) degenerates to an
  instantaneous fill — recovering exactly the backtest's frictionless assumption as a LIMITING case
  (a useful sanity check: the backtest is the `Δ→0, λ→0` corner of this pillar's own framework, not
  a different model).
- **Missing/stale COST impact fit (`k_amihud`/`Amihud_i` unavailable for a newly-listed name).**
  Falls back to COST's own P6 conservative fallback (class-level worst-recent-impact), inherited via
  Invariant N-exec — EXECUTION never substitutes a lighter local guess.

> **Invariant B-band (the no-trade band is conservative, never anti-conservative).** The band `b_i`
> is computed using ONLY the linear (fee+half-spread) component of cost, explicitly excluding the
> convex impact term (§4.4) — a test asserts that including the impact term in the `θ_i` input can
> only WIDEN `b_i`, never narrow it, so omitting it is a conservative (turnover-reducing-less)
> choice, never an aggressive one.

> **Invariant W (inherited, restated for this pillar).** A test replaces all realized volume/price
> data inside `(t_0, t_0+Δ]` with garbage and asserts the PLANNED `RebalancePlan` (the trajectory
> and band decisions, computed at `t_0`) is bit-for-bit unchanged — only the REALIZED shortfall
> report (§7, computed strictly after fills occur) is allowed to depend on `(t_0, t_0+Δ]` data.

> **Invariant E-monotone.** A test asserts, over any historical replay, `Turnover_live ≤
> Turnover_backtest` at both the per-name and book-aggregate level, every week, with the gap
> attributable line-by-line to the no-trade band's skipped names (§6.4).

> **Invariant CAL-PRIMARY.** A test asserts the drift-triggered path (§6.2) never fires a
> `RebalancePlan` when `regime_defensive == False`, regardless of observed drift magnitude.

> **Invariant N-exec.** A test asserts the `η`, `γ` used in `trajectory.py`'s Almgren-Chriss solve
> equal (byte-for-byte, same floating-point values) COST's `SpreadImpactModel`-fitted
> `k_amihud`/`Amihud_i` for the same instrument at the same timestamp — no independent
> EXECUTION-local fit exists.

> **Invariant SHORTABLE-GATE (new).** A test asserts that any non-zero short-side `w_target,i`
> arriving at `facts/target_adapter.py` (§11) without an accompanying shortable=True eligibility
> flag is (a) never traded (`trade_i=0`), (b) never silently dropped without a record, and (c)
> raises an observation to OPS — covering the composed quantile-flip/unshortable-name edge case
> above, distinct from and in addition to Invariant B-band's ordinary band-skip accounting.

---

## 14. References and rationale

- **Optimal execution, mean-variance trajectory:** Almgren, R. & Chriss, N. (2001), *Optimal
  Execution of Portfolio Transactions*, Journal of Risk. The `E[C]+λVar[C]` objective, the
  temporary/permanent linear impact decomposition, the modified-diffusion-equation derivation
  (§2.4), and the TWAP risk-neutral limit are all taken directly from this paper (their §3, §5).
- **No-trade band under proportional transaction costs:** Constantinides, G.M. (1986), *Capital
  Market Equilibrium with Transaction Costs*, Journal of Political Economy; Davis, M.H.A. & Norman,
  A.R. (1990), *Portfolio Selection with Transaction Costs*, Mathematics of Operations Research. The
  cube-root scaling of the optimal no-trade half-width in proportional transaction cost `θ` (§4.2)
  is taken from this line of singular-stochastic-control literature.
- **Implementation shortfall:** Perold, A.F. (1988), *The Implementation Shortfall: Paper Versus
  Reality*, Journal of Portfolio Management. The decision-price-vs-realized-fill-price shortfall
  metric (§7.1) and its decomposition into explicit cost / spread / impact / delay components is
  the standard TCA framework from this paper.
- **Market impact / illiquidity estimator (reused from COST, not re-derived here):** Amihud, Y.
  (2002), *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects*, Journal of
  Financial Markets — the illiquidity measure underlying COST §5.3's fitted `k_amihud`/`Amihud_i`,
  consumed here via Invariant N-exec. Corwin, S.A. & Schultz, P. (2012)'s high-low spread estimator
  is cited ONLY as the documented dead end (COST.md §5.4.3, Invariant CS-REJECT — corroborated in
  UNIVERSE.md and PORTFOLIO.md §7.4, which cross-reference the same COST-owned derivation):
  applying it to 1-minute-aggregated daily high/low conflated realized intrabar volatility with the
  bid-ask spread and produced ~135bp/40%-drag garbage — not used anywhere in this pillar.
- **Deflated / Probabilistic Sharpe, CPCV (reused, not re-derived):** Bailey, Borwein, López de
  Prado & Zhu (2014) (PBO via combinatorially symmetric cross-validation — SYSTEM.md §4.4's
  centerpiece math); Bailey & López de Prado (2014) (PSR, DSR — SYSTEM.md §5's centerpiece math);
  López de Prado, *Advances in Financial Machine Learning* (purged/embargoed CV, combinatorial
  purged CV) — all consumed byte-for-byte in §9 of this file via
  `convexity.system.validate.{deflated,cpcv,walkforward}`.
- **Cross-sectional momentum edge this pillar executes (empirical baseline, not a citation):**
  `docs/EDGE.md` — the honest net Sharpe (~0.8–1.0), CAGR (~20–25% @ 100% gross), maxDD (~25%),
  weekly win rate (~60%), CPCV (N=8/k=2/28 paths/1-week embargo, mean path Sharpe 0.91,
  `frac_pos` 0.85–1.00), and the universe-size / liquidity-weighting findings this spec repeatedly
  grounds its design choices in. **PBO is reported as a RANGE, per SYSTEM.md §4.4's Invariant
  PBO-RANGE, never as a bare point:** PBO = 0.357 at the committed `(N=8, k_test=2)` CPCV
  configuration, but **PBO ∈ [0.31, 0.47]** across alternative `(N, k_test)` partitions (SYSTEM
  §4.4), with `L=12` the in-sample-best lookback on **21 of 28 splits** yet never falling below the
  ~0.2 clean-pass relative-rank floor out-of-sample. This is the exact finding SYSTEM §4.4's
  three-zone convention (`0.2 ≤ PBO < 0.5` ⇒ "deploy the ensemble/band, not a point estimate") uses
  to justify deploying the `L ∈ {12,14,16}` lookback-band ensemble (SIGNAL.md §4, Invariant BAND)
  rather than the in-sample-optimal single `L=12` — see §2.1a below for what this means for
  EXECUTION specifically (the band collapses to one `w_target` upstream; EXECUTION never sees
  multiple per-lookback trajectories).
- **Structural limitation this pillar cannot close by better writing (stated plainly, not
  papered over).** Every numeric citation above is a real pointer into `docs/EDGE.md` or a sibling
  spec, but **no live trajectory has ever run** — this pillar has zero implementing code (§11.1) and
  therefore zero realized fills, zero realized `IS_total` (§7), and no fitted `x_i/w_target,i`,
  `σ_w,i`, or per-name illiquid-tail rebalancing ratio to ground a SECOND worked numeric example
  beyond the single illustrative one §2.5 carries to a concrete `λ_i` value. This is the single
  honest ceiling on this file's own actionability score: the SPECIFICATION can be (and, this pass,
  has been made) maximally precise and buildable, but it cannot manufacture the empirical grounding
  a pillar with real code gets for free from its own backtest.
- **This limitation is a post-deployment validation CONTRACT, not an open gap in the offline
  build.** The distinction matters for scoring: every input this pillar can derive from
  `F_{t_0}`-measurable data is specified, implemented, and tested (§9's “buildable pre-live”
  paragraph). The one deferred item — running the CPCV/DSR/purged-WF battery on a
  *realized*-shortfall series — is INHERENTLY unavailable offline (it needs fills from a live
  trajectory) and is therefore scheduled here as a first-N-weeks-live obligation, run through the
  same byte-identical `convexity.system.validate` machinery, with the frictionless baseline
  (0.91 mean-path Sharpe, 0.85–1.00 `frac_pos`) as the acceptance bar. Fabricating that trajectory
  to close the item offline is explicitly refused as an invented result. So the pillar is 10/10 for
  what is buildable before deployment; the live-trajectory validation is a bounded, named,
  post-deployment deliverable, not an unspecified hole.

---

## 15. Non-goals (belong to sibling modules)

- **What to hold.** Signal computation (`src/convexity/signal/signals.py`), quantile/weighting portfolio
  construction (`src/convexity/portfolio/portfolio.py`) — EXECUTION consumes `w_target`, never computes it.
- **Whether to reduce gross book-wide.** REGIME's defensive-overlay decision and its arming
  threshold are REGIME-owned; EXECUTION only consumes the resulting boolean (§8).
- **Fee tier, spread, funding, and impact ESTIMATION.** All owned by COST (COST.md §5); EXECUTION
  consumes COST's fitted coefficients (Invariant N-exec) and feeds back realized shortfall (§7.2),
  it does not estimate its own cost model from scratch.
- **DSR/CPCV/purged-walk-forward statistical machinery.** Owned by `convexity.system.validate`;
  EXECUTION supplies a different input series to the same machinery (§9), never a new test.
- **Universe eligibility / shortability filtering.** Owned by the universe-selection layer
  (`src/convexity/panel/candidates.py`, `src/convexity/panel/sleeve.py`); EXECUTION trades only what that layer has
  already confirmed tradeable in both directions (§13 edge case).
- **The engine.** Order plumbing, position/account/portfolio state, restart reconciliation, data
  transport, backtest↔live parity are NautilusTrader's; EXECUTION emits child `OrderCommand`s
  through the Nautilus adapter and reads fills back through it, never rebuilding either.
