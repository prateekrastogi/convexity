# COST.md — build spec for the single live cost / funding / carry truth

This file is the authoritative build spec. Build to the contracts and
invariants below. Where this file and your own instinct disagree, this file
wins. Ask before deviating from any **Invariant**.

This module is the **single cost truth** consumed by every other pillar. It
composes with the sibling modules without importing their internals:
- BREAKOUT.md decides trade / no-trade (entry gate).
- TREND.md decides hold / reduce / exit on an open position.
- LEVERAGE.md decides how much (sizing under survival constraints).
- A Carver-Ch25 selection module decides which instruments are eligible at all.
- **COST.md (this) is a PROVIDER, not a decider.** It answers exactly one family
  of questions and emits numbers the deciders consume:

> Given fees, spread, EXECUTION's realized slippage/impact, and perp/roll/MTF
> financing — what does this trade cost round-trip, what will funding/carry cost
> over the remaining hold, and is the expected favorable move large enough to
> pay for all of it?

COST never opens, holds, reduces, or routes anything. It hands BREAKOUT a
**net-of-cost expected-value gate**, hands TREND a **funding-aware hold/reduce
input**, hands LEVERAGE a **carry term** for the stop ceiling, and produces
**post-trade TCA** (implementation-shortfall decomposition, realized-vs-expected,
attribution). It is the one place where a fee, a spread, a funding rate, or a
basis is turned into an economic decision input, so that the four deciders never
each grow their own divergent cost stub.

Scope: full cross-asset — Binance USDT-M perpetual futures (8h funding),
Dukascopy-proxied commodities/metals (storage/contango carry), US equities (MTF
financing), and dated futures (roll/basis carry) — through **one interface, one
model per class**, selected by AMBER metadata, never by forked logic.

Built **on top of NautilusTrader**, which already owns: order plumbing,
position/account/portfolio state, **restart reconciliation**, data transport,
backtest↔live parity, pre-trade risk checks, and funding **ingestion**. COST does
**not** rebuild any of those. It builds the cost/funding **POLICY** on top of the
raw commission accounting and the raw funding events Nautilus already records.

Context: ~$1000 retail book, 1-minute bars, commodity hardware (target a 2-core /
11 GB ARM VM). Retail latency budget is **seconds at bar granularity, not
microseconds**. Every live number must be `F_t`-measurable (causal); future data
appears only inside firewalled validation labels.

---

## 0. Governing principles (read before anything else)

**P1 — One cost truth, consumed four ways.** There is exactly one cost model.
BREAKOUT, TREND, LEVERAGE, and TCA all call the SAME estimators. A trade that
BREAKOUT admitted as net-EV-positive and a trade TREND later holds must have been
priced with identical fee/spread/funding math. Divergent per-pillar cost stubs
(the placeholder `CostModel` in BREAKOUT.md / TREND.md) are explicitly superseded
by this module. Any pillar computing its own funding accrual is the bug.

**P2 — Provider, never decider (the most important rule in this file).** COST
emits numbers and reason codes. It NEVER emits an action. The net-EV **gate**
returns a boolean plus the decomposed margin; BREAKOUT chooses whether to act on
it. The funding-aware hold input returns "forecast residual carry vs forecast
residual favorable move"; TREND chooses to hold/reduce/exit. No `place_order`, no
`close_position`, no `cancel` ever originates here. (See Invariant D.)

**P3 — Cost is a CEILING the edge must clear, evaluated as a `min()`/firewall of
independent components, never a multiplicative score blob.** Total round-trip
cost is an **additive decomposition** of independent, individually-auditable
dollar terms:

```
C_round_trip = C_fee_in + C_fee_out + C_halfspread_in + C_halfspread_out
             + C_impact_in + C_impact_out + C_funding_forecast + C_carry_forecast
```

Each term has its own estimator, its own units (always quote currency and bps of
notional), and its own reason code. There is NO learned weighting, no
`fee × spread × regime` product, no single opaque "cost score". The gate is a
**comparison of two additive sums** (expected favorable excursion vs expected
cost), each decomposable into named lines. (See Invariant B.)

**P4 — Funding/carry forecasting is CAUSAL and NO-ML.** The remaining-hold
funding forecast uses only information available at `t`: the recent realized
funding path, the current basis, and the published next-funding rate where the
venue pre-announces it (Binance publishes the upcoming funding rate continuously).
The forecast is a transparent, monotone, decomposable estimator (an EWMA of
recent funding plus a basis-implied carry term with explicit shrinkage), NEVER a
trained model. No future funding timestamp is read at `t`. (See Invariant C, F.)

**P5 — Accrued is FACT, forecast is ESTIMATE; never confuse them.** Funding
already paid in `[entry, now]` is a deterministic sum over funding events that
actually occurred — read from Nautilus's ingested funding events, not modelled.
Funding over `[now, expected_exit]` is a forecast with an explicit confidence
band. The two are separate fields, separately tested, never summed into one
unlabelled number. Realized TCA uses accrued FACT; the gates use forecast
ESTIMATE. (See Invariant A.)

**P6 — Conservative under uncertainty.** When a cost input is missing, stale, or
its estimator fails diagnostics, COST returns a CONSERVATIVE (higher) cost, never
zero and never infinity. A missing spread falls back to a class-level wide
spread; a missing funding history falls back to the venue's published rate or a
class-level worst-recent-funding; missing impact falls back to a square-root
model at a conservative coefficient. A gate that cannot price a trade fails the
trade closed, it does not pass it free. (See Invariant Q.)

**P7 — Sign discipline is load-bearing.** Funding and carry are SIGNED relative to
the position. A perp long paying positive funding bleeds carry; a perp short in
the same regime EARNS it. Contango is a cost to a long-roll futures position and
a benefit to a short. Every carry term carries the position side and the sign is
unit-tested at the boundary, because a sign error turns a cost gate into an
anti-gate. (See Invariant P.)

Forbidden in v1: any ML / learned cost or funding model (P4); a multiplicative
"cost score" (P3); COST emitting any trading action (P2); reading future funding
timestamps in a live path (P5); per-symbol tuning of the RED cost/forecast
constants on realized P&L (Section 4); rebuilding Nautilus commission accounting,
funding ingestion, or reconciliation (Section 1); treating accrued funding and
forecast funding as one number (P5); a forecast `r_hat` that exceeds the venue
funding cap or adds the published rate as a bias on top of a convex blend (5.5);
reading realized FUTURE volume or a TREND reduce schedule into the gate (Invariant
W); pricing the gate at an optimistic point hold instead of the conservative
`m_max` (Invariant M-hold); pricing the gate's funding at the POINT forecast
instead of the side-selected conservative band (Invariant F-band-gate); consuming
a fixed `funding_band_high` regardless of side, which is anti-conservative for a
short (Invariant P-band); a funding band built as `m` independent draws
(Invariant G-band); a band quantile with no event-count floor or CI edge
(Invariant G-est); a bare inter-contract spread or bare `Δt` carry with no
time-scaling / day-count (Invariant H-time); applying the `(8/24)` interest scaling
to a `per_interval` input (the ~3× double-discount, 5.5b); charging proxy financing
twice (Invariant Y-finance); the rogue `/1e4` on the impact term (Invariant U-dim);
a TCA within-tolerance open sum instead of an exact identity with a named residual
plug (Invariant R).

---

## 1. What COST builds ON TOP OF Nautilus (and what it must NOT rebuild)

Nautilus already provides, and COST consumes as given:

| Nautilus owns (do NOT rebuild)                     | COST consumes it as                              |
|----------------------------------------------------|--------------------------------------------------|
| `Commission` on every fill, per-fill maker/taker   | raw fee FACT for realized TCA (`accrued_fees`)   |
| Funding **ingestion** (funding events, rates)      | the funding-event stream the accrual sums over   |
| Position / account / portfolio state               | signed position notional at each funding stamp   |
| Fill prices, arrival/decision marks                | implementation-shortfall reference prices        |
| Restart **reconciliation**                         | accrued state survives restart; COST re-derives  |
| Backtest↔live parity, data transport               | same estimators run in both; COST adds no I/O    |
| Pre-trade risk checks                               | orthogonal; COST's gate is EV, not risk          |

COST builds the **policy/forecast layer**: turning the raw maker/taker schedule
into an **effective fee by VIP tier from rolling volume**; turning bid/ask into a
**half-spread cost**; consuming **EXECUTION's realized slippage/impact** (COST
does not model microstructure execution — that is EXECUTION.md — it ingests the
realized number and forecasts the *expected* impact for the gate); summing
**accrued funding** from Nautilus's funding events; **forecasting** remaining-hold
funding/carry; and assembling the **net-EV gate**, the **hold input**, and the
**TCA**.

> **Invariant N — no Nautilus re-implementation.** COST does not compute raw
> commissions, does not ingest funding ticks from the exchange, does not manage
> position state, and does not reconcile on restart. It reads those from
> Nautilus. A test asserts COST's accrued-fee number equals the sum of Nautilus
> `Commission` objects for the position (it is a re-key/aggregate, not a re-model).

---

## 2. Architecture: four consumers, one provider

```
                         ┌──────────────────────────────┐
   Nautilus facts ─────► │           COST               │
   (fees, funding events,│  fee tier · spread · impact   │
    fills, positions)    │  accrued funding · forecast   │
   EXECUTION realized ──►│  funding/carry · net-EV math  │
   slippage/impact       └───┬──────┬──────┬──────┬──────┘
                             │      │      │      │
              net-EV gate ◄──┘      │      │      └──► post-trade TCA
              (BREAKOUT)            │      │           (implementation
                                    │      └──► carry term  shortfall +
                 funding-aware hold ┘           (LEVERAGE     attribution)
                 / reduce input                  stop ceiling)
                 (TREND)
```

Five layers, kept independent for testability:

1. **Facts layer** — adapts Nautilus commission/funding/fill objects and
   EXECUTION's realized slippage/impact into COST's frozen types. No modelling.
2. **Component layer** — six independent estimators, each emitting one cost
   component in quote currency and bps, plus diagnostics: `FeeModel`,
   `SpreadModel`, `ImpactModel`, `FundingAccrual`, `FundingForecast`,
   `CarryModel`. Each testable in isolation.
3. **Assembly layer** — `CostBook` sums the components into the additive
   `RoundTripCost` (P3); computes cost-as-fraction-of-edge; never decides.
4. **Consumer layer** — three one-public-method contracts: `NetEVGate` (for
   BREAKOUT), `HoldCarryInput` (for TREND), `CarryTerm` (for LEVERAGE), plus
   `TCA` (post-trade). Each is a thin transform over the assembly layer.
5. **Validation layer** — purged walk-forward, CPCV, baselines, ablation,
   block-bootstrap CIs, multiple-test control, synthetic adversarial data. Never
   imported by layers 1–4.

Why: a bad funding forecast can be isolated from a bad fee-tier model; each
component ablated independently; the binding cost line is always explicit in the
decomposition.

---

## 3. Directory layout

```text
cost/
+-- config/
|   +-- defaults.py          # CostDefaults: RED shared constants (forecast half-life,
|   |                        #   shrinkage, ES-style buffers, fallback spreads/impact)
|   +-- fee_schedule.py       # AMBER: venue maker/taker tiers, VIP volume breakpoints
|   +-- asset_class.py        # AssetClassCostPreset: validated per-class carry model + fallbacks
|   +-- instrument.py         # InstrumentCostSpec: AMBER metadata only (funding interval, etc.)
|   +-- resolve.py            # single merge point (RED unreachable from instrument)
+-- facts/                    # FACTS LAYER — adapt Nautilus + EXECUTION, never model
|   +-- nautilus_adapter.py   # Commission/funding-event/fill -> COST frozen types
|   +-- execution_adapter.py  # ingest EXECUTION's REALIZED slippage/impact
+-- components/               # COMPONENT LAYER — each emits one cost line, never a decision
|   +-- fees.py               # FeeModel: rolling-volume -> VIP tier -> effective maker/taker
|   +-- spread.py             # SpreadModel: half-spread cost from bid/ask (+ fallback)
|   +-- impact.py             # ImpactModel: expected slippage/impact (square-root FRACTION, causal v_hat), uses EXECUTION realized
|   +-- funding_accrual.py    # FundingAccrual: sum of realized funding in [entry, now] (FACT)
|   +-- funding_forecast.py   # FundingForecast: causal piecewise EWMA/basis/published+clamp over m_max (ESTIMATE)
|   +-- carry.py              # CarryModel: ONE interface, per-class (perp/roll/MTF/storage)
+-- assemble/
|   +-- cost_book.py          # CostBook: additive RoundTripCost (P3); cost-as-fraction-of-edge
+-- consume/                  # CONSUMER LAYER — one public method each, thin transforms
|   +-- net_ev_gate.py        # NetEVGate: BREAKOUT entry gate (EV > cost incl. forecast funding)
|   +-- hold_carry.py         # HoldCarryInput: TREND hold/reduce input (residual carry vs residual move)
|   +-- carry_term.py         # CarryTerm: LEVERAGE stop-ceiling carry term
|   +-- tca.py                # TCA: implementation-shortfall decomposition + attribution
+-- validate/
|   +-- labels.py             # realized round-trip cost, realized funding, realized IS (future-only)
|   +-- walkforward.py        # purged walk-forward: forecast-vs-realized funding/cost
|   +-- cpcv.py               # sampled combinatorial purged CV
|   +-- baselines.py          # flat-bps cost, no-funding, last-funding-only, perfect-foresight bound
|   +-- ablation.py           # which cost components matter; funding on/off; forecast vs naive
|   +-- stats.py              # block-bootstrap CIs; DSR/PSR; multiple-test control
|   +-- reports.py
+-- synth/
|   +-- generate.py           # adversarial funding spikes, regime flips, spread blowouts, roll gaps
+-- types.py                  # frozen dataclasses crossing module boundaries
+-- cost_api.py               # public entry point used by the trading layers
tests/
+-- ...                       # unit, firewall, parity, accrual=Nautilus, sign, acceptance, ablation
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest, NautilusTrader (consumed, not
rebuilt). No ML frameworks. `numba` only after correctness tests pass, and only
on the hot accrual/forecast loops if profiling on the target VM shows need.

---

## 4. Parameter governance

Every parameter is exactly one bucket.

- **TEAL — fitted per instrument from its own past data, at runtime, CAUSAL.**
  Examples: the EWMA funding level/half-life behavior on this symbol's own funding
  history; the realized-impact coefficient fitted to this instrument's own past
  fills (from EXECUTION's realized series); the basis EWMA; the spread regime
  buckets (calm/stressed) fitted to the instrument's own spread distribution.
- **AMBER — supplied market metadata.** Venue maker/taker schedule and VIP volume
  breakpoints, funding interval (8h for Binance USDT-M; per-venue), the funding
  **cap**, the **interest-rate baseline** components (`interest_quote`,
  `interest_base`) and their **period unit** `interest_rate_unit ∈
  {per_day, per_interval}` (which gates the 8h scaling, 5.5b — no silent default),
  the venue-published **premium index** (`premium_index_8h`) for the basis anchor
  (5.5b), the inter-contract expiry gap `τ_spread` and `mtf_compounding` flag for
  carry time-scaling (5.6-time), whether the venue pre-publishes
  next funding, contract multiplier/lot, MTF financing rate, storage/borrow rate,
  `commodity_financing_mode` (`unlevered`/`mtf`) and `commodity_basis_observable`
  (5.6-commodity), roll/expiry calendar, settlement currency.
- **RED — shared economic/modelling constants.** The funding-forecast EWMA
  half-life `h` (in 8h STAMPS) → `α = 1 − 2^(−1/h)` (5.5a), the funding-event floor
  `n_funding_min` below which the EWMA falls back (5.5a), the residual-event floor
  `n_residual_min` below which the comonotone band falls back to the class worst
  residual (5.5c), the fee-tier boundary-proximity tolerance `tier_proximity` that
  triggers conservative (worse) tier pricing in the gate (Section 7, Invariant
  J-tier), the basis→carry shrinkage
  λ, the conservative band quantile `q` for the comonotone funding band (5.5c), the
  max-hold quantile that sets `m_max` for the gate (5.5d), the cost-as-fraction-of-
  edge rejection threshold used in reports (NOT in the gate — the gate uses the raw
  inequality), the fallback wide-spread multiple, the conservative impact
  coefficient (fraction-valued, 5.4), the minimum-edge buffer added to the gate (a
  margin of safety), the block length `L_funding` for the funding bootstrap (5.5d),
  and the funding-forecast staleness window.
- **BLUE — finite validated grids** (optional, class-level only): small candidate
  sets for the funding EWMA half-life `h` and the basis shrinkage λ, searched
  offline with promotion gates under DSR/PSR control, NEVER per instrument on P&L.

> **Invariant C-cfg — RED unreachable from per-instrument config.** RED enters
> only via `resolve_config(defaults=...)`. Per-class overrides go through a
> separate `AssetClassCostPreset` argument and only when validated. If a forecast
> half-life, a shrinkage λ, a band quantile `q`, a residual floor `n_residual_min`,
> a fee-tier proximity `tier_proximity`, a max-hold quantile, or a gate
> buffer appears as a per-instrument field, that is the bug. Per-symbol tuning of
> any RED constant on realized P&L is forbidden.

---

## 5. The cost decomposition (full math, additive — P3)

### 5.1 Components and units

Every component is reported in BOTH quote currency (for summing) and basis points
of position notional `Notional = |qty| · price · multiplier` (for cross-instrument
comparison). One round trip has an entry leg and an exit leg.

```
C_round_trip =  C_fee_in     + C_fee_out          # 5.2 fee-tier
             +  C_hs_in       + C_hs_out           # 5.3 half-spread
             +  C_impact_in   + C_impact_out       # 5.4 slippage/impact (from EXECUTION)
             +  C_funding     + C_carry            # 5.5 / 5.6 funding & carry (signed, P7)
```

### 5.2 Fees (effective, VIP-tier, rolling-volume) — Section 7

```
C_fee_leg = liquidity_flag · fee_rate(tier(rolling_30d_volume)) · Notional_leg
```
`liquidity_flag ∈ {maker, taker}`; `fee_rate` is the venue's maker or taker rate
at the VIP tier implied by the rolling 30-day traded volume (Section 7).

### 5.3 Half-spread

The cost of crossing from mid to the touch on each leg:

```
C_hs_leg = (half_spread_bps_at(t_leg) / 1e4) · Notional_leg
half_spread_bps = 1e4 · (ask − bid) / (2 · mid)
```
When bid/ask is unavailable, fall back to the class-level conservative half-spread
(RED, P6) — never zero. Spread is regime-bucketed (calm/stressed) per instrument
(TEAL) and the forecast for the gate uses the current regime's expected
half-spread, not a single global average.

### 5.4 Slippage / impact (expected; realized from EXECUTION)

COST does **not** model order-book execution (that is EXECUTION.md). It ingests
EXECUTION's **realized** per-fill slippage/impact for TCA, and forecasts the
**expected** impact for the gate using a transparent square-root model fitted
(TEAL) to this instrument's own realized series.

**Dimensions (the `/1e4` is a bug magnet — fix it).** The Almgren square-root law
`k · σ · sqrt(Q/V)` is already a **return fraction**, not basis points. So the
impact cost in quote currency is the fraction times notional, with **no `/1e4`**:

```
C_impact_leg = impact_frac_leg · Notional_leg                       # quote ccy
impact_frac_leg = k_i · sigma_i · sqrt( participation_leg )         # a FRACTION
impact_bps_leg  = 1e4 · impact_frac_leg                             # for the bps column only
participation_leg = Notional_leg / V_hat_leg
```
- `sigma_i` is the per-bar return volatility on the same bar clock as the fitted
  `k_i` (a dimensionless return), `k_i` the instrument's fitted dimensionless
  square-root coefficient (TEAL). Their product is a fraction; `sqrt(participation)`
  is dimensionless; so `impact_frac_leg` is a fraction and multiplies notional
  directly. The `/1e4` that appears in the half-spread term (5.3) is correct there
  ONLY because `half_spread_bps` is genuinely in bps; it must NOT appear here. The
  bps figure is recovered by multiplying the fraction by `1e4` for the reporting
  column, never by dividing the cost.
- If `k_i` is instead supplied pre-expressed in bps (an AMBER class fallback may
  be), the fitter records `k_units ∈ {fraction, bps}` on the `CostComponent.reason`
  and converts to a fraction once at ingestion; the assembly math only ever sees a
  fraction. (Invariant U-dim.)

The square-root law is the standard, auditable, monotone impact form; a
conservative linear fallback (RED coefficient, fraction-valued) is used when
`k_i` cannot be fitted (P6). For realized TCA, the EXECUTION-supplied realized
slippage replaces the forecast (Section 9).

**`V_hat_leg` is a CAUSAL volume forecast, never realized future volume
(Invariant W).** `participation_leg = Notional_leg / V_hat_leg` where `V_hat_leg`
is the **expected notional volume over the leg's execution window**, and it MUST
be an `F_t`-measurable forecast built only from PAST volume — e.g. an EWMA of
recent per-bar notional volume scaled to the window length, or a time-of-day
volume profile fitted to past bars (TEAL). For the entry gate the exit-leg window
is in the future; reading the realized volume that will occur in that window is
lookahead and is forbidden. The forecast is the same kind of object as the funding
forecast (5.5): deterministic *window* (how many bars), forecast *volume*. A
conservative LOW volume forecast (which inflates participation and therefore
impact) is the P6 fallback when recent volume is stale. See Invariant W's test:
replacing all post-`t` volume with garbage must leave the gate's impact term
bit-for-bit unchanged.

### 5.5 Perp funding — accrual (FACT) and forecast (ESTIMATE)

**Accrued funding (FACT, P5).** Over `[entry, now]`, summed over the funding
timestamps that ACTUALLY occurred (read from Nautilus's ingested funding events):

```
F_accrued = Σ_{k : t_entry ≤ τ_k ≤ t_now}  side · Notional(τ_k) · r_k
```
where `τ_k` are the funding timestamps (Binance USDT-M: 00:00, 08:00, 16:00 UTC
— 8h interval, AMBER), `r_k` the realized funding rate at `τ_k`, `Notional(τ_k)`
the signed position notional at that instant, and `side ∈ {−1, +1}`. A perp LONG
paying positive funding accrues a COST (`F_accrued > 0` is money out); a SHORT in
the same regime accrues a CREDIT. Sign is unit-tested (P7, Invariant P).
`Notional(τ_k)` must be the position notional AT `τ_k` (it changes with price and
with partial reduces) — COST reads it from Nautilus position state, it does not
assume entry notional.

**Forecast funding (ESTIMATE, CAUSAL, NO-ML — P4).** Over the remaining hold
`[now, t_exit_expected]`, COST forecasts the funding it has not yet paid. Let the
remaining window contain `m` future funding stamps (a count from the fixed 8h
schedule — the *timestamps* are deterministic, the *rates* are not; `m` itself
depends on the hold length and is NOT treated as a known scalar — see "the m
problem" below).

**Per-stamp rate forecast — piecewise, NOT a sum of three terms.** The published
next-funding rate is an OVERRIDE for the first not-yet-paid stamp, not an additive
bias. Writing `r_hat = (1−λ)·r_ewma + λ·r_basis + bias_published` is a bug: the
convex weights `(1−λ)+λ` already sum to 1, so adding a published term on top
inflates EVERY stamp by the published rate and fails the realized-vs-expected
calibration (Invariant E). The correct form is a per-stamp indicator:

```
              ┌ clamp( r_published_next ,  −cap, +cap )     if j == 1 AND r_published_next is F_t-known
r_hat(τ_j) =  │
              └ clamp( (1 − λ)·r_ewma + λ·r_basis ,  −cap, +cap )   otherwise
```
- `j == 1` is the NEXT funding stamp. Binance publishes its rate continuously and
  it is `F_t`-measurable, so when available it is used EXACTLY (the single most
  accurate causal input). It does not enter the blend; it replaces it for that one
  stamp. If the venue does not pre-publish (AMBER `pre_publishes_funding=False`),
  the blend governs `j == 1` too.
- `r_ewma` — EWMA of the realized 8h funding rate (closed form in 5.5a),
  capturing the persistent funding regime (TEAL on this symbol's own funding
  series).
- `r_basis` — the basis-implied funding anchor (closed form in 5.5b), the
  no-arbitrage tether funding mean-reverts toward. Shrinkage weight `λ ∈ [0,1]`
  (RED, BLUE-searchable) tempers how much basis pulls the forecast.
- `cap` — the venue funding CAP (AMBER, e.g. Binance's per-interval clamp). A
  forecast that can exceed the cap is wrong by construction: realized funding
  CANNOT exceed it, so every `r_hat(τ_j)` is clamped to `[−cap, +cap]`. The clamp
  is applied to the published stamp too (a published rate already respects the cap;
  the clamp is a defensive no-op there).

```
F_forecast = Σ_{j=1}^{m}  side · Notional_expected(τ_j) · r_hat(τ_j)
```
with a **confidence band** aggregated over the `m` stamps per 5.5c (a correlated
worst-case sum, NOT `m` independent draws). The point blend `F_forecast` above is
the EXPECTED funding; it is NOT what the gate prices funding at. The gate's binding
funding line is the side-selected conservative band `F_band_conservative(m_max, q)`
of 5.5c (the worse-for-the-position comonotone quantile at its conservative CI
edge), because forecast funding is uncertain WITHIN each stamp and the gate must
clear the upper-CI funding cost, not the point estimate (P6, Gap 1 / Invariant
F-band-gate, 8.1). `F_forecast(m_max)` is reported as telemetry; the gate consumes
the band.

> No future funding timestamp's REALIZED rate is read at `t`. The schedule (which
> 8h boundaries fall in the window) is deterministic; the rates are forecast; the
> published-NEXT rate is the one `F_t`-measurable future-stamp value permitted.
> (Invariant F.)

#### 5.5a EWMA closed form, half-life→α, and the event-count floor

The funding-level EWMA is an `O(1)` streaming recursion over realized 8h funding
events (NOT calendar time — the unit is one funding stamp):

```
r_ewma_k = α · r_k + (1 − α) · r_ewma_{k−1}
α = 1 − 2^(−1/h)              # half-life h in NUMBER OF 8h STAMPS (RED)
ESS = (2 − α) / α             # effective sample size of the EWMA
```
- `h` is the RED half-life expressed in funding stamps. "~3 days" = `h ≈ 9` stamps
  (3 days × 3 stamps/day), giving `α = 1 − 2^(−1/9) ≈ 0.0743` and `ESS ≈ 25.9`
  stamps. The half-life is stated in stamps, never in raw minutes, because the
  funding clock is the event clock.
- **Event-count floor (the same discipline LEVERAGE uses for conditional-MAE and
  EVT).** The EWMA is trusted only when the realized funding history has at least
  `n_funding_min` events (RED, e.g. ≥ `2·h` so `ESS` is meaningfully filled).
  Below the floor the forecast FALLS BACK, in priority order (P6): (1) the
  published-next rate held flat across all `m` stamps if published; else (2) the
  last realized funding rate held flat (the `last-funding-only` baseline of
  Section 12); else (3) the class-level worst-recent-funding fallback. The fallback
  taken is recorded on `CostComponent.reason` (`fallback_ewma_thin`,
  `fallback_published_flat`, `fallback_last_funding`, `fallback_class_worst`).

#### 5.5b Basis → funding mapping (the no-arbitrage anchor, made explicit)

The perp tethers to index via funding; the funding anchor is the **premium of mark
over index plus the venue's interest-rate baseline**, scaled to the 8h interval.
This mirrors Binance's own funding construction (premium index + interest
component, then clamped). Per 8h stamp:

```
premium_8h = (perp_mark − index) / index                  # an 8h-interval fraction (mark already 8h-relative)
r_basis    = clamp( interest_8h + premium_8h ,  −cap, +cap )

# interest_8h: SCALING IS CONDITIONAL ON THE AMBER INPUT UNIT (interest_rate_unit)
interest_8h = (interest_quote − interest_base) · (8 / 24)   if interest_rate_unit == "per_day"
interest_8h = (interest_quote − interest_base)              if interest_rate_unit == "per_interval"   # venue-native; NO rescale
```
- `premium_8h` is the premium of the perp mark over the spot index, expressed
  directly as a per-interval fraction (the mark/index pair is sampled on the
  funding clock, so no separate annualize-then-rescale step is needed).
- **`interest_8h` period convention — pinned, and the `(8/24)` is CONDITIONAL (Gap,
  double-discount).** Binance's NATIVE funding construction publishes the interest
  component PER FUNDING INTERVAL (per 8h), not per day. If the AMBER inputs
  `interest_quote`/`interest_base` are already per-interval (the venue's unit) and
  the code multiplies by `(8/24)` anyway, it DOUBLE-DISCOUNTS and biases `r_basis`
  low by ~3×, corrupting the no-arbitrage anchor. So the `(8/24)` factor is applied
  ONLY when the AMBER metadata declares the inputs are **per-day**; when they are
  **per-interval** (the venue-native case) `interest_8h` is the difference with NO
  rescale. The unit is pinned by an explicit AMBER field
  `interest_rate_unit ∈ {per_day, per_interval}` (no silent default — a missing
  unit is a config error, not an assumed `per_day`). The familiar "published
  0.01%/day → interest_8h ≈ 0.00333%" example holds ONLY for `per_day` inputs; with
  a `per_interval` 0.00333% input the result is 0.00333% with no further scaling.
  For non-Binance venues without an interest component, `interest_8h = 0`.
- The basis anchor is clamped to `[−cap, +cap]` for the same reason `r_hat` is:
  the realized funding it anchors toward cannot exceed the cap. If `perp_mark` or
  `index` is missing/stale, `r_basis` is undefined and the blend degrades to
  EWMA-only with a `fallback_no_basis` reason (P6); the gate does not silently use
  a zero basis.
- **The premium term: consume the venue PREMIUM INDEX, not a single mark/index
  snapshot (pinned, no hand-wave).** Binance's actual funding premium is a
  TIME-WEIGHTED average of `(impact_bid/ask − index)/index` over the interval, not
  a single instantaneous `(perp_mark − index)/index` sample. The earlier claim that
  "the mark is already 8h-relative / annualize-rescale round-trip cancels" is NOT
  generally true for one snapshot — it is an approximation that holds only if the
  snapshot equals the interval's time-weighted mean. To stay faithful to the
  no-arbitrage anchor without re-deriving the venue's index, COST consumes the
  **venue-published premium index** directly where it is exposed (AMBER
  `premium_index_8h`), and uses the instantaneous `(perp_mark − index)/index`
  snapshot only as a documented FALLBACK (reason `premium_from_snapshot`) when the
  published premium index is unavailable. The fallback is flagged so its
  approximation error is visible in realized-vs-expected telemetry (Invariant E),
  rather than asserted away.

> Without this mapping the "no-arbitrage anchor" of P4 is undefined and the basis
> term is unauditable. The interest baseline, the 8h scaling, and the cap are now
> all explicit and individually testable. (Invariant C, F, B.)

#### 5.5c Confidence band — correlated worst-case SUM over m stamps

The band is built from the **forecast-error residual series**
`e_k = r_k − r_hat(τ_k)` (realized rate at stamp `k` minus the forecast COST would
have made for that stamp one step earlier), accumulated in the validation layer
and frozen as a per-class residual distribution (TEAL behavior per instrument
where enough events exist, else class-level). The residual is a one-step-ahead
forecast error, not the raw rate series and not a multi-horizon error.

Funding is HIGHLY autocorrelated (regimes persist for many stamps), so the band
over `m` stamps must NOT be an `m`-independent-draw sum — that understates the tail
carry badly. The band is the **comonotone (perfectly-correlated) worst-case sum**:

```
band_quantile_q over m stamps:
  e_q = block-bootstrap q-quantile of the one-step residual series e_k   (block length L_funding, 5.5d)
  F_band(q) = Σ_{j=1}^{m}  side · Notional_expected(τ_j) · clamp( r_hat(τ_j) + e_q ,  −cap, +cap )
```
i.e. the SAME residual-quantile shock `e_q` is applied to every stamp and the
post-shock rates are re-clamped to the cap, then summed. `q` is RED (the
conservative quantile, e.g. the 90th percentile of the worse-for-the-position
tail). This is conservative by construction (it assumes the funding regime that
hurts persists across the whole hold, which is exactly what autocorrelated funding
does) and it never produces a band beyond what `m·cap·Notional` allows. The two
band fields are `funding_band_low` / `funding_band_high` (the `q` and `1−q`
comonotone sums). The independent-sum alternative is explicitly forbidden as it
understates tail carry. (Invariant G-band.)

**Why a one-step residual quantile is the right shock for an m-step comonotone
path (pinned, not asserted).** `e_k = r_k − r_hat(τ_k)` is the *one-step-ahead*
forecast error. A statistically honest band over an `m`-step path would need a
full joint error model `Var(Σ_j e_{τ_j})` with the residual autocovariance at all
lags — which is an ML-adjacent fitted object COST forbids (P4). The comonotone
construction is the deliberate CONSERVATIVE SUBSTITUTE for that m-step error
model: by applying the SAME single-step quantile shock `e_q` to all `m` stamps it
assumes the residuals are perfectly correlated across the hold (correlation = 1),
which is the upper bound on `Var(Σ e)` for any non-negative-correlation residual
process. Because realized funding residuals ARE strongly positively autocorrelated
(regimes persist), correlation 1 is close to truth and never optimistic. So the
band variance does not enter the gate as a fitted variance term; it enters as a
deterministic worst-case SUM (the `e_q`-shocked comonotone path), and that summed
shock is what the gate's conservative funding line consumes (Gap 1, below).

**Conservative band SELECTION is by SIDE, not a fixed high/low (Gap, sign).**
Adding `+e_q` to `r_hat` is "more positive funding," which hurts a LONG but HELPS
a SHORT. So `funding_band_low`/`funding_band_high` are NOT generically the
conservative/optimistic ends — their roles flip with side. Define the
**side-conservative band** as the comonotone endpoint that MAXIMIZES the position's
funding COST (P7), never a fixed `high`:

```
F_band_conservative(q)  =  max_{shock ∈ {+e_q, −e_q}}  Σ_{j=1}^{m}  side · Notional_expected(τ_j) · clamp( r_hat(τ_j) + shock , −cap, +cap )
```

Equivalently: choose the tail that maximizes `side · Notional · Σ r_hat` (the
comonotone `q` that INCREASES the position's signed funding outflow). For a LONG
(`side=+1`) this is the `+e_q` (more-positive-funding) tail; for a SHORT
(`side=−1`) it is the `−e_q` (more-negative-funding) tail. `funding_band_low` /
`funding_band_high` remain the raw `−e_q` / `+e_q` comonotone sums for telemetry,
but the GATE consumes `F_band_conservative` (5.5c-gate). A funding-EARNING short's
conservative band is a SMALLER credit (a less-negative carry), NEVER an inflated
positive cost — selecting a fixed `high` here would book a phantom cost on a
position that is being paid, which is the anti-gate P7 warns of. (Invariant P-band.)

> **Invariant P-band — confidence-band tail is selected by side, never fixed.**
> The gate's binding funding line is `F_band_conservative(q)` = the comonotone
> endpoint MAXIMIZING `side · Notional · Σ r_hat`, computed by side, not a hardwired
> `funding_band_high`. Test (mirrors Invariant P): a long and a short of equal
> notional in the same funding/basis/residual regime get conservative band
> selections EQUAL IN MAGNITUDE on their respective worse-for-position tails; a
> funding-earning short's conservative band is a smaller credit (magnitude below
> its point credit), never a positive inflated cost. Consuming a fixed
> `funding_band_high` unconditionally fails this test for the short.

**The band estimator `e_q`, its error bars, and its event-count floor (pinned).**
`e_q` is itself ESTIMATED from a finite residual sample and therefore carries
sampling error — the band must not be trusted from a handful of residuals any more
than the EWMA is (5.5a). Pin all three:

- **Estimator.** `e_q` = the `q`-quantile of the one-step residual series `{e_k}`,
  computed by the **block bootstrap** (block length `L_funding`, the SAME funding-
  decorrelation block shared with 5.5d and Section 12 — not a per-call choice): for
  each of `B` block-resamples take the empirical `q`-quantile, then report the
  bootstrap median as `e_q` and the bootstrap `[2.5%, 97.5%]` spread as its CI
  `[e_q_lo, e_q_hi]`. The block bootstrap (not an i.i.d. quantile) is required
  because residuals cluster (P4/Section 12).
- **Error bar enters the gate conservatively (P6).** The gate does not use the
  bootstrap-median `e_q`; it uses the **outer CI edge** of the worse-for-position
  quantile (`e_q_hi` for the long-side `+e_q` tail, the symmetric outer edge for
  the short-side `−e_q` tail), so a wide-CI (thin/unstable) residual sample yields a
  WIDER band and a harder gate, never a falsely tight one. The point `e_q` is
  telemetry; the CI-edge `e_q` is binding.
- **Event-count floor + fallback (the analogue of 5.5a's `n_funding_min`).** The
  residual quantile is trusted only when the residual series has at least
  `n_residual_min` one-step residuals (RED, e.g. `≥ 2·h` so the block bootstrap has
  multiple independent blocks). Below the floor the band FALLS BACK (P6) to a
  **class-level worst-residual** `e_q_class` (the conservative residual quantile
  pooled across the asset class, frozen offline), recorded as `fallback_band_thin`
  on `CostComponent.reason`. A thin residual series never yields a silently narrow
  band; it yields the class worst-case.

> **Invariant G-est — band quantile has a floor and a CI, and enters the gate at
> its conservative CI edge.** `e_q` is a block-bootstrap quantile with a reported
> CI; below `n_residual_min` residuals it falls back to the class-level worst
> residual (`fallback_band_thin`); the gate consumes the outer CI edge of the
> side-selected quantile (Invariant P-band), never the bare point quantile. Test:
> a 3-residual series yields the class fallback band (not a narrow empirical one)
> and a `fallback_band_thin` reason; widening the residual CI widens the gate's
> binding funding band monotonically.

#### 5.5d The `m` problem — funding stamp count from an UNMEASURABLE hold

`m`, the number of 8h stamps paid, is the single largest driver of `F_forecast`: a
1-stamp vs 4-stamp error swamps every other cost line. But `m` is a function of
`expected_exit`, and reference 17's drift power wall `t = SR·√T` says hold length /
edge is **unmeasurable intraday** — the sibling specs refuse to estimate it.
Therefore COST does NOT treat a point `expected_exit` as known. It prices funding
over a **range of holds** and reports the sensitivity:

```
m_max  = ceil( max_hold_quantile_ms / funding_interval_ms )   # conservative cap (P6), AMBER/RED max-hold quantile
F_forecast(m)  evaluated at m ∈ {1, 2, …, m_max}              # the whole curve, O(m_max), cheap
dF/dm  ≈ side · Notional_expected(τ_{m}) · r_hat(τ_{m})       # marginal carry of one more stamp
```
- The **gate** (8.1) prices funding at the **conservative `m_max`** (a caller- or
  class-supplied MAX-hold quantile, P6) — the worst-case stamp count, never an
  optimistic point estimate — so a trade must clear cost even if it is held to the
  plausible maximum. A caller MAY pass a smaller `m` only as additional diagnostic;
  the binding gate value uses `m_max`.
- `HoldHorizon.n_funding_stamps_remaining` is therefore the conservative `m_max`
  on the gate path; any point estimate from a caller is logged as
  `expected_m_caller` but does not relax the gate (Invariant M-hold).
- `dF/dm` (`funding_sensitivity` on the decision) is mandatory telemetry so the
  deciders SEE that funding is the dominant uncertain line and can widen their own
  margin. A trade whose verdict flips between `m=1` and `m=m_max` is flagged
  `funding_m_sensitive`.

> The hold horizon is supplied by the caller as a RANGE/quantile, never as a known
> scalar. COST's job is to make the funding cost of the *unmeasurable* hold
> conservative and its sensitivity visible — not to pretend the hold is known.
> (Invariant M-hold.)

**Notional over the forecast window (a second silent-lookahead hazard).**
`Notional_expected(τ_j)` for the GATE is the **causal worst-case-for-the-position**
assumption: the **current notional held flat to `m_max`** (the position never
shrinks). An "expected reduce schedule supplied by TREND" is a FUTURE/edge quantity
and is FORBIDDEN on the gate path — a reduce path that lowers future notional would
understate funding and is the same class of lookahead as reading future rates.
TREND's reduce intentions are unmeasurable edge (P2 firewall) and do not enter the
gate. The conservative-fallback rule (P6) governs: hold full notional to max-hold.
(A reduce schedule MAY be used only inside post-trade TCA's realized-vs-expected,
where the realized notional path is FACT.) (Invariant W, M-hold.)

### 5.6 Carry — ONE interface, per-class model (cross-asset)

The funding/carry concept generalizes across asset classes through a single
`CarryModel` interface with a per-class implementation selected by AMBER
`asset_class`. All are SIGNED relative to the position (P7) and CAUSAL.

| Class                  | Carry mechanism                       | Per-class formula (per unit time, signed)                                  |
|------------------------|---------------------------------------|----------------------------------------------------------------------------|
| Crypto perp (Binance)  | 8h funding                            | `side · Notional · r_hat` summed over 8h stamps (5.5)                        |
| Dated futures          | roll / contango-backwardation basis   | `side · Notional · (F_far − F_near)/F_near · (Δt / τ_spread)` over hold Δt; roll-gap on roll (5.6-time) |
| US equities (MTF)      | margin financing interest             | `+|borrowed_notional| · mtf_rate · Δt/365` (financing is a COST ⇒ POSITIVE, P7, like perp/futures/storage; cash-long borrows 0) |
| Commodities/metals     | storage + financing − convenience (basis)| see 5.6-commodity below (financing charged ONCE, P7)                       |

- **Perp** routes to `FundingForecast` (5.5).
- **Dated futures** carry is the inter-contract basis between the near and next
  contract, **time-scaled to the hold** (5.6-time below — the raw spread fraction
  is NOT the carry over an arbitrary hold), plus an explicit **roll-gap cost**
  booked when a roll occurs within the hold (the price discontinuity Nautilus's
  stitched series records). Contango is a cost to a long roll, a benefit to a short.
- **MTF equities** carry is the financing interest on the borrowed portion of the
  position (the leverage above 1×), at the broker MTF rate (AMBER), accrued daily
  — directly the same number LEVERAGE.md subtracts in its stop ceiling. It is
  booked **positive = cost** (P7), identical in sign to perp/futures/storage, so
  it enters `(c − max(carry,0))` and TIGHTENS the stop ceiling `L_stop=(c−max(carry,0))/d` (a
  leading minus would make a financing cost RELAX the ceiling — the anti-gate P7
  forbids). Because MTF financing is side-symmetric (`|borrowed_notional|`,
  charged whether long or short), the long/short-antisymmetry Invariant P does
  NOT apply to it; a dedicated sign test instead asserts an MTF charge yields a
  POSITIVE carry that tightens (never relaxes) `L_stop`.

**5.6-commodity — the Dukascopy proxy is a SPOT/CFD price series, so do not
charge financing twice (Invariant Y-finance).** The retail commodity/metals
universe here is Dukascopy-proxied (NatGas/Crude as CFD-like spot price series),
not margined exchange futures. Financing must be booked in exactly ONE place:

| Proxy financing mode (AMBER `commodity_financing_mode`) | Carry formula (signed; day-count per 5.6-time)                          |
|---------------------------------------------------------|----------------------------------------------------------------|
| `unlevered` (cash proxy, no leverage)                   | `side · Notional · (storage_rate / price) · (Δt_days/365)` — financing = 0  |
| `mtf` (the proxy IS held on broker/MTF financing)       | financing booked by the **MTF path** `+|borrowed_notional|·mtf_rate·(Δt_days/365)` (positive=cost); the commodity carry then carries ONLY `side·Notional·(storage_rate/price)·(Δt_days/365)` (storage, no financing) |

In neither mode does the commodity formula include its own `financing` term: an
unlevered proxy pays none, and a financed proxy pays it through the MTF path. The
earlier `(storage + financing − convenience)/price · Δt` double-counted financing
(once in carry, once in any MTF term) and is replaced by the table above. A test
asserts that turning on MTF financing changes total carry by exactly the MTF line
and NOT by an extra commodity-financing term.

> **`storage_rate` units (pinned).** `storage_rate` is AMBER and is **quote
> currency per unit of the underlying per YEAR** (e.g. USD per barrel per year). So
> `storage_rate / price` is a dimensionless ANNUAL fraction of notional, and
> `(storage_rate/price)·(Δt_days/365)` is the dimensionless fraction over the hold;
> multiplied by `Notional` it is quote-ccy carry. This is why the same `Δt_days/365`
> day-count appears on BOTH the storage line and the MTF line (5.6-time) — the two
> carry lines MUST use the identical time convention or a financed-commodity hold
> books storage and financing on inconsistent clocks. A bare `Δt` (no /365) on the
> storage line, as earlier prose had, made storage a per-(unit-time) rate with
> unstated units while MTF was annualized — that inconsistency is removed.

**Convenience yield is usually NOT observable for the retail commodity universe —
fall back to basis-only (P6).** Convenience yield is not directly quotable for a
retail Dukascopy NatGas/Crude proxy. When an observed near/far futures basis is
available (AMBER `commodity_basis_observable=True`), use the **basis directly** as
the net cost-of-carry (it already embeds storage + financing − convenience by
no-arbitrage): `carry = side · Notional · (F_far − F_near)/F_near · (Δt / τ_spread)`
over the hold — the **same time-scaled observed-basis route as dated futures**
(5.6-time; the raw spread fraction is NOT the carry over an arbitrary hold), with
NO separate convenience term to estimate. Only when no basis is observable does it
fall back to the
storage-only formula above with `convenience = 0` (a conservative HIGHER cost to a
long, per P6) and a `fallback_no_convenience` reason. COST never estimates an
unobservable convenience yield component-by-component.

#### 5.6-time — time-scaling, day-count, and simple-vs-compound (pinned for ALL carry classes)

Carry is an accrual over a hold of length `Δt`. The earlier table called the
dated-futures/basis line "annualized basis" yet wrote the RAW inter-contract spread
fraction with no time factor — that books the ENTIRE inter-contract carry no matter
how short the hold (wrong for a sub-`τ` hold) and is dimensionally a per-unit-time
rate with no stated period. Pin the conventions once, used by every class:

- **Inter-contract basis is time-scaled (dated futures + observed-commodity-basis).**
  ```
  carry_basis = side · Notional · (F_far − F_near)/F_near · (Δt / τ_spread)
  ```
  where `τ_spread` is the **calendar time between the two contracts' expiries**
  (`expiry_far − expiry_near`, AMBER from the roll/expiry calendar) and `Δt` the
  hold, **both in the same units** (days). `(F_far − F_near)/F_near` is the basis
  earned/paid over the FULL `τ_spread`; scaling by `Δt/τ_spread` accrues the
  pro-rata fraction actually earned over the hold. This reconciles the "annualized
  basis" prose: the per-year rate is `(F_far−F_near)/F_near · (365/τ_spread_days)`,
  and the hold carry is that annual rate × `Δt_days/365` — which simplifies exactly
  to the formula above. At `Δt = τ_spread` (held to the far expiry) the full
  inter-contract carry is booked; for a short intraday hold it is a small fraction,
  as it must be. The **roll-gap** cost is separate and is booked as a discrete
  event when a roll actually occurs in the hold, NOT scaled by `Δt`.
- **MTF financing is SIMPLE interest, Actual/365 (pinned — the "compounds" prose
  was wrong).**
  ```
  carry_mtf = +|borrowed_notional| · mtf_rate · (Δt_days / 365)   # positive = COST (P7)
  ```
  This is SIMPLE interest, booked **positive = cost** (P7) — identical in sign to
  the §5.6 class table and the §5.6-commodity `mtf` row, so it enters `(c − max(carry,0))`
  and TIGHTENS the stop ceiling `L_stop=(c−max(carry,0))/d` (a leading minus would make a
  financing cost RELAX it — the anti-gate P7 forbids). Earlier §14 prose said MTF financing "compounds over
  week-long holds" — that contradicts the formula and is DELETED: for a retail
  week-long hold the compounding correction on a ~9–18% p.a. rate is a few parts in
  ten-thousand of the financing line, far below the cost model's tolerance, and the
  simple form is the auditable one. Day-count is pinned **Actual/365** (calendar
  days, `mtf_rate` is the annual nominal rate; NOT 360, NOT trading days). If a
  venue genuinely compounds and the difference matters, the compounded form is
  `+borrowed · ((1 + mtf_rate/365)^Δt_days − 1)` (also positive = cost) and MUST be selected explicitly by
  an AMBER `mtf_compounding=True` flag — never silently assumed; the default is
  simple.
- **Storage is the same Actual/365 annual fraction** (5.6-commodity above):
  `side · Notional · (storage_rate/price) · (Δt_days/365)`, `storage_rate` in
  quote-ccy per unit per YEAR. The /365 makes it consistent with the MTF line so a
  financed-commodity hold accrues storage and financing on the IDENTICAL clock.
- **Perp funding is on the STAMP clock, not /365.** Funding accrues per 8h stamp
  (5.5); its time unit is the stamp count `m`, and `interest_8h`'s own day-count is
  pinned separately in 5.5b. Perp carry is the ONE class not on Actual/365, by
  design — the stamp schedule is its native clock.

> **Invariant H-time — every carry class states its time-scaling, day-count, and
> simple/compound convention; no bare `Δt`.** Inter-contract basis carry scales by
> `Δt/τ_spread` (`τ_spread` = far−near expiry gap); MTF and storage use
> Actual/365 SIMPLE accrual (compounding only via explicit AMBER
> `mtf_compounding`); perp uses the stamp clock. Tests: (1) a basis hold of
> `Δt = τ_spread/2` books exactly half the full inter-contract carry, and `Δt→0`
> books `→0` (not the full spread); (2) MTF financing over 7 days equals
> `borrowed·mtf_rate·7/365` to the model tolerance (simple, not compounded) **AND
> is POSITIVE** (a cost, P7) — the SIGN is asserted, not only the magnitude, so a
> stray leading minus on any of the three MTF statements (class table, commodity
> `mtf` row, §5.6-time block) is caught; (3)
> storage and MTF over the same `Δt` use the same `Δt_days/365` factor. A bare
> spread fraction or a bare `Δt` with no period is the bug.

> **Invariant X — one carry interface, per-class body, no forked callers.** Every
> consumer calls `CarryModel.carry(...)`; the class branch is INSIDE the model,
> chosen by AMBER metadata. No consumer special-cases crypto vs equity. A test
> runs the same consumer code path across all four classes with only metadata
> changing.

### 5.7 End-to-end worked example (reproduce this bit-for-bit)

One fully-specified round trip carried through EVERY cost line, in both quote
currency and bps, so a builder can assert against it. It pins the `/1e4`
(half-spread divides, impact does NOT), the fraction-vs-bps convention, the
`interest_8h` unit, the side-selected conservative funding band, and the
point-vs-band gate distinction concretely. Every number below is reproducible from
the inputs with no hidden step.

**Trade and market state (all `F_t`-measurable):**

| Input | Value |
|---|---|
| asset_class | `crypto_perp` (Binance USDT-M), `side = +1` (LONG) |
| Notional | `N = 1000.00` USDT, both legs (held flat to `m_max`) |
| liquidity flags | taker in, taker out; VIP-0 |
| fee schedule | VIP-0 taker `= 0.0005` (a FRACTION = 5.0 bps) |
| bid / ask / mid | `49995 / 50005 / 50000` |
| impact `k_i`, `σ_i` (per-bar) | `k_i = 0.10` (fraction units), `σ_i = 0.004` |
| `v_hat` (causal, per leg window) | `200000` USDT |
| EWMA half-life | `h = 9` stamps → `α = 1 − 2^(−1/9) = 0.074125`, `ESS = 25.98` |
| `r_ewma` (after recursion) | `0.00012` |
| perp_mark / index | `50010 / 50000` → `premium_8h = 0.0002` |
| `interest_rate_unit`, interest diff | `per_interval`, `interest_8h = 0.0000333` (NO `(8/24)`) |
| basis shrink `λ`, cap | `λ = 0.30`, `cap = 0.0075` |
| `published_next_funding` (j==1) | `0.00015` |
| band CI-edge residual `e_q` | `0.00008` (long side: `+e_q` tail) |
| `m_max`, `edge_buffer` | `3` stamps, `2.00` USDT |

**Line by line (no `/1e4` on impact; the fraction multiplies notional directly):**

```
C_fee_in   = 0.0005 · 1000               = 0.50000 USDT   = 5.0000  bps
C_fee_out  = 0.0005 · 1000               = 0.50000 USDT   = 5.0000  bps
half_spread_bps = 1e4·(50005−49995)/(2·50000) = 1.0 bps
C_hs_in    = (1.0/1e4) · 1000            = 0.10000 USDT   = 1.0000  bps   # /1e4 CORRECT (genuinely bps)
C_hs_out   = (1.0/1e4) · 1000            = 0.10000 USDT   = 1.0000  bps
participation = 1000/200000 = 0.005;  sqrt = 0.0707107
impact_frac = 0.10 · 0.004 · 0.0707107   = 0.0000282843  (a FRACTION, NOT bps)
C_impact_in  = impact_frac · 1000        = 0.0282843 USDT = 0.282843 bps  # NO /1e4 (U-dim)
C_impact_out = impact_frac · 1000        = 0.0282843 USDT = 0.282843 bps
```

**Funding forecast (5.5), m_max = 3 stamps, notional flat:**

```
r_basis = clamp(interest_8h + premium_8h, −cap,+cap) = clamp(0.0000333+0.0002) = 0.0002333
r_blend = clamp((1−0.30)·0.00012 + 0.30·0.0002333) = 0.00015399          # stamps j=2,3
r_hat(τ1) = clamp(published_next) = 0.00015                              # j==1 OVERRIDE, not added on top
r_hat = [0.00015, 0.00015399, 0.00015399]
F_forecast(m=3) POINT = Σ side·N·r_hat = 1000·(0.00015+0.00015399+0.00015399)
                      = 0.45798 USDT  = 4.5798 bps         # telemetry, NOT the gate summand
dF/dm = side·N·r_hat(τ3) = 1000·0.00015399 = 0.15399 USDT  # marginal carry of one more stamp
```

**Side-selected conservative funding band (5.5c, long → `+e_q` tail, CI edge):**

```
F_band_conservative = Σ side·N·clamp(r_hat(τj) + 0.00008, −cap,+cap)
                    = 1000·(0.00023 + 0.00023399 + 0.00023399)
                    = 0.69798 USDT  = 6.9798 bps           # THIS is the gate's binding funding line
```

**Round-trip total (the GATE uses the BAND, not the point):**

```
C_round_trip(BAND)  = 0.5+0.5+0.1+0.1+0.0282843+0.0282843+0.69798 = 1.9545485 USDT = 19.5455 bps
C_round_trip(POINT) = 0.5+0.5+0.1+0.1+0.0282843+0.0282843+0.45798 = 1.7145485 USDT = 17.1455 bps  (diagnostic only)
```

**Gate verdict (the marginal-trade demonstration for Invariant F-band-gate).** Take
`expected_favorable = 3.85` USDT, `edge_buffer = 2.00`:

```
passes-at-POINT would need E > 1.7145485 + 2.00 = 3.7145485  →  3.85 > 3.71  → would PASS (WRONG; ignores rate uncertainty)
passes-at-BAND        needs E > 1.9545485 + 2.00 = 3.9545485  →  3.85 < 3.95  → FAILS  (correct; gate prices the band)
margin = 3.85 − 1.9545485 − 2.00 = −0.1045485  →  passes = False
binding_funding = 0.69798  (== funding_band_conservative, NOT funding_forecast 0.45798)
funding_sensitivity = dF/dm = 0.15399;  funding_m_sensitive = (verdict differs between m=1 and m=3)
```

This trade is exactly the headline gap made concrete: priced at the POINT funding
forecast it passes; priced at the side-selected conservative BAND it fails. The
gate must return `passes = False`. (Note the `per_interval` interest unit: had it
been `per_day` at `0.01%/day`, `interest_8h` would be `0.0001·(8/24) = 0.0000333`
— numerically identical here ONLY because the example chose matching inputs; the
unit flag, not the number, is what prevents the ~3× double-discount of 5.5b.)

> **Invariant WE — the worked example is a regression fixture.** A test reproduces
> every number in 5.7 to within floating tolerance from the tabulated inputs:
> `C_fee_*`, `C_hs_*` (with `/1e4`), `C_impact_*` (a fraction × notional, NO `/1e4`,
> equal to `0.0282843` not `2.83e-6`), `F_forecast(3)`, `F_band_conservative`,
> `dF/dm`, both totals, `binding_funding == funding_band_conservative`, and
> `passes == False`. If any constant in the spec changes, this fixture changes with
> it — it is the single anchor pinning the dimensional and band conventions.

---

## 6. Cost-as-fraction-of-edge

A single decomposable diagnostic the deciders and reports use:

```
cost_fraction = C_round_trip / E_favorable_excursion
```
where `E_favorable_excursion` is the expected favorable move in quote currency
supplied by the caller (BREAKOUT's stop-distance-scaled expectation, or TREND's
residual-move estimate). `cost_fraction < 1` is necessary (not sufficient) for a
trade to be worth taking. It is REPORTED with a RED rejection threshold for
monitoring, but the **gate itself uses the raw inequality** (Section 7-gate),
not the ratio, to avoid divide-by-near-zero pathology when edge is tiny. The
ratio is for human/diagnostic consumption and for ablation, not the live boolean.

---

## 7. Fee-tier optimization (rolling volume → tier → effective fee)

```python
class FeeModel:
    def effective_fees(self, rolling_30d_volume: float, fee_schedule: "FeeSchedule",
                       cfg: "ResolvedCostConfig") -> tuple[float, float, dict]:
        """Map rolling 30-day traded volume to the VIP tier, then to the effective
        (maker_rate, taker_rate). rolling_30d_volume is FACT from Nautilus fills
        (F_t-measurable). Tier is a step function over AMBER volume breakpoints;
        no interpolation, no learned schedule. Returns (maker_rate, taker_rate,
        diagnostics{tier, next_breakpoint, volume_to_next}). The volume_to_next
        telemetry surfaces how close the book is to a cheaper tier — informational
        only, never a trade trigger (Invariant D)."""
```

The tier is a deterministic step function; the effective fee feeds 5.2. For a
~$1000 retail book the realistic tier is the base (VIP 0), but the model is built
so the same code reflects a tier change without a rewrite, and so TCA attributes
fee cost at the tier that was actually in force at fill time (FACT).

**The GATE's tier selection is conservative near a breakpoint (Gap, fee-tier
discontinuity).** `effective_fees` above is the FORWARD lookup: given a volume, the
tier in force. But the GATE must choose WHICH tier to PRICE the round-trip fee at,
and that is a forecast question, not a lookup: `rolling_30d_volume` can cross a VIP
breakpoint mid-life, or a trade can sit exactly ON a breakpoint (fee-ambiguous). Per
P6 (conservative under uncertainty), the gate's tier selection is pinned:

```
tier_gate = worse_fee_tier( tier(rolling_30d_volume),
                            tier(rolling_30d_volume) one step UP in fee )   if |volume − nearest_breakpoint| < tier_proximity
            tier(rolling_30d_volume)                                         otherwise
```

- The gate prices the round-trip fee at the **HIGHER-FEE (worse) tier** whenever
  `rolling_30d_volume` is within `tier_proximity` (RED, a fraction of the
  breakpoint, or an absolute volume band) of a VIP breakpoint — including sitting
  exactly on it. Away from any breakpoint it uses the plain step lookup.
- `volume_to_next` (already emitted as telemetry) is the input to this proximity
  test — it stops being pure telemetry and FEEDS the conservative selection. (It
  still never triggers a trade; it only makes the FEE estimate conservative —
  Invariant D intact.)
- TCA still attributes the REALIZED fee at the tier actually in force at fill time
  (FACT); the conservative selection is a GATE-side forecast device only, never
  applied to realized accounting.

---

## 8. The two gates COST hands the deciders (P2 — provider, not decider)

### 8.1 Net-of-cost expected-value gate (consumed by BREAKOUT)

```python
class NetEVGate:
    def evaluate(self, side: "Side", notional: float, expected_favorable: float,
                 liquidity_flag_in: str, liquidity_flag_out: str,
                 expected_hold: "HoldHorizon", market: "CostMarketState",
                 cfg: "ResolvedCostConfig") -> "NetEVDecision":
        """Take-the-trade-only-if gate. Returns NetEVDecision with:
          passes = expected_favorable
                   > C_round_trip(funding priced at the side-selected CONSERVATIVE
                                  BAND F_band_conservative(m_max, q), not the point)
                     + edge_buffer (RED margin of safety)
        and the FULL additive decomposition of C_round_trip (every line, P3),
        plus cost_fraction (Section 6), plus reason_codes. Funding is priced at the
        CONSERVATIVE m_max stamp count from expected_hold (5.5d) — never an
        optimistic point hold — AND, within those stamps, at the side-selected
        conservative confidence band (5.5c, Invariant F-band-gate / P-band), NOT
        the point blend F_forecast: forecast funding is uncertain WITHIN each stamp
        and the gate must clear its upper-CI cost. The point F_forecast(m_max) is
        carried as telemetry only. The decision carries funding_sensitivity (dF/dm)
        and funding_m_sensitive. Impact uses the CAUSAL volume forecast v_hat
        (5.4, Invariant W); forecast notional is current notional held flat to
        m_max (no TREND reduce schedule on the gate, 5.5d). expected_favorable is
        supplied by BREAKOUT (an EDGE quantity owned upstream — COST does NOT
        estimate edge). This method NEVER opens a trade; it returns a boolean and
        the math. (Invariant D, B, W, M-hold.)"""
```

The inequality is additive on both sides (no product). The `edge_buffer` is a RED
margin of safety so a trade that merely ties cost is not taken. When any cost
component is missing/stale, the conservative fallback (P6) inflates
`C_round_trip`, so the gate fails closed.

**The gate's binding funding line is the BAND, not the point (Gap 1, pinned).**
The funding component summed into `C_round_trip` for the `passes` boolean is
`C_funding := F_band_conservative(m_max, q)` (5.5c, side-selected per Invariant
P-band, at its conservative CI edge per Invariant G-est) — NOT the point blend
`F_forecast(m_max)`. The spec already handles the STAMP-COUNT uncertainty
conservatively (`m_max`, 5.5d); the band is what handles the RATE uncertainty
WITHIN each stamp. `NetEVDecision` carries `binding_funding` (the band value
actually summed into the verdict) as a distinct field from the point
`round_trip.funding_forecast`, so the diagnostic and the binding number are never
confused. Pricing funding at the point estimate when the band would have failed
the trade is the headline bug this closes.

> **Invariant F-band-gate — the gate prices funding at the conservative band, not
> the point.** `NetEVGate.passes` consumes `binding_funding =
> F_band_conservative(m_max, q)` (the side-selected, CI-edge comonotone band of
> 5.5c), never `F_forecast(m_max)`. Test: a MARGINAL trade engineered so
> `expected_favorable > total` with funding at the POINT estimate but
> `expected_favorable < total` with funding at the conservative band FAILS the gate
> — i.e. dropping the band back to the point would flip `passes` from False to True.
> A second test asserts `NetEVDecision.binding_funding == funding_band_conservative`
> and that it is the side-correct tail (Invariant P-band). The point
> `funding_forecast` is present but is NOT the summand in `passes`.

### 8.2 Funding-aware hold / reduce input (consumed by TREND)

```python
class HoldCarryInput:
    def residual(self, side: "Side", notional: float, remaining_hold: "HoldHorizon",
                 expected_residual_favorable: float, market: "CostMarketState",
                 cfg: "ResolvedCostConfig") -> "HoldCarryDecision":
        """Funding-aware HOLD input (not a hold decision — P2). Returns:
          forecast_residual_carry  = side-selected CONSERVATIVE-BAND funding+carry over remaining_hold
          expected_residual_move   = supplied by TREND (edge, owned upstream)
          carry_exceeds_residual   = forecast_residual_carry > expected_residual_favorable
        TREND consumes carry_exceeds_residual as ONE input to its hold/reduce/exit
        policy: 'stop holding when forecast carry over remaining life exceeds the
        expected residual favorable move.' COST does not exit the position; it
        states the carry-vs-move relationship and hands it back. The forecast
        carry uses the CAUSAL funding forecast (5.5) priced over the residual
        hold's conservative m_max (5.5d) at the side-selected conservative band
        (5.5c, Invariant P-band/G-est) — the SAME band the gate uses (Invariant T),
        NOT the point forecast — so TREND sees the worse-for-position carry; the
        decision also carries funding_sensitivity so TREND sees the m-uncertainty.
        Signed (P7): a funding-earning short's conservative band is a SMALLER
        credit, never an inflated cost (Invariant P-band).
        (Invariant D, C, G-band, P-band, M-hold, T.)"""
```

This is the funding-side analogue of TREND's death detector: even a still-alive
trend should be released when its forecast carry-to-hold eats the residual move.
Crucially it is signed (P7): a SHORT perp earning funding has NEGATIVE residual
carry, which RELAXES the hold constraint — the input correctly says "carry is
paying you to hold," and TREND can lengthen the hold.

### 8.3 Carry term (consumed by LEVERAGE)

```python
class CarryTerm:
    def over_horizon(self, side: "Side", notional: float, horizon: "HoldHorizon",
                     market: "CostMarketState", cfg: "ResolvedCostConfig"
                     ) -> tuple[float, dict]:
        """Return (carry_quote_ccy, diagnostics) — the forecast funding+carry over
        the holding horizon, SIGNED (a credit is negative). One signed truth (P1) with
        TWO consumers: the net-EV gate (8.1) and the funding-aware hold gate (8.2) consume
        it SIGNED (a credit is real expected income there); the survival stop-ceiling
        consumes only its COST side — LEVERAGE.md's StopCeiling subtracts the CLAMPED carry,
        L_stop = (c − max(carry,0))/d. A positive carry (a cost) tightens the achievable
        leverage; a credit does NOT relax the hard survival cap (it is an unrealized funding
        FORECAST that can flip sign mid-hold, and its EV is already captured in the gate, so
        it must not buy survival headroom). Same forecast as 8.1/8.2 (Invariant T)."""
```

> **Invariant D — COST emits no actions.** No method in `consume/`, `assemble/`,
> or `components/` calls any order/position/cancel API or returns an action enum.
> Returns are numbers, booleans, decompositions, and reason codes only. A test
> greps the live layers for order/close/cancel calls and asserts none exist.

---

## 9. Post-trade TCA — implementation-shortfall decomposition (consumed for review)

```python
class TCA:
    def decompose(self, trade: "ClosedTrade", marks: "ReferenceMarks",
                  nautilus_facts: "NautilusFacts", execution_facts: "ExecutionFacts",
                  cfg: "ResolvedCostConfig") -> "TCAReport":
        """Implementation-shortfall decomposition of a CLOSED trade. Total IS =
        (decision price -> final P&L) gap, attributed to additive, named lines.
        The decomposition is an EXACT accounting IDENTITY by construction (not a
        within-tolerance reconciliation): the named cost lines plus an explicit
        closing residual sum to (gross − net) with ZERO residual by definition.

          gross_minus_net = delay_cost        # decision_price -> arrival_price (signal->actionable lag)
                          + fee_cost           # realized maker/taker fees (FACT, = Nautilus Commission)
                          + half_spread_cost   # realized touch-vs-mid on each leg
                          + impact_cost        # realized slippage/impact (FACT, from EXECUTION)
                          + funding_cost       # F_accrued (FACT, summed from Nautilus funding events, 5.5)
                          + carry_cost         # realized roll/MTF/storage carry over the hold
                          + opportunity_cost   # unfilled/partial-fill or missed-portion shortfall
                          + unattributed_residual   # the explicit PLUG that closes the identity EXACTLY

        SIGN CONVENTION (pinned): every cost line is POSITIVE when it REDUCES P&L
        (money out / shortfall), so the lines SUM to the measured (gross − net)
        gap, which is itself defined positive when net < gross (costs ate P&L). A
        funding CREDIT (short earning funding) is a NEGATIVE cost line (it widens
        net toward gross). `unattributed_residual` is computed as
        (gross − net) − Σ(the seven named lines): it ABSORBS whatever the seven
        lines did not, so the eight-line sum equals (gross − net) EXACTLY, by
        construction, for every trade — there is no open sum that can drift.

        ALL named terms FACT (realized), never forecast. Each in quote ccy and bps.
        Also returns realized-vs-expected per line (the forecast each gate used at
        entry vs what actually happened) so forecast bias is measurable and feeds
        the validation layer. Attribution sums to 100% of IS."""
```

> **Invariant R — TCA is an EXACT identity; the residual is a tested DIAGNOSTIC,
> not a tolerance band.** The eight IS lines (seven named + `unattributed_residual`)
> sum to the measured (gross P&L − net P&L) gap with EXACTLY zero residual for
> every closed trade, because `unattributed_residual` is defined as
> (gross − net) − Σ(seven named lines) — the identity holds by construction, not
> "within tolerance." The DIAGNOSTIC test is then on the MAGNITUDE of the plug:
> `|unattributed_residual|` below a fixed RED tolerance means the seven named lines
> captured the gap; a plug ABOVE tolerance is a missing or double-counted named
> line and fails the test. (Testing an exact-by-construction identity ≈ 0 is
> vacuous; testing the plug magnitude is the real check — and unlike an open
> within-tolerance sum, a real missing line cannot hide as an un-named residual,
> because the residual is named and asserted.) Sign convention is pinned in
> Section 9: each cost line is positive when P&L-reducing, summing to the positive
> (gross − net) gap. `funding_cost` equals `FundingAccrual` (FACT), never the
> forecast.

> **Invariant E — realized-vs-expected is mandatory telemetry.** Every gate
> records the forecast it used (forecast funding, forecast impact, forecast
> spread). TCA pairs each with the realized value so forecast bias per component
> is tracked over time and fed to validation. A systematically negative
> funding-forecast bias means the EWMA/basis blend is miscalibrated.

---

## 10. Shared data objects (`types.py`)

```python
from dataclasses import dataclass
from typing import Literal, Mapping
import numpy as np

Side = Literal[-1, 1]
LiquidityFlag = Literal["maker", "taker"]
AssetClass = Literal["crypto_perp", "dated_future", "equity_mtf", "commodity"]

@dataclass(frozen=True)
class HoldHorizon:                 # hold length is UNMEASURABLE (5.5d, ref 17) -> a RANGE
    max_hold_quantile_ms: int      # conservative max-hold quantile (P6); drives the GATE
    funding_interval_ms: int       # AMBER (Binance USDT-M = 8h)
    m_max: int                     # = ceil(max_hold_quantile_ms / funding_interval_ms); gate stamp count
    expected_m_caller: int | None  # optional caller point estimate, DIAGNOSTIC ONLY (Invariant M-hold)
    # NOTE: no scalar expected_exit drives the gate; funding is priced to m_max and
    # dF/dm is reported. n_funding_stamps_remaining (gate) == m_max.
    @property
    def n_funding_stamps_remaining(self) -> int:  # gate path uses the conservative count
        return self.m_max

@dataclass(frozen=True)
class CostMarketState:             # all F_t-measurable (causal)
    mid: float
    bid: float | None
    ask: float | None
    perp_mark: float | None        # for basis (5.5b)
    index_price: float | None      # for basis (5.5b)
    funding_cap: float             # AMBER venue per-interval cap; r_hat clamped to [-cap,+cap]
    interest_quote: float          # AMBER interest-baseline component (5.5b)
    interest_base: float           # AMBER interest-baseline component (5.5b)
    interest_rate_unit: Literal["per_day", "per_interval"]  # AMBER; gates the (8/24) scaling (5.5b); NO silent default
    premium_index_8h: float | None # AMBER venue-published time-weighted premium index (5.5b); snapshot fallback if None
    published_next_funding: float | None  # venue pre-published rate (Binance); j==1 OVERRIDE (5.5)
    recent_funding_rates: np.ndarray       # realized history -> EWMA (5.5a)
    recent_funding_times: np.ndarray
    n_funding_events: int          # for the event-count floor (5.5a); < floor -> fallback
    sigma: float                   # PER-BAR return vol for impact (5.4), same clock as k_i
    v_hat_exit_window: float       # CAUSAL forecast notional volume over the leg window (5.4, Invariant W)
    rolling_30d_volume: float      # for fee tier (FACT from Nautilus fills)
    futures_basis: float | None    # observed near/far basis for dated futures + commodity (5.6)
    mtf_rate: float | None         # AMBER, equities + financed commodity proxy
    storage_rate: float | None     # AMBER, commodities
    commodity_financing_mode: Literal["unlevered", "mtf"] | None  # AMBER (5.6-commodity)
    commodity_basis_observable: bool  # AMBER; if False, storage-only fallback, convenience=0 (P6)
    spread_regime: Literal["calm", "stressed"]
    stale_flags: tuple[str, ...]   # any input stale -> conservative fallback (P6)

@dataclass(frozen=True)
class CostComponent:               # one additive line, dual-unit
    name: str                      # 'fee_in','half_spread_out','funding_forecast',...
    quote_ccy: float
    bps_of_notional: float
    is_fact: bool                  # True=realized FACT, False=forecast ESTIMATE (P5)
    reason: str

@dataclass(frozen=True)
class RoundTripCost:               # additive decomposition (P3), never a product
    components: tuple[CostComponent, ...]
    total_quote_ccy: float         # = sum(c.quote_ccy)  (Invariant B)
    total_bps: float
    funding_forecast: float        # POINT blend, signed (P7), priced to m_max (5.5d); TELEMETRY, not the gate summand
    funding_band_low: float        # raw −e_q comonotone worst-case sum over m stamps (5.5c)
    funding_band_high: float       # raw +e_q comonotone worst-case sum over m stamps (5.5c)
    funding_band_conservative: float  # SIDE-SELECTED, CI-edge band (Invariant P-band/G-est); the GATE summand
    cost_fraction: float | None    # Section 6, None if edge not supplied

@dataclass(frozen=True)
class NetEVDecision:               # to BREAKOUT (numbers, not an action — P2)
    passes: bool                   # priced at conservative m_max (5.5d) AND conservative band (Invariant F-band-gate)
    expected_favorable: float
    round_trip: RoundTripCost
    binding_funding: float         # = round_trip.funding_band_conservative; the funding line actually summed into passes (Gap 1)
    edge_buffer: float
    margin: float                  # expected_favorable - total - buffer  (total uses binding_funding, not the point forecast)
    funding_sensitivity: float     # dF/dm at m_max (5.5d), mandatory telemetry
    funding_m_sensitive: bool      # True if verdict flips between m=1 and m=m_max
    reason_codes: tuple[str, ...]

@dataclass(frozen=True)
class HoldCarryDecision:           # to TREND (numbers, not an action — P2)
    forecast_residual_carry: float # signed, priced to conservative m_max (5.5d) AND side-selected conservative band (P-band)
    expected_residual_favorable: float
    carry_exceeds_residual: bool   # compares the side-selected conservative band carry vs the residual move
    funding_band_low: float        # raw −e_q comonotone worst-case sum (5.5c)
    funding_band_high: float       # raw +e_q comonotone worst-case sum (5.5c)
    funding_band_conservative: float  # side-selected, CI-edge band (Invariant P-band/G-est)
    funding_sensitivity: float     # dF/dm (5.5d), so TREND sees the m-uncertainty
    reason_codes: tuple[str, ...]

@dataclass(frozen=True)
class TCAReport:                   # post-trade, all FACT (Invariant R)
    delay_cost: float
    fee_cost: float
    half_spread_cost: float
    impact_cost: float
    funding_cost: float            # = FundingAccrual (FACT)
    carry_cost: float
    opportunity_cost: float
    unattributed_residual: float   # explicit PLUG = (gross−net) − Σ(7 named); closes the identity EXACTLY (Invariant R)
    implementation_shortfall: float  # = (gross − net); == sum of the 8 lines above, by construction
    realized_vs_expected: Mapping[str, tuple[float, float]]  # line -> (forecast, realized)
    attribution_pct: Mapping[str, float]                     # sums to 1.0
    engine_version: str
```

---

## 11. Invariants (enforce structurally; ask before breaking)

**Invariant A — accrued FACT vs forecast ESTIMATE are separate fields.** Accrued
funding (realized, summed from Nautilus events) and forecast funding (estimated)
are never merged into one unlabelled number. `CostComponent.is_fact` distinguishes
them. Test: realized TCA uses only `is_fact=True` lines; the gates use the
forecast field; no code path sums an accrued and a forecast funding into one.

**Invariant B — total is an additive sum, never a product.** `RoundTripCost.total
_quote_ccy == sum(c.quote_ccy for c in components)` exactly. No multiplicative
"cost score" exists anywhere. The gate inequality compares two additive sums.
Test: perturbing one component changes the total by exactly that delta.

**Invariant C — funding/carry forecast is causal.** Every input to
`FundingForecast`/`CarryModel` is `F_t`-measurable. The deterministic funding
SCHEDULE may be read ahead (which 8h boundaries fall in the window); no future
realized RATE may be. Test: replacing all funding rates after `t` with garbage
leaves the live forecast unchanged.

**Invariant D — COST emits no actions.** (Section 8.) Live layers contain no
order/position/cancel call and return no action enum.

**Invariant E — realized-vs-expected telemetry is mandatory.** (Section 9.)

**Invariant F — no future funding rate in a live path.** The published-next rate
is allowed (it is `F_t`-measurable); subsequent stamps are forecast. Test asserts
only `published_next_funding` and history feed the live forecast.

**Invariant N — no Nautilus re-implementation.** (Section 1.) `accrued_fees`
equals the sum of Nautilus `Commission`; `F_accrued` sums Nautilus funding
events; positions/reconciliation are read, not rebuilt.

**Invariant P — sign discipline.** Funding and carry are signed by position side.
Test: a long and a short of equal notional in the same funding/basis regime get
carry terms equal in magnitude, opposite in sign; a short earning funding yields
NEGATIVE carry (a credit) in both the gate and the hold input.

**Invariant Q — conservative fallback, never zero/infinity.** Missing or stale
inputs (`stale_flags` non-empty) inflate cost to a conservative class-level
fallback. Test: dropping bid/ask, funding history, and impact data each yields a
finite, strictly POSITIVE cost and a `fallback_*` reason code, and the net-EV gate
fails closed rather than passing free.

**Invariant R — TCA is an exact identity; plug magnitude is the diagnostic.**
(Section 9.) Eight IS lines (seven named + `unattributed_residual`) sum to
(gross − net) EXACTLY by construction; the TEST is that `|unattributed_residual|`
is below the RED tolerance (a real missing/double-counted line shows up as an
over-tolerance NAMED residual, not a hidden open gap); cost-line sign is positive
when P&L-reducing; attribution sums to 1.0.

**Invariant J-tier — the gate prices fees at the conservative tier near a
breakpoint.** When `rolling_30d_volume` is within `tier_proximity` (RED) of a VIP
breakpoint (or sits exactly on it), the gate prices the round-trip fee at the
HIGHER-FEE tier (P6); `volume_to_next` feeds this proximity test, not just
telemetry. Away from a breakpoint the plain step lookup governs. Test: a trade
whose rolling volume straddles a breakpoint within tolerance is priced at the
worse tier (a strictly higher `C_fee` than the forward lookup would give), and the
verdict is computed against that worse fee; the conservative selection is a gate
forecast device only — TCA still books realized fees at the in-force tier (FACT,
Invariant N). Selection never emits a trade (Invariant D).

**Invariant V — validation never imported by live layers.** Nothing in `facts/`,
`components/`, `assemble/`, or `consume/` imports from `validate/`.

**Invariant X — one carry interface, per-class body.** (Section 5.6.) No consumer
forks on asset class; the branch is inside `CarryModel`, keyed by AMBER metadata.

**Invariant T — one cost truth.** A single `CostBook`/`cost_api` instance serves
all four consumers; the gate, the hold input, the carry term, and TCA's forecast
side all call the SAME estimators with the SAME resolved config. Test: the carry
the gate priced at entry equals `CarryTerm.over_horizon` for the same inputs equals
the forecast `HoldCarryInput` uses — bit-for-bit.

**Invariant W — expected volume and forecast notional are `F_t`-measurable, never
realized future.** `v_hat_exit_window` (impact, 5.4) is a forecast from PAST
volume only; `Notional_expected(τ_j)` (funding, 5.5d) is current notional held
flat to `m_max`, never a TREND reduce path. Test (the same pattern as Invariant
C): replacing all post-`t` volume bars AND any future reduce schedule with garbage
leaves the gate's impact term and forecast-funding term bit-for-bit unchanged. A
live path reading realized future volume is the bug.

**Invariant M-hold — the hold is a RANGE, the gate prices the conservative max.**
The funding stamp count `m` is unmeasurable (it depends on hold length; ref 17).
The gate prices funding at `m_max` (the conservative max-hold quantile, P6), NOT
at a caller point estimate. `expected_m_caller` is diagnostic only and cannot
relax the gate. Test: passing a smaller `expected_m_caller` does not change
`passes`; `funding_sensitivity` (dF/dm) and `funding_m_sensitive` are populated;
the gate's BINDING funding value equals `F_band_conservative(m_max, q)` (the point
`F_forecast(m_max)` is telemetry only — consistent with Invariant F-band-gate,
§8.1/§15, so the test cannot re-open the point-forecast bug the band fix closed).

**Invariant U-dim — impact is a return fraction, not bps (no rogue `/1e4`).**
`C_impact_leg = impact_frac_leg · Notional_leg` with `impact_frac_leg = k_i·σ_i·
sqrt(participation)` a dimensionless fraction; the bps column is `1e4·impact_frac`,
recovered by MULTIPLYING, never by dividing the cost by `1e4`. Test: a known
`k_i·σ_i·sqrt(p)` produces a quote-ccy cost equal to that fraction times notional
(NOT 1e4× smaller); the half-spread term (genuinely bps) still divides by `1e4`.
If a fallback `k_i` is supplied in bps, it is converted to a fraction once at
ingestion and `reason` records `k_units`.

**Invariant G-band — funding band is a correlated worst-case SUM, not m
independent draws.** The confidence band over `m` stamps applies the SAME
residual-quantile shock to every stamp (comonotone) and re-clamps to the cap
(5.5c). Test: the band over `m` stamps exceeds the naive `√m`-independent-sum band
in an autocorrelated funding regime, and never exceeds `m·cap·|Notional|`.

**Invariant Y-finance — financing is charged exactly once.** For a Dukascopy
commodity proxy, financing is either zero (`unlevered`) or booked through the MTF
path (`mtf`), never both, and the commodity carry formula carries no `financing`
term of its own (5.6-commodity). Test: enabling MTF financing changes total carry
by exactly the MTF line, not by an additional commodity-financing term;
convenience yield enters only via the observed basis, else is 0 with
`fallback_no_convenience`.

---

## 12. Validation, baselines, and ablation (the validation layer IS part of the model)

Validation scores **forecast cost/funding vs realized cost/funding**, net, on a
**purged walk-forward** with an embargo around funding stamps and rolls, plus
**sampled combinatorial purged CV** (CPCV) for path-robust CIs. Validation labels
(realized round-trip cost, realized funding accrual, realized implementation
shortfall) use future bars and are structurally unreachable from the live layers
(Invariant V).

**Required baselines** (`validate/baselines.py`), all under identical splits,
marks, and embargo:
- **Flat-bps cost** — a constant round-trip bps with no funding (the naive stub
  COST replaces); shows the value added by the funding/carry model.
- **No-funding** — fees+spread+impact only; isolates the funding contribution to
  decision quality.
- **Last-funding-only** — forecast funding = last realized rate held flat;
  the simplest causal forecast, the bar the EWMA+basis blend must beat.
- **Perfect-foresight bound** — uses realized future funding (FUTURE-ONLY,
  firewalled) as an UPPER bound on how good any causal forecast could be; the gap
  to it bounds achievable improvement. Never a live path.

**Required ablations** (`validate/ablation.py`):
- funding forecast on/off in the net-EV gate and the hold input;
- EWMA vs basis vs published-next, each alone and blended;
- impact model on/off; fee-tier dynamic vs flat;
- drop-top-N robustness: remove the N most-impactful trades/regimes and confirm
  the funding model's edge in decision quality is not a handful of funding spikes.

**Required statistics** (`validate/stats.py`):
- **Block-bootstrap CIs** on forecast-vs-realized error and on net decision
  quality, with block length respecting funding-stamp clustering and vol
  clustering.
- **Multiple-test / deflated-or-probabilistic-Sharpe control** when comparing the
  BLUE grid candidates (forecast half-life, basis λ): report the **Deflated Sharpe
  Ratio** / PSR so a half-life that "wins" by chance over many candidates is not
  promoted. Per-symbol PnL tuning is forbidden; promotion is class-level only.
- Report whether one instrument, one funding regime, or one fold drives the
  result.

**Required scored quantities:**
- Forecast funding bias and RMSE per regime (calm/high-funding/regime-flip);
- net-EV gate hit-rate calibration: of trades the gate admitted, what fraction
  were net-cost-positive ex-post (and vice versa for rejected);
- hold-input calibration: when `carry_exceeds_residual` fired, did continued
  holding actually lose to carry;
- TCA reconciliation residual distribution (Invariant R) across all trades.

**Compute budget on the target VM (2-core / 11 GB ARM) — concrete, capped.**
CPCV × block-bootstrap × per-instrument funding series is the one genuinely
combinatorial cost in this file, so it is budgeted as tightly as the rest of the
spec budgets retail constraints. The nesting is bounded by fixed RED grids, not
by `C(N,k)` left to grow:

- **CPCV groups `N = 6`, test size `k = 2`** → `C(6,2) = 15` train/test splits and
  `C(5,1)·…`-recombined paths bounded to **≤ 15 path evaluations**; do NOT raise
  `N` past 8 on this VM (`C(8,2)=28` is the ceiling).
- **Block bootstrap `B = 1000` resamples, block length `L_funding`** set to the
  funding-autocorrelation decorrelation length (≈ `2·h` stamps, the EWMA ESS
  scale, default ~18 stamps ≈ 6 days), shared with 5.5c — one `L_funding`, not a
  per-call choice.
- **Nesting discipline (the blow-up fix):** the block bootstrap runs **once on the
  pooled out-of-fold residual series**, NOT independently inside every CPCV fold.
  CPCV gives the path-robust point estimates; the bootstrap gives the CI on the
  pooled residuals. `15 paths + 1×1000-resample bootstrap`, not `15×1000`. This
  collapses the combinatorial product to a sum.
- **Per-instrument streaming, not in-RAM panels.** Funding/cost series are scored
  one instrument at a time, O(n) streaming, results reduced to per-regime
  summary stats; the full per-bar panel is never held for all instruments at once.
- **Wall-clock / RAM ceiling:** the full validate suite (all baselines, ablations,
  CPCV, the single pooled bootstrap, DSR/PSR) must complete in **≤ ~15 min
  wall-clock and ≤ ~4 GB resident** on the 2-core / 11 GB VM for the live retail
  universe (≤ ~12 instruments, multi-year 8h funding + 1-min bars). The harness
  asserts the RAM ceiling (peak RSS) and logs wall-clock; exceeding either is a
  CI failure, not a silent slowdown. `numba` only on the accrual/forecast inner
  loops and only after correctness tests pass.

These numbers are RED/BLUE config (`N`, `k`, `B`, `L_funding`, the ceilings), not
hardcoded, so the budget can be raised on better hardware — but the DEFAULTS are
sized to run on commodity hardware.

---

## 13. Synthetic adversarial data (`synth/generate.py`)

No external data is required to run tests. Generate adversarial 1-minute paths
with funding/basis/spread overlays so every component and gate is exercised:

- **Calm funding** — small stable funding; fee/spread dominate the gate.
- **Funding spike beyond the cap** — funding jumps an order of magnitude (crypto
  deleveraging) and the EWMA/basis blend would forecast beyond the venue cap; the
  clamp binds (`r_hat == cap`), the hold input fires `carry_exceeds_residual`, and
  the comonotone band widens (Invariant G-band, 5.5b/c clamp).
- **Funding regime flip** — sign flip (longs paid → longs paying); tests the EWMA
  half-life and basis anchor (Invariant P sign at the flip).
- **Published-next divergence** — venue's published next rate diverges from EWMA;
  the `j==1` stamp uses the published rate EXACTLY (override, not added on top —
  removing the published value must change ONLY the first stamp, not all `m`; 5.5).
- **Long vs short hold / m-sensitivity** — vary `m_max` from 1 to several stamps in
  a non-trivial funding regime; the gate prices at `m_max`, `dF/dm` is reported,
  and a smaller `expected_m_caller` leaves `passes` unchanged (Invariant M-hold).
- **Future-volume / reduce-schedule garbage** — replace post-`t` volume and any
  reduce path with noise; the gate's impact and forecast-funding terms are
  unchanged (Invariant W).
- **Commodity proxy financing** — same NatGas proxy under `unlevered` vs `mtf`;
  financing appears once, convenience falls back to basis-only/0 (Invariant
  Y-finance).
- **Band-binding marginal trade** — a trade engineered (as in 5.7) to pass at the
  POINT funding forecast but fail at the side-selected conservative band; the gate
  returns `passes=False` and `binding_funding == funding_band_conservative`
  (Invariant F-band-gate / P-band).
- **Thin residual band** — a 3-residual funding series; the band falls back to the
  class worst residual with `fallback_band_thin`, not a narrow empirical band
  (Invariant G-est).
- **Fee-tier breakpoint straddle** — `rolling_30d_volume` sitting within
  `tier_proximity` of a VIP breakpoint; the gate prices fees at the worse tier;
  TCA still books realized fees at the in-force tier (Invariant J-tier).
- **Interest-unit double-discount** — the SAME venue numbers fed once as
  `per_interval` and once as `per_day`; `interest_8h` rescales only for `per_day`,
  and the `per_interval` path does NOT divide by 3 (Invariant for 5.5b);
  premium taken from the published index vs the snapshot fallback flagged.
- **Carry time-scaling** — a dated-futures hold at `Δt = τ_spread/2` books exactly
  half the inter-contract carry, `Δt→0` books `→0`; an MTF 7-day hold equals
  `borrowed·mtf_rate·7/365` (simple, Invariant H-time).
- **Spread blowout** — bid/ask widens 10× in a stressed session; SpreadModel
  regime switch; net-EV gate rejects marginal trades.
- **Missing/stale inputs** — drop bid/ask, funding history, impact each in turn;
  conservative fallback fires, gate fails closed (Invariant Q).
- **Roll gap (dated futures)** — a roll inside the hold; CarryModel books the
  roll-gap cost and the contango/backwardation sign (Invariant X, P).
- **MTF financing accrual (equities)** — multi-day leveraged hold; daily financing
  matches LEVERAGE's carry term (Invariant T).
- **Short earning funding** — perp short in a positive-funding regime; carry is a
  CREDIT, the hold input RELAXES, the gate's cost falls (Invariant P).

Behavioral assertions: total cost equals the sum of its lines (B); accrued never
mixes with forecast (A); short/long carry are equal-and-opposite (P); `r_hat`
never exceeds the cap and the published rate is an override not a bias (5.5);
impact is a fraction×notional with no `/1e4` (U-dim); the gate's impact and funding
terms are invariant to future volume / reduce-schedule garbage (W); the gate
prices at `m_max` and reports dF/dm (M-hold); the funding band is the comonotone
sum, exceeding the independent-draw sum under autocorrelation (G-band); the gate
prices funding at the side-selected conservative band, not the point, with a
residual floor + CI edge (F-band-gate / P-band / G-est); fees are priced at the
worse tier near a breakpoint (J-tier); carry lines state their time-scaling /
day-count (H-time); `interest_8h` rescales only for `per_day` inputs (5.5b); proxy
financing is charged once (Y-finance); fallbacks are finite-positive and fail the
gate closed (Q); the carry the gate used equals `CarryTerm` equals the hold
input's carry (T); TCA is an exact identity with the plug magnitude below tolerance
(R); the 5.7 worked example reproduces bit-for-bit (WE).

---

## 14. Asset-class cost/carry presets (cross-asset, class-level only)

Per-class presets (validated, class-level — never per instrument on PnL). These
set fallbacks and carry mechanics; live numbers come from TEAL fits.

- **Crypto perp (Binance USDT-M):** 8h funding (00/08/16 UTC); funding cap per
  venue (`cap`, AMBER) — every `r_hat` clamped to `[−cap,+cap]`; basis anchor =
  interest baseline (scaled per `interest_rate_unit` — `(8/24)` only for `per_day`,
  no rescale for the venue-native `per_interval`, 5.5b) + the venue-published
  premium index (snapshot fallback flagged, 5.5b); published
  next-funding available and used as the `j==1` OVERRIDE (not an additive bias);
  high funding-rate variance → wide comonotone conservative band; maker rebate
  / taker fee by VIP tier (retail = VIP 0). 24/7, no roll, no overnight gap in the
  calendar sense but violent funding regime shifts.
- **Dated futures (Dukascopy-proxied):** carry = near/far basis **time-scaled by
  `Δt/τ_spread`** (5.6-time; the per-year rate is `(F_far−F_near)/F_near ·
  365/τ_spread_days`, the hold carry its pro-rata fraction — NOT the raw spread);
  contango is a long cost; roll-gap booked on the stitched-series discontinuity;
  margin SPAN-style (sizing concern, lives in LEVERAGE — COST only carries the
  financing/basis term).
- **US equities (MTF):** carry = MTF financing interest on the borrowed notional
  (the leverage above 1×), broker rate ~9–18% p.a. (AMBER), **SIMPLE interest at
  Actual/365** (5.6-time), accrued over the hold — directly tightens LEVERAGE's
  stop ceiling. (Compounding is off by default; only an explicit AMBER
  `mtf_compounding=True` switches to the compounded form — the earlier "compounds
  over week-long holds" claim is superseded by the pinned simple convention.)
- **Commodities/metals (Dukascopy proxy):** carry via the OBSERVED basis where
  available (it embeds storage + financing − convenience by no-arbitrage), else
  storage-only with convenience = 0 (P6); financing charged ONCE — zero for an
  `unlevered` proxy, via the MTF path for a financed one (Invariant Y-finance,
  5.6-commodity); geopolitical spread blowouts feed the stressed spread regime.

Fee tiers, funding intervals, financing/storage rates, and roll calendars are all
AMBER metadata flowing through the ONE `CarryModel` interface (Invariant X).

---

## 15. Module acceptance gates (CI)

### Fee model
Pass when: rolling-30d volume → VIP tier is a deterministic step over AMBER
breakpoints; effective maker/taker selected by liquidity flag; `volume_to_next`
telemetry emitted and never triggers a trade (Invariant D).

### Spread model
Pass when: half-spread = 1e4·(ask−bid)/(2·mid); regime-bucketed; missing bid/ask →
conservative class fallback, never zero (Invariant Q).

### Impact model
Pass when: square-root impact with TEAL-fitted `k_i`; cost is a return FRACTION
times notional with NO rogue `/1e4` (Invariant U-dim); `participation` uses the
CAUSAL `v_hat` forecast, never realized future volume (Invariant W); conservative
linear fallback when unfit; realized impact ingested from EXECUTION for TCA (not
re-modelled).

### Funding accrual (FACT)
Pass when: `F_accrued` sums Nautilus funding events over actual stamps in
[entry,now] with notional AT each stamp; equals an independent recomputation from
Nautilus state (Invariant N); sign correct for long/short (Invariant P);
`is_fact=True`.

### Funding forecast (ESTIMATE)
Pass when: causal (Invariant C, F); per-stamp PIECEWISE form — published-next is a
`j==1` OVERRIDE, not an additive bias (no whole-curve inflation, 5.5); EWMA closed
form `α=1−2^(−1/h)` with ESS and the `n_funding_min` event-count floor falling back
to published/last-funding (5.5a); basis anchor = interest baseline + premium,
with the `(8/24)` interest scaling applied ONLY when `interest_rate_unit ==
per_day` and SKIPPED for `per_interval` (no ~3× double-discount; 5.5b), premium
taken from the venue-published premium index with a flagged snapshot fallback
(5.5b), clamped to `[−cap,+cap]`; every `r_hat` clamped to the cap; gate priced at
conservative `m_max` with `dF/dm` reported (Invariant M-hold); comonotone
worst-case band with an `n_residual_min` floor + class fallback and a CI-edge
binding value (Invariant G-band/G-est), side-selected (Invariant P-band), and the
GATE consumes that band, NOT the point `F_forecast` (Invariant F-band-gate); beats
last-funding-only baseline on forecast RMSE class-wide; `is_fact=False`; never
reads a future rate.

### Carry model (cross-asset)
Pass when: one interface, per-class body keyed by AMBER (Invariant X); signs
correct per class (Invariant P); roll-gap booked for dated futures; MTF financing
equals LEVERAGE's carry term (Invariant T); commodity proxy charges financing ONCE
and convenience falls back to basis-only/0 (Invariant Y-finance); fallback
finite-positive (Invariant Q).

### Cost book / assembly
Pass when: total = sum of components exactly (Invariant B); accrued and forecast
never merged (Invariant A); dual-unit (quote ccy + bps); deterministic given
inputs.

### Net-EV gate
Pass when: `passes` = expected_favorable > total_round_trip(funding priced at the
side-selected CONSERVATIVE BAND `F_band_conservative(m_max, q)` — NOT the point
`F_forecast`; Invariant F-band-gate / P-band / G-est) + edge_buffer;
`binding_funding == funding_band_conservative` and a marginal trade that passes at
the point estimate but fails at the band is REJECTED (Invariant F-band-gate);
funding priced at conservative `m_max`, `funding_sensitivity` (dF/dm) and
`funding_m_sensitive` populated, and a smaller `expected_m_caller` does NOT relax
the verdict (Invariant M-hold); fee tier priced at the conservative (worse) tier
when within RED proximity of a VIP breakpoint (Invariant J-tier); impact term
causal (Invariant W); additive both sides (no product, Invariant B); fails closed
under stale inputs (Invariant Q); emits no action (Invariant D); full
decomposition + reason codes returned.

### Hold-carry input
Pass when: returns forecast_residual_carry vs expected_residual_favorable and the
boolean; signed so a funding-earning short relaxes the hold (Invariant P); uses
the SAME forecast as the gate (Invariant T); emits no action (Invariant D).

### Carry term (LEVERAGE)
Pass when: returns the signed forecast carry over the horizon equal bit-for-bit to
what the gate/hold input used SIGNED (Invariant T); the survival stop-ceiling consumes it
CLAMPED, `L_stop=(c−max(carry,0))/d` (a credit does not lift the hard cap — its EV is in the gate, not the survival cap).

### TCA
Pass when: the eight IS lines (seven named + `unattributed_residual`) sum to
(gross−net) EXACTLY by construction, and `|unattributed_residual|` is below the RED
tolerance (the diagnostic — a missing/double-counted line surfaces as an
over-tolerance NAMED residual, Invariant R); cost-line sign positive when
P&L-reducing; attribution sums to 1.0; all named lines FACT; realized-vs-expected
emitted per line (Invariant E); `funding_cost`==`FundingAccrual`.

### Validation harness
Pass when: purged walk-forward + sampled CPCV (`N=6, k=2`, ≤15 paths) run; the
block bootstrap (`B=1000`, block `L_funding`) runs ONCE on pooled out-of-fold
residuals, NOT nested per fold; all four baselines produce the same report shape;
ablations run; CIs + DSR/PSR multiple-test control computed; drop-top-N robustness
reported; the suite completes within the **≤15 min wall-clock / ≤4 GB resident**
ceiling on the 2-core/11 GB VM (peak RSS asserted in CI); no live layer imports
validate/ (Invariant V).

---

## 16. Build order

1. `types.py` and `synth/generate.py` (funding/basis/spread/roll overlays).
2. `config/` TEAL/AMBER/RED/BLUE; test Invariant C-cfg (RED unreachable).
3. `facts/nautilus_adapter.py` + `facts/execution_adapter.py`; test Invariant N
   (accrued fees == Σ Nautilus Commission).
4. `components/fees.py` (rolling-volume → tier → effective fee, Section 7).
5. `components/spread.py` (half-spread + regime + fallback, Invariant Q).
6. `components/impact.py` (square-root FRACTION, no `/1e4`, TEAL `k_i`, causal
   `v_hat`, EXECUTION realized ingest; Invariant U-dim/W).
7. `components/funding_accrual.py` (FACT sum, Invariant A/N/P).
8. `components/funding_forecast.py` (causal piecewise `r_hat`: EWMA closed-form
   + event-floor (5.5a), basis anchor + interest with `interest_rate_unit`-gated
   8h scaling + published premium-index/snapshot-fallback + clamp (5.5b), published
   `j==1` override, comonotone band with `n_residual_min` floor + CI edge +
   side-selection (5.5c), `m_max`/dF/dm (5.5d); Invariant
   C/F/G-band/G-est/P-band/M-hold).
9. `components/carry.py` (ONE interface, per-class body; commodity financing-once
   + basis-only convenience; Invariant X/P/Y-finance).
10. `assemble/cost_book.py` (additive RoundTripCost; Invariant B/A; cost_fraction).
11. `consume/net_ev_gate.py` (Section 8.1; price funding at `m_max` AND at the
    side-selected conservative band, not the point; conservative fee tier near a
    breakpoint; Invariant D/B/Q/W/M-hold/F-band-gate/P-band/J-tier).
12. `consume/hold_carry.py` (Section 8.2; Invariant D/P/T).
13. `consume/carry_term.py` (Section 8.3; Invariant T) and `cost_api.py`.
14. `consume/tca.py` (Section 9; Invariant R/E).
15. `validate/labels.py`, `validate/walkforward.py`, `validate/cpcv.py`.
16. `validate/baselines.py` (incl. perfect-foresight bound), `validate/ablation.py`.
17. `validate/stats.py` (block-bootstrap, DSR/PSR), `validate/reports.py`.

Each step complete only when its acceptance gate (Section 15) is green. A module
that fails its gate does not advance. Stop after step 17: whether the EWMA+basis
blend actually beats last-funding-only on your symbols, which cost line dominates
the gate, and whether the funding model changes any real decision are answered by
RUNNING the validation harness on real Binance/Dukascopy data, not by adding
estimators.

---

## 17. References and rationale

- Implementation shortfall: Perold (1988), "The implementation shortfall: paper
  versus reality"; the decision→arrival→fill→opportunity decomposition TCA uses.
- Transaction-cost analysis & impact: Almgren & Chriss (optimal execution);
  Almgren et al. (square-root impact law); Kyle (1985) lambda — the auditable
  monotone impact form, not a learned one. COST forecasts expected impact;
  EXECUTION owns realized microstructure.
- Perp funding mechanics: perpetual swap funding as the no-arbitrage tether of
  perp mark to index; Binance's funding construction = interest-rate baseline +
  clamped premium index (the additive `interest_8h` + clamped premium of 5.5b),
  8h schedule, the per-interval CAP (every `r_hat` clamped to it, 5.5b/c), and the
  pre-published next-funding rate as a `j==1` override (the one causal future-stamp
  input permitted) — NOT an additive bias on every stamp (5.5).
- The drift power wall and the funding stamp count: hold length is unmeasurable
  intraday (ref `t = SR·√T` below), so the number of 8h stamps `m` paid is unknown;
  COST prices funding to a conservative `m_max` and reports dF/dm rather than
  trusting a point hold (5.5d, Invariant M-hold).
- Cost of carry / basis: Hull, cost-of-carry model F = S·e^{(r+u−y)T} (financing +
  storage − convenience yield); the dated-futures and commodity carry terms.
- Roll cost / contango: futures roll yield and the stitched-series gap; carry sign
  by long/short roll (Invariant P).
- EWMA / RiskMetrics: exponentially-weighted estimation for the funding-level
  forecast; transparent, causal, monotone (no ML, P4).
- Block bootstrap (Künsch; Politis & Romano stationary bootstrap): CIs that
  respect funding-stamp and volatility clustering.
- Deflated / Probabilistic Sharpe Ratio (Bailey & López de Prado): multiple-test
  control when choosing the BLUE forecast-grid candidates; guards against a
  half-life that wins by chance.
- Purged / combinatorial purged CV (López de Prado, *Advances in Financial
  Machine Learning*, embargo around funding stamps/rolls): the walk-forward and
  CPCV used here, leakage-free.
- The drift power wall t = SR·√T (BREAKOUT.md/TREND.md/LEVERAGE.md): edge is
  unmeasurable intraday, so COST never estimates edge — it prices COST and lets
  the edge-owning pillars supply `expected_favorable`. The firewall runs the other
  way too: COST supplies cost, never edge.

---

## 18. Non-goals (belong to sibling modules / Nautilus)

- **Trade decisions** — COST is a provider, not a decider (P2). Entry is
  BREAKOUT.md; hold/reduce/exit is TREND.md; sizing is LEVERAGE.md; eligibility is
  the Carver-Ch25 selection module. COST hands them numbers; they decide.
- **Order execution / microstructure** — order routing, child-order scheduling,
  and the realized fill mechanics are EXECUTION.md. COST ingests EXECUTION's
  realized slippage/impact and forecasts the *expected* impact for the gate; it
  does not route orders.
- **Raw commission accounting, funding ingestion, position state, reconciliation,
  backtest↔live transport** — all owned by NautilusTrader (Section 1, Invariant
  N). COST builds the cost/funding POLICY on top, never re-implements the plumbing.
- **Edge / alpha estimation** — `expected_favorable` and `expected_residual_move`
  are supplied by the edge-owning pillars. COST estimates COST, never edge.
