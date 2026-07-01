# OPS.md — build spec for the SPC monitoring / kill-switch / reconciliation layer

This file is the authoritative build spec. Build to the contracts and invariants
below. Where this file and your own instinct disagree, this file wins. Ask
before deviating from any **Invariant**.

**This is a fresh, cross-sectional-specific rewrite.** The prior `OPS.md` was
gutted along with the whole directional-breakout stack (`BREAKOUT.md`,
`TREND.md`, `LEVERAGE.md`, `EXECUTION.md`, `REGIME.md`) when directional
trend/breakout was proven dead — **gross directional R ≈ 0** across every
timeframe, exit, filter, leverage tier, three asset classes, and even inverted
(`docs/EDGE.md` §a; cited here as background only, never as the subject of this
spec). The live edge is **crypto cross-sectional momentum**
(`docs/EDGE.md` §b, `research/xsec/`), and OPS is rebuilt around the risk unit
that strategy actually produces: **next week's portfolio return**, not one
trade's R-multiple. There is no BREAKOUT/TREND/LEVERAGE/EXECUTION to monitor
fills or stops for; there is a weekly-rebalanced, dollar-neutral book whose
health is a *portfolio-level statistical process*, checked far more often than
it trades.

OPS is a **control and monitoring specification built on top of the
statistical machinery SYSTEM and REGIME already define** — it invents no new
probability theory. To be precise about what is and is not shared: `SYSTEM.md`
§4–§5 does NOT itself use CUSUM (its validation battery is CPCV/PBO and
deflated/probabilistic Sharpe, a static offline promotion gate, not a
sequential detector) — OPS's CUSUM engine is a **live, sequential analogue**
of that same validated Sharpe distribution, monitoring, week over week,
whether the REALIZED trailing Sharpe is still consistent with the
distribution `SYSTEM.md` §4–§5 established offline (§4.2 below draws `mu_0`,
`sigma_0` directly from that validated distribution). The CUSUM/ARL
CONSTRUCTION itself (Page 1954 sequential change-detection, Siegmund 1985 ARL
calibration) is, however, genuinely shared machinery: it is the same class of
tool `REGIME.md` §3–§4 independently instantiates for its own market-state
change-detection (REGIME's detector monitors *market* statistics — BTC
realized vol, cross-sectional dispersion — under a disjoint trigger axis from
OPS's *strategy-health* CUSUM; see §4.4's composition note and the
notation-reconciliation remark below §4.3). This is deliberate: one CUSUM/ARL
construction, independently calibrated per monitored stream against its own
null, never a bespoke ad hoc threshold per problem, and never a claim that
SYSTEM's own validation battery uses CUSUM (it does not).

Scope: crypto-only (Binance USDT-M perpetual futures), ~50-name point-in-time
universe (`docs/EDGE.md` §b; the exact breadth/shortability trade-off — net
edge peaking near 60–75 names, and why 50 is the defensible simple choice
once ~15 of the marginal 50 added names turn out effectively unshortable —
is `UNIVERSE.md`'s derivation to own, not re-litigated here; OPS treats
"the live universe" as an AMBER input, whatever size UNIVERSE.md resolves it
to), **weekly Friday-close rebalance** cadence, retail book size, target
compute a 2-core / 11 GB ARM VM. OPS runs on a strictly finer clock than the
strategy trades on (§2), because a control loop that only looks as often as
it rebalances cannot catch a mid-week blowup.

> **A note on `docs/ARCHITECTURE.md`.** That document is currently the
> **pre-pivot, directional-breakout-era** composition root (it opens "a
> multi-pair directional convex-breakout trader," and its OPS-related
> references — an "OPS DD-kill (45%)," `Inv BRAKE`/`Inv KILL`/`Inv ALARM`,
> a "§10 CUSUM cost-drift" fixture with `DEGRADE@15`/`HALT@18` — describe the
> **retired** single-instrument governor, not this spec. None of those
> thresholds, invariant names, or section numbers exist in this document: the
> invariants here are `RECON`/`CADENCE`/`PROV`/`CAL`/`MONO`/`DNEUT`/`SLA`/
> `HALT-ONE`/`C`, and the calibrated boundary is `h* = 6.23` with `DEGRADE` at
> `h*/2` (§4.4). A reader who opens `ARCHITECTURE.md` first and looks for
> those symbols here will not find them. `ARCHITECTURE.md` needs its own
> cross-sectional-pivot rewrite (out of scope for this file, which is
> spec-writing only per its own remit) — until that happens, THIS file, plus
> `SYSTEM.md`, `COST.md`, `REGIME.md`, `PORTFOLIO.md`, `LEVERAGE.md`,
> `EXECUTION.md`, `SIGNAL.md`, and `UNIVERSE.md`, are the authoritative specs,
> and `ARCHITECTURE.md`'s composition map should be treated as stale until
> reconciled.

> **A second, narrower staleness note: `core/types.py`'s `OpsDirective` is
> ALSO pre-pivot and collides with this file's.** Separately from
> `ARCHITECTURE.md`'s document-level staleness above, `src/convexity/
> core/types.py` contains a real, importable `OpsDirective` dataclass
> (`state: KillState`, `flatten`, `cancel_entries`, `reason_codes`,
> `engine_version`) that `src/convexity/core/protocols.py`'s `OpsEngine`
> Protocol currently type-checks against — and it is the SAME retired,
> single-instrument-governor shape `ARCHITECTURE.md` describes, not this
> spec's `OpsDirective` (§9). `EXECUTION.md` §7.2 already names
> `core/protocols.py`'s `OpsEngine.on_obs` as the one real seam it feeds its
> realized `IS_total` through; a builder following both specs verbatim would
> produce two same-named, incompatible `OpsDirective` classes. §9 below
> states the full reconciliation this spec commits to (`core/types.py`'s
> `OpsDirective` is stale and superseded by §9's version) — flagged here too
> because it is exactly the kind of pre-pivot residue `ARCHITECTURE.md`'s
> note above warns about, just scoped to one dataclass instead of the whole
> composition map.

---

## 0. Governing principles (read before anything else)

**P1 — The risk unit is the WEEKLY BOOK RETURN, not a trade R-multiple.** Every
statistic OPS monitors is built from the same object the strategy actually
produces: the realized net-of-cost return of the whole dollar-neutral book
over one rebalance interval (`research/xsec/accounting.py::WeekResult.net`).
There are no per-trade stops to watch, no single-position R-multiples, no
trade-level triple-barrier labels — a rotation out of one name into another
inside an unchanged dollar-neutral book is a *reallocation*, not a "trade"
with its own P&L unit. Any OPS statistic keyed to a single instrument's P&L
instead of the book's weekly net return is the bug. (Former `COST.md`/
`SYSTEM.md` framing keyed everything to a single position's stop-to-target
R-multiple and a per-bar decision; that framing is retired for OPS.)

**P2 — Monitor faster than you trade.** The rebalance clock is weekly; the
control clock is NOT. Position reconciliation runs **daily** (§1), the
kill-switch CUSUM evaluates on every new weekly observation but its
*reconciliation and heartbeat inputs* are checked daily/continuously (§1, §5),
and rebalance-completion SLA monitoring is checked **intraday**, inside the
execution window itself (§5). A control system that only samples at the
frequency it acts cannot distinguish "the strategy is fine" from "the last
data point I saw was fine, five days ago, before the exchange delisted half
the book." (See Invariant CADENCE.)

**P3 — OPS is a PROVIDER of a binary/graded directive, never a decider of
economics.** Mirroring `COST.md` P2 (provider, never decider) and `SYSTEM.md`
P1 (compose, never reimplement): OPS never picks weights, never re-signs a
position, never decides *which* names to hold. It emits one of a small,
totally ordered set of `OpsDirective` states — `RUN`, `DEGRADE`, `HALT` — plus
the specific alarm(s) that produced it. The live loop (or, pre-paper, the
operator) is the only thing that acts on a `HALT` by flattening the book
through the venue. OPS computing a target weight, or vetoing one instrument
while leaving the rest of the book untouched, is out of scope — its unit of
action is the *whole book* or nothing, because dollar-neutrality (P6) is a
book-level invariant that a partial intervention would break. (See Invariant
PROV.)

**P4 — Every live threshold is a calibrated Type-I-error budget, not a
hand-picked number.** The kill-switch is not "halt if Sharpe drops below
X" for an arbitrarily chosen `X`. It is a **CUSUM sequential change-detection
rule** (Page 1954) whose decision boundary `h` is calibrated so that a
**healthy** strategy — one whose true weekly Sharpe and variance match the
validated backtest distribution (`docs/EDGE.md` §b: honest net Sharpe
**~0.8–1.0**; `docs/specs/SYSTEM.md` §4.2: CPCV mean OOS path Sharpe
**≈0.91** vs naive single-path **≈0.98**, haircut only **≈0.07**;
`SYSTEM.md` §4.3: `frac_pos` **0.85–1.00**) — triggers a false HALT only with
a stated, low, explicit annual probability (§4.3). Every other numeric
threshold in this spec (reconciliation tolerance §1.2, SLA budget §5, CUSUM
boundary `h*` §4.3) is likewise tied either to a derivation (§1, §4, §5) or to
a specific EVIDENCE number, cited by name — never invented. (There is no
"DD-kill level" in this spec: the old directional-era `docs/ARCHITECTURE.md`
reference to an "OPS DD-kill (45%)" belonged to the retired single-position
governor and has no cross-sectional analogue here — see §14 Non-goals and the
`ARCHITECTURE.md` staleness note below. OPS's only drawdown-adjacent
mechanism is the CUSUM `DEGRADE`/`HALT` ladder of §4.4, which is a Sharpe-drift
detector, not a raw-equity drawdown trigger; a raw book-equity drawdown stop
is REGIME's mechanism, not OPS's — see §4.4's cross-reference to `REGIME.md`
§5.) (See Invariant CAL.)

**P5 — Accrued is FACT, forecast is never consulted here.** OPS reads only
realized, already-settled quantities: booked positions, exchange positions,
realized weekly net returns, realized fills, realized completion timestamps.
It contains no forecast, no EWMA-projected funding, no expected-slippage
model — those live in COST and are irrelevant to a control loop whose entire
job is to react to what has *already happened*. (Mirrors `COST.md` P5's
FACT/ESTIMATE separation, specialized: OPS is 100% FACT, 0% ESTIMATE.)

**P6 — Dollar-neutrality is a live invariant, not a design aspiration.** The
book satisfies `Σ w_long = −Σ w_short` at every rebalance by construction
(`research/xsec/portfolio.py::quantile_weights`). Between rebalances, an
**exogenous stopping time** — a delisting, a trading halt, a forced perp
settlement — can force an unplanned exit that breaks this identity strictly
between the planned weekly rebalances. OPS is the layer that detects the
break (via reconciliation, §1) and enforces the **renormalization** that
restores it (§3) without waiting for the next scheduled rebalance. A forced
exit that is closed and never renormalized (a "silent" one-legged book until
Friday) is a bug this spec exists to prevent.

**P7 — Escalation is monotone and one-directional per episode.** Once OPS
raises its state from `RUN` to `DEGRADE` or `HALT`, it does not silently
self-heal on the next single good observation — recovery requires an explicit
operator action or a stated cool-down + confirmation rule (§4.4), because a
CUSUM statistic that resets to zero on one good week and immediately
re-accumulates is not a meaningfully different regime detector from no
detector at all. (See Invariant MONO.)

**P8 — No ML, closed-form only.** Consistent with the whole investigation's
constraint and `pyproject.toml`'s "scipy: closed-form (NOT ML)" doctrine: every
detector here is CUSUM (Page 1954), Shewhart, or a simple order-statistic /
tolerance check. No fitted classifier decides HALT vs RUN. No hyperparameter
grid search tunes the kill-switch on live P&L. The ARL calibration (§4.3) is a
closed-form / Monte-Carlo-under-a-fixed-null calculation done ONCE offline, not
a continuously re-fit model.

Forbidden in v1: any per-trade / per-instrument P&L unit as an OPS monitored
series (P1); OPS emitting a partial-book or per-instrument directive (P3); an
un-calibrated threshold not tied to a derivation or an EVIDENCE number (P4);
consuming any forecast/expected quantity (COST's estimates) inside a kill-switch
statistic (P5); a delisting handled by closing the position and leaving the
opposite leg's weight un-renormalized until the next scheduled rebalance (P6);
auto-recovery from HALT/DEGRADE on a single good observation with no cool-down
(P7); any learned/fitted anomaly detector, threshold auto-tuner, or ML
classifier anywhere in the module (P8).

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its
first point of use in prose; this table is a lookup, not a substitute for
those definitions), following the same convention as `LEVERAGE.md` §0.1 and
`SYSTEM.md` §0.1. Week index `t`/`k` runs over Friday-close rebalances (the
same clock `LEVERAGE.md`/`SYSTEM.md` use, `research/xsec/engine.py`); `d`
without a subscript below is OPS's CUSUM reference-value symbol (§4.3), not
a day-count.

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `delta_i,t` | Book-vs-exchange position divergence for instrument `i` at time `t` | data | 1.1 |
| `epsilon_i` | Per-instrument reconciliation tolerance | derived (RED-scaled) | 1.2 |
| `k_lot`, `k_notional` | Lot-rounding and notional-proportional tolerance scale factors | RED | 1.2 |
| `m_warn` | Multiple of `epsilon_i` marking WARN vs BREACH | RED | 1.4 |
| `staleness_max` | Max days a failed reconciliation read stays WARN before escalating | RED | 1.4 |
| `r_k`, `weekly_net_return` | Realized book return, week `k` (`WeekResult.net`) | data | 2, 4.2 |
| `mu_0`, `sigma_0` | Validated-null per-week mean/std-dev the CUSUM standardizes against | RED (EVIDENCE-anchored) | 4.2 |
| `SR_0` | Lower-bound honest net Sharpe anchoring `mu_0` (`= 0.85`) | EVIDENCE | 4.2 |
| `z_k` | Standardized weekly innovation, `(r_k − mu_0)/sigma_0` | derived | 4.2 |
| `C_k`, `C_k^+` | One- and two-sided CUSUM statistics (downward-drift / anomalous-outperformance) | derived | 4.3 |
| `d` | CUSUM reference/allowance value (`≡ k` in REGIME.md's/classical SPC notation, renamed to avoid a clash with the `CUSUM` Python class) | RED | 4.3 |
| `h`, `h*` | CUSUM decision boundary; `h* = 6.23` is the calibrated value shipped | RED (derived) | 4.3 |
| `ARL_0`, `ARL_1` | Average run length under the null / under a real shift | derived | 4.3 |
| `delta_target` | Standardized Sharpe shift size the chart is tuned to detect (`= 1.0`) | RED | 4.3 |
| `n_confirm` | Consecutive healthy weeks required to exit DEGRADE/HALT back to RUN | RED | 4.4 |
| `g_degrade` | Gross multiplier applied while in DEGRADE (`e.g. 0.5`) | RED | 4.4 |
| `PBO` | Probability of backtest overfitting (`= 0.357`, `SYSTEM.md` §4.4) | SYSTEM-owned | 4.3 |
| `frac_pos` | Fraction of CPCV paths with positive OOS Sharpe (`0.85–1.00`) | SYSTEM-owned | 4.1, 4.3 |
| `PSR`, `MinTRL`, `n_obs` | Probabilistic Sharpe, min track-record length, observed weeks (`SYSTEM.md` §5.3–5.5) | SYSTEM-owned | 4.3 |
| `W_exec` | Execution-window budget (`≡ EXECUTION.md`'s `Delta`, AMBER, ~1–4h) | AMBER (EXECUTION-owned) | 5.1 |
| `m_slip` | Slippage-SLA tolerance multiple on modeled friction (`e.g. 2×`) | RED | 5.2 |
| `IS_total` | Realized implementation shortfall per rebalance (`EXECUTION.md` §7.1) | EXECUTION-owned | 5.2 |
| `fee`, `half_spread_i`, `impact_i` | COST's round-trip friction decomposition terms (§5.2's `m_slip` derivation) | COST-owned (`fee` exact; others calibrated) | 5.2 |
| `sigma_0_reestimation_weeks` | Cadence at which `sigma_0` is re-derived from an enlarged sample (`= 52`) | RED | 7, 13 |

---

## 1. Position reconciliation — a control-theoretic invariant check

### 1.1 The invariant

At any instant `t`, the book OPS believes it holds and the position the
exchange actually reports must agree, per instrument, up to a stated
tolerance:

```
delta_i,t = book_position_i,t − exchange_position_i,t
INVARIANT:  |delta_i,t| <= epsilon_i     for every instrument i in the live book
```

`book_position_i,t` is Nautilus's internal position state (the quantity the
strategy's own bookkeeping believes it holds, driven by its own fills);
`exchange_position_i,t` is the ground truth read from Binance's position
endpoint. These are two independently-maintained records of the same physical
fact, and normal operation asserts they never diverge beyond noise. A
non-degenerate `delta_i,t` is evidence of a *specific, enumerable* failure
class: a missed fill acknowledgment, a partial-fill under/over-count, a
liquidation event Nautilus's own risk engine did not originate, a stale
websocket/position-stream reconnect gap, or (worse) a second, unauthorized
process trading the same account.

### 1.2 The tolerance `epsilon_i` (derivation, not a magic number)

A fixed cross-instrument `epsilon` is wrong by construction: a $50,000 BTC
notional position and a $500 low-cap altcoin position round to a lot size at
utterly different absolute-quantity scales, so a single global tolerance is
either too loose for majors (masking a real drift) or triggers permanently on
alt dust (crying wolf until the operator ignores every alert — the classic SPC
failure mode of an uncalibrated control limit). `epsilon_i` is therefore
**scaled to the instrument's typical lot size / notional**, not asserted flat:

```
epsilon_i = max( k_lot · lot_size_i ,  k_notional · target_notional_i / price_i )
```

- `lot_size_i` — the exchange's minimum tradable increment for instrument `i`
  (AMBER metadata, Binance `stepSize`). `k_lot` (RED, e.g. 2) tolerates a
  couple of lots of rounding slop from the Quantizer/lot-rounding step every
  position sizer performs (mirrors `LEVERAGE.md`'s round-down-only Quantizer
  discipline: the position OPS books is a lot-quantized integer, so a
  sub-lot mismatch against a fractional exchange report is expected, not an
  anomaly).
- `k_notional · target_notional_i / price_i` — a tolerance proportional to
  the position's *intended* dollar size (RED `k_notional`, e.g. 0.1%): a name
  sized at $2,000 notional tolerates a few dollars of drift; a name sized at
  $20,000 tolerates proportionally more, because fill-price rounding and
  funding-driven notional drift both scale with position size, not with a
  fixed dollar amount.
- The `max()` of the two floors the tolerance at whichever failure mode binds:
  a tiny position is bounded by the lot-rounding floor even if 0.1% of its
  notional would be sub-lot; a large position is bounded by the proportional
  term even though a couple of lots would be a vanishingly small fraction of
  it.

This is the same tolerance-scaling discipline `COST.md`'s fee-tier and
half-spread fallbacks use (never a bare global constant when the natural unit
is heterogeneous across instruments) — restated here for the reconciliation
check because it is the first place a hand-picked flat epsilon would silently
mis-calibrate across a ~50-name universe spanning majors to micro-caps.

### 1.3 Check cadence — more frequent than the weekly rebalance (P2)

```
reconciliation_cadence = 1 trading day    (RED; strictly < rebalance_days = 7)
```

Reconciliation runs **daily**, not weekly and not only-at-rebalance. A
divergence that opens on Tuesday and is only discovered at Friday's rebalance
has had four days to compound (funding accrual on a phantom position, a
liquidation the strategy doesn't know happened, margin consumed by a leg that
technically isn't there). Daily cadence is the minimum consistent with P2
("monitor faster than you trade") for a weekly-rebalanced book: any cadence
`>= rebalance_days` reduces to "check only at rebalance," which is exactly the
failure mode P2 forbids. A tighter-than-daily cadence (e.g. hourly) is
permitted and cheap (the check is O(n_universe), not a backtest), but daily is
the specified floor.

### 1.4 Escalation policy

```
                     |delta_i,t| <= epsilon_i               -> OK, no action
      epsilon_i  <   |delta_i,t| <= epsilon_i · m_warn       -> WARN  (log + notify, no book action)
      epsilon_i · m_warn < |delta_i,t|                       -> BREACH -> escalate to OPS kill-switch (§4)
```

- `m_warn` (RED, e.g. 3) is a multiple of the base tolerance: a small
  persistent drift a few multiples above `epsilon_i` (e.g. a rounding
  convention mismatch between Nautilus's internal accounting and the venue's
  reported quantity after a funding-driven mark change) is logged and
  surfaced but does not by itself halt trading — it is common enough
  (exchange rounding, in-flight fills at the instant of the snapshot) that
  auto-halting on it would be a nuisance-alarm generator.
- A **BREACH** (`|delta_i,t| > epsilon_i · m_warn`) on ANY instrument is
  escalated directly into the kill-switch machinery (§4) as a `RECON_BREACH`
  alarm — it does not wait for the CUSUM statistic to accumulate evidence,
  because a large position-state divergence is not a *statistical* signal to
  be filtered for noise; it is a direct, already-certain indicator that the
  book's ground truth is wrong RIGHT NOW. `RECON_BREACH` is therefore wired as
  an immediate `HALT` trigger (a Shewhart-style single-point rule, not a
  CUSUM), exactly analogous to how a Shewhart chart's single-point 3σ rule
  fires immediately alongside a CUSUM's cumulative-drift rule in a combined
  SPC scheme (Page 1954 introduced CUSUM explicitly as a complement to, not a
  replacement for, single-point control limits).
- A snapshot read that itself fails (exchange API error, stale websocket) is
  **not** silently treated as `OK` — a failed reconciliation read is itself a
  `RECON_STALE` WARN-tier event, escalating to BREACH if reconciliation has
  been unreadable for more than `staleness_max` days (RED, e.g. 2 days,
  strictly less than one rebalance interval), because "I could not check" must
  never be indistinguishable from "I checked and it's fine" (P6 in spirit:
  never silently assume the invariant holds; the fallback is conservative,
  mirroring `COST.md` P6's "missing input never returns a permissive
  default").

> **Invariant RECON — every live instrument is reconciled at least daily, and
> a breach escalates without waiting on a statistical filter.** Test: a
> synthetic scenario that opens a `1.5 · epsilon_i · m_warn` divergence on one
> instrument produces a `RECON_BREACH` alarm and an `OpsDirective(HALT)` within
> one reconciliation cycle (never later than the next scheduled daily check),
> regardless of what the CUSUM Sharpe-monitor state is at that moment.

---

## 2. Monitored streams (the inputs to every detector below)

Every detector in §4–§5 consumes one of a small, fixed set of realized,
already-settled series (P5). This section is the inventory; it exists so no
detector invents its own private notion of "the book's return."

| Stream | Source | Cadence | Consumed by |
|---|---|---|---|
| `weekly_net_return` | `research/xsec/accounting.py::WeekResult.net` (realized, post-settlement) | weekly (Friday close) | §4 CUSUM kill-switch |
| `position_delta` | book vs exchange position snapshot (§1) | daily | §1 reconciliation |
| `rebalance_completion_time` | realized wall-clock time the weekly rebalance's last fill settles | weekly, checked intraday during the execution window | §5 SLA monitor |
| `realized_slippage` | EXECUTION's realized implementation-shortfall per rebalance (once EXECUTION exists; until then, COST's realized-fill TCA) | weekly | §5 SLA monitor |
| `heartbeat` | liveness ping from the live loop / data feed | continuous (sub-minute) | forced-`HALT` on silence (P6, exogenous-event detection trigger) |

`weekly_net_return` is the ONE series the kill-switch's Sharpe-CUSUM
statistic is built from (P1) — never a per-instrument return, never an
intra-week mark-to-market path (which is not `F_t`-realized-and-settled in the
same sense; it is still subject to the week's own turnover/funding
accounting closing out at `settle_week`).

---

## 3. Delisting / listing mid-week event handling

### 3.1 The exogenous stopping time

Let `tau_delist` be the (random, exogenous) instant an instrument `i` held in
the current book is delisted, permanently halted, or otherwise forced out of
tradability by the exchange. `tau_delist` is **not** a rebalance date in
general:

```
tau_delist  ∈  (t_k, t_{k+1})      strictly between rebalance k and k+1
```

where `t_k` are the scheduled Friday-close rebalance instants. This is a
genuine edge case (not hand-waved): the strategy's book is constructed at
`t_k` under a point-in-time `listed` mask (`research/xsec/panel.py`,
`accounting.py::_asset_week_return` already returns `NaN` for a name not
`listed` at both week endpoints when computing the *backtest* P&L), but a
**live** delisting announcement can land at any wall-clock moment inside the
holding week, and unlike the backtest (which simply excludes the dead week's
return from that name post-hoc), a **live** position must be actively closed
before the venue forces the closure on worse terms.

### 3.2 Forced-liquidation rule

```
ON delisting/halt notice for held instrument i at time tau_delist:
  1. IMMEDIATELY submit a closing order for the full held quantity q_i at
     the last tradeable price (market/IOC at the venue, not a passive limit —
     urgency dominates price improvement once tradability itself is ending).
  2. Record the realized close price p_close,i and timestamp tau_delist.
  3. Mark w_i -> 0 in the LIVE book state immediately (do not wait for the
     next scheduled rebalance to zero it).
  4. Proceed to §3.3 renormalization BEFORE the next scheduled rebalance.
```

"Close at the last tradeable price, immediately" is a strict-urgency policy —
identical in spirit to the forced-liquidation branch a governor/margin-call
handler would take in a single-position risk system, specialized here to a
delisting notice as the trigger. There is no reduce-gradually option: once a
venue announces delisting, liquidity for the name only degrades from that
point (market-makers withdraw ahead of the deadline), so waiting strictly
increases realized slippage in expectation.

### 3.3 Weight renormalization — the invariant-preservation argument

**Claim.** After the forced exit of instrument `i` from a dollar-neutral
book, renormalizing the *surviving* legs by their pre-exit gross restores
`Σ w_long = −Σ w_short` exactly, without requiring a full rebalance.

**Setup.** Let the book immediately before `tau_delist` satisfy
dollar-neutrality by construction (`portfolio.py::quantile_weights`):

```
Σ_{j in L} w_j  =  G/2          (long leg gross)
Σ_{j in S} w_j  =  −G/2         (short leg gross, w_j < 0)
```

where `L`/`S` partition the book into the long/short legs and `G` is the
reference gross exposure (`XSecConfig.gross`). Suppose the delisted name `i`
is in the long leg, `i ∈ L`, with pre-exit weight `w_i > 0` (the short-leg
case is the mirror image by the same argument with `L`/`S` swapped).

**Step 1 — the exit itself breaks the identity.** Forcing `w_i -> 0` while
leaving every other weight untouched produces:

```
Σ_{j in L, j != i} w_j  =  G/2 − w_i          (long leg now UNDER-gross by w_i)
Σ_{j in S} w_j          =  −G/2               (short leg unchanged)
```

The book is no longer dollar-neutral: it carries a net short bias of exactly
`−w_i` (the short leg is now larger in magnitude than the long leg by `w_i`).
This is precisely the invariant breach P6 requires OPS to detect and fix
without waiting for Friday.

**Step 2 — the renormalization.** Scale up every SURVIVING long-leg weight
proportionally so the long leg's gross is restored to `G/2`:

```
w_j'  =  w_j · ( G/2 ) / ( G/2 − w_i )     for every j in L, j != i
       =  w_j · 1 / ( 1 − w_i / (G/2) )
```

**Proof this restores the invariant.**

```
Σ_{j in L, j != i} w_j'  =  [ 1/(1 − w_i/(G/2)) ] · Σ_{j in L, j != i} w_j
                          =  [ 1/(1 − w_i/(G/2)) ] · ( G/2 − w_i )
                          =  ( G/2 − w_i ) / ( 1 − w_i/(G/2) )
                          =  ( G/2 − w_i ) / ( (G/2 − w_i)/(G/2) )        [common denominator]
                          =  G/2                                          ∎
```

So `Σ_{j in L} w_j' = G/2 = −Σ_{j in S} w_j`: dollar-neutrality is restored
exactly, with the short leg untouched. The scale factor
`1/(1 − w_i/(G/2))` is well-defined and finite (> 1) whenever `w_i < G/2`,
i.e. whenever the delisted name is not, absurdly, the entire long leg — a
condition that always holds for a quintile book with `>= 2` names per leg
(the minimum viable cross-section) and is asserted (§3.4 edge case) for a
degenerate single-name leg.

**Why proportional scaling, not equal top-up.** An equal-dollar top-up
(spread `w_i` evenly across the `|L|−1` survivors) would also restore the
leg's gross, but it changes the *relative* weighting scheme the signal chose
(inverse-realized-vol, `portfolio.py::realized_vol`-weighted, per `EDGE.md`
§b's "vol-scaled beats liquidity-capped and equal-weight net of cost"
finding) — it silently converts a vol-scaled position into a blend of
vol-scaled-plus-equal-weight for the remainder of the week. Proportional
rescaling preserves every survivor's *relative* weight
(`w_j'/w_k' = w_j/w_k` for all surviving `j,k`), so the renormalization is a
pure gross-restoration operation orthogonal to the sizing scheme — it does
not need to know or care whether the book was equal-weighted or vol-scaled to
be correct.

**The short-leg-delisting case.** By symmetry (replace `L` with `S`, `w_i`
with `|w_i|`, and the sign flips throughout), the identical proportional
rescale on the surviving short-leg weights restores `Σ w_S = −G/2`. Both legs
are never rescaled simultaneously for a single-name delisting — only the leg
that lost a name.

**Turnover/cost of the renormalization.** The renormalization is itself a
weight change on every surviving name in the affected leg, and therefore
incurs turnover exactly as `accounting.py::settle_week`'s
`turnover = 0.5 · Σ|w_new − w_carried|` formula already charges any weight
change. This cost is real and small (the survivors' weights each move by a
factor of `w_i/(G/2 − w_i)`, typically a few percent for a name sized at
1/N-th of a ~15-20-name leg) and is booked at the SAME round-trip friction
rate `COST.md`/`accounting.py::round_trip_friction` uses — OPS does not
invent a separate cost model for this rebalance-out-of-cycle event, it reuses
the one truth (`COST.md` P1).

### 3.4 Edge cases (explicit, not hand-waved)

- **Delisted name is the entire leg (degenerate `|L| = 1` or `|S| = 1`).**
  The proportional-scale formula's denominator `G/2 − w_i -> 0`, which is a
  genuine singularity (the leg cannot be "topped up" from zero survivors).
  This cannot occur for the deployed universe (`~50` names, quintile legs of
  `~10`, `EDGE.md` §b) under any single-name delisting, but the check is
  explicit: if the affected leg's surviving count would be `0`, OPS emits a
  forced `HALT` (not a renormalization) and defers to the next scheduled full
  rebalance — a single-name book cannot be dollar-neutral by construction and
  is out of scope for a proportional fix.
- **Two names delist inside the same week (independent events).** Apply §3.3
  sequentially: after the first delisting's renormalization restores
  neutrality, the second delisting is a fresh instance of the same invariant
  breach against the now-current (already-renormalized) weights, resolved by
  the identical formula. The proof in §3.3 makes no assumption about the
  weights being the "original" rebalance weights — only that they are
  dollar-neutral immediately before the exit — so sequential application is
  valid by induction.
- **Delisting notice arrives with insufficient liquidity to close at any
  reasonable price (extreme illiquidity, e.g. an exchange freezing
  withdrawals/trading entirely, not just delisting the perp).** The
  forced-liquidation rule (§3.2) degrades to "close what is fillable, mark
  the unfillable residual as a stuck position, and treat it as a
  `RECON_BREACH` (§1.4) on every subsequent daily reconciliation cycle until
  resolved" — this is the one scenario where §3.2's "immediately" cannot be
  literally honored, and OPS's honest response is escalation (§4), not a
  fictitious clean close.
- **A NEW listing appearing mid-week is NOT a symmetric event and requires NO
  live action.** The universe is re-evaluated point-in-time only at the next
  scheduled rebalance (`research/xsec/panel.py`'s point-in-time `listed`
  mask); a name newly meeting eligibility mid-week simply enters the
  candidate set at the next Friday close. There is no "forced entry"
  analogous to forced exit — under-investing in a not-yet-held opportunity is
  not a risk event, only a foregone-gain event, and OPS does not act on
  foregone gains (P3: OPS is a risk/control layer, not a return-seeking one).

> **Invariant DNEUT — dollar-neutrality is restored before the next
> scheduled rebalance, exactly, not approximately.** Test: a synthetic
> mid-week delisting of a randomly chosen long-leg (then short-leg) name,
> starting from an exactly dollar-neutral book, produces a post-renormalization
> book with `|Σw_long + Σw_short| < 1e-9` (floating-point tolerance only) and
> leaves every surviving pairwise weight ratio unchanged to the same
> tolerance. A renormalization that leaves a residual net bias, or that
> silently waits for Friday, fails this test.

---

## 4. The kill-switch — CUSUM sequential change-detection on trailing realized weekly Sharpe

### 4.1 Why CUSUM, and why on THIS series

A fixed threshold ("halt if last week's return < −5%") cannot distinguish a
single bad week — which the backtest's own `frac_pos ≈ 0.85–1.00` says will
happen roughly 1-in-7 to 1-in-15 weeks even in a fully healthy regime — from a
genuine regime break in the strategy's edge. What is needed is a **sequential
test accumulating evidence over time**, which is exactly Page's (1954) CUSUM
construction: designed to detect a *persistent* shift in a monitored
statistic's mean as fast as possible while controlling the false-alarm rate
under the null (no shift). We apply it to the **trailing realized weekly book
Sharpe** (equivalently, to standardized weekly log-return innovations —
either is a valid CUSUM input; we standardize on log-wealth increments, §4.2,
because Sharpe itself is a ratio statistic that is awkward to accumulate
directly in a CUSUM recursion, whereas its numerator, the return, is additive
across weeks and CUSUM's classical derivation is for an additive statistic).

To restate the distinction made in §0: `SYSTEM.md` §4–§5's deflated/
probabilistic-Sharpe and CPCV/PBO machinery is a STATIC offline promotion
gate — it does not use CUSUM. What OPS reuses from `SYSTEM.md` is the
VALIDATED NULL (`mu_0`, `sigma_0`, §4.2) that gate establishes, not its
detection method. The CUSUM/ARL *construction* itself is the piece genuinely
shared with `REGIME.md` §3–§4, which independently instantiates the same
Page (1954) sequential test against its own market-state null — one CUSUM/ARL
engine design, multiple independently-calibrated monitored streams, per §0's
opening framing.

### 4.2 The standardized monitored statistic

Let `r_k` be the realized `weekly_net_return` (§2) at rebalance `k`, and
define the standardized weekly innovation under the validated null
(healthy-strategy hypothesis, EVIDENCE-grounded). Following `SYSTEM.md`
§5.3's tagging convention, every numeric input below is marked
**[MEASURED]** (an `EDGE.md`/`SYSTEM.md`-cited number, by name/section),
**[DERIVED]** (computed from a MEASURED input by a stated formula, no new
assumption), or **[ILLUSTRATIVE]** (a RED policy choice, not itself a
measurement) at first use:

```
SR_0     = 0.85                                                          [MEASURED]
                    # the LOWER end of the honest net Sharpe band "~0.8-1.0"
                    #   (EDGE.md sec b); using the conservative/lower bound
                    #   makes the null a HARDER bar to fail, i.e. slower to
                    #   false-alarm — a healthy strategy tracking the upper
                    #   end of its own honest range is even further from the
                    #   drift the CUSUM is built to catch
mu_0     = SR_0 / sqrt(52.18)                                             [DERIVED]
                    # per-week mean return under the null, in return units
sigma_0  = 0.28 / sqrt(52.18)                                             [DERIVED]
                    # realized weekly-return std-dev under the validated backtest;
                    # 0.28 = ~28% ANNUALIZED vol at 100% gross            [MEASURED]
                    #   (from the SAME weekly_net_return series the honest Sharpe
                    #    was computed on, research/xsec/metrics.py, per EDGE.md/EVIDENCE)

z_k      = ( r_k − mu_0 ) / sigma_0     # standardized weekly innovation, mean 0 / var 1 under H0
```

`PERIODS_PER_YEAR = 52.18` matches `research/xsec/metrics.py`'s own
annualization constant — OPS does not introduce a second, inconsistent
week-count.

### 4.3 The two-sided CUSUM recursion and ARL calibration

**One-sided CUSUM statistics** (Page 1954), detecting a downward shift
(the only direction that matters for a kill-switch — an upward shift in
Sharpe is not a risk event):

```
C_0 = 0
C_k = max( 0 ,  C_{k-1} − z_k − d )         # accumulates evidence of NEGATIVE drift
```

`d` (RED; `≡` REGIME.md §4's `k`, per §0.1's notation table — renamed only to
avoid a local symbol clash with the `CUSUM` Python class in
`ops/detectors/cusum.py`, calibrated independently here against OPS's own
Sharpe-drift null, §4.2) is the classical **reference/allowance value**: a
small positive drift-tolerance below which accumulated evidence is allowed
to reset toward zero — the standard formulation is `d = delta/2` where
`delta` is the standardized shift size the chart is tuned to detect fastest.
We calibrate `d` and the decision boundary `h` (RED) jointly against a
**stated target ARL under the null** using the
classical two-parameter average-run-length design:

```
ARL_0(d, h)  = expected number of weeks between false alarms
               WHEN z_k really is iid N(0,1)   (the healthy-strategy null)
ARL_1(d, h)  = expected number of weeks to detect a shift of size `delta_target`
               once it actually occurs
```

**Calibration target (tied to a stated annual false-alarm budget, not
invented).** We fix a target false-HALT probability of **at most 5% per
year** for a strategy that is genuinely tracking its validated null (P4).
With 52.18 weekly observations per year, the corresponding target
`ARL_0` is:

```
P(>=1 false alarm in a year) <= 0.05
  <=>  P(no false alarm all year) >= 0.95
  <=>  (per-week survival prob)^52.18 >= 0.95     [(*) see justification below]
  <=>  per-week survival prob        >= 0.95^(1/52.18) ≈ 0.999017
  <=>  per-week false-alarm prob     <= 1 − 0.999017 ≈ 0.000983
  <=>  ARL_0  =  1 / (per-week false-alarm prob)
            ≈  1 / 0.000983
            ≈  1,018 weeks   (≈ 19.5 years)
```

**(*) Justifying the geometric-arrival step.** `ARL_0` is, by definition, a
mean first-passage time (Page 1954; Siegmund 1985 Ch. 2's renewal-theoretic
derivation) — the `survival_prob^n` conversion used above is the standard
large-`ARL_0` approximation treating that first-passage time as
geometrically distributed, valid whenever the alarm is rare relative to the
window being budgeted (Siegmund 1985 Ch. 2), exactly the regime here
(`ARL_0 ~ 1000 weeks >> 52 weeks/year`). This is a large-`ARL_0` asymptotic
only — it is deliberately not invoked on the `ARL_1` (detection-speed) side
below, where the shift is real and passage is fast.

This is a deliberately conservative target — a 5%/yr false-halt budget on a
weekly cadence requires an ARL measured in decades, because even a fairly
small per-week false-positive probability compounds over 52 chances/year to
false-alarm. We therefore set the **design target `ARL_0_target = 1,000`
weeks** `[ILLUSTRATIVE — RED policy choice]` (rounding down from the
≈1,018-week `[DERIVED]` figure above for a small safety margin — the
achieved ARL_0 is then, if anything, slightly ABOVE the 5%/yr budget, never
below it) and solve for `(d, h)` by the standard
closed-form CUSUM ARL approximation (Siegmund 1985 exponential-family ARL
formula for a one-sided Gaussian CUSUM), or equivalently by Monte-Carlo
simulation under the `N(0,1)` null (permitted here as a ONE-TIME offline
calibration — not a live-fitted model — consistent with P8):

```
ARL_0(d, h)  ≈  ( exp( 2·d·h ) − 2·d·h − 1 ) / ( 2·d^2 )        [Siegmund 1985, Wald-approximation
                                                                   form for a Gaussian one-sided CUSUM
                                                                   with reference value d and boundary h;
                                                                   valid for d, h > 0]
```

- `d = delta_target/2` where `delta_target` is the standardized Sharpe shift
  the chart should detect reasonably fast — we set `delta_target = 1.0`
  `[ILLUSTRATIVE — RED policy choice]`: a full
  1-standard-deviation-of-weekly-Sharpe shift, i.e. roughly "the strategy's
  realized weekly Sharpe has moved to about zero or below from its
  ~0.85-1.0 honest range" — the shift size that represents "the edge as
  validated is plausibly gone," not a normal week-to-week wobble. This
  gives `d = 0.5` `[DERIVED]`.
- **Why `delta_target = 1.0` and not a more aggressive (smaller) shift.** A
  smaller `delta_target` would detect a smaller Sharpe drift faster, but the
  strategy's own overfitting diagnostic argues against chasing that
  sensitivity: `SYSTEM.md` §4.4 reports **PBO = 0.357** for the deployed
  lookback family (partition-sensitivity range `[0.31, 0.47]` across
  alternative `(N, k_test)` CPCV choices), i.e. the in-sample-best
  configuration's edge over the family median is itself only moderately
  distinguishable from noise. A kill-switch tuned to fire on a SMALL
  standardized drift would be reacting at a sensitivity finer than the
  strategy's own validation can currently distinguish signal from
  partition-noise at — it would risk false-alarming on exactly the kind of
  benign cross-partition Sharpe wobble PBO=0.357 already says is expected
  under a real, but moderately-overfit-flagged, edge. `delta_target = 1.0`
  (a full standardized-Sharpe unit) is deliberately set coarser than that
  noise floor: it targets "the edge is plausibly gone," not "the edge moved
  within its own already-acknowledged uncertainty band." This is the same
  reasoning direction as `mu_0`'s choice of the lower `SR_0 = 0.85` bound
  (conservative in the same direction, harder to false-alarm), applied to the
  shift-size parameter rather than the null mean.
- Solving `ARL_0(0.5, h) = 1000` for `h` via the Siegmund formula (numeric
  root-find/bisection on the exponential — no ML, a one-time scalar solve)
  gives `h* ≈ 6.23` `[DERIVED]`. This module ships `h* = 6.23` as a RED
  constant derived exactly this way, with the derivation reproducible
  offline (`ops/validate/arl.py`, §7) rather than re-solved live.
- With this calibration, the corresponding `ARL_1` (detection speed once a
  true `delta_target = 1.0` shift occurs) is materially shorter than
  `ARL_0` by construction (the whole point of CUSUM vs a fixed-threshold
  rule: fast detection of a real shift, slow false-alarm under the null) —
  `ops/validate/arl.py` reports both numbers, and the offline calibration
  report is a build acceptance gate (§7).
- **The track-record tension this calibration inherits (stated here, not
  deferred to an end-of-document risk note).** `mu_0`/`sigma_0` (§4.2) are
  anchored to the SAME validated backtest distribution that `SYSTEM.md` §5.5
  itself flags as **track-length-insufficient**: `n_obs ≈ 258` weekly
  observations (`SYSTEM.md` §5.2) versus a required `MinTRL ≈ 824` weeks
  (`SYSTEM.md` §5.5) for the deflated-Sharpe gate to certify the edge at its
  stated confidence — the strategy passes on `PSR = 0.994` (the *sign* of the
  edge is not in serious doubt) but fails the DSR gate specifically on
  track-record length, not on a low point estimate (`SYSTEM.md` §5.3-5.5).
  Consequently, `ARL_0_target = 1000` weeks (≈19 years) is calibrated against
  a null whose own parameters are certified only over `~258` observed weeks —
  the CUSUM's false-alarm rate is a correct 5%/yr statement CONDITIONAL on
  `(mu_0, sigma_0)` being the strategy's true long-run parameters, but that
  conditioning assumption is exactly the one `SYSTEM.md` §5.5 says cannot yet
  be certified over 19 years of live history, because only 5 years of
  history exist. This is not a flaw unique to OPS — every live threshold
  calibrated against this backtest inherits the same conditioning — but it
  means Invariant CAL's guarantee should be read as "5%/yr false-HALT rate if
  the validated null holds," not as an unconditional, independently-verified
  19-year guarantee. As `n_obs` accumulates live (crossing `MinTRL` gives
  SYSTEM's DSR gate its own resolution), `mu_0`/`sigma_0` should be
  re-derived from the enlarged sample (§7's slow-cadence `sigma_0`
  re-estimation already provides the mechanism; see §13's edge case) rather
  than left pinned to the original 258-week estimate indefinitely.

**Two-sided extension.** A symmetric upper-sided statistic
`C_k^+ = max(0, C_{k-1}^+ + z_k − d)` is also tracked for completeness/
telemetry (an anomalously GOOD run is itself worth flagging as "check whether
this is real or a data/accounting bug" — a healthy strategy should not show
a persistent multi-sigma-above-null run either) but only `C_k` (the
downward-drift statistic) feeds the kill-switch action in §4.4; `C_k^+`
breaching its own (identically-calibrated) boundary is logged as an
`ANOMALOUS_OUTPERFORMANCE` diagnostic, never a HALT.

### 4.4 Escalation states from the CUSUM statistic

```
                     C_k  <   h*/2               -> RUN     (healthy; no action)
        h*/2   <=   C_k  <   h*                  -> DEGRADE (reduce gross; see below; log CUSUM_WARN)
                     C_k  >=  h*                  -> HALT    (flatten book via §6; log CUSUM_ALARM)
```

- `DEGRADE` (crossing the half-boundary `h*/2`) scales the book's target
  gross by a multiplier `g_degrade = 0.5` `[ILLUSTRATIVE — RED policy
  choice]` — this is a size cut, not a full flatten, and it is the ONE place
  OPS is permitted to influence
  sizing (mirroring `SYSTEM.md`'s exposure-brake composition: it multiplies
  the same gross channel a REGIME governor would, on a *disjoint*
  trigger — reconciliation/Sharpe-drift, not market-state — so a REGIME cut
  and an OPS `DEGRADE` compose multiplicatively, never as a double-counted
  single cut).
- **Caveat: multiplicative composition is not automatically harmless, and
  this spec does not assume it is.** `REGIME.md` §5.2-5.4 documents a
  concrete, tested instance of a drawdown-adjacent gross-cut mechanism being
  net-HARMFUL rather than protective: a causal drawdown-stop on the funding
  carry sleeve caps maxDD at ~24% (down from ~70-75%) but collapses Sharpe
  from 0.57 (with a beta-hedge overlay) to 0.10 and income from ~12.9%/yr to
  ~0.6%/yr (`REGIME.md` §5.3), precisely because carry's drawdown is a
  **persistent regime-bleed** (REGIME's §5.2 taxonomy) where the stop sits
  out the eventual recovery rather than avoiding a genuine loss. OPS's
  `DEGRADE` is a DIFFERENT trigger (Sharpe-drift CUSUM on the strategy's own
  realized weekly return, not a raw-equity drawdown threshold) and momentum's
  documented bad regime is a sharp reversal week, not a slow bleed — closer
  to REGIME §5.2's isolated-tail-event case, where a stop is more likely to
  help than hurt — so the carry finding does NOT mechanically transfer to
  OPS's `DEGRADE` (`REGIME.md` Invariant STOP-XFER makes exactly this
  non-transfer explicit for any drawdown-adjacent mechanism, on any sleeve,
  including OPS's). But "does not mechanically transfer" is not "is proven
  safe": composing OPS's `DEGRADE` multiplier with a future REGIME
  `g_t` cut multiplicatively means a week where BOTH fire simultaneously
  produces a gross cut deeper than either alone intended, and neither this
  spec nor REGIME's has run REGIME §5.4's regime-conditional ablation
  methodology against the SPECIFIC composition of "OPS Sharpe-CUSUM DEGRADE
  active at the same time as a REGIME cut." Until that joint ablation exists,
  treat the multiplicative-composition claim as a design commitment
  (consistent with `ARCHITECTURE.md`'s disjoint-axis intent, itself flagged
  stale above and due for its own pivot rewrite), not as an empirically
  validated one — flagged explicitly here rather than left implicit, per
  this file's own edge-case discipline (§13 adds the corresponding scenario).
- `HALT` (crossing the full boundary `h*`) triggers the forced-flatten
  sequence (§6) — the book is closed via the venue, not merely "not
  rebalanced next week."
- **Reset rule (P7 — monotone, no silent self-heal).** Once `C_k >= h*/2`
  (DEGRADE or HALT), the CUSUM statistic itself does NOT reset to 0 on the
  next good week automatically — the raw recursion `C_k = max(0, C_{k-1} -
  z_k - d)` already decays toward 0 on a run of good weeks (that is intrinsic
  to CUSUM and is fine, it is how the statistic is *supposed* to behave under
  a genuinely healthy resumed regime), but the **operational state**
  (`DEGRADE`/`HALT`) requires an explicit confirmation: `RUN` is only
  re-entered after `C_k` has stayed below `h*/2` for `n_confirm = 4`
  `[ILLUSTRATIVE — RED policy choice]` CONSECUTIVE weekly observations — a
  cool-down/confirmation window, not an
  instant flip back on the statistic alone crossing back under the line once.
  This prevents a statistic that is genuinely re-accumulating from noise
  right at the boundary from flapping `RUN`/`DEGRADE`/`RUN` week to week.

> **Invariant CAL — the kill-switch boundary is calibrated to a stated
> annual false-alarm probability under the validated null, and the
> calibration is reproducible offline.** Test: Monte-Carlo-simulating 100,000
> years of `z_k ~ iid N(0,1)` (the null) through the calibrated `(d, h*)`
> recursion produces an empirical false-HALT rate within Monte-Carlo error of
> the target ≤5%/yr (`ARL_0 ≈ 1000` weeks); simulating the same volume of
> years with a persistent `delta_target = 1.0` downward shift injected shows
> materially faster detection (`ARL_1 << ARL_0`).

> **Invariant MONO — escalation is monotone within an episode; recovery
> requires confirmation, not a single good observation.** Test: a synthetic
> series that crosses into `HALT`, then immediately returns one single
> healthy `z_k`, remains in `HALT` (does not silently revert to `RUN`); only
> `n_confirm` consecutive weeks below `h*/2` restores `RUN`.

---

## 5. Rebalance-failure alerting — the execution-window SLA

### 5.1 The SLA budget

The weekly rebalance is not instantaneous: `n_trades` order legs must fill
inside a bounded **execution window** `W_exec`, which is the SAME quantity
`EXECUTION.md` §2.1 calls `Delta` — the AMBER, venue-liquidity-set intra-day
window `[t_0, t_0+Delta]` immediately following the Friday close over which
EXECUTION's Almgren-Chriss trajectory (Almgren & Chriss 2001, `EXECUTION.md`
§2.4) is planned, with `EXECUTION.md` citing `Delta ≈ 1–4 hours` as the
concrete order of magnitude (short relative to `rebalance_days = 7` so the
backtest's instantaneous-fill approximation stays first-order reasonable, but
long enough that the plan itself does not create the impact being managed).
OPS does not decide execution tactics or set `Delta`'s value — it monitors
whether the REALIZED completion time and REALIZED slippage stayed inside the
budget EXECUTION planned against; `W_exec` in this file is EXECUTION's
`Delta`, referenced not duplicated (OPS carries no independent numeric value
for it, per COST-style P1 provider/decider discipline).

```
t_start        = the scheduled rebalance instant t_k (Friday close)
t_complete     = the wall-clock timestamp of the LAST fill that settles the
                 target book (realized, from Nautilus fill events)
completion_lag = t_complete − t_start

SLA:  completion_lag  <=  W_exec   (:= EXECUTION.md's Delta, AMBER,
                                     ~1-4 hours per EXECUTION.md §2.1;
                                     OPS reads this value, never re-derives it)
```

### 5.2 Slippage SLA

In parallel, the realized round-trip implementation shortfall for the
rebalance — `EXECUTION.md` §7.1's `IS_total` (Perold 1988: decomposed into
fees + realized half-spread + realized market impact + delay/opportunity
cost) — is compared against the cost model the strategy was sized against.
**Ingestion path, corrected.** An earlier draft of this section cited
`cost/facts/execution_adapter.py` as the feed from EXECUTION into COST; that
module does not exist, and `EXECUTION.md` §7.2 (already corrected in that
file, after the same fabrication was caught there) is explicit that COST has
**no** such ingestion seam — `COST.md` §11 (Non-goals) states plainly there
is no EXECUTION pillar in COST's own scope premise. `EXECUTION.md` §7.2
instead commits `IS_total` to feed forward through `OpsEngine.on_obs` — i.e.
THIS spec's own §2 `realized_slippage` stream (`core/protocols.py`'s
`OpsEngine` Protocol, §9's disclosed collision aside) is the correct, single
ingestion point, not a COST-side adapter. §2's stream table is updated
accordingly (`realized_slippage` sourced from EXECUTION's realized
`IS_total` once EXECUTION exists, COST's realized-fill TCA as the interim
proxy) — this file does not name a non-existent COST module as the path.

```
SLA:  realized_slippage_bps  <=  m_slip · expected_slippage_bps
```

`m_slip = 2×` `[ILLUSTRATIVE — RED policy choice, decomposition-anchored
below]` — realized slippage is allowed to run up to a stated
multiple of the expected/modeled friction (`accounting.py::round_trip_friction`
= 13.0 bp `[MEASURED]` round trip at the crypto majors' default parameters, the
EVIDENCE digest's cost model) before it is treated as an anomaly; realized cost
running consistently at 1.0-1.5x the model is ordinary estimation noise,
running at 2x+ is evidence either the cost model needs re-calibration or
market conditions during this specific rebalance were unusually adverse
(a thin-liquidity week, a venue outage during the window) — either way, worth
a flag.

**Why `2×` is a defensible bound, decomposition-anchored (not a bare
convention).** `m_slip` cannot be derived to a point — there is no persisted
cross-sectional dispersion (std/CV) of realized-vs-modeled slippage across
the ~50-name universe anywhere in this repo, and this spec does not invent
one. But the 13bp figure it multiplies is not uniformly uncertain: `COST.md`
§5.1-5.3 decomposes `round_trip_friction = 2·(fee + half_spread_i/2 +
impact_i)` into ONE exact, contractual term and two estimated, calibrated
terms —

```
fee          = 5.0 bp   (Binance USDT-M VIP0 taker rate, §5.1 — a contractual
                          constant, not an estimate; exact by construction,
                          zero realized/modeled deviation possible absent a
                          fee-schedule change)
half_spread  ≈ 1.0 bp per leg (§5.2, c/sqrt(ADV_i) at the median-liquidity
                          anchor — CALIBRATED, carries real estimation error)
impact       ≈ 0.5 bp per leg (§5.3, k_amihud·sqrt(Amihud_i·Notional) at the
                          same anchor — CALIBRATED, carries real estimation
                          error)
round trip   = 2·(5.0 + 1.0/2 + 0.5) bp = 2·(5.0+0.5+0.5) bp = 12 bp
             ≈ 13 bp reported by the flat research baseline
               (accounting.py's un-halved half-spread convention, COST.md §5.5 —
               either parameterization nets the SAME ~1bp/leg empirical spread cost)
```

Of the ~13bp total, **~10bp is the fixed fee component** (5bp fee × 2 legs)
and only **~3bp is the calibrated, stochastic sub-component** (half-spread +
impact, doubled: `2·(0.5+0.5)bp = 2bp` at the §5.2 halved-half-spread
convention, or up to `2·(1.0+0.5)bp = 3bp` at the flat-baseline
un-halved convention — take the wider, more conservative `~3bp` figure). The
fee term cannot deviate from its modeled value in normal operation (a
VIP0-taker order either pays 5bp or the fee schedule itself changed, which is
a `RECON`/governance event, not a slippage surprise); realized-vs-modeled
deviation can therefore only come from the ~3bp stochastic sub-component.
This gives a principled, if still illustrative, sanity floor for `m_slip`:

```
even a generous 3-4x MISS on the ENTIRE stochastic sub-component
  (realized half-spread+impact running 3-4x the modeled ~3bp, i.e. 9-12bp
   instead of 3bp — a large miss by any liquidity-shock standard)
  moves total realized cost to only
    10bp (fixed fee, unchanged) + 9-12bp (stochastic, 3-4x missed)
    = 19-22bp
  ≈ 1.5-1.7x the total 13bp modeled figure

therefore: m_slip = 2x ON THE TOTAL is a MATERIALLY LOOSER bar than a 3-4x
miss confined to just the uncertain (~23% of the total) sub-component —
i.e. the SLA does not fire until realized cost is worse than what even a
severe (3-4x) miss on ONLY the calibrated terms would produce
```

This is the same style of bound `LEVERAGE.md` uses for `G_max = 1.5` (not
derived to a single point from a loss function, but shown to sit safely
outside the range a disciplined stress scenario would produce) — it does not
manufacture a fitted quantile that does not exist in this repo, it derives a
defensible floor from the ALREADY-DERIVED, ALREADY-CITED fee/half-spread/
impact decomposition (`COST.md` §5.1-5.3, no new numbers introduced). The
residual, honest gap — there is still no empirical realized/expected ratio to
calibrate `m_slip` AS a stated-confidence quantile, only this decomposition
bound — remains a disclosed limitation (Risks section), not resolved by this
argument; the argument narrows what "unbacked" means from "no reasoning at
all" to "a real bound exists, but it is not yet a fitted, EVIDENCE-cited
quantile."

### 5.3 Escalation policy

```
  completion_lag <= W_exec  AND  realized_slippage <= m_slip · expected  -> OK
  completion_lag  > W_exec  OR   realized_slippage  > m_slip · expected  -> SLA_BREACH (WARN)
  completion_lag  > 2·W_exec OR  realized_slippage  > 3·m_slip·expected  -> SLA_BREACH_SEVERE -> HALT
```

- A single `SLA_BREACH` (WARN tier) does not halt — it is logged with full
  attribution (which leg was slow/expensive, what the venue conditions were)
  and surfaces in the next reconciliation review. A rebalance that runs a bit
  long or costs a bit more some weeks is expected variance, not a strategy
  failure.
- A `SLA_BREACH_SEVERE` (the book took more than twice its budgeted window,
  or cost more than 3x the multiple-of-expected threshold) escalates directly
  to `HALT` via the same `OpsDirective` channel as the CUSUM kill-switch and
  the reconciliation breach (§1.4, §4.4) — a rebalance that is dramatically
  slow or dramatically expensive is itself evidence something is
  structurally wrong RIGHT NOW (a venue outage, an API failure mid-rebalance
  leaving the book in an inconsistent partial state), and is not something a
  statistical filter should be allowed to average away before acting.
- **Monitoring cadence — intraday, inside the window (P2).** SLA monitoring
  does not wait until `W_exec` has fully elapsed to check; it polls at a RED
  sub-window cadence (e.g. every 15 minutes during an active rebalance) so a
  rebalance that is clearly going to blow its budget (e.g. half the legs
  unfilled at `0.75 · W_exec` elapsed) can escalate before the full window
  expires, not only in a post-mortem after the fact.

> **Invariant SLA — a severe rebalance failure escalates to HALT without
> waiting for the next weekly CUSUM observation.** Test: a synthetic
> rebalance whose last fill settles at `2.5 · W_exec` after the scheduled
> start produces `SLA_BREACH_SEVERE` and an `OpsDirective(HALT)` before the
> following week's `weekly_net_return` observation is even available to the
> CUSUM detector.

---

## 6. The forced-flatten sequence (what HALT actually does)

`HALT`, from any source (§1.4 `RECON_BREACH`, §4.4 CUSUM kill-switch, §5.3
`SLA_BREACH_SEVERE`, or the heartbeat-silence trigger of §2), triggers ONE
shared sequence — there is exactly one flattening procedure, never a
per-trigger bespoke close routine (mirroring `COST.md` Invariant X's "one
interface, per-class body" discipline, specialized here to "one flatten
routine regardless of which alarm fired it"):

```
ON OpsDirective(HALT):
  1. Halt all NEW order submission immediately (no new opens, no rebalance
     continuation, regardless of where in the weekly cycle the alarm fired).
  2. Close every open position via the venue at the best immediately
     achievable price (IOC/market — urgency dominates once HALT has fired,
     identical in spirit to §3.2's delisting closure).
  3. Do NOT attempt to preserve dollar-neutrality during the flatten — a
     HALT is a full-book exit, not a partial rebalance, so §3.3's
     renormalization machinery does not apply (there is no "surviving leg" to
     renormalize; both legs are going to zero).
  4. Log the full reason chain: which alarm(s) fired, their values at the
     instant of the trigger, and the realized flatten fills/cost.
  5. Remain in HALT until an explicit operator action re-arms the system
     (P7 — no auto-restart from HALT; DEGRADE's confirmation-window
     auto-recovery of §4.4 applies ONLY to DEGRADE, never to HALT).
```

> **Invariant HALT-ONE — exactly one flatten routine, triggered identically
> regardless of source.** Test: forcing `RECON_BREACH`, `CUSUM_ALARM`, and
> `SLA_BREACH_SEVERE` independently each invoke the SAME
> `brake.kill_switch.flatten()` code path (asserted by a shared code-path
> test, not three independent implementations), and none of the three
> triggers a `HALT` from which the system silently self-restarts.

---

## 7. Parameter governance

Every OPS parameter is exactly one bucket (same four buckets as every sibling
spec — `core/params.py::Bucket`).

- **TEAL — fitted per-instrument/per-fold from realized history, causal.**
  OPS fits almost nothing (it is a control layer, P8). The one TEAL-ish
  object is `sigma_0`, the realized weekly-return standard deviation used to
  standardize `z_k` (§4.2) — re-estimated periodically from the trailing
  realized `weekly_net_return` series (a rolling causal estimate), not
  hand-fixed forever, but ALSO not re-fit every observation (re-estimating
  `sigma_0` too reactively would let the null itself drift to chase whatever
  is currently happening, defeating the point of a change-detector — see
  Edge cases below). Concrete cadence: `sigma_0_reestimation_weeks = 52`
  (RED, e.g. annual — the same order of cadence as a full validation re-run,
  deliberately `>>` the weekly observation rate and `>>` `n_confirm`'s 4-week
  window, so the null cannot adapt within a single DEGRADE/HALT episode); the
  re-estimation window itself is the FULL trailing realized sample available
  at that point (not a short rolling window), consistent with the Edge-cases
  discussion below.
- **AMBER — supplied metadata.** Per-instrument `lot_size_i` (venue
  `stepSize`), the rebalance calendar/cadence source (`rebalance_days = 7`,
  matching `research/xsec/engine.py::XSecConfig.rebalance_days`), venue
  position/fill endpoints.
- **RED — shared control constants.** `k_lot`, `k_notional` (§1.2 tolerance
  scaling), `m_warn` (§1.4), `staleness_max` (§1.4), `mu_0`'s underlying
  `SR_0` and `sigma_0`'s underlying annualized-vol anchor (§4.2, tied to
  EVIDENCE), `sigma_0_reestimation_weeks` (above), `delta_target` and its
  derived `d` (§4.3), the calibrated CUSUM boundary `h*` (§4.3, derived not
  guessed), `g_degrade` (§4.4), `n_confirm` (§4.4), `W_exec` and `m_slip`
  (§5), the SLA severe-multiple (§5.3), the reconciliation cadence (§1.3).
- **BLUE — N/A for v1.** OPS holds no finite validated search grid: it is a
  control layer whose constants are calibrated once (§4.3's ARL derivation)
  against a stated target, not searched for best backtest performance. There
  is deliberately no BLUE bucket here to tune — Invariant SG-style trial
  accounting (`SYSTEM.md`) does not apply to OPS because OPS is not part of
  the alpha-search surface (P8 in spirit: nothing here is fit on P&L).

> **Invariant C — RED unreachable from per-instrument config** (inherited
> from every sibling spec). RED enters only via `resolve_config(defaults=...)`.
> No reconciliation tolerance multiple, no CUSUM boundary, no SLA budget
> appears as a per-instrument field.

---

## 8. Directory layout

```text
ops/
+-- config/
|   +-- defaults.py          # OpsDefaults: RED shared constants (epsilon scale, h*, W_exec, ...)
|   +-- in_control.py        # the validated null (SR_0, sigma_0) OPS calibrates against — cites EDGE.md §b (SR_0, vol) and SYSTEM.md §4 (frac_pos, PBO, CPCV Sharpe) separately, never conflated
|   +-- asset_class.py       # AssetClassOpsPreset: per-class overrides (crypto-only in v1)
|   +-- severity.py          # WARN/BREACH/DEGRADE/HALT threshold multiples
|   +-- resolve.py           # single merge point (RED unreachable from instrument)
+-- detectors/                # generic SPC primitives (reusable by REGIME, per this spec's framing)
|   +-- base.py               # Detector protocol: on_observation(x) -> DetectorState
|   +-- cusum.py              # CUSUM recursion (Page 1954) + Siegmund ARL approximation
|   +-- shewhart.py           # single-point control-limit rule (RECON_BREACH, SLA_BREACH_SEVERE)
|   +-- ewma.py               # (reserved; not used by v1 kill-switch, available for REGIME reuse)
|   +-- page_hinkley.py       # (reserved; alternative CUSUM variant for future use)
|   +-- sprt.py               # (reserved; sequential-probability-ratio variant for future use)
+-- streams/                  # adapt realized facts into the monitored series of §2
|   +-- whiten.py             # z_k standardization (§4.2)
|   +-- fills.py              # realized fill/completion-time adapter (§5)
|   +-- cost_tca.py           # realized slippage adapter from COST's realized TCA (§5.2)
|   +-- vol.py                # rolling sigma_0 estimator (TEAL, §7)
|   +-- edge.py               # weekly_net_return adapter from research/xsec accounting (§2)
|   +-- recon.py              # book vs exchange position snapshot (§1)
|   +-- heartbeat.py          # liveness signal (§2)
+-- reconcile/
|   +-- streams_map.py        # instrument -> epsilon_i (§1.2), the WARN/BREACH mapping (§1.4)
+-- brake/
|   +-- triggers.py           # RECON_BREACH / CUSUM_ALARM / SLA_BREACH_SEVERE -> OpsDirective
|   +-- alarms.py             # alarm dataclasses + reason-chain logging
|   +-- safe_state.py         # RUN / DEGRADE / HALT state machine + confirmation-window (§4.4)
|   +-- kill_switch.py        # the shared flatten() routine (§6)
|   +-- engine.py             # OpsEngine.on_obs() — the single Protocol entry point
+-- nautilus/
|   +-- adapter.py            # the ONLY nautilus-importing OPS code: reads fills/positions
|   +-- flatten.py            # submits the venue-level closing orders for kill_switch.flatten()
+-- forensics/
|   +-- log.py                # append-only alarm/decision log (the reason chain of every HALT)
|   +-- replay.py             # replay a historical episode's stream through the detectors offline
+-- validate/
|   +-- derive_limits.py      # the epsilon_i / h* / W_exec DERIVATIONS (this file's §1.2/§4.3), not asserted
|   +-- arl.py                # Monte-Carlo ARL_0 / ARL_1 verification (Invariant CAL's test)
|   +-- oos.py                # replay OPS against the historical CPCV paths (SYSTEM.md §4) as a backtest of the monitor ITSELF
|   +-- baselines.py          # fixed-threshold kill-switch baseline, for contrast with CUSUM
|   +-- reports.py
+-- synth/
|   +-- generate.py           # synthetic weekly-return series with injected drift/delisting/recon-breach events
+-- types.py                  # OpsDirective, DetectorState, Alarm — frozen dataclasses crossing boundaries
+-- ops_api.py                # public entry point: OpsEngine.on_obs() -> OpsDirective
tests/
+-- ...                       # unit, invariant (RECON/CAL/MONO/DNEUT/SLA/HALT-ONE), worked examples
```

Stack: Python 3.11+, numpy, scipy (`norm`, offline Monte-Carlo only — not in
the live hot path), pandas, pytest. No ML frameworks (P8).

---

## 9. Shared data objects (`types.py`) — and the STALE `core/types.py` collision

**Cross-consistency disclosure (not silently left for an implementer to find
by diffing two files).** `src/convexity/core/protocols.py` already declares a
real, importable `OpsEngine` Protocol with the EXACT method signature this
spec commits to (`on_obs(self, obs: object) -> OpsDirective`, plus a
`directive() -> OpsDirective` accessor) — confirmed by direct inspection,
this is the same seam `EXECUTION.md` §7.2 names as "the ONE real, existing
monitoring seam in this repo." But `core/types.py`'s existing `OpsDirective`
dataclass is the **PRE-PIVOT, directional-breakout-era shape**:

```python
# src/convexity/core/types.py — EXISTING, PRE-PIVOT, STALE for this spec
@dataclass(frozen=True)
class OpsDirective:
    state: KillState                  # pre-pivot enum, NOT this spec's Literal["RUN","DEGRADE","HALT"]
    flatten: bool
    cancel_entries: bool
    reason_codes: tuple[str, ...]
    engine_version: str
```

against THIS spec's `OpsDirective` (below), which has a materially different
field set (`gross_multiplier: float` and `alarms: tuple[Alarm, ...]` do not
exist in `core/types.py`'s version; `state` is a different type entirely —
`KillState`, a pre-pivot enum, vs. this spec's `Literal["RUN","DEGRADE","HALT"]`).
`core/protocols.py`'s own import block pulls `OpsDirective` from
`core.types` (`from convexity.core.types import (..., OpsDirective, ...)`),
so as written today, the Protocol's return-type annotation resolves to the
STALE shape, not this spec's.

**Why this is flagged here explicitly, mirroring `EXECUTION.md` §7.2's own
corrected-seam disclosure.** A builder who reads `EXECUTION.md` §7.2 is told
`OpsEngine.on_obs(obs: object) -> OpsDirective` is "the ONE real, existing"
composition seam and is directed to route realized `IS_total` through it.
That same builder, following THIS spec's §9 verbatim, would define a SECOND,
incompatible `OpsDirective` — same name, colliding import path, divergent
shape — rather than reconciling with the one `core/protocols.py` already
declares. This is exactly the class of citation/interface mismatch the prior
adversarial review passes exist to catch (cf. `EXECUTION.md` §7.2's own
correction of a previously-fabricated `cost/facts/execution_adapter.py`
seam); the difference here is that BOTH sides of the mismatch are real code
(`core/protocols.py`'s `OpsEngine`/`OpsDirective` import genuinely exist,
confirmed by direct inspection), so the fix is reconciliation, not removal of
a fabrication.

**What this spec commits to (buildable, not silently left ambiguous).**
`core/types.py`'s current `OpsDirective` (`state: KillState`, `flatten`,
`cancel_entries`, `reason_codes`, `engine_version`) is the **STALE, pre-pivot
shape** — it describes the retired single-instrument governor's directive
object (the same directional-breakout era `ARCHITECTURE.md`'s staleness note,
§0, already flags for the composition graph as a whole), not the
cross-sectional book-level directive this spec defines. **THIS spec's
`OpsDirective` (below) supersedes `core/types.py`'s version.** Reconciling the
two is a named, scoped piece of build work, not a decision left implicit for
whoever builds `ops/`:

1. `core/types.py`'s `OpsDirective` dataclass must be replaced with this
   spec's shape (`state: Literal["RUN","DEGRADE","HALT"]`, `gross_multiplier:
   float`, `alarms: tuple[Alarm, ...]`, `as_of: int`) — a rename-in-place, not
   an additive second class, because `core/protocols.py`'s `OpsEngine`
   Protocol and every pillar that imports `OpsDirective` from `core.types`
   must resolve to ONE shape, never two same-named classes shadowing each
   other across modules.
2. `KillState` (the enum `core/types.py`'s stale `OpsDirective.state` field
   references) becomes dead code once (1) lands and should be removed in the
   same change, not left as an orphaned pre-pivot type — a lingering unused
   `KillState` import is exactly the kind of stale cross-reference this
   disclosure exists to prevent from recurring.
3. `Alarm` (this spec's own `ops/types.py`, per §8's directory layout) must
   also be added to (or re-exported from) `core/types.py` if
   `OpsDirective.alarms` is to be constructible outside the `ops/` package —
   otherwise `core/protocols.py`'s Protocol-typed consumers (SYSTEM's
   orchestrator, per the composition graph in `core/protocols.py`'s own
   docstring) cannot type-check against a concrete `Alarm` shape without a
   second import path.
4. This reconciliation is `core/types.py`-owned build work (outside this
   file's remit, which is `docs/specs/OPS.md` and `docs/ARCHITECTURE.md`
   only) — flagged here, and restated in the Risks section, as a concrete,
   scoped prerequisite before `ops/brake/engine.py`'s `OpsEngine.on_obs` can
   be implemented against BOTH `core/protocols.py`'s Protocol AND this
   spec's `OpsDirective` without a naming collision. It is not a hypothetical
   future risk: the collision exists in the repo TODAY, confirmed by direct
   inspection of both files, and blocks nothing about writing `ops/` code
   against THIS spec's shape, but it DOES block `core/protocols.py`'s
   `OpsEngine` Protocol from being satisfiable by that code until (1)-(3)
   land.

```python
@dataclass(frozen=True)
class OpsDirective:
    """The one thing OPS emits (P3). Totally ordered: RUN < DEGRADE < HALT.
    SUPERSEDES core/types.py's current (stale, pre-pivot) OpsDirective — see
    the reconciliation note above; this is the shape core/types.py must be
    updated to before core/protocols.py's OpsEngine Protocol is satisfiable
    by an implementation of this spec."""
    state: Literal["RUN", "DEGRADE", "HALT"]
    gross_multiplier: float          # 1.0 at RUN, g_degrade at DEGRADE, 0.0 at HALT
    alarms: tuple["Alarm", ...]      # the reason chain; empty at RUN
    as_of: int                       # epoch-ms of the observation that produced this directive


@dataclass(frozen=True)
class Alarm:
    """One fired alarm, fully attributed."""
    kind: Literal["RECON_BREACH", "RECON_STALE", "CUSUM_ALARM", "CUSUM_WARN",
                   "SLA_BREACH", "SLA_BREACH_SEVERE", "ANOMALOUS_OUTPERFORMANCE",
                   "HEARTBEAT_SILENCE"]
    detail: str                      # human-readable attribution (which instrument/value/threshold)
    value: float                     # the triggering statistic's value
    threshold: float                 # the threshold it crossed
    as_of: int


@dataclass(frozen=True)
class DetectorState:
    """One detector's internal recursion state, checkpointed for restart-safety."""
    detector: str                    # "cusum_sharpe", "recon", "sla", ...
    statistic: float                 # C_k or the analogous accumulated value
    n_obs: int
    last_update: int
```

---

## 10. Invariants (index; full statements inline above)

- **Invariant RECON (§1.4)** — daily reconciliation; a breach escalates
  without waiting on a statistical filter.
- **Invariant CADENCE (P2)** — every control check runs strictly more
  frequently than the weekly rebalance it protects.
- **Invariant PROV (P3)** — OPS emits only `{RUN, DEGRADE, HALT}` plus
  reason chain; never a weight, never a per-instrument veto.
- **Invariant CAL (§4.3)** — the kill-switch boundary is calibrated to a
  stated annual false-alarm probability under the validated null, calibration
  reproducible offline.
- **Invariant MONO (§4.4, P7)** — escalation is monotone within an episode;
  recovery from DEGRADE requires `n_confirm` consecutive healthy weeks; HALT
  never auto-recovers.
- **Invariant DNEUT (§3.3, P6)** — dollar-neutrality is restored exactly
  before the next scheduled rebalance after any forced exit.
- **Invariant SLA (§5.3)** — a severe rebalance failure escalates to HALT
  without waiting for the next weekly CUSUM observation.
- **Invariant HALT-ONE (§6)** — exactly one flatten routine, invoked
  identically regardless of trigger source.
- **Invariant C** — RED unreachable from per-instrument config (inherited,
  every sibling spec).

---

## 11. Validation

OPS validates itself as a MONITOR, which is a different exercise from
validating a signal: the question is not "does this produce alpha" (it
produces none, by design) but "does this correctly distinguish healthy from
unhealthy episodes, at the calibrated false-alarm rate."

1. **ARL verification (`validate/arl.py`, Invariant CAL's test).**
   Monte-Carlo simulate the null (iid `N(0,1)` `z_k`) and a shifted
   alternative (`delta_target = 1.0` persistent downward shift injected at a
   random week) through the calibrated `(d, h*)` recursion, tens of thousands
   of simulated years each, and report the empirical `ARL_0`/`ARL_1` against
   the design targets (§4.3). This is the acceptance gate for shipping any
   change to `d`, `h*`, or `delta_target`.
2. **Historical replay (`validate/oos.py`).** Replay the CUSUM kill-switch
   against the REAL historical `weekly_net_return` series the strategy's own
   validation produced (the `N=8`-group, `k_test=2`, 28-path CPCV
   construction of `research/xsec/cpcv.py` / `SystemCPCV`, `n_obs ≈ 258`
   weeks (~5 years), `SYSTEM.md` §4.2) and report: did the monitor fire on
   any of the genuinely bad historical stretches (2022 bear incl. LUNA/FTX,
   `docs/EDGE.md` §c's per-regime breakdown), and did it stay quiet through
   the four positive regime windows? This is a sanity check, not a promotion
   gate — the strategy's OWN validation battery (purged-WF §SYSTEM.md §3,
   DSR §SYSTEM.md §5, CPCV/PBO §SYSTEM.md §4, per-regime `docs/EDGE.md` §c)
   already establishes the edge is real; OPS's replay only checks the monitor
   isn't wildly mis-calibrated against the strategy's actual realized
   volatility.
3. **Synthetic injected-event tests (`synth/generate.py`).** Seeded
   generators for: a persistent Sharpe-drift-to-zero regime break (tests
   Invariant CAL's detection side), a single-instrument position-delta
   breach (Invariant RECON), a mid-week delisting on a random long/short-leg
   name (Invariant DNEUT), a slow/expensive rebalance (Invariant SLA), and a
   multi-alarm simultaneous scenario (Invariant HALT-ONE — asserts all three
   trigger sources converge on one flatten call).
4. **Baseline contrast (`validate/baselines.py`).** A naive fixed-threshold
   kill-switch ("halt if any single week's Sharpe-equivalent return is below
   `X`") is implemented purely for contrast, to make explicit in the
   acceptance report how much faster/slower and how much noisier a
   non-sequential rule would be relative to the calibrated CUSUM — this
   baseline is never itself deployable, it exists only to make the CUSUM's
   value legible.

---

## 12. Module acceptance gates (CI)

### Reconciliation
- [ ] `epsilon_i` computed per §1.2 from AMBER `lot_size_i`/`target_notional_i`
      for every universe instrument; no flat global epsilon anywhere.
- [ ] Daily cadence enforced; a stale/failed reconciliation read produces
      `RECON_STALE`, never a silent `OK`.
- [ ] Invariant RECON test passes (synthetic breach escalates within one
      cycle, independent of CUSUM state).

### Delisting / renormalization
- [ ] §3.3's proportional-rescale formula implemented exactly (no equal-
      top-up substitution).
- [ ] Invariant DNEUT test passes to `1e-9` tolerance, both long-leg and
      short-leg delisting directions, and for sequential double-delisting.
- [ ] The degenerate-leg edge case (§3.4) forces `HALT`, never a division
      by a near-zero denominator.

### Kill-switch
- [ ] `mu_0`, `sigma_0` derived from the stated EVIDENCE numbers (`SR_0 =
      0.85`, annualized vol `0.28`), cited in code comments back to
      `docs/EDGE.md` §b (NOT `SYSTEM.md` — these two specific numbers are
      EDGE.md's; see the References-section note on which spec owns which
      number).
- [ ] `frac_pos` (0.85-1.00), CPCV mean OOS path Sharpe (≈0.91 vs naive
      ≈0.98), and PBO (0.357, range [0.31, 0.47]) referenced in code
      comments/docstrings are cited to `SYSTEM.md` §4.2-§4.4, never to
      `docs/EDGE.md` (which does not contain these figures) — a grep-style
      CI check that no code comment attributes these specific numbers to
      EDGE.md is a cheap regression guard against the mis-citation this spec
      itself once carried.
- [ ] `(d, h*)` calibration reproducible via `validate/arl.py`; the shipped
      `h* = 6.23` matches the value the Siegmund-formula solve produces for
      `ARL_0_target = 1000`, `d = 0.5`.
- [ ] Invariant CAL's Monte-Carlo test passes (empirical `ARL_0` within MC
      error of target; `ARL_1 << ARL_0` under the injected shift).
- [ ] Invariant MONO test passes (no single-observation self-heal from
      DEGRADE or HALT).

### SLA monitor
- [ ] Intraday polling cadence implemented (not a post-window check only).
- [ ] Invariant SLA test passes (severe breach halts before the next weekly
      CUSUM observation is even available).

### Flatten
- [ ] Invariant HALT-ONE test passes (all three trigger sources converge on
      the same code path; no independent implementations).
- [ ] No auto-restart from HALT under any test.

### Governance
- [ ] Invariant C test passes (`assert_no_red_in_amber` against OPS's
      `InstrumentSpec`-equivalent fields).
- [ ] No BLUE grid; no ML/fitted component anywhere in `ops/` (an AST
      import-graph check that `ops/` never imports a training/optimizer
      library, consistent with the sibling pillars' firewall tests).

### Cross-consistency (`core/protocols.py` / `core/types.py`)
- [ ] `core/types.py`'s `OpsDirective` matches THIS spec's §9 shape exactly
      (`state: Literal["RUN","DEGRADE","HALT"]`, `gross_multiplier: float`,
      `alarms: tuple[Alarm, ...]`, `as_of: int`) — a single source of truth,
      not two same-named classes. A CI check that `core.types.OpsDirective`
      and `ops.types.OpsDirective` (if the latter still exists as a separate
      definition post-reconciliation) are the SAME object, not merely
      structurally similar, is the regression guard against this collision
      recurring.
- [ ] The pre-pivot `KillState` enum is removed from `core/types.py` once (1)
      lands, with no remaining import of it anywhere in `ops/` or
      `core/protocols.py`.
- [ ] `core/protocols.py`'s `OpsEngine` Protocol is satisfied by
      `ops/brake/engine.py`'s concrete `OpsEngine.on_obs` implementation
      under `mypy --strict` (or equivalent), i.e. the Protocol conformance
      check is not merely "the method exists" but "the return type matches
      exactly" — this is the acceptance test that closes §9's disclosed
      collision, not just documents it.

---

## 13. Edge cases (explicit)

- **`sigma_0` re-estimation cadence vs the null's stability.** If `sigma_0`
  (§7 TEAL) were re-estimated from a short trailing window every week, a
  genuine regime break would itself widen the estimated `sigma_0`, which
  would widen the standardization in §4.2 and make `z_k` LOOK smaller even
  as the true drift is happening — a self-defeating adaptive null. OPS
  therefore re-estimates `sigma_0` on a SLOW cadence
  (`sigma_0_reestimation_weeks = 52`, RED, §7 — annual, the same order of
  cadence as a full validation re-run, not weekly) and the value in
  production is the one anchored to the full validated backtest sample
  (`docs/EDGE.md`), not a rolling short window — this is a deliberate design
  choice, not an oversight, and is why `sigma_0` is listed as TEAL-but-slow
  rather than a per-observation adaptive quantity.
- **The CUSUM kill-switch (§4) and a future REGIME drawdown-stop firing in
  the same episode.** §4.4 already flags, as a design caveat, that OPS's
  `DEGRADE` multiplier and a REGIME gross cut compose multiplicatively by
  construction, and that this composition has NOT itself been validated by
  REGIME §5.4's regime-conditional-ablation methodology. The edge case this
  raises concretely: suppose a genuine cross-sectional reversal week pushes
  `C_k` into `DEGRADE` (§4.4) in the SAME week REGIME's own detector (if
  built and composed) also cuts `g_t` for an unrelated market-state reason
  (e.g. a BTC-vol spike, `REGIME.md` §3). The combined gross cut is the
  PRODUCT of both multipliers — deeper than either alone — for a week whose
  true cause may be a single shared shock (the reversal and the vol spike are
  plausibly the same underlying event, not independent risks), which the
  multiplicative-composition design does not distinguish from two
  independent partial causes each warranting a partial cut. OPS's mitigation
  is only partial: the shared alarm/reason-chain logging (`Alarm` is a tuple,
  §9) records BOTH triggers' values so a post-hoc review can see whether a
  given week's realized gross cut was double-counted against a single
  shock, but OPS does not itself decorrelate or cap the combined multiplier
  — that would require a joint OPS×REGIME calibration this spec's remit does
  not cover (§14 Non-goals). This is logged here as a known, unresolved
  interaction, not silently assumed away.
- **A reconciliation breach and a CUSUM alarm fire in the same cycle.**
  §6's shared flatten routine is idempotent to being invoked with a
  multi-alarm reason chain — `Alarm` is a tuple, not a single value, exactly
  so simultaneous triggers are captured together rather than the second
  trigger being dropped because the state machine "already" transitioned to
  `HALT`.
- **The exchange's own position-report endpoint is itself the thing lying
  (a venue-side bug, not a local one).** Reconciliation as specified cannot
  distinguish "our bookkeeping is wrong" from "the venue's report is wrong" —
  it can only detect that the two disagree. This is a genuine limitation,
  not resolved by more math: the escalation policy (§1.4) treats either
  cause identically (both are a `RECON_BREACH`, both warrant a human look
  before resuming), because the CORRECT remedy differs by cause (fix our
  bookkeeping vs. open a support ticket with the venue) in a way no
  automated rule can safely decide.
- **Funding accrual on a stuck/unfillable position during an extreme
  illiquidity event (§3.4).** A position that cannot be closed continues to
  accrue funding exactly as COST's `FundingAccrual` (FACT, `COST.md` §5.5)
  already sums — OPS does not suspend or special-case funding accounting
  during a stuck-position episode; it is a real, ongoing cost that the
  `RECON_BREACH` escalation (§3.4) surfaces daily until resolved, precisely
  so it cannot be forgotten.

---

## 14. Non-goals (belong to sibling modules)

- **Deciding which instruments to hold, or their target weights.** That is
  `research/xsec/{signals,portfolio}.py` (the live rebalancer, not yet
  built per `EDGE.md` §e) and, eventually, a SYSTEM-equivalent orchestrator
  for the cross-sectional book. OPS only gates the gross multiplier and the
  binary run/halt state.
- **Modeling expected cost or expected funding.** That is `COST.md`; OPS
  consumes only COST's REALIZED facts (§5.2), never its forecasts (P5).
- **Execution tactics, order routing, or the `Delta` time-budget's
  derivation.** That is `EXECUTION.md` §2 (Almgren-Chriss trajectory,
  `Delta ≈ 1-4 hours`, AMBER); OPS only monitors the realized outcome (§5)
  against the budget that spec owns and OPS references, never re-derives.
- **Market-state / regime detection driving a gross multiplier from
  volatility, absorption, or turbulence measures.** That is `REGIME.md` §3-§4;
  OPS's CUSUM/ARL construction is the same class of tool REGIME independently
  instantiates against its own market-state null (§0), but OPS itself
  monitors STRATEGY health (Sharpe drift, reconciliation, SLA), never raw
  market conditions, and the two detectors' outputs compose only via the
  disjoint-trigger gross-multiplier discipline flagged (with its open
  caveat) in §4.4 and §13.
- **The deflated-Sharpe / CPCV / PBO validation battery that establishes the
  edge is real in the first place.** That is `SYSTEM.md` §4-§5 and
  `research/xsec/{cpcv,validate}.py`. OPS's `validate/oos.py` (§11.2)
  REUSES those historical paths for a sanity replay; it does not re-derive
  or duplicate the promotion gate.
- **Universe construction, breadth/shortability trade-offs, and the exact
  N=50 (vs. 60-75, vs. 100) sizing decision.** That is `UNIVERSE.md`'s
  derivation, referenced but not repeated in §0's Scope line above; OPS
  treats the live universe as a resolved AMBER input, whatever size
  UNIVERSE.md settles on.

---

## References

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2),
  100–115. — the CUSUM sequential change-detection construction (§4.1, §4.3)
  and its role as a complement to (not replacement for) single-point Shewhart
  control limits (§1.4's `RECON_BREACH`/`SLA_BREACH_SEVERE` rules).
- Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence Intervals*.
  Springer. — the closed-form exponential-family average-run-length
  approximation for a one-sided Gaussian CUSUM used in §4.3's `(d, h*)`
  calibration.
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5), 94–107. — the deflated/
  probabilistic-Sharpe machinery `SYSTEM.md` §5 builds (PSR, MinTRL), which
  OPS's kill-switch monitors a live sequential analogue of (the trailing
  realized Sharpe against the SAME validated null, §4.2-§4.3) rather than
  re-deriving; also the source of the `n_obs`/MinTRL track-record tension
  discussed in §4.3.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  — combinatorial purged cross-validation and the PBO (probability of
  backtest overfitting) estimator (`research/xsec/cpcv.py`), whose historical
  OOS paths §11.2's replay reuses.
- Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." *Annals of Statistics*, 17(3), 1217–1241. —
  the moving-block-bootstrap principle underlying the block-resampling used
  in the strategy's own validation (`docs/EDGE.md` §c, `SYSTEM.md` §7.4) and
  available to OPS's `validate/arl.py` if a non-Gaussian empirical null is
  substituted for the Monte-Carlo `N(0,1)` null in a future revision.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). "The
  Probability of Backtest Overfitting." *Journal of Computational Finance*.
  — the combinatorially symmetric cross-validation (CSCV) procedure and the
  PBO logit-of-relative-rank statistic (`research/xsec/cpcv.py::compute_pbo`,
  `SYSTEM.md` §4.4), cited here (§4.3) only to justify the conservatism of
  `delta_target = 1.0`; OPS does not compute or re-derive PBO itself.
- `docs/EDGE.md` — the empirical baseline this spec's honest-Sharpe / vol /
  maxDD / win-rate anchors are grounded against: honest net Sharpe
  **~0.8–1.0** (§4.2 `mu_0`), annualized vol **~28%** at 100% gross (§4.2
  `sigma_0`), maxDD **~25%**, weekly win-rate **~60%**, universe **~50–75**
  names (owned by `UNIVERSE.md`), lookback band **12–16d**. EDGE.md does
  **NOT** contain the CPCV path-Sharpe or `frac_pos`/PBO figures — those are
  cited to `SYSTEM.md` immediately below; do not re-cite them to this file.
- `docs/specs/SYSTEM.md` §4.2–§4.4 — the CPCV/PBO validation results this
  spec's kill-switch calibration is grounded against: mean OOS path Sharpe
  **≈0.91** vs naive single-path **≈0.98** (haircut **≈0.07**, `SYSTEM.md`
  §4.2), `frac_pos` **0.85–1.00** (§4.3, Invariant CPCV-STAT), and PBO
  **= 0.357** (partition-sensitivity range **[0.31, 0.47]**, §4.4) — cited
  here (P4, §4.3) rather than to `EDGE.md`, which does not contain these
  numbers. Also §5.2–§5.5 for the DSR track-record finding (PSR = 0.994,
  `n_obs ≈ 258` weeks `<` MinTRL `≈ 824` weeks, "insufficient track record,"
  §5.5) this spec's §4.3 track-record-tension note relies on.
- `docs/specs/REGIME.md` §4 (CUSUM/ARL construction, independently
  calibrated for market-state detection — the `k`/`d` notation
  reconciliation note, §4.3) and §5.2–§5.4 (the drawdown-stop first-passage
  taxonomy, the tested carry-book DD-stop finding — Sharpe 0.57→0.10, income
  12.9%→0.6%/yr, maxDD capped ~24% — and Invariant STOP-XFER's non-transfer
  rule), cited in §4.4's `DEGRADE`-composition caveat and §13's edge case.
- `docs/specs/COST.md` — the one cost/funding/carry truth OPS's SLA monitor
  (§5) and renormalization cost accounting (§3.3) both reuse rather than
  reimplement (COST P1); also §5.1-5.3's fee/half-spread/impact decomposition,
  cited in §5.2's `m_slip` decomposition-anchored bound.
- `docs/specs/EXECUTION.md` §7.1-7.2 — the realized `IS_total` implementation-
  shortfall series this spec's §5.2 slippage SLA compares against, and the
  (corrected) `OpsEngine.on_obs` ingestion path EXECUTION commits to using;
  cited in §5.2's ingestion-path correction and §9's `core/protocols.py`
  cross-consistency disclosure.
- `src/convexity/core/protocols.py` / `src/convexity/core/types.py` — the
  real, existing `OpsEngine` Protocol and its (currently stale, pre-pivot)
  `OpsDirective` dataclass, confirmed by direct inspection; cited in §9's
  disclosure of the naming/shape collision with this spec's own
  `OpsDirective` and the reconciliation this spec commits to.

---

## Risks / open items (flag before building)

- **`m_slip` (§5.2) has no EVIDENCE-cited, fitted-quantile derivation — but
  IS now bounded by a decomposition argument, not a bare convention.**
  `W_exec` is now resolved — it IS `EXECUTION.md` §2.1's `Delta` (~1-4 hours,
  AMBER), referenced directly rather than duplicated (§5.1, updated). `m_slip
  = 2×` still has no EVIDENCE number behind the specific multiple (unlike
  `k_lot`/`k_notional`/`h*`, which all trace to either a derivation or a
  cited constant) in the sense that no persisted cross-sectional dispersion
  of realized-vs-modeled slippage exists in this repo to fit it as a
  quantile — that residual gap is real and this spec does not fabricate the
  missing data. What §5.2 now adds is a **decomposition-anchored bound**:
  `COST.md` §5.1-5.3's round-trip friction is `2·(fee + half_spread_i/2 +
  impact_i)`, of which the 5bp `fee` term (10bp of the 13bp round-trip total)
  is an EXACT, contractual Binance VIP0-taker rate with no realized/modeled
  deviation possible in normal operation, leaving only ~3bp (the doubled
  half-spread+impact terms) as the genuinely stochastic, calibrated
  sub-component. §5.2 shows that even a severe 3-4x miss confined to that
  ~3bp stochastic sub-component alone only pushes total realized cost to
  ~1.5-1.7x the 13bp total — meaningfully LESS than the 2x-on-the-total bar
  `m_slip` sets. This is the same style of argument `LEVERAGE.md` uses for
  `G_max = 1.5` (a bound shown to sit safely outside the operating range,
  not derived to a single fitted point) and closes the "no reasoning at all"
  gap. It does NOT close the "no fitted quantile" gap — revisit once
  EXECUTION.md's realized `IS_total` series (§7.1) has enough live history to
  calibrate `m_slip` empirically (e.g. as a quantile of the realized/expected
  shortfall ratio) rather than relying on the decomposition bound alone.
- **The ARL calibration's Gaussian-null assumption (§4.3) is a
  simplification, and it is calibrated against a track record `SYSTEM.md`
  §5.5 itself flags as too short to fully certify (§4.3's track-record-tension
  note, restated here for visibility).** Weekly cross-sectional momentum
  returns are not perfectly iid Gaussian (the strategy's own validation notes
  fat tails are handled via PSR's skew/kurtosis correction, `SYSTEM.md`
  §5.3's worked `(gamma3, gamma4)` example); the Siegmund closed-form ARL
  approximation assumes Gaussian innovations. Compounding this, `mu_0`/
  `sigma_0` are drawn from the SAME `n_obs ≈ 258`-week sample `SYSTEM.md`
  §5.5 flags as short of `MinTRL ≈ 824` weeks — so the 1000-week (~19-year)
  `ARL_0_target` is a false-alarm-rate statement conditional on a null
  itself not yet independently certified over anything like that horizon.
  Two independent, additive future refinements: (1) substitute a
  block-bootstrap empirical null (Kunsch 1989) for the Monte-Carlo
  Gaussian calibration in `validate/arl.py`, without changing the CUSUM
  recursion itself; (2) re-derive `mu_0`/`sigma_0` from an enlarged sample as
  live `n_obs` accumulates toward `MinTRL` (the `sigma_0_reestimation_weeks`
  mechanism, §7, already provides the update path). Neither is a blocker for
  a v1 deployment — the Gaussian approximation is standard practice for a
  first CUSUM deployment, and the calibration is intentionally conservative
  (`ARL_0_target = 1000` weeks, slightly above the ≈1,018-week requirement
  derived from the stated 5%/yr budget, never below it) — but both should be
  treated as pending, not silently resolved.
- **PBO = 0.357 (`SYSTEM.md` §4.4, partition-sensitivity range
  `[0.31, 0.47]`) is moderate, not clean, and every RED constant that reads
  as "conservative because of the strategy's own uncertainty" (`SR_0`'s
  lower-bound choice §4.2, `delta_target = 1.0`'s coarseness §4.3) is
  conservative BECAUSE of this number, not independently of it.** If a
  future re-validation moves PBO meaningfully outside `[0.31, 0.47]` (either
  direction), the `delta_target`/`d` conservatism argument in §4.3 should be
  revisited — it is not a one-time citation, it is a standing dependency.
- **Live rebalancer does not exist yet (`docs/EDGE.md` §e).** OPS's
  `nautilus/` adapter (§8) is written against the SAME Nautilus fill/position
  event shapes the (not-yet-built) live cross-sectional strategy will emit;
  until that strategy exists, OPS's reconciliation and SLA streams (§2) have
  nothing live to read and can only be exercised against the historical
  replay (§11.2) and synthetic generators (§11.3, §13).
- **The OPS `DEGRADE` × REGIME gross-cut composition (§4.4, §13) has not
  been jointly validated.** Flagged in full in §4.4 and §13; restated here
  because it is exactly the kind of cross-pillar interaction this Risks
  section exists to surface: multiplicative composition is a design
  commitment inherited from the (currently stale, flagged above) intent in
  `ARCHITECTURE.md`, not yet an empirically ablated one for THIS pairing of
  triggers, even though `REGIME.md` §5.4's ablation methodology exists and
  could in principle be pointed at it.
- **`docs/ARCHITECTURE.md` itself needs a cross-sectional-pivot rewrite.**
  Flagged where first relevant (§0); restated here because it is a real
  actionability hazard for any reader who opens the composition root before
  the individual pillar specs — out of scope for this file to fix (this file
  is spec-writing for OPS only), but a standing risk until addressed.
- **`core/types.py`'s `OpsDirective` is stale and collides with this spec's
  (§9), and reconciling the two is unstarted build work.** Confirmed by
  direct inspection: `src/convexity/core/types.py` currently defines
  `OpsDirective` with the pre-pivot shape (`state: KillState`, `flatten`,
  `cancel_entries`, `reason_codes`, `engine_version`), and
  `src/convexity/core/protocols.py`'s `OpsEngine` Protocol — the same seam
  `EXECUTION.md` §7.2 names as its `IS_total` feedback path — currently
  type-checks against that stale shape via its `core.types` import. §9 states
  this spec's `OpsDirective` supersedes it and lists the exact reconciliation
  steps (update `core/types.py`, remove the now-dead `KillState` enum,
  re-export `Alarm`), but none of that reconciliation has been done — it is
  named, scoped, `core/types.py`-owned build work, out of scope for this file
  to perform (this file may only edit `docs/specs/OPS.md` and
  `docs/ARCHITECTURE.md`), and a real blocker for `core/protocols.py`'s
  `OpsEngine` Protocol being satisfiable by any implementation of this spec
  until it lands.
