# EXECUTION.md — build spec for the decision→fill execution engine

This file is the authoritative build spec. Build to the contracts and invariants
below. Where this file and your own instinct disagree, this file wins. Ask before
deviating from any **Invariant**.

This module is INDEPENDENT and composes with three sibling modules and the
NautilusTrader engine without importing their internals:

- BREAKOUT.md decides trade / no-trade and emits the signal, its side, its
  invalidation level, and a stop-distance estimate (entry gate).
- LEVERAGE.md converts a qualified signal into a target size / leverage subject
  to survival constraints (sizing).
- TREND.md decides hold / reduce / exit on an open position (exit + convex
  trail).
- **EXECUTION.md (this) decides HOW TO FILL** — it translates a fully-specified
  target `(instrument, side, size, urgency, invalidation_level)` into venue
  orders, cost-optimally, and rests the protective stop on the exchange.

It answers exactly one question, each time an upstream target changes (new
entry, a reduce, a full exit, a re-sized target):

> Given a target side/size on an instrument, the urgency the signal demands, and
> the price level at which the trade is invalid — what sequence of orders fills
> this at the lowest expected implementation shortfall, without ever chasing
> past the invalidation, and with a venue-resident reduce-only stop that fires
> while the bot is down?

Nothing else. It does not generate the signal, choose the size, decide when to
exit, or rebuild order plumbing. **NautilusTrader already owns** order submission
and lifecycle, position / account / portfolio state, RESTART RECONCILIATION,
data transport, backtest↔live parity, pre-trade risk checks, and funding
INGESTION. This module builds POLICY on top of those primitives — it never
re-implements them.

Scope: full cross-asset (Binance USDT-M perpetuals 24/7; Dukascopy-proxied
commodities, metals, and US equities), 1-minute bars, ~$1000 retail book. Retail
latency budget: SECONDS / bar-level, not microseconds. No live order book is
assumed — the fill model is built from spread, short-horizon volatility, and a
volume proxy that a retail feed actually delivers.

---

## 0. Governing principles (read before anything else)

**P1 — Execution chooses the TACTIC, never the trade.** The side, the size, and
the exit are decided upstream and are INPUTS. Execution decides only the order
type, the limit offset, the cancel/replace cadence, and the abort. It can fill
less than the target (partial, abort) or fill it more cheaply, but it can NEVER
fill a different side, exceed the target size, or move the invalidation. Any
code path where an execution decision changes *what* is traded rather than *how*
is the bug. (See Invariant A.)

**P2 — Cost is the only objective; cost is decomposable and measured against a
benchmark.** The single quantity execution minimizes is **expected
implementation shortfall** (IS) versus the **arrival price** (the mid at the bar
on which the target was issued). IS decomposes additively into spread cost,
market-impact cost, timing/delay cost, and adverse-selection cost (Section 4).
Every tactic is chosen by comparing these components, never by a discretionary
"this feels like a maker market" judgment. The decomposition is logged on every
fill so realized IS can be regressed against the model (Section 12).

**P3 — The maker/taker decision is a firewalled cost comparison, NOT a score.**
The decision to post passive vs cross the spread is

```
take  if   E[cost_taker]  <  E[cost_maker]
post  otherwise
```

where each side is an expected cost in basis points computed from a CAUSAL
fill-probability model (Section 3). It is a `min()` over a small, enumerable set
of tactics by expected cost — never a weighted multiplicative "execution
quality" blob (Invariant B). Each tactic's cost is computed independently so a
bad impact model cannot silently corrupt the spread term.

**P4 — Urgency comes from the SIGNAL, and breakout urgency is adversarial.** A
breakout fires BECAUSE price is already moving in the trade's direction.
Therefore the cost of resting passive is not the half-spread saved — it is the
**expected adverse drift while you wait**, i.e. the price runs away from a
buy-side rester. Passive execution on a breakout is implicitly short a
short-dated option on continuation. The placement problem is an optimal
limit-offset / cancel-replace problem whose chase is bounded by the trade's own
invalidation (P5). On a calm reduce or a mean-reversion-flavoured fill, passive
is cheap; on a live breakout it is often dominated by crossing.

**P5 — The chase can NEVER violate the trade's risk budget.** Every limit reprice
("chase") drifts the fill entry ADVERSELY: for a long breakout running up, the
entry rises AWAY from the (below) invalidation, **widening** the entry→stop
distance and shrinking the R-multiple (you pay more for the same target while
risking more per unit) — so each chase consumes part of the trade's R-budget. The **maximum chase distance** is
a hard fraction of the entry→invalidation distance (RED). If filling would
require crossing past the worst-allowable entry `worst_entry = entry0 + side·χ·|invalidation − entry0|`
(χ RED; the SAME canonical form as §5.3 — explicit `side`, absolute value, and
`entry0` not bare `entry` — so P5 and §5.3 are sign-identical), execution ABORTS the
entry rather than buy a trade whose stop is now too tight to be the stop
LEVERAGE/TREND sized for. Chasing a runaway breakout into a bad entry is the
single most expensive execution error; the abort is non-negotiable. (Invariant
D.)

**P6 — The protective stop lives on the VENUE, not in the bot.** The reduce-only
protective stop is placed as a native exchange order (Binance
`STOP_MARKET`/`STOP` reduce-only; broker-native stop where the proxy venue
supports it) the moment a position is opened, and it is reconciled with
Nautilus on every restart. The bot process is mortal — it crashes, the VM
reboots, the network drops — and a gap can happen while it is down. A stop that
lives only in Python protects nothing across an outage. The venue stop is the
real risk control; the in-process logic is a supervisor that keeps the venue
stop correct, not a substitute for it. (Invariant S.)

**P7 — Every live value is `F_t`-measurable; idempotency is mandatory.** Every
number the tactic uses (spread, short-horizon vol, volume proxy, fill
probability) is computable at or before the bar on which it acts. Future data
appears only in firewalled validation labels (realized adverse selection,
realized IS). Every order carries a deterministic `client_order_id` so a retry
after an ambiguous timeout can never double-fill (Section 7); the engine is a
state machine, not a fire-and-forget loop.

**P8 — Impact degrades to spread at $1k.** At a ~$1000 book the order is a
rounding error against any liquid venue's depth, so the square-root market-impact
term is ~0 and the dominant costs are spread, fees, and adverse selection. The
impact model (Section 4) MUST degrade continuously to spread-only as
participation → 0; it is retained because the same code runs on thin Dukascopy
proxies and because correctness at larger size is cheap to keep. A model that
charges meaningful impact on a $30 notional clip is miscalibrated. (Invariant
I.)

Forbidden in v1: any signal/sizing/exit decision inside execution (P1); a
multiplicative "execution score" instead of a cost `min()` (P3); chasing past
the invalidation budget (P5); an in-process-only stop with no venue order (P6);
non-idempotent order submission (P7); a live-order-book / L2-depth dependency
(retail feed has none — the fill model is spread/vol/volume-proxy based); any ML
or learned execution policy; per-instrument tuning of the RED constants on
realized P&L; re-implementing Nautilus order state, reconciliation, or risk
checks.

---

## 1. The tactic ladder (the whole engine in one decision)

For a target slice, the chosen tactic is the minimum-expected-cost element of a
small, fully-enumerable set, subject to hard gates that can only REMOVE tactics
or force the most aggressive one:

```
tactic* = argmin_{ t in ALLOWED }  E[cost_t]   (basis points vs arrival mid)

ALLOWED = {POST_ONLY_MAKER, LIMIT_IOC, MARKETABLE_LIMIT, MARKET} \ GATED_OUT
```

| Tactic            | What it is                                  | Wins when                                              | Cost dominated by         |
|-------------------|---------------------------------------------|--------------------------------------------------------|---------------------------|
| POST_ONLY_MAKER   | rest at/inside touch, post-only, reprice    | low urgency, wide spread, high maker fill-prob, low drift | spread saved − adverse sel |
| LIMIT_IOC         | marketable limit, immediate-or-cancel       | moderate urgency, want a price cap, partials OK        | half-spread + small slip  |
| MARKETABLE_LIMIT  | limit a few ticks through the touch         | high urgency, bounded slippage required                | half-spread + capped slip |
| MARKET            | cross immediately                           | abort-to-flat, exit on invalidation breach, halt risk  | half-spread + full slip   |

The **breakout-urgency** problem (Section 5) decides between POST_ONLY_MAKER and
the crossing tactics by trading the expected maker savings against the expected
adverse drift while resting. The **gates** (Section 6) force MARKET (or
MARKETABLE_LIMIT) regardless of cost when the action is a protective exit, a
halt/outage is imminent, or the chase budget is exhausted — because at that
point the cost of NOT filling dominates any spread saving.

No tactic is chosen by a score. The expected costs are computed independently
and compared by `min()` (Invariant B).

---

## 2. Architectural split

Five layers, kept independent for testability:

1. **Inputs layer:** receives the fully-specified `ExecTarget` (side, size,
   instrument, urgency, invalidation level, arrival mid, reduce-only flag) from
   upstream and the market microstate (spread, short-horizon vol, volume proxy,
   session/halt flags). Receives NO signal score, NO sizing rationale, NO exit
   reasoning — only the decided target. (Invariant A.)
2. **Microstate layer:** causal estimators of the inputs the cost model needs —
   spread, short-horizon volatility, a volume/turnover proxy, and the
   fill-probability model. Each is testable in isolation and `F_t`-measurable.
3. **Cost layer:** per-tactic expected-cost estimators (spread, impact,
   adverse-selection, delay) emitting basis points + decomposition. No tactic
   estimator may read another tactic's output.
4. **Policy layer:** the tactic `min()`, the breakout-urgency limit-placement /
   cancel-replace solver with the chase budget and abort, the partial-fill
   ladder, the protective-stop placement/reconciliation, and the order-type
   issuer with idempotency/timeouts/retries. Emits Nautilus order commands and a
   decomposable `ExecDecision`.
5. **Validation layer:** IS decomposition, fill-probability calibration,
   adverse-selection and slippage-vs-model regressions, tactic ablation. Never
   imported by the live layers (Invariant V).

Why: a bad fill-probability model can be isolated from a bad impact model; each
tactic ablated independently; the chosen tactic and the binding gate are always
explicit in the decision's reason codes.

The engine wraps a Nautilus `Strategy`/`ExecAlgorithm`: it SUBSCRIBES to
Nautilus order/position/fill events and EMITS `SubmitOrder` /
`ModifyOrder` / `CancelOrder` commands. It never tracks position or account
state itself — it reads Nautilus's `Portfolio`/`Cache`. (Invariant N.)

---

## 3. Directory layout

```text
execution/
+-- config/
|   +-- defaults.py          # PolicyDefaults: RED shared values (chase frac, fee tiers, timeouts)
|   +-- asset_class.py       # AssetClassPreset: validated per-class fill/cost params
|   +-- instrument.py        # InstrumentSpec: AMBER metadata (tick, lot, fee schedule, venue)
|   +-- venue.py             # VenueSpec: order types, reduce-only support, rate limits, stop semantics
|   +-- resolve.py           # single merge point (RED unreachable from instrument)
+-- microstate/             # MICROSTATE LAYER — causal, F_t-measurable
|   +-- spread.py            # rolling effective-spread estimator (Roll / quoted / proxy)
|   +-- vol.py               # short-horizon (seconds-to-minute) volatility for fill-prob & impact
|   +-- volume_proxy.py      # rolling notional volume / turnover; participation denominator
|   +-- fill_prob.py         # CAUSAL P(fill | offset, horizon) model — NO order book
+-- cost/                   # COST LAYER — each tactic's expected IS in bps, decomposed
|   +-- spread_cost.py       # half-spread + fee (maker rebate / taker fee) term
|   +-- impact.py            # square-root impact, degrades to spread-only at tiny size
|   +-- adverse_selection.py # expected fill-conditional adverse drift (the maker's short option)
|   +-- delay_cost.py        # expected drift cost of waiting (timing risk), breakout-aware
|   +-- tactic_cost.py       # assembles per-tactic E[cost] from the four components
+-- policy/                 # POLICY LAYER — the only place orders are emitted
|   +-- tactic_select.py     # argmin E[cost] over ALLOWED tactics; gates
|   +-- chase.py             # breakout limit-placement / cancel-replace ladder + chase budget + abort
|   +-- partial.py           # partial-fill handling + deterministic cancel/replace residual ladder
|   +-- protective_stop.py   # exchange-native reduce-only stop placement + Nautilus reconciliation
|   +-- order_issuer.py      # order-type construction, client_order_id, timeouts, retries, idempotency
|   +-- engine.py            # ExecutionEngine: ties microstate->cost->tactic->orders; reason codes
+-- nautilus/
|   +-- adapter.py           # thin bridge: Nautilus events <-> ExecDecision; reads Portfolio/Cache
|   +-- reconcile.py         # restart: reconcile venue stop + live working orders with intended state
+-- priors/
|   +-- exec_dna.py          # slow daily priors: per-instrument typical spread, fill-rate, fee tier
+-- validate/
|   +-- labels.py            # realized IS, realized adverse selection, realized slippage (firewalled)
|   +-- shortfall.py         # IS decomposition: realized vs expected, per component
|   +-- calibration.py       # fill-probability reliability curve; spread/vol/volume regressions
|   +-- baselines.py         # always-market, always-post, TWAP, naive-cross baselines
|   +-- ablation.py          # tactic on/off; which tactic / gate actually drives cost
|   +-- walkforward.py       # purged walk-forward of realized vs expected IS
|   +-- stats.py             # block-bootstrap CIs, deflated-Sharpe-style multiple-test control
|   +-- reports.py
+-- synth/
|   +-- generate.py          # adversarial fills: runaway breakout, spread blowout, partials, halt, restart
+-- types.py
+-- engine_api.py            # public entry point used by the trading layer
tests/
+-- ...                      # unit, firewall, parity, restart, idempotency, acceptance, ablation
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest, NautilusTrader (engine). No
ML frameworks. Numba allowed only for speed-sensitive rolling microstate
updates, and only after correctness tests exist.

---

## 4. Parameter governance

Every parameter is exactly one bucket.

- **TEAL — fitted per instrument from its own past data, at runtime, causally.**
  Examples: rolling effective spread distribution, short-horizon volatility
  scale, volume-proxy baseline and seasonality, the fitted fill-probability
  curve `P(fill | offset/σ, horizon, g)` per regime bucket, realized
  adverse-selection magnitude per regime, realized maker fill-rate, realized
  slippage-vs-model residual. **Online point-in-time rule (Invariant L):** the
  curves fitted at RUNTIME from the instrument's own realized passive fills
  (`fill_prob`, `adverse_selection`) update online ONLY with fills whose outcome
  horizon has fully elapsed and is embargoed, so a fill can never enter the curve
  it is being scored against — the live causal mirror of the validation layer's
  CPCV/embargo.
- **AMBER — supplied market metadata.** Tick size, lot/step size, contract
  multiplier, venue, maker/taker fee schedule and rebate, order types supported,
  reduce-only support, native stop semantics (trigger type: mark/last/index),
  rate limits, session calendar, min-notional, 24/7-vs-session flag.
- **RED — shared economic / modelling constants.** Max chase fraction χ of the
  entry→invalidation distance (e.g. 0.25–0.40), cancel/replace cadence and the
  reprice tick step, the per-tactic timeout (seconds / bars), max retry count,
  the urgency→tactic mapping thresholds, the adverse-selection-vs-spread
  break-even confidence, the maker fill-probability floor below which posting is
  abandoned (`floor_RED`, the LCB z-multiplier `z`, the calibration-error widening
  coefficient `κ` and threshold `calib_RED` — Section 6.3), the fill-prob
  event-count floors (`N_bucket`, `N_global`, the cold-start `N_red` — Section
  5.5 #2), the cold-start POST_ONLY exploration fraction (PATIENT/calm only), the
  no-fill-split abort floor `P(abort)_floor` (Section 5.1.2), the slippage budget
  per tactic, the marketable-limit through-tick count, the abort policy, and the
  rate-limit backoff schedule.
- **BLUE — finite validated grids** (optional, class-level only): small
  candidate sets for e.g. the fill-probability horizon, the reprice step in
  ticks, the chase fraction χ, searched offline against realized IS with
  promotion gates, NEVER per instrument on P&L.

> **Invariant C — RED unreachable from per-instrument config.** RED enters only
> via `resolve_config(defaults=...)`. Per-class overrides go through a separate
> `AssetClassPreset` argument and only when validated. If χ, a timeout, a retry
> count, the urgency thresholds, or the maker-prob floor appears as a
> per-instrument field, that is the bug.

> **Invariant T — no per-instrument tuning of RED on realized P&L.** χ, the
> timeouts, and the urgency thresholds are class-level constants or BLUE grids
> searched against realized IS (a cost metric), never fitted per symbol to
> maximize trade profit. Execution cost is the objective; trade P&L belongs to
> the signal.

---

## 5. The breakout-urgency problem (the heart of the module)

A breakout signal fires because price has begun to move. Resting a passive buy
below the touch on a market that is bidding up is implicitly writing a free
option on continuation: if the move is real you do not get filled until it has
already run; if it reverses you get filled adversely. This section formalizes
the placement, the cancel/replace chase, the hard chase budget tied to
invalidation, and the abort.

### 5.1 The cost comparison that drives passive-vs-cross

**One unit for every term: signed basis points of implementation shortfall
versus the arrival mid**, for the trade's own side (a cost > 0 is money lost vs
arrival mid; a saving is < 0). Mixing ticks, σ, and bps in one sum is a bug — so
the offset grid is defined in ticks/σ for the fill-probability model and
converted to bps the moment it enters a cost.

Symbols (all bps-vs-arrival-mid unless noted):

- `s` — effective HALF-spread in bps (`spread.py` returns half-spread; Section
  11). Crossing pays `s`; resting AT the touch saves `s`.
- `f_t` — taker fee in bps; `f_m` — maker fee in bps (negative if a rebate).
- `σ_h` — short-horizon vol in bps over fill horizon `h` (`vol.py`).
- `δ` — passive posting offset, measured in TICKS from the touch, INTO the book
  (δ ≥ 0; δ = 0 rests exactly at the touch, δ > 0 rests deeper/away from the
  market). The fill-probability grid is enumerated in `δ/σ_h`; `p_fill(δ, h)` is
  monotone DECREASING in δ (Section 3).
- `tick_bps` — one tick expressed in bps vs the mid the decision faces NOW =
  `10_000 · tick / mid` (AMBER tick, `mid` from CURRENT microstate; at the first
  step this is the arrival mid, at chase step k it is `mid_k` — see 5.2). The
  single conversion factor from the δ grid to bps. (This is the marginal-decision
  benchmark; the IS attribution stays vs the fixed `arrival_mid`, 5.2.)
- `g` — shared latent continuation-drift / regime input (5.1.1) that `p_fill`,
  `A`, and `D` ALL condition on, so they cannot be assembled into an incoherent
  joint.

**Price improvement from a passive fill.** Resting passive does NOT save a flat
`δ`. Resting AT the touch saves the full half-spread `s`; resting `δ` ticks
deeper than the touch captures `s` minus the distance you sit away from the
touch. Define the offset-from-touch in bps and the price-improvement (a saving,
so a negative cost) earned WHEN filled:

```
offset_from_touch_bps = δ · tick_bps                 (δ in ticks, ≥ 0)
PI = s − offset_from_touch_bps    = s − δ·tick_bps    (bps; the spread you keep)
```

At δ = 0 (rest at touch) `PI = s` (you save the whole half-spread); posting one
tick deeper saves `s − tick_bps`. `PI` can go negative for a deep post past the
opposite touch — that is correct (you would be paying up), and such offsets are
dominated and pruned. **This replaces the old `−δ` term, which conflated a
tick/σ offset with a bps saving and could be off by a full half-spread.**

**Taker cost (cross now):**

```
E[cost_taker] = s + f_t          (+ impact, ~0 at $1k — Section 4 / P8)
```

**Maker cost (post passive at offset δ).** Posting either fills within `h` (you
keep `PI`, pay `f_m`, but eat adverse selection `A`) or does NOT fill (the
no-fill branch — handled differently for an entry vs an exit, 5.1.2). Write the
fill branch first:

```
fill branch (per filled unit):  −PI + f_m + A(δ, g)
                              = −(s − δ·tick_bps) + f_m + A(δ, g)
```

where
- `−PI` is the captured price improvement as a (negative) cost; deeper posts
  (larger δ) capture less, so `−PI` rises with δ — the maker branch is NOT
  monotone "more passive is always cheaper";
- `A(δ, g) = adverse-selection cost` = the expected post-fill adverse mid move
  CONDITIONAL on the passive order being filled at offset δ in regime `g` (you
  are filled precisely when informed flow trades through you — Section 4 /
  `adverse_selection.py`). Increasing in `g`: high continuation drift means fills
  cluster on the wrong side of the move, so `A` is large on a live breakout and
  ~0 on a calm reduce. Also offset-dependent, and the sign must be reasoned
  CONDITIONAL ON FILL, not from "the market ran further" (which reads backwards —
  a deeper favorable-side post fills LESS often, not more). For a resting BUY, the
  limit fills when price ticks DOWN to its level (NOT on an up-reversal through
  it); the buyer is adversely selected when that down-flow is informed / the
  breakout is failing, so the **continued DOWN move after the fill** is the
  adverse excursion (a post-fill UP move is FAVORABLE to a filled buyer, not
  adverse). A post placed δ ticks deeper fills only after a LARGER down-move has
  reached it; conditional on a deeper post filling, that down-move was larger
  and — in a continuation/breakout context — more likely a genuine reversal that
  keeps running against the long. So `A` is non-decreasing in δ because, AMONG
  the fills that actually occur at a deeper offset, the continued-adverse
  (downward, for a buy) post-fill move is larger — NOT because a deeper order is
  more likely to be run over (a deeper favorable-side post in fact fills LESS often).

**Side-reflection rule (every maker-branch formula above is stated for a LONG,
`side = +1`; the SHORT mirror is obtained by the side multiplier).** All
price-direction reasoning in 5.1 — the offset `δ` "into the book," `p_fill`
monotonicity, `A`'s adverse direction, `PI`, and `D`'s drift sign — is worked for
a resting BUY. For `side = −1` every price-direction reflects through the side
multiplier and the magnitudes are identical:

- a resting SELL fills when price ticks UP to its level (not on a down-reversal
  through it), mirroring the BUY filling on a down-tick;
- the adverse post-fill excursion for a filled SELL is the continued UP move
  (informed up-flow / failing-breakdown), mirroring the BUY's continued-down
  adverse move; a post-fill DOWN move is FAVORABLE to a filled seller;
- "deeper / away from the market" (larger `δ`) for a SELL is a HIGHER price (above
  the touch), mirroring the BUY's lower price; `offset_from_touch_bps = δ·tick_bps`
  and `PI = s − δ·tick_bps` are unchanged in form;
- `g` is signed IN THE TRADE'S OWN SIDE (the EWMA of mid returns "in trade's
  side," 5.1.1), so `p_fill` non-increasing, `A` non-decreasing, and `D`
  non-decreasing in `g` hold IDENTICALLY for `side = −1` with no sign flips in the
  cost formulas themselves — the side multiplier is absorbed once, in the
  definition of `g` and in `worst_entry`/`side_pays_funding`, and the bps cost
  algebra is side-agnostic thereafter.

The only places a literal `side` multiplier appears are `worst_entry = entry0 +
side·χ·|inv − entry0|` (5.3) and `side_pays_funding` (5.4); every other
maker-branch quantity is written in side-agnostic bps and inherits its direction
from the side-signed `g`. Required test (synth, Section 14): the `side = −1`
maker cost on a price path equals the `side = +1` maker cost on the
sign-mirrored path, term by term (PI, `p_fill`, `A`, `D`, the no-fill split).

The full maker cost and the optimal offset are assembled in 5.1.2 (entry) and
5.1.3 (exit), because the no-fill branch terminates differently. In both:

```
δ* = argmin_{δ on tick grid} E[cost_maker(δ)]
maker/taker choice = min( E[cost_taker], E[cost_maker(δ*)] )
```

**`δ*` is a MYOPIC / greedy per-step optimum, NOT a global optimal-stopping
solution.** The one-shot `argmin` above prices a SINGLE post-or-cross decision at
the current step; it does NOT solve the full multi-step optimal-stopping /
stochastic-control problem over the cancel/replace chase in 5.2 (where each
later step's optimal action depends on the path of future microstates). The
chase is run as a sequence of these greedy per-step `min()`s — at each step
re-solve 5.1 with the CURRENT microstate (5.2) — which is a DELIBERATE,
retail-appropriate choice: the full stochastic-control solution is not worth the
compute or the model risk on a 2-core box, and the per-step greedy policy is
hard-bounded by the chase budget (Invariant D) so its worst case is contained.
An implementer must NOT read the one-shot `argmin` as globally optimal across the
ladder — it is locally optimal given the current state only.

**The greedy regret is not just "contained" — it is BOUNDED, and the abort bound
IS the regret bound.** Two facts make greedy near-optimal here and cap how far it
can sit above the omniscient one-shot optimum:

- **(a) Structural near-optimality (one-step-look-ahead is optimal for a monotone
  stopping problem).** Along an ADVERSE chase (the only case where greedy can be
  wrong — a favorable chase fills cheaply at the first step), the per-step cross
  cost `s + f_t` is non-decreasing as `mid_k` walks away from `arrival_mid`
  (spread does not tighten while a breakout runs, and the cross is taken at the
  worse `mid_k`), and the maker fill/no-fill split is MONOTONE in `g` (favorable
  `p_fill` non-increasing, abort mass non-decreasing — Invariant J). The
  decision "cross now vs post one more step then re-decide" is therefore a
  one-step-look-ahead (OSLA) rule on a problem whose continuation cost is monotone
  in the same `g`, which is the classic condition under which the OSLA / myopic
  stopping rule is OPTIMAL (a "monotone stopping problem"). So greedy is wrong
  only when the monotonicity is locally violated — a transient spread tightening
  or `g` dip mid-chase — never on a clean runaway.
- **(b) Worst-case regret is hard-capped by the chase geometry.** The policy can
  make a wrong post-vs-cross call on at most `max_steps` decisions; each
  mis-decision costs at most one `reprice_tick` of adverse price (in bps) plus one
  per-step delay `D`. Because NO price is ever placed beyond `worst_entry`
  (Invariant D), the accumulated adverse entry drift is hard-capped at
  `χ·|inv − entry0|`. Therefore the worst-case EXCESS expected IS of the greedy
  ladder over the omniscient one-shot optimum is bounded by

  ```
  regret_bps  ≤  10_000 · χ · |inv − entry0| / arrival_mid   (the worst_entry drift in bps)
              +  Σ_{k≤max_steps} D_k                          (the delay accumulated to worst_entry)
  ```

  i.e. **the abort bound IS the regret bound** — the most a greedy chase can lose
  versus an oracle that crossed at the perfect step is exactly the trade's own
  pre-committed chase budget converted to bps plus the delay accrued reaching it,
  both already hard-capped by Invariant D.
- **(c) Do NOT add a DP/backward-induction solver to close this gap.** The reason
  greedy is ACCEPTED is precisely that its regret is a known fraction of the
  trade's own R-budget (a quantity the trade already chose to risk), not an
  unbounded tail. A backward-induction / stochastic-control solver would trade
  that auditable, pre-committed bound for compute cost and model risk on a 2-core
  box with a coarse `p_fill` (Section 5.5) — a worse deal. An implementer who
  reaches for a DP solver here has mis-read the trade-off.

**The breakout asymmetry lives entirely in the shared driver `g`:** as `g`
rises, `p_fill` on the favorable side FALLS, `A` RISES, and `D` (delay, 5.1.2)
RISES — all three move together, so the `min()` flips to crossing exactly when
continuation risk dominates the spread saving, with no discretion. The
internal-consistency of these co-movements is Invariant J.

### 5.1.1 The shared latent-drift / regime input `g` (Invariant J)

`p_fill`, `A`, and `D` are NOT estimated in independent silos. On a breakout they
are driven by the SAME latent continuation drift: the harder price is running in
the trade's direction, the less a favorable-side passive fills, the more adverse
its fill is conditional on filling, and the more the mid runs away if it does
not fill. Estimating them independently would let the maker branch assemble a
physically impossible joint (high `p_fill` AND high `A` AND high `D` at once) and
mis-rank tactics.

Therefore all three condition on a single, causal, `F_t`-measurable regime
input `g`:

```
g = standardized short-horizon signed drift of the CURRENT mid (re-read per step)
  = ( EWMA(recent mid returns observed AT/THROUGH the decision bar, in trade's side) ) / σ_h
                                                          (dimensionless)
```

bucketed into the SAME regime grid the TEAL fits use (e.g. `g` quantile ×
spread-state). `g` is a microstate quantity (price-move based), NOT an upstream
edge forecast — it says "price is moving," which is exactly what a breakout's
own definition already asserts (P4), and carries no expected-return claim
(Invariant A is preserved: `g` is measurable drift, not predicted edge).

**`g` tracks the CURRENT mid, recomputed every chase step — it is NOT frozen at
arrival.** `g` feeds `p_fill`, `A`, and `D` in the step-`k` MARGINAL decision
(5.2), so it must reflect the recent return path observed UP TO step `k`, i.e.
the EWMA of mid returns through `mid_k`, recomputed from the re-read `MicroState`
at each step exactly as `mid_k` is. Anchoring `g` at `arrival_mid` and freezing
it would mis-price the marginal post-vs-cross decision after the mid has already
walked several ticks — the SAME stale-benchmark error 5.2 forbids for the
marginal `min()`, leaking into the maker trio. The IS ATTRIBUTION is still
benchmarked to the fixed `arrival_mid` (5.2); only the regime driver that the
forward decision conditions on tracks the current state. `MicroState.drift_g` is
re-read per step like `mid_k` (Invariant J pinning below).

This is a DELIBERATE, scoped breach of the silo rule in P3/Invariant B: the
firewall that keeps a bad impact model from corrupting the spread term STILL
holds (impact, spread, and fee remain independent), but the maker-branch trio
(`p_fill`, `A`, `D`) MUST share `g` because they share a physical cause. The
firewall protects independent failure modes; it must not force a known-coupled
trio to pretend independence.

> **Invariant J — the maker-branch trio shares one regime driver and respects
> sign relations.** `p_fill`, `A`, and `D` are all conditioned on the same
> causal `g`. Required test (internal consistency): on synthetic data with `g`
> swept from calm to runaway, the assembled maker cost shows favorable-side
> `p_fill` monotone NON-increasing, `A` monotone NON-decreasing, and `D`
> monotone NON-decreasing in `g`; an assembled maker quote that pairs high
> `p_fill` with high `A` and high `D` simultaneously fails the test. The
> spread/impact/fee firewall (Invariant B) is unaffected and separately tested.
> **`drift_g` is re-read per chase step, not fixed at arrival:** `MicroState.drift_g`
> is recomputed from the re-read microstate at every step `k` (the EWMA through
> `mid_k`), exactly like `mid_k`; the IS attribution stays vs `arrival_mid`.
> Required test: on a chase where the mid walks several ticks before a decision,
> the `g` fed to `p_fill`/`A`/`D` at step `k` equals the EWMA through `mid_k`, NOT
> the arrival-anchored value; freezing `g` at arrival is a detected failure.

### 5.1.2 The ENTRY maker cost — no-fill ends in ABORT, not a free cross

On an entry near the chase budget, the no-fill event does NOT terminate in a
guaranteed cross at `(D + s + f_t)`. It terminates in one of two ways:

- the cancel/replace chase repriced toward the market and got filled WITHIN the
  chase budget (5.2/5.3) — pay the delay-plus-cross cost;
- the chase reached `worst_entry` without filling — the entry ABORTS (5.3,
  Invariant D): zero execution cost is charged, and the foregone edge is
  deliberately firewalled OUT (Invariant A — execution does not price the trade's
  alpha).

So the entry maker cost splits the no-fill mass explicitly:

```
E[cost_maker_entry(δ)] =
      p_fill(δ, h, g)              · ( −PI(δ) + f_m + A(δ, g) )          # filled passive
  + ( 1 − p_fill(δ, h, g) ) · [ P(fill_within_budget | no-fill, δ, g) · ( D(δ, g) + s + f_t )
                                + P(abort | no-fill, δ, g)            · 0 ]    # firewalled
```

where
- `D(δ, g) = delay cost` = the expected adverse drift of the arrival mid, in bps,
  accumulated up to the cross, CONDITIONAL on the passive at offset δ NOT filling
  immediately, in regime `g` (`delay_cost.py`). The no-fill conditional drift
  DEPENDS on δ: a deeper post (larger δ) is less likely to fill and only fails to
  fill when the market is running, so its no-fill-conditional drift differs from a
  shallow post's. `D` is therefore a function of δ — see the contract change in
  Section 11.
- `P(fill_within_budget | no-fill, δ, g)` and its complement `P(abort | …)` split
  the no-fill mass into "chase fills before `worst_entry`" vs "chase exhausts the
  budget and aborts." Both are estimated from the same regime `g` and the chase
  geometry (distance from arrival to `worst_entry` in ticks, reprice step).

**Estimation discipline for the no-fill split (same rigor as `p_fill` / `A`).**
This split is a fitted realized-outcome quantity, so it gets the SAME estimation
discipline as the other TEAL fits — it is not exempt just because it is new:

- **Estimator and event-count floor (bucket → global → conservative prior).** The
  split is fitted per regime bucket (the `g`-quantile × spread-state grid, plus a
  chase-geometry bucket on `|worst_entry − arrival0|`-in-ticks) as the realized
  fraction of no-fill chase episodes that filled within budget vs aborted. A
  bucket whose observed chase-episode count is below the AMBER event-count floor
  falls back to the GLOBAL split curve; when even the global curve is below the
  RED terminal floor (see 5.5 #2's terminal-rung values, which this reuses), it
  falls back to a CONSERVATIVE class-level prior — never to an unfloored fit.
- **Online / embargo treatment (Invariant L).** Like `fill_prob` and
  `adverse_selection`, the split is fitted online from the instrument's OWN
  realized chase episodes under the point-in-time / embargo rule: a chase episode
  enters the fit ONLY after its outcome (filled-in-budget or aborted) is fully
  resolved and embargoed, so an in-flight chase never enters the split it is being
  scored against.
- **Conservative thin-data default (bias toward LESS-cheap posting).** When the
  split is unestimable (thin bucket AND thin global), bias `P(abort)` toward the
  value that makes POST_ONLY look LESS cheap — i.e. attribute MORE of the no-fill
  mass to `P(fill_within_budget)`, charging the guaranteed cross cost
  `(D + s + f_t)` to that mass rather than zeroing it as a free abort. Thin data
  must NOT manufacture phantom-cheap posting: the conservative prior sets
  `P(abort) → P(abort)_floor` (a low RED value), pushing the residual mass onto
  the cross. This is the same one-signed conservatism as the funding floor (5.4)
  and the maker-floor gate (6.3): thin/uncertain data can only make the maker
  branch look weakly MORE expensive, never less.
- **Live counterpart of the Section 12 abort check.** This fitted split is the
  LIVE counterpart of the validation layer's "realized abort rate" report
  (Section 12): the realized fraction of entry chases that actually aborted is
  regressed against the fitted `P(abort)`, closing the loop so the split is
  checked against realized aborts, not assumed.

**Why this matters (the bias it removes).** The old form charged the full
crossing cost `(D + s + f_t)` to the ENTIRE no-fill branch — including the mass
that actually aborts for FREE. That systematically OVER-costed the no-fill
branch, which made posting look more expensive than it is and biased the `min()`
toward crossing — but worse, because the abort mass is largest exactly on the
runaway breakouts (P4/P5 warn about), charging it a phantom cross cost
mis-prices precisely the regime the module exists to handle. Splitting it makes
`δ* = argmin_δ E[cost_maker_entry(δ)]` an honest objective.

**Acknowledged, bounded bias.** The abort branch is charged ZERO execution cost.
The trade's foregone edge (the profit you miss by not entering) is real but is
DELIBERATELY excluded — pricing it would import an expected-return term into
execution and violate Invariant A. The consequence is a known, bounded optimism
toward POST_ONLY on high-abort-probability breakouts: the maker branch looks
cheap partly because abort is "free" in this objective. This is acceptable
because (a) the chase budget / abort (Invariant D) is the real risk control, not
the cost ranking, and (b) the validation layer (Section 12) measures realized
abort rate and the IS of aborted-then-re-fired intents so the bias is observed,
not hidden.

**The optimism is BOUNDED, not merely "acknowledged."** The optimism toward
POST_ONLY equals the no-fill abort mass times the cross cost that was zeroed on
it:

```
optimism_bps  =  ( 1 − p_fill(δ,h,g) ) · P(abort | no-fill, δ, g) · ( s + f_t + D(δ,g) )
              ≤  P(abort) · ( s + f_t + D )       (the full per-step cross cost it omits)
```

so it is bounded by the abort probability times the guaranteed-cross cost
`(s + f_t + D)`, and it is MONOTONE INCREASING in `P(abort)`. The bias is
therefore largest exactly on runaway breakouts (where `P(abort)` is near 1) — but
on those, Invariant D (the hard `worst_entry` abort), NOT the cost ranking, is the
operative control, so an over-cheap maker quote cannot cause an over-chase. The
realized-abort-IS report (Section 12) is the observability guarantee that this
bound holds in practice: `optimism_bps` is computable per fill from the logged
`p_abort_if_nofill` and the cross-cost decomposition, and is reported against the
realized abort rate.

### 5.1.3 The EXIT maker cost — no-fill ends in a guaranteed cross

A reduce/exit NEVER aborts (P5/P6): getting flat dominates price. So the no-fill
branch terminates in a guaranteed cross at the budget, with no abort mass:

```
E[cost_maker_exit(δ)] =
      p_fill(δ, h, g) · ( −PI(δ) + f_m + A(δ, g) )
  + ( 1 − p_fill(δ, h, g) ) · ( D(δ, g) + s + f_t )
```

This is the only branch that charges the full cross to all no-fill mass, and it
is correct here precisely because an exit cannot abort. (For perps, a
funding-over-horizon term is added to both branches — see 5.4.)

### 5.2 The cancel/replace chase as a bounded sequence

If POST_ONLY_MAKER is chosen and does not fill within the per-step timeout, the
engine reprices toward the market by the RED reprice step and re-evaluates. This
is a deterministic ladder, not a loop chasing to infinity:

```
step k:  place at  p_k = arrival ± k · reprice_tick      (toward the market)
         wait timeout_step ; on no/partial fill -> step k+1
         BUDGET CHECK before every step (5.3) ; ABORT or CROSS if exhausted
```

Each step re-runs the 5.1 cost comparison with the CURRENT microstate, so a
spread blowout or a vol spike mid-chase can flip the decision to cross. The
cadence (timeout per step, reprice tick step, max steps) is RED.

**Two distinct benchmarks on a long chase — keep them separate.** As the chase
advances the mid moves, so step-k accounting uses TWO reference prices that must
not be conflated:

- **The per-step maker/taker `min()` at step k compares MARGINAL costs measured
  from the CURRENT mid (`mid_k`)** — the decision "post-here-now vs cross-now" is
  a forward choice about the next increment of execution, so `s`, the offset, the
  fill/no-fill split, `A`, and `D` are all evaluated against `mid_k`, the state
  the order actually faces now. Concretely, the conversion factor and offset at
  step k use the CURRENT mid: `tick_bps_k = 10_000·tick / mid_k` and
  `PI_k = s_k − δ·tick_bps_k`. Using the stale arrival mid here would mis-price
  the marginal decision after the mid has walked several ticks.
- **The IS decomposition stays benchmarked to the FIXED `arrival_mid`** (the mid
  at the bar the target was issued — the IS reference in `ExecTarget`). The mid
  drift already realized between arrival and step k is the DELAY cost `D`
  accumulated so far; it is booked into the timing/delay line of the IS
  decomposition against `arrival_mid`, NOT re-baselined away by switching the
  benchmark to `mid_k`. The decomposition's `realized_is_bps` is always vs
  `arrival_mid` (Section 11 contract).

So `tick_bps`/`PI` feeding the step-k DECISION use `mid_k`; the IS ATTRIBUTION
(how much of the total cost was spread vs delay vs adverse vs impact) is summed
vs the fixed `arrival_mid`. The realized mid walk from `arrival_mid` to `mid_k`
is not lost — it is exactly the delay term. Holding the IS benchmark fixed while
the marginal `min()` tracks the moving current state is what keeps long-chase
accounting from drifting (e.g. double-counting or erasing the delay already
paid). `arrival_mid` is fixed for the life of the intent; `mid_k` is AMBER and
re-read every step.

### 5.3 The chase budget tied to invalidation (Invariant D)

The chase is hard-bounded by the trade's own risk. Let `entry0` be the arrival
reference and `inv` the invalidation level from upstream. Define the **max chase
price**:

```
worst_entry = entry0 + side · χ · |inv − entry0|        (χ in [0.25,0.40], RED)
```

The engine MUST NOT place, reprice, or cross at a price beyond `worst_entry`. If
filling the residual would require a price beyond `worst_entry`:

- if the action is an **entry**: ABORT the unfilled residual, cancel working
  orders, emit `chase_budget_exhausted`. A breakout whose fill entry has already
  drifted χ of the original entry→invalidation distance is no longer the trade
  LEVERAGE sized — the adverse entry drift has consumed a fraction χ of its
  R-budget, so the entry→stop distance is now ~(1+χ)× the intended risk per unit
  and the R-multiple is degraded. Chasing further only buys progressively worse
  risk-reward; not trading is the correct, cheap outcome.
- if the action is a **reduce/exit**: do NOT abort — exits never abort (P5/P6).
  Cross immediately at MARKET; getting flat dominates price on a protective
  action.

```python
class ChaseLadder:
    def step(self, target: "ExecTarget", micro: "MicroState",
             working: "WorkingState", cfg: "ResolvedConfig"
             ) -> "ChaseAction":
        """Advance the cancel/replace chase by one deterministic step.

        Re-evaluates the 5.1 maker/taker cost with the CURRENT microstate, then:
        - if within chase budget and posting still wins: reprice toward market by
          reprice_tick and keep the post-only order working;
        - if posting no longer wins (drift/spread flipped the min()): cross with
          the chosen crossing tactic, capped at worst_entry;
        - if the next required price would breach worst_entry: for an ENTRY,
          ABORT the residual (`chase_budget_exhausted`); for a REDUCE/EXIT, cross
          at MARKET regardless.

        worst_entry = entry0 + side*chi*abs(inv-entry0) is recomputed every step
        and is a HARD bound: no order is ever placed beyond it on an entry. The
        ladder is finite (max_steps RED) and deterministic given inputs."""
```

> **Invariant D — the chase never violates the risk budget.** For an entry, no
> order is placed, repriced, or crossed at a price beyond
> `entry0 + side·χ·|inv−entry0|`; exhausting the budget aborts the residual with
> an explicit reason code. Required test: a synthetic runaway breakout that
> sprints toward `inv` produces an abort, never a fill beyond `worst_entry`, and
> never a chase that loops without terminating.

### 5.4 Funding / financing as an execution-timing cost (perps and MTF)

Nautilus INGESTS funding (Invariant N) — it accrues the realized funding cash
flow on the position. But WHETHER to cross now or wait is an execution decision,
and on a perpetual the choice to cross immediately BEFORE a funding stamp versus
waiting PAST it is a real, `F_t`-KNOWN cost: the funding rate and the next
funding timestamp are both observable now (Binance USDT-M posts funding every 8h
at known UTC stamps with a predicted rate). On a PATIENT clip whose chase can
straddle a funding stamp, this term can dominate the half-spread saving and must
enter the cost, or it is a silent gap in the IS decomposition for the primary
asset class.

Model it as an EXPECTED-FUNDING-OVER-HORIZON term added to the maker no-fill
branch (5.1.2/5.1.3) and, where the cross itself spans a stamp, to the taker
branch. For a position of the trade's side over the fill/chase horizon `h`:

```
fund_bps(h) = side_pays_funding · |predicted_funding_rate| · 10_000
              · (number of funding stamps in the HALF-OPEN window [now, now+h))
```

where `side_pays_funding ∈ {0, 1}` is 1 when the trade's side is the one that
PAYS funding over `h` (long pays when funding > 0; short pays when funding < 0),
and 0 when the side RECEIVES (a credit, which we conservatively floor at 0 rather
than book as a negative execution cost — receiving funding is a position
property, not an execution saving). The stamp count is deterministic from the
known schedule and `h`.

**Stamp-count boundary convention (off-by-one pinned).** The resting/chase window
is the HALF-OPEN interval `[now, now + h)`: a clip resting EXACTLY at a funding
stamp boundary does NOT pay that stamp (the stamp is counted only if it falls
strictly inside `[now, now+h)`, i.e. `now ≤ stamp < now+h`). This removes the
boundary ambiguity — a clip that crosses or aborts at the instant of the stamp is
treated as having exited before the charge, matching the venue convention that
the open position AT the stamp is what is charged. `secs_to_next_funding`
(`MicroState`) and `h` make the count `floor`-able deterministically:
`count = number of scheduled stamps t with now ≤ t < now+h`.

**Funding-floor conservatism is one-signed (proven direction, not asserted).**
Flooring the receiving-side credit at 0 (instead of booking a negative execution
cost) can only make the maker branch look WEAKLY MORE expensive than truth, never
cheaper: the floored quantity is a credit (`−|rate|·…`) that would have REDUCED
`E[cost_maker]`; setting it to 0 removes a saving, so the maker branch's cost is
≥ its true value. The error is therefore one-signed toward crossing / abstaining —
the SAFE direction — and can never bias the `min()` toward an unsafe
post-when-you-should-cross. (Same one-signed logic as the abort-bias bound, 5.1.2,
and the maker-floor gate, 6.3.) `fund_bps` is added to whichever branch holds the
position across the stamp:

- a PATIENT maker post that rests across a funding stamp, on the paying side,
  adds `fund_bps` to its no-fill / continued-rest cost — which can flip the
  `min()` toward crossing NOW (before the stamp) even when the spread saving
  favored posting;
- conversely, on the RECEIVING side near a stamp, waiting is (weakly) favored and
  the term is 0 by the floor above.

`predicted_funding_rate` and the next stamp are AMBER/microstate (read from the
venue feed via Nautilus); the per-clip horizon `h` is the chase window. For
commodities/metals/equities (Dukascopy proxies) there is no perpetual funding;
the analogous term is overnight/MTF financing only on holds that cross a session
boundary, which is a HOLD cost owned by LEVERAGE's carry term (Section 8.1 of
LEVERAGE.md), NOT an intra-bar execution-timing cost — so for those classes this
term is **scoped out** with that one-line justification. The class preset
(Section 13) sets `funding_aware = True` only for perps.

> **Invariant W — funding is an execution-timing cost on perps only.** For a
> funding-aware (perp) instrument, the expected-funding-over-horizon term enters
> the maker no-fill branch and the cross cost on the paying side, computed from
> the `F_t`-known predicted rate and next stamp; it is floored at 0 on the
> receiving side; it is identically 0 for non-perp classes (carry there belongs
> to LEVERAGE). Required test: a PATIENT perp clip resting across an imminent
> funding stamp on the paying side has its maker cost raised by `fund_bps` and
> the `min()` can flip to cross-now; the same clip on the receiving side, and any
> commodity/equity clip, has a 0 funding term. The receiving-side floor is
> one-signed-conservative (it can only raise the maker cost, never lower it), and
> the stamp count uses the half-open window `[now, now+h)` so a clip ending
> exactly at a stamp does NOT pay it. Required test: a clip whose horizon ends
> exactly at a stamp counts ZERO stamps; flooring the receiving credit never
> produces a maker cost below the unfloored truth.

### 5.5 The intrinsic coarseness of `p_fill` at 1-min with no order book

State it plainly: **queue position is the dominant driver of whether a passive
limit fills, and queue position is UNOBSERVABLE from 1-min bars with no order
book.** Two orders at the same price with the same microstate fill differently
depending on where they sit in a queue we cannot see. `p_fill(δ/σ, volume, g)` is
therefore a COARSE PROXY — a regime-conditional average fill rate, not a
per-order probability — and it carries irreducibly high variance. The maker
branch of the `min()` is consequently the HIGHEST-VARIANCE leg: the taker branch
is nearly deterministic (`s + f_t` are known), while the maker branch multiplies
the noisy `p_fill` through `−PI + f_m + A` and the no-fill split.

Three consequences, all enforced elsewhere and named here so the limitation is
explicit, not discovered:

1. the maker-floor gate (Section 6.3) gates on a LOWER-CONFIDENCE BOUND of
   `p_fill`, not the point estimate — so a high-mean but wide-CI or
   badly-calibrated `p_fill` is gated OUT automatically, because the maker leg is
   the highest-variance leg and the gate must respond to the VARIANCE, not just
   the mean;
2. `fill_prob.py` is an auditable closed form fitted per regime bucket with an
   event-count floor (Section 3 / Section 11), never a per-order classifier that
   would pretend to a precision the data cannot support. **The bucket → global
   recursion has a TERMINAL base case for the cold-start** (a symbol filled almost
   entirely by crossing tactics has almost no realized passive fills to fit
   `p_fill` at all, so even the global curve is starved). Three rungs:
   - **per-bucket:** use the bucket's own fit when its passive-fill event count
     ≥ `N_bucket` (AMBER per-bucket floor, derived as the count at which the
     fitted `p_fill`'s calibration-CI half-width falls below the maker-floor
     gate's tolerance — see 6.3; a practical RED starting value `N_bucket = 200`
     passive fills/bucket, a BLUE-tunable grid point);
   - **global:** when below `N_bucket`, fall back to the symbol's GLOBAL passive
     curve (all buckets pooled), valid when its event count ≥ `N_global` (RED,
     e.g. `N_global = 1000` passive fills);
   - **terminal RED floor → conservative class prior + maker GATED OFF.** When even
     the global curve's passive-fill count is below the RED floor `N_red` (e.g.
     `N_red = 50` passive fills — the cold-start), `p_fill` falls back to a
     CONSERVATIVE class-level prior curve (AMBER/preset, AMBER reliability) AND the
     maker-floor gate (6.3) treats the symbol as below floor BY DEFAULT, removing
     POST_ONLY until enough OWN passive fills accrue. Cold-start thus defaults to
     CROSSING — the lower-variance leg (5.5's whole point) — instead of trusting a
     prior curve the symbol has not earned.

   **Bootstrap problem (named, not hidden).** Own passive fills are generated ONLY
   by posting, so a pure crossing default never seeds the curve. The resolution: a
   SMALL, GATED exploration of POST_ONLY is permitted to seed the curve — but ONLY
   on PATIENT / calm flow (low `g`, wide spread, IMMEDIATE/URGENT/breakout flow
   NEVER posts during cold-start), capped at a RED exploration fraction of clips
   and removed the moment `N_red` is cleared. The exploration rides the
   lowest-adverse-selection regime so a mis-seeded `p_fill` cannot cost much
   (`A ≈ 0` on calm flow). An implementer who cannot afford even this gated
   exploration MAY instead accept a PERMANENT crossing default for that
   symbol/class, with the one-line justification that the maker savings on calm
   flow are not worth the cold-start adverse-selection risk — this is a valid,
   explicitly-allowed choice, not a defect;
3. calibration is a FIRST-CLASS validation (Section 12) AND a LIVE gate input: a
   reliability curve that sits off the diagonal widens the effective maker-floor
   `floor_eff` continuously (6.3 gate 3 `κ`-term), closing the loop between the
   Section 12 reliability curve and the live gate — the calibration error is read
   as an `F_t` quantity, not merely re-set into the floor manually/offline. The
   maker branch is the leg most likely to be wrong, so it is the leg most heavily
   validated AND the only leg whose gate self-tightens on its own miscalibration.

This is a retail-feed limitation, not a defect to engineer away: an L2 book would
shrink the variance but is explicitly out of scope (P-forbidden list). The
honest move is to acknowledge the coarseness and let the conservative floor +
calibration absorb it.

---

## 6. Tactic selection and the gates

```python
class TacticSelector:
    def select(self, target: "ExecTarget", micro: "MicroState",
               costs: "TacticCosts", cfg: "ResolvedConfig"
               ) -> tuple["Tactic", float, dict]:
        """Return (tactic, chosen_offset_ticks, diagnostics).

        Pure min() over ALLOWED tactic expected costs (Section 1), AFTER the
        gates below have removed tactics or forced the most aggressive one. The
        gates can only make execution MORE aggressive (cross sooner), never less.
        Diagnostics carry every tactic's E[cost] and its decomposition for
        logging; they never alter the choice (Invariant H)."""
```

Hard gates, evaluated before the `min()`:

1. **Protective-exit gate.** If `target.reduce_only and target.urgency ==
   IMMEDIATE` (an invalidation-breach exit handed down by TREND), force
   MARKETABLE_LIMIT or MARKET. Getting flat dominates price (P5/P6). Never post
   passive on a protective exit.
2. **Halt / outage / stale-book gate.** If the venue flags a halt, the feed is
   stale beyond a threshold, or a maintenance window is imminent, force the most
   aggressive ALLOWED tactic for any exit and DISABLE new entries.
3. **Maker-floor gate (keyed to a LOWER-CONFIDENCE BOUND, not the point
   estimate).** Remove POST_ONLY_MAKER from ALLOWED unless the LOWER confidence
   bound of `p_fill(δ*, h, g)` clears the RED maker-probability floor:

   ```
   keep POST_ONLY  iff  p_fill_LCB(δ*, h, g)  ≥  floor_eff
   p_fill_LCB  =  p_fill_point  −  z · ci_halfwidth(N_bucket, calib_error)
   ```

   The gate responds to the VARIANCE of the estimate, not just its mean: a
   high-mean but thin-bucket or badly-calibrated `p_fill` has a wide CI, so its
   LOWER bound falls below the floor and POST_ONLY is removed automatically. This
   is the surgical fix for the gate being qualitative — "conservative" is now a
   DERIVED number, not an asserted "set high." Two terms make it derived:

   - **CI width scales with the bucket event count.** `ci_halfwidth` is the
     calibration CI of the fitted `p_fill` for the regime bucket `δ*` lands in:
     it is `~ 1/√N_bucket` in the bucket's realized passive-fill count (the same
     count that drives the 5.5 #2 bucket → global → prior rungs), so a thin bucket
     widens the CI and gates POST_ONLY out without any manual floor change. When
     the bucket falls to the global / class-prior rung (5.5 #2), the CI is the
     wider global / prior CI; at the cold-start RED rung the LCB is below the floor
     by construction (maker GATED OFF by default).
   - **The floor widens with live calibration error (closes the Section 12 loop).**
     The effective floor `floor_eff = floor_RED + κ · max(0, calib_error −
     calib_RED)` rises when the Section 12 reliability-curve calibration error
     (Brier / off-diagonal calibration error for this instrument's recent passive
     fills) exceeds a RED threshold `calib_RED`. So a reliability curve sitting off
     the diagonal AUTOMATICALLY tightens the live gate — the calibration curve no
     longer "re-sets the floor" manually/offline (the old 5.5 #3 wording); it feeds
     the live gate as a continuously-read `F_t` quantity. `κ`, `floor_RED`, `z`,
     and `calib_RED` are RED constants.

   Removing POST_ONLY whenever the LOWER bound is weak biases toward the
   lower-variance crossing tactics, which is the correct response to a noisy or
   miscalibrated estimate — not a calibration failure but the gate doing its job.
4. **Rate-limit gate.** If the venue rate-limit budget is low, suppress
   cancel/replace churn: widen the reprice cadence and prefer a single
   marketable order over a long maker chase (Section 7).
5. **Min-notional / tick gate.** Snap the order price to the tick and the size to
   the lot/step; if the residual is below venue min-notional, do not emit a dust
   order — fold it into the next slice or mark the residual closed (Section 8).

Urgency is an upstream INPUT enum (the signal decides how urgent it is), mapped
to the cost weighting and the gates by RED thresholds:

```python
Urgency = Literal["PATIENT", "NORMAL", "URGENT", "IMMEDIATE"]
```

- PATIENT: low `D`; passive strongly preferred; long chase budget timeouts.
- NORMAL: balanced; the 5.1 `min()` decides.
- URGENT: `D` weighted up (breakout in progress); short chase; cross quickly.
- IMMEDIATE: gate 1 — cross now (protective exit / abort-to-flat).

---

## 7. Order-type selection, idempotency, timeouts, retries

```python
class OrderIssuer:
    def issue(self, tactic: "Tactic", target: "ExecTarget", price: float,
              qty: float, reduce_only: bool, cfg: "ResolvedConfig"
              ) -> "OrderCommand":
        """Construct the Nautilus order command for the chosen tactic with a
        DETERMINISTIC client_order_id and the right time-in-force / flags:

        - POST_ONLY_MAKER  -> LIMIT, post-only (rejected if it would take), GTC
                              within the step timeout, then cancel/replace.
        - LIMIT_IOC        -> LIMIT, immediate-or-cancel, price = touch (cap).
        - MARKETABLE_LIMIT -> LIMIT, IOC/FOK, price = touch +/- through_ticks
                              (RED) so slippage is bounded by construction.
        - MARKET           -> MARKET (only on protective exit / abort-to-flat /
                              halt gate; never the default entry tactic).

        client_order_id = deterministic hash of (intent_id, slice_idx, attempt)
        so a RETRY after an ambiguous timeout reuses the SAME id and the venue
        de-duplicates — a retry can NEVER double-fill (Invariant Q). Submission,
        acks, fills, and cancels are owned by Nautilus; this method only builds
        the command and hands it over."""
```

Idempotency / timeout / retry state machine (RED-parameterized):

- Each order has a per-tactic timeout. On timeout with no terminal event, query
  Nautilus order state (it is the source of truth) — never assume.
- An ambiguous submit (timeout before ack) is retried with the SAME
  `client_order_id`; the venue rejects the duplicate or returns the existing
  order. Bounded by `max_retries` (RED), then escalate to the gate-forced
  crossing tactic or abort.
- Exponential backoff on rate-limit rejects (`429`/throttle) per the RED backoff
  schedule; the rate-limit gate (Section 6) prevents churn from getting there.
- All lifecycle state (working, partially-filled, cancel-pending, done) is read
  from Nautilus, not duplicated here. The issuer is a command builder + a
  retry/timeout supervisor, not an order book. (Invariant N.)

> **Invariant Q — idempotent submission.** Every order carries a deterministic
> `client_order_id` derived from `(intent_id, slice_idx, attempt)`. A retry after
> an ambiguous timeout reuses the id, so no sequence of retries, reconnects, or
> restarts can produce more total filled quantity than the target. Required
> test: inject duplicate submits, dropped acks, and a mid-flight restart;
> assert filled qty ≤ target qty exactly.

---

## 8. Partial fills and the residual ladder

```python
class PartialFillManager:
    def on_fill(self, target: "ExecTarget", fills: "FillState",
                micro: "MicroState", cfg: "ResolvedConfig"
                ) -> "ResidualPlan":
        """Decide what to do with the UNFILLED residual after a partial fill.

        Deterministic ladder, re-evaluating the tactic on the residual with the
        CURRENT microstate (a partial that filled because the market moved
        through you is information — the residual may now warrant crossing):
        1. recompute remaining_qty = target_qty - filled_qty (signed, never
           overshoots target — Invariant A);
        2. re-run TacticSelector on remaining_qty with the chase budget reduced
           by progress already made toward worst_entry;
        3. if remaining_qty < venue min-notional, do NOT emit a dust order: mark
           the residual closed with `below_min_notional` (a few $ unfilled on a
           $1k book is not worth a taker fee + a second adverse-selection event);
        4. respect the SAME worst_entry bound and abort rule as Section 5.3.

        Returns a ResidualPlan with the residual qty, the next tactic, and
        reason codes. Never increases qty beyond target; never re-opens an
        aborted entry."""
```

Partial-fill discipline:

- The realized average fill price across partials is tracked for the IS
  decomposition (a chase that fills in worse and worse tranches is exactly the
  adverse-selection cost the validation layer measures).
- A reduce/exit residual is ALWAYS pursued to flat (crossing if needed); only an
  entry residual may be abandoned (budget abort or min-notional).
- The protective stop's quantity (Section 9) is reconciled to the ACTUAL filled
  quantity on every fill event — the venue stop must always cover exactly the
  open position, never the intended-but-unfilled size.

---

## 9. The exchange-native protective stop (non-negotiable)

This is the single most important operational guarantee in the module. The stop
that protects the trade lives on the VENUE so it fires while the bot is down.

```python
class ProtectiveStop:
    def place(self, position: "PositionState", invalidation: float,
              venue: "VenueSpec", cfg: "ResolvedConfig") -> "OrderCommand":
        """Place a VENUE-NATIVE reduce-only protective stop the instant a
        position is (partially) open.

        - reduce_only = True ALWAYS (a stop can only flatten, never flip/add).
        - qty = current FILLED position qty (reconciled on every fill event,
          Section 8); the venue stop covers exactly the open size.
        - trigger price = the invalidation level handed down by upstream (TREND
          owns where; execution owns that it RESTS on the venue).
        - order kind: STOP_MARKET where supported (guaranteed fire, accepts
          gap slippage) is the default for the protective stop, because a
          STOP_LIMIT can be jumped by a gap and leave the book naked. A
          STOP_LIMIT is allowed only where the class preset says the venue's
          STOP_MARKET is unsafe.
        - trigger reference: use the venue's mark/index trigger where available
          (AMBER) to avoid last-trade wick triggers; record which was used.
        - client_order_id uses the DEDICATED stop id scheme (Invariant Q-STOP),
          NOT the entry/exit slice scheme: stop_id = hash(intent_id, "STOP",
          amend_generation). amend_generation starts at 0 on first placement and
          is derivable from observed state (the current invalidation level / its
          generation index, carried in WorkingState), so reconciliation can
          recompute the CURRENT expected stop id from the current invalidation and
          adopt exactly that one. The "STOP" tag and a generation counter that is
          INDEPENDENT of slice_idx/attempt guarantee a stop id can never collide
          with an entry/exit slice's id (so an entry-slice retry cannot clobber a
          stop) — see Invariant Q-STOP.

        The stop is submitted to the venue and ALSO registered with Nautilus as a
        contingent/OCO-style order where the adapter supports it, so a fill on the
        stop cancels any working entry/exit and vice-versa. This method returns
        the command; Nautilus submits it."""

    def amend(self, position: "PositionState", new_invalidation: float,
              cfg: "ResolvedConfig") -> "OrderCommand":
        """Modify the resting venue stop when TREND moves the invalidation
        (e.g. the convex trail tightens). Cancel/replace atomically where the
        venue lacks modify; NEVER leave a window with no resting stop — place the
        new stop before cancelling the old, or use venue-native modify.

        Each amend INCREMENTS amend_generation, so the replacement stop gets a
        FRESH, non-colliding client_order_id = hash(intent_id, "STOP",
        amend_generation+1) that the venue accepts (reusing the same id would be
        deduped/rejected by the venue and re-open the very no-stop window amend
        exists to prevent). Because place-before-cancel can crash between the
        place and the cancel, an amend can transiently leave TWO venue stops (the
        new generation and the prior one); reconcile() (below) collapses this to
        exactly one via the multiple-stop case, and the prior-generation id is
        deterministically identifiable (lower generation) so the cleanup is
        unambiguous. amend_generation is derivable from observed state (the current
        invalidation generation in WorkingState), NOT an in-process counter, so it
        survives a restart."""
```

Reconciliation with Nautilus (restart- and gap-proof):

```python
class StopReconciler:
    def reconcile(self, intended: "IntendedState", nautilus_orders,
                  nautilus_position, cfg: "ResolvedConfig") -> "ReconcilePlan":
        """On startup (after Nautilus's own reconciliation completes) and
        periodically, verify the INVARIANT: every open position has exactly one
        resting venue reduce-only stop at the correct (current) invalidation,
        covering the full open qty.

        IDEMPOTENCY AS A PROOF, NOT AN ASSERTION. The target invariant is a PURE
        PREDICATE over OBSERVED state only — (Nautilus position qty, the SET of
        venue reduce-only stops for this intent):

            STOP_OK(pos, stops)  ⇔  |stops| == 1
                                    ∧ stops[0].qty     == pos.qty
                                    ∧ stops[0].trigger == current_invalidation
                                    ∧ stops[0].id      == hash(intent_id,"STOP",cur_gen)
                                    ∧ stops[0].reduce_only

        reconcile() is DEFINED as the projection that drives observed state onto
        STOP_OK: it computes the plan PURELY from (nautilus_position,
        nautilus_orders, current_invalidation) — NO hidden in-process counter,
        no memory of prior runs. Therefore a second run OBSERVES STOP_OK already
        true and emits an EMPTY plan: re-running is a no-op BY CONSTRUCTION, which
        is the proof (not the assertion) of idempotency. Plan is empty ⇔ predicate
        holds.

        Cases handled (each is one way the observed state violates STOP_OK; the
        plan is whatever projects it back onto STOP_OK):
        - position open, NO venue stop found (bot was down through a gap, or stop
          was never placed): place one IMMEDIATELY (this is the gap-protection
          point of P6/Invariant S).
        - position open, venue stop exists at STALE invalidation: amend to
          current (Section 9.amend), advancing amend_generation.
        - MULTIPLE venue reduce-only stops found for one position (the
          place-before-cancel amend window crashed, leaving the new and the prior
          generation both resting): cancel ALL BUT ONE — keep the stop whose
          (trigger, qty) match the CURRENT invalidation and position qty and whose
          id is the current generation hash(intent_id,"STOP",cur_gen); cancel every
          extra (the lower/stale generations). This is the cleanup that makes the
          amend's place-before-cancel safety leak-free across a restart. If NONE of
          the multiple stops matches the current invalidation, amend one to current
          and cancel the rest.
        - venue stop exists, NO position (the stop already fired while down, or
          position closed): cancel the orphan stop, mark the position closed,
          reconcile P&L from Nautilus's filled record.
        - qty mismatch (partial fill happened while down): amend the stop qty to
          the actual position.
        - STOP FIRED PARTIALLY in a thin gap (distinct sub-case of qty-mismatch):
          a STOP_MARKET that reduced only PART of the position in a thin/illiquid
          gap leaves the position partially reduced AND a stale-qty (or fully
          consumed) stop. Reconcile to the ACTUAL remaining position qty: if a
          stop still rests, amend its qty to the remaining position; if it was
          fully consumed by the partial fire, place a fresh stop for the remaining
          qty. The remaining position is read from Nautilus, never inferred from
          the stop's reported fill.

        This method NEVER second-guesses Nautilus's position/account state — it
        READS it as truth and only repairs the POLICY layer (the resting stop's
        existence, price, qty, and the single-stop invariant). It returns a plan of
        order commands; it does not itself submit. Idempotent BY CONSTRUCTION (see
        STOP_OK projection above): running it twice on a state that already
        satisfies STOP_OK emits an empty plan."""
```

> **Invariant S — the stop rests on the venue, always, and is reconciled.** At
> all times an open position has EXACTLY ONE venue-native reduce-only stop
> covering its full filled qty at the current invalidation — formalized as the
> pure predicate `STOP_OK(pos, stops)` over OBSERVED state (Section 9). There is
> NO state in which a position is open with the protective stop living only in the
> bot process, and NO state with two venue stops on one position. `reconcile()` is
> the projection onto `STOP_OK` computed from observed state only (no hidden
> counter), so re-running it on a satisfying state emits an EMPTY plan — that is
> the idempotency PROOF, not an assertion. Required tests: (a) open a position,
> kill the process, simulate an adverse gap through the invalidation, restart —
> the venue stop fired while down and reconciliation finds the position flat;
> (b) restart with an open position and NO venue stop present — reconciliation
> places one before doing anything else; (c) `amend` never leaves a no-stop
> window; (d) crash BETWEEN amend's place and cancel leaving TWO venue stops —
> reconciliation collapses to exactly one (the current-generation id at the
> current invalidation/qty) and cancels the stale-generation extra; (e) a
> partial STOP_MARKET fire in a thin gap — reconciliation re-covers the remaining
> position qty exactly; (f) running reconcile() twice in a row emits an empty plan
> on the second run (predicate already holds).

> **Invariant R — reduce-only on all protective orders.** Every stop and every
> exit residual order carries `reduce_only=True`; no execution command can
> increase or flip a position. Required test: a stop/exit command that would
> grow the position is rejected before submission.

---

## 10. Shared data objects (`types.py`)

```python
from dataclasses import dataclass
from typing import Literal, Mapping

Side = Literal[-1, 1]
Tactic = Literal["POST_ONLY_MAKER", "LIMIT_IOC", "MARKETABLE_LIMIT", "MARKET"]
Urgency = Literal["PATIENT", "NORMAL", "URGENT", "IMMEDIATE"]

@dataclass(frozen=True)
class ExecTarget:            # from upstream; the DECIDED trade. NO signal/sizing/exit rationale.
    intent_id: str           # stable id for this execution intent (idempotency root)
    instrument: str
    side: Side
    target_qty: float        # signed; |qty| is the size LEVERAGE decided. NEVER exceeded.
    arrival_mid: float       # mid at the bar the target was issued (IS benchmark)
    invalidation: float      # price level beyond which the trade is invalid (chase bound)
    urgency: Urgency
    reduce_only: bool        # True for exits/reduces (no abort; pursue to flat)
    # NOTE: deliberately NO breakout_score / expected_return / trend_confidence / kelly_fraction.

@dataclass(frozen=True)
class MicroState:            # all MEASURABLE, F_t-measurable
    mid: float
    half_spread_bps: float
    spread_rung: Literal["quoted","roll","corwin_schultz","class_proxy"]  # which estimator fired (5/Sec 11)
    sigma_horizon_bps: float # short-horizon vol over the fill horizon
    volume_proxy: float      # rolling notional / turnover (participation denominator)
    drift_g: float           # shared latent continuation-drift / regime input (Section 5.1.1)
    tick_bps: float          # one tick in bps vs mid = 10_000 * tick / mid (the delta->bps factor)
    fill_prob_curve: "object"# callable p_fill(offset_ticks, horizon, g), fitted causally (online, Invariant L)
    funding_aware: bool      # True only for perps (Section 5.4)
    predicted_funding_rate: float    # F_t-known next-stamp funding rate (perps; 0 otherwise)
    secs_to_next_funding: float      # F_t-known seconds to next funding stamp (perps; inf otherwise)
    halted: bool
    stale: bool
    rate_limit_ok: bool

@dataclass(frozen=True)
class TacticCosts:           # per-tactic expected IS in bps + decomposition (all signed bps vs arrival mid)
    by_tactic: Mapping[Tactic, float]      # E[cost] in bps
    spread_bps: Mapping[Tactic, float]
    impact_bps: Mapping[Tactic, float]
    adverse_sel_bps: Mapping[Tactic, float]
    delay_bps: Mapping[Tactic, float]
    funding_bps: Mapping[Tactic, float]    # perp funding-over-horizon on the paying side (Section 5.4)
    price_improve_bps: float               # PI(delta*) = s - delta*tick_bps captured if maker fills
    p_fill_at_offset: float                # p_fill(delta*, h, g) point estimate — the high-variance leg (5.5)
    p_fill_lcb_at_offset: float            # LOWER-confidence bound of p_fill(delta*); the maker-floor gate
                                           #   keys on THIS, not the point estimate (Section 6.3 gate 3)
    p_abort_if_nofill: float               # entry no-fill abort mass (Section 5.1.2); 0 for exits
    drift_g: float                         # the shared regime driver used by the maker trio (Invariant J)
    chosen_offset_ticks: float

@dataclass(frozen=True)
class WorkingState:          # READ from Nautilus; not owned here
    filled_qty: float
    avg_fill_price: float
    working_orders: tuple
    chase_step: int
    worst_entry: float       # the hard chase bound (Section 5.3)
    invalidation_generation: int  # increments on each invalidation change; derives the
                                  # stop's amend_generation -> stop client_order_id (Invariant Q-STOP)

@dataclass(frozen=True)
class ExecDecision:          # OUTPUT (the log record; commands go to Nautilus)
    intent_id: str
    instrument: str
    tactic: Tactic
    price: float
    qty: float
    reduce_only: bool
    client_order_id: str     # deterministic (Invariant Q)
    expected_cost_bps: float
    cost_decomposition: Mapping[str, float]  # spread/impact/adverse/delay
    binding_gate: Literal["none","protective_exit","halt","maker_floor","rate_limit","min_notional"]
    chase_step: int
    aborted: bool
    abort_reason: Literal["", "chase_budget_exhausted", "below_min_notional", "halt"]
    reason_codes: tuple[str, ...]
    engine_version: str

@dataclass(frozen=True)
class RealizedFill:          # VALIDATION-ONLY (firewalled); uses future bars
    intent_id: str
    realized_avg_price: float
    realized_is_bps: float           # vs arrival_mid
    realized_spread_bps: float
    realized_impact_bps: float
    realized_adverse_sel_bps: float  # post-fill adverse drift over the horizon
    realized_delay_bps: float
    realized_funding_bps: float      # funding paid/received over the clip horizon (perps; Section 5.4)
    realized_aborted: bool           # entry chase hit worst_entry and aborted (Section 5.1.2)
    filled_qty: float
    maker_filled_qty: float
    taker_filled_qty: float

@dataclass(frozen=True)
class RealizedStopFill:      # VALIDATION-ONLY (firewalled); the protective stop's realized cost
    intent_id: str
    trigger_price: float             # the invalidation level the venue stop rested at
    realized_fill_price: float       # where the STOP_MARKET actually filled
    gap_slippage_bps: float          # realized_fill_price vs trigger_price, signed, in bps
                                     # — the cost a STOP_MARKET pays when price gaps THROUGH the
                                     #   trigger (Section 9). SEPARATELY measured from realized_is_bps:
                                     #   it is the cost that feeds LEVERAGE's gap-survival ceiling, not
                                     #   part of entry/exit implementation shortfall.
    gapped_through: bool             # True if price jumped past the trigger without trading at it
    bot_was_down: bool               # True if the stop fired while the process was offline (Invariant S)
```

---

## 11. Module contracts (one public method each; signatures are the contract)

### microstate/spread.py
```python
class SpreadEstimator:
    def estimate(self, bars, idx: int, cfg) -> float:
        """Return the causal effective HALF-spread in bps at bar idx.

        Estimator ladder (first that is well-defined wins; the chosen rung is
        recorded as a reason code for the validation layer):
        1. QUOTED half-spread where the feed carries bid/ask.
        2. ROLL (1984): half-spread = sqrt(-cov(dp_t, dp_{t-1})) over the rolling
           window. DEGENERATE GUARD — Roll REQUIRES negative serial autocovariance
           of price changes; on trending 1-min crypto/commodity bars the
           autocovariance is FREQUENTLY POSITIVE, making sqrt(-cov) undefined
           (NaN). When cov >= 0 (or within an epsilon of 0), Roll is UNDEFINED and
           MUST NOT be used — fall through to rung 3.
        3. CORWIN-SCHULTZ (2012) high-low spread estimator (already cited in
           Section 18): a high-low-range based estimator that does NOT require
           negative serial covariance and is the documented fallback for trending
           bars. Where its own bounds produce a non-positive estimate, fall
           through to rung 4.
        4. CLASS-LEVEL PROXY FLOOR: the asset-class preset's typical half-spread
           (AMBER/preset), a finite conservative constant, NEVER NaN/inf/zero.

        The return is ALWAYS a finite positive half-spread in bps. F_t-measurable;
        never uses bar idx+1. The rung used is logged so the validation layer can
        report how often Roll degenerates per instrument/regime."""
```

### microstate/vol.py
```python
class HorizonVol:
    def estimate(self, bars, idx: int, horizon_s: float, cfg) -> float:
        """Return short-horizon (seconds-to-one-minute) volatility in bps over
        `horizon_s`, scaled by sqrt(horizon). Drives fill probability (how far
        price wanders to reach a resting offset) and the delay/adverse terms.
        Causal; reuses the family's robust vol estimator, NOT a fresh model."""
```

### microstate/volume_proxy.py
```python
class VolumeProxy:
    def estimate(self, bars, idx: int, window_s: float, cfg) -> float:
        """Return rolling expected NOTIONAL volume over the next window_s, from
        past bar volume and intraday seasonality (no live book). Used only as the
        participation denominator in the impact model and the rate-limit gate."""
```

### microstate/fill_prob.py
```python
class FillProbability:
    def curve(self, micro_inputs, regime_bucket: str, cfg):
        """Return a callable p_fill(offset_ticks, horizon, g) in [0,1]: the causal
        probability a passive limit at `offset_ticks` from the touch fills within
        `horizon`, with NO order book. Model: a monotone function of
        (offset / sigma_horizon), spread, the volume proxy, and the shared regime
        driver `g` (Section 5.1.1), fitted per regime bucket from the instrument's
        OWN realized passive fills (TEAL). Monotone DECREASING in offset and
        DECREASING in `g` on the trade's favorable side (a faster run away from a
        rester lowers its fill probability), INCREASING in sigma/volume. Sharing
        `g` with AdverseSelection and DelayCost is REQUIRED (Invariant J).
        Event-count floor — THREE-RUNG terminal fallback (Section 5.5 #2): a
        bucket with passive-fill count < N_bucket falls back to the symbol's GLOBAL
        passive curve; when even the global count < the RED cold-start floor N_red,
        fall back to a CONSERVATIVE class-level prior curve (AMBER) AND signal the
        caller that the maker-floor gate must treat the symbol as below floor by
        default (cold-start defaults to crossing). There is NO unfloored rung. The
        curve also returns its calibration-CI half-width (scaling ~1/sqrt of the
        active rung's event count) so the maker-floor gate can gate on the LOWER
        confidence bound of p_fill, not the point estimate (Section 6.3). Auditable
        closed form (e.g. logistic in offset/sigma and g), NOT a learned
        classifier.

        COARSENESS (Section 5.5): queue position is unobservable at 1-min with no
        order book, so this is a regime-conditional AVERAGE fill rate, not a
        per-order probability — the highest-variance input to the cost min().

        ONLINE-CAUSAL UPDATE (Invariant L): the TEAL fit is updated online from
        the instrument's own realized passive fills under a strict point-in-time
        rule — a fill is admitted to the curve ONLY after its outcome horizon has
        fully elapsed and is embargoed, so a fill can never enter the curve it is
        itself being scored against. The runtime curve at bar t uses only fills
        whose horizon closed at or before t."""
```

### cost/impact.py
```python
class ImpactModel:
    def cost_bps(self, qty: float, micro: MicroState, cfg) -> float:
        """Square-root market-impact cost in bps:
            impact_bps = eta * sigma_horizon_bps * sqrt( participation )
            participation = |qty| * mid / volume_proxy
        DEGRADES CONTINUOUSLY to ~0 as participation -> 0, so at a $1k clip the
        term vanishes and spread dominates (P8/Invariant I). eta is a RED/class
        constant. Linear small-size limit kept consistent (sqrt(x) -> x near 0 is
        NOT used; the model returns ~0, not a spurious linear charge)."""
```

### cost/adverse_selection.py
```python
class AdverseSelection:
    def cost_bps(self, target: ExecTarget, micro: MicroState, offset_ticks: float,
                 regime_bucket: str, g: float, cfg) -> float:
        """Expected adverse mid-move in bps CONDITIONAL on a passive fill at
        `offset_ticks` — the maker's short-option cost (`A(delta, g)` in Section
        5.1). Fitted per regime bucket from the instrument's realized
        post-passive-fill drift (TEAL). Conditions on the shared driver `g`
        (Section 5.1.1): LARGE when `g` is high (a live breakout — fills cluster
        on the wrong side of continuation), small for PATIENT/calm. Monotone
        NON-decreasing in `g` and in offset_ticks (a deeper fill only triggers
        when the market has run further). Shares `g` with p_fill and DelayCost so
        the maker branch cannot assemble an incoherent joint (Invariant J).

        ONLINE-CAUSAL UPDATE (Invariant L): like p_fill, the realized-drift fit is
        updated online from the instrument's own post-fill outcomes under the same
        point-in-time / embargo rule — a fill's realized adverse drift is admitted
        to the magnitude estimate ONLY after its horizon has fully elapsed, so the
        live magnitude at bar t never includes a fill still being scored."""
```

### cost/delay_cost.py
```python
class DelayCost:
    def cost_bps(self, target: ExecTarget, micro: MicroState, offset_ticks: float,
                 horizon_s: float, g: float, cfg) -> float:
        """Expected adverse drift of the arrival mid over `horizon_s` GIVEN a
        passive order at `offset_ticks` does NOT fill — the breakout runs away
        (`D(delta, g)` in Section 5.1.2). MUST take `offset_ticks`: the no-fill
        event and its conditional drift DEPEND on the posted offset (a deeper post
        is less likely to fill, and it fails to fill precisely when the market is
        running, so its no-fill-conditional drift differs from a shallow post's).
        Without this argument, delta* = argmin_delta E[cost_maker(delta)] would
        optimize only half the delta-dependence (the fill branch via PI and A) and
        miss the no-fill branch's, so the argmin would be inconsistent. Mirrors
        AdverseSelection, which already takes offset_ticks.

        Conditions on the shared regime driver `g` (Section 5.1.1) so D rises
        with the same latent continuation drift that lowers p_fill and raises A
        (Invariant J). Increasing in sigma_horizon and in `g`; ~0 for
        PATIENT/calm. Uses measurable drift `g` and the signal's urgency (an
        input) as drift sign/scale; NEVER an edge/return forecast (Invariant
        A)."""
```

### cost/tactic_cost.py
```python
class TacticCostModel:
    def costs(self, target: ExecTarget, micro: MicroState, cfg) -> TacticCosts:
        """Assemble E[cost] in bps for every ALLOWED tactic and the optimal
        passive offset delta* (Section 5.1). All terms are signed bps vs arrival
        mid (Section 5.1). Steps:
        - compute the shared regime driver `g` ONCE (Section 5.1.1) and pass it to
          p_fill, AdverseSelection, and DelayCost so the maker-branch trio is
          internally consistent (Invariant J);
        - price improvement PI(delta) = s - delta*tick_bps (the captured
          half-spread, NOT a flat -delta);
        - maker cost uses the ENTRY split (fill + no-fill*(P_fill_in_budget*cross
          + P_abort*0), Section 5.1.2) for an entry, and the EXIT form (no-fill ->
          guaranteed cross, Section 5.1.3) for a reduce/exit;
        - for funding-aware (perp) instruments add the funding-over-horizon term
          on the paying side (Section 5.4 / Invariant W);
        - delta* = argmin over the tick grid of the appropriate maker cost.

        The spread, impact, and fee components remain MUTUALLY independent and no
        component reads another TACTIC's cost (Invariant B); the maker-branch
        trio's shared `g` is the one scoped, tested coupling (Invariant J).
        Returns the full decomposition for logging."""
```

### policy/engine.py
```python
class ExecutionEngine:
    def on_target(self, target: ExecTarget, micro: MicroState, working: WorkingState,
                  cfg) -> ExecDecision:
        """The only entry point that turns a target into the next order command.
        Pipeline: build TacticCosts -> apply gates -> TacticSelector.select ->
        if POST_ONLY chosen, hand to ChaseLadder; on partial, PartialFillManager;
        always ensure ProtectiveStop is/where placed and reconciled. Emits exactly
        one ExecDecision with the chosen tactic, price, qty, deterministic
        client_order_id, expected-cost decomposition, binding gate, chase step,
        abort flag, and reason codes. Deterministic given (target, micro,
        working, cfg). Reads position/account state from Nautilus, never tracks
        it itself (Invariant N)."""
```

### nautilus/adapter.py
```python
class NautilusAdapter:
    def bind(self, strategy, cfg) -> None:
        """Wire the engine to a Nautilus Strategy/ExecAlgorithm: subscribe to
        order/fill/position events, expose the engine's OrderCommands as
        SubmitOrder/ModifyOrder/CancelOrder, and read Portfolio/Cache for
        position and account state. This is the ONLY file that touches Nautilus
        APIs; the cost/policy layers stay engine-agnostic and unit-testable
        without a running Nautilus node."""
```

---

## 12. Validation layer (PART OF THE MODEL)

Validation scores REALIZED vs EXPECTED execution cost, net of fees, on purged
walk-forward. It is never imported by the live layers (Invariant V). The
objective is execution COST (IS), never trade P&L (which belongs to the signal).

Required validations:

- **Implementation-shortfall decomposition (realized vs expected).** For every
  fill, decompose realized IS vs arrival mid into spread / impact /
  adverse-selection / delay and regress against the model's expected
  decomposition. Systematic under-prediction of any component ⇒ that estimator
  is miscalibrated. Report per tactic and per regime bucket.
- **Fill-probability calibration.** Reliability curve: bucket passive orders by
  predicted `p_fill`, plot realized fill rate. Must lie on the diagonal within
  CI; a curve that sits below the diagonal means posting is over-optimistic and
  the maker-floor gate is mis-set. Report Brier score / calibration error.
- **Maker fill-rate and adverse selection.** Realized maker fill rate vs target;
  realized post-fill adverse drift conditional on a maker fill (the `A` term) vs
  modeled. The core test of P4: does the breakout regime actually show the large
  adverse selection the model assumes?
- **Realized abort rate vs the fitted no-fill split.** Regress the realized
  fraction of entry chases that hit `worst_entry` and aborted against the fitted
  `P(abort | no-fill, δ, g)` (Section 5.1.2); report the realized `optimism_bps =
  P(abort)·(s+f_t+D)` it implies. This is the observability guarantee for the
  bounded abort-optimism bias (5.1.2) and the live check on the no-fill split's
  conservative thin-data default.
- **Maker-floor LCB / calibration-loop check.** Confirm the live maker-floor gate
  used the LOWER-confidence bound of `p_fill` (not the point estimate) and that
  the effective floor `floor_eff` widened when the reliability-curve calibration
  error exceeded `calib_RED` (Section 6.3). Test: a thin or off-diagonal bucket is
  gated out by the LCB even when its point `p_fill` is high.
- **Slippage-vs-model.** Realized slippage on crossing tactics vs the
  spread+impact prediction; confirm the impact term is ~0 at the $1k clip
  (Invariant I) and grows correctly on thin proxies / larger synthetic size.
- **Tactic ablation.** Turn each tactic off and measure the IS penalty: a tactic
  that never wins is redundant or miscalibrated; a tactic that always wins means
  the others are mis-priced. Report which tactic and which gate drives realized
  cost.
- **Required baselines** (identical fills/costs/benchmark): always-MARKET,
  always-POST-ONLY, naive-cross-on-signal, and a TWAP slice. The engine must
  beat always-MARKET on patient/calm flow and must NOT lose to always-MARKET on
  urgent breakout flow (where posting is the trap it avoids).
- **Stop-fill gap-slippage.** Measure realized `gap_slippage_bps` on every
  STOP_MARKET fire (`RealizedStopFill`), SEPARATELY from entry/exit IS. Report
  the distribution of slippage when price gaps THROUGH the trigger (and when the
  bot was down) per asset class. This is the cost that feeds back to LEVERAGE's
  gap-survival ceiling (LEVERAGE.md Section 8.3) — execution MEASURES it; LEVERAGE
  SIZES against it. A class whose realized gap-through slippage exceeds what
  LEVERAGE's gap model assumed is reported as a sizing-loop feedback.
- **Funding-timing check (perps).** For funding-aware instruments, regress
  realized funding cost on the modeled `fund_bps(h)` (Section 5.4); confirm
  PATIENT clips that crossed before a stamp on the paying side avoided funding the
  model said they would, and that the term is identically 0 on non-perp classes.
- **Maker-branch internal consistency (Invariant J).** On the swept-`g` synthetic
  (5.1.1), assert the assembled maker cost respects the sign relations (`p_fill`
  down, `A` up, `D` up as `g` rises); flag any regime bucket where the live fits
  violate them.
- **Online-update causality (Invariant L).** Assert that the live `fill_prob` and
  `adverse_selection` fits at bar t contain only fills whose outcome horizon
  closed at or before t (point-in-time / embargoed); a deliberately-injected
  not-yet-resolved fill must NOT appear in the curve being used to score it.
- **Restart / reconciliation correctness.** Replay outage-through-a-gap and
  restart-with-no-stop scenarios; assert Invariant S holds and no double-fill
  occurs (Invariant Q).

Required statistics:

- **Block-bootstrap CIs** on IS and on each cost component (blocks respect
  intraday volume/vol clustering, so adjacent fills are not treated as
  independent).
- **Sampled combinatorial purged CV (CPCV)** over time folds, with an embargo
  spanning the fill horizon, so a fill's realized-cost label cannot leak across
  the train/test boundary. Purged walk-forward as the primary split.
- **Multiple-test / deflated-Sharpe-style control.** When comparing the engine
  against N baselines and across BLUE grid candidates, deflate the apparent
  IS-improvement Sharpe for the number of trials (deflated / probabilistic
  Sharpe ratio). A χ-grid winner must clear the deflated threshold, not the raw
  one.
- **Drop-top-N robustness.** Re-compute the IS advantage after dropping the N
  best-improved fills (and the N worst). If the edge over baselines collapses
  when a handful of lucky fills are removed, the tactic advantage is noise, not
  policy.

```python
class ShortfallDecomposer:
    def decompose(self, decisions, realized_fills, cfg) -> "ShortfallReport":
        """Match each ExecDecision to its RealizedFill and produce the
        per-component realized-vs-expected IS table, per tactic and per regime
        bucket, with block-bootstrap CIs. Validation-only; uses future bars for
        the adverse-selection horizon (firewalled, Invariant V)."""
```

---

## 13. Asset-class presets (cross-asset via metadata, not forked logic)

Per-class presets (validated, class-level only — never per instrument). Caps and
starting points; the live tactic is decided by the cost `min()` and the gates:

- **Binance USDT-M perps (crypto):** 24/7, no session gaps but violent intrabar
  moves; maker rebate / low maker fee makes POST_ONLY attractive on PATIENT
  flow, but breakout adverse-selection is large → URGENT crosses. Native
  `STOP_MARKET` reduce-only with MARK-price trigger (avoid last-trade wick
  triggers). Tight tick/step; min-notional small. Rate limits matter — cap
  cancel/replace churn.
- **Commodities / metals (Dukascopy proxy):** session boundaries → real
  open/weekend gaps the venue stop must survive (STOP_MARKET, accept gap
  slippage). Wider, more variable spreads → spread term dominates; longer fill
  horizons. Thinner → impact term non-trivial at larger size (keep it).
- **US equities (Dukascopy proxy):** RTH/ETH split; earnings/overnight gaps →
  STOP_MARKET; lot = shares; fee schedule differs. Wider spreads off-RTH raise
  the maker-floor gate.
- **FX-like proxies:** weekend/news gaps; low spread, modest impact; passive
  often wins on calm flow.

Asset-class differences flow through AMBER metadata (tick, lot, fees, stop
semantics, session calendar, 24/7 flag) and TEAL fits (each instrument's own
spread, fill-prob curve, adverse-selection magnitude) — NOT through forked
tactic logic. The same `ExecutionEngine` runs on all of them.

---

## 14. Synthetic data requirements (`synth/generate.py`)

Generate adversarial fill scenarios so the module is testable with no external
data and no live venue:

- **Calm wide-spread market** (POST_ONLY should win; maker fill-rate high;
  adverse selection ~0).
- **Runaway breakout** (price sprints toward the invalidation): the chase must
  reprice, flip to crossing when `D` dominates, and ABORT at `worst_entry`
  (Invariant D) — never fill beyond it. The ENTRY maker cost must split the
  no-fill branch into chase-fills-in-budget vs aborts-for-free (Section 5.1.2),
  so posting is not phantom-charged the cross on the abort mass.
- **Swept-`g` regime sweep** (calm → runaway continuation drift): assert the
  maker-branch trio respects sign relations — `p_fill` non-increasing, `A`
  non-decreasing, `D` non-decreasing in `g` (Invariant J). Also assert `g` is
  recomputed per chase step from `mid_k` (not frozen at arrival) and the IS
  attribution stays vs `arrival_mid`.
- **Short-side mirror** (`side = −1` on a sign-mirrored price path): assert the
  maker cost equals the `side = +1` cost on the mirrored path term-by-term
  (PI, `p_fill`, `A`, `D`, no-fill split) — the side-reflection rule (5.1).
- **Cold-start with no own passive fills** (symbol filled almost entirely by
  crossing): `p_fill` falls through bucket → global → conservative class prior;
  the maker-floor gate treats the symbol as below floor BY DEFAULT (POST_ONLY
  removed); the gated PATIENT/calm exploration seeds the curve and is removed once
  `N_red` is cleared; URGENT/breakout flow never posts during cold-start
  (Section 5.5 #2).
- **Trending bars where Roll degenerates** (positive serial autocovariance): the
  spread estimator must NOT return NaN/inf — it falls through Roll to
  Corwin-Schultz or the class proxy floor and logs the rung (Section 5 / Section
  11).
- **Perp funding stamp imminent** (PATIENT clip, paying side): the funding term
  raises the maker cost and the `min()` can flip to cross-now; on the receiving
  side and on non-perp classes the funding term is 0 (Invariant W).
- **Not-yet-resolved fill injected into the runtime fit** (online causality): the
  live `fill_prob` / `adverse_selection` curve at bar t must exclude it
  (Invariant L).
- **Spread / vol blowout mid-chase** (the `min()` flips from post to cross
  partway through the ladder).
- **Thin book / low volume** (impact term becomes non-trivial at larger size;
  ~0 at the $1k clip — Invariant I).
- **Heavy partial fills** (residual ladder, min-notional dust handling, average
  fill price tracked).
- **Maker order that never fills within horizon** (maker-floor gate removes
  POST_ONLY; delayed cross is correctly priced as a loss).
- **Halt / stale feed** (halt gate forces aggressive exit, disables new
  entries).
- **Outage through an adverse gap, then restart** (venue STOP_MARKET fired while
  down; reconciliation finds the position flat — Invariant S).
- **Restart with an open position and NO venue stop** (reconciliation places one
  before anything else — Invariant S).
- **Crash inside amend's place-before-cancel window** (TWO venue stops on one
  position): reconciliation collapses to exactly one (current-generation id at the
  current invalidation/qty), cancels the stale-generation extra (Invariant S case
  (d) / Invariant Q-STOP).
- **Partial STOP_MARKET fire in a thin gap** (position partially reduced, stale-qty
  or consumed stop): reconciliation re-covers the exact remaining qty (Invariant S
  case (e)).
- **N successive invalidation amends** (the stop id scheme): each amend produces a
  distinct accepted stop id `hash(intent_id,"STOP",gen)`, reconciliation adopts the
  current-generation id, no stop id collides with any entry/exit slice id
  (Invariant Q-STOP).
- **reconcile() run twice on a satisfying state**: the second run emits an EMPTY
  plan (idempotency-by-construction proof — Invariant S).
- **Duplicate submits / dropped acks / mid-flight restart** (idempotency: filled
  qty ≤ target qty exactly — Invariant Q).
- **Rate-limit pressure** (cancel/replace churn suppressed; single marketable
  order preferred).

Behavioral assertions: the chosen tactic always equals `argmin E[cost]` over the
non-gated set; no order ever prices beyond `worst_entry` on an entry; the venue
stop always covers the exact open qty; retries never overshoot target qty;
diagnostics never change the decision; batch replay and a one-event-at-a-time
stream produce identical decisions (Invariant E).

---

## 15. Module acceptance gates (CI)

### Microstate
Pass when: spread, horizon-vol, and volume-proxy estimators are causal
(`F_t`-measurable, a √horizon test on vol), NaN/inf-free on adversarial data, and
reuse the family's vol estimator rather than a fresh model. The spread estimator
ladder (quoted → Roll → Corwin-Schultz → class proxy) returns a finite positive
half-spread on TRENDING synthetic bars where Roll's serial autocovariance is
positive (sqrt(-cov) undefined) — a NaN/inf acceptance test names this failure
mode explicitly (Section 5) and the chosen rung is logged. The shared regime
driver `g` and `tick_bps` are causal.

### Fill probability
Pass when: `p_fill` is monotone decreasing in offset, increasing in
sigma/volume, bounded [0,1], event-count-floored through the THREE-RUNG terminal
fallback (bucket → global → conservative class prior with maker GATED OFF at the
RED cold-start floor `N_red`; Section 5.5 #2), and is an auditable closed form (no
learned classifier). The cold-start exploration seeds only on PATIENT/calm flow
and never on URGENT/breakout flow.

### Cost components
Pass when: every term is signed bps vs arrival mid (one unit, Section 5.1);
spread/impact/fee compute independently (Invariant B) while the maker-branch trio
(`p_fill`, `A`, `D`) share the one regime driver `g` and respect sign relations
as `g` rises (Invariant J); price improvement is `PI = s − δ·tick_bps` (NOT a
flat `−δ`) so the maker branch cannot be off by a half-spread; `DelayCost` takes
`offset_ticks` so `δ*` optimizes the full δ-dependence (Section 5.1.2); the ENTRY
maker cost splits no-fill into chase-fill vs free-abort and the abort opportunity
cost is excluded by design (firewall, Section 5.1.2); the EXIT maker cost charges
the full cross to all no-fill mass (Section 5.1.3); impact → ~0 as participation
→ 0 (Invariant I); adverse-selection and delay rise with `g` and vanish in
PATIENT/calm; the perp funding term enters the paying side and is 0 elsewhere
(Invariant W); fees use the AMBER schedule.

### Tactic selection + gates
Pass when: chosen tactic = `argmin E[cost]` over ALLOWED (Invariant B); each gate
can only make execution more aggressive; protective-exit and halt gates force
crossing; the maker-floor gate removes POST_ONLY when the LOWER-confidence bound
`p_fill_LCB(δ*) < floor_eff` (gating on variance, not the point estimate), with
the CI widening as the bucket event count falls and `floor_eff` widening as the
Section 12 calibration error exceeds `calib_RED` (Section 6.3);
diagnostics don't alter the choice (Invariant H).

### Chase ladder + abort
Pass when: the ladder is finite and deterministic; `worst_entry =
entry0 + side·χ·|inv−entry0|` is a hard bound never breached on an entry; budget
exhaustion ABORTS an entry residual and CROSSES a reduce/exit; a mid-chase
microstate flip switches post→cross (Invariant D).

### Partial fills
Pass when: residual qty never exceeds target (Invariant A); sub-min-notional
residual is closed with `below_min_notional`, not emitted as dust; reduce/exit
residuals pursue to flat; the stop qty re-reconciles to actual filled qty on
every fill.

### Protective stop + reconciliation
Pass when: a venue-native reduce-only stop exists for every open position at the
current invalidation covering full filled qty — formalized as the `STOP_OK`
predicate (Invariant S); `amend` never leaves a no-stop window and advances
`amend_generation` for a fresh accepted stop id; reconciliation handles no-stop,
stale-stop, orphan-stop, qty-mismatch, MULTIPLE-STOP (place-before-cancel crash),
and PARTIAL-FIRE cases by projecting observed state onto `STOP_OK`, so a second
run emits an empty plan (idempotency-by-construction, not assertion); the stop id
`hash(intent_id,"STOP",amend_generation)` is disjoint from the entry/exit slice id
scheme (Invariant Q-STOP); outage-through-gap test shows the venue stop fired
while down; all protective orders are reduce-only (Invariant R).

### Order issuer / idempotency
Pass when: each tactic maps to the right Nautilus order type/TIF/flags;
`client_order_id` is deterministic in `(intent_id, slice_idx, attempt)`; duplicate
submits, dropped acks, and a mid-flight restart never produce filled qty > target
(Invariant Q); rate-limit backoff is bounded.

### Nautilus adapter / parity
Pass when: the engine reads position/account/portfolio state from Nautilus and
never duplicates it (Invariant N); only `nautilus/` touches Nautilus APIs; batch
replay and one-event-at-a-time streaming yield identical decisions (Invariant E).

### Validation
Pass when: IS decomposition, fill-prob calibration, adverse-selection, and
slippage-vs-model reports are produced with block-bootstrap CIs and CPCV/embargo;
realized STOP_MARKET gap-slippage is measured SEPARATELY from realized IS and fed
back to LEVERAGE's gap-survival ceiling (`RealizedStopFill`); the funding-timing
check runs on perps; the maker-branch internal-consistency (Invariant J) and the
online-update causality (Invariant L) tests pass; required baselines run under
identical fills/costs/benchmark; the engine beats always-MARKET on calm flow and
does not lose to it on urgent flow; deflated-Sharpe multiple-test control and
drop-top-N robustness are applied; nothing in the live layers imports `validate/`
(Invariant V).

---

## 16. Invariants (enforce structurally; ask before breaking)

**Invariant A — execution chooses tactic, never the trade.** No execution path
changes side, exceeds `target_qty`, or moves `invalidation`. `ExecTarget`
carries NO signal/sizing/exit score by construction. Test: replacing any
hypothetical edge/sizing field with arbitrary values leaves every cost, every
tactic, and every order unchanged; residual qty never exceeds target.

**Invariant B — tactic = `min()` of independent expected costs, never a product
of scores.** The chosen tactic is `argmin_t E[cost_t]` over the ALLOWED set; the
spread, impact, and fee components are computed independently so a bad impact
model cannot corrupt the spread term. No weighted multiplicative "execution
quality" combination exists. The ONE scoped exception is the maker-branch trio
(`p_fill`, `A`, `D`), which MUST share the regime driver `g` because they share a
physical cause (Invariant J) — that coupling is required and tested, not a score
blob, and it does not touch the spread/impact/fee firewall.

**Invariant C — RED unreachable from per-instrument config.** (Section 4.)

**Invariant D — the chase never violates the risk budget.** (Section 5.3.)

**Invariant E — batch/live parity.** Batch replay of a fill sequence and a
one-event-at-a-time stream produce identical `ExecDecision`s (Nautilus's own
backtest↔live parity plus the engine's determinism).

**Invariant H — diagnostics are log-only.** Per-tactic cost decompositions and
fill-prob diagnostics exist for monitoring; none may alter the chosen tactic,
price, or qty. Test: the decision is invariant to diagnostic contents.

**Invariant J — the maker-branch trio shares one regime driver and respects sign
relations.** (Section 5.1.1.) `p_fill`, `A`, and `D` all condition on the same
causal `g`; the assembled maker cost shows `p_fill` non-increasing, `A`
non-decreasing, `D` non-decreasing as `g` rises. The spread/impact/fee firewall
(Invariant B) is unaffected.

**Invariant L — runtime TEAL fits update online and causally (point-in-time /
embargoed).** The live `fill_prob` curve and the `adverse_selection` magnitude
are fitted from the instrument's OWN realized passive fills; a fill enters the
fit ONLY after its outcome horizon has fully elapsed, so the live curve at bar t
never contains a fill it is being scored against — no subtle lookahead, parallel
to the validation layer's CPCV/embargo. Test: inject a fill whose horizon has not
closed and assert it is absent from the curve used at t; the curve at t is
identical whether or not future-but-unresolved fills exist.

**Invariant W — funding is an execution-timing cost on perps only.** (Section
5.4.) Enters the maker no-fill branch and the cross cost on the paying side from
the `F_t`-known predicted rate / next stamp; floored at 0 on the receiving side;
identically 0 for non-perp classes.

**Invariant I — impact degrades to spread-only at tiny size.** The impact term
→ 0 continuously as participation → 0, so at the $1k clip spread/fees/adverse
selection dominate. Test: impact_bps at a $30 notional clip on a liquid venue is
negligible; impact grows correctly on thin proxies / synthetic large size.

**Invariant N — Nautilus owns plumbing/state/reconciliation.** Position,
account, portfolio state, order lifecycle, transport, pre-trade risk checks, and
funding ingestion are READ from / delegated to Nautilus. The engine duplicates
none of them; only `nautilus/` imports Nautilus APIs.

**Invariant Q — idempotent submission.** (Section 7.) Deterministic
`client_order_id` = `hash(intent_id, slice_idx, attempt)` for entry/exit SLICES;
no retry/restart sequence overshoots target qty. The slice id scheme is DISJOINT
from the stop id scheme (Invariant Q-STOP): a stop id carries the `"STOP"` tag and
a generation counter and never the slice_idx/attempt, so an entry-slice retry can
never collide with a stop id.

**Invariant Q-STOP — deterministic, generation-versioned stop id.** (Section 9.)
The protective stop's `client_order_id` = `hash(intent_id, "STOP",
amend_generation)`, a scheme DEDICATED to the stop and independent of the
entry/exit `slice_idx`/`attempt`. `amend_generation` starts at 0 and increments
on every invalidation change, and is DERIVABLE FROM OBSERVED STATE (the current
invalidation generation carried in `WorkingState`), not from an in-process
counter — so it survives restart. Consequences, all required-tested: (a) each
amend gets a FRESH non-colliding id the venue accepts (reusing an id would be
deduped and re-open a no-stop window); (b) reconciliation recomputes the CURRENT
expected stop id from the current invalidation generation and adopts exactly that
one, cancelling any stop whose id is a PRIOR generation (the multiple-stop cleanup,
Invariant S case (d)); (c) a stop id can never equal an entry/exit slice id, so an
entry-slice retry cannot clobber the stop. Test: drive N successive invalidation
moves, assert each produces a distinct accepted stop id, reconciliation always
adopts the current-generation id, and no stop id ever equals any slice id.

**Invariant R — reduce-only on all protective orders.** (Section 9.)

**Invariant S — the stop rests on the venue, always, and is reconciled.**
(Section 9.) No state has an open position with the stop living only in-process.

**Invariant T — no per-instrument tuning of RED on realized P&L.** (Section 4.)

**Invariant V — validation never imported by live layers.** Nothing in
`microstate/`, `cost/`, `policy/`, or `nautilus/` imports from `validate/`.

---

## 17. Build order

1. `types.py` and `synth/generate.py` (adversarial fill scenarios first).
2. `config/` with TEAL/AMBER/RED/BLUE + `venue.py`; test Invariant C.
3. `nautilus/adapter.py` skeleton (read-only Portfolio/Cache binding; Invariant
   N) so everything downstream tests against a stub Nautilus.
4. `microstate/spread.py` (quoted → Roll → Corwin-Schultz → class-proxy ladder,
   Roll-degeneracy guard, NaN/inf test on trending bars; Section 5),
   `microstate/vol.py`, `microstate/volume_proxy.py` (causal, reuse family vol),
   and the shared regime driver `g` + `tick_bps` (Section 5.1.1).
5. `microstate/fill_prob.py` (monotone in offset and `g`, THREE-RUNG terminal
   fallback bucket → global → conservative class prior with cold-start maker
   gate-off + gated PATIENT-only exploration, calibration-CI half-width for the
   LCB gate, auditable, ONLINE point-in-time/embargoed update; Invariants J/L,
   Section 5.5 #2).
6. `cost/spread_cost.py`, `cost/impact.py` (Invariant I), `cost/adverse_selection.py`
   (offset + `g`, online update), `cost/delay_cost.py` (offset + `g`, the
   no-fill-conditional drift), then `cost/tactic_cost.py` — PI = s − δ·tick_bps,
   entry no-fill split (fill-in-budget vs free abort), exit guaranteed-cross,
   perp funding term, `δ*` over the grid (Invariants B/J/W).
7. `policy/tactic_select.py` (argmin + gates; maker-floor gate keyed to the
   `p_fill` LCB with calibration-error-widened `floor_eff`, Section 6.3;
   Invariant H).
8. `policy/chase.py` (ladder, `worst_entry` bound, abort; Invariant D).
9. `policy/partial.py` (residual ladder, min-notional, qty discipline; Invariant
   A).
10. `policy/order_issuer.py` (order types, idempotent `client_order_id`,
    timeouts, retries; Invariant Q).
11. `policy/protective_stop.py` (venue-native reduce-only stop, amend with
    generation-versioned stop id `hash(intent_id,"STOP",amend_generation)`;
    Invariants R/S/Q-STOP).
12. `nautilus/reconcile.py` (`StopReconciler` as the `STOP_OK` projection from
    observed state; no-stop / stale / orphan / qty-mismatch / MULTIPLE-stop /
    PARTIAL-fire cases; idempotency-by-construction; Invariants S/Q-STOP).
13. `policy/engine.py` + `engine_api.py` (ties it together; Invariants
    B/E/H/N).
14. `validate/labels.py`, `validate/shortfall.py` (IS decomposition).
15. `validate/calibration.py` (fill-prob reliability, slippage-vs-model).
16. `validate/baselines.py`, `validate/ablation.py`.
17. `validate/walkforward.py` (CPCV/embargo), `validate/stats.py`
    (block-bootstrap, deflated Sharpe, drop-top-N), reports.
18. `priors/exec_dna.py` last.

Each step complete only when its acceptance gate (Section 15) is green. A module
that fails its gate does not advance.

Stop after step 18. The remaining questions — whether the fill-prob curve is
calibrated on YOUR feed, whether breakout adverse selection is as large as
assumed, whether χ is set right, which tactic actually drives realized cost — are
answered by RUNNING the validation harness on real fills, not by adding tactics.

---

## 18. References and rationale

- Implementation shortfall as the execution objective: Perold (1988), *The
  Implementation Shortfall: Paper versus Reality*; decompose vs arrival price.
- Almgren & Chriss (2000), *Optimal Execution of Portfolio Transactions* —
  the timing-risk vs impact trade-off; here the "timing risk" is breakout drift
  `D`, sharpened because the signal is directional by construction.
- Square-root market-impact law: Almgren, Thum, Hauptmann & Li (2005); Tóth et
  al. (2011) *Anomalous price impact*; Kyle & Obizhaeva — `impact ∝ σ·√(Q/V)`,
  which vanishes at retail size (P8/Invariant I).
- Effective-spread estimation without quotes: Roll (1984) serial-covariance
  estimator — VALID ONLY when serial autocovariance of price changes is negative;
  on trending bars it is frequently positive and `sqrt(-cov)` is undefined, so
  Corwin & Schultz (2012) high-low spread is the mandated fallback (Section 5),
  with a class-level proxy floor beneath it. The estimator never returns
  NaN/inf.
- Perpetual-swap funding as a carry/timing cost: Binance USDT-M funding mechanics
  (8h stamps, predicted rate published ahead of the stamp) — the funding rate and
  next stamp are `F_t`-known, so crossing before vs waiting past a stamp is a real
  execution-timing cost on a patient clip (Section 5.4), distinct from the hold
  carry LEVERAGE already owns.
- Adverse selection / the maker's short option: Glosten & Milgrom (1985);
  Glosten & Harris (1988) — a resting limit fills precisely when informed flow
  trades through it; this is the `A` term and the reason passive loses on a
  breakout.
- Optimal limit-order placement vs market order: the maker/taker break-even is a
  standard limit-vs-market cost comparison (e.g. Harris, *Trading & Exchanges*);
  the cancel/replace chase is its sequential, budget-bounded extension.
- Purged / embargoed CV and deflated Sharpe: López de Prado, *Advances in
  Financial Machine Learning* — CPCV, embargo, deflated/probabilistic Sharpe for
  multiple-test control over the tactic/χ grid.
- Block bootstrap for dependent data: Künsch (1989); Politis & Romano
  (stationary bootstrap) — CIs that respect volume/vol clustering across fills.
- NautilusTrader docs: order types, `ExecAlgorithm`, contingent/OCO orders,
  reduce-only, and restart reconciliation — the plumbing this module sits on and
  does NOT re-implement (Invariant N).

---

## 19. Non-goals (belong to sibling modules and to Nautilus)

Signal generation and invalidation level (BREAKOUT.md); position size / leverage
(LEVERAGE.md); hold/reduce/exit timing and where the trail/invalidation sits
(TREND.md); and the raw order plumbing — order submission/lifecycle, position /
account / portfolio state, restart reconciliation primitives, data transport,
backtest↔live parity infrastructure, pre-trade risk checks, and funding
ingestion (NautilusTrader). This module receives a fully-decided target and a
microstate, and emits the cheapest sequence of venue orders that fills it within
the risk budget, with a venue-resident protective stop. It never decides whether
to trade, how much, or when to leave — only how to get the decided position on
and off the book at the lowest cost.
