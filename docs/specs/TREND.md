# TREND_9_10.md - institutional trend-life detector for cheap compute

This file is the authoritative v2 build spec for an independent trend-life
module. It is designed for commodity futures, metals, US equities, and crypto,
with open-position holding periods up to 7 days. It is scoped to trend-life
detection and convex exit management only.

The module does not generate entries. It receives an already-open position:
side, entry index, entry price, and initial invalidation/risk if available. On
every bar it answers:

1. Is the trend still alive enough to hold?
2. Has deterioration or death been detected?
3. Should the position continue, reduce, or exit?
4. How should the convex trail evolve so the left tail is cut while the right
   tail remains open?

Target quality for a retail-compute institutional-style implementation:
- Research architecture: 9/10.
- Statistical realism: 9/10.
- Signal sophistication: 9/10 without black-box ML.
- Cheap-compute practicality: 9/10.
- Detector/policy production readiness: 9/10 for this module only.

Non-goals: entry detection, portfolio allocation, broker connectivity, global
risk, full execution algorithms, and position sizing beyond gap-survival and
reduce/exit policy. Those belong to later modules.

---

## 0. Governing Principle

At intraday-to-7-day horizons, drift is hard to estimate. More 1-minute bars
improve volatility and path estimates, but they do not magically create a
high-power estimate of expected return over a few calendar days. Therefore this
module must not try to confirm that a trend is alive with a pile of correlated
indicators.

The correct stance is:

1. Alive until proven dead.
2. Detect deterioration, invalidation, and death.
3. Use economic thresholds, false-alarm controls, and convexity preservation.
4. Keep every estimator testable apart from the policy that acts on it.

The trend-life module is a death detector plus convex exit policy, not a trend
prediction model.

Forbidden in v1:
- HMM or particle filters.
- Neural networks or learned black-box classifiers.
- Hurst, ADF, KPSS, or Bai-Perron as live decision signals.
- Indicator voting such as 2-of-5 confirmation.
- Multiple sequential detectors for the same latent quantity.
- Per-symbol parameter optimization.

Research diagnostics may compute Hurst/ADF/KPSS/Bai-Perron offline, but live
estimation and policy modules must not import them.

---

## 1. Dynamic Horizon Definition

"Up to 7 days" is a maximum observation cap, not a fixed holding period and not
an exit rule. The live trend horizon is a data-dependent stopping time. This is
the mathematical contract that lets the module capture trend convexity: short
when the thesis is invalidated, long when the right tail keeps extending.

### 1.1 Information set

Let `F_t` be the information available when bar t has closed:

```text
F_t = sigma(bars[0:t], position, resolved_config, online_state[0:t])
```

Every live quantity used by the horizon, belief state, trail, reduce decision,
or exit decision must be `F_t`-measurable. Validation labels may use future
bars, but live horizon logic may not.

### 1.2 Horizon clocks

The code must support four clocks:

```python
HorizonClock = Literal[
    "calendar_days",
    "trading_sessions",
    "active_minutes",
    "bars_since_entry",
]
```

Let `A_t^c` be the elapsed age of the open position under clock c. Examples:
calendar milliseconds, completed trading sessions, active tradable minutes, or
bars since entry. The maximum cap under clock c is `H^c`. The cap time is

```text
tau_cap^c = inf { t >= entry_idx : A_t^c >= H^c }
```

Asset-class defaults:
- US equities: primary clock is trading sessions plus active RTH minutes.
  Calendar days are reported separately.
- Commodity futures and metals futures: primary clock is exchange trading
  sessions, with explicit RTH/ETH policy.
- Crypto: primary clocks are calendar days and active minutes because the
  market trades 24/7.
- Cross-asset reports must never pool durations across clocks without recording
  the clock and normalizing the comparison.

Every decision, label, and report must include:

```python
elapsed_calendar_ms
elapsed_sessions
elapsed_active_minutes
bars_since_entry
active_clock
horizon_cap
cap_fraction_used
```

No module may assume that 7 days equals a fixed number of bars.

### 1.3 Path variables for convexity

For a position with side `s in {-1, +1}` and entry price `P_0`, define the
side-adjusted log PnL path:

```text
X_t = s * log(P_t / P_0)
```

Define the favorable high-watermark, giveback, and accumulated variance:

```text
M_t = max_{entry_idx <= u <= t} X_u
G_t = M_t - X_t
V_t = sum_{u=entry_idx+1..t} sigma_u^2 + gap_var_u
Z_t = X_t / sqrt(max(V_t, eps))
```

`M_t`, `G_t`, `V_t`, and `Z_t` are causal. `M_t` is not future MFE; it is the
running favorable excursion known at t. Future MFE exists only in validation.

### 1.4 Dynamic horizon as a stopping time

The live trend-life horizon is:

```text
tau_live = min(tau_death, tau_cap)
```

where

```text
tau_death = inf { t >= entry_idx : D_t = 1 }
```

and `D_t` is the union of causal death or invalidation events:

```text
D_t = 1 if any of these is true at t:
  jump_against_t
  sequential_deterioration_alarm_t
  trail_or_invalidation_breach_t
```

Reduce-only events are not deaths:

```text
R_t = 1 if hazard/slope/gap-survival state says reduce but not exit
```

`R_t` may lower `size_fraction`, but it must not be counted as trend death
unless the policy explicitly promotes it to a validated hard-exit rule.

Invalid data, exchange halts, missing bars, and untradeable states are not trend
death evidence. They are handled as continue/reduce in live policy or as
`data_censored` in validation, depending on whether the position can still be
observed.

This makes `tau_live` a valid stopping time:

```text
{tau_live <= t} is in F_t for every t.
```

That property is non-negotiable. If a horizon rule requires future MFE, future
return, future volatility, or future "would have recovered" information, it is
a validation label, not live logic.

### 1.5 How dynamic horizon captures convexity

A convex trend exit must be asymmetric:

```text
left tail:  cut quickly when D_t fires or early invalidation is breached
right tail: keep holding while M_t rises and D_t remains false
```

The trail boundary must be a causal function

```text
B_t = b(M_t, V_t, G_t, clock_t, vol_bucket_t, liq_bucket_t)
```

with these monotonicity requirements:

```text
as M_t increases materially, the allowed giveback G_t may widen
as V_t increases, the noise-adjusted trail may widen unless deterioration fires
as early adverse excursion increases before profit, the invalidation stays tight
```

In practical terms: tight early, looser after the trade earns the right to be
loose. A fixed k-bar horizon cannot do this; it exits winners and losers by the
same clock. A dynamic stopping-time horizon exits by path evidence.

Validation must measure this with:
- MFE capture ratio.
- Giveback ratio.
- tail capture under the 7-day cap.
- positive skew.
- cap-binding fraction.
- survival curves by death definition (Section 13a).

The convexity claim is asserted here only qualitatively. It is made an explicit,
quantified, and FALSIFIABLE statement in Section 1.5a (payoff identity) and
Section 1.5b (premium-bleed breakeven). Those two subsections are the
mathematical heart that Sections 13/18/21 detect and gate on. Do not read "tight
early, loose late" as the convexity argument; it is only the mechanism.

### 1.5a Convexity made explicit: the realized round-trip payoff is option-like

Convexity is not a property of the trail schedule by itself; it is a property of
the realized round-trip payoff as a function of the favorable move. State it as
an identity, not a slogan.

Fix one position. Let the trail boundary be `B_t = b(M_t, ...)` (Section 1.5).
The realized exit happens at `tau_live = min(tau_death, tau_cap)`. Write the
terminal favorable high-watermark `M = M_{tau_live}` (in side-adjusted log-PnL
units, so `M >= 0` always) and the giveback surrendered at exit
`G_exit = M - X_{tau_live} = G_{tau_live} >= 0`. The realized round-trip log
payoff is

```text
Pi  =  X_{tau_live}  =  M  -  G_exit
```

The trail schedule controls `G_exit` as a function of `M`. Pin the two regimes
the asymmetric schedule MUST produce (these are the monotonicity requirements of
1.5, restated as a payoff bound):

```text
left tail  (M small, trade never earns out):  Pi  >=  -R0
              the initial-risk cut R0 = initial_risk (in the same log units)
              is the maximum loss; B_t starts at the entry invalidation, so the
              downside is bounded below by construction.
right tail (M large, trend persists):        G_exit  =  g(M)  with  g'(M) < 1
              i.e. giveback grows slower than the move, so Pi = M - g(M) is
              increasing and UNBOUNDED above as M grows.
```

The conjunction "bounded-below loss `-R0`, unbounded-above right tail with
`g'(M) < 1`" is exactly the payoff of a long option position: convex in the
underlying favorable move `M`. Equivalently, `Pi(M)` is a convex function of `M`
when the schedule keeps the marginal giveback `g'(M)` non-increasing in `M` over
the loose/tail-preservation regions (the multiplier widens as profit grows,
Section 10), so the trade gives back a smaller FRACTION of each additional unit
of favorable move the further the trend runs. A symmetric trail has `g'(M)`
roughly constant (and the multiplier fixed), so `Pi(M)` is affine, not convex —
it does not buy the right tail.

Pinned convexity invariant (the schedule must satisfy, tested in 18):

```text
INV-CVX:  d/dM ( marginal giveback g'(M) )  <=  0   over M in the
          loose and tail-preservation regions of the schedule (Section 10),
          AND  Pi(M=0 path) = -R0  (bounded-below initial-risk cut).
```

This is what makes the strategy's payoff option-like rather than just
"asymmetric." The reduce path (`R_t`) lowers `size_fraction` and so scales `Pi`
by the held fraction; it does not change the convex SHAPE of `Pi(M)`.

### 1.5b The lookback-straddle decomposition (Fung-Hsieh), written out

Section 13a.6 and Section 21 name-drop "trend-following payoff ~ long a lookback
straddle (Fung-Hsieh)." Make it a derived statement, not a citation.

A perfect trailing exit that gave back nothing (`G_exit = 0`) would realize the
entire favorable excursion `M = max_u X_u` over the holding window. The payoff
that pays the running maximum of a price path is, by definition, the payoff of a
**lookback straddle** (a lookback call on the favorable side plus a lookback put
on the adverse side; here the side is fixed by the position so we take the
relevant leg). Concretely, over the window `[entry_idx, tau_cap]`,

```text
M_ideal  =  max_{entry_idx <= u <= tau_cap}  X_u        # lookback-straddle leg
Pi       =  M_ideal  -  G_exit  -  R0_realized_on_failures
```

So the realized trend-following payoff decomposes as

```text
trend payoff  =  ( long lookback-straddle leg on the favorable side )
                 -  ( giveback friction G_exit, the imperfect-trail cost )
                 -  ( the stream of initial-risk cuts R0 on trades that fail )
```

The first term is the convex, option-like component Fung-Hsieh identify
empirically; the trail's job is to track `M_ideal` as tightly as the noise floor
allows (minimize `G_exit`) WITHOUT cutting the right tail early (do not collapse
`M_ideal`). The MFE-capture ratio (Section 13) is precisely the empirical
estimate of `1 - E[G_exit]/E[M_ideal]`, i.e. how much of the lookback-straddle
leg the trail actually realizes. This is why MFE capture is the headline
convexity diagnostic: it measures the fraction of the option-like component
retained. The two subtracted terms are the bleed; Section 1.5b's decomposition
is what Section 1.5c (premium bleed) quantifies on the cost side.

### 1.5c Premium bleed and the net-of-cost expectancy breakeven

Cut-losers-fast is not free. The convex right tail of 1.5a is BOUGHT by paying a
stream of small left-tail cuts — every trade that is invalidated early at `-R0`,
plus the spread/slippage/whipsaw on each premature cut. This is the option-
premium analogue: a long straddle is convex but bleeds theta. The convexity
claim of 1.5a is unfalsifiable unless this cost side is stated, so pin it.

Expectancy identity for the trail (in the same log-PnL units as `Pi`, all
quantities measured NET of cost via Section 6 / COST.md estimators):

```text
E[Pi]  =  p_win * avg_win  -  p_loss * avg_loss
            where p_win + p_loss = 1
                  avg_win  = E[Pi | Pi > 0]        (the captured right tail)
                  avg_loss = E[-Pi | Pi <= 0]      (the R0 cut + round-trip cost)
```

Define the **premium bleed** as the cumulative cost of the tight-early cuts that
never developed into a right tail — the trades the trail killed at or near `-R0`
that, absent the trail, were not going to run anyway, plus the friction of
re-cutting:

```text
premium_bleed  =  sum over premature/undeveloped cuts of
                    ( |R0_i|  +  spread_i + slippage_i + fees_i + whipsaw_i )
```

The breakeven condition — the thing that decides whether "ride winners yields
convex P&L" is actually net-positive — is

```text
NET CONVEXITY BREAKEVEN:
    E[ right-tail capture ]   >   premium_bleed
equivalently
    p_win * avg_win           >   p_loss * avg_loss        (E[Pi] > 0 net of cost)
```

A worked illustration (numbers are nominal, in `R0` units, to make the gate
concrete — they are NOT defaults to fit):

```text
suppose  p_win = 0.35,  avg_win = 3.2 R0  (tail captured),
         p_loss = 0.65, avg_loss = 1.15 R0 (cut + round-trip cost)
  E[Pi] = 0.35*3.2 - 0.65*1.15 = 1.12 - 0.7475 = +0.3725 R0  > 0  -> PASS
if costs rise so avg_loss = 1.6 R0 (heavier whipsaw/slippage):
  E[Pi] = 1.12 - 0.65*1.60 = 1.12 - 1.04 = +0.08 R0  -> marginal
if also avg_win erodes to 2.6 R0 (cap clips the tail, Section 13a.6):
  E[Pi] = 0.35*2.6 - 1.04 = 0.91 - 1.04 = -0.13 R0  -> FAIL: bleed wins
```

The third line is the whole point: a strategy that is convex GROSS (positive
skew, good MFE capture) can still be net-negative once the premium bleed and a
cap-clipped right tail are charged. Skew and MFE capture are GROSS diagnostics;
the promotion gate (Section 13) and the convex-exit acceptance gate (Section 18)
must require `E[Pi] > 0` NET of cost, with costs sourced from COST.md (Section 6
note), not the local placeholder.

Reported metrics added in Section 13: `premium_bleed`, `avg_win`, `avg_loss`,
`p_win`, and the net expectancy `E[Pi]` reported ALONGSIDE tail capture, so a
reader can see the right tail and the cost of buying it in the same table.

### 1.6 Fixed cap versus dynamic life

The 7-day cap has only three valid roles:

1. Validation censoring cap: if the trend is still alive at H, the observation
   is right-censored for trend-life survival analysis.
2. Operational maximum-hold cap: if a later trading layer forbids holding past
   H, the exit is marked as `time_cap`, not trend death.
3. Research diagnostic: if many winners are censored at H and tail capture is
   poor, H is clipping convexity and must be reported as cap-binding.

The cap must not be silently treated as the natural death time of the trend.

#### 1.6a The operational cap is a SYSTEM-injected RED parameter (composition)

The operational maximum-hold cap (role 2 above) is NOT a TREND-internal constant.
It is supplied by the orchestrating SYSTEM layer. Pin the boundary:

- The operational cap horizon `H^c` (the cap under the active clock used for
  role 2) is a caller-supplied RED operational value. It is exposed in
  `ResolvedConfig` as `tau_cap` (alias `H_op`), and is set by SYSTEM to its
  contractual hold bound `H_sys` per SYSTEM.md **Invariant HC**.
- `H_sys` is the constant that bounds SYSTEM's integrated-validation embargo
  (the purged-CPCV/DSR firewall must know the maximum realized hold so it can
  embargo correctly). Therefore TREND must NOT realize a hold longer than
  `tau_cap = H_sys`: the engine force-emits `stop_cause = "time_cap"` at the
  first bar where the active clock reaches `H_sys`, marking a still-alive
  position closed.
- Roles 1 (validation censoring cap) and 3 (research diagnostic) may use a
  separate, possibly LARGER, observation cap `H_obs` so the survival analysis
  (Section 13a) can see the tail beyond the operational cut. When `H_obs >
  H_op`, the survival curve treats `H_op`-forced `time_cap` exits as CENSORED
  for trend-death definitions (consistent with 13a.3 competing-risks handling),
  not as deaths.

This is the only legitimate source of the operational cap. TREND does not invent
a 7-day number; "7 days" is the typical value SYSTEM injects, recorded as RED.
See Section 7 Invariant K (revised) and SYSTEM.md Invariant HC.

### 1.7 Dynamic horizon tests

Required tests:
- `tau_live` is a stopping time: changing bars after t cannot change whether
  `tau_live <= t`.
- A persistent synthetic trend has longer realized `tau_live` than a dying
  synthetic trend under the same cap.
- Increasing future MFE after t does not change the decision at t.
- A 7-day survivor is labeled censored for trend-life survival, not dead.
- Cap-binding diagnostics fire when a large share of high-MFE trades survive to
  H with low tail capture.

---

## 2. Architectural Split

The architecture has four layers:

1. Data layer: validates bars, applies calendars, marks gaps, rolls, splits,
   stale bars, session opens, and asset-specific market-structure flags.
2. Estimation layer: emits beliefs and diagnostics only. It never emits actions.
3. Policy layer: maps beliefs and economic rules to continue/reduce/exit.
4. Validation layer: creates future labels, baselines, ablations, and reports.
   It is never imported by live modules.

Why this split matters:

- Bad estimator calibration can be separated from bad exit policy.
- Each component can be ablated cheaply.
- Lookahead surfaces are isolated.
- The same code can run in batch replay and online mode.

---

## 3. Directory Layout

```text
trend/
+-- config/
|   +-- defaults.py          # PolicyDefaults: RED shared policy values
|   +-- instrument.py        # InstrumentSpec: AMBER metadata only
|   +-- asset_class.py       # AssetClassPreset: controlled class overrides
|   +-- cost.py              # CostModel for validation friction
|   +-- tuning.py            # finite cheap-compute grids and promotion gates
|   +-- resolve.py           # single merge point
+-- data/
|   +-- schema.py            # OHLCV validation and bad-bar flags
|   +-- calendar.py          # RTH/ETH/futures/crypto sessions
|   +-- actions.py           # equity split/dividend adjustment hooks
|   +-- rolls.py             # futures roll/stitch metadata hooks
|   +-- quality.py           # missing/stale/outlier bars
|   +-- timing.py            # decision_idx/action_idx conventions
+-- estimate/
|   +-- volatility.py        # robust vol stack + gap variance
|   +-- kalman.py            # class-preset fixed-noise local trend filter
|   +-- drift.py             # variance-integrated normalized drift
|   +-- horizon.py           # causal dynamic horizon / stopping-time state
|   +-- sequential.py        # exactly one deterioration detector
|   +-- jump.py              # jump-against/jump-with diagnostics
|   +-- hazard.py            # optional event-count hazard ladder
|   +-- state.py             # BeliefBuilder, online estimator state
+-- policy/
|   +-- decision.py          # only module that emits ActionDecision
|   +-- convex_exit.py       # asymmetric trail and gap survival
|   +-- reduce.py            # reduce-only rules, no hard exits
+-- priors/
|   +-- market_dna.py        # slow monthly/quarterly priors, not live diagnostics
+-- research/
|   +-- diagnostics.py       # Hurst/ADF/KPSS/PELT/Bai-Perron offline only
+-- validate/
|   +-- labels.py            # future labels, MFE/MAE/capture, validation only
|   +-- survival.py          # censoring-correct trend-life duration (KM all-cause + Aalen-Johansen CIF, p95/p99)
|   +-- walkforward.py       # purged walk-forward
|   +-- cpcv.py              # sampled combinatorial purged CV
|   +-- baselines.py         # cheap baseline suite
|   +-- ablation.py          # component contribution tables
|   +-- stats.py             # confidence intervals and multiple-test controls
|   +-- reports.py           # asset-class and regime reports
+-- tune/
|   +-- search_space.py      # finite grids only
|   +-- autotune.py          # class-level/preset selection
|   +-- promotion.py         # approved/shadow/rejected decisions
|   +-- registry.py          # hashes, windows, selected presets
+-- online/
|   +-- replay.py            # batch/live parity
|   +-- monitor.py           # drift, calibration, data health
|   +-- snapshot.py          # serializable live state
+-- synth/
|   +-- generate.py          # adversarial synthetic trend paths
+-- types.py                 # frozen dataclasses crossing boundaries
+-- engine.py                # TrendLifeEngine
tests/
+-- ...                      # unit, leakage, parity, acceptance, ablation tests
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest. Numba is allowed only after
correctness tests exist. No ML frameworks in v1.

---

## 4. Parameter Governance

Every parameter belongs to exactly one bucket.

### TEAL - fitted per instrument from past data

TEAL values are fitted from the instrument's own past data. They are not
hand-set in `InstrumentSpec`.

Examples:
- Intraday volatility seasonality.
- Liquidity/volume seasonality.
- Robust volatility scale.
- Tail parameter or empirical exceedance profile.
- Gap distribution by session boundary.
- Typical trend duration distribution (estimated censoring-correctly; see 13a).
- Empirical null distribution for sequential statistics.
- Stale/missing-bar profile.

TEAL fitting rule: a value used at bar t must use only data available at or
before t. Session profiles used inside session S must use sessions before S,
unless the profile is explicitly online-recursive.

### AMBER - supplied market metadata

AMBER values are supplied because market structure differs across instruments.
They are not optimized by the trend module.

Examples:
- Asset class, venue, timezone.
- RTH/ETH/24-7 calendar policy.
- Tick size, lot size, multiplier.
- Futures roll rule and expiry metadata.
- Equity corporate-action source.
- Fee, spread, and slippage model for validation.
- Halt/auction flags for equities.
- Funding/outage flags for crypto.
- Bid/ask availability.

### RED - shared economic and modelling choices

RED values live in `PolicyDefaults`. They are shared broadly and may have
validated asset-class presets.

Examples:
- Economic drift floor.
- Sequential detector kind: SPRT or CUSUM, exactly one.
- Target ARL or alpha/beta.
- Trail asymmetry family.
- Gap-survival multiple.
- Reduce/exit priority order.
- Minimum and maximum stop distance.
- Validation horizon clocks and maximum caps.

### BLUE - controlled cheap-compute tuning grids

BLUE values are finite candidate presets searched by `tune/autotune.py`. They
are not free optimization parameters.

Allowed BLUE selections:
- Global default.
- Asset-class preset.
- Liquidity-bucket preset.

Forbidden:
- Per-symbol tuning of Kalman noise ratio, trail multipliers, alpha/beta, or
  ARL based on that symbol's PnL.
- Online self-retuning from recent trade outcomes.
- Expanding a grid after seeing backtest results without recording a new
  research version and multiple-test penalty.

---

## 5. Asset-Class Presets

The module uses one code path. Asset-class differences enter through AMBER
metadata, TEAL fitted profiles, and class-level presets that have passed
validation.

Required preset names:

```python
AssetClass = Literal[
    "commodity_future",
    "precious_metal",
    "industrial_metal",
    "us_equity_liquid",
    "us_equity_illiquid",
    "crypto_major",
    "crypto_alt",
]
```

### Commodity futures

Required metadata:
- Exchange calendar.
- RTH/ETH policy.
- Contract multiplier, tick size, and expiry metadata.
- Roll/stitch rule.
- Limit move flags if available.

Required reports:
- Normal sessions vs roll windows.
- Inventory/report windows when timestamps are supplied.
- Gap events across session boundaries.

### Precious metals

Examples: gold, silver, platinum.

Required metadata:
- Venue/session policy.
- Futures roll metadata if futures are used.
- Settlement/fixing windows if relevant.

Optional context:
- USD index, rates, real-yield proxy.

Required reports:
- Macro-window gap/jump behavior.
- RTH vs overnight behavior.

### Industrial metals

Examples: copper, aluminum, zinc.

Required metadata:
- Venue-specific session and roll policy.
- Liquidity/stale-bar flags, especially for less liquid contracts.

Optional context:
- Growth/risk proxy, China session proxy, copper complex proxy.

### US equities

Required metadata:
- Corporate actions.
- RTH/ETH policy.
- Halt and auction flags if available.
- Liquidity bucket.
- Shortability/borrow flag if short positions are evaluated.

Required reports:
- Earnings windows vs non-earnings windows.
- Opening gap decisions.
- Large/mid/small/liquidity buckets.
- RTH-only and ETH-inclusive variants.

### Crypto

Required metadata:
- Exchange/venue.
- Spot vs perpetual.
- Fee schedule.
- Funding-rate availability for perps.
- Outage/stale data flags.
- UTC day/week segmentation.

Required reports:
- Weekday vs weekend.
- Funding regimes.
- High-volatility liquidation regimes if data exists.
- Exchange-specific performance if multiple venues are used.

---

## 6. Shared Types

All public objects crossing module boundaries are frozen dataclasses.

```python
from dataclasses import dataclass
from typing import Literal, Mapping
import numpy as np

Side = Literal[-1, 1]
Action = Literal["continue", "reduce", "exit"]
DetectorKind = Literal["sprt", "cusum"]
VolMethod = Literal["ewma", "mad", "yang_zhang", "hybrid"]
HorizonClock = Literal[
    "calendar_days",
    "trading_sessions",
    "active_minutes",
    "bars_since_entry",
]
StopCause = Literal[
    "jump_against",
    "seq_alarm",
    "trail_breach",
    "time_cap",
    "data_censored",
    "censored",
]

@dataclass(frozen=True)
class Bars:
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray | None = None
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    is_session_open: np.ndarray | None = None
    is_valid: np.ndarray | None = None
    is_halt: np.ndarray | None = None
    is_roll_window: np.ndarray | None = None
    funding_rate: np.ndarray | None = None

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: str
    venue: str
    timezone: str
    tick_size: float
    lot_size: float
    multiplier: float
    calendar_id: str
    data_policy: Mapping[str, str]
    optional_sources: Mapping[str, bool]

@dataclass(frozen=True)
class Position:
    side: Side
    entry_idx: int
    entry_price: float
    initial_invalidation: float | None = None
    initial_risk: float | None = None

@dataclass(frozen=True)
class CostModel:
    spread_bps: float
    slippage_bps: float
    fees_bps: float
    min_tick_slippage: float

@dataclass(frozen=True)
class ClockState:
    elapsed_calendar_ms: int
    elapsed_sessions: int
    elapsed_active_minutes: int
    bars_since_entry: int

@dataclass(frozen=True)
class HorizonState:
    active_clock: HorizonClock
    horizon_cap: float
    elapsed_value: float
    cap_fraction_used: float
    cap_remaining: float
    integrated_var_since_entry: float
    running_favorable_excursion: float
    running_giveback: float
    cap_binding_warning: bool

@dataclass(frozen=True)
class VolState:
    method: VolMethod
    sigma_log: float
    sigma_price: float
    integrated_var_since_entry: float
    gap_var: float
    vol_bucket: int
    liq_bucket: int
    tail_nu: float
    quality_flag: str

@dataclass(frozen=True)
class BeliefDiagnostics:
    innovation_whiteness: float
    kalman_consistency: float
    detector_margin: float
    jump_flag_rate: float
    hazard_rung: Literal["state", "single", "constant", "disabled"]
    effective_sample_size: int
    tail_nu: float
    vol_method: VolMethod
    vol_quality_flag: str

@dataclass(frozen=True)
class BeliefState:
    idx: int
    clock: ClockState
    horizon: HorizonState
    vol: VolState
    filtered_slope: float
    slope_var: float
    z_drift: float
    adverse_excursion_r: float
    favorable_excursion_r: float
    seq_stat: float
    seq_alarm: bool
    jump_against: bool
    jump_with: bool
    hazard: float
    hazard_model: Literal["state", "single", "constant", "disabled"]
    reason_signals: Mapping[str, float]
    diag: BeliefDiagnostics

@dataclass(frozen=True)
class ExitState:
    trail_level: float
    trail_multiple: float
    high_watermark: float
    low_watermark: float
    open_profit_r: float
    initial_risk: float
    invalidation: float
    size_fraction: float

@dataclass(frozen=True)
class HoldHorizonEstimate:
    """TREND-owned hold-horizon estimate consumed read-only by LEVERAGE and COST.

    This is NOT the live stopping time tau_live (a random F_t-measurable stopping
    time, not a scalar). It is a conservative causal QUANTILE/RANGE of remaining
    hold under the active clock, derived from the censoring-correct trend-life
    survival law (Section 13a) conditioned on cap_fraction_used. Because drift is
    unmeasurable (Section 0), it is reported as a range, never a point scalar
    (consistent with COST 5.5d)."""
    active_clock: HorizonClock
    expected_remaining_lo_ms: int        # conservative low quantile of remaining hold
    expected_remaining_hi_ms: int        # conservative high quantile of remaining hold
    max_hold_quantile_ms: int            # m_max for COST funding gate (e.g. p90 of remaining)
    quantile_used: float                 # which survival quantile backs max_hold_quantile_ms
    op_cap_remaining_ms: int             # hard bound = H_sys - elapsed (Section 1.6a)

@dataclass(frozen=True)
class ResidualMoveEstimate:
    """TREND-owned expected residual FAVORABLE move, consumed read-only by COST's
    funding-aware hold_carry gate (COST 8.2 / 1180-1184). Causal: it is the
    favorable continuation TREND still expects from an open position, NOT a future
    label. Reported with a band because drift is unmeasurable (Section 0)."""
    expected_residual_move_lo: float     # in initial-risk R0 units, conservative low
    expected_residual_move_hi: float     # in R0 units, conservative high
    point_used_by_cost: float            # the single conservative value COST consumes
    units: Literal["initial_risk", "robust_vol"]

@dataclass(frozen=True)
class ActionDecision:
    idx: int
    decision_idx: int
    action_idx: int
    action: Action
    horizon: HorizonState
    stopping_time_fired: bool
    stop_cause: StopCause | None
    reason_codes: tuple[str, ...]
    exit_state: ExitState
    size_fraction: float
    hold_horizon: HoldHorizonEstimate     # TREND-owned; LEVERAGE/COST consume read-only
    residual_move: ResidualMoveEstimate    # TREND-owned edge; COST consumes read-only
    action_price_kind: Literal["next_open", "next_mid", "close_if_live_executable"]
    execution_intent: "ExecutionIntent"   # exit INTENT for EXECUTION; see Section 6a
    engine_version: str
```

`decision_idx` is the bar where the decision becomes known. `action_idx` is the
first bar where the action can be evaluated or executed. If decisions are made
on bar close, `action_idx` is normally `decision_idx + 1`.
`stopping_time_fired` is true only when the policy has reached `tau_live`.
`stop_cause` records whether the stop was trend evidence (`jump_against`,
`seq_alarm`, `trail_breach`), operational (`time_cap`), or censoring/data
related (`data_censored`, `censored`).

```python
@dataclass(frozen=True)
class ExecutionIntent:
    """The exit/trail INTENT TREND hands to EXECUTION. TREND owns the LEVEL and
    the intent + urgency; EXECUTION owns placement/amend/reconcile of the
    venue-resident reduce-only stop (Section 6a). TREND never places orders."""
    invalidation_level: float            # trigger price EXECUTION rests on the venue
    reduce_only: bool                    # always True for protective exits
    urgency: Literal["passive", "immediate"]  # 'immediate' on hard exit (D-events/time_cap)
    kind: Literal["amend_stop", "exit_reduce_only", "none"]
    # 'amend_stop' = trail moved, propagate an invalidation amend (no taker fill)
    # 'exit_reduce_only' = hard exit fired, reduce-only IMMEDIATE
    # 'none' = continue, no execution change
```

The `ExecutionIntent` carried on every `ActionDecision` is how TREND's exit
decision reaches EXECUTION. See Section 6a for the pinned execution boundary.
`HoldHorizonEstimate` and `ResidualMoveEstimate` are TREND-owned outputs consumed
read-only by LEVERAGE (stop sizing) and COST (funding/carry gate); see Section
6b.

### 6.1 CostModel is a validation stub; COST.md supersedes it (composition)

The `CostModel` dataclass above is the local validation-friction STUB only. It
exists so the `validate/` layer can run self-contained in isolation. It is NOT
the authoritative friction model for any net-of-cost gate.

Pin (matches COST.md P3, which states the placeholder `CostModel` in BREAKOUT.md
/ TREND.md are explicitly superseded): for any net-of-cost decision — the net
expectancy gate `E[Pi] > 0` (Section 1.5c), the promotion gate (Section 13), and
anything SYSTEM labels "net-of-cost" — costs MUST be sourced from COST.md's
shared estimators, the same estimators BREAKOUT / LEVERAGE / TCA call. TREND, the
other pillars, and SYSTEM's net-of-cost label must agree on one cost source;
using the local stub for a promotion or net-EV gate is forbidden because it lets
TREND's gate and SYSTEM's label disagree. The local `CostModel` may be used only
for standalone, clearly-labeled validation friction when COST estimators are not
wired in (e.g. a unit test of `validate/labels.py`).

### 6a. Execution boundary: TREND decides exit intent, EXECUTION executes it

TREND owns the invalidation/trail LEVEL and the exit INTENT (plus urgency).
EXECUTION owns placement, amend, and reconcile of the venue-resident reduce-only
stop. TREND never places, amends, or cancels orders directly — NautilusTrader and
EXECUTION own all order plumbing. The positive contract (the prior non-goal only
stated the negative):

- `ExitState.invalidation` / `ExitState.trail_level` is the trigger price that
  EXECUTION rests on the venue as a reduce-only stop. This is the
  `invalidation_level` carried on `ExecutionIntent`.
- When the trail MOVES (a tightening or widening of `trail_level` between bars),
  TREND emits `ExecutionIntent(kind="amend_stop", ...)`; EXECUTION propagates it
  as an invalidation AMEND of the resting venue stop. A trail move is not a new
  order from TREND; it is an amend instruction EXECUTION reconciles. This matches
  EXECUTION.md `ProtectiveStop.place/amend` placing the protective stop at "the
  invalidation level handed down by upstream (TREND/BREAKOUT)."
- When a hard exit fires (any D-event, or `time_cap`), `Action="exit"` maps to
  `ExecutionIntent(kind="exit_reduce_only", reduce_only=True,
  urgency="immediate")` — EXECUTION's protective-exit gate fires on
  `reduce_only + IMMEDIATE urgency`.
- `action_price_kind="close_if_live_executable"` is a VALIDATION/backtest fill
  convention only. In live trading it must be reconciled with EXECUTION's
  venue-native stop semantics (the venue triggers the resting reduce-only stop);
  TREND must not assume a backtest close fill in live mode. Pin: live exits are
  realized by EXECUTION's venue stop, not by a TREND-assumed close.

### 6b. Hold-horizon and residual-move boundary (LEVERAGE / COST consume)

TREND is the caller that supplies the holding-horizon and residual-move inputs
the sizing and cost pillars need for an OPEN position. Pin who owns what:

- LEVERAGE.md sizes the stop as `sigma * sqrt(T)` and needs the holding horizon
  `T` "from the caller." For an open position TREND is that caller. TREND does
  NOT publish a point `T` (the live horizon `tau_live` is a stopping time, not a
  scalar — Section 1.4). It publishes `HoldHorizonEstimate`, a conservative
  quantile/RANGE of remaining hold derived from the survival law (Section 13a)
  conditioned on `cap_fraction_used`. LEVERAGE consumes the range read-only.
- COST.md's funding gate needs a `HoldHorizon` (`max_hold_quantile_ms`, `m_max`);
  TREND supplies it as `HoldHorizonEstimate.max_hold_quantile_ms`. COST's
  `hold_carry` input needs `expected_residual_move` "supplied by TREND" (COST
  8.2, lines 1180-1184); TREND supplies `ResidualMoveEstimate.point_used_by_cost`.
- `expected_residual_move` is TREND-OWNED edge. COST and LEVERAGE consume it
  read-only; they must not re-derive or override it. Consistent with the
  governing principle that drift is unmeasurable (Section 0), it is a band with a
  single conservative point, never a confident point forecast.

### 6c. Entry boundary: TREND consumes BREAKOUT's entry, never re-derives it

TREND consumes an already-open `Position(side, entry_idx, entry_price,
initial_invalidation, initial_risk)` and never recomputes an entry signal, so
there is no double-counting of the breakout entry. Name the source contract
explicitly so the vol/swing fallback cannot be mistaken for a second entry model:

- `initial_invalidation`, `initial_risk`, and the implied entry stop-distance
  come from BREAKOUT.md's entry gate. They are inputs to TREND, owned by BREAKOUT.
- `ConvexExit.initial()` (Section 10) uses BREAKOUT's `initial_invalidation` when
  supplied, and falls back to a causal vol/swing stop ONLY when it is absent
  (e.g. a position handed in without an upstream invalidation). The fallback is a
  degenerate convenience, not an entry model.
- TREND must NOT re-derive, second-guess, or override BREAKOUT's entry timing or
  initial invalidation. It manages the trend life and convex exit of a position
  whose existence and initial risk BREAKOUT already decided.

---

## 7. Invariants

These are structural rules and must be enforced by tests.

### Invariant A - estimation emits beliefs, policy emits actions

Nothing under `estimate/` emits continue/reduce/exit. Only
`policy/decision.py` emits `ActionDecision`.

### Invariant B - exactly one deterioration detector

Configure exactly one of SPRT or CUSUM. Do not run both and reconcile them. Do
not add a second detector for the same latent quantity.

### Invariant C - no RED/BLUE in `InstrumentSpec`

`InstrumentSpec` is AMBER metadata only. No trail multipliers, ARL, alpha/beta,
Kalman ratio, horizon, or policy threshold may be stored there.

### Invariant D - no claimed drift posterior

Do not output `P(trend_alive)` or `P(mu > 0)`. The module outputs filtered
slope, variance-integrated drift z, sequential statistic, jump flags, and
hazard estimates. A posterior is a conceptual guide, not a reported number.

### Invariant E - Kalman noise ratio is a class preset, not fitted per symbol

The Kalman process/observation noise ratio may be set by a validated
asset-class preset. It must not be optimized per symbol and must not adapt
online from recent PnL.

### Invariant F - diagnostics are log-only

`BeliefDiagnostics` may be logged and tested, but policy decisions must not
change when diagnostics are replaced with arbitrary values.

### Invariant G - validation labels are firewalled

Future labels, MFE, MAE, capture ratio, and future returns live only under
`validate/`. Live modules must not import `validate.labels`.

### Invariant H - no lookahead

Every belief and decision at bar t uses only data available at or before t.
Future bars can be used only by validation labels and reports.

### Invariant I - batch/live parity

Batch replay and one-bar online update must produce the same decisions after
warm-up, within deterministic numerical tolerance.

### Invariant J - action timing is explicit

Every decision includes `decision_idx`, `action_idx`, and `action_price_kind`.
Validation must score from `action_idx`, never from an unavailable price.

### Invariant K - dynamic horizon is a causal stopping time

The live trend horizon is `tau_live = min(tau_death, tau_cap)`, where
`tau_death` is triggered only by `F_t`-measurable death events and `tau_cap` is
the maximum observation or operational cap. The 7-day cap is not trend death by
default. Future MFE, future MAE, future returns, and future volatility may not
enter the live horizon. A time-cap exit must be reason-coded as `time_cap`; a
survivor at the validation cap must be labeled `censored`.

TEST: changing bars after t cannot change whether `tau_live <= t`.

### Invariant HC - operational cap is the SYSTEM-injected contractual bound

The operational cap `tau_cap = H_op` in `ResolvedConfig` is supplied by SYSTEM as
`H_sys` (SYSTEM.md Invariant HC), a RED operational value, not a TREND-internal
constant. The engine MUST force-emit `stop_cause="time_cap"` at the first bar
where the active clock reaches `H_sys`, so no realized hold exceeds the
contractual bound that bounds SYSTEM's integrated-validation embargo. The
observation cap `H_obs` used for survival analysis may be larger; `H_op`-forced
exits are censoring (not deaths) for trend-death survival curves.

TEST: with `H_sys` injected, a still-alive synthetic trend is force-marked
`time_cap` exactly at the bar the active clock crosses `H_sys`, and no decision
in the run carries a realized hold beyond `H_sys`.

### Invariant L - realized payoff is option-like (convexity is shown, not asserted)

The convex-exit schedule must satisfy INV-CVX (Section 1.5a): the round-trip
payoff `Pi(M)` is bounded below by the initial-risk cut `-R0` and has marginal
giveback `g'(M)` non-increasing over the loose/tail-preservation regions, so
`Pi(M)` is convex in the favorable move `M`. The MFE-capture ratio is the
empirical estimate of the lookback-straddle-leg fraction retained (Section 1.5b).

TEST: on synthetic persistent trends, fit `Pi` vs realized `M`; the asymmetric
schedule produces a convex (option-like) fit with bounded-below loss, while the
symmetric-trail baseline produces an affine (non-convex) fit. A schedule whose
`g'(M)` increases with `M` over the tail region fails.

### Invariant M - convexity is gated NET of cost, not gross

Positive skew and MFE capture are gross diagnostics and are NOT sufficient. The
convex-exit and promotion gates require measured net expectancy `E[Pi] > 0`
(Section 1.5c) with `premium_bleed` reported alongside tail capture, and with
costs sourced from COST.md (Section 6.1), not the local `CostModel` stub.

TEST: a synthetic regime where gross skew is positive but `premium_bleed`
exceeds right-tail capture must be REJECTED by the gate (the net-negative case of
the Section 1.5c worked example).

### Invariant N - TREND owns exit LEVEL/intent; EXECUTION owns order placement

TREND never places, amends, or cancels venue orders. It emits an
`ExecutionIntent` (level + reduce_only + urgency + kind) on every
`ActionDecision`; a trail move emits `kind="amend_stop"`, a hard exit emits
`kind="exit_reduce_only", urgency="immediate"` (Section 6a). EXECUTION places /
amends / reconciles the venue-resident reduce-only stop.

TEST: no code under `policy/` or `engine.py` calls an order-placement API;
a trail-level change between bars produces an `amend_stop` intent, and any
D-event/`time_cap` exit produces a reduce-only IMMEDIATE intent.

### Invariant O - entry/invalidation come from BREAKOUT and are never overridden

TREND consumes `Position` from BREAKOUT and never recomputes an entry signal.
`ConvexExit.initial()` uses `position.initial_invalidation` when supplied and
only falls back to a causal vol/swing stop when it is absent (Section 6c). TREND
must not re-derive or override BREAKOUT's entry/initial-invalidation logic.

TEST: when `initial_invalidation` is supplied, `ConvexExit.initial()` returns it
unchanged (the vol/swing fallback is not invoked); the fallback path fires only
when `initial_invalidation is None`.

### Invariant P - TREND publishes hold-horizon and residual-move outputs

Every `ActionDecision` carries `HoldHorizonEstimate` (a conservative
quantile/range, never a point scalar) and `ResidualMoveEstimate` (a band with one
conservative point), the TREND-owned inputs LEVERAGE and COST consume read-only
(Section 6b).

TEST: `hold_horizon` is a range with `lo <= max_hold_quantile_ms` and
`op_cap_remaining_ms = H_sys - elapsed`; `residual_move` is a band, not a point;
neither uses future bars.

---

## 8. Data Layer Contracts

### data/schema.py

```python
class BarValidator:
    def validate(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Return bars with is_valid flags.

        Check sorted timestamps, duplicate bars, OHLC consistency, negative or
        zero prices, missing volume where required, stale prices, bad highs/lows,
        and impossible jumps created by corporate actions or futures rolls.
        Do not silently forward-fill tradable prices.
        """
```

### data/calendar.py

```python
class Sessionizer:
    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Attach session-open flags and active-minute/session metadata.

        Must support US equity RTH/ETH, futures sessions, metals sessions, and
        crypto 24/7 UTC segmentation.
        """
```

### data/timing.py

```python
class ActionTimer:
    def action_index(self, decision_idx: int, bars: Bars,
                     policy: str) -> tuple[int, str]:
        """Return action_idx and action_price_kind.

        Default policy: if a decision is known after bar close, action at next
        valid bar. Any same-bar policy must be explicitly marked executable.
        """
```

### data/actions.py

```python
class CorporateActionGuard:
    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Flag or adjust equity split/dividend artifacts.

        A split adjustment must not look like a trend death or a jump.
        """
```

### data/rolls.py

```python
class FuturesRollGuard:
    def apply(self, bars: Bars, spec: InstrumentSpec) -> Bars:
        """Flag roll windows and protect against stitch artifacts.

        Roll artifacts must not trigger trend death by themselves.
        """
```

---

## 9. Estimation Layer Contracts

### estimate/volatility.py

```python
class RobustVolatility:
    def update(self, bars: Bars, idx: int, position: Position,
               cfg: "ResolvedConfig") -> VolState:
        """Return robust causal volatility state.

        Implement a cheap-compute vol stack:
        - EWMA close-to-close log-return volatility.
        - MAD/median robust volatility fallback.
        - Yang-Zhang-style OHLC estimate only when OHLC quality is good.
        - Hybrid method that chooses/fuses methods by data quality.

        The output includes per-bar sigma and integrated variance since entry:

            integrated_var_since_entry = sum_{i=entry_idx+1..idx} sigma_i^2

        Session gaps contribute to gap_var once at the boundary. No full-sample
        volatility, no future quantiles.
        """
```

Why this matters: the old single-current-sigma drift z is too fragile. A trend
held through volatility regimes must be normalized by the variance accumulated
over the path, not by only the current bar's volatility.

### estimate/drift.py

```python
class VarianceIntegratedDrift:
    def update(self, bars: Bars, idx: int, position: Position,
               vol: VolState) -> tuple[float, float, float]:
        """Return (z_drift, favorable_excursion_r, adverse_excursion_r).

        z_drift =
            side * log(price_t / entry_price)
            / sqrt(max(vol.integrated_var_since_entry, eps))

        favorable/adverse excursion are measured in initial-risk units when
        initial_risk exists, otherwise in robust volatility units.
        """
```

### estimate/horizon.py

```python
class DynamicHorizon:
    def update(self, bars: Bars, idx: int, position: Position,
               clock: ClockState, vol: VolState,
               z_drift: float,
               favorable_excursion_r: float,
               adverse_excursion_r: float,
               exit_state: "ExitState | None",
               cfg: "ResolvedConfig") -> HorizonState:
        """Return causal dynamic horizon state.

        This module computes the path state used by the stopping-time contract
        in Section 1:
        - active clock and elapsed value;
        - cap remaining and cap fraction used;
        - integrated variance since entry;
        - running favorable excursion M_t;
        - running giveback G_t;
        - whether the cap is near/binding.

        It does NOT decide continue/reduce/exit by itself. It only makes the
        horizon state explicit so DecisionPolicy can apply the fixed priority
        order. All outputs are F_t-measurable; no future MFE/MAE labels.
        """
```

### estimate/kalman.py

```python
class FixedPresetLocalTrend:
    def __init__(self, noise_ratio: float): ...

    def update(self, price_log: float, sigma_log: float) -> tuple[float, float]:
        """Causal local-level/local-slope filter.

        noise_ratio is selected from a validated asset-class preset. It is not
        estimated per symbol and not changed online. Observations should be
        volatility-normalized or use observation variance from vol state.
        """
```

Allowed noise-ratio preset groups:
- `equity_liquid`
- `equity_illiquid`
- `futures_macro`
- `futures_commodity`
- `precious_metal`
- `industrial_metal`
- `crypto_major`
- `crypto_alt`

If a preset fails batch validation, it may be revised at class level only.

### estimate/sequential.py

```python
class SequentialDeterioration:
    def __init__(self, kind: DetectorKind, cfg: "ResolvedConfig"): ...

    def update(self, side_adjusted_std_return: float,
               vol: VolState) -> tuple[float, bool]:
        """Return (seq_stat, seq_alarm).

        SPRT mode:
        - H0: drift remains economically favorable.
        - H1: drift is dead or unfavorable.
        - alpha/beta are RED or class preset.

        CUSUM mode:
        - One-sided deterioration detector.
        - Threshold from target ARL under empirical null.

        Null calibration must use Student-t or empirical residuals, not naive
        Gaussian critical values.
        """
```

Default recommendation for cheap compute: start with CUSUM. Keep SPRT as a
validated alternative. Do not run both live.

### estimate/jump.py

```python
class JumpDeterioration:
    def update(self, bars: Bars, idx: int, side: Side,
               vol: VolState, cfg: "ResolvedConfig") -> tuple[bool, bool]:
        """Return (jump_against, jump_with).

        Use bipower/Lee-Mykland-style jump logic when the data quality supports
        it. If OHLC or local volatility quality is poor, use robust standardized
        return exceedance with empirical calibration.

        A jump against the position is a candidate hard exit. A jump with the
        position widens the trail and does not exit by itself.
        """
```

### estimate/hazard.py

```python
class TrendDeathHazard:
    def fit(self, events: "EventTable", cfg: "ResolvedConfig") -> None:
        """Fit the event-count fallback ladder.

        Rungs:
        - disabled: default if events are insufficient or validation fails.
        - constant: fewer than about 15 events.
        - single: about 15-50 events; one covariate only.
        - state: 50+ events; 3-5 interpretable covariates.

        The richer rung must beat constant hazard out of sample. If it does
        not, use the simpler rung.
        """

    def update(self, state: Mapping[str, float]) -> tuple[float, str]:
        """Return (hazard, rung). Hazard can reduce, not hard-exit, unless the
        policy explicitly allows a validated class-level hard-exit rule."""
```

Default recommendation: hazard is disabled until there are enough labeled trend
death events. This keeps the v1 cheap and avoids fake sophistication.

### estimate/state.py

```python
class BeliefBuilder:
    def update(self, bars: Bars, idx: int, position: Position,
               cfg: "ResolvedConfig") -> BeliefState:
        """Build the full causal BeliefState for one bar.

        No future labels. No policy action. Deterministic given prior state and
        bars up to idx.
        """
```

---

## 10. Policy Layer Contracts

### policy/convex_exit.py

```python
class ConvexExit:
    def initial(self, bars: Bars, position: Position, vol: VolState,
                cfg: "ResolvedConfig") -> ExitState:
        """Create initial invalidation and trail state.

        Use position.initial_invalidation when supplied by the entry module. If
        absent, use a causal volatility/swing fallback. Initial risk must exceed
        tick/spread noise floor.
        """

    def update(self, bars: Bars, idx: int, position: Position,
               belief: BeliefState, state: ExitState,
               cfg: "ResolvedConfig") -> ExitState:
        """Update asymmetric convex trail.

        Required shape:
        - Tight early trail to cut failed trades.
        - Wider trail as open_profit_r grows.
        - Additional widening after jump_with or volatility expansion if trend
          evidence has not deteriorated.
        - Never tighten because of future MFE labels.
        """

    def breached(self, bars: Bars, idx: int, position: Position,
                 state: ExitState) -> bool:
        """Return whether the causal trail/invalidation was breached."""

    def gap_survival_fraction(self, vol: VolState,
                              cfg: "ResolvedConfig") -> float:
        """Return max size fraction that survives an adverse gap stress.

        This is not a portfolio sizing engine. It only prevents the trend-life
        policy from pretending that stops protect through gaps.
        """
```

Trail schedule must be monotone in open profit:

```text
profit_r <= 0.5: tight multiple
0.5 < profit_r <= 2.0: normal multiple
profit_r > 2.0: loose multiple
profit_r > 4.0: tail-preservation multiple
```

Exact multipliers are RED/BLUE presets, not fitted per symbol.

### policy/reduce.py

```python
class ReducePolicy:
    def fraction(self, belief: BeliefState, state: ExitState,
                 cfg: "ResolvedConfig") -> tuple[float, tuple[str, ...]]:
        """Return reduced size fraction and reason codes.

        Reduce can use hazard, slope flip, gap survival, and severe volatility
        expansion. It should not hard-exit unless DecisionPolicy priority says
        so. This keeps hazard from becoming an underpowered exit classifier.
        """
```

### policy/decision.py

```python
class DecisionPolicy:
    def decide(self, bars: Bars, idx: int, position: Position,
               belief: BeliefState, exit_state: ExitState,
               cfg: "ResolvedConfig") -> ActionDecision:
        """Emit the only live action.

        Fixed priority:
        1. invalid data / halted / untradeable state -> continue or reduce,
           never pretend an executable exit exists.
        2. jump_against -> exit at next actionable bar.
        3. sequential deterioration alarm -> exit.
        4. trail/invalidation breach -> exit.
        5. operational time cap reached -> exit with `time_cap`, not trend death.
        6. hazard/slope/gap survival -> reduce.
        7. otherwise -> continue.

        This method is responsible for setting `stopping_time_fired` and
        `stop_cause` on the ActionDecision. HorizonState supplies causal
        path/cap state; DecisionPolicy supplies the final stopping event.

        Diagnostics must not alter the decision.
        """
```

---

## 11. Engine and Online State

### engine.py

```python
class TrendLifeEngine:
    def run_batch(self, bars: Bars, position: Position,
                  spec: InstrumentSpec,
                  cfg: "ResolvedConfig") -> list[ActionDecision]:
        """Run historical replay with causal state only."""

    def update(self, bar: Bars, position: Position,
               spec: InstrumentSpec,
               cfg: "ResolvedConfig") -> list[ActionDecision]:
        """One-bar live update. Must match run_batch after warm-up."""
```

### online/snapshot.py

```python
class TrendStateSnapshot:
    def snapshot(self) -> bytes:
        """Serialize estimator, trail, and timing state."""

    @classmethod
    def restore(cls, payload: bytes) -> "TrendStateSnapshot":
        """Restore deterministic state."""
```

Batch/live parity is a release gate.

---

## 12. Validation Layer

Validation is part of the model. Without it, the module is just a collection of
plausible exits.

### validate/labels.py

```python
class TrendLifeLabeler:
    def label(self, bars: Bars, position: Position,
              decisions: list[ActionDecision],
              cost: CostModel,
              horizon_caps: Mapping[HorizonClock, float],
              clocks: tuple[HorizonClock, ...]) -> list[dict]:
        """Create future labels for evaluation only.

        Labels include:
        - tau_live under the dynamic stopping-time definition.
        - observed death definition or censoring reason.
        - cap_fraction_used and cap_binding flag.
        - final net PnL from action_idx.
        - MFE and MAE after each decision.
        - MFE capture ratio.
        - round-trip giveback.
        - time-to-exit.
        - gap-through-stop events.
        - trend-tail capture up to each horizon cap.

        If a position is still alive at the cap, label it censored. Do not
        convert the cap into trend death unless the live policy emitted a
        `time_cap` action.
        """
```

### validate/baselines.py

```python
class BaselineSuite:
    def evaluate_all(self, bars: Bars, position: Position,
                     spec: InstrumentSpec, cost: CostModel,
                     splits: "SplitPlan") -> list["Report"]:
        """Evaluate all baselines under identical labels and timing.

        Required baselines:
        - plain volatility-normalized trailing stop.
        - fixed ATR/Chandelier trail.
        - symmetric trailing stop.
        - high-watermark drawdown stop.
        - time stop.
        - SuperTrend-style baseline.
        - no-sequential-detector ablation.
        - no-Kalman ablation.
        - no-jump-override ablation.
        - no-hazard baseline.
        """
```

### validate/walkforward.py

```python
class WalkForward:
    def evaluate(self, bars: Bars, position_sampler: "PositionSampler",
                 spec: InstrumentSpec, cfg: "ResolvedConfig",
                 cost: CostModel, splits: "SplitPlan") -> "Report":
        """Purged, embargoed validation.

        All TEAL fits, null calibrations, hazards, and class-preset choices in
        a fold must be trained only on permitted training history.
        """
```

### validate/ablation.py

```python
class AblationRunner:
    def run(self, bars: Bars, positions: list[Position],
            spec: InstrumentSpec, cfg: "ResolvedConfig",
            cost: CostModel, splits: "SplitPlan") -> list[dict]:
        """Measure component contribution.

        Required ablations:
        - CUSUM vs SPRT, one live at a time.
        - sequential detector on/off.
        - Kalman on/off.
        - jump override on/off.
        - hazard disabled/constant/single/state.
        - asymmetric trail vs symmetric trail.
        - Yang-Zhang vs EWMA vs MAD vs hybrid volatility.
        - class Kalman preset sensitivity.
        """
```

### validate/stats.py

Required statistics:
- Block-bootstrap confidence intervals.
- Median and tail metrics, not only mean/Sharpe.
- Multiple-test correction for BLUE grid trials.
- Report whether one symbol, one fold, or one asset class explains the result.

---

## 13. Required Metrics

Headline metrics:
- Net PnL after costs from `action_idx`.
- MFE capture ratio.
- Giveback ratio: surrendered PnL / prior MFE.
- Positive skew.
- CVaR-95 and worst gap-through-stop loss.
- Trend-life survival curve and quantiles (median, p90, p95, p99) per death
  definition and horizon clock, censoring-correct, with CIs and cap-domination
  flags (Section 13a). Bare median hold duration is a degenerate read of this.
- Premature exit rate.
- Late exit rate.
- Round-trip rate: profitable open PnL later exits flat/negative.
- Tail capture: percentage of 7-day favorable trend retained.
- Net expectancy `E[Pi]`, with `p_win`, `avg_win`, `p_loss`, `avg_loss`
  (Section 1.5c), all NET of cost via COST.md estimators.
- Premium bleed: cumulative cost of tight-early cuts that did not develop
  (initial-risk cuts + spread/slippage/fees/whipsaw), reported ALONGSIDE tail
  capture so the right tail and the cost of buying it appear in the same table.
- Exit reason distribution.
- Reduce reason distribution.
- Batch/live parity status.

Regime breakdowns:
- Asset class.
- Symbol.
- Volatility bucket.
- Liquidity bucket.
- Session segment.
- Gap vs no-gap.
- Roll window vs normal futures session.
- Earnings/event window vs normal equity session.
- Crypto weekday/weekend and funding regime.

Promotion gate:
- Beats plain volatility trail, Chandelier/ATR, symmetric trail, and time stop
  on median MFE capture and giveback ratio.
- Positive NET-of-cost expectancy `E[Pi] > 0` (Section 1.5c), with costs from
  COST.md (Section 6.1) — NOT the local placeholder, and NOT merely gross
  giveback/MFE capture. A finalist that beats baselines on gross capture but is
  net-negative once premium bleed is charged is REJECTED (Invariant M).
- Does not worsen CVaR-95 beyond configured tolerance.
- Asymmetric trail improves positive skew versus symmetric trail AND its realized
  payoff is convex/option-like (INV-CVX, Section 1.5a).
- Sequential detector improves giveback or late-exit rate without excessive
  premature exits.
- Jump override improves gap/jump loss distribution net of premature exits.
- Hazard is enabled only if it beats disabled/constant hazard out of sample.
- Batch/live parity, leakage tests, and label firewall pass.

---

## 13a. Trend-Life Duration and Tail Survival (p95 / p99)

This section formalizes "how long does an intraday breakout trend stay alive,"
including the p95 and p99 of trend life, and answers the binding question for
the 7-day cap: *does the cap amputate the right tail that produces the
convexity?* It lives in `validate/survival.py` and is a measured OUTPUT of the
backtest, not a live estimator. It upgrades the bare "median hold duration"
metric (Section 13) into a censoring-correct survival curve.

### 13a.1 Why a naive percentile is wrong

The intuitive move — collect durations, take the 95th/99th percentile — is
biased by construction here, for two compounding reasons:

1. **Right-censoring by the 7-day cap.** Every trade still alive at the horizon
   cap H is not "dead at H"; its true death time is unknown and `> H`. Treating
   capped survivors as deaths at H truncates exactly the long-lived trades that
   define the tail, so the empirical p95/p99 is biased DOWNWARD — the larger the
   true tail, the worse the bias. This is the single most important point in the
   section: **the quantity you asked for (p99) is the quantity most corrupted by
   the cap, so it must be estimated with a censoring-aware method or not at
   all.**

2. **Definitional dependence.** "Trend death" is not unique. The duration
   distribution — and therefore its p95/p99 — depends entirely on the death
   rule. The module already exposes several candidate death events; the survival
   analysis must be run per death definition and the definition recorded with
   every reported percentile (see 13a.3).

A third, quieter bias is **competing risks**: the 7-day time stop, a jump-against
exit, and a deterioration exit are different exit causes. If you want the
distribution of *trend* death specifically, exits caused by the time cap or by a
mechanical stop are censoring events for the trend-death clock, not trend
deaths. Conflating them contaminates the estimate.

### 13a.2 The estimator: Kaplan-Meier with confidence bands

Let each historical position i contribute a duration `T_i` (in the active
horizon clock, Section 1) and an event indicator `delta_i in {0, 1}`:
`delta_i = 1` if trend death was OBSERVED at `T_i`, `delta_i = 0` if the
observation was CENSORED at `T_i` (capped at H, or exited for a competing-risk
reason being treated as censoring).

Order the distinct observed death times `t_(1) < t_(2) < ... < t_(k)`. At each
`t_(j)` let `d_j` = number of deaths at `t_(j)` and `n_j` = number at risk
(positions still alive and uncensored just before `t_(j)`). The Kaplan-Meier
estimate of the survival function `S(t) = P(trend life > t)` is

```
              ___
S_hat(t) =    | |        ( 1 - d_j / n_j )
            t_(j) <= t
```

Survival quantiles are read off the inverse:

```
p-th quantile of trend life  =  min { t : S_hat(t) <= 1 - p }
median  = q at S_hat = 0.50
p95     = q at S_hat = 0.05      # 95% of trends have died by here
p99     = q at S_hat = 0.01      # 99% of trends have died by here
```

Variance of `S_hat(t)` uses Greenwood's formula

```
Var[S_hat(t)] = S_hat(t)^2 * sum_{t_(j) <= t}  d_j / ( n_j * (n_j - d_j) )
```

and confidence intervals for `S_hat(t)` should be computed on the
complementary-log-log (log-minus-log) scale so the band stays inside (0, 1):

```
CI on  log(-log S_hat(t)) , then invert.
```

Quantile confidence intervals (the CI on p95/p99 itself) follow by inverting the
survival-band: the CI for the p-quantile is the set of t whose `S_hat` band
contains `1 - p`. Report these bands always — a p99 without a band is
meaningless at realistic sample sizes (13a.4).

Two small-sample facts close the "p99 without a band is meaningless" rule
mathematically; pin both:

1. **One-sided / unbounded inverted-band CI.** The inverted-band quantile CI is
   the set `{ t : S_hat band at t contains 1 - p }`. When the log-log band never
   drops to `1 - p` (the band's lower edge stays above `1 - p` out to the last
   observed time — the typical p99 case under heavy tail censoring), the upper
   endpoint of the quantile CI is `+inf`: the data cannot place an upper bound on
   the quantile. The CI must then be reported as a **one-sided / censored lower
   bound** `p_q >= t_last` (or `>= H`), which is exactly the `cap_dominated`
   lower-bound outcome of 13a.4. State the linkage explicitly: an unbounded
   inverted band IS the band-level signature of the event-count/censoring guard
   firing; the two rules must agree and a `SurvivalResult` may not report a
   two-sided CI on a quantile whose band is one-sided.

2. **Greenwood unreliability in the extreme tail.** Greenwood's variance
   `S_hat(t)^2 * sum d_j/(n_j(n_j - d_j))` is driven by the `1/(n_j(n_j - d_j))`
   terms, which blow up as the at-risk count `n_j` shrinks toward the tail —
   precisely the region where p99 lives. So the Greenwood SE (and any band built
   from it) is itself unreliable exactly where p99 sits, independently of the
   censoring bias. This is a second, variance-side reason for the event-count
   gate of 13a.4: even with no censoring, tiny `n_j` makes the tail band
   uninformative. When `n_j` at the relevant time falls below the configured
   minimum (the same threshold backing the p95/p99 event-count gate), the band is
   flagged `tail_band_unreliable` and the quantile is downgraded to a lower bound.

A brief Brownian-bridge / asymptotics note: the log-log band and Greenwood SE
are asymptotic (large-`n_j`) approximations whose coverage degrades in the tail;
for the extreme tail prefer the event-count gate and the censored lower bound
over trusting a nominally-95% band that has poor finite-sample coverage where
`n_j` is single-digit. Do not paper over this with the parametric tail (13a.5)
unless its goodness-of-fit gate passes.

### 13a.3 Death definitions (run the analysis per definition)

Estimate a separate survival curve for each, using only causal, already-defined
module quantities; record which definition produced each percentile:

- **D1 — trailing-stop breach** (the `convex_exit` trail is hit). Practical,
  policy-dependent.
- **D2 — sequential deterioration alarm** (`seq_alarm`, the CUSUM/SPRT death).
  Closest to a statistical "drift no longer favorable."
- **D3 — Kalman slope sign flip** (`filtered_slope` crosses zero against the
  position). A smoothed structural-reversal definition.
- **D4 — fixed retracement from peak** (giveback of a set fraction of MFE).
  Policy-light, comparable across assets.
- **D5 — jump-against** (`jump_against`). A discontinuous-death definition;
  usually rare, treated as a competing risk for the other clocks.

For each definition, exits caused by the OTHER mechanisms (and by the H cap) are
treated as censored, not as deaths, so the curve isolates that one cause.

#### 13a.3a The independent-censoring assumption (and why it is suspect here)

Treating the OTHER death causes as censoring and running ordinary Kaplan-Meier
does NOT estimate the cause-specific quantity for free. KM under cause-specific
censoring estimates a marginal "net" survival `S_cause(t)` ONLY under the
**independent (non-informative) competing-risks assumption**: the time-to-cause-c
and the censoring time (here, time-to-some-other-cause) must be independent given
the covariates. Pin that this assumption is REQUIRED and is exactly the
assumption violated across D1..D5:

- A jump-against (D5) exit is mechanically correlated with an imminent trail
  breach (D1): the same adverse move that gaps through the trail is the move that
  trips the jump detector. So censoring D5 when estimating the D1 curve is
  informative — the censored D5 trades are disproportionately the ones about to
  die by D1. Independent censoring fails.
- D2 (deterioration alarm) and D3 (slope flip) track the same latent drift decay,
  so each censors the other informatively.

Consequences and the required handling:

1. State, in the report, the independent-censoring assumption KM rests on, and
   that it is suspect for any cause-specific (D1..D5) curve where another cause
   can censor it.
2. For the cause-specific quantity, prefer the assumption-FREE alternative: the
   **cumulative incidence function (CIF)** estimated via **Aalen-Johansen** from
   the cause-specific hazards. The CIF `F_c(t) = P(die of cause c by t, in the
   presence of the other causes)` does not require the causes to be independent;
   it is the correct estimand when competing risks are present. Aalen-Johansen is
   a non-parametric, closed-form estimator (cause-specific Nelson-Aalen hazards
   composed into the CIF) — it satisfies the no-ML ethos exactly as KM does.
3. Alternatively, restrict the ordinary KM curve to the **all-cause death
   definition** (union of D1..D5, i.e. `tau_death` of Section 1.4). For the
   all-cause curve the ONLY censoring is the observation cap `H_obs` (and genuine
   data censoring), which IS non-informative about the all-cause death clock, so
   independent censoring genuinely holds and the KM marginal is valid.
4. The report must FLAG, per curve, which estimand it is and whether it rests on
   the independence assumption: tag each survival curve `km_allcause`
   (independence holds), `km_causespecific_independence_assumed` (suspect — KM
   net survival, caveat printed), or `aj_cif` (assumption-free cumulative
   incidence). A cause-specific p95/p99 read off a `km_causespecific...` curve
   must carry an `independence_assumed` flag.

Recommended default: report the all-cause KM curve (3) for the headline trend-
life quantiles, and the Aalen-Johansen CIF (2) when a per-cause decomposition is
needed. Reserve cause-specific KM (1, net survival) for diagnostic comparison
only, always flagged. This keeps the headline number on solid ground and keeps
the decomposition assumption-free.

### 13a.4 The event-count honesty rule (this governs whether p99 is reportable)

Tail quantiles need tail data. The same event-count discipline as the hazard
ladder (Section 9) applies, now as a HARD gate on what may be reported:

- **p50 (median):** ~50+ observed deaths for a stable estimate.
- **p90:** ~100+ observed deaths.
- **p95:** ~200+ observed deaths; otherwise report with explicit wide bands and
  a `low_event_count` flag.
- **p99:** several hundred observed deaths AND the H cap must lie well beyond the
  empirical p99 region (low censoring fraction near the tail). If censoring
  dominates the right tail, **p99 is NOT reported as a point estimate** — only a
  lower bound `p99 >= H` is reported, because the data cannot resolve it.

Censoring-fraction guard: if the fraction of positions censored at H exceeds a
configured threshold (e.g. > ~20-30%), the tail is cap-dominated and any p95/p99
must carry a `cap_dominated` flag. Parametric tail extrapolation (13a.5) is the
only way to say anything beyond H, and it is clearly labeled model-dependent.

### 13a.5 Optional parametric tail (extrapolation beyond the cap, clearly flagged)

Because the cap censors the very tail you asked about, a point p99 often requires
extrapolation. If and only if a goodness-of-fit check passes, fit a parametric
survival model to the censored durations via maximum likelihood and read p99
from the fitted model. The censored log-likelihood is

```
log L = sum_i [ delta_i * log f(T_i ; theta) + (1 - delta_i) * log S(T_i ; theta) ]
```

Candidate families and what each assumes about the hazard `h(t) = f/S`:

- **Exponential** (`h` constant): memoryless trend death — `S(t)=exp(-lambda t)`,
  so `p_q = -ln(1-q)/lambda`, giving `p99 = ln(100)/lambda`. Almost certainly
  too simple (it denies duration dependence), but it is the honest floor model
  and a useful sanity check. A constant-hazard fit is the same object the hazard
  ladder falls back to.
- **Weibull** (`h` monotone in age): `S(t)=exp(-(t/scale)^shape)`. `shape>1`
  means death rate rises with trend age (mean reversion of trends); `shape<1`
  means survivors get hardier (the long-runner phenomenon). `p_q = scale *
  (-ln(1-q))^(1/shape)`. This is the default tail model when one is justified.
- **Log-logistic / log-normal** (non-monotone hazard): allow hazard to rise then
  fall, which fits "most breakouts die early, survivors run long."

Selection by AIC/BIC under the censored likelihood, with a survival-plot /
QQ check; if no family fits acceptably, **do not extrapolate** — fall back to the
KM lower bound `p99 >= H`. Report the fitted family, its parameters, and the CI
from the observed information matrix. A parametric p99 is always tagged
`model_extrapolated`.

### 13a.6 Connecting the tail to the convexity thesis (the actual decision)

The reason to compute this is not curiosity about a number; it is to test
whether the 7-day cap clips the right tail that pays for the strategy
(Fung-Hsieh / positive-skew rationale, Section 21). Two regimes:

- **Cap non-binding:** if the KM (or fitted) p95/p99 lie well inside H, the bulk
  AND the tail of trend life resolve before the cap. The 7-day guillotine is
  free; it removes nothing profitable. This is the expected case for typical
  intraday breakouts, whose median life is short (hours to a couple of
  sessions).
- **Cap binding:** if the tail (p90-p99) extends beyond H, or censoring near H
  is heavy, the cap is amputating the longest-lived survivors — precisely the
  trades that generate convexity. The signature in the existing metrics is
  systematically poor `tail capture` and low `MFE capture` concentrated on the
  largest-MFE trades, with a high censoring fraction at H.

Required cross-checks against metrics already in the spec (Section 13):
- Correlate censoring-at-H fraction with `tail capture`. High censoring + low
  tail capture = the cap is the bottleneck, not the exit logic.
- Stratify the survival curve by the regime buckets (asset class, vol bucket,
  gap vs no-gap). Crypto (24/7) vs RTH equities will have materially different
  duration laws; do not pool them — pooling across asset clocks is explicitly
  misleading (Section 1).
- Decompose by death definition (13a.3): if D2 (deterioration) deaths cluster
  much earlier than D1 (trail) deaths, the detector is exiting before the trail
  would, which is the intended left-tail truncation — confirm it is not also
  cutting the right tail.

### 13a.7 Contract

```python
class TrendLifeSurvival:
    def estimate(self,
                 durations: "np.ndarray",          # T_i in the active clock
                 events: "np.ndarray",             # delta_i in {0,1}
                 horizon_cap: float,               # H in the same clock
                 clock: HorizonClock,
                 death_definition: Literal["D1","D2","D3","D4","D5","all_cause"],
                 estimand: Literal[
                     "km_allcause",
                     "km_causespecific_independence_assumed",
                     "aj_cif",
                 ] = "km_allcause",
                 want_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
                 ) -> "SurvivalResult":
        """Kaplan-Meier survival with Greenwood/log-log CIs; quantiles read off
        S_hat with their inverted-band CIs (reported one-sided / as a lower bound
        when the band is unbounded or n_j is tiny in the tail, 13a.2). Returns,
        per requested quantile, either a point estimate with CI, or a censored
        lower bound (>= H) when the event-count / censoring / band guards (13a.4,
        13a.2) are not met. For cause-specific (D1..D5) estimands, prefer the
        assumption-free Aalen-Johansen cumulative incidence (estimand='aj_cif');
        ordinary cause-specific KM is offered only flagged
        'km_causespecific_independence_assumed' (13a.3a). The headline
        all-cause curve uses estimand='km_allcause' where independent censoring
        genuinely holds. Optionally fits a parametric tail (13a.5) under the
        censored likelihood, AIC/BIC-selected and goodness-of-fit gated, tagged
        `model_extrapolated`. Never pools across horizon clocks or asset classes.
        All inputs are validation outputs; this module is never imported by live
        estimation or policy."""
```

`SurvivalResult` must carry, per quantile: value (or lower bound), CI (marked
one-sided when the inverted band is unbounded, 13a.2), the death definition, the
estimand tag (`km_allcause` / `km_causespecific_independence_assumed` /
`aj_cif`, 13a.3a), the clock, the observed-death count, the at-risk count `n_j`
at the quantile time, the censoring fraction near the tail, and the flags
`low_event_count`, `cap_dominated`, `model_extrapolated`, `independence_assumed`,
and `tail_band_unreliable` as applicable. Reporting a bare scalar p99 without
these is forbidden.

### 13a.8 What this does NOT do

- It does not feed any live decision (Invariant: validation-only, like all of
  `validate/`).
- It does not assert a universal p99 across instruments — the curve is
  per-asset-class and per-death-definition, because there is no single
  distribution.
- It does not extrapolate silently — beyond-cap statements are either a labeled
  parametric model or a censored lower bound, never an unflagged point estimate.

---

## 14. Cheap-Compute Design Rules

This module is designed to run on a retail machine.

Rules:
- Prefer O(n) streaming updates.
- Cache empirical null thresholds by asset-class preset, volatility bucket, and
  detector configuration.
- Run heavy ablations overnight, not inside the live loop.
- Use sampled CPCV when full CPCV is too expensive.
- Keep BLUE grids small and finite.
- Pool calibration by asset class and liquidity bucket when symbol history is
  thin.
- Use vectorized numpy/pandas for research runs; use numba only after tests.
- Avoid per-bar optimization.
- Avoid nested rolling regressions across huge parameter grids.

Suggested v1 live path:

```text
data validation
-> session/timing
-> hybrid robust vol
-> variance-integrated drift
-> dynamic horizon state
-> class-preset Kalman
-> one CUSUM deterioration detector
-> jump-against/jump-with
-> hazard disabled or constant
-> convex asymmetric trail
-> decision
```

Suggested v1 research path:

```text
same live path
-> walk-forward
-> baseline suite
-> ablations
-> sampled CPCV for finalists only
-> registry promotion
```

---

## 15. Auto-Tuning and Promotion

Auto-tuning is allowed only as governed preset selection.

### tune/search_space.py

Finite examples:

```python
TRAIL_PRESETS = ["tight", "balanced", "tail_preserve"]
DETECTOR_KINDS = ["cusum", "sprt"]
ARL_TARGETS = [250, 500, 1000, 2000]
VOL_METHODS = ["ewma", "mad", "hybrid"]
KALMAN_CLASS_PRESETS = [
    "equity_liquid",
    "equity_illiquid",
    "futures_macro",
    "futures_commodity",
    "precious_metal",
    "industrial_metal",
    "crypto_major",
    "crypto_alt",
]
HAZARD_MODES = ["disabled", "constant", "single", "state"]
```

### tune/promotion.py

```python
@dataclass(frozen=True)
class PromotionDecision:
    asset_class: str
    symbol: str
    status: Literal["approved", "shadow", "rejected"]
    engine_version: str
    resolved_config_hash: str
    fit_window: tuple[int, int]
    validation_window: tuple[int, int]
    reason: str
    report_summary: Mapping[str, float]
```

Rules:
- New symbols inherit approved class presets and fit TEAL profiles.
- Insufficient data means `shadow`, not `approved`.
- Failed leakage or batch/live parity means `rejected`.
- Symbol-specific success with non-class parameters remains `shadow`.
- Every trial must be recorded with config hash and data window.

---

## 16. Lookahead Surfaces

Human review and explicit tests are required for:

1. Volatility seasonality:
   - No full-sample intraday means.
   - No future quantiles.
   - Session S uses sessions before S only.

2. Volatility and tail fitting:
   - No full-sample sigma.
   - No full-sample Student-t fit.
   - Winsorization thresholds are fold-local or online-recursive.

3. Drift z:
   - Integrated variance uses only bars from entry to current idx.
   - No future realized volatility in denominator.

4. Dynamic horizon:
   - `tau_live` uses only death events observable at or before t.
   - Future MFE/MAE never enters live horizon state.
   - The 7-day cap is treated as censoring in validation unless an operational
     time-cap exit is explicitly configured and reason-coded.

5. Kalman:
   - Filtering only in live path.
   - No smoothing in live path.

6. Sequential detector:
   - Thresholds trained/calibrated only on training folds.
   - Running statistic uses causal increments only.

7. Jump detector:
   - Local vol uses historical window only.
   - Exceedance calibration is fold-local.

8. Hazard:
   - Event labels come from training folds only.
   - Test fold events do not influence rung or coefficients.

9. Action timing:
   - Score from action_idx.
   - No same-bar exit unless explicitly executable.

10. Validation labels:
   - Future labels are under `validate/` only.
   - Live modules cannot import them.

Required tests:
- Future-needle test.
- Fold-isolation test.
- Label-firewall test.
- Batch/live parity test.
- Same-bar timing rejection test.

---

## 17. Synthetic Data Requirements

`synth/generate.py` must create:

- Persistent trend with low noise.
- Persistent trend with volatility expansion.
- Trend that decays to flat.
- Trend that reverses.
- Jump against position.
- Jump with position.
- Gap against position.
- Gap with position.
- Choppy mean-reverting path.
- Blow-off top / capitulation bottom.
- Futures roll artifact.
- Equity split artifact.
- Equity halt and reopen.
- Crypto weekend liquidity regime.
- Crypto funding shock.
- Missing and stale bars.
- RTH/ETH session transition.

Expected behavior:
- Persistent trends are held longer than baseline symmetric trails.
- Decaying/reversing trends exit with lower giveback than plain trail.
- Jump against exits quickly.
- Jump with widens trail, not exit.
- Gap survival prevents impossible stop assumptions.
- Roll/split artifacts do not trigger false death after data guards.
- Future bars do not change past decisions.
- On persistent trends, the realized payoff `Pi` vs favorable move `M` fits a
  convex (option-like) curve with bounded-below loss `-R0` under the asymmetric
  trail, and an affine curve under the symmetric baseline (Invariant L).
- In a regime where gross skew is positive but premium bleed exceeds tail
  capture, the net-expectancy gate `E[Pi] > 0` rejects the schedule
  (Invariant M).

---

## 18. Module Acceptance Gates

### Data layer

Pass when:
- Bad OHLCV, duplicate timestamps, missing bars, stale bars, roll artifacts,
  split artifacts, and halt flags are detected or flagged.
- Session clocks match asset-class expectations.

### Volatility

Pass when:
- No NaN/inf on adversarial synthetic data.
- Integrated variance is monotone nondecreasing after entry.
- Gap variance is applied once at session boundary.
- Hybrid method falls back when Yang-Zhang quality is poor.

### Variance-integrated drift

Pass when:
- Null z distribution is approximately centered under synthetic no-drift.
- Known drift produces correct sign and growing magnitude.
- Volatility regime shifts do not create fake drift by denominator collapse.

### Dynamic horizon

Pass when:
- `tau_live` is a stopping time: changing future bars after t cannot change
  whether the horizon has stopped by t.
- `HorizonState` uses the configured active clock and never assumes a fixed bar
  count for 7 days.
- Persistent synthetic trends have longer realized live horizons than dying
  synthetic trends under the same cap and entry distribution.
- A survivor at the validation cap is labeled `censored`; an operational cap
  exit is reason-coded `time_cap`, not trend death.
- Cap-binding warnings fire when high-MFE trades frequently survive to the cap
  with poor tail capture.

### Kalman

Pass when:
- Causal filter only.
- Class preset is fixed during run.
- Innovation consistency is healthy on synthetic slow-trend data.
- Preset sensitivity report is generated in validation.

### Sequential detector

Pass when:
- Exactly one mode configured.
- Empirical ARL/alpha-beta is within tolerance under empirical null.
- Pure volatility shift does not reliably trigger deterioration.

### Jump detector

Pass when:
- jump_against and jump_with are correctly signed.
- False jump rate is controlled under no-jump synthetic data.
- Detector degrades gracefully when OHLC quality is poor.

### Hazard

Pass when:
- Disabled by default when events are insufficient.
- Event-count ladder selects correct rung.
- Richer rung must beat constant hazard out of sample or be disabled.

### Convex exit

Pass when:
- Trail multiple widens monotonically with open profit.
- Asymmetric trail beats symmetric trail on skew or MFE capture in synthetic
  persistent trends.
- Gap survival flags or reduces size before session boundary stress.
- Stop/trail never violates tick/spread noise floor.
- The realized payoff `Pi(M)` is option-like: bounded below by `-R0` with a
  convex (non-increasing marginal-giveback) right tail, INV-CVX/Invariant L
  (Section 1.5a). The asymmetric schedule fits convex; symmetric fits affine.
- Measured net expectancy `E[Pi] > 0` NET of cost (Section 1.5c), with costs
  sourced from COST.md (Section 6.1) and `premium_bleed` reported alongside
  right-tail capture. A schedule that is convex GROSS but net-negative once the
  premium bleed is charged FAILS (Invariant M). This requirement replaces the
  weaker "asymmetric beats symmetric on skew" — skew is necessary, not
  sufficient.

### Decision policy

Pass when:
- Fixed priority order is exact.
- Diagnostics do not affect action.
- Only this module emits `ActionDecision`.
- action_idx is explicit and valid.

### Engine

Pass when:
- Batch/live parity holds.
- Future-needle test holds.
- Snapshot/restore gives same next decisions.

### Trend-life survival (validate/survival.py)

Pass when:
- On synthetic data with a KNOWN trend-death law and an imposed horizon cap,
  Kaplan-Meier recovers the true median and p90 within CI, while a naive
  cap-as-death percentile demonstrably UNDER-estimates them (the censoring-bias
  test).
- Quantiles whose event-count or censoring guards (13a.4) fail are returned as
  censored lower bounds with the `cap_dominated` / `low_event_count` flags, not
  as point estimates.
- A quantile whose inverted log-log band is unbounded, or whose tail at-risk
  count `n_j` is below the configured minimum, is returned as a one-sided lower
  bound with `tail_band_unreliable` set (Section 13a.2) — the band-level and
  event-count guards must agree.
- Cause-specific (D1..D5) curves are estimated assumption-free via Aalen-Johansen
  cumulative incidence (`estimand="aj_cif"`), OR the headline curve is the
  all-cause definition (`estimand="km_allcause"`) where independent censoring
  genuinely holds; any ordinary cause-specific KM curve carries the
  `independence_assumed` flag (Section 13a.3a). A cause-specific KM quantile
  reported without that flag fails.
- Parametric tail fits run under the censored likelihood, are AIC/BIC-selected
  and goodness-of-fit gated, and every beyond-cap value is tagged
  `model_extrapolated`.
- Curves are never pooled across horizon clocks or asset classes (a pooling
  attempt raises).
- Module is validation-only: an import from `estimate/` or `policy/` fails.

---

## 19. Build Order

1. `types.py`.
2. `synth/generate.py`.
3. `data/schema.py`, `data/calendar.py`, `data/timing.py`.
4. `data/actions.py` and `data/rolls.py` guards.
5. `config/` with TEAL/AMBER/RED/BLUE separation.
6. `estimate/volatility.py`.
7. `estimate/drift.py`.
8. `estimate/horizon.py`.
9. `estimate/kalman.py`.
10. `estimate/sequential.py` with CUSUM default.
11. `estimate/jump.py`.
12. `estimate/hazard.py` with disabled/default fallback.
13. `policy/convex_exit.py`.
14. `policy/reduce.py`.
15. `policy/decision.py`.
16. `engine.py`.
17. `online/snapshot.py` and `online/replay.py`.
18. `validate/labels.py`.
19. `validate/survival.py` (KM all-cause + Aalen-Johansen CIF + censored tail;
    censoring-bias test and independence-flag test on synthetic).
20. `validate/baselines.py`.
21. `validate/walkforward.py`.
22. `validate/ablation.py`.
23. `validate/cpcv.py` sampled mode.
24. `validate/stats.py` and reports.
25. `tune/` promotion and registry.
26. `research/diagnostics.py` last, with import-firewall tests.

Do not proceed to a later module until the current module passes its acceptance
gate.

---

## 20. Minimum V1 Configuration

For the first real implementation, use this deliberately simple setup:

```text
volatility: hybrid EWMA/MAD, Yang-Zhang only when quality allows
drift: variance-integrated z
kalman: class preset, fixed during run
sequential: CUSUM only
jump: robust jump-against/jump-with
hazard: disabled or constant only
trail: asymmetric convex trail
reduce: gap-survival and severe deterioration only
validation: walk-forward + baselines + ablations
```

This v1 is already strong. Add SPRT, state hazard, and richer context only if
ablation earns them.

---

## 21. References and Rationale

The implementation must be self-contained and test-driven. These references
motivate the design:

- Short-horizon drift is hard to estimate; volatility can be estimated much
  more reliably than mean over a few days.
- Yang-Zhang-style OHLC volatility is useful but must be guarded at 1-minute
  frequency because microstructure noise can dominate.
- Sequential detection via CUSUM/SPRT gives known false-alarm operating
  characteristics when calibrated honestly.
- Jump detection should be treated as a sparse event detector, not an all-hours
  generic volatility alarm.
- Trend following is convex only if the exit policy cuts failed trades while
  preserving the right tail; symmetric trails often damage that convexity. This
  is made explicit, not asserted, in Section 1.5a (the round-trip payoff `Pi(M)`
  is option-like: bounded-below loss `-R0`, unbounded convex right tail) and
  Section 1.5b (the Fung-Hsieh lookback-straddle decomposition is derived: the
  trend payoff = long lookback-straddle leg - giveback friction - initial-risk
  cuts on failures). Fung-Hsieh's empirical finding that trend-following return
  series load like a long lookback straddle is exactly this decomposition.
- Convexity is net only if the right-tail capture exceeds the premium bleed —
  the stream of small left-tail cuts and their friction (Section 1.5c). The
  expectancy identity `E[Pi] = p_win*avg_win - p_loss*avg_loss` makes the claim
  falsifiable; gross skew/MFE capture are not sufficient.
- Survival/hazard models are useful only when event counts justify them.
- Trend-life duration is a right-censored quantity: the holding cap censors the
  longest-lived trends, so duration quantiles (especially p95/p99) require
  Kaplan-Meier with Greenwood/log-log bands, competing-risks handling of
  non-trend exits, and parametric (exponential/Weibull/log-logistic) tail fits
  under the censored likelihood for any beyond-cap statement. A naive percentile
  on capped data is biased downward.
- Cause-specific KM under cause-specific censoring estimates a marginal net
  survival ONLY under independent competing risks, which fails across D1..D5
  (e.g. jump-against is not independent of trail breach). The assumption-free
  estimand is the cumulative incidence function via Aalen-Johansen (cause-
  specific Nelson-Aalen hazards composed into the CIF), or restrict KM to the
  all-cause definition where censoring is only the cap (Section 13a.3a). Both are
  non-parametric/closed-form and respect the no-ML ethos.

---

## 22. Final Rule

If a component passes its unit/statistical acceptance gate but fails to improve
net-of-cost convexity, MFE capture, giveback, or tail risk in ablation, remove
or disable it. Complexity must earn its place.

The target module is not fancy. It is causal, cheap, validated, decomposable,
asset-aware, and hard to fool.
