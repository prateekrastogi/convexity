# OPS.md — build spec for the live monitoring / operations / kill-switch pillar

This file is the authoritative build spec. Build to the contracts and invariants
below. Where this file and your own instinct disagree, this file wins. Ask before
deviating from any **Invariant**.

This module is the **live-safety pillar**. The other six pillars decide *what* to
trade, *how much*, *how to fill*, *what it costs*, and *whether the composed thing
had an edge under firewalled validation*. OPS decides exactly one thing, online,
continuously, cheaply, on every live bar:

> Is the LIVE system still drawing from the same distribution SYSTEM.md validated
> it against — the same realized edge-per-trade, hit-rate, trade-frequency, fill
> quality, cost, funding, and vol the deflated-Sharpe gate certified — and if it
> has DIVERGED, detect that with a statistical test whose false-alarm rate and
> detection delay are both pinned, and BRAKE: degrade gracefully, alert by
> severity, and (the hard binary backstop) HALT.

OPS is what makes a validated system safe to run with real capital. SYSTEM proves
an edge exists *in sample-from-the-validated-distribution*; OPS detects, in
production, when the live world stops matching that distribution — a regime the
validation never saw, a venue that degrades, a cost model that was optimistic, an
edge that has decayed — and stops the bleeding before the drawdown governor or the
account does. It is a **Statistical Process Control (SPC)** layer plus a
**circuit-breaker** plus an **alerting/safe-state** policy, built entirely from
classical, causal, O(1)-per-update change-detection mathematics: **no ML**.

OPS sits on top of **NautilusTrader**, which already owns order plumbing,
position/account/portfolio state, **restart reconciliation**, data transport,
backtest↔live parity, pre-trade risk checks, heartbeats/clock, and funding
ingestion. OPS builds the **monitoring + alarm + kill POLICY** on those primitives.
It does **not** rebuild reconciliation, it **monitors** SYSTEM's reconciliation
output; it does **not** re-implement the cost model, it **consumes** COST's TCA
residuals; it does **not** model fills, it **consumes** EXECUTION's IS stream; it
does **not** size, it **reads** LEVERAGE's governor multiplier and complements it
with a binary halt.

Scope: full cross-asset (Binance USDT-M perpetuals 24/7, Dukascopy-proxied
commodities/metals, US/Indian equities and index futures), 1-minute bars,
~$1000 retail book, target compute = a 2-core / 11 GB ARM VM. Retail latency:
seconds, bar-level — never microseconds. Every monitored quantity is
`F_t`-measurable; every control limit is derived OFFLINE from the SYSTEM-validated
distribution and **frozen**, never refit on live P&L.

---

## 0. Governing principles (read before anything else)

**P1 — OPS monitors; it never trades, never sizes, never re-validates.** OPS
consumes the realized streams the other pillars and Nautilus emit, runs control
charts and sequential tests on them, and emits exactly three kinds of output:
**alarms** (severity-tiered), a **safe-state directive** (deterministic
degradation policy), and a **binary kill** (halt). It never opens, closes,
reduces, routes, or re-sizes a position of its own initiative — the kill directive
is consumed by the live loop, which flattens via EXECUTION's venue-resident
reduce-only stops (P6 of EXECUTION). OPS holds **no edge parameter, no sizing
parameter, no exit parameter**. Any alpha/sizing/exit math living in `ops/` is the
bug. (See Invariant PROV.)

**P2 — The control limits are DERIVED from the validated distribution, never
tuned on live P&L (the most important rule in this file).** Every threshold —
every CUSUM slack `k` and decision interval `h`, every Page-Hinkley `δ`/`λ`, every
EWMA/Shewhart limit `L`, every SPRT boundary `(A, B)` — is computed from the
in-control distribution that SYSTEM.md's firewalled validation produced (the mean
and variance of realized edge-per-trade, the fill-IS distribution, the COST TCA
residual distribution, the realized-vol distribution) plus an explicitly chosen
**Average Run Length (ARL)** / false-alarm target. The limits are then **frozen**
as RED/BLUE governed constants and validated OUT-OF-SAMPLE inside SYSTEM's CPCV
framework. They are **never** re-fitted to maximize backtest or live PnL; they are
never moved because an alarm was "annoying." A limit that drifts because someone
re-tuned it on the live P&L it was meant to police is the four-green-modules trap
re-appearing as an ops failure. (See Invariant LIM, OOS.)

**P3 — Causal, online, O(1) per update.** Every detector is a streaming recursion
over the live stream: one add, one compare, one reset — bounded state, no growing
window, no re-fit, no future bar. CUSUM, Page-Hinkley, EWMA, and Shewhart are all
single-scalar-state recursions; the SPRT accumulates one log-likelihood-ratio
increment per observation. The whole OPS layer costs **O(number of monitored
streams)** per bar — a few dozen scalar updates — and is dwarfed by any one
pillar's per-bar evidence. No detector reads a bar it could not have seen at `t`.
(See Invariant CAUSAL, CHEAP.)

**P4 — Three brakes, three distinct jobs, composed by precedence — never summed.**
There are exactly three size/exposure controls in the stack and OPS owns only the
hardest one:

```
REGIME macro multiplier   (gradual, slow)  : tapers GROSS book exposure as the
                                             macro regime turns risk-off. A smooth
                                             scalar in [0,1] on total gross.
LEVERAGE drawdown governor (gradual, medium): tapers PER-POSITION size as equity
                                             draws down (1.00→0.75→0.50→0.25→0.00
                                             at the 43% cliff). A smooth scalar.
OPS kill-switch           (binary, hard)    : HALTS — stops new entries and/or
                                             flattens to venue stops — when a
                                             monitored stream crosses an SPC limit
                                             or a hard backstop. A {run, degrade,
                                             halt} STATE, not a multiplier.
```

These are **not** multiplied into one number and **not** added. They compose by
`min()` on the size path (REGIME × governor is the existing gradual taper, owned by
LEVERAGE/REGIME) and by **precedence** for the binary state: if OPS says HALT, size
is zero regardless of the two gradual scalars; if OPS says RUN, the gradual scalars
govern. OPS never re-implements the governor's ladder and never tapers gross — it
is the discontinuous backstop the two continuous tapers cannot be (a smooth taper
can ride a structural break all the way down; a binary halt cannot). (See Invariant
BRAKE.)

**P5 — Reconciliation is SYSTEM's; OPS monitors its OUTPUT.** Nautilus +
SYSTEM own position/order/account reconciliation (restart, fill-vs-intent,
venue-vs-book). OPS does **not** re-derive the reconciliation; it **subscribes to
the reconciliation result** and treats a reconciliation MISMATCH as a monitored
binary event with a kill trigger. Re-implementing the reconciliation logic in
`ops/` is the bug; consuming `reconciliation_ok: bool` + the mismatch detail and
firing a circuit-breaker on it is the contract. (See Invariant RECON.)

**P6 — Conservative under uncertainty; fail toward SAFE.** A missing heartbeat, a
stale feed, an unparseable fill, a detector whose own diagnostics fail — every one
resolves to the *safer* state, never the permissive one. A monitored stream that
goes dark is treated as a potential break (stop-new-entries / heartbeat trip), not
as "in control by absence of data." A kill never silently un-trips: re-arming is a
deliberate, gated, logged transition (Section 8), never automatic on the next clean
bar. (See Invariant SAFE.)

**P7 — Decomposable, auditable, replayable.** Every alarm carries the stream, the
detector, the statistic value, the limit it crossed, the sample index, and the
in-control parameters it was tested against. Every kill carries the full trigger
chain. The forensic log is append-only and is the input to COST.md's post-hoc TCA
and to SYSTEM's next overnight re-validation. A halt whose cause cannot be replayed
from the log is an un-auditable halt — forbidden. (See Invariant AUDIT.)

**Forbidden in v1:** any ML / learned anomaly detector (P3 — classical SPC only);
any limit fitted or re-tuned on live P&L (P2); OPS emitting a trade/size/exit of
its own edge logic (P1); re-implementing Nautilus/SYSTEM reconciliation (P5);
re-implementing LEVERAGE's governor ladder or REGIME's gross taper (P4); a control
limit refit online from the live stream it polices (P2 — limits are frozen from the
validated distribution); treating a dark/stale stream as in-control (P6); an
auto-re-arming kill (P6); a single blended "health score" product instead of
independent per-stream detectors with precedence (mirror of LEVERAGE P6 / SYSTEM
P2); per-symbol tuning of any RED limit on realized P&L (Section 4).

---

## 1. What OPS is and is not (the module map)

```text
NautilusTrader            -> heartbeat, clock, order/fill events, position/account
                             state, RESTART RECONCILIATION result, funding events.
SYSTEM.md                 -> the VALIDATED in-control distribution (edge/trade,
                             hit-rate, trade-frequency, per-pillar attribution) +
                             the live reconciliation OK/mismatch flag + the live
                             composed decision stream. OPS reads these; SYSTEM owns
                             validation + reconciliation logic.
COST.md                   -> realized TCA: realized round-trip cost & realized
                             funding vs the FORECAST it made. OPS monitors the TCA
                             RESIDUAL as a stream.
EXECUTION.md              -> realized implementation shortfall, realized fill rate,
                             realized adverse selection, the venue-resident
                             reduce-only stops OPS leans on to flatten.
LEVERAGE.md               -> the drawdown governor multiplier (gradual size taper)
                             + the planned-loss / equity-ladder OPS reads as the
                             drawdown reference. OPS's max-DD kill is the binary
                             backstop ABOVE the governor.
REGIME (macro)            -> the gross-exposure macro multiplier (gradual). OPS
                             does not taper gross; it halts.
```

OPS turns those realized streams into a **live in-control / out-of-control verdict
per stream**, an **alarm**, a **safe-state directive**, and a **binary kill**. It
is the only module whose entire job is to compare LIVE to VALIDATED and brake.

OPS is NOT: an alpha source, a sizing engine, an exit engine, an execution
algorithm, a reconciliation engine, the drawdown governor, the regime taper, or the
Nautilus engine. It holds no model parameter that could be tuned for return — only
SPC limits derived from the validated distribution and frozen.

---

## 2. Architectural split

Five layers, kept independent for testability (mirrors the sibling specs):

1. **Stream layer.** Causal adapters that turn the raw events of Nautilus / the six
   pillars into OPS's frozen **monitored-stream** types — one observation per
   stream per event/bar, each `F_t`-measurable, each tagged with its in-control
   reference parameters. No modelling, no decision. (`streams/`.)
2. **Detector layer.** The SPC detectors — CUSUM, Page-Hinkley, EWMA, Shewhart,
   SPRT — each a pure O(1) streaming recursion over one stream, each emitting an
   `in_control: bool` + the statistic + the limit + diagnostics. Each testable in
   isolation; none reads another detector's output. (`detectors/`.)
3. **Reconciliation layer.** The set of monitored streams mapped to the four
   reconciliation families (fills, cost/funding, vol, edge/frequency) of Section 5,
   each binding a detector to a stream with the limits from Section 6.
   (`reconcile/`.)
4. **Brake layer.** The circuit-breaker triggers (Section 7), the alarm severity
   router + dedup (Section 8), the safe-state / graceful-degradation policy (Section
   9), and the binary kill state machine. The ONLY place a directive is emitted.
   (`brake/`.)
5. **Validation/forensics layer.** Offline derivation of every limit from the
   validated distribution, the ARL/false-alarm math, OOS validation of the limits
   inside SYSTEM's CPCV, replay/forensics, and the audit log feeding COST's TCA.
   Never imported by layers 1–4. (`validate/`, `forensics/`.)

Why: a bad vol detector cannot corrupt the cost detector; each stream's alarm is
independent and its limit independently derived; the binding trigger is always
explicit; the kill is one auditable state machine, not scattered `if`s.

The engine wraps a Nautilus actor/strategy: it SUBSCRIBES to order/fill/position
events + the heartbeat + the pillar streams and EMITS an `OpsDirective`
(`{run, degrade, halt}` + alarms + reasons). It never tracks position/account state
itself — it reads Nautilus's `Portfolio`/`Cache`. (Invariant PROV, RECON.)

---

## 3. Directory layout

```text
ops/
+-- config/
|   +-- defaults.py          # OpsDefaults: RED limits (CUSUM k/h, PH delta/lambda,
|   |                        #   EWMA lambda/L, Shewhart L_s (empirical-tail), SPRT alpha/beta,
|   |                        #   ARL targets, AR(p) prewhiten coeffs phi + rho/VIF per stream,
|   |                        #   warm-up N_warm, flatten retry cadence/max,
|   |                        #   heartbeat/staleness/clock-skew windows, DD-kill level)
|   +-- in_control.py        # the FROZEN validated in-control distribution params
|   |                        #   (mu0, sigma0, phi, rho, tail dof/quantile per stream),
|   |                        #   imported FROM SYSTEM's validation report (read-only)
|   +-- asset_class.py       # AssetClassOpsPreset: per-class limit presets (validated)
|   +-- severity.py          # AMBER: severity tiering + paging routes + dedup windows
|   +-- resolve.py           # single merge point (RED unreachable from instrument)
+-- streams/                 # STREAM LAYER — causal adapters, no modelling
|   +-- whiten.py            # causal prewhitening: z_i = (resid - phi*prior)/sigma_inn
|   |                        #   (frozen AR(p) phi §6.7a) OR non-overlap batch means §6.7b
|   +-- fills.py             # EXECUTION realized IS / fill-rate / adverse-sel -> stream
|   +-- cost_tca.py          # COST realized-vs-forecast cost & funding residual -> stream
|   +-- vol.py               # realized vol vs LEVERAGE-assumed vol -> stream
|   +-- edge.py              # realized edge/trade, hit-rate, trade-frequency -> stream
|   +-- recon.py             # SYSTEM/Nautilus reconciliation OK/mismatch -> binary stream
|   +-- heartbeat.py         # Nautilus heartbeat / feed-staleness / clock-skew -> binary stream
+-- detectors/               # DETECTOR LAYER — O(1) streaming recursions, isolated
|   +-- cusum.py             # two-sided tabular CUSUM (Page) + ARL
|   +-- page_hinkley.py      # Page-Hinkley mean-shift test
|   +-- ewma.py              # EWMA control chart + control limits
|   +-- shewhart.py          # Shewhart 3-sigma / individuals chart
|   +-- sprt.py              # Wald SPRT + CUSUM-of-LLR ("CUSUM-SPRT") change-point form (6.5/6.5a)
|   +-- base.py              # Detector protocol: update(x)->DetectorState; reset()
+-- reconcile/               # binds streams to detectors with Section-6 limits
|   +-- streams_map.py       # which detector(s) police which stream, with which limits
+-- brake/                   # BRAKE LAYER — the only place a directive is emitted
|   +-- triggers.py          # circuit-breaker triggers (Section 7), each w/ justification;
|   |                        #   the 2-of-2 cost AND-rule is one combined detector (6.6a)
|   +-- alarms.py            # severity router + dedup + ARL-set paging cadence (Section 8)
|   +-- safe_state.py        # graceful-degradation policy (Section 9), deterministic
|   +-- kill_switch.py       # binary {run,degrade,halt} machine; re-arm gating;
|   |                        #   flatten_issued latch + flatten retry on reject/partial (§9.1)
|   +-- engine.py            # OpsEngine: streams->detectors->triggers->directive; reasons
+-- nautilus/
|   +-- adapter.py           # thin bridge: Nautilus/pillar events <-> OPS streams/directive
|   +-- flatten.py           # issue flatten via EXECUTION venue reduce-only stops (P6 EXEC)
+-- forensics/
|   +-- log.py               # append-only audit log (every alarm/kill, replayable)
|   +-- replay.py            # reconstruct the trigger chain from the log
+-- validate/                # PART OF THE MODEL; never imported by layers 1-4
|   +-- derive_limits.py     # limits FROM the validated distribution + ARL/FAR math
|   +-- arl.py               # Average Run Length / false-alarm-rate calculators
|   +-- oos.py               # OOS validation of limits inside SYSTEM's CPCV framework
|   +-- baselines.py         # no-monitor, naive-3sigma-only, single-detector baselines
|   +-- reports.py
+-- synth/
|   +-- generate.py          # adversarial drift/jump/staleness/recon-break streams
+-- types.py
+-- ops_api.py               # public entry point used by the live loop
tests/
+-- ...                      # unit, firewall, causality, ARL, parity, kill-state, acceptance
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest, NautilusTrader (consumed, not
rebuilt). No ML frameworks. `numba` only after correctness tests pass, and only if
profiling on the target VM shows a hot detector loop (it will not — O(1) recursions
are already trivial). `uv` for env/deps.

---

## 4. Parameter governance

Every OPS parameter is exactly one bucket. OPS holds **no model parameter** — only
SPC limits and the in-control reference. The dangerous bucket is RED (the limits),
because a mistuned limit either floods alarms (and gets ignored) or goes blind.

- **TEAL — fitted per instrument from its own past data, causal.** OPS itself fits
  almost nothing live: the limits are frozen offline (RED) so they cannot drift on
  the stream they police (P2). The one TEAL-ish object is a **logged, diagnostic-only
  live variance estimate** — and here is a trap that must be pinned shut. The
  standardization `z_i = (x_i − μ₀)/σ₀` is what makes the frozen limit `h` mean a
  fixed thing in raw units; if the *standardizing* `σ` were the live estimate while
  `h` stays frozen, the **effective limit in raw units would move with live vol**
  (`raw trip point = μ₀ + h·σ_live`), which is precisely the "limit refit on the live
  stream" Invariant LIM forbids — smuggled in through the denominator. So:
  - **The FROZEN `σ₀` from the validated distribution is what standardizes** (and
    `μ₀` centres). This is non-negotiable: the ARL₀ the panel pages at is the stated
    one *only* when the denominator is `σ₀`. A live `σ` in the denominator would make
    the real ARL₀ drift every time live vol ≠ validated vol.
  - **A rolling live variance `σ_live` may be computed and LOGGED as a diagnostic
    only** — on the same **embargoed, outcome-resolved** point-in-time basis as
    EXECUTION's `fill_prob` (Invariant L there): an observation never enters the
    estimate it would be compared against. It is telemetry (a persistent `σ_live/σ₀`
    gap is itself a vol-regime signal, surfaced to the `e_vol` stream §5.3), and it
    **never enters the detector's trip computation**. A genuinely live-refit variance
    is permitted ONLY as this diagnostic, NEVER as the standardizing denominator and
    NEVER for the limits.
- **AMBER — supplied metadata.** Heartbeat interval, feed-staleness window per
  venue, clock-skew tolerance per venue, severity tiering + paging routes + alarm
  dedup windows, the reconciliation event schema, per-instrument session/calendar
  (so a session gap is not read as a feed outage), venue/asset-class, the funding
  schedule (from Nautilus).
- **RED — shared SPC/modelling constants (the limits).** The CUSUM slack `k` and
  decision interval `h`; the Page-Hinkley magnitude `δ` and threshold `λ`; the EWMA
  smoothing `λ_ewma` and control-limit width `L`; the Shewhart width `L_s`; the SPRT
  type-I/type-II rates `(α, β)` and the boundaries `(A, B)`; the target **in-control
  ARL₀** and **out-of-control ARL₁** per stream; the heartbeat-miss count that
  trips; the feed-staleness and clock-skew thresholds; the **max-drawdown KILL
  level** (the hard binary backstop, set ABOVE the LEVERAGE governor's 43% flat —
  Section 7.1); the fill-rate-collapse threshold; the re-arm gating window; the
  block length for any bootstrap-derived limit (reused as the §6.7(b) batch-means
  decorrelation length); the **frozen AR(p) prewhitening coefficient(s) `φ`** and the
  measured in-control **lag-1 correlation `ρ` / variance-inflation `VIF`** per stream
  (§6.7); the **empirical/fitted-t Shewhart tail quantile `L_s`** (§6.7(c), not the
  Gaussian 3σ); the per-detector **warm-up `N_warm`** (§6.8); the flatten **retry
  cadence + max attempts** (`flatten_retry_interval`, `flatten_retry_max`, §9.1c);
  the OOS confidence the limits must clear. RED is shared and **unreachable from
  per-instrument config** (Invariant C).
- **BLUE — finite validated grids, class-level only, NEVER per-symbol-on-PnL.** The
  small enumerated candidate sets for the genuinely-uncertain SPC choices: CUSUM
  `(k, h)` pairs at a target ARL₀, EWMA `(λ_ewma, L)` pairs, the SPRT `(α, β)`
  operating point, the ARL₀ target itself (e.g. {500, 1000, 2000} in-control runs
  between false alarms). Each is a finite grid searched OFFLINE inside SYSTEM's CPCV
  framework, promoted by the OOS limit-validation gate (Section 11), never fitted
  continuously, never per symbol, never on raw P&L.

> **Invariant LIM — limits are derived from the validated distribution and frozen,
> and the standardizing `(μ₀, σ₀)` are frozen too.** Every detector limit is computed
> by `validate/derive_limits.py` from the SYSTEM-validated in-control parameters + a
> chosen ARL₀, written into the RED config, and is then immutable at runtime. No live
> code path recomputes a limit from the live stream **and no live code path
> standardizes against a live variance** (the frozen `σ₀` is the denominator; a live
> `σ` is diagnostic-only, §4 TEAL). Tests: (a) replaying a live stream with a
> deliberately shifted **mean** does NOT change any stored limit — the detector
> trips, the limit does not move; (b) replaying a stream with a deliberately shifted
> **live variance** (e.g. live vol 2× validated) does NOT move the detector's trip
> point **in raw units** — the standardization uses `σ₀`, so the raw trip point stays
> `μ₀ + h·σ₀`; the logged `σ_live` changes but the alarm sample does not. (Both are
> the same failure — a limit that moves with the live stream — caught on the mean and
> on the variance.)

> **Invariant C — RED unreachable from per-instrument config** (inherited from the
> sibling specs). RED enters only via `resolve_config(defaults=...)`. No CUSUM `k`,
> no PH `λ`, no SPRT boundary, no DD-kill level appears as a per-instrument field.

> **Invariant OOS — limits validated out-of-sample, never tuned on live P&L.** The
> BLUE limit grids are searched and promoted under SYSTEM.md's purged-walkforward /
> CPCV firewall against an OOS in-control false-alarm-rate and detection-delay
> objective (a *statistical* objective on the monitored stream), NEVER against
> backtest or live trade P&L. Test: the promotion objective function contains no
> P&L term; swapping the P&L series for noise leaves the promoted limits unchanged.

---

## 5. The monitored streams — live-vs-validated reconciliation (the core)

Reconciliation is the heart of OPS: every stream is **realized live quantity vs the
distribution SYSTEM validated**. Each is a causal stream of observations, each
standardized against its in-control parameters so the same detector code runs on
all of them. The four statistical families plus two binary-event families:

### 5.1 Realized fills vs EXECUTION's fill/slippage model

Stream `fills`: per fill, EXECUTION emits realized implementation shortfall
`IS_realized` (bps vs arrival mid, the same benchmark EXECUTION.md §5.2 defines),
realized fill rate (filled fraction of intended size), and realized adverse
selection. The monitored observation is the **residual**:

```
e_fill(i)  = IS_realized(i) − IS_expected(i)        # bps, the model error
e_fillrate(i) = fillrate_realized(i) − fillrate_expected(i)
```

In-control reference: the distribution of `e_fill` over the validation set's fills
(SYSTEM's exec-stub is a *conservative upper bound* on cost per Invariant XC, so
`e_fill` should be in-control NEGATIVE-or-zero-mean — realized cost ≤ modeled cost;
a positive drift means realized fills are WORSE than the conservative model, a real
break). Detectors: **CUSUM** (two-sided, catches a slow IS drift) + **EWMA** (catches
a sustained small shift) on `e_fill`; **Shewhart** on single catastrophic fills.
`e_fillrate` collapse is a circuit-breaker trigger (Section 7.4).

### 5.2 Realized funding/cost vs COST's forecast

Stream `cost_tca`: COST.md's post-trade TCA already produces realized round-trip
cost and realized accrued funding vs the forecast it made (the realized-vs-expected
of COST §9 / Invariant E). The monitored observation is the **TCA residual**:

```
e_cost(i)    = cost_realized(i)    − cost_forecast(i)       # round-trip, bps
e_funding(i) = funding_realized(i) − funding_forecast(i)    # signed, bps (P7 of COST)
```

In-control reference: the validated TCA-residual distribution. COST's forecast is
conservative-band on the gate path but the *realized-vs-expected* residual is what
OPS polices — a persistent positive `e_cost` (realized cost exceeds even the
conservative forecast) is the canonical "cost model has gone stale" break and is
the worked example of Section 10. Detectors: **CUSUM** on `e_cost` (the centerpiece
worked example), **CUSUM** on `e_funding`, **Page-Hinkley** as a second mean-shift
confirmation. A sustained `e_cost` CUSUM breach is a circuit-breaker trigger
(Section 7.3) AND feeds back to COST.md (Invariant XC's "re-tighten the stub")
through the forensic log.

### 5.3 Realized vol vs assumed vol

Stream `vol`: per bar (or per decision), LEVERAGE's vol forecast assumed for sizing
vs the realized short-horizon vol Nautilus's bars deliver. The observation:

```
e_vol(t) = log( realized_vol(t) / assumed_vol(t) )         # log-ratio, ~0 in control
```

(Log-ratio so the stream is symmetric and additive; a realized vol 2× the assumed
is `+0.69`, half is `−0.69`.) In-control reference: the validated vol-forecast-error
distribution. A sustained positive `e_vol` means the book is sized for calmer
markets than it is getting — the leverage ceilings were derived against a vol the
world has left. Detectors: **EWMA** (vol error is autocorrelated; EWMA is the right
smoother) + **CUSUM**. A vol-error breach degrades to stop-new-entries (Section 9)
and pages, because LEVERAGE's own vol-target and tail ceilings should already be
tightening size — OPS's job is to catch the case where the *forecast itself* is
biased, which the in-engine ceilings cannot see.

### 5.4 Realized hit-rate / edge-per-trade / trade-frequency vs the backtest

Stream `edge`: the alpha-decay streams. Per closed trade, realized net-of-cost
edge; rolling realized hit-rate; rolling realized trade-frequency. The observations:

```
e_edge(j)  = edge_realized(j)        # net-of-cost P&L per trade, in R-multiples or bps
hit(j)     = 1{trade j profitable}   # Bernoulli stream
freq(window) = trades per unit time vs validated frequency
```

In-control reference: the **validated edge-per-trade distribution** (mean, variance,
skew) from SYSTEM's deflated-Sharpe gate, the validated hit-rate `p₀`, and the
validated trade-frequency. This is the most important and the hardest stream: edge
is the lowest-frequency, highest-variance stream (one observation per closed trade,
maybe a few per week), so its detector must be **sequential and sample-efficient**:

- **SPRT** on the hit-rate Bernoulli stream `hit(j)` — the Wald test is the right
  *sample-efficient* tool: it decides "edge intact (`p ≥ p₀`)" vs "edge decayed
  (`p ≤ p₁`)" in the *minimum expected number of trades*, with pinned type-I/type-II
  error. It accumulates evidence per trade and declares as soon as the
  log-likelihood ratio crosses a boundary (Section 6.5). **But the SPRT is a
  fixed-hypothesis test, not a change-point detector**, and the live edge stream runs
  for the book's lifetime, so a bare restart-after-reset SPRT does NOT have its design
  `α` as its lifetime false-halt rate — the v1 form is therefore a **CUSUM-of-the-LLR
  ("CUSUM-SPRT")** whose in-control ARL₀ is a single pinned number that enters the
  panel budget like every other CUSUM (Section 6.5a). This is the centerpiece of
  edge-decay detection.
- **CUSUM** on `e_edge(j)` (edge-per-trade drift) as the continuous complement.
- **Trade-frequency** is monitored with a Shewhart chart on the count per window
  (a frequency collapse — the system has stopped finding trades, or a frequency
  explosion — it is over-trading noise — are both breaks; frequency collapse often
  precedes edge decay because the validated opportunity set has changed).

> Edge-decay is detected SLOWLY by construction (low frequency) — which is exactly
> why it must be a *sequential* test (SPRT/CUSUM) and never a fixed-window mean test:
> a fixed window either waits too long (blind through a real decay) or fires on too
> few trades (floods). The sequential tests minimize the expected detection delay at
> a pinned false-alarm rate (Section 6.5–6.6).

### 5.5 Reconciliation mismatch (binary event, monitored not re-derived — P5)

Stream `recon`: SYSTEM/Nautilus emit `reconciliation_ok: bool` + the mismatch detail
(venue-vs-book position delta, working-order mismatch, account-balance mismatch) on
every restart and on a periodic cadence. OPS does **not** re-derive any of this; it
monitors the boolean. A mismatch is a **binary circuit-breaker trigger** (Section
7.5), not a statistical drift — there is no "small acceptable position mismatch";
either the book matches the venue or it does not. (Invariant RECON.)

### 5.6 Heartbeat / staleness / clock-skew (binary event — P6)

Stream `heartbeat`: Nautilus's heartbeat, feed last-tick age per venue, and
clock-skew (local vs venue time). Binary triggers (Section 7.6): `N_hb` consecutive
missed heartbeats, feed age beyond the staleness window (session-aware via AMBER
calendar so a market close is not read as an outage), or clock-skew beyond
tolerance (a skewed clock corrupts every funding-stamp and bar-timestamp
computation downstream). These fail toward SAFE (P6): a dark feed is treated as a
break, not as "in control by silence."

> **Invariant STREAM — every monitored stream is a live-vs-validated residual or a
> binary event, standardized against FROZEN in-control parameters.** No stream is
> monitored against a limit fitted from itself live (P2/LIM). Test: each stream's
> in-control parameters trace to a field in `config/in_control.py` sourced from a
> SYSTEM validation report, not to a live rolling estimate (except the
> standardization variance, embargoed per §4 TEAL).

---

## 6. The detector mathematics — pinned, reproducible, ARL-governed (the centerpiece)

All detectors run on a standardized stream. Write the standardized observation as

```
z_i = ( x_i − μ₀ ) / σ₀
```

where `μ₀`, `σ₀` are the **in-control mean and standard deviation from the validated
distribution** (RED/frozen, §5). Every limit below is derived from `(μ₀, σ₀)` and a
target ARL — never from the live stream. The detectors are independent; their alarms
compose by precedence (Section 7), never by a product (mirror of LEVERAGE P6).

### 6.1 Two-sided tabular CUSUM (Page 1954)

Detects a **persistent small shift** in the mean. The recursion accumulates the
signed deviation beyond a slack `k` (half the shift you want to detect, in σ units),
resetting at zero so it does not drift on noise:

```
C⁺_i = max( 0 , C⁺_{i-1} + z_i − k )        upper (detects an UPWARD shift)
C⁻_i = max( 0 , C⁻_{i-1} − z_i − k )        lower (detects a DOWNWARD shift)
C⁺_0 = C⁻_0 = 0
ALARM when  C⁺_i > h   OR   C⁻_i > h
```

- `k` = `δ_shift / 2` in σ units, where `δ_shift` is the shift magnitude (in σ) the
  chart is tuned to detect quickly (RED/BLUE). For a cost-residual stream tuned to
  catch a 1σ drift, `k = 0.5`.
- `h` = the decision interval (RED/BLUE), chosen jointly with `k` to hit the target
  ARL₀ (Section 6.6). A standard, reproducible operating point: `k = 0.5`,
  `h = 4.77` gives a **TWO-sided** in-control **ARL₀ ≈ 370** and, for a 1σ shift,
  out-of-control **ARL₁ ≈ 10** — i.e. ~370 in-control samples between false alarms
  (on EITHER arm) but a real 1σ break caught in ~10 samples. `h = 5.0` gives the
  two-sided ARL₀ ≈ 469. **The factor of 2 is load-bearing and is the single easiest
  error to make here:** the Siegmund closed form (§6.6) returns the *per-arm*
  (one-sided) ARL — `(k=0.5, h=4.77)` has a one-sided ARL ≈ **743**, and the chart
  above ALARMs when `C⁺ > h` OR `C⁻ > h`, so two near-independent arms each at
  one-sided ARL ≈ 743 combine to a two-sided ARL₀ ≈ 743/2 ≈ **370** (the pinned
  number). Likewise `h = 5.0` → one-sided ≈ 938 → two-sided ≈ 469. The combination
  rule is `1/ARL₀_2sided ≈ 1/ARL₀⁺ + 1/ARL₀⁻ = 2/ARL₀_oneside` for a symmetric
  chart (Section 6.6). The detection-side ARL₁ is unaffected by the factor of 2: a
  1σ UPWARD shift drives only `C⁺` (the `C⁻` arm's `Δ = −1.5σ` gives a per-arm ARL of
  ~10⁷, so its `1/ARL` contribution is negligible), so two-sided ARL₁@1σ ≈ the upper
  arm's ≈ 9.9 ≈ 10. These `(k,h)→ARL` pairs are tabulated and pinned in
  `validate/arl.py` (which solves for `(k,h)` against the **two-sided** ARL₀, §6.6).
- **Causal, O(1):** two scalar states, one update per observation. Resets to 0 on a
  clean stretch, so it is overfitting-proof: it has no parameters fitted on the live
  stream, only `(k,h)` frozen from the validated `(μ₀,σ₀)` + ARL target.

### 6.2 Page-Hinkley test

A second mean-shift detector, used as a confirmation on the cost/funding streams
(an independent statistic so a single chart's quirk does not fire a kill alone). It
tracks the cumulative deviation from the running mean minus a tolerance `δ`, and
alarms when the gap from the running minimum exceeds `λ`:

```
m_T = Σ_{i≤T} ( z_i − δ )                      cumulative (drift-tolerant) sum
M_T = min_{i≤T} m_i                            running minimum
PH_T = m_T − M_T
ALARM (upward shift) when  PH_T > λ
(symmetric form with max for a downward shift)
```

- `δ` = the allowed drift magnitude (in σ; the smallest change deemed *not* worth
  flagging), RED. `λ` = the detection threshold (RED/BLUE), set from the target
  false-alarm rate. Larger `λ` ⇒ fewer false alarms, slower detection.
- Page-Hinkley and CUSUM are mathematically close (PH is a CUSUM variant against a
  running mean); running both gives a cheap **2-of-2 confirmation** option for the
  hard cost-kill (Section 7.3), reducing false halts without blinding. O(1), causal.

### 6.3 EWMA control chart (Roberts 1959)

Detects a **sustained small shift** with memory — the right tool for autocorrelated
streams (vol error, fill IS) where a single-point Shewhart chart is insensitive:

```
S_i = λ_ewma · z_i + (1 − λ_ewma) · S_{i-1}        S_0 = 0  (= in-control mean)
control limits:  ± L · sqrt( λ_ewma / (2 − λ_ewma) · (1 − (1−λ_ewma)^{2i}) )
ALARM when  |S_i| > the (asymptotic) limit  L · sqrt( λ_ewma / (2 − λ_ewma) )
```

- `λ_ewma ∈ (0,1]` (RED/BLUE), small (e.g. 0.1–0.2) for high sensitivity to small
  persistent shifts; `L` the limit width in σ (RED/BLUE), e.g. `L = 3.0`.
- The pair `(λ_ewma = 0.10, L = 2.70)` gives in-control **ARL₀ ≈ 370** and detects a
  1σ shift in **ARL₁ ≈ 10**; `(0.20, 2.86)` gives ARL₀ ≈ 370 with faster large-shift
  detection. These are the standard Lucas–Saccucci (1990) tabulated pairs, pinned in
  `validate/arl.py`. The transient term `(1−(1−λ)^{2i})` matters only for the first
  few samples; the asymptotic limit governs steady state — and that transient is one
  of the reasons a detector spends a governed warm-up `N_warm` with alarms suppressed
  before it is trusted (§6.8).
- O(1): one scalar state. The limit is a closed form in `(λ_ewma, L)` — no live fit.

### 6.4 Shewhart individuals chart (3σ)

The blunt, fast detector for **single large excursions** (a catastrophic fill, a
single absurd cost). Alarms on one point past the limit:

```
ALARM when  |z_i| > L_s          (L_s = 3.0 RED ⇒ in-control ARL₀ ≈ 1/(2·Φ(−3)) ≈ 370)
```

- Shewhart catches *big* shifts in one sample (which CUSUM/EWMA are slower on) but
  is blind to *small persistent* drifts (which CUSUM/EWMA catch). Running Shewhart +
  CUSUM/EWMA together covers both ends of the shift-size spectrum — the standard SPC
  pairing. `L_s = 3.0` ⇒ false alarm every ~370 samples; `L_s = 3.5` ⇒ ~2150.
- O(1): stateless (depends only on the current standardized point).

### 6.5 Sequential Probability Ratio Test (Wald 1945) — the edge-decay detector

The right tool for the low-frequency, high-stakes edge/hit-rate stream (§5.4): it
declares in the **minimum expected number of trades** at a pinned error pair. Test
hit-rate `p`:

```
H₀: p = p₀  (validated hit-rate, edge intact)
H₁: p = p₁  (decayed hit-rate, p₁ < p₀, the smallest decay worth halting for)

per-trade log-likelihood-ratio increment (Bernoulli):
  Λ_j = Λ_{j-1} + [ hit_j · ln(p₁/p₀) + (1 − hit_j) · ln((1−p₁)/(1−p₀)) ]
  Λ_0 = 0

Wald boundaries (from chosen type-I α, type-II β):
  A = ln( (1−β) / α )          B = ln( β / (1−α) )      (B < 0 < A)

DECIDE  "edge decayed (H₁)"  when  Λ_j ≥ A     -> circuit-breaker (Section 7.7)
DECIDE  "edge intact (H₀)"   when  Λ_j ≤ B     -> reset, keep trading
CONTINUE (collect another trade) while  B < Λ_j < A
```

- `(α, β)` RED/BLUE (e.g. `α = 0.05` false-halt-on-good-edge, `β = 0.10`
  miss-a-real-decay). `p₀`, `p₁` from the validated distribution (`p₀` = the gated
  hit-rate; `p₁` = `p₀` minus the smallest economically-meaningful decay, e.g. the
  hit-rate at which net edge = cost).
- **Expected sample number (the FULL Wald ASN, boundary-probability-weighted).** The
  hand-wave `E[N|H] ≈ boundary/increment` is wrong — it omits *which* boundary is hit
  and with what probability. The correct Wald form weights each absorbing boundary by
  its hitting probability under the hypothesis:

  ```
  E[N | H₁] ≈ ( (1−β)·A + β·B ) / E[ ΔΛ | p = p₁ ]
  E[N | H₀] ≈ (   α  ·A + (1−α)·B ) / E[ ΔΛ | p = p₀ ]
  where  A = ln((1−β)/α) > 0,   B = ln(β/(1−α)) < 0,
         E[ ΔΛ | p ] = p·ln(p₁/p₀) + (1−p)·ln((1−p₁)/(1−p₀))   (mean per-trade LLR increment)
  ```

  For `p₀=0.55, p₁=0.45, α=0.05, β=0.10`: `A = +2.890`, `B = −2.251`,
  `E[ΔΛ|H₁] = +0.0201`, `E[ΔΛ|H₀] = −0.0201`, giving **ASN_H₁ ≈ 118 trades** and
  **ASN_H₀ ≈ 99 trades**. (The earlier "40–50" figure was wrong by ~2.5×; it dropped
  the boundary weighting.) So at a few trades/week a decision is **months**, not
  "tens of trades." If a *faster* decision is genuinely wanted, the only honest lever
  is to widen the decay being tested (a larger `p₀−p₁` raises `|E[ΔΛ]|` and shrinks
  ASN) — re-pick `(p₀,p₁)` to the wider gap, do NOT quote a smaller ASN for the
  narrow one. Both `ASN_H₀` and `ASN_H₁` are tabulated in `validate/arl.py` alongside
  the CUSUM/EWMA ARLs (Invariant ARL).
- **Causal, O(1):** one accumulator, one increment per closed trade. **Overfitting-
  proof:** `p₀, p₁, α, β` are all frozen from the validated distribution + the chosen
  error rates; nothing is fit on the live P&L. The CUSUM-on-edge-per-trade (§6.1
  applied to `e_edge`) runs alongside as the continuous complement.
- **It is a fixed-hypothesis test, NOT a change-point detector — pin the repeated-test
  discipline (§6.5a).** A vanilla Wald SPRT decides ONCE between two fixed
  hypotheses; restarting it after a B-crossing reset turns the lifetime stream into a
  *sequence* of SPRT epochs, and the per-epoch `α` compounds. See §6.5a.

### 6.5a Repeated SPRT epochs vs a genuine change-point detector (the edge stream's false-alarm budget)

The SPRT of §6.5 is the right *sample-efficient* tool, but it is a **fixed-hypothesis**
test, and the live edge stream is not a one-shot decision — it runs for the life of the
book. The §6.5 reset ("DECIDE edge intact (H₀) → reset, keep trading") restarts the
test, so over a year the book runs *many* SPRT epochs. The Wald `α` is the false-halt
probability **per epoch**, not over the lifetime: with `K` independent epochs the
lifetime false-halt probability is `≈ 1 − (1−α)^K`, which for `α=0.05` and a dozen
epochs is `> 45%` — nowhere near the design `α`. As written this would make the edge
stream the **one detector whose false-alarm rate is NOT pinned into the panel budget**
the rest of the spec is so careful about. Two admissible fixes, pick one and pin it:

1. **Budget the repeated test explicitly.** Treat each restart as consuming part of a
   *lifetime* false-halt budget: allocate a per-epoch `α_epoch` so that the lifetime
   `1 − (1−α_epoch)^K ≤` the edge stream's share of the panel budget, and enter the
   resulting **per-epoch ARL₀-equivalent** (`≈ ASN_H₀ / α_epoch` in trades) into the
   same `Σ_d(1/ARL₀,d)` panel accounting of §6.6/§8.3. The edge stream's contribution to the
   panel false-page rate is then this ARL₀-equivalent, not the bare `α`.
2. **Replace restart-SPRT with a CUSUM-of-the-LLR ("CUSUM-SPRT") — preferred.** Run a
   one-sided CUSUM on the *same* per-trade Bernoulli log-LR increment `ΔΛ_j` (the §6.1
   recursion with `z_i := ΔΛ_j`, `k := 0`): `G_j = max(0, G_{j-1} + ΔΛ_j)`, alarm when
   `G_j > h_edge`. This is a genuine change-point detector — it auto-resets on a clean
   stretch, never "decides H₀ and stops," and its in-control **ARL₀ is a single pinned
   number** (Siegmund/MC on the `ΔΛ` distribution under `p₀`) that drops straight into
   the §6.6 panel budget like every other CUSUM. The §6.1 CUSUM-on-`e_edge` is already
   half this; the CUSUM-SPRT makes the hit-rate arm match. **v1 default = option 2**;
   option 1 is permitted only if the team wants the SPRT's minimum-ASN property and
   accepts owning the lifetime-budget bookkeeping.

> Either way, the edge stream's false-alarm contribution is `ARL₀`-shaped and enters
> the `Σ_d(1/ARL₀,d)` panel budget of §6.6 — the lifetime false-halt rate of the edge
> detector is a *designed* number, never the raw per-epoch `α`. (Pinned in Invariant
> ARL's reproduction test alongside the CUSUM/EWMA ARLs and the SPRT ASN.)

### 6.6 ARL / false-alarm math — so alarms are neither flooding nor blind

The whole point of deriving limits from `(μ₀, σ₀)` + an ARL target is that we can
**state the false-alarm rate and the detection delay before going live**, and tune
the trade-off deliberately rather than discover it in production.

```
ARL₀  = expected number of IN-CONTROL samples between FALSE alarms  (want LARGE)
ARL₁  = expected number of samples to detect a real shift of size δ (want SMALL)
```

- **CUSUM** (Siegmund's approximation): for a **one-sided** chart with slack `k` and
  interval `h`, `ARL_oneside ≈ ( exp(−2·Δ·(h+1.166)) + 2·Δ·(h+1.166) − 1 ) / (2·Δ²)`
  where `Δ = μ_shift − k` in σ units (`Δ = −k` in control gives the per-arm ARL₀;
  `Δ = δ_shift − k` out of control gives the per-arm ARL₁). **This closed form returns
  the PER-ARM (one-sided) ARL.** The chart of §6.1 is **two-sided** (`C⁺ > h` OR
  `C⁻ > h`), so the in-control ARL₀ the panel actually experiences is the
  two-arm combination
  `1/ARL₀_2sided ≈ 1/ARL₀⁺ + 1/ARL₀⁻ = 2/ARL₀_oneside` (the two arms are
  near-independent in control — see the dependence caveat below — so for a symmetric
  chart `ARL₀_2sided ≈ ARL₀_oneside / 2`). Worked: `(k=0.5,h=4.77)` → one-sided 743 →
  **two-sided 370**; `(k=0.5,h=5.0)` → one-sided 938 → **two-sided 469**. The
  detection-side ARL₁ does NOT get halved — a one-directional shift drives only the
  matching arm (the opposite arm's `Δ = −(δ+k)` makes its `1/ARL` negligible), so
  `ARL₁_2sided ≈ ARL₁_oneside ≈ 10` at 1σ. **`validate/arl.py` and `derive_limits.py`
  solve for `(k, h)` against the TWO-sided ARL₀** (i.e. they target `ARL₀_oneside =
  2 × target_two_sided`), so an implementer who solves the bare one-sided formula for
  ARL₀=370 — landing on `h ≈ 3.5` and roughly *halving* the real limit (doubling the
  false-alarm rate) — is producing the bug this section exists to forbid.
  - **The `÷2` two-sided rule is an APPROXIMATION, and the PINNED `(k,h)` is solved
    against the MC two-sided ARL₀, not the `÷2` shortcut (pin three things):**
    (a) **The two arms are NOT exactly independent in control.** `C⁺` and `C⁻` share
    the *same* `z_i` stream with opposite sign, so a large `|z_i|` pushes one arm up
    while it pushes the other's increment `−|z_i|−k < 0` (resetting it toward 0) — the
    two arms are weakly **negatively** associated in control. Negative association
    makes the union false-alarm rate slightly *less* than the sum, so the true
    two-sided ARL₀ is slightly `≥ ARL₀_oneside / 2` — i.e. **`÷2` is a mildly
    CONSERVATIVE (over-pessimistic on false-alarm rate) seed.** (b) **Error bound:**
    for `k = 0.5` in the operating range the dependence correction is a **few-percent**
    effect — the same order as the Siegmund-vs-MC `9.9`-vs-`10.4` ARL₁ gap (~5%) —
    and `validate/arl.py` reports the gap (`ARL₀_2sided_MC` vs `ARL₀_oneside_MC / 2`).
    (c) **MOST IMPORTANT — the frozen limit is solved against the MC two-sided ARL₀,
    not the analytic `÷2`.** `derive_limits.py` runs the *actual two-arm OR-chart* on
    a dependence-aware in-control MC stream (§6.7) and solves for `h` so the
    **empirical two-sided ARL₀ = target (370)**; the `÷2`/Siegmund form (one-sided 743)
    is retained ONLY as the analytic seed/bracket and the sanity cross-check. Where the
    two disagree the MC value is authoritative (the general rule below).
- **Validity range of the closed forms, and the general rule that the MC solver is
  authoritative.** The Siegmund CUSUM form and the Lucas–Saccucci EWMA tables are
  **design-time APPROXIMATIONS** with a stated range: Siegmund's ARL degrades for
  **large `k`, small `h`, and OFF-DESIGN shift sizes** — it is calibrated near the
  design shift, so `ARL₁` quoted at shifts far from `δ_shift` is unreliable (the §10
  worked `1.5σ` delay is read against the `1σ` design ARL₁ for exactly this reason —
  the realized delay is *consistent with* the `1σ` number, not equal to a `1.5σ`
  closed form). **General principle (pin in Invariant ARL):** *closed-form ARLs
  (Siegmund CUSUM, Lucas–Saccucci EWMA) are the approximation and the bracketing seed;
  the pinned RED limits and their stated ARL₀/ARL₁ are the values `derive_limits.py`
  returns from the dependence-aware Monte-Carlo solver (§6.7), which is authoritative
  wherever the two disagree.* This is already the rule for the AND-rule joint ARL₀
  (§6.6a is MC) and for the one worked ARL₁ (§10/Invariant WE asserts `10.4` against MC,
  not Siegmund); it is hereby the rule for **ALL** limits, not a one-off — every RED
  tabulated ARL and every pinned `(k,h)`/`(λ,L)`/`L_s`/`h_edge` is the MC solver output,
  closed form retained as cross-check + initial bracket.
- **EWMA**: ARL from the Lucas–Saccucci (1990) tables / a Markov-chain
  approximation of the chart statistic; pinned `(λ_ewma, L)` → ARL pairs.
- **Shewhart**: `ARL₀ = 1 / P(|z| > L_s | in control) = 1 / (2Φ(−L_s))`; `L_s=3 ⇒
  370`, `L_s=3.5 ⇒ 2150`.
- **SPRT**: false-decision rates are the design inputs `(α, β)`; the boundaries
  `(A,B)` are closed-form from them; the full boundary-weighted ASN (§6.5) gives the
  expected detection delay (`ASN_H₁ ≈ 118`, `ASN_H₀ ≈ 99` at the worked operating
  point — both tabulated). Because the live edge stream runs *repeated* SPRT epochs,
  its false-alarm contribution to the panel is the §6.5a `ARL₀`-equivalent (a CUSUM-
  SPRT's pinned ARL₀, or the lifetime-budgeted per-epoch `ASN_H₀/α_epoch`), NOT the
  bare per-epoch `α`.
- **Across the whole panel — the family-wise false-alarm budget, counted over
  ALARM SOURCES, not streams.** The naive form `panel rate ≈ S / ARL₀` with `S` =
  number of *streams* is **wrong here**, because several streams run two or three
  detectors at once (`e_cost`: CUSUM + Page-Hinkley; `e_vol`: EWMA + CUSUM; `edge`:
  CUSUM-SPRT + CUSUM-on-`e_edge` + Shewhart-on-frequency). The panel rate is summed
  over the **independent alarm sources**, with two corrections:
  - **Correlated detectors on one stream are NOT two sources.** PH and CUSUM on the
    same `e_cost` are "mathematically close" (§6.2) — their alarms are highly
    dependent, so `S = detectors` *over-counts* just as `S = streams` under-counts.
    When two correlated detectors on one stream are combined by an **AND-rule** (the
    2-of-2 cost confirmation, §6.2/7.3), they form **one combined detector with its
    own joint ARL₀** (§6.6a) and enter the sum as a single source. A lone (OR-style)
    second detector that can fire independently *does* add a source.
  - So: `panel rate ≈ Σ_d (1/ARL₀,d)` over the set of independent alarm sources `d`
    (each single detector at its own ARL₀, each AND-combined pair at its *joint*
    ARL₀, the edge stream at its §6.5a `ARL₀`-equivalent), and the expected time
    between *any* false page is `≈ 1 / Σ_d(1/ARL₀,d)`.
  The paging cadence (Section 8) is set from THIS source-counted rate: pick each
  source's ARL₀ so that `1 / Σ_d(1/ARL₀,d) × (sample interval)` is a human-acceptable
  "page roughly every few weeks of in-control operation" — e.g. ~8 streams resolving
  to ~6 independent sources (two 2-of-2 AND-pairs collapse four detectors into two),
  each at per-source ARL₀ ≈ 2000 samples ⇒ a panel false-page every ~330 samples; at
  a per-trade edge stream that is a sane page rate, at a per-bar vol stream it is too
  hot, so per-source ARL₀ is set per the stream's sample frequency (RED). This is the
  math that makes "page rarely, catch fast" a *designed* property, not a hope — and
  it is mis-budgeted by exactly the over-/under-count factor if `S` is read as raw
  streams or raw detectors instead of independent sources.

### 6.6a The 2-of-2 AND-rule — semantics and joint ARL₀ (the cost confirmation, pinned)

The cost-kill (7.3) fires only when CUSUM on `e_cost` is confirmed by Page-Hinkley.
"Confirmation" must pin two things the prose left implicit:

- **Window semantics (pinned).** The AND-rule is **windowed, not same-sample**:
  the rule is breached when **both** detectors are in their breached state within a
  confirmation window of `M_conf` consecutive samples (RED; v1 `M_conf = 3`). CUSUM
  and PH integrate at slightly different rates, so requiring an identical breach
  sample would make the AND-rule miss real breaks; a short window keeps the
  false-halt reduction while not blinding. (Same-sample, `M_conf = 1`, is a permitted
  stricter setting.)
- **Joint ARL₀ (pinned, not a product).** Two *independent* detectors each at ARL₀
  combined by AND over a window have a joint in-control ARL₀ far larger than either
  alone (that is the whole point — an AND-rule trades a little detection delay for a
  much rarer false halt). But CUSUM and PH on the same stream are **strongly positively
  dependent**, so the independence product `ARL₀,cusum × ARL₀,ph / M_conf` is only a
  loose *upper bound* and would over-state the rarity. The pinned number is the
  **dependence-aware Monte-Carlo joint ARL₀** estimated by `validate/arl.py` from a
  simulated in-control `e_cost` stream run through both recursions with the windowed
  AND — reported with its CI, and entered into the §6.6 panel budget as the cost
  stream's single source. The independence product is logged only as the bounding
  sanity check (`joint_ARL₀_MC ≤ product_bound`). The escalation arm (CUSUM past
  `h₂`, §7.3/Section 10) is a *separate* trigger with its own one-sided ARL and is not
  part of the AND-rule's joint number.

### 6.7 In-control ARL under serial dependence and fat tails (the SPC-on-financial-data trap)

Every ARL number above — the CUSUM `370`, the EWMA `370`, the Shewhart `370`, the
panel budget `Σ_d(1/ARL₀,d)` — is derived for **IID-Gaussian standardized residuals**.
But every monitored stream is the opposite: `e_cost`, `e_fill`, `e_vol` (especially),
and `e_edge` are **autocorrelated** *and* **fat-tailed**. The spec elsewhere uses
"autocorrelated" only as the *reason to pick EWMA* (§5.3, §6.3) and never confronts the
consequence — which is that the as-designed ARL₀ is **not** the realized ARL₀ on a raw
financial residual, so the panel would page far more often than budgeted. This
subsection pins, per stream, the step that makes the raw residual an
approximately-IID input the ARL math actually holds for, and the quantitative
correction. **It is the dominant gap; without it the panel floods in production.**

**(a) Whitening / standardization to an approximately-IID `(μ₀, σ₀)` input.** The
detector is never fed the raw residual directly. Per stream, one of two RED-pinned
prewhitening steps takes the raw residual to an approximately-IID innovation:
- **Prewhitened AR(p) innovation (default for `e_vol`, `e_cost`, `e_fill`).** Fit a
  low-order AR(1) (or AR(p), `p ≤ 2`) to the **IN-CONTROL validation residual OFFLINE**
  (inside SYSTEM's firewall, on validation data only — never live), and **freeze the
  coefficient(s) `φ` as RED**. The detector consumes the prewhitened innovation
  ```
  z_i = ( residual_i − φ·residual_{i-1} ) / σ_innovation        (AR(1) form; σ_innovation = std of the in-control residuals)
  ```
  `z_i` is approximately IID `N(0,1)` in control, so the IID ARL math applies to it.
  `φ` is a frozen RED constant fitted on in-control validation data (not P&L, not
  live) — it is a *closed-form least-squares* AR fit, not ML (P2/Invariant LIM still
  hold: it is frozen offline, never refit on the live stream).
- **Non-overlapping BATCH MEANS (alternative, esp. for `e_edge`).** Average the
  residual over non-overlapping blocks of length `b` chosen so consecutive batch
  means are decorrelated, then run the detector on the batch-mean stream. **Reuse the
  RED block length already pinned for the bootstrap** (§4 RED: "the block length for
  any bootstrap-derived limit") — the same decorrelation length serves both. Batch
  means inflate the per-sample interval, so the per-source ARL₀ is set in *batches*,
  consistent with the per-frequency budgeting of §6.6.

**(b) Direction and conservative correction for positive autocorrelation.** Positive
serial correlation **SHRINKS** the true in-control ARL₀ (the statistic wanders more
persistently, so it crosses `h` sooner ⇒ *more* false pages). For an AR(1) residual
with lag-1 correlation `ρ > 0` fed to a chart calibrated on IID, the effective
variance of the running statistic inflates by the **variance-inflation factor**
```
VIF = (1 + ρ) / (1 − ρ)        (AR(1))
```
so an as-designed ARL₀ of `370` can collapse to a small fraction of that (e.g. `ρ=0.5`
⇒ `VIF=3` ⇒ realized ARL₀ a fraction of design). **Pin:** `derive_limits.py` either
(i) feeds the detector the **prewhitened innovation** of (a) — in which case `ρ≈0` and
no inflation is needed — OR (ii) if a stream is monitored on its *raw* (un-whitened)
residual, **INFLATES `h`** (raises the per-source one-sided ARL₀ target) by the
measured in-control `VIF` so the **REALIZED** two-sided ARL₀ on the autocorrelated
stream hits the design target. The two routes are equivalent (whiten the input, or
loosen the limit by VIF); v1 default is **prewhiten** (route i) so every detector sees
an IID input and the §6.6 numbers stand as written. The measured in-control `ρ`/`VIF`
per stream is recorded in the RED config alongside the limit.

**(c) Fat tails — Shewhart's `1/(2Φ(−L_s))` is Gaussian and WRONG for these streams.**
`ARL₀ = 1/(2Φ(−L_s))` assumes the standardized point is Gaussian. Under a fat-tailed
in-control residual the true exceedance rate at `L_s = 3` is **far higher** than
`2Φ(−3) = 0.0027` (a Student-t with ~4–6 dof puts several times that mass past 3σ), so
the Gaussian formula *over-states* the Shewhart ARL₀ and the chart pages far more than
`1/370`. **Pin:** the Shewhart limit `L_s` is set from the **EMPIRICAL / validated tail
quantile** of the in-control distribution (or a fitted Student-t / EVT tail quantile),
NOT the Gaussian 3σ — `L_s` is the quantile such that the *empirical* in-control
exceedance rate `= 1/ARL₀_target`. **The worked Shewhart `370` is the
empirical-quantile ARL₀**, and the Gaussian `L_s=3 ⇒ 370` is retained only as the
thin-tailed reference (the same closed-form-as-seed rule of §6.6 — the empirical
quantile is authoritative). The CUSUM/EWMA charts are more robust to fat tails than
Shewhart (they integrate, the single point dominates less), but their MC-solved ARL₀
of (a)/(b) is run on the **fat-tailed** in-control synthetic too, so any residual
sensitivity is priced into the pinned `h`.

**(d) The realized-ARL₀ test on a dependent + fat-tailed in-control synthetic** (this
test does not exist today and is the proof the panel will not flood). Invariant ARL's
reproduction test asserts the **REALIZED Monte-Carlo ARL₀** on an in-control synthetic
that is **autocorrelated AND fat-tailed** (an AR(1)-Student-t in-control stream per
the measured `ρ` and dof) — not merely on the Gaussian-IID synthetic — lands within
CI of the design target (`370`) *after* the prewhitening of (a) and/or the `h`-inflation
of (b). The Gaussian-IID synthetic remains as the closed-form cross-check; the
dependent-fat-tailed synthetic is the production-fidelity check.

### 6.8 Detector warm-up / minimum sample (alarms suppressed until recursion + whitening are stable)

A detector must not alarm before its recursion state, its standardization, and its
prewhitening filter are stable. **Although `(μ₀, σ₀)` are FROZEN from SYSTEM** (so
there is *no* live estimation warm-up for the centering — that is the whole point of
§4/Invariant LIM), three things still need a settling period: the **EWMA transient**
`(1 − (1−λ)^{2i})` (§6.3) leaves the early control limit narrower than asymptotic; the
**CUSUM / Page-Hinkley cold start** spends its first samples climbing from `C=0`/`PH=0`
below asymptotic sensitivity; and the **autocorrelation-whitening filter** of §6.7(a)
plus the **diagnostic `σ_live`** (§4 TEAL) each need a minimum **embargoed** sample
before they are trustworthy.

- **(a) Per-detector warm-up `N_warm` (RED).** During the first `N_warm` samples of a
  stream the detector **UPDATES its recursion** (so state is correct the moment warm-up
  ends — no discontinuity) but its alarm is **SUPPRESSED to LOG only** — no PAGE,
  DEGRADE, or HALT. This is fail-safe **for SPC only**: warm-up suppresses the
  *statistical* alarms; it **does NOT suppress the binary heartbeat / reconciliation /
  staleness / clock-skew triggers** (§7.5/§7.6) nor the DD backstop (§7.1) — those are
  not SPC, have no recursion to settle, and need no warm-up (a recon mismatch or dark
  feed in the first bar is still an immediate HALT). `N_warm` is set per detector from
  its settling math: EWMA `N_warm` so `(1−(1−λ)^{2i})` is within a RED tolerance of 1
  (e.g. for `λ=0.10`, ~30–50 samples); CUSUM/PH a handful of samples; the
  prewhitening/`σ_live` warm-up `≥` the AR fit / embargo length of §6.7(a).
- **(b) Tie to the staged-rollout gates of §17.** A stream **cannot be a promotion gate
  (§17.1) until it has cleared BOTH its warm-up `N_warm` AND its `N_stage` minimum
  sample.** "No SPC alarms during warm-up" is a **rollout invariant**: a freshly
  promoted SYSTEM config (new `(μ₀,σ₀)`, new `φ`) restarts every detector's warm-up, so
  the panel does **not** page spuriously on its first live bars after a promotion. A
  stream still inside `N_warm` reports `warming_up` to §17, which **defers** the advance
  exactly as an insufficient sample does — it never advances on it.
- **(c) Invariant.** Folded into Invariant CAUSAL's test (and the synth of §12): a
  detector fed an in-control stream emits **zero PAGE/DEGRADE/HALT before `N_warm`**
  even on a transient excursion inside the warm-up window, while the binary triggers
  (recon/heartbeat/staleness/clock-skew/DD) are **unaffected** by warm-up and fire
  immediately.

> **Invariant ARL — every detector's limits carry a stated ARL₀/ARL₁ and the panel
> false-alarm rate is budgeted over independent ALARM SOURCES.** No limit ships
> without its `(ARL₀, ARL₁ at the design shift)` recorded in the config and its
> contribution to the panel false-alarm rate `Σ_d(1/ARL₀,d)` (summed over independent
> sources, §6.6) accounted in the paging cadence (Section 8). **The closed forms
> (Siegmund CUSUM, Lucas–Saccucci EWMA) are design-time APPROXIMATIONS and the
> bracketing seed; the pinned RED limits and their stated ARL₀/ARL₁ are the
> dependence-aware Monte-Carlo solver output (§6.6/§6.7), which is AUTHORITATIVE
> wherever the two disagree** — for ALL limits, not a one-off. Tests:
> - **Two-sided CUSUM ARL₀, MC-authoritative.** `validate/arl.py` reproduces the
>   *one-sided* Siegmund ARL (`(k=0.5,h=4.77)` → ≈743, `(0.5,5.0)` → ≈938) AND the
>   *two-sided* analytic combination (`÷2` → ≈370, ≈469) as the **seed/cross-check**,
>   and `derive_limits.py` solving for ARL₀=370 returns `h≈4.77` (NOT the ≈3.5 the bare
>   one-sided formula would give). The **PINNED `(k,h)` is solved against the MC
>   two-sided ARL₀**: running the actual two-arm OR-chart on an in-control MC stream and
>   solving for `h` so empirical two-sided ARL₀ = 370 — asserted **distinct from** the
>   analytic `743/2`, and the test asserts the dependence gap (`÷2` is mildly
>   conservative, true two-sided ARL₀ `≥ ARL₀_oneside/2`) is within a few percent (the
>   same order as the `9.9`-vs-`10.4` ARL₁ gap, §6.6).
> - **Realized ARL₀ on a DEPENDENT + FAT-TAILED in-control synthetic (§6.7(d)).** The
>   MC ARL₀ is asserted within CI of `370` on an **autocorrelated AND fat-tailed**
>   (AR(1)-Student-t per the measured `ρ`, dof) in-control synthetic — AFTER the §6.7(a)
>   prewhitening and/or §6.7(b) `h`-inflation by `VIF=(1+ρ)/(1−ρ)` — not merely on the
>   Gaussian-IID synthetic. (This is the test that proves the panel does not flood in
>   production; it FAILS if the detector is fed the raw autocorrelated residual with the
>   IID limit.)
> - **EWMA / Shewhart ARL.** the tabulated `(λ_ewma,L)` ARLs reproduce within tolerance
>   as the cross-check; MC ARL₀ within CI. **Shewhart `L_s` is the EMPIRICAL / fitted-t
>   tail quantile** of the in-control distribution (§6.7(c)), so its ARL₀ is the
>   empirical-quantile `370`, NOT the Gaussian `1/(2Φ(−L_s))` — the test asserts the
>   empirical exceedance rate `=1/ARL₀_target` and FAILS if the Gaussian-3σ limit is
>   used on the fat-tailed stream.
> - **SPRT ASN.** `validate/arl.py` tabulates `ASN_H₀ ≈ 99` and `ASN_H₁ ≈ 118` for
>   `(p₀=0.55,p₁=0.45,α=0.05,β=0.10)` from the boundary-weighted Wald form (§6.5); an
>   MC of the SPRT yields empirical ASNs within CI (and the test FAILS on the old
>   40–50 figure).
> - **Edge-stream lifetime budget.** the edge detector's panel contribution is its
>   §6.5a `ARL₀`-equivalent (CUSUM-SPRT pinned ARL₀, or the lifetime-budgeted
>   `ASN_H₀/α_epoch`), NOT the bare per-epoch `α`; an MC of `K` repeated in-control
>   epochs yields a lifetime false-halt rate matching the budgeted ARL₀-equivalent,
>   not `α`.
> - **Joint AND-rule ARL₀.** the 2-of-2 cost confirmation's dependence-aware MC joint
>   ARL₀ (§6.6a) reproduces within CI, exceeds either single-detector ARL₀, and is
>   `≤` the independence product bound; the panel budget uses the joint number as the
>   cost stream's single source.

> **Invariant CAUSAL — every detector is an O(1) causal recursion, and SPC alarms are
> suppressed until warm-up (§6.8).** Each `update(z_i)` reads only `z_i` and bounded
> prior state, never a future sample, never a growing window. The §6.7(a) prewhitening
> is itself causal (`z_i` reads only `residual_i` and frozen prior residuals/`φ`).
> Tests: feeding the stream one observation at a time vs all at once yields identical
> detector states at every index; replacing all post-`i` data with garbage leaves the
> state at `i` unchanged. **Warm-up:** a detector fed an in-control stream emits zero
> PAGE/DEGRADE/HALT before `N_warm` even on a transient excursion inside the warm-up
> window (its recursion still updates, alarm logs only); the binary triggers
> (recon/heartbeat/staleness/clock-skew §7.5/§7.6, DD backstop §7.1) are UNAFFECTED by
> warm-up and fire in the first bar.

---

## 7. Kill-switch / circuit-breaker logic — triggers, thresholds, responses

The kill state machine has three states — `RUN`, `DEGRADE`, `HALT` — and a set of
triggers, each with a **threshold + statistical justification + a response**. The
state is the `max`-severity (most restrictive) over all active triggers (precedence,
not sum — mirror of SYSTEM's `min()`/precedence). A trigger never *lowers* the
state; only a gated re-arm (Section 8.4) does.

```
RUN      : normal operation; gradual brakes (REGIME × governor) govern size.
DEGRADE  : stop NEW entries; hold/manage existing positions; venue stops live.
HALT     : flatten to venue reduce-only stops; cancel working orders; stop all.
```

| # | Trigger | Threshold (statistical) | Detector | Response | Justification |
|---|---------|------------------------|----------|----------|---------------|
| 7.1 | **Max-drawdown backstop** | equity peak-to-trough ≥ `DD_kill` (RED, set ABOVE governor's 43% flat — e.g. 45%), on the SAME equity series + peak definition the governor uses (§7.1a) | direct equity read (not SPC; a hard boundary) | **HALT** | The binary backstop above LEVERAGE's gradual governor: it **flattens** (not just stops sizing) and **backstops a mis-firing governor**. It does NOT prevent gap-through ruin (only the venue stop + `L_tail` gap ceiling do — §7.1a). Shared `(equity, peak)` reference keeps 45% > 43% by construction (P4, boundary with LEVERAGE Inv. G). |
| 7.2 | **Vol-regime break** | `e_vol` EWMA/CUSUM breach (ARL₀-tuned) | EWMA + CUSUM (§5.3) | **DEGRADE** + page | Realized vol has left the band sizing assumed; LEVERAGE's ceilings tighten size but cannot see a biased *forecast*. Stop new entries until the forecast re-centers. |
| 7.3 | **Cost/funding divergence** | CUSUM on `e_cost` (TCA residual) > `h`, confirmed by Page-Hinkley within `M_conf` samples (windowed AND, §6.6a) | CUSUM + PH (§5.2) 2-of-2 combined detector w/ joint ARL₀ (§6.6a) | **DEGRADE** (→ HALT if `e_cost` CUSUM continues past `h₂`) | Realized cost exceeds even COST's conservative forecast — the cost model is stale, so every net-EV gate decision is now optimistic. The 2-of-2 AND-rule has a (dependence-aware MC) joint ARL₀ much larger than either alone, cutting false halts without blinding. Worked example, Section 10. Feeds COST.md's stub re-tighten (Inv XC). |
| 7.4 | **Fill-rate collapse** | realized fill rate < `fillrate_floor` (RED) over `N_fr` orders, OR `e_fillrate` Shewhart breach | Shewhart + threshold (§5.1) | **DEGRADE** (→ HALT if exits cannot fill) | If orders stop filling, the system cannot manage risk; if a *protective exit* cannot fill, that is a HALT-and-flatten-to-venue-stop condition (the venue stop is the backstop, P6 EXEC). |
| 7.5 | **Reconciliation mismatch** | `reconciliation_ok == False` (binary) | direct monitor (§5.5, P5) | **HALT** | Book disagrees with venue — there is no safe "small mismatch." Halt, flatten to known venue state, page. OPS does not re-derive recon (SYSTEM owns it); it halts on the flag. |
| 7.6 | **Data staleness / heartbeat / clock-skew** | `N_hb` missed heartbeats, feed age > staleness window, or clock-skew > tolerance (all RED, session-aware) | direct monitor (§5.6, P6) | **DEGRADE** (stale feed) / **HALT** (heartbeat loss / clock-skew) | A blind or time-skewed system must not trade. Fail toward SAFE (P6): silence is treated as a break. |
| 7.7 | **Edge decay** | CUSUM-SPRT on hit-rate LLR `G_j > h_edge` (the §6.5a change-point form, ARL₀-budgeted), OR CUSUM on `e_edge` breach | CUSUM-SPRT + CUSUM (§5.4, §6.5a) | **DEGRADE** + page (→ research) | The validated edge no longer holds in production. Stop new entries, let TREND manage open winners out, and page for re-validation. Detected sample-efficiently by the LLR change-point detector at a pinned (lifetime, not per-epoch) false-halt rate. |
| 7.8 | **Funding blowout** | CUSUM on `e_funding`, OR realized funding rate near the venue cap on a paying-side book | CUSUM (§5.2) + COST funding read | **DEGRADE** | A funding regime that bleeds the carry the gate priced; stop adding to the paying side. COST.md's cap/forecast owns the number; OPS owns the brake. |

### 7.1a The drawdown backstop vs the governor — the precise boundary

LEVERAGE.md's drawdown governor is a **gradual size taper** that reaches 0.00 at the
43% portfolio-equity drawdown (its Section 6 ladder). OPS's max-drawdown kill is a
**binary HALT** at `DD_kill` set *above* that — e.g. 45%. Be precise about what it
actually buys, because the obvious justification slightly oversells it:

- **What it buys (legitimate):** (a) it **flattens** rather than merely *stops
  sizing* — the governor at 0.00 stops new size but holds open positions, so an open
  position drifting adversely keeps bleeding equity; the kill closes it. (b) It
  **backstops a MIS-FIRING governor** — if the ladder is mis-configured, frozen, or
  fed a bad equity number, the binary floor is the independent last line. Both are
  real and are the reason it exists.
- **What it does NOT buy (the honest concession):** it does **not** prevent
  gap-through ruin. A single adverse gap can jump equity from, say, 38% DD straight
  *through* both 43% and 45% in one bar — the binary halt then fires AFTER the loss,
  identically to how the governor would. A discrete equity threshold (gradual OR
  binary) is inherently reactive to a gap that clears it in one bar. The only controls
  that bound gap-through loss are the **venue-resident reduce-only stop** (EXEC P6,
  the safe-state floor of Section 9) and LEVERAGE's **`L_tail` gap-survival ceiling**
  (which sizes so a modeled worst gap stays inside the ruin threshold *before* the
  position is ever opened). The DD_kill is a backstop on the *governor*, not on the
  *gap*; do not claim otherwise in the forensic narrative.

So: governor = "size → 0 smoothly approaching 43%"; OPS kill = "flatten hard if we
ever *sit at* ≥ 45%, and flatten regardless of what the governor reports." OPS never
re-implements or overrides the governor ladder (Invariant BRAKE).

**The equity/peak definition is IDENTICAL to the governor's (pinned, so the ordering
cannot invert).** Both the governor's 43% and OPS's 45% are computed on the **same
peak-to-trough drawdown of the same equity series Nautilus exposes** (`Portfolio`
equity; Sections 11/15), with the **same peak definition** (running max of that
series). If the two used different equity snapshots or different peak definitions
they could momentarily disagree and invert the `43% < 45%` ordering — the kill firing
*before* the governor reached zero, or never firing. Pinning a single shared
`(equity_series, peak)` reference makes `DD_kill > DD_governor_flat` hold by
construction at every bar. OPS reads this reference; it does not maintain its own.

> **Invariant KILL — every trigger has a frozen threshold, a named detector/justification, and a deterministic response, and the kill state is precedence (max-severity), never a blended score.** The kill state = the most restrictive response among active triggers; a trigger can only raise the state. No trigger threshold is fitted on live P&L (P2). Test: each trigger fires its tabulated response on a synthetic stream that crosses exactly its limit; two simultaneous triggers (e.g. DEGRADE + HALT) yield HALT; no code path computes a weighted "health score"; a HALT held across multiple triggers/bars issues exactly ONE flatten wave via the `flatten_issued` latch (§9.1a — re-entering HALT does not re-flatten).

> **Invariant BRAKE — three distinct brakes, composed by min()-on-size and precedence-on-state; the DD halt and the governor flat share one equity/peak reference.** REGIME gross taper and LEVERAGE governor are gradual size multipliers composed by `min()` (owned by those modules); OPS contributes only the binary `{run, degrade, halt}` state by precedence. OPS never tapers gross, never re-implements the governor ladder, never multiplies its state into the size scalar. The DD_kill and the governor flat are computed on the **same equity series and the same peak definition** (§7.1a) so `DD_kill (45%) > DD_governor_flat (43%)` holds by construction at every bar. Test: with OPS=RUN, size equals REGIME×governor unchanged; with OPS=HALT, size is 0 regardless of the two scalars; with OPS=DEGRADE, existing size is held (no new entries) regardless of the scalars; the governor's 43% flat and OPS's 45% halt are both present, the halt level is strictly higher, and feeding both the SAME equity/peak snapshot never inverts the ordering (a divergent-snapshot test must fail to construct under the shared-reference contract).

---

## 8. Alerting — severity tiers, dedup, and the ARL-set paging cadence

### 8.1 Severity tiers (the response ladder)

```
LOG      : record to the forensic log only. (a single Shewhart point that self-clears,
           an in-control SPRT continue) — no human, no state change.
PAGE     : notify a human (push notification / message), no automatic state change.
           (an EWMA/CUSUM breach that warrants eyes but not an automatic halt) —
           the DEGRADE-class alarms (7.2, 7.7, 7.8) page.
DEGRADE  : the kill state machine moves to DEGRADE (stop new entries) AND pages.
HALT     : the kill state machine moves to HALT (flatten to venue stops) AND pages
           with highest priority. (7.1, 7.5, 7.6-hard, 7.4-exit, 7.3-escalated.)
```

Severity is a property of the *trigger* (Section 7), tabulated and AMBER-routed, not
re-decided per alarm. A LOG never escalates itself; escalation is a *different
trigger* firing (e.g. CUSUM `h` → page, CUSUM `h₂` → halt).

### 8.2 Dedup and storm suppression

A breached detector that stays breached must not re-page every bar. Dedup is
deterministic: an alarm fingerprint = `(stream, detector, severity)`; the first
crossing pages, subsequent crossings within the AMBER dedup window are coalesced
(counted, logged, not re-paged); a *new* fingerprint or an *escalation* (higher
severity) always pages. This is alarm-state hysteresis, not threshold hysteresis —
the limit never moves (P2/LIM), only the *notification* is suppressed.

### 8.3 The paging cadence is the ARL math (so a human is paged rarely, a real break caught fast)

The panel false-alarm rate `≈ Σ_d(1/ARL₀,d)` summed over the **independent alarm
sources** (Section 6.6 — single detectors at their own ARL₀, 2-of-2 AND-pairs at
their joint ARL₀ §6.6a, the edge stream at its §6.5a ARL₀-equivalent — NOT the raw
stream count nor the raw detector count) sets the *expected page rate in control*.
The RED config picks each source's ARL₀ (per the stream's sample
frequency) so the panel pages a human **roughly every few weeks of in-control
operation** — rare enough that a page means something — while ARL₁ at the design
shift keeps real-break detection in **tens of samples**. This two-sided budget is
the entire reason the limits are derived from ARL rather than picked by eye: a
3σ-by-gut threshold has an *unknown* false-alarm rate; an ARL-derived one is pinned
on both sides.

### 8.4 Re-arming is gated, deliberate, and logged (P6)

A kill never auto-un-trips. Moving HALT→DEGRADE→RUN requires: (a) the triggering
detector back in control for a RED re-arm window (e.g. the detector statistic below
limit for `M_rearm` consecutive samples), AND (b) for HALT, an explicit operator
acknowledgement (the system surfaces the trigger chain and the human re-arms). The
re-arm transition is logged with the full evidence. An automatic re-arm on the next
clean bar is forbidden (a stream that flickers clean-then-broken would otherwise
re-enter a broken regime silently).

> **Invariant ALARM — severity is trigger-defined, dedup is deterministic, paging cadence is ARL-budgeted over independent sources, re-arm is gated.** Test: a detector held breached for 100 bars pages once (plus the coalesced count), not 100 times; an escalation always pages; the simulated in-control panel page rate matches `Σ_d(1/ARL₀,d)` summed over independent alarm sources (§6.6, with AND-pairs counted once at their joint ARL₀ and the edge stream at its §6.5a ARL₀-equivalent) within CI — NOT the raw stream count nor the raw detector count; a HALT does not return to RUN without the re-arm window AND the acknowledgement flag.

---

## 9. Safe-state / graceful-degradation policy (deterministic)

On partial failure, the response is a **deterministic policy**, not discretion, and
it leans on EXECUTION's venue-resident reduce-only stops (P6 of EXECUTION) so the
risk control survives an OPS/bot outage:

```
FAILURE                         -> SAFE-STATE DIRECTIVE
feed gap (single venue stale)   -> stop-NEW-ENTRIES on that venue's instruments;
                                   HOLD existing (venue stops remain live); page.
heartbeat loss (bot blind)      -> HALT: the venue reduce-only stops are already
                                   resting (EXEC P6/S); cancel working entry orders;
                                   do NOT place new orders blind. On recovery,
                                   reconcile (SYSTEM) before re-arming (8.4).
clock-skew over tolerance       -> HALT: every funding-stamp/bar-timestamp is suspect;
                                   flatten is unsafe to compute too, so HOLD to venue
                                   stops and page for manual intervention.
venue down (orders rejected)    -> HOLD to venue stops on that venue; stop-new-entries
                                   there; continue other venues; page.
reconciliation mismatch         -> HALT + flatten to the KNOWN venue state (the venue
                                   is ground truth); page. (Does not re-derive recon.)
exit cannot fill (fill collapse
  on a protective action)       -> escalate to MARKET via EXECUTION (exits never
                                   abort, P5/P6 EXEC); if still unfilled, the venue
                                   reduce-only stop is the backstop; page HALT.
detector self-diagnostic fail   -> treat that stream as OUT of control (P6); degrade
                                   on its trigger class; page.
```

The unifying rule (P6): **every partial failure resolves to the state that cannot
lose more than the venue stop allows.** The venue reduce-only stop — placed by
EXECUTION the moment a position opens and reconciled on every restart — is the floor
under all of this; OPS's degradation policy is about *not making it worse* while the
bot is impaired, never about substituting in-process logic for the venue stop.

### 9.1 The HALT → flatten path: idempotency, race-freedom, and flatten-order failure

OPS emits a flatten **directive** consumed **asynchronously** by the live loop
(P1) — it does not place orders itself. That asynchrony creates three degenerate
paths the kill-switch state machine must pin so the brake is itself safe.

- **(a) Idempotency — the HALT → flatten directive is issued exactly once.** Because
  the directive is consumed asynchronously and triggers re-evaluate every bar, a
  multi-trigger or multi-bar HALT could otherwise fire `N` flatten/cancel waves. The
  `kill_switch` state machine carries a **`flatten_issued` latch**: once HALT is
  latched and a flatten has been issued, subsequent triggers/bars do **NOT** re-issue
  a duplicate flatten/cancel. Re-entering HALT from DEGRADE issues flatten **once**;
  the directive consumed twice by the live loop is a **no-op**. The latch clears only
  on a gated re-arm (§8.4). (Required because the directive is asynchronous — without
  the latch the brake itself becomes a source of order storms.)
- **(b) Race-freedom — HALT concurrent with an in-flight entry/fill.** When HALT fires
  while an entry order is working or a fill is in flight, the ordering is pinned:
  (i) **cancel working ENTRY orders first**, then (ii) issue the **reduce-only flatten**.
  A fill that arrives **between** (i) and (ii) is reconciled into the flatten on the
  next bar (lean on SYSTEM reconciliation, P5) — so a position opened in the race
  window is still flattened. The position is **never uncovered**: EXECUTION's
  venue-resident reduce-only stop (EXEC P6) is **already resting on it the moment it
  opens**, so even a position born in the race window — and even a flatten that has not
  yet been acknowledged — has the venue stop as its floor.
- **(c) Flatten-order failure is a well-defined SAFE state (extends the
  `exit cannot fill` row above to the flatten DIRECTIVE itself).** If the flatten order
  is **REJECTED** (venue down) or only **PARTIALLY** fills, the safe state is
  explicitly: (i) the **venue-resident reduce-only stop remains the floor** (already
  placed, EXEC P6); (ii) OPS stays **latched HALT** and **re-attempts flatten on a RED
  retry cadence** (`flatten_retry_interval`, `flatten_retry_max`); (iii) it pages
  **`HALT-unconfirmed`** (highest priority); and (iv) it **NEVER treats an
  un-acknowledged / un-confirmed flatten as "flat"** — there is no silent transition to
  a safe-looking RUN/clear state while the brake itself is impaired. A flatten is
  "confirmed flat" only on a reconciled venue-vs-book zero-position (SYSTEM, P5),
  never on having merely *sent* the order.

> **Invariant SAFE — degradation is deterministic, leans on venue-resident stops, the HALT→flatten path is idempotent and race-free, and failures fail toward SAFE.** Each failure maps to exactly one directive (above); a missing/stale stream maps to a *more* restrictive state, never a permissive one; no degradation path requires the bot process to remain alive to control risk (the venue stop does). The HALT→flatten directive is **idempotent** (`flatten_issued` latch, §9.1a) and **race-free** (cancel-entries-then-flatten, §9.1b); a **rejected/partial flatten is a latched-HALT-with-retry-and-page state, never a spurious "flat"** (§9.1c). Tests: killing the OPS process in a synthetic open-position scenario leaves the venue reduce-only stop resting and a gap-through fires it without the bot; a stale-feed synthetic produces stop-new-entries + hold, not flatten-blind; a **double-consumed HALT directive issues exactly ONE flatten wave** (latch); a synthetic where the **flatten order is REJECTED** leaves the state **latched HALT with the venue stop resting and a retry + `HALT-unconfirmed` page, NEVER a spurious "flat" verdict**; a HALT firing concurrently with an in-flight entry cancels the entry first, then flattens, and a fill landing in the race window is flattened on the next bar (and was covered by the resting venue stop the moment it opened).

---

## 10. Worked example — a drifting cost stream, the CUSUM crossing, the ARL, the kill (reproduce bit-for-bit)

The centerpiece worked example: the realized cost stream (§5.2, `e_cost`) drifts up
(realized cost begins to exceed even COST's conservative forecast — the cost model
has gone stale), the CUSUM on the standardized residual crosses its limit at a
stated sample, the ARL confirms the timing, and the kill-switch fires. Every number
reproduces from the inputs with no hidden step. This is the regression fixture
(Invariant WE).

**In-control parameters (frozen, from SYSTEM's validation report):**

| Input | Value |
|---|---|
| `μ₀` (in-control mean TCA residual `e_cost`) | `0.0` bps (realized = forecast in control) |
| `σ₀` (in-control std of `e_cost`) | `2.0` bps |
| CUSUM slack `k` (detect a 1σ shift) | `0.5` (σ units) |
| CUSUM interval `h` (DEGRADE) | `4.77` (σ units) → two-sided ARL₀ ≈ 370 (one-sided ≈ 743, §6.6) |
| PH confirmation window `M_conf` (2-of-2, §6.6a) | `3` samples |
| CUSUM interval `h₂` (escalate to HALT) | `8.0` (σ units) |
| design shift to detect | `δ_shift = 1.0σ = 2.0` bps |

**The live stream.** Samples 1–10 are in control (`e_cost` noise around 0); from
sample 11 the cost model goes stale and realized cost runs `+3.0` bps above forecast
every fill (a `+1.5σ` persistent shift). Standardize `z_i = (e_cost_i − 0)/2.0`:

```
i:   1     2     3     4     5     6     7     8     9    10  | 11   12   13   14   15   16   17   18
e:  +0.4 −1.0 +0.8 −0.6 +1.2 −0.4 +0.2 −0.8 +0.6 −0.2 |+3.0 +3.0 +3.0 +3.0 +3.0 +3.0 +3.0 +3.0
z:  +0.2 −0.5 +0.4 −0.3 +0.6 −0.2 +0.1 −0.4 +0.3 −0.1 |+1.5 +1.5 +1.5 +1.5 +1.5 +1.5 +1.5 +1.5
```

**Upper CUSUM recursion** `C⁺_i = max(0, C⁺_{i-1} + z_i − k)`, `k = 0.5`:

```
i=1 : max(0, 0   + 0.2 − 0.5) = 0.0
i=2 : max(0, 0.0 − 0.5 − 0.5) = 0.0
i=3 : max(0, 0.0 + 0.4 − 0.5) = 0.0
i=4 : max(0, 0.0 − 0.3 − 0.5) = 0.0
i=5 : max(0, 0.0 + 0.6 − 0.5) = 0.1
i=6 : max(0, 0.1 − 0.2 − 0.5) = 0.0
i=7 : max(0, 0.0 + 0.1 − 0.5) = 0.0
i=8 : max(0, 0.0 − 0.4 − 0.5) = 0.0
i=9 : max(0, 0.0 + 0.3 − 0.5) = 0.0
i=10: max(0, 0.0 − 0.1 − 0.5) = 0.0        <- in control: C+ pinned near 0 (does NOT drift on noise)
--- shift begins (each step adds z−k = 1.5 − 0.5 = +1.0) ---
i=11: max(0, 0.0 + 1.5 − 0.5) = 1.0
i=12: max(0, 1.0 + 1.0)       = 2.0
i=13:                            3.0
i=14:                            4.0
i=15:                            5.0   <- crosses h = 4.77  => DEGRADE alarm at sample 15
i=16:                            6.0
i=17:                            7.0
i=18:                            8.0   <- crosses h2 = 8.0  => escalate to HALT at sample 18
```

**Detection delay vs the ARL.** The shift began at sample 11; the DEGRADE CUSUM
crossed at sample 15 — a detection delay of **4 samples** for a `+1.5σ` shift. The
exact (tabulated / Markov-chain) out-of-control ARL₁ for `(k=0.5, h=4.77)` at a
1.0σ shift is ≈ 10.4 samples — the Siegmund closed form of §6.6 gives ≈ 9.9, the
two agreeing to ~5% and both ≈ the design target of 10; a 1.5σ shift is larger, so
the realized 4-sample delay is consistent with (faster than) the 1σ ARL₁.
The in-control C⁺ stayed pinned at ~0 through samples 1–10 (two-sided ARL₀ ≈ 370 —
i.e. one-sided ≈ 743 per arm, combined `÷2` per §6.6 — so a false alarm in the first
10 in-control samples is improbable, and none occurred).

**The kill fires (Invariant KILL).** At sample 15 the cost-divergence trigger (7.3)
emits **DEGRADE**: stop new entries, hold existing, page — but only because the
Page-Hinkley arm of the 2-of-2 AND-rule is *also* breached within the `M_conf = 3`
confirmation window (§6.6a); the windowed AND, whose dependence-aware joint ARL₀ is
the cost stream's single panel source (§6.6), is what makes this a real divergence
and not a single-chart artifact. (Had PH not confirmed within the window, the CUSUM
crossing would page but not DEGRADE.) The residual keeps drifting and at sample 18 C⁺ crosses
`h₂ = 8.0`, escalating the trigger to **HALT**: flatten to EXECUTION's venue
reduce-only stops, cancel working orders, page highest-priority, and write the full
trigger chain to the forensic log (which feeds COST.md's stub re-tighten per
Invariant XC). The system does not auto-re-arm (8.4); a human acknowledges and the
cost model is re-validated overnight before RUN resumes.

```
verdict at i=15 : state = DEGRADE, trigger = cost_divergence,
                  C_plus = 5.0 > h = 4.77, stream = cost_tca, sample = 15,
                  in_control = (mu0=0.0, sigma0=2.0, k=0.5, h=4.77,
                                ARL0_2sided≈370 [one-sided≈743], ARL1@1σ≈10)
verdict at i=18 : state = HALT, trigger = cost_divergence_escalated,
                  C_plus = 8.0 ≥ h2 = 8.0, response = flatten_to_venue_stops
```

> **Invariant WE — the worked example is a regression fixture.** A test reproduces
> every number in Section 10 from the tabulated inputs: the `z_i` sequence, the full
> `C⁺_i` recursion (pinned ~0 in control, crossing `h=4.77` at i=15 and `h₂=8.0` at
> i=18), the DEGRADE-then-HALT escalation, and the recorded in-control parameters.
> The Siegmund ARL approximation for `(k=0.5,h=4.77)` reproduces ARL₀ ≈ 370 and
> ARL₁@1σ ≈ 9.9 within tolerance; the exact (Markov-chain) ARL₁@1σ ≈ 10.4 is
> reproduced SEPARATELY by the MC check (the two ARL₁ figures agree to ~5%, so 10.4
> is asserted against the MC value, not against Siegmund's). If any RED limit changes, this fixture changes
> with it — it is the single anchor pinning the detector math and the kill
> escalation.

---

## 11. Validation — limits derived offline, validated OOS inside SYSTEM's framework

Validation is part of the module (mirror of the sibling specs). The OPS-specific
validation is **of the limits themselves**, run inside SYSTEM.md's purged-
walkforward / sampled-CPCV firewall so the limits never see the data they are
promoted on.

### 11.1 Deriving the in-control distribution and the limits

`validate/derive_limits.py` consumes SYSTEM's validation report (the OOS path-return
series, the per-trade edge distribution, the realized-vs-forecast cost/funding
residuals, the vol-forecast errors — all produced under SYSTEM's firewall) and
computes `(μ₀, σ₀)` per stream and, given a target ARL₀ + design shift, the
`(k,h)`, `(λ_ewma,L)`, `L_s`, `(α,β,A,B)`. These are written to the RED config and
frozen. They are derived ONCE per validated config (and re-derived only when SYSTEM
promotes a new config overnight, Section 7.5 of SYSTEM), never live.

### 11.2 OOS validation of the limits (no P&L objective — Invariant OOS)

The BLUE limit grids are searched and promoted by a *statistical* objective on
held-out folds: the in-control **false-alarm rate** (does the panel page roughly at
`Σ_d(1/ARL₀,d)`, the source-counted budget of §6.6, on genuinely in-control OOS data?) and the **detection delay** at injected
synthetic shifts (does ARL₁ hold OOS?). The objective contains **no P&L term**
(Invariant OOS) — OPS limits are graded on detection quality, not on whether
halting happened to save money on one history (which would overfit the limits to
that history's drawdowns).

### 11.3 Baselines and ablations

```
Baselines (the panel must beat these on detection-delay-at-fixed-false-alarm):
  - NO-MONITOR        : no detector; measures the loss a real break causes unhalted.
  - NAIVE-3sigma-ONLY : Shewhart only, no CUSUM/EWMA/SPRT; blind to small persistent
                        drifts (the edge-decay / cost-staleness case) — shows why
                        sequential detectors are needed.
  - FIXED-WINDOW-MEAN : a fixed-N rolling-mean t-test instead of sequential; shows it
                        is either too slow (blind) or too small-N (floods).
Ablations (turn one detector off):
  - which detector catches which synthetic break (drift / jump / staleness / recon);
  - a detector that NEVER fires on any synthetic break is mistuned or redundant;
  - a detector that floods on in-control OOS data has too-tight limits.
```

> If the panel does not beat NO-MONITOR on loss-avoided-per-false-alarm across the
> synthetic breaks, the limits are mistuned — re-derive, do not loosen by gut. If a
> detector never fires even on its designed synthetic break, it is mistuned or
> redundant (mirror of LEVERAGE's "ceiling that never binds" / TREND's Final Rule).

---

## 12. Synthetic data requirements (`synth/generate.py`)

Generate adversarial streams so the module is testable with no live feed:
- **In-control stream** (the ARL₀ check: detectors should rarely fire at the
  *two-sided* per-stream ARL₀; the panel false-page rate matches `Σ_d(1/ARL₀,d)` over
  independent sources, §6.6).
- **Autocorrelated + fat-tailed in-control stream** (AR(1)-Student-t per the measured
  `ρ`/dof, §6.7(d)): the realized MC two-sided ARL₀ lands within CI of `370` ONLY after
  prewhitening (§6.7a) and/or `h`-inflation by `VIF` (§6.7b) — and FAILS (pages far too
  often) if the raw autocorrelated residual is fed the IID limit; the empirical-tail
  Shewhart `L_s` holds its `370` where the Gaussian-3σ limit would flood (§6.7c).
- **Warm-up transient stream** (in-control with a transient excursion inside `N_warm`):
  the detector emits zero PAGE/DEGRADE/HALT before `N_warm` (logs only) while its
  recursion updates correctly; the binary recon/heartbeat/clock-skew triggers fire in
  the first bar regardless of warm-up (§6.8).
- **Live-variance-shift stream** (live vol 2× validated, mean unchanged): the
  detector's trip point in RAW units must NOT move (standardization uses frozen `σ₀`,
  Invariant LIM-(b)); the logged `σ_live/σ₀` diagnostic does move.
- **Repeated edge-epoch stream** (in-control hit-rate, many CUSUM-SPRT epochs): the
  *lifetime* false-halt rate matches the §6.5a budgeted ARL₀-equivalent, NOT the
  per-epoch `α` (which a bare restart-SPRT would inflate to `1−(1−α)^K`).
- **Slow cost drift** (the Section 10 worked example: `e_cost` ramps; CUSUM/PH fire).
- **Vol-regime jump** (`e_vol` step up; EWMA fires; DEGRADE).
- **Edge decay** (hit-rate steps `p₀ → p₁`; SPRT decides H₁ in ~ASN trades).
- **Trade-frequency collapse** (Shewhart on count; the system stops finding trades).
- **Single catastrophic fill** (Shewhart fires in one sample; CUSUM lags — shows the
  Shewhart+CUSUM complementarity).
- **Fill-rate collapse** (orders stop filling; DEGRADE → HALT on exit-fail).
- **Reconciliation mismatch** (binary; immediate HALT, no SPC).
- **Heartbeat loss / feed staleness / clock-skew** (binary; HALT/DEGRADE; session-
  aware so a market close is NOT a false outage).
- **Drawdown ramp through governor → backstop** (equity draws to 43% governor-flat,
  then sits at ≥ 45% → OPS HALT; shows the two-brake boundary on a SHARED equity/peak
  reference so 45% > 43% never inverts).
- **Gap-through-both-thresholds** (equity at 38% DD gaps in one bar past 43% AND 45%):
  the DD_kill fires AFTER the loss — it does NOT prevent gap ruin (§7.1a); only the
  venue reduce-only stop + `L_tail` gap ceiling bound this. The synth asserts the
  honest concession, not a false guarantee.
- **Bot-kill mid-position** (the venue reduce-only stop fires without the bot — the
  safe-state floor of Section 9).
- **Double-consumed HALT directive** (multi-trigger / multi-bar HALT): exactly ONE
  flatten wave is issued (`flatten_issued` latch, §9.1a).
- **HALT during in-flight entry/fill** (race window): entry is cancelled first, then
  flatten issued; a fill landing between is flattened next bar and was covered by the
  resting venue stop the moment it opened (§9.1b).
- **Flatten-order rejected / partial** (venue down on the flatten): state stays latched
  HALT with the venue stop resting, a RED-cadence retry, and a `HALT-unconfirmed` page —
  NEVER a spurious "flat" verdict (§9.1c).
- **Flapping stream** (clean-broken-clean — re-arm must NOT auto-un-trip).

Behavioral assertions: detectors fire on their designed break and stay quiet in
control at the stated ARL₀; the kill state is precedence (max-severity); DEGRADE
holds existing positions and stops new entries; HALT flattens to venue stops;
re-arm requires the window + acknowledgement; the panel false-page rate matches
`Σ_d(1/ARL₀,d)` over independent sources (§6.6, not the raw stream/detector count);
OPS never emits a trade of its own edge logic.

---

## 13. Module acceptance gates (CI)

### Streams
Pass when: every stream is a live-vs-validated residual or a binary event,
standardized against frozen `(μ₀,σ₀)` from `config/in_control.py` (Invariant STREAM);
all observations are `F_t`-measurable (Invariant CAUSAL); no live limit refit (LIM).

### Detectors
Pass when: CUSUM/PH/EWMA/Shewhart/SPRT/CUSUM-SPRT each match their closed-form
recursion and their tabulated ARL (Invariant ARL) — the CUSUM ARL₀ is the *MC*
two-sided value (analytic one-sided Siegmund `÷2` retained as seed/cross-check), the
SPRT ASN is the *boundary-weighted* Wald form (ASN_H₀≈99, ASN_H₁≈118); each is
O(1)/causal (CAUSAL); SPC alarms are suppressed before `N_warm` while the recursion
still updates and the binary triggers are unaffected (§6.8); the §6.7a prewhitening is
causal; detectors are isolated (no detector reads another's output); standardization
uses frozen `σ₀` not a live `σ` (LIM-(b)).

### ARL / limits
Pass when: `validate/arl.py` reproduces ARL₀/ARL₁ for every pinned operating point
within tolerance (MC two-sided CUSUM ARL₀ — distinct from the analytic `÷2`, with the
dependence gap reported; EWMA ARL; **empirical-tail Shewhart `L_s`** §6.7c;
boundary-weighted SPRT ASN; the windowed-AND joint ARL₀ §6.6a; the edge stream's §6.5a
lifetime ARL₀-equivalent); the **realized ARL₀ on the AR(1)-Student-t in-control
synthetic lands within CI of 370 after prewhitening/`VIF`-inflation** (§6.7d) and FAILS
on the raw autocorrelated residual with the IID limit; `derive_limits.py` solving for
ARL₀=370 returns `h≈4.77` not ≈3.5; the closed forms are treated as approximation +
seed with the MC solver authoritative on disagreement; the panel false-alarm rate
`Σ_d(1/ARL₀,d)` over independent sources is budgeted into the paging cadence; limits
trace to the validated distribution and are frozen (LIM, OOS).

### Kill-switch / triggers
Pass when: each trigger fires its tabulated response on a stream crossing exactly
its limit; the kill state is precedence/max-severity, never a blended score
(Invariant KILL); the three brakes compose by min()-on-size + precedence-on-state,
OPS owns only the binary state, and the 45% halt sits strictly above the 43%
governor flat (Invariant BRAKE).

### Alerting
Pass when: severity is trigger-defined; dedup coalesces a held breach to one page +
count; escalation always pages; re-arm requires window + acknowledgement and never
auto-un-trips (Invariant ALARM).

### Safe-state
Pass when: each partial failure maps to its deterministic directive; failures fail
toward SAFE; killing the OPS process leaves the venue reduce-only stop as the floor;
the HALT→flatten path is idempotent (a double-consumed HALT issues exactly one flatten
wave) and race-free (cancel-entries-then-flatten); a rejected/partial flatten stays
latched HALT with the venue stop resting + retry + `HALT-unconfirmed` page and never a
spurious "flat" (Invariant SAFE, §9.1).

### Reconciliation boundary
Pass when: OPS consumes `reconciliation_ok` + detail and HALTs on mismatch, and does
NOT re-derive reconciliation (Invariant RECON, PROV).

### Worked example
Pass when: the Section 10 fixture reproduces bit-for-bit — the `z_i` sequence, the
`C⁺_i` recursion, the DEGRADE@15 / HALT@18 escalation, and the ARL numbers
(Invariant WE).

### Provenance / firewall
Pass when: nothing in `streams/`, `detectors/`, `reconcile/`, `brake/` imports from
`validate/` or `forensics/`; OPS emits no trade/size/exit of its own edge logic; no
RED limit is per-instrument or tuned on live P&L (Invariant PROV, C, OOS).

---

## 14. Invariants (enforce structurally; ask before breaking)

**Invariant PROV — OPS monitors, never trades.** No `ops/` code opens, closes,
reduces, routes, or re-sizes a position from its own edge/sizing/exit logic; OPS
emits only `{run, degrade, halt}` + alarms + a flatten directive consumed by the
live loop via EXECUTION's venue stops. Test: OPS holds no edge/sizing/exit
parameter; the only outputs are the directive and alarms.

**Invariant LIM — limits derived from the validated distribution and frozen.**
(Section 4.) Limits are computed offline from `(μ₀,σ₀)` + ARL target and immutable
at runtime; no live recompute from the policed stream.

**Invariant OOS — limits validated out-of-sample, never tuned on live P&L.**
(Section 4/11.) The promotion objective is detection quality (false-alarm rate +
delay), never P&L.

**Invariant CAUSAL — every detector is an O(1) causal recursion; SPC alarms are
suppressed before warm-up `N_warm`.** (Section 6, §6.8.) No future sample, no growing
window; one-at-a-time and batch states match; binary triggers are unaffected by warm-up.

**Invariant CHEAP — the whole panel is O(streams) per bar.** A few dozen scalar
updates; cheaper than any one pillar's per-bar evidence on the target VM. Test: the
per-bar OPS cost on the target VM is below a RED budget (sub-millisecond for the
scalar recursions) and does not grow with history length.

**Invariant ARL — every detector carries a stated ARL₀/ARL₁ and the panel
false-alarm rate is budgeted; closed forms are approximations/seeds and the
dependence-aware MC solver is authoritative; the realized ARL₀ holds on an
autocorrelated + fat-tailed in-control synthetic.** (Sections 6.6, 6.7.)

**Invariant KILL — frozen thresholds, named justification, deterministic response,
precedence (max-severity) state, never a blended score.** (Section 7.)

**Invariant BRAKE — three distinct brakes: REGIME gross taper + LEVERAGE governor
(gradual, min()-composed, owned by them) + OPS binary state (precedence).** OPS
never tapers gross, never re-implements the governor ladder; the 45% halt is
strictly above the 43% governor flat. (Section 7.1a.)

**Invariant RECON — reconciliation is SYSTEM's; OPS monitors the OUTPUT.** OPS reads
`reconciliation_ok` + detail and halts on mismatch; it does not re-derive
position/order/account reconciliation. (Section 5.5/P5.)

**Invariant STREAM — every monitored stream is a live-vs-validated residual or
binary event, standardized against frozen in-control parameters.** (Section 5.)

**Invariant ALARM — severity trigger-defined, dedup deterministic, paging cadence
ARL-budgeted, re-arm gated (no auto-un-trip).** (Section 8.)

**Invariant SAFE — degradation is deterministic, leans on venue-resident reduce-only
stops, fails toward SAFE, and the HALT→flatten path is idempotent + race-free.** No
degradation path needs the bot alive to control risk; a double-consumed HALT issues
one flatten wave; a rejected/partial flatten stays latched HALT with the venue stop +
retry + page, never a spurious "flat". (Sections 9, 9.1.)

**Invariant AUDIT — every alarm and kill is append-only logged with its full trigger
chain and is replayable; the log feeds COST.md's TCA and SYSTEM's re-validation.**
Test: any halt is reconstructed from the log alone (`forensics/replay.py`) including
the stream, detector, statistic, limit, sample index, and in-control parameters.

**Invariant WE — the worked example is a regression fixture.** (Section 10.)

**Invariant V — validation/forensics never imported by live layers.** Nothing in
`streams/`, `detectors/`, `reconcile/`, `brake/` imports from `validate/`.

---

## 15. Boundaries with neighbouring pillars (pin explicitly)

- **OPS vs SYSTEM.** SYSTEM = validation (the firewalled DSR gate that proves an
  edge exists in-sample-from-the-validated-distribution) + composition +
  reconciliation LOGIC. OPS = continuous LIVE SPC monitoring of whether production
  still matches that validated distribution + alarms + the binary kill on top.
  SYSTEM produces the in-control distribution OPS polices against; OPS never
  re-validates and never re-derives reconciliation. The handoff is one-way at
  promotion time (SYSTEM → OPS limits) and one-way at break time (OPS forensic log →
  SYSTEM's next overnight re-validation).

- **OPS kill-switch vs LEVERAGE governor vs REGIME multiplier — three distinct
  brakes (Section 7.1a, Invariant BRAKE).**
  - REGIME multiplier: **gradual macro GROSS taper** — slow, smooth scalar on total
    gross as the macro regime turns risk-off.
  - LEVERAGE governor: **gradual PER-POSITION size taper** — medium, smooth ladder
    (1.00→…→0.00 at 43% drawdown).
  - OPS kill-switch: **binary safety HALT** — the discontinuous backstop a smooth
    taper cannot be, at a hard floor (45%) above the governor flat, and on any SPC /
    reconciliation / heartbeat trigger.
  Gradual scalars compose by `min()` on size; the binary state composes by
  precedence. They are never summed or multiplied into one number.

- **OPS vs COST.** COST forecasts cost/funding and produces the realized-vs-forecast
  TCA. OPS monitors the TCA RESIDUAL as a stream and brakes on its drift; the
  forensic log feeds COST's conservative-stub re-tighten (Invariant XC). OPS does
  not re-price anything.

- **OPS vs EXECUTION.** EXECUTION fills orders and places the venue-resident
  reduce-only protective stop (P6/Invariant S of EXECUTION). OPS monitors the
  realized IS / fill-rate / adverse-selection streams and, on a kill, FLATTENS via
  those venue stops — leaning on them rather than substituting in-process logic.
  OPS does not choose tactics or model fills.

- **OPS vs LEVERAGE (drawdown reference).** OPS reads the same portfolio equity /
  peak Nautilus exposes and the planned-loss/equity-ladder LEVERAGE defines as the
  drawdown reference; its max-DD kill is strictly above the governor's flat. OPS
  does not size and does not re-implement the ladder.

---

## 16. Build order

1. `types.py` and `synth/generate.py` (the adversarial streams of Section 12).
2. `config/` with TEAL/AMBER/RED/BLUE + `in_control.py` sourced from a SYSTEM report;
   test Invariant C, LIM.
3. `detectors/` (cusum, page_hinkley, ewma, shewhart, sprt + the CUSUM-SPRT
   change-point form of §6.5a) + `validate/arl.py`; test each recursion + its ARL,
   the two-sided CUSUM ARL₀ **solved against the MC two-sided chart** (not the analytic
   `÷2`), the **realized ARL₀ on the AR(1)-Student-t in-control synthetic** after
   prewhitening/`VIF`-inflation (§6.7), the **empirical-tail Shewhart `L_s`** (§6.7c),
   the per-detector **warm-up suppression** (§6.8), the boundary-weighted SPRT ASN, the
   windowed AND-rule joint ARL₀, and the edge-stream lifetime budget (Invariant CAUSAL,
   ARL).
4. `streams/` adapters (fills, cost_tca, vol, edge, recon, heartbeat) + `streams/whiten.py`
   (the §6.7a causal prewhitening / §6.7b batch means); test Invariant STREAM, CAUSAL.
5. `reconcile/streams_map.py` (bind streams to detectors with Section-6 limits).
6. `brake/triggers.py` (Section 7) + `brake/kill_switch.py` (the state machine, with
   the `flatten_issued` latch + flatten-reject retry of §9.1); test Invariant KILL,
   BRAKE (incl. single-flatten-wave idempotency).
7. `brake/alarms.py` (severity + dedup + ARL paging) and `brake/safe_state.py`;
   test Invariant ALARM, SAFE.
8. `brake/engine.py` + `nautilus/adapter.py` + `nautilus/flatten.py` (the live wire
   to Nautilus + EXECUTION venue stops); test Invariant PROV, RECON, SAFE.
9. `forensics/log.py` + `forensics/replay.py`; test Invariant AUDIT.
10. The Section-10 worked-example regression fixture; test Invariant WE.
11. `validate/derive_limits.py`, `validate/oos.py`, `validate/baselines.py`,
    `validate/reports.py`; test Invariant OOS and the baselines/ablations of 11.3.

Each step complete only when its acceptance gate (Section 13) is green. A step that
fails its gate does not advance.

Stop after step 11. The remaining questions — which detector actually catches which
break on YOUR live stream, whether the per-stream ARL₀ pages a human at a tolerable
rate, whether the edge-SPRT's `p₁` is the right decay to halt for — are answered by
RUNNING the panel on the paper → canary → scaled rollout (Section 17), not by adding
detectors.

---

## 17. Operational lifecycle + staged rollout with quantitative go/no-go gates

OPS owns the staged rollout because each stage gate is a **statistical test of
live-vs-validated** — the same reconciliation streams (Section 5), now used as
promotion gates with a minimum sample requirement. The book moves to the next stage
only when the realized streams sit **within the confidence interval of the validated
distribution over a minimum sample**, and demotes immediately on a kill.

```
STAGE         capital        gate to ADVANCE (all must hold over the min sample)
------------  -------------  --------------------------------------------------------
paper         0 (sim live)   realized edge/trade, hit-rate, trade-frequency, fill-IS,
                             and TCA residual each WITHIN the CI of SYSTEM's validated
                             distribution over >= N_paper trades; NO kill fired.
canary        tiny (min lot) same, on REAL fills, over >= N_canary trades; realized
                             cost <= COST's conservative forecast (e_cost CUSUM never
                             breached); fill-rate within CI; recon clean.
scaled        partial        same, over >= N_scaled trades; per-pillar attribution
                             (SYSTEM Section 8) still positive; max realized drawdown
                             well inside the governor ladder; no DEGRADE escalations.
full          target         same, sustained; the panel pages at ~ Σ_d(1/ARL₀,d) (§6.6, in control),
                             not more.
```

### 17.1 Each gate is a statistical test, not a calendar

A stream **cannot be a promotion gate until it has cleared BOTH its warm-up `N_warm`
(§6.8) AND its `N_stage` minimum sample** — a stream still inside warm-up reports
`warming_up` and DEFERS the advance exactly as an insufficient sample does. "No SPC
alarms during warm-up" is a **rollout invariant**: a freshly promoted SYSTEM config
(new `(μ₀,σ₀,φ)`) restarts every detector's warm-up, so the panel does not page
spuriously on its first live bars after a promotion.

The advance test for each stream is a **two-sided confidence check + a sequential
non-inferiority test** against the validated distribution:

- **Realized edge / hit-rate / frequency:** the SPRT of Section 6.5 run in
  *non-inferiority* mode (H₀: live `p ≥ p₀_validated`; advance when the SPRT
  accepts H₀ over `≥ N_stage` trades), AND a block-bootstrap CI on realized
  edge-per-trade that **excludes the no-edge null** and overlaps the validated mean.
  The minimum sample `N_stage` (RED) is set so the test has the power to detect a
  meaningful shortfall (tied to the SPRT's ASN at the chosen `(α,β)`); a smaller
  realized sample DEFERS the advance (`insufficient_stage_sample`), it never
  advances on a calendar date.
- **Fill-IS / TCA residual / fill-rate:** each within the CI of its validated
  distribution (the §5.1/§5.2 streams, in control under their CUSUM/EWMA over the
  stage sample). Realized cost must be `≤` COST's conservative forecast (Invariant
  XC of SYSTEM): an `e_cost` CUSUM breach at any stage **blocks the advance and
  triggers a demote**.
- **Vol / recon / heartbeat:** in control over the stage (no 7.x trigger fired).

### 17.2 Demotion is immediate and gated re-promotion is sequential

A kill at any stage demotes the book one stage (or to paper, by severity) — the same
`{degrade, halt}` machinery (Section 7), no separate logic. Re-promotion re-runs the
stage gate from scratch over a fresh `≥ N_stage` sample (it does not resume mid-gate)
and is subject to the re-arm gating of Section 8.4. Tying the gates to the validated
distribution (SYSTEM) and the cost model (COST) — not to a P&L milestone — is what
keeps the rollout from overfitting to a lucky live streak.

> **Invariant ROLLOUT — every stage gate is a statistical live-vs-validated test
> over a minimum sample, never a calendar or a P&L milestone, and a kill demotes
> immediately.** Test: a synthetic live stream whose realized edge sits below the
> validated CI does NOT advance regardless of elapsed time; an `e_cost` CUSUM breach
> blocks advance and demotes; advancing requires `≥ N_stage` realized trades with all
> streams in control AND every gating detector past its `N_warm` warm-up (§6.8) — a
> stream still warming up DEFERS the advance and a freshly promoted config does not
> page on its first live bars; the advance objective contains no raw-P&L-milestone term.

---

## 18. References and rationale

- **CUSUM:** Page (1954), "Continuous Inspection Schemes" — the tabular two-sided
  CUSUM and its sensitivity to small persistent shifts.
- **Page-Hinkley:** Hinkley (1971); the change-point variant against a running mean,
  used as the independent confirmation on the cost stream.
- **EWMA control chart:** Roberts (1959); Lucas & Saccucci (1990) for the
  `(λ, L) → ARL` tables and the Markov-chain ARL approximation.
- **Shewhart individuals chart:** Shewhart (1931); the 3σ single-point rule and its
  `ARL₀ = 1/(2Φ(−L_s))`, complementary to CUSUM/EWMA (fast on big shifts, blind to
  small drifts).
- **SPRT:** Wald (1945), *Sequential Analysis* — the log-likelihood-ratio test,
  boundaries `A = ln((1−β)/α)`, `B = ln(β/(1−α))`, and the **boundary-weighted**
  Average Sample Number `E[N|H] ≈ (P(hit A)·A + P(hit B)·B)/E[ΔΛ|H]` (§6.5 — NOT the
  unweighted `boundary/increment`); the minimum-expected-sample tool for the
  low-frequency edge-decay stream, but a **fixed-hypothesis** test whose per-epoch
  `α` is not its lifetime false-halt rate under repeated restarts (§6.5a).
- **SPRT → change-point:** Page (1954) / Lorden (1971) — a CUSUM on the Bernoulli
  log-LR ("CUSUM-SPRT") is the change-point detector with a single pinned in-control
  ARL₀, the v1 form for the lifetime edge stream (§6.5a) rather than a restart-SPRT.
- **ARL math:** Siegmund (1985) approximation for the **one-sided (per-arm)** CUSUM
  ARL; the two-sided chart of §6.1 combines two arms as `1/ARL₀_2sided ≈
  2/ARL₀_oneside` (so `(0.5,4.77)`: one-sided 743 → two-sided 370); the family-wise
  panel false-alarm budget summed over **independent alarm sources** `Σ_d(1/ARL₀,d)`
  (AND-pairs once at their joint ARL₀) for setting the paging cadence.
- **SPC as monitoring (not control):** the detectors monitor a process whose
  in-control distribution is GIVEN (by SYSTEM's validation), not estimated live —
  which is precisely why the limits are frozen and overfitting-proof (P2).
- **SPC on autocorrelated / fat-tailed data (§6.7):** the classical residuals-charting
  approach — prewhiten with a frozen low-order AR(p) and chart the innovations
  (Alwan & Roberts 1988; Montgomery, *Introduction to SPC*, ch. on autocorrelated
  data) — and non-overlapping batch means as the alternative decorrelator; the AR(1)
  variance-inflation `VIF=(1+ρ)/(1−ρ)` quantifying how positive autocorrelation shrinks
  the realized ARL₀; and setting the Shewhart limit from the EMPIRICAL / fitted-t (or
  EVT) tail quantile rather than the Gaussian `1/(2Φ(−L_s))` when the in-control
  residual is fat-tailed. The pinned limits are the dependence-aware Monte-Carlo solver
  output; the closed forms are the seed and cross-check.
- **Boundary with the governor / regime taper:** LEVERAGE.md Section 6 (graduated
  drawdown governor) and the REGIME gross taper are the gradual brakes; OPS is the
  binary backstop (P4) — three distinct controls, never blended.
- **Venue-resident stops as the safe-state floor:** EXECUTION.md P6 / Invariant S —
  the reduce-only exchange stop is the risk control that survives a bot outage, on
  which OPS's graceful-degradation policy leans.
- **Conservative-stub feedback loop:** SYSTEM.md Invariant XC — realized cost
  must stay ≤ the conservative exec-stub cost; OPS's TCA-residual monitor is the
  live half of that loop, feeding COST.md's stub re-tighten.

---

## 19. Non-goals (belong to sibling modules)

Edge/entry detection (BREAKOUT.md), sizing under survival constraints and the
drawdown governor (LEVERAGE.md), profitable-exit management (TREND.md), instrument
selection (the Carver-Ch.25 selection module), order tactics and the venue stop
(EXECUTION.md), the cost/funding model and forecast (COST.md), composition +
firewalled validation + reconciliation logic (SYSTEM.md), and the macro regime gross
taper (REGIME). OPS consumes the realized streams these emit, compares LIVE to
VALIDATED with classical SPC, and brakes. It never decides what to trade, how much,
how to fill, what it costs, or whether the edge was real — only whether the live
system is still the validated one, and what to do the moment it is not.
