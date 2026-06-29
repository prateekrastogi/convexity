# Convexity — Implementation Architecture

This is the canonical implementation architecture for the convexity system: a multi-pair
directional convex-breakout trader on Binance USDT-M perpetuals, composing on NautilusTrader. The
**specs in `docs/specs/{BREAKOUT,TREND,LEVERAGE,EXECUTION,COST,SYSTEM,OPS,REGIME}.md` are the
authoritative source** (all ≥9.5/10). This document maps those eight specs onto a single Python
package, fixes the cross-spec interface mismatches the spec `risks` sections flagged, and defines
the shared foundation + test strategy.

---

## 1. Architecture principle

Every pillar is split into two layers:

* **Pillar POLICY (pure-Python, NautilusTrader-FREE).** All the modelling, detection, sizing, cost,
  regime, monitoring and orchestration logic. Depends only on `numpy`/`pandas`/`scipy` and the
  shared `convexity.core`. **Unit-testable without the engine** — the test venv (`.venv-test`) has
  no `nautilus_trader` installed, and every pillar test runs `PYTHONPATH=src .venv-test/bin/python
  -m pytest`.
* **Thin Nautilus ADAPTER.** A small number of clearly-isolated modules — `strategy/`,
  `<pillar>/nautilus/`, `cost/facts/nautilus_adapter.py` — that translate between Nautilus
  events/state and the pure-Python pillar types. These are the *only* files that `import
  nautilus_trader`. They are covered by integration tests on the VM (where nautilus is installed),
  not by the pillar unit suite.

The single live `strategy/convexity_strategy.py` Nautilus `Strategy` subclass **composes** the
pillars: it builds the pure-Python inputs from Nautilus bars/events, calls each pillar's outward
method, and routes the returned commands/intents back through the Nautilus adapters.

This split is *why* the foundation (`convexity.core`) is the thing that makes pillars interoperate:
the boundary objects that cross pillar lines must be one shared type, not redefined per pillar.

---

## 2. Package layout

The specs each write their directory tree at a *bare* top level (`breakout/`, `trend/`, …). The
repo packages everything under `src/convexity/` (`pyproject` `packages=["src/convexity"]`, tests
run `PYTHONPATH=src`). **Resolution (applied uniformly):** every spec path `<pillar>/…` maps to
`src/convexity/<pillar>/…`, imported as `convexity.<pillar>.…`. Every spec-prescribed *filename* is
preserved verbatim under that prefix — `range_geometry.py`, `detect/combine.py`, `validate/arl.py`,
`components/funding_forecast.py`, `ceilings/stop.py`, etc.

```
src/convexity/
  core/                      # SHARED FOUNDATION (implemented now)
    params.py                #   RED/AMBER/BLUE/TEAL governance + typed param registry
    types.py                 #   cross-pillar boundary dataclasses (Bars, Side, EntrySignal, …)
    protocols.py             #   the composition-seam Protocols each pillar implements

  selection/                 # ALREADY IMPLEMENTED (Carver Ch.25 dynamic optimisation + universe)
    dynamic_optimization.py  #   greedy tracking-error optimiser (19 passing tests — DO NOT regress)
    universe.py  UNIVERSE.md

  breakout/                  # BREAKOUT pillar  (entry detector)
    types.py pipeline.py
    config/{defaults,instrument,asset_class,tuning,cost,resolve}.py
    data/{schema,calendar,corporate_actions,rolls,quality}.py
    standardize/{returns,seasonal,volatility,tails,liquidity}.py
    features/{range_geometry,compression,trend_state,flow,jumps,context}.py
    detect/{candidates,glr,bocpd,empirical_null,evidence,combine,dedupe}.py
    validate/{labels,walkforward,cpcv,baselines,ablation,stats,diagnostics}.py
    tune/{search_space,autotune,promotion,registry}.py
    online/{state,replay,monitor}.py
    synth/generate.py

  trend/                     # TREND pillar  (convex trend-life exit)
    types.py engine.py
    config/{defaults,instrument,asset_class,cost,tuning,resolve}.py
    data/{schema,calendar,timing,actions,rolls,quality}.py
    estimate/{volatility,drift,horizon,kalman,sequential,jump,hazard,state}.py
    policy/{convex_exit,reduce,decision}.py
    priors/market_dna.py
    online/{snapshot,replay,monitor}.py
    validate/{labels,survival,baselines,walkforward,cpcv,ablation,stats,reports}.py
    tune/{search_space,autotune,promotion,registry}.py
    research/diagnostics.py
    synth/generate.py

  leverage/                  # LEVERAGE pillar  (min() of seven ceilings)
    types.py engine_api.py
    config/{defaults,asset_class,instrument,margin,resolve}.py
    vol/forecast.py
    ceilings/{stop,vol_target,kelly,tail,portfolio,margin,liquidity}.py
    combine/{kelly_fraction,governor,lot_sizer,engine}.py
    allocate/{risk_budget,clusters}.py
    priors/market_dna.py
    validate/{labels,stress,baselines,ablation,walkforward,stats,reports}.py
    synth/generate.py

  execution/                 # EXECUTION pillar  (tactic selection + protective stop)
    types.py engine_api.py
    config/{defaults,asset_class,instrument,venue,resolve}.py
    microstate/{spread,vol,volume_proxy,fill_prob}.py
    cost/{spread_cost,impact,adverse_selection,delay_cost,tactic_cost}.py
    policy/{tactic_select,chase,partial,order_issuer,protective_stop,engine}.py
    nautilus/{adapter,reconcile}.py          # the ONLY nautilus-importing execution code
    priors/exec_dna.py
    validate/{labels,shortfall,calibration,baselines,ablation,walkforward,stats,reports}.py
    synth/generate.py

  cost/                      # COST pillar  (net-of-cost truth; supersedes the per-pillar stubs)
    types.py cost_api.py
    config/{defaults,fee_schedule,asset_class,instrument,resolve}.py
    facts/{nautilus_adapter,execution_adapter}.py   # the ONLY nautilus-touching cost code
    components/{fees,spread,impact,funding_accrual,funding_forecast,carry}.py
    assemble/cost_book.py
    consume/{net_ev_gate,hold_carry,carry_term,tca}.py
    validate/{labels,walkforward,cpcv,baselines,ablation,stats,reports}.py
    synth/generate.py

  system/                    # SYSTEM pillar  (orchestrator + integrated validation)
    types.py system_api.py
    config/{defaults,universe,tuning,resolve}.py
    modules/{selection,breakout_adapter,leverage_adapter,trend_adapter}.py  # thin pillar adapters
    compose/{cadence,incumbent,budget,iterate,precedence,orchestrator}.py
    nautilus/{adapter,exec_stub}.py
    validate/{labels,walkforward,cpcv,deflated,robustness,attribution,
              baselines,trials,promotion,budget,cache,reports}.py
    synth/generate.py

  ops/                       # OPS pillar  (SPC monitoring + kill switch)
    types.py ops_api.py
    config/{defaults,in_control,asset_class,severity,resolve}.py
    detectors/{base,cusum,page_hinkley,ewma,shewhart,sprt}.py
    streams/{whiten,fills,cost_tca,vol,edge,recon,heartbeat}.py
    reconcile/streams_map.py
    brake/{triggers,alarms,safe_state,kill_switch,engine}.py
    nautilus/{adapter,flatten}.py
    forensics/{log,replay}.py
    validate/{derive_limits,arl,oos,baselines,reports}.py
    synth/generate.py

  regime/                    # REGIME pillar  (global gross multiplier g_t)
    types.py engine_api.py
    config/{defaults,asset_class,resolve}.py
    panel/build.py
    measures/{vol_regime,absorption,turbulence,liquidity}.py
    state/{bands,machine,cost_gate,emit}.py
    validate/{threshold_stability,baselines,ablation,backtest}.py
    synth/generate.py

  strategy/                  # THE Nautilus composition layer (existing stubs to be filled)
    convexity_strategy.py    #   the single Strategy; composes all pillars (see §4)
    breakout.py config.py universe.py     # legacy stubs — absorbed/re-exported (see §6)

  costs/binance_fees.py      # legacy stub — ABSORBED into cost/config/fee_schedule.py (see §6)
  adapters/ run/             # existing data/runtime plumbing (unchanged by this work)
```

Tests mirror the tree under `tests/`: `tests/<pillar>/test_<module>.py`, plus
`tests/<pillar>/test_invariants.py` and `tests/<pillar>/test_worked_examples.py` (see §8).

---

## 3. Shared foundation (`convexity.core`) — implemented

The single source of the shapes and governance that cross pillar boundaries. Pure-Python,
numpy-typed, NautilusTrader-free, ruff-clean (matches the existing `selection/` style).

### `core/params.py` — governance
* `Bucket(str, Enum)` = `RED | AMBER | BLUE | TEAL` — the four governance buckets every spec §2/§4
  uses. The bucket decides *where* a value may enter and *who* may set it.
* `ParamSpec(name, bucket, meaning, spec_ref, blue_grid=None)` — one declared parameter, citing its
  owning spec/section; `blue_grid` records the finite candidate set (rejected on a non-BLUE param).
* `ParamRegistry` Protocol + `StaticParamRegistry` — each pillar declares its parameter set;
  `.of_bucket(RED)` for Invariant-C assertions; `.blue_grid_cardinality()` is the per-pillar trial
  count SYSTEM's ledger deflates against (SYSTEM Inv DSR/SG; caps like REGIME `K_regime_max`).
* `assert_no_red_in_amber(amber_field_names, registry)` — the reusable Invariant-C check: raises if
  a RED/BLUE name leaks into a per-instrument `InstrumentSpec`. **Every pillar's Inv-C test calls
  this.**

### `core/types.py` — cross-pillar boundary objects
The types that flow *between* pillars (rich internal objects stay in each `<pillar>/types.py`):

| Type | Producer → consumer | Spec anchor / firewall |
|---|---|---|
| `Side`/`Direction`/`AssetClass`/`SignalStatus`/`Urgency`/`ExecUrgency`/`RegimeStateName`/`KillState` | shared aliases | one definition, not per-pillar |
| `Bars` | data layer → all | int64 epoch-ms; optional micro arrays None on a kline feed |
| `InstrumentSpec` | metadata → all | **AMBER only** (every Inv C); no RED/BLUE field |
| `CostModel` | stub → BREAKOUT/TREND validation | superseded by COST pillar (COST P1) |
| `HoldHorizon` | BREAKOUT/TREND/LEVERAGE → COST | a RANGE; `m_max` via `from_quantile` (ceil) |
| `EntrySignal` | BREAKOUT `combine` → SYSTEM/EXEC | Inv A (only `combine` builds it), Inv K (downstream reads only direction/status/urgency/levels) |
| `Position` | BREAKOUT entry → TREND | Inv O (TREND never recomputes entry) |
| `SignalInput` | SYSTEM/SELECTION → LEVERAGE | Inv A (no edge field, by absence) |
| `LeverageDecision` | LEVERAGE → SYSTEM/EXEC | regime cut surfaces as `binding_ceiling='portfolio'` |
| `ExecutionIntent`/`HoldHorizonEstimate`/`ResidualMoveEstimate` | TREND → EXEC/LEV/COST | Inv N/P |
| `ExecTarget` | SYSTEM (BREAKOUT+LEV+TREND) → EXEC | Inv A (no edge/sizing/exit score) |
| `CostComponent`/`RoundTripCost` | COST → BREAKOUT/LEV/TREND/REGIME | Inv B (additive), Inv F-band-gate |
| `RegimeState` | REGIME → LEVERAGE/SELECTION | g_t∈{1,.75,.5,.25} floored .25; single H_book channel |
| `OpsDirective` | OPS → live loop | Inv PROV (monitors, never trades) |

### `core/protocols.py` — composition seams
One `typing.Protocol` per pillar's single outward method (`BreakoutModule.propose`,
`LeverageEngine.size_position`, `TrendEngine.update`, `ExecutionEngine.on_target`, `CostEngine`'s
four consumers, `RegimeEngine.regime_state`, `OpsEngine.on_obs`, `SystemOrchestrator.decide`).
SYSTEM's orchestrator and the strategy adapter depend on these Protocols, so each pillar can be
built and tested against a mock satisfying its Protocol before any sibling exists. Pillar-private
return types (`ActionDecision`, `ExecDecision`, `SystemDecision`, …) are typed `object` at the seam
with the spec reference in the docstring — the consumer only needs the boundary fields.

---

## 4. The composition graph (who consumes/emits what)

```
                         ┌─────────────────────────── REGIME ──────────────────────────┐
                         │ panel + vol/absorption/turbulence/liquidity → g_t∈{1,.75,.5,.25}
                         └──────────────────────────────┬──────────────────────────────┘
                                                        g_t (single channel)
                                                         ▼
 SELECTION ─risk_weights/rank─▶ SYSTEM.orchestrator ◀──── LEVERAGE  (g_t·H_book scales L_portfolio)
   (Carver Ch.25)                    │  ▲                     ▲   │
                                     │  │ EntrySignal         │   │ LeverageDecision (size)
                     BREAKOUT.propose ┘  └──────── SignalInput ┘   │
                                     │                            ▼
                                     │  Position           (min() of 7 ceilings × governor; lot-quantize)
                                     ▼                            │
                                  TREND.update ──ExecutionIntent──┤
                                     │  ActionDecision            ▼
                                     │  HoldHorizonEstimate    ExecTarget
                                     │  ResidualMoveEstimate      │
                                     ▼                            ▼
                COST  ◀── net-EV gate (BREAKOUT) ───────────  EXECUTION.on_target
                 │     ◀── hold-carry  (TREND)                    │  OrderCommand / venue reduce-only stop
                 │     ◀── carry-term  (LEVERAGE clamps)          ▼
                 │     ◀── round_trip_estimate (REGIME cost gate)  Nautilus engine
                 └── one CostBook, one resolved config (Inv T)
                                     │
   OPS  ◀── streams: fills/cost_tca/vol/edge/recon/heartbeat ── (EXECUTION/COST/LEVERAGE/SYSTEM)
        ──OpsDirective {run,degrade,halt}+flatten── live loop (flatten via EXECUTION venue stops)
```

**The three exposure brakes compose on disjoint axes (LEVERAGE Inv R/BRAKE, REGIME Inv NODOUBLE,
OPS Inv BRAKE):** REGIME `g_t` reads *market-state* and enters only via `g_t·H_book`; the LEVERAGE
drawdown governor reads *equity-drawdown*; OPS owns the *binary* kill state. They multiply on
disjoint inputs — never a double cut (e.g. CRISIS realized gross is `0.25·full`, **not** `0.25²`).

**Validation firewall (every pillar, Inv C/FW/G/V):** nothing under a pillar's live layers
(`detect/`, `features/`, `standardize/`, `policy/`, `estimate/`, `compose/`, `components/`,
`ceilings/`, `measures/`, `streams/`, `brake/`, `online/`, `nautilus/`) imports from that pillar's
`validate/` (or `forensics/`). Future-looking labels live only under `validate/`. This is asserted
by an AST import-graph test per pillar.

---

## 5. Cross-spec interface mismatches found in the blueprints — and their resolution

These are the conflicts each spec's `risks` section raised. Resolved here so the build is unblocked.

1. **Package placement (all eight specs).** Specs write bare `breakout/`, `cost/` (singular), etc.
   *Resolution:* everything under `src/convexity/<pillar>/`, imported `convexity.<pillar>`. COST is
   `convexity.cost` (singular, matching its prose) as a **new** package; the existing
   `convexity.costs` (plural) holds only `binance_fees.py`, which is **absorbed** into
   `cost/config/fee_schedule.py` (see §6). One fee table only (COST P1).

2. **REGIME `g_t` channel: REGIME.md vs LEVERAGE.md.** REGIME's intro says g_t is "one more min()
   term *and* a scalar on H_book"; LEVERAGE.md (P7/Inv R, the authoritative build spec for sizing)
   says g_t enters **only** through `g_t·H_book`, never a separate min() term. *Resolution:* build
   to LEVERAGE.md — single channel through `L_portfolio`. REGIME's own §3.4 agrees. Encoded in
   `core/types.RegimeState` docstring + `LeverageDecision.binding_ceiling=='portfolio'`.

3. **`ExecTarget` is the load-bearing cross-pillar contract** and was not defined in any sibling
   spec. *Resolution:* defined in `core/types.ExecTarget` with exactly the 8 fields EXECUTION
   prescribes and **no edge field** (Inv A). SYSTEM assembles it from BREAKOUT (side, invalidation),
   LEVERAGE (target_qty), TREND (reduce/exit + invalidation moves + urgency). Owner of `urgency`:
   the signal pillar (BREAKOUT for entries → `'URGENT'`; TREND for exits → `'IMMEDIATE'` on hard
   exit). Owner of `invalidation_generation`: TREND.

4. **`HoldHorizon` producer/`m_max` redundancy (COST risk).** COST stores `m_max` *and* says it
   equals `ceil(max_hold_quantile_ms/funding_interval_ms)`. *Resolution:* `HoldHorizon.from_quantile`
   classmethod computes `m_max` (ceil) so the field can never disagree. `expected_m_caller` is
   diagnostic-only and cannot relax the gate (COST Inv M-hold).

5. **COST carry sign / LEVERAGE clamp location.** COST emits a **signed** carry; LEVERAGE's
   `StopCeiling` applies `L_stop=(c-max(carry,0))/d`. *Resolution:* `CarryTerm.over_horizon` returns
   signed (credit negative); the `max(carry,0)` clamp lives in `leverage/ceilings/stop.py`, NOT in
   COST. COST's Inv T test checks only the signed value is bit-identical across the three consumers;
   the clamp is a LEVERAGE test (Inv CARRY).

6. **`HoldHorizonEstimate` → COST `HoldHorizon` field mapping (TREND risk).** Who computes
   `m_max`? *Resolution:* COST computes it via `HoldHorizon.from_quantile(..., funding_interval_ms)`
   from TREND's `max_hold_quantile_ms`; TREND never computes funding-stamp counts.

7. **CLAUDE.md "Lee-Mykland forbidden" vs TREND/BREAKOUT jump detectors.** *Resolution:* the jump
   detectors (`trend/estimate/jump.py`, `breakout/features/jumps.py`) **default to the robust
   standardized-return exceedance branch** (the CLAUDE.md-compatible path) for crypto; the
   bipower/Lee-Mykland branch is config-gated off by default and documented as the CLAUDE.md
   exception (the forbid was scoped to the *entry signal*, not these exit/feature diagnostics).

8. **`scipy` dependency (TREND/COST/SYSTEM/OPS risk).** KM/Greenwood/Weibull-MLE, PSR/DSR (`norm`),
   CUSUM-ARL MC, Ledoit-Wolf all want scipy. CLAUDE.md discourages scipy in the live engine;
   `.venv-test` *has* scipy 1.18. *Resolution:* scipy is allowed in **validation/research** code and
   in pure-math measure code (it is numpy-adjacent, not a live-engine dep). Add it to a
   `validation`/`research` dependency group in `pyproject`, kept out of the NautilusTrader live
   runtime closure. The pillar policy hot path stays numpy-only where feasible.

9. **`BookState` undefined in LEVERAGE (owned by SYSTEM).** *Resolution:* `system/types.BookState`
   is the canonical definition (equity, peak_equity, free_margin, positions, cov, governor_mult);
   LEVERAGE consumes it via the `LeverageEngine` Protocol (typed `object` at the seam until SYSTEM
   lands; LEVERAGE built/tested against a `BookState` fixture). Same for `SplitPlan`/`PositionSampler`
   (validation support types, defined in each pillar's `validate/` or a shared validation core).

10. **SYSTEM SelectionAdapter contract mismatch.** SYSTEM assumes `select(...)→ProposedPortfolio`;
    the existing `selection/dynamic_optimization.py` exposes `optimise_portfolio→DynOptResult`.
    *Resolution:* `system/modules/selection.py` **maps** `DynOptResult→ProposedPortfolio`
    (keep/eligible from non-zero positions, `risk_weights` from optimised weights). The `rank` field
    (undefined source) is derived as a **RISK measure** — tracking-error-contribution /
    correlation-distance — never edge (Inv A). Selection's 19 tests are untouched.

11. **Nautilus version (1.229) vs spec prose ("generic APIs / 3.11+").** *Resolution:* adapters
    target nautilus 1.229 / py312 exactly; spec API names are illustrative. `FundingRateUpdate`
    interval hardcoded 8h for Binance per CLAUDE.md.

12. **OPS DD-kill (45%) vs LEVERAGE governor flat (43%) shared reference.** *Resolution:* both read
    the SAME equity series + running-max peak (the LEVERAGE-owned reference); `45% > 43%` holds by
    construction (OPS Inv BRAKE §7.1a). The shared reference contract is in `system/types.BookState`.

13. **OPS `§17 rollout` has no spec-prescribed filename.** *Resolution:* `ops/brake/rollout.py`
    (consumes the same §5 streams as the promotion gates; sits beside `engine.py`). Added to the
    build plan, not the initial scaffold (logic-bearing).

---

## 6. Legacy stub reconciliation

* `strategy/breakout.py` → its logic moves to the `breakout/` package; the file becomes a thin
  re-export of `convexity.breakout.detect.combine.BreakoutModule` (or is deleted once
  `convexity_strategy.py` imports the package directly).
* `strategy/config.py`, `strategy/universe.py` → folded into per-pillar `config/` +
  `system/config/universe.py`.
* `strategy/convexity_strategy.py` → the live composer (§4); filled in the SYSTEM/integration phase.
* `costs/binance_fees.py` → absorbed into `cost/config/fee_schedule.py` (AMBER VIP tiers) feeding
  `cost/components/fees.py`. One cost truth (COST P1). The plural `costs/` package is retired.

---

## 7. Governance discipline (enforced, not aspirational)

Every pillar implements `resolve_config(defaults=...)` as the **single merge point** where RED
enters; per-class overrides come through a separate `AssetClassPreset` argument; per-instrument
`InstrumentSpec` carries AMBER only. Each pillar declares its parameters as a
`StaticParamRegistry` and its Inv-C test calls `assert_no_red_in_amber(InstrumentSpec field names,
registry)`. BLUE grid cardinalities are summed into SYSTEM's persistent trial ledger so the
deflated-Sharpe gate counts every searched config (SYSTEM Inv SG/DSR; REGIME `K_regime_max`,
SYSTEM `K_max` caps). TEAL fits are causal/fold-local, never hand knobs.

---

## 8. Test strategy

Four layers, mirroring the spec invariants + worked examples. All run under `PYTHONPATH=src
.venv-test/bin/python -m pytest` with **no nautilus**.

1. **Invariant tests** (`tests/<pillar>/test_invariants.py`). Each named Invariant in a spec becomes
   one test. Recurring shapes:
   * *Firewall/governance (AST):* import-graph scan that live layers never import `validate/`;
     `assert_no_red_in_amber` for Inv C; "only `combine` constructs `EntrySignal`" (Inv A);
     "only `decision` constructs `ActionDecision`".
   * *Causality / no-lookahead:* perturb every bar after `t`, assert output at `≤t` byte-identical
     (BREAKOUT Inv B, TREND Inv H, REGIME Inv CAUSAL, OPS Inv CAUSAL, LEVERAGE Inv CAUSAL-L,
     EXECUTION Inv L).
   * *Batch/live parity + snapshot restore:* `run_batch` == sequential `update`, mid-stream snapshot
     restore reproduces signals (BREAKOUT Inv E, TREND Inv I, EXECUTION Inv E, SYSTEM Inv PAR).
   * *Score/edge firewalls:* scale all scores by `c>0` → downstream decisions byte-identical
     (BREAKOUT Inv K, SYSTEM Inv CB); vary a hypothetical edge field → sizes/tactics unchanged
     (LEVERAGE Inv A, EXECUTION Inv A).
   * *min()/precedence, never a product:* unique-minimum binding tests (LEVERAGE Inv B, REGIME Inv
     MIN, OPS Inv KILL).
   * *Diagnostics are log-only:* randomize diagnostics → decision unchanged (everywhere: Inv H/F).

2. **Worked-example regression fixtures** (`tests/<pillar>/test_worked_examples.py`). Each spec's
   numbered worked example is reproduced **bit-for-bit** (or to the spec's stated tolerance) from
   its tabulated inputs. Load-bearing examples:
   * BREAKOUT: strict-pre-bar barrier identity; inclusive-max-never-fires; same-k 1-tick-vs-kσ;
     entry-timestamp `entry_idx=signal_idx+1`; score-scale invariance; future-needle; cross-sectional
     FDR on a noise universe; `convexity_ratio` formula.
   * TREND: `z_drift` = `log(1.1)/0.1 = 0.9531018`; trail multiplier schedule at 0.4/1.0/3.0/5.0 R;
     `E[Π]` net-convexity PASS/MARGINAL/FAIL (+0.3725 / +0.08 / **−0.13** R₀); KM censoring bias;
     Weibull tail quantile; action timing next-bar.
   * LEVERAGE: `L_stop` carry-sign clamp; `d(2T)/d(T)=√2`; `L_kelly` notional-invariance; governor
     ladder + ramp (`dd=0.42 → 0.0833`); MCTR sum identity; `g_t=1.0` no-overlay equivalence;
     `g_t=0.25` regime cut at full equity; quantization round-down + divergence `0.189189`.
   * EXECUTION: PI at touch / one-tick; entry vs exit maker cost split; `worst_entry` long/short;
     funding half-open window; sqrt impact → 0; `p_fill` LCB gate; `STOP_OK` predicate;
     disjoint slice/stop id schemes.
   * COST: the **§5.7 centerpiece** — every line including `C_impact=0.0282843` (NOT 2.83e-6, Inv
     U-dim), `F_band_conservative=0.69798`, `total_BAND=1.9545485`, `passes=False`, `margin=−0.1045485`;
     EWMA `α=1−2^(−1/9)=0.074125`; carry time-scaling; financing-once.
   * SYSTEM: the **DSR §7.3b FAIL** fixture (DSR=0.6950 @ n_obs=108 < MinTRL≈1114) and the **PASS**
     contrast (DSR≈1.0 @ n_obs=400 ≥ MinTRL≈56); `N_sys=1` BLdP convention; anti-correlated
     ledger → `N_sys=K`; double-consumed-HALT single flatten; DD backstop above governor; **Carver
     [0,1,3]** preserved (the existing 19 selection tests).
   * OPS: the **§10 CUSUM cost-drift** fixture — full `C⁺` recursion, DEGRADE@15 / HALT@18; SPRT
     boundaries `(+2.890,−2.251)` + ASN `(99,118)`; two-sided ARL `743→370`, `h≈4.77` (NOT ≈3.5);
     AR(1)-Student-t realized ARL within CI of 370 only after prewhitening.
   * REGIME: **§6.1** corr-spike `AR_t=0.750, δ_AR=+2.20, g=0.25`, gross `750→250` (NOT 62.50);
     **§6.2** corr→vol handoff dwell continues (CLOCK); **§6.3** drifted-gross cost-gate defers.

3. **Property tests.** Monotonicities the specs assert (e.g. EXECUTION Inv J: p_fill ↓, A ↑, D ↑ in
   g; LEVERAGE Inv U: rounding only reduces risk; OPS Inv ALARM: ARL₀ ↑ with D_min; REGIME Inv
   DSR-MONO: N_sys ↑ as ρ_cl tightens). Implemented with explicit sweeps + assertions (no hypothesis
   dependency required).

4. **Integration tests** (VM-only, nautilus installed). The `strategy/` + `*/nautilus/` adapters:
   batch↔live parity through the real engine, restart reconciliation, stop placement. Marked
   `@pytest.mark.nautilus` and skipped in the pillar venv.

Each pillar also ships `synth/generate.py` so its tests need **no external data** — every invariant
and worked-example fixture is driven by a seeded synthetic generator.
