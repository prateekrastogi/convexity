# PREREG ADDENDUM 1 — session-anchor reproduction diagnostic (frozen 2026-07-04, before running)

**Motivation (owner challenge):** the intraday README's crude result was POSITIVE on futures
(continuous Sharpe 1.01 / gappy 0.83, n≈35 over ~16 months, robust to dropping its own top trades)
while the UTC-day baseline here shows CLUSDT negative. Four construction differences could explain
it: session anchor (MCX 09:00–23:54 IST = 03:30→18:24 UTC, signal at the US-settlement close, vs
our 23:59 UTC dead-overnight close), fire rate (~12/yr vs ~58/yr), period (2023-10→2025-02 vs
2021-06→2026-06), and stop (matched fixed 0.75% vs 0.5×ATR20). This addendum pre-registers the
diagnostic family that separates those causes. **Label: REPRODUCTION DIAGNOSTIC** — results are
reported and counted in the study's N_eff, but adoption still requires the §6 gates of the main
prereg; running MANY anchors and keeping the best would otherwise be textbook snooping, so the
family below is frozen NOW and run on ALL 7 symbols (no crude-only cherry-pick).

## Declared family

Session definitions (bars within [start,end) UTC of each calendar day; OR = first X min of the
session; signal = verbatim strong-close rule evaluated on SESSION high/low/close at session end;
entry at session-end close; stop checked on ALL 24/7 1-min bars during the hold (perp reality);
hold = 5 valid session-days; non-overlap):

| Anchor | Session (UTC) | OR minutes (~8%) | Rationale |
|---|---|---|---|
| MCX-faithful | 03:30–18:24 | 70 | 1:1 reproduction of the original crude construction |
| London | 08:00–16:30 | 40 | European cash window |
| NY/CME | 13:30–19:30 | 30 | US pit/settlement window |
| UTC full day | 00:00–24:00 | 120 / 30 | already run (the main-prereg baseline) |

Stops: {fixed 0.75% (the README matched stop), 0.5×ATR20 (session-based ATR)}. Costs: 12bp RT
primary (6/24 reported). Symbols: all 7. **Total new cells: 3 sessions × 2 stops × 7 symbols = 42**
(+ the already-run UTC cells), all counted in N_eff.

## Declared diagnostics (reported, not selected on)

- Per-symbol per-cell: n, win%, EV bp, per-trade & annualized Sharpe, maxDD, drop-top-3 EV.
- **CL period split**: full sample vs 2023-10-01→2025-02-28 (the original window) vs the rest —
  tests the "period luck" explanation on BOTH the UTC baseline and the MCX-faithful cell.
- Per-year Sharpe for the MCX-faithful CL cell (regime stability).

## Interpretation rules (frozen)

- If MCX-faithful CL is positive with EV/Sharpe of the original's order (and the other symbols'
  behavior is reported alongside): the original crude effect is SESSION-STRUCTURE-DEPENDENT —
  localized to measuring the breakout close at the US settlement; it carries to perps ONLY with
  that session definition.
- If MCX-faithful CL is ~zero/negative on the full sample but positive on 2023-10→2025-02: the
  original result is best explained as PERIOD-SPECIFIC (its own record already flags n=34/16mo as
  anecdotal).
- If nothing reproduces: the original crude positive does not transfer to USD perps at all
  (MCX/INR-specific or luck); record the negative.
- Any positive cell here remains a CANDIDATE only — §6 gates (CPCV OOS, DSR at updated N_eff, PBO,
  drop-top-3, paired CI) still decide adoption.
