"""Invariant CS-REJECT — the Corwin-Schultz dead end, kept as a standing guard (COST.md 5.4.3).

WHAT WAS TRIED AND REJECTED. The Corwin & Schultz (2012) high-low spread estimator was applied to
1-minute-aggregated daily OHLC as an impact proxy. Its separation identity assumes the high-low
range is sampled at a frequency where true volatility scales with sqrt(dt) while the bid-ask
bounce does not; daily highs/lows built from 1-minute bars violate that scaling, so realized
intrabar volatility is misattributed to spread. The failure mode is not "noisy" — it is
systematically, wrongly LARGE: a ~135bp per-round-trip estimate implying a ~40%/yr cost drag, an
order of magnitude above the measured ~13bp round-trip anchor (COST.md Section 5.5) and larger
than the strategy's entire gross return — a cost model that would have falsely killed a real edge.

WHAT THIS MODULE ENFORCES, TWO LAYERS.

  (1) RE-RUN THE ESTIMATOR (the literal Invariant CS-REJECT clause). The spec commits this module to
      "re-run the Corwin-Schultz estimator on the 1-minute-aggregated daily OHLC panel [and]
      reproduce the documented ~135bp/~40%-drag garbage result." ``corwin_schultz_spread_bp`` is a
      RUNNABLE, closed-form implementation of the CS (2012) two-day high-low estimator;
      ``aggregate_minute_to_daily_ohlc`` builds the daily high/low from a 1-minute panel exactly as
      the rejected experiment did (daily high = max of the day's 1-min highs, daily low = min of its
      1-min lows). ``corwin_schultz_round_trip_bp`` runs the estimator over that aggregated panel and
      returns the per-round-trip bps it produces. The standing regression test feeds it a
      1-minute-aggregated daily OHLC panel whose intraday volatility swamps the (near-zero) true
      spread — the exact conflation the spec describes — and asserts the estimator reproduces an
      order-of-magnitude-too-large number (>= 10x the 13bp anchor). This is the load-bearing
      "reproduce the result" guard: it fails loudly if a future refactor of the CS math ever stops
      reproducing the garbage (e.g. someone "fixes" the estimator and quietly resurrects it as a
      viable spread source), which the constant-only assertion of the prior version could not catch.

      This estimator is DELIBERATELY confined to this ``validate/`` reject module and is never
      imported by any cost-path module — it exists only so the negative result is reproducible, not
      so it can be wired into a live cost. Layer (2) enforces that confinement structurally.

  (2) STRUCTURAL ANTI-REINTRODUCTION SCAN (a stronger complement, not a substitute). Corwin-Schultz
      must never be reintroduced as a spread/impact estimator elsewhere in the cost path. The guard
      ``corwin_schultz_offenders`` is an AST scan: any code-level identifier or import matching the
      estimator's name inside the cost-path source roots fails loudly and points back at COST.md
      Section 5.4.3. This module is exempt by filename (it is the one place the dead end is named and
      reproduced); imports of THIS module's own public API are exempt too (a promotion runner or
      baselines consumer legitimately importing the standing test is not a reintroduction). Together
      the two layers satisfy both the literal "reproduce the result" clause AND the doctrine of
      keeping the dangerous estimator out of every module that could ever feed a live rebalance.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterator

import numpy as np

# Documented negative result (COST.md Section 5.4.3) — the garbage estimate CS produced on
# 1-minute-aggregated daily OHLC, vs the measured empirical anchor (COST.md Section 5.5).
CS_REJECTED_ROUND_TRIP_BP: float = 135.0  # ~135bp per round trip (wrong, volatility-conflated)
CS_REJECTED_DRAG_PCT_PER_YEAR: float = 40.0  # ~40%/yr implied cost drag (wrong)
MEASURED_ROUND_TRIP_BP: float = 13.0  # 2*(5 + 1 + 0.5)bp, src/convexity/cost/accounting.py defaults

# COST.md Section 5.4.3 bounds the reproduction to the same ORDER OF MAGNITUDE as the documented
# garbage: the rejected estimate is >= 10x the measured 13bp round-trip anchor. The runnable-estimator
# regression test asserts the reproduced number clears this bound (that is the "reproduce the result"
# check), never that it hits 135bp to the bp — the spec itself only claims order-of-magnitude ("~135bp").
CS_REJECT_ORDER_OF_MAGNITUDE: float = 10.0

# Source roots forming the cost path (repo-relative): the cross-sectional engine + cost pillar are
# now fully promoted under src/convexity (research/xsec no longer exists on disk; the dead root is
# kept in the tuple as a harmless no-op guard — is_dir() skips it below — in case of regression).
# Tests are scanned too — a test EXERCISING Corwin-Schultz would itself be a reintro.
COST_PATH_ROOTS: tuple[str, ...] = ("research/xsec", "src/convexity")

_FORBIDDEN_TOKENS: tuple[str, ...] = ("corwin", "schultz")
_EXEMPT_FILENAME: str = "corwin_schultz_reject.py"  # this guard module names the dead end itself
_GUARD_MODULE_STEM: str = "corwin_schultz_reject"  # importing FROM the guard is legitimate, not a reintro


# =================================================================================================
# LAYER 1 — the runnable Corwin-Schultz estimator (COST.md Section 5.4.3 "re-run ... reproduce").
# Confined to this reject module; never imported by a cost-path module (enforced by Layer 2).
# =================================================================================================

# Corwin & Schultz (2012), Eq. (18): the constant 3 - 2*sqrt(2) appearing in the alpha solution.
_CS_DENOM: float = 3.0 - 2.0 * math.sqrt(2.0)


def aggregate_minute_to_daily_ohlc(
    day_index: np.ndarray, high: np.ndarray, low: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build daily (high, low) from 1-minute bars — exactly the rejected experiment's input.

    ``day_index`` labels each 1-minute bar with its UTC calendar day (ascending, contiguous days may
    repeat); ``high`` / ``low`` are the per-minute bar high/low. The daily high is the max of the
    day's 1-minute highs and the daily low is the min of its 1-minute lows (COST.md Section 5.4.3:
    "daily high/low built by aggregating 1-minute bars"). Returns ``(daily_high, daily_low)`` ordered
    by day. This aggregation is the step that VIOLATES Corwin-Schultz's sampling-frequency assumption:
    a daily range assembled from 1-minute bars is dominated by realized intraday volatility, which
    the estimator then misattributes to the bid-ask spread.
    """
    di = np.asarray(day_index)
    hi = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    if not (di.shape == hi.shape == lo.shape):
        raise ValueError(f"day_index {di.shape}, high {hi.shape}, low {lo.shape} must be aligned")
    days = np.unique(di)
    daily_high = np.empty(days.shape[0], dtype=float)
    daily_low = np.empty(days.shape[0], dtype=float)
    for j, d in enumerate(days):
        m = di == d
        daily_high[j] = float(np.max(hi[m]))
        daily_low[j] = float(np.min(lo[m]))
    return daily_high, daily_low


def corwin_schultz_spread_bp(daily_high: np.ndarray, daily_low: np.ndarray) -> float:
    """Corwin & Schultz (2012) high-low spread estimator, in bps (RUNNABLE, COST.md Section 5.4.3).

    Closed-form, no fit (P7). For each pair of consecutive days ``t, t+1``:

        beta_t  = (ln(H_t / L_t))^2 + (ln(H_{t+1} / L_{t+1}))^2      # sum of two single-day sq log-ranges
        gamma_t = (ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1})))^2        # squared two-day sq log-range
        alpha_t = (sqrt(2*beta_t) - sqrt(beta_t)) / (3 - 2*sqrt(2))
                  - sqrt(gamma_t / (3 - 2*sqrt(2)))                  # CS (2012) Eq. (18)
        S_t     = 2 * (exp(alpha_t) - 1) / (1 + exp(alpha_t))        # implied proportional spread

    Returns ``1e4 * mean(max(S_t, 0))`` — the average estimated proportional spread in bps, negatives
    clamped to 0 per Corwin & Schultz's own convention (a negative single-pair estimate is treated as
    a zero-spread observation, not a negative spread). This is deliberately the ORIGINAL estimator,
    reproduced faithfully so that applying it to a 1-minute-aggregated daily range reproduces the
    documented garbage; it is NOT a corrected or usable spread source.
    """
    h = np.asarray(daily_high, dtype=float)
    lo = np.asarray(daily_low, dtype=float)
    if h.shape != lo.shape:
        raise ValueError(f"daily_high {h.shape} and daily_low {lo.shape} must be aligned")
    if h.shape[0] < 2:
        return float("nan")
    spreads: list[float] = []
    for t in range(h.shape[0] - 1):
        h0, l0, h1, l1 = h[t], lo[t], h[t + 1], lo[t + 1]
        if not (h0 > 0 and l0 > 0 and h1 > 0 and l1 > 0 and h0 >= l0 and h1 >= l1):
            continue
        beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
        gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS_DENOM - math.sqrt(gamma / _CS_DENOM)
        e = math.exp(alpha)
        s = 2.0 * (e - 1.0) / (1.0 + e)
        spreads.append(max(s, 0.0))
    if not spreads:
        return float("nan")
    return 1e4 * float(np.mean(spreads))


def corwin_schultz_round_trip_bp(
    day_index: np.ndarray, high: np.ndarray, low: np.ndarray, *, taker_bp: float = 5.0
) -> float:
    """Re-run CS on 1-minute-aggregated daily OHLC and return the implied round-trip bps (Section 5.4.3).

    Aggregates the 1-minute (``high``, ``low``) to daily OHLC (``aggregate_minute_to_daily_ohlc``),
    runs ``corwin_schultz_spread_bp`` on it, and assembles a per-round-trip cost the SAME way the
    accounting layer does: ``2 * (taker + half_spread)`` with the CS-estimated proportional spread
    treated (charitably to CS) as the full quoted spread crossed once per leg, i.e. half-spread per
    leg = S/2. Even so, on a volatility-dominated daily range the CS spread ``S`` is inflated an order
    of magnitude, and the assembled round-trip lands ~10x the 13bp anchor — reproducing the documented
    ~135bp/~40%-drag dead end (COST.md Section 5.4.3). This is the number the standing regression test
    bounds; it is a reproduction of a REJECTED estimate, never a live cost input.
    """
    dh, dl = aggregate_minute_to_daily_ohlc(day_index, high, low)
    s_bp = corwin_schultz_spread_bp(dh, dl)
    if not math.isfinite(s_bp):
        return float("nan")
    # 2 legs * (taker + half of the estimated quoted spread) — the accounting.round_trip_friction shape.
    return 2.0 * (taker_bp + s_bp / 2.0)


# =================================================================================================
# LAYER 2 — structural anti-reintroduction AST scan (keeps the estimator out of the cost path).
# =================================================================================================


def _imports_the_guard_module(node: ast.Import | ast.ImportFrom) -> bool:
    """True iff ``node`` is an import statement targeting the guard module itself.

    Importing the guard's OWN public API — e.g. ``from convexity.cost.friction.validate.
    corwin_schultz_reject import corwin_schultz_offenders`` — is a legitimate use of the standing
    regression test (baselines/stats/a promotion runner may consume it), NOT a reintroduction of the
    Corwin-Schultz estimator. Its module path (``...corwin_schultz_reject``) and its imported names
    (``corwin_schultz_offenders``, ...) all contain the forbidden tokens, so the guard must exempt the
    import as a whole rather than fire a false positive on it (review gap 4). ``import ....
    corwin_schultz_reject`` (importing the module by dotted path) is exempted the same way.
    """
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.split(".")[-1] == _GUARD_MODULE_STEM:
            return True
        return any(alias.name == _GUARD_MODULE_STEM for alias in node.names)
    return any(alias.name.split(".")[-1] == _GUARD_MODULE_STEM for alias in node.names)


def _identifiers(tree: ast.AST) -> Iterator[str]:
    """Every code-level identifier in ``tree``: names, attributes, defs, args, import targets.

    Import/ImportFrom nodes are handled explicitly (their module path + imported aliases), so an
    import that TARGETS the guard module itself is skipped wholesale — importing the standing
    regression test's own API is not a Corwin-Schultz reintroduction (``_imports_the_guard_module``,
    review gap 4). Non-import ``ast.alias`` nodes never occur, so aliases are only ever reached via
    their parent import node here (no double-yield, no orphan-alias false positive).
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _imports_the_guard_module(node):
                continue  # legitimate import of the guard's own API — exempt as a whole
            if isinstance(node, ast.ImportFrom) and node.module:
                yield node.module
            for alias in node.names:
                yield alias.name
                if alias.asname:
                    yield alias.asname
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg


def _guard_api_local_names(tree: ast.AST) -> set[str]:
    """Local names bound by importing FROM the guard module (review gap 4).

    ``from ....corwin_schultz_reject import corwin_schultz_offenders as g`` binds the local name
    ``g`` (or ``corwin_schultz_offenders`` if un-aliased); downstream USES of that name (``g(root)``,
    ``corwin_schultz_offenders(root)``) are legitimate references to the standing regression test's
    own API, not a reintroduction of the estimator. These bound names are exempt from the forbidden-
    token scan wherever they appear in the file, not merely on the import line itself — otherwise a
    legitimate consumer of the guard's API would trip the guard the moment it CALLED that API.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and _imports_the_guard_module(node):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def corwin_schultz_offenders(
    repo_root: str | Path, *, roots: tuple[str, ...] = COST_PATH_ROOTS
) -> list[str]:
    """AST-scan the cost-path source roots for Corwin-Schultz identifiers/imports.

    Returns ``"relative/path.py::identifier"`` entries for every code-level identifier containing
    a forbidden token (case-insensitive), sorted. Empty list == Invariant CS-REJECT holds.
    Docstring/comment mentions never match (the scan walks the AST, not the raw text); the guard
    module itself is exempt by filename. Imports of the guard module's OWN api (and downstream uses
    of the names they bind) are exempt too (``_guard_api_local_names``, review gap 4) — importing the
    standing regression test is not a reintroduction of the estimator it forbids. A file that fails
    to parse is reported as an offender (``::<unparseable>``) rather than silently skipped — the
    guard must fail loud, never open.
    """
    base = Path(repo_root)
    offenders: list[str] = []
    for root in roots:
        root_dir = base / root
        if not root_dir.is_dir():
            continue
        for py in sorted(root_dir.rglob("*.py")):
            if py.name == _EXEMPT_FILENAME:
                continue
            rel = py.relative_to(base).as_posix()
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                offenders.append(f"{rel}::<unparseable>")
                continue
            exempt = _guard_api_local_names(tree)
            hits = {
                ident
                for ident in _identifiers(tree)
                if ident not in exempt
                and any(tok in ident.lower() for tok in _FORBIDDEN_TOKENS)
            }
            offenders.extend(f"{rel}::{ident}" for ident in sorted(hits))
    return sorted(offenders)
