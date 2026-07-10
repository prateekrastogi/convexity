"""Fleet configuration — every owner-tunable value in ONE place (PAPER.md §12.1).

Values here are OWNER DECISIONS, changeable without code edits elsewhere. Provenance is noted per
value; nothing in this module is derived at runtime.
"""

from __future__ import annotations

# --- Cross-book capital allocation (per 1000 quantized units; owner decision, D1v2) -----------
ALLOC_UNITS: dict[str, int] = {"crypto": 400, "commodity": 300, "equity": 300}
ALLOC_TOTAL = 1000


def alloc_fraction(book: str) -> float:
    return ALLOC_UNITS[book] / ALLOC_TOTAL


# --- Equity book: the D1-validated drawdown-controlled half-Kelly policy ----------------------
# FROZEN half-Kelly constant, derived once from the validated construction's own empirical
# fat-tailed return distribution (D1 program; reference derivation pinned at
# tests/fixtures/equity_book/reference_leverage_policy.py, re-asserted by the parity test —
# deliberately NOT re-derived at boot, which would drift with new data).
EQUITY_L_HALF = 2.325628533609451
# GZ drawdown-brake ceiling (owner's drawdown budget; the D1 waterfall's Step-A policy).
EQUITY_BRAKE_CEILING = 0.50
# Point-in-time signal warmup: all three ensemble rules computable (~252d rule + buffer).
EQUITY_WARMUP_DAYS = 260
# Leveraged/inverse/decay ETF exclusions (validated finding: structural daily-rebalancing decay
# contaminates trend/beta signals — E3/CTA programs; UVXY alone was 20% of an unguarded book's
# gross P&L).
EQUITY_ETF_EXCLUSIONS: frozenset[str] = frozenset({
    "UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT",
})
# Split/corporate-action watch: |1-day move| above this flags a name for human review and
# excludes it from rebalances until cleared (the AAPL-2020 unadjusted-split failure mode).
EQUITY_SPLIT_WATCH_THRESHOLD = 0.25
EQUITY_FLAGS_PATH = "data/universe/equity_flags.json"

# --- Integer exchange leverage dial (per-symbol, set ONCE at book boot; never per-trade) -------
# Binance USDT-M leverage is an integer per symbol. The dial is a CAP + margin divisor; effective
# book leverage is fractional via notional sizing. equity: 3 covers the 2.33x nominal ceiling and
# caps accidental over-leverage at 3x NAV. crypto/commodity: not set by this program (venue
# default stands; observed dial journaled at equity boot for fleet visibility — normalizing them
# is an owner cutover option, not an autonomous change).
LEVERAGE_DIAL: dict[str, int] = {"equity": 3}

# --- Cross-margin software guardrails (PAPER.md §12.2) --------------------------
# Account maintenance-margin-ratio thresholds. Justification: equity book worst historical day
# −10.4% at ~1.9x ≈ −20% book NAV ≈ −6% of account at 300/1000; DERISK at 60% still leaves >3x
# that shock before liquidation territory.
MARGIN_ALERT_RATIO = 0.40
MARGIN_DERISK_RATIO = 0.60
# De-risk priority (levered, directional book flattens FIRST; market-neutral books last).
DERISK_PRIORITY: tuple[str, ...] = ("equity", "commodity", "crypto")
# Liquidation-distance assertion: distance must exceed this multiple of the worst historical
# one-day move (12.1%, the 93-name benchmark's 2020-03-16).
LIQ_DISTANCE_MIN_MULTIPLE = 3.0
WORST_HISTORICAL_DAY = 0.121

# --- Account lock (PAPER.md §12.2) ----------------------------------------------
# lock path is venue-resolved (venue.state_root()/account.lock) — see lock.py, PAPER.md §12.6
ACCOUNT_LOCK_TIMEOUT_S = 300.0

# --- Shared API pacing (PAPER.md §12.2) ------------------------------------------
AUTHED_CALL_MIN_INTERVAL_S = 0.12
PUBLIC_CALL_MIN_INTERVAL_S = 0.06
