"""RED shared, class-level UNIVERSE constants (UNIVERSE.md §8 parameter governance).

Every constant here is RED (shared, class-level, never per-instrument — Invariant C, §8). None is
TEAL: UNIVERSE fits nothing per-instrument at runtime; every input is either observed AMBER metadata
or one of these shared constants (§8, "TEAL — none"). The frozen ``UniverseConfig`` groups them so a
BLUE grid promotion (§8: ``N ∈ {50,60,75,100}``, ``W ∈ {20,30,45}``, the hysteresis band width) is a
new frozen instance, not a mutation, and so ``w_warmup`` cannot drift away from ``W`` (§1.4: it is
DEFINED to equal ``W``, never an independent free parameter — a change to ``W`` propagates to
``tau_history`` automatically without a second governance decision).

The one RED constant with NO assigned numeric value is the short-eligibility strictness of §4.2
(``short_liquid_i(t)``'s threshold): this repo has no short-borrow-availability data, so assigning a
number would be the exact fabrication this file's evidence-grounding standard forbids (§4.2 disclosed
limitation). It is supplied at call time as an INJECTED callable (``shortability.liquidity_gate``),
never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass

# SIGNAL's lookback family {12, 14, 16} trading days (SIGNAL.md §4.2-4.3; restated, not owned here —
# UNIVERSE.md §1.4). L_max = max of the family; feeds the minimum-history warmup (§1.4).
LOOKBACK_FAMILY: tuple[int, ...] = (12, 14, 16)
L_MAX: int = max(LOOKBACK_FAMILY)  # 16 trading days (UNIVERSE.md §1.4)


@dataclass(frozen=True)
class UniverseConfig:
    """RED, class-level UNIVERSE constants (UNIVERSE.md §8). Immutable; one instance per governance value.

    ``W`` — trailing dollar-volume window in days (§2.1; default 30, BLUE grid {20, 30, 45}).
    ``q_in`` / ``q_out`` — hysteresis entry / exit percentile-rank thresholds, ``q_in > q_out`` (§2.2-2.3;
        RED defaults 0.80 / 0.65, band width 0.15 — a deliberately narrower-than-fully-conservative
        operating point, §2.3, BLUE-searchable band width).
    ``N`` — target universe cardinality cap (§3.2; default 50, BLUE grid {50, 60, 75, 100}). A cap on an
        already-liquidity-ranked set, never a separate liquidity model (Invariant SIZE).
    ``min_valid_frac`` — minimum fraction of valid days in the trailing ``W``-window for ``DV_i(d)`` to
        be trusted; below it the name falls to rank 0 (§2.4's ``⌈W/2⌉``-valid-days rule = fraction 0.5).
    ``l_max`` — SIGNAL's ``L_max`` (§1.4); part of the warmup but SIGNAL-owned, restated not tuned here.

    ``w_warmup`` is a derived property pinned to ``W`` (§1.4), never a stored independent field.
    """

    W: int = 30
    q_in: float = 0.80
    q_out: float = 0.65
    N: int = 50
    min_valid_frac: float = 0.5
    l_max: int = L_MAX

    def __post_init__(self) -> None:
        # §2.2 requires q_in > q_out strictly; §2.3's hysteresis argument is undefined otherwise.
        if not (self.q_in > self.q_out):
            raise ValueError(f"hysteresis requires q_in > q_out (UNIVERSE.md §2.2); got {self.q_in} <= {self.q_out}")
        if not (0.0 < self.q_out < self.q_in < 1.0):
            raise ValueError(f"q_in/q_out must lie strictly in (0,1) (UNIVERSE.md §2.3); got {self.q_out}, {self.q_in}")
        if self.N < 1:
            raise ValueError(f"N must be >= 1 (UNIVERSE.md §3.2); got {self.N}")
        if self.W < 1:
            raise ValueError(f"W must be >= 1 (UNIVERSE.md §2.1); got {self.W}")

    @property
    def w_warmup(self) -> int:
        """UNIVERSE's own additional warmup, pinned equal to ``W`` (UNIVERSE.md §1.4).

        NOT an independent tunable: defined to equal §2.1's window ``W`` so a change to ``W`` (the BLUE
        grid) propagates to ``tau_history`` without a second governance decision. Deliberately summed
        with ``L_max`` (not netted against shared calendar days), so ``tau_history`` is a conservative
        UPPER bound on the true minimum history, consistent with P6 (§1.4).
        """
        return self.W


# The deployed RED defaults (UNIVERSE.md §8): W=30, q_in=0.80, q_out=0.65, N=50, min-valid-fraction 0.5.
DEFAULT_UNIVERSE_CONFIG = UniverseConfig()
