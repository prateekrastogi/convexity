"""The ONE Kelly / vol-target sizing term (LEVERAGE.md §2.5, §2.3–§2.4).

Vol targeting and fractional Kelly are the SAME closed-form optimizer (P3, §2.4): ``G_t^{Kelly} =
c_t · SR_LCB_t / σ̂_t`` IS the vol-target rule with the implied ``σ_target = c_t · μ̂ = c_t ·
SR_LCB_t``. There is exactly ONE free sizing dial, ``c`` (Invariant SIZE-ONE) — the reported
``σ_target`` is a diagnostic re-expression, never a second independently-tuned input (§2.4).

The growth-capture identity ``g(c·f*)/g(f*) = c·(2 − c)`` (§2.3) is a pure arithmetic consequence
of the quadratic ``g(f)`` — reproduced exactly (``0.4375`` at ``c=0.25``, ``0.75`` at ``c=0.5``,
§2.3/§7) as an end-to-end consistency check that the derivation is applied consistently.
"""

from __future__ import annotations

from convexity.leverage.types import GTarget


def growth_capture(c: float) -> float:
    """The §2.3 relative-growth-capture identity ``c·(2 − c)`` (LEVERAGE.md §2.3).

    Exact (not an approximation beyond the §2.2 second-order Taylor step): the fraction of
    full-Kelly log-growth captured at fractional-Kelly dial ``c``. Reproduces ``0.4375`` at
    ``c=0.25`` (quarter-Kelly) and ``0.75`` at ``c=0.5`` (half-Kelly) bit-for-bit (§2.3, §7).
    """
    return c * (2.0 - c)


def kelly_vol_target_term(c_t: float, sr_lcb: float, sigma_hat: float) -> GTarget:
    """The fractional-Kelly / vol-target sizing term ``c_t · SR_LCB_t / σ̂_t`` (LEVERAGE.md §2.5).

    ``sigma_hat`` is guaranteed ``> 0`` by ``vol/forecast.sigma_hat_t`` (Invariant VOL-FLOOR), so
    this never divides by zero; a defensive guard still floors a non-positive/NaN ``σ̂_t`` to a
    zero term rather than raising (§8 edge case). ``sr_lcb`` is already ``max(0, ci_low) ≥ 0`` from
    ``edge/lcb.py``, so the term is ``≥ 0`` — the book never runs inverted (§8).

    The diagnostic ``σ_target = c_t · SR_LCB_t`` (§2.4) is carried on the result so a test can
    assert ``kelly_term · σ̂_t == c_t · SR_LCB_t`` to float tolerance (Invariant SIZE-ONE) — there
    is no code path that sets ``σ_target`` independently of ``c``.
    """
    if not (sigma_hat > 0.0) or sigma_hat != sigma_hat:  # non-positive or NaN
        term = 0.0
    else:
        term = c_t * max(0.0, sr_lcb) / sigma_hat
    return GTarget(
        kelly_term=float(term),
        c_t=float(c_t),
        sr_lcb=float(sr_lcb),
        sigma_hat=float(sigma_hat),
        sigma_target=float(c_t * sr_lcb),  # §2.4 diagnostic re-expression (SIZE-ONE)
        growth_capture=growth_capture(c_t),
    )
