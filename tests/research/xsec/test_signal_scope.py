"""SIGNAL Invariant SCOPE (SIGNAL.md §7) + RED-constant governance — hardening tests.

These are the invariant-hardening tests the manifest assigns to a NEW file (not edits to
``signals.py``'s functions nor to the ensemble tests already in ``test_signals.py``). They cover
the one named SIGNAL invariant not yet exercised elsewhere — **SCOPE** — plus a static guard that
the RED signal-construction constants (SIGNAL.md §7) expose no per-symbol / per-sleeve override
surface. Every test fails if the invariant it names is violated.

Invariant SCOPE (SIGNAL.md §7): the cross-sectional momentum edge is crypto-specific — tested and
found NOT to extend to the equity/commodity sleeves (docs/EDGE.md §d). The signal functions are
class-agnostic CODE (mechanically runnable on any panel), and the crypto-only restriction lives in
governance, not in a numeric constant: the functions therefore MUST carry no sleeve/asset-class
parameter and MUST NOT branch on sleeve. The deployable core (``ensemble_momentum_signal``) is also
funding-blind (``w_carry = 0``), so on the momentum-only, zero-funding non-crypto sleeves the panel
layer builds (``build_class_panel(..., zero_funding=True)`` — SIGNAL.md §7 AMBER, panel.py) it
produces a well-defined rank driven only by trailing price, never a silent carry contribution.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from research.xsec import signals as sig
from research.xsec.panel import Panel

_SIGNAL_FNS = (
    sig.momentum_signal,
    sig.carry_signal,
    sig.combined_signal,
    sig.ensemble_momentum_signal,
)


def _build_panel(
    close: np.ndarray, funding: np.ndarray, listed: np.ndarray, symbols: tuple[str, ...]
) -> Panel:
    """Assemble a ``Panel`` directly from arrays (mirrors conftest.make_panel) — no lake I/O."""
    n_days = close.shape[0]
    dates = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=n_days, freq="D"))
    first_day = {s: int(np.flatnonzero(listed[:, j])[0]) for j, s in enumerate(symbols)}
    last_day = {s: int(np.flatnonzero(listed[:, j])[-1]) for j, s in enumerate(symbols)}
    return Panel(
        dates,
        symbols,
        np.where(listed, close, np.nan),
        np.where(listed, funding, np.nan),
        listed,
        first_day,
        last_day,
    )


# The CLOSED allow-set of parameter names any signal function may legitimately expose (SIGNAL.md
# §4.3/§4.4 build-contracts): the panel + as-of index, and the SHARED lookback/skip/weight knobs.
# A structural ALLOWLIST (not a hand-picked blocklist) so that ANY unanticipated parameter — a
# sleeve knob under a novel name (``panel_class``, ``market_tag``), a per-symbol keyed override
# (``lookback_map``, ``symbol_weights``), or anything else — trips this guard, closing the
# "differently-named leak" hole the fixed-blocklist form left open (audit gap: static SCOPE guards
# only caught a fixed forbidden set of NAMES).
_ALLOWED_SIGNAL_PARAMS = {
    "panel",  # the point-in-time data (close, funding, listed) — the ONLY state a signal may read
    "d",  # the as-of row index (Invariant NLA: nothing > d)
    "lookback",  # single-member trailing window (momentum/carry/combined)
    "lookbacks",  # the validated BAND for the ensemble core (§4.3)
    "skip",  # RED skip = 0 (Inv SKIP)
    "carry_lookback",  # combined_signal's carry window (research/diagnostic only)
    "weights",  # ensemble band weights, SHARED not per-symbol (RED, §7)
}


def test_scope_signal_params_are_a_closed_allowlist_no_sleeve_or_override_surface():
    # Invariant SCOPE (§7) + §7 RED discipline, as a SINGLE structural guard: the crypto-only
    # restriction is governance, NOT a code branch, and the RED construction constants are SHARED,
    # unreachable per-symbol. Rather than blocklisting known-bad names (which a leak under a fresh
    # name — ``panel_class``, ``lookback_map`` — slips past), we assert every signal function's
    # parameters are a SUBSET of the closed allow-set above. Any sleeve/asset-class knob OR any
    # per-symbol/per-instrument keyed override, regardless of what it is named, adds a parameter
    # outside the set and FAILS here — the first sign the firewall leaked into the signal layer.
    for fn in _SIGNAL_FNS:
        params = set(inspect.signature(fn).parameters)
        extra = params - _ALLOWED_SIGNAL_PARAMS
        assert not extra, (
            f"{fn.__name__} exposes non-allowlisted parameter(s) {extra} — a sleeve/class knob or "
            f"per-symbol override would appear here (Inv SCOPE / §7 RED). Allowed: {_ALLOWED_SIGNAL_PARAMS}"
        )


def test_scope_ensemble_is_asset_class_agnostic_code():
    # Invariant SCOPE (§7): running against a non-crypto (zero-funding, momentum-only) panel is
    # mechanically POSSIBLE (the code is class-agnostic) and returns a well-defined rank — it is
    # simply NOT a validated/deployable configuration. This documents that the restriction is not
    # a hard-coded refusal in signals.py; it is a governance boundary the promotion gate enforces.
    n_days, syms, d = 40, ("X", "Y", "Z"), 30
    t = np.arange(n_days)
    close = np.column_stack(
        [100.0 * 1.003**t, 100.0 * 1.001**t, 100.0 * 0.999**t]  # a momentum-only sleeve
    )
    funding = np.zeros((n_days, 3))  # equity/commodity: FUNDING = 0 (§7 AMBER, universe.py)
    listed = np.ones((n_days, 3), dtype=bool)
    panel = _build_panel(close, funding, listed, syms)
    rank = sig.ensemble_momentum_signal(panel, d)
    finite = np.isfinite(rank)
    assert rank.shape == (3,)
    assert finite.all()  # a well-defined [0,1] rank on every eligible name
    assert ((rank >= 0.0) & (rank <= 1.0)).all()
    # strongest uptrend (X) ranks top, downtrend (Z) bottom — the same ordering contract as crypto.
    assert rank[0] == rank.max()
    assert rank[2] == rank.min()


def test_scope_core_is_funding_blind_on_zero_funding_sleeve():
    # Invariant SCOPE composed with CARRY-ZERO (§6.2/§7): on a momentum-only sleeve the panel
    # builder zeroes funding on live cells; because the deployable core carries ``w_carry = 0`` it
    # never reads funding, so the zero-funding sleeve's ensemble rank is bit-identical to the SAME
    # price panel carrying arbitrary (nonzero) funding. No carry can leak into a non-crypto book.
    n_days, syms, d = 40, ("X", "Y", "Z"), 30
    t = np.arange(n_days)
    close = np.column_stack([100.0 * 1.003**t, 100.0 * 1.001**t, 100.0 * 0.999**t])
    listed = np.ones((n_days, 3), dtype=bool)
    zero_fund = _build_panel(close, np.zeros((n_days, 3)), listed, syms)
    # same prices, adversarial nonzero funding (as if carry were live) — the core must ignore it.
    nz_fund = _build_panel(
        close, np.random.default_rng(3).normal(size=(n_days, 3)), listed, syms
    )
    np.testing.assert_array_equal(
        sig.ensemble_momentum_signal(zero_fund, d),
        sig.ensemble_momentum_signal(nz_fund, d),
    )


def test_scope_red_constants_are_shared_not_per_symbol_permutation_invariant():
    # §7 RED discipline (mirrors COST Inv C-cfg / SYSTEM Inv C), as a DYNAMIC guard complementing
    # the closed-allowlist static test: the lookback BAND, skip and ensemble weights are SHARED
    # across symbols — no coin gets a "special" lookback. If a signal secretly keyed a construction
    # constant to a symbol's COLUMN POSITION (a positional per-symbol override), the boost would
    # stay pinned to that index instead of following the symbol when columns are reordered. So we
    # assert the signal is EQUIVARIANT under a column permutation: r_perm[i] == r_base[perm[i]] for
    # every symbol. This is COMPLEMENTARY to the relabel guard below (#test_scope_no_hidden...): a
    # relabel preserves column ORDER so it cannot see a positional override, whereas a permutation
    # reorders columns so it can. A truly shared-constant signal is equivariant; a positional
    # per-symbol override breaks it (verified by injection — a position-0 boost trips this test but
    # sails past the relabel test).
    n_days, d = 40, 30
    t = np.arange(n_days)
    close = np.column_stack([100.0 * 1.003**t, 100.0 * 1.001**t, 100.0 * 0.999**t])
    funding = np.column_stack(
        [np.full(n_days, 0.0005), np.full(n_days, -0.0001), np.full(n_days, -0.0007)]
    )
    listed = np.ones((n_days, 3), dtype=bool)
    base = _build_panel(close, funding, listed, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    perm = np.array([2, 0, 1])  # a non-trivial column reordering
    permuted = _build_panel(
        close[:, perm], funding[:, perm], listed[:, perm], ("SOLUSDT", "BTCUSDT", "ETHUSDT")
    )
    cases = (
        (sig.momentum_signal, {"lookback": 12, "skip": 0}),
        (sig.carry_signal, {"lookback": 7}),
        (sig.combined_signal, {"lookback": 12, "skip": 0}),
        (sig.ensemble_momentum_signal, {}),
    )
    for fn, kw in cases:
        r_base = fn(base, d, **kw)
        r_perm = fn(permuted, d, **kw)
        # equivariance: r_perm[i] must equal r_base[perm[i]] for every symbol
        np.testing.assert_array_equal(
            r_perm,
            r_base[perm],
            err_msg=f"{fn.__name__} not permutation-equivariant — a construction constant is keyed "
            f"to symbol identity, not SHARED (§7 RED per-symbol-override bug)",
        )


def test_scope_no_hidden_sleeve_branch_on_panel_contents():
    # Invariant SCOPE (§7), the DISCRIMINATING (non-static) guard: a sleeve special-case need not
    # arrive as a named parameter — it could hide as a branch on the panel's own metadata (symbol
    # names, e.g. ``endswith("USDT")`` for crypto, or the date index). A pure ``F_d``-measurable
    # signal (Invariant NLA/RANK) is a function of ``(close, funding, listed)`` ONLY, so relabeling
    # the symbols and shifting the calendar while holding the three data arrays byte-identical MUST
    # leave every signal bit-for-bit unchanged. This FAILS if any signal function branches on
    # sleeve-identifying metadata — which the static ``inspect.signature`` guards above cannot catch.
    n_days, d = 40, 30
    t = np.arange(n_days)
    close = np.column_stack([100.0 * 1.003**t, 100.0 * 1.001**t, 100.0 * 0.999**t])
    funding = np.column_stack(
        [np.full(n_days, 0.0005), np.full(n_days, -0.0001), np.full(n_days, -0.0007)]
    )
    listed = np.ones((n_days, 3), dtype=bool)
    crypto = _build_panel(close, funding, listed, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    # SAME data arrays, non-crypto-looking symbols + a differently-dated calendar (relabel only).
    relabeled = Panel(
        pd.DatetimeIndex(pd.date_range("1999-06-01", periods=n_days, freq="D")),
        ("AAPL", "GLD", "CL_F"),
        crypto.close,
        crypto.funding,
        crypto.listed,
        crypto.first_day,
        crypto.last_day,
    )
    cases = (
        (sig.momentum_signal, {"lookback": 12, "skip": 0}),
        (sig.carry_signal, {"lookback": 7}),
        (sig.combined_signal, {"lookback": 12, "skip": 0}),
        (sig.ensemble_momentum_signal, {}),
    )
    for fn, kw in cases:
        np.testing.assert_array_equal(
            fn(crypto, d, **kw),
            fn(relabeled, d, **kw),
            err_msg=f"{fn.__name__} output moved under symbol/calendar relabel (hidden sleeve branch?)",
        )
