"""Unit tests for MODELS.jumps -- the Lee-Mykland directional jump-continuation ENTRY. SYNTHETIC ONLY.

The two mandated Phase-C tests (task contract, SYSTEM §C.2), on a KNOWN generating process whose truth
we control -- NO real lake is touched:

  (1) KNOWN-GENERATING-PROCESS RECOVERY -- simulate a Gaussian-diffusion price path with jumps INJECTED
      at controlled times and signs; assert the Lee-Mykland statistic detects the jumps at the right
      bars in the right direction, with CONTROLLED TEST SIZE (a pure jump-free diffusion produces
      essentially no detections at the frozen Gumbel level -- the significance is calibrated, not tuned).
  (2) NO-LOOKAHEAD PINNING -- a test that FAILS if the decision at bar `t` reads bar `t` (or later):
      `signal_at` reads only `cursor.past()`, and a same-bar read through the guarded cursor RAISES;
      the detection at `t` is invariant to mutating `series[t]` (proving no same-bar dependence).

The candidate is imported; the SYSTEM harness / lake are NOT (SYSTEM §C.3 off-ledger, §C.8 data gate).
Deterministic: all paths are seeded (`random.Random`) so the recovery assertions are reproducible.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from entry_exit.base import Cadence, EntryModel, PurityError, ScanCursor, Side, TrainFold
from entry_exit.jumps import (
    MU1,
    LeeMyklandJumpEntry,
    bipower_sigma,
    gumbel_critical_value,
)


# --- Controlled synthetic DGP: a Gaussian-diffusion price path with jumps injected at known bars -----
def _diffusion_prices(n: int, sigma: float, seed: int) -> list[float]:
    """A pure jump-FREE Gaussian random walk of `n` prices: increments iid N(0, sigma^2). Known truth."""
    rng = random.Random(seed)
    p = [0.0]
    for _ in range(n - 1):
        p.append(p[-1] + rng.gauss(0.0, sigma))
    return p


def _inject_jump(prices: list[float], at_return: int, size: float) -> list[float]:
    """Add a jump of magnitude `size` (signed) to the return ENDING at price-index `at_return`.

    Shifts every price from `at_return` onward by `size`, so the single return
    `prices[at_return] - prices[at_return-1]` gains `size` and all others are unchanged -- a clean,
    controlled level shift at a known time and sign (the truth the recovery test asserts against).
    """
    out = list(prices)
    for i in range(at_return, len(out)):
        out[i] += size
    return out


def _train(vals: list[float], warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(vals)))
    return TrainFold(days=days, series=tuple(vals), warmup=warmup)


def _fitted(window: int = 20, alpha: float = 0.01, n_train: int = 300) -> LeeMyklandJumpEntry:
    """A model fit on a jump-free diffusion train fold (pins the Gumbel n; freezes G(alpha))."""
    train_prices = _diffusion_prices(n_train, sigma=0.01, seed=999)
    return LeeMyklandJumpEntry(window=window, alpha=alpha).fit(_train(train_prices))


# ==================================================================================================
# Conformance: the candidate satisfies the EntryModel interface WITHOUT the harness importing it back
# ==================================================================================================
def test_conforms_to_entry_model_protocol():
    m = _fitted()
    assert isinstance(m, EntryModel)              # structural fit + signal_at + k
    assert isinstance(m.k, int) and m.k == 2      # (window K, alpha) -- the two frozen free params
    assert m.cadence is Cadence.DAILY             # DAILY-cadence: strictly F_{t-1}-measurable


def test_signal_before_fit_raises():
    with pytest.raises(RuntimeError):
        LeeMyklandJumpEntry().signal_at(ScanCursor(series=(0.0,) * 30, t=25))


def test_constructor_validates_params():
    with pytest.raises(ValueError):
        LeeMyklandJumpEntry(window=1)             # need an adjacent pair for bipower
    with pytest.raises(ValueError):
        LeeMyklandJumpEntry(alpha=0.0)
    with pytest.raises(ValueError):
        LeeMyklandJumpEntry(alpha=1.0)


# ==================================================================================================
# Building-block correctness (the pieces the recovery test rests on)
# ==================================================================================================
def test_bipower_sigma_recovers_known_diffusion_vol():
    # KNOWN truth: iid N(0, sigma^2) returns => bipower sigma_hat -> sigma. Recover it within tolerance.
    rng = random.Random(7)
    sigma_true = 0.02
    rets = [rng.gauss(0.0, sigma_true) for _ in range(4000)]
    assert bipower_sigma(rets) == pytest.approx(sigma_true, rel=0.06)


def test_bipower_is_robust_to_a_jump_in_the_window():
    # A lone huge jump in the estimation window must NOT inflate the diffusive-vol estimate much
    # (bipower annihilates it via its jump-free neighbour) -- the property Lee-Mykland rely on.
    rng = random.Random(11)
    rets = [rng.gauss(0.0, 0.01) for _ in range(2000)]
    base = bipower_sigma(rets)
    rets_j = list(rets)
    rets_j[1000] += 5.0  # a massive jump
    jumped = bipower_sigma(rets_j)
    # A realized-VARIANCE (sum r^2) estimate would explode; bipower stays within a modest factor.
    assert jumped < 3.0 * base


def test_gumbel_critical_value_monotonic_in_alpha_and_positive():
    # Smaller alpha (stricter) => larger critical value; both positive for a sane sample size.
    strict = gumbel_critical_value(0.001, 300)
    loose = gumbel_critical_value(0.05, 300)
    assert strict > loose > 0.0
    with pytest.raises(ValueError):
        gumbel_critical_value(0.01, 2)            # n < 3: log log n domain error


def test_mu1_constant():
    assert MU1 == pytest.approx(math.sqrt(2.0 / math.pi))


# ==================================================================================================
# (1) KNOWN-GENERATING-PROCESS RECOVERY -- injected jumps detected at the right time AND sign
# ==================================================================================================
def test_recovery_injected_up_jump_detected_as_long():
    # Controlled DGP: jump-free diffusion + one LARGE UP jump on the return ending at price-index 200.
    # The model, deciding to execute at t = 201, reads the return ending at t-1 = 200 -> LONG.
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=1), at_return=200, size=0.20)
    cur = ScanCursor(series=tuple(prices), t=201, cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.LONG


def test_recovery_injected_down_jump_detected_as_short():
    # Same, a large DOWN jump on the return ending at index 200 -> SHORT (directional continuation).
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=2), at_return=200, size=-0.20)
    cur = ScanCursor(series=tuple(prices), t=201, cadence=Cadence.DAILY)
    assert m.signal_at(cur) is Side.SHORT


def test_recovery_jump_detected_at_the_right_TIME_not_the_neighbours():
    # The jump is on the return ending at index 200. Deciding to execute the bar AFTER that return
    # (t=201, reading r ending at 200) fires; deciding well before it (the jump not yet in past())
    # is FLAT -- the detection is localized to the correct time, not smeared onto neighbours.
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=3), at_return=200, size=0.20)
    fires = m.signal_at(ScanCursor(series=tuple(prices), t=201, cadence=Cadence.DAILY))
    # A decision at t=150 cannot see the jump at 200 (it is not in series[:150]) -> FLAT.
    before = m.signal_at(ScanCursor(series=tuple(prices), t=150, cadence=Cadence.DAILY))
    assert fires is Side.LONG
    assert before is Side.FLAT


def test_recovery_test_size_controlled_on_pure_diffusion():
    # CONTROLLED TEST SIZE: a PURE jump-free diffusion (the null) must yield ~no detections at the
    # frozen alpha=0.01 Gumbel level over the whole scan -- the significance is calibrated, not tuned.
    m = _fitted(window=20, alpha=0.01, n_train=600)
    prices = _diffusion_prices(600, sigma=0.01, seed=4242)
    detections = 0
    for t in range(m._warmup, len(prices)):
        if m.signal_at(ScanCursor(series=tuple(prices), t=t, cadence=Cadence.DAILY)) is not Side.FLAT:
            detections += 1
    # Gumbel max-statistic control: the family-wise false-alarm rate is ~alpha, so a single-digit (or
    # zero) count of spurious flags over ~580 bars, never a flood. A leaky/miscalibrated statistic
    # would fire on a large fraction of bars.
    assert detections <= 5


def test_recovery_injected_jump_survives_where_diffusion_is_silent():
    # Direct contrast on the SAME seed/path: with no jump -> FLAT; inject a jump on that return -> a
    # signed signal appears. The signal is CAUSED by the injected jump, not the diffusion noise.
    m = _fitted(window=20, alpha=0.01)
    clean = _diffusion_prices(400, sigma=0.005, seed=77)
    jumped = _inject_jump(clean, at_return=250, size=0.25)
    silent = m.signal_at(ScanCursor(series=tuple(clean), t=251, cadence=Cadence.DAILY))
    fired = m.signal_at(ScanCursor(series=tuple(jumped), t=251, cadence=Cadence.DAILY))
    assert silent is Side.FLAT
    assert fired is Side.LONG


def test_fit_freezes_gumbel_n_and_is_train_only():
    # fit pins the Gumbel sample size n on the TRAIN fold and freezes G(alpha); it is reused OOS
    # unchanged (never re-derived from the TEST length -- a TEST-side re-tune is barred, SYSTEM §C.1).
    m = _fitted(window=20, alpha=0.01, n_train=300)
    frozen_crit = m._crit
    frozen_n = m._n
    assert frozen_crit is not None and frozen_n is not None
    # A far LONGER OOS scan does NOT change the frozen critical value (it is not re-derived from TEST).
    long_prices = _diffusion_prices(5000, sigma=0.01, seed=8)
    _ = m.signal_at(ScanCursor(series=tuple(long_prices), t=4000, cadence=Cadence.DAILY))
    assert m._crit == frozen_crit and m._n == frozen_n


# ==================================================================================================
# Warm-up honesty (the scan fires only after its own burn-in, within the fold)
# ==================================================================================================
def test_warmup_returns_flat_before_burnin():
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=5), at_return=3, size=0.5)
    # window+2 = 22 price points needed; a decision at t < 22 has too little causal history -> FLAT,
    # even sitting right on top of a large injected jump (no pre-sample borrowing).
    assert m.signal_at(ScanCursor(series=tuple(prices), t=10, cadence=Cadence.DAILY)) is Side.FLAT
    assert m._warmup == 22


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING -- the decision at t must NOT read bar t (or later)
# ==================================================================================================
def test_pinning_signal_is_invariant_to_the_same_bar_value():
    # The decision executing at t reads ONLY series[:t]; mutating series[t] (the SAME bar) or any later
    # bar must NOT change the signal. A model that peeked at bar t would flip here -- this FAILS on a
    # 1-bar leak, the mandated pinning standard (SYSTEM §C.2, DAILY cadence).
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=6), at_return=200, size=0.20)
    base = m.signal_at(ScanCursor(series=tuple(prices), t=201, cadence=Cadence.DAILY))
    assert base is Side.LONG
    # Poison bar t=201 (and everything after) with a huge OPPOSITE-sign move; the causal decision at
    # t=201 cannot see it, so the signal is unchanged.
    poisoned = list(prices)
    for i in range(201, len(poisoned)):
        poisoned[i] -= 5.0
    same = m.signal_at(ScanCursor(series=tuple(poisoned), t=201, cadence=Cadence.DAILY))
    assert same is base                     # invariant to bar t and later -> no same-bar/future leak


def test_pinning_only_reads_past_window():
    # signal_at reads cursor.past() only; expose a guarded cursor and confirm a same-bar read RAISES
    # (the F_{t-1} guard) while the model's OWN reads (all < t) never trip it. A wiring that let the
    # decision read bar >= t would break here (SYSTEM §C.2 pinning hook).
    cur = ScanCursor(series=tuple(_diffusion_prices(60, sigma=0.01, seed=9)), t=40, cadence=Cadence.DAILY)
    assert len(cur.past()) == 40            # indices 0..39 -- strictly before the t=40 execution bar
    with pytest.raises(PurityError):
        cur.read(40)                        # same bar t -> leak
    with pytest.raises(PurityError):
        cur.read(41)                        # future bar -> leak
    # The model runs on this cursor without any lookahead read (it stays inside past()).
    _fitted().signal_at(cur)                # no PurityError raised -> all reads are < t


def test_pinning_same_time_decision_uses_return_ending_at_t_minus_1():
    # Explicit F_{t-1} check: the jump is on the return ending at index 200. Executing AT t=200 (whose
    # last causal return ends at 199) does NOT yet see it; executing at t=201 (last return ends at 200)
    # DOES. So the jump that fires an entry at t is measurable from info up to t-1, never bar t.
    m = _fitted(window=20, alpha=0.01)
    prices = _inject_jump(_diffusion_prices(400, sigma=0.005, seed=10), at_return=200, size=0.20)
    at_200 = m.signal_at(ScanCursor(series=tuple(prices), t=200, cadence=Cadence.DAILY))
    at_201 = m.signal_at(ScanCursor(series=tuple(prices), t=201, cadence=Cadence.DAILY))
    assert at_200 is Side.FLAT              # the jump return (ending at 200) is NOT in series[:200]
    assert at_201 is Side.LONG              # it IS in series[:201] -> fires at the next execution bar
