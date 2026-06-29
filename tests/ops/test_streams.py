"""Stream adapters + causal prewhitening (Invariant STREAM, CAUSAL, §5, §6.7)."""

from __future__ import annotations

import math

import numpy as np

from convexity.ops.streams.cost_tca import CostTcaStream, TcaFact
from convexity.ops.streams.edge import EdgeStream, TradeFact
from convexity.ops.streams.fills import FillFact, FillsStream
from convexity.ops.streams.heartbeat import HB_CLOCK_SKEW, HB_MISS, HB_OK, HB_STALE, HeartbeatFact, HeartbeatStream
from convexity.ops.streams.recon import ReconFact, ReconStream
from convexity.ops.streams.vol import VolFact, VolStream
from convexity.ops.streams.whiten import BatchMeans, CausalPrewhitener, standardize
from convexity.ops.types import InControlParams


def test_fills_residual_is_realized_minus_expected():
    s = FillsStream()
    obs = s.observe(FillFact(is_realized_bps=3.0, is_expected_bps=5.0,
                             fillrate_realized=1.0, fillrate_expected=1.0))
    assert obs.value == -2.0           # realized < conservative model → negative (in control)
    assert obs.stream == "fills"


def test_cost_tca_e_cost_residual():
    s = CostTcaStream()
    obs = s.observe_cost(TcaFact(cost_realized_bps=4.0, cost_forecast_bps=1.0))
    assert obs.value == 3.0            # realized exceeds forecast → positive break


def test_vol_log_ratio():
    s = VolStream()
    obs = s.observe(VolFact(realized_vol=0.4, assumed_vol=0.2))
    assert abs(obs.value - math.log(2.0)) < 1e-12   # 2× vol → +0.69


def test_vol_sigma_live_is_diagnostic_only():
    s = VolStream()
    obs = s.observe(VolFact(realized_vol=0.2, assumed_vol=0.2, sigma_live=0.5))
    assert "sigma_live" in obs.meta    # telemetry only (§4 TEAL); never enters a limit


def test_edge_emits_hit_edge_freq():
    s = EdgeStream()
    h = s.observe_hit(TradeFact(edge_realized=1.2, profitable=True))
    e = s.observe_edge(TradeFact(edge_realized=1.2, profitable=True))
    f = s.observe_frequency(2.0, 10.0)
    assert h.value == 1.0 and h.meta["kind"] == 0.0
    assert e.value == 1.2 and e.meta["kind"] == 1.0
    assert f.value == -8.0 and f.meta["kind"] == 2.0   # collapse


def test_recon_binary_mismatch():
    s = ReconStream()
    ok = s.observe(ReconFact(reconciliation_ok=True))
    bad = s.observe(ReconFact(reconciliation_ok=False, detail={"pos_delta": 3.0}))
    assert ok.value == 0.0 and ok.binary
    assert bad.value == 1.0 and bad.meta["pos_delta"] == 3.0


def test_heartbeat_classification_fails_toward_safe():
    s = HeartbeatStream(staleness_window_ms=1000, clock_skew_tol_ms=100)
    assert s.observe(HeartbeatFact(True, 0, 0)).meta["code"] == HB_OK
    assert s.observe(HeartbeatFact(True, 0, 500)).meta["code"] == HB_CLOCK_SKEW
    assert s.observe(HeartbeatFact(False, 0, 0)).meta["code"] == HB_MISS
    assert s.observe(HeartbeatFact(True, 5000, 0)).meta["code"] == HB_STALE
    # session-aware: a gap during a closed session is NOT an outage (P6)
    assert s.observe(HeartbeatFact(True, 5000, 0, in_session=False)).meta["code"] == HB_OK


def test_prewhitener_whitens_ar1_to_iid():
    """§6.7a: the prewhitened innovation is ~unit-variance and ~white in control."""
    ic = InControlParams(stream="cost_tca", mu0=0.0, sigma0=2.0, phi=(0.4,), rho=0.4)
    rng = np.random.default_rng(0)
    # raw AR(1) residual with sigma0=2
    n = 100000
    raw = np.empty(n)
    raw[0] = rng.standard_normal()
    for i in range(1, n):
        raw[i] = 0.4 * raw[i - 1] + rng.standard_normal() * math.sqrt(1 - 0.16)
    raw *= 2.0
    pw = CausalPrewhitener(ic)
    z = np.array([pw.update(x) for x in raw])
    assert abs(z.std() - 1.0) < 0.05
    assert abs(np.corrcoef(z[:-1], z[1:])[0, 1]) < 0.05   # whitened


def test_prewhitener_is_causal():
    """Invariant CAUSAL: the prewhitener reads only residual_i and the frozen prior residual."""
    ic = InControlParams(stream="vol", mu0=0.0, sigma0=1.0, phi=(0.5,))
    x = [0.3, -0.2, 0.5, 0.1, -0.4]
    a = CausalPrewhitener(ic)
    za = [a.update(v) for v in x]
    b = CausalPrewhitener(ic)
    zb = [b.update(v) for v in x[:3]]      # feed only first 3
    assert za[:3] == zb                     # the first 3 are unaffected by later samples


def test_standardize_uses_frozen_sigma0():
    ic = InControlParams(stream="cost_tca", mu0=1.0, sigma0=2.0)
    assert standardize(5.0, ic) == 2.0     # (5-1)/2


def test_batch_means_emits_per_block():
    ic = InControlParams(stream="edge", mu0=0.0, sigma0=1.0)
    bm = BatchMeans(ic, block_length=4)
    out = [bm.update(1.0) for _ in range(8)]
    assert out[0] is None and out[2] is None
    assert out[3] is not None and out[7] is not None   # emits at each block boundary
