"""Registry artifact — detector version, params, fit windows, hashes, trial counts (BREAKOUT §11).

The auto-tuning registry artifact (BREAKOUT §11): config hash, data hash, fit dates, validation
dates, and ALL candidate trials. Critically it records the CROSS-SECTIONAL universe-scan trial count
(how many instruments were evaluated in the onboarding batch + the BLUE-grid size per symbol). That
count is the ``N`` feeding both the cross-sectional FDR/FWER gate (§10, Invariant L) and SYSTEM's
persistent multiplicity ledger (SYSTEM Invariant SG) — without it, a 9.x detector silently ships a
universe of luck.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from convexity.breakout.config.resolve import ResolvedConfig


def config_hash(cfg: ResolvedConfig) -> str:
    """Stable hash of the resolved RED/override config (the promotion artifact key)."""
    payload = {
        "cross_k": cfg.cross_k, "windows": list(cfg.windows), "preset": cfg.preset,
        "barrier_family": cfg.barrier_family, "version": cfg.defaults.detector_version,
        "weights": cfg.weights,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def data_hash(ts: np.ndarray, c: np.ndarray) -> str:
    """Stable hash of the data the fit saw (timestamps + closes)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ts).tobytes())
    h.update(np.ascontiguousarray(c).tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class RegistryArtifact:
    """The §11 onboarding artifact. ``universe_scan_n`` + ``blue_grid_size`` are the load-bearing
    cross-sectional trial counts (Invariant L / SYSTEM Inv SG)."""

    symbol: str
    detector_version: str
    resolved_config_hash: str
    data_hash: str
    fit_window: tuple[int, int]
    validation_window: tuple[int, int]
    universe_scan_n: int  # # instruments evaluated in this onboarding batch (cross-sectional N)
    blue_grid_size: int  # # BLUE candidates searched per symbol (within-symbol N)
    candidate_trials: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
