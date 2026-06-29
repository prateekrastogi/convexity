"""VM-only integration suite (``@pytest.mark.nautilus``) for the live composition layer.

These tests exercise the ONLY nautilus-importing code in the repo — ``strategy/convexity_strategy.py``
and the ``*/nautilus/`` adapters — through a real ``nautilus_trader`` ``BacktestEngine`` driven by
SYNTHETIC bars (no live connectivity). They are skipped wherever ``nautilus_trader`` is not installed
(the pure-Python pillar venv), per ARCHITECTURE §8.4.
"""
