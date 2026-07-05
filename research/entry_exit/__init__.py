"""entry_exit — formal ENTRY/EXIT candidate models + selection/validation harness.

Moved 2026-07-04 from the intraday repo's Phase-C build (intraday@800d209, since reset);
self-contained (numpy/stdlib; zero intraday imports). Target study: directional entry/exit
model selection on COMMODITY/METAL futures — convexity's stitched 1-min lake
(data/stitched/{XAUUSDT,XAGUSDT,CLUSDT,BZUSDT,NATGASUSDT,COPPERUSDT,XPTUSDT,XPDUSDT}...).
READ README.md and PREREGISTRATION.md before running anything.
"""
