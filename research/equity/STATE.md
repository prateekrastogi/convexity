# Equity research state

- **Data phase**: complete. `research/equity/DATA.md` frozen (databento equity data lake, 102
  underlyings + 8 perp-only, full data-quality audit), landed at commit `601ef67` on branch
  `equity`.
- **E1 prereg phase**: complete. `research/equity/E1_PREREG.md` frozen — perp-native suite
  (S1 funding-carry cross-section, S2 perp<->underlying basis mean reversion, S3 overnight/weekend
  structure), N_trials = 179, pre-named deployable = S1 weekly band-ensemble + S3(c)
  overnight-execution overlay. Frozen and committed at commit `e2e586d` on branch `equity`,
  pushed to origin.
- **Next step**: Pipeline phase — write `e1_pipeline.py` implementing the frozen E1_PREREG.md grid
  exactly as declared (no new cells, no post-hoc changes to the pre-named deployable). Reuse
  `research/entry_exit/validation.py`'s `per_obs_sharpe` and
  `research/entry_exit/study/swing_pipeline.py`'s `metrics()` / `cpcv_paths()` / `dsr()` /
  `vol_target()` implementations near-verbatim (note: swing_pipeline.py hardcodes
  `REPO = Path("/home/speculation/convexity")` — treat it as a reference for logic/signatures only,
  not something importable as-is from the equity worktree; e1_pipeline.py should import the
  formulas/pattern, pointing its own REPO/data paths at
  `/home/speculation/convexity-equity/data/{stitched,funding}`). Must include the mandatory
  lookahead canary per sleeve (excluded from real results/family/PBO), the S1 regime-sanity check
  (full-panel median funding rate) and the S3(a) fact-check (overnight vs intraday at
  SPYUSDT/QQQUSDT) BEFORE any headline number is trusted, and must report every headline number
  twice (full universe + on_demo=true subset). Outcome goes in `research/equity/E1_REPORT.md`.
  Full test suite (654 passed, 1 skipped as of `601ef67`) must still pass after pipeline code
  lands: `cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/python -m pytest tests -q`.
