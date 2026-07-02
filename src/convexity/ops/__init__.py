"""OPS — the SPC monitoring / kill-switch / reconciliation layer for the crypto cross-sectional
book (``docs/specs/OPS.md``).

This package is a *control and monitoring* layer built on top of the statistical machinery SYSTEM
and REGIME already define; it invents no new probability theory (OPS.md sec 0). It reads only
realized, already-settled facts (P5) and emits one thing — an :class:`~convexity.core.types.OpsDirective`
(``RUN`` < ``DEGRADE`` < ``HALT``) — never a weight or a per-instrument veto (P3, Inv PROV).

v1 scope, laid out on OPS.md sec 8's named directory tree (all offline / simulatable, no live
exchange dependency):
  * ``config``     — RED shared control constants (sec 7, ``defaults.py``).
  * ``detectors``  — the CUSUM sequential change-detector primitive (sec 4, Page 1954) shared with
                     REGIME's own market-state detector; the Siegmund ARL closed form (sec 4.3).
  * ``streams``    — realized-fact adapters into sec 2's monitored series: ``whiten`` (z_k, sec 4.2),
                     ``recon`` (book-vs-exchange snapshot, sec 1), ``fills`` (rebalance SLA, sec 5).
  * ``reconcile``  — ``streams_map`` : per-instrument ``epsilon_i`` + escalation ladder (sec 1.2/1.4).
  * ``brake``      — the kill-switch state machine (sec 4.4), the single shared ``flatten`` routine
                     (sec 6, Inv HALT-ONE), the alarm->directive ``triggers`` precedence combiner,
                     the ``renormalize`` book repair (sec 3, no sec-8-named home — brake owns forced
                     exits), and ``engine`` : the ``OpsEngine.on_obs`` Protocol entry point (sec 12).
  * ``validate``   — ``arl`` : the reproducible ARL calibration + Monte-Carlo verification (sec 11.1,
                     Inv CAL's acceptance gate).
  * ``ops_api``    — the public entry point re-exporting ``OpsEngine`` + boundary objects (sec 8).
"""
