"""Generic SPC detector primitives (``docs/specs/OPS.md`` sec 8).

These are the shared change-detection tools: the CUSUM recursion (Page 1954) plus its Siegmund
(1985) closed-form ARL approximation. OPS.md sec 0 / sec 8 is explicit that this CUSUM/ARL
*construction* is genuinely shared machinery — the same class of tool REGIME instantiates for its
own market-state detector against a disjoint null — so it lives here as a reusable primitive rather
than being duplicated per problem. OPS's kill-switch (sec 4) is one caller; REGIME is another. The
``d``/``k`` notation reconciliation (OPS.md sec 0.1) is handled by naming the reference value ``d``
in the API (== REGIME/classical SPC's ``k``) to avoid clashing with the ``CUSUM`` class name.
"""
