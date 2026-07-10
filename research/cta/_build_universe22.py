"""Deterministic Stage-A universe subset selection: stratified by trailing dollar-volume tercile
x history-length bin, maximizing sector diversity within each stratum. Rule is fully reproducible
(no randomness) -- see docstring below, reused verbatim in CTA_MATH_NOTES.md and as a function in
ctaA_sweep.py.
"""
import pandas as pd

manifest = pd.read_csv("/home/speculation/convexity-cta/research/cta/_universe_manifest_93.csv")
manifest = manifest.sort_values("symbol").reset_index(drop=True)

# KNOWN-BAD DATA EXCLUSION (found during Stage-A feasibility testing, independently corroborated
# by Axis E's own §E.6 data-quality finding AND research/equity/DATA.md's manifest showing zero
# detected splits for AAPLUSDT): AAPLUSDT's real 2020-08-31 4-for-1 stock split was never detected/
# adjusted by the shared data-lake's split pipeline, producing a fake -74.12% single-day return
# that would corrupt every downstream statistic (worst-day, Kelly/EVT/CDaR leverage, HMM/GARCH
# fits) for any cell that includes it. Verified independently here (not just taking Axis E's word
# for it): direct inspection of data/stitched/AAPLUSDT.parquet confirms the exact date/magnitude,
# and DATA.md's own per-name manifest table lists AAPLUSDT's "Splits" column as "-" (zero detected)
# despite this being a real, undisputed corporate action. This is a known main-branch data lake
# bug, out of scope to fix here (per the host's explicit prohibition on touching anything upstream
# of this worktree) -- the correct, safe response is to exclude it from the Stage-A pilot universe
# and let the SAME deterministic selection rule pick the next-best candidate in its stratum.
KNOWN_BAD_DATA = {"AAPLUSDT"}
manifest = manifest[~manifest["symbol"].isin(KNOWN_BAD_DATA)].reset_index(drop=True)

FORCED = ["SPYUSDT", "QQQUSDT"]
TARGET_TOTAL = 22
N_EXTRA = TARGET_TOTAL - len(FORCED)  # 20

rest = manifest[~manifest["symbol"].isin(FORCED)].copy()

# 3 equal-sized dollar-volume terciles (rank-based, deterministic)
rest = rest.sort_values("dollar_vol").reset_index(drop=True)
n = len(rest)
tercile_edges = [n // 3, 2 * n // 3]
def liq_tercile(i):
    if i < tercile_edges[0]:
        return "low"
    elif i < tercile_edges[1]:
        return "med"
    return "high"
rest["liq_tercile"] = [liq_tercile(i) for i in range(n)]

# 2 history bins split at cross-sectional median span_days
median_span = rest["span_days"].median()
rest["history_bin"] = rest["span_days"].apply(lambda s: "long" if s >= median_span else "short")

rest["stratum"] = rest["liq_tercile"] + "_" + rest["history_bin"]
strata = sorted(rest["stratum"].unique())
print("median_span_days:", median_span)
print("strata:", strata)
print(rest["stratum"].value_counts())

# per-stratum quota: base 20//6=3, remainder 2 distributed to the two strata with the MOST
# candidates available (deterministic, largest-remainder-style allocation)
base_quota = N_EXTRA // len(strata)
remainder = N_EXTRA - base_quota * len(strata)
counts = rest["stratum"].value_counts()
# deterministic tie-break: sort by (count desc, stratum name asc)
order = sorted(strata, key=lambda s: (-counts[s], s))
quota = {s: base_quota for s in strata}
for s in order[:remainder]:
    quota[s] += 1
print("quota:", quota)

selected = list(FORCED)
selected_sectors = set(manifest.set_index("symbol").loc[FORCED, "sector"])

for s in strata:
    cand = rest[rest["stratum"] == s].copy()
    # greedy: prefer a sector not yet represented, tie-break alphabetical symbol
    cand["sector_novel"] = ~cand["sector"].isin(selected_sectors)
    cand = cand.sort_values(["sector_novel", "symbol"], ascending=[False, True])
    picked_this_stratum = []
    for _, row in cand.iterrows():
        if len(picked_this_stratum) >= quota[s]:
            break
        picked_this_stratum.append(row["symbol"])
        selected_sectors.add(row["sector"])
    selected.extend(picked_this_stratum)

selected = sorted(set(selected))
print(f"\nFINAL SELECTED ({len(selected)}):")
sel_df = manifest[manifest["symbol"].isin(selected)].sort_values("symbol")
print(sel_df[["symbol", "sector", "span_days", "dollar_vol"]].to_string())
print("\nUnique sectors covered:", sel_df["sector"].nunique(), "/ 22")
print("\nPython list:")
print(selected)
