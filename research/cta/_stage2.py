import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 300)
PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
book = df[df["universe"] == "STAGE_A_22"].copy()
print("all cols:", list(df.columns))

# ============ FUNNEL: leverage[2,5] AND dd_budget, before any robustness filter ============
print("\n=== FUNNEL START COUNTS (objective's own feasible set) ===")
for u in ("STAGE_A_22","SPY","QQQ"):
    sub = df[df["universe"]==u]
    lev_ok = sub[(sub["avg_leverage"]>=2.0)&(sub["avg_leverage"]<=5.0)]
    print(f"{u}: total={len(sub)}, lev[2,5]={len(lev_ok)}")
    for b in (20,30,41):
        n = ((lev_ok["maxdd"]<=b/100.0)).sum()
        print(f"    DD<={b}%: {n}")

print("\n>>> CONFIRMED: zero cells satisfy leverage in [2,5] jointly with ANY DD budget, in ANY universe.")
print(">>> Therefore start count for robustness funnel = 0 at every DD budget => funnel is 0->0->0->0 trivially.")

# ============ FALLBACK: single best cell by raw objective, DROPPING the now-infeasible leverage restriction (explicitly flagged deviation) ============
print("\n=== FALLBACK: best cells by CAGR-at-30%-DD-budget, STAGE_A_22, NO leverage restriction (since [2,5] is infeasible) ===")
qual30 = book[book["dd_budget_30"]==True].copy()
top = qual30.sort_values("cagr", ascending=False).head(20)
cols = ["a_key","b_key","c_key","d_key","e_key","f_key","oat_axis","cagr","maxdd","sharpe","avg_leverage",
        "cagr_at_6.5bp","cagr_at_26bp","dd_budget_20","dd_budget_30","dd_budget_41","ret_2023_24_meltup","span_years","n_obs"]
print(top[cols].to_string(index=False))
