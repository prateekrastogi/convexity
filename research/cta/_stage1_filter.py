import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 260)

PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
print("TOTAL ROWS:", len(df))
print(df["universe"].value_counts())

book = df[df["universe"] == "STAGE_A_22"].copy()
print("\nSTAGE_A_22 rows:", len(book))

# leverage filter [2,5]
lev_ok = book[(book["avg_leverage"] >= 2.0) & (book["avg_leverage"] <= 5.0)].copy()
print("After leverage-in-[2,5] filter:", len(lev_ok))

for budget in (20, 30, 41):
    col = f"dd_budget_{budget}"
    qual = lev_ok[lev_ok[col] == True].copy()
    print(f"\n--- DD budget {budget}%: qualifying cells (lev-filtered) = {len(qual)} ---")
    if len(qual) == 0:
        continue
    # top decile threshold by primary cagr
    n_top = max(1, int(np.ceil(len(qual) * 0.10)))
    top_primary = qual.sort_values("cagr", ascending=False).head(n_top)
    print(f"  top decile count (by primary 13bp cagr): {n_top}")
    print(f"  best cagr: {qual['cagr'].max():.4f}  threshold cagr(10th pct cutoff): {top_primary['cagr'].min():.4f}")

