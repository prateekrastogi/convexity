import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 260)
PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
book = df[df["universe"] == "STAGE_A_22"].copy()
lev_ok = book[(book["avg_leverage"] >= 2.0) & (book["avg_leverage"] <= 5.0)].copy()
print("lev_ok count:", len(lev_ok))
print(lev_ok["maxdd"].describe())
print("\nmaxdd min (best) rows:")
cols = ["a_key","b_key","c_key","d_key","e_key","f_key","oat_axis","cagr","maxdd","sharpe","avg_leverage","ruin_flag"]
print(lev_ok.sort_values("maxdd").head(15)[cols].to_string(index=False))
print("\ne_key value counts in lev_ok:")
print(lev_ok["e_key"].value_counts())
print("\nruin flag counts in lev_ok:")
print(lev_ok["ruin_flag"].value_counts())

# what about whole book avg_leverage distribution
print("\nfull book avg_leverage describe:")
print(book["avg_leverage"].describe())
print(book["avg_leverage"].quantile([0.5,0.75,0.9,0.95,0.99]))
