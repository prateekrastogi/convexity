import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 300)
PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
book = df[df["universe"] == "STAGE_A_22"].copy()
qual30 = book[book["dd_budget_30"]==True].copy()
print("dd_budget_30 qualifying count (no leverage restriction):", len(qual30))

meltup_start = pd.Timestamp("2023-01-01", tz="UTC")
meltup_end = pd.Timestamp("2024-12-31", tz="UTC")
meltup_years = (meltup_end - meltup_start).days / 365.25
print("meltup_years:", meltup_years)

def ex_meltup_cagr(row):
    total_full = (1+row["cagr"])**row["span_years"] - 1
    denom = 1 + row["ret_2023_24_meltup"]
    if denom <= 0:
        return np.nan
    total_ex = (1+total_full)/denom - 1
    span_ex = row["span_years"] - meltup_years
    if span_ex <= 0:
        return np.nan
    return (1+total_ex)**(1/span_ex) - 1

qual30["cagr_ex_meltup"] = qual30.apply(ex_meltup_cagr, axis=1)

# winner and runner-up (as identified)
winner_mask = (qual30["a_key"]=="kalman_1e-06_1e-08_0.0001") & (qual30["b_key"]=="aggressive") & \
              (qual30["e_key"]=="cdar") & (qual30["f_key"]=="corr_risk_parity_w120_spike") & (qual30["oat_axis"]=="baseline")
winner = qual30[winner_mask].iloc[0]
print("\nWINNER row:")
print(winner[["a_key","b_key","c_key","d_key","e_key","f_key","cagr","maxdd","sharpe","avg_leverage",
              "cagr_at_6.5bp","cagr_at_26bp","ret_2023_24_meltup","cagr_ex_meltup","span_years"]])

runnerup_mask = (qual30["a_key"]=="kalman_1e-06_1e-08_0.0001") & (qual30["b_key"]=="aggressive") & \
                (qual30["e_key"]=="cdar") & (qual30["f_key"]=="corr_risk_parity_w60_nospike") & (qual30["oat_axis"]=="baseline")
runnerup = qual30[runnerup_mask].iloc[0]
print("\nRUNNER-UP row:")
print(runnerup[["a_key","b_key","c_key","d_key","e_key","f_key","cagr","maxdd","sharpe","avg_leverage",
              "cagr_at_6.5bp","cagr_at_26bp","ret_2023_24_meltup","cagr_ex_meltup","span_years"]])

print("\n=== Ex-meltup ranking check: does winner still lead the field among dd_budget_30 qualifiers? ===")
top_ex = qual30.sort_values("cagr_ex_meltup", ascending=False).head(10)
cols = ["a_key","b_key","e_key","f_key","cagr","cagr_ex_meltup","ret_2023_24_meltup","maxdd"]
print(top_ex[cols].to_string(index=False))

print("\nWinner's rank by cagr_ex_meltup among qual30:")
qual30_sorted = qual30.sort_values("cagr_ex_meltup", ascending=False).reset_index(drop=True)
rank = qual30_sorted[winner_mask.reindex(qual30_sorted.index, fill_value=False) if False else
                      (qual30_sorted["a_key"]=="kalman_1e-06_1e-08_0.0001") & (qual30_sorted["b_key"]=="aggressive") &
                      (qual30_sorted["e_key"]=="cdar") & (qual30_sorted["f_key"]=="corr_risk_parity_w120_spike")].index
print("winner rank (0-indexed):", rank.tolist(), "out of", len(qual30_sorted))
