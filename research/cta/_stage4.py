import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 300)
PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
book = df[df["universe"] == "STAGE_A_22"].copy()

cols = ["a_key","b_key","c_key","d_key","e_key","f_key","cagr","maxdd","sharpe","avg_leverage","cagr_at_6.5bp","cagr_at_26bp"]
for b in (20,30,41):
    qual = book[book[f"dd_budget_{b}"]==True]
    print(f"\n=== best cell at DD<=={b}% budget (no leverage restriction), n_qualifying={len(qual)} ===")
    if len(qual):
        print(qual.sort_values("cagr", ascending=False).head(3)[cols].to_string(index=False))

# winner & runner-up full detail across budgets
winner_mask = (book["a_key"]=="kalman_1e-06_1e-08_0.0001") & (book["b_key"]=="aggressive") & \
              (book["e_key"]=="cdar") & (book["f_key"]=="corr_risk_parity_w120_spike") & (book["oat_axis"]=="baseline")
runnerup_mask = (book["a_key"]=="kalman_1e-06_1e-08_0.0001") & (book["b_key"]=="aggressive") & \
                (book["e_key"]=="cdar") & (book["f_key"]=="corr_risk_parity_w60_nospike") & (book["oat_axis"]=="baseline")

w = book[winner_mask].iloc[0]
ru = book[runnerup_mask].iloc[0]
print("\nWINNER full row across budgets/costs:")
print(w[["cagr","maxdd","sharpe","avg_leverage","dd_budget_20","dd_budget_30","dd_budget_41",
         "cagr_at_6.5bp","cagr_at_26bp","cagr_at_fund_plus5","cagr_at_fund_minus5","cagr_delta_vs_benchmark"]])
print("\nRUNNER-UP full row across budgets/costs:")
print(ru[["cagr","maxdd","sharpe","avg_leverage","dd_budget_20","dd_budget_30","dd_budget_41",
         "cagr_at_6.5bp","cagr_at_26bp","cagr_at_fund_plus5","cagr_at_fund_minus5","cagr_delta_vs_benchmark"]])
