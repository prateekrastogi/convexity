"""Read research/cta/results_ctaA_full_grid.parquet and print the required representative
sample directly from the artifact: best 10 / worst 10 cells by CAGR-at-30%-DD-budget (on the
STAGE_A_22 book), and the SPY/QQQ single-instrument cells."""
import pandas as pd

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)

PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
print(f"TOTAL ROWS: {len(df)}")
print("universe counts:\n", df["universe"].value_counts())

cols_show = ["universe", "a_key", "b_key", "c_key", "d_key", "e_key", "f_key", "oat_axis",
             "cagr", "maxdd", "sharpe", "ev_trade_bp", "win_rate", "trades_per_yr", "worst_day",
             "ruin_flag", "avg_leverage", "cagr_delta_vs_benchmark", "dd_budget_30"]

book = df[df["universe"] == "STAGE_A_22"].copy()
qualifying = book[book["dd_budget_30"] == True].copy()  # noqa: E712
print(f"\nSTAGE_A_22 rows: {len(book)}, satisfying <=30% DD budget: {len(qualifying)}")

print("\n=== BEST 10 by CAGR-at-30%-DD-budget (STAGE_A_22) ===")
best10 = qualifying.sort_values("cagr", ascending=False).head(10)
print(best10[cols_show].to_string(index=False))

print("\n=== WORST 10 by CAGR-at-30%-DD-budget (STAGE_A_22, among DD-qualifying cells) ===")
worst10 = qualifying.sort_values("cagr", ascending=True).head(10)
print(worst10[cols_show].to_string(index=False))

print("\n=== WORST 10 by raw CAGR (STAGE_A_22, ALL cells regardless of DD budget) ===")
worst10_all = book.sort_values("cagr", ascending=True).head(10)
print(worst10_all[cols_show].to_string(index=False))

for u in ("SPY", "QQQ"):
    sub = df[df["universe"] == u].copy()
    print(f"\n=== {u}-alone cells: {len(sub)} rows ===")
    if len(sub):
        sub_dd = sub[sub["dd_budget_30"] == True]  # noqa: E712
        print(f"  satisfying <=30% DD budget: {len(sub_dd)}")
        top = (sub_dd if len(sub_dd) else sub).sort_values("cagr", ascending=False).head(5)
        print(top[cols_show].to_string(index=False))

print("\nDONE_REPORT")
