import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 260)
PATH = "/home/speculation/convexity-cta/research/cta/results_ctaA_full_grid.parquet"
df = pd.read_parquet(PATH)
book = df[df["universe"] == "STAGE_A_22"].copy()

for ek in book["e_key"].unique():
    sub = book[book["e_key"] == ek]
    lev_ok = sub[(sub["avg_leverage"]>=2.0)&(sub["avg_leverage"]<=5.0)]
    print(f"e_key={ek:15s} n={len(sub):5d} avg_lev[min,med,max]=({sub['avg_leverage'].min():.2f},{sub['avg_leverage'].median():.2f},{sub['avg_leverage'].max():.2f}) "
          f"lev_in_2_5={len(lev_ok):4d} | maxdd_of_those[min,med]=({lev_ok['maxdd'].min() if len(lev_ok) else np.nan:.3f},{lev_ok['maxdd'].median() if len(lev_ok) else np.nan:.3f})")

print("\nOverall: any cell with avg_leverage in [2,5] AND maxdd<=0.41 (any e_key)?")
cond = (book["avg_leverage"]>=2.0)&(book["avg_leverage"]<=5.0)&(book["maxdd"]<=0.41)
print(book[cond].shape[0])

cond20 = (book["avg_leverage"]>=2.0)&(book["avg_leverage"]<=5.0)&(book["maxdd"]<=0.20)
cond30 = (book["avg_leverage"]>=2.0)&(book["avg_leverage"]<=5.0)&(book["maxdd"]<=0.30)
print("DD<=20%:", book[cond20].shape[0], " DD<=30%:", book[cond30].shape[0], " DD<=41%:", book[cond].shape[0])

# also check SPY, QQQ universes
for u in ("SPY","QQQ"):
    sub = df[df["universe"]==u]
    for budget in (0.20,0.30,0.41):
        cond = (sub["avg_leverage"]>=2.0)&(sub["avg_leverage"]<=5.0)&(sub["maxdd"]<=budget)
        print(u, budget, sub[cond].shape[0])
