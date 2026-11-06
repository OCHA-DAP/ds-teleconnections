import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from src.skill import run_all_combinations
SP=sys.argv[1]
s=pd.read_parquet(f"{SP}/seas5_hti_all.parquet"); e=pd.read_parquet(f"{SP}/era5_hti_all.parquet")
names={"HT":"Haïti","HT01":"Ouest","HT02":"Sud-Est","HT03":"Nord","HT04":"Nord-Est","HT05":"Artibonite","HT06":"Centre","HT07":"Sud","HT08":"Grande'Anse","HT09":"Nord-Ouest","HT10":"Nippes"}
rows=[]
for pc,nm in names.items():
    ss=s[s.pcode==pc].dropna(subset=["mean"]); ee=e[e.pcode==pc].dropna(subset=["mean"])
    sk,_=run_all_combinations(pc,"HTI",nm,ss,ee,detrend=True)
    r=sk[(sk.issued_month==11)&(sk.trimester=="MAM")].iloc[0]
    rows.append(dict(pcode=pc,name=nm,r=r.pearson_r,pct=r.forecast_percentile,P_below=r.prob_lower_tercile,
        r_feb=sk[(sk.issued_month==2)&(sk.trimester=="MAM")].pearson_r.iloc[0], r_mar=sk[(sk.issued_month==3)&(sk.trimester=="MAM")].pearson_r.iloc[0]))
print(pd.DataFrame(rows).round(2).to_string(index=False))
