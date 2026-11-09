import sys, numpy as np, pandas as pd
SP=sys.argv[1]
exec(open(f"{SP}/common.py").read())
for src,X in [("CHIRPS",CH),("ERA5",ER)]:
    X=X[X.pcode.isin(PCS)]
    clim=X[(X.y>=1991)&(X.y<=2020)].groupby(["pcode","m"])["mm"].mean()
    X=X.assign(pct=X.apply(lambda r: r.mm/clim[(r.pcode,r.m)]*100, axis=1))
    print(f"\n{src} national % of 1991-2020 monthly mean (Oct of year-1 .. Jul):")
    for y in [1983,1992,1998,2003,2010,2015,2016,2024]:
        s=X[(X.pcode=="HT")&(((X.y==y-1)&(X.m>=10))|((X.y==y)&(X.m<=7)))].sort_values(["y","m"])
        print(y, " ".join(f"{int(v):4d}" for v in s.pct.values), "  (O N D J F M A M J J)")
    if src=="CHIRPS":
        print("\nCHIRPS 2016 and 1992 by department, Mar-Jun % of normal:")
        for y in [1992,2016]:
            s=X[(X.y==y)&(X.m.between(3,6))].pivot(index="pcode",columns="m",values="pct").round(0).rename(index=NAMES)
            print(y); print(s.to_string())
