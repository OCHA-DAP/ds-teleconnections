import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]
exec(open(f"{SP}/common.py").read())
ms=[3,4,5,6]
C=season_tot(CH, ms); E=season_tot(ER, ms); I=season_tot(IM, ms)
yrs=[y for y in range(2001,2027)]
print("MAMJ totals correlation 2001-2026 (IMERG from 2001 to avoid early TRMM-era gaps):")
rows=[]
for pc in PCS:
    c,e,i=C.loc[yrs,pc],E.loc[yrs,pc],I.loc[yrs,pc]
    rows.append(dict(pcode=pc,name=NAMES[pc],ch_era=stats.pearsonr(c,e)[0],ch_imerg=stats.pearsonr(c,i)[0],era_imerg=stats.pearsonr(e,i)[0],
       mean_ch=c.mean(),mean_imerg=i.mean(),mean_era=e.mean()))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nNational MAMJ % of own mean, El Niño-after years (DJF>=1.0) and moderate (0.5-1.0):")
for y in sorted([y for y in range(1982,2027) if D[y]>=0.5]):
    v=[]
    for X in (C,E,I):
        s=X["HT"]; s=s[s.index>=(2001 if X is I else 1982)]
        v.append(f"{(s.get(y,np.nan)/s.mean()*100):.0f}" if y in s.index else "  -")
    print(y, f"DJF {D[y]:+.2f}", "CHIRPS",v[0],"ERA5",v[1],"IMERG",v[2])
print("\nYear-0 (El Niño developing: DJF<0.5 then following DJF>=1.0):")
for y in [y for y in range(1982,2026) if D.get(y+1,np.nan)>=1.0]:
    print(y, f"DJF {D[y]:+.2f} -> next {D[y+1]:+.2f}", "CHIRPS MAMJ", f"{C.loc[y,'HT']/C['HT'].mean()*100:.0f}%", "ERA5", f"{E.loc[y,'HT']/E['HT'].mean()*100:.0f}%")
