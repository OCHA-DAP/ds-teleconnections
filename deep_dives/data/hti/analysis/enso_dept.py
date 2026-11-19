import sys, numpy as np, pandas as pd
from scipy import stats
SP = sys.argv[1]
def read_psl(p):
    rows=[]
    for line in open(p):
        t=line.split()
        if len(t)==13 and t[0].isdigit():
            for m,v in enumerate(t[1:],1):
                v=float(v); rows.append((int(t[0]),m,np.nan if v<-90 else v))
    return pd.DataFrame(rows,columns=["y","m","v"]).set_index(["y","m"])["v"]
import os
n34 = read_psl(os.environ.get("NINO34_FILE", "cache/live/nino34.data"))
tna = read_psl(os.environ.get("TNA_FILE", "cache/live/tna.data"))
def djf(y): return np.mean([n34.get((y-1,12)), n34.get((y,1)), n34.get((y,2))])
def tna_s(y, ms): return np.mean([tna.get((y,m)) for m in ms])
ch = pd.read_parquet(f"{SP}/chirps_monthly_units.parquet"); ch["y"]=ch.date.dt.year; ch["m"]=ch.date.dt.month
er = pd.read_parquet(f"{SP}/era5_hti_all.parquet"); er=er[er.pcode.str.len()<=4].copy()
er["y"]=er.valid_date.dt.year; er["m"]=er.valid_date.dt.month; er["mm"]=er["mean"]*er.valid_date.dt.days_in_month
names = ch.drop_duplicates("pcode").set_index("pcode")["name"]
SEAS = {"MAMJ":[3,4,5,6], "MAM":[3,4,5], "AMJ":[4,5,6], "MJ":[5,6], "MA":[3,4]}
def season_tot(df, ms):
    g = df[df.m.isin(ms)].groupby(["pcode","y"]).agg(n=("mm","size"), mm=("mm","sum")).reset_index()
    return g[g.n==len(ms)].pivot(index="y", columns="pcode", values="mm")
yrs = list(range(1982, 2027))
D = pd.Series({y: djf(y) for y in yrs}); T = pd.Series({y: tna_s(y,[3,4,5]) for y in yrs})
nino_yrs = [y for y in yrs if D[y] >= 1.0]; nina_yrs=[y for y in yrs if D[y] <= -1.0]
print("El Niño DJF>=1:", nino_yrs, " La Niña DJF<=-1:", nina_yrs)
print("r(DJF n34, MAM TNA) =", round(stats.pearsonr(D, T)[0],2))
out=[]
for sname, ms in SEAS.items():
    C = season_tot(ch, ms).loc[yrs]; E = season_tot(er, ms).reindex(yrs)
    for pc in ["HT","HT01","HT02","HT03","HT04","HT05","HT06","HT07","HT08","HT09","HT10"]:
        c = C[pc]; e = E[pc] if pc in E else None
        lo = c.quantile(1/3); hi = c.quantile(2/3)
        comp = c[nino_yrs]
        row = dict(season=sname, pcode=pc, name=names[pc],
            r_ch_era=stats.pearsonr(c, e)[0] if e is not None else np.nan,
            r_n34_ch=stats.pearsonr(D, c)[0], p_n34_ch=stats.pearsonr(D, c)[1],
            r_n34_era=stats.pearsonr(D, e)[0] if e is not None else np.nan,
            r_tna_ch=stats.pearsonr(T, c)[0],
            nino_pct=(comp/c.mean()*100).mean(), nino_below=int((comp<lo).sum()), nino_above=int((comp>hi).sum()),
            nina_pct=(c[nina_yrs]/c.mean()*100).mean())
        out.append(row)
R = pd.DataFrame(out); R.to_csv(f"{SP}/enso_dept_stats.csv", index=False)
pd.set_option("display.width", 200)
for s in SEAS: print(f"\n== {s} =="); print(R[R.season==s].drop(columns="season").round(2).to_string(index=False))
# per-year national + dept % of normal in El Niño years (MAMJ, CHIRPS)
C = season_tot(ch, SEAS["MAMJ"]).loc[yrs]
print("\nCHIRPS MAMJ % of 1982-2026 mean in El Niño years:"); print((C.loc[nino_yrs]/C.mean()*100).round(0).astype(int).rename(columns=names).to_string())
print("DJF n34:", D[nino_yrs].round(2).to_dict())
