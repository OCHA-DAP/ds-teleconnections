def read_psl(p):
    rows=[]
    for line in open(p):
        t=line.split()
        if len(t)==13 and t[0].isdigit():
            for m,v in enumerate(t[1:],1):
                v=float(v); rows.append((int(t[0]),m,np.nan if v<-90 else v))
    return pd.DataFrame(rows,columns=["y","m","v"]).set_index(["y","m"])["v"]
import os
N34=read_psl(os.environ.get("NINO34_FILE", "cache/live/nino34.data"))  # PSL ERSSTv6 Niño3.4, 1981–2010 base; path relative to the repo root
D=pd.Series({y: np.mean([N34.get((y-1,12)),N34.get((y,1)),N34.get((y,2))]) for y in range(1950,2027)})
CH=pd.read_parquet(f"{SP}/chirps_monthly_units.parquet"); CH["y"]=CH.date.dt.year; CH["m"]=CH.date.dt.month
ER=pd.read_parquet(f"{SP}/era5_hti_all.parquet"); ER=ER[ER.pcode.str.len()<=4].copy()
ER["y"]=ER.valid_date.dt.year; ER["m"]=ER.valid_date.dt.month; ER["mm"]=ER["mean"]*ER.valid_date.dt.days_in_month
IMd=pd.read_parquet(f"{SP}/imerg_hti.parquet"); IMd["y"]=IMd.valid_date.dt.year; IMd["m"]=IMd.valid_date.dt.month; IMd=IMd[IMd.y<2026]  # 2026 IMERG breaks from the earlier record (~2.4x CHIRPS vs 1.2-1.6x): left out
IM=IMd.groupby(["pcode","y","m"]).agg(mm=("mean","sum"),nd=("mean","size")).reset_index()
NAMES=CH.drop_duplicates("pcode").set_index("pcode")["name"]
PCS=["HT"]+[f"HT{i:02d}" for i in range(1,11)]
def season_tot(df, ms):
    g=df[df.m.isin(ms)].groupby(["pcode","y"]).agg(n=("mm","size"),mm=("mm","sum")).reset_index()
    return g[g.n==len(ms)].pivot(index="y",columns="pcode",values="mm")
