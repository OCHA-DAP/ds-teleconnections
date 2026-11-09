"""Aggregate CHIRPS pixel onset/dry-spell metrics to departments; strong-El-Niño analog composites."""
import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]
exec(open(f"{SP}/common.py").read())
O=pd.read_parquet(f"{SP}/onset_chirps_pixels.parquet")
O["no_onset"]=O.onset.isna(); O["onset_c"]=O.onset.fillna(182)
W=pd.read_parquet(f"{SP}/weights_chirps_daily.parquet")[["ix","iy","ADM1_PCODE","w"]]
M=O.merge(W,on=["ix","iy"])
# per-pixel climatology (1991-2020) to define late onset / long dry spell at pixel level, then area share
clim=O[(O.y>=1991)&(O.y<=2020)].groupby(["ix","iy"]).agg(on_q67=("onset_c",lambda s: s.quantile(2/3)), on_med=("onset_c","median"),
      dry_q67=("max_dry",lambda s: s.quantile(2/3)), dry_med=("max_dry","median"))
M=M.join(clim,on=["ix","iy"])
M["late"]=(M.onset_c>M.on_q67).astype(float); M["longdry"]=(M.max_dry>M.dry_q67).astype(float)
M["on_anom"]=M.onset_c-M.on_med
def wavg(g,c): return np.average(g[c],weights=g.w)
A=M.groupby(["ADM1_PCODE","y"]).apply(lambda g: pd.Series({c:wavg(g,c) for c in ["onset_c","on_anom","max_dry","n_dry10","total","false_starts","late","longdry","no_onset"]})).reset_index()
A.to_parquet(f"{SP}/onset_chirps_adm1.parquet")
AN=[1983,1992,1998,2010,2016,2024]
pd.set_option("display.width",250)
def doy2date(d): return (pd.Timestamp("2027-01-01")+pd.Timedelta(days=int(round(d))-1)).strftime("%d %b")
rows=[]
for pc in PCS:
    a=A[A.ADM1_PCODE==pc].set_index("y")
    c=a.loc[1991:2020]
    # area-share-based "late onset year": >50% of dept area late
    late_yr=(a.late>0.5); dry_yr=(a.longdry>0.5)
    k_l=late_yr[AN].sum(); k_d=dry_yr[AN].sum()
    rows.append(dict(pcode=pc,name=NAMES.get(pc,pc),
        onset_clim=doy2date(c.onset_c.median()),
        onset_analogs=" ".join(f"{y%100:02d}:{doy2date(a.onset_c[y])}" for y in AN),
        on_anom_analog_mean=a.on_anom[AN].mean(),
        late_share_clim=late_yr.loc[1981:2025].mean(), late_in_analogs=f"{k_l}/6", P_late=(k_l+1)/(6+3),
        P_late_lo=stats.beta(k_l+1,6-k_l+2).ppf(.1), P_late_hi=stats.beta(k_l+1,6-k_l+2).ppf(.9),
        maxdry_clim=c.max_dry.median(), maxdry_analogs=" ".join(f"{a.max_dry[y]:.0f}" for y in AN),
        longdry_in_analogs=f"{k_d}/6", P_longdry=(k_d+1)/(6+3),
        fs_clim=c.false_starts.median(), fs_analog=a.false_starts[AN].mean(),
        late_2026=a.late.get(2026,np.nan), longdry_2026=a.longdry.get(2026,np.nan)))
R=pd.DataFrame(rows); R.to_csv(f"{SP}/onset_chirps_composites.csv",index=False)
print(R.round(2).to_string(index=False))
# national: year table for context
a=A[A.ADM1_PCODE=="HT"].set_index("y")
print("\nNational by year (late share, longdry share, onset anomaly d, max dry d):")
t=a[["late","longdry","on_anom","max_dry","false_starts","total"]].round(2)
t["DJF"]=D.reindex(t.index).round(2)
print(t.to_string())
# correlations onset metrics vs DJF (all years) and by DJF class
t2=t.dropna()
for c in ["on_anom","max_dry","late","longdry"]:
    print(c, "r vs DJF", round(stats.spearmanr(t2.DJF,t2[c])[0],2))
bins=pd.cut(t2.DJF,[-3,-0.5,0.5,1.0,3],labels=["La Niña","neutral","weak/mod El Niño","strong El Niño"])
print(t2.groupby(bins)[["late","longdry","on_anom","max_dry"]].agg(["mean","count"]).round(2))
