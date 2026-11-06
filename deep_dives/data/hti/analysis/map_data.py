"""Department metrics in rainfall terms (typical mm, expected 2027 range) + simplified geometry for the maps."""
import sys, json, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polylabel
from scipy import stats
SP=sys.argv[1]; exec(open(f"{SP}/common.py").read())
AN=[1983,1992,1998,2010,2016,2024]
L=pd.read_csv(f"{SP}/loo_results_2.7.csv"); L=L[L.season=="MAMJ"].set_index(["src","pcode"])
S5=pd.read_fwf(f"{SP}/seas5_dept_out.txt").set_index("pcode")
ON=pd.read_parquet(f"{SP}/onset_chirps_adm1.parquet")
SYN=pd.read_csv(f"{SP}/dept_synthesis.csv").set_index("pcode")
chS=season_tot(CH,[3,4,5,6]); erS=season_tot(ER,[3,4,5,6]); erMAM=season_tot(ER,[3,4,5])
imS=season_tot(IM,[3,4,5,6])
out={}
for pc in PCS:
    c=chS[pc].loc[1981:2026].dropna(); e=erS[pc].loc[1981:2026].dropna()
    cm=c.loc[1991:2020].mean(); em=e.loc[1991:2020].mean()
    ch_an=[round(float(c[y]/cm*100)) for y in AN]; er_an=[round(float(e[y]/em*100)) for y in AN]
    lc=L.loc[("CHIRPS",pc)]; le=L.loc[("ERA5",pc)]
    # SEAS5 MAM predictive (conditional normal in log space, ERA5 MAM 1981-2026)
    y=np.log1p(erMAM[pc].loc[1981:2026].dropna()); mu,s=y.mean(),y.std(ddof=1); clim=np.expm1(y).loc[1991:2020].mean() if hasattr(y,'loc') else np.expm1(y).mean()
    r=float(S5.loc[pc,"r"]); z=stats.norm.ppf(min(float(S5.loc[pc,"pct"]),97.8)/100)
    m=mu+r*z*s; sd=s*np.sqrt(1-r*r)
    s5=[float(np.expm1(m+k*sd)/clim*100) for k in (0,-1.2816,1.2816)]
    meds=[np.median(ch_an),np.median(er_an),lc.med_pct,le.med_pct,s5[0]]
    lo=np.mean([lc.p10_pct,le.p10_pct,s5[1]]); hi=np.mean([lc.p90_pct,le.p90_pct,s5[2]])
    o=ON[ON.ADM1_PCODE==pc].set_index("y"); ob=o.loc[:2025]
    shift=(o.on_anom[AN]-ob.on_anom.mean()); se=ob.on_anom.std()/np.sqrt(6)
    d=dict(name=NAMES[pc], typ_mm=round(float(cm)), typ_p10=round(float(c.quantile(.1))), typ_p90=round(float(c.quantile(.9))),
        typ_p10_pct=round(float(c.quantile(.1)/cm*100)), typ_p90_pct=round(float(c.quantile(.9)/cm*100)),
        ch_an=ch_an, er_an=er_an,
        ch_reg=[round(lc.med_pct),round(lc.p10_pct),round(lc.p90_pct)], er_reg=[round(le.med_pct),round(le.p10_pct),round(le.p90_pct)],
        s5=[round(v) for v in s5], s5_pct=round(float(S5.loc[pc,"pct"])),
        central=round(float(np.mean(meds))), lo=round(float(lo)), hi=round(float(hi)),
        r_ch=round(float(lc.r),2), r_er=round(float(le.r),2),
        shift=round(float(shift.mean())), shift_se=round(float(se)),
        late_an=int((o.late[AN]>=0.5).sum()), late_base=round(float((ob.late>=0.5).mean()*100)),
        longdry_an=round(float(o.longdry[AN].mean()*100)), longdry_base=round(float(ob.longdry.mean()*100)),
        fs_an=round(float(o.false_starts[AN].mean()),1), fs_base=round(float(ob.false_starts.mean()),1),
        late26=round(float(o.late.get(2026))*100), longdry26=round(float(o.longdry.get(2026))*100))
    d["central_mm"]=round(d["typ_mm"]*d["central"]/100); d["lo_mm"]=round(d["typ_mm"]*d["lo"]/100); d["hi_mm"]=round(d["typ_mm"]*d["hi"]/100)
    out[pc]=d
# national extras
im=imS["HT"].loc[2001:2025]; imm=im.loc[2001:2020].mean()
out["HT"]["im_an"]=[round(float(im[y]/imm*100)) for y in [2010,2016,2024]]
out["HT"]["nmme"]=[125,91,172]
# agreement classes
cls={"HT03":"both","HT04":"both","HT09":"both","HT05":"both","HT06":"both","HT01":"chirps","HT02":"chirps","HT07":"unc","HT08":"unc","HT10":"unc"}
for k,v in cls.items(): out[k]["cls"]=v
# geometry
g=gpd.read_file(f"{SP}/bounds/hti_adm1.gpkg")
lon0,lat1=-74.55,20.15; k=np.cos(np.deg2rad(19.0)); S=1000/(2.95*k)
def P(x,y): return ((x-lon0)*k*S, (lat1-y)*S)
geo={}
for _,row in g.iterrows():
    geom=row.geometry.simplify(0.003,preserve_topology=True)
    polys=list(geom.geoms) if isinstance(geom,MultiPolygon) else [geom]
    polys=[p for p in polys if p.area>0.0015]
    big=max(polys,key=lambda p:p.area); lp=polylabel(big,0.002)
    dstr=""
    for p in polys:
        pts=[P(x,y) for x,y in p.exterior.coords]
        dstr+="M"+"L".join(f"{x:.1f},{y:.1f}" for x,y in pts)+"Z"
    geo[row.ADM1_PCODE]=dict(d=dstr,lx=round(P(lp.x,lp.y)[0],1),ly=round(P(lp.x,lp.y)[1],1))
W=1000; H=round((20.15-17.95)*S)
json.dump(dict(m=out,geo=geo,W=W,H=H),open(f"{SP}/map_data.json","w"))
pd.set_option("display.width",250)
print(pd.DataFrame(out).T[["name","typ_mm","typ_p10","typ_p90","central","lo","hi","central_mm","lo_mm","hi_mm","ch_an","er_an","ch_reg","er_reg","s5","cls"]].to_string())
print(out["HT"]["im_an"], "H", H, "bytes", len(json.dumps(geo)))
