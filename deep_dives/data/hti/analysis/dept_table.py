import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]; exec(open(f"{SP}/common.py").read())
L=pd.read_csv(f"{SP}/loo_results_2.7.csv"); L=L[L.season=="MAMJ"]
S5=pd.read_fwf(f"{SP}/seas5_dept_out.txt")
A=pd.read_parquet(f"{SP}/onset_chirps_adm1.parquet"); AN=[1983,1992,1998,2010,2016,2024]
rows=[]
for pc in PCS:
    c=L[(L.src=="CHIRPS")&(L.pcode==pc)].iloc[0]; e=L[(L.src=="ERA5")&(L.pcode==pc)].iloc[0]; i=L[(L.src=="IMERG")&(L.pcode==pc)].iloc[0]
    s=S5[S5.pcode==pc].iloc[0]; a=A[A.ADM1_PCODE==pc].set_index("y")
    # pct of normal in analogs CHIRPS / ERA5
    rows.append(dict(pcode=pc,name=NAMES[pc],
      ch_below=f"{int(c.below_in_analog)}/6", er_below=f"{int(e.below_in_analog)}/6", im_below=f"{int(i.below_in_analog)}/{int(i.n_analog)}",
      r_ch=c.r, r_er=e.r, P_lin_ch=c.P_below_lin, P_lin_er=e.P_below_lin,
      P_an_ch=c.P_below_analog, P_an_er=e.P_below_analog,
      s5_pct=s.pct, s5_P=s.P_below, s5_r=s.r,
      late_maj_an=int((a.late[AN]>=0.5).sum()), late_maj_base=(a.late.loc[:2025]>=0.5).mean(),
      longdry_an=a.longdry[AN].mean(), longdry_base=a.longdry.loc[:2025].mean(),
      fs_an=a.false_starts[AN].mean(), fs_base=a.false_starts.loc[:2025].mean()))
R=pd.DataFrame(rows)
R["P_consensus"]=R[["P_an_ch","P_an_er","P_lin_ch","P_lin_er","s5_P"]].mean(axis=1)
R.to_csv(f"{SP}/dept_synthesis.csv",index=False)
pd.set_option("display.width",250); print(R.round(2).to_string(index=False))
# percent of normal in each analog year, CHIRPS & ERA5, by dept
for nm,X in [("CHIRPS",CH),("ERA5",ER)]:
    St=season_tot(X,[3,4,5,6]); clim=St.loc[1991:2020].mean()
    print(f"\n{nm} MAMJ % of 1991-2020 mean in analog years"); print((St.loc[AN]/clim*100).round(0).astype(int).rename(columns=NAMES).to_string())
# nonlinearity: national CHIRPS / ERA5 MAMJ tercile by DJF class
for nm,X in [("CHIRPS",CH),("ERA5",ER)]:
    St=season_tot(X,[3,4,5,6])["HT"].loc[1982:2026]; lo,hi=St.quantile([1/3,2/3])
    df=pd.DataFrame({"tot":St,"DJF":D.reindex(St.index)})
    df["cls"]=pd.cut(df.DJF,[-3,-0.5,0.5,1.0,5],labels=["La Niña ≤-0.5","neutral","weak/mod EN 0.5–1.0","strong EN ≥1.0"])
    g=df.groupby("cls",observed=True).agg(n=("tot","size"),below=("tot",lambda s:(s<lo).sum()),above=("tot",lambda s:(s>hi).sum()),pct=("tot",lambda s:(s/St.loc[1991:2020].mean()*100).mean()))
    print(f"\n{nm} national MAMJ by DJF class (lower/upper tercile 1982-2026)"); print(g.round(0).to_string())
    print(df[df.cls=="weak/mod EN 0.5–1.0"].assign(pct=lambda d:(d.tot/St.loc[1991:2020].mean()*100).round(0)).pct.to_dict())
