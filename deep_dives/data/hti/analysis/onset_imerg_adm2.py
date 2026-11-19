import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]
exec(open(f"{SP}/common.py").read()); exec(open(f"{SP}/onset_lib.py").read())
d=pd.read_parquet(f"{SP}/imerg_hti_adm2_fj.parquet")
d["y"]=d.valid_date.dt.year; d["doy"]=d.valid_date.dt.dayofyear-((d.valid_date.dt.is_leap_year)&(d.valid_date.dt.month>2)).astype(int)
rows=[]
for (pc,y),g in d.groupby(["pcode","y"]):
    g=g.sort_values("valid_date"); r=g["mean"].fillna(0).values
    rows.append(dict(pcode=pc,y=y,**onset_dryspell(r,g["doy"].values)))
O=pd.DataFrame(rows); O["adm1"]=O.pcode.str[:4]
O["onset_c"]=O.onset.fillna(182)  # censored: no onset by end of June
O.to_parquet(f"{SP}/onset_imerg_adm2.parquet")
def agg(g): return pd.Series(dict(onset=g.onset_c.median(), late_share=np.nan, max_dry=g.max_dry.mean(), n_dry10=g.n_dry10.mean(), total=g.total.mean(), no_onset=(g.onset.isna()).mean()))
A=pd.concat([O.groupby(["adm1","y"]).apply(agg).reset_index(), O.groupby("y").apply(agg).reset_index().assign(adm1="HT")])
# late onset share: commune onset later than its own 2/3 quantile
q=O.groupby("pcode").onset_c.quantile(2/3); O["late"]=O.onset_c>O.pcode.map(q)
ls=pd.concat([O.groupby(["adm1","y"]).late.mean().reset_index(), O.groupby("y").late.mean().reset_index().assign(adm1="HT")])
A=A.drop(columns="late_share").merge(ls.rename(columns={"late":"late_share"}),on=["adm1","y"])
A.to_parquet(f"{SP}/onset_imerg_adm1.parquet")
nat=A[A.adm1=="HT"].set_index("y")
def doy2(d): return (pd.Timestamp("2001-01-01")+pd.Timedelta(days=int(d)-1)).strftime("%d %b")
print("National (median commune onset; IMERG 1998-2026):")
print(nat[["onset","late_share","no_onset","max_dry","n_dry10","total"]].assign(onset=lambda x: x.onset.map(doy2)).round(2).to_string())
sub=nat[nat.index>=1998]
for c in ["onset","late_share","max_dry","n_dry10","total"]:
    x=D[sub.index]; print(f"r(DJF n34, {c}) = {stats.pearsonr(x, sub[c])[0]:+.2f}", end="; ")
print()
nino=[1998,2010,2016,2024]; dev=[2002,2009,2015,2023]
for lab,ys in [("post-strong-El-Niño",nino),("El Niño developing",dev)]:
    s=sub.loc[ys]; print(lab, "onset", [doy2(v) for v in s.onset], "late_share", s.late_share.round(2).tolist(), "max_dry", s.max_dry.round(1).tolist(), "vs clim median onset", doy2(sub.onset.median()), "clim max_dry", round(sub.max_dry.median(),1))
# by department: post-strong years late-onset share & rank of max_dry
print("\nBy department, post-strong-El-Niño years: late_share (share of communes with onset in their latest third)")
print(A[A.y.isin(nino)].pivot(index="adm1",columns="y",values="late_share").rename(index=NAMES).round(2).to_string())
print("\nBy department, max dry spell percentile rank (higher = longer dry spell) in post-strong years:")
A["md_rank"]=A.groupby("adm1").max_dry.rank(pct=True)
print(A[A.y.isin(nino)].pivot(index="adm1",columns="y",values="md_rank").rename(index=NAMES).round(2).to_string())
