import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]
exec(open(f"{SP}/common.py").read())
nao=pd.read_csv(f"{SP}/idx/nao.txt", sep=r"\s+", header=None, names=["y","m","v"]).set_index(["y","m"])["v"]
TNA=read_psl(f"{SP}/idx/tna_latest.data")
def sm(s,y,pairs): return np.mean([s.get((yy,m),np.nan) for yy,m in [(y+dy,m) for dy,m in pairs]])
yrs=list(range(1982,2027))
P=pd.DataFrame(index=yrs)
P["n34_DJF"]=D[yrs]
P["n34_Oct"]=[N34.get((y-1,10)) for y in yrs]
P["nao_DJF"]=[sm(nao,y,[(-1,12),(0,1),(0,2)]) for y in yrs]
P["nao_JFM"]=[sm(nao,y,[(0,1),(0,2),(0,3)]) for y in yrs]
P["tna_Oct"]=[TNA.get((y-1,10)) for y in yrs]
P["tna_Sep"]=[TNA.get((y-1,9)) for y in yrs]
P["tna_FM"]=[sm(TNA,y,[(0,2),(0,3)]) for y in yrs]
P["tna_MAM"]=[sm(TNA,y,[(0,3),(0,4),(0,5)]) for y in yrs]
for sname,ms in {"MAMJ":[3,4,5,6],"MA":[3,4],"MJ":[5,6]}.items():
    S=season_tot(CH,ms).loc[yrs]; E=season_tot(ER,ms).reindex(yrs)
    print(f"\n{sname}: r with national CHIRPS / ERA5 (n=45; |r|>0.29 ~ p<0.05)")
    for c in P.columns:
        ok=P[c].notna()
        print(f"  {c:8s} CHIRPS {stats.pearsonr(P[c][ok],S['HT'][ok])[0]:+.2f}  ERA5 {stats.pearsonr(P[c][ok],E['HT'][ok])[0]:+.2f}")
print("\nTNA Sep->FM autocorr:", round(stats.pearsonr(P.tna_Sep.dropna(), P.tna_FM[P.tna_Sep.notna()])[0],2),
      " n34_DJF->tna_MAM:", round(stats.pearsonr(P.n34_DJF, P.tna_MAM)[0],2))
nino=[1983,1992,1998,2010,2016,2024]
print("\nPost-strong-El-Niño years: NAO DJF, TNA MAM, CHIRPS MAMJ % mean")
S=season_tot(CH,[3,4,5,6]).loc[yrs]["HT"]
for y in nino: print(y, f"NAO {P.nao_DJF[y]:+.2f}  TNA_MAM {P.tna_MAM[y]:+.2f}  MAMJ {S[y]/S.mean()*100:.0f}%")
print("2027 so far: Oct-2026 n34", N34.get((2026,10)), " TNA Sep 2026", TNA.get((2026,9)), " NAO Oct 2026", nao.get((2026,10)))
