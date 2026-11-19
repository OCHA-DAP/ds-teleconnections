"""LOO cross-validated ENSO-conditioned forecasts of spring rainfall, Haiti national + departments."""
import sys, numpy as np, pandas as pd
from scipy import stats
SP=sys.argv[1]; DJF_MU=float(sys.argv[2]); DJF_SD=float(sys.argv[3])
exec(open(f"{SP}/common.py").read())
rng=np.random.default_rng(0)
def models(x, y, x0):
    """x predictor (DJF n34), y log1p rain. Returns P(below lower tercile) at x0 for: linear-gaussian, analog-bayes, logistic."""
    lo=np.quantile(y,1/3)
    b=stats.linregress(x,y); res=y-(b.intercept+b.slope*x); s=res.std(ddof=2)
    p_lin=stats.norm.cdf((lo-(b.intercept+b.slope*x0))/s)
    ev=(y<lo).astype(float)
    # analog: strong El Niño winters (>=1.0) if x0>=1 else climatology; Beta prior k=3 pseudo-years at 1/3
    if x0>=1.0:
        m=x>=1.0; p_an=(ev[m].sum()+1)/(m.sum()+3)
    else: p_an=1/3
    return p_lin, p_an
rows=[]; preds={}
for src,X,y0 in [("CHIRPS",CH,1982),("ERA5",ER,1982),("IMERG",IM,2001)]:
    for sname,ms in {"MAMJ":[3,4,5,6],"MAM":[3,4,5],"AMJ":[4,5,6]}.items():
        S=season_tot(X,ms); S=S[S.index>=y0]
        for pc in PCS:
            y=np.log1p(S[pc].dropna()); yrs=y.index.values; x=D[yrs].values; yv=y.values
            P=np.zeros((len(yrs),2)); ev=np.zeros(len(yrs))
            for i in range(len(yrs)):
                k=np.arange(len(yrs))!=i
                P[i]=models(x[k],yv[k],x[i]); ev[i]=yv[i]<np.quantile(yv[k],1/3)
            bs_clim=np.mean((1/3-ev)**2)
            bss=[1-np.mean((P[:,j]-ev)**2)/bs_clim for j in range(2)]
            # strong-El-Niño subset hit stats
            m=x>=1.0
            # 2027 forecast with DJF uncertainty
            xs=rng.normal(DJF_MU,DJF_SD,4000)
            b=stats.linregress(x,yv); s=(yv-(b.intercept+b.slope*x)).std(ddof=2)
            draws=b.intercept+b.slope*xs+rng.normal(0,s,4000)
            # parameter uncertainty via bootstrap of the regression
            pb=[]
            for _ in range(500):
                ii=rng.integers(0,len(yv),len(yv)); bb=stats.linregress(x[ii],yv[ii]); ss=(yv[ii]-(bb.intercept+bb.slope*x[ii])).std(ddof=2)
                pb.append(np.mean(stats.norm.cdf((np.quantile(yv,1/3)-(bb.intercept+bb.slope*xs[:400]))/ss)))
            lo_t=np.quantile(yv,1/3); hi_t=np.quantile(yv,2/3)
            clim_mean=np.expm1(yv).mean()
            ev_all=yv<lo_t
            p_an=(ev_all[m].sum()+1)/(m.sum()+3)
            beta=stats.beta(ev_all[m].sum()+1, (m.sum()-ev_all[m].sum())+2)
            rows.append(dict(src=src,season=sname,pcode=pc,name=NAMES[pc],n=len(yv),r=stats.pearsonr(x,yv)[0],
                BSS_lin=bss[0],BSS_analog=bss[1],
                P_below_lin=np.mean(draws<lo_t),P_below_lin_lo=np.quantile(pb,0.1),P_below_lin_hi=np.quantile(pb,0.9),
                P_above_lin=np.mean(draws>hi_t),
                P_below_analog=p_an,P_below_analog_lo=beta.ppf(0.1),P_below_analog_hi=beta.ppf(0.9),
                n_analog=int(m.sum()),below_in_analog=int(ev_all[m].sum()),
                med_pct=np.median(np.expm1(draws))/clim_mean*100,
                p10_pct=np.quantile(np.expm1(draws),0.1)/clim_mean*100,p90_pct=np.quantile(np.expm1(draws),0.9)/clim_mean*100))
R=pd.DataFrame(rows); R.to_csv(f"{SP}/loo_results.csv",index=False)
pd.set_option("display.width",250)
for src in ["CHIRPS","ERA5","IMERG"]:
    print(f"\n===== {src} MAMJ (DJF ~ N({DJF_MU},{DJF_SD})) =====")
    print(R[(R.src==src)&(R.season=="MAMJ")].drop(columns=["src","season"]).round(2).to_string(index=False))
print("\nNational, all seasons:"); print(R[R.pcode=="HT"][["src","season","r","BSS_lin","BSS_analog","P_below_lin","P_below_analog","below_in_analog","n_analog","med_pct","p10_pct","p90_pct"]].round(2).to_string(index=False))
