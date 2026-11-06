"""Skill, 2027 forecast percentiles, skill-calibrated P(below-normal) and Nino3.4 plume.

Usage: uv run python -I 03_analysis.py <nmme_agent_dir>
Reads out/nmme_prec_members.parquet, out/obs_monthly.csv, out/nino34_raw.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(sys.argv[1])
OUT = D / "out"
pr = pd.read_parquet(OUT / "nmme_prec_members.parquet")
obs = pd.read_csv(OUT / "obs_monthly.csv")

SEASONS = {"MAM": [3, 4, 5], "AMJ": [4, 5, 6], "MAMJ": [3, 4, 5, 6]}
FC_YEAR = 2027
COMMON = (1991, 2026)  # target years; Nov starts 1990..2025
LIVE = ["CanESM5", "GEM5.2-NEMO", "CESM1", "CCSM4", "CFSv2"]  # Nov 2026 start posted
SKILL_ONLY = ["GEOS-S2S", "SPEAR"]
SENS = ["CFSv2early"]  # CFSv2 with EARLY_MONTH_SAMPLES realtime (5 members for Nov 2026)
Z80 = stats.norm.ppf(0.9)

# ---------- seasonal totals ----------
def member_seasons(df):
    out = []
    for s, months in SEASONS.items():
        sub = df[df.target_month.isin(months)]
        g = sub.groupby(["model", "seg", "method", "start_year", "member"])["prec_mm"]
        tot = g.sum(min_count=len(months))
        cnt = g.count()
        tot = tot[cnt == len(months)]  # all months present
        out.append(tot.rename("tot").reset_index().assign(season=s))
    m = pd.concat(out)
    m["target_year"] = m["start_year"] + 1
    return m.dropna(subset=["tot"])


mem = member_seasons(pr)
ens = (mem.groupby(["model", "method", "season", "target_year"])
       .agg(ensmean=("tot", "mean"), n_mem=("tot", "size"), seg=("seg", "first")).reset_index())

obs_s = []
for s, months in SEASONS.items():
    sub = obs[obs.month.isin(months)]
    g = sub.groupby("year")
    o = g[["chirps_mm", "era5_mm"]].sum(min_count=len(months))
    o[g[["chirps_mm", "era5_mm"]].count() < len(months)] = np.nan
    obs_s.append(o.reset_index().rename(columns={"year": "target_year"}).assign(season=s))
obs_s = pd.concat(obs_s)
obs_s.to_csv(OUT / "obs_seasonal_totals.csv", index=False)


# ---------- conditional-normal calibration ----------
TF = {"log1p": (np.log1p, np.expm1), "raw": (lambda v: v, lambda v: v)}
q3 = lambda v: np.quantile(v, 1 / 3)


def calib_fit(X, y):
    """X: (n, k) transformed ens-mean per model; y: (n,) transformed obs. Returns params dict."""
    mx, sx = X.mean(0), X.std(0, ddof=1)
    z = ((X - mx) / sx).mean(1)  # equal weight per model on standardised anomalies
    mz, sz = z.mean(), z.std(ddof=1)
    zs = (z - mz) / sz
    mu, sig = y.mean(), y.std(ddof=1)
    r = np.corrcoef(zs, y)[0, 1]
    r_use = max(r, 0.0)  # no negative skill: floor to climatology
    E = mu + r_use * sig * zs  # = (1-r) mu + r F', F' = forecast standardised to obs mean/sd
    res_sd = np.sqrt(((y - E) ** 2).sum() / (len(y) - 2)) if r_use > 0 else sig
    return dict(mx=mx, sx=sx, mz=mz, sz=sz, mu=mu, sig=sig, r=r, r_use=r_use, res_sd=res_sd)


def calib_predict(p, X0):
    z = ((X0 - p["mx"]) / p["sx"]).mean(-1)
    zs = (z - p["mz"]) / p["sz"]
    E = p["mu"] + p["r_use"] * p["sig"] * zs
    return E, p["res_sd"], zs


def roc_auc(score, event):
    event = np.asarray(event, bool)
    n1, n0 = event.sum(), (~event).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = stats.rankdata(score)
    return (rk[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(Xr, yr, years, X0r=None, T_years=None, tf="log1p"):
    """Skill + LOO checks + 2027 forecast for one model (k=1) or the MME (k>1). Inputs in mm."""
    f, finv = TF[tf]
    X, y = f(Xr), f(yr)
    n = len(y)
    E_loo, P_loo, Pc_loo = np.zeros(n), np.zeros(n), np.zeros(n)
    ev_loo, flag = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        tr = np.arange(n) != i
        p = calib_fit(X[tr], y[tr])
        E, s, zs_i = calib_predict(p, X[i])
        T = q3(y[tr])
        E_loo[i] = E
        P_loo[i] = stats.norm.cdf((T - E) / s)
        Pc_loo[i] = stats.norm.cdf((T - p["mu"]) / p["sig"])  # same-fold climatological reference
        ev_loo[i] = y[i] < T
        _, _, zs_tr = calib_predict(p, X[tr])
        flag[i] = zs_i < q3(zs_tr)  # deterministic: ens-mean index in its own lower tercile
    p = calib_fit(X, y)
    zs_all = calib_predict(p, X)[2]
    ev_full = y < q3(y)
    res = dict(
        n_years=n, yr0=int(years.min()), yr1=int(years.max()),
        r_tf=p["r"], r_used=p["r_use"], loo_r=np.corrcoef(E_loo, y)[0, 1],
        auc_ensmean=roc_auc(-zs_all, ev_full),  # rank-based, no fitting involved
        hit_rate=(flag & ev_loo).sum() / max(ev_loo.sum(), 1),
        false_alarm_ratio=(flag & ~ev_loo).sum() / max(flag.sum(), 1),
        n_flag=int(flag.sum()), n_event=int(ev_loo.sum()),
        bss_loo=1 - np.mean((P_loo - ev_loo) ** 2) / np.mean((Pc_loo - ev_loo) ** 2),
        res_sd=p["res_sd"], obs_sd_tf=p["sig"],
    )
    if X0r is not None:
        E, s, zs0 = calib_predict(p, f(X0r))
        E, zs0 = float(np.squeeze(E)), float(np.squeeze(zs0))
        T = q3(y)
        Tn = p["mu"] + p["sig"] * stats.norm.ppf(1 / 3)
        yo = finv(y)
        zc = float(np.clip(zs0, zs_all.min(), zs_all.max()))  # cap at hindcast range (no extrapolation)
        Ec = p["mu"] + p["r_use"] * p["sig"] * zc
        res.update(
            z_2027=zs0, z_hc_max=float(zs_all.max()), z_hc_min=float(zs_all.min()),
            P_below_zcap=stats.norm.cdf((T - Ec) / s),
            P_below=stats.norm.cdf((T - E) / s),
            P_clim_ref=stats.norm.cdf((T - p["mu"]) / p["sig"]),
            P_below_normT=stats.norm.cdf((Tn - E) / s),
            E_mm=float(finv(E)), PI10_mm=float(finv(E - Z80 * s)), PI90_mm=float(finv(E + Z80 * s)),
            obs_mean_mm=yo.mean(), obs_clim10_mm=np.quantile(yo, 0.1), obs_clim90_mm=np.quantile(yo, 0.9),
            obs_lt_mm=float(finv(T)),
        )
        if T_years is not None:  # sensitivity: tercile from a sub-period (e.g. 1991-2020)
            T2 = q3(y[(years >= T_years[0]) & (years <= T_years[1])])
            res["P_below_T9120"] = stats.norm.cdf((T2 - E) / s)
            res["obs_lt9120_mm"] = float(finv(T2))
    return res


def member_loo_frac(mm, years):
    """Per year: fraction of that year's members below the lower tercile of all OTHER years' members."""
    out = {}
    for yv in years:
        others = mm[(mm.target_year != yv) & mm.target_year.isin(years)].tot.values
        cur = mm[mm.target_year == yv].tot.values
        out[yv] = (cur < q3(others)).mean()
    return pd.Series(out)


def obs_loo_event(yv):
    return np.array([yv[i] < q3(np.delete(yv, i)) for i in range(len(yv))])


SKILL_DROP = ("z_", "P_", "E_", "PI", "obs_m", "obs_c", "obs_l")


def fc_record(ev, ev_raw, ev_e):
    return dict(r_tf=ev["r_tf"], r_used=ev["r_used"], z_2027=ev["z_2027"], z_hc_max=ev["z_hc_max"],
                P_below_zcap=ev["P_below_zcap"],
                P_below_cal=ev["P_below"], P_clim_ref=ev["P_clim_ref"], P_below_normT=ev["P_below_normT"],
                P_below_T9120=ev["P_below_T9120"], P_below_raw=ev_raw["P_below"],
                P_clim_ref_raw=ev_raw["P_clim_ref"], P_below_vsERA5=ev_e["P_below"],
                P_clim_ref_vsERA5=ev_e["P_clim_ref"],
                E_mm=ev["E_mm"], PI10_mm=ev["PI10_mm"], PI90_mm=ev["PI90_mm"],
                E_pct=100 * ev["E_mm"] / ev["obs_mean_mm"],
                PI10_pct=100 * ev["PI10_mm"] / ev["obs_mean_mm"], PI90_pct=100 * ev["PI90_mm"] / ev["obs_mean_mm"],
                obs_mean_mm=ev["obs_mean_mm"], obs_clim10_mm=ev["obs_clim10_mm"],
                obs_clim90_mm=ev["obs_clim90_mm"], obs_LT_mm=ev["obs_lt_mm"], obs_LT9120_mm=ev["obs_lt9120_mm"],
                loo_r=ev["loo_r"])


# ---------- per-model skill + forecast ----------
skill_rows, fc_rows, memp = [], [], {}
models = LIVE + SENS + SKILL_ONLY
for method in ["frac", "touch"]:
    for season in SEASONS:
        o = obs_s[obs_s.season == season].set_index("target_year")[["chirps_mm", "era5_mm"]]
        for model in models:
            e = ens[(ens.model == model) & (ens.method == method) & (ens.season == season)].set_index("target_year")
            if e.empty:
                continue
            mm = mem[(mem.model == model) & (mem.method == method) & (mem.season == season)]
            hc = e[e.index < FC_YEAR].join(o, how="inner")
            periods = {"common": hc[(hc.index >= COMMON[0]) & (hc.index <= COMMON[1])], "full": hc}
            if model.startswith("CFSv2"):
                periods["hindcast_only_1983-2011"] = hc[hc.index <= 2011]
            for pname, h in periods.items():
                h = h.dropna(subset=["ensmean", "chirps_mm"])
                if len(h) < 10:
                    continue
                he = h.dropna(subset=["era5_mm"])
                row = dict(model=model, method=method, season=season, period=pname,
                           r_chirps=stats.pearsonr(h.ensmean, h.chirps_mm)[0],
                           rho_chirps=stats.spearmanr(h.ensmean, h.chirps_mm)[0],
                           r_era5=stats.pearsonr(he.ensmean, he.era5_mm)[0],
                           rho_era5=stats.spearmanr(he.ensmean, he.era5_mm)[0],
                           n_era5=len(he), mean_members=h.n_mem.mean())
                fc_ok = FC_YEAR in e.index and pname == "common"
                X0 = np.array([[e.loc[FC_YEAR, "ensmean"]]]) if fc_ok else None
                Xh = h.ensmean.values[:, None]
                ev = evaluate(Xh, h.chirps_mm.values, h.index.values, X0, T_years=(1991, 2020))
                ev_e = evaluate(he.ensmean.values[:, None], he.era5_mm.values, he.index.values, X0)
                row.update({k: v for k, v in ev.items() if not k.startswith(SKILL_DROP)})
                pm = member_loo_frac(mm, h.index.values)
                row["auc_members_loo"] = roc_auc(pm.values, obs_loo_event(h.chirps_mm.values))
                row.update(loo_r_era5=ev_e["loo_r"], auc_ensmean_era5=ev_e["auc_ensmean"])
                memp[(model, method, season, pname)] = pm
                skill_rows.append(row)
                if fc_ok:
                    ev_raw = evaluate(Xh, h.chirps_mm.values, h.index.values, X0, T_years=(1991, 2020), tf="raw")
                    clim_e = h.ensmean.values
                    fv = e.loc[FC_YEAR, "ensmean"]
                    clim_m = mm[mm.target_year.isin(h.index)].tot.values
                    f_m = mm[mm.target_year == FC_YEAR].tot.values
                    fr = dict(model=model, method=method, season=season,
                              clim_years=f"{h.index.min()}-{h.index.max()}", n_clim_years=len(h),
                              n_members_2027=len(f_m), ensmean_2027_mm=fv,
                              ensmean_pct_of_model_clim=100 * fv / clim_e.mean(),
                              ensmean_percentile=100 * (clim_e < fv).mean(),
                              ensmean_rank=f"{int((clim_e < fv).sum()) + 1}/{len(clim_e) + 1}",
                              frac_members_below_LT=(f_m < q3(clim_m)).mean(),
                              frac_members_above_UT=(f_m > np.quantile(clim_m, 2 / 3)).mean(),
                              **fc_record(ev, ev_raw, ev_e))
                    if model.startswith("CFSv2"):  # sensitivity: vs 1983-2011 hindcast-only climatology
                        hh = e[e.index <= 2011]
                        fr["ensmean_percentile_vs_hc8311"] = 100 * (hh.ensmean.values < fv).mean()
                        fr["frac_below_LT_vs_hc8311"] = (f_m < q3(mm[mm.target_year <= 2011].tot.values)).mean()
                    fc_rows.append(fr)

        # ---------- multi-model ensemble: equal weight per model on standardised (transformed) ens means ----------
        for mname, mlist in [("MME-5", LIVE), ("MME-5_CFSv2early", LIVE[:4] + SENS)]:
            wide = (ens[(ens.method == method) & (ens.season == season) & ens.model.isin(mlist)]
                    .pivot(index="target_year", columns="model", values="ensmean")[mlist])
            hc = wide[(wide.index >= COMMON[0]) & (wide.index <= COMMON[1])].dropna().join(o, how="inner")
            hc = hc.dropna(subset=["chirps_mm"])
            Xh = hc[mlist].values
            X0 = wide.loc[[FC_YEAR], mlist].values.astype(float)
            ev = evaluate(Xh, hc.chirps_mm.values, hc.index.values, X0, T_years=(1991, 2020))
            ev_raw = evaluate(Xh, hc.chirps_mm.values, hc.index.values, X0, T_years=(1991, 2020), tf="raw")
            he = hc.dropna(subset=["era5_mm"])
            ev_e = evaluate(he[mlist].values, he.era5_mm.values, he.index.values, X0)
            p = calib_fit(np.log1p(Xh), np.log1p(hc.chirps_mm.values))
            zs = calib_predict(p, np.log1p(Xh))[2]
            zse = calib_predict(p, np.log1p(he[mlist].values))[2]
            row = dict(model=mname, method=method, season=season, period="common",
                       r_chirps=stats.pearsonr(zs, hc.chirps_mm)[0], rho_chirps=stats.spearmanr(zs, hc.chirps_mm)[0],
                       r_era5=stats.pearsonr(zse, he.era5_mm)[0], rho_era5=stats.spearmanr(zse, he.era5_mm)[0],
                       n_era5=len(he))
            row.update({k: v for k, v in ev.items() if not k.startswith(SKILL_DROP)})
            pm = pd.concat([memp[(m, method, season, "common")] for m in mlist], axis=1).loc[hc.index].mean(1)
            row["auc_members_loo"] = roc_auc(pm.values, obs_loo_event(hc.chirps_mm.values))
            row.update(loo_r_era5=ev_e["loo_r"], auc_ensmean_era5=ev_e["auc_ensmean"])
            skill_rows.append(row)
            fr_models = [r for r in fc_rows if r["model"] in mlist and r["method"] == method and r["season"] == season]
            fc_rows.append(dict(model=mname, method=method, season=season,
                                clim_years=f"{hc.index.min()}-{hc.index.max()}", n_clim_years=len(hc),
                                n_members_2027=sum(r["n_members_2027"] for r in fr_models),
                                ensmean_percentile=100 * (zs < ev["z_2027"]).mean(),
                                ensmean_rank=f"{int((zs < ev['z_2027']).sum()) + 1}/{len(zs) + 1}",
                                frac_members_below_LT=np.mean([r["frac_members_below_LT"] for r in fr_models]),
                                frac_members_above_UT=np.mean([r["frac_members_above_UT"] for r in fr_models]),
                                **fc_record(ev, ev_raw, ev_e)))

skill = pd.DataFrame(skill_rows)
fc = pd.DataFrame(fc_rows)
skill.to_csv(OUT / "skill_by_model.csv", index=False)
fc.to_csv(OUT / "forecast_2027_by_model.csv", index=False)
ens.to_csv(OUT / "ensmean_seasonal_totals.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
cols = ["model", "season", "period", "n_years", "yr0", "yr1", "r_chirps", "rho_chirps", "r_era5", "rho_era5",
        "r_tf", "loo_r", "auc_ensmean", "auc_members_loo", "hit_rate", "false_alarm_ratio", "n_flag", "n_event",
        "bss_loo", "loo_r_era5"]
print("\n=== SKILL (frac, common & CFSv2 hindcast-only) ===")
print(skill[(skill.method == "frac") & (skill.period != "full")][cols].round(2).to_string(index=False))
print("\n=== SKILL full record (frac) ===")
print(skill[(skill.method == "frac") & (skill.period == "full")][cols].round(2).to_string(index=False))
print("\n=== SKILL touch sensitivity (common) ===")
print(skill[(skill.method == "touch") & (skill.period == "common")][cols].round(2).to_string(index=False))
print("\n=== FORECAST 2027 ===")
fcols = ["model", "method", "season", "n_members_2027", "ensmean_pct_of_model_clim", "ensmean_percentile",
         "ensmean_rank", "frac_members_below_LT", "frac_members_above_UT", "r_used", "z_2027", "z_hc_max",
         "P_below_zcap", "P_below_cal", "P_clim_ref",
         "P_below_normT", "P_below_T9120", "P_below_raw", "P_clim_ref_raw", "P_below_vsERA5", "P_clim_ref_vsERA5"]
print(fc[fcols].round(3).to_string(index=False))
print(fc[fc.method == "frac"][["model", "season", "E_mm", "PI10_mm", "PI90_mm", "E_pct", "PI10_pct", "PI90_pct",
                                "obs_mean_mm", "obs_clim10_mm", "obs_clim90_mm", "obs_LT_mm", "obs_LT9120_mm"]].round(0).to_string(index=False))
if "ensmean_percentile_vs_hc8311" in fc:
    print(fc[fc.model.str.startswith("CFSv2")][["model"] + ["method", "season", "ensmean_percentile_vs_hc8311", "frac_below_LT_vs_hc8311"]])

# obs-obs agreement
for s in SEASONS:
    oo = obs_s[(obs_s.season == s) & obs_s.target_year.between(*COMMON)].dropna()
    print(s, "CHIRPS vs ERA5 r =", round(stats.pearsonr(oo.chirps_mm, oo.era5_mm)[0], 2), "n", len(oo),
          "CHIRPS mean", round(oo.chirps_mm.mean()), "ERA5 mean", round(oo.era5_mm.mean()))

# ---------- Nino3.4 ----------
nr = pd.read_parquet(OUT / "nino34_raw.parquet")
NSEAS = {"NDJ": [0.5, 1.5, 2.5], "DJF": [1.5, 2.5, 3.5], "JFM": [2.5, 3.5, 4.5],
         "FMA": [3.5, 4.5, 5.5], "MAM": [4.5, 5.5, 6.5]}
BASE = {m: (1991, 2020) for m in LIVE}
# CFSv2: avoid the 1999 hindcast discontinuity in tropical Pacific SST; FORECAST sst for 2011-2025
# not downloaded (server-side timeouts), so base = 1999-2010 hindcast Nov starts (CPC practice).
BASE["CFSv2"] = (1999, 2010)
nino_rows = []
for model in LIVE:
    d = nr[nr.model == model].dropna(subset=["sst"])
    b0, b1 = BASE[model]
    clim = d[d.start_year.between(b0, b1)].groupby("L").sst.mean()  # Nov starts b0..b1
    d = d.assign(anom=d.sst - d.L.map(clim))
    for s, Ls in NSEAS.items():
        sub = d[d.L.isin(Ls)]
        g = sub.groupby(["start_year", "member"]).anom
        a = g.mean()[g.count() == 3].rename("nino34_anom").reset_index()
        a["season"] = s
        a["model"] = model
        nino_rows.append(a)
nino = pd.concat(nino_rows)
n27 = nino[nino.start_year == 2026][["model", "member", "season", "nino34_anom"]]
n27.to_csv(OUT / "nino34_2027_members.csv", index=False)
nino.to_csv(OUT / "nino34_all_starts_members.csv", index=False)
summ = []
for s in NSEAS:
    v = n27[n27.season == s].nino34_anom
    # equal weight per model (each model's members weighted 1/(n_mem*n_models)) + pooled
    w = n27[n27.season == s].groupby("model").nino34_anom.transform(lambda x: 1 / len(x))
    order = np.argsort(v.values)
    cw = np.cumsum(w.values[order]) / w.sum()
    pq = lambda p: v.values[order][np.searchsorted(cw, p)]
    summ.append(dict(season=s, n_members=len(v), mean_pooled=v.mean(),
                     p10_pooled=v.quantile(.1), p50_pooled=v.quantile(.5), p90_pooled=v.quantile(.9),
                     mean_eqw=np.average(v, weights=w), p10_eqw=pq(.1), p50_eqw=pq(.5), p90_eqw=pq(.9),
                     **{f"mean_{m}": n27[(n27.season == s) & (n27.model == m)].nino34_anom.mean() for m in LIVE}))
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "nino34_2027_summary.csv", index=False)
print("\n=== Nino3.4 2027 plume (anom degC) ===")
print(summ.round(2).to_string(index=False))
print(nr.groupby("model").units.first())
