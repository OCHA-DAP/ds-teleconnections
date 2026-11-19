"""Dry Corridor (GT, HN, SV): NMME Nov-start skill and 2027 amounts for MJJA and SO; Nino3.4 plume + verification.

Usage: uv run python -I 03_analysis.py <nmme_dir> <psl_nino34.data> [<psl_tna.data>]
Reads out/nmme_prec_members.parquet, out/obs_monthly.csv, out/nino34_raw.parquet.
Rainfall outputs are AMOUNTS only (central estimate + 10-90% range); no P(below-normal) is computed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(sys.argv[1])
OUT = D / "out"
NINO_OBS = Path(sys.argv[2])
pr = pd.read_parquet(OUT / "nmme_prec_members.parquet")
obs = pd.read_csv(OUT / "obs_monthly.csv")

SEASONS = {"MJJA": [5, 6, 7, 8], "SO": [9, 10]}  # SO = Sep-Oct only (Postrera proxy; NOT SON)
FC_YEAR = 2027
COMMON = (1991, 2026)  # target years; Nov starts 1990..2025
CLIM = (1991, 2020)  # observed reference mean for "% of normal"
CANDIDATES = ["CanESM5", "GEM5.2-NEMO", "CESM1", "CCSM4", "CFSv2"]
SKILL_ONLY = ["SPEAR"]  # no Nov 2026 start (re-checked 2026-11-09)
SENS = ["CFSv2early"]  # CFSv2 with EARLY_MONTH_SAMPLES realtime
COUNTRIES = ["GT", "HN", "SV"]
Z80 = stats.norm.ppf(0.9)
HAS_ERA5 = obs["era5_mm"].notna().any()


# ---------- seasonal totals ----------
def member_seasons(df):
    out = []
    for s, months in SEASONS.items():
        sub = df[df.target_month.isin(months)]
        g = sub.groupby(["country", "model", "seg", "method", "start_year", "member"])["prec_mm"]
        tot = g.sum(min_count=len(months))
        tot = tot[g.count() == len(months)]  # all months present
        out.append(tot.rename("tot").reset_index().assign(season=s))
    m = pd.concat(out)
    m["target_year"] = m["start_year"] + 1
    return m.dropna(subset=["tot"])


mem = member_seasons(pr)
ens = (mem.groupby(["country", "model", "method", "season", "target_year"])
       .agg(ensmean=("tot", "mean"), n_mem=("tot", "size"), seg=("seg", "first")).reset_index())
LIVE = {s: [m for m in CANDIDATES if ((ens.model == m) & (ens.season == s) & (ens.target_year == FC_YEAR)).any()]
        for s in SEASONS}
print("Live models (Nov 2026 start with every month of the season):")
for s, ms in LIVE.items():
    nm = {m: int(ens[(ens.model == m) & (ens.season == s) & (ens.target_year == FC_YEAR)].n_mem.max()) for m in ms}
    print(f"  {s}: {nm}  total {sum(nm.values())}")
for m in ["CFSv2", "CFSv2early"]:
    x = ens[(ens.model == m) & (ens.season == "MJJA") & (ens.country == "GT") & (ens.method == "frac")]
    print(f"  {m} MJJA members by target year:", x.set_index("target_year").n_mem.loc[2008:].to_dict())
pd.DataFrame([dict(season=s, model=m) for s, ms in LIVE.items() for m in ms]).to_csv(OUT / "live_models.csv",
                                                                                     index=False)

obs_s = []
for s, months in SEASONS.items():
    sub = obs[obs.month.isin(months)]
    g = sub.groupby(["country", "year"])
    cols = ["chirps_mm", "era5_mm"]
    o = g[cols].sum(min_count=len(months))
    o[g[cols].count() < len(months)] = np.nan
    obs_s.append(o.reset_index().rename(columns={"year": "target_year"}).assign(season=s))
obs_s = pd.concat(obs_s)
obs_s.to_csv(OUT / "obs_seasonal_totals.csv", index=False)

# ---------- conditional-normal calibration ----------
TF = {"log1p": (np.log1p, np.expm1), "raw": (lambda v: v, lambda v: v)}
q3 = lambda v: np.quantile(v, 1 / 3)  # noqa: E731


def calib_fit(X, y):
    """X: (n, k) transformed ens-mean per model; y: (n,) transformed obs."""
    mx, sx = X.mean(0), X.std(0, ddof=1)
    z = ((X - mx) / sx).mean(1)  # equal weight per model on standardised anomalies
    mz, sz = z.mean(), z.std(ddof=1)
    zs = (z - mz) / sz
    mu, sig = y.mean(), y.std(ddof=1)
    r = np.corrcoef(zs, y)[0, 1]
    r_use = max(r, 0.0)  # no negative skill: floor to climatology
    E = mu + r_use * sig * zs
    res_sd = np.sqrt(((y - E) ** 2).sum() / (len(y) - 2)) if r_use > 0 else sig
    return dict(mx=mx, sx=sx, mz=mz, sz=sz, mu=mu, sig=sig, r=r, r_use=r_use, res_sd=res_sd)


def calib_predict(p, X0):
    z = ((X0 - p["mx"]) / p["sx"]).mean(-1)
    zs = (z - p["mz"]) / p["sz"]
    return p["mu"] + p["r_use"] * p["sig"] * zs, p["res_sd"], zs


def roc_auc(score, event):
    event = np.asarray(event, bool)
    n1, n0 = event.sum(), (~event).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = stats.rankdata(score)
    return (rk[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(Xr, yr, years, X0r=None, tf="log1p"):
    """Hindcast skill (incl. LOO) for one model (k=1) or the MME (k>1); optional 2027 amounts. Inputs in mm."""
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
        P_loo[i] = stats.norm.cdf((T - E) / s)  # hindcast-only (for BSS skill metric)
        Pc_loo[i] = stats.norm.cdf((T - p["mu"]) / p["sig"])
        ev_loo[i] = y[i] < T
        _, _, zs_tr = calib_predict(p, X[tr])
        flag[i] = zs_i < q3(zs_tr)
    p = calib_fit(X, y)
    zs_all = calib_predict(p, X)[2]
    ev_full = y < q3(y)
    res = dict(
        n_years=n, yr0=int(years.min()), yr1=int(years.max()),
        r_tf=p["r"], r_used=p["r_use"], loo_r=np.corrcoef(E_loo, y)[0, 1],
        auc_ensmean=roc_auc(-zs_all, ev_full),
        hit_rate=(flag & ev_loo).sum() / max(ev_loo.sum(), 1),
        false_alarm_ratio=(flag & ~ev_loo).sum() / max(flag.sum(), 1),
        n_flag=int(flag.sum()), n_event=int(ev_loo.sum()),
        bss_loo=1 - np.mean((P_loo - ev_loo) ** 2) / np.mean((Pc_loo - ev_loo) ** 2),
        res_sd=p["res_sd"], obs_sd_tf=p["sig"],
    )
    if X0r is not None:
        E, s, zs0 = calib_predict(p, f(X0r))
        E, zs0 = float(np.squeeze(E)), float(np.squeeze(zs0))
        zc = float(np.clip(zs0, zs_all.min(), zs_all.max()))  # capped at hindcast range (no extrapolation)
        Ec = p["mu"] + p["r_use"] * p["sig"] * zc
        yo = finv(y)
        inclim = (years >= CLIM[0]) & (years <= CLIM[1])
        res.update(
            z_2027=zs0, z_hc_min=float(zs_all.min()), z_hc_max=float(zs_all.max()),
            rank_2027=int((zs_all < zs0).sum()) + 1, rank_of=n + 1,
            E_mm=float(finv(E)), PI10_mm=float(finv(E - Z80 * s)), PI90_mm=float(finv(E + Z80 * s)),
            E_zcap_mm=float(finv(Ec)),
            E_clim_mm=float(finv(p["mu"])), PI10_clim_mm=float(finv(p["mu"] - Z80 * p["sig"])),
            PI90_clim_mm=float(finv(p["mu"] + Z80 * p["sig"])),
            obs_mean9120_mm=float(yo[inclim].mean()), n_clim=int(inclim.sum()),
        )
    return res


SKILL_DROP = ("z_", "E_", "PI", "obs_m", "rank_", "n_clim")


def member_loo_frac(mm, years):
    out = {}
    for yv in years:
        others = mm[(mm.target_year != yv) & mm.target_year.isin(years)].tot.values
        cur = mm[mm.target_year == yv].tot.values
        out[yv] = (cur < q3(others)).mean()
    return pd.Series(out)


def obs_loo_event(yv):
    return np.array([yv[i] < q3(np.delete(yv, i)) for i in range(len(yv))])


def amounts(ev):
    m = ev["obs_mean9120_mm"]
    return dict(E_mm=ev["E_mm"], PI10_mm=ev["PI10_mm"], PI90_mm=ev["PI90_mm"],
                E_pct=100 * ev["E_mm"] / m, PI10_pct=100 * ev["PI10_mm"] / m, PI90_pct=100 * ev["PI90_mm"] / m,
                E_zcap_mm=ev["E_zcap_mm"], E_clim_mm=ev["E_clim_mm"], E_clim_pct=100 * ev["E_clim_mm"] / m,
                PI10_clim_pct=100 * ev["PI10_clim_mm"] / m, PI90_clim_pct=100 * ev["PI90_clim_mm"] / m,
                obs_mean9120_mm=m, r_used=ev["r_used"], z_2027=ev["z_2027"], z_hc_min=ev["z_hc_min"],
                z_hc_max=ev["z_hc_max"], loo_r=ev["loo_r"])


def safe_r(a, b, fn=stats.pearsonr):
    return fn(a, b)[0] if len(a) >= 10 else np.nan


# ---------- per-model skill + forecast ----------
skill_rows, fc_rows, memp = [], [], {}
for method in ["frac", "touch"]:
    for cc in COUNTRIES:
        for season in SEASONS:
            o = obs_s[(obs_s.country == cc) & (obs_s.season == season)].set_index("target_year")[
                ["chirps_mm", "era5_mm"]]
            for model in CANDIDATES + SENS + SKILL_ONLY:
                sel = (ens.country == cc) & (ens.model == model) & (ens.method == method) & (ens.season == season)
                e = ens[sel].set_index("target_year")
                if e.empty:
                    continue
                mm = mem[(mem.country == cc) & (mem.model == model) & (mem.method == method) & (mem.season == season)]
                hc = e[e.index < FC_YEAR].join(o, how="inner")
                periods = {"common": hc[(hc.index >= COMMON[0]) & (hc.index <= COMMON[1])], "full": hc}
                if model.startswith("CFSv2"):
                    periods["hindcast_only_1983-2011"] = hc[hc.index <= 2011]
                for pname, h in periods.items():
                    h = h.dropna(subset=["ensmean", "chirps_mm"])
                    if len(h) < 10:
                        continue
                    he = h.dropna(subset=["era5_mm"])
                    row = dict(country=cc, model=model, method=method, season=season, period=pname,
                               r_chirps=stats.pearsonr(h.ensmean, h.chirps_mm)[0],
                               rho_chirps=stats.spearmanr(h.ensmean, h.chirps_mm)[0],
                               r_era5=safe_r(he.ensmean, he.era5_mm),
                               rho_era5=safe_r(he.ensmean, he.era5_mm, stats.spearmanr),
                               n_era5=len(he), mean_members=h.n_mem.mean())
                    fc_ok = FC_YEAR in e.index and pname == "common" and model in LIVE[season] + SENS
                    X0 = np.array([[e.loc[FC_YEAR, "ensmean"]]]) if fc_ok else None
                    ev = evaluate(h.ensmean.values[:, None], h.chirps_mm.values, h.index.values, X0)
                    row.update({k: v for k, v in ev.items() if not k.startswith(SKILL_DROP)})
                    if len(he) >= 10:
                        ev_e = evaluate(he.ensmean.values[:, None], he.era5_mm.values, he.index.values)
                        row.update(loo_r_era5=ev_e["loo_r"], auc_ensmean_era5=ev_e["auc_ensmean"])
                    pm = member_loo_frac(mm, h.index.values)
                    row["auc_members_loo"] = roc_auc(pm.values, obs_loo_event(h.chirps_mm.values))
                    memp[(cc, model, method, season, pname)] = pm
                    skill_rows.append(row)
                    if fc_ok:
                        clim_e = h.ensmean.values
                        fv = e.loc[FC_YEAR, "ensmean"]
                        fc_rows.append(dict(
                            country=cc, model=model, method=method, season=season,
                            clim_years=f"{h.index.min()}-{h.index.max()}", n_clim_years=len(h),
                            n_members_2027=int(e.loc[FC_YEAR, "n_mem"]), ensmean_2027_mm=fv,
                            ensmean_pct_of_model_clim=100 * fv / clim_e.mean(),
                            ensmean_rank=f"{int((clim_e < fv).sum()) + 1}/{len(clim_e) + 1}",
                            r_chirps=row["r_chirps"], r_era5=row["r_era5"], loo_r_era5=row.get("loo_r_era5"),
                            **amounts(ev)))
                        if model.startswith("CFSv2"):
                            hh = e[(e.index <= 2011)]
                            fc_rows[-1]["ensmean_rank_vs_hc8311"] = (
                                f"{int((hh.ensmean.values < fv).sum()) + 1}/{len(hh) + 1}")

            # ---------- MME: equal weight per model on standardised (log1p) ensemble means ----------
            live = LIVE[season]
            mmes = [(f"MME-{len(live)}", live)]
            if "CFSv2" in live:
                mmes.append((f"MME-{len(live)}_CFSv2early", [m if m != "CFSv2" else "CFSv2early" for m in live]))
                mmes.append((f"MME-{len(live) - 1}_noCFSv2", [m for m in live if m != "CFSv2"]))
            for mname, mlist in mmes:
                wide = (ens[(ens.country == cc) & (ens.method == method) & (ens.season == season)
                            & ens.model.isin(mlist)]
                        .pivot(index="target_year", columns="model", values="ensmean")[mlist])
                hc = wide[(wide.index >= COMMON[0]) & (wide.index <= COMMON[1])].dropna().join(o, how="inner")
                hc = hc.dropna(subset=["chirps_mm"])
                Xh = hc[mlist].values
                X0 = wide.loc[[FC_YEAR], mlist].values.astype(float)
                ev = evaluate(Xh, hc.chirps_mm.values, hc.index.values, X0)
                he = hc.dropna(subset=["era5_mm"])
                p = calib_fit(np.log1p(Xh), np.log1p(hc.chirps_mm.values))
                zs = calib_predict(p, np.log1p(Xh))[2]
                zse = calib_predict(p, np.log1p(he[mlist].values))[2]
                row = dict(country=cc, model=mname, method=method, season=season, period="common",
                           r_chirps=stats.pearsonr(zs, hc.chirps_mm)[0],
                           rho_chirps=stats.spearmanr(zs, hc.chirps_mm)[0],
                           r_era5=safe_r(zse, he.era5_mm), rho_era5=safe_r(zse, he.era5_mm, stats.spearmanr),
                           n_era5=len(he))
                row.update({k: v for k, v in ev.items() if not k.startswith(SKILL_DROP)})
                if len(he) >= 10:
                    ev_e = evaluate(he[mlist].values, he.era5_mm.values, he.index.values)
                    row.update(loo_r_era5=ev_e["loo_r"], auc_ensmean_era5=ev_e["auc_ensmean"])
                pm = pd.concat([memp[(cc, m, method, season, "common")] for m in mlist], axis=1).loc[hc.index].mean(1)
                row["auc_members_loo"] = roc_auc(pm.values, obs_loo_event(hc.chirps_mm.values))
                skill_rows.append(row)
                fr_models = [r for r in fc_rows if r["model"] in mlist and r["method"] == method
                             and r["season"] == season and r["country"] == cc]
                fc_rows.append(dict(country=cc, model=mname, method=method, season=season,
                                    clim_years=f"{hc.index.min()}-{hc.index.max()}", n_clim_years=len(hc),
                                    n_members_2027=sum(r["n_members_2027"] for r in fr_models),
                                    ensmean_rank=f"{ev['rank_2027']}/{ev['rank_of']}",
                                    r_chirps=row["r_chirps"], r_era5=row["r_era5"],
                                    loo_r_era5=row.get("loo_r_era5"), **amounts(ev)))

skill = pd.DataFrame(skill_rows)
fc = pd.DataFrame(fc_rows)
skill.to_csv(OUT / "skill_by_model.csv", index=False)
fc.to_csv(OUT / "forecast_2027_by_model.csv", index=False)
ens.to_csv(OUT / "ensmean_seasonal_totals.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)
print("\nERA5 national rows present:", bool(HAS_ERA5))
cols = ["country", "model", "season", "period", "n_years", "yr0", "yr1", "r_chirps", "rho_chirps", "r_era5",
        "r_tf", "loo_r", "loo_r_era5", "auc_ensmean", "auc_members_loo", "hit_rate", "false_alarm_ratio", "bss_loo"]
cols = [c for c in cols if c in skill]
print("\n=== SKILL (frac, common 1991-2026 & CFSv2 hindcast-only) ===")
print(skill[(skill.method == "frac") & (skill.period != "full")][cols].round(2).to_string(index=False))
print("\n=== SKILL full record (frac) ===")
print(skill[(skill.method == "frac") & (skill.period == "full")][cols].round(2).to_string(index=False))
print("\n=== SKILL touch sensitivity (MMEs, common) ===")
print(skill[(skill.method == "touch") & (skill.period == "common") & skill.model.str.startswith("MME")][cols]
      .round(2).to_string(index=False))

print("\n=== FORECAST 2027: amounts (frac; conditional-normal on log1p, r floored at 0) ===")
print("E = calibrated central estimate (back-transformed median); PI = 10-90% range; pct = % of CHIRPS 1991-2020 mean;")
print("E_clim = what the same method gives with zero skill (climatological median); rank 1 = driest.")
acols = ["country", "season", "model", "n_members_2027", "ensmean_rank", "ensmean_pct_of_model_clim", "r_chirps",
         "r_era5", "loo_r", "loo_r_era5", "r_used", "E_mm", "PI10_mm", "PI90_mm", "E_pct", "PI10_pct", "PI90_pct",
         "E_clim_pct", "obs_mean9120_mm"]
f1 = fc[fc.method == "frac"].sort_values(["country", "season"])
print(f1[acols].round(2).to_string(index=False))
print("\nz of 2027 vs hindcast range, and central with z capped at the hindcast range (frac):")
print(f1[["country", "season", "model", "z_2027", "z_hc_min", "z_hc_max", "E_mm", "E_zcap_mm"]].round(2)
      .to_string(index=False))
if "ensmean_rank_vs_hc8311" in fc:
    print("\nCFSv2 2027 ens-mean rank vs the 1983-2011 hindcast-only climatology:")
    print(f1[f1.model.str.startswith("CFSv2")][["country", "season", "model", "ensmean_rank",
                                                  "ensmean_rank_vs_hc8311"]].to_string(index=False))
print("\n=== FORECAST 2027 touch sensitivity (MMEs) ===")
print(fc[(fc.method == "touch") & fc.model.str.startswith("MME")][["country", "season", "model", "ensmean_rank",
                                                                   "r_chirps", "loo_r", "E_mm", "PI10_mm", "PI90_mm",
                                                                   "E_pct"]].round(2).to_string(index=False))

print("\n=== Observed seasonal totals 1991-2020 and CHIRPS-ERA5 agreement 1991-2026 ===")
for cc in COUNTRIES:
    for s in SEASONS:
        oo = obs_s[(obs_s.country == cc) & (obs_s.season == s) & obs_s.target_year.between(*COMMON)]
        o9 = oo[oo.target_year.between(*CLIM)]
        both = oo.dropna()
        r = stats.pearsonr(both.chirps_mm, both.era5_mm)[0] if len(both) >= 10 else np.nan
        print(f"{cc} {s}: CHIRPS mean {o9.chirps_mm.mean():.0f} mm (sd {o9.chirps_mm.std():.0f}, "
              f"p10-p90 {o9.chirps_mm.quantile(.1):.0f}-{o9.chirps_mm.quantile(.9):.0f}), "
              f"ERA5 mean {o9.era5_mm.mean():.0f} mm; CHIRPS vs ERA5 r = {r:.2f} (n {len(both)}); "
              f"last CHIRPS year {int(oo.dropna(subset=['chirps_mm']).target_year.max())}")


# ======================= SST indices: Nino3.4 and TNA =======================
def read_psl(path, col):
    """PSL 13-column monthly text format; values < -90 are missing."""
    lines = path.read_text().splitlines()
    y0, y1 = map(int, lines[0].split()[:2])
    rec = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) == 13 and p[0].isdigit() and y0 <= int(p[0]) <= y1:
            for mth, v in enumerate(p[1:], 1):
                v = float(v)
                rec.append((int(p[0]), mth, np.nan if v < -90 else v))
    return pd.DataFrame(rec, columns=["year", "month", col])


def sst_anoms(raw, seasons):
    """Per-model anomalies vs each model's own Nov-start lead climatology; season means per start/member."""
    live = [m for m in CANDIDATES if ((raw.model == m) & (raw.start_year == FC_YEAR - 1)).any()]
    base = {m: (1991, 2020) for m in live}
    # CFSv2: avoid the 1999 hindcast discontinuity; FORECAST sst for 2011-2025 not downloaded (server-side
    # timeouts), so base = 1999-2010 hindcast Nov starts (CPC practice).
    base["CFSv2"] = (1999, 2010)
    rows = []
    for model in live:
        d = raw[raw.model == model].dropna(subset=["sst"])
        b0, b1 = base[model]
        clim = d[d.start_year.between(b0, b1)].groupby("L").sst.mean()
        d = d.assign(anom=d.sst - d.L.map(clim))
        for s, Ls in seasons.items():
            g = d[d.L.isin(Ls)].groupby(["start_year", "member"]).anom
            a = g.mean()[g.count() == len(Ls)].rename("anom").reset_index()
            rows.append(a.assign(season=s, model=model))
    return pd.concat(rows), live


def wq(v, w, p):
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return v[o][np.searchsorted(cw, p)]


def plume(a27, seasons, live):
    summ = []
    for s in seasons:
        x = a27[a27.season == s]
        if x.empty:
            continue
        v = x.anom.values
        w = x.groupby("model").anom.transform(lambda t: 1 / len(t)).values  # equal weight per model
        summ.append(dict(season=s, n_models=x.model.nunique(), n_members=len(v),
                         mean_eqw=np.average(v, weights=w), p10_eqw=wq(v, w, .1), p50_eqw=wq(v, w, .5),
                         p90_eqw=wq(v, w, .9), mean_pooled=v.mean(), p10_pooled=np.quantile(v, .1),
                         p50_pooled=np.quantile(v, .5), p90_pooled=np.quantile(v, .9),
                         **{f"mean_{m}": x[x.model == m].anom.mean() for m in live}))
    return pd.DataFrame(summ)


def obs_season(no, start_year, Ls, col):
    vals = []
    for L in Ls:
        k = int(np.floor(L))
        r = no[(no.year == start_year + (11 - 1 + k) // 12) & (no.month == (11 - 1 + k) % 12 + 1)][col]
        vals.append(r.iloc[0] if len(r) else np.nan)
    return np.mean(vals) if np.all(np.isfinite(vals)) else np.nan


def shares(mu, sd):
    return dict(P_ge_plus05=1 - stats.norm.cdf((0.5 - mu) / sd),
                P_neutral=stats.norm.cdf((0.5 - mu) / sd) - stats.norm.cdf((-0.5 - mu) / sd),
                P_le_minus05=stats.norm.cdf((-0.5 - mu) / sd))


def verify(an, live, no, col, seasons, cal_seasons, ndj_mme, alt_col=None, enso_shares=False):
    """MME mean (equal weight; models with a complete 1990-2025 Nov-start record) vs observed; calibrated 2027."""
    emean = an.groupby(["model", "season", "start_year"]).anom.mean().unstack("model")
    starts = range(COMMON[0] - 1, COMMON[1])  # Nov 1990 .. Nov 2025
    ver_rows, ver_years, cal_rows = [], [], []
    for s, Ls in seasons.items():
        if s not in emean.index.get_level_values("season"):
            continue
        es = emean.xs(s, level="season")
        vm = [m for m in live if m in es and es[m].reindex(starts).notna().all()]
        if not vm:
            continue
        rec = pd.DataFrame({"start_year": list(starts)})
        rec["target_year"] = rec.start_year + 1
        rec["fc"] = es[vm].mean(1).reindex(starts).values
        rec["obs"] = [obs_season(no, y, Ls, col) for y in starts]
        if alt_col:
            rec["obs_alt"] = [obs_season(no, y, Ls, alt_col) for y in starts]
        rec["mme_ndj_nino34"] = ndj_mme.reindex(starts).values
        rec = rec.dropna(subset=["fc", "obs"])
        err = rec.fc - rec.obs
        bias, sd_e = err.mean(), err.std(ddof=1)
        strong = rec.mme_ndj_nino34 >= 1.0
        f27 = float(es.loc[FC_YEAR - 1, vm].mean()) if FC_YEAR - 1 in es.index else np.nan
        row = dict(season=s, models="+".join(vm), n_models=len(vm), n=len(rec), yr0=int(rec.target_year.min()),
                   yr1=int(rec.target_year.max()), r=stats.pearsonr(rec.fc, rec.obs)[0],
                   rmse=np.sqrt((err ** 2).mean()), bias=bias, rmse_biascorr=sd_e,
                   slope_obs_on_fc=np.polyfit(rec.fc, rec.obs, 1)[0], sd_fc=rec.fc.std(), sd_obs=rec.obs.std(),
                   n_strong=int(strong.sum()), bias_strong=err[strong].mean(),
                   rmse_strong=np.sqrt((err[strong] ** 2).mean()), mme_2027=f27)
        if alt_col:
            ea = rec.fc - rec.obs_alt
            row.update(bias_alt=ea.mean(), rmse_alt=np.sqrt((ea ** 2).mean()))
        ver_rows.append(row)
        ver_years.append(rec.assign(season=s, err=err))
        if s in cal_seasons:
            mu, sd = f27 - bias, sd_e
            # sensitivity A: bias and error sd from the strong-El-Nino years only (model NDJ >= 1.0)
            mu_s, sd_s = f27 - err[strong].mean(), err[strong].std(ddof=1)
            # sensitivity B: linear regression of observed on the MME mean (corrects amplitude as well as bias)
            b1, b0 = np.polyfit(rec.fc, rec.obs, 1)
            mu_r = b0 + b1 * f27
            sd_r = np.sqrt(((rec.obs - (b0 + b1 * rec.fc)) ** 2).sum() / (len(rec) - 2))
            c = dict(season=s, models=row["models"], mme_2027_raw=f27, bias=bias, mean_calibrated=mu, sd=sd,
                     p10=mu + stats.norm.ppf(.1) * sd, p50=mu, p90=mu + stats.norm.ppf(.9) * sd)
            if enso_shares:
                c.update(shares(mu, sd))
                if alt_col:
                    ea = rec.fc - rec.obs_alt
                    mu8 = f27 - ea.mean()
                    c.update(mean_cal_altbase=mu8,
                             **{f"{k}_altbase": v for k, v in shares(mu8, ea.std(ddof=1)).items()})
            c.update(mean_strongbias=mu_s, sd_strong=sd_s, p10_strongbias=mu_s + stats.norm.ppf(.1) * sd_s,
                     p90_strongbias=mu_s + stats.norm.ppf(.9) * sd_s,
                     mean_regression=mu_r, sd_regression=sd_r, p10_regression=mu_r + stats.norm.ppf(.1) * sd_r,
                     p90_regression=mu_r + stats.norm.ppf(.9) * sd_r)
            if enso_shares:
                c.update(**{f"{k}_strongbias": v for k, v in shares(mu_s, sd_s).items()},
                         **{f"{k}_regression": v for k, v in shares(mu_r, sd_r).items()})
            cal_rows.append(c)
    return pd.DataFrame(ver_rows), pd.concat(ver_years), pd.DataFrame(cal_rows)


# ---------- Nino3.4 ----------
no = read_psl(NINO_OBS, "nino34_8110")
mclim = no[no.year.between(1991, 2020)].groupby("month").nino34_8110.mean()
no["nino34_9120"] = no.nino34_8110 - no.month.map(mclim)  # re-based to 1991-2020 (current ONI convention)
no.to_csv(OUT / "nino34_obs_monthly.csv", index=False)
last = no.dropna().iloc[-1]
print(f"\nObserved Nino3.4 (PSL ERSSTv6): last month {int(last.year)}-{int(last.month):02d} = "
      f"{last.nino34_8110:.2f} (1981-2010 base), {last.nino34_9120:.2f} (1991-2020 base). "
      f"Mean 1991-2020 of the 1981-2010-base anomaly, by month: {mclim.round(2).to_dict()}")

NSEAS = {"NDJ": [0.5, 1.5, 2.5], "DJF": [1.5, 2.5, 3.5], "JFM": [2.5, 3.5, 4.5], "FMA": [3.5, 4.5, 5.5],
         "MAM": [4.5, 5.5, 6.5], "AMJ": [5.5, 6.5, 7.5], "MJJ": [6.5, 7.5, 8.5], "JJA": [7.5, 8.5, 9.5],
         "JAS": [8.5, 9.5, 10.5], "ASO": [9.5, 10.5, 11.5], "SO": [10.5, 11.5]}
nr = pd.read_parquet(OUT / "nino34_raw.parquet")
nino, LIVE_SST = sst_anoms(nr, NSEAS)
nino.rename(columns={"anom": "nino34_anom"}).to_csv(OUT / "nino34_all_starts_members.csv", index=False)
n27 = nino[nino.start_year == FC_YEAR - 1]
n27.rename(columns={"anom": "nino34_anom"})[["model", "member", "season", "nino34_anom"]].to_csv(
    OUT / "nino34_2027_members.csv", index=False)
summ = plume(n27, NSEAS, LIVE_SST)
summ.to_csv(OUT / "nino34_2027_summary.csv", index=False)
print("\n=== Nino3.4 2027 plume, Nov 2026 starts (anomaly degC; each model vs its own Nov-start climatology) ===")
print("Members per model:", n27[n27.season == "NDJ"].groupby("model").member.nunique().to_dict())
print(summ.round(2).to_string(index=False))
print("SST units:", nr.groupby("model").units.first().to_dict())

ndj_mme = nino[nino.season == "NDJ"].groupby(["model", "start_year"]).anom.mean().unstack(0).mean(1)
ver, vy, cal = verify(nino, LIVE_SST, no, "nino34_9120", NSEAS, ("MJJ", "JAS", "SO"), ndj_mme,
                      alt_col="nino34_8110", enso_shares=True)
ver.to_csv(OUT / "nino34_hindcast_verification.csv", index=False)
vy.to_csv(OUT / "nino34_hindcast_mme_by_year.csv", index=False)
cal.to_csv(OUT / "nino34_2027_calibrated.csv", index=False)
print("\n=== Nino3.4 hindcast verification: MME mean (equal weight, models with a complete 1990-2025 Nov-start"
      " record) vs PSL ERSSTv6 re-based to 1991-2020 ===")
print("bias = forecast - observed; rmse_biascorr = sd of errors; *_strong = years with model NDJ >= 1.0;"
      " *_alt = vs the index as published (1981-2010 base)")
print(ver.drop(columns="models").round(2).to_string(index=False))
print("models:", ver.groupby("models").season.apply(list).to_dict())
print("\nPer-year MME forecast vs observed after strong El Nino winters (model NDJ >= 1.0):")
vs = vy[(vy.mme_ndj_nino34 >= 1.0) & vy.season.isin(["MJJ", "JAS", "SO"])]
print(vs.pivot_table(index=["target_year", "mme_ndj_nino34"], columns="season", values=["fc", "obs"])
      .round(2).to_string())
print("\n=== Calibrated 2027 Nino3.4: N(MME mean - bias, sd of hindcast errors), 1991-2020 base ===")
print("primary = mean_calibrated +- sd; A = *_strongbias (bias + sd from model-NDJ>=1.0 years); B = *_regression (OLS obs on MME)")
print(cal.set_index("season").drop(columns="models").T.round(2).to_string())

# ---------- TNA (tropical North Atlantic, 5.5-23.5N, 57.5-15W) ----------
TNA_OBS = Path(sys.argv[3]) if len(sys.argv) > 3 else None
if TNA_OBS is not None and (OUT / "tna_raw.parquet").exists():
    TSEAS = {"AMJ": [5.5, 6.5, 7.5], "MJJ": [6.5, 7.5, 8.5], "JAS": [8.5, 9.5, 10.5], "SO": [10.5, 11.5]}
    to = read_psl(TNA_OBS, "tna")  # PSL TNA (HadISST1.1), base as given (1991-2020 climatology)
    to.to_csv(OUT / "tna_obs_monthly.csv", index=False)
    lt = to.dropna().iloc[-1]
    print(f"\nObserved TNA (PSL, HadISST1.1, 1991-2020 climatology as given): last month "
          f"{int(lt.year)}-{int(lt.month):02d} = {lt.tna:.2f}")
    tr = pd.read_parquet(OUT / "tna_raw.parquet")
    tna, LIVE_TNA = sst_anoms(tr, TSEAS)
    tna.rename(columns={"anom": "tna_anom"}).to_csv(OUT / "tna_all_starts_members.csv", index=False)
    t27 = tna[tna.start_year == FC_YEAR - 1]
    t27.rename(columns={"anom": "tna_anom"})[["model", "member", "season", "tna_anom"]].to_csv(
        OUT / "tna_2027_members.csv", index=False)
    tsum = plume(t27, TSEAS, LIVE_TNA)
    tsum.to_csv(OUT / "tna_2027_summary.csv", index=False)
    print("\n=== TNA 2027 plume, Nov 2026 starts (anomaly degC; each model vs its own Nov-start climatology) ===")
    print("Members per model:", t27[t27.season == "AMJ"].groupby("model").member.nunique().to_dict())
    print(tsum.round(2).to_string(index=False))
    tver, tvy, tcal = verify(tna, LIVE_TNA, to, "tna", TSEAS, ("AMJ", "MJJ", "JAS", "SO"), ndj_mme)
    tver.to_csv(OUT / "tna_hindcast_verification.csv", index=False)
    tvy.to_csv(OUT / "tna_hindcast_mme_by_year.csv", index=False)
    tcal.to_csv(OUT / "tna_2027_calibrated.csv", index=False)
    print("\n=== TNA hindcast verification: MME mean vs PSL TNA (bias = forecast - observed) ===")
    print(tver.drop(columns="models").round(2).to_string(index=False))
    print("models:", tver.groupby("models").season.apply(list).to_dict())
    print("\n=== Calibrated 2027 TNA: N(MME mean - bias, sd of hindcast errors) ===")
    print(tcal.set_index("season").drop(columns="models").T.round(2).to_string())
    # joint behaviour of hindcast errors (are Nino3.4 and TNA misses correlated?)
    jrows = []
    for s in ("MJJ", "JAS", "SO"):
        a = vy[vy.season == s].set_index("target_year")
        b = tvy[tvy.season == s].set_index("target_year")
        j = a[["err", "fc", "obs"]].join(b[["err", "fc", "obs"]], lsuffix="_nino34", rsuffix="_tna", how="inner")
        if len(j) >= 10:
            jrows.append(dict(season=s, n=len(j), r_errors=stats.pearsonr(j.err_nino34, j.err_tna)[0],
                              r_fc=stats.pearsonr(j.fc_nino34, j.fc_tna)[0],
                              r_obs=stats.pearsonr(j.obs_nino34, j.obs_tna)[0]))
    jr = pd.DataFrame(jrows)
    jr.to_csv(OUT / "tna_nino34_error_correlation.csv", index=False)
    print("\nCorrelation of hindcast errors (and of forecasts / observations), Nino3.4 vs TNA, same targets:")
    print(jr.round(2).to_string(index=False))
else:
    print("\nTNA: no obs path or no tna_raw.parquet; skipped")
