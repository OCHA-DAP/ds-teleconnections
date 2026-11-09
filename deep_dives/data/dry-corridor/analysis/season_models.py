"""Seasonal totals (Primera MJJA, Postrera SON) by unit: typical amounts, ocean-index relations and 2027 amounts.
Usage: season_models.py SP INPUTS_JSON
INPUTS_JSON gives the calibrated NMME November-start forecasts on the observed index scales:
  {"n34_mjj": [mu, sd], "tna_mjj": [mu, sd], "n34_son": [mu, sd]}
  primera : log1p(MJJA) ~ Niño3.4(MJJ) - TNA(MJJ)      (the Pacific-Atlantic gradient; both observed, PSL)
  postrera: log1p(SON)  ~ Niño3.4(SON)                  (the Atlantic adds little and not consistently)
  2027 draws: Niño3.4 and TNA ~ N(mu, sd) independently; regression parameters by bootstrap over years.
  after   : the six seasons after a strong El Niño winter (1983, 1992, 1998, 2010, 2016, 2024).
  scen    : Postrera at Niño3.4 SON = +1 / 0 / -1. Primera sensitivity: gradient at the uncorrected NMME Niño3.4 (+1.25).
  lag     : correlation with the preceding DJF Niño3.4 (reported only)."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
SP = sys.argv[1]; INP = json.load(open(sys.argv[2]))
exec(open(f"{SP}/scripts/common.py").read())
TNA = read_psl(os.environ.get("TNA_FILE", f"{SP}/idx/tna.data"))
tna = lambda y, ms: float(np.mean([TNA.get((y, m)) for m in ms]))
NI["TNA_MJJ"] = [tna(y, [5, 6, 7]) if y <= 2026 else np.nan for y in NI.index]
rng = np.random.default_rng(0)
Z = stats.norm.ppf(0.9)
def design(sea, yrs):
    if sea == "primera": return np.column_stack([np.ones(len(yrs)), NI.loc[yrs, "MJJ"] - NI.loc[yrs, "TNA_MJJ"]])
    return np.column_stack([np.ones(len(yrs)), NI.loc[yrs, "SON"]])
def fit(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; return b, np.sqrt(e @ e / (len(y) - X.shape[1]))
def loo_r(X, y):
    p = [X[i] @ fit(np.delete(X, i, 0), np.delete(y, i))[0] for i in range(len(y))]; return np.corrcoef(p, y)[0, 1]
def draws_x(sea, n):
    if sea == "primera":
        return np.column_stack([np.ones(n), rng.normal(*INP["n34_mjj"], n) - rng.normal(*INP["tna_mjj"], n)])
    return np.column_stack([np.ones(n), rng.normal(*INP["n34_son"], n)])
def predictive(sea, X, y, nb=400, nd=25):
    out = []
    for _ in range(nb):
        ii = rng.integers(0, len(y), len(y)); b, s = fit(X[ii], y[ii])
        out.append(draws_x(sea, nd) @ b + rng.normal(0, s, nd))
    return np.expm1(np.concatenate(out))
rows, tots = [], []
for src, Xd in [("CHIRPS", CH), ("ERA5", ER)]:
    for sea, ms in SEASONS.items():
        S = season_tot(Xd, ms); S = S[S.index >= 1981]
        for pc in S.columns:
            if pc not in NAMES.index: continue
            v = S[pc].dropna(); hist = v[v.index <= 2025]
            base = hist[(hist.index >= 1991) & (hist.index <= 2020)]; typ = base.median(); typ_mean = base.mean()
            tots.append(pd.DataFrame({"src": src, "season": sea, "pcode": pc, "y": v.index, "mm": v.values, "pct": 100 * v.values / typ}))
            hy = hist.index.values; y = np.log1p(hist.values)
            X = design(sea, hy); b, s = fit(X, y)
            d = predictive(sea, X, y)
            xl = NI.loc[hy, "DJF"].values; xc = NI.loc[hy, CONC[sea]].values
            X1 = np.column_stack([np.ones(len(hy)), xc]); b1, s1 = fit(X1, y)
            mu1 = INP["n34_mjj" if sea == "primera" else "n34_son"]
            d1 = np.expm1(np.column_stack([np.ones(10000), rng.normal(*mu1, 10000)]) @ b1 + rng.normal(0, s1, 10000))
            r = dict(src=src, season=sea, pcode=pc, name=NAMES[pc], level=LEVEL[pc], iso3=ISO[pc], n=len(y),
                     typical=typ, typical_mean=typ_mean, clim_p10=np.quantile(hist, 0.1), clim_p90=np.quantile(hist, 0.9),
                     clim_p10_pct=100 * np.quantile(hist, 0.1) / typ, clim_p90_pct=100 * np.quantile(hist, 0.9) / typ,
                     r_n34=stats.pearsonr(xc, y)[0], r_lag=stats.pearsonr(xl, y)[0], loo_r=loo_r(X, y),
                     slope=b[1], t_slope=b[1] / np.sqrt(np.diag(s**2 * np.linalg.inv(X.T @ X)))[1],
                     reg_med=100 * np.median(d) / typ, n34only_med=100 * np.median(d1) / typ, loo_r_n34only=loo_r(X1, y), reg_p10=100 * np.quantile(d, .1) / typ, reg_p90=100 * np.quantile(d, .9) / typ)
            scen = [("rawnmme", INP["n34_mjj_raw"] - INP["tna_mjj"][0])] if sea == "primera" else [("nino", 1.0), ("neutral", 0.0), ("nina", -1.0)]
            for nm, x0 in scen:
                xr0 = np.array([1, x0])
                c = np.expm1(xr0 @ b + np.array([-Z, 0, Z]) * s)
                r.update({f"scen_{nm}_p10": 100 * c[0] / typ, f"scen_{nm}_med": 100 * c[1] / typ, f"scen_{nm}_p90": 100 * c[2] / typ})
            for yy in AFTER: r[f"a{yy}"] = 100 * hist.get(yy, np.nan) / typ
            r["after_med"] = np.nanmedian([r[f"a{yy}"] for yy in AFTER]); r["after_mean"] = np.nanmean([r[f"a{yy}"] for yy in AFTER])
            r["y2026"] = 100 * v.get(2026, np.nan) / typ; r["y2025"] = 100 * v.get(2025, np.nan) / typ; r["y2023"] = 100 * v.get(2023, np.nan) / typ
            r["fit2023"] = 100 * np.expm1(design(sea, np.array([2023]))[0] @ b) / typ
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{SP}/season_results.csv", index=False)
pd.concat(tots).to_csv(f"{SP}/season_totals.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)
cols = ["src", "season", "pcode", "typical", "r_n34", "r_lag", "loo_r", "t_slope", "reg_med", "reg_p10", "reg_p90", "n34only_med", "loo_r_n34only", "after_med"] + \
       [f"a{y}" for y in AFTER] + ["scen_rawnmme_med", "scen_nino_med", "scen_neutral_med", "scen_nina_med", "y2026"]
print(R[R.level != 1].sort_values(["season", "pcode", "src"])[cols].round(2).to_string(index=False))
