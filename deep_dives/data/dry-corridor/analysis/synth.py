"""Page data for /enso/dry-corridor/: per unit and season, the typical amounts, each evidence line and the combined
outlook; NMME national rows; national series for the charts; map outlines. Usage: synth.py SP -> page_data.json
Combined outlook (same recipe for every unit): central = mean of the central values of the CHIRPS and ERA5 past-season
medians and the CHIRPS and ERA5 regressions; likely range = mean of the two regressions' 10th and 90th percentiles.
All percentages are of the 1991-2020 median of the dataset concerned; mm figures are on the CHIRPS scale."""
import sys, json, os, numpy as np, pandas as pd
SP = sys.argv[1]
exec(open(f"{SP}/scripts/common.py").read())
R = pd.read_csv(f"{SP}/season_results.csv")
geo = json.load(open(f"{SP}/geo.json"))
SUBS = {"primera": {"MJ": [5, 6], "JA": [7, 8]}, "postrera": {"SO": [9, 10], "ON": [10, 11]}}
r1 = lambda v: None if pd.isna(v) else int(round(float(v)))
def sub_pct(X, ms):
    S = season_tot(X, ms); return 100 * S / S.loc[1991:2020].median()
SUBP = {(src, k): sub_pct(X, ms) for src, X in [("CHIRPS", CH), ("ERA5", ER)] for sd in SUBS.values() for k, ms in sd.items()}
TIM = pd.read_parquet(f"{SP}/daily_metrics_units.parquet") if os.path.exists(f"{SP}/daily_metrics_units.parquet") else None
# COD-AB Honduras names come without accents
NAME_FIX = {"Atlantida": "Atlántida", "Colon": "Colón", "Copan": "Copán", "Cortes": "Cortés", "El Paraiso": "El Paraíso",
            "Francisco Morazan": "Francisco Morazán", "Intibuca": "Intibucá", "Islas de La Bahia": "Islas de la Bahía",
            "Santa Barbara": "Santa Bárbara"}
OC = pd.read_csv(f"{SP}/onset_check.csv")
U = {}
for pc in NAMES.index:
    u = {"name": NAME_FIX.get(NAMES[pc], NAMES[pc]), "iso3": ISO[pc], "level": int(LEVEL[pc]), "aoi": any(pc in v for v in AOI.values())}
    for sea, key in [("primera", "pri"), ("postrera", "pos")]:
        c = R[(R.pcode == pc) & (R.season == sea) & (R.src == "CHIRPS")].iloc[0]
        e = R[(R.pcode == pc) & (R.season == sea) & (R.src == "ERA5")].iloc[0]
        ch_an = [r1(c[f"a{y}"]) for y in AFTER]; er_an = [r1(e[f"a{y}"]) for y in AFTER]
        ch_reg = [r1(c.reg_med), r1(c.reg_p10), r1(c.reg_p90)]; er_reg = [r1(e.reg_med), r1(e.reg_p10), r1(e.reg_p90)]
        central = r1(np.mean([np.median(ch_an), np.median(er_an), c.reg_med, e.reg_med]))
        lo = r1(np.mean([c.reg_p10, e.reg_p10])); hi = r1(np.mean([c.reg_p90, e.reg_p90]))
        T = c.typical
        d = {"typ_mm": r1(T), "typ_p10": r1(c.clim_p10), "typ_p90": r1(c.clim_p90),
             "typ_p10_pct": r1(c.clim_p10_pct), "typ_p90_pct": r1(c.clim_p90_pct),
             "ch_an": ch_an, "er_an": er_an, "ch_reg": ch_reg, "er_reg": er_reg,
             "central": central, "lo": lo, "hi": hi,
             "central_mm": r1(T * central / 100), "lo_mm": r1(T * lo / 100), "hi_mm": r1(T * hi / 100),
             "r_ch": round(float(c.loo_r), 2), "r_er": round(float(e.loo_r), 2), "y2026": r1(c.y2026),
             "y2026_er": r1(e.y2026), "y2025": r1(c.get("y2025")), "y2025_er": r1(e.get("y2025")), "y2023": r1(c.get("y2023")), "y2023_er": r1(e.get("y2023")),
             "n34only": [r1(c.n34only_med), r1(e.n34only_med)], "loo_n34only": [round(float(c.loo_r_n34only), 2), round(float(e.loo_r_n34only), 2)]}
        for k in SUBS[sea]:
            P = SUBP[("CHIRPS", k)][pc]; Pe = SUBP[("ERA5", k)][pc]
            d[f"{k}_ch"] = [r1(P.get(y)) for y in AFTER]; d[f"{k}_er"] = [r1(Pe.get(y)) for y in AFTER]
            d[f"{k}_2026"] = r1(P.get(2026)) if 2026 in P.index else None
        if sea == "postrera":
            d["scen"] = {nm: [r1(np.mean([c[f"scen_{nm}_{q}"], e[f"scen_{nm}_{q}"]])) for q in ("med", "p10", "p90")] for nm in ("nino", "neutral", "nina")}
        else:
            d["raw_nmme_reg"] = r1(np.mean([c.scen_rawnmme_med, e.scen_rawnmme_med]))
        u[key] = d
    if TIM is not None:
        t = TIM[TIM.unit == pc].set_index("y")
        if len(t):
            base = t.loc[1991:2020]
            u["tim"] = {"onset_typ": r1(base.onset.median()), "onset_an": [r1(t.loc[y, "onset_anom"]) for y in AFTER],
                        "late_an": [r1(100 * t.loc[y, "onset_bad"]) for y in AFTER], "late_base": r1(100 * base.onset_bad.mean()),
                        "onset_2026": r1(t.loc[2026, "onset_anom"]) if 2026 in t.index else None,
                        "late_2026": r1(100 * t.loc[2026, "onset_bad"]) if 2026 in t.index else None,
                        "end_typ": r1(base.end.median()), "end_an": [r1(t.loc[y, "end_anom"]) for y in AFTER],
                        "early_an": [r1(100 * t.loc[y, "end_bad"]) for y in AFTER], "early_base": r1(100 * base.end_bad.mean())}
            o = OC[OC.unit == pc].set_index("group")
            if len(o):
                u["tim"].update({"main_after": round(float(o.loc["after", "main_anom"]), 1), "alt_after": round(float(o.loc["after", "alt_anom"]), 1),
                                 "main_late_after": r1(o.loc["after", "main_late"]), "alt_late_after": r1(o.loc["after", "alt_late"]),
                                 "main_late_other": r1(o.loc["other", "main_late"]), "alt_late_other": r1(o.loc["other", "alt_late"]),
                                 "main_during": round(float(o.loc["during", "main_anom"]), 1)})
    U[pc] = u
# NMME national rows (% of the CHIRPS 1991-2020 median of the same months)
F = pd.read_csv(f"{SP}/nmme/out/forecast_2027_by_model.csv")
F = F[(F.method == "frac") & (F.model.isin(["MME-5", "MME-4"]))]
NM = {}
for r in F.itertuples():
    ms = [5, 6, 7, 8] if r.season == "MJJA" else [9, 10]
    med = season_tot(CH, ms).loc[1991:2020, r.country].median()
    NM.setdefault(r.country, {})["pri" if r.season == "MJJA" else "so"] = {
        "c": r1(100 * r.E_mm / med), "lo": r1(100 * r.PI10_mm / med), "hi": r1(100 * r.PI90_mm / med),
        "rank": r.ensmean_rank, "loo_r": round(float(r.loo_r), 2), "r": round(float(r.r_chirps), 2), "model": r.model}
# national series for charts
T = pd.read_csv(f"{SP}/season_totals.csv")
TNA = read_psl(os.environ.get("TNA_FILE", f"{SP}/idx/tna.data"))
pts = []
for y in range(1981, 2027):
    row = {"y": y, "n_mjj": round(nseas(y, [5, 6, 7]), 2), "n_son": None if pd.isna(nseas(y, [9, 10, 11])) else round(nseas(y, [9, 10, 11]), 2),
           "tna_mjj": round(float(np.mean([TNA.get((y, m)) for m in (5, 6, 7)])), 2), "djf": round(nseas(y, [0, 1, 2]), 2)}
    for pc in ["GT", "HN", "SV", "GT_AOI", "HN_AOI", "SV_AOI"]:
        for sea, key in [("primera", "pri"), ("postrera", "pos")]:
            for src, s2 in [("CHIRPS", "c"), ("ERA5", "e")]:
                v = T[(T.src == src) & (T.season == sea) & (T.pcode == pc) & (T.y == y)].pct
                row[f"{pc}_{key}_{s2}"] = r1(v.iloc[0]) if len(v) else None
    pts.append(row)
INP = json.load(open(f"{SP}/enso_inputs.json"))
from scipy import stats
z = stats.norm.ppf(0.9)
gmu = INP["n34_mjj"][0] - INP["tna_mjj"][0]; gsd = float(np.hypot(INP["n34_mjj"][1], INP["tna_mjj"][1]))
band = {"pri": [round(gmu - z * gsd, 2), round(gmu + z * gsd, 2)], "pos": [round(INP["n34_son"][0] - z * INP["n34_son"][1], 2), round(INP["n34_son"][0] + z * INP["n34_son"][1], 2)]}
so26 = {}
for src, X in [("CHIRPS", CH), ("ERA5", ER)]:
    for nm, ms in [("SO", [9, 10]), ("Sep", [9])]:
        S = season_tot(X, ms)
        if 2026 in S.index: so26[f"{src}_{nm}"] = {pc: r1(100 * S.loc[2026, pc] / S.loc[1991:2020, pc].median()) for pc in ["GT", "HN", "SV", "GT_AOI", "HN_AOI", "SV_AOI"]}
out = {"u": U, "nmme": NM, "pts": pts, "geo": geo, "after": AFTER, "band": band, "y2026_late": so26, "inputs": {k: v for k, v in INP.items() if not k.startswith("_")}}
json.dump(out, open(f"{SP}/page_data.json", "w"), ensure_ascii=False)
for pc in ["GT", "HN", "SV", "GT_AOI", "HN_AOI", "SV_AOI"]:
    for k in ("pri", "pos"):
        d = U[pc][k]
        print(pc, k, "typ", d["typ_mm"], f'({d["typ_p10"]}-{d["typ_p90"]}; {d["typ_p10_pct"]}-{d["typ_p90_pct"]}%)', "central", d["central"], f'({d["lo"]}-{d["hi"]})',
              d["central_mm"], f'{d["lo_mm"]}-{d["hi_mm"]}', "ch_an", d["ch_an"], "er_an", d["er_an"], "reg", d["ch_reg"], d["er_reg"])
print(json.dumps(NM))
dept = {pc: u for pc, u in U.items() if u["level"] == 1}
for k in ("pri", "pos"):
    ch = sorted((u[k]["central"], u["name"]) for u in dept.values()); print(k, "dept central range", ch[:3], ch[-3:])
print("band", band); print("2026 late", so26)
print(os.path.getsize(f"{SP}/page_data.json") // 1000, "kB")
