"""Page data for /enso/pacific/: SEAS5 November 2026 forecast, ENSO relation, past seasons, CERF.

usage: synth.py SP OUT_DIR
  SP      work dir holding data/ (seas5.parquet, era5.parquet, sstoi.indices, oni.ascii.txt, nina34_psl.data,
          {iso}_adm{0,1}.gpkg, rasters/{seas5,era5}/*.nc, ghcnd_monthly.pkl, cerf_pacific.csv, chirps_units.parquet)
  OUT_DIR where page_data.json and results/*.csv go
All rainfall percentages are of the 1991-2020 mean of the same series (forecasts: of the mean November forecast).
"""
import json, sys, glob, os
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr


def _open(f):
    return xr.open_dataset(f)["__xarray_dataarray_variable__"]


SP = sys.argv[1].rstrip("/") + "/"
OUTD = sys.argv[2].rstrip("/") + "/"
D = SP + "data/"
os.makedirs(OUTD + "results", exist_ok=True)

s = pd.read_parquet(D + "seas5.parquet")
e = pd.read_parquet(D + "era5.parquet")
oi = pd.read_csv(D + "sstoi.indices", sep=r"\s+")
oi.columns = ["yr", "mon", "n12", "n12a", "n3", "n3a", "n4", "n4a", "n34", "n34a"]
octi = oi[oi.mon == 10].set_index("yr")[["n12a", "n3a", "n4a", "n34a"]]
octi["n3m4"] = octi.n3a - octi.n4a
oni = pd.read_csv(D + "oni.ascii.txt", sep=r"\s+")
oni_aso = oni[oni.SEAS == "ASO"].set_index("YR").ANOM
psl = pd.read_csv(D + "nina34_psl.data", sep=r"\s+", skiprows=1, header=None, nrows=2026 - 1948 + 1, engine="python").set_index(0)
psl_oct = psl[10].astype(float).replace(-99.99, np.nan)
INDICES = {"oisst_oct": octi.n34a, "oni_aso": oni_aso, "psl_oct": psl_oct}

DAYS = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
TRI = {"NDJ": [11, 12, 1], "DJF": [12, 1, 2], "JFM": [1, 2, 3], "FMA": [2, 3, 4], "MAM": [3, 4, 5], "NA": [11, 12, 1, 2, 3, 4]}
TRIS = ["NDJ", "DJF", "JFM", "FMA", "MAM"]
UNITS = {  # page unit -> admin pcodes averaged (equal weight; country units are the admin-0 polygon)
    "FJ": ["FJ"], "VU": ["VU"], "TO": ["TO"], "SB": ["SB"],
    "SB_SE": ["SB05", "SB08", "SB09"], "SB_C": ["SB04", "SB06", "SB07"], "SB_NW": ["SB01", "SB02", "SB03"],
}
UNIT_NAMES = {"FJ": "Fiji", "VU": "Vanuatu", "TO": "Tonga", "SB": "Solomon Islands",
              "SB_SE": "Solomon Islands south-east (Temotu, Makira-Ulawa, Rennell-Bellona)",
              "SB_C": "Solomon Islands centre (Guadalcanal, Malaita, Central)",
              "SB_NW": "Solomon Islands north-west (Choiseul, Western, Isabel)"}


def obs_series(pcs, months, wet=True):
    ee = e[e.pcode.isin(pcs)].copy()
    ee["m"] = ee.valid_date.dt.month
    ee["y"] = ee.valid_date.dt.year
    ee = ee[ee.m.isin(months)].copy()
    ee["sy"] = np.where(ee.m >= 11, ee.y, ee.y - 1) if wet else ee.y
    ee["mm"] = ee["mean"] * ee.m.map(DAYS)
    g = ee.groupby(["pcode", "sy"]).agg(mm=("mm", "sum"), n=("mm", "count")).reset_index()  # count: NaN months don't pass
    g = g[g.n == len(months)]
    w = g.pivot(index="sy", columns="pcode", values="mm")
    return w.dropna().mean(axis=1) if len(w.columns) == len(pcs) else w.mean(axis=1)


def fc_series(pcs, months):
    ss = s[s.pcode.isin(pcs) & (s.issued_date.dt.month == 11)].copy()
    ss["m"] = ss.valid_date.dt.month
    ss = ss[ss.m.isin(months) & (ss.leadtime <= 6)]
    ss["mm"] = ss["mean"] * ss.m.map(DAYS)
    g = ss.groupby(["pcode", ss.issued_date.dt.year.rename("iy")]).agg(mm=("mm", "sum"), n=("mm", "count")).reset_index()
    g = g[g.n == len(months)]
    return g.pivot(index="iy", columns="pcode", values="mm").mean(axis=1)


def pair(pcs, months):
    df = pd.DataFrame({"obs": obs_series(pcs, months), "fc": fc_series(pcs, months)}).join(octi)
    df["op"] = df.obs / df.loc[1991:2020].obs.mean() * 100
    df["fp"] = df.fc / df.loc[1991:2020].fc.mean() * 100
    return df


def r(a, b):
    j = pd.concat([a, b], axis=1).dropna()
    return float(j.corr().iloc[0, 1])


def rank_dry(series, year):
    v = series.dropna()
    return int((v < v.loc[year]).sum() + 1), int(len(v))


units = {}
for u, pcs in UNITS.items():
    rec = {"name": UNIT_NAMES[u], "tri": {}}
    for t, months in TRI.items():
        df = pair(pcs, months)
        h = df.loc[1982:2025]
        ho = h.dropna(subset=["op"])
        rk, n = rank_dry(df.fp, 2026)
        rec["tri"][t] = {
            "fc": round(float(df.loc[2026, "fp"]), 1), "rank": rk, "n": n,
            "skill": round(r(ho.fp, ho.op), 2),
            "r_fc_n34": round(r(h.fp, h.n34a), 2), "r_obs_n34": round(r(ho.op, ho.n34a), 2),
        }
    # ENSO-only estimates for Nov-Apr (forecast and observed): straight-line fits on three Niño3.4 series known at
    # issuance, each with and without a linear year term (the November forecasts rise over 1982-2025), at the 2026 value
    df = pair(pcs, TRI["NA"])
    est = {}
    for k, ix in INDICES.items():
        d = df.join(ix.rename("ix"))
        h = d.loc[1982:2025]
        ho = h.dropna(subset=["op"])
        x26 = float(ix.loc[2026])
        e1 = {"max_hind": round(float(ix.loc[1982:2025].max()), 2), "max_year": int(ix.loc[1982:2025].idxmax()), "v2026": round(x26, 2)}
        for trend in (False, True):
            for key, frame, col in (("fc", h, "fp"), ("obs", ho, "op")):
                X = np.column_stack([np.ones(len(frame)), frame.ix.values] + ([frame.index.values - 2000.0] if trend else []))
                beta, *_ = np.linalg.lstsq(X, frame[col].values, rcond=None)
                x = np.array([1.0, x26] + ([26.0] if trend else []))
                resid = frame[col].values - X @ beta
                e1[f"{key}{'_trend' if trend else ''}"] = round(float(x @ beta), 1)
                # neutral-year value (same fit at Niño3.4 = 0, trend at 2026) and the Niño term at the 2026 value
                e1[f"{key}{'_trend' if trend else ''}_neutral"] = round(float(x @ beta - beta[1] * x26), 1)
                e1[f"{key}{'_trend' if trend else ''}_nino"] = round(float(beta[1] * x26), 1)
                e1[f"{key}{'_trend' if trend else ''}_sd"] = round(float(resid.std(ddof=X.shape[1])), 1)
        # share of variation: squared correlation, raw and after removing a linear trend from both series
        def _r2(frame, col, detrend):
            a, b = frame[col].values, frame.ix.values
            if detrend:
                yy = frame.index.values.astype(float)
                a = a - np.polyval(np.polyfit(yy, a, 1), yy)
                b = b - np.polyval(np.polyfit(yy, b, 1), yy)
            return float(np.corrcoef(a, b)[0, 1] ** 2)
        e1.update(r2_fc=round(_r2(h, "fp", False), 3), r2_fc_dt=round(_r2(h, "fp", True), 3),
                  r2_obs=round(_r2(ho, "op", False), 3), r2_obs_dt=round(_r2(ho, "op", True), 3),
                  r_fc=round(r(h.fp, h.ix), 2), r_obs=round(r(ho.op, ho.ix), 2))
        est[k] = e1
    h = df.loc[1982:2025]
    rec["fc_trend_decade"] = round(float(np.polyfit(h.index.values, h.fp.values, 1)[0] * 10), 1)
    # observed rainfall that past forecasts of this size were followed by: fit of ERA5 % on forecast %, 1982-2025
    ho = h.dropna(subset=["op"])
    bo, ao = np.polyfit(ho.fp, ho.op, 1)
    sd = float((ho.op - (ao + bo * ho.fp)).std(ddof=2))
    x0 = float(df.loc[2026, "fp"])
    ex = float(ao + bo * x0)
    # 80% prediction interval including the fit's own uncertainty (2026 lies outside the range of past forecasts)
    from scipy import stats
    nn = len(ho)
    se = sd * np.sqrt(1 + 1 / nn + (x0 - ho.fp.mean()) ** 2 / ((ho.fp - ho.fp.mean()) ** 2).sum())
    q = float(stats.t.ppf(0.9, nn - 2))
    rec["obs_equiv"] = {"exp": round(ex, 1), "lo": round(ex - q * se, 1), "hi": round(ex + q * se, 1), "slope": round(float(bo), 2)}
    # observed rainfall vs other ENSO indices (r², Nov-Apr, 1982-2025)
    def _r2dt(a, b):
        yy = a.index.values.astype(float)
        a = a.values - np.polyval(np.polyfit(yy, a.values, 1), yy)
        b = b.values - np.polyval(np.polyfit(yy, b.values, 1), yy)
        return float(np.corrcoef(a, b)[0, 1] ** 2)
    rec["r2_obs_idx"] = {k: round(_r2dt(ho.op, ho[k]), 3) for k in ("n34a", "n3a", "n4a", "n3m4")}  # after removing trends
    # January-March observed rainfall in the strong El Niño seasons (October Niño3.4 >= 1.5 °C)
    jfm = pair(pcs, TRI["JFM"])
    strong = [int(y) for y in octi.index if 1982 <= y <= 2025 and octi.loc[y, "n34a"] >= 1.5]
    rec["jfm_strong"] = {y: round(float(jfm.loc[y, "op"]), 1) for y in strong}
    rec["enso_est"] = est
    # scatter + history (Nov-Apr), years 1981-2026
    rec["years"] = [int(y) for y in df.index]
    rec["obs"] = [None if pd.isna(v) else round(float(v), 1) for v in df.op]
    rec["fc"] = [None if pd.isna(v) else round(float(v), 1) for v in df.fp]
    rec["obs_mm_clim"] = round(float(df.loc[1991:2020].obs.mean()))
    rec["fc_mm_2026"] = round(float(df.loc[2026, "fc"]))
    rec["fc_mm_clim"] = round(float(df.loc[1991:2020].fc.mean()))
    # rain so far: Jul-Oct and Aug-Oct 2026 (and the Jul-Oct series, for comparisons with past years)
    for key, months in {"JASO": [7, 8, 9, 10], "ASO": [8, 9, 10]}.items():
        o = obs_series(pcs, months, wet=False)
        cl = o.loc[1991:2020].mean()
        rk, n = rank_dry(o, 2026)
        rec[key] = {"pct": round(float(o.loc[2026] / cl * 100), 1), "rank": rk, "n": n}
        if key == "JASO":
            rec["jaso_pct"] = {int(y): round(float(v / cl * 100), 1) for y, v in o.items()}
    units[u] = rec
    pd.DataFrame({"year": df.index, "obs_pct": df.op.round(1), "fc_pct": df.fp.round(1), "oct_n34_oisst": df.n34a}).to_csv(
        OUTD + f"results/nov_apr_{u}.csv", index=False)

# Niño3.4 index values for the scatter x axis
n34 = {"years": [int(y) for y in octi.index], "oct": [round(float(v), 2) for v in octi.n34a],
       "n3": [round(float(v), 2) for v in octi.n3a], "n4": [round(float(v), 2) for v in octi.n4a]}

# Timing: r(ERA5 trimester rainfall, concurrent Niño3.4), 1982-2025
mon = oi.set_index(pd.to_datetime(dict(year=oi.yr, month=oi.mon, day=1))).n34a
n3m = mon.rolling(3).mean()
NAMES12 = ["JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF"]
ORDER = ["JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ"]
timing = {}
for u, pcs in UNITS.items():
    ee = e[e.pcode.isin(pcs)].groupby("valid_date")["mean"].mean().sort_index().rolling(3).mean()
    row = {}
    for i, nm in enumerate(NAMES12):
        endm = (i + 2) % 12 + 1
        j = pd.concat([ee[ee.index.month == endm].rename("p"), n3m[n3m.index.month == endm].rename("n")], axis=1).dropna().loc["1982":"2025"]
        row[nm] = round(float(j.corr().iloc[0, 1]), 2)
    timing[u] = [row[k] for k in ORDER]
timing["order"] = ORDER

# Provinces (admin 1)
prov = []
for iso, iso2 in [("fji", "FJ"), ("vut", "VU"), ("ton", "TO"), ("slb", "SB")]:
    g = gpd.read_file(D + f"{iso}_adm1.gpkg")
    for pc, nm in zip(g.adm1_src, g.adm1_name):
        nm = {"Rennell-Bell": "Rennell-Bellona", "Northern  Division": "Northern Division"}.get(nm, nm)
        rec = {"pc": pc, "name": nm, "c": iso2}
        if not s.loc[s.pcode == pc, "mean"].notna().any():
            rec["note"] = "too small for the SEAS5 grid"
        else:
            df = pair([pc], TRI["NA"])
            ho = df.loc[1982:2025].dropna(subset=["op"])
            rk, n = rank_dry(df.fp, 2026)
            rec.update(fc=round(float(df.loc[2026, "fp"]), 1), rank=rk, n=n,
                       skill=round(r(ho.fp, ho.op), 2) if len(ho) > 20 else None,
                       tri={t: round(float(pair([pc], TRI[t]).loc[2026, "fp"]), 1) for t in TRIS})
        if not e[e.pcode == pc].empty:
            o = obs_series([pc], [7, 8, 9, 10], wet=False)
            if 2026 in o.index and np.isfinite(o.loc[2026]) and o.loc[1991:2020].mean() > 0:
                rk2, n2 = rank_dry(o, 2026)
                rec["JASO"] = {"pct": round(float(o.loc[2026] / o.loc[1991:2020].mean() * 100), 1), "rank": rk2, "n": n2}
        prov.append(rec)
pd.json_normalize(prov).to_csv(OUTD + "results/provinces.csv", index=False)

# Regional grids: SEAS5 Nov-Apr 2026 (% of mean Nov forecast 1991-2020); ERA5 Nov-Apr in past El Niño seasons (% of 1991-2020)
def seas5_total(y):
    fs = [D + f"rasters/seas5/precip_em_i{y}-11-01_lt{l}.nc" for l in range(6)]
    months = [11, 12, 1, 2, 3, 4]
    tot = None
    for f, m in zip(fs, months):
        a = _open(f) * DAYS[m]
        tot = a if tot is None else tot + a
    return tot
clim = sum(seas5_total(y) for y in range(1991, 2021)) / 30
s26 = seas5_total(2026)
g_s = (s26 / clim * 100)

def era5_month(y, m):
    f = glob.glob(D + f"rasters/era5/precip_reanalysis_v{y}-{m:02d}-01.nc")
    return _open(f[0]) * DAYS[m] if f else None
def era5_total(sy):
    parts = [era5_month(sy if m >= 11 else sy + 1, m) for m in [11, 12, 1, 2, 3, 4]]
    if any(p is None for p in parts):
        return None
    return sum(parts)
eclim = sum(era5_total(y) for y in range(1991, 2021)) / 30
grids = {"seas5_2026": g_s}
for y in [1982, 1997, 2015, 2023]:
    t = era5_total(y)
    if t is not None:
        grids[f"era5_{y}"] = t / eclim * 100
def pack(da):
    da = da.sortby("y", ascending=False).sortby("x")
    return {"x0": float(da.x.min()), "dx": float(abs(da.x[1] - da.x[0])), "y0": float(da.y.max()), "dy": float(abs(da.y[1] - da.y[0])),
            "nx": int(da.sizes["x"]), "ny": int(da.sizes["y"]),
            "v": [None if not np.isfinite(v) else int(round(float(v))) for v in da.values.ravel()]}
regional = {k: pack(v) for k, v in grids.items()}
# average over the band from Vanuatu to Tonga (12-22°S, 165°E-175°W), each layer as % of its own 1991-2020 mean
BAND = dict(y=slice(-12, -22), x=slice(165, 185))
def _band(da):
    da = da.sortby("y", ascending=False).sortby("x")
    w = np.cos(np.deg2rad(da.y))
    return float(da.sel(**BAND).weighted(w.sel(y=BAND["y"])).mean())
band = {"seas5_2026": round(_band(s26) / _band(clim) * 100, 1)}
for y in [1982, 1997, 2015, 2023]:
    t_ = era5_total(y)
    band[f"era5_{y}"] = round(_band(t_) / _band(eclim) * 100, 1)

# Outlines (lon shifted to 0-360 so Fiji and Tonga sit east of 180)
def shift(geom):
    from shapely.ops import transform
    return transform(lambda x, y, z=None: ((np.asarray(x) % 360), y), geom)
outl = {}
for iso, iso2 in [("fji", "FJ"), ("vut", "VU"), ("ton", "TO"), ("slb", "SB")]:
    for lvl in (0, 1):
        g = gpd.read_file(D + f"{iso}_adm{lvl}.gpkg").explode(index_parts=False)
        g["geometry"] = g.geometry.apply(shift)
        g = g[g.geometry.area > 0.0004]  # drop specks below ~2 km across
        g["geometry"] = g.geometry.simplify(0.01, preserve_topology=True)
        key = "adm1_src" if lvl == 1 else None
        feats = []
        for _, rw in g.iterrows():
            coords = [[[round(x, 3), round(y, 3)] for x, y in rw.geometry.exterior.coords]]
            feats.append({"pc": rw[key] if key else iso2, "ring": coords[0]})
        outl[f"{iso2}_{lvl}"] = feats

# Fiji gauges (GHCN-Daily, near-complete months), Nov-Apr % of 1991-2020 and r with ERA5 division
gm = pd.read_pickle(D + "ghcnd_monthly.pkl")
GAUGES = {"FJM00091680": ("Nadi", "FJ2"), "FJN00077648": ("Lautoka", "FJ2"), "FJM00091657": ("Labasa", "FJ1"),
          "FJM00091683": ("Nausori", "FJ4"), "FJM00091690": ("Suva (Laucala Bay)", "FJ4"), "FJM00091691": ("Lakeba", "FJ3")}
def _jaso_gauge(m):
    j = m[m.index.month.isin([7, 8, 9, 10])]
    g = j.groupby(j.index.year).agg(["sum", "size"])
    g = g[g["size"] == 4]["sum"]
    cl = g.loc[1991:2020]
    return {int(y): round(float(g.loc[y] / cl.mean() * 100)) for y in (1997, 2015) if y in g.index and len(cl) >= 25}


gauges = []
for sid, (nm, pc) in GAUGES.items():
    m = gm[sid].to_frame("mm")
    m["mo"] = m.index.month
    m["y"] = m.index.year
    w = m[m.mo.isin([11, 12, 1, 2, 3, 4])].copy()
    w["sy"] = np.where(w.mo >= 11, w.y, w.y - 1)
    gs = w.groupby("sy").agg(t=("mm", "sum"), n=("mm", "size"))
    gs = gs[gs.n == 6].t
    cl = gs.loc[1991:2020]
    o = obs_series([pc], TRI["NA"])
    gauges.append({"name": nm, "pc": pc, "y0": int(gs.index.min()), "y1": int(gs.index.max()), "n": int(len(gs)), "n_clim": int(len(cl)),
                   "r_era5": round(r(gs, o), 2), "r_n34": round(r(gs, octi.n34a), 2),
                   "skill_gauge": round(r(gs.loc[1982:], fc_series([pc], TRI["NA"])), 2),
                   "era5_div_1997": round(float(o.loc[1997] / o.loc[1991:2020].mean() * 100)),
                   "pct": {int(y): round(float(gs.loc[y] / cl.mean() * 100)) for y in [1982, 1997, 2009, 2015] if y in gs.index},
                   "jaso_pct": _jaso_gauge(gm[sid])})

# CERF
cerf = pd.read_csv(SP + "cerf_pacific.csv")
cerf_rows = [{"code": rw.application_code, "iso3": rw.country_iso3, "type": rw.emergency_type, "title": rw.title,
              "amount": int(rw.amount_approved), "date": str(rw.erc_endorsement_date)[:10],
              "valid": (f"{int(rw.valid_year_start)}-{int(rw.valid_month_start):02d} to {int(rw.valid_year_end)}-{int(rw.valid_month_end):02d}"
                        if pd.notna(rw.valid_year_start) else None)} for rw in cerf.itertuples()]

page = {"issued": "2026-11", "units": units, "n34": n34, "timing": timing, "provinces": prov, "regional": regional,
        "outlines": outl, "gauges": gauges, "cerf": cerf_rows, "band": band,
        "indices_2026": {"oisst_oct": float(octi.loc[2026, "n34a"]), "oisst_oct_n3": float(octi.loc[2026, "n3a"]),
                         "oisst_oct_n4": float(octi.loc[2026, "n4a"]), "oisst_oct_n12": float(octi.loc[2026, "n12a"]),
                         "oni_aso": float(oni_aso.loc[2026]), "psl_oct": float(psl_oct.loc[2026])}}
if os.path.exists(D + "chirps_jaso.json"):
    page["chirps_jaso"] = json.load(open(D + "chirps_jaso.json"))
json.dump(page, open(OUTD + "page_data.json", "w"), separators=(",", ":"))
print("wrote", OUTD + "page_data.json", os.path.getsize(OUTD + "page_data.json") // 1000, "kB")
for u, v in units.items():
    print(u, {t: (v["tri"][t]["fc"], v["tri"][t]["rank"], v["tri"][t]["skill"]) for t in TRI}, "JASO", v["JASO"], "est", {k: (x["fc"], x["obs"]) for k, x in v["enso_est"].items()})
