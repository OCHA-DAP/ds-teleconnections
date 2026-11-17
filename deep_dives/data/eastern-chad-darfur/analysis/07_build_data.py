"""Assemble everything the page needs into one JSON (../page_data.json, committed).

Choices made here (the page's Method block repeats them):
- Rainfall = June-September total, CHIRPS v2 and v3 shown as a range, point = their mean.
  Anomaly = % of the level the 1981-2025 linear trend gives for that year (the
  Sahel has wetted since the 1980s, so % of a fixed average flatters recent years);
  % of the 1991-2020 average is kept for tooltips.
- Vegetation comes first (it carries the timing of the rain, not just the total):
  MODIS NDVI season greening (Jul-Oct minus Mar-May) as % of its 2001-2025 trend
  level (09_greening.py) is the headline and the x-axis; JRC ASAP biomass (cumulative
  FPAR z-score, cropland, 21 Oct) and FAO VHI (Jul-Oct mean) are shown detrended the
  same way: the value minus the level a 2001-2025 straight line gives for that year
  (FAO VHI fitted on 1984-2025). All three greened strongly over 2001-2025, so
  undetrended anomalies flatter recent years.
- Seasons from 2023 on are war seasons in Darfur and refugee-influx seasons in
  eastern Chad: plotted, but kept out of the analogue summaries.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PAGE_DATA, PROC, RAW, SIDES, load_adm  # noqa: E402

OUT = PAGE_DATA
WAR_FROM = 2023
N_ANALOGUE = 5


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(
        float(x), n)


def rainfall():
    m = pd.read_parquet(PROC / "rain" / "monthly_side.parquet")
    m["y"] = m["month"].str[:4].astype(int)
    m["m"] = m["month"].str[5:].astype(int)
    ch = m[m["product"].isin(["chirps_v2", "chirps_v3"])]
    j = (ch[ch["m"].between(6, 9)].groupby(["product", "side", "y"])["mm"]
         .agg(["sum", "count"]).query("count == 4")["sum"].rename("mm").reset_index())
    out = []
    for (p, s), g in j.groupby(["product", "side"]):
        g = g.sort_values("y").copy()
        fit = g[g["y"].between(1981, 2025)]
        b = np.polyfit(fit["y"], fit["mm"], 1)
        g["trend"] = np.polyval(b, g["y"])
        g["pct_trend"] = 100 * g["mm"] / g["trend"]
        g["pct_norm"] = 100 * g["mm"] / g.loc[g["y"].between(1991, 2020), "mm"].mean()
        out.append(g)
    d = pd.concat(out)
    w = d.pivot_table(index=["side", "y"], columns="product",
                      values=["mm", "pct_trend", "pct_norm"])
    w.columns = [f"{a}_{b[-2:]}" for a, b in w.columns]
    w["pct_trend"] = w[["pct_trend_v2", "pct_trend_v3"]].mean(axis=1)
    w["pct_norm"] = w[["pct_norm_v2", "pct_norm_v3"]].mean(axis=1)
    w = w.reset_index()
    # ERA5 and IMERG Late JJAS as % of their own 1991-2020 / 2001-2020 mean (comparison only)
    alt = {}
    for p, base in (("era5", (1991, 2020)), ("imerg_late", (2001, 2020))):
        q = m[(m["product"] == p) & m["m"].between(6, 9)].groupby(["side", "y"])["mm"].agg(
            ["sum", "count"]).query("count == 4")["sum"]
        for s in SIDES:
            qs = q.loc[s]
            alt.setdefault(s, {})[p] = {int(y): r(100 * v / qs.loc[base[0]:base[1]].mean())
                                        for y, v in qs.items() if y >= 2020}
    for s in SIDES:
        jj = j[j["side"] == s].groupby(["y"])["mm"].mean()  # mean of v2/v3
        alt[s]["chirps_2001_2020"] = r(100 * jj.loc[2026] / jj.loc[2001:2020].mean())
    # monthly 2026 vs 1991-2020 mean, May-Oct
    mon = []
    for s in SIDES:
        for mm in range(5, 11):
            x = ch[(ch["side"] == s) & (ch["m"] == mm)]
            norm = x[x["y"].between(1991, 2020)].groupby("product")["mm"].mean()
            now = x[x["y"] == 2026].set_index("product")["mm"]
            mon.append({"side": s, "m": mm, "norm": r(norm.mean()),
                        "v2": r(now.get("chirps_v2")), "v3": r(now.get("chirps_v3"))})
    return w, alt, mon


def by_adm1():
    """June-September 2026 rain (% of 1991-2020, v2 and v3) and ASAP biomass at 21 Oct, per
    province/state (ASAP reports the five Darfur states and the three Chad provinces)."""
    m = pd.read_parquet(PROC / "rain" / "monthly_adm1.parquet")
    m["y"] = m["month"].str[:4].astype(int)
    m["mo"] = m["month"].str[5:].astype(int)
    j = m[m["product"].isin(["chirps_v2", "chirps_v3"]) & m["mo"].between(6, 9)].groupby(
        ["product", "adm1_pcode", "y"])["mm"].sum()
    cl = j.loc[:, :, 1991:2020].groupby(["product", "adm1_pcode"]).mean()
    p = (100 * j.xs(2026, level="y") / cl).unstack("product")
    yu = pd.read_parquet(PROC / "veg" / "yearly_unit.parquet")
    yu = yu[(yu["indicator"] == "zfparc_crop") & (yu["year"] == 2026)]
    name2pc = {"Wadi Fira": "TCD17", "Ouaddaï": "TCD14", "Sila": "TCD21",
               "North Darfur": "SD02", "South Darfur": "SD03", "West Darfur": "SD04",
               "East Darfur": "SD05", "Central Darfur": "SD06"}
    bio = {name2pc[u]: v for u, v in zip(yu["unit"], yu["value"]) if u in name2pc}
    grn = pd.read_parquet(PROC / "ndvi" / "adm1_2026.parquet").set_index("adm1_pcode")
    out = []
    for pc, row in p.iterrows():
        side = "chad" if pc.startswith("TCD") else "darfur"
        out.append({"pc": pc, "side": side, "v2": r(row["chirps_v2"], 0),
                    "v3": r(row["chirps_v3"], 0), "bio": r(bio.get(pc), 2),
                    "grn": r(grn.loc[pc, "pct_detr_2026"], 0) if pc in grn.index else None})
    return out


def veg_map():
    """2026 detrended greening at 0.04 deg as base64 uint8 (value = % clipped 0-254, 255 = none)."""
    import base64
    z = np.load(PROC / "ndvi" / "map_2026.npz")
    pct = z["pct"]
    u8 = np.where(np.isfinite(pct), np.clip(np.round(pct), 0, 254), 255).astype(np.uint8)
    return {"lat0": r(float(z["lat"][0]), 3), "lon0": r(float(z["lon"][0]), 3),
            "res": r(float(z["lon"][1] - z["lon"][0]), 3), "ny": int(u8.shape[0]),
            "nx": int(u8.shape[1]), "b64": base64.b64encode(u8.tobytes()).decode()}


def detrend(years, vals, base):
    """Value minus the level of a straight line fitted over the base years."""
    years, vals = np.asarray(years, float), np.asarray(vals, float)
    m = (years >= base[0]) & (years <= base[1]) & np.isfinite(vals)
    b = np.polyfit(years[m], vals[m], 1)
    return vals - np.polyval(b, years), b[0]


def vegetation():
    ys = pd.read_parquet(PROC / "veg" / "yearly_side.parquet")
    v = ys.pivot_table(index=["side", "year"], columns="indicator", values="value").reset_index()
    gr = pd.read_parquet(PROC / "ndvi" / "side_years.parquet").rename(columns={"year": "year"})
    v = v.merge(gr[["side", "year", "pct_raw", "pct_detr", "pct_detr_area", "pct_detr_floor"]],
                on=["side", "year"], how="outer")
    v = v.rename(columns={"pct_raw": "grn_raw", "pct_detr": "grn", "pct_detr_area": "grn_area",
                          "pct_detr_floor": "grn_floor"})
    slopes = {"grn": gr.groupby("side")["slope_per_decade"].first().round(1).to_dict()}
    for col, base in (("zfparc_crop", (2001, 2025)), ("zfparc_range", (2001, 2025)),
                      ("vhi", (1984, 2025))):
        v[col + "_d"] = np.nan
        slopes[col] = {}
        for s in SIDES:
            m = (v["side"] == s) & v[col].notna()
            d, b = detrend(v.loc[m, "year"], v.loc[m, col], base)
            v.loc[m, col + "_d"] = d
            slopes[col][s] = round(10 * b, 3)
    v = v.sort_values(["side", "year"])
    dk = pd.read_parquet(PROC / "veg" / "dekadal_side.parquet")
    dk = dk[dk["indicator"].isin(["zfparc_crop"])].copy()
    dk["date"] = pd.to_datetime(dk["date"])
    dk["year"] = dk["date"].dt.year
    dk["md"] = dk["date"].dt.strftime("%m-%d")
    dk = dk[(dk["md"] >= "05-01") & (dk["md"] <= "11-21")]
    # detrend each dekad's series across years (same 2001-2025 straight line)
    parts = []
    for (ind, s, md), h in dk.groupby(["indicator", "side", "md"]):
        h = h.copy()
        if h["year"].between(2001, 2025).sum() >= 15:
            h["value"], _ = detrend(h["year"], h["value"], (2001, 2025))
            parts.append(h)
    dk = pd.concat(parts)
    env = []
    for (ind, s), g in dk.groupby(["indicator", "side"]):
        past = g[g["year"].between(2001, 2025)]
        for md, h in past.groupby("md"):
            cur = g[(g["year"] == 2026) & (g["md"] == md)]["value"]
            env.append({"ind": ind, "side": s, "md": md,
                        "min": r(h["value"].min(), 2), "p10": r(h["value"].quantile(.1), 2),
                        "med": r(h["value"].median(), 2), "p90": r(h["value"].quantile(.9), 2),
                        "max": r(h["value"].max(), 2),
                        "y2026": r(cur.iloc[0], 2) if len(cur) else None})
    return v, env, slopes


def outcomes():
    f = pd.read_parquet(PROC / "fewsnet" / "phase_share.parquet")
    f = f[f["pop_unclassified"] <= 0.25]
    i = pd.read_parquet(PROC / "ipcch" / "phase_share.parquet")
    y = pd.read_parquet(PROC / "yields" / "darfur_yield.parquet")
    return f, i, y


def analogues(rain, veg, fn, ic, yl):
    """Pre-war seasons closest to 2026 (lowest detrended greening; driest), and what followed."""
    out = {}
    for s in SIDES:
        v = veg[(veg["side"] == s) & (veg["year"] < WAR_FROM)].dropna(subset=["grn"])
        low = v.nsmallest(N_ANALOGUE, "grn")
        vg = low["year"].astype(int).tolist()
        rr = rain[(rain["side"] == s) & (rain["y"].between(2001, WAR_FROM - 1))]
        dry = rr.nsmallest(N_ANALOGUE, "pct_trend")["y"].tolist()
        rec = {"veg": sorted(vg), "rain": sorted(dry)}
        for key, yrs in (("veg", vg), ("rain", dry)):
            lean = fn[(fn["side"] == s) & (fn["map"] == "lean")].set_index("year")
            rest = [yy for yy in lean.index if yy - 1 not in yrs and yy < WAR_FROM]
            rec[f"fn_lean_{key}"] = {int(yy + 1): r(100 * lean.loc[yy + 1, "share3"])
                                     for yy in yrs if yy + 1 in lean.index}
            rec[f"fn_lean_{key}_others_median"] = r(100 * lean.loc[rest, "share3"].median())
            if s == "darfur":
                yy_ = yl.set_index("year")
                rec[f"yield_{key}"] = {int(a): r(yy_.loc[a, "yield_pct_local"], 0)
                                       for a in yrs if a in yy_.index}
        rec["veg_range"] = [r(low["grn"].min(), 0), r(low["grn"].max(), 0)]
        rb = rain[(rain["side"] == s)].set_index("y")["pct_trend"]
        rec["rain_of_veg"] = {int(a): r(rb.get(a), 0) for a in vg}
        # driest pre-war seasons on the full rainfall record (1981-)
        full = rain[(rain["side"] == s) & (rain["y"] < WAR_FROM)]
        rec["rain_full"] = {int(a): r(b, 0) for a, b in
                            full.nsmallest(N_ANALOGUE, "pct_trend")[["y", "pct_trend"]].values}
        if s == "darfur":
            yy_ = yl.set_index("year")
            rec["yield_rain_full"] = {int(a): r(yy_.loc[a, "yield_pct_local"], 0)
                                      for a in rec["rain_full"] if a in yy_.index}
            rec["yield_2023"] = r(yy_.loc[2023, "yield_pct_local"], 0)
            # driest third of seasons since 1981 (pre-war) and their yields
            z = pd.concat([rain[rain["side"] == s].set_index("y")["pct_trend"],
                           yy_["yield_pct_local"]], axis=1, keys=["x", "y"]).dropna()
            z = z[(z.index < WAR_FROM) & ~z.index.isin(BAD_YIELD)]
            d = z[z["x"] <= z["x"].quantile(1 / 3)]
            rec["dry_third"] = {"n": int(len(d)), "below": int((d["y"] < 100).sum()),
                                "median": r(d["y"].median(), 0), "max_rain": r(d["x"].max(), 0),
                                "drier_than_2026": [int(a) for a in d.index[d["x"] <= rb.get(2026)]]}
        out[s] = rec
    return out


def ranks(rain, veg):
    """2026 rank within 2001-2026 on unrounded values (1 = driest / lowest)."""
    out = {}
    for s in SIDES:
        rr = rain[(rain["side"] == s) & (rain["y"] >= 2001)].set_index("y")
        vv = veg[(veg["side"] == s) & (veg["year"] >= 2001)].set_index("year")
        def rk(ser):
            ser = ser.dropna()
            return int((ser < ser.loc[2026]).sum() + 1)
        out[s] = {"grn": rk(vv["grn"]), "n_grn": int(vv["grn"].notna().sum()),
                  "bio_d": rk(vv["zfparc_crop_d"]), "vhi_d": rk(vv["vhi_d"]),
                  "rain": rk(rr["pct_trend"]), "rain_v2": rk(rr["pct_trend_v2"]),
                  "rain_v3": rk(rr["pct_trend_v3"]), "bio": rk(vv["zfparc_crop"]),
                  "vhi": rk(vv["vhi"]), "n": int(len(rr))}
    return out


def vstat(veg):
    """2026 against the other seasons since 2001 for each vegetation measure: rank, the next
    lowest season and whether 2026 is clear of it. 'level' = within the margin below, or not
    lowest on one of the greening checks (area-weighted, fixed floor). The page words its
    claims from these flags."""
    margin = {"grn": 10.0, "bio_d": 0.2, "vhi_d": 0.02}
    out = {}
    for s in SIDES:
        vv = veg[(veg["side"] == s) & (veg["year"] >= 2001)].set_index("year")
        rec = {}
        col = {"bio_d": "zfparc_crop_d"}
        for k in ("grn", "grn_area", "grn_floor", "grn_raw", "bio_d", "vhi_d", "vhi"):
            x = vv[col.get(k, k)].dropna()
            others = x.drop(2026)
            rec[k] = {"v": r(x.loc[2026], 3), "rank": int((x < x.loc[2026]).sum() + 1),
                      "n": int(len(x)), "next_y": int(others.idxmin()),
                      "next_v": r(others.min(), 3)}
        for k, mg in margin.items():
            gap = rec[k]["next_v"] - rec[k]["v"] if rec[k]["rank"] == 1 else 0
            level = rec[k]["rank"] > 1 or gap < mg
            if k == "grn":
                level = level or rec["grn_area"]["rank"] > 1 or rec["grn_floor"]["rank"] > 1
            rec[k]["level"] = bool(level)
            if level:  # the season it is level with: the closest-lowest other season
                rec[k]["with_y"] = rec[k]["next_y"] if k != "grn" or rec["grn_floor"]["rank"] == 1 \
                    else rec["grn_floor"]["next_y"]
        out[s] = rec
    return out


# 2018 and 2019 report the same harvested area (3,903,000 ha) to the hectare: copied figures, a
# reason that does not depend on the season indicators. Other high years (2014, 2015) stay in;
# rank correlation keeps single outliers from driving r.
BAD_YIELD = (2018, 2019)


def corr_matrices(rain, veg, fn, ic, yl):
    """Pairwise Spearman rank correlation per side, seasons before the war: predictors and yields
    up to the 2022 season, next-lean-season outcomes up to the 2021 season (lean 2022). Used for
    the build guards; the page computes its matrix in the browser from the same series."""
    out = {}
    for s in SIDES:
        t = pd.DataFrame(index=pd.Index(range(1981, 2027), name="y"))
        t["rain"] = rain[rain["side"] == s].set_index("y")["pct_trend"]
        vv = veg[veg["side"] == s].set_index("year")
        # ASI stays undetrended: a straight line through a 0-100% share that sits near zero in
        # most recent years runs below zero by 2025 and turns ordinary years into "stress"
        for k, c in (("grn", "grn"), ("bio", "zfparc_crop_d"), ("vhi", "vhi_d"), ("asi", "asi")):
            t[k] = vv[c]
        if s == "darfur":
            y = yl.set_index("year")
            t["yield"] = y.loc[~y.index.isin(BAD_YIELD), "yield_pct_local"]
        lean = fn[(fn["side"] == s) & (fn["map"] == "lean")].set_index("year")
        t["fn3"] = (100 * lean["share3"]).rename(lambda z: z - 1)
        t["fnm"] = lean["mean_phase"].rename(lambda z: z - 1)
        grp = "residents" if s == "chad" else "all"
        ii = ic[(ic["side"] == s) & (ic["kind"] == "lean") & (ic["group"] == grp)].set_index("year")
        t["ic3"] = (100 * ii["share3"]).rename(lambda z: z - 1)
        for c in t.columns:
            last = 2021 if c in ("fn3", "fnm", "ic3") else 2022
            t.loc[t.index > last, c] = np.nan
        cols = list(t.columns)
        R, N = [], []
        for a in cols:
            R.append([]), N.append([])
            for b in cols:
                if a == b:
                    N[-1].append(int(t[a].notna().sum()))
                    R[-1].append(1.0)
                    continue
                z = t[[a, b]].dropna()
                N[-1].append(int(len(z)))
                R[-1].append(r(z[a].corr(z[b], method="spearman"), 2)
                             if len(z) >= 3 and z[a].std() > 0 and z[b].std() > 0 else None)
        out[s] = {"vars": cols, "r": R, "n": N}
    return out


def geometry():
    a2 = load_adm(2)
    a2["geometry"] = a2.geometry.simplify(0.01, preserve_topology=True)
    a1 = a2.dissolve("adm1_pcode", as_index=False)[["adm1_pcode", "side", "geometry"]]
    a1["geometry"] = a1.geometry.simplify(0.01, preserve_topology=True)
    ctx = []
    for iso in ("sdn", "tcd"):
        g = gpd.read_file(f"zip://{RAW / 'codab' / f'{iso}.shp.zip'}!{iso}_adm0.shp").to_crs(4326)
        ctx.append(g[["geometry"]].assign(iso=iso.upper()))
    ctx = pd.concat(ctx)
    ctx["geometry"] = ctx.geometry.simplify(0.03, preserve_topology=True)

    def gj(df, props):
        df = df.copy()
        df["geometry"] = df.geometry.apply(_orient)
        js = json.loads(df[props + ["geometry"]].to_json(drop_id=True))
        for ft in js["features"]:
            ft["geometry"] = _round(ft["geometry"])
        return js
    return gj(a2, ["adm2_pcode", "adm2_name", "adm1_pcode", "side"]), \
        gj(a1, ["adm1_pcode", "side"]), gj(ctx, ["iso"])


def _orient(geom):
    """d3-geo wants exterior rings clockwise (the reverse of GeoJSON's RFC 7946 order)."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry.polygon import orient
    if isinstance(geom, Polygon):
        return orient(geom, sign=-1.0)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(g, sign=-1.0) for g in geom.geoms])
    return geom


def _round(geom):
    def rc(c):
        return [rc(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 3),
                                                                            round(c[1], 3)]
    geom["coordinates"] = rc(geom["coordinates"])
    return geom


def main():
    rain, alt, mon = rainfall()
    veg, env, slopes = vegetation()
    fn, ic, yl = outcomes()
    ana = analogues(rain, veg, fn, ic, yl)
    hn = pd.read_parquet(PROC / "hnrp" / "adm2.parquet")
    grid = pd.read_parquet(PROC / "rain" / "grid_2026.parquet")
    grid = grid[(grid["frac_chad"] + grid["frac_darfur"] > 0) | True]
    grid["pct"] = 100 * (grid["chirps_v2_jjas_2026"] / grid["chirps_v2_jjas_clim"]
                         + grid["chirps_v3_jjas_2026"] / grid["chirps_v3_jjas_clim"]) / 2
    a2, a1, ctx = geometry()
    adm1_names = {pc: n for s in SIDES.values() for pc, n in s["adm1"].items()}

    D = {
        "sides": {k: v["label"] for k, v in SIDES.items()},
        "adm1": adm1_names,
        "war_from": WAR_FROM,
        "rain": [{"side": x.side, "y": int(x.y), "pt": r(x.pct_trend), "pn": r(x.pct_norm),
                  "pt2": r(x.pct_trend_v2), "pt3": r(x.pct_trend_v3),
                  "mm2": r(x.mm_v2, 0), "mm3": r(x.mm_v3, 0)} for x in rain.itertuples()],
        "rain_alt": alt,
        "rain_month": mon,
        "adm1_2026": by_adm1(),
        "veg": [{"side": x.side, "y": int(x.year), "grn": r(x.grn, 1), "grn_raw": r(x.grn_raw, 1),
                 "grn_area": r(x.grn_area, 1), "grn_floor": r(x.grn_floor, 1),
                 "bio": r(x.zfparc_crop, 2), "bio_d": r(x.zfparc_crop_d, 2),
                 "bior_d": r(x.zfparc_range_d, 2), "vhi": r(x.vhi, 3), "vhi_d": r(x.vhi_d, 3),
                 "asi": r(x.asi, 1)} for x in veg.itertuples()],
        "veg_slopes": slopes,
        "vstat": vstat(veg),
        "corr": corr_matrices(rain, veg, fn, ic, yl),
        "vmap": veg_map(),
        "veg_dekad": env,
        "yield": [{"y": int(x.year), "pct": r(x.yield_pct_local, 0),
                   "pct_src": r(x.yield_pct_src_median, 0),
                   "tha": r(x.yield_t_ha, 2), "src": x.source, "n": int(x.n_states)}
                  for x in yl.itertuples()],
        "fews": [{"side": x.side, "y": int(x.year), "map": x.map, "round": x.round,
                  "layer": x.layer, "s3": r(100 * x.share3), "s4": r(100 * x.share4),
                  "mp": r(x.mean_phase, 2)} for x in fn.itertuples()],
        "ipcch": [{"side": x.side, "y": int(x.year), "kind": x.kind, "src": x.source,
                   "grp": x.group, "s3": r(100 * x.share3), "s4": r(100 * x.share4),
                   "s5": r(100 * x.share5), "pop": int(x.pop), "an": x.analysis,
                   "per": x.period, "flag": (x.flag if isinstance(x.flag, str) else "")}
                  for x in ic.itertuples()],
        "analogues": ana,
        "ranks": ranks(rain, veg),
        "hnrp": [{"pc": x.adm2_pcode, "name": x.adm2_name, "a1": x.adm1_name,
                  "side": x.side, "s25": r(x.sev_2025, 0), "s26": r(x.sev_2026, 0),
                  "pin": r(x.pin_2025, 0), "pop": r(x.pop_2025, 0)} for x in hn.itertuples()],
        "grid": [{"lat": x.lat, "lon": x.lon, "belt": bool(x.belt), "pct": r(x.pct, 0)}
                 for x in grid.itertuples() if np.isfinite(x.pct)],
        "geo": {"adm2": a2, "adm1": a1, "ctx": ctx},
    }
    OUT.write_text(json.dumps(D, separators=(",", ":"), ensure_ascii=False))
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB")
    print(json.dumps(ana, indent=1))
    print(rain[rain["y"] == 2026].round(1).to_string())


if __name__ == "__main__":
    main()
