"""National area means (GT, HN, SV) for NMME prec (per member/lead/start), CHIRPS and ERA5; tidy Nino3.4.

Usage: uv run python -I 02_areamean.py <nmme_dir> <bounds_dir> <chirps_monthly.nc> <era5.parquet|->
  bounds_dir holds {gtm,hnd,slv}_adm0.gpkg; pass "-" for ERA5 to skip it.
Writes <nmme_dir>/out/{weights_nmme.csv, cells_per_country.csv, obs_monthly.csv,
                       nmme_prec_members.parquet, nino34_raw.parquet}
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

D = Path(sys.argv[1])
RAW, OUT = D / "raw", D / "out"
OUT.mkdir(exist_ok=True)
BDIR, CHIRPS_NC, ERA5_PQ = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
EA = "+proj=laea +lat_0=15 +lon_0=-87 +datum=WGS84"  # equal-area, centred on Central America
COUNTRIES = {"GT": "gtm", "HN": "hnd", "SV": "slv"}
POLY = {}
for cc, iso in COUNTRIES.items():
    g = gpd.read_file(BDIR / f"{iso}_adm0.gpkg").to_crs(EA)
    POLY[cc] = g.geometry.union_all().simplify(100)  # 100 m tolerance (keeps CHIRPS overlay cheap)


def s_to_ym(s):
    s = np.floor(np.asarray(s, dtype=float) + 1e-6).astype(int)
    return 1960 + s // 12, s % 12 + 1


def cell_weights(lon, lat, poly_ea):
    """2-D arrays (lat, lon): intersect area (km2) of each cell with the country, and cell area (km2)."""
    lon = np.asarray(lon, float)
    lon = np.where(lon > 180, lon - 360, lon)
    lat = np.asarray(lat, float)
    dx = np.median(np.abs(np.diff(np.sort(lon)))) if lon.size > 1 else 1.0
    dy = np.median(np.abs(np.diff(np.sort(lat)))) if lat.size > 1 else 1.0
    LO, LA = np.meshgrid(lon, lat)
    boxes = shapely.box(LO - dx / 2, LA - dy / 2, LO + dx / 2, LA + dy / 2)
    gs = gpd.GeoSeries(boxes.ravel(), crs=4326).to_crs(EA)
    inter = gs.intersection(poly_ea).area.values.reshape(LO.shape) / 1e6
    carea = gs.area.values.reshape(LO.shape) / 1e6
    return inter, carea


def area_means(da, poly_ea, xdim="X", ydim="Y"):
    """Fractional-area-weighted mean and all-touched (cell-area-weighted) mean."""
    inter, carea = cell_weights(da[xdim].values, da[ydim].values, poly_ea)
    co = {ydim: da[ydim], xdim: da[xdim]}
    w_frac = xr.DataArray(inter, dims=(ydim, xdim), coords=co)
    w_touch = xr.DataArray((inter > 0).astype(float) * carea, dims=(ydim, xdim), coords=co)
    m_frac = (da * w_frac).sum([xdim, ydim], min_count=1) / w_frac.where(da.notnull()).sum([xdim, ydim])
    m_touch = (da * w_touch).sum([xdim, ydim], min_count=1) / w_touch.where(da.notnull()).sum([xdim, ydim])
    return m_frac, m_touch, inter, carea


DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# ---------------- NMME prec ----------------
rows, wrows, grids = [], [], {}
for f in sorted(RAW.glob("prec_*.nc")):
    model, seg = f.stem.split("_")[1], f.stem.split("_")[2]
    ds = xr.open_dataset(f, decode_times=False)
    da = ds["prec"]
    units = da.attrs.get("units")
    yy, mm = s_to_ym(da.S.values)
    assert (mm == 11).all(), (f, mm)
    lon = np.where(da.X.values > 180, da.X.values - 360, da.X.values)
    grids[(model, seg)] = (tuple(lon), tuple(da.Y.values))
    for cc, poly in POLY.items():
        m_frac, m_touch, inter, carea = area_means(da, poly)
        if seg in ("H", "A"):  # one weights table per model
            for j, la in enumerate(da.Y.values):
                for i, lo in enumerate(lon):
                    if inter[j, i] > 0:
                        wrows.append(dict(model=model, country=cc, lon=lo, lat=la, country_km2=inter[j, i],
                                          cell_km2=carea[j, i], frac=inter[j, i] / carea[j, i]))
        for name, arr in (("frac", m_frac), ("touch", m_touch)):
            df = arr.to_dataframe(name="v").reset_index()
            rows.append(df.assign(method=name, model=model, seg=seg, country=cc, units=units))
    print(f.name, units, "X", lon.min(), "..", lon.max(), "Y", da.Y.values.min(), "..", da.Y.values.max(),
          "L", da.L.values.tolist(), "M", da.M.size, "S", int(yy[0]), int(yy[-1]), flush=True)
if len(set(grids.values())) != 1:
    print("NOTE: NMME grids differ between files:", {k: (v[0][:2], v[1][:2]) for k, v in grids.items()})

pr = pd.concat(rows)
pr["start_year"], _ = s_to_ym(pr["S"].values)
pr["lead_m"] = np.floor(pr["L"]).astype(int)  # target month: Nov start + floor(L) months
pr["target_month"] = (11 - 1 + pr["lead_m"]) % 12 + 1
pr["target_year"] = pr["start_year"] + (11 - 1 + pr["lead_m"]) // 12
assert (pr["units"] == "mm/day").all(), pr["units"].unique()
pr["prec_mm"] = pr["v"] * pr["target_month"].map(DAYS)
pr = pr.rename(columns={"M": "member"})
pr = pr[["country", "model", "seg", "method", "start_year", "member", "L", "target_year", "target_month",
         "prec_mm"]]
pr.to_parquet(OUT / "nmme_prec_members.parquet", index=False)
w = pd.DataFrame(wrows)
w.to_csv(OUT / "weights_nmme.csv", index=False)
cells = (w[w.model == w.model.iloc[0]].groupby("country")
         .agg(n_cells_touched=("frac", "size"), eff_cells=("frac", "sum"), country_km2=("country_km2", "sum"),
              max_cell_frac=("frac", "max")))
cells.to_csv(OUT / "cells_per_country.csv")
print("\nNMME 1-degree cells touched per country (eff_cells = sum of covered fractions):")
print(cells.round(2).to_string())
print(w[w.model == w.model.iloc[0]].assign(frac=lambda x: x.frac.round(3))
      [["country", "lon", "lat", "frac"]].sort_values(["country", "lat", "lon"]).to_string(index=False))

# ---------------- CHIRPS ----------------
ch = xr.open_dataset(CHIRPS_NC, decode_times=False)["precipitation"]
print("\nCHIRPS units", ch.attrs.get("units"), dict(ch.sizes), "X", float(ch.X.min()), float(ch.X.max()),
      "Y", float(ch.Y.min()), float(ch.Y.max()))
cy, cm = s_to_ym(ch["T"].values)
obs = []
for cc, poly in POLY.items():
    c_frac, c_touch, cinter, _ = area_means(ch, poly)
    print(cc, "CHIRPS cells touched", int((cinter > 0).sum()), "country km2 covered", round(cinter.sum()))
    obs.append(pd.DataFrame({"country": cc, "year": cy, "month": cm,
                             "chirps_mm": c_frac.values, "chirps_touch_mm": c_touch.values}))
obs = pd.concat(obs)

# ---------------- ERA5 (national rows) ----------------
if ERA5_PQ != "-" and Path(ERA5_PQ).exists():
    e = pd.read_parquet(ERA5_PQ)
    e = e[(e["adm_level"] == 0) & e["pcode"].isin(COUNTRIES)]
    e = pd.DataFrame({"country": e.pcode.values, "year": e.valid_date.dt.year.values,
                      "month": e.valid_date.dt.month.values, "era5_mmday": e["mean"].values})
    e["era5_mm"] = e["era5_mmday"] * e["month"].map(DAYS)  # Feb leap days ignored (Feb not used)
    obs = obs.merge(e[["country", "year", "month", "era5_mm"]], on=["country", "year", "month"], how="outer")
else:
    print("ERA5 not supplied: ERA5 columns left empty")
    obs["era5_mm"] = np.nan
obs = obs.sort_values(["country", "year", "month"])
obs.to_csv(OUT / "obs_monthly.csv", index=False)
print(obs.groupby("country").tail(2).to_string(index=False))

# ---------------- SST box means (Nino3.4, TNA) raw ----------------
for box in ("nino34", "tna"):
    nrows = []
    for f in sorted(RAW.glob(f"{box}_*.nc")):
        model, seg = f.stem.split("_")[1], f.stem.split("_")[2]
        ds = xr.open_dataset(f, decode_times=False)
        da = ds["sst"]
        df = da.to_dataframe(name="sst").reset_index()
        df["start_year"], smon = s_to_ym(df["S"].values)
        assert (smon == 11).all()
        df = df.rename(columns={"M": "member"}).assign(model=model, seg=seg, units=da.attrs.get("units"))
        nrows.append(df[["model", "seg", "start_year", "member", "L", "sst", "units"]])
        print(f.name, da.attrs.get("units"), dict(da.sizes), "L", float(da.L.min()), "-", float(da.L.max()),
              "n NaN", int(da.isnull().sum()), flush=True)
    if nrows:
        pd.concat(nrows).to_parquet(OUT / f"{box}_raw.parquet", index=False)
    else:
        print(f"no {box} files")

# ---------------- TNA land-mask check: one ungridded field vs the server-side box mean ----------------
for f in sorted(RAW.glob("tnagrid_*.nc")):
    model = f.stem.split("_")[1]
    g = xr.open_dataset(f, decode_times=False)["sst"].squeeze()
    nan = int(g.isnull().sum())
    lat_w = np.cos(np.deg2rad(g.Y))
    seg = [p for p in RAW.glob(f"tna_{model}_*.nc")]
    bm = np.nan
    for p in seg:
        t = xr.open_dataset(p, decode_times=False)["sst"]
        yy, _ = s_to_ym(t.S.values)
        if 2026 in yy:
            bm = float(t.isel(S=list(yy).index(2026)).sel(L=0.5, M=g.M.item()))
    print(f"TNA grid {model}: {g.sizes['X']}x{g.sizes['Y']} cells, NaN (land) cells {nan}; local unweighted mean "
          f"{float(g.mean()):.3f} (min {float(g.min()):.2f}, max {float(g.max()):.2f}), cos-lat weighted {float(g.weighted(lat_w).mean()):.3f}; server box mean {bm:.3f}",
          flush=True)
