"""Haiti national area means for NMME prec (per member/lead/start), CHIRPS and ERA5; tidy Nino3.4.

Usage: uv run python -I 02_areamean.py <nmme_agent_dir> <hti_adm0.gpkg> <hti_era5.parquet>
Writes <dir>/out/{weights_nmme.csv, obs_monthly.csv, nmme_prec_members.parquet, nino34_raw.parquet}
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
poly = gpd.read_file(sys.argv[2]).to_crs(4326).geometry.union_all()
EA = "+proj=laea +lat_0=19 +lon_0=-73 +datum=WGS84"  # local equal-area
poly_ea = gpd.GeoSeries([poly], crs=4326).to_crs(EA).iloc[0]


def s_to_ym(s):
    s = np.floor(np.asarray(s, dtype=float) + 1e-6).astype(int)
    return 1960 + s // 12, s % 12 + 1


def cell_weights(lon, lat):
    """Return 2-D arrays (lat, lon): intersect area (km2) of each cell with Haiti, cell area."""
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


def area_means(da, xdim="X", ydim="Y"):
    """Fractional-area-weighted mean and all-touched (cell-area-weighted) mean."""
    inter, carea = cell_weights(da[xdim].values, da[ydim].values)
    w_frac = xr.DataArray(inter, dims=(ydim, xdim), coords={ydim: da[ydim], xdim: da[xdim]})
    touch = (inter > 0).astype(float) * carea
    w_touch = xr.DataArray(touch, dims=(ydim, xdim), coords={ydim: da[ydim], xdim: da[xdim]})
    m_frac = (da * w_frac).sum([xdim, ydim], min_count=1) / w_frac.where(da.notnull()).sum([xdim, ydim])
    m_touch = (da * w_touch).sum([xdim, ydim], min_count=1) / w_touch.where(da.notnull()).sum([xdim, ydim])
    return m_frac, m_touch, inter, carea


DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# ---------------- NMME prec ----------------
rows, wrows = [], []
for f in sorted(RAW.glob("prec_*.nc")):
    model, seg = f.stem.split("_")[1], f.stem.split("_")[2]
    ds = xr.open_dataset(f, decode_times=False)
    da = ds["prec"]
    units = da.attrs.get("units")
    m_frac, m_touch, inter, carea = area_means(da)
    if not wrows or wrows[-1]["model"] != model:
        lon = np.where(da.X.values > 180, da.X.values - 360, da.X.values)
        for j, la in enumerate(da.Y.values):
            for i, lo in enumerate(lon):
                if inter[j, i] > 0:
                    wrows.append(dict(model=model, lon=lo, lat=la, haiti_km2=inter[j, i],
                                      cell_km2=carea[j, i], frac=inter[j, i] / carea[j, i]))
    yy, mm = s_to_ym(da.S.values)
    assert (mm == 11).all(), (f, mm)
    for name, arr in (("frac", m_frac), ("touch", m_touch)):
        df = arr.to_dataframe(name="v").reset_index()
        df["method"] = name
        rows.append(df.assign(model=model, seg=seg, units=units))
    print(f.name, units, "X", da.X.values, "Y", da.Y.values, "M", da.M.size,
          "S", int(yy[0]), int(yy[-1]), flush=True)

pr = pd.concat(rows)
pr["start_year"], _ = s_to_ym(pr["S"].values)
# target month: Nov start + floor(L) months
pr["lead_m"] = np.floor(pr["L"]).astype(int)
pr["target_month"] = (11 - 1 + pr["lead_m"]) % 12 + 1
pr["target_year"] = pr["start_year"] + (11 - 1 + pr["lead_m"]) // 12
assert (pr["units"] == "mm/day").all(), pr["units"].unique()
pr["prec_mm"] = pr["v"] * pr["target_month"].map(DAYS)
pr = pr.rename(columns={"M": "member"})
pr = pr[["model", "seg", "method", "start_year", "member", "L", "target_year", "target_month", "prec_mm"]]
pr.to_parquet(OUT / "nmme_prec_members.parquet", index=False)
pd.DataFrame(wrows).to_csv(OUT / "weights_nmme.csv", index=False)

# ---------------- CHIRPS ----------------
ch = xr.open_dataset(RAW / "chirps_hti_monthly.nc", decode_times=False)["precipitation"]
print("CHIRPS units", ch.attrs.get("units"), ch.sizes)
c_frac, c_touch, cinter, _ = area_means(ch)
cy, cm = s_to_ym(ch["T"].values)
obs = pd.DataFrame({"year": cy, "month": cm, "chirps_mm": c_frac.values, "chirps_touch_mm": c_touch.values})

# ---------------- ERA5 ----------------
e = pd.read_parquet(sys.argv[3])
e = e[e["adm_level"] == 0]
e = pd.DataFrame({"year": e.valid_date.dt.year, "month": e.valid_date.dt.month, "era5_mmday": e["mean"].values})
e["era5_mm"] = e["era5_mmday"] * e["month"].map(DAYS)  # Feb leap-days ignored (Feb not used)
obs = obs.merge(e[["year", "month", "era5_mm"]], on=["year", "month"], how="outer").sort_values(["year", "month"])
obs.to_csv(OUT / "obs_monthly.csv", index=False)
print(obs.tail(3))

# ---------------- Nino3.4 raw ----------------
nrows = []
for f in sorted(RAW.glob("nino34_*.nc")):
    model, seg = f.stem.split("_")[1], f.stem.split("_")[2]
    ds = xr.open_dataset(f, decode_times=False)
    da = ds["sst"]
    df = da.to_dataframe(name="sst").reset_index()
    df["start_year"], _ = s_to_ym(df["S"].values)
    df = df.rename(columns={"M": "member"}).assign(model=model, seg=seg, units=da.attrs.get("units"))
    nrows.append(df[["model", "seg", "start_year", "member", "L", "sst", "units"]])
    print(f.name, da.attrs.get("units"), da.sizes, flush=True)
pd.concat(nrows).to_parquet(OUT / "nino34_raw.parquet", index=False)
