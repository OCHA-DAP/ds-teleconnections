"""CHIRPS monthly (IRI Data Library) averaged over Tonga's and Fiji's land cells, as a second rainfall record next to ERA5.

usage: chirps_units.py SP
Reads SP/data/chirps/chirps_{ton,fji_w,fji_e}.nc and SP/data/{ton,fji}_adm0.gpkg; writes SP/data/chirps_units.parquet
(columns: unit, year, obs_pct = November–April total as % of 1991–2020, r_era5) and prints the checks.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy

SP = sys.argv[1].rstrip("/") + "/"
D = SP + "data/"
e = pd.read_parquet(D + "era5.parquet")
DAYS = {11: 30, 12: 31, 1: 31, 2: 28.25, 3: 31, 4: 30}


def chirps_monthly(files, iso):
    geom = gpd.read_file(D + f"{iso}_adm0.gpkg").geometry.union_all()
    parts = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        da = ds["precipitation"]
        t0 = pd.Timestamp("1960-01-01")
        times = [t0 + pd.DateOffset(months=int(np.floor(m))) for m in da["T"].values]
        xx, yy = np.meshgrid(da["X"].values, da["Y"].values)
        mask = contains_xy(geom, xx, yy)
        if mask.sum() == 0:
            continue
        v = da.values[:, mask]  # mm/month
        parts.append(pd.DataFrame(v, index=pd.DatetimeIndex(times)))
    allv = pd.concat(parts, axis=1)
    return allv.mean(axis=1), allv.shape[1]


def nov_apr(m):
    df = m.to_frame("mm")
    df["mo"], df["y"] = df.index.month, df.index.year
    df = df[df.mo.isin(DAYS)]
    df["sy"] = np.where(df.mo >= 11, df.y, df.y - 1)
    g = df.groupby("sy").agg(t=("mm", "sum"), n=("mm", "size"))
    return g[g.n == 6].t


rows = []
for unit, iso, files in [("TO", "ton", [D + "chirps/chirps_ton.nc"]), ("FJ", "fji", [D + "chirps/chirps_fji_w.nc", D + "chirps/chirps_fji_e.nc"])]:
    m, ncells = chirps_monthly(files, iso)
    c = nov_apr(m)
    ee = e[e.pcode == unit].set_index("valid_date")["mean"]
    ee = ee * ee.index.month.map(lambda mo: DAYS.get(mo, 30))
    era = nov_apr(ee)
    j = pd.concat([c.rename("chirps"), era.rename("era5")], axis=1).dropna()
    r = j.corr().iloc[0, 1]
    pct = c / c.loc[1991:2020].mean() * 100
    print(unit, "CHIRPS land cells:", ncells, "| r(CHIRPS, ERA5) Nov-Apr:", round(r, 2),
          "| CHIRPS % 1982/1997/2015/2023:", [round(pct.get(y, np.nan)) for y in (1982, 1997, 2015, 2023)])
    # Jul-Oct 2026 so far
    jaso = m[m.index.month.isin([7, 8, 9, 10])]
    jy = jaso.groupby(jaso.index.year).agg(["sum", "size"])
    jy = jy[jy["size"] == 4]["sum"]
    if 2026 in jy.index:
        print("   Jul-Oct 2026:", round(jy.loc[2026] / jy.loc[1991:2020].mean() * 100), "% of 1991-2020, rank", int((jy < jy.loc[2026]).sum() + 1), "of", len(jy))
    for y, v in pct.items():
        rows.append({"unit": unit, "year": int(y), "obs_pct": round(float(v), 1), "r_era5": round(float(r), 2), "n_cells": int(ncells)})
pd.DataFrame(rows).to_parquet(D + "chirps_units.parquet")
