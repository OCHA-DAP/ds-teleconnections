"""July–October rainfall from CHIRPS v2 (CHC COGs fetched by chc_window.py) over Tonga's and Fiji's land, next to ERA5.

usage: chc_units.py SP
Reads SP/data/chc/{ton,fji_e,fji_w}_jaso.npz, SP/data/{ton,fji}_adm0.gpkg and SP/data/era5.parquet; writes
SP/data/chirps_jaso.json: per unit, the CHIRPS July–October % of 1991–2020 by year, the 2026 rank (driest first) and the
correlation with ERA5's July–October totals over 1981–2025.
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy

SP = sys.argv[1].rstrip("/") + "/"
D = SP + "data/"
e = pd.read_parquet(D + "era5.parquet")
DAYS = {7: 31, 8: 31, 9: 30, 10: 31}


def unit_series(files, iso):
    geom = gpd.read_file(D + f"{iso}_adm0.gpkg").geometry.union_all()
    tot, ncell = None, 0
    for f in files:
        z = np.load(D + "chc/" + f)
        xx, yy = np.meshgrid(z["x"], z["y"])
        m = contains_xy(geom, xx, yy) & np.isfinite(z["data"]).all(axis=0)
        ncell += int(m.sum())
        s = pd.Series(np.nansum(z["data"][:, m], axis=1), index=pd.PeriodIndex(z["ym"], freq="M"))
        tot = s if tot is None else tot.add(s)
    mean = tot / ncell  # mm/month averaged over land cells
    g = mean.groupby(mean.index.year).agg(["sum", "size"])
    return g[g["size"] == 4]["sum"], ncell


def era5_jaso(pc):
    ee = e[(e.pcode == pc) & e.valid_date.dt.month.isin(DAYS)].copy()
    ee["mm"] = ee["mean"] * ee.valid_date.dt.month.map(DAYS)
    g = ee.groupby(ee.valid_date.dt.year).agg(mm=("mm", "sum"), n=("mm", "size"))
    return g[g.n == 4].mm


out = {}
for unit, iso, files in [("TO", "ton", ["ton_jaso.npz"]), ("FJ", "fji", ["fji_e_jaso.npz", "fji_w_jaso.npz"])]:
    if not all(os.path.exists(D + "chc/" + f) for f in files):
        print(unit, "skipped: CHIRPS files missing")
        continue
    c, n = unit_series(files, iso)
    era = era5_jaso(unit)
    pct = c / c.loc[1991:2020].mean() * 100
    epct = era / era.loc[1991:2020].mean() * 100
    j = pd.concat([c.rename("c"), era.rename("e")], axis=1).dropna().loc[1981:2025]
    out[unit] = {"n_cells": n, "r_era5": round(float(j.corr().iloc[0, 1]), 2),
                 "pct": {int(y): round(float(v), 1) for y, v in pct.items()},
                 "rank2026": int((c < c.loc[2026]).sum() + 1), "n": int(len(c))}
    print(unit, "cells", n, "| r(CHIRPS, ERA5) Jul-Oct 1981-2025:", out[unit]["r_era5"],
          "| 2026 CHIRPS", round(pct.loc[2026]), "% rank", out[unit]["rank2026"], "of", len(c),
          "| ERA5", round(epct.loc[2026]), "% | 2015 CHIRPS", round(pct.loc[2015]), "% ERA5", round(epct.loc[2015]), "%")
json.dump(out, open(D + "chirps_jaso.json", "w"))
