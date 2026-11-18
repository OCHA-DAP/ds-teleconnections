"""0.5-degree cells of the Sudano-Sahelian belt for the pooled conflict test: mask, country, rain, greening.

Design in <work>/processed/belt/prespec_2026-11-18.md (written before any event-indicator join).

Cells: the CHIRPS 0.5 deg grid over BELT_BBOX, assigned to the country and admin-1 unit (team CODAB)
holding most of the cell's land, in the twelve belt countries. Kept if the CHIRPS v2 1991-2020
monthly climatology has June-September 100-1200 mm, its wettest month in July-September and March-May
at most 25% of the year: one rainy season, so the greening recipe's dry floor and season hold.

Per cell and year:
  rain         CHIRPS v2 June-September total, % of the cell's 1981-2025 linear trend
  rain_up      the same over belt cells UP_LAT deg north of the cell, within UP_LON deg of longitude
  grn          MODIS greening (08_ndvi.py belt: 4 km), recipe of 09_greening.py per cell: each pixel
               divided by its 2001-2024 mean, cell mean over vegetated pixels, % of the cell's own
               2001-2024 line; cells need MIN_PIX vegetated pixels
  grn_up       the same over vegetated pixels in the upstream band

Writes <work>/processed/belt/cells.parquet and cell_years.parquet.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW  # noqa: E402

OUT = PROC / "belt"
ISO3 = ["MRT", "SEN", "GMB", "MLI", "BFA", "NER", "NGA", "CMR", "CAF", "SSD", "TCD", "SDN"]
BELT_BBOX = (-17.75, 3.75, 38.25, 18.75)   # 0.5 deg cell CENTRES (CHIRPS grid); edges are +-0.25
JJAS_RANGE = (100.0, 1200.0)
MAM_MAX_SHARE = 0.25
MIN_PIX = 25          # 4 km pixels (about 400 km2)
UP_LAT, UP_LON = (0.5, 2.5), 1.0
BASE = (2001, 2024)
VALID_PCT = (0.0, 300.0)


def trend_pct(years, s, base):
    m = (years >= base[0]) & (years <= base[1]) & np.isfinite(s)
    if m.sum() < 10:
        return np.full(len(years), np.nan)
    return 100 * s / np.polyval(np.polyfit(years[m], s[m], 1), years)


def adm1():
    from ocha_stratus import codab
    out = []
    for iso in ISO3:
        g = codab.load_codab_from_blob(iso.lower(), admin_level=1).to_crs(4326)
        name = next(c for c in ("ADM1_EN", "ADM1_FR", "ADM1_REF") if c in g)
        g = g[["ADM1_PCODE", name, "geometry"]].rename(columns={"ADM1_PCODE": "adm1_pcode", name: "adm1_name"})
        g["iso3"] = iso
        out.append(g)
    a = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=4326)
    a["geometry"] = a.geometry.simplify(0.01)
    return a


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = BELT_BBOX
    ds = xr.open_dataset(RAW / "belt" / "chirps_v2_grid05.nc").sortby("lat")
    pr = ds["precip"].sel(lat=slice(y0, y1), lon=slice(x0, x1)).load()
    lat, lon = pr.lat.values, pr.lon.values
    mon = pr.sel(time=slice("1991-01-01", "2020-12-31")).groupby("time.month").mean()
    jjas = mon.sel(month=[6, 7, 8, 9]).sum("month").values
    ann = mon.sum("month").values
    mam = mon.sel(month=[3, 4, 5]).sum("month").values
    wet = mon.month.values[np.argmax(np.nan_to_num(mon.values, nan=-1.0), axis=0)]  # sea cells: no rain data
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cells = pd.DataFrame({"lat": LA.ravel(), "lon": LO.ravel(), "jjas_clim": jjas.ravel(),
                          "mam_share": (mam / np.where(ann > 0, ann, np.nan)).ravel(), "wettest": wet.ravel()})
    cells["cell"] = [f"{a:+06.2f}_{b:+07.2f}" for a, b in zip(cells["lat"], cells["lon"])]

    # country and admin-1 holding most of the cell's land
    a = adm1()
    g = gpd.GeoDataFrame(cells[["cell"]], geometry=[box(b - .25, a_ - .25, b + .25, a_ + .25)
                                                    for a_, b in zip(cells["lat"], cells["lon"])], crs=4326)
    ov = gpd.overlay(g, a, how="intersection", keep_geom_type=True)
    ov["area"] = ov.geometry.area
    # the country with most land in the cell, then its largest admin-1 piece
    ctry = ov.groupby(["cell", "iso3"])["area"].sum().reset_index().sort_values("area").groupby("cell").tail(1)
    ov = ov.merge(ctry[["cell", "iso3"]], on=["cell", "iso3"])
    best = ov.sort_values("area").groupby("cell").tail(1).set_index("cell")
    land = gpd.overlay(g, a, how="intersection", keep_geom_type=True).assign(
        area=lambda d: d.geometry.area).groupby("cell")["area"].sum()
    cells = cells.join(best[["iso3", "adm1_pcode", "adm1_name"]], on="cell")
    cells["land_share"] = cells["cell"].map(land / 0.25).fillna(0)
    cells["in_belt"] = (cells["iso3"].notna() & (cells["land_share"] >= 0.5)
                        & cells["jjas_clim"].between(*JJAS_RANGE) & cells["wettest"].isin([7, 8, 9])
                        & (cells["mam_share"] <= MAM_MAX_SHARE))

    # rain: June-September total per year, % of the cell's 1981-2025 trend; upstream band mean
    yrs = np.arange(1981, 2026)
    jj = np.stack([pr.sel(time=[f"{y}-{m:02d}-01" for m in (6, 7, 8, 9)]).sum("time").values for y in yrs])
    rain = np.full(jj.shape, np.nan)
    rain_up = np.full(jj.shape, np.nan)
    belt100 = jjas >= JJAS_RANGE[0]
    for i in range(len(lat)):
        for j in range(len(lon)):
            rain[:, i, j] = trend_pct(yrs, jj[:, i, j], (1981, 2025))
            band = ((lat >= lat[i] + UP_LAT[0]) & (lat <= lat[i] + UP_LAT[1]))[:, None] & \
                   (np.abs(lon - lon[j]) <= UP_LON)[None, :] & belt100
            if band.any():
                rain_up[:, i, j] = trend_pct(yrs, jj[:, band].mean(1), (1981, 2025))

    # greening: pixels (0.04 deg) -> cells, recipe of 09_greening.py
    z = np.load(PROC / "belt" / "greening.npz")
    gyears, glat, glon, gr = z["years"], z["lat"], z["lon"], z["greening"]
    base = (gyears >= BASE[0]) & (gyears <= BASE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        gmean = np.nanmean(gr[base], axis=0)
        norm = gr / gmean
    ok = (gmean >= 0.03) & (np.isfinite(gr[base]).sum(0) >= 20)
    ci = np.floor((glat - (y0 - 0.25)) / 0.5).astype(int)   # pixel -> cell row/col, from the cells' true edges
    cj = np.floor((glon - (x0 - 0.25)) / 0.5).astype(int)
    cell_belt100 = belt100[np.clip(ci, 0, len(lat) - 1)][:, np.clip(cj, 0, len(lon) - 1)]
    ok &= cell_belt100
    norm[:, ~ok] = np.nan
    PLA, PLO = np.meshgrid(glat, glon, indexing="ij")
    grn = np.full((len(gyears), len(lat), len(lon)), np.nan)
    grn_up = np.full_like(grn, np.nan)
    npix = np.zeros((len(lat), len(lon)), int)
    rows_of = {i: (ci == i) for i in range(len(lat))}
    cols_of = {j: (cj == j) for j in range(len(lon))}
    for i in range(len(lat)):
        rmask = rows_of[i]
        if not rmask.any():
            continue
        uprows = (glat >= lat[i] + UP_LAT[0]) & (glat <= lat[i] + UP_LAT[1])
        for j in range(len(lon)):
            m = np.outer(rmask, cols_of[j]) & ok
            npix[i, j] = int(m.sum())
            if npix[i, j] >= MIN_PIX:
                s = np.nanmean(norm[:, m], axis=1)
                grn[:, i, j] = trend_pct(gyears, s, BASE)
            u = np.outer(uprows, np.abs(glon - lon[j]) <= UP_LON) & ok
            if u.sum() >= MIN_PIX:
                grn_up[:, i, j] = trend_pct(gyears, np.nanmean(norm[:, u], axis=1), BASE)
    cells["npix"] = npix.ravel()
    # a % of trend is meaningless where the line runs near zero (sparse vegetation on the dry edge):
    # drop a cell's series if any year falls outside VALID_PCT (rule fixed before any join)
    for arr, flag in ((grn, "grn_ok"), (grn_up, "grn_up_ok")):
        badc = np.any((arr < VALID_PCT[0]) | (arr > VALID_PCT[1]), axis=0)
        arr[:, badc] = np.nan
        cells[flag] = ~badc.ravel()

    rows = []
    for k, y in enumerate(sorted(set(yrs) | set(gyears.tolist()))):
        r = {"year": int(y), "cell": cells["cell"].values}
        yi = np.where(yrs == y)[0]
        gi = np.where(gyears == y)[0]
        r["rain"] = rain[yi[0]].ravel() if len(yi) else np.nan
        r["rain_up"] = rain_up[yi[0]].ravel() if len(yi) else np.nan
        r["grn"] = grn[gi[0]].ravel() if len(gi) else np.nan
        r["grn_up"] = grn_up[gi[0]].ravel() if len(gi) else np.nan
        rows.append(pd.DataFrame(r))
    cy = pd.concat(rows, ignore_index=True)
    keep = set(cells.loc[cells["in_belt"], "cell"])
    cy = cy[cy["cell"].isin(keep)]
    cells.to_parquet(OUT / "cells.parquet", index=False)
    cy.to_parquet(OUT / "cell_years.parquet", index=False)
    a.to_parquet(OUT / "adm1.parquet", index=False)
    kb = cells[cells["in_belt"]]
    print(f"[belt] {len(kb)} cells in the belt; with greening: {(kb['npix'] >= MIN_PIX).sum()}")
    print(kb.groupby("iso3").size().to_dict())


if __name__ == "__main__":
    main()
