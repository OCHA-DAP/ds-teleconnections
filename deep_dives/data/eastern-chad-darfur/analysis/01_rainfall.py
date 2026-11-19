"""Rainfall per side and per state, 1981-2026, on the 0.5 deg grid.

History comes from the packed 0.5 deg cubes of ds-precip-intercomparison
(CHIRPS v2/v3, ERA5, IMERG Late; cache/packed/*.npz, 1981 - Aug 2026). Months the
cache lacks (CHIRPS Sep-Oct 2026, and CHIRPS v3 Aug 2026, which CHC revised after the
cache was built) are read from the CHC COGs one at a time with a pause (CHC rate-limits
parallel reads), block-averaged 10x10 from 0.05 deg to the same 0.5 deg grid.

Writes <work>/processed/rain/:
  monthly_side.parquet   product, side, month, mm   (belt-masked, area-weighted)
  monthly_adm1.parquet   product, adm1_pcode, month, mm
  grid_2026.parquet      cell lat/lon, side fractions, belt flag, JJAS 2026 mm and
                         the 1991-2020 JJAS mean, per CHIRPS version (for the map)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import BBOX, BELT_MIN_JJAS_MM, PROC, RAW, load_adm  # noqa: E402

PACKED = Path.home() / "OCHA/repos/ds-precip-intercomparison/cache/packed"
CHC = "https://data.chc.ucsb.edu/products/CHIRPS"
COG = {
    "chirps_v2": CHC + "/v2.0/global_monthly/cogs/chirps-v2.0.{y}.{m:02d}.cog",
    "chirps_v3": CHC + "/v3.0/monthly/global/cogs/chirps-v3.0.{y}.{m:02d}.cog",
}
# (product, month) to read from CHC instead of the cache
FETCH = [
    ("chirps_v3", "2026-08"),
    ("chirps_v2", "2026-09"),
    ("chirps_v3", "2026-09"),
    ("chirps_v2", "2026-10"),
    ("chirps_v3", "2026-10"),
]
PRODUCTS = ["chirps_v2", "chirps_v3", "era5", "imerg_late"]
RES = 0.5
OUT = PROC / "rain"


def grid():
    x0, y0, x1, y1 = BBOX
    lon = np.arange(x0 + RES / 2, x1, RES)
    lat = np.arange(y0 + RES / 2, y1, RES)
    return np.round(lat, 2), np.round(lon, 2)


def fractions(lat, lon, gdf, key):
    """Fraction of each 0.5 deg cell inside each polygon group (rasterised at 0.01 deg)."""
    f = 50
    x0, y0, x1, y1 = BBOX
    tr = from_origin(x0, y1, RES / f, RES / f)
    shape = (lat.size * f, lon.size * f)
    out = {}
    for k, sub in gdf.groupby(key):
        m = rasterize(sub.geometry, out_shape=shape, transform=tr, fill=0, default_value=1)
        m = m.reshape(lat.size, f, lon.size, f).mean(axis=(1, 3))[::-1]  # -> ascending lat
        out[k] = m
    return out


def from_cache(product, lat, lon):
    cells = np.load(PACKED / "cells.npz")
    p = np.load(PACKED / f"{product}.npz")
    t = p["time"]
    cube = np.full((t.size, lat.size, lon.size), np.nan, np.float32)
    iy = np.searchsorted(lat, cells["lat"])
    ix = np.searchsorted(lon, cells["lon"])
    ok = (
        (iy < lat.size)
        & (ix < lon.size)
        & np.isclose(lat[np.minimum(iy, lat.size - 1)], cells["lat"])
        & np.isclose(lon[np.minimum(ix, lon.size - 1)], cells["lon"])
    )
    cube[:, iy[ok], ix[ok]] = p["precip"][:, ok]
    return pd.Index(t), cube


def from_chc(product, ym, lat, lon):
    y, m = map(int, ym.split("-"))
    url = "/vsicurl/" + COG[product].format(y=y, m=m)
    x0, y0, x1, y1 = BBOX
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(url) as src:
            w = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            a = src.read(1, window=w).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
    a[a < 0] = np.nan
    f = 10
    ny, nx = lat.size * f, lon.size * f
    if a.shape != (ny, nx):
        raise ValueError(f"{product} {ym}: window {a.shape} != {(ny, nx)}")
    b = a.reshape(lat.size, f, lon.size, f)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(b, axis=(1, 3))
    return mean[::-1]  # north-up -> ascending lat


def area_mean(cube, w):
    """Weighted mean over cells for each time step; w is (lat, lon), NaN-aware."""
    v = np.isfinite(cube)
    num = np.where(v, cube, 0) * w
    den = v * w
    return num.sum(axis=(1, 2)) / np.maximum(den.sum(axis=(1, 2)), 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lat, lon = grid()
    coslat = np.cos(np.deg2rad(lat))[:, None]
    a1 = load_adm(1)
    f_side = fractions(lat, lon, a1, "side")
    f_adm1 = fractions(lat, lon, a1, "adm1_pcode")

    cubes = {}
    for p in PRODUCTS:
        t, c = from_cache(p, lat, lon)
        cubes[p] = dict(zip(t, c))
    # cached chirps_v3 Aug 2026 is the pre-revision field; replace it
    raw_dir = RAW / "chirps_bbox"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for p, ym in FETCH:
        f = raw_dir / f"{p}_{ym}.npy"
        if not f.exists():
            print(f"[chc] {p} {ym}", flush=True)
            np.save(f, from_chc(p, ym, lat, lon))
            time.sleep(6)
        cubes[p][ym] = np.load(f)
    for p in PRODUCTS:
        cubes[p] = dict(sorted(cubes[p].items()))

    # belt mask from CHIRPS v2 1991-2020 JJAS mean
    s = cubes["chirps_v2"]
    yrs = range(1991, 2021)
    jjas = np.stack(
        [np.sum(np.stack([s[f"{y}-{m:02d}"] for m in (6, 7, 8, 9)]), axis=0) for y in yrs]
    )
    clim = jjas.mean(axis=0)
    belt = clim >= BELT_MIN_JJAS_MM

    rows_side, rows_adm1 = [], []
    for p, ser in cubes.items():
        cube = np.stack(list(ser.values()))
        months = list(ser)
        for side, fr in f_side.items():
            w = fr * coslat * belt
            rows_side.append(pd.DataFrame({"product": p, "side": side, "month": months,
                                           "mm": area_mean(cube, w)}))
        for pc, fr in f_adm1.items():
            w = fr * coslat * belt
            if w.sum() == 0:
                continue
            rows_adm1.append(pd.DataFrame({"product": p, "adm1_pcode": pc, "month": months,
                                           "mm": area_mean(cube, w)}))
    pd.concat(rows_side).to_parquet(OUT / "monthly_side.parquet", index=False)
    pd.concat(rows_adm1).to_parquet(OUT / "monthly_adm1.parquet", index=False)

    # grid for the map
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    g = pd.DataFrame({"lat": lat2.ravel(), "lon": lon2.ravel(), "belt": belt.ravel(),
                      "frac_chad": f_side["chad"].ravel(), "frac_darfur": f_side["darfur"].ravel()})
    for p in ("chirps_v2", "chirps_v3"):
        ser = cubes[p]
        cl = np.mean([sum(ser[f"{y}-{m:02d}"] for m in (6, 7, 8, 9)) for y in yrs], axis=0)
        g[f"{p}_jjas_clim"] = cl.ravel()
        g[f"{p}_jjas_2026"] = sum(ser[f"2026-{m:02d}"] for m in (6, 7, 8, 9)).ravel()
    g.to_parquet(OUT / "grid_2026.parquet", index=False)
    print(f"[rain] belt cells: chad {int((belt * (f_side['chad'] > 0)).sum())}, "
          f"darfur {int((belt * (f_side['darfur'] > 0)).sum())}")


if __name__ == "__main__":
    main()
