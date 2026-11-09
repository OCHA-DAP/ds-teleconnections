"""Fractional-coverage weights of CHIRPS 0.05° cells per Haiti department and national."""
import sys, numpy as np, xarray as xr, geopandas as gpd, pandas as pd
from shapely.geometry import box
nc, bdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ds = xr.open_dataset(nc, decode_times=False)
xs, ys = ds["X"].values.astype(float), ds["Y"].values.astype(float)
dx, dy = abs(xs[1]-xs[0]), abs(ys[1]-ys[0])
cells = gpd.GeoDataFrame({"ix": np.repeat(np.arange(len(xs)), len(ys)), "iy": np.tile(np.arange(len(ys)), len(xs))},
    geometry=[box(x-dx/2, y-dy/2, x+dx/2, y+dy/2) for x in xs for y in ys], crs=4326)
a1 = gpd.read_file(f"{bdir}/hti_adm1.gpkg")[["ADM1_PCODE","ADM1_FR","geometry"]]
a0 = gpd.read_file(f"{bdir}/hti_adm0.gpkg"); a0 = a0.assign(ADM1_PCODE="HT", ADM1_FR="Haïti")[["ADM1_PCODE","ADM1_FR","geometry"]]
units = pd.concat([a0, a1], ignore_index=True)
ea = 6933  # equal-area CRS for area weights
ov = gpd.overlay(cells.to_crs(ea), units.to_crs(ea), how="intersection", keep_geom_type=True)
ov["w"] = ov.area
ov = ov[["ix","iy","ADM1_PCODE","ADM1_FR","w"]]
ov["X"] = xs[ov.ix]; ov["Y"] = ys[ov.iy]
ov.to_parquet(out); print(ov.groupby("ADM1_PCODE").agg(n=("w","size"), km2=("w", lambda s: s.sum()/1e6)).round(0))
