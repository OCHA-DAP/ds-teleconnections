"""Fractional-coverage weights of CHIRPS 0.05° cells per unit (countries, departments, framework groups).
Usage: weights.py <grid nc> <bounds dir> <out parquet> <scripts dir>"""
import sys, numpy as np, xarray as xr, geopandas as gpd, pandas as pd
from shapely.geometry import box
nc, bdir, out, sdir = sys.argv[1:5]
exec(open(f"{sdir}/units.py").read())
ds = xr.open_dataset(nc, decode_times=False)
xs, ys = ds["X"].values.astype(float), ds["Y"].values.astype(float)
dx, dy = abs(xs[1]-xs[0]), abs(ys[1]-ys[0])
IX, IY = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
cells = gpd.GeoDataFrame({"ix": IX.ravel(), "iy": IY.ravel()},
    geometry=[box(xs[i]-dx/2, ys[j]-dy/2, xs[i]+dx/2, ys[j]+dy/2) for i, j in zip(IX.ravel(), IY.ravel())], crs=4326)
U = load_units(bdir)
ea = "+proj=laea +lat_0=15 +lon_0=-87 +datum=WGS84"
cells_ea = cells.to_crs(ea); U_ea = U.to_crs(ea)
U_ea["geometry"] = U_ea.geometry.simplify(200)
ov = gpd.overlay(cells_ea, U_ea[["unit", "name", "iso3", "level", "geometry"]], how="intersection", keep_geom_type=True)
ov["w"] = ov.area
ov = pd.DataFrame(ov.drop(columns="geometry"))
ov["X"] = xs[ov.ix]; ov["Y"] = ys[ov.iy]
ov.to_parquet(out)
s = ov.groupby(["unit", "name"]).agg(n=("w", "size"), km2=("w", lambda v: v.sum()/1e6)).round(0)
print(s.to_string())
