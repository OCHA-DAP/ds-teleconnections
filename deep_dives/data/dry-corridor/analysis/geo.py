"""Simplified department outlines for the page map, projected to SVG units (equirectangular, cos 15°N),
plus label anchors. Usage: geo.py SP -> geo.json"""
import sys, json, numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import Polygon, MultiPolygon
SP = sys.argv[1]
exec(open(f"{SP}/scripts/units.py").read())
U = load_units(f"{SP}/bounds")
U = U[U.level.isin([0, 1])].copy()
x0, x1, y0, y1 = -92.35, -83.05, 12.9, 17.9
K = np.cos(np.radians(15)); W = 760
sc = W / ((x1 - x0) * K); H = (y1 - y0) * sc
def P(x, y): return ((x - x0) * K * sc, (y1 - y) * sc)
def path(geom, tol):
    g = geom.simplify(tol, preserve_topology=True)
    polys = [g] if isinstance(g, Polygon) else list(g.geoms)
    out = []
    for p in polys:
        if p.area < 0.0008: continue  # drop tiny islets
        c = [P(x, y) for x, y in p.exterior.coords]
        out.append("M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in c) + "Z")
    return "".join(out)
geo = {"w": round(W), "h": round(H), "u": {}}
for r in U.itertuples():
    tol = 0.01 if r.level == 1 else 0.008
    lp = r.geometry.representative_point() if r.level == 1 else r.geometry.centroid
    geo["u"][r.unit] = {"d": path(r.geometry, tol), "lx": round(P(lp.x, lp.y)[0], 1), "ly": round(P(lp.x, lp.y)[1], 1)}
json.dump(geo, open(f"{SP}/geo.json", "w"))
print(geo["w"], geo["h"], len(geo["u"]), sum(len(v["d"]) for v in geo["u"].values()) // 1000, "kB of paths")
