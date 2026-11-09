"""Global ENSO season map: the data behind docs/enso-map/.

For every 0.25° ERA5 land cell and every rolling three-month season, the
correlation of seasonal rainfall with Niño3.4, written as small binary files
that the interactive viewer (docs/enso-map/index.html) reads and paints.

The method is the survey's pixel pass, unchanged (teleconnection_survey.py):
total Pearson r between the season's ERA5 rainfall and Niño3.4, at the lag of
0-3 months with the largest |r| (the survey's default lag cap), 1981-2025, the
same rainy-season rule (the season holds >= 25 % of the year's rain and
averages >= 0.25 mm/day) and p < 0.05. Only the area and the land mask differ:

  * area: 180°W-180°E, 60°S-84°N. The survey's maps stop at 100°W, 36°S and
    56°N (no western North America, New Zealand or southern South America).
  * land: Natural Earth 1:10m land plus its minor-islands layer, so atolls and
    small islands get cells. The survey rasterises the 1:110m country layer.

The survey's pixel ENSO map shows each cell's strongest season; this map shows
every season separately. Each run checks the two against each other on the
cells they share (the strongest season here must equal the survey's value).

    uv run python enso_season_map.py
    uv run python enso_season_map.py --cache-dir ../ds-teleconnections/cache \\
        --survey-npz ../ds-teleconnections/out/corr_px_display_l3.npz

The first run needs read access to the prod raster blob (ERA5 monthly COGs,
~2.4 GB download, ~1.8 GB on disk under cache/era5_pixel_global/). Later runs
are offline. The Niño3.4 series is the survey's pinned cache/nino34.data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon

import teleconnection_survey as ts

EXTENT_X = (-180.0, 180.0)
EXTENT_Y = (-60.0, 84.0)
CACHE_NAME = "era5_pixel_global"
MAX_LAG = ts.LAG_CAPS["l3"]          # the survey's default lag cap (3 months)
INDEX = "nino34"
OUT_DIR = Path("docs/enso-map/data")

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
NE_LAND = ("ne_10m_land", "ne_10m_minor_islands")
NE_COUNTRIES = "ne_50m_admin_0_countries"
NE_PLACES = "ne_50m_populated_places_simple"
BASEMAP_SIMPLIFY_DEG = 0.02     # ~2 km: well under one 0.25° cell at the map's top zoom

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# flags byte, one per land cell and season (read by docs/enso-map/index.html)
F_LAG = 0b11            # bits 0-1: best lag in months (0-3)
F_RAINY = 1 << 2        # the season passes the rainy-season rule here
F_SIG = 1 << 3          # p < alpha at the best lag
F_VALID = 1 << 4        # a correlation could be computed at all
F_CLASS_SHIFT = 5       # bits 5-7: 0 none, 1 +moderate, 2 +strong, 3 -moderate, 4 -strong
                        # (set wherever p < alpha, also outside the rainy season: the
                        # viewer can show the drier part of the year on request)
R_NAN = -128            # int8 sentinel for "no r"


def _ne_layer(cache_dir: Path, name: str) -> gpd.GeoDataFrame:
    path = cache_dir / f"naturalearth_{name}.geojson"
    if not path.exists():
        resp = requests.get(NE_BASE + f"{name}.geojson", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return gpd.read_file(path)


def land_layer(cache_dir: Path) -> gpd.GeoDataFrame:
    """Natural Earth 1:10m land + minor islands, one GeoDataFrame."""
    parts = [_ne_layer(cache_dir, n)[["geometry"]] for n in NE_LAND]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")


def _polys(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _bbox(geom) -> list[float] | None:
    """[south, west, north, east] for zooming to a country, antimeridian-aware.

    Countries that cross 180° (Fiji, Russia, Kiribati, the US with the
    Aleutians) get a bbox in 0-360° longitudes when that is the tighter one;
    the viewer passes it to Leaflet as is (east > 180 is a world copy)."""
    polys = _polys(geom)
    if not polys:
        return None
    b = np.array([p.bounds for p in polys])        # minx, miny, maxx, maxy
    s, n = float(b[:, 1].min()), float(b[:, 3].max())
    w0, e0 = float(b[:, 0].min()), float(b[:, 2].max())
    cx = (b[:, 0] + b[:, 2]) / 2
    shifted = np.where(cx[:, None] < 0, b[:, [0, 2]] + 360, b[:, [0, 2]])
    w1, e1 = float(shifted[:, 0].min()), float(shifted[:, 1].max())
    # shifted only when clearly tighter: a bare `<` let floating-point noise (~1e-14) move
    # 38 wholly western countries (UK, Brazil, Caribbean...) into 0-360 longitudes
    w, e = (w1, e1) if (e1 - w1) < (e0 - w0) - 1e-6 else (w0, e0)
    return [round(s, 2), round(w, 2), round(n, 2), round(e, 2)]


def countries(cache_dir: Path) -> list[dict]:
    g = _ne_layer(cache_dir, NE_COUNTRIES)
    out = []
    for _, row in g.iterrows():
        bb = _bbox(row.geometry)
        if bb is None or row["ADM0_A3"] == "ATA":
            continue
        name = row["NAME_LONG"] or row["NAME"]
        alt = row["NAME"] if row["NAME"] != name else None
        out.append({"name": name, "alt": alt, "iso3": row["ADM0_A3"], "bbox": bb})
    return sorted(out, key=lambda d: d["name"])


def _round_coords(c, nd: int = 3):
    if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
        return [round(float(v), nd) for v in c]
    return [_round_coords(v, nd) for v in c]


def basemap(cache_dir: Path) -> dict:
    """Country polygons and labels for the viewer's own basemap.

    The page draws in Equal Earth, so it cannot use Web Mercator basemap tiles:
    land, borders and names come from Natural Earth instead (1:50m countries,
    simplified per feature; country label points and zoom levels from the same
    layer; 1:50m populated places with their own zoom levels).
    """
    import shapely
    from shapely.geometry import mapping

    g = _ne_layer(cache_dir, NE_COUNTRIES)
    g = g[g.geometry.notna()]               # Antarctica stays in the land fill (no label, no search)
    # simplified as one coverage, so neighbours keep one shared border (per-feature
    # simplify drew shared borders twice at the top zooms), then densified to 0.5 degree:
    # in Equal Earth meridians curve, so a long straight lon/lat edge (the 180 degree cut,
    # meridian borders) must carry intermediate vertices to follow them
    simplified = shapely.coverage_simplify(g.geometry.values, BASEMAP_SIMPLIFY_DEG)
    feats = []
    for geom in shapely.segmentize(simplified, 0.5):
        if geom is None or geom.is_empty:
            continue
        m = mapping(geom)
        feats.append({"type": "Feature", "properties": {},
                      "geometry": {"type": m["type"],
                                   "coordinates": _round_coords(m["coordinates"])}})
    country_labels = sorted(
        ([row["NAME"], round(float(row["LABEL_Y"]), 2), round(float(row["LABEL_X"]), 2),
          float(row["MIN_LABEL"]), int(row["LABELRANK"])]
         for _, row in g.iterrows() if row["ADM0_A3"] != "ATA"),
        key=lambda v: (v[3], v[4]))
    pl = _ne_layer(cache_dir, NE_PLACES)
    city_labels = sorted(
        ([row["name"], round(float(row["latitude"]), 3), round(float(row["longitude"]), 3),
          float(row["min_zoom"]), int(row["adm0cap"])]
         for _, row in pl.iterrows()),
        key=lambda v: (v[3], -v[4]))
    return {"countries": {"type": "FeatureCollection", "features": feats},
            "labels": {"countries": country_labels, "cities": city_labels}}
    # (the page merges the two lists into one priority order: by zoom, countries first on ties)


def _class(r: np.ndarray, show: np.ndarray) -> np.ndarray:
    """The survey's four |r| classes (teleconnection_survey._paint_r_bins)."""
    c = np.zeros(r.shape, dtype="uint8")
    rr = np.where(show, r, 0.0)
    c[(rr >= ts._R_MIN) & (rr < ts._R_STRONG)] = 1
    c[rr >= ts._R_STRONG] = 2
    c[(rr <= -ts._R_MIN) & (rr > -ts._R_STRONG)] = 3
    c[rr <= -ts._R_STRONG] = 4
    return c


def _gz(path: Path, data: bytes) -> None:
    # mtime=0 keeps the bytes identical across rebuilds of the same data
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))


def tri_months(tri: str) -> str:
    m = ts._TRIMESTER_MONTHS[tri]
    return f"{MONTH_ABBR[m[0] - 1]}–{MONTH_ABBR[m[-1] - 1]}"


def check_against_survey(npz_path: Path, px: ts.PixelData,
                         r_strongest: np.ndarray) -> dict:
    """Compare the strongest significant season per cell with the survey's
    pixel ENSO map (total, 3-month cap) on the cells both grids have."""
    z = np.load(npz_path, allow_pickle=False)
    xs, ys, fi = z["x"], z["y"], z["flat_idx"]
    r_s = z[f"total_{INDEX}_r"].astype("float64")
    ny_s, nx_s = len(ys), len(xs)
    rows_s, cols_s = np.divmod(fi, nx_s)
    lon, lat = xs[cols_s], ys[rows_s]

    d = float(abs(px.x[1] - px.x[0]))
    col = np.rint((lon - px.x[0]) / d).astype(int)
    row = np.rint((px.y[0] - lat) / d).astype(int)
    pos_grid = np.full(px.shape[0] * px.shape[1], -1, dtype="int64")
    pos_grid[px.flat_idx] = np.arange(px.n_cells)
    mine = pos_grid[row * px.shape[1] + col]
    shared = mine >= 0
    a = r_s[shared]
    b = r_strongest[mine[shared]].astype("float64")
    both = np.isfinite(a) & np.isfinite(b)
    out = {
        "survey_cells": int(len(fi)),
        "shared_cells": int(shared.sum()),
        "both_have_r": int(both.sum()),
        "only_survey": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
        "only_here": int((~np.isfinite(a) & np.isfinite(b)).sum()),
        "max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", default=str(ts.CONFIG["cache_dir"]),
                    help="the survey's cache folder (pinned nino34.data, ERA5 stack)")
    ap.add_argument("--survey-npz", default=str(ts.CONFIG["parquet_dir"] / "corr_px_display_l3.npz"),
                    help="the survey's pixel display file, for the cross-check")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cfg = {**ts.CONFIG, "cache_dir": cache_dir}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = ts.load_indices(cfg)[[INDEX]]
    gdf = land_layer(cache_dir)
    px = ts.pixel_trimester_rainfall_era5(cfg, gdf, EXTENT_X, EXTENT_Y, CACHE_NAME)
    res = ts.pixel_sweep(px, indices, cfg, max_lag=MAX_LAG)
    j = list(indices.columns).index(INDEX)

    # --- cross-check with the survey's pixel map --------------------------
    disp = ts.pixel_reduce_for_display(res, px, list(indices.columns), cfg["alpha"])
    npz = Path(args.survey_npz)
    check = None
    if npz.exists():
        check = check_against_survey(npz, px, disp[INDEX]["r"])
        print("check vs survey pixel map:", json.dumps(check))
        if (check["both_have_r"] == 0 or check["only_survey"] or check["only_here"]
                or check["max_abs_diff"] > 1e-6):
            raise SystemExit("the strongest season here does not reproduce the survey's "
                             "pixel ENSO map on the shared cells; not writing the viewer data")
    else:
        print(f"no survey file at {npz}; cross-check skipped")

    # --- per-season files ---------------------------------------------------
    annual = np.zeros(px.n_cells, dtype="float64")
    for tri in ts._ANNUAL_TRIMESTERS:
        annual += px.tri_clim[tri]

    # floor, not rint: the viewer tests the stored value against 0.25 mm/day, and
    # stored >= 25 must mean the float passed the survey's >= 0.25 test
    def mm100(v: np.ndarray) -> np.ndarray:
        return np.clip(np.floor(np.nan_to_num(v, nan=0.0) * 100), 0, 65535).astype("<u2")

    seasons, digests, counts = [], [], {}
    for tri in ts.TRIMESTERS:
        rt = res[tri]
        r = rt["r"][j].astype("float64")
        p = rt["p"][j].astype("float64")
        lag = rt["lag"][j]
        n = rt["n"][j]
        valid = np.isfinite(r)
        rainy = px.rainy[tri]
        sig = valid & (p < cfg["alpha"])
        cls = _class(r, sig)

        flags = (np.where(valid, lag, 0).astype("uint8") & F_LAG)
        flags |= np.where(rainy, F_RAINY, 0).astype("uint8")
        flags |= np.where(sig, F_SIG, 0).astype("uint8")
        flags |= np.where(valid, F_VALID, 0).astype("uint8")
        flags |= (cls << F_CLASS_SHIFT).astype("uint8")
        # truncated towards zero, not rounded, so the r the viewer prints never
        # crosses the 0.30 / 0.50 class edges the colour was decided on (0.296 -> 0.29)
        r8 = np.where(valid, np.clip(np.trunc(r * 100), -100, 100), R_NAN).astype("int8")

        blob = mm100(px.tri_clim[tri]).tobytes() + r8.tobytes() + flags.tobytes()
        _gz(out_dir / f"{tri}.bin.gz", blob)
        digests.append(hashlib.sha1(blob).hexdigest())

        n_by_lag = []
        for L in range(MAX_LAG + 1):
            vals = np.unique(n[valid & (lag == L)])
            n_by_lag.append([int(v) for v in vals])
        yrs = px.tri_years[tri]
        seasons.append({
            "code": tri, "months": tri_months(tri), "end_month": ts.TRIMESTERS[tri],
            "years": [int(yrs[0]), int(yrs[-1])],
            "n_by_lag": n_by_lag,
        })
        rc = np.where(rainy, cls, 0)    # as mapped by default (the survey's rule)
        counts[tri] = {
            "rainy": int(rainy.sum()),
            "strong_pos": int((rc == 2).sum()), "mod_pos": int((rc == 1).sum()),
            "mod_neg": int((rc == 3).sum()), "strong_neg": int((rc == 4).sum()),
        }
        print(tri, counts[tri])

    land = np.zeros(px.shape[0] * px.shape[1], dtype=bool)
    land[px.flat_idx] = True
    land_bytes = np.packbits(land).tobytes()
    _gz(out_dir / "land.bin.gz", land_bytes)
    # annual rounded UP (season rounded down): the share the viewer prints, clim / annual,
    # then never exceeds the true share, so a season skipped at 24.97% cannot read "25.0%"
    ann_bytes = np.clip(np.ceil(annual * 100), 0, 65535).astype("<u2").tobytes()
    _gz(out_dir / "annual.bin.gz", ann_bytes)
    digests += [hashlib.sha1(land_bytes).hexdigest(), hashlib.sha1(ann_bytes).hexdigest()]

    # the JSON files are fetched with ?v=<version> too, so they count towards it
    for name, obj in (("countries.json", countries(cache_dir)), ("basemap.json", basemap(cache_dir))):
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        (out_dir / name).write_text(text)
        digests.append(hashlib.sha1(text.encode()).hexdigest())

    nino_last = indices[INDEX].dropna().index.max()
    meta = {
        "version": hashlib.sha1("".join(digests).encode()).hexdigest()[:12],
        "built": dt.date.today().isoformat(),
        "period": [cfg["start_year"], cfg["end_year"]],
        "grid": {"lon0": float(px.x[0]), "lat0": float(px.y[0]),
                 "d": float(abs(px.x[1] - px.x[0])),
                 "nx": int(px.shape[1]), "ny": int(px.shape[0])},
        "n_land": int(px.n_cells),
        "max_lag": MAX_LAG,
        "thresholds": {"r_min": ts._R_MIN, "r_strong": ts._R_STRONG,
                       "alpha": cfg["alpha"], "rainy_share": 0.25,
                       "min_mm_day": ts.PIXEL_MIN_TRI_MM_DAY,
                       "min_years": cfg["min_years"]},
        "nino34_last_month": nino_last.strftime("%Y-%m"),
        "seasons": seasons,
        "counts": counts,
        "survey_check": check,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    sizes = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"wrote {out_dir} ({sizes / 1e6:.1f} MB), {px.n_cells:,} land cells, "
          f"version {meta['version']}")


if __name__ == "__main__":
    main()
