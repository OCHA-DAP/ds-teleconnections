# /// script
# requires-python = ">=3.11"
# dependencies = ["pystac-client", "rasterio", "geopandas", "pyarrow", "numpy"]
# ///
"""Copernicus Global Flood Monitoring (GFM) over the Gaza Strip: one JSON line per Sentinel-1 scene.

GFM (https://global-flood.emergency.copernicus.eu/) maps flood water in every Sentinel-1 scene at
20 m. This reads its STAC catalogue at EODC (https://stac.eodc.eu/api/v1, collection GFM) for every
scene whose footprint touches the Gaza Strip and counts, inside the Gaza polygon (COD-AB PS02, from
cache/gaza/pse_adm1.parquet), the pixels the scene covers, the pixels in GFM's exclusion mask, the
pixels of the ensemble flood extent and of each of the three algorithms (TUW, DLR, LIST).

Most STAC items only graze Gaza (the footprints are generous): rows with n_valid = 0 are kept so the
scan can resume, and the page uses only scenes that cover at least 90% of Gaza.

    uv run --script tools/gaza_gfm_scan.py                       # scan 2015 → today, resumable
    uv run --script tools/gaza_gfm_scan.py --map <STAC item id>  # also save that scene's layers for the map

Output: cache/gaza/gfm_passes.jsonl (append-only) and cache/gaza/gfm_map_<item>.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
CACHE = Path("cache/gaza")
STAC = "https://stac.eodc.eu/api/v1"
ALGOS = [("exclusion_mask", "n_excl"), ("tuw_flood_extent", "n_tuw"), ("dlr_flood_extent", "n_dlr"),
         ("list_flood_extent", "n_list"), ("advisory_flags", "n_adv")]
MAP_LAYERS = ["ensemble_flood_extent", "exclusion_mask", "reference_water_mask", "tuw_flood_extent",
              "dlr_flood_extent", "list_flood_extent"]


def gaza() -> gpd.GeoDataFrame:
    g = gpd.read_parquet(CACHE / "pse_adm1.parquet")
    return g[g.ADM1_PCODE == "PS02"]


class Scanner:
    def __init__(self):
        self.g = gaza()
        self._masks: dict = {}
        self._lock = threading.Lock()

    def window(self, src):
        """Window, transform and polygon mask of Gaza on this scene's grid (cached per grid)."""
        key = (str(src.crs), tuple(src.transform)[:6])
        with self._lock:
            if key not in self._masks:
                g = self.g.to_crs(src.crs); b = tuple(g.total_bounds)
                w = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
                tr = src.window_transform(w)
                m = ~geometry_mask(g.geometry, out_shape=(int(w.height), int(w.width)), transform=tr)
                self._masks[key] = (b, m, tr)
        return self._masks[key]

    @staticmethod
    def read(href: str, b: tuple, shape: tuple) -> np.ndarray:
        with rasterio.open(href) as s:
            w = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
            a = s.read(1, window=w, boundless=True, fill_value=255)
        return a[:shape[0], :shape[1]]

    def stats(self, it) -> dict:
        o = {"id": it.id, "dt": str(it.datetime)}
        if "ensemble_flood_extent" not in it.assets:
            o["err"] = "no ensemble layer"; return o
        with rasterio.open(it.assets["ensemble_flood_extent"].href) as src:
            b, m, _ = self.window(src)
            w = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            fl = src.read(1, window=w, boundless=True, fill_value=255)[:m.shape[0], :m.shape[1]]
        valid = m & (fl != 255)
        o.update(n_gaza=int(m.sum()), n_valid=int(valid.sum()), n_flood=int((valid & (fl == 1)).sum()))
        if o["n_valid"] == 0:
            return o
        for layer, k in ALGOS:
            if layer not in it.assets:
                continue
            try:
                a = self.read(it.assets[layer].href, b, m.shape)
            except Exception as e:  # noqa: BLE001 - one missing layer should not lose the scene
                o[k] = None; o.setdefault("lerr", []).append(f"{layer}:{repr(e)[:80]}"); continue
            if layer == "advisory_flags":
                o[k] = int((valid & (a != 0) & (a != 255)).sum())
            else:
                o[k] = int((valid & (a == 1)).sum())
                if layer == "exclusion_mask":
                    o["n_visible"] = int((valid & (a == 0)).sum())
        return o

    def save_map(self, it, out: Path) -> None:
        with rasterio.open(it.assets["ensemble_flood_extent"].href) as src:
            b, m, tr = self.window(src)
            crs = src.crs.to_string()
        arrs = {k: self.read(it.assets[k].href, b, m.shape) for k in MAP_LAYERS if k in it.assets}
        np.savez_compressed(out, mask=m, transform=np.array(tr)[:6], crs=np.array(crs), **arrs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--map", help="STAC item id whose layers to save for the map")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = CACHE / "gfm_passes.jsonl"
    sc = Scanner()
    client = Client.open(STAC)
    if args.map:
        it = next(client.search(collections=["GFM"], ids=[args.map]).items())
        sc.save_map(it, CACHE / f"gfm_map_{args.map}.npz"); print("saved map layers for", args.map)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            transient = [e for e in r.get("lerr", []) if "404" not in e]   # a 404 layer (e.g. advisory_flags) will not appear later
            if "err" not in r and not transient:      # failed scenes and transient layer errors are retried; the reader keeps the last line per id
                done.add(r["id"])
    items = list(client.search(collections=["GFM"], intersects=sc.g.geometry.iloc[0].__geo_interface__,
                               datetime=f"{args.start}/{args.end}T23:59:59Z").items())
    todo = [i for i in items if i.id not in done]
    print(f"{len(items)} STAC items touch Gaza, {len(todo)} not yet scanned", flush=True)
    with ThreadPoolExecutor(args.threads) as ex, out.open("a") as f:
        futs = {ex.submit(sc.stats, i): i for i in todo}
        for n, fu in enumerate(as_completed(futs)):
            try:
                o = fu.result()
            except Exception as e:  # noqa: BLE001 - logged, retried on the next run
                it = futs[fu]; o = {"id": it.id, "dt": str(it.datetime), "err": repr(e)[:200]}
            f.write(json.dumps(o) + "\n"); f.flush()
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
