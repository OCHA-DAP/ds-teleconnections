"""MODIS NDVI season greening, 2001-2026, on a 0.02 deg grid over the AOI box.

Greening = mean NDVI of the 16-day composites starting 12 Jul - 30 Sep (covering mid-July to
mid-October; the same six composites every year) minus the mean of those starting 1 Mar -
31 May of the same year (the dry-season floor). Planetary Computer has no MOD13A1 composites
from June 2025 to February 2026, so 2025 has no value. Raw NDVI as % of
average understates deficits on sparse Sahel vegetation; the seasonal rise does not.

Source: MOD13A1 v6.1 (Terra, 500 m, 16-day) COGs from the Microsoft Planetary Computer,
read at overview level 2 (about 2 km) to keep the transfer small. Items are picked by
their id (MOD13A1.*), not the `platform` property, which drops years on this catalogue.
Pixels flagged cloudy/snow/fill in pixel_reliability are left out of the composite mean.
Each item read is cached as .npy, so a rerun only fetches what is missing.

Writes <work>/processed/ndvi/greening.npz  (years, lat, lon, greening[year, lat, lon],
n_season[year, lat, lon], n_dry[...])
"""

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import BBOX, PROC, RAW  # noqa: E402

RES = 0.02
OVR = 1  # overview index: 0 = 1 km, 1 = 2 km for a 2400-px 500 m tile
CACHE = RAW / "ndvi_items"
OUT = PROC / "ndvi"
DST_CRS = "EPSG:4326"


def grid():
    x0, y0, x1, y1 = BBOX
    nx, ny = round((x1 - x0) / RES), round((y1 - y0) / RES)
    return from_origin(x0, y1, RES, RES), (ny, nx)


def window_of(d: date) -> str | None:
    md = (d.month, d.day)
    if (3, 1) <= md <= (5, 31):
        return "dry"
    if (7, 1) <= md <= (10, 14):  # Terra 16-day starts 12 Jul .. 30 Sep (15 Oct in leap years excluded)
        return "season"
    return None


def read_item(item, dst_tr, shape):
    f = CACHE / f"{item.id}.npy"
    if f.exists():
        return np.load(f)
    item = pc.sign(item)
    out = np.full(shape, np.nan, np.float32)
    for attempt in range(4):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3"):
                with rasterio.open(item.assets["500m_16_days_NDVI"].href, overview_level=OVR) as s:
                    ndvi = s.read(1).astype(np.float32)
                    tr, crs = s.transform, s.crs
                with rasterio.open(item.assets["500m_16_days_pixel_reliability"].href,
                                   overview_level=OVR) as s:
                    rel = s.read(1)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  failed {item.id}: {e}", flush=True)
                return None
            time.sleep(5 * (attempt + 1))
            item = pc.sign(item)
    ndvi[(ndvi <= -2000) | ~np.isin(rel, (0, 1))] = np.nan
    ndvi /= 10000.0
    reproject(ndvi, out, src_transform=tr, src_crs=crs, dst_transform=dst_tr, dst_crs=DST_CRS,
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    np.save(f, out)
    return out


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dst_tr, shape = grid()
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = []
    for y in range(2001, 2027):
        s = cat.search(collections=["modis-13A1-061"], bbox=list(BBOX),
                       datetime=f"{y}-03-01/{y}-10-31")
        for it in s.items():
            if not it.id.startswith("MOD13A1"):
                continue
            d = (it.datetime or it.properties.get("start_datetime"))
            d = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
            w = window_of(d)
            if w:
                items.append((y, w, d, it))
    print(f"[ndvi] {len(items)} tile-composites", flush=True)
    sums = {}
    t0 = time.time()

    def work(rec):
        return rec, read_item(rec[3], dst_tr, shape)

    with ThreadPoolExecutor(6) as ex:
        for i, ((y, w, d, it), a) in enumerate(ex.map(work, items)):
            if a is None:
                continue
            key = (y, w, d)
            m = sums.setdefault(key, np.full(shape, np.nan, np.float32))
            np.copyto(m, a, where=np.isfinite(a) & ~np.isfinite(m))  # mosaic tiles of one composite
            if i % 100 == 0:
                print(f"  {i}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    years = sorted({k[0] for k in sums})
    g = np.full((len(years),) + shape, np.nan, np.float32)
    ns = np.zeros((len(years),) + shape, np.int8)
    nd = np.zeros((len(years),) + shape, np.int8)
    for i, y in enumerate(years):
        season = [v for k, v in sums.items() if k[0] == y and k[1] == "season"]
        dry = [v for k, v in sums.items() if k[0] == y and k[1] == "dry"]
        if not season or not dry:
            continue
        S, Dr = np.stack(season), np.stack(dry)
        ns[i], nd[i] = np.isfinite(S).sum(0), np.isfinite(Dr).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g[i] = np.nanmean(S, 0) - np.nanmean(Dr, 0)
        g[i][(ns[i] < 3) | (nd[i] < 2)] = np.nan
        print(f"  {y}: {len(season)} season, {len(dry)} dry composites "
              f"(dates {min(k[2] for k in sums if k[0] == y)} .. {max(k[2] for k in sums if k[0] == y)})")
    x0, y0, x1, y1 = BBOX
    lat = y1 - RES / 2 - RES * np.arange(shape[0])
    lon = x0 + RES / 2 + RES * np.arange(shape[1])
    np.savez_compressed(OUT / "greening.npz", years=np.array(years), lat=lat, lon=lon,
                        greening=g, n_season=ns, n_dry=nd)
    print(f"[ndvi] wrote {OUT / 'greening.npz'}")


if __name__ == "__main__":
    main()
