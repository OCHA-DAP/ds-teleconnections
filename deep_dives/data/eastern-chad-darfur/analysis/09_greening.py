"""Detrended MODIS greening: per side and year, per state/province for 2026, and the 2026 map.

Same recipe as the Chad evaluation-villages page (ds-aa-tcd-drought), which the user asked
for there: each pixel's greening is divided by its own 2001-2025 mean (cell-normalised),
units are averaged, a straight line is fitted to the unit's 2001-2025 series, and a year
is expressed as % of the level that line gives for it. A pixel on the map is its
cell-normalised 2026 greening as % of its admin-2 unit's trend level for 2026 (a
pixel-level trend over 25 noisy years is mostly noise).

Pixels count only inside the rainfall belt (0.5 deg cells with >= 100 mm in an average
June-September) and where the average greening is at least MIN_GREEN (bare ground and
desert, where a ratio is meaningless, drop out).

Writes <work>/processed/ndvi/:
  side_years.parquet   side, year, pct_raw (% of the 2001-2025 mean), pct_detr (% of trend)
  adm1_2026.parquet    adm1_pcode, pct_detr_2026, pct_raw_2026
  map_2026.npz         lat, lon (0.04 deg), pct (detrended 2026, NaN outside the mask)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import BBOX, PROC, load_adm  # noqa: E402

MIN_GREEN = 0.03
BASE = (2001, 2025)
OUT = PROC / "ndvi"


def trend_level(years, series, base=BASE):
    m = (years >= base[0]) & (years <= base[1]) & np.isfinite(series)
    b = np.polyfit(years[m], series[m], 1)
    return np.polyval(b, years), b[0]


def main():
    z = np.load(OUT / "greening.npz")
    years, lat, lon, g = z["years"], z["lat"], z["lon"], z["greening"]
    ny, nx = lat.size, lon.size
    res = float(lat[0] - lat[1])
    x0, y0, x1, y1 = BBOX
    tr = from_origin(x0, y1, res, res)

    # belt mask from the 0.5 deg rainfall grid
    rg = pd.read_parquet(PROC / "rain" / "grid_2026.parquet")
    belt05 = {(round(a, 2), round(b, 2)): bool(c) for a, b, c in zip(rg["lat"], rg["lon"], rg["belt"])}
    clat = np.floor(lat / 0.5) * 0.5 + 0.25
    clon = np.floor(lon / 0.5) * 0.5 + 0.25
    belt = np.array([[belt05.get((round(a, 2), round(b, 2)), False) for b in clon] for a in clat])

    base = (years >= BASE[0]) & (years <= BASE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(g[base], axis=0)
        norm = g / mean
    ok = belt & (mean >= MIN_GREEN) & (np.isfinite(g[base]).sum(0) >= 20)
    norm[:, ~ok] = np.nan

    a2 = load_adm(2).reset_index(drop=True)
    a2["id"] = np.arange(1, len(a2) + 1)
    ids = rasterize(zip(a2.geometry, a2["id"]), out_shape=(ny, nx), transform=tr, fill=0,
                    dtype="int32")
    side_of = dict(zip(a2["id"], a2["side"]))
    adm1_of = dict(zip(a2["id"], a2["adm1_pcode"]))

    def unit_series(mask):
        return np.array([np.nanmean(norm[i][mask]) if np.isfinite(norm[i][mask]).any() else np.nan
                         for i in range(len(years))])

    # per side
    rows = []
    for side in ("chad", "darfur"):
        mask = np.isin(ids, [k for k, v in side_of.items() if v == side])
        s = unit_series(mask)
        lvl, slope = trend_level(years, s)
        for y, v, t in zip(years, s, lvl):
            rows.append({"side": side, "year": int(y), "pct_raw": 100 * v,
                         "pct_detr": 100 * v / t, "slope_per_decade": 1000 * slope})
        print(f"[greening] {side}: {mask.sum() and np.isfinite(norm[-1][mask]).sum()} pixels in 2026, "
              f"trend {1000 * slope:+.1f} %-pts/decade, 2026 raw {100 * s[-1]:.0f}% detr "
              f"{100 * s[-1] / lvl[-1]:.0f}%")
    pd.DataFrame(rows).to_parquet(OUT / "side_years.parquet", index=False)

    # per province/state
    rows = []
    for pc in sorted(set(adm1_of.values())):
        mask = np.isin(ids, [k for k, v in adm1_of.items() if v == pc])
        s = unit_series(mask)
        lvl, _ = trend_level(years, s)
        rows.append({"adm1_pcode": pc, "pct_detr_2026": 100 * s[-1] / lvl[-1],
                     "pct_raw_2026": 100 * s[-1]})
    pd.DataFrame(rows).to_parquet(OUT / "adm1_2026.parquet", index=False)

    # map: pixel 2026 as % of its admin-2 trend level for 2026
    pct = np.full((ny, nx), np.nan, np.float32)
    for k in a2["id"]:
        mask = ids == k
        s = unit_series(mask)
        if np.isfinite(s[base]).sum() < 15 or not np.isfinite(s[-1]):
            continue
        lvl, _ = trend_level(years, s)
        pct[mask] = 100 * norm[-1][mask] / lvl[-1]
    # 0.04 deg for the page
    c = pct[: ny // 2 * 2, : nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2)
    with np.errstate(invalid="ignore"):
        coarse = np.nanmean(c, axis=(1, 3))
    clat2 = lat[: ny // 2 * 2].reshape(-1, 2).mean(1)
    clon2 = lon[: nx // 2 * 2].reshape(-1, 2).mean(1)
    np.savez_compressed(OUT / "map_2026.npz", lat=clat2, lon=clon2, pct=coarse.astype(np.float32))
    v = coarse[np.isfinite(coarse)]
    print(f"[greening] map: {v.size} cells, median {np.median(v):.0f}%, "
          f"10th-90th {np.percentile(v, 10):.0f}-{np.percentile(v, 90):.0f}%")


if __name__ == "__main__":
    main()
