"""FEWS NET classification maps -> share of each side's population in Phase 3+ / 4+.

Maps used, per year Y:
  lean  = the projection for the lean season made at its start: the June round's
          near-term (ML1, Jun-Sep) from 2016, the July round's ML1 (Jul-Sep) before.
  post  = the October round's current situation (CS, post-harvest).
From July 2011 (before April 2011 FEWS NET used its own severity scale).
Sources: FEWS NET's regional shapefile archive (East Africa for Sudan, West Africa for
Chad) to 2021; the FEWS NET Data Warehouse round packages (one zip per country and
round) from 2022. FEWS NET published nothing between Dec 2024 and Oct 2025.

Population weights: WorldPop 2018 1 km (constrained, UN-adjusted), fixed for all
years, so the series moves only with the maps. Units FEWS NET did not classify
(no phase, or 66/88/99 = water/park/no data) are left out of the denominator and
their population share is reported as `pop_unclassified`. IDP-camp layers are left out.

Writes data/processed/fewsnet/phase_share.parquet
  side, year, map (lean/post), round, layer, share3, share4, share5, mean_phase,
  pop_unclassified   (shares = people living in areas classified at that phase or worse)
"""

import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW, load_adm  # noqa: E402

FN = RAW / "fewsnet"
OUT = PROC / "fewsnet"
LEGACY = "https://shapefiles.fews.net/HFIC/{r}/{n}{ym}.zip"
REGION = {"darfur": ("EA", "east-africa", "SD"), "chad": ("WA", "west-africa", "TD")}
PKG = "https://fdw.fews.net/api/ipcpackage/?country_code={c}&collection_date={d}"
HDR = {"User-Agent": "OCHA-CHD data science"}


def rounds():
    """(year, map, round 'YYYYMM', layer) for every year, plus the latest outlooks.

    Starts with July 2011: until April 2011 FEWS NET mapped on its own Food Insecurity
    Severity Scale, not comparable with the IPC-compatible phases used since.
    `junproj` = the June round's projection for October-January, to compare with what
    October then showed."""
    out = []
    for y in range(2011, 2027):
        lean_m = "07" if y <= 2015 else "06"
        out.append((y, "lean", f"{y}{lean_m}", "ML1"))
        out.append((y, "post", f"{y}10", "CS"))
        if y >= 2016:
            out.append((y, "junproj", f"{y}06", "ML2"))
    # latest projections reaching into 2027 (FEWS NET's own outlook for this season)
    for ym in ("202606", "202609", "202610"):
        out.append((2027, "outlook", ym, "ML1"))
        out.append((2027, "outlook", ym, "ML2"))
    return out


def get_zip(side, ym):
    r, n, c = REGION[side]
    f = FN / f"{side}_{ym}.zip"
    if f.exists() and f.stat().st_size > 1000:
        return f  # failures are not cached: a round published later is fetched on re-run
    y = int(ym[:4])
    url = (LEGACY.format(r=r, n=n, ym=ym) if y <= 2021
           else PKG.format(c=c, d=f"{ym[:4]}-{ym[4:]}-01"))
    resp = requests.get(url, headers=HDR, timeout=600)
    time.sleep(2)
    ok = resp.status_code == 200 and resp.content[:2] == b"PK"
    print(f"[fewsnet] {side} {ym}: {'ok' if ok else 'none'} ({len(resp.content)} B)", flush=True)
    if not ok:
        return None
    f.write_bytes(resp.content)
    return f


def read_layer(z, layer):
    names = [n for n in zipfile.ZipFile(z).namelist()
             if n.endswith(f"_{layer}.shp") and "IDP" not in n]
    if not names:
        return None
    g = gpd.read_file(f"zip://{z}!{names[0]}").to_crs(4326)
    g["phase"] = pd.to_numeric(g[layer], errors="coerce")
    g.loc[~g["phase"].between(1, 5), "phase"] = np.nan
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FN.mkdir(parents=True, exist_ok=True)
    with rasterio.open(RAW / "worldpop_2018_bbox.tif") as src:
        pop = src.read(1).astype(np.float64)
        tr, shape = src.transform, src.shape
    pop[~np.isfinite(pop) | (pop < 0)] = 0
    a1 = load_adm(1)
    side_mask = {s: rasterize(g.geometry, out_shape=shape, transform=tr, fill=0,
                              default_value=1).astype(bool)
                 for s, g in a1.groupby("side")}

    rows = []
    for y, kind, ym, layer in rounds():
        for side in REGION:
            z = get_zip(side, ym)
            if z is None:
                continue
            g = read_layer(z, layer)
            if g is None:
                print(f"[fewsnet] {side} {ym}: no {layer} layer")
                continue
            g = g.cx[19:29, 8:21]
            ph = np.zeros(shape, np.uint8)  # 0 = not classified / outside
            shapes = [(geom, int(p)) for geom, p in zip(g.geometry, g["phase"])
                      if geom is not None and np.isfinite(p)]
            if shapes:
                # draw low phases first so a higher phase wins on shared edges
                shapes.sort(key=lambda t: t[1])
                ph = rasterize(shapes, out_shape=shape, transform=tr, fill=0, dtype="uint8")
            m = side_mask[side]
            tot = pop[m].sum()
            cl = pop[m & (ph > 0)].sum()
            rec = {"side": side, "year": y, "map": kind, "round": ym, "layer": layer,
                   "pop_unclassified": 1 - cl / tot if tot else np.nan}
            for k in (3, 4, 5):
                rec[f"share{k}"] = pop[m & (ph >= k)].sum() / cl if cl else np.nan
            sel = m & (ph > 0)
            rec["mean_phase"] = (pop[sel] * ph[sel]).sum() / cl if cl else np.nan
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "phase_share.parquet", index=False)
    print(d.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
