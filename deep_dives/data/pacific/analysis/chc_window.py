"""CHIRPS v2 monthly from the CHC COGs, windowed reads one file at a time (CHC rate-limits parallel reads).

usage: chc_window.py OUT_NPZ X0 X1 Y0 Y1 MONTHS YEAR0 YEAR1 [EXTRA_YYYY-MM ...]
Reads months MONTHS (comma list) of YEAR0..YEAR1 plus any extra year-months; saves {"ym": [...], "data": cube, "x", "y"}.
"""
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import from_bounds

out, x0, x1, y0, y1 = sys.argv[1], *map(float, sys.argv[2:6])
months = [int(m) for m in sys.argv[6].split(",")]
yms = [(y, m) for y in range(int(sys.argv[7]), int(sys.argv[8]) + 1) for m in months]
yms += [tuple(map(int, s.split("-"))) for s in sys.argv[9:]]
URL = "/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_monthly/cogs/chirps-v2.0.{y}.{m:02d}.cog"
arrs, got = [], []
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="5"):
    for y, m in yms:
        try:
            with rasterio.open(URL.format(y=y, m=m)) as src:
                w = from_bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w).astype("float32")
                a[a < 0] = np.nan
                tr = src.window_transform(w)
            arrs.append(a)
            got.append(f"{y}-{m:02d}")
        except Exception as ex:  # noqa: BLE001
            print("fail", y, m, ex, flush=True)
        time.sleep(0.7)
xs = tr.c + tr.a * (np.arange(arrs[0].shape[1]) + 0.5)
ys = tr.f + tr.e * (np.arange(arrs[0].shape[0]) + 0.5)
np.savez_compressed(out, ym=np.array(got), data=np.stack(arrs), x=xs, y=ys)
print("saved", out, len(got), "months", flush=True)
