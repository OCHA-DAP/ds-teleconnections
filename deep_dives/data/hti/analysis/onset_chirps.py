"""Pixel-level onset / dry spells from CHIRPS daily (Feb–Jul), aggregated to departments (area-weighted)."""
import sys, glob, os, numpy as np, pandas as pd, xarray as xr
SP=sys.argv[1]
exec(open(f"{SP}/onset_lib.py").read())
W=pd.read_parquet(f"{SP}/weights_chirps_daily.parquet")
pix=W[["ix","iy"]].drop_duplicates().reset_index(drop=True)
rows=[]
for fp in sorted(glob.glob(f"{SP}/chirps_fj/chirps_hti_fj_*.nc")):
    y=int(fp[-7:-3]); ds=xr.open_dataset(fp, decode_times=False)
    dates=pd.date_range(f"{y}-02-01", f"{y}-07-31")
    P=ds["prcp"].transpose("T","X","Y").values
    assert P.shape[0]==len(dates), (y, P.shape)
    doy=(dates.dayofyear - ((dates.is_leap_year)&(dates.month>2)).astype(int)).values
    for ix,iy in zip(pix.ix.values, pix.iy.values):
        r=P[:,ix,iy]
        if np.isnan(r).any(): continue
        o=onset_dryspell(r, doy); rows.append(dict(y=y, ix=ix, iy=iy, **o))
    print(y, flush=True)
O=pd.DataFrame(rows); O.to_parquet(f"{SP}/onset_chirps_pixels.parquet")
