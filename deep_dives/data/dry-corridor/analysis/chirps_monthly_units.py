import sys, numpy as np, xarray as xr, pandas as pd
nc, wp, out = sys.argv[1:4]
ds = xr.open_dataset(nc, decode_times=False)
T = ds["T"].values
dates = [pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(np.floor(v))) for v in T]
p = ds["precipitation"].transpose("T", "X", "Y").values
w = pd.read_parquet(wp)
rows = []
for (u, nm, iso, lv), g in w.groupby(["unit", "name", "iso3", "level"]):
    vals = p[:, g.ix.values, g.iy.values]; ww = g.w.values
    m = np.nansum(vals * ww, axis=1) / np.nansum(np.where(np.isnan(vals), 0, 1) * ww, axis=1)
    rows.append(pd.DataFrame({"unit": u, "name": nm, "iso3": iso, "level": lv, "date": dates, "mm": m}))
df = pd.concat(rows); df.to_parquet(out)
print(df.date.min(), df.date.max(), df.unit.nunique()); print(df[df.unit == "SV"].tail(4))
