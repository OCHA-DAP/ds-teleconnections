import sys, numpy as np, xarray as xr, pandas as pd
nc, wp, out = sys.argv[1:4]
ds = xr.open_dataset(nc, decode_times=False)
print(ds["T"].attrs, ds["precipitation"].attrs.get("units"))
T = ds["T"].values  # months since 1960-01-01
t0 = pd.Timestamp("1960-01-01")
dates = [t0 + pd.DateOffset(months=int(np.floor(v))) for v in T]
p = ds["precipitation"].transpose("T","X","Y").values  # mm/month
w = pd.read_parquet(wp)
rows = []
for pc, g in w.groupby("ADM1_PCODE"):
    vals = p[:, g.ix.values, g.iy.values]
    ww = g.w.values
    m = np.nansum(vals * ww, axis=1) / np.nansum(np.where(np.isnan(vals), 0, 1) * ww, axis=1)
    rows.append(pd.DataFrame({"pcode": pc, "name": g.ADM1_FR.iloc[0], "date": dates, "mm": m}))
df = pd.concat(rows); df.to_parquet(out)
print(df.date.min(), df.date.max()); print(df[df.pcode=="HT"].tail(4))
