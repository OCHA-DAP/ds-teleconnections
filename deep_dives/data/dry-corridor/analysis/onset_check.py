"""Sensitivity of the Primera onset result: (a) a simpler rule (first day from 15 Apr starting a 5-day total >= 25 mm, no
false-start test); (b) composites by ENSO situation. Usage: onset_check.py SP -> onset_check.csv"""
import sys, glob, numpy as np, pandas as pd, xarray as xr
SP = sys.argv[1]
exec(open(f"{SP}/scripts/common.py").read())
W = pd.read_parquet(f"{SP}/weights_daily.parquet")
units = ["GT", "HN", "SV", "GT_AOI", "HN_AOI", "SV_AOI"]
pix = W[W.unit.isin(units)][["ix", "iy"]].drop_duplicates().reset_index(drop=True)
rows = []
for fp in sorted(glob.glob(f"{SP}/chirps_d/chirps_cadc_an_*.nc")):
    y = int(fp[-7:-3]); ds = xr.open_dataset(fp, decode_times=False)
    P = np.nan_to_num(ds["prcp"].transpose("T", "X", "Y").values[:, pix.ix.values, pix.iy.values])
    s5 = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), 0); s5 = s5[5:] - s5[:-5]
    ok = s5[14:106] >= 25
    on = np.where(ok.any(0), 14 + np.argmax(ok, 0), 106).astype(float)
    rows.append(pd.DataFrame({"y": y, "ix": pix.ix, "iy": pix.iy, "on5": on}))
O = pd.concat(rows)
O = O.merge(O[(O.y >= 1991) & (O.y <= 2020)].groupby(["ix", "iy"]).on5.agg(med="median", q2=lambda s: s.quantile(2/3)).reset_index(), on=["ix", "iy"])
O["anom"] = O.on5 - O.med; O["late"] = (O.on5 > O.q2).astype(float)
M = O.merge(W[W.unit.isin(units)][["ix", "iy", "unit", "w"]], on=["ix", "iy"])
A = M.groupby(["unit", "y"]).apply(lambda d: pd.Series({"anom": np.average(d.anom, weights=d.w), "late": np.average(d.late, weights=d.w)}), include_groups=False).reset_index()
B = pd.read_parquet(f"{SP}/daily_metrics_units.parquet")
yrs = sorted(A.y.unique())
grp = {y: ("after" if y in AFTER else "during" if NI.loc[y, "MJJ"] >= 0.5 else "nina" if NI.loc[y, "MJJ"] <= -0.5 else "other") for y in yrs}
out = []
for u in units:
    a = A[A.unit == u].set_index("y"); b = B[B.unit == u].set_index("y")
    for g in ["after", "during", "nina", "other"]:
        ys = [y for y in yrs if grp[y] == g]
        out.append(dict(unit=u, group=g, n=len(ys), main_anom=b.loc[ys, "onset_anom"].mean(), main_late=100 * b.loc[ys, "onset_bad"].mean(),
                        main_nlate=int((b.loc[ys, "onset_bad"] > 0.5).sum()), alt_anom=a.loc[ys, "anom"].mean(), alt_late=100 * a.loc[ys, "late"].mean(),
                        alt_nlate=int((a.loc[ys, "late"] > 0.5).sum())))
    print(u, "after-years, alt rule:", {y: (round(a.loc[y, "anom"], 1), round(100 * a.loc[y, "late"])) for y in AFTER})
C = pd.DataFrame(out); C.to_csv(f"{SP}/onset_check.csv", index=False)
pd.set_option("display.width", 200); print(C.round(1).to_string(index=False))
print("during years:", [y for y in yrs if grp[y] == "during"])
