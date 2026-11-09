"""Area-weighted unit-year timing metrics from the per-cell CHIRPS metrics, as departures from each cell's 1991-2020
median and as the share of the unit in the cell's worst third (late onset, long dry spell, early end of rains).
Usage: daily_agg.py SP   -> daily_metrics_units.parquet, timing_composites.csv"""
import sys, numpy as np, pandas as pd
SP = sys.argv[1]
exec(open(f"{SP}/scripts/common.py").read())
O = pd.read_parquet(f"{SP}/daily_metrics_pixels.parquet"); O = O[O.ok]
W = pd.read_parquet(f"{SP}/weights_daily.parquet")
MET = {"onset": "late", "canic": "long", "dry_so": "long", "end": "early"}
base = O[(O.y >= 1991) & (O.y <= 2020)]
q = base.groupby(["ix", "iy"]).agg(**{f"{k}_med": (k, "median") for k in MET},
                                   **{f"{k}_q1": (k, lambda s: s.quantile(1/3)) for k in MET},
                                   **{f"{k}_q2": (k, lambda s: s.quantile(2/3)) for k in MET}).reset_index()
O = O.merge(q, on=["ix", "iy"])
for k, bad in MET.items():
    O[f"{k}_anom"] = O[k] - O[f"{k}_med"]
    O[f"{k}_bad"] = ((O[k] > O[f"{k}_q2"]) if bad in ("late", "long") else (O[k] < O[f"{k}_q1"])).astype(float)
    O.loc[O[k].isna(), f"{k}_bad"] = np.nan
M = O.merge(W[["ix", "iy", "unit", "w"]], on=["ix", "iy"])
cols = [f"{k}{s}" for k in MET for s in ("", "_anom", "_bad")]
def wavg(d):
    out = {}
    for c in cols:
        v = d[c].astype(float); ok = v.notna()
        out[c] = np.average(v[ok], weights=d.w[ok]) if ok.any() else np.nan
    return pd.Series(out)
U = M.groupby(["unit", "y"]).apply(wavg, include_groups=False).reset_index()
U.to_parquet(f"{SP}/daily_metrics_units.parquet")
rows = []
for u, g in U.groupby("unit"):
    g = g.set_index("y")
    r = dict(unit=u)
    for k in MET:
        r[f"{k}_typ"] = g.loc[1991:2020, k].median()
        r[f"{k}_anom_after"] = g.loc[AFTER, f"{k}_anom"].mean()
        r[f"{k}_bad_after"] = 100 * g.loc[AFTER, f"{k}_bad"].mean()
        for yy in AFTER: r[f"{k}_anom_{yy}"] = g.loc[yy, f"{k}_anom"] if yy in g.index else np.nan
    rows.append(r)
C = pd.DataFrame(rows); C.to_csv(f"{SP}/timing_composites.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(C[C.unit.str.len() != 4].round(1).to_string(index=False))
