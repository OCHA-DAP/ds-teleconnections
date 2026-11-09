# exec'd by the analysis scripts: needs SP defined. Index file paths from env (NINO34_FILE), default = repo cache.
import os, numpy as np, pandas as pd, geopandas as gpd
def read_psl(p):
    rows = []
    for line in open(p):
        t = line.split()
        if len(t) == 13 and t[0].isdigit():
            for m, v in enumerate(t[1:], 1):
                v = float(v); rows.append((int(t[0]), m, np.nan if v < -90 else v))
    return pd.DataFrame(rows, columns=["y", "m", "v"]).set_index(["y", "m"])["v"]
N34 = read_psl(os.environ.get("NINO34_FILE", "cache/live/nino34.data"))  # PSL ERSSTv6, published on a 1981-2010 base
_clim = N34.loc[1991:2020].groupby(level="m").mean()
N34 = N34 - N34.index.get_level_values("m").map(_clim).values  # re-based to 1991-2020, month by month (as the NMME and TNA)
def nseas(y, ms):  # ms may include 0 (= Dec of y-1)
    v = [N34.get((y - 1, 12)) if m == 0 else N34.get((y, m)) for m in ms]
    return np.nan if any(pd.isna(x) for x in v) else float(np.mean(v))
YRS = range(1981, 2028)
NI = pd.DataFrame({"DJF": {y: nseas(y, [0, 1, 2]) for y in YRS}, "MJJ": {y: nseas(y, [5, 6, 7]) for y in YRS},
                   "SON": {y: nseas(y, [9, 10, 11]) for y in YRS}})
AFTER = [1983, 1992, 1998, 2010, 2016, 2024]  # springs after a strong El Niño winter (DJF >= 1.0)
assert AFTER == [y for y in range(1982, 2027) if NI.DJF[y] >= 1.0], NI[NI.DJF >= 1.0]
SEASONS = {"primera": [5, 6, 7, 8], "postrera": [9, 10, 11]}
CONC = {"primera": "MJJ", "postrera": "SON"}
AOI = {"GT_AOI": ["GT20", "GT19", "GT21", "GT02"], "HN_AOI": ["HN07", "HN08"], "SV_AOI": ["SV01", "SV12"]}
# --- CHIRPS monthly by unit
_U = pd.read_parquet(f"{SP}/weights_daily.parquet").drop_duplicates("unit").set_index("unit")
NAMES, LEVEL, ISO = _U["name"], _U["level"], _U["iso3"]
if os.path.exists(f"{SP}/chirps_monthly_units.parquet"):
    CH = pd.read_parquet(f"{SP}/chirps_monthly_units.parquet").rename(columns={"unit": "pcode"})
    CH["y"] = CH.date.dt.year; CH["m"] = CH.date.dt.month
else:
    CH = None
# --- ERA5 monthly by unit (+ framework groups as area-weighted means of their departments)
ER = pd.read_parquet(f"{SP}/era5_cadc_all.parquet")
ER["y"] = ER.valid_date.dt.year; ER["m"] = ER.valid_date.dt.month; ER["mm"] = ER["mean"] * ER.valid_date.dt.days_in_month
_A = pd.read_parquet(f"{SP}/weights_daily.parquet").groupby("unit").w.sum()
_g = []
for k, pcs in AOI.items():
    s = ER[ER.pcode.isin(pcs)].assign(w=lambda d: d.pcode.map(_A))
    s = s.groupby(["y", "m"]).apply(lambda d: np.average(d.mm, weights=d.w), include_groups=False).rename("mm").reset_index()
    _g.append(s.assign(pcode=k))
ER = pd.concat([ER[["pcode", "y", "m", "mm"]]] + _g, ignore_index=True)
def season_tot(df, ms):
    g = df[df.m.isin(ms)].groupby(["pcode", "y"]).agg(n=("mm", "size"), mm=("mm", "sum")).reset_index()
    return g[g.n == len(ms)].pivot(index="y", columns="pcode", values="mm")
