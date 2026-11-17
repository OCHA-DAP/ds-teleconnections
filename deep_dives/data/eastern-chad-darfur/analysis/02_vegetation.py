"""Vegetation per side and year: FAO ASIS (1984-2026) and JRC ASAP (2001-2026).

Every year's value is computed the same way from the dekadal series, so 2026 (season
not closed in the annual summaries yet) is comparable with the past:

  asi    FAO ASIS Agricultural Stress Index, cropland: mean of the dekads 1 Jul - 21 Oct
         (% of cropland with vegetation health below 35).
  vhi    FAO ASIS Vegetation Health Index: mean of the dekads 1 Jul - 21 Oct (0-1).
  zfparc_crop / zfparc_range
         JRC ASAP cumulative-FPAR z-score ("biomass"), cropland / rangeland during the
         growing cycle: value at the dekad of 21 Oct (the last dekad with a value between
         21 Sep and 1 Nov when 21 Oct is missing).

Side value = mean of the units weighted by their area inside the rainfall belt (cells
with >= 100 mm in an average June-September; cropland area per unit is not published).
ASIS reports Darfur as the three pre-2012 states and Wadi Fira under its old name
Biltine; the weights map them onto the current states.

Writes data/processed/veg/{yearly_side,yearly_unit,dekadal_side}.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW  # noqa: E402

OUT = PROC / "veg"

# unit (as the source names it) -> (side, list of COD adm1 pcodes it covers)
ASIS_UNITS = {
    "Biltine": ("chad", ["TCD17"]),
    "Ouaddai": ("chad", ["TCD14"]),
    "Sila": ("chad", ["TCD21"]),
    "Northern Darfur": ("darfur", ["SD02"]),
    "Southern Darfur": ("darfur", ["SD03", "SD05"]),
    "Western Darfur": ("darfur", ["SD04", "SD06"]),
}
ASAP_UNITS = {
    "Wadi Fira": ("chad", ["TCD17"]),
    "Ouaddaï": ("chad", ["TCD14"]),
    "Sila": ("chad", ["TCD21"]),
    "North Darfur": ("darfur", ["SD02"]),
    "South Darfur": ("darfur", ["SD03"]),
    "East Darfur": ("darfur", ["SD05"]),
    "West Darfur": ("darfur", ["SD04"]),
    "Central Darfur": ("darfur", ["SD06"]),
}


def belt_weights():
    """Belt area (cos-lat weighted 0.5 deg cell fractions) per COD adm1 pcode."""
    import geopandas as gpd  # noqa: F401  (load_adm needs it)
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from aoi import BBOX, load_adm

    g = pd.read_parquet(PROC / "rain" / "grid_2026.parquet")
    lat = np.sort(g["lat"].unique())
    lon = np.sort(g["lon"].unique())
    belt = g.pivot(index="lat", columns="lon", values="belt").loc[lat, lon].values
    a1 = load_adm(1)
    f, res = 50, 0.5
    x0, y0, x1, y1 = BBOX
    tr = from_origin(x0, y1, res / f, res / f)
    w = {}
    for pc, sub in a1.groupby("adm1_pcode"):
        m = rasterize(sub.geometry, out_shape=(lat.size * f, lon.size * f), transform=tr,
                      fill=0, default_value=1)
        fr = m.reshape(lat.size, f, lon.size, f).mean(axis=(1, 3))[::-1]
        w[pc] = float((fr * belt * np.cos(np.deg2rad(lat))[:, None]).sum())
    return w


def unit_weight(units, w):
    return {u: sum(w[p] for p in pcs) for u, (_, pcs) in units.items()}


def asis_dekadal(kind):
    fn = {"asi": "ASI_Dekad_Season1_data.csv", "vhi": "vhi_adm1_dekad_data.csv"}[kind]
    out = []
    for iso in ("TCD", "SDN"):
        d = pd.read_csv(RAW / "asis" / f"{iso}_{fn}")
        d = d[d["Province"].isin(ASIS_UNITS)]
        out.append(pd.DataFrame({"unit": d["Province"], "date": pd.to_datetime(d["Date"]),
                                 "value": d["Data"].astype(float)}))
    d = pd.concat(out, ignore_index=True)
    d["indicator"] = kind
    return d


def asap_dekadal(name):
    out = []
    for iso in ("TCD", "SDN"):
        d = pd.read_csv(RAW / "asap" / f"{iso}_{name}.csv")
        d = d[d["region_name"].isin(ASAP_UNITS)]
        out.append(pd.DataFrame({"unit": d["region_name"],
                                 "date": pd.to_datetime(d["date"].astype(str)),
                                 "value": d["value"].astype(float)}))
    d = pd.concat(out, ignore_index=True)
    d["indicator"] = name
    return d


def season_value(d, how):
    d = d.reset_index(drop=True).copy()
    d["year"] = d["date"].dt.year
    md = d["date"].dt.strftime("%m-%d")
    if how == "mean_jul_oct":
        s = d[(md >= "07-01") & (md <= "10-21")]
        return s.groupby(["indicator", "unit", "year"])["value"].agg(["mean", "count"]) \
            .query("count >= 10")["mean"].rename("value").reset_index()
    # end of season: 21 Oct, else the last value between 21 Sep and 1 Nov
    d["md"] = md
    s = d[(md >= "09-21") & (md <= "11-01")].dropna(subset=["value"]).copy()
    s["pri"] = (s["md"] != "10-21").astype(int)
    s = s.sort_values(["pri", "date"], ascending=[True, False])
    return s.groupby(["indicator", "unit", "year"]).head(1)[
        ["indicator", "unit", "year", "value"]]


def to_side(df, units, wts, key="year"):
    df = df.copy()
    df["side"] = df["unit"].map(lambda u: units[u][0])
    df["w"] = df["unit"].map(wts)
    g = df.groupby(["indicator", "side", key])
    out = g.apply(lambda x: pd.Series({"value": np.average(x["value"], weights=x["w"]),
                                       "n_units": len(x)}), include_groups=False)
    out = out.reset_index()
    n_need = {s: sum(1 for v in units.values() if v[0] == s) for s in ("chad", "darfur")}
    return out[out["n_units"] == out["side"].map(n_need)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = belt_weights()
    asis = pd.concat([asis_dekadal("asi"), asis_dekadal("vhi")])
    asap = pd.concat([asap_dekadal("zfparc_crop"), asap_dekadal("zfparc_range")])
    y_asis = season_value(asis, "mean_jul_oct")
    y_asap = season_value(asap, "eos")
    yu = pd.concat([y_asis, y_asap])
    yu.to_parquet(OUT / "yearly_unit.parquet", index=False)
    ys = pd.concat([to_side(y_asis, ASIS_UNITS, unit_weight(ASIS_UNITS, w)),
                    to_side(y_asap, ASAP_UNITS, unit_weight(ASAP_UNITS, w))])
    ys.to_parquet(OUT / "yearly_side.parquet", index=False)
    # dekadal side series for the within-season charts
    dk = []
    for d, units in ((asis, ASIS_UNITS), (asap, ASAP_UNITS)):
        d = d.dropna(subset=["value"])
        dk.append(to_side(d, units, unit_weight(units, w), key="date"))
    pd.concat(dk).to_parquet(OUT / "dekadal_side.parquet", index=False)
    p = ys.pivot_table(index="year", columns=["indicator", "side"], values="value")
    print(p.loc[2000:].round(2).to_string())


if __name__ == "__main__":
    main()
