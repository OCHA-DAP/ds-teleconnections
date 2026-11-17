"""Darfur cereal yields from the FEWS NET Data Warehouse (official statistics).

Sorghum + millet, traditional rainfed sector (the smallholder crop), main season.
Greater Darfur yield = summed production / summed harvested area over the Darfur
states reported that year (3 states to 2011, 5 from 2012).

The level of the series shifts several times (high in 1975-83, low 1985-2013, a jump in
2014-15, and again when the source changes to FAO CFSAM in 2016), so the anomaly is
taken against the surrounding years: yield as % of the median yield of the years
within 5 years either side that come from the same source (the year itself left
out; at least 4 years). `yield_pct_src_median` (whole same-source median) is kept
for comparison.

No Chad series exists in the Data Warehouse (checked: country_code=TD returns
nothing), so this link of the chain is Darfur only.

Writes data/processed/yields/darfur_yield.parquet
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW  # noqa: E402

URL = "https://fdw.fews.net/api/cropproductionfacts/?country_code=SD&format=csv"
SRC = RAW / "fdw" / "cropproductionfacts_SD.csv"
OUT = PROC / "yields"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        SRC.write_bytes(requests.get(URL, timeout=300).content)
    d = pd.read_csv(SRC, low_memory=False)
    d = d[
        d["admin_1"].astype(str).str.contains("Darfur")
        & d["product"].isin(["Sorghum", "Millet"])
        & d["crop_production_system"].eq("traditional_rainfed")
        & d["season_name"].eq("Main")
        & d["status"].eq("Collected")
        & d["indicator"].isin(["Quantity Produced", "Area Harvested"])
    ].copy()
    d["year"] = d["season_year"].str[-4:].astype(int)
    d["source"] = d["source_organization"].str.contains("CFSAM").map({True: "FAO CFSAM",
                                                                       False: "Ministry"})
    w = d.pivot_table(index=["year", "source", "admin_1", "product"], columns="indicator",
                      values="value", aggfunc="sum").dropna()
    w = w[w["Area Harvested"] > 0].reset_index()
    g = w.groupby(["year", "source"]).agg(
        prod_t=("Quantity Produced", "sum"),
        area_ha=("Area Harvested", "sum"),
        n_states=("admin_1", "nunique"),
    ).reset_index()
    # a year reported by both sources (2016) keeps the CFSAM figure
    g = g.sort_values("source").drop_duplicates("year", keep="first").sort_values("year")
    g["yield_t_ha"] = g["prod_t"] / g["area_ha"]
    g["yield_pct_src_median"] = 100 * g["yield_t_ha"] / g.groupby("source")[
        "yield_t_ha"].transform("median")

    def local(row, k=5):
        w = g[(g["source"] == row["source"]) & ((g["year"] - row["year"]).abs() <= k)
              & (g["year"] != row["year"])]
        return 100 * row["yield_t_ha"] / w["yield_t_ha"].median() if len(w) >= 4 else None
    g["yield_pct_local"] = g.apply(local, axis=1)
    g.to_parquet(OUT / "darfur_yield.parquet", index=False)
    print(g.tail(30).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
