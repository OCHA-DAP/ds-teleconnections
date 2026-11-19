"""Inputs for the Pacific page, written to SP/data/ (read-only access everywhere).

usage: get_inputs.py SP
  - SEAS5 and ERA5 admin 0/1 monthly statistics for FJI, VUT, TON, SLB (and PNG) from the prod database
  - CERF allocations to Pacific countries, with the drought-period supplement, from the dev database (aa schema)
  - NOAA indices: CPC OISST Niño regions (sstoi.indices), CPC ONI, PSL Niño3.4 (ERSST v6)
  - COD-AB admin 0/1 boundaries (fieldmaps, via ocha-stratus)
  - GHCN-Daily station files for Fiji, Vanuatu, Tonga and Solomon Islands -> near-complete monthly totals
Database hosts come from the environment (DSCI_AZ_DB_{DEV,PROD}_*; through the laptop tunnel set *_HOST=127.0.0.1:1543x).
"""
import io
import os
import sys

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import requests
from sqlalchemy import text

SP = sys.argv[1].rstrip("/") + "/"
D = SP + "data/"
os.makedirs(D + "ghcnd", exist_ok=True)
ISO = ("FJI", "VUT", "TON", "SLB", "PNG")

with stratus.get_engine("prod").connect() as c:
    q = f"select * from public.{{t}} where iso3 in {ISO} and adm_level <= 1"
    pd.read_sql(text(q.format(t="seas5")), c, parse_dates=["valid_date", "issued_date"]).to_parquet(D + "seas5.parquet")
    pd.read_sql(text(q.format(t="era5")), c, parse_dates=["valid_date"]).to_parquet(D + "era5.parquet")

with stratus.get_engine("dev").connect() as c:
    pd.read_sql(text(
        "select a.application_code, a.year, a.country_iso3, a.window_name, a.emergency_type, a.title, a.amount_approved, "
        "a.erc_endorsement_date, s.not_tc, s.valid_month_start, s.valid_year_start, s.valid_month_end, s.valid_year_end, s.notes "
        "from aa.cerf_allocation a left join aa.cerf_supplement s using(application_code) "
        "where a.region_name in ('Melanesia', 'Micronesia', 'Polynesia') order by country_iso3, erc_endorsement_date"), c
    ).to_csv(SP + "cerf_pacific.csv", index=False)

for url, name in [("https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices", "sstoi.indices"),
                  ("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", "oni.ascii.txt"),
                  ("https://psl.noaa.gov/data/correlation/nina34.anom.data", "nina34_psl.data")]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    open(D + name, "w").write(r.text)

for iso in ("FJI", "VUT", "TON", "SLB"):
    for lvl in (0, 1):
        stratus.codab.load_codab_from_fieldmaps(iso, lvl).to_file(D + f"{iso.lower()}_adm{lvl}.gpkg")

STATIONS = ["FJM00091680", "FJN00077648", "FJM00091657", "FJM00091683", "FJM00091690", "FJM00091691",
            "NHM00091557", "NH000091554", "NH000091568", "BP000091517", "BP000091503", "TN000091780"]
monthly = {}
for sid in STATIONS:
    r = requests.get(f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{sid}.csv", timeout=300)
    r.raise_for_status()
    open(D + f"ghcnd/{sid}.csv", "w").write(r.text)
    d = pd.read_csv(io.StringIO(r.text), usecols=["DATE", "PRCP"], low_memory=False, parse_dates=["DATE"]).dropna(subset=["PRCP"])
    d["PRCP"] = d.PRCP / 10.0  # tenths of mm
    g = d.groupby(d.DATE.dt.to_period("M")).agg(mm=("PRCP", "sum"), n=("PRCP", "size"))
    g = g[g.n >= g.index.days_in_month - 2]  # near-complete months only
    m = g.mm
    m.index = m.index.to_timestamp()
    monthly[sid] = m
pd.to_pickle(monthly, D + "ghcnd_monthly.pkl")
print("inputs written to", D)
