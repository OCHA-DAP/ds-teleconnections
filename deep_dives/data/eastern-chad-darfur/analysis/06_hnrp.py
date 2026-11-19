"""HNRP layers for the map: intersectoral severity (2025, 2026) and people in need (2025).

Source: the team HPC mirror on the dev DB (hpc.severity_admin = JIAF final severity
from the HDX workbooks; hpc.needs_admin = HAPI intersectoral PiN). The 2026 HNRPs
publish PiN only nationally for Sudan and Chad, so the PiN layer is 2025.

Writes data/processed/hnrp/adm2.parquet  (adm2_pcode, side, sev_2025, sev_2026,
pin_2025, pop_2025)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import engine, stratus  # noqa: E402,F401
from aoi import PROC, load_adm  # noqa: E402

OUT = PROC / "hnrp"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = engine()
    a2 = load_adm(2)
    codes = a2["adm2_pcode"].tolist()
    sev = pd.read_sql(
        "select admin2_code, year, final_severity from hpc.severity_admin "
        "where iso3 in ('SDN','TCD') and admin2_code = any(%(c)s) and admin3_code is null",
        e, params={"c": codes})
    sev = sev.pivot_table(index="admin2_code", columns="year", values="final_severity",
                          aggfunc="max")
    sev.columns = [f"sev_{c}" for c in sev.columns]
    pin = pd.read_sql(
        "select admin2_code, population_status, population from hpc.needs_admin "
        "where location_code in ('SDN','TCD') and admin_level = 2 "
        "and sector_code = 'Intersectoral' and category = 'total' "
        "and reference_period_start = '2025-01-01' and admin2_code = any(%(c)s)",
        e, params={"c": codes})
    pin = pin.pivot_table(index="admin2_code", columns="population_status",
                          values="population", aggfunc="sum")
    pin = pin.rename(columns={"INN": "pin_2025", "all": "pop_2025"})[["pin_2025", "pop_2025"]]
    d = a2[["adm2_pcode", "adm2_name", "adm1_name", "side"]].merge(
        sev, left_on="adm2_pcode", right_index=True, how="left").merge(
        pin, left_on="adm2_pcode", right_index=True, how="left")
    d.to_parquet(OUT / "adm2.parquet", index=False)
    print(d.groupby("side")[["sev_2025", "sev_2026", "pin_2025", "pop_2025"]].agg(
        ["count", "sum", "max"]).round(0).to_string())
    print(d[d[["sev_2026", "pin_2025"]].isna().any(axis=1)][["side", "adm2_name"]].to_string())


if __name__ == "__main__":
    main()
