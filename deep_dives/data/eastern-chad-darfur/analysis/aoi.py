"""Area of interest: Chad's three border provinces and the five Darfur states.

The two sides are kept apart throughout (one point per year per side). The five
Darfur states are summed into Greater Darfur so that series which still use the
three pre-2012 states (FAO ASIS, FEWS NET yields before 2012) line up.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

import os

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blob_sync import DATA, REPO  # noqa: E402,F401  work folder (ECD_WORK, mirrored on dev blob)

RAW = DATA / "raw"
PROC = DATA / "processed"
PAGE_DATA = PROC / "page_data.json"  # the public page's data; on blob, never committed

SIDES = {
    "chad": {
        "label": "Eastern Chad",
        "iso3": "TCD",
        "adm1": {"TCD14": "Ouaddaï", "TCD17": "Wadi Fira", "TCD21": "Sila"},
    },
    "darfur": {
        "label": "Darfur",
        "iso3": "SDN",
        "adm1": {
            "SD02": "North Darfur",
            "SD03": "South Darfur",
            "SD04": "West Darfur",
            "SD05": "East Darfur",
            "SD06": "Central Darfur",
        },
    },
}

# lon_min, lat_min, lon_max, lat_max — covers both sides with a margin, on 0.5 deg edges
BBOX = (19.5, 8.5, 28.5, 20.5)

# Cells (0.5 deg) whose 1991-2020 June-September CHIRPS v2 mean is below this are
# treated as desert and left out of every area mean.
BELT_MIN_JJAS_MM = 100.0


def _read(iso: str, level: int) -> gpd.GeoDataFrame:
    z = RAW / "codab" / f"{iso}.shp.zip"
    return gpd.read_file(f"zip://{z}!{iso}_adm{level}.shp").to_crs(4326)


def load_adm(level: int = 1) -> gpd.GeoDataFrame:
    """Admin units of the AOI (level 1 or 2) with a `side` column."""
    out = []
    for side, s in SIDES.items():
        g = _read(s["iso3"].lower(), level)
        g = g[g["ADM1_PCODE"].isin(s["adm1"])].copy()
        g["side"] = side
        g["adm1_pcode"] = g["ADM1_PCODE"]
        g["adm1_name"] = g["ADM1_PCODE"].map(s["adm1"])
        if level == 2:
            name = "ADM2_EN" if "ADM2_EN" in g else "ADM2_FR"
            g["adm2_pcode"] = g["ADM2_PCODE"]
            g["adm2_name"] = g[name]
        keep = ["side", "adm1_pcode", "adm1_name"] + (
            ["adm2_pcode", "adm2_name"] if level == 2 else []
        )
        out.append(g[keep + ["geometry"]])
    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=4326)
