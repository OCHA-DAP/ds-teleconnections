"""ACLED CAST (Conflict Alert System) forecasts for Chad and Sudan, all rows the API returns.

CAST forecasts the number of political violence events per admin-1 and month (total, battles,
explosions/remote violence, violence against civilians), up to six months ahead, and stores
the observed count once the month is over. Restricted like the events: kept in <work>/raw/acled/.
Writes <work>/raw/acled/cast_tcd_sdn.parquet.
"""

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import RAW  # noqa: E402

import importlib  # noqa: E402

acled = importlib.import_module("10_acled")
API = "https://acleddata.com/api/cast/read"


def fetch(country: str, tok: str) -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        params = {"country": country, "limit": 5000, "_format": "json"}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(4):
            try:
                r = requests.get(API, params=params, headers={"Authorization": f"Bearer {tok}"}, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        j = r.json()
        rows += j.get("data", [])
        cursor = j.get("next_cursor") or r.headers.get("X-Next-Cursor")
        if not j.get("data") or not cursor:
            return pd.DataFrame(rows)


def main():
    tok = acled.token()
    df = pd.concat([fetch(c, tok) for c in ("Chad", "Sudan")], ignore_index=True)
    df["download_date"] = date.today().isoformat()
    df.to_parquet(RAW / "acled" / "cast_tcd_sdn.parquet", index=False)
    print(f"[cast] {len(df)} rows")


if __name__ == "__main__":
    main()
