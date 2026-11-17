"""ACLED events for Chad and Sudan, all years, from the ACLED API (OAuth password grant).

Restricted data: ACLED limits redistribution, so the events stay in the work folder
(<work>/raw/acled/) and only counts reach the (password-protected) page.

Needs ACLED_USERNAME / ACLED_PASSWORD in the environment (the team's ds-acled-fetcher uses
the same account). Writes <work>/raw/acled/acled_tcd_sdn.parquet (all fields as delivered,
plus `download_date`).
"""

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import RAW  # noqa: E402

TOKEN_URL = "https://acleddata.com/oauth/token"
API = "https://acleddata.com/api/acled/read"
OUT = RAW / "acled"
PAGE = 5000


def token() -> str:
    r = requests.post(TOKEN_URL, data={"grant_type": "password", "client_id": "acled",
                                       "username": os.environ["ACLED_USERNAME"],
                                       "password": os.environ["ACLED_PASSWORD"]}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def fetch(country: str, tok: str) -> pd.DataFrame:
    """All events for one country; the API pages with a cursor (next_cursor), not page=."""
    rows, cursor, page = [], None, 1
    while True:
        params = {"country": country, "limit": PAGE, "_format": "json"}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(4):
            try:
                r = requests.get(API, params=params, headers={"Authorization": f"Bearer {tok}"},
                                 timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                print(f"  retry {country} p{page}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        j = r.json()
        d = j.get("data", [])
        rows += d
        cursor = j.get("next_cursor") or r.headers.get("X-Next-Cursor")
        print(f"  {country} page {page}: {len(d)} (total {len(rows)})", flush=True)
        if not d or not cursor:
            return pd.DataFrame(rows)
        page += 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    df = pd.concat([fetch(c, tok) for c in ("Chad", "Sudan")], ignore_index=True)
    df["download_date"] = date.today().isoformat()
    df.to_parquet(OUT / "acled_tcd_sdn.parquet", index=False)
    print(f"[acled] {len(df)} events, {df['event_date'].min()} .. {df['event_date'].max()}")


if __name__ == "__main__":
    main()
