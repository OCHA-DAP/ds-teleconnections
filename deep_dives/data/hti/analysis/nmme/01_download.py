"""Download NMME Nov-start prec (Haiti box) + Nino3.4 SST (server-side box mean) and CHIRPS
from the IRI Data Library. Sequential, polite, idempotent (skips files that already exist).

Usage: uv run python -I 01_download.py <raw_dir>
"""
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(sys.argv[1])
RAW.mkdir(parents=True, exist_ok=True)
B = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"

# (model, segment, dataset path, first Nov start year, last Nov start year)
SEGMENTS = [
    ("CanESM5", "H", ".CanSIPS-IC4/.CanESM5/.HINDCAST/.MONTHLY", 1990, 2023),
    ("CanESM5", "F", ".CanSIPS-IC4/.CanESM5/.FORECAST/.MONTHLY", 2024, 2026),
    ("GEM5.2-NEMO", "H", ".CanSIPS-IC4/.GEM5.2-NEMO/.HINDCAST/.MONTHLY", 1990, 2023),
    ("GEM5.2-NEMO", "F", ".CanSIPS-IC4/.GEM5.2-NEMO/.FORECAST/.MONTHLY", 2024, 2026),
    ("CESM1", "A", ".COLA-RSMAS-CESM1/.MONTHLY", 1982, 2026),
    ("CCSM4", "A", ".COLA-RSMAS-CCSM4/.MONTHLY", 1982, 2026),
    ("CFSv2", "H", ".NCEP-CFSv2/.HINDCAST/.MONTHLY", 1982, 2010),
    # CFSv2 realtime: PENTAD_SAMPLES (28 members/yr, 20 for Nov 2026) is primary; EARLY_MONTH_SAMPLES
    # (5-12 members since 2022, 5 for Nov 2026) is kept as the "CFSv2early" sensitivity.
    ("CFSv2", "F", ".NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY", 2011, 2026),
    ("CFSv2early", "H", ".NCEP-CFSv2/.HINDCAST/.MONTHLY", 1982, 2010),
    ("CFSv2early", "F", ".NCEP-CFSv2/.FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY", 2011, 2026),
    # No Nov 2026 start posted yet (latest S = Oct 2026) -> skill-only rows
    ("GEOS-S2S", "H", ".NASA-GEOSS2S/.HINDCAST/.MONTHLY", 1981, 2016),
    ("GEOS-S2S", "F", ".NASA-GEOSS2S/.FORECAST/.MONTHLY", 2017, 2026),
    ("SPEAR", "H", ".GFDL-SPEAR/.HINDCAST/.MONTHLY", 1991, 2020),
    ("SPEAR", "F", ".GFDL-SPEAR/.FORECAST/.MONTHLY", 2021, 2026),
]
SST_MODELS = {"CanESM5", "GEM5.2-NEMO", "CESM1", "CCSM4", "CFSv2"}


def q(s):
    return urllib.parse.quote(s, safe="/.()-:")


def s_sel(y0, y1):
    return f"S/(0000 1 Nov {y0})(0000 1 Nov {y1})RANGEEDGES/S/12/STEP/"


def fetch(url, dest, tries=5):
    if dest.exists() and dest.stat().st_size > 0:
        print("skip", dest.name, flush=True)
        return True
    for i in range(tries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=600) as r:
                data = r.read()
            if not data[:4] in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
                raise RuntimeError(f"not netCDF: {data[:200]!r}")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            print(f"ok {dest.name} {len(data)/1e6:.2f} MB {time.time()-t0:.0f}s", flush=True)
            time.sleep(3)
            return True
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} {dest.name} try {i}", flush=True)
            if e.code < 500:
                return False
        except Exception as e:  # noqa: BLE001
            print(f"  err {dest.name} try {i}: {str(e)[:200]}", flush=True)
        time.sleep(10 * 2**i)
    return False


def main():
    chirps = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/"
              ".precipitation/X/(-74.6)(-71.5)RANGEEDGES/Y/(17.9)(20.2)RANGEEDGES/data.nc")
    fetch(q(chirps), RAW / "chirps_hti_monthly.nc")
    for model, seg, path, y0, y1 in SEGMENTS:
        # last S available for GEOS/SPEAR forecasts is Oct 2026 -> cap at 2025
        if model in ("GEOS-S2S", "SPEAR") and y1 == 2026:
            y1 = 2025
        prec = (f"{B}/{path}/.prec/" + s_sel(y0, y1) + "L/(4.5)(7.5)RANGE/"
                "X/(-76)(-70)RANGE/Y/(16)(22)RANGE/data.nc")
        fetch(q(prec), RAW / f"prec_{model}_{seg}.nc")
        if model in SST_MODELS:
            # CFSv2 FORECAST sst box-mean over 2011-2026 times out server-side (>600 s twice):
            # fetch only the Nov 2026 start; CFSv2 anomaly base = 1999-2010 hindcast Nov starts.
            sy0 = 2026 if (model == "CFSv2" and seg == "F") else y0
            sst = (f"{B}/{path}/.sst/" + s_sel(sy0, y1) + "L/(0.5)(6.5)RANGE/"
                   "X/(190)(240)RANGEEDGES/Y/(-5)(5)RANGEEDGES/[X/Y]average/data.nc")
            fetch(q(sst), RAW / f"nino34_{model}_{seg}.nc")


if __name__ == "__main__":
    main()
