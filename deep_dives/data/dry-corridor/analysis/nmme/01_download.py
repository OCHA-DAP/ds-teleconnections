"""Download NMME Nov-start prec (Central American Dry Corridor box) + Nino3.4 SST (server-side box mean)
from the IRI Data Library. Sequential, polite, idempotent (skips files that already exist).

prec: X(-93)(-82) Y(12)(19), L 6.5-11.5 (May-Oct of the following year; MJJA = 6.5-9.5, SO = 10.5-11.5).
sst : Nino3.4 box mean (190-240E, 5S-5N), L 0.5-11.5.
CHIRPS is NOT fetched here (a separate monthly file is passed to 02).

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
    # CFSv2 realtime: PENTAD_SAMPLES is primary; EARLY_MONTH_SAMPLES is the "CFSv2early" sensitivity.
    ("CFSv2", "F", ".NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY", 2011, 2026),
    ("CFSv2early", "H", ".NCEP-CFSv2/.HINDCAST/.MONTHLY", 1982, 2010),
    ("CFSv2early", "F", ".NCEP-CFSv2/.FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY", 2011, 2026),
    # Re-checked 2026-11-09 (00_meta.py): latest SPEAR start = Oct 2026 -> skill-only, Nov starts to 2025.
    # GEOS-S2S also stops at Oct 2026 AND its leads end at L 8.5 (Jul), so it cannot form MJJA or SO:
    # not downloaded.
    ("SPEAR", "H", ".GFDL-SPEAR/.HINDCAST/.MONTHLY", 1991, 2020),
    ("SPEAR", "F", ".GFDL-SPEAR/.FORECAST/.MONTHLY", 2021, 2025),
]
SST_MODELS = {"CanESM5", "GEM5.2-NEMO", "CESM1", "CCSM4", "CFSv2"}  # live models (Nov 2026 start posted)


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
            if data[:4] not in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
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


BOXES = {  # SST index boxes, 0-360 longitudes
    "nino34": "X/(190)(240)RANGEEDGES/Y/(-5)(5)RANGEEDGES/",  # 170W-120W, 5S-5N
    "tna": "X/(302.5)(345)RANGEEDGES/Y/(5.5)(23.5)RANGEEDGES/",  # 57.5W-15W, 5.5N-23.5N (CPC/PSL TNA)
}


def sst_url(path, y0, y1, box="nino34"):
    return (f"{B}/{path}/.sst/" + s_sel(y0, y1) + "L/(0.5)(11.5)RANGE/" + BOXES[box] + "[X/Y]average/data.nc")


def sst_boxmeans(box):
    for model, seg, path, y0, y1 in SEGMENTS:
        if model not in SST_MODELS:
            continue
        dest = RAW / f"{box}_{model}_{seg}.nc"
        if model == "CFSv2" and seg == "F":
            # As for Haiti: the CFSv2 FORECAST (PENTAD_SAMPLES) sst box mean over 2011-2026 times out
            # server-side (>600 s twice in Nov 2026) -> fetch only the Nov 2026 start;
            # CFSv2 anomaly base = 1999-2010 hindcast Nov starts.
            fetch(q(sst_url(path, 2026, 2026, box)), dest, tries=3)
            continue
        if fetch(q(sst_url(path, y0, y1, box)), dest, tries=2):
            continue
        # timeout fallback for other segments: two halves of the start range
        mid = (y0 + y1) // 2
        ok1 = fetch(q(sst_url(path, y0, mid, box)), RAW / f"{box}_{model}_{seg}a.nc", tries=2)
        ok2 = fetch(q(sst_url(path, mid + 1, y1, box)), RAW / f"{box}_{model}_{seg}b.nc", tries=2)
        print(f"  fallback halves {box} {model} {seg}: {ok1} {ok2}", flush=True)


def main():
    for model, seg, path, y0, y1 in SEGMENTS:
        prec = (f"{B}/{path}/.prec/" + s_sel(y0, y1) + "L/(6.5)(11.5)RANGE/"
                "X/(-93)(-82)RANGE/Y/(12)(19)RANGE/data.nc")
        fetch(q(prec), RAW / f"prec_{model}_{seg}.nc")
    sst_boxmeans("nino34")
    sst_boxmeans("tna")
    # land-mask check for the TNA box (it touches West Africa): one field (Nov 2026 start, L 0.5, first member)
    # per live model, ungridded, to compare with the server-side box mean.
    for model, seg, path, y0, y1 in SEGMENTS:
        if model in SST_MODELS and y1 == 2026 and model != "CFSv2early":
            url = (f"{B}/{path}/.sst/S/(0000 1 Nov 2026)VALUE/L/(0.5)VALUE/M/(1)VALUE/" + BOXES["tna"] + "data.nc")
            fetch(q(url), RAW / f"tnagrid_{model}.nc", tries=2)


if __name__ == "__main__":
    main()
