# /// script
# requires-python = ">=3.11"
# dependencies = ["eccodes", "numpy", "pandas", "pyarrow", "requests"]
# ///
"""ECMWF IFS forecasts of total precipitation over Gaza, from ECMWF's open-data archive on AWS.

ECMWF publishes its real-time open data (CC-BY-4.0) at 0.25° and keeps it in a public bucket
(https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/, 0.25° from early 2024; layout
YYYYMMDD/HHz/ifs/0p25/{oper,enfo}/, without the ifs/ level in early 2024). Each GRIB file has a
.index of byte ranges, so only the total-precipitation messages are downloaded: the HRES run
(stream oper) and the 50 perturbed ENS members (stream enfo). tp is accumulated from the start of
the run; window totals are differences between two steps.

Kept per message: the 0.25° box 30.25–32.5°N, 33.5–35.75°E and the Gaza mean, weighted by the share of
the Gaza Strip (COD-AB PS02) in each 0.25° cell — the same weights as the page's ERA5 cells.

    uv run --script tools/ifs_open_data_tp.py OUT.parquet --runs 2026110400,2026110412 --valid 2026110600,2026110800
    uv run --script tools/ifs_open_data_tp.py OUT.parquet --window 2026-11-06 --leads 1,3,5   # runs 1/3/5 days before a 48 h window

Output rows: run, step, valid, member (hres, pf01…pf50), gaza (mm accumulated since the run), box.
Appends to OUT and skips runs already complete, so an interrupted download resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import eccodes
import numpy as np
import pandas as pd
import requests

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
# share of the Gaza Strip in each 0.25° cell (cell centre lat_lon), from levant_deep_dive.overlap_weights
WEIGHTS = {(31.25, 34.25): 0.43929670427579404, (31.5, 34.25): 0.11828360426486645, (31.5, 34.5): 0.4424196914593388}
LAT = np.round(np.arange(32.5, 30.24, -0.25), 2)
LON = np.round(np.arange(33.5, 35.76, 0.25), 2)


def _get(url: str, headers: dict | None = None, ok=(200,), tries: int = 25) -> requests.Response:
    """GET with retries. The bucket answers a share of parallel range requests with a quick 503 SlowDown;
    a short jittered pause is enough (a long exponential backoff stalls the whole download)."""
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code in ok or r.status_code == 404:
                return r
        except requests.RequestException:
            pass
        time.sleep((0.3 + random.random()) * (1 if k < 10 else 4))
    raise RuntimeError(f"{url} failed after {tries} tries")


def steps_ok(stream: str, step: int) -> bool:
    """Steps published for 00/12 UTC runs: 3-hourly to 144 h, then 6-hourly to 360 h (HRES and ENS)."""
    return (step <= 144 and step % 3 == 0) or (144 < step <= 360 and step % 6 == 0)


def messages(run: datetime, step: int, stream: str) -> list[tuple]:
    d, hh = run.strftime("%Y%m%d"), run.hour
    typ = "fc" if stream == "oper" else "ef"
    for lvl in ("ifs/0p25", "0p25"):
        root = f"{BASE}/{d}/{hh:02d}z/{lvl}/{stream}/{d}{hh:02d}0000-{step}h-{stream}-{typ}"
        r = _get(root + ".index")
        if r.status_code == 200:
            break
    else:
        print(f"  missing: {run:%Y-%m-%d %HZ} +{step}h {stream}", flush=True)
        return []
    out = []
    for line in r.text.splitlines():
        j = json.loads(line)
        if j.get("param") != "tp" or j.get("type") == "cf":
            continue
        mem = "hres" if stream == "oper" else f"pf{int(j['number']):02d}"
        out.append((root + ".grib2", j["_offset"], j["_length"], mem, run, step))
    return out


def fetch(url, off, ln, mem, run, step) -> dict:
    r = _get(url, headers={"Range": f"bytes={off}-{off + ln - 1}"}, ok=(200, 206))
    if len(r.content) != ln:
        raise RuntimeError(f"short read {url}")
    h = eccodes.codes_new_from_message(r.content)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        la0, lo0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dla, dlo = eccodes.codes_get(h, "jDirectionIncrementInDegrees"), eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        units = eccodes.codes_get(h, "units")
        v = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    lats = np.round(la0 - dla * np.arange(nj), 3)
    lons = np.round(lo0 + dlo * np.arange(ni), 3); lons = np.where(lons >= 180, lons - 360, lons)
    iy = [int(np.argmin(abs(lats - a))) for a in LAT]; ix = [int(np.argmin(abs(lons - o))) for o in LON]
    assert np.allclose(lats[iy], LAT) and np.allclose(lons[ix], LON), "grid is not the 0.25° open-data grid"
    bx = v[np.ix_(iy, ix)] * (1000.0 if units == "m" else 1.0)
    gm = sum(w * bx[list(LAT).index(la), list(LON).index(lo)] for (la, lo), w in WEIGHTS.items()) / sum(WEIGHTS.values())
    return dict(run=pd.Timestamp(run), step=step, valid=pd.Timestamp(run + timedelta(hours=step)), member=mem,
                gaza=float(gm), box=bx.astype("float32").ravel().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", help="comma-separated YYYYMMDDHH")
    ap.add_argument("--valid", help="comma-separated YYYYMMDDHH: the step boundaries wanted")
    ap.add_argument("--window", help="YYYY-MM-DD: a 48 h window from 00 UTC; with --leads")
    ap.add_argument("--leads", default="1,3,5", help="days before the window start (00 UTC runs)")
    ap.add_argument("--streams", default="oper,enfo")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    p = lambda s: datetime.strptime(s, "%Y%m%d%H")
    if a.window:
        w0 = datetime.fromisoformat(a.window)
        runs = [w0 - timedelta(days=int(k)) for k in a.leads.split(",")]
        valids = [w0, w0 + timedelta(hours=48)]
    else:
        runs, valids = [p(r) for r in a.runs.split(",")], [p(v) for v in a.valid.split(",")]
    streams = a.streams.split(",")
    have = pd.read_parquet(a.out) if os.path.exists(a.out) else None
    for run in runs:
        jobs = [(run, int((vt - run).total_seconds() // 3600), st) for vt in valids for st in streams]
        jobs = [j for j in jobs if j[1] > 0 and steps_ok(j[2], j[1])]
        if have is not None:
            sub = have[have.run == pd.Timestamp(run)]
            got = sub.groupby([sub.step, sub.member.eq("hres")]).size()
            jobs = [j for j in jobs if got.get((j[1], j[2] == "oper"), 0) < (1 if j[2] == "oper" else 50)]
        if not jobs:
            continue
        with ThreadPoolExecutor(a.threads) as ex:
            per_file = list(ex.map(lambda j: messages(*j), jobs))
        # interleave the files so that parallel requests mostly hit different objects
        msgs = [m for grp in zip(*[iter(f) for f in per_file]) for m in grp] if per_file else []
        msgs += [m for f in per_file for m in f[min(map(len, per_file)):]] if per_file else []
        with ThreadPoolExecutor(a.threads) as ex:
            rows = list(ex.map(lambda m: fetch(*m), msgs))
        df = pd.DataFrame(rows)
        have = df if have is None else pd.concat([have, df]).drop_duplicates(["run", "step", "member"], keep="last").reset_index(drop=True)
        have.to_parquet(a.out)
        print(f"{run:%Y-%m-%d %HZ}: {len(msgs)} messages (total {len(have)})", flush=True)


if __name__ == "__main__":
    main()
