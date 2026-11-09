"""Per-cell season timing from CHIRPS daily (1 Apr-30 Nov), 1981-2026.
  onset   Primera onset: first day from 15 Apr with >=1 mm and a 3-day total >=20 mm, not followed by a dry spell of
          >=10 days (<1 mm) within the next 30 days; searched to 15 Jul (later = no onset, coded 16 Jul).
  canic   canícula: longest run of dry days (<1 mm) between 1 Jul and 31 Aug.
  end     Postrera end of rains: last day between 1 Sep and 28 Nov that starts a 3-day total >=20 mm (no such day = 1 Sep).
  dry_so  longest run of dry days between 1 Sep and 31 Oct.
Dates are stored as days after 1 Apr. Usage: daily_metrics.py SP"""
import sys, glob, os, numpy as np, pandas as pd, xarray as xr
SP = sys.argv[1]
W = pd.read_parquet(f"{SP}/weights_daily.parquet")
pix = W[["ix", "iy"]].drop_duplicates().sort_values(["ix", "iy"]).reset_index(drop=True)
WET, ACC = 1.0, 20.0
def idx(dates, m, d):
    hit = np.where((dates.month == m) & (dates.day == d))[0]
    return int(hit[0]) if len(hit) else None
def longest(dry, a, b):
    run = np.zeros(dry.shape[1], np.int16); best = run.copy()
    for t in range(a, b + 1):
        run = np.where(dry[t], run + 1, 0).astype(np.int16); best = np.maximum(best, run)
    return best
out = []
for fp in sorted(glob.glob(f"{SP}/chirps_d/chirps_cadc_an_*.nc")):
    y = int(fp[-7:-3]); ds = xr.open_dataset(fp, decode_times=False)
    dates = pd.DatetimeIndex(pd.to_datetime(np.floor(ds["T"].values.astype("float64") - 2440587.5), unit="D"))
    assert dates[0] == pd.Timestamp(f"{y}-04-01") and (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all(), (y, dates[0])
    P = ds["prcp"].transpose("T", "X", "Y").values[:, pix.ix.values, pix.iy.values].astype("float32")  # (T, N)
    ok = ~np.isnan(P).any(0)
    P = np.nan_to_num(P)
    T, N = P.shape
    dry = P < WET
    s3 = np.full_like(P, np.nan); s3[:T-2] = P[:-2] + P[1:-1] + P[2:]
    run = np.zeros((T, N), np.int16); r = np.zeros(N, np.int16)
    for t in range(T):
        r = np.where(dry[t], r + 1, 0).astype(np.int16); run[t] = r
    D10 = (run >= 10).astype(np.int32); C = np.vstack([np.zeros((1, N), np.int32), np.cumsum(D10, 0)])
    rec = dict(y=np.full(N, y), ix=pix.ix.values, iy=pix.iy.values, ok=ok)
    a, b = idx(dates, 4, 15), idx(dates, 7, 15)
    onset = np.full(N, np.nan)
    if b is not None and b + 32 < T:
        for d in range(a, b + 1):
            bad = (C[d + 33] - C[d + 12]) > 0
            good = (P[d] >= WET) & (s3[d] >= ACC) & ~bad & np.isnan(onset)
            onset[good] = d
        onset[np.isnan(onset)] = b + 1
    rec["onset"] = onset
    j1, a31 = idx(dates, 7, 1), idx(dates, 8, 31)
    rec["canic"] = longest(dry, j1, a31).astype(float) if a31 is not None else np.full(N, np.nan)
    s1, o31, n28 = idx(dates, 9, 1), idx(dates, 10, 31), idx(dates, 11, 28)
    rec["dry_so"] = longest(dry, s1, o31).astype(float) if o31 is not None else np.full(N, np.nan)
    if n28 is not None:
        hit = s3[s1:n28 + 1] >= ACC
        last = np.where(hit.any(0), s1 + (hit.shape[0] - 1 - np.argmax(hit[::-1], 0)), s1)
        rec["end"] = last.astype(float)
    else:
        rec["end"] = np.full(N, np.nan)
    out.append(pd.DataFrame(rec)); print(y, T, dates[-1].date(), int(ok.sum()), flush=True)
O = pd.concat(out, ignore_index=True); O.to_parquet(f"{SP}/daily_metrics_pixels.parquet")
print(O.groupby("y")[["onset", "canic", "end", "dry_so"]].median().tail(8))
