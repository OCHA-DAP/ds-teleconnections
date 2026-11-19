"""CHIRPS v2.0 daily, 1 Apr-30 Nov, Guatemala/Honduras/El Salvador box, from the IRI Data Library (never the CHC server)."""
import sys, os, time, requests
out = sys.argv[1]; years = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
os.makedirs(out, exist_ok=True)
base = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/"
loc = "X/(-92.3)(-83.0)RANGEEDGES/Y/(12.9)(17.9)RANGEEDGES/"
for y in years:
    fp = os.path.join(out, f"chirps_cadc_an_{y}.nc")
    if os.path.exists(fp) and os.path.getsize(fp) > 1_000_000: continue
    last = "31 Oct" if y == 2026 else "30 Nov"
    url = base + loc + f"T/(1 Apr {y})({last} {y})RANGEEDGES/data.nc"
    for attempt in range(4):
        try:
            t = time.time(); r = requests.get(url, timeout=900)
            if r.status_code == 200 and r.content[:3] in (b"CDF", b"\x89HD"):
                open(fp + ".part", "wb").write(r.content); os.rename(fp + ".part", fp)
                print(y, len(r.content)//1e6, "MB", round(time.time()-t), "s", flush=True); break
            print(y, "status", r.status_code, r.content[:200], flush=True)
        except Exception as e:
            print(y, "err", e, flush=True)
        time.sleep(30 * (attempt + 1))
    time.sleep(2)
