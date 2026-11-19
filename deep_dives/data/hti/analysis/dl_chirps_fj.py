import sys, os, time, requests
out = sys.argv[1]; years = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
base = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/"
loc = "X/(-74.55)(-71.55)RANGEEDGES/Y/(17.95)(20.15)RANGEEDGES/"
for y in years:
    fp = os.path.join(out, f"chirps_hti_fj_{y}.nc")
    if os.path.exists(fp) and os.path.getsize(fp) > 50000: continue
    url = base + loc + f"T/(1 Feb {y})(31 Jul {y})RANGEEDGES/data.nc"
    for attempt in range(4):
        try:
            t = time.time(); r = requests.get(url, timeout=600)
            if r.status_code == 200 and r.content[:3] in (b"CDF", b"\x89HD"):
                open(fp, "wb").write(r.content); print(y, len(r.content)//1e6, "MB", round(time.time()-t), "s", flush=True); break
            print(y, "status", r.status_code, r.content[:200], flush=True)
        except Exception as e:
            print(y, "err", e, flush=True)
        time.sleep(30 * (attempt + 1))
    time.sleep(2)
