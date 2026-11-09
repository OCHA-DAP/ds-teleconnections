import sys, requests, time
out = sys.argv[1]
url = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
       "X/(-92.3)(-83.0)RANGEEDGES/Y/(12.9)(17.9)RANGEEDGES/data.nc")
for a in range(4):
    try:
        t=time.time(); r = requests.get(url, timeout=1200)
        if r.status_code == 200 and r.content[:3] in (b"CDF", b"\x89HD"):
            open(out, "wb").write(r.content); print(len(r.content)//1e6, "MB", round(time.time()-t), "s"); break
        print("status", r.status_code, r.content[:200])
    except Exception as e: print("err", e)
    time.sleep(30*(a+1))
