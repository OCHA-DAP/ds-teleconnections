import sys, requests, time
out_dir = sys.argv[1]
boxes = {"slb": (155.3, 170.4, -12.5, -4.9), "vut": (166.4, 170.4, -20.4, -12.9),
         "fji_e": (176.7, 180.0, -21.9, -12.3), "fji_w": (-180.0, -178.0, -21.9, -12.3),
         "ton": (-176.4, -173.5, -22.5, -15.4)}
only = sys.argv[2:] or list(boxes)
for k, (x0, x1, y0, y1) in boxes.items():
    if k not in only:
        continue
    url = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
           f"X/({x0})({x1})RANGEEDGES/Y/({y0})({y1})RANGEEDGES/data.nc")
    for a in range(4):
        try:
            t = time.time(); r = requests.get(url, timeout=1200)
            if r.status_code == 200 and r.content[:3] in (b"CDF", b"\x89HD"):
                open(f"{out_dir}/chirps_{k}.nc", "wb").write(r.content); print(k, len(r.content)/1e6, "MB", round(time.time()-t), "s"); break
            print(k, "status", r.status_code, r.content[:200])
        except Exception as e: print(k, "err", e)
        time.sleep(30*(a+1))
