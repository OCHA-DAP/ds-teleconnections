"""Fetch S/L/M grids for candidate NMME datasets via OPeNDAP ascii (sequential, polite)."""
import json, time, urllib.request, sys
B = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"
DS = {
 "CanESM5_F": ".CanSIPS-IC4/.CanESM5/.FORECAST/.MONTHLY",
 "CanESM5_H": ".CanSIPS-IC4/.CanESM5/.HINDCAST/.MONTHLY",
 "GEM52_F": ".CanSIPS-IC4/.GEM5.2-NEMO/.FORECAST/.MONTHLY",
 "GEM52_H": ".CanSIPS-IC4/.GEM5.2-NEMO/.HINDCAST/.MONTHLY",
 "CanSIPSIC4_F": ".CanSIPS-IC4/.FORECAST/.MONTHLY",
 "CanSIPSIC4_H": ".CanSIPS-IC4/.HINDCAST/.MONTHLY",
 "GEOS_F": ".NASA-GEOSS2S/.FORECAST/.MONTHLY",
 "GEOS_H": ".NASA-GEOSS2S/.HINDCAST/.MONTHLY",
 "SPEAR_F": ".GFDL-SPEAR/.FORECAST/.MONTHLY",
 "SPEAR_H": ".GFDL-SPEAR/.HINDCAST/.MONTHLY",
 "CESM1": ".COLA-RSMAS-CESM1/.MONTHLY",
 "CCSM4": ".COLA-RSMAS-CCSM4/.MONTHLY",
 "CFS_F": ".NCEP-CFSv2/.FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY",
 "CFS_H": ".NCEP-CFSv2/.HINDCAST/.MONTHLY",
}
def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode()
        except Exception as e:
            print("  retry", i, e, file=sys.stderr); time.sleep(5 * 2**i)
    return None
out = {}
for k, p in DS.items():
    d = {}
    for var in ["prec"]:
        txt = get(f"{B}/{p}/.{var}/dods.asc?S,L,M")
        time.sleep(2)
        if txt is None: d[var] = None; continue
        g = {}
        cur = None
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            if line[0].isalpha():
                cur = line.split(",")[0]; g[cur] = []
            elif cur:
                g[cur] += [float(x) for x in line.split(",") if x.strip()]
        d[var] = g
    out[k] = d
    g = d["prec"] or {}
    S = g.get("S", [])
    def ym(s): s=int(s); return f"{1960+s//12}-{s%12+1:02d}"
    print(k, "S:", len(S), ym(S[0]) if S else None, "->", ym(S[-1]) if S else None,
          "L:", g.get("L"), "M:", len(g.get("M", [])), flush=True)
json.dump(out, open(sys.argv[1], "w"))
