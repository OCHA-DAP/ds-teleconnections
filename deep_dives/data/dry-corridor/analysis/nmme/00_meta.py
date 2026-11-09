"""Check latest start (S) and lead axis (L) of candidate NMME forecast datasets (prec), sequentially.

Usage: uv run python -I 00_meta.py <out_json>
"""
import json
import sys
import time
import urllib.request

B = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"
DS = {
    "SPEAR_F": ".GFDL-SPEAR/.FORECAST/.MONTHLY",
    "GEOS_F": ".NASA-GEOSS2S/.FORECAST/.MONTHLY",
}


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            print("  retry", i, e, file=sys.stderr)
            time.sleep(5 * 2**i)
    return None


def ym(s):
    s = int(s)
    return f"{1960 + s // 12}-{s % 12 + 1:02d}"


out = {}
for k, p in DS.items():
    txt = get(f"{B}/{p}/.prec/dods.asc?S,L")
    time.sleep(2)
    g, cur = {}, None
    for line in (txt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isalpha():
            cur = line.split(",")[0]
            g[cur] = []
        elif cur:
            g[cur] += [float(x) for x in line.split(",") if x.strip()]
    out[k] = g
    S = g.get("S", [])
    print(k, "S:", len(S), ym(S[0]) if S else None, "->", ym(S[-1]) if S else None,
          "last 3:", [ym(s) for s in S[-3:]], "L:", g.get("L"), flush=True)
json.dump(out, open(sys.argv[1], "w"))
