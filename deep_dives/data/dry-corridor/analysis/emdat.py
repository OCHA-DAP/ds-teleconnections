"""EM-DAT drought records for Guatemala, Honduras and El Salvador since 1981 (team blob snapshot, read-only), with the
ENSO situation of the start year: 'el_nino' if Niño3.4 (1991-2020 base) reached +0.5 in Jun-Aug or Sep-Nov of that year,
'after' if the preceding Dec-Feb was >= +1.0 (and not el_nino), else 'other'. Usage: emdat.py SP -> emdat_droughts.csv"""
import sys, numpy as np, pandas as pd, ocha_stratus as stratus
SP = sys.argv[1]
exec(open(f"{SP}/scripts/common.py").read())
rows = []
for iso in ("GTM", "HND", "SLV"):
    d = stratus.emdat.load_emdat_from_blob(iso, include_historic=True, stage="dev")
    d = d[d["Disaster Type"].str.contains("Drought", na=False) & (d["Start Year"] >= 1981)]
    rows.append(d)
E = pd.concat(rows)
E = E[["DisNo.", "ISO", "Start Year", "Start Month", "End Year", "End Month", "Total Affected"]].rename(columns={"Start Year": "y"})
def situ(y):
    jja, son = nseas(y, [6, 7, 8]), nseas(y, [9, 10, 11])
    if max(jja, son if not np.isnan(son) else -9) >= 0.5: return "el_nino"
    if y in AFTER: return "after"
    return "other"
E["situation"] = E.y.map(situ); E["jja"] = E.y.map(lambda y: round(nseas(y, [6, 7, 8]), 2)); E["son"] = E.y.map(lambda y: round(nseas(y, [9, 10, 11]), 2))
E = E.sort_values(["y", "ISO"]); E.to_csv(f"{SP}/emdat_droughts.csv", index=False)
print(E.to_string(index=False))
Y = E.drop_duplicates("y"); print("start years:", len(Y), Y.groupby("situation").y.apply(list).to_dict())
print("newest record start:", E.y.max())
