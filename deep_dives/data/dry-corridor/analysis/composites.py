"""Season and sub-season rainfall (% of the 1991-2020 median) by ENSO situation, national and framework areas, CHIRPS and ERA5.
Groups: 'after' = after a strong El Niño winter; 'during' = El Niño (concurrent Niño3.4 >= +0.5) during the season, not
'after'; 'nina' = concurrent <= -0.5; 'other'. Usage: composites.py SP -> composites.csv, subseason_after.csv"""
import sys, numpy as np, pandas as pd
SP = sys.argv[1]
exec(open(f"{SP}/scripts/common.py").read())
SUB = {"primera": [5, 6, 7, 8], "MJ": [5, 6], "JA": [7, 8], "postrera": [9, 10, 11], "SO": [9, 10], "ON": [10, 11]}
CONCS = {"primera": [5, 6, 7], "MJ": [5, 6, 7], "JA": [7, 8, 9], "postrera": [9, 10, 11], "SO": [9, 10, 11], "ON": [9, 10, 11]}
PCS = ["GT", "HN", "SV", "GT_AOI", "HN_AOI", "SV_AOI"]
rows, sub = [], []
for src, X in [("CHIRPS", CH), ("ERA5", ER)]:
    for s, ms in SUB.items():
        S = season_tot(X, ms); P = 100 * S / S.loc[1991:2020].median()
        yrs = [y for y in range(1981, 2026) if y in P.index]
        c = {y: nseas(y, CONCS[s]) for y in yrs}
        G = {"after": AFTER, "during": [y for y in yrs if c[y] >= 0.5 and y not in AFTER],
             "nina": [y for y in yrs if c[y] <= -0.5 and y not in AFTER], }
        G["other"] = [y for y in yrs if y not in sum(G.values(), [])]
        for g, ys in G.items():
            for pc in PCS:
                v = P.loc[ys, pc]; q = P.loc[yrs, pc].quantile(1 / 3)
                rows.append(dict(src=src, season=s, group=g, pcode=pc, n=len(ys), median=v.median(), mean=v.mean(),
                                 driest_third=int((v < q).sum()), years=" ".join(map(str, ys))))
        for pc in PCS:
            sub.append(dict(src=src, season=s, pcode=pc, **{str(y): P.loc[y, pc] for y in AFTER},
                            y2026=P.loc[2026, pc] if 2026 in P.index else np.nan))
C = pd.DataFrame(rows); C.to_csv(f"{SP}/composites.csv", index=False)
B = pd.DataFrame(sub); B.to_csv(f"{SP}/subseason_after.csv", index=False)
pd.set_option("display.width", 250)
for s in SUB:
    t = C[(C.season == s)].pivot_table(index=["group"], columns=["src", "pcode"], values="median").round(0)
    print("\n", s, "median % of typical"); print(t.to_string())
    print(C[(C.season == s) & (C.src == "CHIRPS") & (C.pcode == "GT")][["group", "n", "years"]].to_string(index=False))
print("\nAfter-years by sub-season, CHIRPS"); print(B[B.src == "CHIRPS"].round(0).to_string(index=False))
