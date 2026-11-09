"""Regression figures quoted in the page text, from season_results.csv. Usage: results_summary.py SP -> regression_summary.json"""
import sys, json, pandas as pd
SP = sys.argv[1]
R = pd.read_csv(f"{SP}/season_results.csv")
six = ["GT_AOI", "HN_AOI", "SV_AOI", "GT", "HN", "SV"]
P = R[(R.season == "primera") & R.pcode.isin(six)].set_index(["src", "pcode"]).sort_index()
Q = R[(R.season == "postrera") & R.pcode.isin(six)].set_index(["src", "pcode"]).sort_index()
r = lambda s: [int(round(v)) for v in s]
out = {"pri_reg": r(P.reg_med), "pri_n34only": r(P.n34only_med), "pri_loo": [round(v, 2) for v in P.loo_r],
       "pri_loo_n34only": [round(v, 2) for v in P.loo_r_n34only], "fit2023": r(P.fit2023),
       "raw_nmme": r((P.scen_rawnmme_med.xs("CHIRPS") + P.scen_rawnmme_med.xs("ERA5")) / 2),
       "pos_loo": [round(v, 2) for v in Q.loo_r]}
json.dump(out, open(f"{SP}/regression_summary.json", "w")); print(out)
