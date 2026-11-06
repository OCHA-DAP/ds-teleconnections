import os, sys
# database connection settings come from the environment (ocha-stratus)
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.datasources.seas5 import load_seas5
from src.datasources.era5 import load_era5
from src.skill import run_all_combinations
s = load_seas5("HT"); e = load_era5("HT")
s.to_parquet(os.path.join(os.path.dirname(__file__), "hti_seas5.parquet"))
cols = ["issued_month","trimester","n_years","pearson_r","forecast_percentile","forecast_rp","flood_rp","prob_lower_tercile","prob_rp"]
for dt in (True, False):
    sk, _ = run_all_combinations("HT","HTI","Haiti", s, e, detrend=dt)
    print(f"detrend={dt}"); print(sk[(sk.issued_month.isin([11])) & (sk.trimester.isin(["DJF","JFM","FMA","MAM","AMJ","MJJ"]))][cols].round(3).to_string(index=False))
    print(sk[(sk.issued_month.isin([1,2,3])) & (sk.trimester.isin(["MAM","AMJ"]))][["issued_month","trimester","pearson_r"]].round(2).to_string(index=False))
