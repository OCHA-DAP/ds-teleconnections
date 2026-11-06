import os, sys
# database connection settings come from the environment (ocha-stratus)
import ocha_stratus as stratus, pandas as pd
from sqlalchemy import text
out = sys.argv[1]
e = stratus.get_engine("prod")
with e.connect() as c:
    era = pd.read_sql(text("SELECT pcode, valid_date, mean FROM public.era5 WHERE iso3='HTI' ORDER BY pcode, valid_date"), c, parse_dates=["valid_date"])
    s5 = pd.read_sql(text("SELECT pcode, issued_date, valid_date, leadtime, mean FROM public.seas5 WHERE iso3='HTI' AND extract(month from issued_date) IN (11,12,1,2,3)"), c, parse_dates=["valid_date","issued_date"])
print(era.pcode.unique(), era.valid_date.max()); print(s5.pcode.unique(), s5.issued_date.max())
era.to_parquet(f"{out}/era5_hti_all.parquet"); s5.to_parquet(f"{out}/seas5_hti_all.parquet")
