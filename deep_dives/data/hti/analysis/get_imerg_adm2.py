import os, sys
# database connection settings come from the environment (ocha-stratus)
import ocha_stratus as stratus, pandas as pd
from sqlalchemy import text
e = stratus.get_engine("prod")
with e.connect() as c:
    df = pd.read_sql(text("SELECT pcode, valid_date, mean FROM public.imerg WHERE iso3='HTI' AND adm_level=2 AND extract(month from valid_date) BETWEEN 2 AND 7"), c, parse_dates=["valid_date"])
df.to_parquet(f"{sys.argv[1]}/imerg_hti_adm2_fj.parquet"); print(df.shape, df.pcode.nunique(), df.valid_date.min(), df.valid_date.max())
