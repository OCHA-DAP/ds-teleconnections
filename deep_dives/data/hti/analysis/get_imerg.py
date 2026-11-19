import os, sys
# database connection settings come from the environment (ocha-stratus)
import ocha_stratus as stratus, pandas as pd
from sqlalchemy import text
e = stratus.get_engine("prod")
with e.connect() as c:
    print(pd.read_sql(text("SELECT adm_level, count(*), min(valid_date), max(valid_date), count(distinct pcode) FROM public.imerg WHERE iso3='HTI' GROUP BY 1"), c))
    df = pd.read_sql(text("SELECT pcode, adm_level, valid_date, mean FROM public.imerg WHERE iso3='HTI' AND adm_level<=1"), c, parse_dates=["valid_date"])
df.to_parquet(f"{sys.argv[1]}/imerg_hti.parquet"); print(df.shape)
