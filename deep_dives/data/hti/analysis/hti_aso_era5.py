import os
# database connection settings come from the environment (ocha-stratus)
import ocha_stratus as stratus
import pandas as pd
from sqlalchemy import text

eng = stratus.get_engine("prod")
with eng.connect() as c:
    print(pd.read_sql(text("SELECT pcode, adm_level, name FROM public.polygon WHERE iso3='HTI' AND adm_level=0"), c))
    df = pd.read_sql(text("SELECT * FROM public.era5 WHERE pcode='HT' ORDER BY valid_date"), c, parse_dates=["valid_date"])
print(df.columns.tolist())
print(df.tail(6))
df.to_parquet(os.path.join(os.path.dirname(__file__), "hti_era5.parquet"))
