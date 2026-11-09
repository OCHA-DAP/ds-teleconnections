"""ERA5 monthly precipitation by country and department, Guatemala/Honduras/El Salvador (team database, read-only)."""
import sys, ocha_stratus as stratus, pandas as pd
from sqlalchemy import text
out = sys.argv[1]
e = stratus.get_engine("prod")
with e.connect() as c:
    era = pd.read_sql(text("SELECT iso3, pcode, adm_level, valid_date, mean FROM public.era5 WHERE iso3 IN ('GTM','HND','SLV') AND adm_level IN (0,1) ORDER BY pcode, valid_date"), c, parse_dates=["valid_date"])
print(era.groupby(["iso3", "adm_level"]).pcode.nunique(), era.valid_date.min(), era.valid_date.max())
era.to_parquet(f"{out}/era5_cadc_all.parquet")
