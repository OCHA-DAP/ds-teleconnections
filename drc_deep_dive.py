"""DRC — El Niño, the Indian Ocean Dipole and floods (builder = "drc" in deep_dives/drc.toml).

Own builder, like Extrême-Nord's: the question is not "does El Niño change the rains" in general but how
many people the 2026/27 event could put under water, where and when. The page is plot-led and
ENCRYPTED: it sits beside a partner's internal scenario note, and that note's figures live only in
deep_dives/private/drc-partner.toml (gitignored) and in the encrypted HTML. Nothing unencrypted from this
page is committed: figures are embedded in the HTML before staticrypt runs.

    uv run python drc_deep_dive.py                 # build out/drc/ (plaintext, gitignored)
    SITE_PASSWORD=... uv run python drc_deep_dive.py --encrypt   # + docs/enso/drc/ (staticrypt)

Inputs (cached under cache/drc/; fetched when missing or stale): FloodScan population exposure
(team prod DB app.floodscan_exposure: SFED flooded fraction >= 5% x WorldPop 2020, daily, ADM1/ADM2),
EM-DAT (team blob snapshot via ocha-stratus), HNRP 2026 PiN (team dev DB hpc.pin_admin), CHIRPS v3 and
ERA5 at 0.5° (ds-precip-intercomparison cubes on the dev blob), Lake Tanganyika and Mweru levels (NASA
GSFC G-REALM 10-day altimetry), NOAA CPC ONI, NOAA PSL Niño3.4, DMI (NOAA OSMC weekly OISST from
Sep 1981, NOAA PSL HadISST before).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
SLUG = "drc"
CACHE = ROOT / "cache" / "drc"
OUT = ROOT / "out" / "drc"
FIGDIR = OUT / "fig"
DOCS = ROOT / "docs" / "enso" / "drc"
SPEC = ROOT / "deep_dives" / "drc.toml"
PRIVATE = ROOT / "deep_dives" / "private" / "drc-partner.toml"
for d in (CACHE, FIGDIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #
LANG = "en"


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


def _t(en: str, fr: str) -> str:
    return fr if LANG == "fr" else en


def _num(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",") if LANG == "fr" else s


MONTHS = {"en": "JASONDJFMAMJ", "fr": "JASONDJFMAMJ"}

# --------------------------------------------------------------------------- #
# Style (validated categorical pair: orange = El Niño + positive IOD, blue = positive IOD alone)
# --------------------------------------------------------------------------- #
BOTH_C = "#eb6834"
IOD_C = "#2a78d6"
NOW_C = "#0b0b0b"
OTHER_C = "#c9c8c2"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
# brown (drier / negative) <-> grey <-> blue (wetter / positive), as on the COD ENSO slides
DIV = ["#7a4a12", "#b77a35", "#e2c48e", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9a9993", "axes.linewidth": 0.6,
    "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

# Analogue seasons (July–June, labelled by the starting year). Fixed once, used everywhere.
BOTH = [1997, 2015, 2023]          # strong El Niño + positive IOD (SON ONI >= 1.5, SON DMI >= 0.4)
BOTH_PRE = [1972, 1982]            # same class, before FloodScan (1982 is in CHIRPS)
IOD_ONLY = [2019]                  # very strong positive IOD (SON DMI >= 1.0), weak El Niño (SON ONI < 1.0): 2019 only
NOW = 2026
FS_START = pd.Timestamp("1998-01-12")


def season_label(y: int) -> str:
    return f"{y}/{(y + 1) % 100:02d}"


def season_color(y: int) -> str:
    return BOTH_C if y in BOTH + BOTH_PRE else IOD_C if y in IOD_ONLY else NOW_C if y == NOW else OTHER_C


# --------------------------------------------------------------------------- #
# Data access (cache first; fetch when missing)
# --------------------------------------------------------------------------- #
def _stale(p: Path, days: float) -> bool:
    return not p.exists() or (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).days > days


def _db(name: str, sql: str, stage: str, max_age: float = 30) -> pd.DataFrame:
    p = CACHE / f"{name}.parquet"
    if _stale(p, max_age):
        try:
            import ocha_stratus as stratus
            from sqlalchemy import text
            for k, port in (("DEV", 15432), ("PROD", 15433)):
                os.environ.setdefault(f"DSCI_AZ_DB_{k}_HOST", f"127.0.0.1:{port}")
            with stratus.get_engine(stage=stage).connect() as c:
                pd.read_sql(text(sql), c).to_parquet(p, index=False)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({name}: DB unreachable, using cache: {str(e)[:80]})")
    return pd.read_parquet(p)


def floodscan_adm1() -> pd.DataFrame:
    df = _db("fs_adm01", "select adm_level, pcode, valid_date, sum from app.floodscan_exposure "
             "where iso3 ilike 'cod' and adm_level in ('0', '1')", "prod", max_age=3)
    df = df[df.adm_level == "1"].copy()
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["sum"] = df["sum"].astype(float)
    return df


def floodscan_adm2() -> pd.DataFrame:
    df = _db("fs_adm2", "select pcode, valid_date, sum from app.floodscan_exposure "
             "where iso3 ilike 'cod' and adm_level = '2'", "prod", max_age=3)
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["sum"] = df["sum"].astype(float)
    return df


def pin() -> pd.DataFrame:
    df = _db("hpc_pin_admin", "select * from hpc.pin_admin where iso3 = 'COD'", "dev")
    return df[df.year == df.year.max()]


def emdat() -> pd.DataFrame:
    p = CACHE / "emdat_cod_floods.parquet"
    if not p.exists():
        import ocha_stratus as stratus
        df = stratus.emdat.load_emdat_from_blob("COD", include_historic=True, stage="dev")
        df[df["Disaster Type"] == "Flood"].to_parquet(p)
    return pd.read_parquet(p)


def boundaries(level: int) -> gpd.GeoDataFrame:
    p = CACHE / f"cod_adm{level}.parquet"
    if not p.exists():
        import ocha_stratus as stratus
        cc = stratus.get_container_client(container_name="projects", stage="dev")
        raw = cc.get_blob_client("ds-floodexposure-monitoring/raw/codab/cod.shp.zip").download_blob().readall()
        tmp = CACHE / "codab"
        zipfile.ZipFile(io.BytesIO(raw)).extractall(tmp)
        for lv in (0, 1, 2):
            gpd.read_file(tmp / f"cod_adm{lv}.shp").to_parquet(CACHE / f"cod_adm{lv}.parquet")
    return gpd.read_parquet(p)


def _fetch(url: str, p: Path, days: float) -> Path:
    if _stale(p, days):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({p.name}: fetch failed, using cache: {str(e)[:80]})")
    return p


def _psl_table(text: str) -> pd.Series:
    lines = text.splitlines()
    y0, y1 = (int(v) for v in lines[0].split()[:2])
    rows = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 13 and parts[0].isdigit() and y0 <= int(parts[0]) <= y1:
            rows[int(parts[0])] = [float(v) for v in parts[1:]]
    s = pd.DataFrame(rows).T.stack()
    s.index = pd.to_datetime([f"{y}-{m + 1:02d}-01" for y, m in s.index])
    return s.where(s > -90).dropna()


def oni() -> pd.DataFrame:
    """CPC Oceanic Niño Index (official), one row per 3-month season."""
    p = _fetch("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", CACHE / "oni.txt", 7)
    df = pd.read_csv(p, sep=r"\s+")
    df.columns = ["seas", "year", "total", "anom"]
    return df


def oni_season(seas: str) -> pd.Series:
    df = oni()
    return df[df.seas == seas].set_index("year")["anom"]


def nino34() -> pd.Series:
    p = _fetch("https://psl.noaa.gov/data/correlation/nina34.anom.data", CACHE / "nina34_anom.data", 7)
    return _psl_table(p.read_text()).rename("nino34")


def dmi() -> pd.Series:
    """Monthly DMI: NOAA OSMC weekly (OISST) from Oct 1981, NOAA PSL HadISST before."""
    w = _fetch("https://stateoftheocean.osmc.noaa.gov/sur/data/dmi.nc", CACHE / "osmc_dmi.nc", 7)
    h = _fetch("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data", CACHE / "dmi_had_long.txt", 30)
    weekly = xr.open_dataset(w)["DMI"].to_series()
    m = weekly.resample("MS").mean()
    m = m[m.index >= "1981-10-01"]
    had = _psl_table(h.read_text())
    return pd.concat([had[had.index < m.index[0]], m]).rename("dmi")


def dmi_weekly() -> pd.Series:
    return xr.open_dataset(CACHE / "osmc_dmi.nc")["DMI"].to_series()


def seasonal(s: pd.Series, months: list[int]) -> pd.Series:
    """Mean over calendar months (13 = next January ...), indexed by the year of the first month."""
    out = {}
    for y in range(s.index.year.min(), s.index.year.max() + 1):
        dates = [pd.Timestamp(y + (m - 1) // 12, (m - 1) % 12 + 1, 1) for m in months]
        if all(d in s.index for d in dates):
            out[y] = s.loc[dates].mean()
    return pd.Series(out, name=s.name)


def rain(product: str = "chirps_v3") -> xr.DataArray:
    p = CACHE / f"{product}_cod05.nc"
    if not p.exists():
        import ocha_stratus as stratus
        cc = stratus.get_container_client(container_name="projects", stage="dev")
        raw = cc.get_blob_client(f"ds-precip-intercomparison/processed/grid05/{product}.nc").download_blob().readall()
        ds = xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
        lat = slice(-14.5, 6.5) if ds.lat.values[0] < ds.lat.values[-1] else slice(6.5, -14.5)
        ds.sel(lat=lat, lon=slice(11.0, 32.0)).load().to_netcdf(p)
    ds = xr.open_dataset(p)
    da = ds["precip"]
    if "valid_frac" in ds:
        da = da.where(ds["valid_frac"] > 0.5)
    return da.sortby("lat")


def rain_season(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    t = pd.DatetimeIndex(da.time.values)
    out = []
    for y in range(t.year.min(), t.year.max() + 1):
        dates = [pd.Timestamp(y + (m - 1) // 12, (m - 1) % 12 + 1, 1) for m in months]
        if all(d in t for d in dates):
            out.append(da.sel(time=dates).sum("time", skipna=False).expand_dims(year=[y]))
    return xr.concat(out, "year")


GREALM = {"tanganyika": "000315", "mweru": "000416"}


def lake(name: str) -> pd.Series:
    """G-REALM 10-day lake level, metres above mean sea level (EGM2008 column)."""
    p = CACHE / f"{name}_grealm.parquet"
    if _stale(p, 20):
        try:
            tgz = CACHE / "lake.10.tar.gz"
            _fetch("https://earth.gsfc.nasa.gov/gwm/zip/lake.10.tar.gz", tgz, 20)
            with tarfile.open(tgz) as tf:
                for nm, lid in GREALM.items():
                    rows = []
                    for ln in tf.extractfile(f"lake{lid}.10d.2.txt").read().decode().splitlines():
                        f = ln.split()
                        if len(f) >= 16 and f[2].isdigit() and len(f[2]) == 8 and float(f[14]) < 9000:
                            rows.append((pd.Timestamp(f[2]), float(f[14])))
                    pd.Series(dict(rows)).sort_index().to_frame("h").to_parquet(CACHE / f"{nm}_grealm.parquet")
            tgz.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({name}: G-REALM unreachable, using cache: {str(e)[:80]})")
    return pd.read_parquet(p)["h"]


def fao_maize() -> pd.DataFrame:
    """FAOSTAT maize production (t), selected southern African countries, from the public bulk file."""
    p = CACHE / "fao_maize_production.csv"
    if not p.exists():
        z = _fetch("https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_Africa.zip", CACHE / "fao_africa.zip", 90)
        df = pd.read_csv(zipfile.ZipFile(z).open("Production_Crops_Livestock_E_Africa_NOFLAG.csv"), encoding="latin-1")
        m = df[(df["Item"] == "Maize (corn)") & (df["Element"] == "Production")]
        ycols = [c for c in m.columns if c.startswith("Y") and c[1:].isdigit()]
        long = m.melt(id_vars=["Area"], value_vars=ycols, var_name="year", value_name="t")
        long["year"] = long.year.str[1:].astype(int)
        long.to_csv(p, index=False)
    return pd.read_csv(p)


def wfp_prices() -> pd.DataFrame:
    """WFP retail prices for DRC (HDX), monthly; runs of >= 6 identical local prices are dropped (carried forward)."""
    p = _fetch("https://data.humdata.org/dataset/07a6d6fb-828d-4391-afb8-78b1fc32caa3/resource/c9906fd4-9947-450a-b816-682907463f2a/"
               "download/wfp_food_prices_cod.csv", CACHE / "wfp_food_prices_cod.csv", 30)
    df = pd.read_csv(p, skiprows=[1])
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df["usdprice"] = df["usdprice"].astype(float)
    return df


def lubumbashi_maize() -> dict[str, pd.Series]:
    df = wfp_prices()
    out = {}
    for c in ("Maize", "Maize flour"):
        s = df[(df.market == "Lubumbashi") & (df.commodity == c) & (df.pricetype == "Retail")].sort_values("date").set_index("date")
        run = (s["price"] != s["price"].shift()).cumsum()
        stale = s.groupby(run)["price"].transform("size") >= 6
        out[c] = s.loc[~stale, "usdprice"]
    return out


# --------------------------------------------------------------------------- #
# Regions (FloodScan sees river and lake floods; it does not see flash floods on the Kivu slopes)
# --------------------------------------------------------------------------- #
REGIONS = {
    "corridor": dict(en="Congo and Ubangi rivers", fr="Fleuve Congo et Oubangui",
                     pcodes=["CD41", "CD42", "CD43", "CD44", "CD45", "CD51", "CD33", "CD52"], window=(10, 2)),
    "kinshasa": dict(en="Kinshasa", fr="Kinshasa", pcodes=["CD10"], window=(11, 2)),
    "tanganyika": dict(en="Tanganyika", fr="Tanganyika", pcodes=["CD74"], window=(2, 6)),
    "southeast": dict(en="South-east wetlands", fr="Zones humides du Sud-Est",
                      pcodes=["CD73", "CD71", "CD72"], window=(2, 5)),
}
REGION_SUB = {
    "corridor": ("Équateur, Ubangi, Mongala, Tshuapa, Tshopo, Maï-Ndombe, Bas-Uele",
                 "Équateur, Ubangi, Mongala, Tshuapa, Tshopo, Maï-Ndombe, Bas-Uele"),
    "kinshasa": ("river floods only; city flash floods are not seen", "crues du fleuve seulement ; crues éclair urbaines non vues"),
    "tanganyika": ("lake shore and the Lukuga", "rives du lac et Lukuga"),
    "southeast": ("Haut-Lomami, Haut-Katanga, Lualaba", "Haut-Lomami, Haut-Katanga, Lualaba"),
}
SMOOTH_DAYS = 15  # centred rolling median: removes the short rain-onset bursts over Katanga towns


def _in_window(month: pd.Series, window: tuple[int, int]) -> pd.Series:
    a, b = window
    return month.between(a, b) if a <= b else (month >= a) | (month <= b)


def smooth(s: pd.Series) -> pd.Series:
    return s.asfreq("D").rolling(SMOOTH_DAYS, center=True, min_periods=SMOOTH_DAYS // 2).median()


def by_season(s: pd.Series) -> pd.DataFrame:
    df = s.to_frame("v")
    df["season"] = df.index.year - (df.index.month < 7)
    start = pd.to_datetime(df["season"].astype(str) + "-07-01").values
    df["day"] = (df.index.values - start).astype("timedelta64[D]").astype(int)
    return df


def region_series(fs: pd.DataFrame) -> dict[str, pd.Series]:
    return {k: smooth(fs[fs.pcode.isin(r["pcodes"])].groupby("valid_date")["sum"].sum()) for k, r in REGIONS.items()}


def region_peaks(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Peak smoothed exposure per region and season, inside the region's flood window."""
    rows = []
    for k, s in series.items():
        df = by_season(s.dropna())
        df = df[_in_window(df.index.month.to_series(index=df.index), REGIONS[k]["window"])]
        for season, g in df.groupby("season"):
            rows.append(dict(region=k, season=season, peak=g["v"].max(), date=g["v"].idxmax(), n=len(g)))
    pk = pd.DataFrame(rows)
    last_full = pd.Timestamp.now().year - 1 - (pd.Timestamp.now().month < 7)
    return pk[pk.season <= last_full]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def partial_r(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Partial correlation of y (year, cell) with x (year), controlling for z (year)."""
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    rx, ry = resid(x, z), resid(y, z)
    rx, ry = rx - rx.mean(), ry - ry.mean(0)
    return (rx @ ry) / np.sqrt((rx**2).sum() * (ry**2).sum(0))


DRY_SEASON_MM = 150
RAIN_SEASONS = {"SON": [9, 10, 11], "OND": [10, 11, 12], "NDJ": [11, 12, 13], "DJF": [12, 13, 14], "JFM": [13, 14, 15]}
INDEX_MONTHS = {"SON": [9, 10, 11], "OND": [9, 10, 11], "NDJ": [10, 11, 12], "DJF": [11, 12, 13], "JFM": [11, 12, 13]}


def driver_maps(product: str = "chirps_v3", y0: int = 1981, y1: int = 2025) -> dict:
    """Partial r maps: rain vs Niño3.4 holding the DMI fixed, and vs the DMI holding Niño3.4 fixed."""
    da, n34, d = rain(product), nino34(), dmi()
    out = {}
    for sname, months in RAIN_SEASONS.items():
        rs = rain_season(da, months).sel(year=slice(y0, y1))
        yrs = rs.year.values
        e = seasonal(n34, INDEX_MONTHS[sname]).reindex(yrs).values
        i = seasonal(d, INDEX_MONTHS[sname]).reindex(yrs).values
        ok = ~np.isnan(e) & ~np.isnan(i)
        Y = rs.values[ok].reshape(ok.sum(), -1)
        good = ~np.isnan(Y).any(0)
        for key, x, z in (("enso", e[ok], i[ok]), ("iod", i[ok], e[ok])):
            r = np.full(Y.shape[1], np.nan)
            r[good] = partial_r(Y[:, good], x, z)
            out[(key, sname)] = xr.DataArray(r.reshape(rs.shape[1:]), coords=dict(lat=rs.lat, lon=rs.lon))
        out["n"] = int(ok.sum())
    return out


def rain_anomalies(product: str = "chirps_v3") -> dict:
    da = rain(product)
    out = {}
    for sname in ("OND", "JFM"):
        rs = rain_season(da, RAIN_SEASONS[sname])
        clim = rs.sel(year=slice(1991, 2020)).mean("year")
        for y in BOTH_PRE[1:] + BOTH + IOD_ONLY:
            if y in rs.year:
                # cells whose normal 3-month total is under 150 mm are in their dry season: % anomalies mean little there
                out[(sname, y)] = (rs.sel(year=y) / clim * 100).where(clim >= DRY_SEASON_MM, -1)
    return out


EMDAT_REGION = {  # DisNo. -> region row, read from EM-DAT's Location field
    "1990-0114-COD": "kinshasa", "1994-0546-COD": "east", "1997-0295-COD": "corridor", "1999-0531-COD": "kinshasa",
    "2001-0209-COD": "kinshasa", "2001-0683-COD": "corridor", "2002-0080-COD": "east", "2003-0566-COD": "corridor",
    "2003-0641-COD": "west", "2006-0145-COD": "west", "2006-0172-COD": "kinshasa", "2006-0673-COD": "corridor",
    "2007-0527-COD": "kinshasa", "2008-0139-COD": "west", "2009-0544-COD": "west", "2010-0133-COD": "southeast",
    "2010-0357-COD": "corridor", "2011-0516-COD": "east", "2011-0517-COD": "west", "2012-0046-COD": "tanganyika",
    "2014-0432-COD": "east", "2015-0536-COD": "kinshasa", "2015-0549-COD": "corridor", "2015-0535-COD": "east",
    "2016-0451-COD": "corridor", "2016-0507-COD": "west", "2017-0391-COD": "east", "2018-0005-COD": "kinshasa",
    "2019-0561-COD": "several", "2020-0123-COD": "east", "2020-0128-COD": "southeast", "2020-0175-COD": "east",
    "2020-0152-COD": "east", "2020-0419-COD": "east", "2020-0531-COD": "west", "2022-0232-COD": "east",
    "2022-0746-COD": "east", "2022-0770-COD": "east", "2022-0805-COD": "kinshasa", "2023-0281-COD": "east",
    "2023-0769-COD": "east", "2023-0857-COD": "east", "2024-0268-COD": "several", "2024-0389-COD": "tanganyika",
    "2025-0245-COD": "kinshasa", "2025-0273-COD": "tanganyika", "2025-0347-COD": "east", "2025-0437-COD": "kinshasa",
}
EMDAT_ROWS = {
    "several": ("Several regions", "Plusieurs régions"), "kinshasa": ("Kinshasa", "Kinshasa"),
    "corridor": ("Congo and Ubangi rivers", "Fleuve Congo et Oubangui"),
    "east": ("East (Kivu, Ituri, Maniema)", "Est (Kivu, Ituri, Maniema)"), "tanganyika": ("Tanganyika", "Tanganyika"),
    "southeast": ("South-east", "Sud-Est"), "west": ("Kasaï and west", "Kasaï et ouest"),
}


def emdat_events() -> pd.DataFrame:
    df = emdat().copy()
    df["region"] = df["DisNo."].map(EMDAT_REGION)
    missing = df[df.region.isna()]["DisNo."].tolist()
    if missing:
        raise ValueError(f"EM-DAT events without a region in EMDAT_REGION: {missing}")
    df["date"] = pd.to_datetime(dict(year=df["Start Year"], month=df["Start Month"].fillna(6).astype(int), day=15))
    df["season"] = df.date.dt.year - (df.date.dt.month < 7)
    df["affected"] = df["Total Affected"].fillna(0)
    df["deaths"] = df["Total Deaths"].fillna(0)
    return df


HNRP_ADM2_FIX = {"CD3106": "CD3107"}  # Kasongo-Lunda: HNRP 2026 code vs the CODAB used by FloodScan


def territory_cross(fs2: pd.DataFrame) -> gpd.GeoDataFrame:
    """Per territory: smoothed seasonal peak in the El Niño + IOD seasons vs a typical season; 2026 PiN."""
    fs2 = fs2[fs2.valid_date >= FS_START].copy()
    piv = fs2.pivot_table(index="valid_date", columns="pcode", values="sum").asfreq("D")
    sm = piv.rolling(SMOOTH_DAYS, center=True, min_periods=SMOOTH_DAYS // 2).median()
    season = sm.index.year - (sm.index.month < 7)
    pk = sm.groupby(season).max()
    full = [y for y in pk.index if 1998 <= y <= 2025]
    t = pd.DataFrame({"analog": pk.loc[[y for y in BOTH if y >= 1998]].mean(), "typical": pk.loc[full].median()})
    p = pin()
    by2 = p.groupby(p["admin2_code"].replace(HNRP_ADM2_FIX))["final_pin"].sum()
    missing = sorted(set(by2.index) - set(t.index))
    if missing:
        raise ValueError(f"HNRP admin2 codes not in the FloodScan territories: {missing} (extend HNRP_ADM2_FIX)")
    t["pin"] = by2.reindex(t.index).fillna(0)
    g = boundaries(2).set_index("ADM2_PCODE").join(t, how="left")
    return g


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> Path:
    p = FIGDIR / f"{name}_{LANG}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _source(fig, text: str, y: float = -0.02):
    fig.text(0.01, y, text, fontsize=7, color=TEXT2, ha="left", va="top")


def _place(ax, items: list[tuple[float, float, str]], dx: float = 0.16, dy: float = 0.065, fontsize: float = 7.5) -> None:
    """Label points (x, y, text) just above-right of each, pushing a label down when it would sit on one
    already placed (axes-fraction units). Call after the axis limits are final."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    placed: list[tuple[float, float]] = []
    for x, y, text in sorted(items, key=lambda t: -t[1]):
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        ly = fy + 0.015
        while any(abs(fx - px) < dx and abs(ly - py) < dy for px, py in placed):
            ly -= dy
        placed.append((fx, ly))
        moved = abs(ly - fy - 0.015) > 1e-6
        ax.annotate(text, (x, y), xytext=(fx + 0.012, ly), textcoords="axes fraction", fontsize=fontsize, color="#0b0b0b",
                    va="bottom", arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.5, shrinkA=0, shrinkB=2) if moved else None)


def fig_setup() -> Path:
    """Where 2026 sits: SON ONI vs SON DMI, 1950–2025, with 2026 so far."""
    son = oni_season("SON")
    d = seasonal(dmi(), [9, 10, 11])
    df = pd.DataFrame({"oni": son, "dmi": d}).dropna().loc[1950:2025]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _grid(ax, "both")
    ax.axhline(0, color="#9a9993", lw=0.6)
    ax.axvline(0, color="#9a9993", lw=0.6)
    offsets = {1972: (-26, 5), 1982: (5, -11), 1994: (5, -11), 2006: (-28, 4)}
    for y, r in df.iterrows():
        c = season_color(y)
        ax.scatter(r.oni, r.dmi, s=34 if c != OTHER_C else 16, color=c, edgecolor=SURFACE, lw=1, zorder=3 if c != OTHER_C else 2)
        if y in BOTH + BOTH_PRE + IOD_ONLY + [1994, 2006]:
            ax.annotate(str(y), (r.oni, r.dmi), xytext=offsets.get(y, (5, 3)), textcoords="offset points", fontsize=8,
                        color="#0b0b0b" if c != OTHER_C else TEXT2)
    # 2026: latest ONI season (JAS) and September DMI
    o = oni()
    latest = o.iloc[-1]
    dm = dmi()
    d26 = dm[dm.index >= "2026-09-01"].iloc[0]
    ax.scatter(latest.anom, d26, s=90, marker="*", color=NOW_C, zorder=5)
    ax.annotate(_t(f"2026 so far\n(ONI {latest.seas} {latest.anom:+.1f}, DMI Sep {d26:+.1f})",
                   f"2026 à ce jour\n(ONI {latest.seas} {_num(latest.anom, 1)}, DMI sept. {_num(d26, 1)})"),
                (latest.anom, d26), xytext=(-4, -62), textcoords="offset points", fontsize=8, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.6, shrinkB=6))
    ax.set_xlabel(_t("El Niño strength: ONI, Sep–Nov (°C)", "Force d'El Niño : ONI, sept.–nov. (°C)"))
    ax.set_ylabel(_t("Indian Ocean Dipole: DMI, Sep–Nov (°C)", "Dipôle de l'océan Indien : DMI, sept.–nov. (°C)"))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BOTH_C, label=_t("strong El Niño + positive IOD", "El Niño fort + IOD positif")),
        Line2D([], [], marker="o", ls="", color=IOD_C, label=_t("very strong positive IOD, weak El Niño", "IOD positif très fort, El Niño faible")),
        Line2D([], [], marker="*", ls="", color=NOW_C, ms=10, label="2026"),
    ], loc="upper left", fontsize=8)
    _source(fig, _t("Sources: NOAA CPC ONI; DMI from NOAA OSMC (OISST) since Oct 1981, NOAA PSL (HadISST) before.",
                    "Sources : NOAA CPC (ONI) ; DMI : NOAA OSMC (OISST) depuis oct. 1981, NOAA PSL (HadISST) avant."))
    return _save(fig, "setup")


def _map_axes(ax, adm1):
    adm1.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
    ax.set_xlim(12, 31.5)
    ax.set_ylim(-13.6, 5.6)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _clip(da: xr.DataArray, adm0) -> np.ndarray:
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    lat, lon = da.lat.values, da.lon.values
    res = float(abs(lon[1] - lon[0]))
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res, res)
    m = geometry_mask(adm0.geometry, out_shape=(lat.size, lon.size), transform=tr, invert=True, all_touched=True)
    return np.where(m[::-1] if lat[0] < lat[-1] else m, da.values, np.nan)


def fig_drivers(maps: dict, maps_era5: dict) -> Path:
    adm0, adm1 = boundaries(0), boundaries(1)
    bounds = [-1, -0.5, -0.3, 0.3, 0.5, 1]
    cmap = mcolors.ListedColormap([DIV[0], DIV[1], "#ebeae6", DIV[5], DIV[6]])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    seasons = ["SON", "OND", "NDJ", "JFM"]
    rows = [("iod", maps, _t("Positive Dipole\n(CHIRPS)", "Dipôle positif\n(CHIRPS)")),
            ("iod", maps_era5, _t("Positive Dipole\n(ERA5)", "Dipôle positif\n(ERA5)")),
            ("enso", maps, _t("El Niño\n(CHIRPS)", "El Niño\n(CHIRPS)"))]
    fig, axes = plt.subplots(len(rows), len(seasons), figsize=(10.4, 8.4))
    for i, (key, m, lab) in enumerate(rows):
        for j, s in enumerate(seasons):
            ax = axes[i, j]
            da = m[(key, s)]
            ax.pcolormesh(da.lon, da.lat, _clip(da, adm0), cmap=cmap, norm=norm, shading="nearest")
            _map_axes(ax, adm1)
            adm0.boundary.plot(ax=ax, color="#52514e", lw=0.6)
            if i == 0:
                ax.set_title(_t({"SON": "Sep–Nov", "OND": "Oct–Dec", "NDJ": "Nov–Jan", "JFM": "Jan–Mar"}[s],
                                {"SON": "sept.–nov.", "OND": "oct.–déc.", "NDJ": "nov.–janv.", "JFM": "janv.–mars"}[s]), fontsize=10)
        axes[i, 0].text(-0.1, 0.5, lab, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="center",
                        fontsize=9.5, fontweight="bold")
    fig.legend(handles=[Patch(color=DIV[6], label=_t("much wetter (r ≥ 0.5)", "beaucoup plus humide (r ≥ 0,5)")),
                        Patch(color=DIV[5], label=_t("wetter (0.3–0.5)", "plus humide (0,3–0,5)")),
                        Patch(color="#ebeae6", label=_t("no clear signal", "pas de signal net")),
                        Patch(color=DIV[1], label=_t("drier (−0.3 to −0.5)", "plus sec (−0,3 à −0,5)")),
                        Patch(color=DIV[0], label=_t("much drier (r ≤ −0.5)", "beaucoup plus sec (r ≤ −0,5)"))],
               loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.02, hspace=0.08, bottom=0.06)
    _source(fig, _t(f"Partial correlation of 3-month rainfall with the Dipole index (El Niño held fixed) or with Niño3.4 (Dipole held fixed); "
                    f"0.5° grid, 1981–2025 (n = {maps['n']}); |r| ≥ 0.3 ≈ p < 0.05. El Niño in ERA5: no clear signal in at least 96% of cells in every season shown.",
                    f"Corrélation partielle de la pluie sur 3 mois avec l'indice du dipôle (El Niño fixé) ou avec Niño3.4 (dipôle fixé) ; grille 0,5°, "
                    f"1981–2025 (n = {maps['n']}) ; |r| ≥ 0,3 ≈ p < 0,05. El Niño dans ERA5 : pas de signal net dans au moins 96 % des cellules à chaque saison."),
            y=-0.02)
    return _save(fig, "drivers")


def fig_rain_years(anoms: dict) -> Path:
    adm0, adm1 = boundaries(0), boundaries(1)
    years = [y for y in BOTH_PRE[1:] + BOTH + IOD_ONLY]
    bounds = [-2, 0, 60, 80, 90, 110, 125, 150, 1000]
    cmap = mcolors.ListedColormap(["#d6d5cf"] + DIV)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(2, len(years), figsize=(2.3 * len(years), 5.4))
    for i, s in enumerate(("OND", "JFM")):
        for j, y in enumerate(years):
            ax = axes[i, j]
            da = anoms.get((s, y))
            if da is not None:
                ax.pcolormesh(da.lon, da.lat, _clip(da, adm0), cmap=cmap, norm=norm, shading="nearest")
            _map_axes(ax, adm1)
            adm0.boundary.plot(ax=ax, color="#52514e", lw=0.6)
            if i == 0:
                ax.set_title(season_label(y), color=season_color(y), fontsize=11)
        axes[i, 0].text(-0.1, 0.5, _t({"OND": "Oct–Dec", "JFM": "Jan–Mar"}[s], {"OND": "oct.–déc.", "JFM": "janv.–mars"}[s]),
                        transform=axes[i, 0].transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    labels = ["< 60%", "60–80%", "80–90%", "90–110%", "110–125%", "125–150%", "> 150%"]
    handles = [Patch(color=c, label=lab) for c, lab in zip(DIV, labels)]
    handles.append(Patch(color="#d6d5cf", label=_t(f"dry season (normal < {DRY_SEASON_MM} mm)", f"saison sèche (normale < {DRY_SEASON_MM} mm)")))
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), title=_t("rainfall, % of the 1991–2020 average", "pluie, % de la moyenne 1991–2020"),
               title_fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.06, bottom=0.12)
    _source(fig, _t("CHIRPS v3, 0.5°. Orange titles: strong El Niño + positive IOD; blue: very strong positive IOD with a weak El Niño.",
                    "CHIRPS v3, 0,5°. Titres orange : El Niño fort + IOD positif ; bleu : IOD positif très fort, El Niño faible."), y=-0.05)
    return _save(fig, "rain_years")


def _season_lines(ax, s: pd.Series) -> list[tuple[float, float, str]]:
    """Draw one line per season; return peak labels for the highlighted seasons (placed later)."""
    df = by_season(s)
    labels = []
    for season, g in df.groupby("season"):
        c = season_color(season)
        hl = c != OTHER_C
        ax.plot(g["day"], g["v"] / 1000, color=c, lw=1.8 if hl else 0.6, zorder=3 if hl else 1,
                ls=(0, (4, 1.5)) if season == 2015 else (0, (1, 1.2)) if season == 1997 else "-")
        if hl and season != NOW and g["v"].max() >= 1000:
            k = g["v"].idxmax()
            lab = season_label(season) + (_t(" (from Jan)", " (dès janv.)") if season == 1997 else "")
            labels.append((g.loc[k, "day"], g.loc[k, "v"] / 1000, lab))
    ticks = [0, 31, 62, 92, 123, 153, 184, 215, 243, 274, 304, 335]
    ax.set_xticks(ticks)
    ax.set_xticklabels(list(MONTHS[LANG]))
    ax.set_xlim(0, 365)
    _grid(ax)
    return labels


def _titled(ax, title: str, sub: str) -> None:
    ax.set_title(title, loc="left", pad=15)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.5, color=TEXT2, va="bottom")


def fig_flood_cycle(series: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    for ax, (k, r) in zip(axes.flat, REGIONS.items()):
        labels = _season_lines(ax, series[k])
        _titled(ax, _t(r["en"], r["fr"]), _t(*REGION_SUB[k]))
        top = max(v for _, v, _ in labels) if labels else ax.get_ylim()[1]
        ax.set_ylim(0, max(ax.get_ylim()[1], top * 1.12))
        _place(ax, labels)
    for ax in axes[:, 0]:
        ax.set_ylabel(_t("people in flooded areas (thousands)", "personnes en zone inondée (milliers)"))
    fig.legend(handles=[
        Line2D([], [], color=BOTH_C, lw=1.8, label=_t("strong El Niño + positive IOD", "El Niño fort + IOD positif")),
        Line2D([], [], color=IOD_C, lw=1.8, label=_t("very strong positive IOD, weak El Niño (2019/20)", "IOD positif très fort, El Niño faible (2019/20)")),
        Line2D([], [], color=NOW_C, lw=1.8, label=_t("2026/27 so far", "2026/27 à ce jour")),
        Line2D([], [], color=OTHER_C, lw=1, label=_t("other seasons since 1998", "autres saisons depuis 1998"))],
        loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    _source(fig, _t(f"FloodScan flooded fraction (≥ 5%) × WorldPop 2020 population, summed by region; {SMOOTH_DAYS}-day running median. Season = July–June. FloodScan starts 12 Jan 1998.",
                    f"Fraction inondée FloodScan (≥ 5 %) × population WorldPop 2020, somme par région ; médiane glissante sur {SMOOTH_DAYS} jours. Saison = juillet–juin. FloodScan commence le 12 janv. 1998."),
            y=-0.035)
    return _save(fig, "flood_cycle")


PEAK_REGIONS = ["corridor", "tanganyika", "southeast"]  # Kinshasa: one river flood in the record (2023/24), no range to show


def fig_flood_peaks(pk: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(len(PEAK_REGIONS), 1, figsize=(8.6, 5.2))
    for ax, k in zip(axes, PEAK_REGIONS):
        r = REGIONS[k]
        d = pk[pk.region == k]
        med = d[d.season >= 1998].peak.median()
        ax.axvline(med / 1000, color=TEXT2, lw=0.8)
        ax.text(med / 1000, 0.97, _t(f" median {_num(med / 1000)}k", f" médiane {_num(med / 1000)} k"), fontsize=7.5,
                color=TEXT2, transform=ax.get_xaxis_transform(), ha="left", va="top")
        rng = np.random.default_rng(1)
        hl_rows = []
        for _, row in d.iterrows():
            c = season_color(row.season)
            hl = c != OTHER_C
            yj = 0 if hl else rng.uniform(-0.25, 0.25)
            ax.scatter(row.peak / 1000, yj, s=60 if hl else 22, color=c, edgecolor=SURFACE, lw=1, zorder=3 if hl else 2)
            if hl:
                hl_rows.append(row)
        span = d.peak.max() / 1000
        last_x, up = -1e9, True
        for row in sorted(hl_rows, key=lambda r_: r_.peak):
            x = row.peak / 1000
            up = (not up) if x - last_x < 0.09 * span else True
            last_x = x
            ax.annotate(season_label(row.season) + ("*" if row.season == 1997 else ""), (x, 0),
                        xytext=(0, 8 if up else -15), textcoords="offset points", ha="center", fontsize=7.5)
        ax.set_yticks([])
        ax.set_ylim(-0.7, 0.8)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(left=0)
        _grid(ax, "x")
        ax.set_title(_t(r["en"], r["fr"]), loc="left", fontsize=9.5)
    axes[-1].set_xlabel(_t("highest number of people in flooded areas in the season (thousands)",
                           "nombre maximal de personnes en zone inondée sur la saison (milliers)"))
    fig.tight_layout(h_pad=1.2)
    _source(fig, _t("One dot per season, 1997/98–2025/26, peak of the 15-day running median inside each region's flood months "
                    "(rivers Oct–Feb, Tanganyika Feb–Jun, south-east Feb–May). *1997/98 from 12 January only. Kinshasa is left out: "
                    "one river flood in the record (2023/24). A past range, not a forecast.",
                    "Un point par saison, 1997/98–2025/26 : pic de la médiane glissante sur 15 jours pendant les mois de crue de chaque "
                    "région (fleuves oct.–févr., Tanganyika févr.–juin, Sud-Est févr.–mai). *1997/98 à partir du 12 janvier seulement. "
                    "Kinshasa n'y figure pas : une seule crue du fleuve dans l'archive (2023/24). Une fourchette passée, pas une prévision."),
            y=-0.01)
    return _save(fig, "flood_peaks")


def lake_rises(s: pd.Series) -> pd.DataFrame:
    rows = []
    for y in range(s.index.year.min(), s.index.year.max() + 1):
        o = s[f"{y}-10-01":f"{y}-10-31"]
        nxt = s[f"{y}-10-01":f"{y + 1}-07-31"]
        if len(o) and len(nxt) and nxt.index.max() >= pd.Timestamp(f"{y + 1}-06-01"):
            rows.append(dict(season=y, oct=o.mean(), peak=nxt.max(), peak_date=nxt.idxmax()))
    df = pd.DataFrame(rows).set_index("season")
    df["rise"] = df.peak - df.oct
    return df


def fig_lake() -> tuple[Path, dict]:
    t, m = lake("tanganyika"), lake("mweru")
    rz = lake_rises(t)
    oct26 = t["2026-10-01":].mean()
    rec = rz.peak.max()
    rec_season = int(rz.peak.idxmax())
    scen = [(season_label(y), oct26 + rz.loc[y, "rise"], season_color(y)) for y in [1997, 2023, 2019, 2015] if y in rz.index]
    scen.append((_t("typical season (median)", "saison typique (médiane)"), oct26 + rz.rise.median(), TEXT2))
    fig = plt.figure(figsize=(10.4, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1], height_ratios=[1.15, 1], hspace=0.42, wspace=0.42)
    ax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for a, s, name in ((ax, t, "Tanganyika"), (ax2, m, "Mweru")):
        a.plot(s.index, s.values, color="#3f3e3a", lw=0.8)
        for y in BOTH + IOD_ONLY:
            a.axvspan(pd.Timestamp(f"{y}-10-01"), pd.Timestamp(f"{y + 1}-06-30"), color=season_color(y), alpha=0.15, lw=0)
        _grid(a)
        a.set_title(_t(f"Lake {name}", f"Lac {name}"), loc="left")
        a.set_xlim(pd.Timestamp("1992-06-01"), pd.Timestamp("2027-03-01"))
    ax.axhline(rec, color=TEXT2, lw=0.7)
    ax.text(pd.Timestamp("1993-01-01"), rec + 0.05, _t(f"{season_label(rec_season)} flood peak, {rec:.2f} m",
                                                       f"pic de crue {season_label(rec_season)}, {_num(rec, 2)} m"),
            fontsize=7.5, color=TEXT2)
    ax.annotate(_t(f"Oct 2026: {oct26:.2f} m", f"oct. 2026 : {_num(oct26, 2)} m"), (t.index.max(), t.iloc[-1]),
                xytext=(-6, -22), textcoords="offset points", ha="right", fontsize=7.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.5))
    ax.set_ylabel(_t("water level (m above sea level)", "niveau (m au-dessus de la mer)"))
    ax2.set_ylabel("")
    # every season's October-to-peak rise
    _grid(ax3, "x")
    rng = np.random.default_rng(2)
    for y, r in rz.iterrows():
        c = season_color(y)
        hl = c != OTHER_C
        ax3.scatter(r.rise, 0 if hl else rng.uniform(-0.25, 0.25), s=60 if hl else 22, color=c, edgecolor=SURFACE, lw=1,
                    zorder=3 if hl else 2)
    last_x, up = -9, True
    for y in sorted([y for y in BOTH + IOD_ONLY if y in rz.index], key=lambda y: rz.loc[y, "rise"]):
        x = rz.loc[y, "rise"]
        up = (not up) if x - last_x < 0.15 else True
        last_x = x
        ax3.annotate(season_label(y), (x, 0), xytext=(0, 8 if up else -15), textcoords="offset points", ha="center", fontsize=7.5)
    ax3.axvline(rz.rise.median(), color=TEXT2, lw=0.8)
    ax3.text(rz.rise.median(), 0.97, _t(f" median +{rz.rise.median():.1f} m", f" médiane +{_num(rz.rise.median(), 1)} m"),
             transform=ax3.get_xaxis_transform(), ha="left", va="top", fontsize=7.5, color=TEXT2)
    ax3.set_yticks([])
    ax3.set_ylim(-0.7, 0.8)
    ax3.spines["left"].set_visible(False)
    ax3.set_xlabel(_t("rise from October to the next peak (m)", "hausse d'octobre au pic suivant (m)"))
    ax3.set_title(_t("How much the lake rose, season by season", "Hausse du lac, saison par saison"), loc="left")
    # where those rises would take the lake from October 2026
    _grid(ax4, "x")
    for i, (lab, v, c) in enumerate(scen):
        ax4.plot([oct26, v], [i, i], color=c, lw=1.6)
        ax4.scatter([v], [i], color=c, s=40, zorder=3, edgecolor=SURFACE, lw=1)
    ax4.axvline(rec, color=TEXT2, lw=0.8)
    ax4.text(rec, 0.99, _t(f" {season_label(rec_season)}\n peak", f" pic\n {season_label(rec_season)}"),
             transform=ax4.get_xaxis_transform(), ha="left", va="top", fontsize=7.5, color=TEXT2)
    ax4.set_yticks(range(len(scen)))
    ax4.set_yticklabels([_t(f"rise as in {l}", f"hausse comme en {l}") if "/" in l else l for l, _, _ in scen], fontsize=7.5)
    ax4.set_ylim(len(scen) - 0.5, -0.5)
    ax4.set_xlim(oct26 - 0.15, max(v for _, v, _ in scen) + 0.15)
    ax4.set_xlabel(_t("peak level in 2027 (m)", "niveau du pic en 2027 (m)"))
    ax4.set_title(_t("From the October 2026 level", "À partir du niveau d'octobre 2026"), loc="left")
    _source(fig, _t(f"NASA GSFC G-REALM satellite altimetry, 10-day, EGM2008 datum (not the national gauge datum); latest {t.index.max():%d %b %Y}. "
                    "Shading: orange El Niño + positive IOD, blue very strong positive IOD with a weak El Niño. Past rises have not depended on the October level "
                    f"(r = {rz.rise.corr(rz.oct):.2f}). Bottom right: a past range, not a forecast.",
                    f"Altimétrie satellite NASA GSFC G-REALM, pas de 10 jours, référence EGM2008 (pas celle des échelles nationales) ; "
                    f"dernière mesure {t.index.max():%d/%m/%Y}. Fond : orange El Niño + IOD positif, bleu IOD positif très fort avec El Niño faible. "
                    f"Les hausses passées ne dépendent pas du niveau d'octobre (r = {_num(rz.rise.corr(rz.oct), 2)}). En bas à droite : une fourchette passée, pas une prévision."), y=0.02)
    return _save(fig, "lake"), dict(oct26=oct26, rises=rz, scen=scen, record=rec, record_season=rec_season)


def fig_reported(ev: pd.DataFrame) -> Path:
    ev = ev[ev.date >= "1997-07-01"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), gridspec_kw=dict(height_ratios=[1, 1.5]), sharex=True)
    tot = ev.groupby("season")["affected"].sum().reindex(range(1997, ev.season.max() + 1), fill_value=0)
    x = [pd.Timestamp(f"{y + 1}-01-01") for y in tot.index]
    ax1.bar(x, tot.values / 1e6, width=300, color=[season_color(y) for y in tot.index], edgecolor=SURFACE, lw=1)
    for y, v in tot.items():
        if y in BOTH + IOD_ONLY or v > 0.3e6:
            ax1.annotate(_t(f"{v / 1e6:.2f} M", f"{_num(v / 1e6, 2)} M"), (pd.Timestamp(f"{y + 1}-01-01"), v / 1e6),
                         xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)
    ax1.set_ylabel(_t("people reported affected\nper season (millions)", "personnes affectées\nsignalées par saison (millions)"))
    _grid(ax1)
    rows = list(EMDAT_ROWS)
    for i, k in enumerate(rows):
        e = ev[ev.region == k]
        ax2.scatter(e.date, [i] * len(e), s=np.clip(e.affected / 4000, 6, 700), color=[season_color(s) for s in e.season],
                    alpha=0.85, edgecolor="#3f3e3a", lw=0.4)
        big = e[(e.deaths >= 150)]
        for _, r in big.iterrows():
            ax2.annotate(_t(f"{int(r.deaths):,} dead", f"{_num(r.deaths)} morts"), (r.date, i), xytext=(0, -11),
                         textcoords="offset points", ha="center", fontsize=6.5, color=TEXT2)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([_t(*EMDAT_ROWS[k]) for k in rows])
    ax2.set_ylim(len(rows) - 0.4, -0.6)
    for y in BOTH + IOD_ONLY:
        for a in (ax1, ax2):
            a.axvspan(pd.Timestamp(f"{y}-07-01"), pd.Timestamp(f"{y + 1}-06-30"), color=season_color(y), alpha=0.08, lw=0)
    _grid(ax2, "x")
    for s_ in (50_000, 500_000, 2_000_000):
        ax2.scatter([], [], s=np.clip(s_ / 4000, 6, 700), color=OTHER_C, edgecolor="#3f3e3a", lw=0.4,
                    label=_t(f"{s_:,} affected", f"{_num(s_)} affectés"))
    ax2.legend(loc="lower left", fontsize=7.5, ncol=3, bbox_to_anchor=(0, -0.28), labelspacing=1.2, borderpad=0.8)
    fig.tight_layout()
    _source(fig, _t(f"EM-DAT (CRED/UCLouvain), flood records for DRC, team snapshot to {ev['Last Update'].max()}. Reported counts are a floor: "
                    "small events and earlier years are under-recorded. Seasons July–June; shading: orange El Niño + positive IOD, blue very strong positive IOD with a weak El Niño.",
                    f"EM-DAT (CRED/UCLouvain), inondations en RDC, copie de l'équipe à jour au {ev['Last Update'].max()}. Les chiffres signalés sont "
                    "un minimum : petits événements et années anciennes sous-enregistrés. Saisons juillet–juin ; fond : orange El Niño + IOD positif, bleu IOD positif très fort avec El Niño faible."),
            y=-0.06)
    return _save(fig, "reported")


# Kivu territories with deadly flash floods in EM-DAT (>= 30 deaths in one event), and the events placed in each
# (read from EM-DAT's Location field: Bushushu/Nyamukubi/Bunyakiri are in Kalehe, Bihambwe/Rubaya/Sake in Masisi,
# Baraka/Kasaba in Fizi; events naming Bukavu and Uvira together are counted under Uvira).
EAST_FLASH_EVENTS = {
    "CD6205": ["2014-0432-COD", "2023-0281-COD", "2022-0746-COD"],                   # Kalehe
    "CD6201": ["1994-0546-COD", "2015-0535-COD", "2022-0232-COD", "2023-0857-COD"],  # Bukavu
    "CD6208": ["2002-0080-COD", "2020-0175-COD"],                                    # Uvira
    "CD6103": ["2017-0391-COD", "2022-0770-COD", "2020-0419-COD"],                   # Masisi
    "CD6210": ["2023-0769-COD", "2025-0347-COD"],                                    # Fizi
}
EAST_FLASH = list(EAST_FLASH_EVENTS)


def fig_needs(g: gpd.GeoDataFrame, ev: pd.DataFrame, top_n: int = 12) -> Path:
    adm1 = boundaries(1)
    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_axes((0.0, 0.08, 0.40, 0.80))
    axe = fig.add_axes((0.585, 0.12, 0.25, 0.76))
    axp = fig.add_axes((0.85, 0.12, 0.14, 0.76))
    bins = [0, 1000, 5000, 20000, 50000, 1e9]
    blues = ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]
    g = g.copy()
    g["cls"] = pd.cut(g["analog"].fillna(0), bins, labels=False, include_lowest=True)
    g.plot(ax=ax, color=[blues[int(c)] for c in g["cls"]], edgecolor="#ffffff", lw=0.3)
    adm1.boundary.plot(ax=ax, color="#52514e", lw=0.5)
    flash = g.loc[g.index.isin(EAST_FLASH)].representative_point()
    ax.scatter(flash.x, flash.y, marker="^", s=30, color="#3f3e3a", zorder=4)
    ax.set_axis_off()
    ax.set_title(_t("People in flooded areas, El Niño + IOD seasons", "Personnes en zone inondée, saisons El Niño + IOD"),
                 loc="left", fontsize=9.5)
    h = [Patch(color=c, label=lab) for c, lab in zip(blues, ["< 1k", "1–5k", "5–20k", "20–50k", "> 50k"])]
    h.append(Line2D([], [], marker="^", ls="", color="#3f3e3a", label=_t("deadly flash floods (EM-DAT)", "crues éclair meurtrières (EM-DAT)")))
    ax.legend(handles=h, loc="lower left", fontsize=7.5, title=_t("people, territory peak", "personnes, pic du territoire"), title_fontsize=7.5)
    # ranked territories + the flash-flood group
    d = g.dropna(subset=["analog"]).sort_values("analog", ascending=False).head(top_n)
    deaths = {pc: int(ev[ev["DisNo."].isin(ids)]["deaths"].sum()) for pc, ids in EAST_FLASH_EVENTS.items()}
    f = g.loc[EAST_FLASH].assign(deaths=pd.Series(deaths)).sort_values("deaths", ascending=False)
    names = [f"{r.ADM2_FR} ({r.ADM1_FR})" for _, r in d.iterrows()] + [f"{r.ADM2_FR} ({r.ADM1_FR})" for _, r in f.iterrows()]
    y = list(range(len(d))) + [len(d) + 1 + i for i in range(len(f))]
    yd, yf = y[: len(d)], y[len(d):]
    _grid(axe, "x")
    axe.hlines(yd, d["typical"] / 1000, d["analog"] / 1000, color=OTHER_C, lw=1.2)
    axe.scatter(d["typical"] / 1000, yd, color=OTHER_C, s=28, zorder=3, edgecolor=SURFACE, lw=1)
    axe.scatter(d["analog"] / 1000, yd, color=BOTH_C, s=40, zorder=3, edgecolor=SURFACE, lw=1)
    for yy, (_, r) in zip(yf, f.iterrows()):
        axe.text(1, yy, _t(f"not seen by satellite · {r.deaths:,} dead since 1994", f"invisible au satellite · {_num(r.deaths)} morts depuis 1994"),
                 fontsize=7, color=TEXT2, va="center")
    axe.set_yticks(y)
    axe.set_yticklabels(names, fontsize=7.5)
    axe.set_ylim(max(y) + 0.6, -0.6)
    axe.set_xlim(left=0)
    axe.set_xlabel(_t("people in flooded areas (thousands)", "personnes en zone inondée (milliers)"))
    axe.legend(handles=[Line2D([], [], marker="o", ls="", color=BOTH_C, label=_t("El Niño + IOD seasons", "saisons El Niño + IOD")),
                        Line2D([], [], marker="o", ls="", color=OTHER_C, label=_t("typical season", "saison typique"))],
               loc="center right", fontsize=7.5, bbox_to_anchor=(1.0, 0.42))
    axe.set_title(_t("Most exposed territories", "Territoires les plus exposés"), loc="left", fontsize=9.5)
    _grid(axp, "x")
    pins = list(d["pin"]) + list(f["pin"])
    axp.barh(y, np.array(pins) / 1e6, color="#8a8984", height=0.6)
    axp.set_yticks(y)
    axp.set_yticklabels([])
    axp.set_ylim(max(y) + 0.6, -0.6)
    axp.set_xlabel(_t("people in need, 2026 (millions)", "personnes dans le besoin, 2026 (millions)"))
    axp.set_title(_t("Needs 2026", "Besoins 2026"), loc="left", fontsize=9.5)
    _source(fig, _t("FloodScan flooded fraction × WorldPop 2020: per territory, the seasonal peak of the 15-day running median; El Niño + IOD = mean of 2015/16 "
                    "and 2023/24, typical = median of 1998/99–2025/26. PiN: HNRP 2026 (OCHA), health zones summed to territories. Deaths: EM-DAT floods since 1990.",
                    "Fraction inondée FloodScan × WorldPop 2020 : par territoire, pic saisonnier de la médiane glissante sur 15 jours ; El Niño + IOD = moyenne "
                    "de 2015/16 et 2023/24, typique = médiane de 1998/99–2025/26. PiN : HNRP 2026 (OCHA), zones de santé agrégées par territoire. "
                    "Décès : EM-DAT, inondations depuis 1990."), y=0.02)
    return _save(fig, "needs")


def southeast_rain(pcodes: tuple[str, ...] = ("CD71",)) -> pd.Series:
    """CHIRPS v3 January–March rain over Haut-Katanga, % of the 1991–2020 mean, by season (year of December)."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    rs = rain_season(rain("chirps_v3"), RAIN_SEASONS["JFM"])
    lat, lon = rs.lat.values, rs.lon.values
    res = float(lon[1] - lon[0])
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res, res)
    adm1 = boundaries(1)
    m = geometry_mask(adm1[adm1.ADM1_PCODE.isin(pcodes)].geometry, out_shape=(lat.size, lon.size), transform=tr,
                      invert=True, all_touched=False)[::-1]
    ser = pd.Series(np.nanmean(np.where(m, rs.values, np.nan), axis=(1, 2)), index=rs.year.values)
    return ser / ser.loc[1991:2020].mean() * 100


def strong_nino() -> list[int]:
    """Strong El Niño seasons: Sep–Nov ONI >= 1.5 °C (the same El Niño test as the analogue seasons)."""
    o = oni_season("SON")
    return [int(y) for y in o[o >= 1.5].index]


def fig_southeast() -> tuple[Path, dict]:
    sn = strong_nino()
    jfm = southeast_rain()
    fao = fao_maize()
    zmb = fao[fao.Area == "Zambia"].set_index("year")["t"].loc[1990:] / 1e6
    px = lubumbashi_maize()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1.25], wspace=0.28))
    # (a) Jan–Mar rain in Haut-Katanga
    dry3 = jfm.quantile(1 / 3)
    _grid(a1)
    a1.bar([y + 1 for y in jfm.index], jfm.values - 100, bottom=100, width=0.8,
           color=[BOTH_C if y in sn else OTHER_C for y in jfm.index])
    a1.axhline(100, color=TEXT2, lw=0.6)
    a1.axhline(dry3, color=TEXT2, lw=0.6, ls=(0, (3, 2)))
    a1.text(1983, dry3 - 1, _t("driest third", "tiers le plus sec"), fontsize=7, color=TEXT2, va="top")
    for y in sn:
        if y in jfm.index:
            a1.annotate(str(y + 1), (y + 1, jfm[y]), xytext=(0, 3 if jfm[y] >= 100 else -9), textcoords="offset points",
                        ha="center", fontsize=7)
    a1.set_ylabel(_t("% of the 1991–2020 average", "% de la moyenne 1991–2020"))
    a1.set_title(_t("Jan–Mar rain, Haut-Katanga", "Pluie janv.–mars, Haut-Katanga"), loc="left")
    # (b) Zambia maize
    _grid(a2)
    a2.bar(zmb.index, zmb.values, width=0.8, color=[BOTH_C if y - 1 in sn else OTHER_C for y in zmb.index])
    for y in zmb.index:
        if y - 1 in sn:
            a2.annotate(str(y), (y, zmb[y]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    a2.set_ylabel(_t("million tonnes", "millions de tonnes"))
    a2.set_title(_t("Zambia's maize harvest", "Récolte de maïs en Zambie"), loc="left")
    # (c) Lubumbashi prices
    _grid(a3)
    for y in sn:
        if y >= 2010:  # the harvest year after the event and the lean season that follows it
            a3.axvspan(pd.Timestamp(f"{y + 1}-01-01"), pd.Timestamp(f"{y + 2}-06-30"), color=BOTH_C, alpha=0.12, lw=0)
    for c, col, lab in (("Maize flour", "#3f3e3a", _t("maize flour", "farine de maïs")), ("Maize", "#8a8984", _t("maize grain", "maïs grain"))):
        s_ = px[c][px[c].index >= "2012-01-01"]
        a3.plot(s_.index, s_.values, color=col, lw=1.2, marker=".", ms=3, label=lab)
    a3.set_ylabel(_t("US$ per kg (retail)", "US$ par kg (détail)"))
    a3.set_title(_t("Maize prices, Lubumbashi", "Prix du maïs, Lubumbashi"), loc="left")
    a3.legend(loc="upper left", fontsize=7.5)
    a3.set_ylim(bottom=0)
    fig.legend(handles=[Patch(color=BOTH_C, label=_t("after a strong El Niño (Sep–Nov ONI ≥ 1.5 °C)", "après un El Niño fort (ONI sept.–nov. ≥ 1,5 °C)")),
                        Patch(color=OTHER_C, label=_t("other years", "autres années"))], loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    _source(fig, _t("CHIRPS v3 0.5° (Haut-Katanga mean); FAOSTAT; WFP retail prices (HDX), US$, months where the local price repeats for six "
                    "months or more are left out (not updated). Bars and shading are coloured for the harvest or lean season after the event.",
                    "CHIRPS v3 0,5° (moyenne Haut-Katanga) ; FAOSTAT ; prix de détail PAM (HDX), en US$, les mois où le prix local se répète six "
                    "mois ou plus sont exclus (non mis à jour). Barres et fond colorés pour la récolte ou la soudure qui suit l'événement."), y=-0.1)
    n = [y for y in sn if y in jfm.index]
    dry = [y for y in n if jfm[y] <= dry3]
    stats = dict(se_n=len(n), se_dry=len(dry), se_dry_years=", ".join(season_label(y) for y in dry),
                 se_wet_years=", ".join(season_label(y) for y in n if y not in dry),
                 zmb_drop=_num((1 - zmb[2024] / zmb[2023]) * 100), zmb_drop98=_num((1 - zmb[1998] / zmb[1997]) * 100))
    return _save(fig, "southeast"), stats


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
EXTRA_CSS = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0 8px}
.tile{border:1px solid #e2e7e7;border-top:3px solid #eb6834;border-radius:6px;padding:12px 14px;background:#fff}
.tile .big{font-size:24px;font-weight:700;margin:0 0 4px;line-height:1.15}
.tile p{font-size:13px;color:var(--n8);margin:0}
section h2{font-size:18px;line-height:1.35}
.note{font-size:13px;color:var(--n7);margin:-6px 0 14px}
.lang{float:right;font-size:13px;margin-top:4px}
details{margin:24px 0}details summary{cursor:pointer;font-weight:600}
details li{font-size:13.5px;margin:0 0 8px;max-width:90ch}
.private{background:#fbf3ea;border-left:5px solid #eb6834;padding:10px 14px;border-radius:4px}
.private table{width:100%}.private td{font-size:13px}
.tablewrap{overflow-x:auto}
"""


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def values(a: dict) -> dict:
    """Numbers the TOML headlines quote; printed by report()."""
    lk, pk, ev = a["lake"], a["peaks"], a["events"]
    rz = lk["rises"]
    strong = [y for y in (1997, 2019, 2023) if y in rz.index]
    t = lake("tanganyika")
    octs = t[t.index.month == 10].groupby(t[t.index.month == 10].index.year).mean()
    rank = int(octs.rank(ascending=False)[NOW])
    ordinal_en = {1: "highest", 2: "second-highest", 3: "third-highest", 4: "fourth-highest"}.get(rank, f"{rank}th-highest")
    ordinal_fr = {1: "le plus haut", 2: "le 2e plus haut", 3: "le 3e plus haut"}.get(rank, f"le {rank}e plus haut")
    aff = ev.groupby("season")["affected"].sum()
    aff_both = [aff.get(y, 0) for y in BOTH if y >= 2015]
    corr = pk[pk.region == "corridor"].set_index("season")["peak"].rank(ascending=False)
    top = sorted(int(corr[y]) for y in BOTH if y in corr.index)
    nat = smooth(floodscan_adm1().groupby("valid_date")["sum"].sum())
    fs_end = nat.dropna().index.max()
    return dict(
        data_end=f"{fs_end:%d %b %Y}" if LANG == "en" else f"{fs_end:%d/%m/%Y}",
        corridor_top="Top 3" if top == [1, 2, 3] else "#" + ", ".join(map(str, top)),
        lake_oct=_num(lk["oct26"], 1), lake_oct_rank_en=ordinal_en, lake_oct_rank_fr=ordinal_fr, lake_n=len(octs),
        lake_gap=_num(lk["record"] - lk["oct26"], 1),
        rise_lo=_num(min(rz.loc[strong, "rise"]), 1), rise_hi=_num(max(rz.loc[strong, "rise"]), 1),
        aff_lo=_num(min(aff_both) / 1e6, 1), aff_hi=_num(max(aff_both) / 1e6, 1),
        exposed_2324=_num(nat["2023-07-01":"2024-06-30"].max() / 1e6, 2), affected_2324=_num(aff.get(2023, 0) / 1e6, 2),
        emdat_end=str(ev["Last Update"].max()),
        **a.get("se", {}),
    )


def range_table(pk: pd.DataFrame, series: dict) -> str:
    cols = [(y, season_label(y) + ("*" if y == 1997 else "")) for y in (2015, 2023, 1997, 2019)]
    head = "".join(f'<th class="num" style="color:{season_color(y)}">{lab}</th>' for y, lab in cols)
    rows = []
    for k in ["corridor", "kinshasa", "tanganyika", "southeast"]:
        d = pk[pk.region == k].set_index("season")["peak"]
        cells = "".join(f'<td class="num">{_num(d.get(y, np.nan) / 1000)}</td>' for y, _ in cols)
        rows.append(f'<tr><td>{_t(REGIONS[k]["en"], REGIONS[k]["fr"])}</td><td class="num">{_num(d.loc[1998:].median() / 1000)}</td>{cells}</tr>')
    return ('<div class="tablewrap"><table><tr><th>' + _t("People in flooded areas, season peak (thousands)", "Personnes en zone inondée, pic de la saison (milliers)")
            + f'</th><th class="num">{_t("typical season", "saison typique")}</th>{head}</tr>' + "".join(rows) + "</table></div>"
            + f'<p class="note">{_t("*1997/98 from 12 January 1998. Typical = median of 1998/99–2025/26.", "*1997/98 à partir du 12 janvier 1998. Typique = médiane de 1998/99–2025/26.")}</p>')


def private_block(v: dict) -> str:
    if not PRIVATE.exists():
        return ""
    pv = tomllib.loads(PRIVATE.read_text(encoding="utf-8"))
    L = LANG
    rows = "".join(f"<tr><td><b>{r['channel_' + L]}</b></td><td>{r['partner_' + L].format(**v)}</td><td>{r['ours_' + L].format(**v)}</td></tr>"
                   for r in pv["rows"])
    return (f'<section id="partner" class="private"><h2>{pv["title_" + L]}</h2><p class="note">{pv["note_" + L]}</p>'
            f'<div class="tablewrap"><table><tr><th></th><th>{pv["col_partner_" + L]}</th><th>{pv["col_ours_" + L]}</th></tr>{rows}</table></div></section>')


def render(spec: dict, a: dict, figs: dict) -> str:
    import enso_deep_dive as edd
    v = values(a)
    L = LANG
    pg = spec["page"]
    home = "../" if L == "en" else "../../"
    out = [edd.HEAD.format(title=html.escape(pg["title_" + L]), desc=html.escape(pg["sub_" + L]), css=edd.CSS + EXTRA_CSS,
                           home=home, home_label=_t("ENSO deep dives", "Analyses ENSO par pays"))]
    out[0] = out[0].replace('<html lang="en">', f'<html lang="{L}">')
    out.append(f'<a class="lang" href="{"fr/" if L == "en" else "../"}">{"Français" if L == "en" else "English"}</a>')
    out.append(f'<p class="eyebrow">{_t("Restricted · El Niño 2026/27", "Accès restreint · El Niño 2026/27")}</p>')
    out.append(f'<h1>{pg["title_" + L]}</h1><p class="meta">{pg["sub_" + L]}<br>{pg["dateline_" + L].format(**v)}</p>')
    out.append('<div class="tiles">' + "".join(f'<div class="tile"><p class="big">{t["big"].format(**v)}</p><p>{t[L].format(**v)}</p></div>'
                                                for t in spec["tiles"]) + "</div>")
    for sec in spec["sections"]:
        out.append(f'<section id="{sec["id"]}"><h2>{sec[L].format(**v)}</h2>')
        if sec.get("note_" + L):
            out.append(f'<p class="note">{sec["note_" + L].format(**v)}</p>')
        out.append(f'<figure><img src="{_img(figs[sec["fig"]])}" alt="{html.escape(sec[L].format(**v))}"></figure>')
        if sec.get("table") == "range":
            out.append(range_table(a["peaks"], a["series"]))
        out.append("</section>")
    out.append(private_block(v))
    notes = "".join(f"<li>{n.format(**v)}</li>" for n in spec["notes"][L])
    out.append(f'<details><summary>{_t("About the data", "À propos des données")}</summary><ul>{notes}</ul></details>')
    out.append(f'<p class="small">{_t("OCHA Centre for Humanitarian Data · built", "OCHA Centre de données humanitaires · généré le")} '
               f'{pd.Timestamp.now():%Y-%m-%d}</p>')
    out.append(edd.FOOT)
    return "\n".join(out)


def encrypt() -> None:
    """staticrypt both pages into docs/enso/drc/ and check no plaintext survives (password from SITE_PASSWORD)."""
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        raise SystemExit("set SITE_PASSWORD to encrypt")
    for rel in ("index.html", "fr/index.html"):
        src, dst = OUT / rel, (DOCS / rel).parent
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["npx", "-y", "staticrypt@3.5.4", str(src), "-d", str(dst), "--short", "--remember", "30",
                        "--template-title", "DRC floods and El Niño: restricted",
                        "--template-instructions", "OCHA Centre for Humanitarian Data. Password shared internally."],
                       check=True, env=dict(os.environ, STATICRYPT_PASSWORD=pw), stdout=subprocess.DEVNULL)
        enc = (DOCS / rel).read_text()
        if "staticrypt" not in enc:
            raise SystemExit(f"{DOCS / rel} is not a staticrypt page")
        for marker in ('class="tile"', "data:image/png", "EM-DAT", "FloodScan", "Tanganyika", "WFP", "PAM", "METTELSAT", "IPC 3+"):
            if marker in enc:
                raise SystemExit(f"{DOCS / rel} contains plaintext ({marker})")
    print(f"  encrypted -> {DOCS}/index.html, fr/index.html")
    write_index()


def write_index() -> None:
    """Re-render the /enso/ card index from every deep_dives/*.toml (the DRC card says 'restricted')."""
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def analyse() -> dict:
    fs = floodscan_adm1()
    series = region_series(fs)
    a = dict(series=series, peaks=region_peaks(series), maps=driver_maps(), maps_era5=driver_maps("era5"), anoms=rain_anomalies(),
             events=emdat_events(), cross=territory_cross(floodscan_adm2()))
    return a


def figures(a: dict) -> dict:
    f = {}
    f["setup"] = fig_setup()
    f["drivers"] = fig_drivers(a["maps"], a["maps_era5"])
    f["rain_years"] = fig_rain_years(a["anoms"])
    f["flood_cycle"] = fig_flood_cycle(a["series"])
    f["flood_peaks"] = fig_flood_peaks(a["peaks"])
    f["lake"], a["lake"] = fig_lake()
    f["reported"] = fig_reported(a["events"])
    f["needs"] = fig_needs(a["cross"], a["events"])
    f["southeast"], a["se"] = fig_southeast()
    return f


def report(a: dict) -> None:
    pk = a["peaks"]
    for k in REGIONS:
        d = pk[pk.region == k].set_index("season")["peak"]
        print(f"  {k}: median {d.median():,.0f}; rank of analogues:",
              {season_label(y): (round(d[y]), int(d.rank(ascending=False)[y])) for y in BOTH + IOD_ONLY if y in d.index}, f"of {len(d)}")
    lk = a.get("lake")
    if lk:
        print(f"  lake Oct 2026 {lk['oct26']:.2f}; record {lk['record']:.2f} ({lk['record_season']}); scen", {k: round(v, 2) for k, v, _ in lk["scen"]})
    ev = a["events"]
    print("  EM-DAT affected by season:", ev.groupby("season")["affected"].sum().loc[lambda s: s > 100000].round(-3).to_dict())
    if lk:
        print("  headline values:", values(a))


def build(spec: dict | None = None, *_args, encrypt_pages: bool = False) -> dict:
    """Called by enso_deep_dive.main() for builder = "drc" (grid, gdf, indices unused), or standalone."""
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    a = analyse()
    for lang, rel in (("en", "index.html"), ("fr", "fr/index.html")):
        set_lang(lang)
        figs = figures(a)
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_text(render(spec, a, figs), encoding="utf-8")
        if lang == "en":
            report(a)
    set_lang("en")
    print(f"  wrote {OUT}/index.html and fr/index.html (plaintext, gitignored)")
    if encrypt_pages:
        encrypt()
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encrypt", action="store_true", help="also staticrypt into docs/enso/drc/ (needs SITE_PASSWORD)")
    args = ap.parse_args()
    build(encrypt_pages=args.encrypt)


if __name__ == "__main__":
    main()
