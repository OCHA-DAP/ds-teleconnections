"""Venezuela — El Niño, La Niña, floods and drought (builder = "ven" in deep_dives/ven.toml).

Own builder, plot-led like the DRC page: one short headline per figure, numbers filled from the build.
The question is what the 2026/27 El Niño means for Venezuela, state by state: rain, floods, drought,
the Guri reservoir, vegetation and harvests, set against every El Niño and La Niña since 1981.

    uv run python ven_deep_dive.py              # build out/ven/ (gitignored) to review
    uv run python ven_deep_dive.py --publish    # + docs/enso/ven/ and the /enso/ card index

Inputs (cached under cache/ven/; fetched when missing or stale): CHIRPS v3 and ERA5 at 0.5° (ds-precip-
intercomparison cubes on the dev blob); ERA5 and SEAS5 admin-1 raster stats (team prod DB, through the laptop
tunnel); state boundaries (COD-AB via FieldMaps); NOAA CPC ONI; EM-DAT (team blob snapshot); DesInventar
Venezuela (UNDRR, records to early 2016); Guri reservoir level (NASA GSFC G-REALM altimetry); JRC ASAP
vegetation (FPAR) by GAUL admin-1; FAOSTAT crop production (Americas bulk file).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import tarfile
import textwrap
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
SLUG = "ven"
CACHE = ROOT / "cache" / "ven"
OUT = ROOT / "out" / "ven"
FIGDIR = OUT / "fig"
DOCS = ROOT / "docs" / "enso" / SLUG
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
for d in (CACHE, FIGDIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Style: brown = El Niño / drier, blue = La Niña / wetter, as on the other ENSO pages
# --------------------------------------------------------------------------- #
NINO_C = "#b5541b"
STRONG_C = "#7a2d0b"
NINA_C = "#2a78d6"
NEUTRAL_C = "#c9c8c2"
NOW_C = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
DIV = ["#7a4a12", "#b77a35", "#e2c48e", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9a9993", "axes.linewidth": 0.6,
    "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

NOW = 2026
Y0, Y1 = 1981, 2025   # ENSO years (July–June) with full rain records in both products
BOX = dict(lat=(0.0, 13.0), lon=(-74.0, -59.0))


def year_label(y: int) -> str:
    return f"{y}/{(y + 1) % 100:02d}"


# --------------------------------------------------------------------------- #
# Data access (cache first; fetch when missing or stale)
# --------------------------------------------------------------------------- #
def _stale(p: Path, days: float) -> bool:
    return not p.exists() or (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).days > days


def _fetch(url: str, p: Path, days: float) -> Path:
    if _stale(p, days):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({p.name}: fetch failed, using cache: {str(e)[:80]})")
    return p


def _db(name: str, sql: str, stage: str, max_age: float = 30) -> pd.DataFrame:
    """Team database through the laptop tunnel (the shell's FQDN host is not reachable from outside the VNet)."""
    p = CACHE / f"{name}.parquet"
    if _stale(p, max_age):
        try:
            import ocha_stratus as stratus
            from sqlalchemy import text
            os.environ["DSCI_AZ_DB_DEV_HOST"] = "127.0.0.1:15432"
            os.environ["DSCI_AZ_DB_PROD_HOST"] = "127.0.0.1:15433"
            with stratus.get_engine(stage=stage).connect() as c:
                pd.read_sql(text(sql), c).to_parquet(p, index=False)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({name}: DB unreachable, using cache: {str(e)[:80]})")
    return pd.read_parquet(p)


def states() -> gpd.GeoDataFrame:
    """The 24 states (COD-AB via FieldMaps); pcode VE01…VE24. Federal Dependencies are not in the boundaries."""
    p = CACHE / "ven_adm1.parquet"
    if not p.exists():
        import ocha_stratus as stratus
        g = stratus.codab.load_codab_from_fieldmaps("VEN", 1)
        g = g[g.adm1_name.notna()].rename(columns={"adm1_src": "pcode", "adm1_name": "name"})[["pcode", "name", "geometry"]]
        g.to_parquet(p)
    return gpd.read_parquet(p).sort_values("pcode").reset_index(drop=True)


def era5_adm1() -> pd.DataFrame:
    """ERA5 monthly mean precipitation (mm/day) per state, team raster stats (prod)."""
    df = _db("era5_adm1", "select pcode, valid_date, mean from public.era5 where iso3 = 'VEN' and adm_level = 1", "prod", 20)
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["mm"] = df["mean"] * df.valid_date.dt.days_in_month
    return df


def seas5_adm1() -> pd.DataFrame:
    """SEAS5 monthly ensemble-mean precipitation (mm/day) per state by issuance and lead, team raster stats (prod)."""
    now = pd.Timestamp.now()
    expected = (now if now.day >= 6 else now - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    p = CACHE / "seas5_adm1.parquet"
    stale = not p.exists() or pd.Timestamp(pd.read_parquet(p, columns=["issued_date"]).issued_date.max()) < expected
    df = _db("seas5_adm1", "select pcode, issued_date, valid_date, leadtime, mean from public.seas5 "
             "where iso3 = 'VEN' and adm_level = 1", "prod", 0 if stale else 3650)
    for c in ("issued_date", "valid_date"):
        df[c] = pd.to_datetime(df[c])
    return df


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


def rain(product: str = "chirps_v3") -> xr.DataArray:
    """Monthly totals (mm) at 0.5° over Venezuela, from the ds-precip-intercomparison cubes (dev blob)."""
    p = CACHE / f"{product}_ven05.nc"
    if not p.exists():
        import ocha_stratus as stratus
        cc = stratus.get_container_client(container_name="projects", stage="dev")
        raw = cc.get_blob_client(f"ds-precip-intercomparison/processed/grid05/{product}.nc").download_blob().readall()
        ds = xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
        lat = slice(*BOX["lat"]) if ds.lat.values[0] < ds.lat.values[-1] else slice(*BOX["lat"][::-1])
        ds.sel(lat=lat, lon=slice(*BOX["lon"])).load().to_netcdf(p)
    ds = xr.open_dataset(p)
    da = ds["precip"]
    if "valid_frac" in ds:
        da = da.where(ds["valid_frac"] > 0.5)
    return da.sortby("lat")


def emdat() -> pd.DataFrame:
    p = CACHE / "emdat_ven.parquet"
    if _stale(p, 60):
        import ocha_stratus as stratus
        stratus.emdat.load_emdat_from_blob("VEN", include_historic=True, stage="dev").to_parquet(p)
    return pd.read_parquet(p)


DI_URL = "https://www.desinventar.net/DesInventar/download/DI_export_ven.zip"
DI_FIELDS = ["serial", "level0", "level1", "name0", "name1", "evento", "lugar", "fechano", "fechames", "fechadia",
             "muertos", "desaparece", "afectados", "damnificados", "evacuados", "vivdest", "vivafec", "nhectareas", "fuentes"]


def desinventar() -> pd.DataFrame:
    """DesInventar Venezuela disaster records (UNDRR; one row per event × place). The national database stops early in 2016."""
    p = CACHE / "desinventar_ven.parquet"
    if not p.exists():
        import xml.etree.ElementTree as ET
        z = _fetch(DI_URL, CACHE / "DI_export_ven.zip", 3650)
        rows = []
        with zipfile.ZipFile(z).open("DI_export_ven.xml") as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag == "TR" and el.find("fechano") is not None:
                    rows.append({k: (el.findtext(k) or "").strip() for k in DI_FIELDS})
                    el.clear()
        df = pd.DataFrame(rows)
        for c in DI_FIELDS[7:-1]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(p, index=False)
    df = pd.read_parquet(p)
    return df[df.fechano.between(1900, 2016)]


GURI_ID = "000432"


def guri() -> pd.Series:
    """Guri reservoir level, 10-day satellite altimetry (NASA GSFC G-REALM, metres above EGM2008)."""
    p = CACHE / "guri_grealm.parquet"
    if _stale(p, 20):
        try:
            tgz = _fetch("https://earth.gsfc.nasa.gov/gwm/zip/lake.10.tar.gz", CACHE / "lake.10.tar.gz", 20)
            rows = []
            with tarfile.open(tgz) as tf:
                for ln in tf.extractfile(f"lake{GURI_ID}.10d.2.txt").read().decode().splitlines():
                    f = ln.split()
                    if len(f) >= 16 and f[2].isdigit() and len(f[2]) == 8 and float(f[14]) < 9000:
                        rows.append((pd.Timestamp(f[2]), float(f[14])))
            pd.Series(dict(rows)).sort_index().to_frame("h").to_parquet(p)
            tgz.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  (G-REALM unreachable, using cache: {str(e)[:80]})")
    return pd.read_parquet(p)["h"]


ASAP_EXPORT = "https://agricultural-production-hotspots.ec.europa.eu/export/rum/export.php"
ASAP_VEN = 202   # ASAP's own country id (asap0_id), not an ISO code


def asap(variable_id: int, class_id: int, classesset_id: int, sensor_id: int = 3) -> pd.DataFrame:
    """JRC ASAP dekadal indicator means per GAUL admin-1 unit (ds-asap-trends pattern; open data, 2001-05 on)."""
    key = f"asap_{variable_id}_{class_id}_{classesset_id}_{sensor_id}"
    p = CACHE / f"{key}.csv"
    if _stale(p, 20):
        r = requests.get(ASAP_EXPORT, params=dict(gaul_level=1, country_id=ASAP_VEN, variable_id=variable_id, class_id=class_id,
                                                  classesset_id=classesset_id, sensor_id=sensor_id), timeout=300)
        r.raise_for_status()
        if not r.text.lstrip().startswith("country_id,"):
            if not p.exists():
                raise ValueError(f"unexpected ASAP response: {r.text[:200]!r}")
        else:
            p.write_text(r.text)
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.loc[df["value"] <= -999, "value"] = np.nan
    return df


def faostat() -> pd.DataFrame:
    """FAOSTAT crop production, area and yield for Venezuela, with flags (Americas bulk file)."""
    p = CACHE / "faostat_ven.parquet"
    if not p.exists():
        z = _fetch("https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_Americas.zip",
                   CACHE / "fao_americas.zip", 3650)
        zf = zipfile.ZipFile(z)
        name = next(n for n in zf.namelist() if n.endswith(".csv") and "NOFLAG" not in n and "Americas" in n)
        df = pd.read_csv(zf.open(name), encoding="latin-1")
        df = df[df["Area"].str.startswith("Venezuela")]
        ycols = [c for c in df.columns if c.startswith("Y") and c[1:].isdigit()]
        fcols = [c + "F" for c in ycols if c + "F" in df.columns]
        v = df.melt(id_vars=["Item", "Element", "Unit"], value_vars=ycols, var_name="year", value_name="value")
        f = df.melt(id_vars=["Item", "Element"], value_vars=fcols, var_name="year", value_name="flag")
        v["year"] = v.year.str[1:].astype(int)
        f["year"] = f.year.str[1:-1].astype(int)
        v.merge(f, on=["Item", "Element", "year"], how="left").to_parquet(p, index=False)
        (CACHE / "fao_americas.zip").unlink(missing_ok=True)
    return pd.read_parquet(p)




# --------------------------------------------------------------------------- #
# Rain years and ENSO phases. A rain year runs from May of year Y to April of Y+1 (the May–November rains,
# then the December–April dry season) and takes the phase of the Sep–Nov ONI of year Y, as on the DRC page.
# --------------------------------------------------------------------------- #
def phases() -> pd.DataFrame:
    df = oni_season("SON").to_frame("oni")
    df["phase"] = np.where(df.oni >= 0.5, "nino", np.where(df.oni <= -0.5, "nina", "neutral"))
    df["strong"] = df.oni >= 1.5
    return df


def phase_color(y: int, ph: pd.DataFrame) -> str:
    if y == NOW:
        return NOW_C
    if y not in ph.index:
        return NEUTRAL_C
    r = ph.loc[y]
    return STRONG_C if r.strong else NINO_C if r.phase == "nino" else NINA_C if r.phase == "nina" else NEUTRAL_C


def phase_years(ph: pd.DataFrame, phase: str, y0: int = Y0, y1: int = Y1) -> list[int]:
    return [int(y) for y in ph.index[ph.phase == phase] if y0 <= y <= y1]


REGIONS = {  # name: (pcodes, states)
    "North-west": (["VE23", "VE11", "VE13"], "Zulia, Falcón, Lara"),
    "Andes": (["VE20", "VE14", "VE21"], "Táchira, Mérida, Trujillo"),
    "Western Llanos": (["VE04", "VE06", "VE18", "VE09"], "Apure, Barinas, Portuguesa, Cojedes"),
    "Centre-north": (["VE01", "VE24", "VE15", "VE05", "VE08", "VE22"],
                     "Distrito Capital, La Guaira, Miranda, Aragua, Carabobo, Yaracuy"),
    "Central and eastern Llanos": (["VE12", "VE03", "VE16"], "Guárico, Anzoátegui, Monagas"),
    "North-east coast": (["VE19", "VE17"], "Sucre, Nueva Esparta"),
    "Guayana and Delta": (["VE07", "VE02", "VE10"], "Bolívar, Amazonas, Delta Amacuro"),
}
PCODE_REGION = {p: r for r, (ps, _) in REGIONS.items() for p in ps}
STATE_ORDER = [p for ps, _ in REGIONS.values() for p in ps]


# --------------------------------------------------------------------------- #
# Rain by state and season
# --------------------------------------------------------------------------- #
SEASONS = ["MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM"]   # from May of the rain year
SEASON_START = {s: 5 + i for i, s in enumerate(SEASONS)}                                 # 13 = January of Y+1
DRY_MM = 60   # a 3-month normal below this is dry season: percentages of so little rain mean little


def _state_weights(da: xr.DataArray, st: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    """Share of each 0.5° cell inside each state (rasterised at 0.025°), on the cube's south-to-north rows."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    lat, lon, sub = da.lat.values, da.lon.values, 20
    res = float(abs(lon[1] - lon[0]))
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res / sub, res / sub)
    lab = rasterize([(g, i + 1) for i, g in enumerate(st.geometry)], out_shape=(lat.size * sub, lon.size * sub),
                    transform=tr, fill=0)
    return {pc: (lab == i + 1).reshape(lat.size, sub, lon.size, sub).mean((1, 3))[::-1] for i, pc in enumerate(st.pcode)}


def state_monthly(product: str = "chirps_v3") -> pd.DataFrame:
    """Monthly rain (mm) per state: CHIRPS area-weighted from the 0.5° cube; ERA5 from the team's admin-1 stats."""
    if product == "era5":
        return era5_adm1().pivot(index="valid_date", columns="pcode", values="mm")
    p = CACHE / f"{product}_state_monthly.parquet"
    da = rain(product)
    if not p.exists() or pd.Timestamp(pd.read_parquet(p).index.max()) < pd.Timestamp(da.time.values[-1]):
        v = da.values
        out = {}
        for pc, w in _state_weights(da, states()).items():
            ok = ~np.isnan(v).any(0) & (w > 0)
            out[pc] = (v[:, ok] * w[ok]).sum(1) / w[ok].sum()
        pd.DataFrame(out, index=pd.DatetimeIndex(da.time.values)).to_parquet(p)
    return pd.read_parquet(p)


def _dates(y: int, months: list[int]) -> list[pd.Timestamp]:
    return [pd.Timestamp(y + (m - 1) // 12, (m - 1) % 12 + 1, 1) for m in months]


def period_totals(mon: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Total over the given months (counted from January of the rain year; 13 = next January), per rain year."""
    out = {}
    for y in range(mon.index.year.min(), mon.index.year.max() + 1):
        d = _dates(y, months)
        if all(x in mon.index for x in d) and not mon.loc[d].isna().any().any():
            out[y] = mon.loc[d].sum()
    return pd.DataFrame(out).T


def season_totals(mon: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: period_totals(mon, [m0, m0 + 1, m0 + 2]) for s, m0 in SEASON_START.items()}


def composite(tot: dict[str, pd.DataFrame], years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median % anomaly (vs 1991–2020) per state × season over `years`, and the share of those years on the
    median's side; NaN where the season's normal is under DRY_MM."""
    med, agree = {}, {}
    for s in SEASONS:
        t = tot[s]
        clim = t.loc[1991:2020].mean()
        pct = t.loc[[y for y in years if y in t.index]] / clim * 100 - 100
        m = pct.median()
        med[s] = m.where(clim >= DRY_MM)
        agree[s] = (np.sign(pct) == np.sign(m)).mean().where(clim >= DRY_MM)
    return pd.DataFrame(med), pd.DataFrame(agree)


MAP_PERIODS = {"Jun–Sep": [6, 7, 8, 9], "Oct–Dec": [10, 11, 12], "Jan–Mar": [13, 14, 15], "Apr–May": [16, 17]}


def map_composite(years: list[int], product: str = "chirps_v3") -> dict[str, tuple[xr.DataArray, np.ndarray]]:
    """Median % anomaly per 0.5° cell for each map period over `years`, and the dry-season cells (under 20 mm a month)."""
    da = rain(product)
    t = pd.DatetimeIndex(da.time.values)
    out = {}
    for name, months in MAP_PERIODS.items():
        tot = {}
        for y in range(t.year.min(), t.year.max() + 1):
            d = _dates(y, months)
            if all(x in t for x in d):
                tot[y] = da.sel(time=d).sum("time", skipna=False)
        stack = xr.concat(list(tot.values()), pd.Index(list(tot), name="year"))
        clim = stack.sel(year=slice(1991, 2020)).mean("year")
        pct = (stack.sel(year=[y for y in years if y in tot]) / clim * 100 - 100).median("year")
        dry = (clim < 20 * len(months)).values & clim.notnull().values
        out[name] = (pct.where(clim >= 20 * len(months)), dry)
    return out


# --------------------------------------------------------------------------- #
# Reported impacts
# --------------------------------------------------------------------------- #
def emdat_wet() -> pd.DataFrame:
    """EM-DAT floods, wet mass movements and storms since 1980, dated to the middle of their start month."""
    em = emdat()
    em = em[em["Disaster Type"].isin(["Flood", "Mass movement (wet)", "Storm"]) & (em["Start Year"] >= 1980)].copy()
    em["month"] = em["Start Month"].fillna(6).astype(int)
    em["date"] = pd.to_datetime(dict(year=em["Start Year"], month=em["month"], day=15))
    em["ry"] = em["Start Year"] - (em["month"] < 5)
    em["affected"] = em["Total Affected"].fillna(0)
    em["deaths"] = em["Total Deaths"].fillna(0)
    return em


DI_WET = ["Inundación", "Lluvias", "Avenida torrencial", "Flujo de detritos", "Deslizamiento", "Tempestad", "Marejada"]
DI_Y0, DI_Y1 = 1975, 2012   # rain years whose ±3-year window sits inside the database (thin before the mid-1970s; stops early 2016)
DI_WIN = 3


def di_wet() -> pd.DataFrame:
    """DesInventar flood, rain, torrent, debris-flow, landslide and storm records, one per event and municipality."""
    di = desinventar()
    di = di[di.evento.isin(DI_WET)].copy()
    di["pcode"] = "VE" + di.level0.astype(str).str.zfill(2)
    m = di.fechames.replace(0, np.nan).fillna(6).clip(1, 12).astype(int)
    di["ry"] = di.fechano.astype(int) - (m < 5)
    di["region"] = di.pcode.map(PCODE_REGION)
    return di[di.region.notna()]


def di_index(di: pd.DataFrame) -> pd.DataFrame:
    """Records per region and rain year divided by the region's mean over the 2 × DI_WIN surrounding rain years:
    1 = an ordinary year for its time. Reporting effort in DesInventar changes several-fold between decades,
    so raw counts cannot be compared across them."""
    cnt = (di.groupby(["region", "ry"]).size().unstack(fill_value=0)
           .reindex(index=list(REGIONS), columns=range(DI_Y0 - DI_WIN, DI_Y1 + DI_WIN + 1), fill_value=0))
    cnt.loc["Venezuela"] = cnt.sum()
    out = {}
    for y in range(DI_Y0, DI_Y1 + 1):
        nb = [x for x in range(y - DI_WIN, y + DI_WIN + 1) if x != y]
        out[y] = cnt[y] / cnt[nb].mean(axis=1).replace(0, np.nan)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Guri, vegetation, harvests
# --------------------------------------------------------------------------- #
def guri_years(ph: pd.DataFrame) -> pd.DataFrame:
    """Per rain year: the Aug–Nov peak and the Feb–Jun low that follows (G-REALM, m above EGM2008)."""
    g = guri()
    rows = []
    for y in range(1992, g.index.max().year + 1):
        hi, lo = g[f"{y}-08-01":f"{y}-11-30"], g[f"{y + 1}-02-01":f"{y + 1}-06-30"]
        rows.append(dict(year=y, peak=hi.max() if len(hi) >= 6 else np.nan, low=lo.min() if len(lo) >= 9 else np.nan))
    df = pd.DataFrame(rows).set_index("year")
    df["drop"] = df.peak - df.low
    return df.join(ph)


def _norm(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def asap_dry(class_id: int) -> pd.DataFrame:
    """JRC ASAP FPAR z-score (whole land-cover mask, year-round), mean over November–April per rain year and state."""
    a = asap(220, class_id, 2)
    a = a[a.date.dt.month.isin([11, 12, 1, 2, 3, 4])].copy()
    a["ry"] = a.date.dt.year - (a.date.dt.month < 5)
    st = states()
    lookup = {_norm(n): p for p, n in zip(st.pcode, st.name)}
    a["pcode"] = a.region_name.map(lambda n: lookup.get(_norm(n)))
    if a.pcode.isna().any():
        raise ValueError(f"ASAP units without a state: {sorted(a.loc[a.pcode.isna(), 'region_name'].unique())}")
    n = a.groupby(["pcode", "ry"])["value"].count().unstack()
    m = a.groupby(["pcode", "ry"])["value"].mean().unstack()
    return m.where(n >= 15)          # a full Nov–Apr has 18 dekads


def yields() -> pd.DataFrame:
    """Maize and rice yield (t/ha) and its anomaly against the two years either side (FAOSTAT, flags kept)."""
    f = faostat()
    out = []
    for item, key in (("Maize (corn)", "maize"), ("Rice", "rice")):
        s = f[(f.Item == item) & (f.Element == "Yield")].set_index("year").sort_index()
        y = s["value"].astype(float) / (1000 if s["Unit"].iloc[0] == "kg/ha" else 10000 if s["Unit"].iloc[0] == "100 g/ha" else 1)
        nb = pd.concat([y.shift(k) for k in (-2, -1, 1, 2)], axis=1)
        anom = (y / nb.mean(axis=1, skipna=True).where(nb.notna().sum(axis=1) >= 2) - 1) * 100
        out.append(pd.DataFrame(dict(crop=key, yield_t=y, anom=anom, flag=s["flag"])))
    return pd.concat(out).reset_index()


# --------------------------------------------------------------------------- #
# SEAS5, October issuance: where this year's forecast ranks among the 1981–2025 forecasts for the same months
# (the seas5-skill app's admin-level method: trimester means, log1p, scaled to ERA5, both detrended)
# --------------------------------------------------------------------------- #
FC_SEASONS = {"OND": [10, 11, 12], "NDJ": [11, 12, 1], "DJF": [12, 1, 2], "JFM": [1, 2, 3], "FMA": [2, 3, 4]}


def seas5_rank(issued_month: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    s, e = seas5_adm1(), era5_adm1()
    s = s[s.issued_date.dt.month == issued_month].dropna(subset=["mean"])
    year = int(s.issued_date.dt.year.max())
    pct, skill, raw = {}, {}, {}
    for code, months in FC_SEASONS.items():
        wraps = 1 in months and 12 in months
        for pc in STATE_ORDER:
            f = s[(s.pcode == pc) & s.valid_date.dt.month.isin(months)]
            f = f.groupby(f.issued_date.dt.year)["mean"].mean()
            if not wraps and min(months) < issued_month:
                f.index = f.index + 1
            raw_f = f.copy()
            o = e[(e.pcode == pc) & e.valid_date.dt.month.isin(months)]
            sy = np.where((o.valid_date.dt.month > 6) | (not wraps), o.valid_date.dt.year, o.valid_date.dt.year - 1)
            g = o.groupby(sy)["mean"]
            o = g.mean()[g.count() == 3]
            f, o = np.log1p(f.clip(lower=0)), np.log1p(o.clip(lower=0))
            hist = f.index.intersection(o.index)
            f = (f - f[hist].mean()) / max(f[hist].std(ddof=1), 1e-9) * o[hist].std(ddof=1) + o[hist].mean()
            x = hist.values.astype(float)
            a_, b_ = np.polyfit(x, f[hist].values, 1)
            f = f - (a_ * f.index.values.astype(float) + b_) + f[hist].mean()
            a_, b_ = np.polyfit(x, o[hist].values, 1)
            o = o - (a_ * o.index.values.astype(float) + b_) + o[hist].mean()
            cur = int(f.index.max())
            if cur in o.index:
                raise ValueError(f"{code}: the latest SEAS5 issuance is already observed")
            h = f[hist].values
            pct.setdefault(code, {})[pc] = 100.0 * float((h < f[cur]).sum()) / len(h)
            raw.setdefault(code, {})[pc] = bool(raw_f.drop(cur).min() > raw_f[cur])
            skill.setdefault(code, {})[pc] = float(np.corrcoef(f[hist].values, o[hist].values)[0, 1])
    return pd.DataFrame(pct), pd.DataFrame(skill), year, pd.DataFrame(raw)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> Path:
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _source(fig, text: str, y: float = -0.02):
    fig.text(0.01, y, textwrap.fill(text, 170), fontsize=7, color=TEXT2, ha="left", va="top")


def _phase_legend(ax, ph: pd.DataFrame, loc="upper left", now: bool = False, **kw):
    h = [Line2D([], [], marker="o", ls="", color=STRONG_C, label="strong El Niño"),
         Line2D([], [], marker="o", ls="", color=NINO_C, label="El Niño"),
         Line2D([], [], marker="o", ls="", color=NEUTRAL_C, label="neutral"),
         Line2D([], [], marker="o", ls="", color=NINA_C, label="La Niña")]
    if now:
        h.append(Line2D([], [], marker="*", ls="", color=NOW_C, ms=10, label="2026"))
    ax.legend(handles=h, loc=loc, fontsize=8, **kw)


MONTH = "JFMAMJJASOND"
SEASON_LABEL = {s: f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][(m - 1) % 12]}–"
                   f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][(m + 1) % 12]}"
                for s, m in SEASON_START.items()}
PCT_BOUNDS = [-100, -40, -25, -10, 10, 25, 40, 100]
PCT_CMAP = mcolors.ListedColormap(DIV)
PCT_NORM = mcolors.BoundaryNorm(PCT_BOUNDS, PCT_CMAP.N)
DRY_GREY = "#dcdbd6"
plt.rcParams["hatch.color"] = "#c9c8c2"
plt.rcParams["hatch.linewidth"] = 0.6


def _hatch_cells(ax, mask: np.ndarray):
    """Dry-season cells of an imshow grid: white with grey hatching, so they cannot be read as 'near normal'."""
    from matplotlib.patches import Rectangle
    for i, j in zip(*np.where(mask)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#ffffff", hatch="////", edgecolor="#c9c8c2", lw=0, zorder=2))


DRY_PATCH = Patch(facecolor="#ffffff", hatch="////", edgecolor="#c9c8c2", lw=0)


def oni_now() -> dict:
    """The latest ONI season; the page assumes 2026 is a strong El Niño."""
    last = oni().iloc[-1]
    if last.anom < 1.5:
        raise ValueError(f"latest ONI {last.anom:+.2f}: 2026 is no longer a strong El Niño, reword the page")
    names = {"JJA": "Jun–Aug", "JAS": "Jul–Sep", "ASO": "Aug–Oct", "SON": "Sep–Nov", "MJJ": "May–Jul"}
    return dict(oni_now=f"{last.anom:+.1f}", oni_seas=names.get(last.seas, last.seas))


def _clip(da: xr.DataArray, geom) -> np.ndarray:
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    lat, lon = da.lat.values, da.lon.values
    res = float(abs(lon[1] - lon[0]))
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res, res)
    m = geometry_mask(geom, out_shape=(lat.size, lon.size), transform=tr, invert=True, all_touched=True)
    return m[::-1] if lat[0] < lat[-1] else m


def _edges(c: np.ndarray) -> np.ndarray:
    d = np.diff(c) / 2
    return np.concatenate([[c[0] - d[0]], c[:-1] + d, [c[-1] + d[-1]]])


def _map_axes(ax, st):
    st.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
    ax.set_xlim(-73.6, -59.6)
    ax.set_ylim(0.6, 12.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def fig_maps(ph: pd.DataFrame) -> Path:
    st = states()
    rows = [("El Niño", phase_years(ph, "nino")), ("La Niña", phase_years(ph, "nina"))]
    fig, axes = plt.subplots(len(rows), len(MAP_PERIODS), figsize=(11, 5.6))
    for i, (lab, yrs) in enumerate(rows):
        comp = map_composite(yrs)
        for j, (per, (da, dry_cells)) in enumerate(comp.items()):
            ax = axes[i, j]
            inside = _clip(da, st.geometry)
            v = np.where(inside, da.values, np.nan)
            dry = (inside & dry_cells).astype(float)
            ax.pcolormesh(da.lon, da.lat, v, cmap=PCT_CMAP, norm=PCT_NORM, shading="nearest")
            if dry.any():
                ax.pcolor(_edges(da.lon.values), _edges(da.lat.values), np.ma.masked_where(dry == 0, dry),
                          cmap=mcolors.ListedColormap(["#ffffff"]), hatch="////", lw=0)
            _map_axes(ax, st)
            if i == 0:
                ax.set_title(per + (" (year 1)" if j < 2 else " (year 2)"), fontsize=9.5)
            if j == 0:
                ax.text(-0.04, 0.5, f"{lab}\n{len(yrs)} years", transform=ax.transAxes, ha="right", va="center",
                        fontsize=9.5, fontweight="bold")
    cax = fig.add_axes([0.3, 0.1, 0.4, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=PCT_NORM, cmap=PCT_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks(PCT_BOUNDS[1:-1])
    cb.set_label("rain, % above or below the 1991–2020 normal (median of the years)", fontsize=8)
    DRY_PATCH.set_label("dry season (under 20 mm a month)")
    fig.legend(handles=[DRY_PATCH], loc="lower right",
               bbox_to_anchor=(0.98, 0.06), fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.2, wspace=0.03, hspace=0.06)
    _source(fig, "Year 1 = the year El Niño or La Niña develops (rain years May–April, 1981/82–2025/26; phase from the Sep–Nov ONI). "
                 "Source: CHIRPS v3 at 0.5°.", y=0.0)
    return _save(fig, "maps")


def _state_panel(ax, med: pd.DataFrame, names: pd.Series, title: str, labels: bool):
    m = med.reindex(STATE_ORDER)
    ax.imshow(m.values, cmap=PCT_CMAP, norm=PCT_NORM, aspect="auto")
    _hatch_cells(ax, m.isna().values)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}".replace("-", "−"), ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(v) >= 25 else "#0b0b0b")
    ax.set_xticks(range(len(SEASONS)), [SEASON_LABEL[s] for s in SEASONS], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(STATE_ORDER)), [names[p] for p in STATE_ORDER] if labels else [], fontsize=7.5)
    ax.tick_params(length=0)
    y = -0.5
    for r, (ps, _) in REGIONS.items():
        y += len(ps)
        ax.axhline(y, color=SURFACE, lw=2.2)
    ax.axvline(5.5, color=NOW_C, lw=0.8, ls=(0, (2, 2)))
    ax.set_title(title, fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_states(tot: dict, ph: pd.DataFrame) -> Path:
    names = states().set_index("pcode")["name"]
    nino, nina = phase_years(ph, "nino"), phase_years(ph, "nina")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 7.4), sharey=True)
    _state_panel(axes[0], composite(tot, nino)[0], names, f"El Niño ({len(nino)} years)", True)
    _state_panel(axes[1], composite(tot, nina)[0], names, f"La Niña ({len(nina)} years)", False)
    y = -0.5
    for r, (ps, _) in REGIONS.items():
        axes[1].text(len(SEASONS) - 0.3, y + len(ps) / 2, r, ha="left", va="center", fontsize=8, fontweight="bold")
        y += len(ps)
    fig.subplots_adjust(wspace=0.04, right=0.84)
    _source(fig, "Each cell: median rain over the El Niño (or La Niña) years, % above or below the state's 1991–2020 normal for "
                 "those three months. Left of the dashed line: the rains of the year El Niño or La Niña develops; right: the "
                 "dry season and the next spring. Hatched: dry season (normal under 60 mm in three months). Rain years May–April, 1981/82–2025/26. "
                 "Distrito Capital, La Guaira and Nueva Esparta rest on one or two partial 0.5° cells. Source: CHIRPS v3, state means.")
    return _save(fig, "states")


def _strip(ax, pct: pd.DataFrame, ph: pd.DataFrame, now: bool):
    """One row per region, split into El Niño / neutral / La Niña sub-rows; strong El Niño years larger and darker."""
    regs = list(pct.columns)
    off = {"nino": -0.24, "neutral": 0.0, "nina": 0.24}
    for i, r in enumerate(regs):
        for y, v in pct[r].dropna().items():
            if y == NOW or y not in ph.index:
                continue
            c = phase_color(y, ph)
            ax.scatter(v, i + off[ph.loc[y, "phase"]], s=40 if c == STRONG_C else 18, color=c, edgecolor=SURFACE, lw=0.6,
                       zorder=4 if c == STRONG_C else 3)
        if now and NOW in pct.index and not np.isnan(pct.loc[NOW, r]):
            v = pct.loc[NOW, r]
            ax.scatter(v, i + off["nino"], s=120, marker="*", color=NOW_C, zorder=5)
            ax.annotate(f"{v:.0f}%", (v, i + off["nino"]), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=7.5, fontweight="bold")
        if i:
            ax.axhline(i - 0.5, color=GRID, lw=0.8)
    ax.axvline(100, color="#9a9993", lw=0.7)
    ax.set_yticks(range(len(regs)), regs)
    ax.set_ylim(len(regs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    _grid(ax, "x")
    ax.set_xlabel("rain, % of the 1991–2020 normal")


def _region_pct(mon: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    t = period_totals(mon, months)
    out = {r: t[ps].mean(axis=1) for r, (ps, _) in REGIONS.items()}
    df = pd.DataFrame(out)
    return df / df.loc[1991:2020].mean() * 100


def fig_now(mon: pd.DataFrame, mon_e: pd.DataFrame, ph: pd.DataFrame) -> tuple[Path, dict]:
    last = mon.dropna(how="all").index.max()
    months = list(range(5, last.month + 1))
    pct = _region_pct(mon, months).loc[1981:]
    pe = _region_pct(mon_e, months).loc[1981:]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    _strip(ax, pct, ph, now=True)
    _phase_legend(ax, ph, loc="lower left", now=True, ncols=5, bbox_to_anchor=(0, 1.0))
    rank = pct.rank().loc[NOW]
    rk_e = pe.rank().loc[NOW]
    span = f"May–{last:%b}"
    _source(fig, f"{span} rain, each year 1981–{NOW}, one dot per year. 2026 was among the six driest of "
                 f"{len(pct)} in every region in CHIRPS v3 and among the seven driest in ERA5. Sources: CHIRPS v3 (state means), "
                 "ERA5 (team admin-1 statistics).")
    v = dict(now_span=span, now_lo=f"{pct.loc[NOW].min():.0f}", now_hi=f"{pct.loc[NOW].max():.0f}",
             now_rank_max=int(rank.max()), now_n=len(pct), era5_rank_max=int(rk_e.max()),
             era5_lo=f"{pe.loc[NOW].min():.0f}", era5_hi=f"{pe.loc[NOW].max():.0f}")
    if v["now_rank_max"] > 6 or v["era5_rank_max"] > 7:
        raise ValueError(f"2026 ranks changed ({v['now_rank_max']}, {v['era5_rank_max']}): reword the fig_now note")
    return _save(fig, "now"), v


def fig_dry(mon: pd.DataFrame, ph: pd.DataFrame) -> tuple[Path, dict]:
    months = [11, 12, 13, 14, 15]
    pct = _region_pct(mon, months).loc[1981:Y1]
    strong = [y for y in ph.index[ph.strong] if Y0 <= y <= Y1]
    if len(strong) != 4:
        raise ValueError(f"strong El Niño years are now {strong}: reword 'four strong El Niño years'")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    _strip(ax, pct, ph, now=False)
    _phase_legend(ax, ph, loc="lower left", ncols=4, bbox_to_anchor=(0, 1.0))
    _source(fig, f"November–March rain, rain years 1981/82–{Y1}/{(Y1 + 1) % 100:02d}, one dot per year; strong El Niño: "
                 + ", ".join(f"{y}/{(y + 1) % 100:02d}" for y in strong) + ". Source: CHIRPS v3, state means.")
    s = pct.loc[strong]
    nino = pct.loc[phase_years(ph, "nino")]
    v = dict(strong_below=int((s < 100).sum().sum()), strong_cells=int(s.size),
             strong_med=f"{100 - s.stack().median():.0f}")
    dry_third = {}
    for r in pct.columns:
        q = pct[r].quantile(1 / 3)
        dry_third[r] = int((nino[r] <= q).sum())
    v["dry_third_min"], v["dry_third_max"], v["nino_n"] = min(dry_third.values()), max(dry_third.values()), len(nino)
    v["dry_third_llanos"] = dry_third["Western Llanos"]
    fewest = sorted(r for r, n in dry_third.items() if n == v["dry_third_min"])
    v["dry_third_fewest"] = " and the ".join(r.lower() if r != "Western Llanos" else "western Llanos" for r in fewest)
    if v["strong_below"] != v["strong_cells"] or "Western Llanos" not in fewest:
        raise ValueError("the November–March pattern changed: reword the dry-season headline and note")
    return _save(fig, "dry"), v


EM_LABELS = {  # DisNo.: (label, x offset, y offset in points)
    "1999-0547-VEN": ("Dec 1999, Vargas: 30,000 dead", 10, 0), "2021-0538-VEN": ("Aug 2021, Mérida and 10+ states", -8, 0),
    "2022-0648-VEN": ("Oct 2022, Las Tejerías", -8, 6), "2015-0296-VEN": ("Jun 2015, Apure", 6, -2),
    "2005-0073-VEN": ("Feb 2005: 76 dead", 6, 0), "2010-0616-VEN": ("Nov–Dec 2010", -8, 4), "1987-0156-VEN": ("Sep 1987, Aragua", 6, 0),
    "2002-0445-VEN": ("Jul 2002, Llanos", 0, 10), "1993-0059-VEN": ("Aug 1993, storm Bret", 6, 0)}


def fig_emdat(ph: pd.DataFrame) -> tuple[Path, dict]:
    em = emdat_wet()
    em["phase"] = ph.phase.reindex(em.ry).values
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[3.2, 1]))
    _grid(ax)
    for _, r in em.iterrows():
        y = max(r.affected, 50)
        ax.scatter(r.date, y, s=18 + np.sqrt(r.deaths) * 3, color=phase_color(int(r.ry), ph), edgecolor=SURFACE, lw=0.6,
                   zorder=3, marker="o" if r.affected > 0 else "v")
        if r["DisNo."] in EM_LABELS:
            lab, dx, dy = EM_LABELS[r["DisNo."]]
            ax.annotate(lab, (r.date, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, va="center",
                        ha="right" if dx < 0 else "center" if dx == 0 else "left")
    ax.set_yscale("log")
    ax.set_ylim(30, 4e6)
    ax.set_ylabel("people affected (EM-DAT)")
    _phase_legend(ax, ph, loc="upper left", ncols=2)
    t = em.groupby("phase")[["affected", "deaths"]].sum().reindex(["nino", "neutral", "nina"]).fillna(0)
    years = ph.loc[1980:Y1, "phase"].value_counts()
    per = t.div(years, axis=0)
    for i, (k, c) in enumerate((("nino", NINO_C), ("neutral", NEUTRAL_C), ("nina", NINA_C))):
        w = float(per.loc[k, "affected"]) / 1000
        bx.barh(i, w, color=c, height=0.6)
        bx.text(w, i, f" {w:,.0f}k", va="center", fontsize=8)
    bx.set_yticks([0, 1, 2], [f"El Niño\n({years['nino']} yrs)", f"neutral\n({years['neutral']} yrs)", f"La Niña\n({years['nina']} yrs)"])
    bx.invert_yaxis()
    bx.set_xlabel("people affected per year\n(thousands)")
    bx.set_xlim(0, per["affected"].max() / 1000 * 1.45)
    _source(fig, "Floods, wet landslides and storms reported to EM-DAT since 1980, by the rain year (May–April) they fall in. Marker size: deaths; "
                 "triangles: no affected figure. Totals per year of each phase. Source: EM-DAT, CRED/UCLouvain (team snapshot).")
    v = dict(em_nina_aff=f"{t.loc['nina', 'affected'] / 1e6:.1f}", em_nino_aff=f"{t.loc['nino', 'affected'] / 1e6:.1f}",
             em_nina_dead=f"{t.loc['nina', 'deaths']:,.0f}", em_nino_dead=f"{t.loc['nino', 'deaths']:,.0f}",
             em_ratio=f"{per.loc['nina', 'affected'] / per.loc['nino', 'affected']:.0f}",
             em_ratio_ex=f"{(t.loc['nina', 'affected'] - em.loc[em['DisNo.'].isin(['1999-0547-VEN', '2021-0538-VEN']), 'affected'].sum()) / years['nina'] / per.loc['nino', 'affected']:.1f}",
             em_n_nino=int((em.phase == "nino").sum()), em_n_nina=int((em.phase == "nina").sum()),
             em_nino_yr=f"{per.loc['nino', 'affected'] / 1000:,.0f},000", em_nina_yr=f"{per.loc['nina', 'affected'] / 1000:,.0f},000",
             em_neutral_yr=f"{per.loc['neutral', 'affected'] / 1000:,.0f},000")
    return _save(fig, "emdat"), v


def fig_desinventar(ph: pd.DataFrame) -> tuple[Path, dict]:
    idx = di_index(di_wet())
    groups = {p: phase_years(ph, p, DI_Y0, DI_Y1) for p in ("nino", "neutral", "nina")}
    m = pd.DataFrame({p: idx[ys].median(axis=1) for p, ys in groups.items()})
    mean_nina = float(idx[groups["nina"]].mean(axis=1)["Venezuela"])
    rows = list(m.index)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _grid(ax, "x")
    for i, r in enumerate(rows):
        ax.plot([m.loc[r, "nino"], m.loc[r, "nina"]], [i, i], color="#c9c8c2", lw=1.6, zorder=1)
        for p, c in (("nino", NINO_C), ("neutral", NEUTRAL_C), ("nina", NINA_C)):
            ax.scatter(m.loc[r, p], i, s=52, color=c, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.axvline(1, color="#9a9993", lw=0.7)
    ax.set_yticks(range(len(rows)), [f"{r}" if r != "Venezuela" else "Venezuela (all)" for r in rows])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "Venezuela (all)":
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("flood and landslide records, relative to an ordinary year for the period (1 = ordinary); median of the years")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f"{lab} ({len(groups[p])} years)")
                       for p, c, lab in (("nino", NINO_C, "El Niño"), ("neutral", NEUTRAL_C, "neutral"), ("nina", NINA_C, "La Niña"))],
              loc="lower left", fontsize=8, ncols=3, bbox_to_anchor=(0, 1.0))
    _source(fig, f"Records of floods, heavy rain, torrents, debris flows, landslides and storms, one per event and municipality, rain years "
                 f"{DI_Y0}/{(DI_Y0 + 1) % 100:02d}–{DI_Y1}/{(DI_Y1 + 1) % 100:02d}. Each year's count is divided by the mean of the three "
                 "rain years either side, because reporting changes several-fold between decades. Source: DesInventar Venezuela (UNDRR), "
                 "which stops in early 2016.")
    regs = m.drop(index="Venezuela")
    above = regs.index[regs["nina"] > 1]
    v = dict(di_nino_lo=f"{regs['nino'].min():.1f}", di_nino_hi=f"{regs['nino'].max():.1f}",
             di_fewer_lo=f"{(1 - regs['nino'].max()) * 100:.0f}", di_fewer_hi=f"{(1 - regs['nino'].min()) * 100:.0f}",
             di_nina_above=len(above), di_regions=len(regs), di_mean_nina=f"{mean_nina:.1f}",
             di_llanos_nina=f"{m.loc['Western Llanos', 'nina']:.1f}", di_cn_nina=f"{m.loc['Centre-north', 'nina']:.1f}",
             di_n=int(len(di_wet())))
    if (regs["nino"] >= 1).any() or sorted(set(regs.index) - set(above)) != ["Centre-north", "Western Llanos"]:
        raise ValueError(f"the DesInventar pattern changed (La Niña above 1 in {list(above)}): reword the floods headline")
    return _save(fig, "desinventar"), v


def fig_guri(ph: pd.DataFrame) -> tuple[Path, dict]:
    g = guri()
    gy = guri_years(ph)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw=dict(width_ratios=[3, 1.1]))
    _grid(ax)
    for y in gy.index[(gy.phase == "nino")]:
        ax.axvspan(pd.Timestamp(y, 12, 1), pd.Timestamp(y + 1, 6, 1), color=NINO_C, alpha=0.13, lw=0)
    ax.plot(g.index, g.values, color="#3b3a37", lw=0.8)
    for yr, lab in ((2009, "2010 rationing"), (2015, "2016 rationing")):
        lo = gy.loc[yr, "low"]
        d = g[f"{yr + 1}-02-01":f"{yr + 1}-06-30"].idxmin()
        ax.annotate(lab, (d, lo), xytext=(6, -2), textcoords="offset points", fontsize=7.5, va="top")
    drops = gy.loc[gy.phase == "nino", "drop"].dropna()
    peak = float(g[f"{NOW}-08-01":].max())
    lo_hi = (peak - drops.max(), peak - drops.min())
    x = pd.Timestamp(NOW + 1, 5, 1)
    ax.plot([x, x], lo_hi, color=NOW_C, lw=3, solid_capstyle="butt")
    ax.annotate(f"May {NOW + 1} if it falls\nlike past El Niño years", (x, lo_hi[0]), xytext=(-8, -2), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("Guri reservoir level (m, satellite)")
    ax.set_xlim(pd.Timestamp(1992, 6, 1), pd.Timestamp(NOW + 1, 12, 31))
    ax.legend(handles=[Patch(color=NINO_C, alpha=0.25, label="dry season after El Niño")], loc="lower left", fontsize=8)
    # right: drawdown per rain year by phase
    _grid(bx)
    cats = [("nina", "La Niña", NINA_C), ("neutral", "neutral", NEUTRAL_C), ("nino", "El Niño", NINO_C)]
    for i, (p, lab, c) in enumerate(cats):
        d = -gy.loc[gy.phase == p, "drop"].dropna()
        xs = i + np.linspace(-0.18, 0.18, len(d)) if len(d) > 1 else [i]
        for xx, (yr, val) in zip(xs, d.sort_values(ascending=False).items()):
            cc = STRONG_C if gy.loc[yr, "strong"] else c
            bx.scatter(xx, val, s=26, color=cc, edgecolor=SURFACE, lw=0.6, zorder=3)
            if p == "nino" and (gy.loc[yr, "strong"] or yr == 2009):
                bx.annotate(f"{year_label(yr)}", (xx, val), xytext=(-5 if yr in (2009, 2023) else 5, 0), textcoords="offset points",
                            fontsize=6.5, va="center", ha="right" if yr in (2009, 2023) else "left")
        bx.hlines(d.median(), i - 0.28, i + 0.28, color=NOW_C, lw=1.2)
    bx.set_xticks(range(3), [lab for _, lab, _ in cats])
    bx.axhline(0, color="#9a9993", lw=0.6)
    bx.set_ylabel("change, Aug–Nov peak to Feb–Jun low (m)")
    _source(fig, "Satellite altimetry, 10-day, heights above the EGM2008 geoid (not the operator's gauge datum: compare changes, not "
                 "levels). Black bar: this year's August–October peak minus the smallest and largest falls after past El Niño years. "
                 "Lines: medians. Source: NASA GSFC G-REALM (Guri, target 432).")
    med = gy.groupby("phase")["drop"].median()
    v = dict(guri_nino=f"{med['nino']:.0f}", guri_nina=f"{med['nina']:.0f}", guri_peak=f"{peak:.1f}",
             guri_lo=f"{lo_hi[0]:.0f}", guri_hi=f"{lo_hi[1]:.0f}", guri_low16=f"{gy.loc[2015, 'low']:.0f}",
             guri_low10=f"{gy.loc[2009, 'low']:.0f}", guri_peak15=f"{gy.loc[2015, 'peak']:.0f}", guri_n_nino=len(drops),
             next_year=NOW + 1)
    v["guri_gap10"] = f"{lo_hi[0] - gy.loc[2009, 'low']:.0f}"
    if not 0 < lo_hi[0] - gy.loc[2009, "low"] < 3:
        raise ValueError("the Guri range moved against the 2010 low: reword the Guri note")
    return _save(fig, "guri"), v


def fig_veg(ph: pd.DataFrame) -> tuple[Path, dict]:
    crop, rng = asap_dry(1), asap_dry(2)
    nat = crop.mean()
    nat_r = rng.mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    _grid(ax)
    yrs = [y for y in nat.index if y < NOW]
    ax.bar(yrs, nat.loc[yrs], color=[phase_color(y, ph) for y in yrs], width=0.75)
    ax.scatter(yrs, nat_r.loc[yrs], marker="_", s=60, color=NOW_C, zorder=3, label="rangeland")
    ax.axhline(0, color="#9a9993", lw=0.6)
    ax.set_ylabel("vegetation (FPAR) z-score, Nov–Apr")
    ax.set_xticks(yrs[::2], [year_label(y) for y in yrs[::2]], rotation=90, fontsize=7.5)
    h = [Patch(color=STRONG_C, label="strong El Niño"), Patch(color=NINO_C, label="El Niño"), Patch(color=NEUTRAL_C, label="neutral"),
         Patch(color=NINA_C, label="La Niña"), Line2D([], [], marker="_", ls="", ms=10, color=NOW_C, label="rangeland")]
    ax.legend(handles=h, loc="lower left", fontsize=7.5, ncols=5, bbox_to_anchor=(0, 1.0))
    st = states()
    nino = [y for y in phase_years(ph, "nino", 2000, NOW - 1)]
    sm = st.merge(crop[nino].mean(axis=1).rename("z"), left_on="pcode", right_index=True, how="left")
    zb = [-1, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 1]
    cm = mcolors.ListedColormap(DIV)
    sm.plot(ax=bx, column="z", cmap=cm, norm=mcolors.BoundaryNorm(zb, cm.N), edgecolor="#ffffff", lw=0.5,
            missing_kwds=dict(color=DRY_GREY))
    bx.set_axis_off()
    bx.set_title(f"Nov–Apr after El Niño, cropland\n(mean of {len(nino)} rain years)", fontsize=9)
    cax = fig.add_axes([0.72, 0.1, 0.22, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(zb, cm.N), cmap=cm), cax=cax, orientation="horizontal")
    cb.set_ticks(zb[1:-1])
    cb.ax.tick_params(labelsize=7)
    _source(fig, "Bars: cropland, mean of the state values; dashes: rangeland. Z-score of FPAR (the share of sunlight the canopy absorbs) "
                 "against each 10-day period's 2001–present record, November–April, rain years 2000/01–2025/26. "
                 "Source: JRC ASAP (MODIS FPAR, GAUL admin-1).")
    worst = nat.loc[yrs].nsmallest(3)
    after = [y for y in worst.index if ph.phase.get(y) == "nino"]
    v = dict(veg_worst=", ".join(year_label(y) for y in sorted(worst.index)), veg_worst_nino=len(after),
             veg_2023=f"{nat.loc[2023]:+.1f}".replace("-", "−"),
             veg_states_neg=int((crop[nino].mean(axis=1) < 0).sum()), veg_states_n=int(crop[nino].mean(axis=1).notna().sum()))
    if len(after) != 3 or nat.loc[2023] < -0.1:
        raise ValueError("the vegetation pattern changed: reword the vegetation headline")
    return _save(fig, "veg"), v


def fig_yields(ph: pd.DataFrame) -> tuple[Path, dict]:
    y = yields()
    y = y[y.year >= 1980]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    for ax, (crop, lab) in zip(axes, (("maize", "Maize"), ("rice", "Rice"))):
        d = y[y.crop == crop].set_index("year")
        _grid(ax)
        ax.axvspan(2013.5, 2019.5, color="#ebeae6", lw=0)
        ax.bar(d.index, d.anom, color=[phase_color(int(t), ph) if t < NOW else NEUTRAL_C for t in d.index], width=0.75,
               hatch=["////" if f != "A" else "" for f in d.flag], edgecolor=SURFACE, lw=0)
        ax.axhline(0, color="#9a9993", lw=0.6)
        ax.set_ylabel(f"{lab} yield\n% vs years around", fontsize=8)
        ax.set_ylim(-30, 30)
        for t in d.index[(d.anom < -9) & (d.index < 2014)]:
            if ph.phase.get(t) == "nino":
                ax.annotate(str(t), (t, d.loc[t, "anom"]), xytext=(0, -9), textcoords="offset points", ha="center", fontsize=7)
    axes[0].text(2016.5, 26, "economic crisis", ha="center", va="top", fontsize=7.5, color=TEXT2)
    h = [Patch(color=STRONG_C, label="strong El Niño"), Patch(color=NINO_C, label="El Niño"), Patch(color=NEUTRAL_C, label="neutral"),
         Patch(color=NINA_C, label="La Niña")]
    axes[1].legend(handles=h, loc="lower left", fontsize=7.5, ncols=4)
    _source(fig, "National yield in each calendar year (the main harvest follows the May–October rains) against the mean of the two years "
                 "either side. Hatched: FAOSTAT estimate, not an official figure. Source: FAOSTAT.")
    d = y[(y.crop == "maize") & (y.year < 2014)].set_index("year")
    nino = d.loc[[t for t in phase_years(ph, "nino", 1980, 2013)], "anom"]
    worst = d["anom"].loc[2000:2013].nsmallest(3)
    v = dict(maize_worst=", ".join(str(t) for t in sorted(worst.index)),
             maize_worst_nino=int(sum(ph.phase.get(t) == "nino" for t in worst.index)),
             maize_strong=", ".join(f"{t} {d.loc[t, 'anom']:+.0f}%".replace("-", "−") for t in (1982, 1997) if t in d.index),
             maize_nino_med=f"{nino.median():+.0f}".replace("-", "−"), maize_nino_n=len(nino),
             **{f"maize_{t}": f"{abs(y[(y.crop == 'maize') & (y.year == t)].anom.iloc[0]):.0f}"
                for t in (1982, 1997, 2002, 2009, 2015, 2023)})
    a = {t: y[(y.crop == "maize") & (y.year == t)].anom.iloc[0] for t in (1982, 1997, 2002, 2009, 2015, 2023)}
    if not (a[2002] < -8 and a[2009] < -8 and a[2015] < -8 and all(abs(a[t]) < 6 for t in (1982, 1997, 2023))):
        raise ValueError(f"maize anomalies changed {a}: reword the yields headline")
    return _save(fig, "yields"), v


FC_BOUNDS = [0, 10, 33.4, 66.6, 90, 100.01]
FC_CMAP = mcolors.ListedColormap([DIV[0], DIV[2], "#ebeae6", DIV[4], DIV[6]])
FC_NORM = mcolors.BoundaryNorm(FC_BOUNDS, FC_CMAP.N)


def fig_forecast(mon_e: pd.DataFrame) -> tuple[Path, dict]:
    pr, sk, year, raw = seas5_rank()
    names = states().set_index("pcode")["name"]
    pr, sk = pr.reindex(STATE_ORDER), sk.reindex(STATE_ORDER)
    clim = {}
    for code, months in FC_SEASONS.items():
        mm = [m if m >= 10 else m + 12 for m in months]
        clim[code] = period_totals(mon_e, mm).loc[1991:2020].mean()
    clim = pd.DataFrame(clim).reindex(STATE_ORDER)
    dry = clim < DRY_MM
    st = states()
    labels = {"OND": "Oct–Dec", "NDJ": "Nov–Jan", "DJF": "Dec–Feb", "JFM": "Jan–Mar", "FMA": "Feb–Apr"}
    fig, axes = plt.subplots(1, len(FC_SEASONS), figsize=(13, 3.6))
    for ax, code in zip(axes, FC_SEASONS):
        g = st.merge(pr[code].rename("p"), left_on="pcode", right_index=True)
        g["dry"] = g.pcode.map(dry[code])
        g["weak"] = g.pcode.map(sk[code] < 0.3) & ~g["dry"]
        g[~g["dry"]].plot(ax=ax, column="p", cmap=FC_CMAP, norm=FC_NORM, edgecolor="#ffffff", lw=0.5)
        if g["dry"].any():
            g[g["dry"]].plot(ax=ax, facecolor="#ffffff", hatch="////", edgecolor="#c9c8c2", lw=0.5)
        if g["weak"].any():
            g[g["weak"]].plot(ax=ax, facecolor="none", hatch="....", edgecolor="#52514e", lw=0)
        st.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
        ax.set_title(labels[code], fontsize=9.5)
        ax.set_xlim(-73.6, -59.6)
        ax.set_ylim(0.6, 12.4)
        ax.set_aspect("equal")
        ax.set_axis_off()
    h = [Patch(color=c, label=l) for c, l in zip([DIV[0], DIV[2], "#ebeae6", DIV[4], DIV[6]],
                                                 ["driest tenth", "drier third", "middle third", "wetter third", "wettest tenth"])]
    h += [Patch(facecolor="#ffffff", hatch="....", edgecolor="#52514e", lw=0, label="weak skill"), DRY_PATCH]
    DRY_PATCH.set_label("dry season")
    fig.legend(handles=h, loc="lower center", ncols=7, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.03)
    _source(fig, f"Where the {year} October forecast ranks among the 45 October forecasts for 1981–2025, by state: ensemble mean, "
                 "detrended. Dotted: weak skill (hindcast correlation with ERA5 under 0.3). Dry season: normal under 60 mm in three months. "
                 "Source: ECMWF SEAS5 and ERA5, team admin-1 statistics.", y=0.0)
    wet_season = ~dry
    v = dict(fc_year=year, fc_driest=int(((pr[["OND", "NDJ", "DJF"]] < 10) & wet_season[["OND", "NDJ", "DJF"]]).sum().sum()),
             fc_cells=int(wet_season[["OND", "NDJ", "DJF"]].sum().sum()),
             fc_r_lo=f"{sk[['OND', 'NDJ', 'DJF']].min().min():.1f}", fc_r_hi=f"{sk[['OND', 'NDJ', 'DJF']].max().max():.1f}",
             fc_raw_driest=int(raw.reindex(STATE_ORDER)["NDJ"].sum()), fc_states=len(STATE_ORDER))
    return _save(fig, "forecast"), v

# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
EXTRA_CSS = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0 8px}
.tile{border:1px solid #e2e7e7;border-top:3px solid #b5541b;border-radius:6px;padding:12px 14px;background:#fff}
.tile .big{font-size:24px;font-weight:700;margin:0 0 4px;line-height:1.15}
.tile p{font-size:13px;color:var(--n8);margin:0}
section h2{font-size:18px;line-height:1.35}
.note{font-size:13px;color:var(--n7);margin:-6px 0 14px}
details{margin:24px 0}details summary{cursor:pointer;font-weight:600}
details li{font-size:13.5px;margin:0 0 8px;max-width:90ch}
"""


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def analyse() -> dict:
    ph = phases()
    mon, mon_e = state_monthly("chirps_v3"), state_monthly("era5")
    return dict(ph=ph, mon=mon, mon_e=mon_e, tot=season_totals(mon), tot_e=season_totals(mon_e))


def era5_agreement(a: dict) -> dict:
    """How often ERA5's El Niño median has the same sign as CHIRPS's, where CHIRPS shows at least ±10%."""
    nino = phase_years(a["ph"], "nino")
    c, e = composite(a["tot"], nino)[0], composite(a["tot_e"], nino)[0]
    m = ((c.abs() >= 10) & e.notna()).values
    same = (np.sign(c.values) == np.sign(e.values))[m]
    return dict(era5_agree=f"{same.mean() * 100:.0f}", era5_cells=int(m.sum()))


def figures(a: dict) -> tuple[dict, dict]:
    ph, f, v = a["ph"], {}, {}
    v |= oni_now()
    f["maps"] = fig_maps(ph)
    f["states"] = fig_states(a["tot"], ph)
    v |= era5_agreement(a)
    f["now"], x = fig_now(a["mon"], a["mon_e"], ph); v |= x
    f["dry"], x = fig_dry(a["mon"], ph); v |= x
    f["forecast"], x = fig_forecast(a["mon_e"]); v |= x
    f["emdat"], x = fig_emdat(ph); v |= x
    f["desinventar"], x = fig_desinventar(ph); v |= x
    f["guri"], x = fig_guri(ph); v |= x
    f["veg"], x = fig_veg(ph); v |= x
    f["yields"], x = fig_yields(ph); v |= x
    nino, nina = phase_years(ph, "nino"), phase_years(ph, "nina")
    v |= dict(n_nino=len(nino), n_nina=len(nina), strong_years=", ".join(str(y) for y in ph.index[ph.strong] if y >= Y0),
              data_end=f"{a['mon'].dropna(how='all').index.max():%B %Y}", era5_end=f"{a['mon_e'].dropna(how='all').index.max():%B %Y}",
              emdat_end=str(emdat()["Last Update"].max()) if "Last Update" in emdat() else "")
    return f, v


def render(spec: dict, figs: dict, v: dict) -> str:
    import enso_deep_dive as edd
    pg = spec["page"]
    out = [edd.HEAD.format(title=html.escape(pg["title"]), desc=html.escape(pg["sub"]), css=edd.CSS + EXTRA_CSS,
                           home="../", home_label="ENSO deep dives")]
    out.append('<p class="eyebrow">El Niño 2026/27</p>')
    out.append(f'<h1>{pg["title"]}</h1><p class="meta">{pg["sub"].format(**v)}<br>{pg["dateline"].format(**v)}</p>')
    out.append('<div class="tiles">' + "".join(f'<div class="tile"><p class="big">{t["big"].format(**v)}</p><p>{t["text"].format(**v)}</p></div>'
                                                for t in spec["tiles"]) + "</div>")
    for sec in spec["sections"]:
        if sec.get("part"):
            out.append(f'<h2 style="margin-top:36px">{sec["part"]}</h2>')
        out.append(f'<section id="{sec["id"]}"><h3>{sec["head"].format(**v)}</h3>')
        if sec.get("note"):
            out.append(f'<p class="note">{sec["note"].format(**v)}</p>')
        out.append(f'<figure><img src="{_img(figs[sec["fig"]])}" alt="{html.escape(sec["head"].format(**v))}"></figure></section>')
    notes = "".join(f"<li>{n.format(**v)}</li>" for n in spec["notes"])
    out.append(f'<details><summary>About the data</summary><ul>{notes}</ul></details>')
    out.append(f'<p class="small">OCHA Centre for Humanitarian Data · built {pd.Timestamp.now():%Y-%m-%d}</p>')
    out.append(edd.FOOT)
    return "\n".join(out)


def write_index() -> None:
    """Re-render the /enso/ card index from every deep_dives/*.toml."""
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def build(spec: dict | None = None, *_args, publish: bool = False) -> dict:
    """Called by enso_deep_dive.main() for builder = "ven" (grid, gdf, indices unused), or standalone."""
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    a = analyse()
    figs, v = figures(a)
    page = render(spec, figs, v)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print("  values:", v)
    print(f"  wrote {OUT}/index.html")
    if publish:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "index.html").write_text(page, encoding="utf-8")
        write_index()
        print(f"  published -> {DOCS}/index.html and docs/enso/index.html")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/ven/ and the /enso/ card index")
    args = ap.parse_args()
    build(publish=args.publish)


if __name__ == "__main__":
    main()
