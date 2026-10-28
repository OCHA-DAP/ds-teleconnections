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
import re
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
from matplotlib import ticker as mticker
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
RAINY_C = "#e6f0f7"
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
# Language: the page is built in English (out/ven/) and again in Spanish (out/ven/es/) from the same analysis,
# figures included. The narrative lives in deep_dives/ven.toml and deep_dives/i18n/ven.es.toml (kept out of
# deep_dives/*.toml, which the dispatcher and the card index glob); the strings below are figure labels and captions.
# --------------------------------------------------------------------------- #
LANG = "en"
ES_SPEC = ROOT / "deep_dives" / "i18n" / f"{SLUG}.es.toml"
MON_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MON_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MONTH_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
            "diciembre"]
REGION_ES = {"North-west": "Noroeste", "Andes": "Andes", "Western Llanos": "Llanos occidentales", "Centre-north": "Centro-norte",
             "Central and eastern Llanos": "Llanos centrales y orientales", "North-east coast": "Costa nororiental",
             "Guayana and Delta": "Guayana y Delta"}
REGION_ES_IN = {"North-west": "el noroeste", "Andes": "los Andes", "Western Llanos": "los Llanos occidentales",
                "Centre-north": "el centro-norte", "Central and eastern Llanos": "los Llanos centrales y orientales",
                "North-east coast": "la costa nororiental", "Guayana and Delta": "Guayana y el Delta"}


def set_lang(lang: str) -> None:
    global LANG, FIGDIR
    LANG = lang
    FIGDIR = (OUT / "es" if lang == "es" else OUT) / "fig"
    FIGDIR.mkdir(parents=True, exist_ok=True)


def _t(en: str, es: str) -> str:
    return es if LANG == "es" else en


def _num(s: str) -> str:
    """A formatted number with a true minus; in Spanish, decimal comma and a point between thousands."""
    s = s.replace("-", "−")
    return s.translate(str.maketrans(",.", ".,")) if LANG == "es" else s


NUM_RE = re.compile(r"[+−-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+−-]?\d+\.\d+")


def _localise(v: dict) -> dict:
    """Page values that are plain formatted numbers, in the page's language (everything else is left alone)."""
    return {k: _num(x) if isinstance(x, str) and NUM_RE.fullmatch(x) else x for k, x in v.items()}


def _mon(m: int) -> str:
    """Month abbreviation; m counts on past 12 into the next year."""
    return (MON_ES if LANG == "es" else MON_EN)[(m - 1) % 12]


def _span(m0: int, m1: int) -> str:
    return f"{_mon(m0)}–{_mon(m1)}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _month_year(t: pd.Timestamp) -> str:
    return f"{MONTH_ES[t.month - 1]} de {t.year}" if LANG == "es" else f"{t:%B %Y}"


NUM_WORDS = {2: ("two", "dos"), 3: ("three", "tres"), 4: ("four", "cuatro"), 5: ("five", "cinco"), 6: ("six", "seis"),
             7: ("seven", "siete"), 8: ("eight", "ocho"), 9: ("nine", "nueve"), 10: ("ten", "diez")}


def _words(n: int) -> str:
    return _t(*NUM_WORDS[n]) if n in NUM_WORDS else str(n)


def _reg(r: str) -> str:
    return REGION_ES.get(r, r) if LANG == "es" else r


def _phase_name(p: str) -> str:
    return {"nino": "El Niño", "neutral": _t("neutral", "neutro"), "nina": "La Niña"}[p]


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
    if product == "chirps_v3":
        ext = _chirps_recent(ds)
        if ext is not None:
            ds = xr.concat([ds, ext], "time")
    da = ds["precip"]
    if "valid_frac" in ds:
        da = da.where(ds["valid_frac"] > 0.5)
    return da.sortby("lat")


CHIRPS_TIF = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/latam/tifs/chirps-v3.0.{y}.{m:02d}.tif"


def _chirps_recent(cube: xr.Dataset) -> xr.Dataset | None:
    """CHIRPS v3 months newer than the team cube, from CHC's monthly tifs, put on the cube's 0.5° grid the way the cube is
    made (mean of the valid 0.05° cells, and their share as valid_frac); for August 2026, where both exist, identical."""
    import rasterio
    out, t = [], pd.Timestamp(cube.time.values[-1]) + pd.DateOffset(months=1)
    while t < pd.Timestamp.now().normalize().replace(day=1):
        f = CACHE / "chirps_v3_tifs" / f"chirps-v3.0.{t.year}.{t.month:02d}.tif"
        if not f.exists():
            r = requests.get(CHIRPS_TIF.format(y=t.year, m=t.month), timeout=300)
            if r.status_code != 200:
                break
            if r.content[:4] not in (b"II*\x00", b"MM\x00*"):   # an error page served as 200: don't cache it as a tif
                raise ValueError(f"CHC returned something other than a GeoTIFF for {t:%Y-%m}")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(r.content)
        with rasterio.open(f) as src:
            arr, tr = src.read(1).astype(float), src.transform
        arr[arr < 0] = np.nan
        g = xr.DataArray(arr, dims=("lat", "lon"), coords=dict(lat=tr.f + tr.e * (np.arange(arr.shape[0]) + 0.5),
                                                               lon=tr.c + tr.a * (np.arange(arr.shape[1]) + 0.5)))
        g = g.sel(lat=slice(BOX["lat"][1], BOX["lat"][0]), lon=slice(*BOX["lon"]))
        grid = dict(lat=cube.lat, lon=cube.lon, method="nearest", tolerance=0.01)
        mean = g.coarsen(lat=10, lon=10, boundary="trim").mean().reindex(**grid)
        frac = g.notnull().coarsen(lat=10, lon=10, boundary="trim").mean().reindex(**grid)
        out.append(xr.Dataset(dict(precip=mean, valid_frac=frac)).expand_dims(time=[t]))
        t += pd.DateOffset(months=1)
    return xr.concat(out, "time") if out else None


def emdat() -> pd.DataFrame:
    p = CACHE / "emdat_ven.parquet"
    if _stale(p, 60):
        import ocha_stratus as stratus
        stratus.emdat.load_emdat_from_blob("VEN", include_historic=True, stage="dev").to_parquet(p)
    return pd.read_parquet(p)


def cerf() -> pd.DataFrame:
    """CERF allocations to Venezuela (the team's OneGMS mirror in the dev database; regional VZR allocations not included)."""
    return _db("cerf_ven", "select application_code, year, window_name, emergency_type, title, amount_approved, "
                           "erc_endorsement_date from aa.cerf_allocation where country_iso3 = 'VEN'", "dev")


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
# Three-month windows in calendar order. For El Niño and La Niña years a window starting in January–April is taken from
# the year after the event develops (the end of its rain year), the rest from the year it develops: 13 = January of Y+1.
SEASONS = ["JFM", "FMA", "MAM", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF"]
SEASON_START = {s: m + 12 if m < 5 else m for s, m in zip(SEASONS, [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])}
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


# --------------------------------------------------------------------------- #
# INAMEH rainy-season zones: deep_dives/data/ven_inameh_zones.py digitizes INAMEH's map of rainy-season onset. Each zone's
# rainy season runs from its onset month to November (the end FEWS NET uses), its dry season from December to the onset.
# --------------------------------------------------------------------------- #
ZONES_FILE = ROOT / "deep_dives" / "data" / "ven_inameh_rainy_season_zones.geojson"
ZONE_ONSET = {"A": 4, "B": 4, "C": 5, "D": 6, "E": 6}   # month of INAMEH's onset fortnight
ZONE_ONSET_TXT = {"A": ("early April", "inicios de abril"), "B": ("late April", "fines de abril"),
                  "C": ("late May", "fines de mayo"), "D": ("early June", "inicios de junio"), "E": ("late June", "fines de junio")}
ZONE_FILL = {"A": "#cfe5da", "B": "#a6d0e2", "C": "#6fb1b0", "D": "#9fb7e1", "E": "#4d84cc"}
FEWS_DIR = ROOT / "deep_dives" / "data" / "fewsnet"   # FEWS NET figures, reproduced with credit (see the page notes)


def zones() -> gpd.GeoDataFrame:
    return gpd.read_file(ZONES_FILE)


def zone_monthly(product: str = "chirps_v3") -> pd.DataFrame:
    """Monthly rain (mm) per INAMEH zone, area-weighted from the 0.5° cube as for the states."""
    p = CACHE / f"{product}_zone_monthly.parquet"
    da = rain(product)
    if (not p.exists() or p.stat().st_mtime < ZONES_FILE.stat().st_mtime
            or pd.Timestamp(pd.read_parquet(p).index.max()) < pd.Timestamp(da.time.values[-1])):
        v = da.values
        out = {}
        for z, w in _state_weights(da, zones().rename(columns={"zone": "pcode"})).items():
            ok = ~np.isnan(v).any(0) & (w > 0)
            out[z] = (v[:, ok] * w[ok]).sum(1) / w[ok].sum()
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


# Three-month seasons in FEWS NET's order (its El Niño maps show June–August, September–November, December–February),
# then March–May. Same rule as elsewhere: Dec–Feb starts in December, the year the event develops; Mar–May is the year after.
MAP_PERIODS = {"Jun–Aug": [6, 7, 8], "Sep–Nov": [9, 10, 11], "Dec–Feb": [12, 13, 14], "Mar–May": [15, 16, 17]}
def _map_stacks(product: str = "chirps_v3") -> dict[str, xr.DataArray]:
    """Total rain per 0.5° cell for each map period, one layer per rain year."""
    da = rain(product)
    t = pd.DatetimeIndex(da.time.values)
    out = {}
    for name, months in MAP_PERIODS.items():
        tot = {}
        for y in range(t.year.min(), t.year.max() + 1):
            d = _dates(y, months)
            if all(x in t for x in d):
                tot[y] = da.sel(time=d).sum("time", skipna=False)
        out[name] = xr.concat(list(tot.values()), pd.Index(list(tot), name="year"))
    return out


def map_normal(product: str = "chirps_v3") -> dict[str, xr.DataArray]:
    """1991–2020 normal per 0.5° cell for each map period, in mm a month."""
    return {name: stack.sel(year=slice(1991, 2020)).mean("year") / len(MAP_PERIODS[name])
            for name, stack in _map_stacks(product).items()}


def map_composite(years: list[int], product: str = "chirps_v3") -> dict[str, tuple[xr.DataArray, np.ndarray]]:
    """Per 0.5° cell and map period, as FEWS NET maps El Niño: how often `years` fell in the driest third (negative %) or the
    wettest third (positive %) of 1991–2020, whichever is more frequent; and the dry-season cells (under 20 mm a month)."""
    out = {}
    for name, stack in _map_stacks(product).items():
        n = len(MAP_PERIODS[name])
        base = stack.sel(year=slice(1991, 2020))
        clim = base.mean("year")
        sel = stack.sel(year=[y for y in years if y in stack.year])
        below = (sel < base.quantile(1 / 3, "year")).mean("year") * 100
        above = (sel > base.quantile(2 / 3, "year")).mean("year") * 100
        freq = xr.where(below >= above, -below, above)
        dry = (clim < 20 * n).values & clim.notnull().values
        out[name] = (freq.where(clim >= 20 * n), dry)
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
    if LANG == "es":   # decimal commas on every plain numeric axis (dates, logs and labelled ticks keep their own)
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if type(axis.get_major_formatter()) is mticker.ScalarFormatter and axis.get_scale() == "linear":
                    axis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _num(f"{x:g}")))
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _source(fig, text: str, y: float = -0.02, width: int = 170):
    fig.text(0.01, y, textwrap.fill(text, width), fontsize=7, color=TEXT2, ha="left", va="top")


def _phase_legend(ax, ph: pd.DataFrame, loc="upper left", now: bool = False, **kw):
    h = [Line2D([], [], marker="o", ls="", color=STRONG_C, label=_t("strong El Niño", "El Niño fuerte")),
         Line2D([], [], marker="o", ls="", color=NINO_C, label="El Niño"),
         Line2D([], [], marker="o", ls="", color=NEUTRAL_C, label=_phase_name("neutral")),
         Line2D([], [], marker="o", ls="", color=NINA_C, label="La Niña")]
    if now:
        h.append(Line2D([], [], marker="*", ls="", color=NOW_C, ms=10, label="2026"))
    ax.legend(handles=h, loc=loc, fontsize=8, **kw)


def _month_letters() -> str:
    return _t("JFMAMJJASOND", "EFMAMJJASOND")
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
    m0 = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"].index(last.seas)
    return dict(oni_now=f"{last.anom:+.1f}", oni_seas=_span(m0, m0 + 2) if m0 else _span(12, 14))


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


# FEWS NET's frequency scale (its colour bar, sampled): driest-third frequency in browns, wettest-third in greens, white under 40%
FREQ_BOUNDS = [-100] + [-b + 1e-9 for b in (80, 70, 60, 50, 40)] + [40, 50, 60, 70, 80, 100]   # bins hold their edge on both sides
FREQ_CMAP = mcolors.ListedColormap(["#84540c", "#b17a28", "#d5a45f", "#eacfa0", "#f6ebcb", "#ffffff",
                                    "#ceecea", "#9bdad1", "#6ab2af", "#2c8d86", "#08615d"])
FREQ_NORM = mcolors.BoundaryNorm(FREQ_BOUNDS, FREQ_CMAP.N)
NORMAL_BOUNDS = [0, 20, 50, 100, 150, 200, 300, 450]
NORMAL_CMAP = mcolors.ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.08, 0.95, len(NORMAL_BOUNDS) - 1)))
NORMAL_NORM = mcolors.BoundaryNorm(NORMAL_BOUNDS, NORMAL_CMAP.N)


def fig_maps(ph: pd.DataFrame) -> tuple[Path, dict]:
    """Top row: normal rain per period (the seasonality); below: how often El Niño and La Niña years fell in the driest or
    wettest third, on FEWS NET's scale, so the figure reads beside FEWS NET's maps."""
    st = states()
    rows = [("El Niño", phase_years(ph, "nino")), ("La Niña", phase_years(ph, "nina"))]
    fig, axes = plt.subplots(len(rows) + 1, len(MAP_PERIODS), figsize=(11, 6.7))
    for j, (per, da) in enumerate(map_normal().items()):
        ax = axes[0, j]
        ax.pcolormesh(da.lon, da.lat, np.where(_clip(da, st.geometry), da.values.clip(max=NORMAL_BOUNDS[-1] - 1), np.nan),
                      cmap=NORMAL_CMAP, norm=NORMAL_NORM, shading="nearest")
        _map_axes(ax, st)
        mm = MAP_PERIODS[per]
        ax.set_title(_cap(_span(mm[0], mm[-1])), fontsize=9.5)
    axes[0, 0].text(-0.04, 0.5, "Normal\n1991–2020", transform=axes[0, 0].transAxes, ha="right", va="center",
                    fontsize=9.5, fontweight="bold")
    share50 = {}
    for i, (lab, yrs) in enumerate(rows, start=1):
        comp = map_composite(yrs)
        for j, (per, (da, dry_cells)) in enumerate(comp.items()):
            ax = axes[i, j]
            inside = _clip(da, st.geometry)
            v = np.where(inside, da.values, np.nan)
            dry = (inside & dry_cells).astype(float)
            if i == 1:   # El Niño: share of the (not dry-season) country in the driest third at least half the time
                share50[per] = float(np.nanmean(np.where(np.isnan(v[inside]), np.nan, v[inside] <= -50)) * 100)
            ax.pcolormesh(da.lon, da.lat, v, cmap=FREQ_CMAP, norm=FREQ_NORM, shading="nearest")
            if dry.any():
                ax.pcolor(_edges(da.lon.values), _edges(da.lat.values), np.ma.masked_where(dry == 0, dry),
                          cmap=mcolors.ListedColormap(["#ffffff"]), hatch="////", lw=0)
            _map_axes(ax, st)
            if j == 0:
                ax.text(-0.04, 0.5, f"{lab}\n{len(yrs)} {_t('years', 'años')}", transform=ax.transAxes, ha="right",
                        va="center", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.07, wspace=0.03, hspace=0.06)
    top, mid, low = (axes[k, -1].get_position() for k in range(3))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORMAL_NORM, cmap=NORMAL_CMAP), orientation="vertical",
                      cax=fig.add_axes([0.875, top.y0 + 0.01, 0.013, top.height - 0.02]))
    cb.set_ticks(NORMAL_BOUNDS[1:-1])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(_t("rain, mm a month", "lluvia, mm al mes"), fontsize=8)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=FREQ_NORM, cmap=FREQ_CMAP), orientation="vertical",
                      cax=fig.add_axes([0.875, low.y0 + 0.01, 0.013, mid.y1 - low.y0 - 0.02]))
    ticks = [-80, -70, -60, -50, -40, 40, 50, 60, 70, 80]
    cb.set_ticks(ticks, labels=[f"{abs(t)}%" for t in ticks])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(_t("% of the years in the driest third (brown) or wettest third (green)",
                    "% de los años en el tercio más seco (marrón) o más húmedo (verde)"), fontsize=8)
    DRY_PATCH.set_label(_t("dry season (under 20 mm a month)", "estación seca (menos de 20 mm al mes)"))
    fig.legend(handles=[DRY_PATCH], loc="upper right", bbox_to_anchor=(0.99, 0.065), fontsize=8, frameon=False)
    _source(fig, _t("El Niño and La Niña rows: how often the season fell in the driest or wettest third of 1991–2020; white: "
                    "neither reaches 40% (a third would be chance), the scale FEWS NET uses. Jun–Aug, Sep–Nov and Dec–Feb from the year "
                    "the event develops (Dec–Feb runs into the next year), Mar–May from the year after (phase from the Sep–Nov ONI, "
                    "1981–2025). Source: CHIRPS v3 at 0.5°.",
                    "Filas de El Niño y La Niña: con qué frecuencia la temporada cayó en el tercio más seco o más húmedo de 1991–2020; "
                    "blanco: ninguno llega al 40% (por azar sería un tercio), la escala de FEWS NET. Jun–ago, sep–nov y dic–feb del año "
                    "en que se desarrolla el evento (dic–feb se extiende al año siguiente), mar–may del año siguiente (fase según el ONI "
                    "de septiembre–noviembre, 1981–2025). Fuente: CHIRPS v3 a 0,5°."), y=0.045, width=140)
    jn = (share50["Jun–Aug"] + share50["Sep–Nov"]) / 2
    if (share50["Dec–Feb"] < 60 or not 15 <= jn <= 45 or share50["Mar–May"] > 6
            or max(share50["Jun–Aug"], share50["Sep–Nov"]) >= share50["Dec–Feb"] or abs(share50["Jun–Aug"] - share50["Sep–Nov"]) > 10):
        raise ValueError(f"the El Niño driest-third pattern changed ({ {k: round(x) for k, x in share50.items()} }): reword the maps headline")
    return _save(fig, "maps"), dict(djf50=f"{share50['Dec–Feb']:.0f}", jn50=f"{jn:.0f}")


def _grid_panel(ax, med: pd.DataFrame, labels: list[str] | None, title: str):
    """Rows × three-month windows of median % change, numbers in the cells; dry-season windows hatched."""
    ax.imshow(med.values, cmap=PCT_CMAP, norm=PCT_NORM, aspect="auto")
    _hatch_cells(ax, med.isna().values)
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            v = med.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{0.0 if abs(v) < 0.5 else v:+.0f}".replace("-", "−"), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) >= 25 else "#0b0b0b")
    ax.set_xticks(range(len(SEASONS)), [_cap(_span(SEASON_START[s], SEASON_START[s] + 2)) for s in SEASONS], rotation=90, fontsize=7.5)
    if labels:
        ax.set_yticks(range(med.shape[0]), labels, fontsize=8)
    else:   # shared y axis: hide this panel's labels without clearing the other's
        ax.tick_params(labelleft=False)
    ax.tick_params(length=0)
    for i in range(1, med.shape[0]):
        ax.axhline(i - 0.5, color=SURFACE, lw=1.5)
    ax.axvline(2.5, color=NOW_C, lw=0.8, ls=(0, (2, 2)))
    ax.set_title(title, fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)


def season_contrast(mon: pd.DataFrame, ph: pd.DataFrame) -> dict:
    """The maps' headline, by region: El Niño's median change in December–February (dry season) and in the June–November
    rains (% and mm against 1991–2020), leaving out regions where December–February is nearly rainless (normal under DRY_MM,
    as on the maps); that March–May shows no consistent change; and that Apure, the survey map's wetter patch, is normal or
    drier from June to January."""
    nino = phase_years(ph, "nino")
    reg = pd.DataFrame({r: mon[ps].mean(axis=1) for r, (ps, _) in REGIONS.items()})
    out = {}
    for k, months in (("djf", [12, 13, 14]), ("jn", list(range(6, 12))), ("mam", [15, 16, 17])):
        t = period_totals(reg, months)
        clim, sel = t.loc[1991:2020].mean(), t.loc[[y for y in nino if y in t.index]]
        out[k] = ((sel / clim - 1) * 100).median(), (sel - clim).median(), clim
    (djf, djf_mm, djf_clim), (jn, jn_mm, _), (mam, _, _) = out["djf"], out["jn"], out["mam"]
    rainless = sorted(djf_clim.index[djf_clim < DRY_MM])
    djf, djf_mm = djf.drop(rainless), djf_mm.drop(rainless)
    apure = []
    for m0 in range(6, 12):   # three-month windows starting June–November, the year El Niño develops
        t = period_totals(mon[["VE04"]], [m0, m0 + 1, m0 + 2])["VE04"]
        apure.append(float((t.loc[[y for y in nino if y in t.index]] / t.loc[1991:2020].mean() - 1).median() * 100))
    most = {"North-east coast", "Andes", "Central and eastern Llanos"}
    if ((djf > -20).any() or (jn > 0).any() or jn.min() <= djf.max() + 10 or not (mam.min() < 0 < mam.max())
            or mam.abs().max() > 10 or set(jn.nsmallest(3).index) != most or set(jn_mm.nsmallest(3).index) != most
            or rainless != ["Western Llanos"] or max(apure) > 10):
        raise ValueError(f"the season contrast changed (DJF {djf.round().tolist()}, Jun–Nov {jn.round().tolist()}, "
                         f"MAM {mam.round().tolist()}, rainless DJF {rainless}, Apure Jun–Jan {[round(a) for a in apure]}): "
                         "reword the rain-maps headline and note")
    f = lambda x: f"{x:.0f}"  # noqa: E731
    return dict(djf_lo=f(-djf.max()), djf_hi=f(-djf.min()), djf_mm_lo=f(-djf_mm.max()), djf_mm_hi=f(-djf_mm.min()),
                jn_lo=f(-jn.max()), jn_hi=f(-jn.min()), jn_mm_lo=f(-jn_mm.max()), jn_mm_hi=f(-jn_mm.min()))


def zone_seasons(zm: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Per zone and rain year: rainy-season (onset month to November) and dry-season (December to the onset, into the next
    calendar year) rain as % of the 1991–2020 normal, with the normals in mm."""
    out = {}
    for season in ("rainy", "dry"):
        pct, norm = {}, {}
        for z, m0 in ZONE_ONSET.items():
            t = period_totals(zm[[z]], list(range(m0, 12)) if season == "rainy" else list(range(12, m0 + 12)))[z]
            norm[z] = t.loc[1991:2020].mean()
            pct[z] = t / norm[z] * 100
        out[season] = (pd.DataFrame(pct), pd.Series(norm))
    return out


def zone_values(zs: dict, zm: pd.DataFrame, ph: pd.DataFrame) -> dict:
    """The zones headline: El Niño cuts the dry season in the south (A, B) and the rainy season in the north (C–E)."""
    nino = phase_years(ph, "nino")
    (rain_pct, _), (dry_pct, dry_norm) = zs["rainy"], zs["dry"]
    rm, dm = (rain_pct.reindex(nino) - 100).median(), (dry_pct.reindex(nino) - 100).median()
    below = (dry_pct.reindex(nino) < 100).sum()
    south, north = ["A", "B"], ["C", "D", "E"]
    clim = zm.loc["1991":"2020"].groupby(zm.loc["1991":"2020"].index.month).mean()
    onset = [int(clim.index[(clim[z] >= clim[z].mean()).values][0]) for z in ZONE_ONSET]
    lat = zones().set_index("zone").geometry.to_crs(3857).centroid.y.loc[list(ZONE_ONSET)]
    a_dry_mm = dry_norm["A"] / ZONE_ONSET["A"]
    if ((dm[south] > -15).any() or below[south].min() < 12 or (rm[south].abs() > 5).any() or (rm[north] > -5).any()
            or not ((dm[north] < 0) & (dm[north] > dm[south].max())).all() or onset != sorted(onset)
            or not lat.is_monotonic_increasing or a_dry_mm < DRY_MM):
        raise ValueError(f"the zone pattern changed (rainy {rm.round().tolist()}, dry {dm.round().tolist()}, below {below.tolist()}, "
                         f"onset order {onset}): reword the zones headline, note and tile")
    f = lambda x: f"{x:.0f}"  # noqa: E731
    return dict(ab_dry_lo=f(-dm[south].max()), ab_dry_hi=f(-dm[south].min()), ab_dry_below=int(below[south].min()),
                ab_rain_max=f(np.ceil(rm[south].abs().max())), cde_rain_lo=f(-rm[north].max()), cde_rain_hi=f(-rm[north].min()),
                cde_dry_lo=f(-dm[north].max()), cde_dry_hi=f(-dm[north].min()), a_dry_mm=f(a_dry_mm),
                a_dry_span=_span(12, 11 + ZONE_ONSET["A"]))


def _zone_label(z: str) -> str:
    return f"{z} · {_t(*ZONE_ONSET_TXT[z])}"


def fig_zones(zs: dict, ph: pd.DataFrame) -> Path:
    """Left: the INAMEH zones; middle and right: each zone's rainy and dry season, one dot per rain year, by ENSO phase."""
    z, st = zones(), states()
    fig = plt.figure(figsize=(14.5, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 0.38, 1, 1], wspace=0.08)   # the second column leaves room for zone labels
    ax, r1, r2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    z.plot(ax=ax, color=z.zone.map(ZONE_FILL), edgecolor="none")
    _map_axes(ax, st)
    for _, row in z.iterrows():
        pt = max(getattr(row.geometry, "geoms", [row.geometry]), key=lambda q: q.area).representative_point()
        ax.text(pt.x, pt.y, row.zone, ha="center", va="center", fontsize=12, fontweight="bold", color="#0b0b0b")
    ax.set_title(_t("INAMEH rainy-season zones", "Zonas de temporada de lluvias del INAMEH"), fontsize=9.5)
    labels = {k: _zone_label(k) for k in ZONE_ONSET}
    for axx, season, title in ((r1, "rainy", _t("Rainy season\n(onset month to November)", "Temporada de lluvias\n(del inicio a noviembre)")),
                               (r2, "dry", _t("Dry season\n(December to onset; zone A: less rainy)",
                                              "Estación seca\n(de diciembre al inicio; zona A: menos lluviosa)"))):
        _strip(axx, zs[season][0].loc[Y0:Y1].rename(columns=labels), ph, now=False)
        axx.set_title(title, fontsize=9.5)
    r2.tick_params(labelleft=False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in (
        (STRONG_C, _t("strong El Niño", "El Niño fuerte")), (NINO_C, "El Niño"), (NEUTRAL_C, _phase_name("neutral")), (NINA_C, "La Niña"))],
        loc="lower center", ncols=4, fontsize=8, bbox_to_anchor=(0.66, 0.93), frameon=False)
    fig.subplots_adjust(top=0.8)
    _source(fig, _t(f"Each dot is one rain year, 1981/82–{year_label(Y1)}: the zone's rain in its rainy and dry season, as % of the "
                    "1991–2020 normal. Zones digitized from INAMEH's map of rainy-season onset onto the 0.5° grid (edges approximate); "
                    "the rainy season runs from the onset month to November, as FEWS NET. Sources: CHIRPS v3; zones: INAMEH.",
                    f"Cada punto es un año hidrológico, 1981/82–{year_label(Y1)}: la lluvia de la zona en su temporada de lluvias y en "
                    "su estación seca, en % de la normal 1991–2020. Zonas digitalizadas del mapa del INAMEH de inicio de la temporada "
                    "de lluvias en la malla de 0,5° (bordes aproximados); la temporada de lluvias va del mes de inicio a noviembre, "
                    "como en FEWS NET. Fuentes: CHIRPS v3; zonas: INAMEH."), y=0.0)
    return _save(fig, "zones")


def fig_fewsnet() -> Path:
    """FEWS NET's maps of below/above-normal rain frequency in strong El Niño events, cropped to northern South America."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (k, m0, m1) in zip(axes, (("jja", 6, 8), ("son", 9, 11), ("djf", 12, 14))):
        ax.imshow(plt.imread(FEWS_DIR / f"elnino_precip_{k}.png"))
        ax.set_title(_cap(_span(m0, m1)), fontsize=9.5)
        ax.set_axis_off()
    cax = fig.add_axes([0.28, 0.0, 0.44, 0.13])
    cax.imshow(plt.imread(FEWS_DIR / "elnino_precip_colorbar.png"))
    cax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.16, wspace=0.03)
    _source(fig, _t("FEWS NET: how often rain was below (brown) or above (green) normal in strong and very strong El Niño events, "
                    "1950–2025; a frequency, not an amount. From the FEWS NET El Niño Western Hemisphere briefing, August 2026 "
                    "(pages 4–6), cropped to northern South America.",
                    "FEWS NET: con qué frecuencia la lluvia estuvo por debajo (marrón) o por encima (verde) de lo normal en los eventos "
                    "de El Niño fuertes y muy fuertes, 1950–2025; una frecuencia, no una cantidad. Del informe de FEWS NET sobre El "
                    "Niño en el hemisferio occidental, agosto de 2026 (páginas 4–6, en inglés), recortado al norte de Sudamérica."),
            y=-0.02)
    return _save(fig, "fewsnet")


CALENDAR_FILE = FEWS_DIR / "ve_seasonal_calendar_rural.csv"   # digitized by deep_dives/data/fewsnet/ve_seasonal_calendar_digitize.py
CAL_FILL = {"hurricane": "#1d3a5f", "prep": "#4f352e", "harvest": "#0d5d40", "labour": "#434343"}   # FEWS NET's colours
CAL_RAINY, CAL_LEAN = (4, 11), (5, 8)   # months from 1 January: FEWS NET's rainy season May–Nov, lean season Jun–Aug


def fig_calendar(k: float = 1.0) -> Path:
    """FEWS NET's seasonal calendar for rural Venezuela, redrawn from the digitized table so it can be labelled in Spanish.
    k > 1: type k times larger and no caption, for a slide (ven_slides.py), saved as calendar_slide."""
    cal = pd.read_csv(CALENDAR_FILE, keep_default_na=False)
    rows = cal.drop_duplicates("order").set_index("order")[_t("row_en", "row_es")]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    fig.subplots_adjust(left=0.24, right=0.99, top=0.88, bottom=0.1)   # before the labels, which are sized to the month width
    ax.axvspan(*CAL_RAINY, color="#bddce8", lw=0, zorder=0)
    ax.text(sum(CAL_RAINY) / 2, -1.05, _t("Rainy season", "Temporada de lluvias"), ha="center", va="center", fontsize=10.5 * k,
            style="italic", color="#2b4c8c")
    for r in cal.itertuples():
        ax.barh(r.order, r.end - r.start, left=r.start, height=0.5, color=CAL_FILL[r.kind], edgecolor=SURFACE, lw=1.5, zorder=2)
        label = r.text_es if LANG == "es" else r.text_en
        if label:   # shrink a label to fit its bar: bold characters are up to about 0.75 of the font size wide
            month_pt = fig.get_figwidth() * 72 * (0.99 - 0.24) / 12
            size = min(8.5 * k, 0.9 * (r.end - r.start) * month_pt / (0.75 * len(label)))
            ax.text((r.start + r.end) / 2, r.order, label, ha="center", va="center", fontsize=size, color="#ffffff",
                    fontweight="bold", zorder=3)
    for y in np.arange(len(rows) - 1) + 0.5:
        ax.axhline(y, color="#9a9993", lw=0.5, ls=(0, (1, 2)), zorder=1)
    lean = plt.Rectangle((CAL_LEAN[0], -0.62), CAL_LEAN[1] - CAL_LEAN[0], len(rows) + 0.3, fill=False, edgecolor="#e8383d", lw=2,
                         zorder=4, clip_on=False)
    ax.add_patch(lean)
    ax.text(sum(CAL_LEAN) / 2, len(rows) - 0.05, _t("Lean season", "Temporada de escasez"), ha="center", va="top", fontsize=10.5 * k,
            fontweight="bold", color="#e8383d")
    ax.set_xlim(0, 12)
    ax.set_ylim(len(rows) + 0.75, -1.45)
    ax.set_yticks(rows.index, rows.values, fontsize=9.5 * k)
    ax.set_xticks(np.arange(12) + 0.5, [_mon(m).upper() for m in range(1, 13)], fontsize=9.5 * k)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.set_title(_t("Rural areas", "Zonas rurales"), fontsize=11 * k, fontweight="bold", pad=24 * k)
    if k > 1:
        return _save(fig, "calendar_slide")
    if cal.loc[cal.wraps.astype(str) == "True", "row_en"].unique().tolist() != ["Cash crop harvest"]:
        raise ValueError("the calendar's year-end crossing changed: reword the caption")
    _source(fig, _t("Redrawn from FEWS NET’s seasonal calendar for Venezuela (rural areas), fews.net/latin-america-and-caribbean/"
                    "venezuela. Winter and summer are Venezuela’s two crop cycles: winter crops are planted with the rains (April–"
                    "June), summer crops from about October. The coffee and sugar-cane harvest runs from October to March, across the "
                    "year end; the other bars stop where they end.",
                    "Redibujado del calendario estacional de FEWS NET para Venezuela (zonas rurales), fews.net/latin-america-and-"
                    "caribbean/venezuela. Invierno y verano son los dos ciclos de cultivo de Venezuela: los de invierno se siembran con "
                    "las lluvias (abril–junio), los de verano desde octubre aproximadamente. La cosecha de café y caña de azúcar va de "
                    "octubre a marzo, de un año al siguiente; las demás barras terminan donde acaban."), y=0.0, width=150)
    return _save(fig, "calendar")


def fig_states(totz: dict, ph: pd.DataFrame) -> tuple[Path, dict]:
    """By INAMEH zone and three-month window: median change in the El Niño and the La Niña years."""
    nino, nina = phase_years(ph, "nino"), phase_years(ph, "nina")
    med_nino, med_nina = composite(totz, nino)[0].loc[list(ZONE_ONSET)], composite(totz, nina)[0].loc[list(ZONE_ONSET)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.4), sharey=True)
    _grid_panel(axes[0], med_nino, [_zone_label(z) for z in ZONE_ONSET], f"El Niño ({len(nino)} {_t('years', 'años')})")
    _grid_panel(axes[1], med_nina, None, f"La Niña ({len(nina)} {_t('years', 'años')})")
    fig.subplots_adjust(wspace=0.04)
    _source(fig, _t("Each cell: median rain over the El Niño (or La Niña) years, % above or below the zone's 1991–2020 normal for "
                    "those three months. Left of the dashed line: windows starting January–March, from the year after El Niño or "
                    "La Niña develops; right: windows starting May–December, from the year it develops. Rain years May–April, "
                    "1981/82–2025/26. Sources: CHIRPS v3; zones: INAMEH.",
                    "Cada celda: mediana de la lluvia en los años de El Niño (o de La Niña), en % por encima o por debajo de la normal "
                    "1991–2020 de la zona para esos tres meses. A la izquierda de la línea discontinua: los períodos que empiezan entre "
                    "enero y marzo, del año siguiente al desarrollo de El Niño o La Niña; a la derecha: los que empiezan entre mayo y "
                    "diciembre, del año en que se desarrolla. Años hidrológicos de mayo a abril, 1981/82–2025/26. Fuentes: CHIRPS v3; "
                    "zonas: INAMEH."), y=-0.12)
    core = med_nino[["NDJ", "DJF", "JFM"]]
    rains = med_nino[["JJA", "JAS", "ASO", "SON"]]
    if (core.isna().any().any() or (core > -15).any().any() or (rains.loc["E"] > -12).any()
            or rains.loc[["A", "B", "C", "D"]].abs().max().max() >= rains.loc["E"].abs().min()
            or (med_nino.loc[["C", "D", "E"], ["FMA", "MAM"]].abs() >= 10).any().any()):
        raise ValueError(f"the zone × season pattern changed:\n{med_nino.round()}\nreword the grid headline and note")
    f = lambda x: f"{x:.0f}"  # noqa: E731
    return _save(fig, "states"), dict(core_lo=f(-core.max().max()), core_hi=f(-core.min().min()),
                                      e_rain_lo=f(-rains.loc["E"].max()), e_rain_hi=f(-rains.loc["E"].min()))


def fig_season(zm: pd.DataFrame, ph: pd.DataFrame) -> Path:
    """Normal monthly rain by INAMEH zone, January to December, each zone's rainy season shaded, with the median of the
    El Niño and La Niña years (January–April from the year after the event develops) and this year so far on top."""
    months = [13, 14, 15, 16] + list(range(5, 13))
    clim = zm.loc["1991":"2020"].groupby(zm.loc["1991":"2020"].index.month).mean().loc[[(m - 1) % 12 + 1 for m in months]]

    def phase_median(phase: str) -> tuple[pd.DataFrame, int]:
        yrs = [y for y in phase_years(ph, phase) if all(d in zm.index for d in _dates(y, months))]
        return pd.DataFrame(np.median([zm.loc[_dates(y, months)].values for y in yrs], axis=0), columns=zm.columns), len(yrs)

    nino, n_nino = phase_median("nino")
    nina, n_nina = phase_median("nina")
    cur = zm.reindex([pd.Timestamp(NOW, m, 1) for m in range(1, 13)])
    cur_last = cur.dropna(how="all").index.max()
    fig, axes = plt.subplots(1, len(ZONE_ONSET), figsize=(11, 2.8), sharey=True)
    x = np.arange(len(months))
    for ax, z in zip(axes, ZONE_ONSET):
        ax.axvspan(ZONE_ONSET[z] - 1.5, 10.5, color=RAINY_C, lw=0, zorder=0)
        ax.bar(x, clim[z].values, color=NEUTRAL_C, width=0.8, zorder=1)
        ax.plot(x, nino[z].values, color=NINO_C, lw=1.4, marker="o", ms=2.5, zorder=3)
        ax.plot(x, nina[z].values, color=NINA_C, lw=1.4, marker="o", ms=2.5, zorder=3)
        ax.plot(x, cur[z].values, color=NOW_C, lw=1.4, marker="o", ms=2.5, zorder=4)
        ax.set_title(_zone_label(z), fontsize=8.5)
        ax.set_xticks(x, [_month_letters()[(m - 1) % 12] for m in months], fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        _grid(ax)
    axes[0].set_ylabel(_t("rain, mm a month", "lluvia, mm al mes"), fontsize=8)
    fig.legend(handles=[Patch(color=NEUTRAL_C, label="normal (1991–2020)"),
                        Patch(color=RAINY_C, label=_t("rainy season", "temporada de lluvias")),
                        Line2D([], [], color=NINO_C, marker="o", ms=3,
                               label=_t(f"El Niño years (median of {n_nino})", f"años de El Niño (mediana de {n_nino})")),
                        Line2D([], [], color=NINA_C, marker="o", ms=3,
                               label=_t(f"La Niña years (median of {n_nina})", f"años de La Niña (mediana de {n_nina})")),
                        Line2D([], [], color=NOW_C, marker="o", ms=3,
                               label=_t(f"{NOW} (to {cur_last:%B})", f"{NOW} (hasta {MONTH_ES[cur_last.month - 1]})"))],
               loc="lower center", ncols=5, fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2, wspace=0.12)
    _source(fig, _t("INAMEH rainy-season zones, titled by the onset fortnight; shaded: rainy season, from the onset month to November. "
                    "El Niño and La Niña: January–April from the year after the event develops, May–December from the year it "
                    "develops; events 1981–2025. 2026 is the calendar year so far: its January–April come before this El Niño. "
                    "Sources: CHIRPS v3; zones: INAMEH.",
                    "Zonas de temporada de lluvias del INAMEH, tituladas por la quincena de inicio; sombreado: temporada de lluvias, "
                    "del mes de inicio a noviembre. El Niño y La Niña: enero–abril del año siguiente al desarrollo del evento, "
                    "mayo–diciembre del año en que se desarrolla; eventos de 1981–2025. 2026 es el año calendario hasta ahora: su "
                    "enero–abril es anterior a este El Niño. Fuentes: CHIRPS v3; zonas: INAMEH."), y=-0.09)
    return _save(fig, "season")


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
    ax.set_yticks(range(len(regs)), [_reg(r) for r in regs])
    ax.set_ylim(len(regs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    _grid(ax, "x")
    ax.set_xlabel(_t("rain, % of the 1991–2020 normal", "lluvia, % de la normal 1991–2020"))


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
    span = _t(f"May–{last:%B}", f"mayo–{MONTH_ES[last.month - 1]}")
    _source(fig, _t(f"{span} rain, each year 1981–{NOW}, one dot per year. 2026 was among the {_words(int(rank.max()))} driest of "
                    f"{len(pct)} in every region in CHIRPS v3 and among the {_words(int(rk_e.max()))} driest in ERA5. October data come "
                    "out in November. Sources: CHIRPS v3 (state means), "
                    "ERA5 (team admin-1 statistics).",
                    f"Lluvia de {span}, cada año de 1981 a {NOW}, un punto por año. 2026 estuvo entre los {_words(int(rank.max()))} más "
                    f"secos de {len(pct)} en todas las regiones según CHIRPS v3 y entre los {_words(int(rk_e.max()))} más secos según "
                    "ERA5. Los datos de octubre salen en noviembre. Fuentes: CHIRPS v3 "
                    "(medias por estado), ERA5 (estadísticas admin-1 del equipo)."))
    v = dict(now_span=span, now_lo=f"{pct.loc[NOW].min():.0f}", now_hi=f"{pct.loc[NOW].max():.0f}", now_rank_txt=_words(int(rank.max())),
             now_rank_max=int(rank.max()), now_n=len(pct), era5_rank_max=int(rk_e.max()),
             era5_lo=f"{pe.loc[NOW].min():.0f}", era5_hi=f"{pe.loc[NOW].max():.0f}")
    if not 2 <= v["now_rank_max"] <= 6 or v["era5_rank_max"] > 7:   # rank 1 everywhere needs "the driest", not "among the"
        raise ValueError(f"2026 ranks changed ({v['now_rank_max']}, {v['era5_rank_max']}): reword the fig_now note")
    return _save(fig, "now"), v


EM_LABELS = {  # DisNo.: (English label, Spanish label, x offset, y offset in points)
    "1999-0547-VEN": ("Dec 1999, Vargas: 30,000 dead", "dic 1999, Vargas: 30.000 muertos", 10, 0),
    "2021-0538-VEN": ("Aug 2021, Mérida and 10+ states", "ago 2021, Mérida y más de 10 estados", -8, 0),
    "2022-0648-VEN": ("Oct 2022, Las Tejerías", "oct 2022, Las Tejerías", -8, 6),
    "2015-0296-VEN": ("Jun 2015, Apure", "jun 2015, Apure", 6, -2),
    "2005-0073-VEN": ("Feb 2005: 76 dead", "feb 2005: 76 muertos", 6, 0),
    "2010-0616-VEN": ("Nov–Dec 2010", "nov–dic 2010", -8, 4),
    "1987-0156-VEN": ("Sep 1987, Aragua", "sep 1987, Aragua", 6, 0),
    "2002-0445-VEN": ("Jul 2002, Llanos", "jul 2002, Llanos", 0, 10),
    "1993-0059-VEN": ("Aug 1993, storm Bret", "ago 1993, tormenta Bret", 6, 0)}
CERF_EMDAT = {  # CERF flood allocation: the EM-DAT event it responded to, and its label in English and Spanish
    "CERF-VEN-25-RR-1490": ("2025-0541-VEN", "the May–June 2025 floods in Apure and 9 more states",
                            "las inundaciones de mayo–junio de 2025 en Apure y otros 9 estados")}
CERF_KNOWN = {  # every Venezuela allocation the page's CERF lines were written against (one flood, none for drought)
    "18-RR-VEN-33275", "19-UF-VEN-38565", "21-UF-VEN-48704", "22-UF-VEN-55398", "23-UF-VEN-61207", "CERF-VEN-25-RR-1490",
    "CERF-VEN-25-UF-1452", "CERF-VEN-26-RR-1539", "CERF-VEN-26-RR-1555"}


def fig_emdat(ph: pd.DataFrame) -> tuple[Path, dict]:
    em = emdat_wet()
    em = em[em.ry.between(1980, Y1)].copy()   # a later event would join a phase total with no year in the denominator
    snap = str(emdat()["Last Update"].max())
    em["phase"] = ph.phase.reindex(em.ry).values
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[3.2, 1]))
    _grid(ax)
    for _, r in em.iterrows():
        y = max(r.affected, 50)
        ax.scatter(r.date, y, s=18 + np.sqrt(r.deaths) * 3, color=phase_color(int(r.ry), ph), edgecolor=SURFACE, lw=0.6,
                   zorder=3, marker="o" if r.affected > 0 else "v")
        if r["DisNo."] in EM_LABELS:
            en, es, dx, dy = EM_LABELS[r["DisNo."]]
            ax.annotate(_t(en, es), (r.date, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, va="center",
                        ha="right" if dx < 0 else "center" if dx == 0 else "left")
    c = cerf()
    cerf_txt = []
    for _, a in c[c.emergency_type.isin(["Flood", "Storm"])].iterrows():
        dis, en, es = CERF_EMDAT[a.application_code]
        r = em[em["DisNo."] == dis].iloc[0]
        mn = f"{a.amount_approved / 1e6:.0f}"
        ax.scatter(r.date, max(r.affected, 50), s=130, facecolor="none", edgecolor=NOW_C, lw=1.1, zorder=4)
        ax.annotate(_t(f"CERF\nUS${mn}M", f"CERF\nUS$ {mn} M"), (r.date, max(r.affected, 50)), xytext=(9, 0),
                    textcoords="offset points", fontsize=7, ha="left", va="center")
        ax.set_xlim(right=pd.Timestamp(f"{r.date.year + 4}-01-01"))
        when = _month_year(pd.Timestamp(a.erc_endorsement_date))
        cerf_txt.append(_t(f"{en}, for which CERF allocated US${mn} million in {when}",
                           f"{es}, para las que el CERF asignó {mn} millones de US$ en {when}"))
    ax.set_yscale("log")
    ax.set_ylim(30, 4e6)
    ax.set_ylabel(_t("people affected (EM-DAT)", "personas afectadas (EM-DAT)"))
    _phase_legend(ax, ph, loc="upper left", ncols=2)
    t = em.groupby("phase")[["affected", "deaths"]].sum().reindex(["nino", "neutral", "nina"]).fillna(0)
    years = ph.loc[1980:Y1, "phase"].value_counts()
    per = t.div(years, axis=0)
    for i, (k, c) in enumerate((("nino", NINO_C), ("neutral", NEUTRAL_C), ("nina", NINA_C))):
        w = float(per.loc[k, "affected"]) / 1000
        bx.barh(i, w, color=c, height=0.6)
        bx.text(w, i, _t(f" {w:,.0f}k", f" {w:,.0f} mil"), va="center", fontsize=8)
    bx.set_yticks([0, 1, 2], [f"{_phase_name(p)}\n({years[p]} {_t('yrs', 'años')})" for p in ("nino", "neutral", "nina")])
    bx.invert_yaxis()
    bx.set_xlabel(_t("people affected per year\n(thousands)", "personas afectadas por año\n(miles)"))
    bx.set_xlim(0, per["affected"].max() / 1000 * 1.45)
    _source(fig, _t("Floods, wet landslides and storms reported to EM-DAT since 1980, by the rain year (May–April) they fall in. "
                    f"Marker size: deaths; triangles: no affected figure. Ring: {'; '.join(cerf_txt)}. Totals per year of each phase, rain "
                    f"years 1980/81–{year_label(Y1)}; the EM-DAT snapshot ends {snap}, so {year_label(Y1)} is partial. "
                    "Sources: EM-DAT, CRED/UCLouvain (team snapshot); CERF.",
                    "Inundaciones, deslizamientos por lluvia y tormentas registrados en EM-DAT desde 1980, según el año hidrológico "
                    "(mayo–abril) en que ocurren. Tamaño del marcador: muertes; triángulos: sin cifra de afectados. Anillo: "
                    f"{'; '.join(cerf_txt)}. Totales por año de cada fase, años hidrológicos 1980/81–{year_label(Y1)}; la versión de "
                    f"EM-DAT termina el {snap}, así que {year_label(Y1)} está incompleto. Fuentes: EM-DAT, CRED/UCLouvain (versión "
                    "del equipo); CERF."))
    v = dict(em_nina_aff=f"{t.loc['nina', 'affected'] / 1e6:.1f}", em_nino_aff=f"{t.loc['nino', 'affected'] / 1e6:.1f}",
             em_nina_dead=f"{t.loc['nina', 'deaths']:,.0f}", em_nino_dead=f"{t.loc['nino', 'deaths']:,.0f}",
             em_ratio=f"{per.loc['nina', 'affected'] / per.loc['nino', 'affected']:.0f}",
             em_ratio_ex=f"{(t.loc['nina', 'affected'] - em.loc[em['DisNo.'].isin(['1999-0547-VEN', '2021-0538-VEN']), 'affected'].sum()) / years['nina'] / per.loc['nino', 'affected']:.1f}",
             em_n_nino=int((em.phase == "nino").sum()), em_n_nina=int((em.phase == "nina").sum()),
             em_nino_yr=f"{per.loc['nino', 'affected'] / 1000:,.0f},000", em_nina_yr=f"{per.loc['nina', 'affected'] / 1000:,.0f},000",
             em_neutral_yr=f"{per.loc['neutral', 'affected'] / 1000:,.0f},000")
    ratio_ex = (t.loc['nina', 'affected'] - em.loc[em['DisNo.'].isin(['1999-0547-VEN', '2021-0538-VEN']), 'affected'].sum()) \
        / years['nina'] / per.loc['nino', 'affected']
    if per.loc["nina", "affected"] / per.loc["nino", "affected"] < 3 or ratio_ex < 1.5:
        raise ValueError("the EM-DAT El Niño/La Niña contrast weakened: reword the floods headline and note")
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
    total = _t("Venezuela (all)", "Venezuela (total)")
    ax.set_yticks(range(len(rows)), [_reg(r) if r != "Venezuela" else total for r in rows])
    for lab in ax.get_yticklabels():
        if lab.get_text() == total:
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(_t("flood and landslide records, relative to an ordinary year for the period (1 = ordinary); median of the years",
                     "registros de inundaciones y deslizamientos frente a un año ordinario del período (1 = ordinario); mediana de los años"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f"{_phase_name(p)} ({len(groups[p])} {_t('years', 'años')})")
                       for p, c in (("nino", NINO_C), ("neutral", NEUTRAL_C), ("nina", NINA_C))],
              loc="lower left", fontsize=8, ncols=3, bbox_to_anchor=(0, 1.0))
    _source(fig, _t(f"Records of floods, heavy rain, torrents, debris flows, landslides and storms, one per event and municipality, "
                    f"rain years {year_label(DI_Y0)}–{year_label(DI_Y1)}. Each year's count is divided by the mean of the three rain "
                    "years either side, because reporting changes several-fold between decades. Source: DesInventar Venezuela "
                    "(UNDRR), which stops in early 2016.",
                    "Registros de inundaciones, lluvias, avenidas torrenciales, flujos de detritos, deslizamientos y tormentas, uno por "
                    f"evento y municipio, años hidrológicos {year_label(DI_Y0)}–{year_label(DI_Y1)}. El recuento de cada año se divide "
                    "por la media de los tres años hidrológicos a cada lado, porque la cantidad de registros se multiplica o se divide "
                    "varias veces de una década a otra. "
                    "Fuente: DesInventar Venezuela (UNDRR), que termina a principios de 2016."))
    regs = m.drop(index="Venezuela")
    above = regs.index[regs["nina"] > 1]
    v = dict(di_nino_lo=f"{regs['nino'].min():.2f}", di_nino_hi=f"{regs['nino'].max():.2f}",
             di_fewer_lo=f"{(1 - regs['nino'].max()) * 100:.0f}", di_fewer_hi=f"{(1 - regs['nino'].min()) * 100:.0f}",
             di_nina_above=len(above), di_regions=len(regs), di_mean_nina=f"{mean_nina:.1f}",
             di_llanos_nina=f"{m.loc['Western Llanos', 'nina']:.2f}", di_cn_nina=f"{m.loc['Centre-north', 'nina']:.2f}",
             di_n=f"{len(di_wet()):,}")
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
    for yr, lab in ((2009, _t("2010 rationing", "racionamiento\nde 2010")), (2015, _t("2016 rationing", "racionamiento de 2016"))):
        lo = gy.loc[yr, "low"]
        d = g[f"{yr + 1}-02-01":f"{yr + 1}-06-30"].idxmin()
        ax.annotate(lab, (d, lo), xytext=(-6, -2) if yr == 2009 else (6, -2), textcoords="offset points", fontsize=7.5,
                    va="top", ha="right" if yr == 2009 else "left")
    drops = gy.loc[gy.phase == "nino", "drop"].dropna()
    peak = float(g[f"{NOW}-08-01":].max())
    lo_hi = (peak - drops.max(), peak - drops.min())
    x = pd.Timestamp(NOW + 1, 5, 1)
    ax.plot([x, x], lo_hi, color=NOW_C, lw=3, solid_capstyle="butt")
    ax.annotate(_t(f"May {NOW + 1} if it falls\nlike past El Niño years", f"Mayo de {NOW + 1} si baja\ncomo tras otros El Niño"),
                (x, lo_hi[0]), xytext=(-8, -2), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, fontweight="bold")
    ax.set_ylabel(_t("Guri reservoir level (m, satellite)", "nivel del embalse de Guri (m, satélite)"))
    ax.set_xlim(pd.Timestamp(1992, 6, 1), pd.Timestamp(NOW + 1, 12, 31))
    ax.legend(handles=[Patch(color=NINO_C, alpha=0.25, label=_t("dry season after El Niño", "estación seca tras El Niño"))],
              loc="lower left", fontsize=8)
    # right: drawdown per rain year by phase
    _grid(bx)
    cats = [(p, _phase_name(p), c) for p, c in (("nina", NINA_C), ("neutral", NEUTRAL_C), ("nino", NINO_C))]
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
    bx.set_ylabel(_t(f"change, {_span(8, 11)} peak to {_span(2, 6)} low (m)",
                     f"cambio del máximo de {_span(8, 11)}\nal mínimo de {_span(2, 6)} (m)"))
    _source(fig, _t("Satellite altimetry, 10-day, heights above the EGM2008 geoid (not the operator's gauge datum: compare changes, "
                    "not levels). Black bar: this year's August–October peak minus the smallest and largest falls after past El Niño "
                    "years. Lines: medians. Source: NASA GSFC G-REALM (Guri, target 432).",
                    "Altimetría satelital cada 10 días, alturas sobre el geoide EGM2008 (no la cota de referencia del operador: comparar "
                    "cambios, no niveles). Barra negra: el máximo de agosto a octubre de este año menos la menor y la mayor bajada tras "
                    "años de El Niño anteriores. Líneas: medianas. Fuente: NASA GSFC G-REALM (Guri, objetivo 432)."))
    med = gy.groupby("phase")["drop"].median()
    v = dict(guri_nino=f"{med['nino']:.0f}", guri_nina=f"{med['nina']:.0f}", guri_peak=f"{peak:.1f}",
             guri_lo=f"{lo_hi[0]:.0f}", guri_hi=f"{lo_hi[1]:.0f}", guri_low16=f"{gy.loc[2015, 'low']:.0f}",
             guri_low10=f"{gy.loc[2009, 'low']:.0f}", guri_peak15=f"{gy.loc[2015, 'peak']:.0f}", guri_n_nino=len(drops),
             next_year=NOW + 1)
    v["guri_gap10"] = f"{lo_hi[0] - gy.loc[2009, 'low']:.0f}"
    if not 0 < lo_hi[0] - gy.loc[2009, "low"] < 3 or peak < g.max() - 3:
        raise ValueError("the Guri range moved against the 2010 low: reword the Guri note")
    return _save(fig, "guri"), v


def _enso_scatter(ax, x: pd.Series, y: pd.Series, ph: pd.DataFrame, label: dict[int, str], hollow: set[int] = frozenset(),
                  oni: bool = True):
    """One point per year: Sep–Nov ONI (or, with oni=False, another driver) against an outcome, coloured by phase;
    `hollow` years drawn open (left out of r)."""
    _grid(ax, "both")
    for t in y.dropna().index:
        if t not in x.index:
            continue
        c = phase_color(int(t), ph)
        ax.scatter(x[t], y[t], s=44 if c == STRONG_C else 26, zorder=3, lw=1,
                   color="none" if t in hollow else c, edgecolor=c if t in hollow else SURFACE)
        if t in label:
            ax.annotate(label[t], (x[t], y[t]), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#9a9993", lw=0.6)
    if not oni:
        ax.axvline(0, color="#9a9993", lw=0.6)
        return
    for v in (-0.5, 0.5):
        ax.axvline(v, color="#9a9993", lw=0.5, ls=(0, (3, 3)))
    ax.set_xlabel(f"ONI, {_span(9, 11)} (°C): La Niña ← → El Niño")


def _oni_now_line(ax):
    """Dashed vertical line at the latest ONI, so this year can be read off the scatter."""
    last = oni().iloc[-1]
    ax.axvline(last.anom, color=NOW_C, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"{NOW} {_t('so far', 'hasta ahora')}: {_num(f'{last.anom:+.1f}')} ({oni_now()['oni_seas']})", (last.anom, 1),
                xycoords=("data", "axes fraction"), xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=NOW_C)


def fig_veg(ph: pd.DataFrame) -> tuple[Path, dict]:
    crop, rng = asap_dry(1), asap_dry(2)
    nat = crop.mean()
    nat_r = rng.mean()
    yrs = [y for y in nat.index if y < NOW]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    worst = nat.loc[yrs].nsmallest(3)
    lab = {y: year_label(y) for y in list(worst.index) + [y for y in yrs if ph.loc[y, "strong"]]}
    _enso_scatter(ax, ph.oni, nat.loc[yrs], ph, lab)
    _oni_now_line(ax)
    ax.set_ylabel(_t(f"cropland vegetation (FPAR) z-score, {_span(11, 16)}", f"vegetación de cultivos (FPAR), puntaje z, {_span(11, 16)}"))
    _phase_legend(ax, ph, loc="lower left", ncols=4, bbox_to_anchor=(0, 1.0))
    st = states()
    nino = [y for y in phase_years(ph, "nino", 2000, NOW - 1)]
    sm = st.merge(crop[nino].mean(axis=1).rename("z"), left_on="pcode", right_index=True, how="left")
    zb = [-1, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 1]
    cm = mcolors.ListedColormap(DIV)
    sm.plot(ax=bx, column="z", cmap=cm, norm=mcolors.BoundaryNorm(zb, cm.N), edgecolor="#ffffff", lw=0.5,
            missing_kwds=dict(color=DRY_GREY))
    bx.set_axis_off()
    bx.set_title(_t(f"Nov–Apr after El Niño, cropland\n(mean of {len(nino)} rain years)",
                    f"Nov–abr tras El Niño, cultivos\n(media de {len(nino)} años hidrológicos)"), fontsize=9)
    cax = fig.add_axes([0.66, 0.1, 0.26, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(zb, cm.N), cmap=cm), cax=cax, orientation="horizontal")
    cb.set_ticks(zb[1:-1], labels=[_num(f"{z:g}") for z in zb[1:-1]])
    cb.ax.tick_params(labelsize=7)
    r = float(np.corrcoef(ph.oni.reindex(yrs), nat.loc[yrs])[0, 1])
    r_rng = float(np.corrcoef(ph.oni.reindex(yrs), nat_r.loc[yrs])[0, 1])
    rc, rr = _num(f"{r:+.2f}"), _num(f"{r_rng:+.2f}")
    _source(fig, _t(f"One point per rain year, 2000/01–{year_label(NOW - 1)}: cropland, mean of the state values. Z-score of FPAR "
                    "(the share of sunlight the canopy absorbs) against each 10-day period's 2001–present record, November–April. "
                    f"Correlation with the ONI: cropland {rc}, rangeland {rr}. Source: JRC ASAP (MODIS FPAR, GAUL admin-1).",
                    f"Un punto por año hidrológico, 2000/01–{year_label(NOW - 1)}: cultivos, media de los valores por estado. Puntaje z "
                    "de la FPAR (la fracción de la luz solar que absorbe la cubierta vegetal) frente al registro de cada período de 10 "
                    "días desde 2001, de noviembre a abril. Correlación con el ONI: cultivos "
                    f"{rc}, pastizales {rr}. Fuente: JRC ASAP (MODIS FPAR, GAUL admin-1)."))
    after = [y for y in worst.index if ph.phase.get(y) == "nino"]
    v = dict(veg_worst=", ".join(year_label(y) for y in sorted(worst.index)), veg_worst_nino=len(after),
             veg_2023=f"{nat.loc[2023]:+.1f}".replace("-", "−"), veg_r=f"{r:+.2f}".replace("-", "−"),
             veg_states_neg=int((crop[nino].mean(axis=1) < 0).sum()), veg_states_n=int(crop[nino].mean(axis=1).notna().sum()))
    if len(after) != 3 or nat.loc[2023] < -0.1 or r > -0.4:
        raise ValueError("the vegetation pattern changed: reword the vegetation headline")
    return _save(fig, "veg"), v


def fig_vegnow() -> tuple[Path, dict]:
    """The latest JRC ASAP dekad: FPAR z-score by state over cropland and over rangeland."""
    st = states()
    lookup = {_norm(n): pc for pc, n in zip(st.pcode, st.name)}
    names = st.set_index("pcode")["name"]
    zb = [-3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3]
    cm, nm = mcolors.ListedColormap(DIV), mcolors.BoundaryNorm(zb, len(DIV))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    cur = {}
    for ax, cid, title in zip(axes, (1, 2), (_t("Cropland", "Cultivos"), _t("Rangeland", "Pastizales"))):
        a = asap(220, cid, 2)
        last = a.date.max()
        z = a[a.date == last].assign(pcode=lambda d: d.region_name.map(lambda n: lookup.get(_norm(n)))).set_index("pcode")["value"]
        cur[cid] = (z, last)
        st.merge(z.rename("z"), left_on="pcode", right_index=True, how="left").plot(
            ax=ax, column="z", cmap=cm, norm=nm, edgecolor="#ffffff", lw=0.5, missing_kwds=dict(color=DRY_GREY))
        ax.set_xlim(-73.6, -59.6)
        ax.set_ylim(0.6, 12.4)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    cax = fig.add_axes([0.3, 0.17, 0.4, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal")
    cb.set_ticks(zb[1:-1], labels=[_num(f"{t:g}") for t in zb[1:-1]])
    cb.ax.tick_params(labelsize=8)
    cb.set_label(_t("vegetation (FPAR) z-score; ASAP's warning level is −1", "vegetación (FPAR), puntaje z; el umbral de alerta de ASAP es −1"),
                 fontsize=8)
    crop, last = cur[1]
    d0 = last.day
    d1 = last.days_in_month if d0 == 21 else d0 + 9   # ASAP dekads: 1–10, 11–20, 21–end of month
    when = _t(f"{d0}–{d1} {last:%B %Y}", f"{d0}–{d1} de {MONTH_ES[last.month - 1]} de {last.year}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.27, wspace=0.05)
    _source(fig, _t(f"{when}. JRC ASAP: z-score of MODIS FPAR (the share of sunlight the canopy absorbs) against the same ten days in "
                    "2001–present, averaged over cropland or rangeland in each state (GAUL admin-1). Grey: no data.",
                    f"{when}. JRC ASAP: puntaje z de la FPAR de MODIS (la fracción de la luz solar que absorbe la cubierta vegetal) frente "
                    "a los mismos diez días desde 2001, promediado sobre cultivos o pastizales en cada estado (GAUL admin-1). Gris: sin "
                    "datos."), y=0.03)
    warn = [names[pc] for pc in crop.sort_values().index if crop[pc] < -1]
    nat = float(crop.mean())
    if not (abs(nat) < 0.3 and 1 <= len(warn) <= 5):
        raise ValueError(f"the vegetation-now pattern changed (national {nat:+.2f}, under −1: {warn}): reword the section")
    joined = (", ".join(warn[:-1]) + _t(" and ", " y ") + warn[-1]) if len(warn) > 1 else warn[0]
    return _save(fig, "vegnow"), dict(veg_now_date=when, veg_warn=joined, veg_now_below=int((crop < 0).sum()),
                                      veg_now_n=int(crop.notna().sum()))


CRISIS = set(range(2014, 2020))
CRISIS_NB = set(range(2012, 2022))   # the crisis and the years whose two-either-side comparison reaches into it
MAIZE_STATES = {"VE18": "Portuguesa", "VE12": "Guárico", "VE06": "Barinas", "VE22": "Yaracuy", "VE09": "Cojedes",
                "VE03": "Anzoátegui"}


def _r(a: pd.Series, b: pd.Series, yrs) -> float:
    ok = [t for t in yrs if t in a.dropna().index and t in b.dropna().index]
    return float(np.corrcoef(a.loc[ok], b.loc[ok])[0, 1])


def yield_r_range(ph: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Correlation of yield with the ONI under other ways of removing the trend and the crisis: only the crisis years left
    out; neighbours taken from non-crisis years only; deviation from a 9-year running median of non-crisis years;
    year-on-year change. With the page's own method, the range of all five, per crop."""
    y = yields()
    out = {}
    for crop in ("maize", "rice"):
        d = y[y.crop == crop].set_index("year")
        t = d.yield_t
        base = [k for k in range(1980, NOW) if k not in CRISIS]
        tc = t.where(~t.index.isin(CRISIS))
        nb = pd.concat([tc.shift(k) for k in (-2, -1, 1, 2)], axis=1)
        rs = [_r(ph.oni, d.anom, [k for k in base if k not in CRISIS_NB]),
              _r(ph.oni, d.anom, base),
              _r(ph.oni, (t / nb.mean(axis=1).where(nb.notna().sum(axis=1) >= 2) - 1) * 100, base),
              _r(ph.oni, np.log(t) - np.log(tc).rolling(9, center=True, min_periods=5).median(), base),
              _r(ph.oni, np.log(t).diff(), [k for k in base if k - 1 not in CRISIS])]
        out[crop] = (min(rs), max(rs))
    return out


def fig_yields(mon: pd.DataFrame, ph: pd.DataFrame) -> tuple[Path, dict]:
    y = yields()
    y = y[(y.year >= 1980) & (y.year < NOW)]
    last = int(y.year.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    rs = {}
    names = {t: str(t) for t in (1982, 1997, 2002, 2009, 2015, 2023)}
    for ax, (crop, lab) in zip(axes, (("maize", _t("Maize", "Maíz")), ("rice", _t("Rice", "Arroz")))):
        d = y[y.crop == crop].set_index("year")["anom"]
        _enso_scatter(ax, ph.oni, d, ph, names if crop == "maize" else {}, hollow=CRISIS_NB)
        ok = [t for t in d.dropna().index if t not in CRISIS_NB and t in ph.index]
        rs[crop] = _r(ph.oni, d, ok)
        ax.set_title(f"{lab} {_t('against the ONI', 'frente al ONI')}\nr = {_num(f'{rs[crop]:+.2f}')}: "
                     f"{_t('no clear relation', 'sin relación clara')}", fontsize=10)
    # does national maize follow rain where it grows? May–Sep, the main season's growing months
    t = period_totals(mon[list(MAIZE_STATES)], [5, 6, 7, 8, 9]).mean(axis=1)
    rain_pct = (t / t.loc[1991:2020].mean() - 1) * 100
    maize = y[y.crop == "maize"].set_index("year")["anom"]
    _enso_scatter(axes[2], rain_pct, maize, ph, {t: str(t) for t in (2002, 2009)}, hollow=CRISIS_NB, oni=False)
    rs["rain"] = _r(rain_pct, maize, [k for k in range(1980, NOW) if k not in CRISIS_NB])
    r_rain = _num(f"{rs['rain']:+.2f}")
    axes[2].set_title(_t(f"Maize against {_span(5, 9)} rain in the maize states",
                         f"Maíz frente a la lluvia de {_span(5, 9)}\nen los estados maiceros")
                      + f"\nr = {r_rain}: {_t('no clear relation', 'sin relación clara')}", fontsize=10)
    axes[2].set_xlabel(_t("rain, % above or below normal: drier ← → wetter", "lluvia, % sobre o bajo lo normal\nmás seco ← → más húmedo"))
    axes[0].set_ylabel(_t("yield, % against the two years either side", "rendimiento, % frente a los dos años a cada lado"))
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
         ((STRONG_C, _t("strong El Niño", "El Niño fuerte")), (NINO_C, "El Niño"), (NEUTRAL_C, _phase_name("neutral")), (NINA_C, "La Niña"))]
    h.append(Line2D([], [], marker="o", ls="", color="none", markeredgecolor=TEXT2,
                    label=_t("2012–2021: the 2014–2019 crisis, or compared with it", "2012–2021: la crisis de 2014–2019 o comparados con ella")))
    fig.legend(handles=h, loc="lower center", fontsize=7.5, ncols=5, bbox_to_anchor=(0.5, 1.06 if LANG == "es" else 1.0), frameon=False)
    _source(fig, _t(f"One point per calendar year, 1980–{last} (the main harvest follows the May–October rains). Detrended: each "
                    "year's yield against the mean of the two years either side. Open circles, left out of r: the 2014–2019 economic "
                    f"crisis and the years whose comparison years fall in it. Maize states: {', '.join(MAIZE_STATES.values())} "
                    "(CHIRPS v3). Sources: FAOSTAT, CHIRPS v3.",
                    f"Un punto por año calendario, 1980–{last} (la cosecha principal sigue a las lluvias de mayo a octubre). Sin "
                    "tendencia: el rendimiento de cada año frente a la media de los dos años a cada lado. Círculos vacíos, excluidos de "
                    "r: la crisis económica de 2014–2019 y los años cuya comparación incluye años de la crisis. Estados maiceros: "
                    f"{', '.join(MAIZE_STATES.values())} (CHIRPS v3). Fuentes: FAOSTAT, CHIRPS v3."))
    alt = yield_r_range(ph)
    fmt = lambda x: f"{x:+.2f}".replace("-", "−")  # noqa: E731
    v = dict(maize_r=fmt(rs["maize"]), rice_r=fmt(rs["rice"]), maize_rain_r=fmt(rs["rain"]),
             maize_r_lo=fmt(alt["maize"][0]), maize_r_hi=fmt(alt["maize"][1]),
             rice_r_lo=fmt(alt["rice"][0]), rice_r_hi=fmt(alt["rice"][1]),
             rain_2002=f"{rain_pct[2002]:+.0f}".replace("-", "−"), rain_2009=f"{rain_pct[2009]:+.0f}".replace("-", "−"),
             **{f"maize_{t}": f"{abs(y[(y.crop == 'maize') & (y.year == t)].anom.iloc[0]):.0f}"
                for t in (1982, 1997, 2002, 2009, 2015, 2023)})
    a = {t: y[(y.crop == "maize") & (y.year == t)].anom.iloc[0] for t in (1982, 1997, 2002, 2009, 2015, 2023)}
    if not (a[2002] < -8 and a[2009] < -8 and all(abs(a[t]) < 6 for t in (1982, 1997, 2023))
            and abs(rs["maize"]) < 0.3 and abs(rs["rice"]) < 0.3
            and max(abs(x) for lohi in alt.values() for x in lohi) < 0.3):
        raise ValueError(f"maize anomalies or correlations changed {a} {rs} {alt}: reword the yields headline")
    if not (abs(rs["rain"]) < 0.2 and rain_pct[2002] > 0 and rain_pct[2009] < -10):
        raise ValueError(f"maize vs rain changed (r {rs['rain']:+.2f}, 2002 {rain_pct[2002]:+.0f}%, "
                         f"2009 {rain_pct[2009]:+.0f}%): reword the yields note")
    return _save(fig, "yields"), v


# The team's seas5-skill viewer (ds-seas5-skill docs/app.js classify() and STYLE, src/season.py): same thresholds,
# precedence (outside the rainy season, then low skill, then the forecast class) and colours.
SEAS5_T = {"sev_rp": 3, "vsev_rp": 10, "r_mod": 0.3, "r_high": 0.5, "rainy_share": 0.15}
SEAS5_FILL = {"vsev_dry": "#7f5619", "sev_dry": "#dda555", "none": "#e2e8e8", "sev_wet": "#74a1e8", "vsev_wet": "#134ead",
              "off_season": "#b1c1c2", "low_skill": "#ffffff"}
SEAS5_HATCH_WHITE, SEAS5_HATCH_GREY, SEAS5_CROSS = (1, 1, 1, 0.7), "#9db1b3", "#b1c1c2"


def seas5_class(pct: float, r: float, rainy: bool) -> tuple[str, str]:
    """(class, skill) for one forecast cell, as the viewer's classify()."""
    if np.isnan(pct) or np.isnan(r):
        raise ValueError("SEAS5 rank or skill missing for a state: the viewer would show it as not monitored")
    if not rainy:
        return "off_season", ""
    if r < SEAS5_T["r_mod"]:
        return "low_skill", ""
    v, sev = 100 / SEAS5_T["vsev_rp"], 100 / SEAS5_T["sev_rp"]
    cls = "vsev" if pct <= v or pct >= 100 - v else "sev" if pct <= sev or pct >= 100 - sev else "none"
    if cls != "none":
        cls += "_dry" if pct < 50 else "_wet"
    return cls, "high" if r >= SEAS5_T["r_high"] else "mod"


def seas5_rainy() -> pd.DataFrame:
    """In season per state × window, as the viewer's src/season.py: the three months carry at least 15% of the year's rain
    in the full-record ERA5 monthly climatology."""
    e = era5_adm1()
    mc = e.groupby(["pcode", e.valid_date.dt.month])["mean"].mean().unstack()
    return pd.DataFrame({code: 3 * mc[months].mean(axis=1) / mc.sum(axis=1) >= SEAS5_T["rainy_share"]
                         for code, months in FC_SEASONS.items()})


def fig_forecast() -> tuple[Path, dict]:
    pr, sk, year, raw = seas5_rank()
    pr, sk = pr.reindex(STATE_ORDER), sk.reindex(STATE_ORDER)
    rainy = seas5_rainy().reindex(STATE_ORDER)
    st = states()
    fig, axes = plt.subplots(1, len(FC_SEASONS), figsize=(13, 3.45))
    for ax, code in zip(axes, FC_SEASONS):
        cls = {pc: seas5_class(pr.loc[pc, code], sk.loc[pc, code], bool(rainy.loc[pc, code])) for pc in STATE_ORDER}
        g = st.assign(cls=st.pcode.map(lambda pc: cls[pc][0]), skill=st.pcode.map(lambda pc: cls[pc][1]))
        g.plot(ax=ax, color=g.cls.map(SEAS5_FILL), edgecolor="none", lw=0)
        for mask, colour, hatch in ((g.cls == "low_skill", SEAS5_CROSS, "xxx"),
                                    ((g.skill == "mod") & (g.cls != "none"), SEAS5_HATCH_WHITE, "////"),
                                    ((g.skill == "mod") & (g.cls == "none"), SEAS5_HATCH_GREY, "////")):
            if mask.any():
                g[mask].plot(ax=ax, facecolor="none", edgecolor=colour, hatch=hatch, lw=0)
        st.boundary.plot(ax=ax, color="#ffffff", lw=0.6)
        if (g.cls == "low_skill").any():
            g[g.cls == "low_skill"].boundary.plot(ax=ax, color="#c4d0d1", lw=0.6)
        ax.set_title(_cap(_span(FC_SEASONS[code][0], FC_SEASONS[code][-1])), fontsize=9.5)
        ax.set_xlim(-73.6, -59.6)
        ax.set_ylim(0.6, 12.4)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fc = [Patch(facecolor=SEAS5_FILL[k], label=lab) for k, lab in (
        ("vsev_dry", _t("Strongly below normal", "Muy por debajo de lo normal")), ("sev_dry", _t("Below normal", "Por debajo de lo normal")),
        ("none", _t("Roughly normal", "Cerca de lo normal")), ("sev_wet", _t("Above normal", "Por encima de lo normal")),
        ("vsev_wet", _t("Strongly above normal", "Muy por encima de lo normal")),
        ("off_season", _t("Outside rainy season", "Fuera de la temporada de lluvias")))]
    skill = [Patch(facecolor="#ffffff", edgecolor="#c4d0d1", label=_t("High — solid", "Alta — liso")),
             Patch(facecolor="#ffffff", edgecolor=SEAS5_HATCH_GREY, hatch="////", label=_t("Moderate — hatched", "Moderada — rayado")),
             Patch(facecolor="#ffffff", edgecolor=SEAS5_CROSS, hatch="xxx", label=_t("Low — not shown", "Baja — no se muestra"))]
    fig.legend(handles=fc, loc="lower center", ncols=6, fontsize=7.5, bbox_to_anchor=(0.5, 0.12), title=_t("Forecast", "Pronóstico"),
               title_fontsize=8)
    fig.legend(handles=skill, loc="lower center", ncols=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), title=_t("Skill", "Destreza"),
               title_fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.3, wspace=0.03)
    _source(fig, _t(f"Where the {year} October forecast ranks among the 45 October forecasts for 1981–2025, by state (ensemble "
                    "mean, detrended), classed as in the team's seas5-skill viewer: strongly below or above normal = the driest or "
                    "wettest 10% of forecasts, below or above = the driest or wettest third. Skill is the hindcast correlation with "
                    "ERA5: high from 0.5, moderate 0.3–0.5, low under 0.3. Outside rainy season: the three months bring less than 15% "
                    "of the year's rain (ERA5). Source: ECMWF SEAS5 and ERA5, team admin-1 statistics.",
                    f"Posición del pronóstico de octubre de {year} entre los 45 pronósticos de octubre de 1981–2025, por estado (media "
                    "del conjunto, sin tendencia), clasificada como en el visor seas5-skill del equipo: muy por debajo o por encima de "
                    "lo normal = el 10% más seco o más húmedo de los pronósticos; por debajo o por encima = el tercio más seco o más "
                    "húmedo. La destreza es la correlación de los pronósticos retrospectivos con ERA5: alta a partir de 0,5, moderada "
                    "0,3–0,5, baja por debajo de 0,3. Fuera de la temporada de lluvias: los tres meses aportan menos del 15% de la "
                    "lluvia del año (ERA5). Fuente: ECMWF SEAS5 y ERA5, estadísticas admin-1 del equipo."), y=-0.06)
    in_season = rainy[["OND", "NDJ", "DJF"]]
    sk_in = sk[["OND", "NDJ", "DJF"]].where(in_season)
    v = dict(fc_year=year, fc_driest=int(((pr[["OND", "NDJ", "DJF"]] <= 100 / SEAS5_T["vsev_rp"]) & in_season).sum().sum()),
             fc_cells=int(in_season.sum().sum()),
             fc_r_lo=f"{sk_in.min().min():.1f}", fc_r_hi=f"{sk_in.max().max():.1f}",
             fc_raw_driest=int(raw.reindex(STATE_ORDER)["NDJ"].sum()), fc_states=len(STATE_ORDER))
    west = [p for r in ("North-west", "Western Llanos", "Centre-north") for p in REGIONS[r][0]]
    shown = rainy.loc[west, "FMA"] & (sk.loc[west, "FMA"] >= SEAS5_T["r_mod"])
    if (v["fc_driest"] < v["fc_cells"] or sk_in.min().min() < SEAS5_T["r_mod"]
            or pr.loc[west, "FMA"].median() < 66 or shown.any()):
        raise ValueError("the SEAS5 pattern changed (Oct–Feb strongly below normal and skilful in every rainy-season "
                         "state-season; Feb–Apr wet but masked, low skill or off-season, in the west and centre): reword the "
                         "forecast headline and note")
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
.lead{font-size:16px;line-height:1.5;border-left:4px solid #b5541b;background:#fff;padding:12px 16px;margin:6px 0 14px;max-width:90ch}
.lang{display:inline-block;margin:0 8px 14px 0;padding:5px 11px;font-size:13px;font-weight:500;border:1px solid var(--b1);border-radius:4px;background:var(--b05);text-decoration:none}
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


def cerf_check() -> None:
    """The page labels one CERF flood allocation and says there has been none for drought; stop if any allocation is added
    (a drought response might be coded as food insecurity, not Drought) or the flood one changes."""
    c = cerf()
    wet = c[c.emergency_type.isin(["Flood", "Storm"])]
    if set(c.application_code) != CERF_KNOWN or set(wet.application_code) != set(CERF_EMDAT) or (c.emergency_type == "Drought").any():
        raise ValueError(f"CERF allocations to Venezuela changed (new: {sorted(set(c.application_code) - CERF_KNOWN)}): "
                         "check them, then update CERF_KNOWN, CERF_EMDAT and the drought line")


def figures(a: dict) -> tuple[dict, dict]:
    ph, f, v = a["ph"], {}, {}
    v |= oni_now()
    f["maps"], x = fig_maps(ph); v |= x
    v |= season_contrast(a["mon"], ph)
    zm = zone_monthly()
    f["states"], x = fig_states(season_totals(zm), ph); v |= x
    zs = zone_seasons(zm)
    v |= zone_values(zs, zm, ph)
    f["season"] = fig_season(zm, ph)
    f["zones"] = fig_zones(zs, ph)
    f["fewsnet"] = fig_fewsnet()
    f["calendar"] = fig_calendar()
    v |= era5_agreement(a)
    f["now"], x = fig_now(a["mon"], a["mon_e"], ph); v |= x
    f["forecast"], x = fig_forecast(); v |= x
    cerf_check()
    f["emdat"], x = fig_emdat(ph); v |= x
    f["desinventar"], x = fig_desinventar(ph); v |= x
    f["guri"], x = fig_guri(ph); v |= x
    f["veg"], x = fig_veg(ph); v |= x
    f["vegnow"], x = fig_vegnow(); v |= x
    f["yields"], x = fig_yields(a["mon"], ph); v |= x
    nino, nina = phase_years(ph, "nino"), phase_years(ph, "nina")
    v |= dict(n_nino=len(nino), n_nina=len(nina), strong_years=", ".join(str(y) for y in ph.index[ph.strong] if y >= Y0),
              data_end=_month_year(a["mon"].dropna(how="all").index.max()), era5_end=_month_year(a["mon_e"].dropna(how="all").index.max()),
              emdat_end=str(emdat()["Last Update"].max()) if "Last Update" in emdat() else "")
    return f, _localise(v)


def render(spec: dict, figs: dict, v: dict) -> str:
    """The page in the current language; the Spanish page sits one level down, at es/."""
    import enso_deep_dive as edd
    pg = spec["page"]
    es = LANG == "es"
    head = edd.HEAD.format(title=html.escape(pg["title"]), desc=html.escape(pg["sub"]), css=edd.CSS + EXTRA_CSS,
                           home="../../" if es else "../", home_label=_t("ENSO deep dives", "Análisis ENSO por país"))
    out = [head.replace('<html lang="en">', '<html lang="es">') if es else head]
    out.append('<p class="eyebrow">El Niño 2026/27</p>')
    out.append(f'<h1>{pg["title"]}</h1><p class="meta">{pg["sub"].format(**v)}<br>{pg["dateline"].format(**v)}</p>')
    out.append('<a class="lang" href="../" lang="en">English version</a>' if es else
               '<a class="lang" href="es/" lang="es">Versión en español</a>')
    out.append(f'<a class="lang" href="{"../slides/es/" if es else "slides/"}">'   # the short deck (ven_slides.py)
               f'{_t("Slides (PowerPoint)", "Diapositivas (PowerPoint)")}</a>')
    if pg.get("lead"):
        out.append(f'<p class="lead">{pg["lead"].format(**v)}</p>')
    out.append('<div class="tiles">' + "".join(f'<div class="tile"><p class="big">{t["big"].format(**v)}</p><p>{t["text"].format(**v)}</p></div>'
                                                for t in spec["tiles"]) + "</div>")
    for sec in spec["sections"]:
        if sec.get("part"):
            out.append(f'<h2 style="margin-top:36px">{sec["part"]}</h2>')
            if sec.get("part_note"):
                out.append(f'<p class="note">{sec["part_note"].format(**v)}</p>')
        out.append(f'<section id="{sec["id"]}"><h3>{sec["head"].format(**v)}</h3>')
        if sec.get("note"):
            out.append(f'<p class="note">{sec["note"].format(**v)}</p>')
        for fig in [sec["fig"]] if isinstance(sec["fig"], str) else sec["fig"]:
            out.append(f'<figure><img src="{_img(figs[fig])}" alt="{html.escape(sec["head"].format(**v))}"></figure>')
        out.append("</section>")
    notes = "".join(f"<li>{n.format(**v)}</li>" for n in spec["notes"])
    out.append(f'<details><summary>{_t("About the data", "Sobre los datos")}</summary><ul>{notes}</ul></details>')
    out.append(f'<p class="small">{_t("OCHA Centre for Humanitarian Data · built", "Centro de Datos Humanitarios de OCHA · generado el")} '
               f'{pd.Timestamp.now():%Y-%m-%d}</p>')
    out.append(edd.FOOT)
    return "\n".join(out)


def _placeholders(x) -> set[str]:
    import string
    if isinstance(x, str):
        return {f for _, f, _, _ in string.Formatter().parse(x) if f}
    if isinstance(x, dict):
        return set().union(*map(_placeholders, x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*map(_placeholders, x)) if x else set()
    return set()


def _numbers(page: str, lang: str) -> list[str]:
    """Every number in the page's visible text (figures and markup dropped), with the separators normalised."""
    text = re.sub(r"<img[^>]*>|<[^>]+>", " ", page[page.index("<body>"):])
    nums = re.findall(r"\d(?:[\d.,]*\d)?", html.unescape(text))
    return sorted(n.replace(".", "").replace(",", ".") if lang == "es" else n.replace(",", "") for n in nums)


def check_parity(spec: dict, es_spec: dict, pages: dict[str, str]) -> None:
    """The Spanish page must say the same things: same placeholders, same sections, same numbers."""
    if _placeholders(spec) != _placeholders(es_spec):
        raise ValueError(f"placeholders differ: EN only {_placeholders(spec) - _placeholders(es_spec)}, "
                         f"ES only {_placeholders(es_spec) - _placeholders(spec)}")
    def shape(sp: dict) -> list:
        fields = [sp["page"].get(k, "") for k in ("title", "sub", "dateline", "lead")] + list(sp["notes"])
        fields += [t[k] for t in sp["tiles"] for k in ("big", "text")]
        fields += [x.get(k, "") for x in sp["sections"] for k in ("part", "part_note", "head", "note")]
        return ([(x["id"], x["fig"], sorted(x)) for x in sp["sections"]] + [len(sp["tiles"]), len(sp["notes"])]
                + [(bool(f), sorted(_placeholders(f))) for f in fields])
    if shape(spec) != shape(es_spec):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(shape(spec), shape(es_spec))) if a != b]
        raise ValueError(f"the Spanish TOML does not match deep_dives/ven.toml field by field: {diff[:3]}")
    from collections import Counter
    en, es = Counter(_numbers(pages["en"], "en")), Counter(_numbers(pages["es"], "es"))
    if en != es:
        raise ValueError(f"numbers differ between the English and Spanish pages: EN only {dict(en - es)}, ES only {dict(es - en)}")


def write_index() -> None:
    """Re-render the /enso/ card index from every deep_dives/*.toml."""
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def build(spec: dict | None = None, *_args, publish: bool = False) -> dict:
    """Called by enso_deep_dive.main() for builder = "ven" (grid, gdf, indices unused), or standalone. Builds the English
    page and, from the same analysis, the Spanish one (deep_dives/i18n/ven.es.toml)."""
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    es_spec = tomllib.loads(ES_SPEC.read_text(encoding="utf-8"))
    a = analyse()
    pages, vals = {}, {}
    try:
        for lang, sp in (("en", spec), ("es", es_spec)):
            set_lang(lang)
            figs, vals[lang] = figures(a)
            pages[lang] = render(sp, figs, vals[lang])
    finally:
        set_lang("en")
    check_parity(spec, es_spec, pages)
    for lang, page in pages.items():
        d = OUT / "es" if lang == "es" else OUT
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page, encoding="utf-8")
        print(f"  wrote {d}/index.html")
    print("  values:", vals["en"])
    if publish:
        for lang, page in pages.items():
            d = DOCS / "es" if lang == "es" else DOCS
            d.mkdir(parents=True, exist_ok=True)
            (d / "index.html").write_text(page, encoding="utf-8")
        write_index()
        print(f"  published -> {DOCS}/index.html, {DOCS}/es/index.html and docs/enso/index.html")
    return vals["en"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/ven/ and the /enso/ card index")
    args = ap.parse_args()
    build(publish=args.publish)


if __name__ == "__main__":
    main()
