"""Angola — the 2026/27 rainy season: forecast, rain so far, how much is El Niño, food security (builder = "ago").

Plot-led like the Venezuela page: one plain headline per figure, a short note at most, numbers filled from the build
(the build stops when a headline's claim stops holding). The questions: how dry is the coming season forecast to be
(return period, with the forecast's skill), what has fallen so far, how much of it is El Niño, and where food
insecurity already is.

    uv run python ago_deep_dive.py              # build out/ago/ (gitignored) to review
    uv run python ago_deep_dive.py --publish    # + docs/enso/ago/ and the /enso/ card index

Inputs (cached under cache/ago/; fetched when missing or stale): CHIRPS v3 and ERA5 at 0.5° (ds-precip-intercomparison
cubes on the dev blob), CHIRPS v3 monthly and pentad tifs from CHC for the months after the cube; ERA5 and SEAS5
admin-0/1 raster stats (team prod DB, through the laptop tunnel); province boundaries (COD-AB via FieldMaps); NOAA CPC
ONI; JRC ASAP biomass (zFPARc) by GAUL admin-1; FEWS NET classification (ds-fewsnet-mirror) and population estimates
(FEWS NET Data Warehouse); IPC (team mirror, dev DB); CERF allocations (team OneGMS mirror, dev DB).

Target areas: put p-codes in [targets] in deep_dives/ago.toml; every province statistic is also computed for each
target group (area-weighted), and the groups get a row in the table and an outline on the maps.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import textwrap
import tomllib
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
SLUG = "ago"
CACHE = ROOT / "cache" / "ago"
OUT = ROOT / "out" / "ago"
FIGDIR = OUT / "fig"
DOCS = ROOT / "docs" / "enso" / SLUG
SPEC = ROOT / "deep_dives" / f"{SLUG}.toml"
for _d in (CACHE, FIGDIR):
    _d.mkdir(parents=True, exist_ok=True)

# Palette: the Venezuela page's (dataviz-validated) phase colours and diverging ramp
NINO_C, STRONG_C, NINA_C, NEUTRAL_C, NOW_C = "#b5541b", "#7a2d0b", "#2a78d6", "#c9c8c2", "#0b0b0b"
TEXT, TEXT2, GRID = "#1f2324", "#5e6a6b", "#e6e5e1"
DIV = ["#7a4a12", "#b77a35", "#e2c48e", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]
IPC_COLOURS = {1: "#CDFACD", 2: "#FAE61E", 3: "#E67800", 4: "#C80000", 5: "#640000"}
IPC_LABELS = {1: "1 Minimal", 2: "2 Stressed", 3: "3 Crisis", 4: "4 Emergency", 5: "5 Famine"}
plt.rcParams["hatch.color"] = "#c9c8c2"
plt.rcParams["hatch.linewidth"] = 0.6

NOW = 2026                  # season 2026/27; seasons are labelled by the year they start (Nov 2026 – Mar 2027 = 2026)
Y0, Y1 = 1981, 2025         # complete seasons in the record
EXTENT = dict(lon=(11.5, 24.2), lat=(-18.2, -4.3))
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TRIMESTERS = {"".join("JFMAMJJASOND"[(m + k) % 12] for k in range(3)): [(m + k) % 12 + 1 for k in range(3)] for m in range(12)}
WINDOWS: dict[str, list[int]] = {}   # the five trimesters an issuance reaches (leads 0–4); set by set_issued()


def set_issued(im: int) -> None:
    WINDOWS.clear()
    for lead in range(5):
        m0 = (im - 1 + lead) % 12
        code = next(c for c, ms in TRIMESTERS.items() if ms[0] == m0 + 1)
        WINDOWS[code] = TRIMESTERS[code]
    if SEASON[0] != im and (SEASON[0] - im) % 12 > 1:
        raise ValueError("the Nov–Mar season starts more than one month after the issuance")


SEASON = [11, 12, 1, 2, 3]  # "the season": November–March, inside the October forecast and the core of the rains everywhere
SEASON_LBL = "Nov–Mar"
SEAS5_T = dict(sev_rp=3, vsev_rp=10, r_mod=0.3, r_high=0.5, rainy_share=0.15)   # seas5-skill viewer thresholds


def _span(m0: int, m1: int) -> str:
    return f"{MON[(m0 - 1) % 12]}–{MON[(m1 - 1) % 12]}"


def win_label(code: str) -> str:
    m = WINDOWS[code]
    return _span(m[0], m[-1])


def season_label(y: int) -> str:
    return f"{y}/{str(y + 1)[2:]}"


# --------------------------------------------------------------------------- #
# Data access (cache first; fetch when missing or stale)
# --------------------------------------------------------------------------- #
def _stale(p: Path, days: float) -> bool:
    return not p.exists() or (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).days > days


def _fetch(url: str, p: Path, days: float, **kw) -> Path:
    if _stale(p, days):
        try:
            r = requests.get(url, timeout=300, **kw)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            if not p.exists():
                raise
            print(f"  ({p.name}: fetch failed, using cache: {str(e)[:80]})")
    return p


def _db(name: str, sql: str, stage: str, max_age: float = 30) -> pd.DataFrame:
    """Team database through the laptop tunnel (the FQDN host is not reachable from outside the VNet)."""
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


def provinces() -> gpd.GeoDataFrame:
    """The 18 provinces (COD-AB via FieldMaps; pcodes AO01–AO18 = the team's polygon table), north to south."""
    p = CACHE / "ago_adm1.parquet"
    if not p.exists():
        import ocha_stratus as stratus
        g = stratus.codab.load_codab_from_fieldmaps("AGO", 1)
        g = g[g.adm1_name.notna()].rename(columns={"adm1_src": "pcode", "adm1_name": "name"})[["pcode", "name", "geometry"]]
        g.to_parquet(p)
    g = gpd.read_parquet(p)
    g["lat"] = g.geometry.representative_point().y
    g["km2"] = g.to_crs("ESRI:102022").area / 1e6   # Africa Albers equal area
    return g.sort_values("lat", ascending=False).reset_index(drop=True)


def era5_adm() -> pd.DataFrame:
    """ERA5 monthly mean precipitation (mm/day) per province and nationally (team raster stats, prod)."""
    df = _db("era5_adm1", "select pcode, adm_level, valid_date, mean from public.era5 where iso3 = 'AGO' and adm_level in (0, 1)",
             "prod", 20)
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    return df


def seas5_adm() -> pd.DataFrame:
    """SEAS5 monthly ensemble-mean precipitation (mm/day) per province and nationally, by issuance and lead (prod)."""
    now = pd.Timestamp.now()
    expected = (now if now.day >= 7 else now - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    p = CACHE / "seas5_adm1.parquet"
    stale = not p.exists() or pd.Timestamp(pd.read_parquet(p, columns=["issued_date"]).issued_date.max()) < expected
    df = _db("seas5_adm1", "select pcode, adm_level, issued_date, valid_date, leadtime, mean from public.seas5 "
             "where iso3 = 'AGO' and adm_level in (0, 1)", "prod", 0 if stale else 3650)
    for c in ("issued_date", "valid_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def oni() -> pd.DataFrame:
    p = _fetch("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", CACHE / "oni.txt", 7)
    df = pd.read_csv(p, sep=r"\s+")
    df.columns = ["seas", "year", "total", "anom"]
    return df


def oni_season(seas: str) -> pd.Series:
    df = oni()
    return df[df.seas == seas].set_index("year")["anom"]


def phases() -> pd.DataFrame:
    """Each season (Nov Y – Mar Y+1) classed by the Sep–Nov ONI of year Y, as on the sibling pages: El Niño ≥ +0.5 °C,
    strong ≥ +1.5 °C, La Niña ≤ −0.5 °C."""
    df = oni_season("SON").to_frame("oni")
    df["phase"] = np.where(df.oni >= 0.5, "nino", np.where(df.oni <= -0.5, "nina", "neutral"))
    df["strong"] = df.oni >= 1.5
    return df


def phase_years(ph: pd.DataFrame, phase: str, strong: bool = False) -> list[int]:
    s = ph[(ph.phase == phase) & (ph.strong if strong else True)]
    return [int(y) for y in s.index if Y0 <= y <= Y1]


def phase_color(y: int, ph: pd.DataFrame) -> str:
    if y == NOW:
        return NOW_C
    if y not in ph.index:
        return NEUTRAL_C
    r = ph.loc[y]
    return STRONG_C if r.strong else NINO_C if r.phase == "nino" else NINA_C if r.phase == "nina" else NEUTRAL_C


def oni_now() -> dict:
    last = oni().iloc[-1]
    if last.anom < 1.5:
        raise ValueError(f"latest ONI {last.anom:+.2f}: 2026 is no longer a strong El Niño, reword the page")
    seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    m0 = seas.index(last.seas)
    return dict(oni_now=f"{last.anom:+.1f}", oni_seas=_span(m0, m0 + 2) if m0 else _span(12, 2), oni_jas=float(oni_season("JAS")[NOW]))


# ---- rain grids ------------------------------------------------------------------------------------------------------
CHIRPS_MONTHLY = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/chirps-v3.0.{y}.{m:02d}.tif"


def _read_tif(content_or_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import rasterio
    with (rasterio.MemoryFile(content_or_path).open() if isinstance(content_or_path, bytes)
          else rasterio.open(content_or_path)) as src:
        win = rasterio.windows.from_bounds(10.5, -19.0, 25.0, -3.5, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype("float32")
        t = src.window_transform(win)
    a[a < 0] = np.nan          # CHC tifs carry no nodata flag: negatives are the sea mask
    lat = t.f + t.e * (np.arange(a.shape[0]) + 0.5)
    lon = t.c + t.a * (np.arange(a.shape[1]) + 0.5)
    return a, lat, lon


def _to_cube_grid(a: np.ndarray, lat: np.ndarray, lon: np.ndarray, cube: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """0.05° tif → the cube's 0.5° cells, as the cube is built: mean of the valid 0.05° cells and their share."""
    g = xr.DataArray(a, dims=("lat", "lon"), coords=dict(lat=lat, lon=lon))
    grid = dict(lat=cube.lat, lon=cube.lon, method="nearest", tolerance=0.01)
    lat0 = float(cube.lat.min()) - 0.25
    lon0 = float(cube.lon.min()) - 0.25
    g = g.sel(lat=slice(float(cube.lat.max()) + 0.25, lat0), lon=slice(lon0, float(cube.lon.max()) + 0.25))
    mean = g.coarsen(lat=10, lon=10, boundary="trim").mean().reindex(**grid)
    frac = g.notnull().coarsen(lat=10, lon=10, boundary="trim").mean().reindex(**grid)
    return mean, frac


def rain(product: str = "chirps_v3") -> xr.DataArray:
    """Monthly totals (mm) at 0.5° over Angola (ds-precip-intercomparison cubes, dev blob); CHIRPS extended with CHC's final
    monthly tifs for the months after the cube."""
    p = CACHE / f"{product}_ago05.nc"
    if not p.exists():
        import ocha_stratus as stratus
        cc = stratus.get_container_client(container_name="projects", stage="dev")
        raw = cc.get_blob_client(f"ds-precip-intercomparison/processed/grid05/{product}.nc").download_blob().readall()
        ds = xr.open_dataset(io.BytesIO(raw), engine="h5netcdf")
        lat = slice(-18.5, -4.0) if ds.lat.values[0] < ds.lat.values[-1] else slice(-4.0, -18.5)
        ds.sel(lat=lat, lon=slice(11.0, 24.5)).load().to_netcdf(p)
    ds = xr.open_dataset(p).sortby("lat")
    if product == "chirps_v3":
        ext, t = [], pd.Timestamp(ds.time.values[-1]) + pd.DateOffset(months=1)
        while t < pd.Timestamp.now().normalize().replace(day=1):
            f = CACHE / "chirps_v3_tifs" / f"chirps-v3.0.{t.year}.{t.month:02d}.tif"
            if not f.exists():
                latest = t + pd.DateOffset(months=1) >= pd.Timestamp.now().normalize().replace(day=1)
                try:
                    r = requests.get(CHIRPS_MONTHLY.format(y=t.year, m=t.month), timeout=(30, 300))
                except requests.exceptions.ConnectionError:
                    if not latest:           # only the month just ended may be missing
                        raise
                    print(f"  CHC unreachable: CHIRPS monthly for {t:%Y-%m} not checked (preliminary pentads cover it)")
                    break
                if r.status_code != 200:
                    break
                if r.content[:4] not in (b"II*\x00", b"MM\x00*"):
                    raise ValueError(f"CHC returned something other than a GeoTIFF for {t:%Y-%m}")
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
            mean, frac = _to_cube_grid(*_read_tif(f), ds)
            ext.append(xr.Dataset(dict(precip=mean.expand_dims(time=[t]), valid_frac=frac)))
            t += pd.DateOffset(months=1)
        precip = xr.concat([ds["precip"]] + [e["precip"] for e in ext], "time") if ext else ds["precip"]
        return precip.where(ds["valid_frac"] > 0)     # any land in the cell: the cube's value is the mean of its land pixels
    return ds["precip"]


def _weights(da: xr.DataArray, gdf: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    """Share of each 0.5° cell inside each polygon (rasterised at 0.025°), on the cube's south-to-north rows."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    lat, lon, sub = da.lat.values, da.lon.values, 20
    res = float(abs(lon[1] - lon[0]))
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res / sub, res / sub)
    out = {}
    for pc, g in zip(gdf.pcode, gdf.geometry):   # one polygon at a time: groups overlap their provinces
        m = rasterize([(g, 1)], out_shape=(lat.size * sub, lon.size * sub), transform=tr, fill=0)
        out[pc] = m.reshape(lat.size, sub, lon.size, sub).mean((1, 3))[::-1]
    return out


def _area_mean(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean over the cells of a (time, lat, lon) stack; cells missing at any time are left out."""
    w = w * np.cos(np.deg2rad(LAT2D))
    ok = np.isfinite(v).mean(0) >= 0.95                 # cells with data at (nearly) every time step
    ok &= w > 0
    if not ok.any():
        raise ValueError("a polygon has no rain cells")
    x = v[:, ok]
    ww = np.where(np.isfinite(x), w[ok], 0.0)
    return np.nansum(x * ww, 1) / ww.sum(1)


LAT2D = None


def prov_monthly(product: str = "chirps_v3") -> pd.DataFrame:
    """Monthly rain (mm) per province, per target group and nationally ('AO'), area-weighted from the 0.5° cube."""
    global LAT2D
    da = rain(product)
    LAT2D = np.repeat(da.lat.values[:, None], da.lon.size, 1)
    p = CACHE / f"{product}_prov_monthly.parquet"
    if p.exists() and pd.Timestamp(pd.read_parquet(p).index.max()) >= pd.Timestamp(da.time.values[-1]) and set(UNITS) <= set(pd.read_parquet(p).columns):
        return pd.read_parquet(p)
    v = da.values
    out = {pc: _area_mean(v, w) for pc, w in _weights(da, unit_polygons()).items()}
    df = pd.DataFrame(out, index=pd.DatetimeIndex(da.time.values))
    df.to_parquet(p)
    return df


# ---- target groups ---------------------------------------------------------------------------------------------------
GROUPS: dict[str, dict] = {}   # key -> {"label", "pcodes"}; filled from the TOML by set_groups()
UNITS: list[str] = []


def set_groups(spec: dict) -> None:
    global UNITS
    GROUPS.clear()
    for key, g in spec.get("groups", {}).items():
        GROUPS[key] = dict(label=g["label"], pcodes=list(g["pcodes"]), target=bool(g.get("target", False)))
    st = provinces()
    bad = {p for g in GROUPS.values() for p in g["pcodes"]} - set(st.pcode)
    if bad:
        raise ValueError(f"group p-codes not in the province layer: {sorted(bad)} (municipality targets need the pixel route)")
    UNITS = list(st.pcode) + list(GROUPS) + ["AO"]


def unit_polygons() -> gpd.GeoDataFrame:
    st = provinces()[["pcode", "name", "geometry"]]
    extra = [dict(pcode=k, name=g["label"], geometry=st[st.pcode.isin(g["pcodes"])].union_all()) for k, g in GROUPS.items()]
    extra.append(dict(pcode="AO", name="Angola", geometry=st.union_all()))
    return pd.concat([st, gpd.GeoDataFrame(extra, crs=st.crs)], ignore_index=True)


def unit_name(u: str) -> str:
    if u == "AO":
        return "Angola"
    if u in GROUPS:
        return GROUPS[u]["label"]
    return dict(zip(provinces().pcode, provinces().name))[u]


def _group_series(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Province mm/day series → each group's area-weighted mean (provinces' 'mean' is already a spatial mean)."""
    km2 = provinces().set_index("pcode").km2
    out = [df]
    for k, g in GROUPS.items():
        sub = df[df.pcode.isin(g["pcodes"])].copy()
        sub["w"] = sub.pcode.map(km2)
        n = sub.groupby(key_cols)["pcode"].nunique()
        m = sub.groupby(key_cols).apply(lambda x: np.average(x["mean"], weights=x["w"]), include_groups=False)
        m = m[n == len(g["pcodes"])].rename("mean").reset_index()
        m["pcode"] = k
        out.append(m)
    return pd.concat(out, ignore_index=True)


# ---- other sources ---------------------------------------------------------------------------------------------------
ASAP_EXPORT = "https://agricultural-production-hotspots.ec.europa.eu/export/rum/export.php"
ASAP_AGO = 2   # ASAP's own country id (asap0_id), not an ISO code


def asap_biomass(class_id: int) -> pd.DataFrame:
    """JRC ASAP zFPARc (cumulative FPAR anomaly during the growing cycle, MODIS) per GAUL admin-1 unit and dekad."""
    p = CACHE / f"asap_zfparc_g1_{class_id}.csv"
    if _stale(p, 9):
        r = requests.get(ASAP_EXPORT, params=dict(gaul_level=1, country_id=ASAP_AGO, class_id=class_id, variable_id=240,
                                                  classesset_id=1, sensor_id=3), timeout=600)
        r.raise_for_status()
        if r.text.startswith("country_id,"):
            p.write_text(r.text)
        elif not p.exists():
            raise ValueError(f"unexpected ASAP response: {r.text[:200]!r}")
    a = pd.read_csv(p, dtype={"date": str})
    a["date"] = pd.to_datetime(a["date"], format="%Y%m%d")
    a.loc[a["value"] <= -999, "value"] = np.nan
    name2pc = {n: pc for pc, n in zip(provinces().pcode, provinces().name)}
    a["pcode"] = a.region_name.map(name2pc)
    if a.pcode.isna().any():
        raise ValueError(f"ASAP units without a province: {sorted(a.loc[a.pcode.isna(), 'region_name'].unique())}")
    return a


def cerf() -> pd.DataFrame:
    return _db("cerf_ago", "select application_code, year, window_name, emergency_type, title, amount_approved, "
                           "erc_endorsement_date from aa.cerf_allocation where country_iso3 = 'AGO'", "dev")


def ipc() -> pd.DataFrame:
    return _db("ipc_ago", "select * from ipc.population where iso3 = 'AGO'", "dev")


def fews_population() -> pd.DataFrame:
    p = _fetch("https://fdw.fews.net/api/ipcpopulationsize.csv?country_code=AO", CACHE / "fdw_ipcpopulationsize_AO.csv", 3)
    df = pd.read_csv(p, encoding="utf-8-sig")
    for c in ("projection_start", "projection_end", "reporting_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def fews_classes() -> dict:
    import enso_deep_dive as edd
    cwd = os.getcwd()
    os.chdir(ROOT)               # edd caches under the relative cache/ directory
    try:
        f = edd.load_fews("AGO")
    finally:
        os.chdir(cwd)
    if f is None:
        raise ValueError("FEWS NET classification for Angola not available")
    return f


# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #
def season_totals(mon: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Totals over `months` per season (labelled by the year of its first month), complete seasons only."""
    df = mon.copy()
    first = months[0]
    df["sy"] = np.where(df.index.month >= first, df.index.year, df.index.year - 1)
    df = df[df.index.month.isin(months)]
    g = df.groupby("sy")
    t, n = g.sum(min_count=1), g.size()
    return t[n == len(months)]


def pct_rank(t: pd.DataFrame, y: int, y0: int = Y0, y1: int = Y1) -> pd.Series:
    """Where season y ranks among the record seasons (share of record seasons at or below it, %)."""
    ref = t.loc[y0:y1]
    return (ref.le(t.loc[y]) .sum() / ref.notna().sum() * 100)


def rp_dry(t: pd.DataFrame, y: int, y0: int = Y0, y1: int = Y1) -> pd.Series:
    """Weibull return period of season y on the dry side among the record seasons plus y: (n + 1) / rank."""
    ref = t.loc[y0:y1]
    rank = ref.lt(t.loc[y]).sum() + 1
    return (ref.notna().sum() + 1) / rank


# --------------------------------------------------------------------------- #
# SEAS5: the team's seas5-skill method (src/skill.py run_all_combinations, detrend=True), per province, group and
# nationally: trimester or season means of the ensemble mean, log1p, scaled to ERA5 over the overlap, both detrended;
# rank among the hindcasts, Weibull RP = (n + 1) / rank as the viewer, Pearson r, and P(lower tercile) from the
# conditional regression with the empirical residual sigma.
# --------------------------------------------------------------------------- #
def _seas5_pairs(issued_month: int, months: list[int]) -> dict[str, tuple[pd.Series, pd.Series]]:
    s, e = seas5_adm(), era5_adm()
    s = _group_series(s[s.issued_date.dt.month == issued_month].dropna(subset=["mean"]), ["issued_date", "valid_date"])
    e = _group_series(e.dropna(subset=["mean"]), ["valid_date"])
    first = months[0]
    out = {}
    for u in UNITS:
        f = s[(s.pcode == u) & s.valid_date.dt.month.isin(months)]
        n = f.groupby(f.issued_date.dt.year)["mean"].count()
        f = f.groupby(f.issued_date.dt.year)["mean"].mean()[n == len(months)]
        # season year = the year of the window's first month (the October issuance reaches forward only)
        lead_first = (first - issued_month) % 12
        f.index = f.index + ((issued_month - 1 + lead_first) // 12)
        o = e[(e.pcode == u) & e.valid_date.dt.month.isin(months)]
        sy = np.where(o.valid_date.dt.month >= first, o.valid_date.dt.year, o.valid_date.dt.year - 1)
        g = o.groupby(sy)["mean"]
        o = g.mean()[g.count() == len(months)]
        out[u] = (f, o)
    return out


def _skill_stats(f: pd.Series, o: pd.Series) -> dict:
    raw = f.copy()
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
        raise ValueError("the latest SEAS5 issuance is already observed")
    h = f[hist].values
    r = float(np.corrcoef(h, o[hist].values)[0, 1])
    return dict(year=cur, n=len(h), pct=100.0 * float((h <= f[cur]).sum()) / len(h),
                rp_dry=(len(h) + 1) / (int((h < f[cur]).sum()) + 1), rp_wet=(len(h) + 1) / (int((h > f[cur]).sum()) + 1),
                r=r, z=float((f[cur] - h.mean()) / h.std(ddof=1)),
                raw_driest=bool(raw.drop(cur).loc[hist].min() > raw[cur]), f=f, o=o, hist=hist)


def seas5_table(issued_month: int) -> dict[str, pd.DataFrame]:
    """{window: DataFrame(index=unit) of year, n, pct, rp_dry, rp_wet, r, z, raw_driest} for the five trimesters and
    the Nov–Mar season."""
    out = {}
    for code, months in list(WINDOWS.items()) + [("season", SEASON)]:
        rows = {}
        for u, (f, o) in _seas5_pairs(issued_month, months).items():
            s = _skill_stats(f, o)
            rows[u] = {k: v for k, v in s.items() if k not in ("f", "o", "hist")}
        out[code] = pd.DataFrame(rows).T
    return out


def seas5_series(issued_month: int, months: list[int]) -> dict[str, dict]:
    return {u: _skill_stats(f, o) for u, (f, o) in _seas5_pairs(issued_month, months).items()}


def seas5_rainy() -> pd.DataFrame:
    """In season per unit × window, as the viewer's src/season.py: the three months carry at least 15% of the year's rain
    in the full-record ERA5 monthly climatology."""
    e = _group_series(era5_adm().dropna(subset=["mean"]), ["valid_date"])
    mc = e.groupby(["pcode", e.valid_date.dt.month])["mean"].mean().unstack()
    return pd.DataFrame({code: 3 * mc[m].mean(axis=1) / mc.sum(axis=1) >= SEAS5_T["rainy_share"] for code, m in WINDOWS.items()})


def seas5_class(pct: float, r: float, rainy: bool) -> tuple[str, str]:
    """(class, skill) for one forecast cell, as the viewer's classify()."""
    if np.isnan(pct) or np.isnan(r):
        raise ValueError("SEAS5 rank or skill missing for a province")
    if not rainy:
        return "off_season", ""
    if r < SEAS5_T["r_mod"]:
        return "low_skill", ""
    v, sev = 100 / SEAS5_T["vsev_rp"], 100 / SEAS5_T["sev_rp"]
    cls = "vsev" if pct <= v or pct >= 100 - v else "sev" if pct <= sev or pct >= 100 - sev else "none"
    if cls != "none":
        cls += "_dry" if pct < 50 else "_wet"
    return cls, "high" if r >= SEAS5_T["r_high"] else "mod"


SEAS5_FILL = {"vsev_dry": "#a35a00", "sev_dry": "#e3a54f", "none": "#f2f2ee", "sev_wet": "#7fb4e0", "vsev_wet": "#1f5fa3",
              "off_season": "#b1c1c2", "low_skill": "#ffffff"}
SEAS5_HATCH_WHITE, SEAS5_HATCH_GREY, SEAS5_CROSS = (1, 1, 1, 0.7), "#9db1b3", "#b1c1c2"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> Path:
    p = FIGDIR / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _source(fig, text: str, y: float = -0.02, width: int = 170):
    fig.text(0.01, y, textwrap.fill(text, width), fontsize=7, color=TEXT2, ha="left", va="top")


def _map_axes(ax, outline_targets: bool = True):
    st = provinces()
    st.boundary.plot(ax=ax, color="#ffffff", lw=0.5)
    if outline_targets:
        for g in GROUPS.values():
            if g["target"]:
                gpd.GeoSeries([st[st.pcode.isin(g["pcodes"])].union_all()], crs=st.crs).boundary.plot(ax=ax, color=TEXT, lw=1.2)
    ax.set_xlim(*EXTENT["lon"])
    ax.set_ylim(*EXTENT["lat"])
    ax.set_aspect("equal")
    ax.set_axis_off()


def _edges(c: np.ndarray) -> np.ndarray:
    d = np.diff(c) / 2
    return np.concatenate([[c[0] - d[0]], c[:-1] + d, [c[-1] + d[-1]]])


def _clip(da: xr.DataArray) -> np.ndarray:
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    lat, lon = da.lat.values, da.lon.values
    res = float(abs(lon[1] - lon[0]))
    tr = from_origin(lon.min() - res / 2, lat.max() + res / 2, res, res)
    m = geometry_mask(provinces().geometry, out_shape=(lat.size, lon.size), transform=tr, invert=True, all_touched=True)
    return m[::-1] if lat[0] < lat[-1] else m


def _label_provinces(ax, fontsize=6.5, color=TEXT2):
    for _, r in provinces().iterrows():
        pt = r.geometry.representative_point()
        ax.text(pt.x, pt.y, r["name"], fontsize=fontsize, color=color, ha="center", va="center")


# ---- 1. forecast -----------------------------------------------------------------------------------------------------
def fig_forecast(tab: dict[str, pd.DataFrame], rainy: pd.DataFrame, issued: str) -> Path:
    st = provinces()
    codes = list(WINDOWS) + ["season"]
    fig, axes = plt.subplots(1, len(codes), figsize=(14, 3.5))
    for ax, code in zip(axes, codes):
        t = tab[code]
        if code == "season":
            cls = {pc: seas5_class(t.loc[pc, "pct"], t.loc[pc, "r"], True) for pc in st.pcode}
        else:
            cls = {pc: seas5_class(t.loc[pc, "pct"], t.loc[pc, "r"], bool(rainy.loc[pc, code])) for pc in st.pcode}
        g = st.assign(cls=st.pcode.map(lambda pc: cls[pc][0]), skill=st.pcode.map(lambda pc: cls[pc][1]))
        g.plot(ax=ax, color=g.cls.map(SEAS5_FILL), edgecolor="none", lw=0)
        for mask, colour, hatch in ((g.cls == "low_skill", SEAS5_CROSS, "xxx"),
                                    ((g.skill == "mod") & (g.cls != "none"), SEAS5_HATCH_WHITE, "////"),
                                    ((g.skill == "mod") & (g.cls == "none"), SEAS5_HATCH_GREY, "////")):
            if mask.any():
                g[mask].plot(ax=ax, facecolor="none", edgecolor=colour, hatch=hatch, lw=0)
        _map_axes(ax)
        if (g.cls == "low_skill").any():
            g[g.cls == "low_skill"].boundary.plot(ax=ax, color="#c4d0d1", lw=0.6)
        ax.set_title(f"{SEASON_LBL} (season)" if code == "season" else win_label(code), fontsize=9.5,
                     fontweight="bold" if code == "season" else "normal")
    fc = [Patch(facecolor=SEAS5_FILL[k], label=lab) for k, lab in (
        ("vsev_dry", "Driest 10% of forecasts"), ("sev_dry", "Driest third"), ("none", "Middle third"),
        ("sev_wet", "Wettest third"), ("vsev_wet", "Wettest 10%"), ("off_season", "Outside rainy season"))]
    skill = [Patch(facecolor="#ffffff", edgecolor="#c4d0d1", label="High (r ≥ 0.5) — solid"),
             Patch(facecolor="#ffffff", edgecolor=SEAS5_HATCH_GREY, hatch="////", label="Moderate (0.3–0.5) — hatched"),
             Patch(facecolor="#ffffff", edgecolor=SEAS5_CROSS, hatch="xxx", label="Low (< 0.3) — not shown")]
    fig.legend(handles=fc, loc="lower center", ncols=6, fontsize=7.5, bbox_to_anchor=(0.5, 0.1), title="Forecast", title_fontsize=8)
    fig.legend(handles=skill, loc="lower center", ncols=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04), title="Skill", title_fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.3, wspace=0.03)
    n = int(tab["season"]["n"].iloc[0])
    _source(fig, f"ECMWF SEAS5, issued {issued}: where this year's ensemble-mean forecast ranks among the {n} hindcasts for the same "
                 f"months ({Y0}–{Y1}), by province, detrended and classed as in the team's seas5-skill viewer. Skill = hindcast "
                 "correlation with ERA5. Outside rainy season: the three months bring under 15% of the year's rain (ERA5). The "
                 f"{SEASON_LBL} map is the five-month total and is shown for every province. Source: ECMWF SEAS5 and ERA5, team "
                 "admin-1 statistics.", y=-0.07)
    return _save(fig, "forecast")


# ---- 2. rain so far --------------------------------------------------------------------------------------------------
PENTAD_DIR = CACHE / "chirps_pentads"


PENTAD_URL = {"final": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/pentads/africa/tifs/chirps-v3.0.{y}.10.{p}.tif",
              "prelim": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/prelim/pentads/africa/tifs/chirps-v3.0.{y}.10.{p}.tif"}


def _pentad(y: int, p: int, prelim: bool) -> bool:
    """One October pentad cut to the Angola box and cached as npz; preliminary files are re-fetched once a day (CHC revises
    them). False when CHC does not have it yet."""
    import rasterio
    f = PENTAD_DIR / f"{y}_{p}.npz"
    if f.exists() and not (prelim and _stale(f, 1)):
        return True
    for attempt in range(3):   # CHC drops the odd request
        try:
            r = requests.get(PENTAD_URL["prelim" if prelim else "final"].format(y=y, p=p), timeout=300)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            return f.exists()
        if r.status_code == 200 and r.content[:4] in (b"II*\x00", b"MM\x00*"):
            break
    else:
        return f.exists()
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        win = rasterio.windows.from_bounds(11.0, -18.5, 24.5, -4.0, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype("float32")
        t = src.window_transform(win)
    a[a < 0] = np.nan
    PENTAD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, data=a, transform=np.array(t)[:6])
    return True


def october_to_date(cube: xr.DataArray) -> tuple[pd.DataFrame, int]:
    """1 October to the latest CHIRPS v3 pentad of October 2026 (preliminary) per unit, and the same pentads 1981–2025
    (final)."""
    npent = 0
    for p in range(1, 7):
        if not _pentad(NOW, p, prelim=True):
            break
        npent = p
    if npent == 0:
        raise ValueError("no October 2026 CHIRPS pentads yet")
    for y in range(Y0, NOW):
        for p in range(1, npent + 1):
            if not _pentad(y, p, prelim=False):
                raise ValueError(f"CHIRPS final pentad {y}-10-{p} not available")
    years = {}
    for f in sorted(PENTAD_DIR.glob("*_*.npz")):
        y, p = (int(x) for x in f.stem.split("_"))
        years.setdefault(y, {})[p] = f
    out = {}
    w = _weights(cube, unit_polygons())
    for y, ps in sorted(years.items()):
        if any(p not in ps for p in range(1, npent + 1)):
            continue
        tot = None
        for p in range(1, npent + 1):
            z = np.load(ps[p])
            t = z["transform"]
            a = z["data"]
            lat = t[5] + t[4] * (np.arange(a.shape[0]) + 0.5)
            lon = t[2] + t[0] * (np.arange(a.shape[1]) + 0.5)
            tot = a if tot is None else tot + a
        mean, frac = _to_cube_grid(tot, lat, lon, cube.to_dataset(name="precip"))
        v = mean.where(frac > 0).sortby("lat").values[None]
        out[y] = {u: float(_area_mean(v, wu)[0]) for u, wu in w.items()}
    return pd.DataFrame(out).T, npent


EARLY_MM = 40


def so_far(mon: pd.DataFrame, cube: xr.DataArray) -> dict:
    """Rain since 1 September 2026 (CHIRPS final September + prelim October to date) against the same dates since 1981,
    and the last full season."""
    octd, npent = october_to_date(cube)
    sep = mon[mon.index.month == 9].copy()
    sep.index = sep.index.year
    tot = (sep.reindex(octd.index) + octd).dropna(how="all")
    if NOW not in tot.index or tot.loc[NOW].isna().any():
        raise ValueError("September 2026 CHIRPS missing")
    normal = tot.loc[1991:2020].mean()
    share = tot.loc[NOW] / normal * 100
    rank = pct_rank(tot, NOW)
    # under 40 mm normally falls on these dates: before the rains, percentages of so little mean little; left out
    early = normal < EARLY_MM
    last = season_totals(mon, SEASON)
    return dict(share=share, rank=rank, rp=rp_dry(tot, NOW), early=early, normal=normal, end_day=31 if npent == 6 else 5 * npent,
                last_rank=pct_rank(last, NOW - 1, Y0, Y1 - 1), last_share=last.loc[NOW - 1] / last.loc[1991:2020].mean() * 100,
                last_rp=rp_dry(last, NOW - 1, Y0, Y1 - 1))


def fig_now(sf: dict, veg: pd.DataFrame) -> Path:
    st = provinces()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    bounds = [0, 25, 50, 75, 90, 110, 125, 150, 300]
    cmap = mcolors.ListedColormap(["#7a4a12", "#b77a35", "#e2c48e", "#efe3cc", "#f0efec", "#cfe0f5", "#9ec5f4", "#3987e5"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    panels = [(axes[0], sf["last_share"], None, f"Last season, {SEASON_LBL} {season_label(NOW - 1)}\n% of normal (rank in brackets)",
               sf["last_rank"]),
              (axes[1], sf["share"], sf["early"], f"This season so far: 1 Sep – {sf['end_day']} Oct 2026\n% of normal (rank in brackets)",
               sf["rank"])]
    for ax, val, early, title, rank in panels:
        g = st.assign(v=st.pcode.map(val), r=st.pcode.map(rank))
        g.plot(ax=ax, column="v", cmap=cmap, norm=norm, edgecolor="none")
        if early is not None:
            m = st.pcode.map(early).astype(bool)
            st[m].plot(ax=ax, facecolor="#ffffff", edgecolor="#c9c8c2", hatch="////", lw=0)
        _map_axes(ax)
        for _, r in g.iterrows():
            if early is not None and bool(early[r.pcode]):
                continue
            pt = r.geometry.representative_point()
            ax.text(pt.x, pt.y, f"{r.v:.0f}%\n({r.r:.0f})", fontsize=6.3, ha="center", va="center", color=TEXT)
        ax.set_title(title, fontsize=9)
    # vegetation: latest ASAP biomass on cropland
    ax = axes[2]
    g = st.assign(v=st.pcode.map(veg["value"]))
    vb = [-3, -1.5, -1, -0.5, 0.5, 1, 1.5, 3]
    vc = mcolors.ListedColormap(["#7a4a12", "#b77a35", "#e2c48e", "#f0efec", "#bfe0c4", "#7fbf8a", "#2e8b57"])
    g.plot(ax=ax, column="v", cmap=vc, norm=mcolors.BoundaryNorm(vb, vc.N), edgecolor="none")
    _map_axes(ax)
    for _, r in g.iterrows():
        pt = r.geometry.representative_point()
        ax.text(pt.x, pt.y, f"{r.v:+.1f}", fontsize=6.3, ha="center", va="center", color=TEXT)
    ax.set_title(f"Crop biomass, end of last season\n(ASAP zFPARc, {veg['date'].iloc[0]:%d %b %Y})", fontsize=9)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], orientation="horizontal", fraction=0.04, pad=0.02,
                      aspect=50)
    cb.set_ticks(bounds[1:-1])
    cb.set_label("% of the 1991–2020 normal for the same dates", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(vb, vc.N), cmap=vc), ax=axes[2], orientation="horizontal",
                       fraction=0.04, pad=0.02)
    cb2.set_ticks(vb[1:-1])
    cb2.set_label("z-score (below −1: ASAP warning level)", fontsize=8)
    cb2.ax.tick_params(labelsize=7.5)
    DRY = Patch(facecolor="#ffffff", hatch="////", edgecolor="#c9c8c2", lw=0, label=f"rains not started\n(normal under\n{EARLY_MM} mm)")
    axes[1].legend(handles=[DRY], loc="upper right", bbox_to_anchor=(1.02, 1.0), fontsize=6.8, frameon=False,
                   handlelength=1.5)
    _source(fig, f"Rank = percentile among the past seasons ({Y0}–{NOW - 1}; {Y0}–{NOW - 2} for last season) for the same dates (0 = driest). CHIRPS v3: final through September, "
                 f"preliminary pentads for 1–{sf['end_day']} October 2026; same pentads in past years. Biomass: JRC ASAP cumulative "
                 "FPAR anomaly over cropland during the growing cycle, GAUL admin-1.", y=-0.06, width=180)
    return _save(fig, "now")


# ---- 3. how much is El Niño -----------------------------------------------------------------------------------------
def _cell_seasons(da: xr.DataArray, months: list[int]) -> pd.DataFrame:
    """Seasonal totals per 0.5° cell, (season year × cell)."""
    v = da.values.reshape(da.time.size, -1)
    df = pd.DataFrame(v, index=pd.DatetimeIndex(da.time.values))
    return season_totals(df, months)


def fig_enso_maps(ph: pd.DataFrame) -> tuple[Path, dict]:
    """Row 1: correlation of seasonal rain with the Sep–Nov ONI (CHIRPS); row 2: ERA5 (sign check); row 3: how often El
    Niño seasons fell in the driest / wettest third (CHIRPS, FEWS NET's scale)."""
    das = {"CHIRPS v3": rain("chirps_v3"), "ERA5": rain("era5")}
    cols = list(WINDOWS) + ["season"]
    nino = phase_years(ph, "nino")
    son = ph.oni
    fig, axes = plt.subplots(3, len(cols), figsize=(14, 8.4))
    rnorm = mcolors.BoundaryNorm([-1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1], len(DIV))
    rcmap = mcolors.ListedColormap(DIV)
    FB = [-100] + [-b + 1e-9 for b in (80, 70, 60, 50, 40)] + [40, 50, 60, 70, 80, 100]
    FC = mcolors.ListedColormap(["#84540c", "#b17a28", "#d5a45f", "#eacfa0", "#f6ebcb", "#ffffff",
                                 "#ceecea", "#9bdad1", "#6ab2af", "#2c8d86", "#08615d"])
    FN = mcolors.BoundaryNorm(FB, FC.N)
    agree, share_neg = [], {}
    ann = das["CHIRPS v3"].sel(time=slice("1991", "2020")).groupby("time.year").sum().mean("year").values
    for j, code in enumerate(cols):
        months = SEASON if code == "season" else WINDOWS[code]
        rs = {}
        for i, (lab, da) in enumerate(das.items()):
            t = _cell_seasons(da, months).loc[Y0:Y1]
            x = son.reindex(t.index).values
            r = t.apply(lambda c: np.corrcoef(c.values, x)[0, 1] if c.notna().all() and c.std() > 0 else np.nan).values
            r = r.reshape(da.lat.size, da.lon.size)
            normal = t.loc[1991:2020].mean().values.reshape(da.lat.size, da.lon.size)
            dry = normal < 0.15 * ann * (len(months) / 3)   # the window brings under 15% of the year's rain (per 3 months)
            inside = _clip(da)
            r = np.where(inside & ~dry, r, np.nan)
            rs[lab] = r
            ax = axes[i, j]
            ax.pcolormesh(da.lon, da.lat, r, cmap=rcmap, norm=rnorm, shading="nearest")
            dm = (inside & dry).astype(float)
            if dm.any():
                ax.pcolor(_edges(da.lon.values), _edges(da.lat.values), np.ma.masked_where(dm == 0, dm),
                          cmap=mcolors.ListedColormap(["#ffffff"]), hatch="////", lw=0)
            _map_axes(ax)
            if i == 0:
                ax.set_title(f"{SEASON_LBL} (season)" if code == "season" else win_label(code), fontsize=9.5,
                             fontweight="bold" if code == "season" else "normal")
            if j == 0:
                ax.text(-0.04, 0.5, f"{lab}\ncorrelation\nwith ONI", transform=ax.transAxes, ha="right", va="center", fontsize=9,
                        fontweight="bold")
            if lab == "CHIRPS v3":
                tc = t
                inside_c, dry_c = inside, dry
        a, b = rs["CHIRPS v3"], rs["ERA5"]
        m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) >= 0.2)
        agree.append((np.sign(a[m]) == np.sign(b[m])).mean() * 100 if m.any() else np.nan)
        # frequency of El Niño seasons in the driest / wettest third (CHIRPS)
        ref = tc.loc[1991:2020]
        lo, hi = ref.quantile(1 / 3), ref.quantile(2 / 3)
        en = tc.loc[nino]
        fdry = (en.le(lo)).mean() * 100
        fwet = (en.ge(hi)).mean() * 100
        f = np.where(fdry >= fwet, -fdry, fwet).reshape(a.shape)
        f = np.where(inside_c & ~dry_c, f, np.nan)
        ax = axes[2, j]
        ax.pcolormesh(das["CHIRPS v3"].lon, das["CHIRPS v3"].lat, f, cmap=FC, norm=FN, shading="nearest")
        dm = (inside_c & dry_c).astype(float)
        if dm.any():
            ax.pcolor(_edges(das["CHIRPS v3"].lon.values), _edges(das["CHIRPS v3"].lat.values), np.ma.masked_where(dm == 0, dm),
                      cmap=mcolors.ListedColormap(["#ffffff"]), hatch="////", lw=0)
        _map_axes(ax)
        if j == 0:
            ax.text(-0.04, 0.5, f"El Niño\n{len(nino)} seasons\n(CHIRPS)", transform=ax.transAxes, ha="right", va="center", fontsize=9,
                    fontweight="bold")
        ok = np.isfinite(a)
        share_neg[code] = float((a[ok] < 0).mean() * 100)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.08, wspace=0.03, hspace=0.06)
    p1, p3 = axes[0, -1].get_position(), axes[1, -1].get_position()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap=rcmap), cax=fig.add_axes([0.895, p3.y0 + 0.01, 0.012, p1.y1 - p3.y0 - 0.02]))
    cb.set_ticks([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("correlation with the Sep–Nov ONI: brown = drier in El Niño", fontsize=8)
    p = axes[2, -1].get_position()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=FN, cmap=FC), cax=fig.add_axes([0.895, p.y0 + 0.005, 0.012, p.height - 0.01]))
    ticks = [-80, -60, -40, 40, 60, 80]
    cb.set_ticks(ticks, labels=[f"{abs(t)}%" for t in ticks])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("% of El Niño seasons in the\ndriest (brown) / wettest (green) third", fontsize=7.5)
    DRY = Patch(facecolor="#ffffff", hatch="////", edgecolor="#c9c8c2", lw=0, label="dry season (under 15% of the year's rain)")
    fig.legend(handles=[DRY], loc="lower right", bbox_to_anchor=(0.99, 0.03), fontsize=7.5, frameon=False)
    _source(fig, f"Seasons {Y0}–{Y1}, each classed by the Sep–Nov ONI of the year it starts ({len(nino)} El Niño seasons; strong "
                 f"≥ +1.5 °C: {', '.join(season_label(y) for y in phase_years(ph, 'nino', True))}). r² is the share of the year-to-year "
                 "variation that goes with ENSO (r = 0.5 → 25%). Third row: white = neither driest nor wettest third reaches 40% "
                 "(chance is 33%), FEWS NET's scale. Sources: CHIRPS v3 and ERA5 at 0.5°, NOAA CPC ONI.", y=0.055, width=190)
    v = dict(era5_agree=f"{np.nanmin(agree):.0f}–{np.nanmax(agree):.0f}", season_neg=f"{share_neg['season']:.0f}")
    return _save(fig, "enso_maps"), v


def enso_by_unit(mon: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    """Per unit: correlation of the season's rain with the Sep–Nov ONI, and El Niño seasons in the driest / wettest third."""
    t = season_totals(mon, SEASON).loc[Y0:Y1]
    x = ph.oni.reindex(t.index)
    ref = t.loc[1991:2020]
    lo, hi = ref.quantile(1 / 3), ref.quantile(2 / 3)
    nino, strong = phase_years(ph, "nino"), phase_years(ph, "nino", True)
    return pd.DataFrame(dict(r=t.apply(lambda c: np.corrcoef(c, x)[0, 1]),
                             nino_dry=t.loc[nino].le(lo).sum(), nino_wet=t.loc[nino].ge(hi).sum(), n_nino=len(nino),
                             strong_dry=t.loc[strong].le(lo).sum(), n_strong=len(strong)))


def clim_pos(ref: pd.Series, x: float) -> float:
    """Position of x in the 1991–2020 distribution, 0–1, the inverse of the (linear) quantile used for the terciles, so
    x is in the driest third exactly when the result is ≤ 1/3."""
    q = np.linspace(0, 1, 1001)
    return float(np.interp(x, np.quantile(ref.dropna().values, q), q))


def fig_strong(ss: dict[str, dict], tot: pd.DataFrame, ph: pd.DataFrame, issued: str) -> tuple[Path, dict]:
    """Left: the October SEAS5 forecast of the season in past strong El Niño years and now; right: the rain that came."""
    strong = phase_years(ph, "nino", True)
    units = list(provinces().pcode) + list(GROUPS) + ["AO"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.27 * len(units) + 1.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    y = np.arange(len(units))[::-1]
    markers = dict(zip(strong, ["o", "s", "D", "^", "v", "P"]))
    for yi, u in zip(y, units):
        s = ss[u]
        h = s["f"].loc[s["hist"]]
        for yr in strong:
            fp = 100 * (h <= h.loc[yr]).sum() / len(h)
            axes[0].scatter(fp, yi, marker=markers[yr], s=22, facecolor="none", edgecolor=STRONG_C, lw=0.9, zorder=3)
        axes[0].scatter(s["pct"], yi, marker="*", s=90, color=NOW_C, zorder=4)
        t = tot[u]
        for yr in strong:
            op = 100 * clim_pos(t.loc[1991:2020], t.loc[yr])
            axes[1].scatter(op, yi, marker=markers[yr], s=22, facecolor="none", edgecolor=STRONG_C, lw=0.9, zorder=3)
    for ax, title in ((axes[0], f"October SEAS5 forecast of {SEASON_LBL}\n(rank among forecasts, %)"),
                      (axes[1], f"{SEASON_LBL} rain that followed\n(CHIRPS, position in 1991–2020, %)")):
        ax.axvspan(0, 100 / 3, color="#f6ebcb", zorder=0)
        ax.axvspan(200 / 3, 100, color="#e3eef9", zorder=0)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 33, 50, 67, 100])
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y, [unit_name(u) for u in units], fontsize=8)
    for lab, u in zip(axes[0].get_yticklabels(), units):
        if u in GROUPS or u == "AO":
            lab.set_fontweight("bold")
    # right: the 2026 forecast anomaly and the part SEAS5's hindcasts associate with the July–September ONI
    ax = axes[2]
    oj = oni_season("JAS")
    parts = {}
    for yi, u in zip(y, units):
        s = ss[u]
        h = s["f"].loc[s["hist"]]
        z = (h - h.mean()) / h.std(ddof=1)
        x = oj.reindex(h.index)
        b, a = np.polyfit(x.values, z.values, 1)
        part = a + b * oj[NOW]
        parts[u] = dict(z=s["z"], enso=part, r_oni=float(np.corrcoef(x, z)[0, 1]))
        ax.barh(yi + 0.17, s["z"], height=0.32, color="#5e6a6b")
        ax.barh(yi - 0.17, part, height=0.32, color=NINO_C)
    ax.axvline(0, color=TEXT, lw=0.6)
    ax.set_title("2026 forecast anomaly (SD) and the part\nthat goes with El Niño in SEAS5's hindcasts", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(handles=[Patch(color="#5e6a6b", label="2026 forecast"), Patch(color=NINO_C, label=f"ENSO part (JAS ONI {oj[NOW]:+.1f})")],
              fontsize=7, loc="lower left", frameon=False)
    h = [Line2D([], [], marker=markers[yr], ls="", mfc="none", mec=STRONG_C, label=season_label(yr)) for yr in strong]
    h.append(Line2D([], [], marker="*", ls="", color=NOW_C, ms=10, label=f"{season_label(NOW)} (forecast)"))
    fig.legend(handles=h, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.45, 0.0), frameon=False, ncols=len(h))
    fig.subplots_adjust(wspace=0.08, top=0.9, bottom=0.1)
    _source(fig, f"Strong El Niño = Sep–Nov ONI ≥ +1.5 °C. Shading: driest third (sand), wettest third (blue). Left: October "
                 f"issuances, detrended as in the seas5-skill viewer, ranked among {Y0}–{Y1}. Right: forecast z-score of the "
                 f"{season_label(NOW)} season ({issued}) and a straight-line fit of the {Y0}–{Y1} hindcast z-scores on the "
                 f"Jul–Sep ONI evaluated at {oj[NOW]:+.2f} °C, above any year in the hindcast record "
                 f"(max {oj.loc[Y0:Y1].max():+.2f}): an extrapolation.", y=0.0, width=185)
    return _save(fig, "strong"), parts


# ---- the south's record: rain, vegetation, CERF and FEWS NET ---------------------------------------------------------
def fig_record(tot: pd.DataFrame, veg_season: pd.DataFrame, ph: pd.DataFrame, cf: pd.DataFrame, pop: pd.DataFrame,
               ipc_ph3: pd.DataFrame, key: str) -> tuple[Path, dict]:
    lbl = GROUPS[key]["label"]
    t = tot[key].loc[Y0:Y1]
    share = t / t.loc[1991:2020].mean() * 100
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 0.9, 1.1], hspace=0.12))
    ax = axes[0]
    ax.bar(share.index, share.values - 100, bottom=100, color=[phase_color(y, ph) for y in share.index], width=0.8)
    ax.axhline(100, color=TEXT, lw=0.6)
    lo = t.loc[1991:2020].quantile(1 / 3) / t.loc[1991:2020].mean() * 100
    ax.axhline(lo, color=TEXT2, lw=0.7, ls="--")
    ax.text(Y0 - 0.6, lo, "driest third", fontsize=7, color=TEXT2, va="bottom")
    ax.set_ylabel(f"{SEASON_LBL} rain\n% of normal", fontsize=8.5)
    ax.set_title(f"{lbl}: {SEASON_LBL} rain, crop biomass, CERF drought allocations and FEWS NET peak needs, by season", fontsize=10,
                 loc="left")
    hh = [Patch(color=STRONG_C, label="strong El Niño"), Patch(color=NINO_C, label="El Niño"), Patch(color=NEUTRAL_C, label="neutral"),
          Patch(color=NINA_C, label="La Niña")]
    ax.legend(handles=hh, fontsize=7.5, ncols=4, loc="upper left", frameon=False)
    ax = axes[1]
    v = veg_season[key].dropna()
    ax.bar(v.index, v.values, color=[phase_color(y, ph) for y in v.index], width=0.8)
    ax.axhline(0, color=TEXT, lw=0.6)
    ax.axhline(-1, color=TEXT2, lw=0.7, ls="--")
    ax.text(Y0 - 0.6, -1, "ASAP warning", fontsize=7, color=TEXT2, va="top")
    ax.set_ylabel("crop biomass\nend of season (z)", fontsize=8.5)
    ax = axes[2]
    for _, r in pop.iterrows():
        ax.plot([r.sy, r.sy], [r.low_value / 1e6, r.high_value / 1e6], color="#E67800", lw=5, solid_capstyle="butt")
    for _, r in ipc_ph3.iterrows():
        ax.scatter([r.sy], [r.value / 1e6], marker="D", s=26, color="#C80000", zorder=4)
    ax.set_ylabel("people in IPC\nPhase 3+ (millions)", fontsize=8.5)
    ax2 = ax.twinx()
    for i, (_, r) in enumerate(cf.sort_values("erc_endorsement_date").iterrows()):
        ax2.scatter([r.sy], [r.amount_approved / 1e6], marker="v", s=46, color=TEXT, zorder=5)
        ax2.annotate(f"{pd.Timestamp(r.erc_endorsement_date):%b %Y}", (r.sy, r.amount_approved / 1e6),
                     xytext=(7, 0), va="center", ha="left",
                     textcoords="offset points", fontsize=6.3, color=TEXT2)
    ax2.set_ylim(0, 9)
    ax2.set_ylabel("CERF drought\nallocation (US$ M)", fontsize=8.5)
    ax.set_ylim(0, max(5.2, pop.high_value.max() / 1e6 + 0.3) if len(pop) else 5)
    ax.legend(handles=[Line2D([], [], color="#E67800", lw=5, label="FEWS NET peak needs, Phase 3+ (range, national)"),
                       Line2D([], [], marker="D", ls="", color="#C80000", label="IPC analysis, Phase 3+ (areas analysed)"),
                       Line2D([], [], marker="v", ls="", color=TEXT, label="CERF drought allocation (right axis)")],
              fontsize=7.5, loc="upper left", frameon=False)
    for a in axes:
        a.tick_params(labelsize=8)
        a.grid(axis="y", color=GRID, lw=0.5)
        a.set_axisbelow(True)
        for s in ("top",):
            a.spines[s].set_visible(False)
    yrs = list(range(Y0, NOW + 1, 5))
    axes[-1].set_xticks(yrs, [season_label(y) for y in yrs])
    axes[-1].set_xlim(Y0 - 1, NOW + 0.8)
    _source(fig, f"Seasons labelled by the year they start ({season_label(2015)} = Nov 2015 – Mar 2016), coloured by the Sep–Nov ONI. "
                 f"Rain: CHIRPS v3, area mean of {lbl}. Biomass: JRC ASAP zFPARc on cropland, last dekad of April, area-weighted "
                 "over the same provinces. FEWS NET peak needs: the year's highest national Phase 3+ estimate (FEWS NET Data "
                 "Warehouse), placed on the season under way or last ended at the middle of its window. IPC: the two Angola analyses "
                 "(2019, 2021; southern provinces only). CERF: allocations coded Drought (OneGMS), placed on the season whose failed "
                 "rains they responded to.", y=0.03, width=175)
    return _save(fig, "record"), {}


def fig_fews(fews: dict, pop: pd.DataFrame) -> Path:
    st = provinces()
    order = [sc for sc in ("CS", "ML2") if sc in fews["picks"]]
    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.8))
    gdf = fews["gdf"].to_crs(st.crs)
    for ax, sc in zip(np.atleast_1d(axes), order):
        pk = fews["picks"][sc]
        g = gdf.assign(phase=gdf.fnid.map(pk["phase"]))
        if g.phase.isna().any():
            g[g.phase.isna()].plot(ax=ax, color="#ececec", edgecolor="white", lw=0.3)
        for ph_, col in IPC_COLOURS.items():
            sub = g[g.phase == ph_]
            if len(sub):
                sub.plot(ax=ax, color=col, edgecolor="white", lw=0.3)
        _map_axes(ax)
        st.boundary.plot(ax=ax, color=TEXT2, lw=0.5)
        lbl = {"CS": "Current situation", "ML1": "Projection", "ML2": "Projection"}[sc]
        a, b = pd.Timestamp(pk["start"]), pd.Timestamp(pk["end"])
        win = f"{a:%b %Y}" if (a.year, a.month) == (b.year, b.month) else f"{a:%b %Y} – {b:%b %Y}"
        ax.set_title(f"{lbl}, {win}\n(FEWS NET, {pd.Timestamp(pk['round'] + '-01'):%B %Y})", fontsize=9)
    fig.legend(handles=[Patch(color=IPC_COLOURS[k], label=IPC_LABELS[k]) for k in range(1, 6)] + [Patch(color="#ececec", label="not classified")],
               frameon=False, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.02, bottom=0.08)
    _source(fig, "FEWS NET acute food insecurity classification (IPC-compatible), on FEWS NET's own livelihood-zone × admin units; grey "
                 "lines: provinces. Via the team's FEWS NET mirror.", y=-0.03, width=170)
    return _save(fig, "fews")


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def _fmt_rp(row) -> str:
    return f"1 in {row.rp_dry:.0f} dry" if row.pct < 50 else f"1 in {row.rp_wet:.0f} wet"


def province_table(tab: dict[str, pd.DataFrame], en: pd.DataFrame, sf: dict, parts: dict, fews_max: pd.Series) -> str:
    units = list(provinces().pcode) + list(GROUPS) + ["AO"]
    t = tab["season"]
    head = ("<tr><th>Province</th>"
            f"<th class='num'>Forecast {SEASON_LBL}<br>(return period)</th><th class='num'>Skill r</th>"
            "<th class='num'>2026 forecast (SD) / ENSO part<br>(hindcast r with ONI)</th>"
            "<th class='num'>El Niño seasons<br>driest / wettest third</th><th class='num'>Rain–ONI r</th>"
            f"<th class='num'>{SEASON_LBL} {season_label(NOW - 1)}<br>% of normal</th>"
            f"<th class='num'>1 Sep–{sf['end_day']} Oct 2026<br>% of normal</th>"
            "<th>FEWS NET highest phase<br>Oct 2026–Jan 2027</th></tr>")
    rows = []
    for u in units:
        r, e, p = t.loc[u], en.loc[u], parts[u]
        cls = ' class="hl"' if u in GROUPS or u == "AO" else ""
        early = bool(sf["early"].get(u, False))
        so = "—" if early else f"{sf['share'][u]:.0f}%"
        rows.append(f"<tr{cls}><td>{html.escape(unit_name(u))}</td><td class='num'>{_fmt_rp(r)}</td>"
                    + (f"<td class='num' style='color:#9aa3ad'>{r.r:.2f} (low)</td>" if r.r < SEAS5_T["r_mod"]
                       else f"<td class='num'>{r.r:.2f}</td>")
                    + f"<td class='num'>{p['z']:+.1f} / {p['enso']:+.1f} ({p['r_oni']:+.2f})</td>"
                    f"<td class='num'>{e.nino_dry:.0f} / {e.nino_wet:.0f} of {e.n_nino:.0f}</td><td class='num'>{e.r:+.2f}</td>"
                    f"<td class='num'>{sf['last_share'][u]:.0f}%</td>"
                    f"<td class='num'>{so}</td>"
                    f"<td>{fews_max.get(u, '—')}</td></tr>")
    return f'<div style="overflow-x:auto"><table class="skill">{head}{"".join(rows)}</table></div>'


def fews_by_province(fews: dict) -> pd.Series:
    """Highest FEWS NET phase (ML2) among the units covering at least 10% of each province or group."""
    st = unit_polygons()
    pk = fews["picks"]["ML2"]
    g = fews["gdf"].to_crs(st.crs).assign(phase=lambda d: d.fnid.map(pk["phase"])).dropna(subset=["phase"])
    eq = "ESRI:102022"
    gi = gpd.overlay(st.to_crs(eq)[["pcode", "geometry"]], g.to_crs(eq)[["fnid", "phase", "geometry"]], how="intersection")
    gi["a"] = gi.area
    tot = st.to_crs(eq).set_index("pcode").area
    gi = gi[gi.a / gi.pcode.map(tot) >= 0.10]
    best = gi.groupby("pcode").phase.max()
    return best.map(lambda x: IPC_LABELS[int(x)])


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
EXTRA_CSS = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0 8px}
.tile{border:1px solid #e2e7e7;border-top:3px solid #b5541b;border-radius:6px;padding:12px 14px;background:#fff}
.tile .big{font-size:24px;font-weight:700;margin:0 0 4px;line-height:1.15}
.tile p{font-size:13px;color:var(--n8);margin:0}
section h3{font-size:16px;line-height:1.4}
.note{font-size:13px;color:var(--n7);margin:-6px 0 14px;max-width:95ch}
details{margin:24px 0}details summary{cursor:pointer;font-weight:600}
details li{font-size:13.5px;margin:0 0 8px;max-width:90ch}
.wrap{max-width:1120px}
table.skill td{white-space:nowrap}table.skill th{font-size:11.5px;vertical-align:bottom}
table.lit td{font-size:13px;vertical-align:top}table.lit th{font-size:11.5px}table.lit td:first-child{white-space:nowrap}
"""


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def analyse(spec: dict) -> tuple[dict, dict]:
    set_groups(spec)
    im = int(spec.get("issued_month", 10))
    ph = phases()
    mon = prov_monthly("chirps_v3")
    cube = rain("chirps_v3")
    v: dict = {}
    f: dict = {}
    v |= oni_now()
    set_issued(im)
    s5 = seas5_adm()
    issued_ts = s5[s5.issued_date.dt.month == im].issued_date.max()
    issued = f"{issued_ts:%B %Y}"
    tab = seas5_table(im)
    rainy = seas5_rainy()
    f["forecast"] = fig_forecast(tab, rainy, issued)
    sf = so_far(mon, cube)
    a = asap_biomass(1)
    # end of last season = last dekad of April 2026, kept beside the latest dekad
    eos = a[(a.date.dt.month == 4) & (a.date.dt.day == 21)]
    eos_last = eos[eos.date.dt.year == NOW].set_index("pcode")[["value", "date"]]
    f["now"] = fig_now(sf, eos_last)
    x, vv = fig_enso_maps(ph)
    f["enso_maps"] = x
    v |= vv
    tot = season_totals(mon, SEASON)
    en = enso_by_unit(mon, ph)
    ss = seas5_series(im, SEASON)
    f["strong"], parts = fig_strong(ss, tot, ph, issued)
    # CERF drought allocations → the season whose failed rains they responded to (hand-checked in the TOML)
    cf = cerf()
    cf = cf[cf.emergency_type == "Drought"].copy()
    known = {k: int(x) for k, x in spec["cerf_season"].items()}
    if set(cf.application_code) != set(known):
        raise ValueError(f"CERF drought allocations to Angola changed: new {sorted(set(cf.application_code) - set(known))}, "
                         f"gone {sorted(set(known) - set(cf.application_code))}: add their season to [cerf_season]")
    cf["sy"] = cf.application_code.map(known)
    # FEWS NET peak needs (national, Phase 3+) → the season under way or last ended at the middle of the peak window
    pop = fews_population()
    pk = pop[(pop.scenario_name == "Peak Needs") & (pop.phase.astype(str) == "3+")].copy()
    mid = pk.projection_start + (pk.projection_end - pk.projection_start) / 2
    pk["sy"] = np.where(mid.dt.month >= 11, mid.dt.year, mid.dt.year - 1)
    pk = pk.sort_values("reporting_date").groupby("sy").tail(1)
    # IPC Phase 3+ (current period) per analysis
    ip = ipc()
    ip3 = ip[(ip.level == "national") & (ip.period_type == "current") & (ip.phase.astype(str) == "3+")][["analysis_date", "population"]].copy()
    ip3["sy"] = pd.to_datetime(ip3.analysis_date).dt.year - 1
    ip3 = ip3.rename(columns={"population": "value"})
    veg_season = eos.assign(sy=eos.date.dt.year - 1)
    km2 = provinces().set_index("pcode").km2
    vs = veg_season.pivot_table(index="sy", columns="pcode", values="value")
    for k, g in GROUPS.items():
        w = km2[g["pcodes"]]
        vs[k] = (vs[g["pcodes"]] * w).sum(1) / w.sum()
    key = spec["record_group"]
    f["record"], _ = fig_record(tot, vs, ph, cf, pk, ip3, key)
    fews = fews_classes()
    f["fews"] = fig_fews(fews, pop)
    fmax = fews_by_province(fews)
    table = province_table(tab, en, sf, parts, fmax)
    # ---- values for the narrative, with guards ------------------------------------------------------------------------
    S = tab["season"]
    sk = spec["headline_group"]
    s = S.loc[sk]
    nino_cerf = cf[cf.sy.map(lambda y: ph.phase.get(y) == "nino")]
    nina_cerf = cf[cf.sy.map(lambda y: ph.phase.get(y) == "nina")]
    ml2 = fews["picks"]["ML2"]
    n3 = sum(1 for p in ml2["phase"].values() if p >= 3)
    latest_pk = pop[(pop.scenario_name == "Peak Needs")].sort_values("reporting_date").iloc[-1]
    cs = pop[(pop.scenario_name == "Current Situation")].sort_values("reporting_date").iloc[-1]
    wet = [u for u in provinces().pcode if S.loc[u, "pct"] >= 90]
    dry = [u for u in provinces().pcode if S.loc[u, "pct"] <= 10]
    hp = parts[sk]
    v |= dict(
        issued=issued, fc_year=int(s.year), n_hind=int(s.n), hl_label=GROUPS[sk]["label"],
        hl_rp=f"{s.rp_dry:.0f}", hl_pct=f"{s.pct:.0f}", hl_r=f"{s.r:.2f}",
        hl_z=f"{hp['z']:+.1f}", hl_enso=f"{hp['enso']:+.1f}",
        n_dry=len(dry), dry_names=", ".join(unit_name(u) for u in dry), n_wet=len(wet), wet_names=", ".join(unit_name(u) for u in wet),
        nat_pct=f"{S.loc['AO', 'pct']:.0f}",
        dry_r_lo=f"{S.loc[dry, 'r'].min():.2f}", dry_r_hi=f"{S.loc[dry, 'r'].max():.2f}",
        wet_r_lo=f"{S.loc[wet, 'r'].min():.2f}", wet_r_hi=f"{S.loc[wet, 'r'].max():.2f}",
        raw_driest=int(S.loc[dry, "raw_driest"].sum()),
        so_end=sf["end_day"], last_season=season_label(NOW - 1),
        n_cerf=len(cf), n_cerf_nino=len(nino_cerf), n_cerf_nina=len(nina_cerf),
        cerf_nino_seasons=", ".join(season_label(y) + (f" ({n})" if n > 1 else "") for y, n in nino_cerf.sy.value_counts().sort_index().items()),
        cerf_nina_seasons=", ".join(season_label(y) + (f" ({n})" if n > 1 else "") for y, n in nina_cerf.sy.value_counts().sort_index().items()),
        n_cerf_nino_seasons=nino_cerf.sy.nunique(), n_cerf_nina_seasons=nina_cerf.sy.nunique(),
        fews_peak_round=f"{latest_pk.reporting_date:%B %Y}",
        fews_n3=n3, fews_units=len(ml2["phase"]), fews_round=f"{pd.Timestamp(ml2['round'] + '-01'):%B %Y}",
        fews_peak=latest_pk.description.replace(" - ", "–"), fews_peak_win=f"{latest_pk.projection_start:%b %Y} – {latest_pk.projection_end:%b %Y}",
        fews_cs=cs.description.replace(" - ", "–"), fews_cs_month=f"{cs.projection_start:%B %Y}",
        strong_years=", ".join(season_label(y) for y in phase_years(ph, "nino", True)),
        en_r_neg=", ".join(unit_name(u) for u in provinces().pcode if en.loc[u, "r"] <= -0.25),
        en_r_pos_n=int((en.loc[list(provinces().pcode), "r"] >= 0.25).sum()),
        hl_nino_dry=int(en.loc[sk, "nino_dry"]), hl_nino_wet=int(en.loc[sk, "nino_wet"]), n_nino=int(en.loc[sk, "n_nino"]),
        hl_en_r=f"{en.loc[sk, 'r']:+.2f}",
    )
    # guards: the headlines below were written against this pattern
    if s.pct > 10 or s.r >= SEAS5_T["r_mod"] + 0.15:
        raise ValueError(f"{sk}: the season forecast is no longer among the driest 10% with low skill: reword the forecast lines")
    if not wet or not dry:
        raise ValueError("the forecast no longer splits into a dry south and a wet north: reword")
    if hp["enso"] >= 0 or abs(hp["enso"]) >= abs(hp["z"]):
        raise ValueError("the ENSO part is no longer a minority share of the south's dry forecast: reword")
    v["hl_enso_share"] = f"{hp['enso'] / hp['z'] * 100:.0f}"
    started = [u for u in provinces().pcode if not bool(sf["early"][u])]
    ipcv = ip3.set_index(pd.to_datetime(ip3.analysis_date).dt.year)["value"]
    v |= dict(rp_cap=f"{s.n + 1:.0f}", last_south=f"{sf['last_share'][sk]:.0f}",
              so_lo=f"{sf['share'][started].min():.0f}", so_hi=f"{sf['share'][started].max():.0f}",
              ipc19=f"{ipcv[2019] / 1e3:,.0f} thousand", ipc21=f"{ipcv[2021] / 1e6:.1f} million",
              hind_years=f"{int(tab['season'].loc[sk, 'year']) - int(s.n)}–{int(tab['season'].loc[sk, 'year']) - 1}")
    south = GROUPS[sk]["pcodes"]
    v |= dict(dry_r_med=f"{S.loc[dry, 'r'].median():.2f}", wet_r_med=f"{S.loc[wet, 'r'].median():.2f}",
              nino_dry_by=", ".join(f"{unit_name(u)} {int(en.loc[u, 'nino_dry'])}" for u in sorted(south, key=lambda u: -en.loc[u, 'nino_dry'])),
              hl_r2=f"{en.loc[sk, 'r'] ** 2 * 100:.0f}")
    # past strong El Niño years: October forecast and outcome for the headline group, by province
    strong = phase_years(ph, "nino", True)
    def fpct(u, yr):
        h = ss[u]["f"].loc[ss[u]["hist"]]
        return 100 * (h <= h.loc[yr]).sum() / len(h)
    tt = tot.loc[Y0:Y1]
    opct = {yr: 100 * clim_pos(tot[sk].loc[1991:2020], tot.loc[yr, sk]) for yr in strong}
    if sum(o <= 100 / 3 for o in opct.values()) != int(en.loc[sk, "strong_dry"]):
        raise ValueError("strong El Niño driest-third count disagrees with the tercile count")
    v |= dict(n_strong=len(strong), strong_out_dry=sum(o <= 100 / 3 for o in opct.values()),
              strong_out_dry_years=", ".join(season_label(y) for y, o in opct.items() if o <= 100 / 3),
              unlike=", ".join(unit_name(u) for u in south if min(fpct(u, yr) for yr in strong) > 100 / 3),
              alike=", ".join(unit_name(u) for u in south if min(fpct(u, yr) for yr in strong) <= 10),
              alike_years=", ".join(season_label(yr) for yr in strong if all(fpct(u, yr) <= 10 for u in south
                                                                              if min(fpct(u, y2) for y2 in strong) <= 10)))
    # the headline group's driest seasons and their phases
    d5 = tt[sk].nsmallest(5)
    v |= dict(driest5=", ".join(f"{season_label(y)} ({ {'nino': 'El Niño', 'nina': 'La Niña', 'neutral': 'neutral'}[ph.phase[y]]})"
                                  for y in d5.index),
              driest5_nino=int(sum(ph.phase[y] == "nino" for y in d5.index)))
    # rain so far / last season
    lr = sf["last_rank"][list(provinces().pcode)]
    v |= dict(last_driest=", ".join(unit_name(u) for u in lr.index if lr[u] == 0),
              so_low=", ".join(f"{unit_name(u)} {sf['share'][u]:.0f}%" for u in started if round(sf["share"][u]) < 75),
              fews_p3_prov=", ".join(unit_name(u) for u in provinces().pcode if fmax.get(u) == IPC_LABELS[3]))
    if not set(u for u in provinces().pcode if fmax.get(u) == IPC_LABELS[3]) <= set(GROUPS[sk]["pcodes"]) | {"AO02"}:
        raise ValueError("FEWS NET Phase 3 areas now reach beyond the south: reword the food-security headline")
    skilled = [u for u in dry if S.loc[u, "r"] >= SEAS5_T["r_mod"]]
    unskilled = [u for u in dry if S.loc[u, "r"] < SEAS5_T["r_mod"]]
    v |= dict(sk_names=" and ".join(unit_name(u) for u in skilled), unsk_names=", ".join(unit_name(u) for u in unskilled),
              hl_r_oni=f"{hp['r_oni']:.2f}", early_names=", ".join(unit_name(u) for u in provinces().pcode if bool(sf["early"][u])))
    if not skilled:
        raise ValueError("no dry province with skill ≥ 0.3: reword the skill headline")
    if not 10 <= abs(hp["enso"] / hp["z"]) * 100 <= 40:
        raise ValueError("ENSO share outside 10–40%: reword the tile and the strong-section headline")
    if not 25 <= S.loc["AO", "pct"] <= 75:
        raise ValueError("the national forecast is no longer near the middle: reword 'the two cancel'")
    if set(ipcv.index) != {2019, 2021}:
        raise ValueError(f"IPC analyses for Angola changed ({sorted(ipcv.index)}): reword 'analysed twice'")
    return f, v | dict(table=table)


def literature_table(items: list[dict]) -> str:
    """Published studies on El Niño and Angola's rains, as listed in the TOML ([[literature]]): one row each, linked."""
    head = "<tr><th>Study</th><th>Area, season</th><th>Angola in it</th><th>Finding</th></tr>"
    rows = "".join(f"<tr><td><a href='{html.escape(x['url'])}'>{html.escape(x['cite'])}</a></td><td>{html.escape(x['area'])}</td>"
                   f"<td>{html.escape(x['angola'])}</td><td>{html.escape(x['finding'])}</td></tr>" for x in items)
    return f'<div style="overflow-x:auto"><table class="lit">{head}{rows}</table></div>'


def render(spec: dict, figs: dict, v: dict) -> str:
    import enso_deep_dive as edd
    pg = spec["page"]
    head = edd.HEAD.format(title=html.escape(pg["title"]), desc=html.escape(pg["sub"]), css=edd.CSS + EXTRA_CSS,
                           home="../", home_label="ENSO deep dives")
    out = [head, '<p class="eyebrow">El Niño 2026/27</p>',
           f'<h1>{pg["title"]}</h1><p class="meta">{pg["sub"].format(**v)}<br>{pg["dateline"].format(**v)}</p>']
    out.append('<div class="tiles">' + "".join(f'<div class="tile"><p class="big">{t["big"].format(**v)}</p><p>{t["text"].format(**v)}</p></div>'
                                                for t in spec["tiles"]) + "</div>")
    for sec in spec["sections"]:
        if sec.get("part"):
            out.append(f'<h2>{sec["part"]}</h2>')
        out.append(f'<section id="{sec["id"]}"><h3>{sec["head"].format(**v)}</h3>')
        if sec.get("note"):
            out.append(f'<p class="note">{sec["note"].format(**v)}</p>')
        for fig in [sec["fig"]] if isinstance(sec["fig"], str) else sec["fig"]:
            if fig == "table":
                out.append(v["table"])
            elif fig == "literature":
                out.append(literature_table(spec["literature"]))
            else:
                out.append(f'<figure><img src="{_img(figs[fig])}" alt="{html.escape(sec["head"].format(**v))}"></figure>')
        out.append("</section>")
    notes = "".join(f"<li>{n.format(**v)}</li>" for n in spec["notes"])
    out.append(f'<details><summary>About the data</summary><ul>{notes}</ul></details>')
    out.append(f'<p class="small">OCHA Centre for Humanitarian Data · built {pd.Timestamp.now():%Y-%m-%d}</p>')
    out.append(edd.FOOT)
    return "\n".join(out)


def write_index() -> None:
    import enso_deep_dive as edd
    specs = [tomllib.loads(p.read_text(encoding="utf-8")) | {"slug": p.stem} for p in sorted((ROOT / "deep_dives").glob("*.toml"))]
    (ROOT / "docs" / "enso" / "index.html").write_text(edd.render_index(specs, 0), encoding="utf-8")


def build(spec: dict | None = None, *_args, publish: bool = False) -> dict:
    spec = spec or tomllib.loads(SPEC.read_text(encoding="utf-8"))
    figs, v = analyse(spec)
    page = render(spec, figs, v)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"  wrote {OUT}/index.html")
    print("  values:", {k: x for k, x in v.items() if k != "table"})
    if publish:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "index.html").write_text(page, encoding="utf-8")
        write_index()
        print(f"  published -> {DOCS}/index.html and docs/enso/index.html")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="also write docs/enso/ago/ and the /enso/ card index")
    build(publish=ap.parse_args().publish)


if __name__ == "__main__":
    main()
