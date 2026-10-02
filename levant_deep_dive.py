"""Gaza — ENSO deep dive (bespoke builder for deep_dives/gaza.toml).

The generic builder in enso_deep_dive.py works on a country's ERA5 cells and a drought framing.
Gaza does not fit either: Gaza is 365 km², three 0.25° ERA5 cells, and the winter hazard is
too much rain, not too little. This module reuses the series' page chrome, SEAS5 skill figure and
pixel correlation maps, and adds what a Gaza page needs:

* four independent rainfall records, because one reanalysis cell is not enough evidence —
  ERA5 hourly point series for the three cells over Gaza (CDS time-series dataset, 1950–),
  GPCC gauge analysis (1° cell over Gaza and the southern coastal plain, 1891–), IMERG late
  v7 daily at 0.1° (team blob, 1998–), and the Beer Sheva rain gauge (GHCN-Daily, 1921–2016);
* a stationarity test, because the El Niño link in this region is known to switch on and off;
* daily metrics (rain days, heavy-rain days, wettest day, cold nights, wind) by ENSO phase;
* the last three winters' daily rainfall over Gaza against reported impacts (curated in the TOML).

    uv run python levant_deep_dive.py gaza   # or: uv run python enso_deep_dive.py --only gaza

Downloads are cached under cache/gaza/. CDS needs a key in ~/.cdsapirc (any ECMWF data store
URL; the CDS endpoint is forced here); IMERG and the COD-AB need the team's blob credentials.
"""
from __future__ import annotations

import html
import io
import re
import tomllib
import zipfile
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import colors as mcolors
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import signal, stats
from shapely.geometry import box

import enso_deep_dive as edd
import teleconnection_survey as ts

COD_CACHE = Path("cache/gaza")               # COD-AB polygons for the whole oPt (shared by both pages)
UTM = 32636                                   # metres, for overlap areas
REGION = (32.5, 29.5, 37.0, 34.0)             # map extent for the correlation maps: w, s, e, n


@dataclass
class Area:
    """Everything that differs between the Gaza and West Bank pages. Defaults are Gaza's; a TOML
    `[area]` table overrides any field. ERA5 cell weights are the share of the area's COD-AB polygon
    in each 0.25° cell, computed at run time; GPCC cells are weighted the same way at 1°."""
    slug: str = "gaza"
    name: str = "Gaza"
    pcode: str = "PS02"
    cache: str = "cache/gaza"
    era5_cells: dict = field(default_factory=lambda: {"n": (31.50, 34.50), "s": (31.25, 34.25), "nw": (31.50, 34.25)})
    gpcc_cells: list = field(default_factory=lambda: [(31.5, 34.5)])
    gpcc_desc: str = ("GPCC is a 1° cell (31–32°N, 34–35°E) that takes in the southern coastal plain and the north-western Negev "
                      "as well as Gaza; ")
    gpcc_title: str = "Gaza and the southern coastal plain"
    gpcc_label: str = "GPCC gauge analysis (1° cell)"
    era5_label: str = "ERA5 (3 cells over Gaza)"
    imerg_label: str = "IMERG late v7 (Gaza, 0.1°)"
    gauges: list = field(default_factory=lambda: [["IS000051690", "", ""]])   # GHCN-D ids spliced by date: [id, from, to]
    gauge_name: str = "Beer Sheva gauge"
    gauge_label: str = "Beer Sheva rain gauge"
    gauge_short: str = "Beer Sheva"
    meta_gauge: str = "Beer Sheva gauge 1921–2016"
    imerg_box: tuple = (33.95, 30.95, 34.85, 31.85)          # w, s, e, n
    imerg_split: tuple | None = (31.45, 31.35)                # north of / south of (latitude)
    seas5_box: tuple = (34.0, 31.0, 34.8, 31.8)
    era5_cold: bool = False                                    # ERA5 cells partly sea: no cold-night metric
    cold_thresh: float = 5.0
    stationarity_records: str = ("Two gauge-based records (the GPCC analysis, and the Beer Sheva gauge 46 km inland, which is one of GPCC\'s "
                                 "inputs) and a reanalysis that assimilates no rain gauges (ERA5).")
    daily_cold_note: str = "the Beer Sheva gauge only, because the ERA5 cells are partly sea and rarely get that cold."
    daily_note: str = ("ERA5 spreads rain over a 25 km cell and understates heavy days; IMERG is satellite-only; Beer Sheva is drier "
                       "and 46 km inland.")
    byron: bool = True                                         # Gaza's Byron-specific wording in section 5
    impact_caption: str = ("Hazard: R rain and flooding, S sea surge or high tide, W wind, C cold. Weather columns cover the window from the day "
                           "before the reported start to the reported end: IMERG wettest day over Gaza (north = cells at 31.4–31.6°N: North Gaza, "
                           "Gaza governorate and most of Deir al Balah; south = 31.2–31.4°N: Khan Younis and Rafah); window totals from IMERG and from ERA5 (which smooths rain over 25 km cells and "
                           "runs lower on heavy days; the two disagree on single days by a factor of two or more, so read them as a range); ERA5 lowest daily minimum temperature and highest hourly 10 m wind, averaged "
                           "over the three cells over Gaza (partly sea, so milder and less gusty than an exposed tent site). Figures in <em>italics</em> "
                           "are from Gaza authorities (Ministry of Health, Civil Defence, Government Media Office) as relayed by the UN; the rest are UN "
                           "agency or cluster figures.")


A = Area()
OUT = edd.OUT_DIR / A.slug
CACHE = Path(A.cache)


def set_area(spec: dict) -> None:
    """Point the module at the page being built (its `[area]` table over Gaza's defaults)."""
    global A, OUT, CACHE
    kw = dict(spec.get("area", {}))
    for k in ("era5_cells",):
        if k in kw:
            kw[k] = {kk: tuple(v) for kk, v in kw[k].items()}
    for k in ("gpcc_cells",):
        if k in kw:
            kw[k] = [tuple(v) for v in kw[k]]
    for k in ("imerg_box", "imerg_split", "seas5_box"):
        if k in kw and kw[k] is not None:
            kw[k] = tuple(kw[k]) if kw[k] else None
    A = Area(**({"slug": spec["slug"]} | kw))
    OUT = edd.OUT_DIR / A.slug
    CACHE = Path(A.cache)
    SRC_LABEL.update({"GPCC": A.gpcc_label, "ERA5": A.era5_label, "IMERG": A.imerg_label, "gauge": A.gauge_label})

WET = [10, 11, 12, 1, 2, 3, 4]                # the rainy season, Oct–Apr
TRIS = {"OND": [10, 11, 12], "NDJ": [11, 12, 1], "DJF": [12, 1, 2], "JFM": [1, 2, 3]}
SPLIT = 1979                                  # the stationarity break (see the running correlation)
ENSO_THRESH = edd.ENSO_THRESH
C_EN, C_NEU, C_LN = edd.C_ELNINO, edd.C_NEUTRAL, edd.C_LANINA
C_TEXT, C_MUTED = edd.C_TEXT, edd.C_MUTED
SRC_COL = {"GPCC": "#1F5F96", "ERA5": "#269777", "IMERG": "#8a4f7d", "gauge": "#C0782F"}  # validated (dataviz)
WET_SEQ = mcolors.LinearSegmentedColormap.from_list("wet", ["#F4F7FA", "#BFD9EE", "#5E9FD2", "#1F5F96"])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def area_polygons() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """COD-AB (FieldMaps via ocha-stratus): the area (admin 1, e.g. PS02 Gaza, PS01 West Bank) and its governorates."""
    p1, p2 = COD_CACHE / "pse_adm1.parquet", COD_CACHE / "pse_adm2.parquet"
    if not (p1.exists() and p2.exists()):
        from ocha_stratus import codab
        COD_CACHE.mkdir(parents=True, exist_ok=True)
        codab.load_codab_from_blob("pse", admin_level=1).to_crs("EPSG:4326").to_parquet(p1)
        codab.load_codab_from_blob("pse", admin_level=2).to_crs("EPSG:4326").to_parquet(p2)
    a1, a2 = gpd.read_parquet(p1), gpd.read_parquet(p2)
    return a1[a1.ADM1_PCODE == A.pcode], a2[a2.ADM1_PCODE == A.pcode]


def overlap_weights(poly, centres: dict[str, tuple[float, float]], res: float) -> dict[str, float]:
    g = gpd.GeoSeries([poly], crs=4326).to_crs(UTM).iloc[0]
    w = {}
    for k, (la, lo) in centres.items():
        c = gpd.GeoSeries([box(lo - res / 2, la - res / 2, lo + res / 2, la + res / 2)], crs=4326).to_crs(UTM).iloc[0]
        w[k] = c.intersection(g).area / g.area
    return w


def _cds_client():
    import cdsapi
    import yaml
    key = yaml.safe_load((Path.home() / ".cdsapirc").read_text())["key"]
    return cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key, quiet=True)


def era5_cell(key: str, end: str | None = None, cells: dict | None = None, cache: Path | None = None) -> pd.DataFrame:
    """Daily series for one ERA5 cell from the CDS ERA5 hourly time-series dataset (1950–).

    pr: mm, 00–24 UTC (ERA5's hourly tp is the accumulation over the hour ending at the stamp,
    so it is shifted back one hour before summing); tmin/tmax: °C; wmax: highest hourly 10 m
    wind, m/s. Re-fetched when the cache is more than 20 days behind today.
    """
    cells, cache = cells or A.era5_cells, cache or CACHE
    path = cache / f"era5_{key}.parquet"
    if path.exists():
        d = pd.read_parquet(path)
        if d.index[-1] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=20):
            return d
    la, lo = cells[key]
    end = end or (pd.Timestamp.now().normalize() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    print(f"  fetching ERA5 hourly time series for cell {key} ({la}, {lo}) 1950–{end} from CDS…", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"era5_{key}.zip"
    _cds_client().retrieve("reanalysis-era5-single-levels-timeseries", {
        "variable": ["total_precipitation", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
        "location": {"longitude": lo, "latitude": la}, "date": [f"1950-01-01/{end}"], "data_format": "csv"}, str(tmp))
    with zipfile.ZipFile(tmp) as z:
        h = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), parse_dates=["valid_time"]).set_index("valid_time").sort_index()
    tmp.unlink()
    assert abs(h.latitude.iloc[0] - la) < 1e-6 and abs(h.longitude.iloc[0] - lo) < 1e-6, "CDS returned a different cell"
    d = _era5_daily(h)
    d.to_parquet(path)
    return d


def _era5_daily(h: pd.DataFrame) -> pd.DataFrame:
    tp = h.tp.shift(-1) * 1000.0
    ws = np.hypot(h.u10, h.v10)
    d = pd.DataFrame({"pr": tp.resample("D").sum(min_count=24), "tmin": h.t2m.resample("D").min() - 273.15,
                      "tmax": h.t2m.resample("D").max() - 273.15, "wmax": ws.resample("D").max()})
    return d.dropna(subset=["pr"])


def era5_area(weights: dict[str, float]) -> pd.DataFrame:
    """Overlap-weighted mean of the daily series of the ERA5 cells over the area."""
    cells = {k: era5_cell(k) for k in A.era5_cells}
    idx = next(iter(cells.values())).index
    for d in cells.values():
        idx = idx.intersection(d.index)
    tot = sum(weights.values())
    return sum(cells[k].loc[idx] * (w / tot) for k, w in weights.items())


def gpcc_monthly(poly=None) -> pd.Series:
    """GPCC Full Data (v2020, to 2019) + Monitoring (after), combined 1° product via NOAA PSL NCSS: the
    area's cell, or the overlap-weighted mean of its cells."""
    path = CACHE / "gpcc_comb_1deg.nc"
    if not path.exists():
        la = [c[0] for c in A.gpcc_cells]; lo = [c[1] for c in A.gpcc_cells]
        url = ("https://psl.noaa.gov/thredds/ncss/grid/Datasets/gpcc/combined/"
               "precip.comb.v2020to2019-v2020monitorafter.total.nc?var=precip"
               f"&north={max(la) + 0.5}&south={min(la) - 0.75}&west={min(lo) - 0.75}&east={max(lo) + 0.5}&temporal=all&accept=netcdf")
        r = requests.get(url, timeout=300); r.raise_for_status()
        CACHE.mkdir(parents=True, exist_ok=True); path.write_bytes(r.content)
    import xarray as xr
    with xr.open_dataset(path) as ds:
        if len(A.gpcc_cells) == 1:
            s = ds.precip.sel(lat=A.gpcc_cells[0][0], lon=A.gpcc_cells[0][1]).to_series()
        else:
            w = overlap_weights(poly, {i: c for i, c in enumerate(A.gpcc_cells)}, 1.0)
            tw = sum(w.values())
            s = sum(ds.precip.sel(lat=c[0], lon=c[1]).to_series() * (w[i] / tw) for i, c in enumerate(A.gpcc_cells))
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s.dropna()


def gauge_daily() -> pd.DataFrame:
    """The area's rain gauge: one or more GHCN-Daily stations spliced by date ([id, from, to])."""
    parts = []
    for sid, a, b in A.gauges:
        d = ghcn_daily(sid)
        parts.append(d.loc[(a or None):(b or None)])
    return pd.concat(parts).sort_index()


def ghcn_daily(station: str) -> pd.DataFrame:
    path = CACHE / f"ghcn_{station}.csv"
    if not path.exists():
        r = requests.get(f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station}.csv", timeout=120)
        r.raise_for_status(); path.write_bytes(r.content)
    d = pd.read_csv(path, usecols=["DATE", "PRCP", "TMIN"], parse_dates=["DATE"]).set_index("DATE")
    return pd.DataFrame({"pr": d.PRCP / 10.0, "tmin": d.TMIN / 10.0})


def imerg_daily(weights_fn, box: tuple | None = None, path: Path | None = None, split_lat: tuple | None = None) -> pd.DataFrame:
    """IMERG late run v7 daily (mm/day) from the prod raster blob, windowed COG reads over `box`,
    cached as a small cube (`path`) and topped up with new days on each run. Returns the area-weighted
    daily mean and, if `split_lat` is given, the means north of its first and south of its second
    latitude (for Gaza: Gaza City and North Gaza vs Khan Younis and Rafah)."""
    import ocha_stratus as stratus
    import rasterio
    from rasterio.windows import from_bounds
    box = box or A.imerg_box; path = path or CACHE / "imerg_box.npz"
    split_lat = split_lat if split_lat is not None else A.imerg_split
    have = np.load(path) if path.exists() else None
    done = set(pd.to_datetime(have["dates"]).strftime("%Y-%m-%d")) if have is not None else set()
    try:
        cc = stratus.get_container_client("raster", stage="prod")
        names = [b.name for b in cc.list_blobs(name_starts_with="imerg/daily/late/v7/processed/") if b.name.endswith(".tif")]
        todo = [n for n in names if re.search(r"(\d{4}-\d{2}-\d{2})", n).group(1) not in done]
    except Exception as e:  # noqa: BLE001
        print(f"  (IMERG blob not reachable, using the cache: {e})"); todo = []
    if todo:
        print(f"  reading {len(todo)} IMERG day(s) from blob…", flush=True)
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")

        def one(n):
            with rasterio.Env(**env), rasterio.open(cc.get_blob_client(n).url) as src:
                w = from_bounds(*box, src.transform).round_offsets().round_lengths()
                return re.search(r"(\d{4}-\d{2}-\d{2})", n).group(1), src.read(1, window=w), src.window_transform(w)
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(one, sorted(todo)))
        tr = res[0][2]; ny, nx = res[0][1].shape
        x = tr.c + tr.a * (np.arange(nx) + 0.5); y = tr.f + tr.e * (np.arange(ny) + 0.5)
        pr = np.stack([r[1] for r in res]).astype("float32"); dates = pd.to_datetime([r[0] for r in res])
        if have is not None:
            assert np.allclose(have["x"], x) and np.allclose(have["y"], y), "IMERG window moved"
            pr = np.concatenate([have["pr"], pr]); dates = pd.to_datetime(have["dates"]).append(dates)
        o = np.argsort(dates.values)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pr=pr[o], dates=dates.values[o].astype("datetime64[D]"), x=x, y=y)
        have = np.load(path)
    pr = np.where(have["pr"] < 0, np.nan, have["pr"]); x, y = have["x"], have["y"]
    W = weights_fn(x, y)
    f = lambda M: np.nansum(pr * M, axis=(1, 2)) / M.sum()
    out = pd.DataFrame({"pr": f(W)}, index=pd.to_datetime(have["dates"]))
    if split_lat:
        out["north"] = f(W * (y[:, None] >= split_lat[0])); out["south"] = f(W * (y[:, None] <= split_lat[1]))
    return out


def nino_long() -> pd.Series:
    """HadISST1 Niño3.4 anomaly, 1870– (NOAA PSL). Used only where the record starts before 1950."""
    path = CACHE / "nino34_long_hadisst.data"
    if not path.exists():
        r = requests.get("https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data", timeout=60)
        r.raise_for_status(); path.write_text(r.text)
    return _month_start(ts._parse_psl(path.read_text()))


def nino_pinned() -> pd.Series:
    """The series' pinned Niño3.4 (NOAA PSL, ERSST v5 basis, 1950–): same as every other deep dive."""
    return _month_start(ts._parse_psl(Path("cache/nino34.data").read_text()))


def _month_start(s: pd.Series) -> pd.Series:
    s = s[s > -90].copy()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s


# --------------------------------------------------------------------------- #
# Seasonal aggregation
# --------------------------------------------------------------------------- #
def season_year(idx: pd.DatetimeIndex) -> np.ndarray:
    """Rainy seasons are labelled by the year they start (Oct 2025 – Apr 2026 = 2025)."""
    return np.where(idx.month >= 8, idx.year, idx.year - 1)


def _cal_year(y: int, m: int) -> int:
    """Calendar year of month m in the season year y (August–July, labelled by the August year),
    so Jan–Mar 1980 belongs to 1979 like Oct–Dec 1979 does."""
    return y if m >= 8 else y + 1


def seasonal(monthly: pd.Series, months: list[int]) -> pd.Series:
    """Season totals from a monthly series, labelled by season year (see _cal_year), only for
    seasons with every month present."""
    out = {}
    for y in range(monthly.index.year.min() - 1, monthly.index.year.max() + 1):
        ts_ = [pd.Timestamp(_cal_year(y, m), m, 1) for m in months]
        if all(t in monthly.index and np.isfinite(monthly[t]) for t in ts_):
            out[y] = float(monthly[ts_].sum())
    return pd.Series(out, dtype=float)


def nino_season(nino: pd.Series, years, months: list[int]) -> pd.Series:
    out = {}
    for y in years:
        ts_ = [pd.Timestamp(_cal_year(y, m), m, 1) for m in months]
        if all(t in nino.index for t in ts_):
            out[y] = float(nino[ts_].mean())
    return pd.Series(out, dtype=float)


def phase_of(v) -> np.ndarray:
    v = np.asarray(v)
    return np.where(v >= ENSO_THRESH, "El Niño", np.where(v <= -ENSO_THRESH, "La Niña", "Neutral"))


def winter_daily_metrics(d: pd.DataFrame, wind_thresh: float | None = None, min_days: int = 200) -> pd.DataFrame:
    """Per rainy season (Oct–Apr) from a daily frame with pr (and optionally tmin, wmax): total, rain
    days (≥ 1 mm), days ≥ 10 and ≥ 20 mm, wettest day, nights below 5 °C, days with the hourly
    10 m wind at or above `wind_thresh`."""
    w = d[d.index.month.isin(WET)].copy()
    w["sy"] = season_year(w.index)
    g = w.groupby("sy")
    m = pd.DataFrame({"n": g.pr.count(), "total": g.pr.sum(), "d1": g.pr.apply(lambda s: int((s >= 1).sum())),
                      "d10": g.pr.apply(lambda s: int((s >= 10).sum())), "d20": g.pr.apply(lambda s: int((s >= 20).sum())),
                      "rx1": g.pr.max()})
    if "tmin" in w:
        m["cold"] = g.tmin.apply(lambda s: int((s < A.cold_thresh).sum()) if s.notna().sum() >= min_days else np.nan)
    if "wmax" in w and wind_thresh is not None:
        m["windy"] = g.wmax.apply(lambda s: int((s >= wind_thresh).sum()))
    return m[m.n >= min_days]


# --------------------------------------------------------------------------- #
# Grid (ERA5 monthly pixel stack shared with the survey)
# --------------------------------------------------------------------------- #
def region_country(grid: edd.Grid, gaza: gpd.GeoDataFrame, ne: gpd.GeoDataFrame) -> edd.Country:
    """An edd.Country over REGION: mask = land cells (Natural Earth + Gaza), geom = Gaza,
    so the survey's map helpers draw the southern Levant with Gaza outlined."""
    res = float(abs(grid.x[1] - grid.x[0]))
    w, s, e, n = REGION
    ix = np.where((grid.x >= w) & (grid.x <= e))[0]; iy = np.where((grid.y >= s) & (grid.y <= n))[0]
    c0, c1, r0, r1 = ix[0], ix[-1] + 1, iy[0], iy[-1] + 1
    lon, lat = grid.x[c0:c1], grid.y[r0:r1]
    tr = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    land_polys = list(ne[ne.intersects(box(w - 1, s - 1, e + 1, n + 1))].geometry) + list(gaza.geometry)
    land = rasterize(((g, 1) for g in land_polys), out_shape=(len(lat), len(lon)), transform=tr, fill=0,
                     all_touched=False, dtype="uint8").astype(bool)
    land |= rasterize(((g, 1) for g in gaza.geometry), out_shape=(len(lat), len(lon)), transform=tr, fill=0,
                      all_touched=True, dtype="uint8").astype(bool)
    sub = grid.stack[:, r0:r1, c0:c1]
    if grid.ext is not None:
        sub = np.concatenate([np.asarray(sub), np.asarray(grid.ext[:, r0:r1, c0:c1])], axis=0)
    nb = ne[ne.intersects(box(w - 3, s - 3, e + 3, n + 3))]
    return edd.Country(iso3="GAZ", geom=gaza.geometry, lat=lat, lon=lon, mask=land, sub=np.asarray(sub), neighbours=nb)


def wet_stack(c: edd.Country, grid: edd.Grid) -> tuple[np.ndarray, np.ndarray]:
    """(n_seasons, ny, nx) Oct–Apr mean rainfall (mm/day), labelled by the October year."""
    out, yrs = [], []
    for y in grid.years:
        keys = [(y if m >= 10 else y + 1, m) for m in WET]
        if all(k in grid.mpos for k in keys):
            out.append(c.sub[[grid.mpos[k] for k in keys]].mean(0)); yrs.append(y)
    return np.array(out, dtype="float64"), np.array(yrs)


def gaza_cells(c: edd.Country, gaza: gpd.GeoDataFrame) -> np.ndarray:
    res = float(abs(c.lon[1] - c.lon[0]))
    tr = from_origin(c.lon[0] - res / 2, c.lat[0] + res / 2, res, res)
    return rasterize(((g, 1) for g in gaza.geometry), out_shape=c.mask.shape, transform=tr, fill=0,
                     all_touched=True, dtype="uint8").astype(bool)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def corr(x, y) -> tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def running_corr(x: pd.Series, y: pd.Series, half: int = 15, min_n: int = 25) -> pd.Series:
    """Centred (2·half + 1)-year running Pearson r, labelled by the window's centre year."""
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    out = {}
    for c in range(j.index.min() + half, j.index.max() - half + 1):
        w = j.loc[c - half:c + half]
        if len(w) >= min_n:
            out[c] = stats.pearsonr(w.x, w.y)[0]
    return pd.Series(out, dtype=float)


def thirds(v: pd.Series) -> pd.Series:
    """Tercile of each value within its own record: 'wettest', 'middle', 'driest'."""
    p = v.rank(pct=True)
    return pd.Series(np.where(p > 2 / 3, "wettest", np.where(p <= 1 / 3, "driest", "middle")), index=v.index)


def phase_table(tot: pd.Series, nino: pd.Series, lo: int, hi: int) -> list[dict]:
    """Counts of seasons per ENSO phase in the wettest/middle/driest third, terciles computed
    within the period itself (so each period is judged against its own climate)."""
    j = pd.concat([tot.rename("tot"), nino.rename("nino")], axis=1).dropna().loc[lo:hi]
    j["third"] = thirds(j.tot); j["phase"] = phase_of(j.nino)
    rows = []
    for ph in ("El Niño", "Neutral", "La Niña"):
        g = j[j.phase == ph]
        rows.append(dict(phase=ph, n=len(g), wet=int((g.third == "wettest").sum()), mid=int((g.third == "middle").sum()),
                         dry=int((g.third == "driest").sum()), years=list(g.index)))
    return rows


def analyse(spec: dict, grid: edd.Grid, ne: gpd.GeoDataFrame, indices: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    gaza, govs = area_polygons()
    poly = gaza.geometry.iloc[0]
    wts = overlap_weights(poly, A.era5_cells, 0.25)

    def imerg_weights(x, y):
        return np.array([[overlap_weights(poly, {"c": (la, lo)}, 0.1)["c"] for lo in x] for la in y])

    era5 = era5_area(wts)
    imerg = imerg_daily(imerg_weights)
    gpcc = gpcc_monthly(poly)
    bs = gauge_daily()
    n_long, n_pin = nino_long(), nino_pinned()
    end_era5 = era5.index[-1]

    # Monthly totals per source (only complete months)
    def monthly(d: pd.Series, min_frac: float = 0.9) -> pd.Series:
        m = d.resample("MS").agg(["sum", "count"])
        full = m["count"] >= min_frac * m.index.days_in_month
        return m["sum"].where(full).dropna()
    mon = {"ERA5": monthly(era5.pr), "IMERG": monthly(imerg.pr), "GPCC": gpcc, "gauge": monthly(bs.pr.dropna(), 0.85)}

    # Rainy-season totals and ENSO (DJF Niño3.4). Pinned ERSST v5 from 1950, HadISST before.
    tot = {k: seasonal(v, WET) for k, v in mon.items()}
    years_all = range(1880, 2027)
    djf_pin = nino_season(n_pin, years_all, [12, 1, 2])
    djf_long = nino_season(n_long, years_all, [12, 1, 2])

    # 1. Stationarity: 31-year running r per source (HadISST so the long records share one index).
    # Periods before 1979 and the full record use HadISST throughout; 1979– uses the pinned series.
    run = {k: running_corr(djf_long, v) for k, v in tot.items() if k != "IMERG"}
    per, diff = {}, {}
    for k, v in tot.items():
        for tag, lo, hi, nin in [("pre", 1891, SPLIT - 1, djf_long), ("post", SPLIT, 2025, djf_pin),
                                 ("recent", 1998, 2025, djf_pin), ("full", 1891, 2025, djf_long)]:
            j = pd.concat([v.rename("v"), nin.rename("n")], axis=1).dropna().loc[lo:hi]
            if len(j) >= 20:
                r, p, n = corr(j.n, j.v)
                rd = stats.pearsonr(signal.detrend(j.n), signal.detrend(j.v))[0]
                per[(k, tag)] = dict(r=r, p=p, n=n, r_detr=float(rd), first=int(j.index.min()), last=int(j.index.max()),
                                     index="HadISST" if nin is djf_long else "ERSST v5 (pinned)")
        if (k, "pre") in per and (k, "post") in per:
            a_, b_ = per[(k, "pre")], per[(k, "post")]
            z = (np.arctanh(b_["r"]) - np.arctanh(a_["r"])) / np.sqrt(1 / (a_["n"] - 3) + 1 / (b_["n"] - 3))
            diff[k] = dict(z=float(z), p=float(2 * stats.norm.sf(abs(z))))

    # 2. By trimester, concurrent and with the Aug–Oct Niño3.4 you know before the season starts
    tri_rows = []
    aso = nino_season(n_pin, years_all, [8, 9, 10])
    for k in ("GPCC", "ERA5", "IMERG"):
        for code, mm in list(TRIS.items()) + [("Oct–Apr", WET)]:
            s = seasonal(mon[k], mm).loc[SPLIT:2025]
            conc = nino_season(n_pin, s.index, mm if len(mm) == 3 else [12, 1, 2])
            r_c, p_c, n_c = corr(conc.reindex(s.index), s)
            r_l, p_l, _ = corr(aso.reindex(s.index), s)
            tri_rows.append(dict(src=k, season=code, r=r_c, p=p_c, n=n_c, r_aso=r_l, p_aso=p_l,
                                 first=int(s.index.min()), last=int(s.index.max())))

    # 3. Phase tables (terciles within each period), GPCC for the long view, ERA5 alongside
    ptab = {("GPCC", 1891, SPLIT - 1): phase_table(tot["GPCC"], djf_long, 1891, SPLIT - 1),
            ("GPCC", SPLIT, 2025): phase_table(tot["GPCC"], djf_pin, SPLIT, 2025),
            ("ERA5", SPLIT, 2025): phase_table(tot["ERA5"], djf_pin, SPLIT, 2025)}

    j = pd.concat([tot["GPCC"].rename("t"), djf_long.rename("n")], axis=1).dropna().loc[SPLIT:2025]
    j["third"] = thirds(j.t)
    en_had = dict(n=int((j.n >= ENSO_THRESH).sum()), wet=int(((j.n >= ENSO_THRESH) & (j.third == "wettest")).sum()))

    # Strong El Niño winters (DJF ≥ +1.5 on the pinned series) with each record's rank
    strong = []
    for y in djf_pin[djf_pin >= 1.5].index:
        row = dict(year=int(y), nino=float(djf_pin[y]))
        for k in ("GPCC", "ERA5", "IMERG", "gauge"):
            v = tot[k]
            ref = v.loc[SPLIT:2025] if y >= SPLIT else v.loc[1950:SPLIT - 1]
            if y in v.index and y in ref.index:
                row[k] = dict(mm=float(v[y]), pct=float(ref.rank(pct=True)[y]), third=thirds(ref)[y])
        strong.append(row)

    # 4. Daily metrics by phase (since SPLIT): ERA5 cells, IMERG and the gauge over the area
    w_era5 = era5[era5.index.month.isin(WET)]
    wind95 = float(w_era5.loc[f"{SPLIT}":].wmax.quantile(0.95))
    dm = {"ERA5": winter_daily_metrics(era5, wind95), "IMERG": winter_daily_metrics(imerg[["pr"]]),
          "gauge": winter_daily_metrics(bs, min_days=190)}
    daily_rows = []
    for k, m in dm.items():
        m = m.loc[SPLIT:2025].copy()
        m["nino"] = djf_pin.reindex(m.index); m = m.dropna(subset=["nino"]); m["phase"] = phase_of(m.nino)
        dm[k] = m
        for col in ["total", "d1", "d10", "d20", "rx1", "cold", "windy"]:
            if col not in m or m[col].isna().all() or (col == "cold" and k == "ERA5" and not A.era5_cold):   # Gaza's ERA5 cells are part sea
                continue
            mm_ = m.dropna(subset=[col])
            r, p, n = corr(mm_.nino, mm_[col])
            by = mm_.groupby("phase")[col].mean()
            daily_rows.append(dict(src=k, metric=col, r=r, p=p, n=n, first=int(mm_.index.min()), last=int(mm_.index.max()),
                                   en=float(by.get("El Niño", np.nan)), neu=float(by.get("Neutral", np.nan)),
                                   ln=float(by.get("La Niña", np.nan))))

    # 5. Grid: pixel correlation maps + El Niño wet composite over the southern Levant
    c = region_country(grid, gaza, ne)
    gz_cells = gaza_cells(c, gaza)
    annual = sum(edd.season_stack(c, grid, edd.season_months(t))[0].mean(0) for t in ["DJF", "MAM", "JAS", "OND"])

    def analysable(months):
        s = edd.season_stack(c, grid, months)[0].mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return c.mask & (annual > 0) & (s / annual >= 0.25) & (s >= ts.PIXEL_MIN_TRI_MM_DAY)
    panels, map_rows = [], []
    for code, mm in TRIS.items():
        R, yrs = edd.season_stack(c, grid, mm)
        r, k = edd.best_lag_corr(R, yrs, indices, mm, 3)
        ok = analysable(mm)
        v = r[ok & gz_cells]
        panels.append(dict(r=r, analysable=ok, title=f"{code} vs Niño3.4\n(best lag 0–3 mo)"))
        vv = r[ok]
        map_rows.append(dict(season=code, gaza=float(np.nanmean(v)) if v.size else np.nan, n=int(ok.sum()),
                             pos=float((vv >= 0.3).mean()) if vv.size else np.nan, sig=float((ts._pearson_p(vv, len(yrs)) < 0.05).mean()) if vv.size else np.nan))
    edd.fig_corr_maps(c, panels, OUT / "corr_maps.png", "Southern Levant")
    # Oct–Apr composite and wettest-third hit rate per cell, 1981–
    S, sy = wet_stack(c, grid)
    nin = djf_pin.reindex(sy).values
    en = nin >= ENSO_THRESH
    Z = (S - S.mean(0)) / S.std(0)
    pct_ = (S.argsort(0).argsort(0) + 1) / S.shape[0]
    comp = Z[en].mean(0); hit = (pct_[en] > 2 / 3).mean(0)
    ok_wet = c.mask & (S.mean(0) >= ts.PIXEL_MIN_TRI_MM_DAY)
    fig_wet_composite(c, comp, hit, ok_wet, int(en.sum()), int(sy.min()), int(sy.max()), OUT / "composite_maps.png")

    # 6. SEAS5: skill + current forecast for the cells over Gaza
    gc = edd.Country(iso3="GAZ", geom=gaza.geometry, lat=c.lat, lon=c.lon, mask=gz_cells, sub=c.sub, neighbours=c.neighbours)
    skill = edd.seas5_skill_issued(gc, {A.name: gz_cells}, int(spec.get("skill_issued_month", 9)))
    if skill:
        edd.fig_skill_issued(gc, grid, {}, skill, OUT / "skill_issued.png", A.name, area_label=A.name)

    # SEAS5 raw ensemble mean: rank cross-check by window and the forecast by month
    seas5_raw = seas5_raw_ranks(int(spec.get("skill_issued_month", 9)), box=A.seas5_box)
    seas5_mon = seas5_monthly(seas5_raw, mon["ERA5"]) if seas5_raw else None
    if seas5_mon:
        fig_seas5_monthly(seas5_mon, seas5_raw["year"], seas5_raw["month"], OUT / "seas5_monthly.png")

    # 7. Figures on the Gaza series
    fig_cycle(mon, OUT / "seasonal_cycle.png")
    fig_stationarity(run, per, OUT / "stationarity.png")
    fig_history(tot, djf_pin, OUT / "phase_history.png")
    fig_daily(dm, OUT / "daily_by_phase.png")
    events = spec.get("impacts", [])
    fig_winters(imerg, events, spec.get("timeline_winters", [2023, 2024, 2025]), OUT / "winters.png")

    # Storm statistics, the event catalogue and last winter as reference points
    st = storm_stats(imerg.pr, era5.pr, djf_pin)
    cold = cold_stats(era5.tmin, djf_pin)
    data_dir = edd.DEEP_DIR / "data"
    ev = load_events(data_dir / f"{A.slug}_events.csv", imerg, era5) if (data_dir / f"{A.slug}_events.csv").exists() else None
    totals = pd.read_csv(data_dir / f"{A.slug}_season_totals.csv") if (data_dir / f"{A.slug}_season_totals.csv").exists() else None
    ref = reference_winters(ev, totals, st["ei"]) if (ev is not None and totals is not None) else None
    fig_storm_tiers(st, djf_pin, OUT / "storm_tiers.png")
    fig_first_storm(st, djf_pin, OUT / "first_storm.png")
    if ev is not None:
        fig_event_impacts(ev, OUT / "event_impacts.png", st["ei"])

    # Rainfall on the dates of reported impacts, and how often such days occur
    ev_rows = impact_rain(events, imerg, era5)
    pre_rows = impact_rain(spec.get("prewar", []), imerg, era5)
    wet_days = imerg[imerg.index.month.isin(WET)].pr
    freq = {t: float((wet_days >= t).groupby(season_year(wet_days.index)).sum().loc[1998:2025].mean()) for t in (10, 20, 30, 50)}

    nn = edd.NINO_LATEST.dropna() if edd.NINO_LATEST is not None else n_pin
    nn = nn[nn > -90]
    return dict(wts=wts, per=per, run=run, tri_rows=tri_rows, ptab=ptab, strong=strong, daily_rows=daily_rows,
                wind95=wind95, map_rows=map_rows, diff=diff, en_had=en_had, st=st, cold=cold, ev=ev, ref=ref, gauge_last=int(dm["gauge"].index.max()), seas5_raw=seas5_raw, seas5_mon=seas5_mon, skill=skill, tot=tot, djf=djf_pin, ev_rows=ev_rows, pre_rows=pre_rows, freq=freq,
                end_era5=end_era5, end_imerg=imerg.index[-1], n_grid=(int(sy.min()), int(sy.max())), n_en_grid=int(en.sum()),
                comp_gaza=float(np.nanmean(comp[gz_cells])), hit_gaza=float(np.nanmean(hit[gz_cells])),
                hit_region=float(np.nanmedian(hit[ok_wet])), comp_region_pos=float((comp[ok_wet] > 0).mean()),
                nino_now=dict(value=float(nn.iloc[-1]), date=nn.index[-1], mean3=float(nn.iloc[-3:].mean())),
                imerg_last=imerg, era5=era5)


def seas5_raw_ranks(issued_month: int = 9, box=(34.0, 31.0, 34.8, 31.8)) -> dict | None:
    """Cross-check of the skill cube: the raw SEAS5 ensemble-mean precipitation (prod raster blob,
    precip_em_i<YYYY>-<MM>-01_lt<k>.tif) averaged over a box over Gaza, for every issuance of
    `issued_month` since 1981. For each three-month window: where the latest issuance ranks among
    all of them (1 = wettest) and the three wettest hindcast years. Cached; refreshed when a new
    year's issuance appears."""
    import ocha_stratus as stratus
    import rasterio
    from rasterio.windows import from_bounds
    path = CACHE / f"seas5_em_m{issued_month:02d}.parquet"
    df = pd.read_parquet(path) if path.exists() else None
    try:
        cc = stratus.get_container_client("raster", stage="prod")
        pre = "seas5/monthly/processed/precip_em_i"
        years = sorted({int(b.name[len(pre):len(pre) + 4]) for b in cc.list_blobs(name_starts_with=pre)
                        if b.name[len(pre) + 5:len(pre) + 7] == f"{issued_month:02d}"})
        todo = [y for y in years if df is None or y not in df.index]
    except Exception as e:  # noqa: BLE001
        print(f"  (SEAS5 blob not reachable, using the cache: {e})"); todo = []
    if todo:
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

        def one(a):
            y, lt = a
            with rasterio.Env(**env), rasterio.open(cc.get_blob_client(f"{pre}{y}-{issued_month:02d}-01_lt{lt}.tif").url) as src:
                w = from_bounds(*box, src.transform).round_offsets().round_lengths()
                return y, lt, float(np.nanmean(src.read(1, window=w)))
        with ThreadPoolExecutor(24) as ex:
            res = list(ex.map(one, [(y, lt) for y in todo for lt in range(7)]))
        new = pd.DataFrame(res, columns=["y", "lt", "v"]).pivot(index="y", columns="lt", values="v")
        new.columns = [str(c) for c in new.columns]
        df = new if df is None else pd.concat([df, new]).sort_index()
        CACHE.mkdir(parents=True, exist_ok=True); df.to_parquet(path)
    if df is None or df.empty:
        return None
    by_start = {v[0]: k for k, v in ts._TRIMESTER_MONTHS.items()}
    latest = int(df.index.max()); out = []
    for lt0 in range(5):
        code = by_start[((issued_month - 1 + lt0) % 12) + 1]
        v = df[[str(lt0), str(lt0 + 1), str(lt0 + 2)]].mean(axis=1)
        rank = int((v > v[latest]).sum()) + 1
        top = [int(y) for y in v.drop(latest).sort_values(ascending=False).index[:3]]
        out.append(dict(code=code, rank=rank, n=len(v), ratio=float(v[latest] / v.drop(latest).mean()), top=top))
    return dict(year=latest, month=issued_month, rows=out, em=df)


def seas5_monthly(sr: dict, era5_monthly: pd.Series) -> list[dict]:
    """The same raw ensemble-mean files, month by month: each lead month's hindcast mean and latest
    forecast (mm/month), the forecast's rank among all issuances (1 = wettest), and the detrended
    correlation of the hindcast ensemble mean with ERA5 rainfall over Gaza (1981 to the last
    complete year) as the month's skill."""
    em, latest, im = sr["em"], sr["year"], sr["month"]
    rows = []
    for lt in range(7):
        m = (im - 1 + lt) % 12 + 1
        yoff = 1 if im + lt > 12 else 0
        days = np.array([pd.Timestamp(y + yoff, m, 1).days_in_month for y in em.index])
        v = em[str(lt)] * days
        hind = v.drop(latest)
        obs = pd.Series({y: era5_monthly.get(pd.Timestamp(y + yoff, m, 1), np.nan) for y in hind.index}).dropna()
        hh = hind.loc[obs.index]
        r = float(stats.pearsonr(signal.detrend(hh.values), signal.detrend(obs.values))[0])
        rows.append(dict(month=m, lead=lt, hind=float(hind.mean()), fc=float(v[latest]), pct=float(v[latest] / hind.mean() - 1),
                         rank=int((v > v[latest]).sum()) + 1, n=len(v), r=r, obs_clim=float(obs.mean()), n_skill=len(obs)))
    return rows


def reference_winters(ev: pd.DataFrame, totals: pd.DataFrame, storms_im: pd.DataFrame) -> dict:
    """Last winter (2025/26) as two reference points: as it happened, and without its one storm of 50 mm
    or more (Byron). Household-impacts are given two ways because the sources disagree: the sum of the
    per-event rows, and the Shelter Cluster's monthly snapshots (first storm + December + January) plus
    the February–March event rows. A household is counted once per storm that affected it."""
    e = ev[ev.winter == "2025/26"]
    byron_ids = []
    for _, sr in storms_im[(storms_im.sy == 2025) & (storms_im["max"] >= 50)].iterrows():
        m = (e.start - pd.Timedelta(days=1) <= sr.end) & (e.end >= sr.start) & e.hh_affected_un.notna()
        byron_ids += list(e[m].sort_values("hh_affected_un").tail(1).event_id)
    b = e[e.event_id.isin(byron_ids)]
    tv = lambda pat: totals[(totals.winter == "2025/26") & totals.metric.str.contains(pat, case=False, regex=True)]
    dec = float(tv(r"^Families affected, December").value.iloc[0]); jan = float(tv(r"^Families affected, January").value.iloc[0])
    first = float(e[e.start == e.start.min()].hh_affected_un.iloc[0])
    late = float(e[e.start >= "2026-02-01"].hh_affected_un.sum())
    rows_all = float(e.hh_affected_un.sum()); snaps_all = first + dec + jan + late
    hh_b = float(b.hh_affected_un.sum()); tents_all = float(e.tents_damaged_un.sum()); tents_b = float(b.tents_damaged_un.sum())
    storm_deaths = tv(r"^Storm-related deaths").sort_values("as_of").iloc[-1]
    cold = tv(r"^Child hypothermia deaths").sort_values(["as_of", "value"])
    cold_at = cold[cold.as_of <= storm_deaths.as_of].value.max(); cold_final = cold.value.max()
    collapse_max = float(tv(r"^Building-collapse deaths").value.max())
    return dict(rows_all=rows_all, snaps_all=snaps_all, rows_wo=rows_all - hh_b, snaps_wo=snaps_all - hh_b,
                tents_all=tents_all, tents_wo=tents_all - tents_b, byron_hh=hh_b, byron_tents=tents_b,
                deaths=float(storm_deaths.value + (cold_final - cold_at)), deaths_hi=collapse_max + float(cold_final),
                deaths_asof=storm_deaths.as_of, cold_final=float(cold_final),
                dec=dec, jan=jan, first=first, late=late)


def impact_rain(events: list[dict], imerg: pd.DataFrame, era5: pd.DataFrame) -> list[dict]:
    """For each dated impact, over the window [start − 1 day, end]: the largest IMERG daily total over
    Gaza (and its northern and southern halves), the IMERG window total, the lowest ERA5 daily
    minimum temperature and the highest hourly ERA5 10 m wind."""
    rows = []
    for e in events:
        a = pd.Timestamp(e["start"]) - pd.Timedelta(days=1); b = pd.Timestamp(e.get("end", e["start"]))
        w, x = imerg.loc[a:b], era5.loc[a:b]
        rows.append(dict(e, im_max=float(w.pr.max()) if len(w) else np.nan, im_sum=float(w.pr.sum()) if len(w) else np.nan,
                         im_max_n=float(w.north.max()) if len(w) else np.nan, im_max_s=float(w.south.max()) if len(w) else np.nan,
                         era5_sum=float(x.pr.sum()) if len(x) else np.nan, era5_max=float(x.pr.max()) if len(x) else np.nan,
                         tmin=float(x.tmin.min()) if len(x) else np.nan, wmax=float(x.wmax.max()) if len(x) else np.nan))
    return rows


# --------------------------------------------------------------------------- #
# Storm events: counts per winter by tier, first damaging storm, cold nights
# --------------------------------------------------------------------------- #
TIERS = [10, 20, 50]                          # IMERG wettest day of a storm over Gaza (mm)
TIER_LABEL = {10: "10–20 mm", 20: "20–50 mm", 50: "≥ 50 mm (Byron-class)"}


def storms(d: pd.Series, wet: float = 1.0, gap: int = 2) -> pd.DataFrame:
    """Rain spells in the rainy season: runs of days with ≥ `wet` mm, merged across lulls of up to
    `gap` dry days (so Byron, 8–17 December 2025, is one storm). One row per storm: start, end,
    wettest day, total, season year."""
    d = d[d.index.month.isin(WET)].dropna()
    out, cur, last = [], None, None
    for t, v in d.items():
        if v < wet:
            continue
        if cur is not None and (t - last).days <= gap + 1:
            cur[1] = t; cur[2] = max(cur[2], v); cur[3] += v
        else:
            if cur is not None:
                out.append(cur)
            cur = [t, t, v, v]
        last = t
    if cur is not None:
        out.append(cur)
    e = pd.DataFrame(out, columns=["start", "end", "max", "total"])
    e["sy"] = season_year(pd.DatetimeIndex(e.start))
    return e


def storm_counts(e: pd.DataFrame, years, thresholds: dict[int, float]) -> pd.DataFrame:
    """Storms per winter at or above each tier's threshold (tier key → threshold in this record's mm)."""
    return pd.DataFrame({t: e[e["max"] >= thr].groupby("sy").size().reindex(years, fill_value=0)
                         for t, thr in thresholds.items()})


def matched_thresholds(e_ref: pd.DataFrame, e_other: pd.DataFrame, lo: int, hi: int) -> dict[int, float]:
    """Thresholds in `e_other` (ERA5) that give the same number of storms over [lo, hi] as each IMERG
    tier does in `e_ref` (frequency matching: ERA5 spreads rain over 25 km cells and runs lower)."""
    r = e_ref[(e_ref.sy >= lo) & (e_ref.sy <= hi)]; o = np.sort(e_other[(e_other.sy >= lo) & (e_other.sy <= hi)]["max"].values)[::-1]
    return {t: float(o[int((r["max"] >= t).sum()) - 1]) for t in TIERS}


def storm_stats(imerg: pd.Series, era5: pd.Series, djf: pd.Series) -> dict:
    ei, ee = storms(imerg), storms(era5)
    thr_e = matched_thresholds(ei, ee, 1998, 2025)
    ci = storm_counts(ei, range(1998, 2026), {t: float(t) for t in TIERS})
    ce = storm_counts(ee, range(1950, 2026), thr_e)
    rows = []
    for name, c, lo in [("IMERG", ci, 1998), ("ERA5", ce, SPLIT)]:
        c = c.loc[lo:2025].copy(); ph = pd.Series(phase_of(djf.reindex(c.index)), index=c.index)
        for t in TIERS:
            for label, m in [("all", ph.notna()), ("El Niño", ph == "El Niño"), ("La Niña", ph == "La Niña")]:
                v = c.loc[m, t]
                rows.append(dict(src=name, tier=t, group=label, n=int(m.sum()), mean=float(v.mean()), p10=float(v.quantile(.1)),
                                 p90=float(v.quantile(.9)), p_any=float((v >= 1).mean())))
    # first storm of 20 mm class, and storms of that class in October–November
    first = []
    for name, e, thr, lo in [("IMERG", ei, 20.0, 1998), ("ERA5", ee, thr_e[20], SPLIT)]:
        f = e[e["max"] >= thr].groupby("sy").start.min().reindex(range(lo, 2026))
        days = pd.Series({y: (d - pd.Timestamp(y, 10, 1)).days if pd.notna(d) else np.nan for y, d in f.items()})
        on = e[(e["max"] >= thr) & pd.DatetimeIndex(e.start).month.isin([10, 11])].groupby("sy").size().reindex(days.index, fill_value=0)
        ph = pd.Series(phase_of(djf.reindex(days.index)), index=days.index)
        for label, m in [("all", ph.notna()), ("El Niño", ph == "El Niño"), ("La Niña", ph == "La Niña")]:
            dd = days[m]
            first.append(dict(src=name, group=label, n=int(m.sum()), median_day=float(dd.median()),
                              p_before_dec=float((dd < 61).mean()), p_octnov=float((on[m] >= 1).mean())))
    return dict(ei=ei, ee=ee, thr_e=thr_e, ci=ci, ce=ce, rows=rows, first=first)


def cold_stats(tmin: pd.Series, djf: pd.Series, lo: int = SPLIT, hi: int = 2025) -> dict:
    t = tmin[tmin.index.month.isin(WET)].dropna(); sy = season_year(t.index)
    c8 = (t <= 8).groupby(sy).sum().loc[lo:hi]; mn = t.groupby(sy).min().loc[lo:hi]
    r, p, _ = corr(djf.reindex(c8.index), c8)
    return dict(median8=float(c8.median()), p10=float(c8.quantile(.1)), p90=float(c8.quantile(.9)), r=r, p=p,
                trend=float(np.polyfit(c8.index, c8.values, 1)[0] * 10), recent={int(y): int(c8[y]) for y in c8.index[-3:]},
                coldest_median=float(mn.median()))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
SRC_LABEL = {"GPCC": A.gpcc_label, "ERA5": A.era5_label, "IMERG": A.imerg_label, "gauge": A.gauge_label}   # reset by set_area()


def _short(k: str) -> str:
    return A.gauge_short if k == "gauge" else k


def _disp(k: str) -> str:
    return A.gauge_name if k == "gauge" else k
MON_ORDER = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]


def _save(fig, out: Path):
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_cycle(mon: dict[str, pd.Series], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=150)
    x = np.arange(12)
    ax.axvspan(1.5, 8.5, color="#eef4f9", zorder=0)
    ax.text(1.6, 0.97, "rainy season,\nOctober–April", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8.5, color=C_MUTED)
    for k in ("GPCC", "ERA5", "IMERG"):
        s = mon[k]
        s = s.loc["1991":"2020"] if k != "IMERG" else s.loc["1998-08":"2025-07"]
        clim = s.groupby(s.index.month).mean().reindex(MON_ORDER)
        yrs = "1991–2020" if k != "IMERG" else "1998–2025"
        ax.plot(x, clim.values, color=SRC_COL[k], lw=2, marker="o", ms=4.5, label=f"{SRC_LABEL[k]}, {yrs}")
    ax.set_xticks(x, [edd.MONTH_NAMES[m - 1] for m in MON_ORDER], fontsize=8.5)
    ax.set_ylabel("mm per month", fontsize=9, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax.set_title(f"{A.name}: monthly rainfall climatology", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def fig_stationarity(run: dict[str, pd.Series], per: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=150)
    crit = 0.355                                    # |r| for p < 0.05 with n = 31
    ax.axhspan(-crit, crit, color="#f1f3f3", zorder=0)
    ax.text(1921, crit + 0.02, "inside the grey band: not significant for a 31-year window (p ≥ 0.05)", fontsize=7.5, color=C_MUTED, va="bottom")
    ax.axhline(0, color="#b8bfbf", lw=0.8)
    ax.axvline(SPLIT, color=C_TEXT, lw=1, ls=(0, (4, 3)))
    ax.text(SPLIT + 0.8, -0.52, f"{SPLIT}", fontsize=8, color=C_TEXT)
    for k in ("GPCC", "gauge", "ERA5"):
        s = run.get(k)
        if s is None or s.empty:
            continue
        ax.plot(s.index, s.values, color=SRC_COL[k], lw=2, label=SRC_LABEL[k])
        ax.text(s.index[-1] + 0.6, s.values[-1], _short(k), fontsize=8, color=C_TEXT, va="center")
    ax.set_xlim(1900, 2014); ax.set_ylim(-0.6, 0.85)
    ax.set_ylabel("Pearson r, Oct–Apr rain vs DJF Niño3.4", fontsize=9, color=C_MUTED)
    ax.set_xlabel("centre year of the 31-year window", fontsize=9, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("The El Niño link is recent: 31-year running correlation in three records", fontsize=10, color=C_TEXT, loc="left")
    edd._style_ax(ax)
    _save(fig, out)


def fig_history(tot: dict[str, pd.Series], djf: pd.Series, out: Path) -> None:
    g = tot["GPCC"].loc[1950:2025]
    n = djf.reindex(g.index)
    ph = phase_of(n.values)
    base = g.loc[1991:2020].mean()
    anom = 100 * (g / base - 1)
    fig, ax = plt.subplots(figsize=(9.6, 3.8), dpi=150)
    col = pd.Series(ph).map({"El Niño": C_EN, "Neutral": C_NEU, "La Niña": C_LN}).values
    ax.bar(g.index, anom.values, color=col, width=0.8, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#8a9495", lw=0.8)
    ax.axvline(SPLIT - 0.5, color=C_TEXT, lw=1, ls=(0, (4, 3)))
    for y in g.index[(n.values >= 1.5)]:
        ax.text(y, anom[y] + (4 if anom[y] >= 0 else -4), f"{y}/{str(y + 1)[2:]}", ha="center",
                va="bottom" if anom[y] >= 0 else "top", fontsize=7, color=C_TEXT, rotation=90)
    ax.set_ylabel("% of the 1991–2020 mean", fontsize=9, color=C_MUTED)
    ax.set_title(f"{A.gpcc_title}: rainy-season totals by ENSO phase (GPCC, Oct–Apr)", fontsize=10, color=C_TEXT, loc="left")
    ax.legend(handles=[edd.Patch(color=C_EN, label=f"El Niño (DJF Niño3.4 ≥ +{ENSO_THRESH})"), edd.Patch(color=C_NEU, label="Neutral"),
                       edd.Patch(color=C_LN, label=f"La Niña (≤ −{ENSO_THRESH})")], frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_ylim(anom.min() - 15, anom.max() + 30)
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    z, se = np.arctanh(r), 1 / np.sqrt(max(n - 3, 1))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


METRIC_LABEL = {"total": "Rainy-season total", "d1": "Rain days (≥ 1 mm)", "d10": "Days ≥ 10 mm", "d20": "Days ≥ 20 mm",
                "rx1": "Wettest day", "cold": "Nights below 5 °C", "windy": "Windy days (top 5%)"}


def _metric_label(m: str) -> str:
    return f"Nights below {A.cold_thresh:g} °C" if m == "cold" else METRIC_LABEL[m]


def fig_daily(dm: dict[str, pd.DataFrame], out: Path) -> None:
    """Forest plot: correlation of each winter metric with DJF Niño3.4, per record, with 95% CI."""
    metrics = ["total", "d1", "d10", "d20", "rx1", "cold", "windy"]
    srcs = ["ERA5", "IMERG", "gauge"]
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
    ax.axvline(0, color="#8a9495", lw=0.8)
    for i, mt in enumerate(metrics):
        for j, k in enumerate(srcs):
            m = dm[k]
            if mt not in m or m[mt].isna().all() or (mt == "cold" and k == "ERA5" and not A.era5_cold):
                continue
            mm_ = m.dropna(subset=[mt])
            r, _, n = corr(mm_.nino, mm_[mt])
            if np.isnan(r):
                continue
            lo, hi = _fisher_ci(r, n)
            yv = i + (j - 1) * 0.24
            ax.plot([lo, hi], [yv, yv], color=SRC_COL[k], lw=2, solid_capstyle="round")
            ax.plot(r, yv, "o", color=SRC_COL[k], ms=6.5, mec="white", mew=1.2)
    ax.set_yticks(range(len(metrics)), [_metric_label(m) for m in metrics], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(-0.75, 0.95)
    ax.set_xlabel("Pearson r with DJF Niño3.4 (dot) and 95% interval (line)", fontsize=9, color=C_MUTED)
    spans = {k: f"{int(dm[k].index.min())}–{int(dm[k].index.max())}" for k in srcs}
    ax.legend(handles=[plt.Line2D([], [], color=SRC_COL[k], marker="o", lw=2, label=f"{SRC_LABEL[k]}, {spans[k]}") for k in srcs],
              frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    ax.set_title(f"What El Niño changes in a {A.name} winter (rainy seasons since {SPLIT})", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(axis="x", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def fig_winters(imerg: pd.DataFrame, events: list[dict], winters: list[int], out: Path) -> None:
    """Daily IMERG rainfall over the area for each winter, with the dated impacts (numbered as in the table)."""
    fig, axes = plt.subplots(len(winters), 1, figsize=(9.6, 2.35 * len(winters) + 0.4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    num = {id(e): i + 1 for i, e in enumerate(events)}
    ymax = 0
    for ax, y in zip(axes, winters):
        d = imerg.pr.loc[f"{y}-10-01":f"{y + 1}-04-30"]
        ymax = max(ymax, float(d.max()) if len(d) else 0)
    for ax, y in zip(axes, winters):
        d = imerg.pr.loc[f"{y}-10-01":f"{y + 1}-04-30"]
        ax.bar(d.index, d.values, width=0.9, color="#5E9FD2", edgecolor="none")
        ax.axhline(20, color="#8a9495", lw=0.7, ls=(0, (3, 3)))
        last, lift = None, 0
        for e in events:
            t0 = pd.Timestamp(e["start"])
            if not (pd.Timestamp(f"{y}-10-01") <= t0 <= pd.Timestamp(f"{y + 1}-04-30")):
                continue
            lift = (1 - lift) if (last is not None and (t0 - last).days < 7) else 0   # stagger neighbours
            last = t0
            ax.plot(t0, ymax * 1.08, "v", color=C_TEXT, ms=6.5, mec="white", mew=0.8, clip_on=False)
            ax.text(t0, ymax * (1.18 + 0.13 * lift), str(num[id(e)]), ha="center", va="bottom", fontsize=7.5, color=C_TEXT, clip_on=False)
        tot = float(d.sum())
        ax.text(0.005, 0.95, f"{y}/{str(y + 1)[2:]}  ·  Oct–Apr total {tot:.0f} mm", transform=ax.transAxes,
                fontsize=8.5, color=C_TEXT, va="top", fontweight="bold")
        ax.set_xlim(pd.Timestamp(f"{y}-10-01"), pd.Timestamp(f"{y + 1}-04-30"))
        ax.set_ylim(0, ymax * 1.05)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
        ax.tick_params(labelsize=8)
        edd._style_ax(ax)
    axes[len(winters) // 2].set_ylabel(f"mm per day ({A.name} mean)", fontsize=9, color=C_MUTED)
    axes[0].set_title(f"Daily rainfall over {A.name} (IMERG) and dated reports of winter-weather impacts (▼, numbered as in the table)\n"
                      "dotted line: 20 mm/day", fontsize=9.5, color=C_TEXT, loc="left", pad=22)
    fig.tight_layout()
    _save(fig, out)


def fig_seas5_monthly(rows: list[dict], year: int, im: int, out: Path) -> None:
    """Two panels on one month axis: forecast vs hindcast mean (mm/month), and the month's skill."""
    x = np.arange(len(rows)); w = 0.38
    labels = [edd.MONTH_NAMES[r["month"] - 1] for r in rows]
    fig, (ax, sx) = plt.subplots(2, 1, figsize=(9.6, 6.4), dpi=150, sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 2], hspace=0.18))
    hind = [r["hind"] for r in rows]; fc = [r["fc"] for r in rows]
    ax.bar(x - w / 2 - 0.01, hind, w, color="#C9D0D0", label="SEAS5 hindcast mean, 1981–" + str(year - 1))
    ax.bar(x + w / 2 + 0.01, fc, w, color="#1F5F96", label=f"{edd.MONTH_NAMES[im - 1]} {year} forecast")
    top = max(max(hind), max(fc))
    for i, r in enumerate(rows):
        ax.text(x[i] + w / 2 + 0.01, r["fc"] + top * 0.02, f"{_pct_signed(r['pct'])}\n{r['rank']} of {r['n']}",
                ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    ax.set_ylim(0, top * 1.28)
    ax.set_ylabel("mm per month (ensemble mean)", fontsize=9, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{A.name}: the {edd.MONTH_NAMES[im - 1]} {year} SEAS5 forecast by month (label: forecast vs hindcast mean, and rank, 1 = wettest)",
                 fontsize=10, color=C_TEXT, loc="left")
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    lo, hi = edd.SKILL_THRESH["r_mod"], edd.SKILL_THRESH["r_high"]
    for y0, y1, c, lab in [(-0.3, 0, "#f7ecea", "negative"), (0, lo, "#f3f7f6", "low"), (lo, hi, "#e3f1ec", "moderate"), (hi, 1, "#cfe7de", "high")]:
        sx.axhspan(y0, y1, color=c, zorder=0)
        sx.text(len(rows) - 0.45, (y0 + y1) / 2, lab, fontsize=7.5, color=C_MUTED, va="center", ha="right", style="italic")
    cols = [edd.SKILL_CATS.get(edd.skill_cat(r["r"]), "#cccccc") for r in rows]
    sx.bar(x, [r["r"] for r in rows], 0.5, color=cols, edgecolor="#5e6a6b", linewidth=0.6)
    for i, r in enumerate(rows):
        sx.text(x[i], r["r"] + (0.03 if r["r"] >= 0 else -0.03), f"{r['r']:+.2f}".replace("-", "−"), ha="center", va="bottom" if r["r"] >= 0 else "top", fontsize=7.5, color=C_TEXT)
    sx.axhline(0, color="#8a9495", lw=0.8)
    sx.set_ylim(-0.3, 0.8)
    sx.set_ylabel("skill (r)", fontsize=9, color=C_MUTED)
    sx.set_title(f"Skill of the {edd.MONTH_NAMES[im - 1]} issuance for each month: detrended r with ERA5 over {A.name}, 1981–{year - 1}",
                 fontsize=9.5, color=C_TEXT, loc="left")
    sx.set_xticks(x, labels, fontsize=9)
    edd._style_ax(sx)
    _save(fig, out)


def fig_storm_tiers(st: dict, djf: pd.Series, out: Path) -> None:
    """Storms per winter by tier (stacked, exclusive bins), IMERG 1998–, El Niño winters marked."""
    cols = {10: "#BFD9EE", 20: "#5E9FD2", 50: "#1F5F96"}
    c = st["ci"].loc[1998:2025]
    x = np.array(c.index)
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=150)
    b10 = (c[10] - c[20]).values; b20 = (c[20] - c[50]).values; b50 = c[50].values
    ax.bar(x, b10, 0.8, color=cols[10], edgecolor="white", linewidth=0.5, label=TIER_LABEL[10])
    ax.bar(x, b20, 0.8, bottom=b10, color=cols[20], edgecolor="white", linewidth=0.5, label=TIER_LABEL[20])
    ax.bar(x, b50, 0.8, bottom=b10 + b20, color=cols[50], edgecolor="white", linewidth=0.5, label=TIER_LABEL[50])
    en = (djf.reindex(c.index) >= ENSO_THRESH).values
    ax.plot(x[en], c[10].values[en] + 0.5, "v", color=C_EN, ms=5.5, mec="white", mew=0.6, ls="", label="El Niño winter")
    ax.set_xlim(x.min() - 0.8, x.max() + 0.8); ax.set_ylim(0, c[10].max() + 2)
    ticks = [y for y in x if y % 3 == 1]
    ax.set_xticks(ticks, [f"{y}/{str(y + 1)[2:]}" for y in ticks], fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("storms per winter", fontsize=9, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=4)
    ax.set_title(f"{A.name}: rain storms per October–April season, by the storm's wettest day (IMERG)", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def fig_first_storm(st: dict, djf: pd.Series, out: Path) -> None:
    """Date of each winter's first storm with a wettest day of 20 mm or more (IMERG), by ENSO phase."""
    e = st["ei"]; f = e[e["max"] >= 20].groupby("sy").start.min().reindex(range(1998, 2026))
    ph = pd.Series(phase_of(djf.reindex(f.index)), index=f.index)
    order = ["El Niño", "Neutral", "La Niña"]; colc = {"El Niño": C_EN, "Neutral": C_NEU, "La Niña": C_LN}
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=150)
    for i, p_ in enumerate(order):
        yrs = f.index[ph == p_]
        days = np.array([(f[y] - pd.Timestamp(y, 10, 1)).days if pd.notna(f[y]) else np.nan for y in yrs])
        jit = (np.arange(len(yrs)) % 3 - 1) * 0.12
        ax.plot(days, np.full(len(yrs), i) + jit, "o", color=colc[p_], ms=8, mec="white", mew=0.8, ls="")
        for y, d_, j in zip(yrs, days, jit):
            if y >= 2023 and np.isfinite(d_):
                ax.annotate(f"{y}/{str(y + 1)[2:]}", (d_, i + j), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=C_TEXT)
        n_before = int(np.nansum(days < 61)); ax.text(183, i, f"{n_before} of {len(yrs)} before 1 Dec", va="center", fontsize=8, color=C_MUTED)
    ax.axvline(61, color=C_TEXT, lw=1, ls=(0, (4, 3)))
    ax.text(62, 2.45, "1 December", fontsize=8, color=C_TEXT)
    ticks = [0, 31, 61, 92, 123, 151, 182]
    ax.set_xticks(ticks, ["1 Oct", "1 Nov", "1 Dec", "1 Jan", "1 Feb", "1 Mar", "1 Apr"], fontsize=8.5)
    ax.set_yticks(range(3), order, fontsize=9); ax.set_ylim(-0.6, 2.7); ax.set_xlim(-3, 225)
    ax.invert_yaxis()
    ax.set_title(f"{A.name}: date of each winter's first storm of 20 mm or more (IMERG, 1998/99–2025/26)", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(axis="x", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def load_events(path: Path, imerg: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """The curated event catalogue (deep_dives/data/*_events.csv) with the weather over each event's
    window [start − 1 day, end]: IMERG and ERA5 wettest day, ERA5 coldest night and strongest wind."""
    ev = pd.read_csv(path)
    ev["start"] = pd.to_datetime(ev["start"].fillna(ev["end"])); ev["end"] = pd.to_datetime(ev["end"])
    rows = []
    for _, r in ev.iterrows():
        a, b = r.start - pd.Timedelta(days=1), r.end
        w, x = imerg.loc[a:b], era5.loc[a:b]
        rows.append(dict(im_max=float(w.pr.max()) if len(w) else np.nan, era5_max=float(x.pr.max()) if len(x) else np.nan,
                         tmin=float(x.tmin.min()) if len(x) else np.nan, wmax=float(x.wmax.max()) if len(x) else np.nan))
    ev = pd.concat([ev, pd.DataFrame(rows, index=ev.index)], axis=1)
    scope = ev["hh_affected_scope"].fillna("").str.lower()
    ev["hh_complete"] = scope.str.contains("strip-wide")
    return ev


def fig_event_impacts(ev: pd.DataFrame, out: Path, storms_im: pd.DataFrame | None = None) -> None:
    """Households affected per event vs the event's wettest day over Gaza, 2024/25 and 2025/26.
    Filled = Gaza-wide alert counts; hollow = partial-site floors or response counts. A horizontal
    line joins the ERA5 and IMERG estimates of the wettest day (they disagree)."""
    scope = ev["hh_affected_scope"].fillna("").str.lower()
    dated = ~(scope.str.contains("response count") | scope.str.contains("rain date not stated") | scope.str.contains("mixed causes") | scope.str.contains("receiving packages")
              | scope.str.contains("carried over"))
    d = ev[ev.hh_affected_un.notna() & ev.winter.isin(["2024/25", "2025/26"]) & dated].copy()
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=150)
    col = {"2025/26": "#1F5F96", "2024/25": "#C0782F"}
    for _, r in d.iterrows():
        lo, hi = sorted([r.era5_max, r.im_max])
        ax.plot([lo, hi], [r.hh_affected_un] * 2, color=col[r.winter], lw=1.4, alpha=0.6, solid_capstyle="round")
        rain = "R" in r.hazard and r.im_max >= 10
        mk = "o" if rain else ("^" if "S" in r.hazard and "W" not in r.hazard else "s")
        ax.plot(r.im_max, r.hh_affected_un, mk, ms=8, color=col[r.winter], mfc=col[r.winter] if r.hh_complete else "white", mew=1.6)
    # storms of 20 mm or more in the three war winters with no dated UN household count: drawn at the floor
    if storms_im is not None:
        big = storms_im[(storms_im.sy >= 2023) & (storms_im["max"] >= 20)]
        k = 0
        for _, sr in big.sort_values("max").iterrows():
            hit = ((d.start - pd.Timedelta(days=1) <= sr.end) & (d.end >= sr.start)).any()
            if not hit:
                ax.plot(sr["max"], 115, "x", color=C_TEXT, ms=8, mew=1.8)
                ax.annotate(f"{sr.start:%b %Y}", (sr["max"], 115), xytext=(0, 8 + 10 * (k % 2)), textcoords="offset points",
                            ha="center", fontsize=7, color=C_MUTED)
                k += 1
    lab = {"GZ-2526-01": "first storm, 14–15 Nov 2025 (Gaza-wide)", "GZ-2526-04": "Byron, 8–17 Dec 2025 (Gaza-wide)",
           "GZ-2526-08": "12–16 Jan 2026: wind, cold (106 sites)",
           "GZ-2526-07": "9–10 Jan 2026 (34 northern sites)", "GZ-2526-18": "25–26 Mar 2026", "GZ-2526-16": "14 Mar 2026: sandstorm, wind",
           "GZ-2425-01": "24–25 Nov 2024: high tides", "GZ-2425-06": "5–6 Feb 2025: wind"}
    off = {"GZ-2425-06": (-6, -24), "GZ-2526-18": (8, -12)}
    for _, r in d.iterrows():
        if r.event_id in lab:
            ax.annotate(lab[r.event_id], (r.im_max, r.hh_affected_un), xytext=off.get(r.event_id, (7, 4)), textcoords="offset points",
                        fontsize=7.5, color=C_TEXT)
    ax.set_yscale("log"); ax.set_xlim(0, 72); ax.set_ylim(90, 150000)
    ax.set_yticks([100, 300, 1000, 3000, 10000, 30000, 100000], ["100", "300", "1,000", "3,000", "10,000", "30,000", "100,000"])
    ax.set_xlabel(f"wettest day of the event over {A.name}, mm (dot: IMERG; line extends to ERA5)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("households affected (log scale)", fontsize=9, color=C_MUTED)
    ax.axvline(20, color="#b8bfbf", lw=0.8, ls=(0, (3, 3))); ax.axvline(50, color="#b8bfbf", lw=0.8, ls=(0, (3, 3)))
    handles = [plt.Line2D([], [], marker="o", color=col["2025/26"], ls="", ms=7, label="2025/26"),
               plt.Line2D([], [], marker="o", color=col["2024/25"], ls="", ms=7, label="2024/25"),
               plt.Line2D([], [], marker="o", color=C_MUTED, mfc="white", ls="", ms=7, mew=1.4, label="hollow: partial-site floor or response count"),
               plt.Line2D([], [], marker="s", color=C_MUTED, ls="", ms=7, label="square: wind/cold with little rain"),
               plt.Line2D([], [], marker="^", color=C_MUTED, ls="", ms=7, label="triangle: high tide"),
               plt.Line2D([], [], marker="x", color=C_TEXT, ls="", ms=7, mew=1.6, label="storm ≥ 20 mm with no dated UN count")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    ax.text(0.99, 0.30, "events counted only as assistance delivered,\nwith no rain date, are left out", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=C_MUTED)
    ax.set_title(f"{A.name}: households affected per weather event vs that event's heaviest rain", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    _save(fig, out)


def fig_wet_composite(c: edd.Country, comp: np.ndarray, hit: np.ndarray, ok: np.ndarray, n_en: int,
                      y0: int, y1: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), dpi=150)
    fig.subplots_adjust(top=0.8, bottom=0.03, left=0.02, right=0.98, wspace=0.18)
    ext = edd._extent(c)
    grey = mcolors.ListedColormap(["#ececec"])
    m1 = edd._pcolor(axes[0], c, np.where(ok, comp, np.nan), edd.DIVERGING, -1.2, 1.2)
    edd._pcolor(axes[0], c, np.where(c.mask & ~ok, 0.0, np.nan), grey, -1, 1)
    m2 = edd._pcolor(axes[1], c, np.where(ok, hit * 100, np.nan), WET_SEQ, 0, 100)
    edd._pcolor(axes[1], c, np.where(c.mask & ~ok, 0.0, np.nan), grey, -1, 1)
    for ax in axes:
        edd._draw_country(ax, c, ext)
    axes[0].set_title(f"Mean Oct–Apr rainfall anomaly in the\n{n_en} El Niño winters (SD)", fontsize=9.5, loc="left", color=C_TEXT)
    axes[1].set_title("Share of El Niño winters in the cell's\nwettest third (chance = 33%)", fontsize=9.5, loc="left", color=C_TEXT)
    for m, ax, lab in [(m1, axes[0], "standard deviations"), (m2, axes[1], "% of El Niño winters")]:
        cb = fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=8, colors=C_MUTED)
        cb.set_label(lab, fontsize=9, color=C_MUTED)
    fig.suptitle(f"Southern Levant: what El Niño (DJF Niño3.4 ≥ +{ENSO_THRESH}) did to the rainy season,\nper ERA5 cell, {y0}/{str(y0 + 1)[2:]}–{y1}/{str(y1 + 1)[2:]}; "
                 f"{A.name} outlined; grey = too dry to analyse", fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.995, va="top")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _r(v) -> str:
    return edd.fmt_r(v)


def _p(v) -> str:
    return "—" if v is None or np.isnan(v) else ("&lt;0.001" if v < 0.001 else f"{v:.3f}")


def _pct_signed(v: float) -> str:
    return f"{100 * v:+.0f}%".replace("-", "−")


def _mm(v) -> str:
    return "—" if v is None or np.isnan(v) else f"{v:.0f} mm"


def _yr(y: int) -> str:
    return f"{y}/{str(y + 1)[2:]}"


def _ord(p: float) -> str:
    n = int(round(100 * p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def impact_table(rows: list[dict], numbered: bool) -> str:
    o = ['<div style="overflow-x:auto"><table><thead><tr>' + ('<th>#</th>' if numbered else '') +
         f'<th>Dates</th><th>Hazard</th><th>What was reported</th><th class="num">IMERG wettest day<br><span class="small">{A.name + (" · north · south" if A.imerg_split else "")}</span></th>'
         '<th class="num">Window total<br><span class="small">IMERG · ERA5</span></th><th class="num">ERA5 coldest night · strongest wind</th><th>Source</th></tr></thead><tbody>']
    for i, e in enumerate(rows, 1):
        d = e["start"] if e.get("end", e["start"]) == e["start"] else f'{e["start"]} to {e["end"]}'
        rain = ("—" if np.isnan(e["im_max"]) else f'{e["im_max"]:.0f} mm<br><span class="small">{e["im_max_n"]:.0f} · {e["im_max_s"]:.0f}</span>')
        met = "—" if np.isnan(e["tmin"]) else f'{e["tmin"]:.0f} °C · {e["wmax"]:.0f} m/s'
        o.append('<tr>' + (f'<td>{i}</td>' if numbered else '') + f'<td style="white-space:nowrap">{html.escape(d)}</td>'
                 f'<td>{html.escape(e.get("hazard", ""))}</td><td>{e["what_html"]}</td><td class="num">{rain}</td>'
                 f'<td class="num" style="white-space:nowrap">{_mm(e["im_sum"])} · {_mm(e["era5_sum"])}</td><td class="num" style="white-space:nowrap">{met}</td><td class="small">{e["source_html"]}</td></tr>')
    o.append('</tbody></table></div>')
    return "\n".join(o)


def _nword(n: int) -> str:
    return ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] if n <= 10 else str(n)


def _round_to(v: float, step: int) -> str:
    return f"{int(round(v / step) * step):,}"


def render_range(spec: dict, a: dict) -> str:
    """Section 5: how many storms a winter brings (IMERG), when the first damaging one comes, what one
    storm of each kind did last winter, and last winter as two reference points. No forecast totals."""
    st, ref, ev = a["st"], a.get("ref"), a.get("ev")
    T = spec.get("titles", {})
    row = {(r["src"], r["tier"], r["group"]): r for r in st["rows"]}
    fs = {(r["src"], r["group"]): r for r in st["first"]}
    o = [f'<h2>{html.escape(T.get("range", "5. What this winter could bring"))}</h2>', spec.get("range_intro_html", "")]
    # how many storms
    i10, i20, i50 = row[("IMERG", 10, "all")], row[("IMERG", 20, "all")], row[("IMERG", 50, "all")]
    e50, l50 = row[("IMERG", 50, "El Niño")], row[("IMERG", 50, "La Niña")]
    o.append('<h3>How many storms a winter brings</h3>')
    o.append(f'<p>Over the 28 winters of IMERG (1998/99–2025/26), a {A.name} winter brought on average {i10["mean"]:.0f} storms whose wettest day '
             f'reached 10 mm (in most winters {i10["p10"]:.0f}–{i10["p90"]:.0f}), {i20["mean"]:.0f} that reached 20 mm ({i20["p10"]:.0f}–{i20["p90"]:.0f}), '
             f'and a storm of 50 mm or more{", like Byron," if A.byron else ""} in {100 * i50["p_any"]:.0f}% of winters. El Niño winters brought slightly more of the '
             f'smaller storms ({row[("IMERG", 10, "El Niño")]["mean"]:.0f} of 10 mm or more, against {row[("IMERG", 10, "La Niña")]["mean"]:.0f} in La Niña winters) '
             f'but not more of the largest: {round(e50["p_any"] * e50["n"])} of {e50["n"]} El Niño winters had a storm of 50 mm or more, against '
             f'{round(l50["p_any"] * l50["n"])} of {l50["n"]} La Niña winters. Last winter, with {int(st["ci"].loc[2025, 10])}, {int(st["ci"].loc[2025, 20])} and '
             f'{int(st["ci"].loc[2025, 50])}, was among the more active.</p>')
    o.append('<figure><img src="storm_tiers.png" alt="Rain storms per winter by size"><figcaption>A storm is a run of rainy days (1 mm or more), '
             f'merged across lulls of up to two dry days, sized by its wettest day over {A.name} (IMERG, area-weighted). ' + ('Byron, 8–17 December 2025, '
             'counts as one storm (rain from 8 December, the storm proper from 10 December). ERA5 is not used for the largest storms: its 25 km '
             'cells smooth them so much that it gives Byron a wettest day of 18 mm, which would not place it among the record\'s 14 largest '
             'storms.' if A.byron else '') + '</figcaption></figure>')
    # first storm
    fe, fa = fs[("IMERG", "El Niño")], fs[("IMERG", "all")]
    o.append('<h3>When the first storm of 20 mm or more comes</h3>')
    o.append(spec.get("first_storm_html", ""))
    o.append(f'<figure><img src="first_storm.png" alt="Date of the first storm of 20 mm or more, by ENSO phase"><figcaption>Each dot is one winter: '
             f'the start date of its first storm with a wettest day of 20 mm or more over {A.name} (IMERG). Before 1 December in '
             f'{round(fe["p_before_dec"] * fe["n"])} of {fe["n"]} El Niño winters and {round(fa["p_before_dec"] * fa["n"])} of {fa["n"]} winters overall. '
             'Eight El Niño winters is a small sample: the El Niño share is within chance (Fisher exact test, p ≈ 0.2). 2007/08, a La Niña '
             'winter, had no storm of 20 mm or more and has no dot.</figcaption></figure>')
    # what one storm of each kind did
    if ev is not None and spec.get("storm_kinds"):
        e = ev.set_index("event_id")
        o.append('<h3>What one storm of each kind did last winter</h3>' + spec.get("storm_kinds_intro_html", ""))
        o.append('<div style="overflow-x:auto"><table><thead><tr><th>Kind of storm</th><th>Example</th><th class="num">Wettest day<br>'
                 '<span class="small">IMERG · ERA5</span></th><th class="num">Households affected</th><th class="num">Tents and shelters damaged</th>'
                 '<th>Deaths</th><th>How to read the numbers</th></tr></thead><tbody>')
        for k in spec["storm_kinds"]:
            r = e.loc[k["event_id"]]
            hh = "—" if pd.isna(r.hh_affected_un) else f'{k.get("hh_prefix", "")}{int(r.hh_affected_un):,}'
            tn = "—" if pd.isna(r.tents_damaged_un) else f'{k.get("tents_prefix", "")}{int(r.tents_damaged_un):,}'
            o.append(f'<tr><td><strong>{html.escape(k["kind"])}</strong></td><td style="white-space:nowrap">{html.escape(k["example"])}</td>'
                     f'<td class="num">{r.im_max:.0f} · {r.era5_max:.0f} mm</td><td class="num">{hh}</td><td class="num">{tn}</td>'
                     f'<td>{k.get("deaths_html", "")}</td><td class="small">{k["read_html"]}</td></tr>')
        o.append('</tbody></table></div>')
        o.append('<figure><img src="event_impacts.png" alt="Households affected per event against the event\'s heaviest rain"><figcaption>'
                 f'Every event of 2024/25 and 2025/26 with a dated UN or cluster household count. Filled markers are {A.name}-wide counts; hollow ones '
                 'cover only the sites assessed, so they are floors. Crosses at the bottom are storms of 20 mm or more since October 2023 for which no dated '
                 'household count was published. Impacts at the same rainfall differ tenfold between events: exposure, shelter condition and '
                 'what was counted matter as much as the rain.</figcaption></figure>')
    # reference points
    if ref:
        o.append('<h3>Last winter in numbers</h3>' + spec.get("reference_intro_html", ""))
        o.append('<table><thead><tr><th></th><th class="num">Household-impacts</th><th class="num">Tents and shelters damaged</th><th class="num">Deaths</th></tr></thead><tbody>')
        o.append(f'<tr><td>Byron alone (8–17 December 2025)</td><td class="num">≥ {ref["byron_hh"]:,.0f}</td>'
                 f'<td class="num">&gt; {ref["byron_tents"]:,.0f}</td><td class="num">12 (Health Cluster)</td></tr>')
        o.append(f'<tr><td>All other events</td><td class="num">about {_round_to(ref["rows_wo"], 5000)}–{_round_to(ref["snaps_wo"], 5000)}</td>'
                 f'<td class="num">about {_round_to(ref["tents_wo"], 500)}</td><td class="num">—</td></tr>')
        o.append(f'<tr class="hl"><td>Whole winter</td><td class="num">about {_round_to(ref["rows_all"], 5000)}–{_round_to(ref["snaps_all"], 5000)}</td>'
                 f'<td class="num">about {_round_to(ref["tents_all"], 500)}</td><td class="num">{ref["deaths"]:.0f}–{ref["deaths_hi"]:.0f}</td></tr>')
        o.append('</tbody></table>')
        o.append(f'<p class="small">Household-impacts count a household once for every storm that affected it; the same households were hit '
                 f'repeatedly. The lower figure sums the per-event counts; the higher one uses the Shelter Cluster\'s monthly snapshots '
                 f'(December {ref["dec"]:,.0f} families, January {ref["jan"]:,.0f}) with the first storm and the February–March events. '
                 f'“All other events” takes Byron\'s own counts ({ref["byron_hh"]:,.0f} households, {ref["byron_tents"]:,.0f} tents) out of both; '
                 f'for December that subtracts a Site Management Cluster count (132 sites) from a Shelter Cluster snapshot (537 sites), two sources '
                 f'with different coverage, so the remainder is approximate. '
                 f'Tents are the sum of the UN and cluster counts by event. Deaths: {ref["deaths"]:.0f} is the Ministry of Health\'s '
                 f'storm-related total to {pd.Timestamp(ref["deaths_asof"]):%-d %B %Y} (relayed by OCHA) plus the child cold deaths it reported afterwards '
                 f'({ref["cold_final"]:.0f} in all); the higher figure uses its other count of 25 collapse deaths for the same period. '
                 f'Neither is UN-verified.</p>')
    o.append(spec.get("range_after_html", ""))
    return "\n".join(o)


def render(spec: dict, a: dict) -> str:
    ours, cat = spec["assessment"], spec["catalogue"]
    T = lambda k, d: html.escape(spec.get("titles", {}).get(k, d))
    o = [edd.HEAD.format(title=f"{A.name} — ENSO deep dive", desc=html.escape(ours["one_line"]), css=edd.CSS,
                         home="../", home_label="ENSO country deep dives")]
    o.append(f'<p class="eyebrow">ENSO country deep dive</p><h1>{html.escape(A.name)}</h1>')
    o.append(f'<p class="meta">{html.escape(spec.get("subtitle", ""))} &nbsp;·&nbsp; GPCC 1891–2025, ERA5 1950–{a["end_era5"]:%Y}, '
             f'IMERG 1998–{a["end_imerg"]:%Y}, {A.meta_gauge} &nbsp;·&nbsp; Niño3.4 (NOAA)</p>')
    # verdict
    o.append('<div class="verdict">')
    o.append(f'<div class="card"><p class="lbl">Survey catalogue says</p><p class="big">El Niño → {html.escape(cat["direction"])}, {html.escape(cat["season"])}</p>'
             f'<p>{edd.chip(cat["evidence"])} <span class="small">source: {cat["source_html"]}</span></p></div>')
    o.append(f'<div class="card"><p class="lbl">This review assesses</p><p class="big">El Niño → {html.escape(ours["direction"])}, {html.escape(ours["season"])}</p>'
             f'<p>{edd.chip(ours["evidence"])} <span class="small">{html.escape(ours["evidence_note"])}</span></p></div>')
    o.append('</div>')
    o.append(f'<div class="summary">{spec["summary_html"]}</div>')
    if spec.get("before_html"):
        o.append(spec["before_html"])

    # 1. Forecasts
    nn = a["nino_now"]
    o.append(f'<h2>{T("forecast", "1. What the forecasts say for this winter")}</h2>')
    o.append(f'<p class="small">Latest Niño3.4 in NOAA PSL\'s current series: {nn["value"]:+.2f} °C for {nn["date"]:%B %Y} '
             f'(three-month mean {nn["mean3"]:+.2f}). The historical analysis on this page uses the series\' pinned Niño3.4 '
             f'(ERSST v5 basis, about 0.2 °C cooler than the current ERSST v6 series), and HadISST only where a record starts before 1950.</p>')
    o.append(spec.get("forecast_html", ""))
    fc = spec.get("forecasts", [])
    if fc:
        o.append('<div style="overflow-x:auto"><table><thead><tr><th>Forecast</th><th>Issued</th><th>Oct–Dec</th><th>Nov–Jan</th>'
                 '<th>Dec–Feb</th><th>Temperature</th></tr></thead><tbody>')
        for f in fc:
            o.append(f'<tr><td>{f["source_html"]}</td><td style="white-space:nowrap">{html.escape(f.get("issued", ""))}</td>'
                     f'<td>{f.get("ond", "")}</td><td>{f.get("ndj", "")}</td><td>{f.get("djf", "")}</td><td>{f.get("temp", "")}</td></tr>')
        o.append('</tbody></table></div>')
        o.append(spec.get("forecast_table_note", ""))
    if a.get("skill"):
        sk_html = (edd.render_skill(spec, a, "OND", spec.get("titles", {}).get("skill", f"SEAS5 for {A.name}: skill and the September forecast"))
                 .replace("whole country (bars) and zones (lines)", f"the cells over {A.name} (bars)")
                 .replace("for the whole country", f"for the cells over {A.name}")
                 .replace("of the's annual rain", f"of {A.name}'s annual rain")
                   )
        # the shared auto-summary names the zone before each window ("Gaza SON, Gaza OND"): drop it
        o.append(re.sub(rf"\b{re.escape(A.name)} (?=[A-Z]{{3}}\b)", "", sk_html))
    sr = a.get("seas5_raw")
    if sr:
        o.append(f'<p class="small">Cross-check from the raw ensemble-mean files (box 31.0–31.8°N, 34.0–34.8°E), {edd.MONTH_NAMES[sr["month"] - 1]} issuances '
                 f'1981–{sr["year"]}: where the {sr["year"]} forecast ranks among all {sr["rows"][0]["n"]} (1 = wettest), its ratio to the hindcast mean, '
                 'and the three wettest hindcast years.</p>')
        o.append('<table><thead><tr><th>Window</th><th class="num">Rank</th><th class="num">Ratio to mean</th><th>Wettest hindcasts (issuance year)</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{r["code"]}</td><td class="num">{r["rank"]} of {r["n"]}</td><td class="num">{r["ratio"]:.2f}</td>'
                           f'<td>{", ".join(str(y) for y in r["top"])}</td></tr>' for r in sr["rows"])
                 + '</tbody></table>')
    sm = a.get("seas5_mon")
    if sm:
        o.append(f'<h3>The same forecast, month by month</h3>{spec.get("monthly_html", "")}')
        o.append('<figure><img src="seas5_monthly.png" alt="SEAS5 forecast and skill by month"><figcaption>Top: the SEAS5 ensemble-mean '
                 f'rainfall for each month of the September issuance, averaged over a box on {A.name} ({A.seas5_box[1]:.1f}–{A.seas5_box[3]:.1f}°N, {A.seas5_box[0]:.1f}–{A.seas5_box[2]:.1f}°E), against the mean of '
                 'the same month in the 1981–2025 September hindcasts; labels give the forecast as a percentage above or below that mean and its '
                 'rank among all 46 September issuances (1 = wettest). Bottom: the skill of that month\'s forecast, the correlation between the '
                 f'detrended hindcast ensemble mean and detrended ERA5 rainfall over {A.name}, on the app\'s low / moderate / high bands. A single month '
                 'is noisier than a three-month window, so monthly skill is lower than in the figure above. September gets about 3 mm in an '
                 'average year, so its skill says little.</figcaption></figure>')
        o.append('<table><thead><tr><th>Month</th><th class="num">Hindcast mean</th><th class="num">Forecast</th><th class="num">vs mean</th>'
                 f'<th class="num">Rank (1 = wettest)</th><th class="num">Skill r</th><th class="num">ERA5 mean, {A.name}</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{edd.MONTH_NAMES[r["month"] - 1]}</td><td class="num">{r["hind"]:.0f} mm</td><td class="num">{r["fc"]:.0f} mm</td>'
                           f'<td class="num">{_pct_signed(r["pct"])}</td><td class="num">{r["rank"]} of {r["n"]}</td>'
                           f'<td class="num">{_r(r["r"])} {edd.skill_chip(edd.skill_cat(r["r"]))}</td><td class="num">{r["obs_clim"]:.0f} mm</td></tr>' for r in sm)
                 + '</tbody></table>')
        o.append('<p class="small">SEAS5 ensemble-mean totals are smoother than any single year, so compare the forecast with the hindcast mean and '
                 f'rank, not with observed rainfall. ERA5 mean: the {_nword(len(a["wts"]))} cells over {A.name}, same years, for scale.</p>')

    # 2. ENSO and the rainy season
    o.append(f'<h2>{T("enso", f"2. How much El Niño matters for a {A.name} winter")}</h2>')
    o.append(spec.get("enso_intro_html", ""))
    nw = len(a["wts"])
    o.append(f'<figure><img src="seasonal_cycle.png" alt="Monthly rainfall climatology for {A.name}"><figcaption>Monthly climatology from three '
             f'records. {A.gpcc_desc}'
             f'ERA5 is the mean of the {_nword(nw)} 0.25° cells over {A.name}, weighted by the share of {A.name} in each'
             + (f' ({", ".join(f"{100 * v:.0f}%" for v in a["wts"].values())})' if nw <= 4 else '')
             + f'; IMERG is the area-weighted mean of the 0.1° cells over {A.name}.'
             '</figcaption></figure>')
    o.append(f'<h3>The link switched on in the late 1970s</h3>{spec.get("stationarity_html", "")}')
    o.append(f'<figure><img src="stationarity.png" alt="Running correlation between {A.name} rainfall and Niño3.4"><figcaption>Pearson r between the '
             'October–April total and December–February Niño3.4 (HadISST) in centred 31-year windows. ' + A.stationarity_records + '</figcaption></figure>')
    per = a["per"]
    o.append('<table><thead><tr><th>Record</th><th>Winters</th><th>Niño3.4</th><th class="num">r</th><th class="num">p</th>'
             '<th class="num">r, detrended</th></tr></thead><tbody>')
    for k in ("GPCC", "gauge", "ERA5", "IMERG"):
        for tag in ("pre", "post", "recent", "full"):
            v = per.get((k, tag))
            if not v or (k == "IMERG" and tag != "recent") or (tag == "recent" and k == "gauge"):
                continue
            cls = ' class="hl"' if tag == "post" else ""
            o.append(f'<tr{cls}><td>{html.escape(SRC_LABEL[k])}</td><td>{_yr(v["first"])} – {_yr(v["last"])}</td><td class="small">{v["index"]}</td>'
                     f'<td class="num">{_r(v["r"])}</td><td class="num">{_p(v["p"])}</td><td class="num">{_r(v["r_detr"])}</td></tr>')
    o.append('</tbody></table>')
    dd = a["diff"]
    o.append('<p class="small">Is the change real? A Fisher z-test for the difference between the correlations before and after '
             f'{SPLIT}: ' + "; ".join(f'{html.escape(_disp(k))} z = {v["z"]:.1f}, p {"&lt; 0.001" if v["p"] < 0.001 else "= " + _p(v["p"])}' for k, v in dd.items()) + '. '
             f'The {SPLIT} break was not chosen blind: it is where the literature places the change (Price et al. 1998; Alpert et al. 2005) '
             'and the start of the satellite era. p-values for the period after the break are conditional on choosing it; '
             'the full-record rows show what an unsplit analysis gives.</p>')
    o.append(f'<h3>What El Niño winters have looked like since {SPLIT}</h3>{spec.get("history_html", "")}')
    o.append('<figure><img src="phase_history.png" alt="Rainy-season totals by ENSO phase"><figcaption>October–April GPCC totals as a percentage '
             'of the 1991–2020 mean, coloured by the ENSO phase of the same winter (December–February Niño3.4, pinned ERSST v5 series throughout, so '
             'a few 1950s–70s winters are classed differently from the HadISST-based table below). Winters with Niño3.4 ≥ +1.5 °C are labelled; '
             f'the dashed line marks {SPLIT}.</figcaption></figure>')
    o.append('<table><thead><tr><th>Record, period</th><th>ENSO phase</th><th class="num">Winters</th><th class="num">Wettest third</th>'
             '<th class="num">Middle third</th><th class="num">Driest third</th></tr></thead><tbody>')
    for (k, lo, hi), rows in a["ptab"].items():
        for r in rows:
            cls = ' class="hl"' if (r["phase"] == "El Niño" and lo == SPLIT) else ""
            o.append(f'<tr{cls}><td>{html.escape(k)}, {_yr(lo)} – {_yr(min(hi, int(a["tot"][k].index.max())))}</td><td>{r["phase"]}</td><td class="num">{r["n"]}</td>'
                     f'<td class="num">{r["wet"]}</td><td class="num">{r["mid"]}</td><td class="num">{r["dry"]}</td></tr>')
    o.append('</tbody></table>')
    o.append('<p class="small">Terciles are computed within each period, so each period is judged against its own climate. '
             f'Phase from December–February Niño3.4 (±0.5 °C): HadISST before {SPLIT}, the pinned ERSST v5 series from {SPLIT}. '
             f'On HadISST, {a["en_had"]["n"]} winters since {SPLIT} count as El Niño rather than 14, and {a["en_had"]["wet"]} of them '
             'were in the GPCC wettest third.</p>')
    o.append('<h3>Every strong El Niño winter since 1950</h3>' + spec.get("strong_html", ""))
    o.append('<table><thead><tr><th>Winter</th><th class="num">Niño3.4 DJF</th>'
             + "".join(f'<th class="num">{_short(k)}</th>' for k in ("GPCC", "ERA5", "gauge", "IMERG"))
             + '<th>Note</th></tr></thead><tbody>')
    for s_ in a["strong"]:
        cells = []
        for k in ("GPCC", "ERA5", "gauge", "IMERG"):
            v = s_.get(k)
            cells.append('<td class="num">—</td>' if not v else
                         f'<td class="num">{v["mm"]:.0f} mm<br><span class="small">{_ord(v["pct"])} pct · {v["third"]}</span></td>')
        note = spec.get("strong_notes", {}).get(str(s_["year"]), "")
        o.append(f'<tr><td>{_yr(s_["year"])}</td><td class="num">{s_["nino"]:+.1f}</td>{"".join(cells)}<td class="small">{note}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">Niño3.4 ≥ +1.5 °C in December–February on the pinned series. Percentiles are within {SPLIT}–2025 for winters '
             f'from {SPLIT}, within 1950–{SPLIT - 1} before.</p>')
    o.append(f'<h3>Which part of the winter</h3>{spec.get("trimester_html", "")}')
    o.append('<table><thead><tr><th>Record</th><th>Window</th><th class="num">r, concurrent Niño3.4</th><th class="num">p</th>'
             '<th class="num">r, Aug–Oct Niño3.4</th><th class="num">p</th></tr></thead><tbody>')
    for t in a["tri_rows"]:
        if t["src"] == "IMERG":
            continue
        cls = ' class="hl"' if t["season"] == "Oct–Apr" else ""
        o.append(f'<tr{cls}><td>{t["src"]}</td><td>{t["season"]}</td><td class="num">{_r(t["r"])}</td><td class="num">{_p(t["p"])}</td>'
                 f'<td class="num">{_r(t["r_aso"])}</td><td class="num">{_p(t["p_aso"])}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">{SPLIT}–2025. Concurrent = Niño3.4 averaged over the same three months (December–February for October–April). '
             'August–October Niño3.4 is the value known when the season starts.</p>')
    o.append(f'<h3>Across the region</h3>{spec.get("maps_html", "")}')
    o.append('<figure><img src="corr_maps.png" alt="Pixel correlation maps, southern Levant"><figcaption>Pearson r between three-month rainfall and '
             'Niño3.4 for each ERA5 0.25° cell, keeping the lag (0–3 months, index leading) with the largest |r| as the survey does. '
             f'Blue = wetter under El Niño. Grey cells hold under a quarter of their annual rain in that window. {A.name} is outlined.</figcaption></figure>')
    o.append(f'<figure><img src="composite_maps.png" alt="El Niño composite and wettest-third hit rate"><figcaption>Left: mean standardised '
             f'October–April anomaly over the {a["n_en_grid"]} El Niño winters of {_yr(a["n_grid"][0])}–{_yr(a["n_grid"][1])} '
             f'(cells over {A.name}: {a["comp_gaza"]:+.2f} SD; positive in {100 * a["comp_region_pos"]:.0f}% of analysed cells). '
             f'Right: the share of those winters in each cell\'s wettest third ({A.name}: {100 * a["hit_gaza"]:.0f}%; regional median '
             f'{100 * a["hit_region"]:.0f}%; chance is 33%).</figcaption></figure>')

    # 3. Daily weather
    o.append(f'<h2>{T("daily", "3. What changes in an El Niño winter, and what does not")}</h2>')
    o.append(spec.get("daily_html", ""))
    o.append('<figure><img src="daily_by_phase.png" alt="Correlation of winter weather metrics with Niño3.4"><figcaption>Each dot is the '
             'correlation between one October–April metric and December–February Niño3.4 across the winters since '
             f'{SPLIT} (IMERG from 1998, {A.gauge_short} to {a["gauge_last"]}); lines are 95% intervals. Windy days: days whose highest hourly ERA5 10 m wind reaches '
             f'the top 5% of winter days ({a["wind95"]:.1f} m/s; ERA5 winds are cell averages and understate gusts). Nights below 5 °C: '
             + A.daily_cold_note + '</figcaption></figure>')
    o.append('<table><thead><tr><th>Metric</th><th>Record</th><th class="num">El Niño</th><th class="num">Neutral</th><th class="num">La Niña</th>'
             '<th class="num">r</th><th class="num">p</th></tr></thead><tbody>')
    unit = {"total": " mm", "rx1": " mm"}
    for mt in ["total", "d1", "d10", "d20", "rx1", "cold", "windy"]:
        for r in [r for r in a["daily_rows"] if r["metric"] == mt]:
            u = unit.get(mt, "")
            o.append(f'<tr><td>{_metric_label(mt)}</td><td>{html.escape(_disp(r["src"]))} {r["first"]}–{r["last"]}</td>'
                     f'<td class="num">{r["en"]:.1f}{u}</td><td class="num">{r["neu"]:.1f}{u}</td><td class="num">{r["ln"]:.1f}{u}</td>'
                     f'<td class="num">{_r(r["r"])}</td><td class="num">{_p(r["p"])}</td></tr>')
    o.append('</tbody></table>')
    o.append('<p class="small">Means per winter, by ENSO phase. The records differ in level: '
             + A.daily_note + ' Compare phases within a record, not across records.</p>')

    # 4. Impacts
    o.append(f'<h2>{T("impacts", f"4. What winter weather does in {A.name}")}</h2>')
    o.append(spec.get("impacts_intro_html", ""))
    fr = a["freq"]
    o.append(f'<p class="small">How often the rain that has caused these impacts comes: over 1998–2025, IMERG puts an average of '
             f'{fr[10]:.1f} days of ≥ 10 mm, {fr[20]:.1f} of ≥ 20 mm, {fr[30]:.1f} of ≥ 30 mm and {fr[50]:.1f} of ≥ 50 mm over {A.name} in each October–April.</p>')
    o.append(f'<figure><img src="winters.png" alt="Daily rainfall over {A.name}, 2023/24 to 2025/26, with impacts"><figcaption>IMERG late run, '
             f'area-weighted mean over {A.name}. Markers are the start dates of the reported impacts in the table below.</figcaption></figure>')
    o.append(impact_table(a["ev_rows"], numbered=True))
    o.append(f'<p class="small">{A.impact_caption}</p>')
    if a.get("pre_rows"):
        o.append(f'<h3>Before the war</h3>{spec.get("prewar_html", "")}')
        o.append(impact_table(a["pre_rows"], numbered=False))
    o.append(spec.get("impacts_after_html", ""))
    if a.get("st"):
        o.append(render_range(spec, a))
    for sec in spec.get("sections_after", []):
        o.append(f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}')
    if spec.get("references"):
        o.append('<h2>References</h2><ul class="refs">' + "".join(f'<li>{r["html"]}</li>' for r in spec["references"]) + '</ul>')
    o.append(f'<p class="small">Generated by <code>levant_deep_dive.py</code> from <code>deep_dives/{A.slug}.toml</code>. '
             'Grid method and Niño3.4 series as in the <a href="../../survey/">global survey</a>.</p>')
    o.append(edd.FOOT)
    return "\n".join(o)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def build(spec: dict, grid: edd.Grid, ne: gpd.GeoDataFrame, indices: pd.DataFrame) -> dict:
    """Called by enso_deep_dive.main() for a TOML with builder = "gaza".

    enso_deep_dive runs as __main__, so its module globals (END_YEAR, NINO_LATEST) are not the ones
    this module sees through `import enso_deep_dive`; set them here from the same sources."""
    set_area(spec)
    edd.END_YEAR = int(grid.years[-1])
    latest = ts.CONFIG["cache_dir"] / "nino34_latest.data"
    edd.NINO_LATEST = ts._parse_psl(latest.read_text()) if latest.exists() else None
    a = analyse(spec, grid, ne, indices)
    (OUT / "index.html").write_text(render(spec, a), encoding="utf-8")
    return a


def main() -> None:
    import sys
    slug = sys.argv[1] if len(sys.argv) > 1 else "gaza"
    cfg = dict(ts.CONFIG, max_lag=3)
    spec = tomllib.loads((edd.DEEP_DIR / f"{slug}.toml").read_text()) | {"slug": slug}
    edd.extend_pixel_cache(cfg)
    grid = edd.load_grid(cfg)
    build(spec, grid, ts.load_admin0_gdf(cfg), ts.load_indices(cfg))
    print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
