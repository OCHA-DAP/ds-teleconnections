"""Gaza and the West Bank — ENSO deep dive (builder for deep_dives/gaza-west-bank.toml and its parts).

The generic builder in enso_deep_dive.py works on a country's ERA5 cells and a drought framing.
Gaza and the West Bank do not fit either: both are a handful of 0.25° ERA5 cells, and the winter
hazard is too much rain (and sea, wind, cold or snow), not too little. This module reuses the
series' page chrome, SEAS5 skill figure and pixel correlation maps, and adds:

* independent rainfall records, because a few reanalysis cells are not enough evidence: ERA5
  hourly point series for the cells over the area (CDS time-series dataset, 1950–), GPCC gauge
  analysis (1° cells, 1891–), IMERG late v7 daily at 0.1° (team blob, 1998–), and a long rain
  gauge (GHCN-Daily: Beer Sheva for Gaza, Jerusalem for the West Bank);
* a stationarity test, because the El Niño link in this region is known to switch on and off;
* daily metrics (rain days, heavy-rain days, wettest day, cold nights, wind) by ENSO phase;
* storm counts, first-storm timing and the last winters' rainfall against reported impacts;
* for the West Bank ([agri] in its TOML), crop years against rain and El Niño by agro-ecological zone
  (levant_agri.py: FAOSTAT, PCBS, NOAA STAR vegetation health, MODIS NDVI).

Both areas share one page, deep_dives/gaza-west-bank.toml (builder "levant-combined"); each area is a part,
deep_dives/parts/<slug>.toml, whose [area] table overrides the Area dataclass (Gaza defaults).

    uv run python levant_deep_dive.py gaza-west-bank   # or: uv run python enso_deep_dive.py --only gaza-west-bank

Downloads are cached under the area's cache directory. CDS needs a key in ~/.cdsapirc (any ECMWF
data store URL; the CDS endpoint is forced here); IMERG and the COD-AB need the team's blob credentials.
"""
from __future__ import annotations

import html
import io
import re
import textwrap
import tomllib
import unicodedata
import zipfile
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    ref: str = "Gaza"                                          # the name in running prose ("the West Bank")
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
    seas5_pcode: str = "PS02"                                  # this area in the team's raster stats (public.seas5, admin 1)
    era5_cold: bool = False                                    # ERA5 cells partly sea: no cold-night metric
    cold_thresh: float = 5.0
    cold_snap_thresh: float = 8.0                              # ERA5 night minimum counted as a cold night in the ENSO comparison
    stationarity_records: str = ("Two gauge-based records (the GPCC analysis, and the Beer Sheva gauge 46 km inland, which is one of GPCC\'s "
                                 "inputs) and a reanalysis that assimilates no rain gauges (ERA5).")
    daily_cold_note: str = "the Beer Sheva gauge only, because the ERA5 cells are partly sea and rarely get that cold."
    daily_note: str = ("ERA5 spreads rain over a 25 km cell and understates heavy days; IMERG is satellite-only; Beer Sheva is drier "
                       "and 46 km inland.")
    byron: bool = True                                         # Gaza's Byron-specific wording in section 5
    event_sources: str = ("Counts: OCHA, UNRWA and UNICEF reports before October 2023; since, counts published by OCHA, mostly from the "
                          "Site Management Cluster and the Shelter Cluster.")
    impact_caption: str = ("Hazard: R rain and flooding, S sea surge or high tide, W wind, C cold. Weather columns cover the window from the day "
                           "before the reported start to the reported end: IMERG wettest day over Gaza (north = cells at 31.4–31.6°N: North Gaza, "
                           "Gaza governorate and most of Deir al Balah; south = 31.2–31.4°N: Khan Younis and Rafah); window totals from IMERG and from ERA5 (which smooths rain over 25 km cells and "
                           "runs lower on heavy days; the two disagree on single days by a factor of two or more, so read them as a range); ERA5 lowest daily minimum temperature and highest hourly 10 m wind, averaged "
                           "over the three cells over Gaza (partly sea, so milder and less gusty than an exposed displacement site). Figures in <em>italics</em> "
                           "are from the Ministry of Health in Gaza, the Palestinian Civil Defense or the Government Media Office, as relayed by the UN; the rest are UN "
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
    OUT = Path(spec["out_dir"]) if spec.get("out_dir") else edd.OUT_DIR / A.slug
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
    d = pd.read_csv(path, usecols=lambda c: c in ("DATE", "PRCP", "TMIN"), parse_dates=["DATE"]).set_index("DATE")
    return pd.DataFrame({"pr": d.PRCP / 10.0, "tmin": d.TMIN / 10.0 if "TMIN" in d else np.nan}, index=d.index)


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
    path = COD_CACHE / "nino34_long_hadisst.data"     # shared, not per area
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
    outline = gaza                                 # the maps outline this area, or every area of a combined page
    if spec.get("map_outline"):
        a1 = gpd.read_parquet(COD_CACHE / "pse_adm1.parquet")
        outline = a1[a1.ADM1_PCODE.isin(spec["map_outline"])]
    c = region_country(grid, outline, ne)
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
    fig_wet_composite(c, comp, hit, ok_wet, int(en.sum()), int(sy.min()), int(sy.max()), OUT / "composite_maps.png", spec.get("map_label"))

    # 6. SEAS5: skill + current forecast for the cells over Gaza
    gc = edd.Country(iso3="GAZ", geom=gaza.geometry, lat=c.lat, lon=c.lon, mask=gz_cells, sub=c.sub, neighbours=c.neighbours)
    skill = edd.seas5_skill_issued(gc, {A.name: gz_cells}, int(spec.get("skill_issued_month", 9)))
    # The app's pixel cube is recomputed a day after SEAS5 lands. Until then its forecast layers belong to an older
    # issuance: take the return periods of the new one from the team's admin-1 raster stats instead (same method),
    # keep the cube's hindcast skill, and blank the in-season windows, which need the latest ERA5 month.
    db = seas5_db_trimesters(A.seas5_pcode, int(spec.get("skill_issued_month", 9))) if skill else None
    if skill and db and (skill.get("issued_year") or 0) < db["year"]:
        row = skill["rows"][0]
        for t in skill["trimesters"]:
            v = db["rows"].get(t["code"]) if t["lead"] >= 0 else None
            row["rp"][t["code"]] = v["rp"] if v else float("nan")
            row["pct"][t["code"]] = v["pct"] if v else float("nan")
        skill.update(cube_year=skill.get("issued_year"), issued_year=db["year"], db=db)
    elif skill and db:
        print(f"  note: the skill cube holds the {skill.get('issued_year')} issuance, so SEAS5 return periods are pixel medians from the cube "
              "again; re-check the SEAS5 paragraphs, summary item 1 and key message 1 against the regenerated tables.")
    now = pd.Timestamp.now()
    im_ = int(spec.get("skill_issued_month", 9))
    if db and (im_ < now.month or (im_ == now.month and now.day >= 5)) and db["year"] != now.year:
        print(f"  WARNING: the cached raster stats end before the {now.year}-{im_:02d} issuance (latest for that month: {db['year']}); "
              "the SEAS5 forecast on the page is not this year's.")
    if skill:
        edd.fig_skill_issued(gc, grid, {}, skill, OUT / "skill_issued.png", A.name, area_label=A.name,
                             rp_label="admin-1 mean, team database" if skill.get("db") else "median pixel")

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
    week = week_ladder(imerg.pr, bs.pr)
    cold = cold_stats(era5.tmin, djf_pin)
    cold_enso = cold_by_enso(era5.tmin, djf_pin, A.cold_snap_thresh)
    data_dir = edd.DEEP_DIR / "data"
    ev = load_events(data_dir / f"{A.slug}_events.csv", imerg, era5) if (data_dir / f"{A.slug}_events.csv").exists() else None
    totals = pd.read_csv(data_dir / f"{A.slug}_season_totals.csv") if (data_dir / f"{A.slug}_season_totals.csv").exists() else None
    ref = reference_winters(ev, totals, st["ei"]) if (ev is not None and totals is not None) else None
    evp = load_events(data_dir / f"{A.slug}_prewar_events.csv", imerg, era5) if (data_dir / f"{A.slug}_prewar_events.csv").exists() else None
    wi = (winter_impacts(evp, ev, imerg.pr, st, djf_pin, ref, spec["winter_impacts"], dm=dm, era5=era5, db=db)
          if (evp is not None and ref is not None and ref["byron_ids"] and spec.get("winter_impacts")) else None)
    fig_storm_tiers(st, djf_pin, OUT / "storm_tiers.png")
    fig_first_storm(st, djf_pin, OUT / "first_storm.png")
    if ev is not None:
        fig_event_impacts(ev, OUT / "event_impacts.png", st["ei"])
    if wi is not None:
        fig_winter_impacts(wi, OUT / "winter_impacts.png")
        if "rx1" in wi["S"]:
            fig_winter_hazards(wi, OUT / "winter_hazards.png")
        fig_event_hazards(wi, st, OUT / "event_hazards.png")
        if wi["ond"] and wi["ond"]["top"]:
            fig_forecast_quarter(wi, OUT / "forecast_quarter.png")
    bars = winter_counts(spec, wi, djf_pin)
    if bars is not None:
        fig_winter_counts(bars, OUT / "winter_counts.png")

    # Rainfall on the dates of reported impacts, and how often such days occur
    ev_rows = impact_rain(events, imerg, era5)
    pre_rows = impact_rain(spec.get("prewar", []), imerg, era5)
    wet_days = imerg[imerg.index.month.isin(WET)].pr
    freq = {t: float((wet_days >= t).groupby(season_year(wet_days.index)).sum().loc[1998:2025].mean()) for t in (10, 20, 30, 50)}

    nn = edd.NINO_LATEST.dropna() if edd.NINO_LATEST is not None else n_pin
    nn = nn[nn > -90]
    out = dict(week=week, wts=wts, per=per, run=run, tri_rows=tri_rows, ptab=ptab, strong=strong, daily_rows=daily_rows,
                wind95=wind95, map_rows=map_rows, diff=diff, en_had=en_had, st=st, cold=cold, cold_enso=cold_enso, bars=bars, ev=ev, ref=ref, wi=wi, gauge_last=int(dm["gauge"].index.max()), seas5_raw=seas5_raw, seas5_mon=seas5_mon, skill=skill, tot=tot, djf=djf_pin, ev_rows=ev_rows, pre_rows=pre_rows, freq=freq,
                end_era5=end_era5, end_imerg=imerg.index[-1], n_grid=(int(sy.min()), int(sy.max())), n_en_grid=int(en.sum()),
                comp_gaza=float(np.nanmean(comp[gz_cells])), hit_gaza=float(np.nanmean(hit[gz_cells])),
                hit_region=float(np.nanmedian(hit[ok_wet])), comp_region_pos=float((comp[ok_wet] > 0).mean()),
                nino_now=dict(value=float(nn.iloc[-1]), date=nn.index[-1], mean3=float(nn.iloc[-3:].mean())),
                imerg_last=imerg, era5=era5)
    if spec.get("agri"):                          # West Bank: crop years against rain and El Niño
        import levant_agri
        out["agri"] = levant_agri.analyse(spec, out)
    return out


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


SEAS5_DB_CACHE = COD_CACHE / "seas5_adm1_db.parquet"      # prod public.seas5, admin 1, both areas
ERA5_DB_CACHE = COD_CACHE / "era5_adm1_db.parquet"        # prod public.era5, admin 1


def seas5_db_tables(iso3: str = "PSE") -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The team's admin-1 raster stats for SEAS5 (monthly ensemble mean by issuance and lead, mm/day) and ERA5
    (monthly mean, mm/day), from the prod database, cached. The database is reachable only inside the VNet or
    through the laptop tunnel (DSCI_AZ_DB_PROD_HOST pointing at it), so it is queried only when the cache lacks the
    issuance that should exist by now, and a failed connection falls back to the cache."""
    now = pd.Timestamp.now()
    expected = (now if now.day >= 5 else now - pd.DateOffset(months=1)).to_period("M").to_timestamp()   # SEAS5 lands on the 5th
    stale = not SEAS5_DB_CACHE.exists() or pd.read_parquet(SEAS5_DB_CACHE, columns=["issued_date"]).issued_date.max() < expected
    if stale:
        try:
            import ocha_stratus as stratus
            with stratus.get_engine("prod").connect() as c:
                s = pd.read_sql("SELECT pcode, issued_date, valid_date, leadtime, mean FROM public.seas5 WHERE iso3 = %s AND adm_level = 1",
                                c, params=(iso3,), parse_dates=["issued_date", "valid_date"])
                e = pd.read_sql("SELECT pcode, valid_date, mean FROM public.era5 WHERE iso3 = %s AND adm_level = 1",
                                c, params=(iso3,), parse_dates=["valid_date"])
            s.to_parquet(SEAS5_DB_CACHE); e.to_parquet(ERA5_DB_CACHE)
        except Exception as ex:  # noqa: BLE001
            print(f"  (prod database not reachable, using the cached raster stats: {ex})")
    if not (SEAS5_DB_CACHE.exists() and ERA5_DB_CACHE.exists()):
        return None
    return pd.read_parquet(SEAS5_DB_CACHE), pd.read_parquet(ERA5_DB_CACHE)


def seas5_db_trimesters(pcode: str, issued_month: int) -> dict | None:
    """What the latest issuance of `issued_month` forecasts for one admin-1 unit, per fully forecast trimester
    (leads 0–4), computed from the team's raster stats the way the seas5-skill app computes its admin-level
    product (src/skill.py, detrended): trimester means per issuance year, log1p, SEAS5 scaled to ERA5's mean
    and spread over the overlap years, both detrended linearly, then the latest forecast's Weibull rank among
    the hindcasts. Used while the app's pixel cube still holds the previous issuance. In-season trimesters are
    left out: they blend observed months, and ERA5 for the month before an issuance lands a day after SEAS5."""
    tabs = seas5_db_tables()
    if tabs is None:
        return None
    s, e = (t[t.pcode == pcode].dropna(subset=["mean"]) for t in tabs)
    s = s[s.issued_date.dt.month == issued_month]
    if s.empty:
        return None
    issued_year = int(s.issued_date.dt.year.max())
    out = {}
    for code, months in ts._TRIMESTER_MONTHS.items():
        lead = (months[0] - issued_month) % 12
        if lead > 4:
            continue
        wraps = 1 in months and 12 in months
        f = s[s.valid_date.dt.month.isin(months)]
        f = f.groupby(f.issued_date.dt.year)["mean"].mean()
        f.index = f.index + (1 if (not wraps and min(months) < issued_month) else 0)           # season year
        o = e[e.valid_date.dt.month.isin(months)]
        sy = np.where((o.valid_date.dt.month > 6) | (not wraps), o.valid_date.dt.year, o.valid_date.dt.year - 1)
        g = o.groupby(sy)["mean"]
        mm = (o["mean"] * o.valid_date.dt.days_in_month).groupby(sy).sum()[g.count() == 3]      # ERA5 trimester total, mm
        o = g.mean()[g.count() == 3]
        f, o = np.log1p(f.clip(lower=0)), np.log1p(o.clip(lower=0))
        hist = f.index.intersection(o.index)
        if len(hist) < 10:
            continue
        f = (f - f[hist].mean()) / max(f[hist].std(ddof=1), 1e-9) * o[hist].std(ddof=1) + o[hist].mean()
        x = hist.values.astype(float)
        a_, b_ = np.polyfit(x, f[hist].values, 1)
        f = f - (a_ * f.index.values.astype(float) + b_) + f[hist].mean()
        a_, b_ = np.polyfit(x, o[hist].values, 1)
        o = o - (a_ * o.index.values.astype(float) + b_) + o[hist].mean()
        cur_year = int(f.index.max())
        if cur_year in o.index:                    # the latest forecast must be a real forecast, not a past season
            continue
        cur, h = float(f[cur_year]), f[hist].values
        dry, wet = (len(h) + 1) / (int((h < cur).sum()) + 1), (len(h) + 1) / (int((h > cur).sum()) + 1)
        pct_ = 100.0 * float((h <= cur).sum()) / len(h)
        out[code] = dict(lead=lead, r=float(np.corrcoef(f[hist].values, o[hist].values)[0, 1]), n=len(hist),
                         pct=pct_, rp=dry if pct_ < 50 else -wet, season_year=cur_year, cur=cur,
                         hind=pd.DataFrame(dict(f=f[hist], mm=mm[hist])))
    return dict(year=issued_year, month=issued_month, pcode=pcode, rows=out) if out else None


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


def week_ladder(imerg: pd.Series, gauge: pd.Series) -> dict:
    """The wettest 7 days of each winter (rolling sum of daily totals, by season year), in mm and as a share of the
    source's own mean annual (August–July) total: the scale against which a 7-day forecast total can be read."""
    out = {}
    for k, d in (("IMERG", imerg.loc["1998-08":]), ("gauge", gauge.dropna())):
        sy = season_year(d.index)
        cnt = d.groupby(sy).count()
        full = cnt.index[cnt >= 330]
        ann = d.groupby(sy).sum().loc[full]
        mx = d.rolling(7).sum().groupby(season_year(d.index)).max()
        if k == "IMERG":                              # every complete winter (to April), including the last one
            last = int(season_year(d.index[-1:])[0]) - (0 if d.index[-1].month in (5, 6, 7) else 1)
            mx = mx.loc[1998:last]
        else:
            mx = mx.loc[full]
        m = float(ann.mean())
        out[k] = dict(mean=m, first=int(ann.index.min()), last=int(ann.index.max()), n=len(mx),
                      med=float(mx.median()), q80=float(mx.quantile(0.8)), mx=float(mx.max()), mx_year=int(mx.idxmax()),
                      share30=int((mx >= 0.3 * m).sum()))
    b = imerg.loc["2025-12-08":"2025-12-21"].rolling(7).sum()
    out["byron"] = dict(mm=float(b.max()), end=b.idxmax()) if b.notna().any() else None
    return out


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
    return dict(byron_ids=byron_ids, rows_all=rows_all, snaps_all=snaps_all, rows_wo=rows_all - hh_b, snaps_wo=snaps_all - hh_b,
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
                         im_max_n=float(w.north.max()) if (len(w) and "north" in w) else np.nan,
                         im_max_s=float(w.south.max()) if (len(w) and "south" in w) else np.nan,
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


TOTAL_TIERS = [25, 50, 100]                   # a storm's total rain over the area (IMERG mm)


def storm_counts(e: pd.DataFrame, years, thresholds: dict[int, float], col: str = "max") -> pd.DataFrame:
    """Storms per winter at or above each tier's threshold (tier key → threshold in this record's mm), sized by
    the storm's wettest day (`max`) or its total rain (`total`)."""
    return pd.DataFrame({t: e[e[col] >= thr].groupby("sy").size().reindex(years, fill_value=0)
                         for t, thr in thresholds.items()})


def matched_thresholds(e_ref: pd.DataFrame, e_other: pd.DataFrame, lo: int, hi: int, tiers=None, col: str = "max") -> dict[int, float]:
    """Thresholds in `e_other` (ERA5) that give the same number of storms over [lo, hi] as each IMERG
    tier does in `e_ref` (frequency matching: ERA5 spreads rain over 25 km cells and runs lower)."""
    r = e_ref[(e_ref.sy >= lo) & (e_ref.sy <= hi)]; o = np.sort(e_other[(e_other.sy >= lo) & (e_other.sy <= hi)][col].values)[::-1]
    return {t: float(o[int((r[col] >= t).sum()) - 1]) for t in (TIERS if tiers is None else tiers)}


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
    # the same count by a storm's total rain, by phase: what an El Niño winter has brought against the others
    thr_t = matched_thresholds(ei, ee, 1998, 2025, TOTAL_TIERS, "total")
    cti = storm_counts(ei, range(1998, 2026), {t: float(t) for t in TOTAL_TIERS}, "total")
    cte = storm_counts(ee, range(1950, 2026), thr_t, "total")
    by_total = []
    for name, c, lo in [("IMERG", cti, 1998), ("ERA5", cte, SPLIT)]:
        c = c.loc[lo:2025]; ph = pd.Series(phase_of(djf.reindex(c.index)), index=c.index)
        for t in TOTAL_TIERS:
            p_ = float(stats.mannwhitneyu(c.loc[ph == "El Niño", t], c.loc[ph != "El Niño", t], alternative="two-sided").pvalue)
            for label, m in [("El Niño", ph == "El Niño"), ("Neutral", ph == "Neutral"), ("La Niña", ph == "La Niña"), ("other", ph != "El Niño")]:
                v = c.loc[m, t]
                by_total.append(dict(src=name, tier=t, group=label, n=int(m.sum()), lo=int(v.min()), hi=int(v.max()), mean=float(v.mean()),
                                     med=float(v.median()), any=int((v >= 1).sum()), p=p_, first=int(c.index.min())))
    return dict(ei=ei, ee=ee, thr_e=thr_e, ci=ci, ce=ce, rows=rows, first=first, by_total=by_total, thr_t=thr_t, cti=cti)


def cold_by_enso(tmin: pd.Series, djf: pd.Series, thr: float, lo: int = SPLIT, hi: int = 2025) -> dict:
    """Cold nights per winter (ERA5 minimum at or below `thr`), the winter's coldest night and its cold snaps (runs of
    two or more cold nights) against winter Niño3.4: the correlation of each, and of the night count two degrees either
    side of the threshold, so that the answer does not hang on the threshold chosen."""
    t = tmin[tmin.index.month.isin(WET)].dropna(); sy = season_year(t.index); yrs = range(lo, hi + 1)
    count = lambda th: (t <= th).groupby(sy).sum().reindex(yrs)
    b = (t <= thr).astype(int); run = b.groupby((b != b.shift()).cumsum()).transform("size") * b
    m = dict(nights=count(thr), coldest=t.groupby(sy).min().reindex(yrs),
             snaps=((b == 1) & (b.shift(fill_value=0) == 0) & (run >= 2)).groupby(sy).sum().reindex(yrs))
    n = djf.reindex(m["nights"].index); ph = pd.Series(phase_of(n), index=n.index)
    out = dict(thr=thr, lo=lo, hi=hi, nights=m["nights"], nino=n, phase=ph,
               mean={g: float(m["nights"][ph == g].mean()) for g in ("El Niño", "Neutral", "La Niña")},
               n={g: int((ph == g).sum()) for g in ("El Niño", "Neutral", "La Niña")})
    for k, v in m.items():
        out[f"r_{k}"], out[f"p_{k}"] = (float(x) for x in stats.pearsonr(n, v))
    out["r_other"] = {thr + d: float(stats.pearsonr(n, count(thr + d))[0]) for d in (-2, 2)}
    return out


def winter_counts(spec: dict, wi: dict | None, djf: pd.Series) -> dict | None:
    """People reported affected by winter weather, one figure per winter with its ENSO phase: from the event catalogues
    where the area has them (wi), otherwise from the counts listed in the area's TOML. Winters with a report but no
    number are kept as such. The phase test is El Niño winters against the others, on the winters of one period."""
    if wi is not None:
        S = wi["S"].loc[wi["first"]:]
        rows = pd.DataFrame(dict(people=S.people, people_hi=S.people_hi, n_ev=S.n_ev, kind=""))
        test = rows.loc[:SINCE - 1]; split = SINCE
    elif spec.get("winter_counts"):
        hh = float(spec.get("hh_size", 4.8)); ev = spec.get("prewar", []) + spec.get("impacts", [])
        sy = pd.Series(season_year(pd.DatetimeIndex([e["start"] for e in ev])))
        yrs = range(int(sy.min()), 2026)
        rows = pd.DataFrame(dict(people=np.nan, people_hi=np.nan, n_ev=sy.value_counts().reindex(yrs, fill_value=0).values, kind=""), index=yrs)
        for c in spec["winter_counts"]:
            y = int(c["winter"])
            rows.loc[y, "people"] = c.get("people", c.get("households", np.nan) * hh if "households" in c else np.nan)
            rows.loc[y, "people_hi"] = c.get("people_hi", c.get("households_hi", np.nan) * hh if "households_hi" in c else np.nan)
            rows.loc[y, "kind"] = c.get("kind", "")
        test = rows; split = None
    else:
        return None
    rows["phase"] = phase_of(djf.reindex(rows.index))
    c = test.dropna(subset=["people"]); c = c.assign(phase=rows.phase.reindex(c.index))
    en, rest = c[c.phase == "El Niño"].people, c[c.phase != "El Niño"].people
    p = float(stats.mannwhitneyu(en, rest, alternative="two-sided").pvalue) if len(en) >= 2 and len(rest) >= 2 else np.nan
    return dict(rows=rows, split=split, p=p, n_test=len(c), by_phase={g: sorted(c[c.phase == g].people) for g in ("El Niño", "Neutral", "La Niña")})


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
    ends = []
    for k in ("GPCC", "gauge", "ERA5"):
        s = run.get(k)
        if s is None or s.empty:
            continue
        ax.plot(s.index, s.values, color=SRC_COL[k], lw=2, label=SRC_LABEL[k])
        ends.append([s.values[-1], s.index[-1], _short(k)])
    ends.sort(key=lambda e: e[0])                      # nudge end labels apart (at least 0.05 in r)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05)
    for yv, xv, lab in ends:
        ax.text(xv + 0.6, yv, lab, fontsize=8, color=C_TEXT, va="center")
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
    ax.set_ylabel("% above or below the 1991–2020 mean", fontsize=9, color=C_MUTED)
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
    axes[0].set_title(f"Daily rainfall over {A.ref} (IMERG) and dated reports of winter-weather impacts (▼, numbered as in the table)\n"
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
    sx.set_title(f"Skill of the {edd.MONTH_NAMES[im - 1]} issuance for each month: detrended r with ERA5 over {A.ref}, 1981–{year - 1}",
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
    ax.bar(x, b50, 0.8, bottom=b10 + b20, color=cols[50], edgecolor="white", linewidth=0.5,
           label=TIER_LABEL[50] if A.byron else "≥ 50 mm")
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
    ev["start"] = pd.to_datetime(ev["start"].fillna(ev["end"])); ev["end"] = pd.to_datetime(ev["end"]).fillna(ev["start"])
    rows = []
    for _, r in ev.iterrows():
        a, b = r.start - pd.Timedelta(days=1), r.end
        w, x = imerg.loc[a:b], era5.loc[a:b]
        rows.append(dict(im_max=float(w.pr.max()) if len(w) else np.nan, im_sum=float(w.pr.sum()) if len(w) else np.nan,
                         era5_max=float(x.pr.max()) if len(x) else np.nan,
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
    # storms of 20 mm or more in the three winters since October 2023 with no dated UN household count: drawn at the floor
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
    ax.set_xlabel(f"wettest day of the event over {A.ref}, mm (dot: IMERG; line extends to ERA5)", fontsize=9, color=C_MUTED)
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


# --------------------------------------------------------------------------- #
# Reported impact per winter against that winter's rain, before and since October 2023
SINCE = 2023                                     # first season year of the current situation (October 2023)
SIZE_TIERS = [(10, 20), (20, 50), (50, np.inf)]  # IMERG wettest day of a storm (mm), as in TIERS
UNDATED_HH = "response count|rain date not stated|mixed causes|receiving packages"
C_REGIME = {False: SRC_COL["GPCC"], True: SRC_COL["gauge"]}


HAZARDS = [("total", "Rain, October–April", "mm"), ("rx7", "Wettest 7 days", "mm"), ("rx1", "Wettest day", "mm"),
           ("n20", "Storms of 20 mm or more", ""), ("d10", "Days of 10 mm or more", ""), ("rd", "Rain days (1 mm or more)", ""),
           ("windy", "Windy days, top 5% (ERA5)", ""), ("cold8", "Cold nights, ≤ 8 °C (ERA5)", "")]


EVENT_HAZARDS = [("im_max", "Wettest day, mm (IMERG)"), ("im_sum", "Rain over the event, mm (IMERG)")]   # ERA5's area-mean wind and
#                 night minimum do not resolve the gusts and cold the reports describe, so they are not plotted per event
HAZARD_WORD = {"R": "rain", "S": "sea", "W": "wind", "C": "cold"}


def forecast_analogues(db: dict | None, code: str, share: float = 0.2) -> dict | None:
    """What a trimester's rain was (ERA5, mm) in the hindcast years whose forecast from the same issuance month ranked
    closest to this year's: a fifth of the hindcasts, so the wettest fifth when this year's forecast is among them.
    The database holds SEAS5's ensemble mean, not its members, so this is the forecast's range, not its spread."""
    row = (db or {}).get("rows", {}).get(code)
    if not row or len(row["hind"]) < 20:
        return None
    h = row["hind"].sort_values("f")
    k = int(np.ceil(share * len(h)))
    pos = int((h.f < row["cur"]).sum())                                    # hindcasts drier than this year's forecast
    top = pos >= len(h) - k + 1                                           # fewer than k hindcasts are wetter: the wettest fifth
    lo = len(h) - k if top else min(max(pos - k // 2, 0), len(h) - k)
    an = h.iloc[lo:lo + k]
    near = an.mm[(an.mm - h.mm.median()).abs() <= 0.15 * h.mm.median()]
    return dict(code=code, issued_year=db["year"], issued_month=db["month"], n=len(h), k=k, drier=pos, top=top,
                n_near=len(near), far_lo=float(an.mm.drop(near.index).min()) if len(near) < k else np.nan,
                drier_by_year={int(y): int((h.f < v).sum()) for y, v in h.f.items()},
                years=sorted(int(y) for y in an.index), lo=float(an.mm.min()), hi=float(an.mm.max()), med=float(an.mm.median()),
                an=an.mm.sort_values(), clim=h.mm.sort_index(), clim_med=float(h.mm.median()),
                clim_lo=float(h.mm.quantile(.1)), clim_hi=float(h.mm.quantile(.9)), r=row["r"])


def winter_impacts(evp: pd.DataFrame, ev: pd.DataFrame, imerg: pd.Series, st: dict, djf: pd.Series, ref: dict, cfg: dict,
                   dm: dict | None = None, era5: pd.DataFrame | None = None, db: dict | None = None) -> dict:
    """Every winter of the two event catalogues on one row (rain over the area against what was counted),
    the same storm sizes before and since October 2023, and what last winter's counts imply for a winter
    with no, one or two storms of 50 mm or more at this year's exposure. People are as reported where a
    source gives people; households are converted with one stated household size per period. A count is
    'dated' when it belongs to the event's own rain dates (not assistance delivered later)."""
    hh = {False: float(cfg["hh_size_before"]), True: float(cfg["hh_size_since"])}
    e = pd.concat([evp, ev], ignore_index=True)
    e["sy"] = e.winter.str[:4].astype(int); e["since"] = e.sy >= SINCE
    e["derived"] = e.people_affected_un.isna() & e.hh_affected_un.notna()
    e["people"] = e.people_affected_un.where(~e.derived, e.hh_affected_un * e.since.map(hh))
    scope = e.hh_affected_scope.fillna("").str.lower()
    e["dated"] = e.people.notna() & ~((e.derived & scope.str.contains(UNDATED_HH)) | scope.str.contains("carried over"))
    first = int(e.sy.min())

    # one row per winter
    wet = imerg[imerg.index.month.isin(WET)]
    d = imerg.loc["1998-08":]
    S = pd.DataFrame(dict(total=wet.groupby(season_year(wet.index)).sum(), rx7=d.rolling(7).sum().groupby(season_year(d.index)).max(),
                          n10=st["ci"][10], n20=st["ci"][20], n50=st["ci"][50])).loc[st["ci"].index]
    S["phase"] = phase_of(djf.reindex(S.index))
    g = e.groupby("sy")
    S["n_ev"] = g.size().reindex(S.index, fill_value=0)
    S["people"] = g.people.sum(min_count=1); S["derived"] = g.derived.any().reindex(S.index, fill_value=False)
    S["n_cnt"] = e[e.people.notna()].groupby("sy").size().reindex(S.index, fill_value=0)
    S["tents"] = g.tents_damaged_un.sum(min_count=1); S["tents_auth"] = g.tents_authority.sum(min_count=1)
    S["deaths"] = e[["deaths_collapse", "deaths_cold", "deaths_other"]].sum(axis=1, min_count=1).groupby(e.sy).sum(min_count=1)
    ry = int(e[e.event_id.isin(ref["byron_ids"])].sy.iloc[0])           # the reference winter (last winter)
    S["people_hi"] = np.nan; S.loc[ry, "people_hi"] = ref["snaps_all"] * hh[True]
    if dm is not None and era5 is not None:              # other measures of the hazard, for the panels
        S["rx1"], S["d10"], S["rd"] = dm["IMERG"].rx1, dm["IMERG"].d10, dm["IMERG"].d1
        S["windy"] = dm["ERA5"].windy
        t = era5.tmin[era5.index.month.isin(WET)].dropna()
        S["cold8"] = (t <= 8).groupby(season_year(t.index)).sum()
    q4 = e[e.start.dt.month.isin([10, 11, 12]) & e.dated]   # dated counts for events of October–December, against that quarter's rain
    S["people_ond"] = q4.groupby("sy").people.sum(min_count=1)
    S["people_ond_hi"] = np.nan; S.loc[ry, "people_ond_hi"] = (ref["first"] + ref["dec"]) * hh[True]
    ond = forecast_analogues(db, "OND")
    w4 = imerg[imerg.index.month.isin([10, 11, 12])]
    S["q4_imerg"] = w4.groupby(w4.index.year).sum()

    # which measure of rain tracks the counts: winters before October 2023, and single events in each period
    C = S.loc[first:SINCE - 1].dropna(subset=["people"])
    corr_w = {k: float(stats.spearmanr(C[k], C.people)[0]) for k in ("total", "rx7", "n20")}
    corr_all = {k: float(stats.spearmanr(C[k], C.people)[0]) for k, *_ in HAZARDS if k in C}
    U = e[e.dated].sort_values("start").copy(); U["lp"] = np.log10(U.people)
    U["no"] = range(1, len(U) + 1)                       # the number each count carries in the event figure and its table
    corr_ev = {k: {s_: tuple(float(v) for v in stats.spearmanr(U[U.since == s_][k], U[U.since == s_].lp)) for s_ in (False, True)} for k, _ in EVENT_HAZARDS}
    U["missed"] = (U.im_max < SIZE_TIERS[0][0]) & (U.era5_max >= SIZE_TIERS[0][0])       # rain that IMERG did not see but ERA5 did
    corr_e = {s_: dict(n=int((U.since == s_).sum()), **{k: tuple(float(v) for v in stats.spearmanr(U[U.since == s_][k], U[U.since == s_].lp))
                                                       for k in ("im_max", "im_sum")}) for s_ in (False, True)}
    wet_ev = U[U.since & (U.im_max >= SIZE_TIERS[0][0])]                  # since October 2023, with a day of 10 mm or more in IMERG
    corr_wet = dict(n=len(wet_ev), rho=float(stats.spearmanr(wet_ev.im_max, wet_ev.lp)[0]) if len(wet_ev) >= 4 else np.nan)

    # the level shift: log10(people) = a + shift * since + slope * log10(storm rain + 1), on events with a dated count
    def fit(u: pd.DataFrame) -> dict:
        X = np.column_stack([np.ones(len(u)), u.since.astype(float), np.log10(u.im_sum + 1)])
        b = np.linalg.lstsq(X, u.lp.values, rcond=None)[0]
        res = u.lp.values - X @ b; dof = len(u) - 3
        se = np.sqrt(np.diag(res @ res / dof * np.linalg.inv(X.T @ X))); t = stats.t.ppf(0.975, dof)
        return dict(n=len(u), mult=float(10 ** b[1]), lo=float(10 ** (b[1] - t * se[1])), hi=float(10 ** (b[1] + t * se[1])),
                    slope=float(b[2]), slope_se=float(se[2]))
    top = {s_: U[U.since == s_].sort_values("im_sum").iloc[-1] for s_ in (False, True)}      # the largest storm of each period
    shift = dict(all=fit(U), no_top=fit(U[~U.event_id.isin([top[False].event_id, top[True].event_id])]),
                 geo=float(10 ** (U[U.since].lp.mean() - U[~U.since].lp.mean())),
                 med=float(U[U.since].people.median() / U[~U.since].people.median()),
                 med_before=float(U[~U.since].people.median()), med_since=float(U[U.since].people.median()),
                 top={k: dict(people=float(v.people), rain=float(v.im_sum), start=v.start, end=v.end) for k, v in top.items()})

    # storm for storm: storms of each size in each period, how many left a report, and what was counted
    sto = st["ei"]; sto = sto[(sto.sy >= first) & (sto.sy <= S.index.max())]
    hit = lambda r: ((e.start - pd.Timedelta(days=1) <= r.end) & (e.end >= r.start)).any()
    counts = lambda u: sorted(zip(u.people, u.derived))  # (people, converted from households?)
    tiers = []                                           # a count is sized by the wettest day in its own event window
    for lo, hi_ in SIZE_TIERS:
        for s_ in (False, True):
            x = sto[(sto["max"] >= lo) & (sto["max"] < hi_) & ((sto.sy >= SINCE) == s_)]
            tiers.append(dict(lo=lo, hi=hi_, since=s_, n=len(x), reported=int(sum(hit(r) for r in x.itertuples())),
                              people=counts(U[(U.im_max >= lo) & (U.im_max < hi_) & (U.since == s_)])))
    dry_ev = U.im_max < SIZE_TIERS[0][0]                 # under 10 mm in IMERG: little rain only if ERA5 agrees
    low = {s_: counts(U[dry_ev & (U.era5_max < SIZE_TIERS[0][0]) & (U.since == s_)]) for s_ in (False, True)}
    per = pd.DataFrame({lo: st["ci"][lo] - (st["ci"][int(hi_)] if np.isfinite(hi_) else 0) for lo, hi_ in SIZE_TIERS})   # storms per winter, by size
    en_w = per[S.phase.reindex(per.index) == "El Niño"]
    expect = {lo: dict(lo=int(en_w[lo].min()), hi=int(en_w[lo].max()), med=float(en_w[lo].median()), n=len(en_w),
                       all_lo=int(per[lo].min()), all_hi=int(per[lo].max()), n_all=len(per), first=int(per.index.min())) for lo, _ in SIZE_TIERS}
    missed = [dict(start=r.start, end=r.end, people=float(r.people), derived=bool(r.derived), im=float(r.im_max), era5=float(r.era5_max))
              for r in U[dry_ev & (U.era5_max >= SIZE_TIERS[0][0])].itertuples()]

    # this winter's rain: the range of El Niño winters in the same record
    en = S[S.phase == "El Niño"]
    rain = dict(en_n=len(en), en_min=float(en.total.min()), en_max=float(en.total.max()), en_med=float(en.total.median()),
                en_q25=float(en.total.quantile(.25)), en_q75=float(en.total.quantile(.75)), n=len(S), med=float(S.total.median()),
                driest=int(S.total.idxmin()), wettest=int(S.total.idxmax()))
    big = lambda x, k: int((x.n50 == k).sum()) if k < 2 else int((x.n50 >= 2).sum())
    freq = {k: dict(all=big(S, k), en=big(en, k), years=[int(y) for y in S.index[S.n50 >= 2]] if k == 2 else []) for k in (0, 1, 2)}

    # scenarios: last winter's alert-based counts without, with and with twice its largest storm, at this year's exposure
    byron, byron_hi = float(e[e.event_id.isin(ref["byron_ids"])].people.sum()), float(ref["byron_hh"] * hh[True])
    lo_all, hi_all = float(e[e.sy == ry].people.sum()), float(ref["snaps_all"] * hh[True])
    scale = float(cfg["people_in_sites_now"]) / float(cfg["people_in_sites_then"])
    scen = [dict(key=k, lo=lo_, hi=hi_, lo_s=lo_ * scale, hi_s=hi_ * scale, **freq[n_])
            for k, n_, lo_, hi_ in [("none", 0, lo_all - byron, hi_all - byron_hi), ("one", 1, lo_all, hi_all),
                                    ("two", 2, lo_all + byron, hi_all + byron_hi)]]
    return dict(S=S, e=e, U=U, first=first, ry=ry, hh=hh, corr_w=corr_w, corr_e=corr_e, corr_wet=corr_wet, shift=shift, tiers=tiers,
                low=low, missed=missed, rain=rain, scen=scen, scale=scale, byron=byron, byron_hi=byron_hi, n_all=len(S), ond=ond,
                corr_all=corr_all, n_before=len(C), corr_ev=corr_ev, expect=expect, survey=float(cfg["survey_flooded_share"]) * float(cfg["people_in_sites_now"]),
                sites_now=float(cfg["people_in_sites_now"]), sites_then=float(cfg["people_in_sites_then"]))


def fig_winter_impacts(wi: dict, out: Path) -> None:
    """Left: people reported affected per winter against October–April rain, log scale, before and since
    October 2023. Right: the winters since October 2023 on a linear scale with the three scenarios. The
    shaded band on both is the range of rain in the record's El Niño winters."""
    S, rain, ry = wi["S"], wi["rain"], wi["ry"]
    C = S.loc[wi["first"]:].dropna(subset=["people"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=150, gridspec_kw=dict(width_ratios=[1, 1], wspace=0.24))
    for x in (ax, bx):
        x.axvspan(rain["en_min"], rain["en_max"], color="#BFD9EE", alpha=0.45, lw=0, zorder=0)
        x.set_xlim(80, 520); x.set_xlabel(f"October–April rain over {A.ref}, mm (IMERG)", fontsize=9, color=C_MUTED)
    # left: every counted winter
    off = {2009: (-8, 6), 2012: (-8, -12), 2013: (-8, 7), 2014: (-9, -13), 2015: (8, -3), 2019: (7, 5), 2024: (8, -3), ry: (-10, -4)}
    ax.text((rain["en_min"] + rain["en_max"]) / 2, 4500, f"rain in the {_nword(rain['en_n'])} El Niño\nwinters since {_yr(int(S.index.min()))}:\n"
            f"{rain['en_min']:.0f}–{rain['en_max']:.0f} mm", ha="center", va="center", fontsize=7.5, color=SRC_COL["GPCC"])
    for y, r in C.iterrows():
        c = C_REGIME[y >= SINCE]
        if pd.notna(r.people_hi):
            ax.plot([r.total] * 2, [r.people, r.people_hi], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r.total, r.people, "o", ms=8, color=c)
        dx, dy = off.get(y, (7, 4))
        ax.annotate(_yr(y), (r.total, r.people), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=C_TEXT,
                    ha="right" if dx < 0 else "left")
    ax.plot(S.total, [13] * len(S), "|", ms=7, mew=1.1, color=C_MUTED, alpha=0.7)
    ax.set_yscale("log"); ax.set_ylim(10, 4e6)
    ax.set_yticks([10, 100, 1000, 10000, 100000, 1000000], ["10", "100", "1,000", "10,000", "100,000", "1 million"])
    ax.set_ylabel("people reported affected in the winter (log scale)", fontsize=9, color=C_MUTED)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=C_REGIME[False], ls="", ms=7, label="before October 2023"),
                       plt.Line2D([], [], marker="o", color=C_REGIME[True], ls="", ms=7, label="since October 2023"),
                       plt.Line2D([], [], marker="|", color=C_MUTED, ls="", ms=7, mew=1.1, label=f"rain in each of the {len(S)} winters")],
              frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Counted winters, before and since October 2023", fontsize=10, color=C_TEXT, loc="left")
    # right: since October 2023, with the scenarios for this winter
    lab = {"none": "no storm ≥ 50 mm", "one": "one, as last winter", "two": "two such storms"}
    often = lambda sc: f"\n{sc['all']} of {wi['n_all']} winters"
    for sc in wi["scen"]:
        bx.add_patch(plt.Rectangle((rain["en_min"], sc["lo_s"] / 1e6), rain["en_max"] - rain["en_min"], (sc["hi_s"] - sc["lo_s"]) / 1e6,
                                   fc=C_REGIME[True], ec=C_REGIME[True], alpha=0.35, lw=1.0, zorder=2))
        bx.text(rain["en_min"] - 6, (sc["lo_s"] + sc["hi_s"]) / 2e6, lab[sc["key"]] + often(sc), ha="right", va="center", fontsize=8,
                color=C_TEXT, linespacing=1.15)
    for y, r in C.loc[SINCE:].iterrows():
        if pd.notna(r.people_hi):
            bx.plot([r.total] * 2, [r.people / 1e6, r.people_hi / 1e6], color=C_REGIME[True], lw=2.2, solid_capstyle="round", zorder=3)
        bx.plot(r.total, r.people / 1e6, "o", ms=8, color=C_REGIME[True], zorder=3)
        bx.annotate(_yr(y) + (" as counted" if y == ry else ""), (r.total, r.people / 1e6), xytext=(8, 5) if y != ry else (0, -15),
                    textcoords="offset points", fontsize=8, color=C_TEXT, ha="left" if y != ry else "center")
    for v, t, ls in [(wi["survey"], f"if {100 * wi['survey'] / wi['sites_now']:.0f}% are flooded at least once, as surveyed last winter (each person once)", (0, (4, 3))),
                     (wi["sites_now"], "people in displacement sites, September 2026", (0, (1, 2)))]:
        bx.axhline(v / 1e6, color=C_MUTED, lw=1.0, ls=ls, zorder=1)
        bx.text(86, v / 1e6 + 0.02, t, fontsize=7.5, color=C_MUTED, va="bottom")
    bx.set_ylim(0, 1.9); bx.set_yticks([0, 0.5, 1.0, 1.5], ["0", "0.5", "1.0", "1.5"]); bx.set_ylabel("people reported affected in the winter, millions", fontsize=9, color=C_MUTED)
    bx.text((rain["en_min"] + rain["en_max"]) / 2, 0.02, "El Niño winters", ha="center", va="bottom", fontsize=7.5, color=SRC_COL["GPCC"])
    bx.set_title("Planning scenarios for 2026/27, not a forecast", fontsize=10, color=C_TEXT, loc="left")
    for x in (ax, bx):
        x.grid(color="#eceff0", lw=0.8); x.set_axisbelow(True); edd._style_ax(x)
    _save(fig, out)


def fig_winter_hazards(wi: dict, out: Path) -> None:
    """People reported affected per winter (log scale) against eight measures of that winter's weather, one panel each,
    before and since October 2023. The shaded band in each panel is that measure's range over the record's El Niño winters."""
    S, ry = wi["S"], wi["ry"]
    C = S.loc[wi["first"]:].dropna(subset=["people"])
    en = S[S.phase == "El Niño"]
    cols = [h for h in HAZARDS if h[0] in S]
    fig, axs = plt.subplots(2, 4, figsize=(11.6, 6.0), dpi=150, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.42))
    for ax, (k, title, unit) in zip(axs.ravel(), cols):
        v = S[k].dropna(); pad = 0.08 * (v.max() - v.min())
        ax.axvspan(en[k].min(), en[k].max(), color="#BFD9EE", alpha=0.45, lw=0, zorder=0)
        ax.plot(v, [13] * len(v), "|", ms=6, mew=1.0, color=C_MUTED, alpha=0.6)
        for y, r in C.iterrows():
            c = C_REGIME[y >= SINCE]
            if pd.notna(r.people_hi):
                ax.plot([r[k]] * 2, [r.people, r.people_hi], color=c, lw=2.0, solid_capstyle="round")
            ax.plot(r[k], r.people, "o", ms=6.5, color=c)
            if y >= SINCE or r.people == C.loc[:SINCE - 1].people.max():
                right = r[k] > v.min() + 0.72 * (v.max() - v.min())
                ax.annotate(_yr(y), (r[k], r.people), xytext=(-6 if right else 6, 4 if y >= SINCE else -11), textcoords="offset points",
                            fontsize=7, color=C_TEXT, ha="right" if right else "left")
        ax.set_xlim(v.min() - pad, v.max() + pad)
        ax.set_yscale("log"); ax.set_ylim(10, 2e6)
        ax.set_title(title + (f", {unit}" if unit else ""), fontsize=8.5, color=C_TEXT, loc="left")
        ax.set_title(f"ρ {wi['corr_all'][k]:+.2f}", fontsize=7.5, color=C_REGIME[False], loc="right")
        if not unit:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
        ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True); edd._style_ax(ax); ax.tick_params(labelsize=7.5)
    for ax in axs[:, 0]:
        ax.set_yticks([10, 100, 1000, 10000, 100000, 1000000], ["10", "100", "1,000", "10,000", "100,000", "1 million"])
        ax.set_ylabel("people reported affected", fontsize=8.5, color=C_MUTED)
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=C_REGIME[False], ls="", ms=6, label="before October 2023"),
                        plt.Line2D([], [], marker="o", color=C_REGIME[True], ls="", ms=6, label="since October 2023"),
                        plt.Rectangle((0, 0), 1, 1, fc="#BFD9EE", alpha=0.6, label="range in El Niño winters"),
                        plt.Line2D([], [], marker="|", color=C_MUTED, ls="", ms=7, mew=1.0, label=f"each of the {len(S)} winters"),
                        plt.Line2D([], [], ls="", marker="", label=f"ρ: rank correlation, {_nword(wi['n_before'])} winters before October 2023")],
               frameon=False, fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.985), handletextpad=0.4, columnspacing=1.4)
    _save(fig, out)


def fig_event_hazards(wi: dict, st: dict, out: Path) -> None:
    """People counted per event (log scale) against the rain over the event's own dates, before and since October 2023.
    Above the dots: how many storms with at least that much rain a winter has brought, El Niño winters against the others."""
    U = wi["U"]; k = "im_sum"
    bt = {(r["tier"], r["group"]): r for r in st["by_total"] if r["src"] == "IMERG"}
    rg = lambda r: f"{r['lo']}–{r['hi']}" if r["lo"] != r["hi"] else f"{r['lo']}"
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
    v = U[k]; pad = 0.05 * (v.max() - v.min())
    x0, x1 = v.min() - pad, v.max() + pad
    ax.set_xlim(x0, x1); ax.set_yscale("log"); ax.set_ylim(10, 6e7)
    edges = [x0] + [float(t) for t in TOTAL_TIERS] + [x1]            # bands of total rain: alternate shading, a firm line between them
    for n_, (a_, b_) in enumerate(zip(edges, edges[1:])):
        if n_ % 2:
            ax.fill_between([a_, b_], 10, 6e5, color="#e3e8ea", alpha=0.75, lw=0, zorder=0)
    ax.axhspan(6e5, 6e7, color="#BFD9EE", alpha=0.45, lw=0, zorder=0); ax.axhline(6e5, color="#6f7b7c", lw=0.8, zorder=1)
    en0 = bt[(TOTAL_TIERS[0], "El Niño")]; ot0 = bt[(TOTAL_TIERS[0], "other")]
    ax.text(x0 + 0.012 * (x1 - x0), 3.6e7, f"Storms per winter with at least this much rain: fewest–most in the {en0['n']} El Niño winters since "
            f"{_yr(en0['first'])}, and in the other {ot0['n']}", ha="left", va="center", fontsize=7.5, color=SRC_COL["GPCC"])
    ax.text(x0 + 0.012 * (x1 - x0), 2.0e7, "every rain spell of that size, damaging or not; a past range, not a forecast", ha="left", va="center",
            fontsize=6.8, color=C_MUTED)
    for t, ytxt in zip(TOTAL_TIERS, (8.5e6, 3.4e6, 1.35e6)):             # each row starts at its own threshold
        ax.vlines(t, 10, ytxt * 1.55, color="#6f7b7c", lw=1.3, zorder=1)
        e_, o_ = bt[(t, "El Niño")], bt[(t, "other")]
        ax.text(t + 0.012 * (x1 - x0), ytxt, f"{t} mm or more:", ha="left", va="center", fontsize=7.5, color=C_MUTED)
        ax.text(t + 0.150 * (x1 - x0), ytxt, f"El Niño {rg(e_)}", ha="left", va="center", fontsize=9, fontweight="bold", color=SRC_COL["GPCC"])
        ax.text(t + 0.287 * (x1 - x0), ytxt, f"other winters {rg(o_)}", ha="left", va="center", fontsize=7.5, color=C_MUTED)
    for r in U.itertuples():
        ax.plot(getattr(r, k), r.people, "o", ms=8, color=C_REGIME[r.since], mfc="white" if r.missed else C_REGIME[r.since], mew=1.6, zorder=3)
    ax.set_yticks([10, 100, 1000, 10000, 100000], ["10", "100", "1,000", "10,000", "100,000"])
    ax.set_xlabel(f"rain over {A.ref} during the event, mm (IMERG)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("people counted in the event (log scale)", fontsize=9, color=C_MUTED)
    n = {s_: int((U.since == s_).sum()) for s_ in (False, True)}
    hs = [plt.Line2D([], [], marker="o", color=C_REGIME[False], ls="", ms=7, label=f"before October 2023 ({n[False]} counts)"),
          plt.Line2D([], [], marker="o", color=C_REGIME[True], ls="", ms=7, label=f"since October 2023 ({n[True]} counts)")]
    if U.missed.any():
        hs.append(plt.Line2D([], [], marker="o", color=C_REGIME[True], mfc="white", mew=1.5, ls="", ms=7, label="hollow: rain IMERG missed"))
    ax.legend(handles=hs, frameon=False, fontsize=8, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.09), handletextpad=0.4, columnspacing=1.6)
    ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True); edd._style_ax(ax)
    import textwrap
    fig.text(0.125, 0.0, textwrap.fill(A.event_sources, 125), fontsize=7.5, color=C_MUTED, ha="left", va="top", linespacing=1.3)
    _save(fig, out)


PHASE_COL = {"El Niño": C_EN, "Neutral": C_NEU, "La Niña": C_LN}


def fig_winter_counts(bars: dict, out: Path) -> None:
    """People reported affected by winter weather, one bar per winter (log scale), coloured by the winter's ENSO phase.
    A hollow dot marks a winter with a report but no number."""
    R = bars["rows"]; x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=150)
    floor = 10
    top = np.nanmax(R[["people", "people_hi"]].values)
    for i, (y, r) in enumerate(R.iterrows()):
        c = PHASE_COL[r.phase]
        if pd.notna(r.people):
            ax.bar(i, r.people - floor, bottom=floor, color=c, width=0.72, edgecolor="white", linewidth=0.4, zorder=2)
            if pd.notna(r.people_hi):
                ax.plot([i, i], [r.people, r.people_hi], color=c, lw=2.0, solid_capstyle="round", zorder=3)
            if r.kind:
                ax.text(i, max(r.people, r.people_hi if pd.notna(r.people_hi) else 0) * 1.25, "\n".join(textwrap.wrap(r.kind, 18)), ha="center", va="bottom",
                        fontsize=6.8, color=C_TEXT, linespacing=1.1)
        elif r.n_ev:
            ax.plot(i, floor * 1.6, "o", ms=6, mfc="white", mew=1.4, color=c, zorder=3)
    if bars["split"] is not None and bars["split"] in R.index:
        k = list(R.index).index(bars["split"]) - 0.5
        ax.axvline(k, color=C_TEXT, lw=1, ls=(0, (4, 3)))
        ax.text(k + 0.12, top * 6, "since October 2023", fontsize=7.5, color=C_TEXT, va="top", ha="left")
    ax.set_yscale("log"); ax.set_ylim(floor, top * 8)
    ticks = [v for v in (10, 100, 1000, 10000, 100000, 1000000) if v <= top * 8]
    ax.set_yticks(ticks, [f"{v:,}" if v < 1e6 else "1 million" for v in ticks])
    ax.set_xticks(x, [_yr(int(y)) for y in R.index], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.7, len(R) - 0.3)
    ax.set_ylabel("people reported affected (log scale)", fontsize=9, color=C_MUTED)
    ax.legend(handles=[edd.Patch(color=C_EN, label="El Niño winter"), edd.Patch(color=C_NEU, label="neutral"), edd.Patch(color=C_LN, label="La Niña"),
                       plt.Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.4, ls="", ms=6, label="impact reported, no number")],
              frameon=False, fontsize=8, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.12), handletextpad=0.5, columnspacing=1.4)
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True); edd._style_ax(ax)
    _save(fig, out)


def fig_cold_enso(parts: list[dict], out: Path) -> None:
    """Cold nights per winter against winter Niño3.4, one panel per area: each dot a winter, coloured by its phase,
    with the mean of each phase as a short line."""
    fig, axs = plt.subplots(1, len(parts), figsize=(10.4, 4.3), dpi=150, gridspec_kw=dict(wspace=0.2))
    for ax, p in zip(np.atleast_1d(axs), parts):
        c = p["a"]["cold_enso"]; n, v, ph = c["nino"], c["nights"], c["phase"]
        for g, col in PHASE_COL.items():
            m = ph == g
            ax.plot(n[m], v[m], "o", ms=6.5, color=col, alpha=0.9, zorder=3)
            ax.plot([n[m].min(), n[m].max()], [c["mean"][g]] * 2, color=col, lw=2.0, alpha=0.9, zorder=2)
        for t in (-ENSO_THRESH, ENSO_THRESH):
            ax.axvline(t, color="#b8bfbf", lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_title(f"{p['area'].name}: nights of {c['thr']:g} °C or colder", fontsize=9.5, color=C_TEXT, loc="left")
        ax.set_title(f"r = {c['r_nights']:+.2f}, p = {c['p_nights']:.2f}", fontsize=8.5, color=C_MUTED, loc="right")
        ax.set_xlabel("December–February Niño3.4, °C", fontsize=9, color=C_MUTED)
        ax.set_ylabel(f"cold nights per winter (ERA5, {_yr(c['lo'])}–{_yr(c['hi'])})", fontsize=9, color=C_MUTED)
        ax.set_ylim(bottom=-0.6)
        ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True); edd._style_ax(ax)
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=C_LN, ls="", ms=6.5, label="La Niña winter"),
                        plt.Line2D([], [], marker="o", color=C_NEU, ls="", ms=6.5, label="neutral"),
                        plt.Line2D([], [], marker="o", color=C_EN, ls="", ms=6.5, label="El Niño winter"),
                        plt.Line2D([], [], color=C_MUTED, lw=2, label="mean of each phase")],
               frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    _save(fig, out)


def fig_forecast_quarter(wi: dict, out: Path) -> None:
    """People reported affected in October–December (log scale) against that quarter's ERA5 rain, with the range of rain
    in the hindcast years whose October forecast was as wet as this year's."""
    S, q, ry = wi["S"], wi["ond"], wi["ry"]
    x = q["clim"]                                                    # ERA5 quarter totals by year (team raster stats)
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax.axvspan(q["lo"], q["hi"], color="#BFD9EE", alpha=0.5, lw=0, zorder=0)
    ax.axvline(q["clim_med"], color=C_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(q["clim_med"] - 2, 1.3e6, f"normal\n{q['clim_med']:.0f} mm", ha="right", va="top", fontsize=7.5, color=C_MUTED)
    ax.text((q["lo"] + q["hi"]) / 2, 1.3e6, f"rain in the {_nword(q['k'])} years with an October forecast\nin the wettest fifth, like {q['issued_year']}'s: "
            f"{q['lo']:.0f}–{q['hi']:.0f} mm", ha="center", va="top", fontsize=7.5, color=SRC_COL["GPCC"])
    ax.plot(x, [13] * len(x), "|", ms=7, mew=1.1, color=C_MUTED, alpha=0.7)
    ax.plot(q["an"], [13] * len(q["an"]), "|", ms=9, mew=1.6, color=SRC_COL["GPCC"])
    C, late = S.loc[wi["first"]:], []
    for y, r in C.iterrows():
        if y not in x.index:
            continue
        c = C_REGIME[y >= SINCE]
        if pd.notna(r.people_ond):
            if pd.notna(r.people_ond_hi):
                ax.plot([x[y]] * 2, [r.people_ond, r.people_ond_hi], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(x[y], r.people_ond, "o", ms=8, color=c)
            ax.annotate(_yr(y), (x[y], r.people_ond), xytext=(8, -3), textcoords="offset points", fontsize=8, color=C_TEXT)
        elif pd.notna(r.people):                                    # counted, but only for storms after December
            late.append(y)
    for i, y in enumerate(sorted(late, key=lambda y: x[y])):
        ax.plot(x[y], 30, "o", ms=7, mfc="white", mew=1.4, color=C_REGIME[y >= SINCE])
        ax.annotate(_yr(y), (x[y], 30), xytext=(0, 8 + 9 * (i % 3)), textcoords="offset points", fontsize=7, color=C_MUTED, ha="center")
    ax.set_yscale("log"); ax.set_ylim(10, 2e6); ax.set_xlim(max(0, x.min() - 12), max(x.max(), q["hi"]) + 14)
    ax.set_yticks([10, 100, 1000, 10000, 100000, 1000000], ["10", "100", "1,000", "10,000", "100,000", "1 million"])
    ax.set_xlabel(f"October–December rain over {A.ref}, mm (ERA5)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("people reported affected, October–December (log scale)", fontsize=9, color=C_MUTED)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=C_REGIME[False], ls="", ms=7, label="before October 2023"),
                       plt.Line2D([], [], marker="o", color=C_REGIME[True], ls="", ms=7, label="since October 2023"),
                       plt.Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.4, ls="", ms=7, label="counted only for storms after December"),
                       plt.Line2D([], [], marker="|", color=C_MUTED, ls="", ms=7, mew=1.1, label=f"each October–December, {int(x.index.min())}–{int(x.index.max())}")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title(f"{A.name}: people affected in October–December against that quarter's rain (ERA5)", fontsize=10, color=C_TEXT, loc="left")
    ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True); edd._style_ax(ax)
    _save(fig, out)


def fig_wet_composite(c: edd.Country, comp: np.ndarray, hit: np.ndarray, ok: np.ndarray, n_en: int,
                      y0: int, y1: int, out: Path, label: str | None = None) -> None:
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
                 f"{label or A.name} outlined; grey = too dry to analyse", fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.995, va="top")
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
        rain = ("—" if np.isnan(e["im_max"]) else f'{e["im_max"]:.0f} mm'
                + ("" if np.isnan(e["im_max_n"]) else f'<br><span class="small">{e["im_max_n"]:.0f} · {e["im_max_s"]:.0f}</span>'))
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


def render_range(spec: dict, a: dict, heading: bool = True) -> str:
    """Section 5: how many storms a winter brings (IMERG), when the first damaging one comes, what one
    storm of each kind did last winter, last winter as two reference points and, where both event
    catalogues exist, the winter-by-winter record with scenarios scaled from last winter (no forecast totals)."""
    st, ref, ev = a["st"], a.get("ref"), a.get("ev")
    T = spec.get("titles", {})
    row = {(r["src"], r["tier"], r["group"]): r for r in st["rows"]}
    fs = {(r["src"], r["group"]): r for r in st["first"]}
    o = ([f'<h2>{html.escape(T.get("range", "5. What this winter could bring"))}</h2>'] if heading else []) + [spec.get("range_intro_html", "")]
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
             f'merged across lulls of up to two dry days, sized by its wettest day over {A.ref} (IMERG, area-weighted). ' + ('Byron, 8–17 December 2025, '
             'counts as one storm (rain from 8 December, the storm proper from 10 December). ERA5 is not used for the largest storms: its 25 km '
             'cells smooth them so much that it gives Byron a wettest day of 18 mm, which would not place it among the record\'s 14 largest '
             'storms.' if A.byron else '') + '</figcaption></figure>')
    if st.get("by_total"):
        o.append(render_storm_totals(st))
    # first storm
    fe, fa = fs[("IMERG", "El Niño")], fs[("IMERG", "all")]
    e_ = st["ei"]; f_ = e_[e_["max"] >= 20].groupby("sy").start.min().reindex(range(1998, 2026))
    ph_ = pd.Series(phase_of(a["djf"].reindex(f_.index)), index=f_.index)
    early = pd.Series({y: pd.notna(d) and d < pd.Timestamp(y, 12, 1) for y, d in f_.items()})
    en_ = ph_ == "El Niño"
    p_fish = stats.fisher_exact([[int(early[en_].sum()), int((~early[en_]).sum())],
                                 [int(early[~en_].sum()), int((~early[~en_]).sum())]])[1]
    none_ = [f"{y}/{str(y + 1)[2:]} ({ph_[y]})" for y in f_.index if pd.isna(f_[y])]
    o.append('<h3>When the first storm of 20 mm or more comes</h3>')
    o.append(spec.get("first_storm_html", ""))
    o.append(f'<figure><img src="first_storm.png" alt="Date of the first storm of 20 mm or more, by ENSO phase"><figcaption>Each dot is one winter: '
             f'the start date of its first storm with a wettest day of 20 mm or more over {A.ref} (IMERG). Before 1 December in '
             f'{round(fe["p_before_dec"] * fe["n"])} of {fe["n"]} El Niño winters and {round(fa["p_before_dec"] * fa["n"])} of {fa["n"]} winters overall. '
             + (f'Eight El Niño winters is a small sample: against the other winters the difference is within chance (Fisher exact test, p = {p_fish:.2f}).'
                if p_fish >= 0.05 else
                f'Against the other winters the difference is unlikely to be chance alone (Fisher exact test, p = {p_fish:.2f}), though eight El Niño winters is a small sample.')
             + (f' {", ".join(none_)} had no storm of 20 mm or more and {"has" if len(none_) == 1 else "have"} no dot.' if none_ else '')
             + '</figcaption></figure>')
    # how big a week of rain is
    wk = a.get("week")
    if wk:
        im, ga = wk["IMERG"], wk["gauge"]
        pc = lambda v, m: f"{100 * v / m:.0f}%"
        o.append('<h3>How big a week of rain is</h3>')
        o.append(f'<p>A 7-day rainfall forecast can be read against the year: in a typical winter the wettest week over {A.ref} brings about '
                 f'{pc(im["med"], im["mean"])} of an average year\'s rain (IMERG), and one winter in five brings a week of {pc(im["q80"], im["mean"])} or more'
                 + (f' ({im["share30"]} of the {im["n"]} winters since 1998/99 had a week of 30% or more)' if round(100 * im["q80"] / im["mean"]) != 30 else '')
                 + '. The table is a scale for reading a forecast, '
                 'not a damage predictor.<!--more-->' + spec.get("week_html", "") + '</p>')
        o.append(f'<table><thead><tr><th>Wettest 7 days of a winter</th><th class="num">IMERG over {html.escape(A.ref)}</th>'
                 f'<th class="num">Share of the year</th><th class="num">{html.escape(A.gauge_short)} gauge</th><th class="num">Share of the year</th></tr></thead><tbody>')
        o.append(f'<tr><td>Average year (for scale)</td><td class="num">{im["mean"]:.0f} mm</td><td class="num">100%</td>'
                 f'<td class="num">{ga["mean"]:.0f} mm</td><td class="num">100%</td></tr>')
        o.append(f'<tr><td>Typical winter (median)</td><td class="num">{im["med"]:.0f} mm</td><td class="num">{pc(im["med"], im["mean"])}</td>'
                 f'<td class="num">{ga["med"]:.0f} mm</td><td class="num">{pc(ga["med"], ga["mean"])}</td></tr>')
        o.append(f'<tr><td>One winter in five</td><td class="num">{im["q80"]:.0f} mm</td><td class="num">{pc(im["q80"], im["mean"])}</td>'
                 f'<td class="num">{ga["q80"]:.0f} mm</td><td class="num">{pc(ga["q80"], ga["mean"])}</td></tr>')
        o.append(f'<tr><td>Wettest in the record</td><td class="num">{im["mx"]:.0f} mm<br><span class="small">{_yr(im["mx_year"])}</span></td>'
                 f'<td class="num">{pc(im["mx"], im["mean"])}</td><td class="num">{ga["mx"]:.0f} mm<br><span class="small">{_yr(ga["mx_year"])}</span></td>'
                 f'<td class="num">{pc(ga["mx"], ga["mean"])}</td></tr>')
        if wk["byron"]:
            e_ = wk["byron"]["end"]
            o.append(f'<tr class="hl"><td>Storm Byron, 7 days to {e_:%-d} December 2025</td><td class="num">{wk["byron"]["mm"]:.0f} mm</td>'
                     f'<td class="num">{pc(wk["byron"]["mm"], im["mean"])}</td><td class="num">—</td><td class="num">—</td></tr>')
        o.append('</tbody></table>')
        o.append(f'<p class="small">Highest 7-day running total in each season (August–July), divided by the same source\'s mean annual total: IMERG '
                 f'{_yr(im["first"])}–{_yr(im["last"])} (area-weighted over {html.escape(A.ref)}), {html.escape(A.gauge_name)} {_yr(ga["first"])}–{_yr(ga["last"])} '
                 '(seasons with at least 330 days reported). These are observed scales: a forecast system\'s 7-day totals carry its own biases, so a '
                 'forecast compares best as a share of that system\'s own climatological annual total. Totals for a week ahead are much less certain '
                 'than for the next two or three days, and a 7-day total says nothing about how much fell in an hour, which is what drives flash '
                 'floods.</p>')

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
    if a.get("wi"):
        o.append(render_winter_impacts(spec, a))
    return "\n".join(o)


def render_winter_counts(spec: dict, bars: dict) -> str:
    """Section 4, last part for an area: the counts by winter coloured by ENSO phase, and whether the phases differ."""
    bp = bars["by_phase"]; lst = lambda v: _join(_sig(x) if x >= 1000 else f"{x:,.0f}" for x in v)
    said = [f'the {_nword(len(bp[g]))} counted {g if g != "Neutral" else "neutral"} winter{"s" if len(bp[g]) != 1 else ""} ({lst(bp[g])} people)'
            for g in ("El Niño", "Neutral", "La Niña") if bp[g]]
    none = [g for g in ("El Niño", "Neutral", "La Niña") if not bp[g]]
    o = ['<h3>Counts by winter and ENSO phase</h3>',
         f'<figure><img src="winter_counts.png" alt="People reported affected by winter weather in {html.escape(A.ref)}, by winter and ENSO phase">'
         f'<figcaption>{spec.get("winter_counts_caption", "")}</figcaption></figure>',
         f'<p>The counts do not separate by ENSO phase. ' + (("Before October 2023, " + _join(said)) if bars["split"] else (_join(said)[0].upper() + _join(said)[1:]))
         + (' cannot be told apart' + (f' ({_pv(bars["p"])} for El Niño against the rest)' if pd.notna(bars["p"]) else '')
            if len(said) > 1 else ' are all there is')
         + (f', and no {_join(none)} winter has a count' if none else '') + '. ' + spec.get("winter_counts_html", "") + '</p>']
    return "\n".join(o)


def _pv(p: float) -> str:
    return "p &lt; 0.01" if p < 0.01 else f"p = {p:.2f}"


def render_storm_totals(st: dict) -> str:
    """How many storms of a given total rain a winter has brought, El Niño winters against neutral and La Niña ones:
    a past range for this winter, with the test of whether the phases differ. IMERG first; ERA5's longer record,
    at thresholds matched to IMERG's, as the check."""
    bt = {(r["src"], r["tier"], r["group"]): r for r in st["by_total"]}
    rg = lambda r: f'{r["lo"]}–{r["hi"]}' if r["lo"] != r["hi"] else f'{r["lo"]}'
    t0, t1, t2 = TOTAL_TIERS; thr = st["thr_t"]
    en, ne, ln, ot = (bt[("IMERG", t0, g)] for g in ("El Niño", "Neutral", "La Niña", "other"))
    e_en, e_ot = bt[("ERA5", t0, "El Niño")], bt[("ERA5", t0, "other")]
    chance = lambda p: "is within chance" if p >= 0.05 else "is unlikely to be chance"
    big_en, big_ot = bt[("IMERG", t2, "El Niño")], bt[("IMERG", t2, "other")]
    mid = bt[("IMERG", t1, "El Niño")]
    o = [f'<p>Counted by a storm\'s total rain, El Niño winters have brought {rg(en)} storms of {t0} mm or more ({en["mean"]:.1f} on average), against '
         f'{rg(ne)} in neutral winters ({ne["mean"]:.1f}) and {rg(ln)} in La Niña winters ({ln["mean"]:.1f}). On IMERG\'s {_nword(en["n"])} El Niño '
         f'winters the difference from the other {ot["n"]} {chance(en["p"])} ({_pv(en["p"])}); in ERA5 since {e_en["first"]}, with {e_en["n"]} El Niño '
         f'winters and thresholds matched to IMERG\'s, the gap is {e_en["mean"] - e_ot["mean"]:.1f} storms a winter and {chance(e_en["p"])} ({_pv(e_en["p"])}). '
         f'The bigger storms do not follow the phase: {t1} mm or more came {rg(mid)} times in an El Niño winter and {rg(bt[("IMERG", t1, "other")])} in the '
         f'others, and a storm of {t2} mm or more came in {big_en["any"]} of {big_en["n"]} El Niño winters and {big_ot["any"]} of {big_ot["n"]} others. '
         'These are past ranges, not a forecast.</p>']
    o.append('<table><thead><tr><th>Storms per winter with a total of</th>'
             + "".join(f'<th class="num">{g}<br><span class="small">{bt[("IMERG", t0, g)]["n"]} winters</span></th>' for g in ("El Niño", "Neutral", "La Niña"))
             + '<th class="num">El Niño against the rest<br><span class="small">IMERG · ERA5</span></th></tr></thead><tbody>')
    for t in TOTAL_TIERS:
        o.append(f'<tr><td>{t} mm or more</td>' + "".join(
            f'<td class="num">{rg(bt[("IMERG", t, g)])} <span class="small">(mean {bt[("IMERG", t, g)]["mean"]:.1f})</span></td>' for g in ("El Niño", "Neutral", "La Niña"))
            + f'<td class="num">{_pv(bt[("IMERG", t, "El Niño")]["p"])} · {_pv(bt[("ERA5", t, "El Niño")]["p"])}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">Fewest to most storms per winter, IMERG {_yr(en["first"])}–{_yr(2025)}. A storm is a run of rainy days as defined above, here sized by '
             f'its total rain over {html.escape(A.ref)}. The last column tests El Niño winters against all others (Mann–Whitney): first on IMERG, then on ERA5 '
             f'{_yr(e_en["first"])}–{_yr(2025)} with thresholds that give the same number of storms as IMERG\'s over the shared years '
             f'({", ".join(f"{thr[t]:.0f}" for t in TOTAL_TIERS)} mm).</p>')
    return "\n".join(o)


def _sig(v: float, n: int = 2) -> str:
    """A count rounded to n significant figures (reported counts are rarely better than that)."""
    k = int(np.floor(np.log10(abs(v)))) - (n - 1) if v else 0
    return f"{int(round(v / 10 ** k) * 10 ** k):,}" if k > 0 else f"{v:,.0f}"


def _join(items) -> str:
    v = list(items)
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]


def _span(v: list[tuple[float, bool]]) -> str:
    """The range of a sorted list of counts: three significant figures as reported, two where converted from households."""
    f = [_sig(x, 2 if derived else 3) for x, derived in v]
    return "—" if not f else (f[0] if f[0] == f[-1] else f"{f[0]}–{f[-1]}")


def render_winter_impacts(spec: dict, a: dict) -> str:
    """Section 5, last part: rain against reported impact winter by winter; the same storm sizes before and
    since October 2023; and what last winter's counts imply for this winter's exposure. The last is a set
    of scenarios scaled from one winter, labelled as such: not a forecast and not a fitted model."""
    wi, st, cfg = a["wi"], a["st"], spec["winter_impacts"]
    S, sh, rain, ry, hh = wi["S"], wi["shift"], wi["rain"], wi["ry"], wi["hh"]
    T = S.loc[wi["first"]:]
    n_w, n_all = len(T), len(S)
    o = ['<h3>Rain and reported impact, winter by winter</h3>' + cfg.get("table_intro_html", "")]

    # one row per winter
    def people(y, r) -> str:
        if pd.isna(r.people):
            return '<span class="small">not counted</span>' if r.n_ev else '<span class="small">none found</span>'
        if y >= SINCE:
            return f'≥ {_sig(r.people)}' if pd.isna(r.people_hi) else f'{_sig(r.people)}–{_sig(r.people_hi)}'
        return f'about {_sig(r.people)}' if r.derived else f'{r.people:,.0f}'

    def tents(r) -> str:
        if pd.isna(r.tents):
            return "—"
        un = f'about {_round_to(r.tents, 500)}' if r.tents >= 5000 else f'about {_sig(r.tents)}'
        return un + (f'<br><em>about {_sig(r.tents_auth, 3)}</em>' if pd.notna(r.tents_auth) and r.tents_auth > r.tents else '')

    def deaths(y, r) -> str:
        if y == ry:
            return f'<em>{a["ref"]["deaths"]:.0f}–{a["ref"]["deaths_hi"]:.0f}</em>'
        return cfg.get("deaths", {}).get(str(y), "—" if pd.isna(r.deaths) else f'{r.deaths:.0f}')

    head = lambda t: f'<tr><td colspan="9" class="small" style="background:var(--n05)"><strong>{t}</strong></td></tr>'
    o.append('<div style="overflow-x:auto"><table><thead><tr><th>Winter</th><th>ENSO</th><th class="num">Rain, Oct–Apr</th>'
             '<th class="num">Wettest 7 days</th><th class="num">Storms ≥ 20 mm</th><th class="num">People reported affected</th>'
             '<th class="num">Tents and shelters damaged</th><th class="num">Deaths</th><th>What was counted</th></tr></thead><tbody>')
    o.append(head(cfg["before_label"]))
    for y, r in T.iterrows():
        if y == SINCE:
            o.append(head(cfg["since_label"]))
        o.append(f'<tr><td style="white-space:nowrap">{_yr(y)}</td><td style="white-space:nowrap">{r.phase}</td><td class="num" style="white-space:nowrap">{r.total:.0f} mm</td>'
                 f'<td class="num" style="white-space:nowrap">{r.rx7:.0f} mm</td><td class="num">{r.n20:.0f}</td><td class="num" style="white-space:nowrap">{people(y, r)}</td>'
                 f'<td class="num" style="white-space:nowrap">{tents(r)}</td><td class="num" style="white-space:nowrap">{deaths(y, r)}</td>'
                 f'<td class="small">{cfg.get("notes", {}).get(str(y), "")}</td></tr>')
    o.append('</tbody></table></div>')
    o.append(f'<p class="small">People are as the source gives them, or households multiplied by {hh[False]:.1f} before October 2023 and {hh[True]:.1f} '
             f'since, shown as “about” or rounded. Rain: IMERG, area-weighted over {html.escape(A.ref)}; the wettest 7 days is the highest 7-day running '
             'total of the season; storms as defined above, sized by their wettest day. Household sizes: the average household in Gaza (PCBS census 2017) '
             'and people per household in displacement sites (September 2026). A person is counted once for each '
             f'storm that affected them. {_yr(ry)}: the lower figure sums the per-event counts and the higher uses the Shelter Cluster\'s monthly snapshots, '
             'as in “Last winter in numbers”. Tents and shelters: UN and cluster counts. Figures in italics come from the Government Media Office and '
             'Palestinian Civil Defense (tents) or the Ministry of Health in Gaza (deaths), relayed by OCHA. Deaths before October 2023 are those UN or Red Cross and Red Crescent reports attribute to the storms. '
             '“Not counted” means an impact was reported without a number; “none found” means no weather impact was found in UN reporting for that winter.</p>')

    # which measure of the weather tracks the counts: the panels, then what they do not show
    cw, ca, ce, cwet = wi["corr_w"], wi["corr_all"], wi["corr_e"], wi["corr_wet"]
    n_cw, n_en = wi["n_before"], rain["en_n"]
    names = {"rx7": "wettest week", "rx1": "wettest day", "total": "season total", "n20": "the number of storms of 20 mm or more",
             "d10": "days of 10 mm or more", "rd": "rain days", "windy": "windy days", "cold8": "cold nights"}
    rk = sorted((k for k in ca if k in names), key=ca.get, reverse=True)
    k1, k2, k3 = rk[:3]
    weak = [k for k in rk[3:] if abs(ca[k]) < 0.3]; neg = [k for k in rk[3:] if ca[k] <= -0.3]
    if "rx1" in S:
        o.append(f'<figure><img src="winter_hazards.png" alt="People reported affected per winter against eight measures of the winter\'s weather">'
                 f'<figcaption>Each dot is one winter with a count, against eight measures of that winter\'s weather over {A.ref} (IMERG unless marked). '
                 f'Shaded: the measure\'s range in the {_nword(n_en)} El Niño winters since {_yr(int(S.index.min()))}, which spans most of each axis. '
                 f'ρ: rank correlation across the {_nword(n_cw)} counted winters before October 2023. Bar on {_yr(ry)}: the two alert-based sums. Windy days: '
                 f'the top 5% of ERA5 winter days. Cold nights: an ERA5 minimum of 8 °C or less.</figcaption></figure>')
    o.append(f'<p>Before October 2023 the counts followed the winter\'s biggest storm. Across the {_nword(n_cw)} counted winters, the {names[k1]} '
             f'(rank correlation {ca[k1]:+.2f}) and {names[k2]} ({ca[k2]:+.2f}) rank them best, ahead of the {names[k3]} ({ca[k3]:+.2f})'
             + (f'; {" and ".join(names[k] for k in neg)} run the other way ({", ".join(f"{ca[k]:+.2f}" for k in neg)})' if neg else '')
             + (f', and {_join(names[k] for k in weak)} show no relation' if weak else '') + f'. {_nword(n_cw).capitalize()} winters is a small sample. '
             + cfg.get("corr_note_html", "") + ' Since October 2023 the two counted winters sit far above every earlier one on every measure'
             + (f', and the driest winter of the {n_all}-winter record, {_yr(rain["driest"])}, has more people reported affected (at least '
                f'{_sig(S.people[rain["driest"]])}) than the wettest, {_yr(rain["wettest"])} ({_sig(S.people[rain["wettest"]])})'
                if rain["driest"] >= SINCE > rain["wettest"] and S.people.get(rain["driest"], np.nan) > S.people.get(rain["wettest"], np.nan) else '')
             + '.</p>')

    # storm for storm
    tr = {(t["lo"], t["since"]): t for t in wi["tiers"]}
    cell = lambda t: (f'<td class="num">{t["n"]}</td><td class="num">{t["reported"]}</td><td class="num" style="white-space:nowrap">{_span(t["people"])}'
                      + (f' <span class="small">({len(t["people"])})</span>' if t["people"] else '') + '</td>')
    o.append('<h3>Storm for storm, before and since October 2023</h3>' + cfg.get("storm_for_storm_html", ""))
    o.append(f'<div style="overflow-x:auto"><table><thead><tr><th rowspan="2">Storm size<br><span class="small">wettest day, IMERG</span></th>'
             f'<th colspan="3">Before October 2023 <span class="small">({_yr(wi["first"])}–{_yr(SINCE - 1)}, {SINCE - wi["first"]} winters)</span></th>'
             f'<th colspan="3">Since October 2023 <span class="small">({_yr(SINCE)}–{_yr(int(S.index.max()))}, {int(S.index.max()) - SINCE + 1} winters)</span></th></tr>'
             '<tr><th class="num">Storms</th><th class="num">With a reported impact</th><th class="num">People counted</th>'
             '<th class="num">Storms</th><th class="num">With a reported impact</th><th class="num">People counted</th></tr></thead><tbody>')
    for lo, hi_ in SIZE_TIERS:
        o.append(f'<tr><td>{f"{lo}–{hi_:.0f} mm" if np.isfinite(hi_) else f"{lo} mm or more"}</td>{cell(tr[(lo, False)])}{cell(tr[(lo, True)])}</tr>')
    o.append('</tbody></table></div>')
    low, missed = wi["low"], wi["missed"]
    miss = " ".join(f'The event of {m["start"]:%-d}–{m["end"]:%-d %B %Y} ({_sig(m["people"], 2 if m["derived"] else 3)} people) is not in the table: IMERG has '
                    f'{m["im"]:.1f} mm on its wettest day and ERA5 {m["era5"]:.0f} mm, so IMERG missed most of its rain.' for m in missed).replace(
                        "is not in the table:", "is not in the size tiers above:")
    o.append(f'<p class="small">Storms over {html.escape(A.ref)} in IMERG, as defined above. “With a reported impact”: any entry in the event '
             'catalogue, with or without a count, overlaps the storm (from the day before the entry starts to the day it ends). “People counted”: the '
             'range across events with a dated count, sized by the wettest day in the event\'s own dates, with the number of counts in brackets. Counts '
             f'of assistance delivered, with no rain date, are left out. {miss} ' + cfg.get("tier_note", "") + '</p>')
    U, evs = wi["U"], cfg.get("events", {})
    if set(U.event_id) - set(evs):
        raise ValueError(f"[winter_impacts.events] lacks {sorted(set(U.event_id) - set(evs))}")
    o.append(f'<figure><img src="event_hazards.png" alt="People counted per event against the rain over the event\'s dates">'
             f'<figcaption>{html.escape(A.event_sources)} Each dot is one event with a dated count, against the rain over {A.ref} from the '
             f'day before the event to its last day; the table below lists each with its source. Top: how many storms with at least that much rain a '
             f'winter has brought, in El Niño winters and in the others. That is every rain spell of that size, whether or not anyone was counted: a '
             f'past range, not a forecast.</figcaption></figure>')
    o.append('<div style="overflow-x:auto"><table><thead><tr><th>Dates</th><th>Hazard</th><th class="num">Rain over the event</th>'
             '<th class="num">People counted</th><th>What the count is</th><th>Source</th></tr></thead><tbody>')
    for r in U.itertuples():
        if r.since and not U[U.no < r.no].since.any():
            o.append('<tr><td colspan="6" class="small" style="background:var(--n05)"><strong>Since October 2023</strong></td></tr>')
        d = (f'{r.start:%-d %b %Y}' if r.start == r.end else f'{r.start:%-d %b %Y}–{r.end:%-d %b %Y}' if r.start.year != r.end.year
             else f'{r.start:%-d}–{r.end:%-d %b %Y}' if r.start.month == r.end.month else f'{r.start:%-d %b}–{r.end:%-d %b %Y}')
        d = ("about " if str(getattr(r, "date_quality", "")) == "M" else "") + d
        ev_ = evs.get(r.event_id, {})
        src = html.escape(ev_.get("source", str(r.source_publisher)))
        o.append(f'<tr><td style="white-space:nowrap">{d}</td><td>{", ".join(HAZARD_WORD[h] for h in r.hazard.split("+"))}</td>'
                 f'<td class="num" style="white-space:nowrap">{r.im_sum:.0f} mm{"*" if r.missed else ""}</td><td class="num" style="white-space:nowrap">'
                 f'{("about " + _sig(r.people)) if r.derived else f"{r.people:,.0f}"}</td><td class="small">{ev_.get("what", "")}</td>'
                 f'<td class="small" style="white-space:nowrap"><a href="{html.escape(str(r.source_url_primary))}">{src}</a></td></tr>')
    o.append('</tbody></table></div>')
    o.append(f'<p class="small">“About”: the source gives households or families, converted at {hh[False]:.1f} people before October 2023 and '
             f'{hh[True]:.1f} since. Rain: IMERG over {html.escape(A.ref)}, from the day before the event to its last day. '
             + "".join(f'* IMERG missed this storm\'s rain: ERA5 has {r.era5_max:.0f} mm. ' for r in U[U.missed].itertuples())
             + 'Wind and cold are not plotted: ERA5\'s area-mean wind and night minimum do not resolve the gusts and the cold in tents that the reports '
             'describe; the hazard column is from the reports. Events reported without a '
             'number, and counts of assistance delivered with no rain date, are in the tables of section 4 but not here.</p>')
    t20b, t20s = tr[(20, False)], tr[(20, True)]
    f_all, sh_ = sh["all"], sh
    tb, ts = sh["top"][False], sh["top"][True]
    did = f'all {t20s["n"]}' if t20s["reported"] == t20s["n"] else f'{t20s["reported"]} of {t20s["n"]}'
    o.append(f'<p>Two things changed. More storms do harm: {t20b["reported"]} of the {t20b["n"]} storms of 20–50 mm left an entry in the event catalogue '
             f'before October 2023, and {did} since.'
             + (f' {_nword(len(low[True])).capitalize()} events with less than 10 mm of rain (high tides and wind), with no counterpart before October 2023, '
                f'were counted at {_span(low[True])} people each.' if low[True] and not low[False] else '')
             + f' And each does more: the counts are tens to hundreds of times higher, about {sh_["med"]:.0f} times on medians ({_sig(sh_["med_since"])} '
             f'people against {_sig(sh_["med_before"])}) and, as a scale check allowing for each storm\'s rain, about {_sig(f_all["mult"])} times (95% '
             f'interval about {_sig(f_all["lo"])} to {_sig(f_all["hi"])}, on {f_all["n"]} events). ' + cfg.get("top_storms_html", "").format(
                 since_rain=_round_to(ts["rain"], 10), since_people=f'{ts["people"]:,.0f}', before_rain=_round_to(tb["rain"], 10),
                 before_people=f'{tb["people"]:,.0f}', ratio=f'{ts["people"] / tb["people"]:.0f}')
             + f'<!--more-->The scale check is a model fitted to the {f_all["n"]} events with a dated count: one slope on the storm\'s rain total, plus a '
             f'step in October 2023. Without the largest storm of each period the step is about {_sig(sh_["no_top"]["mult"])} times; on geometric means, '
             f'ignoring rain, about {sh_["geo"]:.0f}. ' + cfg.get("model_note", "")
             + (f' Since October 2023 rain alone does not order the counts: across the {_nword(ce[True]["n"])} dated counts, people affected and rain are '
                f'barely related (rank correlation {ce[True]["im_max"][0]:+.2f} with the wettest day), though among the {_nword(cwet["n"])} with a day of '
                f'10 mm or more the bigger storms were counted higher ({cwet["rho"]:+.2f}).' if cwet["n"] >= 4 else '') + '</p>')
    o.append(cfg.get("multiplier_caveat_html", ""))

    # what this October's forecast implies for October–December
    q = wi.get("ond")
    if q and q["top"]:
        x, top = q["clim"], int(S.loc[wi["first"]:SINCE - 1].people_ond.idxmax())
        o.append(f'<h3>What the October forecast implies for October–December</h3>'
                 f'<p>SEAS5\'s October {q["issued_year"]} forecast for October–December is wetter than {q["drier"]} of its {q["n"]} past October forecasts once '
                 f'the trend is taken out (section 1). In the {_nword(q["k"])} years with a forecast in that wettest fifth, ERA5 rain over {A.ref} in the '
                 f'quarter was {q["lo"]:.0f}–{q["hi"]:.0f} mm: {_nword(q["n_near"])} were close to the normal {q["clim_med"]:.0f} mm and '
                 f'{_nword(q["k"] - q["n_near"])} had {q["far_lo"]:.0f} mm or more. '
                 + (f'Last year\'s quarter, with Byron, had {x[ry]:.0f} mm after an October forecast in the driest fifth. '
                    if q["drier_by_year"].get(ry, q["n"]) < q["k"] else f'Last year\'s quarter, with Byron, had {x[ry]:.0f} mm. ')
                 + f'ERA5 flattens the biggest storms: it has the quarters of Storm Alexa and Byron at {x[top]:.0f} and {x[ry]:.0f} mm, where IMERG has '
                 f'{S.q4_imerg[top]:.0f} and {S.q4_imerg[ry]:.0f} mm. Read where a winter sits, not its millimetres.</p>')
        o.append(f'<figure><img src="forecast_quarter.png" alt="People reported affected in October–December against that quarter\'s ERA5 rain, with the range '
                 f'implied by the October forecast"><figcaption>People counted in events dated October–December against that quarter\'s rain in ERA5. Shaded: the rain in the '
                 f'{_nword(q["k"])} years whose October forecast ranked in SEAS5\'s wettest fifth, as this year\'s does: the range of outcomes after such '
                 f'a forecast, not the spread of the ensemble. Those years (dark ticks): {", ".join(str(y) for y in q["years"])}. ERA5 from the team\'s '
                 f'admin-1 statistics, {int(x.index.min())}–{int(x.index.max())}. {_yr(ry)}: the bar reaches the Shelter Cluster\'s first-storm and December '
                 f'counts.</figcaption></figure>')

    # scenarios
    sc = {x_["key"]: x_ for x_ in wi["scen"]}
    rng = lambda x_, k="": f'about {_sig(x_["lo" + k])}–{_sig(x_["hi" + k])}'
    o.append('<h3>Scenarios for winter 2026/27</h3>' + cfg.get("scenario_intro_html", ""))
    o.append(f'<figure><img src="winter_impacts.png" alt="People reported affected per winter against rain, with scenarios for 2026/27"><figcaption>'
             f'Left: counted winters against October–April rain over {A.ref} (IMERG), log scale. Right: the two counted winters since October 2023 on a '
             f'linear scale, with the three scenarios as boxes, scaled from last winter\'s counts, not fitted to the dots, and the same wherever in the El '
             f'Niño range the winter falls. Bar on {_yr(ry)}: the two alert-based sums. The boxes span the rain of '
             f'the {_nword(n_en)} El Niño winters since {_yr(int(S.index.min()))} ({rain["en_min"]:.0f}–{rain["en_max"]:.0f} mm). Dots and boxes '
             f'count a person once per storm; the two horizontal lines count each person once.</figcaption></figure>')
    often = lambda x_: f'{x_["all"]} of {n_all} · {x_["en"]} of {n_en}'
    o.append('<div style="overflow-x:auto"><table><thead><tr><th>Scenario</th><th class="num">How often<br><span class="small">all winters · El Niño winters</span></th>'
             f'<th>Built from</th><th class="num">People-impacts as counted last winter</th><th class="num">At this September\'s exposure<br>'
             f'<span class="small">× {wi["scale"]:.2f}</span></th></tr></thead><tbody>')
    o.append(f'<tr><td>No storm of 50 mm or more</td><td class="num">{often(sc["none"])}</td><td>last winter without Byron</td>'
             f'<td class="num">{rng(sc["none"])}</td><td class="num">{rng(sc["none"], "_s")}</td></tr>')
    o.append(f'<tr class="hl"><td>One such storm, as last winter</td><td class="num">{often(sc["one"])}</td><td>last winter as counted</td>'
             f'<td class="num">{rng(sc["one"])}</td><td class="num">{rng(sc["one"], "_s")}</td></tr>')
    o.append(f'<tr><td>Two such storms</td><td class="num">{often(sc["two"])}'
             + (f'<br><span class="small">{", ".join(_yr(y) for y in sc["two"]["years"])}</span>' if sc["two"]["years"] else '')
             + f'</td><td>last winter plus a second Byron</td><td class="num">{rng(sc["two"])}</td><td class="num">{rng(sc["two"], "_s")}</td></tr>')
    o.append('</tbody></table></div>')
    two_none = [y for y in sc["two"]["years"] if S.n_ev[y] == 0]
    o.append(f'<p class="small">People-impacts count a person once for each storm that affected them. Each range runs from the sum of the per-event counts '
             f'(people as reported where a source gives them, {hh[True]:.1f} per household otherwise) to the Shelter Cluster\'s monthly snapshots (households at '
             f'{hh[True]:.1f} people), as in “Last winter in numbers”. Byron is taken out or added at the Shelter Cluster\'s {wi["byron"]:,.0f} people in '
             f'the lower figures and at the Site Management Cluster\'s {a["ref"]["byron_hh"]:,.0f} households (about {_sig(wi["byron_hi"], 3)} people) in '
             f'the higher ones. “How often” counts IMERG winters {_yr(int(S.index.min()))}–{_yr(int(S.index.max()))}. '
             + (f'The only winter with two such storms, {_yr(two_none[0])}, was before October 2023 and left no entry in the event catalogue. '
                if len(sc["two"]["years"]) == 1 and two_none else '') +
             f'The no-storm case is last winter with Byron taken out: {S.n20[ry] - 1:.0f} storms of 20 mm or more remain '
             f'({next(r_["mean"] for r_ in st["rows"] if (r_["src"], r_["tier"], r_["group"]) == ("IMERG", 20, "El Niño")):.0f} in an average El Niño winter). '
             f'Exposure: about {wi["sites_now"] / 1e6:.1f} million people in displacement sites in September 2026 against about '
             f'{wi["sites_then"] / 1e6:.1f} million in early December 2025; the two counts used different methods.</p>')
    dry = rain["driest"]
    o.append('<ul>'
             '<li><strong>One winter, one storm.</strong> Every figure is last winter\'s count rescaled, and the two larger cases rest on Byron alone.</li>'
             f'<li><strong>Alert counts are floors.</strong> The Shelter Cluster\'s household survey found {100 * wi["survey"] / wi["sites_now"]:.0f}% of '
             f'households flooded at least once last winter' + cfg.get("survey_note", "") + f'. At the same share, about {wi["survey"] / 1e6:.1f} million of '
             f'the {wi["sites_now"] / 1e6:.1f} million people in displacement sites would be flooded at least once this winter, each counted once. Last '
             'winter the alerts caught a third or less of the households the survey found.</li>'
             + cfg.get("scenario_limits_html", "") +
             (f'<li><strong>A dry winter is not a safe one.</strong> {_yr(dry)}, the driest winter of the record ({S.total[dry]:.0f} mm), still had at least '
              f'{_sig(S.people[dry])} people reported affected, by high tides and wind. No El Niño winter since {_yr(int(S.index.min()))} has been that dry, '
              'but sea surge, wind and cold come whatever the rain.</li>' if dry >= SINCE and pd.notna(S.people[dry]) else '')
             + '<li><strong>Deaths are not scaled.</strong> They came from collapsing buildings and cold nights, which do not follow the number of people in sites.</li>'
             '</ul>')
    return "\n".join(o)


def render_blocks(spec: dict, a: dict, headings: bool = True) -> dict[str, list[str]]:
    """The page in named blocks (lists of HTML chunks), without the section headings, so a page can be
    assembled for one area (render) or interleaved with another area's (render_combined)."""
    ours, cat = spec["assessment"], spec.get("catalogue")
    T = lambda k, d: html.escape(spec.get("titles", {}).get(k, d))
    B: dict[str, list[str]] = {k: [] for k in ['verdict', 'summary', 'before', 'forecast_lead', 'forecast_text', 'forecast_area', 'enso', 'daily', 'impacts', 'range', 'agri', 'after', 'refs']}
    # verdict (a part of a combined page has no catalogue row or summary of its own)
    B["verdict"].append('<div class="verdict">')
    if cat:
        B["verdict"].append(f'<div class="card"><p class="lbl">Survey catalogue says</p><p class="big">El Niño → {html.escape(cat["direction"])}, {html.escape(cat["season"])}</p>'
                 f'<p>{edd.chip(cat["evidence"])} <span class="small">source: {cat["source_html"]}</span></p></div>')
    B["verdict"].append(f'<div class="card"><p class="lbl">This review assesses</p><p class="big">El Niño → {html.escape(ours["direction"])}, {html.escape(ours["season"])}</p>'
             f'<p>{edd.chip(ours["evidence"])} <span class="small">{html.escape(ours["evidence_note"])}</span></p></div>')
    B["verdict"].append('</div>')
    if spec.get("summary_html"):
        B["summary"].append(f'<div class="summary">{spec["summary_html"]}</div>')
    if spec.get("before_html"):
        B["before"].append(spec["before_html"])

    # 1. Forecasts
    nn = a["nino_now"]
    B["forecast_lead"].append(f'<p class="small">Latest Niño3.4 in NOAA PSL\'s current series: {nn["value"]:+.2f} °C for {nn["date"]:%B %Y} '
             f'(three-month mean {nn["mean3"]:+.2f}). The historical analysis on this page uses the series\' pinned Niño3.4 '
             f'(ERSST v5 basis, about 0.2 °C cooler than the current ERSST v6 series), and HadISST only where a record starts before 1950.</p>')
    B["forecast_text"].append(spec.get("forecast_html", ""))
    fc = spec.get("forecasts", [])
    if fc:
        B["forecast_text"].append('<div style="overflow-x:auto"><table><thead><tr><th>Forecast</th><th>Issued</th><th>Oct–Dec</th><th>Nov–Jan</th>'
                 '<th>Dec–Feb</th><th>Temperature</th></tr></thead><tbody>')
        for f in fc:
            B["forecast_text"].append(f'<tr><td>{f["source_html"]}</td><td style="white-space:nowrap">{html.escape(f.get("issued", ""))}</td>'
                     f'<td>{f.get("ond", "")}</td><td>{f.get("ndj", "")}</td><td>{f.get("djf", "")}</td><td>{f.get("temp", "")}</td></tr>')
        B["forecast_text"].append('</tbody></table></div>')
        B["forecast_text"].append(spec.get("forecast_table_note", ""))
    if a.get("skill"):
        sk_html = (edd.render_skill(spec, a, "OND", spec.get("titles", {}).get("skill", f"SEAS5 for {A.name}: skill and the September forecast"))
                 .replace("whole country (bars) and zones (lines)", f"the cells over {A.ref} (bars)")
                 .replace("for the whole country", f"for the cells over {A.ref}")
                 .replace("of the's annual rain", f"of {A.ref}'s annual rain")
                   )
        sk = a["skill"]
        if sk.get("db"):       # return periods from the database while the app's pixel cube lags the new issuance
            import calendar
            mn = calendar.month_name[sk["issued_month"]]
            sk_html = (sk_html
                       .replace("forecast_rp / flood_rp), median pixel;", "forecast_rp / flood_rp), here for the area mean (see the note above the figure);")
                       .replace(f"median return period of the {edd.MONTH_NAMES[sk['issued_month'] - 1]} {sk['issued_year']} forecast anomaly",
                                f"return period of the {edd.MONTH_NAMES[sk['issued_month'] - 1]} {sk['issued_year']} forecast anomaly for the area mean"))
            note = (f'<p class="small"><strong>Source of the {mn} {sk["issued_year"]} return periods.</strong> The seas5-skill app\'s pixel product, which this '
                    f'page normally reads, is recomputed the day after a new forecast lands, so the return periods here come from the team\'s '
                    f'raster statistics for {html.escape(A.ref)} as one area (the ensemble mean averaged over the admin-1 polygon, in the team '
                    f'database), ranked against the {mn} hindcasts of 1981–{sk["issued_year"] - 1} with the app\'s own method. Skill is still the '
                    'median pixel correlation from the app\'s cube, which depends on the hindcasts and not on the new forecast. Windows that had '
                    'already started at issuance are left blank: they blend in observed months, and the latest ERA5 month arrives a day after the forecast.</p>')
            sk_html = sk_html.replace('<figure><img src="skill_issued.png"', note + '<figure><img src="skill_issued.png"', 1)
            # the app's alert rule, read with both measures of skill: the cube's pixel median and the area mean's own r
            lo_ = edd.SKILL_THRESH["r_mod"]; row_ = sk["rows"][0]; clear, border = [], []
            for t in sk["trimesters"]:
                v = sk["db"]["rows"].get(t["code"])
                if t["lead"] < 0 or not v or abs(v["rp"]) < 3:
                    continue
                rc, ra = row_["r"].get(t["code"], float("nan")), v["r"]
                if rc >= lo_ and ra >= lo_:
                    clear.append(t["code"])
                elif rc >= lo_ or ra >= lo_:
                    border.append(f'{t["code"]} (pixel-median r {rc:.2f}, area-mean r {ra:.2f})')
            rule = ("Under the app\'s rule, an alert needs the return period <em>and</em> at least moderate skill. Until the pixel product is "
                    "out this is provisional: " + (f'{", ".join(clear)} meet both conditions on either measure of skill' if clear else
                                                    'no window meets both conditions on both measures of skill')
                    + (f'; {", ".join(border)} {"sits" if len(border) == 1 else "sit"} at the moderate-skill boundary' if border else '') + '.')
            sk_html, n_sub = re.subn(r"Under the app\'s rule — .*?(?:this issuance would raise an alert for [^.]*\.|because the skill behind them is low\.)",
                                     lambda m_: rule, sk_html, count=1, flags=re.S)
            assert n_sub == 1, "alert sentence not found in the skill block"
        # the shared auto-summary names the zone before each window ("Gaza SON, Gaza OND"): drop it
        B["forecast_area"].append(re.sub(rf"\b{re.escape(A.name)} (?=[A-Z]{{3}}\b)", "", sk_html))
    sr = a.get("seas5_raw")
    if sr:
        B["forecast_area"].append(f'<p class="small">Cross-check from the raw ensemble-mean files (box {A.seas5_box[1]:.1f}–{A.seas5_box[3]:.1f}°N, '
                 f'{A.seas5_box[0]:.1f}–{A.seas5_box[2]:.1f}°E), {edd.MONTH_NAMES[sr["month"] - 1]} issuances '
                 f'1981–{sr["year"]}: where the {sr["year"]} forecast ranks among all {sr["rows"][0]["n"]} (1 = wettest), its ratio to the hindcast mean, '
                 'and the three wettest hindcast years.</p>')
        B["forecast_area"].append('<table><thead><tr><th>Window</th><th class="num">Rank</th><th class="num">Ratio to mean</th><th>Wettest hindcasts (issuance year)</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{r["code"]}</td><td class="num">{r["rank"]} of {r["n"]}</td><td class="num">{r["ratio"]:.2f}</td>'
                           f'<td>{", ".join(str(y) for y in r["top"])}</td></tr>' for r in sr["rows"])
                 + '</tbody></table>')
    sm = a.get("seas5_mon")
    if sm:
        B["forecast_area"].append(f'<h3>The same forecast, month by month</h3>{spec.get("monthly_html", "")}')
        import calendar
        imn = calendar.month_name[sr["month"]]
        dry0 = (f' {calendar.month_name[sm[0]["month"]]} gets about {sm[0]["obs_clim"]:.0f} mm in an average year, so its skill says little.'
                if sm[0]["obs_clim"] < 5 else '')
        B["forecast_area"].append('<figure><img src="seas5_monthly.png" alt="SEAS5 forecast and skill by month"><figcaption>Top: the SEAS5 ensemble-mean '
                 f'rainfall for each month of the {imn} issuance, averaged over a box on {A.ref} ({A.seas5_box[1]:.1f}–{A.seas5_box[3]:.1f}°N, {A.seas5_box[0]:.1f}–{A.seas5_box[2]:.1f}°E), against the mean of '
                 f'the same month in the 1981–{sr["year"] - 1} {imn} hindcasts; labels give the forecast as a percentage above or below that mean and its '
                 f'rank among all {sm[0]["n"]} {imn} issuances (1 = wettest). Bottom: the skill of that month\'s forecast, the correlation between the '
                 f'detrended hindcast ensemble mean and detrended ERA5 rainfall over {A.ref}, on the app\'s low / moderate / high bands. A single month '
                 f'is noisier than a three-month window, so monthly skill is lower than in the figure above.{dry0}</figcaption></figure>')
        B["forecast_area"].append('<table><thead><tr><th>Month</th><th class="num">Hindcast mean</th><th class="num">Forecast</th><th class="num">vs mean</th>'
                 f'<th class="num">Rank (1 = wettest)</th><th class="num">Skill r</th><th class="num">ERA5 mean, {A.name}</th></tr></thead><tbody>'
                 + "".join(f'<tr><td>{edd.MONTH_NAMES[r["month"] - 1]}</td><td class="num">{r["hind"]:.0f} mm</td><td class="num">{r["fc"]:.0f} mm</td>'
                           f'<td class="num">{_pct_signed(r["pct"])}</td><td class="num">{r["rank"]} of {r["n"]}</td>'
                           f'<td class="num">{_r(r["r"])} {edd.skill_chip(edd.skill_cat(r["r"]))}</td><td class="num">{r["obs_clim"]:.0f} mm</td></tr>' for r in sm)
                 + '</tbody></table>')
        B["forecast_area"].append('<p class="small">SEAS5 ensemble-mean totals are smoother than any single year, so compare the forecast with the hindcast mean and '
                 f'rank, not with observed rainfall. ERA5 mean: the {_nword(len(a["wts"]))} cells over {A.ref}, same years, for scale.</p>')

    # 2. ENSO and the rainy season
    B["enso"].append(spec.get("enso_intro_html", ""))
    nw = len(a["wts"])
    B["enso"].append(f'<figure><img src="seasonal_cycle.png" alt="Monthly rainfall climatology for {A.name}"><figcaption>Monthly climatology from three '
             f'records. {A.gpcc_desc}'
             f'ERA5 is the mean of the {_nword(nw)} 0.25° cells over {A.ref}, weighted by the share of {A.ref} in each'
             + (f' ({", ".join(f"{100 * v:.0f}%" for v in a["wts"].values())})' if nw <= 4 else '')
             + f'; IMERG is the area-weighted mean of the 0.1° cells over {A.ref}.'
             '</figcaption></figure>')
    B["enso"].append(f'<h3>The link switched on in the late 1970s</h3>{spec.get("stationarity_html", "")}')
    B["enso"].append(f'<figure><img src="stationarity.png" alt="Running correlation between {A.name} rainfall and Niño3.4"><figcaption>Pearson r between the '
             'October–April total and December–February Niño3.4 (HadISST) in centred 31-year windows. ' + A.stationarity_records + '</figcaption></figure>')
    per = a["per"]
    B["enso"].append('<table><thead><tr><th>Record</th><th>Winters</th><th>Niño3.4</th><th class="num">r</th><th class="num">p</th>'
             '<th class="num">r, detrended</th></tr></thead><tbody>')
    for k in ("GPCC", "gauge", "ERA5", "IMERG"):
        for tag in ("pre", "post", "recent", "full"):
            v = per.get((k, tag))
            if not v or (k == "IMERG" and tag != "recent") or (tag == "recent" and k == "gauge"):
                continue
            cls = ' class="hl"' if tag == "post" else ""
            B["enso"].append(f'<tr{cls}><td>{html.escape(SRC_LABEL[k])}</td><td>{_yr(v["first"])} – {_yr(v["last"])}</td><td class="small">{v["index"]}</td>'
                     f'<td class="num">{_r(v["r"])}</td><td class="num">{_p(v["p"])}</td><td class="num">{_r(v["r_detr"])}</td></tr>')
    B["enso"].append('</tbody></table>')
    dd = a["diff"]
    B["enso"].append('<p class="small">Is the change real? A Fisher z-test for the difference between the correlations before and after '
             f'{SPLIT}: ' + "; ".join(f'{html.escape(_disp(k))} z = {v["z"]:.1f}, p {"&lt; 0.001" if v["p"] < 0.001 else "= " + _p(v["p"])}' for k, v in dd.items()) + '. '
             f'The {SPLIT} break was not chosen blind: it is where the literature places the change (Price et al. 1998; Alpert et al. 2005) '
             'and the start of the satellite era. p-values for the period after the break are conditional on choosing it; '
             'the full-record rows show what an unsplit analysis gives.</p>')
    B["enso"].append(f'<h3>What El Niño winters have looked like since {SPLIT}</h3>{spec.get("history_html", "")}')
    B["enso"].append('<figure><img src="phase_history.png" alt="Rainy-season totals by ENSO phase"><figcaption>October–April GPCC totals as a percentage above or below '
             'the 1991–2020 mean, coloured by the ENSO phase of the same winter (December–February Niño3.4, pinned ERSST v5 series throughout, so '
             'a few 1950s–70s winters are classed differently from the HadISST-based table below). Winters with Niño3.4 ≥ +1.5 °C are labelled; '
             f'the dashed line marks {SPLIT}.</figcaption></figure>')
    B["enso"].append('<table><thead><tr><th>Record, period</th><th>ENSO phase</th><th class="num">Winters</th><th class="num">Wettest third</th>'
             '<th class="num">Middle third</th><th class="num">Driest third</th></tr></thead><tbody>')
    for (k, lo, hi), rows in a["ptab"].items():
        for r in rows:
            cls = ' class="hl"' if (r["phase"] == "El Niño" and lo == SPLIT) else ""
            B["enso"].append(f'<tr{cls}><td>{html.escape(k)}, {_yr(lo)} – {_yr(min(hi, int(a["tot"][k].index.max())))}</td><td>{r["phase"]}</td><td class="num">{r["n"]}</td>'
                     f'<td class="num">{r["wet"]}</td><td class="num">{r["mid"]}</td><td class="num">{r["dry"]}</td></tr>')
    B["enso"].append('</tbody></table>')
    B["enso"].append('<p class="small">Terciles are computed within each period, so each period is judged against its own climate. '
             f'Phase from December–February Niño3.4 (±0.5 °C): HadISST before {SPLIT}, the pinned ERSST v5 series from {SPLIT}. '
             f'On HadISST, {a["en_had"]["n"]} winters since {SPLIT} count as El Niño rather than 14, and {a["en_had"]["wet"]} of them '
             'were in the GPCC wettest third.</p>')
    B["enso"].append('<h3>Every strong El Niño winter since 1950</h3>' + spec.get("strong_html", ""))
    B["enso"].append('<table><thead><tr><th>Winter</th><th class="num">Niño3.4 DJF</th>'
             + "".join(f'<th class="num">{_short(k)}</th>' for k in ("GPCC", "ERA5", "gauge", "IMERG"))
             + '<th>Note</th></tr></thead><tbody>')
    for s_ in a["strong"]:
        cells = []
        for k in ("GPCC", "ERA5", "gauge", "IMERG"):
            v = s_.get(k)
            cells.append('<td class="num">—</td>' if not v else
                         f'<td class="num">{v["mm"]:.0f} mm<br><span class="small">{_ord(v["pct"])} pct · {v["third"]}</span></td>')
        note = spec.get("strong_notes", {}).get(str(s_["year"]), "")
        B["enso"].append(f'<tr><td>{_yr(s_["year"])}</td><td class="num">{s_["nino"]:+.1f}</td>{"".join(cells)}<td class="small">{note}</td></tr>')
    B["enso"].append('</tbody></table>')
    B["enso"].append(f'<p class="small">Niño3.4 ≥ +1.5 °C in December–February on the pinned series. Percentiles are within {SPLIT}–2025 for winters '
             f'from {SPLIT}, within 1950–{SPLIT - 1} before.</p>')
    B["enso"].append(f'<h3>Which part of the winter</h3>{spec.get("trimester_html", "")}')
    B["enso"].append('<table><thead><tr><th>Record</th><th>Window</th><th class="num">r, concurrent Niño3.4</th><th class="num">p</th>'
             '<th class="num">r, Aug–Oct Niño3.4</th><th class="num">p</th></tr></thead><tbody>')
    for t in a["tri_rows"]:
        if t["src"] == "IMERG":
            continue
        cls = ' class="hl"' if t["season"] == "Oct–Apr" else ""
        B["enso"].append(f'<tr{cls}><td>{t["src"]}</td><td>{t["season"]}</td><td class="num">{_r(t["r"])}</td><td class="num">{_p(t["p"])}</td>'
                 f'<td class="num">{_r(t["r_aso"])}</td><td class="num">{_p(t["p_aso"])}</td></tr>')
    B["enso"].append('</tbody></table>')
    B["enso"].append(f'<p class="small">{SPLIT}–2025. Concurrent = Niño3.4 averaged over the same three months (December–February for October–April). '
             'August–October Niño3.4 is the value known when the season starts.</p>')
    B["enso"].append(f'<h3>Across the region</h3>{spec.get("maps_html", "")}')
    B["enso"].append('<figure><img src="corr_maps.png" alt="Pixel correlation maps, southern Levant"><figcaption>Pearson r between three-month rainfall and '
             'Niño3.4 for each ERA5 0.25° cell, keeping the lag (0–3 months, index leading) with the largest |r| as the survey does. '
             f'Blue = wetter under El Niño. Grey cells hold under a quarter of their annual rain in that window. {A.ref[0].upper() + A.ref[1:]} is outlined.</figcaption></figure>')
    B["enso"].append(f'<figure><img src="composite_maps.png" alt="El Niño composite and wettest-third hit rate"><figcaption>Left: mean standardized '
             f'October–April anomaly over the {a["n_en_grid"]} El Niño winters of {_yr(a["n_grid"][0])}–{_yr(a["n_grid"][1])} '
             f'(cells over {A.ref}: {a["comp_gaza"]:+.2f} SD; positive in {100 * a["comp_region_pos"]:.0f}% of analysed cells). '
             f'Right: the share of those winters in each cell\'s wettest third ({A.name}: {100 * a["hit_gaza"]:.0f}%; regional median '
             f'{100 * a["hit_region"]:.0f}%; chance is 33%).</figcaption></figure>')

    # 3. Daily weather
    B["daily"].append(spec.get("daily_html", ""))
    B["daily"].append('<figure><img src="daily_by_phase.png" alt="Correlation of winter weather metrics with Niño3.4"><figcaption>Each dot is the '
             'correlation between one October–April metric and December–February Niño3.4 across the winters since '
             f'{SPLIT} (IMERG from 1998, {A.gauge_short} to {a["gauge_last"]}); lines are 95% intervals. Windy days: days whose highest hourly ERA5 10 m wind reaches '
             f'the top 5% of winter days ({a["wind95"]:.1f} m/s; ERA5 winds are cell averages and understate gusts). {_metric_label("cold")}: '
             + A.daily_cold_note + '</figcaption></figure>')
    B["daily"].append('<table><thead><tr><th>Metric</th><th>Record</th><th class="num">El Niño</th><th class="num">Neutral</th><th class="num">La Niña</th>'
             '<th class="num">r</th><th class="num">p</th></tr></thead><tbody>')
    unit = {"total": " mm", "rx1": " mm"}
    for mt in ["total", "d1", "d10", "d20", "rx1", "cold", "windy"]:
        for r in [r for r in a["daily_rows"] if r["metric"] == mt]:
            u = unit.get(mt, "")
            B["daily"].append(f'<tr><td>{_metric_label(mt)}</td><td>{html.escape(_disp(r["src"]))} {r["first"]}–{r["last"]}</td>'
                     f'<td class="num">{r["en"]:.1f}{u}</td><td class="num">{r["neu"]:.1f}{u}</td><td class="num">{r["ln"]:.1f}{u}</td>'
                     f'<td class="num">{_r(r["r"])}</td><td class="num">{_p(r["p"])}</td></tr>')
    B["daily"].append('</tbody></table>')
    B["daily"].append('<p class="small">Means per winter, by ENSO phase. The records differ in level: '
             + A.daily_note + ' Compare phases within a record, not across records.</p>')

    # 4. Impacts
    B["impacts"].append(spec.get("impacts_intro_html", ""))
    fr = a["freq"]
    B["impacts"].append(f'<p class="small">How often the rain that has caused these impacts comes: over 1998–2025, IMERG puts an average of '
             f'{fr[10]:.1f} days of ≥ 10 mm, {fr[20]:.1f} of ≥ 20 mm, {fr[30]:.1f} of ≥ 30 mm and {fr[50]:.1f} of ≥ 50 mm over {A.ref} in each October–April.</p>')
    B["impacts"].append(f'<figure><img src="winters.png" alt="Daily rainfall over {A.ref}, 2023/24 to 2025/26, with impacts"><figcaption>IMERG late run, '
             f'area-weighted mean over {A.ref}. Markers are the start dates of the reported impacts in the table below.</figcaption></figure>')
    B["impacts"].append(impact_table(a["ev_rows"], numbered=True))
    B["impacts"].append(f'<p class="small">{A.impact_caption}</p>')
    if a.get("pre_rows"):
        B["impacts"].append(f'<h3>Before October 2023</h3>{spec.get("prewar_html", "")}')
        B["impacts"].append(impact_table(a["pre_rows"], numbered=False))
    B["impacts"].append(spec.get("impacts_after_html", ""))
    if a.get("bars"):
        B["impacts"].append(render_winter_counts(spec, a["bars"]))
    if a.get("st"):
        B["range"].append(render_range(spec, a, headings))
    if a.get("agri"):
        import levant_agri
        B["agri"].append(levant_agri.render(spec, a, headings))
    for sec in spec.get("sections_after", []):
        B["after"].append(f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}')
    if spec.get("references"):
        B["refs"].append('<h2>References</h2><ul class="refs">' + "".join(f'<li>{r["html"]}</li>' for r in spec["references"]) + '</ul>')
    return B


def render(spec: dict, a: dict) -> str:
    ours, cat = spec["assessment"], spec["catalogue"]
    T = lambda k, d: html.escape(spec.get("titles", {}).get(k, d))
    o = [edd.HEAD.format(title=f"{A.name} — ENSO deep dive", desc=html.escape(ours["one_line"]), css=edd.CSS,
                         home="../", home_label="ENSO country deep dives")]
    o.append(f'<p class="eyebrow">ENSO country deep dive</p><h1>{html.escape(A.name)}</h1>')
    o.append(f'<p class="meta">{html.escape(spec.get("subtitle", ""))} &nbsp;·&nbsp; GPCC 1891–2025, ERA5 1950–{a["end_era5"]:%Y}, '
             f'IMERG 1998–{a["end_imerg"]:%Y}, {A.meta_gauge} &nbsp;·&nbsp; Niño3.4 (NOAA)</p>')
    B = render_blocks(spec, a)
    for k in ("verdict", "summary", "before"):
        o.extend(B[k])

    o.append(f'<h2>{T("forecast", "1. What the forecasts say for this winter")}</h2>')
    o.extend(B["forecast_lead"] + B["forecast_text"] + B["forecast_area"])

    o.append(f'<h2>{T("enso", f"2. How much El Niño matters for a {A.name} winter")}</h2>')
    o.extend(B["enso"])

    o.append(f'<h2>{T("daily", "3. What changes in an El Niño winter, and what does not")}</h2>')
    o.extend(B["daily"])

    o.append(f'<h2>{T("impacts", f"4. What winter weather does in {A.name}")}</h2>')
    o.extend(B["impacts"] + B["range"] + B["agri"] + B["after"] + B["refs"])
    o.append(f'<p class="small">Generated by <code>levant_deep_dive.py</code> from <code>deep_dives/{A.slug}.toml</code>. '
             'Grid method and Niño3.4 series as in the <a href="../../survey/">global survey</a>.</p>')
    o.append(edd.FOOT)
    return ("\n".join(o)).replace("<!--more-->", " ")   # the fold marker is for the combined page


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


PARTS_DIR = edd.DEEP_DIR / "parts"                 # area TOMLs that are built only as parts of a combined page
COMBINED_CSS = ("h3.area{font-size:17px;margin:28px 0 8px;padding:2px 0 2px 10px;border-left:4px solid var(--b5)}"
                "h4{font-size:15px;margin:20px 0 6px}h5{font-size:14px;margin:16px 0 4px}"
                "@media (min-width:641px){.verdict{grid-template-columns:1fr 1fr 1fr}}"
                ".keymsg{border:1px solid #cfe3dc;border-top:4px solid var(--b5);border-radius:6px;background:#fff;padding:14px 22px 10px;margin:4px 0 20px}"
                ".keymsg h2{border:0;margin:0 0 2px;padding:0;font-size:20px}.keymsg ol{margin:10px 0 8px;padding-left:22px}"
                ".keymsg li{margin:0 0 11px;max-width:92ch}.keymsg li::marker{font-weight:700;color:var(--b6)}.keymsg p.small{margin:0 0 6px}"
                "details.summary summary{cursor:pointer;font-weight:700}details.summary[open] summary{margin-bottom:10px}")
# Navigation on the combined page: a sticky section bar, a contents block, links between the areas of a
# section, a list of each long part's subsections, and an anchor on every heading.
NAV_CSS = ("h2,h3,h4,h5,#key-messages,#contents{scroll-margin-top:48px}"
           ".secbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;margin:0 -44px 16px;padding:7px 44px;"
           "background:#fff;border-bottom:1px solid #e2e7e7;font-size:12.5px;white-space:nowrap;overflow-x:auto;scrollbar-width:none}"
           ".secbar::-webkit-scrollbar{display:none}.secbar .item{display:inline-flex;align-items:center}"
           ".secbar a{color:var(--n8);text-decoration:none;padding:4px 6px;border-radius:4px}.secbar a:hover,.secbar a:focus-visible{background:var(--b05);color:var(--b7)}"
           ".secbar .item.cur>a{background:var(--b05);color:var(--b7);font-weight:700}.secbar .sub{display:none;font-size:12px;color:var(--n7)}"
           ".secbar .item.cur .sub,.secbar .item:focus-within .sub{display:inline}.secbar .sub a{padding:3px 5px}.secbar .sub a.cur{color:var(--b7);font-weight:700;text-decoration:underline}"
           ".contents{background:var(--n05);border:1px solid #e2e7e7;border-radius:6px;padding:12px 18px 8px;margin:16px 0 10px}"
           ".contents .lbl{font-size:11px;letter-spacing:.11em;text-transform:uppercase;font-weight:700;color:var(--n7);margin:0 0 4px}"
           ".contents ul{list-style:none;margin:0;padding:0}.contents li{display:flex;justify-content:space-between;align-items:baseline;gap:6px 18px;"
           "flex-wrap:wrap;padding:5px 0;border-top:1px solid #e6eaea}.contents li:first-child{border-top:0}.contents li>a{font-weight:500;text-decoration:none}"
           ".contents li>a:hover{text-decoration:underline}.contents .areas{font-size:13px;white-space:nowrap}.contents .areas a{margin-left:14px}"
           ".areahead{display:flex;justify-content:space-between;align-items:baseline;gap:4px 16px;flex-wrap:wrap;margin:28px 0 8px}"
           ".areahead h3.area{margin:0}.areahead .alt{margin:0;font-size:12.5px;color:var(--n7);max-width:none}.tscroll{overflow-x:auto}"
           ".inpart{font-size:12.5px;color:var(--n7);max-width:none;margin:0 0 12px;line-height:1.85}.inpart span{white-space:nowrap}"
           ".anchor{margin-left:7px;font:400 .8em 'Roboto',system-ui,sans-serif;color:var(--n7);text-decoration:none;opacity:0;user-select:none}"
           ".anchor::after{content:'#'}"
           "h2:hover .anchor,h3:hover .anchor,h4:hover .anchor,h5:hover .anchor{opacity:1}"
           ".keymsg .see,.summary .see{font-size:13px;color:var(--n7);white-space:nowrap}"
           "details.note{margin:2px 0 14px}details.note>summary{cursor:pointer;font-size:12.5px;color:var(--b6);width:fit-content}"
           "details.note>summary:hover{text-decoration:underline}details.note[open]>summary{margin-bottom:4px}details.note p{margin:0 0 6px}"
           "figcaption details.note{display:inline}figcaption details.note[open]{display:block;margin-top:4px}"
           ".foldall{font-size:12.5px;margin:6px 0 0}.foldall button{font:inherit;color:var(--b6);background:none;border:0;padding:0;cursor:pointer;text-decoration:underline}"
           "@media(max-width:1000px){.secbar{-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 28px),transparent);"
           "mask-image:linear-gradient(90deg,#000 calc(100% - 28px),transparent)}}"
           "@media(max-width:640px){.secbar{margin:0 -18px 12px;padding:6px 18px}.contents .areas a{margin:0 14px 0 0}}"
           "@media print{.secbar,.anchor,.areahead .alt,.inpart,.foldall,details.note>summary{display:none}.tscroll,div[style*='overflow-x']{overflow:visible}}")
NAV_JS = """<script>
(function(){
  var bar=document.querySelector('.secbar'); if(!bar) return;
  function pair(a){return {a:a, el:document.getElementById(a.getAttribute('href').slice(1))};}
  var secs=[].map.call(bar.querySelectorAll('.item>a'),pair), subs=[].map.call(bar.querySelectorAll('.sub a'),pair), last=null, tick=0;
  function update(){
    tick=0;
    var y=bar.getBoundingClientRect().bottom+16, cur=null, sub=null;
    secs.forEach(function(s){ if(s.el && s.el.getBoundingClientRect().top<=y) cur=s; });
    if(window.innerHeight+window.scrollY>=document.documentElement.scrollHeight-1) cur=secs[secs.length-1];
    subs.forEach(function(s){ if(s.el && s.el.getBoundingClientRect().top<=y) sub=s; });
    secs.forEach(function(s){ s.a.parentNode.classList.toggle('cur', s===cur); if(s===cur) s.a.setAttribute('aria-current','true'); else s.a.removeAttribute('aria-current'); });
    subs.forEach(function(s){
      var on = s===sub && !!cur && s.a.parentNode.parentNode===cur.a.parentNode;
      s.a.classList.toggle('cur', on); if(on) s.a.setAttribute('aria-current','location'); else s.a.removeAttribute('aria-current');
    });
    if(cur && cur!==last){
      last=cur; var it=cur.a.parentNode, b=bar.getBoundingClientRect(), r=it.getBoundingClientRect();
      if(r.left<b.left || r.right>b.right)
        bar.scrollTo({left: bar.scrollLeft + r.left - b.left - (bar.clientWidth - it.offsetWidth)/2,
                      behavior: window.matchMedia && matchMedia('(prefers-reduced-motion:reduce)').matches ? 'auto' : 'smooth'});
    }
  }
  window.addEventListener('scroll', function(){ if(!tick) tick=setTimeout(update, 60); }, {passive:true});
  window.addEventListener('resize', update); window.addEventListener('load', update); update();
})();
(function(){
  var notes=[].slice.call(document.querySelectorAll('details.note')), btn=document.querySelector('.foldall button'), open=false;
  if(btn && notes.length){ btn.parentNode.hidden=false; btn.addEventListener('click', function(){
    open=!open; notes.forEach(function(d){ d.open=open; }); btn.textContent=open ? 'Hide the notes and method details' : 'Show all notes and method details ('+notes.length+')'; });
    btn.textContent='Show all notes and method details ('+notes.length+')'; }
  var shut=[];
  window.addEventListener('beforeprint', function(){ shut=[].filter.call(document.querySelectorAll('details'), function(d){ return !d.open; }); shut.forEach(function(d){ d.open=true; }); });
  window.addEventListener('afterprint', function(){ shut.forEach(function(d){ d.open=false; }); shut=[]; });
})();
</script>"""
HEADING = re.compile(r"<(h[2-5])\b([^>]*)>(.*?)</\1>", re.S)


def _slug(t: str) -> str:
    t = unicodedata.normalize("NFKD", html.unescape(re.sub(r"<[^>]+>", "", t))).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "part"


def _scroll_tables(page: str) -> str:
    """A table wider than a narrow screen scrolls sideways in its own box instead of widening the whole page."""
    def wrap(m):
        return m.group(0) if re.search(r"<div[^>]*overflow-x:auto[^>]*>\s*$", page[:m.start()]) else f'<div class="tscroll">{m.group(0)}</div>'
    return re.sub(r"<table\b.*?</table>", wrap, page, flags=re.S)


def _words(t: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", t).split())


def _fold(page: str, note_words: int = 45, caption_words: int = 60, lead_words: int = 22) -> str:
    """Keep the page short to read without dropping anything: a long method note under a table or figure folds
    behind its own first words, and a long figure caption keeps its opening sentences and folds the rest."""
    start = page.find('id="contents"')                       # the key messages and the summary above it are left as written

    def note(m):
        t = m.group(1)
        if m.start() < start or _words(t) <= note_words:
            return m.group(0)
        lead = re.match(r"\s*<strong>(.*?)</strong>\s*", t)
        if lead:
            return f'<details class="note"><summary>{lead.group(1).rstrip(".:")}</summary><p class="small">{t[lead.end():]}</p></details>'
        for b in re.finditer(r"(?<=[a-z0-9)%”])\.\s+(?=[A-Z“])", t):          # first sentence stays: it is often the table's legend
            if t.count("<", 0, b.start()) == t.count(">", 0, b.start()) and _words(t[:b.start()]) >= 6:
                return (f'<p class="small">{t[:b.start() + 1]}</p><details class="note"><summary>Notes and method</summary>'
                        f'<p class="small">{t[b.end():]}</p></details>')
        return m.group(0)

    def caption(m):
        t = m.group(1)
        if _words(t) <= caption_words:
            return m.group(0)
        for b in re.finditer(r"(?<=[a-z0-9)%”])\.\s+(?=[A-Z“])", t):          # a sentence end outside tags, after enough words
            if (t.count("<", 0, b.start()) == t.count(">", 0, b.start()) and _words(t[:b.start()]) >= lead_words
                    and not re.match(r"(Bottom|Right|Middle|Lower)\b", t[b.end():])):          # keep a figure's panels together
                return (f'<figcaption>{t[:b.start() + 1]} <details class="note"><summary>More about this figure</summary>'
                        f'<span>{t[b.end():]}</span></details></figcaption>')
        return m.group(0)

    page = re.sub(r"<p>((?:(?!</p>).)*?)\s*<!--more-->\s*(.*?)</p>", lambda m: f'<p>{m.group(1)}</p><details class="note"><summary>More detail</summary>'
                  f'<p>{m.group(2)}</p></details>', page, flags=re.S)             # a paragraph's detail, marked where it is written
    page = re.sub(r'<p class="small">(.*?)</p>', note, page, flags=re.S)
    return re.sub(r"<figcaption>(.*?)</figcaption>", caption, page, flags=re.S)


def _anchors(page: str) -> str:
    """Give every heading an id and a link to itself, and list a part's subsections under its heading when it
    has three or more. Ids are nested: a subsection's id starts with the id of the area heading (or, outside
    the area blocks, the section heading) it sits under, which also keeps repeated titles apart."""
    hs = [dict(tag=m.group(1), attrs=m.group(2), inner=m.group(3), m=m) for m in HEADING.finditer(page)]
    used = set(re.findall(r'\bid="([^"]+)"', page))
    sec = area = None
    for h in hs:
        got = re.search(r'\bid="([^"]+)"', h["attrs"])
        if h["tag"] == "h2":
            sec, area, parent = h, None, None
        elif 'class="area"' in h["attrs"]:
            area, parent = h, None
        else:
            parent = area or sec
            top = "h4" if area else "h3"                 # the first level of headings under an area or section heading
            if parent is not None and h["tag"] == top:
                parent.setdefault("kids", []).append(h)
        if got:
            h["id"] = got.group(1)
        else:
            base = "-".join(x for x in (parent["id"] if parent else "", _slug(h["inner"])) if x)
            hid, k = base, 2
            while hid in used:
                hid, k = f"{base}-{k}", k + 1
            used.add(hid); h["id"] = hid
    out, pos = [], 0
    for h in hs:
        m = h["m"]
        attrs = h["attrs"] if 'id="' in h["attrs"] else f'{h["attrs"]} id="{h["id"]}"'
        out.append(page[pos:m.start()])
        out.append(f'<{h["tag"]}{attrs}>{h["inner"]}<a class="anchor" href="#{h["id"]}" aria-hidden="true" tabindex="-1"></a></{h["tag"]}>')
        pos = m.end()
        if len(h.get("kids", [])) >= 3:
            links = " ".join(f'<span><a href="#{k["id"]}">{re.sub(r"<[^>]+>", "", k["inner"])}</a>{"" if k is h["kids"][-1] else " ·"}</span>'
                             for k in h["kids"])
            lst = f'<p class="inpart">In this {"part" if "area" in h["attrs"] else "section"}: {links}</p>'
            if page[:m.start()].endswith('<div class="areahead">'):                  # after the heading row, not inside it
                end = page.index("</div>", pos) + 6
                out.append(page[pos:end]); pos = end
            out.append(lst)
    out.append(page[pos:])
    page = "".join(out)
    ids = re.findall(r'\bid="([^"]+)"', page)
    dup = {i for i in ids if ids.count(i) > 1}
    dead = {h for h in re.findall(r'href="#([^"]+)"', page) if h not in set(ids)}
    if dup or dead:
        raise ValueError(f"navigation: duplicate ids {sorted(dup)}, links to nothing {sorted(dead)}")
    return page


def _demote(chunks: list[str]) -> str:
    """Headings inside an area's block, nested under the area heading: h4 → h5, and h3 and the SEAS5 block's own h2 → h4."""
    t = "\n".join(chunks)
    t = re.sub(r"<(/?)h4\b", r"<\1h5", t)
    return re.sub(r"<(/?)h[23]\b", r"<\1h4", t)


def _per_cent(page: str) -> str:
    """OCHA writes "per cent" in running text and keeps "%" for tables and graphics: convert the text nodes of
    plain paragraphs and list items; tables, figure captions and small notes are left alone."""
    def block(m):
        return m.group(1) + re.sub(r"(<[^>]+>)|(\d)%", lambda x: x.group(1) or x.group(2) + " per cent", m.group(2)) + m.group(3)
    page = re.sub(r"(<p>)(.*?)(</p>)", block, page, flags=re.S)
    return re.sub(r"(<li>)(.*?)(</li>)", block, page, flags=re.S)


def _prefix_src(t: str, sub: str) -> str:
    """Point an area block's relative figure paths at its subfolder."""
    return re.sub(r'(src=")(?![a-z]+:|/|\.\./)', rf"\g<1>{sub}/", t)


def build_combined(cspec: dict, grid: edd.Grid, ne: gpd.GeoDataFrame, indices: pd.DataFrame) -> list[dict]:
    """One page for several areas (builder = "levant-combined"): each part in `parts` is a TOML under
    deep_dives/parts/, analysed as its own area with figures in docs/enso/<slug>/<part>/, then the pages are
    interleaved section by section. The combined TOML supplies the summary, the shared forecast text and
    table, the regional-maps text, the caveats and the verdict for the index card."""
    edd.END_YEAR = int(grid.years[-1])
    latest = ts.CONFIG["cache_dir"] / "nino34_latest.data"
    edd.NINO_LATEST = ts._parse_psl(latest.read_text()) if latest.exists() else None
    shared = {k: cspec[k] for k in ("forecast_html", "forecasts", "forecast_table_note") if k in cspec}
    parts = []
    for slug in cspec["parts"]:
        pspec = tomllib.loads((PARTS_DIR / f"{slug}.toml").read_text()) | {"slug": slug} | shared | {
            "out_dir": str(edd.OUT_DIR / cspec["slug"] / slug), "map_outline": cspec.get("map_outline"),
            "map_label": cspec["name"]}
        set_area(pspec)
        print(f"  part: {A.name}", flush=True)
        a = analyse(pspec, grid, ne, indices)
        parts.append(dict(slug=slug, spec=pspec, a=a, area=A, B=render_blocks(pspec, a, headings=False)))
    out = edd.OUT_DIR / cspec["slug"]
    if all(p["a"].get("cold_enso") for p in parts):
        fig_cold_enso(parts, out / "cold_enso.png")
    (out / "index.html").write_text(render_combined(cspec, parts), encoding="utf-8")
    for p in parts[1:]:                                      # the regional maps are shown once, from the first part
        for f in ("corr_maps.png", "composite_maps.png"):
            (out / p["slug"] / f).unlink(missing_ok=True)
    for old in cspec.get("redirect_from", []):               # the areas' former standalone pages
        d = edd.OUT_DIR / old
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Moved: {html.escape(cspec["name"])}</title>'
            f'<meta http-equiv="refresh" content="0; url=../{cspec["slug"]}/"><link rel="canonical" href="../{cspec["slug"]}/"></head>'
            f'<body><p>This page is now part of <a href="../{cspec["slug"]}/">{html.escape(cspec["name"])}</a>.</p></body></html>\n',
            encoding="utf-8")
    return parts


def render_combined(cspec: dict, parts: list[dict]) -> str:
    ours, cat = cspec["assessment"], cspec["catalogue"]
    T = lambda k, d: html.escape(cspec.get("titles", {}).get(k, d))
    p0 = parts[0]
    o = [edd.HEAD.format(title=f'{cspec["name"]} — ENSO deep dive', desc=html.escape(ours["one_line"]), css=edd.CSS + COMBINED_CSS + NAV_CSS,
                         home="../", home_label="ENSO country deep dives")]
    # what each section holds, known before anything is written: the bar, the contents block and the links between areas use it
    names = {p["slug"]: p["area"].name for p in parts}
    sec_keys = ["forecast", "enso", "daily", "impacts", "range"]
    has = {k: [p for p in parts if any(c.strip() for c in p["B"][{"forecast": "forecast_area"}.get(k, k)])] for k in sec_keys}
    sibs = {k: [(f'{p["slug"]}-{k}', names[p["slug"]]) for p in has[k]] + ([("region-enso", "Both areas")] if k == "enso" else [])
               + ([("cold-daily", "Cold snaps")] if k == "daily" and all(p["a"].get("cold_enso") for p in parts) else []) for k in sec_keys}
    agri = [p for p in parts if p["B"]["agri"]]
    after = [(_slug(sec["title"]), sec["title"]) for sec in cspec.get("sections_after", [])]
    toc = ([(k, cspec.get("titles", {}).get(k, k), sibs[k]) for k in sec_keys] + ([("farming", cspec.get("titles", {}).get("agri", "6. Farming"), [])] if agri else [])
           + [(i, t, []) for i, t in after] + [("references", "References", [])])
    short = cspec.get("nav", {})
    sec_no = {i: n for n, (i, _, _) in enumerate(toc, 1) if n <= len(sec_keys) + bool(agri)}

    targets = {k: (sec_no[k], None) for k in sec_no} | {a: (sec_no[k], n) for k in sec_keys for a, n in sibs[k]}

    def see(ids: list[str]) -> str:
        """Links from a key message to the sections that carry its evidence (a wrong anchor is a KeyError naming it)."""
        def one(i):
            n, area = targets[i]
            return f'<a href="#{i}">section {n}' + (f', {html.escape(area)}' if area else '') + '</a>'
        return ' <span class="see">See ' + " and ".join(one(i) for i in ids) + '.</span>' if ids else ''

    o.append(f'<p class="eyebrow">{html.escape(cspec.get("eyebrow", "ENSO country deep dive"))}</p><h1>{html.escape(cspec["name"])}</h1>')
    o.append(f'<p class="meta">{html.escape(cspec.get("subtitle", ""))} &nbsp;·&nbsp; GPCC 1891–2025, ERA5 1950–{p0["a"]["end_era5"]:%Y}, '
             f'IMERG 1998–{p0["a"]["end_imerg"]:%Y}, ' + ", ".join(f'{p["area"].meta_gauge} ({p["area"].name})' for p in parts)
             + ' &nbsp;·&nbsp; Niño3.4 (NOAA)</p>')
    item = lambda i, label, sub=(): ('<span class="item">' + f'<a href="#{i}">{label}</a>' + ('<span class="sub">' + "".join(
        f'<a href="#{a}">{html.escape(n)}</a>' for a, n in sub) + '</span>' if sub else '') + '</span>')
    o.append('<nav class="secbar" aria-label="Sections of this page">'
             + (item("key-messages", "Key messages") if cspec.get("key_messages") else '') + item("contents", "Contents")
             + "".join(item(i, html.escape(short.get(i, t)) if i not in sec_no else f'{sec_no[i]}&nbsp;{html.escape(short.get(i, t.split(". ", 1)[-1]))}', sub)
                       for i, t, sub in toc) + '</nav>')
    # key messages for the coming winter, ahead of everything else
    if cspec.get("key_messages"):
        o.append(f'<div class="keymsg" id="key-messages"><h2 id="key-messages-heading">{html.escape(cspec.get("key_messages_title", "Key messages"))}</h2>'
                 + (f'<p class="small">{html.escape(cspec["key_messages_dateline"])}</p>' if cspec.get("key_messages_dateline") else "")
                 + "<ol>" + "".join(f'<li><strong>{html.escape(m["lead"])}</strong> {html.escape(m["text"])}{see(m.get("see", []))}</li>'
                                    for m in cspec["key_messages"]) + "</ol>"
                 + (f'<p class="small">{html.escape(cspec["key_messages_footer"])}</p>' if cspec.get("key_messages_footer") else "") + '</div>')
    # verdict: the survey's catalogue row once, then this review's grade for each area
    o.append('<div class="verdict">')
    o.append(f'<div class="card"><p class="lbl">Survey catalogue says</p><p class="big">El Niño → {html.escape(cat["direction"])}, {html.escape(cat["season"])}</p>'
             f'<p>{edd.chip(cat["evidence"])} <span class="small">source: {cat["source_html"]}</span></p></div>')
    for p in parts:
        po = p["spec"]["assessment"]
        o.append(f'<div class="card"><p class="lbl">This review assesses: {html.escape(p["area"].name)}</p><p class="big">El Niño → '
                 f'{html.escape(po["direction"])}, {html.escape(po["season"])}</p>'
                 f'<p>{edd.chip(po["evidence"])} <span class="small">{html.escape(po["evidence_note"])}</span></p></div>')
    o.append('</div>')
    by_no = {n: i for i, n in sec_no.items()}           # the summary's numbered items link to their sections
    summary = re.sub(r"<p><strong>(\d+)\. ([^<]*)</strong>", lambda m: (f'<p><strong><a href="#{by_no[int(m.group(1))]}">{m.group(1)}. {m.group(2)}</a></strong>'
                                                                       if int(m.group(1)) in by_no else m.group(0)), cspec["summary_html"])
    if cspec.get("key_messages"):       # the section-by-section summary folds away once the key messages lead the page
        o.append(f'<details class="summary"><summary>Summary by section</summary>{summary}</details>')
    else:
        o.append(f'<div class="summary">{summary}</div>')
    o.append('<nav class="contents" id="contents" aria-labelledby="contents-label"><p class="lbl" id="contents-label">Contents</p><ul>' + "".join(
        f'<li><a href="#{i}">{html.escape(t)}</a>' + ('<span class="areas">' + "".join(f'<a href="#{a}">{html.escape(n)}</a>' for a, n in sub) + '</span>' if sub else '')
        + '</li>' for i, t, sub in toc) + '</ul><p class="foldall" hidden><button type="button"></button></p></nav>')
    o.append(cspec.get("before_html", ""))

    def area_head(anchor: str, label: str, key: str) -> str:
        """An area's heading within a section, with links to the section's other areas beside it."""
        alt = " · ".join(f'<a href="#{a}">{html.escape(n)}</a>' for a, n in sibs[key] if a != anchor)
        return (f'<div class="areahead"><h3 class="area" id="{anchor}">{html.escape(label)}</h3>'
                + (f'<p class="alt">Also in this section: {alt}</p>' if alt else '') + '</div>')

    def area_block(p: dict, chunks: list[str], anchor: str) -> str:
        return area_head(f'{p["slug"]}-{anchor}', p["area"].name, anchor) + '\n' + _prefix_src(_demote(chunks), p["slug"])

    # 1. Forecasts: shared text and table once (built with the first area), then each area's SEAS5
    o.append(f'<h2 id="forecast">{T("forecast", "1. What the forecasts say for this winter")}</h2>')
    o.extend(p0["B"]["forecast_lead"] + p0["B"]["forecast_text"])
    for p in parts:
        o.append(area_block(p, p["B"]["forecast_area"], "forecast"))

    # 2. El Niño link: each area, then the regional maps once (both areas outlined)
    o.append(f'<h2 id="enso">{T("enso", "2. How much El Niño matters")}</h2>')
    if cspec.get("enso_intro_html"):
        o.append(cspec["enso_intro_html"])
    for p in parts:
        ch = p["B"]["enso"]
        cut = next(i for i, c in enumerate(ch) if c.startswith("<h3>Across the region</h3>"))
        p["maps"] = ch[cut + 1:]
        o.append(area_block(p, ch[:cut], "enso"))
    o.append(area_head("region-enso", "Both areas: across the region", "enso") + cspec.get("maps_html", ""))
    o.append(_prefix_src(p0["maps"][0], p0["slug"]).replace(f'{p0["area"].ref[0].upper() + p0["area"].ref[1:]} is outlined.',
                                                             f'{html.escape(cspec["name"])} are outlined.'))
    a0 = p0["a"]
    o.append(f'<figure><img src="{p0["slug"]}/composite_maps.png" alt="El Niño composite and wettest-third hit rate"><figcaption>Left: mean '
             f'standardized October–April anomaly over the {a0["n_en_grid"]} El Niño winters of {_yr(a0["n_grid"][0])}–{_yr(a0["n_grid"][1])} '
             f'(positive in {100 * a0["comp_region_pos"]:.0f}% of analysed cells). Right: the share of those winters in each cell\'s wettest '
             f'third (regional median {100 * a0["hit_region"]:.0f}%; chance is 33%). '
             + "Cells over " + "; over ".join(f'{p["area"].ref}: {p["a"]["comp_gaza"]:+.2f} SD and {100 * p["a"]["hit_gaza"]:.0f}% in the wettest third' for p in parts)
             + '.</figcaption></figure>')

    # 3–5. Each area in turn
    for key, title, blocks in (("daily", "3. What changes in an El Niño winter, and what does not", ("daily",)),
                               ("impacts", "4. What winter weather does", ("impacts",)),
                               ("range", "5. What this winter could bring", ("range",))):
        o.append(f'<h2 id="{key}">{T(key, title)}</h2>')
        for p in has[key]:
            o.append(area_block(p, sum((p["B"][b] for b in blocks), []), key))
        if key == "daily" and ("cold-daily", "Cold snaps") in sibs[key]:
            cs = [p["a"]["cold_enso"] for p in parts]
            rs = [c[k] for c in cs for k in ("r_coldest", "r_snaps")] + [r for c in cs for r in c["r_other"].values()]
            o.append(area_head("cold-daily", "Both areas: cold snaps and El Niño", key))
            o.append('<p>Cold snaps show no link to El Niño in either area. Since ' + f'{cs[0]["lo"]}' + ' the number of cold nights in a winter does not follow winter '
                     'Niño3.4: ' + "; ".join(f'{html.escape(p["area"].name)}, nights of {c["thr"]:g} °C or colder, r = {c["r_nights"]:+.2f} ({_pv(c["p_nights"])})'
                                           for p, c in zip(parts, cs))
                     + f'. Nor do the winter\'s coldest night, the number of cold snaps of two nights or more, or the night count two degrees either side '
                     f'of each threshold (correlations between {min(rs):+.2f} and {max(rs):+.2f}). ' + cspec.get("cold_html", "") + '</p>')
            o.append('<figure><img src="cold_enso.png" alt="Cold nights per winter against winter Niño3.4, Gaza and the West Bank"><figcaption>Each dot is one '
                     f'winter, {_yr(cs[0]["lo"])}–{_yr(cs[0]["hi"])}: nights whose ERA5 minimum over the area was at or below the threshold, against that '
                     'winter\'s Niño3.4. Lines: the mean of each phase (' + "; ".join(
                         f'{html.escape(p["area"].name)} {c["mean"]["El Niño"]:.1f} in El Niño winters, {c["mean"]["Neutral"]:.1f} neutral, '
                         f'{c["mean"]["La Niña"]:.1f} La Niña' for p, c in zip(parts, cs)) + '). ERA5 is an area mean and runs warmer than the coldest places.'
                     '</figcaption></figure>')
    # 6. Farming: the areas that have it, under their own headings
    for p in parts:
        if p["B"]["agri"]:
            o.append(f'<h2 id="farming">{T("agri", "6. Farming")}</h2>')
            o.append(_prefix_src("\n".join(p["B"]["agri"]), p["slug"]))
    for sec in cspec.get("sections_after", []):
        o.append(f'<h2 id="{_slug(sec["title"])}">{html.escape(sec["title"])}</h2>{sec["html"]}')
    refs, seen = [], set()
    for r in [r for p in parts for r in p["spec"].get("references", [])] + cspec.get("references", []):
        if r["html"] not in seen:
            seen.add(r["html"]); refs.append(r)
    o.append('<h2 id="references">References</h2><ul class="refs">' + "".join(f'<li>{r["html"]}</li>' for r in refs) + '</ul>')
    o.append(f'<p class="small">Generated by <code>levant_deep_dive.py</code> from <code>deep_dives/{cspec["slug"]}.toml</code> and '
             + ", ".join(f'<code>deep_dives/parts/{p["slug"]}.toml</code>' for p in parts)
             + '. Grid method and Niño3.4 series as in the <a href="../../survey/">global survey</a>.</p>')
    o.append(NAV_JS + edd.FOOT)
    return _anchors(_scroll_tables(_fold(_per_cent("\n".join(o)))))


def main() -> None:
    import sys
    slug = sys.argv[1] if len(sys.argv) > 1 else "gaza-west-bank"
    cfg = dict(ts.CONFIG, max_lag=3)
    path = edd.DEEP_DIR / f"{slug}.toml"
    if not path.exists():
        sys.exit(f"{slug}: no deep_dives/{slug}.toml" + (" (it is a part of a combined page: build that page, e.g. gaza-west-bank)"
                                                        if (PARTS_DIR / f"{slug}.toml").exists() else ""))
    spec = tomllib.loads(path.read_text()) | {"slug": slug}
    edd.extend_pixel_cache(cfg)
    grid = edd.load_grid(cfg)
    if spec.get("builder") == "levant-combined":
        build_combined(spec, grid, ts.load_admin0_gdf(cfg), ts.load_indices(cfg))
        print(f"wrote {edd.OUT_DIR / slug}/index.html")
    else:
        build(spec, grid, ts.load_admin0_gdf(cfg), ts.load_indices(cfg))
        print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
