"""
Global teleconnection survey: max-correlation maps per climate index.

For each country x index:
  - 12 rolling trimesters (NDJ, DJF, JFM, … OND), year-labeled by first month
    (DJF/NDJ label by December/November year; all others by their own year)
  - sweep lags up to a max (3 or 6 months, in-app toggle; index leads rainfall)
  - per trimester, keep lag with max |r|  (this freezes a best-lag per index/tri)
  - filter to p < 0.05
  - if significant corrs exist in both signs -> hatch (bidirectional)
  - else -> show strongest, colored by r

Two passes, both off the SAME frozen best-lags:
  - TOTAL    : pairwise Pearson r (direct + shared-with-other-modes)
  - PARTIAL  : r unique to the index, holding the other indices constant
               (residuals method; controls = other indices at their own best lag)
Divergence total->partial = signal is shared among modes, not absent.

Output: two choropleths per index (total + partial), ENSO composites, HTML report.

Data: ERA5 monthly country-level precip from team DB via ocha-stratus.
"""

from __future__ import annotations
import io
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import ocha_stratus as stratus

# Brown (drought) → neutral → blue (flood), matching ds-seas5-skill
DROUGHT_FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "drought_flood",
    [
        (0.00, "#7B3A1A"),
        (0.35, "#C8844A"),
        (0.50, "#F5F0EC"),
        (0.65, "#71B3E5"),
        (1.00, "#0D40B0"),
    ],
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CONFIG = {
    "cache_dir": Path("cache"),
    "out_dir": Path("docs/maps"),
    "parquet_dir": Path("out"),
    "docs_dir": Path("docs"),
    "start_year": 1981,
    "end_year": 2025,
    "max_lag_months": 6,
    "min_years": 25,
    "alpha": 0.05,
    "partial_exclude": ["pdo"],
}

# Lag-cap variants generated for the in-app toggle. Tag -> max lag (months).
# Default view is the tighter 3-month cap (index leads rainfall by at most one
# full non-overlapping season); 6-month is available for the long-lag view.
LAG_CAPS = {"l3": 3, "l6": 6}
DEFAULT_LAG_TAG = "l3"

# Spatial-resolution variants generated for the in-app toggle. The country pass
# uses DB-side ERA5 admin-0 raster stats; the pixel pass runs the identical
# method on every ERA5 land cell in the map viewport, read from the ERA5 monthly
# COGs on blob. Tag -> label.
RES_VARIANTS = {"adm0": "Country (ADM0)", "px": "Pixel (0.25°)"}
DEFAULT_RES_TAG = "adm0"

# Pixel pass: source COGs, filters and render settings.
PIXEL_BLOB_PREFIX = "era5/monthly/processed/"
_PIXEL_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
PIXEL_DOWNLOAD_WORKERS = 8
PIXEL_CHUNK = 40_000          # cells per batched-linear-algebra block
PIXEL_MIN_TRI_MM_DAY = 0.25   # hyper-arid cut for the pixel pass (see docstring)
PIXEL_DOMINANT_MARGIN = 0.10  # |r| gap the top mode must beat the runner-up by
PIXEL_MAP_W = 15.0            # inches; matches the country maps
PIXEL_MAP_DPI = 150           # ~2x the native 0.25 deg grid across the viewport

# All 12 rolling 3-month windows: name -> end month.
# Year label by the first month of the window:
#   NDJ (Nov-Dec-Jan): first=Nov, end=Jan → offset -1 (Jan_year - 1 = Nov_year)
#   DJF (Dec-Jan-Feb): first=Dec, end=Feb → offset -1
#   All others: first and end month in same calendar year → offset 0
TRIMESTERS = {
    "NDJ": 1, "DJF": 2, "JFM": 3, "FMA": 4,
    "MAM": 5, "AMJ": 6, "MJJ": 7, "JJA": 8,
    "JAS": 9, "ASO": 10, "SON": 11, "OND": 12,
}
TRIMESTER_YEAR_OFFSET = {
    "NDJ": -1, "DJF": -1,
    "JFM": 0, "FMA": 0, "MAM": 0, "AMJ": 0,
    "MJJ": 0, "JJA": 0, "JAS": 0, "ASO": 0, "SON": 0, "OND": 0,
}
# Non-overlapping canonical set — used only for annual-total normalisation
_ANNUAL_TRIMESTERS = ("DJF", "MAM", "JAS", "OND")

INDEX_SOURCES = {
    "nino34": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
    "dmi":    "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "tna":    "https://psl.noaa.gov/data/correlation/tna.data",
    "tsa":    "https://psl.noaa.gov/data/correlation/tsa.data",
    "amm":    "https://psl.noaa.gov/data/correlation/amm.data",
    "pdo":    "https://psl.noaa.gov/data/correlation/pdo.data",
}

INDEX_LABELS = {
    "nino34": "Niño3.4 (ENSO)",
    "dmi":    "IOD (Indian Ocean Dipole)",
    "tna":    "TNA (Tropical N. Atlantic)",
    "tsa":    "TSA (Tropical S. Atlantic)",
    "amm":    "AMM (Atlantic Meridional Mode)",
    "pdo":    "PDO (Pacific Decadal Oscillation)",
}

NATURALEARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

# Threshold reference lines shown on the time series panel.
# For ENSO this is the standard ±0.5 °C ONI threshold.
# For other indices ±0.4 is a reasonable "notable event" reference.
INDEX_THRESHOLDS = {
    "nino34": 0.5,
    "dmi":    0.4,
    "tna":    0.4,
    "tsa":    0.4,
    "amm":    0.5,
    "pdo":    0.5,
}

# Map viewport — matches ds-seas5-skill global view
MAP_XLIM = (-100, 180)
MAP_YLIM = (-36, 56)
_MAP_DX = MAP_XLIM[1] - MAP_XLIM[0]   # 280°
_MAP_DY = MAP_YLIM[1] - MAP_YLIM[0]   # 92°
_DOT_R = 0.0035 * _MAP_DX             # ≈ 0.98° — same formula as ds-seas5-skill


# Discrete 2-bin correlation colors (matching ds-seas5-skill palette)
_R_STRONG  = 0.45   # |r| threshold for dark vs light bin
_R_MIN     = 0.30   # |r| below this → gray (effectively unshown)
_R_LEGEND  = [
    ("#0D40B0", "#092E88", f"Positive, strong (r ≥ {_R_STRONG})"),
    ("#71B3E5", "#4A90C8", f"Positive, moderate (r {_R_MIN}–{_R_STRONG})"),
    ("#C8844A", "#A06030", f"Negative, moderate (r −{_R_MIN}–−{_R_STRONG})"),
    ("#7B3A1A", "#5A2A0A", f"Negative, strong (r ≤ −{_R_STRONG})"),
]


def _r_colors(r: float) -> tuple[str, str]:
    """Map correlation value to (facecolor, edgecolor) from the discrete 2-bin scheme."""
    if not np.isfinite(r):
        return "#e8e8e8", "#999"
    if r >=  _R_STRONG: return "#0D40B0", "#092E88"
    if r >=  _R_MIN:    return "#71B3E5", "#4A90C8"
    if r <= -_R_STRONG: return "#7B3A1A", "#5A2A0A"
    if r <= -_R_MIN:    return "#C8844A", "#A06030"
    return "#e8e8e8", "#999"


# Time series panel colors: red/green warm-cool palette, clearly distinct from the
# brown/blue drought-flood map palette.
_TS_STRONG_WARM  = "#B2182B"   # dark red
_TS_WEAK_WARM    = "#F4A582"   # light salmon
_TS_WEAK_COOL    = "#A6DBA0"   # light green
_TS_STRONG_COOL  = "#1B7837"   # dark green

# Categorical colors for the dominant-index map (muted pastel variants)
INDEX_COLORS = {
    "nino34": "#F08080",
    "dmi":    "#FFBB77",
    "tna":    "#88CC88",
    "tsa":    "#C49AC9",
    "amm":    "#88AEDD",
    "pdo":    "#C9956A",
}

def _ts_color(val: float, thresh: float) -> str:
    """Discrete fill color for a raw index value relative to its threshold."""
    if val >=  thresh: return _TS_STRONG_WARM
    if val >   0:      return _TS_WEAK_WARM
    if val >  -thresh: return _TS_WEAK_COOL
    return _TS_STRONG_COOL


def _draw_ts_panel(ax, series: pd.Series, index_name: str) -> None:
    """Draw the past-10-year time series of a climate index on ax."""
    thresh = INDEX_THRESHOLDS.get(index_name, 0.5)
    s = series.dropna()
    if s.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        return

    today  = pd.Timestamp.today().normalize()
    cutoff = pd.Timestamp("2010-01-01")
    recent = s[s.index >= cutoff]

    # Filled areas by threshold band
    kw = dict(interpolate=True, linewidth=0)
    ax.fill_between(recent.index, thresh, recent,
                     where=recent >= thresh,  color=_TS_STRONG_WARM, alpha=0.85, **kw)
    ax.fill_between(recent.index, 0, recent,
                     where=(recent >= 0) & (recent < thresh), color=_TS_WEAK_WARM, alpha=0.60, **kw)
    ax.fill_between(recent.index, 0, recent,
                     where=(recent < 0) & (recent > -thresh), color=_TS_WEAK_COOL, alpha=0.60, **kw)
    ax.fill_between(recent.index, -thresh, recent,
                     where=recent <= -thresh, color=_TS_STRONG_COOL, alpha=0.85, **kw)

    # Thin line on top
    ax.plot(recent.index, recent, color="#333", linewidth=0.7, zorder=3)

    # Reference lines
    ax.axhline(0,       color="#888", linewidth=0.5, zorder=2)
    ax.axhline( thresh, color=_TS_STRONG_WARM, linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(-thresh, color=_TS_STRONG_COOL,  linewidth=0.8, linestyle="--", zorder=2)

    # Threshold labels on right margin
    ax.text(1.01,  thresh, f"+{thresh}", transform=ax.get_yaxis_transform(),
            fontsize=10, va="center", color=_TS_STRONG_WARM)
    ax.text(1.01, -thresh, f"−{thresh}", transform=ax.get_yaxis_transform(),
            fontsize=10, va="center", color=_TS_STRONG_COOL)

    # Current value marker + label
    cur_val  = recent.iloc[-1]
    cur_date = recent.index[-1]
    ax.scatter([cur_date], [cur_val], s=30, zorder=5,
                color=_ts_color(cur_val, thresh), edgecolors="#333", linewidths=0.5)
    ax.annotate(f" {cur_val:+.2f}",
                xy=(cur_date, cur_val), xytext=(3, 0),
                textcoords="offset points", fontsize=11,
                color=_ts_color(cur_val, thresh), fontweight="bold", va="center")

    # X-axis: one label per year, angled; extend right edge to today
    years = sorted({d.year for d in recent.index} | {today.year})
    ax.set_xticks([pd.Timestamp(y, 7, 1) for y in years])
    ax.set_xticklabels([str(y) for y in years], rotation=60, ha="right", fontsize=10)

    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Index value", fontsize=10, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(cutoff, today)

    # Latest-data date label
    latest = recent.dropna().index[-1] if recent.dropna().size else None
    if latest is not None:
        ax.text(0.98, 0.97, f"Latest: {latest.strftime('%b %Y')}",
                transform=ax.transAxes, fontsize=10, color="#666",
                ha="right", va="top")


def _is_dot_country(geom) -> bool:
    """True for small islands / archipelagos that should be shown as dots."""
    if geom is None:
        return False
    if geom.area < 0.5:
        return True
    max_p = max(p.area for p in geom.geoms) if hasattr(geom, "geoms") else geom.area
    compact = 4 * np.pi * geom.area / (geom.length ** 2) if geom.length > 0 else 1.0
    return max_p < 1.5 and compact < 0.4


def _diagonal_halves(geom):
    """Split geometry into (upper_right, lower_left) halves via the \\-diagonal
    of its bounding box.  Each half covers exactly half the bbox area, giving a
    visually balanced 50/50 split for bidirectional countries."""
    from shapely.geometry import Polygon as _SPoly
    minx, miny, maxx, maxy = geom.bounds
    pad = max(maxx - minx, maxy - miny) + 2
    upper_right = _SPoly([(minx-pad, maxy+pad), (maxx+pad, maxy+pad), (maxx+pad, miny-pad)])
    lower_left  = _SPoly([(minx-pad, maxy+pad), (maxx+pad, miny-pad), (minx-pad, miny-pad)])
    return geom.intersection(upper_right), geom.intersection(lower_left)


def _dot_center(geom) -> tuple[float, float]:
    """Representative point of the largest sub-polygon; wraps antimeridian islands."""
    largest = max(geom.geoms, key=lambda p: p.area) if hasattr(geom, "geoms") else geom
    rp = largest.representative_point()
    cx, cy = rp.x, rp.y
    if cx < MAP_XLIM[0]:
        cx_wrap = cx + 360
        cx = cx_wrap if cx_wrap <= MAP_XLIM[1] else MAP_XLIM[1] - _DOT_R * 1.5
    return cx, cy


# --------------------------------------------------------------------------- #
# Index loading
# --------------------------------------------------------------------------- #
def _parse_psl(text: str) -> pd.Series:
    """Parse NOAA PSL fixed-format monthly series -> Series indexed by month-end."""
    lines = text.splitlines()
    yr0, yr1 = (int(x) for x in lines[0].split()[:2])
    recs = {}
    missing = None
    for ln in lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        if len(parts) == 1:
            try:
                missing = float(parts[0])
            except ValueError:
                pass
            continue
        try:
            yr = int(parts[0])
        except ValueError:
            continue
        if yr < yr0 or yr > yr1:
            continue
        vals = [float(v) for v in parts[1:13]]
        for m, v in enumerate(vals, start=1):
            recs[pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0)] = v
    s = pd.Series(recs).sort_index()
    if missing is not None:
        s = s.where(s != missing)
    return s.where(s > -900)


def load_indices(cfg: dict) -> pd.DataFrame:
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in INDEX_SOURCES.items():
        cache = cfg["cache_dir"] / f"{name}.data"
        if cache.exists():
            text = cache.read_text()
        else:
            text = requests.get(url, timeout=60).text
            cache.write_text(text)
        out[name] = _parse_psl(text)
    df = pd.DataFrame(out)
    # No end-year cap: full series needed for the time series panel.
    # The correlation analysis is naturally bounded by the rainfall years (via index intersection).
    return df.loc[f"{cfg['start_year']}":]


# --------------------------------------------------------------------------- #
# Admin0 boundaries (Natural Earth 110m, cached)
# --------------------------------------------------------------------------- #
def load_admin0_gdf(cfg: dict) -> gpd.GeoDataFrame:
    """Download Natural Earth 110m countries once, cache to cache/. Returns
    GeoDataFrame with a lowercase 'iso3' column."""
    cache = cfg["cache_dir"] / "naturalearth_admin0.geojson"
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print("Downloading Natural Earth admin0 boundaries...")
        r = requests.get(NATURALEARTH_URL, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
    gdf = gpd.read_file(cache).to_crs("EPSG:4326")
    # ISO_A3_EH has codes for Kosovo/Taiwan/etc.; fall back to ISO_A3
    if "ISO_A3_EH" in gdf.columns:
        gdf["iso3"] = gdf["ISO_A3_EH"].where(
            gdf["ISO_A3_EH"] != "-99", gdf.get("ISO_A3", "-99")
        ).str.upper()
    else:
        gdf["iso3"] = gdf["ISO_A3"].str.upper()
    # Remap unrecognized territories (-99) to their de facto parent state
    # so they display the parent's signal rather than appearing as gray holes.
    _REMAP = {"Somaliland": "SOM"}
    if "NAME" in gdf.columns:
        mask = gdf["iso3"] == "-99"
        gdf.loc[mask, "iso3"] = gdf.loc[mask, "NAME"].map(_REMAP).fillna("-99")
    # Drop any remaining unrecognized entries (Kosovo, N. Cyprus, etc.)
    gdf = gdf[gdf["iso3"] != "-99"]
    # Dissolve duplicate iso3 rows (e.g. Somalia + remapped Somaliland → single SOM polygon)
    gdf = gdf[["iso3", "geometry"]].dissolve(by="iso3").reset_index()
    return gdf


# --------------------------------------------------------------------------- #
# Rainfall -> country trimester series (ERA5 from DB)
# --------------------------------------------------------------------------- #
_TRIMESTER_MONTHS = {
    "NDJ": [11, 12,  1],
    "DJF": [12,  1,  2],
    "JFM": [ 1,  2,  3],
    "FMA": [ 2,  3,  4],
    "MAM": [ 3,  4,  5],
    "AMJ": [ 4,  5,  6],
    "MJJ": [ 5,  6,  7],
    "JJA": [ 6,  7,  8],
    "JAS": [ 7,  8,  9],
    "ASO": [ 8,  9, 10],
    "SON": [ 9, 10, 11],
    "OND": [10, 11, 12],
}


def country_trimester_rainfall_era5(cfg: dict) -> pd.DataFrame:
    """Query ERA5 monthly country-level precip from DB; return season_year-indexed
    DataFrame with MultiIndex columns (iso3, trimester).

    DB values are mm/day (monthly means). Trimester = mean of the 3 monthly means.
    adm_level=0 = whole-country polygon.
    DJF season_year = December's year (Dec 2024 / Jan-Feb 2025 → 2024).
    Only complete seasons (all 3 months present) are kept.
    """
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT iso3, valid_date, mean FROM public.era5 "
            "WHERE adm_level = 0 "
            "AND valid_date >= %(start)s AND valid_date <= %(end)s",
            conn,
            params={
                "start": f"{cfg['start_year']}-01-01",
                "end": f"{cfg['end_year']}-12-31",
            },
            parse_dates=["valid_date"],
        )
    df["iso3"] = df["iso3"].str.upper()

    results = {}
    for tri, months in _TRIMESTER_MONTHS.items():
        is_wrapping = 1 in months and 12 in months
        df_tri = df[df["valid_date"].dt.month.isin(months)].copy()
        if is_wrapping:
            # DJF: Jan/Feb belong to the previous December's year
            df_tri["season_year"] = df_tri["valid_date"].apply(
                lambda d: d.year if d.month > 6 else d.year - 1
            )
        else:
            df_tri["season_year"] = df_tri["valid_date"].dt.year

        # Drop incomplete seasons
        counts = df_tri.groupby(["iso3", "season_year"])["valid_date"].count()
        complete = counts[counts == 3].index
        df_tri = df_tri.set_index(["iso3", "season_year"]).loc[complete].reset_index()

        seasonal = df_tri.groupby(["iso3", "season_year"])["mean"].mean()
        for (iso, yr), val in seasonal.items():
            if (iso, tri) not in results:
                results[(iso, tri)] = {}
            results[(iso, tri)][yr] = val

    out = pd.DataFrame(
        {k: pd.Series(v) for k, v in results.items()}
    )
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["iso3", "trimester"])
    return out


# --------------------------------------------------------------------------- #
# Correlation with lag sweep
# --------------------------------------------------------------------------- #
def _index_trimester_mean(
    idx: pd.Series, end_month: int, lag: int, year_offset: int = 0
) -> pd.Series:
    """Mean of 3-month index window ending `lag` months before trimester end,
    returned as a year-indexed series using the same year convention as rainfall."""
    shifted = idx.shift(lag)
    win = shifted.rolling(3).mean()
    sub = win[win.index.month == end_month]
    return pd.Series(sub.values, index=sub.index.year + year_offset)


@dataclass
class CorrResult:
    iso3: str
    index: str
    trimester: str
    lag: int
    r: float
    p: float
    n: int


def sweep(rain: pd.DataFrame, indices: pd.DataFrame, cfg: dict,
          max_lag: int | None = None) -> pd.DataFrame:
    """TOTAL pass. For every iso3 x trimester x index, sweep lags and keep the
    lag with max |r|. The chosen lag per (iso3, trimester, index) is the frozen
    best-lag reused by the PARTIAL pass, so both passes are strictly comparable.

    max_lag overrides cfg['max_lag_months'] when given (used for the lag-cap
    toggle variants)."""
    if max_lag is None:
        max_lag = cfg["max_lag_months"]
    rows: list[CorrResult] = []
    isos = rain.columns.get_level_values("iso3").unique()
    for iso in isos:
        for tri, end_m in TRIMESTERS.items():
            if (iso, tri) not in rain.columns:
                continue
            y = rain[(iso, tri)].dropna()
            if len(y) < cfg["min_years"]:
                continue
            offset = TRIMESTER_YEAR_OFFSET[tri]
            for name in indices.columns:
                best = None
                for lag in range(max_lag + 1):
                    x = _index_trimester_mean(indices[name], end_m, lag, year_offset=offset)
                    j = y.index.intersection(x.index)
                    if len(j) < cfg["min_years"]:
                        continue
                    xv, yv = x.loc[j].values, y.loc[j].values
                    m = np.isfinite(xv) & np.isfinite(yv)
                    if m.sum() < cfg["min_years"]:
                        continue
                    r, p = stats.pearsonr(xv[m], yv[m])
                    if best is None or abs(r) > abs(best.r):
                        best = CorrResult(iso, name, tri, lag, r, p, int(m.sum()))
                if best is not None:
                    rows.append(best)
    return pd.DataFrame([r.__dict__ for r in rows])


# --------------------------------------------------------------------------- #
# PARTIAL correlation pass (residuals method, frozen lags from sweep)
# --------------------------------------------------------------------------- #
def _partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray):
    """Partial r of x,y controlling for Z columns, via residual correlation.
    Returns (r, p, dof)."""
    Z1 = np.column_stack([np.ones(len(x)), Z]) if Z.size else np.ones((len(x), 1))
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    r, _ = stats.pearsonr(rx, ry)
    k = Z.shape[1] if Z.size else 0
    dof = len(x) - k - 2
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan, dof
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, dof


def partial_pass(
    rain: pd.DataFrame, indices: pd.DataFrame, total: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """For each iso3 x trimester, build ONE aligned matrix using each index at its
    frozen best-lag (from `total`), then compute partial r of every index vs
    rainfall holding the others constant. cfg['partial_exclude'] (e.g. pdo)
    are excluded from the control set but still have their own partial computed."""
    exclude = set(cfg.get("partial_exclude", []))
    rows: list[CorrResult] = []
    best_lag = {(r.iso3, r.trimester, r.index): int(r.lag)
                for r in total.itertuples()}

    isos = rain.columns.get_level_values("iso3").unique()
    for iso in isos:
        for tri, end_m in TRIMESTERS.items():
            if (iso, tri) not in rain.columns:
                continue
            y = rain[(iso, tri)].dropna()
            if len(y) < cfg["min_years"]:
                continue
            offset = TRIMESTER_YEAR_OFFSET[tri]
            cols = {}
            for name in indices.columns:
                lag = best_lag.get((iso, tri, name))
                if lag is None:
                    continue
                cols[name] = _index_trimester_mean(
                    indices[name], end_m, lag, year_offset=offset
                )
            if not cols:
                continue
            mat = pd.DataFrame(cols)
            mat["__y"] = y
            mat = mat.dropna()
            if len(mat) < cfg["min_years"]:
                continue
            yv = mat["__y"].values
            for name in indices.columns:
                if name not in mat.columns:
                    continue
                controls = [c for c in mat.columns
                            if c not in ("__y", name) and c not in exclude]
                Z = mat[controls].values if controls else np.empty((len(mat), 0))
                r, p, _ = _partial_corr(mat[name].values, yv, Z)
                rows.append(CorrResult(
                    iso, name, tri, best_lag[(iso, tri, name)], r, p, int(len(mat))
                ))
    return pd.DataFrame([r.__dict__ for r in rows])


# --------------------------------------------------------------------------- #
# Reduce to per-country display record (per index)
# --------------------------------------------------------------------------- #
def rainy_trimesters(rain: pd.DataFrame) -> set[tuple[str, str]]:
    """Return {(iso3, trimester)} where the trimester mean is ≥25% of the
    country's annual mean.

    Annual mean is computed from the 4 non-overlapping canonical trimesters
    (DJF+MAM+JAS+OND), which together cover each month exactly once. Summing
    all 12 rolling windows would triple-count every month.
    """
    clim = rain.mean()  # Series: MultiIndex (iso3, trimester) → mean mm/day
    rainy: set[tuple[str, str]] = set()
    for iso in clim.index.get_level_values("iso3").unique():
        annual = sum(
            clim.get((iso, t), np.nan)
            for t in _ANNUAL_TRIMESTERS
            if pd.notna(clim.get((iso, t), np.nan))
        )
        if annual <= 0:
            continue
        for tri in TRIMESTERS:
            v = clim.get((iso, tri), np.nan)
            if pd.notna(v) and v / annual >= 0.25:
                rainy.add((iso, tri))
    return rainy


def reduce_for_display(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """For each iso3 x index: keep significant trimester-best corrs, then decide
    single vs bidirectional. Returns one row per iso3 x index for mapping.

    Bidirectional requires a non-overlapping positive+negative pair — with 12
    rolling windows an overlapping pair (e.g. JFM+ and FMA−) would be spurious.
    """
    sig = df[df["p"] < alpha].copy()
    out = []
    for (iso, name), g in sig.groupby(["iso3", "index"]):
        best = g.loc[g["r"].abs().idxmax()]
        best_months = set(_TRIMESTER_MONTHS[best["trimester"]])
        # Opposite-sign candidates that share no months with the best trimester
        opp = g[
            (g["r"] * best["r"] < 0) &
            g["trimester"].apply(
                lambda t: not bool(best_months & set(_TRIMESTER_MONTHS[t]))
            )
        ]
        is_bi = len(opp) > 0
        rec = {"iso3": iso, "index": name, "bidirectional": is_bi}
        if is_bi:
            other = opp.loc[opp["r"].abs().idxmax()]
            bp = best if best["r"] > 0 else other
            bn = other if best["r"] > 0 else best
            rec.update(
                r_pos=bp["r"], tri_pos=bp["trimester"], lag_pos=int(bp["lag"]),
                r_neg=bn["r"], tri_neg=bn["trimester"], lag_neg=int(bn["lag"]),
                r=best["r"], trimester=best["trimester"], lag=int(best["lag"]),
            )
        else:
            rec.update(r=best["r"], trimester=best["trimester"], lag=int(best["lag"]),
                       r_pos=np.nan, r_neg=np.nan)
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# PIXEL-LEVEL PASS (native 0.25 deg ERA5 grid)
#
# Same method as the country pass, run on every land grid cell in the map
# viewport instead of on country polygons:
#   monthly ERA5 precip COGs -> 12 rolling trimester means per cell
#   -> lag sweep (total r) -> partial r holding the other modes constant
#   -> rainy-season + significance filter -> RGB raster maps.
#
# Everything is vectorised over cells: a single (n_years, n_cells) matrix per
# trimester, so the whole grid is one Pearson/regression call per (index, lag)
# rather than a Python loop over ~413k cells.
# --------------------------------------------------------------------------- #
def _tri_month_pairs(tri: str, season_year: int) -> list[tuple[int, int]]:
    """The three (calendar_year, month) pairs making up `tri` of `season_year`.

    Mirrors the country pass exactly: NDJ/DJF wrap, so their Jan/Feb months
    belong to the following calendar year; every other window sits inside one.
    """
    months = _TRIMESTER_MONTHS[tri]
    wrapping = 1 in months and 12 in months
    return [
        (season_year + (0 if (not wrapping or m > 6) else 1), m)
        for m in months
    ]


def pixel_monthly_era5(cfg: dict) -> tuple[np.ndarray, list[tuple[int, int]],
                                           np.ndarray, np.ndarray]:
    """Monthly ERA5 precip (mm/day) for the map viewport, as a disk-backed array.

    Downloads `era5/monthly/processed/*.tif` COGs from the team raster blob
    container once, subsets each to MAP_XLIM/MAP_YLIM, and caches the stack to
    cache/era5_pixel/monthly.npy so re-runs are free.

    Returns (stack, ym, x, y) where stack is a read-only memmap of shape
    (n_months, ny, nx), ym is the matching list of (year, month), and x/y are
    the cell-centre coordinates (y descending, as in the source COG).
    """
    cache_dir: Path = cfg["cache_dir"] / "era5_pixel"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path, meta_path = cache_dir / "monthly.npy", cache_dir / "meta.json"

    want = [(yr, m)
            for yr in range(cfg["start_year"], cfg["end_year"] + 1)
            for m in range(1, 13)]

    if arr_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["ym"] and [tuple(t) for t in meta["ym"]] == want[:len(meta["ym"])]:
            stack = np.load(arr_path, mmap_mode="r")
            print(f"ERA5 pixel cache: {stack.shape[0]} months, "
                  f"grid {stack.shape[1]}x{stack.shape[2]}")
            return (stack, [tuple(t) for t in meta["ym"]],
                    np.asarray(meta["x"]), np.asarray(meta["y"]))

    from concurrent.futures import ThreadPoolExecutor

    container = stratus.get_container_client("raster", stage="prod")
    available = {
        _PIXEL_DATE_RE.search(b.name).group(0): b.name
        for b in container.list_blobs(name_starts_with=PIXEL_BLOB_PREFIX)
        if _PIXEL_DATE_RE.search(b.name)
    }

    # Grid window, read once from the first available COG
    probe = stratus.open_blob_cog(next(iter(available.values())),
                                  container_name="raster",
                                  container_client=container)
    xs, ys = probe.x.values, probe.y.values
    xsel = np.where((xs >= MAP_XLIM[0] - 1e-6) & (xs <= MAP_XLIM[1] + 1e-6))[0]
    ysel = np.where((ys >= MAP_YLIM[0] - 1e-6) & (ys <= MAP_YLIM[1] + 1e-6))[0]
    x0, x1, y0, y1 = xsel[0], xsel[-1], ysel[0], ysel[-1]
    x, y = xs[x0:x1 + 1], ys[y0:y1 + 1]

    ym = [(yr, m) for (yr, m) in want if f"{yr}-{m:02d}-01" in available]
    missing = len(want) - len(ym)
    if missing:
        print(f"  {missing} month(s) have no ERA5 COG on blob and are skipped")

    stack = np.lib.format.open_memmap(
        arr_path, mode="w+", dtype="float32", shape=(len(ym), len(y), len(x))
    )

    def _fetch(job):
        i, (yr, m) = job
        da = stratus.open_blob_cog(available[f"{yr}-{m:02d}-01"],
                                   container_name="raster",
                                   container_client=container)
        stack[i] = da.isel(band=0, y=slice(y0, y1 + 1),
                           x=slice(x0, x1 + 1)).values.astype("float32")
        return i

    print(f"Downloading {len(ym)} ERA5 monthly COGs "
          f"({len(y)}x{len(x)} cells each)...")
    with ThreadPoolExecutor(PIXEL_DOWNLOAD_WORKERS) as ex:
        for done, _ in enumerate(ex.map(_fetch, enumerate(ym)), start=1):
            if done % 60 == 0:
                print(f"  {done}/{len(ym)}")
    stack.flush()
    del stack

    meta_path.write_text(json.dumps(
        {"ym": ym, "x": x.tolist(), "y": y.tolist()}
    ))
    return np.load(arr_path, mmap_mode="r"), ym, x, y


@dataclass
class PixelData:
    """Trimester rainfall on the ERA5 land grid, plus the masks the maps need.

    All analysis arrays are indexed by *land cell* (length n_cells), not by full
    grid position — ocean is ~2/3 of the viewport and carries no rainfall
    signal worth mapping.  `scatter()` puts a per-cell vector back on the grid.
    """
    x: np.ndarray                       # (nx,) cell-centre longitudes
    y: np.ndarray                       # (ny,) cell-centre latitudes, descending
    shape: tuple[int, int]              # (ny, nx)
    flat_idx: np.ndarray                # (n_cells,) flat position of each land cell
    tri_years: dict[str, np.ndarray]    # trimester -> season years with full data
    tri_clim: dict[str, np.ndarray]     # trimester -> climatological mean mm/day
    rainy: dict[str, np.ndarray]        # trimester -> analysable-season mask
    analyzed: np.ndarray                # (n_cells,) any analysable season at all
    _stack: np.ndarray                  # (n_months, ny, nx) memmap
    _mpos: dict[tuple[int, int], int]   # (year, month) -> month index in _stack

    @property
    def n_cells(self) -> int:
        return len(self.flat_idx)

    def trimester(self, tri: str) -> np.ndarray:
        """(n_years, n_cells) float32 mean mm/day for `tri`, read from the memmap."""
        years = self.tri_years[tri]
        out = np.empty((len(years), self.n_cells), dtype="float32")
        for k, season_year in enumerate(years):
            pos = [self._mpos[p] for p in _tri_month_pairs(tri, int(season_year))]
            out[k] = self._stack[pos].mean(axis=0).reshape(-1)[self.flat_idx]
        return out

    def scatter(self, vals: np.ndarray, fill=np.nan) -> np.ndarray:
        """Place a per-cell vector back on the (ny, nx) grid."""
        grid = np.full(self.shape[0] * self.shape[1], fill, dtype=vals.dtype)
        grid[self.flat_idx] = vals
        return grid.reshape(self.shape)


def pixel_land_mask(gdf: gpd.GeoDataFrame, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rasterise the admin-0 layer onto the ERA5 grid -> (ny, nx) bool land mask."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    res = float(abs(x[1] - x[0]))
    transform = from_origin(x[0] - res / 2, y[0] + res / 2, res, res)
    return rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None),
        out_shape=(len(y), len(x)),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def pixel_trimester_rainfall_era5(cfg: dict, gdf: gpd.GeoDataFrame) -> PixelData:
    """Build the pixel-level equivalent of `country_trimester_rainfall_era5`.

    Applies two per-cell season filters, both needed at this resolution:
      - rainy season: trimester climatology >= 25% of the annual total (same
        rule as the country pass);
      - wet enough: trimester climatology >= PIXEL_MIN_TRI_MM_DAY, which drops
        hyper-arid cells (Sahara, Rub' al Khali, Taklamakan) where correlations
        against near-zero rainfall are numerically strong but meaningless.
    """
    stack, ym, x, y = pixel_monthly_era5(cfg)
    land = pixel_land_mask(gdf, x, y)
    flat_idx = np.flatnonzero(land.reshape(-1))
    mpos = {p: i for i, p in enumerate(ym)}
    print(f"Pixel grid: {land.shape[0]}x{land.shape[1]} cells, "
          f"{len(flat_idx)} on land")
    # Cached so the report can quote the real count even on a --skip-px run.
    (cfg["cache_dir"] / "era5_pixel" / "cells.json").write_text(
        json.dumps({"n_cells": int(len(flat_idx)),
                    "shape": [int(v) for v in land.shape]})
    )

    tri_years = {}
    for tri in TRIMESTERS:
        tri_years[tri] = np.array([
            sy for sy in range(cfg["start_year"], cfg["end_year"] + 1)
            if all(p in mpos for p in _tri_month_pairs(tri, sy))
        ])

    px = PixelData(x=x, y=y, shape=land.shape, flat_idx=flat_idx,
                   tri_years=tri_years, tri_clim={}, rainy={},
                   analyzed=np.zeros(len(flat_idx), dtype=bool),
                   _stack=stack, _mpos=mpos)

    for tri in TRIMESTERS:
        px.tri_clim[tri] = px.trimester(tri).mean(axis=0)

    annual = np.zeros(px.n_cells, dtype="float64")
    for tri in _ANNUAL_TRIMESTERS:
        annual += px.tri_clim[tri]
    with np.errstate(invalid="ignore", divide="ignore"):
        for tri in TRIMESTERS:
            clim = px.tri_clim[tri]
            px.rainy[tri] = (
                (annual > 0)
                & (clim / annual >= 0.25)
                & (clim >= PIXEL_MIN_TRI_MM_DAY)
                & (len(px.tri_years[tri]) >= cfg["min_years"])
            )
            px.analyzed |= px.rainy[tri]

    n_pairs = sum(int(m.sum()) for m in px.rainy.values())
    print(f"Analysable (cell, trimester) pairs: {n_pairs}; "
          f"{int(px.analyzed.sum())} of {px.n_cells} land cells have >=1")
    return px


# --------------------------------------------------------------------------- #
# Vectorised correlation helpers
# --------------------------------------------------------------------------- #
def _centered(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-centre Y (n, m) and return (Yc, column L2 norms)."""
    Yc = Y - Y.mean(axis=0)
    return Yc, np.sqrt((Yc ** 2).sum(axis=0))


def _pearson_p(r: np.ndarray, n: np.ndarray | int, k: int = 0) -> np.ndarray:
    """Two-tailed p for Pearson/partial r with `k` controlled variables."""
    dof = np.asarray(n, dtype="float64") - k - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(dof / np.clip(1 - r ** 2, 1e-12, None))
        p = 2 * stats.t.sf(np.abs(t), dof)
    return np.where(np.isfinite(r) & (dof > 0), p, np.nan)


def _index_lag_series(indices: pd.DataFrame, name: str, tri: str, lag: int) -> pd.Series:
    """Index `name` as a year-indexed series for trimester `tri` at `lag` months."""
    return _index_trimester_mean(
        indices[name], TRIMESTERS[tri], lag, year_offset=TRIMESTER_YEAR_OFFSET[tri]
    ).dropna()


def pixel_sweep(px: PixelData, indices: pd.DataFrame, cfg: dict,
                max_lag: int) -> dict[str, dict[str, np.ndarray]]:
    """TOTAL pass on the grid: per (trimester, index) keep the lag with max |r|.

    Vectorised over cells — one Pearson call per (index, lag) covering the whole
    grid. Returns res[trimester] = {r, p, lag, n}, each (n_indices, n_cells).
    """
    cols = list(indices.columns)
    res: dict[str, dict[str, np.ndarray]] = {}

    for tri in TRIMESTERS:
        Y = px.trimester(tri)
        yrs = px.tri_years[tri]
        y_ok = Y.std(axis=0) > 0

        r_best = np.full((len(cols), px.n_cells), np.nan, dtype="float32")
        lag_best = np.full((len(cols), px.n_cells), -1, dtype="int8")
        n_best = np.zeros((len(cols), px.n_cells), dtype="int16")
        cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

        for j, name in enumerate(cols):
            for lag in range(max_lag + 1):
                xs = _index_lag_series(indices, name, tri, lag)
                common = np.intersect1d(yrs, xs.index.values)
                if len(common) < cfg["min_years"]:
                    continue
                rowsel = np.isin(yrs, common)
                key = rowsel.tobytes()
                if key not in cache:
                    cache[key] = _centered(Y[rowsel])
                Yc, sy = cache[key]
                xc = xs.loc[common].values.astype("float64")
                xc -= xc.mean()
                sx = np.sqrt((xc ** 2).sum())
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = (xc @ Yc) / (sx * sy)
                upd = (
                    y_ok & np.isfinite(r)
                    & (np.isnan(r_best[j]) | (np.abs(r) > np.abs(r_best[j])))
                )
                r_best[j, upd] = r[upd]
                lag_best[j, upd] = lag
                n_best[j, upd] = len(common)

        res[tri] = {
            "r": r_best, "lag": lag_best, "n": n_best,
            "p": _pearson_p(r_best, n_best).astype("float32"),
        }
    return res


def _partial_vs_last(V: np.ndarray) -> np.ndarray:
    """Partial correlation of each leading column of V with its last column,
    each controlling for all the other columns.

    V: (c, n, q) -> (c, q-1). Uses the precision-matrix identity
    r_ij|rest = -P_ij / sqrt(P_ii P_jj), which is algebraically identical to the
    residual regression used by the country pass but batched over cells.
    """
    c, n, q = V.shape
    Vc = V - V.mean(axis=1, keepdims=True)
    sd = np.sqrt((Vc ** 2).sum(axis=1))
    ok = sd > 0
    Vs = Vc / np.where(ok, sd, 1.0)[:, None, :]
    C = np.einsum("cni,cnj->cij", Vs, Vs)          # unit diagonal -> correlation
    C[:, np.arange(q), np.arange(q)] += 1e-9       # guard exact collinearity
    P = np.linalg.inv(C)
    d = np.sqrt(np.abs(np.diagonal(P, axis1=1, axis2=2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = -P[:, :-1, -1] / (d[:, :-1] * d[:, -1:])
    r = np.clip(r, -1.0, 1.0)
    r[~(ok[:, :-1] & ok[:, -1:])] = np.nan
    return r


def pixel_partial_pass(px: PixelData, indices: pd.DataFrame, cfg: dict,
                       total: dict[str, dict[str, np.ndarray]],
                       max_lag: int) -> dict[str, dict[str, np.ndarray]]:
    """PARTIAL pass on the grid, off the frozen best-lags from `pixel_sweep`.

    For each trimester every index enters at its own per-cell best lag, then each
    index's partial r vs rainfall is computed holding the others constant.
    cfg['partial_exclude'] indices are kept out of the control set but still get
    their own partial, exactly as in the country pass — which means one model per
    excluded index plus one shared model for everything else.
    """
    cols = list(indices.columns)
    exclude = set(cfg.get("partial_exclude", []))
    base = [j for j, c in enumerate(cols) if c not in exclude]
    if not base:
        base = list(range(len(cols)))

    # target index -> the model it is estimated in (its controls + itself)
    models: dict[tuple[int, ...], list[int]] = {tuple(base): list(base)}
    for j, c in enumerate(cols):
        if c in exclude:
            models.setdefault(tuple(sorted(base + [j])), []).append(j)

    res: dict[str, dict[str, np.ndarray]] = {}
    for tri in TRIMESTERS:
        yrs = px.tri_years[tri]
        # One year set per trimester: seasons where every index is available at
        # every candidate lag, so the frozen per-cell lag mix is always aligned.
        common = yrs
        series: dict[tuple[int, int], pd.Series] = {}
        for j, name in enumerate(cols):
            for lag in range(max_lag + 1):
                s = _index_lag_series(indices, name, tri, lag)
                series[(j, lag)] = s
                common = np.intersect1d(common, s.index.values)
        if len(common) < cfg["min_years"]:
            nan = np.full((len(cols), px.n_cells), np.nan, dtype="float32")
            res[tri] = {"r": nan, "p": nan.copy(),
                        "lag": total[tri]["lag"].copy(),
                        "n": np.zeros((len(cols), px.n_cells), dtype="int16")}
            continue

        rowsel = np.isin(yrs, common)
        Yt = px.trimester(tri)[rowsel].T.astype("float32")      # (n_cells, n_yrs)
        X = np.stack([
            np.stack([series[(j, lag)].loc[common].values for lag in range(max_lag + 1)])
            for j in range(len(cols))
        ]).astype("float32")                                    # (n_idx, n_lags, n_yrs)

        lag_best = np.clip(total[tri]["lag"], 0, max_lag).astype("int64")
        r_out = np.full((len(cols), px.n_cells), np.nan, dtype="float32")
        p_out = np.full((len(cols), px.n_cells), np.nan, dtype="float32")
        n_ctrl = np.zeros(len(cols), dtype="int64")

        for start in range(0, px.n_cells, PIXEL_CHUNK):
            sl = slice(start, min(start + PIXEL_CHUNK, px.n_cells))
            c = sl.stop - sl.start
            M = np.empty((c, len(common), len(cols)), dtype="float32")
            for j in range(len(cols)):
                M[:, :, j] = X[j][lag_best[j, sl]]
            for model, targets in models.items():
                V = np.concatenate(
                    [M[:, :, list(model)], Yt[sl][:, :, None]], axis=2
                ).astype("float64")
                rp = _partial_vs_last(V)                        # (c, len(model))
                for t in targets:
                    r_out[t, sl] = rp[:, list(model).index(t)]
                    n_ctrl[t] = len(model) - 1

        for j in range(len(cols)):
            p_out[j] = _pearson_p(r_out[j], len(common), k=int(n_ctrl[j]))

        res[tri] = {
            "r": r_out, "p": p_out,
            "lag": total[tri]["lag"].copy(),
            "n": np.full((len(cols), px.n_cells), len(common), dtype="int16"),
        }
    return res


# --------------------------------------------------------------------------- #
# Reduce to per-cell display record (per index)
# --------------------------------------------------------------------------- #
_TRI_ORDER = list(TRIMESTERS)
_TRI_OVERLAP = np.array([
    [bool(set(_TRIMESTER_MONTHS[a]) & set(_TRIMESTER_MONTHS[b])) for b in _TRI_ORDER]
    for a in _TRI_ORDER
])


def pixel_reduce_for_display(res: dict[str, dict[str, np.ndarray]], px: PixelData,
                             cols: list[str], alpha: float) -> dict[str, dict[str, np.ndarray]]:
    """Per (cell, index): strongest significant trimester, plus the bidirectional
    test. Same rules as `reduce_for_display`, vectorised over cells — including
    the requirement that an opposite-sign season share no month with the best one.
    """
    cell = np.arange(px.n_cells)
    rainy = np.stack([px.rainy[t] for t in _TRI_ORDER])            # (12, n_cells)
    out: dict[str, dict[str, np.ndarray]] = {}

    for j, name in enumerate(cols):
        R = np.stack([res[t]["r"][j] for t in _TRI_ORDER]).astype("float64")
        P = np.stack([res[t]["p"][j] for t in _TRI_ORDER]).astype("float64")
        L = np.stack([res[t]["lag"][j] for t in _TRI_ORDER])
        sig = rainy & np.isfinite(R) & (P < alpha)

        score = np.where(sig, np.abs(R), -1.0)
        bt = score.argmax(axis=0)
        has = score.max(axis=0) > 0
        r_best = np.where(has, R[bt, cell], np.nan)
        lag_best = L[bt, cell]

        # opposite sign AND no shared month with the best trimester
        opp = (sig.T & (R.T * r_best[:, None] < 0) & ~_TRI_OVERLAP[bt])
        is_bi = opp.any(axis=1) & has
        oscore = np.where(opp, np.abs(R.T), -1.0)
        ot = oscore.argmax(axis=1)
        r_other = np.where(is_bi, R.T[cell, ot], np.nan)
        lag_other = L.T[cell, ot]
        best_is_pos = r_best > 0

        out[name] = {
            "r": r_best,
            "lag": np.where(has, lag_best, -1),
            "tri": np.where(has, bt, -1),
            "bidirectional": is_bi,
            "r_pos": np.where(best_is_pos, r_best, r_other),
            "r_neg": np.where(best_is_pos, r_other, r_best),
            "lag_pos": np.where(best_is_pos, lag_best, lag_other),
            "lag_neg": np.where(best_is_pos, lag_other, lag_best),
        }
    return out


# --------------------------------------------------------------------------- #
# Pixel map rendering
# --------------------------------------------------------------------------- #
_PX_OCEAN    = "#FFFFFF"   # no land -> not part of the analysis
_PX_ARID     = "#F4F1EA"   # land, but no season passes the wet-enough filter
_PX_NOSIG    = "#E8E8E8"   # land with an analysed season, no significant signal
_PX_TIED     = "#CBD1D9"   # dominant-mode map: modes within noise of each other
_PX_OUTLINE  = "#9AA3B0"   # country boundaries drawn over the raster


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _pixel_extent(px: PixelData) -> tuple[float, float, float, float]:
    """imshow extent covering full cell edges (y descending -> origin='upper')."""
    half = float(abs(px.x[1] - px.x[0])) / 2
    return (float(px.x[0]) - half, float(px.x[-1]) + half,
            float(px.y[-1]) - half, float(px.y[0]) + half)


def _pixel_base_rgb(px: PixelData) -> np.ndarray:
    """(ny, nx, 3) uint8 base: ocean / arid land / analysed-but-no-signal."""
    img = np.empty((*px.shape, 3), dtype="uint8")
    img[:] = _hex_to_rgb(_PX_OCEAN)
    for mask, color in ((np.ones(px.n_cells, bool), _PX_ARID),
                        (px.analyzed, _PX_NOSIG)):
        sel = px.scatter(mask.astype("uint8"), fill=0).astype(bool)
        img[sel] = _hex_to_rgb(color)
    return img


def _paint_r_bins(img: np.ndarray, px: PixelData, r: np.ndarray) -> None:
    """Colour cells by the discrete |r| bins used on the country maps."""
    bins = [
        (r >= _R_STRONG, "#0D40B0"),
        ((r >= _R_MIN) & (r < _R_STRONG), "#71B3E5"),
        ((r <= -_R_MIN) & (r > -_R_STRONG), "#C8844A"),
        (r <= -_R_STRONG, "#7B3A1A"),
    ]
    for mask, color in bins:
        m = np.where(np.isfinite(r), mask, False)
        if not m.any():
            continue
        sel = px.scatter(m.astype("uint8"), fill=0).astype(bool)
        img[sel] = _hex_to_rgb(color)


def _finish_pixel_ax(ax, px: PixelData, gdf: gpd.GeoDataFrame) -> None:
    gdf.boundary.plot(ax=ax, color=_PX_OUTLINE, linewidth=0.3, zorder=4)
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _overlay_bidirectional(ax, px: PixelData, mask: np.ndarray) -> None:
    """Hatch the regions that are significant in both directions across
    non-overlapping seasons — the raster analogue of the split-diagonal fill."""
    if not mask.any():
        return
    grid = px.scatter(mask.astype("float32"), fill=0.0)
    xs = np.asarray(px.x, dtype="float64")
    ys = np.asarray(px.y, dtype="float64")
    # White hatch: the four correlation fills range from pale blue to dark brown,
    # and white is the one stroke colour that reads on all of them.
    with plt.rc_context({"hatch.linewidth": 0.6, "hatch.color": "#FFFFFF"}):
        ax.contourf(xs, ys, grid, levels=[0.5, 1.5], colors="none",
                    hatches=["////"], zorder=3)


def plot_pixel_index_map(disp: dict[str, dict[str, np.ndarray]], px: PixelData,
                         gdf: gpd.GeoDataFrame, index: str, out_dir: Path,
                         kind: str = "total", lag_tag: str = "l6") -> None:
    """Per-cell correlation map for one index, saved as map_px_{lag}_{kind}_{index}.png."""
    d = disp[index]
    img = _pixel_base_rgb(px)
    _paint_r_bins(img, px, d["r"])

    fig, ax = plt.subplots(figsize=(PIXEL_MAP_W, PIXEL_MAP_W * _MAP_DY / _MAP_DX))
    ax.imshow(img, extent=_pixel_extent(px), origin="upper",
              interpolation="nearest", zorder=1)
    _overlay_bidirectional(ax, px, d["bidirectional"] & np.isfinite(d["r"]))
    _finish_pixel_ax(ax, px, gdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"map_px_{lag_tag}_{kind}_{index}.png",
                dpi=PIXEL_MAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pixel_dominant_map(disp_total: dict[str, dict[str, np.ndarray]],
                            px: PixelData, gdf: gpd.GeoDataFrame,
                            cols: list[str], out_dir: Path,
                            lag_tag: str = "l6") -> None:
    """Each cell coloured by the index with the strongest significant total r.

    Two departures from the country version, both forced by the resolution:
      - only the top mode is shown; the runner-up shown as a split diagonal on
        the country map is not legible in a single 0.25 deg cell;
      - the top mode must beat the runner-up by PIXEL_DOMINANT_MARGIN in |r|,
        otherwise the cell is drawn as "no single dominant mode". Country means
        average enough grid cells to stabilise the arg-max; a single cell does
        not, and without the margin the map is confetti wherever two collinear
        modes are within noise of each other — which is most of the world.
    """
    R = np.stack([np.abs(disp_total[c]["r"]) for c in cols])
    R = np.where(np.isfinite(R) & (R >= _R_MIN), R, -1.0)
    top = R.argmax(axis=0)
    srt = np.sort(R, axis=0)
    has = srt[-1] > 0
    # gap to the runner-up; infinite when only one mode is significant at all
    gap = np.where(srt[-2] > 0, srt[-1] - srt[-2], np.inf)
    clear = has & (gap >= PIXEL_DOMINANT_MARGIN)

    img = _pixel_base_rgb(px)
    sel = px.scatter((has & ~clear).astype("uint8"), fill=0).astype(bool)
    img[sel] = _hex_to_rgb(_PX_TIED)
    for j, name in enumerate(cols):
        m = clear & (top == j)
        if not m.any():
            continue
        sel = px.scatter(m.astype("uint8"), fill=0).astype(bool)
        img[sel] = _hex_to_rgb(INDEX_COLORS.get(name, "#CCCCCC"))

    fig, ax = plt.subplots(figsize=(PIXEL_MAP_W, PIXEL_MAP_W * _MAP_DY / _MAP_DX))
    ax.imshow(img, extent=_pixel_extent(px), origin="upper",
              interpolation="nearest", zorder=1)
    _finish_pixel_ax(ax, px, gdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"map_px_dominant_{lag_tag}.png",
                dpi=PIXEL_MAP_DPI, bbox_inches="tight")
    plt.close(fig)


def pixel_enso_composite_maps(px: PixelData, indices: pd.DataFrame,
                              gdf: gpd.GeoDataFrame, cfg: dict) -> None:
    """Per-cell El Nino / La Nina composites, mirroring `enso_composite_maps`:
    each cell is scored in its own headline season (the analysable trimester with
    the largest ENSO response) and coloured by the mean standardised anomaly."""
    classes = {}
    for tri, end_m in TRIMESTERS.items():
        nino = _index_trimester_mean(indices["nino34"], end_month=end_m, lag=0,
                                     year_offset=TRIMESTER_YEAR_OFFSET[tri])
        nino = nino[(nino.index >= cfg["start_year"]) &
                    (nino.index <= cfg["end_year"])].dropna()
        classes[tri] = nino

    best_signal = np.zeros(px.n_cells)
    best = {"ElNino": np.full(px.n_cells, np.nan),
            "LaNina": np.full(px.n_cells, np.nan)}

    for tri in TRIMESTERS:
        yrs = px.tri_years[tri]
        nino = classes[tri]
        common = np.intersect1d(yrs, nino.index.values)
        if len(common) < 5:
            continue
        vals = nino.loc[common].values
        el, la = vals >= 0.5, vals <= -0.5
        if el.sum() < 3 or la.sum() < 3:
            continue
        Y = px.trimester(tri)[np.isin(yrs, common)].astype("float64")
        mean, std = Y.mean(axis=0), Y.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (Y - mean) / std
        el_anom, la_anom = z[el].mean(axis=0), z[la].mean(axis=0)
        signal = np.maximum(np.abs(el_anom), np.abs(la_anom))
        upd = (px.rainy[tri] & (std > 0) & np.isfinite(signal)
               & (signal > best_signal))
        best_signal[upd] = signal[upd]
        best["ElNino"][upd] = el_anom[upd]
        best["LaNina"][upd] = la_anom[upd]

    finite = np.concatenate([v[np.isfinite(v)] for v in best.values()])
    vmax = max(float(np.quantile(np.abs(finite), 0.95)), 0.5) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for phase, fname in (("ElNino", "enso_px_elnino.png"),
                         ("LaNina", "enso_px_lanina.png")):
        anom = best[phase]
        img = _pixel_base_rgb(px).astype("float32") / 255.0
        m = np.isfinite(anom)
        if m.any():
            rgba = DROUGHT_FLOOD_CMAP(norm(anom[m]))[:, :3]
            sel = px.scatter(m.astype("uint8"), fill=0).astype(bool)
            img[sel] = rgba
        fig, ax = plt.subplots(figsize=(PIXEL_MAP_W,
                                        PIXEL_MAP_W * _MAP_DY / _MAP_DX))
        ax.imshow(img, extent=_pixel_extent(px), origin="upper",
                  interpolation="nearest", zorder=1)
        _finish_pixel_ax(ax, px, gdf)
        sm = plt.cm.ScalarMappable(cmap=DROUGHT_FLOOD_CMAP, norm=norm)
        fig.colorbar(sm, ax=ax, shrink=0.5,
                     label="anomaly (SDs from mean)")
        fig.savefig(out_dir / fname, dpi=PIXEL_MAP_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"Pixel ENSO composites written (colour scale +/-{vmax:.2f} SD)")


# --------------------------------------------------------------------------- #
# Collinearity check
# --------------------------------------------------------------------------- #
def print_collinearity(indices: pd.DataFrame) -> None:
    """Print 6x6 Pearson correlation matrix among climate indices."""
    corr = indices.corr().round(2)
    print("\nIndex collinearity (Pearson r):")
    print(corr.to_string())
    print()


# --------------------------------------------------------------------------- #
# Correlation maps
# --------------------------------------------------------------------------- #
def _plot_base_layer(ax, g_poly, analyzed_isos: set[str]) -> None:
    """Draw the two-shade gray base layer: lighter for unanalyzed, medium for no-signal."""
    in_analysis = g_poly["iso3"].isin(analyzed_isos)
    g_poly[~in_analysis].plot(ax=ax, color="#F8F8F8", edgecolor="#EBEBEB", linewidth=0.3)
    g_poly[ in_analysis].plot(ax=ax, color="#E8E8E8", edgecolor="#CCCCCC", linewidth=0.3)


def plot_index_map(
    disp: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    index: str,
    out_dir: Path,
    kind: str = "total",
    end_year: int = 2025,
    indices: pd.DataFrame | None = None,
    analyzed_isos: set[str] | None = None,
    lag_tag: str = "l6",
) -> None:
    """kind: 'total' (pairwise r) or 'partial' (other modes held constant).
    lag_tag: which max-lag variant this is ('l3'/'l6'), used in the filename.

    Saves ts_{index}.png (time series panel, lag/kind-independent, generated
    once) and map_{lag_tag}_{kind}_{index}.png (choropleth only), so the HTML
    can make only the map zoomable while the TS panel stays static beside it."""
    sub = disp[disp["index"] == index]
    g = gdf.merge(sub, on="iso3", how="left")

    is_dot = g.geometry.apply(_is_dot_country)
    g_poly = g[~is_dot]
    g_dot  = g[is_dot]

    ts_w  = 6.5
    map_w = 15.0
    fig_h = map_w * _MAP_DY / _MAP_DX
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Time series panel (lag/kind-independent; generate once) ---
    ts_path = out_dir / f"ts_{index}.png"
    if not ts_path.exists() and indices is not None and index in indices.columns:
        index_label_ts = INDEX_LABELS.get(index, index.upper())
        fig_ts, ax_ts = plt.subplots(figsize=(ts_w, fig_h))
        fig_ts.subplots_adjust(left=0.14, right=0.86, top=0.85, bottom=0.14)
        ax_ts.set_title(f"{index_label_ts}\nHistorical values", fontsize=10, pad=6)
        _draw_ts_panel(ax_ts, indices[index], index)
        fig_ts.savefig(ts_path, dpi=200, bbox_inches="tight")
        plt.close(fig_ts)

    # --- Choropleth map ---
    fig, ax = plt.subplots(figsize=(map_w, fig_h))
    _plot_base_layer(ax, g_poly, analyzed_isos or set())

    has_sig  = g_poly["r"].notna()
    is_bi    = g_poly["bidirectional"].fillna(False) == True  # noqa: E712
    uni_poly = g_poly[has_sig & ~is_bi]
    bi_poly  = g_poly[has_sig &  is_bi]

    # Unidirectional countries: discrete 2-bin color
    if len(uni_poly):
        fc = [_r_colors(r)[0] for r in uni_poly["r"]]
        ec = [_r_colors(r)[1] for r in uni_poly["r"]]
        uni_poly.plot(ax=ax, color=fc, edgecolor=ec, linewidth=0.3)

    # Legend moved to HTML — keeping map PNG clean for inline zoom

    # Bidirectional countries: diagonal 50/50 split
    # upper-right = positive r (flood/blue), lower-left = negative r (drought/brown)
    for _, row in bi_poly.iterrows():
        geom    = row.geometry
        col_pos = _r_colors(row["r_pos"])[0]
        col_neg = _r_colors(row["r_neg"])[0]
        upper, lower = _diagonal_halves(geom)
        if not upper.is_empty:
            gpd.GeoDataFrame(geometry=[upper], crs="EPSG:4326").plot(
                ax=ax, color=col_pos, edgecolor="none")
        if not lower.is_empty:
            gpd.GeoDataFrame(geometry=[lower], crs="EPSG:4326").plot(
                ax=ax, color=col_neg, edgecolor="none")
        gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").plot(
            ax=ax, facecolor="none", edgecolor="#555", linewidth=0.4)
        if not upper.is_empty and abs(row["r_pos"]) >= _R_MIN:
            rp = upper.representative_point()
            ax.annotate(f"{row['tri_pos']}\nL{int(row['lag_pos'])}",
                        (rp.x, rp.y), ha="center", va="center",
                        fontsize=4.5, color="black")
        if not lower.is_empty and abs(row["r_neg"]) >= _R_MIN:
            rp = lower.representative_point()
            ax.annotate(f"{row['tri_neg']}\nL{int(row['lag_neg'])}",
                        (rp.x, rp.y), ha="center", va="center",
                        fontsize=4.5, color="black")

    # Unidirectional annotations
    for _, row in uni_poly.iterrows():
        if abs(row["r"]) < _R_MIN:
            continue
        c = row.geometry.representative_point()
        ax.annotate(f"{row['trimester']}\nL{int(row['lag'])}",
                    (c.x, c.y), ha="center", va="center",
                    fontsize=4.5, color="black")

    # Dot countries
    for _, row in g_dot.iterrows():
        cx, cy = _dot_center(row.geometry)
        if cy < MAP_YLIM[0] or cy > MAP_YLIM[1]:
            continue
        if row.get("bidirectional") is True:
            col_pos = _r_colors(row.get("r_pos", np.nan))[0]
            col_neg = _r_colors(row.get("r_neg", np.nan))[0]
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R, -90,  90,
                                         facecolor=col_pos, edgecolor="#555",
                                         linewidth=0.5, zorder=5))
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R,  90, 270,
                                         facecolor=col_neg, edgecolor="#555",
                                         linewidth=0.5, zorder=5))
            ax.annotate(f"{row['tri_pos']}/L{int(row['lag_pos'])}",
                        (cx, cy + _DOT_R * 1.4), ha="center", va="bottom",
                        fontsize=3.5, color="#222", zorder=6)
        else:
            r_val = row.get("r", np.nan)
            if pd.notna(r_val):
                face, edge = _r_colors(r_val)
                ax.annotate(f"{row['trimester']}/L{int(row['lag'])}",
                            (cx, cy + _DOT_R * 1.4), ha="center", va="bottom",
                            fontsize=3.5, color="#222", zorder=6)
            elif (analyzed_isos and row["iso3"] in analyzed_isos):
                face, edge = "#E8E8E8", "#CCCCCC"
            else:
                face, edge = "#F8F8F8", "#EBEBEB"
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=face,
                                          edgecolor=edge, linewidth=0.5, zorder=5))

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"map_{lag_tag}_{kind}_{index}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# ENSO composite maps
# --------------------------------------------------------------------------- #
def enso_composite_maps(
    rain: pd.DataFrame,
    indices: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    cfg: dict,
    rainy: set[tuple[str, str]] | None = None,
    analyzed_isos: set[str] | None = None,
) -> None:
    """Two maps: mean rainfall anomaly (SDs) in El Niño / La Niña years vs
    climatology. Each country is classified by the Niño3.4 value concurrent
    with its own headline trimester (not a single DJF year label), so MAM
    in decay years and JAS in developing years are correctly identified."""
    # Per-trimester concurrent Niño3.4 classification
    enso_classes: dict[str, pd.Series] = {}
    for tri, end_m in TRIMESTERS.items():
        offset = TRIMESTER_YEAR_OFFSET[tri]
        nino = _index_trimester_mean(indices["nino34"], end_month=end_m, lag=0,
                                     year_offset=offset)
        nino = nino[(nino.index >= cfg["start_year"]) &
                    (nino.index <= cfg["end_year"])].dropna()
        ec = pd.Series("Neutral", index=nino.index, dtype=str)
        ec[nino >= 0.5]  = "ElNino"
        ec[nino <= -0.5] = "LaNina"
        enso_classes[tri] = ec

    isos = rain.columns.get_level_values("iso3").unique()
    composite_rows = []

    for iso in isos:
        best_signal = 0.0
        best_el: dict | None = None
        best_la: dict | None = None

        for tri in TRIMESTERS:
            if rainy is not None and (iso, tri) not in rainy:
                continue
            if (iso, tri) not in rain.columns:
                continue
            y = rain[(iso, tri)].dropna()
            enso_class = enso_classes[tri]
            common = y.index.intersection(enso_class.index)
            if len(common) < 5:
                continue
            y_c = y.loc[common]
            enso_c = enso_class.loc[common]

            clim_mean = y_c.mean()
            clim_std = y_c.std()
            if clim_std == 0 or pd.isna(clim_std):
                continue

            el_idx = enso_c[enso_c == "ElNino"].index
            la_idx = enso_c[enso_c == "LaNina"].index
            el_vals = y_c.loc[y_c.index.intersection(el_idx)]
            la_vals = y_c.loc[y_c.index.intersection(la_idx)]

            if len(el_vals) < 3 or len(la_vals) < 3:
                continue

            el_anom = (el_vals - clim_mean) / clim_std
            la_anom = (la_vals - clim_mean) / clim_std
            signal = max(abs(el_anom.mean()), abs(la_anom.mean()))

            if signal > best_signal:
                best_signal = signal
                best_el = {"iso3": iso, "trimester": tri, "anom": el_anom.mean(),
                           "n": len(el_vals)}
                best_la = {"iso3": iso, "trimester": tri, "anom": la_anom.mean(),
                           "n": len(la_vals)}

        if best_el is not None:
            composite_rows.append({**best_el, "phase": "ElNino"})
            composite_rows.append({**best_la, "phase": "LaNina"})  # type: ignore[arg-type]

    comp_df = pd.DataFrame(composite_rows)
    end_year = cfg["end_year"]
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared colorscale across both phases
    if len(comp_df):
        vmax = max(float(comp_df["anom"].abs().quantile(0.95)), 0.5)
    else:
        vmax = 1.0
    enso_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    enso_cmap = DROUGHT_FLOOD_CMAP

    for phase, label, fname in [
        ("ElNino", "El Niño", "enso_elnino.png"),
        ("LaNina", "La Niña", "enso_lanina.png"),
    ]:
        sub = comp_df[comp_df["phase"] == phase] if len(comp_df) else pd.DataFrame()
        g = gdf.merge(sub, on="iso3", how="left") if len(sub) else gdf.copy()

        is_dot = g.geometry.apply(_is_dot_country)
        g_poly = g[~is_dot]
        g_dot = g[is_dot]

        map_w = 19.0
        fig, ax = plt.subplots(figsize=(map_w, map_w * _MAP_DY / _MAP_DX))

        _plot_base_layer(ax, g_poly, analyzed_isos or set())

        has_poly = g_poly[g_poly["anom"].notna()] if "anom" in g_poly.columns else pd.DataFrame()
        if len(has_poly):
            has_poly.plot(ax=ax, column="anom", cmap=DROUGHT_FLOOD_CMAP, norm=enso_norm,
                          edgecolor="#666", linewidth=0.3, legend=True,
                          legend_kwds={"label": "anomaly (SDs from mean)", "shrink": 0.5})
            for _, row in has_poly.iterrows():
                c = row.geometry.representative_point()
                ax.annotate(row["trimester"], (c.x, c.y),
                            ha="center", va="center", fontsize=4.5, color="black")

        # Dot countries
        for _, row in g_dot.iterrows():
            cx, cy = _dot_center(row.geometry)
            if cy < MAP_YLIM[0] or cy > MAP_YLIM[1]:
                continue
            anom_val = row.get("anom", np.nan) if "anom" in row.index else np.nan
            if pd.notna(anom_val):
                face = enso_cmap(enso_norm(anom_val))
                edge = "#666"
                ax.annotate(row["trimester"], (cx, cy + _DOT_R * 1.4),
                            ha="center", va="bottom", fontsize=3.5, color="#222", zorder=6)
            elif analyzed_isos and row["iso3"] in analyzed_isos:
                face, edge = "#E8E8E8", "#CCCCCC"
            else:
                face, edge = "#F8F8F8", "#EBEBEB"
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=face, edgecolor=edge,
                                         linewidth=0.5, zorder=5))

        # Title in HTML
        ax.set_xlim(MAP_XLIM)
        ax.set_ylim(MAP_YLIM)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Dominant-index map
# --------------------------------------------------------------------------- #
def _build_dominant_two(disp_total: pd.DataFrame) -> pd.DataFrame:
    """For each country return its top-1 index and, where available, a top-2
    index that (a) is a different index and (b) acts on a completely
    non-overlapping trimester.  DJF and OND share December so they are
    considered overlapping and will not co-appear."""
    records = []
    for iso, grp in disp_total.groupby("iso3"):
        ranked = (
            grp.assign(abs_r=grp["r"].abs())
            .sort_values("abs_r", ascending=False)
        )
        first = ranked.iloc[0]
        months1 = set(_TRIMESTER_MONTHS[first["trimester"]])
        rec: dict = dict(
            iso3=iso,
            index1=first["index"],  r1=first["r"],
            trimester1=first["trimester"], lag1=int(first["lag"]),
            index2=None, r2=np.nan, trimester2=None, lag2=None,
        )
        for _, cand in ranked.iloc[1:].iterrows():
            if cand["index"] == first["index"]:
                continue
            if not (months1 & set(_TRIMESTER_MONTHS[cand["trimester"]])):
                rec.update(
                    index2=cand["index"],  r2=cand["r"],
                    trimester2=cand["trimester"], lag2=int(cand["lag"]),
                )
                break
        records.append(rec)
    return pd.DataFrame(records)


def plot_dominant_index_map(
    disp_total: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    end_year: int = 2025,
    analyzed_isos: set[str] | None = None,
    lag_tag: str = "l6",
) -> None:
    """Each country: dominant index (upper-right, color) + optional second index
    on a non-overlapping trimester (lower-left diagonal, different color)."""
    dom = _build_dominant_two(disp_total[disp_total["r"].abs() >= _R_MIN])
    g   = gdf.merge(dom, on="iso3", how="left")

    is_dot = g.geometry.apply(_is_dot_country)
    g_poly = g[~is_dot]
    g_dot  = g[is_dot]

    map_w = 19.0
    fig, ax = plt.subplots(figsize=(map_w, map_w * _MAP_DY / _MAP_DX))

    _plot_base_layer(ax, g_poly, analyzed_isos or set())

    has_sig  = g_poly[g_poly["index1"].notna()]
    has_two  = has_sig[has_sig["index2"].notna()]
    has_one  = has_sig[has_sig["index2"].isna()]

    # Single-index countries: solid fill
    if len(has_one):
        fc = [INDEX_COLORS.get(idx, "#cccccc") for idx in has_one["index1"]]
        has_one.plot(ax=ax, color=fc, edgecolor="#555", linewidth=0.3)
        for _, row in has_one.iterrows():
            c = row.geometry.representative_point()
            ax.annotate(f"{row['trimester1']}\nL{row['lag1']}",
                        (c.x, c.y), ha="center", va="center",
                        fontsize=4.5, color="#111", fontweight="bold")

    # Two-index countries: diagonal split
    for _, row in has_two.iterrows():
        geom  = row.geometry
        col1  = INDEX_COLORS.get(row["index1"], "#cccccc")
        col2  = INDEX_COLORS.get(row["index2"], "#cccccc")
        upper, lower = _diagonal_halves(geom)
        if not upper.is_empty:
            gpd.GeoDataFrame(geometry=[upper], crs="EPSG:4326").plot(
                ax=ax, color=col1, edgecolor="none")
        if not lower.is_empty:
            gpd.GeoDataFrame(geometry=[lower], crs="EPSG:4326").plot(
                ax=ax, color=col2, edgecolor="none")
        gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").plot(
            ax=ax, facecolor="none", edgecolor="#555", linewidth=0.4)
        if not upper.is_empty:
            rp = upper.representative_point()
            ax.annotate(f"{row['trimester1']}\nL{row['lag1']}",
                        (rp.x, rp.y), ha="center", va="center",
                        fontsize=4.5, color="#111", fontweight="bold")
        if not lower.is_empty:
            rp = lower.representative_point()
            ax.annotate(f"{row['trimester2']}\nL{row['lag2']}",
                        (rp.x, rp.y), ha="center", va="center",
                        fontsize=4.5, color="#111", fontweight="bold")

    # Dot countries
    for _, row in g_dot.iterrows():
        cx, cy = _dot_center(row.geometry)
        if cy < MAP_YLIM[0] or cy > MAP_YLIM[1]:
            continue
        if pd.isna(row.get("index1")):
            if analyzed_isos and row["iso3"] in analyzed_isos:
                face, edge = "#E8E8E8", "#CCCCCC"
            else:
                face, edge = "#F8F8F8", "#EBEBEB"
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=face,
                                          edgecolor=edge, linewidth=0.5, zorder=5))
        elif pd.notna(row.get("index2")):
            col1 = INDEX_COLORS.get(row["index1"], "#cccccc")
            col2 = INDEX_COLORS.get(row["index2"], "#cccccc")
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R, -90,  90,
                                         facecolor=col1, edgecolor="#333",
                                         linewidth=0.5, zorder=5))
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R,  90, 270,
                                         facecolor=col2, edgecolor="#333",
                                         linewidth=0.5, zorder=5))
            ax.annotate(f"{row['trimester1']}/L{row['lag1']}",
                        (cx, cy + _DOT_R * 1.4), ha="center", va="bottom",
                        fontsize=3.5, color="#222", zorder=6)
        else:
            face = INDEX_COLORS.get(row["index1"], "#cccccc")
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=face,
                                          edgecolor="#333", linewidth=0.5, zorder=5))
            ax.annotate(f"{row['trimester1']}/L{row['lag1']}",
                        (cx, cy + _DOT_R * 1.4), ha="center", va="bottom",
                        fontsize=3.5, color="#222", zorder=6)

    # Legend
    handles = [
        mpatches.Patch(facecolor=INDEX_COLORS[name], edgecolor="#333",
                       linewidth=0.5, label=INDEX_LABELS[name])
        for name in INDEX_SOURCES
    ] + [
        mpatches.Patch(facecolor="#E8E8E8", edgecolor="#CCCCCC",
                       linewidth=0.5, label="No reliable signal"),
        mpatches.Patch(facecolor="#F8F8F8", edgecolor="#EBEBEB",
                       linewidth=0.5, label="Not calculated"),
    ]
    # Legend and title in HTML
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"map_dominant_{lag_tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Index correlation matrix
# --------------------------------------------------------------------------- #
def plot_index_corr_matrix(indices: pd.DataFrame, out_dir: Path, end_year: int = 2025) -> None:
    """Heatmap of Pearson r between all climate indices over the analysis period."""
    cols = [c for c in INDEX_SOURCES if c in indices.columns]
    corr = indices.loc[:str(end_year), cols].corr()
    labels = [INDEX_LABELS[c] for c in cols]
    n = len(cols)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            ax.add_patch(mpatches.Rectangle(
                (j, n - 1 - i), 1, 1,
                facecolor=cmap(norm(r)), edgecolor="white", linewidth=1,
            ))
            ax.text(j + 0.5, n - 0.5 - i, f"{r:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(r) > 0.55 else "#333")

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.75, label="Pearson r")

    ax.set_title(f"Index collinearity (Pearson r, 1981–{end_year})", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "index_corr_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Documented-impacts (literature) map
# --------------------------------------------------------------------------- #
# Dominant documented El Niño rainfall response per country in its main
# humanitarian rainy season (see the "El Niño" literature section of the report
# for the per-link sources). Curated from peer-reviewed +
# operational literature, cross-checked against this study's ERA5 results.
# Documented ENSO rainfall response per country, from the literature review
# (7 humanitarian-priority regions + a per-region sweep of the rest of the
# monitored world). Colored on the map; see the report's table for sources.
_LIT_WET = {  # El Nino wetter
    "AFG", "ARE", "ARG", "BDI", "BEN", "BHS", "BTN", "CHL", "CIV",
    "DZA", "EGY", "GHA", "IRN", "IRQ", "KAZ", "KEN", "KGZ", "KIR",
    "KWT", "LBN", "LBY", "OMN", "PAK", "PRY", "PSE", "QAT", "RWA",
    "SAU", "SOM", "SYR", "TGO", "TUN", "TUR", "UGA", "URY", "UZB"
}
_LIT_DRY = {  # El Nino drier
    "AGO", "ATG", "BES", "BFA", "BLZ", "BRB", "BWA", "CMR", "COG",
    "COL", "CPV", "CRI", "CYM", "DJI", "DMA", "DOM", "ERI", "FJI",
    "GAB", "GIN", "GLP", "GMB", "GNB", "GNQ", "GRD", "GTM", "GUF",
    "GUY", "HND", "HTI", "IDN", "IND", "JAM", "KHM", "KNA", "LAO",
    "LCA", "LSO", "MDV", "MLI", "MMR", "MOZ", "MRT", "MTQ", "MWI",
    "MYS", "NAM", "NER", "NGA", "NIC", "NPL", "PAN", "PHL", "PNG",
    "PRI", "SDN", "SEN", "SLB", "SLV", "SSD", "SUR", "SWZ", "SYC",
    "TCA", "TCD", "THA", "TLS", "TON", "TTO", "VCT", "VEN", "VGB",
    "VIR", "VNM", "VUT", "ZAF", "ZMB", "ZWE"
}
_LIT_BI = {  # opposite-sign seasons (diagonal split)
    "BOL", "BRA", "CHN", "COD", "CUB", "ECU", "ETH", "FSM", "MAR",
    "MDG", "MEX", "MHL", "PER", "TZA", "YEM"
}
_LIT_SEASON = {
    "AFG": "OND", "AGO": "DJF", "ARE": "MAM", "ARG": "NDJ", "ATG":
    "ASO", "BDI": "OND", "BEN": "ASO", "BES": "OND", "BFA": "JJAS",
    "BHS": "DJF", "BLZ": "JJA", "BRB": "ASO", "BTN": "JJA", "BWA":
    "DJF", "CHL": "JJA", "CIV": "ASO", "CMR": "JJA", "COG": "MAM",
    "COL": "DJF", "CPV": "ASO", "CRI": "JAS", "CYM": "JAS", "DJI":
    "JJAS", "DMA": "ASO", "DOM": "MJJ", "DZA": "JFM", "EGY": "JFM",
    "ERI": "JJAS", "FJI": "DJF", "GAB": "MAM", "GHA": "ASO", "GIN":
    "JJAS", "GLP": "ASO", "GMB": "JJAS", "GNB": "JJAS", "GNQ": "MAM",
    "GRD": "DJF", "GTM": "JAS", "GUF": "MAM", "GUY": "JJA", "HND":
    "JAS", "HTI": "MJJ", "IDN": "JAS", "IND": "JJAS", "IRN": "SON",
    "IRQ": "SON", "JAM": "ASO", "KAZ": "MAM", "KEN": "OND", "KGZ":
    "MAM", "KHM": "JAS", "KIR": "DJF", "KNA": "ASO", "KWT": "DJF",
    "LAO": "JAS", "LBN": "DJF", "LBY": "DJF", "LCA": "ASO", "LSO":
    "DJF", "MDV": "MJJ", "MLI": "JJAS", "MMR": "JAS", "MOZ": "DJF",
    "MRT": "JJAS", "MTQ": "ASO", "MWI": "DJF", "MYS": "JAS", "NAM":
    "DJF", "NER": "JJAS", "NGA": "JJAS", "NIC": "JAS", "NPL": "JJAS",
    "OMN": "MAM", "PAK": "OND", "PAN": "JAS", "PHL": "JAS", "PNG":
    "DJF", "PRI": "AMJ", "PRY": "OND", "PSE": "DJF", "QAT": "DJF",
    "RWA": "OND", "SAU": "MAM", "SDN": "JJAS", "SEN": "JJAS", "SLB":
    "DJF", "SLV": "JAS", "SOM": "OND", "SSD": "JJAS", "SUR": "MAM",
    "SWZ": "DJF", "SYC": "NDJ", "SYR": "SON", "TCA": "ASO", "TCD":
    "JJAS", "TGO": "ASO", "THA": "JAS", "TLS": "DJF", "TON": "DJF",
    "TTO": "DJF", "TUN": "JFM", "TUR": "DJF", "UGA": "OND", "URY":
    "OND", "UZB": "MAM", "VCT": "ASO", "VEN": "JJA", "VGB": "ASO",
    "VIR": "ASO", "VNM": "JAS", "VUT": "DJF", "ZAF": "DJF", "ZMB":
    "DJF", "ZWE": "DJF"
}
_LIT_BI_SEASON = {
    "BOL": ("DJF", "DJF"), "BRA": ("OND", "FMA"), "CHN": ("JJA", "JJA"),
    "COD": ("OND", "MAM"), "CUB": ("DJF", "JAS"), "ECU": ("JFM", "MAM"),
    "ETH": ("OND", "JJAS"), "FSM": ("OND", "MAM"), "MAR": ("NDJ",
    "NDJ"), "MDG": ("NDJ", "DJF"), "MEX": ("DJF", "JJA"), "MHL": ("OND",
    "FMA"), "PER": ("DJF", "DJF"), "TZA": ("OND", "DJF"), "YEM": ("DJF",
    "JJA")
}

# Full monitored universe (the 153 ERA5-analysis countries). Monitored
# countries with no documented link are drawn light grey; non-monitored
# countries (not researched, e.g. USA / most of Europe) are near-white.
_LIT_SEARCHED = {
    "AFG", "AGO", "ALB", "ARE", "ARG", "ARM", "ATG", "AZE", "BDI",
    "BEN", "BES", "BFA", "BGD", "BGR", "BHS", "BLR", "BLZ", "BMU",
    "BOL", "BRA", "BRB", "BTN", "BWA", "CAF", "CHL", "CHN", "CIV",
    "CMR", "COD", "COG", "COL", "COM", "CPV", "CRI", "CUB", "CYM",
    "DJI", "DMA", "DOM", "DZA", "ECU", "EGY", "ERI", "ESH", "ETH",
    "FJI", "FSM", "GAB", "GEO", "GHA", "GIN", "GLP", "GMB", "GNB",
    "GNQ", "GRD", "GTM", "GUF", "GUY", "HND", "HTI", "HUN", "IDN",
    "IRN", "IRQ", "JAM", "KAZ", "KEN", "KGZ", "KHM", "KIR", "KNA",
    "KWT", "LAO", "LBN", "LBR", "LBY", "LCA", "LKA", "LSO", "MAR",
    "MDA", "MDG", "MDV", "MEX", "MHL", "MLI", "MMR", "MNG", "MOZ",
    "MRT", "MTQ", "MUS", "MWI", "MYS", "NAM", "NER", "NGA", "NIC",
    "NPL", "OMN", "PAK", "PAN", "PER", "PHL", "PNG", "POL", "PRI",
    "PRK", "PRY", "PSE", "QAT", "ROU", "RUS", "RWA", "SAU", "SDN",
    "SEN", "SLB", "SLE", "SLV", "SOM", "SSD", "STP", "SUR", "SVK",
    "SWZ", "SYC", "SYR", "TCA", "TCD", "TGO", "THA", "TLS", "TON",
    "TTO", "TUN", "TUR", "TZA", "UGA", "UKR", "URY", "UZB", "VCT",
    "VEN", "VGB", "VIR", "VNM", "VUT", "YEM", "ZAF", "ZMB", "ZWE"
}

_BLUE, _BLUE_E = "#86BCE8", "#5E9FD2"    # wetter
_BROWN, _BROWN_E = "#D29A6C", "#B17E50"  # drier
_GREY_NE, _GREY_NE_E = "#DBDBDB", "#BDBDBD"    # monitored, no documented link
_GREY_NS, _GREY_NS_E = "#FCFCFC", "#DCDCDC"    # not monitored (near-white)

# Per-country evidence strength -> map shade (robust dark, single-study pale)
_LIT_EVIDENCE = {
    "AFG": "moderate", "AGO": "robust", "ARE": "moderate", "ARG":
    "robust", "ATG": "moderate", "BDI": "moderate", "BEN": "single-study",
    "BES": "moderate", "BFA": "moderate", "BHS": "moderate", "BLZ":
    "robust", "BOL": "robust", "BRA": "robust", "BRB": "moderate", "BTN":
    "moderate", "BWA": "robust", "CHL": "robust", "CHN": "robust", "CIV":
    "moderate", "CMR": "moderate", "COD": "moderate", "COG": "moderate",
    "COL": "robust", "CPV": "moderate", "CRI": "robust", "CUB": "robust",
    "CYM": "single-study", "DJI": "robust", "DMA": "single-study", "DOM":
    "robust", "DZA": "moderate", "ECU": "robust", "EGY": "moderate",
    "ERI": "robust", "ETH": "robust", "FJI": "robust", "FSM": "robust",
    "GAB": "moderate", "GHA": "moderate", "GIN": "moderate", "GLP":
    "single-study", "GMB": "moderate", "GNB": "moderate", "GNQ":
    "single-study", "GRD": "single-study", "GTM": "robust", "GUF":
    "single-study", "GUY": "moderate", "HND": "robust", "HTI": "robust",
    "IDN": "robust", "IND": "moderate", "IRN": "moderate", "IRQ":
    "moderate", "JAM": "robust", "KAZ": "robust", "KEN": "robust", "KGZ":
    "robust", "KHM": "robust", "KIR": "robust", "KNA": "single-study",
    "KWT": "moderate", "LAO": "robust", "LBN": "moderate", "LBY":
    "single-study", "LCA": "single-study", "LSO": "robust", "MAR":
    "moderate", "MDG": "moderate", "MDV": "moderate", "MEX": "robust",
    "MHL": "robust", "MLI": "moderate", "MMR": "robust", "MOZ": "robust",
    "MRT": "moderate", "MTQ": "single-study", "MWI": "robust", "MYS":
    "robust", "NAM": "robust", "NER": "moderate", "NGA": "moderate",
    "NIC": "robust", "OMN": "moderate", "PAK": "moderate", "PAN":
    "robust", "PER": "robust", "PHL": "robust", "PNG": "robust", "PRI":
    "robust", "PRY": "robust", "PSE": "single-study", "QAT":
    "single-study", "RWA": "moderate", "SAU": "moderate", "SDN": "robust",
    "SEN": "moderate", "SLB": "robust", "SLV": "robust", "SOM": "robust",
    "SSD": "robust", "SUR": "moderate", "SWZ": "robust", "SYC":
    "single-study", "SYR": "moderate", "TCA": "single-study", "TCD":
    "moderate", "TGO": "single-study", "THA": "robust", "TLS": "robust",
    "TON": "robust", "TTO": "moderate", "TUN": "moderate", "TUR":
    "robust", "TZA": "robust", "UGA": "robust", "URY": "robust", "UZB":
    "moderate", "VCT": "single-study", "VEN": "moderate", "VGB":
    "single-study", "VIR": "moderate", "VNM": "robust", "VUT": "robust",
    "YEM": "single-study", "ZAF": "robust", "ZMB": "robust", "ZWE":
    "robust"
}
_RAMP = {
    # (direction, evidence) -> (fill, edge).  direction 'wet' = El Nino wetter.
    ("wet", "robust"):       ("#2E7DBD", "#1F5F96"),
    ("wet", "moderate"):     ("#86BCE8", "#5E9FD2"),
    ("wet", "single-study"): ("#D4E7F7", "#5E9FD2"),
    ("dry", "robust"):       ("#9C6730", "#7A4E22"),
    ("dry", "moderate"):     ("#D29A6C", "#B17E50"),
    ("dry", "single-study"): ("#F0DAC2", "#B17E50"),
}


def plot_literature_enso_map(gdf: gpd.GeoDataFrame, out_dir: Path,
                             phase: str = "elnino") -> None:
    """Choropleth of documented ENSO rainfall impacts, in the same visual style
    as the correlation maps, annotated with each signal's season. ``phase`` is
    'elnino' or 'lanina'; the La Niña map is the typical (approximate) opposite,
    since ENSO is asymmetric. Saved as maps/lit_enso_{phase}.png."""
    flip = phase == "lanina"

    def _fill(direction, iso):
        """(face, edge) by direction ('wet'/'dry'), country evidence, and phase."""
        eff = ("dry" if direction == "wet" else "wet") if flip else direction
        ev = _LIT_EVIDENCE.get(iso, "moderate")
        return _RAMP.get((eff, ev), _RAMP[(eff, "moderate")])

    is_dot = gdf.geometry.apply(_is_dot_country)
    g_poly, g_dot = gdf[~is_dot], gdf[is_dot]

    map_w = 15.0
    plt.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(figsize=(map_w, map_w * _MAP_DY / _MAP_DX))

    # Base layers: not-searched countries (near-white, fine outline) and
    # searched-but-no-documented-effect countries (light grey).
    not_searched = g_poly[~g_poly["iso3"].isin(_LIT_SEARCHED)]
    if len(not_searched):
        not_searched.plot(ax=ax, color=_GREY_NS, edgecolor=_GREY_NS_E, linewidth=0.3)
    documented = _LIT_WET | _LIT_DRY | _LIT_BI
    no_effect = g_poly[g_poly["iso3"].isin(_LIT_SEARCHED - documented)]
    if len(no_effect):
        no_effect.plot(ax=ax, color=_GREY_NE, edgecolor=_GREY_NE_E, linewidth=0.3)

    wet_poly = g_poly[g_poly["iso3"].isin(_LIT_WET)]
    dry_poly = g_poly[g_poly["iso3"].isin(_LIT_DRY)]
    # Shade by evidence strength (robust = strong, single-study = pale)
    for direction, sub in (("wet", wet_poly), ("dry", dry_poly)):
        for ev in ("robust", "moderate", "single-study"):
            part = sub[sub["iso3"].map(lambda i: _LIT_EVIDENCE.get(i, "moderate")) == ev]
            if len(part):
                eff = ("dry" if direction == "wet" else "wet") if flip else direction
                face, edge = _RAMP.get((eff, ev), _RAMP[(eff, "moderate")])
                hatch = "///" if ev == "single-study" else None
                part.plot(ax=ax, color=face, edgecolor=edge, linewidth=0.3, hatch=hatch)

    def _label(x, y, text):
        ax.annotate(text, (x, y), ha="center", va="center", fontsize=5.0,
                    color="black", zorder=7)

    # Unidirectional season labels
    for _, row in pd.concat([wet_poly, dry_poly]).iterrows():
        season = _LIT_SEASON.get(row["iso3"])
        if not season:
            continue
        p = row.geometry.representative_point()
        _label(p.x, p.y, season)

    # Bidirectional: diagonal split (upper half = the country's "wet season"
    # under El Niño) + a season label in each half. Colors follow the phase.
    for _, row in g_poly[g_poly["iso3"].isin(_LIT_BI)].iterrows():
        geom = row.geometry
        upper, lower = _diagonal_halves(geom)
        if not upper.is_empty:
            gpd.GeoDataFrame(geometry=[upper], crs="EPSG:4326").plot(
                ax=ax, color=_fill("wet", row["iso3"])[0], edgecolor="none")
        if not lower.is_empty:
            gpd.GeoDataFrame(geometry=[lower], crs="EPSG:4326").plot(
                ax=ax, color=_fill("dry", row["iso3"])[0], edgecolor="none")
        gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").plot(
            ax=ax, facecolor="none", edgecolor="#555", linewidth=0.4)
        if _LIT_EVIDENCE.get(row["iso3"]) == "single-study":
            gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="#8a8a8a", linewidth=0.0, hatch="///")
        wet_s, dry_s = _LIT_BI_SEASON.get(row["iso3"], ("", ""))
        if not upper.is_empty and wet_s:
            rp = upper.representative_point()
            _label(rp.x, rp.y, wet_s)
        if not lower.is_empty and dry_s:
            rp = lower.representative_point()
            _label(rp.x, rp.y, dry_s)

    def _dot_label(cx, cy, text):
        if text:
            ax.annotate(text, (cx, cy + _DOT_R * 1.5), ha="center", va="bottom",
                        fontsize=3.8, color="black", zorder=7)

    for _, row in g_dot.iterrows():
        iso = row["iso3"]
        cx, cy = _dot_center(row.geometry)
        if cy < MAP_YLIM[0] or cy > MAP_YLIM[1]:
            continue
        if iso in _LIT_BI:
            # split dot: right half = wet season, left half = dry season
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R, -90, 90,
                                        facecolor=_fill("wet", iso)[0], edgecolor="#555",
                                        linewidth=0.5, zorder=5))
            ax.add_patch(mpatches.Wedge((cx, cy), _DOT_R, 90, 270,
                                        facecolor=_fill("dry", iso)[0], edgecolor="#555",
                                        linewidth=0.5, zorder=5))
            wet_s, dry_s = _LIT_BI_SEASON.get(iso, ("", ""))
            _dot_label(cx, cy, "/".join(s for s in (wet_s, dry_s) if s))
        elif iso in _LIT_WET or iso in _LIT_DRY:
            face, edge = _fill("wet" if iso in _LIT_WET else "dry", iso)
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=face,
                                         edgecolor=edge, linewidth=0.5, zorder=5))
            _dot_label(cx, cy, _LIT_SEASON.get(iso, ""))
        elif iso in _LIT_SEARCHED:
            # monitored, no documented link
            ax.add_patch(mpatches.Circle((cx, cy), _DOT_R, facecolor=_GREY_NE,
                                         edgecolor=_GREY_NE_E, linewidth=0.5, zorder=5))

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"lit_enso_{phase}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def generate_html_report(cfg: dict) -> None:
    """Write docs/index.html with all 14 maps embedded."""
    docs_dir: Path = cfg["docs_dir"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    end_year = cfg["end_year"]

    corr_legend = """<div class="map-legend">
              <span><span class="sw" style="background:#0D40B0;border-color:#092E88"></span>Positive strong (r≥0.45)</span>
              <span><span class="sw" style="background:#71B3E5;border-color:#4A90C8"></span>Positive moderate (0.30–0.45)</span>
              <span><span class="sw" style="background:#C8844A;border-color:#A06030"></span>Negative moderate</span>
              <span><span class="sw" style="background:#7B3A1A;border-color:#5A2A0A"></span>Negative strong (r≤−0.45)</span>
              <span><span class="sw" style="background:#E8E8E8;border-color:#CCCCCC"></span>No signal</span>
              <span><span class="sw" style="background:#F8F8F8;border-color:#EBEBEB"></span>Not calculated</span>
            </div>"""

    # caption tail is resolution-specific: the country maps split a polygon
    # diagonally for a both-signs country, the pixel maps hatch the cells.
    _bidir_note = {
        "adm0": " Split diagonal = significant in both directions across non-overlapping seasons.",
        "px":   " Hatching = significant in both directions across non-overlapping seasons;"
                " the fill shows the stronger of the two.",
    }
    _kind_meta = {
        "total":   ("total association",
                    "<strong>Total association</strong> — pairwise Pearson r, p&lt;0.05."),
        "partial": ("unique signal",
                    "<strong>Unique signal</strong> — partial r, other climate modes held constant. "
                    "Shrinkage vs Total = shared variance, not absent signal."),
    }
    # Pixel maps share the correlation bins but have their own base categories
    # (ocean vs arid land vs analysed-no-signal) and use a hatch, not a split
    # diagonal, to flag cells significant in both directions.
    px_corr_legend = f"""<div class="map-legend">
              <span><span class="sw" style="background:#0D40B0;border-color:#092E88"></span>Positive strong (r≥0.45)</span>
              <span><span class="sw" style="background:#71B3E5;border-color:#4A90C8"></span>Positive moderate (0.30–0.45)</span>
              <span><span class="sw" style="background:#C8844A;border-color:#A06030"></span>Negative moderate</span>
              <span><span class="sw" style="background:#7B3A1A;border-color:#5A2A0A"></span>Negative strong (r≤−0.45)</span>
              <span><span class="sw hatch-sw"></span>Both signs across seasons</span>
              <span><span class="sw" style="background:{_PX_NOSIG};border-color:#CCCCCC"></span>No signal</span>
              <span><span class="sw" style="background:{_PX_ARID};border-color:#DED8C9"></span>Arid — no wet season</span>
              <span><span class="sw" style="background:{_PX_OCEAN};border-color:#D5D5D5"></span>Ocean / outside grid</span>
            </div>"""

    _lag_label = {"l3": "0–3 mo lag", "l6": "0–6 mo lag"}
    _res_label = {"adm0": "country mean", "px": "0.25° grid cell"}
    # Land-cell count of the pixel grid, read back from the cached run so the
    # methodology text always quotes the number actually analysed even when the
    # pixel pass was skipped this run.
    _px_cells_hint = cfg.get("px_cells_hint")
    if _px_cells_hint is None:
        _cells_json = cfg["cache_dir"] / "era5_pixel" / "cells.json"
        _px_cells_hint = (
            f'{json.loads(_cells_json.read_text())["n_cells"]:,}'
            if _cells_json.exists() else "all"
        )

    def _map_item(name, label, kind, lag_tag, res="adm0"):
        kind_title, caption = _kind_meta[kind]
        img = (f"maps/map_{lag_tag}_{kind}_{name}.png" if res == "adm0"
               else f"maps/map_px_{lag_tag}_{kind}_{name}.png")
        legend = corr_legend if res == "adm0" else px_corr_legend
        extra = _bidir_note[res] if kind == "total" else ""
        return f"""
      <div class="map-item" data-kind="{kind}" data-lag="{lag_tag}" data-res="{res}">
        <div class="map-with-ts">
          <div class="ts-col"><img style="width:100%;height:auto;display:block;" src="maps/ts_{name}.png" alt="{label} historical values"></div>
          <div class="map-col">
            <p class="map-title">Correlation of {label} with total seasonal rainfall — {kind_title} ({_lag_label[lag_tag]}, {_res_label[res]})</p>
            <div class="map-zoom"><img src="{img}" alt="{label} {kind} correlation {lag_tag} {res}"></div>
            {legend}
          </div>
        </div>
        <p>{caption}{extra}</p>
      </div>"""

    index_sections = []
    for name in INDEX_SOURCES:
        label = INDEX_LABELS.get(name, name.upper())
        items = "".join(
            _map_item(name, label, kind, lag_tag, res)
            for res in RES_VARIANTS
            for lag_tag in LAG_CAPS
            for kind in ("total", "partial")
        )
        index_sections.append(f"""
  <section>
    <h2>{label}</h2>
    <div class="map-pair">{items}
    </div>
  </section>""")

    # Dominant-mode map legend, derived from the actual INDEX_COLORS so swatches match.
    def _edge(hex_c):  # slightly darker border
        h = hex_c.lstrip("#")
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return "#" + "".join(f"{int(c*0.7):02X}" for c in rgb)
    dom_swatches = "".join(
        f'<span><span class="sw" style="background:{INDEX_COLORS[n]};'
        f'border-color:{_edge(INDEX_COLORS[n])}"></span>{INDEX_LABELS[n]}</span>'
        for n in INDEX_SOURCES
    )
    dominant_legend = f"""<div class="map-legend">
        {dom_swatches}
        <span><span class="sw" style="background:#E8E8E8;border-color:#CCCCCC"></span>No signal</span>
        <span><span class="sw" style="background:#F8F8F8;border-color:#EBEBEB"></span>Not calculated</span>
      </div>"""
    px_dominant_legend = f"""<div class="map-legend">
        {dom_swatches}
        <span><span class="sw" style="background:{_PX_TIED};border-color:#AEB6C0"></span>No single dominant mode (top two within {PIXEL_DOMINANT_MARGIN:.2f} of each other)</span>
        <span><span class="sw" style="background:{_PX_NOSIG};border-color:#CCCCCC"></span>No signal</span>
        <span><span class="sw" style="background:{_PX_ARID};border-color:#DED8C9"></span>Arid — no wet season</span>
        <span><span class="sw" style="background:{_PX_OCEAN};border-color:#D5D5D5"></span>Ocean / outside grid</span>
      </div>"""
    dominant_items = "".join(
        f"""<div class="map-item" data-lag="{lag_tag}" data-res="adm0" style="max-width:100%">
      <div class="map-zoom"><img src="maps/map_dominant_{lag_tag}.png" alt="Dominant climate mode map {lag_tag}"></div>
      {dominant_legend}
    </div>"""
        for lag_tag in LAG_CAPS
    ) + "".join(
        f"""<div class="map-item" data-lag="{lag_tag}" data-res="px" style="max-width:100%">
      <div class="map-zoom"><img src="maps/map_px_dominant_{lag_tag}.png" alt="Dominant climate mode map, pixel {lag_tag}"></div>
      {px_dominant_legend}
      <p>Top mode only — the runner-up mode shown as a split diagonal on the country map is not legible at 0.25°. A cell is
      coloured only where the leading mode's |r| exceeds the runner-up's by at least {PIXEL_DOMINANT_MARGIN:.2f}; a single
      grid cell does not average enough area to make the arg-max over six collinear modes stable, so cells where the top two
      are within noise of each other are left grey rather than assigned to an arbitrary winner.</p>
    </div>"""
        for lag_tag in LAG_CAPS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERA5 Precipitation Teleconnection Analysis</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 0.75rem 1rem;
      background: #f8f9fa;
      color: #1a1a1a;
      line-height: 1.5;
    }}
    h1 {{ font-size: 1.5rem; border-bottom: 3px solid #005a9c; padding-bottom: 0.5rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.05rem; color: #005a9c; margin-top: 2rem; margin-bottom: 0.5rem; }}
    .meta {{ font-size: 0.8rem; color: #666; margin-bottom: 0.5rem; }}
    .note {{ font-size: 0.82rem; color: #444; background: #e8f0fa; border-left: 3px solid #005a9c; padding: 0.5rem 0.75rem; margin-bottom: 1.5rem; }}
    .map-pair {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    @media (max-width: 900px) {{ .map-pair {{ grid-template-columns: 1fr; }} }}
    .map-item {{ padding: 0.2rem 0; }}
    .map-with-ts {{ display: flex; align-items: flex-start; gap: 0; }}
    .map-col {{ flex: 1; min-width: 0; display: flex; flex-direction: column; background: white; border: 1px solid #dde3ec; border-radius: 4px; padding: 0.5rem; }}
    .ts-col {{ flex: 0 0 30%; max-width: 30%; display: flex; flex-direction: column; background: white; border: 1px solid #dde3ec; border-radius: 4px; padding: 0.5rem; margin-right: 0.4rem; }}
    .map-zoom {{ overflow: hidden; position: relative; line-height: 0; border-radius: 2px; }}
    .map-zoom img {{ width: 100%; height: auto; display: block; transform-origin: 0 0; cursor: default; }}
    .map-title {{ font-size: 0.88rem; font-weight: 600; color: #1a1a1a; margin: 0 0 0.4rem; }}
    .map-legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; font-size: 0.72rem; color: #444; margin: 0.35rem 0 0.1rem; align-items: center; }}
    .sw {{ display: inline-block; width: 1em; height: 1em; border: 1px solid; vertical-align: middle; margin-right: 0.25em; border-radius: 2px; }}
    .hatch-sw {{ border-color: #999; background:
      repeating-linear-gradient(45deg, #333 0 1px, transparent 1px 3px), #E8E8E8; }}
    .map-item p {{ font-size: 0.75rem; color: #555; margin: 0.4rem 0 0 0; }}
    .map-item p strong {{ color: #222; }}
    section {{ margin-bottom: 2rem; }}
    .enso-note {{ font-size: 0.82rem; color: #444; margin-bottom: 0.5rem; }}
    hr {{ border: none; border-top: 1px solid #dde3ec; margin: 2rem 0; }}
    .view-toggle {{
      position: sticky; top: 0; z-index: 100;
      display: flex; flex-wrap: wrap; gap: 0.4rem 0.5rem; align-items: center;
      background: #f8f9fa; padding: 0.5rem 0; margin-bottom: 1rem;
      border-bottom: 1px solid #dde3ec;
    }}
    .view-toggle span {{ font-size: 0.82rem; color: #555; margin-right: 0.25rem; }}
    .toggle-btn {{
      padding: 0.3rem 0.85rem; border: 1px solid #005a9c;
      background: white; color: #005a9c; border-radius: 4px;
      cursor: pointer; font-size: 0.82rem; transition: background 0.15s;
    }}
    .toggle-btn.active {{ background: #005a9c; color: white; }}
    .toggle-btn:hover:not(.active) {{ background: #e8f0fa; }}
    h1, h2 {{ scroll-margin-top: 0.5rem; }}
    .anchor-link {{ margin-left: 0.4rem; color: #b3bccb; text-decoration: none; font-weight: 400; opacity: 0; transition: opacity 0.12s; }}
    h1:hover .anchor-link, h2:hover .anchor-link {{ opacity: 1; }}
    .anchor-link:hover {{ color: #005a9c; }}
  </style>
</head>
<body>
  <h1>ERA5 Precipitation Teleconnection Analysis</h1>
  <p class="meta">ERA5 1981–{end_year} &nbsp;·&nbsp; Pearson r &nbsp;·&nbsp; p &lt; 0.05 &nbsp;·&nbsp; Gray = no reliable signal &nbsp;·&nbsp; Split diagonal (country) / hatching (pixel) = both signs across seasons</p>
  <div class="view-toggle">
    <span>Resolution:</span>
    <button class="toggle-btn active" data-group="res" data-show="adm0">Country (ADM0)</button>
    <button class="toggle-btn" data-group="res" data-show="px">Pixel (0.25°)</button>
    <span style="margin-left:1.25rem;">View:</span>
    <button class="toggle-btn" data-group="view" data-show="total">Total association</button>
    <button class="toggle-btn active" data-group="view" data-show="partial">Unique signal</button>
    <span style="margin-left:1.25rem;">Max lag:</span>
    <button class="toggle-btn active" data-group="lag" data-show="l3">3 months</button>
    <button class="toggle-btn" data-group="lag" data-show="l6">6 months</button>
  </div>
  <div class="note">
    Each map shows the <strong>strongest significant correlation</strong> between a climate index and trimester rainfall across all countries.
    <strong>Total</strong> maps show the raw pairwise association (including signal shared with other modes).
    <strong>Unique signal</strong> maps show the partial correlation, holding other modes constant —
    shrinkage from Total to Unique indicates that modes share variance, not that the signal is absent.
    The <strong>Max lag</strong> toggle limits how far the index may lead rainfall: 3 months keeps the
    index within one preceding non-overlapping season (cleaner, forecast-relevant); 6 months also admits
    longer prior-season relationships. Trimester labels follow the start-of-season convention
    (DJF 2024 = Dec 2024 – Feb 2025).
    The <strong>Resolution</strong> toggle switches the unit of analysis: <em>Country</em> runs the method on
    ERA5 admin-0 means (one series per country, 153 countries), <em>Pixel</em> runs the identical method
    independently on every ERA5 land cell at 0.25° (~0.25° ≈ 28 km at the equator). The pixel maps expose
    sub-national structure and gradients that a country mean averages away — but each cell is a single
    ~45-year series with no spatial pooling, so read coherent regions rather than isolated cells.
  </div>

{"".join(index_sections)}
  <hr>
  <section>
    <h2>Index Collinearity</h2>
    <p class="enso-note">Pearson r between climate indices over the full analysis period (1981–{end_year}). High collinearity (e.g. AMM–TNA r≈0.81) means total vs unique signal may diverge substantially for those modes — shrinkage in the unique-signal map reflects shared variance, not absent signal.</p>
    <div class="map-item" style="max-width:640px">
      <img src="maps/index_corr_matrix.png" style="width:100%;height:auto;display:block;border-radius:2px;" alt="Index collinearity matrix">
    </div>
  </section>

  <hr>
  <section>
    <h2>Dominant Climate Mode</h2>
    <p class="enso-note" data-res="adm0">Each country colored by the index with the strongest significant total correlation (|r|≥{_R_MIN}). Split diagonal = second-strongest index (different mode), shown only when it acts on a non-overlapping trimester. Respects the Max lag toggle above.</p>
    <p class="enso-note" data-res="px">Each 0.25° land cell colored by the index with the strongest significant total correlation (|r|≥{_R_MIN}), where that mode leads the runner-up by at least {PIXEL_DOMINANT_MARGIN:.2f}. Respects the Max lag toggle above.</p>
    {dominant_items}
  </section>

  <hr>
  <section>
    <h2>ENSO Composites — El Niño &amp; La Niña</h2>
    <p class="enso-note">Mean rainfall anomaly (standard deviations from climatology) in each unit's headline trimester when Niño3.4 ≥ ±0.5 concurrent with that trimester. The two maps are independent — ENSO impacts are asymmetric. Respects the Resolution toggle above.</p>
    <div class="map-pair">
      <div class="map-item" data-res="adm0">
        <div class="map-zoom"><img src="maps/enso_elnino.png" alt="El Niño composite rainfall anomaly"></div>
        <p><strong>El Niño composite</strong> — mean anomaly when Niño3.4 ≥ +0.5 (brown = drier than normal, blue = wetter).</p>
      </div>
      <div class="map-item" data-res="adm0">
        <div class="map-zoom"><img src="maps/enso_lanina.png" alt="La Niña composite rainfall anomaly"></div>
        <p><strong>La Niña composite</strong> — mean anomaly when Niño3.4 ≤ −0.5. Roughly opposite to El Niño in ENSO-sensitive regions, but magnitude and pattern differ.</p>
      </div>
      <div class="map-item" data-res="px">
        <div class="map-zoom"><img src="maps/enso_px_elnino.png" alt="El Niño composite rainfall anomaly, pixel"></div>
        <p><strong>El Niño composite (0.25° grid)</strong> — mean anomaly when Niño3.4 ≥ +0.5 (brown = drier than normal, blue = wetter), scored per cell in its own headline season.</p>
      </div>
      <div class="map-item" data-res="px">
        <div class="map-zoom"><img src="maps/enso_px_lanina.png" alt="La Niña composite rainfall anomaly, pixel"></div>
        <p><strong>La Niña composite (0.25° grid)</strong> — mean anomaly when Niño3.4 ≤ −0.5.</p>
      </div>
    </div>
  </section>

  <hr>
  <section>
    <h2>Reference — Typical Global ENSO Impacts</h2>
    <p class="enso-note">
      Generalized seasonal impact patterns associated with El Niño and La Niña, for context.
      These are schematic climatological composites, not derived from this analysis. Each map shows
      Northern Hemisphere winter (top) and summer (bottom). Source:
      <a href="https://www.pmel.noaa.gov/elnino/impacts-of-el-nino" target="_blank" rel="noopener">NOAA PMEL</a>,
      via climate.gov.
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;max-width:1140px;margin:0 auto;">
      <div class="map-item">
        <img src="maps/enso_impacts_elnino_climategov.jpg" style="width:100%;height:auto;display:block;border-radius:2px;" alt="Typical global El Niño impacts for winter and summer">
        <p><strong>El Niño</strong> — typical global impacts for winter (top) and summer (bottom). Credit: climate.gov.</p>
      </div>
      <div class="map-item">
        <img src="maps/enso_impacts_lanina_climategov.jpg" style="width:100%;height:auto;display:block;border-radius:2px;" alt="Typical global La Niña impacts for winter and summer">
        <p><strong>La Niña</strong> — typical global impacts for winter (top) and summer (bottom). Credit: climate.gov.</p>
      </div>
    </div>
  </section>

  <hr>
  <section>
    <h2>Documented ENSO Impacts (Literature)</h2>
    <p class="enso-note">
      The generalized NOAA and IRI impact maps above are built from a few canonical global studies and miss
      several real teleconnections — most notably summer rainfall over <strong>highland Yemen and the
      southwestern Arabian Peninsula</strong>. The maps and table below are a curated, citation-backed catalogue
      of documented ENSO–seasonal-rainfall links covering <strong>all 153 monitored countries</strong> (those in
      the ERA5 analysis), researched region by region from peer-reviewed literature plus authoritative operational
      sources (NOAA, IRI, ICPAC/IGAD). Rows tagged
      <span style="background:#7B3A1A;color:#fff;font-size:0.62rem;padding:0.05rem 0.3rem;border-radius:3px;">gap vs NOAA/IRI</span>
      are well-supported in the literature yet absent or under-represented on the standard maps. The final
      column reports what <strong>this study's own ERA5 analysis</strong> found, as an independent check.
    </p>
    <div style="display:flex;gap:0.5rem;margin:0.2rem 0 0.6rem;">
      <button class="toggle-btn ensoph-btn active" data-phase="elnino">El Niño</button>
      <button class="toggle-btn ensoph-btn" data-phase="lanina">La Niña</button>
    </div>
    <div class="ensoph" data-phase="elnino">
      <div class="map-zoom" style="max-width:1140px;margin:0 auto;"><img src="maps/lit_enso_elnino.png" alt="Documented El Niño rainfall impacts across humanitarian-priority regions"></div>
      <p class="enso-note" style="margin:0.3rem auto 0;max-width:1140px;font-size:0.73rem;color:#666;">Showing <strong>El&nbsp;Niño</strong> — dominant documented rainfall response in each country's main humanitarian rainy season.</p>
    </div>
    <div class="ensoph" data-phase="lanina" style="display:none;">
      <div class="map-zoom" style="max-width:1140px;margin:0 auto;"><img src="maps/lit_enso_lanina.png" alt="Typical La Niña rainfall impacts across humanitarian-priority regions"></div>
      <p class="enso-note" style="margin:0.3rem auto 0;max-width:1140px;font-size:0.73rem;color:#666;">Showing <strong>La&nbsp;Niña</strong> — the typical (approximate) opposite of El&nbsp;Niño; ENSO is asymmetric, so the real La&nbsp;Niña pattern can differ in magnitude.</p>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:0.35rem 0.9rem;font-size:0.72rem;color:#444;margin:0.5rem 0 0.2rem;align-items:center;"><span style="color:#555;">El&nbsp;Niño wetter:</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #6FA8D6;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);color:#1a1a1a;">single study</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #5E9FD2;background:#86BCE8;color:#1a1a1a;">moderate</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #1F5F96;background:#2E7DBD;color:#1a1a1a;">robust</span><span style="color:#555;margin-left:0.5em;">drier:</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #C49A6E;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);color:#1a1a1a;">single study</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #B17E50;background:#D29A6C;color:#1a1a1a;">moderate</span><span style="display:inline-block;padding:0.05rem 0.4rem;border-radius:3px;font-size:0.66rem;border:1px solid #7A4E22;background:#9C6730;color:#1a1a1a;">robust</span><span style="color:#777;">(shade = confidence)</span><span><span style="display:inline-block;width:0.9em;height:0.9em;border-radius:2px;vertical-align:middle;margin-right:0.3em;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);border:1px solid #777;"></span>Opposing signals (split)</span><span><span style="display:inline-block;width:0.9em;height:0.9em;border-radius:2px;vertical-align:middle;margin-right:0.3em;background:#DBDBDB;border:1px solid #BDBDBD;"></span>Monitored — no documented link</span><span><span style="display:inline-block;width:0.9em;height:0.9em;border-radius:2px;vertical-align:middle;margin-right:0.3em;background:#FCFCFC;border:1px solid #DCDCDC;"></span>Not monitored (not researched)</span></div>
    <p class="enso-note" style="margin:0.2rem auto 0.9rem;max-width:1140px;font-size:0.73rem;color:#666;">
      Countries whose seasons respond in opposite directions (e.g. Ethiopia, Yemen, Brazil, Peru, China) are
      split; 3-month season codes (e.g. OND, JJAS, DJF) label each signal. <strong>Grey</strong> = monitored but
      no robust documented ENSO link in the literature; <strong>near-white</strong> = not monitored (not
      researched — e.g. USA, most of Europe). The curated highlights are in the table; <strong>every monitored
      country is in the collapsible catalogue below</strong>.
    </p>
<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;font-size:0.78rem;line-height:1.4;min-width:760px;"><thead><tr><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">Region / area</th><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">Season</th><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">El&nbsp;Niño</th><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">La&nbsp;Niña</th><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">Evidence &amp; source</th><th style="text-align:left;padding:0.35rem 0.5rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;">This study (ERA5)</th></tr></thead><tbody><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Horn of Africa</span><br><span style="color:#222;">Kenya, S. Somalia, SE Ethiopia</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">OND &#8220;short rains&#8221;</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Robust, widely replicated<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033121" target="_blank" rel="noopener" style="color:#2563b8;">Park et&nbsp;al. 2020 (JGR-A)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — SOM OND r=+0.65, KEN +0.58, ETH +0.59</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Horn of Africa</span><br><span style="color:#222;">Ethiopia / Sudan / S. Sudan highlands</span> <span style="background:#7B3A1A;color:#fff;font-size:0.62rem;padding:0.05rem 0.3rem;border-radius:3px;white-space:nowrap;">gap vs NOAA/IRI</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJAS (kiremt / Blue Nile)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Robust<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — ETH JAS r=&#8722;0.63, SSD &#8722;0.59, SDN &#8722;0.40</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Horn of Africa</span><br><span style="color:#222;">East Africa</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">MAM &#8220;long rains&#8221;</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#E8E8E8;text-align:center;color:#555;">weak / none</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#E8E8E8;text-align:center;color:#555;">weak / none</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Long rains only weakly tied to ENSO<br><a href="https://link.springer.com/article/10.1007/s00382-018-4239-7" target="_blank" rel="noopener" style="color:#2563b8;">Wainwright et&nbsp;al. 2018 (Clim&nbsp;Dyn)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — little MAM ENSO signal in the Horn</td></tr><tr style="background:#ffffff;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Sahel &amp; W. Africa</span><br><span style="color:#222;">Senegal, Mali, Niger, Chad, N. Nigeria</span> <span style="background:#7B3A1A;color:#fff;font-size:0.62rem;padding:0.05rem 0.3rem;border-radius:3px;white-space:nowrap;">gap vs NOAA/IRI</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJAS monsoon</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Direction robust; mechanism contested<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — SEN JJA &#8722;0.49, MLI JAS &#8722;0.41, TCD &#8722;0.43</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Middle East / Arabia</span><br><span style="color:#222;">SW Arabia / highland Yemen</span> <span style="background:#7B3A1A;color:#fff;font-size:0.62rem;padding:0.05rem 0.3rem;border-radius:3px;white-space:nowrap;">gap vs NOAA/IRI</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJA summer rains</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier (drought)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Single robust modelling+obs study; <strong>direct</strong> Pacific teleconnection<br><a href="https://www.nature.com/articles/s41612-017-0003-7" target="_blank" rel="noopener" style="color:#2563b8;">Atif et&nbsp;al. 2017 (npj&nbsp;Clim&nbsp;Atmos&nbsp;Sci)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — YEM JAS r=&#8722;0.59 (summer); winter NDJ +0.53</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Middle East / Arabia</span><br><span style="color:#222;">Iraq, Iran, Levant, N. Arabia</span> <span style="background:#7B3A1A;color:#fff;font-size:0.62rem;padding:0.05rem 0.3rem;border-radius:3px;white-space:nowrap;">gap vs NOAA/IRI</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">SON&#8211;DJF autumn&#8211;winter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Multiple studies; autumn phase-transition evidence<br><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn&#8211;ENSO study 2019</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — IRQ SON +0.63, IRN OND +0.59, SAU MAM +0.51</td></tr><tr style="background:#ffffff;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">South Asia</span><br><span style="color:#222;">India (all-India monsoon)</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJAS</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Robust historically but <strong>non-stationary</strong> (weakened since ~1980s)<br><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8605027/" target="_blank" rel="noopener" style="color:#2563b8;">Indian-monsoon non-stationarity reviews</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">India is not in the ERA5 set (no admin-0 extract); the ENSO&#8211;monsoon link is documented but weakening</td></tr><tr style="background:#ffffff;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">South Asia</span><br><span style="color:#222;">Pakistan, Afghanistan</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">OND &amp; MAM (winter&#8211;spring)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Established<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — PAK OND +0.55, MAM +0.54; AFG AMJ +0.59</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Central America</span><br><span style="color:#222;">Guatemala, Honduras, El&nbsp;Salvador, Nicaragua (Dry Corridor)</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJA&#8211;SON (incl. can&#237;cula)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier (drought)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Canonical<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — GTM ASO &#8722;0.61, HND &#8722;0.60, NIC &#8722;0.61</td></tr><tr style="background:#ffffff;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">Southern Africa</span><br><span style="color:#222;">Zimbabwe, Zambia, Malawi, Mozambique, S.&nbsp;Africa, Botswana</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">DJF austral summer</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier (drought)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Robust, widely replicated<br><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees — ZWE DJF &#8722;0.68, ZAF NDJ &#8722;0.62, MOZ JFM &#8722;0.50</td></tr><tr style="background:#f7f9fc;"><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;"><span style="color:#5a6b85;">SE Asia / Maritime Continent</span><br><span style="color:#222;">Indonesia, Philippines, Vietnam, Thailand</span></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">JJA&#8211;SON dry season</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier (drought)</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Very robust (drought &amp; fire risk)<br><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.35rem 0.5rem;border-bottom:1px solid #e3e8ef;vertical-align:top;">Agrees strongly — IDN ASO &#8722;0.87, PHL FMA &#8722;0.86, THA &#8722;0.81</td></tr></tbody></table></div>
    <p class="enso-note" style="margin-top:0.7rem;font-size:0.74rem;color:#666;">
      Notes: directions are the dominant documented response — ENSO teleconnections are asymmetric, so the
      La&nbsp;Niña column is the typical (not guaranteed) opposite of El&nbsp;Niño. “Evidence” flags how well
      replicated each link is. The Yemen/SW-Arabia summer link is a <strong>direct</strong> Pacific
      teleconnection (Atif et&nbsp;al. 2017), not an Indian-Ocean-Dipole artefact. The ENSO–Indian-monsoon
      correlation is non-stationary and has weakened since the 1980s, consistent with the weak all-India signal
      in our ERA5 analysis. Framing on map incompleteness follows Lenssen, Goddard &amp; Mason
      (<a href="https://journals.ametsoc.org/view/journals/wefo/35/6/WAF-D-19-0235.1.xml" target="_blank" rel="noopener" style="color:#2563b8;">2020</a>),
      who detect many additional teleconnections when the standard maps are updated.
    </p>
                <details style="margin-top:0.9rem;">
      <summary style="cursor:pointer;font-weight:600;font-size:0.9rem;color:#1a1a1a;">Complete per-country catalogue (128 documented of 154 monitored) — click to expand</summary>
      <p class="enso-note" style="margin:0.4rem 0 0.5rem;font-size:0.73rem;">
        Every monitored country (153 ERA5-analysis countries), researched region by region from the literature.
        Cell shade encodes <strong>confidence</strong> (hatched/pale = single study, solid/dark = robust); the
        “La&nbsp;Niña” column shows the documented opposite where known (<em>italic</em> = inferred inverse).
        “—” = no robust documented ENSO rainfall link (grey on the map). Final column is this study's strongest
        significant ERA5 Niño3.4 correlation (season and sign), or “no sig”. Rows tagged
        <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;">gap</span>
        are documented links absent/under-represented on the standard NOAA/IRI maps.
      </p>
<div style="overflow-x:auto;max-height:460px;overflow-y:auto;border:1px solid #dde3ec;border-radius:4px;">
<table style="border-collapse:collapse;width:100%;font-size:0.72rem;line-height:1.35;min-width:720px;">
<thead><tr><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">Country</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">El&nbsp;Niño</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">La&nbsp;Niña</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">Season</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">Evidence</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">Source</th><th style="text-align:left;padding:0.3rem 0.45rem;border-bottom:2px solid #c4cdda;font-weight:600;color:#1a1a1a;background:#eef2f7;position:sticky;top:0;">ERA5 Niño3.4</th></tr></thead><tbody>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Afghanistan</strong> <span style="color:#8a93a3;">AFG</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.55</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Albania</strong> <span style="color:#8a93a3;">ALB</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/JCLI-D-14-00008.1" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti et al. 2002 (GRL, Euro-Mediterranean rainfall &amp; ENSO); Zhang et al. 2014 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Algeria</strong> <span style="color:#8a93a3;">DZA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JFM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-021-05768-y" target="_blank" rel="noopener" style="color:#2563b8;">Euro-Mediterranean late-winter ENSO study 2021 (Clim. Dyn.); Mariotti et al. 2002 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">SON +0.30</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Angola</strong> <span style="color:#8a93a3;">AGO</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF +0.34</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Antigua and Barbuda</strong> <span style="color:#8a93a3;">ATG</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.43</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Argentina</strong> <span style="color:#8a93a3;">ARG</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">NDJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013&lt;0035:CVITSO&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Grimm et al. 2000 (J. Climate); Vera et al. 2004 (GRL); Cai et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.54</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Armenia</strong> <span style="color:#8a93a3;">ARM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics) - review finds no coherent Caucasus signal</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Azerbaijan</strong> <span style="color:#8a93a3;">AZE</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bahamas</strong> <span style="color:#8a93a3;">BHS</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Giannini, Kushnir &amp; Cane 2000 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Barbados</strong> <span style="color:#8a93a3;">BRB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Giannini, Kushnir &amp; Cane 2000 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.46</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Belarus</strong> <span style="color:#8a93a3;">BLR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.32</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Belize</strong> <span style="color:#8a93a3;">BLZ</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); FEWS NET Central America/Belize analyses</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.51</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Benin</strong> <span style="color:#8a93a3;">BEN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bermuda</strong> <span style="color:#8a93a3;">BMU</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">No robust peer-reviewed ENSO-Bermuda rainfall teleconnection</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JFM +0.42</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>BGD</strong> <span style="color:#8a93a3;">BGD</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS +0.32</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bhutan</strong> <span style="color:#8a93a3;">BTN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.jag.2024.104238" target="_blank" rel="noopener" style="color:#2563b8;">Gyeltshen et al. 2024 (Int. J. Applied Earth Observation and Geoinformation); Kuenzang et al. 2021 (Atmosphere - MDPI, Bhutan ENSO/IOD case study)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bolivia</strong> <span style="color:#8a93a3;">BOL</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF wet / DJF dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/(SICI)1097-0088(19991115)19:13&lt;1579::AID-JOC441&gt;3.0.CO;2-N" target="_blank" rel="noopener" style="color:#2563b8;">Vuille 1999 (Int. J. Climatol.); Garreaud &amp; Aceituno 2001 (J. Climate); Garreaud et al. 2003 (Palaeo3)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.31</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bonaire (Caribbean Netherlands)</strong> <span style="color:#8a93a3;">BES</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); regional ABC-islands climatology (Curacao/Bonaire)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.66</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Botswana</strong> <span style="color:#8a93a3;">BWA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.52</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Brazil</strong> <span style="color:#8a93a3;">BRA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / FMA dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2003)016&lt;0263:TENIOT&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Grimm 2003/2004 (J. Climate); Grimm &amp; Tedeschi 2009 (J. Climate); Coelho et al. 2002 (Int. J. Climatol.); Cai et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>British Virgin Islands</strong> <span style="color:#8a93a3;">VGB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00484-018-1494-6" target="_blank" rel="noopener" style="color:#2563b8;">Mendez-Lazaro et al. 2018 (Int. J. Biometeorol., PR/USVI region); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.45</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Bulgaria</strong> <span style="color:#8a93a3;">BGR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/JCLI-D-14-00008.1" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti et al. 2002 (GRL); Zhang et al. 2014 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MJJ +0.35</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Burkina Faso</strong> <span style="color:#8a93a3;">BFA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.38</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Burundi</strong> <span style="color:#8a93a3;">BDI</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/2016RG000544" target="_blank" rel="noopener" style="color:#2563b8;">Indeje et al. 2000 (Int. J. Climatol.); Nicholson 2017 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.46</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cambodia</strong> <span style="color:#8a93a3;">KHM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">AMJ −0.72</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cameroon</strong> <span style="color:#8a93a3;">CMR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.3912" target="_blank" rel="noopener" style="color:#2563b8;">Diatta &amp; Fink 2014 (Int. J. Climatol.); Giannini, Saravanan &amp; Chang 2003/2008 on Sahel ENSO; Nicholson 2018 (Glob. Planet. Change) review; Cameroon agroecological ENSO study 2025 (Earth Syst. Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cape Verde</strong> <span style="color:#8a93a3;">CPV</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Janicot, Trzaska &amp; Poccard 2001 (Clim. Dyn.); Camberlin et al. 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.36</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cayman Islands</strong> <span style="color:#8a93a3;">CYM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.45</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Central African Republic</strong> <span style="color:#8a93a3;">CAF</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.gloplacha.2017.12.014" target="_blank" rel="noopener" style="color:#2563b8;">Nicholson 2018 (Glob. Planet. Change) review of African rainfall; Balas et al. 2007 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.38</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Chad</strong> <span style="color:#8a93a3;">TCD</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.43</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Chile</strong> <span style="color:#8a93a3;">CHL</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2003)016&lt;0281:SOTERA&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Montecinos &amp; Aceituno 2003 (J. Climate); Garreaud et al. 2017 (Int. J. Climatol., megadrought)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS+ / AMJ−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>China</strong> <span style="color:#8a93a3;">CHN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA wet / JJA dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013&lt;3206:STRBSC&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Zhang et al. 1999; Wu et al. 2003 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Colombia</strong> <span style="color:#8a93a3;">COL</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-010-0931-y" target="_blank" rel="noopener" style="color:#2563b8;">Poveda et al. 2011 (Climate Dynamics); Poveda &amp; Mesa 1997 (GRL); Cordoba-Machado et al. 2015 (Climate Dynamics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Comoros</strong> <span style="color:#8a93a3;">COM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">No robust peer-reviewed rainfall teleconnection (limited regional studies, e.g. Salim et al. conference work on northern Madagascar/Comoros)</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ +0.47</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Costa Rica</strong> <span style="color:#8a93a3;">CRI</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cote d&#x27;Ivoire</strong> <span style="color:#8a93a3;">CIV</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MJJ −0.46</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Cuba</strong> <span style="color:#8a93a3;">CUB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF wet / JAS dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.47</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Democratic Republic of the Congo</strong> <span style="color:#8a93a3;">COD</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / MAM dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.1456" target="_blank" rel="noopener" style="color:#2563b8;">Balas, Nicholson &amp; Klotter 2007 (Int. J. Climatol.); Preethi, Ratnam et al. 2015 (Sci. Rep.); Hua et al. 2016 (Clim. Dyn.); Nicholson 2018 (Glob. Planet. Change) review</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Djibouti</strong> <span style="color:#8a93a3;">DJI</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Dominica</strong> <span style="color:#8a93a3;">DMA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.47</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Dominican Republic</strong> <span style="color:#8a93a3;">DOM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MJJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Giannini, Kushnir &amp; Cane 2000 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM+ / JAS−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Ecuador</strong> <span style="color:#8a93a3;">ECU</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JFM wet / MAM dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.jhydrol.2008.12.026" target="_blank" rel="noopener" style="color:#2563b8;">Rossel &amp; Cadier 2009 (J. Hydrology); Vicente-Serrano et al. 2017 (Int. J. Climatol.); Bendix &amp; Bendix 2006</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Egypt</strong> <span style="color:#8a93a3;">EGY</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JFM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.jaridenv.2021.104575" target="_blank" rel="noopener" style="color:#2563b8;">Rainfall variability over Sinai Peninsula and its teleconnection to El Nino SST 2021 (J. Arid Environ.); Euro-Med late-winter ENSO study 2021 (Clim. Dyn.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>El Salvador</strong> <span style="color:#8a93a3;">SLV</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Equatorial Guinea</strong> <span style="color:#8a93a3;">GNQ</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.1456" target="_blank" rel="noopener" style="color:#2563b8;">Balas, Nicholson &amp; Klotter 2007 (Int. J. Climatol.); Dezfuli &amp; Nicholson 2013 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">AMJ −0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Eritrea</strong> <span style="color:#8a93a3;">ERI</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Eswatini</strong> <span style="color:#8a93a3;">SWZ</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1088/1748-9326/aacc4c" target="_blank" rel="noopener" style="color:#2563b8;">Pomposi et al. 2018 (Environmental Research Letters); Steinkopf &amp; Engelbrecht 2025 (Environmental Research Letters)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.59</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Ethiopia</strong> <span style="color:#8a93a3;">ETH</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / JJAS dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000; Korecha &amp; Barnston 2007</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.63</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Fiji</strong> <span style="color:#8a93a3;">FJI</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-009-0716-3" target="_blank" rel="noopener" style="color:#2563b8;">Vincent et al. 2011 (Clim. Dyn.); Cai et al. 2012 (Nature); Brown et al. 2020 (Nat. Rev. Earth Environ.); PACCSAP Fiji report 2014; McGree et al. 2014 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>French Guiana</strong> <span style="color:#8a93a3;">GUF</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.2364" target="_blank" rel="noopener" style="color:#2563b8;">Bovolo et al. 2012 (Int. J. Climatol.); Yoon &amp; Zeng 2010 (Amazon ENSO, Climate Dynamics); regional Meteo-France analyses</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.69</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Gabon</strong> <span style="color:#8a93a3;">GAB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.5194/piahs-384-181-2021" target="_blank" rel="noopener" style="color:#2563b8;">Marechal et al. 2021 (Proc. IAHS 384); Dezfuli &amp; Nicholson 2013 Part I (J. Climate); Balas et al. 2007 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Gambia</strong> <span style="color:#8a93a3;">GMB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.38</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Georgia</strong> <span style="color:#8a93a3;">GEO</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Ghana</strong> <span style="color:#8a93a3;">GHA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Grenada</strong> <span style="color:#8a93a3;">GRD</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Sookram et al. (ENSO dry-season Trinidad, regional analogue)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.54</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Guadeloupe</strong> <span style="color:#8a93a3;">GLP</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.43</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Guatemala</strong> <span style="color:#8a93a3;">GTM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.61</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Guinea</strong> <span style="color:#8a93a3;">GIN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.32</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Guinea-Bissau</strong> <span style="color:#8a93a3;">GNB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Guyana</strong> <span style="color:#8a93a3;">GUY</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.2364" target="_blank" rel="noopener" style="color:#2563b8;">Bovolo et al. 2012 (Int. J. Climatol.); IRI/CariCOF &amp; WMO regional analyses; Giannini et al. 2000 (Caribbean ENSO)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.55</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Haiti</strong> <span style="color:#8a93a3;">HTI</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MJJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); FEWS NET Haiti climate analyses</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Honduras</strong> <span style="color:#8a93a3;">HND</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.60</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Hungary</strong> <span style="color:#8a93a3;">HUN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics); Zhang et al. 2014 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>India</strong> <span style="color:#8a93a3;">IND</span> <span style="color:#999;font-size:0.66rem;">(not monitored)</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8605027/" target="_blank" rel="noopener" style="color:#2563b8;">Indian-monsoon non-stationarity reviews</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Indonesia</strong> <span style="color:#8a93a3;">IDN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND −0.70</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Iran</strong> <span style="color:#8a93a3;">IRN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">SON</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Iraq</strong> <span style="color:#8a93a3;">IRQ</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">SON</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Jamaica</strong> <span style="color:#8a93a3;">JAM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2001JC001097" target="_blank" rel="noopener" style="color:#2563b8;">Chen &amp; Taylor 2002 (Int. J. Climatol.); Taylor, Enfield &amp; Chen 2002 (JGR-Oceans); Giannini, Kushnir &amp; Cane 2000 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.48</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Kazakhstan</strong> <span style="color:#8a93a3;">KAZ</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1038/s41612-024-00742-x" target="_blank" rel="noopener" style="color:#2563b8;">Yao et al. 2024 (npj Climate and Atmospheric Science)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Kenya</strong> <span style="color:#8a93a3;">KEN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033121" target="_blank" rel="noopener" style="color:#2563b8;">Park et al. 2020 (JGR-A)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ +0.61</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Kiribati</strong> <span style="color:#8a93a3;">KIR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.pacificclimatechangescience.org/wp-content/uploads/2013/06/6_PCCSP_Kiribati_8pp.pdf" target="_blank" rel="noopener" style="color:#2563b8;">PCCSP/PACCSAP Kiribati country report 2011/2014; Australian BoM-CSIRO Pacific Climate Change Science Program 2011</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF +0.91</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Kuwait</strong> <span style="color:#8a93a3;">KWT</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/2017JD027263" target="_blank" rel="noopener" style="color:#2563b8;">Sandeep &amp; Ajayamohan 2018 (J. Geophys. Res. Atmos., Arabian Gulf); Marcella &amp; Eltahir / Kuwait hydroclimatology studies</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.40</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Kyrgyzstan</strong> <span style="color:#8a93a3;">KGZ</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1038/s41612-024-00742-x" target="_blank" rel="noopener" style="color:#2563b8;">Yao et al. 2024 (npj Climate and Atmospheric Science)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">AMJ +0.52</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Laos</strong> <span style="color:#8a93a3;">LAO</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS+ / AMJ−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Lebanon</strong> <span style="color:#8a93a3;">LBN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2001GL014248" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti, Zeng &amp; Lau 2002 (Geophys. Res. Lett.); Hoell et al. 2024 (J. Climate, Fertile Crescent)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.47</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Lesotho</strong> <span style="color:#8a93a3;">LSO</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.48</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Liberia</strong> <span style="color:#8a93a3;">LBR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Libya</strong> <span style="color:#8a93a3;">LBY</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-021-05768-y" target="_blank" rel="noopener" style="color:#2563b8;">Euro-Mediterranean late-winter ENSO study 2021 (Clim. Dyn.); Mariotti et al. 2002 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.30</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>LKA</strong> <span style="color:#8a93a3;">LKA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">SON +0.49</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Madagascar</strong> <span style="color:#8a93a3;">MDG</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">NDJ wet / DJF dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1088/1748-9326/ade60e" target="_blank" rel="noopener" style="color:#2563b8;">Steinkopf &amp; Engelbrecht 2025 (Environmental Research Letters); FEWS NET El Nino impact assessment 2024</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Malawi</strong> <span style="color:#8a93a3;">MWI</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">FMA −0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Malaysia</strong> <span style="color:#8a93a3;">MYS</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Maldives</strong> <span style="color:#8a93a3;">MDV</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MJJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.ijdrr.2020.101726" target="_blank" rel="noopener" style="color:#2563b8;">Foley &amp; Kelman 2020 (Int. J. Disaster Risk Reduction 50:101726); follow-up Kelman et al. 2025 (Theoretical and Applied Climatology)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.64</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mali</strong> <span style="color:#8a93a3;">MLI</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Marshall Islands</strong> <span style="color:#8a93a3;">MHL</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / FMA dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.pacificrisa.org/wp-content/uploads/2015/11/Pacific-Region-EL-NINO-Fact-Sheet_RMI_2015-FINAL-v2.pdf" target="_blank" rel="noopener" style="color:#2563b8;">Pacific ENSO Applications Climate Center (PEAC) / NWS; Pacific RISA &#x27;El Nino and its Impacts on the RMI&#x27; 2015; PACCSAP/PCCSP Marshall Islands country report 2011/2014</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA +0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Martinique</strong> <span style="color:#8a93a3;">MTQ</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MJJ+ / ASO−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mauritania</strong> <span style="color:#8a93a3;">MRT</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.54</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mauritius</strong> <span style="color:#8a93a3;">MUS</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">No robust peer-reviewed rainfall teleconnection; cf. Mascarene-High variability studies (Xulu et al. 2020, Sci. Rep.)</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mexico</strong> <span style="color:#8a93a3;">MEX</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF wet / JJA dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/JCLI4045.1" target="_blank" rel="noopener" style="color:#2563b8;">Magana, Vazquez, Perez &amp; Perez 2003 (Geofisica Internacional); Pavia, Graef &amp; Reyes 2006 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.58</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Micronesia (Federated States of)</strong> <span style="color:#8a93a3;">FSM</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / MAM dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.weather.gov/media/peac/one_pagers/El%20Nino%20Impacts%20on%20the%20Eastern%20FSM.pdf" target="_blank" rel="noopener" style="color:#2563b8;">Pacific ENSO Applications Climate Center (PEAC) / NWS; PACCSAP FSM country report 2014; Lander &amp; Khosrowpanah; PEAC El Nino impacts on Eastern FSM fact sheet</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA+ / OND−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Moldova</strong> <span style="color:#8a93a3;">MDA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti et al. 2002 (GRL); Bronnimann 2007 (Rev. Geophys.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mongolia</strong> <span style="color:#8a93a3;">MNG</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1126/sciadv.1701832" target="_blank" rel="noopener" style="color:#2563b8;">Hessl et al. 2018 (Science Advances); Wu et al. 2019 (Climate Dynamics, EASM-ENSO)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Morocco</strong> <span style="color:#8a93a3;">MAR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">NDJ wet / NDJ dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-021-05768-y" target="_blank" rel="noopener" style="color:#2563b8;">Lopez-Parages &amp; Rodriguez-Fonseca 2012 / Mariotti et al. 2002 (Mediterranean ENSO); Euro-Med late-winter ENSO study 2021 (Clim. Dyn.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Mozambique</strong> <span style="color:#8a93a3;">MOZ</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JFM −0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Myanmar</strong> <span style="color:#8a93a3;">MMR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS+ / AMJ−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Namibia</strong> <span style="color:#8a93a3;">NAM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JFM −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Nicaragua</strong> <span style="color:#8a93a3;">NIC</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.61</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Niger</strong> <span style="color:#8a93a3;">NER</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.31</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Nigeria</strong> <span style="color:#8a93a3;">NGA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.40</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>North Korea</strong> <span style="color:#8a93a3;">PRK</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-013-1953-z" target="_blank" rel="noopener" style="color:#2563b8;">Choi et al. 2014 (Climate Dynamics); Ho et al. 2016 (Asia-Pacific J. Atmos. Sci.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.35</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>NPL</strong> <span style="color:#8a93a3;">NPL</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MJJ −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Oman</strong> <span style="color:#8a93a3;">OMN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">FMA +0.42</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Pakistan</strong> <span style="color:#8a93a3;">PAK</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">FMA+ / JAS−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Palestine</strong> <span style="color:#8a93a3;">PSE</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2001GL014248" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti, Zeng &amp; Lau 2002 (Geophys. Res. Lett.); Price, Stone &amp; Rind 1998 (J. Climate, E. Med. rainfall-ENSO)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.56</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Panama</strong> <span style="color:#8a93a3;">PAN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.72</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Papua New Guinea</strong> <span style="color:#8a93a3;">PNG</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/JCLI-D-13-00322.1" target="_blank" rel="noopener" style="color:#2563b8;">Murphy, Power &amp; McGree 2014 (J. Climate); Cai et al. 2011 (J. Climate); PACCSAP/PCCSP PNG country report 2011/2014</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JFM +0.39</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Paraguay</strong> <span style="color:#8a93a3;">PRY</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013&lt;0035:CVITSO&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Grimm et al. 2000 (J. Climate); Barros et al. 2000/2008; Cai et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.44</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Peru</strong> <span style="color:#8a93a3;">PER</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF wet / DJF dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.5194/adgeo-14-231-2008" target="_blank" rel="noopener" style="color:#2563b8;">Lagos et al. 2008 (Adv. Geosci.); Garreaud &amp; Aceituno 2001 (J. Climate); Sulca et al. 2018 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.32</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Philippines</strong> <span style="color:#8a93a3;">PHL</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND −0.74</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Poland</strong> <span style="color:#8a93a3;">POL</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.49</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Puerto Rico</strong> <span style="color:#8a93a3;">PRI</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">AMJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00484-018-1494-6" target="_blank" rel="noopener" style="color:#2563b8;">Mendez-Lazaro et al. 2018 (Int. J. Biometeorology, Teleconnections between ENSO and rainfall/drought in Puerto Rico); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM+ / JAS−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Qatar</strong> <span style="color:#8a93a3;">QAT</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/2017JD027263" target="_blank" rel="noopener" style="color:#2563b8;">Sandeep &amp; Ajayamohan 2018 (J. Geophys. Res. Atmos., Arabian Gulf); Hoell et al. 2024 (J. Climate, Middle East/SW Asia)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.41</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Republic of the Congo</strong> <span style="color:#8a93a3;">COG</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/JCLI-D-11-00653.1" target="_blank" rel="noopener" style="color:#2563b8;">Balas, Nicholson &amp; Klotter 2007 (Int. J. Climatol.); Dezfuli &amp; Nicholson 2013 Part I Boreal Spring (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Romania</strong> <span style="color:#8a93a3;">ROU</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Mariotti et al. 2002 (GRL, Euro-Mediterranean rainfall &amp; ENSO); Bronnimann 2007 (Rev. Geophys.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Russia</strong> <span style="color:#8a93a3;">RUS</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.37</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Rwanda</strong> <span style="color:#8a93a3;">RWA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/2016RG000544" target="_blank" rel="noopener" style="color:#2563b8;">Indeje et al. 2000 (Int. J. Climatol.); Nicholson 2017 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ +0.49</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Saint Kitts and Nevis</strong> <span style="color:#8a93a3;">KNA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.45</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Saint Lucia</strong> <span style="color:#8a93a3;">LCA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); CariCOF outlooks</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MJJ+ / ASO−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Saint Vincent and the Grenadines</strong> <span style="color:#8a93a3;">VCT</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013%3C0297:IVOCRE%3E2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Giannini, Kushnir &amp; Cane 2000 (J. Climate); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Sao Tome and Principe</strong> <span style="color:#8a93a3;">STP</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1016/j.gloplacha.2017.12.014" target="_blank" rel="noopener" style="color:#2563b8;">Balas et al. 2007 (Int. J. Climatol.); Nicholson 2018 (Glob. Planet. Change) review (Atlantic-dominated western-equatorial regime)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Saudi Arabia</strong> <span style="color:#8a93a3;">SAU</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.51</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Senegal</strong> <span style="color:#8a93a3;">SEN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.49</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Seychelles</strong> <span style="color:#8a93a3;">SYC</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">NDJ</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.icpac.net/" target="_blank" rel="noopener" style="color:#2563b8;">Seychelles Meteorological Authority / regional operational guidance (ICPAC GHACOF)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Sierra Leone</strong> <span style="color:#8a93a3;">SLE</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Slovakia</strong> <span style="color:#8a93a3;">SVK</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.32</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Solomon Islands</strong> <span style="color:#8a93a3;">SLB</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-009-0716-3" target="_blank" rel="noopener" style="color:#2563b8;">Vincent et al. 2011 (Clim. Dyn.); Cai et al. 2011/2012; PACCSAP Solomon Islands country report 2014; Brown et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.33</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Somalia</strong> <span style="color:#8a93a3;">SOM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033121" target="_blank" rel="noopener" style="color:#2563b8;">Park et al. 2020 (JGR-A)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.65</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>South Africa</strong> <span style="color:#8a93a3;">ZAF</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.62</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>South Sudan</strong> <span style="color:#8a93a3;">SSD</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.59</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Sudan</strong> <span style="color:#8a93a3;">SDN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.40</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Suriname</strong> <span style="color:#8a93a3;">SUR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.2364" target="_blank" rel="noopener" style="color:#2563b8;">Bovolo et al. 2012 (Int. J. Climatol.); IRI/CariCOF regional outlooks; Nurmohamed et al. (Suriname hydrology)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JJA −0.64</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Syria</strong> <span style="color:#8a93a3;">SYR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">SON</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.47</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Tanzania</strong> <span style="color:#8a93a3;">TZA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND wet / DJF dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/2016RG000544" target="_blank" rel="noopener" style="color:#2563b8;">Nicholson 2017 (Reviews of Geophysics); Palmer et al. 2023 (Nature Reviews Earth &amp; Environment); Indeje et al. 2000 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ +0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Thailand</strong> <span style="color:#8a93a3;">THA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">AMJ −0.70</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Timor-Leste</strong> <span style="color:#8a93a3;">TLS</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2003)016&lt;1775:IRVIOE&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Hendon 2003 (J. Climate 16:1775); Aldrian &amp; Susanto 2003 (Int. J. Climatology 23:1435); Moron et al. 2009 (Int. J. Climatology, Indonesian monsoon onset predictability)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.72</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Togo</strong> <span style="color:#8a93a3;">TGO</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/joc.633" target="_blank" rel="noopener" style="color:#2563b8;">Camberlin, Janicot &amp; Poccard 2001 (Int. J. Climatol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Tonga</strong> <span style="color:#8a93a3;">TON</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1038/s43017-020-0078-2" target="_blank" rel="noopener" style="color:#2563b8;">Vincent et al. 2011 (Clim. Dyn.); Cai et al. 2012 (Nature); PACCSAP Tonga country report 2014; Brown et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.63</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Trinidad and Tobago</strong> <span style="color:#8a93a3;">TTO</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Sookram &amp; ... (Impact of ENSO on dry-season rainfall in Trinidad); Jury, Malmgren &amp; Winter 2007 (JGR-Atmos)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.57</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Tunisia</strong> <span style="color:#8a93a3;">TUN</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JFM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1007/s00382-021-05768-y" target="_blank" rel="noopener" style="color:#2563b8;">Euro-Mediterranean late-winter ENSO study 2021 (Clim. Dyn.); Mariotti, Zeng &amp; Lau 2002 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Turkey</strong> <span style="color:#8a93a3;">TUR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/hyp.7219" target="_blank" rel="noopener" style="color:#2563b8;">Karabork &amp; Kahya 2009 (Hydrological Processes); Karabork, Kahya &amp; Karaca 2005 (Hydrological Processes)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">AMJ +0.39</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Turks and Caicos Islands</strong> <span style="color:#8a93a3;">TCA</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#F0DAC2,#F0DAC2 3px,#D29A6C 3px,#D29A6C 5px);text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:repeating-linear-gradient(45deg,#D4E7F7,#D4E7F7 3px,#86BCE8 3px,#86BCE8 5px);text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Giannini, Kushnir &amp; Cane 2000 (J. Climate)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.30</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>UAE</strong> <span style="color:#8a93a3;">ARE</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.researchgate.net/publication/337134434_The_Impact_of_ENSO_Phase_Transition_on_the_Atmospheric_Circulation_Precipitation_and_Temperature_in_the_Middle_East_Autumn" target="_blank" rel="noopener" style="color:#2563b8;">Middle East autumn–ENSO study 2019</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">FMA +0.44</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Uganda</strong> <span style="color:#8a93a3;">UGA</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1002/(SICI)1097-0088(200001)20:1&lt;19::AID-JOC449&gt;3.0.CO;2-0" target="_blank" rel="noopener" style="color:#2563b8;">Indeje et al. 2000 (Int. J. Climatol.); Nicholson 2017 (Reviews of Geophysics); Palmer et al. 2023 (Nature Reviews Earth &amp; Environment)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND+ / JAS−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Ukraine</strong> <span style="color:#8a93a3;">UKR</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006RG000199" target="_blank" rel="noopener" style="color:#2563b8;">Bronnimann 2007 (Reviews of Geophysics); Zhang et al. 2014 (J. Climate, seasonal ENSO-Europe precip)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.34</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>United States Virgin Islands</strong> <span style="color:#8a93a3;">VIR</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">ASO</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1029/2006JD007541" target="_blank" rel="noopener" style="color:#2563b8;">Jury, Malmgren &amp; Winter 2007 (JGR-Atmos); Mendez-Lazaro et al. 2018 (Int. J. Biometeorol.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">JAS −0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Uruguay</strong> <span style="color:#8a93a3;">URY</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">OND</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1175/1520-0442(2000)013&lt;0035:CVITSO&gt;2.0.CO;2" target="_blank" rel="noopener" style="color:#2563b8;">Grimm et al. 2000 (J. Climate); Cazes-Boezio et al. 2003 (J. Climate); Mernild et al. (Uruguay studies)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND +0.65</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Uzbekistan</strong> <span style="color:#8a93a3;">UZB</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">MAM</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1038/s41612-024-00742-x" target="_blank" rel="noopener" style="color:#2563b8;">Yao et al. 2024 (npj Climate and Atmospheric Science)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">MAM +0.50</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Vanuatu</strong> <span style="color:#8a93a3;">VUT</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.1038/nclimate1726" target="_blank" rel="noopener" style="color:#2563b8;">Vincent et al. 2011 (Clim. Dyn.); Cai et al. 2012 (Nature); PACCSAP Vanuatu country report 2014; Brown et al. 2020 (Nat. Rev. Earth Environ.)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">NDJ −0.70</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Venezuela</strong> <span style="color:#8a93a3;">VEN</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#D29A6C;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#86BCE8;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JJA</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">moderate</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://doi.org/10.3406/bifea.2002.6943" target="_blank" rel="noopener" style="color:#2563b8;">Pulwarty et al. 1992/1998; Cardenas et al. 2002 (Bull. IFEA); Poveda et al. 2006 review</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">ASO −0.67</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Vietnam</strong> <span style="color:#8a93a3;">VNM</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">JAS</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999GL011140" target="_blank" rel="noopener" style="color:#2563b8;">Dai &amp; Wigley 2000 (GRL)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">OND −0.69</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Western Sahara</strong> <span style="color:#8a93a3;">ESH</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#DBDBDB;text-align:center;color:#666;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">—</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">No country-specific peer-reviewed ENSO rainfall study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">no sig</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Yemen</strong> <span style="color:#8a93a3;">YEM</span> <span style="background:#7B3A1A;color:#fff;font-size:0.58rem;padding:0.02rem 0.25rem;border-radius:3px;white-space:nowrap;">gap</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:linear-gradient(135deg,#86BCE8 50%,#D29A6C 50%);text-align:center;color:#1a1a1a;">split</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF wet / JJA dry</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">single-study</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://www.nature.com/articles/s41612-017-0003-7" target="_blank" rel="noopener" style="color:#2563b8;">Atif et al. 2017 (npj Clim Atmos Sci)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">FMA+ / JAS−</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Zambia</strong> <span style="color:#8a93a3;">ZMB</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.40</td></tr>
<tr><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><strong>Zimbabwe</strong> <span style="color:#8a93a3;">ZWE</span></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#9C6730;text-align:center;color:#1a1a1a;">drier</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;background:#2E7DBD;text-align:center;color:#1a1a1a;">wetter</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">DJF</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;">robust</td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;"><a href="https://journals.ametsoc.org/view/journals/bams/82/4/1520-0477_2001_082_0619_ppaawe_2_3_co_2.xml" target="_blank" rel="noopener" style="color:#2563b8;">Mason &amp; Goddard 2001 (BAMS)</a></td><td style="padding:0.3rem 0.45rem;border-bottom:1px solid #e8ecf1;vertical-align:top;color:#555;">DJF −0.68</td></tr>
</tbody></table></div>
    </details>

  </section>

  <hr>
  <section>
    <h2>Methodology</h2>
    <div class="note" style="font-size:0.82rem;line-height:1.6;">
      <h3 style="font-size:0.95rem;margin:0 0 0.6rem;">Data</h3>
      <p style="margin:0 0 0.6rem;">
        Rainfall is drawn from ERA5 reanalysis (ECMWF), extracted as country-level area-weighted mean precipitation
        for 153 countries at admin-0 level, covering 1981–{end_year}. Units are mm/day; a trimester value is the
        mean of the three constituent monthly means (not a sum). Climate mode indices are monthly anomaly series
        downloaded from NOAA PSL: <strong>Niño3.4</strong> (ERSSTv5 SST anomaly in 5°N–5°S, 120–170°W),
        <strong>IOD</strong> (Dipole Mode Index, HadISST), <strong>TNA</strong> (Tropical North Atlantic SST anomaly),
        <strong>TSA</strong> (Tropical South Atlantic SST anomaly), <strong>AMM</strong> (Atlantic Meridional Mode),
        and <strong>PDO</strong> (Pacific Decadal Oscillation, NOAA).
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Spatial resolution (Country vs Pixel)</h3>
      <p style="margin:0 0 0.6rem;">
        The <strong>Resolution</strong> toggle selects the unit of analysis; the method below is identical in
        both cases. <em>Country (ADM0)</em> uses the pre-computed ERA5 admin-0 raster statistics held in the
        team database — one area-weighted mean series per country, 153 countries. <em>Pixel (0.25°)</em> reads
        the source ERA5 monthly precipitation COGs directly and runs the whole pipeline independently on every
        land grid cell in the map viewport (~{_px_cells_hint} cells at 0.25°, ≈28 km at the equator), with no
        spatial smoothing or pooling between cells.
      </p>
      <p style="margin:0 0 0.6rem;">
        The pixel view exists because a national mean can hide as much as it shows: countries spanning more
        than one rainfall regime (Kenya, Ethiopia, Indonesia, Brazil) average opposing signals toward zero, and
        a signal confined to one basin or one side of a mountain range disappears entirely. It carries two
        costs. First, each cell is a single ~45-year series, so at p&nbsp;&lt;&nbsp;0.05 a few percent of cells
        will pass by chance; because no field-significance or false-discovery correction is applied, isolated
        coloured cells should be read as noise and only spatially coherent regions as signal. Second, an extra
        filter is needed that the country pass does not require: a cell–season is analysed only if its
        climatological mean exceeds {PIXEL_MIN_TRI_MM_DAY} mm/day, which removes hyper-arid cells (Sahara, Rub'&nbsp;al&nbsp;Khali,
        Taklamakan interiors) where correlations against near-zero rainfall are numerically large but
        meaningless. Those cells are drawn in the “arid — no wet season” shade. Ocean cells are not analysed.
      </p>
      <p style="margin:0 0 0.6rem;">
        On the pixel maps, cells significant in both directions across non-overlapping seasons are
        <strong>hatched</strong> rather than split diagonally, and the coloured fill shows the stronger of the
        two. The pixel dominant-mode map shows the top mode only, and only where it leads the runner-up
        by at least {PIXEL_DOMINANT_MARGIN:.2f} in |r| — one cell does not average enough area to make
        the arg-max over six collinear modes stable, so near-ties are left grey rather than assigned a
        winner.
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Seasonal aggregation</h3>
      <p style="margin:0 0 0.6rem;">
        All 12 rolling 3-month windows are assessed for every country: NDJ, DJF, JFM, FMA, MAM, AMJ, MJJ,
        JJA, JAS, ASO, SON, OND. Using rolling windows rather than four fixed seasons avoids misaligning
        the analysis window with the actual rainy season (e.g. a country whose rains peak Oct–Dec is better
        captured by OND than by JAS or DJF). Year labels use the first month of the window: NDJ and DJF are
        labeled by November and December respectively (e.g. DJF 2024 = Dec 2024–Feb 2025); all others by
        the year of their first month.
      </p>
      <p style="margin:0 0 0.6rem;">
        A country–trimester pair is included only if that trimester's climatological mean rainfall is at
        least 25% of the country's annual mean, suppressing correlations in dry seasons. Annual mean is
        computed from the four non-overlapping canonical trimesters (DJF + MAM + JAS + OND), which together
        cover each calendar month exactly once. Multiple rolling windows can qualify for the same country.
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Correlation sweep (total association)</h3>
      <p style="margin:0 0 0.6rem;">
        For each country × trimester × index combination, Pearson r is computed across all lags up to the
        selected maximum (index leading rainfall). The lag with the highest |r| is retained as the "best lag"
        for that combination. The <strong>Max lag</strong> toggle controls this cap: <em>3 months</em> (default)
        restricts the index to at most one preceding non-overlapping season, which keeps the relationship within
        the same evolving event and is the more forecast-relevant view; <em>6 months</em> additionally admits
        prior-season relationships (e.g. a previous winter's ENSO state predicting the following monsoon), which
        can have the opposite sign to the concurrent signal. Significance is assessed at p &lt; 0.05 (two-tailed).
        Results below |r| = {_R_MIN} are treated as no reliable signal and shown in gray; |r| ≥ {_R_STRONG} is
        shown in a darker shade. The best-lag values are <em>frozen</em> after the total pass and reused in the
        partial pass below.
      </p>
      <p style="margin:0 0 0.6rem;">
        On each per-index map, a country is shown in a single color if all its significant correlations for that
        index have the same sign. It is shown as a split diagonal if it has significant correlations in
        <em>both</em> directions across non-overlapping trimesters (e.g. ENSO drives wet conditions in one
        season and dry in another). With 12 rolling windows, the non-overlapping check is required to avoid
        pairing near-identical windows (e.g. JFM and FMA) as spuriously "bidirectional".
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Partial correlation (unique signal)</h3>
      <p style="margin:0 0 0.6rem;">
        The unique-signal maps show partial correlations: the correlation between an index and rainfall after
        removing the linear influence of all other indices. This is implemented via the residuals method —
        for a given country, trimester, and target index, both the rainfall series and the target index series
        are separately regressed on all other indices (at their own frozen best lags), and the Pearson r of
        the residuals is taken as the partial correlation. PDO still has its own unique signal computed, but
        it is excluded from the control set — i.e. it is never regressed out of the other modes — because as a
        low-frequency mode strongly collinear with ENSO, controlling for it would absorb genuine ENSO signal.
        Best lags are identical to those from the total pass.
      </p>
      <p style="margin:0 0 0.6rem;">
        When an index is significant in the unique-signal view but not in the total view, this is a
        <em>suppressor variable</em> effect — the index has real predictive value that is masked in the raw
        correlation because it partially cancels shared variance from another index. These are shown in the
        unique-signal maps. The reverse — total significant, unique not — simply means the signal is largely
        shared with other modes.
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">ENSO composites</h3>
      <p style="margin:0 0 0.6rem;">
        El Niño and La Niña composites are computed per-country using the concurrent trimester's Niño3.4
        value (not a single annual classification). For each country, the trimester with the strongest
        correlation to Niño3.4 is used as the headline season. Years in which the concurrent Niño3.4 mean
        ≥ +0.5 are composited as El Niño; years ≤ −0.5 as La Niña. Anomalies are expressed in standard
        deviations from the full-period climatological mean for that country–trimester. The two maps are
        independent — ENSO teleconnections are asymmetric and the La Niña composite is not simply the
        mirror image of El Niño.
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Documented-impacts literature review</h3>
      <p style="margin:0 0 0.6rem;">
        The “Documented ENSO Impacts (Literature)” section complements the data-driven maps with teleconnections
        established in the scientific literature, which together are more complete than the generalized NOAA and
        IRI impact maps. It was assembled with a multi-agent deep-research pass: the question was decomposed into
        per-region search angles, parallel web searches retrieved candidate sources, and roughly two dozen
        peer-reviewed articles plus authoritative operational sources (NOAA, IRI, ICPAC/IGAD) were fetched and
        read. From these, falsifiable directional claims (region × season × phase × sign) were extracted and each
        was <strong>adversarially verified</strong> by an independent panel — a claim was retained only if it
        survived attempts to refute it, and dropped otherwise. This filtering removed plausible-but-unsupported
        statements; for example, the framing of the Yemen signal as Indian-Ocean-Dipole-mediated was refuted, so
        it is reported as a <em>direct</em> Pacific teleconnection (Atif et&nbsp;al. 2017). Coverage was then
        extended in a second region-by-region sweep over the rest of the monitored world (South America, the
        Caribbean, Central/East Africa, East Asia, the Pacific and the mid-latitudes), so that every one of the
        153 monitored countries has been researched.
      </p>
      <p style="margin:0 0 0.6rem;">
        Verified regional findings were combined with canonical global ENSO-rainfall climatologies (Dai &amp;
        Wigley 2000; Mason &amp; Goddard 2001; Lenssen, Goddard &amp; Mason 2020) into the per-link table. Each
        row carries a citation and an <em>evidence-strength</em> flag (robust and widely replicated vs.
        single-study or contested), and a “gap vs NOAA/IRI” tag where a well-supported link is absent or
        under-represented on the standard maps. Every documented link is cross-referenced against this study's
        own ERA5 correlation result (agree / partial / weak) as an independent check.
      </p>
      <p style="margin:0 0 0.6rem;">
        The map colors each country by its dominant documented <strong>El&nbsp;Niño</strong> response, labelled
        with the relevant 3-month season; countries that respond in opposite directions across seasons are shown
        split. <strong>Grey</strong> means a monitored country was researched but has no robust documented ENSO
        rainfall link; <strong>near-white</strong> means the country is not monitored and was not researched. The
        <strong>La&nbsp;Niña</strong> map is the mechanical inverse of El&nbsp;Niño — a first-order picture only,
        since ENSO is asymmetric. Evidence strength is uneven (robust in well-studied regions; single-study or
        contested elsewhere — see the per-country catalogue), and the “gap vs NOAA/IRI” tags are editorial
        judgements.
      </p>
      <h3 style="font-size:0.95rem;margin:0.8rem 0 0.6rem;">Caveats</h3>
      <ul style="margin:0;padding-left:1.2rem;">
        <li>Pearson r assumes a linear, stationary relationship. Teleconnections can be non-linear and
            non-stationary; correlations over 1981–{end_year} may not represent future conditions.</li>
        <li>ERA5 reanalysis is model-dependent. In data-sparse regions the precipitation field is more
            influenced by the model background than by observations.</li>
        <li>Multiple-testing is not corrected for (no FDR/Bonferroni). With ~600 country–trimester combinations
            per index, some false positives are expected at p &lt; 0.05.</li>
        <li>Partial correlations do not partition the total explained variance cleanly when predictors are
            collinear (e.g. AMM and TNA share r ≈ 0.81). Unique-signal maps should be read as indicating
            whether an index contributes <em>beyond</em> the others, not as a clean attribution.</li>
        <li>Coverage is limited to countries where ERA5 admin-0 extracts are available (153 countries).
            Small island states are shown as dots rather than choropleth polygons.</li>
      </ul>
    </div>
  </section>

  <script>
    // Inline scroll-to-zoom on every .map-zoom container.
    // Scroll: zoom centred on cursor. Drag: pan when zoomed in. Double-click: reset.
    document.querySelectorAll('.map-zoom').forEach(container => {{
      const img = container.querySelector('img');
      let scale = 1, ox = 0, oy = 0, dragging = false, sx, sy;
      const apply = () => {{
        img.style.transform = `translate(${{ox}}px,${{oy}}px) scale(${{scale}})`;
      }};
      const clamp = () => {{
        const cw = container.clientWidth, ch = container.clientHeight;
        const iw = img.offsetWidth * scale, ih = img.offsetHeight * scale;
        ox = Math.min(0, Math.max(ox, cw - iw));
        oy = Math.min(0, Math.max(oy, ch - ih));
      }};
      container.addEventListener('wheel', e => {{
        e.preventDefault();
        const r = container.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
        const imgX = (mx - ox) / scale, imgY = (my - oy) / scale;
        scale = Math.min(Math.max(scale * factor, 1), 12);
        ox = mx - imgX * scale;
        oy = my - imgY * scale;
        clamp();
        apply();
        img.style.cursor = scale > 1 ? 'grab' : 'default';
      }}, {{passive: false}});
      img.addEventListener('mousedown', e => {{
        if (scale === 1) return;
        dragging = true; sx = e.clientX - ox; sy = e.clientY - oy;
        img.style.cursor = 'grabbing'; e.preventDefault();
      }});
      window.addEventListener('mousemove', e => {{
        if (!dragging) return;
        ox = e.clientX - sx; oy = e.clientY - sy;
        clamp(); apply();
      }});
      window.addEventListener('mouseup', () => {{
        dragging = false;
        img.style.cursor = scale > 1 ? 'grab' : 'default';
      }});
      container.addEventListener('dblclick', () => {{
        scale = 1; ox = 0; oy = 0; apply();
        img.style.cursor = 'default';
      }});
    }});

    const cur = {{ view: 'partial', lag: '{DEFAULT_LAG_TAG}', res: '{DEFAULT_RES_TAG}' }};
    function applyView() {{
      document.querySelectorAll('.map-pair').forEach(p => {{
        p.style.gridTemplateColumns = '1fr';
      }});
      document.querySelectorAll('[data-kind], [data-lag], [data-res]').forEach(item => {{
        const ok = (!item.dataset.kind || item.dataset.kind === cur.view)
                && (!item.dataset.lag  || item.dataset.lag  === cur.lag)
                && (!item.dataset.res  || item.dataset.res  === cur.res);
        item.style.display = ok ? '' : 'none';
      }});
    }}
    document.querySelectorAll('.toggle-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const group = btn.dataset.group;
        if (!group) return;  // phase toggle (.ensoph-btn) handled separately
        document.querySelectorAll('.toggle-btn[data-group="' + group + '"]')
          .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        cur[group] = btn.dataset.show;
        applyView();
      }});
    }});
    // Apply default on load
    applyView();

    // ENSO phase toggle (El Niño / La Niña) for the documented-impacts maps
    document.querySelectorAll('.ensoph-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const phase = btn.dataset.phase;
        document.querySelectorAll('.ensoph-btn').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.ensoph').forEach(d => {{
          d.style.display = (d.dataset.phase === phase) ? '' : 'none';
        }});
      }});
    }});

    // Stable heading ids + hover permalinks, with deep-link support (#section)
    (function () {{
      const slug = t => t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
      const seen = {{}};
      document.querySelectorAll('h1, h2').forEach(h => {{
        let id = h.id || slug(h.textContent);
        seen[id] = (seen[id] || 0) + 1;
        if (seen[id] > 1) id += '-' + seen[id];
        h.id = id;
        const a = document.createElement('a');
        a.href = '#' + id;
        a.className = 'anchor-link';
        a.textContent = '#';
        a.setAttribute('aria-label', 'Link to this section');
        h.appendChild(a);
      }});
      if (location.hash) {{
        const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
        if (el) el.scrollIntoView();
      }}
    }})();
  </script>
</body>
</html>"""

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML report written to {docs_dir / 'index.html'}")


# --------------------------------------------------------------------------- #
def run_country_pass(cfg: dict, indices: pd.DataFrame,
                     gdf: gpd.GeoDataFrame) -> None:
    """Country (ADM0) resolution: ERA5 admin-0 raster stats from the team DB."""
    rain = country_trimester_rainfall_era5(cfg)
    n_countries = rain.columns.get_level_values("iso3").nunique()
    print(f"Loaded ERA5 rainfall: {n_countries} countries, "
          f"years {rain.index.min()}–{rain.index.max()}")

    rainy = rainy_trimesters(rain)
    print(f"Rainy (iso3, trimester) pairs: {len(rainy)} of {rain.columns.nunique()} total")
    rainy_df = pd.DataFrame(list(rainy), columns=["iso3", "trimester"])

    analyzed_isos: set[str] = set(rain.columns.get_level_values("iso3").unique())

    # Regenerate TS panels fresh each run (they are lag/kind-independent)
    for _ts in cfg["out_dir"].glob("ts_*.png"):
        _ts.unlink()

    # One full analysis+map pass per lag-cap variant (toggle in the app)
    for lag_tag, max_lag in LAG_CAPS.items():
        total = sweep(rain, indices, cfg, max_lag=max_lag)
        total.to_parquet(cfg["parquet_dir"] / f"corr_total_{lag_tag}.parquet")

        partial = partial_pass(rain, indices, total, cfg)
        partial.to_parquet(cfg["parquet_dir"] / f"corr_partial_{lag_tag}.parquet")

        total_rainy = total.merge(rainy_df, on=["iso3", "trimester"])
        disp_total  = reduce_for_display(total_rainy, cfg["alpha"])
        disp_total.to_parquet(cfg["parquet_dir"] / f"corr_display_total_{lag_tag}.parquet")

        # Partial: restrict to rainy seasons, then drop suppressor-only signals
        # (sig partial but not total — "unique fraction of a non-existent total"
        # is not interpretable for a survey product)
        total_sig_keys = set(zip(disp_total["iso3"], disp_total["index"]))
        partial_rainy  = partial.merge(rainy_df, on=["iso3", "trimester"])
        disp_partial_all = reduce_for_display(partial_rainy, cfg["alpha"])
        partial_sig_keys = set(zip(disp_partial_all["iso3"], disp_partial_all["index"]))
        n_suppressor = len(partial_sig_keys - total_sig_keys)
        print(f"[{lag_tag}] max_lag={max_lag}; suppressor-only partial signals "
              f"excluded: {n_suppressor}")

        disp_partial = disp_partial_all[
            disp_partial_all.apply(lambda r: (r["iso3"], r["index"]) in total_sig_keys, axis=1)
        ].reset_index(drop=True)
        disp_partial.to_parquet(cfg["parquet_dir"] / f"corr_display_partial_{lag_tag}.parquet")

        for kind, disp in (("total", disp_total), ("partial", disp_partial_all)):
            for name in INDEX_SOURCES:
                plot_index_map(disp, gdf, name, cfg["out_dir"],
                               kind=kind, end_year=cfg["end_year"], indices=indices,
                               analyzed_isos=analyzed_isos, lag_tag=lag_tag)

        plot_dominant_index_map(disp_total, gdf, cfg["out_dir"], end_year=cfg["end_year"],
                                analyzed_isos=analyzed_isos, lag_tag=lag_tag)

    enso_composite_maps(rain, indices, gdf, cfg, rainy=rainy, analyzed_isos=analyzed_isos)
    print(f"country pass done across {n_countries} countries.")


def run_pixel_pass(cfg: dict, indices: pd.DataFrame,
                   gdf: gpd.GeoDataFrame) -> None:
    """Pixel resolution: the same method on every ERA5 land cell (0.25 deg),
    read from the ERA5 monthly COGs on blob."""
    px = pixel_trimester_rainfall_era5(cfg, gdf)
    cfg["px_cells_hint"] = f"{px.n_cells:,}"   # quoted in the methodology text
    cols = list(indices.columns)

    for lag_tag, max_lag in LAG_CAPS.items():
        px_total = pixel_sweep(px, indices, cfg, max_lag=max_lag)
        px_partial = pixel_partial_pass(px, indices, cfg, px_total, max_lag=max_lag)

        px_disp = {
            "total":   pixel_reduce_for_display(px_total, px, cols, cfg["alpha"]),
            "partial": pixel_reduce_for_display(px_partial, px, cols, cfg["alpha"]),
        }
        n_sup = sum(
            int((np.isfinite(px_disp["partial"][c]["r"])
                 & ~np.isfinite(px_disp["total"][c]["r"])).sum())
            for c in cols
        )
        n_sig = sum(int(np.isfinite(px_disp["total"][c]["r"]).sum()) for c in cols)
        print(f"[px/{lag_tag}] max_lag={max_lag}; significant (cell, index) totals: "
              f"{n_sig}; suppressor-only partials: {n_sup}")

        np.savez_compressed(
            cfg["parquet_dir"] / f"corr_px_display_{lag_tag}.npz",
            x=px.x, y=px.y, flat_idx=px.flat_idx, cols=np.array(cols),
            trimesters=np.array(list(TRIMESTERS)),
            **{f"{kind}_{c}_{k}": v
               for kind, d in px_disp.items() for c in cols
               for k, v in d[c].items()},
        )

        for kind, disp in px_disp.items():
            for name in INDEX_SOURCES:
                plot_pixel_index_map(disp, px, gdf, name, cfg["out_dir"],
                                     kind=kind, lag_tag=lag_tag)
        plot_pixel_dominant_map(px_disp["total"], px, gdf, cols, cfg["out_dir"],
                                lag_tag=lag_tag)

    pixel_enso_composite_maps(px, indices, gdf, cfg)
    print(f"pixel pass done across {px.n_cells} land cells.")


def main(cfg: dict = CONFIG, skip: tuple[str, ...] = ()) -> None:
    """Full run. `skip` may contain 'adm0' and/or 'px' to leave that resolution's
    PNGs untouched — the two passes are independent, and the country pass needs
    DB access while the pixel pass only needs blob."""
    indices = load_indices(cfg)
    print_collinearity(indices)
    gdf = load_admin0_gdf(cfg)
    cfg["parquet_dir"].mkdir(parents=True, exist_ok=True)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    if "adm0" not in skip:
        # Regenerate TS panels fresh each run (they are lag/kind-independent)
        for _ts in cfg["out_dir"].glob("ts_*.png"):
            _ts.unlink()
        run_country_pass(cfg, indices, gdf)
    else:
        print("skipping country (ADM0) pass")

    if "px" not in skip:
        run_pixel_pass(cfg, indices, gdf)
    else:
        print("skipping pixel pass")

    # Resolution-independent panels
    plot_index_corr_matrix(indices, cfg["out_dir"], end_year=cfg["end_year"])
    plot_literature_enso_map(gdf, cfg["out_dir"], phase="elnino")
    plot_literature_enso_map(gdf, cfg["out_dir"], phase="lanina")
    generate_html_report(cfg)
    print(f"done; lag caps {list(LAG_CAPS.values())}, "
          f"resolutions {[r for r in RES_VARIANTS if r not in skip]}.")


if __name__ == "__main__":
    import sys
    _skip = tuple(r for r in RES_VARIANTS if f"--skip-{r}" in sys.argv[1:])
    main(skip=_skip)
