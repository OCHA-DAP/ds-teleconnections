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

    _kind_meta = {
        "total":   ("total association",
                    "<strong>Total association</strong> — pairwise Pearson r, p&lt;0.05. "
                    "Split diagonal = significant in both directions across non-overlapping seasons."),
        "partial": ("unique signal",
                    "<strong>Unique signal</strong> — partial r, other climate modes held constant. "
                    "Shrinkage vs Total = shared variance, not absent signal."),
    }
    _lag_label = {"l3": "0–3 mo lag", "l6": "0–6 mo lag"}

    def _map_item(name, label, kind, lag_tag):
        kind_title, caption = _kind_meta[kind]
        return f"""
      <div class="map-item" data-kind="{kind}" data-lag="{lag_tag}">
        <div class="map-with-ts">
          <div class="ts-col"><img style="width:100%;height:auto;display:block;" src="maps/ts_{name}.png" alt="{label} historical values"></div>
          <div class="map-col">
            <p class="map-title">Correlation of {label} with total seasonal rainfall — {kind_title} ({_lag_label[lag_tag]})</p>
            <div class="map-zoom"><img src="maps/map_{lag_tag}_{kind}_{name}.png" alt="{label} {kind} correlation {lag_tag}"></div>
            {corr_legend}
          </div>
        </div>
        <p>{caption}</p>
      </div>"""

    index_sections = []
    for name in INDEX_SOURCES:
        label = INDEX_LABELS.get(name, name.upper())
        items = "".join(
            _map_item(name, label, kind, lag_tag)
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
    dominant_items = "".join(
        f"""<div class="map-item" data-lag="{lag_tag}" style="max-width:100%">
      <div class="map-zoom"><img src="maps/map_dominant_{lag_tag}.png" alt="Dominant climate mode map {lag_tag}"></div>
      {dominant_legend}
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
    .map-item p {{ font-size: 0.75rem; color: #555; margin: 0.4rem 0 0 0; }}
    .map-item p strong {{ color: #222; }}
    section {{ margin-bottom: 2rem; }}
    .enso-note {{ font-size: 0.82rem; color: #444; margin-bottom: 0.5rem; }}
    hr {{ border: none; border-top: 1px solid #dde3ec; margin: 2rem 0; }}
    .view-toggle {{
      position: sticky; top: 0; z-index: 100;
      display: flex; gap: 0.5rem; align-items: center;
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
  </style>
</head>
<body>
  <h1>ERA5 Precipitation Teleconnection Analysis</h1>
  <p class="meta">ERA5 1981–{end_year} &nbsp;·&nbsp; Pearson r &nbsp;·&nbsp; p &lt; 0.05 &nbsp;·&nbsp; Gray = no reliable signal &nbsp;·&nbsp; Split diagonal = both signs across seasons</p>
  <div class="view-toggle">
    <span>View:</span>
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
    <p class="enso-note">Each country colored by the index with the strongest significant total correlation (|r|≥{_R_MIN}). Split diagonal = second-strongest index (different mode), shown only when it acts on a non-overlapping trimester. Respects the Max lag toggle above.</p>
    {dominant_items}
  </section>

  <hr>
  <section>
    <h2>ENSO Composites — El Niño &amp; La Niña</h2>
    <p class="enso-note">Mean rainfall anomaly (standard deviations from climatology) in each country's headline trimester when Niño3.4 ≥ ±0.5 concurrent with that trimester. The two maps are independent — ENSO impacts are asymmetric.</p>
    <div class="map-pair">
      <div class="map-item">
        <div class="map-zoom"><img src="maps/enso_elnino.png" alt="El Niño composite rainfall anomaly"></div>
        <p><strong>El Niño composite</strong> — mean anomaly when Niño3.4 ≥ +0.5 (brown = drier than normal, blue = wetter).</p>
      </div>
      <div class="map-item">
        <div class="map-zoom"><img src="maps/enso_lanina.png" alt="La Niña composite rainfall anomaly"></div>
        <p><strong>La Niña composite</strong> — mean anomaly when Niño3.4 ≤ −0.5. Roughly opposite to El Niño in ENSO-sensitive regions, but magnitude and pattern differ.</p>
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

    let curView = 'partial', curLag = '{DEFAULT_LAG_TAG}';
    function applyView() {{
      document.querySelectorAll('.map-pair').forEach(p => {{
        p.style.gridTemplateColumns = '1fr';
      }});
      document.querySelectorAll('[data-kind], [data-lag]').forEach(item => {{
        const okKind = !item.dataset.kind || item.dataset.kind === curView;
        const okLag  = !item.dataset.lag  || item.dataset.lag  === curLag;
        item.style.display = (okKind && okLag) ? '' : 'none';
      }});
    }}
    document.querySelectorAll('.toggle-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const group = btn.dataset.group;
        document.querySelectorAll('.toggle-btn[data-group="' + group + '"]')
          .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (group === 'view') curView = btn.dataset.show;
        else curLag = btn.dataset.show;
        applyView();
      }});
    }});
    // Apply default on load
    applyView();
  </script>
</body>
</html>"""

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML report written to {docs_dir / 'index.html'}")


# --------------------------------------------------------------------------- #
def main(cfg: dict = CONFIG) -> None:
    indices = load_indices(cfg)
    print_collinearity(indices)

    rain = country_trimester_rainfall_era5(cfg)
    n_countries = rain.columns.get_level_values("iso3").nunique()
    print(f"Loaded ERA5 rainfall: {n_countries} countries, "
          f"years {rain.index.min()}–{rain.index.max()}")

    gdf = load_admin0_gdf(cfg)

    cfg["parquet_dir"].mkdir(parents=True, exist_ok=True)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

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

    # Lag-independent panels
    enso_composite_maps(rain, indices, gdf, cfg, rainy=rainy, analyzed_isos=analyzed_isos)
    plot_index_corr_matrix(indices, cfg["out_dir"], end_year=cfg["end_year"])
    generate_html_report(cfg)

    print(f"done across {n_countries} countries; lag caps {list(LAG_CAPS.values())}.")


if __name__ == "__main__":
    main()
