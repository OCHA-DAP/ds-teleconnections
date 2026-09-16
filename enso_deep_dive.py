"""ENSO country deep dives — per-country evidence reviews published under docs/enso/.

Each country is one TOML file in deep_dives/ (curated narrative + literature) and
this script adds the reproducible ERA5 evidence: a seasonal-cycle chart, pixel maps
of the Niño3.4 correlation, an ENSO-phase rainfall history, El Niño composite and
drought hit-rate maps, and the phase/drought tables. Everything numeric is computed
here from the same cached ERA5 pixel stack and NOAA indices the survey uses (no DB
access needed); the narrative lives in the TOML so a human can edit it.

    uv run python enso_deep_dive.py            # all countries
    uv run python enso_deep_dive.py --only eri # one

Requires the pixel cache (cache/era5_pixel/, built by a survey run) and the
survey's out/corr_*.parquet tables for the country-level cross-check (optional).
"""
from __future__ import annotations

import argparse
import html
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import from_origin

import teleconnection_survey as ts

DEEP_DIR = Path("deep_dives")
OUT_DIR = Path("docs/enso")
SITE_TITLE = "Teleconnections"
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Same conventions as the survey: brown = drier under El Niño (negative r),
# blue = wetter. ENSO phase colours (validated categorical triplet).
C_ELNINO, C_NEUTRAL, C_LANINA = "#D1495B", "#9AA3AD", "#2E7DBD"
C_TEXT, C_MUTED = "#1f2324", "#5e6a6b"
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "brbu", ["#7A4E22", "#B17E50", "#E8CDB0", "#F3F3F1", "#BFD9EE", "#5E9FD2", "#1F5F96"])
ENSO_THRESH = 0.5
NINO_LATEST = None   # latest NOAA Niño3.4 series (current-state line only); set in main()
GRADES = {"robust": "#9C6730", "moderate": "#D29A6C", "single-study": "#F0DAC2", "none": "#DBDBDB"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    stack: np.ndarray               # (n_months, ny, nx) memmap, full viewport
    mpos: dict[tuple[int, int], int]
    x: np.ndarray
    y: np.ndarray
    years: np.ndarray = field(default_factory=lambda: np.arange(1981, 2026))


def load_grid(cfg: dict) -> Grid:
    d = cfg["cache_dir"] / "era5_pixel"
    meta = json.loads((d / "meta.json").read_text())
    stack = np.load(d / "monthly.npy", mmap_mode="r")
    ym = [tuple(t) for t in meta["ym"]]
    return Grid(stack=stack, mpos={p: i for i, p in enumerate(ym)},
                x=np.asarray(meta["x"]), y=np.asarray(meta["y"]),
                years=np.arange(cfg["start_year"], cfg["end_year"] + 1))


@dataclass
class Country:
    iso3: str
    geom: gpd.GeoSeries
    lat: np.ndarray                 # bbox cell-centre latitudes (descending)
    lon: np.ndarray
    mask: np.ndarray                # (ny, nx) inside-country
    sub: np.ndarray                 # (n_months, ny, nx) monthly mm/day, bbox
    neighbours: gpd.GeoDataFrame


def cut_country(grid: Grid, gdf: gpd.GeoDataFrame, iso3: str, pad: int = 2) -> Country:
    sel = gdf[gdf.iso3 == iso3]
    if sel.empty:
        raise SystemExit(f"{iso3}: not in the Natural Earth layer")
    res = float(abs(grid.x[1] - grid.x[0]))
    tr = from_origin(grid.x[0] - res / 2, grid.y[0] + res / 2, res, res)
    full = rasterize(((g, 1) for g in sel.geometry), out_shape=(len(grid.y), len(grid.x)),
                     transform=tr, fill=0, all_touched=True, dtype="uint8").astype(bool)
    iy, ix = np.nonzero(full)
    r0, r1 = max(iy.min() - pad, 0), min(iy.max() + pad + 1, len(grid.y))
    c0, c1 = max(ix.min() - pad, 0), min(ix.max() + pad + 1, len(grid.x))
    bbox = sel.total_bounds
    nb = gdf[gdf.intersects(sel.geometry.union_all().buffer(3.0)) & (gdf.iso3 != iso3)]
    return Country(iso3=iso3, geom=sel.geometry, lat=grid.y[r0:r1], lon=grid.x[c0:c1],
                   mask=full[r0:r1, c0:c1], sub=grid.stack[:, r0:r1, c0:c1], neighbours=nb)


def season_months(code: str) -> list[int]:
    """'JAS' -> [7,8,9]; also accepts 4-month codes like 'JJAS' and 'OND'."""
    if code in ts._TRIMESTER_MONTHS:
        return list(ts._TRIMESTER_MONTHS[code])
    letters = "JFMAMJJASOND"
    # find the start position whose letters match
    for s in range(12):
        if all(letters[(s + k) % 12] == code[k] for k in range(len(code))):
            return [((s + k) % 12) + 1 for k in range(len(code))]
    raise ValueError(code)


def season_stack(c: Country, grid: Grid, months: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(n_years, ny, nx) mean mm/day for the season, year labelled by its first month."""
    wrap = 12 in months and 1 in months
    out, yrs = [], []
    for sy in grid.years:
        pos = []
        for m in months:
            cy = sy + (1 if (wrap and m <= 6) else 0)
            if (cy, m) not in grid.mpos:
                pos = None
                break
            pos.append(grid.mpos[(cy, m)])
        if pos:
            out.append(c.sub[pos].mean(axis=0))
            yrs.append(sy)
    return np.array(out, dtype="float64"), np.array(yrs)


def nino_series(indices: pd.DataFrame, months: list[int], lag: int) -> pd.Series:
    s = indices["nino34"].shift(lag).rolling(len(months)).mean()
    sub = s[s.index.month == months[-1]]
    off = -1 if (12 in months and 1 in months) else 0
    return pd.Series(sub.values, index=sub.index.year + off).dropna()


def cell_corr(R: np.ndarray, yrs: np.ndarray, n: pd.Series) -> tuple[np.ndarray, int]:
    common = np.intersect1d(yrs, n.index.values)
    Y = R[np.isin(yrs, common)].reshape(len(common), -1)
    X = n.loc[common].values
    Xc, Yc = X - X.mean(), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc @ Yc) / (np.sqrt((Xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(0)))
    return r.reshape(R.shape[1:]), len(common)


def best_lag_corr(R, yrs, indices, months, max_lag=3):
    rs = np.stack([cell_corr(R, yrs, nino_series(indices, months, lag))[0] for lag in range(max_lag + 1)])
    k = np.nanargmax(np.where(np.isnan(rs), -1, np.abs(rs)), axis=0)
    return np.take_along_axis(rs, k[None], 0)[0], k


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9d0d0")
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#e6eaea", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_seasonal_cycle(c: Country, grid: Grid, headline: list[int], out: Path, name: str,
                       zones: dict[str, np.ndarray] | None = None) -> None:
    """Monthly climatology, whole country plus optional sub-zones (lines)."""
    clim = np.zeros(12)
    for m in range(1, 13):
        pos = [grid.mpos[(y, m)] for y in grid.years if (y, m) in grid.mpos]
        clim[m - 1] = c.sub[pos][:, c.mask].mean()
    # Rotate the month axis so the headline season sits in the middle of the chart
    # (a Nov–Apr season is unreadable when the year starts in January).
    mid = headline[len(headline) // 2] - 1
    order = [(mid + 6 + k) % 12 for k in range(12)]        # 0-based month indices, left to right
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    cols = ["#9C6730" if (m + 1) in headline else "#D8C3AC" for m in order]
    ax.bar(range(12), clim[order], color=cols, width=0.72)
    if zones:
        for (label, zm), col in zip(zones.items(), ["#1F5F96", "#5E9FD2", "#18614c"]):
            z = np.array([c.sub[[grid.mpos[(y, m)] for y in grid.years if (y, m) in grid.mpos]][:, zm].mean()
                          for m in range(1, 13)])
            ax.plot(range(12), z[order], color=col, lw=2, marker="o", ms=4, label=label)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.set_xticks(range(12)); ax.set_xticklabels([MONTH_NAMES[m] for m in order])
    if order[0] != 0:  # mark where the calendar year turns over
        jan = order.index(0)
        ax.axvline(jan - 0.5, color="#c9d0d0", lw=0.8, ls=(0, (3, 3)))
        ax.text(jan - 0.55, clim.max() * 0.5, "1 Jan", fontsize=7.5, color=C_MUTED, rotation=90, ha="right", va="center")
    ax.set_ylabel("mm / day", fontsize=9, color=C_MUTED)
    ax.set_title(f"{name}: ERA5 monthly rainfall climatology, 1981–{grid.years[-1]} "
                 f"(dark bars = headline season)", fontsize=10, color=C_TEXT, loc="left")
    _style_ax(ax)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_zone_map(c: Country, zones: dict[str, np.ndarray], out: Path, name: str, head: str) -> None:
    """Locator map: which cells belong to which zone (same colours as the zone lines elsewhere)."""
    w, h = _panel_size(c, base=3.6, floor=2.2)
    fig, ax = plt.subplots(figsize=(w + 2.6, h + 0.9), dpi=150)
    ext = _extent(c)
    cols = ["#1F5F96", "#5E9FD2", "#18614c", "#8a4f7d"]
    idx = np.full(c.mask.shape, np.nan)
    for i, (label, zm) in enumerate(zones.items()):
        idx[zm] = i
    n = len(zones)
    cmap = mcolors.ListedColormap(cols[:n])
    _pcolor(ax, c, idx, cmap, -0.5, n - 0.5)
    # cells inside the country but in no zone (no headline season): light grey
    none = c.mask & np.isnan(idx)
    _pcolor(ax, c, np.where(none, 0.0, np.nan), mcolors.ListedColormap(["#ececec"]), -1, 1)
    _draw_country(ax, c, ext)
    ax.set_title(f"{name}: zones used on this page", fontsize=10, color=C_TEXT, loc="left")
    handles = [Patch(color=cols[i], label=f"{lbl} — {int(zm.sum())} cells") for i, (lbl, zm) in enumerate(zones.items())]
    if none.any():
        handles.append(Patch(color="#ececec", label=f"no {head} season ({int(none.sum())} cells)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def _draw_country(ax, c: Country, grid_extent):
    c.neighbours.boundary.plot(ax=ax, color="#b8bfbf", linewidth=0.6)
    c.geom.boundary.plot(ax=ax, color="#1f2324", linewidth=1.1)
    ax.set_xlim(grid_extent[0], grid_extent[1]); ax.set_ylim(grid_extent[2], grid_extent[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c9d0d0")


def _extent(c: Country):
    res = float(abs(c.lon[1] - c.lon[0]))
    return (c.lon[0] - res / 2, c.lon[-1] + res / 2, c.lat[-1] - res / 2, c.lat[0] + res / 2)


def _pcolor(ax, c: Country, vals, cmap, vmin, vmax):
    res = float(abs(c.lon[1] - c.lon[0]))
    xe = np.append(c.lon - res / 2, c.lon[-1] + res / 2)
    ye = np.append(c.lat + res / 2, c.lat[-1] - res / 2)
    return ax.pcolormesh(xe, ye, np.ma.masked_invalid(vals), cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="flat", edgecolors="none")


def _panel_size(c: Country, base: float = 4.6, floor: float = 2.4) -> tuple[float, float]:
    """(width, height) in inches for one map panel, following the country's shape: tall narrow
    countries get narrow panels, wide flat countries get short ones."""
    ext = _extent(c)
    aspect = (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6)   # lat range / lon range
    if aspect >= 1:
        return float(np.clip(base / aspect, floor, base)), base
    return base, float(np.clip(base * aspect, floor, base))


def fig_corr_maps(c: Country, panels: list[dict], out: Path, name: str) -> None:
    n = len(panels)
    w, h = _panel_size(c)
    ncols = 2 if (n >= 4 and w > 3.2) else n
    nrows = int(np.ceil(n / ncols))
    H = h * nrows + 1.4 + 0.5 * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols + 1.4, H), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    ext = _extent(c)
    fig.subplots_adjust(top=1 - 0.95 / H, bottom=0.45 / H, left=0.02, right=0.98, wspace=0.12, hspace=0.28)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, p in zip(axes, panels):
        r = np.where(p["analysable"], p["r"], np.nan)
        m = _pcolor(ax, c, np.where(c.mask, r, np.nan), DIVERGING, -0.7, 0.7)
        # cells inside the country but not analysable for this season: hatch
        na = c.mask & ~p["analysable"]
        _pcolor(ax, c, np.where(na, 0.0, np.nan), mcolors.ListedColormap(["#ececec"]), -1, 1)
        _draw_country(ax, c, ext)
        ax.set_title(p["title"], fontsize=9.5, color=C_TEXT, loc="left")
        if p.get("note"):
            ax.set_xlabel(p["note"], fontsize=7.5, color=C_MUTED, loc="left")
    cb = fig.colorbar(m, ax=axes[:n].tolist(), shrink=0.7 if nrows == 1 else 0.5, pad=0.03)
    cb.set_label("Pearson r (Niño3.4 vs rainfall)", fontsize=9, color=C_MUTED)
    cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
    cb.ax.text(0.5, 1.03, "wetter under\nEl Niño", transform=cb.ax.transAxes, fontsize=7.5, color=C_MUTED, va="bottom", ha="center")
    cb.ax.text(0.5, -0.03, "drier under\nEl Niño", transform=cb.ax.transAxes, fontsize=7.5, color=C_MUTED, va="top", ha="center")
    fig.suptitle(f"{name}: pixel-level Niño3.4 correlation\n(ERA5 0.25°, 1981–2025; grey = season too small to analyse)",
                 fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.995, va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_composite_maps(c: Country, comp: np.ndarray, hit: np.ndarray, analysable: np.ndarray,
                       season: str, n_en: int, out: Path, name: str, outlines=None) -> None:
    w, h = _panel_size(c)
    fig, axes = plt.subplots(1, 2, figsize=(2 * w + 2.8, h + 1.3), dpi=150)
    fig.subplots_adjust(top=1 - 1.0 / (h + 1.3), bottom=0.2 / (h + 1.3), left=0.02, right=0.98, wspace=0.35)
    ext = _extent(c)
    ok = c.mask & analysable
    m1 = _pcolor(axes[0], c, np.where(ok, comp, np.nan), DIVERGING, -1.2, 1.2)
    _pcolor(axes[0], c, np.where(c.mask & ~analysable, 0.0, np.nan), mcolors.ListedColormap(["#ececec"]), -1, 1)
    _draw_country(axes[0], c, ext)
    axes[0].set_title(f"Mean {season} rainfall anomaly\nin the {n_en} El Niño years (SD)", fontsize=9.5, loc="left", color=C_TEXT)
    cb = fig.colorbar(m1, ax=axes[0], shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_label("standard deviations", fontsize=9, color=C_MUTED)
    seq = mcolors.LinearSegmentedColormap.from_list("dry", ["#F7F1EA", "#E8CDB0", "#B17E50", "#7A4E22"])
    m2 = _pcolor(axes[1], c, np.where(ok, hit * 100, np.nan), seq, 0, 100)
    _pcolor(axes[1], c, np.where(c.mask & ~analysable, 0.0, np.nan), mcolors.ListedColormap(["#ececec"]), -1, 1)
    _draw_country(axes[1], c, ext)
    if outlines is not None:
        for ax in axes:
            outlines.boundary.plot(ax=ax, color="#ffffff", linewidth=0.35, alpha=0.9)
    axes[1].set_title(f"Share of El Niño years in the cell's\ndriest third of {season} seasons (chance = 33%)"
                      + ("\nwhite outlines = FEWS NET units" if outlines is not None else ""), fontsize=9.5, loc="left", color=C_TEXT)
    cb = fig.colorbar(m2, ax=axes[1], shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_label("% of El Niño years", fontsize=9, color=C_MUTED)
    fig.suptitle(f"{name}: what El Niño (concurrent Niño3.4 ≥ +{ENSO_THRESH})\ndid to the {season} season, per cell",
                 fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.995, va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_phase_history(df: pd.DataFrame, season: str, out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=150)
    col = df.phase.map({"El Niño": C_ELNINO, "Neutral": C_NEUTRAL, "La Niña": C_LANINA})
    ax.bar(df.index, df.z, color=col, width=0.78)
    t = df.rain.quantile(1 / 3)
    zt = (t - df.rain.mean()) / df.rain.std()
    ax.axhline(zt, color="#7A4E22", lw=1, ls=(0, (4, 3)))
    ax.text(df.index[-1] + 0.9, zt - 0.06, "driest-third\nthreshold", fontsize=7.5, color="#7A4E22", va="top", ha="right")
    ax.axhline(0, color="#9aa3ad", lw=0.8)
    for yr, row in df[df.phase == "El Niño"].iterrows():
        ax.text(yr, row.z + (0.08 if row.z >= 0 else -0.08), str(yr), fontsize=7.5, color=C_ELNINO,
                ha="center", va="bottom" if row.z >= 0 else "top", rotation=90)
    ax.set_ylabel(f"{season} rainfall anomaly (SD)", fontsize=9, color=C_MUTED)
    ax.set_xlim(df.index[0] - 1, df.index[-1] + 1)
    ax.legend(handles=[Patch(color=C_ELNINO, label=f"El Niño (Niño3.4 ≥ +{ENSO_THRESH} in {season})"),
                       Patch(color=C_NEUTRAL, label="Neutral"),
                       Patch(color=C_LANINA, label=f"La Niña (≤ −{ENSO_THRESH})")],
              frameon=False, fontsize=8.5, loc="upper right", ncol=3)
    ax.set_title(f"{name}: {season} rainfall over the rain-fed cells, by concurrent ENSO phase", fontsize=10, color=C_TEXT, loc="left")
    _style_ax(ax)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_zone_history(zone_dfs: dict[str, pd.DataFrame], season: str, out: Path, name: str) -> None:
    """One phase-history panel per zone, shared axes, so a north–south split is visible."""
    n = len(zone_dfs)
    fig, axes = plt.subplots(n, 1, figsize=(9.6, 2.35 * n + 0.6), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, df) in zip(axes, zone_dfs.items()):
        col = df.phase.map({"El Niño": C_ELNINO, "Neutral": C_NEUTRAL, "La Niña": C_LANINA})
        ax.bar(df.index, df.z, color=col, width=0.78)
        t = df.rain.quantile(1 / 3)
        ax.axhline((t - df.rain.mean()) / df.rain.std(), color="#7A4E22", lw=1, ls=(0, (4, 3)))
        ax.axhline(0, color="#9aa3ad", lw=0.8)
        for yr, row in df[df.phase == "El Niño"].iterrows():
            ax.text(yr, row.z + (0.08 if row.z >= 0 else -0.08), str(yr), fontsize=6.5, color=C_ELNINO,
                    ha="center", va="bottom" if row.z >= 0 else "top", rotation=90)
        en = df[df.phase == "El Niño"]
        r = np.corrcoef(df.rain, df.nino)[0, 1]
        ax.set_title(f"{label} — r = {fmt_r(r)}; {int((en.pct <= 1 / 3).sum())} of {len(en)} El Niño seasons in the driest third",
                     fontsize=9.5, color=C_TEXT, loc="left")
        ax.set_ylabel("SD", fontsize=8.5, color=C_MUTED)
        _style_ax(ax)
    axes[0].legend(handles=[Patch(color=C_ELNINO, label="El Niño"), Patch(color=C_NEUTRAL, label="Neutral"),
                            Patch(color=C_LANINA, label="La Niña")], frameon=False, fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlim(df.index[0] - 1, df.index[-1] + 1)
    fig.suptitle(f"{name}: standardised {season} rainfall by zone and concurrent ENSO phase", fontsize=10,
                 color=C_TEXT, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# SEAS5 forecast skill (from the seas5-skill app's detrended per-pixel skill cube)
# --------------------------------------------------------------------------- #
# Same thresholds, categories and colours as the app's skill map
# (ds-seas5-skill/pipeline/export_skill_raster_site.py). Skill is the temporal Pearson r
# between the detrended SEAS5 trimester forecast and detrended ERA5, per 0.4° pixel, for a
# given issued month; the lead is the number of months from the issue month to the first
# month of the trimester (negative = issued in-season, elapsed months observed).
SKILL_CUBE = Path("cache/skill_stats_grid_detrended.nc")
SKILL_BLOB = "ds-seas5-skill/processed/raster/skill_stats_grid_detrended.nc"
SKILL_THRESH = {"r_mod": 0.30, "r_high": 0.50}
SKILL_LEADS = [4, 3, 2, 1, 0, -1, -2]
SKILL_CATS = {"negative": "#f3dad7", "low": "#bee0d6", "moderate": "#7dc1ad", "high": "#1e795f"}


def skill_cat(r: float) -> str:
    if r is None or np.isnan(r):
        return "—"
    if r < 0:
        return "negative"
    if r < SKILL_THRESH["r_mod"]:
        return "low"
    if r < SKILL_THRESH["r_high"]:
        return "moderate"
    return "high"


def _ensure_skill_cube() -> Path | None:
    if SKILL_CUBE.exists():
        return SKILL_CUBE
    try:
        import ocha_stratus as stratus
        print("  downloading the SEAS5 skill cube from the DEV blob (one-time)…", flush=True)
        SKILL_CUBE.parent.mkdir(parents=True, exist_ok=True)
        SKILL_CUBE.write_bytes(stratus.load_blob_data(SKILL_BLOB, stage="dev"))
        return SKILL_CUBE
    except Exception as e:  # noqa: BLE001
        print(f"  (SEAS5 skill cube not available: {e})")
        return None


def seas5_skill_issued(c: Country, zones: dict[str, np.ndarray], issued_month: int,
                       leads: list[int] = SKILL_LEADS) -> dict | None:
    """SEAS5 skill of one issuance, per zone, for every complete trimester it covers.

    Returns {"issued_month", "trimesters": [{"code", "lead", "start"}], "rows": [{"zone", "n_cells",
    "r": {code: median pixel r}, "frac_mod": {code: share of cells ≥ moderate}}]}. The app's cube
    is on the SEAS5 0.4° grid; it is sampled at this page's 0.25° ERA5 cell centres (nearest
    neighbour) so the zone masks apply unchanged.
    """
    path = _ensure_skill_cube()
    if path is None:
        return None
    import xarray as xr
    ds = xr.open_dataset(path)
    by_start = {v[0]: k for k, v in ts._TRIMESTER_MONTHS.items()}
    tris = []
    for lead in sorted(leads):
        start = ((issued_month - 1 + lead) % 12) + 1
        if start in by_start:
            tris.append(dict(code=by_start[start], lead=lead, start=start))
    def grid(var, code):
        da = ds[var].sel(issued_month=issued_month, trimester=code)
        return da.sel(x=xr.DataArray(c.lon, dims="lon"), y=xr.DataArray(c.lat, dims="lat"), method="nearest").values

    # Vintage: the cube's current forecast for this issuance, labelled by the lead-0 window's season year
    lead0 = [t for t in tris if t["lead"] == 0]
    issued_year = None
    if lead0:
        yv = ds["current_forecast_year"].sel(issued_month=issued_month, trimester=lead0[0]["code"]).values
        issued_year = int(yv) if np.isfinite(yv) else None
    rows = []
    for label, zm in zones.items():
        if not zm.any():
            continue
        r_by, fm_by, rp_by, pc_by = {}, {}, {}, {}
        for t in tris:
            v = grid("pearson_r", t["code"])[zm]
            v = v[np.isfinite(v)]
            r_by[t["code"]] = float(np.median(v)) if v.size else float("nan")
            fm_by[t["code"]] = float((v >= SKILL_THRESH["r_mod"]).mean()) if v.size else float("nan")
            # signed return period of the current forecast: +dry RP if the forecast sits below its
            # hindcast median, −wet RP otherwise (the app's forecast_rp / flood_rp, Weibull)
            pc = grid("forecast_percentile", t["code"])[zm]
            dry = grid("forecast_rp", t["code"])[zm]
            wet = grid("flood_rp", t["code"])[zm]
            ok = np.isfinite(pc) & np.isfinite(dry) & np.isfinite(wet)
            signed = np.where(pc < 50, dry, -wet)[ok]
            rp_by[t["code"]] = float(np.median(signed)) if signed.size else float("nan")
            pc_by[t["code"]] = float(np.median(pc[ok])) if ok.any() else float("nan")
        rows.append(dict(zone=label, n_cells=int(zm.sum()), r=r_by, frac_mod=fm_by, rp=rp_by, pct=pc_by))
    ds.close()
    return dict(issued_month=issued_month, issued_year=issued_year, trimesters=tris, rows=rows)


def fig_skill_issued(c: Country, grid: Grid, zones: dict[str, np.ndarray], skill: dict, out: Path,
                     name: str, off_share: float = 0.15) -> None:
    """Climatology on top, skill heatmap underneath, on a shared month axis.

    Months run from two before the issuance to six after (the 7-month SEAS5 horizon). Each
    trimester column sits on its middle month, so the reader sees which part of the rainy
    season each forecast window covers. Cells whose trimester holds under `off_share` of the
    zone's annual rain (the app's rainy-season mask) are faded.
    """
    im = skill["issued_month"]
    months = [((im - 3 + k) % 12) + 1 for k in range(9)]          # issued−2 … issued+6
    xpos = {m: k for k, m in enumerate(months)}
    tris = skill["trimesters"]
    rows = skill["rows"]

    def monthly(mask):
        return np.array([c.sub[[grid.mpos[(y, m)] for y in grid.years if (y, m) in grid.mpos]][:, mask].mean()
                         for m in range(1, 13)])

    n = len(rows)
    fig, (ax, hx, rx) = plt.subplots(3, 1, figsize=(9.6, 9.0), dpi=150, sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.4, 2.4, 2.4], hspace=0.12))
    # --- top: climatology
    clim_all = monthly(c.mask)
    ax.bar(range(9), clim_all[[m - 1 for m in months]], color="#D8C3AC", width=0.72, label="Whole country")
    zone_clim = {}
    for (label, zm), col in zip(zones.items(), ["#1F5F96", "#5E9FD2", "#18614c", "#8a4f7d"]):
        z = monthly(zm); zone_clim[label] = z
        ax.plot(range(9), z[[m - 1 for m in months]], color=col, lw=2, marker="o", ms=4, label=label)
    ax.axvline(xpos[im], color=C_TEXT, lw=1, ls=(0, (4, 3)))
    ax.text(xpos[im] + 0.08, ax.get_ylim()[1] * 0.97, f"issued\n1 {MONTH_NAMES[im - 1]}", fontsize=8, color=C_TEXT, va="top")
    ax.set_ylabel("mm / day", fontsize=9, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="best", ncol=1)
    solo_fig = len(rows) == 1
    ax.set_title(f"{name}: skill of the {MONTH_NAMES[im - 1]} SEAS5 issuance{'' if solo_fig else ' by zone'}, against the rainy season",
                 fontsize=10, color=C_TEXT, loc="left")
    _style_ax(ax)
    # --- bottom: median pixel skill per zone as lines over the trimester middle months, on the
    # app's low / moderate / high bands. Hollow markers = window is off-season for that zone.
    zone_cols = dict(zip(zones.keys(), ["#1F5F96", "#5E9FD2", "#18614c", "#8a4f7d"]))
    lo, hi = SKILL_THRESH["r_mod"], SKILL_THRESH["r_high"]
    ymin, ymax = -0.15, 1.0
    hx.axhspan(ymin, 0, color="#f3dad7", alpha=0.6, lw=0)
    hx.axhspan(0, lo, color=SKILL_CATS["low"], alpha=0.35, lw=0)
    hx.axhspan(lo, hi, color=SKILL_CATS["moderate"], alpha=0.35, lw=0)
    hx.axhspan(hi, ymax, color=SKILL_CATS["high"], alpha=0.35, lw=0)
    for yv, lbl in ((ymin + 0) / 2, "negative"), (lo / 2, "low"), ((lo + hi) / 2, "moderate"), ((hi + ymax) / 2, "high"):
        hx.text(8.55, yv, lbl, fontsize=7.5, color=C_MUTED, ha="right", va="center", style="italic")
    # already-observed windows sit left of the issuance
    hx.axvspan(-0.6, xpos[im] - 0.5, color="#ffffff", alpha=0.55, lw=0)
    hx.text(xpos[im] - 0.6, ymin + 0.04, "windows already\nunder way ", fontsize=7, color=C_MUTED, ha="right", va="bottom")
    xs = [xpos.get(((t["start"]) % 12) + 1) for t in tris]
    for row in rows:
        lbl = row["zone"]
        z = zone_clim.get(lbl, clim_all); annual = z.sum()
        ys = [row["r"].get(t["code"], np.nan) for t in tris]
        col = zone_cols.get(lbl, "#3f4748")
        is_all = lbl not in zone_cols
        solo = len(rows) == 1
        hx.plot(xs, ys, color=col, lw=2.2 if (not is_all or solo) else 1.6, ls="-" if (not is_all or solo) else (0, (4, 2)), zorder=3)
        for t, x, yv in zip(tris, xs, ys):
            if x is None or np.isnan(yv):
                continue
            share = z[[((t["start"] - 1 + k) % 12) for k in range(3)]].sum() / annual if annual > 0 else 0
            on = share >= off_share
            hx.plot([x], [yv], marker="o", ms=5.5, color=col, mfc=col if on else "white", mew=1.6, zorder=4)
    hx.set_ylim(ymin, ymax); hx.set_yticks([0, lo, hi, 1.0])
    hx.set_ylabel("median pixel r", fontsize=9, color=C_MUTED)
    hx.set_xlim(-0.6, 8.6); hx.set_xticks(range(9))
    code_at = {xpos.get(((t["start"]) % 12) + 1): t["code"] for t in tris}
    rx.set_xticks(range(9))
    rx.set_xticklabels([MONTH_NAMES[m - 1] + (f"\n{code_at[k]}" if k in code_at else "") for k, m in enumerate(months)])
    hx.axvline(xpos[im], color=C_TEXT, lw=1, ls=(0, (4, 3)), alpha=0.6)
    hx.tick_params(colors=C_MUTED, labelsize=8.5)
    hx.yaxis.grid(False)
    for sp in ("top", "right"):
        hx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        hx.spines[sp].set_color("#c9d0d0")
    hx.legend(handles=[plt.Line2D([], [], color="#3f4748", ls=(0, (4, 2)) if len(rows) > 1 else "-", lw=1.6, label="Whole country"),
                       plt.Line2D([], [], color=C_MUTED, marker="o", mfc="white", ls="", mew=1.6,
                                  label="hollow = off-season window" + ("" if solo_fig else " for that zone") + " (<15% of annual rain)")],
              frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    hx.set_title("Skill of this issuance (median pixel r" + (", whole country)" if len(rows) == 1 else ")"), fontsize=9.5, color=C_TEXT, loc="left")

    # --- third panel: the current forecast's return period, dry above the axis, wet below (log scale)
    yr = skill.get("issued_year")
    sev, vsev, rmax = 3.0, 10.0, 46.0
    tr = lambda v: np.sign(v) * np.log10(max(abs(v), 1.0))          # signed log10
    ylim = np.log10(rmax) * 1.05
    rx.axhspan(np.log10(sev), np.log10(vsev), color="#E8CDB0", alpha=0.45, lw=0)
    rx.axhspan(np.log10(vsev), ylim, color="#B17E50", alpha=0.35, lw=0)
    rx.axhspan(-np.log10(vsev), -np.log10(sev), color="#BFD9EE", alpha=0.45, lw=0)
    rx.axhspan(-ylim, -np.log10(vsev), color="#5E9FD2", alpha=0.35, lw=0)
    rx.axhline(0, color="#9aa3ad", lw=0.8)
    rx.text(8.55, (np.log10(sev) + np.log10(vsev)) / 2, "dry, severe (≥3 yr)", fontsize=7, color="#7A4E22", ha="right", va="center", style="italic")
    rx.text(8.55, (np.log10(vsev) + ylim) / 2, "dry, very severe (≥10 yr)", fontsize=7, color="#7A4E22", ha="right", va="center", style="italic")
    rx.text(8.55, -(np.log10(sev) + np.log10(vsev)) / 2, "wet, severe", fontsize=7, color="#1F5F96", ha="right", va="center", style="italic")
    rx.text(8.55, -(np.log10(vsev) + ylim) / 2, "wet, very severe", fontsize=7, color="#1F5F96", ha="right", va="center", style="italic")
    rx.axvspan(-0.6, xpos[im] - 0.5, color="#ffffff", alpha=0.55, lw=0)
    for row in rows:
        lbl = row["zone"]
        z = zone_clim.get(lbl, clim_all); annual = z.sum()
        col = zone_cols.get(lbl, "#3f4748"); is_all = lbl not in zone_cols
        ys = [tr(row["rp"].get(t["code"], np.nan)) if np.isfinite(row["rp"].get(t["code"], np.nan)) else np.nan for t in tris]
        rx.plot(xs, ys, color=col, lw=2.2 if (not is_all or len(rows) == 1) else 1.6, ls="-" if (not is_all or len(rows) == 1) else (0, (4, 2)), zorder=3)
        for t, x, yv in zip(tris, xs, ys):
            if x is None or np.isnan(yv):
                continue
            share = z[[((t["start"] - 1 + k) % 12) for k in range(3)]].sum() / annual if annual > 0 else 0
            on = share >= off_share
            skilled = row["r"].get(t["code"], np.nan) >= lo
            rx.plot([x], [yv], marker="o", ms=5.5, color=col, mfc=col if skilled else "white", mew=1.6,
                    alpha=1.0 if on else 0.35, zorder=4)
    ticks = [1, 2, 3, 5, 10, 20, 45]
    rx.set_yticks([-np.log10(v) for v in ticks[::-1] if v > 1] + [0] + [np.log10(v) for v in ticks if v > 1])
    rx.set_yticklabels([f"{v}" for v in ticks[::-1] if v > 1] + ["1"] + [f"{v}" for v in ticks if v > 1])
    rx.set_ylim(-ylim, ylim)
    rx.set_ylabel("return period (yr)\nwet ◂   ▸ dry", fontsize=8.5, color=C_MUTED)
    rx.axvline(xpos[im], color=C_TEXT, lw=1, ls=(0, (4, 3)), alpha=0.6)
    rx.tick_params(colors=C_MUTED, labelsize=8.5)
    for sp in ("top", "right"):
        rx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        rx.spines[sp].set_color("#c9d0d0")
    rx.set_title(f"What this issuance forecasts ({MONTH_NAMES[im - 1]} {yr if yr else ''}): return period of the forecast "
                 f"anomaly, median pixel" + ("" if len(rows) == 1 else " per zone"), fontsize=9.5, color=C_TEXT, loc="left")
    rx.legend(handles=[plt.Line2D([], [], color=C_MUTED, marker="o", ls="", mew=1.6, label="filled = skill ≥ moderate" if solo_fig else "filled = zone skill ≥ moderate"),
                       plt.Line2D([], [], color=C_MUTED, marker="o", mfc="white", ls="", mew=1.6, label="hollow = low skill"),
                       plt.Line2D([], [], color=C_MUTED, marker="o", ls="", alpha=0.35, label="faded = off-season" if solo_fig else "faded = off-season for that zone")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Food security context: FEWS NET IPC-compatible classification (ds-fewsnet-mirror)
# --------------------------------------------------------------------------- #
# Classification + unit registry from the mirror's public site JSON (regenerated daily from the
# team's dev DB); unit geometry from the dev blob. Rules from the mirror's README: published
# map = assistance false; phase null = not classified (never Phase 1); drop the admin0 FAOB
# series; key every row on both the collection round and the projection window.
FEWS_SITE = "https://ocha-dap.github.io/ds-fewsnet-mirror/data"
FEWS_BLOB = "ds-fewsnet-mirror/processed/units/{iso3}.geojson"
IPC_COLOURS = {1: "#CDFACD", 2: "#FAE61E", 3: "#E67800", 4: "#C80000", 5: "#640000"}
IPC_LABELS = {1: "1 Minimal", 2: "2 Stressed", 3: "3 Crisis", 4: "4 Emergency", 5: "5 Famine"}


def _fetch_json(url: str, cache: Path):
    import requests
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")) < pd.Timedelta(days=1):
        return json.loads(cache.read_text())
    r = requests.get(url, timeout=60); r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(r.text)
    return r.json()


def _window(start, end) -> str:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    return f"{a:%b %Y}" if (a.year, a.month) == (b.year, b.month) else f"{a:%b %Y}–{b:%b %Y}"


def load_fews(iso3: str) -> dict | None:
    """Latest FEWS NET picture for a country: current situation (latest round that carries a CS)
    and the latest round's ML1 / ML2 projections, one phase per FNID, plus unit geometry."""
    try:
        cls = _fetch_json(f"{FEWS_SITE}/classification/{iso3}.json", Path(f"cache/fews_classification_{iso3}.json"))
        units = _fetch_json(f"{FEWS_SITE}/units/{iso3}.json", Path(f"cache/fews_units_{iso3}.json"))
    except Exception as e:  # noqa: BLE001
        print(f"  (FEWS NET classification not available: {e})"); return None
    geo_path = Path(f"cache/fews_units_{iso3}.geojson")
    if not geo_path.exists():
        try:
            import ocha_stratus as stratus
            raw = stratus.load_blob_data(FEWS_BLOB.format(iso3=iso3), stage="dev")
            geo_path.write_bytes(raw if isinstance(raw, (bytes, bytearray)) else raw.read())
        except Exception as e:  # noqa: BLE001
            print(f"  (FEWS NET geometry not available: {e})"); return None
    gdf = gpd.read_file(geo_path)
    cols = cls["columns"]; U = cls["units"]; docs = cls["docs"]; st = cls["statuses"]
    rows = pd.DataFrame([dict(zip(cols, r)) for r in cls["rows"]])
    rows["fnid"] = rows.u.map(lambda i: U[i][0]); rows["unit_type"] = rows.u.map(lambda i: U[i][2])
    rows["doc"] = rows.doc.map(lambda i: docs[i]); rows["status"] = rows.st.map(lambda i: st[i])
    rows = rows[(rows.assistance == 0) & (rows.unit_type != "admin0") & (rows.status == "Collected") & rows.phase.notna()]
    latest = rows.reporting_date.max()
    picks = {}
    cs_rounds = rows[rows.scenario == "CS"].reporting_date
    if len(cs_rounds):
        rd = cs_rounds.max(); sub = rows[(rows.scenario == "CS") & (rows.reporting_date == rd)]
        picks["CS"] = dict(round=rd, doc=sub.doc.iloc[0], start=sub.projection_start.iloc[0], end=sub.projection_end.iloc[0],
                           phase=sub.set_index("fnid").phase.astype(int).to_dict())
    for sc in ("ML1", "ML2"):
        sub = rows[(rows.scenario == sc) & (rows.reporting_date == latest)]
        if len(sub):
            picks[sc] = dict(round=latest, doc=sub.doc.iloc[0], start=sub.projection_start.iloc[0], end=sub.projection_end.iloc[0],
                             phase=sub.set_index("fnid").phase.astype(int).to_dict())
    # Per year: the pre-season outlook for the October–January window (the same product as the
    # current medium-term projection) — latest round issued by September whose ML1/ML2 window starts
    # 1 October; fallback to an October-issued ML1 (marked). Shares are of classified units (unit
    # counts: older FNID vintages carry no geometry in the mirror).
    hist = {}

    def share(h):
        return dict(n=int(len(h)), p3=float((h.phase == 3).mean()), p4=float((h.phase == 4).mean()),
                    p5=float((h.phase == 5).mean()))

    ml = rows[rows.scenario.isin(["ML1", "ML2"])]
    cs = rows[rows.scenario == "CS"]
    years = sorted({int(x[:4]) for x in rows.projection_start})
    for yr in years:
        entry = {}
        # pre-season outlook: window starting 1 Oct Y, latest round issued by Sep Y (fallback: Oct-issued)
        g = ml[ml.projection_start == f"{yr}-10-01"]
        pre = g[g.reporting_date <= f"{yr}-09"]
        pick = pre if len(pre) else g[g.reporting_date == f"{yr}-10"]
        if len(pick):
            rd = pick.reporting_date.max(); h = pick[pick.reporting_date == rd]
            entry["pre"] = share(h) | dict(round=rd, doc=h.doc.iloc[0], scenario=h.scenario.iloc[0], pre_season=bool(len(pre)),
                                           window=_window(h.projection_start.iloc[0], h.projection_end.iloc[0]))
        # in-season outlook: window starting 1 Feb Y+1, issued Oct Y – Jan Y+1 (earliest such round = the October outlook)
        g = ml[(ml.projection_start == f"{yr + 1}-02-01") & (ml.reporting_date >= f"{yr}-10") & (ml.reporting_date <= f"{yr + 1}-01")]
        if len(g):
            rd = g.reporting_date.min(); h = g[g.reporting_date == rd]
            entry["mid"] = share(h) | dict(round=rd, doc=h.doc.iloc[0], scenario=h.scenario.iloc[0],
                                           window=_window(h.projection_start.iloc[0], h.projection_end.iloc[0]))
        # observed: current situation at the lean-season peak, Jan–Apr Y+1 (round with the largest Phase 3+ share)
        g = cs[(cs.projection_start >= f"{yr + 1}-01-01") & (cs.projection_start <= f"{yr + 1}-04-30")]
        if len(g):
            best = None
            for rd, h in g.groupby("reporting_date"):
                v = share(h); v["round"] = rd; v["window"] = _window(h.projection_start.iloc[0], h.projection_end.iloc[0])
                if best is None or (v["p3"] + v["p4"] + v["p5"]) > (best["p3"] + best["p4"] + best["p5"]):
                    best = v
            entry["obs"] = best
        if entry:
            hist[yr] = entry
    ucols = units["columns"]; ureg = pd.DataFrame([dict(zip(ucols, r)) for r in units["rows"]]).set_index("fnid")
    key = "fnid" if "fnid" in gdf.columns else [c for c in gdf.columns if c.lower() in ("fnid", "pcode")][0]
    gdf = gdf.rename(columns={key: "fnid"})
    return dict(gdf=gdf, picks=picks, units=ureg, generated=cls.get("generated_at"), history=hist)


def fig_fews_maps(c: Country, fews: dict, hit: np.ndarray | None, analysable: np.ndarray | None,
                  season: str, out: Path, name: str) -> None:
    """FEWS NET phase maps (CS / ML1 / ML2) drawn on FEWS NET's own unit polygons, with the El Niño
    driest-third hit-rate map alongside so drought exposure and food insecurity can be read together."""
    order = [sc for sc in ("CS", "ML1", "ML2") if sc in fews["picks"]]
    n = len(order) + (1 if hit is not None else 0)
    w, h = _panel_size(c)
    ncols = 2 if (n >= 4 and w > 3.2) else n
    nrows = int(np.ceil(n / ncols))
    H = h * nrows + 1.6 + 0.6 * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols + 1.6, H), dpi=150)
    axes = np.atleast_1d(axes).ravel(); ext = _extent(c)
    fig.subplots_adjust(top=1 - 1.0 / H, bottom=0.55 / H, left=0.02, right=0.98, wspace=0.12, hspace=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    gdf = fews["gdf"]
    for ax, sc in zip(axes, order):
        pk = fews["picks"][sc]
        g = gdf.copy(); g["phase"] = g.fnid.map(pk["phase"])
        g[g.phase.isna()].plot(ax=ax, color="#ececec", edgecolor="white", linewidth=0.3)
        for ph, col in IPC_COLOURS.items():
            sub = g[g.phase == ph]
            if len(sub):
                sub.plot(ax=ax, color=col, edgecolor="white", linewidth=0.3)
        _draw_country(ax, c, ext)
        lbl = {"CS": "Current situation", "ML1": "Near-term projection", "ML2": "Medium-term projection"}[sc]
        ax.set_title(f"{lbl}\n{_window(pk['start'], pk['end'])} · round {pk['round']}", fontsize=9, color=C_TEXT, loc="left")
    if hit is not None:
        ax = axes[n - 1]
        seq = mcolors.LinearSegmentedColormap.from_list("dry", ["#F7F1EA", "#E8CDB0", "#B17E50", "#7A4E22"])
        ok = c.mask & analysable
        m = _pcolor(ax, c, np.where(ok, hit * 100, np.nan), seq, 0, 100)
        gdf.boundary.plot(ax=ax, color="#ffffff", linewidth=0.35, alpha=0.9)
        _draw_country(ax, c, ext)
        ax.set_title(f"El Niño years in the cell's driest third\nof {season} (chance 33%), FEWS NET units outlined", fontsize=9, color=C_TEXT, loc="left")
        cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.03); cb.ax.tick_params(labelsize=7.5, colors=C_MUTED)
    fig.legend(handles=[Patch(color=IPC_COLOURS[k], label=IPC_LABELS[k]) for k in range(1, 6)] + [Patch(color="#ececec", label="not classified")],
               frameon=False, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{name}: FEWS NET acute food insecurity (IPC-compatible, not allowing for assistance), on FEWS NET's own units"
                 + (" — beside the El Niño drought hit-rate" if hit is not None else ""),
                 fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.995, va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Admin-1 view from the team's ERA5 raster stats (public.era5, per-admin monthly means)
# --------------------------------------------------------------------------- #
def _pcode_col(gdf) -> str:
    for c in gdf.columns:
        if c.lower() in ("adm1_pcode", "pcode", "adm1_code"):
            return c
    raise KeyError(f"no admin-1 pcode column in {list(gdf.columns)}")


def load_adm1(iso3: str):
    """CODAB admin-1 polygons (FieldMaps via ocha-stratus) with the DB's names, cached locally."""
    cache = Path(f"cache/adm1_{iso3}.parquet")
    if cache.exists():
        return gpd.read_parquet(cache)
    from ocha_stratus import codab
    g = codab.load_codab_from_blob(iso3.lower(), admin_level=1).to_crs("EPSG:4326")
    pc = _pcode_col(g)
    name_col = next((c for c in g.columns if c.lower() in ("adm1_en", "adm1_name", "name")), None)
    g = g.rename(columns={pc: "pcode"})[["pcode", "geometry"] + ([name_col] if name_col else [])]
    if name_col:
        g = g.rename(columns={name_col: "name"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(cache)
    return g


def load_era5_adm1(iso3: str) -> pd.DataFrame:
    """Monthly ERA5 precipitation means per admin-1 (mm/day) from the prod DB, cached as parquet."""
    cache = Path(f"cache/era5_adm1_{iso3}.parquet")
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")) < pd.Timedelta(days=30):
        return pd.read_parquet(cache)
    import os
    os.environ.setdefault("PGSSLMODE", "require")
    import ocha_stratus as stratus
    eng = stratus.get_engine(stage="prod")
    df = pd.read_sql("SELECT e.pcode, e.valid_date, e.mean, e.count, p.name FROM public.era5 e "
                     "LEFT JOIN public.polygon p ON p.pcode = e.pcode AND p.adm_level = 1 "
                     "WHERE e.iso3 = %(iso)s AND e.adm_level = 1 ORDER BY e.pcode, e.valid_date", eng, params={"iso": iso3})
    df["valid_date"] = pd.to_datetime(df.valid_date)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def adm1_enso_stats(iso3: str, months: list[int], indices: pd.DataFrame, years: np.ndarray) -> pd.DataFrame | None:
    """Per admin-1: headline-season mean series from the raster stats, concurrent Niño3.4 r,
    and the share of El Niño / La Niña seasons in the unit's own driest third."""
    try:
        df = load_era5_adm1(iso3)
    except Exception as e:  # noqa: BLE001
        print(f"  (ERA5 admin-1 raster stats not available: {e})"); return None
    if df.empty:
        return None
    wrap = 12 in months and 1 in months
    n0 = nino_series(indices, months, 0)
    rows = []
    for pcode, g in df.groupby("pcode"):
        s = g.set_index("valid_date")["mean"]
        vals, yrs = [], []
        for sy in years:
            ts = [pd.Timestamp(year=sy + (1 if (wrap and m <= 6) else 0), month=m, day=1) for m in months]
            if all(t in s.index for t in ts):
                vals.append(float(np.mean([s[t] for t in ts]))); yrs.append(sy)
        d = pd.DataFrame({"rain": vals}, index=yrs).join(n0.rename("nino")).dropna()
        if len(d) < 20:
            continue
        d["pct"] = d.rain.rank(pct=True)
        d["phase"] = np.where(d.nino >= ENSO_THRESH, "El Niño", np.where(d.nino <= -ENSO_THRESH, "La Niña", "Neutral"))
        en, ln = d[d.phase == "El Niño"], d[d.phase == "La Niña"]
        z = (d.rain - d.rain.mean()) / d.rain.std()
        rows.append(dict(pcode=pcode, name=g["name"].iloc[0] if pd.notna(g["name"].iloc[0]) else pcode,
                         n_px=int(g["count"].iloc[0]), r=float(np.corrcoef(d.rain, d.nino)[0, 1]),
                         en_hit=float((en.pct <= 1 / 3).mean()) if len(en) else np.nan, n_en=len(en),
                         ln_hit=float((ln.pct <= 1 / 3).mean()) if len(ln) else np.nan,
                         en_z=float(z[en.index].mean()) if len(en) else np.nan,
                         seasons=", ".join(str(y) for y in en[en.pct <= 1 / 3].index)))
    return pd.DataFrame(rows).sort_values("r")


def fig_adm1_maps(c: Country, adm: gpd.GeoDataFrame, stats: pd.DataFrame, season: str, out: Path, name: str) -> None:
    g = adm.merge(stats, on="pcode", how="left")
    panels = [("r", DIVERGING, -0.7, 0.7, f"{season} rainfall vs Niño3.4\n(admin-1 mean series, concurrent)", "Pearson r"),
              ("en_hit", mcolors.LinearSegmentedColormap.from_list("dry", ["#F7F1EA", "#E8CDB0", "#B17E50", "#7A4E22"]), 1 / 3, 1,
               f"El Niño {season} seasons in the\nunit's driest third (chance 33%)", "share (scale starts at chance)")]
    n = len(panels)
    w, h = _panel_size(c)
    fig, axes = plt.subplots(1, n, figsize=(w * n + 1.8, h + 1.1), dpi=150, layout="constrained")
    axes = np.atleast_1d(axes); ext = _extent(c)
    for ax, (col, cmap, vmin, vmax, title, cblabel) in zip(axes, panels):
        g.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.8,
               missing_kwds=dict(color="#ececec"))
        c.neighbours.boundary.plot(ax=ax, color="#b8bfbf", linewidth=0.6)
        for _, row in g.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            pt = row.geometry.representative_point()
            lbl = str(row.get("name_y", row.get("name", "")) or "")
            ax.text(pt.x, pt.y, lbl, fontsize=6, ha="center", va="center", color=C_TEXT,
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(linewidth=1.5, foreground="white")])
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9d0d0")
        ax.set_title(title, fontsize=9, color=C_TEXT, loc="left")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax)); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.03); cb.ax.tick_params(labelsize=7.5, colors=C_MUTED)
        cb.set_label(cblabel, fontsize=8, color=C_MUTED)
    fig.suptitle(f"{name}: admin-1 view from the team's ERA5 raster stats (public.era5, per-province monthly means)",
                 fontsize=10, color=C_TEXT, x=0.01, ha="left")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Season-by-season table: rainfall, ENSO phase, CERF drought allocations, FEWS NET population
# --------------------------------------------------------------------------- #
def load_cerf_drought(iso3: str) -> pd.DataFrame:
    """CERF drought allocations for a country from the team's OneGMS mirror (aa.cerf_allocation,
    dev DB) joined to the drought-period supplement (aa.cerf_supplement). Cached as parquet."""
    cache = Path(f"cache/cerf_drought_{iso3}.parquet")
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")) < pd.Timedelta(days=7):
        return pd.read_parquet(cache)
    import os
    os.environ.setdefault("PGSSLMODE", "require")
    import ocha_stratus as stratus
    eng = stratus.get_engine(stage="dev")
    df = pd.read_sql(
        "SELECT a.application_code, a.year, a.window_name, a.emergency_type, a.title, a.amount_approved, "
        "a.first_project_approved_date, s.valid_month_start, s.valid_year_start, s.valid_month_end, s.valid_year_end, s.not_drought "
        "FROM aa.cerf_allocation a LEFT JOIN aa.cerf_supplement s ON s.application_code = a.application_code "
        "WHERE a.country_iso3 = %(iso)s AND a.emergency_type = 'Drought' ORDER BY a.first_project_approved_date", eng, params={"iso": iso3})
    df = df[df.not_drought.isna() | (df.not_drought == False)]  # noqa: E712
    df["first_project_approved_date"] = pd.to_datetime(df.first_project_approved_date)
    cache.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(cache)
    return df


def season_table(iso3: str, df: pd.DataFrame, head: str, fews: dict | None, forecast_year: int | None = None) -> list[dict]:
    """One row per headline season since 1981: rainfall anomaly/rank, ENSO phase, CERF drought
    allocations attributed to the season, FEWS NET population in Phase 3+ (peak in the consumption
    year Apr Y+1 – Mar Y+2) and the share of FEWS NET units in Phase 3+ / 4+ at that peak."""
    n_all = len(df)
    try:
        cerf = load_cerf_drought(iso3)
    except Exception as e:  # noqa: BLE001
        print(f"  (CERF mirror not available: {e})"); cerf = pd.DataFrame()
    outlook = fews["history"] if (fews and fews.get("history")) else {}

    def build_row(yr: int, r):
        cy0, cy1 = pd.Timestamp(yr + 1, 4, 1), pd.Timestamp(yr + 2, 3, 31)   # consumption year
        # CERF: supplement valid period first, else allocation date in the consumption year
        cerf_txt, cerf_usd = "", 0.0
        if len(cerf):
            def season_of(a):
                """Season the allocation responds to: the supplement's dated rainy season when present
                (starts Jul–Dec → that year; Jan–Jun → the year before), else the consumption year the
                allocation date falls in (Apr Y+1 – Mar Y+2 → season Y)."""
                if pd.notna(a.valid_year_start):
                    y0 = int(a.valid_year_start); m0 = int(a.valid_month_start) if pd.notna(a.valid_month_start) else 11
                    return (y0 if m0 >= 7 else y0 - 1), True
                d = a.first_project_approved_date
                return (d.year - 2 if d.month <= 3 else d.year - 1), False
            att = [season_of(a) for _, a in cerf.iterrows()]
            hits = cerf[[y == yr for y, _ in att]]
            dated = [ok for (y, ok) in att if y == yr]
            cerf_usd = float(hits.amount_approved.sum()) if len(hits) else 0.0
            if len(hits):
                cerf_txt = f'US$ {hits.amount_approved.sum() / 1e6:.1f} M · ' + "; ".join(
                    f'{a.first_project_approved_date:%b %Y} ({a.window_name.split()[0]}, {a.amount_approved / 1e6:.1f} M{"" if ok else ", by date"})'
                    for (_, a), ok in zip(hits.iterrows(), dated))
        # FEWS NET pre-season outlook for Oct Y – Jan Y+1 (issued Jun–Sep Y): shares of units in Phase 3 / 4 / 5
        fw = (outlook.get(yr) or {}).get("pre")
        if fw:
            fews_txt = f'{pct(fw["p3"])} / {pct(fw["p4"])} / {pct(fw["p5"])}'
            fews_src = f'{fw["window"]} · round {fw["round"]} ({fw["scenario"]}{"" if fw["pre_season"] else ", issued in October"}) · {fw["n"]} units'
            p3, p4, p5 = fw["p3"], fw["p4"], fw["p5"]
        else:
            fews_txt, fews_src, p3, p4, p5 = "", "", np.nan, np.nan, np.nan
        fews_all = outlook.get(yr) or {}
        extra = dict(cerf_usd=cerf_usd, p3=p3, p4=p4, p5=p5, fews=fews_txt, fews_src=fews_src, fews_all=fews_all)
        if r is None:
            return dict(year=yr, label=f"{yr}/{str(yr + 1)[-2:]}", z=None, rank=None, n=n_all, nino=None, phase="Neutral",
                        cerf=cerf_txt, forecast=None, gap=True, **extra)
        return dict(year=yr, label=f"{yr}/{str(yr + 1)[-2:]}", z=float(r.z), rank=int(round(r.pct * n_all)), n=n_all,
                    nino=float(r.nino), phase=r.phase, cerf=cerf_txt, forecast=None, gap=False, **extra)

    rows = [build_row(int(yr), r) for yr, r in df.iterrows()]
    last = int(df.index.max())
    for yr in range(last + 1, (forecast_year or last + 1)):
        rows.append(build_row(yr, None))
    return rows


def forecast_row(a: dict, head: str, fews: dict | None) -> dict | None:
    """A final row for the season now being forecast: SEAS5 return period, current ENSO state,
    FEWS NET's projection (units in Phase 3+ / 4+) and its projected Peak Needs population."""
    sk, nn = a.get("skill"), a.get("nino_now")
    if not sk or not nn:
        return None
    yr = int(sk.get("issued_year") or int(a["df"].index.max()) + 1)
    hm0 = ts._TRIMESTER_MONTHS[head][0]
    if hm0 < sk["issued_month"] and hm0 <= 6:     # season starts next calendar year (e.g. JFM issued in Sep)
        yr = yr  # season year labels the year the season's first month... keep the issuance year as the season label
    wc = [r for r in sk["rows"] if r["zone"].startswith("Whole country")] or sk["rows"]
    rp = wc[0]["rp"].get(head, np.nan); r_sk = wc[0]["r"].get(head, np.nan)
    rp_txt = "—" if np.isnan(rp) else (f'SEAS5 {MONTH_NAMES[sk["issued_month"] - 1]} issuance: {"dry" if rp > 0 else "wet"}, return period {abs(rp):.0f} yr ({skill_cat(r_sk)} skill)')
    fews_all = (fews or {}).get("history", {}).get(yr) or {}
    fw = fews_all.get("pre")
    if fw:
        fews_txt = f'{pct(fw["p3"])} / {pct(fw["p4"])} / {pct(fw["p5"])}'
        fews_src = f'{fw["window"]} · round {fw["round"]} ({fw["scenario"]}) · {fw["n"]} units'
        p3f, p4f, p5f = fw["p3"], fw["p4"], fw["p5"]
    else:
        fews_txt, fews_src, p3f, p4f, p5f = "", "", np.nan, np.nan, np.nan
    return dict(year=yr, label=f"{yr}/{str(yr + 1)[-2:]} (forecast)", z=None, rank=None, n=None, rp_txt=rp_txt,
                nino=nn["value"], phase=nn["phase"].replace("neutral", "Neutral"), nino_date=nn["date"],
                cerf="", forecast=True, rp=rp, cerf_usd=0.0, p3=p3f, p4=p4f, p5=p5f, fews=fews_txt, fews_src=fews_src, fews_all=fews_all)


def fig_seasons(rows: list[dict], head: str, out: Path, name: str, start_year: int = 2006) -> None:
    """Stacked time series of the season table: rainfall by ENSO phase, CERF drought allocations,
    FEWS NET people in Phase 3+ (published ranges), share of FEWS NET units in Phase 3+."""
    all_hist = [r for r in rows if not r.get("forecast") and not r.get("gap")]
    z_all = pd.Series([r["z"] for r in all_hist])
    thr = float(z_all.quantile(1 / 3))                       # driest-third threshold from the whole record
    first_fews = min([r["year"] for r in rows if r.get("fews_all")] or [start_year])
    rows = [r for r in rows if r["year"] >= start_year]
    hist = [r for r in rows if not r.get("forecast") and not r.get("gap")]
    fc = next((r for r in rows if r.get("forecast")), None)
    yrs = np.array([r["year"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.6), dpi=150, sharex=True,
                             gridspec_kw=dict(height_ratios=[2.0, 1.1, 1.9], hspace=0.12))
    ax = axes[0]
    col = {"El Niño": C_ELNINO, "Neutral": C_NEUTRAL, "La Niña": C_LANINA}
    ax.bar([r["year"] for r in hist], [r["z"] for r in hist], color=[col[r["phase"]] for r in hist], width=0.78)
    ax.axhline(thr, color="#7A4E22", lw=1, ls=(0, (4, 3)))
    ax.text(yrs.min() - 0.4, thr - 0.05, "driest third", fontsize=6.5, color="#7A4E22", va="top", ha="left")
    ax.axhline(0, color="#9aa3ad", lw=0.8)
    for r in hist:
        if r["phase"] == "El Niño":
            ax.text(r["year"], r["z"] + (0.08 if r["z"] >= 0 else -0.08), r["label"], fontsize=6.5, color=C_ELNINO,
                    ha="center", va="bottom" if r["z"] >= 0 else "top", rotation=90)
    if fc and fc.get("rp") is not None and np.isfinite(fc["rp"]):
        from scipy.stats import norm
        pctl = 1 / abs(fc["rp"]); z_eq = float(norm.ppf(pctl if fc["rp"] > 0 else 1 - pctl))
        ax.bar([fc["year"]], [z_eq], color="none", edgecolor=C_ELNINO if fc["phase"] == "El Niño" else C_TEXT, hatch="///", width=0.78, lw=1.2)
        ax.text(fc["year"], z_eq - 0.08, "forecast", fontsize=6.5, color=C_TEXT, ha="center", va="top", rotation=90)
    lo = min(float(z_all.min()), z_eq if (fc and fc.get("rp") is not None and np.isfinite(fc["rp"])) else 0) - 0.9
    ax.set_ylim(lo, float(z_all.max()) + 0.4)
    ax.set_ylabel(f"{head} rainfall (SD)", fontsize=8.5, color=C_MUTED)
    ax.legend(handles=[Patch(color=C_ELNINO, label="El Niño"), Patch(color=C_NEUTRAL, label="Neutral"), Patch(color=C_LANINA, label="La Niña"),
                       Patch(facecolor="none", edgecolor=C_TEXT, hatch="///", label="SEAS5 forecast (return period → percentile)")],
              frameon=False, fontsize=7.5, loc="upper left", ncol=4)
    ax.set_title(f"{name}: season by season — rainfall and ENSO, CERF drought funding, FEWS NET outlooks and observed", fontsize=10, color=C_TEXT, loc="left")
    _style_ax(ax)
    ax = axes[1]
    ax.bar([r["year"] for r in rows], [r.get("cerf_usd", 0) / 1e6 for r in rows], color="#B17E50", width=0.78)
    for r in rows:
        if r.get("cerf_usd", 0) > 0:
            ax.text(r["year"], r["cerf_usd"] / 1e6 + 0.3, f'{r["cerf_usd"] / 1e6:.0f}', fontsize=6.5, color=C_TEXT, ha="center", va="bottom")
    ax.set_ylabel("CERF drought\nallocations (US$ M)", fontsize=8.5, color=C_MUTED); _style_ax(ax)
    ax = axes[2]
    plt.rcParams["hatch.linewidth"] = 0.6
    kinds = [("pre", -0.28, "pre-season outlook (issued Jun–Sep, for Oct–Jan)", dict(hatch="///", edgecolor="#5a3a12", linewidth=0)),
             ("mid", 0.0, "in-season outlook (issued Oct, for Feb–May)", dict(hatch="\\\\\\", edgecolor="#5a3a12", linewidth=0)),
             ("obs", 0.28, "observed at the lean-season peak (Jan–Apr)", dict(edgecolor="none", linewidth=0))]
    for key, off, _, style in kinds:
        for r in rows:
            e = (r.get("fews_all") or {}).get(key)
            if not e:
                continue
            x = r["year"] + off; base = 0.0
            for ph in (3, 4, 5):
                v = 100 * e[f"p{ph}"]
                if v > 0:
                    ax.bar([x], [v], bottom=base, width=0.26, color=IPC_COLOURS[ph], **style)
                    base += v
    ax.set_ylabel("FEWS NET units in\nPhase 3 / 4 / 5 (%)", fontsize=8.5, color=C_MUTED); ax.set_ylim(0, 105); _style_ax(ax)
    handles = [Patch(color=IPC_COLOURS[3], label="Phase 3"), Patch(color=IPC_COLOURS[4], label="Phase 4"), Patch(color=IPC_COLOURS[5], label="Phase 5"),
               Patch(facecolor="#d9d9d9", hatch="///", edgecolor="#5a3a12", linewidth=0, label="left: pre-season outlook (Jun–Sep, for Oct–Jan)"),
               Patch(facecolor="#d9d9d9", hatch="\\\\\\", edgecolor="#5a3a12", linewidth=0, label="middle: in-season outlook (Oct, for Feb–May)"),
               Patch(facecolor="#d9d9d9", edgecolor="none", label="right: observed, lean-season peak (Jan–Apr)")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", ncol=2)
    first_p3 = min([r["year"] for r in rows if any((e.get("p3", 0) + e.get("p4", 0) + e.get("p5", 0)) > 0 for e in (r.get("fews_all") or {}).values())] or [yrs[0]])
    if first_p3 > yrs[0]:
        ax.text(first_p3 - 0.6, 40, f"FEWS NET classifies from {first_fews}/{str(first_fews + 1)[-2:]};\nno unit in Phase 3+ until {first_p3}/{str(first_p3 + 1)[-2:]} ▸",
                fontsize=7, color=C_MUTED, ha="right", va="center")
    ax.set_xlim(yrs.min() - 1, yrs.max() + 1)
    ax.set_xticks(list(yrs)); ax.set_xticklabels([f"{y}/{str(y + 1)[-2:]}" for y in yrs], rotation=90)
    ax.tick_params(labelsize=8, colors=C_MUTED)
    for a_ in axes[1:]:
        a_.tick_params(labelsize=8, colors=C_MUTED)
    if fc:
        for a_ in axes:
            a_.axvspan(fc["year"] - 0.5, fc["year"] + 0.5, color="#f3f3f1", zorder=0)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def render_seasons(spec: dict, a: dict, head: str, title: str | None = None) -> str:
    rows = a["seasons"]
    out = [f'<h2>{html.escape(title) if title else f"Season by season: {head} rainfall, ENSO, CERF and FEWS NET"}</h2>']
    out.append(spec.get("seasons_html", ""))
    out.append(f'<figure><img src="seasons.png" alt="Season-by-season time series"><figcaption>The table as a time series: '
               f'{head} rainfall anomaly coloured by ENSO phase (hatched = the SEAS5 forecast for the coming season, its return period '
               f'converted to a percentile); CERF drought allocations by season; and, per season, three FEWS NET readings as stacked '
               f'shares of units in Phase 3, 4 and 5 — the pre-season outlook (issued June–September for October–January, the product '
               f'we have now), the in-season outlook (issued in October for February–May) and the observed current situation at the '
               f'lean-season peak (January–April). From 2006/07, when CERF began. Shaded column = the season being forecast.</figcaption></figure>')
    rows = list(reversed(rows))   # recent first, forecast on top
    max_cerf = max([r.get("cerf_usd", 0) for r in rows] + [1.0])

    def shade(v, vmax, rgb=(230, 120, 0), floor=0.08):
        """Background colour: darker for worse; white when zero/absent."""
        if v is None or not np.isfinite(v) or v <= 0:
            return ""
        a = floor + 0.55 * min(v / vmax, 1.0)
        return f' style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f})"'

    def shade_rain(r):
        if r.get("forecast") or r.get("gap") or r.get("rank") is None:
            return ""
        q = r["rank"] / r["n"]            # 0 = driest
        if q <= 0.5:
            return f' style="background:rgba(122,78,34,{0.08 + 0.6 * (0.5 - q) / 0.5:.2f})"'
        return f' style="background:rgba(94,159,210,{0.05 + 0.3 * (q - 0.5) / 0.5:.2f})"'

    out.append('<div style="overflow-x:auto"><table class="skill seasons"><thead><tr><th>Season</th>'
               f'<th class="num">{head} rainfall<br><span class="small">anomaly (SD) · rank, 1 = driest</span></th>'
               f'<th>ENSO<br><span class="small">Niño3.4 in {head}</span></th>'
               '<th>CERF drought allocation<br><span class="small">for this season\'s drought</span></th>'
               '<th class="num">FEWS NET outlook for Oct–Jan, issued before the season<br><span class="small">share of units in Phase 3 / 4 / 5</span></th></tr></thead><tbody>')
    for r in rows:
        cls = ' class="hl"' if r["phase"] == "El Niño" else ""
        ph = {"El Niño": "#D1495B", "La Niña": "#2E7DBD", "Neutral": "#9AA3AD"}[r["phase"]]
        if r.get("forecast"):
            rain = f'<span class="small">{html.escape(r["rp_txt"])}</span>'
            enso = (f'<span class="chip" style="background:{ph};color:#fff;margin:0">{r["phase"]}</span> '
                    f'<span class="small">{r["nino"]:+.2f} in {r["nino_date"]:%b %Y}</span>')
            cls = ' class="hl" style="border-top:2px solid #c9d0d0"'
        elif r.get("gap"):
            rain = '<span class="small">ERA5 season not yet in the cache</span>'
            enso = '<span class="small">—</span>'; cls = ""
        else:
            rain = f'{fmt_r(r["z"])} · {r["rank"]} of {r["n"]}'
            enso = f'<span class="chip" style="background:{ph};color:#fff;margin:0">{r["phase"]}</span> <span class="small">{r["nino"]:+.2f}</span>'
        cls = cls.replace(' class="hl"', "")   # phase is carried by the chip; row highlight would fight the cell shading
        p3p = (r.get("p3", np.nan) + r.get("p4", np.nan) + r.get("p5", np.nan)) if np.isfinite(r.get("p3", np.nan)) else np.nan
        out.append(f'<tr{cls}><td>{"<strong>" + r["label"] + "</strong>" if r.get("forecast") else r["label"]}</td>'
                   f'<td class="num"{shade_rain(r)}>{rain}</td><td>{enso}</td>'
                   f'<td{shade(r.get("cerf_usd", 0), max_cerf, (177, 126, 80))}>{html.escape(r["cerf"]) if r["cerf"] else "<span class=small>—</span>"}</td>'
                   f'<td class="num"{shade(p3p, 1.0)}>{r.get("fews") or "<span class=small>—</span>"}' + (f'<br><span class="small">{html.escape(r["fews_src"])}</span>' if r.get("fews_src") else "") + '</td></tr>')
    out.append('</tbody></table></div>')
    out.append('<p class="small">Rainfall: ERA5 area mean over the rain-fed cells, standardised over 1981–2025. ENSO phase: concurrent Niño3.4 '
               f'(≥ +{ENSO_THRESH} El Niño, ≤ −{ENSO_THRESH} La Niña; pinned NOAA series). CERF: Rapid Response / Underfunded Emergencies applications with '
               'emergency type “Drought” from the team\'s OneGMS mirror, attributed to the rainy season named in the CERF drought-period '
               'supplement (or, failing that, to the season whose harvest year the allocation fell in; “by date”). FEWS NET: the outlook for '
               'October–January issued between June and September of that year — the same product as the current medium-term projection, so '
               'each row shows the pre-season picture that season started from — as shares of FEWS NET\'s classified units in Phase 3, 4 and 5 '
               '(unit counts, since older unit vintages carry no geometry; FEWS NET classifies areas and publishes no population by phase). '
               '“Issued in October” marks a year with no June–September outlook. Cells are shaded darker for worse outcomes; most recent '
               'season first, the forecast season on top.</p>')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Analysis for one country
# --------------------------------------------------------------------------- #
def analyse(spec: dict, grid: Grid, gdf: gpd.GeoDataFrame, indices: pd.DataFrame, cfg: dict,
            out_dir: Path) -> dict:
    iso3, name = spec["iso3"], spec["name"]
    c = cut_country(grid, gdf, iso3)
    head = spec["headline_season"]
    hm = season_months(head)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annual total from the four canonical trimesters (same rule as the survey).
    clim = {t: season_stack(c, grid, season_months(t))[0].mean(0) for t in ["DJF", "MAM", "JAS", "OND"]}
    annual = sum(clim.values())

    def analysable(months):
        s = season_stack(c, grid, months)[0].mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return c.mask & (annual > 0) & (s / annual >= 0.25) & (s >= ts.PIXEL_MIN_TRI_MM_DAY), s

    ok_head, clim_head = analysable(hm)
    # Zones: bounded by headline-season climatology (mm/day) and/or latitude (decimal degrees,
    # south negative; a cell centred exactly on a bound goes to the zone south of it),
    # intersected with the analysable headline-season cells.
    lat2d = np.repeat(c.lat[:, None], len(c.lon), axis=1)
    zones = {}
    for z in spec.get("zones", []):
        zm = ok_head & (clim_head >= z.get("min_mm_day", 0.0)) & (clim_head < z.get("max_mm_day", 1e9))
        zm &= (lat2d > z.get("min_lat", -90.0)) & (lat2d <= z.get("max_lat", 90.0))
        zones[z["label"]] = zm
    fig_seasonal_cycle(c, grid, hm, out_dir / "seasonal_cycle.png", name, zones or None)
    if zones:
        fig_zone_map(c, zones, out_dir / "zones_map.png", name, head)
    # `zones_analysis = false` keeps the zones for the seasonal cycle + locator map only (to show a
    # unimodal, uniform country) and runs everything downstream on the whole country.
    if not spec.get("zones_analysis", True):
        zones = {}

    # Correlation maps: headline + any extra seasons, best lag 0..3 (as in the survey)
    panels, summaries = [], []
    map_seasons = list(spec.get("map_seasons", []))
    if head not in map_seasons:
        map_seasons.insert(0, head)
    for code in map_seasons:
        months = season_months(code)
        R, yrs = season_stack(c, grid, months)
        r, k = best_lag_corr(R, yrs, indices, months, cfg.get("max_lag", 3))
        ok, s = analysable(months)
        v = r[ok]
        p = ts._pearson_p(v, len(yrs))
        share = (s / annual)[c.mask]
        summ = dict(season=code, n_cells=int(ok.sum()), n_country=int(c.mask.sum()),
                    median_r=float(np.nanmedian(v)) if v.size else float("nan"),
                    min_r=float(np.nanmin(v)) if v.size else float("nan"),
                    max_r=float(np.nanmax(v)) if v.size else float("nan"),
                    frac_sig_neg=float(((p < .05) & (v < 0)).mean()) if v.size else 0.0,
                    frac_sig_pos=float(((p < .05) & (v > 0)).mean()) if v.size else 0.0,
                    frac_strong_neg=float((v <= -0.5).mean()) if v.size else 0.0,
                    frac_strong_pos=float((v >= 0.5).mean()) if v.size else 0.0,
                    frac_mod_neg=float((v <= -0.3).mean()) if v.size else 0.0,
                    frac_mod_pos=float((v >= 0.3).mean()) if v.size else 0.0,
                    mean_share=float(np.nanmean(share)))
        summaries.append(summ)
        panels.append(dict(r=r, analysable=ok, title=f"{code} vs Niño3.4\n(best lag 0–{cfg.get('max_lag', 3)} mo)",
                           note=f"{summ['n_cells']} of {summ['n_country']} cells\nhave a {code} season"))
    fig_corr_maps(c, panels, out_dir / "corr_maps.png", name)

    # Phase history on the area mean of the analysable headline-season cells
    R, yrs = season_stack(c, grid, hm)
    n0 = nino_series(indices, hm, 0)
    am = R[:, ok_head].mean(1)
    df = pd.DataFrame({"rain": am}, index=yrs).join(n0.rename("nino")).dropna()
    df["z"] = (df.rain - df.rain.mean()) / df.rain.std()
    df["pct"] = df.rain.rank(pct=True)
    df["phase"] = np.where(df.nino >= ENSO_THRESH, "El Niño", np.where(df.nino <= -ENSO_THRESH, "La Niña", "Neutral"))
    fig_phase_history(df, head, out_dir / "phase_history.png", name)
    phase_rows = []
    for ph in ("El Niño", "Neutral", "La Niña"):
        g = df[df.phase == ph]
        phase_rows.append(dict(phase=ph, n=len(g), mean_z=g.z.mean(), tercile=(g.pct <= 1 / 3).mean(),
                               quintile=(g.pct <= 0.2).mean(), wettest=(g.pct > 2 / 3).mean()))
    en = df[df.phase == "El Niño"].sort_index()
    r_area = float(np.corrcoef(df.rain, df.nino)[0, 1])
    n1 = nino_series(indices, hm, 1)
    d1 = df.join(n1.rename("n1")).dropna()
    r_area_lag1 = float(np.corrcoef(d1.rain, d1.n1)[0, 1])
    driest = df[df.pct <= 1 / 3]

    # Composite + hit-rate maps
    en_mask = np.isin(yrs, en.index.values)
    Z = (R - R.mean(0)) / R.std(0)
    comp = Z[en_mask].mean(0)
    pct = (R.argsort(0).argsort(0) + 1) / R.shape[0]
    hit = (pct[en_mask] <= 1 / 3).mean(0)
    fews = load_fews(iso3) if spec.get("food_security") == "fews" else None
    outlines = fews["gdf"] if (fews and fews["picks"]) else None
    fig_composite_maps(c, comp, hit, ok_head, head, int(en_mask.sum()), out_dir / "composite_maps.png", name, outlines)
    zone_rows, zone_dfs = [], {}
    for label, zm in zones.items():
        if not zm.any():
            zone_rows.append(dict(zone=label, n_cells=0, r=np.nan, comp=np.nan, hit=np.nan, hit_zone=np.nan))
            continue
        zdf = pd.DataFrame({"rain": R[:, zm].mean(1)}, index=yrs).join(n0.rename("nino")).dropna()
        zdf["z"] = (zdf.rain - zdf.rain.mean()) / zdf.rain.std()
        zdf["pct"] = zdf.rain.rank(pct=True)
        zdf["phase"] = np.where(zdf.nino >= ENSO_THRESH, "El Niño", np.where(zdf.nino <= -ENSO_THRESH, "La Niña", "Neutral"))
        zen = zdf[zdf.phase == "El Niño"]
        zone_dfs[label] = zdf
        zone_rows.append(dict(zone=label, n_cells=int(zm.sum()), r=float(np.corrcoef(zdf.rain, zdf.nino)[0, 1]),
                              comp=float(np.nanmedian(comp[zm])), hit=float(np.nanmedian(hit[zm])),
                              hit_zone=float((zen.pct <= 1 / 3).mean()) if len(zen) else np.nan))
    if zone_dfs and spec.get("zone_history"):
        fig_zone_history(zone_dfs, head, out_dir / "zone_history.png", name)

    # SEAS5 forecast skill for the headline trimester, per zone + whole country
    skill = None
    if spec.get("skill", True):
        # default: issued one month before the headline season starts
        im = int(spec.get("skill_issued_month", ((hm[0] - 2) % 12) + 1))
        skill = seas5_skill_issued(c, dict(zones) | {"Whole country": ok_head}, im)
        if skill:
            fig_skill_issued(c, grid, zones, skill, out_dir / "skill_issued.png", name)

    # Food security context (FEWS NET), rasterised to the page grid for zone shares
    fews_out = None
    if fews and fews["picks"]:
        if True:
            fig_fews_maps(c, fews, hit, ok_head, head, out_dir / "fews_maps.png", name)
            fews_out = dict(picks={k: {kk: vv for kk, vv in v.items() if kk != "phase"} for k, v in fews["picks"].items()},
                            generated=fews["generated"],
                            n_units={k: len(v["phase"]) for k, v in fews["picks"].items()},
                            p3_units={k: int(sum(1 for p in v["phase"].values() if p >= 3)) for k, v in fews["picks"].items()})

    seasons = None
    if spec.get("season_table"):
        fyear = int(skill["issued_year"]) if (skill and skill.get("issued_year")) else None
        seasons = season_table(iso3, df, head, fews, fyear)

    # Admin-1 view: the team's ERA5 per-admin raster stats + CODAB polygons
    adm1_out = None
    if spec.get("adm1", True):
        stats = adm1_enso_stats(iso3, hm, indices, grid.years)
        if stats is not None and len(stats):
            try:
                adm = load_adm1(iso3)
            except Exception as e:  # noqa: BLE001
                print(f"  (admin-1 polygons not available: {e})"); adm = None
            if adm is not None:
                fig_adm1_maps(c, adm, stats, head, out_dir / "adm1_maps.png", name)
            adm1_out = dict(stats=stats, has_map=adm is not None)

    # Country-level survey cross-check (optional: needs out/ parquet)
    adm0 = None
    try:
        tot = pd.read_parquet(cfg["parquet_dir"] / "corr_total_l3.parquet")
        par = pd.read_parquet(cfg["parquet_dir"] / "corr_partial_l3.parquet")
        t = tot[(tot.iso3 == iso3) & (tot["index"] == "nino34")].set_index("trimester")
        p_ = par[(par.iso3 == iso3) & (par["index"] == "nino34")].set_index("trimester")
        rows = []
        for tri in ts.TRIMESTERS:
            if tri not in t.index:
                continue
            _, s_tri = analysable(season_months(tri))
            rows.append(dict(trimester=tri, r=t.loc[tri, "r"], p=t.loc[tri, "p"], lag=int(t.loc[tri, "lag"]),
                             r_partial=p_.loc[tri, "r"] if tri in p_.index else np.nan,
                             share=float(np.nanmean((s_tri / annual)[c.mask]))))
        adm0 = pd.DataFrame(rows)
    except Exception as e:  # noqa: BLE001
        print(f"  ({iso3}: country-level parquet not available: {e})")

    nino_now = (NINO_LATEST if NINO_LATEST is not None else indices["nino34"]).dropna()
    nino_now = nino_now[nino_now > -90]
    now_val = float(nino_now.iloc[-1]); now_date = nino_now.index[-1]
    now_phase = "El Niño" if now_val >= ENSO_THRESH else "La Niña" if now_val <= -ENSO_THRESH else "neutral"
    # 3-month running mean, NOAA-style
    now_3m = float(nino_now.iloc[-3:].mean())

    result = dict(country=c, summaries=summaries, df=df, phase_rows=phase_rows, en=en, driest=driest,
                nino_now=dict(value=now_val, date=now_date, phase=now_phase, mean3=now_3m),
                r_area=r_area, r_area_lag1=r_area_lag1, comp_med=float(np.nanmedian(comp[ok_head])),
                comp_frac=float((comp[ok_head] < -0.5).mean()), hit_med=float(np.nanmedian(hit[ok_head])),
                zone_rows=zone_rows, skill=skill, fews=fews_out, adm1=adm1_out, seasons=seasons, adm0=adm0, n_cells=int(c.mask.sum()),
                n_head=int(ok_head.sum()))
    if seasons is not None:
        fr = forecast_row(result, head, fews)
        if fr:
            result["seasons"] = seasons + [fr]
        fig_seasons(result["seasons"], head, out_dir / "seasons.png", name)
    return result


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--b5:#269777;--b6:#1e795f;--b7:#18614c;--b05:#e9f5f1;--b1:#d4eae4;--n9:#1f2324;--n8:#3f4748;--n7:#5e6a6b;--n05:#f5f7f7;}
*{box-sizing:border-box}
body{margin:0;background:var(--n05);color:var(--n9);font-family:'Roboto',system-ui,-apple-system,'Segoe UI','Helvetica Neue',Arial,sans-serif;line-height:1.55;font-size:15px}
.wrap{max-width:1000px;margin:0 auto;background:#fff;min-height:100vh;box-shadow:0 0 40px rgba(31,35,36,.06);padding:14px 44px 48px}
.home-link{display:inline-block;margin:0 0 18px;padding:6px 12px;font:500 13px/1 'Roboto',system-ui,sans-serif;color:var(--b6);background:var(--b05);border:1px solid var(--b1);border-radius:4px;text-decoration:none}
.home-link:hover{background:var(--b1)}
h1{font-family:'Merriweather',Georgia,serif;font-size:30px;line-height:1.2;margin:6px 0 6px}
h2{font-family:'Merriweather',Georgia,serif;font-size:20px;margin:36px 0 10px;padding-top:8px;border-top:1px solid #e2e7e7}
h3{font-size:15px;margin:22px 0 6px}
p{margin:0 0 12px;max-width:78ch}
.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:700;color:var(--n7);margin:0}
.meta{color:var(--n7);font-size:13px;margin-bottom:22px}
.verdict{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0 8px}
.card{border:1px solid #e2e7e7;border-radius:6px;padding:14px 16px;background:#fff}
.card .lbl{font-size:11px;letter-spacing:.11em;text-transform:uppercase;font-weight:700;color:var(--n7);margin:0 0 6px}
.card .big{font-size:20px;font-weight:700;margin:0 0 4px}
.chip{display:inline-block;padding:2px 9px;border-radius:3px;font-size:12px;font-weight:600;color:#1a1a1a;margin-right:6px}
.summary{background:var(--b05);border-left:5px solid var(--b5);padding:12px 16px;border-radius:4px;margin:16px 0 6px}
.summary p{margin:0 0 8px}.summary p:last-child{margin:0}
figure{margin:16px 0 22px}figure img{width:100%;height:auto;display:block;border:1px solid #eef1f1;border-radius:4px}
figcaption{font-size:12.5px;color:var(--n7);margin-top:6px;max-width:90ch}
table{border-collapse:collapse;font-size:13.5px;margin:8px 0 16px}
th,td{padding:5px 10px;border-bottom:1px solid #e8ecf1;text-align:left;vertical-align:top}
th{background:#eef2f7;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.hl{background:#fbf3ea}
.refs li{margin:0 0 6px;font-size:14px}
table.skill{font-size:12.5px;width:100%}table.seasons td,table.seasons th{vertical-align:top}table.skill th,table.skill td{padding:5px 6px}table.skill .chip{font-size:11px;padding:1px 6px}table.skill .small{font-size:11px}
a{color:var(--b6)}
.small{font-size:13px;color:var(--n7)}
@media(max-width:640px){.wrap{padding:12px 18px 36px}.verdict{grid-template-columns:1fr}h1{font-size:24px}}
"""

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="wrap">
<a class="home-link" href="{home}">&larr; {home_label}</a>
"""

FOOT = """
</div>
</body>
</html>
"""


def chip(grade: str) -> str:
    return f'<span class="chip" style="background:{GRADES.get(grade, GRADES["none"])};">{html.escape(grade)}</span>'


def fmt_r(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}".replace("-", "−")


def pct(v) -> str:
    return f"{100 * v:.0f}%"


DEFAULT_ORDER = ["verdict", "summary", "before", "era5", "drought", "seasons", "adm1", "skill", "fews", "after", "refs"]


def render_country(spec: dict, a: dict, end_year: int) -> str:
    """Assemble the page from named blocks. `section_order` in the TOML reorders or drops blocks;
    `titles` overrides a block's h2 (e.g. to phrase it as a question)."""
    name, head = spec["name"], spec["headline_season"]
    cat, ours = spec["catalogue"], spec["assessment"]
    en, df = a["en"], a["df"]
    titles = spec.get("titles", {})
    T = lambda key, default: html.escape(titles.get(key, default))

    out = [HEAD.format(title=f"{name} — ENSO deep dive", desc=html.escape(ours["one_line"]), css=CSS,
                       home="../", home_label="ENSO country deep dives")]
    out.append(f'<p class="eyebrow">ENSO country deep dive</p><h1>{html.escape(name)}</h1>')
    out.append(f'<p class="meta">{html.escape(spec.get("subtitle", ""))} &nbsp;·&nbsp; ERA5 0.25° 1981–{end_year} '
               f'&nbsp;·&nbsp; Niño3.4 (NOAA PSL) &nbsp;·&nbsp; {a["n_cells"]} grid cells, {a["n_head"]} with a {head} season</p>')

    def block_verdict():
        o = ['<div class="verdict">']
        o.append(f'<div class="card"><p class="lbl">Survey catalogue says</p><p class="big">El Niño → {html.escape(cat["direction"])}, {html.escape(cat["season"])}</p>'
                 f'<p>{chip(cat["evidence"])} <span class="small">source: {cat["source_html"]}</span></p></div>')
        o.append(f'<div class="card"><p class="lbl">This review assesses</p><p class="big">El Niño → {html.escape(ours["direction"])}, {html.escape(ours["season"])}</p>'
                 f'<p>{chip(ours["evidence"])} <span class="small">{html.escape(ours["evidence_note"])}</span></p></div>')
        o.append('</div>')
        return o

    def block_summary():
        return [f'<div class="summary">{spec["summary_html"]}</div>']

    def block_before():
        return [f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}' for sec in spec.get("sections_before", [])]

    def block_after():
        return [f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}' for sec in spec.get("sections_after", [])]

    def block_era5():
        o = [f'<h2>{T("era5", "What ERA5 shows")}</h2>']
        nn = a.get("nino_now")
        if nn:
            o.append(f'<p class="small">Latest Niño3.4 in NOAA PSL\'s current series: {nn["value"]:+.2f} °C for {nn["date"]:%B %Y} '
                     f'(three-month mean {nn["mean3"]:+.2f}), i.e. {nn["phase"]} conditions by the ±{ENSO_THRESH} threshold used on this page. '
                     f'The historical analysis below uses the survey\'s pinned Niño3.4 series (NOAA PSL, ERSST v5 basis), which runs '
                     f'about 0.2 °C cooler than the current ERSST v6 series.</p>')
        o.append(spec.get("era5_intro_html",
                 '<p>Everything in this section is computed from the same ERA5 monthly grid and NOAA Niño3.4 index the '
                 'survey uses, restricted to the cells inside the country. A cell–season is analysed only if that season '
                 'holds at least a quarter of the cell\'s annual rainfall and averages at least 0.25 mm/day (the survey\'s '
                 'rainy-season and aridity filters).</p>'))
        o.append(f'<h3>Seasonal cycle</h3>{spec.get("seasonal_cycle_html", "")}')
        o.append('<figure><img src="seasonal_cycle.png" alt="Monthly rainfall climatology"><figcaption>Area-mean monthly '
                 'rainfall over all grid cells in the country. Dark bars mark the headline season used below.</figcaption></figure>')
        if spec.get("zones"):
            o.append('<figure style="max-width:620px"><img src="zones_map.png" alt="Zone locator map"><figcaption>Where the zones '
                     'are: the 0.25° cells assigned to each zone used in the charts and tables on this page'
                     + (' (latitude bands)' if any("min_lat" in z or "max_lat" in z for z in spec["zones"]) else
                        f' (bands of {head} climatology)') + '.</figcaption></figure>')
        o.append(f'<h3>Pixel-level correlation with Niño3.4</h3>{spec.get("corr_html", "")}')
        o.append('<figure><img src="corr_maps.png" alt="Pixel-level Niño3.4 correlation maps"><figcaption>Pearson r between '
                 'seasonal rainfall and Niño3.4 for each 0.25° cell, keeping the lag (0–3 months, index leading) with the '
                 'largest |r|, exactly as the survey\'s pixel pass does. Brown = drier under El Niño, blue = wetter. Grey cells '
                 'have no analysable season in that window.</figcaption></figure>')
        o.append('<table><thead><tr><th>Season</th><th class="num">Cells analysed</th><th class="num">Share of annual rain</th>'
                 '<th class="num">Median r</th><th class="num">Range</th><th class="num">Significant (p&lt;0.05)</th>'
                 '<th class="num">|r| ≥ 0.30</th><th class="num">|r| ≥ 0.50</th></tr></thead><tbody>')
        for s_ in a["summaries"]:
            neg = s_["median_r"] < 0
            o.append(f'<tr{" class=hl" if s_["season"] == head else ""}><td>{s_["season"]}</td><td class="num">{s_["n_cells"]} / {s_["n_country"]}</td>'
                     f'<td class="num">{pct(s_["mean_share"])}</td><td class="num">{fmt_r(s_["median_r"])}</td>'
                     f'<td class="num">{fmt_r(s_["min_r"])} to {fmt_r(s_["max_r"])}</td>'
                     f'<td class="num">{pct(s_["frac_sig_neg"] if neg else s_["frac_sig_pos"])} {"negative" if neg else "positive"}</td>'
                     f'<td class="num">{pct(s_["frac_mod_neg"] if neg else s_["frac_mod_pos"])}</td>'
                     f'<td class="num">{pct(s_["frac_strong_neg"] if neg else s_["frac_strong_pos"])}</td></tr>')
        o.append('</tbody></table>')
        if a["adm0"] is not None and spec.get("show_adm0", True):
            o.append(f'<h3>Country-level view (the survey\'s ADM0 pass)</h3>{spec.get("adm0_html", "")}')
            o.append('<table><thead><tr><th>Trimester</th><th class="num">Share of annual rain</th><th class="num">Total r (best lag)</th>'
                     '<th class="num">Lag (mo)</th><th class="num">p</th><th class="num">Unique-signal r (partial)</th></tr></thead><tbody>')
            for _, r in a["adm0"].iterrows():
                rainy = r.share >= 0.25
                if not rainy and spec.get("hide_filtered_adm0", False):
                    continue
                cls = ' class="hl"' if r.trimester == head else ""
                style = "" if rainy else ' style="color:#9aa3ad;"'
                o.append(f'<tr{cls}{style}><td>{r.trimester}{"" if rainy else " <span class=small>filtered *</span>"}</td>'
                         f'<td class="num">{pct(r.share)}</td><td class="num">{fmt_r(r.r)}</td><td class="num">{r.lag}</td>'
                         f'<td class="num">{r.p:.3f}</td><td class="num">{fmt_r(r.r_partial)}</td></tr>')
            o.append('</tbody></table>')
            if not spec.get("hide_filtered_adm0", False):
                o.append('<p class="small">* Below the survey\'s rainy-season filter (trimester climatology under 25% of the annual '
                         'mean), so the survey never shows these correlations at country level; '
                         + spec.get("adm0_filtered_note", "they are listed here for completeness.") + '</p>')
        return o

    def block_drought():
        ph = {r["phase"]: r for r in a["phase_rows"]}
        o = [f'<h2>{T("drought", f"El Niño and {head} drought")}</h2>{spec.get("drought_html", "")}']
        o.append(f'<figure><img src="phase_history.png" alt="{head} rainfall history by ENSO phase"><figcaption>Standardised '
                 f'{head} rainfall, averaged over the cells with a {head} season, coloured by the ENSO phase of the same '
                 f'season (Niño3.4 ≥ +{ENSO_THRESH} El Niño, ≤ −{ENSO_THRESH} La Niña). El Niño years are labelled. '
                 f'Area-mean correlation with concurrent Niño3.4: r = {fmt_r(a["r_area"])}; with Niño3.4 one month earlier: '
                 f'r = {fmt_r(a["r_area_lag1"])}.</figcaption></figure>')
        o.append('<table><thead><tr><th>ENSO phase (concurrent)</th><th class="num">Seasons</th><th class="num">Mean anomaly (SD)</th>'
                 '<th class="num">In driest third</th><th class="num">In driest fifth</th><th class="num">In wettest third</th></tr></thead><tbody>')
        for r in a["phase_rows"]:
            o.append(f'<tr><td>{r["phase"]}</td><td class="num">{r["n"]}</td><td class="num">{fmt_r(r["mean_z"])}</td>'
                     f'<td class="num">{pct(r["tercile"])}</td><td class="num">{pct(r["quintile"])}</td><td class="num">{pct(r["wettest"])}</td></tr>')
        o.append('</tbody></table>')
        o.append(f'<h3>Every El Niño {head} season since 1981</h3>')
        o.append('<table><thead><tr><th>Year</th><th class="num">Niño3.4 (' + head + ')</th><th class="num">Rainfall anomaly (SD)</th>'
                 '<th class="num">Rank (1 = driest)</th><th>Outcome</th></tr></thead><tbody>')
        n_all = len(df)
        for yr, r in en.iterrows():
            rank = int(round(r.pct * n_all))
            outcome = ("driest fifth" if r.pct <= 0.2 else "driest third" if r.pct <= 1 / 3 else
                       "wettest third" if r.pct > 2 / 3 else "near normal")
            o.append(f'<tr><td>{yr}</td><td class="num">{r.nino:+.2f}</td><td class="num">{fmt_r(r.z)}</td>'
                     f'<td class="num">{rank} of {n_all}</td><td>{outcome}</td></tr>')
        o.append('</tbody></table>')
        dr = a["driest"]
        ph_counts = dr.phase.value_counts().to_dict()
        o.append(f'<p class="small">The {len(dr)} driest-third {head} seasons: '
                 + ", ".join(f"{y} ({p})" for y, p in zip(dr.sort_index().index, dr.sort_index().phase)) + ". "
                 f'Phase split: {", ".join(f"{k} {v}" for k, v in ph_counts.items())}.</p>')
        o.append(f'<figure><img src="composite_maps.png" alt="El Niño composite and drought hit-rate maps"><figcaption>Left: mean '
                 f'standardised {head} anomaly across the El Niño years, per cell (median over analysable cells '
                 f'{fmt_r(a["comp_med"])} SD; {pct(a["comp_frac"])} of cells below −0.5 SD). Right: the share of El Niño years '
                 f'that landed in the cell\'s own driest third (median {pct(a["hit_med"])}; chance is 33%).'
                 + (' White outlines are FEWS NET\'s reporting units.' if spec.get("food_security") == "fews" else "") + '</figcaption></figure>')
        if a["zone_rows"]:
            o.append(f'<h3>By zone</h3>{spec.get("zones_html", "")}')
            o.append('<table><thead><tr><th>Zone</th><th class="num">Cells</th><th class="num">Zone-mean r (concurrent)</th>'
                     '<th class="num">El Niño composite (median SD)</th><th class="num">El Niño years in the zone\'s driest third</th>'
                     '<th class="num">Per-cell median</th></tr></thead><tbody>')
            for z in a["zone_rows"]:
                o.append(f'<tr><td>{html.escape(z["zone"])}</td><td class="num">{z["n_cells"]}</td><td class="num">{fmt_r(z["r"])}</td>'
                         f'<td class="num">{fmt_r(z["comp"])}</td><td class="num">{pct(z["hit_zone"]) if not np.isnan(z["hit_zone"]) else "—"}</td>'
                         f'<td class="num">{pct(z["hit"]) if not np.isnan(z["hit"]) else "—"}</td></tr>')
            o.append('</tbody></table>')
            if spec.get("zone_history"):
                o.append(f'<figure><img src="zone_history.png" alt="{head} rainfall history by zone and ENSO phase"><figcaption>'
                         f'Standardised {head} rainfall for each zone\'s area mean, coloured by concurrent ENSO phase; dashed line is '
                         f'the zone\'s own driest-third threshold. {spec.get("zone_history_caption", "")}</figcaption></figure>')
        return o

    blocks = {
        "verdict": block_verdict, "summary": block_summary, "before": block_before, "era5": block_era5,
        "drought": block_drought,
        "adm1": lambda: [render_adm1(spec, a, head, T("adm1", "By province: the same numbers from the team\'s ERA5 raster stats"))] if a.get("adm1") else [],
        "skill": lambda: [render_skill(spec, a, head, titles.get("skill"))] if a.get("skill") else [],
        "fews": lambda: [render_fews(spec, a, head, T("fews", "Food security context: FEWS NET"))] if a.get("fews") else [],
        "seasons": lambda: [render_seasons(spec, a, head, titles.get("seasons"))] if a.get("seasons") else [],
        "after": block_after,
        "refs": lambda: (['<h2>References</h2><ul class="refs">'] + [f'<li>{ref["html"]}</li>' for ref in spec["references"]] + ['</ul>']) if spec.get("references") else [],
    }
    for key in spec.get("section_order", DEFAULT_ORDER):
        out.extend(blocks[key]())
    out.append(f'<p class="small">Generated by <code>enso_deep_dive.py</code> from <code>deep_dives/{spec["slug"]}.toml</code>. '
               f'Method and data as in the <a href="../../survey/">global survey</a>.</p>')
    out.append(FOOT)
    return "\n".join(out)


def skill_chip(cat: str) -> str:
    if cat not in SKILL_CATS:
        return "—"
    fg = "#ffffff" if cat == "high" else "#1a1a1a"
    return f'<span class="chip" style="background:{SKILL_CATS[cat]};color:{fg};margin:0">{cat}</span>'


def forecast_summary(sk: dict) -> str:
    """One auto-written paragraph on what the current issuance forecasts, zone by zone, with the
    app's alert rule applied (|RP| ≥ 3 years at moderate-or-better skill)."""
    im, yr = sk["issued_month"], sk.get("issued_year")
    mon = MONTH_NAMES[im - 1]
    parts, alerts, ceiling = [], [], False
    for r in sk["rows"]:
        hits = []
        for t in sk["trimesters"]:
            code = t["code"]
            rp = r["rp"].get(code, np.nan)
            if np.isnan(rp) or abs(rp) < 3 or t["lead"] < 0:
                continue
            skilled = r["r"].get(code, np.nan) >= SKILL_THRESH["r_mod"]
            if abs(rp) >= 45:
                ceiling = True
            hits.append(f'{code} {"dry" if rp > 0 else "wet"} {abs(rp):.0f} yr ({skill_cat(r["r"].get(code, np.nan))} skill)')
            if skilled:
                alerts.append(f'{r["zone"].split(" (")[0]} {code}')
        if hits:
            parts.append(f'<strong>{html.escape(r["zone"].split(" (")[0])}:</strong> ' + ", ".join(hits))
    if not parts:
        body = (f'The {mon} {yr or ""} issuance has no window at or beyond the app\'s 3-year return-period threshold in any zone.')
    else:
        national = len(sk["rows"]) == 1
        body = (f'<strong>What the {mon} {yr or ""} issuance forecasts.</strong> Windows at or beyond the app\'s 3-year '
                f'return-period threshold{"" if national else ", by zone"} (in-season windows excluded): '
                + "; ".join(parts if not national else [pp.split("</strong> ", 1)[1] for pp in parts]) + ". ")
        body += ("Under the app\'s rule — an alert needs the return period <em>and</em> at least moderate skill — "
                 + (f'this issuance would raise an alert for {", ".join(a.replace("Whole country ", "") for a in alerts)}.' if alerts else
                    "none of these would raise an alert, because the skill behind them is low."))
    if ceiling:
        body += (" A return period at the ceiling (about 46 years) means the forecast is the most extreme of the 45-year "
                 "hindcast for that window, not a calibrated 1-in-46 probability; read it as “beyond the record”.")
    return f'<p>{body}</p>'


def render_skill(spec: dict, a: dict, head: str, title: str | None = None) -> str:
    sk = a["skill"]
    im = sk["issued_month"]
    mon = MONTH_NAMES[im - 1]
    yr = sk.get("issued_year")
    national = len(sk["rows"]) == 1
    out = [f'<h2>{html.escape(title) if title else f"Can SEAS5 forecast it? Skill of the {mon} issuance" + ("" if national else ", by zone")}</h2>']
    out.append("" if not spec.get("skill_intro", True) else f'<p>A teleconnection is only useful for anticipatory action if the seasonal forecast can carry it. '
               f'The figure reads the seas5-skill app\'s per-pixel skill cube — the temporal Pearson r between the '
               f'detrended ECMWF SEAS5 trimester forecast and detrended ERA5, per 0.4° pixel — for forecasts issued on '
               f'1 {mon}, sampled at this page\'s cells and summarised as the median pixel r per zone. Bins are the app\'s '
               f'(<em>low</em> &lt; {SKILL_THRESH["r_mod"]:.2f} ≤ <em>moderate</em> &lt; {SKILL_THRESH["r_high"]:.2f} ≤ <em>high</em>). '
               f'Each point is one three-month window the issuance covers, drawn on its middle month under the rainy-season '
               f'climatology, so the reader sees which part of the season each forecast window reaches and how much skill it has there.</p>')
    out.append(spec.get("skill_html", ""))
    out.append(forecast_summary(sk))
    out.append(f'<figure><img src="skill_issued.png" alt="SEAS5 skill of the {mon} issuance by zone"><figcaption>Top: '
               f'monthly rainfall climatology, whole country (bars) and zones (lines), from two months before the issuance '
               f'to the end of the seven-month SEAS5 horizon. Bottom: median pixel skill of the {mon} issuance for each '
               f'three-month window, plotted on the window\'s middle month, '
               + ("for the whole country" if national else "one line per zone and a dashed line for the whole country")
               + f', over the app\'s low / moderate / high bands. Hollow markers are windows holding under 15% '
               f'of {"the" if national else "that zone"}\'s annual rain (the app\'s off-season mask); windows left of the dashed vertical had '
               f'already started at issuance, so part of them is observed rather than forecast. Third panel: the return period of '
               f'the {mon} {yr if yr else ""} forecast anomaly in each window (Weibull rank of the forecast among its own hindcasts, '
               f'the app\'s forecast_rp / flood_rp), median pixel{"" if national else " per zone"}; dry seasons plot above the axis and wet below, '
               f'with the app\'s severe (3-year) and very severe (10-year) alert bands. Filled markers mean {"the" if national else "the zone" + chr(39) + "s"} skill '
               f'there is at least moderate, the app\'s condition for raising an alert.</figcaption></figure>')
    # compact table of the same numbers, with the share of cells at moderate-or-better
    tris = sk["trimesters"]
    out.append('<div style="overflow-x:auto"><table class="skill"><thead><tr><th>Zone</th><th class="num">Cells</th>'
               + "".join(f'<th class="num">{t["code"]}<br><span class="small">{"in-season" if t["lead"] < 0 else f"{t["lead"]}-mo lead"}</span></th>' for t in tris)
               + '</tr></thead><tbody>')
    for r in sk["rows"]:
        cells = []
        for t in tris:
            v = r["r"].get(t["code"], np.nan)
            if np.isnan(v):
                cells.append('<td class="num">—</td>'); continue
            rp = r["rp"].get(t["code"], np.nan)
            rp_txt = "—" if np.isnan(rp) else (f"dry {rp:.1f} yr" if rp > 0 else f"wet {-rp:.1f} yr")
            cells.append(f'<td class="num" style="white-space:nowrap">{fmt_r(v)} {skill_chip(skill_cat(v))}'
                         f'<br><span class="small">{pct(r["frac_mod"][t["code"]])} ≥ mod. · {rp_txt}</span></td>')
        out.append(f'<tr><td>{html.escape(r["zone"])}</td><td class="num">{r["n_cells"]}</td>{"".join(cells)}</tr>')
    out.append('</tbody></table></div>')
    yr = sk.get("issued_year")
    out.append(f'<p class="small">Each cell: median pixel r, its bin, the share of the zone\'s cells at moderate-or-better skill, and the '
               f'median return period of the {mon} {yr if yr else ""} forecast anomaly (dry = forecast below its hindcast median). '
               'Skill source: <code>skill_stats_grid_detrended.nc</code> (seas5-skill, DEV blob), the same cube behind '
               'the app\'s pixel skill map. Median of pixel correlations, not the correlation of the area mean, so it is a '
               'conservative summary for a coherent area.</p>')
    return "\n".join(out)


def render_adm1(spec: dict, a: dict, head: str, title: str | None = None) -> str:
    ad = a["adm1"]; st = ad["stats"]
    out = [f'<h2>{title or "By province: the same numbers from the team&#39;s ERA5 raster stats"}</h2>']
    out.append("" if not spec.get("adm1_intro", True) else '<p>The zones above are analysis bands; operational units are provinces. This section repeats the '
               'headline-season analysis on the team\'s standard per-admin ERA5 raster stats (monthly means per admin-1 '
               'unit from <code>public.era5</code>, the same table the SEAS5 skill app and the drought triggers use), so '
               'nothing here is recomputed from pixels: each province\'s season series is the stored mean, correlated with '
               'concurrent Niño3.4 and ranked against its own history.</p>')
    out.append(spec.get("adm1_html", ""))
    if ad["has_map"]:
        out.append(f'<figure><img src="adm1_maps.png" alt="Admin-1 maps"><figcaption>Left: Pearson r between the province\'s '
                   f'{head} mean rainfall and concurrent Niño3.4. Right: share of El Niño {head} seasons in the province\'s own '
                   f'driest third. '
                   + ' Boundaries: CODAB admin-1 via FieldMaps.</figcaption></figure>')
    out.append('<table><thead><tr><th>Province</th><th class="num">ERA5 pixels</th><th class="num">r (concurrent)</th>'
               '<th class="num">El Niño mean (SD)</th><th class="num">El Niño seasons in driest third</th><th class="num">La Niña in driest third</th>'
               + '<th>El Niño driest-third seasons</th></tr></thead><tbody>')
    for _, r in st.iterrows():
        out.append(f'<tr><td>{html.escape(str(r["name"]))}</td><td class="num">{r.n_px}</td><td class="num">{fmt_r(r.r)}</td>'
                   f'<td class="num">{fmt_r(r.en_z)}</td><td class="num">{pct(r.en_hit)} of {r.n_en}</td><td class="num">{pct(r.ln_hit)}</td>'
                   + f'<td class="small">{html.escape(r.seasons)}</td></tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


def render_fews(spec: dict, a: dict, head: str, title: str | None = None) -> str:
    fw = a["fews"]
    order = [sc for sc in ("CS", "ML1", "ML2") if sc in fw["picks"]]
    lbl = {"CS": "Current situation", "ML1": "Near-term projection", "ML2": "Medium-term projection"}
    out = [f'<h2>{title or "Food security context: FEWS NET"}</h2>']
    out.append("" if not spec.get("fews_intro", True) else '<p>FEWS NET\'s IPC-compatible acute food insecurity classification, from the team\'s daily mirror of the '
               'FEWS NET Data Warehouse (<a href="https://ocha-dap.github.io/ds-fewsnet-mirror/">ds-fewsnet-mirror</a>), drawn '
               'on FEWS NET\'s own livelihood-zone × district units. This is the published map — the “not allowing for '
               'assistance” series; grey means FEWS NET did not classify the unit, which is not Phase 1. FEWS NET classifies '
               'areas, not populations, and publishes no population-in-phase figures, so the maps are shown at the level FEWS NET '
               'reports them and are not aggregated here. FEWS NET\'s analysis is IPC-compatible but independent of the IPC/CH '
               'consensus.</p>')
    out.append(spec.get("food_security_html", ""))
    out.append('<figure><img src="fews_maps.png" alt="FEWS NET food insecurity phases"><figcaption>'
               + "; ".join(f'{lbl[sc]}: {_window(fw["picks"][sc]["start"], fw["picks"][sc]["end"])}, '
                           f'from the {fw["picks"][sc]["round"]} round ({html.escape(fw["picks"][sc]["doc"])}), '
                           f'{fw["p3_units"][sc]} of {fw["n_units"][sc]} classified units in Phase 3+' for sc in order)
               + f'. Last panel: the El Niño driest-third hit-rate for {head} from the drought section with the FEWS NET unit outlines on top, so the two can be read together. '
               f'Mirror snapshot {html.escape(str(fw["generated"])[:10])}. Source: FEWS NET.</figcaption></figure>')
    return "\n".join(out)


def render_index(specs: list[dict], end_year: int) -> str:
    out = [HEAD.format(title="ENSO country deep dives", desc="Per-country reviews of the ENSO–rainfall evidence: literature grade, ERA5 pixel correlations, and drought odds.",
                       css=CSS + ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}"
                       ".k{display:flex;flex-direction:column;text-decoration:none;color:inherit;border:1px solid #e2e7e7;border-top:3px solid var(--b5);border-radius:6px;padding:16px 18px 12px;background:#fff}"
                       ".k:hover{border-color:var(--b5);box-shadow:0 2px 12px rgba(31,35,36,.08)}.k h2{border:0;margin:0 0 6px;font-size:18px;padding:0}.k p{font-size:14px;flex:1}"
                       ".k .foot{font-size:12px;color:var(--n7);margin-top:8px}",
                       home="../", home_label=SITE_TITLE)]
    out.append('<p class="eyebrow">Teleconnections</p><h1>ENSO country deep dives</h1>')
    out.append('<p class="meta">Where the survey\'s one-line literature grade deserves a closer look: per country, why the catalogue '
               'says what it says, what the peer-reviewed literature actually supports, what ERA5 shows at country and pixel '
               'resolution, and what that means for drought.</p>')
    out.append('<div class="grid">')
    for s in sorted(specs, key=lambda s: s["name"]):
        cat, ours = s["catalogue"], s["assessment"]
        out.append(f'<a class="k" href="{s["slug"]}/"><h2>{html.escape(s["name"])}</h2>'
                   f'<p>{html.escape(ours["one_line"])}</p>'
                   f'<div class="foot">catalogue {chip(cat["evidence"])} → reviewed {chip(ours["evidence"])}</div></a>')
    out.append('</div>')
    out.append('<h2>How a deep dive is built</h2>'
               '<p>Each page pairs a curated evidence review (why the survey catalogue graded the country as it did, and what '
               'the literature actually supports) with figures recomputed from the survey\'s own ERA5 0.25° grid and the NOAA '
               'Niño3.4 index: the seasonal cycle, pixel-level correlation maps, a season-by-season rainfall history coloured by '
               'ENSO phase, and El Niño composite and drought hit-rate maps. The drought framing is deliberate: a correlation '
               'coefficient understates an asymmetric response, so each page reports how often El Niño seasons fell in the '
               'driest third, against a 33% base rate.</p>'
               '<p class="small">Add a country by dropping a <code>deep_dives/&lt;slug&gt;.toml</code> into the repo and running '
               '<code>uv run python enso_deep_dive.py</code>; see the <a href="https://github.com/OCHA-DAP/ds-teleconnections#country-deep-dives">README</a>.</p>')
    out.append(FOOT)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="slug of a single country to (re)build")
    args = ap.parse_args()
    cfg = dict(ts.CONFIG, max_lag=3)
    specs = [tomllib.loads(p.read_text()) | {"slug": p.stem} for p in sorted(DEEP_DIR.glob("*.toml"))]
    if not specs:
        raise SystemExit("no deep_dives/*.toml found")
    grid = load_grid(cfg)
    gdf = ts.load_admin0_gdf(cfg)
    # The analysis uses the survey's pinned Niño3.4 series (cache/nino34.data) so every published
    # number stays reproducible; the *latest* NOAA series is fetched separately (weekly) and used
    # only for the "current ENSO state" line. NOAA moved nina34.anom.data to ERSST v6 in 2026,
    # which shifts anomalies by ~+0.2 °C and changes phase counts — do not mix the two.
    latest = cfg["cache_dir"] / "nino34_latest.data"
    if not latest.exists() or (pd.Timestamp.now() - pd.Timestamp(latest.stat().st_mtime, unit="s")) > pd.Timedelta(days=7):
        try:
            import requests
            txt = requests.get(ts.INDEX_SOURCES["nino34"], timeout=60).text
            if len(txt) > 1000:
                latest.write_text(txt)
        except Exception as e:  # noqa: BLE001
            print(f"  (Niño3.4 latest refresh failed: {e})")
    global NINO_LATEST
    NINO_LATEST = ts._parse_psl(latest.read_text()) if latest.exists() else None
    indices = ts.load_indices(cfg)
    for spec in specs:
        if args.only and spec["slug"] != args.only:
            continue
        print(f"{spec['iso3']}: {spec['name']}")
        a = analyse(spec, grid, gdf, indices, cfg, OUT_DIR / spec["slug"])
        (OUT_DIR / spec["slug"] / "index.html").write_text(render_country(spec, a, cfg["end_year"]), encoding="utf-8")
        for s in a["summaries"]:
            print(f"  {s['season']}: {s['n_cells']} cells, median r {s['median_r']:+.2f}, sig {pct(max(s['frac_sig_neg'], s['frac_sig_pos']))}")
        print(f"  El Niño composite median {a['comp_med']:+.2f} SD; hit-rate median {pct(a['hit_med'])}")
    (OUT_DIR / "index.html").write_text(render_index(specs, cfg["end_year"]), encoding="utf-8")
    print(f"wrote {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
