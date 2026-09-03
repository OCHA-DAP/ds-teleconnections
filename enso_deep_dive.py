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
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    cols = ["#9C6730" if (m + 1) in headline else "#D8C3AC" for m in range(12)]
    ax.bar(range(12), clim, color=cols, width=0.72)
    if zones:
        for (label, zm), col in zip(zones.items(), ["#1F5F96", "#5E9FD2", "#18614c"]):
            z = [c.sub[[grid.mpos[(y, m)] for y in grid.years if (y, m) in grid.mpos]][:, zm].mean()
                 for m in range(1, 13)]
            ax.plot(range(12), z, color=col, lw=2, marker="o", ms=4, label=label)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("mm / day", fontsize=9, color=C_MUTED)
    ax.set_title(f"{name}: ERA5 monthly rainfall climatology, 1981–{grid.years[-1]} "
                 f"(dark bars = headline season)", fontsize=10, color=C_TEXT, loc="left")
    _style_ax(ax)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


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


def fig_corr_maps(c: Country, panels: list[dict], out: Path, name: str) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 1.2, 4.6), dpi=150)
    axes = np.atleast_1d(axes)
    ext = _extent(c)
    for ax, p in zip(axes, panels):
        r = np.where(p["analysable"], p["r"], np.nan)
        m = _pcolor(ax, c, np.where(c.mask, r, np.nan), DIVERGING, -0.7, 0.7)
        # cells inside the country but not analysable for this season: hatch
        na = c.mask & ~p["analysable"]
        _pcolor(ax, c, np.where(na, 0.0, np.nan), mcolors.ListedColormap(["#ececec"]), -1, 1)
        _draw_country(ax, c, ext)
        ax.set_title(p["title"], fontsize=10, color=C_TEXT, loc="left")
        if p.get("note"):
            ax.text(0.01, 0.01, p["note"], transform=ax.transAxes, fontsize=7.5, color=C_MUTED, va="bottom")
    cb = fig.colorbar(m, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Pearson r (Niño3.4 vs rainfall)", fontsize=9, color=C_MUTED)
    cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
    cb.ax.text(0.5, 1.03, "wetter under\nEl Niño", transform=cb.ax.transAxes, fontsize=7.5, color=C_MUTED, va="bottom", ha="center")
    cb.ax.text(0.5, -0.03, "drier under\nEl Niño", transform=cb.ax.transAxes, fontsize=7.5, color=C_MUTED, va="top", ha="center")
    fig.suptitle(f"{name}: pixel-level Niño3.4 correlation (ERA5 0.25°, 1981–2025; grey = season too small to analyse)",
                 fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.86)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_composite_maps(c: Country, comp: np.ndarray, hit: np.ndarray, analysable: np.ndarray,
                       season: str, n_en: int, out: Path, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=150)
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
    axes[1].set_title(f"Share of El Niño years in the cell's\ndriest third of {season} seasons (chance = 33%)", fontsize=9.5, loc="left", color=C_TEXT)
    cb = fig.colorbar(m2, ax=axes[1], shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=8, colors=C_MUTED)
    cb.set_label("% of El Niño years", fontsize=9, color=C_MUTED)
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(f"{name}: what El Niño (concurrent Niño3.4 ≥ +{ENSO_THRESH}) did to the {season} season, per cell",
                 fontsize=10, color=C_TEXT, x=0.01, ha="left", y=0.9)
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
    zones = {}
    for z in spec.get("zones", []):
        zm = ok_head & (clim_head >= z["min_mm_day"]) & (clim_head < z.get("max_mm_day", 1e9))
        zones[z["label"]] = zm
    fig_seasonal_cycle(c, grid, hm, out_dir / "seasonal_cycle.png", name, zones or None)

    # Correlation maps: headline + any extra seasons, best lag 0..3 (as in the survey)
    panels, summaries = [], []
    for code in [head] + [s for s in spec.get("map_seasons", []) if s != head]:
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
        panels.append(dict(r=r, analysable=ok, title=f"{code} rainfall vs Niño3.4 (best lag 0–{cfg.get('max_lag', 3)} mo)",
                           note=f"{summ['n_cells']} of {summ['n_country']} cells have a {code} season"))
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
    fig_composite_maps(c, comp, hit, ok_head, head, int(en_mask.sum()), out_dir / "composite_maps.png", name)
    zone_rows = []
    for label, zm in zones.items():
        zone_rows.append(dict(zone=label, n_cells=int(zm.sum()), comp=float(np.nanmedian(comp[zm])) if zm.any() else np.nan,
                              hit=float(np.nanmedian(hit[zm])) if zm.any() else np.nan))

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

    return dict(country=c, summaries=summaries, df=df, phase_rows=phase_rows, en=en, driest=driest,
                r_area=r_area, r_area_lag1=r_area_lag1, comp_med=float(np.nanmedian(comp[ok_head])),
                comp_frac=float((comp[ok_head] < -0.5).mean()), hit_med=float(np.nanmedian(hit[ok_head])),
                zone_rows=zone_rows, adm0=adm0, n_cells=int(c.mask.sum()), n_head=int(ok_head.sum()))


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


def render_country(spec: dict, a: dict, end_year: int) -> str:
    name, head = spec["name"], spec["headline_season"]
    cat, ours = spec["catalogue"], spec["assessment"]
    s0 = a["summaries"][0]
    en, df = a["en"], a["df"]
    ph = {r["phase"]: r for r in a["phase_rows"]}

    out = [HEAD.format(title=f"{name} — ENSO deep dive", desc=html.escape(ours["one_line"]), css=CSS,
                       home="../", home_label="ENSO country deep dives")]
    out.append(f'<p class="eyebrow">ENSO country deep dive</p><h1>{html.escape(name)}</h1>')
    out.append(f'<p class="meta">{html.escape(spec.get("subtitle", ""))} &nbsp;·&nbsp; ERA5 0.25° 1981–{end_year} '
               f'&nbsp;·&nbsp; Niño3.4 (NOAA PSL) &nbsp;·&nbsp; {a["n_cells"]} grid cells, {a["n_head"]} with a {head} season</p>')

    # Verdict cards
    out.append('<div class="verdict">')
    out.append(f'<div class="card"><p class="lbl">Survey catalogue says</p><p class="big">El Niño → {html.escape(cat["direction"])}, {html.escape(cat["season"])}</p>'
               f'<p>{chip(cat["evidence"])} <span class="small">source: {cat["source_html"]}</span></p></div>')
    out.append(f'<div class="card"><p class="lbl">This review assesses</p><p class="big">El Niño → {html.escape(ours["direction"])}, {html.escape(ours["season"])}</p>'
               f'<p>{chip(ours["evidence"])} <span class="small">{html.escape(ours["evidence_note"])}</span></p></div>')
    out.append('</div>')
    out.append(f'<div class="summary">{spec["summary_html"]}</div>')

    # Narrative sections before the data
    for sec in spec.get("sections_before", []):
        out.append(f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}')

    # ---- ERA5 evidence ----
    out.append('<h2>What ERA5 shows</h2>')
    out.append('<p>Everything in this section is computed from the same ERA5 monthly grid and NOAA Niño3.4 index the '
               'survey uses, restricted to the cells inside the country. A cell–season is analysed only if that season '
               'holds at least a quarter of the cell\'s annual rainfall and averages at least 0.25 mm/day (the survey\'s '
               'rainy-season and aridity filters).</p>')
    out.append(f'<h3>Seasonal cycle</h3>{spec.get("seasonal_cycle_html", "")}')
    out.append('<figure><img src="seasonal_cycle.png" alt="Monthly rainfall climatology"><figcaption>Area-mean monthly '
               'rainfall over all grid cells in the country. Dark bars mark the headline season used below.</figcaption></figure>')

    out.append(f'<h3>Pixel-level correlation with Niño3.4</h3>{spec.get("corr_html", "")}')
    out.append('<figure><img src="corr_maps.png" alt="Pixel-level Niño3.4 correlation maps"><figcaption>Pearson r between '
               'seasonal rainfall and Niño3.4 for each 0.25° cell, keeping the lag (0–3 months, index leading) with the '
               'largest |r|, exactly as the survey\'s pixel pass does. Brown = drier under El Niño, blue = wetter. Grey cells '
               'have no analysable season in that window.</figcaption></figure>')
    out.append('<table><thead><tr><th>Season</th><th class="num">Cells analysed</th><th class="num">Share of annual rain</th>'
               '<th class="num">Median r</th><th class="num">Range</th><th class="num">Significant (p&lt;0.05)</th>'
               '<th class="num">|r| ≥ 0.30</th><th class="num">|r| ≥ 0.50</th></tr></thead><tbody>')
    for s in a["summaries"]:
        neg = s["median_r"] < 0
        out.append(f'<tr{" class=hl" if s["season"] == head else ""}><td>{s["season"]}</td><td class="num">{s["n_cells"]} / {s["n_country"]}</td>'
                   f'<td class="num">{pct(s["mean_share"])}</td><td class="num">{fmt_r(s["median_r"])}</td>'
                   f'<td class="num">{fmt_r(s["min_r"])} to {fmt_r(s["max_r"])}</td>'
                   f'<td class="num">{pct(s["frac_sig_neg"] if neg else s["frac_sig_pos"])} {"negative" if neg else "positive"}</td>'
                   f'<td class="num">{pct(s["frac_mod_neg"] if neg else s["frac_mod_pos"])}</td>'
                   f'<td class="num">{pct(s["frac_strong_neg"] if neg else s["frac_strong_pos"])}</td></tr>')
    out.append('</tbody></table>')

    if a["adm0"] is not None:
        out.append(f'<h3>Country-level view (the survey\'s ADM0 pass)</h3>{spec.get("adm0_html", "")}')
        out.append('<table><thead><tr><th>Trimester</th><th class="num">Share of annual rain</th><th class="num">Total r (best lag)</th>'
                   '<th class="num">Lag (mo)</th><th class="num">p</th><th class="num">Unique-signal r (partial)</th></tr></thead><tbody>')
        for _, r in a["adm0"].iterrows():
            rainy = r.share >= 0.25
            cls = ' class="hl"' if r.trimester == head else ""
            style = "" if rainy else ' style="color:#9aa3ad;"'
            out.append(f'<tr{cls}{style}><td>{r.trimester}{"" if rainy else " <span class=small>filtered *</span>"}</td>'
                       f'<td class="num">{pct(r.share)}</td><td class="num">{fmt_r(r.r)}</td><td class="num">{r.lag}</td>'
                       f'<td class="num">{r.p:.3f}</td><td class="num">{fmt_r(r.r_partial)}</td></tr>')
        out.append('</tbody></table>')
        out.append('<p class="small">* Below the survey\'s rainy-season filter (trimester climatology under 25% of the annual '
                   'mean), so the survey never shows these correlations at country level; they are listed here because the '
                   'coastal winter signal is part of the story.</p>')

    # ---- Drought ----
    out.append(f'<h2>El Niño and {head} drought</h2>{spec.get("drought_html", "")}')
    out.append(f'<figure><img src="phase_history.png" alt="{head} rainfall history by ENSO phase"><figcaption>Standardised '
               f'{head} rainfall, averaged over the cells with a {head} season, coloured by the ENSO phase of the same '
               f'season (Niño3.4 ≥ +{ENSO_THRESH} El Niño, ≤ −{ENSO_THRESH} La Niña). El Niño years are labelled. '
               f'Area-mean correlation with concurrent Niño3.4: r = {fmt_r(a["r_area"])}; with Niño3.4 one month earlier: '
               f'r = {fmt_r(a["r_area_lag1"])}.</figcaption></figure>')
    out.append('<table><thead><tr><th>ENSO phase (concurrent)</th><th class="num">Seasons</th><th class="num">Mean anomaly (SD)</th>'
               '<th class="num">In driest third</th><th class="num">In driest fifth</th><th class="num">In wettest third</th></tr></thead><tbody>')
    for r in a["phase_rows"]:
        out.append(f'<tr><td>{r["phase"]}</td><td class="num">{r["n"]}</td><td class="num">{fmt_r(r["mean_z"])}</td>'
                   f'<td class="num">{pct(r["tercile"])}</td><td class="num">{pct(r["quintile"])}</td><td class="num">{pct(r["wettest"])}</td></tr>')
    out.append('</tbody></table>')
    out.append(f'<h3>Every El Niño {head} season since 1981</h3>')
    out.append('<table><thead><tr><th>Year</th><th class="num">Niño3.4 (' + head + ')</th><th class="num">Rainfall anomaly (SD)</th>'
               '<th class="num">Rank (1 = driest)</th><th>Outcome</th></tr></thead><tbody>')
    n_all = len(df)
    for yr, r in en.iterrows():
        rank = int(round(r.pct * n_all))
        outcome = ("driest fifth" if r.pct <= 0.2 else "driest third" if r.pct <= 1 / 3 else
                   "wettest third" if r.pct > 2 / 3 else "near normal")
        out.append(f'<tr><td>{yr}</td><td class="num">{r.nino:+.2f}</td><td class="num">{fmt_r(r.z)}</td>'
                   f'<td class="num">{rank} of {n_all}</td><td>{outcome}</td></tr>')
    out.append('</tbody></table>')
    dr = a["driest"]
    ph_counts = dr.phase.value_counts().to_dict()
    out.append(f'<p class="small">The {len(dr)} driest-third {head} seasons: '
               + ", ".join(f"{y} ({p})" for y, p in zip(dr.sort_index().index, dr.sort_index().phase)) + ". "
               f'Phase split: {", ".join(f"{k} {v}" for k, v in ph_counts.items())}.</p>')
    out.append(f'<figure><img src="composite_maps.png" alt="El Niño composite and drought hit-rate maps"><figcaption>Left: mean '
               f'standardised {head} anomaly across the El Niño years, per cell (median over analysable cells '
               f'{fmt_r(a["comp_med"])} SD; {pct(a["comp_frac"])} of cells below −0.5 SD). Right: the share of El Niño years '
               f'that landed in the cell\'s own driest third (median {pct(a["hit_med"])}; chance is 33%).</figcaption></figure>')
    if a["zone_rows"]:
        out.append('<table><thead><tr><th>Zone (by ' + head + ' climatology)</th><th class="num">Cells</th><th class="num">El Niño composite (median SD)</th>'
                   '<th class="num">El Niño years in driest third (median)</th></tr></thead><tbody>')
        for z in a["zone_rows"]:
            out.append(f'<tr><td>{html.escape(z["zone"])}</td><td class="num">{z["n_cells"]}</td><td class="num">{fmt_r(z["comp"])}</td><td class="num">{pct(z["hit"])}</td></tr>')
        out.append('</tbody></table>')

    for sec in spec.get("sections_after", []):
        out.append(f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}')

    if spec.get("references"):
        out.append('<h2>References</h2><ul class="refs">')
        for ref in spec["references"]:
            out.append(f'<li>{ref["html"]}</li>')
        out.append('</ul>')
    out.append(f'<p class="small">Generated by <code>enso_deep_dive.py</code> from <code>deep_dives/{spec["slug"]}.toml</code>. '
               f'Method and data as in the <a href="../../survey/">global survey</a>.</p>')
    out.append(FOOT)
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
