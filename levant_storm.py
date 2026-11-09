"""Gaza, this winter so far: one storm read three ways — the rain, how far ahead ECMWF's forecasts saw
it, and what the two flood satellites the team uses (GFM and FloodScan) saw — set against the page's
own event catalogue before and since October 2023.

Called from levant_deep_dive when the area's TOML has a [storm_now] table (Gaza only). Inputs, all
under cache/gaza/ (the page never downloads GRIB or scans STAC itself):

* IMERG late v7 over Gaza: the page's own daily series and storm list (``a["imerg_last"]``, ``a["st"]["ei"]``).
* ECMWF IFS open data, total precipitation, HRES and the 50 perturbed ENS members, Gaza-weighted:
  ``ifs_tp_<date>.parquet`` (every run in the fortnight before the storm) and ``ifs_tp_past_storms.parquet``
  (runs 1, 3 and 5 days before earlier storms), written by ``tools/ifs_open_data_tp.py``.
* Copernicus GFM (Sentinel-1, 20 m) per-scene counts inside the Gaza polygon: ``gfm_passes.jsonl``, and the
  layers of the scene after the storm for the map, ``gfm_map_<item>.npz``; both from ``tools/gaza_gfm_scan.py``.
* FloodScan (AER) SFED from the team's prod raster blob. Its Africa grid (30°W–60°E, 50°S–40°N) covers Gaza,
  with the daily record from 12 January 1998; ``floodscan_box`` keeps a 12 × 12-pixel box (1/12°) and tops it
  up on each run, like IMERG.
"""
from __future__ import annotations

import html
import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import enso_deep_dive as edd
import levant_deep_dive as L

FS_BOX = (33.9167, 30.9167, 34.9167, 31.9167)   # w, s, e, n: 12 × 12 FloodScan pixels around Gaza
GFM_PX_KM2 = 20 * 20 / 1e6                       # one GFM pixel (Equi7 20 m)
SINCE = pd.Timestamp("2023-10-07")
C_ENS, C_HRES, C_OBS = "#1F5F96", "#1f2324", "#C0782F"   # page palette: SRC_COL GPCC, ink, gauge
C_TEXT, C_MUTED = edd.C_TEXT, edd.C_MUTED


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def floodscan_box(path: Path | None = None) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """FloodScan v5 daily SFED and MFED (flooded fraction of each 1/12° pixel) over FS_BOX, from the prod
    raster blob (floodscan/daily/v5/processed/aer_area_300s_v<date>_v05r01.tif, the Africa grid). Cached as
    a cube and topped up with new days; bands are picked by name (their order is not stable in the archive)."""
    import ocha_stratus as stratus
    import rasterio
    from rasterio.windows import from_bounds
    path = path or L.COD_CACHE / "floodscan_box.npz"
    have = np.load(path) if path.exists() else None
    done = set(pd.to_datetime(have["dates"]).strftime("%Y-%m-%d")) if have is not None else set()
    try:
        cc = stratus.get_container_client("raster", stage="prod")
        names = [b.name for b in cc.list_blobs(name_starts_with="floodscan/daily/v5/processed/aer_area_300s_v") if b.name.endswith(".tif")]
        todo = [n for n in names if re.search(r"v(\d{4}-\d{2}-\d{2})", n).group(1) not in done]
    except Exception as e:  # noqa: BLE001
        print(f"  (FloodScan blob not reachable, using the cache: {e})"); todo = []
    if todo:
        print(f"  reading {len(todo)} FloodScan day(s) from blob…", flush=True)
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")

        def one(n):
            with rasterio.Env(**env), rasterio.open(cc.get_blob_client(n).url) as src:
                w = from_bounds(*FS_BOX, src.transform).round_offsets().round_lengths()
                a, d = src.read(window=w), list(src.descriptions)
                return (re.search(r"v(\d{4}-\d{2}-\d{2})", n).group(1), np.stack([a[d.index("SFED")], a[d.index("MFED")]]),
                        src.window_transform(w))
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(one, sorted(todo)))
        tr = res[0][2]; ny, nx = res[0][1].shape[-2:]
        x = tr.c + tr.a * (np.arange(nx) + 0.5); y = tr.f + tr.e * (np.arange(ny) + 0.5)
        cube = np.stack([r[1] for r in res]).astype("float32"); dates = pd.to_datetime([r[0] for r in res])
        if have is not None:
            assert np.allclose(have["x"], x) and np.allclose(have["y"], y), "FloodScan window moved"
            cube = np.concatenate([have["cube"], cube]); dates = pd.to_datetime(have["dates"]).append(dates)
        o = np.argsort(dates.values)
        np.savez_compressed(path, cube=cube[o], dates=dates.values[o].astype("datetime64[D]"), x=x, y=y, bands=np.array(["SFED", "MFED"]))
        have = np.load(path)
    return have["cube"], pd.to_datetime(have["dates"]), have["x"], have["y"]


def floodscan_gaza(poly) -> tuple[pd.Series, np.ndarray]:
    """Gaza-weighted SFED: the share of Gaza's area that FloodScan calls flooded, each pixel weighted by the share of
    the Gaza Strip inside it (most of these pixels are part sea, part Gaza, part Israel)."""
    cube, dates, x, y = floodscan_box()
    W = np.array([[L.overlap_weights(poly, {"c": (la, lo)}, 1 / 12)["c"] for lo in x] for la in y])
    sf = np.nan_to_num(cube[:, 0])
    return pd.Series((sf * W).sum(axis=(1, 2)) / W.sum(), index=dates), W


def gfm_scenes(path: Path | None = None, min_cover: float = 0.9) -> pd.DataFrame:
    """One row per Sentinel-1 acquisition that covers at least `min_cover` of the Gaza Strip. STAC items are slices of a
    scene, about 25 s apart; slices less than 3 minutes apart are one acquisition and their counts are summed (their overlap
    inside Gaza is small; coverage is capped at 1)."""
    path = path or L.COD_CACHE / "gfm_passes.jsonl"
    rows = {}
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["id"]] = r                                   # the last line per item wins (retries)
    d = pd.DataFrame(rows.values())
    d = d[d.get("err").isna() & (d.n_valid > 0)].copy() if "err" in d else d[d.n_valid > 0].copy()
    d["dt"] = pd.to_datetime(d.dt, utc=True).dt.tz_localize(None)
    d = d.sort_values("dt")
    first = d.dt.diff().gt(pd.Timedelta(minutes=3)).cumsum()       # slices of one acquisition are ~25 s apart
    d["acq"] = d.groupby(first).dt.transform("min").dt.floor("min")
    num = ["n_valid", "n_flood", "n_excl", "n_visible", "n_tuw", "n_dlr", "n_list"]
    for c in num:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    g = d.groupby("acq").agg(n_gaza=("n_gaza", "max"), **{c: (c, "sum") for c in num}, items=("id", "size")).reset_index()
    g["cover"] = (g.n_valid / g.n_gaza).clip(upper=1)
    g = g[g.cover >= min_cover].copy()
    g["flood_km2"] = g.n_flood * GFM_PX_KM2
    g["excl_share"] = g.n_excl / g.n_valid
    for c in ("tuw", "dlr", "list"):
        g[f"{c}_km2"] = g[f"n_{c}"] * GFM_PX_KM2
    return g.sort_values("acq").reset_index(drop=True)


def ifs_windows(path: Path, w0: pd.Timestamp, w1: pd.Timestamp) -> pd.DataFrame:
    """Rain over Gaza between w0 and w1 (UTC) in each run and member: tp(w1) − tp(w0), with tp(run start) = 0 for runs
    that start inside the window (those forecast only the rest of it)."""
    f = pd.read_parquet(path)
    f = f[f.member != "cf"]
    p = f.pivot_table(index=["run", "member"], columns="valid", values="gaza")
    out = []
    for (run, mem), r in p.iterrows():
        if run >= w1 or w1 not in r or pd.isna(r[w1]):
            continue
        a = 0.0 if run >= w0 else r.get(w0, np.nan)
        if pd.isna(a):
            continue
        out.append(dict(run=run, member=mem, mm=float(r[w1] - a), partial=bool(run > w0)))
    return pd.DataFrame(out)


def ens_summary(w: pd.DataFrame, thresholds=(20, 40)) -> pd.DataFrame:
    """Per run: HRES, ENS median and 10th–90th percentiles, share of members at or above each threshold."""
    rows = []
    if w.empty:
        return pd.DataFrame()
    for run, g in w.groupby("run"):
        e = g[g.member.str.startswith("pf")].mm
        h = g[g.member == "hres"].mm
        r = dict(run=run, partial=bool(g.partial.any()), n=len(e), hres=float(h.iloc[0]) if len(h) else np.nan,
                 med=float(e.median()), p10=float(e.quantile(0.1)), p90=float(e.quantile(0.9)), mx=float(e.max()))
        for t in thresholds:
            r[f"P{t}"] = float((e >= t).mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("run")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(spec: dict, a: dict) -> dict:
    L.set_area(spec)            # when levant_deep_dive runs as __main__, this import is a second copy of it
    cfg = spec["storm_now"]
    imerg = a["imerg_last"]; pr = imerg.pr
    s0, s1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    ei = a["st"]["ei"]
    hit = ei[(ei.start <= s1) & (ei.end >= s0)]
    if hit.empty:
        raise ValueError(f"[storm_now] {cfg['start']}–{cfg['end']}: no IMERG storm on those dates (IMERG ends {pr.index[-1]:%d %b %Y})")
    S = hit.iloc[0]
    day = pr.loc[S.start:S.end].idxmax(); dmax = float(pr.loc[day])
    rain = dict(start=S.start, end=S.end, total=float(S.total), max=dmax, day=day, sy=int(S.sy),
                north=float(imerg.north.loc[day]), south=float(imerg.south.loc[day]),
                rank_all=int((pr > dmax).sum()) + 1, n_all=int(pr.notna().sum()), first_year=pr.index[0].year, last=pr.index[-1],
                rank_month=int((pr[pr.index.month == day.month] > dmax).sum()) + 1)
    # first storm whose wettest day reached 20 mm, each season (days after 1 September)
    f20 = ei[ei["max"] >= 20].groupby("sy").start.min()
    doy = lambda t: (t - pd.Timestamp(t.year if t.month >= 8 else t.year - 1, 9, 1)).days
    past = f20[(f20.index >= 1998) & (f20.index < rain["sy"])]
    rain["first20"] = S["max"] >= 20 and f20.get(rain["sy"]) == S.start
    rain["first_earlier"] = [t for t in past if doy(t) < doy(S.start)]
    rain["first_n"] = rain["sy"] - 1998                      # winters 1998/99 … last winter (some never had one)
    rain["first_since"] = {int(y): f20.get(y) for y in range(SINCE.year, rain["sy"])}
    # storms of a similar size before October 2023 and what the catalogue has on them
    lo, hi = cfg.get("similar_mm", [35, 50])
    sim = ei[(ei["max"] >= lo) & (ei["max"] < hi) & (ei.start < SINCE) & (ei.start >= pr.index[0])].copy()
    pre = spec.get("prewar", [])
    sim["report"] = [next((p for p in pre if pd.Timestamp(p["start"]) <= e + pd.Timedelta(days=2) and
                           pd.Timestamp(p.get("end", p["start"])) >= s - pd.Timedelta(days=1)), None) for s, e in zip(sim.start, sim.end)]
    rain["similar"] = sim

    # ECMWF forecasts
    w0, w1 = pd.Timestamp(cfg["window"][0]), pd.Timestamp(cfg["window"][1])
    fdir = L.COD_CACHE
    fc = None
    if (fdir / cfg["ifs_cache"]).exists():
        win = ifs_windows(fdir / cfg["ifs_cache"], w0, w1)
        fc = dict(w0=w0, w1=w1, runs=ens_summary(win), obs=float(pr.loc[w0:w1 - pd.Timedelta(days=1)].sum()), members=win)
        r = fc["runs"][~fc["runs"].partial]
        lead = lambda t: (w0 - t) / pd.Timedelta(days=1)
        first = lambda m: next((t for t in r.index if m(r.loc[t]) and all(m(r.loc[u]) for u in r.index if u >= t)), None)
        fc["first_med20"] = first(lambda q: q.med >= 20)
        fc["first_p50"] = first(lambda q: q.P20 >= 0.5)
        fc["first_hres20"] = first(lambda q: q.hres >= 20)
        fc["any_p40"] = r[r.P40 > 0]
        fc["lead"] = lead
        # the 24 h of the wettest IMERG day alone
        d0, d1 = rain["day"], rain["day"] + pd.Timedelta(days=1)
        fc["day"] = ens_summary(ifs_windows(fdir / cfg["ifs_cache"], d0, d1)) if d0 >= w0 else None
    pastfc = []
    if (fdir / "ifs_tp_past_storms.parquet").exists():
        for w in cfg.get("past_windows", []):
            p0 = pd.Timestamp(w["start"]); p1 = p0 + pd.Timedelta(days=2)
            ws = ens_summary(ifs_windows(fdir / "ifs_tp_past_storms.parquet", p0, p1))
            if ws.empty:
                continue
            ws["lead_d"] = ((p0 - ws.index) / pd.Timedelta(days=1)).round().astype(int)
            e5 = a["era5"].pr.loc[p0:p1 - pd.Timedelta(days=1)] if a.get("era5") is not None else pd.Series(dtype=float)
            pastfc.append(dict(start=p0, label=w["label"], obs=float(pr.loc[p0:p1 - pd.Timedelta(days=1)].sum()),
                               era5=float(e5.sum()) if len(e5) == 2 else np.nan, runs=ws))

    # GFM
    gaza, govs = L.area_polygons(); poly = gaza.geometry.iloc[0]
    gfm = None
    if (fdir / "gfm_passes.jsonl").exists():
        sc = gfm_scenes()
        after = sc[(sc.acq >= S.start) & (sc.acq < S.end + pd.Timedelta(days=cfg.get("gfm_days_after", 4)))]
        sc["since"] = sc.acq >= SINCE
        day = sc.acq.dt.normalize()
        sc["rain3"] = [float(pr.loc[d - pd.Timedelta(days=2): d].sum()) for d in day]      # the scene's day and the two before
        sc["rain9"] = [float(pr.loc[d - pd.Timedelta(days=8): d].sum()) for d in day]
        wet, dry = sc[sc.rain3 >= 5], sc[sc.rain9 < 1]
        before = sc[(sc.acq < S.start) & (sc.acq >= S.start - pd.Timedelta(days=7))]
        mk = sc[sc.n_excl > 0]                              # a few scenes come without the exclusion layer
        gfm = dict(scenes=sc, after=after, before=before, first=sc.acq.min(), n=len(sc), excl_share=float(mk.excl_share.median()),
                   excl_range=(float(mk.excl_share.min()), float(mk.excl_share.max())), n_nomask=int((sc.n_excl == 0).sum()),
                   top_wet=wet.loc[wet.flood_km2.idxmax()] if len(wet) else None, n_wet=len(wet),
                   top_dry=dry.loc[dry.flood_km2.idxmax()] if len(dry) else None,
                   tuw_since=float(sc[sc.acq >= "2024-01-01"].tuw_km2.median()), tuw_pre=float(sc[(sc.acq >= "2021-01-01") & (sc.acq < SINCE)].tuw_km2.median()))
        mp = fdir / f"gfm_map_{cfg['gfm_map']}.npz"
        if mp.exists():
            gfm["map"] = dict(np.load(mp, allow_pickle=False))
            gfm["gov"] = gfm_by_governorate(gfm["map"], govs)
            gfm["boxes"] = gfm_by_box(gfm["map"], cfg.get("gfm_boxes", []))

    # FloodScan
    fs, fw = floodscan_gaza(poly)
    st = ei[(ei.start >= fs.index[0]) & (ei.total >= 25)].copy()
    st["fs"] = [fs.loc[s - pd.Timedelta(days=1): e + pd.Timedelta(days=5)].max() for s, e in zip(st.start, st.end)]
    big = st.nlargest(10, "total")
    floodscan = dict(series=fs, w=fw, last=fs.index[-1], now=float(fs.loc[S.start - pd.Timedelta(days=1):].max()),
                     storms=st, n25=len(st), n25_hit=int((st.fs >= 0.01).sum()), big=big,
                     top=fs.groupby(np.where(fs.index.month >= 8, fs.index.year, fs.index.year - 1)).max().sort_values(ascending=False).head(6))

    # the catalogue against both satellites
    table = event_table(spec, a, pr, fs, gfm["scenes"] if gfm else None, S, cfg)
    out = dict(cfg=cfg, rain=rain, fc=fc, pastfc=pastfc, gfm=gfm, fs=floodscan, table=table)
    if fc is not None:
        fig_forecast(out, L.OUT / "storm_forecast.png")
    if gfm is not None and "map" in gfm:
        fig_satellites(out, L.OUT / "storm_satellites.png")
    return out


def gfm_by_governorate(m: dict, govs) -> list[dict]:
    """Share of each governorate in GFM's exclusion mask on the mapped scene."""
    from affine import Affine
    from rasterio.features import geometry_mask
    tr = Affine(*m["transform"][:6]); inside = m["mask"]
    g = govs.to_crs(str(m["crs"]))
    out = []
    for _, r in g.iterrows():
        mm = ~geometry_mask([r.geometry], out_shape=inside.shape, transform=tr) & inside
        if mm.sum() == 0:
            continue
        out.append(dict(name=r.ADM2_EN, km2=float(mm.sum() * GFM_PX_KM2), excl=float((m["exclusion_mask"][mm] == 1).mean())))
    return out


def gfm_by_box(m: dict, boxes: list[dict]) -> list[dict]:
    """Share of a lon/lat box (inside Gaza) in GFM's exclusion mask, e.g. Gaza City or the Al Mawasi coast."""
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import geometry_mask
    from shapely.geometry import box
    tr = Affine(*m["transform"][:6]); inside = m["mask"]
    out = []
    for b in boxes:
        g = gpd.GeoSeries([box(*b["box"])], crs=4326).to_crs(str(m["crs"]))
        mm = ~geometry_mask(g, out_shape=inside.shape, transform=tr) & inside
        out.append(dict(name=b["name"], km2=float(mm.sum() * GFM_PX_KM2), excl=float((m["exclusion_mask"][mm] == 1).mean())))
    return out


def event_table(spec, a, pr, fs, sc, S, cfg) -> list[dict]:
    """Every catalogued rain event (before October 2023 from [[prewar]], since from the events CSV), plus this storm:
    IMERG wettest day, FloodScan's largest Gaza share from the day before to five days after, and GFM's largest flooded
    area in any scene from the start to three days after the end (GFM only since 2015)."""
    rows = [dict(start=pd.Timestamp(p["start"]), end=pd.Timestamp(p.get("end", p["start"])), what=re.sub("<[^>]+>", "", p["what_html"]),
                 since=False) for p in spec.get("prewar", []) if "R" in p["hazard"].split()]
    ev = a.get("ev")
    if ev is not None:
        for _, r in ev.iterrows():
            if "R" in str(r.hazard).split("+"):
                rows.append(dict(start=r.start, end=r.end, what=str(r.description), since=True))
    rows.append(dict(start=S.start, end=S.end, what=cfg["label"], since=True, now=True))
    out = []
    for r in rows:
        s, e = r["start"], r["end"]
        rr = pr.loc[s - pd.Timedelta(days=1): e]
        f = fs.loc[s - pd.Timedelta(days=1): e + pd.Timedelta(days=5)]
        g = None
        if sc is not None and s >= sc.acq.min():
            gg = sc[(sc.acq >= s) & (sc.acq < e + pd.Timedelta(days=4))]
            g = dict(n=len(gg), km2=float(gg.flood_km2.max()) if len(gg) else np.nan)
        out.append(dict(r, rain=float(rr.max()) if len(rr) else np.nan, fs=float(f.max()) if len(f) else np.nan, gfm=g))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_forecast(r: dict, out: Path) -> None:
    """Rain forecast over Gaza for the storm's 48 hours, from every ECMWF run in the fortnight before: ENS 10th–90th
    percentile bar and median, HRES, against IMERG."""
    fc = r["fc"]; R = fc["runs"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=150)
    x = np.arange(len(R))
    for i, (t, q) in enumerate(R.iterrows()):
        alpha = 0.45 if q.partial else 1.0
        ax.vlines(i, q.p10, q.p90, color=C_ENS, lw=5, alpha=0.35 * alpha, zorder=2)
        ax.plot(i, q.med, "o", ms=7, color=C_ENS, alpha=alpha, zorder=3, mec="white", mew=1)
        ax.plot(i, q.hres, "D", ms=5.5, color=C_HRES, alpha=alpha, zorder=4, mec="white", mew=0.8)
        ax.text(i, -6.5, f"{100 * q.P20:.0f}", ha="center", va="center", fontsize=7.5, color=C_MUTED if q.P20 < 0.5 else C_ENS,
                fontweight="bold" if q.P20 >= 0.5 else "normal")
    ax.axhline(fc["obs"], color=C_OBS, lw=2, zorder=1)
    ax.text(len(R) - 0.4, fc["obs"] + 1.2, f"IMERG: {fc['obs']:.0f} mm fell", ha="right", va="bottom", fontsize=8.5, color=C_OBS, fontweight="bold")
    ax.axhline(20, color="#9aa5a6", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.4, 21, "20 mm", ha="left", va="bottom", fontsize=7.5, color=C_MUTED)
    ax.text(-0.9, -6.5, "% of members\n≥ 20 mm", ha="right", va="center", fontsize=7, color=C_MUTED, linespacing=1.1)
    ahead = lambda t: (fc["w0"] - t) / pd.Timedelta(days=1)
    ax.set_xticks(x, [(f"{t:%-d %b}" if t.hour == 0 else "") + f"\n{t:%H}Z\n" + (f"{ahead(t):g}" if ahead(t) >= 0 else "inside") for t in R.index], fontsize=7.5)
    ax.set_xlim(-0.6, len(R) - 0.4); ax.set_ylim(-11, max(fc["obs"], R.p90.max(), R.hres.max()) * 1.12)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60]); ax.set_ylim(bottom=-11)
    ax.set_ylabel(f"rain over Gaza, {fc['w0']:%-d}–{(fc['w1'] - pd.Timedelta(days=1)):%-d %b} (mm)", fontsize=9, color=C_MUTED)
    ax.set_xlabel("ECMWF run: date, hour it started (UTC) and, below, days before the window opened", fontsize=9, color=C_MUTED)
    hs = [plt.Line2D([], [], color=C_ENS, lw=5, alpha=0.35, label="ENS, 10th–90th percentile of 50 members"),
          plt.Line2D([], [], marker="o", color=C_ENS, ls="", ms=7, label="ENS median"),
          plt.Line2D([], [], marker="D", color=C_HRES, ls="", ms=5.5, label="HRES (the single high-resolution run)")]
    ax.legend(handles=hs, frameon=False, fontsize=8, loc="upper left", ncol=1)
    edd._style_ax(ax); ax.spines["bottom"].set_position(("data", -11))
    L._save(fig, out)


def _map_classes(m: dict) -> tuple[np.ndarray, tuple]:
    """GFM classes on the scene's 20 m Equi7 grid, reprojected (nearest) to a lat/lon grid of about 20 m for display:
    0 outside Gaza, 1 mapped and dry, 2 excluded, 3 permanent water, 4 flagged by one algorithm only, 5 GFM flood map."""
    import rasterio.warp
    from affine import Affine
    cls = _classes_native(m)
    w, s_, e, n = 34.2, 31.2, 34.59, 31.61
    res = 0.0002
    dst = np.zeros((int(round((n - s_) / res)), int(round((e - w) / res))), np.uint8)
    rasterio.warp.reproject(cls, dst, src_transform=Affine(*m["transform"][:6]), src_crs=str(m["crs"]),
                            dst_transform=Affine(res, 0, w, 0, -res, n), dst_crs="EPSG:4326",
                            resampling=rasterio.warp.Resampling.nearest, src_nodata=0, dst_nodata=0)
    return dst, (w, e, s_, n)


def fig_satellites(r: dict, out: Path) -> None:
    """Left: GFM on the scene after the storm (what it could map, what it mapped). Right: both satellites' records over Gaza,
    one panel each, with the catalogued events."""
    g, F = r["gfm"], r["fs"]
    ink = "#3d4647"
    fig = plt.figure(figsize=(10.4, 5.8), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.1], height_ratios=[1, 1], wspace=0.14, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    cls, ext = _map_classes(g["map"])
    cmap = ListedColormap(["#ffffff", "#dfe9ee", "#9aa5a6", "#7fb0d8", "#E8A33D", "#B2182B"])
    ax.imshow(cls, cmap=cmap, vmin=0, vmax=5, interpolation="nearest", extent=ext)
    ax.set_aspect(1 / np.cos(np.deg2rad(31.4)))
    ax.set_axis_off()
    m = g["map"]; inside = m["mask"]; n_in = inside.sum()
    nat = _classes_native(m)
    c0 = lambda k: int((nat == k).sum())
    lab = [(1, f"mapped, no flood ({100 * c0(1) / n_in:.0f}%)"), (2, f"excluded: GFM cannot map ({100 * c0(2) / n_in:.0f}%)"),
           (4, f"flagged by one algorithm only ({c0(4) * GFM_PX_KM2:.1f} km²)"), (5, f"GFM flood map ({c0(5) * GFM_PX_KM2:.1f} km²)")]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap(k / 5)) for k, _ in lab], labels=[t for _, t in lab], frameon=False, fontsize=7.5,
              loc="upper left", bbox_to_anchor=(-0.02, 0.02), handlelength=1.2)
    t = pd.Timestamp(str(r["cfg"]["gfm_map_time"]))
    ax.set_title(f"GFM, Sentinel-1 scene of\n{t:%-d %B %Y, %H:%M} UTC", fontsize=9, color=C_TEXT, loc="left")
    sc = g["scenes"]
    x0, x1 = pd.Timestamp("1998-01-01"), F["last"] + pd.Timedelta(days=90)
    gaza_km2 = _gaza_km2(g)
    for i, k in enumerate(("fs", "gfm")):
        a = fig.add_subplot(gs[i, 1])
        if k == "fs":
            v = F["series"] * 100; v = v[v > 0]
            a.vlines(v.index, 0, v.values, color=ink, lw=1.1)
            ymax = 32
            a.set_title("FloodScan (AER, 1/12°): share of Gaza called flooded, daily", fontsize=8.5, color=C_TEXT, loc="left")
        else:
            v = sc.set_index("acq").flood_km2 / gaza_km2 * 100
            cap = 1.0
            a.vlines(v.index, 0, v.clip(upper=cap).values, color=ink, lw=1.1)
            r9 = sc.set_index("acq").rain9
            for tt, val in v[v > cap].items():
                why = "after nine dry days:\nalmost certainly a false detection" if r9.loc[tt] < 1 else "(axis cut)"
                a.annotate(f"{val:.0f}% on {tt:%-d %b %Y}, {why}", (tt, cap), xytext=(-8, -2),
                           textcoords="offset points", ha="right", va="top", fontsize=7, color=C_MUTED)
            ymax = cap * 1.08
            a.axvspan(x0, g["first"], color="#f1f3f4", lw=0)
            a.text(x0 + pd.Timedelta(days=200), ymax * 0.5, f"no GFM before {g['first']:%Y}", fontsize=7.5, color=C_MUTED, va="center")
            a.set_title(f"GFM (Sentinel-1, 20 m): share of Gaza in the flood map, per scene (axis cut at {cap:.0f}%)", fontsize=8.5, color=C_TEXT, loc="left")
        for row in r["table"]:
            now = bool(row.get("now"))
            a.plot(row["start"], ymax, marker="v", ms=6 if now else 4.5, color=L.C_REGIME[bool(row["since"])], ls="", clip_on=False,
                   zorder=5 if now else 4, mec=C_TEXT if now else "none", mew=1.0)
        a.set_xlim(x0, x1); a.set_ylim(0, ymax)
        a.set_ylabel("% of Gaza", fontsize=8, color=C_MUTED)
        edd._style_ax(a); a.tick_params(labelsize=8)
    hs = [plt.Line2D([], [], marker="v", color=L.C_REGIME[False], ls="", ms=5, label="catalogued rain event, before October 2023"),
          plt.Line2D([], [], marker="v", color=L.C_REGIME[True], ls="", ms=5, label="since October 2023 (outlined: this storm)")]
    fig.legend(handles=hs, frameon=False, fontsize=7.5, loc="lower right", ncol=2, bbox_to_anchor=(0.9, -0.01))
    L._save(fig, out)


def _classes_native(m: dict) -> np.ndarray:
    inside = m["mask"]
    cls = np.zeros(inside.shape, np.uint8); cls[inside] = 1
    cls[inside & (m["exclusion_mask"] == 1)] = 2
    cls[inside & (m["reference_water_mask"] == 1)] = 3
    one = np.zeros(inside.shape, bool)
    for k in ("tuw_flood_extent", "dlr_flood_extent", "list_flood_extent"):
        if k in m:
            one |= (m[k] == 1)
    cls[inside & one] = 4
    cls[inside & (m["ensemble_flood_extent"] == 1)] = 5
    return cls


# --------------------------------------------------------------------------- #
# Page text
# --------------------------------------------------------------------------- #
def _d(t) -> str:
    return f"{pd.Timestamp(t):%-d %b %Y}"


def render(spec: dict, a: dict) -> str:
    r = a["storm_now"]; cfg = r["cfg"]; R = r["rain"]; fc = r["fc"]; g = r["gfm"]; F = r["fs"]
    o = [f'<h3 id="{cfg.get("anchor", "storm-now")}">{html.escape(cfg["title"])}</h3>']
    o.append(cfg.get("lead_html", ""))
    # rain
    since = ", ".join(f"{L._yr(y)}: {t:%-d %b}" if pd.notna(t) else f"{L._yr(y)}: none" for y, t in R["first_since"].items())
    ear = R["first_earlier"]
    o.append(f'<p><strong>The rain.</strong> IMERG puts {R["max"]:.0f} mm over Gaza on {R["day"]:%-d %B} ({R["north"]:.0f} mm in the north, '
             f'{R["south"]:.0f} mm in the south) and {R["total"]:.0f} mm over the storm. That is {_rank(R["rank_all"], "wettest day")} of '
             f'{R["n_all"]:,} since {R["first_year"]} and {_rank(R["rank_month"], "wettest " + R["day"].strftime("%B") + " day")}. '
             + (f'It is also the first storm of the winter whose wettest day reached 20 mm, and it came earlier than in any of the winters '
                f'since October 2023 ({since}); in the {R["first_n"]} winters of IMERG before this one, the first came earlier only {len(ear)} times '
                f'({", ".join(f"{t:%-d %b %Y}" for t in ear)}). ' if R["first20"] else '')
             + 'Storms that begin in October–December are the one kind El Niño winters have brought more often (see "How many storms a winter brings" below); '
               'one storm does not show that this winter is following that pattern.'
             + '<!--more--> ' + cfg.get("rain_caveat_html", "") + '</p>')
    sim = R["similar"]
    if len(sim):
        rep = sim[sim.report.notna()]
        o.append(f'<p class="small">Storms of a similar size before October 2023 (IMERG wettest day {cfg.get("similar_mm", [35, 50])[0]}–'
                 f'{cfg.get("similar_mm", [35, 50])[1]} mm): {len(sim)}, of which {len(rep)} {"is" if len(rep) == 1 else "are"} in the catalogue of section 4 — '
                 + "; ".join(f'{_d(s.start)}: {re.sub("<[^>]+>", "", s.report["what_html"]).rstrip(". ")}' for s in rep.itertuples())
                 + '. ' + cfg.get("similar_html", "") + '</p>')
    # forecasts
    if fc is not None:
        Rr = fc["runs"]; full = Rr[~Rr.partial]; lead = fc["lead"]
        fm, fp, fh = fc["first_med20"], fc["first_p50"], fc["first_hres20"]
        best = full.loc[full.med.idxmax()]
        win = f'{fc["w0"]:%-d}–{(fc["w1"] - pd.Timedelta(days=1)):%-d} {fc["w0"]:%B}'
        when = lambda t: f'{t:%-d %B} {t:%H}:00 UTC'
        ahead = lambda t: f'{lead(t):.1f} days before the window opened'
        f1 = next((t for t in full.index if full.loc[t, "med"] >= 20), None)        # first run whose median reached 20 mm
        q1 = full.loc[f1] if f1 is not None else None
        dip = full[(full.index > f1) & (full.med < 20)] if f1 is not None else full.iloc[:0]
        part = Rr[Rr.partial]
        txt = '<p><strong>Could it have been forecast?</strong> '
        if f1 is None:
            txt += f'No ECMWF run before the window put the middle of its 50 members at 20 mm or more over Gaza for {win}. '
        else:
            txt += (f'Only partly. Over Gaza for {win}, ECMWF\'s ensemble kept its middle member at {full.loc[:f1].med.iloc[:-1].min():.0f}–'
                    f'{full.loc[:f1].med.iloc[:-1].max():.0f} mm until the run of {when(f1)}, {ahead(f1)}, which put it at {q1.med:.0f} mm '
                    f'({100 * q1.P20:.0f}% of members at 20 mm or more; the high-resolution run, HRES, {q1.hres:.0f} mm). ')
            if len(dip) and fm is not None and fm > f1:
                txt += (f'The next runs backed off (median {dip.med.min():.0f}–{dip.med.max():.0f} mm), and the forecast held at 20 mm or more only from '
                        f'{when(fm)}, {ahead(fm)}. ')
        txt += (('Even so, n' if f1 is not None else 'N') + f'o run before the window expected as much as fell: the wettest ensemble median, {best.med:.0f} mm '
                f'(run of {when(best.name)}), is about {100 * best.med / fc["obs"]:.0f}% of the {fc["obs"]:.0f} mm IMERG measured, and at most '
                f'{100 * full.P40.max():.0f}% of members reached 40 mm')
        txt += (f' ({100 * part.P40.max():.0f}% in the run that started on {when(part.index[0])}, inside the window).' if len(part) else '.')
        o.append(txt
                 + '<!--more-->' + cfg.get("forecast_note_html", "") + '</p>')
        o.append(f'<figure><img src="storm_forecast.png" alt="ECMWF forecasts of the rain over Gaza for {html.escape(cfg["label"])}, by run">'
                 f'<figcaption>Rain over the Gaza Strip for {fc["w0"]:%-d} 00:00 – {fc["w1"]:%-d %B} 00:00 UTC in each ECMWF run of the fortnight before, '
                 'from ECMWF\'s open data (0.25°, the 0.25° cells weighted by their share of Gaza). Bars: the middle 80% of the 50 ensemble members; '
                 'dots: their median; diamonds: HRES. Numbers under the axis: the share of members with 20 mm or more. The window opens at 00:00 UTC on '
                 f'{fc["w0"]:%-d %B}; the rain began that evening. Faded: the run that started inside the window. IMERG is a satellite estimate and itself '
                 'uncertain; ERA5 for these days is not yet available.</figcaption></figure>')
        if r["pastfc"]:
            o.append(_past_summary(r["pastfc"], cfg))
            o.append(_past_table(r["pastfc"]))
    # satellites
    if g is not None and F is not None:
        o.append(_satellite_text(r))
        o.append(f'<figure><img src="storm_satellites.png" alt="What GFM and FloodScan saw over Gaza"><figcaption>{_sat_caption(r)}</figcaption></figure>')
        o.append(_event_table_html(r))
    o.append(cfg.get("after_html", ""))
    return "\n".join(o)


def _rank(n: int, what: str) -> str:
    """'the wettest day', 'the third-wettest day', 'the 21st-wettest day'."""
    if n == 1:
        return f"the {what}"
    w = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}")
    return f"the {w}-{what}"


def _past_summary(pastfc: list[dict], cfg: dict) -> str:
    """Earlier storms split by how much ERA5 (the model's own reanalysis) measured (cut at `past_era5_mm`, which sits in a gap of
    the ERA5 values); how the ensemble did with each group."""
    cut = cfg.get("past_era5_mm", 12)
    both = [p for p in pastfc if p["era5"] >= cut]
    only = [p for p in pastfc if not p["era5"] >= cut]
    p3 = lambda p: p["runs"][p["runs"].lead_d == 3].P20
    b3 = [float(p3(p).iloc[0]) for p in both if len(p3(p))]
    o_max = max((float(p["runs"].P20.max()) for p in only), default=np.nan)
    rng = lambda ps: f'{min(p["era5"] for p in ps):.0f}–{max(p["era5"] for p in ps):.0f} mm'
    t = (f'<p>In the {_word(len(pastfc))} earlier storms that the archive covers, the forecasts did well with the {_word(len(both))} that ERA5 also saw '
         f'({rng(both)} in ERA5 against {min(p["obs"] for p in both):.0f}–{max(p["obs"] for p in both):.0f} mm in IMERG): three days ahead, '
         f'{100 * min(b3):.0f}–{100 * max(b3):.0f}% of members had 20 mm or more for each of them. The {_word(len(only))} that ERA5 barely saw '
         f'({rng(only)}, against {min(p["obs"] for p in only):.0f}–{max(p["obs"] for p in only):.0f} mm in IMERG) got almost nothing from the forecasts '
         f'at any lead (at most {100 * o_max:.0f}% of members). '
         if b3 and only else '<p>')
    return t + cfg.get("past_note_html", "") + '</p>'


def _past_table(pastfc: list[dict]) -> str:
    rows = []
    for p in pastfc:
        R = p["runs"].sort_values("lead_d", ascending=False)
        cells = []
        for k in (5, 3, 1):
            q = R[R.lead_d == k]
            cells.append(f'<td>{q.med.iloc[0]:.0f} · {100 * q.P20.iloc[0]:.0f}%</td>' if len(q) else '<td>—</td>')
        e5 = "—" if np.isnan(p["era5"]) else f'{p["era5"]:.0f}'
        rows.append(f'<tr><td>{html.escape(p["label"])}</td><td>{p["obs"]:.0f}</td><td>{e5}</td>' + "".join(cells) + '</tr>')
    return ('<p class="small">The earlier storms of the open-data archive (from February 2024): rain over Gaza in the 48 hours from 00 UTC on the '
            'storm\'s first day, measured (IMERG and ERA5) and forecast (the ensemble median · the share of members at 20 mm or more) in the 00 UTC runs 5, 3 and '
            '1 days before. The two measurements disagree for some storms: ERA5 smooths rain over 25 km cells and comes from the same ECMWF model as the '
            'forecasts, while IMERG has both missed and invented storms here.</p>'
            '<table><thead><tr><th>Storm (48 h from)</th><th>IMERG, mm</th><th>ERA5, mm</th><th>5 days ahead</th><th>3 days ahead</th><th>1 day ahead</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table>')


def _satellite_text(r: dict) -> str:
    g, F, cfg = r["gfm"], r["fs"], r["cfg"]
    af = g["after"]
    gov = sorted(g.get("gov", []), key=lambda q: -q["excl"])
    gtxt = (f' — from {100 * gov[-1]["excl"]:.0f}% in {gov[-1]["name"]} to {100 * gov[0]["excl"]:.0f}% in {gov[0]["name"]}' if gov else "")
    tw, td = g["top_wet"], g["top_dry"]
    p1 = (f'<p><strong>What the flood satellites saw.</strong> Neither of the two satellite flood products the team uses picked up this flood. '
          f'<em>GFM</em> (Copernicus Global Flood Monitoring) maps water in every Sentinel-1 radar scene at 20 m. '
          f'Sentinel-1 covered all of Gaza {_times(len(af))} in the {cfg.get("gfm_days_after", 4)} days from the start of the storm'
          + (f' (first at {af.acq.iloc[0]:%-d %B %H:%M} UTC)' if len(af) else '') + ', and GFM mapped '
          + (f'{af.flood_km2.max():.2f} km² of flood water' if len(af) and af.flood_km2.max() > 0 else 'no flood water') + '. '
          f'GFM cannot map {100 * g["excl_share"]:.0f}% of Gaza at all{gtxt}: its exclusion mask covers built-up land (taken from a 2018 map of buildings) and '
          'surfaces that look like water to the radar, such as sand. The mask has been the same in every scene since 2015 '
          f'({100 * g["excl_range"][0]:.1f}–{100 * g["excl_range"][1]:.1f}% of Gaza), so it still treats destroyed neighbourhoods as built-up'
          + (': it leaves out ' + " and ".join(f'{100 * b["excl"]:.0f}% of {b["name"]}' for b in g["boxes"]) if g.get("boxes") else '')
          + '. In the part it can map, one algorithm flagged '
          f'{_one_alg_km2(g):.1f} km² on {pd.Timestamp(str(cfg["gfm_map_time"])):%-d %B}; GFM publishes a pixel as flooded only when most of its three algorithms agree. '
          f'In the {g["n"]:,} scenes that covered Gaza since {g["first"]:%B %Y}, the most GFM mapped within three days of rain was {tw.flood_km2:.1f} km² '
          f'({tw.acq:%-d %B %Y}, about {100 * tw.flood_km2 / _gaza_km2(g):.1f}% of Gaza).'
          + (f' Its largest map of all, {td.flood_km2:.0f} km² on {td.acq:%-d %B %Y}, came after nine days without rain and matches no report.'
             if td is not None and td.flood_km2 > tw.flood_km2 else '')
          + '<!--more-->'
          + (f' The single-algorithm flags are not a sign of this flood either: since 2024 the TUW algorithm flags about {g["tuw_since"]:.0f} km² in a typical '
             f'scene (against {_km2(g["tuw_pre"])} in 2021–2023), and it flagged less after the storm '
             f'({_tuw_after(g):.1f} km²) than in the dry days before it ({", ".join(f"{q.tuw_km2:.0f}" for q in g["before"].itertuples())} km²).'
             if len(g["before"]) else '')
          + cfg.get("gfm_note_html", "") + '</p>')
    big = F["big"].sort_values("total", ascending=False)
    seen = big[big.fs >= 0.01]; missed = big[big.fs < 0.01]
    wadi = missed[(missed.start <= "2010-01-19") & (missed.end >= "2010-01-18")]
    p2 = (f'<p><em>FloodScan</em> does cover Gaza: its Africa grid reaches 40°N and 60°E, and the team\'s archive holds every day since January 1998. '
          f'It shows nothing for this storm so far (file of {F["last"]:%-d %B}). Its pixels are about 9 km across and every one that touches Gaza is '
          f'part sea and part Israel, so it rarely sees Gaza\'s floods: of the {F["n25"]} IMERG storms of 25 mm or more since 1998, it put at least 1% of '
          f'Gaza under water within five days for {F["n25_hit"]}. Of the {_word(len(big))} largest it registered {_word(len(seen))}: '
          + ", ".join(f'{q.start:%B %Y} ({100 * q.fs:.0f}%)' if q.fs >= 0.1 else f'{q.start:%B %Y} ({100 * q.fs:.1f}%)' for q in seen.itertuples())
          + f'; for the other {_word(len(missed))} it showed less than 1%' + (', including the Wadi Gaza flood of January 2010' if len(wadi) else '') + '. '
          'Neither product can stand in for reports from the sites; for warning, the rain forecasts are the usable signal.</p>')
    return p1 + p2


def _times(n: int) -> str:
    return {0: "no time", 1: "once", 2: "twice"}.get(n, f"{n} times")


def _word(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n, str(n))


def _km2(v: float) -> str:
    return "less than 0.1 km²" if v < 0.1 else f"{v:.1f} km²"


def _gaza_km2(g: dict) -> float:
    return float(g["scenes"].n_gaza.median() * GFM_PX_KM2)


def _tuw_after(g: dict) -> float:
    return float(g["after"].tuw_km2.iloc[0]) if len(g["after"]) else float("nan")


def _one_alg_km2(g: dict) -> float:
    m = g.get("map")
    if m is None:
        return float("nan")
    one = np.zeros(m["mask"].shape, bool)
    for k in ("tuw_flood_extent", "dlr_flood_extent", "list_flood_extent"):
        if k in m:
            one |= (m[k] == 1)
    return float((one & m["mask"] & (m["ensemble_flood_extent"] != 1)).sum() * GFM_PX_KM2)


def _sat_caption(r: dict) -> str:
    return ('Left: the GFM layers of the first Sentinel-1 scene after the storm, inside the Gaza Strip (COD-AB). Grey: GFM\'s exclusion mask; orange: pixels '
            'flagged by only one of GFM\'s three algorithms, which GFM does not publish as flooded; red: GFM\'s flood map. Right: the two records over Gaza. '
            'FloodScan: daily share of Gaza\'s area called flooded, each 1/12° pixel weighted by its share of the Gaza Strip (team prod archive, Africa grid). '
            'GFM: share of Gaza in the flood map, for each scene that covered at least 90% of Gaza. Triangles: the rain events of the catalogue in section 4.')


def _event_table_html(r: dict) -> str:
    rows = []
    for e in r["table"]:
        g = e["gfm"]
        gtxt = "—" if g is None else ("no scene" if g["n"] == 0 else f'{g["km2"]:.2f} km² ({g["n"]})')
        what = textwrap.shorten(e["what"], 110, placeholder="…")
        cls = ' class="now"' if e.get("now") else ""
        rows.append(f'<tr{cls}><td>{_d(e["start"])}</td><td>{html.escape(what)}</td><td>{e["rain"]:.0f}</td>'
                    f'<td>{100 * e["fs"]:.1f}%</td><td>{gtxt}</td></tr>')
    return ('<details class="note"><summary>Every catalogued rain event against both satellites</summary>'
            '<table><thead><tr><th>Start</th><th>Reported</th><th>IMERG wettest day, mm</th><th>FloodScan, largest share of Gaza</th>'
            '<th>GFM, largest flood map (scenes)</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
            '<p class="small">IMERG over the event window from the day before; FloodScan from the day before to five days after; GFM in scenes '
            'from the start to three days after the end (— before the GFM archive begins in 2015).</p></details>')
