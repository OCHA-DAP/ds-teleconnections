"""West Bank farming section for the levant ENSO deep dive: crop years against rainfall and El Niño.

Called from levant_deep_dive when the page's TOML has an [agri] table. Public sources only:

* FAOSTAT crops and livestock (QCL), Palestine (West Bank and Gaza together), 1994–2022. The subset
  used here is committed as deep_dives/data/pse_faostat_crops.csv; it was cut from the bulk file
  https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip
  (``fetch_faostat`` rebuilds it). Only official figures (flag A) enter the field-crop index.
* PCBS by governorate (olive presses survey; field crops), committed as deep_dives/data/pcbs_*.csv
  with the source of every row.
* NOAA STAR Blended Vegetation Health (4 km), weekly means for the West Bank (province 2 of PSE),
  1982–: the Vegetation Condition Index (VCI, from NDVI) over cropland and over all land.
* Zone rainfall: the page's ERA5 cells, overlap-weighted over each zone's governorates.

One convention throughout: crop year Y is the harvest of Y. Field crops harvested in May–June Y
and olives picked in October–November Y both follow the rainy season October Y−1 to April Y,
which the rest of the page labels Y−1 (``season_year``).
"""
from __future__ import annotations

import html
import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats

import enso_deep_dive as edd
import levant_deep_dive as L

DATA = edd.DEEP_DIR / "data"
FAO_CSV = DATA / "pse_faostat_crops.csv"
FAO_BULK = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
VHP_URL = ("https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?provinceID=2&country=PSE"
           "&adminVHversion=GC_current&yearlyTag=Weekly&type=Mean&TagCropland={tag}&year1=1982&year2={y2}")
FIELD = ["Wheat", "Barley"]                     # the rainfed cereals: official yields most years
PULSES = ["Lentils, dry", "Chick peas, dry"]    # imputed (flag I/X) in 2009–2017: shown, not pooled
PHASES = ["El Niño", "Neutral", "La Niña"]
PCOL = {"El Niño": L.C_EN, "Neutral": L.C_NEU, "La Niña": L.C_LN}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_faostat(items: list[str]) -> None:
    """Rebuild the committed FAOSTAT subset (Palestine; area, production, yield) from the bulk file."""
    r = requests.get(FAO_BULK, timeout=600); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith("All_Data_(Normalized).csv"))
        d = pd.read_csv(z.open(name), encoding="latin-1")
    d = d[d.Area.str.startswith("Palest") & d.Item.isin(items) & d.Element.isin(["Area harvested", "Production", "Yield"]) & (d.Flag != "M")]
    d = d.rename(columns=str.lower)[["item", "element", "year", "unit", "value", "flag"]].sort_values(["item", "element", "year"])
    d.to_csv(FAO_CSV, index=False)


def faostat() -> pd.DataFrame:
    return pd.read_csv(FAO_CSV)


def fao_series(fao: pd.DataFrame, item: str, element: str = "Yield", official: bool = False) -> pd.Series:
    d = fao[(fao.item == item) & (fao.element == element)]
    if official:
        d = d[d.flag == "A"]
    s = d.set_index("year").value.astype(float)
    if item in FIELD:          # FAOSTAT repeats 2013 as 2014 for wheat and barley (area, production, yield)
        s = s.drop(2014, errors="ignore")
    return s[s > 0]


def detrended_pct(s: pd.Series) -> pd.Series:
    """Percentage above or below a log-linear trend fitted over the series itself."""
    t = s.index.values.astype(float); ly = np.log(s.values)
    b = np.polyfit(t, ly, 1)
    return pd.Series(100 * (np.exp(ly - np.polyval(b, t)) - 1), index=s.index)


def olive_bearing(s: pd.Series) -> pd.Series:
    """Olive production with alternate bearing taken out: % above or below what last year's crop
    predicts (log production regressed on the previous year's)."""
    ly = np.log(s); prev = ly.shift(1)
    d = pd.DataFrame({"y": ly, "p": prev}).dropna()
    b = np.polyfit(d.p, d.y, 1)
    return pd.Series(100 * (np.exp(d.y - np.polyval(b, d.p)) - 1), index=d.index)


def vhp(tag: str) -> pd.DataFrame:
    """NOAA STAR weekly VH means for the West Bank (tag 'crop' = cropland, 'land' = all land)."""
    path = L.CACHE / f"vhp_{tag}.txt"
    if not path.exists() or (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days > 20:
        r = requests.get(VHP_URL.format(tag=tag, y2=pd.Timestamp.now().year), timeout=120); r.raise_for_status()
        path.write_text(r.text)
    rows = [l.replace("<tt><pre>", "").strip().rstrip(",") for l in path.read_text().splitlines()
            if re.match(r"^(<tt><pre>)?\d{4},", l)]
    d = pd.DataFrame([list(map(float, r.split(","))) for r in rows], columns=["year", "week", "smn", "smt", "vci", "tci", "vhi"])
    d = d[d.vci >= 0]
    d.index = pd.to_datetime(d.year.astype(int).astype(str) + "-01-01") + pd.to_timedelta((d.week - 1) * 7, "D")
    return d


def spring_vci(d: pd.DataFrame, months=(3, 4), min_weeks: int = 7) -> pd.Series:
    """Mean VCI over March–April of each year (the peak of the rainfed season)."""
    w = d[d.index.month.isin(months)]
    g = w.groupby(w.index.year).vci
    s = g.mean()[g.count() >= min_weeks]
    return s


MODIS_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MODIS_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs"
SPRING_DOY = (33, 49, 65, 81, 97)              # MOD13Q1 16-day composites starting 2 Feb – 7 Apr (to 22 April)


def modis_zone_ndvi(govs, zones: list[dict]) -> pd.DataFrame:
    """Mean NDVI per zone for each spring MOD13Q1 composite (Terra, 250 m, collection 6.1), read from the
    Microsoft Planetary Computer's public COGs with windowed reads; fill values dropped. Cached and topped
    up with new composites. Columns: one per zone plus "West Bank"; index: composite start date."""
    import geopandas as gpd
    import rasterio
    from concurrent.futures import ThreadPoolExecutor
    from rasterio.features import rasterize
    from rasterio.windows import from_bounds
    path = L.CACHE / "modis_ndvi_zones.parquet"
    have = pd.read_parquet(path) if path.exists() else None
    shapes = {z["name"]: govs[govs.ADM2_EN.isin(z["governorates"])].union_all() for z in zones}
    shapes["West Bank"] = govs.union_all()
    w, s_, e, n = govs.total_bounds
    items = []
    for y in range(2000, pd.Timestamp.now().year + 1):
        r = requests.post(MODIS_STAC, json={"collections": ["modis-13Q1-061"], "bbox": [w, s_, e, n],
                                            "datetime": f"{y}-01-25/{y}-04-30", "limit": 200}, timeout=120)
        r.raise_for_status()
        for f in r.json()["features"]:
            prod, a_, tile = f["id"].split(".")[:3]
            if prod == "MOD13Q1" and int(a_[5:8]) in SPRING_DOY and int(a_[1:5]) == y:
                items.append((pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=int(a_[5:8]) - 1), tile, f["assets"]["250m_16_days_NDVI"]["href"]))
    done = set(have.index) if have is not None else set()
    todo = [it for it in items if it[0] not in done]
    if todo:
        print(f"  reading {len(todo)} MODIS NDVI tile-composites…", flush=True)
        token = requests.get(MODIS_SAS, timeout=60).json()["token"]
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
        masks = {}

        def one(it):
            date, tile, href = it
            with rasterio.Env(**env), rasterio.open(f"{href}?{token}") as src:
                g = gpd.GeoSeries(list(shapes.values()), crs=4326).to_crs(src.crs)
                win = from_bounds(*g.total_bounds, src.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                a = src.read(1, window=win).astype("float32")
                tr = src.window_transform(win)
                key = (tile, win.col_off, win.row_off, win.width, win.height)
                if key not in masks:
                    masks[key] = {k: rasterize([(geom, 1)], out_shape=a.shape, transform=tr, fill=0, all_touched=False).astype(bool)
                                  for k, geom in zip(shapes, g)}
            ok = a > -2000
            return date, {k: (float(a[m & ok].sum()) * 1e-4, int((m & ok).sum())) for k, m in masks[key].items()}
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(one, todo))
        acc = {}
        for date, d in res:
            for k, (sm, ct) in d.items():
                a0 = acc.setdefault((date, k), [0.0, 0])
                a0[0] += sm; a0[1] += ct
        new = pd.Series({k: v[0] / v[1] if v[1] else np.nan for k, v in acc.items()}).unstack()
        have = new if have is None else pd.concat([have, new]).groupby(level=0).last()
        have = have.sort_index(); have.to_parquet(path)
    return have


def zone_rain(govs, zones: list[dict]) -> pd.DataFrame:
    """October–April ERA5 totals per zone (overlap-weighted over its governorates), labelled by season year."""
    cells = {k: L.era5_cell(k) for k in L.A.era5_cells}
    idx = next(iter(cells.values())).index
    for d in cells.values():
        idx = idx.intersection(d.index)
    out = {}
    for z in zones:
        poly = govs[govs.ADM2_EN.isin(z["governorates"])].union_all()
        w = {k: v for k, v in L.overlap_weights(poly, L.A.era5_cells, 0.25).items() if v > 0}
        tw = sum(w.values())
        daily = sum(cells[k].loc[idx, "pr"] * (v / tw) for k, v in w.items())
        m = daily.resample("MS").agg(["sum", "count"])
        mon = m["sum"].where(m["count"] >= 0.9 * m.index.days_in_month).dropna()
        out[z["name"]] = L.seasonal(mon, L.WET)
    return pd.DataFrame(out)


def pct_of_mean(s: pd.Series, lo: int = 1991, hi: int = 2020) -> pd.Series:
    return 100 * (s / s.loc[lo:hi].mean() - 1)


def spring_ndvi(d: pd.DataFrame, lo: int = 2001, hi: int = 2020) -> pd.DataFrame:
    """Spring greenness per zone and year: each composite as a percentage above or below the lo–hi mean of
    the same composite (so a missing composite does not bias the year), averaged over the year's composites."""
    d = d[d.index.year >= lo]
    ref = d[d.index.year <= hi]
    clim = ref.groupby(ref.index.dayofyear).mean()
    an = 100 * (d / clim.reindex(d.index.dayofyear).values - 1)
    return an.groupby(an.index.year).mean()


def pcbs_olives() -> pd.DataFrame:
    """PCBS Olive Presses Survey: olives pressed (t) by governorate and harvest year."""
    d = pd.read_csv(DATA / "pcbs_olive_presses.csv")
    return d.pivot_table(index="harvest_year", columns="governorate", values="olives_pressed_t", aggfunc="first")


def _detr(s: pd.Series) -> pd.Series:
    t = s.index.values.astype(float)
    return s - np.polyval(np.polyfit(t, s.values, 1), t)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def by_phase(v: pd.Series, phase: pd.Series) -> list[dict]:
    rows = []
    for p in PHASES:
        x = v[phase.reindex(v.index) == p].dropna()
        rows.append(dict(phase=p, n=len(x), mean=float(x.mean()) if len(x) else np.nan, median=float(x.median()) if len(x) else np.nan,
                         lo=float(x.min()) if len(x) else np.nan, hi=float(x.max()) if len(x) else np.nan, above=int((x > 0).sum()),
                         years=[int(y) for y in x.index]))
    return rows


def analyse(spec: dict, a: dict) -> dict:
    L.set_area(spec)            # when levant_deep_dive runs as __main__, this import is a second copy of it
    ag = spec["agri"]
    _, govs = L.area_polygons()
    djf = a["djf"]                                       # DJF Niño3.4 (pinned), by season year
    crop_phase = pd.Series(L.phase_of(djf.values), index=djf.index + 1)   # by crop year
    crop_nino = pd.Series(djf.values, index=djf.index + 1)
    rain_wb = pct_of_mean(a["tot"]["ERA5"])
    crop_rain = pd.Series(rain_wb.values, index=rain_wb.index + 1)

    # Zone rainfall: same El Niño signal everywhere?
    zr = zone_rain(govs, ag["zones"])
    zrows = []
    for z in zr:
        s = zr[z].loc[L.SPLIT:2025]
        r, p, n = L.corr(djf.reindex(s.index), s)
        t = L.thirds(s); en = djf.reindex(s.index) >= L.ENSO_THRESH
        zrows.append(dict(zone=z, r=r, p=p, n=n, en_n=int(en.sum()), en_wet=int((t[en] == "wettest").sum()),
                          en_dry=int((t[en] == "driest").sum()), en_pct=float(pct_of_mean(zr[z])[en[en].index].mean())))
    zc = zr.loc[L.SPLIT:2025].corr()
    z_min_r = float(zc.values[np.triu_indices(len(zc), 1)].min())

    # National field crops (FAOSTAT, official yields only)
    fao = faostat()
    fy = {i: detrended_pct(fao_series(fao, i, official=True)) for i in FIELD}
    field = pd.DataFrame(fy).mean(axis=1, skipna=False).dropna()
    field_r = L.corr(crop_rain.reindex(field.index), field)
    field_e = L.corr(crop_nino.reindex(field.index), field)
    field_ph = by_phase(field, crop_phase)
    pulses = {i: detrended_pct(fao_series(fao, i, official=True)) for i in PULSES}

    # Olives: national production, alternate bearing out
    oprod = fao_series(fao, "Olives", "Production")
    oflag = fao[(fao.item == "Olives") & (fao.element == "Production")].set_index("year").flag
    oab = olive_bearing(oprod)
    ol_r = L.corr(crop_rain.reindex(oab.index), oab)
    ol_e = L.corr(crop_nino.reindex(oab.index), oab)
    ol_ph = by_phase(oab, crop_phase)
    ol_ac = float(np.corrcoef(np.log(oprod).values[1:], np.log(oprod).values[:-1])[0, 1])

    # Spring vegetation (VCI), cropland and all land
    veg = {}
    for tag in ("crop", "land"):
        s = spring_vci(vhp(tag))
        veg[tag] = dict(s=s, r_rain=L.corr(crop_rain.reindex(s.index), s), r_nino=L.corr(crop_nino.reindex(s.index), s),
                        ph=by_phase(s - s.median(), crop_phase))

    # By zone: spring greenness (MODIS), olives pressed (PCBS) and where the cereals are (2021 census)
    ndvi = spring_ndvi(modis_zone_ndvi(govs, ag["zones"]))
    olv = pcbs_olives()
    area = pd.read_csv(DATA / "pcbs_cereal_area_2021.csv").groupby("governorate").area_dunum.sum()
    wb_area = float(area["West Bank total"])
    olv_pal = olv["Palestine total"].loc[2003:2019]
    zones = []
    for z in ag["zones"]:
        nm = z["name"]
        zrain = pd.Series(pct_of_mean(zr[nm]).values, index=zr[nm].index + 1)
        v = ndvi[nm].dropna()
        zz = dict(name=nm, ndvi=v, ndvi_r=L.corr(zrain.reindex(v.index), v), ndvi_e=L.corr(crop_nino.reindex(v.index), v),
                  ndvi_rd=L.corr(_detr(zrain.reindex(v.index).dropna()), _detr(v.loc[zrain.reindex(v.index).dropna().index])),
                  ndvi_ed=L.corr(crop_nino.reindex(v.index), _detr(v)), ndvi_ph=by_phase(v, crop_phase), ndvi_ph_d=by_phase(_detr(v), crop_phase),
                  ndvi_trend=float(np.polyfit(v.index, v.values, 1)[0] * 10), zrain=zrain,
                  cereal_share=float(sum(area.get(g, 0) for g in z.get("pcbs_area", [])) / wb_area))
        if z.get("pcbs_olive"):
            o = olv[z["pcbs_olive"]].loc[2003:2019].sum(axis=1, min_count=len(z["pcbs_olive"])).dropna()
            ab = olive_bearing(o)
            keep = ab.index != 2019
            zz.update(olive_rho=float(stats.spearmanr(zrain.reindex(ab.index), ab)[0]),
                      olive_r_no2019=float(stats.pearsonr(zrain.reindex(ab.index)[keep], ab[keep])[0]))
            zz.update(olive=o, olive_ab=ab, olive_r=L.corr(zrain.reindex(ab.index), ab), olive_e=L.corr(crop_nino.reindex(ab.index), ab),
                      olive_ph=by_phase(ab, crop_phase), olive_share=float((o / olv_pal.reindex(o.index)).mean()))
        zones.append(zz)
    wbz = ndvi["West Bank"].dropna()
    ndvi_wb = dict(s=wbz, r=L.corr(crop_rain.reindex(wbz.index), wbz), e=L.corr(crop_nino.reindex(wbz.index), wbz), ph=by_phase(wbz, crop_phase))
    olive_wb_share = float((olv["West Bank total"].loc[2003:2019] / olv_pal).mean())

    # Strong El Niño crop years (DJF ≥ 1.5) and the dry and wet extremes, for the table
    strong = [int(y) + 1 for y in djf[djf >= 1.5].index if y + 1 >= 1983]
    rows = []
    for y in sorted(set(strong)):
        rows.append(dict(year=y, nino=float(crop_nino.get(y, np.nan)), rain=float(crop_rain.get(y, np.nan)),
                         field=float(field.get(y, np.nan)), olive=float(oab.get(y, np.nan)),
                         vci=float(veg["crop"]["s"].get(y, np.nan)), vci_land=float(veg["land"]["s"].get(y, np.nan))))

    L_out = L.OUT
    fig_by_phase(field, oab, veg["crop"]["s"], crop_phase, L_out / "agri_by_phase.png")
    fig_field_rain(field, crop_rain, crop_phase, L_out / "agri_field_rain.png")
    fig_olives(oprod, oflag, crop_phase, L_out / "agri_olives.png")
    fig_vci(veg, crop_phase, L_out / "agri_vci.png")
    fig_zone_ndvi(zones, crop_phase, L_out / "agri_zone_ndvi.png")
    fig_zone_olives(zones, crop_phase, L_out / "agri_zone_olives.png")
    return dict(zones=zones, ndvi_wb=ndvi_wb, olive_wb_share=olive_wb_share, wb_cereal_share=wb_area / float(area["Palestine total"]),
                zr=zr, zrows=zrows, z_min_r=z_min_r, field=field, field_r=field_r, field_e=field_e, field_ph=field_ph,
                fy=fy, pulses=pulses, oprod=oprod, oab=oab, ol_r=ol_r, ol_e=ol_e, ol_ph=ol_ph, ol_ac=ol_ac, veg=veg,
                strong=rows, crop_rain=crop_rain, crop_nino=crop_nino, crop_phase=crop_phase)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _strip(ax, v: pd.Series, phase: pd.Series, label_years=(), unit="%", zero=0.0):
    for i, p in enumerate(PHASES):
        x = v[phase.reindex(v.index) == p].dropna()
        jit = (np.arange(len(x)) % 5 - 2) * 0.07
        ax.plot(np.full(len(x), i) + jit, x.values, "o", color=PCOL[p], ms=7, mec="white", mew=0.8, ls="")
        if len(x):
            ax.plot([i - 0.28, i + 0.28], [x.median()] * 2, color=L.C_TEXT, lw=1.6)
            ax.text(i + 0.32, x.median(), f"median {x.median():+.0f}{unit}" if unit == "%" else f"median {x.median():.0f}",
                    va="center", fontsize=7.5, color=L.C_MUTED)
        for y, xx, j in zip(x.index, x.values, jit):
            if y in label_years:
                ax.annotate(str(y), (i + j, xx), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=L.C_TEXT)
    ax.axhline(zero, color=L.C_MUTED, lw=0.8)
    ax.set_xticks(range(3), [f"{p}\n(n = {int((phase.reindex(v.dropna().index) == p).sum())})" for p in PHASES], fontsize=8.5)
    ax.set_xlim(-0.6, 2.9)
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)


def fig_by_phase(field: pd.Series, oab: pd.Series, vci: pd.Series, phase: pd.Series, out: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9), dpi=150)
    _strip(axs[0], field, phase, label_years=(1999, 2008, 2009, 1998, 2003, 2016))
    axs[0].set_title("Wheat and barley yield\n(% above or below trend)", fontsize=9.5, color=L.C_TEXT, loc="left")
    _strip(axs[1], oab, phase, label_years=(1998, 2016, 2019))
    axs[1].set_title("Olive crop, alternate bearing taken out\n(% above or below what last year predicts)", fontsize=9.5, color=L.C_TEXT, loc="left")
    _strip(axs[2], vci, phase, label_years=(1983, 1998, 2016, 2024, 1999, 2000, 2025), unit="", zero=50)
    axs[2].set_title("Spring vegetation condition, cropland\n(VCI, March–April; 50 = mid-range)", fontsize=9.5, color=L.C_TEXT, loc="left")
    axs[2].set_ylim(0, 100)
    fig.suptitle("West Bank crop years by the ENSO phase of the winter before the harvest", fontsize=10.5, color=L.C_TEXT, x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    L._save(fig, out)


def fig_field_rain(field: pd.Series, rain: pd.Series, phase: pd.Series, out: Path) -> None:
    d = pd.DataFrame({"f": field, "r": rain.reindex(field.index), "p": phase.reindex(field.index)}).dropna()
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
    for p in PHASES:
        x = d[d.p == p]
        ax.plot(x.r, x.f, "o", color=PCOL[p], ms=7, mec="white", mew=0.8, ls="", label=p)
    for y, r_ in d.iterrows():
        ax.annotate(str(y), (r_.r, r_.f), xytext=(5, 3), textcoords="offset points", fontsize=7, color=L.C_MUTED)
    b = np.polyfit(d.r, d.f, 1); xx = np.linspace(d.r.min(), d.r.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=L.C_MUTED, lw=1, ls=(0, (4, 3)))
    ax.axhline(0, color=L.C_MUTED, lw=0.8); ax.axvline(0, color=L.C_MUTED, lw=0.8)
    ax.set_xlabel("October–April rainfall before the harvest, % above or below the 1991–2020 mean (ERA5, West Bank)", fontsize=8.5, color=L.C_MUTED)
    ax.set_ylabel("Wheat and barley yield, % above or below trend", fontsize=8.5, color=L.C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Wheat and barley yields against the winter's rain (Palestine, 1994–2021)", fontsize=10, color=L.C_TEXT, loc="left")
    ax.tick_params(labelsize=8.5)
    ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    L._save(fig, out)


def fig_olives(prod: pd.Series, flag: pd.Series, phase: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=150)
    for y, v in prod.items():
        p = phase.get(y, "Neutral")
        off = flag.get(y, "A") != "A"
        ax.bar(y, v / 1000, color="white" if off else PCOL[p], edgecolor=PCOL[p], lw=1.2, width=0.8)
    ax.set_ylabel("thousand tonnes of olives", fontsize=8.5, color=L.C_MUTED)
    ax.set_title("Olive production, West Bank and Gaza (FAOSTAT), coloured by the ENSO phase of the winter before the harvest",
                 fontsize=9.5, color=L.C_TEXT, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PHASES] + [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=L.C_MUTED)]
    ax.legend(handles, PHASES + ["hollow: FAO estimate, not an official figure"], frameon=False, fontsize=7.5, ncol=4, loc="upper left")
    ax.set_ylim(0, prod.max() / 1000 * 1.18)
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    L._save(fig, out)


def fig_zone_ndvi(zones: list[dict], phase: pd.Series, out: Path) -> None:
    fig, axs = plt.subplots(len(zones), 1, figsize=(9.6, 1.9 * len(zones) + 0.6), dpi=150, sharex=True, sharey=True)
    for ax, z in zip(axs, zones):
        v = z["ndvi"]
        for y, x in v.items():
            ax.bar(y, x, color=PCOL[phase.get(y, "Neutral")], width=0.8)
        ax.axhline(0, color=L.C_MUTED, lw=0.8)
        e = _ph(z["ndvi_ph"], "El Niño")
        ax.set_title(f'{z["name"]}: El Niño years above normal {e["above"]} of {e["n"]}; r with the zone\'s rain {z["ndvi_r"][0]:+.2f}',
                     fontsize=9, color=L.C_TEXT, loc="left")
        ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
        edd._style_ax(ax)
    axs[len(zones) // 2].set_ylabel("spring NDVI, % above or below\nthe 2001–2020 mean", fontsize=8.5, color=L.C_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PHASES]
    axs[0].legend(handles, [f"{p} winter" for p in PHASES], frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.42))
    fig.tight_layout()
    L._save(fig, out)


def fig_zone_olives(zones: list[dict], phase: pd.Series, out: Path) -> None:
    zs = [z for z in zones if "olive_ab" in z]
    fig, axs = plt.subplots(1, len(zs), figsize=(4.8 * len(zs), 3.9), dpi=150, sharey=True)
    for ax, z in zip(np.atleast_1d(axs), zs):
        ab = z["olive_ab"]; r = z["zrain"].reindex(ab.index)
        for p in PHASES:
            m = phase.reindex(ab.index) == p
            ax.plot(r[m], ab[m], "o", color=PCOL[p], ms=8, mec="white", mew=0.8, ls="", label=p)
        for y in ab.index:
            ax.annotate(str(y), (r[y], ab[y]), xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=L.C_MUTED)
        ax.axhline(0, color=L.C_MUTED, lw=0.8); ax.axvline(0, color=L.C_MUTED, lw=0.8)
        ax.set_title(f'{z["name"]}\nr with rain {z["olive_r"][0]:+.2f}, with Niño3.4 {z["olive_e"][0]:+.2f} ({z["olive_r"][2]} years)', fontsize=9, color=L.C_TEXT, loc="left")
        ax.set_xlabel("Oct–Apr rain before the harvest, % vs 1991–2020 mean (ERA5)", fontsize=8, color=L.C_MUTED)
        ax.grid(color="#eceff0", lw=0.8); ax.set_axisbelow(True)
        edd._style_ax(ax)
    np.atleast_1d(axs)[0].set_ylabel("olives pressed, % above or below\nwhat the previous crop predicts", fontsize=8.5, color=L.C_MUTED)
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    L._save(fig, out)


def fig_vci(veg: dict, phase: pd.Series, out: Path) -> None:
    s, sl = veg["crop"]["s"], veg["land"]["s"]
    fig, ax = plt.subplots(figsize=(9.6, 3.5), dpi=150)
    ax.axhspan(0, 35, color="#f6eeee", zorder=0)
    ax.text(s.index.max() + 1.3, 17, "below 35:\npoor", fontsize=7.5, color=L.C_MUTED, va="center", ha="left")
    for y, v in s.items():
        ax.bar(y, v, color=PCOL[phase.get(y, "Neutral")], width=0.8)
    dots, = ax.plot(sl.index, sl.values, "o", color=L.C_TEXT, ms=3.2, ls="")
    ax.axhline(50, color=L.C_MUTED, lw=0.8)
    ax.set_ylim(0, 100); ax.set_xlim(s.index.min() - 1, s.index.max() + 1)
    ax.set_ylabel("VCI, March–April mean", fontsize=8.5, color=L.C_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in PHASES]
    ax.legend(handles + [dots], [f"cropland, {p} winter" for p in PHASES] + ["all land, incl. rangeland"], frameon=False, fontsize=7.5, ncol=4, loc="upper left")
    ax.set_title("Spring vegetation condition in the West Bank (NOAA STAR, 4 km)", fontsize=10, color=L.C_TEXT, loc="left")
    ax.grid(axis="y", color="#eceff0", lw=0.8); ax.set_axisbelow(True)
    edd._style_ax(ax)
    L._save(fig, out)


# --------------------------------------------------------------------------- #
# Page section
# --------------------------------------------------------------------------- #
def _ph(rows: list[dict], p: str) -> dict:
    return next(r for r in rows if r["phase"] == p)


def _nw(n: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


def _signed(v: float, unit: str = "%") -> str:
    if not np.isfinite(v):
        return "—"
    return f"0{unit}" if round(abs(v)) == 0 else f"{'+' if v > 0 else '−'}{abs(v):.0f}{unit}"


def render(spec: dict, a: dict) -> str:
    ag, g = spec["agri"], a["agri"]
    T = spec.get("titles", {})
    o = [f'<h2>{html.escape(T.get("agri", "6. Farming"))}</h2>', ag.get("intro_html", "")]
    zr = {r["zone"]: r for r in g["zrows"]}
    zd = {z["name"]: z for z in g["zones"]}

    # The three zones
    o.append('<div style="overflow-x:auto"><table><thead><tr><th>Zone</th><th>Governorates</th><th>What it grows</th>'
             '<th class="num">Share of West Bank wheat and barley area</th><th class="num">Share of olives pressed</th>'
             '<th class="num">Rain vs Niño3.4, r</th><th class="num">El Niño winters in the wettest third</th>'
             '<th class="num">Spring greenness after El Niño winters</th></tr></thead><tbody>')
    for z in ag["zones"]:
        r, d = zr[z["name"]], zd[z["name"]]
        e = _ph(d["ndvi_ph"], "El Niño")
        osh = f'{100 * d["olive_share"]:.0f}%' if "olive_share" in d else "—"
        o.append(f'<tr><td><strong>{html.escape(z["name"])}</strong></td><td class="small">{html.escape(", ".join(z.get("governorates_label", z["governorates"])))}</td>'
                 f'<td class="small">{z["grows_html"]}</td><td class="num">{100 * d["cereal_share"]:.0f}%</td><td class="num">{osh}</td>'
                 f'<td class="num">{L._r(r["r"])}</td><td class="num">{r["en_wet"]} of {r["en_n"]}</td>'
                 f'<td class="num">{e["above"]} of {e["n"]} above normal<br><span class="small">median {_signed(e["median"])}</span></td></tr>')
    o.append('</tbody></table></div>')
    o.append('<p class="small">Wheat and barley area: PCBS Agricultural Census 2021, by governorate (Tubas includes the northern valleys). '
             f'Olives pressed: PCBS Olive Presses Survey, share of all olives pressed in Palestine, 2003–2019 mean (PCBS merges Jenin and Tubas; '
             f'the West Bank as a whole pressed {100 * g["olive_wb_share"]:.0f}%; Jericho has no presses). Rain: October–April ERA5 over each '
             f'zone\'s governorates, {L.SPLIT}/{str(L.SPLIT + 1)[2:]}–2025/26, against December–February Niño3.4 (pinned series). Spring greenness: '
             'MODIS NDVI, February–April, 2001–2026, against the 2001–2020 normal (see below). The three zones\' winters move together (r ≥ '
             f'{g["z_min_r"]:.2f} between any two) and carry the same El Niño tilt; the zones differ in what they grow. ERA5\'s 25 km cells smooth '
             f'the West Bank\'s steep rain gradient (its Jordan Valley mean is about {g["zr"].loc[1991:2020, "Jordan Valley"].mean():.0f} mm, against '
             '100–200 mm at the valley floor), so only anomalies are used.</p>')
    o.append(ag.get("zones_after_html", ""))

    # Spring greenness by zone
    zs = [zd[z["name"]] for z in ag["zones"]]
    by = lambda k, p: _ph(zd[k]["ndvi_ph"], p)
    names = [z["name"] for z in ag["zones"]]
    pr = {z["name"]: z.get("prose", z["name"]) for z in ag["zones"]}
    o.append('<h3>Spring greenness, zone by zone</h3>')
    o.append('<p>The one yearly measure available for every zone is satellite greenness. In the spring after an El Niño winter, it was above '
             'normal in ' + ", ".join(f'{by(k, "El Niño")["above"]} of {by(k, "El Niño")["n"]} years in the {html.escape(pr[k])}' for k in names)
             + '; after La Niña winters, in ' + ", ".join(f'{by(k, "La Niña")["above"]} of {by(k, "La Niña")["n"]}' for k in names) + '. '
             'The drier the zone, the larger the swing: the median El Niño spring was '
             + ", ".join(f'{_signed(by(k, "El Niño")["median"])} in the {html.escape(pr[k])}' for k in names) + '. '
             + ag.get("ndvi_html", "") + '</p>')
    o.append('<figure><img src="agri_zone_ndvi.png" alt="Spring greenness by zone and year"><figcaption>MODIS Terra 16-day NDVI at 250 m '
             '(MOD13Q1, collection 6.1), mean over each zone\'s governorates for the five composites from 2 February to 22 April; each composite '
             'is compared with its own 2001–2020 mean and the year\'s value is the average of those percentages (2023 and 2026 miss one and '
             'two composites). Colour: ENSO phase of the winter before. Correlations with the zone\'s rain and with Niño3.4: '
             + "; ".join(f'{html.escape(z["name"])} r = {L._r(z["ndvi_r"][0])} and {L._r(z["ndvi_e"][0])} (p {L._p(z["ndvi_e"][1])})' for z in zs)
             + f' ({len(zs[0]["ndvi"])} springs). The Jordan Valley\'s governorates take in the dry eastern slopes as well as irrigated land, so its '
             'swings are mostly rangeland. Greenness has risen since 2001 in the Highlands and the Jordan Valley ('
             + ", ".join(f'{_signed(z["ndvi_trend"])} a decade' for z in zs[1:]) + '). With the trend removed the correlations are much the same, and '
             'El Niño springs above normal number ' + ", ".join(f'{_ph(z["ndvi_ph_d"], "El Niño")["above"]}' for z in zs)
             + f' of {_ph(zs[0]["ndvi_ph_d"], "El Niño")["n"]} (La Niña: ' + ", ".join(f'{_ph(z["ndvi_ph_d"], "La Niña")["above"]}' for z in zs)
             + f' of {_ph(zs[0]["ndvi_ph_d"], "La Niña")["n"]}), in the order above.'
             '</figcaption></figure>')

    # National: cereals, olives, long vegetation record, by phase
    fe, fl, fn = (_ph(g["field_ph"], p) for p in ("El Niño", "La Niña", "Neutral"))
    ve, vl = _ph(g["veg"]["crop"]["ph"], "El Niño"), _ph(g["veg"]["crop"]["ph"], "La Niña")
    le_, ll_ = _ph(g["veg"]["land"]["ph"], "El Niño"), _ph(g["veg"]["land"]["ph"], "La Niña")
    oe = _ph(g["ol_ph"], "El Niño")
    worst = g["field"].nsmallest(4)
    wph = [g["crop_phase"].get(y, "") for y in worst.index]
    worst_txt = ", ".join(f"{y} ({_signed(v)})" for y, v in worst.items())
    o.append('<h3>Harvests after El Niño winters</h3>')
    o.append(f'<p>Cereal yields are published only for Palestine as a whole, but the West Bank holds {100 * g["wb_cereal_share"]:.0f}% of its wheat and '
             f'barley area and presses {100 * g["olive_wb_share"]:.0f}% of its olives. Rainfed wheat and barley yields were above trend in '
             f'{fe["above"]} of {fe["n"]} crop years after El Niño winters, against {fl["above"]} of {fl["n"]} after La Niña winters. The '
             f'{_nw(len(worst))} worst harvests since 1994, {worst_txt}, '
             + ('all followed La Niña winters' if all(p == "La Niña" for p in wph) else f'include {sum(p == "La Niña" for p in wph)} after La Niña winters')
             + f'. El Niño years were not bumper years: their median, {_signed(fe["median"])}, is close to that of neutral years ({_signed(fn["median"])}). '
             'What an El Niño winter has done is make a bad cereal year less likely. The longer vegetation record, which also covers pasture, '
             f'tells the same story: spring vegetation was above its median in {ve["above"]} of {ve["n"]} El Niño years on cropland and '
             f'{le_["above"]} of {le_["n"]} on all land, against {vl["above"]} of {vl["n"]} and {ll_["above"]} of {ll_["n"]} after La Niña winters. '
             f'Olives show no El Niño signal ({oe["above"]} of {oe["n"]} El Niño years above what the previous crop predicts).</p>')
    o.append(ag.get("phase_html", ""))
    o.append('<figure><img src="agri_by_phase.png" alt="Cereal yields, olive crop and spring vegetation by ENSO phase"><figcaption>Each dot is one crop '
             'year, coloured by the ENSO phase of the winter before the harvest (December–February Niño3.4, ±0.5 °C); bars are medians. '
             f'Left: wheat and barley, the mean of their yields\' percentages above or below a log-linear trend, official FAOSTAT figures only '
             f'({min(g["field"].index)}–{max(g["field"].index)}, {len(g["field"])} years). Middle: olive production (FAOSTAT) after taking out '
             f'alternate bearing. Right: the NOAA STAR Vegetation Condition Index over West Bank cropland, mean of March and April '
             f'({min(g["veg"]["crop"]["s"].index)}–{max(g["veg"]["crop"]["s"].index)}; no data for 2004).</figcaption></figure>')
    o.append('<table><thead><tr><th>Measure</th><th>Years</th><th class="num">El Niño</th><th class="num">Neutral</th><th class="num">La Niña</th>'
             '<th class="num">r with rain</th><th class="num">r with Niño3.4</th></tr></thead><tbody>')

    def row(name, yrs, ph, rr, re_, base="above trend", unit="%"):
        cells = "".join(f'<td class="num">{_ph(ph, p)["above"]} of {_ph(ph, p)["n"]}<br><span class="small">median {_signed(_ph(ph, p)["median"], unit)}</span></td>'
                        for p in PHASES)
        return (f'<tr><td>{name}<br><span class="small">years {base}</span></td><td>{yrs}</td>{cells}'
                f'<td class="num">{L._r(rr[0])}<br><span class="small">p {L._p(rr[1])}</span></td><td class="num">{L._r(re_[0])}<br><span class="small">p {L._p(re_[1])}</span></td></tr>')
    fi, oi = g["field"].index, g["oab"].index
    o.append(row("Wheat and barley yield", f"{fi.min()}–{fi.max()}", g["field_ph"], g["field_r"], g["field_e"]))
    o.append(row("Olive crop, alternate bearing out", f"{oi.min()}–{oi.max()}", g["ol_ph"], g["ol_r"], g["ol_e"], base="above what the previous crop predicts"))
    for t, nm in (("crop", "Spring VCI, cropland"), ("land", "Spring VCI, all land")):
        v = g["veg"][t]
        o.append(row(nm, f'{v["s"].index.min()}–{v["s"].index.max()}', v["ph"], v["r_rain"], v["r_nino"], base="above the record's median", unit=" pts"))
    w = g["ndvi_wb"]
    o.append(row("Spring NDVI, whole West Bank", f'{w["s"].index.min()}–{w["s"].index.max()}', w["ph"], w["r"], w["e"], base="above the 2001–2020 normal"))
    o.append('</tbody></table>')
    o.append('<p class="small">“r with rain”: against the October–April ERA5 total over the West Bank before the harvest; “r with Niño3.4”: against '
             'December–February Niño3.4 of that winter. Medians are percentages above or below trend or normal, or VCI points above or below the '
             'record\'s median. All of these years fall after 1979, inside the period in which the El Niño link to West Bank rain exists.</p>')

    # Cereals
    o.append('<h3>Rainfed cereals</h3>' + ag.get("field_html", ""))
    o.append('<figure><img src="agri_field_rain.png" alt="Wheat and barley yield against rainfall"><figcaption>Wheat and barley yields (official '
             'FAOSTAT figures for Palestine; percentage above or below trend) against the rainfall of the winter before the harvest. '
             f'r = {L._r(g["field_r"][0])} (p {L._p(g["field_r"][1])}, {g["field_r"][2]} years). FAOSTAT repeats its 2013 figures as 2014 and gives '
             'estimated or imputed figures for 2009, 2017 and (barley) 2022; those years are left out.</figcaption></figure>')

    # Olives
    oz = [z for z in zs if "olive_ab" in z]
    o.append('<h3>Olives</h3>' + ag.get("olive_html", ""))
    o.append('<figure><img src="agri_olives.png" alt="Olive production by year"><figcaption>Olive production (FAOSTAT, Palestine; for 2017–2019 '
             'it equals the olives pressed in the PCBS survey). Big and small crops alternate (r between one year and the next: '
             f'{g["ol_ac"]:+.2f}); the colour is the ENSO phase of the winter before the harvest. FAOSTAT\'s harvested area halves from 2010 '
             'without a matching change in production, so production is used rather than yield.</figcaption></figure>')
    o.append('<figure><img src="agri_zone_olives.png" alt="Olives pressed by zone against rainfall"><figcaption>Olives pressed in each zone\'s '
             'governorates (PCBS Olive Presses Survey, 2003–2019), as a percentage above or below what the previous year\'s crop predicts, '
             'against the zone\'s October–April rain. The survey counts olives brought to presses, not table olives or olives pressed at home. '
             + " ".join(f'{html.escape(z["name"])}: Spearman ρ = {z["olive_rho"]:+.2f}; Pearson r without 2019 = {z["olive_r_no2019"]:+.2f}.' for z in oz)
             + '</figcaption></figure>')

    # Long vegetation record
    o.append('<h3>Spring vegetation since 1982</h3>' + ag.get("veg_html", ""))
    o.append('<figure><img src="agri_vci.png" alt="Spring vegetation condition by year"><figcaption>NOAA STAR Blended Vegetation Health, '
             'Vegetation Condition Index (VCI): where each week\'s NDVI sits between the lowest (0) and highest (100) on record for that week and '
             'place, averaged by NOAA over the West Bank. Bars: cropland; dots: all land, which adds the rangeland of the eastern slopes. '
             f'March–April mean; 2004 has a data gap. r with Niño3.4: cropland {L._r(g["veg"]["crop"]["r_nino"][0])}, all land '
             f'{L._r(g["veg"]["land"]["r_nino"][0])}.</figcaption></figure>')

    # Strong El Niño crop years
    o.append('<h3>Every strong El Niño crop year since 1983</h3>' + ag.get("strong_html", ""))
    o.append('<div style="overflow-x:auto"><table><thead><tr><th>Crop year</th><th class="num">Niño3.4 DJF</th><th class="num">Rain, Oct–Apr</th>'
             '<th class="num">Wheat and barley</th><th class="num">Olives, bearing out</th><th class="num">VCI cropland</th>'
             '<th class="num">VCI all land</th><th class="num">Spring NDVI</th><th>Note</th></tr></thead><tbody>')
    notes = ag.get("strong_notes", {})
    for r in g["strong"]:
        nd = w["s"].get(r["year"], np.nan)
        o.append(f'<tr><td>{r["year"]} <span class="small">(winter {r["year"] - 1}/{str(r["year"])[2:]})</span></td><td class="num">{r["nino"]:+.1f}</td>'
                 f'<td class="num">{_signed(r["rain"])}</td><td class="num">{_signed(r["field"])}</td><td class="num">{_signed(r["olive"])}</td>'
                 f'<td class="num">{r["vci"]:.0f}</td><td class="num">{r["vci_land"]:.0f}</td><td class="num">{_signed(nd)}</td>'
                 f'<td class="small">{notes.get(str(r["year"]), "")}</td></tr>')
    o.append('</tbody></table></div>')
    o.append('<p class="small">Niño3.4 ≥ +1.5 °C in December–February. Rain as a percentage above or below the 1991–2020 mean (ERA5, West Bank); '
             'yields and olives as in the table above (— = no official figure); VCI 50 is the middle of the record\'s range; spring NDVI for the '
             'whole West Bank against the 2001–2020 normal (from 2001).</p>')
    o.append(ag.get("after_html", ""))
    return "\n".join(o)
