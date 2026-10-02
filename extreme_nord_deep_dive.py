"""Cameroon, Extrême-Nord — ENSO deep dive (builder = "extreme_nord" in deep_dives/extreme-nord.toml).

Own builder, like Gaza's: one admin-1 region, and the question is not only "does El Niño dry the rains"
but what the 2026 El Niño has already done and what is left — the 2026 season (over), the Logone
floods (peak Oct–Nov), the hot season after the event (Mar–May 2027) and the 2027 rains. ERA5 is not
the spine here: over this region it lacks the post-1980s rainfall recovery that every gauge-based
record shows and gives 2026 a deficit the vegetation record does not support, so the rainfall
sections lead with gauge records (GPCC, CHIRPS) and show ERA5 as one voice among five.

    uv run python enso_deep_dive.py --only extreme-nord    # via the dispatcher
    uv run python extreme_nord_deep_dive.py                # standalone

Inputs (cached under cache/cmr/; re-fetched when stale, cache used when a source is unreachable):
GPCC 1° (NOAA PSL combined Full v2020 + Monitoring, 1891–2025; DWD first guess and monitoring for
2026), CHIRPS v3 (CHC, windowed COG reads) and v2 (IRI Data Library), IMERG late run / ERA5 /
FloodScan admin-level stats (team prod DB), ERA5 2 m temperature (CDS monthly means), JRC ASAP
indicators, Cadre Harmonisé via HAPI (team dev DB, ipc schema), the SEAS5 skill cube, NOAA Niño3.4.
"""
from __future__ import annotations

import html
import os
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import stats
from shapely.geometry import box

import enso_deep_dive as edd
import teleconnection_survey as ts

SLUG = "extreme-nord"
CACHE = Path("cache/cmr")
OUT = edd.OUT_DIR / SLUG
ISO3, PCODE = "CMR", "CM004"
BOX = (13.2, 9.8, 15.9, 13.3)                  # w, s, e: the region plus a margin
BASE = (2001, 2025)                            # common baseline for 2026 (IMERG starts 2001)
RAINS = [6, 7, 8, 9]                           # June–September
JAS = [7, 8, 9]
THR = edd.ENSO_THRESH
STRONG = 1.3                                   # "strong" for the analogue table: JAS Niño3.4 ≥ +1.3

C_EN, C_NEU, C_LN = edd.C_ELNINO, edd.C_NEUTRAL, edd.C_LANINA
C_TEXT, C_MUTED = edd.C_TEXT, edd.C_MUTED
SRC_COL = {"GPCC": "#1F5F96", "CHIRPS v3": "#C0782F", "CHIRPS v2": "#C0782F", "IMERG": "#8a4f7d", "ERA5": "#269777"}
PRODUCTS = ["GPCC", "CHIRPS v3", "CHIRPS v2", "IMERG", "ERA5"]
DEPTS = {"CM004001": "Diamaré", "CM004002": "Logone-et-Chari", "CM004003": "Mayo-Danay",
         "CM004004": "Mayo-Kani", "CM004005": "Mayo-Sava", "CM004006": "Mayo-Tsanaga"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _age_days(p: Path) -> float:
    return (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).total_seconds() / 86400 if p.exists() else 1e9


def _db(name: str, sql: str, stage: str = "prod", max_age: float = 1.0, **kw) -> pd.DataFrame:
    """A team-DB query cached as parquet; the cache is used when the DB is unreachable (it sits
    behind private endpoints: run `db-tunnel up` and export its env for a refresh)."""
    path = CACHE / f"{name}.parquet"
    if _age_days(path) < max_age:
        return pd.read_parquet(path)
    try:
        import ocha_stratus as stratus
        df = pd.read_sql(sql, stratus.get_engine(stage=stage), **kw)
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  ({name}: DB not reachable, using the cache from {pd.Timestamp(path.stat().st_mtime, unit='s'):%Y-%m-%d}: {str(e)[:80]})")
            return pd.read_parquet(path)
        raise


def region() -> gpd.GeoDataFrame:
    adm = edd.load_adm1(ISO3)
    return adm[adm.pcode == PCODE]


def depts() -> gpd.GeoDataFrame:
    p = CACHE / "adm2_CMR.parquet"
    if not p.exists():
        from ocha_stratus import codab
        codab.load_codab_from_blob("cmr", admin_level=2).to_crs("EPSG:4326").to_parquet(p)
    g = gpd.read_parquet(p)
    pc = next(c for c in g.columns if c.lower().startswith("adm2_pcode"))
    g = g.rename(columns={pc: "pcode"})
    g["pcode"] = g.pcode.str.upper()
    return g[g.pcode.isin(DEPTS)].assign(name=lambda d: d.pcode.map(DEPTS))


def db_series() -> dict[str, pd.DataFrame]:
    q = "SELECT pcode, adm_level, valid_date, mean FROM public.{t} WHERE iso3 = 'CMR' AND adm_level IN (1, 2) ORDER BY valid_date"
    im = _db("imerg_adm12", q.format(t="imerg"), parse_dates=["valid_date"])
    er = _db("era5_adm12", q.format(t="era5"), parse_dates=["valid_date"])
    fs = _db("floodscan_adm12", "SELECT pcode, adm_level, valid_date, band, mean, max, count FROM public.floodscan "
             "WHERE iso3 = 'CMR' AND adm_level IN (1, 2) AND band = 'SFED' ORDER BY valid_date", parse_dates=["valid_date"])
    return {"imerg": im, "era5": er, "floodscan": fs}


def imerg_daily(im: pd.DataFrame, pcode: str = PCODE) -> pd.Series:
    return im[im.pcode == pcode].set_index("valid_date")["mean"].sort_index()


def imerg_monthly(im: pd.DataFrame, pcode: str = PCODE) -> pd.Series:
    d = imerg_daily(im, pcode)
    m = d.resample("MS").sum(min_count=25)
    return m[m.index >= "2001-01-01"]          # 1998–2000 are partial in the DB


def era5_monthly(er: pd.DataFrame, pcode: str = PCODE) -> pd.Series:
    d = er[er.pcode == pcode].set_index("valid_date")["mean"].sort_index()
    return d * d.index.days_in_month           # mm/day → mm/month


def _cell_weights(lats, lons, res: float) -> dict[tuple[float, float], float]:
    geom = region().geometry.union_all()
    w = {}
    for la in lats:
        for lo in lons:
            a = geom.intersection(box(lo - res / 2, la - res / 2, lo + res / 2, la + res / 2)).area
            if a > 0:
                w[(float(la), float(lo))] = a
    return w


def gpcc_monthly() -> pd.Series:
    """GPCC Full Data v2020 (to 2019) + Monitoring v2022 (after), 1°, NOAA PSL; area-weighted over the region."""
    import xarray as xr
    path = CACHE / "gpcc_comb_1deg.nc"
    if not path.exists() or _age_days(path) > 60:
        w, s, e, n = BOX
        url = ("https://psl.noaa.gov/thredds/ncss/grid/Datasets/gpcc/combined/"
               "precip.comb.v2020to2019-v2020monitorafter.total.nc?var=precip"
               f"&north={n + 0.5}&south={s - 0.5}&west={w - 0.5}&east={e + 0.5}&temporal=all&accept=netcdf")
        try:
            r = requests.get(url, timeout=300); r.raise_for_status(); path.write_bytes(r.content)
        except Exception as ex:  # noqa: BLE001
            if not path.exists():
                raise
            print(f"  (GPCC refresh failed, using the cache: {ex})")
    with xr.open_dataset(path) as ds:
        wts = _cell_weights(ds.lat.values, ds.lon.values, 1.0)
        tot = sum(wts.values())
        s_ = sum(ds.precip.sel(lat=la, lon=lo).to_series() * (wt / tot) for (la, lo), wt in wts.items())
    s_.index = pd.DatetimeIndex(s_.index).to_period("M").to_timestamp()
    return s_.dropna()


def gpcc_2026(year: int = 2026) -> list[dict]:
    """DWD's GPCC first guess and monitoring products for the current year, area-weighted the same
    way, with the number of gauges inside the region's cells (the honest part of the story)."""
    import gzip
    import xarray as xr
    rows = []
    for kind, stem in [("first guess", "first_guess/{y}/first_guess_monthly_{y}_{m:02d}.nc"),
                       ("monitoring", "monitoring_v2022/{y}/monitoring_v2022_10_{y}_{m:02d}.nc")]:
        for m in RAINS:
            path = CACHE / Path(stem.format(y=year, m=m)).name
            if not path.exists():
                try:
                    r = requests.get("https://opendata.dwd.de/climate_environment/GPCC/" + stem.format(y=year, m=m) + ".gz", timeout=120)
                    if r.status_code != 200:
                        continue
                    path.write_bytes(gzip.decompress(r.content))
                except Exception:  # noqa: BLE001
                    continue
            with xr.open_dataset(path, decode_times=False) as ds:
                wts = _cell_weights(ds.lat.values[(ds.lat.values > 8) & (ds.lat.values < 15)],
                                    ds.lon.values[(ds.lon.values > 12) & (ds.lon.values < 17)], 1.0)
                tot = sum(wts.values())
                p = sum(float(ds["p"].sel(lat=la, lon=lo).values.squeeze()) * wt / tot for (la, lo), wt in wts.items())
                inside = [k for k, wt in wts.items() if wt / tot > 0.05]
                g = int(sum(float(ds["s"].sel(lat=la, lon=lo).values.squeeze()) for la, lo in inside)) if "s" in ds else None
            rows.append(dict(kind=kind, month=m, p=p, gauges=g))
    return rows


def chirps3_box() -> tuple[np.ndarray, pd.DatetimeIndex, object]:
    """CHIRPS v3 monthly (CHC Africa COGs), May–September of every year, windowed reads over BOX;
    cached and topped up with months published since."""
    import rasterio
    from affine import Affine
    from rasterio.windows import from_bounds
    path = CACHE / "chirps3_box.npz"
    have = np.load(path) if path.exists() else None
    done = set(pd.to_datetime(have["dates"]).strftime("%Y-%m")) if have is not None else set()
    now = pd.Timestamp.now()
    want = [(y, m) for y in range(1981, now.year + 1) for m in [5] + RAINS
            if f"{y}-{m:02d}" not in done and pd.Timestamp(y, m, 1) < now - pd.Timedelta(days=20)]
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")

    def one(ym):
        y, m = ym
        u = f"/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/chirps-v3.0.{y}.{m:02d}.tif"
        try:
            with rasterio.Env(**env), rasterio.open(u) as src:
                w = from_bounds(*BOX, src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w).astype("float32"); nod = src.nodata; tr = src.window_transform(w)
        except Exception:  # noqa: BLE001
            return None
        if nod is not None:
            a[a == nod] = np.nan
        a[a < 0] = np.nan
        return y, m, a, tr
    got = []
    if want:
        with ThreadPoolExecutor(12) as ex:
            got = [g for g in ex.map(one, want) if g is not None]
    if got:
        pr = np.stack([g[2] for g in got]); dates = pd.to_datetime([f"{g[0]}-{g[1]:02d}-01" for g in got])
        tr = np.array(got[0][3])[:6]
        if have is not None:
            assert np.allclose(have["transform"], tr), "CHIRPS window moved"
            pr = np.concatenate([have["pr"], pr]); dates = pd.to_datetime(have["dates"]).append(dates)
        o = np.argsort(dates.values)
        np.savez_compressed(path, pr=pr[o], dates=dates.values[o].astype("datetime64[D]"), transform=tr)
        have = np.load(path)
    return have["pr"], pd.to_datetime(have["dates"]), Affine(*have["transform"])


def chirps3_series(geom) -> pd.Series:
    pr, dates, tr = chirps3_box()
    m = rasterize([(geom, 1)], out_shape=pr.shape[1:], transform=tr, fill=0, dtype="uint8").astype(bool)
    return pd.Series(np.nanmean(pr[:, m], axis=1), index=dates)


def chirps2_monthly() -> pd.Series:
    """CHIRPS v2.0 monthly from the IRI Data Library (1981–), region mean."""
    import xarray as xr
    path = CACHE / "chirps_monthly.nc"
    if _age_days(path) > 20:
        w, s, e, n = BOX
        url = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
               f"X/{w}/{e}/RANGEEDGES/Y/{s}/{n}/RANGEEDGES/data.nc")
        try:
            r = requests.get(url, timeout=300); r.raise_for_status(); path.write_bytes(r.content)
        except Exception as ex:  # noqa: BLE001
            if not path.exists():
                raise
            print(f"  (CHIRPS v2 refresh failed, using the cache: {ex})")
    ds = xr.open_dataset(path, decode_times=False)
    t = pd.date_range("1981-01-01", periods=ds.sizes["T"], freq="MS")
    X, Y = ds.X.values, ds.Y.values[::-1]
    tr = from_origin(X[0] - 0.025, Y[0] + 0.025, 0.05, 0.05)
    m = rasterize([(region().geometry.union_all(), 1)], out_shape=(len(Y), len(X)), transform=tr, fill=0, dtype="uint8").astype(bool)
    P = ds.precipitation.values[:, ::-1, :]
    return pd.Series(np.nanmean(P[:, m], axis=1), index=t)


def era5_t2m() -> pd.Series:
    """ERA5 monthly-mean 2 m temperature (°C), region mean of the 0.25° cells touching it (CDS)."""
    import xarray as xr
    path = CACHE / "era5_t2m_monthly.nc"
    if _age_days(path) > 30:
        try:
            import cdsapi
            import yaml
            key = yaml.safe_load((Path.home() / ".cdsapirc").read_text())["key"]
            now = pd.Timestamp.now()
            tmp = path.with_suffix(".tmp.nc")
            cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key, quiet=True).retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {"product_type": ["monthly_averaged_reanalysis"], "variable": ["2m_temperature"],
                 "year": [str(y) for y in range(1981, now.year + 1)], "month": [f"{m:02d}" for m in range(1, 13)],
                 "time": ["00:00"], "area": [13.25, 13.25, 10.0, 15.75], "data_format": "netcdf", "download_format": "unarchived"},
                str(tmp))
            tmp.replace(path)
        except Exception as ex:  # noqa: BLE001
            if not path.exists():
                raise
            print(f"  (ERA5 temperature refresh failed, using the cache: {ex})")
    with xr.open_dataset(path) as ds:
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        lat, lon = ds.latitude.values, ds.longitude.values
        tr = from_origin(lon[0] - 0.125, lat[0] + 0.125, 0.25, 0.25)
        m = rasterize([(region().geometry.union_all(), 1)], out_shape=(len(lat), len(lon)), transform=tr, fill=0,
                      all_touched=True, dtype="uint8").astype(bool)
        s = pd.Series(np.nanmean(ds.t2m.values[:, m], axis=1) - 273.15, index=pd.to_datetime(ds[tdim].values)).sort_index()
    s.index = s.index.to_period("M").to_timestamp()
    return s


ASAP_IND = {"zfparc_crop": (240, 1, 1, 3), "zfparc_range": (240, 2, 1, 3), "spi3_crop": (40, 1, 1, 4)}   # ids as in ds-asap-trends


def asap() -> pd.DataFrame:
    """JRC ASAP per-unit dekadal indicators for Cameroon (asap0_id 37), as in ds-asap-trends."""
    path = CACHE / "asap_cmr.parquet"
    if _age_days(path) < 5:
        return pd.read_parquet(path)
    import io
    frames = []
    try:
        meta_ok = True
        for key, (var, cls, cset, sensor) in ASAP_IND.items():
            r = requests.get("https://agricultural-production-hotspots.ec.europa.eu/export/rum/export.php",
                             params=dict(gaul_level=1, country_id=37, variable_id=var, class_id=cls, classesset_id=cset, sensor_id=sensor), timeout=300)
            if not r.text.lstrip().startswith("country_id,"):
                meta_ok = False; break
            d = pd.read_csv(io.StringIO(r.text), dtype={"date": str}); d["key"] = key; frames.append(d)
        if meta_ok:
            pd.concat(frames).to_parquet(path)
    except Exception as ex:  # noqa: BLE001
        print(f"  (ASAP refresh failed, using the cache: {ex})")
    return pd.read_parquet(path)


def cadre_harmonise() -> pd.DataFrame:
    """CH results for Cameroon from the team's HAPI/HDX mirror (dev DB, ipc schema)."""
    return _db("ch_population_cmr", "SELECT * FROM ipc.population WHERE iso3 = 'CMR'", stage="dev", max_age=7)


def gpcc_gauges() -> pd.Series:
    """Mean number of gauges per June–September month in the 1° cells that make up the region
    (GPCC Full Data v2022, DWD; 1891–2020). Downloaded once: ~300 MB of decade files, kept as a box."""
    import gzip
    import xarray as xr
    path = CACHE / "gpcc_v2022_box.nc"
    if not path.exists():
        def one(d):
            u = f"https://opendata.dwd.de/climate_environment/GPCC/full_data_monthly_v2022/10/full_data_monthly_v2022_{d}_{d + 9}_10.nc.gz"
            r = requests.get(u, timeout=600); r.raise_for_status()
            fd, tmp = tempfile.mkstemp(suffix=".nc"); os.write(fd, gzip.decompress(r.content)); os.close(fd)
            with xr.open_dataset(tmp) as ds:
                sub = ds[["numgauge"]].sel(lat=slice(14, 9), lon=slice(13, 16)).load()
            os.remove(tmp)
            return sub
        with ThreadPoolExecutor(4) as ex:
            xr.concat(list(ex.map(one, range(1891, 2021, 10))), "time").to_netcdf(path)
    with xr.open_dataset(path) as ds:
        wts = _cell_weights(ds.lat.values, ds.lon.values, 1.0); tot = sum(wts.values())
        g = sum(ds["numgauge"].sel(lat=la, lon=lo).to_series() for (la, lo), wt in wts.items() if wt / tot > 0.05)
    g.index = pd.DatetimeIndex(g.index)
    g = g[g.index.month.isin(RAINS)]
    return g.groupby(g.index.year).mean()


def oni_table() -> pd.DataFrame:
    """CPC's official ONI (three-month running Niño3.4, ERSST v5), for the 2026 seasons that the
    pinned monthly series (which ends June 2026) does not reach."""
    path = CACHE / "oni.ascii.txt"
    if _age_days(path) > 5:
        try:
            r = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60)
            r.raise_for_status(); path.write_text(r.text)
        except Exception as ex:  # noqa: BLE001
            print(f"  (ONI refresh failed: {ex})")
    rows = [ln.split() for ln in path.read_text().splitlines()[1:] if ln.strip()]
    return pd.DataFrame([(r[0], int(r[1]), float(r[3])) for r in rows], columns=["seas", "year", "anom"])


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def seasonal(m: pd.Series, months: list[int], y0: int = 1891, y1: int = 2100) -> pd.Series:
    out = {}
    for y in range(y0, y1 + 1):
        k = [pd.Timestamp(y, x, 1) for x in months]
        if all(t in m.index and pd.notna(m[t]) for t in k):
            out[y] = float(m[k].sum())
    return pd.Series(out, dtype=float)


def detrend(s: pd.Series) -> pd.Series:
    return s - np.polyval(np.polyfit(s.index.values.astype(float), s.values, 1), s.index.values.astype(float))


def thirds(s: pd.Series) -> pd.Series:
    return pd.qcut(s, 3, labels=["dry", "mid", "wet"])


def counts(t: pd.Series) -> list[int]:
    return [int((t == k).sum()) for k in ("dry", "mid", "wet")]


def relationship(name: str, rains: pd.Series, nino: pd.Series, y0: int, y1: int) -> dict:
    s = rains.loc[y0:y1]
    c = s.index.intersection(nino.index)
    s, n = s[c], nino[c]
    r, p = stats.pearsonr(s, n)
    d = detrend(s)
    rd, pdd = stats.pearsonr(d, n)
    t = thirds(d)
    return dict(src=name, y0=int(c.min()), y1=int(c.max()), n=len(c), r=r, p=p, rd=rd, pd=pdd,
                trend=np.polyfit(c.values.astype(float), s.values, 1)[0] * 10,
                en=counts(t[n >= THR]), ln=counts(t[n <= -THR]), n_en=int((n >= THR).sum()), n_ln=int((n <= -THR).sum()),
                pct=s.rank(pct=True))


def running_corr(x: pd.Series, y: pd.Series, half: int = 15, min_n: int = 25) -> pd.Series:
    c = x.index.intersection(y.index); x, y = x[c], y[c]
    out = {}
    for t in c:
        w = (c >= t - half) & (c <= t + half)
        if w.sum() >= min_n and (t - half) >= c.min() and (t + half) <= c.max():
            out[t] = stats.pearsonr(x[w], y[w])[0]
    return pd.Series(out)


def phase(v: float) -> str:
    return "El Niño" if v >= THR else ("La Niña" if v <= -THR else "Neutral")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(spec: dict, indices: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    a: dict = {}
    reg = region(); geom = reg.geometry.union_all()
    # the survey's pinned Niño3.4 file (NOAA PSL, ERSST v5), read whole: load_indices() trims it to 1981
    pinned = ts._parse_psl((ts.CONFIG["cache_dir"] / "nino34.data").read_text())
    indices = pd.DataFrame({"nino34": pinned[pinned > -90]})
    nino = indices["nino34"]
    njas = edd.nino_series(indices, JAS, 0)              # 1950–
    nndj = edd.nino_series(indices, [11, 12, 1], 0)       # labelled by the November year
    a["nino_last"] = (nino.dropna().index[-1], float(nino.dropna().iloc[-1]))

    # rainfall records
    db = db_series()
    mon = {"GPCC": gpcc_monthly(), "CHIRPS v3": chirps3_series(geom), "CHIRPS v2": chirps2_monthly(),
           "IMERG": imerg_monthly(db["imerg"]), "ERA5": era5_monthly(db["era5"])}
    a["last_month"] = {k: v.dropna().index.max() for k, v in mon.items()}
    rains = {k: seasonal(v, RAINS) for k, v in mon.items()}

    # 1. the relationship, 1981–2025 (IMERG 2001–), detrended terciles
    rel = [relationship(k, rains[k], njas, 2001 if k == "IMERG" else 1981, 2025) for k in PRODUCTS]
    a["rel"] = rel
    # the longer gauge record: from 1950 (the pinned Niño3.4 series; GPCC's cells over the region hold
    # no gauge before 1927 and one or two until 1950, so earlier "data" is climatology or far interpolation)
    nlong = njas.loc[1950:2025]
    g = rains["GPCC"]
    a["gauges"] = gpcc_gauges()
    a["gpcc_long"] = relationship("GPCC", g, nlong, 1950, 2025)
    a["gpcc_periods"] = []
    for y0, y1 in [(1950, 1980), (1981, 2025)]:
        r_ = relationship("GPCC", g, nlong, y0, y1)
        a["gpcc_periods"].append(dict(y0=y0, y1=y1, n=r_["n"], rd=r_["rd"], pd=r_["pd"], en=r_["en"], n_en=r_["n_en"]))
    a["running"] = running_corr(detrend(g.loc[1950:2025]), nlong)
    # history figure frame (1950–2025: the pinned series' span)
    hist = pd.DataFrame({"gpcc": g.loc[1950:2025], "chirps": rains["CHIRPS v3"].loc[1981:2025], "nino": njas.loc[1950:2025]})
    hist["z"] = (hist.gpcc - hist.gpcc.mean()) / hist.gpcc.std()
    hist["zc"] = (hist.chirps - hist.chirps.mean()) / hist.chirps.std()
    hist["phase"] = hist.nino.map(phase)
    a["hist"] = hist
    # strong events: JAS Niño3.4 ≥ STRONG, with each record's percentile (100 = wettest)
    strong = []
    for y in nlong.index[(nlong >= STRONG)]:
        if y > 2025:
            continue
        row = dict(year=int(y), nino=float(nlong[y]))
        for k in ["GPCC", "CHIRPS v3", "IMERG", "ERA5"]:
            s = rains[k].loc[:2025]
            s = s.loc[1950:] if k == "GPCC" else s.loc[1981:]
            row[k] = float(s.rank(pct=True)[y] * 100) if y in s.index else None
        strong.append(row)
    a["strong"] = strong
    oni = oni_table()
    a["oni_2026"] = {r.seas: r.anom for r in oni[oni.year == 2026].itertuples()}
    a["oni_last"] = oni.iloc[-1].to_dict()

    # 2. the 2026 season, every product against 2001–2025
    rows = []
    for k in PRODUCTS:
        m = mon[k]
        r = dict(src=k, months={})
        for mm in [5] + RAINS:
            t = pd.Timestamp(2026, mm, 1)
            if t in m.index and pd.notna(m[t]):
                b = m[(m.index.month == mm) & (m.index.year >= BASE[0]) & (m.index.year <= BASE[1])].mean()
                r["months"][mm] = m[t] / b * 100
        jja = seasonal(m, [6, 7, 8], BASE[0], 2026)
        if 2026 in jja.index:
            r["jja"] = jja[2026] / jja.loc[BASE[0]:BASE[1]].mean() * 100
            r["jja_rank"] = int(jja.rank()[2026]); r["jja_n"] = len(jja)
        rows.append(r)
    g26 = gpcc_2026()
    gm = mon["GPCC"]
    for kind in ["first guess", "monitoring"]:
        sub = [x for x in g26 if x["kind"] == kind]
        if not sub:
            continue
        r = dict(src=f"GPCC {kind}", months={}, gauges={})
        for x in sub:
            b = gm[(gm.index.month == x["month"]) & (gm.index.year >= BASE[0]) & (gm.index.year <= BASE[1])].mean()
            r["months"][x["month"]] = x["p"] / b * 100; r["gauges"][x["month"]] = x["gauges"]
        if all(mm in r["months"] for mm in (6, 7, 8)):
            tot = sum(x["p"] for x in sub if x["month"] in (6, 7, 8))
            r["jja"] = tot / seasonal(gm, [6, 7, 8], BASE[0], BASE[1]).mean() * 100
        rows.append(r)
    a["y2026"] = rows
    # per department, June–August, the two products that disagree most and IMERG
    dg = depts()
    drows = []
    for _, d in dg.sort_values("name").iterrows():
        c3 = seasonal(chirps3_series(d.geometry), [6, 7, 8], BASE[0], 2026)
        im = seasonal(imerg_monthly(db["imerg"], d.pcode), [6, 7, 8], BASE[0], 2026)
        im4 = seasonal(imerg_monthly(db["imerg"], d.pcode), RAINS, BASE[0], 2026)
        drows.append(dict(name=d["name"], c3=c3[2026] / c3.loc[BASE[0]:BASE[1]].mean() * 100,
                          im=im[2026] / im.loc[BASE[0]:BASE[1]].mean() * 100,
                          im4=im4[2026] / im4.loc[BASE[0]:BASE[1]].mean() * 100 if 2026 in im4.index else np.nan,
                          im4_rank=int(im4.rank()[2026]) if 2026 in im4.index else None, n=len(im4)))
    a["dept2026"] = drows
    # IMERG daily, cumulative from 1 May
    d = imerg_daily(db["imerg"])
    cum = {}
    for y in range(BASE[0], 2027):
        s = d[f"{y}-05-01":f"{y}-10-31"]
        if len(s) < 100:
            continue
        s = s.cumsum(); s.index = s.index.dayofyear - pd.Timestamp(f"{y}-05-01").dayofyear
        cum[y] = s
    a["imerg_cum"] = pd.DataFrame(cum)
    a["imerg_last"] = d.index.max()

    # vegetation and SPI (ASAP)
    asp = asap(); asp["date"] = pd.to_datetime(asp.date.astype(str), format="%Y%m%d")
    en = asp[asp.region_name == "Extrême-Nord"]
    veg = {}
    for key in ASAP_IND:
        s = en[en.key == key].set_index("date")["value"].sort_index()
        veg[key] = s
    a["veg"] = veg
    zc = veg["zfparc_crop"]; last = zc.index.max()
    same = zc[(zc.index.month == last.month) & (zc.index.day == last.day)]; same.index = same.index.year
    a["veg_last"] = dict(date=last, val=float(zc.iloc[-1]), rank=int(same.rank()[last.year]), n=len(same))
    sp = veg["spi3_crop"]; same = sp[(sp.index.month == last.month) & (sp.index.day == last.day)]; same.index = same.index.year
    a["spi_last"] = dict(val=float(sp[last]) if last in sp.index else np.nan, rank=int(same.rank()[last.year]) if last.year in same.index else None, n=len(same), y0=int(same.index.min()))
    eos = zc[(zc.index.month == 10) & (zc.index.day == 21)]; eos.index = eos.index.year
    c = eos.index.intersection(njas.index)
    a["veg_enso"] = dict(r=stats.pearsonr(eos[c], njas[c])[0], p=stats.pearsonr(eos[c], njas[c])[1], n=len(c),
                         en={int(y): float(eos[y]) for y in c if njas[y] >= THR})
    june = zc[(zc.index.year == 2026) & (zc.index.month == 6)]
    a["veg_june"] = float(june.min()) if len(june) else np.nan

    # 3. floods (FloodScan SFED: flooded fraction of the unit)
    fs = db["floodscan"].copy(); fs["year"] = fs.valid_date.dt.year; fs["doy"] = fs.valid_date.dt.dayofyear
    a["fs_last"] = fs.valid_date.max()
    cut = int(a["fs_last"].dayofyear)
    frows, fcurves = [], {}
    for p_, nm in [(PCODE, "Extrême-Nord")] + [(k, v) for k, v in DEPTS.items() if k in ("CM004002", "CM004003", "CM004001", "CM004004")]:
        x = fs[fs.pcode == p_]
        ytd = x[x.doy <= cut].groupby("year")["mean"].max()
        full = x[x.year <= 2025].groupby("year")["mean"].max()
        frows.append(dict(name=nm, ytd=float(ytd[2026]), rank=int(ytd.rank(ascending=False)[2026]), n=len(ytd),
                          med=float(ytd.loc[:2025].median()), full_med=float(full.median()),
                          top=[(int(y), float(v)) for y, v in full.sort_values(ascending=False).head(5).items()]))
        fcurves[nm] = x.pivot_table(index="doy", columns="year", values="mean")
    a["floods"] = frows; a["flood_curves"] = fcurves
    pk = fs[(fs.pcode == PCODE) & (fs.year <= 2025)].groupby("year")["mean"].max()
    c = pk.index.intersection(njas.index)
    a["flood_enso"] = dict(r=stats.pearsonr(pk[c], njas[c])[0], p=stats.pearsonr(pk[c], njas[c])[1],
                           rho=stats.spearmanr(pk[c], njas[c])[0], n=len(c),
                           top=[(int(y), float(pk[y]), float(njas[y])) for y in pk.sort_values(ascending=False).index[:6]])
    lc = fs[(fs.pcode == "CM004002") & (fs.year <= 2025)].groupby("year")["mean"].max()
    c = lc.index.intersection(njas.index)
    a["flood_enso_lc"] = dict(r=stats.pearsonr(lc[c], njas[c])[0], p=stats.pearsonr(lc[c], njas[c])[1], n=len(c))

    # 4. heat: Mar–May after the El Niño winter
    t2 = era5_t2m()
    mam = seasonal(t2, [3, 4, 5], 1981, 2026) / 3
    a["t2m_last"] = t2.index.max()
    prev = nndj.copy(); prev.index = prev.index + 1
    m_ = mam.loc[:2026]
    an = detrend(m_)
    c = an.index.intersection(prev.index)
    a["heat"] = dict(r=stats.pearsonr(an[c], prev[c])[0], p=stats.pearsonr(an[c], prev[c])[1], n=len(c),
                     trend=np.polyfit(m_.index.values.astype(float), m_.values, 1)[0] * 10,
                     df=pd.DataFrame({"an": an[c], "nino": prev[c], "mam": m_[c]}))
    clim = t2[(t2.index.year >= 1991) & (t2.index.year <= 2020)]
    clim = clim.groupby(clim.index.month).mean()
    a["t2m_clim"] = clim
    a["t2m_2026"] = {mm: float(t2[pd.Timestamp(2026, mm, 1)] - clim[mm]) for mm in range(1, 13) if pd.Timestamp(2026, mm, 1) in t2.index}
    a["mam_rank_raw"] = {int(y): int(m_.rank(ascending=False)[y]) for y in m_.index}
    a["mam_hottest"] = [(int(y), float(v)) for y, v in m_.sort_values(ascending=False).head(5).items()]

    # 5. food security (Cadre Harmonisé)
    ch = cadre_harmonise()
    l1 = ch[(ch.level == "level1") & (ch.level1_name == "Far-North") & (ch.phase == "3+")].copy()
    l1["analysis_date"] = pd.to_datetime(l1.analysis_date); l1["start"] = pd.to_datetime(l1.reference_period_start)
    a["ch"] = l1.sort_values(["analysis_date", "start"])[["analysis_date", "period_type", "start", "population", "fraction"]]
    ar = ch[(ch.level == "area") & (ch.level1_name == "Far-North")].copy()
    ar["analysis_date"] = pd.to_datetime(ar.analysis_date)
    lastd = ar.analysis_date.max()
    x = ar[ar.analysis_date == lastd]
    piv = x.pivot_table(index=["area_name", "period_type"], columns="phase", values="population", aggfunc="sum")
    fr = x[x.phase == "3+"].set_index(["area_name", "period_type"])["fraction"]
    a["ch_dept"] = (piv, fr, lastd)

    # 6. the following year: rains after an El Niño winter (NDJ ≥ 1.0)
    after = []
    for y in nndj.index[nndj >= 1.0]:
        y1 = int(y) + 1
        if y1 > 2025:
            continue
        row = dict(event=f"{int(y)}/{str(y1)[2:]}", ndj=float(nndj[y]), jas=float(njas.get(y1, np.nan)))
        for k in ["GPCC", "CHIRPS v3", "ERA5"]:
            s = rains[k].loc[1950 if k == "GPCC" else 1981:2025]
            row[k] = float(s.rank(pct=True)[y1] * 100) if y1 in s.index else None
        row["mam"] = float(an[y1]) if y1 in an.index else None
        row["mam_rank"] = int(an.rank(ascending=False)[y1]) if y1 in an.index else None
        after.append(row)
    a["after"] = after
    a["after_r"] = {}
    for k in ["GPCC", "CHIRPS v3", "ERA5"]:
        s = rains[k].loc[1982:2025]; c = s.index.intersection(prev.index)
        a["after_r"][k] = stats.pearsonr(detrend(s[c]), prev[c])

    # SEAS5 skill for the July–September rains, by issue month
    sk = []
    path = edd._ensure_skill_cube()
    if path is not None:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            sub = ds.pearson_r.sel(y=slice(13.1, 10.0), x=slice(13.4, 15.7))
            for im_ in [1, 2, 3, 4, 5, 6, 7]:
                v = sub.sel(issued_month=im_, trimester="JAS").values
                v = v[np.isfinite(v)]
                sk.append(dict(month=im_, lead=7 - im_, r=float(np.median(v)) if v.size else np.nan))
    a["skill"] = sk

    figures(a)
    return a


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight"); plt.close(fig)


def figures(a: dict) -> None:
    fig_history(a); fig_running(a); fig_2026(a); fig_vegetation(a); fig_floods(a); fig_heat(a); fig_ch(a)


def fig_history(a: dict) -> None:
    h = a["hist"]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    col = h.phase.map({"El Niño": C_EN, "Neutral": C_NEU, "La Niña": C_LN})
    ax.bar(h.index, h.z, color=col, width=0.78)
    ax.plot(h.index, h.zc, "o", ms=3.6, mfc="white", mec=C_TEXT, mew=0.9)
    zt = (h.gpcc.quantile(1 / 3) - h.gpcc.mean()) / h.gpcc.std()
    ax.axhline(0, color="#9aa3ad", lw=0.8)
    for yr, row in h[h.phase == "El Niño"].iterrows():
        ax.text(yr, min(row.z, row.zc if pd.notna(row.zc) else row.z) - 0.1, str(yr), fontsize=7, color=C_EN, ha="center", va="top", rotation=90)
    ax.set_xlim(1949, 2026.2)
    ax.set_ylabel("June–September rainfall\nanomaly (SD)", fontsize=9, color=C_MUTED)
    ax.legend(handles=[Patch(color=C_EN, label=f"El Niño (Jul–Sep Niño3.4 ≥ +{THR})"), Patch(color=C_NEU, label="Neutral"),
                       Patch(color=C_LN, label=f"La Niña (≤ −{THR})"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=C_TEXT, label="CHIRPS v3 (1981–)")],
              frameon=False, fontsize=8, loc="upper left", ncol=4, bbox_to_anchor=(0, 1.13))
    ax.set_title("Extrême-Nord: June–September rainfall, GPCC gauge analysis (bars) and CHIRPS v3 (dots), by ENSO phase",
                 fontsize=9.5, color=C_TEXT, loc="left", pad=22)
    edd._style_ax(ax)
    fig.tight_layout(); _save(fig, "history.png")


def fig_running(a: dict) -> None:
    run = a["running"]
    fig, ax = plt.subplots(figsize=(9.6, 2.8))
    crit = 0.355  # |r| for p = 0.05, n = 31
    ax.axhspan(-crit, crit, color="#eef1f1")
    ax.axhline(0, color="#9aa3ad", lw=0.8)
    ax.plot(run.index, run.values, color=SRC_COL["GPCC"], lw=2)
    ax.text(1966, crit + 0.03, "inside the grey band: not significant for one 31-year window", fontsize=7.5, color=C_MUTED, va="bottom")
    ax.set_ylim(-0.75, 0.45); ax.set_xlim(1964, 2011)
    ax.set_ylabel("r, 31-year window", fontsize=9, color=C_MUTED)
    ax.set_title("GPCC June–September rainfall (detrended) against July–September Niño3.4, 31-year running correlation",
                 fontsize=9.5, color=C_TEXT, loc="left")
    edd._style_ax(ax)
    fig.tight_layout(); _save(fig, "running.png")


def fig_2026(a: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    rows = [r for r in a["y2026"] if "jja" in r]
    order = ["ERA5", "IMERG", "GPCC first guess", "GPCC monitoring", "CHIRPS v3", "CHIRPS v2"]
    rows = sorted(rows, key=lambda r: order.index(r["src"]) if r["src"] in order else 99)
    labels = [r["src"] for r in rows]
    vals = [r["jja"] for r in rows]
    cols = [SRC_COL.get(r["src"].split(" first")[0].split(" monitoring")[0], "#1F5F96") for r in rows]
    y = np.arange(len(rows))
    bars = ax1.barh(y, vals, color=cols, height=0.62)
    for b, r in zip(bars, rows):
        if r["src"] in ("CHIRPS v2",) or "GPCC" in r["src"]:
            b.set_alpha(0.55)
    ax1.axvline(100, color=C_TEXT, lw=0.9)
    for yi, v in zip(y, vals):
        ax1.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=C_TEXT)
    ax1.set_yticks(y, labels, fontsize=8.5); ax1.invert_yaxis(); ax1.set_xlim(0, 135)
    ax1.set_xlabel(f"June–August 2026, % of the {BASE[0]}–{BASE[1]} mean", fontsize=8.5, color=C_MUTED)
    ax1.set_title("Five rainfall products, one season", fontsize=9.5, color=C_TEXT, loc="left")
    edd._style_ax(ax1); ax1.xaxis.grid(True, color="#e6eaea"); ax1.yaxis.grid(False)
    cum = a["imerg_cum"]
    base = cum[[c for c in cum.columns if c <= BASE[1]]]
    x = cum.index.values
    ax2.fill_between(x, base.quantile(0.1, axis=1), base.quantile(0.9, axis=1), color="#e8e2ee", lw=0, label=f"{BASE[0]}–{BASE[1]}, 10th–90th percentile")
    ax2.plot(x, base.median(axis=1), color="#a58cb0", lw=1.2, label="median")
    s26 = cum[2026].dropna()
    ax2.plot(s26.index, s26.values, color=SRC_COL["IMERG"], lw=2.2, label="2026")
    ticks = [0, 31, 61, 92, 123, 153]
    ax2.set_xticks(ticks, ["1 May", "1 Jun", "1 Jul", "1 Aug", "1 Sep", "1 Oct"], fontsize=8.5)
    ax2.set_xlim(0, 183)
    ax2.set_ylabel("mm since 1 May", fontsize=8.5, color=C_MUTED)
    ax2.set_title(f"IMERG, Extrême-Nord mean, cumulative to {a['imerg_last']:%d %b %Y}", fontsize=9.5, color=C_TEXT, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    edd._style_ax(ax2)
    fig.tight_layout(); _save(fig, "season2026.png")


def fig_vegetation(a: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax, key, ttl in [(axes[0], "zfparc_crop", "Cropland biomass (ASAP zFPARc)"), (axes[1], "spi3_crop", "3-month rainfall index (ASAP SPI-3, cropland)")]:
        s = a["veg"][key]
        s = s[s.index.month.isin([5, 6, 7, 8, 9, 10])]
        df = pd.DataFrame({"v": s.values, "year": s.index.year, "k": s.index.strftime("%m-%d")})
        piv = df.pivot_table(index="k", columns="year", values="v")
        base = piv[[c for c in piv.columns if BASE[0] <= c <= BASE[1]]]
        x = np.arange(len(piv))
        ax.fill_between(x, base.min(axis=1), base.max(axis=1), color="#eef1f1", lw=0, label=f"{base.columns.min()}–{BASE[1]} range")
        for yy, cc in [(2015, C_EN), (2023, "#e8a0a9")]:
            if yy in piv:
                ax.plot(x, piv[yy], color=cc, lw=1.1, ls=(0, (3, 2)), label=f"{yy} (El Niño)")
        if 2026 in piv:
            ax.plot(x, piv[2026], color=C_TEXT, lw=2.2, label="2026")
        ax.axhline(-1, color="#7A4E22", lw=0.9, ls=(0, (4, 3)))
        ax.text(len(x) - 0.5, -1.05, "ASAP warning threshold", fontsize=7, color="#7A4E22", ha="right", va="top")
        ax.axhline(0, color="#9aa3ad", lw=0.7)
        labs = [f"{pd.Timestamp('2001-' + k):%d %b}" if k.endswith("-01") else "" for k in piv.index]
        ax.set_xticks(x[[i for i, l in enumerate(labs) if l]], [l for l in labs if l], fontsize=8)
        ax.set_title(f"Extrême-Nord: {ttl}", fontsize=9.5, color=C_TEXT, loc="left")
        ax.set_ylabel("z-score", fontsize=8.5, color=C_MUTED)
        edd._style_ax(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); _save(fig, "vegetation.png")


def fig_floods(a: dict) -> None:
    names = ["Logone-et-Chari", "Mayo-Danay"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=False)
    for ax, nm in zip(axes, names):
        piv = a["flood_curves"][nm]
        piv = piv.loc[152:365]
        base = piv[[c for c in piv.columns if c <= 2025]]
        x = piv.index.values
        ax.fill_between(x, base.quantile(0.1, axis=1) * 100, base.quantile(0.9, axis=1) * 100, color="#dce8f3", lw=0, label="1998–2025, 10th–90th percentile")
        ax.plot(x, base.median(axis=1) * 100, color="#7fa6c9", lw=1.2, label="median")
        if 2024 in piv:
            ax.plot(x, piv[2024] * 100, color=C_NEU, lw=1, ls=(0, (3, 2)), label="2024 (record floods)")
        ax.plot(x, piv[2026] * 100, color="#1F5F96", lw=2.2, label="2026")
        ticks = [152, 182, 213, 244, 274, 305, 335]
        ax.set_xticks(ticks, ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8.5)
        ax.set_xlim(152, 365)
        ax.set_ylabel("% of the department flooded", fontsize=8.5, color=C_MUTED)
        ax.set_title(f"{nm}: flooded area (FloodScan SFED)", fontsize=9.5, color=C_TEXT, loc="left")
        edd._style_ax(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); _save(fig, "floods.png")


def fig_heat(a: dict) -> None:
    df = a["heat"]["df"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    col = [C_EN if v >= 1.0 else (C_LN if v <= -1.0 else C_NEU) for v in df.nino]
    ax.scatter(df.nino, df.an, c=col, s=34, edgecolor="white", lw=0.6, zorder=3)
    sl, ic = np.polyfit(df.nino, df.an, 1)
    xx = np.linspace(df.nino.min(), df.nino.max(), 10)
    ax.plot(xx, ic + sl * xx, color=C_MUTED, lw=1, ls=(0, (4, 3)))
    for y, r in df.iterrows():
        if r.nino >= 1.0 or r.an >= 0.8 or y in (1992,):
            ax.text(r.nino + 0.04, r.an, str(y), fontsize=7.5, color=C_TEXT, va="center")
    ax.axhline(0, color="#9aa3ad", lw=0.7); ax.axvline(0, color="#9aa3ad", lw=0.7)
    ax.set_xlabel("Niño3.4 the winter before (November–January, °C)", fontsize=8.5, color=C_MUTED)
    ax.set_ylabel("March–May temperature anomaly\n(°C, trend removed)", fontsize=8.5, color=C_MUTED)
    h = a["heat"]
    ax.set_title(f"Extrême-Nord hot season after El Niño: ERA5, 1982–{df.index.max()} (r = {h['r']:+.2f})", fontsize=9.5, color=C_TEXT, loc="left")
    edd._style_ax(ax)
    fig.tight_layout(); _save(fig, "heat.png")


def fig_ch(a: dict) -> None:
    ch = a["ch"]
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    cur = ch[ch.period_type == "current"]; pro = ch[ch.period_type != "current"]
    ax.plot(cur.start, cur.population / 1e6, "o", color="#E67800", ms=7, label="current, at the time of the analysis")
    oct_ = pro[pro.analysis_date.dt.month >= 9]; mar = pro[pro.analysis_date.dt.month < 9]
    ax.plot(oct_.start - pd.Timedelta(days=12), oct_.population / 1e6, "s", mfc="white", mec="#E67800", mew=1.6, ms=7,
            label="June–August projection made the previous Oct/Nov")
    ax.plot(mar.start + pd.Timedelta(days=12), mar.population / 1e6, "D", mfc="white", mec="#9a5200", mew=1.4, ms=6,
            label="June–August projection updated in March")
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("people in Phase 3+ (millions)", fontsize=8.5, color=C_MUTED)
    ax.set_title("Extrême-Nord: Cadre Harmonisé, people in Crisis or worse (Phase 3+)",
                 fontsize=9.5, color=C_TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    edd._style_ax(ax)
    fig.tight_layout(); _save(fig, "cadre_harmonise.png")


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _r(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}".replace("-", "−")


def _p(v) -> str:
    return "—" if v is None or np.isnan(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}" if v < 0.01 else f"{v:.2f}")


def _sg(v: float, nd: int) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def _pc(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}%"


def _n(v) -> str:
    return f"{v:,.0f}".replace(",", " ")


def _m(v) -> str:
    return f"{v / 1e6:.2f}".rstrip("0").rstrip(".") + " million"


MONTHS = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}


def render(spec: dict, a: dict) -> str:
    T = spec.get("titles", {}).get
    o = [edd.HEAD.format(title=html.escape(spec["page_title"]), desc=html.escape(spec["assessment"]["one_line"]),
                         css=edd.CSS + ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:16px 0}"
                         ".card .big{font-size:16px}.watch td:first-child{white-space:nowrap}"
                         "table{display:block;max-width:100%;overflow-x:auto}.wrap{overflow-wrap:anywhere}",
                         home="../", home_label="ENSO deep dives")]
    o.append('<p class="eyebrow">Teleconnections · ENSO deep dive</p>')
    o.append(f'<h1>{html.escape(spec["name"])}</h1>')
    o.append(f'<p class="meta">{html.escape(spec["subtitle"])}. Data to {a["imerg_last"]:%d %B %Y}; built {pd.Timestamp.now():%d %B %Y}.</p>')
    o.append(f'<div class="summary">{spec.get("summary_html", "")}</div>')
    o.append('<div class="cards">' + "".join(
        f'<div class="card"><p class="lbl">{html.escape(c["label"])}</p><p class="big">{html.escape(c["big"])}</p>'
        f'<p class="small">{c["note_html"]}</p></div>' for c in spec.get("cards", [])) + '</div>')

    # 1. forecasts and calendar
    o.append(f'<h2>{T("forecast", "1. Where things stand")}</h2>')
    o.append(spec.get("forecast_html", ""))
    o.append(spec.get("calendar_html", ""))

    # 2. the relationship
    o.append(f'<h2>{T("enso", "2. How much El Niño matters for the rains")}</h2>')
    o.append(spec.get("enso_html", ""))
    o.append('<figure><img src="history.png" alt="June–September rainfall by ENSO phase, 1950–2025"><figcaption>Bars: GPCC '
             'gauge analysis, area-weighted over the region, standardised over 1950–2025. Dots: CHIRPS v3 over the region, '
             'standardised over 1981–2025. Colour: Niño3.4 averaged over July–September (NOAA PSL, ERSST v5 basis). El Niño '
             'seasons are labelled.</figcaption></figure>')
    o.append('<table><thead><tr><th>Record</th><th>Years</th><th class="num">r</th><th class="num">r, trend removed</th><th class="num">p</th>'
             '<th class="num">El Niño seasons: driest / middle / wettest third</th><th class="num">La Niña: driest / middle / wettest</th>'
             '<th class="num">Trend, mm per decade</th></tr></thead><tbody>')
    for r in a["rel"]:
        flag = ' class="hl"' if r["src"] == "ERA5" else ""
        o.append(f'<tr{flag}><td>{r["src"]}</td><td>{r["y0"]}–{r["y1"]}</td><td class="num">{_r(r["r"])}</td><td class="num">{_r(r["rd"])}</td>'
                 f'<td class="num">{_p(r["pd"])}</td><td class="num">{" / ".join(map(str, r["en"]))} of {r["n_en"]}</td>'
                 f'<td class="num">{" / ".join(map(str, r["ln"]))} of {r["n_ln"]}</td><td class="num">{f"{r['trend']:+.0f}".replace("-", "−")}</td></tr>')
    gl = a["gpcc_long"]
    o.append(f'<tr><td>GPCC, full record</td><td>{gl["y0"]}–{gl["y1"]}</td><td class="num">{_r(gl["r"])}</td><td class="num">{_r(gl["rd"])}</td>'
             f'<td class="num">{_p(gl["pd"])}</td><td class="num">{" / ".join(map(str, gl["en"]))} of {gl["n_en"]}</td>'
             f'<td class="num">{" / ".join(map(str, gl["ln"]))} of {gl["n_ln"]}</td><td class="num">{f"{gl['trend']:+.0f}".replace("-", "−")}</td></tr>')
    o.append('</tbody></table>')
    o.append('<p class="small">June–September totals over the region against July–September Niño3.4 (concurrent). Thirds are taken on '
             'the detrended series, so the post-1980s recovery of Sahel rainfall does not pile the early years into the driest third. '
             'El Niño and La Niña: Niño3.4 ≥ +0.5 and ≤ −0.5. p is for the detrended r. The highlighted ERA5 row is discussed in the text.</p>')
    o.append(spec.get("enso_after_table_html", ""))
    o.append('<figure><img src="running.png" alt="31-year running correlation, GPCC vs Niño3.4"><figcaption>Each point is the correlation '
             'over the 31 seasons centred on that year (1950–2025): negative throughout, but in most single windows too weak to '
             'pass a significance test on its own.</figcaption></figure>')
    o.append(f'<h3>{T("strong", "Strong El Niño seasons")}</h3>{spec.get("strong_html", "")}')
    o.append('<table><thead><tr><th>Season</th><th class="num">Jul–Sep Niño3.4</th><th class="num">GPCC</th><th class="num">CHIRPS v3</th>'
             '<th class="num">IMERG</th><th class="num">ERA5</th><th>Notes</th></tr></thead><tbody>')
    for s in a["strong"]:
        note = spec.get("strong_notes", {}).get(str(s["year"]), "")
        o.append(f'<tr><td>{s["year"]}</td><td class="num">{s["nino"]:+.1f}</td>' +
                 "".join(f'<td class="num">{_pc(s[k])}</td>' for k in ["GPCC", "CHIRPS v3", "IMERG", "ERA5"]) +
                 f'<td class="small">{note}</td></tr>')
    o26 = a["oni_2026"]
    jja = o26.get("JJA")
    o.append(f'<tr class="hl"><td>2026</td><td class="num">{"+%.1f" % jja if jja is not None else "—"}*</td>'
             f'<td colspan="4" class="small">see section 3</td><td class="small">{spec.get("strong_notes", {}).get("2026", "")}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">Percentile of the June–September total within each record (0 = driest, 100 = wettest): GPCC within 1950–2025, '
             f'the others within their own years (CHIRPS and ERA5 from 1981, IMERG from 2001). Niño3.4: NOAA PSL, ERSST v5. '
             f'*2026: CPC\'s official ONI for June–August (ERSST v5); the pinned monthly series behind the rest of the table ends in '
             f'{a["nino_last"][0]:%B %Y}.</p>')

    o.append(f'<h3>{T("gauges", "The rain gauges behind the records")}</h3>{spec.get("gauges_html", "")}')
    gd = a["gauges"].groupby((a["gauges"].index // 10) * 10).mean()
    gd = gd[gd.index >= 1920]
    o.append('<table><thead><tr><th></th>' + "".join(f'<th class="num">{d}s</th>' for d in gd.index) + '</tr></thead><tbody>'
             '<tr><td>Gauges per month, June–September</td>' + "".join(f'<td class="num">{v:.1f}</td>' for v in gd.values) + '</tr></tbody></table>')
    o.append('<p class="small">GPCC Full Data v2022 (DWD), number of gauges in the 1° cells that make up the region, averaged over '
             'June–September months and over each decade (the 2020s: 2020 only, the last year of the product).</p>')

    # 3. the 2026 season
    o.append(f'<h2>{T("season", "3. What the 2026 season delivered")}</h2>')
    o.append(spec.get("season_html", ""))
    o.append('<figure><img src="season2026.png" alt="June–August 2026 rainfall by product, and IMERG cumulative rainfall"><figcaption>Left: '
             f'June–August 2026 over the region as a share of each product\'s own {BASE[0]}–{BASE[1]} mean (paler bars: CHIRPS v2, '
             'and GPCC 2026 products interpolated from gauges outside the region). Right: IMERG late run, daily, region mean.</figcaption></figure>')
    o.append('<table><thead><tr><th>Product</th><th>What it is</th>' + "".join(f'<th class="num">{MONTHS[m]}</th>' for m in [5] + RAINS) +
             f'<th class="num">Jun–Aug</th><th class="num">Rank, driest = 1</th></tr></thead><tbody>')
    what = spec.get("product_notes", {})
    for r in a["y2026"]:
        cells = "".join(f'<td class="num">{_pc(r["months"].get(m))}</td>' for m in [5] + RAINS)
        rank = f'{r["jja_rank"]} of {r["jja_n"]}' if "jja_rank" in r else "—"
        g = ""
        if r.get("gauges"):
            g = f' Gauges in the 1° cells overlapping the region (two of them mostly in Chad): {", ".join(f"{MONTHS[m]} {v}" for m, v in r["gauges"].items())}.'
        o.append(f'<tr><td>{r["src"]}</td><td class="small">{what.get(r["src"], "")}{g}</td>{cells}'
                 f'<td class="num"><strong>{_pc(r.get("jja"))}</strong></td><td class="num">{rank}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">Each month and season as a share of the same product\'s {BASE[0]}–{BASE[1]} mean (one baseline for all, '
             f'set by IMERG\'s start). Rank among {BASE[0]}–2026. Blank: not yet published.</p>')
    o.append(f'<h3>{T("depts", "By department")}</h3>{spec.get("depts_html", "")}')
    o.append('<table><thead><tr><th>Department</th><th class="num">CHIRPS v3, Jun–Aug</th><th class="num">IMERG, Jun–Aug</th>'
             '<th class="num">IMERG, Jun–Sep</th><th class="num">IMERG Jun–Sep rank, driest = 1</th></tr></thead><tbody>')
    for d in a["dept2026"]:
        o.append(f'<tr><td>{d["name"]}</td><td class="num">{_pc(d["c3"])}</td><td class="num">{_pc(d["im"])}</td><td class="num">{_pc(d["im4"])}</td>'
                 f'<td class="num">{d["im4_rank"]} of {d["n"]}</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<h3>{T("veg", "What the vegetation shows")}</h3>{spec.get("veg_html", "")}')
    o.append('<figure><img src="vegetation.png" alt="ASAP cropland biomass and SPI-3, 2026"><figcaption>JRC ASAP, Extrême-Nord, '
             'dekadal. Left: z-score of cumulative fAPAR on cropland during the growing cycle (biomass compared with the same point '
             'in past seasons). Right: 3-month standardised precipitation index on cropland. Grey: range of past seasons.</figcaption></figure>')
    o.append(f'<h3>{T("reports", "What regional and field reports say")}</h3>{spec.get("reports_html", "")}')

    # 4. floods
    o.append(f'<h2>{T("floods", "4. Floods")}</h2>')
    o.append(spec.get("floods_html", ""))
    o.append('<figure><img src="floods.png" alt="FloodScan flooded area, Logone-et-Chari and Mayo-Danay"><figcaption>AER FloodScan SFED '
             '(standard flood extent depiction), daily share of the department flagged as flooded.</figcaption></figure>')
    o.append('<table><thead><tr><th>Unit</th><th class="num">Peak flooded share to ' + f'{a["fs_last"]:%d %b}' + ', 2026</th>'
             '<th class="num">Median of past years to that date</th><th class="num">2026 rank (1 = most flooded)</th>'
             '<th>Largest whole-season peaks, 1998–2025</th></tr></thead><tbody>')
    for f in a["floods"]:
        top = ", ".join(f"{y} ({v * 100:.0f}%)" for y, v in f["top"][:4])
        o.append(f'<tr><td>{f["name"]}</td><td class="num">{f["ytd"] * 100:.1f}%</td><td class="num">{f["med"] * 100:.1f}%</td>'
                 f'<td class="num">{f["rank"]} of {f["n"]}</td><td class="small">{top}</td></tr>')
    o.append('</tbody></table>')
    fe = a["flood_enso"]
    o.append(spec.get("floods_after_html", "").format(r=_r(fe["r"]), p=_p(fe["p"]), n=fe["n"],
             top=", ".join(f'{y} ({phase(n_)})' for y, _, n_ in fe["top"][:5]),
             r_lc=_r(a["flood_enso_lc"]["r"])))

    # 5. heat
    o.append(f'<h2>{T("heat", "5. The hot season after El Niño")}</h2>')
    o.append(spec.get("heat_html", ""))
    o.append('<figure><img src="heat.png" alt="March–May temperature against the previous winter\'s Niño3.4"><figcaption>ERA5 monthly 2 m '
             'temperature, mean over the 0.25° cells touching the region, March–May, linear trend removed '
             f'({a["heat"]["trend"]:+.2f} °C per decade). Red: winters with Niño3.4 ≥ +1.0; blue: ≤ −1.0.</figcaption></figure>')
    o.append('<table><thead><tr><th>El Niño winter</th><th class="num">Nov–Jan Niño3.4</th><th class="num">March–May after: anomaly, trend removed</th>'
             '<th class="num">Rank, hottest = 1</th></tr></thead><tbody>')
    for r in [r for r in a["after"] if r["mam"] is not None]:
        o.append(f'<tr><td>{r["event"]}</td><td class="num">{r["ndj"]:+.1f}</td><td class="num">{_sg(r["mam"], 2)} °C</td>'
                 f'<td class="num">{r["mam_rank"]} of {a["heat"]["n"]}</td></tr>')
    o.append('</tbody></table>')
    o.append(spec.get("heat_after_html", ""))

    # 6. food security
    o.append(f'<h2>{T("food", "6. Food security")}</h2>')
    o.append(spec.get("food_html", ""))
    o.append('<figure><img src="cadre_harmonise.png" alt="Cadre Harmonisé Phase 3+ in Extrême-Nord"><figcaption>Cadre Harmonisé '
             'results for Extrême-Nord via HDX/HAPI (the team\'s mirror). Hollow markers are projections for the June–August lean '
             'season: squares from the October/November analysis the year before, diamonds from the March update.</figcaption></figure>')
    piv, fr, lastd = a["ch_dept"]
    o.append(f'<table><thead><tr><th>Department</th><th class="num">Phase 3+, current (Oct–Dec 2025)</th><th class="num">Phase 3+, '
             f'projected Jun–Aug 2026</th><th class="num">of whom Phase 4</th><th class="num">Share in Phase 3+</th></tr></thead><tbody>')
    for nm in sorted({i[0] for i in piv.index}):
        cur = piv.loc[(nm, "current")] if (nm, "current") in piv.index else None
        pro = piv.loc[(nm, "first projection")] if (nm, "first projection") in piv.index else None
        if pro is None:
            continue
        o.append(f'<tr><td>{nm.replace("-chari", "-Chari").replace("-danay", "-Danay").replace("-kani", "-Kani").replace("-sava", "-Sava").replace("-tsanaga", "-Tsanaga").replace("Diamare", "Diamaré")}</td>'
                 f'<td class="num">{_n(cur["3+"]) if cur is not None else "—"}</td><td class="num">{_n(pro["3+"])}</td>'
                 f'<td class="num">{_n(pro.get("4", 0))}</td><td class="num">{fr.get((nm, "first projection"), np.nan) * 100:.0f}%</td></tr>')
    o.append('</tbody></table>')
    o.append(f'<p class="small">From the Cadre Harmonisé analysis of {lastd:%B %Y}, the latest in the HDX/HAPI record for Cameroon.</p>')
    o.append(spec.get("food_after_html", ""))

    # 7. 2027
    o.append(f'<h2>{T("next", "7. The 2027 rains")}</h2>')
    o.append(spec.get("next_html", ""))
    o.append('<table><thead><tr><th>El Niño winter</th><th class="num">Nov–Jan Niño3.4</th><th class="num">Jul–Sep Niño3.4, next year</th>'
             '<th class="num">Next June–September rain: GPCC</th><th class="num">CHIRPS v3</th><th class="num">ERA5</th></tr></thead><tbody>')
    for r in a["after"]:
        o.append(f'<tr><td>{r["event"]}</td><td class="num">{r["ndj"]:+.1f}</td><td class="num">{_sg(r["jas"], 1)}</td>' +
                 "".join(f'<td class="num">{_pc(r[k])}</td>' for k in ["GPCC", "CHIRPS v3", "ERA5"]) + '</tr>')
    o.append('</tbody></table>')
    ar = a["after_r"]
    o.append(f'<p class="small">Percentile of the following June–September (0 = driest): GPCC within 1950–2025, CHIRPS v3 and ERA5 within 1981–2025. Across all years since 1982, the correlation '
             f'between a winter\'s Niño3.4 and the next rainy season (trend removed) is {_r(ar["GPCC"][0])} in GPCC, '
             f'{_r(ar["CHIRPS v3"][0])} in CHIRPS v3 and {_r(ar["ERA5"][0])} in ERA5, none significant.</p>')
    if a["skill"]:
        o.append(f'<h3>{T("skill", "When seasonal forecasts start to help")}</h3>{spec.get("skill_html", "")}')
        mname = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July"}
        o.append('<table><thead><tr><th>SEAS5 issued in</th>' + "".join(f'<th class="num">{mname[s["month"]]}</th>' for s in a["skill"]) +
                 '</tr></thead><tbody><tr><td>Median r, July–September, Extrême-Nord cells</td>' +
                 "".join(f'<td class="num">{edd.skill_chip(edd.skill_cat(s["r"]))}{_r(np.floor(s["r"] * 100) / 100)}</td>' for s in a["skill"]) + '</tr></tbody></table>')
        o.append('<p class="small">From the team\'s SEAS5 skill cube (ensemble mean against ERA5, 1981–2025, detrended): low &lt; 0.30, '
                 'moderate 0.30–0.50. The reference is ERA5, whose weaknesses over this region are discussed in section 2.</p>')

    # 8. watch list
    o.append(f'<h2>{T("watch", "8. What to watch, and when")}</h2>')
    o.append('<table class="watch"><thead><tr><th>When</th><th>What</th><th>Why it matters here</th></tr></thead><tbody>' +
             "".join(f'<tr><td>{w["when"]}</td><td>{w["what_html"]}</td><td class="small">{w["why_html"]}</td></tr>' for w in spec.get("watch", [])) +
             '</tbody></table>')
    for sec in spec.get("sections_after", []):
        o.append(f'<h2>{html.escape(sec["title"])}</h2>{sec["html"]}')
    if spec.get("references"):
        o.append('<h2>References</h2><ul class="refs">' + "".join(f'<li>{r["html"]}</li>' for r in spec["references"]) + '</ul>')
    o.append('<p class="small">Generated by <code>extreme_nord_deep_dive.py</code> from <code>deep_dives/extreme-nord.toml</code>. '
             'Niño3.4 series as in the <a href="../../survey/">global survey</a>.</p>')
    o.append(edd.FOOT)
    return "\n".join(o)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def build(spec: dict, grid, ne, indices: pd.DataFrame) -> dict:
    """Called by enso_deep_dive.main() for a TOML with builder = "extreme_nord" (grid and ne unused:
    this page reads its own records, not the survey's ERA5 pixel stack)."""
    a = analyse(spec, indices)
    report(a)
    (OUT / "index.html").write_text(render(spec, a), encoding="utf-8")
    return a


def report(a: dict) -> None:
    """The numbers the TOML narrative quotes, printed so it can be reconciled after every rebuild."""
    print("  relationship (detrended r, El Niño thirds):", {r["src"]: (round(r["rd"], 2), r["en"]) for r in a["rel"]},
          "GPCC long", round(a["gpcc_long"]["rd"], 2), a["gpcc_long"]["en"], a["gpcc_long"]["n_en"])
    print("  GPCC by period:", [(p["y0"], p["y1"], round(p["rd"], 2), round(p["pd"], 3), p["en"]) for p in a["gpcc_periods"]])
    print("  strong:", [(s["year"], round(s["nino"], 2), {k: (round(s[k]) if s[k] is not None else None) for k in ["GPCC", "CHIRPS v3", "IMERG", "ERA5"]}) for s in a["strong"]])
    print("  2026:", [(r["src"], {m: round(v) for m, v in r["months"].items()}, round(r["jja"]) if "jja" in r else None, r.get("jja_rank")) for r in a["y2026"]])
    print("  depts:", [(d["name"], round(d["c3"]), round(d["im"]), d["im4"], d["im4_rank"]) for d in a["dept2026"]])
    print("  veg:", a["veg_last"], "spi", a["spi_last"], "june min", round(a["veg_june"], 2), "veg~enso", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a["veg_enso"].items()})
    print("  floods:", [(f["name"], round(f["ytd"] * 100, 1), f["rank"], f["n"]) for f in a["floods"]], "enso", {k: v for k, v in a["flood_enso"].items()})
    print("  heat:", {k: v for k, v in a["heat"].items() if k != "df"}, "hottest MAM", a["mam_hottest"], "2026 T anomalies", {k: round(v, 1) for k, v in a["t2m_2026"].items()})
    print("  after:", [(r["event"], round(r["jas"], 2), r["GPCC"], r["CHIRPS v3"], r["ERA5"], r["mam"], r["mam_rank"]) for r in a["after"]])
    print("  skill JAS:", [(s["month"], round(s["r"], 2)) for s in a["skill"]], "ONI 2026", a["oni_2026"])
    print("  CH:", a["ch"].tail(4).to_dict("records"))


def main() -> None:
    cfg = dict(ts.CONFIG, max_lag=3)
    spec = tomllib.loads((edd.DEEP_DIR / f"{SLUG}.toml").read_text()) | {"slug": SLUG}
    build(spec, None, None, ts.load_indices(cfg))
    print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
