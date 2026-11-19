"""Poor seasons and resource-type violence in Darfur: data for the restricted conflict page.

Restricted: built from ACLED events, so the output stays in the work folder
(<work>/processed/conflict/conflict_data.json) and is never committed; the page is encrypted.

Design (fixed before any drought-conflict relationship was looked at):
  * Outcome: ACLED "resource-type" events (conflict_classes.py), headline; communal second.
  * Test area: Darfur's localities. Eastern Chad is shown descriptively, not tested (the page
    says why).
  * Seasons SEASONS: chosen from ACLED's coverage history before any test; the war (from
    15 April 2023) changes who is coded, so nothing after it is tested.
  * Windows after season Y: Oct–Jan (primary: harvest, herds moving south), Mar–Jun of Y+1
    (pre-specified; late dry season) and Jun–Jul of Y+1 (chosen afterwards from the seasonal
    cycle of events, not from any drought relationship). The later windows stop at
    SEASONS_LATE (Mar–Jun 2023 straddles the start of the war), in the per-season panels too.
  * Drought: MODIS greening per locality (09_greening.py recipe: cell-normalised, own
    2001-2024 line, % of trend), localities with at least MIN_PIX valid pixels. Upstream:
    the same for the pastoral zone (rainfall-belt cells with a 1991-2020 June-September mean
    below PASTORAL_MAX_MM, both sides of the border) - one number per year, so it is used
    only in the per-season comparison.
  * Tests: two-way (locality and season) demeaned slope of event presence on greening, and
    a locality-only version; significance by permuting whole seasons of greening (the
    spatial pattern of a season kept intact; 64 x 9 is not the sample size). Out-of-sample
    skill: leave-one-season-out logistic model, baseline (locality propensity, events and war
    events in the previous 12 months: all known at forecast time) with and without greening.
  * Sensitivity: greening in units of each locality's own 2001-2024 standard deviation
    (sparse northern vegetation swings far more in % terms and would otherwise dominate), and
    the larger vegetated localities only (BIG_PIX pixels or more).
  * Falsification: the same test with war events as the outcome, and with the NEXT
    season's greening (placebo) as the predictor.
  * Added after the first results, from the literature (McGuirk & Nunn 2025): a window during
    the season itself (Jul–Oct of Y) and "upstream" greening per locality (vegetated belt
    pixels UP_LAT degrees north of it, within UP_LON of its longitude: where the herds that
    arrive after the rains come from). Labelled as added on the page.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import BBOX, PAGE_DATA, PROC, RAW, load_adm  # noqa: E402
from conflict_classes import classify  # noqa: E402

OUT = PROC / "conflict"
SEASONS = list(range(2014, 2023))
SEASONS_LATE = list(range(2014, 2022))
MIN_PIX = 25
PASTORAL_MAX_MM = 300.0
UP_LAT = (0.5, 2.5)  # upstream band: degrees north of the locality
UP_LON = 1.0         # and within this many degrees of its longitude
BIG_PIX = 400        # sensitivity: localities with at least this many vegetated pixels
BASE = (2001, 2024)
NPERM = 5000
NPERM_SKILL = 500
RNG = np.random.default_rng(20261117)
CLASSES = ["resource", "resource_strict", "communal", "war"]


def window(name: str, y: int):
    return {
        "oct_jan": (f"{y}-10-01", f"{y + 1}-02-01"),
        "mar_jun": (f"{y + 1}-03-01", f"{y + 1}-07-01"),
        "jun_jul": (f"{y + 1}-06-01", f"{y + 1}-08-01"),
        "jul_oct": (f"{y}-07-01", f"{y}-11-01"),  # during the season (added after the first results)
        "prior": (f"{y - 1}-10-01", f"{y}-10-01"),
    }[name]


# ---------------------------------------------------------------- greening per locality
def trend_pct(years, s):
    m = (years >= BASE[0]) & (years <= BASE[1]) & np.isfinite(s)
    b = np.polyfit(years[m], s[m], 1)
    return 100 * s / np.polyval(b, years)


def greening(a2: gpd.GeoDataFrame):
    z = np.load(PROC / "ndvi" / "greening.npz")
    years, lat, lon, g = z["years"], z["lat"], z["lon"], z["greening"]
    ny, nx = lat.size, lon.size
    res = float(lat[0] - lat[1])
    x0, y0, x1, y1 = BBOX
    rg = pd.read_parquet(PROC / "rain" / "grid_2026.parquet")
    k = lambda a, b: (round(a, 2), round(b, 2))  # noqa: E731
    belt05 = {k(a, b): bool(c) for a, b, c in zip(rg["lat"], rg["lon"], rg["belt"])}
    clim05 = {k(a, b): m for a, b, m in zip(rg["lat"], rg["lon"], rg["chirps_v2_jjas_clim"])}
    clat = np.floor(lat / 0.5) * 0.5 + 0.25
    clon = np.floor(lon / 0.5) * 0.5 + 0.25
    belt = np.array([[belt05.get(k(a, b), False) for b in clon] for a in clat])
    clim = np.array([[clim05.get(k(a, b), np.nan) for b in clon] for a in clat])
    base = (years >= BASE[0]) & (years <= BASE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(g[base], axis=0)
        norm = g / mean
    ok = belt & (mean >= 0.03) & (np.isfinite(g[base]).sum(0) >= 20)
    norm[:, ~ok] = np.nan
    a2 = a2.reset_index(drop=True)
    ids = rasterize(zip(a2.geometry, np.arange(1, len(a2) + 1)), out_shape=(ny, nx),
                    transform=from_origin(x0, y1, res, res), fill=0, dtype="int32")

    def series(mask):
        return np.array([np.nanmean(norm[i][mask]) if np.isfinite(norm[i][mask]).any() else np.nan
                         for i in range(len(years))])

    rows, npix = [], {}
    LAT, LON = lat[:, None], lon[None, :]
    cen = a2.geometry.representative_point()
    for i, pc in enumerate(a2["adm2_pcode"], start=1):
        mask = (ids == i) & ok
        npix[pc] = int(mask.sum())
        # upstream: vegetated belt pixels UP_LAT degrees north of the locality, within UP_LON of its longitude
        # (Darfur herds move south after the rains; added after the first results, from McGuirk & Nunn 2025)
        cy, cx = cen.iloc[i - 1].y, cen.iloc[i - 1].x
        up = ok & (LAT >= cy + UP_LAT[0]) & (LAT <= cy + UP_LAT[1]) & (np.abs(LON - cx) <= UP_LON)
        sl = trend_pct(years, series(mask)) if npix[pc] >= MIN_PIX else np.full(len(years), np.nan)
        su = trend_pct(years, series(up)) if up.sum() >= MIN_PIX else np.full(len(years), np.nan)
        if not (np.isfinite(sl).any() or np.isfinite(su).any()):
            continue
        for y, v, u in zip(years, sl, su):
            rows.append({"adm2_pcode": pc, "year": int(y), "grn": float(v), "grn_up": float(u)})
    unit = pd.DataFrame(rows)
    past = (ids > 0) & ok & (clim < PASTORAL_MAX_MM)
    pz = pd.DataFrame({"year": years.astype(int), "grn_pastoral": trend_pct(years, series(past))})
    print(f"[conflict] greening: {unit.adm2_pcode.nunique()} localities with >= {MIN_PIX} pixels; "
          f"pastoral zone {int(past.sum())} pixels")
    return unit, pz, npix


# ---------------------------------------------------------------- events
def events(a2):
    ev = classify(pd.read_parquet(RAW / "acled" / "acled_tcd_sdn.parquet"))
    pts = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["longitude"].astype(float),
                                                           ev["latitude"].astype(float)), crs=4326)
    j = gpd.sjoin(pts, a2[["adm2_pcode", "side", "geometry"]].rename(columns={"side": "side_poly"}),
                  how="left", predicate="within").drop(columns="index_right")
    j = j[~j.index.duplicated()]
    j.loc[j["side_poly"] != j["side"], "adm2_pcode"] = np.nan  # ACLED admin1 and the polygon disagree
    j["geo_precision"] = j["geo_precision"].astype(int)
    return pd.DataFrame(j.drop(columns="geometry"))


def count(ev, start, end, cls=None, units=None, precise=True):
    m = (ev["date"] >= start) & (ev["date"] < end)
    if cls:
        m &= ev[cls]
    if precise:
        m &= ev["geo_precision"] < 3
    s = ev[m].groupby("adm2_pcode").size()
    return s.reindex(units, fill_value=0) if units is not None else s


# ---------------------------------------------------------------- statistics
def demean(m, two_way=True):
    """m: seasons x units. Remove unit means (and season means for two-way), balanced panel."""
    r = m - m.mean(axis=0, keepdims=True)
    if two_way:
        r = r - r.mean(axis=1, keepdims=True)
    return r


def slope(x, y, two_way=True):
    xr, yr = demean(x, two_way), demean(y, two_way)
    return float((xr * yr).sum() / (xr * xr).sum()), float(np.corrcoef(xr.ravel(), yr.ravel())[0, 1])


def perm_test(x, y, two_way=True, n=NPERM):
    """Slope of y on x, with a null from permuting whole seasons (rows) of x."""
    b, r = slope(x, y, two_way)
    null = np.array([slope(x[RNG.permutation(x.shape[0])], y, two_way)[0] for _ in range(n)])
    p_two = float((np.abs(null) >= abs(b)).mean())
    p_neg = float((null <= b).mean())  # pre-specified direction: poorer greening, more events
    q = np.quantile(-1000 * null, [0.025, 0.975])
    return {"slope": b, "r": r, "p": p_two, "p_dir": p_neg,
            # percentage points of probability per 10 points of greening LOST
            "pp_per_10": -10 * 100 * b,
            # 95% range of that number when greening is unrelated to events: smaller effects can't be told apart
            "null_lo": float(q[0]), "null_hi": float(q[1])}


def logit_fit(X, y, lam=1e-2):
    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + lam * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        gr = X.T @ (p - y)
        gr[1:] += 2 * lam * w[1:]
        return gr

    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def loso_skill(pres, prior, war, grn):
    """Leave-one-season-out Brier scores: baseline vs baseline + greening. Arrays seasons x units."""
    S, U = pres.shape
    pb, pf = np.empty_like(pres, dtype=float), np.empty_like(pres, dtype=float)
    for s in range(S):
        tr = np.arange(S) != s
        k, ntr = pres[tr].sum(0), tr.sum()

        def prop(idx, held_out):
            """Locality propensity (log-odds) from the training seasons, leaving each training row's
            own season out so training and test rows are built the same way."""
            if held_out:
                return np.tile(np.log((k + 0.5) / (ntr - k + 0.5)), len(idx))
            kk = k[None, :] - pres[idx]
            return np.log((kk + 0.5) / (ntr - 1 - kk + 0.5)).ravel()

        def X(idx, with_g, held_out=False):
            n = len(idx)
            cols = [np.ones(n * U), prop(idx, held_out), prior[idx].ravel(), np.log1p(war[idx]).ravel()]
            if with_g:
                cols.append((grn[idx].ravel() - 100) / 10)
            return np.column_stack(cols)

        tri = np.where(tr)[0]
        for with_g, out in ((False, pb), (True, pf)):
            w = logit_fit(X(tri, with_g), pres[tri].ravel())
            out[s] = 1 / (1 + np.exp(-(X(np.array([s]), with_g, held_out=True) @ w)))
    bb = float(((pb - pres) ** 2).mean())
    bf = float(((pf - pres) ** 2).mean())
    return {"brier_base": bb, "brier_full": bf, "bss": 1 - bf / bb}


def skill_test(pres, prior, war, grn, n=NPERM_SKILL):
    obs = loso_skill(pres, prior, war, grn)
    null = np.array([loso_skill(pres, prior, war, grn[RNG.permutation(grn.shape[0])])["bss"]
                     for _ in range(n)])
    obs["p"] = float((null >= obs["bss"]).mean())
    obs["null_q90"] = float(np.quantile(null, 0.9))
    return obs


def clean(o):
    """NaN/inf -> None and numpy scalars -> Python, recursively (strict JSON for the page)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a2 = load_adm(2)
    unit_g, pz, npix = greening(a2)
    ev = events(a2)
    dar_units = sorted(a2.loc[a2["side"] == "darfur", "adm2_pcode"])
    gw = unit_g.pivot(index="year", columns="adm2_pcode", values="grn")
    gu = unit_g.pivot(index="year", columns="adm2_pcode", values="grn_up")
    yrs = sorted(set(SEASONS) | {y + 1 for y in SEASONS})
    test_units = [u for u in dar_units if u in gw.columns and gw.loc[yrs, u].notna().all()]
    up_units = [u for u in dar_units if u in gu.columns and gu.loc[yrs, u].notna().all()]
    print(f"[conflict] Darfur localities in the test: {len(test_units)} of {len(dar_units)} "
          f"(upstream: {len(up_units)})")

    def mat(seasons, wname, cls, presence=True, units=None):
        rows = []
        for y in seasons:
            c = count(ev, *window(wname, y), cls, units or test_units)
            rows.append((c > 0).astype(int).values if presence else c.values)
        return np.array(rows, dtype=float)

    G = lambda seasons, lag=0: gw.loc[[y + lag for y in seasons], test_units].values  # noqa: E731
    tests = []

    def add(label, key, seasons, wname, cls, xlag=0, kind="test", up=False, units=None):
        units = units or (up_units if up else test_units)
        y = mat(seasons, wname, cls, units=units)
        x = (gu if up else gw).loc[[s + xlag for s in seasons], units].values
        r2 = perm_test(x, y, two_way=True)
        r1 = perm_test(x, y, two_way=False)
        tests.append({"key": key, "label": label, "kind": kind, "window": wname, "cls": cls,
                      "xlag": xlag, "up": up, "seasons": [seasons[0], seasons[-1]], "n_seasons": len(seasons),
                      "n_units": len(units), "rate": float(y.mean()),
                      "two_way": r2, "unit_only": r1})
        print(f"  {key:<22} rate {y.mean():.2f}  two-way {r2['pp_per_10']:+.1f} pp/10pts "
              f"p={r2['p']:.3f}  unit-only {r1['pp_per_10']:+.1f} p={r1['p']:.3f}")

    print("[conflict] tests (pp = change in probability of >= 1 event per 10 points of greening lost)")
    add("Resource-type, Oct–Jan", "res_octjan", SEASONS, "oct_jan", "resource")
    add("Resource-type, Mar–Jun (pre-specified)", "res_marjun", SEASONS_LATE, "mar_jun", "resource")
    add("Resource-type, Jun–Jul (from seasonal cycle)", "res_junjul", SEASONS_LATE, "jun_jul", "resource")
    add("Communal, Oct–Jan", "com_octjan", SEASONS, "oct_jan", "communal", kind="second")
    add("Resource-type, larger localities only, Oct–Jan", "big_octjan", SEASONS, "oct_jan", "resource",
        kind="sensitivity", units=[u for u in test_units if npix[u] >= BIG_PIX])
    add("Resource-type, strict war filter, Oct–Jan", "ress_octjan", SEASONS, "oct_jan", "resource_strict",
        kind="sensitivity")
    add("War events, Oct–Jan (should show less)", "war_octjan", SEASONS, "oct_jan", "war", kind="falsification")
    add("Next season's greening (placebo), Oct–Jan", "plc_octjan", SEASONS, "oct_jan", "resource", xlag=1,
        kind="falsification")
    # added after the first results, from the literature (McGuirk & Nunn 2025: drought in herders' areas
    # raises violence in nearby farmland during the growing season)
    add("Resource-type, during the season (Jul–Oct)", "res_juloct", SEASONS, "jul_oct", "resource", kind="added")
    add("Upstream greening, resource-type, Jul–Oct", "up_juloct", SEASONS, "jul_oct", "resource", kind="added", up=True)
    add("Upstream greening, resource-type, Oct–Jan", "up_octjan", SEASONS, "oct_jan", "resource", kind="added", up=True)

    # per standard deviation of each locality's own greening (the unit published effects use)
    def per_sd(key, g, units, wname):
        t = next(t for t in tests if t["key"] == key)
        x = g.loc[SEASONS, units].values / g.loc[BASE[0]:BASE[1], units].std().values[None, :]
        r = perm_test(x, mat(SEASONS, wname, "resource", units=units), two_way=True)
        t["per_sd"] = {"pp": r["pp_per_10"] / 10, "lo": r["null_lo"] / 10, "hi": r["null_hi"] / 10, "p": r["p"]}
        print(f"  {key} per SD: {t['per_sd']['pp']:+.1f} pp (chance {t['per_sd']['lo']:+.1f}..{t['per_sd']['hi']:+.1f})")

    per_sd("res_octjan", gw, test_units, "oct_jan")
    per_sd("up_juloct", gu, up_units, "jul_oct")

    # out-of-sample skill, primary only
    pres = mat(SEASONS, "oct_jan", "resource")
    prior = mat(SEASONS, "prior", "resource")
    war = mat(SEASONS, "prior", "war", presence=False)  # previous 12 months: known when forecasting
    skill = skill_test(pres, prior, war, G(SEASONS))
    print(f"[conflict] LOSO skill: Brier base {skill['brier_base']:.4f} full {skill['brier_full']:.4f} "
          f"BSS {skill['bss']:+.3f} (perm p={skill['p']:.3f}, null q90 {skill['null_q90']:+.3f})")

    # ---------------------------------------------- per-season aggregates (all precision levels)
    def tot(side, start, end, cls=None):
        m = (ev["side"] == side) & (ev["date"] >= start) & (ev["date"] < end)
        if cls:
            m &= ev[cls]
        return int(m.sum())

    season_rows = []
    for y in range(2012, 2026):
        for side in ("darfur", "chad"):
            r = {"side": side, "y": y}
            for w in ("oct_jan", "mar_jun", "jun_jul"):
                s, e = window(w, y)
                if pd.Timestamp(e) > ev["date"].max() + pd.Timedelta(days=1):
                    continue
                for c in CLASSES:
                    r[f"{c}_{w}"] = tot(side, s, e, c)
                r[f"all_{w}"] = tot(side, s, e)
                if side == "darfur":
                    r[f"units_res_{w}"] = int((count(ev, s, e, "resource", test_units) > 0).sum())
            season_rows.append(r)
    seasons_df = pd.DataFrame(season_rows)

    yearly = (ev.groupby(["side", "year"])[CLASSES].sum().astype(int)
              .join(ev.groupby(["side", "year"]).size().rename("all")))
    # UCDP GED (independent coding; only conflicts reaching 25 deaths in a year are included):
    # non-state events between communal groups (dyads of rebel factions or militias dropped)
    ged = pd.read_csv(RAW / "ucdp" / "ged251-csv.zip", low_memory=False,
                      usecols=["year", "type_of_violence", "latitude", "longitude", "country", "dyad_name"])
    ged = ged[ged["country"].isin(["Chad", "Sudan"]) & (ged["type_of_violence"] == 2)
              & ~ged["dyad_name"].str.contains(r"SLM|JEM|SPLM|SRF|Janjaweed|PDF|RSF|Rapid Support|FACT|UFR",
                                               regex=True)]
    gp = gpd.sjoin(gpd.GeoDataFrame(ged, geometry=gpd.points_from_xy(ged["longitude"], ged["latitude"]), crs=4326),
                   a2[["side", "geometry"]], predicate="within")
    yearly = yearly.join(gp.groupby(["side", "year"]).size().rename("ucdp_nonstate"), how="outer")
    yearly["ucdp_nonstate"] = yearly["ucdp_nonstate"].fillna(0).where(
        yearly.index.get_level_values("year") <= 2024)  # GED 25.1 ends in 2024
    yearly = yearly.reset_index()
    yearly[CLASSES + ["all"]] = yearly[CLASSES + ["all"]].fillna(0).astype(int)
    ev["ym"] = ev["date"].dt.strftime("%Y-%m")
    monthly = (ev[ev["date"] >= "2023-01-01"].groupby(["side", "ym"])[CLASSES].sum().astype(int)
               .join(ev[ev["date"] >= "2023-01-01"].groupby(["side", "ym"]).size().rename("all"))
               .reset_index())
    pre = ev[(ev["side"] == "darfur") & (ev["date"] >= "2014-01-01") & (ev["date"] < "2023-01-01")]
    cycle = pre.groupby(pre["date"].dt.month)[CLASSES].sum().astype(int)
    cycle["all"] = pre.groupby(pre["date"].dt.month).size()

    # ---------------------------------------------- maps: propensity and 2026 greening
    prop = pd.DataFrame({"adm2_pcode": dar_units})
    pr = np.array([(count(ev, *window("oct_jan", y), "resource", dar_units) > 0).values for y in SEASONS])
    prop["share_res"] = pr.mean(0)
    prop["n_res"] = sum(count(ev, *window("oct_jan", y), "resource", dar_units).values for y in SEASONS)
    g26 = unit_g[unit_g["year"] == 2026].set_index("adm2_pcode")["grn"]
    gmin = unit_g[unit_g["year"].between(*BASE)].groupby("adm2_pcode")["grn"].min()
    units = a2[["side", "adm1_pcode", "adm1_name", "adm2_pcode", "adm2_name"]].copy()
    units["npix"] = units["adm2_pcode"].map(npix)
    units["grn_2026"] = units["adm2_pcode"].map(g26)
    units["grn_min_2001_2024"] = units["adm2_pcode"].map(gmin)
    units["in_test"] = units["adm2_pcode"].isin(test_units)
    units = units.merge(prop, on="adm2_pcode", how="left")
    since = ev[(ev["date"] >= "2026-10-01")]
    units["res_since_oct26"] = units["adm2_pcode"].map(since[since["resource"]].groupby("adm2_pcode").size()).fillna(0).astype(int)

    # range of the test seasons' Darfur-wide and per-locality greening vs 2026
    tg = gw.loc[SEASONS, test_units]
    rng = {"test_unit_min": float(np.nanmin(tg.values)), "test_unit_max": float(np.nanmax(tg.values)),
           "n_test_units": len(test_units),
           # typical year-to-year spread (SD over 2001-2024, median over localities), to put published
           # effects "per standard deviation" into this page's units
           "sd_local": float(gw.loc[BASE[0]:BASE[1], test_units].std().median()),
           "sd_up": float(gu.loc[BASE[0]:BASE[1], up_units].std().median())}

    # ---------------------------------------------- per-season comparison (one point per season)
    pub = json.loads(PAGE_DATA.read_text())
    veg = {(r["side"], r["y"]): r for r in pub["veg"]}
    rain = {(r["side"], r["y"]): r for r in pub["rain"]}
    pzd = dict(zip(pz["year"], pz["grn_pastoral"]))
    pts = []
    for y in range(2012, 2027):
        v, ra = veg.get(("darfur", y), {}), rain.get(("darfur", y), {})
        s = seasons_df[(seasons_df["side"] == "darfur") & (seasons_df["y"] == y)]
        r = {"y": y, "grn": v.get("grn"), "bio_d": v.get("bio_d"), "vhi_d": v.get("vhi_d"),
             "rain": ra.get("pt"), "grn_pastoral": None if not np.isfinite(pzd.get(y, np.nan)) else float(pzd[y]),
             "kind": "test" if y in SEASONS else ("war" if y >= 2023 else "early")}
        if len(s):
            r.update({k: (None if pd.isna(val) else int(val)) for k, val in s.iloc[0].items()
                      if k not in ("side", "y")})
        pts.append(r)
    sc = pd.DataFrame(pts)
    sc["share_res_oct_jan"] = 100 * sc["resource_oct_jan"] / sc["all_oct_jan"]
    sc_t = sc[sc["kind"] == "test"].copy()
    # every per-season panel the page offers (x by y), as plotted and with both series detrended over the
    # tested seasons; the later windows stop at SEASONS_LATE. Only the tally reaches the page (forking paths).
    def resid(df, k):
        return df[k] - np.polyval(np.polyfit(df["y"], df[k], 1), df["y"])

    late = {"resource_mar_jun", "resource_jun_jul"}
    panels = []
    for xk in ("grn", "grn_pastoral", "bio_d", "vhi_d", "rain"):
        for yk in ("resource_oct_jan", "share_res_oct_jan", "res_detr", "units_res_oct_jan", "communal_oct_jan",
                   "war_oct_jan", "resource_mar_jun", "resource_jun_jul"):
            base_y = "resource_oct_jan" if yk == "res_detr" else yk
            ok = sc_t[sc_t["y"] <= (SEASONS_LATE[-1] if yk in late else SEASONS[-1])].dropna(subset=[xk, base_y])
            yv = resid(ok, base_y) if yk == "res_detr" else ok[yk]
            panels.append({"x": xk, "y": yk, "n": len(ok), "p": float(spearmanr(ok[xk], yv)[1]),
                           "p_dt": float(spearmanr(resid(ok, xk), resid(ok, base_y))[1])})
    pn = pd.DataFrame(panels)
    panel_tally = {"n_stats": 2 * len(pn), "pass05": int((pn["p"] < 0.05).sum() + (pn["p_dt"] < 0.05).sum())}
    print(f"[conflict] per-season panels: {panel_tally['pass05']} of {panel_tally['n_stats']} rank correlations at p < 0.05")

    # ---------------------------------------------- CAST
    cast = pd.read_parquet(RAW / "acled" / "cast_tcd_sdn.parquet")
    cast = cast[cast["admin1"].isin(["Ouaddai", "Wadi Fira", "Sila", "North Darfur", "South Darfur",
                                     "West Darfur", "Central Darfur", "East Darfur"])].copy()
    cast["m"] = pd.to_datetime(cast["month"] + " " + cast["year"].astype(str), format="%B %Y")
    cast["issued"] = pd.to_datetime(cast["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    cast["side"] = cast["admin1"].map(lambda s: "darfur" if "Darfur" in s else "chad")
    cast_m = (cast.groupby(["side", "m"]).agg(fc=("total_forecast", "sum"), obs=("total_observed", "sum"),
                                              n=("admin1", "size"), issued=("issued", "max"))
              .reset_index())
    cast_m["m"] = cast_m["m"].dt.strftime("%Y-%m")
    cast_last_issue = str(cast["issued"].max())

    data = {
        "built": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "acled_download": str(ev["download_date"].iloc[0]) if "download_date" in ev else None,
        "acled_last_event": ev["date"].max().strftime("%Y-%m-%d"),
        "design": {"seasons": [SEASONS[0], SEASONS[-1]], "seasons_late": [SEASONS_LATE[0], SEASONS_LATE[-1]],
                   "min_pix": MIN_PIX, "pastoral_max_mm": PASTORAL_MAX_MM, "nperm": NPERM,
                   "up_lat": list(UP_LAT), "up_lon": UP_LON, "n_units_up": len(up_units),
                   "nperm_skill": NPERM_SKILL, "n_units_darfur": len(dar_units), "n_units_test": len(test_units)},
        "tests": tests, "skill": skill, "range": rng, "panels": panel_tally,
        "season": json.loads(sc.to_json(orient="records")),
        "yearly": json.loads(yearly.to_json(orient="records")),
        "monthly": json.loads(monthly.to_json(orient="records")),
        "cycle": json.loads(cycle.reset_index(names="m").to_json(orient="records")),
        "units": json.loads(units.to_json(orient="records")),
        "cast": json.loads(cast_m.to_json(orient="records")), "cast_last_issue": cast_last_issue,
    }
    (OUT / "conflict_data.json").write_text(json.dumps(clean(data), allow_nan=False))
    print(f"[conflict] wrote {OUT / 'conflict_data.json'}")


if __name__ == "__main__":
    main()
