"""Pooled Sudano-Sahelian test: poor seasons and resource-type violence in 0.5-degree cells.

Restricted (ACLED): output stays in <work>/processed/belt/belt_data.json, never committed.
The design is fixed in <work>/processed/belt/prespec_2026-11-18.md (written before any event was
joined with an indicator); this script implements it and nothing else is claimed from it.

Inputs: 13_belt_cells.py (cells, cell_years), 10_acled.py (tcd_sdn + belt), fetch_belt_veg.sh
(ASAP + ASIS admin-1 series) with the providers' unit boundaries (ASAP gaul1_asap_v06, FAO GAUL 2015).

Model: presence (0/1) of at least one event in the cell and window; slope on the predictor after
removing cell means and country x season means (alternating projections on the unbalanced panel);
inference by permuting whole seasons of the predictor (each season's map kept). Predictors in units
of the cell's own 2001-2024 standard deviation, signed so that + = poorer conditions.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW  # noqa: E402
from conflict_classes import classify  # noqa: E402

OUT = PROC / "belt"
SEASONS = list(range(2010, 2025))
SDN_LAST = 2022
ATTR_MAX = 0.5
EXTREME = 60.0
BIG_PIX = 100
NPERM_P = 2000      # primaries
NPERM_S = 2000      # everything else (the plan: 2,000 draws throughout)
NPERM_TALLY = 1000  # correlation-matrix tally
SDN_WAR = "2023-04-15"  # Sudan: no window ending after the start of the war
MIN_EVENTS_COUNTRY = 50
BASE = (2001, 2024)
RNG = np.random.default_rng(20261118)
if __import__("os").environ.get("BELT_SMOKE"):  # quick end-to-end check, not for the page
    NPERM_P = NPERM_S = NPERM_TALLY = 20
ISO = {"Mauritania": "MRT", "Senegal": "SEN", "Gambia": "GMB", "Mali": "MLI", "Burkina Faso": "BFA",
       "Niger": "NER", "Nigeria": "NGA", "Cameroon": "CMR", "Central African Republic": "CAF",
       "South Sudan": "SSD", "Chad": "TCD", "Sudan": "SDN"}
NAME = {v: k for k, v in ISO.items()}
CLASSES = ["resource", "resource_strict", "communal", "war"]


def window(name, y):
    return {"jul_oct": (f"{y}-07-01", f"{y}-11-01"), "oct_jan": (f"{y}-10-01", f"{y + 1}-02-01"),
            "mar_jun": (f"{y + 1}-03-01", f"{y + 1}-07-01")}[name]


def window_years(name, y):
    return {"jul_oct": [y], "oct_jan": [y, y + 1], "mar_jun": [y + 1]}[name]


def cell_id(lat, lon):
    return f"{np.floor(lat / .5) * .5 + .25:+06.2f}_{np.floor(lon / .5) * .5 + .25:+07.2f}"


# ------------------------------------------------------------------ data
def load_events(belt_cells):
    ev = pd.concat([pd.read_parquet(RAW / "acled" / "acled_tcd_sdn.parquet"),
                    pd.read_parquet(RAW / "acled" / "acled_belt.parquet")], ignore_index=True)
    a = classify(ev, area=None)
    a["cell"] = [cell_id(float(y), float(x)) for y, x in zip(a["latitude"], a["longitude"])]
    a = a[a["cell"].isin(belt_cells) & (a["geo_precision"].astype(int) < 3)].copy()
    a["iso3_ev"] = a["country"].map(ISO)
    return a


def attribution(ev):
    """Share (and number) of farmer/pastoralist-tagged or keyword events that the war filter removes, by country-year."""
    k = ev[ev["farmherd"] | ev["kw"]]
    g = k.groupby(["iso3_ev", "year"])["war"]
    return g.mean(), g.size()


def admin1_series(cells):
    """ASAP (crop, range: 21 Oct, value minus its 2001-2024 line) and ASIS (ASI 21 Sep; VHI Jul-Oct mean,
    minus line) per cell, mapped to the providers' own units by majority area."""
    g = gpd.GeoDataFrame(cells[["cell"]], geometry=gpd.points_from_xy(cells["lon"], cells["lat"]).buffer(0.25, cap_style=3),
                         crs=4326)
    out = {}
    # ASAP
    asap_u = gpd.read_file(f"zip://{RAW / 'belt' / 'asap' / 'gaul1_asap_v06.zip'}!gaul1_asap_v06.shp",
                           bbox=(-18, 3.5, 38.5, 19))[["asap1_id", "geometry"]]
    m = _majority(g, asap_u, "asap1_id")
    rows = []
    for f in sorted((RAW / "belt" / "asap").glob("*_zfparc_*.csv")):
        d = pd.read_csv(f)
        d["md"] = d["date"].astype(str).str[4:]
        d = d[d["md"] == "1021"].assign(year=lambda x: x["date"].astype(str).str[:4].astype(int))
        d["var"] = "asap_crop" if "crop" in f.stem else "asap_range"
        rows.append(d[["region_id", "year", "var", "value"]])
    asap = pd.concat(rows).drop_duplicates(["region_id", "year", "var"])
    for var in ("asap_crop", "asap_range"):
        p = asap[asap["var"] == var].pivot(index="year", columns="region_id", values="value")
        out[var] = _to_cells(_detrend(p), m)
    # ASIS (GAUL 2015 admin-1 codes)
    gaul = gpd.read_file(f"zip://{RAW / 'belt' / 'gaul' / 'g2015_2014_1.zip'}!g2015_2014_1.shp", bbox=(-18, 3.5, 38.5, 19))
    code = next(c for c in gaul.columns if c.upper() == "ADM1_CODE")
    mg = _majority(g, gaul[[code, "geometry"]].rename(columns={code: "ADM1_CODE"}), "ADM1_CODE")
    asi, vhi = [], []
    for f in sorted((RAW / "belt" / "asis").glob("*_ASI_Dekad_Season1_data.csv")):
        d = pd.read_csv(f, skipinitialspace=True, encoding="latin-1")  # FAO files are not UTF-8
        d["Date"] = pd.to_datetime(d["Date"])
        d = d[(d["Date"].dt.month == 9) & (d["Date"].dt.day == 21)]
        asi.append(d.assign(year=d["Date"].dt.year)[["ADM1_CODE", "year", "Data"]])
    for f in sorted((RAW / "belt" / "asis").glob("*_vhi_adm1_dekad_data.csv")):
        d = pd.read_csv(f, skipinitialspace=True, encoding="latin-1")  # FAO files are not UTF-8
        d["Date"] = pd.to_datetime(d["Date"])
        d = d[d["Date"].dt.month.isin([7, 8, 9, 10])]
        vhi.append(d.assign(year=d["Date"].dt.year).groupby(["ADM1_CODE", "year"], as_index=False)["Data"].mean())
    if asi:
        a = pd.concat(asi).drop_duplicates(["ADM1_CODE", "year"]).pivot(index="year", columns="ADM1_CODE", values="Data")
        out["asi"] = _to_cells(a, mg)            # bounded share: not detrended (as on the public page)
    if vhi:
        v = pd.concat(vhi).drop_duplicates(["ADM1_CODE", "year"]).pivot(index="year", columns="ADM1_CODE", values="Data")
        out["vhi"] = _to_cells(_detrend(v), mg)
    return out


def _majority(g, units, key):
    ov = gpd.overlay(g, units, how="intersection", keep_geom_type=True)
    ov["a"] = ov.geometry.area
    return ov.sort_values("a").groupby("cell").tail(1).set_index("cell")[key]


def _detrend(p):
    yrs = p.index.values.astype(float)
    out = p.copy()
    for c in p.columns:
        s = p[c].values.astype(float)
        m = (yrs >= BASE[0]) & (yrs <= BASE[1]) & np.isfinite(s)
        out[c] = s - np.polyval(np.polyfit(yrs[m], s[m], 1), yrs) if m.sum() >= 10 else np.nan
    return out


def _to_cells(p, mapping):
    """year x unit table -> year x cell table through the cell -> unit mapping."""
    cols = {c: p[u] for c, u in mapping.items() if u in p.columns}
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ statistics
def demean(v, groups, iters=200, tol=1e-9):
    r = v.astype(float).copy()
    for _ in range(iters):
        mx = 0.0
        for g, n in groups:
            m = np.bincount(g, weights=r, minlength=len(n)) / n
            r -= m[g]
            mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return r


class Panel:
    """Rows = kept (cell, season) with a complete predictor; fixed effects; permutation of seasons."""

    def __init__(self, Y, K, X, cells, fe="country", lag=0, seasons=None, C=None):
        seasons = seasons or SEASONS
        Y, K = Y.loc[seasons], K.loc[seasons]
        C = C.loc[seasons] if C is not None else None
        S = len(seasons)
        Xs = X.reindex(index=[s + lag for s in seasons], columns=Y.columns).T.values  # cells x seasons
        ok_cell = np.isfinite(Xs).all(1)
        ci, si = np.where(K.T.values & ok_cell[:, None])
        self.ci, self.si = ci, si
        self.y = Y.T.values[ci, si].astype(float)
        self.Xs = Xs
        meta = cells.set_index("cell").loc[Y.columns]
        unit = (meta["iso3"] if fe == "country" else meta["adm1_pcode"]).values
        g1 = pd.factorize(ci)[0]
        g2 = pd.factorize(pd.Series(unit[ci]).astype(str) + "_" + pd.Series(si).astype(str))[0]
        self.groups = [(g1, np.bincount(g1)), (g2, np.bincount(g2))]
        self.yr = demean(self.y, self.groups)
        self.n, self.S = len(self.y), S
        self.n_cells = len(np.unique(ci))
        self.rate = float(self.y.mean()) if len(self.y) else float("nan")
        self.events = int(self.y.sum())                      # cell-seasons with at least one event
        self.n_events = int(C.T.values[ci, si].sum()) if C is not None else None   # events in those rows
        iso = meta["iso3"].values[ci]
        self.iso_events = {k: int(self.y[iso == k].sum()) for k in np.unique(iso)}

    def slope(self, perm=None):
        s = self.si if perm is None else perm[self.si]
        x = demean(self.Xs[self.ci, s], self.groups)
        return float((x * self.yr).sum() / (x * x).sum())

    def test(self, nperm):
        b = self.slope()
        null = np.array([self.slope(RNG.permutation(self.S)) for _ in range(nperm)])
        q = np.quantile(null, [0.025, 0.975])
        # smallest effect found 4 times in 5 at the plan's two-sided p < 0.025 (normal approximation to the null)
        mde80 = (2.241 + 0.842) * null.std()
        return {"pp": 100 * b, "lo": 100 * q[0], "hi": 100 * q[1], "p": float((np.abs(null) >= abs(b)).mean()),
                "null_sd": 100 * float(null.std()), "mde80": 100 * float(mde80),
                "n": self.n, "cells": self.n_cells, "rate": self.rate, "events": self.events,
                "n_events": self.n_events, "iso_events": self.iso_events, "nperm": nperm}


def scaled(X, sign=-1.0):
    """year x cell table -> + = poorer, in units of each cell's own 2001-2024 standard deviation."""
    sd = X.loc[BASE[0]:BASE[1]].std()
    return sign * X / sd.where(sd > 0)


# ------------------------------------------------------------------ main
def main():
    cells = pd.read_parquet(OUT / "cells.parquet")
    cells = cells[cells["in_belt"]].reset_index(drop=True)
    cy = pd.read_parquet(OUT / "cell_years.parquet")
    belt = set(cells["cell"])
    ev = load_events(belt)
    attr, attr_n = attribution(ev)
    print(f"[belt] {len(cells)} cells, {len(ev)} events placed in belt cells")

    # outcome matrices (season x cell) and kept masks per window
    cols = cells["cell"].values
    iso_of = dict(zip(cells["cell"], cells["iso3"]))

    def counts(wname, cls):
        rows = []
        for y in SEASONS:
            s, e = window(wname, y)
            m = (ev["date"] >= s) & (ev["date"] < e) & ev[cls]
            rows.append(ev[m].groupby("cell").size().reindex(cols, fill_value=0).values)
        return pd.DataFrame(rows, index=SEASONS, columns=cols)

    def kept(wname, drop_iso=(), attr_floor=0):
        k = np.zeros((len(SEASONS), len(cols)), bool)
        for i, y in enumerate(SEASONS):
            war_end = window(wname, y)[1] > SDN_WAR
            for j, c in enumerate(cols):
                iso = iso_of[c]
                if iso in drop_iso or (iso == "SDN" and war_end):
                    continue
                k[i, j] = all(attr.get((iso, yy), 0.0) <= ATTR_MAX or attr_n.get((iso, yy), 0) < attr_floor
                              for yy in window_years(wname, y))
        return pd.DataFrame(k, index=SEASONS, columns=cols)

    Cmat = {(w, c): counts(w, c) for w in ("jul_oct", "oct_jan", "mar_jun") for c in CLASSES}
    Ymat = {k: (v > 0).astype(int) for k, v in Cmat.items()}
    Kmat = {w: kept(w) for w in ("jul_oct", "oct_jan", "mar_jun")}
    piv = {v: cy.pivot(index="year", columns="cell", values=v) for v in ("grn", "grn_up", "rain", "rain_up")}
    adm1 = admin1_series(cells)
    piv.update(adm1)
    X = {v: scaled(p, sign=(+1.0 if v == "asi" else -1.0)) for v, p in piv.items()}
    X10 = {v: -piv[v] / 10 for v in ("grn", "grn_up")}                      # per 10 points lost
    Xext = {"grn": (piv["grn"] < EXTREME).astype(float).where(piv["grn"].notna())}

    tests = []

    def run(key, label, kind, w, cls, xv, nperm=NPERM_S, Xsrc=None, lag=0, fe="country", cell_mask=None,
            drop_iso=(), unit="sd", seasons=None, attr_floor=0):
        Y = Ymat[(w, cls)]
        K = kept(w, drop_iso, attr_floor) if (drop_iso or attr_floor) else Kmat[w]
        if cell_mask is not None:
            K = K & pd.DataFrame(np.repeat(cell_mask[None, :], len(SEASONS), 0), index=SEASONS, columns=cols)
        P = Panel(Y, K, (Xsrc or X)[xv], cells, fe=fe, lag=lag, seasons=seasons, C=Cmat[(w, cls)])
        if P.n == 0 or P.events == 0:
            return None
        ss = seasons or SEASONS
        r = P.test(nperm) | {"key": key, "label": label, "kind": kind, "window": w, "cls": cls, "x": xv,
                             "lag": lag, "fe": fe, "drop": list(drop_iso), "unit": unit, "seasons": [ss[0], ss[-1]]}
        tests.append(r)
        print(f"  {key:<26} {r['pp']:+6.2f} pp ({r['lo']:+.2f}..{r['hi']:+.2f}) p={r['p']:.3f} "
              f"n={r['n']} cells={r['cells']} events={r['events']}", flush=True)
        return r

    print("[belt] primaries")
    p1 = run("P1", "Upstream greening, resource-type, Jul–Oct", "primary", "jul_oct", "resource", "grn_up", NPERM_P)
    p2 = run("P2", "Own greening, resource-type, Oct–Jan", "primary", "oct_jan", "resource", "grn", NPERM_P)
    run("P1_10", "Upstream greening, per 10 points", "primary_units", "jul_oct", "resource", "grn_up", NPERM_S, X10, unit="10pts")
    run("P2_10", "Own greening, per 10 points", "primary_units", "oct_jan", "resource", "grn", NPERM_S, X10, unit="10pts")
    print("[belt] secondary")
    run("S_rain_own", "Own rain, resource-type, Oct–Jan", "secondary", "oct_jan", "resource", "rain")
    run("S_rain_own_jo", "Own rain, resource-type, Jul–Oct", "secondary", "jul_oct", "resource", "rain")
    run("S_rain_up", "Upstream rain, resource-type, Jul–Oct", "secondary", "jul_oct", "resource", "rain_up")
    run("S_rain_up_oj", "Upstream rain, resource-type, Oct–Jan", "secondary", "oct_jan", "resource", "rain_up")
    run("S_comm", "Own greening, communal, Oct–Jan", "secondary", "oct_jan", "communal", "grn")
    run("S_marjun", "Own greening, resource-type, Mar–Jun", "secondary", "mar_jun", "resource", "grn")
    for v, lab in (("asap_crop", "JRC cropland biomass"), ("asap_range", "JRC rangeland biomass"),
                   ("asi", "FAO cropland stress (ASI)"), ("vhi", "FAO vegetation health")):
        if v in X:
            run(f"S_{v}_oj", f"{lab}, resource-type, Oct–Jan", "secondary", "oct_jan", "resource", v)
            run(f"S_{v}_jo", f"{lab}, resource-type, Jul–Oct", "secondary", "jul_oct", "resource", v)
    print("[belt] sensitivity")
    big = cells["npix"].values >= BIG_PIX
    for base, w, xv in (("P1", "jul_oct", "grn_up"), ("P2", "oct_jan", "grn")):
        run(f"{base}_noSDN", f"{base} without Sudan", "sensitivity", w, "resource", xv, drop_iso=("SDN",))
        run(f"{base}_adm1", f"{base}, admin-1 × season effects", "sensitivity", w, "resource", xv, fe="adm1")
        run(f"{base}_strict", f"{base}, strict war filter", "sensitivity", w, "resource_strict", xv)
        run(f"{base}_big", f"{base}, larger vegetated cells", "sensitivity", w, "resource", xv, cell_mask=big)
        run(f"{base}_attr10", f"{base}, attribution rule only on 10+ events", "sensitivity", w, "resource", xv,
            attr_floor=10)
    print("[belt] checks")
    run("C_war_P1", "War events, P1 set-up (should show less)", "check", "jul_oct", "war", "grn_up")
    run("C_war_P2", "War events, P2 set-up (should show less)", "check", "oct_jan", "war", "grn")
    run("C_placebo", "Next season's greening (placebo), Oct–Jan", "check", "oct_jan", "resource", "grn", lag=1,
        seasons=SEASONS[:-1])  # 2025 has no MODIS season
    print("[belt] extreme seasons")
    ex = run("E_extreme", f"Own greening below {EXTREME:.0f}% of trend, Oct–Jan", "extreme", "oct_jan", "resource",
             "grn", NPERM_S, Xext, unit="indicator")

    # unique resource-type events in the rows the two primaries test, per country
    Kp2 = Kmat["oct_jan"]
    tested = set()
    for w, xv in (("jul_oct", "grn_up"), ("oct_jan", "grn")):
        complete = set(piv[xv].reindex(index=SEASONS).columns[piv[xv].reindex(index=SEASONS).notna().all()])
        for y in SEASONS:
            s, e = window(w, y)
            keep_cells = set(Kmat[w].columns[Kmat[w].loc[y].values]) & complete
            tested |= set(ev.index[(ev["date"] >= s) & (ev["date"] < e) & ev["resource"] & ev["cell"].isin(keep_cells)])
    ev_country = ev.loc[sorted(tested), "cell"].map(iso_of).value_counts().reindex(sorted(set(cells["iso3"])),
                                                                                   fill_value=0).astype(int).to_dict()
    big_iso = [i for i, n in ev_country.items() if n >= MIN_EVENTS_COUNTRY]
    with_events = sorted({k for t in (p1, p2) for k, v in t["iso_events"].items() if v > 0})
    print("[belt] leave one country out:", with_events, "| per country:", big_iso)
    loo, per = [], []
    for iso in with_events:
        for base, w, xv in (("P1", "jul_oct", "grn_up"), ("P2", "oct_jan", "grn")):
            r = run(f"{base}_wo_{iso}", f"{base} without {NAME[iso]}", "loo", w, "resource", xv, drop_iso=(iso,))
            if r:
                loo.append(r | {"iso3": iso, "base": base})
            if iso in big_iso:
                r = run(f"{base}_only_{iso}", f"{base}, {NAME[iso]} only", "country", w, "resource", xv,
                        cell_mask=np.array([iso_of[c] == iso for c in cols]))
                if r:
                    per.append(r | {"iso3": iso, "base": base})
    # where own greening is missing: resource-type events 2010-2025 in cells with / without a complete own series
    complete_own = set(piv["grn"].reindex(index=SEASONS).columns[piv["grn"].reindex(index=SEASONS).notna().all()])
    e10 = ev[ev["resource"] & ev["date"].between("2010-01-01", "2025-12-31")]
    coverage = {iso: {"events": int(n), "in_own": int(e10[(e10["cell"].map(iso_of) == iso) & e10["cell"].isin(complete_own)].shape[0])}
                for iso, n in e10["cell"].map(iso_of).value_counts().items()}

    # what the pre-specification lets the page say about 2026
    def shows(r):
        return r is not None and r["pp"] > 0 and r["p"] < 0.025
    lic = {"p1": shows(p1), "p2": shows(p2), "extreme": ex["pp"] > 0 and ex["p"] < 0.05}
    for base in ("P1", "P2"):
        rr = [t for t in tests if t["key"] == f"{base}_noSDN"] + [t for t in loo if t["base"] == base and t["iso3"] in big_iso]
        lic[f"{base}_signs_hold"] = all(t["pp"] > 0 for t in rr)
    lic["license"] = bool((lic["p1"] and lic["P1_signs_hold"] or lic["p2"] and lic["P2_signs_hold"]) and lic["extreme"])
    print("[belt] 2026 licence:", lic)

    # correlation matrix on values demeaned by cell and country x season (pooled, without Sudan, per country)
    mvars = [("grn", "Greening (own)"), ("grn_up", "Greening (upstream)"), ("rain", "Rain (own)"),
             ("rain_up", "Rain (upstream)")] + [(v, l) for v, l in (("asap_crop", "JRC cropland biomass"),
             ("asap_range", "JRC rangeland biomass"), ("asi", "FAO ASI (cropland stressed)"),
             ("vhi", "FAO vegetation health")) if v in piv]
    outs = [(("jul_oct", "resource"), "Resource-type, Jul–Oct"), (("oct_jan", "resource"), "Resource-type, Oct–Jan"),
            (("oct_jan", "communal"), "Communal, Oct–Jan"), (("oct_jan", "war"), "Wider wars, Oct–Jan")]
    K = Kmat["oct_jan"] & Kmat["jul_oct"]
    ci, si = np.where(K.T.values)
    meta = cells.set_index("cell").loc[cols]
    iso_r = meta["iso3"].values[ci]

    def resid(vals, sel):
        ok = np.isfinite(vals) & sel
        out = np.full(len(vals), np.nan)
        if ok.sum() < 30:
            return out
        g1 = pd.factorize(ci[ok])[0]
        g2 = pd.factorize(pd.Series(iso_r[ok]).astype(str) + "_" + pd.Series(si[ok]).astype(str))[0]
        out[ok] = demean(vals[ok], [(g1, np.bincount(g1)), (g2, np.bincount(g2))])
        return out

    raw = {}
    for v, _ in mvars:
        Xv = piv[v].reindex(index=SEASONS, columns=cols).T.values
        raw[v] = Xv[ci, si]
    for (wc, lab) in outs:
        raw[lab] = Ymat[wc].T.values[ci, si].astype(float)
    names = [l for _, l in mvars] + [l for _, l in outs]
    keys = [v for v, _ in mvars] + [l for _, l in outs]
    matrices = {}
    for setname, sel in [("all", np.ones(len(ci), bool)), ("noSDN", iso_r != "SDN")] + \
                        [(iso, iso_r == iso) for iso in big_iso]:
        R = {k: resid(raw[k], sel) for k in keys}
        M = np.full((len(keys), len(keys)), np.nan)
        N = np.zeros((len(keys), len(keys)), int)
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                ok = np.isfinite(R[a]) & np.isfinite(R[b])
                N[i, j] = int(ok.sum())
                if ok.sum() >= 30 and np.nanstd(R[a][ok]) > 0 and np.nanstd(R[b][ok]) > 0:
                    M[i, j] = spearmanr(R[a][ok], R[b][ok])[0]
        matrices[setname] = {"r": np.round(M, 3).tolist(), "n": N.tolist()}
        if setname in ("all", "noSDN"):   # tally: indicator x event entries with season-permutation p < 0.05
            ps = []
            for v, _ in mvars:
                Xv = piv[v].reindex(index=SEASONS, columns=cols).T.values
                for (_, lab) in outs:
                    ok0 = np.isfinite(R[v]) & np.isfinite(R[lab])
                    if ok0.sum() < 30:
                        continue
                    obs = spearmanr(R[v][ok0], R[lab][ok0])[0]
                    null = []
                    for _ in range(NPERM_TALLY):
                        rv = resid(Xv[ci, RNG.permutation(len(SEASONS))[si]], sel)
                        ok = np.isfinite(rv) & np.isfinite(R[lab])
                        null.append(spearmanr(rv[ok], R[lab][ok])[0])
                    ps.append(float((np.abs(np.array(null)) >= abs(obs)).mean()))
            matrices[setname]["tally"] = {"pass05": int(sum(q < 0.05 for q in ps)), "n": len(ps), "nperm": NPERM_TALLY}
            print(f"  matrix {setname}: {matrices[setname]['tally']}", flush=True)

    # maps: resource-type propensity (share of kept seasons with an event, Oct-Jan) and 2026 greening per cell
    Y = Ymat[("oct_jan", "resource")].where(Kp2)
    cmap = pd.DataFrame({"cell": cols, "lat": meta["lat"].values, "lon": meta["lon"].values, "iso3": meta["iso3"].values,
                         "share": Y.mean().values, "nseas": Kp2.sum().values,
                         "grn26": piv["grn"].reindex(index=[2026]).iloc[0].reindex(cols).values if 2026 in piv["grn"].index else np.nan})

    # country outlines for the page's map (d3-geo wants clockwise rings)
    from shapely.geometry.polygon import orient
    from shapely.geometry import MultiPolygon, Polygon
    a0 = gpd.read_parquet(OUT / "adm1.parquet").dissolve("iso3").reset_index()
    a0["geometry"] = a0.geometry.simplify(0.05)

    def cw(geom):
        if isinstance(geom, Polygon):
            return orient(geom, sign=-1.0)
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([orient(g, sign=-1.0) for g in geom.geoms])
        return geom
    a0["geometry"] = a0.geometry.apply(cw)
    adm0 = json.loads(a0[["iso3", "geometry"]].to_json())

    data = {"built": pd.Timestamp.today().strftime("%Y-%m-%d"), "adm0": adm0,
            "coverage": coverage, "with_events": with_events,
            "design": {"seasons": [SEASONS[0], SEASONS[-1]], "sdn_last": SDN_LAST, "sdn_war": SDN_WAR, "attr_max": ATTR_MAX,
                       "nperm_tally": NPERM_TALLY,
                       "extreme": EXTREME, "big_pix": BIG_PIX, "nperm_p": NPERM_P, "nperm_s": NPERM_S,
                       "min_events_country": MIN_EVENTS_COUNTRY, "n_cells": len(cells),
                       "countries": sorted(set(cells["iso3"]))},
            "names": NAME, "ev_country": ev_country, "tests": tests, "loo": loo, "per_country": per,
            "licence": lic, "matrix": {"names": names, "n_pred": len(mvars), "sets": matrices},
            "attr": {f"{k[0]}_{k[1]}": round(float(v), 3) for k, v in attr.items() if 2010 <= k[1] <= 2025},
            "cells": json.loads(cmap.to_json(orient="records"))}
    (OUT / "belt_data.json").write_text(json.dumps(_clean(data), allow_nan=False))
    print(f"[belt] wrote {OUT / 'belt_data.json'}")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    main()
