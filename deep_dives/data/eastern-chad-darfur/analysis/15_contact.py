"""Contact-zone test: McGuirk & Nunn's nearest-neighbour design in the belt, and a bridge to this page's outcome.

Plan: <work>/processed/belt/prespec_2026-11-19_contact.md (written before any outcome was joined with the
contact measures). Inputs, all in the work folder:
  raw/mcguirk_nunn/x/EMG_NN_THPCCC_replication_data_072024.dta   McGuirk & Nunn (2025) replication data,
      Zenodo record 13308366 v2.1, CC-BY 4.0: own / nearest-neighbour ethnic territory scores per 0.5 deg cell
  raw/mcguirk_nunn/n1_identity.parquet   nearest neighbour per cell, recovered by matching each cell's
      neighbour-rain series to the one territory with the identical own-rain series (see prespec)
  raw/belt/chirps_v2_grid05.nc, processed/belt/{cells,cell_years}.parquet, processed/belt/greening.npz
  raw/acled/*.parquet, raw/ucdp/GEDEvent_v25_1.csv, raw/ucdp/nonstate/NonState_v25_1.csv

Model (their Table 2): y ~ Rn + Rn*Tn + Rg + Rg*Tg + Rc + Rc*Tg | cell + country x season, where R = rain in the
nearest neighbouring territory (n), own territory (g) and own cell (c), T = transhumant-pastoralism score.
Statistic: gamma1 (Rn*Tn). Inference: season permutation of the whole rain map (all R move together).

Modes: --null  C1 permutation null only (detectable effect; the estimate is not computed)
       (none)  everything; writes <work>/processed/belt/contact_data.json
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoi import PROC, RAW  # noqa: E402

B = importlib.import_module("14_belt_conflict")   # events, windows, attribution, demeaning, constants

OUT = PROC / "belt"
MN = RAW / "mcguirk_nunn"
SEASONS = list(range(2010, 2025))
SEASONS_MN = list(range(1997, 2020))   # their ACLED years (Table 2: 7,722 cells x 23 years)
NPERM = 2000
HALF = 0.5                             # median score of contact neighbours
RNG = np.random.default_rng(20261119)
MIN_TERR_PIX = 100
PUBLISHED = {"ucdp_all_10": -0.0110, "acled_all_10": -0.0096, "acled_nonstate_10": -0.0040}
PUB_SE = {"ucdp_all_10": 0.0033, "acled_all_10": 0.0038, "acled_nonstate_10": 0.0037}
PUB_PCT = {"ucdp_all_10": -37.51, "acled_all_10": -13.60, "acled_nonstate_10": -7.99}
if __import__("os").environ.get("BELT_SMOKE"):
    NPERM = 20


def cid(lat, lon):
    return f"{lat:+06.2f}_{lon:+07.2f}"


# ------------------------------------------------------------------ data
def mn_frame():
    """Their cell-year data (cached columns)."""
    f = MN / "mn_calib.parquet"
    if not f.exists():
        cols = ["cell", "x", "y", "MAP_name", "year", "country", "KG_climate", "ucdp_all_10", "acled_all_10",
                "acled_nonstate_10", "prec_gpcc", "g_prec_gpcc", "n1_g_prec_gpcc", "herdXEApXn12",
                "n1_herdXEApXn12", "herdXEApXn1234", "n1_herdXEApXn1234", "n1_prec_gpcc_X_herdXEApXn12",
                "g_prec_gpcc_X_herdXEApXn12", "prec_gpcc_X_herdXEApXn12"]
        pd.read_stata(MN / "x" / "EMG_NN_THPCCC_replication_data_072024.dta", columns=cols).to_parquet(f, index=False)
    d = pd.read_parquet(f)
    d["year"] = d["year"].astype(int)
    return d


def neighbour_identity(d):
    """Each cell's nearest neighbouring territory: the one territory whose own-rain series (g_prec_gpcc,
    1989-2018) equals the cell's neighbour-rain series (n1_g_prec_gpcc). Cached in n1_identity.parquet."""
    f = MN / "n1_identity.parquet"
    if not f.exists():
        s = d[(d["year"] >= 1989) & (d["year"] <= 2018)]
        G = s.drop_duplicates(["MAP_name", "year"]).pivot(index="MAP_name", columns="year", values="g_prec_gpcc").round(4)
        N = s.drop_duplicates(["cell", "year"]).pivot(index="cell", columns="year", values="n1_g_prec_gpcc").round(4)
        key = {tuple(r): k for k, r in zip(G.index, G.values)}
        if len(key) != len(G):
            raise SystemExit("two territories share a rain series: the neighbour cannot be identified")
        pd.Series([key.get(tuple(r)) for r in N.values], index=N.index, name="n1_name").to_frame().to_parquet(f)
    return pd.read_parquet(f)["n1_name"]


def chirps_fields(mcell):
    """CHIRPS June-September and calendar-year totals (mm) per replication cell, 1981-2025."""
    ds = xr.open_dataset(RAW / "belt" / "chirps_v2_grid05.nc").sortby("lat")
    af = ds["precip"].sel(lat=slice(-36, 38), lon=slice(-18, 52), time=slice("1981-01-01", "2025-12-31")).load()
    pr = af.sel(lat=xr.DataArray(mcell["y"].values, dims="p"), lon=xr.DataArray(mcell["x"].values, dims="p"))
    t = pd.DatetimeIndex(pr.time.values)
    v = pr.values                                                       # time x cells
    yrs = np.arange(1981, 2026)
    jjas = np.stack([v[(t.year == y) & t.month.isin([6, 7, 8, 9])].sum(0) for y in yrs])
    ann = np.stack([v[t.year == y].sum(0) if (t.year == y).sum() == 12 else np.full(v.shape[1], np.nan) for y in yrs])
    allnan = np.isnan(v).all(0)
    jjas[:, allnan] = np.nan
    ann[:, allnan] = np.nan
    return {k: pd.DataFrame(a, index=yrs, columns=mcell["cell"].values) for k, a in (("jjas", jjas), ("ann", ann))}


def territory_fields(cellfield, mcell):
    """year x cell table of territory means: own territory (g) and nearest neighbouring territory (n)."""
    terr = cellfield.T.groupby(mcell.set_index("cell").loc[cellfield.columns, "MAP_name"].values).mean().T
    own = mcell.set_index("cell").loc[cellfield.columns, "MAP_name"]
    nb = mcell.set_index("cell").loc[cellfield.columns, "n1_name"]
    g = pd.DataFrame({c: terr[t] if t in terr else np.nan for c, t in own.items()}, index=cellfield.index)
    n = pd.DataFrame({c: terr[t] if isinstance(t, str) and t in terr else np.nan for c, t in nb.items()},
                     index=cellfield.index)
    return g, n


def greening_fields(mcell):
    """Territory greening (% of each territory's own 2001-2024 line, recipe of 13_belt_cells.py) for own and
    neighbouring territories; territories need MIN_TERR_PIX vegetated 4 km pixels inside the belt box."""
    z = np.load(PROC / "belt" / "greening.npz")
    gyears, glat, glon, gr = z["years"], z["lat"], z["lon"], z["greening"]
    base = (gyears >= 2001) & (gyears <= 2024)
    with np.errstate(invalid="ignore", divide="ignore"):
        gmean = np.nanmean(gr[base], axis=0)
        norm = gr / gmean
    ok = (gmean >= 0.03) & (np.isfinite(gr[base]).sum(0) >= 20)
    ds = xr.open_dataset(RAW / "belt" / "chirps_v2_grid05.nc").sortby("lat")
    clim = ds["precip"].sel(lat=slice(3.5, 19.0), lon=slice(-18.0, 38.5), time=slice("1991-01-01", "2020-12-31"))
    clim = clim.where(clim.time.dt.month.isin([6, 7, 8, 9]), drop=True).groupby("time.year").sum().mean("year")
    lat0 = np.floor(glat / .5) * .5 + .25
    lon0 = np.floor(glon / .5) * .5 + .25
    jj = clim.sel(lat=xr.DataArray(np.repeat(lat0, len(lon0))), lon=xr.DataArray(np.tile(lon0, len(lat0)))).values
    ok &= (jj.reshape(len(lat0), len(lon0)) >= 100)
    norm[:, ~ok] = np.nan
    # per 0.5 deg cell: sum of normalised values and count per year, and number of vegetated pixels
    keys = np.array([cid(a, b) for a in lat0 for b in lon0]).reshape(len(lat0), len(lon0))
    flat_keys = keys.ravel()
    codes, uniq = pd.factorize(flat_keys)
    nflat = norm.reshape(len(gyears), -1)
    fin = np.isfinite(nflat)
    S = np.stack([np.bincount(codes, weights=np.where(fin[k], nflat[k], 0.0), minlength=len(uniq)) for k in range(len(gyears))])
    N = np.stack([np.bincount(codes, weights=fin[k].astype(float), minlength=len(uniq)) for k in range(len(gyears))])
    P = np.bincount(codes, weights=ok.ravel().astype(float), minlength=len(uniq))
    S, N = pd.DataFrame(S, index=gyears, columns=uniq), pd.DataFrame(N, index=gyears, columns=uniq)
    P = pd.Series(P, index=uniq)
    m = mcell.set_index("key")
    m = m[m.index.isin(uniq)]
    terr = {}
    for t, cs in m.groupby("MAP_name").groups.items():
        cs = list(cs)
        if P.loc[cs].sum() < MIN_TERR_PIX:
            continue
        s = S[cs].sum(axis=1) / N[cs].sum(axis=1).replace(0, np.nan)
        pct = B13.trend_pct(gyears.astype(float), s.values, (2001, 2024))
        if np.all((pct >= 0) & (pct <= 300) | ~np.isfinite(pct)):
            terr[t] = pd.Series(pct, index=gyears)
    own = mcell.set_index("cell")["MAP_name"]
    nb = mcell.set_index("cell")["n1_name"]
    g = pd.DataFrame({c: terr.get(t, pd.Series(np.nan, index=gyears)) for c, t in own.items()})
    n = pd.DataFrame({c: terr.get(t, pd.Series(np.nan, index=gyears)) if isinstance(t, str) else pd.Series(np.nan, index=gyears)
                      for c, t in nb.items()})
    return g, n, terr


B13 = importlib.import_module("13_belt_cells")


# ------------------------------------------------------------------ statistics
def demean_cols(X, groups):
    return np.column_stack([B.demean(X[:, k], groups, iters=1000, tol=1e-10) for k in range(X.shape[1])])


def cgm_se(Xr, e, clusters):
    """Two-way cluster-robust SEs (Cameron, Gelbach & Miller), each term with G/(G-1)*(N-1)/(N-K)."""
    n, k = Xr.shape
    bread = np.linalg.inv(Xr.T @ Xr)
    sc = Xr * e[:, None]

    def meat(g):
        codes, uniq = pd.factorize(g)
        S = np.column_stack([np.bincount(codes, weights=sc[:, j], minlength=len(uniq)) for j in range(k)])
        G = len(uniq)
        return (G / (G - 1)) * ((n - 1) / (n - k)) * (S.T @ S)

    a, b = clusters
    both = pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str)
    V = bread @ (meat(a) + meat(b) - meat(both.values)) @ bread
    return np.sqrt(np.clip(np.diag(V), 0, None))


class Panel2:
    """Rows = kept (cell, season) with every regressor; the M&N specification; season permutation."""

    def __init__(self, Y, K, R, T, unit, seasons, Cn=None):
        """Y, K: season x cell; R: {'n','g','c'} -> season x cell rain (or greening); T: {'n','g'} -> cell Series;
        unit: cell -> country code for country x season effects; Cn: season x cell event counts (optional)."""
        cols = Y.columns
        Y, K = Y.loc[seasons, cols], K.loc[seasons, cols]
        self.R = {k: v.reindex(index=seasons, columns=cols).T.values for k, v in R.items()}   # cells x seasons
        self.T = {k: v.reindex(cols).values.astype(float) for k, v in T.items()}
        ok_cell = np.isfinite(self.T["n"]) & np.isfinite(self.T["g"])
        for v in self.R.values():
            ok_cell &= np.isfinite(v).all(1)
        ci, si = np.where(K.T.values & ok_cell[:, None])
        self.ci, self.si, self.cols, self.S = ci, si, cols, len(seasons)
        self.seasons = seasons
        self.y = Y.T.values[ci, si].astype(float)
        u = pd.Series(unit).reindex(cols).astype(str).values
        g1 = pd.factorize(ci)[0]
        g2 = pd.factorize(pd.Series(u[ci]) + "_" + pd.Series(si).astype(str))[0]
        self.groups = [(g1, np.bincount(g1)), (g2, np.bincount(g2))]
        self.yr = B.demean(self.y, self.groups, iters=1000, tol=1e-10)
        self.iso = u[ci]
        self.n, self.n_cells = len(self.y), len(np.unique(ci))
        self.contact = self.T["n"][ci] > 0
        self.cnt = Cn.loc[seasons, cols].T.values[ci, si].astype(int) if Cn is not None else None
        rn = B.demean(self.R["n"][ci, si], [self.groups[0]])
        self.sd_n = float(rn.std())                                   # within-cell SD of neighbour rain, all rows
        # the unit used on the page: a typical year-to-year swing in rain next door, where the neighbour herds
        self.sd_nc = float(rn[self.contact].std()) if self.contact.any() else float("nan")

    def X(self, perm=None):
        s = self.si if perm is None else perm[self.si]
        rn, rg, rc = (self.R[k][self.ci, s] for k in ("n", "g", "c"))
        tn, tg = self.T["n"][self.ci], self.T["g"][self.ci]
        return np.column_stack([rn, rn * tn, rg, rg * tg, rc, rc * tg])

    def beta(self, perm=None):
        Xr = demean_cols(self.X(perm), self.groups)
        return np.linalg.lstsq(Xr, self.yr, rcond=None)[0], Xr

    def null(self, nperm):
        return np.array([self.beta(RNG.permutation(self.S))[0][1] for _ in range(nperm)])

    def units(self, g1):
        """Points of probability, and % of the outcome mean, for a season one contact-row SD drier next door,
        neighbour scoring HALF against a farming neighbour; + = more events after drier seasons next door."""
        pp = -100 * g1 * HALF * self.sd_nc
        return {"pp_half": pp, "rel_half": pp / self.y.mean() if self.y.mean() > 0 else float("nan")}

    def describe(self):
        return {"n": self.n, "cells": self.n_cells, "contact_cells": int(len(np.unique(self.ci[self.contact]))),
                "rate": float(self.y.mean()), "events": int(self.y.sum()),
                "events_contact": int(self.y[self.contact].sum()), "sd_n": self.sd_n, "sd_nc": self.sd_nc,
                "iso_events_contact": {k: int(self.y[self.contact & (self.iso == k)].sum()) for k in np.unique(self.iso)},
                "iso_nevents_contact": ({k: int(self.cnt[self.contact & (self.iso == k)].sum()) for k in np.unique(self.iso)}
                                        if self.cnt is not None else None),
                "iso_contact_cells": {k: int(len(np.unique(self.ci[self.contact & (self.iso == k)]))) for k in np.unique(self.iso)},
                "seasons": [self.seasons[0], self.seasons[-1]]}

    def test(self, nperm, clusters=None):
        b, Xr = self.beta()
        nl = self.null(nperm)
        g1 = float(b[1])
        q = np.quantile(nl, [0.025, 0.975])
        out = {"gamma1": g1, "beta": [float(v) for v in b], "p": float((np.abs(nl) >= abs(g1)).mean()),
               "null_sd": float(nl.std()), "null_lo": float(q[0]), "null_hi": float(q[1]),
               "mde80_gamma": float(3.083 * nl.std()), "nperm": nperm,
               "mde80_rel": 100 * 3.083 * float(nl.std()) * HALF * self.sd_nc / self.y.mean()} | self.units(g1) | self.describe()
        if clusters is not None:
            e = self.yr - Xr @ b
            out["se_cgm"] = float(cgm_se(Xr, e, clusters)[1])
        return out


# ------------------------------------------------------------------ main
def main(null_only=False):
    cells = pd.read_parquet(OUT / "cells.parquet")
    cells = cells[cells["in_belt"]].reset_index(drop=True)
    d = mn_frame()
    n1 = neighbour_identity(d)
    mcell = d.drop_duplicates("cell")[["cell", "x", "y", "MAP_name", "country", "KG_climate", "herdXEApXn12",
                                       "n1_herdXEApXn12", "herdXEApXn1234", "n1_herdXEApXn1234"]].copy()
    mcell["n1_name"] = mcell["cell"].map(n1)
    mcell["key"] = [cid(a, b) for a, b in zip(mcell["y"], mcell["x"])]
    bel = cells.merge(mcell, left_on="cell", right_on="key", how="inner", suffixes=("", "_mn"))
    bel = bel.rename(columns={"cell_mn": "mncell"})
    print(f"[contact] belt cells {len(cells)}, matched {len(bel)}, with neighbour score "
          f"{bel['n1_herdXEApXn12'].notna().sum()}, contact (score>0) {(bel['n1_herdXEApXn12'] > 0).sum()}")

    # ---- rain fields (CHIRPS), on replication cells, then onto belt cells
    rf = chirps_fields(mcell)
    fields = {}
    for k in ("jjas", "ann"):
        g, n = territory_fields(rf[k], mcell)
        fields[k] = {"c": rf[k], "g": g, "n": n}

    def on_belt(tab):            # replication-cell columns -> belt-cell columns
        return tab[bel["mncell"].values].set_axis(bel["cell"].values, axis=1)

    R = {k: {r: on_belt(v) for r, v in f.items()} for k, f in fields.items()}
    T = {"n": bel.set_index("cell")["n1_herdXEApXn12"], "g": bel.set_index("cell")["herdXEApXn12"]}
    Tb = {"n": bel.set_index("cell")["n1_herdXEApXn1234"], "g": bel.set_index("cell")["herdXEApXn1234"]}
    unit = bel.set_index("cell")["iso3"]

    # ---- outcomes on belt cells
    ev = B.load_events(set(bel["cell"]))
    attr, attr_n = B.attribution(ev)
    ev["any"] = ev["event_type"].isin(["Battles", "Violence against civilians"])
    ev["nonstate"] = ev["any"] & (ev["inter1"] != "State forces") & (ev["inter2"].fillna("") != "State forces")
    ucdp = ucdp_communal(set(bel["cell"]))
    cols = bel["cell"].values
    iso_of = dict(zip(bel["cell"], bel["iso3"]))

    def win(w, y):
        return (f"{y}-01-01", f"{y + 1}-01-01") if w == "cal" else B.window(w, y)

    def wyears(w, y):
        return [y] if w == "cal" else B.window_years(w, y)

    def ymat(src, cls, w, seasons, counts=False):
        rows = []
        for y in seasons:
            s, e = win(w, y)
            m = (src["date"] >= s) & (src["date"] < e) & src[cls]
            rows.append(src[m].groupby("cell").size().reindex(cols, fill_value=0).values)
        a = np.array(rows)
        return pd.DataFrame(a if counts else (a > 0).astype(int), index=seasons, columns=cols)

    def kept(w, seasons, attr_rule, drop_iso=()):
        k = np.zeros((len(seasons), len(cols)), bool)
        for i, y in enumerate(seasons):
            war_end = win(w, y)[1] > B.SDN_WAR
            for j, c in enumerate(cols):
                iso = iso_of[c]
                if iso in drop_iso or (iso == "SDN" and war_end):
                    continue
                k[i, j] = (not attr_rule) or all(attr.get((iso, yy), 0.0) <= B.ATTR_MAX for yy in wyears(w, y))
        return pd.DataFrame(k, index=seasons, columns=cols)

    def panel(cls="resource", w="jul_oct", rain="jjas", seasons=SEASONS, src=None, drop_iso=(), Tsel=None,
              Rsel=None, only_iso=None):
        src = ev if src is None else src
        Cn = ymat(src, cls, w, seasons, counts=True)
        Y = (Cn > 0).astype(int)
        K = kept(w, seasons, attr_rule=(cls in ("resource", "resource_strict")), drop_iso=drop_iso)
        if only_iso:
            K = K & np.repeat(np.array([iso_of[c] == only_iso for c in cols])[None, :], len(seasons), 0)
        return Panel2(Y, K, Rsel or R[rain], Tsel or T, unit, seasons, Cn=Cn)

    if null_only:
        P = panel()
        nl = P.null(NPERM)
        print(f"[contact] C1 permutation null: rows {P.n}, cells {P.n_cells}, contact cells "
              f"{P.describe()['contact_cells']}, events {int(P.y.sum())} (contact {int(P.y[P.contact].sum())}); "
              f"null SD {nl.std():.5f}; 80% detectable effect {100 * 3.083 * nl.std() * HALF * P.sd_n:.2f} pp "
              f"per SD drier next door at score 0.5 (SD {P.sd_n:.1f} mm); "
              f"{100 * 3.083 * nl.std() * P.sd_n / P.y.mean():.1f}% of the mean at score 1 "
              f"(page unit: {100 * 3.083 * nl.std() * HALF * P.sd_nc / P.y.mean():.1f}% at 0.5, contact-row SD {P.sd_nc:.1f} mm)")
        return

    res = {"calibration": {}, "bridge": [], "primary": None, "secondary": [], "loo": [], "countries": []}

    # ---- A: calibration on their data
    print("[contact] A: calibration")
    chk = np.nanmax(np.abs(d["n1_prec_gpcc_X_herdXEApXn12"] - d["n1_g_prec_gpcc"] * d["n1_herdXEApXn12"]))
    print(f"  interaction check (max |stored - product|): {chk:.2e}")
    belt_mn = set(bel["mncell"])
    # their % unit: within-cell SD of own-cell rain over all their cells and years (Stata loneway on the full file)
    pc = d.dropna(subset=["prec_gpcc"])
    sd_paper = float(np.sqrt(((pc["prec_gpcc"] - pc.groupby("cell")["prec_gpcc"].transform("mean")) ** 2).sum()
                             / (len(pc) - pc["cell"].nunique())))
    res["sd_paper"] = sd_paper
    for scope in ("africa", "belt"):
        for yv, yrs in (("ucdp_all_10", range(1989, 2019)), ("acled_all_10", SEASONS_MN), ("acled_nonstate_10", SEASONS_MN)):
            dd = d[d["year"].isin(list(yrs))]
            if scope == "belt":
                dd = dd[dd["cell"].isin(belt_mn)]
            Y = dd.pivot(index="year", columns="cell", values=yv)
            K = Y.notna()
            Rm = {"n": dd.pivot(index="year", columns="cell", values="n1_g_prec_gpcc"),
                  "g": dd.pivot(index="year", columns="cell", values="g_prec_gpcc"),
                  "c": dd.pivot(index="year", columns="cell", values="prec_gpcc")}
            mc = mcell.set_index("cell")
            Tm = {"n": mc["n1_herdXEApXn12"], "g": mc["herdXEApXn12"]}
            P = Panel2(Y.fillna(0), K, Rm, Tm, mc["country"], list(yrs))
            b, Xr = P.beta()
            e = P.yr - Xr @ b
            cl = (P.ci, pd.Series(mc["KG_climate"].reindex(P.cols).values[P.ci]).astype(str) + "_" + pd.Series(P.si).astype(str))
            se = cgm_se(Xr, e, (P.ci, cl[1].values))
            row = {"scope": scope, "outcome": yv, "gamma1": float(b[1]), "se_cgm": float(se[1]), "n": P.n,
                   "cells": P.n_cells, "rate": float(P.y.mean()), "pct_paper": 100 * b[1] * sd_paper / P.y.mean(),
                   "sd_n": P.sd_n, "sd_nc": P.sd_nc, "contact_cells": P.describe()["contact_cells"]} | P.units(float(b[1]))
            if scope == "africa":
                row |= {"published": PUBLISHED[yv], "published_se": PUB_SE[yv], "published_pct": PUB_PCT[yv]}
            nl = P.null(NPERM)
            q = np.quantile(nl, [0.025, 0.975])
            row |= {"p": float((np.abs(nl) >= abs(b[1])).mean()), "null_sd": float(nl.std()),
                    "null_lo": float(q[0]), "null_hi": float(q[1]), "nperm": NPERM,
                    "mde80_rel": 100 * 3.083 * float(nl.std()) * HALF * P.sd_nc / P.y.mean()}
            res["calibration"].setdefault(scope, []).append(row)
            print(f"  {scope:<6} {yv:<18} g1={b[1]:+.4f} (se {se[1]:.4f}) n={P.n} cells={P.n_cells}"
                  + (f" published {PUBLISHED[yv]:+.4f} ({PUB_SE[yv]:.4f}) {row['pct_paper']:+.2f}% vs {PUB_PCT[yv]:+.2f}%" if scope == "africa" else "")
            + f" p_perm={row['p']:.3f} rel_half={row['rel_half']:+.1f}% mde={row['mde80_rel']:.0f}%",
                  flush=True)

    # ---- bridge and primary
    print("[contact] bridge")

    def run(bucket, key, label, nperm=NPERM, **kw):
        P = panel(**kw)
        if P.n == 0 or P.y.sum() == 0:
            return None
        r = P.test(nperm) | {"key": key, "label": label}
        (res[bucket].append(r) if isinstance(res[bucket], list) else res.__setitem__(bucket, r))
        print(f"  {key:<14} g1={r['gamma1']:+.5f} -> {r['pp_half']:+.2f} pp, {r['rel_half']:+.1f}% of mean "
              f"(mde {r['mde80_rel']:.0f}%) p={r['p']:.3f}; n={r['n']} contact cells={r['contact_cells']} "
              f"events={r['events']} (contact {r['events_contact']})", flush=True)
        return r

    run("bridge", "B1", "This project's data (CHIRPS, its ACLED pull, events within 25 km, its countries and cells), "
        "their outcome and years", cls="any", w="cal", rain="ann", seasons=SEASONS_MN)
    run("bridge", "B2", "… seasons 2010–2024", cls="any", w="cal", rain="ann")
    run("bridge", "B3", "… June–September rain, July–October events", cls="any", w="jul_oct", rain="jjas")
    print("[contact] primary")
    c1 = run("primary", "C1", "Resource-type, July–October (= B4)")
    print("[contact] secondary")
    run("secondary", "S_nonstate", "ACLED non-state, July–October", cls="nonstate")
    run("secondary", "S_ucdp", "UCDP communal, July–October", cls="ucdp", src=ucdp)
    run("secondary", "S_broad", "Broad herder score", Tsel=Tb)
    run("secondary", "S_noSDN", "Without Sudan", drop_iso=("SDN",))
    run("secondary", "S_octjan", "October–January", w="oct_jan")
    gG, gN, terr = greening_fields(mcell)
    gR = {"c": B13_cell_grn(bel), "g": on_belt(gG), "n": on_belt(gN)}
    run("secondary", "S_greening", "Greening instead of rain", Rsel=gR)
    print("[contact] leave one country out / by country")
    base = panel()
    ev_c = base.describe()["iso_nevents_contact"]          # resource-type events in contact rows, by country
    res["contact_events_by_country"] = ev_c
    for iso in sorted(k for k, v in ev_c.items() if v > 0):
        run("loo", f"LOO_{iso}", f"Without {B.NAME[iso]}", drop_iso=(iso,))
    for iso in sorted(k for k, v in ev_c.items() if v >= B.MIN_EVENTS_COUNTRY):
        run("countries", f"CTRY_{iso}", B.NAME[iso], only_iso=iso)

    # ---- added after review (not in the plan): rain with each cell's own linear trend over the tested seasons
    # removed (the in-sample trend is the one cell and country x season effects leave in place)
    print("[contact] post-review: detrended rain")
    res["post_review"] = []

    def detr(tab, span=(SEASONS[0], SEASONS[-1])):
        yrs = tab.index.values.astype(float)
        out = tab.copy()
        for c in tab.columns:
            v = tab[c].values.astype(float)
            m = np.isfinite(v) & (yrs >= span[0]) & (yrs <= span[1])
            if m.sum() >= 10:
                out[c] = v - np.polyval(np.polyfit(yrs[m], v[m], 1), yrs) + v[m].mean()
        return out

    Rd = {k: {r: detr(v) for r, v in f.items()} for k, f in R.items()}
    run("post_review", "D_C1", "Primary, rain detrended per cell", Rsel=Rd["jjas"])
    run("post_review", "D_B2", "B2, rain detrended per cell", cls="any", w="cal", Rsel=Rd["ann"])

    # ---- licence (prespec): only if C1 shows a link
    link = c1 is not None and c1["gamma1"] < 0 and c1["p"] < 0.025
    nos = next((r for r in res["secondary"] if r["key"] == "S_noSDN"), None)
    big = [r for r in res["loo"] if ev_c.get(r["key"][4:], 0) >= B.MIN_EVENTS_COUNTRY]
    res["licence"] = {"c1_link": bool(link), "noSDN_sign": bool(nos and nos["gamma1"] < 0),
                      "loo_sign_all": bool(big and all(r["gamma1"] < 0 for r in big)),
                      "granted": bool(link and nos and nos["gamma1"] < 0 and big and all(r["gamma1"] < 0 for r in big))}
    print(f"[contact] licence: {res['licence']}")

    # ---- map + coverage
    # the page gets scores only; ethnic territory names stay in the work folder (contact_neighbours.csv)
    res["cells"] = [{"cell": c, "lat": float(a), "lon": float(o), "iso3": i, "thp_n": _f(tn), "thp_g": _f(tg)}
                    for c, a, o, i, tn, tg in zip(bel["cell"], bel["lat"], bel["lon"], bel["iso3"],
                                                  bel["n1_herdXEApXn12"], bel["herdXEApXn12"])]
    cc = bel[bel["n1_herdXEApXn12"] > 0]
    (cc.groupby(["n1_name", "iso3"]).agg(cells=("cell", "size"), thp=("n1_herdXEApXn12", "first"))
       .reset_index().to_csv(OUT / "contact_neighbours.csv", index=False))
    res["coverage"] = {"belt_cells": len(cells), "matched": len(bel), "with_score": int(bel["n1_herdXEApXn12"].notna().sum()),
                       "contact": int(len(cc)), "contact_by_country": cc.groupby("iso3").size().to_dict(),
                       "pkg_cells": int(len(mcell)), "pkg_matched": int(mcell["n1_name"].notna().sum()),
                       "no_identity": int((bel["n1_herdXEApXn12"].notna() & bel["n1_name"].isna()).sum()),
                       "no_identity_contact": int(((bel["n1_herdXEApXn12"] > 0) & bel["n1_name"].isna()).sum()),
                       "no_own_score": int((bel["n1_herdXEApXn12"].notna() & bel["herdXEApXn12"].isna()).sum()),
                       "greening_territories": len(terr)}
    (OUT / "contact_data.json").write_text(json.dumps(_sig(res), indent=1))
    print(f"[contact] wrote {OUT / 'contact_data.json'}")


def _sig(o):
    """JSON-safe copy with floats to 6 significant digits (coefficients per mm are ~1e-5: fixed decimals lose them)."""
    if isinstance(o, dict):
        return {str(k): _sig(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sig(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.6g}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _f(v):
    return None if pd.isna(v) else float(v)


def B13_cell_grn(bel):
    cy = pd.read_parquet(OUT / "cell_years.parquet")
    return cy.pivot(index="year", columns="cell", values="grn").reindex(columns=bel["cell"].values)


def ucdp_communal(belt_cells):
    """UCDP GED type 2 events in non-state conflicts between informally organised groups (org 3), within 25 km."""
    ns = pd.read_csv(RAW / "ucdp" / "nonstate" / "NonState_v25_1.csv")
    org3 = set(ns.loc[ns["org"] == 3, "conflict_id"])
    u = pd.read_csv(RAW / "ucdp" / "GEDEvent_v25_1.csv", usecols=["type_of_violence", "conflict_new_id", "where_prec",
                                                                  "latitude", "longitude", "date_start"], low_memory=False)
    u = u[(u["type_of_violence"] == 2) & u["conflict_new_id"].isin(org3) & (u["where_prec"] <= 2)].copy()
    u["cell"] = [B.cell_id(float(a), float(b)) for a, b in zip(u["latitude"], u["longitude"])]
    u = u[u["cell"].isin(belt_cells)]
    u["date"] = pd.to_datetime(u["date_start"])
    u["ucdp"] = True
    return u


if __name__ == "__main__":
    main(null_only="--null" in sys.argv)
