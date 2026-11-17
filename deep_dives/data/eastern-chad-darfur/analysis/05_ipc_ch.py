"""Share of people in Phase 3+ in the lean season: CH for eastern Chad, IPC for Darfur.

Lean season Y follows the harvest of season Y-1.

Chad (Cadre Harmonise, public HDX consolidated file): the projection for June-August
Y (the March exercise; in 2023 the July update for the eastern provinces) and the
current situation for September-December Y (post-harvest). Residents by departement;
refugees and returnees (separate rows since 2024) as a second group.

Darfur (IPC, state level, team mirror ipc.population): the analysis whose period
covers August Y — a current analysis when there is one, otherwise the most recent
projection. 2025 has no analysis covering the lean season; the September 2025
current analysis (end of the lean season) is used and flagged.

Writes data/processed/ipcch/phase_share.parquet
  side, year, kind (lean/post), source, analysis, period, share3, share4, share5, pop
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import engine, stratus  # noqa: E402,F401
from aoi import PROC, RAW  # noqa: E402

OUT = PROC / "ipcch"
HDX_CH = "https://data.humdata.org/api/3/action/package_show?id=cadre-harmonise"
CH_FILE = RAW / "ch" / "cadre_harmonise_hdx.xlsx"
CHAD_ADM1 = {"Ouaddai", "Ouaddaï", "Wadi Fira", "Sila"}
DARFUR = ["North Darfur", "South Darfur", "West Darfur", "East Darfur", "Central Darfur",
          "Central Darfur State"]


def chad():
    """CH from the public HDX consolidated file (usethisperiod == 'Y' rows only).

    Residents = rows with a departement p-code; refugee and returnee rows (analysed
    separately since 2024, no p-code) are kept as a second population group."""
    if not CH_FILE.exists():
        CH_FILE.parent.mkdir(parents=True, exist_ok=True)
        meta = requests.get(HDX_CH, timeout=60).json()["result"]["resources"]
        url = next(r["url"] for r in meta if r["name"].endswith(".xlsx"))
        CH_FILE.write_bytes(requests.get(url, timeout=600).content)
    x = pd.read_excel(CH_FILE)
    x = x[x["adm0_name"].isin(["Chad", "Tchad"]) & x["adm1_name"].isin(CHAD_ADM1)
          & x["usethisperiod"].eq("Y")]
    camp = x["adm2_name"].astype(str).str.contains("Refug|Réfug|Returnee|Retourn", case=False)
    x["group"] = camp.map({True: "refugees", False: "residents"})
    sel = {
        "lean": x["chtype"].eq("projected") & x["reference_label"].eq("Jun-Aug"),
        "post": x["chtype"].eq("current") & x["reference_label"].eq("Sep-Dec"),
    }
    n_dep = 9  # departements of the three provinces (2014 was analysed by province)
    rows = []
    for kind, m in sel.items():
        for (yr, grp), g in x[m].groupby(["reference_year", "group"]):
            ok = g["population"].notna() & g["phase35"].notna()
            pop = g.loc[ok, "population"].sum()
            if pop == 0:
                continue
            flag = ""
            if grp == "residents" and ok.sum() < n_dep and g["adm2_pcod2"].notna().any():
                miss = ", ".join(sorted(g.loc[~ok, "adm2_name"]))
                flag = f"{ok.sum()} of {n_dep} departements" + (f" (not analysed: {miss})"
                                                                if miss else "")
            ex = sorted({f"{int(e)}-{lbl}" for e, lbl in
                         zip(g["exercise_year"], g["exercise_label"])})
            g = g[ok]
            rows.append({
                "side": "chad", "year": int(yr), "kind": kind, "source": "CH",
                "group": grp, "analysis": "/".join(ex),
                "period": "Jun-Aug" if kind == "lean" else "Sep-Dec",
                "share3": g["phase35"].sum() / pop,
                "share4": (g["phase4"].sum() + g["phase5"].sum()) / pop,
                "share5": g["phase5"].sum() / pop,
                "pop": pop, "n_units": len(g), "flag": flag,
            })
    return pd.DataFrame(rows)


def darfur():
    e = engine()
    q = """
      select analysis_date, period_type, reference_period_start, reference_period_end,
             level1_name, phase, population
      from ipc.population
      where iso3 = 'SDN' and level = 'level1' and analysis_date >= '2017-01-01'
        and level1_name = any(%(s)s)
    """
    d = pd.read_sql(q, e, params={"s": DARFUR})
    d["level1_name"] = d["level1_name"].str.replace(" State", "", regex=False)
    for c in ("analysis_date", "reference_period_start", "reference_period_end"):
        d[c] = pd.to_datetime(d[c])
    rows = []
    for y in range(2017, 2027):
        aug = pd.Timestamp(f"{y}-08-15")
        c = d[(d["reference_period_start"] <= aug) & (d["reference_period_end"] >= aug)]
        flag = ""
        if c.empty:  # fall back to an analysis of a lean-season month (Jun-Sep)
            c = d[(d["reference_period_start"].dt.year == y)
                  & d["reference_period_start"].dt.month.between(6, 9)]
            flag = "outside Aug"
        if c.empty:
            continue
        c = c.assign(cur=c["period_type"].eq("current"))
        best = c.sort_values(["cur", "analysis_date"], ascending=False).iloc[0]
        g = c[(c["analysis_date"] == best["analysis_date"])
              & (c["period_type"] == best["period_type"])
              & (c["reference_period_start"] == best["reference_period_start"])]
        piv = g.pivot_table(index="level1_name", columns="phase", values="population",
                            aggfunc="sum")
        if piv.shape[0] < 5:
            flag = (flag + f" {piv.shape[0]} of 5 states").strip()
        # A projection can cover fewer localities than the analysis while `all` keeps the
        # full population (Jun-Sep 2026 covers about half of Darfur). Divide by the people
        # actually classified and report the coverage.
        phases = [k for k in ("1", "2", "3", "4", "5") if k in piv]
        classified = piv[phases].sum(axis=1)
        pop = classified.sum()
        cover = pop / piv["all"].sum()
        if cover < 0.98:
            flag = (flag + f" covers {cover:.0%} of the population").strip()
        p = {k: piv[k].sum() for k in ("3", "4", "5") if k in piv}
        rows.append({
            "side": "darfur", "year": y, "kind": "lean", "source": "IPC", "group": "all",
            "analysis": best["analysis_date"].strftime("%Y-%m"),
            "period": f"{best['reference_period_start']:%b %Y}-{best['reference_period_end']:%b %Y}"
                      f" ({best['period_type']})",
            "share3": (p.get("3", 0) + p.get("4", 0) + p.get("5", 0)) / pop,
            "share4": (p.get("4", 0) + p.get("5", 0)) / pop,
            "share5": p.get("5", 0) / pop,
            "pop": pop, "coverage": cover, "n_units": piv.shape[0], "flag": flag,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.concat([chad(), darfur()], ignore_index=True)
    d.to_parquet(OUT / "phase_share.parquet", index=False)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
